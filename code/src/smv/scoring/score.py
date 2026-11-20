"""`smv score`: evaluate every stored result against gold, write per-image counts, long rows and summaries."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smv.data.index import load_golds, load_index
from smv.orchestrator.store import ResultStore
from smv.prompts.registry import PromptRegistry
from smv.schema.config import load_schema_config
from smv.scoring.confidence import offer_confidences
from smv.scoring.evaluate import COUNT_COLS, MODES, EvalConfig, evaluate_image, perception_metrics, prf
from smv.scoring.ocr_regex import lines_to_prediction
from smv.scoring.parse import ocr_text_to_lines, parse_output
from smv.settings import Settings
from smv.stats.bootstrap import bootstrap_f1, bootstrap_ratio

log = logging.getLogger("smv.score")
OCR_REGEX = "OCR-REGEX"


def _latest(records: list[dict]) -> dict[tuple, dict]:
    """Latest record per request key; an ok record beats a later error only if the error is older."""
    out: dict[tuple, dict] = {}
    for r in records:
        k = (r["image_sha"], int(r.get("seed", 0)), r.get("thinking", "off"))
        prev = out.get(k)
        if prev is None or r.get("status") == "ok" or prev.get("status") != "ok":
            out[k] = r
    return out


def score_results(st: Settings, models: list[str] | None = None, prompts: list[str] | None = None,
                  out_dir: Path | None = None, tau: float | None = None, with_rows: bool = True,
                  overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    scfg = {**st.scoring_cfg(), **(overrides or {})}
    lex = st.lexicon()
    ecfg = EvalConfig.from_cfg(scfg, lex, tau)
    index = load_index(st)
    golds = load_golds(st)
    out_dir = out_dir or (st.out_dir / "scores" / "all")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not golds:
        msg = "no gold labels found (outputs/data/gold is empty): nothing to score"
        (out_dir / "README.txt").write_text(msg + "\n")
        return {"message": msg, "n_rows": 0}
    reg = PromptRegistry(st.prompts_dir)
    store = ResultStore(st.results_dir)
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in index.itertuples(index=False):
        base = r.base_image_id if isinstance(r.base_image_id, str) and r.base_image_id else r.image_id
        if base in golds:
            by_sha[r.sha256].append({"image_id": r.image_id, "base": base, "split": r.split,
                                     "perturbation": r.perturbation or ""})
    rep_limit = int(scfg.get("repetition_min_count", 5))
    rows: list[dict] = []
    long_rows: list[dict] = []
    models = models or store.models()
    for model in models:
        for pid in store.prompts_of(model):
            if prompts and pid.split("@")[0] not in prompts and pid != "OCR":
                continue
            base_pid = pid.split("@", 1)[0]
            kind = "ocr" if base_pid == "OCR" else (reg.get(base_pid).output if base_pid in reg.specs else "json")
            if kind == "free":
                continue
            for (sha, seed, thinking), rec in _latest(store.load(model, pid)).items():
                for img in by_sha.get(sha, []):
                    ann = golds[img["base"]]
                    meta = {"model": model, "seed": seed, "thinking": thinking, "split": img["split"],
                            "image_id": img["image_id"], "base_image_id": img["base"],
                            "perturbation": img["perturbation"], "latency_s": rec.get("latency_s"),
                            "prompt_tokens": rec.get("prompt_tokens"), "completion_tokens": rec.get("completion_tokens"),
                            "finish_reason": rec.get("finish_reason", ""), "request_status": rec.get("status")}
                    if kind == "ocr":
                        lines = rec.get("ocr_lines") or ocr_text_to_lines(rec.get("text", ""))
                        pm = perception_metrics(ann, lines) if rec.get("status") == "ok" else {}
                        pred = lines_to_prediction(lines) if rec.get("status") == "ok" else None
                        for mode in MODES:
                            c, lr = evaluate_image(ann, pred, ecfg, mode, with_rows and mode == "strict",
                                                   "strict" if pred is not None else "failed")
                            rows.append({**meta, "prompt": OCR_REGEX, "ocr_source": model, "mode": mode,
                                         "parse_status": "strict" if pred is not None else "failed",
                                         "truncated": False, "repetition": False, "think_chars": 0, **c, **pm})
                            long_rows += [{**meta, "prompt": OCR_REGEX, **x} for x in lr]
                        continue
                    if kind == "lines":
                        pr = parse_output(rec.get("text", ""), "lines", rec.get("finish_reason", ""),
                                          repetition_limit=rep_limit) if rec.get("status") == "ok" else None
                        pm = perception_metrics(ann, pr.lines) if pr is not None else {}
                        rows.append({**meta, "prompt": pid, "mode": "strict",
                                     "parse_status": pr.status if pr else "failed",
                                     "truncated": bool(pr and pr.truncated), "repetition": bool(pr and pr.repetition),
                                     "think_chars": pr.think_chars if pr else 0, **pm})
                        continue
                    if rec.get("status") == "ok":
                        pr = parse_output(rec.get("text", ""), "json", rec.get("finish_reason", ""),
                                          repetition_limit=rep_limit)
                        if pr.prediction is not None and rec.get("lp"):      # run with logprobs: confidence per offer
                            cs = offer_confidences(rec.get("text", ""), rec, pr.prediction)
                            meta = {**meta, "conf_offers": cs["offers"], "conf_mapped": cs["with_conf"]}
                    else:
                        pr = None
                    for mode in MODES:
                        c, lr = evaluate_image(ann, pr.prediction if pr else None, ecfg, mode,
                                               with_rows and mode == "strict", pr.status if pr else "failed")
                        rows.append({**meta, "prompt": pid, "mode": mode, "parse_status": pr.status if pr else "failed",
                                     "truncated": bool(pr and pr.truncated), "repetition": bool(pr and pr.repetition),
                                     "think_chars": pr.think_chars if pr else 0, **c})
                        long_rows += [{**meta, "prompt": pid, **x} for x in lr]
    df = pd.DataFrame(rows)
    for col in COUNT_COLS:
        if col not in df.columns:
            df[col] = 0
    if not df.empty:
        df[COUNT_COLS] = df[COUNT_COLS].fillna(0)
    df.to_parquet(out_dir / "per_image.parquet", index=False)
    if with_rows:
        pd.DataFrame(long_rows).to_csv(out_dir / "long_offers.csv.gz", index=False, compression="gzip")
    summary = aggregate(df, ["model", "prompt", "thinking", "split", "mode"], scfg) if not df.empty else pd.DataFrame()
    summary.to_csv(out_dir / "summary.csv", index=False)
    meta = {"tau": ecfg.tau, "n_rows": len(df), "n_long_rows": len(long_rows), "n_gold": len(golds),
            "models": models}
    (out_dir / "score_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _ratio(n, d):
    n, d = float(np.sum(n)), float(np.sum(d))
    return n / d if d else float("nan")


def aggregate(df: pd.DataFrame, by: list[str], scfg: dict[str, Any] | None = None, ci: bool = True,
              n_boot: int | None = None) -> pd.DataFrame:
    scfg = scfg or {}
    b = scfg.get("bootstrap", {})
    n_boot = n_boot or int(b.get("n", 10000))
    seed = int(b.get("seed", 2027))
    out = []
    for keys, g in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        r: dict[str, Any] = dict(zip(by, keys))
        r["n_images"] = g["base_image_id"].nunique() if "base_image_id" in g else len(g)
        r["n_rows"] = len(g)
        nan = float("nan")
        for rule in ("strict", "lenient"):
            tp, fp, fn = g[f"off_tp_{rule}"], g[f"off_fp_{rule}"], g[f"off_fn_{rule}"]
            # nothing to score (e.g. transcription-only prompt): report n/a, not a perfect score
            p, rc, f = prf(tp.sum(), fp.sum(), fn.sum()) if (tp.sum() + fp.sum() + fn.sum()) > 0 else (nan, nan, nan)
            r[f"offer_P_{rule}"], r[f"offer_R_{rule}"], r[f"offer_F1_{rule}"] = p, rc, f
            per = [prf(a, b_, c)[2] for a, b_, c in zip(tp, fp, fn) if a + b_ + c > 0]
            r[f"offer_F1_macro_{rule}"] = float(np.mean(per)) if per else float("nan")
            r[f"PNA_{rule}"] = _ratio(g[f"pna_ok_{rule}"], g["paired"])
            r[f"IPA_{rule}"] = _ratio(g[f"inh_tp_{rule}"], g["inh_n"])
            r[f"SAA_{rule}"] = _ratio(g[f"sp_tp_{rule}"], g["sp_n"])
            r[f"variant_F1_{rule}"] = prf(g[f"var_tp_{rule}"].sum(), g[f"var_fp_{rule}"].sum(), g[f"var_fn_{rule}"].sum())[2] \
                if g["var_gold"].sum() else float("nan")
        ip, ir, ifi = (prf(g["item_tp"].sum(), g["item_fp"].sum(), g["item_fn"].sum())
                       if (g["item_tp"].sum() + g["item_fp"].sum() + g["item_fn"].sum()) > 0 else (nan, nan, nan))
        r.update({"item_P": ip, "item_R": ir, "item_F1": ifi, "NNA": _ratio(g["nna_ok"], g["nna_n"]),
                  "hallucination_rate": _ratio(g["item_fp"], g["item_tp"] + g["item_fp"]),
                  "omission_rate": _ratio(g["item_fn"], g["gold_items"]),
                  "desc_as_item_rate": _ratio(g["fp_desc_as_item"], g["item_tp"] + g["item_fp"]),
                  "abstain_correct": _ratio(g["abstain_ok"], g["abstain_targets"]),
                  "edits_per_100_offers": 100 * _ratio(g["off_fp_strict"] + g["off_fn_strict"], g["gold_offers"]),
                  "null_price_ok": _ratio(g["null_ok"], g["null_n"]),
                  "null_price_fabricated": _ratio(g["null_fab"], g["null_n"]),
                  "over_abstain_rate": _ratio(g["over_abstain"], g["paired"]),
                  "magnitude_err_rate": _ratio(g["magnitude_err"], g["paired"]),
                  "magnitude_share_of_wrong": _ratio(g["magnitude_err"], g["price_wrong"]),
                  "unit_acc": _ratio(g["unit_ok"], g["unit_n"]), "old_price_acc": _ratio(g["old_ok"], g["old_n"]),
                  "valid_json_strict": float((g["parse_status"] == "strict").mean()),
                  "valid_json_repaired": float(g["parse_status"].isin(["strict", "lenient"]).mean()),
                  "truncation_rate": float(g["truncated"].astype(bool).mean()),
                  "repetition_rate": float(g["repetition"].astype(bool).mean()),
                  "pt_price_recall": _ratio(g["pt_price_hit"], g["pt_price_n"]),
                  "pt_token_recall": _ratio(g["pt_tok_hit"], g["pt_tok_n"]),
                  "latency_median_s": float(pd.to_numeric(g["latency_s"], errors="coerce").median()),
                  "latency_p90_s": float(pd.to_numeric(g["latency_s"], errors="coerce").quantile(0.9)),
                  "completion_tokens_mean": float(pd.to_numeric(g["completion_tokens"], errors="coerce").mean()),
                  "think_chars_mean": float(pd.to_numeric(g.get("think_chars", 0), errors="coerce").mean())})
        if ci and g["gold_offers"].sum() > 0:
            f, lo, hi = bootstrap_f1(g["off_tp_strict"], g["off_fp_strict"], g["off_fn_strict"], n_boot, seed)
            r["offer_F1_strict_lo"], r["offer_F1_strict_hi"] = lo, hi
            for name, num, den in (("PNA_strict", "pna_ok_strict", "paired"), ("IPA_strict", "inh_tp_strict", "inh_n"),
                                   ("NNA", "nna_ok", "nna_n")):
                if g[den].sum() > 0:
                    _, lo, hi = bootstrap_ratio(g[num], g[den], max(1000, n_boot // 5), seed)
                    r[f"{name}_lo"], r[f"{name}_hi"] = lo, hi
        out.append(r)
    return pd.DataFrame(out)


def attach_meta(df: pd.DataFrame, index: pd.DataFrame, dense_threshold: int = 60) -> pd.DataFrame:
    meta = index[index["perturbation"].fillna("") == ""][["image_id", "ccl", "tags", "venue_tier", "venue_type", "medium",
                                                          "lighting", "angle", "n_offers"]].rename(columns={"image_id": "base_image_id"})
    out = df.merge(meta, on="base_image_id", how="left", suffixes=("", "_img"))
    out["dense"] = out["n_offers"].fillna(0) > dense_threshold
    out["non_chain"] = out["venue_tier"].fillna("") != "chain"
    return out


def slice_table(df: pd.DataFrame, index: pd.DataFrame, dim: str, schema_cfg_path: Path,
                by: list[str] | None = None, scfg: dict | None = None, dense_threshold: int = 60) -> pd.DataFrame:
    """Aggregate by a slicing dimension: ccl | tag | tag_group | venue_tier | non_chain | dense | medium | lighting |
    angle | split."""
    by = by or ["model", "prompt", "thinking"]
    d = attach_meta(df, index, dense_threshold)
    if dim in ("tag", "tag_group"):
        cfg = load_schema_config(schema_cfg_path)
        d = d.assign(tag=d["tags"].fillna("").str.split(";")).explode("tag")
        d = d[d["tag"].fillna("") != ""]
        if dim == "tag_group":
            d["tag_group"] = d["tag"].map(lambda t: cfg.tag_group_of(t) or "?")
            d = d.drop_duplicates(subset=[*by, "split", "mode", "seed", "base_image_id", "perturbation", "tag_group"])
    if dim not in d.columns:
        raise KeyError(f"unknown slice {dim}")
    return aggregate(d, [*by, dim], scfg, ci=False)
