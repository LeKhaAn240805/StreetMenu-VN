"""Which offers should a person check, and what do cheap ensembles gain? (no GPU needed)

Confidence signals for every predicted offer of a reference run, all compared on the same question, "is this offer right
under the strict rule?":
    token probability   from the log-probabilities of the answer (run P1@lp, scoring/confidence.py)
    stated confidence   the number the model writes when asked (prompt P-Conf)
    prompt agreement    share of the other prompts of the same model that return the same item with the same price
    sample agreement    share of the sampled answers (P1@t07, five seeds) that return it
    model agreement     share of the other models of the main table that return it
A signal is useful when wrong offers get low values: AUROC, and the share of all wrong offers found when a person reviews
the 10/20/30% least confident offers. Ensembles keep an (item, price) pair returned by a strict majority of their members
and are scored like any other prediction."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from smv.report import figures as F  # noqa: E402
from smv.report.extras import load_display_names, pretty  # noqa: E402
from smv.report.tables import write_table  # noqa: E402
from smv.schema.prediction import Prediction, PredItem, PredOffer  # noqa: E402
from smv.scoring.normalize import normalize_name  # noqa: E402
from smv.stats.bootstrap import bootstrap_f1, micro_f1, paired_bootstrap  # noqa: E402

TUNING = ["qwen3vl_8b", "gemma4_e4b", "internvl35_8b"]
PROMPT_PANEL = ["P0", "P2", "P-2S", "P1a", "P1b", "P1c", "P1-VI"]
WRONG = {"price_wrong", "abstain_wrong", "FP_hallucination", "FP_desc_as_item", "FP_component_as_item", "FP_inactive_fp",
         "FP_extra_offer"}
NAME_MATCH = 85          # rapidfuzz ratio above which two normalised names are the same item
C3 = ["off_tp_strict", "off_fp_strict", "off_fn_strict"]


# ------------------------------------------------------------------------------------------------ predictions of a run
class Runs:
    """Parsed predictions of stored runs, per base image (test split, clean images, thinking off)."""

    def __init__(self, st, split: str):
        from smv.data.index import load_golds, load_index
        from smv.orchestrator.store import ResultStore
        from smv.scoring.evaluate import EvalConfig

        self.st = st
        self.store = ResultStore(st.results_dir)
        idx = load_index(st)
        idx = idx[(idx["perturbation"].fillna("") == "") & (idx["split"] == split)]
        self.sha_to_image = dict(zip(idx["sha256"], idx["image_id"]))
        self.golds = load_golds(st)
        self.images = [i for i in idx["image_id"] if i in self.golds]
        self.lex = st.lexicon()
        self.words = self.lex.get("words", {})
        self.ecfg = EvalConfig.from_cfg(st.scoring_cfg(), self.lex)
        self._cache: dict[tuple, dict[str, Prediction | None]] = {}

    def has(self, model: str, prompt: str) -> bool:
        return self.store.path(model, prompt).is_file()

    def get(self, model: str, prompt: str, seed: int = 0) -> dict[str, Prediction | None]:
        key = (model, prompt, seed)
        if key not in self._cache:
            from smv.scoring.parse import parse_output

            out: dict[str, Prediction | None] = {}
            for r in self.store.load(model, prompt):
                im = self.sha_to_image.get(r.get("image_sha"))
                if im is None or int(r.get("seed", 0)) != seed or r.get("thinking", "off") != "off":
                    continue
                if r.get("status") != "ok":
                    out.setdefault(im, None)
                    continue
                out[im] = parse_output(r.get("text", ""), "json", r.get("finish_reason", "")).prediction
            self._cache[key] = out
        return self._cache[key]

    def norm(self, name: str | None) -> str:
        return normalize_name(name or "", "lenient", self.words)

    def pairs(self, model: str, prompt: str, seed: int = 0) -> dict[str, list[tuple[str, int | None]]]:
        """Per image, the (normalised item name, price) pairs of a run."""
        out = {}
        for im, pred in self.get(model, prompt, seed).items():
            out[im] = [(self.norm(it.name or it.raw_name), o.price_vnd) for it in (pred.items if pred else [])
                       for o in it.offers]
        return out


def _same(a: str, b: str) -> bool:
    return a == b or (bool(a) and bool(b) and fuzz.ratio(a, b) >= NAME_MATCH)


def _supported(name: str, price, pairs: list[tuple[str, int | None]]) -> bool:
    return any(p == price and _same(name, n) for n, p in pairs)


# ------------------------------------------------------------------------------------------------ signals
def reference_offers(long: pd.DataFrame, model: str, prompt: str, split: str, seed: int = 0) -> pd.DataFrame:
    """Predicted offers with a price of one run, with `ok` = right under the strict rule. Offers paired with a gold offer
    flagged ambiguous are left out (rule R4: they are in no precision or recall)."""
    d = long[(long["model"] == model) & (long["prompt"] == prompt) & (long["split"] == split) & (long["seed"] == seed)
             & (long["thinking"] == "off") & (long["perturbation"].fillna("") == "")]
    d = d[d["pred_price"].notna() & (d["status"].isin(WRONG) | (d["status"] == "TP"))]
    d = d[~(d["gold_ambiguous"].astype(str).str.lower() == "true")]
    return d.assign(ok=(d["status"] == "TP").astype(int))


def agreement(runs: Runs, ref: pd.DataFrame, panel: list[tuple[str, str, int]]) -> pd.Series:
    """Share of the panel runs that return the same item with the same price (runs without an answer for the image do not
    vote)."""
    tables = [runs.pairs(*p) for p in panel]
    vals = []
    for im, name, price in zip(ref["base_image_id"], ref["pred_item"], ref["pred_price"]):
        n = runs.norm(name)
        votes = [_supported(n, int(price), t[im]) for t in tables if im in t]
        vals.append(float(np.mean(votes)) if votes else np.nan)
    return pd.Series(vals, index=ref.index)


def signal_metrics(ok: np.ndarray, conf: np.ndarray, seed: int = 2027) -> dict[str, float]:
    """AUROC of the confidence for "the offer is right", and what a review of the least confident offers finds."""
    from scipy.stats import rankdata

    keep = ~np.isnan(conf)
    ok, conf = ok[keep].astype(int), conf[keep]
    n, n_bad = len(ok), int((ok == 0).sum())
    out = {"n": n, "covered": float(keep.mean()) if len(keep) else float("nan"),
           "error_rate": n_bad / n if n else float("nan")}
    if n == 0 or n_bad == 0 or n_bad == n:
        return {**out, "auroc": float("nan"), "caught10": float("nan"), "caught20": float("nan"), "caught30": float("nan"),
                "precision_at80": float("nan")}
    r = rankdata(conf)
    out["auroc"] = float((r[ok == 1].sum() - (n - n_bad) * (n - n_bad + 1) / 2) / ((n - n_bad) * n_bad))
    order = np.lexsort((np.random.default_rng(seed).random(n), conf))        # least confident first, ties at random
    bad_sorted = (ok[order] == 0)
    for k in (10, 20, 30):
        out[f"caught{k}"] = float(bad_sorted[: int(np.ceil(n * k / 100))].sum() / n_bad)
    kept = order[int(np.ceil(n * 0.2)):]
    out["precision_at80"] = float(ok[kept].mean()) if len(kept) else float("nan")
    return out


def risk_curve(ok: np.ndarray, conf: np.ndarray, seed: int = 2027) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among the offers that are kept when the least confident ones are sent to a person, against the share
    kept (coverage)."""
    keep = ~np.isnan(conf)
    ok, conf = ok[keep].astype(int), conf[keep]
    order = np.lexsort((np.random.default_rng(seed).random(len(ok)), -conf))  # most confident first
    bad = np.cumsum(ok[order] == 0)
    k = np.arange(1, len(ok) + 1)
    return k / len(ok), bad / k


def confidence_study(st, long: pd.DataFrame, runs: Runs, models_main: list[str], split: str, main_p: str,
                     labels: dict[str, str], out_tbl: Path, out_fig: Path) -> list[Path]:
    rows, curves = [], {}
    cands = [m for m in dict.fromkeys(TUNING + ["qwen35_4b"] + models_main) if runs.has(m, main_p)]
    for m in cands:
        sigs: list[tuple[str, np.ndarray, np.ndarray]] = []
        ref = reference_offers(long, m, main_p, split)
        if ref.empty:
            continue
        ok = ref["ok"].to_numpy()
        lp = reference_offers(long, m, f"{main_p}@lp", split)
        if not lp.empty and lp["pred_conf"].notna().any():
            sigs.append(("token probability (item and price)", lp["ok"].to_numpy(), lp["pred_conf"].to_numpy(float)))
            sigs.append(("token probability (price only)", lp["ok"].to_numpy(), lp["pred_conf_price"].to_numpy(float)))
        pc = reference_offers(long, m, "P-Conf", split)
        if not pc.empty and pc["pred_conf"].notna().any():
            sigs.append(("stated confidence (P-Conf)", pc["ok"].to_numpy(), pc["pred_conf"].to_numpy(float)))
        panel = [(m, p, 0) for p in PROMPT_PANEL if runs.has(m, p) and runs.get(m, p)]
        if len(panel) >= 3:
            sigs.append((f"agreement of {len(panel)} other prompts", ok, agreement(runs, ref, panel).to_numpy(float)))
        samples = [(m, f"{main_p}@t07", s) for s in range(1, 6) if runs.has(m, f"{main_p}@t07") and runs.get(m, f"{main_p}@t07", s)]
        if len(samples) >= 3:
            sigs.append((f"agreement of {len(samples)} sampled answers", ok, agreement(runs, ref, samples).to_numpy(float)))
        only_main = m in TUNING or any(s[0].startswith(("token", "stated")) for s in sigs)
        others = [(o, main_p, 0) for o in models_main if o != m]
        if len(others) >= 3 and only_main:
            sigs.append((f"agreement of {len(others)} other models", ok, agreement(runs, ref, others).to_numpy(float)))
        for name, y, c in sigs:
            r = signal_metrics(y, c)
            rows.append({"model": m, "signal": name, **r})
            curves[(m, name)] = risk_curve(y, c)
    if not rows:
        return []
    t = pd.DataFrame(rows)
    t.to_csv(out_tbl.with_name(out_tbl.name + "_values.csv"), index=False)
    show = pd.DataFrame([{"Model": labels.get(r.model, r.model), "Signal": r.signal, "Offers": int(r.n),
                          "Wrong (%)": f"{r.error_rate * 100:.1f}", "AUROC": f"{r.auroc:.3f}",
                          "Errors found, review 10%": f"{r.caught10 * 100:.0f}",
                          "review 20%": f"{r.caught20 * 100:.0f}", "review 30%": f"{r.caught30 * 100:.0f}",
                          "Precision of the 80% kept": f"{r.precision_at80 * 100:.1f}"} for r in t.itertuples()])
    paths = write_table(show, out_tbl, "Which signal finds the wrong offers? (test, strict price rule)", "tab:confidence",
                        note="Every predicted offer with a price is right or wrong under the strict rule. AUROC: chance that a "
                             "right offer gets a higher value than a wrong one (0.5 = no information). Errors found: share of "
                             "all wrong offers among the 10/20/30% least confident. Missed items are not offers and no signal "
                             "sees them.")
    show_models = [m for m in cands if any(k[0] == m for k in curves)][:2]
    if show_models:
        F._style()
        fig, axes = plt.subplots(1, len(show_models), figsize=(3.4 * len(show_models), 2.8), squeeze=False)
        for ax, m in zip(axes[0], show_models):
            for k, ((_mm, name), (cov, risk)) in enumerate([kv for kv in curves.items() if kv[0][0] == m]):
                ax.plot(cov * 100, risk * 100, linewidth=1.5, color=F.SERIES[k % len(F.SERIES)], label=name)
            ax.set_xlabel("Offers accepted without review (%)")
            ax.set_ylabel("Wrong among the accepted (%)")
            ax.set_xlim(20, 100)
            ax.set_title(labels.get(m, m))
            ax.legend(fontsize=5.6, loc="upper left")
        paths += F._save(fig, out_fig)
    return paths


# ------------------------------------------------------------------------------------------------ ensembles
def vote(preds: list[Prediction | None], norm) -> Prediction:
    """Keep an (item, price) pair returned by a strict majority of the members that answered."""
    members = [p for p in preds if p is not None]
    need = len(members) // 2 + 1
    clusters: list[dict] = []
    for k, p in enumerate(members):
        for it in p.items:
            n = norm(it.name or it.raw_name)
            if not n:
                continue
            c = next((c for c in clusters if _same(n, c["key"])), None)
            if c is None:
                c = {"key": n, "names": Counter(), "raws": Counter(), "votes": defaultdict(set), "attrs": defaultdict(Counter)}
                clusters.append(c)
            c["names"][it.name or it.raw_name or ""] += 1
            c["raws"][it.raw_name or it.name or ""] += 1
            for o in it.offers:
                c["votes"][o.price_vnd].add(k)
                c["attrs"][o.price_vnd][(o.size, o.temp, o.portion, o.price_max_vnd, o.raw_price)] += 1
    items = []
    for c in clusters:
        offers = []
        for price, who in c["votes"].items():
            if len(who) >= need:
                size, temp, portion, pmax, raw = c["attrs"][price].most_common(1)[0][0]
                offers.append(PredOffer(size=size, temp=temp, portion=portion, price_vnd=price, price_max_vnd=pmax,
                                        raw_price=raw))
        if offers:
            items.append(PredItem(raw_name=c["raws"].most_common(1)[0][0], name=c["names"].most_common(1)[0][0],
                                  offers=offers))
    return Prediction(items=items)


def _score(runs: Runs, preds: dict[str, Prediction | None]) -> pd.DataFrame:
    from smv.scoring.evaluate import evaluate_image

    rows = []
    for im in runs.images:
        if im not in preds:
            continue
        c, _ = evaluate_image(runs.golds[im], preds[im], runs.ecfg, "strict", False,
                              "strict" if preds[im] is not None else "failed")
        rows.append({"base_image_id": im, **{k: c[k] for k in C3}})
    return pd.DataFrame(rows).set_index("base_image_id") if rows else pd.DataFrame(columns=C3)


def ensemble_table(runs: Runs, models_main: list[str], main_p: str, labels: dict[str, str], out_base: Path,
                   n_boot: int = 4000) -> list[Path]:
    specs: list[tuple[str, list[tuple[str, str, int]]]] = []
    tun = [m for m in TUNING if runs.has(m, main_p)]
    if len(tun) >= 3:
        specs.append((f"{len(tun)} tuning models, {main_p}", [(m, main_p, 0) for m in tun]))
    if len(models_main) >= 5:
        specs.append((f"all {len(models_main)} models of the main table, {main_p}", [(m, main_p, 0) for m in models_main]))
    for m in tun:
        ps = [p for p in [main_p, *PROMPT_PANEL] if runs.has(m, p) and runs.get(m, p)]
        if len(ps) >= 3:
            specs.append((f"{labels.get(m, m)}, {len(ps)} prompts", [(m, p, 0) for p in ps]))
    for m in models_main:
        pid = f"{main_p}@t07"
        seeds = [s for s in range(1, 6) if runs.has(m, pid) and runs.get(m, pid, s)]
        if len(seeds) >= 3:
            specs.append((f"{labels.get(m, m)}, {len(seeds)} sampled answers", [(m, pid, s) for s in seeds]))
    rows = []
    for title, members in specs:
        tables = [runs.get(*k) for k in members]
        images = [im for im in runs.images if sum(im in t for t in tables) == len(tables)]
        if len(images) < 20:
            continue
        ens = _score(runs, {im: vote([t[im] for t in tables], runs.norm) for im in images})
        singles = {k: _score(runs, {im: t[im] for im in images}) for k, t in zip(members, tables)}
        best_k = max(singles, key=lambda k: micro_f1(*(singles[k][c] for c in C3)))
        best = singles[best_k]
        # the reference a reader has in mind: the greedy main run of the same model, if the ensemble is about one model
        one_model = len({k[0] for k in members}) == 1
        ref_name, ref = "best member", best
        if one_model and runs.has(members[0][0], main_p):
            ref_name, ref = f"{main_p}, greedy", _score(runs, {im: runs.get(members[0][0], main_p)[im] for im in images
                                                               if im in runs.get(members[0][0], main_p)})
            ref = ref.reindex(ens.index).fillna(0)
        f, lo, hi = bootstrap_f1(ens[C3[0]], ens[C3[1]], ens[C3[2]], n_boot)
        d = paired_bootstrap(tuple(ens[c].to_numpy() for c in C3), tuple(ref[c].to_numpy() for c in C3), n_boot)
        tp, fp, fn = (float(ens[c].sum()) for c in C3)
        rows.append({"Ensemble (strict majority)": title, "Images": len(images),
                     "Offer F1 [95% CI]": f"{f * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]",
                     "Precision": f"{100 * tp / max(tp + fp, 1):.1f}", "Recall": f"{100 * tp / max(tp + fn, 1):.1f}",
                     "Reference": ref_name, "Reference F1": f"{micro_f1(*(ref[c] for c in C3)) * 100:.1f}",
                     "Δ [95% CI]": f"{d['diff'] * 100:+.1f} [{d['ci_lo'] * 100:+.1f}, {d['ci_hi'] * 100:+.1f}]",
                     "Members F1 (min-max)": "-".join(f"{v * 100:.1f}" for v in (
                         min(micro_f1(*(s[c] for c in C3)) for s in singles.values()),
                         max(micro_f1(*(s[c] for c in C3)) for s in singles.values())))})
    if not rows:
        return []
    return write_table(pd.DataFrame(rows), out_base, "Majority-vote ensembles of answers that already exist (test, strict rule)",
                       "tab:ensembles",
                       note="An (item, price) pair is kept when a strict majority of the members return it (names compared "
                            "after normalisation, without diacritics). Δ: ensemble minus reference on the same images, paired "
                            "bootstrap.")


# ------------------------------------------------------------------------------------------------ driver
def build_trust(st, name: str, df: pd.DataFrame, specs: dict, scfg: dict, main_p: str, split: str) -> list[Path]:
    out = st.repo / "reports" / name
    tdir, fdir = out / "tables", out / "figures"
    lo = st.out_dir / "scores" / name / "long_offers.csv.gz"
    if not lo.is_file():
        return []
    long = pd.read_csv(lo, low_memory=False)
    for col in ("pred_conf", "pred_conf_price", "pred_conf_name"):
        if col not in long.columns:
            long[col] = np.nan
    names = load_display_names(st.repo)
    labels = {m: pretty(m, specs, names) for m in df["model"].unique()}
    main = df[(df["mode"] == "strict") & (df["split"] == split) & (df["prompt"] == main_p) & (df["thinking"] == "off")
              & (df["perturbation"].fillna("") == "") & df["gold_offers"].notna()]
    from smv.report.extras import _order_models

    models_main = _order_models(main) if not main.empty else []
    runs = Runs(st, split)
    paths: list[Path] = []
    errors: list[str] = []
    for label, fn in (("confidence", lambda: confidence_study(st, long, runs, models_main, split, main_p, labels,
                                                              tdir / "table_confidence_signals", fdir / "fig_risk_coverage")),
                      ("ensembles", lambda: ensemble_table(runs, models_main, main_p, labels, tdir / "table_ensembles"))):
        try:
            paths.extend(fn() or [])
        except Exception:  # noqa: BLE001 - one failing asset must not take the others with it
            import traceback

            errors.append(f"== {label}\n{traceback.format_exc()}")
    err = out / "trust_error.txt"
    if errors:
        err.write_text("\n".join(errors))
    elif err.exists():
        err.unlink()
    return paths
