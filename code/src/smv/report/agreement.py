"""Inter-annotator agreement: Offer F1 between label sets (same scorer as models), price match,
Cohen kappa (two annotators) and Fleiss kappa (>= 3) for CCL and every tag."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from smv.data.index import load_agreement_sets, load_golds
from smv.report.tables import fmt, write_table
from smv.schema.config import load_schema_config
from smv.schema.prediction import annotation_to_prediction
from smv.scoring.evaluate import EvalConfig, evaluate_image, prf
from smv.settings import Settings
from smv.stats.bootstrap import cohen_kappa, fleiss_kappa


def agreement_report(st: Settings) -> pd.DataFrame:
    golds = load_golds(st)
    sets = load_agreement_sets(st)
    if not sets:
        return pd.DataFrame()
    ecfg = EvalConfig.from_cfg(st.scoring_cfg(), st.lexicon())
    cfg = load_schema_config(st.schema_config_path)
    rows = []
    per_image_raters: dict[str, list] = defaultdict(list)
    for ann_name, anns in sets.items():
        tp = fp = fn = paired = price_ok = 0
        ccl_a, ccl_b, tags_a, tags_b = [], [], [], []
        for iid, other in anns.items():
            gold = golds.get(iid)
            if gold is None:
                continue
            c, _ = evaluate_image(gold, annotation_to_prediction(other), ecfg, "strict")
            tp += c.get("off_tp_strict", 0)
            fp += c.get("off_fp_strict", 0)
            fn += c.get("off_fn_strict", 0)
            paired += c.get("paired", 0)
            price_ok += c.get("pna_ok_strict", 0)
            ccl_a.append(gold.ccl)
            ccl_b.append(other.ccl)
            tags_a.append(set(gold.tags))
            tags_b.append(set(other.tags))
            per_image_raters[iid].append(other)
        p, r, f = prf(tp, fp, fn)
        row = {"Annotator vs gold": ann_name, "Images": len(ccl_a), "Offer F1": fmt(f), "Price match": fmt(price_ok / paired if paired else float("nan")),
               "kappa CCL": f"{cohen_kappa(ccl_a, ccl_b):.2f}" if ccl_a else "–"}
        kappas = []
        for tag in cfg.tags:
            a = [tag in t for t in tags_a]
            b = [tag in t for t in tags_b]
            if any(a) or any(b):
                k = cohen_kappa(a, b)
                kappas.append(k)
                row[f"kappa {tag}"] = f"{k:.2f}"
        row["kappa tags (mean)"] = f"{sum(kappas) / len(kappas):.2f}" if kappas else "–"
        rows.append(row)
    df = pd.DataFrame(rows).fillna("–")
    # Fleiss kappa over images with >= 3 raters (gold + >= 2 others)
    multi = {iid: [golds[iid], *others] for iid, others in per_image_raters.items() if len(others) >= 2 and iid in golds}
    if multi:
        df.attrs["fleiss_ccl"] = fleiss_kappa([[a.ccl for a in anns] for anns in multi.values()])
    out = st.repo / "reports" / "tables"
    main_cols = [c for c in df.columns if not c.startswith("kappa ") or c in ("kappa CCL", "kappa tags (mean)")]
    write_table(df[main_cols], out / "agreement", "Inter-annotator agreement", "tab:agreement")
    df.to_csv(out / "agreement_per_tag.csv", index=False)
    return df[main_cols]
