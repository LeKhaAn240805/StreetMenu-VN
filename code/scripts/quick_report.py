#!/usr/bin/env python3
"""Console summary of scored results (after `smv score --plan ...`): headline metrics per model/prompt/split,
parse health, price-error mix, and the worst images. Text only.

    python scripts/quick_report.py tune                 # scores/<plan>/ ; split dev
    python scripts/quick_report.py pilot test P1 --worst 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smv.settings import get_settings  # noqa: E402

COLS = ["model", "prompt", "split", "n_images", "offer_P_strict", "offer_R_strict", "offer_F1_strict", "offer_F1_lenient",
        "item_F1", "PNA_strict", "NNA", "IPA_strict", "null_price_ok", "magnitude_err_rate", "valid_json_strict",
        "valid_json_repaired", "truncation_rate", "repetition_rate", "completion_tokens_mean", "latency_median_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("split", nargs="?", default="dev")
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--worst", type=int, default=0, help="list the N images with the lowest Offer F1 per model")
    a = ap.parse_args()
    st = get_settings()
    sdir = st.out_dir / "scores" / a.plan
    s = pd.read_csv(sdir / "summary.csv")
    s = s[(s["split"] == a.split) & (s["mode"] == "strict") & (s["thinking"] == "off")]
    if a.prompt:
        s = s[s["prompt"] == a.prompt]
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(s[[c for c in COLS if c in s.columns]].round(3).to_string(index=False))
    per = pd.read_parquet(sdir / "per_image.parquet")
    per = per[(per["split"] == a.split) & (per["mode"] == "strict") & (per["thinking"] == "off")]
    if a.prompt:
        per = per[per["prompt"] == a.prompt]
    print("\nparse status:")
    print(per.groupby(["model", "prompt"])["parse_status"].value_counts().unstack(fill_value=0).to_string())
    lo = sdir / "long_offers.csv.gz"
    if lo.is_file():
        long = pd.read_csv(lo)
        long = long[(long["split"] == a.split) & (long["thinking"] == "off")]
        if a.prompt:
            long = long[long["prompt"] == a.prompt]
        print("\nerror hints (offers):")
        e = long[long["error_hint"].fillna("") != ""]
        print(e.groupby(["model", "prompt", "error_hint"]).size().unstack(fill_value=0).to_string())
        print("\nstatus counts:")
        print(long.groupby(["model", "prompt", "status"]).size().unstack(fill_value=0).to_string())
    if a.worst:
        per = per.assign(den=2 * per["off_tp_strict"] + per["off_fp_strict"] + per["off_fn_strict"])
        per["f1"] = (2 * per["off_tp_strict"] / per["den"].where(per["den"] > 0)).fillna(1.0)
        for (m, p), g in per.groupby(["model", "prompt"]):
            w = g.sort_values("f1").head(a.worst)
            print(f"\nworst images {m} {p}:")
            print(w[["base_image_id", "gold_offers", "pred_offers", "off_tp_strict", "off_fp_strict", "off_fn_strict", "f1",
                     "parse_status", "truncated"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
