"""Systematic, text-only error sample for manual E1-E8 coding (no images)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from smv.settings import Settings


def sample_errors(st: Settings, name: str = "all", n_per_model: int = 40, seed: int = 2027) -> Path:
    src = st.out_dir / "scores" / name / "long_offers.csv.gz"
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found: run `smv score` first")
    df = pd.read_csv(src)
    df = df[~df["status"].isin(["TP", "abstain_ok", "unscored", "range_second_end"])]
    out_rows = []
    for _key, g in df.groupby(["model", "prompt"]):
        # stratify by suggested code so rare error types are represented
        per = max(1, n_per_model // max(g["error_hint"].nunique(), 1))
        parts = [x.sample(min(len(x), per), random_state=seed) for _, x in g.groupby(g["error_hint"].fillna(""))]
        s = pd.concat(parts).head(n_per_model)
        out_rows.append(s)
    out = pd.concat(out_rows) if out_rows else df.head(0)
    out = out.assign(error_code_manual="", annotator_note="")
    path = st.repo / "reports" / name / "error_sample_for_coding.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
