"""papers/models_table.tex: family, organisation, parameters, licence, pinned revision, date."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from smv.models.registry import load_all
from smv.models.verify import read_verified
from smv.report.tables import write_table
from smv.settings import Settings


def write_models_table(st: Settings, tiers: tuple[str, ...] = ("pilot", "full", "optional"), only_run: bool = True) -> Path:
    """Models of the study. With only_run (default) just the models that have results in the store are listed."""
    from smv.report.extras import load_display_names, pretty

    verified = read_verified(st.model_configs_dir / "VERIFIED.csv")
    specs = load_all(st.model_configs_dir)
    names = load_display_names(st.repo)
    ran = {p.name for p in st.results_dir.iterdir() if p.is_dir() and any(p.glob("*.jsonl"))} if st.results_dir.is_dir() else set()
    rows = []
    for key, s in sorted(specs.items(), key=lambda kv: (kv[1].role, kv[1].family, kv[1].params_b or 0, kv[0])):
        if s.backend == "mock" or s.tier not in tiers or not s.enabled or (only_run and key not in ran):
            continue
        v = verified.get(key, {})
        rows.append({"Model": pretty(key, specs, names), "Family": s.family, "Role": s.role,
                     "Params (B)": f"{s.params_b:g}" if s.params_b else "–", "Weights": s.quant or "bf16",
                     "Licence": s.license or v.get("license", ""), "HF repo": s.hf_id or "pip",
                     "Revision": (s.revision or "")[:8]})
    df = pd.DataFrame(rows)
    paths = write_table(df, st.repo / "papers" / "models_table", "Models evaluated (revisions pinned)", "tab:models",
                        "lllrllll")
    return paths[1]
