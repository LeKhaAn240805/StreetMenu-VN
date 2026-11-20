"""`smv plan estimate`: GPU hours and download size per model, from throughput measured in manifests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smv.orchestrator.runner import Runner
from smv.settings import Settings

DEFAULT_REQ_PER_H = {"vlm": 600.0, "ocr": 1500.0, "llm": 2000.0}


def _measured(st: Settings) -> dict[str, float]:
    """Best measured requests/hour per model over every manifest (any plan)."""
    out: dict[str, float] = {}
    for p in st.out_dir.glob("*/*/manifests/*.json"):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        v = m.get("throughput_req_per_h")
        if v:
            out[m["model_key"]] = max(out.get(m["model_key"], 0.0), float(v))
    return out


def estimate_plan(st: Settings, plan_path: Path) -> pd.DataFrame:
    r = Runner(st, plan_path, dry_run=True)
    plans = r.expand()
    meas = _measured(st)
    rows = []
    for mp in plans:
        if mp.spec is None:
            continue
        rate = meas.get(mp.key)
        src = "measured" if rate else "default"
        rate = rate or DEFAULT_REQ_PER_H.get(mp.spec.role, 600.0)
        todo = len(mp.todo)
        rows.append({"model": mp.key, "role": mp.spec.role, "requests_todo": todo, "blocked": len(mp.blocked),
                     "req_per_h": round(rate, 0), "rate_source": src, "gpu_h": round(todo / rate + 0.1, 2) if todo else 0.0,
                     "download_gb": mp.spec.download_gb or 0.0})
    df = pd.DataFrame(rows)
    if not df.empty:
        total = {"model": "TOTAL", "role": "", "requests_todo": int(df["requests_todo"].sum()),
                 "blocked": int(df["blocked"].sum()), "req_per_h": None, "rate_source": "",
                 "gpu_h": round(df["gpu_h"].sum(), 2), "download_gb": round(df["download_gb"].sum(), 1)}
        df = df.astype({"req_per_h": "object"})
        df.loc[len(df)] = total
    return df
