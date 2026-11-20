#!/usr/bin/env python3
"""Summarise a trial run (status.csv + manifests + scores) into a Markdown table.

usage: python scripts/trial_summary.py [outputs/_trial] [trial_all] > docs/TRIAL_REPORT.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/_trial")
    plan = sys.argv[2] if len(sys.argv) > 2 else "trial_all"
    status = {r["model_key"]: r for r in csv.DictReader(open(out / plan / "status.csv", encoding="utf-8"))}
    man = {}
    for p in sorted((out / plan).glob("*/manifests/*.json")):
        m = json.loads(p.read_text())
        man[m["model_key"]] = m          # last run wins
    summ = out / "scores" / plan / "summary.csv"
    s = pd.read_csv(summ) if summ.is_file() else pd.DataFrame()
    if not s.empty:
        s = s[(s["mode"] == "strict") & (s["thinking"] == "off")]
    rows = []
    for key, st in status.items():
        m = man.get(key, {})
        tf = m.get("tune_final") or {}

        def f1(prompt: str, key: str = key) -> str:
            if s.empty:
                return ""
            v = s[(s["model"] == key) & (s["prompt"] == prompt)]["offer_F1_strict"]
            return f"{100 * v.iloc[0]:.1f}" if len(v) and v.iloc[0] == v.iloc[0] else ""

        rows.append({
            "model": key, "state": st["state"], "ok/jobs": f"{st['n_done']}/{st['n_jobs']}",
            "backend": m.get("backend", st.get("backend", "")),
            "seqs": tf.get("max_num_seqs", ""), "len": tf.get("max_model_len", ""),
            "start s": tf.get("start_s", ""), "VRAM GB": f"{m['vram_peak_mib'] / 1024:.1f}" if m.get("vram_peak_mib") else "",
            "req/h": m.get("throughput_req_per_h") or "", "F1 P1": f1("P1"), "F1 OCR-REGEX": f1("OCR-REGEX"),
            "F1 P-OCR+": f1("P-OCR+@ocr_ppocr"), "message": (st.get("message") or "")[:90],
        })
    df = pd.DataFrame(rows)
    print(f"# Trial run `{plan}` on synthetic menus\n")
    print(f"{(df['state'] == 'done').sum()} done, {(df['state'] != 'done').sum()} not done, {len(df)} models.\n")
    print("| " + " | ".join(df.columns) + " |")
    print("|" + "|".join("---" for _ in df.columns) + "|")
    for _, r in df.iterrows():
        print("| " + " | ".join(str(x) for x in r.values) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
