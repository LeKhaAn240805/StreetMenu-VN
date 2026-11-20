#!/usr/bin/env python3
"""Gate for the image-budget study: did the per-request image budget really change the visual tokens?

For every model in configs/plans/budget_probe.yaml, compare the mean prompt tokens of the probe run (P1@<smallest tag>) with
the model's ordinary P1 run on the same images. Exit 0 when every model shows at least MIN_DROP fewer prompt tokens, 1 when a
model ignored the kwargs (then the full budget plan would only repeat the default run). Prints one line per model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from smv.settings import get_settings  # noqa: E402

MIN_DROP = 0.15


def mean_tokens(path: Path, shas: set[str] | None = None) -> tuple[float, set[str]]:
    vals, seen = [], set()
    if not path.is_file():
        return float("nan"), seen
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("prompt_tokens") and (shas is None or r["image_sha"] in shas):
            vals.append(float(r["prompt_tokens"]))
            seen.add(r["image_sha"])
    return (sum(vals) / len(vals) if vals else float("nan")), seen


def main() -> int:
    st = get_settings()
    plan = yaml.safe_load((REPO / "configs" / "plans" / "budget_probe.yaml").read_text(encoding="utf-8"))
    ok_all = True
    for stage in plan["stages"]:
        for model in stage["models"]:
            for entry in stage["prompts"]:
                pid = f"{entry['id']}@{entry['tag']}"
                probe, shas = mean_tokens(st.results_dir / model / f"{pid}.jsonl")
                base, _ = mean_tokens(st.results_dir / model / f"{entry['id']}.jsonl", shas)
                if not shas or probe != probe or base != base:
                    print(f"{model} {pid}: no usable records (probe={probe}, default={base})")
                    ok_all = False
                    continue
                drop = 1 - probe / base
                good = drop >= MIN_DROP
                ok_all &= good
                print(f"{model} {pid}: prompt tokens {probe:.0f} vs default {base:.0f} on {len(shas)} images "
                      f"(drop {100 * drop:.0f}%) -> {'budget honoured' if good else 'IGNORED'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
