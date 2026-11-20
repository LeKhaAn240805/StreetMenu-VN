#!/usr/bin/env python3
"""Freeze prompts after a format-health check of their dev runs (no accuracy is looked at, nothing is tuned).

    python scripts/freeze_if_sane.py --prompts P-OCR+ P-Oracle --models llm_qwen3_8b llm_vylinh_3b --min-valid 0.8
    python scripts/freeze_if_sane.py --prompts P1-C --models qwen35_4b qwen35_2b --compact P1-C:P1:0.75
    python scripts/freeze_if_sane.py --prompts P-Conf --models qwen35_4b qwen35_2b --need-confidence P-Conf

For every model and every result id that starts with one of the prompts, the share of dev requests that returned a parseable
JSON object must reach --min-valid. Optional format checks on the same dev records:
  --compact NEW:BASE:RATIO   on every model NEW must return at least 85% as many offers as BASE (nothing was dropped), and
                             on at least half of the models it must spend at most RATIO times the output tokens per offer
                             of BASE (the layout really is shorter; a model that ignores the layout is not an error)
  --need-confidence PROMPT   at least 80% of the offers of PROMPT carry a numeric confidence
All good -> the prompts are frozen (exit 0). Otherwise nothing is frozen (exit 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smv.orchestrator.store import ResultStore  # noqa: E402
from smv.prompts.registry import PromptRegistry  # noqa: E402
from smv.scoring.parse import parse_output  # noqa: E402
from smv.settings import get_settings  # noqa: E402


def dev_stats(store: ResultStore, model: str, pid: str, dev: set[str]) -> dict[str, float]:
    """Counts over the latest dev record of every image: records, parseable, output tokens, offers, offers with confidence."""
    latest: dict[tuple, dict] = {}
    for r in store.load(model, pid):
        if r.get("image_sha") in dev and r.get("thinking", "off") == "off":
            k = (r["image_sha"], int(r.get("seed", 0)))
            if k not in latest or r.get("status") == "ok" or latest[k].get("status") != "ok":
                latest[k] = r
    out = {"n": len(latest), "good": 0, "tokens": 0, "offers": 0, "with_conf": 0, "truncated": 0}
    for r in latest.values():
        if r.get("status") != "ok":
            continue
        pr = parse_output(r.get("text", ""), "json", r.get("finish_reason", ""))
        if pr.prediction is None or pr.status == "failed":
            continue
        out["good"] += 1
        out["tokens"] += int(r.get("completion_tokens") or 0)
        out["truncated"] += int(bool(pr.truncated))
        for it in pr.prediction.items:
            out["offers"] += len(it.offers)
            out["with_conf"] += sum(1 for o in it.offers if o.confidence is not None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--min-valid", type=float, default=0.8)
    ap.add_argument("--compact", default=None, help="NEW:BASE:RATIO, output tokens per offer of NEW against BASE")
    ap.add_argument("--need-confidence", default=None, help="prompt whose offers must carry a numeric confidence")
    a = ap.parse_args()
    st = get_settings()
    store = ResultStore(st.results_dir)
    dev = {r.sha256 for r in __import__("smv.data.index", fromlist=["load_index"]).load_index(st).itertuples(index=False)
           if r.split == "dev"}
    ok = True
    compact_votes: list[bool] = []
    for model in a.models:
        seen = {pid.split("@")[0] for pid in store.prompts_of(model)} & set(a.prompts)
        for missing in sorted(set(a.prompts) - seen):          # every prompt must have been run by every model on dev
            print(f"{model} {missing}: no dev results at all")
            ok = False
        for pid in store.prompts_of(model):
            if pid.split("@")[0] not in a.prompts:
                continue
            s = dev_stats(store, model, pid, dev)
            if not s["n"]:
                print(f"{model} {pid}: no dev results")
                ok = False
                continue
            share = s["good"] / s["n"]
            print(f"{model} {pid}: {s['good']}/{s['n']} parseable ({100 * share:.0f}%), {s['truncated']} truncated")
            ok &= share >= a.min_valid
        if a.compact:
            new, base, ratio = a.compact.split(":")
            sn, sb = dev_stats(store, model, new, dev), dev_stats(store, model, base, dev)
            if not sn["offers"] or not sb["offers"]:
                print(f"{model}: compact check impossible (offers {new}={sn['offers']}, {base}={sb['offers']})")
                ok = False
            else:
                tn, tb = sn["tokens"] / sn["offers"], sb["tokens"] / sb["offers"]
                kept = sn["offers"] >= 0.85 * sb["offers"]
                compact_votes.append(tn <= float(ratio) * tb)
                print(f"{model} {new} vs {base}: {tn:.0f} vs {tb:.0f} output tokens per offer, {sn['offers']} vs "
                      f"{sb['offers']} offers, truncated {sn['truncated']} vs {sb['truncated']} -> "
                      f"{'compact' if compact_votes[-1] else 'not compact'}{'' if kept else ', OFFERS LOST'}")
                ok &= kept
        if a.need_confidence:
            s = dev_stats(store, model, a.need_confidence, dev)
            share = s["with_conf"] / s["offers"] if s["offers"] else 0.0
            print(f"{model} {a.need_confidence}: {100 * share:.0f}% of {s['offers']} offers carry a confidence")
            ok &= share >= 0.8
    if a.compact and compact_votes and sum(compact_votes) * 2 < len(compact_votes):
        print(f"compact layout followed by only {sum(compact_votes)} of {len(compact_votes)} models")
        ok = False
    if not ok:
        print("format check failed: nothing frozen")
        return 1
    fz = PromptRegistry(st.prompts_dir).freeze(a.prompts)
    print("frozen:", {k: v[:12] for k, v in fz.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
