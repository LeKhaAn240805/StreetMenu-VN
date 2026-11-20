#!/usr/bin/env python
"""Pre-download upcoming model weights, one at a time, using the pipeline's own downloader.

Why: the runner only prefetches ONE model ahead and only WITHIN a plan, so the first model of each new
plan (and the model after an NFS stall) downloads with the GPU idle. Staging a few models ahead across
those boundaries removes that idle gap. Safe to run alongside the lanes: it writes the same completion
marker `ensure_weights` writes, so when a lane reaches the model it sees it is complete and skips the
download. Downloads sequentially to avoid starving a lane's own live download of NFS bandwidth.

    python scripts/prestage.py qwen35_9b ministral3_8b gemma4_31b_awq ...
"""
import sys
import time

from smv.settings import build_settings
from smv.models.registry import load_model
from smv.runtime.weights import ensure_weights, is_complete, repo_size_gb


def main(keys: list[str]) -> int:
    st = build_settings()
    for key in keys:
        try:
            spec = load_model(st.model_configs_dir, key)
        except Exception as e:  # noqa: BLE001
            print(f"[prestage] {key}: cannot load spec: {e}", flush=True)
            continue
        if is_complete(st, spec):
            print(f"[prestage] {key}: already on disk, skip", flush=True)
            continue
        try:
            gb = repo_size_gb(spec)
        except Exception:  # noqa: BLE001
            gb = -1
        print(f"[prestage] {key}: downloading ({gb:.1f} GB) {spec.hf_id} ...", flush=True)
        t0 = time.time()
        try:
            ensure_weights(st, spec)
            print(f"[prestage] {key}: done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[prestage] {key}: FAILED {e}", flush=True)
    print("[prestage] all requested models processed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
