"""Download model weights into $SMV_MODELS_DIR/<key> (pinned revision, no shared cache), prefetch in the
background, and delete after a model is done."""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from smv.models.registry import ModelSpec
from smv.runtime.disk import DiskFullError, can_fit, clean_xet_cache, dir_size_gb, safe_rmtree
from smv.settings import Settings

log = logging.getLogger("smv.weights")
MARKER = ".smv_download_complete.json"


def model_dir(st: Settings, key: str) -> Path:
    return st.models_dir / key


def _matches(name: str, allow: list[str] | None, ignore: list[str] | None) -> bool:
    if allow and not any(fnmatch.fnmatch(name, p) for p in allow):
        return False
    if ignore and any(fnmatch.fnmatch(name, p) for p in ignore):
        return False
    return True


def repo_size_gb(spec: ModelSpec, token: str | None = None) -> float:
    """Total size of the files that will be downloaded (model_info(files_metadata=True))."""
    if spec.backend == "mock":
        return float(spec.mock.get("fake_weights_mb", 1)) / 1000
    if not spec.hf_id:
        return float(spec.download_gb or 1.0)
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(spec.hf_id, revision=spec.revision, files_metadata=True)
    total = sum((s.size or 0) for s in info.siblings if _matches(s.rfilename, spec.allow_patterns, spec.ignore_patterns))
    return total / 1e9


def is_complete(st: Settings, spec: ModelSpec) -> bool:
    m = model_dir(st, spec.key) / MARKER
    if not m.is_file():
        return False
    try:
        data = json.loads(m.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("revision") == (spec.revision or "") and data.get("hf_id") == (spec.hf_id or "")


def _write_marker(st: Settings, spec: ModelSpec, seconds: float) -> None:
    d = model_dir(st, spec.key)
    d.mkdir(parents=True, exist_ok=True)
    (d / MARKER).write_text(json.dumps({"hf_id": spec.hf_id or "", "revision": spec.revision or "",
                                        "seconds": round(seconds, 1), "size_gb": round(dir_size_gb(d), 3),
                                        "ts": time.time()}))


def ensure_weights(st: Settings, spec: ModelSpec, size_gb: float | None = None) -> Path:
    """Download if needed (checks free space first). Returns the local directory."""
    d = model_dir(st, spec.key)
    if is_complete(st, spec):
        return d
    dcfg = st.cfg.get("download", {})
    if size_gb is None:
        size_gb = repo_size_gb(spec, os.environ.get("HF_TOKEN") or None)
    have = dir_size_gb(d) if d.exists() else 0.0
    ok, free, req = can_fit(st.models_dir, max(size_gb - have, 0.0), float(dcfg.get("margin", 0.2)),
                            float(dcfg.get("min_free_gb", 30)))
    if not ok:
        raise DiskFullError(f"not enough disk for {spec.key}: free {free:.1f} GB < required {req:.1f} GB "
                            f"(model {size_gb:.1f} GB + {dcfg.get('margin', 0.2):.0%} + min_free_gb)")
    t0 = time.time()
    d.mkdir(parents=True, exist_ok=True)
    if spec.backend == "mock":
        mb = int(spec.mock.get("fake_weights_mb", 1))
        with open(d / "fake_model.safetensors", "wb") as f:
            f.write(os.urandom(1024) * (mb * 1024))
        (d / "config.json").write_text(json.dumps({"architectures": ["MockForCausalLM"]}))
    elif spec.hf_id:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=spec.hf_id, revision=spec.revision, local_dir=str(d),
                          allow_patterns=spec.allow_patterns, ignore_patterns=spec.ignore_patterns,
                          max_workers=int(dcfg.get("max_workers", 8)), token=os.environ.get("HF_TOKEN") or None)
    # OCR engines without an HF repo download into this directory themselves at first use
    _write_marker(st, spec, time.time() - t0)
    return d


def delete_weights(st: Settings, spec: ModelSpec) -> list[str]:
    removed = []
    d = model_dir(st, spec.key)
    if d.exists() or d.is_symlink():
        safe_rmtree(d, st.models_dir)
        removed.append(str(d))
    removed += clean_xet_cache(st.hf_home)
    return removed


@dataclass
class Prefetch:
    key: str
    proc: subprocess.Popen
    log_path: Path

    def done(self) -> bool:
        return self.proc.poll() is not None

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)


def start_prefetch(st: Settings, spec: ModelSpec, log_path: Path, env: dict[str, str]) -> Prefetch:
    """Download the next model in a background process (same code path as ensure_weights)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    cmd = [sys.executable, "-m", "smv.runtime.weights", "download", spec.key]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(st.repo),
                            start_new_session=True)
    return Prefetch(spec.key, proc, log_path)


def _main(argv: list[str]) -> int:
    from smv.models.registry import load_model
    from smv.settings import get_settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(argv) != 2 or argv[0] != "download":
        print("usage: python -m smv.runtime.weights download <model_key>", file=sys.stderr)
        return 2
    st = get_settings()
    spec = load_model(st.model_configs_dir, argv[1])
    try:
        ensure_weights(st, spec)
    except Exception as e:  # noqa: BLE001
        print(f"PREFETCH_FAILED {type(e).__name__}: {e}", flush=True)
        return 1
    print("PREFETCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
