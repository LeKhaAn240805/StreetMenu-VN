"""Run manifests (reproducibility record) and environment versions."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from smv.settings import Settings

_VERSIONS_SNIPPET = r"""
import json, sys
out = {"python": sys.version.split()[0]}
for m in ("vllm", "torch", "transformers", "paddleocr", "paddle", "easyocr", "vietocr"):
    try:
        mod = __import__(m)
        out[m] = getattr(mod, "__version__", "?")
    except Exception:
        pass
try:
    import torch
    out["cuda"] = torch.version.cuda
except Exception:
    pass
print(json.dumps(out))
"""


def git_commit(repo: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=20).stdout.strip()
        return (sha or "no-commit") + ("+dirty" if dirty else "")
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


@lru_cache(maxsize=8)
def python_versions(python: str) -> dict[str, Any]:
    if not Path(python).exists():
        return {"error": f"{python} not found"}
    try:
        out = subprocess.run([python, "-c", _VERSIONS_SNIPPET], capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError) as e:
        return {"error": str(e)[:200]}


def gpu_info(gpu_id: int) -> dict[str, Any]:
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu_id), "--query-gpu=name,driver_version,memory.total,pci.bus_id",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        name, drv, mem, bus = [p.strip() for p in out.split(",")]
        return {"index": gpu_id, "name": name, "driver": drv, "memory_total": mem, "pci_bus_id": bus}
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return {"index": gpu_id}


def env_versions(st: Settings, backend: str) -> dict[str, Any]:
    py = {"vllm": st.vllm_python, "ocr": st.ocr_python}.get(backend, st.main_python)
    return {"backend_python": str(py), "backend": python_versions(str(py)),
            "orchestrator": python_versions(str(st.main_python)), "host": platform.node()}


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
