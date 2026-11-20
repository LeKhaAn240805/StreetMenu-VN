"""Environment for child processes: GPU pinning, CPU affinity, thread caps, caches off $HOME."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from smv.settings import Settings

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM")


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def cache_env(st: Settings) -> dict[str, str]:
    root = st.cache_root
    return {
        "HF_HOME": str(st.hf_home),
        "HF_HUB_CACHE": str(st.hf_home / "hub"),
        "XDG_CACHE_HOME": str(root / "xdg"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
        "VLLM_CONFIG_ROOT": str(root / "vllm_config"),
        "TRITON_CACHE_DIR": str(root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
        "TORCH_HOME": str(root / "torch"),
        "CUDA_CACHE_PATH": str(root / "cuda_cache"),
        "FLASHINFER_WORKSPACE_BASE": str(root / "flashinfer"),
        "MPLCONFIGDIR": str(root / "matplotlib"),
        "PADDLE_PDX_CACHE_HOME": str(root / "paddlex"),
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "EASYOCR_MODULE_PATH": str(root / "easyocr"),
        "UV_CACHE_DIR": str(root / "uv"),
        "PIP_CACHE_DIR": str(root / "pip"),
        "NUMBA_CACHE_DIR": str(root / "numba"),
        "OUTLINES_CACHE_DIR": str(root / "outlines"),
    }


def child_env(st: Settings, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(cache_env(st))
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(st.gpu_id)
    n = str(st.num_threads)
    for v in THREAD_VARS:
        env[v] = "false" if v == "TOKENIZERS_PARALLELISM" else n
    env["SMV_NUM_THREADS"] = n
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    for k in ("XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def taskset_prefix(st: Settings) -> list[str]:
    if shutil.which("taskset") is None:
        return []
    return ["taskset", "-c", st.cpu_affinity]


def apply_in_process(st: Settings) -> None:
    """Pin the current process (orchestrator / scorer) to the configured CPUs and thread caps."""
    os.environ.update(cache_env(st))
    for v in THREAD_VARS:
        os.environ[v] = "false" if v == "TOKENIZERS_PARALLELISM" else str(st.num_threads)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(st.gpu_id)
    try:
        cpus = parse_cpu_list(st.cpu_affinity)
        if cpus and hasattr(os, "sched_setaffinity"):
            avail = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(cpus)
            use = [c for c in cpus if c in avail] or list(avail)
            os.sched_setaffinity(0, use)
    except (OSError, ValueError):
        pass
    try:
        import torch  # noqa: F401  (only if already importable; never required)

        torch.set_num_threads(st.num_threads)
    except Exception:  # noqa: BLE001
        pass
