"""vLLM general plugin with small compatibility shims (loaded in every vLLM process).

1. vLLM 0.32 + Triton 3.7: the DeepEncoder relative-position attention kernel (DeepSeek-OCR, DeepSeek-OCR-2)
   reads the module-level float LOG2E inside @triton.jit; Triton >= 3.7 only accepts globals that are
   `tl.constexpr`. Wrapping the same value in tl.constexpr keeps the maths identical.
2. transformers' remote-code import check is regex based: it also lists imports that sit inside a function
   on a code path we never take (Molmo: `import tensorflow` only for resize_method="tensorflow"; PhoGPT/MPT:
   `triton_pre_mlir` only for attn_impl="triton"). Those names are dropped from the check; a real use would
   still fail at call time with a normal ImportError.
Each shim is idempotent and silently skipped when the module or attribute is absent (other vLLM versions).
"""
from __future__ import annotations

import logging

log = logging.getLogger("smv_vllm_patches")


def _constexpr_log2e() -> None:
    try:
        import triton.language as tl
        from vllm.model_executor.models import deepencoder
    except Exception:  # noqa: BLE001 - module absent in this vLLM version
        return
    val = getattr(deepencoder, "LOG2E", None)
    if isinstance(val, float):
        deepencoder.LOG2E = tl.constexpr(val)
        log.info("smv_vllm_patches: deepencoder.LOG2E wrapped in tl.constexpr")


# lazily imported, optional code paths only (see 2. above); keep in sync with smv/backends/hf_worker.py
OPTIONAL_REMOTE_IMPORTS = frozenset({"tensorflow", "triton_pre_mlir"})


def _relax_remote_import_check() -> None:
    try:
        from transformers import dynamic_module_utils as dmu
    except Exception:  # noqa: BLE001
        return
    orig = dmu.get_imports
    if getattr(orig, "_smv_relaxed", False):
        return

    def get_imports(filename):
        return [m for m in orig(filename) if m.split(".")[0] not in OPTIONAL_REMOTE_IMPORTS]

    get_imports._smv_relaxed = True
    dmu.get_imports = get_imports


def register() -> None:
    _constexpr_log2e()
    _relax_remote_import_check()
