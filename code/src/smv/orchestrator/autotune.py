"""OOM ladder: lower max_num_seqs, then max_model_len, then the image-token limit."""
from __future__ import annotations

from smv.backends.vllm_server import TuneState
from smv.models.registry import ModelSpec


def initial_tune(spec: ModelSpec) -> TuneState:
    v = spec.vllm
    return TuneState(max_num_seqs=v.max_num_seqs, max_model_len=v.max_model_len,
                     mm_processor_kwargs=v.mm_processor_kwargs, gpu_memory_utilization=v.gpu_memory_utilization)


def _lower(values: list[int], cur: int) -> int | None:
    lower = sorted({x for x in values if x < cur}, reverse=True)
    return lower[0] if lower else None


def next_tune(spec: ModelSpec, cur: TuneState, failure: str) -> TuneState | None:
    """Next configuration after a failure of kind `failure` (oom | kv_too_small | preemption)."""
    at = spec.autotune
    seqs = list(at.max_num_seqs) or [cur.max_num_seqs]
    lens = list(at.max_model_len) or [cur.max_model_len]
    mm_steps = list(at.mm_processor_kwargs)

    def with_(**kw) -> TuneState:
        d = cur.as_dict()
        d.pop("autotune_step", None)
        d.pop("note", None)
        d.update(kw)
        return TuneState(step=cur.step + 1, note=f"after {failure}", **d)

    def lower_seqs():
        n = _lower(seqs, cur.max_num_seqs)
        return with_(max_num_seqs=n) if n else None

    def lower_len():
        n = _lower(lens, cur.max_model_len)
        return with_(max_model_len=n) if n else None

    def lower_mm():
        if not mm_steps:
            return None
        try:
            idx = mm_steps.index(cur.mm_processor_kwargs) + 1 if cur.mm_processor_kwargs in mm_steps else 0
        except ValueError:
            idx = 0
        if idx >= len(mm_steps):
            return None
        base = dict(spec.vllm.mm_processor_kwargs or {})
        base.update(mm_steps[idx])
        return with_(mm_processor_kwargs=base)

    order = {"kv_too_small": (lower_len, lower_seqs, lower_mm),
             "preemption": (lower_seqs, lower_len, lower_mm)}.get(failure, (lower_seqs, lower_len, lower_mm))
    for fn in order:
        t = fn()
        if t is not None:
            return t
    return None
