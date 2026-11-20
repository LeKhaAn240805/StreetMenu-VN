"""Model specs (configs/models/<key>.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class VllmCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    max_model_len: int = 16384
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.90
    limit_mm_per_prompt: dict[str, int] | None = Field(default_factory=lambda: {"image": 1})
    trust_remote_code: bool = False
    mm_processor_kwargs: dict[str, Any] | None = None
    max_num_batched_tokens: int | None = None
    enforce_eager: bool = False
    tokenizer_mode: str | None = None
    hf_overrides: dict[str, Any] | None = None
    chat_template: str | None = None     # path relative to the repo, if the model needs a custom template
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class AutotuneCfg(BaseModel):
    max_num_seqs: list[int] = Field(default_factory=lambda: [16, 8, 4, 2, 1])
    max_model_len: list[int] = Field(default_factory=lambda: [16384, 12288, 8192])
    mm_processor_kwargs: list[dict[str, Any]] = Field(default_factory=list)
    preemption_limit: int = 200


class ThinkingCfg(BaseModel):
    supported: bool = False
    default_on: bool = False
    off: dict[str, Any] = Field(default_factory=dict)   # extra request body for thinking=off
    on: dict[str, Any] = Field(default_factory=dict)    # extra request body for thinking=on
    strip: list[str] = Field(default_factory=lambda: ["<think>", "</think>"])
    max_tokens_on: int | None = None


class HfCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    adapter: str = "generic"           # generic | internvl_chat | minicpmv | ovis
    model_class: str = "AutoModelForImageTextToText"
    batch_size: int = 4
    batch_ladder: list[int] = Field(default_factory=lambda: [4, 2, 1])
    python: str = "main"               # main (main) | vllm (vLLM venv)
    dtype: str | None = "bfloat16"
    max_new_tokens: int | None = None
    attn_implementation: str | None = None
    max_num_tiles: int | None = None   # InternVL-style tiling


class OcrCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    engine: str = "vlm_prompt"         # vlm_prompt | paddleocr | easyocr | ppocr_vietocr
    prompt: str = "OCR:"
    system: str | None = None
    output: str = "text"               # text | markdown | json_layout
    lang: str = "vi"
    max_tokens: int = 8192
    options: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    key: str
    hf_id: str | None = None
    revision: str | None = None
    family: str = ""
    org: str = ""
    params_b: float | None = None
    license: str = ""
    gated: bool | str = False
    download_gb: float | None = None
    role: str = "vlm"                  # vlm | ocr | llm
    groups: list[str] = Field(default_factory=list)
    tier: str = "full"                 # pilot | full | optional
    priority: int = 50
    backend: str = "vllm"              # vllm | hf | ocr | mock
    text_only: bool = False
    dtype: str | None = "bfloat16"
    quant: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    keep_weights: bool = False
    enabled: bool = True
    note: str = ""
    vllm: VllmCfg = Field(default_factory=VllmCfg)
    autotune: AutotuneCfg = Field(default_factory=AutotuneCfg)
    thinking: ThinkingCfg = Field(default_factory=ThinkingCfg)
    sampling: dict[str, Any] = Field(default_factory=dict)
    concurrency: int | None = None
    hf: HfCfg = Field(default_factory=HfCfg)
    ocr: OcrCfg | None = None
    fallback_backend: str | None = None   # tried when the primary backend cannot load the model
    mock: dict[str, Any] = Field(default_factory=dict)

    @property
    def needs_download(self) -> bool:
        return bool(self.hf_id) and self.backend != "mock"


def load_model(path_or_dir: Path, key: str | None = None) -> ModelSpec:
    p = path_or_dir / f"{key}.yaml" if key else path_or_dir
    if not p.is_file():
        raise FileNotFoundError(f"model config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return ModelSpec.model_validate(data)


def load_all(models_dir: Path) -> dict[str, ModelSpec]:
    out = {}
    for p in sorted(models_dir.glob("*.yaml")):
        spec = load_model(p)
        out[spec.key] = spec
    return out


def dump_model(spec: ModelSpec, path: Path) -> None:
    data = spec.model_dump(exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
