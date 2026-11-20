"""Plans (configs/plans/*.yaml) and their expansion into concrete requests."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROLE_ORDER = {"ocr": 0, "vlm": 1, "llm": 2}


class PromptEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    optional: bool = False
    ocr_from: list[str] = Field(default_factory=list)
    after: str | None = None
    seeds: list[int] | None = None
    thinking: list[str] | None = None
    # Per-request vLLM mm_processor_kwargs (image budget). The result id becomes "<id>@<tag>", e.g. P1@px1m, so the same
    # prompt can be run at several visual-token budgets inside ONE server session (server flag --trust-request-mm-kwargs).
    mm: dict[str, Any] | None = None
    # Request-body overrides for this entry only: a decoding variant such as {repetition_penalty: 1.05} or
    # {temperature: 0.7, top_p: 0.95}, or {logprobs: true} to store token log-probabilities. Needs a `tag` too
    # (result id "<id>@<tag>"); the override is part of the prompt hash, so changing it re-runs the requests.
    sampling: dict[str, Any] | None = None
    tag: str | None = None

    @model_validator(mode="after")
    def _mm_needs_tag(self) -> PromptEntry:
        if (self.mm or self.sampling) and not self.tag:
            raise ValueError(f"prompt {self.id}: `mm` and `sampling` need a `tag` (the suffix of the result id)")
        return self


class Stage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    name: str = ""
    models: list[str]
    prompts: list[PromptEntry | str] = Field(default_factory=list)
    thinking: list[str] = Field(default_factory=lambda: ["off"])
    seeds: list[int] | None = None
    splits: list[str] | None = None
    optional: bool = False            # whole stage runs only with --with-optional / include_optional

    def prompt_entries(self) -> list[PromptEntry]:
        return [PromptEntry(id=p) if isinstance(p, str) else p for p in self.prompts]


class Plan(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    seeds: list[int] = Field(default_factory=lambda: [0])
    splits: list[str] = Field(default_factory=lambda: ["dev", "test"])
    include_optional: bool = False
    prefetch: bool = True
    keep_weights: bool = False
    min_free_gb: float | None = None
    limit: int | None = None
    # Share of the card this plan may use (0-1). Set it on two plans with small models to run them side by side: every
    # server is capped at this fraction and the run waits for that much free memory instead of an empty card.
    gpu_share: float | None = None
    stages: list[Stage]

    def ordered_models(self, with_optional: bool = True) -> list[tuple[Stage, str]]:
        """(stage, model_key) in run order: OCR stages, then VLM, then LLM (stable inside a role)."""
        out = []
        stages = [s for s in self.stages if with_optional or not s.optional]
        for st in sorted(stages, key=lambda s: ROLE_ORDER.get(s.role, 9)):
            for m in st.models:
                out.append((st, m))
        # a model listed in several stages runs once, at its first position, with all its prompts
        seen: dict[str, int] = {}
        uniq = []
        for st, m in out:
            if m in seen:
                continue
            seen[m] = len(uniq)
            uniq.append((st, m))
        return uniq

    def stages_of(self, model_key: str, with_optional: bool = True) -> list[Stage]:
        return [s for s in self.stages if model_key in s.models and (with_optional or not s.optional)]


def load_plan(path: Path) -> Plan:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Plan.model_validate(data)


@dataclass
class Job:
    """One request to run: key + what is needed to build its body."""
    model_key: str
    prompt_id: str            # e.g. P1, P-OCR+@ocr_ppocr, OCR
    base_prompt: str          # P-OCR+ / OCR / P1 ...
    image_id: str
    image_sha: str
    image_path: str
    split: str
    seed: int
    thinking: str             # off | on
    phase: int = 1            # 1: single stage; 2: needs a stage-1 output
    depends_on: str | None = None   # prompt id of the stage-1 output (same model) or OCR model key
    ocr_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.image_sha, self.seed, self.thinking)

    def key_dict(self) -> dict[str, Any]:
        return {"model_key": self.model_key, "prompt_id": self.prompt_id, "image_sha": self.image_sha,
                "image_id": self.image_id, "seed": self.seed, "thinking": self.thinking, "split": self.split}
