"""Prompt registry: load prompts/prompts.yaml, render templates, hash and freeze."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from smv.schema.annotation import Annotation

IMAGE_MARK = "[image]"


@dataclass
class PromptSpec:
    id: str
    file: str
    system: str = "en"
    kind: str = "image"           # image | text
    output: str = "json"          # json | lines | free
    status: str = "final"
    inputs: list[str] = field(default_factory=list)
    lines_from: str | None = None  # P-T (same model) | ocr
    stage1: str | None = None


@dataclass
class Rendered:
    system: str | None
    text: str
    image: bool


class PromptRegistry:
    def __init__(self, root: Path):
        self.root = root
        raw = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
        self.specs = {pid: PromptSpec(id=pid, **v) for pid, v in raw["prompts"].items()}
        self.schema_text = (root / "schema.txt").read_text(encoding="utf-8").rstrip("\n")
        self._system = {}
        for lang in ("en", "vi"):
            p = root / f"system_{lang}.txt"
            if p.is_file():
                self._system[lang] = p.read_text(encoding="utf-8").strip()

    def get(self, pid: str) -> PromptSpec:
        base = pid.split("@", 1)[0]
        if base not in self.specs:
            raise KeyError(f"unknown prompt {pid!r}; known: {sorted(self.specs)}")
        return self.specs[base]

    def template(self, pid: str) -> str:
        return (self.root / self.get(pid).file).read_text(encoding="utf-8").rstrip("\n")

    def system_text(self, pid: str) -> str | None:
        lang = self.get(pid).system
        return self._system.get(lang) if lang and lang != "none" else None

    def sha(self, pid: str) -> str:
        spec = self.get(pid)
        h = hashlib.sha256()
        h.update((self.system_text(pid) or "").encode())
        h.update(b"\n---\n")
        h.update(self.template(pid).encode())
        if "{SCHEMA}" in self.template(pid):
            h.update(b"\n---\n")
            h.update(self.schema_text.encode())
        h.update(f"\n---\n{spec.kind}|{spec.output}".encode())
        return h.hexdigest()

    def render(self, pid: str, **values: str) -> Rendered:
        spec = self.get(pid)
        text = self.template(pid)
        image = spec.kind == "image"
        if text.startswith(IMAGE_MARK):
            text = text[len(IMAGE_MARK):].lstrip("\n")
        text = text.replace("{SCHEMA}", self.schema_text)
        for key in spec.inputs:
            if key not in values:
                raise ValueError(f"prompt {pid} needs {key}")
            text = text.replace("{" + key + "}", values[key])
        return Rendered(self.system_text(pid), text, image)

    # ---- freezing ------------------------------------------------------------------------------
    @property
    def frozen_path(self) -> Path:
        return self.root / "FROZEN.json"

    def frozen(self) -> dict[str, str]:
        if not self.frozen_path.is_file():
            return {}
        data = json.loads(self.frozen_path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.get("prompts", {}).items()}

    def freeze(self, pids: list[str], include_drafts: bool = False) -> dict[str, str]:
        cur = self.frozen()
        for pid in pids:
            spec = self.get(pid)
            if spec.status == "draft" and not include_drafts:
                raise ValueError(f"{pid} is a draft; check it and set status: final first")
            if pid in cur and cur[pid] != self.sha(pid):
                raise ValueError(f"{pid} was frozen with a different hash; frozen prompts must not change")
            cur[pid] = self.sha(pid)
        self.frozen_path.write_text(json.dumps({"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
                                                "prompts": cur}, indent=2), encoding="utf-8")
        return cur

    def is_frozen(self, pid: str) -> bool:
        base = pid.split("@", 1)[0]
        fz = self.frozen()
        if base not in fz or fz[base] != self.sha(base):
            return False
        spec = self.get(base)
        # a two-stage prompt is frozen only if its stage-1 prompt is frozen too
        if spec.stage1 and not self.is_frozen(spec.stage1):
            return False
        if spec.lines_from == "P-T" and not self.is_frozen("P-T"):
            return False
        return True

    def check(self) -> list[dict]:
        fz = self.frozen()
        rows = []
        for pid, spec in self.specs.items():
            sha = self.sha(pid)
            rows.append({"prompt": pid, "status": spec.status, "sha256": sha[:16],
                         "frozen": pid in fz, "changed_since_freeze": pid in fz and fz[pid] != sha})
        return rows


# ---------------------------------------------------------------------------------- oracle lines
_HEADER_NOTE = re.compile(r"\s*\((?:[^)]*(?:tiêu đề|tieu de|header|cột|cot|ngoặc|ngoac|gộp|gop)[^)]*)\)\s*$",
                          re.IGNORECASE)
_LINE_SOURCES = {"item_line", "badge", "handwritten_override"}


def oracle_lines(ann: Annotation) -> list[str]:
    """P-Oracle input from the gold label: one line per item 'raw_name raw_price', section titles as
    their own line (inherited prices stay on the title line, as on the menu)."""
    secs = ann.section_map()
    lines: list[str] = []
    cur_sec = object()
    for it in ann.items:
        if it.section_id != cur_sec:
            cur_sec = it.section_id
            sec = secs.get(it.section_id) if it.section_id else None
            if sec and sec.raw_title:
                lines.append(sec.raw_title)
        prices: list[str] = []
        for o in it.offers:
            if o.price_source not in _LINE_SOURCES:
                continue
            rp = _HEADER_NOTE.sub("", o.raw_price or "").strip()
            if rp and rp not in prices:
                prices.append(rp)
        line = " ".join([it.raw_name.strip()] + prices).strip()
        if line:
            lines.append(line)
    return lines
