"""Read ref/schema_config.json at run time (enums, tags, price warning range).

Nothing in the code base lists enum values or tags by hand; everything comes from this file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SchemaConfig:
    raw: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.raw.get("schema_version", ""))

    @property
    def price_min_warn(self) -> int:
        return int(self.raw.get("price_min_warn", 1000))

    @property
    def price_max_warn(self) -> int:
        return int(self.raw.get("price_max_warn", 5_000_000))

    def _fields(self, level: str) -> list[dict[str, Any]]:
        return list(self.raw.get(f"{level}_fields", []))

    def field(self, level: str, key: str) -> dict[str, Any] | None:
        for f in self._fields(level):
            if f.get("key") == key:
                return f
        return None

    def options(self, level: str, key: str) -> list[str]:
        f = self.field(level, key)
        return [str(o) for o in (f or {}).get("options", [])]

    def enum_fields(self, level: str) -> dict[str, tuple[str, list[str]]]:
        """{key: (type, options)} for select/multi fields of a level (image, section, item, offer)."""
        out = {}
        for f in self._fields(level):
            if f.get("type") in ("select", "multi") and f.get("options"):
                out[f["key"]] = (f["type"], [str(o) for o in f["options"]])
        return out

    def required_fields(self, level: str) -> list[str]:
        return [f["key"] for f in self._fields(level) if f.get("req")]

    @property
    def tag_groups(self) -> dict[str, list[str]]:
        """{prefix: [codes]} in file order, e.g. {"P_": ["P_IMPLICIT_K", ...]}."""
        return {g["prefix"]: [t["code"] for t in g.get("tags", [])] for g in self.raw.get("tag_groups", [])}

    @property
    def tag_group_labels(self) -> dict[str, str]:
        return {g["prefix"]: g.get("label", g["prefix"]) for g in self.raw.get("tag_groups", [])}

    @property
    def tags(self) -> list[str]:
        return [c for codes in self.tag_groups.values() for c in codes]

    def tag_group_of(self, tag: str) -> str | None:
        for prefix, codes in self.tag_groups.items():
            if tag in codes:
                return prefix
        return None

    @property
    def ccl_levels(self) -> list[str]:
        return [o for o in self.options("image", "ccl") if o != ""]


@lru_cache(maxsize=8)
def _load(path: str) -> SchemaConfig:
    with open(path, encoding="utf-8") as f:
        return SchemaConfig(json.load(f))


def load_schema_config(path: Path | str) -> SchemaConfig:
    return _load(str(Path(path).resolve()))
