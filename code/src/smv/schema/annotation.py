"""Pydantic v2 models for the v3 label files written by the web tool.

Types are coerced leniently (the web tool writes "" for empty selects and numbers as strings);
enum values are NOT checked here but in smv.data.validate against schema_config.json.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _empty_to_none(v: Any) -> Any:
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _to_int(v: Any) -> Any:
    v = _empty_to_none(v)
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return v
    if isinstance(v, float):
        return int(round(v))
    if isinstance(v, str):
        s = v.strip().replace(".", "").replace(",", "").replace(" ", "")
        try:
            return int(float(s))
        except ValueError:
            return v
    return v


def _to_float(v: Any) -> Any:
    v = _empty_to_none(v)
    if v is None or isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    if isinstance(v, str):
        try:
            return float(v.replace(",", "."))
        except ValueError:
            return v
    return v


def _to_list(v: Any) -> Any:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return v


def _to_str(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return v


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ComboComponent(_Base):
    name: str = ""
    qty: str | None = None
    ref_price_vnd: int | None = None

    _qty = field_validator("qty", mode="before")(lambda v: None if v in (None, "") else str(v))
    _price = field_validator("ref_price_vnd", mode="before")(_to_int)


class Offer(_Base):
    size: str = ""
    size_rank: int | None = None
    temp: str = ""
    portion: str = ""
    quantity: float | None = None
    qty_unit: str = ""
    raw_price: str = ""
    price_vnd: int | None = None
    old_price_vnd: int | None = None
    price_kind: str = "fixed"
    price_min_vnd: int | None = None
    price_max_vnd: int | None = None
    unit_qty: float | None = None
    unit: str = ""
    price_source: str = "item_line"
    evidence: str = ""
    ambiguous: bool = False  # not in the web tool; an item-level flag is propagated in scoring
    notes: str = ""

    _ints = field_validator("size_rank", "price_vnd", "old_price_vnd", "price_min_vnd", "price_max_vnd",
                            mode="before")(_to_int)
    _floats = field_validator("quantity", "unit_qty", mode="before")(_to_float)
    _strs = field_validator("size", "temp", "portion", "qty_unit", "raw_price", "unit", "evidence", "notes",
                            mode="before")(_to_str)

    @field_validator("price_kind", "price_source", mode="before")
    @classmethod
    def _default_kind(cls, v: Any, info) -> Any:
        if v in (None, ""):
            return "fixed" if info.field_name == "price_kind" else "item_line"
        return v


class Item(_Base):
    item_id: int | str | None = None
    section_id: str | None = None
    item_code: str = ""
    raw_name: str = ""
    raw_name_2: str = ""
    norm_name: str = ""
    alt_norm_names: list[str] = Field(default_factory=list)
    description: str = ""
    category: str = ""
    is_add_on: bool = False
    is_combo: bool = False
    combo_components: list[ComboComponent] = Field(default_factory=list)
    serves: str = ""
    option_groups: str = ""
    attributes: list[str] = Field(default_factory=list)
    availability: str = ""
    item_status: str = "active"
    legibility: str = ""
    ambiguous: bool = False
    ambig_reason: str = ""
    bbox: Any = None
    notes: str = ""
    offers: list[Offer] = Field(default_factory=list)

    _lists = field_validator("alt_norm_names", "attributes", mode="before")(_to_list)
    _sec = field_validator("section_id", mode="before")(_empty_to_none)
    _strs = field_validator("item_code", "raw_name", "raw_name_2", "norm_name", "description", "category",
                            "serves", "option_groups", "availability", "legibility", "ambig_reason", "notes",
                            mode="before")(_to_str)

    @field_validator("item_status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return "active" if v in (None, "") else v

    @field_validator("combo_components", mode="before")
    @classmethod
    def _components(cls, v: Any) -> Any:
        if v in (None, ""):
            return []
        out = []
        for c in v:
            if isinstance(c, str):  # "name ; qty ; price"
                parts = [p.strip() for p in c.split(";")]
                out.append({"name": parts[0], "qty": parts[1] if len(parts) > 1 else None,
                            "ref_price_vnd": parts[2] if len(parts) > 2 else None})
            else:
                out.append(c)
        return out


class Section(_Base):
    section_id: str
    raw_title: str = ""
    norm_title: str = ""
    shared_price_raw: str = ""
    notes: str = ""

    _strs = field_validator("raw_title", "norm_title", "shared_price_raw", "notes", mode="before")(_to_str)


class IgnoreRegion(_Base):
    desc: str = ""
    text: str = ""  # optional: literal text in the region (used to ignore predictions)
    bbox: Any = None

    _strs = field_validator("desc", "text", mode="before")(_to_str)


class Annotation(_Base):
    schema_version: str = "3.0"
    image_id: str = ""
    file_name: str = ""
    file_sha256: str = ""
    group_id: str = ""
    near_duplicate_of: str = ""
    split: str = "unassigned"
    ccl: int | None = None
    tags: list[str] = Field(default_factory=list)
    venue_tier: str = ""
    venue_type: str = ""
    medium: str = ""
    lighting: str = ""
    angle: str = ""
    source: str = ""
    city: str = ""
    languages: list[str] = Field(default_factory=list)
    price_unit_default: str = ""
    tax_note: str = ""
    has_prices: bool = True
    pii_present: bool = False
    pii_types: list[str] = Field(default_factory=list)
    redaction_checked: bool = False
    consent_id: str = ""
    sections: list[Section] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    ignore_regions: list[IgnoreRegion] = Field(default_factory=list)
    annotator: str = ""
    status: str = "draft"
    version: int | None = None
    notes: str = ""

    _ccl = field_validator("ccl", "version", mode="before")(_to_int)
    _lists = field_validator("tags", "languages", "pii_types", mode="before")(_to_list)
    _strs = field_validator("schema_version", "image_id", "file_name", "file_sha256", "group_id",
                            "near_duplicate_of", "venue_tier", "venue_type", "medium", "lighting", "angle",
                            "source", "city", "price_unit_default", "tax_note", "consent_id", "annotator",
                            "notes", mode="before")(_to_str)

    @field_validator("split", "status", mode="before")
    @classmethod
    def _default_split(cls, v: Any, info) -> Any:
        if v in (None, ""):
            return "unassigned" if info.field_name == "split" else "draft"
        return v

    @field_validator("ignore_regions", mode="before")
    @classmethod
    def _regions(cls, v: Any) -> Any:
        if v in (None, ""):
            return []
        return [{"desc": r} if isinstance(r, str) else r for r in v]

    # ---- helpers ---------------------------------------------------------
    def n_items(self) -> int:
        return len(self.items)

    def n_offers(self) -> int:
        return sum(len(it.offers) for it in self.items)

    def section_map(self) -> dict[str, Section]:
        return {s.section_id: s for s in self.sections}
