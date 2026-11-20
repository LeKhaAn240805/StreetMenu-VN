"""Model output schema (prompts_v1.md section 6.2 + optional price_max_vnd).

The parser coerces sloppy model output (prices as strings, missing lists) into this model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from smv.schema.annotation import Annotation


def _str_or_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    s = str(v).strip()
    return None if s == "" or s.lower() in ("null", "none") else s


class PredOffer(BaseModel):
    model_config = ConfigDict(extra="allow")
    size: str | None = None
    temp: str | None = None
    portion: str | None = None
    raw_price: str | None = None
    price_vnd: int | None = None
    price_max_vnd: int | None = None
    old_price_vnd: int | None = None
    unit: str | None = None
    confidence: float | None = None

    _strs = field_validator("size", "temp", "portion", "raw_price", "unit", mode="before")(_str_or_none)

    @field_validator("price_vnd", "price_max_vnd", "old_price_vnd", mode="before")
    @classmethod
    def _price(cls, v: Any) -> Any:
        from smv.scoring.normalize import parse_price_value

        return parse_price_value(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> Any:
        try:
            return None if v in (None, "") else float(v)
        except (TypeError, ValueError):
            return None


class PredItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    section: str | None = None
    raw_name: str | None = None
    name: str | None = None
    description: str | None = None
    is_topping: bool = False
    is_combo: bool = False
    offers: list[PredOffer] = Field(default_factory=list)

    _strs = field_validator("section", "raw_name", "name", "description", mode="before")(_str_or_none)

    @field_validator("is_topping", "is_combo", mode="before")
    @classmethod
    def _bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "có")
        return bool(v)

    @field_validator("offers", mode="before")
    @classmethod
    def _offers(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return [o for o in v if isinstance(o, (dict, BaseModel))]


class Prediction(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: list[PredItem] = Field(default_factory=list)

    @field_validator("items", mode="before")
    @classmethod
    def _items(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return [i for i in v if isinstance(i, (dict, BaseModel))]


class Transcript(BaseModel):
    """Output of P-T: {"lines": [...]}."""
    lines: list[str] = Field(default_factory=list)


def annotation_to_prediction(ann: Annotation) -> Prediction:
    """Turn a label into a 'perfect' prediction (used for annotator agreement and mock servers)."""
    secs = ann.section_map()
    items = []
    for it in ann.items:
        sec = secs.get(it.section_id) if it.section_id else None
        offers = []
        for o in it.offers:
            pv = o.price_min_vnd if o.price_kind == "range" and o.price_min_vnd is not None else o.price_vnd
            offers.append(PredOffer(size=o.size or None, temp=o.temp or None, portion=o.portion or None,
                                    raw_price=o.raw_price or None, price_vnd=pv,
                                    price_max_vnd=o.price_max_vnd if o.price_kind == "range" else None,
                                    old_price_vnd=o.old_price_vnd, unit=o.unit or None))
        items.append(PredItem(section=(sec.raw_title if sec else None), raw_name=it.raw_name,
                              name=it.norm_name or it.raw_name, description=it.description or None,
                              is_topping=it.is_add_on, is_combo=it.is_combo, offers=offers))
    return Prediction(items=items)


def export_json_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in (("annotation_v3", Annotation), ("prediction", Prediction), ("transcript", Transcript)):
        p = out_dir / f"{name}.schema.json"
        p.write_text(json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p)
    return written
