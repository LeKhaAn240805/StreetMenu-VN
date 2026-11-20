"""Label checks against schema_config.json enums and the consistency rules of the spec."""
from __future__ import annotations

from dataclasses import dataclass

from smv.schema.annotation import Annotation
from smv.schema.config import SchemaConfig


@dataclass
class Issue:
    level: str        # error | warning
    code: str
    where: str
    msg: str


def _check_enum(issues: list[Issue], cfg: SchemaConfig, level: str, obj, where: str) -> None:
    for key, (ftype, options) in cfg.enum_fields(level).items():
        if not hasattr(obj, key) and key not in (obj.model_extra or {}):
            continue
        val = getattr(obj, key, None)
        if val is None:
            continue
        if ftype == "multi" or isinstance(val, list):
            bad = [v for v in val if str(v) not in options]
            if bad:
                issues.append(Issue("error", "enum", where, f"{key}: {bad} not in {options}"))
        else:
            sval = "" if val is None else str(val)
            if sval not in options and not (sval == "" and "" in options):
                if sval == "" and level in ("item", "offer"):
                    continue
                issues.append(Issue("error", "enum", where, f"{key}={sval!r} not in {options}"))


def validate_annotation(ann: Annotation, cfg: SchemaConfig) -> list[Issue]:
    issues: list[Issue] = []
    img = ann.file_name or ann.image_id
    _check_enum(issues, cfg, "image", ann, img)
    allowed_tags = set(cfg.tags)
    bad_tags = [t for t in ann.tags if t not in allowed_tags]
    if bad_tags:
        issues.append(Issue("error", "tag", img, f"unknown tags {bad_tags}"))
    if ann.ccl is not None and str(ann.ccl) not in cfg.ccl_levels:
        issues.append(Issue("error", "ccl", img, f"ccl={ann.ccl} not in {cfg.ccl_levels}"))
    if ann.split != "ood" and ann.ccl is None and ann.status == "done":
        issues.append(Issue("warning", "ccl_missing", img, "ccl is empty"))
    if ann.schema_version and not str(ann.schema_version).startswith(cfg.version.split(".")[0]):
        issues.append(Issue("warning", "schema_version", img, f"schema_version {ann.schema_version} != {cfg.version}"))

    section_ids = [s.section_id for s in ann.sections]
    dup = {s for s in section_ids if section_ids.count(s) > 1}
    if dup:
        issues.append(Issue("error", "section_dup", img, f"duplicate section_id {sorted(dup)}"))
    for s in ann.sections:
        _check_enum(issues, cfg, "section", s, f"{img}/{s.section_id}")

    lo, hi = cfg.price_min_warn, cfg.price_max_warn
    n_priced = 0
    item_ids = []
    for it in ann.items:
        w = f"{img}/item{it.item_id}"
        item_ids.append(it.item_id)
        _check_enum(issues, cfg, "item", it, w)
        if it.section_id is not None and it.section_id not in section_ids:
            issues.append(Issue("error", "section_ref", w, f"section_id {it.section_id!r} not in sections"))
        for req in cfg.required_fields("item"):
            if req in ("category",):
                continue
            if not getattr(it, req, None):
                issues.append(Issue("warning", "required", w, f"{req} is empty"))
        for j, o in enumerate(it.offers):
            wo = f"{w}/offer{j + 1}"
            _check_enum(issues, cfg, "offer", o, wo)
            for name in ("price_vnd", "old_price_vnd", "price_min_vnd", "price_max_vnd"):
                v = getattr(o, name)
                if v is None:
                    continue
                if not isinstance(v, int):
                    issues.append(Issue("error", "price_type", wo, f"{name}={v!r} is not an integer"))
                    continue
                if o.price_kind == "delta" and name == "price_vnd":
                    if not (0 < v <= hi):
                        issues.append(Issue("warning", "price_range", wo, f"{name}={v} outside (0, {hi}]"))
                elif not (lo <= v <= hi):
                    issues.append(Issue("warning", "price_range", wo, f"{name}={v} outside [{lo}, {hi}]"))
            if o.price_vnd is not None:
                n_priced += 1
                if not o.raw_price.strip():
                    issues.append(Issue("error", "raw_price_missing", wo, "price_vnd set but raw_price empty"))
            if o.price_kind == "range":
                if o.price_min_vnd is None or o.price_max_vnd is None:
                    issues.append(Issue("error", "range_bounds", wo, "range without price_min_vnd/price_max_vnd"))
                elif o.price_min_vnd > o.price_max_vnd:
                    issues.append(Issue("error", "range_order", wo, "price_min_vnd > price_max_vnd"))
            elif o.price_min_vnd is not None or o.price_max_vnd is not None:
                issues.append(Issue("warning", "range_fields", wo, f"min/max set but price_kind={o.price_kind}"))
            if o.size_rank is not None and o.size_rank < 1:
                issues.append(Issue("warning", "size_rank", wo, "size_rank < 1"))
    dup_items = {i for i in item_ids if i is not None and item_ids.count(i) > 1}
    if dup_items:
        issues.append(Issue("error", "item_dup", img, f"duplicate item_id {sorted(dup_items, key=str)}"))
    if not ann.has_prices and n_priced:
        issues.append(Issue("error", "has_prices", img, f"has_prices=false but {n_priced} offers have price_vnd"))
    return issues


def validate_dataset(golds: dict[str, Annotation]) -> list[Issue]:
    """Cross-image rules: a group_id must not span two splits; near_duplicate_of must exist."""
    issues: list[Issue] = []
    group_splits: dict[str, set[str]] = {}
    names = {a.file_name for a in golds.values()}
    for a in golds.values():
        if a.group_id and a.split in ("dev", "test"):
            group_splits.setdefault(a.group_id, set()).add(a.split)
        if a.near_duplicate_of and a.near_duplicate_of not in names:
            issues.append(Issue("warning", "near_dup_ref", a.file_name,
                                f"near_duplicate_of {a.near_duplicate_of!r} is not a known file_name"))
    for g, splits in group_splits.items():
        if len(splits) > 1:
            issues.append(Issue("error", "group_split_leak", g, f"group_id {g} appears in splits {sorted(splits)}"))
    return issues
