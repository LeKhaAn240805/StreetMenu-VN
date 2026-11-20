"""Find label files and match them to images (relative path -> file_sha256 -> file_name)."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from smv.data.scan import ImageFile
from smv.schema.annotation import Annotation


@dataclass
class LabelFile:
    path: Path
    annotator: str              # folder name under _labels/ (or annotator field)
    base_dir: Path | None       # directory the label's relative path is relative to
    rel_guess: str              # relative image path decoded from the file name
    ann: Annotation | None
    error: str = ""
    adjudicated: bool = False


@dataclass
class MatchResult:
    by_sha: dict[str, list[LabelFile]] = field(default_factory=lambda: defaultdict(list))
    orphan_labels: list[LabelFile] = field(default_factory=list)
    invalid_labels: list[LabelFile] = field(default_factory=list)
    match_method: dict[str, str] = field(default_factory=dict)  # label path -> method


def _decode_rel(name: str) -> str:
    stem = name[:-5] if name.endswith(".json") else name
    return stem.replace("__", "/")


def load_label_file(path: Path, annotator: str, base_dir: Path | None, adjudicated: bool = False) -> LabelFile:
    rel = _decode_rel(path.name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ann = Annotation.model_validate(data)
        if not ann.annotator:
            ann.annotator = annotator
        return LabelFile(path, ann.annotator or annotator, base_dir, rel, ann, adjudicated=adjudicated)
    except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as e:
        return LabelFile(path, annotator, base_dir, rel, None, error=f"{type(e).__name__}: {str(e)[:300]}",
                         adjudicated=adjudicated)


def _is_label_json(p: Path) -> bool:
    return p.suffix == ".json" and p.name not in ("schema_config.json",) and not p.name.startswith(".")


def find_labels(images_root: Path | None, labels_dir: Path | None, adjudicated_dir: Path | None) -> list[LabelFile]:
    """labels/<annotator>/*.json (copy of _labels) wins; otherwise images/**/_labels/<annotator>/*.json.
    adjudicated/*.json (or adjudicated/<any>/*.json) are always loaded and flagged."""
    out: list[LabelFile] = []
    if labels_dir and labels_dir.is_dir():
        for ann_dir in sorted(p for p in labels_dir.iterdir() if p.is_dir()):
            for f in sorted(ann_dir.glob("*.json")):
                if _is_label_json(f):
                    out.append(load_label_file(f, ann_dir.name, images_root))
    elif images_root and images_root.is_dir():
        for lab_root in sorted(images_root.rglob("_labels")):
            if not lab_root.is_dir():
                continue
            for ann_dir in sorted(p for p in lab_root.iterdir() if p.is_dir()):
                for f in sorted(ann_dir.glob("*.json")):
                    if _is_label_json(f):
                        out.append(load_label_file(f, ann_dir.name, lab_root.parent))
    if adjudicated_dir and adjudicated_dir.is_dir():
        for f in sorted(adjudicated_dir.rglob("*.json")):
            if _is_label_json(f):
                out.append(load_label_file(f, "adjudicated", images_root, adjudicated=True))
    return out


def match_labels(images: list[ImageFile], labels: list[LabelFile], images_root: Path) -> MatchResult:
    res = MatchResult()
    by_rel = {im.rel_path: im for im in images}
    by_rel_lower = {im.rel_path.lower(): im for im in images}
    by_sha = {im.sha256: im for im in images}
    by_name: dict[str, list[ImageFile]] = defaultdict(list)
    for im in images:
        by_name[im.file_name].append(im)

    for lf in labels:
        if lf.ann is None:
            res.invalid_labels.append(lf)
            continue
        target = None
        method = ""
        # 1. relative path (label base dir may be a sub-folder of the image root)
        rel = lf.rel_guess
        if lf.base_dir is not None:
            try:
                prefix = lf.base_dir.resolve().relative_to(images_root.resolve()).as_posix()
                if prefix not in ("", "."):
                    rel = f"{prefix}/{rel}"
            except ValueError:
                pass
        if rel in by_rel:
            target, method = by_rel[rel], "path"
        elif rel.lower() in by_rel_lower:
            target, method = by_rel_lower[rel.lower()], "path_case"
        # 2. sha256
        if target is None and lf.ann.file_sha256 and lf.ann.file_sha256 in by_sha:
            target, method = by_sha[lf.ann.file_sha256], "sha256"
        # 3. file name (only when unique)
        if target is None and lf.ann.file_name:
            cands = by_name.get(lf.ann.file_name, [])
            if len(cands) == 1:
                target, method = cands[0], "file_name"
        if target is None:
            res.orphan_labels.append(lf)
            continue
        res.by_sha[target.sha256].append(lf)
        res.match_method[str(lf.path)] = method
    return res
