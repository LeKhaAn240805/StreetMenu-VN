"""`smv ingest`: scan images + labels, choose gold, validate, split, preprocess, write data_index."""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from smv.data.gold import select_gold
from smv.data.labels import find_labels, match_labels
from smv.data.preprocess import preprocess_all
from smv.data.scan import scan_images
from smv.data.splits import assign_splits, build_groups, read_splits_csv, write_splits_csv
from smv.data.validate import validate_annotation, validate_dataset
from smv.schema.annotation import Annotation
from smv.schema.config import load_schema_config
from smv.settings import Settings

log = logging.getLogger("smv.ingest")

INDEX_COLUMNS = ["image_id", "sha256", "rel_path", "src_path", "cache_path", "width", "height", "split",
                 "group_id", "ccl", "tags", "venue_tier", "venue_type", "medium", "lighting", "angle", "source",
                 "has_prices", "n_items", "n_offers", "dense", "has_gold", "gold_reason", "gold_annotator",
                 "n_done_sets", "base_image_id", "perturbation"]


def derive_image_id(rel_path: str, prefix: str) -> str:
    stem = rel_path.rsplit(".", 1)[0] if "." in Path(rel_path).name else rel_path
    return prefix + stem.replace("/", "__")


def run_ingest(st: Settings, preprocess: bool = True) -> dict[str, Any]:
    if st.images_dir is None or not st.images_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {st.images_dir} (set SMV_DATA_DIR / SMV_IMAGES_DIR)")
    icfg = st.cfg.get("ingest", {})
    scfg = load_schema_config(st.schema_config_path)
    images = scan_images(st.images_dir, icfg.get("image_exts", []), icfg.get("skip_dirs", []))
    labels = find_labels(st.images_dir, st.labels_dir, st.adjudicated_dir)
    match = match_labels(images, labels, st.images_dir)
    report: dict[str, Any] = {"images_dir": str(st.images_dir), "n_images": len(images),
                              "n_label_files": len(labels), "n_invalid_labels": len(match.invalid_labels),
                              "orphan_labels": [str(lf.path) for lf in match.orphan_labels],
                              "invalid_labels": {str(lf.path): lf.error for lf in match.invalid_labels},
                              "match_methods": dict(Counter(match.match_method.values()))}

    sha_counts = Counter(im.sha256 for im in images)
    report["duplicate_files"] = [im.rel_path for im in images if sha_counts[im.sha256] > 1]

    rows: list[dict[str, Any]] = []
    golds: dict[str, Annotation] = {}
    agreement_sets: dict[str, list[Annotation]] = {}
    gold_reasons = Counter()
    n_draft = n_review = 0
    used_ids: set[str] = set()
    warnings: list[str] = []
    prefix = icfg.get("image_id_prefix", "smv_")
    gold_annotators = list(icfg.get("gold_annotators", []) or [])
    env_ga = os.environ.get("SMV_GOLD_ANNOTATORS", "").strip()
    if env_ga:
        gold_annotators = [x.strip() for x in env_ga.split(",") if x.strip()]

    for im in images:
        choice = select_gold(match.by_sha.get(im.sha256, []), gold_annotators)
        n_draft += choice.n_draft
        n_review += choice.n_needs_review
        gold_reasons[choice.reason] += 1
        warnings += [f"{im.rel_path}: {w}" for w in choice.warnings]
        ann = choice.gold.ann if choice.gold else None
        iid = (ann.image_id if ann and ann.image_id else derive_image_id(im.rel_path, prefix))
        if iid in used_ids:
            iid = derive_image_id(im.rel_path.replace(".", "_"), prefix)
            k = 2
            while iid in used_ids:
                iid = f"{iid}_{k}"
                k += 1
        used_ids.add(iid)
        if ann is not None:
            ann = ann.model_copy(deep=True)
            ann.image_id = iid
            ann.file_name = ann.file_name or im.file_name
            golds[iid] = ann
            agreement_sets[iid] = [lf.ann for lf in choice.others if lf.ann is not None]
        rows.append({
            "image_id": iid, "sha256": im.sha256, "rel_path": im.rel_path, "src_path": str(im.path),
            "cache_path": str(st.image_cache / f"{im.sha256}.jpg"), "width": 0, "height": 0,
            "split": ann.split if ann else "unassigned", "group_id": ann.group_id if ann else "",
            "near_duplicate_of": ann.near_duplicate_of if ann else "", "file_name": im.file_name,
            "ccl": ann.ccl if ann else None, "tags": ";".join(ann.tags) if ann else "",
            "venue_tier": ann.venue_tier if ann else "", "venue_type": ann.venue_type if ann else "",
            "medium": ann.medium if ann else "", "lighting": ann.lighting if ann else "",
            "angle": ann.angle if ann else "", "source": ann.source if ann else "",
            "has_prices": ann.has_prices if ann else None, "n_items": ann.n_items() if ann else 0,
            "n_offers": ann.n_offers() if ann else 0,
            "dense": bool(ann and ann.n_offers() > int(st.cfg.get("dense_threshold", 60))),
            "has_gold": ann is not None, "gold_reason": choice.reason,
            "gold_annotator": choice.gold.annotator if choice.gold else "",
            "n_done_sets": 1 + len(choice.others) if choice.gold else len(choice.others),
            "base_image_id": iid, "perturbation": "",
        })

    # ---- splits: splits.csv > label split > automatic (group-aware) ---------------------------
    manual = read_splits_csv(st.splits_csv) if st.splits_csv else {}
    for r in rows:
        m = manual.get(r["image_id"])
        if m:
            r["split"] = m.get("split") or r["split"]
            r["group_id"] = m.get("group_id") or r["group_id"]
    groups = build_groups(rows)
    for r in rows:
        if not r["group_id"]:
            r["group_id"] = groups[r["image_id"]]
    scfg_split = st.cfg.get("splits", {})
    split_of = assign_splits(rows, scfg_split.get("ratio", {"dev": 0.2, "test": 0.8}), int(scfg_split.get("seed", 2027)))
    for r in rows:
        r["split"] = split_of[r["image_id"]]
        if r["image_id"] in golds:
            golds[r["image_id"]].split = r["split"]
    st.data_out.mkdir(parents=True, exist_ok=True)
    write_splits_csv(st.data_out / "splits_auto.csv", rows)
    # Write $SMV_DATA_DIR/splits.csv only once every image has a gold label: a split made before group_id /
    # near_duplicate_of are known would otherwise be frozen and override the labels later.
    all_gold = bool(rows) and all(r["has_gold"] for r in rows)
    if st.splits_csv and scfg_split.get("write_back", True) and not st.splits_csv.exists() and all_gold:
        try:
            write_splits_csv(st.splits_csv, rows)
            report["splits_csv_written"] = str(st.splits_csv)
        except OSError as e:
            warnings.append(f"could not write {st.splits_csv}: {e}")

    # ---- validation ------------------------------------------------------------------------
    issues = []
    for iid, ann in golds.items():
        issues += [(iid, i) for i in validate_annotation(ann, scfg)]
    issues += [("", i) for i in validate_dataset(golds)]
    # group leakage after automatic split
    gs: dict[str, set] = {}
    for r in rows:
        if r["split"] in ("dev", "test"):
            gs.setdefault(r["group_id"], set()).add(r["split"])
    leaks = [g for g, s in gs.items() if len(s) > 1]

    # ---- preprocessing ---------------------------------------------------------------------
    pcfg = st.cfg.get("preprocess", {})
    if preprocess:
        jobs = []
        seen = set()
        for r in rows:
            if r["sha256"] in seen:
                continue
            seen.add(r["sha256"])
            jobs.append((r["src_path"], r["cache_path"], r["sha256"]))
        st.image_cache.mkdir(parents=True, exist_ok=True)
        res = {p.sha256: p for p in preprocess_all(jobs, int(pcfg.get("max_side", 2048)),
                                                   int(pcfg.get("jpeg_quality", 95)), int(pcfg.get("workers", 10)))}
        errors = {}
        for r in rows:
            p = res.get(r["sha256"])
            if p:
                r["width"], r["height"] = p.width, p.height
                if p.error:
                    errors[r["rel_path"]] = p.error
        report["preprocess_errors"] = errors

    # ---- write -------------------------------------------------------------------------------
    st.data_out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df[INDEX_COLUMNS + [c for c in df.columns if c not in INDEX_COLUMNS]]
    df.to_parquet(st.data_index_path, index=False)
    df.to_csv(st.data_out / "data_index.csv", index=False)
    gold_dir = st.data_out / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    for old in gold_dir.glob("*.json"):
        old.unlink()
    for iid, ann in golds.items():
        (gold_dir / f"{iid}.json").write_text(ann.model_dump_json(indent=1), encoding="utf-8")
    agr_dir = st.data_out / "agreement_sets"
    if agr_dir.exists():
        for old in agr_dir.rglob("*.json"):
            old.unlink()
    for iid, anns in agreement_sets.items():
        for a in anns:
            d = agr_dir / (a.annotator or "unknown")
            d.mkdir(parents=True, exist_ok=True)
            a2 = a.model_copy(deep=True)
            a2.image_id = iid
            (d / f"{iid}.json").write_text(a2.model_dump_json(indent=1), encoding="utf-8")
    pd.DataFrame([{"image_id": iid, "level": i.level, "code": i.code, "where": i.where, "msg": i.msg}
                  for iid, i in issues]).to_csv(st.data_out / "validation.csv", index=False)

    report.update({
        "n_gold": len(golds), "gold_reasons": dict(gold_reasons), "n_draft_labels": n_draft,
        "n_needs_review_labels": n_review, "n_validation_errors": sum(1 for _, i in issues if i.level == "error"),
        "n_validation_warnings": sum(1 for _, i in issues if i.level == "warning"),
        "orphan_images": [r["rel_path"] for r in rows if not r["has_gold"]],
        "splits": dict(Counter(r["split"] for r in rows)), "group_split_leaks": leaks, "warnings": warnings,
        "n_agreement_images": sum(1 for v in agreement_sets.values() if v),
    })
    (st.data_out / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def load_index(st: Settings) -> pd.DataFrame:
    if not st.data_index_path.is_file():
        raise FileNotFoundError(f"{st.data_index_path} not found: run `smv ingest` first")
    df = pd.read_parquet(st.data_index_path)
    pert = st.data_out / "perturbed_index.parquet"
    if pert.is_file():
        df = pd.concat([df, pd.read_parquet(pert)], ignore_index=True)
    return df


def load_golds(st: Settings) -> dict[str, Annotation]:
    gold_dir = st.data_out / "gold"
    out = {}
    for p in sorted(gold_dir.glob("*.json")) if gold_dir.is_dir() else []:
        out[p.stem] = Annotation.model_validate_json(p.read_text(encoding="utf-8"))
    return out


def load_agreement_sets(st: Settings) -> dict[str, dict[str, Annotation]]:
    """{annotator: {image_id: Annotation}} of non-gold `done` label sets."""
    root = st.data_out / "agreement_sets"
    out: dict[str, dict[str, Annotation]] = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out[d.name] = {p.stem: Annotation.model_validate_json(p.read_text(encoding="utf-8")) for p in d.glob("*.json")}
    return out


def select_images(df: pd.DataFrame, split: str | None = None, limit: int | None = None,
                  only_gold: bool = False) -> pd.DataFrame:
    sel = df
    if split and split != "all":
        sel = sel[sel["split"] == split]
    if only_gold:
        sel = sel[sel["has_gold"]]
    sel = sel.sort_values("image_id")
    if limit:
        sel = sel.head(limit)
    return sel
