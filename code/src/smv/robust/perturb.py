"""Deterministic perturbations of cached CCL-1/2 images (only via file paths; nothing is displayed).
Writes cache/perturbed/<sha>__<kind>.jpg and outputs/data/perturbed_index.parquet (split 'robust')."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from smv.data.index import load_index
from smv.settings import Settings

KINDS = ("rot+15", "rot-15", "perspective", "blur", "jpeg30", "half", "glare")


def _perturb(src: Path, dst: Path, kind: str, seed: int) -> None:
    from PIL import Image, ImageFilter

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if kind.startswith("rot"):
            ang = 15 if kind == "rot+15" else -15
            out = im.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        elif kind == "perspective":
            d = 0.12
            quad = (w * d, h * 0.02, 0, h, w, h, w * (1 - d), h * 0.02)
            out = im.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        elif kind == "blur":
            out = im.filter(ImageFilter.GaussianBlur(radius=max(1.5, w / 800)))
        elif kind == "half":
            out = im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
        elif kind == "glare":
            rng = np.random.default_rng(seed)
            cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
            r = 0.35 * min(w, h)
            yy, xx = np.mgrid[0:h, 0:w]
            mask = np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1) ** 2 * 0.85
            arr = np.asarray(im, dtype=np.float32)
            arr = arr * (1 - mask[..., None]) + 255 * mask[..., None]
            out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        else:
            out = im
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, "JPEG", quality=30 if kind == "jpeg30" else 92)


def make_perturbed(st: Settings, ccl_levels: list[int], kinds: tuple[str, ...] = KINDS) -> int:
    idx = load_index(st)
    idx = idx[(idx["perturbation"].fillna("") == "") & idx["has_gold"] & idx["ccl"].isin(ccl_levels)]
    rows = []
    for r in idx.itertuples(index=False):
        for kind in kinds:
            seed = int(hashlib.sha256(f"{r.sha256}{kind}".encode()).hexdigest()[:8], 16)
            dst = st.perturbed_cache / f"{r.sha256}__{kind}.jpg"
            if not dst.is_file():
                _perturb(Path(r.cache_path), dst, kind, seed)
            sha = hashlib.sha256(dst.read_bytes()).hexdigest()
            row = r._asdict()
            row.update({"image_id": f"{r.image_id}@{kind}", "sha256": sha, "cache_path": str(dst), "split": "robust",
                        "base_image_id": r.image_id, "perturbation": kind, "src_path": str(r.cache_path)})
            rows.append(row)
    pd.DataFrame(rows).to_parquet(st.data_out / "perturbed_index.parquet", index=False)
    return len(rows)
