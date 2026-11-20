"""One-off image preprocessing into cache/images/<sha256>.jpg (EXIF rotation, HEIC/WEBP -> JPEG,
RGB, longest side <= max_side). Runs in a process pool of at most 10 workers.
Images are only read and written by code; nothing is displayed."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreResult:
    sha256: str
    out_path: str
    width: int = 0
    height: int = 0
    orig_width: int = 0
    orig_height: int = 0
    error: str = ""


def _register_heif() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:  # pragma: no cover
        pass


def preprocess_one(src: str, dst: str, sha256: str, max_side: int, quality: int) -> PreResult:
    import os

    from PIL import Image, ImageOps

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _register_heif()
    try:
        dst_p = Path(dst)
        if dst_p.is_file() and dst_p.stat().st_size > 0:
            with Image.open(dst_p) as im:
                w, h = im.size
            return PreResult(sha256, dst, w, h, 0, 0)
        with Image.open(src) as im:
            im.load()
            ow, oh = im.size
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.split()[-1])
                    im = bg
                else:
                    im = im.convert("RGB")
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_p.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=quality, optimize=True)
            tmp.replace(dst_p)
            return PreResult(sha256, dst, im.size[0], im.size[1], ow, oh)
    except Exception as e:  # noqa: BLE001 - report every failure per image
        return PreResult(sha256, dst, error=f"{type(e).__name__}: {e}")


def preprocess_all(jobs: list[tuple[str, str, str]], max_side: int = 2048, quality: int = 95,
                   workers: int = 10) -> list[PreResult]:
    """jobs: (src_path, dst_path, sha256)."""
    if not jobs:
        return []
    workers = max(1, min(workers, 10, len(jobs)))
    if workers == 1:
        return [preprocess_one(s, d, h, max_side, quality) for s, d, h in jobs]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(preprocess_one, s, d, h, max_side, quality) for s, d, h in jobs]
        return [f.result() for f in futs]
