"""Scan the image tree. Images are only touched as files (hash, path); never displayed."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageFile:
    path: Path            # absolute
    rel_path: str         # relative to the image root, '/' separators
    sha256: str
    size_bytes: int

    @property
    def file_name(self) -> str:
        return Path(self.rel_path).name

    @property
    def label_stem(self) -> str:
        """Name the web tool uses for the label file (without .json)."""
        return self.rel_path.replace("/", "__")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_image_paths(root: Path, exts: Iterable[str], skip_dirs: Iterable[str]) -> list[Path]:
    exts = {e.lower() for e in exts}
    skip = set(skip_dirs)
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in skip or part.startswith(".") for part in rel_parts[:-1]):
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return out


def scan_images(root: Path, exts: Iterable[str], skip_dirs: Iterable[str], workers: int = 10) -> list[ImageFile]:
    from concurrent.futures import ThreadPoolExecutor

    paths = iter_image_paths(root, exts, skip_dirs)

    def one(p: Path) -> ImageFile:
        return ImageFile(path=p.resolve(), rel_path=p.relative_to(root).as_posix(), sha256=sha256_file(p),
                         size_bytes=p.stat().st_size)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 10))) as ex:
        return list(ex.map(one, paths))
