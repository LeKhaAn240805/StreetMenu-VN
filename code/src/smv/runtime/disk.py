"""Disk space checks and the only deletion helper of the project (bounded to a root directory)."""
from __future__ import annotations

import fnmatch
import shutil
import time
from pathlib import Path


class UnsafeDeleteError(RuntimeError):
    pass


class DiskFullError(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total / 1e9


def can_fit(path: Path, need_gb: float, margin: float, min_free_gb: float) -> tuple[bool, float, float]:
    """Return (ok, free_gb, required_gb) where required = need * (1 + margin) + min_free_gb."""
    free = free_gb(path)
    req = need_gb * (1 + margin) + min_free_gb
    return free >= req, free, req


def safe_rmtree(target: Path, root: Path) -> bool:
    """Delete `target` only if, after resolving symlinks, it is strictly inside `root`."""
    root_r = Path(root).resolve()
    tgt = Path(target)
    if not tgt.exists() and not tgt.is_symlink():
        return False
    tgt_r = tgt.resolve()
    if tgt_r == root_r or root_r not in tgt_r.parents:
        raise UnsafeDeleteError(f"refusing to delete {tgt_r}: not strictly inside {root_r}")
    if str(root_r) in ("/", str(Path.home().resolve())):
        raise UnsafeDeleteError(f"refusing to use {root_r} as a deletion root")
    if tgt.is_symlink():
        tgt.unlink()
        return True
    if not tgt_r.is_dir():
        tgt_r.unlink()
        return True
    # NFS: a cached directory listing can miss files written moments ago (ENOTEMPTY on rmdir), and files
    # still open elsewhere become .nfsXXXX until closed. Retry a few times before giving up.
    last: OSError | None = None
    for attempt in range(8):
        try:
            shutil.rmtree(tgt_r)
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            last = e
            time.sleep(min(2 ** attempt, 15))
    raise last if last else OSError(f"could not delete {tgt_r}")


def clean_xet_cache(hf_home: Path) -> list[str]:
    """Remove hf_xet chunk caches below HF_HOME (they can hold a full copy of downloaded shards)."""
    removed = []
    hf_home = Path(hf_home)
    if not hf_home.is_dir():
        return removed
    xet = hf_home / "xet"
    if not xet.is_dir():
        return removed
    cands = [s for s in xet.rglob("*") if s.is_dir() and fnmatch.fnmatch(s.name, "chunk-cache")]
    for sub in cands:
        try:
            if sub.exists() and safe_rmtree(sub, hf_home):
                removed.append(str(sub))
        except (UnsafeDeleteError, OSError):
            pass
    return removed
