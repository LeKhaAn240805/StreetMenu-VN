"""Group-aware dev/test split. Groups merge near_duplicate_of links and same-group_id images."""
from __future__ import annotations

import csv
import random
from pathlib import Path


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_groups(rows: list[dict]) -> dict[str, str]:
    """rows: dicts with image_id, file_name, group_id, near_duplicate_of. Returns image_id -> group key.
    Chain venues are grouped by their group_id (annotators give one group_id per shop/chain)."""
    uf = _UF()
    by_name = {r["file_name"]: r["image_id"] for r in rows if r.get("file_name")}
    for r in rows:
        iid = r["image_id"]
        uf.find(iid)
        if r.get("group_id"):
            uf.union(iid, "G::" + r["group_id"])
        nd = r.get("near_duplicate_of")
        if nd and nd in by_name:
            uf.union(iid, by_name[nd])
    out = {}
    for r in rows:
        root = uf.find(r["image_id"])
        # prefer a human group id as the group key
        out[r["image_id"]] = root[3:] if root.startswith("G::") else (r.get("group_id") or root)
    # make keys consistent inside a component
    comp: dict[str, str] = {}
    for r in rows:
        root = uf.find(r["image_id"])
        comp.setdefault(root, out[r["image_id"]])
    return {r["image_id"]: comp[uf.find(r["image_id"])] for r in rows}


def assign_splits(rows: list[dict], ratio: dict[str, float], seed: int) -> dict[str, str]:
    """Assign `unassigned` images to splits by whole groups. Groups that already contain a dev/test image
    keep that split. Returns image_id -> split."""
    groups = build_groups(rows)
    fixed: dict[str, str] = {}
    for r in rows:
        s = r.get("split") or "unassigned"
        if s in ("dev", "test", "ood"):
            fixed.setdefault(groups[r["image_id"]], s)
    members: dict[str, list[str]] = {}
    for r in rows:
        members.setdefault(groups[r["image_id"]], []).append(r["image_id"])
    free = sorted(g for g in members if g not in fixed)
    rng = random.Random(seed)
    rng.shuffle(free)
    names = list(ratio)
    total = sum(len(members[g]) for g in members if fixed.get(g) != "ood")
    target = {n: ratio[n] * total for n in names}
    have = {n: sum(len(members[g]) for g, s in fixed.items() if s == n) for n in names}
    for g in free:
        # give the group to the split furthest below its target
        n = max(names, key=lambda k: (target[k] - have[k], -names.index(k)))
        fixed[g] = n
        have[n] += len(members[g])
    out = {}
    for r in rows:
        s = r.get("split") or "unassigned"
        out[r["image_id"]] = s if s in ("dev", "test", "ood") else fixed[groups[r["image_id"]]]
    return out


def read_splits_csv(path: Path) -> dict[str, dict]:
    if not path or not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["image_id"]: r for r in csv.DictReader(f) if r.get("image_id")}


def write_splits_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "split", "group_id"])
        w.writeheader()
        for r in rows:
            w.writerow({"image_id": r["image_id"], "split": r["split"], "group_id": r["group_id"]})
