#!/bin/bash
# Synthetic dataset for real-model trial runs: every image in split dev (no frozen prompts needed).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${1:-outputs/_trial/data}
rm -rf "$D" && mkdir -p "$D"
cp -r tests/fixtures/menus/images "$D/images"
python=${PY:-python}
$python - "$D" <<'PY'
import csv, sys, pathlib
d = pathlib.Path(sys.argv[1])
names = sorted(p.name for p in (d / "images" / "_labels" / "synth").glob("s*.json"))
with open(d / "splits.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image_id", "split", "group_id"])
    for n in names:
        stem = n[:-5].rsplit(".", 1)[0]
        w.writerow([f"smv_{stem}", "dev", f"T_{stem}"])
    w.writerow(["smv_sub__orphan_no_label", "dev", "T_orphan"])
PY
echo "$D"
