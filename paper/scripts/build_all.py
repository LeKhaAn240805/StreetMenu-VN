"""Regenerate every figure, table and number macro of the paper, then compile main.pdf.

Run from anywhere:  python scripts/build_all.py   (needs matplotlib, numpy, pillow, fonttools, rapidfuzz and a TeX
Live with latexmk). Fig. 1 crops the redacted images in $SMV_IMAGES (default ../data/images) or,
without them, reuses the cached crops in data/crops. The model results are read from the paper kit in $SMV_KIT or from ../results
(needs pandas and pyarrow) or, without it, from the cached CSVs in data/results.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["make_bib.py", "dataset_stats.py", "gold_reliability.py", "error_coding.py", "price_examples.py", "schematics.py",
           "protocol_tables.py", "prep_results.py", "model_results.py"]
for s in SCRIPTS:
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / s)], cwd=ROOT, check=True)
if "--no-pdf" not in sys.argv:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=ROOT, check=True)
