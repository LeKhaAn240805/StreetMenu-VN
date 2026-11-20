"""Extract the model results used by the paper from the paper kit of the GPU server into small CSVs in data/results/.

The kit (StreetMenuVN_paper_kit_v2.zip) is read in place, without unpacking: $SMV_KIT, else
../server/StreetMenuVN_paper_kit_v2.zip next to the paper folder, else the results/ and code/ folders of the public
repository. Its score files were written by the released scorer
(claude/outputs/scores/all: summary.csv, per_image.parquet, long_offers.csv.gz) and its report tables by the same
pipeline. Nothing here re-scores a prediction: the script only filters, aggregates counts, and copies values, so every
number in Table 6 and Fig. 6 can be traced to a kit file. Reading the parquet file needs pyarrow; once the CSVs exist,
model_results.py rebuilds the table and figure without the kit.

Main setting = test split, prompt P1, thinking off, names with diacritics ("strict" name mode), strict price rule,
no image perturbation, seed 0.
"""
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from common import DATA, ROOT

KIT = Path(os.environ.get("SMV_KIT") or ROOT.parent / "server" / "StreetMenuVN_paper_kit_v2.zip")
# In the public repository the kit files sit in results/ and code/ next to paper/.
REPO_MAP = {"claude/outputs/scores/all/": "results/scores/", "claude/reports/": "results/reports/",
            "claude/configs/": "code/configs/", "StreetMenuVN_final/": "data/"}


class RepoKit:
    """Reads kit paths from the repository folders, so the same code serves the zip and the repository."""

    def __init__(self, root):
        self.root = root

    def _path(self, name):
        for a, b in REPO_MAP.items():
            if name.startswith(a):
                return self.root / (b + name[len(a):])
        raise KeyError(name)

    def open(self, name):
        return open(self._path(name), "rb")

    def read(self, name):
        return self._path(name).read_bytes()

OUT = DATA / "results"
SC = "claude/outputs/scores/all/"
TB = "claude/reports/all/tables/"
INHERITED = {"column_header", "section_header", "global_header", "merged_cell", "bracket"}
SOURCE = {"item_line": "line", "badge": "badge", "handwritten_override": "hand", **{s: "inh" for s in INHERITED}}

if KIT.exists():
    z = zipfile.ZipFile(KIT)
elif (ROOT.parent / "results" / "scores" / "summary.csv").exists():
    z = RepoKit(ROOT.parent)
else:
    print(f"note: kit not found at {KIT} (set SMV_KIT); using the cached CSVs in {OUT}")
    sys.exit(0)
OUT.mkdir(parents=True, exist_ok=True)


def read(name, **kw):
    with z.open(name) as f:
        data = f.read()
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    if name.endswith(".gz"):
        kw["compression"] = "gzip"
    return pd.read_csv(io.BytesIO(data), **kw)


def main_rows(d, split="test", prompt="P1", thinking="off"):
    keep = (d["split"] == split) & (d["prompt"] == prompt) & (d["thinking"] == thinking)
    if "mode" in d:
        keep &= d["mode"] == "strict"
    if "perturbation" in d:
        keep &= d["perturbation"].fillna("").astype(str).eq("")
    if "seed" in d:
        keep &= d["seed"].astype(str).eq("0")
    return d[keep]


names = {}
for line in z.read("claude/configs/display_names.yaml").decode("utf-8").splitlines():
    m = re.match(r"^([a-z0-9_]+):\s*(.+?)\s*$", line)
    if m:
        names[m.group(1)] = m.group(2)
rev = {v: k for k, v in names.items()}

# ----------------------------------------------------------------------------- per-model scores (main setting)
summ = read(SC + "summary.csv")
S = main_rows(summ).set_index("model")
pi = main_rows(read(SC + "per_image.parquet"))
splits = read("StreetMenuVN_final/splits.csv")
pi = pi.merge(splits[["image_id", "ccl", "in_blind_audit"]], on="image_id", how="left")


def f1(d):
    tp, fp, fn = d.off_tp_strict.sum(), d.off_fp_strict.sum(), d.off_fn_strict.sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


rows = []
for key, d in pi.groupby("model"):
    s = S.loc[key]
    cut = d.truncated.astype(bool)
    r = {"model": key, "name": names.get(key, key), "n_images": len(d),
         "offer_f1": s.offer_F1_strict, "offer_f1_lo": s.offer_F1_strict_lo, "offer_f1_hi": s.offer_F1_strict_hi,
         "offer_f1_check": f1(d), "offer_f1_lenient": s.offer_F1_lenient, "item_f1": s.item_F1,
         "pna": s.PNA_strict, "ipa": s.IPA_strict, "nna": s.NNA, "valid_json": s.valid_json_repaired,
         "truncated": s.truncation_rate, "repetition": s.repetition_rate, "latency_med_s": s.latency_median_s,
         "tokens_mean": s.completion_tokens_mean, "hallucination_rate": s.hallucination_rate,
         "omission_rate": s.omission_rate, "magnitude_share_of_wrong": s.magnitude_share_of_wrong,
         "null_price_ok": s.null_price_ok, "f1_complete": f1(d[~cut]), "f1_cut": f1(d[cut]),
         "gold_share_cut": d.gold_offers[cut].sum() / d.gold_offers.sum(),
         "audit_f1": f1(d[d.in_blind_audit == 1]), "audit_n": int((d.in_blind_audit == 1).sum())}
    for lv in (1, 2, 3, 4):
        r[f"ccl{lv}"] = f1(d[d.ccl == lv])
        r[f"ccl{lv}_n"] = int((d.ccl == lv).sum())
    rows.append(r)
M = pd.DataFrame(rows).set_index("model")

# recall and conditional price accuracy of gold offers by the source of their price
lo = main_rows(read(SC + "long_offers.csv.gz", dtype={"side": str}, low_memory=False,
                    usecols=["model", "seed", "thinking", "split", "prompt", "perturbation", "side", "status",
                             "gold_source"]))
lo = lo[lo.side == "gold"].assign(src=lambda x: x.gold_source.map(SOURCE))
scored = lo[lo.status.isin(["TP", "price_wrong", "FN_item_missing", "FN_offer_missing"])]
paired = lo[lo.status.isin(["TP", "price_wrong"])]
for name, d in (("rec", scored), ("acc", paired)):
    t = d.assign(ok=d.status.eq("TP")).groupby(["model", "src"]).ok.agg(["mean", "size"]).unstack()
    for src in ("line", "inh", "badge", "hand"):
        M[f"{name}_{src}"] = t["mean"][src]
        M[f"{name}_{src}_n"] = t["size"][src]

dep = read(TB + "table6_deployment.csv").replace("–", pd.NA)
dep["model"] = dep["Model"].map(rev)
dep = dep.set_index("model")
M["img_per_h"] = pd.to_numeric(dep["Images/h"], errors="coerce")
M["vram_gb"] = pd.to_numeric(dep["VRAM peak (GB)"], errors="coerce")
M["quant"] = dep["Quant"]
M.to_csv(OUT / "models.csv", float_format="%.6f")

# ----------------------------------------------------------------------------- secondary settings
alt = []
for prompt, split, thinking in (("P1@long", "test", "off"), ("P1", "test", "on"), ("P1", "ood", "off"),
                                ("P0", "test", "off"), ("P2", "test", "off")):
    for key, s in main_rows(summ, split, prompt, thinking).set_index("model").iterrows():
        alt.append({"model": key, "prompt": prompt, "split": split, "thinking": thinking, "n_images": s.n_images,
                    "offer_f1": s.offer_F1_strict, "offer_f1_lo": s.offer_F1_strict_lo,
                    "offer_f1_hi": s.offer_F1_strict_hi, "truncated": s.truncation_rate,
                    "latency_med_s": s.latency_median_s, "tokens_mean": s.completion_tokens_mean})
pd.DataFrame(alt).to_csv(OUT / "settings.csv", index=False, float_format="%.6f")

lng = read(TB + "table_longout.csv")
lng.insert(0, "model", lng["Model"].map(rev))
lng.to_csv(OUT / "longout.csv", index=False)
pr = read(TB + "table3_prompts.csv")
pr.insert(0, "model", pr["Model"].map(rev))
pr.to_csv(OUT / "prompts.csv", index=False)
for src, dst in (("pairwise_tests.csv", "pairwise.csv"), ("table_ranking_sensitivity.csv", "sensitivity.csv")):
    read(TB + src).to_csv(OUT / dst, index=False)

# human-gold agreement computed by the kit with the model scorer (tau 0.70)
txt = z.read("claude/reports/dataset/agreement_human_blind.txt").decode("utf-8")
row = next(line for line in txt.splitlines() if line.strip().startswith("human_blind"))
_, n, off, price, k_ccl, k_tags = row.split()
pd.DataFrame([{"images": int(n), "offer_f1": float(off) / 100, "price_match": float(price) / 100,
               "kappa_ccl": float(k_ccl), "kappa_tags": float(k_tags)}]).to_csv(OUT / "agreement_tau070.csv",
                                                                                 index=False)

bad = M[(M.offer_f1 - M.offer_f1_check).abs() > 1e-6]
print(f"{len(M)} models written to {OUT}; F1 recomputed from per-image counts differs for {len(bad)} model(s)")
