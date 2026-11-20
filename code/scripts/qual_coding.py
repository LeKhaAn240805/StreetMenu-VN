#!/usr/bin/env python3
"""Mixed-methods error analysis: turn the scored errors into a worksheet two people code by hand, then measure how much
they agree and what the themes are. This is the qualitative half of the evaluation (the quantitative half is `smv score`);
TAIMMA asks for mixed methods, and a benchmark paper is stronger when a human reading of the failures backs the metric.

    python scripts/qual_coding.py sample --n 150            # -> audit/qual/coding_sheet.csv + codebook.md (do this once)
    #   two people fill theme_coder1 / theme_coder2 (+ a short note) independently, reading the real images themselves
    python scripts/qual_coding.py score                      # -> reports/all/tables/table_qual_*.{md,tex}, kappa + themes

The worksheet carries only text already in the labels and the model outputs (raw_name, raw_price, gold vs predicted);
it never contains an image. The coders open the images on their own. The sample is stratified by model, by the automatic
error family and by complexity level (CCL) so every kind of failure is represented, not only the common ones."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smv.data.index import load_index  # noqa: E402
from smv.settings import get_settings  # noqa: E402
from smv.stats.bootstrap import cohen_kappa  # noqa: E402

MAIN_MODELS = ["qwen3vl_8b", "gemma4_e4b", "internvl35_8b", "ministral3_8b", "qwen35_9b", "minicpmv45"]
WRONG = {"price_wrong", "abstain_wrong", "FP_hallucination", "FP_desc_as_item", "FP_component_as_item",
         "FN_item_missing", "FN_offer_missing", "name_mismatch"}

# The qualitative codebook: themes a human assigns, grouped. Each automatic hint (configs/scoring.yaml) suggests a theme,
# but the coder decides from the image and may pick another, or "other" with a note. Kept deliberately short so two people
# can hold it in mind.
THEMES = {
    "read_print": "Misread printed text or digits (the price/name is printed clearly but transcribed wrong)",
    "read_hand": "Misread handwriting (the price/name is handwritten or written over)",
    "read_scale": "Currency scale (digits right, magnitude wrong: 15 vs 15,000; k/dot/ngàn)",
    "struct_inherit": "Did not apply a shared/inherited price (section or column header, bracket, 'same price')",
    "struct_variant": "Sizes/options mishandled (missing, merged, or mapped to the wrong price)",
    "struct_spatial": "Paired a price with the wrong item (layout, column, badge, distance)",
    "name_norm": "Name or section wrong (normalisation, gloss, composed section name) though the item is right",
    "halluc": "Invented an item or price, or output a description/heading/distractor as an item",
    "omit_dense": "Missed an item or offer on a dense menu (not because the answer was cut)",
    "omit_trunc": "Missed because the answer was cut off (truncation/repetition/invalid JSON)",
    "other": "None of the above (write a note)",
}
HINT_TO_THEME = {"E1": "read_print", "E2": "read_scale", "E3": "struct_inherit", "E4": "struct_variant",
                 "E5": "struct_spatial", "E6": "name_norm", "E7": "halluc", "E8": "omit_dense", "E0": "omit_trunc"}
SHEET = Path("audit/qual/coding_sheet.csv")
CTX = ["case_id", "model", "image_id", "ccl", "auto_hint", "auto_theme", "status", "gold_raw_name", "gold_raw_price",
       "gold_price", "pred_raw_name", "pred_raw_price", "pred_price", "truncated"]
CODE_COLS = ["theme_coder1", "note_coder1", "theme_coder2", "note_coder2"]


def _load_long(st) -> pd.DataFrame:
    lo = st.out_dir / "scores" / "all" / "long_offers.csv.gz"
    if not lo.is_file():
        sys.exit("no long_offers.csv.gz yet: run `smv score` first")
    d = pd.read_csv(lo, low_memory=False)
    ccl = load_index(st).drop_duplicates("image_id").set_index("image_id")["ccl"]
    d["ccl"] = d["base_image_id"].map(ccl)
    return d


def sample(st, n: int, seed: int) -> None:
    d = _load_long(st)
    d = d[(d["prompt"] == "P1") & (d["split"] == "test") & (d["thinking"] == "off")
          & (d["model"].isin(MAIN_MODELS)) & d["status"].isin(WRONG) & d["error_hint"].isin(HINT_TO_THEME)]
    if d.empty:
        sys.exit("no scored P1/test errors for the main models yet")
    d = d.assign(auto_theme=d["error_hint"].map(HINT_TO_THEME),
                 family=d["error_hint"].map(lambda h: HINT_TO_THEME[h].split("_")[0]))
    # stratify by automatic family x CCL, then fill proportionally but give every (family) at least a few
    rng = np.random.default_rng(seed)
    groups = list(d.groupby(["family", "ccl"], dropna=False))
    per = max(1, n // max(len(groups), 1))
    picks = []
    for _, g in groups:
        picks.append(g.sample(min(len(g), per), random_state=int(rng.integers(1 << 30))))
    out = pd.concat(picks)
    if len(out) < n:                        # top up at random from the rest to reach n
        rest = d.drop(out.index)
        if not rest.empty:
            out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # shuffle so coders see a mix
    out["case_id"] = [f"Q{i:03d}" for i in range(len(out))]
    out["truncated"] = (out["finish_reason"].fillna("") == "length")       # long_offers has no `truncated`; derive it
    for c in ("gold_raw_name", "gold_raw_price", "pred_raw_name", "pred_raw_price"):
        if c not in out:
            out[c] = ""
    sheet = out.reindex(columns=CTX)
    for c in CODE_COLS:
        sheet[c] = ""
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET, index=False)
    _write_codebook(SHEET.parent / "codebook.md", len(sheet))
    counts = out.groupby(["family", "ccl"]).size().unstack(fill_value=0)
    print(f"wrote {len(sheet)} cases to {SHEET}")
    print(f"codebook: {SHEET.parent / 'codebook.md'}")
    print("cases by automatic family x CCL:")
    print(counts.to_string())


def _write_codebook(path: Path, n: int) -> None:
    lines = [
        "# Error coding guide (StreetMenu-VN qualitative analysis)",
        "",
        f"Written for: two people who code the {n} cases in `coding_sheet.csv`. Work **independently** (do not discuss "
        "while coding); we measure how often you agree.",
        "",
        "## How to code one case",
        "1. Open the real image for `image_id` (you may look at images; the pipeline never does).",
        "2. Compare what the model wrote (`pred_*`) with the gold (`gold_*`) and the image.",
        "3. Put the single best theme code in `theme_coder1` (or `theme_coder2` for the second coder). Add a few words "
        "in the note when it helps or when you pick `other`.",
        "4. One theme per case: the *main* reason the prediction is wrong. `auto_theme` is the scorer's guess — you may "
        "agree or override it.",
        "",
        "## Themes",
        "",
        "| Code | Meaning |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {v} |" for k, v in THEMES.items()]
    lines += [
        "",
        "## Notes",
        "- Judge the *cause*, not the symptom: a missing price caused by a cut-off answer is `omit_trunc`, not `omit_dense` "
        "(`truncated` is true in the sheet).",
        "- `name_norm` is only when the item and price are essentially right but the name/section is off.",
        "- If two causes apply, pick the one a reader would fix first.",
        "- `gold_price`/`pred_price` are integers in đồng (null shown empty).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score(st) -> None:
    from smv.report.tables import write_table

    if not SHEET.is_file():
        sys.exit(f"{SHEET} not found: run `qual_coding.py sample` and have two people fill it")
    d = pd.read_csv(SHEET, dtype=str).fillna("")
    coded1 = d[d["theme_coder1"] != ""]                                    # all cases the primary coder coded -> themes
    both = d[(d["theme_coder1"] != "") & (d["theme_coder2"] != "")]        # double-coded subset (if any) -> Cohen's kappa
    out = st.repo / "reports" / "all" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    if coded1.empty:
        sys.exit(f"{SHEET}: no rows coded yet — fill theme_coder1 ({len(d)} cases total)")
    bad = sorted({t for c in ("theme_coder1", "theme_coder2") for t in d[c].unique() if t and t not in THEMES})
    if bad:
        print(f"warning: theme codes not in the codebook (treated as written): {bad}")

    paths: list = []
    # Inter-coder agreement is only reported when a second coder double-coded at least some cases. One coder alone
    # still yields the theme distribution below (a single-coder qualitative analysis); note the missing kappa in Limitations.
    if not both.empty:
        k_cc = cohen_kappa(list(both["theme_coder1"]), list(both["theme_coder2"]))
        agree = float((both["theme_coder1"] == both["theme_coder2"]).mean())
        k_a1 = cohen_kappa(list(both["theme_coder1"]), list(both["auto_theme"]))
        k_a2 = cohen_kappa(list(both["theme_coder2"]), list(both["auto_theme"]))
        cons = both[both["theme_coder1"] == both["theme_coder2"]]
        auto_ok = float((cons["theme_coder1"] == cons["auto_theme"]).mean()) if not cons.empty else float("nan")
        rows = [{"Measure": "Cases double-coded (2 coders)", "Value": f"{len(both)} of {len(coded1)}"},
                {"Measure": "Raw agreement (coder 1 = coder 2)", "Value": f"{agree * 100:.0f}%"},
                {"Measure": "Cohen's κ (coder 1 vs coder 2)", "Value": f"{k_cc:.2f}"},
                {"Measure": "Cohen's κ (coder 1 vs automatic)", "Value": f"{k_a1:.2f}"},
                {"Measure": "Cohen's κ (coder 2 vs automatic)", "Value": f"{k_a2:.2f}"},
                {"Measure": "Consensus cases where the scorer's code matches", "Value": f"{auto_ok * 100:.0f}%"}]
        paths += write_table(pd.DataFrame(rows), out / "table_qual_agreement",
                             "Inter-coder agreement on the error themes (double-coded subset)", "tab:qualagree",
                             note="Cohen's κ on the theme of each double-coded case. 'Automatic' is the scorer's error code "
                                  "mapped to a theme; the last row is how often the scorer agrees when the coders agree.")
        kmsg = f"κ(coders) = {k_cc:.2f} on {len(both)} double-coded, agreement {agree * 100:.0f}%"
    else:
        print("note: only theme_coder1 is filled — single coder, so Cohen's κ is skipped (themes-only report).")
        kmsg = "single coder (no κ)"

    # theme frequency over everything the primary coder coded (the qualitative finding)
    coded1 = coded1.assign(theme=coded1["theme_coder1"])
    freq = coded1["theme"].value_counts()
    tr = [{"Theme": THEMES.get(t, t), "Code": t, "Cases": int(c), "Share (%)": f"{100 * c / len(coded1):.0f}"}
          for t, c in freq.items()]
    title = ("Human-coded failure themes (one coder; P1, test, main models)" if both.empty else
             "Human-coded failure themes (primary coder; κ-validated on a double-coded subset; P1, test)")
    paths += write_table(pd.DataFrame(tr), out / "table_qual_themes", title, "tab:qualthemes",
                         note="One theme per case: the main reason the prediction is wrong.")
    (out.parent / "qual_examples.md").write_text(_examples(coded1), encoding="utf-8")
    print(f"coded {len(coded1)}/{len(d)} cases; {kmsg}")
    for p in paths:
        print(" ", p)


def _examples(coded: pd.DataFrame, per: int = 2) -> str:
    lines = ["# Representative coded failures (one or two per theme)", ""]
    for t, g in coded.groupby("theme"):
        lines.append(f"## {t} — {THEMES.get(t, t)}")
        for _, r in g.head(per).iterrows():
            note = r.get("note_coder1") or r.get("note_coder2") or ""
            lines.append(f"- **{r['case_id']}** ({r['model']}, CCL {r['ccl']}): gold `{r['gold_raw_name']}` = "
                         f"`{r['gold_raw_price']}` → pred `{r['pred_raw_name']}` = `{r['pred_raw_price']}`"
                         + (f" — {note}" if note else ""))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="build the coding worksheet and the codebook")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=2027)
    sub.add_parser("score", help="measure agreement and theme frequencies from the filled worksheet")
    a = ap.parse_args()
    st = get_settings()
    if a.cmd == "sample":
        sample(st, a.n, a.seed)
    else:
        score(st)


if __name__ == "__main__":
    main()
