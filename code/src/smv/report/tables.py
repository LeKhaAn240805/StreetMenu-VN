"""Paper tables: Markdown + LaTeX (booktabs, Springer CCIS friendly)."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from smv.data.index import load_index
from smv.schema.config import load_schema_config
from smv.settings import Settings

_TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\~{}",
            "^": r"\^{}", "Δ": r"$\Delta$", "−": r"$-$", "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$",
            "×": r"$\times$", "τ": r"$\tau$", "κ": r"$\kappa$"}


def tex_escape(s) -> str:
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    return "".join(_TEX_ESC.get(ch, ch) for ch in s)


def fmt(v, digits: int = 1, pct: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, (int,)) and not pct:
        return str(v)
    if isinstance(v, float):
        return f"{v * 100:.{digits}f}" if pct else f"{v:.{digits}f}"
    return str(v)


def fmt_ci(v, lo, hi, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return fmt(v, digits)
    return f"{v * 100:.{digits}f} [{lo * 100:.{digits}f}, {hi * 100:.{digits}f}]"


def write_table(df: pd.DataFrame, out_base: Path, caption: str, label: str, col_align: str | None = None,
                note: str = "") -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    md = out_base.with_suffix(".md")
    tex = out_base.with_suffix(".tex")
    csv = out_base.with_suffix(".csv")
    df.to_csv(csv, index=False)
    lines = [f"**{caption}**", "", "| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    if note:
        lines += ["", note]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    align = col_align or ("l" + "r" * (len(df.columns) - 1))
    t = [r"\begin{table}[t]", r"\centering", rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}",
         r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
         " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        t.append(" & ".join(tex_escape(x) for x in r.values) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}"]
    if note:
        t.append(rf"\par\smallskip{{\footnotesize {tex_escape(note)}}}")
    t.append(r"\end{table}")
    tex.write_text("\n".join(t) + "\n", encoding="utf-8")
    return [md, tex, csv]


# ------------------------------------------------------------------------------------------ Table 1
def dataset_tables(st: Settings, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or st.repo / "reports" / "tables"
    idx = load_index(st)
    idx = idx[idx["perturbation"].fillna("") == ""]
    g = idx[idx["has_gold"]]
    cfg = load_schema_config(st.schema_config_path)
    paths: list[Path] = []
    rows = []
    for split, d in [("all", g)] + [(s, g[g["split"] == s]) for s in sorted(g["split"].unique())]:
        rows.append({"Split": split, "Images": len(d), "Items": int(d["n_items"].sum()), "Offers": int(d["n_offers"].sum()),
                     "Dense (>60 offers)": int(d["dense"].sum()),
                     "Chain": int((d["venue_tier"] == "chain").sum()),
                     "Street/small": int(d["venue_tier"].isin(["street_vendor", "small_shop"]).sum())})
    t1 = pd.DataFrame(rows)
    note = (f"{len(idx)} images scanned, {len(g)} with a usable gold label "
            f"({len(idx) - len(g)} without: unlabeled, draft only or orphan).")
    paths += write_table(t1, out_dir / "table1_dataset", "Dataset statistics by split", "tab:dataset", note=note)
    # CCL
    ccl = g.groupby("ccl", dropna=False).agg(Images=("image_id", "count"), Offers=("n_offers", "sum")).reset_index()
    ccl["ccl"] = ccl["ccl"].map(lambda x: "–" if pd.isna(x) else f"CCL-{int(x)}")
    paths += write_table(ccl.rename(columns={"ccl": "Level"}), out_dir / "table1b_ccl", "Images per context complexity level",
                         "tab:ccl")
    # tags
    tag_rows = []
    tags_series = g["tags"].fillna("").str.split(";")
    for prefix, codes in cfg.tag_groups.items():
        for code in codes:
            n = int(tags_series.map(lambda ts, c=code: c in ts).sum())
            tag_rows.append({"Group": prefix.rstrip("_"), "Tag": code, "Images": n,
                             "Share": f"{100 * n / max(len(g), 1):.1f}%"})
    paths += write_table(pd.DataFrame(tag_rows), out_dir / "table1c_tags", "Phenomenon tags (22 tags in 6 groups)",
                         "tab:tags", "llrr")
    # venue tier / type
    vt = g.groupby("venue_tier", dropna=False).size().reset_index(name="Images").rename(columns={"venue_tier": "Venue tier"})
    vt["Venue tier"] = vt["Venue tier"].replace("", "–")
    paths += write_table(vt, out_dir / "table1d_venue_tier", "Images by venue tier", "tab:venue")
    vty = g.groupby("venue_type", dropna=False).size().reset_index(name="Images").rename(columns={"venue_type": "Venue type"})
    vty["Venue type"] = vty["Venue type"].replace("", "–")
    paths += write_table(vty, out_dir / "table1e_venue_type", "Images by venue type", "tab:venuetype")
    return paths
