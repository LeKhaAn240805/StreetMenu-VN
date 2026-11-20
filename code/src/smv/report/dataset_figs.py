"""Dataset-side figures and tables for the paper (needs only `smv ingest`, no model results).

Everything is computed from the gold labels and the data index. No figure contains a menu image."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from smv.data.index import load_golds, load_index  # noqa: E402
from smv.report import figures as F  # noqa: E402
from smv.report.tables import write_table  # noqa: E402
from smv.schema.config import load_schema_config  # noqa: E402
from smv.settings import Settings  # noqa: E402

GROUP_NAMES = {"P_": "Price", "N_": "Name", "L_": "Logic", "S_": "Spatial", "V_": "Visual", "D_": "Distractor"}


def offer_table(st: Settings) -> pd.DataFrame:
    """One row per gold offer with image-level context (used by several tables and figures)."""
    rows = []
    for iid, ann in load_golds(st).items():
        for it in ann.items:
            for o in it.offers:
                rows.append({
                    "image_id": iid, "split": ann.split, "ccl": ann.ccl, "venue_tier": ann.venue_tier,
                    "domain": getattr(ann, "domain", "") or "", "item_status": it.item_status,
                    "legibility": it.legibility, "price_kind": o.price_kind, "price_source": o.price_source,
                    "evidence": o.evidence, "price_vnd": o.price_vnd, "ambiguous": bool(o.ambiguous),
                    "is_add_on": bool(it.is_add_on), "has_size": bool(o.size), "has_temp": bool(o.temp)})
    return pd.DataFrame(rows)


def _tag_order(cfg) -> list[tuple[str, str]]:
    return [(prefix, code) for prefix, codes in cfg.tag_groups.items() for code in codes]


# ------------------------------------------------------------------------------------------ figures
def fig_dataset_overview(st: Settings, out: Path) -> list[Path]:
    """(a) images per CCL level and split, (b) tag prevalence by group, (c) offers per image, (d) price source."""
    F._style()
    idx = load_index(st)
    idx = idx[(idx["perturbation"].fillna("") == "") & idx["has_gold"]]
    cfg = load_schema_config(st.schema_config_path)
    off = offer_table(st)
    menu = idx[idx["split"].isin(["dev", "test"])]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), gridspec_kw={"width_ratios": [1, 1.25], "hspace": 0.5,
                                                                    "wspace": 0.55})
    # (a) CCL by split
    ax = axes[0, 0]
    levels = [1, 2, 3, 4]
    bottoms = np.zeros(len(levels))
    for k, split in enumerate(["dev", "test"]):
        vals = np.array([int(((menu["split"] == split) & (menu["ccl"] == lv)).sum()) for lv in levels])
        ax.bar([f"CCL-{lv}" for lv in levels], vals, bottom=bottoms, color=F.SERIES[k], edgecolor=F.SURFACE,
               label=split, width=0.7)
        bottoms += vals
    for x, tot in zip(range(len(levels)), bottoms):
        ax.text(x, tot + 3, f"{int(tot)}", ha="center", fontsize=7, color=F.INK2)
    ax.set_ylabel("Images")
    ax.set_title("(a) Context complexity level")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)

    # (b) tag prevalence by group
    ax = axes[0, 1]
    tags = menu["tags"].fillna("").str.split(";")
    order = _tag_order(cfg)
    ylabels, vals, cols = [], [], []
    gcolor = {p: F.SERIES[i % 8] for i, p in enumerate(cfg.tag_groups)}
    for prefix, code in order:
        n = int(tags.map(lambda ts, c=code: c in ts).sum())
        ylabels.append(code)
        vals.append(100 * n / max(len(menu), 1))
        cols.append(gcolor[prefix])
    y = np.arange(len(ylabels))
    ax.barh(y, vals, color=cols, height=0.72, edgecolor=F.SURFACE)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=5.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Menu images carrying the tag (%)")
    ax.set_title("(b) Phenomenon tags (22, six groups)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=gcolor[p]) for p in cfg.tag_groups]
    ax.legend(handles, [GROUP_NAMES.get(p, p) for p in cfg.tag_groups], ncol=1, loc="center right", fontsize=6)

    # (c) offers per image
    ax = axes[1, 0]
    n = menu["n_offers"].clip(lower=1)
    bins = np.unique(np.round(np.logspace(0, np.log10(max(n.max(), 2) + 1), 16)).astype(int))
    ax.hist(n, bins=bins, color=F.SERIES[0], edgecolor=F.SURFACE)
    ax.set_xscale("log")
    thr = int(st.cfg.get("dense_threshold", 60))
    ax.axvline(thr, color=F.SERIES[1], linewidth=1.5)
    ax.text(thr * 1.08, ax.get_ylim()[1] * 0.92, f"dense: > {thr}\n({int((menu['n_offers'] > thr).sum())} images)",
            fontsize=6.5, color=F.INK2, va="top")
    ax.set_xlabel("Offers per image (log scale)")
    ax.set_ylabel("Images")
    ax.set_title(f"(c) Menu size (median {int(n.median())}, max {int(n.max())})")

    # (d) price source
    ax = axes[1, 1]
    menu_off = off[off["split"].isin(["dev", "test"])]
    src = menu_off["price_source"].value_counts()
    labels = [s.replace("_", " ") for s in src.index]
    ax.barh(labels, 100 * src.values / src.sum(), color=F.SERIES[2], height=0.7, edgecolor=F.SURFACE)
    for i, v in enumerate(100 * src.values / src.sum()):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=6.5, color=F.INK2)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of offers (%)")
    ax.set_title(f"(d) Where the price is written ({len(menu_off):,} offers)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(100 * src.values / src.sum()) * 1.18)
    return F._save(fig, out)


def fig_tag_cooccurrence(st: Settings, out: Path) -> list[Path]:
    """P(column tag | row tag) on menu images: which phenomena arrive together."""
    F._style()
    idx = load_index(st)
    menu = idx[(idx["perturbation"].fillna("") == "") & idx["has_gold"] & idx["split"].isin(["dev", "test"])]
    cfg = load_schema_config(st.schema_config_path)
    codes = [c for _, c in _tag_order(cfg)]
    sets = [set(t.split(";")) if t else set() for t in menu["tags"].fillna("")]
    m = np.array([[1 if c in s else 0 for c in codes] for s in sets], dtype=float)
    cnt = m.T @ m
    diag = np.diag(cnt).copy()
    cond = np.divide(cnt, diag[:, None], out=np.zeros_like(cnt), where=diag[:, None] > 0)
    cmap = LinearSegmentedColormap.from_list("blue", F.SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    im = ax.imshow(cond * 100, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=70, ha="right", fontsize=6)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([f"{c} ({int(d)})" for c, d in zip(codes, diag)], fontsize=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(codes)):
        for j in range(len(codes)):
            if i != j and cond[i, j] >= 0.5:
                ax.text(j, i, f"{cond[i, j] * 100:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if cond[i, j] > 0.6 else F.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("P(column tag | row tag), %")
    cb.outline.set_visible(False)
    ax.set_title("Tag co-occurrence on menu images (numbers: >= 50%)")
    return F._save(fig, out)


def fig_price_magnitude(st: Settings, out: Path) -> list[Path]:
    """Distribution of gold prices (log10 VND) by venue tier: why implicit-thousands errors are costly."""
    F._style()
    off = offer_table(st)
    off = off[off["split"].isin(["dev", "test"]) & off["price_vnd"].notna() & (off["price_vnd"] > 0)]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    bins = np.linspace(3, 6.8, 39)
    tiers = [t for t in ["street_vendor", "small_shop", "chain", "supermarket_foodcourt"] if (off["venue_tier"] == t).any()]
    for k, tier in enumerate(tiers):
        v = np.log10(off.loc[off["venue_tier"] == tier, "price_vnd"].astype(float))
        h, _ = np.histogram(v, bins=bins)
        ax.step(bins[:-1], 100 * h / max(h.sum(), 1), where="post", color=F.SERIES[k], linewidth=1.8,
                label=f"{tier.replace('_', ' ')} (n={len(v):,})")
    ticks = [3, 4, 5, 6]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1k", "10k", "100k", "1M"])
    ax.set_xlabel("Price (VND, log scale)")
    ax.set_ylabel("Share of offers per bin (%)")
    ax.legend(fontsize=6.5)
    return F._save(fig, out)


# ------------------------------------------------------------------------------------------ tables
def tables(st: Settings, out_dir: Path) -> list[Path]:
    off = offer_table(st)
    idx = load_index(st)
    idx = idx[(idx["perturbation"].fillna("") == "") & idx["has_gold"]]
    paths: list[Path] = []
    splits = ["dev", "test", "ood"]

    def piv(col: str, title: str, label: str, name: str) -> None:
        t = off.pivot_table(index=col, columns="split", values="image_id", aggfunc="count", fill_value=0)
        t = t.reindex(columns=[s for s in splits if s in t.columns])
        t["Total"] = t.sum(axis=1)
        t = t.sort_values("Total", ascending=False)
        t.loc["All"] = t.sum()
        t = t.reset_index().rename(columns={col: col.replace("_", " ")})
        for c in t.columns[1:]:
            t[c] = t[c].astype(int)
        paths.extend(write_table(t, out_dir / name, title, label))

    piv("price_kind", "Gold offers by price kind and split", "tab:pricekind", "table1f_price_kind")
    piv("price_source", "Gold offers by price source and split", "tab:pricesource", "table1g_price_source")
    piv("evidence", "Gold offers by evidence (printed, handwritten, overwritten)", "tab:evidence", "table1h_evidence")

    # out-of-domain split
    ood = idx[idx["split"] == "ood"]
    domains = {iid: getattr(a, "domain", "") for iid, a in load_golds(st).items()}
    ood = ood.assign(domain=ood["image_id"].map(domains))
    t = ood.groupby("domain").agg(Images=("image_id", "count"), Items=("n_items", "sum"), Offers=("n_offers", "sum"))
    t = t.reset_index().rename(columns={"domain": "Domain"})
    paths.extend(write_table(t, out_dir / "table1i_ood_domains",
                             "Out-of-domain split: price lists that are not food menus", "tab:ood"))

    # menu size by CCL
    menu = idx[idx["split"].isin(["dev", "test"])]
    rows = []
    for lv, d in [("All", menu)] + [(f"CCL-{int(k)}", g) for k, g in menu.groupby("ccl")]:
        n = d["n_offers"]
        rows.append({"Level": lv, "Images": len(d), "Offers (median)": int(n.median()), "Offers (p90)": int(n.quantile(0.9)),
                     "Offers (max)": int(n.max()), "Dense (>60)": int((n > 60).sum())})
    paths.extend(write_table(pd.DataFrame(rows), out_dir / "table1j_density", "Menu size by context complexity level",
                             "tab:density"))

    # scoring-set definition (rules R4/R6): what enters the main precision/recall
    rows = []
    for split in ["dev", "test", "ood"]:
        d = off[off["split"] == split]
        active = (d["item_status"] == "active") & (d["legibility"] != "illegible")
        main = d[active & ~d["ambiguous"]]
        rows.append({"Split": split, "Offers": len(d), "Inactive/illegible item": int((~active).sum()),
                     "Ambiguous": int((active & d["ambiguous"]).sum()), "In main set": len(main),
                     "of which no price (blank is right)": int(main["price_vnd"].isna().sum()),
                     "of which range": int((main["price_kind"] == "range").sum())})
    paths.extend(write_table(pd.DataFrame(rows), out_dir / "table1k_scoring_set",
                             "Gold offers entering the main precision/recall (rules R4 and R6)", "tab:scoringset"))

    # label provenance and review (disclosure table: every label starts as an AI draft that humans review)
    rows = []
    golds = load_golds(st)
    groups = [("All", list(golds.values()))]
    for lv in (1, 2, 3, 4):
        groups.append((f"CCL-{lv}", [a for a in golds.values() if a.ccl == lv]))
    groups.append(("OOD", [a for a in golds.values() if a.split == "ood"]))
    for name, anns in groups:
        if not anns:
            continue
        rev = [getattr(a, "review", None) or {} for a in anns]
        secs = [r.get("seconds") for r in rev if r.get("seconds")]
        n = len(anns)
        rows.append({"Group": name, "Images": n,
                     "AI draft": sum(1 for a in anns if getattr(a, "label_origin", "") == "claude_draft"),
                     "Confirmed unchanged": sum(1 for r in rev if r.get("status") == "confirmed"),
                     "Edited by reviewer": sum(1 for r in rev if r.get("status") == "edited"),
                     "Share edited (%)": f"{100 * sum(1 for r in rev if r.get('changed')) / n:.0f}",
                     "Median review time (s)": int(np.median(secs)) if secs else "–"})
    paths.extend(write_table(pd.DataFrame(rows), out_dir / "table1l_provenance",
                             "Label provenance: AI draft, then human review, then rule-based adjudication",
                             "tab:provenance",
                             note="Every label began as a draft written by an AI model (label_origin = claude_draft), was reviewed "
                                  "by one of two annotators and adjudicated by explicit rules; blind double annotation of 30 "
                                  "test images is reported separately."))
    return paths


def build_dataset_assets(st: Settings, out_root: Path | None = None) -> list[Path]:
    out_root = out_root or st.repo / "reports" / "dataset"
    paths: list[Path] = []
    paths += tables(st, out_root / "tables")
    paths += fig_dataset_overview(st, out_root / "figures" / "fig_dataset_overview")
    paths += fig_tag_cooccurrence(st, out_root / "figures" / "fig_tag_cooccurrence")
    paths += fig_price_magnitude(st, out_root / "figures" / "fig_price_magnitude")
    return paths


__all__ = ["build_dataset_assets", "offer_table", "Counter"]
