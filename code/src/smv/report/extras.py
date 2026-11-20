"""Extra paper assets computed from scored results (`smv score` first): menu-size analysis, per-image score
distribution, pairwise significance, capability map, out-of-domain table, robustness, ranking sensitivity,
OCR-to-LLM cascade, calibration, text-only error examples, improved scale curves.

Every function takes the per-image frame written by `smv score` (one row per model x prompt x image x mode) and
returns the written paths. No figure contains a menu image."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from smv.report import figures as F  # noqa: E402
from smv.report.tables import fmt, write_table  # noqa: E402
from smv.stats.bootstrap import bootstrap_f1, holm, paired_bootstrap  # noqa: E402

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
SIZE_BINS = [(1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 60, "41-60"),
             (61, 10_000, ">60")]


# ------------------------------------------------------------------------------------------ naming
def load_display_names(repo: Path) -> dict[str, str]:
    p = repo / "configs" / "display_names.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.is_file() else {}


_QUANT_LABEL = {"awq": "AWQ-4bit", "fp8-weight-only": "FP8", "w4a16-qat": "QAT-4bit", "modelopt-fp8": "FP8",
                "gptq-int4": "GPTQ-4bit"}
_SUFFIXES = ("-Instruct-2512-BF16", "-Instruct-2512", "-BF16", "-Instruct", "-it-qat-w4a16-ct", "-it-AWQ-4bit",
             "-it-AWQ", "-AWQ-4bit", "-AWQ", "-FP8-Dynamic", "-FP8", "-it", "-Chat", "-hf", "-2508")


def pretty(key: str, specs: dict | None = None, names: dict[str, str] | None = None) -> str:
    """Display name: override file > Hugging Face repo name without release suffixes (+ quantisation) > key."""
    if names and key in names:
        return names[key]
    spec = (specs or {}).get(key)
    hf = getattr(spec, "hf_id", None)
    if not hf:
        return key
    base = hf.split("/")[-1]
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if base.endswith(suf) and len(base) > len(suf) + 2:
                base = base[: -len(suf)]
                changed = True
                break
    quant = getattr(spec, "quant", None)
    if quant:
        base += f" ({_QUANT_LABEL.get(quant, quant)})"
    return base


def family_colors(models: list[str], specs: dict | None) -> dict[str, str]:
    fams: list[str] = []
    for m in models:
        f = getattr(specs.get(m), "family", None) if specs else None
        f = f or m
        if f not in fams:
            fams.append(f)
    pal = F.SERIES + ["#7a7974", "#14b8c4"]
    return {m: pal[fams.index((getattr(specs.get(m), "family", None) if specs else None) or m) % len(pal)] for m in models}


# ------------------------------------------------------------------------------------------ helpers
def image_f1(df: pd.DataFrame, rule: str = "strict") -> pd.Series:
    tp, fp, fn = df[f"off_tp_{rule}"], df[f"off_fp_{rule}"], df[f"off_fn_{rule}"]
    den = 2 * tp + fp + fn
    return (2 * tp / den.where(den > 0)).fillna(1.0)


def _ci_f1(g: pd.DataFrame, n: int = 2000) -> tuple[float, float, float]:
    return bootstrap_f1(g["off_tp_strict"], g["off_fp_strict"], g["off_fn_strict"], n, 2027)


def _order_models(main: pd.DataFrame) -> list[str]:
    s = main.groupby("model")[["off_tp_strict", "off_fp_strict", "off_fn_strict"]].sum()
    f = 2 * s["off_tp_strict"] / (2 * s["off_tp_strict"] + s["off_fp_strict"] + s["off_fn_strict"]).replace(0, np.nan)
    return list(f.sort_values(ascending=False).index)


# ------------------------------------------------------------------------------------------ figures
def fig_density(main: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], out: Path,
                max_models: int = 8) -> list[Path]:
    """Offer F1 by menu size (gold offers per image) and the share of outputs cut at the token limit."""
    F._style()
    models = _order_models(main)[:max_models]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    xs = np.arange(len(SIZE_BINS))
    counts = []
    for k, m in enumerate(models):
        d = main[main["model"] == m]
        ys, lo, hi = [], [], []
        for a, b, _ in SIZE_BINS:
            g = d[(d["gold_offers"] >= a) & (d["gold_offers"] <= b)]
            if len(g) < 2 or (g[["off_tp_strict", "off_fp_strict", "off_fn_strict"]].sum().sum() == 0):
                ys.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            f, l_, h_ = _ci_f1(g)
            ys.append(f * 100)
            lo.append(l_ * 100)
            hi.append(h_ * 100)
        ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
        ax.plot(xs + (k - len(models) / 2) * 0.04, ys, marker=MARKERS[k % len(MARKERS)], color=colors[m],
                linewidth=1.6, markersize=4.5, markeredgecolor=F.SURFACE, label=labels.get(m, m))
        ax.fill_between(xs + (k - len(models) / 2) * 0.04, lo, hi, color=colors[m], alpha=0.10, linewidth=0)
    base = main[main["model"] == models[0]] if models else main
    counts = [int(((base["gold_offers"] >= a) & (base["gold_offers"] <= b)).sum()) for a, b, _ in SIZE_BINS]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\n(n={c})" for (_, _, lab), c in zip(SIZE_BINS, counts)], fontsize=6.5)
    ax.set_xlabel("Gold offers in the image")
    ax.set_ylabel("Offer F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Accuracy falls with menu size")
    ax.legend(fontsize=5.8, ncol=2, loc="lower left")
    ax = axes[1]
    for k, m in enumerate(models):
        d = main[main["model"] == m]
        ys = []
        for a, b, _ in SIZE_BINS:
            g = d[(d["gold_offers"] >= a) & (d["gold_offers"] <= b)]
            ys.append(100 * float(g["truncated"].astype(bool).mean()) if len(g) else np.nan)
        ax.plot(xs + (k - len(models) / 2) * 0.04, ys, marker=MARKERS[k % len(MARKERS)], color=colors[m],
                linewidth=1.6, markersize=4.5, markeredgecolor=F.SURFACE)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, _, lab in SIZE_BINS], fontsize=6.5)
    ax.set_xlabel("Gold offers in the image")
    ax.set_ylabel("Outputs cut at the token limit (%)")
    ax.set_ylim(0, max(5.0, ax.get_ylim()[1]))
    ax.set_title("(b) Dense menus overflow the output budget")
    return F._save(fig, out)


def fig_ecdf(main: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], out: Path,
             max_models: int = 8) -> list[Path]:
    """Empirical CDF of the per-image Offer F1: the mean hides how many images a model fails completely."""
    F._style()
    models = _order_models(main)[:max_models]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for m in models:
        v = np.sort(image_f1(main[main["model"] == m]).values * 100)
        ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=colors[m], linewidth=1.6,
                label=labels.get(m, m))
    ax.set_xlabel("Per-image Offer F1 (%)")
    ax.set_ylabel("Images with F1 at most x (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, loc="upper left")
    return F._save(fig, out)


def fig_pairwise(main: pd.DataFrame, labels: dict[str, str], out: Path, n_boot: int = 4000,
                 max_models: int = 10) -> tuple[list[Path], pd.DataFrame]:
    """Row minus column Offer F1 (points) with paired-bootstrap p-values, Holm-corrected over all pairs."""
    F._style()
    models = _order_models(main)[:max_models]
    cols = ["off_tp_strict", "off_fp_strict", "off_fn_strict"]
    piv = {m: main[main["model"] == m].set_index("base_image_id")[cols] for m in models}
    common = None
    for m in models:
        common = piv[m].index if common is None else common.intersection(piv[m].index)
    pairs, pv = [], []
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if j <= i:
                continue
            r = paired_bootstrap(tuple(piv[a].loc[common, c].values for c in cols),
                                 tuple(piv[b].loc[common, c].values for c in cols), n_boot)
            pairs.append((a, b, r))
            pv.append(r["p_boot"])
    adj = holm(pv) if pv else []
    rows = [{"model_a": a, "model_b": b, "diff": r["diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p_boot"],
             "p_holm": q} for (a, b, r), q in zip(pairs, adj)]
    table = pd.DataFrame(rows)
    k = len(models)
    mat = np.full((k, k), np.nan)
    star = [["" for _ in range(k)] for _ in range(k)]
    for r in rows:
        i, j = models.index(r["model_a"]), models.index(r["model_b"])
        mat[i, j], mat[j, i] = r["diff"] * 100, -r["diff"] * 100
        s = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else "*" if r["p_holm"] < 0.05 else ""
        star[i][j] = star[j][i] = s
    fig, ax = plt.subplots(figsize=(0.55 * k + 2.2, 0.5 * k + 1.6))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#fcfcfb", "#2a78d6"])
    ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(k))
    ax.set_xticklabels([labels.get(m, m) for m in models], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(k))
    ax.set_yticklabels([labels.get(m, m) for m in models], fontsize=6.5)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for i in range(k):
        for j in range(k):
            if i != j and np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.1f}{star[i][j]}", ha="center", va="center", fontsize=5.8, color=F.INK)
    ax.set_title("Offer F1 difference, row minus column (points)\n* p<.05, ** p<.01, *** p<.001 (paired bootstrap, Holm)",
                 fontsize=7)
    return F._save(fig, out), table


def fig_scale(overall: pd.DataFrame, specs: dict, labels: dict[str, str], out: Path) -> list[Path]:
    """Offer F1 against parameters, one line per family (>= 2 sizes); hollow markers are 4-bit or FP8 checkpoints."""
    F._style()
    rows = []
    for m in overall.index:
        s = specs.get(m)
        if s and s.params_b and s.family and s.role == "vlm":
            rows.append({"model": m, "family": s.family, "params": float(s.params_b), "f1": overall.loc[m, "offer_F1_strict"],
                         "lo": overall.loc[m].get("offer_F1_strict_lo", np.nan),
                         "hi": overall.loc[m].get("offer_F1_strict_hi", np.nan), "quant": bool(s.quant)})
    d = pd.DataFrame(rows)
    if d.empty:
        return []
    d = d[d.groupby("family")["model"].transform("count") >= 2]
    if d.empty:
        return []
    fams = list(dict.fromkeys(d.sort_values("f1", ascending=False)["family"]))
    pal = F.SERIES + ["#7a7974", "#14b8c4"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for k, fam in enumerate(fams):
        g = d[d["family"] == fam].sort_values("params")
        c = pal[k % len(pal)]
        ax.plot(g["params"], g["f1"] * 100, color=c, linewidth=1.6, zorder=2)
        for _, r in g.iterrows():
            ax.errorbar([r["params"]], [r["f1"] * 100], yerr=[[max(0, (r["f1"] - r["lo"]) * 100) if np.isfinite(r["lo"]) else 0],
                                                          [max(0, (r["hi"] - r["f1"]) * 100) if np.isfinite(r["hi"]) else 0]],
                        color=c, linewidth=0.8, capsize=1.5, zorder=2)
            ax.scatter([r["params"]], [r["f1"] * 100], s=26, zorder=3, color=(F.SURFACE if r["quant"] else c),
                       edgecolor=c, linewidth=1.3, marker=MARKERS[k % len(MARKERS)])
        ax.plot([], [], color=c, linewidth=1.6, marker=MARKERS[k % len(MARKERS)], markersize=5,
                markeredgecolor=c, label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B, log scale; hollow = 4-bit or FP8 weights)")
    ax.set_ylabel("Offer F1 (%)")
    ax.legend(fontsize=6.2, loc="lower right", ncol=2)
    return F._save(fig, out)


# ------------------------------------------------------------------------------------------ tables
CAPABILITIES = ["perception", "structure", "normalization", "reasoning"]


def table_capability(agg: pd.DataFrame, labels: dict[str, str], out_base: Path) -> list[Path]:
    """Four capabilities of schema v3.1 (capability_map): what each metric measures, reported side by side."""
    rows = []
    for m, r in agg.iterrows():
        rows.append({"Model": labels.get(m, m),
                     "Perception: price recall (P-T)": fmt(r.get("pt_price_recall")),
                     "Structure: Item F1": fmt(r.get("item_F1")), "Structure: Offer F1": fmt(r.get("offer_F1_strict")),
                     "Structure: Variant F1": fmt(r.get("variant_F1_strict")), "Structure: SAA": fmt(r.get("SAA_strict")),
                     "Normalization: PNA": fmt(r.get("PNA_strict")), "Normalization: NNA": fmt(r.get("NNA")),
                     "Reasoning: IPA": fmt(r.get("IPA_strict")), "Reasoning: blank price right": fmt(r.get("null_price_ok"))})
    return write_table(pd.DataFrame(rows), out_base,
                       "Four capabilities (perception, structure, normalization, reasoning), schema capability map",
                       "tab:capabilities",
                       note="Perception needs the transcription prompt P-T; a dash means it was not run.")


def f1_of(g: pd.DataFrame) -> float:
    tp, fp, fn = g["off_tp_strict"].sum(), g["off_fp_strict"].sum(), g["off_fn_strict"].sum()
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")


def table_ood(df: pd.DataFrame, domains: dict[str, str], labels: dict[str, str], main_p: str, out_base: Path) -> list[Path]:
    """In-domain (menus, test) against the out-of-domain price lists (services, retail)."""
    d = df[(df["prompt"] == main_p) & (df["thinking"] == "off") & (df["mode"] == "strict")]
    if not (d["split"] == "ood").any():
        return []
    d = d.assign(domain=d["base_image_id"].map(domains))
    rows = []
    for m in _order_models(d[d["split"] == "ood"]):
        g = d[d["model"] == m]
        r = {"Model": labels.get(m, m), "Test (menus)": fmt(f1_of(g[g["split"] == "test"]))}
        o = g[g["split"] == "ood"]
        r["OOD all"] = fmt(f1_of(o))
        for dom in ("service", "retail"):
            sub = o[o["domain"] == dom]
            r[f"OOD {dom} (n={sub['base_image_id'].nunique()})"] = fmt(f1_of(sub)) if len(sub) else "–"
        r["Gap (OOD - test)"] = f"{(f1_of(o) - f1_of(g[g['split'] == 'test'])) * 100:+.1f}"
        rows.append(r)
    return write_table(pd.DataFrame(rows), out_base, f"Out-of-domain price lists ({main_p}; Offer F1 %)", "tab:ood",
                       note="OOD images are service and retail price lists, not food menus; prompts were frozen before this split was run.")


def fig_robustness(df: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], main_p: str,
                   out: Path) -> list[Path]:
    """Offer F1 on perturbed copies against the clean copies of the same images."""
    d = df[(df["prompt"] == main_p) & (df["thinking"] == "off") & (df["mode"] == "strict")]
    pert = d[d["perturbation"].fillna("") != ""]
    if pert.empty:
        return []
    F._style()
    clean = d[(d["perturbation"].fillna("") == "") & d["base_image_id"].isin(pert["base_image_id"].unique())]
    kinds = sorted(pert["perturbation"].unique())
    models = [m for m in _order_models(pert) if (clean["model"] == m).any()]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        ys = []
        for kind in kinds:
            g = pert[(pert["model"] == m) & (pert["perturbation"] == kind)]
            c = clean[(clean["model"] == m) & clean["base_image_id"].isin(g["base_image_id"])]
            ys.append((f1_of(g) - f1_of(c)) * 100 if len(g) and len(c) else np.nan)
        ax.bar(np.arange(len(kinds)) + (k - (len(models) - 1) / 2) * w, ys, width=w * 0.9, color=colors[m],
               edgecolor=F.SURFACE, label=labels.get(m, m))
    ax.axhline(0, color=F.INK2, linewidth=0.8)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=20, ha="right")
    ax.set_ylabel("Offer F1 change vs clean image (points)")
    ax.legend(fontsize=6, ncol=min(len(models), 4), loc="lower left")
    ax.grid(axis="x", visible=False)
    return F._save(fig, out)


def table_cascade(df: pd.DataFrame, labels: dict[str, str], split: str, out_base: Path) -> list[Path]:
    """Image to JSON in one step against OCR text to JSON, and the human-transcript upper bound (P-Oracle)."""
    d = df[(df["split"] == split) & (df["mode"] == "strict") & (df["thinking"] == "off")]
    srcs = sorted({p for p in d["prompt"].unique() if p.startswith("P-OCR+@")})
    if not srcs and not (d["prompt"] == "P-Oracle").any():
        return []
    cols = ["P1"] + srcs + ["P-Oracle"]
    rows = []
    for m in dict.fromkeys(d["model"]):
        r = {"Model": labels.get(m, m)}
        have = False
        for c in cols:
            g = d[(d["model"] == m) & (d["prompt"] == c)]
            r[c.replace("P-OCR+@", "OCR: ").replace("ocr_", "")] = fmt(f1_of(g)) if len(g) else "–"
            have |= len(g) > 0 and c != "P1"
        if have:
            rows.append(r)
    if not rows:
        return []
    return write_table(pd.DataFrame(rows), out_base, f"Image to JSON against OCR text to JSON ({split}; Offer F1 %)",
                       "tab:cascade", note="P1: the model reads the image. OCR: the model reads the transcript of the named OCR engine "
                                           "(plus the image for VLMs). P-Oracle: a human transcript of the item lines.")


def table_sensitivity(rankings: dict[str, pd.Series], out_base: Path, main_key: str) -> list[Path]:
    """Kendall tau and top-1 agreement between the main ranking and rankings under other scoring choices."""
    from scipy.stats import kendalltau

    base = rankings[main_key]
    rows = []
    for name, s in rankings.items():
        common = base.index.intersection(s.index)
        tau = kendalltau(base.loc[common], s.loc[common])[0] if len(common) > 2 else float("nan")
        rows.append({"Scoring variant": name, "Kendall tau vs main": f"{tau:.3f}" if tau == tau else "–",
                     "Top-1": s.idxmax(), "Top-1 same": "yes" if s.idxmax() == base.idxmax() else "no",
                     "Models": len(common)})
    return write_table(pd.DataFrame(rows), out_base, "Ranking stability under scoring choices", "tab:sensitivity")


def error_examples(long: pd.DataFrame, labels: dict[str, str], out_base: Path, per_code: int = 4,
                   seed: int = 2027) -> list[Path]:
    """Text-only examples of each suggested error code (gold text beside the prediction)."""
    d = long[long["error_hint"].fillna("") != ""]
    if d.empty:
        return []
    rows = []
    rng = np.random.default_rng(seed)
    for code, g in d.groupby("error_hint"):
        g = g[g["gold_raw_name"].fillna("").astype(str).str.len().between(3, 40)]
        take = g.iloc[rng.permutation(len(g))[:per_code]] if len(g) else g
        for _, r in take.iterrows():
            rows.append({"Code": code, "Model": labels.get(r.get("model"), r.get("model")), "Gold item": r.get("gold_raw_name", ""),
                         "Gold raw price": r.get("gold_raw_price", ""), "Gold VND": r.get("gold_price", ""),
                         "Predicted item": r.get("pred_item", ""), "Predicted VND": r.get("pred_price", "")})
    return write_table(pd.DataFrame(rows), out_base, "Text-only error examples by suggested code", "tab:errorexamples")


def reliability(long: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], out: Path) -> tuple[list[Path], pd.DataFrame]:
    """Reliability diagram for the confidence prompt: does confidence track whether price_vnd is right?"""
    d = long[(long["prompt"] == "P-Conf") & long["pred_conf"].notna() & long["status"].isin(["TP", "price_wrong"])]
    if d.empty:
        return [], pd.DataFrame()
    F._style()
    d = d.assign(ok=(d["status"] == "TP").astype(float))
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=F.GRID, linewidth=1.2)
    rows = []
    for k, (m, g) in enumerate(d.groupby("model")):
        b = np.clip(np.digitize(g["pred_conf"].astype(float), edges) - 1, 0, 9)
        acc = [g["ok"][b == i].mean() if (b == i).any() else np.nan for i in range(10)]
        conf = [g["pred_conf"].astype(float)[b == i].mean() if (b == i).any() else np.nan for i in range(10)]
        n = np.array([(b == i).sum() for i in range(10)])
        ece = float(np.nansum(np.abs(np.array(acc) - np.array(conf)) * n) / max(n.sum(), 1))
        rows.append({"model": m, "n": int(n.sum()), "ECE": ece, "mean_conf": float(g["pred_conf"].astype(float).mean()),
                     "accuracy": float(g["ok"].mean())})
        ax.plot(conf, acc, marker=MARKERS[k % len(MARKERS)], color=colors.get(m, F.SERIES[k % 8]), linewidth=1.4,
                markersize=4, label=f"{labels.get(m, m)} (ECE {ece:.2f})")
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Share of prices that are right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5.8, loc="upper left")
    return F._save(fig, out), pd.DataFrame(rows)


def fig_budget(df: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], split: str,
               out: Path) -> tuple[list[Path], pd.DataFrame]:
    """Offer F1 against the visual budget: the same frozen prompt P1 at several per-request image budgets (P1@tag),
    the model's own default budget being its ordinary P1 run. x = mean prompt tokens per image (log)."""
    d = df[(df["split"] == split) & (df["mode"] == "strict") & (df["thinking"] == "off")
           & df["prompt"].str.split("@").str[0].eq("P1") & (df["perturbation"].fillna("") == "")]
    # only the image-budget tags (px..., tok..., tile...): P1@lp, P1@rp105 and P1@t07 are decoding variants, not budgets
    d = d[~d["prompt"].str.contains("@") | d["prompt"].str.split("@").str[1].fillna("").str.match(r"(px|tok|tile)")]
    models = [m for m in d["model"].unique() if d[(d["model"] == m) & d["prompt"].str.contains("@")].shape[0]]
    if not models:
        return [], pd.DataFrame()
    F._style()
    rows = []
    for m in models:
        for pid, g in d[d["model"] == m].groupby("prompt"):
            rows.append({"model": m, "prompt": pid, "tag": pid.split("@")[1] if "@" in pid else "default",
                         "offer_F1": f1_of(g), "prompt_tokens": float(pd.to_numeric(g["prompt_tokens"], errors="coerce").mean()),
                         "latency_median_s": float(pd.to_numeric(g["latency_s"], errors="coerce").median()),
                         "n_images": g["base_image_id"].nunique()})
    t = pd.DataFrame(rows).sort_values(["model", "prompt_tokens"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"wspace": 0.3})
    for k, m in enumerate(models):
        g = t[t["model"] == m]
        for ax, ycol, scale in ((axes[0], "offer_F1", 100), (axes[1], "latency_median_s", 1)):
            ax.plot(g["prompt_tokens"], g[ycol] * scale, marker=MARKERS[k % len(MARKERS)], color=colors.get(m, F.SERIES[k % 8]),
                    linewidth=1.6, markersize=5, markeredgecolor=F.SURFACE, label=labels.get(m, m))
        for _, r in g.iterrows():
            axes[0].annotate(r["tag"], (r["prompt_tokens"], r["offer_F1"] * 100), xytext=(3, -8), textcoords="offset points",
                             fontsize=5.5, color=F.INK2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Prompt tokens per request (log scale)")
    axes[0].set_ylabel("Offer F1 (%)")
    axes[1].set_ylabel("Median latency per image (s)")
    axes[0].set_title("(a) Accuracy against visual budget")
    axes[1].set_title("(b) Cost of the budget")
    axes[0].legend(fontsize=6, loc="lower right")
    return F._save(fig, out), t


def headroom(long: pd.DataFrame, index: pd.DataFrame, labels: dict[str, str], main_p: str, split: str,
             out_fig: Path, out_tbl: Path, top_k: int = 12) -> tuple[list[Path], list[Path]]:
    """What no model solves. A gold offer is solved by a model when its row is a true positive. Panel (a): share of gold
    offers solved by at least one of the k best models (oracle ensemble) and by all of them. Panel (b): which tags and
    price sources are over-represented among the offers that no model solves."""
    d = long[(long["prompt"] == main_p) & (long["split"] == split) & (long["thinking"] == "off")]
    g = d[d["side"] == "gold"].copy()
    if g.empty or g["model"].nunique() < 2:
        return [], []
    key = ["base_image_id", "gold_item", "gold_size", "gold_temp", "gold_portion", "gold_price", "gold_raw_price"]
    g[key] = g[key].fillna("")
    g["solved"] = (g["status"] == "TP").astype(int)
    g = g[~g["status"].isin(["unscored", "abstain_ok", "abstain_wrong"])]
    piv = g.groupby(key + ["model"])["solved"].max().unstack("model").fillna(0)
    order = list(piv.mean().sort_values(ascending=False).index)[:top_k]
    piv = piv[order]
    cum_any = [float(piv[order[: k + 1]].max(axis=1).mean()) * 100 for k in range(len(order))]
    cum_all = [float(piv[order[: k + 1]].min(axis=1).mean()) * 100 for k in range(len(order))]
    F._style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.45})
    ax = axes[0]
    xs = np.arange(1, len(order) + 1)
    ax.plot(xs, cum_any, marker="o", color=F.SERIES[0], linewidth=1.8, markersize=4.5, label="solved by at least one")
    ax.plot(xs, cum_all, marker="s", color=F.SERIES[1], linewidth=1.8, markersize=4.5, label="solved by all")
    ax.set_xlabel("Models pooled (best first)")
    ax.set_ylabel("Gold offers solved (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(xs[:: max(1, len(xs) // 6)])
    ax.set_title("(a) Headroom of an oracle ensemble")
    ax.legend(fontsize=6, loc="center right")
    never = piv.max(axis=1) == 0
    tagged = piv.reset_index()[["base_image_id"]].assign(never=never.values)
    meta = index[["image_id", "tags"]].rename(columns={"image_id": "base_image_id"})
    tagged = tagged.merge(meta, on="base_image_id", how="left")
    tagged["tags"] = tagged["tags"].fillna("").str.split(";")
    rows = []
    for tag in sorted({t for ts in tagged["tags"] for t in ts if t}):
        has = tagged["tags"].map(lambda ts, t=tag: t in ts)
        if has.sum() < 30:
            continue
        rows.append({"tag": tag, "offers": int(has.sum()), "share_unsolved_with_tag": float(never[has.values].mean()) * 100,
                     "share_unsolved_overall": float(never.mean()) * 100})
    t = pd.DataFrame(rows).sort_values("share_unsolved_with_tag", ascending=False)
    ax = axes[1]
    top = t.head(10).iloc[::-1]
    ax.barh(top["tag"], top["share_unsolved_with_tag"], color=F.SERIES[2], height=0.7, edgecolor=F.SURFACE)
    ax.axvline(float(never.mean()) * 100, color=F.SERIES[1], linewidth=1.4)
    ax.tick_params(axis="y", length=0, labelsize=6)
    ax.set_xlabel("Offers no model solves (%), by image tag")
    ax.set_title(f"(b) Unsolved offers: {float(never.mean()) * 100:.1f}% overall (line)")
    ax.grid(axis="y", visible=False)
    paths = F._save(fig, out_fig)
    summary = pd.DataFrame({"k_models": xs, "model_added": [labels.get(m, m) for m in order], "solved_by_any_pct": cum_any,
                            "solved_by_all_pct": cum_all})
    tbl = write_table(summary.assign(solved_by_any_pct=lambda x: x["solved_by_any_pct"].round(1),
                                     solved_by_all_pct=lambda x: x["solved_by_all_pct"].round(1)),
                      out_tbl, f"Oracle ensemble headroom ({main_p}, {split})", "tab:headroom",
                      note="Models are added best first; a gold offer counts as solved when the model's row is a true positive.")
    t.to_csv(out_tbl.with_name(out_tbl.name + "_tags.csv"), index=False)
    return paths, tbl


def fig_price_severity(long: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], main_p: str, split: str,
                       out: Path, out_tbl: Path, max_models: int = 8) -> tuple[list[Path], list[Path]]:
    """How wrong are the wrong prices? log10(predicted / gold) for every offer with a wrong price: a scale mistake is a
    bar at +-1 or +-2 (a factor 10 or 100), a near miss sits close to 0. The table lists the share of wrong prices by size."""
    d = long[(long["prompt"] == main_p) & (long["split"] == split) & (long["thinking"] == "off")
             & (long["status"] == "price_wrong")].copy()
    d["gold_price"] = pd.to_numeric(d["gold_price"], errors="coerce")
    d["pred_price"] = pd.to_numeric(d["pred_price"], errors="coerce")
    d = d[(d["gold_price"] > 0) & (d["pred_price"] > 0)]
    if d.empty:
        return [], []
    d["logr"] = np.log10(d["pred_price"] / d["gold_price"])
    d["rel"] = (d["pred_price"] - d["gold_price"]).abs() / d["gold_price"]
    models = d.groupby("model").size().sort_values(ascending=False).index.tolist()[:max_models]
    F._style()
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for k, m in enumerate(models):
        v = np.sort(d.loc[d["model"] == m, "logr"].values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=colors.get(m, F.SERIES[k % 8]),
                linewidth=1.5, label=f"{labels.get(m, m)} (n={len(v)})")
    for x in (-2, -1, 1, 2):
        ax.axvline(x, color=F.GRID, linewidth=0.8)
    ax.set_xlabel("log10(predicted price / gold price) of wrong prices")
    ax.set_ylabel("Wrong prices at most x (%)")
    ax.set_xlim(-3, 3)
    ax.legend(fontsize=5.8, loc="upper left")
    paths = F._save(fig, out)
    rows = []
    for m in models:
        g = d[d["model"] == m]
        rows.append({"Model": labels.get(m, m), "Wrong prices": len(g),
                     "within 10% (%)": f"{100 * (g['rel'] <= 0.10).mean():.0f}",
                     "10-100% off (%)": f"{100 * ((g['rel'] > 0.10) & (g['rel'] <= 1)).mean():.0f}",
                     "more than 2x off (%)": f"{100 * (g['rel'] > 1).mean():.0f}",
                     "factor 10 or more (%)": f"{100 * (g['logr'].abs() >= 0.9).mean():.0f}",
                     "median abs log10 ratio": f"{g['logr'].abs().median():.2f}"})
    tbl = write_table(pd.DataFrame(rows), out_tbl, f"How wrong are the wrong prices ({main_p}, {split})", "tab:severity",
                      note="Relative error |predicted - gold| / gold of every offer whose price is wrong.")
    return paths, tbl


def finite(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and math.isfinite(v)


# ------------------------------------------------------------------------------------------ driver
def ranking_variants(st, models: list[str], main_p: str, split: str, df: pd.DataFrame, scfg: dict) -> dict[str, pd.Series]:
    """Offer F1 per model under alternative scoring choices (all on the same split and prompt)."""
    import tempfile

    from smv.scoring.score import score_results

    def f1_by_model(d: pd.DataFrame, rule: str) -> pd.Series:
        g = d.groupby("model")[[f"off_tp_{rule}", f"off_fp_{rule}", f"off_fn_{rule}"]].sum()
        return 2 * g[f"off_tp_{rule}"] / (2 * g[f"off_tp_{rule}"] + g[f"off_fp_{rule}"] + g[f"off_fn_{rule}"])

    sel = df[(df["split"] == split) & (df["prompt"] == main_p) & (df["thinking"] == "off")
             & (df["perturbation"].fillna("") == "")]
    out: dict[str, pd.Series] = {}
    out[f"main (strict price, strict names, tau {float(scfg.get('tau', 0.8)):.2f}, repeats merged)"] = f1_by_model(sel[sel["mode"] == "strict"], "strict")
    out["lenient price rule"] = f1_by_model(sel[sel["mode"] == "strict"], "lenient")
    out["names without diacritics"] = f1_by_model(sel[sel["mode"] == "lenient"], "strict")
    micro_img = sel[sel["mode"] == "strict"].assign(f1=lambda x: image_f1(x))
    out["macro over images"] = micro_img.groupby("model")["f1"].mean()
    base_tau = float(scfg.get("tau", 0.8))
    variants = [(f"tau {t:.2f}", {"tau": t}, None) for t in (round(base_tau - 0.10, 2), round(base_tau + 0.10, 2))]
    variants.append(("repeated items counted separately", {}, {"dedupe_repeats": False}))
    for label, kw, ov in variants:
        with tempfile.TemporaryDirectory(dir=str(st.cache_root / "tmp") if (st.cache_root / "tmp").is_dir() else None) as td:
            score_results(st, models, prompts=[main_p], out_dir=Path(td), with_rows=False, overrides=ov, **kw)
            d2 = pd.read_parquet(Path(td) / "per_image.parquet")
        d2 = d2[(d2["split"] == split) & (d2["thinking"] == "off") & (d2["mode"] == "strict")
                & (d2["perturbation"].fillna("") == "")]
        if len(d2):
            out[label] = f1_by_model(d2, "strict")
    return out


def index_for_headroom(st) -> pd.DataFrame:
    from smv.data.index import load_index

    idx = load_index(st)
    return idx[idx["perturbation"].fillna("") == ""]


def build_extras(st, name: str, df: pd.DataFrame, specs: dict, scfg: dict, main_p: str, split: str,
                 overall: pd.DataFrame) -> list[Path]:
    """All extra assets for plan `name`; each asset is skipped when its input was not run."""
    from smv.data.index import load_golds
    from smv.scoring.score import aggregate

    out = st.repo / "reports" / name
    tdir, fdir = out / "tables", out / "figures"
    paths: list[Path] = []
    names = load_display_names(st.repo)
    labels = {m: pretty(m, specs, names) for m in df["model"].unique()}
    strict = df[df["mode"] == "strict"]
    json_df = strict[strict["gold_offers"].notna() & ~strict["prompt"].isin(["P-T"])]
    clean = json_df[json_df["perturbation"].fillna("") == ""]
    main = clean[(clean["split"] == split) & (clean["prompt"] == main_p) & (clean["thinking"] == "off")]
    if main.empty:
        return paths
    models = _order_models(main)
    colors = family_colors(models, specs)
    paths += fig_density(main, labels, colors, fdir / "fig_f1_by_menu_size")
    main.assign(f1=image_f1(main)).to_csv(fdir / "per_image_f1_main.csv", index=False)
    paths += fig_ecdf(main, labels, colors, fdir / "fig_f1_ecdf")
    if len(models) >= 2:
        p, tbl = fig_pairwise(main, labels, fdir / "fig_pairwise_delta")
        paths += p
        tbl.to_csv(tdir / "pairwise_tests.csv", index=False)
    ov = overall.reindex(models)
    paths += fig_scale(ov, specs, labels, fdir / "fig_scale_curves")
    # capability map: P-T perception comes from its own prompt
    cap = aggregate(main, ["model"], scfg, ci=False).set_index("model")
    pt = strict[(strict["prompt"] == "P-T") & (strict["split"] == split)]
    if not pt.empty:
        ptagg = pt.groupby("model")[["pt_price_hit", "pt_price_n"]].sum()
        cap["pt_price_recall"] = ptagg["pt_price_hit"] / ptagg["pt_price_n"].replace(0, np.nan)
    paths += table_capability(cap.reindex(models), labels, tdir / "table_capabilities")
    golds = load_golds(st)
    domains = {iid: getattr(a, "domain", "") for iid, a in golds.items()}
    paths += table_ood(strict, domains, labels, main_p, tdir / "table_ood")
    paths += fig_robustness(json_df, labels, colors, main_p, fdir / "fig_robustness")
    pb, bt = fig_budget(df, labels, colors, split, fdir / "fig_budget")
    paths += pb
    if not bt.empty:
        bt.to_csv(tdir / "budget_study.csv", index=False)
    paths += table_cascade(strict, labels, split, tdir / "table_cascade")
    lo = st.out_dir / "scores" / name / "long_offers.csv.gz"
    if lo.is_file():
        long = pd.read_csv(lo)
        long = long[(long["split"] == split) & (long["thinking"] == "off")]
        paths += error_examples(long[long["prompt"] == main_p], labels, tdir / "table_error_examples")
        psev, tsev = fig_price_severity(long, labels, colors, main_p, split, fdir / "fig_price_severity",
                                        tdir / "table_price_severity")
        paths += psev + tsev
        ph, tb = headroom(long, index_for_headroom(st), labels, main_p, split, fdir / "fig_headroom", tdir / "table_headroom")
        paths += ph + tb
        p, rel = reliability(long, labels, colors, fdir / "fig_reliability")
        paths += p
        if not rel.empty:
            rel.to_csv(tdir / "reliability.csv", index=False)
    if len(models) >= 3:
        try:
            rk = ranking_variants(st, models, main_p, split, df, scfg)
            paths += table_sensitivity(rk, tdir / "table_ranking_sensitivity", next(iter(rk)))
        except Exception as e:  # noqa: BLE001 - the sensitivity re-scoring must not break the report
            (out / "sensitivity_error.txt").write_text(repr(e))
    return paths
