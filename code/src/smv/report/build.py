"""`smv report`: every paper table + figure from outputs/scores/<name>/ (run `smv score` first)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from smv.data.index import load_index
from smv.models.registry import load_all
from smv.report import figures as F
from smv.report.tables import dataset_tables, fmt, fmt_ci, write_table
from smv.scoring.score import aggregate, attach_meta, slice_table
from smv.settings import Settings
from smv.stats.bootstrap import holm, paired_bootstrap, permutation_test


def _pick_split(df: pd.DataFrame) -> str:
    for s in ("test", "dev", "robust"):
        if (df["split"] == s).any():
            return s
    return str(df["split"].iloc[0])


def _main_prompt(df: pd.DataFrame) -> str:
    for p in ("P1", "P0", "P2"):
        if (df["prompt"] == p).any():
            return p
    return str(df["prompt"].iloc[0])


def load_manifests(st: Settings, plan: str | None = None) -> pd.DataFrame:
    rows = []
    roots = [st.plan_dir(plan)] if plan and st.plan_dir(plan).is_dir() else [p for p in st.out_dir.iterdir() if p.is_dir()]
    for root in roots:
        for p in root.glob("*/manifests/*.json"):
            try:
                m = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            m["manifest_path"] = str(p)
            rows.append(m)
    return pd.DataFrame(rows)


def build_report(st: Settings, name: str = "all") -> list[Path]:
    sdir = st.out_dir / "scores" / name
    out = st.repo / "reports" / name
    tdir, fdir = out / "tables", out / "figures"
    paths: list[Path] = []
    try:
        paths += dataset_tables(st, tdir)
    except FileNotFoundError:
        pass
    pi = sdir / "per_image.parquet"
    if not pi.is_file():
        (out / "README.md").parent.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text("No scores yet: run `smv score --plan ...` (needs gold labels).\n")
        return paths + [out / "README.md"]
    df = pd.read_parquet(pi)
    if df.empty:
        return paths
    scfg = st.scoring_cfg()
    index = load_index(st)
    specs = load_all(st.model_configs_dir)
    from smv.report.extras import load_display_names, pretty

    _names = load_display_names(st.repo)

    def lab(m: str) -> str:
        """Display name used in every table and figure (configs/display_names.yaml, else the Hugging Face repo name)."""
        return pretty(m, specs, _names)
    strict = df[df["mode"] == "strict"]
    json_df = strict[strict["gold_offers"].notna() & ~strict["prompt"].isin(["P-T"])]
    main_p = _main_prompt(json_df)
    split = _pick_split(json_df[json_df["prompt"] == main_p])
    main = json_df[(json_df["split"] == split) & (json_df["prompt"] == main_p) & (json_df["thinking"] == "off")]
    if main.empty:
        main = json_df[(json_df["split"] == split) & (json_df["prompt"] == main_p)]
    models = list(dict.fromkeys(main["model"]))
    if not models:
        return paths

    # ---- Table 2: model x CCL ----------------------------------------------------------------
    overall = aggregate(main, ["model"], scfg).set_index("model")
    by_ccl = slice_table(main, index, "ccl", st.schema_config_path, ["model"], scfg).copy()
    rows = []
    for m in models:
        o = overall.loc[m]
        sp = specs.get(m)
        r = {"Model": lab(m), "Params (B)": (f"{sp.params_b:g}" if sp and sp.params_b else "–"),
             "Offer F1 [95% CI]": fmt_ci(o["offer_F1_strict"], o.get("offer_F1_strict_lo"), o.get("offer_F1_strict_hi")),
             "Macro F1": fmt(o["offer_F1_macro_strict"])}
        for lv in (1, 2, 3, 4):
            v = by_ccl[(by_ccl["model"] == m) & (by_ccl["ccl"] == lv)]["offer_F1_strict"]
            r[f"CCL-{lv}"] = fmt(float(v.iloc[0])) if len(v) else "–"
        r.update({"PNA": fmt(o["PNA_strict"]), "NNA": fmt(o["NNA"]), "IPA": fmt(o["IPA_strict"]),
                  "Item F1": fmt(o["item_F1"]), "Blank price ok": fmt(o["null_price_ok"])})
        rows.append(r)
    paths += write_table(pd.DataFrame(rows), tdir / "table2_main",
                         f"Main results ({main_p}, split {split}, strict price rule, thinking off; %)", "tab:main",
                         note="CI: 10,000 image-level bootstrap resamples. PNA/NNA/IPA as defined in Section 7.3.")
    # lenient vs strict
    len_rows = [{"Model": lab(m), "Offer F1 strict": fmt(overall.loc[m, "offer_F1_strict"]),
                 "Offer F1 lenient price": fmt(overall.loc[m, "offer_F1_lenient"]),
                 "Offer F1 macro": fmt(overall.loc[m, "offer_F1_macro_strict"]),
                 "Variant F1": fmt(overall.loc[m, "variant_F1_strict"]), "SAA": fmt(overall.loc[m, "SAA_strict"]),
                 "Halluc.": fmt(overall.loc[m, "hallucination_rate"]), "Omission": fmt(overall.loc[m, "omission_rate"]),
                 "Desc→item": fmt(overall.loc[m, "desc_as_item_rate"]), "Abstain ok": fmt(overall.loc[m, "abstain_correct"]),
                 "Edits/100 offers": f"{overall.loc[m, 'edits_per_100_offers']:.0f}",
                 "Mag. err": fmt(overall.loc[m, "magnitude_err_rate"]),
                 "JSON strict": fmt(overall.loc[m, "valid_json_strict"]),
                 "JSON repaired": fmt(overall.loc[m, "valid_json_repaired"]),
                 "Trunc.": fmt(overall.loc[m, "truncation_rate"]), "Repet.": fmt(overall.loc[m, "repetition_rate"])}
                for m in models]
    paths += write_table(pd.DataFrame(len_rows), tdir / "table2b_all_metrics", f"All metrics ({main_p}, {split}; %)",
                         "tab:allmetrics")
    lenient_name = aggregate(json_df[(json_df["split"] == split) & (json_df["prompt"] == main_p)], ["model"], scfg, ci=False)
    lenient_mode = aggregate(df[(df["mode"] == "lenient") & (df["split"] == split) & (df["prompt"] == main_p)
                                & (df["thinking"] == "off")], ["model"], scfg, ci=False)
    if not lenient_mode.empty:
        nm = lenient_name.set_index("model")["offer_F1_strict"].to_frame("names with diacritics").join(
            lenient_mode.set_index("model")["offer_F1_strict"].to_frame("names without diacritics"))
        nm = nm.reset_index().rename(columns={"model": "Model"})
        nm["Model"] = nm["Model"].map(lab)
        for c in nm.columns[1:]:
            nm[c] = nm[c].map(fmt)
        paths += write_table(nm, tdir / "table2c_name_modes", "Offer F1 with strict vs lenient name normalisation (%)",
                             "tab:namemodes")

    # ---- Table 3: prompts (paired tests, Holm) --------------------------------------------------------
    pdf = json_df[(json_df["split"] == split) & (json_df["thinking"] == "off")]
    prompts = [p for p in ("P0", "P1", "P2", "P-2S", "P-NL", "P-Conf", "P1-VI", "P1a", "P1b", "P1c")
               if (pdf["prompt"] == p).any()]
    rows, pvals, keys = [], [], []
    for m in dict.fromkeys(pdf["model"]):
        r = {"Model": lab(m)}
        agg = aggregate(pdf[pdf["model"] == m], ["prompt"], scfg, ci=False).set_index("prompt")
        if not any(p in agg.index for p in prompts):
            continue
        for p in prompts:
            r[p] = fmt(agg.loc[p, "offer_F1_strict"]) if p in agg.index else "–"
        for p in prompts:
            if p in ("P0",) or "P0" not in agg.index or p not in agg.index:
                continue
            a = pdf[(pdf["model"] == m) & (pdf["prompt"] == p)].set_index(["base_image_id", "seed"])
            b = pdf[(pdf["model"] == m) & (pdf["prompt"] == "P0")].set_index(["base_image_id", "seed"])
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            ta = tuple(a.loc[common, c].values for c in ("off_tp_strict", "off_fp_strict", "off_fn_strict"))
            tb = tuple(b.loc[common, c].values for c in ("off_tp_strict", "off_fp_strict", "off_fn_strict"))
            pb = paired_bootstrap(ta, tb, int(scfg.get("bootstrap", {}).get("n", 10000)))
            pp = permutation_test(ta, tb, int(scfg.get("bootstrap", {}).get("n", 10000)))
            r[f"Δ{p}−P0"] = f"{pb['diff'] * 100:+.1f}"
            pvals.append(pp)
            keys.append((len(rows), f"p {p}"))
        rows.append(r)
    if rows:
        adj = holm(pvals) if pvals else []
        for (ri, col), p in zip(keys, adj):
            rows[ri][col + " (Holm)"] = f"{p:.3f}"
        paths += write_table(pd.DataFrame(rows).fillna("–"), tdir / "table3_prompts",
                             f"Offer F1 by prompt (split {split}; %) with paired permutation tests vs P0", "tab:prompts",
                             note="p-values: paired approximate randomisation over images (10,000 shuffles), Holm-adjusted.")

    # ---- thinking on vs off --------------------------------------------------------------------
    th = json_df[(json_df["split"] == split) & (json_df["prompt"] == main_p)]
    if (th["thinking"] == "on").any():
        agg = aggregate(th, ["model", "thinking"], scfg, ci=False)
        piv = agg.pivot(index="model", columns="thinking", values="offer_F1_strict")
        tok = agg.pivot(index="model", columns="thinking", values="completion_tokens_mean")
        lat = agg.pivot(index="model", columns="thinking", values="latency_median_s")
        tr = [{"Model": lab(m), "F1 off": fmt(piv.loc[m].get("off")), "F1 on": fmt(piv.loc[m].get("on")),
               "Tokens off": fmt(tok.loc[m].get("off"), 0, False), "Tokens on": fmt(tok.loc[m].get("on"), 0, False),
               "Latency off (s)": fmt(lat.loc[m].get("off"), 1, False), "Latency on (s)": fmt(lat.loc[m].get("on"), 1, False)}
              for m in piv.index]
        paths += write_table(pd.DataFrame(tr), tdir / "table3b_thinking", f"Thinking on vs off ({main_p}, {split})",
                             "tab:thinking")

    # ---- tag groups, tags, venue tier, non-chain ------------------------------------------------
    tg = slice_table(main, index, "tag_group", st.schema_config_path, ["model"], scfg)
    if not tg.empty:
        piv = tg.pivot(index="model", columns="tag_group", values="offer_F1_strict").reindex(models)
        piv.columns = [c.rstrip("_") for c in piv.columns]
        out_tg = piv.reset_index().rename(columns={"model": "Model"})
        out_tg["Model"] = out_tg["Model"].map(lab)
        for c in out_tg.columns[1:]:
            out_tg[c] = out_tg[c].map(fmt)
        paths += write_table(out_tg, tdir / "table4_tag_groups", f"Offer F1 by tag group ({main_p}, {split}; %)",
                             "tab:taggroups")
    tags = slice_table(main, index, "tag", st.schema_config_path, ["model"], scfg)
    if not tags.empty:
        tags.to_csv(tdir / "table4b_tags_long.csv", index=False)
        mat = tags.pivot(index="model", columns="tag", values="offer_F1_strict").reindex(models)
        paths += F.tag_heatmap(mat.rename(index=lab), fdir / "fig_tag_heatmap")
        mat.to_csv(fdir / "fig_tag_heatmap.csv")
    nc = slice_table(main, index, "non_chain", st.schema_config_path, ["model"], scfg)
    vt = slice_table(main, index, "venue_tier", st.schema_config_path, ["model"], scfg)
    if not nc.empty:
        rows = []
        for m in models:
            r = {"Model": lab(m)}
            for flag, col in ((True, "Non-chain"), (False, "Chain")):
                v = nc[(nc["model"] == m) & (nc["non_chain"] == flag)]["offer_F1_strict"]
                r[col] = fmt(float(v.iloc[0])) if len(v) else "–"
            for tier in ("street_vendor", "small_shop", "supermarket_foodcourt"):
                v = vt[(vt["model"] == m) & (vt["venue_tier"] == tier)]["offer_F1_strict"]
                r[tier] = fmt(float(v.iloc[0])) if len(v) else "–"
            rows.append(r)
        paths += write_table(pd.DataFrame(rows), tdir / "table5_venue", f"Offer F1 by venue tier ({main_p}, {split}; %)",
                             "tab:venue_results", note="Non-chain menus are less likely to appear in pre-training data.")
    for dim in ("dense", "medium", "lighting", "angle"):
        try:
            slice_table(main, index, dim, st.schema_config_path, ["model"], scfg).to_csv(tdir / f"slice_{dim}.csv", index=False)
        except KeyError:
            pass

    # ---- deployment table -------------------------------------------------------------------
    man = load_manifests(st, None if name == "all" else name)
    # runs that shared the card with another plan (gpu_share) say nothing about the speed or memory of the model alone
    solo = man[man["gpu_share"].isna()] if not man.empty and "gpu_share" in man else man
    dep = []
    for m in models:
        s = specs.get(m)
        mm = solo[solo["model_key"] == m] if not solo.empty else pd.DataFrame()
        vram = mm["vram_peak_mib"].dropna().max() if not mm.empty and "vram_peak_mib" in mm else None
        thr = mm["throughput_img_per_h"].dropna().max() if not mm.empty and "throughput_img_per_h" in mm else None
        o = overall.loc[m]
        dep.append({"Model": lab(m), "Backend": s.backend if s else "", "Quant": (s.quant or "bf16") if s else "",
                    "Download (GB)": fmt(s.download_gb, 1, False) if s and s.download_gb else "–",
                    "VRAM peak (GB)": fmt(vram / 1024, 1, False) if vram and vram == vram else "–",
                    "Images/h": fmt(thr, 0, False) if thr and thr == thr else "–",
                    "Latency med (s)": fmt(o["latency_median_s"], 1, False), "Latency p90 (s)": fmt(o["latency_p90_s"], 1, False)})
    paths += write_table(pd.DataFrame(dep), tdir / "table6_deployment",
                         "Deployment cost on one RTX A5000 24 GB (requests served concurrently by vLLM)", "tab:deploy")

    # ---- figures -------------------------------------------------------------------------------
    if not by_ccl.empty:
        paths += F.ccl_bars(by_ccl[by_ccl["model"].isin(models)].assign(model=lambda x: x["model"].map(lab)),
                            fdir / "fig_ccl_bars")
        by_ccl.to_csv(fdir / "fig_ccl_bars.csv", index=False)
    lo = sdir / "long_offers.csv.gz"
    if lo.is_file():
        long = pd.read_csv(lo)
        long = long[(long["prompt"] == main_p) & (long["split"] == split) & (long["thinking"] == "off")]
        long = long[long["error_hint"].fillna("") != ""]
        if not long.empty:
            codes = [c for c in scfg.get("error_codes", {}) if c in set(long["error_hint"])]
            counts = long.groupby(["model", "error_hint"]).size().unstack(fill_value=0).reindex(models).fillna(0)
            paths += F.error_bars(counts.rename(index=lab), fdir / "fig_error_distribution", codes)
            counts.to_csv(fdir / "fig_error_distribution.csv")
    fam_rows = []
    for m in models:
        s = specs.get(m)
        if s and s.params_b and s.family and not s.quant:
            fam_rows.append({"model": m, "family": s.family, "params_b": s.params_b,
                             "offer_F1_strict": overall.loc[m, "offer_F1_strict"]})
    fam = pd.DataFrame(fam_rows)
    if not fam.empty:
        fam = fam[fam.groupby("family")["model"].transform("count") >= 2]
        if not fam.empty:
            paths += F.size_curves(fam, fdir / "fig_size_curves")
            fam.to_csv(fdir / "fig_size_curves.csv", index=False)
    par = overall.reset_index()[["model", "offer_F1_strict", "latency_median_s"]]
    paths += F.pareto(par.assign(model=par["model"].map(lab)), fdir / "fig_pareto_latency")
    par.to_csv(fdir / "fig_pareto_latency.csv", index=False)

    # ---- extra assets (menu size, distributions, pairwise tests, capability map, OOD, robustness, ...) ------
    try:
        from smv.report.extras import build_extras

        paths += build_extras(st, name, df, specs, scfg, main_p, split, overall)
    except Exception:  # noqa: BLE001 - extras must never break the main report; the traceback is kept
        import traceback

        (out / "extras_error.txt").write_text(traceback.format_exc())

    try:                                    # truncation, compact output, decoding check, ranking stability, ...
        from smv.report.depth import build_depth

        paths += build_depth(st, name, df, specs, scfg, main_p, split)
    except Exception:  # noqa: BLE001
        import traceback

        (out / "depth_error.txt").write_text(traceback.format_exc())

    try:                                    # confidence signals for review, majority-vote ensembles
        from smv.report.trust import build_trust

        paths += build_trust(st, name, df, specs, scfg, main_p, split)
    except Exception:  # noqa: BLE001
        import traceback

        (out / "trust_error.txt").write_text(traceback.format_exc())

    try:                                    # models table of the experimental-setup section (only models with results)
        from smv.report.models_table import write_models_table

        paths.append(write_models_table(st))
    except Exception:  # noqa: BLE001
        import traceback

        (out / "models_table_error.txt").write_text(traceback.format_exc())

    # ---- registry -----------------------------------------------------------------------------
    paths.append(write_registry(st, df, man, name))
    return paths


def write_registry(st: Settings, df: pd.DataFrame, man: pd.DataFrame, name: str) -> Path:
    path = st.repo / "reports" / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    scfg = st.scoring_cfg()
    agg = aggregate(df[df["mode"] == "strict"], ["model", "prompt", "split", "seed", "thinking"], scfg, ci=False)
    specs = load_all(st.model_configs_dir)
    old = []
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r.get("plan") != name]
    rows = old
    for _, r in agg.iterrows():
        mm = man[man["model_key"] == r["model"]] if not man.empty else pd.DataFrame()
        rows.append({"plan": name, "model": r["model"], "revision": (specs[r["model"]].revision if r["model"] in specs else ""),
                     "prompt": r["prompt"], "split": r["split"], "seed": r["seed"], "thinking": r["thinking"],
                     "n_images": r["n_images"], "offer_F1_strict": round(r["offer_F1_strict"], 4),
                     "offer_F1_lenient": round(r["offer_F1_lenient"], 4), "item_F1": round(r["item_F1"], 4),
                     "manifest": mm["manifest_path"].iloc[-1] if not mm.empty else ""})
    fields = ["plan", "model", "revision", "prompt", "split", "seed", "thinking", "n_images", "offer_F1_strict",
              "offer_F1_lenient", "item_F1", "manifest"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


__all__ = ["build_report", "attach_meta"]
