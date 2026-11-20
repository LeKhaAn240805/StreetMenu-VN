#!/usr/bin/env python3
"""Write reports/PAPER_ASSETS.md: every generated table and figure, where it fits in the paper and what claim it supports.

    python scripts/paper_assets_index.py            # scans reports/<plan>/ for every plan that has results

Only files that exist are listed. Descriptions are written here, numbers are never copied (cite the CSV twin)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS: dict[str, tuple[str, str, str]] = {
    # name: (section, what it shows, claim it can support)
    "table1_dataset": ("Dataset", "images, items, offers, dense menus, chain share per split", "size and composition"),
    "table1b_ccl": ("Dataset", "images per context-complexity level", "difficulty is skewed toward CCL-4"),
    "table1c_tags": ("Dataset", "prevalence of the 22 phenomenon tags", "which phenomena the benchmark covers"),
    "table1d_venue_tier": ("Dataset", "venue tiers", "street vendors and small shops dominate"),
    "table1f_price_kind": ("Dataset", "offers by price kind", "fixed, unit, range, delta, bundle, market"),
    "table1g_price_source": ("Dataset", "where prices are written", "a third of prices are not on the item line"),
    "table1h_evidence": ("Dataset", "printed, handwritten, overwritten prices", "handwriting is common"),
    "table1i_ood_domains": ("Dataset", "out-of-domain split composition", "service and retail price lists"),
    "table1j_density": ("Dataset", "menu size by CCL", "dense menus exist at every level"),
    "table1k_scoring_set": ("Evaluation", "offers entering the main precision/recall", "R4/R6 definition, counts"),
    "table1l_provenance": ("Dataset / Limitations", "AI draft, human review, adjudication", "label provenance and edit rate"),
    "fig_dataset_overview": ("Dataset", "CCL, tags, menu size, price source in one figure", "overview figure"),
    "fig_tag_cooccurrence": ("Dataset", "which phenomena co-occur", "tags are entangled, not independent"),
    "fig_price_magnitude": ("Dataset", "price scale by venue tier", "why currency-scale errors are costly"),
    "table2_main": ("Results", "main table: Offer F1 with CI, by CCL, PNA, NNA, IPA, Item F1", "model ranking and difficulty gradient"),
    "table2b_all_metrics": ("Results / Appendix", "every metric per model", "error profile beyond F1"),
    "table2c_name_modes": ("Appendix", "names with and without diacritics", "diacritics are not what separates models"),
    "table3_prompts": ("Results", "P0, P1, P2 and other prompts with paired tests", "instructions help or do not"),
    "table3b_thinking": ("Results", "thinking on versus off, tokens and latency", "does reasoning pay for its cost"),
    "table4_tag_groups": ("Results", "Offer F1 by tag group", "which phenomenon family hurts"),
    "fig_tag_heatmap": ("Results", "model by tag heatmap", "fine-grained weaknesses"),
    "table5_venue": ("Results", "chain versus non-chain and venue tiers", "contamination check"),
    "table6_deployment": ("Results", "download, VRAM, images per hour, latency", "deployment cost on one 24 GB GPU"),
    "fig_ccl_bars": ("Results", "Offer F1 by CCL per model", "difficulty gradient"),
    "fig_error_distribution": ("Analysis", "share of suggested error codes E0-E8", "what kind of mistakes dominate"),
    "fig_size_curves": ("Results", "older scale figure (one line per family)", "superseded by fig_scale_curves"),
    "fig_scale_curves": ("Results", "Offer F1 against parameters for each family", "does scale help inside a family"),
    "fig_pareto_latency": ("Results", "accuracy against latency with the Pareto front", "cost-accuracy trade-off"),
    "fig_f1_by_menu_size": ("Analysis", "F1 and truncation by number of offers", "dense menus break output budgets"),
    "fig_f1_ecdf": ("Analysis", "distribution of per-image F1", "means hide complete failures"),
    "fig_pairwise_delta": ("Results / Appendix", "pairwise differences with Holm-adjusted tests", "which gaps are significant"),
    "table_capabilities": ("Analysis", "perception, structure, normalization, reasoning side by side", "capability map of schema v3.1"),
    "table_ood": ("Results", "menus versus service and retail price lists", "domain shift"),
    "fig_robustness": ("Analysis", "F1 change under rotation, blur, JPEG, half resolution, glare", "robustness"),
    "table_cascade": ("Results", "image to JSON versus OCR text to JSON versus human transcript", "perception versus structuring"),
    "table_ranking_sensitivity": ("Appendix", "ranking stability under scoring choices", "conclusions do not hinge on the metric"),
    "table_error_examples": ("Analysis", "text-only examples per error code", "qualitative error analysis"),
    "fig_reliability": ("Analysis", "stated confidence against accuracy", "calibration of price confidence"),
    "fig_budget": ("Analysis", "Offer F1 against visual tokens at several image budgets", "resolution limits reading"),
}


def main() -> None:
    lines = ["# Paper assets", "",
             "Generated by `scripts/paper_assets_index.py`. Each asset has a `.csv` twin with the numbers; LaTeX tables are "
             "`.tex`, figures are `.pdf` (vector) and `.png`. No figure shows a menu photo.", "",
             "| Asset | Where | Shows | Supports | Files |", "|---|---|---|---|---|"]
    seen = False
    for d in sorted((REPO / "reports").iterdir()):
        if not d.is_dir():
            continue
        for sub in ("tables", "figures"):
            for f in sorted((d / sub).glob("*")) if (d / sub).is_dir() else []:
                stem = f.stem
                if stem not in ASSETS or f.suffix not in (".tex", ".pdf"):
                    continue
                sec, what, claim = ASSETS[stem]
                twin = ", ".join(sorted(x.suffix[1:] for x in (d / sub).glob(stem + ".*")))
                lines.append(f"| `{d.name}/{sub}/{stem}` | {sec} | {what} | {claim} | {twin} |")
                seen = True
    if not seen:
        lines.append("| (nothing generated yet) | | | | |")
    out = REPO / "reports" / "PAPER_ASSETS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
