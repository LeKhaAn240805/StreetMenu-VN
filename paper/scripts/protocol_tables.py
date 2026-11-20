"""Tables that come from documented facts rather than from the label files:
(1) comparison with related benchmarks (numbers taken from each paper, see comments) and
(2) the scoring conventions R1-R8 of the released scorer (StreetMenuVN_final/README.md)."""
from common import TABS

# ----------------------------------------------------------------------------- related benchmarks
# Every number below is copied from the cited paper (checked against the PDF text):
#   SROIE: "1000 whole scanned receipt images"; FUNSD: "199 real, fully annotated, scanned forms";
#   MC-OCR: "2,436 Vietnamese receipts" captured with mobile devices; ReceiptBench: "10,656 high-quality images";
#   CC-OCR: "39 subsets with 7,058 full annotated images"; OmniDocBench: 981 PDF pages of 9 document types;
#   MTVQA: 2,116 test images, 9 languages incl. Vietnamese; ViTextVQA: "over 16,000 images";
#   MOTBench: 65 Chinese + 65 English menu images, 2,328 + 1,467 dish items.
RELATED = [
    # name, cite key, language, domain, images, target
    ("SROIE", "Huang2021SROIE", "en", "scanned receipts", "1,000", "4 key fields"),
    ("FUNSD", "Jaume2019FUNSD", "en", "scanned forms", "199", "entities and links"),
    ("MC-OCR", "vu2021mcocr", "vi", "phone-captured receipts", "2,436", "4 key fields"),
    ("ReceiptBench", "Wang2026ReceiptBench", "multi", "receipts, invoices", "10,656", "fields, line items"),
    ("CC-OCR", "Yang2024CCOCR", "multi", "text-rich images", "7,058", "OCR, parsing, KIE"),
    ("OmniDocBench", "Ouyang2024OmniDocBench", "zh, en", "PDF pages", "981", "page parsing"),
    ("MTVQA", "tang-etal-2025-mtvqa", "9 lang.", "scene-text images", "2,116", "QA pairs"),
    ("ViTextVQA", "Nguyen2024ViTextVQA", "vi", "scene-text images", "$>$16,000", "QA pairs"),
    ("MOTBench", "Wu2025MOTBench", "zh, en", "web menus", "130", "dishes, prices, translation"),
]
L = ["\\begin{tabular}{@{}lllrl@{}}", "\\toprule",
     "Dataset & Lang. & Domain & Images & Extraction target \\\\", "\\midrule"]
for name, key, lang, dom, n, tgt in RELATED:
    L.append(f"{name}~\\cite{{{key}}} & {lang} & {dom} & {n} & {tgt} \\\\")
L += ["\\midrule",
      "\\textbf{StreetMenu-VN} & vi (+en) & in-situ street menus & \\nImagesAll & offers, price provenance, tags \\\\",
      "\\bottomrule", "\\end{tabular}"]
(TABS / "tab_related.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

# ----------------------------------------------------------------------------- scoring conventions
RULES = [
    ("R1", "Price range", "An unsized ``A--B'' is one \\texttt{range} offer priced A. The lenient score accepts "
                          "either end or two offers."),
    ("R2", "Surcharge", "``+X'' after an explicit base price is coded as the absolute price. The \\texttt{delta} "
                        "kind is kept for add-ons and unknown bases."),
    ("R3", "Unlabeled sizes", "A bare price sequence is read as sizes \\#1, \\#2, \\#3 from left to right."),
    ("R4", "Exclusions", "Ambiguous offers and crossed-out, taped-over or illegible items are excluded. Predictions "
                         "matched to them are neutral."),
    ("R5", "Override", "An overwritten or pasted-over price is scored at its current value."),
    ("R6", "Market price", "Market-price lines have no price: only abstaining is correct."),
    ("R7", "Null attribute", "A null size or temperature in the prediction matches any gold value."),
    ("R8", "Repeated board", "Identical items on two boards of one stall are merged before matching."),
]
L = ["\\begin{tabular}{@{}llp{9.0cm}@{}}", "\\toprule", "Rule & Case & Convention \\\\", "\\midrule"]
for rid, case, text in RULES:
    L.append(f"{rid} & {case} & {text} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(TABS / "tab_rules.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

print("tables: tab_related, tab_rules")
