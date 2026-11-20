# Error coding guide (StreetMenu-VN qualitative analysis)

Written for: two people who code the 150 cases in `coding_sheet.csv`. Work **independently** (do not discuss while coding); we measure how often you agree.

## How to code one case
1. Open the real image for `image_id` (you may look at images; the pipeline never does).
2. Compare what the model wrote (`pred_*`) with the gold (`gold_*`) and the image.
3. Put the single best theme code in `theme_coder1` (or `theme_coder2` for the second coder). Add a few words in the note when it helps or when you pick `other`.
4. One theme per case: the *main* reason the prediction is wrong. `auto_theme` is the scorer's guess — you may agree or override it.

## Themes

| Code | Meaning |
|---|---|
| `read_print` | Misread printed text or digits (the price/name is printed clearly but transcribed wrong) |
| `read_hand` | Misread handwriting (the price/name is handwritten or written over) |
| `read_scale` | Currency scale (digits right, magnitude wrong: 15 vs 15,000; k/dot/ngàn) |
| `struct_inherit` | Did not apply a shared/inherited price (section or column header, bracket, 'same price') |
| `struct_variant` | Sizes/options mishandled (missing, merged, or mapped to the wrong price) |
| `struct_spatial` | Paired a price with the wrong item (layout, column, badge, distance) |
| `name_norm` | Name or section wrong (normalisation, gloss, composed section name) though the item is right |
| `halluc` | Invented an item or price, or output a description/heading/distractor as an item |
| `omit_dense` | Missed an item or offer on a dense menu (not because the answer was cut) |
| `omit_trunc` | Missed because the answer was cut off (truncation/repetition/invalid JSON) |
| `other` | None of the above (write a note) |

## Notes
- Judge the *cause*, not the symptom: a missing price caused by a cut-off answer is `omit_trunc`, not `omit_dense` (`truncated` is true in the sheet).
- `name_norm` is only when the item and price are essentially right but the name/section is off.
- If two causes apply, pick the one a reader would fix first.
- `gold_price`/`pred_price` are integers in đồng (null shown empty).
