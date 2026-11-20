# StreetMenu-VN: evaluation protocol (v1.0 labels, scorer of 14 Nov 2026)

Written for: the authors drafting the Evaluation and Experimental Setup sections, and for reviewers who need to
reproduce the numbers. Everything here is implemented in `src/smv/` and driven by `configs/`.

## 1. Task and output format

Input: one photo of a Vietnamese menu, price board or price list. Output: one JSON object
`{"items": [{"section", "raw_name", "name", "description", "is_topping", "is_combo",
"offers": [{"size", "temp", "portion", "raw_price", "price_vnd", "price_max_vnd", "old_price_vnd", "unit"}]}]}`
(`ref/model_output_schema_v1.1.json`). `raw_*` fields copy the text; `name` is the standard Vietnamese name with
diacritics; `price_vnd` is an integer number of dong, `null` when the menu shows no readable number. One *offer* is one
purchasable option of an item (a size, hot/iced, a portion, a quantity tier, a price in an unlabeled row).

## 2. Data and splits

346 redacted photos, one gold JSON each (`adjudicated/`, schema v3.1). Splits are made by whole venue groups
(`group_id`, seed 2027), so no group appears in two splits: **dev** 39 images (590 items, 788 offers), **test** 274
(4,345 items, 5,081 offers), **ood** 33 (309 items, 376 offers; service and retail price lists that are not food menus).
CCL-1..4 is the context complexity level of the image, computed from tags by `ccl_rule` of the schema (1 plain line
prices, 2 names need composing, 3 prices or options must be inferred, 4 layout, handwriting or damage).
Every label began as an AI draft that two annotators reviewed and a rule pass adjudicated; the provenance table
(`reports/dataset/tables/table1l_provenance`) reports the edit rate per CCL. Blind double annotation of 30 test images is
a separate agreement study.

## 3. Label conventions the scorer follows

| Rule | Convention | Effect on scoring |
|---|---|---|
| R1 | "A-B" without a size is one `range` offer, `price_vnd = min`, `price_max_vnd = max` | strict: price = min (and max if the model gives one); lenient: either end, or two predicted offers priced exactly {min, max} |
| R2 | "+X" on an item with a clear base price is the absolute price (`fixed`, raw "+X"); `delta` only for add-on lines | exact price; lenient also accepts base + delta for gold deltas |
| R3 | prices in a row with no label are tiers `#1, #2, #3` | paired by order or by price, never ambiguous |
| R4 | not in the main precision/recall: offers flagged ambiguous, items whose `item_status` is not `active`, items with `legibility = illegible` | a prediction that matches such an item is neither right nor wrong (ignored) |
| R5 | a handwritten or pasted price over a printed one is `evidence = overwritten`, `price_vnd` is the current price | exact price |
| R6 | "tùy loại", "thời giá" and similar: `price_kind = market`, `price_vnd = null`, not ambiguous | blank is correct; any number is an invented price |
| R7 | an attribute (size, temperature) left null by the model matches every gold value | attributes steer the pairing only, correctness is the price |
| R8 | an item repeated verbatim in one image (same name, same prices) | counted once, in gold and in the prediction |

## 4. Scoring

1. **Parsing.** strict JSON, then lenient (code fences removed, outer braces cut, `json-repair`), else failed. Reasoning
   text before the answer is removed. Truncation (`finish_reason = length`) and repetition (one item more than five times)
   are flagged and reported, never silently repaired away.
2. **Item matching.** Names are NFC-lowercased with numbering and trailing punctuation removed and a small lexicon applied
   ("cf" = "cà phê"). Similarity is 1 for equality else the normalised Indel ratio (RapidFuzz), taken as the maximum over
   the predicted `name` and `raw_name` against the gold `norm_name`, `alt_norm_names`, `raw_name` and `raw_name_2`.
   The Hungarian algorithm assigns predicted to gold items; a pair counts when similarity >= tau (tau = 0.70, chosen on dev from
   the price agreement of matched pairs, see section 6; names keep their diacritics in the main score, a second score drops
   them). A parenthesised gloss ("Rau (vegetables)") is also tried removed, at a weight just below an exact match.
3. **Unmatched predictions** are classified: ignored (they match an inactive or illegible gold item, or an ignored region),
   description written as an item, combo component written as an item, otherwise hallucinated item (false positive).
4. **Offer matching** inside a matched item pair: Hungarian on attribute agreement (size, rank, temperature, portion,
   quantity) plus price agreement, with tier order for unlabeled prices.
5. **Price rules** are applied in two versions, strict (main) and lenient, as in the table above.

## 5. Metrics

* **Offer F1** (main): micro over all offers of the split. A true positive is a matched offer with the right price. A
  false positive is a predicted priced offer that is wrong, extra or inside a hallucinated item; a false negative is a
  scored gold offer that is missing or wrong. Gold offers without a price are scored: leaving the price blank is a true
  positive, a number is a false positive and a false negative. Macro F1 over images is also reported.
* **Item F1** over active gold items; hallucination rate (false-positive items / predicted items), omission rate
  (false-negative items / gold items), description-as-item rate.
* **PNA** price normalisation accuracy: matched priced offers with exactly the right price. **NNA** name normalisation
  accuracy: matched items whose predicted `name` equals the gold standard name. **IPA** inherited-price accuracy: offers
  whose price is written in a header, column, bracket or merged cell. **SAA** spatial association accuracy: offers on
  images tagged detached or free layout, or whose price sits in a badge. **Variant F1** over items with two or more offers.
* **Blank-price accuracy**: share of gold offers with no price on the menu that the model leaves blank.
* **Magnitude errors**: wrong prices with |log10(predicted / gold)| >= 0.9 (a currency-scale mistake such as "15" for 15,000).
* **Format health**: valid JSON (strict, after repair), truncation rate, repetition rate; tokens, latency, peak VRAM, images/hour.
* **Perception** (prompt P-T): recall of gold `raw_price` strings and of `raw_name` tokens in a transcript.

Statistics: percentile bootstrap over images (10,000 resamples) for micro F1 and ratios; paired bootstrap and paired
approximate randomisation for model or prompt differences, Holm-adjusted; chain and non-chain venues are reported
separately because chain menus may be in pre-training data. Ranking stability is checked against the lenient price rule,
names without diacritics, macro averaging, tau 0.60/0.80 and counting repeated items separately.

## 6. Discipline

Choice of tau. Item F1 rises monotonically as tau falls, so maximising it would reward looseness. Instead every predicted item
of the dev runs was paired with a gold item at similarity >= 0.5, the pairs were binned by name similarity, and per bin the
share of pairs whose price sets intersect was computed (a wrong pairing rarely shares a price; chance level 20%, measured
on random gold items of the same image). Agreement is 80-95% for every bin from 0.60 upward and 59% for 0.50-0.60, so
tau = 0.70 was set inside the plateau (`outputs/scores/tau_evidence.csv`, `reports/tune/`).

All choices (prompt wording, tau, generation parameters) were fixed on dev. The runner refuses to run test, ood or the
robustness split for a prompt whose SHA-256 is not in `prompts/FROZEN.json`. Decoding is greedy (temperature 0,
max 8,192 new tokens, seed 0, the model's own generation config ignored), one request per image, the same system prompt
for every model. Images are decoded once (EXIF rotation, RGB, longest side at most 2,048 px, JPEG q95) and shared by all
models; each model keeps its own processor limits. Every run writes `outputs/<plan>/<model>/manifests/*.json` with the
model revision, library versions, prompt hashes, GPU, seeds, wall time and peak VRAM.
