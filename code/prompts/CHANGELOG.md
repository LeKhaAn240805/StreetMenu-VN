# Prompt changelog

## v1.4 (16/11/2026): P1-C added (compact output); P1a, P1b, P1c final; nothing frozen was changed
- P1a, P1b and P1c (three rewordings of P1: rules in another order, terse bullets, one paragraph) were compared with P1 rule
  by rule on 15/11/2026, set to `final`, checked on dev (39/39 parseable on the three tuning models) and frozen.
- P1-VI frozen on 16/11/2026 after the same dev format check.
- **P1-C**: the context and the ten rules of P1, word for word; only the last paragraph (how to write the answer) differs:
  one line, no indentation, keys with a null or false value left out (every item keeps `raw_name`, `name` and at least one
  offer, every offer keeps `price_vnd`), schema shown minified. Reason, found in the main run (P1, test): the indented
  JSON with every key spelled out costs about 140 output tokens per offer, so every menu with more than about 60 offers
  stops at the 8,192-token output limit; for the best models 45-57% of all missed offers sit in those truncated answers.
  P1-C asks the same question with a shorter answer, to separate "cannot read the menu" from "ran out of output".
  It was written once, after the main run and before any P1-C result on test existed; the freeze gate
  (`scripts/freeze_if_sane.py --compact P1-C:P1:0.75`) looks at format on dev only: parseable answers, no offers lost
  against P1, fewer output tokens per offer. The main table stays P1; P1-C is reported next to it.
- P-Conf (P1 plus a stated confidence per offer, text unchanged since v1.2) goes through the same dev gate
  (`--need-confidence P-Conf`) before its test run.

## v1.3 (14/11/2026), FROZEN 14/11/2026 for P0, P1, P2, P-T, P-2S (SHA-256 in prompts/FROZEN.json)
Chosen on dev only (39 images). P0 and P-T are unchanged.
- Rule 1 (P1 and every prompt that reuses P1 rules): "A heading that groups several items (for example "SODA: 22K"
  above a list of flavours) is not an item: list each flavour or dish under it as its own item." Reason: on a dev menu
  with shared-price section headings the model listed the seven headings as items and lost 95 offers.
- Dev evidence: Gemma 4 E4B P1 beats P0 by 12 Offer F1 points (the header menu is enumerated again); Qwen3-VL-8B P1 is
  on par with P0 on Offer F1 and far better on names (NNA 0.76 against 0.49). P1-VI, P-OCR+, P-NL, P-Conf and P-Oracle stay
  unfrozen until their own dev check.
- Rule 2: "If a translation in another language is printed next to the name, leave it out of raw_name and name."
  Reason: 36% of the images are bilingual (tag N_FOREIGN); the gold keeps the gloss in raw_name_2, so a gloss folded into
  the name is an error of the name, not a different item (the scorer also ignores parenthesised glosses when matching).

## v1.2 (14/11/2026), label conventions of the final labels (README of StreetMenuVN_final, rules R1-R8)
- Rule 4: "12k5" means 12500, "1tr9" means 1900000 and "Free" means 0.
- Rule 6: prices in a row with no label ("20k 25k 30k") are one offer each in reading order with size null (R3); a
  surcharge such as "+5k" for a bigger size on an item with a clear base price makes the price of that option the full
  price (base + 5000, raw_price "+5k"), and only a separate add-on or topping line keeps the increment (R2).
- Rule 7: a handwritten or pasted-over price replaces the printed one even when it is higher (R5).
- Rule 10: a word instead of a number ("tùy loại", "thời giá", "liên hệ") or a missing price gives price_vnd null with the
  words in raw_price (R6).
- P1-VI, P1b and P1c carry the same additions.

## v1.1 (02/11/2026), not frozen
Source: ref/prompts_v1.md (Word sections 6.2-6.4), built by scripts/build_prompts.py.
- {SCHEMA}: added `"price_max_vnd": null` to offers (spec A, DE_XUAT B1). JSON indented with 2 spaces
  (the Word export had lost the indentation); keys and value descriptions unchanged.
- P1 rule 8 (and every prompt that reuses P1 rules: P2, P-Conf, P-2S, P-OCR+, P-Oracle, P-NL, P1-VI):
  added "If a price is a range such as 45k-55k, put the lower bound in price_vnd and the upper bound in
  price_max_vnd." (DE_XUAT B1). P1-VI has the Vietnamese equivalent.
- P-NL split into two prompts: P-NL.s1 (stage 1, image, free list) and P-NL (stage 2, text only).
- P1a/P1b/P1c: first drafts written from the descriptions in 6.4; status draft until a second person
  checks that they are equivalent to P1. P-Adv not written (optional).
- Every prompt uses the shared system prompt (EN; VI for P0-VI and P1-VI), as stated in 6.4.
