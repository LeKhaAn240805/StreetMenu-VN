# StreetMenu-VN

StreetMenu-VN is a dataset and evaluation protocol for offer-level menu extraction from photographs of Vietnamese
street-food menus. It accompanies the paper *StreetMenu-VN: A Dataset and Evaluation Protocol for Offer-Level Menu
Extraction with Local Vision-Language Models* (TAIMMA 2027, under review).

The release contains 346 redacted in-situ photographs with 5,244 items and 6,245 offers, the gold labels, the scorer
and evaluation pipeline, the scored model outputs, the blind re-annotation, the error-coding materials and the scripts
that produce every table and figure of the paper.

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | The dataset: `images/` (346 redacted photographs), `adjudicated/` (one gold label file per image), `gold.jsonl` (the same labels, one image per line), `splits.csv`, `schema/` (label schema v3.1), `stats.json`, `excluded.csv` (3 removed images), `final_changes.csv` and `tag_changes.csv` (edits made during adjudication), `audit/` (audit samples), `SHA256SUMS.txt` and `scripts/verify_data.py` |
| `code/` | The evaluation pipeline `smv`: data ingestion, model runners (vLLM, transformers, OCR baselines), prompts, the offer-level scorer with rules R1 to R8, statistics and report generation |
| `results/` | Scored outputs of every run (`scores/summary.csv`, `scores/per_image.parquet`, `scores/long_offers.csv.gz` with every gold and predicted offer, `scores/score_meta.json`) and the generated report tables and figures (`reports/`) |
| `annotation/blind_audit/` | Blind re-annotation of 30 test images by two annotators (A and B): raw labels, agreement results and scripts |
| `annotation/error_coding/` | Mixed-methods error analysis: the 150-case coding sheet, the codebook, the codes of each coder (C1 to C4), the merged and adjudicated codes and the merge script |
| `tools/labeler/` | The browser-based labeling tool used to create and review the labels (open `index.html`) |
| `paper/` | Scripts and inputs that regenerate every number, table and figure of the paper (`paper/scripts/build_all.py`) |

## Data format

Each file in `data/adjudicated/` describes one image. An image holds a list of items and each item holds one or more
offers. An offer is one price for one item variant (size, temperature or option) and records the price in VND, its
written form, its kind (fixed, range, market price and others) and its provenance (own line, header, merged cell,
bracket or handwritten override). Phenomenon tags and the contextual complexity level (CCL) are stored per image. The
full field list is in `data/schema/schema_config_v3.1.json` and `data/schema/annotation_schema.json`.

Splits are group-aware (images of the same venue stay in one split, seed 2027): dev 39 images, test 274 images and
out-of-distribution (ood) 33 images.

Check the data folder with

```bash
cd data && python scripts/verify_data.py .
```

## Evaluation

The pipeline needs Python 3.10 or later and, for model runs, one GPU with 24 GB.

```bash
cd code
cp .env.example .env          # paths default to ../data and folders inside code/
pip install -e .
make ingest                   # validate labels and images, build the split index
make status PLAN=configs/plans/c1_main.yaml
scripts/run_plan.sh configs/plans/c1_main.yaml   # download, serve and query each model in turn
make score report PLAN=configs/plans/c1_main.yaml
```

The raw model responses are not in the repository because of their size. The scored results in `results/scores/` are
enough to rebuild every table: `long_offers.csv.gz` lists each gold and predicted offer with its match.

Run plans are in `code/configs/plans/`, model configurations in `code/configs/models/`, frozen prompts in
`code/prompts/` (`FROZEN.json` holds their hashes) and the scoring rules in `code/configs/scoring.yaml` and
`code/docs/EVALUATION_PROTOCOL.md`. The main setting of the paper is the test split, prompt P1, thinking off, strict
name matching and seed 0.

## Reproducing the paper

```bash
pip install matplotlib numpy pillow fonttools rapidfuzz pandas pyarrow
python paper/scripts/build_all.py --no-pdf
```

The scripts read `results/` and their own inputs in `paper/data/` and write the number macros and tables to
`paper/tables/` and the figures to `paper/figs/`. Fig. 1 is cropped from `data/images/`.

## Privacy and ethics

Faces, phone numbers and other personal data in the photographs were covered with solid boxes before release. An image
showing a child was removed. Annotators and coders are identified only by pseudonyms (A, B and C1 to C4). If you find
personal data that was missed, please open an issue and the image will be redacted or removed.

## License

The code (`code/`, `paper/scripts/`, `tools/`, `annotation/*/scripts/`, `data/scripts/`) is released under the MIT
License (`LICENSE`). The images, labels, scores and annotation data are released under the Creative Commons Attribution
4.0 International License (`LICENSE-DATA.md`).

## Citation

The citation will be added after the review.
