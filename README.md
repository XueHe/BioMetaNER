# BioMetaNER

**Taming Label Noise in Metagenomics Named Entity Recognition**

Submission preparation repository for schema-first transfer from schema-aligned
Silver annotations to heterogeneous Bronze supervision. The primary system uses
four independent BIO token classifiers for MIC, DIS, CHE and HAB. E6 first
fine-tunes on Silver, then adapts only the classifier on Bronze.

> Private preparation snapshot, not a completed public data release. Training
> data, original abstracts, annotations, raw predictions, model weights, API
> credentials and reviewer correspondence are intentionally excluded. See
> [release status](docs/RELEASE_STATUS.md) before making the repository public.

## Audited Results

Golden v2: 41 abstracts, 1,725 entities. Partial matching is type-aware,
prediction-order greedy one-to-one character overlap, not maximum matching.
Scores below are equal-weight four-type Macro F1 (%), averaged across available
training seeds; the dispersion is sample SD, not a bootstrap CI.

| Configuration | PubMedBERT Macro F1 |
| --- | ---: |
| E6: Silver full FT, then Bronze head-only adaptation | 88.59 +/- 0.13 |
| Custom EM label fusion under the E6 schedule | 79.17 +/- 2.94 |
| Noise-masked Bronze loss, seed 42 | 74.37 |
| Silver-disagreement filtering, seed 42 | 67.56 |

The E1-E9 count tables cover five backbones and seeds 42, 1 and 2. Recomputed
strict-span results and 2,000-draw paired abstract bootstrap summaries are also
included. See [results and caveats](docs/RESULTS.md). Missing or incompatible
baselines are not silently filled with manuscript values.

## Quick Start: No GPU Required

Python 3.10+ is recommended. Evaluation and count aggregation use only the
standard library; training and plotting dependencies are optional.

```sh
python -m unittest discover -s tests -v
python scripts/evaluate.py --golden examples/gold.jsonl \
  --predictions examples/predictions.jsonl --allow-unpinned \
  --output outputs/example_evaluation.json
python scripts/summarize.py --counts results/figure4/per_seed_per_type.csv \
  --output outputs/summary_recomputed.csv
```

For the real Golden file, omit `--allow-unpinned`. Its exact SHA-256 is checked,
along with document IDs, text identity and the 41/1,725 corpus dimensions.
See [data layout and access](docs/DATA.md).

Local checks have run. GitHub CI is currently an [inactive template](docs/ci/README.md)
because the login lacks workflow-upload permission; this is not a claim of passing hosted CI.

## Training and Inference

```sh
python -m pip install -r requirements-training.txt
python scripts/train.py --model pubmedbert --experiment E6 --seed 42 \
  --data-root /path/to/private/data --model-paths /path/to/models.local.json \
  --output-root outputs/training
python scripts/predict.py --experiment-dir outputs/training/E6_pubmedbert_silver_full_bronze_lastlayer_seed42 \
  --golden /path/to/private/data/golden_metasigndb_v2_for_eval.jsonl \
  --output outputs/predictions_golden_v2.jsonl
python scripts/evaluate.py --golden /path/to/private/data/golden_metasigndb_v2_for_eval.jsonl \
  --predictions outputs/predictions_golden_v2.jsonl --output outputs/paper_evaluation.json
```

Model files must already be cached or supplied as local directories. The
training entry point records input hashes for new runs. The original trainer's
automatic evaluation is diagnostic: use `predict.py` and `evaluate.py` for the
paper protocol. The paper inference path retains raw-logit span ranking; the
trainer's diagnostic path uses softmax-based span ranking. These are not assumed
interchangeable. No new training was performed while preparing this repository.

## Reproduce Analyses

The pinned analysis dependencies require Python 3.11+.

```sh
python -m pip install -r requirements-analysis.txt
python scripts/plot_figure4.py --summary results/figure4/summary.csv --output outputs/figure4
python scripts/bootstrap.py --counts results/bootstrap/document_counts.json.gz \
  --output outputs/bootstrap --n-resamples 2000
python scripts/check_release.py
```

The bootstrap input contains per-document integer counts, not text or spans.
Document indices are paired across all configurations and fixed training seeds.
See [evaluation protocol](docs/EVALUATION.md) and [experiment definitions](docs/EXPERIMENTS.md).

## Contents

- `biometaner/`: canonical matching, evaluation, aggregation and inference helpers.
- `training/`: archived training implementations with portable path configuration.
- `scripts/`: explicit training, inference, evaluation, statistics and figure entry points.
- `results/`: count-derived results, uncertainty estimates and verification metadata.
- `prompts/`: supplied abstract-level NER prompt templates, not a claim that every LLM run used them.
- `examples/`: synthetic examples only.
- `docs/`: data requirements, limitations, provenance and release checklist.

Related private dictionary project: [XueHe/MITC](https://github.com/XueHe/MITC).
This repository is separate from that dictionary/API implementation.

## License and Citation

A redistribution license has not yet been selected with all contributors.
Private hosting does not grant permission to redistribute upstream corpora or
models. See [license status](LICENSE_STATUS.md). Citation metadata is provisional
until the manuscript author list and publication identifier are finalized.
