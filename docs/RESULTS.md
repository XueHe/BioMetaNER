# Results and Qualification

`results/figure4/per_seed_per_type.csv` has 540 integer-count rows: 9 experiments,
5 backbones, 3 seeds and 4 entity types. `summary.csv` contains 225 type/Macro
summaries. These partial results were verified against the pinned Golden v2
texts, gold spans and saved predictions before export.

`results/bootstrap/per_seed_counts.csv` additionally includes strict counts.
`summary_ci.csv` and `paired_differences_ci.csv` retain unrounded fractions.
Per-document bootstrap inputs contain integer counts and pseudonymous document
indices only; the same index identifies the same abstract across all runs.

`results/baselines/` distinguishes verified evaluation records from qualified
or unresolved manuscript rows. Important corrections:

Golden was also used to compare candidate Silver annotators. It must not be
described as strictly held out from annotator selection. Historical prompt
development and model-selection provenance are incomplete; this repository does
not establish their independence from Golden.

- Custom EM label fusion under E6: three-seed Macro 79.17 +/- 2.94, not the
  manuscript's single selected seed. Seed 1 is 79.42 when recomputed from counts.
- Noise-masked loss: MIC/DIS/CHE/HAB = 92.14/85.00/81.11/39.25; Macro 74.37,
  seed 42 only.
- The old Training-Dynamics label is incorrect: DC20 uses Silver disagreement,
  not epoch dynamics. Its F1s are 95.01/79.51/78.59/17.12; Macro 67.56. Threshold
  ties remove 67.41% of HAB records, not the nominal 20%. This is not a repaired
  filter experiment.
- GLiNER zero-shot can be re-scored on v2: Macro 72.83. Its original texts differ
  only by index-preserving whitespace. This is NOT the unverified fine-tuned
  GLiNER 65.68 row; fine-tuning output was not located.
- MINT Dictionary 69.20 was not traced to original Golden v2 predictions/counts.
- DynClean's raw NMS counts total 1720 gold entities, not 1725. Its count-derived
  Macro 51.58 cannot be treated as the same-corpus main comparison.
- The LLM/partial integer CSV supports the claimed Claude Macro 90.77 (not micro
  92.37), but the raw annotation inputs/model metadata are unavailable. Its
  associated evaluation/postprocessing source differs and includes expert-assisted
  propagation. Same-matcher independent-LLM equivalence has NOT been established.

`results/joint/` retains count-derived diagnostic results. Joint E3 Silver-only
Macro is 88.77 +/- 0.38; E6 stage1 is 88.56 +/- 0.15; stage2 is 68.36 +/- 0.16.
For each of the three E6 stage2 seeds, HAB TP=0, FP=0, FN=202. Differences in
supervision, hyperparameters and selection prevent an architecture-only causal
claim. The original 88.4 stage1 number is not confirmed by retained checkpoints.

## What Has Not Been Reproduced Here

This preparation did not retrain any model or regenerate predictions. Count
aggregation and matching utilities have automated tests. Full fresh-machine
GPU retraining, access-controlled data delivery and a public model release are
pending. Historical training source/data hashes are incomplete for some runs.
The observed dependency versions are not asserted to be an exhaustive lockfile
for every historical experiment.
