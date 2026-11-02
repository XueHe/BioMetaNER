# Experiment Definitions

FT updates the full encoder and classifier. LP/lastlayer updates the classifier
only; it is not the final Transformer block. Four independent models are trained
for MIC/DIS/CHE/HAB. The primary E-series uses linear three-label BIO heads.

| Experiment | Stage 1 | Stage 2 |
| --- | --- | --- |
| E1 | Bronze LP | - |
| E2 | Bronze FT | - |
| E3 | Silver FT | - |
| E4 | Bronze FT | Silver FT |
| E5 | Bronze FT | Silver LP |
| E6 | Silver FT | Bronze LP |
| E7 | Silver LP | - |
| E8 | Silver FT | Bronze FT |
| E9 | Bronze LP | Silver FT |

Five backbones: BERT-base, BioBERT, SciBERT, PubMedBERT, BioMegatron.
Available audited seeds: 42, 1, 2 for all 45 backbone/configuration pairs.
Final-stage outputs are used; E9 is not a second name for E1 or E4.

E6/E9 keep source-specific update scope fixed and invert order. Their actual
stage budgets, learning rates, stopping rules and optimization histories remain
source/phase-specific. They are not claimed to be identical schedules except
for one swapped line of code. E6/E8 compare frozen versus unrestricted Bronze
adaptation after Silver. See `ENTITY_TUNING`, `STAGE_TUNING`, `cfg_for_stage`
in the archived binary training source for actual settings.

In full-FT stages, HAB B/I may use weight 3, and label smoothing decays from
0.1 to zero. In head-only stages, HAB class weighting and label smoothing are
disabled. Do not describe [1,3,3] as identical across every stage.

## Joint Diagnostic

The optional Joint baseline shares an encoder and a 9-label BIO head. Unobserved
types in source annotations are treated as O, not a partial-label loss. The
archived implementation also merges Bronze text by `text[:120]`; the audit found
a prefix collision. Learning-rate budgets, dev selection, data exposure and
class weighting differ from the binary pipeline. Its HAB collapse is a result
of that implementation/configuration, not proof that all joint architectures
must fail. Historical behavior is documented, not silently repaired here.

## Training Provenance

Portability edits change paths, not model/loss/schedule functions. The new
`scripts/train.py` saves hashes for the actual inputs of any future run. This
does not retroactively provide old hashes. Some historical Binary E3 seed
configurations differ; do not treat their aggregate as a perfectly controlled
architecture-only comparison.
