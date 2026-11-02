# Training Implementations

The binary trainer preserves the experimental E1-E9 schedules, loss settings,
early stopping and per-type tuning. Portability edits replace machine-specific
data/model paths; they do not claim that all historical seeds used the final
source configuration. Original source SHA-256 values are retained in the release
manifest. Use `scripts/train.py`, not a default all-model/all-experiment launch.

`train_softcrf.py` is a historical dependency providing the optional CRF class.
The primary E1-E9 results are linear BIO heads, not a new soft-label/CRF result.
Its old heuristic noise priors are not the schema-first contribution.

`train_ner_joint.py` is a diagnostic joint-head implementation. It treats
unobserved entity types as O, merges Bronze records by the first 120 characters,
and uses training settings/dev selection that differ from the binary pipeline.
Do not silently repair those behaviors and attribute the old measurements to
the repaired implementation. See the documented comparison limitations.
