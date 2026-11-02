# Evaluation and Uncertainty

The canonical partial matcher is the original Golden v2 type-aware greedy
one-to-one character-overlap matcher. Iterate predictions in their saved order;
match the first unmatched same-type gold span with any nonempty character
overlap. There is no IoU threshold, confidence re-sort, boundary normalization,
sentence restriction, or maximum bipartite matching inside the matcher.

The two matcher function bodies are copied without alteration from the source
whose SHA-256 is:

```text
8575708511494ca7653fc0cc78e255c7ef9e4ab4f49ce1e030d43733697625bb
```

For strict matching, use same-type identical character boundaries with
one-to-one multiset counts. Duplicate predictions count as false positives.

F1 = 2 TP / (2 TP + FP + FN). A zero denominator maps to zero. Compute each
type's F1 within each training seed. Macro F1 is the equal-weight mean of MIC,
DIS, CHE and HAB within that seed, then averaged across seeds. Sample SD uses
ddof=1; it is unavailable for one seed. Never pool counts across seeds or average
rounded F1 fields. Only the final display multiplies by 100 and rounds to two
decimals. Micro F1 is a different statistic.

## Inference Versus Matching

Paper inference uses 512-token windows with stride 256 and four model heads.
Per-token labels are argmax of logits. An entity's score is the mean selected
BIO-tag logit; all type/window spans are greedily de-overlapped in descending
score. Scores are unnormalized heuristics, not calibrated cross-head
probabilities. Original inference function bodies and tie ordering are retained.

The original trainer's diagnostic inference uses softmax-based scores and its
scorer re-sorts predictions. Therefore automatic `eval_golden.json` is not the
canonical paper artifact. Use the separate saved-prediction inference/evaluation
entry points. No new inference was run during repository preparation.

## Paired Abstract Bootstrap

The saved 2000-draw analysis resamples 41 complete abstracts with replacement.
Each draw's indices are reused for every configuration, backbone and training
seed. Within each seed, sum sampled document counts, compute four type F1s and
their Macro mean, then average over the fixed training seeds. Contrasts use the
intersection of available seed IDs. No training-seed resampling is performed.

The 95% interval is the 2.5/97.5 percentile with linear interpolation. Seed
sample SD and document-bootstrap CI quantify different variability. Intervals
are marginal, unadjusted for multiple comparisons. The fraction of positive
bootstrap differences is not a formal p-value. Point estimates use the original
full corpus, not the bootstrap mean.
