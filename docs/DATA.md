# Data Layout and Access

Original text, Silver/Golden annotations, source corpora and pretrained weights
are intentionally not in this preparation repository. Obtain authorized copies
from the project maintainers and the original data providers. No public download
URL, data license, archive DOI, or model release is implied by this snapshot.

Expected private data directory:

```text
private-data/
  silver_new.jsonl
  golden_metasigndb_v2_for_eval.jsonl
  bronze_type_by_source_80_10_10_filtered_v1/
    MIC/{train,dev,test}.jsonl
    DIS/{train,dev,test}.jsonl
    CHE/{train,dev,test}.jsonl
    HAB/{train,dev,test}.jsonl
```

Each JSONL record has `pmid`, `text`, and `entities`. Each entity has `start`,
`end`, and `type`; offsets are zero-based Python character indices with an
exclusive end. Entity types are MIC (microorganism), DIS (disease), CHE (chemical),
and HAB (habitat). Retain per-entity source metadata when preparing multi-source
data; a document-level merged source label is not equivalent.

Golden v2 contains 41 abstracts and 1725 entities: MIC 497, DIS 606, CHE 420,
HAB 202. SHA-256:

```text
e874a9d0148cf8dd460e7ac56ee36bcf2267792d70c6852fb3b9d20bbf1cde63
```

Do not replace this file with an older `golden.jsonl` or a later annotation
revision. All text and offset equality checks are intentional. Synthetic
examples can bypass the hash with `--allow-unpinned`; that option is not a way
to label results from another gold version as Golden v2.

The current Silver file has 600 records. The original trainer shuffles by seed
and uses 480 for training and 120 for development. Historical file hashes and
split IDs were not saved for every old run; count verification cannot restore
missing historical training provenance.

`--model-paths` takes a local JSON mapping, for example:

```json
{"pubmedbert": "/path/to/pretrained/pubmedbert-snapshot"}
```

The recorded historical model IDs and snapshot revisions are in
`provenance/backbones.json`. Their upstream availability and redistribution
permissions must be checked before a public model/data release.

Upstream Bronze resources include BC5CDR, BioRED, NCBI Disease, S800,
GutBrainIE and BB2019. Dataset-specific access terms and citations must be
confirmed separately. No blanket license is asserted for their union.
