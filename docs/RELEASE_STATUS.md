# Submission Release Checklist

Current status: public code and audited-result release, authorized by the owner
on 2026-11-02. This is not a complete corpus release or a claim of fresh GPU reproduction.

Prepared:

- [x] Separate BioMetaNER repository; existing MITC dictionary repository preserved.
- [x] E1-E9 binary trainer and diagnostic Joint source, portable data/model paths.
- [x] Pinned matcher and Golden hash, count-only result tables, uncertainty protocol.
- [x] Explicit baseline qualifications; no unsupported manuscript values filled in.
- [x] Synthetic example, local tests, security/content check and an inactive CI template.
- [x] No original text/annotations, raw predictions, checkpoints, API keys or emails.
- [x] PubMedBERT E6 seed-42 checkpoints published separately at
  https://huggingface.co/jasminhx/BioMetaNER-schemefirst (four experts).

Remaining work before claiming a fully licensed, complete reproduction release:

- [ ] Confirm all code copyright holders and select a code license with coauthors.
- [ ] Confirm corpus/annotation redistribution terms and a stable access mechanism.
- [ ] Finalize the manuscript author list, citation, DOI and publication version.
- [ ] Run an independent fresh-environment GPU training/inference reproduction.
- [ ] Authorize GitHub workflow uploads and activate the template in docs/ci/.
- [ ] Resolve or remove MINT Dictionary and fine-tuned GLiNER unsupported rows.
- [ ] Keep incompatible DynClean and qualified LLM records out of a homogeneous comparison.
- [ ] Review Joint/Binary data and optimization confounds in the manuscript.
- [ ] Finish a complete immutable dependency/model/data manifest for future runs.
- [ ] Decide a versioned archival release (e.g. DOI) and any additional checkpoint seeds.
- [x] Re-run credential/content checks before the owner-requested visibility change.

The owner explicitly requested public visibility for this repository and the
linked Hugging Face model repository. Only curated code/numerical results and
the separately documented model bundle are released, not the underlying corpora.
Workflow files must still be checked on GitHub; passing local CPU tests is not a
substitute for full GPU reproduction. No new permissive code license is implied.
