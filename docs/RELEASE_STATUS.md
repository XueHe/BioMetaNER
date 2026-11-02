# Submission Release Checklist

Current status: named PRIVATE preparation repository, not a public release.

Prepared:

- [x] Separate BioMetaNER repository; existing MITC dictionary repository preserved.
- [x] E1-E9 binary trainer and diagnostic Joint source, portable data/model paths.
- [x] Pinned matcher and Golden hash, count-only result tables, uncertainty protocol.
- [x] Explicit baseline qualifications; no unsupported manuscript values filled in.
- [x] Synthetic example, local tests, security/content check and an inactive CI template.
- [x] No original text/annotations, raw predictions, checkpoints, API keys or emails.

Before public release or representing the repository as complete reproduction:

- [ ] Confirm all code copyright holders and select a code license with coauthors.
- [ ] Confirm corpus/annotation redistribution terms and a stable access mechanism.
- [ ] Finalize the manuscript author list, citation, DOI and publication version.
- [ ] Run an independent fresh-environment GPU training/inference reproduction.
- [ ] Authorize GitHub workflow uploads and activate the template in docs/ci/.
- [ ] Resolve or remove MINT Dictionary and fine-tuned GLiNER unsupported rows.
- [ ] Keep incompatible DynClean and qualified LLM records out of a homogeneous comparison.
- [ ] Review Joint/Binary data and optimization confounds in the manuscript.
- [ ] Finish a complete immutable dependency/model/data manifest for future runs.
- [ ] Decide model-weight distribution and versioned archival release (e.g. DOI).
- [ ] Re-run credential/content checks before changing visibility or tagging a release.

No request to make this repository public has been made. Privacy is verified
after repository creation and push. Workflow files must be checked on GitHub;
passing local CPU tests is not a substitute for full GPU reproduction.
