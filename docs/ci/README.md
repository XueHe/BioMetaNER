# CI Template: Not Enabled

`checks.yml` defines two CPU jobs (Python 3.10 and 3.12). It is intentionally
outside `.github/workflows/` and is not an active GitHub workflow.

The current GitHub login can create repositories and push ordinary code but
lacks the OAuth `workflow` permission required to upload workflow definitions.
Repository preparation did not change login authorization or bypass that limit.
After appropriate authorization, the workflow can be activated in the standard
GitHub directory. Its presence here is not evidence that GitHub CI ran.

Local unit tests, count/CI regression checks and security/content checks were
run successfully; see `provenance/local_validation.json`. No fresh GPU training
reproduction was attempted.
