Image-generation execution artifacts should be stored under this directory.

Project-scoped persistent state should stay under `projects/img_gen/<project_id>/`.
Per-run outputs should resolve to `artifacts/img_gen/<project_id>/run_<timestamp>/...`
once the phase 6 runtime is implemented.
