This directory stores image-generation projects keyed by `project_id`.

Long-lived project state such as `visual_bible`, imported references, promoted assets,
and derived context should live here. Per-run outputs should stay under `artifacts/img_gen/`.

Recommended import strategy for `imported_projects`:

- store a manifest that points to the source `project_id`;
- list exactly which asset ids or versions are being reused;
- keep hashes and metadata for reproducibility;
- avoid copying the entire source project by default.

When the user decides to freeze or promote an imported asset into the current project,
the current project may create a local snapshot copy of that specific asset only.
