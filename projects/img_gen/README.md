This directory stores image-generation projects keyed by `project_id`.

Long-lived project state such as `visual_bible`, imported references, promoted assets,
and derived context should live here. Per-run outputs should stay under `artifacts/img_gen/`.

Recommended project tree:

```text
projects/img_gen/<project_id>/
  project.json
  visual_bibles/
    manifest.json
    versions/
  references/
    manifest.json
    images/
    text/
    audio/
    video/
  imported_projects/
    manifest.json
  promoted_assets/
    manifest.json
    images/
  derived_context/
    manifest.json
    transcripts/
    summaries/
    scene_lists/
  working_sets/
    manifest.json
```

Recommended import strategy for `imported_projects`:

- store a manifest that points to the source `project_id`;
- list exactly which asset ids, labels, roles, tags, or active items are being reused;
- keep hashes and metadata for reproducibility;
- avoid copying the entire source project by default.

When the user decides to freeze or promote an imported asset into the current project,
the current project may create a local snapshot copy of that specific asset only.

Recommended `visual_bible` version contents:

- `label`, `summary`, `notes`, and optional `relative_path`
- `prompt_anchor` and `negative_prompt_anchor`
- `style_profile` as structured JSON-like metadata
- `characters`, `environments`, and `props` as named reusable entities
- `composition_rules` and `continuity_rules` for cross-run consistency
