Phase 6 image-generation modules will live in this package.

Planned structure:

- project-scoped state keyed by `project_id`;
- reusable `visual_bible` versions;
- reference stores for images, text, audio, and video-derived context;
- run-level manifests and preview generation;
- a dedicated web interface that can later be mounted into a shared `main_app.py`.

Current core modules:

- `model_paths.py`
  - canonical filesystem layout for local checkpoints, adapters, SAM weights,
    upscalers, cache, and optional `external/ComfyUI`
- `image_models.py`
  - local registry and download helpers for model assets under `models/img_gen`
  - exposes `ensure_model_storage`, `list_model_artifacts`,
    `resolve_model_source_path`, category-aware path inspection/validation,
    status refresh for manual downloads, and download helpers for Hugging Face snapshots
  - includes default aliases for the SD 1.5 base checkpoint and a dedicated
    SD 1.5 inpainting model (`sd15_inpainting_base`)
- `runtime_env.py`
  - runtime diagnostics for local generation
  - reports installed package versions, import errors, CUDA availability,
    checkpoint readiness, and CPU-only `torch` builds
  - validates the correct model category for base checkpoints versus inpainting checkpoints
- `image_generation.py`
  - local runners for `txt2img`, `img2img`, and `inpainting`
  - writes a unified `generation_manifest.json` plus mode-specific manifests
  - supports optional `IP-Adapter` and one `ControlNet` at a time for `txt2img` / `img2img`
  - supports per-run `safety_checker` toggle and records when NSFW was signaled
  - applies an internal SD 1.5 quality preset with prompt/negative enrichment
  - switches to a higher-quality `DPM++ 2M Karras` scheduler when supported
  - uses mode-specific defaults tuned for the current `RTX 2060 6 GB` profile
  - applies blur to inpainting masks to improve seam integration
  - expects local aliases registered for `ip_adapter` and `controlnet`
  - saves outputs under `artifacts/img_gen/<project_id>/run_<timestamp>/...`
    or `artifacts/img_gen/_ad_hoc/...` when no project is provided
  - inpainting validates whether the checkpoint is a dedicated inpaint model
- `project_paths.py`
  - canonical filesystem layout for `projects/img_gen/<project_id>/...`
- `schemas.py`
  - base records for project metadata, references, imports, promoted assets, derived context, and working sets
  - `visual_bible` versions now support `prompt_anchor`, `negative_prompt_anchor`,
    `style_profile`, `characters`, `environments`, `props`, `composition_rules`,
    and `continuity_rules`
  - `VisualStyleProfileRecord` models style keywords, palette, lighting, camera,
    mood, render, and negative keyword groups
  - `VisualEntityRecord` models each personagem/ambiente/prop with `label`,
    anchors, tags, references, `attributes`, and lock flag
  - imported project manifests now use explicit selector payloads such as
    `reference_ids`, `reference_roles`, `reference_tags`, `visual_bible_ids`,
    `visual_bible_labels`, `character_labels`, `environment_labels`, `prop_labels`,
    `promoted_asset_ids`, `derived_context_ids`, and `working_set_ids`
- `project_registry.py`
  - create, list, load, update, and persist project-scoped manifests
  - exposes `set_active_visual_bible`, `set_active_working_set`, `load_project_bundle`,
    and promotion of generated images into `promoted_assets` / `references`
- `webapp/app.py`
  - project-centric web interface for creating projects, registering references,
    versioning `visual_bible`, importing prior projects, assembling working sets,
    generating local `txt2img` / `img2img`, preparing inpainting runs,
    and exposing optional IP-Adapter / ControlNet controls,
    previewing recent runs, promoting approved outputs,
    and registering local aliases for `checkpoints`, `inpainting`, `IP-Adapter`,
    and `ControlNet` without editing `models/img_gen/registry.json` manually
  - surfaces safety-checker state in the forms and preview grid to help diagnose black-image false positives
  - defaults now prioritize image quality while preserving the simplified prompt-first flow
