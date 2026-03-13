Phase 6 image-generation modules will live in this package.

Planned structure:

- project-scoped state keyed by `project_id`;
- reusable `visual_bible` versions;
- reference stores for images, text, audio, and video-derived context;
- run-level manifests and preview generation;
- a dedicated web interface that can later be mounted into a shared `main_app.py`.
