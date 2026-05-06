"""Filesystem layout helpers for phase 7 audio projects."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict


DEFAULT_AUDIO_GEN_PROJECTS_DIR = os.path.join("projects", "audio_gen")
DEFAULT_AUDIO_GEN_ARTIFACTS_DIR = os.path.join("artifacts", "audio_gen")


@dataclass(frozen=True)
class ProjectPaths:
    project_id: str
    projects_root: str
    project_dir: str
    project_file: str
    references_dir: str
    references_manifest: str
    references_audio_dir: str
    references_text_dir: str
    derived_context_dir: str
    derived_context_manifest: str
    derived_transcripts_dir: str
    derived_analyses_dir: str
    beat_blueprints_dir: str
    beat_blueprints_manifest: str
    beat_blueprint_versions_dir: str
    mix_sessions_dir: str
    mix_sessions_manifest: str
    masters_dir: str
    masters_manifest: str
    capture_sessions_dir: str
    capture_sessions_manifest: str
    working_sets_dir: str
    working_sets_manifest: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectArtifactPaths:
    project_id: str
    artifacts_root: str
    project_artifacts_dir: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def build_project_paths(
    project_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    project_dir = os.path.join(projects_root, project_id)
    references_dir = os.path.join(project_dir, "references")
    derived_context_dir = os.path.join(project_dir, "derived_context")
    beat_blueprints_dir = os.path.join(project_dir, "beat_blueprints")
    mix_sessions_dir = os.path.join(project_dir, "mix_sessions")
    masters_dir = os.path.join(project_dir, "masters")
    capture_sessions_dir = os.path.join(project_dir, "capture_sessions")
    working_sets_dir = os.path.join(project_dir, "working_sets")

    return ProjectPaths(
        project_id=project_id,
        projects_root=projects_root,
        project_dir=project_dir,
        project_file=os.path.join(project_dir, "project.json"),
        references_dir=references_dir,
        references_manifest=os.path.join(references_dir, "manifest.json"),
        references_audio_dir=os.path.join(references_dir, "audio"),
        references_text_dir=os.path.join(references_dir, "text"),
        derived_context_dir=derived_context_dir,
        derived_context_manifest=os.path.join(derived_context_dir, "manifest.json"),
        derived_transcripts_dir=os.path.join(derived_context_dir, "transcripts"),
        derived_analyses_dir=os.path.join(derived_context_dir, "analyses"),
        beat_blueprints_dir=beat_blueprints_dir,
        beat_blueprints_manifest=os.path.join(beat_blueprints_dir, "manifest.json"),
        beat_blueprint_versions_dir=os.path.join(beat_blueprints_dir, "versions"),
        mix_sessions_dir=mix_sessions_dir,
        mix_sessions_manifest=os.path.join(mix_sessions_dir, "manifest.json"),
        masters_dir=masters_dir,
        masters_manifest=os.path.join(masters_dir, "manifest.json"),
        capture_sessions_dir=capture_sessions_dir,
        capture_sessions_manifest=os.path.join(capture_sessions_dir, "manifest.json"),
        working_sets_dir=working_sets_dir,
        working_sets_manifest=os.path.join(working_sets_dir, "manifest.json"),
    )


def build_project_artifact_paths(
    project_id: str,
    artifacts_root: str = DEFAULT_AUDIO_GEN_ARTIFACTS_DIR,
) -> ProjectArtifactPaths:
    return ProjectArtifactPaths(
        project_id=project_id,
        artifacts_root=artifacts_root,
        project_artifacts_dir=os.path.join(artifacts_root, project_id),
    )
