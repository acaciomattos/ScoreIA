"""Filesystem layout helpers for image-generation projects."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict


DEFAULT_IMG_GEN_PROJECTS_DIR = os.path.join("projects", "img_gen")
DEFAULT_IMG_GEN_ARTIFACTS_DIR = os.path.join("artifacts", "img_gen")


@dataclass(frozen=True)
class ProjectPaths:
    project_id: str
    projects_root: str
    project_dir: str
    project_file: str
    visual_bibles_dir: str
    visual_bibles_manifest: str
    visual_bible_versions_dir: str
    references_dir: str
    references_manifest: str
    references_images_dir: str
    references_text_dir: str
    references_audio_dir: str
    references_video_dir: str
    imported_projects_dir: str
    imported_projects_manifest: str
    promoted_assets_dir: str
    promoted_assets_manifest: str
    promoted_assets_images_dir: str
    derived_context_dir: str
    derived_context_manifest: str
    derived_transcripts_dir: str
    derived_summaries_dir: str
    derived_scene_lists_dir: str
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
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    project_dir = os.path.join(projects_root, project_id)
    visual_bibles_dir = os.path.join(project_dir, "visual_bibles")
    references_dir = os.path.join(project_dir, "references")
    imported_projects_dir = os.path.join(project_dir, "imported_projects")
    promoted_assets_dir = os.path.join(project_dir, "promoted_assets")
    derived_context_dir = os.path.join(project_dir, "derived_context")
    working_sets_dir = os.path.join(project_dir, "working_sets")

    return ProjectPaths(
        project_id=project_id,
        projects_root=projects_root,
        project_dir=project_dir,
        project_file=os.path.join(project_dir, "project.json"),
        visual_bibles_dir=visual_bibles_dir,
        visual_bibles_manifest=os.path.join(visual_bibles_dir, "manifest.json"),
        visual_bible_versions_dir=os.path.join(visual_bibles_dir, "versions"),
        references_dir=references_dir,
        references_manifest=os.path.join(references_dir, "manifest.json"),
        references_images_dir=os.path.join(references_dir, "images"),
        references_text_dir=os.path.join(references_dir, "text"),
        references_audio_dir=os.path.join(references_dir, "audio"),
        references_video_dir=os.path.join(references_dir, "video"),
        imported_projects_dir=imported_projects_dir,
        imported_projects_manifest=os.path.join(imported_projects_dir, "manifest.json"),
        promoted_assets_dir=promoted_assets_dir,
        promoted_assets_manifest=os.path.join(promoted_assets_dir, "manifest.json"),
        promoted_assets_images_dir=os.path.join(promoted_assets_dir, "images"),
        derived_context_dir=derived_context_dir,
        derived_context_manifest=os.path.join(derived_context_dir, "manifest.json"),
        derived_transcripts_dir=os.path.join(derived_context_dir, "transcripts"),
        derived_summaries_dir=os.path.join(derived_context_dir, "summaries"),
        derived_scene_lists_dir=os.path.join(derived_context_dir, "scene_lists"),
        working_sets_dir=working_sets_dir,
        working_sets_manifest=os.path.join(working_sets_dir, "manifest.json"),
    )


def build_project_artifact_paths(
    project_id: str,
    artifacts_root: str = DEFAULT_IMG_GEN_ARTIFACTS_DIR,
) -> ProjectArtifactPaths:
    return ProjectArtifactPaths(
        project_id=project_id,
        artifacts_root=artifacts_root,
        project_artifacts_dir=os.path.join(artifacts_root, project_id),
    )
