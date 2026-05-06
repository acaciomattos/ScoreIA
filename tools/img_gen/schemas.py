"""Core schemas for image-generation projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


IMG_GEN_SCHEMA_VERSION = "1.0"


@dataclass
class ProjectRecord:
    project_id: str
    slug: str
    name: str
    project_type: str = "single_image"
    status: str = "active"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    active_visual_bible_id: Optional[str] = None
    active_working_set_id: Optional[str] = None
    import_policy: str = "pointer_manifest"
    snapshot_policy: str = "promote_on_freeze"
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = IMG_GEN_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualStyleProfileRecord:
    style_keywords: List[str] = field(default_factory=list)
    palette_keywords: List[str] = field(default_factory=list)
    lighting_keywords: List[str] = field(default_factory=list)
    camera_keywords: List[str] = field(default_factory=list)
    mood_keywords: List[str] = field(default_factory=list)
    render_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualEntityRecord:
    label: str
    summary: str = ""
    prompt_anchor: str = ""
    negative_prompt_anchor: str = ""
    reference_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    locked: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualBibleRecord:
    version_id: str
    label: str
    summary: str = ""
    status: str = "draft"
    created_at: str = ""
    updated_at: str = ""
    relative_path: Optional[str] = None
    notes: str = ""
    source_reference_ids: List[str] = field(default_factory=list)
    prompt_anchor: str = ""
    negative_prompt_anchor: str = ""
    style_profile: VisualStyleProfileRecord = field(default_factory=VisualStyleProfileRecord)
    characters: List[VisualEntityRecord] = field(default_factory=list)
    environments: List[VisualEntityRecord] = field(default_factory=list)
    props: List[VisualEntityRecord] = field(default_factory=list)
    composition_rules: List[str] = field(default_factory=list)
    continuity_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceAssetRecord:
    reference_id: str
    source_type: str
    role: str
    caption: str
    relative_path: Optional[str] = None
    status: str = "active"
    origin: str = "user_upload"
    tags: List[str] = field(default_factory=list)
    source_project_id: Optional[str] = None
    source_run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportedProjectSelector:
    reference_ids: List[str] = field(default_factory=list)
    reference_roles: List[str] = field(default_factory=list)
    reference_tags: List[str] = field(default_factory=list)
    visual_bible_ids: List[str] = field(default_factory=list)
    visual_bible_labels: List[str] = field(default_factory=list)
    character_labels: List[str] = field(default_factory=list)
    environment_labels: List[str] = field(default_factory=list)
    prop_labels: List[str] = field(default_factory=list)
    promoted_asset_ids: List[str] = field(default_factory=list)
    promoted_asset_roles: List[str] = field(default_factory=list)
    promoted_asset_tags: List[str] = field(default_factory=list)
    derived_context_ids: List[str] = field(default_factory=list)
    working_set_ids: List[str] = field(default_factory=list)
    include_active_visual_bible: bool = False
    include_active_working_set: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportedProjectRecord:
    import_id: str
    source_project_id: str
    alias: str = ""
    selection_mode: str = "explicit_assets"
    import_strategy: str = "pointer_manifest"
    snapshot_policy: str = "promote_on_freeze"
    selectors: ImportedProjectSelector = field(default_factory=ImportedProjectSelector)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotedAssetRecord:
    asset_id: str
    asset_type: str
    role: str
    caption: str
    relative_path: Optional[str] = None
    status: str = "active"
    source_project_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_asset_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedContextRecord:
    context_id: str
    source_type: str
    label: str
    summary: str = ""
    relative_path: Optional[str] = None
    source_reference_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkingSetRecord:
    working_set_id: str
    label: str
    description: str = ""
    reference_ids: List[str] = field(default_factory=list)
    promoted_asset_ids: List[str] = field(default_factory=list)
    imported_project_ids: List[str] = field(default_factory=list)
    derived_context_ids: List[str] = field(default_factory=list)
    is_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_items_manifest(
    project_id: str,
    manifest_type: str,
    items: Optional[List[Dict[str, Any]]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {
        "schema_version": IMG_GEN_SCHEMA_VERSION,
        "project_id": project_id,
        "manifest_type": manifest_type,
        "items": items or [],
    }
    payload.update(extra)
    return payload
