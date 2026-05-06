"""Core schemas for phase 7 audio-generation projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


AUDIO_GEN_SCHEMA_VERSION = "1.0"


@dataclass
class ProjectRecord:
    project_id: str
    slug: str
    name: str
    project_type: str = "music_production"
    status: str = "active"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    active_beat_blueprint_id: Optional[str] = None
    active_working_set_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = AUDIO_GEN_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceAssetRecord:
    reference_id: str
    source_type: str
    role: str
    caption: str
    relative_path: Optional[str] = None
    source_url: Optional[str] = None
    provider: Optional[str] = None
    provider_id: Optional[str] = None
    status: str = "active"
    origin: str = "user_upload"
    tags: List[str] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    key_signature: Optional[str] = None
    source_project_id: Optional[str] = None
    source_run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedContextRecord:
    context_id: str
    context_type: str
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
class BeatBlueprintLayerRecord:
    name: str
    role: str
    pattern: str = ""
    source_reference_ids: List[str] = field(default_factory=list)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeatBlueprintRecord:
    version_id: str
    label: str
    summary: str = ""
    status: str = "draft"
    created_at: str = ""
    updated_at: str = ""
    relative_path: Optional[str] = None
    source_reference_ids: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    target_bpm: Optional[float] = None
    bars: int = 4
    steps_per_bar: int = 16
    layers: List[BeatBlueprintLayerRecord] = field(default_factory=list)
    arrangement_notes: List[str] = field(default_factory=list)
    mix_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MixSessionRecord:
    session_id: str
    label: str
    artifact_path: Optional[str] = None
    source_run_id: Optional[str] = None
    target_peak_db: Optional[float] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterSessionRecord:
    master_id: str
    label: str
    artifact_path: Optional[str] = None
    source_run_id: Optional[str] = None
    target_lufs: Optional[float] = None
    true_peak_db: Optional[float] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaptureSessionRecord:
    capture_id: str
    label: str
    source_reference_id: Optional[str] = None
    promoted_reference_id: Optional[str] = None
    source_url: Optional[str] = None
    provider: Optional[str] = None
    provider_id: Optional[str] = None
    device_name: Optional[str] = None
    source_run_id: Optional[str] = None
    raw_capture_path: Optional[str] = None
    clean_capture_path: Optional[str] = None
    status: str = "captured"
    cleanup_status: str = "pending"
    keep_capture: bool = False
    expires_at: Optional[str] = None
    notes: str = ""
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
    capture_ids: List[str] = field(default_factory=list)
    beat_blueprint_ids: List[str] = field(default_factory=list)
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
        "schema_version": AUDIO_GEN_SCHEMA_VERSION,
        "project_id": project_id,
        "manifest_type": manifest_type,
        "items": items or [],
    }
    payload.update(extra)
    return payload
