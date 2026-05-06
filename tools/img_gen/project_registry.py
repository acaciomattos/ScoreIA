"""Registry and persistence helpers for image-generation projects."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from tools.img_gen.project_paths import (
    DEFAULT_IMG_GEN_PROJECTS_DIR,
    ProjectPaths,
    build_project_artifact_paths,
    build_project_paths,
)
from tools.img_gen.schemas import (
    DerivedContextRecord,
    ImportedProjectRecord,
    ImportedProjectSelector,
    ProjectRecord,
    PromotedAssetRecord,
    ReferenceAssetRecord,
    VisualBibleRecord,
    VisualEntityRecord,
    VisualStyleProfileRecord,
    WorkingSetRecord,
    build_items_manifest,
)


MANIFEST_ATTRS = {
    "visual_bibles": "visual_bibles_manifest",
    "references": "references_manifest",
    "imported_projects": "imported_projects_manifest",
    "promoted_assets": "promoted_assets_manifest",
    "derived_context": "derived_context_manifest",
    "working_sets": "working_sets_manifest",
}


VISUAL_ENTITY_REQUIRED_LABELS = {
    "characters": "personagem",
    "environments": "ambiente",
    "props": "prop",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "img-project"


def _build_project_id(name: str) -> str:
    return f"{_slugify(name)}-{uuid.uuid4().hex[:8]}"


def _build_item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify_filename(value: str) -> str:
    base_name = os.path.splitext(os.path.basename(_clean_text(value)))[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_name.lower()).strip("-")
    return slug or "generated-image"


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_str_list(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    normalized: List[str] = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _normalize_mapping(value: Optional[Dict[str, Any]], field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"O campo '{field_name}' deve ser um objeto/dict.")
    return dict(value)


def _normalize_style_profile(
    style_profile: Optional[Any],
) -> VisualStyleProfileRecord:
    if isinstance(style_profile, VisualStyleProfileRecord):
        return style_profile

    payload = _normalize_mapping(style_profile, "style_profile")
    metadata = _normalize_mapping(payload.get("metadata"), "style_profile.metadata")
    known_fields = {
        "style_keywords",
        "palette_keywords",
        "lighting_keywords",
        "camera_keywords",
        "mood_keywords",
        "render_keywords",
        "negative_keywords",
        "metadata",
    }
    extra_fields = {
        key: value
        for key, value in payload.items()
        if key not in known_fields and value not in (None, "", [], {})
    }
    if extra_fields:
        metadata["extra_fields"] = extra_fields

    return VisualStyleProfileRecord(
        style_keywords=_normalize_str_list(payload.get("style_keywords")),
        palette_keywords=_normalize_str_list(payload.get("palette_keywords")),
        lighting_keywords=_normalize_str_list(payload.get("lighting_keywords")),
        camera_keywords=_normalize_str_list(payload.get("camera_keywords")),
        mood_keywords=_normalize_str_list(payload.get("mood_keywords")),
        render_keywords=_normalize_str_list(payload.get("render_keywords")),
        negative_keywords=_normalize_str_list(payload.get("negative_keywords")),
        metadata=metadata,
    )


def _normalize_visual_bible_entities(
    items: Optional[Iterable[Any]],
    bucket_name: str,
) -> List[VisualEntityRecord]:
    entity_label = VISUAL_ENTITY_REQUIRED_LABELS.get(bucket_name, bucket_name)
    normalized_items: List[VisualEntityRecord] = []

    for index, item in enumerate(items or [], start=1):
        if isinstance(item, VisualEntityRecord):
            normalized_items.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(
                f"Cada item em '{bucket_name}' deve ser um objeto/dict. "
                f"Recebido na posicao {index}: {type(item).__name__}."
            )

        payload = dict(item)
        label = _clean_text(payload.get("label") or payload.get("name"))
        if not label:
            raise ValueError(
                f"Cada {entity_label} no visual_bible deve informar 'label'. "
                f"Item invalido na posicao {index}."
            )

        payload["label"] = label
        summary = _clean_text(
            payload.get("summary") or payload.get("description") or ""
        )
        prompt_anchor = _clean_text(payload.get("prompt_anchor"))
        negative_prompt_anchor = _clean_text(payload.get("negative_prompt_anchor"))
        notes = _clean_text(payload.get("notes"))
        reference_ids = _normalize_str_list(payload.get("reference_ids"))
        tags = _normalize_str_list(payload.get("tags"))
        locked = _normalize_bool(payload.get("locked", False))
        metadata = _normalize_mapping(payload.get("metadata"), f"{bucket_name}.metadata")
        attributes = _normalize_mapping(payload.get("attributes"), f"{bucket_name}.attributes")

        known_fields = {
            "label",
            "name",
            "summary",
            "description",
            "prompt_anchor",
            "negative_prompt_anchor",
            "reference_ids",
            "tags",
            "notes",
            "locked",
            "attributes",
            "metadata",
        }
        for key, value in payload.items():
            if key in known_fields or value in (None, "", [], {}):
                continue
            attributes.setdefault(key, value)

        normalized_items.append(
            VisualEntityRecord(
                label=label,
                summary=summary,
                prompt_anchor=prompt_anchor,
                negative_prompt_anchor=negative_prompt_anchor,
                reference_ids=reference_ids,
                tags=tags,
                notes=notes,
                locked=locked,
                attributes=attributes,
                metadata=metadata,
            )
        )

    return normalized_items


def _normalize_imported_project_selectors(
    selectors: Optional[Any],
) -> ImportedProjectSelector:
    if isinstance(selectors, ImportedProjectSelector):
        return selectors
    else:
        payload = _normalize_mapping(selectors, "selectors") if selectors is not None else {}

    return ImportedProjectSelector(
        reference_ids=_normalize_str_list(payload.get("reference_ids")),
        reference_roles=_normalize_str_list(payload.get("reference_roles")),
        reference_tags=_normalize_str_list(payload.get("reference_tags")),
        visual_bible_ids=_normalize_str_list(payload.get("visual_bible_ids")),
        visual_bible_labels=_normalize_str_list(payload.get("visual_bible_labels")),
        character_labels=_normalize_str_list(payload.get("character_labels")),
        environment_labels=_normalize_str_list(payload.get("environment_labels")),
        prop_labels=_normalize_str_list(payload.get("prop_labels")),
        promoted_asset_ids=_normalize_str_list(payload.get("promoted_asset_ids")),
        promoted_asset_roles=_normalize_str_list(payload.get("promoted_asset_roles")),
        promoted_asset_tags=_normalize_str_list(payload.get("promoted_asset_tags")),
        derived_context_ids=_normalize_str_list(payload.get("derived_context_ids")),
        working_set_ids=_normalize_str_list(payload.get("working_set_ids")),
        include_active_visual_bible=_normalize_bool(
            payload.get("include_active_visual_bible", False)
        ),
        include_active_working_set=_normalize_bool(
            payload.get("include_active_working_set", False)
        ),
    )


def _merge_selector_list(
    selectors: Dict[str, Any],
    key: str,
    values: Optional[Iterable[Any]],
) -> None:
    merged = _normalize_str_list(selectors.get(key))
    merged.extend(_normalize_str_list(values))
    selectors[key] = _normalize_str_list(merged)


def _default_manifest_payload(project_id: str, manifest_name: str) -> Dict[str, Any]:
    base = build_items_manifest(project_id, manifest_name)
    if manifest_name == "visual_bibles":
        base["active_visual_bible_id"] = None
    if manifest_name == "working_sets":
        base["active_working_set_id"] = None
    return base


def _resolve_existing_project_paths(
    project_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    paths = build_project_paths(project_id, projects_root=projects_root)
    if not os.path.exists(paths.project_file):
        raise FileNotFoundError(f"Projeto nao encontrado: {project_id}")
    return paths


def _ensure_existing_project_storage(
    project_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    paths = _resolve_existing_project_paths(project_id, projects_root=projects_root)
    _ensure_project_dirs(paths)

    for manifest_name in MANIFEST_ATTRS:
        manifest_path = _resolve_manifest_path(paths, manifest_name)
        if not os.path.exists(manifest_path):
            _write_json(
                manifest_path,
                _default_manifest_payload(project_id, manifest_name),
            )

    artifact_paths = build_project_artifact_paths(project_id)
    os.makedirs(artifact_paths.project_artifacts_dir, exist_ok=True)
    return paths


def _ensure_project_dirs(paths: ProjectPaths) -> None:
    directories = [
        paths.project_dir,
        paths.visual_bibles_dir,
        paths.visual_bible_versions_dir,
        paths.references_dir,
        paths.references_images_dir,
        paths.references_text_dir,
        paths.references_audio_dir,
        paths.references_video_dir,
        paths.imported_projects_dir,
        paths.promoted_assets_dir,
        paths.promoted_assets_images_dir,
        paths.derived_context_dir,
        paths.derived_transcripts_dir,
        paths.derived_summaries_dir,
        paths.derived_scene_lists_dir,
        paths.working_sets_dir,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def _resolve_manifest_path(paths: ProjectPaths, manifest_name: str) -> str:
    try:
        return getattr(paths, MANIFEST_ATTRS[manifest_name])
    except KeyError as exc:
        raise ValueError(f"Manifesto nao suportado: {manifest_name}") from exc


def bootstrap_project_storage(
    project_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    paths = build_project_paths(project_id, projects_root=projects_root)
    _ensure_project_dirs(paths)

    for manifest_name in MANIFEST_ATTRS:
        manifest_path = _resolve_manifest_path(paths, manifest_name)
        if not os.path.exists(manifest_path):
            _write_json(
                manifest_path,
                _default_manifest_payload(project_id, manifest_name),
            )

    artifact_paths = build_project_artifact_paths(project_id)
    os.makedirs(artifact_paths.project_artifacts_dir, exist_ok=True)
    return paths


def create_project(
    name: str,
    project_type: str = "single_image",
    description: str = "",
    tags: Optional[Iterable[str]] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("O nome do projeto deve ser informado.")

    resolved_project_id = project_id or _build_project_id(clean_name)
    paths = build_project_paths(resolved_project_id, projects_root=projects_root)
    if os.path.exists(paths.project_file):
        raise FileExistsError(f"O projeto '{resolved_project_id}' ja existe.")

    now = _now_iso()
    project = ProjectRecord(
        project_id=resolved_project_id,
        slug=_slugify(clean_name),
        name=clean_name,
        project_type=project_type,
        description=description.strip(),
        tags=_normalize_str_list(tags),
        created_at=now,
        updated_at=now,
        metadata=_normalize_mapping(metadata, "metadata"),
    )

    bootstrap_project_storage(resolved_project_id, projects_root=projects_root)
    _write_json(paths.project_file, project.to_dict())
    return {"project": project.to_dict(), "paths": paths.to_dict()}


def load_project(
    project_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    return _read_json(
        _resolve_existing_project_paths(project_id, projects_root=projects_root).project_file
    )


def save_project(
    project_payload: Dict[str, Any],
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    project_id = (project_payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id ausente no payload do projeto.")

    paths = bootstrap_project_storage(project_id, projects_root=projects_root)
    payload = dict(project_payload)
    payload["updated_at"] = _now_iso()
    _write_json(paths.project_file, payload)
    return payload


def update_project(
    project_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
    **changes: Any,
) -> Dict[str, Any]:
    project = load_project(project_id, projects_root=projects_root)
    project.update(changes)
    return save_project(project, projects_root=projects_root)


def list_projects(
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> List[Dict[str, Any]]:
    if not os.path.exists(projects_root):
        return []

    projects: List[Dict[str, Any]] = []
    for entry in os.listdir(projects_root):
        project_file = os.path.join(projects_root, entry, "project.json")
        if not os.path.exists(project_file):
            continue
        try:
            projects.append(_read_json(project_file))
        except Exception:
            continue

    return sorted(
        projects,
        key=lambda item: (
            item.get("updated_at", ""),
            item.get("created_at", ""),
            item.get("project_id", ""),
        ),
        reverse=True,
    )


def load_manifest(
    project_id: str,
    manifest_name: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    return _read_json(_resolve_manifest_path(paths, manifest_name))


def save_manifest(
    project_id: str,
    manifest_name: str,
    payload: Dict[str, Any],
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    manifest_path = _resolve_manifest_path(paths, manifest_name)
    persisted = dict(payload)
    persisted["project_id"] = project_id
    persisted["manifest_type"] = manifest_name
    _write_json(manifest_path, persisted)
    return persisted


def _append_manifest_item(
    project_id: str,
    manifest_name: str,
    item_payload: Dict[str, Any],
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, manifest_name, projects_root=projects_root)
    manifest.setdefault("items", []).append(item_payload)
    save_manifest(project_id, manifest_name, manifest, projects_root=projects_root)
    update_project(project_id, projects_root=projects_root)
    return item_payload


def _replace_manifest_item(
    project_id: str,
    manifest_name: str,
    item_key: str,
    item_id: str,
    item_payload: Dict[str, Any],
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, manifest_name, projects_root=projects_root)
    for index, item in enumerate(manifest.get("items", [])):
        if item.get(item_key) != item_id:
            continue
        manifest["items"][index] = item_payload
        save_manifest(project_id, manifest_name, manifest, projects_root=projects_root)
        update_project(project_id, projects_root=projects_root)
        return item_payload
    raise ValueError(f"Item nao encontrado em '{manifest_name}': {item_id}")


def _find_manifest_item(
    manifest: Dict[str, Any],
    item_key: str,
    item_id: str,
) -> Optional[Dict[str, Any]]:
    for item in manifest.get("items", []):
        if item.get(item_key) == item_id:
            return item
    return None


def add_visual_bible_version(
    project_id: str,
    label: str,
    summary: str = "",
    notes: str = "",
    relative_path: Optional[str] = None,
    source_reference_ids: Optional[Iterable[str]] = None,
    prompt_anchor: str = "",
    negative_prompt_anchor: str = "",
    style_profile: Optional[Dict[str, Any]] = None,
    characters: Optional[Iterable[Dict[str, Any]]] = None,
    environments: Optional[Iterable[Dict[str, Any]]] = None,
    props: Optional[Iterable[Dict[str, Any]]] = None,
    composition_rules: Optional[Iterable[str]] = None,
    continuity_rules: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    set_active: bool = True,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_label = _clean_text(label)
    if not clean_label:
        raise ValueError("O visual_bible deve informar um 'label'.")

    now = _now_iso()
    version = VisualBibleRecord(
        version_id=_build_item_id("vb"),
        label=clean_label,
        summary=_clean_text(summary),
        notes=_clean_text(notes),
        relative_path=relative_path,
        source_reference_ids=_normalize_str_list(source_reference_ids),
        prompt_anchor=_clean_text(prompt_anchor),
        negative_prompt_anchor=_clean_text(negative_prompt_anchor),
        style_profile=_normalize_style_profile(style_profile),
        characters=_normalize_visual_bible_entities(characters, "characters"),
        environments=_normalize_visual_bible_entities(environments, "environments"),
        props=_normalize_visual_bible_entities(props, "props"),
        composition_rules=_normalize_str_list(composition_rules),
        continuity_rules=_normalize_str_list(continuity_rules),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )

    item = _append_manifest_item(
        project_id,
        "visual_bibles",
        version.to_dict(),
        projects_root=projects_root,
    )

    if set_active:
        set_active_visual_bible(
            project_id,
            version.version_id,
            projects_root=projects_root,
        )

    return item


def load_visual_bible_version(
    project_id: str,
    version_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, "visual_bibles", projects_root=projects_root)
    version = _find_manifest_item(manifest, "version_id", version_id)
    if version is None:
        raise ValueError(f"Visual_bible nao encontrado no projeto: {version_id}")
    return version


def update_visual_bible_version(
    project_id: str,
    version_id: str,
    label: str,
    summary: str = "",
    notes: str = "",
    relative_path: Optional[str] = None,
    source_reference_ids: Optional[Iterable[str]] = None,
    prompt_anchor: str = "",
    negative_prompt_anchor: str = "",
    style_profile: Optional[Dict[str, Any]] = None,
    characters: Optional[Iterable[Dict[str, Any]]] = None,
    environments: Optional[Iterable[Dict[str, Any]]] = None,
    props: Optional[Iterable[Dict[str, Any]]] = None,
    composition_rules: Optional[Iterable[str]] = None,
    continuity_rules: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    set_active: Optional[bool] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_label = _clean_text(label)
    if not clean_label:
        raise ValueError("O visual_bible deve informar um 'label'.")

    existing = load_visual_bible_version(
        project_id,
        version_id,
        projects_root=projects_root,
    )
    now = _now_iso()
    version = VisualBibleRecord(
        version_id=version_id,
        label=clean_label,
        summary=_clean_text(summary),
        status=_clean_text(existing.get("status")) or "draft",
        created_at=existing.get("created_at") or now,
        updated_at=now,
        relative_path=relative_path,
        notes=_clean_text(notes),
        source_reference_ids=_normalize_str_list(source_reference_ids),
        prompt_anchor=_clean_text(prompt_anchor),
        negative_prompt_anchor=_clean_text(negative_prompt_anchor),
        style_profile=_normalize_style_profile(style_profile),
        characters=_normalize_visual_bible_entities(characters, "characters"),
        environments=_normalize_visual_bible_entities(environments, "environments"),
        props=_normalize_visual_bible_entities(props, "props"),
        composition_rules=_normalize_str_list(composition_rules),
        continuity_rules=_normalize_str_list(continuity_rules),
        metadata=_normalize_mapping(metadata, "metadata"),
    )

    item = _replace_manifest_item(
        project_id,
        "visual_bibles",
        "version_id",
        version_id,
        version.to_dict(),
        projects_root=projects_root,
    )

    if set_active:
        set_active_visual_bible(
            project_id,
            version_id,
            projects_root=projects_root,
        )

    return item


def add_reference_asset(
    project_id: str,
    source_type: str,
    role: str,
    caption: str,
    relative_path: Optional[str] = None,
    origin: str = "user_upload",
    tags: Optional[Iterable[str]] = None,
    source_project_id: Optional[str] = None,
    source_run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    asset = ReferenceAssetRecord(
        reference_id=_build_item_id("ref"),
        source_type=source_type,
        role=role,
        caption=caption.strip(),
        relative_path=relative_path,
        origin=origin,
        tags=_normalize_str_list(tags),
        source_project_id=source_project_id,
        source_run_id=source_run_id,
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "references",
        asset.to_dict(),
        projects_root=projects_root,
    )


def add_imported_project(
    project_id: str,
    source_project_id: str,
    alias: str = "",
    selection_mode: str = "explicit_assets",
    selectors: Optional[Any] = None,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    import_strategy: str = "pointer_manifest",
    snapshot_policy: str = "promote_on_freeze",
    selected_reference_ids: Optional[Iterable[str]] = None,
    selected_visual_bible_ids: Optional[Iterable[str]] = None,
    selected_promoted_asset_ids: Optional[Iterable[str]] = None,
    selected_derived_context_ids: Optional[Iterable[str]] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_source_project_id = _clean_text(source_project_id)
    if not clean_source_project_id:
        raise ValueError("O projeto importado deve informar um source_project_id.")

    if isinstance(selectors, ImportedProjectSelector):
        selector_payload = selectors.to_dict()
    else:
        selector_payload = dict(selectors or {})
    _merge_selector_list(selector_payload, "reference_ids", selected_reference_ids)
    _merge_selector_list(selector_payload, "visual_bible_ids", selected_visual_bible_ids)
    _merge_selector_list(selector_payload, "promoted_asset_ids", selected_promoted_asset_ids)
    _merge_selector_list(selector_payload, "derived_context_ids", selected_derived_context_ids)

    now = _now_iso()
    imported = ImportedProjectRecord(
        import_id=_build_item_id("import"),
        source_project_id=clean_source_project_id,
        alias=_clean_text(alias),
        selection_mode=_clean_text(selection_mode) or "explicit_assets",
        selectors=_normalize_imported_project_selectors(selector_payload),
        notes=_clean_text(notes),
        metadata=_normalize_mapping(metadata, "metadata"),
        import_strategy=import_strategy,
        snapshot_policy=snapshot_policy,
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "imported_projects",
        imported.to_dict(),
        projects_root=projects_root,
    )


def add_promoted_asset(
    project_id: str,
    asset_type: str,
    role: str,
    caption: str,
    relative_path: Optional[str] = None,
    source_project_id: Optional[str] = None,
    source_run_id: Optional[str] = None,
    source_asset_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    asset = PromotedAssetRecord(
        asset_id=_build_item_id("asset"),
        asset_type=asset_type,
        role=role,
        caption=caption.strip(),
        relative_path=relative_path,
        source_project_id=source_project_id,
        source_run_id=source_run_id,
        source_asset_id=source_asset_id,
        tags=_normalize_str_list(tags),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "promoted_assets",
        asset.to_dict(),
        projects_root=projects_root,
    )


def promote_generated_image(
    project_id: str,
    source_image_path: str,
    caption: str,
    role: str = "other",
    source_run_id: Optional[str] = None,
    source_project_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    create_reference: bool = True,
    create_promoted_asset: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_source_image_path = _clean_text(source_image_path)
    if not clean_source_image_path:
        raise ValueError("source_image_path deve ser informado para promover a imagem.")
    if not os.path.exists(clean_source_image_path):
        raise FileNotFoundError(f"Imagem gerada nao encontrada: {clean_source_image_path}")
    if not create_reference and not create_promoted_asset:
        raise ValueError(
            "Selecione pelo menos um destino para promover a imagem: referencia ou promoted_asset."
        )

    clean_caption = _clean_text(caption) or os.path.basename(clean_source_image_path)
    clean_role = _clean_text(role) or "other"
    clean_source_run_id = _clean_text(source_run_id) or None
    clean_source_project_id = _clean_text(source_project_id) or project_id
    clean_tags = _normalize_str_list(tags)
    metadata_payload = _normalize_mapping(metadata, "metadata")

    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    _, original_extension = os.path.splitext(clean_source_image_path)
    extension = original_extension or ".png"
    filename = f"{_slugify_filename(clean_source_image_path)}-{datetime.now().strftime('%Y%m%d%H%M%S')}{extension}"

    promoted_relative_path = None
    if create_promoted_asset:
        promoted_abs_path = os.path.join(paths.promoted_assets_images_dir, filename)
        shutil.copy2(clean_source_image_path, promoted_abs_path)
        promoted_relative_path = os.path.relpath(promoted_abs_path, paths.project_dir)

    reference_relative_path = promoted_relative_path
    if create_reference and not reference_relative_path:
        reference_abs_path = os.path.join(paths.references_images_dir, filename)
        shutil.copy2(clean_source_image_path, reference_abs_path)
        reference_relative_path = os.path.relpath(reference_abs_path, paths.project_dir)

    promoted_asset = None
    reference_asset = None

    if create_promoted_asset:
        promoted_metadata = dict(metadata_payload)
        promoted_metadata.setdefault("promotion_source", "generated_run")
        promoted_asset = add_promoted_asset(
            project_id=project_id,
            asset_type="image",
            role=clean_role,
            caption=clean_caption,
            relative_path=promoted_relative_path,
            source_project_id=clean_source_project_id,
            source_run_id=clean_source_run_id,
            source_asset_id=os.path.basename(clean_source_image_path),
            tags=clean_tags,
            metadata=promoted_metadata,
            projects_root=projects_root,
        )

    if create_reference:
        reference_metadata = dict(metadata_payload)
        reference_metadata.setdefault("promotion_source", "generated_run")
        if promoted_asset:
            reference_metadata["promoted_asset_id"] = promoted_asset["asset_id"]
        reference_asset = add_reference_asset(
            project_id=project_id,
            source_type="image",
            role=clean_role,
            caption=clean_caption,
            relative_path=reference_relative_path,
            origin="project_generated",
            tags=clean_tags,
            source_project_id=clean_source_project_id,
            source_run_id=clean_source_run_id,
            metadata=reference_metadata,
            projects_root=projects_root,
        )

    return {
        "promoted_asset": promoted_asset,
        "reference_asset": reference_asset,
        "copied_relative_path": promoted_relative_path or reference_relative_path,
    }


def add_derived_context(
    project_id: str,
    source_type: str,
    label: str,
    summary: str = "",
    relative_path: Optional[str] = None,
    source_reference_ids: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    context = DerivedContextRecord(
        context_id=_build_item_id("ctx"),
        source_type=source_type,
        label=label.strip(),
        summary=summary.strip(),
        relative_path=relative_path,
        source_reference_ids=_normalize_str_list(source_reference_ids),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "derived_context",
        context.to_dict(),
        projects_root=projects_root,
    )


def add_working_set(
    project_id: str,
    label: str,
    description: str = "",
    reference_ids: Optional[Iterable[str]] = None,
    promoted_asset_ids: Optional[Iterable[str]] = None,
    imported_project_ids: Optional[Iterable[str]] = None,
    derived_context_ids: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    set_active: bool = False,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    working_set = WorkingSetRecord(
        working_set_id=_build_item_id("ws"),
        label=label.strip(),
        description=description.strip(),
        reference_ids=_normalize_str_list(reference_ids),
        promoted_asset_ids=_normalize_str_list(promoted_asset_ids),
        imported_project_ids=_normalize_str_list(imported_project_ids),
        derived_context_ids=_normalize_str_list(derived_context_ids),
        is_active=bool(set_active),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    item = _append_manifest_item(
        project_id,
        "working_sets",
        working_set.to_dict(),
        projects_root=projects_root,
    )

    if set_active:
        set_active_working_set(
            project_id,
            working_set.working_set_id,
            projects_root=projects_root,
        )

    return item


def set_active_visual_bible(
    project_id: str,
    version_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, "visual_bibles", projects_root=projects_root)
    version = _find_manifest_item(manifest, "version_id", version_id)
    if version is None:
        raise ValueError(f"Visual_bible nao encontrado no projeto: {version_id}")

    manifest["active_visual_bible_id"] = version_id
    save_manifest(project_id, "visual_bibles", manifest, projects_root=projects_root)
    update_project(
        project_id,
        projects_root=projects_root,
        active_visual_bible_id=version_id,
    )
    return version


def set_active_working_set(
    project_id: str,
    working_set_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, "working_sets", projects_root=projects_root)
    active_working_set = None
    for item in manifest.get("items", []):
        is_target = item.get("working_set_id") == working_set_id
        item["is_active"] = is_target
        if is_target:
            active_working_set = item

    if active_working_set is None:
        raise ValueError(f"Working set nao encontrado no projeto: {working_set_id}")

    manifest["active_working_set_id"] = working_set_id
    save_manifest(project_id, "working_sets", manifest, projects_root=projects_root)
    update_project(
        project_id,
        projects_root=projects_root,
        active_working_set_id=working_set_id,
    )
    return active_working_set


def load_project_bundle(
    project_id: str,
    projects_root: str = DEFAULT_IMG_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    project = load_project(project_id, projects_root=projects_root)
    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    manifests = {
        manifest_name: load_manifest(project_id, manifest_name, projects_root=projects_root)
        for manifest_name in MANIFEST_ATTRS
    }

    active_visual_bible = _find_manifest_item(
        manifests["visual_bibles"],
        "version_id",
        project.get("active_visual_bible_id") or manifests["visual_bibles"].get("active_visual_bible_id"),
    )
    active_working_set = _find_manifest_item(
        manifests["working_sets"],
        "working_set_id",
        project.get("active_working_set_id") or manifests["working_sets"].get("active_working_set_id"),
    )

    return {
        "project": project,
        "paths": paths.to_dict(),
        "manifests": manifests,
        "active_visual_bible": active_visual_bible,
        "active_working_set": active_working_set,
    }
