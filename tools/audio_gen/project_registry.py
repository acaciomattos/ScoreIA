"""Registry and persistence helpers for phase 7 audio-generation projects."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

from tools.audio_gen.project_paths import (
    DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    ProjectPaths,
    build_project_artifact_paths,
    build_project_paths,
)
from tools.audio_gen.schemas import (
    BeatBlueprintLayerRecord,
    BeatBlueprintRecord,
    CaptureSessionRecord,
    DerivedContextRecord,
    MasterSessionRecord,
    MixSessionRecord,
    ProjectRecord,
    ReferenceAssetRecord,
    WorkingSetRecord,
    build_items_manifest,
)


MANIFEST_ATTRS = {
    "references": "references_manifest",
    "derived_context": "derived_context_manifest",
    "beat_blueprints": "beat_blueprints_manifest",
    "mix_sessions": "mix_sessions_manifest",
    "masters": "masters_manifest",
    "capture_sessions": "capture_sessions_manifest",
    "working_sets": "working_sets_manifest",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "audio-project"


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


def _normalize_float(value: Any) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    return float(text)


def _dedupe_str_list(values: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _build_reference_metadata_filename(provider: str, resource_type: str, provider_id: str) -> str:
    provider_slug = _slugify(provider)
    resource_slug = _slugify(resource_type)
    id_slug = _slugify(provider_id)
    return f"{provider_slug}_{resource_slug}_{id_slug}_metadata.json"


def _canonicalize_youtube_reference(source_url: str) -> Dict[str, Any]:
    parsed = urlparse(_clean_text(source_url))
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    query = parse_qs(parsed.query)

    if host in {"youtu.be"}:
        video_id = path.split("/")[0]
        if not video_id:
            raise ValueError("URL do YouTube invalida: video ID ausente.")
        return {
            "source_type": "youtube_url",
            "provider": "youtube",
            "provider_id": video_id,
            "resource_type": "video",
            "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
            "display_label": f"YouTube video {video_id}",
        }

    if host.endswith("youtube.com"):
        path_parts = [part for part in path.split("/") if part]
        if parsed.path == "/watch":
            video_id = _clean_text((query.get("v") or [""])[0])
            if not video_id:
                raise ValueError("URL do YouTube invalida: parametro 'v' ausente.")
            return {
                "source_type": "youtube_url",
                "provider": "youtube",
                "provider_id": video_id,
                "resource_type": "video",
                "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
                "display_label": f"YouTube video {video_id}",
            }

        if path_parts and path_parts[0] == "playlist":
            playlist_id = _clean_text((query.get("list") or [""])[0])
            if not playlist_id:
                raise ValueError("URL do YouTube invalida: playlist ID ausente.")
            return {
                "source_type": "youtube_url",
                "provider": "youtube",
                "provider_id": playlist_id,
                "resource_type": "playlist",
                "canonical_url": f"https://www.youtube.com/playlist?list={playlist_id}",
                "display_label": f"YouTube playlist {playlist_id}",
            }

        if len(path_parts) >= 2 and path_parts[0] == "shorts":
            video_id = _clean_text(path_parts[1])
            if not video_id:
                raise ValueError("URL do YouTube invalida: short ID ausente.")
            return {
                "source_type": "youtube_url",
                "provider": "youtube",
                "provider_id": video_id,
                "resource_type": "video",
                "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
                "display_label": f"YouTube short {video_id}",
            }

        if len(path_parts) >= 2 and path_parts[0] == "channel":
            channel_id = _clean_text(path_parts[1])
            return {
                "source_type": "youtube_url",
                "provider": "youtube",
                "provider_id": channel_id,
                "resource_type": "channel",
                "canonical_url": f"https://www.youtube.com/channel/{channel_id}",
                "display_label": f"YouTube channel {channel_id}",
            }

        if path_parts and path_parts[0].startswith("@"):
            handle = _clean_text(path_parts[0])
            return {
                "source_type": "youtube_url",
                "provider": "youtube",
                "provider_id": handle.lstrip("@"),
                "resource_type": "channel",
                "canonical_url": f"https://www.youtube.com/{handle}",
                "display_label": f"YouTube creator {handle}",
            }

    raise ValueError("URL do YouTube nao suportada pelo parser atual.")


def _canonicalize_spotify_reference(source_url: str) -> Dict[str, Any]:
    clean_url = _clean_text(source_url)
    if clean_url.lower().startswith("spotify:"):
        uri_parts = clean_url.split(":")
        if len(uri_parts) < 3:
            raise ValueError("URI do Spotify invalida.")
        resource_type = _clean_text(uri_parts[1]).lower()
        provider_id = _clean_text(uri_parts[2])
        if resource_type not in {"track", "album", "artist", "playlist", "show", "episode"}:
            raise ValueError("Tipo de recurso do Spotify nao suportado.")
        return {
            "source_type": "spotify_url",
            "provider": "spotify",
            "provider_id": provider_id,
            "resource_type": resource_type,
            "canonical_url": f"https://open.spotify.com/{resource_type}/{provider_id}",
            "display_label": f"Spotify {resource_type} {provider_id}",
        }

    parsed = urlparse(clean_url)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    if not host.endswith("spotify.com") or len(path_parts) < 2:
        raise ValueError("URL do Spotify nao suportada pelo parser atual.")

    if path_parts[0] not in {"track", "album", "artist", "playlist", "show", "episode"}:
        raise ValueError("Tipo de recurso do Spotify nao suportado.")

    resource_type = _clean_text(path_parts[0]).lower()
    provider_id = _clean_text(path_parts[1])
    if not provider_id:
        raise ValueError("URL do Spotify invalida: provider_id ausente.")

    return {
        "source_type": "spotify_url",
        "provider": "spotify",
        "provider_id": provider_id,
        "resource_type": resource_type,
        "canonical_url": f"https://open.spotify.com/{resource_type}/{provider_id}",
        "display_label": f"Spotify {resource_type} {provider_id}",
    }


def parse_reference_source_url(source_url: str) -> Dict[str, Any]:
    clean_url = _clean_text(source_url)
    if not clean_url:
        raise ValueError("A URL da referencia deve ser informada.")

    parsed = urlparse(clean_url)
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()

    if scheme in {"http", "https"} and (host == "youtu.be" or host.endswith("youtube.com")):
        return _canonicalize_youtube_reference(clean_url)

    if clean_url.lower().startswith("spotify:") or (scheme in {"http", "https"} and host.endswith("spotify.com")):
        return _canonicalize_spotify_reference(clean_url)

    raise ValueError("A URL informada nao pertence a um provedor suportado (YouTube ou Spotify).")


def _infer_reference_caption(
    parsed_reference: Dict[str, Any],
    explicit_caption: str = "",
    imported_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    caption = _clean_text(explicit_caption)
    if caption:
        return caption

    metadata = imported_metadata or {}
    for key in ("title", "name", "label"):
        value = _clean_text(metadata.get(key))
        if value:
            return value

    return _clean_text(parsed_reference.get("display_label")) or "reference-link"


def _default_manifest_payload(project_id: str, manifest_name: str) -> Dict[str, Any]:
    base = build_items_manifest(project_id, manifest_name)
    if manifest_name == "beat_blueprints":
        base["active_beat_blueprint_id"] = None
    if manifest_name == "working_sets":
        base["active_working_set_id"] = None
    return base


def _resolve_manifest_path(paths: ProjectPaths, manifest_name: str) -> str:
    try:
        return getattr(paths, MANIFEST_ATTRS[manifest_name])
    except KeyError as exc:
        raise ValueError(f"Manifesto nao suportado: {manifest_name}") from exc


def _ensure_project_dirs(paths: ProjectPaths) -> None:
    directories = [
        paths.project_dir,
        paths.references_dir,
        paths.references_audio_dir,
        paths.references_text_dir,
        paths.derived_context_dir,
        paths.derived_transcripts_dir,
        paths.derived_analyses_dir,
        paths.beat_blueprints_dir,
        paths.beat_blueprint_versions_dir,
        paths.mix_sessions_dir,
        paths.masters_dir,
        paths.capture_sessions_dir,
        paths.working_sets_dir,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def bootstrap_project_storage(
    project_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
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


def _resolve_existing_project_paths(
    project_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    paths = build_project_paths(project_id, projects_root=projects_root)
    if not os.path.exists(paths.project_file):
        raise FileNotFoundError(f"Projeto nao encontrado: {project_id}")
    return paths


def _ensure_existing_project_storage(
    project_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> ProjectPaths:
    paths = _resolve_existing_project_paths(project_id, projects_root=projects_root)
    return bootstrap_project_storage(paths.project_id, projects_root=projects_root)


def create_project(
    name: str,
    project_type: str = "music_production",
    description: str = "",
    tags: Optional[Iterable[str]] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_name = _clean_text(name)
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
        description=_clean_text(description),
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
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    return _read_json(
        _resolve_existing_project_paths(project_id, projects_root=projects_root).project_file
    )


def save_project(
    project_payload: Dict[str, Any],
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    project_id = _clean_text(project_payload.get("project_id"))
    if not project_id:
        raise ValueError("project_id ausente no payload do projeto.")

    paths = bootstrap_project_storage(project_id, projects_root=projects_root)
    payload = dict(project_payload)
    payload["updated_at"] = _now_iso()
    _write_json(paths.project_file, payload)
    return payload


def update_project(
    project_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    **changes: Any,
) -> Dict[str, Any]:
    project = load_project(project_id, projects_root=projects_root)
    project.update(changes)
    return save_project(project, projects_root=projects_root)


def list_projects(
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
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
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    return _read_json(_resolve_manifest_path(paths, manifest_name))


def save_manifest(
    project_id: str,
    manifest_name: str,
    payload: Dict[str, Any],
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    persisted = dict(payload)
    persisted["project_id"] = project_id
    persisted["manifest_type"] = manifest_name
    _write_json(_resolve_manifest_path(paths, manifest_name), persisted)
    return persisted


def _append_manifest_item(
    project_id: str,
    manifest_name: str,
    item_payload: Dict[str, Any],
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, manifest_name, projects_root=projects_root)
    manifest.setdefault("items", []).append(item_payload)
    save_manifest(project_id, manifest_name, manifest, projects_root=projects_root)
    update_project(project_id, projects_root=projects_root)
    return item_payload


def _find_manifest_item(
    manifest: Dict[str, Any],
    item_key: str,
    item_id: str,
) -> Optional[Dict[str, Any]]:
    for item in manifest.get("items", []):
        if item.get(item_key) == item_id:
            return item
    return None


def _replace_manifest_item(
    project_id: str,
    manifest_name: str,
    item_key: str,
    item_payload: Dict[str, Any],
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, manifest_name, projects_root=projects_root)
    target_id = _clean_text(item_payload.get(item_key))
    if not target_id:
        raise ValueError(f"Item de manifesto sem chave '{item_key}'.")

    replaced = False
    items = manifest.get("items", [])
    for index, current in enumerate(items):
        if _clean_text(current.get(item_key)) == target_id:
            items[index] = item_payload
            replaced = True
            break

    if not replaced:
        raise ValueError(f"Item nao encontrado no manifesto '{manifest_name}': {target_id}")

    manifest["items"] = items
    save_manifest(project_id, manifest_name, manifest, projects_root=projects_root)
    update_project(project_id, projects_root=projects_root)
    return item_payload


def add_reference_asset(
    project_id: str,
    source_type: str,
    role: str,
    caption: str,
    relative_path: Optional[str] = None,
    source_url: Optional[str] = None,
    provider: Optional[str] = None,
    provider_id: Optional[str] = None,
    origin: str = "user_upload",
    tags: Optional[Iterable[str]] = None,
    tempo_bpm: Optional[float] = None,
    key_signature: Optional[str] = None,
    source_project_id: Optional[str] = None,
    source_run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    asset = ReferenceAssetRecord(
        reference_id=_build_item_id("ref"),
        source_type=_clean_text(source_type),
        role=_clean_text(role),
        caption=_clean_text(caption),
        relative_path=relative_path,
        source_url=_clean_text(source_url) or None,
        provider=_clean_text(provider) or None,
        provider_id=_clean_text(provider_id) or None,
        origin=_clean_text(origin) or "user_upload",
        tags=_normalize_str_list(tags),
        tempo_bpm=tempo_bpm,
        key_signature=_clean_text(key_signature) or None,
        source_project_id=_clean_text(source_project_id) or None,
        source_run_id=_clean_text(source_run_id) or None,
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


def update_reference_asset(
    project_id: str,
    reference_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    **changes: Any,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, "references", projects_root=projects_root)
    item = _find_manifest_item(manifest, "reference_id", reference_id)
    if item is None:
        raise ValueError(f"Referencia nao encontrada no projeto: {reference_id}")

    updated = dict(item)
    updated.update(changes)
    updated["reference_id"] = item["reference_id"]
    updated["updated_at"] = _now_iso()
    updated["caption"] = _clean_text(updated.get("caption"))
    updated["role"] = _clean_text(updated.get("role"))
    updated["source_type"] = _clean_text(updated.get("source_type"))
    updated["relative_path"] = _clean_text(updated.get("relative_path")) or None
    updated["source_url"] = _clean_text(updated.get("source_url")) or None
    updated["provider"] = _clean_text(updated.get("provider")) or None
    updated["provider_id"] = _clean_text(updated.get("provider_id")) or None
    updated["origin"] = _clean_text(updated.get("origin")) or "user_upload"
    updated["status"] = _clean_text(updated.get("status")) or "active"
    updated["tempo_bpm"] = _normalize_float(updated.get("tempo_bpm"))
    updated["key_signature"] = _clean_text(updated.get("key_signature")) or None
    updated["source_project_id"] = _clean_text(updated.get("source_project_id")) or None
    updated["source_run_id"] = _clean_text(updated.get("source_run_id")) or None
    updated["tags"] = _normalize_str_list(updated.get("tags"))
    updated["metadata"] = _normalize_mapping(updated.get("metadata"), "metadata")

    return _replace_manifest_item(
        project_id,
        "references",
        "reference_id",
        updated,
        projects_root=projects_root,
    )


def add_reference_link(
    project_id: str,
    source_url: str,
    role: str = "reference",
    caption: str = "",
    origin: str = "external_link",
    tags: Optional[Iterable[str]] = None,
    tempo_bpm: Optional[float] = None,
    key_signature: Optional[str] = None,
    imported_metadata: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    parsed_reference = parse_reference_source_url(source_url)
    project_paths = bootstrap_project_storage(project_id, projects_root=projects_root)

    imported_payload = _normalize_mapping(imported_metadata, "imported_metadata")
    extra_metadata = _normalize_mapping(metadata, "metadata")
    merged_metadata: Dict[str, Any] = {
        "resource_type": parsed_reference["resource_type"],
        "canonical_url": parsed_reference["canonical_url"],
    }
    merged_metadata.update(extra_metadata)

    relative_path = None
    if imported_payload:
        metadata_filename = _build_reference_metadata_filename(
            parsed_reference["provider"],
            parsed_reference["resource_type"],
            parsed_reference["provider_id"],
        )
        stored_metadata_path = os.path.join(project_paths.references_text_dir, metadata_filename)
        _write_json(stored_metadata_path, imported_payload)
        relative_path = os.path.relpath(stored_metadata_path, project_paths.project_dir)
        merged_metadata["imported_metadata_path"] = relative_path
        merged_metadata["imported_metadata_keys"] = sorted(imported_payload.keys())

    normalized_tags = _dedupe_str_list(
        [
            *(tags or []),
            parsed_reference["provider"],
            parsed_reference["resource_type"],
        ]
    )

    return add_reference_asset(
        project_id=project_id,
        source_type=parsed_reference["source_type"],
        role=_clean_text(role) or "reference",
        caption=_infer_reference_caption(
            parsed_reference,
            explicit_caption=caption,
            imported_metadata=imported_payload,
        ),
        relative_path=relative_path,
        source_url=parsed_reference["canonical_url"],
        provider=parsed_reference["provider"],
        provider_id=parsed_reference["provider_id"],
        origin=_clean_text(origin) or "external_link",
        tags=normalized_tags,
        tempo_bpm=tempo_bpm,
        key_signature=key_signature,
        metadata=merged_metadata,
        projects_root=projects_root,
    )


def create_reference_links_working_set(
    project_id: str,
    label: str,
    links: Iterable[Any],
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    set_active: bool = False,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    registered_references: List[Dict[str, Any]] = []
    reference_ids: List[str] = []

    for item in links:
        if isinstance(item, str):
            payload: Dict[str, Any] = {"source_url": item}
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            raise TypeError("Cada item de 'links' deve ser uma string URL ou um dict com os dados da referencia.")

        source_url = _clean_text(payload.get("source_url"))
        if not source_url:
            raise ValueError("Toda referencia de link deve informar 'source_url'.")

        reference = add_reference_link(
            project_id=project_id,
            source_url=source_url,
            role=_clean_text(payload.get("role")) or "reference",
            caption=_clean_text(payload.get("caption")),
            origin=_clean_text(payload.get("origin")) or "external_link",
            tags=payload.get("tags"),
            tempo_bpm=payload.get("tempo_bpm"),
            key_signature=payload.get("key_signature"),
            imported_metadata=payload.get("imported_metadata"),
            metadata=payload.get("metadata"),
            projects_root=projects_root,
        )
        registered_references.append(reference)
        reference_ids.append(reference["reference_id"])

    working_set = add_working_set(
        project_id=project_id,
        label=label,
        description=description,
        reference_ids=reference_ids,
        metadata=metadata,
        set_active=set_active,
        projects_root=projects_root,
    )

    return {
        "working_set": working_set,
        "registered_references": registered_references,
    }


def load_reference_assets(
    project_id: str,
    reference_ids: Optional[Iterable[str]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> List[Dict[str, Any]]:
    manifest = load_manifest(project_id, "references", projects_root=projects_root)
    items = list(manifest.get("items", []))
    normalized_reference_ids = _normalize_str_list(reference_ids)
    if not normalized_reference_ids:
        return items

    selected = []
    selected_ids = set(normalized_reference_ids)
    for item in items:
        if item.get("reference_id") in selected_ids:
            selected.append(item)
    return selected


def find_reference_asset(
    project_id: str,
    reference_id: Optional[str] = None,
    source_url: Optional[str] = None,
    provider: Optional[str] = None,
    provider_id: Optional[str] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Optional[Dict[str, Any]]:
    manifest = load_manifest(project_id, "references", projects_root=projects_root)
    items = list(manifest.get("items", []))

    normalized_reference_id = _clean_text(reference_id)
    if normalized_reference_id:
        return _find_manifest_item(manifest, "reference_id", normalized_reference_id)

    normalized_source_url = _clean_text(source_url)
    normalized_provider = _clean_text(provider).lower()
    normalized_provider_id = _clean_text(provider_id)
    for item in items:
        if normalized_source_url and _clean_text(item.get("source_url")) == normalized_source_url:
            return item
        if normalized_provider and normalized_provider_id:
            if _clean_text(item.get("provider")).lower() == normalized_provider and _clean_text(
                item.get("provider_id")
            ) == normalized_provider_id:
                return item
    return None


def add_derived_context(
    project_id: str,
    context_type: str,
    label: str,
    summary: str = "",
    relative_path: Optional[str] = None,
    source_reference_ids: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    context = DerivedContextRecord(
        context_id=_build_item_id("ctx"),
        context_type=_clean_text(context_type),
        label=_clean_text(label),
        summary=_clean_text(summary),
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


def add_beat_blueprint(
    project_id: str,
    label: str,
    summary: str = "",
    relative_path: Optional[str] = None,
    source_reference_ids: Optional[Iterable[str]] = None,
    style_tags: Optional[Iterable[str]] = None,
    target_bpm: Optional[float] = None,
    bars: int = 4,
    steps_per_bar: int = 16,
    layers: Optional[Iterable[Dict[str, Any]]] = None,
    arrangement_notes: Optional[Iterable[str]] = None,
    mix_notes: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    set_active: bool = True,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    clean_label = _clean_text(label)
    if not clean_label:
        raise ValueError("O beat blueprint deve informar um 'label'.")

    normalized_layers: List[BeatBlueprintLayerRecord] = []
    for item in layers or []:
        payload = dict(item or {})
        normalized_layers.append(
            BeatBlueprintLayerRecord(
                name=_clean_text(payload.get("name")),
                role=_clean_text(payload.get("role")) or "sample",
                pattern=_clean_text(payload.get("pattern")),
                source_reference_ids=_normalize_str_list(payload.get("source_reference_ids")),
                notes=_clean_text(payload.get("notes")),
                metadata=_normalize_mapping(payload.get("metadata"), "layers.metadata"),
            )
        )

    now = _now_iso()
    blueprint = BeatBlueprintRecord(
        version_id=_build_item_id("beat"),
        label=clean_label,
        summary=_clean_text(summary),
        relative_path=relative_path,
        source_reference_ids=_normalize_str_list(source_reference_ids),
        style_tags=_normalize_str_list(style_tags),
        target_bpm=_normalize_float(target_bpm),
        bars=max(1, int(bars)),
        steps_per_bar=max(1, int(steps_per_bar)),
        layers=normalized_layers,
        arrangement_notes=_normalize_str_list(arrangement_notes),
        mix_notes=_normalize_str_list(mix_notes),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )

    item = _append_manifest_item(
        project_id,
        "beat_blueprints",
        blueprint.to_dict(),
        projects_root=projects_root,
    )

    if set_active:
        set_active_beat_blueprint(
            project_id,
            blueprint.version_id,
            projects_root=projects_root,
        )

    return item


def add_mix_session(
    project_id: str,
    label: str,
    artifact_path: Optional[str] = None,
    source_run_id: Optional[str] = None,
    target_peak_db: Optional[float] = None,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    session = MixSessionRecord(
        session_id=_build_item_id("mix"),
        label=_clean_text(label),
        artifact_path=_clean_text(artifact_path) or None,
        source_run_id=_clean_text(source_run_id) or None,
        target_peak_db=_normalize_float(target_peak_db),
        notes=_clean_text(notes),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "mix_sessions",
        session.to_dict(),
        projects_root=projects_root,
    )


def add_master_session(
    project_id: str,
    label: str,
    artifact_path: Optional[str] = None,
    source_run_id: Optional[str] = None,
    target_lufs: Optional[float] = None,
    true_peak_db: Optional[float] = None,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    session = MasterSessionRecord(
        master_id=_build_item_id("master"),
        label=_clean_text(label),
        artifact_path=_clean_text(artifact_path) or None,
        source_run_id=_clean_text(source_run_id) or None,
        target_lufs=_normalize_float(target_lufs),
        true_peak_db=_normalize_float(true_peak_db),
        notes=_clean_text(notes),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "masters",
        session.to_dict(),
        projects_root=projects_root,
    )


def add_working_set(
    project_id: str,
    label: str,
    description: str = "",
    reference_ids: Optional[Iterable[str]] = None,
    capture_ids: Optional[Iterable[str]] = None,
    beat_blueprint_ids: Optional[Iterable[str]] = None,
    derived_context_ids: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    set_active: bool = False,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    working_set = WorkingSetRecord(
        working_set_id=_build_item_id("ws"),
        label=_clean_text(label),
        description=_clean_text(description),
        reference_ids=_normalize_str_list(reference_ids),
        capture_ids=_normalize_str_list(capture_ids),
        beat_blueprint_ids=_normalize_str_list(beat_blueprint_ids),
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


def add_capture_session(
    project_id: str,
    label: str,
    source_reference_id: Optional[str] = None,
    promoted_reference_id: Optional[str] = None,
    source_url: Optional[str] = None,
    provider: Optional[str] = None,
    provider_id: Optional[str] = None,
    device_name: Optional[str] = None,
    source_run_id: Optional[str] = None,
    raw_capture_path: Optional[str] = None,
    clean_capture_path: Optional[str] = None,
    status: str = "captured",
    cleanup_status: str = "pending",
    keep_capture: bool = False,
    expires_at: Optional[str] = None,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    now = _now_iso()
    session = CaptureSessionRecord(
        capture_id=_build_item_id("cap"),
        label=_clean_text(label),
        source_reference_id=_clean_text(source_reference_id) or None,
        promoted_reference_id=_clean_text(promoted_reference_id) or None,
        source_url=_clean_text(source_url) or None,
        provider=_clean_text(provider) or None,
        provider_id=_clean_text(provider_id) or None,
        device_name=_clean_text(device_name) or None,
        source_run_id=_clean_text(source_run_id) or None,
        raw_capture_path=_clean_text(raw_capture_path) or None,
        clean_capture_path=_clean_text(clean_capture_path) or None,
        status=_clean_text(status) or "captured",
        cleanup_status=_clean_text(cleanup_status) or "pending",
        keep_capture=bool(keep_capture),
        expires_at=_clean_text(expires_at) or None,
        notes=_clean_text(notes),
        metadata=_normalize_mapping(metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )
    return _append_manifest_item(
        project_id,
        "capture_sessions",
        session.to_dict(),
        projects_root=projects_root,
    )


def update_capture_session(
    project_id: str,
    capture_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    **changes: Any,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, "capture_sessions", projects_root=projects_root)
    item = _find_manifest_item(manifest, "capture_id", capture_id)
    if item is None:
        raise ValueError(f"Sessao de captura nao encontrada no projeto: {capture_id}")

    updated = dict(item)
    updated.update(changes)
    updated["capture_id"] = item["capture_id"]
    updated["updated_at"] = _now_iso()
    for field_name in (
        "label",
        "source_reference_id",
        "promoted_reference_id",
        "source_url",
        "provider",
        "provider_id",
        "device_name",
        "source_run_id",
        "raw_capture_path",
        "clean_capture_path",
        "status",
        "cleanup_status",
        "expires_at",
        "notes",
    ):
        updated[field_name] = _clean_text(updated.get(field_name)) or None

    updated["label"] = updated["label"] or item.get("label") or "capture"
    updated["status"] = updated["status"] or "captured"
    updated["cleanup_status"] = updated["cleanup_status"] or "pending"
    updated["keep_capture"] = bool(updated.get("keep_capture"))
    updated["metadata"] = _normalize_mapping(updated.get("metadata"), "metadata")

    return _replace_manifest_item(
        project_id,
        "capture_sessions",
        "capture_id",
        updated,
        projects_root=projects_root,
    )


def load_capture_sessions(
    project_id: str,
    capture_ids: Optional[Iterable[str]] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> List[Dict[str, Any]]:
    manifest = load_manifest(project_id, "capture_sessions", projects_root=projects_root)
    items = list(manifest.get("items", []))
    normalized_capture_ids = _normalize_str_list(capture_ids)
    if not normalized_capture_ids:
        return items

    selected_ids = set(normalized_capture_ids)
    return [item for item in items if item.get("capture_id") in selected_ids]


def find_capture_session(
    project_id: str,
    capture_id: Optional[str] = None,
    source_reference_id: Optional[str] = None,
    promoted_reference_id: Optional[str] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Optional[Dict[str, Any]]:
    manifest = load_manifest(project_id, "capture_sessions", projects_root=projects_root)
    items = list(manifest.get("items", []))

    normalized_capture_id = _clean_text(capture_id)
    if normalized_capture_id:
        return _find_manifest_item(manifest, "capture_id", normalized_capture_id)

    normalized_source_reference_id = _clean_text(source_reference_id)
    normalized_promoted_reference_id = _clean_text(promoted_reference_id)
    for item in items:
        if normalized_source_reference_id and _clean_text(item.get("source_reference_id")) == normalized_source_reference_id:
            return item
        if normalized_promoted_reference_id and _clean_text(item.get("promoted_reference_id")) == normalized_promoted_reference_id:
            return item
    return None


def link_external_reference_to_audio(
    project_id: str,
    external_reference_id: str,
    audio_reference_id: str,
    capture_id: Optional[str] = None,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    **metadata_updates: Any,
) -> Dict[str, Any]:
    external_reference = find_reference_asset(
        project_id,
        reference_id=external_reference_id,
        projects_root=projects_root,
    )
    if external_reference is None:
        raise ValueError(f"Referencia externa nao encontrada: {external_reference_id}")

    metadata = dict(external_reference.get("metadata") or {})
    linked_audio_reference_ids = _dedupe_str_list(
        [
            *(metadata.get("linked_audio_reference_ids") or []),
            audio_reference_id,
        ]
    )
    metadata["linked_audio_reference_ids"] = linked_audio_reference_ids
    metadata["linked_audio_reference_id"] = audio_reference_id
    if _clean_text(capture_id):
        metadata["linked_capture_id"] = _clean_text(capture_id)
        metadata["linked_capture_ids"] = _dedupe_str_list(
            [
                *(metadata.get("linked_capture_ids") or []),
                capture_id,
            ]
        )
    metadata.update(metadata_updates)

    return update_reference_asset(
        project_id,
        external_reference_id,
        projects_root=projects_root,
        metadata=metadata,
    )


def set_active_beat_blueprint(
    project_id: str,
    version_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    manifest = load_manifest(project_id, "beat_blueprints", projects_root=projects_root)
    version = _find_manifest_item(manifest, "version_id", version_id)
    if version is None:
        raise ValueError(f"Beat blueprint nao encontrado no projeto: {version_id}")

    manifest["active_beat_blueprint_id"] = version_id
    save_manifest(project_id, "beat_blueprints", manifest, projects_root=projects_root)
    update_project(
        project_id,
        projects_root=projects_root,
        active_beat_blueprint_id=version_id,
    )
    return version


def set_active_working_set(
    project_id: str,
    working_set_id: str,
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
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
    projects_root: str = DEFAULT_AUDIO_GEN_PROJECTS_DIR,
) -> Dict[str, Any]:
    project = load_project(project_id, projects_root=projects_root)
    paths = _ensure_existing_project_storage(project_id, projects_root=projects_root)
    manifests = {
        manifest_name: load_manifest(project_id, manifest_name, projects_root=projects_root)
        for manifest_name in MANIFEST_ATTRS
    }

    active_beat_blueprint = _find_manifest_item(
        manifests["beat_blueprints"],
        "version_id",
        project.get("active_beat_blueprint_id")
        or manifests["beat_blueprints"].get("active_beat_blueprint_id"),
    )
    active_working_set = _find_manifest_item(
        manifests["working_sets"],
        "working_set_id",
        project.get("active_working_set_id")
        or manifests["working_sets"].get("active_working_set_id"),
    )

    return {
        "project": project,
        "paths": paths.to_dict(),
        "manifests": manifests,
        "active_beat_blueprint": active_beat_blueprint,
        "active_working_set": active_working_set,
    }
