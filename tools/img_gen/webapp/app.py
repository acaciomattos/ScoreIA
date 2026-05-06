import json
import os
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, abort, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from tools.img_gen.image_generation import (
    DEFAULT_IMG2IMG_GUIDANCE,
    DEFAULT_IMG2IMG_IMAGES,
    DEFAULT_IMG2IMG_STEPS,
    DEFAULT_IMG2IMG_STRENGTH,
    DEFAULT_INPAINT_GUIDANCE,
    DEFAULT_INPAINT_IMAGES,
    DEFAULT_INPAINT_MASK_BLUR,
    DEFAULT_INPAINT_STEPS,
    DEFAULT_INPAINT_STRENGTH,
    DEFAULT_TXT2IMG_GUIDANCE,
    DEFAULT_TXT2IMG_IMAGES,
    DEFAULT_TXT2IMG_STEPS,
    generate_img2img,
    generate_inpainting,
    generate_txt2img,
)
from tools.img_gen.image_models import (
    DEFAULT_SD15_BASE_ALIAS,
    DEFAULT_SD15_CONTROLNET_ALIAS,
    DEFAULT_SD15_INPAINTING_ALIAS,
    DEFAULT_SD15_IP_ADAPTER_ALIAS,
    list_model_artifacts,
)
from tools.img_gen.image_models import (
    inspect_model_artifact_path,
    refresh_all_model_artifact_statuses,
    register_model_artifact,
)
from tools.img_gen.model_paths import build_model_paths
from tools.img_gen.project_paths import (
    DEFAULT_IMG_GEN_ARTIFACTS_DIR,
    DEFAULT_IMG_GEN_PROJECTS_DIR,
    build_project_artifact_paths,
    build_project_paths,
)
from tools.img_gen.project_registry import (
    add_derived_context,
    add_imported_project,
    add_reference_asset,
    add_visual_bible_version,
    add_working_set,
    create_project,
    list_projects,
    load_project_bundle,
    promote_generated_image,
    set_active_visual_bible,
    set_active_working_set,
    update_visual_bible_version,
)


app = Flask(__name__)


PROJECT_TYPES = [
    ("single_image", "Imagem pontual"),
    ("story", "Historia"),
    ("ebook", "E-book"),
    ("music_campaign", "Arte para musica"),
    ("visual_set", "Conjunto visual"),
]

REFERENCE_SOURCE_TYPES = [
    ("image", "Imagem"),
    ("text", "Texto"),
    ("audio", "Audio"),
    ("video", "Video"),
]

REFERENCE_ROLES = [
    ("character", "Personagem"),
    ("environment", "Ambiente"),
    ("style", "Estilo"),
    ("composition", "Composicao"),
    ("mood", "Humor"),
    ("product", "Produto"),
    ("other", "Outro"),
]

DERIVED_SOURCE_TYPES = [
    ("text", "Texto"),
    ("audio", "Audio"),
    ("video", "Video"),
]

IMPORT_STRATEGIES = [
    ("pointer_manifest", "Manifesto com ponteiro"),
]

SNAPSHOT_POLICIES = [
    ("promote_on_freeze", "Snapshot ao promover/congelar"),
    ("never_snapshot", "Nunca snapshotar"),
]

SELECTION_MODES = [
    ("explicit_assets", "Selecao explicita"),
    ("active_only", "Somente itens ativos"),
    ("tag_query", "Por tags/papeis"),
]

MODEL_REGISTRATION_CATEGORIES = [
    {"value": "checkpoints", "label": "Checkpoint base"},
    {"value": "inpainting", "label": "Inpainting"},
    {"value": "ip_adapter", "label": "IP-Adapter"},
    {"value": "controlnet", "label": "ControlNet"},
]

MODEL_SOURCES = [
    ("manual_local", "Pasta local manual"),
    ("huggingface", "Snapshot do Hugging Face"),
]

STYLE_PROFILE_FIELDS = [
    {
        "name": "style_keywords",
        "label": "Style keywords",
        "placeholder": "storybook, cinematic, editorial",
    },
    {
        "name": "palette_keywords",
        "label": "Palette keywords",
        "placeholder": "amber, teal, desaturated blue",
    },
    {
        "name": "lighting_keywords",
        "label": "Lighting keywords",
        "placeholder": "soft rim light, moonlight, warm practicals",
    },
    {
        "name": "camera_keywords",
        "label": "Camera keywords",
        "placeholder": "close-up, wide shot, low angle",
    },
    {
        "name": "mood_keywords",
        "label": "Mood keywords",
        "placeholder": "melancholic, adventurous, intimate",
    },
    {
        "name": "render_keywords",
        "label": "Render keywords",
        "placeholder": "illustration, painterly, detailed texture",
    },
    {
        "name": "negative_keywords",
        "label": "Negative keywords",
        "placeholder": "extra fingers, blurry, duplicate face",
    },
]

VISUAL_ENTITY_BUCKETS = [
    {
        "prefix": "character",
        "label": "Characters",
        "singular_label": "Character",
        "summary_placeholder": "Funcao narrativa e identidade visual",
        "attributes_placeholder": "hair=ruivo; eyes=verdes; wardrobe=casaco longo",
    },
    {
        "prefix": "environment",
        "label": "Environments",
        "singular_label": "Environment",
        "summary_placeholder": "Descricao do ambiente e atmosfera",
        "attributes_placeholder": "palette=azul; scale=monumental; weather=mist",
    },
]

VISUAL_ENTITY_SLOT_COUNT = 3
VISUAL_ENTITY_FORM_KEY_MAP = {
    "character": "characters",
    "environment": "environments",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md", ".json"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_optional_identifier(value: Any) -> Optional[str]:
    text = _clean_text(value)
    if not text:
        return None
    if text.lower() in {"none", "null", "undefined"}:
        return None
    return text


def _parse_csv(value: Any) -> List[str]:
    text = _clean_text(value)
    if not text:
        return []
    return [chunk.strip() for chunk in text.split(",") if chunk and chunk.strip()]


def _parse_lines(value: Any) -> List[str]:
    text = _clean_text(value)
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line and line.strip()]


def _parse_bool_token(value: Any) -> bool:
    return _clean_text(value).lower() in {"1", "true", "yes", "y", "on", "sim"}


def _checkbox_enabled(form: Dict[str, Any], field_name: str, default: bool = False) -> bool:
    if field_name not in form:
        return default
    return form.get(field_name) == "on"


def _format_csv(values: Any) -> str:
    if not values:
        return ""
    return ", ".join(str(value).strip() for value in values if str(value).strip())


def _format_lines(values: Any) -> str:
    if not values:
        return ""
    return "\n".join(str(value).strip() for value in values if str(value).strip())


def _parse_int(
    value: Any,
    field_name: str,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
) -> int:
    text = _clean_text(value)
    if not text:
        if default is None:
            raise ValueError(f"O campo '{field_name}' deve ser informado.")
        return default
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"O campo '{field_name}' deve ser um numero inteiro.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"O campo '{field_name}' deve ser maior ou igual a {minimum}.")
    return parsed


def _parse_optional_int(value: Any, field_name: str) -> Optional[int]:
    text = _clean_text(value)
    if not text:
        return None
    return _parse_int(text, field_name)


def _parse_optional_float(value: Any, field_name: str) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    return _parse_float(text, field_name, minimum=0.0)


def _parse_float(
    value: Any,
    field_name: str,
    default: Optional[float] = None,
    minimum: Optional[float] = None,
) -> float:
    text = _clean_text(value)
    if not text:
        if default is None:
            raise ValueError(f"O campo '{field_name}' deve ser informado.")
        return default
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"O campo '{field_name}' deve ser um numero.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"O campo '{field_name}' deve ser maior ou igual a {minimum}.")
    return parsed


def _parse_json_dict(raw_value: Any, field_name: str) -> Dict[str, Any]:
    text = _clean_text(raw_value)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"O campo '{field_name}' deve conter JSON valido.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"O campo '{field_name}' deve conter um objeto JSON.")
    return payload


def _parse_attributes(raw_value: Any) -> Dict[str, Any]:
    text = _clean_text(raw_value)
    if not text:
        return {}

    attributes: Dict[str, Any] = {}
    for chunk in text.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        if "=" in piece:
            key, value = piece.split("=", 1)
        elif ":" in piece:
            key, value = piece.split(":", 1)
        else:
            key, value = piece, "true"
        clean_key = _clean_text(key)
        if clean_key:
            attributes[clean_key] = _clean_text(value)
    return attributes


def _format_attributes(attributes: Any) -> str:
    if not isinstance(attributes, dict) or not attributes:
        return ""

    parts = []
    for key, value in attributes.items():
        clean_key = _clean_text(key)
        clean_value = _clean_text(value)
        if not clean_key:
            continue
        if clean_value in {"", "true"}:
            parts.append(clean_key)
        else:
            parts.append(f"{clean_key}={clean_value}")
    return "; ".join(parts)


def _parse_visual_entities(raw_value: Any, field_name: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for index, line in enumerate(_parse_lines(raw_value), start=1):
        parts = [part.strip() for part in line.split("|")]
        while len(parts) < 6:
            parts.append("")

        label, summary, raw_attrs, raw_refs, raw_tags, raw_locked = parts[:6]
        if not label:
            raise ValueError(
                f"O campo '{field_name}' contem uma linha sem label na posicao {index}."
            )

        entity: Dict[str, Any] = {
            "label": label,
            "summary": summary,
            "attributes": _parse_attributes(raw_attrs),
            "reference_ids": _parse_csv(raw_refs),
            "tags": _parse_csv(raw_tags),
            "locked": _parse_bool_token(raw_locked) if raw_locked else False,
        }
        entities.append(entity)

    return entities


def _parse_visual_entities_from_slots(
    form: Dict[str, Any],
    prefix: str,
    slot_count: int = VISUAL_ENTITY_SLOT_COUNT,
    project_id: Optional[str] = None,
    files: Optional[Any] = None,
    reference_role: str = "other",
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []

    for index in range(slot_count):
        base = f"{prefix}_{index}_"
        label = _clean_text(form.get(f"{base}label"))
        summary = _clean_text(form.get(f"{base}summary"))
        prompt_anchor = _clean_text(form.get(f"{base}prompt_anchor"))
        negative_prompt_anchor = _clean_text(form.get(f"{base}negative_prompt_anchor"))
        notes = _clean_text(form.get(f"{base}notes"))
        attributes = _parse_attributes(form.get(f"{base}attributes"))
        reference_ids = _parse_csv(form.get(f"{base}reference_ids"))
        tags = _parse_csv(form.get(f"{base}tags"))
        locked = form.get(f"{base}locked") == "on"
        uploaded_reference = files.get(f"{base}reference_upload") if files else None
        has_uploaded_reference = bool(
            uploaded_reference and _clean_text(getattr(uploaded_reference, "filename", ""))
        )

        if not any(
            [
                label,
                summary,
                prompt_anchor,
                negative_prompt_anchor,
                notes,
                attributes,
                reference_ids,
                tags,
                locked,
                has_uploaded_reference,
            ]
        ):
            continue

        if not label:
            raise ValueError(
                f"O item {index + 1} de '{prefix}' precisa informar um label."
            )

        if has_uploaded_reference:
            if not project_id:
                raise ValueError(
                    "Abra um projeto antes de enviar imagem de referencia no visual_bible."
                )
            uploaded_relative_path = _save_uploaded_reference_file(
                project_id,
                uploaded_reference,
                "image",
            )
            created_reference = add_reference_asset(
                project_id=project_id,
                source_type="image",
                role=reference_role,
                caption=f"Referencia visual de {label}",
                relative_path=uploaded_relative_path,
                origin="visual_bible_upload",
                tags=list(dict.fromkeys(tags + [prefix, "visual_bible"])),
                metadata={
                    "created_from": "visual_bible_entity_upload",
                    "entity_bucket": prefix,
                    "entity_label": label,
                },
            )
            reference_ids = list(
                dict.fromkeys(reference_ids + [created_reference["reference_id"]])
            )

        entities.append(
            {
                "label": label,
                "summary": summary,
                "prompt_anchor": prompt_anchor,
                "negative_prompt_anchor": negative_prompt_anchor,
                "notes": notes,
                "attributes": attributes,
                "reference_ids": reference_ids,
                "tags": tags,
                "locked": locked,
            }
        )

    return entities


def _build_style_profile(form: Dict[str, Any]) -> Dict[str, Any]:
    raw_json = _clean_text(form.get("visual_bible_style_profile_json"))
    if raw_json:
        return _parse_json_dict(raw_json, "visual_bible_style_profile_json")

    payload: Dict[str, Any] = {}
    for field in STYLE_PROFILE_FIELDS:
        key = field["name"]
        values = _parse_csv(form.get(f"style_profile_{key}"))
        if values:
            payload[key] = values

    metadata = _parse_json_dict(
        form.get("visual_bible_style_profile_metadata_json"),
        "visual_bible_style_profile_metadata_json",
    )
    if metadata:
        payload["metadata"] = metadata
    return payload


def _build_visual_bible_entities(
    form: Dict[str, Any],
    bucket_prefix: str,
    legacy_field_name: str,
    project_id: Optional[str] = None,
    files: Optional[Any] = None,
    reference_role: str = "other",
) -> List[Dict[str, Any]]:
    legacy_value = _clean_text(form.get(legacy_field_name))
    if legacy_value:
        return _parse_visual_entities(legacy_value, legacy_field_name)
    return _parse_visual_entities_from_slots(
        form,
        bucket_prefix,
        project_id=project_id,
        files=files,
        reference_role=reference_role,
    )


def _find_bundle_item(
    bundle: Dict[str, Any],
    manifest_name: str,
    item_key: str,
    item_id: str,
) -> Optional[Dict[str, Any]]:
    if not item_id:
        return None
    for item in bundle["manifests"][manifest_name].get("items", []):
        if item.get(item_key) == item_id:
            return item
    return None


def _build_visual_bible_form_state(
    version: Dict[str, Any],
    is_active: bool = False,
) -> Dict[str, Any]:
    form_state = {
        "visual_bible_label": version.get("label", ""),
        "visual_bible_relative_path": version.get("relative_path") or "",
        "visual_bible_source_reference_ids": _format_csv(version.get("source_reference_ids")),
        "visual_bible_summary": version.get("summary", ""),
        "visual_bible_prompt_anchor": version.get("prompt_anchor", ""),
        "visual_bible_negative_prompt_anchor": version.get("negative_prompt_anchor", ""),
        "visual_bible_composition_rules": _format_lines(version.get("composition_rules")),
        "visual_bible_continuity_rules": _format_lines(version.get("continuity_rules")),
        "visual_bible_notes": version.get("notes", ""),
        "editing_visual_bible_id": version.get("version_id", ""),
    }
    if is_active:
        form_state["visual_bible_set_active"] = "on"

    style_profile = version.get("style_profile") or {}
    for field in STYLE_PROFILE_FIELDS:
        form_state[f"style_profile_{field['name']}"] = _format_csv(
            style_profile.get(field["name"])
        )

    style_profile_metadata = style_profile.get("metadata") or {}
    if style_profile_metadata:
        form_state["visual_bible_style_profile_metadata_json"] = json.dumps(
            style_profile_metadata,
            ensure_ascii=False,
            indent=2,
        )

    version_metadata = version.get("metadata") or {}
    if version_metadata:
        form_state["visual_bible_metadata_json"] = json.dumps(
            version_metadata,
            ensure_ascii=False,
            indent=2,
        )

    for bucket in VISUAL_ENTITY_BUCKETS:
        items = version.get(VISUAL_ENTITY_FORM_KEY_MAP[bucket["prefix"]]) or []
        for index, item in enumerate(items):
            base = f"{bucket['prefix']}_{index}_"
            form_state[f"{base}label"] = item.get("label", "")
            form_state[f"{base}summary"] = item.get("summary", "")
            form_state[f"{base}prompt_anchor"] = item.get("prompt_anchor", "")
            form_state[f"{base}negative_prompt_anchor"] = item.get(
                "negative_prompt_anchor",
                "",
            )
            form_state[f"{base}attributes"] = _format_attributes(item.get("attributes"))
            form_state[f"{base}reference_ids"] = _format_csv(item.get("reference_ids"))
            form_state[f"{base}tags"] = _format_csv(item.get("tags"))
            form_state[f"{base}notes"] = item.get("notes", "")
            if item.get("locked"):
                form_state[f"{base}locked"] = "on"

    return form_state


def _resolve_visual_entity_slot_count(form_state: Dict[str, Any]) -> int:
    max_index = VISUAL_ENTITY_SLOT_COUNT - 1
    for key in form_state:
        for bucket in VISUAL_ENTITY_BUCKETS:
            prefix = f"{bucket['prefix']}_"
            if not key.startswith(prefix):
                continue
            index_text = key[len(prefix) :].split("_", 1)[0]
            if index_text.isdigit():
                max_index = max(max_index, int(index_text))
    return max_index + 1


def _build_import_selectors(form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reference_ids": _parse_csv(form.get("selector_reference_ids")),
        "reference_roles": _parse_csv(form.get("selector_reference_roles")),
        "reference_tags": _parse_csv(form.get("selector_reference_tags")),
        "visual_bible_ids": _parse_csv(form.get("selector_visual_bible_ids")),
        "visual_bible_labels": _parse_csv(form.get("selector_visual_bible_labels")),
        "character_labels": _parse_csv(form.get("selector_character_labels")),
        "environment_labels": _parse_csv(form.get("selector_environment_labels")),
        "prop_labels": _parse_csv(form.get("selector_prop_labels")),
        "promoted_asset_ids": _parse_csv(form.get("selector_promoted_asset_ids")),
        "promoted_asset_roles": _parse_csv(form.get("selector_promoted_asset_roles")),
        "promoted_asset_tags": _parse_csv(form.get("selector_promoted_asset_tags")),
        "derived_context_ids": _parse_csv(form.get("selector_derived_context_ids")),
        "working_set_ids": _parse_csv(form.get("selector_working_set_ids")),
        "include_active_visual_bible": form.get("selector_include_active_visual_bible") == "on",
        "include_active_working_set": form.get("selector_include_active_working_set") == "on",
    }


def _resolve_form_action(form: Dict[str, Any]) -> str:
    return _clean_text(form.get("form_action") or "open_project") or "open_project"


def _resolve_tab_state(form: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        "primary": _clean_optional_identifier(
            form.get("active_primary_tab") or args.get("active_primary_tab")
        ),
        "context": _clean_optional_identifier(
            form.get("active_context_tab") or args.get("active_context_tab")
        ),
        "generation": _clean_optional_identifier(
            form.get("active_generation_tab") or args.get("active_generation_tab")
        ),
    }


def _compact_prompt_text(value: Any, max_words: int = 18) -> str:
    text = _clean_text(value).replace("\r", " ").replace("\n", " ")
    if not text:
        return ""
    words = [chunk for chunk in text.split() if chunk]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _join_prompt_parts(parts: List[str], max_words: Optional[int] = None) -> str:
    normalized: List[str] = []
    seen = set()
    for part in parts:
        text = _clean_text(part)
        if not text:
            continue
        if text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    joined = ", ".join(normalized)
    if max_words is not None:
        return _compact_prompt_text(joined, max_words=max_words)
    return joined


def _format_entity_attributes(attributes: Any) -> str:
    if not isinstance(attributes, dict) or not attributes:
        return ""
    parts: List[str] = []
    for key, value in list(attributes.items())[:4]:
        clean_key = _clean_text(key)
        clean_value = _clean_text(value)
        if not clean_key or not clean_value:
            continue
        parts.append(f"{clean_key} {clean_value}")
    return ", ".join(parts)


def _select_visual_bible_entities(
    items: List[Dict[str, Any]],
    base_prompt: str,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    prompt_lower = _clean_text(base_prompt).lower()
    matched: List[Dict[str, Any]] = []
    locked: List[Dict[str, Any]] = []

    for item in items:
        label = _clean_text(item.get("label")).lower()
        if label and label in prompt_lower:
            matched.append(item)
        elif item.get("locked"):
            locked.append(item)

    if matched:
        return matched + [item for item in locked if item not in matched]
    if len(items) == 1:
        return items
    return locked[:2]


def _build_entity_prompt_fragment(entity_type: str, entity: Dict[str, Any]) -> str:
    label = _clean_text(entity.get("label"))
    summary = _compact_prompt_text(entity.get("summary"), max_words=14)
    prompt_anchor = _compact_prompt_text(entity.get("prompt_anchor"), max_words=10)
    attribute_text = _format_entity_attributes(entity.get("attributes"))
    return _join_prompt_parts(
        [
            f"{entity_type} {label}" if label else "",
            summary,
            prompt_anchor,
            attribute_text,
        ],
        max_words=20,
    )


def _build_reference_lookup(bundle: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not bundle:
        return {}
    return {
        _clean_text(item.get("reference_id")): item
        for item in bundle["manifests"]["references"].get("items", [])
        if _clean_text(item.get("reference_id"))
    }


def _is_image_reference_asset(reference: Dict[str, Any]) -> bool:
    return (
        _clean_text(reference.get("source_type")) == "image"
        and bool(_clean_text(reference.get("relative_path")))
    )


def _flatten_reference_metadata(metadata: Any) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return ""

    parts: List[str] = []
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)):
            text = _clean_text(value)
            if text:
                parts.append(f"{key} {text}")
        if len(parts) >= 4:
            break
    return ", ".join(parts)


def _score_reference_for_entity(
    entity_type: str,
    entity: Dict[str, Any],
    reference: Dict[str, Any],
) -> int:
    if not _is_image_reference_asset(reference):
        return -1

    score = 0
    reference_role = _clean_text(reference.get("role"))
    if reference_role == entity_type:
        score += 30
    elif entity_type == "environment" and reference_role in {"composition", "style"}:
        score += 8

    label = _clean_text(entity.get("label")).lower()
    searchable_text = " ".join(
        [
            _clean_text(reference.get("caption")),
            _format_csv(reference.get("tags")),
            _flatten_reference_metadata(reference.get("metadata")),
            _clean_text(reference.get("relative_path")),
        ]
    ).lower()
    if label and label in searchable_text:
        score += 100

    if _clean_text(reference.get("origin")) == "visual_bible_upload":
        score += 5
    return score


def _resolve_project_relative_image_path(
    project_id: Optional[str],
    relative_path: Any,
) -> Optional[str]:
    clean_project_id = _clean_text(project_id)
    clean_relative_path = _clean_text(relative_path)
    if not clean_project_id or not clean_relative_path:
        return None

    project_paths = build_project_paths(clean_project_id)
    absolute_path = os.path.abspath(
        os.path.join(project_paths.project_dir, clean_relative_path)
    )
    project_root = os.path.abspath(project_paths.project_dir)
    try:
        is_inside_project = os.path.commonpath([absolute_path, project_root]) == project_root
    except ValueError:
        return None
    if not is_inside_project or not os.path.exists(absolute_path):
        return None
    return absolute_path


def _resolve_entity_reference_assets(
    bundle: Optional[Dict[str, Any]],
    entity_type: str,
    entity: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not bundle:
        return []

    reference_lookup = _build_reference_lookup(bundle)
    resolved: List[Dict[str, Any]] = []
    seen_reference_ids = set()

    for reference_id in entity.get("reference_ids") or []:
        reference = reference_lookup.get(_clean_text(reference_id))
        if reference and _is_image_reference_asset(reference):
            clean_reference_id = _clean_text(reference.get("reference_id"))
            if clean_reference_id and clean_reference_id not in seen_reference_ids:
                resolved.append(reference)
                seen_reference_ids.add(clean_reference_id)

    if resolved:
        return resolved[:2]

    candidates: List[Dict[str, Any]] = []
    scored_candidates: List[tuple[int, Dict[str, Any]]] = []
    for reference in bundle["manifests"]["references"].get("items", []):
        if not _is_image_reference_asset(reference):
            continue
        candidates.append(reference)
        score = _score_reference_for_entity(entity_type, entity, reference)
        if score > 0:
            scored_candidates.append((score, reference))

    if scored_candidates:
        ordered = sorted(
            scored_candidates,
            key=lambda item: (
                item[0],
                _clean_text(item[1].get("created_at")),
            ),
            reverse=True,
        )
        deduped: List[Dict[str, Any]] = []
        for _, reference in ordered:
            clean_reference_id = _clean_text(reference.get("reference_id"))
            if clean_reference_id in seen_reference_ids:
                continue
            deduped.append(reference)
            seen_reference_ids.add(clean_reference_id)
            if len(deduped) >= 2:
                break
        if deduped:
            return deduped

    role_matched = [
        reference
        for reference in candidates
        if _clean_text(reference.get("role")) == entity_type
    ]
    if len(role_matched) == 1:
        return role_matched
    return []


def _build_reference_prompt_fragment(
    entity_type: str,
    entity: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> str:
    label = _clean_text(entity.get("label"))
    captions = [_clean_text(reference.get("caption")) for reference in references[:2]]
    metadata_notes = [
        _flatten_reference_metadata(reference.get("metadata"))
        for reference in references[:1]
    ]

    if entity_type == "character":
        reference_lead = (
            f"manter {label} fiel a referencia visual do projeto, mesma identidade, mesmos traços faciais e cabelo"
            if label
            else "manter personagem fiel a referencia visual do projeto, mesma identidade e mesmos traços"
        )
    else:
        reference_lead = (
            f"manter ambiente {label} fiel a referencia visual do projeto"
            if label
            else "manter ambiente fiel a referencia visual do projeto"
        )

    compact_captions: List[str] = []
    for caption in captions:
        clean_caption = _compact_prompt_text(caption, max_words=10)
        lower_caption = clean_caption.lower()
        if any(
            marker in lower_caption
            for marker in ("ao lado", "junto de", "perto de", "segurando", "com ")
        ):
            continue
        if label and label.lower() not in lower_caption:
            continue
        compact_captions.append(clean_caption)
    compact_metadata = [
        _compact_prompt_text(note, max_words=8) for note in metadata_notes
    ]
    return _join_prompt_parts(
        [reference_lead] + compact_captions + compact_metadata,
        max_words=18,
    )


def _resolve_visual_reference_context(
    bundle: Optional[Dict[str, Any]],
    base_prompt: str,
) -> Dict[str, Any]:
    if not bundle:
        return {"prompt_parts": [], "reference_candidates": []}

    visual_bible = bundle.get("active_visual_bible") or {}
    project = bundle.get("project") or {}
    project_id = _clean_text(project.get("project_id"))
    prompt_parts: List[str] = []
    reference_candidates: List[Dict[str, Any]] = []
    seen_reference_ids = set()

    for entity_type, bucket_name in (
        ("character", "characters"),
        ("environment", "environments"),
    ):
        for entity in _select_visual_bible_entities(
            visual_bible.get(bucket_name) or [],
            base_prompt,
        ):
            matched_references = _resolve_entity_reference_assets(
                bundle,
                entity_type,
                entity,
            )
            if not matched_references:
                continue

            prompt_parts.append(
                _build_reference_prompt_fragment(
                    entity_type,
                    entity,
                    matched_references,
                )
            )

            for reference in matched_references:
                reference_id = _clean_text(reference.get("reference_id"))
                if not reference_id or reference_id in seen_reference_ids:
                    continue
                seen_reference_ids.add(reference_id)
                reference_candidates.append(
                    {
                        "entity_type": entity_type,
                        "entity_label": _clean_text(entity.get("label")),
                        "reference_id": reference_id,
                        "caption": _clean_text(reference.get("caption")),
                        "relative_path": _clean_text(reference.get("relative_path")),
                        "absolute_path": _resolve_project_relative_image_path(
                            project_id,
                            reference.get("relative_path"),
                        ),
                    }
                )

    return {
        "prompt_parts": prompt_parts,
        "reference_candidates": reference_candidates,
    }


def _build_visual_bible_prompt_context(
    bundle: Dict[str, Any],
    base_prompt: str,
) -> Dict[str, Any]:
    visual_bible = bundle.get("active_visual_bible") or {}
    style_profile = visual_bible.get("style_profile") or {}
    reference_context = _resolve_visual_reference_context(bundle, base_prompt)
    prompt_parts = [
        "seguir biblia visual do projeto com alta fidelidade",
    ]
    negative_parts = [
        _compact_prompt_text(visual_bible.get("negative_prompt_anchor"), max_words=16),
        _format_csv(style_profile.get("negative_keywords")),
    ]

    for entity in _select_visual_bible_entities(
        visual_bible.get("characters") or [],
        base_prompt,
    ):
        prompt_parts.append(_build_entity_prompt_fragment("character", entity))
        negative_parts.append(_clean_text(entity.get("negative_prompt_anchor")))

    prompt_parts.extend(reference_context["prompt_parts"])

    for entity in _select_visual_bible_entities(
        visual_bible.get("environments") or [],
        base_prompt,
    ):
        prompt_parts.append(_build_entity_prompt_fragment("environment", entity))
        negative_parts.append(_clean_text(entity.get("negative_prompt_anchor")))

    prompt_parts.extend(
        [
            _compact_prompt_text(visual_bible.get("prompt_anchor"), max_words=16),
            _compact_prompt_text(visual_bible.get("summary"), max_words=18),
            _format_csv(style_profile.get("style_keywords")),
            _format_csv(style_profile.get("palette_keywords")),
            _format_csv(style_profile.get("lighting_keywords")),
            _format_csv(style_profile.get("camera_keywords")),
            _format_csv(style_profile.get("mood_keywords")),
            _format_csv(style_profile.get("render_keywords")),
        ]
    )
    prompt_parts.extend((visual_bible.get("composition_rules") or [])[:2])
    prompt_parts.extend((visual_bible.get("continuity_rules") or [])[:2])

    return {
        "prompt_parts": prompt_parts,
        "negative_parts": negative_parts,
        "reference_candidates": reference_context["reference_candidates"],
    }


def _compose_generation_prompts(
    bundle: Optional[Dict[str, Any]],
    prompt: str,
    negative_prompt: str,
    include_visual_bible_context: bool,
) -> Dict[str, str]:
    base_prompt = _clean_text(prompt)
    base_negative = _clean_text(negative_prompt)
    if not include_visual_bible_context or not bundle:
        return {
            "prompt": _join_prompt_parts([base_prompt], max_words=60),
            "negative_prompt": _join_prompt_parts([base_negative], max_words=40),
            "reference_candidates": [],
        }

    visual_context = _build_visual_bible_prompt_context(bundle, base_prompt)

    return {
        "prompt": _join_prompt_parts(
            [base_prompt] + visual_context["prompt_parts"],
            max_words=60,
        ),
        "negative_prompt": _join_prompt_parts(
            [base_negative] + visual_context["negative_parts"],
            max_words=40,
        ),
        "reference_candidates": visual_context["reference_candidates"],
    }


def _resolve_ip_adapter_inputs(
    reference_candidates: List[Dict[str, Any]],
    explicit_alias: Optional[str],
    explicit_image_path: Optional[str],
    explicit_scale: Optional[float],
) -> Dict[str, Any]:
    ready_ip_adapters = list_model_artifacts(category="ip_adapter", status="ready")
    selected_reference = next(
        (
            item
            for item in reference_candidates
            if item.get("entity_type") == "character" and _clean_text(item.get("absolute_path"))
        ),
        None,
    ) or next(
        (
            item
            for item in reference_candidates
            if _clean_text(item.get("absolute_path"))
        ),
        None,
    )

    resolved_alias = _clean_text(explicit_alias) or None
    resolved_image_path = _clean_text(explicit_image_path) or None
    resolved_scale = explicit_scale

    if (
        not resolved_image_path
        and selected_reference is not None
        and (resolved_alias or ready_ip_adapters)
    ):
        resolved_image_path = _clean_text(selected_reference.get("absolute_path")) or None

    if not resolved_alias and resolved_image_path and ready_ip_adapters:
        preferred_alias = next(
            (
                item
                for item in ready_ip_adapters
                if item.get("alias") == DEFAULT_SD15_IP_ADAPTER_ALIAS
            ),
            ready_ip_adapters[0],
        )
        resolved_alias = preferred_alias.get("alias")
        if resolved_scale is None:
            metadata = preferred_alias.get("metadata") or {}
            try:
                resolved_scale = float(metadata.get("default_scale", 0.72))
            except (TypeError, ValueError):
                resolved_scale = 0.72

    if resolved_alias and resolved_scale is None:
        matching_artifact = next(
            (
                item
                for item in ready_ip_adapters
                if item.get("alias") == resolved_alias
            ),
            None,
        )
        if matching_artifact is not None:
            metadata = matching_artifact.get("metadata") or {}
            try:
                resolved_scale = float(metadata.get("default_scale", 0.72))
            except (TypeError, ValueError):
                resolved_scale = 0.72

    return {
        "alias": resolved_alias,
        "image_path": resolved_image_path,
        "scale": resolved_scale,
        "reference_used": selected_reference,
    }


def _safe_root(storage: str) -> Optional[str]:
    roots = {
        "artifact": os.path.abspath(DEFAULT_IMG_GEN_ARTIFACTS_DIR),
        "project": os.path.abspath(DEFAULT_IMG_GEN_PROJECTS_DIR),
    }
    return roots.get(storage)


def _is_safe_under_root(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _build_public_file_ref(abs_path: Optional[str]) -> Optional[Dict[str, str]]:
    clean_path = _clean_text(abs_path)
    if not clean_path or not os.path.exists(clean_path):
        return None

    absolute_path = os.path.abspath(clean_path)
    for storage in ("artifact", "project"):
        root = _safe_root(storage)
        if not root or not _is_safe_under_root(absolute_path, root):
            continue
        relative_path = os.path.relpath(absolute_path, root).replace("\\", "/")
        return {
            "storage": storage,
            "relative_path": relative_path,
            "url": url_for(
                "serve_storage_file",
                storage=storage,
                relative_path=relative_path,
            ),
        }
    return None


def _workspace_relative_path(abs_path: str) -> str:
    workspace_root = os.path.abspath(".")
    absolute_path = os.path.abspath(abs_path)
    if _is_safe_under_root(absolute_path, workspace_root):
        return os.path.relpath(absolute_path, workspace_root).replace("\\", "/")
    return absolute_path


def _resolve_user_image_path(raw_path: Any, field_name: str) -> str:
    clean_path = _clean_text(raw_path)
    if not clean_path:
        raise ValueError(f"O campo '{field_name}' deve apontar para uma imagem.")

    candidates = [
        os.path.abspath(clean_path),
        os.path.abspath(os.path.join(os.getcwd(), clean_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Arquivo nao encontrado para '{field_name}': {clean_path}")


def _resolve_existing_local_path(raw_path: Any, field_name: str) -> str:
    clean_path = _clean_text(raw_path)
    if not clean_path:
        raise ValueError(f"O campo '{field_name}' deve ser informado.")

    candidates = [
        os.path.abspath(clean_path),
        os.path.abspath(os.path.join(os.getcwd(), clean_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Caminho nao encontrado para '{field_name}': {clean_path}")


def _infer_source_type_from_filename(filename: str) -> str:
    extension = os.path.splitext(_clean_text(filename).lower())[1]
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in TEXT_EXTENSIONS:
        return "text"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return "image"


def _save_uploaded_reference_file(
    project_id: str,
    uploaded_file: Any,
    source_type: str,
) -> Optional[str]:
    filename = _clean_text(getattr(uploaded_file, "filename", ""))
    if not filename:
        return None

    project_paths = build_project_paths(project_id)
    clean_source_type = _clean_text(source_type) or _infer_source_type_from_filename(filename)
    if clean_source_type == "text":
        target_dir = project_paths.references_text_dir
    elif clean_source_type == "audio":
        target_dir = project_paths.references_audio_dir
    elif clean_source_type == "video":
        target_dir = project_paths.references_video_dir
    else:
        clean_source_type = "image"
        target_dir = project_paths.references_images_dir

    os.makedirs(target_dir, exist_ok=True)
    safe_name = secure_filename(filename) or f"reference-{uuid.uuid4().hex[:8]}"
    stem, extension = os.path.splitext(safe_name)
    final_name = f"{stem}-{uuid.uuid4().hex[:8]}{extension.lower()}"
    abs_path = os.path.join(target_dir, final_name)
    uploaded_file.save(abs_path)
    return os.path.relpath(abs_path, project_paths.project_dir).replace("\\", "/")


def _save_uploaded_runtime_image(
    project_id: Optional[str],
    uploaded_file: Any,
) -> Optional[str]:
    filename = _clean_text(getattr(uploaded_file, "filename", ""))
    if not filename:
        return None

    artifact_paths = build_project_artifact_paths(project_id or "_ad_hoc")
    target_dir = os.path.join(artifact_paths.project_artifacts_dir, "web_uploads")
    os.makedirs(target_dir, exist_ok=True)
    safe_name = secure_filename(filename) or f"input-{uuid.uuid4().hex[:8]}.png"
    stem, extension = os.path.splitext(safe_name)
    final_name = f"{stem}-{uuid.uuid4().hex[:8]}{extension.lower()}"
    abs_path = os.path.join(target_dir, final_name)
    uploaded_file.save(abs_path)
    return _workspace_relative_path(abs_path)


def _build_model_registration_metadata(form: Dict[str, Any], category: str) -> Dict[str, Any]:
    metadata = _parse_json_dict(
        form.get("model_metadata_json"),
        "model_metadata_json",
    )
    clean_category = _clean_text(category)
    if clean_category == "ip_adapter":
        weight_name = _clean_text(form.get("model_ip_adapter_weight_name"))
        subfolder = _clean_text(form.get("model_ip_adapter_subfolder"))
        image_encoder_folder = _clean_text(form.get("model_ip_adapter_image_encoder_folder"))
        default_scale = _clean_text(form.get("model_ip_adapter_default_scale"))

        if weight_name:
            metadata["weight_name"] = weight_name
        if subfolder:
            metadata["subfolder"] = subfolder
        if image_encoder_folder:
            metadata["image_encoder_folder"] = image_encoder_folder
        if default_scale:
            try:
                metadata["default_scale"] = float(default_scale)
            except ValueError as exc:
                raise ValueError(
                    "O campo 'model_ip_adapter_default_scale' deve ser numerico."
                ) from exc
    return metadata


def _load_generation_manifest(manifest_dir: str) -> Optional[Dict[str, Any]]:
    candidate_names = (
        "generation_manifest.json",
        "txt2img_manifest.json",
        "img2img_manifest.json",
        "inpaint_manifest.json",
    )
    for name in candidate_names:
        candidate_path = os.path.join(manifest_dir, name)
        if not os.path.exists(candidate_path):
            continue
        try:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            continue
    return None


def _build_project_image_candidates(
    bundle: Optional[Dict[str, Any]],
    generation_runs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    seen = set()

    def append_candidate(path: Optional[str], label: str) -> None:
        clean_path = _clean_text(path)
        if not clean_path:
            return
        absolute_path = os.path.abspath(clean_path)
        if absolute_path in seen or not os.path.exists(absolute_path):
            return
        candidates.append(
            {
                "path": _workspace_relative_path(absolute_path),
                "label": label,
            }
        )
        seen.add(absolute_path)

    if bundle:
        project_dir = bundle["paths"]["project_dir"]
        for item in bundle["manifests"]["references"].get("items", []):
            relative_path = _clean_text(item.get("relative_path"))
            if not relative_path:
                continue
            append_candidate(
                os.path.join(project_dir, relative_path),
                f"Referencia: {item.get('caption') or item.get('reference_id')}",
            )
        for item in bundle["manifests"]["promoted_assets"].get("items", []):
            relative_path = _clean_text(item.get("relative_path"))
            if not relative_path:
                continue
            append_candidate(
                os.path.join(project_dir, relative_path),
                f"Promoted: {item.get('caption') or item.get('asset_id')}",
            )

    for run in generation_runs:
        append_candidate(
            run.get("image_path"),
            f"Run {run.get('run_id')} ({run.get('mode')})",
        )

    return candidates


def _build_import_source_image_candidates(
    source_bundle: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not source_bundle:
        return []

    project_dir = source_bundle["paths"]["project_dir"]
    candidates: List[Dict[str, Any]] = []

    def append_candidate(
        item_type: str,
        item_id: str,
        caption: str,
        relative_path: str,
        role: str = "",
    ) -> None:
        absolute_path = os.path.join(project_dir, relative_path)
        public_ref = _build_public_file_ref(absolute_path)
        if not public_ref:
            return
        candidates.append(
            {
                "item_type": item_type,
                "item_id": item_id,
                "caption": caption,
                "role": role,
                "relative_path": relative_path,
                "image_url": public_ref["url"],
            }
        )

    for item in source_bundle["manifests"]["references"].get("items", []):
        if _clean_text(item.get("source_type")) != "image":
            continue
        relative_path = _clean_text(item.get("relative_path"))
        if not relative_path:
            continue
        append_candidate(
            item_type="reference",
            item_id=_clean_text(item.get("reference_id")),
            caption=_clean_text(item.get("caption")) or _clean_text(item.get("reference_id")),
            relative_path=relative_path,
            role=_clean_text(item.get("role")),
        )

    for item in source_bundle["manifests"]["promoted_assets"].get("items", []):
        relative_path = _clean_text(item.get("relative_path"))
        if not relative_path:
            continue
        append_candidate(
            item_type="promoted_asset",
            item_id=_clean_text(item.get("asset_id")),
            caption=_clean_text(item.get("caption")) or _clean_text(item.get("asset_id")),
            relative_path=relative_path,
            role=_clean_text(item.get("role")),
        )

    return candidates


def _load_recent_generation_runs(
    project_id: Optional[str],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    clean_project_id = _clean_text(project_id)
    if not clean_project_id:
        return []

    artifact_paths = build_project_artifact_paths(clean_project_id)
    project_artifacts_dir = artifact_paths.project_artifacts_dir
    if not os.path.exists(project_artifacts_dir):
        return []

    runs: List[Dict[str, Any]] = []
    for entry in os.listdir(project_artifacts_dir):
        run_dir = os.path.join(project_artifacts_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        manifest = _load_generation_manifest(os.path.join(run_dir, "manifests"))
        if manifest is None:
            continue

        outputs = manifest.get("outputs") or {}
        request_payload = manifest.get("request") or {}
        generation_metadata = manifest.get("generation_metadata") or {}
        image_paths = outputs.get("image_paths") or []
        if not image_paths and outputs.get("image_path"):
            image_paths = [outputs.get("image_path")]
        if not image_paths:
            continue

        prompt = _clean_text(request_payload.get("prompt"))
        total_images = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            image_ref = _build_public_file_ref(image_path)
            if image_ref is None:
                continue
            runs.append(
                {
                    "mode": _clean_text(manifest.get("mode") or "txt2img"),
                    "run_id": _clean_text(request_payload.get("run_id")) or entry,
                    "generated_at": _clean_text(manifest.get("generated_at")),
                    "prompt": prompt,
                    "negative_prompt": _clean_text(request_payload.get("negative_prompt")),
                    "prompt_excerpt": (
                        prompt[:140] + "..."
                        if prompt and len(prompt) > 140
                        else prompt
                    ),
                    "seed": manifest.get("seed"),
                    "width": request_payload.get("width"),
                    "height": request_payload.get("height"),
                    "num_inference_steps": request_payload.get("num_inference_steps"),
                    "guidance_scale": request_payload.get("guidance_scale"),
                    "num_images": request_payload.get("num_images") or total_images,
                    "image_index": index,
                    "total_images": total_images,
                    "device_strategy": (manifest.get("runtime") or {}).get("device_strategy"),
                    "generation_metadata": generation_metadata,
                    "safety_checker_mode": _clean_text(
                        generation_metadata.get("safety_checker_mode")
                    ),
                    "nsfw_triggered": bool(generation_metadata.get("nsfw_triggered")),
                    "image_path": image_path,
                    "image_ref": image_ref,
                    "inputs": manifest.get("inputs") or {},
                }
            )

    runs.sort(
        key=lambda item: (
            item.get("generated_at", ""),
            item.get("run_id", ""),
        ),
        reverse=True,
    )
    return runs[:limit]


def _build_generation_form_defaults(
    bundle: Optional[Dict[str, Any]],
    form_state: Dict[str, Any],
) -> Dict[str, Any]:
    resolved = dict(form_state)
    defaults = {
        "generation_model_alias": DEFAULT_SD15_BASE_ALIAS,
        "generation_width": "512",
        "generation_height": "512",
        "generation_num_inference_steps": str(DEFAULT_TXT2IMG_STEPS),
        "generation_guidance_scale": str(DEFAULT_TXT2IMG_GUIDANCE),
        "generation_num_images": str(DEFAULT_TXT2IMG_IMAGES),
        "generation_include_visual_bible_context": "on",
        "generation_safety_checker_enabled": "on",
        "generation_ip_adapter_alias": "",
        "generation_ip_adapter_scale": "0.7",
        "generation_controlnet_alias": "",
        "generation_controlnet_conditioning_scale": "0.8",
        "img2img_model_alias": DEFAULT_SD15_BASE_ALIAS,
        "img2img_width": "512",
        "img2img_height": "512",
        "img2img_num_inference_steps": str(DEFAULT_IMG2IMG_STEPS),
        "img2img_guidance_scale": str(DEFAULT_IMG2IMG_GUIDANCE),
        "img2img_num_images": str(DEFAULT_IMG2IMG_IMAGES),
        "img2img_strength": str(DEFAULT_IMG2IMG_STRENGTH),
        "img2img_include_visual_bible_context": "on",
        "img2img_safety_checker_enabled": "on",
        "img2img_ip_adapter_alias": "",
        "img2img_ip_adapter_scale": "0.7",
        "img2img_controlnet_alias": "",
        "img2img_controlnet_conditioning_scale": "0.8",
        "model_category": "inpainting",
        "model_alias": DEFAULT_SD15_INPAINTING_ALIAS,
        "inpaint_model_alias": DEFAULT_SD15_INPAINTING_ALIAS,
        "inpaint_width": "512",
        "inpaint_height": "512",
        "inpaint_num_inference_steps": str(DEFAULT_INPAINT_STEPS),
        "inpaint_guidance_scale": str(DEFAULT_INPAINT_GUIDANCE),
        "inpaint_num_images": str(DEFAULT_INPAINT_IMAGES),
        "inpaint_strength": str(DEFAULT_INPAINT_STRENGTH),
        "inpaint_mask_blur_radius": str(DEFAULT_INPAINT_MASK_BLUR),
        "inpaint_include_visual_bible_context": "on",
        "inpaint_safety_checker_enabled": "on",
        "promotion_caption": "",
        "promotion_role": "other",
    }
    for key, value in defaults.items():
        resolved.setdefault(key, value)

    if bundle and not _clean_text(resolved.get("generation_prompt")):
        visual_bible = bundle.get("active_visual_bible") or {}
        resolved["generation_prompt"] = _clean_text(visual_bible.get("summary"))
    if bundle and not _clean_text(resolved.get("generation_negative_prompt")):
        visual_bible = bundle.get("active_visual_bible") or {}
        style_profile = visual_bible.get("style_profile") or {}
        resolved["generation_negative_prompt"] = _join_prompt_parts(
            [
                visual_bible.get("negative_prompt_anchor"),
                _format_csv(style_profile.get("negative_keywords")),
            ]
        )
    if bundle and not _clean_text(resolved.get("img2img_negative_prompt")):
        resolved["img2img_negative_prompt"] = resolved.get("generation_negative_prompt", "")
    if bundle and not _clean_text(resolved.get("inpaint_negative_prompt")):
        resolved["inpaint_negative_prompt"] = resolved.get("generation_negative_prompt", "")
    return resolved


def _build_generation_success_message(result: Dict[str, Any], verb: str) -> str:
    message = f"{verb}. Run: {result['run_id']}."
    generation_metadata = result.get("generation_metadata") or {}
    if generation_metadata.get("nsfw_triggered"):
        message += " O safety checker sinalizou NSFW; o preview pode ter sido bloqueado."
    elif generation_metadata.get("safety_checker_mode") == "disabled":
        message += " Safety checker desligado nesta execucao."
    return message


def _resolve_current_project_id(
    projects: List[Dict[str, Any]],
    requested_project_id: Optional[str],
) -> Optional[str]:
    del projects
    return _clean_optional_identifier(requested_project_id)


def _build_page_context(
    current_project_id: Optional[str],
    editing_visual_bible_id: Optional[str] = None,
    success_message: Optional[str] = None,
    error_message: Optional[str] = None,
    form_state: Optional[Dict[str, Any]] = None,
    active_tabs: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    projects = list_projects()
    resolved_project_id = _resolve_current_project_id(projects, current_project_id)
    bundle = None
    resolved_error_message = error_message
    if resolved_project_id:
        try:
            bundle = load_project_bundle(resolved_project_id)
        except FileNotFoundError:
            resolved_error_message = (
                f"{resolved_error_message} "
                if resolved_error_message
                else ""
            ) + f"Projeto nao encontrado: {resolved_project_id}."
            fallback_project_id = _resolve_current_project_id(projects, None)
            if fallback_project_id and fallback_project_id != resolved_project_id:
                resolved_project_id = fallback_project_id
                bundle = load_project_bundle(resolved_project_id)
            else:
                resolved_project_id = None

    available_import_projects = [
        project
        for project in projects
        if project.get("project_id") != resolved_project_id
    ]

    resolved_form_state = dict(form_state or {})
    resolved_editing_visual_bible_id = _clean_text(editing_visual_bible_id)
    editing_visual_bible = None
    if bundle and resolved_editing_visual_bible_id:
        editing_visual_bible = _find_bundle_item(
            bundle,
            "visual_bibles",
            "version_id",
            resolved_editing_visual_bible_id,
        )
        if editing_visual_bible is None:
            resolved_error_message = (
                f"{resolved_error_message} "
                if resolved_error_message
                else ""
            ) + f"Visual_bible nao encontrado: {resolved_editing_visual_bible_id}."
            resolved_editing_visual_bible_id = None
        elif not resolved_form_state:
            resolved_form_state = _build_visual_bible_form_state(
                editing_visual_bible,
                is_active=(
                    bundle["project"].get("active_visual_bible_id")
                    == resolved_editing_visual_bible_id
                ),
            )

    manifest_counts = {}
    if bundle:
        manifest_counts = {
            manifest_name: len(bundle["manifests"][manifest_name].get("items", []))
            for manifest_name in bundle["manifests"]
        }

    generation_runs = _load_recent_generation_runs(resolved_project_id or "_ad_hoc")
    ready_checkpoints = list_model_artifacts(category="checkpoints", status="ready")
    ready_inpainting_models = list_model_artifacts(category="inpainting", status="ready")
    ready_ip_adapters = list_model_artifacts(category="ip_adapter", status="ready")
    ready_controlnets = list_model_artifacts(category="controlnet", status="ready")
    model_artifacts_by_category = {
        category["value"]: list_model_artifacts(category=category["value"])
        for category in MODEL_REGISTRATION_CATEGORIES
    }
    model_storage_paths = build_model_paths().to_dict()
    resolved_form_state = _build_generation_form_defaults(bundle, resolved_form_state)
    project_image_candidates = _build_project_image_candidates(bundle, generation_runs)
    import_source_project_id = _clean_optional_identifier(
        resolved_form_state.get("import_source_project_id")
    )
    import_source_bundle = None
    import_source_image_candidates: List[Dict[str, Any]] = []
    if import_source_project_id and import_source_project_id != resolved_project_id:
        try:
            import_source_bundle = load_project_bundle(import_source_project_id)
            import_source_image_candidates = _build_import_source_image_candidates(
                import_source_bundle
            )
        except FileNotFoundError:
            resolved_error_message = (
                f"{resolved_error_message} " if resolved_error_message else ""
            ) + f"Projeto de importacao nao encontrado: {import_source_project_id}."

    resolved_active_tabs = dict(active_tabs or {})

    return {
        "projects": projects,
        "current_project_id": resolved_project_id,
        "current_bundle": bundle,
        "available_import_projects": available_import_projects,
        "manifest_counts": manifest_counts,
        "success_message": success_message,
        "error_message": resolved_error_message,
        "form_state": resolved_form_state,
        "editing_visual_bible_id": resolved_editing_visual_bible_id,
        "editing_visual_bible": editing_visual_bible,
        "project_types": PROJECT_TYPES,
        "reference_source_types": REFERENCE_SOURCE_TYPES,
        "reference_roles": REFERENCE_ROLES,
        "derived_source_types": DERIVED_SOURCE_TYPES,
        "import_strategies": IMPORT_STRATEGIES,
        "snapshot_policies": SNAPSHOT_POLICIES,
        "selection_modes": SELECTION_MODES,
        "model_registration_categories": MODEL_REGISTRATION_CATEGORIES,
        "model_sources": MODEL_SOURCES,
        "model_artifacts_by_category": model_artifacts_by_category,
        "model_storage_paths": model_storage_paths,
        "default_sd15_base_alias": DEFAULT_SD15_BASE_ALIAS,
        "default_sd15_inpainting_alias": DEFAULT_SD15_INPAINTING_ALIAS,
        "default_sd15_ip_adapter_alias": DEFAULT_SD15_IP_ADAPTER_ALIAS,
        "default_sd15_controlnet_alias": DEFAULT_SD15_CONTROLNET_ALIAS,
        "style_profile_fields": STYLE_PROFILE_FIELDS,
        "visual_entity_buckets": VISUAL_ENTITY_BUCKETS,
        "visual_entity_slot_range": range(
            _resolve_visual_entity_slot_count(resolved_form_state)
        ),
        "ready_checkpoints": ready_checkpoints,
        "ready_inpainting_models": ready_inpainting_models,
        "ready_ip_adapters": ready_ip_adapters,
        "ready_controlnets": ready_controlnets,
        "generation_runs": generation_runs,
        "latest_generation_run": generation_runs[0] if generation_runs else None,
        "project_image_candidates": project_image_candidates,
        "import_source_project_id": import_source_project_id,
        "import_source_bundle": import_source_bundle,
        "import_source_image_candidates": import_source_image_candidates,
        "initial_primary_tab": resolved_active_tabs.get("primary"),
        "initial_context_tab": resolved_active_tabs.get("context"),
        "initial_generation_tab": resolved_active_tabs.get("generation"),
    }


@app.get("/files/<storage>/<path:relative_path>")
def serve_storage_file(storage: str, relative_path: str):
    root = _safe_root(storage)
    if root is None:
        abort(404)

    normalized_relative_path = _clean_text(relative_path).replace("/", os.sep)
    target_path = os.path.abspath(os.path.join(root, normalized_relative_path))
    if not _is_safe_under_root(target_path, root) or not os.path.exists(target_path):
        abort(404)
    return send_file(target_path)


@app.route("/", methods=["GET", "POST"])
def index():
    success_message = None
    error_message = None
    form_state = request.form.to_dict(flat=True) if request.method == "POST" else {}
    active_tabs = _resolve_tab_state(request.form, request.args)

    current_project_id = _clean_optional_identifier(
        request.args.get("project_id")
        or request.form.get("current_project_id")
        or request.form.get("project_id")
    )
    editing_visual_bible_id = (
        request.args.get("editing_visual_bible_id")
        or request.form.get("editing_visual_bible_id")
    )

    if request.method == "POST":
        action = _resolve_form_action(request.form)
        try:
            if action == "open_project":
                current_project_id = _clean_optional_identifier(
                    request.form.get("project_id")
                )
                active_tabs["primary"] = active_tabs.get("primary") or "project"
                editing_visual_bible_id = None
                form_state = {}
                if not current_project_id:
                    raise ValueError("Selecione um projeto para abrir.")
                success_message = "Projeto carregado."

            elif action == "create_project":
                created = create_project(
                    name=request.form.get("project_name"),
                    project_type=request.form.get("project_type") or "single_image",
                    description=request.form.get("project_description") or "",
                    tags=_parse_csv(request.form.get("project_tags")),
                    metadata=_parse_json_dict(
                        request.form.get("project_metadata_json"),
                        "project_metadata_json",
                    ),
                )
                current_project_id = created["project"]["project_id"]
                active_tabs["primary"] = active_tabs.get("primary") or "project"
                editing_visual_bible_id = None
                form_state = {}
                success_message = f"Projeto criado: {created['project']['project_id']}."

            elif action == "register_local_model_artifact":
                category = _clean_text(request.form.get("model_category"))
                alias = _clean_text(request.form.get("model_alias"))
                if not alias:
                    raise ValueError("Informe um alias para o modelo local.")
                local_path = _resolve_existing_local_path(
                    request.form.get("model_local_path"),
                    "model_local_path",
                )
                metadata = _build_model_registration_metadata(request.form, category)
                inspection = inspect_model_artifact_path(
                    category=category,
                    local_path=local_path,
                    format="diffusers",
                    metadata=metadata,
                )
                registered = register_model_artifact(
                    category=category,
                    alias=alias,
                    repo_id=_clean_text(request.form.get("model_repo_id")) or None,
                    local_path=_workspace_relative_path(inspection["local_path"]),
                    status=inspection.get("status") or "ready",
                    source=_clean_text(request.form.get("model_source")) or "manual_local",
                    format="diffusers",
                    metadata=inspection.get("metadata"),
                )
                form_state = {}
                success_message = (
                    f"Modelo registrado: {registered['alias']} "
                    f"({registered['category']}, status={registered['status']})."
                )

            elif action == "refresh_model_registry":
                refreshed_items = refresh_all_model_artifact_statuses()
                success_message = (
                    f"Registry de modelos atualizado. Itens verificados: {len(refreshed_items)}."
                )

            elif action == "generate_txt2img":
                bundle = load_project_bundle(current_project_id) if current_project_id else None
                composed_prompts = _compose_generation_prompts(
                    bundle=bundle,
                    prompt=request.form.get("generation_prompt") or "",
                    negative_prompt=request.form.get("generation_negative_prompt") or "",
                    include_visual_bible_context=_checkbox_enabled(
                        request.form,
                        "generation_include_visual_bible_context",
                        default=True,
                    ),
                )
                ip_adapter_inputs = _resolve_ip_adapter_inputs(
                    reference_candidates=composed_prompts.get("reference_candidates") or [],
                    explicit_alias=_clean_text(request.form.get("generation_ip_adapter_alias")) or None,
                    explicit_image_path=_clean_text(
                        request.form.get("generation_ip_adapter_image_path")
                    )
                    or None,
                    explicit_scale=_parse_optional_float(
                        request.form.get("generation_ip_adapter_scale"),
                        "generation_ip_adapter_scale",
                    ),
                )
                generated = generate_txt2img(
                    prompt=composed_prompts["prompt"],
                    negative_prompt=composed_prompts["negative_prompt"],
                    model_alias=(
                        _clean_text(request.form.get("generation_model_alias"))
                        or DEFAULT_SD15_BASE_ALIAS
                    ),
                    width=_parse_int(
                        request.form.get("generation_width"),
                        "generation_width",
                        default=512,
                        minimum=256,
                    ),
                    height=_parse_int(
                        request.form.get("generation_height"),
                        "generation_height",
                        default=512,
                        minimum=256,
                    ),
                    num_inference_steps=_parse_int(
                        request.form.get("generation_num_inference_steps"),
                        "generation_num_inference_steps",
                        default=DEFAULT_TXT2IMG_STEPS,
                        minimum=1,
                    ),
                    guidance_scale=_parse_float(
                        request.form.get("generation_guidance_scale"),
                        "generation_guidance_scale",
                        default=DEFAULT_TXT2IMG_GUIDANCE,
                        minimum=0.0,
                    ),
                    num_images=_parse_int(
                        request.form.get("generation_num_images"),
                        "generation_num_images",
                        default=DEFAULT_TXT2IMG_IMAGES,
                        minimum=1,
                    ),
                    ip_adapter_alias=ip_adapter_inputs["alias"],
                    ip_adapter_image_path=ip_adapter_inputs["image_path"],
                    ip_adapter_scale=ip_adapter_inputs["scale"],
                    controlnet_alias=_clean_text(request.form.get("generation_controlnet_alias")) or None,
                    control_image_path=_clean_text(request.form.get("generation_control_image_path")) or None,
                    controlnet_conditioning_scale=(
                        _parse_float(
                            request.form.get("generation_controlnet_conditioning_scale"),
                            "generation_controlnet_conditioning_scale",
                            default=0.8,
                            minimum=0.0,
                        )
                        if _clean_text(request.form.get("generation_controlnet_alias"))
                        else None
                    ),
                    safety_checker_enabled=(
                        _checkbox_enabled(
                            request.form,
                            "generation_safety_checker_enabled",
                            default=True,
                        )
                    ),
                    seed=_parse_optional_int(request.form.get("generation_seed"), "generation_seed"),
                    project_id=current_project_id,
                )
                success_message = _build_generation_success_message(
                    generated,
                    "Imagem gerada com sucesso",
                )
                if ip_adapter_inputs.get("reference_used"):
                    success_message += (
                        f" Referencia visual detectada automaticamente para "
                        f"{ip_adapter_inputs['reference_used'].get('entity_label') or 'o projeto'}."
                    )

            elif action == "generate_img2img":
                bundle = load_project_bundle(current_project_id) if current_project_id else None
                uploaded_init_image = request.files.get("img2img_init_image_upload")
                init_image_path = _clean_text(request.form.get("img2img_init_image_path"))
                if uploaded_init_image and _clean_text(uploaded_init_image.filename):
                    init_image_path = _save_uploaded_runtime_image(
                        current_project_id,
                        uploaded_init_image,
                    ) or init_image_path
                composed_prompts = _compose_generation_prompts(
                    bundle=bundle,
                    prompt=request.form.get("img2img_prompt") or "",
                    negative_prompt=request.form.get("img2img_negative_prompt") or "",
                    include_visual_bible_context=_checkbox_enabled(
                        request.form,
                        "img2img_include_visual_bible_context",
                        default=True,
                    ),
                )
                ip_adapter_inputs = _resolve_ip_adapter_inputs(
                    reference_candidates=composed_prompts.get("reference_candidates") or [],
                    explicit_alias=_clean_text(request.form.get("img2img_ip_adapter_alias")) or None,
                    explicit_image_path=_clean_text(
                        request.form.get("img2img_ip_adapter_image_path")
                    )
                    or None,
                    explicit_scale=_parse_optional_float(
                        request.form.get("img2img_ip_adapter_scale"),
                        "img2img_ip_adapter_scale",
                    ),
                )
                generated = generate_img2img(
                    prompt=composed_prompts["prompt"],
                    init_image_path=_resolve_user_image_path(
                        init_image_path,
                        "img2img_init_image_path",
                    ),
                    negative_prompt=composed_prompts["negative_prompt"],
                    model_alias=(
                        _clean_text(request.form.get("img2img_model_alias"))
                        or DEFAULT_SD15_BASE_ALIAS
                    ),
                    width=_parse_optional_int(request.form.get("img2img_width"), "img2img_width"),
                    height=_parse_optional_int(request.form.get("img2img_height"), "img2img_height"),
                    strength=_parse_float(
                        request.form.get("img2img_strength"),
                        "img2img_strength",
                        default=DEFAULT_IMG2IMG_STRENGTH,
                        minimum=0.01,
                    ),
                    num_inference_steps=_parse_int(
                        request.form.get("img2img_num_inference_steps"),
                        "img2img_num_inference_steps",
                        default=DEFAULT_IMG2IMG_STEPS,
                        minimum=1,
                    ),
                    guidance_scale=_parse_float(
                        request.form.get("img2img_guidance_scale"),
                        "img2img_guidance_scale",
                        default=DEFAULT_IMG2IMG_GUIDANCE,
                        minimum=0.0,
                    ),
                    num_images=_parse_int(
                        request.form.get("img2img_num_images"),
                        "img2img_num_images",
                        default=DEFAULT_IMG2IMG_IMAGES,
                        minimum=1,
                    ),
                    ip_adapter_alias=ip_adapter_inputs["alias"],
                    ip_adapter_image_path=ip_adapter_inputs["image_path"],
                    ip_adapter_scale=ip_adapter_inputs["scale"],
                    controlnet_alias=_clean_text(request.form.get("img2img_controlnet_alias")) or None,
                    control_image_path=_clean_text(request.form.get("img2img_control_image_path")) or None,
                    controlnet_conditioning_scale=(
                        _parse_float(
                            request.form.get("img2img_controlnet_conditioning_scale"),
                            "img2img_controlnet_conditioning_scale",
                            default=0.8,
                            minimum=0.0,
                        )
                        if _clean_text(request.form.get("img2img_controlnet_alias"))
                        else None
                    ),
                    safety_checker_enabled=(
                        _checkbox_enabled(
                            request.form,
                            "img2img_safety_checker_enabled",
                            default=True,
                        )
                    ),
                    seed=_parse_optional_int(request.form.get("img2img_seed"), "img2img_seed"),
                    project_id=current_project_id,
                )
                success_message = _build_generation_success_message(
                    generated,
                    "Refinamento img2img concluido",
                )
                if ip_adapter_inputs.get("reference_used"):
                    success_message += (
                        f" Referencia visual detectada automaticamente para "
                        f"{ip_adapter_inputs['reference_used'].get('entity_label') or 'o projeto'}."
                    )

            elif action == "generate_inpainting":
                bundle = load_project_bundle(current_project_id) if current_project_id else None
                uploaded_init_image = request.files.get("inpaint_init_image_upload")
                init_image_path = _clean_text(request.form.get("inpaint_init_image_path"))
                if uploaded_init_image and _clean_text(uploaded_init_image.filename):
                    init_image_path = _save_uploaded_runtime_image(
                        current_project_id,
                        uploaded_init_image,
                    ) or init_image_path
                uploaded_mask_image = request.files.get("inpaint_mask_image_upload")
                mask_image_path = _clean_text(request.form.get("inpaint_mask_image_path"))
                if uploaded_mask_image and _clean_text(uploaded_mask_image.filename):
                    mask_image_path = _save_uploaded_runtime_image(
                        current_project_id,
                        uploaded_mask_image,
                    ) or mask_image_path
                composed_prompts = _compose_generation_prompts(
                    bundle=bundle,
                    prompt=request.form.get("inpaint_prompt") or "",
                    negative_prompt=request.form.get("inpaint_negative_prompt") or "",
                    include_visual_bible_context=_checkbox_enabled(
                        request.form,
                        "inpaint_include_visual_bible_context",
                        default=True,
                    ),
                )
                inpaint_model_alias = _clean_text(
                    request.form.get("inpaint_model_alias")
                ) or DEFAULT_SD15_INPAINTING_ALIAS or _clean_text(request.form.get("img2img_model_alias")) or DEFAULT_SD15_BASE_ALIAS
                generated = generate_inpainting(
                    prompt=composed_prompts["prompt"],
                    init_image_path=_resolve_user_image_path(
                        init_image_path,
                        "inpaint_init_image_path",
                    ),
                    mask_image_path=_resolve_user_image_path(
                        mask_image_path,
                        "inpaint_mask_image_path",
                    ),
                    negative_prompt=composed_prompts["negative_prompt"],
                    model_alias=inpaint_model_alias,
                    width=_parse_optional_int(request.form.get("inpaint_width"), "inpaint_width"),
                    height=_parse_optional_int(request.form.get("inpaint_height"), "inpaint_height"),
                    strength=_parse_float(
                        request.form.get("inpaint_strength"),
                        "inpaint_strength",
                        default=DEFAULT_INPAINT_STRENGTH,
                        minimum=0.01,
                    ),
                    num_inference_steps=_parse_int(
                        request.form.get("inpaint_num_inference_steps"),
                        "inpaint_num_inference_steps",
                        default=DEFAULT_INPAINT_STEPS,
                        minimum=1,
                    ),
                    guidance_scale=_parse_float(
                        request.form.get("inpaint_guidance_scale"),
                        "inpaint_guidance_scale",
                        default=DEFAULT_INPAINT_GUIDANCE,
                        minimum=0.0,
                    ),
                    num_images=_parse_int(
                        request.form.get("inpaint_num_images"),
                        "inpaint_num_images",
                        default=DEFAULT_INPAINT_IMAGES,
                        minimum=1,
                    ),
                    mask_blur_radius=_parse_float(
                        request.form.get("inpaint_mask_blur_radius"),
                        "inpaint_mask_blur_radius",
                        default=DEFAULT_INPAINT_MASK_BLUR,
                        minimum=0.0,
                    ),
                    safety_checker_enabled=(
                        _checkbox_enabled(
                            request.form,
                            "inpaint_safety_checker_enabled",
                            default=True,
                        )
                    ),
                    seed=_parse_optional_int(request.form.get("inpaint_seed"), "inpaint_seed"),
                    project_id=current_project_id,
                )
                success_message = _build_generation_success_message(
                    generated,
                    "Inpainting concluido",
                )

            elif action == "promote_generated_output":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de promover imagens geradas.")
                source_image_path = _clean_text(request.form.get("source_image_path"))
                artifact_root = _safe_root("artifact")
                if not source_image_path or artifact_root is None:
                    raise ValueError("Imagem gerada invalida para promocao.")
                absolute_source_path = os.path.abspath(source_image_path)
                if not _is_safe_under_root(absolute_source_path, artifact_root):
                    raise ValueError("A imagem informada nao pertence ao storage de artifacts.")

                promotion_result = promote_generated_image(
                    project_id=current_project_id,
                    source_image_path=absolute_source_path,
                    caption=(
                        request.form.get("promotion_caption")
                        or request.form.get("fallback_caption")
                        or os.path.basename(absolute_source_path)
                    ),
                    role=request.form.get("promotion_role") or "other",
                    source_run_id=_clean_text(request.form.get("source_run_id")) or None,
                    source_project_id=current_project_id,
                    tags=_parse_csv(request.form.get("promotion_tags")),
                    create_reference=request.form.get("promotion_create_reference") == "on",
                    create_promoted_asset=(
                        request.form.get("promotion_create_promoted_asset") == "on"
                    ),
                    metadata={
                        "generated_image_path": absolute_source_path,
                        "generation_prompt": _clean_text(request.form.get("generation_prompt")),
                    },
                )
                created_labels: List[str] = []
                if promotion_result.get("promoted_asset"):
                    created_labels.append(
                        f"promoted_asset {promotion_result['promoted_asset']['asset_id']}"
                    )
                if promotion_result.get("reference_asset"):
                    created_labels.append(
                        f"referencia {promotion_result['reference_asset']['reference_id']}"
                    )
                success_message = (
                    "Imagem promovida para o projeto: " + ", ".join(created_labels) + "."
                )

            elif action == "add_visual_bible":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de adicionar um visual_bible.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "visual_bible"
                created = add_visual_bible_version(
                    project_id=current_project_id,
                    label=request.form.get("visual_bible_label"),
                    summary=request.form.get("visual_bible_summary") or "",
                    notes=request.form.get("visual_bible_notes") or "",
                    relative_path=_clean_text(request.form.get("visual_bible_relative_path")) or None,
                    source_reference_ids=_parse_csv(
                        request.form.get("visual_bible_source_reference_ids")
                    ),
                    prompt_anchor=request.form.get("visual_bible_prompt_anchor") or "",
                    negative_prompt_anchor=request.form.get(
                        "visual_bible_negative_prompt_anchor"
                    )
                    or "",
                    style_profile=_build_style_profile(request.form),
                    characters=_build_visual_bible_entities(
                        request.form,
                        "character",
                        "visual_bible_characters",
                        project_id=current_project_id,
                        files=request.files,
                        reference_role="character",
                    ),
                    environments=_build_visual_bible_entities(
                        request.form,
                        "environment",
                        "visual_bible_environments",
                    ),
                    props=[],
                    composition_rules=_parse_lines(
                        request.form.get("visual_bible_composition_rules")
                    ),
                    continuity_rules=_parse_lines(
                        request.form.get("visual_bible_continuity_rules")
                    ),
                    metadata=_parse_json_dict(
                        request.form.get("visual_bible_metadata_json"),
                        "visual_bible_metadata_json",
                    ),
                    set_active=request.form.get("visual_bible_set_active") == "on",
                )
                editing_visual_bible_id = None
                form_state = {}
                success_message = f"Visual_bible registrado: {created['version_id']}."

            elif action == "start_edit_visual_bible":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de editar um visual_bible.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "visual_bible"
                editing_visual_bible_id = (
                    request.form.get("editing_visual_bible_id")
                    or request.form.get("version_id")
                )
                if not _clean_text(editing_visual_bible_id):
                    raise ValueError("Selecione uma versao de visual_bible para editar.")
                form_state = {}
                success_message = "Visual_bible carregado para edicao."

            elif action == "cancel_edit_visual_bible":
                active_tabs["primary"] = "context"
                active_tabs["context"] = "visual_bible"
                editing_visual_bible_id = None
                form_state = {}
                success_message = "Edicao de visual_bible cancelada."

            elif action == "update_visual_bible":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de editar um visual_bible.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "visual_bible"
                editing_visual_bible_id = (
                    request.form.get("editing_visual_bible_id")
                    or request.form.get("version_id")
                )
                if not _clean_text(editing_visual_bible_id):
                    raise ValueError("Selecione uma versao de visual_bible para atualizar.")
                updated = update_visual_bible_version(
                    project_id=current_project_id,
                    version_id=editing_visual_bible_id,
                    label=request.form.get("visual_bible_label"),
                    summary=request.form.get("visual_bible_summary") or "",
                    notes=request.form.get("visual_bible_notes") or "",
                    relative_path=_clean_text(request.form.get("visual_bible_relative_path")) or None,
                    source_reference_ids=_parse_csv(
                        request.form.get("visual_bible_source_reference_ids")
                    ),
                    prompt_anchor=request.form.get("visual_bible_prompt_anchor") or "",
                    negative_prompt_anchor=request.form.get(
                        "visual_bible_negative_prompt_anchor"
                    )
                    or "",
                    style_profile=_build_style_profile(request.form),
                    characters=_build_visual_bible_entities(
                        request.form,
                        "character",
                        "visual_bible_characters",
                        project_id=current_project_id,
                        files=request.files,
                        reference_role="character",
                    ),
                    environments=_build_visual_bible_entities(
                        request.form,
                        "environment",
                        "visual_bible_environments",
                    ),
                    props=[],
                    composition_rules=_parse_lines(
                        request.form.get("visual_bible_composition_rules")
                    ),
                    continuity_rules=_parse_lines(
                        request.form.get("visual_bible_continuity_rules")
                    ),
                    metadata=_parse_json_dict(
                        request.form.get("visual_bible_metadata_json"),
                        "visual_bible_metadata_json",
                    ),
                    set_active=request.form.get("visual_bible_set_active") == "on",
                )
                editing_visual_bible_id = updated["version_id"]
                form_state = {}
                success_message = (
                    f"Visual_bible atualizado: {updated['version_id']}."
                )

            elif action == "add_reference_asset":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de adicionar referencias.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "references"
                uploaded_reference = request.files.get("reference_file_upload")
                source_type = request.form.get("reference_source_type") or "image"
                uploaded_relative_path = None
                if uploaded_reference and _clean_text(uploaded_reference.filename):
                    uploaded_relative_path = _save_uploaded_reference_file(
                        current_project_id,
                        uploaded_reference,
                        source_type,
                    )
                created = add_reference_asset(
                    project_id=current_project_id,
                    source_type=source_type,
                    role=request.form.get("reference_role") or "other",
                    caption=request.form.get("reference_caption") or "",
                    relative_path=uploaded_relative_path
                    or _clean_text(request.form.get("reference_relative_path"))
                    or None,
                    origin=request.form.get("reference_origin") or "user_upload",
                    tags=_parse_csv(request.form.get("reference_tags")),
                    source_project_id=_clean_text(
                        request.form.get("reference_source_project_id")
                    )
                    or None,
                    source_run_id=_clean_text(request.form.get("reference_source_run_id"))
                    or None,
                    metadata=_parse_json_dict(
                        request.form.get("reference_metadata_json"),
                        "reference_metadata_json",
                    ),
                )
                success_message = f"Referencia registrada: {created['reference_id']}."

            elif action == "preview_import_source":
                active_tabs["primary"] = "context"
                active_tabs["context"] = "imports"
                success_message = "Projeto de origem carregado para selecao visual."

            elif action == "add_imported_project":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de importar outro projeto.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "imports"
                selected_reference_ids = request.form.getlist("selected_reference_ids")
                selected_promoted_asset_ids = request.form.getlist(
                    "selected_promoted_asset_ids"
                )
                if not selected_reference_ids and not selected_promoted_asset_ids:
                    raise ValueError(
                        "Selecione pelo menos uma imagem do projeto de origem para importar."
                    )
                created = add_imported_project(
                    project_id=current_project_id,
                    source_project_id=request.form.get("import_source_project_id"),
                    alias=request.form.get("import_alias")
                    or request.form.get("import_source_project_id")
                    or "",
                    selection_mode="explicit_assets",
                    selectors={},
                    notes=request.form.get("import_notes") or "",
                    metadata=_parse_json_dict(
                        request.form.get("import_metadata_json"),
                        "import_metadata_json",
                    ),
                    import_strategy="pointer_manifest",
                    snapshot_policy="promote_on_freeze",
                    selected_reference_ids=selected_reference_ids,
                    selected_promoted_asset_ids=selected_promoted_asset_ids,
                )
                success_message = f"Projeto importado registrado: {created['import_id']}."

            elif action == "add_derived_context":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de adicionar contexto derivado.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "imports"
                created = add_derived_context(
                    project_id=current_project_id,
                    source_type=request.form.get("derived_source_type") or "text",
                    label=request.form.get("derived_label") or "",
                    summary=request.form.get("derived_summary") or "",
                    relative_path=_clean_text(request.form.get("derived_relative_path")) or None,
                    source_reference_ids=_parse_csv(
                        request.form.get("derived_source_reference_ids")
                    ),
                    metadata=_parse_json_dict(
                        request.form.get("derived_metadata_json"),
                        "derived_metadata_json",
                    ),
                )
                success_message = f"Contexto derivado registrado: {created['context_id']}."

            elif action == "add_working_set":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de criar um working set.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "library"
                created = add_working_set(
                    project_id=current_project_id,
                    label=request.form.get("working_set_label") or "",
                    description=request.form.get("working_set_description") or "",
                    reference_ids=_parse_csv(request.form.get("working_set_reference_ids")),
                    promoted_asset_ids=_parse_csv(
                        request.form.get("working_set_promoted_asset_ids")
                    ),
                    imported_project_ids=_parse_csv(
                        request.form.get("working_set_imported_project_ids")
                    ),
                    derived_context_ids=_parse_csv(
                        request.form.get("working_set_derived_context_ids")
                    ),
                    metadata=_parse_json_dict(
                        request.form.get("working_set_metadata_json"),
                        "working_set_metadata_json",
                    ),
                    set_active=request.form.get("working_set_set_active") == "on",
                )
                success_message = f"Working set registrado: {created['working_set_id']}."

            elif action == "set_active_visual_bible":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de alterar o visual_bible ativo.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "visual_bible"
                active = set_active_visual_bible(
                    current_project_id,
                    request.form.get("version_id") or "",
                )
                success_message = f"Visual_bible ativo atualizado para {active['version_id']}."

            elif action == "set_active_working_set":
                if not _clean_text(current_project_id):
                    raise ValueError("Abra um projeto antes de alterar o working set ativo.")
                active_tabs["primary"] = "context"
                active_tabs["context"] = "library"
                active = set_active_working_set(
                    current_project_id,
                    request.form.get("working_set_id") or "",
                )
                success_message = (
                    f"Working set ativo atualizado para {active['working_set_id']}."
                )

            else:
                raise ValueError(f"Acao nao suportada: {action}")

        except Exception as exc:
            error_message = str(exc)

    return render_template(
        "index.html",
        **_build_page_context(
            current_project_id=current_project_id,
            editing_visual_bible_id=editing_visual_bible_id,
            success_message=success_message,
            error_message=error_message,
            form_state=form_state,
            active_tabs=active_tabs,
        ),
    )


if __name__ == "__main__":
    app.run(debug=True)
