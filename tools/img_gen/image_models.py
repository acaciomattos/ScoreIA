"""Registry and download helpers for local image-generation model assets."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from tools.img_gen.model_paths import (
    DEFAULT_COMFYUI_DIR,
    DEFAULT_IMG_GEN_MODELS_DIR,
    ImgGenModelPaths,
    build_model_paths,
)


IMG_GEN_MODEL_REGISTRY_VERSION = "1.0"
DEFAULT_SD15_BASE_REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_SD15_BASE_ALIAS = "sd15_runwayml_base"
DEFAULT_SD15_INPAINTING_REPO_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_SD15_INPAINTING_ALIAS = "sd15_inpainting_base"
DEFAULT_SD15_IP_ADAPTER_ALIAS = "sd15_ip_adapter_base"
DEFAULT_SD15_CONTROLNET_ALIAS = "sd15_controlnet_base"

MODEL_CATEGORY_ATTRS = {
    "checkpoints": "checkpoints_dir",
    "inpainting": "inpainting_dir",
    "controlnet": "controlnet_dir",
    "ip_adapter": "ip_adapter_dir",
    "sam": "sam_dir",
    "upscalers": "upscalers_dir",
    "cache": "cache_dir",
}

DIFFUSERS_SNAPSHOT_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/*",
    "vae/*",
    "feature_extractor/*",
    "safety_checker/*",
]

DIFFUSERS_REQUIRED_ENTRIES = (
    "model_index.json",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "unet",
    "vae",
)

CONTROLNET_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
)

IP_ADAPTER_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", _clean_text(value).lower()).strip("-")
    return slug or "img-model"


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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


def _is_ready_diffusers_snapshot(local_path: str) -> bool:
    clean_local_path = _clean_text(local_path)
    if not clean_local_path or not os.path.isdir(clean_local_path):
        return False

    for entry_name in DIFFUSERS_REQUIRED_ENTRIES:
        if not os.path.exists(os.path.join(clean_local_path, entry_name)):
            return False
    return True


def _find_existing_weight_file(local_path: str, candidates: Iterable[str]) -> Optional[str]:
    clean_local_path = _clean_text(local_path)
    if not clean_local_path:
        return None

    for candidate in candidates:
        clean_candidate = _clean_text(candidate)
        if not clean_candidate:
            continue
        candidate_path = os.path.join(clean_local_path, clean_candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _infer_single_weight_file(local_path: str, subfolder: Optional[str] = None) -> Optional[str]:
    clean_local_path = _clean_text(local_path)
    clean_subfolder = _clean_text(subfolder)
    search_root = (
        os.path.join(clean_local_path, clean_subfolder)
        if clean_subfolder
        else clean_local_path
    )
    if not os.path.isdir(search_root):
        return None

    matches = []
    for entry_name in os.listdir(search_root):
        entry_path = os.path.join(search_root, entry_name)
        if not os.path.isfile(entry_path):
            continue
        if entry_name.lower().endswith(IP_ADAPTER_WEIGHT_EXTENSIONS):
            matches.append(entry_name)
    if len(matches) == 1:
        return matches[0]
    return None


def _read_unet_in_channels(local_path: str) -> Optional[int]:
    config_path = os.path.join(_clean_text(local_path), "unet", "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        config = _read_json(config_path)
    except Exception:
        return None
    value = config.get("in_channels")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_model_artifact_path(
    category: str,
    local_path: str,
    format: str = "diffusers",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clean_category = _clean_text(category)
    clean_format = _clean_text(format) or "diffusers"
    clean_local_path = os.path.abspath(_clean_text(local_path))
    normalized_metadata = _normalize_mapping(metadata, "metadata")

    if not clean_local_path:
        raise ValueError("local_path deve ser informado.")
    if not os.path.exists(clean_local_path):
        raise FileNotFoundError(f"Modelo local nao encontrado: {clean_local_path}")

    inspection: Dict[str, Any] = {
        "category": clean_category,
        "format": clean_format,
        "local_path": clean_local_path,
        "exists": True,
        "is_dir": os.path.isdir(clean_local_path),
        "status": "ready",
        "metadata": dict(normalized_metadata),
    }

    if clean_category == "checkpoints":
        if clean_format != "diffusers":
            raise ValueError("Checkpoints do MVP devem estar em formato diffusers.")
        if not _is_ready_diffusers_snapshot(clean_local_path):
            raise ValueError(
                "Checkpoint diffusers incompleto. Esperado: model_index.json, scheduler/, "
                "text_encoder/, tokenizer/, unet/ e vae/."
            )
        inspection["metadata"]["detected_locally"] = True
        return inspection

    if clean_category == "inpainting":
        if clean_format != "diffusers":
            raise ValueError("Modelos de inpainting do MVP devem estar em formato diffusers.")
        if not _is_ready_diffusers_snapshot(clean_local_path):
            raise ValueError(
                "Modelo de inpainting diffusers incompleto. Esperado: model_index.json, scheduler/, "
                "text_encoder/, tokenizer/, unet/ e vae/."
            )
        in_channels = _read_unet_in_channels(clean_local_path)
        if in_channels is None or in_channels < 9:
            raise ValueError(
                "O modelo informado nao parece ser de inpainting dedicado. "
                "Esperado unet/config.json com in_channels >= 9."
            )
        inspection["metadata"]["unet_in_channels"] = in_channels
        return inspection

    if clean_category == "controlnet":
        if not inspection["is_dir"]:
            raise ValueError("ControlNet deve apontar para um diretorio local.")
        config_path = os.path.join(clean_local_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError("ControlNet diffusers precisa de config.json na raiz do diretorio.")
        weight_path = _find_existing_weight_file(clean_local_path, CONTROLNET_REQUIRED_FILES[1:])
        if weight_path is None:
            raise ValueError(
                "ControlNet diffusers precisa de um arquivo de pesos como "
                "diffusion_pytorch_model.safetensors ou diffusion_pytorch_model.bin."
            )
        inspection["metadata"]["detected_weight_name"] = os.path.basename(weight_path)
        return inspection

    if clean_category == "ip_adapter":
        if not inspection["is_dir"]:
            raise ValueError("IP-Adapter deve apontar para um diretorio local.")
        subfolder = _clean_text(inspection["metadata"].get("subfolder"))
        weight_name = _clean_text(inspection["metadata"].get("weight_name"))
        if not weight_name:
            inferred_weight = _infer_single_weight_file(clean_local_path, subfolder=subfolder)
            if not inferred_weight:
                raise ValueError(
                    "IP-Adapter precisa de metadata.weight_name ou de um unico arquivo de pesos "
                    "(.safetensors/.bin) no diretorio informado."
                )
            weight_name = inferred_weight
            inspection["metadata"]["weight_name"] = weight_name

        weight_relative_path = os.path.join(subfolder, weight_name) if subfolder else weight_name
        weight_path = os.path.join(clean_local_path, weight_relative_path)
        if not os.path.exists(weight_path):
            raise ValueError(
                f"O arquivo de pesos do IP-Adapter nao foi encontrado: {weight_relative_path}."
            )

        image_encoder_folder = inspection["metadata"].get("image_encoder_folder")
        if image_encoder_folder is not None:
            clean_image_encoder_folder = _clean_text(image_encoder_folder)
            if clean_image_encoder_folder:
                inspection["metadata"]["image_encoder_folder"] = clean_image_encoder_folder
                encoder_path = os.path.join(clean_local_path, clean_image_encoder_folder)
                if not os.path.exists(encoder_path):
                    raise ValueError(
                        f"O image_encoder_folder informado nao existe: {clean_image_encoder_folder}."
                    )
            else:
                inspection["metadata"].pop("image_encoder_folder", None)

        default_scale = inspection["metadata"].get("default_scale")
        if default_scale is not None and _clean_text(default_scale):
            try:
                inspection["metadata"]["default_scale"] = float(default_scale)
            except (TypeError, ValueError) as exc:
                raise ValueError("metadata.default_scale do IP-Adapter deve ser numerico.") from exc
        return inspection

    if clean_category in {"sam", "upscalers", "cache"}:
        return inspection

    raise ValueError(f"Categoria de modelo nao suportada: {clean_category}")


def _default_registry_payload() -> Dict[str, Any]:
    return {
        "schema_version": IMG_GEN_MODEL_REGISTRY_VERSION,
        "updated_at": _now_iso(),
        "items": [],
    }


def _resolve_category_dir(paths: ImgGenModelPaths, category: str) -> str:
    attr_name = MODEL_CATEGORY_ATTRS.get(category)
    if not attr_name:
        raise ValueError(f"Categoria de modelo nao suportada: {category}")
    return getattr(paths, attr_name)


def ensure_model_storage(
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> ImgGenModelPaths:
    paths = build_model_paths(models_root=models_root, comfyui_root=comfyui_root)
    directories = [
        paths.models_root,
        paths.checkpoints_dir,
        paths.inpainting_dir,
        paths.controlnet_dir,
        paths.ip_adapter_dir,
        paths.sam_dir,
        paths.upscalers_dir,
        paths.cache_dir,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(paths.registry_file):
        _write_json(paths.registry_file, _default_registry_payload())

    return paths


def load_model_registry(
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    paths = ensure_model_storage(models_root=models_root, comfyui_root=comfyui_root)
    return _read_json(paths.registry_file)


def save_model_registry(
    registry_payload: Dict[str, Any],
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    paths = ensure_model_storage(models_root=models_root, comfyui_root=comfyui_root)
    payload = dict(registry_payload)
    payload["schema_version"] = IMG_GEN_MODEL_REGISTRY_VERSION
    payload["updated_at"] = _now_iso()
    _write_json(paths.registry_file, payload)
    return payload


def list_model_artifacts(
    category: Optional[str] = None,
    status: Optional[str] = None,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> List[Dict[str, Any]]:
    registry = load_model_registry(models_root=models_root, comfyui_root=comfyui_root)
    items = list(registry.get("items", []))
    if category:
        items = [item for item in items if item.get("category") == category]
    if status:
        items = [item for item in items if item.get("status") == status]
    return sorted(
        items,
        key=lambda item: (
            item.get("updated_at", ""),
            item.get("created_at", ""),
            item.get("alias", ""),
        ),
        reverse=True,
    )


def register_model_artifact(
    category: str,
    alias: str,
    repo_id: Optional[str],
    local_path: str,
    status: str = "ready",
    source: str = "huggingface",
    format: str = "diffusers",
    metadata: Optional[Dict[str, Any]] = None,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    clean_category = _clean_text(category)
    _resolve_category_dir(
        build_model_paths(models_root=models_root, comfyui_root=comfyui_root),
        clean_category,
    )

    clean_alias = _clean_text(alias) or _slugify(repo_id)
    clean_repo_id = _clean_text(repo_id)
    clean_local_path = _clean_text(local_path)
    if not clean_alias or not clean_local_path:
        raise ValueError("alias e local_path devem ser informados para registrar um modelo.")
    if not clean_repo_id:
        clean_repo_id = f"local::{clean_alias}"

    registry = load_model_registry(models_root=models_root, comfyui_root=comfyui_root)
    now = _now_iso()
    payload = {
        "alias": clean_alias,
        "category": clean_category,
        "repo_id": clean_repo_id,
        "local_path": clean_local_path,
        "source": _clean_text(source) or "huggingface",
        "format": _clean_text(format) or "diffusers",
        "status": _clean_text(status) or "ready",
        "metadata": _normalize_mapping(metadata, "metadata"),
        "updated_at": now,
    }

    for item in registry.get("items", []):
        if item.get("alias") == clean_alias and item.get("category") == clean_category:
            item.update(payload)
            item.setdefault("created_at", now)
            save_model_registry(
                registry,
                models_root=models_root,
                comfyui_root=comfyui_root,
            )
            return item

    payload["created_at"] = now
    registry.setdefault("items", []).append(payload)
    save_model_registry(
        registry,
        models_root=models_root,
        comfyui_root=comfyui_root,
    )
    return payload


def refresh_model_artifact_status(
    category: str,
    alias: str,
    repo_id: Optional[str] = None,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Optional[Dict[str, Any]]:
    artifact = resolve_model_artifact(
        category=category,
        alias=alias,
        repo_id=repo_id,
        models_root=models_root,
        comfyui_root=comfyui_root,
        refresh=False,
    )
    if not artifact:
        return None

    clean_local_path = _clean_text(artifact.get("local_path"))
    canonical_repo_id = artifact.get("repo_id") or repo_id or ""
    if category == "checkpoints" and alias == DEFAULT_SD15_BASE_ALIAS:
        canonical_repo_id = DEFAULT_SD15_BASE_REPO_ID

    metadata = _normalize_mapping(artifact.get("metadata"), "metadata")
    metadata.pop("error", None)
    try:
        inspection = inspect_model_artifact_path(
            category=artifact.get("category") or category,
            local_path=clean_local_path,
            format=artifact.get("format") or "diffusers",
            metadata=metadata,
        )
    except Exception as exc:
        return register_model_artifact(
            category=artifact.get("category") or category,
            alias=artifact.get("alias") or alias,
            repo_id=canonical_repo_id,
            local_path=clean_local_path,
            status="invalid",
            source=artifact.get("source") or "huggingface",
            format=artifact.get("format") or "diffusers",
            metadata={
                **metadata,
                "error": str(exc),
            },
            models_root=models_root,
            comfyui_root=comfyui_root,
        )

    if inspection.get("status") == "ready":
        return register_model_artifact(
            category=artifact.get("category") or category,
            alias=artifact.get("alias") or alias,
            repo_id=canonical_repo_id,
            local_path=clean_local_path,
            status="ready",
            source=artifact.get("source") or "huggingface",
            format=artifact.get("format") or "diffusers",
            metadata=inspection.get("metadata") or metadata,
            models_root=models_root,
            comfyui_root=comfyui_root,
        )

    return artifact


def refresh_all_model_artifact_statuses(
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> List[Dict[str, Any]]:
    refreshed_items: List[Dict[str, Any]] = []
    for item in list_model_artifacts(
        models_root=models_root,
        comfyui_root=comfyui_root,
    ):
        refreshed = refresh_model_artifact_status(
            category=item.get("category") or "",
            alias=item.get("alias") or "",
            repo_id=item.get("repo_id"),
            models_root=models_root,
            comfyui_root=comfyui_root,
        )
        if refreshed is not None:
            refreshed_items.append(refreshed)
    return refreshed_items


def resolve_model_artifact(
    category: str,
    alias: Optional[str] = None,
    repo_id: Optional[str] = None,
    refresh: bool = True,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Optional[Dict[str, Any]]:
    clean_category = _clean_text(category)
    clean_alias = _clean_text(alias)
    clean_repo_id = _clean_text(repo_id)

    for item in list_model_artifacts(
        category=clean_category,
        models_root=models_root,
        comfyui_root=comfyui_root,
    ):
        if clean_alias and item.get("alias") != clean_alias:
            continue
        if clean_repo_id and item.get("repo_id") != clean_repo_id:
            continue
        resolved_item = item
        if refresh:
            refreshed = refresh_model_artifact_status(
                category=clean_category,
                alias=resolved_item.get("alias") or clean_alias,
                repo_id=resolved_item.get("repo_id") or clean_repo_id,
                models_root=models_root,
                comfyui_root=comfyui_root,
            )
            if refreshed:
                resolved_item = refreshed
        return resolved_item

    if refresh and clean_category == "checkpoints" and clean_alias == DEFAULT_SD15_BASE_ALIAS:
        paths = ensure_model_storage(models_root=models_root, comfyui_root=comfyui_root)
        default_local_path = os.path.join(paths.checkpoints_dir, DEFAULT_SD15_BASE_ALIAS)
        if _is_ready_diffusers_snapshot(default_local_path):
            return register_model_artifact(
                category="checkpoints",
                alias=DEFAULT_SD15_BASE_ALIAS,
                repo_id=clean_repo_id or DEFAULT_SD15_BASE_REPO_ID,
                local_path=default_local_path,
                status="ready",
                metadata={
                    "allow_patterns": list(DIFFUSERS_SNAPSHOT_PATTERNS),
                    "detected_locally": True,
                },
                models_root=models_root,
                comfyui_root=comfyui_root,
            )
    if refresh and clean_category == "inpainting" and clean_alias == DEFAULT_SD15_INPAINTING_ALIAS:
        paths = ensure_model_storage(models_root=models_root, comfyui_root=comfyui_root)
        default_local_path = os.path.join(paths.inpainting_dir, DEFAULT_SD15_INPAINTING_ALIAS)
        try:
            inspection = inspect_model_artifact_path(
                category="inpainting",
                local_path=default_local_path,
                format="diffusers",
                metadata={},
            )
        except Exception:
            inspection = None
        if inspection and inspection.get("status") == "ready":
            return register_model_artifact(
                category="inpainting",
                alias=DEFAULT_SD15_INPAINTING_ALIAS,
                repo_id=clean_repo_id or DEFAULT_SD15_INPAINTING_REPO_ID,
                local_path=default_local_path,
                status="ready",
                metadata={
                    **(inspection.get("metadata") or {}),
                    "detected_locally": True,
                    "allow_patterns": list(DIFFUSERS_SNAPSHOT_PATTERNS),
                },
                models_root=models_root,
                comfyui_root=comfyui_root,
            )
    return None


def resolve_model_source_path(
    category: str,
    alias: Optional[str] = None,
    repo_id: Optional[str] = None,
    required: bool = False,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Optional[str]:
    artifact = resolve_model_artifact(
        category=category,
        alias=alias,
        repo_id=repo_id,
        models_root=models_root,
        comfyui_root=comfyui_root,
    )
    if artifact and os.path.exists(artifact.get("local_path", "")):
        return artifact["local_path"]
    if required:
        raise FileNotFoundError(
            f"Nenhum modelo local encontrado para category='{category}', alias='{alias}', repo_id='{repo_id}'."
        )
    return None


def download_huggingface_snapshot(
    repo_id: str,
    category: str = "checkpoints",
    alias: Optional[str] = None,
    local_dir_name: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
    token: Optional[str] = None,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub nao esta disponivel. Instale a dependencia antes de baixar modelos."
        ) from exc

    paths = ensure_model_storage(models_root=models_root, comfyui_root=comfyui_root)
    category_dir = _resolve_category_dir(paths, category)
    clean_alias = _clean_text(alias) or _slugify(repo_id)
    target_dir = os.path.join(category_dir, _clean_text(local_dir_name) or clean_alias)

    register_model_artifact(
        category=category,
        alias=clean_alias,
        repo_id=repo_id,
        local_path=target_dir,
        status="downloading",
        metadata={
            "allow_patterns": _normalize_str_list(allow_patterns),
            "ignore_patterns": _normalize_str_list(ignore_patterns),
        },
        models_root=models_root,
        comfyui_root=comfyui_root,
    )

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            allow_patterns=_normalize_str_list(allow_patterns) or None,
            ignore_patterns=_normalize_str_list(ignore_patterns) or None,
            token=token,
        )
    except Exception as exc:
        error_payload = register_model_artifact(
            category=category,
            alias=clean_alias,
            repo_id=repo_id,
            local_path=target_dir,
            status="download_error",
            metadata={
                "allow_patterns": _normalize_str_list(allow_patterns),
                "ignore_patterns": _normalize_str_list(ignore_patterns),
                "error": str(exc),
            },
            models_root=models_root,
            comfyui_root=comfyui_root,
        )
        error_payload["error"] = str(exc)
        return error_payload

    return register_model_artifact(
        category=category,
        alias=clean_alias,
        repo_id=repo_id,
        local_path=target_dir,
        status="ready",
        metadata={
            "allow_patterns": _normalize_str_list(allow_patterns),
            "ignore_patterns": _normalize_str_list(ignore_patterns),
        },
        models_root=models_root,
        comfyui_root=comfyui_root,
    )


def download_default_sd15_base_model(
    token: Optional[str] = None,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    return download_huggingface_snapshot(
        repo_id=DEFAULT_SD15_BASE_REPO_ID,
        category="checkpoints",
        alias=DEFAULT_SD15_BASE_ALIAS,
        allow_patterns=DIFFUSERS_SNAPSHOT_PATTERNS,
        token=token,
        models_root=models_root,
        comfyui_root=comfyui_root,
    )


def download_default_sd15_inpainting_model(
    token: Optional[str] = None,
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    return download_huggingface_snapshot(
        repo_id=DEFAULT_SD15_INPAINTING_REPO_ID,
        category="inpainting",
        alias=DEFAULT_SD15_INPAINTING_ALIAS,
        allow_patterns=DIFFUSERS_SNAPSHOT_PATTERNS,
        token=token,
        models_root=models_root,
        comfyui_root=comfyui_root,
    )


def describe_model_storage(
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> Dict[str, Any]:
    paths = ensure_model_storage(models_root=models_root, comfyui_root=comfyui_root)
    registry = load_model_registry(models_root=models_root, comfyui_root=comfyui_root)
    return {
        "paths": paths.to_dict(),
        "registry_items": registry.get("items", []),
    }
