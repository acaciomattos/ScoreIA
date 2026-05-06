"""Local image-generation runners for txt2img, img2img, and inpainting."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from tools.img_gen.image_models import DEFAULT_SD15_BASE_ALIAS, resolve_model_artifact
from tools.img_gen.project_paths import (
    DEFAULT_IMG_GEN_ARTIFACTS_DIR,
    build_project_artifact_paths,
)
from tools.img_gen.runtime_env import ensure_runtime_ready


DEFAULT_TXT2IMG_STEPS = 18
DEFAULT_TXT2IMG_GUIDANCE = 7.5
DEFAULT_TXT2IMG_IMAGES = 2

DEFAULT_IMG2IMG_STEPS = 20
DEFAULT_IMG2IMG_GUIDANCE = 7.2
DEFAULT_IMG2IMG_STRENGTH = 0.30
DEFAULT_IMG2IMG_IMAGES = 1

DEFAULT_INPAINT_STEPS = 24
DEFAULT_INPAINT_GUIDANCE = 7.5
DEFAULT_INPAINT_STRENGTH = 0.75
DEFAULT_INPAINT_IMAGES = 1
DEFAULT_INPAINT_MASK_BLUR = 6.0

DEFAULT_SCHEDULER_PRESET = "dpmpp_2m_karras"
DEFAULT_QUALITY_PRESET = "sd15_quality"

QUALITY_PROMPT_SUFFIXES = {
    "txt2img": [
        "masterpiece",
        "best quality",
        "highly detailed",
        "sharp focus",
        "coherent composition",
        "cinematic lighting",
        "rich textures",
    ],
    "img2img": [
        "best quality",
        "highly detailed",
        "refined details",
        "sharp focus",
        "clean edges",
        "coherent anatomy",
    ],
    "inpaint": [
        "best quality",
        "highly detailed",
        "seamless integration",
        "matching lighting",
        "matching perspective",
        "clean edges",
    ],
}

QUALITY_NEGATIVE_SUFFIXES = {
    "txt2img": [
        "worst quality",
        "low quality",
        "blurry",
        "deformed",
        "bad anatomy",
        "bad hands",
        "extra fingers",
        "text",
        "watermark",
        "logo",
        "signature",
        "cropped",
        "out of frame",
    ],
    "img2img": [
        "worst quality",
        "low quality",
        "blurry",
        "deformed",
        "bad anatomy",
        "bad hands",
        "extra fingers",
        "text",
        "watermark",
        "logo",
        "signature",
        "oversmoothed",
        "double face",
        "duplicate subject",
    ],
    "inpaint": [
        "worst quality",
        "low quality",
        "blurry",
        "deformed",
        "bad anatomy",
        "bad hands",
        "extra fingers",
        "text",
        "watermark",
        "logo",
        "signature",
        "mismatched edges",
        "visible seam",
        "bad inpainting",
    ],
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _split_prompt_terms(value: str) -> list[str]:
    terms: list[str] = []
    for raw_term in str(value or "").replace("\n", ",").split(","):
        term = raw_term.strip()
        if term:
            terms.append(term)
    return terms


def _merge_prompt_terms(*values: str | list[str]) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        terms = value if isinstance(value, list) else _split_prompt_terms(str(value or ""))
        for term in terms:
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(term)
    return ", ".join(merged)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_dimension(value: int) -> int:
    normalized = max(256, int(value))
    normalized = max(256, (normalized // 8) * 8)
    return normalized


def _resolve_existing_path(path: str) -> str:
    clean_path = _clean_text(path)
    if not clean_path:
        raise ValueError("O caminho de entrada da imagem deve ser informado.")

    candidates = [
        os.path.abspath(clean_path),
        os.path.abspath(os.path.join(os.getcwd(), clean_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Arquivo nao encontrado: {clean_path}")


def _resolve_run_dir(
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
    artifacts_root: str = DEFAULT_IMG_GEN_ARTIFACTS_DIR,
) -> Dict[str, str]:
    resolved_run_id = _clean_text(run_id) or f"run_{_now_stamp()}"
    if project_id:
        project_dir = build_project_artifact_paths(
            project_id,
            artifacts_root=artifacts_root,
        ).project_artifacts_dir
    else:
        project_dir = os.path.join(artifacts_root, "_ad_hoc")

    run_dir = os.path.join(project_dir, resolved_run_id)
    generated_dir = os.path.join(run_dir, "generated")
    manifests_dir = os.path.join(run_dir, "manifests")
    inputs_dir = os.path.join(run_dir, "inputs")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)
    return {
        "project_dir": project_dir,
        "run_id": resolved_run_id,
        "run_dir": run_dir,
        "generated_dir": generated_dir,
        "manifests_dir": manifests_dir,
        "inputs_dir": inputs_dir,
    }


@dataclass
class Txt2ImgRequest:
    prompt: str
    negative_prompt: str = ""
    model_alias: str = DEFAULT_SD15_BASE_ALIAS
    width: int = 512
    height: int = 512
    num_inference_steps: int = DEFAULT_TXT2IMG_STEPS
    guidance_scale: float = DEFAULT_TXT2IMG_GUIDANCE
    num_images: int = DEFAULT_TXT2IMG_IMAGES
    ip_adapter_alias: Optional[str] = None
    ip_adapter_image_path: Optional[str] = None
    ip_adapter_scale: Optional[float] = None
    controlnet_alias: Optional[str] = None
    control_image_path: Optional[str] = None
    controlnet_conditioning_scale: Optional[float] = None
    safety_checker_enabled: bool = True
    quality_preset: str = DEFAULT_QUALITY_PRESET
    seed: Optional[int] = None
    project_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Img2ImgRequest:
    prompt: str
    init_image_path: str
    negative_prompt: str = ""
    model_alias: str = DEFAULT_SD15_BASE_ALIAS
    width: Optional[int] = None
    height: Optional[int] = None
    strength: float = DEFAULT_IMG2IMG_STRENGTH
    num_inference_steps: int = DEFAULT_IMG2IMG_STEPS
    guidance_scale: float = DEFAULT_IMG2IMG_GUIDANCE
    num_images: int = DEFAULT_IMG2IMG_IMAGES
    ip_adapter_alias: Optional[str] = None
    ip_adapter_image_path: Optional[str] = None
    ip_adapter_scale: Optional[float] = None
    controlnet_alias: Optional[str] = None
    control_image_path: Optional[str] = None
    controlnet_conditioning_scale: Optional[float] = None
    safety_checker_enabled: bool = True
    quality_preset: str = DEFAULT_QUALITY_PRESET
    seed: Optional[int] = None
    project_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InpaintRequest:
    prompt: str
    init_image_path: str
    mask_image_path: str
    negative_prompt: str = ""
    model_alias: str = DEFAULT_SD15_BASE_ALIAS
    width: Optional[int] = None
    height: Optional[int] = None
    strength: float = DEFAULT_INPAINT_STRENGTH
    num_inference_steps: int = DEFAULT_INPAINT_STEPS
    guidance_scale: float = DEFAULT_INPAINT_GUIDANCE
    num_images: int = DEFAULT_INPAINT_IMAGES
    safety_checker_enabled: bool = True
    quality_preset: str = DEFAULT_QUALITY_PRESET
    mask_blur_radius: float = DEFAULT_INPAINT_MASK_BLUR
    seed: Optional[int] = None
    project_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_text_prompt(prompt: str) -> None:
    if not _clean_text(prompt):
        raise ValueError("O prompt deve ser informado para gerar imagem.")


def _validate_canvas_size(width: int, height: int) -> Tuple[int, int]:
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("width e height devem ser multiplos de 8.")
    if width < 256 or height < 256:
        raise ValueError("width e height devem ser no minimo 256.")
    return width, height


def _validate_txt2img_request(request: Txt2ImgRequest) -> Txt2ImgRequest:
    _validate_text_prompt(request.prompt)
    _validate_canvas_size(request.width, request.height)
    if int(request.num_images) < 1:
        raise ValueError("num_images deve ser maior ou igual a 1.")
    _validate_optional_reference_controls(
        request.ip_adapter_alias,
        request.ip_adapter_image_path,
        request.ip_adapter_scale,
        request.controlnet_alias,
        request.control_image_path,
        request.controlnet_conditioning_scale,
    )
    return request


def _validate_image_mode_request(
    prompt: str,
    image_path: str,
    width: Optional[int],
    height: Optional[int],
    strength: float,
) -> Tuple[str, Optional[int], Optional[int], float]:
    _validate_text_prompt(prompt)
    _resolve_existing_path(image_path)
    if (width is None) != (height is None):
        raise ValueError("width e height devem ser informados juntos para img2img/inpainting.")
    if width is not None and height is not None:
        _validate_canvas_size(width, height)
    if not (0 < float(strength) <= 1):
        raise ValueError("strength deve estar no intervalo (0, 1].")
    return image_path, width, height, float(strength)


def _validate_optional_reference_controls(
    ip_adapter_alias: Optional[str],
    ip_adapter_image_path: Optional[str],
    ip_adapter_scale: Optional[float],
    controlnet_alias: Optional[str],
    control_image_path: Optional[str],
    controlnet_conditioning_scale: Optional[float],
) -> None:
    clean_ip_alias = _clean_text(ip_adapter_alias)
    clean_ip_image_path = _clean_text(ip_adapter_image_path)
    clean_control_alias = _clean_text(controlnet_alias)
    clean_control_image_path = _clean_text(control_image_path)

    if clean_ip_alias and not clean_ip_image_path:
        raise ValueError("Informe ip_adapter_image_path quando usar um ip_adapter_alias.")
    if clean_ip_image_path and not clean_ip_alias:
        raise ValueError("Informe ip_adapter_alias quando usar uma imagem de referencia do IP-Adapter.")
    if clean_ip_image_path:
        _resolve_existing_path(clean_ip_image_path)
    if ip_adapter_scale is not None and float(ip_adapter_scale) < 0:
        raise ValueError("ip_adapter_scale deve ser maior ou igual a 0.")

    if clean_control_alias and not clean_control_image_path:
        raise ValueError("Informe control_image_path quando usar um controlnet_alias.")
    if clean_control_image_path and not clean_control_alias:
        raise ValueError("Informe controlnet_alias quando usar control_image_path.")
    if clean_control_image_path:
        _resolve_existing_path(clean_control_image_path)
    if controlnet_conditioning_scale is not None and float(controlnet_conditioning_scale) < 0:
        raise ValueError("controlnet_conditioning_scale deve ser maior ou igual a 0.")


def _validate_img2img_request(request: Img2ImgRequest) -> Img2ImgRequest:
    _validate_image_mode_request(
        request.prompt,
        request.init_image_path,
        request.width,
        request.height,
        request.strength,
    )
    _validate_optional_reference_controls(
        request.ip_adapter_alias,
        request.ip_adapter_image_path,
        request.ip_adapter_scale,
        request.controlnet_alias,
        request.control_image_path,
        request.controlnet_conditioning_scale,
    )
    if int(request.num_images) < 1:
        raise ValueError("num_images deve ser maior ou igual a 1.")
    return request


def _validate_inpaint_request(request: InpaintRequest) -> InpaintRequest:
    _validate_image_mode_request(
        request.prompt,
        request.init_image_path,
        request.width,
        request.height,
        request.strength,
    )
    _resolve_existing_path(request.mask_image_path)
    if int(request.num_images) < 1:
        raise ValueError("num_images deve ser maior ou igual a 1.")
    return request


def _prepare_runtime(model_alias: str, model_category: str = "checkpoints"):
    runtime = ensure_runtime_ready(
        model_alias=model_alias,
        model_category=model_category,
    )
    model_artifact = runtime["model_artifact"]
    model_path = model_artifact["local_path"]

    import torch

    device = "cuda" if runtime["cuda_available"] else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    return runtime, model_artifact, model_path, torch, device, torch_dtype


def _resolve_quality_preset(
    quality_preset: Optional[str],
) -> Optional[str]:
    clean_preset = _clean_text(quality_preset)
    if not clean_preset:
        return DEFAULT_QUALITY_PRESET
    if clean_preset.lower() in {"none", "off", "disabled"}:
        return None
    return clean_preset


def _prepare_effective_prompts(
    mode: str,
    prompt: str,
    negative_prompt: str,
    quality_preset: Optional[str],
) -> Dict[str, Any]:
    resolved_preset = _resolve_quality_preset(quality_preset)
    base_prompt = _clean_text(prompt)
    base_negative_prompt = _clean_text(negative_prompt)

    if not resolved_preset:
        return {
            "quality_preset": None,
            "user_prompt": base_prompt,
            "user_negative_prompt": base_negative_prompt,
            "effective_prompt": base_prompt,
            "effective_negative_prompt": base_negative_prompt,
            "positive_suffix": [],
            "negative_suffix": [],
        }

    positive_suffix = QUALITY_PROMPT_SUFFIXES.get(mode, [])
    negative_suffix = QUALITY_NEGATIVE_SUFFIXES.get(mode, [])
    return {
        "quality_preset": resolved_preset,
        "user_prompt": base_prompt,
        "user_negative_prompt": base_negative_prompt,
        "effective_prompt": _merge_prompt_terms(base_prompt, positive_suffix),
        "effective_negative_prompt": _merge_prompt_terms(base_negative_prompt, negative_suffix),
        "positive_suffix": list(positive_suffix),
        "negative_suffix": list(negative_suffix),
    }


def _resolve_local_model_artifact(category: str, alias: Optional[str]) -> Optional[Dict[str, Any]]:
    clean_alias = _clean_text(alias)
    if not clean_alias:
        return None
    artifact = resolve_model_artifact(category, alias=clean_alias)
    if artifact is None:
        raise ValueError(f"Modelo local nao encontrado para category='{category}', alias='{clean_alias}'.")
    if artifact.get("status") != "ready":
        raise ValueError(
            f"O modelo '{clean_alias}' da categoria '{category}' nao esta pronto para uso. "
            f"Status atual: {artifact.get('status')}."
        )
    return artifact


def _load_optional_ip_adapter(
    pipeline: Any,
    alias: Optional[str],
    scale: Optional[float],
) -> Optional[Dict[str, Any]]:
    artifact = _resolve_local_model_artifact("ip_adapter", alias)
    if artifact is None:
        return None

    metadata = artifact.get("metadata") or {}
    weight_name = _clean_text(metadata.get("weight_name"))
    if not weight_name:
        raise ValueError(
            f"O ip_adapter '{artifact['alias']}' precisa de metadata.weight_name para ser carregado."
        )

    kwargs: Dict[str, Any] = {}
    subfolder = _clean_text(metadata.get("subfolder"))
    if subfolder:
        kwargs["subfolder"] = subfolder
    image_encoder_folder = metadata.get("image_encoder_folder", "image_encoder")
    if image_encoder_folder is not None:
        kwargs["image_encoder_folder"] = image_encoder_folder

    pipeline.load_ip_adapter(
        artifact["local_path"],
        weight_name=weight_name,
        **kwargs,
    )
    resolved_scale = (
        float(scale)
        if scale is not None
        else float(metadata.get("default_scale", 0.7))
    )
    if hasattr(pipeline, "set_ip_adapter_scale"):
        pipeline.set_ip_adapter_scale(resolved_scale)

    return {
        "artifact": artifact,
        "scale": resolved_scale,
    }


def _load_controlnet_model(
    alias: Optional[str],
    torch_dtype: Any,
):
    artifact = _resolve_local_model_artifact("controlnet", alias)
    if artifact is None:
        return None, None

    from diffusers import ControlNetModel

    controlnet_model = ControlNetModel.from_pretrained(
        artifact["local_path"],
        torch_dtype=torch_dtype,
    )
    return controlnet_model, artifact


def _configure_scheduler(pipeline: Any) -> Dict[str, Any]:
    if not hasattr(pipeline, "scheduler") or pipeline.scheduler is None:
        return {
            "preset": "pipeline_default",
            "class_name": None,
            "use_karras_sigmas": None,
            "timestep_spacing": None,
            "error": None,
        }

    scheduler = pipeline.scheduler
    scheduler_info = {
        "preset": "pipeline_default",
        "class_name": scheduler.__class__.__name__,
        "use_karras_sigmas": getattr(scheduler.config, "use_karras_sigmas", None),
        "timestep_spacing": getattr(scheduler.config, "timestep_spacing", None),
        "error": None,
    }

    try:
        from diffusers import DPMSolverMultistepScheduler

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            timestep_spacing="trailing",
        )
        resolved_scheduler = pipeline.scheduler
        scheduler_info.update(
            {
                "preset": DEFAULT_SCHEDULER_PRESET,
                "class_name": resolved_scheduler.__class__.__name__,
                "use_karras_sigmas": getattr(
                    resolved_scheduler.config,
                    "use_karras_sigmas",
                    None,
                ),
                "timestep_spacing": getattr(
                    resolved_scheduler.config,
                    "timestep_spacing",
                    None,
                ),
            }
        )
    except Exception as exc:
        scheduler_info["error"] = str(exc)

    return scheduler_info


def _configure_pipeline(pipeline: Any, device: str) -> str:
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()

    device_strategy = device
    if device == "cuda" and hasattr(pipeline, "enable_model_cpu_offload"):
        try:
            pipeline.enable_model_cpu_offload()
            device_strategy = "cuda+cpu_offload"
        except Exception:
            pipeline = pipeline.to("cuda")
    else:
        pipeline = pipeline.to(device)
    return device_strategy


def _configure_safety_checker(pipeline: Any, enabled: bool) -> str:
    if enabled:
        return "enabled"

    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if hasattr(pipeline, "requires_safety_checker"):
        pipeline.requires_safety_checker = False
    return "disabled"


def _extract_generation_metadata(result: Any, safety_checker_mode: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "safety_checker_mode": safety_checker_mode,
    }
    nsfw_content_detected = getattr(result, "nsfw_content_detected", None)
    if nsfw_content_detected is not None:
        if isinstance(nsfw_content_detected, (list, tuple)):
            metadata["nsfw_content_detected"] = list(nsfw_content_detected)
            metadata["nsfw_triggered"] = any(bool(item) for item in nsfw_content_detected)
        else:
            metadata["nsfw_content_detected"] = nsfw_content_detected
            metadata["nsfw_triggered"] = bool(nsfw_content_detected)
    return metadata


def _resolve_seed_and_generator(torch_module: Any, device: str, seed: Optional[int]):
    resolved_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch_module.Generator(device=generator_device).manual_seed(resolved_seed)
    return resolved_seed, generator


def _snapshot_input_file(source_path: str, target_dir: str, target_name: str) -> str:
    clean_source_path = _resolve_existing_path(source_path)
    _, extension = os.path.splitext(clean_source_path)
    resolved_extension = extension or ".png"
    target_path = os.path.join(target_dir, f"{target_name}{resolved_extension}")
    shutil.copy2(clean_source_path, target_path)
    return target_path


def _save_generated_images(images: Any, target_dir: str, prefix: str, run_id: str) -> Tuple[str, list[str]]:
    image_paths: list[str] = []
    for index, image in enumerate(images, start=1):
        suffix = f"_{index:02d}" if len(images) > 1 else ""
        image_filename = f"{prefix}_{run_id}{suffix}.png"
        image_path = os.path.join(target_dir, image_filename)
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths[0], image_paths


def _prepare_source_image(
    image_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: str = "RGB",
    blur_radius: float = 0.0,
):
    from PIL import Image, ImageFilter

    resolved_path = _resolve_existing_path(image_path)
    with Image.open(resolved_path) as handle:
        image = handle.convert(mode)

    if width is None and height is None:
        target_size = (
            _normalize_dimension(image.width),
            _normalize_dimension(image.height),
        )
    else:
        target_size = (_normalize_dimension(width or image.width), _normalize_dimension(height or image.height))

    if image.size != target_size:
        resample = Image.LANCZOS if mode == "RGB" else Image.NEAREST
        image = image.resize(target_size, resample)

    if blur_radius and blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    return image, resolved_path


def _checkpoint_supports_inpainting(model_path: str) -> bool:
    unet_config_path = os.path.join(model_path, "unet", "config.json")
    if not os.path.exists(unet_config_path):
        return False
    try:
        config = _read_json(unet_config_path)
    except Exception:
        return False
    return int(config.get("in_channels", 0) or 0) >= 9


def _write_generation_manifests(
    mode: str,
    request_payload: Dict[str, Any],
    effective_request_payload: Dict[str, Any],
    runtime: Dict[str, Any],
    model_artifact: Dict[str, Any],
    run_paths: Dict[str, str],
    image_path: str,
    image_paths: Optional[list[str]],
    seed: int,
    device_strategy: str,
    scheduler_info: Optional[Dict[str, Any]] = None,
    input_paths: Optional[Dict[str, str]] = None,
    auxiliary_models: Optional[Dict[str, Any]] = None,
    generation_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    manifest = {
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "request": request_payload,
        "effective_request": effective_request_payload,
        "runtime": {
            "device_strategy": device_strategy,
            "cuda_available": runtime["cuda_available"],
            "cuda_device_name": runtime["cuda_device_name"],
            "model_artifact": model_artifact,
            "scheduler": scheduler_info or {},
        },
        "inputs": input_paths or {},
        "auxiliary_models": auxiliary_models or {},
        "generation_metadata": generation_metadata or {},
        "outputs": {
            "image_path": image_path,
            "image_paths": image_paths or [image_path],
            "run_dir": run_paths["run_dir"],
        },
        "seed": seed,
    }
    mode_manifest_path = os.path.join(run_paths["manifests_dir"], f"{mode}_manifest.json")
    generation_manifest_path = os.path.join(run_paths["manifests_dir"], "generation_manifest.json")
    _write_json(mode_manifest_path, manifest)
    _write_json(generation_manifest_path, manifest)
    return mode_manifest_path, generation_manifest_path


def _build_generation_result(
    mode: str,
    image_path: str,
    image_paths: Optional[list[str]],
    mode_manifest_path: str,
    generation_manifest_path: str,
    run_paths: Dict[str, str],
    seed: int,
    device_strategy: str,
    model_artifact: Dict[str, Any],
    effective_request_payload: Optional[Dict[str, Any]] = None,
    scheduler_info: Optional[Dict[str, Any]] = None,
    generation_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "image_path": image_path,
        "image_paths": image_paths or [image_path],
        "manifest_path": mode_manifest_path,
        "generation_manifest_path": generation_manifest_path,
        "run_dir": run_paths["run_dir"],
        "run_id": run_paths["run_id"],
        "seed": seed,
        "device_strategy": device_strategy,
        "model_artifact": model_artifact,
        "effective_request": effective_request_payload or {},
        "scheduler": scheduler_info or {},
        "generation_metadata": generation_metadata or {},
    }


def generate_txt2img(
    prompt: str,
    negative_prompt: str = "",
    model_alias: str = DEFAULT_SD15_BASE_ALIAS,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = DEFAULT_TXT2IMG_STEPS,
    guidance_scale: float = DEFAULT_TXT2IMG_GUIDANCE,
    num_images: int = DEFAULT_TXT2IMG_IMAGES,
    ip_adapter_alias: Optional[str] = None,
    ip_adapter_image_path: Optional[str] = None,
    ip_adapter_scale: Optional[float] = None,
    controlnet_alias: Optional[str] = None,
    control_image_path: Optional[str] = None,
    controlnet_conditioning_scale: Optional[float] = None,
    safety_checker_enabled: bool = True,
    quality_preset: str = DEFAULT_QUALITY_PRESET,
    seed: Optional[int] = None,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
    artifacts_root: str = DEFAULT_IMG_GEN_ARTIFACTS_DIR,
) -> Dict[str, Any]:
    request = _validate_txt2img_request(
        Txt2ImgRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_alias=model_alias,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            ip_adapter_alias=ip_adapter_alias,
            ip_adapter_image_path=ip_adapter_image_path,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_alias=controlnet_alias,
            control_image_path=control_image_path,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            safety_checker_enabled=safety_checker_enabled,
            quality_preset=quality_preset,
            seed=seed,
            project_id=project_id,
            run_id=run_id,
        )
    )

    runtime, model_artifact, model_path, torch_module, device, torch_dtype = _prepare_runtime(
        request.model_alias,
        model_category="checkpoints",
    )

    auxiliary_models: Dict[str, Any] = {}
    input_snapshots: Dict[str, str] = {}
    control_kwargs: Dict[str, Any] = {}
    controlnet_model = None
    controlnet_artifact = None
    if _clean_text(request.controlnet_alias):
        controlnet_model, controlnet_artifact = _load_controlnet_model(
            request.controlnet_alias,
            torch_dtype=torch_dtype,
        )
        auxiliary_models["controlnet"] = {
            "artifact": controlnet_artifact,
            "conditioning_scale": (
                float(request.controlnet_conditioning_scale)
                if request.controlnet_conditioning_scale is not None
                else 0.8
            ),
        }

    if controlnet_model is not None:
        from diffusers import StableDiffusionControlNetPipeline

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_path,
            controlnet=controlnet_model,
            torch_dtype=torch_dtype,
        )
    else:
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )
    scheduler_info = _configure_scheduler(pipeline)
    safety_checker_mode = _configure_safety_checker(
        pipeline,
        request.safety_checker_enabled,
    )
    device_strategy = _configure_pipeline(pipeline, device)
    ip_adapter_runtime = _load_optional_ip_adapter(
        pipeline,
        request.ip_adapter_alias,
        request.ip_adapter_scale,
    )
    if ip_adapter_runtime is not None:
        auxiliary_models["ip_adapter"] = {
            "artifact": ip_adapter_runtime["artifact"],
            "scale": ip_adapter_runtime["scale"],
        }
    resolved_seed, generator = _resolve_seed_and_generator(torch_module, device, request.seed)

    if _clean_text(request.ip_adapter_image_path):
        ip_image, resolved_ip_image_path = _prepare_source_image(
            request.ip_adapter_image_path,
            width=request.width,
            height=request.height,
            mode="RGB",
        )
        control_kwargs["ip_adapter_image"] = ip_image
    else:
        resolved_ip_image_path = None

    if _clean_text(request.control_image_path):
        control_image, resolved_control_image_path = _prepare_source_image(
            request.control_image_path,
            width=request.width,
            height=request.height,
            mode="RGB",
        )
        control_kwargs["control_image"] = control_image
        control_kwargs["controlnet_conditioning_scale"] = auxiliary_models["controlnet"][
            "conditioning_scale"
        ]
    else:
        resolved_control_image_path = None

    effective_prompting = _prepare_effective_prompts(
        mode="txt2img",
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        quality_preset=request.quality_preset,
    )

    result = pipeline(
        prompt=effective_prompting["effective_prompt"],
        negative_prompt=effective_prompting["effective_negative_prompt"] or None,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.num_images,
        generator=generator,
        **control_kwargs,
    )
    generation_metadata = _extract_generation_metadata(result, safety_checker_mode)

    images = result.images
    run_paths = _resolve_run_dir(
        project_id=request.project_id,
        run_id=request.run_id,
        artifacts_root=artifacts_root,
    )
    image_path, image_paths = _save_generated_images(
        images,
        run_paths["generated_dir"],
        "txt2img",
        run_paths["run_id"],
    )

    if resolved_ip_image_path:
        input_snapshots["ip_adapter_image_path"] = _snapshot_input_file(
            resolved_ip_image_path,
            run_paths["inputs_dir"],
            "ip_adapter_image",
        )
    if resolved_control_image_path:
        input_snapshots["control_image_path"] = _snapshot_input_file(
            resolved_control_image_path,
            run_paths["inputs_dir"],
            "control_image",
        )

    mode_manifest_path, generation_manifest_path = _write_generation_manifests(
        mode="txt2img",
        request_payload=request.to_dict(),
        effective_request_payload={
            **request.to_dict(),
            "prompt": effective_prompting["effective_prompt"],
            "negative_prompt": effective_prompting["effective_negative_prompt"],
            "user_prompt": effective_prompting["user_prompt"],
            "user_negative_prompt": effective_prompting["user_negative_prompt"],
            "quality_preset": effective_prompting["quality_preset"],
        },
        runtime=runtime,
        model_artifact=model_artifact,
        run_paths=run_paths,
        image_path=image_path,
        image_paths=image_paths,
        seed=resolved_seed,
        device_strategy=device_strategy,
        scheduler_info=scheduler_info,
        input_paths=input_snapshots,
        auxiliary_models=auxiliary_models,
        generation_metadata={
            **generation_metadata,
            "quality_preset": effective_prompting["quality_preset"],
            "quality_positive_suffix": effective_prompting["positive_suffix"],
            "quality_negative_suffix": effective_prompting["negative_suffix"],
        },
    )
    return _build_generation_result(
        mode="txt2img",
        image_path=image_path,
        image_paths=image_paths,
        mode_manifest_path=mode_manifest_path,
        generation_manifest_path=generation_manifest_path,
        run_paths=run_paths,
        seed=resolved_seed,
        device_strategy=device_strategy,
        model_artifact=model_artifact,
        effective_request_payload={
            **request.to_dict(),
            "prompt": effective_prompting["effective_prompt"],
            "negative_prompt": effective_prompting["effective_negative_prompt"],
            "user_prompt": effective_prompting["user_prompt"],
            "user_negative_prompt": effective_prompting["user_negative_prompt"],
            "quality_preset": effective_prompting["quality_preset"],
        },
        scheduler_info=scheduler_info,
        generation_metadata={
            **generation_metadata,
            "quality_preset": effective_prompting["quality_preset"],
            "quality_positive_suffix": effective_prompting["positive_suffix"],
            "quality_negative_suffix": effective_prompting["negative_suffix"],
        },
    )


def generate_img2img(
    prompt: str,
    init_image_path: str,
    negative_prompt: str = "",
    model_alias: str = DEFAULT_SD15_BASE_ALIAS,
    width: Optional[int] = None,
    height: Optional[int] = None,
    strength: float = DEFAULT_IMG2IMG_STRENGTH,
    num_inference_steps: int = DEFAULT_IMG2IMG_STEPS,
    guidance_scale: float = DEFAULT_IMG2IMG_GUIDANCE,
    num_images: int = DEFAULT_IMG2IMG_IMAGES,
    ip_adapter_alias: Optional[str] = None,
    ip_adapter_image_path: Optional[str] = None,
    ip_adapter_scale: Optional[float] = None,
    controlnet_alias: Optional[str] = None,
    control_image_path: Optional[str] = None,
    controlnet_conditioning_scale: Optional[float] = None,
    safety_checker_enabled: bool = True,
    quality_preset: str = DEFAULT_QUALITY_PRESET,
    seed: Optional[int] = None,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
    artifacts_root: str = DEFAULT_IMG_GEN_ARTIFACTS_DIR,
) -> Dict[str, Any]:
    request = _validate_img2img_request(
        Img2ImgRequest(
            prompt=prompt,
            init_image_path=init_image_path,
            negative_prompt=negative_prompt,
            model_alias=model_alias,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            ip_adapter_alias=ip_adapter_alias,
            ip_adapter_image_path=ip_adapter_image_path,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_alias=controlnet_alias,
            control_image_path=control_image_path,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            safety_checker_enabled=safety_checker_enabled,
            quality_preset=quality_preset,
            seed=seed,
            project_id=project_id,
            run_id=run_id,
        )
    )

    runtime, model_artifact, model_path, torch_module, device, torch_dtype = _prepare_runtime(
        request.model_alias,
        model_category="checkpoints",
    )
    init_image, resolved_init_path = _prepare_source_image(
        request.init_image_path,
        width=request.width,
        height=request.height,
        mode="RGB",
    )

    auxiliary_models: Dict[str, Any] = {}
    input_snapshots: Dict[str, str] = {}
    pipeline_kwargs: Dict[str, Any] = {}
    controlnet_model = None
    controlnet_artifact = None
    if _clean_text(request.controlnet_alias):
        controlnet_model, controlnet_artifact = _load_controlnet_model(
            request.controlnet_alias,
            torch_dtype=torch_dtype,
        )
        auxiliary_models["controlnet"] = {
            "artifact": controlnet_artifact,
            "conditioning_scale": (
                float(request.controlnet_conditioning_scale)
                if request.controlnet_conditioning_scale is not None
                else 0.8
            ),
        }

    if controlnet_model is not None:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline

        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path,
            controlnet=controlnet_model,
            torch_dtype=torch_dtype,
        )
    else:
        from diffusers import AutoPipelineForImage2Image

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )
    scheduler_info = _configure_scheduler(pipeline)
    safety_checker_mode = _configure_safety_checker(
        pipeline,
        request.safety_checker_enabled,
    )
    device_strategy = _configure_pipeline(pipeline, device)
    ip_adapter_runtime = _load_optional_ip_adapter(
        pipeline,
        request.ip_adapter_alias,
        request.ip_adapter_scale,
    )
    if ip_adapter_runtime is not None:
        auxiliary_models["ip_adapter"] = {
            "artifact": ip_adapter_runtime["artifact"],
            "scale": ip_adapter_runtime["scale"],
        }
    resolved_seed, generator = _resolve_seed_and_generator(torch_module, device, request.seed)

    if _clean_text(request.ip_adapter_image_path):
        ip_image, resolved_ip_image_path = _prepare_source_image(
            request.ip_adapter_image_path,
            width=init_image.width,
            height=init_image.height,
            mode="RGB",
        )
        pipeline_kwargs["ip_adapter_image"] = ip_image
    else:
        resolved_ip_image_path = None

    if _clean_text(request.control_image_path):
        control_image, resolved_control_image_path = _prepare_source_image(
            request.control_image_path,
            width=init_image.width,
            height=init_image.height,
            mode="RGB",
        )
        pipeline_kwargs["control_image"] = control_image
        pipeline_kwargs["controlnet_conditioning_scale"] = auxiliary_models["controlnet"][
            "conditioning_scale"
        ]
    else:
        resolved_control_image_path = None

    effective_prompting = _prepare_effective_prompts(
        mode="img2img",
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        quality_preset=request.quality_preset,
    )

    result = pipeline(
        prompt=effective_prompting["effective_prompt"],
        negative_prompt=effective_prompting["effective_negative_prompt"] or None,
        image=init_image,
        strength=request.strength,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.num_images,
        generator=generator,
        **pipeline_kwargs,
    )
    generation_metadata = _extract_generation_metadata(result, safety_checker_mode)

    images = result.images
    run_paths = _resolve_run_dir(
        project_id=request.project_id,
        run_id=request.run_id,
        artifacts_root=artifacts_root,
    )
    snapped_init_path = _snapshot_input_file(
        resolved_init_path,
        run_paths["inputs_dir"],
        "img2img_init",
    )
    input_snapshots["init_image_path"] = snapped_init_path
    if resolved_ip_image_path:
        input_snapshots["ip_adapter_image_path"] = _snapshot_input_file(
            resolved_ip_image_path,
            run_paths["inputs_dir"],
            "img2img_ip_adapter",
        )
    if resolved_control_image_path:
        input_snapshots["control_image_path"] = _snapshot_input_file(
            resolved_control_image_path,
            run_paths["inputs_dir"],
            "img2img_control",
        )
    image_path, image_paths = _save_generated_images(
        images,
        run_paths["generated_dir"],
        "img2img",
        run_paths["run_id"],
    )

    request_payload = request.to_dict()
    request_payload["init_image_path"] = resolved_init_path
    request_payload["width"] = init_image.width
    request_payload["height"] = init_image.height
    mode_manifest_path, generation_manifest_path = _write_generation_manifests(
        mode="img2img",
        request_payload=request_payload,
        effective_request_payload={
            **request_payload,
            "prompt": effective_prompting["effective_prompt"],
            "negative_prompt": effective_prompting["effective_negative_prompt"],
            "user_prompt": effective_prompting["user_prompt"],
            "user_negative_prompt": effective_prompting["user_negative_prompt"],
            "quality_preset": effective_prompting["quality_preset"],
        },
        runtime=runtime,
        model_artifact=model_artifact,
        run_paths=run_paths,
        image_path=image_path,
        image_paths=image_paths,
        seed=resolved_seed,
        device_strategy=device_strategy,
        scheduler_info=scheduler_info,
        input_paths=input_snapshots,
        auxiliary_models=auxiliary_models,
        generation_metadata={
            **generation_metadata,
            "quality_preset": effective_prompting["quality_preset"],
            "quality_positive_suffix": effective_prompting["positive_suffix"],
            "quality_negative_suffix": effective_prompting["negative_suffix"],
        },
    )
    return _build_generation_result(
        mode="img2img",
        image_path=image_path,
        image_paths=image_paths,
        mode_manifest_path=mode_manifest_path,
        generation_manifest_path=generation_manifest_path,
        run_paths=run_paths,
        seed=resolved_seed,
        device_strategy=device_strategy,
        model_artifact=model_artifact,
        effective_request_payload={
            **request_payload,
            "prompt": effective_prompting["effective_prompt"],
            "negative_prompt": effective_prompting["effective_negative_prompt"],
            "user_prompt": effective_prompting["user_prompt"],
            "user_negative_prompt": effective_prompting["user_negative_prompt"],
            "quality_preset": effective_prompting["quality_preset"],
        },
        scheduler_info=scheduler_info,
        generation_metadata={
            **generation_metadata,
            "quality_preset": effective_prompting["quality_preset"],
            "quality_positive_suffix": effective_prompting["positive_suffix"],
            "quality_negative_suffix": effective_prompting["negative_suffix"],
        },
    )


def generate_inpainting(
    prompt: str,
    init_image_path: str,
    mask_image_path: str,
    negative_prompt: str = "",
    model_alias: str = DEFAULT_SD15_BASE_ALIAS,
    width: Optional[int] = None,
    height: Optional[int] = None,
    strength: float = DEFAULT_INPAINT_STRENGTH,
    num_inference_steps: int = DEFAULT_INPAINT_STEPS,
    guidance_scale: float = DEFAULT_INPAINT_GUIDANCE,
    num_images: int = DEFAULT_INPAINT_IMAGES,
    safety_checker_enabled: bool = True,
    quality_preset: str = DEFAULT_QUALITY_PRESET,
    mask_blur_radius: float = DEFAULT_INPAINT_MASK_BLUR,
    seed: Optional[int] = None,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
    artifacts_root: str = DEFAULT_IMG_GEN_ARTIFACTS_DIR,
) -> Dict[str, Any]:
    request = _validate_inpaint_request(
        InpaintRequest(
            prompt=prompt,
            init_image_path=init_image_path,
            mask_image_path=mask_image_path,
            negative_prompt=negative_prompt,
            model_alias=model_alias,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            safety_checker_enabled=safety_checker_enabled,
            quality_preset=quality_preset,
            mask_blur_radius=mask_blur_radius,
            seed=seed,
            project_id=project_id,
            run_id=run_id,
        )
    )

    runtime, model_artifact, model_path, torch_module, device, torch_dtype = _prepare_runtime(
        request.model_alias,
        model_category="inpainting",
    )
    if not _checkpoint_supports_inpainting(model_path):
        raise ValueError(
            "O checkpoint selecionado nao suporta inpainting. "
            "Use um modelo de inpainting dedicado (UNet com in_channels=9)."
        )

    init_image, resolved_init_path = _prepare_source_image(
        request.init_image_path,
        width=request.width,
        height=request.height,
        mode="RGB",
    )
    mask_image, resolved_mask_path = _prepare_source_image(
        request.mask_image_path,
        width=init_image.width,
        height=init_image.height,
        mode="L",
        blur_radius=request.mask_blur_radius,
    )

    from diffusers import AutoPipelineForInpainting

    pipeline = AutoPipelineForInpainting.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    scheduler_info = _configure_scheduler(pipeline)
    safety_checker_mode = _configure_safety_checker(
        pipeline,
        request.safety_checker_enabled,
    )
    device_strategy = _configure_pipeline(pipeline, device)
    resolved_seed, generator = _resolve_seed_and_generator(torch_module, device, request.seed)
    effective_prompting = _prepare_effective_prompts(
        mode="inpaint",
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        quality_preset=request.quality_preset,
    )

    result = pipeline(
        prompt=effective_prompting["effective_prompt"],
        negative_prompt=effective_prompting["effective_negative_prompt"] or None,
        image=init_image,
        mask_image=mask_image,
        strength=request.strength,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.num_images,
        generator=generator,
    )
    generation_metadata = _extract_generation_metadata(result, safety_checker_mode)

    images = result.images
    run_paths = _resolve_run_dir(
        project_id=request.project_id,
        run_id=request.run_id,
        artifacts_root=artifacts_root,
    )
    snapped_init_path = _snapshot_input_file(
        resolved_init_path,
        run_paths["inputs_dir"],
        "inpaint_init",
    )
    snapped_mask_path = _snapshot_input_file(
        resolved_mask_path,
        run_paths["inputs_dir"],
        "inpaint_mask",
    )
    image_path, image_paths = _save_generated_images(
        images,
        run_paths["generated_dir"],
        "inpaint",
        run_paths["run_id"],
    )

    request_payload = request.to_dict()
    request_payload["init_image_path"] = resolved_init_path
    request_payload["mask_image_path"] = resolved_mask_path
    request_payload["width"] = init_image.width
    request_payload["height"] = init_image.height
    mode_manifest_path, generation_manifest_path = _write_generation_manifests(
        mode="inpaint",
        request_payload=request_payload,
        effective_request_payload={
            **request_payload,
            "prompt": effective_prompting["effective_prompt"],
            "negative_prompt": effective_prompting["effective_negative_prompt"],
            "user_prompt": effective_prompting["user_prompt"],
            "user_negative_prompt": effective_prompting["user_negative_prompt"],
            "quality_preset": effective_prompting["quality_preset"],
        },
        runtime=runtime,
        model_artifact=model_artifact,
        run_paths=run_paths,
        image_path=image_path,
        image_paths=image_paths,
        seed=resolved_seed,
        device_strategy=device_strategy,
        scheduler_info=scheduler_info,
        input_paths={
            "init_image_path": snapped_init_path,
            "mask_image_path": snapped_mask_path,
        },
        generation_metadata={
            **generation_metadata,
            "quality_preset": effective_prompting["quality_preset"],
            "quality_positive_suffix": effective_prompting["positive_suffix"],
            "quality_negative_suffix": effective_prompting["negative_suffix"],
            "mask_blur_radius": request.mask_blur_radius,
        },
    )
    return _build_generation_result(
        mode="inpaint",
        image_path=image_path,
        image_paths=image_paths,
        mode_manifest_path=mode_manifest_path,
        generation_manifest_path=generation_manifest_path,
        run_paths=run_paths,
        seed=resolved_seed,
        device_strategy=device_strategy,
        model_artifact=model_artifact,
        effective_request_payload={
            **request_payload,
            "prompt": effective_prompting["effective_prompt"],
            "negative_prompt": effective_prompting["effective_negative_prompt"],
            "user_prompt": effective_prompting["user_prompt"],
            "user_negative_prompt": effective_prompting["user_negative_prompt"],
            "quality_preset": effective_prompting["quality_preset"],
        },
        scheduler_info=scheduler_info,
        generation_metadata={
            **generation_metadata,
            "quality_preset": effective_prompting["quality_preset"],
            "quality_positive_suffix": effective_prompting["positive_suffix"],
            "quality_negative_suffix": effective_prompting["negative_suffix"],
            "mask_blur_radius": request.mask_blur_radius,
        },
    )
