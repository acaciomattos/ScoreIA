"""Filesystem layout helpers for local image-generation model assets."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict


DEFAULT_IMG_GEN_MODELS_DIR = os.path.join("models", "img_gen")
DEFAULT_COMFYUI_DIR = os.path.join("external", "ComfyUI")


@dataclass(frozen=True)
class ImgGenModelPaths:
    models_root: str
    registry_file: str
    checkpoints_dir: str
    inpainting_dir: str
    controlnet_dir: str
    ip_adapter_dir: str
    sam_dir: str
    upscalers_dir: str
    cache_dir: str
    comfyui_root: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def build_model_paths(
    models_root: str = DEFAULT_IMG_GEN_MODELS_DIR,
    comfyui_root: str = DEFAULT_COMFYUI_DIR,
) -> ImgGenModelPaths:
    return ImgGenModelPaths(
        models_root=models_root,
        registry_file=os.path.join(models_root, "registry.json"),
        checkpoints_dir=os.path.join(models_root, "checkpoints"),
        inpainting_dir=os.path.join(models_root, "inpainting"),
        controlnet_dir=os.path.join(models_root, "controlnet"),
        ip_adapter_dir=os.path.join(models_root, "ip_adapter"),
        sam_dir=os.path.join(models_root, "sam"),
        upscalers_dir=os.path.join(models_root, "upscalers"),
        cache_dir=os.path.join(models_root, "cache"),
        comfyui_root=comfyui_root,
    )
