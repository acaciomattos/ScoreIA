"""Filesystem layout helpers for phase 7 audio models."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict


DEFAULT_AUDIO_GEN_MODELS_DIR = os.path.join("models", "audio_gen")


@dataclass(frozen=True)
class AudioModelPaths:
    models_root: str
    whisper_dir: str
    separation_dir: str
    cache_dir: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def build_model_paths(
    models_root: str = DEFAULT_AUDIO_GEN_MODELS_DIR,
) -> AudioModelPaths:
    return AudioModelPaths(
        models_root=models_root,
        whisper_dir=os.path.join(models_root, "whisper"),
        separation_dir=os.path.join(models_root, "separation"),
        cache_dir=os.path.join(models_root, "cache"),
    )
