"""Minimal MP4 audio extraction helpers for manual/API tests."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2


@dataclass
class Mp4AudioExtractionRequest:
    uploaded_video_path: str
    output_path: Optional[str] = None
    output_dir: Optional[str] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    overwrite: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str, fallback: str = "uploaded-video") -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or fallback


def _resolve_existing_path(path: str) -> str:
    clean_path = _clean_text(path)
    if not clean_path:
        raise ValueError("O caminho do video .mp4 deve ser informado.")

    candidates = [
        os.path.abspath(clean_path),
        os.path.abspath(os.path.join(os.getcwd(), clean_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Arquivo nao encontrado: {clean_path}")


def _resolve_ffmpeg_path() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("`ffmpeg` nao foi encontrado no PATH.")
    return ffmpeg_path


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _raise_command_error(command_name: str, result: subprocess.CompletedProcess[str]) -> None:
    stderr = _clean_text(result.stderr)
    stdout = _clean_text(result.stdout)
    message = stderr or stdout or f"Falha ao executar {command_name}."
    raise RuntimeError(message)


def _resolve_output_path(
    source_path: str,
    output_path: Optional[str],
    output_dir: Optional[str],
) -> str:
    if _clean_text(output_path):
        resolved_output_path = os.path.abspath(str(output_path))
        os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
        return resolved_output_path

    resolved_output_dir = os.path.abspath(_clean_text(output_dir) or os.path.dirname(source_path))
    os.makedirs(resolved_output_dir, exist_ok=True)

    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    output_filename = f"{_slugify(source_stem)}_audio.wav"
    return os.path.join(resolved_output_dir, output_filename)


def extract_audio_from_mp4(request: Mp4AudioExtractionRequest) -> Dict[str, Any]:
    source_path = _resolve_existing_path(request.uploaded_video_path)
    extension = os.path.splitext(source_path)[1].lower()
    if extension != ".mp4":
        raise ValueError("uploaded_video_path deve apontar para um arquivo .mp4.")

    output_path = _resolve_output_path(
        source_path,
        output_path=request.output_path,
        output_dir=request.output_dir,
    )
    if os.path.exists(output_path) and not request.overwrite:
        raise FileExistsError(f"O arquivo de saida ja existe: {output_path}")

    ffmpeg_path = _resolve_ffmpeg_path()
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y" if request.overwrite else "-n",
        "-i",
        source_path,
        "-vn",
        "-ar",
        str(int(request.sample_rate or DEFAULT_SAMPLE_RATE)),
        "-ac",
        str(int(request.channels or DEFAULT_CHANNELS)),
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    result = _run_command(command)
    if result.returncode != 0:
        _raise_command_error("ffmpeg mp4 audio extraction", result)
    if not os.path.exists(output_path):
        raise RuntimeError(f"O ffmpeg terminou sem criar o WAV esperado: {output_path}")

    return {
        "request": request.to_dict(),
        "source_video_path": source_path,
        "output_path": output_path,
        "output_size_bytes": os.path.getsize(output_path),
        "sample_rate": int(request.sample_rate or DEFAULT_SAMPLE_RATE),
        "channels": int(request.channels or DEFAULT_CHANNELS),
        "source_mode": "uploaded_video",
        "device_name": "uploaded-mp4",
        "opened_url": False,
        "ffmpeg_path": ffmpeg_path,
    }


def extract_uploaded_mp4_audio_to_wav(
    uploaded_video_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    overwrite: bool = True,
) -> Dict[str, Any]:
    return extract_audio_from_mp4(
        Mp4AudioExtractionRequest(
            uploaded_video_path=uploaded_video_path,
            output_path=output_path,
            output_dir=output_dir,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
    )
