"""WASAPI loopback capture helpers for phase 7 audio workflows."""

from __future__ import annotations

import os
import time
import wave
import webbrowser
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - guarded by runtime diagnostics
    np = None  # type: ignore[assignment]


DEFAULT_LOOPBACK_SAMPLE_RATE = 48000
DEFAULT_LOOPBACK_CHANNELS = 2
DEFAULT_LOOPBACK_BLOCK_FRAMES = 2048


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _optional_import_soundcard():
    try:
        import soundcard as soundcard_module  # type: ignore[import-not-found]

        return soundcard_module
    except Exception:
        return None


def _require_soundcard():
    soundcard_module = _optional_import_soundcard()
    if soundcard_module is None:
        raise RuntimeError(
            "A captura WASAPI loopback precisa do pacote opcional `soundcard`. "
            "Instale na `.venv310` com `pip install soundcard`."
        )
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")
    return soundcard_module


def _resolve_loopback_microphone(
    soundcard_module: Any,
    device_name: str = "",
):
    normalized_device_name = _clean_text(device_name).lower()
    microphones = list(soundcard_module.all_microphones(include_loopback=True))
    if not microphones:
        raise RuntimeError("Nenhum device loopback foi encontrado pelo backend `soundcard`.")

    if normalized_device_name:
        for microphone in microphones:
            microphone_name = _clean_text(getattr(microphone, "name", ""))
            if normalized_device_name in microphone_name.lower():
                return microphone
        raise ValueError(f"Nenhum device loopback corresponde a '{device_name}'.")

    default_speaker = soundcard_module.default_speaker()
    default_speaker_name = _clean_text(getattr(default_speaker, "name", ""))
    for microphone in microphones:
        microphone_name = _clean_text(getattr(microphone, "name", ""))
        if default_speaker_name and default_speaker_name.lower() in microphone_name.lower():
            return microphone

    return microphones[0]


def list_speaker_devices() -> List[Dict[str, Any]]:
    soundcard_module = _optional_import_soundcard()
    if soundcard_module is None:
        return []

    default_speaker_name = _clean_text(getattr(soundcard_module.default_speaker(), "name", ""))
    devices: List[Dict[str, Any]] = []
    for speaker in soundcard_module.all_speakers():
        name = _clean_text(getattr(speaker, "name", ""))
        devices.append(
            {
                "name": name,
                "id": _clean_text(getattr(speaker, "id", "")) or None,
                "is_default": bool(default_speaker_name and default_speaker_name.lower() == name.lower()),
            }
        )
    return devices


def list_loopback_devices() -> List[Dict[str, Any]]:
    soundcard_module = _optional_import_soundcard()
    if soundcard_module is None:
        return []

    default_speaker_name = _clean_text(getattr(soundcard_module.default_speaker(), "name", ""))
    devices: List[Dict[str, Any]] = []
    for microphone in soundcard_module.all_microphones(include_loopback=True):
        name = _clean_text(getattr(microphone, "name", ""))
        devices.append(
            {
                "name": name,
                "id": _clean_text(getattr(microphone, "id", "")) or None,
                "is_default": bool(default_speaker_name and default_speaker_name.lower() in name.lower()),
            }
        )
    return devices


def list_audio_routing_devices() -> Dict[str, Any]:
    speakers = list_speaker_devices()
    loopback_devices = list_loopback_devices()
    return {
        "speakers": speakers,
        "loopback_devices": loopback_devices,
        "default_speaker": next((item for item in speakers if item.get("is_default")), None),
        "default_loopback": next((item for item in loopback_devices if item.get("is_default")), None),
    }


def open_reference_url(source_url: str) -> bool:
    clean_url = _clean_text(source_url)
    if not clean_url:
        raise ValueError("A URL da referencia deve ser informada para abertura assistida.")
    return bool(webbrowser.open(clean_url))


def _write_float_audio_to_wav(
    samples: "np.ndarray",
    output_path: str,
    sample_rate: int,
) -> str:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    channels = int(pcm.shape[1]) if pcm.ndim == 2 else 1

    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return output_path


def capture_wasapi_loopback(
    output_path: str,
    duration_seconds: float,
    device_name: str = "",
    sample_rate: int = DEFAULT_LOOPBACK_SAMPLE_RATE,
    channels: int = DEFAULT_LOOPBACK_CHANNELS,
    block_frames: int = DEFAULT_LOOPBACK_BLOCK_FRAMES,
    open_url: Optional[str] = None,
    launch_delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    if float(duration_seconds) <= 0:
        raise ValueError("duration_seconds deve ser maior que zero.")

    soundcard_module = _require_soundcard()
    microphone = _resolve_loopback_microphone(soundcard_module, device_name=device_name)
    if open_url:
        open_reference_url(open_url)
    if float(launch_delay_seconds) > 0:
        time.sleep(float(launch_delay_seconds))

    target_frames = int(round(float(duration_seconds) * int(sample_rate)))
    chunks: List["np.ndarray"] = []

    with microphone.recorder(
        samplerate=int(sample_rate),
        channels=int(channels),
        blocksize=int(block_frames),
    ) as recorder:
        captured_frames = 0
        while captured_frames < target_frames:
            requested_frames = min(int(block_frames), target_frames - captured_frames)
            chunk = np.asarray(recorder.record(numframes=requested_frames), dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk[:, None]
            if chunk.shape[1] < int(channels):
                repeats = int(channels) // int(chunk.shape[1]) + 1
                chunk = np.repeat(chunk, repeats, axis=1)[:, : int(channels)]
            elif chunk.shape[1] > int(channels):
                chunk = chunk[:, : int(channels)]
            chunks.append(chunk)
            captured_frames += int(chunk.shape[0])

    if not chunks:
        raise RuntimeError("Nenhum frame foi capturado via WASAPI loopback.")

    audio = np.concatenate(chunks, axis=0)
    if audio.shape[0] > target_frames:
        audio = audio[:target_frames]

    _write_float_audio_to_wav(audio, output_path, sample_rate=int(sample_rate))
    return {
        "output_path": output_path,
        "duration_seconds": round(float(audio.shape[0] / int(sample_rate)), 4),
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "device_name": _clean_text(getattr(microphone, "name", "")) or None,
        "opened_url": bool(open_url),
    }
