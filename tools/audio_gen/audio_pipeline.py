"""Local audio-processing runners for phase 7."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import wave
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Set

import requests

try:
    import numpy as np
except Exception:  # pragma: no cover - handled by runtime diagnostics
    np = None  # type: ignore[assignment]

try:
    from scipy import signal
except Exception:  # pragma: no cover - handled by runtime diagnostics
    signal = None  # type: ignore[assignment]

from tools.audio_gen.model_paths import DEFAULT_AUDIO_GEN_MODELS_DIR
from tools.audio_gen.capture_loopback import capture_wasapi_loopback, list_loopback_devices
from tools.audio_gen.capture_loopback import list_audio_routing_devices
from tools.audio_gen.project_paths import (
    DEFAULT_AUDIO_GEN_ARTIFACTS_DIR,
    DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    build_project_artifact_paths,
)
from tools.audio_gen.youtube_background import launch_youtube_background_player, stop_background_player
from tools.audio_gen.project_registry import (
    add_beat_blueprint,
    add_capture_session,
    add_derived_context,
    add_master_session,
    add_mix_session,
    add_reference_asset,
    add_reference_link,
    add_working_set,
    find_reference_asset,
    find_capture_session,
    link_external_reference_to_audio,
    load_capture_sessions,
    load_project_bundle,
    update_reference_asset,
    update_capture_session,
)
from tools.audio_gen.runtime_env import ensure_runtime_ready


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_TARGET_PEAK_DB = -1.0
DEFAULT_TARGET_LUFS = -14.0
DEFAULT_TRUE_PEAK_DB = -1.0
DEFAULT_LOUDNESS_RANGE = 7.0
DEFAULT_BEAT_BARS = 4
DEFAULT_BEAT_STEPS_PER_BAR = 16
ACTIVE_PATTERN_CHARS = {"x", "X", "1", "o", "O", "*"}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", _clean_text(value).lower()).strip("-")
    return slug or "audio"


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


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


def _resolve_existing_path(path: str) -> str:
    clean_path = _clean_text(path)
    if not clean_path:
        raise ValueError("O caminho do audio deve ser informado.")

    candidates = [
        os.path.abspath(clean_path),
        os.path.abspath(os.path.join(os.getcwd(), clean_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Arquivo nao encontrado: {clean_path}")


def _ffmpeg_null_output() -> str:
    return "NUL" if os.name == "nt" else "/dev/null"


def _escape_ffmpeg_filter_value(value: str) -> str:
    escaped = str(value).replace("\\", "/")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(",", "\\,")
    return escaped


def _normalize_ffmpeg_filter_path(path: str) -> str:
    normalized = _clean_text(path)
    if not normalized:
        return normalized
    try:
        absolute = os.path.abspath(normalized)
        cwd = os.path.abspath(os.getcwd())
        if os.path.splitdrive(absolute)[0].lower() == os.path.splitdrive(cwd)[0].lower():
            normalized = os.path.relpath(absolute, cwd)
        else:
            normalized = absolute
    except Exception:
        normalized = os.path.abspath(normalized)
    return normalized.replace("\\", "/")


def _run_command(args: List[str]) -> subprocess.CompletedProcess[str]:
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


def _resolve_run_dir(
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
    artifacts_root: str = DEFAULT_AUDIO_GEN_ARTIFACTS_DIR,
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
    directories = {
        "project_dir": project_dir,
        "run_id": resolved_run_id,
        "run_dir": run_dir,
        "inputs_dir": os.path.join(run_dir, "inputs"),
        "normalized_dir": os.path.join(run_dir, "normalized"),
        "transcripts_dir": os.path.join(run_dir, "transcripts"),
        "separated_dir": os.path.join(run_dir, "separated"),
        "analyses_dir": os.path.join(run_dir, "analyses"),
        "beats_dir": os.path.join(run_dir, "beats"),
        "mixes_dir": os.path.join(run_dir, "mixes"),
        "masters_dir": os.path.join(run_dir, "masters"),
        "manifests_dir": os.path.join(run_dir, "manifests"),
        "captures_dir": os.path.join(run_dir, "captures"),
        "captures_raw_dir": os.path.join(run_dir, "captures", "raw"),
        "captures_clean_dir": os.path.join(run_dir, "captures", "clean"),
    }
    for key, directory in directories.items():
        if key.endswith("_dir"):
            os.makedirs(directory, exist_ok=True)
    return directories


def _copy_into_directory(source_path: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    destination = os.path.join(target_dir, os.path.basename(source_path))
    if os.path.abspath(source_path) != os.path.abspath(destination):
        shutil.copy2(source_path, destination)
    return destination


def _normalize_audio_to_wav(
    source_path: str,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> str:
    ffmpeg_path = ensure_runtime_ready("mix")["ffmpeg_path"]
    result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            source_path,
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    if result.returncode != 0:
        _raise_command_error("ffmpeg", result)
    return output_path


def _load_wav_array(path: str) -> tuple["np.ndarray", int]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError("A leitura WAV interna espera audio PCM 16-bit.")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels)
    return samples, sample_rate


def _can_load_wav_direct(path: str, sample_rate: int, channels: int) -> bool:
    if os.path.splitext(path)[1].lower() != ".wav":
        return False
    try:
        with wave.open(path, "rb") as handle:
            return (
                int(handle.getframerate()) == int(sample_rate)
                and int(handle.getnchannels()) == int(channels)
                and int(handle.getsampwidth()) == 2
            )
    except Exception:
        return False


def _load_audio_array(
    source_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> tuple["np.ndarray", int]:
    resolved_source_path = _resolve_existing_path(source_path)
    if _can_load_wav_direct(resolved_source_path, int(sample_rate), int(channels)):
        return _load_wav_array(resolved_source_path)

    temporary_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary_wav.close()
    try:
        _normalize_audio_to_wav(
            source_path=resolved_source_path,
            output_path=temporary_wav.name,
            sample_rate=sample_rate,
            channels=channels,
        )
        return _load_wav_array(temporary_wav.name)
    finally:
        if os.path.exists(temporary_wav.name):
            os.unlink(temporary_wav.name)


_REFERENCE_AUDIO_CACHE: Dict[tuple[str, int, int], tuple["np.ndarray", int]] = {}
_REFERENCE_CHOP_CANDIDATE_CACHE: Dict[tuple[Any, ...], List[tuple[float, int, Dict[str, float]]]] = {}
_SECTION_MIX_PROFILE_CACHE: Dict[tuple[Any, ...], Dict[str, Any]] = {}


def _load_reference_audio_array(
    source_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> tuple["np.ndarray", int]:
    resolved_path = os.path.abspath(_resolve_existing_path(source_path))
    key = (resolved_path, int(sample_rate), int(channels))
    cached = _REFERENCE_AUDIO_CACHE.get(key)
    if cached is None:
        cached = _load_audio_array(resolved_path, sample_rate=int(sample_rate), channels=int(channels))
        _REFERENCE_AUDIO_CACHE[key] = (cached[0].copy(), int(cached[1]))
    samples, resolved_sample_rate = cached
    return samples.copy(), int(resolved_sample_rate)


def _rank_reference_chop_candidates(
    reference_path: str,
    stereo: "np.ndarray",
    sample_rate: int,
    chop_samples: int,
    style_tokens: Iterable[str],
    variant: str,
) -> List[tuple[float, int, Dict[str, float]]]:
    resolved_path = os.path.abspath(_resolve_existing_path(reference_path))
    normalized_tokens = tuple(sorted({str(token).lower() for token in style_tokens if str(token).strip()}))
    cache_key = (
        resolved_path,
        int(sample_rate),
        int(chop_samples),
        _clean_text(variant).lower() or "stab",
        normalized_tokens,
    )
    cached = _REFERENCE_CHOP_CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return [(float(score), int(start), dict(components)) for score, start, components in cached]

    if np is None:
        return []
    search_start = int(stereo.shape[0] * 0.18)
    search_end = max(search_start + 1, int(stereo.shape[0] * 0.82))
    step = max(512, int(chop_samples) // 2)
    ranked_candidates: List[tuple[float, int, Dict[str, float]]] = []
    resolved_variant = _clean_text(variant).lower() or "stab"
    for start in range(search_start, min(search_end, stereo.shape[0] - int(chop_samples)), step):
        window = stereo[start : start + int(chop_samples)]
        score, components = _score_reference_crate_window(
            window,
            role="chop",
            sample_rate=sample_rate,
            style_tokens=normalized_tokens,
            variant=resolved_variant,
        )
        if score <= 0.0:
            continue
        ranked_candidates.append((float(score), int(start), dict(components)))
    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    _REFERENCE_CHOP_CANDIDATE_CACHE[cache_key] = [(float(score), int(start), dict(components)) for score, start, components in ranked_candidates]
    return [(float(score), int(start), dict(components)) for score, start, components in ranked_candidates]


def _select_representative_sample_layers(
    role_by_name: Dict[str, str],
    available_layers: Iterable[str],
) -> List[str]:
    selected: List[str] = []
    seen: Set[str] = set()
    available = [name for name in available_layers if name in role_by_name]
    if not available:
        return selected

    def _add(name: str) -> None:
        if name and name not in seen:
            selected.append(name)
            seen.add(name)

    priority_exact = [
        "melody",
        "texture",
        "chop",
        "chop-hook",
        "chop-phrase",
        "resample-motif",
        "tonal-glue",
        "hook-memory-bus",
        "sample-groove-response-bus",
    ]
    for priority_name in priority_exact:
        for layer_name in available:
            if _clean_text(layer_name).lower() == priority_name:
                _add(layer_name)

    chop_candidates = [name for name in available if role_by_name.get(name) == "chop" and name not in seen]
    for keyword in ("hook", "phrase", "answer", "transition", "alt"):
        for layer_name in chop_candidates:
            if keyword in _clean_text(layer_name).lower():
                _add(layer_name)
                break

    role_limits = {
        "melody": 1,
        "lead": 1,
        "texture": 1,
        "resample": 1,
        "tonal_glue": 1,
        "hook_memory": 1,
        "sample_groove": 1,
        "chop": 3,
    }
    counts: Dict[str, int] = {}
    filtered: List[str] = []
    for layer_name in selected + available:
        role_name = role_by_name.get(layer_name, "")
        limit = int(role_limits.get(role_name, 1))
        current = int(counts.get(role_name, 0))
        if current >= limit:
            continue
        if layer_name in filtered:
            continue
        filtered.append(layer_name)
        counts[role_name] = current + 1
    return filtered


def _select_sample_stack_focus_layers(
    role_by_name: Dict[str, str],
    available_layers: Iterable[str],
    include_response_bus: bool = False,
    max_layers: Optional[int] = None,
) -> List[str]:
    available = [name for name in available_layers if name in role_by_name]
    if not available:
        return []
    resolved_max_layers = max(1, int(max_layers or (5 if include_response_bus else 4)))

    selected: List[str] = []
    seen: Set[str] = set()

    def _add_first(predicate: Any) -> None:
        if len(selected) >= resolved_max_layers:
            return
        for layer_name in available:
            if layer_name in seen:
                continue
            if predicate(layer_name):
                selected.append(layer_name)
                seen.add(layer_name)
                return

    _add_first(lambda name: role_by_name.get(name) in {"melody", "lead"})
    _add_first(lambda name: role_by_name.get(name) == "chop" and "hook" in _clean_text(name).lower())
    _add_first(lambda name: role_by_name.get(name) == "chop")
    _add_first(lambda name: role_by_name.get(name) == "texture")
    _add_first(lambda name: role_by_name.get(name) == "resample")
    _add_first(lambda name: role_by_name.get(name) == "tonal_glue")
    if include_response_bus:
        _add_first(lambda name: role_by_name.get(name) == "sample_groove")

    for layer_name in available:
        if len(selected) >= resolved_max_layers:
            break
        if layer_name in seen:
            continue
        selected.append(layer_name)
        seen.add(layer_name)
    return selected


def _select_representative_drum_layers(
    role_by_name: Dict[str, str],
    available_layers: Iterable[str],
) -> List[str]:
    selected: List[str] = []
    seen: Set[str] = set()
    available = [name for name in available_layers if name in role_by_name]
    if not available:
        return selected

    def _add(name: str) -> None:
        if name and name not in seen:
            selected.append(name)
            seen.add(name)

    priority_exact = [
        "kick",
        "snare",
        "clap",
        "hat",
        "perc",
        "drum-bed",
        "drum-fill",
        "drum-variation",
    ]
    for priority_name in priority_exact:
        for layer_name in available:
            if _clean_text(layer_name).lower() == priority_name:
                _add(layer_name)

    role_limits = {
        "kick": 1,
        "snare": 1,
        "clap": 1,
        "hat": 1,
        "hihat": 1,
        "perc": 1,
        "drum_bed": 1,
        "drum_fill": 1,
        "drum_variation": 1,
    }
    counts: Dict[str, int] = {}
    filtered: List[str] = []
    for layer_name in selected + available:
        role_name = role_by_name.get(layer_name, "")
        limit = int(role_limits.get(role_name, 1))
        current = int(counts.get(role_name, 0))
        if current >= limit:
            continue
        if layer_name in filtered:
            continue
        filtered.append(layer_name)
        counts[role_name] = current + 1
    return filtered


def _select_drum_character_layers(
    role_by_name: Dict[str, str],
    available_layers: Iterable[str],
) -> List[str]:
    available = [name for name in available_layers if name in role_by_name]
    if not available:
        return []

    selected: List[str] = []
    seen: Set[str] = set()

    def _add_first(predicate: Any) -> None:
        for layer_name in available:
            if layer_name in seen:
                continue
            if predicate(layer_name):
                selected.append(layer_name)
                seen.add(layer_name)
                return

    _add_first(lambda name: role_by_name.get(name) == "kick")
    _add_first(lambda name: role_by_name.get(name) == "snare")
    _add_first(lambda name: role_by_name.get(name) == "clap")
    _add_first(lambda name: role_by_name.get(name) in {"hat", "hihat"})
    _add_first(lambda name: role_by_name.get(name) == "drum_variation")
    _add_first(lambda name: role_by_name.get(name) == "perc")
    _add_first(lambda name: role_by_name.get(name) == "drum_fill")
    _add_first(lambda name: role_by_name.get(name) == "drum_bed")

    prioritized: List[str] = []
    prioritized_seen: Set[str] = set()
    for layer_name in selected:
        role_name = role_by_name.get(layer_name, "")
        if role_name == "clap" and any(role_by_name.get(existing) == "snare" for existing in prioritized):
            continue
        if role_name == "drum_fill" and any(role_by_name.get(existing) == "drum_variation" for existing in prioritized):
            continue
        if role_name == "perc" and any(role_by_name.get(existing) == "drum_variation" for existing in prioritized):
            continue
        if layer_name not in prioritized_seen:
            prioritized.append(layer_name)
            prioritized_seen.add(layer_name)
        if len(prioritized) >= 4:
            break

    for layer_name in available:
        if len(prioritized) >= 4:
            break
        if layer_name in prioritized_seen:
            continue
        prioritized.append(layer_name)
        prioritized_seen.add(layer_name)
    return prioritized


def _write_audio_array_to_wav(
    samples: "np.ndarray",
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]

    clipped = np.clip(array, -1.0, 1.0)
    int_samples = (clipped * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(int(array.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(int_samples.tobytes())
    return output_path


def _delete_file_if_exists(path: Optional[str]) -> bool:
    target_path = _clean_text(path)
    if not target_path or not os.path.isfile(target_path):
        return False
    os.remove(target_path)
    return True


def _build_expiration_timestamp(hours: int) -> Optional[str]:
    if int(hours or 0) <= 0:
        return None
    return (datetime.now() + timedelta(hours=int(hours))).isoformat(timespec="seconds")


def _linear_from_db(value_db: float) -> float:
    return float(10 ** (float(value_db) / 20.0))


def _hygienize_audio_capture(
    input_path: str,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    silence_threshold_db: float = -42.0,
    trim_padding_ms: int = 120,
    normalize_peak_db: float = DEFAULT_TARGET_PEAK_DB,
) -> Dict[str, Any]:
    samples, resolved_sample_rate = _load_audio_array(
        input_path,
        sample_rate=sample_rate,
        channels=channels,
    )
    stereo = _ensure_stereo(samples)
    mono_envelope = np.max(np.abs(stereo), axis=1)
    threshold = _linear_from_db(silence_threshold_db)
    active_indices = np.flatnonzero(mono_envelope >= threshold)
    padding_samples = int(max(0, trim_padding_ms) * resolved_sample_rate / 1000)

    if active_indices.size:
        start_index = max(0, int(active_indices[0]) - padding_samples)
        end_index = min(int(stereo.shape[0]), int(active_indices[-1]) + 1 + padding_samples)
        cleaned = stereo[start_index:end_index]
    else:
        start_index = 0
        end_index = int(stereo.shape[0])
        cleaned = stereo

    peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    gain_applied = 1.0
    target_peak_linear = _linear_from_db(normalize_peak_db)
    if peak > 0:
        gain_applied = target_peak_linear / peak
        cleaned = np.clip(cleaned * gain_applied, -1.0, 1.0)

    _write_audio_array_to_wav(cleaned, output_path, sample_rate=resolved_sample_rate)
    duration_seconds = float(cleaned.shape[0] / resolved_sample_rate) if resolved_sample_rate else 0.0
    return {
        "output_path": output_path,
        "trimmed_start_ms": round(start_index * 1000.0 / resolved_sample_rate, 2) if resolved_sample_rate else 0.0,
        "trimmed_end_ms": round((stereo.shape[0] - end_index) * 1000.0 / resolved_sample_rate, 2)
        if resolved_sample_rate
        else 0.0,
        "peak_before": round(peak, 6),
        "gain_applied": round(gain_applied, 6),
        "duration_seconds": round(duration_seconds, 4),
        "sample_rate": resolved_sample_rate,
        "channels": int(cleaned.shape[1]) if cleaned.ndim == 2 else 1,
    }


def _ffprobe_audio_metadata(source_path: str) -> Dict[str, Any]:
    status = ensure_runtime_ready("analysis")
    ffprobe_path = status["ffprobe_path"]
    result = _run_command(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            source_path,
        ]
    )
    if result.returncode != 0:
        _raise_command_error("ffprobe", result)
    return json.loads(result.stdout or "{}")


def _to_db(value: float, floor: float = -120.0) -> float:
    if value <= 0:
        return floor
    return float(20.0 * math.log10(value))


def _pan_gains(pan: float) -> tuple[float, float]:
    clean_pan = max(-1.0, min(1.0, float(pan)))
    left_gain = 1.0 if clean_pan <= 0 else 1.0 - clean_pan
    right_gain = 1.0 if clean_pan >= 0 else 1.0 + clean_pan
    return left_gain, right_gain


def _ensure_stereo(samples: "np.ndarray") -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    return array


def _apply_gain_and_pan(samples: "np.ndarray", gain_db: float = 0.0, pan: float = 0.0) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    stereo = _ensure_stereo(samples).copy()
    gain = float(10 ** (float(gain_db) / 20.0))
    stereo *= gain
    left_gain, right_gain = _pan_gains(pan)
    stereo[:, 0] *= left_gain
    stereo[:, 1] *= right_gain
    return stereo


def _parse_pattern(pattern: str, bars: int, steps_per_bar: int) -> List[bool]:
    total_steps = max(1, int(bars)) * max(1, int(steps_per_bar))
    clean_pattern = "".join(char for char in str(pattern or "") if not char.isspace() and char != "|")
    if not clean_pattern:
        clean_pattern = "x" + "-" * (steps_per_bar - 1)

    if len(clean_pattern) == steps_per_bar:
        clean_pattern = clean_pattern * bars
    elif len(clean_pattern) < total_steps:
        repeats = math.ceil(total_steps / len(clean_pattern))
        clean_pattern = (clean_pattern * repeats)[:total_steps]
    else:
        clean_pattern = clean_pattern[:total_steps]

    return [char in ACTIVE_PATTERN_CHARS for char in clean_pattern]


def _pattern_steps_to_text(steps: Iterable[bool]) -> str:
    return "".join("x" if bool(step) else "-" for step in steps)


def _resolve_total_bars_for_duration(
    tempo_bpm: float,
    target_duration_seconds: Optional[float],
    minimum_bars: int,
) -> int:
    resolved_minimum_bars = max(1, int(minimum_bars))
    if target_duration_seconds is None:
        return resolved_minimum_bars
    try:
        requested_duration = float(target_duration_seconds)
    except Exception:
        return resolved_minimum_bars
    if requested_duration <= 0 or tempo_bpm <= 0:
        return resolved_minimum_bars
    seconds_per_bar = 240.0 / float(tempo_bpm)
    estimated_bars = int(math.ceil(requested_duration / seconds_per_bar))
    return max(resolved_minimum_bars, estimated_bars)


def _build_arrangement_sections(total_bars: int) -> List[Dict[str, Any]]:
    def _section(
        name: str,
        bars: int,
        *,
        song_role: Optional[str] = None,
        section_family: Optional[str] = None,
        energy: float,
        drum_density: float,
        melodic_density: float,
        chop_density: float,
        bass_density: float,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "bars": int(max(1, bars)),
            "song_role": _clean_text(song_role or name).lower() or name,
            "section_family": _clean_text(section_family or song_role or name).lower() or name,
            "energy": float(energy),
            "drum_density": float(drum_density),
            "melodic_density": float(melodic_density),
            "chop_density": float(chop_density),
            "bass_density": float(bass_density),
        }

    total_bars = max(1, int(total_bars))
    if total_bars <= 4:
        return [
            _section(
                "main_a",
                total_bars,
                song_role="hook",
                section_family="core_loop",
                energy=0.92,
                drum_density=0.92,
                melodic_density=0.9,
                chop_density=0.82,
                bass_density=0.96,
            )
        ]
    if total_bars <= 8:
        intro_bars = 1
        outro_bars = 1
        main_a_bars = max(2, total_bars - intro_bars - outro_bars)
        return [
            _section("intro", intro_bars, song_role="intro", section_family="intro", energy=0.62, drum_density=0.55, melodic_density=0.5, chop_density=0.42, bass_density=0.74),
            _section("main_a", main_a_bars, song_role="hook", section_family="core_loop", energy=0.94, drum_density=0.92, melodic_density=0.9, chop_density=0.88, bass_density=0.98),
            _section("outro", outro_bars, song_role="outro", section_family="outro", energy=0.54, drum_density=0.48, melodic_density=0.46, chop_density=0.36, bass_density=0.66),
        ]

    if total_bars >= 24:
        intro_bars = 4
        fill_bars = 1
        break_bars = 2
        outro_bars = 2
        remaining = total_bars - intro_bars - fill_bars - break_bars - outro_bars
        main_a_bars = max(6, int(round(remaining * 0.37)))
        main_b_bars = max(5, int(round(remaining * 0.32)))
        main_c_bars = max(4, remaining - main_a_bars - main_b_bars)
        overflow = intro_bars + main_a_bars + fill_bars + main_b_bars + break_bars + main_c_bars + outro_bars - total_bars
        if overflow > 0:
            main_a_bars = max(6, main_a_bars - overflow)
            main_c_bars = max(4, total_bars - intro_bars - main_a_bars - fill_bars - main_b_bars - break_bars - outro_bars)
        return [
            _section("intro", intro_bars, song_role="intro", section_family="intro", energy=0.6, drum_density=0.52, melodic_density=0.46, chop_density=0.38, bass_density=0.72),
            _section("main_a", main_a_bars, song_role="verse", section_family="verse", energy=0.88, drum_density=0.84, melodic_density=0.8, chop_density=0.74, bass_density=0.94),
            _section("fill", fill_bars, song_role="pre_hook", section_family="transition", energy=1.08, drum_density=1.18, melodic_density=0.68, chop_density=1.06, bass_density=0.84),
            _section("main_b", main_b_bars, song_role="hook", section_family="hook", energy=1.0, drum_density=1.02, melodic_density=0.96, chop_density=0.98, bass_density=1.0),
            _section("break", break_bars, song_role="break", section_family="break", energy=0.54, drum_density=0.44, melodic_density=0.36, chop_density=0.52, bass_density=0.62),
            _section("main_c", main_c_bars, song_role="hook", section_family="hook_variation", energy=1.06, drum_density=1.08, melodic_density=1.04, chop_density=1.1, bass_density=1.02),
            _section("outro", outro_bars, song_role="outro", section_family="outro", energy=0.5, drum_density=0.42, melodic_density=0.34, chop_density=0.3, bass_density=0.58),
        ]

    intro_bars = 2
    outro_bars = 1
    fill_bars = 1
    remaining = total_bars - intro_bars - outro_bars - fill_bars
    main_a_bars = max(3, int(math.ceil(remaining * 0.5)))
    main_b_bars = max(2, remaining - main_a_bars)
    overflow = intro_bars + outro_bars + fill_bars + main_a_bars + main_b_bars - total_bars
    if overflow > 0:
        main_a_bars = max(2, main_a_bars - overflow)

    return [
        _section("intro", intro_bars, song_role="intro", section_family="intro", energy=0.68, drum_density=0.6, melodic_density=0.56, chop_density=0.45, bass_density=0.78),
        _section("main_a", main_a_bars, song_role="verse", section_family="verse", energy=0.94, drum_density=0.9, melodic_density=0.88, chop_density=0.84, bass_density=0.98),
        _section("fill", fill_bars, song_role="pre_hook", section_family="transition", energy=1.08, drum_density=1.16, melodic_density=0.72, chop_density=1.08, bass_density=0.86),
        _section("main_b", main_b_bars, song_role="hook", section_family="hook", energy=1.02, drum_density=1.04, melodic_density=1.0, chop_density=1.02, bass_density=1.0),
        _section("outro", outro_bars, song_role="outro", section_family="outro", energy=0.56, drum_density=0.5, melodic_density=0.48, chop_density=0.38, bass_density=0.68),
    ]


def _resolve_section_song_role(section: Optional[Dict[str, Any]]) -> str:
    song_role = _clean_text((section or {}).get("song_role")).lower()
    if song_role:
        return song_role
    section_name = _clean_text((section or {}).get("name")).lower()
    return {
        "main_a": "verse",
        "fill": "pre_hook",
        "main_b": "hook",
        "break": "break",
        "main_c": "hook",
    }.get(section_name, section_name or "verse")


def _resolve_section_family(section: Optional[Dict[str, Any]]) -> str:
    family = _clean_text((section or {}).get("section_family")).lower()
    if family:
        return family
    song_role = _resolve_section_song_role(section)
    return {
        "hook": "hook",
        "pre_hook": "transition",
        "break": "break",
        "outro": "outro",
        "intro": "intro",
    }.get(song_role, "verse")


def _midi_to_frequency(midi_note: float) -> float:
    return float(440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0)))


def _frequency_to_midi(frequency_hz: float) -> float:
    return float(69.0 + (12.0 * math.log2(max(float(frequency_hz), 1e-6) / 440.0)))


def _key_name_to_pitch_class(key_name: str) -> Optional[int]:
    normalized = _clean_text(key_name).upper().replace("♯", "#").replace("♭", "B")
    mapping = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    return mapping.get(normalized)


def _pitch_class_to_key_name(pitch_class: Optional[int]) -> Optional[str]:
    if pitch_class is None:
        return None
    mapping = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
    }
    return mapping.get(int(pitch_class) % 12)


def _transpose_key_name(key_name: Optional[str], semitone_shift: int) -> Optional[str]:
    pitch_class = _key_name_to_pitch_class(key_name or "")
    if pitch_class is None:
        return None
    return _pitch_class_to_key_name(int(pitch_class) + int(semitone_shift))


def _scale_intervals(scale_name: str) -> List[int]:
    normalized = _clean_text(scale_name).lower()
    if normalized in {"minor", "aeolian", "dorian", "phrygian"}:
        return [0, 2, 3, 5, 7, 8, 10]
    return [0, 2, 4, 5, 7, 9, 11]


def _nearest_scale_midi(target_midi: float, root_pitch_class: int, scale_intervals: List[int]) -> int:
    base = int(round(target_midi))
    best_candidate = base
    best_distance = float("inf")
    for candidate in range(base - 24, base + 25):
        pitch_class = candidate % 12
        if ((pitch_class - root_pitch_class) % 12) not in scale_intervals:
            continue
        distance = abs(candidate - target_midi)
        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate
    return int(best_candidate)


def _extract_reference_pitch_context(blueprint: Dict[str, Any], sub_frequency_hz: float) -> Dict[str, Any]:
    render_overrides = dict(blueprint.get("_render_overrides") or {})
    key_name = ""
    scale_name = ""
    key_confidence = None
    source_path = ""
    anchor_locked = False
    for summary in blueprint.get("reference_summaries") or []:
        analysis = summary.get("analysis") or {}
        key_name = _clean_text(analysis.get("estimated_key"))
        scale_name = _clean_text(analysis.get("estimated_scale"))
        key_confidence = analysis.get("estimated_key_confidence")
        source_path = _clean_text(summary.get("path"))
        if key_name:
            break

    override_key_name = _clean_text(render_overrides.get("tonal_anchor_key"))
    override_scale_name = _clean_text(render_overrides.get("tonal_anchor_scale"))
    override_confidence = render_overrides.get("tonal_anchor_confidence")
    if override_key_name:
        key_name = override_key_name
        source_path = _clean_text(render_overrides.get("tonal_anchor_source")) or source_path
        anchor_locked = True
    if override_scale_name:
        scale_name = override_scale_name
        anchor_locked = True
    if override_confidence is not None:
        key_confidence = override_confidence

    try:
        normalized_confidence = float(key_confidence) if key_confidence is not None else None
    except Exception:
        normalized_confidence = None

    if (
        not anchor_locked
        and source_path
        and os.path.exists(source_path)
        and (not key_name or normalized_confidence is None or normalized_confidence < 0.1)
    ):
        refreshed = _compute_librosa_descriptors(source_path, sample_rate=22050)
        if refreshed:
            refreshed_key = _clean_text(refreshed.get("estimated_key"))
            refreshed_scale = _clean_text(refreshed.get("estimated_scale"))
            refreshed_confidence = refreshed.get("estimated_key_confidence")
            if refreshed_key:
                key_name = refreshed_key
            if refreshed_scale:
                scale_name = refreshed_scale
            try:
                normalized_confidence = float(refreshed_confidence) if refreshed_confidence is not None else normalized_confidence
            except Exception:
                pass

    style_tags = {tag.lower() for tag in _normalize_instrumental_style_tags(blueprint.get("style_tags"))}
    if not scale_name:
        scale_name = "minor" if any(token in style_tags for token in ("dark", "boom bap", "old school", "sub-heavy")) else "major"

    root_pitch_class = _key_name_to_pitch_class(key_name)
    if root_pitch_class is None:
        root_pitch_class = int(round(_frequency_to_midi(max(sub_frequency_hz, 55.0)))) % 12

    scale_intervals = _scale_intervals(scale_name)
    bass_root_midi = _nearest_scale_midi(_frequency_to_midi(max(sub_frequency_hz, 50.0)), root_pitch_class, scale_intervals)
    if bass_root_midi < 34:
        bass_root_midi += 12
    elif bass_root_midi > 48:
        bass_root_midi -= 12

    melody_root_midi = _nearest_scale_midi(bass_root_midi + 24, root_pitch_class, scale_intervals)
    if scale_name.lower() == "minor":
        if any(token in style_tags for token in ("dark", "boom bap", "old school", "griselda", "wu tang")):
            section_progressions = {
                "intro": [0, 5],
                "main_a": [0, 5, 3, 4],
                "fill": [4, 5],
                "break": [5, 3],
                "main_b": [0, 3, 5, 4],
                "main_c": [0, 6, 5, 4],
                "outro": [0, 5],
            }
        else:
            section_progressions = {
                "intro": [0, 3],
                "main_a": [0, 3, 5, 4],
                "fill": [5, 4],
                "break": [3, 5],
                "main_b": [0, 5, 3, 4],
                "main_c": [0, 4, 5, 3],
                "outro": [0, 3],
            }
    else:
        section_progressions = {
            "intro": [0, 4],
            "main_a": [0, 4, 5, 3],
            "fill": [5, 4],
            "break": [4, 5],
            "main_b": [0, 5, 4, 3],
            "main_c": [0, 3, 4, 5],
            "outro": [0, 4],
        }
    return {
        "estimated_key": key_name or None,
        "estimated_scale": scale_name,
        "estimated_key_confidence": round(float(normalized_confidence), 4) if normalized_confidence is not None else None,
        "anchor_locked": bool(anchor_locked),
        "root_pitch_class": int(root_pitch_class),
        "scale_intervals": scale_intervals,
        "bass_root_midi": int(bass_root_midi),
        "melody_root_midi": int(melody_root_midi),
        "section_progressions": {
            **section_progressions,
            "verse": list(section_progressions.get("main_a") or [0]),
            "pre_hook": list(section_progressions.get("fill") or section_progressions.get("main_b") or [0]),
            "hook": list(section_progressions.get("main_b") or [0]),
            "hook_variation": list(section_progressions.get("main_c") or section_progressions.get("main_b") or [0]),
            "transition": list(section_progressions.get("fill") or [0]),
            "core_loop": list(section_progressions.get("main_a") or [0]),
        },
        "source_path": source_path or None,
    }


def _add_synth_note(
    buffer: "np.ndarray",
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
    frequency_hz: float,
    amplitude: float,
    role: str,
    section_name: Optional[str] = None,
    phrase_function: Optional[str] = None,
    pitch_slide_semitones: float = 0.0,
) -> None:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")
    start_index = int(round(max(0.0, start_seconds) * sample_rate))
    length = max(32, int(round(max(0.03, duration_seconds) * sample_rate)))
    end_index = min(buffer.shape[0], start_index + length)
    if end_index <= start_index:
        return

    t = np.arange(end_index - start_index, dtype=np.float32) / float(sample_rate)
    role_name = (_clean_text(role) or "melody").lower()
    section_type = (_clean_text(section_name) or "main_a").lower()
    phrase_type = _clean_text(phrase_function).lower()
    if role_name == "bass":
        section_drive = {
            "intro": 1.06,
            "verse": 1.18,
            "pre_hook": 1.12,
            "hook": 1.24,
            "main_a": 1.18,
            "fill": 1.12,
            "main_b": 1.24,
            "break": 0.98,
            "hook_variation": 1.28,
            "main_c": 1.28,
            "outro": 1.02,
        }.get(section_type, 1.18)
        overtone_gain = {
            "intro": 0.16,
            "verse": 0.24,
            "pre_hook": 0.19,
            "hook": 0.28,
            "main_a": 0.24,
            "fill": 0.19,
            "main_b": 0.28,
            "break": 0.12,
            "hook_variation": 0.3,
            "main_c": 0.3,
            "outro": 0.15,
        }.get(section_type, 0.24)
        grit_gain = {
            "intro": 0.04,
            "verse": 0.08,
            "pre_hook": 0.06,
            "hook": 0.1,
            "main_a": 0.08,
            "fill": 0.06,
            "main_b": 0.1,
            "break": 0.03,
            "hook_variation": 0.11,
            "main_c": 0.11,
            "outro": 0.04,
        }.get(section_type, 0.08)
        slide = float(pitch_slide_semitones or 0.0)
        if phrase_type in {"approach_up", "slide_answer"} and abs(slide) < 1e-6:
            slide = -2.0
        elif phrase_type in {"turnaround", "drop_return"} and abs(slide) < 1e-6:
            slide = 2.0
        if abs(slide) > 1e-6:
            slide_len = max(1, min(t.size, int(round(t.size * (0.34 if phrase_type == "turnaround" else 0.24)))))
            slide_curve = np.ones_like(t, dtype=np.float32)
            slide_curve[:slide_len] = np.linspace(2.0 ** (slide / 12.0), 1.0, slide_len, dtype=np.float32)
            freq_curve = np.maximum(20.0, frequency_hz * slide_curve)
            phase = 2.0 * np.pi * np.cumsum(freq_curve).astype(np.float32) / float(sample_rate)
            overtone_phase = 2.0 * np.pi * np.cumsum(freq_curve * 2.0).astype(np.float32) / float(sample_rate)
            grit_phase = 2.0 * np.pi * np.cumsum(freq_curve * 3.0).astype(np.float32) / float(sample_rate)
        else:
            phase = 2.0 * np.pi * frequency_hz * t
            overtone_phase = 2.0 * np.pi * frequency_hz * 2.0 * t
            grit_phase = 2.0 * np.pi * frequency_hz * 3.0 * t
        body = np.sin(phase)
        overtone = overtone_gain * np.sin(overtone_phase + 0.25)
        grit = grit_gain * np.sin(grit_phase + 0.18)
        click = 0.0
        if section_type in {"main_b", "main_c", "fill", "hook", "pre_hook"} or phrase_type in {"approach_up", "turnaround"}:
            click_env = np.exp(-t * 55.0)
            click = 0.018 * np.sin(2.0 * np.pi * frequency_hz * 6.0 * t + 0.35) * click_env
        signal_mono = np.tanh((body + overtone + grit + click) * section_drive)
        if phrase_type in {"muted_pickup", "ghost_low"}:
            signal_mono = np.tanh(signal_mono * 0.82)
        attack = 0.006 if phrase_type in {"approach_up", "turnaround"} else 0.007 if section_type in {"main_b", "main_c", "fill", "hook"} else 0.009
        release = 0.09 if phrase_type in {"muted_pickup", "ghost_low"} else 0.12 if section_type == "break" else 0.16 if section_type in {"intro", "outro"} else 0.2
    else:
        shimmer_gain = {
            "intro": 0.14,
            "verse": 0.2,
            "pre_hook": 0.22,
            "hook": 0.28,
            "main_a": 0.22,
            "fill": 0.24,
            "main_b": 0.28,
            "break": 0.1,
            "hook_variation": 0.3,
            "main_c": 0.3,
            "outro": 0.14,
        }.get(section_type, 0.22)
        air_gain = {
            "intro": 0.05,
            "verse": 0.08,
            "pre_hook": 0.1,
            "hook": 0.13,
            "main_a": 0.1,
            "fill": 0.12,
            "main_b": 0.14,
            "break": 0.04,
            "hook_variation": 0.16,
            "main_c": 0.16,
            "outro": 0.06,
        }.get(section_type, 0.1)
        body2_gain = {
            "intro": 0.03,
            "verse": 0.055,
            "pre_hook": 0.05,
            "hook": 0.07,
            "main_a": 0.06,
            "fill": 0.05,
            "main_b": 0.07,
            "break": 0.025,
            "hook_variation": 0.08,
            "main_c": 0.08,
            "outro": 0.03,
        }.get(section_type, 0.06)
        drive = {
            "intro": 0.98,
            "verse": 1.06,
            "pre_hook": 1.08,
            "hook": 1.16,
            "main_a": 1.08,
            "fill": 1.06,
            "main_b": 1.16,
            "break": 0.94,
            "hook_variation": 1.2,
            "main_c": 1.2,
            "outro": 1.0,
        }.get(section_type, 1.08)
        body = np.sin(2.0 * np.pi * frequency_hz * t)
        shimmer = shimmer_gain * np.sin(2.0 * np.pi * frequency_hz * 2.0 * t)
        air = air_gain * np.sin(2.0 * np.pi * frequency_hz * 3.0 * t + 0.4)
        body2 = body2_gain * np.sin(2.0 * np.pi * frequency_hz * 4.0 * t + 0.2)
        transient = 0.0
        if section_type in {"fill", "main_b", "main_c"}:
            transient_env = np.exp(-t * 42.0)
            transient = 0.03 * np.sin(2.0 * np.pi * frequency_hz * 5.0 * t + 0.1) * transient_env
        signal_mono = np.tanh((body + shimmer + air + body2 + transient) * drive)
        attack = 0.011 if section_type in {"fill", "main_b", "main_c"} else 0.016
        release = 0.1 if section_type == "break" else 0.14 if section_type in {"intro", "outro"} else 0.18

    env = np.ones_like(t, dtype=np.float32)
    attack_len = min(len(env), max(1, int(round(attack * sample_rate))))
    env[:attack_len] *= np.linspace(0.0, 1.0, attack_len, dtype=np.float32)
    release_len = min(len(env), max(1, int(round(release * sample_rate))))
    env[-release_len:] *= np.linspace(1.0, 0.0, release_len, dtype=np.float32)
    sustain_curve = np.exp(-t * ((2.8 if section_type == "break" else 2.2 if role_name == "bass" else 3.9 if section_type == "break" else 3.0)))
    mono = signal_mono * env * sustain_curve * float(amplitude)
    if role_name == "bass":
        stereo = _stereo_from_mono(mono)
    else:
        width = {
            "intro": 0.1,
            "main_a": 0.18,
            "fill": 0.2,
            "main_b": 0.28,
            "break": 0.08,
            "main_c": 0.32,
            "outro": 0.12,
        }.get(section_type, 0.18)
        detune = 1.003 if section_type in {"main_b", "main_c"} else 1.0018
        left = mono + (np.sin(2.0 * np.pi * frequency_hz * detune * t + 0.08) * (0.045 + width * 0.08))
        right = mono + (np.sin(2.0 * np.pi * frequency_hz * (2.0 - detune) * t - 0.05) * (0.04 + width * 0.075))
        stereo = np.stack([left, right], axis=1).astype(np.float32)
        stereo = _normalize_audio_buffer(stereo, peak=0.95)
    buffer[start_index:end_index] += stereo[: end_index - start_index]


def _resolve_boombap_bass_pocket_events(
    section_song_role: str,
    section_family: str,
    bar_index: int,
    bars: int,
    steps_per_bar: int,
    creativity: float,
    rng: "np.random.Generator",
) -> List[Dict[str, Any]]:
    quarter = max(1, int(steps_per_bar) // 4)
    eighth = max(1, int(steps_per_bar) // 8)
    sixteenth = max(1, int(steps_per_bar) // 16)
    role = _clean_text(section_song_role).lower() or _clean_text(section_family).lower() or "verse"
    bar_phase = int(bar_index) % 4
    is_last_bar = int(bar_index) >= max(1, int(bars)) - 1

    if role == "intro":
        template = [(0, "root_anchor", 6, 1.0, 0, 0.0)]
        if is_last_bar:
            template.append((max(0, steps_per_bar - quarter), "approach_up", 2, 0.74, 4, -2.0))
    elif role == "verse":
        templates = [
            [(0, "kick_lock", 5, 1.0, 0, 0.0), (quarter + eighth, "pocket_answer", 2, 0.72, 3, 0.0), (steps_per_bar - quarter, "muted_pickup", 2, 0.58, 4, -1.0)],
            [(0, "root_anchor", 4, 0.92, 0, 0.0), (quarter * 2 + eighth, "pocket_answer", 2, 0.68, 3, 0.0)],
            [(0, "kick_lock", 6, 1.02, 0, 0.0), (quarter * 2, "ghost_low", 2, 0.52, 5, 0.0), (steps_per_bar - eighth, "approach_up", 2, 0.72, 4, -2.0)],
            [(0, "root_anchor", 4, 0.94, 0, 0.0), (quarter + eighth, "pocket_answer", 2, 0.7, 3, 0.0), (steps_per_bar - quarter, "turnaround", 3, 0.82, 5, 2.0)],
        ]
        template = list(templates[bar_phase])
    elif role == "pre_hook":
        template = [
            (0, "kick_lock", 4, 0.94, 0, 0.0),
            (quarter + sixteenth, "approach_up", 2, 0.68, 4, -2.0),
            (quarter * 2 + eighth, "pocket_answer", 2, 0.74, 5, 0.0),
            (steps_per_bar - eighth, "turnaround", 2, 0.78, 4, 2.0),
        ]
    elif role == "hook":
        templates = [
            [(0, "kick_lock", 5, 1.08, 0, 0.0), (quarter, "short_answer", 2, 0.68, 4, 0.0), (quarter * 2 + eighth, "pocket_answer", 2, 0.78, 3, 0.0), (steps_per_bar - quarter, "approach_up", 2, 0.7, 5, -2.0)],
            [(0, "root_anchor", 5, 1.0, 0, 0.0), (quarter + eighth, "pocket_answer", 2, 0.72, 4, 0.0), (quarter * 3, "turnaround", 3, 0.84, 3, 2.0)],
        ]
        template = list(templates[bar_phase % 2])
    elif role == "break":
        template = [(0, "drop_return", 4, 0.74, 0, 2.0)]
        if is_last_bar:
            template.append((steps_per_bar - quarter, "approach_up", 2, 0.58, 5, -2.0))
    elif role == "outro":
        template = [(0, "root_anchor", 5, 0.78, 0, 0.0)]
        if bar_phase % 2 == 1:
            template.append((quarter * 2, "ghost_low", 2, 0.48, 3, 0.0))
    else:
        template = [(0, "kick_lock", 5, 1.0, 0, 0.0), (quarter * 2, "pocket_answer", 2, 0.68, 3, 0.0)]

    if role in {"verse", "hook"} and rng.random() < creativity * 0.22:
        template.append((max(0, steps_per_bar - eighth - sixteenth), "muted_pickup", 1, 0.46, 4, -1.0))
    if role == "verse" and bar_phase == 1 and rng.random() < 0.28:
        template = [item for item in template if item[1] != "muted_pickup"]
    if is_last_bar and role in {"verse", "pre_hook", "hook"}:
        template.append((max(0, steps_per_bar - 2), "turnaround", 1, 0.52, 5, 2.0))

    seen_steps: set[int] = set()
    events: List[Dict[str, Any]] = []
    for step, function_name, duration_steps, amplitude_scale, degree, slide in template:
        resolved_step = max(0, min(int(steps_per_bar) - 1, int(step)))
        if resolved_step in seen_steps:
            continue
        seen_steps.add(resolved_step)
        events.append(
            {
                "step": resolved_step,
                "phrase_function": function_name,
                "duration_steps": max(1, int(duration_steps)),
                "amplitude_scale": float(amplitude_scale),
                "degree": int(degree),
                "pitch_slide_semitones": float(slide),
            }
        )
    return events


def _build_boom_bap_motif_memory(
    pitch_context: Dict[str, Any],
    sections: List[Dict[str, Any]],
    creativity: float,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng((int(seed) ^ 0x5A17B11D) & 0xFFFFFFFF)
    scale_intervals = list(pitch_context.get("scale_intervals") or [0, 2, 3, 5, 7, 8, 10])
    motif_presets = [
        [0, 2, 4, 2, 0, 5, 4, 2],
        [0, 3, 4, 2, 0, 5, 3, 2],
        [0, 2, 5, 4, 2, 0, 4, 2],
        [0, 4, 5, 4, 2, 0, 3, 2],
    ]
    answer_presets = [
        [4, 2, 0, 2, 4, 2],
        [5, 4, 2, 0, 2, 4],
        [4, 5, 4, 2, 0, 2],
    ]
    hook_presets = [
        [0, 4, 5, 7, 5, 4, 2, 0],
        [0, 5, 7, 5, 4, 2, 5, 4],
        [0, 4, 7, 5, 4, 2, 7, 5],
    ]
    fragment_presets = [
        [0, 2, 4],
        [4, 2, 0],
        [0, 5, 4],
        [5, 4, 2],
    ]
    anchor = list(motif_presets[int(rng.integers(0, len(motif_presets)))])
    answer = list(answer_presets[int(rng.integers(0, len(answer_presets)))])
    hook = list(hook_presets[int(rng.integers(0, len(hook_presets)))])
    fragment = list(fragment_presets[int(rng.integers(0, len(fragment_presets)))])
    if creativity >= 0.42 and len(anchor) >= 8:
        swap_index = int(rng.integers(1, min(6, len(anchor))))
        anchor[swap_index] = int((anchor[swap_index] + 1) % max(1, len(scale_intervals)))
    section_roles = [_resolve_section_song_role(section) for section in sections]
    motif_map = {
        "intro": fragment + anchor[:2],
        "verse": anchor,
        "pre_hook": answer + fragment[:2],
        "hook": hook,
        "hook_variation": hook[2:] + hook[:2],
        "break": fragment[::-1],
        "outro": anchor[:4] + [anchor[0]],
        "transition": answer,
        "core_loop": anchor,
    }
    return {
        "anchor": anchor,
        "answer": answer,
        "hook": hook,
        "fragment": fragment,
        "section_map": motif_map,
        "section_roles": section_roles,
    }


def _build_phrase_note_events(
    role: str,
    sections: List[Dict[str, Any]],
    tempo_bpm: float,
    steps_per_bar: int,
    pitch_context: Dict[str, Any],
    creativity: float,
    seed: int,
) -> List[Dict[str, Any]]:
    resolved_role = (_clean_text(role) or "melody").lower()
    scale_intervals = list(pitch_context["scale_intervals"])
    bass_root = int(pitch_context["bass_root_midi"])
    melody_root = int(pitch_context["melody_root_midi"])
    section_progressions = dict(pitch_context.get("section_progressions") or {})
    rng = np.random.default_rng(seed)
    seconds_per_bar = 240.0 / float(tempo_bpm)
    step_seconds = seconds_per_bar / float(steps_per_bar)

    bass_walk_degrees = [0, 0, 3, 0, 4, 3, 0, 5]
    boombap_bass_phrases = {
        "intro": [0, 0, 5, 0],
        "verse": [0, 0, 3, 0, 4, 0, 3, 0],
        "pre_hook": [0, 4, 5, 4, 3, 4],
        "hook": [0, 0, 4, 3, 5, 4, 3, 0],
        "core_loop": [0, 0, 3, 0, 4, 0],
        "break": [0, 5, 0],
        "outro": [0, 3, 0, 5],
    }
    boombap_motif_phrases = {
        "intro": [0, 2, 4, 2, 0],
        "verse": [0, 2, 4, 2, 0, 5, 4, 2],
        "pre_hook": [4, 5, 6, 5, 4, 2, 1],
        "hook": [0, 4, 5, 7, 5, 4, 2, 0],
        "core_loop": [0, 2, 4, 2, 0, 5],
        "break": [4, 2, 0],
        "outro": [0, 2, 4, 2, 0],
    }
    motif_steps = [0, 3, 6, 8, 10, 12, 14]
    motif_memory = _build_boom_bap_motif_memory(
        pitch_context=pitch_context,
        sections=sections,
        creativity=creativity,
        seed=seed,
    )
    events: List[Dict[str, Any]] = []
    current_bar_start = 0.0

    for section_index, section in enumerate(sections):
        section_name = _clean_text(section.get("name")) or f"section_{section_index + 1}"
        section_song_role = _resolve_section_song_role(section)
        section_family = _resolve_section_family(section)
        previous_section_name = _clean_text((sections[section_index - 1] if section_index > 0 else {}).get("name")).lower()
        next_section_name = _clean_text((sections[section_index + 1] if section_index + 1 < len(sections) else {}).get("name")).lower()
        previous_song_role = _resolve_section_song_role(sections[section_index - 1] if section_index > 0 else None)
        next_song_role = _resolve_section_song_role(sections[section_index + 1] if section_index + 1 < len(sections) else None)
        bars = max(1, int(section.get("bars") or 1))
        progression = list(
            section_progressions.get(section_name)
            or section_progressions.get(section_song_role)
            or section_progressions.get(section_family)
            or section_progressions.get("main_a")
            or [0]
        )
        bass_density = float(section.get("bass_density") or 1.0)
        melodic_density = float(section.get("melodic_density") or 1.0)

        for bar_index in range(bars):
            is_first_bar = bar_index == 0
            is_last_bar = bar_index >= bars - 1
            progression_degree = int(progression[bar_index % len(progression)]) if progression else 0
            progression_offset = scale_intervals[progression_degree % len(scale_intervals)]
            if resolved_role == "bass":
                bass_min_midi = bass_root - 5
                bass_max_midi = min(53, bass_root + (7 if section_song_role in {"intro", "break", "outro"} else 9))
                pocket_events = _resolve_boombap_bass_pocket_events(
                    section_song_role=section_song_role,
                    section_family=section_family,
                    bar_index=bar_index,
                    bars=bars,
                    steps_per_bar=steps_per_bar,
                    creativity=creativity,
                    rng=rng,
                )
                if is_first_bar and (previous_section_name == "break" or previous_song_role == "break") and section_song_role == "hook":
                    pocket_events.insert(
                        1,
                        {
                            "step": max(0, steps_per_bar // 8),
                            "phrase_function": "drop_return",
                            "duration_steps": 3,
                            "amplitude_scale": 0.74,
                            "degree": 3,
                            "pitch_slide_semitones": 2.0,
                        },
                    )
                elif is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}):
                    pocket_events = [event for event in pocket_events if event.get("step") in {0, max(0, steps_per_bar // 2), max(0, steps_per_bar - 2)}]
                elif is_last_bar and (next_section_name in {"main_b", "main_c"} or next_song_role == "hook"):
                    pocket_events.append(
                        {
                            "step": max(0, steps_per_bar - max(1, steps_per_bar // 4)),
                            "phrase_function": "approach_up",
                            "duration_steps": 2,
                            "amplitude_scale": 0.76,
                            "degree": 4,
                            "pitch_slide_semitones": -2.0,
                        }
                    )
                seen_pocket_steps: set[int] = set()
                normalized_pocket_events: List[Dict[str, Any]] = []
                for pocket_event in pocket_events:
                    pocket_step = max(0, min(steps_per_bar - 1, int(pocket_event.get("step") or 0)))
                    if pocket_step in seen_pocket_steps:
                        continue
                    seen_pocket_steps.add(pocket_step)
                    normalized_pocket_events.append({**pocket_event, "step": pocket_step})
                for event_index, pocket_event in enumerate(normalized_pocket_events):
                    step = int(pocket_event.get("step") or 0)
                    phrase = boombap_bass_phrases.get(section_song_role) or boombap_bass_phrases.get(section_family) or bass_walk_degrees
                    phrase_offset = (bar_index % 4) * (2 if section_song_role in {"verse", "hook"} else 1)
                    degree = int(pocket_event.get("degree") if pocket_event.get("degree") is not None else phrase[(phrase_offset + event_index) % len(phrase)])
                    if section_song_role == "verse" and bar_index % 4 == 3 and event_index == len(normalized_pocket_events) - 1:
                        degree = 5
                    elif section_song_role == "hook" and bar_index % 2 == 1 and event_index == len(normalized_pocket_events) - 1:
                        degree = 0
                    midi_note = bass_root + progression_offset + scale_intervals[degree % len(scale_intervals)]
                    phrase_function = _clean_text(pocket_event.get("phrase_function")).lower() or ("turnaround" if is_last_bar else "bass_call" if event_index == 0 else "bass_response")
                    if phrase_function in {"ghost_low", "drop_return"}:
                        midi_note -= 12
                    if section_song_role == "pre_hook" and event_index >= 2:
                        midi_note += 12
                    if section_song_role == "hook" and event_index == len(normalized_pocket_events) - 1 and rng.random() < 0.45:
                        midi_note += 12
                    if section_family == "hook_variation" and event_index in {1, 3} and rng.random() < 0.42:
                        midi_note += 12
                    if section_song_role == "intro" and is_last_bar and event_index == len(normalized_pocket_events) - 1:
                        midi_note += 12
                    if section_song_role == "break" and is_last_bar:
                        midi_note -= 12
                    if (next_section_name in {"main_b", "main_c"} or next_song_role == "hook") and is_last_bar and event_index == len(normalized_pocket_events) - 1:
                        midi_note += 12
                    if next_section_name == "outro" and is_last_bar:
                        midi_note -= 12
                    soft_upper = min(52, bass_root + (6 if section_song_role in {"intro", "break", "outro"} else 8))
                    while midi_note > soft_upper:
                        midi_note -= 12
                    midi_note = _nearest_scale_midi(midi_note, pitch_context["root_pitch_class"], scale_intervals)
                    while midi_note > bass_max_midi:
                        midi_note -= 12
                    midi_note = max(bass_min_midi, min(bass_max_midi, midi_note))
                    duration_steps = int(pocket_event.get("duration_steps") or (6 if step == 0 and section_song_role in {"verse", "hook"} else 5 if step == 0 else 3 if section_song_role != "hook" else 2))
                    if section_family == "hook_variation" and step != 0:
                        duration_steps = 2
                    if is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}) and event_index == len(normalized_pocket_events) - 1:
                        duration_steps = max(2, duration_steps - 1)
                    if is_first_bar and (previous_section_name == "break" or previous_song_role == "break") and event_index == 0:
                        duration_steps = max(duration_steps, 6)
                    amplitude = (0.48 if phrase_function in {"kick_lock", "root_anchor"} else 0.36 if phrase_function in {"pocket_answer", "short_answer"} else 0.28) * bass_density
                    amplitude *= float(pocket_event.get("amplitude_scale") or 1.0)
                    if phrase_function in {"ghost_low", "muted_pickup"}:
                        amplitude *= 0.74
                    if section_family == "hook_variation" and event_index in {0, len(normalized_pocket_events) - 1}:
                        amplitude *= 1.08
                    if is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}):
                        amplitude *= 0.92
                    elif is_last_bar and (next_section_name in {"main_b", "main_c"} or next_song_role == "hook"):
                        amplitude *= 1.03
                    if rng.random() < creativity * 0.22:
                        midi_note += 12 if rng.random() < 0.25 else 0
                    events.append(
                        {
                            "start_seconds": round(float(current_bar_start + (step * step_seconds)), 4),
                            "duration_seconds": round(float(duration_steps * step_seconds), 4),
                            "frequency_hz": round(_midi_to_frequency(midi_note), 3),
                            "amplitude": round(float(amplitude), 4),
                            "midi_note": int(midi_note),
                            "section_name": section_name,
                            "song_role": section_song_role,
                            "section_family": section_family,
                            "phrase_function": phrase_function,
                            "pitch_slide_semitones": round(float(pocket_event.get("pitch_slide_semitones") or 0.0), 4),
                            "pocket_step": int(step),
                        }
                    )
            else:
                melody_min_midi = melody_root - (5 if section_song_role in {"intro", "outro"} else 3)
                melody_max_midi = melody_root + (19 if section_song_role == "intro" else 24 if section_song_role in {"verse", "hook"} else 22)
                active_steps_map = {
                    "intro": [8, 12],
                    "verse": [0, 6, 10, 14],
                    "pre_hook": [6, 8, 11, 14],
                    "hook": [0, 3, 7, 10, 12, 14],
                    "core_loop": [0, 4, 8, 12],
                    "main_a": [0, 4, 8, 12],
                    "fill": [8, 10, 12, 14],
                    "main_b": [0, 3, 6, 8, 11, 14],
                    "break": [10],
                    "main_c": [0, 2, 4, 7, 9, 12, 14],
                    "outro": [0, 6],
                }
                active_steps = active_steps_map.get(section_song_role, active_steps_map.get(section_family, active_steps_map.get(section_name, [0, 4, 8, 12])))
                if is_first_bar and (previous_section_name == "break" or previous_song_role == "break") and section_song_role == "hook":
                    active_steps = list(dict.fromkeys([0, 2, 4, 7, 9, 12, 14]))
                elif is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}):
                    active_steps = list(dict.fromkeys([8, 10, 12, 14]))
                elif is_last_bar and (next_section_name in {"main_b", "main_c"} or next_song_role == "hook"):
                    active_steps = list(dict.fromkeys(active_steps + [max(0, steps_per_bar - 2)]))
                motif_pool_map = {
                    "intro": [0, 2, 4, 2, 0, 4],
                    "verse": [0, 2, 4, 2, 0, 5, 4, 2],
                    "pre_hook": [4, 5, 6, 5, 4, 2],
                    "hook": [0, 4, 5, 7, 4, 2, 5, 4],
                    "main_a": [0, 2, 4, 2, 0, 5, 4, 2],
                    "fill": [4, 5, 6, 5, 4, 2],
                    "main_b": [0, 4, 5, 7, 4, 2, 5, 4],
                    "break": [4, 2, 0],
                    "main_c": [0, 4, 7, 5, 4, 2, 7, 5],
                    "outro": [0, 2, 4, 2, 0, 2],
                }
                motif_pool = (
                    list((motif_memory.get("section_map") or {}).get(section_song_role) or [])
                    or list((motif_memory.get("section_map") or {}).get(section_family) or [])
                    or boombap_motif_phrases.get(section_song_role)
                    or boombap_motif_phrases.get(section_family)
                    or motif_pool_map.get(section_song_role)
                    or motif_pool_map.get(section_family)
                    or motif_pool_map.get(section_name, [0, 2, 4, 2, 0, 5, 4, 2])
                )
                if is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}):
                    motif_pool = list(motif_memory.get("answer") or [4, 5, 6, 5, 4, 2, 1])
                elif is_first_bar and (previous_section_name == "break" or previous_song_role == "break") and section_song_role == "hook":
                    motif_pool = list(motif_memory.get("hook") or [0, 4, 7, 5, 4, 7, 9, 5])
                for event_index, step in enumerate(active_steps):
                    motif_offset = ((bar_index % 4) * (1 if section_song_role == "verse" else 2))
                    degree = motif_pool[(motif_offset + event_index) % len(motif_pool)]
                    if section_song_role == "verse" and bar_index % 4 == 1 and event_index in {1, 3}:
                        degree = 0
                    elif section_song_role == "hook" and bar_index % 2 == 1 and event_index in {0, len(active_steps) - 1}:
                        degree = 5
                    midi_note = melody_root + progression_offset + scale_intervals[degree % len(scale_intervals)]
                    if section_song_role == "hook" and event_index % 3 == 0:
                        midi_note += 12
                    if section_song_role == "pre_hook" and event_index >= 2:
                        midi_note += 12
                    if rng.random() < creativity * 0.16:
                        midi_note += 12 if rng.random() < 0.4 else -12
                    if section_song_role == "verse" and event_index == 0 and bar_index % 4 == 3 and rng.random() < 0.55:
                        continue
                    if section_song_role == "break" and event_index == 0 and rng.random() < 0.4:
                        midi_note -= 12
                    if is_first_bar and (previous_section_name == "break" or previous_song_role == "break") and event_index == 0:
                        midi_note += 12
                    if is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}) and event_index >= max(1, len(active_steps) - 2):
                        midi_note += 12
                    if is_last_bar and next_section_name == "outro":
                        midi_note -= 12 if event_index == len(active_steps) - 1 else 0
                    midi_note = _nearest_scale_midi(midi_note, pitch_context["root_pitch_class"], scale_intervals)
                    midi_note = max(melody_min_midi, min(melody_max_midi, midi_note))
                    duration_steps = 3 if step in motif_steps else 2
                    if section_song_role == "hook" and event_index in {1, 4}:
                        duration_steps = 4
                    if section_family == "hook_variation" and event_index in {0, 3, 5}:
                        duration_steps = 4
                    if is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}) and event_index >= max(1, len(active_steps) - 2):
                        duration_steps = max(duration_steps, 4)
                    if is_first_bar and (previous_section_name == "break" or previous_song_role == "break") and event_index in {0, 1}:
                        duration_steps = max(duration_steps, 4)
                    amplitude = (0.2 if section_song_role == "verse" else 0.22 if section_song_role != "pre_hook" else 0.18) * melodic_density
                    if section_song_role == "hook" and event_index in {0, len(active_steps) - 1}:
                        amplitude *= 1.08
                    if section_family == "hook_variation" and event_index in {0, 2, 4}:
                        amplitude *= 1.12
                    if is_last_bar and (next_section_name in {"fill", "break"} or next_song_role in {"pre_hook", "break"}):
                        amplitude *= 0.96
                    elif is_first_bar and (previous_section_name == "break" or previous_song_role == "break"):
                        amplitude *= 1.06
                    events.append(
                        {
                            "start_seconds": round(float(current_bar_start + (step * step_seconds)), 4),
                            "duration_seconds": round(float(duration_steps * step_seconds), 4),
                            "frequency_hz": round(_midi_to_frequency(midi_note), 3),
                            "amplitude": round(float(amplitude), 4),
                            "midi_note": int(midi_note),
                            "section_name": section_name,
                            "song_role": section_song_role,
                            "section_family": section_family,
                            "phrase_function": (
                                "motif_fragment"
                                if section_song_role in {"intro", "break", "outro"} and event_index < max(1, len(active_steps) // 2)
                                else "motif_answer"
                                if section_song_role == "hook" and event_index >= max(1, len(active_steps) // 2)
                                else "motif_anchor"
                            ),
                            "motif_memory": {
                                "anchor": list(motif_memory.get("anchor") or [])[:8],
                                "active_pool": list(motif_pool)[:8],
                            },
                        }
                    )
            current_bar_start += seconds_per_bar

    return events


def _synthesize_phrase_track(
    role: str,
    output_path: str,
    sample_rate: int,
    tempo_bpm: float,
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    pitch_context: Dict[str, Any],
    creativity: float,
    seed: int,
) -> Dict[str, Any]:
    total_bars = sum(max(1, int(section.get("bars") or 1)) for section in sections)
    total_seconds = (240.0 / float(tempo_bpm)) * total_bars
    total_samples = max(64, int(math.ceil(total_seconds * sample_rate)))
    buffer = np.zeros((total_samples, 2), dtype=np.float32)
    note_events = _build_phrase_note_events(
        role=role,
        sections=sections,
        tempo_bpm=tempo_bpm,
        steps_per_bar=steps_per_bar,
        pitch_context=pitch_context,
        creativity=creativity,
        seed=seed,
    )
    for event in note_events:
        _add_synth_note(
            buffer=buffer,
            sample_rate=sample_rate,
            start_seconds=float(event["start_seconds"]),
            duration_seconds=float(event["duration_seconds"]),
            frequency_hz=float(event["frequency_hz"]),
            amplitude=float(event["amplitude"]),
            role=role,
            section_name=_clean_text(event.get("song_role")) or _clean_text(event.get("section_name")) or None,
            phrase_function=_clean_text(event.get("phrase_function")) or None,
            pitch_slide_semitones=float(event.get("pitch_slide_semitones") or 0.0),
        )

    stereo = _normalize_audio_buffer(buffer, peak=0.78 if _clean_text(role).lower() == "bass" else 0.62)
    _write_audio_array_to_wav(stereo, output_path, sample_rate=sample_rate)
    return {
        "role": _clean_text(role).lower(),
        "output_path": output_path,
        "duration_seconds": round(float(total_seconds), 4),
        "peak": round(float(np.max(np.abs(stereo))) if stereo.size else 0.0, 6),
        "note_events_count": len(note_events),
        "note_events_preview": note_events[:8],
        "phrase_functions": sorted(
            {
                _clean_text(event.get("phrase_function")).lower()
                for event in note_events
                if _clean_text(event.get("phrase_function"))
            }
        ),
        "render_mode": "phrase_track",
        "pitch_context": {
            "estimated_key": pitch_context.get("estimated_key"),
            "estimated_scale": pitch_context.get("estimated_scale"),
            "bass_root_midi": pitch_context.get("bass_root_midi"),
            "melody_root_midi": pitch_context.get("melody_root_midi"),
        },
    }


def _build_single_trigger_pattern(total_bars: int, steps_per_bar: int) -> str:
    total_steps = max(1, int(total_bars)) * max(1, int(steps_per_bar))
    return "x" + ("-" * max(0, total_steps - 1))


def _crate_unit_curve(value: float, factor: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, float(value)) * float(factor)))))


def _crate_triangle_fit(value: float, center: float, width: float) -> float:
    if width <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (abs(float(value) - float(center)) / float(width)))))


def _score_reference_crate_window(
    window: "np.ndarray",
    role: str,
    sample_rate: int,
    style_tokens: Optional[Iterable[str]] = None,
    variant: Optional[str] = None,
) -> tuple[float, Dict[str, float]]:
    if np is None:
        return 0.0, {}

    audio = np.asarray(window, dtype=np.float32)
    if audio.size == 0:
        return 0.0, {}
    if audio.ndim == 1:
        mono = audio.astype(np.float32)
    else:
        mono = audio.mean(axis=1).astype(np.float32)
    if mono.size < 32:
        return 0.0, {}
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    mono = mono - float(np.mean(mono))

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    if rms < 0.004 or peak < 0.006:
        return 0.0, {
            "level": round(_crate_unit_curve(rms, 30.0), 4),
            "silence_penalty": 1.0,
        }

    diff = np.diff(mono).astype(np.float32) if mono.size > 1 else np.zeros(1, dtype=np.float32)
    abs_diff = np.abs(diff)
    mean_diff = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    peak_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
    diff_threshold = max(float(np.percentile(abs_diff, 78)) if abs_diff.size > 8 else 0.0, mean_diff * 1.8, 0.0015)
    rhythmic_activity = float(np.mean(abs_diff > diff_threshold)) if abs_diff.size else 0.0
    zcr = float(np.mean(np.signbit(mono[:-1]) != np.signbit(mono[1:]))) if mono.size > 1 else 0.0
    crest = float(peak / max(rms, 1e-6))

    max_fft_samples = 4096
    if mono.size > max_fft_samples:
        stride = max(1, int(math.ceil(mono.size / max_fft_samples)))
        fft_mono = mono[::stride][:max_fft_samples]
        effective_sr = float(sample_rate) / float(stride)
    else:
        fft_mono = mono
        effective_sr = float(sample_rate)
    if fft_mono.size >= 64:
        fft_window = np.hanning(fft_mono.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(fft_mono * fft_window)).astype(np.float64)
        freqs = np.fft.rfftfreq(fft_mono.size, d=1.0 / max(effective_sr, 1.0))
        power = spectrum * spectrum
        total_power = float(np.sum(power) + 1e-12)

        def _band_ratio(low_hz: float, high_hz: float) -> float:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            if not np.any(mask):
                return 0.0
            return float(np.sum(power[mask]) / total_power)

        body_ratio = _band_ratio(120.0, 900.0)
        mid_ratio = _band_ratio(900.0, 3600.0)
        air_ratio = _band_ratio(3600.0, max(3700.0, effective_sr * 0.48))
        low_mud_ratio = _band_ratio(20.0, 120.0)
        centroid = float(np.sum(freqs * spectrum) / max(float(np.sum(spectrum)), 1e-12))
        centroid_norm = float(max(0.0, min(1.0, centroid / max(effective_sr * 0.5, 1.0))))
    else:
        body_ratio = 0.0
        mid_ratio = 0.0
        air_ratio = 0.0
        low_mud_ratio = 0.0
        centroid_norm = 0.0

    tokens = {str(token).lower() for token in (style_tokens or []) if str(token).strip()}
    resolved_role = _clean_text(role).lower()
    resolved_variant = _clean_text(variant).lower()
    boombap_bias = 1.0 if _uses_boombap_profile(tokens) else 0.0
    dark_bias = 1.0 if "dark" in tokens else 0.0

    level = _crate_unit_curve(rms, 24.0)
    attack = _crate_unit_curve(peak_diff, 18.0)
    motion = _crate_unit_curve(mean_diff, 95.0)
    rhythm = float(max(0.0, min(1.0, rhythmic_activity * 10.0)))
    body = float(max(0.0, min(1.0, body_ratio * 4.0)))
    mids = float(max(0.0, min(1.0, mid_ratio * 3.2)))
    air = float(max(0.0, min(1.0, air_ratio * 3.0)))
    warmth = float(max(0.0, min(1.0, (body_ratio * 3.6) + (low_mud_ratio * 0.6))))
    tonal_balance = 1.0 - min(1.0, abs(centroid_norm - (0.18 if dark_bias else 0.24)) * 3.2)
    noise_penalty = max(
        0.0,
        (zcr - 0.24) * 2.2,
        (air_ratio - (0.42 if dark_bias else 0.52)) * 1.4,
        (crest - 18.0) / 18.0,
    )
    mud_penalty = max(0.0, (low_mud_ratio - 0.42) * 1.6)

    if resolved_role == "texture":
        crest_fit = _crate_triangle_fit(crest, 4.8, 5.5)
        score = (
            (level * 0.23)
            + (body * (0.23 + (0.03 * boombap_bias)))
            + (mids * 0.14)
            + (motion * 0.14)
            + (warmth * 0.12)
            + (tonal_balance * 0.08)
            + (crest_fit * 0.06)
            - (noise_penalty * 0.17)
            - (mud_penalty * 0.08)
        )
    elif resolved_role == "perc":
        crest_fit = _crate_triangle_fit(crest, 7.5, 7.0)
        low_penalty = max(0.0, (body_ratio + low_mud_ratio - 0.46) * 1.35)
        score = (
            (attack * 0.25)
            + (rhythm * 0.24)
            + (motion * 0.16)
            + (crest_fit * 0.13)
            + (level * 0.1)
            + (air * 0.08)
            + (mids * 0.04)
            - (noise_penalty * 0.1)
            - (low_penalty * 0.14)
        )
    else:
        phrase_bias = 1.0 if resolved_variant == "phrase" else 0.0
        crest_fit = _crate_triangle_fit(crest, 6.2 if phrase_bias else 8.0, 6.0)
        attack_weight = 0.16 if phrase_bias else 0.25
        body_weight = 0.24 if phrase_bias else 0.18
        rhythm_weight = 0.16 if phrase_bias else 0.18
        score = (
            (attack * attack_weight)
            + (body * (body_weight + (0.03 * boombap_bias)))
            + (rhythm * rhythm_weight)
            + (level * 0.15)
            + (mids * 0.12)
            + (motion * 0.08)
            + (crest_fit * 0.08)
            + (tonal_balance * 0.05)
            - (noise_penalty * 0.13)
            - (mud_penalty * 0.07)
        )

    components = {
        "level": round(level, 4),
        "attack": round(attack, 4),
        "motion": round(motion, 4),
        "rhythm": round(rhythm, 4),
        "body": round(body, 4),
        "mids": round(mids, 4),
        "air": round(air, 4),
        "warmth": round(warmth, 4),
        "tonal_balance": round(float(max(0.0, min(1.0, tonal_balance))), 4),
        "crest_fit": round(crest_fit, 4),
        "noise_penalty": round(float(max(0.0, min(1.0, noise_penalty))), 4),
        "mud_penalty": round(float(max(0.0, min(1.0, mud_penalty))), 4),
        "rms": round(rms, 6),
        "crest": round(crest, 4),
        "zero_crossing_rate": round(zcr, 4),
    }
    return float(max(0.0, min(1.0, score))), components


def _extract_reference_texture_track(
    reference_path: str,
    output_path: str,
    sample_rate: int,
    tempo_bpm: float,
    total_bars: int,
    sections: Optional[List[Dict[str, Any]]] = None,
    style_tags: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    source_path = _resolve_existing_path(reference_path)
    source_audio, _ = _load_reference_audio_array(source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    stereo = _ensure_stereo(source_audio)
    total_seconds = (240.0 / float(tempo_bpm)) * max(1, int(total_bars))
    total_samples = max(64, int(math.ceil(total_seconds * sample_rate)))
    seconds_per_bar = 240.0 / float(tempo_bpm)
    segment_seconds = min(max(seconds_per_bar, 1.4), 3.0)
    segment_length = min(stereo.shape[0], max(2048, int(round(segment_seconds * sample_rate))))

    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    secondary_start = 0
    secondary_segment = None
    chosen_score = 0.0
    chosen_components: Dict[str, float] = {}
    secondary_score = 0.0
    secondary_components: Dict[str, float] = {}
    if stereo.shape[0] <= segment_length:
        segment = stereo.copy()
        chosen_start = 0
        secondary_segment = np.flip(segment, axis=0).copy()
        chosen_score, chosen_components = _score_reference_crate_window(
            segment,
            role="texture",
            sample_rate=sample_rate,
            style_tokens=style_tokens,
        )
        secondary_score, secondary_components = chosen_score, dict(chosen_components)
    else:
        start_min = int(stereo.shape[0] * 0.15)
        start_max = max(start_min + 1, int(stereo.shape[0] * 0.7))
        step = max(512, segment_length // 3)
        ranked_candidates: List[tuple[float, int, Dict[str, float]]] = []
        chosen_start = start_min
        for start in range(start_min, min(start_max, stereo.shape[0] - segment_length), step):
            window = stereo[start : start + segment_length]
            score, components = _score_reference_crate_window(
                window,
                role="texture",
                sample_rate=sample_rate,
                style_tokens=style_tokens,
            )
            if score <= 0.0:
                continue
            ranked_candidates.append((score, start, components))
        if ranked_candidates:
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen_start = ranked_candidates[0][1]
            chosen_score = ranked_candidates[0][0]
            chosen_components = ranked_candidates[0][2]
            secondary_candidate = ranked_candidates[1] if len(ranked_candidates) > 1 else ranked_candidates[0]
            secondary_start = secondary_candidate[1]
            secondary_score = secondary_candidate[0]
            secondary_components = secondary_candidate[2]
        segment = stereo[chosen_start : chosen_start + segment_length].copy()
        secondary_segment = stereo[secondary_start : secondary_start + segment_length].copy()
        if not chosen_components:
            chosen_score, chosen_components = _score_reference_crate_window(
                segment,
                role="texture",
                sample_rate=sample_rate,
                style_tokens=style_tokens,
            )
        if not secondary_components and secondary_segment is not None:
            secondary_score, secondary_components = _score_reference_crate_window(
                secondary_segment,
                role="texture",
                sample_rate=sample_rate,
                style_tokens=style_tokens,
            )

    if signal is not None and segment.shape[0] > 128:
        lowpass_hz = 4200.0 if "dark" in style_tokens else 6200.0
        try:
            hp_b, hp_a = signal.butter(2, 180.0 / (sample_rate * 0.5), btype="highpass")
            lp_b, lp_a = signal.butter(2, lowpass_hz / (sample_rate * 0.5), btype="lowpass")
            for channel in range(segment.shape[1]):
                filtered = signal.filtfilt(hp_b, hp_a, segment[:, channel]).astype(np.float32)
                filtered = signal.filtfilt(lp_b, lp_a, filtered).astype(np.float32)
                segment[:, channel] = filtered
                if secondary_segment is not None and secondary_segment.shape[0] > 0:
                    filtered_alt = signal.filtfilt(hp_b, hp_a, secondary_segment[:, channel]).astype(np.float32)
                    filtered_alt = signal.filtfilt(lp_b, lp_a, filtered_alt).astype(np.float32)
                    secondary_segment[:, channel] = filtered_alt
        except Exception:
            pass

    fade_samples = min(max(64, int(round(sample_rate * 0.03))), max(1, segment.shape[0] // 8))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    segment[:fade_samples] *= fade_in[:, np.newaxis]
    segment[-fade_samples:] *= fade_out[:, np.newaxis]
    if secondary_segment is not None and secondary_segment.shape[0] > 0:
        secondary_segment[:fade_samples] *= fade_in[:, np.newaxis]
        secondary_segment[-fade_samples:] *= fade_out[:, np.newaxis]

    resolved_sections = list(sections or [])
    if not resolved_sections:
        resolved_sections = [{"name": "main_a", "bars": max(1, int(total_bars)), "energy": 0.92}]
    section_buffers: List["np.ndarray"] = []
    seconds_per_bar = 240.0 / float(tempo_bpm)
    for section_index, section in enumerate(resolved_sections):
        section_name = _clean_text(section.get("name")).lower() or f"section_{section_index + 1}"
        bars = max(1, int(section.get("bars") or 1))
        section_seconds = seconds_per_bar * bars
        section_samples = max(64, int(round(section_seconds * sample_rate)))
        if section_name == "break":
            base_segment = np.flip(segment, axis=0).copy()
        elif section_name in {"main_b", "main_c"} and secondary_segment is not None:
            base_segment = secondary_segment.copy()
        else:
            base_segment = segment.copy()
        repeats = int(math.ceil(section_samples / max(1, base_segment.shape[0])))
        section_audio = np.tile(base_segment, (repeats, 1))[:section_samples]
        section_gain = {
            "intro": 0.26,
            "main_a": 0.32,
            "fill": 0.24,
            "main_b": 0.36,
            "break": 0.18,
            "main_c": 0.38,
            "outro": 0.22,
        }.get(section_name, 0.32)
        width_gain = {
            "intro": 0.08,
            "main_a": 0.12,
            "fill": 0.1,
            "main_b": 0.18,
            "break": 0.05,
            "main_c": 0.22,
            "outro": 0.08,
        }.get(section_name, 0.12)
        mid = ((section_audio[:, 0] + section_audio[:, 1]) * 0.5).astype(np.float32)
        side = ((section_audio[:, 0] - section_audio[:, 1]) * 0.5).astype(np.float32)
        side *= (1.0 + width_gain)
        section_audio = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
        section_audio *= float(section_gain)
        if section_index > 0:
            crossfade = min(max(128, int(round(sample_rate * 0.045))), section_audio.shape[0] // 4, section_buffers[-1].shape[0] // 4)
            if crossfade > 0:
                fade_prev = np.linspace(1.0, 0.0, crossfade, dtype=np.float32)[:, np.newaxis]
                fade_next = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)[:, np.newaxis]
                section_buffers[-1][-crossfade:] *= fade_prev
                section_audio[:crossfade] *= fade_next
        section_buffers.append(section_audio)

    tiled = np.concatenate(section_buffers, axis=0)[:total_samples]
    tiled = _normalize_audio_buffer(tiled, peak=0.35)
    _write_audio_array_to_wav(tiled, output_path, sample_rate=sample_rate)
    return {
        "role": "texture",
        "output_path": output_path,
        "duration_seconds": round(float(total_seconds), 4),
        "peak": round(float(np.max(np.abs(tiled))) if tiled.size else 0.0, 6),
        "render_mode": "reference_texture",
        "reference_path": source_path,
        "segment_start_seconds": round(float(chosen_start / float(sample_rate)), 4),
        "secondary_segment_start_seconds": round(float(secondary_start / float(sample_rate)), 4),
        "segment_duration_seconds": round(float(segment.shape[0] / float(sample_rate)), 4),
        "crate_score": round(float(chosen_score), 6),
        "crate_score_components": chosen_components,
        "secondary_crate_score": round(float(secondary_score), 6),
        "secondary_crate_score_components": secondary_components,
        "crate_digging_profile": "boombap_texture_v1",
    }


def _extract_reference_chop_sample(
    reference_path: str,
    output_path: str,
    sample_rate: int,
    tempo_bpm: float,
    selection_rank: int = 0,
    style_tags: Optional[Iterable[Any]] = None,
    chop_variant: Optional[str] = None,
    bank_role: Optional[str] = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    source_path = _resolve_existing_path(reference_path)
    source_audio, _ = _load_reference_audio_array(source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    stereo = _ensure_stereo(source_audio)
    beat_seconds = 60.0 / max(float(tempo_bpm), 1.0)
    resolved_variant = _clean_text(chop_variant).lower() or "stab"
    resolved_bank = _clean_text(bank_role).lower() or "main"
    if resolved_variant == "phrase":
        chop_seconds = min(max(beat_seconds * 0.95, 0.46), 0.92)
    elif resolved_variant == "alt":
        chop_seconds = min(max(beat_seconds * 0.62, 0.24), 0.52)
    else:
        chop_seconds = min(max(beat_seconds * 0.5, 0.18), 0.42)
    if resolved_bank == "hook":
        chop_seconds = min(max(chop_seconds * 1.08, 0.24), 0.84)
    elif resolved_bank == "transition":
        chop_seconds = min(max(chop_seconds * 0.82, 0.16), 0.54)
    chop_samples = min(stereo.shape[0], max(2048, int(round(chop_seconds * sample_rate))))

    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    chosen_score = 0.0
    chosen_components: Dict[str, float] = {}
    if stereo.shape[0] <= chop_samples:
        chosen_start = 0
        chop = stereo.copy()
        chosen_score, chosen_components = _score_reference_crate_window(
            chop,
            role="chop",
            sample_rate=sample_rate,
            style_tokens=style_tokens,
            variant=resolved_variant,
        )
    else:
        chosen_start = int(stereo.shape[0] * 0.18)
        base_candidates = _rank_reference_chop_candidates(
            reference_path=source_path,
            stereo=stereo,
            sample_rate=sample_rate,
            chop_samples=chop_samples,
            style_tokens=style_tokens,
            variant=resolved_variant,
        )
        ranked_candidates: List[tuple[float, int, Dict[str, float]]] = []
        for score, start, components in base_candidates:
            bank_bonus = 0.0
            if resolved_bank == "hook":
                bank_bonus = float(components.get("body") or 0.0) * 0.06
            elif resolved_bank == "answer":
                bank_bonus = float(components.get("mids") or 0.0) * 0.04
            elif resolved_bank == "transition":
                bank_bonus = float(components.get("attack") or 0.0) * 0.05
            ranked_candidates.append((min(1.0, float(score) + bank_bonus), int(start), dict(components)))
        if ranked_candidates:
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            resolved_rank = max(0, min(int(selection_rank or 0), len(ranked_candidates) - 1))
            chosen_start = ranked_candidates[resolved_rank][1]
            chosen_score = ranked_candidates[resolved_rank][0]
            chosen_components = ranked_candidates[resolved_rank][2]
        chop = stereo[chosen_start : chosen_start + chop_samples].copy()
        if not chosen_components:
            chosen_score, chosen_components = _score_reference_crate_window(
                chop,
                role="chop",
                sample_rate=sample_rate,
                style_tokens=style_tokens,
                variant=resolved_variant,
            )

    boombap_profile = _uses_boombap_profile(style_tokens)
    if signal is not None and chop.shape[0] > 128:
        lowpass_hz = 4800.0 if resolved_variant == "phrase" else (5200.0 if "dark" in style_tokens else 7600.0)
        highpass_hz = 140.0 if resolved_variant == "phrase" else 220.0
        if resolved_bank == "hook":
            lowpass_hz = min(8400.0, lowpass_hz + 900.0)
        elif resolved_bank == "transition":
            highpass_hz = min(480.0, highpass_hz + 70.0)
        try:
            hp_b, hp_a = signal.butter(2, highpass_hz / (sample_rate * 0.5), btype="highpass")
            lp_b, lp_a = signal.butter(2, lowpass_hz / (sample_rate * 0.5), btype="lowpass")
            for channel in range(chop.shape[1]):
                filtered = signal.filtfilt(hp_b, hp_a, chop[:, channel]).astype(np.float32)
                filtered = signal.filtfilt(lp_b, lp_a, filtered).astype(np.float32)
                chop[:, channel] = filtered
        except Exception:
            pass

    fade_samples = min(max(64, int(round(sample_rate * (0.035 if resolved_variant == "phrase" else 0.02)))), max(1, chop.shape[0] // 6))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    chop[:fade_samples] *= fade_in[:, np.newaxis]
    chop[-fade_samples:] *= fade_out[:, np.newaxis]
    mono = chop.mean(axis=1).astype(np.float32)
    transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
    transient = np.clip(transient, -0.2, 0.2)
    left = chop[:, 0] + (transient * 0.18)
    right = chop[:, 1] - (transient * 0.14)
    stereo_width = 0.1 if resolved_variant == "phrase" else (0.12 if "dark" in style_tokens else 0.18)
    if resolved_bank == "hook":
        stereo_width += 0.04
    elif resolved_bank == "transition":
        stereo_width -= 0.03
    mid = ((left + right) * 0.5).astype(np.float32)
    side = ((left - right) * 0.5).astype(np.float32) * (1.0 + stereo_width)
    chop = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
    chop = np.tanh(chop * (1.04 if resolved_variant == "phrase" else (1.08 if "dark" in style_tokens else 1.12)))
    output_gain = 0.72 if resolved_variant == "phrase" else 0.86
    peak_target = 0.64 if resolved_variant == "phrase" else 0.68
    if resolved_bank == "hook":
        output_gain *= 1.06
        peak_target = min(0.74, peak_target + 0.03)
    elif resolved_bank == "transition":
        output_gain *= 0.9
    chop = _normalize_audio_buffer(chop * output_gain, peak=peak_target)
    _write_audio_array_to_wav(chop, output_path, sample_rate=sample_rate)
    return {
        "role": "chop",
        "output_path": output_path,
        "duration_seconds": round(float(chop.shape[0] / float(sample_rate)), 4),
        "peak": round(float(np.max(np.abs(chop))) if chop.size else 0.0, 6),
        "render_mode": "reference_chop",
        "chop_variant": resolved_variant,
        "chop_bank_role": resolved_bank,
        "reference_path": source_path,
        "segment_start_seconds": round(float(chosen_start / float(sample_rate)), 4),
        "selection_rank": int(selection_rank),
        "crate_score": round(float(chosen_score), 6),
        "crate_score_components": chosen_components,
        "crate_digging_profile": f"boombap_chop_{resolved_variant}_{resolved_bank}_v1",
    }


def _extract_reference_perc_layer(
    reference_path: str,
    output_path: str,
    sample_rate: int,
    tempo_bpm: float,
    style_tags: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    source_path = _resolve_existing_path(reference_path)
    stereo, _ = _load_reference_audio_array(source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    stereo = _ensure_stereo(stereo).astype(np.float32)
    mono = stereo.mean(axis=1).astype(np.float32)
    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}

    if mono.size == 0:
        raise ValueError("Referencia vazia para extracao da camada percussiva.")

    if signal is not None and mono.size > 2048:
        try:
            hp_cutoff = 850.0 if "dark" in style_tokens else 1050.0
            hp_b, hp_a = signal.butter(2, hp_cutoff / (sample_rate * 0.5), btype="highpass")
            proc_mono = signal.filtfilt(hp_b, hp_a, mono).astype(np.float32)
        except Exception:
            proc_mono = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
    else:
        proc_mono = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)

    beat_seconds = 60.0 / max(float(tempo_bpm), 1.0)
    segment_seconds = min(max(beat_seconds * 0.8, 0.32), 0.74)
    segment_samples = min(proc_mono.shape[0], max(2048, int(round(segment_seconds * sample_rate))))

    chosen_start = 0
    chosen_score = 0.0
    chosen_components: Dict[str, float] = {}
    if proc_mono.shape[0] > segment_samples:
        start_min = int(proc_mono.shape[0] * 0.08)
        start_max = max(start_min + 1, int(proc_mono.shape[0] * 0.82))
        step = max(512, segment_samples // 3)
        ranked_candidates: List[tuple[float, int, Dict[str, float]]] = []
        for start in range(start_min, min(start_max, proc_mono.shape[0] - segment_samples), step):
            window = proc_mono[start : start + segment_samples]
            score, components = _score_reference_crate_window(
                window,
                role="perc",
                sample_rate=sample_rate,
                style_tokens=style_tokens,
            )
            if score <= 0.0:
                continue
            ranked_candidates.append((score, start, components))
        if ranked_candidates:
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen_start = ranked_candidates[0][1]
            chosen_score = ranked_candidates[0][0]
            chosen_components = ranked_candidates[0][2]

    perc = proc_mono[chosen_start : chosen_start + segment_samples].copy()
    if not chosen_components:
        chosen_score, chosen_components = _score_reference_crate_window(
            perc,
            role="perc",
            sample_rate=sample_rate,
            style_tokens=style_tokens,
        )
    fade_samples = min(max(64, int(round(sample_rate * 0.018))), max(1, perc.shape[0] // 8))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    perc[:fade_samples] *= fade_in
    perc[-fade_samples:] *= fade_out

    if signal is not None and perc.size > 512:
        try:
            lp_cutoff = 7800.0 if "dark" in style_tokens else 9800.0
            lp_b, lp_a = signal.butter(2, lp_cutoff / (sample_rate * 0.5), btype="lowpass")
            perc = signal.filtfilt(lp_b, lp_a, perc).astype(np.float32)
        except Exception:
            pass

    delayed = np.zeros_like(perc)
    delay_samples = max(16, int(round(sample_rate * 0.0075)))
    if delay_samples < perc.shape[0]:
        delayed[delay_samples:] = perc[:-delay_samples]
    stereo_perc = np.stack(
        [
            perc + (delayed * 0.12),
            perc - (delayed * 0.09),
        ],
        axis=1,
    ).astype(np.float32)
    stereo_perc = np.tanh(stereo_perc * (1.12 if "dark" in style_tokens else 1.08))
    stereo_perc = _normalize_audio_buffer(stereo_perc * 0.68, peak=0.5)
    _write_audio_array_to_wav(stereo_perc, output_path, sample_rate=sample_rate)
    return {
        "role": "perc",
        "output_path": output_path,
        "duration_seconds": round(float(stereo_perc.shape[0] / float(sample_rate)), 4),
        "peak": round(float(np.max(np.abs(stereo_perc))) if stereo_perc.size else 0.0, 6),
        "render_mode": "reference_perc_layer",
        "reference_path": source_path,
        "segment_start_seconds": round(float(chosen_start / float(sample_rate)), 4),
        "segment_duration_seconds": round(float(segment_samples / float(sample_rate)), 4),
        "crate_score": round(float(chosen_score), 6),
        "crate_score_components": chosen_components,
        "crate_digging_profile": "boombap_perc_v1",
    }


def _extract_reference_drum_bed_track(
    reference_path: str,
    output_path: str,
    sample_rate: int,
    tempo_bpm: float,
    total_bars: int,
    sections: Optional[List[Dict[str, Any]]] = None,
    style_tags: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    source_path = _resolve_existing_path(reference_path)
    source_audio, _ = _load_reference_audio_array(source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    stereo = _ensure_stereo(source_audio).astype(np.float32)
    mono = stereo.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        raise ValueError("Referencia vazia para extracao da cama de bateria.")

    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    seconds_per_bar = 240.0 / float(tempo_bpm)
    total_seconds = seconds_per_bar * max(1, int(total_bars))
    total_samples = max(64, int(math.ceil(total_seconds * sample_rate)))
    segment_seconds = min(max(seconds_per_bar * 2.0, 2.4), 5.2)
    segment_samples = min(stereo.shape[0], max(4096, int(round(segment_seconds * sample_rate))))

    proc = mono.copy()
    if signal is not None and proc.size > 2048:
        try:
            hp_b, hp_a = signal.butter(2, (95.0 if "dark" in style_tokens else 120.0) / (sample_rate * 0.5), btype="highpass")
            lp_b, lp_a = signal.butter(2, (6500.0 if "dark" in style_tokens else 7600.0) / (sample_rate * 0.5), btype="lowpass")
            proc = signal.filtfilt(hp_b, hp_a, proc).astype(np.float32)
            proc = signal.filtfilt(lp_b, lp_a, proc).astype(np.float32)
        except Exception:
            proc = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
    transient = np.concatenate([[0.0], np.diff(proc)]).astype(np.float32)
    search_audio = (proc * 0.58) + (transient * 0.42)

    chosen_start = 0
    chosen_score = 0.0
    chosen_components: Dict[str, float] = {}
    if search_audio.shape[0] > segment_samples:
        start_min = int(search_audio.shape[0] * 0.1)
        start_max = max(start_min + 1, int(search_audio.shape[0] * 0.78))
        step = max(1024, segment_samples // 3)
        ranked_candidates: List[tuple[float, int, Dict[str, float]]] = []
        for start in range(start_min, min(start_max, search_audio.shape[0] - segment_samples), step):
            window = search_audio[start : start + segment_samples]
            score, components = _score_reference_crate_window(
                window,
                role="perc",
                sample_rate=sample_rate,
                style_tokens=style_tokens,
            )
            if score <= 0.0:
                continue
            body_bonus = float(components.get("body") or 0.0) * 0.08
            rhythm_bonus = float(components.get("rhythm") or 0.0) * 0.06
            adjusted_score = min(1.0, float(score) + body_bonus + rhythm_bonus)
            ranked_candidates.append((adjusted_score, start, components))
        if ranked_candidates:
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen_score, chosen_start, chosen_components = ranked_candidates[0]

    bed = search_audio[chosen_start : chosen_start + segment_samples].copy()
    if not chosen_components:
        chosen_score, chosen_components = _score_reference_crate_window(
            bed,
            role="perc",
            sample_rate=sample_rate,
            style_tokens=style_tokens,
        )

    if signal is not None and bed.size > 1024:
        try:
            body_b, body_a = signal.butter(2, [180.0 / (sample_rate * 0.5), 1250.0 / (sample_rate * 0.5)], btype="bandpass")
            body = signal.filtfilt(body_b, body_a, bed).astype(np.float32)
            bed = (bed * 0.72) + (body * 0.36)
        except Exception:
            pass
    bed = np.tanh(bed * (1.2 if _uses_boombap_profile(style_tokens) else 1.08)).astype(np.float32)
    fade_samples = min(max(128, int(round(sample_rate * 0.035))), max(1, bed.shape[0] // 8))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    bed[:fade_samples] *= fade_in
    bed[-fade_samples:] *= fade_out

    resolved_sections = list(sections or [])
    if not resolved_sections:
        resolved_sections = [{"name": "main_a", "bars": max(1, int(total_bars)), "energy": 0.92}]
    section_buffers: List["np.ndarray"] = []
    for section_index, section in enumerate(resolved_sections):
        section_name = _resolve_section_song_role(section)
        bars = max(1, int(section.get("bars") or 1))
        section_samples = max(64, int(round(seconds_per_bar * bars * sample_rate)))
        base_segment = np.flip(bed, axis=0).copy() if section_name == "break" else bed.copy()
        repeats = int(math.ceil(section_samples / max(1, base_segment.shape[0])))
        section_audio = np.tile(base_segment, repeats)[:section_samples]
        section_gain = {
            "intro": 0.38,
            "verse": 0.52,
            "pre_hook": 0.6,
            "hook": 0.66,
            "break": 0.28,
            "outro": 0.34,
        }.get(section_name, 0.5)
        if section_index % 2 == 1:
            section_audio = np.roll(section_audio, max(1, int(round(sample_rate * 0.011))))
        section_audio *= float(section_gain)
        if section_buffers:
            crossfade = min(max(128, int(round(sample_rate * 0.045))), section_audio.shape[0] // 4, section_buffers[-1].shape[0] // 4)
            if crossfade > 0:
                section_buffers[-1][-crossfade:] *= np.linspace(1.0, 0.0, crossfade, dtype=np.float32)
                section_audio[:crossfade] *= np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
        section_buffers.append(section_audio.astype(np.float32))

    tiled_mono = np.concatenate(section_buffers, axis=0)[:total_samples]
    delayed = np.zeros_like(tiled_mono)
    delay_samples = max(16, int(round(sample_rate * 0.0065)))
    if delay_samples < tiled_mono.shape[0]:
        delayed[delay_samples:] = tiled_mono[:-delay_samples]
    stereo_bed = np.stack([tiled_mono + (delayed * 0.08), tiled_mono - (delayed * 0.07)], axis=1).astype(np.float32)
    stereo_bed = _normalize_audio_buffer(stereo_bed * 0.52, peak=0.42)
    _write_audio_array_to_wav(stereo_bed, output_path, sample_rate=sample_rate)
    return {
        "role": "drum_bed",
        "output_path": output_path,
        "duration_seconds": round(float(stereo_bed.shape[0] / float(sample_rate)), 4),
        "peak": round(float(np.max(np.abs(stereo_bed))) if stereo_bed.size else 0.0, 6),
        "render_mode": "reference_drum_bed",
        "reference_path": source_path,
        "segment_start_seconds": round(float(chosen_start / float(sample_rate)), 4),
        "segment_duration_seconds": round(float(segment_samples / float(sample_rate)), 4),
        "crate_score": round(float(chosen_score), 6),
        "crate_score_components": chosen_components,
        "crate_digging_profile": "boombap_drum_bed_v1",
    }


def _build_internal_resample_pattern(
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    seed: int,
) -> str:
    rng = np.random.default_rng((int(seed) ^ 0x2C91A5F3) & 0xFFFFFFFF)
    quarter = max(1, steps_per_bar // 4)
    eighth = max(1, steps_per_bar // 8)
    all_steps: List[bool] = []
    for section in sections:
        song_role = _resolve_section_song_role(section)
        bars = max(1, int(section.get("bars") or 1))
        for bar_index in range(bars):
            steps = [False] * steps_per_bar
            if song_role == "intro":
                candidates = [quarter * 2]
                probability = 0.22 + (creativity * 0.08)
            elif song_role == "verse":
                candidates = [0] if bar_index % 2 == 0 else [quarter * 2]
                probability = 0.3 + (creativity * 0.1)
            elif song_role == "pre_hook":
                candidates = [quarter * 2, max(0, steps_per_bar - eighth - 1)]
                probability = 0.38 + (creativity * 0.14)
            elif song_role == "hook":
                candidates = [0, quarter * 2, max(0, steps_per_bar - quarter)]
                probability = 0.44 + (creativity * 0.16)
            elif song_role == "break":
                candidates = [quarter * 2]
                probability = 0.2 + (creativity * 0.08)
            else:
                candidates = [max(0, steps_per_bar - quarter)]
                probability = 0.24 + (creativity * 0.1)
            for idx in candidates:
                if 0 <= idx < steps_per_bar and rng.random() < probability:
                    steps[idx] = True
            all_steps.extend(steps)
    return _pattern_steps_to_text(all_steps)


def _render_internal_resample_layer(
    source_path: str,
    output_path: str,
    sample_rate: int,
    tempo_bpm: float,
    total_bars: int,
    sections: Optional[List[Dict[str, Any]]] = None,
    style_tags: Optional[Iterable[Any]] = None,
    source_role: Optional[str] = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")
    resolved_source_path = _resolve_existing_path(source_path)
    audio, _ = _load_audio_array(resolved_source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    stereo = _ensure_stereo(audio).astype(np.float32)
    if stereo.size == 0:
        raise ValueError("Camada base vazia para resample interno.")
    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    total_seconds = (240.0 / float(tempo_bpm)) * max(1, int(total_bars))
    total_samples = max(64, int(math.ceil(total_seconds * sample_rate)))
    seconds_per_bar = 240.0 / float(tempo_bpm)
    segment_seconds = min(max(seconds_per_bar * 0.75, 0.42), 1.35)
    segment_length = min(stereo.shape[0], max(2048, int(round(segment_seconds * sample_rate))))
    step = max(512, segment_length // 2)
    chosen_start = 0
    chosen_score = 0.0
    chosen_components: Dict[str, float] = {}
    ranked_candidates: List[tuple[float, int, Dict[str, float]]] = []
    search_end = max(1, stereo.shape[0] - segment_length)
    for start in range(0, search_end, step):
        window = stereo[start : start + segment_length]
        score, components = _score_reference_crate_window(
            window,
            role="chop",
            sample_rate=sample_rate,
            style_tokens=style_tokens,
            variant="phrase" if _clean_text(source_role).lower() in {"melody", "lead"} else "stab",
        )
        if score <= 0.0:
            continue
        ranked_candidates.append((score, start, components))
    if ranked_candidates:
        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen_score, chosen_start, chosen_components = ranked_candidates[min(1, len(ranked_candidates) - 1)]
    segment = stereo[chosen_start : chosen_start + segment_length].copy()
    if segment.shape[0] < 128:
        segment = stereo.copy()
    reversed_segment = np.flip(segment, axis=0).copy()
    stretched = _resize_audio_to_length(reversed_segment, max(128, int(round(segment.shape[0] * 1.18))))
    chopped = stretched[: max(128, int(round(stretched.shape[0] * 0.82)))].copy()
    mono = chopped.mean(axis=1).astype(np.float32)
    transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
    processed = np.stack([mono + (transient * 0.1), mono - (transient * 0.08)], axis=1).astype(np.float32)
    if signal is not None and processed.shape[0] > 128:
        try:
            hp_b, hp_a = signal.butter(2, 180.0 / (sample_rate * 0.5), btype="highpass")
            lp_b, lp_a = signal.butter(2, (5200.0 if "dark" in style_tokens else 7600.0) / (sample_rate * 0.5), btype="lowpass")
            for channel in range(processed.shape[1]):
                processed[:, channel] = signal.filtfilt(hp_b, hp_a, processed[:, channel]).astype(np.float32)
                processed[:, channel] = signal.filtfilt(lp_b, lp_a, processed[:, channel]).astype(np.float32)
        except Exception:
            pass
    processed = np.tanh(processed * 1.12).astype(np.float32)
    fade_samples = min(max(64, int(round(sample_rate * 0.02))), max(1, processed.shape[0] // 6))
    processed[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, np.newaxis]
    processed[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, np.newaxis]

    resolved_sections = list(sections or [])
    if not resolved_sections:
        resolved_sections = [{"name": "main_a", "bars": max(1, int(total_bars)), "energy": 0.92}]
    section_buffers: List["np.ndarray"] = []
    for section in resolved_sections:
        song_role = _resolve_section_song_role(section)
        bars = max(1, int(section.get("bars") or 1))
        section_samples = max(64, int(round(seconds_per_bar * bars * sample_rate)))
        source_clip = processed.copy()
        if song_role in {"break", "outro"}:
            source_clip = np.flip(source_clip, axis=0).copy()
        if song_role == "hook":
            source_clip = _resize_audio_to_length(source_clip, max(128, int(round(source_clip.shape[0] * 0.94))))
        repeats = int(math.ceil(section_samples / max(1, source_clip.shape[0])))
        section_audio = np.tile(source_clip, (repeats, 1))[:section_samples]
        gain = {
            "intro": 0.18,
            "verse": 0.24,
            "pre_hook": 0.28,
            "hook": 0.34,
            "break": 0.16,
            "outro": 0.18,
        }.get(song_role, 0.22)
        section_buffers.append((section_audio * gain).astype(np.float32))
    tiled = np.concatenate(section_buffers, axis=0)[:total_samples]
    tiled = _normalize_audio_buffer(tiled * 0.72, peak=0.44)
    _write_audio_array_to_wav(tiled, output_path, sample_rate=sample_rate)
    return {
        "role": "resample",
        "output_path": output_path,
        "duration_seconds": round(float(tiled.shape[0] / float(sample_rate)), 4),
        "peak": round(float(np.max(np.abs(tiled))) if tiled.size else 0.0, 6),
        "render_mode": "internal_resample_layer",
        "source_path": resolved_source_path,
        "source_role": _clean_text(source_role).lower() or None,
        "segment_start_seconds": round(float(chosen_start / float(sample_rate)), 4),
        "segment_duration_seconds": round(float(segment.shape[0] / float(sample_rate)), 4),
        "crate_score": round(float(chosen_score), 6),
        "crate_score_components": chosen_components,
        "crate_digging_profile": "internal_resample_motif_v1",
    }


def _build_reference_perc_pattern(
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    total_steps = sum(max(1, int(section.get("bars") or 1)) for section in sections) * steps_per_bar
    if total_steps <= 0:
        return ""

    pattern_steps: List[bool] = []
    half_step = max(1, steps_per_bar // 2)
    quarter_step = max(1, steps_per_bar // 4)
    for section_index, section in enumerate(sections):
        section_name = _clean_text(section.get("name")).lower() or f"section_{section_index + 1}"
        song_role = _resolve_section_song_role(section)
        bars = max(1, int(section.get("bars") or 1))
        density = float(section.get("chop_density") or 1.0)
        for bar_index in range(bars):
            bar_steps = [False] * steps_per_bar
            if song_role == "intro":
                candidate_steps = [half_step]
            elif song_role == "verse":
                candidate_steps = [quarter_step, half_step + quarter_step]
            elif song_role == "pre_hook":
                candidate_steps = [quarter_step, half_step, half_step + quarter_step, max(0, steps_per_bar - 2)]
            elif song_role == "hook":
                candidate_steps = [quarter_step, half_step, half_step + quarter_step, max(0, steps_per_bar - quarter_step), max(0, steps_per_bar - 2)]
            elif song_role == "break":
                candidate_steps = [half_step]
            else:
                candidate_steps = [quarter_step]

            for step in candidate_steps:
                if 0 <= step < steps_per_bar and rng.random() < (0.46 + (creativity * 0.22)) * max(0.72, density):
                    bar_steps[step] = True
            if section_name in {"fill", "main_b", "main_c"} or song_role in {"pre_hook", "hook"}:
                pickup_step = max(0, steps_per_bar - quarter_step)
                if rng.random() < 0.32 + (creativity * 0.18):
                    bar_steps[pickup_step] = True
            pattern_steps.extend(bar_steps)

    return _pattern_steps_to_text(pattern_steps)


def _build_reference_chop_pattern(
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    quarter = max(1, steps_per_bar // 4)
    eighth = max(1, steps_per_bar // 8)
    all_steps: List[bool] = []
    for section_index, section in enumerate(sections):
        section_name = _clean_text(section.get("name")) or "main_a"
        song_role = _resolve_section_song_role(section)
        section_key = song_role or section_name.lower()
        chop_density = float(section.get("chop_density") or 1.0)
        bars = max(1, int(section.get("bars") or 1))
        previous_section = sections[section_index - 1] if section_index > 0 else {}
        next_section = sections[section_index + 1] if section_index + 1 < len(sections) else {}
        previous_section_name = _resolve_section_song_role(previous_section) or _clean_text(previous_section.get("name")).lower()
        next_section_name = _resolve_section_song_role(next_section) or _clean_text(next_section.get("name")).lower()
        for bar_index in range(bars):
            is_first_bar = bar_index == 0
            is_last_bar = bar_index >= bars - 1
            steps = [False] * steps_per_bar
            if section_key == "intro":
                for idx in (quarter * 2, steps_per_bar - quarter):
                    if idx < steps_per_bar:
                        steps[idx] = True
            elif section_key == "fill":
                for idx in (quarter * 2, quarter * 2 + max(1, quarter // 2), steps_per_bar - quarter, steps_per_bar - 1):
                    if idx < steps_per_bar:
                        steps[idx] = True
            elif section_key == "outro":
                if quarter * 2 < steps_per_bar:
                    steps[quarter * 2] = True
            elif section_key == "break":
                for idx in (quarter * 2,):
                    if idx < steps_per_bar:
                        steps[idx] = True
            else:
                if section_key == "verse":
                    base_hits = [quarter, quarter * 2 + max(1, quarter // 2), steps_per_bar - quarter]
                elif section_key == "pre_hook":
                    base_hits = [quarter, quarter * 2, quarter * 3, max(0, steps_per_bar - eighth - 1)]
                elif section_key == "hook":
                    base_hits = [0, quarter, quarter * 2 + max(1, quarter // 2), quarter * 3, steps_per_bar - quarter]
                else:
                    base_hits = [quarter, quarter * 2 + max(1, quarter // 2), steps_per_bar - quarter]
                for idx in base_hits:
                    if idx < steps_per_bar:
                        steps[idx] = True
                if section_key in {"main_a", "verse"} and rng.random() < (0.24 + (creativity * 0.18)) * chop_density:
                    idx = max(0, quarter * 2 - eighth)
                    if idx < steps_per_bar:
                        steps[idx] = True
                if section_key in {"main_b", "pre_hook"}:
                    extra = [max(0, quarter - 1), quarter * 3]
                    for idx in extra:
                        if idx < steps_per_bar and rng.random() < (0.55 + (creativity * 0.25)) * chop_density:
                            steps[idx] = True
                    if rng.random() < (0.34 + (creativity * 0.22)) * chop_density:
                        idx = max(0, steps_per_bar - eighth - 1)
                        if idx < steps_per_bar:
                            steps[idx] = True
                if section_key in {"main_c", "hook"}:
                    extra = [max(0, quarter - 1), quarter * 2, quarter * 3, max(0, steps_per_bar - eighth - 1)]
                    for idx in extra:
                        if idx < steps_per_bar and rng.random() < (0.62 + (creativity * 0.22)) * chop_density:
                            steps[idx] = True
                if bar_index % 2 == 1 and rng.random() < 0.22 + (creativity * 0.2):
                    idx = max(0, steps_per_bar - quarter - 1)
                    steps[idx] = True
                if bar_index % 4 == 3 and section_key in {"verse", "pre_hook", "hook", "main_b", "main_c"}:
                    for idx in (max(0, steps_per_bar - eighth), max(0, steps_per_bar - 2)):
                        if idx < steps_per_bar and rng.random() < (0.3 + (creativity * 0.18)) * max(0.75, chop_density):
                            steps[idx] = True
            if is_last_bar and next_section_name in {"fill", "break", "main_b", "main_c", "pre_hook", "hook"}:
                transition_hits = [max(0, steps_per_bar - quarter), max(0, steps_per_bar - eighth - 1)]
                if next_section_name in {"main_b", "main_c", "pre_hook", "hook"}:
                    transition_hits.append(max(0, steps_per_bar - 2))
                for idx in transition_hits:
                    if idx < steps_per_bar and rng.random() < (0.34 + (creativity * 0.18)) * max(0.75, chop_density):
                        steps[idx] = True
            if is_first_bar and previous_section_name in {"fill", "break"}:
                relaunch_hits = [quarter, quarter * 2]
                if section_key in {"main_b", "main_c", "pre_hook", "hook"}:
                    relaunch_hits.append(max(0, quarter - 1))
                for idx in relaunch_hits:
                    if idx < steps_per_bar and rng.random() < (0.42 + (creativity * 0.18)) * max(0.8, chop_density):
                        steps[idx] = True
            all_steps.extend(steps)
    return _pattern_steps_to_text(all_steps)


def _build_reference_counter_chop_pattern(
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    quarter = max(1, steps_per_bar // 4)
    eighth = max(1, steps_per_bar // 8)
    all_steps: List[bool] = []
    for section in sections:
        section_name = _clean_text(section.get("name")).lower() or "main_a"
        song_role = _resolve_section_song_role(section)
        section_key = song_role or section_name
        chop_density = float(section.get("chop_density") or 1.0)
        bars = max(1, int(section.get("bars") or 1))
        for bar_index in range(bars):
            steps = [False] * steps_per_bar
            if section_key in {"intro", "outro"}:
                candidate_steps = [max(0, quarter - 1)]
            elif section_key == "break":
                candidate_steps = [max(0, quarter * 3 - 1)]
            elif section_key == "fill":
                candidate_steps = [max(0, quarter - 1), quarter * 2, max(0, steps_per_bar - eighth - 1)]
            else:
                candidate_steps = [max(0, quarter - 1), quarter * 2, max(0, quarter * 3 - 1)]
                if section_key in {"main_b", "main_c", "pre_hook", "hook"}:
                    candidate_steps.extend([max(0, steps_per_bar - quarter - 1), max(0, steps_per_bar - 2)])
                if section_key == "hook":
                    candidate_steps.append(0)
            for idx in candidate_steps:
                if 0 <= idx < steps_per_bar and rng.random() < (0.28 + (creativity * 0.18)) * max(0.72, chop_density):
                    steps[idx] = True
            if bar_index % 2 == 0 and section_key in {"main_b", "main_c", "pre_hook", "hook"} and rng.random() < 0.24 + (creativity * 0.12):
                idx = max(0, quarter * 2 - eighth)
                if idx < steps_per_bar:
                    steps[idx] = True
            all_steps.extend(steps)
    return _pattern_steps_to_text(all_steps)


def _build_reference_phrase_chop_pattern(
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    quarter = max(1, steps_per_bar // 4)
    eighth = max(1, steps_per_bar // 8)
    all_steps: List[bool] = []
    for section_index, section in enumerate(sections):
        section_name = _clean_text(section.get("name")).lower() or "main_a"
        song_role = _resolve_section_song_role(section)
        chop_density = float(section.get("chop_density") or 1.0)
        bars = max(1, int(section.get("bars") or 1))
        next_song_role = _resolve_section_song_role(sections[section_index + 1] if section_index + 1 < len(sections) else None)
        for bar_index in range(bars):
            steps = [False] * steps_per_bar
            is_first_bar = bar_index == 0
            is_last_bar = bar_index >= bars - 1
            if song_role == "intro":
                candidate_steps = [0] if bar_index == 0 else [quarter * 2]
                probability = 0.26 + (creativity * 0.12)
            elif song_role == "verse":
                candidate_steps = [0] if bar_index % 2 == 0 else [quarter * 2]
                if bar_index % 4 == 3:
                    candidate_steps.append(max(0, steps_per_bar - quarter))
                probability = 0.42 + (creativity * 0.18)
            elif song_role == "pre_hook":
                candidate_steps = [0, max(0, steps_per_bar - quarter), max(0, steps_per_bar - eighth - 1)]
                probability = 0.52 + (creativity * 0.2)
            elif song_role == "hook":
                candidate_steps = [0, quarter * 2]
                if bar_index % 2 == 1:
                    candidate_steps.append(max(0, steps_per_bar - quarter))
                probability = 0.58 + (creativity * 0.22)
            elif song_role == "break":
                candidate_steps = [quarter * 2]
                probability = 0.22 + (creativity * 0.1)
            else:
                candidate_steps = [0]
                probability = 0.34 + (creativity * 0.14)

            if is_first_bar and song_role == "hook":
                candidate_steps.append(max(0, quarter - 1))
            if is_last_bar and next_song_role in {"pre_hook", "hook", "break"}:
                candidate_steps.append(max(0, steps_per_bar - eighth - 1))

            for idx in candidate_steps:
                if 0 <= idx < steps_per_bar and rng.random() < probability * max(0.62, chop_density):
                    steps[idx] = True
            all_steps.extend(steps)
    return _pattern_steps_to_text(all_steps)


def _build_reference_bank_chop_pattern(
    bank_role: str,
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    seed: int,
) -> str:
    normalized_bank = _clean_text(bank_role).lower() or "answer"
    rng = np.random.default_rng((int(seed) ^ (zlib.crc32(normalized_bank.encode("utf-8")) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    quarter = max(1, steps_per_bar // 4)
    eighth = max(1, steps_per_bar // 8)
    all_steps: List[bool] = []
    for section in sections:
        song_role = _resolve_section_song_role(section)
        bars = max(1, int(section.get("bars") or 1))
        chop_density = float(section.get("chop_density") or 1.0)
        for bar_index in range(bars):
            steps = [False] * steps_per_bar
            if normalized_bank == "answer":
                candidates = [quarter * 2, max(0, steps_per_bar - quarter)]
                if song_role in {"hook", "pre_hook"}:
                    candidates.append(max(0, quarter - 1))
                probability = 0.34 + (creativity * 0.16)
            elif normalized_bank == "hook":
                candidates = [0, quarter, quarter * 2, max(0, steps_per_bar - quarter)]
                probability = 0.46 + (creativity * 0.18) if song_role == "hook" else 0.18 + (creativity * 0.08)
            else:
                candidates = [max(0, steps_per_bar - eighth - 1)]
                if song_role in {"pre_hook", "break", "outro"}:
                    candidates.append(max(0, steps_per_bar - 2))
                probability = 0.28 + (creativity * 0.14)
            for idx in candidates:
                if 0 <= idx < steps_per_bar and rng.random() < probability * max(0.66, chop_density):
                    steps[idx] = True
            all_steps.extend(steps)
    return _pattern_steps_to_text(all_steps)


def _render_creative_mix_bus(
    input_audio_path: str,
    output_path: str,
    style_tags: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    status = ensure_runtime_ready("mix")
    ffmpeg_path = status["ffmpeg_path"]
    input_path = _resolve_existing_path(input_audio_path)
    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    dark_profile = any(token in style_tokens for token in ("dark", "boom bap", "old school", "sub-heavy"))
    boombap_profile = _uses_boombap_profile(style_tokens)

    filter_chain = [
        "highpass=f=28",
        f"lowpass=f={'12800' if boombap_profile else ('14500' if dark_profile else '16800')}",
        f"equalizer=f={68 if boombap_profile else 62}:t=q:w=1.0:g={'3.8' if boombap_profile else '3.4'}",
        f"equalizer=f={145 if boombap_profile else 180}:t=q:w=1.0:g={'1.6' if boombap_profile else '1.2'}",
        f"equalizer=f={320 if boombap_profile else 280}:t=q:w=1.1:g={'-1.8' if boombap_profile else '-1.4'}",
        f"equalizer=f={1700 if boombap_profile else (2600 if dark_profile else 3600)}:t=q:w=1.0:g={1.2 if boombap_profile else (0.9 if dark_profile else 1.4)}",
        f"equalizer=f={6900 if boombap_profile else (7800 if dark_profile else 9200)}:t=q:w=0.9:g={0.15 if boombap_profile else (0.45 if dark_profile else 0.72)}",
        f"acompressor=threshold={'0.2' if boombap_profile else '0.19'}:ratio={'2.55' if boombap_profile else '2.35'}:attack=7:release={'148' if boombap_profile else '135'}:makeup={'1.28' if boombap_profile else '1.2'}",
        f"volume={'1.1' if boombap_profile else '1.08'}",
        "alimiter=limit=0.94",
    ]
    filter_graph = ",".join(filter_chain)
    result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            input_path,
            "-af",
            filter_graph,
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    if result.returncode != 0:
        _raise_command_error("ffmpeg creative mix", result)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "filter_graph": filter_graph,
        "stderr": result.stderr,
    }


def _derive_vocal_fit_mix_profile(
    vocal_analysis: Dict[str, Any],
    instrumental_analysis: Optional[Dict[str, Any]] = None,
    phrase_blocks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vocal_payload = vocal_analysis.get("analysis") if "analysis" in vocal_analysis else vocal_analysis
    instrumental_payload = (
        instrumental_analysis.get("analysis") if isinstance(instrumental_analysis, dict) and "analysis" in instrumental_analysis else (instrumental_analysis or {})
    )
    active_ratio = max(0.0, min(1.0, 1.0 - float(vocal_payload.get("silence_ratio") or 0.0)))
    vocal_peak_dbfs = float(vocal_payload.get("peak_dbfs") or -9.0)
    vocal_centroid_hz = float(vocal_payload.get("spectral_centroid_hz") or 3200.0)
    instrumental_correlation = instrumental_payload.get("stereo_correlation")
    try:
        instrumental_correlation = float(instrumental_correlation) if instrumental_correlation is not None else None
    except Exception:
        instrumental_correlation = None
    dense_long_bars = 0
    open_bars = 0
    for block in list((phrase_blocks or {}).get("blocks") or []):
        profile_name = _clean_text(block.get("profile")).lower()
        block_bars = int(block.get("bars") or 0)
        if profile_name == "dense_long":
            dense_long_bars += block_bars
        if profile_name.startswith("open_"):
            open_bars += block_bars

    if active_ratio >= 0.5 or dense_long_bars > open_bars + 4:
        profile_name = "tight_focus"
        space_scale = 0.82
        width_scale = 0.78
        glue_comp_scale = 1.16
        presence_scale = 0.9
        saturation_scale = 0.96
    elif active_ratio <= 0.28 and (instrumental_correlation is None or instrumental_correlation > 0.45):
        profile_name = "open_support"
        space_scale = 1.14
        width_scale = 1.18
        glue_comp_scale = 0.9
        presence_scale = 1.08
        saturation_scale = 1.03
    else:
        profile_name = "balanced_glue"
        space_scale = 1.0
        width_scale = 1.0
        glue_comp_scale = 1.0
        presence_scale = 1.0
        saturation_scale = 1.0

    if vocal_centroid_hz > 3600.0:
        presence_scale *= 0.95
    if vocal_peak_dbfs > -4.5:
        glue_comp_scale *= 1.05

    return {
        "profile": profile_name,
        "active_ratio": round(float(active_ratio), 4),
        "space_scale": round(float(space_scale), 4),
        "width_scale": round(float(width_scale), 4),
        "glue_comp_scale": round(float(glue_comp_scale), 4),
        "presence_scale": round(float(presence_scale), 4),
        "saturation_scale": round(float(saturation_scale), 4),
        "dense_long_bars": int(dense_long_bars),
        "open_bars": int(open_bars),
        "vocal_peak_dbfs": round(float(vocal_peak_dbfs), 2),
        "vocal_centroid_hz": round(float(vocal_centroid_hz), 2),
        "instrumental_stereo_correlation": round(float(instrumental_correlation), 4)
        if instrumental_correlation is not None
        else None,
    }


def _build_vocal_fit_ui_summary(
    fit_plan: Dict[str, Any],
    vocal_mix_profile: Optional[Dict[str, Any]],
    phrase_blocks_manifest: Optional[Dict[str, Any]],
    macrosections_manifest: Optional[Dict[str, Any]],
    hotspot_manifest: Optional[Dict[str, Any]],
    output_duration_seconds: Optional[float],
    output_path: str,
    prepared_vocal_path: str,
    ducked_instrumental_path: str,
    adaptation_profile: str,
    transcription_status: str,
    resolved_pitch_shift_semitones: int = 0,
) -> Dict[str, Any]:
    blocks = list((phrase_blocks_manifest or {}).get("blocks") or [])
    sections = list((macrosections_manifest or {}).get("sections") or [])
    profile_counts: Dict[str, int] = {}
    for block in blocks:
        profile_name = _clean_text(block.get("profile")).lower() or "unclassified"
        profile_counts[profile_name] = int(profile_counts.get(profile_name, 0)) + int(block.get("bars") or 0)
    section_counts: Dict[str, int] = {}
    for section in sections:
        section_name = _clean_text(section.get("section_type")).lower() or "unclassified"
        section_counts[section_name] = int(section_counts.get(section_name, 0)) + int(section.get("bars") or 0)

    vocal_key = dict(fit_plan.get("vocal_key") or {})
    instrumental_key = fit_plan.get("instrumental_key") or {}
    applied_pitch_shift = int(resolved_pitch_shift_semitones or 0)
    effective_vocal_key_name = _clean_text(vocal_key.get("key")) or None
    if applied_pitch_shift:
        effective_vocal_key_name = _transpose_key_name(effective_vocal_key_name, applied_pitch_shift) or effective_vocal_key_name
    effective_vocal_scale = _clean_text(vocal_key.get("scale")) or None
    effective_key_distance = _key_distance_semitones(
        _clean_text(instrumental_key.get("key")) or None,
        effective_vocal_key_name,
    )
    pitch_aligned = False
    if effective_key_distance is not None and effective_key_distance == 0:
        instrumental_scale = _clean_text(instrumental_key.get("scale")) or None
        pitch_aligned = (
            not instrumental_scale
            or not effective_vocal_scale
            or instrumental_scale == effective_vocal_scale
        )
    return {
        "mode": "vocal_fit",
        "instrumental_adaptation_profile": adaptation_profile,
        "mix_profile": _clean_text((vocal_mix_profile or {}).get("profile")) or "balanced_glue",
        "transcription_status": _clean_text(transcription_status) or "disabled",
        "duration_seconds": round(float(output_duration_seconds or 0.0), 4) if output_duration_seconds is not None else None,
        "tempo_bpm": round(float(fit_plan.get("instrumental_tempo_bpm") or 0.0), 2)
        if fit_plan.get("instrumental_tempo_bpm") is not None
        else None,
        "vocal_key": {
            "key": effective_vocal_key_name,
            "scale": effective_vocal_scale,
        },
        "instrumental_key": {
            "key": _clean_text(instrumental_key.get("key")) or None,
            "scale": _clean_text(instrumental_key.get("scale")) or None,
        },
        "time_aligned": bool(fit_plan.get("should_time_align")),
        "pitch_aligned": bool(pitch_aligned),
        "applied_pitch_shift_semitones": applied_pitch_shift,
        "instrumental_extended": any(
            "re-renderizado" in _clean_text(item).lower() or "rerender" in _clean_text(item).lower()
            for item in list(fit_plan.get("recommendations") or [])
        ),
        "phrase_blocks_count": len(blocks),
        "phrase_profile_bars": profile_counts,
        "macrosections_count": len(sections),
        "macrosection_bars": section_counts,
        "hotspots": {
            "sibilance": int((hotspot_manifest or {}).get("sibilance_event_count") or 0),
            "breath": int((hotspot_manifest or {}).get("breath_event_count") or 0),
            "plosive": int((hotspot_manifest or {}).get("plosive_event_count") or 0),
            "harshness": int((hotspot_manifest or {}).get("harshness_event_count") or 0),
        },
        "outputs": {
            "mix": output_path,
            "prepared_vocal": prepared_vocal_path,
            "instrumental_ready": ducked_instrumental_path,
        },
    }


def _extract_transcript_segments(transcript_payload: Any) -> List[Dict[str, Any]]:
    def _normalize_segment_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        try:
            raw_start = item.get("start")
            raw_end = item.get("end")
            start_seconds = float(item.get("start_time") or raw_start or 0.0)
            end_seconds = float(item.get("end_time") or raw_end or start_seconds)
        except Exception:
            return None
        if raw_start is not None and raw_end is not None:
            try:
                start_value = float(raw_start)
                end_value = float(raw_end)
                if start_value > 1000.0 or end_value > 1000.0:
                    start_seconds = start_value / 1000.0
                    end_seconds = end_value / 1000.0
            except Exception:
                pass
        text = _clean_text(item.get("text") or item.get("sentence") or item.get("content"))
        if not text or end_seconds <= start_seconds:
            return None
        return {
            "start_seconds": round(float(start_seconds), 4),
            "end_seconds": round(float(end_seconds), 4),
            "text": text,
        }

    if isinstance(transcript_payload, dict):
        segments = transcript_payload.get("segments")
        if isinstance(segments, list):
            normalized_segments: List[Dict[str, Any]] = []
            for item in segments:
                normalized_item = _normalize_segment_item(item)
                if not normalized_item:
                    continue
                normalized_segments.append(normalized_item)
            return normalized_segments
    if isinstance(transcript_payload, list):
        normalized_segments: List[Dict[str, Any]] = []
        for item in transcript_payload:
            normalized_item = _normalize_segment_item(item)
            if not normalized_item:
                continue
            normalized_segments.append(normalized_item)
        return normalized_segments
    if isinstance(transcript_payload, str):
        normalized_segments: List[Dict[str, Any]] = []
        for raw_line in transcript_payload.splitlines():
            line = _clean_text(raw_line)
            if not line:
                continue
            try:
                parsed_line = json.loads(line)
            except Exception:
                continue
            normalized_item = _normalize_segment_item(parsed_line)
            if not normalized_item:
                continue
            normalized_segments.append(normalized_item)
        return normalized_segments
    return []


def _derive_vocal_transcript_line_events(
    transcript_manifest: Optional[Dict[str, Any]],
    sample_rate: int,
    tempo_bpm: float,
    macrosections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    transcript_payload = (transcript_manifest or {}).get("transcript")
    segments = _extract_transcript_segments(transcript_payload)
    if not segments:
        return {
            "line_events": [],
            "line_events_count": 0,
            "source": "transcript_segments_missing",
        }

    line_events: List[Dict[str, Any]] = []
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    section_gain_map = {
        "intro": 0.0,
        "verse": 0.02,
        "pre_hook": 0.06,
        "hook": 0.1,
        "break": -0.02,
        "transition": 0.04,
        "bridge": 0.04,
        "outro": -0.03,
    }
    def _classify_line_profile(text: str, word_count: int, density: float) -> str:
        stripped = _clean_text(text).strip()
        if not stripped:
            return "unknown"
        if (
            (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("(") and stripped.endswith(")"))
            or (stripped.startswith("*") and stripped.endswith("*"))
        ):
            return "non_lyric_event"
        if word_count <= 2 and density < 2.6:
            return "adlib"
        if density >= 4.8 or word_count >= 10:
            return "dense_lyric"
        if any(marker in stripped for marker in ("!", "?")) or (stripped.isupper() and len(stripped) >= 8):
            return "emphatic_lyric"
        if density <= 1.4 and word_count <= 4:
            return "sparse_lyric"
        return "standard_lyric"

    for segment in segments:
        start_seconds = float(segment.get("start_seconds") or 0.0)
        end_seconds = float(segment.get("end_seconds") or start_seconds)
        text = _clean_text(segment.get("text"))
        if not text or end_seconds <= start_seconds:
            continue
        start_sample = int(round(start_seconds * sample_rate))
        end_sample = int(round(end_seconds * sample_rate))
        duration_seconds = max(0.01, end_seconds - start_seconds)
        words = [token for token in re.findall(r"\w+", text, flags=re.UNICODE) if token]
        word_count = len(words)
        density = float(word_count / duration_seconds)
        line_profile = _classify_line_profile(text, word_count, density)
        emphasis_db = 0.0
        if density >= 4.8:
            emphasis_db += 0.08
        elif density >= 3.2:
            emphasis_db += 0.04
        if len(text) >= 42:
            emphasis_db += 0.03
        if any(marker in text for marker in ("!", "?", "…", "...")):
            emphasis_db += 0.04
        if text.isupper() and len(text) >= 8:
            emphasis_db += 0.03
        if line_profile == "dense_lyric":
            emphasis_db += 0.04
        elif line_profile == "emphatic_lyric":
            emphasis_db += 0.05
        elif line_profile == "sparse_lyric":
            emphasis_db += 0.01
        elif line_profile == "adlib":
            emphasis_db -= 0.02
        elif line_profile == "non_lyric_event":
            emphasis_db -= 0.06
        section_type = _resolve_macrosection_type_for_bar(int(start_sample // samples_per_bar), macrosections)
        emphasis_db += float(section_gain_map.get(section_type, 0.0))
        release_db = -0.04 if density >= 4.0 else -0.02
        if line_profile == "dense_lyric":
            release_db = -0.05
        elif line_profile == "emphatic_lyric":
            release_db = -0.03
        elif line_profile == "adlib":
            release_db = -0.01
        elif line_profile == "non_lyric_event":
            release_db = 0.0
        line_events.append(
            {
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_seconds": round(float(start_seconds), 4),
                "end_seconds": round(float(end_seconds), 4),
                "start_bar": int(max(0, start_sample // samples_per_bar)),
                "end_bar": int(max(0, (max(start_sample, end_sample - 1)) // samples_per_bar)),
                "text": text[:180],
                "word_count": int(word_count),
                "density": round(float(density), 4),
                "line_profile": line_profile,
                "section_type": section_type or None,
                "emphasis_db": round(float(emphasis_db), 4),
                "release_db": round(float(release_db), 4),
            }
        )

    return {
        "source": "transcript_segments",
        "line_events": line_events[:240],
        "line_events_count": len(line_events),
    }


def _run_vocal_fit_transcription(
    vocal_path: str,
    label: str,
    project_id: Optional[str],
    run_id: str,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    for use_gpu in (True, False):
        try:
            manifest = transcribe_audio(
                TranscriptionRequest(
                    audio_path=vocal_path,
                    label=f"{label}-vocal-fit-transcript",
                    language="auto",
                    output_format="json",
                    use_gpu=use_gpu,
                    project_id=project_id,
                    run_id=run_id,
                    persist_to_project=False,
                )
            )
            manifest["transcription_status"] = "ok"
            manifest["transcription_attempts"] = attempts + [{"use_gpu": bool(use_gpu), "status": "ok"}]
            return manifest
        except Exception as exc:
            attempts.append(
                {
                    "use_gpu": bool(use_gpu),
                    "status": "error",
                    "error": _clean_text(str(exc))[:1200],
                }
            )
    return {
        "transcription_status": "failed",
        "transcription_attempts": attempts,
        "transcript": None,
        "transcript_path": None,
    }


def _resolve_macrosection_type_for_bar(
    bar_index: Optional[int],
    macrosections_manifest: Optional[Dict[str, Any]],
) -> str:
    if bar_index is None or not macrosections_manifest:
        return ""
    try:
        normalized_bar_index = int(bar_index)
    except Exception:
        return ""
    sections = list((macrosections_manifest or {}).get("sections") or [])
    bar_to_section = {
        int(key): int(value)
        for key, value in dict((macrosections_manifest or {}).get("bar_section_map") or {}).items()
    }
    section_index = bar_to_section.get(normalized_bar_index)
    if section_index is None:
        return ""
    if section_index < 0 or section_index >= len(sections):
        return ""
    return _clean_text(sections[section_index].get("section_type")).lower()


def _render_vocal_fit_glue_bus(
    input_audio_path: str,
    vocal_path: str,
    output_path: str,
    vocal_analysis: Optional[Dict[str, Any]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mix_profile: Optional[Dict[str, Any]] = None,
    macrosections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_path = _resolve_existing_path(input_audio_path)
    vocal_input_path = _resolve_existing_path(vocal_path)

    if signal is None:
        status = ensure_runtime_ready("mix")
        ffmpeg_path = status["ffmpeg_path"]
        filter_graph = ",".join(
            [
                "highpass=f=28",
                "equalizer=f=260:t=q:w=1.0:g=-0.8",
                "equalizer=f=3300:t=q:w=1.0:g=0.35",
                "acompressor=threshold=0.17:ratio=1.8:attack=12:release=180:makeup=0.6",
                "alimiter=limit=0.93",
            ]
        )
        result = _run_command(
            [
                ffmpeg_path,
                "-hide_banner",
                "-y",
                "-i",
                input_path,
                "-af",
                filter_graph,
                "-c:a",
                "pcm_s16le",
                output_path,
            ]
        )
        if result.returncode != 0:
            _raise_command_error("ffmpeg vocal fit glue bus", result)
        return {
            "input_path": input_path,
            "vocal_path": vocal_input_path,
            "output_path": output_path,
            "mode": "ffmpeg_fallback_glue_bus",
            "filter_graph": filter_graph,
        }

    mix_samples, resolved_sample_rate = _load_audio_array(
        input_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    vocal_samples, _ = _load_audio_array(
        vocal_input_path,
        sample_rate=resolved_sample_rate or sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(mix_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "input_path": input_path,
            "vocal_path": vocal_input_path,
            "output_path": output_path,
            "mode": "contextual_glue_bus",
            "active_ratio": 0.0,
            "compression_events_count": 0,
        }

    aligned_vocal = _pad_or_trim_audio(vocal_samples, total_samples)
    activity = _compute_vocal_activity_envelope(aligned_vocal, sample_rate=effective_sample_rate)
    vocal_envelope = activity["envelope"].astype(np.float32)
    active_ratio = float(activity.get("active_ratio") or 0.0)

    mono = stereo.mean(axis=1).astype(np.float32)
    rms_window = max(64, int(round(0.035 * effective_sample_rate)))
    kernel = np.ones(rms_window, dtype=np.float32) / float(rms_window)
    rms = np.sqrt(np.convolve(np.square(mono), kernel, mode="same") + 1e-8).astype(np.float32)
    glue_comp_scale = float((mix_profile or {}).get("glue_comp_scale") or 1.0)
    presence_scale = float((mix_profile or {}).get("presence_scale") or 1.0)
    saturation_scale = float((mix_profile or {}).get("saturation_scale") or 1.0)
    threshold = 0.18 - (active_ratio * 0.018 * glue_comp_scale)
    compression_strength = np.clip((rms - threshold) / max(0.04, 1.0 - threshold), 0.0, 1.0).astype(np.float32)
    ratio_amount = (0.12 + (active_ratio * 0.08)) * glue_comp_scale
    vocal_bias = 0.04 * vocal_envelope
    gain_curve = 1.0 - ((compression_strength * ratio_amount) + vocal_bias)
    macrosection_events: List[Dict[str, Any]] = []
    if macrosections:
        sections = list((macrosections or {}).get("sections") or [])
        bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}
        section_curve = np.ones(total_samples, dtype=np.float32)
        samples_per_bar = max(1, int(round((240.0 / max(1.0, float((vocal_analysis.get("analysis") if isinstance(vocal_analysis, dict) and "analysis" in vocal_analysis else (vocal_analysis or {})).get("tempo_estimate_bpm_librosa") or (vocal_analysis.get("analysis") if isinstance(vocal_analysis, dict) and "analysis" in vocal_analysis else (vocal_analysis or {})).get("tempo_estimate_bpm") or 92.0))) * effective_sample_rate)))
        section_scale_map = {
            "intro": 0.96,
            "verse": 1.0,
            "pre_hook": 1.04,
            "hook": 1.08,
            "break": 0.9,
            "transition": 0.97,
            "bridge": 0.99,
            "outro": 0.95,
        }
        for bar_index, section_index in bar_to_section.items():
            if section_index < 0 or section_index >= len(sections):
                continue
            section_type = _clean_text(sections[section_index].get("section_type")).lower()
            scale = float(section_scale_map.get(section_type, 1.0))
            start_index = max(0, bar_index * samples_per_bar)
            end_index = min(total_samples, start_index + samples_per_bar)
            if end_index <= start_index:
                continue
            section_curve[start_index:end_index] *= scale
            macrosection_events.append(
                {
                    "bar_index": int(bar_index),
                    "section_type": section_type or None,
                    "compression_scale": round(float(scale), 4),
                }
            )
        tilt_window = max(64, int(round(0.05 * effective_sample_rate)))
        if section_curve.size > tilt_window:
            kernel = np.hanning(tilt_window).astype(np.float32)
            kernel_sum = float(np.sum(kernel))
            if kernel_sum > 0:
                kernel /= kernel_sum
                section_curve = np.convolve(section_curve, kernel, mode="same").astype(np.float32)
        compression_strength *= np.clip(section_curve, 0.88, 1.12)
    gain_curve = np.clip(gain_curve, 0.78, 1.0).astype(np.float32)

    tilt_window = max(64, int(round(0.05 * effective_sample_rate)))
    tilt_kernel = np.hanning(tilt_window).astype(np.float32)
    tilt_sum = float(np.sum(tilt_kernel))
    if tilt_sum > 0:
        tilt_kernel /= tilt_sum
        gain_curve = np.convolve(gain_curve, tilt_kernel, mode="same").astype(np.float32)
    gain_curve = np.clip(gain_curve, 0.8, 1.0)

    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    low_mid_low_hz = 180.0
    low_mid_high_hz = min(420.0, nyquist_hz * 0.52)
    presence_low_hz = 2500.0
    presence_high_hz = min(4200.0, nyquist_hz * 0.82)
    if presence_high_hz <= presence_low_hz:
        presence_low_hz = 2200.0
        presence_high_hz = min(3800.0, nyquist_hz * 0.74)
    low_mid_sos = signal.butter(2, [low_mid_low_hz, low_mid_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    presence_sos = signal.butter(2, [presence_low_hz, presence_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")

    analysis = vocal_analysis.get("analysis") if isinstance(vocal_analysis, dict) and "analysis" in vocal_analysis else (vocal_analysis or {})
    silence_ratio = float((analysis or {}).get("silence_ratio") or 0.0)
    low_mid_reduction = np.clip(0.08 + (active_ratio * 0.06), 0.06, 0.16)
    presence_lift = np.clip((0.025 + ((1.0 - silence_ratio) * 0.03)) * presence_scale, 0.018, 0.07)
    saturation_drive = (1.08 + (active_ratio * 0.08)) * saturation_scale

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        low_mid_band = signal.sosfiltfilt(low_mid_sos, channel).astype(np.float32)
        presence_band = signal.sosfiltfilt(presence_sos, channel).astype(np.float32)
        voiced_low_mid_reduction = low_mid_reduction * (0.4 + (0.6 * vocal_envelope))
        voiced_presence_lift = presence_lift * (0.35 + (0.65 * vocal_envelope))
        glued = channel - (low_mid_band * voiced_low_mid_reduction) + (presence_band * voiced_presence_lift)
        glued *= gain_curve
        glued = np.tanh(glued * saturation_drive) / np.tanh(saturation_drive)
        processed_channels.append(glued.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    compression_event_count = int(np.count_nonzero(compression_strength > 0.08))
    return {
        "input_path": input_path,
        "vocal_path": vocal_input_path,
        "output_path": output_path,
        "mode": "contextual_glue_bus",
        "active_ratio": round(float(active_ratio), 4),
        "mix_profile": _clean_text((mix_profile or {}).get("profile")) or "balanced_glue",
        "threshold": round(float(threshold), 4),
        "low_mid_band_hz": [round(float(low_mid_low_hz), 2), round(float(low_mid_high_hz), 2)],
        "presence_band_hz": [round(float(presence_low_hz), 2), round(float(presence_high_hz), 2)],
        "low_mid_reduction": round(float(low_mid_reduction), 4),
        "presence_lift": round(float(presence_lift), 4),
        "saturation_drive": round(float(saturation_drive), 4),
        "compression_events_count": int(compression_event_count),
        "macrosection_events": macrosection_events[:240],
        "macrosection_events_count": len(macrosection_events),
    }


def _apply_contextual_vocal_space(
    vocal_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    mix_profile: Optional[Dict[str, Any]] = None,
    macrosections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_vocal_space_fallback",
            "contextual_events": [],
            "contextual_events_count": 0,
        }

    blocks = list(phrase_blocks.get("blocks") or [])
    stages = dict(phrase_blocks.get("bar_stages") or {})
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    send_envelope = np.zeros(total_samples, dtype=np.float32)
    width_envelope = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    stage_profile_adjustments = {
        "dense_long": {"entry": (0.02, 0.12), "sustain": (0.03, 0.16), "release": (0.07, 0.26), "recovery": (0.1, 0.34)},
        "dense_short": {"entry": (0.025, 0.14), "sustain": (0.04, 0.18), "release": (0.08, 0.28), "recovery": (0.11, 0.36)},
        "open_long": {"entry": (0.04, 0.18), "sustain": (0.055, 0.24), "release": (0.11, 0.38), "recovery": (0.15, 0.48)},
        "open_short": {"entry": (0.05, 0.22), "sustain": (0.065, 0.28), "release": (0.13, 0.42), "recovery": (0.17, 0.54)},
    }

    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        if block_index is None:
            block_index = bar_to_block.get(bar_index - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        send_envelope[start_index:end_index] += float(stage_adjustment[0])
        width_envelope[start_index:end_index] += float(stage_adjustment[1])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "space_send": round(float(stage_adjustment[0]), 4),
                "space_width": round(float(stage_adjustment[1]), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.04 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            send_envelope = np.convolve(send_envelope, kernel, mode="same").astype(np.float32)
            width_envelope = np.convolve(width_envelope, kernel, mode="same").astype(np.float32)

    space_scale = float((mix_profile or {}).get("space_scale") or 1.0)
    width_scale = float((mix_profile or {}).get("width_scale") or 1.0)
    macrosection_events: List[Dict[str, Any]] = []
    if macrosections:
        sections = list((macrosections or {}).get("sections") or [])
        bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}
        section_send_curve = np.ones(total_samples, dtype=np.float32)
        section_width_curve = np.ones(total_samples, dtype=np.float32)
        section_scale_map = {
            "intro": (0.78, 0.82),
            "verse": (0.96, 0.94),
            "pre_hook": (1.02, 1.0),
            "hook": (0.88, 0.92),
            "break": (1.14, 1.08),
            "transition": (1.08, 1.04),
            "bridge": (1.0, 1.02),
            "outro": (1.12, 1.1),
        }
        for bar_index, section_index in bar_to_section.items():
            if section_index < 0 or section_index >= len(sections):
                continue
            section_type = _clean_text(sections[section_index].get("section_type")).lower()
            send_scale, width_section_scale = section_scale_map.get(section_type, (1.0, 1.0))
            start_index = max(0, bar_index * samples_per_bar)
            end_index = min(total_samples, start_index + samples_per_bar)
            if end_index <= start_index:
                continue
            section_send_curve[start_index:end_index] *= float(send_scale)
            section_width_curve[start_index:end_index] *= float(width_section_scale)
            macrosection_events.append(
                {
                    "bar_index": int(bar_index),
                    "section_type": section_type or None,
                    "send_scale": round(float(send_scale), 4),
                    "width_scale": round(float(width_section_scale), 4),
                }
            )
        if total_samples > smoothing_samples:
            kernel = np.hanning(smoothing_samples).astype(np.float32)
            kernel_sum = float(np.sum(kernel))
            if kernel_sum > 0:
                kernel /= kernel_sum
                section_send_curve = np.convolve(section_send_curve, kernel, mode="same").astype(np.float32)
                section_width_curve = np.convolve(section_width_curve, kernel, mode="same").astype(np.float32)
        send_envelope *= np.clip(section_send_curve, 0.72, 1.18)
        width_envelope *= np.clip(section_width_curve, 0.78, 1.16)
    send_envelope = np.clip(send_envelope * space_scale, 0.0, 0.2)
    width_envelope = np.clip(width_envelope * width_scale, 0.0, 0.6)
    mono = stereo.mean(axis=1).astype(np.float32)
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    highpass_hz = 180.0
    lowpass_hz = min(6400.0, nyquist_hz * 0.88)
    hp_sos = signal.butter(2, highpass_hz, btype="highpass", fs=effective_sample_rate, output="sos")
    lp_sos = signal.butter(2, lowpass_hz, btype="lowpass", fs=effective_sample_rate, output="sos")
    wet_mono = signal.sosfiltfilt(hp_sos, mono).astype(np.float32)
    wet_mono = signal.sosfiltfilt(lp_sos, wet_mono).astype(np.float32)

    def _delay(signal_mono: "np.ndarray", delay_ms: float, feedback: float) -> "np.ndarray":
        delay_samples = max(1, int(round((delay_ms / 1000.0) * effective_sample_rate)))
        delayed = np.zeros_like(signal_mono)
        if delay_samples < signal_mono.shape[0]:
            delayed[delay_samples:] += signal_mono[:-delay_samples]
        if feedback > 0 and (delay_samples * 2) < signal_mono.shape[0]:
            delayed[delay_samples * 2 :] += signal_mono[: -(delay_samples * 2)] * float(feedback)
        return delayed.astype(np.float32)

    left_return = (_delay(wet_mono, 46.0, 0.28) + (_delay(wet_mono, 92.0, 0.0) * 0.42)).astype(np.float32)
    right_return = (_delay(wet_mono, 63.0, 0.24) + (_delay(wet_mono, 118.0, 0.0) * 0.38)).astype(np.float32)
    wet = np.stack([left_return, right_return], axis=1)
    wet *= send_envelope[:, np.newaxis]
    width = width_envelope[:, np.newaxis]
    wet[:, 0] *= (1.0 + width[:, 0] * 0.28)
    wet[:, 1] *= (1.0 + width[:, 0] * 0.34)
    processed = np.clip(stereo + wet, -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_vocal_space",
        "mix_profile": _clean_text((mix_profile or {}).get("profile")) or "balanced_glue",
        "highpass_hz": round(float(highpass_hz), 2),
        "lowpass_hz": round(float(lowpass_hz), 2),
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "macrosection_events": macrosection_events[:240],
        "macrosection_events_count": len(macrosection_events),
        "smoothing_samples": int(smoothing_samples),
        "max_space_send": round(float(np.max(send_envelope)) if send_envelope.size else 0.0, 4),
        "max_space_width": round(float(np.max(width_envelope)) if width_envelope.size else 0.0, 4),
    }


def _apply_transition_space(
    audio_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    macrosections: Optional[Dict[str, Any]] = None,
    target_mode: str = "vocal",
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_samples, resolved_sample_rate = _load_audio_array(
        audio_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(input_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": f"{target_mode}_transition_space_fallback",
            "transition_events": [],
            "transition_events_count": 0,
        }

    normalized_mode = "instrumental" if _clean_text(target_mode).lower() == "instrumental" else "vocal"
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    send_envelope = np.zeros(total_samples, dtype=np.float32)
    width_envelope = np.zeros(total_samples, dtype=np.float32)
    transition_events: List[Dict[str, Any]] = []

    if normalized_mode == "instrumental":
        section_event_map = {
            "verse": (0.02, 0.12),
            "pre_hook": (0.03, 0.14),
            "hook": (0.055, 0.24),
            "break": (0.07, 0.16),
            "transition": (0.05, 0.2),
            "bridge": (0.04, 0.16),
            "outro": (0.05, 0.18),
        }
        release_event = (0.03, 0.12)
        hp_hz = 240.0
        lp_hz = 7600.0
        left_delay_ms = (38.0, 76.0)
        right_delay_ms = (52.0, 98.0)
        width_scale_factor = 0.46
    else:
        section_event_map = {
            "verse": (0.028, 0.1),
            "pre_hook": (0.04, 0.14),
            "hook": (0.05, 0.16),
            "break": (0.08, 0.12),
            "transition": (0.055, 0.16),
            "bridge": (0.045, 0.14),
            "outro": (0.072, 0.18),
        }
        release_event = (0.05, 0.18)
        hp_hz = 180.0
        lp_hz = 6400.0
        left_delay_ms = (46.0, 92.0)
        right_delay_ms = (63.0, 118.0)
        width_scale_factor = 0.32

    def _apply_window(
        center_sample: int,
        pre_samples: int,
        post_samples: int,
        send_amount: float,
        width_amount: float,
        event_type: str,
        section_type: Optional[str],
        stage_name: Optional[str] = None,
    ) -> None:
        start = max(0, int(center_sample - pre_samples))
        end = min(total_samples, int(center_sample + post_samples))
        if end <= start:
            return
        window = np.hanning(max(4, end - start)).astype(np.float32)
        if window.size != (end - start):
            window = np.ones(end - start, dtype=np.float32)
        send_envelope[start:end] += window * float(send_amount)
        width_envelope[start:end] += window * float(width_amount)
        transition_events.append(
            {
                "type": event_type,
                "section_type": section_type or None,
                "stage": stage_name or None,
                "center_sample": int(center_sample),
                "line_profile": None,
                "send_amount": round(float(send_amount), 4),
                "width_amount": round(float(width_amount), 4),
            }
        )

    sections = list((macrosections or {}).get("sections") or [])
    for section_index, section in enumerate(sections):
        if section_index == 0:
            continue
        section_type = _clean_text(section.get("section_type")).lower()
        event_amounts = section_event_map.get(section_type)
        if not event_amounts:
            continue
        start_bar = int(section.get("start_bar") or 0)
        center_sample = int(start_bar * samples_per_bar)
        _apply_window(
            center_sample=center_sample,
            pre_samples=max(64, int(round(samples_per_bar * 0.18))),
            post_samples=max(64, int(round(samples_per_bar * 0.34))),
            send_amount=float(event_amounts[0]),
            width_amount=float(event_amounts[1]),
            event_type="macrosection_transition",
            section_type=section_type,
        )

    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}
    for bar_index, stage_name in stages.items():
        if _clean_text(stage_name).lower() != "release":
            continue
        center_sample = int((bar_index + 1) * samples_per_bar)
        section_type = _resolve_macrosection_type_for_bar(bar_index, macrosections)
        _apply_window(
            center_sample=center_sample,
            pre_samples=max(64, int(round(samples_per_bar * 0.08))),
            post_samples=max(64, int(round(samples_per_bar * 0.22))),
            send_amount=float(release_event[0]),
            width_amount=float(release_event[1]),
            event_type="phrase_release_transition",
            section_type=section_type,
            stage_name="release",
        )

    for event in list((transcript_line_manifest or {}).get("line_events") or []):
        end_sample = max(0, min(total_samples, int(event.get("end_sample") or 0)))
        if end_sample <= 0:
            continue
        section_type = _clean_text(event.get("section_type")).lower() or None
        density = float(event.get("density") or 0.0)
        word_count = int(event.get("word_count") or 0)
        line_profile = _clean_text(event.get("line_profile")).lower()
        text_preview = _clean_text(event.get("text"))
        punctuation_boost = 0.0
        if text_preview.endswith(("!", "?")):
            punctuation_boost = 0.01 if normalized_mode == "instrumental" else 0.014
        elif text_preview.endswith((".", ",")):
            punctuation_boost = 0.004 if normalized_mode == "instrumental" else 0.008
        if normalized_mode == "instrumental":
            send_amount = 0.018 + min(0.02, max(0.0, density - 2.0) * 0.004) + punctuation_boost
            width_amount = 0.08 + min(0.1, max(0.0, density - 1.6) * 0.02) + (punctuation_boost * 4.0)
        else:
            send_amount = 0.028 + min(0.024, max(0.0, density - 2.0) * 0.005) + punctuation_boost
            width_amount = 0.1 + min(0.08, max(0.0, density - 1.6) * 0.016) + (punctuation_boost * 3.2)
        if word_count >= 10:
            send_amount += 0.004 if normalized_mode == "instrumental" else 0.006
            width_amount += 0.012
        if line_profile == "dense_lyric":
            send_amount += 0.005
            width_amount += 0.014
        elif line_profile == "emphatic_lyric":
            send_amount += 0.007
            width_amount += 0.01
        elif line_profile == "sparse_lyric":
            send_amount -= 0.003
        elif line_profile == "adlib":
            send_amount -= 0.006
            width_amount -= 0.01
        elif line_profile == "non_lyric_event":
            send_amount -= 0.01
            width_amount -= 0.014
        section_profile_adjustments = {
            ("dense_lyric", "verse"): (0.004, -0.012),
            ("dense_lyric", "hook"): (0.007, 0.016),
            ("dense_lyric", "break"): (0.005, -0.006),
            ("emphatic_lyric", "hook"): (0.01, 0.018),
            ("emphatic_lyric", "transition"): (0.008, 0.014),
            ("sparse_lyric", "hook"): (-0.002, 0.008),
            ("adlib", "hook"): (-0.006, -0.006),
            ("non_lyric_event", "verse"): (-0.008, -0.012),
            ("non_lyric_event", "break"): (-0.01, -0.016),
        }
        profile_section_delta = section_profile_adjustments.get((line_profile, section_type or ""))
        if profile_section_delta:
            send_amount += float(profile_section_delta[0])
            width_amount += float(profile_section_delta[1])
        _apply_window(
            center_sample=end_sample,
            pre_samples=max(64, int(round(samples_per_bar * 0.04))),
            post_samples=max(64, int(round(samples_per_bar * 0.18))),
            send_amount=float(send_amount),
            width_amount=float(width_amount),
            event_type="transcript_line_transition",
            section_type=section_type,
            stage_name="line_release",
        )
        if transition_events:
            transition_events[-1]["line_profile"] = line_profile or None

    smoothing_samples = max(64, int(round(0.03 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            send_envelope = np.convolve(send_envelope, kernel, mode="same").astype(np.float32)
            width_envelope = np.convolve(width_envelope, kernel, mode="same").astype(np.float32)

    if normalized_mode == "instrumental":
        send_envelope = np.clip(send_envelope, 0.0, 0.11)
        width_envelope = np.clip(width_envelope, 0.0, 0.42)
    else:
        send_envelope = np.clip(send_envelope, 0.0, 0.14)
        width_envelope = np.clip(width_envelope, 0.0, 0.34)

    mono = stereo.mean(axis=1).astype(np.float32)
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    hp_sos = signal.butter(2, hp_hz, btype="highpass", fs=effective_sample_rate, output="sos")
    lp_sos = signal.butter(2, min(lp_hz, nyquist_hz * 0.9), btype="lowpass", fs=effective_sample_rate, output="sos")
    wet_mono = signal.sosfiltfilt(hp_sos, mono).astype(np.float32)
    wet_mono = signal.sosfiltfilt(lp_sos, wet_mono).astype(np.float32)

    def _delay(signal_mono: "np.ndarray", delay_ms: float, feedback: float) -> "np.ndarray":
        delay_samples = max(1, int(round((delay_ms / 1000.0) * effective_sample_rate)))
        delayed = np.zeros_like(signal_mono)
        if delay_samples < signal_mono.shape[0]:
            delayed[delay_samples:] += signal_mono[:-delay_samples]
        if feedback > 0 and (delay_samples * 2) < signal_mono.shape[0]:
            delayed[delay_samples * 2 :] += signal_mono[: -(delay_samples * 2)] * float(feedback)
        return delayed.astype(np.float32)

    left_return = (
        _delay(wet_mono, left_delay_ms[0], 0.24) +
        (_delay(wet_mono, left_delay_ms[1], 0.0) * 0.34)
    ).astype(np.float32)
    right_return = (
        _delay(wet_mono, right_delay_ms[0], 0.2) +
        (_delay(wet_mono, right_delay_ms[1], 0.0) * 0.3)
    ).astype(np.float32)
    wet = np.stack([left_return, right_return], axis=1)
    wet *= send_envelope[:, np.newaxis]

    mid = ((stereo[:, 0] + stereo[:, 1]) * 0.5).astype(np.float32)
    side = ((stereo[:, 0] - stereo[:, 1]) * 0.5).astype(np.float32)
    side *= (1.0 + (width_envelope * width_scale_factor))
    widened = np.stack([mid + side, mid - side], axis=1)
    processed = np.clip(widened + wet, -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": f"{normalized_mode}_transition_space",
        "transition_events": transition_events[:240],
        "transition_events_count": len(transition_events),
        "transcript_line_events_count": len(list((transcript_line_manifest or {}).get("line_events") or [])),
        "max_send": round(float(np.max(send_envelope)) if send_envelope.size else 0.0, 4),
        "max_width": round(float(np.max(width_envelope)) if width_envelope.size else 0.0, 4),
        "highpass_hz": round(float(hp_hz), 2),
        "lowpass_hz": round(float(min(lp_hz, nyquist_hz * 0.9)), 2),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_vocal_phrase_contrast(
    vocal_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    hotspot_manifest: Optional[Dict[str, Any]] = None,
    macrosections: Optional[Dict[str, Any]] = None,
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "vocal_phrase_contrast_fallback",
            "contrast_events": [],
            "contrast_events_count": 0,
        }

    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    envelope = np.ones(total_samples, dtype=np.float32)
    contrast_events: List[Dict[str, Any]] = []
    blocks = list((phrase_blocks or {}).get("blocks") or [])
    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}
    bar_to_block = {int(key): int(value) for key, value in dict((phrase_blocks or {}).get("bar_to_block") or {}).items()}

    stage_gain_db = {
        "entry": 0.28,
        "sustain": 0.08,
        "release": -0.14,
        "recovery": -0.06,
    }
    profile_stage_adjustments = {
        "dense_long": {"entry": 0.14, "sustain": 0.08, "release": -0.06, "recovery": -0.04},
        "dense_short": {"entry": 0.1, "sustain": 0.05, "release": -0.04, "recovery": -0.03},
        "open_long": {"entry": 0.06, "sustain": 0.02, "release": -0.08, "recovery": -0.05},
        "open_short": {"entry": 0.04, "sustain": 0.0, "release": -0.1, "recovery": -0.06},
    }
    section_gain_adjustments = {
        "intro": -0.04,
        "verse": 0.0,
        "pre_hook": 0.05,
        "hook": 0.11,
        "break": -0.08,
        "transition": 0.03,
        "bridge": 0.02,
        "outro": -0.05,
    }
    line_profile_section_emphasis = {
        ("dense_lyric", "verse"): 0.03,
        ("dense_lyric", "hook"): 0.012,
        ("dense_lyric", "break"): 0.018,
        ("emphatic_lyric", "hook"): 0.03,
        ("emphatic_lyric", "transition"): 0.022,
        ("sparse_lyric", "hook"): -0.004,
        ("adlib", "hook"): -0.012,
        ("non_lyric_event", "verse"): -0.02,
        ("non_lyric_event", "break"): -0.01,
    }
    line_profile_section_release = {
        ("dense_lyric", "verse"): -0.014,
        ("dense_lyric", "hook"): 0.006,
        ("emphatic_lyric", "hook"): 0.008,
        ("emphatic_lyric", "transition"): 0.004,
        ("adlib", "hook"): 0.004,
        ("non_lyric_event", "verse"): 0.01,
        ("non_lyric_event", "break"): 0.008,
    }

    def _apply_gain_window(start_index: int, end_index: int, gain_db: float) -> None:
        start = max(0, int(start_index))
        end = min(total_samples, int(end_index))
        if end <= start:
            return
        gain_linear = float(10 ** (float(gain_db) / 20.0))
        envelope[start:end] *= gain_linear

    for bar_index, stage_name in stages.items():
        block_index = bar_to_block.get(int(bar_index))
        if block_index is None:
            block_index = bar_to_block.get(int(bar_index) - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        gain_db = float(stage_gain_db.get(_clean_text(stage_name).lower(), 0.0))
        gain_db += float((profile_stage_adjustments.get(block_profile) or {}).get(_clean_text(stage_name).lower(), 0.0))
        section_type = _resolve_macrosection_type_for_bar(int(bar_index), macrosections)
        gain_db += float(section_gain_adjustments.get(section_type, 0.0))
        if gain_db == 0.0:
            continue
        start = int(bar_index * samples_per_bar)
        end = min(total_samples, start + samples_per_bar)
        _apply_gain_window(start, end, gain_db)
        contrast_events.append(
            {
                "type": "bar_stage_contrast",
                "bar_index": int(bar_index),
                "stage": _clean_text(stage_name).lower() or None,
                "block_profile": block_profile or None,
                "section_type": section_type or None,
                "gain_db": round(float(gain_db), 4),
            }
        )

    def _apply_hotspot_boost(events: List[Dict[str, Any]], gain_db: float, event_type: str) -> None:
        for event in events:
            start_index = max(0, int(event.get("start_sample") or 0))
            end_index = min(total_samples, int(event.get("end_sample") or 0))
            if end_index <= start_index:
                continue
            pre_roll = max(32, int(round(0.01 * effective_sample_rate)))
            release = max(32, int(round(0.04 * effective_sample_rate)))
            start = max(0, start_index - pre_roll)
            end = min(total_samples, end_index + release)
            ramp = np.hanning(max(4, end - start)).astype(np.float32)
            ramp_peak = float(10 ** (gain_db / 20.0)) - 1.0
            envelope[start:end] *= (1.0 + (ramp * ramp_peak))
            contrast_events.append(
                {
                    "type": event_type,
                    "start_sample": int(start_index),
                    "end_sample": int(end_index),
                    "section_type": _resolve_macrosection_type_for_bar(int(start_index // samples_per_bar), macrosections) or None,
                    "gain_db": round(float(gain_db), 4),
                }
            )

    for event in list((transcript_line_manifest or {}).get("line_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        line_profile = _clean_text(event.get("line_profile")).lower()
        section_type = _clean_text(event.get("section_type")).lower()
        emphasis_db = float(event.get("emphasis_db") or 0.0)
        release_db = float(event.get("release_db") or 0.0)
        if line_profile == "dense_lyric":
            emphasis_db += 0.03
            release_db -= 0.01
        elif line_profile == "emphatic_lyric":
            emphasis_db += 0.04
        elif line_profile == "adlib":
            emphasis_db -= 0.03
        elif line_profile == "non_lyric_event":
            emphasis_db -= 0.08
            release_db = max(0.0, release_db)
        emphasis_db += float(line_profile_section_emphasis.get((line_profile, section_type), 0.0))
        release_db += float(line_profile_section_release.get((line_profile, section_type), 0.0))
        if emphasis_db != 0.0:
            _apply_gain_window(start_index, end_index, emphasis_db)
        if release_db != 0.0:
            release_end = min(total_samples, end_index + max(32, int(round(0.06 * effective_sample_rate))))
            _apply_gain_window(end_index, release_end, release_db)
        contrast_events.append(
            {
                "type": "transcript_line_contrast",
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "section_type": section_type or None,
                "line_profile": line_profile or None,
                "word_count": event.get("word_count"),
                "density": event.get("density"),
                "gain_db": round(float(emphasis_db), 4),
                "release_db": round(float(release_db), 4),
                "text_preview": _clean_text(event.get("text"))[:80] or None,
            }
        )

    _apply_hotspot_boost(list((hotspot_manifest or {}).get("plosive_events") or []), 0.14, "plosive_phrase_support")
    _apply_hotspot_boost(list((hotspot_manifest or {}).get("harshness_events") or []), 0.08, "harshness_presence_support")

    smoothing_samples = max(64, int(round(0.03 * effective_sample_rate)))
    if envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.9, 1.08).astype(np.float32)

    processed = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "vocal_phrase_contrast",
        "contrast_events": contrast_events[:240],
        "contrast_events_count": len(contrast_events),
        "transcript_line_events_count": len(list((transcript_line_manifest or {}).get("line_events") or [])),
        "smoothing_samples": int(smoothing_samples),
    }


def _resolve_analysis_tempo(analysis_payload: Dict[str, Any]) -> Optional[float]:
    if not analysis_payload:
        return None
    analysis = analysis_payload.get("analysis") if "analysis" in analysis_payload else analysis_payload
    candidates = []
    for key in ("tempo_estimate_bpm_librosa", "tempo_estimate_bpm"):
        value = analysis.get(key)
        try:
            if value is not None:
                candidates.append(float(value))
        except Exception:
            continue
    if not candidates:
        return None
    return round(float(candidates[0]), 2)


def _resolve_analysis_key(analysis_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not analysis_payload:
        return {"key": None, "scale": None, "confidence": None}
    analysis = analysis_payload.get("analysis") if "analysis" in analysis_payload else analysis_payload
    return {
        "key": _clean_text(analysis.get("estimated_key")) or None,
        "scale": _clean_text(analysis.get("estimated_scale")) or None,
        "confidence": analysis.get("estimated_key_confidence"),
    }


def _key_distance_semitones(key_a: Optional[str], key_b: Optional[str]) -> Optional[int]:
    pitch_a = _key_name_to_pitch_class(key_a or "")
    pitch_b = _key_name_to_pitch_class(key_b or "")
    if pitch_a is None or pitch_b is None:
        return None
    raw_distance = abs(pitch_a - pitch_b)
    return min(raw_distance, 12 - raw_distance)


def _signed_pitch_shift_to_target(source_key: Optional[str], target_key: Optional[str]) -> Optional[int]:
    source_pitch = _key_name_to_pitch_class(source_key or "")
    target_pitch = _key_name_to_pitch_class(target_key or "")
    if source_pitch is None or target_pitch is None:
        return None
    difference = int(target_pitch - source_pitch)
    while difference > 6:
        difference -= 12
    while difference < -6:
        difference += 12
    return difference


def _build_vocal_fit_plan(
    instrumental_analysis: Dict[str, Any],
    vocal_analysis: Dict[str, Any],
    request: VocalFitRequest,
) -> Dict[str, Any]:
    instrumental_tempo = _resolve_analysis_tempo(instrumental_analysis)
    vocal_tempo = _resolve_analysis_tempo(vocal_analysis)
    instrumental_key = _resolve_analysis_key(instrumental_analysis)
    vocal_key = _resolve_analysis_key(vocal_analysis)

    tempo_ratio = None
    tempo_adjustment_ratio = None
    should_time_align = False
    if instrumental_tempo and vocal_tempo and vocal_tempo > 0:
        tempo_ratio = float(instrumental_tempo / vocal_tempo)
        tempo_adjustment_ratio = abs(tempo_ratio - 1.0)
        should_time_align = (
            bool(request.auto_time_align)
            and tempo_adjustment_ratio <= max(0.0, float(request.max_tempo_adjustment_ratio))
            and 0.5 <= tempo_ratio <= 2.0
        )

    key_distance = _key_distance_semitones(instrumental_key["key"], vocal_key["key"])
    recommended_pitch_shift = _signed_pitch_shift_to_target(vocal_key["key"], instrumental_key["key"])
    should_pitch_align = False
    recommendations: List[str] = []
    if should_time_align and tempo_ratio is not None:
        recommendations.append(f"Ajustar o vocal em tempo com fator {tempo_ratio:.4f} para encaixar no instrumental.")
    elif instrumental_tempo and vocal_tempo and tempo_adjustment_ratio is not None:
        recommendations.append(
            "Nao aplicar time-stretch automatico porque a diferenca de BPM esta fora da faixa segura configurada."
        )

    if key_distance is not None:
        if key_distance == 0 and instrumental_key["scale"] == vocal_key["scale"]:
            recommendations.append("Tonalidades parecem compativeis para encaixe direto.")
        elif key_distance <= 2:
            recommendations.append("Tonalidades proximas; ajuste melodico ou semitom leve pode funcionar.")
        else:
            recommendations.append("Tonalidades distantes; considerar transposicao do vocal ou reharmonizacao do beat.")
    else:
        recommendations.append("Tonalidade do vocal ou do instrumental nao foi estimada com confianca suficiente.")

    try:
        instrumental_conf = float(instrumental_key.get("confidence") or 0.0)
        vocal_conf = float(vocal_key.get("confidence") or 0.0)
    except Exception:
        instrumental_conf = 0.0
        vocal_conf = 0.0
    min_confidence = min(instrumental_conf, vocal_conf)
    if (
        bool(request.auto_pitch_align)
        and recommended_pitch_shift is not None
        and abs(int(recommended_pitch_shift)) > 0
        and abs(int(recommended_pitch_shift)) <= int(max(0, request.max_pitch_shift_semitones))
        and min_confidence >= 0.075
    ):
        should_pitch_align = True
        recommendations.append(
            f"Aplicar pitch shift de {int(recommended_pitch_shift):+d} semitom(ns) no vocal para aproximar a tonalidade do instrumental."
        )

    recommendations.append("Aplicar ducking no instrumental durante a presenca vocal para abrir espaco na faixa media.")
    recommendations.append("Priorizar cortes leves entre 2 kHz e 5 kHz no instrumental se a voz competir nessa faixa.")

    instrumental_rms = (instrumental_analysis.get("analysis") or instrumental_analysis).get("rms_dbfs")
    vocal_rms = (vocal_analysis.get("analysis") or vocal_analysis).get("rms_dbfs")
    auto_vocal_gain_db = 0.0
    auto_instrumental_gain_db = 0.0
    try:
        if instrumental_rms is not None and vocal_rms is not None:
            delta = float(instrumental_rms) - float(vocal_rms)
            auto_vocal_gain_db = max(-3.0, min(2.0, (delta * 0.35) - 0.4))
            auto_instrumental_gain_db = max(-1.0, min(1.5, (-delta * 0.1) + 0.25))
    except Exception:
        auto_vocal_gain_db = 0.0
        auto_instrumental_gain_db = 0.0

    return {
        "instrumental_tempo_bpm": instrumental_tempo,
        "vocal_tempo_bpm": vocal_tempo,
        "instrumental_duration_seconds": (instrumental_analysis.get("analysis") or instrumental_analysis).get("duration_seconds"),
        "vocal_duration_seconds": (vocal_analysis.get("analysis") or vocal_analysis).get("duration_seconds"),
        "tempo_ratio": round(float(tempo_ratio), 6) if tempo_ratio is not None else None,
        "tempo_adjustment_ratio": round(float(tempo_adjustment_ratio), 6) if tempo_adjustment_ratio is not None else None,
        "should_time_align": bool(should_time_align),
        "recommended_pitch_shift_semitones": int(recommended_pitch_shift) if recommended_pitch_shift is not None else None,
        "should_pitch_align": bool(should_pitch_align),
        "instrumental_key": instrumental_key,
        "vocal_key": vocal_key,
        "key_distance_semitones": key_distance,
        "auto_vocal_gain_db": round(float(auto_vocal_gain_db), 4),
        "auto_instrumental_gain_db": round(float(auto_instrumental_gain_db), 4),
        "recommendations": recommendations,
    }


def _apply_instrumental_tonal_override_to_fit_plan(
    fit_plan: Dict[str, Any],
    pitch_context: Dict[str, Any],
    request: VocalFitRequest,
) -> Dict[str, Any]:
    overridden_key = _clean_text(pitch_context.get("estimated_key"))
    if not overridden_key:
        return fit_plan

    overridden_scale = _clean_text(pitch_context.get("estimated_scale")) or _clean_text(
        (fit_plan.get("instrumental_key") or {}).get("scale")
    )
    overridden_confidence = pitch_context.get("estimated_key_confidence")
    updated_plan = dict(fit_plan)
    updated_plan["instrumental_key"] = {
        "key": overridden_key,
        "scale": overridden_scale or None,
        "confidence": round(float(overridden_confidence), 4) if overridden_confidence is not None else None,
    }

    vocal_key = dict(updated_plan.get("vocal_key") or {})
    key_distance = _key_distance_semitones(overridden_key, vocal_key.get("key"))
    recommended_pitch_shift = _signed_pitch_shift_to_target(vocal_key.get("key"), overridden_key)
    updated_plan["key_distance_semitones"] = key_distance
    updated_plan["recommended_pitch_shift_semitones"] = (
        int(recommended_pitch_shift) if recommended_pitch_shift is not None else None
    )

    try:
        instrumental_conf = float(overridden_confidence or 0.0)
        vocal_conf = float(vocal_key.get("confidence") or 0.0)
    except Exception:
        instrumental_conf = 0.0
        vocal_conf = 0.0

    should_pitch_align = (
        bool(request.auto_pitch_align)
        and recommended_pitch_shift is not None
        and abs(int(recommended_pitch_shift)) > 0
        and abs(int(recommended_pitch_shift)) <= int(max(0, request.max_pitch_shift_semitones))
        and min(instrumental_conf, vocal_conf) >= 0.075
    )
    if key_distance is not None and key_distance <= 1 and overridden_scale == _clean_text(vocal_key.get("scale")):
        should_pitch_align = False
    updated_plan["should_pitch_align"] = bool(should_pitch_align)

    recommendations = [
        text
        for text in list(updated_plan.get("recommendations") or [])
        if "tonalidade do vocal como ancora" not in text.lower()
        and "pitch shift" not in text.lower()
        and "tonalidades parecem compativeis" not in text.lower()
        and "tonalidades proximas" not in text.lower()
        and "tonalidades distantes" not in text.lower()
    ]
    recommendations.insert(0, "Beat adaptado re-renderizado usando a tonalidade do vocal como ancora.")
    if key_distance is not None:
        if key_distance == 0 and overridden_scale == _clean_text(vocal_key.get("scale")):
            recommendations.insert(1, "Tonalidades ficaram compativeis apos o rerender do beat.")
        elif key_distance <= 2:
            recommendations.insert(1, "Tonalidades ficaram proximas apos o rerender; ajuste leve opcional pode finalizar o encaixe.")
        else:
            recommendations.insert(1, "Mesmo com o rerender tonal, ainda existe distancia harmonica perceptivel entre vocal e beat.")
    if should_pitch_align and recommended_pitch_shift is not None:
        recommendations.append(
            f"Aplicar pitch shift de {int(recommended_pitch_shift):+d} semitom(ns) no vocal para finalizar a aproximacao tonal."
        )
    updated_plan["recommendations"] = recommendations
    return updated_plan


def _resolve_audio_duration_seconds(path: str) -> float:
    resolved_path = _resolve_existing_path(path)
    probe = _ffprobe_audio_metadata(resolved_path)
    duration = probe.get("format", {}).get("duration")
    try:
        if duration is not None:
            return round(float(duration), 4)
    except Exception:
        pass
    with wave.open(resolved_path, "rb") as handle:
        frame_rate = handle.getframerate() or DEFAULT_SAMPLE_RATE
        return round(float(handle.getnframes()) / float(frame_rate), 4)


def _pad_or_trim_audio(samples: "np.ndarray", target_samples: int) -> "np.ndarray":
    stereo = _ensure_stereo(samples)
    if stereo.shape[0] == target_samples:
        return stereo
    if stereo.shape[0] > target_samples:
        return stereo[:target_samples].copy()
    pad = np.zeros((target_samples - stereo.shape[0], stereo.shape[1]), dtype=np.float32)
    return np.vstack([stereo, pad])


def _compute_vocal_activity_envelope(
    vocal_samples: "np.ndarray",
    sample_rate: int,
    frame_ms: float = 40.0,
    release_ms: float = 180.0,
) -> Dict[str, Any]:
    mono = _ensure_stereo(vocal_samples).mean(axis=1)
    abs_mono = np.abs(mono)
    frame_size = max(32, int(round(sample_rate * (frame_ms / 1000.0))))
    if frame_size > 1:
        kernel = np.ones(frame_size, dtype=np.float32) / float(frame_size)
        smoothed = np.convolve(abs_mono, kernel, mode="same")
    else:
        smoothed = abs_mono

    release_size = max(32, int(round(sample_rate * (release_ms / 1000.0))))
    if release_size > 1:
        release_kernel = np.exp(-np.linspace(0.0, 5.0, release_size, dtype=np.float32))
        release_kernel /= float(np.sum(release_kernel) or 1.0)
        smoothed = np.convolve(smoothed, release_kernel, mode="same")

    peak = float(np.max(smoothed)) if smoothed.size else 0.0
    normalized = smoothed / peak if peak > 0 else smoothed
    active_threshold = max(0.08, float(np.percentile(normalized, 68))) if normalized.size else 0.08
    active_mask = normalized >= active_threshold
    return {
        "envelope": normalized.astype(np.float32),
        "active_threshold": round(float(active_threshold), 6),
        "active_ratio": round(float(np.mean(active_mask)) if active_mask.size else 0.0, 4),
    }


def _detect_vocal_issue_hotspots(
    vocal_path: str,
    sample_rate: int,
    tempo_bpm: float,
    phrase_blocks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    mono = stereo.mean(axis=1).astype(np.float32)
    activity = _compute_vocal_activity_envelope(stereo, sample_rate=effective_sample_rate)
    total_samples = mono.shape[0]
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    blocks = list((phrase_blocks or {}).get("blocks") or [])
    bar_to_block = {int(key): int(value) for key, value in dict((phrase_blocks or {}).get("bar_to_block") or {}).items()}
    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}

    if total_samples <= 0 or signal is None:
        return {
            "sibilance_events": [],
            "sibilance_event_count": 0,
            "breath_events": [],
            "breath_event_count": 0,
            "plosive_events": [],
            "plosive_event_count": 0,
            "harshness_events": [],
            "harshness_event_count": 0,
            "active_threshold": activity.get("active_threshold"),
        }

    def _band_envelope(low_hz: float, high_hz: float) -> np.ndarray:
        nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
        band_low = max(100.0, min(float(low_hz), nyquist_hz * 0.9))
        band_high = max(band_low + 200.0, min(float(high_hz), nyquist_hz * 0.98))
        sos = signal.butter(2, [band_low, band_high], btype="bandpass", fs=effective_sample_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, mono).astype(np.float32)
        frame_size = max(32, int(round(effective_sample_rate * 0.02)))
        kernel = np.ones(frame_size, dtype=np.float32) / float(frame_size)
        env = np.convolve(np.abs(filtered), kernel, mode="same")
        peak = float(np.max(env)) if env.size else 0.0
        return (env / peak).astype(np.float32) if peak > 0 else env.astype(np.float32)

    sibilance_env = _band_envelope(4800.0, 9200.0)
    breath_env = _band_envelope(6500.0, 12000.0)
    plosive_env = _band_envelope(85.0, 260.0)
    harshness_env = _band_envelope(2200.0, 4800.0)
    base_env = activity["envelope"]
    active_threshold = float(activity.get("active_threshold") or 0.08)
    sibilance_ratio = sibilance_env / np.maximum(base_env, 0.055)
    breath_ratio = breath_env / np.maximum(base_env, 0.04)
    plosive_ratio = plosive_env / np.maximum(base_env, 0.06)
    harshness_ratio = harshness_env / np.maximum(base_env, 0.06)
    sibilance_threshold = max(1.08, float(np.percentile(sibilance_ratio, 78))) if sibilance_ratio.size else 1.08
    breath_threshold = max(1.12, float(np.percentile(breath_ratio, 82))) if breath_ratio.size else 1.12
    plosive_threshold = max(1.1, float(np.percentile(plosive_ratio, 79))) if plosive_ratio.size else 1.1
    harshness_threshold = max(1.06, float(np.percentile(harshness_ratio, 78))) if harshness_ratio.size else 1.06
    min_event_samples = max(32, int(round(effective_sample_rate * 0.045)))
    max_event_samples = max(min_event_samples, int(round(effective_sample_rate * 0.42)))
    min_plosive_samples = max(32, int(round(effective_sample_rate * 0.03)))
    max_plosive_samples = max(min_plosive_samples, int(round(effective_sample_rate * 0.22)))
    min_harshness_samples = max(32, int(round(effective_sample_rate * 0.04)))
    max_harshness_samples = max(min_harshness_samples, int(round(effective_sample_rate * 0.35)))

    def _collect_events(
        mask: np.ndarray,
        ratio: np.ndarray,
        event_type: str,
        min_length_samples: int = min_event_samples,
        max_length_samples: int = max_event_samples,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        start_index: Optional[int] = None
        for sample_index, active in enumerate(list(mask.astype(bool)) + [False]):
            if active and start_index is None:
                start_index = sample_index
            elif not active and start_index is not None:
                end_index = sample_index
                length = end_index - start_index
                if min_length_samples <= length <= max_length_samples:
                    bar_index = int(start_index // samples_per_bar)
                    block_index = bar_to_block.get(bar_index)
                    if block_index is None:
                        block_index = bar_to_block.get(bar_index - 1)
                    block_profile = None
                    if block_index is not None and 0 <= block_index < len(blocks):
                        block_profile = _clean_text(blocks[block_index].get("profile")) or None
                    stage_name = stages.get(bar_index)
                    mean_ratio = float(np.mean(ratio[start_index:end_index])) if end_index > start_index else 0.0
                    peak_ratio = float(np.max(ratio[start_index:end_index])) if end_index > start_index else 0.0
                    events.append(
                        {
                            "type": event_type,
                            "start_sample": int(start_index),
                            "end_sample": int(end_index),
                            "start_seconds": round(float(start_index / float(effective_sample_rate)), 4),
                            "end_seconds": round(float(end_index / float(effective_sample_rate)), 4),
                            "duration_ms": round(float((end_index - start_index) * 1000.0 / float(effective_sample_rate)), 2),
                            "bar_index": int(bar_index),
                            "stage": stage_name or None,
                            "block_profile": block_profile,
                            "mean_ratio": round(mean_ratio, 4),
                            "peak_ratio": round(peak_ratio, 4),
                        }
                    )
                start_index = None
        return events

    sibilance_mask = (
        (sibilance_ratio >= sibilance_threshold)
        & (sibilance_env >= np.percentile(sibilance_env, 70) if sibilance_env.size else 0.0)
        & (base_env >= active_threshold * 0.72)
    )
    breath_mask = (
        (breath_ratio >= breath_threshold)
        & (breath_env >= np.percentile(breath_env, 72) if breath_env.size else 0.0)
        & (base_env >= active_threshold * 0.22)
        & (base_env <= active_threshold * 1.05)
    )
    plosive_mask = (
        (plosive_ratio >= plosive_threshold)
        & (plosive_env >= np.percentile(plosive_env, 70) if plosive_env.size else 0.0)
        & (base_env >= active_threshold * 0.5)
    )
    harshness_mask = (
        (harshness_ratio >= harshness_threshold)
        & (harshness_env >= np.percentile(harshness_env, 70) if harshness_env.size else 0.0)
        & (base_env >= active_threshold * 0.62)
    )

    sibilance_events = _collect_events(sibilance_mask, sibilance_ratio, "sibilance")
    breath_events = _collect_events(breath_mask, breath_ratio, "breath")
    plosive_events = _collect_events(
        plosive_mask,
        plosive_ratio,
        "plosive",
        min_length_samples=min_plosive_samples,
        max_length_samples=max_plosive_samples,
    )
    harshness_events = _collect_events(
        harshness_mask,
        harshness_ratio,
        "harshness",
        min_length_samples=min_harshness_samples,
        max_length_samples=max_harshness_samples,
    )
    return {
        "sibilance_events": sibilance_events[:240],
        "sibilance_event_count": len(sibilance_events),
        "breath_events": breath_events[:240],
        "breath_event_count": len(breath_events),
        "plosive_events": plosive_events[:240],
        "plosive_event_count": len(plosive_events),
        "harshness_events": harshness_events[:240],
        "harshness_event_count": len(harshness_events),
        "active_threshold": round(float(active_threshold), 6),
        "sibilance_ratio_threshold": round(float(sibilance_threshold), 4),
        "breath_ratio_threshold": round(float(breath_threshold), 4),
        "plosive_ratio_threshold": round(float(plosive_threshold), 4),
        "harshness_ratio_threshold": round(float(harshness_threshold), 4),
    }


def _apply_vocal_activity_ducking(
    instrumental_path: str,
    vocal_path: str,
    output_path: str,
    sample_rate: int,
    ducking_strength: float,
    target_duration_seconds: Optional[float] = None,
    instrumental_gain_db: float = 0.0,
    macrosections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, _ = _load_audio_array(instrumental_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    vocal_samples, _ = _load_audio_array(vocal_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)

    target_samples = max(
        instrumental_samples.shape[0],
        vocal_samples.shape[0],
        int(round(float(target_duration_seconds or 0.0) * sample_rate)) if target_duration_seconds else 0,
    )
    instrumental = _pad_or_trim_audio(instrumental_samples, target_samples)
    vocal = _pad_or_trim_audio(vocal_samples, target_samples)
    instrumental *= _linear_from_db(instrumental_gain_db)

    activity = _compute_vocal_activity_envelope(vocal, sample_rate=sample_rate)
    envelope = activity["envelope"]
    strength = max(0.05, min(0.95, float(ducking_strength)))
    duck_floor = max(0.58, 1.0 - (strength * 0.55))
    duck_curve = duck_floor + ((1.0 - envelope) * (1.0 - duck_floor))
    contextual_events: List[Dict[str, Any]] = []
    if macrosections:
        sections = list((macrosections or {}).get("sections") or [])
        bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}
        if sections:
            samples_per_bar = max(1, int(round((target_samples / max(1, len(bar_to_section) or max(1, len(sections)))))))
            section_scale_map = {
                "intro": 1.08,
                "verse": 1.02,
                "pre_hook": 0.98,
                "hook": 0.92,
                "break": 1.14,
                "transition": 1.06,
                "bridge": 1.0,
                "outro": 0.96,
            }
            section_curve = np.ones(target_samples, dtype=np.float32)
            for bar_index, section_index in bar_to_section.items():
                if section_index < 0 or section_index >= len(sections):
                    continue
                section_type = _clean_text(sections[section_index].get("section_type")).lower()
                scale = float(section_scale_map.get(section_type, 1.0))
                start_index = max(0, int(bar_index * samples_per_bar))
                end_index = min(target_samples, start_index + samples_per_bar)
                if end_index <= start_index:
                    continue
                section_curve[start_index:end_index] *= scale
                contextual_events.append(
                    {
                        "bar_index": int(bar_index),
                        "section_type": section_type or None,
                        "duck_scale": round(float(scale), 4),
                    }
                )
            smooth = max(64, int(round(0.06 * sample_rate)))
            if section_curve.size > smooth:
                kernel = np.hanning(smooth).astype(np.float32)
                kernel_sum = float(np.sum(kernel))
                if kernel_sum > 0:
                    kernel /= kernel_sum
                    section_curve = np.convolve(section_curve, kernel, mode="same").astype(np.float32)
            duck_curve = duck_curve * np.clip(section_curve, 0.88, 1.16)
    duck_curve = np.clip(duck_curve, duck_floor, 1.0).astype(np.float32)
    ducked = instrumental * duck_curve[:, np.newaxis]
    ducked = _normalize_audio_buffer(ducked, peak=0.92)
    _write_audio_array_to_wav(ducked, output_path, sample_rate=sample_rate)
    return {
        "output_path": output_path,
        "active_threshold": activity["active_threshold"],
        "active_ratio": activity["active_ratio"],
        "duck_floor": round(float(duck_floor), 6),
        "target_duration_seconds": round(float(target_samples / float(sample_rate)), 4),
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
    }


def _apply_vocal_spectral_space(
    instrumental_path: str,
    output_path: str,
    vocal_analysis: Dict[str, Any],
    phrase_blocks: Optional[Dict[str, Any]] = None,
    tempo_bpm: Optional[float] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, Any]:
    analysis = vocal_analysis.get("analysis") if "analysis" in vocal_analysis else vocal_analysis
    try:
        vocal_centroid = float(analysis.get("spectral_centroid_hz") or 3200.0)
    except Exception:
        vocal_centroid = 3200.0
    centroid_hz = max(1800.0, min(5200.0, vocal_centroid))
    upper_presence_hz = max(2600.0, min(6800.0, centroid_hz * 1.32))
    active_ratio = max(0.0, min(1.0, 1.0 - float(analysis.get("silence_ratio") or 0.0)))
    primary_cut_db = -1.0 - (active_ratio * 0.85)
    secondary_cut_db = -0.7 - (active_ratio * 0.55)
    filter_chain = ",".join(
        [
            "highpass=f=28",
            "lowpass=f=16500",
            f"equalizer=f={centroid_hz:.1f}:t=q:w=1.1:g={primary_cut_db:.2f}",
            f"equalizer=f={upper_presence_hz:.1f}:t=q:w=1.4:g={secondary_cut_db:.2f}",
        ]
    )

    if signal is None:
        ffmpeg_path = ensure_runtime_ready("mix")["ffmpeg_path"]
        result = _run_command(
            [
                ffmpeg_path,
                "-hide_banner",
                "-y",
                "-i",
                instrumental_path,
                "-af",
                filter_chain,
                "-c:a",
                "pcm_s16le",
                output_path,
            ]
        )
        if result.returncode != 0:
            _raise_command_error("ffmpeg vocal spectral space", result)
        return {
            "output_path": output_path,
            "mode": "static_ffmpeg_fallback",
            "filter_chain": filter_chain,
            "centroid_hz": round(float(centroid_hz), 2),
            "upper_presence_hz": round(float(upper_presence_hz), 2),
            "active_ratio": round(float(active_ratio), 4),
            "primary_cut_db": round(float(primary_cut_db), 4),
            "secondary_cut_db": round(float(secondary_cut_db), 4),
        }

    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]

    primary_db_envelope = np.full(total_samples, float(primary_cut_db), dtype=np.float32)
    secondary_db_envelope = np.full(total_samples, float(secondary_cut_db), dtype=np.float32)
    profile_bar_counts: Dict[str, int] = {}
    contextual_events: List[Dict[str, Any]] = []

    if phrase_blocks and tempo_bpm and total_samples > 0:
        blocks = list(phrase_blocks.get("blocks") or [])
        stages = {int(key): value for key, value in dict(phrase_blocks.get("bar_stages") or {}).items()}
        bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
        samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
        stage_profile_adjustments = {
            "dense_long": {
                "entry": (-0.55, -0.42),
                "sustain": (-0.42, -0.34),
                "release": (0.12, 0.16),
                "recovery": (0.18, 0.24),
            },
            "dense_short": {
                "entry": (-0.42, -0.32),
                "sustain": (-0.32, -0.24),
                "release": (0.1, 0.14),
                "recovery": (0.14, 0.2),
            },
            "open_long": {
                "entry": (-0.16, -0.12),
                "sustain": (-0.06, -0.02),
                "release": (0.24, 0.3),
                "recovery": (0.36, 0.44),
            },
            "open_short": {
                "entry": (-0.08, -0.06),
                "sustain": (0.04, 0.08),
                "release": (0.32, 0.38),
                "recovery": (0.46, 0.54),
            },
        }

        for block in blocks:
            profile_name = _clean_text(block.get("profile")).lower() or "unclassified"
            profile_bar_counts[profile_name] = int(profile_bar_counts.get(profile_name, 0)) + int(block.get("bars") or 0)

        for bar_index, stage_name in stages.items():
            block_index = bar_to_block.get(int(bar_index))
            if block_index is None:
                block_index = bar_to_block.get(int(bar_index) - 1)
            block_profile = ""
            if block_index is not None and 0 <= block_index < len(blocks):
                block_profile = _clean_text(blocks[block_index].get("profile")).lower()
            band_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
            if not band_adjustment:
                continue
            start_index = max(0, int(bar_index) * samples_per_bar)
            end_index = min(total_samples, start_index + samples_per_bar)
            if end_index <= start_index:
                continue
            primary_db_envelope[start_index:end_index] += float(band_adjustment[0])
            secondary_db_envelope[start_index:end_index] += float(band_adjustment[1])
            contextual_events.append(
                {
                    "bar_index": int(bar_index),
                    "stage": str(stage_name),
                    "block_profile": block_profile or None,
                    "primary_adjust_db": round(float(band_adjustment[0]), 4),
                    "secondary_adjust_db": round(float(band_adjustment[1]), 4),
                }
            )

        smoothing_samples = max(64, int(round(0.08 * effective_sample_rate)))
        if total_samples > smoothing_samples:
            kernel = np.hanning(smoothing_samples).astype(np.float32)
            kernel_sum = float(np.sum(kernel))
            if kernel_sum > 0:
                kernel /= kernel_sum
                primary_db_envelope = np.convolve(primary_db_envelope, kernel, mode="same").astype(np.float32)
                secondary_db_envelope = np.convolve(secondary_db_envelope, kernel, mode="same").astype(np.float32)

    primary_db_envelope = np.clip(primary_db_envelope, -3.4, -0.08)
    secondary_db_envelope = np.clip(secondary_db_envelope, -2.6, -0.06)
    primary_reduction = (1.0 - np.clip(np.power(10.0, primary_db_envelope / 20.0), 0.0, 1.0)).astype(np.float32)
    secondary_reduction = (1.0 - np.clip(np.power(10.0, secondary_db_envelope / 20.0), 0.0, 1.0)).astype(np.float32)

    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    primary_low_hz = max(150.0, centroid_hz * 0.68)
    primary_high_hz = min(nyquist_hz * 0.96, centroid_hz * 1.24)
    secondary_low_hz = max(2200.0, upper_presence_hz * 0.76)
    secondary_high_hz = min(nyquist_hz * 0.98, upper_presence_hz * 1.18)
    if primary_high_hz <= primary_low_hz:
        primary_low_hz = max(150.0, centroid_hz * 0.58)
        primary_high_hz = min(nyquist_hz * 0.96, primary_low_hz * 1.45)
    if secondary_high_hz <= secondary_low_hz:
        secondary_low_hz = max(2200.0, upper_presence_hz * 0.7)
        secondary_high_hz = min(nyquist_hz * 0.98, secondary_low_hz * 1.28)

    primary_sos = signal.butter(2, [primary_low_hz, primary_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    secondary_sos = signal.butter(2, [secondary_low_hz, secondary_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    highpass_sos = signal.butter(2, 28.0, btype="highpass", fs=effective_sample_rate, output="sos")
    lowpass_sos = signal.butter(2, min(16500.0, nyquist_hz * 0.96), btype="lowpass", fs=effective_sample_rate, output="sos")

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        primary_band = signal.sosfiltfilt(primary_sos, channel).astype(np.float32)
        secondary_band = signal.sosfiltfilt(secondary_sos, channel).astype(np.float32)
        cleaned = channel - (primary_band * primary_reduction) - (secondary_band * secondary_reduction)
        cleaned = signal.sosfiltfilt(highpass_sos, cleaned).astype(np.float32)
        cleaned = signal.sosfiltfilt(lowpass_sos, cleaned).astype(np.float32)
        processed_channels.append(cleaned.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)

    return {
        "output_path": output_path,
        "mode": "contextual_phrase_space" if phrase_blocks else "contextual_static_space",
        "filter_chain": filter_chain,
        "centroid_hz": round(float(centroid_hz), 2),
        "upper_presence_hz": round(float(upper_presence_hz), 2),
        "active_ratio": round(float(active_ratio), 4),
        "primary_cut_db": round(float(primary_cut_db), 4),
        "secondary_cut_db": round(float(secondary_cut_db), 4),
        "primary_band_hz": [round(float(primary_low_hz), 2), round(float(primary_high_hz), 2)],
        "secondary_band_hz": [round(float(secondary_low_hz), 2), round(float(secondary_high_hz), 2)],
        "profile_bar_counts": profile_bar_counts or None,
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
    }


def _apply_vocal_stereo_space(
    instrumental_path: str,
    output_path: str,
    phrase_blocks: Optional[Dict[str, Any]],
    tempo_bpm: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_stereo_space",
            "contextual_events": [],
            "contextual_events_count": 0,
            "profile_bar_counts": None,
        }

    mid = ((stereo[:, 0] + stereo[:, 1]) * 0.5).astype(np.float32)
    side = ((stereo[:, 0] - stereo[:, 1]) * 0.5).astype(np.float32)
    mid_gain_db = np.zeros(total_samples, dtype=np.float32)
    side_gain_db = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    profile_bar_counts: Dict[str, int] = {}

    blocks = list((phrase_blocks or {}).get("blocks") or [])
    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}
    bar_to_block = {int(key): int(value) for key, value in dict((phrase_blocks or {}).get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    stage_profile_adjustments = {
        "dense_long": {
            "entry": (-1.05, 0.35),
            "sustain": (-0.88, 0.28),
            "release": (-0.24, 0.48),
            "recovery": (-0.08, 0.72),
        },
        "dense_short": {
            "entry": (-0.9, 0.32),
            "sustain": (-0.72, 0.24),
            "release": (-0.18, 0.44),
            "recovery": (0.0, 0.66),
        },
        "open_long": {
            "entry": (-0.46, 0.24),
            "sustain": (-0.3, 0.2),
            "release": (0.08, 0.56),
            "recovery": (0.18, 0.84),
        },
        "open_short": {
            "entry": (-0.32, 0.2),
            "sustain": (-0.18, 0.16),
            "release": (0.12, 0.62),
            "recovery": (0.24, 0.95),
        },
    }

    for block in blocks:
        profile_name = _clean_text(block.get("profile")).lower() or "unclassified"
        profile_bar_counts[profile_name] = int(profile_bar_counts.get(profile_name, 0)) + int(block.get("bars") or 0)

    for bar_index, stage_name in stages.items():
        block_index = bar_to_block.get(int(bar_index))
        if block_index is None:
            block_index = bar_to_block.get(int(bar_index) - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        start_index = max(0, int(bar_index) * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        mid_gain_db[start_index:end_index] += float(stage_adjustment[0])
        side_gain_db[start_index:end_index] += float(stage_adjustment[1])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "mid_adjust_db": round(float(stage_adjustment[0]), 4),
                "side_adjust_db": round(float(stage_adjustment[1]), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.09 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            mid_gain_db = np.convolve(mid_gain_db, kernel, mode="same").astype(np.float32)
            side_gain_db = np.convolve(side_gain_db, kernel, mode="same").astype(np.float32)

    mid_gain = np.power(10.0, np.clip(mid_gain_db, -2.4, 0.6) / 20.0).astype(np.float32)
    side_gain = np.power(10.0, np.clip(side_gain_db, -0.3, 1.8) / 20.0).astype(np.float32)
    processed_mid = mid * mid_gain
    processed_side = side * side_gain
    processed_left = processed_mid + processed_side
    processed_right = processed_mid - processed_side
    processed = np.stack([processed_left, processed_right], axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_stereo_space",
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "profile_bar_counts": profile_bar_counts or None,
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_contextual_vocal_clarity(
    vocal_path: str,
    output_path: str,
    vocal_analysis: Dict[str, Any],
    phrase_blocks: Optional[Dict[str, Any]],
    tempo_bpm: float,
    macrosections: Optional[Dict[str, Any]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    analysis = vocal_analysis.get("analysis") if "analysis" in vocal_analysis else vocal_analysis

    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_vocal_clarity_fallback",
            "contextual_events": [],
            "contextual_events_count": 0,
            "profile_bar_counts": None,
        }

    try:
        centroid_hz = float(analysis.get("spectral_centroid_hz") or 3000.0)
    except Exception:
        centroid_hz = 3000.0
    presence_low_hz = max(1800.0, min(2400.0, centroid_hz * 0.78))
    presence_high_hz = max(3600.0, min(5200.0, centroid_hz * 1.34))
    body_low_hz = 170.0
    body_high_hz = 420.0
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    presence_high_hz = min(presence_high_hz, nyquist_hz * 0.95)

    presence_gain_db = np.zeros(total_samples, dtype=np.float32)
    body_gain_db = np.zeros(total_samples, dtype=np.float32)
    side_gain_db = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    profile_bar_counts: Dict[str, int] = {}
    macrosection_events: List[Dict[str, Any]] = []

    blocks = list((phrase_blocks or {}).get("blocks") or [])
    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}
    bar_to_block = {int(key): int(value) for key, value in dict((phrase_blocks or {}).get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    stage_profile_adjustments = {
        "dense_long": {
            "entry": (0.95, -0.42, -0.55),
            "sustain": (0.78, -0.3, -0.46),
            "release": (0.42, -0.14, -0.18),
            "recovery": (0.2, 0.0, 0.0),
        },
        "dense_short": {
            "entry": (0.82, -0.34, -0.48),
            "sustain": (0.64, -0.24, -0.38),
            "release": (0.36, -0.1, -0.12),
            "recovery": (0.18, 0.0, 0.04),
        },
        "open_long": {
            "entry": (0.56, -0.2, -0.24),
            "sustain": (0.44, -0.14, -0.16),
            "release": (0.28, -0.04, 0.06),
            "recovery": (0.16, 0.02, 0.12),
        },
        "open_short": {
            "entry": (0.48, -0.16, -0.2),
            "sustain": (0.34, -0.08, -0.08),
            "release": (0.22, -0.02, 0.08),
            "recovery": (0.12, 0.04, 0.16),
        },
    }
    macrosection_adjustments = {
        "intro": (0.06, -0.04, -0.08),
        "verse": (0.02, -0.02, -0.02),
        "pre_hook": (0.08, -0.04, -0.05),
        "hook": (0.16, -0.06, -0.12),
        "break": (-0.04, 0.02, 0.04),
        "transition": (0.04, -0.02, 0.0),
        "bridge": (0.02, -0.01, 0.02),
        "outro": (-0.02, 0.02, 0.06),
    }

    for block in blocks:
        profile_name = _clean_text(block.get("profile")).lower() or "unclassified"
        profile_bar_counts[profile_name] = int(profile_bar_counts.get(profile_name, 0)) + int(block.get("bars") or 0)

    for bar_index, stage_name in stages.items():
        block_index = bar_to_block.get(int(bar_index))
        if block_index is None:
            block_index = bar_to_block.get(int(bar_index) - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        section_type = _resolve_macrosection_type_for_bar(bar_index, macrosections)
        if section_type:
            macro_adjustment = macrosection_adjustments.get(section_type)
            if macro_adjustment:
                stage_adjustment = (
                    float(stage_adjustment[0]) + float(macro_adjustment[0]),
                    float(stage_adjustment[1]) + float(macro_adjustment[1]),
                    float(stage_adjustment[2]) + float(macro_adjustment[2]),
                )
                macrosection_events.append(
                    {
                        "bar_index": int(bar_index),
                        "section_type": section_type or None,
                        "presence_adjust_db": round(float(macro_adjustment[0]), 4),
                        "body_adjust_db": round(float(macro_adjustment[1]), 4),
                        "side_adjust_db": round(float(macro_adjustment[2]), 4),
                    }
                )
        start_index = max(0, int(bar_index) * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        presence_gain_db[start_index:end_index] += float(stage_adjustment[0])
        body_gain_db[start_index:end_index] += float(stage_adjustment[1])
        side_gain_db[start_index:end_index] += float(stage_adjustment[2])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "presence_adjust_db": round(float(stage_adjustment[0]), 4),
                "body_adjust_db": round(float(stage_adjustment[1]), 4),
                "side_adjust_db": round(float(stage_adjustment[2]), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.08 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            presence_gain_db = np.convolve(presence_gain_db, kernel, mode="same").astype(np.float32)
            body_gain_db = np.convolve(body_gain_db, kernel, mode="same").astype(np.float32)
            side_gain_db = np.convolve(side_gain_db, kernel, mode="same").astype(np.float32)

    presence_sos = signal.butter(
        2,
        [presence_low_hz, presence_high_hz],
        btype="bandpass",
        fs=effective_sample_rate,
        output="sos",
    )
    body_sos = signal.butter(
        2,
        [body_low_hz, min(body_high_hz, nyquist_hz * 0.8)],
        btype="bandpass",
        fs=effective_sample_rate,
        output="sos",
    )
    highpass_sos = signal.butter(2, 90.0, btype="highpass", fs=effective_sample_rate, output="sos")
    lowpass_sos = signal.butter(2, min(14000.0, nyquist_hz * 0.94), btype="lowpass", fs=effective_sample_rate, output="sos")

    left = stereo[:, 0].astype(np.float32)
    right = stereo[:, 1].astype(np.float32)
    presence_gain = np.power(10.0, np.clip(presence_gain_db, 0.0, 1.4) / 20.0).astype(np.float32) - 1.0
    body_gain = 1.0 - np.power(10.0, np.clip(body_gain_db, -0.7, 0.2) / 20.0).astype(np.float32)
    side_gain = np.power(10.0, np.clip(side_gain_db, -1.0, 0.25) / 20.0).astype(np.float32)

    processed_channels: List[np.ndarray] = []
    for channel in (left, right):
        channel_presence = signal.sosfiltfilt(presence_sos, channel).astype(np.float32)
        channel_body = signal.sosfiltfilt(body_sos, channel).astype(np.float32)
        processed = channel + (channel_presence * presence_gain) - (channel_body * body_gain)
        processed = signal.sosfiltfilt(highpass_sos, processed).astype(np.float32)
        processed = signal.sosfiltfilt(lowpass_sos, processed).astype(np.float32)
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    mid = ((processed[:, 0] + processed[:, 1]) * 0.5).astype(np.float32)
    side = ((processed[:, 0] - processed[:, 1]) * 0.5).astype(np.float32)
    side *= side_gain
    centered_left = mid + side
    centered_right = mid - side
    centered = np.stack([centered_left, centered_right], axis=1)
    centered = _normalize_audio_buffer(centered, peak=0.92)
    _write_audio_array_to_wav(centered, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_vocal_clarity",
        "presence_band_hz": [round(float(presence_low_hz), 2), round(float(presence_high_hz), 2)],
        "body_band_hz": [round(float(body_low_hz), 2), round(float(min(body_high_hz, nyquist_hz * 0.8)), 2)],
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "macrosection_events": macrosection_events[:240],
        "macrosection_events_count": len(macrosection_events),
        "profile_bar_counts": profile_bar_counts or None,
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_vocal_phrase_level_automation(
    vocal_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    macrosections: Optional[Dict[str, Any]],
    sample_rate: int,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    total_samples = stereo.shape[0]
    stages = dict(phrase_blocks.get("bar_stages") or {})
    blocks = list(phrase_blocks.get("blocks") or [])
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    sections = list((macrosections or {}).get("sections") or [])
    bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}
    stage_gain_db = {
        "entry": 0.32,
        "sustain": 0.18,
        "release": -0.08,
        "recovery": -0.24,
    }
    profile_stage_adjustments = {
        "dense_long": {"entry": 0.22, "sustain": 0.16, "release": -0.04, "recovery": -0.08},
        "dense_short": {"entry": 0.16, "sustain": 0.1, "release": -0.02, "recovery": -0.05},
        "open_long": {"entry": 0.08, "sustain": 0.03, "release": -0.02, "recovery": -0.08},
        "open_short": {"entry": 0.02, "sustain": -0.02, "release": -0.04, "recovery": -0.1},
    }
    section_gain_adjustments = {
        "intro": -0.08,
        "verse": 0.0,
        "pre_hook": 0.04,
        "hook": 0.12,
        "break": -0.1,
        "transition": -0.02,
        "bridge": 0.02,
        "outro": -0.04,
    }

    envelope = np.ones(total_samples, dtype=np.float32)
    applied_stage_gains: List[Dict[str, Any]] = []
    macrosection_events: List[Dict[str, Any]] = []
    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        if block_index is None:
            block_index = bar_to_block.get(bar_index - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        gain_db = float(stage_gain_db.get(stage_name, 0.0))
        gain_db += float((profile_stage_adjustments.get(block_profile) or {}).get(stage_name, 0.0))
        section_index = bar_to_section.get(bar_index)
        section_type = ""
        if section_index is not None and 0 <= section_index < len(sections):
            section_type = _clean_text(sections[section_index].get("section_type")).lower()
            section_gain_db = float(section_gain_adjustments.get(section_type, 0.0))
            gain_db += section_gain_db
            if section_gain_db != 0.0:
                macrosection_events.append(
                    {
                        "bar_index": int(bar_index),
                        "section_type": section_type or None,
                        "gain_db": round(float(section_gain_db), 4),
                    }
                )
        if gain_db == 0.0:
            continue
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        envelope[start_index:end_index] *= float(10 ** (gain_db / 20.0))
        applied_stage_gains.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_index": int(block_index) if block_index is not None else None,
                "block_profile": block_profile or None,
                "gain_db": round(gain_db, 4),
            }
        )

    transition_events: List[Dict[str, Any]] = []

    def _multiply_envelope_window(
        start_index: int,
        end_index: int,
        start_gain: float,
        end_gain: float,
        event_type: str,
        block_index: int,
        block_profile: str,
    ) -> None:
        start = max(0, int(start_index))
        end = min(total_samples, int(end_index))
        if end <= start:
            return
        ramp = np.linspace(float(start_gain), float(end_gain), end - start, dtype=np.float32)
        envelope[start:end] *= ramp
        transition_events.append(
            {
                "type": event_type,
                "block_index": int(block_index),
                "block_profile": block_profile or None,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_gain": round(float(start_gain), 4),
                "end_gain": round(float(end_gain), 4),
            }
        )

    for block_index, block in enumerate(blocks):
        start_bar = int(block.get("start_bar") or 0)
        end_bar = int(block.get("end_bar") or start_bar)
        block_profile = _clean_text(block.get("profile")).lower()
        block_start = start_bar * samples_per_bar
        block_end = min(total_samples, (end_bar + 1) * samples_per_bar)
        pre_roll = max(64, int(round(samples_per_bar * 0.1)))
        attack = max(64, int(round(samples_per_bar * 0.16)))
        release = max(64, int(round(samples_per_bar * 0.18)))
        recovery = max(64, int(round(samples_per_bar * 0.24)))
        if block_profile == "dense_long":
            pre_entry_end = 0.98
            entry_start = 1.04
            release_end = 0.99
            recovery_start_gain = 0.96
        elif block_profile == "dense_short":
            pre_entry_end = 0.985
            entry_start = 1.03
            release_end = 0.995
            recovery_start_gain = 0.97
        elif block_profile == "open_long":
            pre_entry_end = 0.99
            entry_start = 1.02
            release_end = 0.995
            recovery_start_gain = 0.975
        else:
            pre_entry_end = 0.992
            entry_start = 1.015
            release_end = 0.998
            recovery_start_gain = 0.98

        _multiply_envelope_window(
            start_index=block_start - pre_roll,
            end_index=block_start,
            start_gain=1.0,
            end_gain=pre_entry_end,
            event_type="pre_entry",
            block_index=block_index,
            block_profile=block_profile,
        )
        _multiply_envelope_window(
            start_index=block_start,
            end_index=block_start + attack,
            start_gain=entry_start,
            end_gain=1.0,
            event_type="entry_settle",
            block_index=block_index,
            block_profile=block_profile,
        )
        _multiply_envelope_window(
            start_index=block_end - release,
            end_index=block_end,
            start_gain=1.0,
            end_gain=release_end,
            event_type="release_settle",
            block_index=block_index,
            block_profile=block_profile,
        )
        _multiply_envelope_window(
            start_index=block_end,
            end_index=block_end + recovery,
            start_gain=recovery_start_gain,
            end_gain=1.0,
            event_type="recovery_relax",
            block_index=block_index,
            block_profile=block_profile,
        )

    smoothing_samples = max(64, int(round(0.06 * effective_sample_rate)))
    if smoothing_samples > 1 and envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.84, 1.18).astype(np.float32)

    automated = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    automated = _normalize_audio_buffer(automated, peak=0.92)
    _write_audio_array_to_wav(automated, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "applied_stage_gains": applied_stage_gains,
        "applied_stage_gains_count": len(applied_stage_gains),
        "macrosection_events": macrosection_events[:240],
        "macrosection_events_count": len(macrosection_events),
        "transition_events": transition_events,
        "transition_events_count": len(transition_events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_contextual_vocal_sibilance_control(
    vocal_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    hotspot_manifest: Optional[Dict[str, Any]] = None,
    macrosections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_vocal_sibilance_fallback",
            "contextual_events": [],
            "contextual_events_count": 0,
        }

    blocks = list(phrase_blocks.get("blocks") or [])
    stages = dict(phrase_blocks.get("bar_stages") or {})
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    sibilance_cut_db = np.zeros(total_samples, dtype=np.float32)
    air_cut_db = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    macrosection_events: List[Dict[str, Any]] = []
    stage_profile_adjustments = {
        "dense_long": {"entry": (-1.25, -0.42), "sustain": (-1.0, -0.34), "release": (-0.62, -0.18), "recovery": (-0.18, 0.0)},
        "dense_short": {"entry": (-1.05, -0.36), "sustain": (-0.82, -0.28), "release": (-0.48, -0.14), "recovery": (-0.14, 0.0)},
        "open_long": {"entry": (-0.72, -0.24), "sustain": (-0.52, -0.18), "release": (-0.28, -0.08), "recovery": (-0.08, 0.02)},
        "open_short": {"entry": (-0.58, -0.2), "sustain": (-0.38, -0.12), "release": (-0.22, -0.06), "recovery": (-0.04, 0.04)},
    }
    macrosection_adjustments = {
        "intro": (-0.08, -0.02),
        "verse": (-0.02, 0.0),
        "pre_hook": (-0.08, -0.02),
        "hook": (-0.16, -0.06),
        "break": (0.06, 0.02),
        "transition": (-0.04, -0.02),
        "bridge": (-0.02, 0.0),
        "outro": (0.04, 0.02),
    }

    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        if block_index is None:
            block_index = bar_to_block.get(bar_index - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        section_type = _resolve_macrosection_type_for_bar(bar_index, macrosections)
        if section_type:
            macro_adjustment = macrosection_adjustments.get(section_type)
            if macro_adjustment:
                stage_adjustment = (
                    float(stage_adjustment[0]) + float(macro_adjustment[0]),
                    float(stage_adjustment[1]) + float(macro_adjustment[1]),
                )
                macrosection_events.append(
                    {
                        "bar_index": int(bar_index),
                        "section_type": section_type or None,
                        "sibilance_cut_db": round(float(macro_adjustment[0]), 4),
                        "air_cut_db": round(float(macro_adjustment[1]), 4),
                    }
                )
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        sibilance_cut_db[start_index:end_index] += float(stage_adjustment[0])
        air_cut_db[start_index:end_index] += float(stage_adjustment[1])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "sibilance_cut_db": round(float(stage_adjustment[0]), 4),
                "air_cut_db": round(float(stage_adjustment[1]), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.05 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            sibilance_cut_db = np.convolve(sibilance_cut_db, kernel, mode="same").astype(np.float32)
            air_cut_db = np.convolve(air_cut_db, kernel, mode="same").astype(np.float32)

    hotspot_events_applied: List[Dict[str, Any]] = []
    for event in list((hotspot_manifest or {}).get("sibilance_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        extra_sibilance_cut = -0.42
        extra_air_cut = -0.16
        sibilance_cut_db[start_index:end_index] += extra_sibilance_cut
        air_cut_db[start_index:end_index] += extra_air_cut
        hotspot_events_applied.append(
            {
                "type": "sibilance_hotspot",
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "extra_sibilance_cut_db": round(float(extra_sibilance_cut), 4),
                "extra_air_cut_db": round(float(extra_air_cut), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
            }
        )

    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    sibilance_low_hz = 4800.0
    sibilance_high_hz = min(9200.0, nyquist_hz * 0.94)
    air_low_hz = min(9800.0, nyquist_hz * 0.7)
    air_high_hz = min(13500.0, nyquist_hz * 0.98)
    if air_high_hz <= air_low_hz:
        air_low_hz = max(7000.0, nyquist_hz * 0.58)
        air_high_hz = min(13500.0, nyquist_hz * 0.96)

    sibilance_sos = signal.butter(2, [sibilance_low_hz, sibilance_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    air_sos = signal.butter(2, [air_low_hz, air_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    sibilance_gain = np.power(10.0, np.clip(sibilance_cut_db, -2.0, 0.0) / 20.0).astype(np.float32)
    air_gain = np.power(10.0, np.clip(air_cut_db, -0.8, 0.1) / 20.0).astype(np.float32)

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        sibilance_band = signal.sosfiltfilt(sibilance_sos, channel).astype(np.float32)
        air_band = signal.sosfiltfilt(air_sos, channel).astype(np.float32)
        processed = channel + (sibilance_band * (sibilance_gain - 1.0)) + (air_band * (air_gain - 1.0))
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_vocal_sibilance_control",
        "sibilance_band_hz": [round(float(sibilance_low_hz), 2), round(float(sibilance_high_hz), 2)],
        "air_band_hz": [round(float(air_low_hz), 2), round(float(air_high_hz), 2)],
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "macrosection_events": macrosection_events[:240],
        "macrosection_events_count": len(macrosection_events),
        "hotspot_events": hotspot_events_applied[:120],
        "hotspot_events_count": len(hotspot_events_applied),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_contextual_instrumental_brightness_space(
    instrumental_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    hotspot_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_brightness_space_fallback",
            "contextual_events": [],
            "contextual_events_count": 0,
        }

    blocks = list(phrase_blocks.get("blocks") or [])
    stages = dict(phrase_blocks.get("bar_stages") or {})
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    brightness_cut_db = np.zeros(total_samples, dtype=np.float32)
    air_cut_db = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    stage_profile_adjustments = {
        "dense_long": {"entry": (-1.15, -0.55), "sustain": (-0.92, -0.46), "release": (-0.42, -0.2), "recovery": (-0.08, 0.0)},
        "dense_short": {"entry": (-0.98, -0.46), "sustain": (-0.78, -0.36), "release": (-0.34, -0.16), "recovery": (-0.04, 0.0)},
        "open_long": {"entry": (-0.62, -0.28), "sustain": (-0.48, -0.22), "release": (-0.18, -0.08), "recovery": (0.0, 0.06)},
        "open_short": {"entry": (-0.5, -0.22), "sustain": (-0.34, -0.16), "release": (-0.12, -0.04), "recovery": (0.06, 0.12)},
    }

    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        if block_index is None:
            block_index = bar_to_block.get(bar_index - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        brightness_cut_db[start_index:end_index] += float(stage_adjustment[0])
        air_cut_db[start_index:end_index] += float(stage_adjustment[1])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "brightness_cut_db": round(float(stage_adjustment[0]), 4),
                "air_cut_db": round(float(stage_adjustment[1]), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.06 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            brightness_cut_db = np.convolve(brightness_cut_db, kernel, mode="same").astype(np.float32)
            air_cut_db = np.convolve(air_cut_db, kernel, mode="same").astype(np.float32)

    hotspot_events_applied: List[Dict[str, Any]] = []
    for event in list((hotspot_manifest or {}).get("sibilance_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        extra_brightness_cut = -0.28
        extra_air_cut = -0.12
        brightness_cut_db[start_index:end_index] += extra_brightness_cut
        air_cut_db[start_index:end_index] += extra_air_cut
        hotspot_events_applied.append(
            {
                "type": "sibilance_hotspot_space",
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "extra_brightness_cut_db": round(float(extra_brightness_cut), 4),
                "extra_air_cut_db": round(float(extra_air_cut), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
            }
        )

    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    brightness_low_hz = 5200.0
    brightness_high_hz = min(9800.0, nyquist_hz * 0.94)
    air_low_hz = min(9800.0, nyquist_hz * 0.7)
    air_high_hz = min(14000.0, nyquist_hz * 0.98)
    if air_high_hz <= air_low_hz:
        air_low_hz = max(7000.0, nyquist_hz * 0.58)
        air_high_hz = min(14000.0, nyquist_hz * 0.96)

    brightness_sos = signal.butter(2, [brightness_low_hz, brightness_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    air_sos = signal.butter(2, [air_low_hz, air_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    brightness_gain = np.power(10.0, np.clip(brightness_cut_db, -1.8, 0.2) / 20.0).astype(np.float32)
    air_gain = np.power(10.0, np.clip(air_cut_db, -0.9, 0.12) / 20.0).astype(np.float32)

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        brightness_band = signal.sosfiltfilt(brightness_sos, channel).astype(np.float32)
        air_band = signal.sosfiltfilt(air_sos, channel).astype(np.float32)
        processed = channel + (brightness_band * (brightness_gain - 1.0)) + (air_band * (air_gain - 1.0))
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_brightness_space",
        "brightness_band_hz": [round(float(brightness_low_hz), 2), round(float(brightness_high_hz), 2)],
        "air_band_hz": [round(float(air_low_hz), 2), round(float(air_high_hz), 2)],
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "hotspot_events": hotspot_events_applied[:120],
        "hotspot_events_count": len(hotspot_events_applied),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_vocal_breath_control(
    vocal_path: str,
    output_path: str,
    hotspot_manifest: Optional[Dict[str, Any]],
    tempo_bpm: float,
    macrosections: Optional[Dict[str, Any]],
    sample_rate: int,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "applied_breath_events": [],
            "applied_breath_events_count": 0,
            "smoothing_samples": 0,
        }

    envelope = np.ones(total_samples, dtype=np.float32)
    applied_events: List[Dict[str, Any]] = []
    macrosection_events: List[Dict[str, Any]] = []
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    section_gain_scale = {
        "intro": 0.98,
        "verse": 1.0,
        "pre_hook": 1.02,
        "hook": 0.96,
        "break": 1.08,
        "transition": 1.01,
        "bridge": 1.02,
        "outro": 1.06,
    }
    for event in list((hotspot_manifest or {}).get("breath_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        pre_roll = max(32, int(round(0.012 * effective_sample_rate)))
        release = max(32, int(round(0.045 * effective_sample_rate)))
        target_gain = 0.84 if _clean_text(event.get("block_profile")).lower() == "dense_long" else 0.88
        section_type = _resolve_macrosection_type_for_bar(int(start_index // samples_per_bar), macrosections)
        target_gain *= float(section_gain_scale.get(section_type, 1.0))
        target_gain = float(np.clip(target_gain, 0.82, 0.94))
        ramp_in_start = max(0, start_index - pre_roll)
        ramp_in = np.linspace(1.0, float(target_gain), start_index - ramp_in_start or 1, dtype=np.float32)
        envelope[ramp_in_start:start_index] *= ramp_in[: max(0, start_index - ramp_in_start)]
        envelope[start_index:end_index] *= float(target_gain)
        recovery_end = min(total_samples, end_index + release)
        ramp_out = np.linspace(float(target_gain), 1.0, recovery_end - end_index or 1, dtype=np.float32)
        envelope[end_index:recovery_end] *= ramp_out[: max(0, recovery_end - end_index)]
        applied_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "target_gain": round(float(target_gain), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
                "section_type": section_type or None,
            }
        )
        if section_type:
            macrosection_events.append(
                {
                    "section_type": section_type or None,
                    "target_gain": round(float(target_gain), 4),
                    "start_sample": int(start_index),
                    "end_sample": int(end_index),
                }
            )

    smoothing_samples = max(64, int(round(0.025 * effective_sample_rate)))
    if envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.8, 1.02).astype(np.float32)

    processed = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "applied_breath_events": applied_events[:120],
        "applied_breath_events_count": len(applied_events),
        "macrosection_events": macrosection_events[:120],
        "macrosection_events_count": len(macrosection_events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_vocal_plosive_control(
    vocal_path: str,
    output_path: str,
    hotspot_manifest: Optional[Dict[str, Any]],
    tempo_bpm: float,
    macrosections: Optional[Dict[str, Any]],
    sample_rate: int,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "vocal_plosive_control_fallback",
            "applied_plosive_events": [],
            "applied_plosive_events_count": 0,
            "band_hz": [90.0, 260.0],
            "smoothing_samples": 0,
        }

    band_gain = np.ones(total_samples, dtype=np.float32)
    body_gain = np.ones(total_samples, dtype=np.float32)
    applied_events: List[Dict[str, Any]] = []
    macrosection_events: List[Dict[str, Any]] = []
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    section_band_scale = {
        "intro": 1.0,
        "verse": 1.0,
        "pre_hook": 0.98,
        "hook": 0.94,
        "break": 1.06,
        "transition": 0.98,
        "bridge": 0.98,
        "outro": 1.04,
    }
    section_body_scale = {
        "intro": 1.0,
        "verse": 1.0,
        "pre_hook": 0.99,
        "hook": 0.96,
        "break": 1.04,
        "transition": 0.99,
        "bridge": 0.99,
        "outro": 1.02,
    }

    for event in list((hotspot_manifest or {}).get("plosive_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        pre_roll = max(32, int(round(0.008 * effective_sample_rate)))
        post_roll = max(32, int(round(0.055 * effective_sample_rate)))
        block_profile = _clean_text(event.get("block_profile")).lower()
        stage_name = _clean_text(event.get("stage")).lower()
        event_band_gain = 0.72 if block_profile == "dense_long" else 0.78
        event_body_gain = 0.9 if stage_name == "entry" else 0.93
        section_type = _resolve_macrosection_type_for_bar(int(start_index // samples_per_bar), macrosections)
        event_band_gain *= float(section_band_scale.get(section_type, 1.0))
        event_body_gain *= float(section_body_scale.get(section_type, 1.0))
        event_band_gain = float(np.clip(event_band_gain, 0.68, 0.84))
        event_body_gain = float(np.clip(event_body_gain, 0.86, 0.97))

        ramp_start = max(0, start_index - pre_roll)
        ramp_in_band = np.linspace(1.0, float(event_band_gain), start_index - ramp_start or 1, dtype=np.float32)
        ramp_in_body = np.linspace(1.0, float(event_body_gain), start_index - ramp_start or 1, dtype=np.float32)
        band_gain[ramp_start:start_index] *= ramp_in_band[: max(0, start_index - ramp_start)]
        body_gain[ramp_start:start_index] *= ramp_in_body[: max(0, start_index - ramp_start)]
        band_gain[start_index:end_index] *= float(event_band_gain)
        body_gain[start_index:end_index] *= float(event_body_gain)
        recovery_end = min(total_samples, end_index + post_roll)
        ramp_out_band = np.linspace(float(event_band_gain), 1.0, recovery_end - end_index or 1, dtype=np.float32)
        ramp_out_body = np.linspace(float(event_body_gain), 1.0, recovery_end - end_index or 1, dtype=np.float32)
        band_gain[end_index:recovery_end] *= ramp_out_band[: max(0, recovery_end - end_index)]
        body_gain[end_index:recovery_end] *= ramp_out_body[: max(0, recovery_end - end_index)]
        applied_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "band_gain": round(float(event_band_gain), 4),
                "body_gain": round(float(event_body_gain), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
                "section_type": section_type or None,
            }
        )
        if section_type:
            macrosection_events.append(
                {
                    "section_type": section_type or None,
                    "band_gain": round(float(event_band_gain), 4),
                    "body_gain": round(float(event_body_gain), 4),
                    "start_sample": int(start_index),
                    "end_sample": int(end_index),
                }
            )

    smoothing_samples = max(64, int(round(0.018 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            band_gain = np.convolve(band_gain, kernel, mode="same").astype(np.float32)
            body_gain = np.convolve(body_gain, kernel, mode="same").astype(np.float32)

    band_gain = np.clip(band_gain, 0.65, 1.02)
    body_gain = np.clip(body_gain, 0.84, 1.02)
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    plosive_low_hz = 90.0
    plosive_high_hz = min(260.0, nyquist_hz * 0.46)
    body_low_hz = 180.0
    body_high_hz = min(420.0, nyquist_hz * 0.62)
    if body_high_hz <= body_low_hz:
        body_low_hz = 140.0
        body_high_hz = min(380.0, nyquist_hz * 0.54)

    plosive_sos = signal.butter(2, [plosive_low_hz, plosive_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    body_sos = signal.butter(2, [body_low_hz, body_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        plosive_band = signal.sosfiltfilt(plosive_sos, channel).astype(np.float32)
        body_band = signal.sosfiltfilt(body_sos, channel).astype(np.float32)
        processed = channel + (plosive_band * (band_gain - 1.0)) + (body_band * (body_gain - 1.0))
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "vocal_plosive_control",
        "band_hz": [round(float(plosive_low_hz), 2), round(float(plosive_high_hz), 2)],
        "body_band_hz": [round(float(body_low_hz), 2), round(float(body_high_hz), 2)],
        "applied_plosive_events": applied_events[:120],
        "applied_plosive_events_count": len(applied_events),
        "macrosection_events": macrosection_events[:120],
        "macrosection_events_count": len(macrosection_events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_instrumental_hotspot_micro_duck(
    instrumental_path: str,
    output_path: str,
    hotspot_manifest: Optional[Dict[str, Any]],
    sample_rate: int,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "applied_hotspot_events": [],
            "applied_hotspot_events_count": 0,
            "smoothing_samples": 0,
        }

    envelope = np.ones(total_samples, dtype=np.float32)
    applied_events: List[Dict[str, Any]] = []

    def _apply_event(event: Dict[str, Any], target_gain: float, event_type: str, pre_ms: float, post_ms: float) -> None:
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            return
        pre_roll = max(32, int(round((pre_ms / 1000.0) * effective_sample_rate)))
        post_roll = max(32, int(round((post_ms / 1000.0) * effective_sample_rate)))
        ramp_in_start = max(0, start_index - pre_roll)
        ramp_in = np.linspace(1.0, float(target_gain), start_index - ramp_in_start or 1, dtype=np.float32)
        envelope[ramp_in_start:start_index] *= ramp_in[: max(0, start_index - ramp_in_start)]
        envelope[start_index:end_index] *= float(target_gain)
        recovery_end = min(total_samples, end_index + post_roll)
        ramp_out = np.linspace(float(target_gain), 1.0, recovery_end - end_index or 1, dtype=np.float32)
        envelope[end_index:recovery_end] *= ramp_out[: max(0, recovery_end - end_index)]
        applied_events.append(
            {
                "type": event_type,
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "target_gain": round(float(target_gain), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
            }
        )

    for event in list((hotspot_manifest or {}).get("sibilance_events") or []):
        _apply_event(event, target_gain=0.93, event_type="sibilance_micro_duck", pre_ms=18.0, post_ms=28.0)
    for event in list((hotspot_manifest or {}).get("breath_events") or []):
        _apply_event(event, target_gain=0.9, event_type="breath_micro_duck", pre_ms=24.0, post_ms=38.0)

    smoothing_samples = max(64, int(round(0.02 * effective_sample_rate)))
    if envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.86, 1.0).astype(np.float32)

    processed = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "applied_hotspot_events": applied_events[:200],
        "applied_hotspot_events_count": len(applied_events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_transcript_line_support(
    instrumental_path: str,
    output_path: str,
    transcript_line_manifest: Optional[Dict[str, Any]],
    sample_rate: int,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    line_events = list((transcript_line_manifest or {}).get("line_events") or [])
    if total_samples <= 0 or not line_events:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "transcript_line_support_fallback",
            "line_support_events": [],
            "line_support_events_count": 0,
            "transcript_line_events_count": len(line_events),
        }

    envelope = np.ones(total_samples, dtype=np.float32)
    line_support_events: List[Dict[str, Any]] = []
    section_gain_map = {
        "intro": 0.985,
        "verse": 0.955,
        "pre_hook": 0.945,
        "hook": 0.935,
        "break": 0.97,
        "transition": 0.948,
        "bridge": 0.952,
        "outro": 0.978,
    }
    line_profile_section_body_adjustments = {
        ("dense_lyric", "verse"): -0.018,
        ("dense_lyric", "hook"): -0.008,
        ("dense_lyric", "break"): -0.012,
        ("emphatic_lyric", "hook"): -0.01,
        ("emphatic_lyric", "transition"): -0.008,
        ("sparse_lyric", "hook"): 0.004,
        ("adlib", "hook"): 0.008,
        ("non_lyric_event", "verse"): 0.012,
        ("non_lyric_event", "break"): 0.014,
    }
    line_profile_section_recovery_adjustments = {
        ("dense_lyric", "verse"): -0.006,
        ("dense_lyric", "hook"): 0.01,
        ("emphatic_lyric", "hook"): 0.012,
        ("emphatic_lyric", "transition"): 0.008,
        ("adlib", "hook"): -0.004,
        ("sparse_lyric", "hook"): 0.004,
        ("non_lyric_event", "verse"): -0.008,
        ("non_lyric_event", "break"): -0.01,
    }

    def _clamp_gain(value: float) -> float:
        return max(0.88, min(1.02, float(value)))

    for event in line_events:
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        section_type = _clean_text(event.get("section_type")).lower()
        density = float(event.get("density") or 0.0)
        word_count = int(event.get("word_count") or 0)
        line_profile = _clean_text(event.get("line_profile")).lower()
        text_preview = _clean_text(event.get("text"))
        body_gain = float(section_gain_map.get(section_type, 0.965))
        if density >= 4.8:
            body_gain -= 0.018
        elif density >= 3.4:
            body_gain -= 0.012
        elif density >= 2.2:
            body_gain -= 0.006
        if word_count >= 12:
            body_gain -= 0.008
        elif word_count >= 8:
            body_gain -= 0.004
        if any(marker in text_preview for marker in ("!", "?")):
            body_gain -= 0.004
        if line_profile == "dense_lyric":
            body_gain -= 0.014
        elif line_profile == "emphatic_lyric":
            body_gain -= 0.01
        elif line_profile == "sparse_lyric":
            body_gain -= 0.004
        elif line_profile == "adlib":
            body_gain += 0.01
        elif line_profile == "non_lyric_event":
            body_gain += 0.018
        body_gain += float(line_profile_section_body_adjustments.get((line_profile, section_type), 0.0))
        body_gain = _clamp_gain(body_gain)

        pre_roll = max(32, int(round(0.026 * effective_sample_rate)))
        recovery_ramp = max(32, int(round(0.05 * effective_sample_rate)))
        settle_ramp = max(32, int(round(0.09 * effective_sample_rate)))
        ramp_start = max(0, start_index - pre_roll)
        if start_index > ramp_start:
            ramp_in = np.linspace(1.0, body_gain, start_index - ramp_start, dtype=np.float32)
            envelope[ramp_start:start_index] *= ramp_in
        envelope[start_index:end_index] *= float(body_gain)

        recovery_gain = 1.0
        if section_type in {"pre_hook", "hook", "transition", "bridge"}:
            recovery_gain = 1.012
        elif section_type in {"verse"}:
            recovery_gain = 1.006
        if text_preview.endswith(("!", "?", ".")):
            recovery_gain += 0.004
        if line_profile == "dense_lyric":
            recovery_gain += 0.004
        elif line_profile == "emphatic_lyric":
            recovery_gain += 0.006
        elif line_profile == "adlib":
            recovery_gain -= 0.004
        elif line_profile == "non_lyric_event":
            recovery_gain -= 0.008
        recovery_gain += float(line_profile_section_recovery_adjustments.get((line_profile, section_type), 0.0))
        recovery_gain = _clamp_gain(recovery_gain)

        recovery_end = min(total_samples, end_index + recovery_ramp)
        if recovery_end > end_index:
            ramp_out = np.linspace(body_gain, recovery_gain, recovery_end - end_index, dtype=np.float32)
            envelope[end_index:recovery_end] *= ramp_out

        settle_end = min(total_samples, recovery_end + settle_ramp)
        if settle_end > recovery_end:
            settle = np.linspace(recovery_gain, 1.0, settle_end - recovery_end, dtype=np.float32)
            envelope[recovery_end:settle_end] *= settle

        line_support_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "section_type": section_type or None,
                "line_profile": line_profile or None,
                "density": round(float(density), 4),
                "word_count": int(word_count),
                "body_gain": round(float(body_gain), 4),
                "recovery_gain": round(float(recovery_gain), 4),
                "text_preview": text_preview[:80] or None,
            }
        )

    smoothing_samples = max(64, int(round(0.025 * effective_sample_rate)))
    if envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.88, 1.015).astype(np.float32)

    processed = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "transcript_line_support",
        "line_support_events": line_support_events[:240],
        "line_support_events_count": len(line_support_events),
        "transcript_line_events_count": len(line_events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_contextual_instrumental_low_mid_space(
    instrumental_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    hotspot_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_low_mid_space_fallback",
            "contextual_events": [],
            "contextual_events_count": 0,
            "hotspot_events": [],
            "hotspot_events_count": 0,
        }

    blocks = list(phrase_blocks.get("blocks") or [])
    stages = dict(phrase_blocks.get("bar_stages") or {})
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    low_mid_cut_db = np.zeros(total_samples, dtype=np.float32)
    mud_cut_db = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    stage_profile_adjustments = {
        "dense_long": {"entry": (-0.82, -0.52), "sustain": (-0.7, -0.42), "release": (-0.28, -0.18), "recovery": (-0.06, 0.0)},
        "dense_short": {"entry": (-0.7, -0.44), "sustain": (-0.56, -0.34), "release": (-0.22, -0.14), "recovery": (-0.04, 0.0)},
        "open_long": {"entry": (-0.42, -0.24), "sustain": (-0.3, -0.18), "release": (-0.12, -0.08), "recovery": (0.0, 0.04)},
        "open_short": {"entry": (-0.34, -0.18), "sustain": (-0.22, -0.12), "release": (-0.08, -0.04), "recovery": (0.04, 0.08)},
    }

    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        if block_index is None:
            block_index = bar_to_block.get(bar_index - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        low_mid_cut_db[start_index:end_index] += float(stage_adjustment[0])
        mud_cut_db[start_index:end_index] += float(stage_adjustment[1])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "low_mid_cut_db": round(float(stage_adjustment[0]), 4),
                "mud_cut_db": round(float(stage_adjustment[1]), 4),
            }
        )

    hotspot_events_applied: List[Dict[str, Any]] = []
    for event in list((hotspot_manifest or {}).get("plosive_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        extra_low_mid_cut = -0.48
        extra_mud_cut = -0.22
        low_mid_cut_db[start_index:end_index] += extra_low_mid_cut
        mud_cut_db[start_index:end_index] += extra_mud_cut
        hotspot_events_applied.append(
            {
                "type": "plosive_low_mid_space",
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "extra_low_mid_cut_db": round(float(extra_low_mid_cut), 4),
                "extra_mud_cut_db": round(float(extra_mud_cut), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
            }
        )

    smoothing_samples = max(64, int(round(0.05 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            low_mid_cut_db = np.convolve(low_mid_cut_db, kernel, mode="same").astype(np.float32)
            mud_cut_db = np.convolve(mud_cut_db, kernel, mode="same").astype(np.float32)

    low_mid_cut_db = np.clip(low_mid_cut_db, -1.8, 0.14)
    mud_cut_db = np.clip(mud_cut_db, -0.9, 0.1)
    low_mid_gain = np.power(10.0, low_mid_cut_db / 20.0).astype(np.float32)
    mud_gain = np.power(10.0, mud_cut_db / 20.0).astype(np.float32)
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    low_mid_low_hz = 130.0
    low_mid_high_hz = min(320.0, nyquist_hz * 0.42)
    mud_low_hz = 240.0
    mud_high_hz = min(520.0, nyquist_hz * 0.66)
    if mud_high_hz <= mud_low_hz:
        mud_low_hz = 220.0
        mud_high_hz = min(460.0, nyquist_hz * 0.58)

    low_mid_sos = signal.butter(2, [low_mid_low_hz, low_mid_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    mud_sos = signal.butter(2, [mud_low_hz, mud_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        low_mid_band = signal.sosfiltfilt(low_mid_sos, channel).astype(np.float32)
        mud_band = signal.sosfiltfilt(mud_sos, channel).astype(np.float32)
        processed = channel + (low_mid_band * (low_mid_gain - 1.0)) + (mud_band * (mud_gain - 1.0))
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_low_mid_space",
        "low_mid_band_hz": [round(float(low_mid_low_hz), 2), round(float(low_mid_high_hz), 2)],
        "mud_band_hz": [round(float(mud_low_hz), 2), round(float(mud_high_hz), 2)],
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "hotspot_events": hotspot_events_applied[:120],
        "hotspot_events_count": len(hotspot_events_applied),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_vocal_harshness_control(
    vocal_path: str,
    output_path: str,
    hotspot_manifest: Optional[Dict[str, Any]],
    tempo_bpm: float,
    macrosections: Optional[Dict[str, Any]],
    sample_rate: int,
) -> Dict[str, Any]:
    vocal_samples, resolved_sample_rate = _load_audio_array(
        vocal_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(vocal_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "vocal_harshness_control_fallback",
            "applied_harshness_events": [],
            "applied_harshness_events_count": 0,
            "band_hz": [2300.0, 4600.0],
            "smoothing_samples": 0,
        }

    harshness_gain = np.ones(total_samples, dtype=np.float32)
    air_companion_gain = np.ones(total_samples, dtype=np.float32)
    applied_events: List[Dict[str, Any]] = []
    macrosection_events: List[Dict[str, Any]] = []
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    section_harshness_scale = {
        "intro": 1.0,
        "verse": 1.0,
        "pre_hook": 0.98,
        "hook": 0.94,
        "break": 1.06,
        "transition": 0.98,
        "bridge": 0.99,
        "outro": 1.02,
    }
    section_air_scale = {
        "intro": 1.0,
        "verse": 1.0,
        "pre_hook": 0.99,
        "hook": 0.96,
        "break": 1.05,
        "transition": 0.99,
        "bridge": 1.0,
        "outro": 1.03,
    }
    for event in list((hotspot_manifest or {}).get("harshness_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        pre_roll = max(32, int(round(0.01 * effective_sample_rate)))
        post_roll = max(32, int(round(0.05 * effective_sample_rate)))
        block_profile = _clean_text(event.get("block_profile")).lower()
        event_harshness_gain = 0.78 if block_profile == "dense_long" else 0.84
        event_air_gain = 0.93 if _clean_text(event.get("stage")).lower() == "entry" else 0.96
        section_type = _resolve_macrosection_type_for_bar(int(start_index // samples_per_bar), macrosections)
        event_harshness_gain *= float(section_harshness_scale.get(section_type, 1.0))
        event_air_gain *= float(section_air_scale.get(section_type, 1.0))
        event_harshness_gain = float(np.clip(event_harshness_gain, 0.74, 0.88))
        event_air_gain = float(np.clip(event_air_gain, 0.9, 0.99))
        ramp_start = max(0, start_index - pre_roll)
        ramp_in_h = np.linspace(1.0, float(event_harshness_gain), start_index - ramp_start or 1, dtype=np.float32)
        ramp_in_a = np.linspace(1.0, float(event_air_gain), start_index - ramp_start or 1, dtype=np.float32)
        harshness_gain[ramp_start:start_index] *= ramp_in_h[: max(0, start_index - ramp_start)]
        air_companion_gain[ramp_start:start_index] *= ramp_in_a[: max(0, start_index - ramp_start)]
        harshness_gain[start_index:end_index] *= float(event_harshness_gain)
        air_companion_gain[start_index:end_index] *= float(event_air_gain)
        recovery_end = min(total_samples, end_index + post_roll)
        ramp_out_h = np.linspace(float(event_harshness_gain), 1.0, recovery_end - end_index or 1, dtype=np.float32)
        ramp_out_a = np.linspace(float(event_air_gain), 1.0, recovery_end - end_index or 1, dtype=np.float32)
        harshness_gain[end_index:recovery_end] *= ramp_out_h[: max(0, recovery_end - end_index)]
        air_companion_gain[end_index:recovery_end] *= ramp_out_a[: max(0, recovery_end - end_index)]
        applied_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "harshness_gain": round(float(event_harshness_gain), 4),
                "air_companion_gain": round(float(event_air_gain), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
                "section_type": section_type or None,
            }
        )
        if section_type:
            macrosection_events.append(
                {
                    "section_type": section_type or None,
                    "harshness_gain": round(float(event_harshness_gain), 4),
                    "air_companion_gain": round(float(event_air_gain), 4),
                    "start_sample": int(start_index),
                    "end_sample": int(end_index),
                }
            )

    smoothing_samples = max(64, int(round(0.02 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            harshness_gain = np.convolve(harshness_gain, kernel, mode="same").astype(np.float32)
            air_companion_gain = np.convolve(air_companion_gain, kernel, mode="same").astype(np.float32)

    harshness_gain = np.clip(harshness_gain, 0.72, 1.02)
    air_companion_gain = np.clip(air_companion_gain, 0.9, 1.02)
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    harsh_low_hz = 2300.0
    harsh_high_hz = min(4600.0, nyquist_hz * 0.84)
    air_low_hz = min(5200.0, nyquist_hz * 0.72)
    air_high_hz = min(7800.0, nyquist_hz * 0.92)
    if air_high_hz <= air_low_hz:
        air_low_hz = max(4200.0, nyquist_hz * 0.58)
        air_high_hz = min(7200.0, nyquist_hz * 0.86)

    harsh_sos = signal.butter(2, [harsh_low_hz, harsh_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    air_sos = signal.butter(2, [air_low_hz, air_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        harsh_band = signal.sosfiltfilt(harsh_sos, channel).astype(np.float32)
        air_band = signal.sosfiltfilt(air_sos, channel).astype(np.float32)
        processed = channel + (harsh_band * (harshness_gain - 1.0)) + (air_band * (air_companion_gain - 1.0))
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "vocal_harshness_control",
        "band_hz": [round(float(harsh_low_hz), 2), round(float(harsh_high_hz), 2)],
        "air_band_hz": [round(float(air_low_hz), 2), round(float(air_high_hz), 2)],
        "applied_harshness_events": applied_events[:120],
        "applied_harshness_events_count": len(applied_events),
        "macrosection_events": macrosection_events[:120],
        "macrosection_events_count": len(macrosection_events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_contextual_instrumental_presence_space(
    instrumental_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    hotspot_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0 or signal is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "contextual_presence_space_fallback",
            "contextual_events": [],
            "contextual_events_count": 0,
            "hotspot_events": [],
            "hotspot_events_count": 0,
        }

    blocks = list(phrase_blocks.get("blocks") or [])
    stages = dict(phrase_blocks.get("bar_stages") or {})
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    presence_cut_db = np.zeros(total_samples, dtype=np.float32)
    upper_mid_cut_db = np.zeros(total_samples, dtype=np.float32)
    contextual_events: List[Dict[str, Any]] = []
    stage_profile_adjustments = {
        "dense_long": {"entry": (-0.72, -0.34), "sustain": (-0.64, -0.28), "release": (-0.24, -0.12), "recovery": (-0.06, 0.02)},
        "dense_short": {"entry": (-0.62, -0.28), "sustain": (-0.54, -0.22), "release": (-0.2, -0.1), "recovery": (-0.04, 0.04)},
        "open_long": {"entry": (-0.34, -0.16), "sustain": (-0.28, -0.12), "release": (-0.1, -0.04), "recovery": (0.0, 0.08)},
        "open_short": {"entry": (-0.28, -0.12), "sustain": (-0.18, -0.08), "release": (-0.06, -0.02), "recovery": (0.06, 0.12)},
    }
    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        if block_index is None:
            block_index = bar_to_block.get(bar_index - 1)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        stage_adjustment = (stage_profile_adjustments.get(block_profile) or {}).get(stage_name)
        if not stage_adjustment:
            continue
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        presence_cut_db[start_index:end_index] += float(stage_adjustment[0])
        upper_mid_cut_db[start_index:end_index] += float(stage_adjustment[1])
        contextual_events.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_profile": block_profile or None,
                "presence_cut_db": round(float(stage_adjustment[0]), 4),
                "upper_mid_cut_db": round(float(stage_adjustment[1]), 4),
            }
        )

    hotspot_events_applied: List[Dict[str, Any]] = []
    for event in list((hotspot_manifest or {}).get("harshness_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        extra_presence_cut = -0.42
        extra_upper_mid_cut = -0.18
        presence_cut_db[start_index:end_index] += extra_presence_cut
        upper_mid_cut_db[start_index:end_index] += extra_upper_mid_cut
        hotspot_events_applied.append(
            {
                "type": "harshness_presence_space",
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "extra_presence_cut_db": round(float(extra_presence_cut), 4),
                "extra_upper_mid_cut_db": round(float(extra_upper_mid_cut), 4),
                "block_profile": event.get("block_profile"),
                "stage": event.get("stage"),
            }
        )

    smoothing_samples = max(64, int(round(0.045 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            presence_cut_db = np.convolve(presence_cut_db, kernel, mode="same").astype(np.float32)
            upper_mid_cut_db = np.convolve(upper_mid_cut_db, kernel, mode="same").astype(np.float32)

    presence_cut_db = np.clip(presence_cut_db, -1.6, 0.14)
    upper_mid_cut_db = np.clip(upper_mid_cut_db, -0.8, 0.12)
    presence_gain = np.power(10.0, presence_cut_db / 20.0).astype(np.float32)
    upper_mid_gain = np.power(10.0, upper_mid_cut_db / 20.0).astype(np.float32)
    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)
    presence_low_hz = 2200.0
    presence_high_hz = min(4200.0, nyquist_hz * 0.82)
    upper_mid_low_hz = 3500.0
    upper_mid_high_hz = min(6200.0, nyquist_hz * 0.9)
    if upper_mid_high_hz <= upper_mid_low_hz:
        upper_mid_low_hz = 3000.0
        upper_mid_high_hz = min(5600.0, nyquist_hz * 0.84)
    presence_sos = signal.butter(2, [presence_low_hz, presence_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    upper_mid_sos = signal.butter(2, [upper_mid_low_hz, upper_mid_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        presence_band = signal.sosfiltfilt(presence_sos, channel).astype(np.float32)
        upper_mid_band = signal.sosfiltfilt(upper_mid_sos, channel).astype(np.float32)
        processed = channel + (presence_band * (presence_gain - 1.0)) + (upper_mid_band * (upper_mid_gain - 1.0))
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "contextual_presence_space",
        "presence_band_hz": [round(float(presence_low_hz), 2), round(float(presence_high_hz), 2)],
        "upper_mid_band_hz": [round(float(upper_mid_low_hz), 2), round(float(upper_mid_high_hz), 2)],
        "contextual_events": contextual_events[:240],
        "contextual_events_count": len(contextual_events),
        "hotspot_events": hotspot_events_applied[:120],
        "hotspot_events_count": len(hotspot_events_applied),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_phrase_energy_automation(
    instrumental_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    macrosections: Optional[Dict[str, Any]] = None,
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    total_samples = stereo.shape[0]
    stages = dict(phrase_blocks.get("bar_stages") or {})
    blocks = list(phrase_blocks.get("blocks") or [])
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    sections = list((macrosections or {}).get("sections") or [])
    bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}
    stage_gain_db = {
        "entry": -0.65,
        "sustain": -0.3,
        "release": 0.22,
        "recovery": 0.75,
    }
    profile_stage_adjustments = {
        "dense_long": {"entry": -0.22, "sustain": -0.16, "release": 0.06, "recovery": 0.12},
        "dense_short": {"entry": -0.14, "sustain": -0.08, "release": 0.04, "recovery": 0.08},
        "open_long": {"entry": 0.0, "sustain": 0.06, "release": 0.12, "recovery": 0.16},
        "open_short": {"entry": 0.06, "sustain": 0.1, "release": 0.14, "recovery": 0.2},
    }
    section_gain_adjustments = {
        "intro": -0.24,
        "verse": -0.04,
        "pre_hook": 0.06,
        "hook": 0.24,
        "break": -0.32,
        "transition": -0.1,
        "bridge": 0.02,
        "outro": -0.12,
    }

    envelope = np.ones(total_samples, dtype=np.float32)
    applied_stage_gains: List[Dict[str, Any]] = []
    macrosection_events: List[Dict[str, Any]] = []
    for bar_key, stage_name in stages.items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_index = bar_to_block.get(bar_index)
        block_profile = ""
        if block_index is not None and 0 <= block_index < len(blocks):
            block_profile = _clean_text(blocks[block_index].get("profile")).lower()
        gain_db = float(stage_gain_db.get(stage_name, 0.0))
        gain_db += float((profile_stage_adjustments.get(block_profile) or {}).get(stage_name, 0.0))
        section_index = bar_to_section.get(bar_index)
        section_type = ""
        if section_index is not None and 0 <= section_index < len(sections):
            section_type = _clean_text(sections[section_index].get("section_type")).lower()
            section_gain_db = float(section_gain_adjustments.get(section_type, 0.0))
            gain_db += section_gain_db
            if section_gain_db != 0.0:
                macrosection_events.append(
                    {
                        "bar_index": int(bar_index),
                        "section_type": section_type or None,
                        "gain_db": round(float(section_gain_db), 4),
                    }
                )
        if gain_db == 0.0:
            continue
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        envelope[start_index:end_index] *= float(10 ** (gain_db / 20.0))
        applied_stage_gains.append(
            {
                "bar_index": int(bar_index),
                "stage": str(stage_name),
                "block_index": int(block_index) if block_index is not None else None,
                "block_profile": block_profile or None,
                "gain_db": round(gain_db, 4),
            }
        )

    transition_events: List[Dict[str, Any]] = []

    def _multiply_envelope_window(
        start_index: int,
        end_index: int,
        start_gain: float,
        end_gain: float,
        event_type: str,
        block_index: int,
    ) -> None:
        start = max(0, int(start_index))
        end = min(total_samples, int(end_index))
        if end <= start:
            return
        ramp = np.linspace(float(start_gain), float(end_gain), end - start, dtype=np.float32)
        envelope[start:end] *= ramp
        transition_events.append(
            {
                "type": event_type,
                "block_index": int(block_index),
                "start_sample": int(start),
                "end_sample": int(end),
                "start_gain": round(float(start_gain), 4),
                "end_gain": round(float(end_gain), 4),
            }
        )

    for block_index, block in enumerate(blocks):
        start_bar = int(block.get("start_bar") or 0)
        end_bar = int(block.get("end_bar") or start_bar)
        block_profile = _clean_text(block.get("profile")).lower()
        block_start = start_bar * samples_per_bar
        block_end = min(total_samples, (end_bar + 1) * samples_per_bar)
        pre_roll = max(64, int(round(samples_per_bar * 0.16)))
        attack = max(64, int(round(samples_per_bar * 0.22)))
        release = max(64, int(round(samples_per_bar * 0.24)))
        recovery = max(64, int(round(samples_per_bar * 0.32)))
        if block_profile == "dense_long":
            pre_entry_end = 0.92
            entry_start = 0.88
            release_end = 1.03
            recovery_start_gain = 1.05
        elif block_profile == "dense_short":
            pre_entry_end = 0.93
            entry_start = 0.9
            release_end = 1.04
            recovery_start_gain = 1.06
        elif block_profile == "open_long":
            pre_entry_end = 0.95
            entry_start = 0.93
            release_end = 1.06
            recovery_start_gain = 1.09
        else:
            pre_entry_end = 0.96
            entry_start = 0.94
            release_end = 1.08
            recovery_start_gain = 1.11

        _multiply_envelope_window(
            start_index=block_start - pre_roll,
            end_index=block_start,
            start_gain=1.0,
            end_gain=pre_entry_end,
            event_type="pre_entry",
            block_index=block_index,
        )
        _multiply_envelope_window(
            start_index=block_start,
            end_index=block_start + attack,
            start_gain=entry_start,
            end_gain=1.0,
            event_type="entry_attack",
            block_index=block_index,
        )
        _multiply_envelope_window(
            start_index=block_end - release,
            end_index=block_end,
            start_gain=1.0,
            end_gain=release_end,
            event_type="release_lift",
            block_index=block_index,
        )
        recovery_start_index = block_end
        _multiply_envelope_window(
            start_index=recovery_start_index,
            end_index=recovery_start_index + recovery,
            start_gain=recovery_start_gain,
            end_gain=1.0,
            event_type="recovery_return",
            block_index=block_index,
        )

    line_energy_events: List[Dict[str, Any]] = []
    line_profile_energy_map = {
        ("dense_lyric", "verse"): (0.94, 0.92, 1.01),
        ("dense_lyric", "hook"): (0.97, 0.96, 1.03),
        ("dense_lyric", "break"): (0.95, 0.94, 0.995),
        ("emphatic_lyric", "hook"): (0.98, 0.99, 1.045),
        ("emphatic_lyric", "transition"): (0.97, 0.98, 1.03),
        ("standard_lyric", "hook"): (0.985, 0.985, 1.018),
        ("sparse_lyric", "hook"): (0.99, 0.99, 1.02),
        ("adlib", "hook"): (0.995, 0.998, 1.012),
        ("non_lyric_event", "verse"): (1.004, 1.002, 1.0),
        ("non_lyric_event", "break"): (1.0, 0.998, 0.996),
    }
    default_line_profile_energy = {
        "dense_lyric": (0.955, 0.95, 1.02),
        "emphatic_lyric": (0.972, 0.975, 1.03),
        "standard_lyric": (0.988, 0.988, 1.012),
        "sparse_lyric": (0.992, 0.994, 1.014),
        "adlib": (0.996, 0.998, 1.008),
        "non_lyric_event": (1.0, 1.0, 1.0),
    }
    for event in list((transcript_line_manifest or {}).get("line_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        line_profile = _clean_text(event.get("line_profile")).lower()
        section_type = _clean_text(event.get("section_type")).lower() or _resolve_macrosection_type_for_bar(
            int(start_index // samples_per_bar),
            macrosections,
        )
        word_count = int(event.get("word_count") or 0)
        density = float(event.get("density") or 0.0)
        pre_entry_end, body_gain, recovery_gain = line_profile_energy_map.get(
            (line_profile, section_type),
            default_line_profile_energy.get(line_profile, (1.0, 1.0, 1.0)),
        )
        if line_profile == "dense_lyric" and density >= 4.6:
            pre_entry_end -= 0.01
            body_gain -= 0.012
            recovery_gain += 0.008
        if line_profile in {"dense_lyric", "emphatic_lyric"} and word_count >= 10:
            recovery_gain += 0.008
        pre_roll = max(32, int(round(samples_per_bar * 0.06)))
        settle = max(32, int(round(samples_per_bar * 0.08)))
        recovery = max(32, int(round(samples_per_bar * 0.16)))
        _multiply_envelope_window(
            start_index=start_index - pre_roll,
            end_index=start_index,
            start_gain=1.0,
            end_gain=float(pre_entry_end),
            event_type="line_pre_entry",
            block_index=-1,
        )
        _multiply_envelope_window(
            start_index=start_index,
            end_index=end_index,
            start_gain=float(body_gain),
            end_gain=float(body_gain),
            event_type="line_body_energy",
            block_index=-1,
        )
        _multiply_envelope_window(
            start_index=end_index,
            end_index=min(total_samples, end_index + settle),
            start_gain=1.0,
            end_gain=float(recovery_gain),
            event_type="line_release_lift",
            block_index=-1,
        )
        _multiply_envelope_window(
            start_index=min(total_samples, end_index + settle),
            end_index=min(total_samples, end_index + settle + recovery),
            start_gain=float(recovery_gain),
            end_gain=1.0,
            event_type="line_recovery_return",
            block_index=-1,
        )
        line_energy_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "section_type": section_type or None,
                "line_profile": line_profile or None,
                "pre_entry_end": round(float(pre_entry_end), 4),
                "body_gain": round(float(body_gain), 4),
                "recovery_gain": round(float(recovery_gain), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.11 * effective_sample_rate)))
    if smoothing_samples > 1 and envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.3, 1.25).astype(np.float32)

    automated = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    automated = _normalize_audio_buffer(automated, peak=0.92)
    _write_audio_array_to_wav(automated, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "applied_stage_gains": applied_stage_gains,
        "applied_stage_gains_count": len(applied_stage_gains),
        "macrosection_events": macrosection_events[:240],
        "macrosection_events_count": len(macrosection_events),
        "transition_events": transition_events,
        "transition_events_count": len(transition_events),
        "line_energy_events": line_energy_events[:240],
        "line_energy_events_count": len(line_energy_events),
        "transcript_line_events_count": len(list((transcript_line_manifest or {}).get("line_events") or [])),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_macrosection_groove_automation(
    instrumental_path: str,
    output_path: str,
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    macrosections: Optional[Dict[str, Any]] = None,
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    if total_samples <= 0:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "macrosection_groove_automation_fallback",
            "groove_events": [],
            "groove_events_count": 0,
        }

    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}
    envelope = np.ones(total_samples, dtype=np.float32)
    groove_events: List[Dict[str, Any]] = []
    sections = list((macrosections or {}).get("sections") or [])
    bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}

    section_profiles = {
        "intro": {"pre_end": 0.992, "attack_peak": 1.012, "settle_end": 1.0},
        "verse": {"pre_end": 0.988, "attack_peak": 1.018, "settle_end": 0.998},
        "pre_hook": {"pre_end": 0.98, "attack_peak": 1.03, "settle_end": 1.006},
        "hook": {"pre_end": 0.972, "attack_peak": 1.044, "settle_end": 1.012},
        "break": {"pre_end": 0.996, "attack_peak": 0.986, "settle_end": 0.978},
        "transition": {"pre_end": 0.978, "attack_peak": 1.032, "settle_end": 1.004},
        "bridge": {"pre_end": 0.986, "attack_peak": 1.02, "settle_end": 1.002},
        "outro": {"pre_end": 0.994, "attack_peak": 1.008, "settle_end": 0.996},
    }
    stage_strength = {
        "entry": 1.0,
        "sustain": 0.7,
        "release": 0.5,
        "recovery": 0.85,
    }

    def _apply_ramp(start: int, end: int, start_gain: float, end_gain: float) -> None:
        clamped_start = max(0, int(start))
        clamped_end = min(total_samples, int(end))
        if clamped_end <= clamped_start:
            return
        ramp = np.linspace(float(start_gain), float(end_gain), clamped_end - clamped_start, dtype=np.float32)
        envelope[clamped_start:clamped_end] *= ramp

    estimated_bars = max(1, int(math.ceil(total_samples / float(samples_per_bar))))
    for bar_index in range(estimated_bars):
        section_index = bar_to_section.get(bar_index)
        section_type = ""
        if section_index is not None and 0 <= section_index < len(sections):
            section_type = _clean_text(sections[section_index].get("section_type")).lower()
        profile = section_profiles.get(section_type or "verse", section_profiles["verse"])
        stage_name = _clean_text(stages.get(bar_index)).lower() or "sustain"
        strength = float(stage_strength.get(stage_name, 0.7))

        bar_start = int(bar_index * samples_per_bar)
        bar_end = min(total_samples, bar_start + samples_per_bar)
        if bar_end <= bar_start:
            continue
        pre_roll = max(64, int(round(samples_per_bar * 0.12)))
        attack = max(64, int(round(samples_per_bar * 0.16)))
        settle = max(64, int(round(samples_per_bar * 0.24)))

        pre_end = 1.0 - ((1.0 - float(profile["pre_end"])) * strength)
        attack_peak = 1.0 + ((float(profile["attack_peak"]) - 1.0) * strength)
        settle_end = 1.0 + ((float(profile["settle_end"]) - 1.0) * strength)

        _apply_ramp(bar_start - pre_roll, bar_start, 1.0, pre_end)
        _apply_ramp(bar_start, bar_start + attack, attack_peak, 1.0)
        _apply_ramp(bar_start + attack, min(bar_end, bar_start + attack + settle), 1.0, settle_end)

        groove_events.append(
            {
                "bar_index": int(bar_index),
                "section_type": section_type or None,
                "stage": stage_name or None,
                "pre_end": round(float(pre_end), 4),
                "attack_peak": round(float(attack_peak), 4),
                "settle_end": round(float(settle_end), 4),
            }
        )

    line_groove_events: List[Dict[str, Any]] = []
    line_profile_groove_map = {
        ("dense_lyric", "verse"): (0.986, 1.008, 0.994, 0.994, 1.012, 0.998),
        ("dense_lyric", "hook"): (0.992, 1.016, 1.002, 0.988, 1.026, 1.006),
        ("dense_lyric", "break"): (0.994, 0.996, 0.988, 0.992, 1.004, 0.994),
        ("emphatic_lyric", "hook"): (0.988, 1.024, 1.006, 0.984, 1.034, 1.01),
        ("emphatic_lyric", "transition"): (0.99, 1.018, 1.004, 0.986, 1.028, 1.008),
        ("standard_lyric", "hook"): (0.994, 1.012, 1.002, 0.992, 1.02, 1.004),
        ("adlib", "hook"): (0.997, 1.01, 1.004, 0.995, 1.014, 1.006),
        ("non_lyric_event", "verse"): (0.999, 1.0, 0.999, 0.998, 1.004, 1.0),
        ("non_lyric_event", "break"): (1.0, 0.998, 0.996, 0.998, 1.0, 0.998),
    }
    default_line_profile_groove = {
        "dense_lyric": (0.99, 1.012, 0.998, 0.988, 1.02, 1.004),
        "emphatic_lyric": (0.992, 1.016, 1.002, 0.988, 1.026, 1.006),
        "standard_lyric": (0.996, 1.006, 1.0, 0.994, 1.014, 1.002),
        "sparse_lyric": (0.998, 1.004, 1.002, 0.996, 1.008, 1.004),
        "adlib": (0.999, 1.004, 1.002, 0.997, 1.008, 1.004),
        "non_lyric_event": (1.0, 1.0, 1.0, 0.998, 1.002, 1.0),
    }
    for event in list((transcript_line_manifest or {}).get("line_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        line_profile = _clean_text(event.get("line_profile")).lower()
        section_type = _clean_text(event.get("section_type")).lower() or _resolve_macrosection_type_for_bar(
            int(start_index // samples_per_bar),
            macrosections,
        )
        body_pre_end, body_attack_peak, body_settle_end, release_pre_end, release_attack_peak, release_settle_end = line_profile_groove_map.get(
            (line_profile, section_type),
            default_line_profile_groove.get(line_profile, (1.0, 1.0, 1.0, 0.998, 1.006, 1.001)),
        )
        density = float(event.get("density") or 0.0)
        if line_profile == "dense_lyric" and density >= 4.6:
            body_pre_end -= 0.004
            body_attack_peak -= 0.004
            release_attack_peak += 0.004
        if line_profile == "emphatic_lyric":
            release_attack_peak += 0.004
            release_settle_end += 0.002

        pre_roll = max(64, int(round(samples_per_bar * 0.05)))
        attack = max(64, int(round(samples_per_bar * 0.08)))
        settle = max(64, int(round(samples_per_bar * 0.12)))
        release = max(64, int(round(samples_per_bar * 0.1)))
        release_settle = max(64, int(round(samples_per_bar * 0.16)))

        _apply_ramp(start_index - pre_roll, start_index, 1.0, body_pre_end)
        _apply_ramp(start_index, min(end_index, start_index + attack), body_attack_peak, 1.0)
        _apply_ramp(start_index + attack, min(end_index, start_index + attack + settle), 1.0, body_settle_end)
        _apply_ramp(end_index - release, end_index, 1.0, release_pre_end)
        _apply_ramp(end_index, min(total_samples, end_index + attack), release_attack_peak, 1.0)
        _apply_ramp(end_index + attack, min(total_samples, end_index + attack + release_settle), 1.0, release_settle_end)

        line_groove_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "section_type": section_type or None,
                "line_profile": line_profile or None,
                "body_pre_end": round(float(body_pre_end), 4),
                "body_attack_peak": round(float(body_attack_peak), 4),
                "body_settle_end": round(float(body_settle_end), 4),
                "release_pre_end": round(float(release_pre_end), 4),
                "release_attack_peak": round(float(release_attack_peak), 4),
                "release_settle_end": round(float(release_settle_end), 4),
            }
        )

    smoothing_samples = max(64, int(round(0.035 * effective_sample_rate)))
    if envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.clip(envelope, 0.9, 1.07).astype(np.float32)

    processed = np.clip(stereo * envelope[:, np.newaxis], -1.0, 1.0)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "macrosection_groove_automation",
        "groove_events": groove_events[:240],
        "groove_events_count": len(groove_events),
        "line_groove_events": line_groove_events[:240],
        "line_groove_events_count": len(line_groove_events),
        "transcript_line_events_count": len(list((transcript_line_manifest or {}).get("line_events") or [])),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_macrosection_harmonic_support(
    instrumental_path: str,
    output_path: str,
    vocal_key: Optional[Dict[str, Any]],
    phrase_blocks: Dict[str, Any],
    tempo_bpm: float,
    sample_rate: int,
    macrosections: Optional[Dict[str, Any]] = None,
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instrumental_samples, resolved_sample_rate = _load_audio_array(
        instrumental_path,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
    stereo = _ensure_stereo(instrumental_samples)
    effective_sample_rate = resolved_sample_rate or sample_rate
    total_samples = stereo.shape[0]
    root_pitch_class = _key_name_to_pitch_class(_clean_text((vocal_key or {}).get("key")))
    if total_samples <= 0 or signal is None or root_pitch_class is None:
        _write_audio_array_to_wav(stereo, output_path, sample_rate=effective_sample_rate)
        return {
            "output_path": output_path,
            "mode": "macrosection_harmonic_support_fallback",
            "root_key": _clean_text((vocal_key or {}).get("key")) or None,
            "root_scale": _clean_text((vocal_key or {}).get("scale")) or None,
            "harmonic_events": [],
            "harmonic_events_count": 0,
        }

    confidence = vocal_key.get("confidence")
    try:
        confidence_scale = float(confidence) if confidence is not None else 0.12
    except Exception:
        confidence_scale = 0.12
    confidence_scale = max(0.12, min(1.0, confidence_scale * 2.0))

    root_midi = 36 + int(root_pitch_class)
    while root_midi < 41:
        root_midi += 12
    while root_midi > 52:
        root_midi -= 12
    fifth_midi = root_midi + 7
    while fifth_midi > 64:
        fifth_midi -= 12
    upper_root_midi = root_midi + 12

    root_frequency_hz = _midi_to_frequency(root_midi)
    fifth_frequency_hz = _midi_to_frequency(fifth_midi)
    upper_root_frequency_hz = _midi_to_frequency(upper_root_midi)

    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * effective_sample_rate)))
    total_bars = max(1, int(math.ceil(total_samples / float(samples_per_bar))))
    stages = {int(key): value for key, value in dict((phrase_blocks or {}).get("bar_stages") or {}).items()}
    sections = list((macrosections or {}).get("sections") or [])
    bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}

    root_gain_db = np.zeros(total_samples, dtype=np.float32)
    fifth_gain_db = np.zeros(total_samples, dtype=np.float32)
    upper_root_gain_db = np.zeros(total_samples, dtype=np.float32)
    harmonic_events: List[Dict[str, Any]] = []

    section_gain_profiles = {
        "intro": (0.16, 0.08, 0.04),
        "verse": (0.28, 0.16, 0.08),
        "pre_hook": (0.36, 0.22, 0.12),
        "hook": (0.52, 0.34, 0.18),
        "break": (-0.08, -0.04, -0.04),
        "transition": (0.22, 0.14, 0.08),
        "bridge": (0.3, 0.18, 0.1),
        "outro": (0.1, 0.04, 0.02),
    }
    stage_scale_map = {
        "entry": 1.0,
        "sustain": 0.92,
        "release": 0.7,
        "recovery": 0.82,
    }

    for bar_index in range(total_bars):
        section_type = _resolve_macrosection_type_for_bar(bar_index, macrosections)
        gains = section_gain_profiles.get(section_type or "verse", section_gain_profiles["verse"])
        stage_name = _clean_text(stages.get(bar_index)).lower() or "sustain"
        scale = float(stage_scale_map.get(stage_name, 0.9)) * confidence_scale
        start_index = max(0, bar_index * samples_per_bar)
        end_index = min(total_samples, start_index + samples_per_bar)
        if end_index <= start_index:
            continue
        root_gain_db[start_index:end_index] += float(gains[0]) * scale
        fifth_gain_db[start_index:end_index] += float(gains[1]) * scale
        upper_root_gain_db[start_index:end_index] += float(gains[2]) * scale
        harmonic_events.append(
            {
                "bar_index": int(bar_index),
                "section_type": section_type or None,
                "stage": stage_name or None,
                "root_gain_db": round(float(gains[0]) * scale, 4),
                "fifth_gain_db": round(float(gains[1]) * scale, 4),
                "upper_root_gain_db": round(float(gains[2]) * scale, 4),
            }
        )

    line_harmonic_events: List[Dict[str, Any]] = []
    line_profile_harmonic_map = {
        ("dense_lyric", "verse"): (-0.12, -0.08, -0.07, 0.02, 0.01, 0.0),
        ("dense_lyric", "hook"): (0.06, 0.03, -0.02, 0.08, 0.05, 0.03),
        ("dense_lyric", "break"): (-0.05, -0.03, -0.04, 0.03, 0.02, 0.01),
        ("emphatic_lyric", "hook"): (0.08, 0.04, 0.02, 0.11, 0.06, 0.04),
        ("emphatic_lyric", "transition"): (0.05, 0.03, 0.01, 0.09, 0.05, 0.03),
        ("standard_lyric", "hook"): (0.02, 0.01, 0.0, 0.04, 0.02, 0.01),
        ("sparse_lyric", "hook"): (0.03, 0.02, 0.01, 0.05, 0.03, 0.02),
        ("adlib", "hook"): (-0.02, -0.01, 0.0, 0.04, 0.02, 0.02),
        ("non_lyric_event", "verse"): (-0.08, -0.04, -0.03, 0.0, 0.0, 0.0),
        ("non_lyric_event", "break"): (-0.1, -0.06, -0.04, -0.01, 0.0, 0.0),
    }
    default_line_profile_harmonics = {
        "dense_lyric": (-0.08, -0.05, -0.04, 0.05, 0.03, 0.02),
        "emphatic_lyric": (0.04, 0.02, 0.01, 0.06, 0.03, 0.02),
        "standard_lyric": (0.0, 0.0, 0.0, 0.02, 0.01, 0.0),
        "sparse_lyric": (0.01, 0.0, 0.0, 0.03, 0.02, 0.01),
        "adlib": (-0.02, -0.01, 0.0, 0.02, 0.01, 0.01),
        "non_lyric_event": (-0.06, -0.03, -0.02, 0.0, 0.0, 0.0),
    }
    for event in list((transcript_line_manifest or {}).get("line_events") or []):
        start_index = max(0, int(event.get("start_sample") or 0))
        end_index = min(total_samples, int(event.get("end_sample") or 0))
        if end_index <= start_index:
            continue
        line_profile = _clean_text(event.get("line_profile")).lower()
        section_type = _clean_text(event.get("section_type")).lower() or _resolve_macrosection_type_for_bar(
            int(start_index // samples_per_bar),
            macrosections,
        )
        density = float(event.get("density") or 0.0)
        word_count = int(event.get("word_count") or 0)
        profile_gains = line_profile_harmonic_map.get(
            (line_profile, section_type),
            default_line_profile_harmonics.get(line_profile, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        body_root, body_fifth, body_upper, recovery_root, recovery_fifth, recovery_upper = [float(value) for value in profile_gains]
        if density >= 4.6 and line_profile == "dense_lyric":
            body_root -= 0.04
            body_fifth -= 0.03
            body_upper -= 0.03
        if word_count >= 10 and line_profile in {"dense_lyric", "emphatic_lyric"}:
            recovery_root += 0.02
            recovery_fifth += 0.01
        pre_roll = max(32, int(round(0.02 * effective_sample_rate)))
        recovery = max(32, int(round(0.08 * effective_sample_rate)))
        body_start = max(0, start_index - pre_roll)
        body_end = end_index
        root_gain_db[body_start:body_end] += body_root
        fifth_gain_db[body_start:body_end] += body_fifth
        upper_root_gain_db[body_start:body_end] += body_upper
        recovery_end = min(total_samples, end_index + recovery)
        if recovery_end > end_index:
            root_gain_db[end_index:recovery_end] += np.linspace(body_root, recovery_root, recovery_end - end_index, dtype=np.float32)
            fifth_gain_db[end_index:recovery_end] += np.linspace(body_fifth, recovery_fifth, recovery_end - end_index, dtype=np.float32)
            upper_root_gain_db[end_index:recovery_end] += np.linspace(body_upper, recovery_upper, recovery_end - end_index, dtype=np.float32)
        line_harmonic_events.append(
            {
                "start_sample": int(start_index),
                "end_sample": int(end_index),
                "section_type": section_type or None,
                "line_profile": line_profile or None,
                "body_root_gain_db": round(body_root, 4),
                "body_fifth_gain_db": round(body_fifth, 4),
                "body_upper_root_gain_db": round(body_upper, 4),
                "recovery_root_gain_db": round(recovery_root, 4),
                "recovery_fifth_gain_db": round(recovery_fifth, 4),
                "recovery_upper_root_gain_db": round(recovery_upper, 4),
            }
        )

    smoothing_samples = max(64, int(round(0.08 * effective_sample_rate)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            root_gain_db = np.convolve(root_gain_db, kernel, mode="same").astype(np.float32)
            fifth_gain_db = np.convolve(fifth_gain_db, kernel, mode="same").astype(np.float32)
            upper_root_gain_db = np.convolve(upper_root_gain_db, kernel, mode="same").astype(np.float32)

    nyquist_hz = max(100.0, float(effective_sample_rate) / 2.0)

    def _band(low_hz: float, high_hz: float) -> Tuple[float, float]:
        band_low = max(40.0, min(float(low_hz), nyquist_hz * 0.88))
        band_high = max(band_low + 18.0, min(float(high_hz), nyquist_hz * 0.96))
        return band_low, band_high

    root_low_hz, root_high_hz = _band(root_frequency_hz * 0.82, root_frequency_hz * 1.22)
    fifth_low_hz, fifth_high_hz = _band(fifth_frequency_hz * 0.84, fifth_frequency_hz * 1.2)
    upper_root_low_hz, upper_root_high_hz = _band(upper_root_frequency_hz * 0.86, upper_root_frequency_hz * 1.16)

    root_sos = signal.butter(2, [root_low_hz, root_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    fifth_sos = signal.butter(2, [fifth_low_hz, fifth_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")
    upper_root_sos = signal.butter(2, [upper_root_low_hz, upper_root_high_hz], btype="bandpass", fs=effective_sample_rate, output="sos")

    root_gain = np.power(10.0, np.clip(root_gain_db, -0.4, 0.9) / 20.0).astype(np.float32) - 1.0
    fifth_gain = np.power(10.0, np.clip(fifth_gain_db, -0.3, 0.65) / 20.0).astype(np.float32) - 1.0
    upper_root_gain = np.power(10.0, np.clip(upper_root_gain_db, -0.2, 0.42) / 20.0).astype(np.float32) - 1.0

    processed_channels: List[np.ndarray] = []
    for channel_index in range(stereo.shape[1]):
        channel = stereo[:, channel_index].astype(np.float32)
        root_band = signal.sosfiltfilt(root_sos, channel).astype(np.float32)
        fifth_band = signal.sosfiltfilt(fifth_sos, channel).astype(np.float32)
        upper_root_band = signal.sosfiltfilt(upper_root_sos, channel).astype(np.float32)
        processed = channel + (root_band * root_gain) + (fifth_band * fifth_gain) + (upper_root_band * upper_root_gain)
        processed_channels.append(processed.astype(np.float32))

    processed = np.stack(processed_channels, axis=1)
    processed = _normalize_audio_buffer(processed, peak=0.92)
    _write_audio_array_to_wav(processed, output_path, sample_rate=effective_sample_rate)
    return {
        "output_path": output_path,
        "mode": "macrosection_harmonic_support",
        "root_key": _clean_text((vocal_key or {}).get("key")) or None,
        "root_scale": _clean_text((vocal_key or {}).get("scale")) or None,
        "root_band_hz": [round(float(root_low_hz), 2), round(float(root_high_hz), 2)],
        "fifth_band_hz": [round(float(fifth_low_hz), 2), round(float(fifth_high_hz), 2)],
        "upper_root_band_hz": [round(float(upper_root_low_hz), 2), round(float(upper_root_high_hz), 2)],
        "harmonic_events": harmonic_events[:240],
        "harmonic_events_count": len(harmonic_events),
        "line_harmonic_events": line_harmonic_events[:240],
        "line_harmonic_events_count": len(line_harmonic_events),
        "transcript_line_events_count": len(list((transcript_line_manifest or {}).get("line_events") or [])),
        "smoothing_samples": int(smoothing_samples),
    }


def _render_vocal_aware_arrangement_mix(
    pattern_manifest: Dict[str, Any],
    bar_overrides: Dict[int, Dict[str, Any]],
    step_overrides: Optional[Dict[int, Dict[str, Any]]],
    phrase_blocks: Optional[Dict[str, Any]],
    macrosections: Optional[Dict[str, Any]],
    output_path: str,
    sample_rate: int,
    bars: int,
    steps_per_bar: int,
    tempo_bpm: float,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    seconds_per_bar = 240.0 / float(tempo_bpm)
    seconds_per_step = seconds_per_bar / float(max(1, steps_per_bar))
    total_samples = max(64, int(round(max(1, int(bars)) * seconds_per_bar * sample_rate)))
    mix_buffer = np.zeros((total_samples, 2), dtype=np.float32)
    applied_overrides: List[Dict[str, Any]] = []
    role_source_cache: Dict[str, np.ndarray] = {}
    blocks = list((phrase_blocks or {}).get("blocks") or [])
    bar_to_block = {int(key): int(value) for key, value in dict((phrase_blocks or {}).get("bar_to_block") or {}).items()}
    sections = list((macrosections or {}).get("sections") or [])
    bar_to_section = {int(key): int(value) for key, value in dict((macrosections or {}).get("bar_section_map") or {}).items()}

    def _resolve_block_profile(bar_index: int) -> str:
        block_index = bar_to_block.get(int(bar_index))
        if block_index is None:
            block_index = bar_to_block.get(int(bar_index) - 1)
        if block_index is None or block_index < 0 or block_index >= len(blocks):
            return ""
        return _clean_text(blocks[block_index].get("profile")).lower()

    def _resolve_section_type(bar_index: int) -> str:
        section_index = bar_to_section.get(int(bar_index))
        if section_index is None:
            section_index = bar_to_section.get(int(bar_index) - 1)
        if section_index is None or section_index < 0 or section_index >= len(sections):
            return ""
        return _clean_text(sections[section_index].get("section_type")).lower()

    def _merge_override_dicts(
        base_override: Optional[Dict[str, Any]],
        detail_override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "mute_roles": [],
            "duck_gain_db": 0.0,
            "role_gain_db": {},
        }
        for candidate in (base_override or {}, detail_override or {}):
            merged["mute_roles"].extend(
                [_clean_text(item).lower() for item in candidate.get("mute_roles") or [] if _clean_text(item)]
            )
            merged["duck_gain_db"] += float(candidate.get("duck_gain_db") or 0.0)
            role_gain_db = dict(candidate.get("role_gain_db") or {})
            for role_name, gain_db in role_gain_db.items():
                normalized_role = _clean_text(role_name).lower()
                merged["role_gain_db"][normalized_role] = float(merged["role_gain_db"].get(normalized_role, 0.0)) + float(gain_db)
        merged["mute_roles"] = sorted(set(merged["mute_roles"]))
        return merged

    for layer in pattern_manifest.get("rendered_layers") or []:
        source_path = _resolve_existing_path(layer["source_path"])
        sample_audio, _ = _load_audio_array(source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
        sample_audio = _ensure_stereo(sample_audio)
        role = _clean_text(layer.get("role")).lower()
        role_source_cache.setdefault(role, sample_audio)

        for trigger in layer.get("trigger_positions") or []:
            if isinstance(trigger, int):
                start_seconds = float(trigger) * (seconds_per_bar / float(steps_per_bar))
                step_index = int(trigger)
                gain_offset_db = 0.0
            else:
                start_seconds = float(trigger.get("start_seconds") or 0.0)
                step_index = int(trigger.get("step_index") or 0)
                gain_offset_db = float(trigger.get("gain_offset_db") or 0.0)

            bar_index = min(max(0, int(step_index // max(1, steps_per_bar))), max(0, bars - 1))
            override = _merge_override_dicts(
                bar_overrides.get(bar_index),
                (step_overrides or {}).get(step_index),
            )
            if override and role in {item.lower() for item in override.get("mute_roles") or []}:
                applied_overrides.append(
                    {"bar_index": bar_index, "step_index": step_index, "role": role, "action": "muted"}
                )
                continue

            role_gain_db = float(((override or {}).get("role_gain_db") or {}).get(role, 0.0))
            extra_gain_db = float((override or {}).get("duck_gain_db") or 0.0) + role_gain_db
            gain_linear = float(10 ** ((gain_offset_db + extra_gain_db) / 20.0))
            triggered = np.clip(sample_audio * gain_linear, -1.0, 1.0)
            start_index = max(0, int(round(start_seconds * sample_rate)))
            end_index = min(mix_buffer.shape[0], start_index + triggered.shape[0])
            if end_index <= start_index:
                continue
            mix_buffer[start_index:end_index] += triggered[: end_index - start_index]
            if override and extra_gain_db != 0.0:
                applied_overrides.append(
                    {
                        "bar_index": bar_index,
                        "step_index": step_index,
                        "role": role,
                        "action": "role_gain" if role_gain_db != 0.0 else "gain",
                        "gain_db": round(extra_gain_db, 4),
                    }
                )

    transition_injections: List[Dict[str, Any]] = []

    def _inject_role_accent(
        role_name: str,
        start_seconds: float,
        gain_db: float,
        event_type: str,
        bar_index: int,
        step_index: int,
    ) -> None:
        sample_audio = role_source_cache.get(role_name)
        if sample_audio is None:
            return
        start_index = max(0, int(round(start_seconds * sample_rate)))
        end_index = min(mix_buffer.shape[0], start_index + sample_audio.shape[0])
        if end_index <= start_index:
            return
        gain_linear = float(10 ** (float(gain_db) / 20.0))
        triggered = np.clip(sample_audio * gain_linear, -1.0, 1.0)
        mix_buffer[start_index:end_index] += triggered[: end_index - start_index]
        transition_injections.append(
            {
                "type": event_type,
                "bar_index": int(bar_index),
                "step_index": int(step_index),
                "role": role_name,
                "gain_db": round(float(gain_db), 4),
                "block_profile": _resolve_block_profile(bar_index) or None,
            }
        )

    for bar_key, stage_name in dict((phrase_blocks or {}).get("bar_stages") or {}).items():
        try:
            bar_index = int(bar_key)
        except Exception:
            continue
        block_profile = _resolve_block_profile(bar_index)
        if stage_name == "release":
            accent_step = max(0, steps_per_bar - max(1, steps_per_bar // 4))
            accent_time = (bar_index * seconds_per_bar) + (accent_step * seconds_per_step)
            section_type = _resolve_section_type(bar_index)
            snare_gain = -1.35 if block_profile == "dense_long" else -1.2 if block_profile == "dense_short" else -0.85
            clap_gain = -1.8 if block_profile == "dense_long" else -1.65 if block_profile == "dense_short" else -1.15
            chop_gain = -4.2 if block_profile == "dense_long" else -3.8 if block_profile == "dense_short" else -2.6
            if section_type == "hook":
                snare_gain += 0.22
                clap_gain += 0.2
                chop_gain += 0.35
            elif section_type == "break":
                snare_gain -= 0.28
                clap_gain -= 0.22
                chop_gain -= 0.4
            _inject_role_accent("snare", accent_time, snare_gain, "release_accent", bar_index, accent_step)
            _inject_role_accent("clap", accent_time, clap_gain, "release_accent", bar_index, accent_step)
            _inject_role_accent("chop", accent_time, chop_gain, "release_texture", bar_index, accent_step)
            if block_profile in {"open_long", "open_short"}:
                _inject_role_accent(
                    "hat",
                    accent_time - seconds_per_step,
                    -5.4 if block_profile == "open_long" else -4.9,
                    "release_hat_pickup",
                    bar_index,
                    max(0, accent_step - 1),
                )
                _inject_role_accent(
                    "melody",
                    accent_time - seconds_per_step,
                    -4.6 if block_profile == "open_long" else -4.0,
                    "release_melody_pickup",
                    bar_index,
                    max(0, accent_step - 1),
                )
        elif stage_name == "recovery":
            accent_step = 0
            accent_time = bar_index * seconds_per_bar
            section_type = _resolve_section_type(bar_index)
            kick_gain = -0.95 if block_profile == "dense_long" else -0.82 if block_profile == "dense_short" else -0.45
            snare_gain = -1.6 if block_profile == "dense_long" else -1.45 if block_profile == "dense_short" else -1.05
            hat_gain = -5.8 if block_profile == "dense_long" else -5.2 if block_profile == "dense_short" else -4.3
            chop_gain = -4.9 if block_profile == "dense_long" else -4.5 if block_profile == "dense_short" else -3.35
            if section_type == "hook":
                kick_gain += 0.28
                snare_gain += 0.24
                hat_gain += 0.4
                chop_gain += 0.5
            elif section_type == "break":
                kick_gain -= 0.22
                snare_gain -= 0.18
                hat_gain -= 0.65
                chop_gain -= 0.5
            _inject_role_accent("kick", accent_time, kick_gain, "recovery_downbeat", bar_index, accent_step)
            _inject_role_accent(
                "snare",
                accent_time + (seconds_per_step * max(1, steps_per_bar // 2)),
                snare_gain,
                "recovery_backbeat",
                bar_index,
                max(1, steps_per_bar // 2),
            )
            _inject_role_accent("hat", accent_time + seconds_per_step, hat_gain, "recovery_hat_pickup", bar_index, 1)
            _inject_role_accent(
                "chop",
                accent_time + (seconds_per_step * max(1, steps_per_bar // 4)),
                chop_gain,
                "recovery_texture",
                bar_index,
                max(1, steps_per_bar // 4),
            )
            if block_profile in {"open_long", "open_short"}:
                _inject_role_accent(
                    "melody",
                    accent_time,
                    -4.4 if block_profile == "open_long" else -3.8,
                    "recovery_melody_return",
                    bar_index,
                    accent_step,
                )
                _inject_role_accent(
                    "texture",
                    accent_time + (seconds_per_step * max(1, steps_per_bar // 4)),
                    -6.8 if block_profile == "open_long" else -6.1,
                    "recovery_texture_swell",
                    bar_index,
                    max(1, steps_per_bar // 4),
                )

    mix_buffer = _normalize_audio_buffer(mix_buffer, peak=0.92)
    _write_audio_array_to_wav(mix_buffer, output_path, sample_rate=sample_rate)
    return {
        "output_path": output_path,
        "applied_overrides": applied_overrides[:500],
        "applied_overrides_count": len(applied_overrides),
        "transition_injections": transition_injections,
        "transition_injections_count": len(transition_injections),
    }


def _derive_vocal_fit_render_profile(vocal_analysis: Dict[str, Any]) -> Dict[str, Any]:
    analysis = vocal_analysis.get("analysis") if "analysis" in vocal_analysis else vocal_analysis
    silence_ratio = float(analysis.get("silence_ratio") or 0.0)
    active_ratio = max(0.0, min(1.0, 1.0 - silence_ratio))
    dense_vocal = active_ratio >= 0.52
    medium_vocal = active_ratio >= 0.36

    return {
        "creativity": 0.34 if dense_vocal else (0.42 if medium_vocal else 0.5),
        "fill_probability": 0.18 if dense_vocal else (0.25 if medium_vocal else 0.34),
        "add_melodic_motif": not dense_vocal,
        "add_reference_texture": True,
        "add_reference_chops": not dense_vocal,
        "timing_jitter_ratio": 0.032 if dense_vocal else 0.04,
        "velocity_jitter_db": 1.3 if dense_vocal else 1.8,
        "active_ratio": round(active_ratio, 4),
    }


def _normalize_instrumental_adaptation_profile(value: Optional[str]) -> str:
    normalized = _clean_text(value).lower()
    if normalized in {"auto", "dry", "spaced", "dense", "hook_support"}:
        return normalized
    return "auto"


def _apply_instrumental_adaptation_profile(
    base_profile: Dict[str, Any],
    adaptation_profile: str,
) -> Dict[str, Any]:
    resolved = dict(base_profile)
    if adaptation_profile == "dry":
        resolved["creativity"] = min(float(resolved.get("creativity", 0.4)), 0.34)
        resolved["fill_probability"] = min(float(resolved.get("fill_probability", 0.25)), 0.18)
        resolved["add_reference_texture"] = False
        resolved["add_reference_chops"] = False
        resolved["add_melodic_motif"] = False
        resolved["timing_jitter_ratio"] = min(float(resolved.get("timing_jitter_ratio", 0.04)), 0.032)
        resolved["velocity_jitter_db"] = min(float(resolved.get("velocity_jitter_db", 1.6)), 1.25)
    elif adaptation_profile == "spaced":
        resolved["creativity"] = max(float(resolved.get("creativity", 0.4)), 0.44)
        resolved["fill_probability"] = min(max(float(resolved.get("fill_probability", 0.25)), 0.22), 0.3)
        resolved["add_reference_texture"] = True
        resolved["add_reference_chops"] = bool(resolved.get("add_reference_chops", True))
        resolved["add_melodic_motif"] = True
        resolved["timing_jitter_ratio"] = max(float(resolved.get("timing_jitter_ratio", 0.04)), 0.038)
        resolved["velocity_jitter_db"] = max(float(resolved.get("velocity_jitter_db", 1.6)), 1.7)
    elif adaptation_profile == "dense":
        resolved["creativity"] = max(float(resolved.get("creativity", 0.4)), 0.46)
        resolved["fill_probability"] = max(float(resolved.get("fill_probability", 0.25)), 0.32)
        resolved["add_reference_texture"] = True
        resolved["add_reference_chops"] = True
        resolved["add_melodic_motif"] = True
        resolved["timing_jitter_ratio"] = max(float(resolved.get("timing_jitter_ratio", 0.04)), 0.04)
        resolved["velocity_jitter_db"] = max(float(resolved.get("velocity_jitter_db", 1.6)), 1.95)
    elif adaptation_profile == "hook_support":
        resolved["creativity"] = max(float(resolved.get("creativity", 0.4)), 0.42)
        resolved["fill_probability"] = max(float(resolved.get("fill_probability", 0.25)), 0.24)
        resolved["add_reference_texture"] = True
        resolved["add_reference_chops"] = True
        resolved["add_melodic_motif"] = True
    resolved["adaptation_profile"] = adaptation_profile
    return resolved


def _compute_bar_activity_levels(
    vocal_path: str,
    sample_rate: int,
    bars: int,
    tempo_bpm: float,
) -> Dict[str, Any]:
    vocal_samples, _ = _load_audio_array(vocal_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    activity = _compute_vocal_activity_envelope(vocal_samples, sample_rate=sample_rate)
    envelope = activity["envelope"]
    samples_per_bar = max(1, int(round((240.0 / float(tempo_bpm)) * sample_rate)))
    levels: List[float] = []
    for bar_index in range(max(1, int(bars))):
        start = bar_index * samples_per_bar
        end = min(envelope.shape[0], start + samples_per_bar)
        if end <= start:
            levels.append(0.0)
            continue
        levels.append(float(np.mean(envelope[start:end])))
    dense_threshold = max(0.24, float(np.percentile(levels, 70))) if levels else 0.24
    medium_threshold = max(0.14, float(np.percentile(levels, 45))) if levels else 0.14
    return {
        "levels": [round(float(level), 4) for level in levels],
        "dense_threshold": round(float(dense_threshold), 4),
        "medium_threshold": round(float(medium_threshold), 4),
    }


def _compute_step_activity_levels(
    vocal_path: str,
    sample_rate: int,
    bars: int,
    tempo_bpm: float,
    steps_per_bar: int,
) -> Dict[str, Any]:
    vocal_samples, _ = _load_audio_array(vocal_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    activity = _compute_vocal_activity_envelope(vocal_samples, sample_rate=sample_rate)
    envelope = activity["envelope"]
    total_steps = max(1, int(bars)) * max(1, int(steps_per_bar))
    samples_per_step = max(1, int(round(((240.0 / float(tempo_bpm)) / float(max(1, int(steps_per_bar)))) * sample_rate)))
    levels: List[float] = []
    for step_index in range(total_steps):
        start = step_index * samples_per_step
        end = min(envelope.shape[0], start + samples_per_step)
        if end <= start:
            levels.append(0.0)
            continue
        levels.append(float(np.mean(envelope[start:end])))

    dense_threshold = max(0.28, float(np.percentile(levels, 76))) if levels else 0.28
    medium_threshold = max(0.18, float(np.percentile(levels, 56))) if levels else 0.18
    return {
        "levels": [round(float(level), 4) for level in levels],
        "dense_threshold": round(float(dense_threshold), 4),
        "medium_threshold": round(float(medium_threshold), 4),
    }


def _detect_vocal_phrase_blocks(
    bar_activity: Dict[str, Any],
    max_gap_bars: int = 1,
    min_phrase_bars: int = 2,
) -> Dict[str, Any]:
    levels = [float(item) for item in list(bar_activity.get("levels") or [])]
    medium_threshold = float(bar_activity.get("medium_threshold") or 0.0)
    active_flags = [level >= medium_threshold for level in levels]
    phrase_bars = [False] * len(active_flags)

    index = 0
    while index < len(active_flags):
        if not active_flags[index]:
            index += 1
            continue
        start = index
        end = index
        gap_count = 0
        cursor = index + 1
        while cursor < len(active_flags):
            if active_flags[cursor]:
                end = cursor
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > max_gap_bars:
                    break
            cursor += 1
        if (end - start + 1) >= max(1, int(min_phrase_bars)):
            for phrase_index in range(start, end + 1):
                phrase_bars[phrase_index] = True
        index = max(cursor, index + 1)

    blocks: List[Dict[str, Any]] = []
    start_index = None
    for bar_index, active in enumerate(phrase_bars + [False]):
        if active and start_index is None:
            start_index = bar_index
        elif not active and start_index is not None:
            end_index = bar_index - 1
            phrase_levels = levels[start_index : end_index + 1]
            blocks.append(
                {
                    "start_bar": int(start_index),
                    "end_bar": int(end_index),
                    "bars": int(end_index - start_index + 1),
                    "mean_level": round(float(sum(phrase_levels) / max(1, len(phrase_levels))), 4),
                    "peak_level": round(float(max(phrase_levels) if phrase_levels else 0.0), 4),
                }
            )
            start_index = None

    dense_threshold = float(bar_activity.get("dense_threshold") or 0.0)
    medium_threshold = float(bar_activity.get("medium_threshold") or 0.0)
    bar_to_block: Dict[int, int] = {}
    for block_index, block in enumerate(blocks):
        mean_level = float(block.get("mean_level") or 0.0)
        block_bars = int(block.get("bars") or 0)
        if mean_level >= dense_threshold and block_bars >= 8:
            profile = "dense_long"
        elif mean_level >= dense_threshold:
            profile = "dense_short"
        elif mean_level >= medium_threshold and block_bars >= 8:
            profile = "open_long"
        else:
            profile = "open_short"
        block["profile"] = profile
        for bar_index in range(int(block["start_bar"]), int(block["end_bar"]) + 1):
            bar_to_block[bar_index] = block_index

    stages: Dict[int, str] = {}
    for block in blocks:
        start_bar = int(block["start_bar"])
        end_bar = int(block["end_bar"])
        for bar_index in range(start_bar, end_bar + 1):
            if bar_index == start_bar:
                stages[bar_index] = "entry"
            elif bar_index == end_bar:
                stages[bar_index] = "release"
            else:
                stages[bar_index] = "sustain"
        recovery_bar = end_bar + 1
        if recovery_bar < len(levels) and recovery_bar not in stages and levels[recovery_bar] < medium_threshold:
            stages[recovery_bar] = "recovery"

    return {
        "blocks": blocks,
        "phrase_bars": [index for index, active in enumerate(phrase_bars) if active],
        "bar_stages": {str(key): value for key, value in stages.items()},
        "bar_to_block": {str(key): int(value) for key, value in bar_to_block.items()},
    }


def _detect_vocal_macrosections(
    phrase_blocks: Dict[str, Any],
    total_bars: int,
    adaptation_profile: str = "auto",
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bars = max(1, int(total_bars))
    blocks = list(phrase_blocks.get("blocks") or [])
    sections: List[Dict[str, Any]] = []
    bar_section_map: Dict[int, int] = {}
    densest_block_index = None
    densest_score = float("-inf")
    block_line_stats: List[Dict[str, Any]] = []
    hook_candidate_index = None
    hook_candidate_score = float("-inf")

    for block in blocks:
        block_line_stats.append(
            {
                "dense_lyric": 0,
                "emphatic_lyric": 0,
                "standard_lyric": 0,
                "sparse_lyric": 0,
                "adlib": 0,
                "non_lyric_event": 0,
                "line_events": 0,
                "lyric_score": 0.0,
            }
        )

    line_events = list((transcript_line_manifest or {}).get("line_events") or [])
    for event in line_events:
        start_bar = int(event.get("start_bar") or 0)
        end_bar = int(event.get("end_bar") or start_bar)
        line_profile = _clean_text(event.get("line_profile")).lower() or "standard_lyric"
        matched_index = None
        for block_index, block in enumerate(blocks):
            block_start = int(block.get("start_bar") or 0)
            block_end = int(block.get("end_bar") or block_start)
            if end_bar < block_start or start_bar > block_end:
                continue
            matched_index = block_index
            break
        if matched_index is None:
            continue
        stats = block_line_stats[matched_index]
        stats["line_events"] = int(stats["line_events"]) + 1
        stats[line_profile] = int(stats.get(line_profile, 0)) + 1
        lyric_score_map = {
            "dense_lyric": 1.1,
            "emphatic_lyric": 1.25,
            "standard_lyric": 0.55,
            "sparse_lyric": 0.25,
            "adlib": 0.08,
            "non_lyric_event": -0.18,
        }
        stats["lyric_score"] = float(stats.get("lyric_score") or 0.0) + float(lyric_score_map.get(line_profile, 0.2))

    for block_index, block in enumerate(blocks):
        bars_in_block = max(1, int(block.get("bars") or 0))
        score = float(block.get("mean_level") or 0.0) + (0.03 * bars_in_block)
        if score > densest_score:
            densest_score = score
            densest_block_index = block_index
        stats = block_line_stats[block_index] if block_index < len(block_line_stats) else {}
        hook_score = (
            score
            + (float(stats.get("lyric_score") or 0.0) / float(bars_in_block))
            + (0.12 if _clean_text(block.get("profile")).lower() == "dense_long" else 0.0)
            + (0.06 if _clean_text(block.get("profile")).lower() == "open_long" else 0.0)
        )
        if hook_score > hook_candidate_score:
            hook_candidate_score = hook_score
            hook_candidate_index = block_index

    def _append_section(start_bar: int, end_bar: int, section_type: str, source_block_index: Optional[int] = None) -> None:
        if end_bar < start_bar:
            return
        section_index = len(sections)
        sections.append(
            {
                "section_index": int(section_index),
                "section_type": section_type,
                "start_bar": int(start_bar),
                "end_bar": int(end_bar),
                "bars": int(end_bar - start_bar + 1),
                "source_block_index": int(source_block_index) if source_block_index is not None else None,
            }
        )
        for bar_index in range(int(start_bar), int(end_bar) + 1):
            bar_section_map[bar_index] = section_index

    previous_end = -1
    active_section_count = 0
    for block_index, block in enumerate(blocks):
        start_bar = int(block.get("start_bar") or 0)
        end_bar = int(block.get("end_bar") or start_bar)
        if start_bar > previous_end + 1:
            gap_start = previous_end + 1
            gap_end = start_bar - 1
            gap_type = "intro" if active_section_count == 0 else ("break" if (gap_end - gap_start + 1) >= 2 else "transition")
            _append_section(gap_start, gap_end, gap_type, None)

        profile_name = _clean_text(block.get("profile")).lower()
        stats = block_line_stats[block_index] if block_index < len(block_line_stats) else {}
        bars_in_block = max(1, int(block.get("bars") or 0))
        line_event_count = int(stats.get("line_events") or 0)
        dense_event_count = int(stats.get("dense_lyric") or 0)
        emphatic_event_count = int(stats.get("emphatic_lyric") or 0)
        lyric_focus = float(stats.get("lyric_score") or 0.0) / float(bars_in_block)
        is_hook_candidate = (
            hook_candidate_index is not None
            and block_index == hook_candidate_index
            and (
                len(blocks) == 1
                or bars_in_block >= 8
                or line_event_count >= 6
                or (dense_event_count + emphatic_event_count) >= 3
                or lyric_focus >= 0.22
            )
        )
        if adaptation_profile == "hook_support" and block_index == densest_block_index:
            section_type = "hook"
        elif is_hook_candidate and profile_name in {"dense_long", "open_long", "dense_short"}:
            section_type = "hook"
        elif adaptation_profile == "dense" and profile_name in {"dense_long", "dense_short"}:
            section_type = "hook"
        elif profile_name == "dense_long":
            section_type = "hook"
        elif block_index == 0 and is_hook_candidate:
            section_type = "hook"
        elif block_index == 0:
            section_type = "verse"
        elif profile_name == "dense_short":
            section_type = "pre_hook" if active_section_count > 0 else "verse"
        elif profile_name == "open_long":
            section_type = "verse" if active_section_count == 0 and not is_hook_candidate else ("hook" if is_hook_candidate else "bridge")
        else:
            section_type = "bridge" if active_section_count > 1 else "verse"

        _append_section(start_bar, end_bar, section_type, block_index)
        previous_end = end_bar
        active_section_count += 1

    if previous_end < bars - 1:
        trailing_start = max(0, previous_end + 1)
        trailing_type = "outro" if active_section_count > 0 else "intro"
        _append_section(trailing_start, bars - 1, trailing_type, None)

    return {
        "sections": sections,
        "bar_section_map": {str(key): int(value) for key, value in bar_section_map.items()},
        "densest_block_index": int(densest_block_index) if densest_block_index is not None else None,
        "hook_candidate_index": int(hook_candidate_index) if hook_candidate_index is not None else None,
    }


def _build_vocal_aware_section_overrides(
    vocal_path: str,
    sample_rate: int,
    bars: int,
    tempo_bpm: float,
    steps_per_bar: int,
    adaptation_profile: str = "auto",
    transcript_line_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    activity = _compute_bar_activity_levels(vocal_path, sample_rate=sample_rate, bars=bars, tempo_bpm=tempo_bpm)
    phrase_blocks = _detect_vocal_phrase_blocks(activity)
    macrosections = _detect_vocal_macrosections(
        phrase_blocks=phrase_blocks,
        total_bars=bars,
        adaptation_profile=_normalize_instrumental_adaptation_profile(adaptation_profile),
        transcript_line_manifest=transcript_line_manifest,
    )
    blocks = list(phrase_blocks.get("blocks") or [])
    bar_to_block = {int(key): int(value) for key, value in dict(phrase_blocks.get("bar_to_block") or {}).items()}
    sections = list(macrosections.get("sections") or [])
    bar_to_section = {int(key): int(value) for key, value in dict(macrosections.get("bar_section_map") or {}).items()}

    def _resolve_block_profile(bar_index: int) -> str:
        block_index = bar_to_block.get(int(bar_index))
        if block_index is None:
            block_index = bar_to_block.get(int(bar_index) - 1)
        if block_index is None or block_index < 0 or block_index >= len(blocks):
            return ""
        return _clean_text(blocks[block_index].get("profile")).lower()

    def _resolve_section_type(bar_index: int) -> str:
        section_index = bar_to_section.get(int(bar_index))
        if section_index is None:
            section_index = bar_to_section.get(int(bar_index) - 1)
        if section_index is None or section_index < 0 or section_index >= len(sections):
            return ""
        return _clean_text(sections[section_index].get("section_type")).lower()

    step_activity = _compute_step_activity_levels(
        vocal_path=vocal_path,
        sample_rate=sample_rate,
        bars=bars,
        tempo_bpm=tempo_bpm,
        steps_per_bar=steps_per_bar,
    )
    dense_bars: List[int] = []
    medium_bars: List[int] = []
    for bar_index, level in enumerate(activity["levels"]):
        if level >= activity["dense_threshold"]:
            dense_bars.append(bar_index)
        elif level >= activity["medium_threshold"]:
            medium_bars.append(bar_index)

    overrides: Dict[int, Dict[str, Any]] = {}
    for bar_index in dense_bars:
        bar_profile = _resolve_block_profile(bar_index)
        if bar_profile == "dense_long":
            overrides[bar_index] = {
                "mute_roles": ["texture", "chop"],
                "duck_gain_db": -1.1,
                "role_gain_db": {
                    "hat": -4.8,
                    "melody": -2.8,
                    "lead": -2.8,
                    "bass": -1.05,
                    "kick": -0.65,
                    "snare": -0.55,
                    "clap": -0.55,
                },
            }
        elif bar_profile == "dense_short":
            overrides[bar_index] = {
                "mute_roles": ["texture", "chop"],
                "duck_gain_db": -1.0,
                "role_gain_db": {
                    "hat": -4.5,
                    "melody": -2.55,
                    "lead": -2.55,
                    "bass": -0.95,
                    "kick": -0.6,
                    "snare": -0.48,
                    "clap": -0.48,
                },
            }
        elif bar_profile == "open_long":
            overrides[bar_index] = {
                "mute_roles": ["texture"],
                "duck_gain_db": -0.52,
                "role_gain_db": {
                    "hat": -2.0,
                    "melody": -1.25,
                    "lead": -1.25,
                    "bass": -0.45,
                    "kick": -0.35,
                    "snare": -0.22,
                    "clap": -0.22,
                },
            }
        elif bar_profile == "open_short":
            overrides[bar_index] = {
                "mute_roles": ["texture"],
                "duck_gain_db": -0.4,
                "role_gain_db": {
                    "hat": -1.55,
                    "melody": -0.95,
                    "lead": -0.95,
                    "bass": -0.3,
                    "kick": -0.22,
                    "snare": -0.12,
                    "clap": -0.12,
                },
            }
        else:
            overrides[bar_index] = {
                "mute_roles": ["texture", "chop"],
                "duck_gain_db": -0.95,
                "role_gain_db": {
                    "hat": -4.2,
                    "melody": -2.4,
                    "lead": -2.4,
                    "bass": -0.9,
                    "kick": -0.55,
                    "snare": -0.45,
                    "clap": -0.45,
                },
            }
    for bar_index in medium_bars:
        bar_profile = _resolve_block_profile(bar_index)
        if bar_profile == "dense_long":
            medium_override = {
                "mute_roles": ["texture", "chop"],
                "duck_gain_db": -0.62,
                "role_gain_db": {
                    "hat": -2.5,
                    "melody": -1.55,
                    "lead": -1.55,
                    "bass": -0.45,
                },
            }
        elif bar_profile == "dense_short":
            medium_override = {
                "mute_roles": ["texture"],
                "duck_gain_db": -0.52,
                "role_gain_db": {
                    "hat": -2.15,
                    "melody": -1.3,
                    "lead": -1.3,
                    "bass": -0.38,
                },
            }
        elif bar_profile == "open_long":
            medium_override = {
                "mute_roles": [],
                "duck_gain_db": -0.18,
                "role_gain_db": {
                    "hat": -1.05,
                    "melody": -0.58,
                    "lead": -0.58,
                    "bass": -0.15,
                },
            }
        elif bar_profile == "open_short":
            medium_override = {
                "mute_roles": [],
                "duck_gain_db": -0.08,
                "role_gain_db": {
                    "hat": -0.55,
                    "melody": -0.3,
                    "lead": -0.3,
                    "bass": -0.08,
                },
            }
        else:
            medium_override = {
                "mute_roles": ["texture"],
                "duck_gain_db": -0.45,
                "role_gain_db": {
                    "hat": -1.9,
                    "melody": -1.15,
                    "lead": -1.15,
                    "bass": -0.35,
                },
            }
        overrides.setdefault(
            bar_index,
            medium_override,
        )

    for bar_key, stage in (phrase_blocks.get("bar_stages") or {}).items():
        bar_index = int(bar_key)
        bar_profile = _resolve_block_profile(bar_index)
        section_type = _resolve_section_type(bar_index)
        stage_override = overrides.setdefault(
            bar_index,
            {
                "mute_roles": [],
                "duck_gain_db": 0.0,
                "role_gain_db": {},
            },
        )
        if stage == "entry":
            stage_override["mute_roles"] = sorted(set(list(stage_override.get("mute_roles") or []) + ["chop"]))
            stage_override["duck_gain_db"] = float(stage_override.get("duck_gain_db") or 0.0) - (
                0.24 if bar_profile == "dense_long" else 0.2 if bar_profile == "dense_short" else 0.12
            )
            stage_role_gain = dict(stage_override.get("role_gain_db") or {})
            stage_role_gain["hat"] = float(stage_role_gain.get("hat", 0.0)) - (
                1.35 if bar_profile == "dense_long" else 1.15 if bar_profile == "dense_short" else 0.8
            )
            stage_role_gain["melody"] = float(stage_role_gain.get("melody", 0.0)) - (
                1.05 if bar_profile == "dense_long" else 0.92 if bar_profile == "dense_short" else 0.55
            )
            stage_role_gain["lead"] = float(stage_role_gain.get("lead", 0.0)) - (
                1.05 if bar_profile == "dense_long" else 0.92 if bar_profile == "dense_short" else 0.55
            )
            stage_override["role_gain_db"] = stage_role_gain
        elif stage == "sustain":
            stage_role_gain = dict(stage_override.get("role_gain_db") or {})
            stage_role_gain["bass"] = float(stage_role_gain.get("bass", 0.0)) - (
                0.24 if bar_profile == "dense_long" else 0.18 if bar_profile == "dense_short" else 0.08
            )
            if bar_profile in {"dense_long", "dense_short"}:
                stage_role_gain["melody"] = float(stage_role_gain.get("melody", 0.0)) - 0.18
                stage_role_gain["lead"] = float(stage_role_gain.get("lead", 0.0)) - 0.18
            stage_override["role_gain_db"] = stage_role_gain
        elif stage == "release":
            stage_role_gain = dict(stage_override.get("role_gain_db") or {})
            stage_role_gain["hat"] = float(stage_role_gain.get("hat", 0.0)) + (
                0.42 if bar_profile == "dense_long" else 0.52 if bar_profile == "dense_short" else 0.82
            )
            stage_role_gain["snare"] = float(stage_role_gain.get("snare", 0.0)) + (
                0.28 if bar_profile == "dense_long" else 0.35 if bar_profile == "dense_short" else 0.55
            )
            if bar_profile in {"open_long", "open_short"}:
                stage_role_gain["chop"] = float(stage_role_gain.get("chop", 0.0)) + 0.65
                stage_role_gain["texture"] = float(stage_role_gain.get("texture", 0.0)) + 0.38
            stage_override["role_gain_db"] = stage_role_gain
            stage_override["duck_gain_db"] = float(stage_override.get("duck_gain_db") or 0.0) + (
                0.08 if bar_profile == "dense_long" else 0.1 if bar_profile == "dense_short" else 0.18
            )
        elif stage == "recovery":
            stage_role_gain = dict(stage_override.get("role_gain_db") or {})
            stage_role_gain["hat"] = float(stage_role_gain.get("hat", 0.0)) + (
                0.95 if bar_profile == "dense_long" else 1.05 if bar_profile == "dense_short" else 1.55
            )
            stage_role_gain["snare"] = float(stage_role_gain.get("snare", 0.0)) + (
                0.48 if bar_profile == "dense_long" else 0.58 if bar_profile == "dense_short" else 0.82
            )
            stage_role_gain["chop"] = float(stage_role_gain.get("chop", 0.0)) + (
                0.45 if bar_profile == "dense_long" else 0.58 if bar_profile == "dense_short" else 1.2
            )
            stage_role_gain["texture"] = float(stage_role_gain.get("texture", 0.0)) + (
                0.22 if bar_profile == "dense_long" else 0.28 if bar_profile == "dense_short" else 0.75
            )
            if bar_profile in {"open_long", "open_short"}:
                stage_role_gain["melody"] = float(stage_role_gain.get("melody", 0.0)) + 0.52
                stage_role_gain["lead"] = float(stage_role_gain.get("lead", 0.0)) + 0.52
            stage_override["role_gain_db"] = stage_role_gain
            stage_override["duck_gain_db"] = float(stage_override.get("duck_gain_db") or 0.0) + (
                0.18 if bar_profile == "dense_long" else 0.22 if bar_profile == "dense_short" else 0.34
            )

        section_role_gain = dict(stage_override.get("role_gain_db") or {})
        if section_type == "intro":
            stage_override["mute_roles"] = sorted(set(list(stage_override.get("mute_roles") or []) + ["chop", "texture"]))
            section_role_gain["hat"] = float(section_role_gain.get("hat", 0.0)) - 0.8
            section_role_gain["melody"] = float(section_role_gain.get("melody", 0.0)) - 0.65
        elif section_type == "verse":
            section_role_gain["bass"] = float(section_role_gain.get("bass", 0.0)) - 0.08
        elif section_type == "pre_hook":
            section_role_gain["snare"] = float(section_role_gain.get("snare", 0.0)) + 0.18
            section_role_gain["hat"] = float(section_role_gain.get("hat", 0.0)) + 0.25
        elif section_type == "hook":
            section_role_gain["kick"] = float(section_role_gain.get("kick", 0.0)) + 0.26
            section_role_gain["snare"] = float(section_role_gain.get("snare", 0.0)) + 0.32
            section_role_gain["bass"] = float(section_role_gain.get("bass", 0.0)) + 0.24
            section_role_gain["melody"] = float(section_role_gain.get("melody", 0.0)) + 0.34
            section_role_gain["lead"] = float(section_role_gain.get("lead", 0.0)) + 0.34
        elif section_type == "bridge":
            section_role_gain["melody"] = float(section_role_gain.get("melody", 0.0)) - 0.18
            section_role_gain["lead"] = float(section_role_gain.get("lead", 0.0)) - 0.18
            if adaptation_profile != "dense":
                stage_override["mute_roles"] = sorted(set(list(stage_override.get("mute_roles") or []) + ["chop"]))
        elif section_type == "break":
            stage_override["mute_roles"] = sorted(set(list(stage_override.get("mute_roles") or []) + ["texture", "chop"]))
            section_role_gain["hat"] = float(section_role_gain.get("hat", 0.0)) - 0.55
            section_role_gain["bass"] = float(section_role_gain.get("bass", 0.0)) - 0.18
        elif section_type == "outro":
            stage_override["mute_roles"] = sorted(set(list(stage_override.get("mute_roles") or []) + ["chop"]))
            section_role_gain["hat"] = float(section_role_gain.get("hat", 0.0)) - 0.35
            section_role_gain["texture"] = float(section_role_gain.get("texture", 0.0)) + 0.12

        if adaptation_profile == "dry":
            stage_override["mute_roles"] = sorted(set(list(stage_override.get("mute_roles") or []) + ["texture", "chop"]))
            section_role_gain["hat"] = float(section_role_gain.get("hat", 0.0)) - 0.4
            section_role_gain["texture"] = float(section_role_gain.get("texture", 0.0)) - 0.6
        elif adaptation_profile == "spaced":
            section_role_gain["texture"] = float(section_role_gain.get("texture", 0.0)) + 0.32
            section_role_gain["melody"] = float(section_role_gain.get("melody", 0.0)) + 0.12
        elif adaptation_profile == "dense":
            section_role_gain["hat"] = float(section_role_gain.get("hat", 0.0)) + 0.18
            section_role_gain["bass"] = float(section_role_gain.get("bass", 0.0)) + 0.14
            section_role_gain["chop"] = float(section_role_gain.get("chop", 0.0)) + 0.2
        elif adaptation_profile == "hook_support" and section_type == "hook":
            section_role_gain["snare"] = float(section_role_gain.get("snare", 0.0)) + 0.18
            section_role_gain["melody"] = float(section_role_gain.get("melody", 0.0)) + 0.22
            section_role_gain["lead"] = float(section_role_gain.get("lead", 0.0)) + 0.22

        stage_override["role_gain_db"] = section_role_gain

    step_overrides: Dict[int, Dict[str, Any]] = {}
    for step_index, level in enumerate(step_activity["levels"]):
        step_bar_index = min(max(0, int(step_index // max(1, steps_per_bar))), max(0, bars - 1))
        step_profile = _resolve_block_profile(step_bar_index)
        if level >= step_activity["dense_threshold"]:
            if step_profile == "dense_long":
                step_overrides[step_index] = {
                    "mute_roles": ["chop"],
                    "duck_gain_db": -0.3,
                    "role_gain_db": {
                        "hat": -2.55,
                        "melody": -1.8,
                        "lead": -1.8,
                    },
                }
            elif step_profile == "dense_short":
                step_overrides[step_index] = {
                    "mute_roles": ["chop"],
                    "duck_gain_db": -0.27,
                    "role_gain_db": {
                        "hat": -2.35,
                        "melody": -1.68,
                        "lead": -1.68,
                    },
                }
            elif step_profile in {"open_long", "open_short"}:
                step_overrides[step_index] = {
                    "mute_roles": [],
                    "duck_gain_db": -0.16,
                    "role_gain_db": {
                        "hat": -1.25,
                        "melody": -0.9,
                        "lead": -0.9,
                    },
                }
            else:
                step_overrides[step_index] = {
                    "mute_roles": ["chop"],
                    "duck_gain_db": -0.25,
                    "role_gain_db": {
                        "hat": -2.2,
                        "melody": -1.6,
                        "lead": -1.6,
                    },
                }
        elif level >= step_activity["medium_threshold"]:
            if step_profile == "dense_long":
                step_overrides[step_index] = {
                    "mute_roles": [],
                    "duck_gain_db": -0.16,
                    "role_gain_db": {
                        "hat": -1.1,
                        "melody": -0.7,
                    },
                }
            elif step_profile == "dense_short":
                step_overrides[step_index] = {
                    "mute_roles": [],
                    "duck_gain_db": -0.14,
                    "role_gain_db": {
                        "hat": -0.98,
                        "melody": -0.6,
                    },
                }
            elif step_profile == "open_long":
                step_overrides[step_index] = {
                    "mute_roles": [],
                    "duck_gain_db": -0.04,
                    "role_gain_db": {
                        "hat": -0.35,
                        "melody": -0.18,
                        "chop": 0.12,
                    },
                }
            elif step_profile == "open_short":
                step_overrides[step_index] = {
                    "mute_roles": [],
                    "duck_gain_db": 0.0,
                    "role_gain_db": {
                        "hat": -0.18,
                        "melody": -0.08,
                        "chop": 0.22,
                    },
                }
            else:
                step_overrides[step_index] = {
                    "mute_roles": [],
                    "duck_gain_db": -0.12,
                    "role_gain_db": {
                        "hat": -0.9,
                        "melody": -0.55,
                    },
                }

    profile_bar_counts: Dict[str, int] = {}
    for block in blocks:
        profile_name = _clean_text(block.get("profile")).lower() or "unclassified"
        profile_bar_counts[profile_name] = int(profile_bar_counts.get(profile_name, 0)) + int(block.get("bars") or 0)

    def _ensure_bar_override(bar_index: int) -> Dict[str, Any]:
        return overrides.setdefault(
            int(bar_index),
            {
                "mute_roles": [],
                "duck_gain_db": 0.0,
                "role_gain_db": {},
            },
        )

    def _ensure_step_override(step_index: int) -> Dict[str, Any]:
        return step_overrides.setdefault(
            int(step_index),
            {
                "mute_roles": [],
                "duck_gain_db": 0.0,
                "role_gain_db": {},
            },
        )

    transcript_line_events = list((transcript_line_manifest or {}).get("line_events") or [])
    line_profile_counts: Dict[str, int] = {}
    line_aware_bar_count = 0
    line_aware_step_count = 0
    line_bar_markers: Set[int] = set()
    line_step_markers: Set[int] = set()
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    samples_per_step = max(1, int(round(samples_per_bar / float(max(1, steps_per_bar)))))
    for event in transcript_line_events:
        start_sample = max(0, int(event.get("start_sample") or 0))
        end_sample = max(start_sample, int(event.get("end_sample") or 0))
        line_profile = _clean_text(event.get("line_profile")).lower() or "unknown"
        line_profile_counts[line_profile] = int(line_profile_counts.get(line_profile, 0)) + 1
        start_bar = max(0, min(bars - 1, int(start_sample // samples_per_bar))) if bars > 0 else 0
        end_bar = max(start_bar, min(bars - 1, int(end_sample // samples_per_bar))) if bars > 0 else start_bar
        section_type = _clean_text(event.get("section_type")).lower() or _resolve_section_type(start_bar)
        word_count = int(event.get("word_count") or 0)
        density = float(event.get("density") or 0.0)

        for bar_index in range(start_bar, end_bar + 1):
            line_bar_markers.add(int(bar_index))
            bar_override = _ensure_bar_override(bar_index)
            role_gain = dict(bar_override.get("role_gain_db") or {})
            mute_roles = set(bar_override.get("mute_roles") or [])
            if line_profile == "dense_lyric":
                mute_roles.update({"texture", "chop"})
                role_gain["hat"] = float(role_gain.get("hat", 0.0)) - (2.2 if section_type == "verse" else 1.45 if section_type == "hook" else 1.8)
                role_gain["melody"] = float(role_gain.get("melody", 0.0)) - (1.55 if section_type == "verse" else 0.95 if section_type == "hook" else 1.2)
                role_gain["lead"] = float(role_gain.get("lead", 0.0)) - (1.55 if section_type == "verse" else 0.95 if section_type == "hook" else 1.2)
                if section_type != "hook":
                    role_gain["bass"] = float(role_gain.get("bass", 0.0)) - 0.22
                bar_override["duck_gain_db"] = float(bar_override.get("duck_gain_db") or 0.0) - (0.18 if section_type == "verse" else 0.1 if section_type == "hook" else 0.14)
            elif line_profile == "emphatic_lyric":
                mute_roles.update({"chop"})
                role_gain["hat"] = float(role_gain.get("hat", 0.0)) - (1.35 if section_type == "hook" else 1.7)
                role_gain["melody"] = float(role_gain.get("melody", 0.0)) - (0.9 if section_type == "hook" else 1.15)
                role_gain["lead"] = float(role_gain.get("lead", 0.0)) - (0.9 if section_type == "hook" else 1.15)
                bar_override["duck_gain_db"] = float(bar_override.get("duck_gain_db") or 0.0) - (0.08 if section_type == "hook" else 0.12)
            elif line_profile == "adlib":
                role_gain["texture"] = float(role_gain.get("texture", 0.0)) + 0.12
                if section_type == "hook":
                    role_gain["chop"] = float(role_gain.get("chop", 0.0)) + 0.08
            elif line_profile == "non_lyric_event":
                if section_type in {"verse", "break"}:
                    role_gain["texture"] = float(role_gain.get("texture", 0.0)) + 0.18
                    role_gain["chop"] = float(role_gain.get("chop", 0.0)) + 0.12
                    bar_override["duck_gain_db"] = float(bar_override.get("duck_gain_db") or 0.0) + 0.04
            elif line_profile == "sparse_lyric" and section_type == "hook":
                role_gain["melody"] = float(role_gain.get("melody", 0.0)) + 0.08
                role_gain["lead"] = float(role_gain.get("lead", 0.0)) + 0.08

            if word_count >= 10 and line_profile in {"dense_lyric", "emphatic_lyric"}:
                role_gain["hat"] = float(role_gain.get("hat", 0.0)) - 0.2
            if density >= 4.5 and line_profile == "dense_lyric":
                bar_override["duck_gain_db"] = float(bar_override.get("duck_gain_db") or 0.0) - 0.06

            bar_override["mute_roles"] = sorted(mute_roles)
            bar_override["role_gain_db"] = role_gain

        start_step = max(0, min((bars * steps_per_bar) - 1, int(start_sample // samples_per_step))) if bars > 0 else 0
        end_step = max(start_step, min((bars * steps_per_bar) - 1, int(end_sample // samples_per_step))) if bars > 0 else start_step
        for step_index in range(start_step, end_step + 1):
            line_step_markers.add(int(step_index))
            step_override = _ensure_step_override(step_index)
            step_role_gain = dict(step_override.get("role_gain_db") or {})
            step_mute_roles = set(step_override.get("mute_roles") or [])
            if line_profile == "dense_lyric":
                step_role_gain["hat"] = float(step_role_gain.get("hat", 0.0)) - 1.1
                step_role_gain["melody"] = float(step_role_gain.get("melody", 0.0)) - 0.85
                step_role_gain["lead"] = float(step_role_gain.get("lead", 0.0)) - 0.85
                if section_type != "hook":
                    step_mute_roles.add("chop")
                step_override["duck_gain_db"] = float(step_override.get("duck_gain_db") or 0.0) - 0.08
            elif line_profile == "emphatic_lyric":
                step_role_gain["hat"] = float(step_role_gain.get("hat", 0.0)) - 0.7
                step_role_gain["melody"] = float(step_role_gain.get("melody", 0.0)) - 0.52
                step_role_gain["lead"] = float(step_role_gain.get("lead", 0.0)) - 0.52
            elif line_profile == "non_lyric_event":
                step_role_gain["texture"] = float(step_role_gain.get("texture", 0.0)) + 0.08
            step_override["mute_roles"] = sorted(step_mute_roles)
            step_override["role_gain_db"] = step_role_gain

        recovery_step = end_step + 1
        if 0 <= recovery_step < (bars * steps_per_bar):
            recovery_override = _ensure_step_override(recovery_step)
            recovery_gain = dict(recovery_override.get("role_gain_db") or {})
            if line_profile == "dense_lyric" and section_type == "hook":
                recovery_gain["snare"] = float(recovery_gain.get("snare", 0.0)) + 0.22
                recovery_gain["hat"] = float(recovery_gain.get("hat", 0.0)) + 0.35
            elif line_profile == "emphatic_lyric" and section_type in {"hook", "transition"}:
                recovery_gain["snare"] = float(recovery_gain.get("snare", 0.0)) + 0.18
                recovery_gain["chop"] = float(recovery_gain.get("chop", 0.0)) + 0.16
            elif line_profile == "adlib" and section_type == "hook":
                recovery_gain["texture"] = float(recovery_gain.get("texture", 0.0)) + 0.12
            recovery_override["role_gain_db"] = recovery_gain
            line_step_markers.add(int(recovery_step))

    return {
        "bar_activity": activity,
        "phrase_blocks": phrase_blocks,
        "macrosections": macrosections,
        "step_activity": step_activity,
        "bar_overrides": overrides,
        "step_overrides": step_overrides,
        "profile_bar_counts": profile_bar_counts,
        "transcript_line_profiles": line_profile_counts,
        "transcript_line_events_count": len(transcript_line_events),
        "line_aware_bar_count": len(line_bar_markers),
        "line_aware_step_count": len(line_step_markers),
    }


def _mutate_section_bar_pattern(
    base_bar: List[bool],
    role: str,
    section: Dict[str, Any],
    section_bar_index: int,
    section_bars: int,
    steps_per_bar: int,
    creativity: float,
    fill_probability: float,
    previous_section_name: Optional[str],
    next_section_name: Optional[str],
    rng: "np.random.Generator",
) -> List[bool]:
    section_name = _clean_text(section.get("name")) or "main_a"
    drum_density = float(section.get("drum_density") or 1.0)
    resolved_role = (_clean_text(role) or "sample").lower()
    pattern = list(base_bar)
    half_step = max(1, steps_per_bar // 2)
    quarter_step = max(1, steps_per_bar // 4)
    eighth_step = max(1, steps_per_bar // 8)
    is_first_bar = int(section_bar_index) == 0
    is_last_bar = int(section_bar_index) >= max(0, int(section_bars) - 1)
    previous_name = _clean_text(previous_section_name).lower()
    next_name = _clean_text(next_section_name).lower()

    def _set_step(step: int, active: bool = True) -> None:
        if 0 <= step < len(pattern):
            pattern[step] = active

    if section_name == "intro":
        if resolved_role == "kick":
            pattern = [False] * len(pattern)
            _set_step(0, True)
            if section_bar_index % 2 == 1 and steps_per_bar > 8:
                _set_step(half_step, True)
        elif resolved_role in {"snare", "clap"}:
            pattern = [False] * len(pattern)
            if section_bar_index > 0:
                _set_step(half_step, True)
        elif resolved_role in {"hat", "hihat"}:
            pattern = [(index % quarter_step == 0) for index in range(len(pattern))]
        elif resolved_role == "bass":
            pattern = [False] * len(pattern)
            _set_step(0, True)
        else:
            pattern = [False] * len(pattern)
            _set_step(0, True)
    elif section_name == "fill":
        if resolved_role in {"hat", "hihat"}:
            pattern = [True if index >= len(pattern) - quarter_step else pattern[index] for index in range(len(pattern))]
        elif resolved_role in {"snare", "clap"}:
            _set_step(max(0, len(pattern) - quarter_step), True)
            _set_step(max(0, len(pattern) - 2), True)
        elif resolved_role == "kick":
            _set_step(max(0, len(pattern) - quarter_step), True)
        elif resolved_role == "bass":
            pattern = [False] * len(pattern)
            _set_step(0, True)
    elif section_name == "main_b":
        if resolved_role in {"hat", "hihat"}:
            for step in range(1, len(pattern), 2):
                if not pattern[step] and rng.random() < 0.22 + (creativity * 0.35):
                    pattern[step] = True
            if rng.random() < 0.22 + (creativity * 0.22):
                _set_step(max(0, len(pattern) - eighth_step - 1), True)
        if resolved_role == "kick":
            for step in (max(1, quarter_step - 1), half_step + max(1, quarter_step // 2), len(pattern) - quarter_step):
                if 0 <= step < len(pattern) and rng.random() < 0.18 + (creativity * 0.2):
                    pattern[step] = True
            if rng.random() < 0.16 + (creativity * 0.16):
                _set_step(max(0, len(pattern) - quarter_step - 1), True)
        if resolved_role in {"snare", "clap"} and rng.random() < fill_probability * 0.45:
            _set_step(max(0, len(pattern) - quarter_step), True)
        if resolved_role == "clap" and rng.random() < 0.2 + (creativity * 0.2):
            _set_step(max(0, half_step - 1), True)
        if resolved_role == "bass" and rng.random() < 0.32 + (creativity * 0.2):
            _set_step(half_step + max(1, quarter_step // 2), True)
    elif section_name == "break":
        if resolved_role in {"hat", "hihat"}:
            pattern = [False] * len(pattern)
            _set_step(half_step, True)
        elif resolved_role in {"snare", "clap"}:
            pattern = [False] * len(pattern)
            _set_step(half_step, True)
        elif resolved_role == "kick":
            pattern = [False] * len(pattern)
            _set_step(0, True)
        elif resolved_role == "bass":
            pattern = [False] * len(pattern)
            _set_step(0, True)
        else:
            pattern = [False] * len(pattern)
    elif section_name == "outro":
        if resolved_role in {"hat", "hihat"}:
            pattern = [(index % quarter_step == 0) for index in range(len(pattern))]
        elif resolved_role in {"kick", "bass"}:
            trimmed = [False] * len(pattern)
            _set_step(0, True)
            if resolved_role == "kick" and len(pattern) > half_step and section_bar_index == 0:
                trimmed[half_step] = True
            pattern = trimmed
        else:
            pattern = [False] * len(pattern)
            if resolved_role in {"snare", "clap"}:
                _set_step(half_step, True)

    if section_name in {"main_a", "main_b", "main_c"}:
        if resolved_role in {"kick", "snare", "clap", "bass"}:
            for step in range(len(pattern)):
                if pattern[step]:
                    continue
                chance = 0.0
                if resolved_role == "kick" and step % quarter_step != 0:
                    chance = (0.04 + (creativity * 0.08)) * drum_density
                elif resolved_role in {"snare", "clap"} and step not in {half_step, len(pattern) - quarter_step}:
                    chance = (0.015 + (creativity * 0.05)) * drum_density
                elif resolved_role == "bass" and step % quarter_step == 0:
                    chance = 0.05 + (creativity * 0.08)
                if chance and rng.random() < chance:
                    pattern[step] = True
            if resolved_role == "kick" and rng.random() < 0.12 + (creativity * 0.12):
                _set_step(max(0, half_step - 1), True)
            if resolved_role in {"snare", "clap"} and rng.random() < 0.14 + (creativity * 0.14):
                _set_step(max(0, len(pattern) - eighth_step - 1), True)
            if section_name == "main_c" and resolved_role == "kick" and rng.random() < 0.18 + (creativity * 0.16):
                _set_step(max(0, quarter_step * 3 - 1), True)

        if resolved_role in {"hat", "hihat"}:
            for step in range(len(pattern)):
                if step % 2 == 1 and not pattern[step] and rng.random() < creativity * 0.12 * drum_density:
                    pattern[step] = True
            if rng.random() < 0.18 + (creativity * 0.18):
                _set_step(max(0, len(pattern) - quarter_step - 1), True)

    if is_last_bar and next_name in {"fill", "break", "main_b", "main_c"}:
        if resolved_role in {"hat", "hihat"}:
            tail_steps = [max(0, len(pattern) - quarter_step), max(0, len(pattern) - eighth_step - 1)]
            if next_name in {"main_b", "main_c"}:
                tail_steps.append(max(0, len(pattern) - 2))
            for idx in tail_steps:
                if rng.random() < 0.52 + (creativity * 0.2):
                    _set_step(idx, True)
        elif resolved_role == "kick":
            pickup_step = max(0, len(pattern) - quarter_step - 1)
            if next_name in {"main_b", "main_c"} and rng.random() < 0.38 + (creativity * 0.16):
                _set_step(pickup_step, True)
            elif next_name in {"fill", "break"} and rng.random() < 0.22 + (creativity * 0.12):
                _set_step(max(0, len(pattern) - eighth_step - 1), True)
        elif resolved_role in {"snare", "clap"}:
            accent_step = max(0, len(pattern) - quarter_step)
            if next_name in {"fill", "main_b", "main_c"} and rng.random() < 0.3 + (fill_probability * 0.35):
                _set_step(accent_step, True)
        elif resolved_role == "bass" and next_name in {"break", "fill"}:
            pattern = [False if idx >= len(pattern) - quarter_step else active for idx, active in enumerate(pattern)]

    if is_first_bar and previous_name in {"fill", "break"}:
        if resolved_role == "kick":
            _set_step(0, True)
            if section_name in {"main_b", "main_c"}:
                _set_step(max(1, quarter_step - 1), True)
                if rng.random() < 0.28 + (creativity * 0.14):
                    _set_step(half_step, True)
        elif resolved_role in {"snare", "clap"} and section_name in {"main_b", "main_c"}:
            _set_step(half_step, True)
            if resolved_role == "clap" and rng.random() < 0.22 + (creativity * 0.14):
                _set_step(max(0, half_step - 1), True)
        elif resolved_role in {"hat", "hihat"}:
            if previous_name == "break" and section_name in {"main_b", "main_c"}:
                for idx in range(1, len(pattern), 2):
                    if rng.random() < 0.34 + (creativity * 0.18):
                        _set_step(idx, True)
            elif section_name == "fill":
                for idx in range(max(0, len(pattern) - quarter_step), len(pattern)):
                    _set_step(idx, True)
        elif resolved_role == "bass":
            _set_step(0, True)
            if section_name == "main_c" and rng.random() < 0.32 + (creativity * 0.16):
                _set_step(half_step, True)

    return pattern


def _compose_arranged_pattern(
    pattern: str,
    role: str,
    sections: List[Dict[str, Any]],
    steps_per_bar: int,
    creativity: float,
    fill_probability: float,
    seed: int,
) -> str:
    base_bar = _parse_pattern(pattern, bars=1, steps_per_bar=steps_per_bar)
    all_steps: List[bool] = []
    rng = np.random.default_rng(seed)

    for section_index, section in enumerate(sections):
        bars = max(1, int(section.get("bars") or 1))
        previous_section_name = _clean_text((sections[section_index - 1] if section_index > 0 else {}).get("name"))
        next_section_name = _clean_text((sections[section_index + 1] if section_index + 1 < len(sections) else {}).get("name"))
        for bar_index in range(bars):
            all_steps.extend(
                _mutate_section_bar_pattern(
                    base_bar=base_bar,
                    role=role,
                    section=section,
                    section_bar_index=bar_index,
                    section_bars=bars,
                    steps_per_bar=steps_per_bar,
                    creativity=creativity,
                    fill_probability=fill_probability,
                    previous_section_name=previous_section_name or None,
                    next_section_name=next_section_name or None,
                    rng=rng,
                )
            )

    return _pattern_steps_to_text(all_steps)


def _estimate_tempo_bpm(mono_samples: "np.ndarray", sample_rate: int) -> Optional[float]:
    if np is None or signal is None:
        return None
    if len(mono_samples) < sample_rate:
        return None

    frame_size = 1024
    hop_size = 512
    if len(mono_samples) < frame_size:
        return None

    frame_count = 1 + (len(mono_samples) - frame_size) // hop_size
    energies = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_size
        frame = mono_samples[start : start + frame_size]
        energies[index] = float(np.sqrt(np.mean(frame * frame) + 1e-12))

    onset_env = np.maximum(0.0, np.diff(energies))
    if not np.any(onset_env > 0):
        return None

    onset_env = onset_env - float(np.mean(onset_env))
    autocorr = np.correlate(onset_env, onset_env, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    min_bpm = 60.0
    max_bpm = 200.0
    min_lag = max(1, int((60.0 / max_bpm) * sample_rate / hop_size))
    max_lag = max(min_lag + 1, int((60.0 / min_bpm) * sample_rate / hop_size))
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if max_lag <= min_lag:
        return None

    target_region = autocorr[min_lag:max_lag]
    if target_region.size == 0:
        return None

    best_lag = int(np.argmax(target_region)) + min_lag
    bpm = 60.0 * sample_rate / (hop_size * best_lag)
    return round(float(bpm), 2)


def _optional_import_librosa():
    try:
        import librosa  # type: ignore

        return librosa
    except Exception:
        return None


def _optional_import_basic_pitch_predict():
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="resampy")
            root_logger.setLevel(max(previous_level, logging.ERROR))
            from basic_pitch.inference import predict  # type: ignore

        return predict
    except Exception:
        return None
    finally:
        root_logger.setLevel(previous_level)


def _estimate_key_from_chroma(chroma_mean: "np.ndarray") -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    if chroma_mean.size != 12 or float(np.sum(chroma_mean)) <= 0:
        return {"key": None, "scale": None, "confidence": None}

    normalized = chroma_mean / float(np.sum(chroma_mean))
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float32,
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float32,
    )
    major_profile /= float(np.sum(major_profile))
    minor_profile /= float(np.sum(minor_profile))

    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best_score = -1.0
    best_key = None
    best_scale = None

    for index, pitch_class in enumerate(pitch_classes):
        major_score = float(np.dot(normalized, np.roll(major_profile, index)))
        if major_score > best_score:
            best_score = major_score
            best_key = pitch_class
            best_scale = "major"

        minor_score = float(np.dot(normalized, np.roll(minor_profile, index)))
        if minor_score > best_score:
            best_score = minor_score
            best_key = pitch_class
            best_scale = "minor"

    return {
        "key": best_key,
        "scale": best_scale,
        "confidence": round(best_score, 4) if best_score >= 0 else None,
    }


def _compute_librosa_descriptors(source_path: str, sample_rate: int = 22050) -> Dict[str, Any]:
    librosa = _optional_import_librosa()
    if librosa is None:
        return {}

    mono, effective_sample_rate = librosa.load(source_path, sr=sample_rate, mono=True)
    if mono.size == 0:
        return {}

    tempo_value, beat_frames = librosa.beat.beat_track(y=mono, sr=effective_sample_rate)
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    onset_envelope = librosa.onset.onset_strength(y=mono, sr=effective_sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=effective_sample_rate)
    spectral_flatness = librosa.feature.spectral_flatness(y=mono)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=effective_sample_rate)
    chroma_mean = np.mean(chroma, axis=1) if chroma.size else np.zeros(12, dtype=np.float32)
    harmonic_signal, percussive_signal = librosa.effects.hpss(mono)

    harmonic_energy = float(np.mean(np.abs(harmonic_signal))) if harmonic_signal.size else 0.0
    percussive_energy = float(np.mean(np.abs(percussive_signal))) if percussive_signal.size else 0.0
    total_hp_energy = harmonic_energy + percussive_energy
    harmonic_ratio = harmonic_energy / total_hp_energy if total_hp_energy > 0 else None
    percussive_ratio = percussive_energy / total_hp_energy if total_hp_energy > 0 else None

    key_estimate = _estimate_key_from_chroma(chroma_mean)
    dominant_pitch_class = None
    if chroma_mean.size == 12 and float(np.sum(chroma_mean)) > 0:
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        dominant_pitch_class = pitch_classes[int(np.argmax(chroma_mean))]

    beat_times = librosa.frames_to_time(beat_frames, sr=effective_sample_rate).tolist() if len(beat_frames) else []
    return {
        "tempo_estimate_bpm_librosa": round(float(tempo_value), 2) if tempo_value else None,
        "beat_frame_count": int(len(beat_frames)),
        "beat_times_preview_seconds": [round(float(item), 3) for item in beat_times[:16]],
        "onset_strength_mean": round(float(np.mean(onset_envelope)), 4) if onset_envelope.size else None,
        "spectral_bandwidth_hz": round(float(np.mean(spectral_bandwidth)), 2) if spectral_bandwidth.size else None,
        "spectral_flatness": round(float(np.mean(spectral_flatness)), 6) if spectral_flatness.size else None,
        "dominant_pitch_class": dominant_pitch_class,
        "estimated_key": key_estimate["key"],
        "estimated_scale": key_estimate["scale"],
        "estimated_key_confidence": key_estimate["confidence"],
        "harmonic_ratio": round(float(harmonic_ratio), 4) if harmonic_ratio is not None else None,
        "percussive_ratio": round(float(percussive_ratio), 4) if percussive_ratio is not None else None,
        "chroma_profile": [round(float(value), 4) for value in chroma_mean.tolist()],
    }


def _summarize_basic_pitch_note_events(note_events: List[Any], duration_seconds: Optional[float]) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    if not note_events:
        return {
            "note_count": 0,
            "mean_note_length_seconds": None,
            "pitch_range_midi": None,
            "mean_confidence": None,
            "dominant_pitch_classes": [],
            "notes_per_second": None,
        }

    note_lengths: List[float] = []
    pitches: List[int] = []
    confidences: List[float] = []
    pitch_class_counts = np.zeros(12, dtype=np.int32)

    for event in note_events:
        try:
            start, end, midi_pitch, confidence = event[:4]
        except Exception:
            continue

        start_value = float(start)
        end_value = float(end)
        midi_value = int(midi_pitch)
        confidence_value = float(confidence)
        note_lengths.append(max(0.0, end_value - start_value))
        pitches.append(midi_value)
        confidences.append(confidence_value)
        pitch_class_counts[midi_value % 12] += 1

    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    ranked_pitch_classes = np.argsort(pitch_class_counts)[::-1]
    dominant_pitch_classes = [
        pitch_classes[int(index)] for index in ranked_pitch_classes if pitch_class_counts[int(index)] > 0
    ][:4]

    duration_value = float(duration_seconds) if duration_seconds else None
    return {
        "note_count": int(len(pitches)),
        "mean_note_length_seconds": round(float(np.mean(note_lengths)), 4) if note_lengths else None,
        "pitch_range_midi": {
            "min": int(min(pitches)),
            "max": int(max(pitches)),
        }
        if pitches
        else None,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "dominant_pitch_classes": dominant_pitch_classes,
        "notes_per_second": round(float(len(pitches) / duration_value), 4)
        if duration_value and duration_value > 0
        else None,
    }


def _extract_basic_pitch_data(
    source_path: str,
    output_midi_path: Optional[str] = None,
    output_notes_path: Optional[str] = None,
    midi_tempo: float = 120.0,
) -> Dict[str, Any]:
    predict = _optional_import_basic_pitch_predict()
    if predict is None:
        return {}

    _, midi_data, note_events = predict(
        source_path,
        midi_tempo=float(midi_tempo or 120.0),
    )

    if output_midi_path:
        os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
        midi_data.write(output_midi_path)

    notes_payload = []
    for event in note_events:
        try:
            start, end, midi_pitch, confidence = event[:4]
            pitch_bends = event[4] if len(event) > 4 else None
        except Exception:
            continue

        notes_payload.append(
            {
                "start_seconds": round(float(start), 4),
                "end_seconds": round(float(end), 4),
                "midi_pitch": int(midi_pitch),
                "confidence": round(float(confidence), 4),
                "pitch_bends": [int(item) for item in list(pitch_bends)] if pitch_bends else [],
            }
        )

    if output_notes_path:
        _write_json(output_notes_path, {"note_events": notes_payload})

    return {
        "note_events": notes_payload,
        "summary": _summarize_basic_pitch_note_events(note_events, None),
        "midi_path": output_midi_path,
        "notes_path": output_notes_path,
    }


def _compute_audio_descriptors(samples: "np.ndarray", sample_rate: int) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    stereo = _ensure_stereo(samples)
    mono = stereo.mean(axis=1)
    abs_mono = np.abs(mono)
    peak = float(np.max(abs_mono)) if abs_mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    silence_ratio = float(np.mean(abs_mono < 0.001)) if abs_mono.size else 0.0
    zero_crossing_rate = float(np.mean(np.diff(np.signbit(mono)).astype(np.float32))) if len(mono) > 1 else 0.0

    dominant_frequencies: List[float] = []
    spectral_centroid_hz = None
    spectral_rolloff_85_hz = None
    sub_energy_ratio = None
    mid_energy_ratio = None
    high_energy_ratio = None

    if signal is not None and len(mono) >= 2048:
        frequencies, _, stft = signal.stft(
            mono,
            fs=sample_rate,
            nperseg=2048,
            noverlap=1536,
            boundary=None,
        )
        mean_magnitude = np.abs(stft).mean(axis=1)
        total_energy = float(np.sum(mean_magnitude))
        if total_energy > 0:
            spectral_centroid_hz = float(np.sum(frequencies * mean_magnitude) / total_energy)
            cumulative = np.cumsum(mean_magnitude)
            rolloff_index = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
            rolloff_index = min(rolloff_index, len(frequencies) - 1)
            spectral_rolloff_85_hz = float(frequencies[rolloff_index])
            dominant_indices = np.argsort(mean_magnitude)[-5:][::-1]
            dominant_frequencies = [
                round(float(frequencies[index]), 2)
                for index in dominant_indices
                if mean_magnitude[index] > 0
            ]

            sub_mask = frequencies < 200
            mid_mask = (frequencies >= 200) & (frequencies < 2000)
            high_mask = frequencies >= 2000
            sub_energy_ratio = float(np.sum(mean_magnitude[sub_mask]) / total_energy)
            mid_energy_ratio = float(np.sum(mean_magnitude[mid_mask]) / total_energy)
            high_energy_ratio = float(np.sum(mean_magnitude[high_mask]) / total_energy)

    tempo_estimate_bpm = _estimate_tempo_bpm(mono, sample_rate)
    left = stereo[:, 0]
    right = stereo[:, 1]
    stereo_balance = float(np.mean(np.abs(left)) - np.mean(np.abs(right)))
    stereo_correlation = None
    if len(left) > 1 and np.std(left) > 0 and np.std(right) > 0:
        stereo_correlation = float(np.corrcoef(left, right)[0, 1])

    dynamic_low = float(np.percentile(abs_mono, 10)) if abs_mono.size else 0.0
    dynamic_high = float(np.percentile(abs_mono, 95)) if abs_mono.size else 0.0
    dynamic_range_db = _to_db(dynamic_high + 1e-12) - _to_db(dynamic_low + 1e-12)

    mood_tags: List[str] = []
    if tempo_estimate_bpm is not None:
        if tempo_estimate_bpm < 85:
            mood_tags.append("slow")
        elif tempo_estimate_bpm < 115:
            mood_tags.append("mid-tempo")
        else:
            mood_tags.append("up-tempo")
    if spectral_centroid_hz is not None:
        if spectral_centroid_hz < 1200:
            mood_tags.append("dark")
        elif spectral_centroid_hz > 3000:
            mood_tags.append("bright")
        else:
            mood_tags.append("balanced")
    if sub_energy_ratio is not None and sub_energy_ratio > 0.35:
        mood_tags.append("sub-heavy")

    return {
        "duration_seconds": round(float(len(mono) / sample_rate), 4) if sample_rate else 0.0,
        "sample_rate": int(sample_rate),
        "channels": int(stereo.shape[1]),
        "peak_dbfs": round(_to_db(peak), 2),
        "rms_dbfs": round(_to_db(rms), 2),
        "crest_factor_db": round(_to_db((peak + 1e-12) / (rms + 1e-12), floor=0.0), 2),
        "dynamic_range_db": round(float(dynamic_range_db), 2),
        "silence_ratio": round(silence_ratio, 4),
        "zero_crossing_rate": round(zero_crossing_rate, 4),
        "tempo_estimate_bpm": tempo_estimate_bpm,
        "spectral_centroid_hz": round(float(spectral_centroid_hz), 2) if spectral_centroid_hz is not None else None,
        "spectral_rolloff_85_hz": round(float(spectral_rolloff_85_hz), 2) if spectral_rolloff_85_hz is not None else None,
        "dominant_frequencies_hz": dominant_frequencies,
        "sub_energy_ratio": round(float(sub_energy_ratio), 4) if sub_energy_ratio is not None else None,
        "mid_energy_ratio": round(float(mid_energy_ratio), 4) if mid_energy_ratio is not None else None,
        "high_energy_ratio": round(float(high_energy_ratio), 4) if high_energy_ratio is not None else None,
        "stereo_balance": round(float(stereo_balance), 4),
        "stereo_correlation": round(float(stereo_correlation), 4) if stereo_correlation is not None else None,
        "mood_tags": mood_tags,
    }


def _summarize_analysis(analysis: Dict[str, Any]) -> str:
    tempo = analysis.get("tempo_estimate_bpm")
    tempo_text = f"{tempo} BPM" if tempo is not None else "tempo nao estimado"
    centroid = analysis.get("spectral_centroid_hz")
    centroid_text = f"centroide {centroid} Hz" if centroid is not None else "centroide indisponivel"
    key = analysis.get("estimated_key")
    scale = analysis.get("estimated_scale")
    key_text = f"tonalidade estimada {key} {scale}" if key and scale else "tonalidade indisponivel"
    melody_summary = analysis.get("melodic_summary") or {}
    note_count = melody_summary.get("note_count")
    note_text = f"{note_count} notas detectadas" if note_count else "sem extracao melodica"
    mood_tags = ", ".join(analysis.get("mood_tags") or []) or "sem tags"
    return f"{tempo_text}; {centroid_text}; {key_text}; {note_text}; perfil {mood_tags}"


def _infer_reference_role_from_path(path: str) -> str:
    filename = os.path.basename(path).lower()
    if "no_vocals" in filename or "instrumental" in filename or "accompaniment" in filename:
        return "instrumental"
    for token, role in (
        ("kick", "kick"),
        ("snare", "snare"),
        ("clap", "clap"),
        ("hat", "hat"),
        ("hihat", "hat"),
        ("808", "bass"),
        ("bass", "bass"),
        ("melody", "melody"),
        ("lead", "lead"),
        ("vocal", "vocal"),
        ("vox", "vocal"),
        ("loop", "loop"),
    ):
        if token in filename:
            return role
    return "sample"


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return {}


def _stereo_from_mono(mono: "np.ndarray") -> "np.ndarray":
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim == 1:
        mono = mono[:, np.newaxis]
    if mono.shape[1] == 1:
        mono = np.repeat(mono, 2, axis=1)
    return np.clip(mono, -1.0, 1.0)


def _normalize_audio_buffer(samples: "np.ndarray", peak: float = 0.95) -> "np.ndarray":
    samples = np.asarray(samples, dtype=np.float32)
    current_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if current_peak <= 0:
        return samples
    gain = min(float(peak) / current_peak, 4.0)
    return np.clip(samples * gain, -1.0, 1.0)


def _extract_reference_sub_frequency(blueprint: Dict[str, Any]) -> float:
    summaries = blueprint.get("reference_summaries") or []
    candidates: List[float] = []
    for summary in summaries:
        analysis = summary.get("analysis") or {}
        for value in analysis.get("dominant_frequencies_hz") or []:
            try:
                frequency = float(value)
            except Exception:
                continue
            if 35.0 <= frequency <= 120.0:
                candidates.append(frequency)
    if not candidates:
        return 55.0
    return round(float(median(candidates)), 2)


def _normalize_instrumental_style_tags(style_tags: Optional[Iterable[Any]]) -> List[str]:
    tags = _normalize_str_list(style_tags)
    normalized = {tag.lower() for tag in tags}
    defaults = [
        "hip hop",
        "rap",
        "boom bap",
        "old school",
        "dusty",
        "sampled",
        "vinyl",
        "soulful",
    ]
    for token in defaults:
        if token not in normalized:
            tags.append(token)
            normalized.add(token)
    return tags


def _uses_boombap_profile(style_tags: Optional[Iterable[Any]]) -> bool:
    normalized = {tag.lower() for tag in _normalize_instrumental_style_tags(style_tags)}
    return any(
        token in normalized
        for token in (
            "hip hop",
            "hip-hop",
            "rap",
            "boom bap",
            "boombap",
            "old school",
            "90s hip hop",
            "golden era",
        )
    )


def _resolve_default_blueprint_pattern(role: str, style_tags: Optional[Iterable[Any]] = None) -> str:
    resolved_role = (_clean_text(role) or "sample").lower()
    boombap_profile = _uses_boombap_profile(style_tags)
    if boombap_profile:
        patterns = {
            "kick": "x---x---x--x----",
            "snare": "----x-------x---",
            "clap": "----x-------x---",
            "hat": "x-xx-x-xx-xx-x-x",
            "hihat": "x-xx-x-xx-xx-x-x",
            "bass": "x-----x---x-----",
            "melody": "x-------x-------",
            "lead": "x-------x-------",
        }
    else:
        patterns = {
            "kick": "x---x---x---x---",
            "snare": "----x-------x---",
            "clap": "----x-------x---",
            "hat": "x-x-x-x-x-x-x-x-",
            "hihat": "x-x-x-x-x-x-x-x-",
            "bass": "x-------x-------",
            "melody": "x---------------",
            "lead": "x---------------",
        }
    return patterns.get(resolved_role, "x---------------")


def _resolve_primary_reference_audio_path(blueprint: Dict[str, Any]) -> str:
    for summary in blueprint.get("reference_summaries") or []:
        candidate = _clean_text(summary.get("path"))
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def _extract_reference_drum_shot(
    reference_path: str,
    sample_rate: int,
    role: str,
    style_tags: Optional[Iterable[Any]] = None,
    variant_name: Optional[str] = None,
) -> Optional["np.ndarray"]:
    if np is None:
        return None
    source_path = _resolve_existing_path(reference_path)
    source_audio, _ = _load_reference_audio_array(source_path, sample_rate=sample_rate, channels=DEFAULT_CHANNELS)
    stereo = _ensure_stereo(source_audio).astype(np.float32)
    if stereo.size == 0:
        return None

    resolved_role = (_clean_text(role) or "sample").lower()
    resolved_variant = _clean_text(variant_name).lower() or "default"
    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    boombap_profile = _uses_boombap_profile(style_tokens)

    shot_seconds_map = {
        "kick": 0.18 if boombap_profile else 0.15,
        "snare": 0.2 if boombap_profile else 0.17,
        "clap": 0.22 if boombap_profile else 0.19,
        "hat": 0.1,
        "hihat": 0.1,
    }
    hp_cutoff_map = {
        "kick": 45.0,
        "snare": 180.0,
        "clap": 220.0,
        "hat": 2400.0,
        "hihat": 2400.0,
    }
    lp_cutoff_map = {
        "kick": 2400.0,
        "snare": 5200.0,
        "clap": 6200.0,
        "hat": 9800.0,
        "hihat": 9800.0,
    }
    shot_seconds = shot_seconds_map.get(resolved_role, 0.16)
    shot_samples = min(stereo.shape[0], max(1024, int(round(shot_seconds * sample_rate))))
    if shot_samples <= 0:
        return None

    search_start = int(stereo.shape[0] * 0.06)
    search_end = max(search_start + 1, int(stereo.shape[0] * 0.86) - shot_samples)
    step = max(2048, shot_samples // 2)
    ranked_candidates: List[tuple[float, int]] = []
    for start in range(search_start, search_end, step):
        window = stereo[start : start + shot_samples]
        mono = window.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
        attack = float(np.max(np.abs(np.diff(mono)))) if mono.size > 1 else 0.0
        crest = float(np.max(np.abs(mono))) if mono.size else 0.0
        if resolved_role == "kick":
            low_focus = float(np.mean(np.abs(window[:, 0]))) + float(np.mean(np.abs(window[:, 1])))
            score = (rms * 0.4) + (attack * 0.18) + (crest * 0.22) + (low_focus * 0.2)
        elif resolved_role in {"hat", "hihat"}:
            score = (attack * 0.52) + (crest * 0.28) + (rms * 0.2)
        else:
            score = (attack * 0.4) + (crest * 0.24) + (rms * 0.36)
        ranked_candidates.append((score, start))
    if not ranked_candidates:
        return None
    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    resolved_rank = 1 if resolved_variant in {"alt", "ghost"} and len(ranked_candidates) > 1 else 0
    chosen_start = ranked_candidates[resolved_rank][1]
    shot = stereo[chosen_start : chosen_start + shot_samples].copy()

    if signal is not None and shot.shape[0] > 128:
        try:
            hp_cutoff = hp_cutoff_map.get(resolved_role, 180.0)
            lp_cutoff = lp_cutoff_map.get(resolved_role, 6500.0)
            hp_b, hp_a = signal.butter(2, hp_cutoff / (sample_rate * 0.5), btype="highpass")
            lp_b, lp_a = signal.butter(2, lp_cutoff / (sample_rate * 0.5), btype="lowpass")
            for channel in range(shot.shape[1]):
                filtered = signal.filtfilt(hp_b, hp_a, shot[:, channel]).astype(np.float32)
                filtered = signal.filtfilt(lp_b, lp_a, filtered).astype(np.float32)
                shot[:, channel] = filtered
        except Exception:
            pass

    mono = shot.mean(axis=1).astype(np.float32)
    transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
    transient = np.clip(transient, -0.3, 0.3)
    if resolved_role in {"snare", "clap", "hat", "hihat"}:
        shot[:, 0] += transient * (0.11 if boombap_profile else 0.08)
        shot[:, 1] -= transient * (0.09 if boombap_profile else 0.06)
    fade_samples = min(max(32, int(round(sample_rate * 0.012))), max(1, shot.shape[0] // 6))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    shot[:fade_samples] *= fade_in[:, np.newaxis]
    shot[-fade_samples:] *= fade_out[:, np.newaxis]
    gain = {
        "kick": 0.48,
        "snare": 0.34,
        "clap": 0.3,
        "hat": 0.22,
        "hihat": 0.22,
    }.get(resolved_role, 0.24)
    if resolved_variant == "ghost":
        gain *= 0.52
    return _normalize_audio_buffer(np.tanh(shot * (1.16 if boombap_profile else 1.08)) * gain, peak=0.62 if resolved_role == "kick" else 0.52)


def _hybridize_one_shot_with_reference(
    base_sample: "np.ndarray",
    reference_shot: Optional["np.ndarray"],
    role: str,
    variant_name: Optional[str] = None,
    style_tags: Optional[Iterable[Any]] = None,
) -> "np.ndarray":
    if np is None or reference_shot is None:
        return base_sample
    base = _ensure_stereo(base_sample).astype(np.float32)
    ref = _ensure_stereo(reference_shot).astype(np.float32)
    if base.size == 0 or ref.size == 0:
        return base

    resolved_role = (_clean_text(role) or "sample").lower()
    resolved_variant = _clean_text(variant_name).lower() or "default"
    boombap_profile = _uses_boombap_profile(style_tags)
    target_len = max(base.shape[0], ref.shape[0])
    base_padded = np.zeros((target_len, 2), dtype=np.float32)
    ref_padded = np.zeros((target_len, 2), dtype=np.float32)
    base_padded[: base.shape[0]] = base
    ref_padded[: ref.shape[0]] = ref

    blend = {
        "kick": 0.28,
        "snare": 0.34,
        "clap": 0.3,
        "hat": 0.24,
        "hihat": 0.24,
    }.get(resolved_role, 0.0)
    if boombap_profile:
        blend += 0.06
    if resolved_variant == "ghost":
        blend *= 0.72
    if blend <= 0:
        return base

    hybrid = (base_padded * (1.0 - blend)) + (ref_padded * blend)
    if resolved_role in {"snare", "clap", "hat", "hihat"}:
        mono = hybrid.mean(axis=1).astype(np.float32)
        transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
        hybrid[:, 0] += transient * 0.05
        hybrid[:, 1] -= transient * 0.04
    return _normalize_audio_buffer(np.tanh(hybrid * (1.08 if boombap_profile else 1.03)), peak=0.9)


def _infer_blueprint_swing(style_tags: Iterable[Any], target_bpm: float) -> float:
    normalized_tags = {tag.lower() for tag in _normalize_instrumental_style_tags(style_tags)}
    swing = 0.055
    if target_bpm <= 95:
        swing += 0.038
    if any(token in normalized_tags for token in ("boom bap", "old school", "lofi", "soulful", "dark boom bap")):
        swing += 0.042
    if any(token in normalized_tags for token in ("trap", "drill", "club")):
        swing -= 0.03
    return round(max(0.02, min(0.22, swing)), 3)


def _synthesize_role_one_shot(
    role: str,
    output_path: str,
    sample_rate: int,
    target_bpm: float,
    sub_frequency_hz: float,
    style_tags: Optional[Iterable[Any]] = None,
    variant_name: Optional[str] = None,
    reference_path: Optional[str] = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")

    resolved_role = (_clean_text(role) or "sample").lower()
    resolved_variant = _clean_text(variant_name).lower() or "default"
    ghost_variant = resolved_variant == "ghost"
    style_tokens = {token.lower() for token in _normalize_instrumental_style_tags(style_tags)}
    boombap_profile = _uses_boombap_profile(style_tokens)
    seed = zlib.crc32(f"{resolved_role}:{resolved_variant}:{sample_rate}:{round(target_bpm, 2)}".encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    def _mono_time(duration: float) -> "np.ndarray":
        size = max(32, int(round(float(duration) * sample_rate)))
        return np.arange(size, dtype=np.float32) / float(sample_rate)

    if resolved_role == "kick":
        duration = 0.42 if target_bpm < 95 else 0.32
        t = _mono_time(duration)
        start_hz = min(max(sub_frequency_hz * 2.25, 82.0), 150.0)
        end_hz = max(sub_frequency_hz * 0.92, 42.0)
        if resolved_variant in {"alt", "tight"}:
            start_hz *= 1.08
            end_hz *= 1.04
        phase = 2.0 * np.pi * np.cumsum(np.linspace(start_hz, end_hz, t.size, dtype=np.float32)) / sample_rate
        body = np.sin(phase)
        sub = (0.18 if resolved_variant in {"alt", "tight"} else 0.24) * np.sin(2.0 * np.pi * (end_hz * 0.5) * t)
        envelope = np.exp(-t * 13.0) * (1.0 - np.exp(-t * 240.0))
        click_freq = 2150.0 if resolved_variant in {"alt", "tight"} else 1850.0
        click = (0.18 if ghost_variant else (0.22 if resolved_variant in {"alt", "tight"} else 0.16)) * np.sin(2.0 * np.pi * click_freq * t) * np.exp(-t * 95.0)
        thump = np.sin(2.0 * np.pi * max(34.0, end_hz * 0.72) * t + 0.2) * np.exp(-t * 11.5) * (0.12 if boombap_profile else 0.05)
        mono = np.tanh((body + sub + click + thump) * envelope * ((1.06 if ghost_variant else 1.4) if boombap_profile else (1.34 if resolved_variant in {"alt", "tight"} else 1.28)))
    elif resolved_role in {"snare", "clap"}:
        duration = 0.24 if resolved_role == "snare" else 0.28
        t = _mono_time(duration)
        noise = rng.standard_normal(t.size, dtype=np.float32)
        bright_noise = noise - np.concatenate((np.zeros(1, dtype=np.float32), noise[:-1] * 0.94))
        tone_freq = (210.0 if resolved_role == "snare" else 290.0) if resolved_variant == "alt" else (174.0 if ghost_variant else (190.0 if resolved_role == "snare" else 260.0))
        snap_freq = (3600.0 if resolved_role == "snare" else 3200.0) if resolved_variant == "alt" else (2100.0 if ghost_variant else (3200.0 if resolved_role == "snare" else 2800.0))
        tone = np.sin(2.0 * np.pi * tone_freq * t)
        snap = np.sin(2.0 * np.pi * snap_freq * t) * np.exp(-t * (88.0 if resolved_variant == "alt" else 75.0))
        envelope = np.exp(-t * ((20.0 if resolved_role == "snare" else 17.5) if resolved_variant == "alt" else ((13.0 if resolved_role == "snare" else 11.8) if ghost_variant else (18.0 if resolved_role == "snare" else 16.0))))
        dust = np.sin(2.0 * np.pi * 1100.0 * t + 0.37) * np.exp(-t * 26.0) * (0.08 if boombap_profile else 0.0)
        mono = np.tanh((bright_noise * envelope * ((0.46 if resolved_role == "snare" else 0.4) if ghost_variant else (0.88 if boombap_profile else (0.92 if resolved_variant == "alt" else 0.86)))) + (tone * np.exp(-t * 24.0) * (0.12 if ghost_variant else 0.24)) + (snap * (0.05 if ghost_variant else (0.14 if boombap_profile else (0.16 if resolved_variant == "alt" else 0.12)))) + (dust * (0.4 if ghost_variant else 1.0)))
        if resolved_role == "clap":
            delays = (0.009, 0.019, 0.031) if resolved_variant == "alt" else (0.012, 0.026)
            for delay_ms in delays:
                offset = int(round(delay_ms * sample_rate))
                mono[offset:] += bright_noise[: mono.size - offset] * np.exp(-(t[: mono.size - offset]) * 24.0) * 0.25
    elif resolved_role in {"hat", "hihat"}:
        duration = 0.12 if resolved_variant == "alt" else 0.09
        t = _mono_time(duration)
        noise = rng.standard_normal(t.size, dtype=np.float32)
        bright_noise = noise - np.concatenate((np.zeros(1, dtype=np.float32), noise[:-1] * 0.985))
        metallic = np.sin(2.0 * np.pi * (8200.0 if boombap_profile else (8800.0 if resolved_variant == "alt" else 7600.0)) * t) * (0.1 if resolved_variant == "alt" else 0.08)
        air = np.sin(2.0 * np.pi * (9600.0 if boombap_profile else (11200.0 if resolved_variant == "alt" else 9800.0)) * t + 0.18) * (0.015 if boombap_profile else (0.03 if resolved_variant == "alt" else 0.0))
        envelope = np.exp(-t * ((44.0 if "dark" not in style_tokens else 38.0) if boombap_profile else ((52.0 if "dark" not in style_tokens else 42.0) if resolved_variant == "alt" else (66.0 if "dark" not in style_tokens else 54.0))))
        mono = np.tanh((bright_noise + metallic + air) * envelope * (0.74 if boombap_profile else (0.82 if resolved_variant == "alt" else 0.76)))
    elif resolved_role == "bass":
        duration = max(0.28, min(0.62, (60.0 / max(float(target_bpm), 1.0)) * 0.72))
        t = _mono_time(duration)
        fundamental = max(42.0, min(sub_frequency_hz, 80.0))
        body = np.sin(2.0 * np.pi * fundamental * t)
        overtone = np.sin(2.0 * np.pi * fundamental * 2.0 * t + 0.3) * 0.22
        drive = np.sin(2.0 * np.pi * fundamental * 3.0 * t + 0.1) * ((0.12 if resolved_variant == "alt" else 0.09) if boombap_profile else (0.09 if resolved_variant == "alt" else 0.06))
        detune = np.sin(2.0 * np.pi * (fundamental * 0.5) * t + 0.28) * (0.13 if boombap_profile else 0.05)
        envelope = np.exp(-t * 6.2) * (1.0 - np.exp(-t * 65.0))
        mono = np.tanh((body + overtone + drive + detune) * envelope * ((1.34 if resolved_variant == "alt" else 1.28) if boombap_profile else (1.28 if resolved_variant == "alt" else 1.22)))
    elif resolved_role in {"melody", "lead"}:
        duration = 0.38
        t = _mono_time(duration)
        frequency = 220.0 if "dark" in style_tokens else 277.18
        body = np.sin(2.0 * np.pi * frequency * t)
        shimmer = np.sin(2.0 * np.pi * frequency * 2.0 * t) * (0.18 if boombap_profile else (0.21 if resolved_variant == "alt" else 0.17))
        air = np.sin(2.0 * np.pi * frequency * 4.0 * t + 0.2) * (0.03 if boombap_profile else (0.08 if resolved_variant == "alt" else 0.05))
        key_click = np.sin(2.0 * np.pi * (frequency * 8.0) * t + 0.6) * np.exp(-t * 58.0) * (0.09 if boombap_profile else 0.0)
        wobble = np.sin(2.0 * np.pi * 4.2 * t + 0.12) * (0.015 if boombap_profile else 0.0)
        envelope = np.exp(-t * (6.0 if boombap_profile else 7.2)) * (1.0 - np.exp(-t * 42.0))
        mono = np.tanh((body + shimmer + air + key_click + wobble) * envelope * (1.04 if boombap_profile else (1.12 if resolved_variant == "alt" else 1.08)))
    else:
        duration = 0.2
        t = _mono_time(duration)
        noise = rng.standard_normal(t.size, dtype=np.float32)
        envelope = np.exp(-t * 18.0)
        mono = noise * envelope * 0.35

    stereo = _stereo_from_mono(_normalize_audio_buffer(mono, peak=0.9))
    reference_blend = 0.0
    if reference_path and resolved_role in {"kick", "snare", "clap", "hat", "hihat"}:
        reference_shot = _extract_reference_drum_shot(
            reference_path=reference_path,
            sample_rate=sample_rate,
            role=resolved_role,
            style_tags=style_tags,
            variant_name=resolved_variant,
        )
        if reference_shot is not None:
            stereo = _hybridize_one_shot_with_reference(
                base_sample=stereo,
                reference_shot=reference_shot,
                role=resolved_role,
                variant_name=resolved_variant,
                style_tags=style_tags,
            )
            reference_blend = 1.0
    _write_audio_array_to_wav(stereo, output_path, sample_rate=sample_rate)
    return {
        "role": resolved_role,
        "variant_name": resolved_variant,
        "output_path": output_path,
        "duration_seconds": round(float(stereo.shape[0] / float(sample_rate)), 4),
        "peak": round(float(np.max(np.abs(stereo))) if stereo.size else 0.0, 6),
        "reference_path": _resolve_existing_path(reference_path) if reference_path and os.path.exists(reference_path) else None,
        "hybrid_reference_blend": bool(reference_blend),
    }


def _resolve_blueprint_render_role_defaults(role: str) -> Dict[str, float]:
    resolved_role = (_clean_text(role) or "sample").lower()
    defaults = {
        "kick": {"gain_db": 2.1, "pan": 0.0},
        "snare": {"gain_db": -0.4, "pan": 0.0},
        "clap": {"gain_db": -2.4, "pan": 0.03},
        "hat": {"gain_db": -6.1, "pan": 0.16},
        "hihat": {"gain_db": -6.1, "pan": 0.16},
        "perc": {"gain_db": -9.6, "pan": 0.1},
        "bass": {"gain_db": -1.6, "pan": 0.0},
        "melody": {"gain_db": -6.4, "pan": -0.12},
        "lead": {"gain_db": -6.0, "pan": -0.1},
        "texture": {"gain_db": -11.2, "pan": 0.0},
        "chop": {"gain_db": -7.2, "pan": 0.12},
        "drum_bed": {"gain_db": -14.4, "pan": 0.0},
        "resample": {"gain_db": -11.8, "pan": -0.06},
    }
    return defaults.get(resolved_role, {"gain_db": -4.0, "pan": 0.0})


def _resolve_project_blueprint_payload(
    project_id: str,
    blueprint_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = load_project_bundle(project_id)
    manifests = bundle.get("manifests") or {}
    beat_blueprints_manifest = manifests.get("beat_blueprints") or {}
    selected_blueprint = None
    desired_version_id = _clean_text(blueprint_version_id)

    if desired_version_id:
        for item in beat_blueprints_manifest.get("items", []):
            if _clean_text(item.get("version_id")) == desired_version_id:
                selected_blueprint = item
                break
    if selected_blueprint is None:
        selected_blueprint = bundle.get("active_beat_blueprint")
    if selected_blueprint is None:
        raise ValueError(f"Nenhum beat blueprint ativo foi encontrado para o projeto '{project_id}'.")

    from tools.audio_gen.project_registry import bootstrap_project_storage

    project_paths = bootstrap_project_storage(project_id)
    relative_path = _clean_text(selected_blueprint.get("relative_path"))
    artifact_path = _clean_text((selected_blueprint.get("metadata") or {}).get("artifact_path"))
    candidate_paths = []
    if relative_path:
        candidate_paths.append(os.path.join(project_paths.project_dir, relative_path))
    if artifact_path:
        candidate_paths.append(os.path.abspath(artifact_path))

    resolved_artifact_path = ""
    artifact_payload: Dict[str, Any] = {}
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            resolved_artifact_path = candidate
            artifact_payload = _safe_read_json(candidate)
            if artifact_payload:
                break

    if not artifact_payload:
        artifact_payload = {
            "label": selected_blueprint.get("label"),
            "summary": selected_blueprint.get("summary"),
            "target_bpm": selected_blueprint.get("target_bpm"),
            "bars": selected_blueprint.get("bars"),
            "steps_per_bar": selected_blueprint.get("steps_per_bar"),
            "style_tags": selected_blueprint.get("style_tags") or [],
            "recommended_layers": selected_blueprint.get("layers") or [],
            "arrangement_notes": selected_blueprint.get("arrangement_notes") or [],
            "mix_notes": selected_blueprint.get("mix_notes") or [],
            "reference_summaries": [],
            "external_reference_context": [],
        }

    return {
        "bundle": bundle,
        "blueprint_record": selected_blueprint,
        "artifact_path": resolved_artifact_path or None,
        "blueprint": artifact_payload,
    }


def _build_pattern_layers_from_blueprint(
    blueprint: Dict[str, Any],
    generated_samples_dir: str,
    sample_rate: int,
    total_bars: int,
    steps_per_bar: int,
    creativity: float,
    fill_probability: float,
    seed: int,
    add_supporting_bass: bool = True,
    add_melodic_motif: bool = True,
    add_reference_texture: bool = True,
    add_reference_chops: bool = True,
) -> Dict[str, Any]:
    os.makedirs(generated_samples_dir, exist_ok=True)

    target_bpm = float(blueprint.get("target_bpm") or 92.0)
    style_tags = _normalize_instrumental_style_tags(blueprint.get("style_tags"))
    boombap_profile = _uses_boombap_profile(style_tags)
    recommended_layers = list(blueprint.get("recommended_layers") or [])
    role_names = {_clean_text(item.get("role")).lower() for item in recommended_layers}
    sub_frequency_hz = _extract_reference_sub_frequency(blueprint)
    primary_reference_path = _resolve_primary_reference_audio_path(blueprint)
    sections = _build_arrangement_sections(total_bars)
    pitch_context = _extract_reference_pitch_context(blueprint, sub_frequency_hz)

    if add_supporting_bass and "bass" not in role_names:
        recommended_layers.append(
            {
                "name": "bass",
                "role": "bass",
                "pattern": _resolve_default_blueprint_pattern("bass", style_tags),
                "notes": "Camada de suporte adicionada automaticamente para sustentar o low-end do beat.",
                "metadata": {"auto_added": True},
            }
        )
    if "snare" in role_names and "clap" not in role_names and total_bars >= 8:
        recommended_layers.append(
            {
                "name": "clap",
                "role": "clap",
                "pattern": _resolve_default_blueprint_pattern("clap", style_tags),
                "notes": "Camada de clap adicionada automaticamente para reforcar o backbeat e dar mais corpo ao groove.",
                "metadata": {"auto_added": True},
            }
        )
    if role_names.intersection({"kick"}) and not any(_clean_text(item.get("name")).lower() == "kick-alt" for item in recommended_layers) and total_bars >= 16:
        recommended_layers.append(
            {
                "name": "kick-alt",
                "role": "kick",
                "pattern": "--------x--x-x--" if boombap_profile else "--------x---x---",
                "notes": "Camada complementar de kick adicionada automaticamente para reforcar pickups e retornos com uma variante de source material mais apertada.",
                "metadata": {"auto_added": True, "layer_variant": "alt", "render_mode": "one_shot_alt"},
            }
        )
    if any(_clean_text(item.get("role")).lower() == "clap" for item in recommended_layers) and not any(_clean_text(item.get("name")).lower() == "clap-alt" for item in recommended_layers) and total_bars >= 16:
        recommended_layers.append(
            {
                "name": "clap-alt",
                "role": "clap",
                "pattern": "------------x--x" if boombap_profile else "------------x---",
                "notes": "Variante complementar de clap adicionada automaticamente para acentos de retorno e secoes mais abertas.",
                "metadata": {"auto_added": True, "layer_variant": "alt", "render_mode": "one_shot_alt"},
            }
        )
    if any(_clean_text(item.get("role")).lower() == "snare" for item in recommended_layers) and not any(_clean_text(item.get("name")).lower() == "snare-alt" for item in recommended_layers) and total_bars >= 20:
        recommended_layers.append(
            {
                "name": "snare-alt",
                "role": "snare",
                "pattern": "----------x-x-x-" if boombap_profile else "------------x-x-",
                "notes": "Variante complementar de snare adicionada automaticamente para reforcar transicoes em arranjos mais longos.",
                "metadata": {"auto_added": True, "layer_variant": "alt", "render_mode": "one_shot_alt"},
            }
        )
    if role_names.intersection({"hat", "hihat"}) and not any(_clean_text(item.get("name")).lower() == "hat-alt" for item in recommended_layers) and total_bars >= 12:
        recommended_layers.append(
            {
                "name": "hat-alt",
                "role": "hat",
                "pattern": "-x-xx-x-x-xx-x-x" if boombap_profile else "-x-x-x-x-x-x-x-x",
                "notes": "Camada complementar de hat adicionada automaticamente para variar o source material local e reforcar o groove nas secoes principais.",
                "metadata": {"auto_added": True, "layer_variant": "alt", "render_mode": "one_shot_alt"},
            }
        )
    if any(_clean_text(item.get("role")).lower() == "snare" for item in recommended_layers) and not any(_clean_text(item.get("name")).lower() == "snare-ghost" for item in recommended_layers) and total_bars >= 12:
        recommended_layers.append(
            {
                "name": "snare-ghost",
                "role": "snare",
                "pattern": "------x-------x-",
                "notes": "Ghost snare automatica para empurrar o bounce do boombap sem deixar a bateria reta demais.",
                "metadata": {"auto_added": True, "layer_variant": "ghost", "render_mode": "one_shot_alt"},
            }
        )
    if add_melodic_motif and not role_names.intersection({"melody", "lead"}):
        recommended_layers.append(
            {
                "name": "melody",
                "role": "melody",
                "pattern": _resolve_default_blueprint_pattern("melody", style_tags),
                "notes": "Motivo melodico sintetizado automaticamente a partir do contexto tonal da referencia.",
                "metadata": {"auto_added": True, "render_mode": "phrase_track"},
            }
        )
    if add_reference_texture and blueprint.get("reference_summaries"):
        recommended_layers.append(
            {
                "name": "texture",
                "role": "texture",
                "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                "notes": "Textura derivada automaticamente da referencia principal para reduzir o timbre sintetico puro.",
                "metadata": {"auto_added": True, "render_mode": "reference_texture"},
            }
        )
        if total_bars >= 12:
            recommended_layers.append(
                {
                    "name": "perc-top",
                    "role": "perc",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Camada percussiva derivada automaticamente da referencia para adicionar top-loop e mais sensacao de source material real.",
                    "metadata": {"auto_added": True, "render_mode": "reference_perc_layer"},
                }
            )
            recommended_layers.append(
                {
                    "name": "drum-bed",
                    "role": "drum_bed",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Cama percussiva/room derivada da referencia para colar a bateria e trazer sujeira sampleada de boombap.",
                    "metadata": {"auto_added": True, "render_mode": "reference_drum_bed"},
                }
            )
    if add_reference_chops and blueprint.get("reference_summaries"):
        recommended_layers.append(
            {
                "name": "chop",
                "role": "chop",
                "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                "notes": "Chop ritmico derivado da referencia principal para trazer mais cola timbrica.",
                "metadata": {"auto_added": True, "render_mode": "reference_chop"},
            }
        )
        if total_bars >= 12:
            recommended_layers.append(
                {
                    "name": "chop-alt",
                    "role": "chop",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Variante complementar de chop derivada da referencia para adicionar mais personalidade ritmica.",
                    "metadata": {"auto_added": True, "render_mode": "reference_chop_alt", "layer_variant": "alt"},
                }
            )
        if total_bars >= 12:
            recommended_layers.append(
                {
                    "name": "chop-phrase",
                    "role": "chop",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Chop fraseado mais longo derivado da referencia para criar call/response sampleado entre secoes.",
                    "metadata": {"auto_added": True, "render_mode": "reference_chop_phrase", "layer_variant": "phrase"},
                }
            )
        if total_bars >= 12:
            recommended_layers.append(
                {
                    "name": "chop-answer",
                    "role": "chop",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Banco de resposta derivado da referencia para responder ao chop principal em espacos do groove.",
                    "metadata": {"auto_added": True, "render_mode": "reference_chop_answer", "layer_variant": "answer"},
                }
            )
            recommended_layers.append(
                {
                    "name": "chop-hook",
                    "role": "chop",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Banco de hook derivado da referencia para reforcar identidade no refrão.",
                    "metadata": {"auto_added": True, "render_mode": "reference_chop_hook", "layer_variant": "hook"},
                }
            )
            recommended_layers.append(
                {
                    "name": "chop-transition",
                    "role": "chop",
                    "pattern": _build_single_trigger_pattern(total_bars, steps_per_bar),
                    "notes": "Banco de transicao derivado da referencia para pickups e bordas de secao.",
                    "metadata": {"auto_added": True, "render_mode": "reference_chop_transition", "layer_variant": "transition"},
                }
            )

    if not recommended_layers:
        raise ValueError("O blueprint nao possui camadas recomendadas para renderizacao.")

    layers: List[PatternBeatLayer] = []
    synthesized_samples: List[Dict[str, Any]] = []
    for item in recommended_layers:
        role = _clean_text(item.get("role")) or "sample"
        pattern = _clean_text(item.get("pattern")) or "x---------------"
        name = _clean_text(item.get("name")) or role
        layer_variant = _clean_text((item.get("metadata") or {}).get("layer_variant")).lower()
        defaults = _resolve_blueprint_render_role_defaults(role)
        sample_path = os.path.join(generated_samples_dir, f"{_slugify(name)}_{_slugify(role)}.wav")
        layer_seed = (seed ^ (zlib.crc32(f"{name}:{role}".encode("utf-8")) & 0xFFFFFFFF)) & 0xFFFFFFFF
        if role.lower() == "texture":
            reference_path = ""
            for summary in blueprint.get("reference_summaries") or []:
                candidate = _clean_text(summary.get("path"))
                if candidate and os.path.exists(candidate):
                    reference_path = candidate
                    break
            if not reference_path:
                continue
            arranged_pattern = _build_single_trigger_pattern(total_bars, steps_per_bar)
            synthesized_samples.append(
                _extract_reference_texture_track(
                    reference_path=reference_path,
                    output_path=sample_path,
                    sample_rate=sample_rate,
                    tempo_bpm=target_bpm,
                    total_bars=total_bars,
                    sections=sections,
                    style_tags=style_tags,
                )
            )
        elif role.lower() == "drum_bed":
            reference_path = ""
            for summary in blueprint.get("reference_summaries") or []:
                candidate = _clean_text(summary.get("path"))
                if candidate and os.path.exists(candidate):
                    reference_path = candidate
                    break
            if not reference_path:
                continue
            arranged_pattern = _build_single_trigger_pattern(total_bars, steps_per_bar)
            synthesized_samples.append(
                _extract_reference_drum_bed_track(
                    reference_path=reference_path,
                    output_path=sample_path,
                    sample_rate=sample_rate,
                    tempo_bpm=target_bpm,
                    total_bars=total_bars,
                    sections=sections,
                    style_tags=style_tags,
                )
            )
        elif role.lower() == "perc":
            reference_path = ""
            for summary in blueprint.get("reference_summaries") or []:
                candidate = _clean_text(summary.get("path"))
                if candidate and os.path.exists(candidate):
                    reference_path = candidate
                    break
            if not reference_path:
                continue
            arranged_pattern = _build_reference_perc_pattern(
                sections=sections,
                steps_per_bar=steps_per_bar,
                creativity=creativity,
                seed=layer_seed,
            )
            synthesized_samples.append(
                _extract_reference_perc_layer(
                    reference_path=reference_path,
                    output_path=sample_path,
                    sample_rate=sample_rate,
                    tempo_bpm=target_bpm,
                    style_tags=style_tags,
                )
            )
        elif role.lower() == "chop":
            reference_path = ""
            for summary in blueprint.get("reference_summaries") or []:
                candidate = _clean_text(summary.get("path"))
                if candidate and os.path.exists(candidate):
                    reference_path = candidate
                    break
            if not reference_path:
                continue
            if layer_variant == "alt":
                arranged_pattern = _build_reference_counter_chop_pattern(
                    sections=sections,
                    steps_per_bar=steps_per_bar,
                    creativity=creativity,
                    seed=layer_seed,
                )
            elif layer_variant == "phrase":
                arranged_pattern = _build_reference_phrase_chop_pattern(
                    sections=sections,
                    steps_per_bar=steps_per_bar,
                    creativity=creativity,
                    seed=layer_seed,
                )
            elif layer_variant in {"answer", "hook", "transition"}:
                arranged_pattern = _build_reference_bank_chop_pattern(
                    bank_role=layer_variant,
                    sections=sections,
                    steps_per_bar=steps_per_bar,
                    creativity=creativity,
                    seed=layer_seed,
                )
            else:
                arranged_pattern = _build_reference_chop_pattern(
                    sections=sections,
                    steps_per_bar=steps_per_bar,
                    creativity=creativity,
                    seed=layer_seed,
                )
            synthesized_samples.append(
                _extract_reference_chop_sample(
                    reference_path=reference_path,
                    output_path=sample_path,
                    sample_rate=sample_rate,
                    tempo_bpm=target_bpm,
                    selection_rank=(
                        2 if layer_variant == "phrase"
                        else 1 if layer_variant == "alt"
                        else 2 if layer_variant == "answer"
                        else 0 if layer_variant == "hook"
                        else 3 if layer_variant == "transition"
                        else 0
                    ),
                    style_tags=style_tags,
                    chop_variant="phrase" if layer_variant == "phrase" else "alt" if layer_variant == "alt" else "stab",
                    bank_role=layer_variant or "main",
                )
            )
            if layer_variant == "alt":
                defaults = dict(defaults)
                defaults["gain_db"] = float(defaults["gain_db"]) - 2.1
                defaults["pan"] = -0.14
            elif layer_variant == "phrase":
                defaults = dict(defaults)
                defaults["gain_db"] = float(defaults["gain_db"]) - 3.4
                defaults["pan"] = -0.08
            elif layer_variant == "answer":
                defaults = dict(defaults)
                defaults["gain_db"] = float(defaults["gain_db"]) - 2.6
                defaults["pan"] = -0.2
            elif layer_variant == "hook":
                defaults = dict(defaults)
                defaults["gain_db"] = float(defaults["gain_db"]) - 1.2
                defaults["pan"] = 0.06
            elif layer_variant == "transition":
                defaults = dict(defaults)
                defaults["gain_db"] = float(defaults["gain_db"]) - 3.1
                defaults["pan"] = 0.18
        elif role.lower() in {"bass", "melody", "lead"}:
            arranged_pattern = "x" + ("-" * ((total_bars * steps_per_bar) - 1))
            synthesized_samples.append(
                _synthesize_phrase_track(
                    role=role,
                    output_path=sample_path,
                    sample_rate=sample_rate,
                    tempo_bpm=target_bpm,
                    sections=sections,
                    steps_per_bar=steps_per_bar,
                    pitch_context=pitch_context,
                    creativity=creativity,
                    seed=layer_seed,
                )
            )
        else:
            arranged_pattern = _compose_arranged_pattern(
                pattern=pattern,
                role=role,
                sections=sections,
                steps_per_bar=steps_per_bar,
                creativity=creativity,
                fill_probability=fill_probability,
                seed=layer_seed,
            )
            synthesized_samples.append(
                _synthesize_role_one_shot(
                    role=role,
                    output_path=sample_path,
                    sample_rate=sample_rate,
                    target_bpm=target_bpm,
                    sub_frequency_hz=sub_frequency_hz,
                    style_tags=style_tags,
                    variant_name=layer_variant,
                    reference_path=primary_reference_path or None,
                )
            )
        layer_gain_db = float(defaults["gain_db"])
        layer_pan = float(defaults["pan"])
        if layer_variant == "alt":
            if role.lower() in {"hat", "hihat"}:
                layer_gain_db -= 1.1
                layer_pan = -0.18 if layer_pan >= 0 else layer_pan
            elif role.lower() in {"snare", "clap", "kick"}:
                layer_gain_db -= 0.7
                layer_pan = -layer_pan if abs(layer_pan) > 1e-6 else -0.04
        elif layer_variant == "ghost":
            layer_gain_db -= 5.2
            if role.lower() == "snare":
                layer_pan = -0.08
        layers.append(
            PatternBeatLayer(
                audio_path=sample_path,
                pattern=arranged_pattern,
                name=name,
                role=role,
                gain_db=layer_gain_db,
                pan=layer_pan,
                metadata={
                    "sections": sections,
                    "layer_variant": layer_variant or None,
                    "render_mode": _clean_text((item.get("metadata") or {}).get("render_mode"))
                    or _clean_text((synthesized_samples[-1] if synthesized_samples else {}).get("render_mode"))
                    or None,
                    "section_contrast_enabled": True,
                },
            )
        )

    if total_bars >= 12:
        resample_sources = [
            item
            for item in synthesized_samples
            if _clean_text(item.get("role")).lower() in {"chop", "melody", "texture"}
            and _clean_text(item.get("output_path"))
            and os.path.exists(_clean_text(item.get("output_path")))
        ]
        if resample_sources:
            priority = {"chop": 0, "melody": 1, "texture": 2}
            resample_sources.sort(
                key=lambda item: (
                    priority.get(_clean_text(item.get("role")).lower(), 9),
                    -float(item.get("crate_score") or 0.0),
                    -float(item.get("peak") or 0.0),
                )
            )
            source_item = resample_sources[0]
            resample_path = os.path.join(generated_samples_dir, "resample_motif_resample.wav")
            resample_pattern = _build_internal_resample_pattern(
                sections=sections,
                steps_per_bar=steps_per_bar,
                creativity=creativity,
                seed=seed,
            )
            resample_manifest = _render_internal_resample_layer(
                source_path=_clean_text(source_item.get("output_path")),
                output_path=resample_path,
                sample_rate=sample_rate,
                tempo_bpm=target_bpm,
                total_bars=total_bars,
                sections=sections,
                style_tags=style_tags,
                source_role=_clean_text(source_item.get("role")),
            )
            synthesized_samples.append(resample_manifest)
            layers.append(
                PatternBeatLayer(
                    audio_path=resample_path,
                    pattern=resample_pattern,
                    name="resample-motif",
                    role="resample",
                    gain_db=float(_resolve_blueprint_render_role_defaults("resample")["gain_db"]),
                    pan=float(_resolve_blueprint_render_role_defaults("resample")["pan"]),
                    metadata={
                        "sections": sections,
                        "layer_variant": "motif",
                        "render_mode": "internal_resample_layer",
                        "source_role": _clean_text(source_item.get("role")).lower() or None,
                        "source_path": _clean_text(source_item.get("output_path")) or None,
                        "section_contrast_enabled": True,
                    },
                )
            )

    return {
        "layers": layers,
        "synthesized_samples": synthesized_samples,
        "sub_frequency_hz": sub_frequency_hz,
        "sections": sections,
        "pitch_context": pitch_context,
    }


def _resolve_pattern_section_for_step(
    step_index: int,
    steps_per_bar: int,
    sections: Optional[Iterable[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if not resolved_sections:
        return None
    bar_index = max(0, int(step_index)) // max(1, int(steps_per_bar))
    consumed = 0
    for section in resolved_sections:
        bars = max(1, int((section or {}).get("bars") or 1))
        if bar_index < consumed + bars:
            return dict(section)
        consumed += bars
    return dict(resolved_sections[-1])


def _resolve_pattern_section_mix_key(section: Optional[Dict[str, Any]]) -> str:
    song_role = _clean_text((section or {}).get("song_role")).lower()
    if song_role:
        return song_role
    return _clean_text((section or {}).get("name")).lower() or "main_a"


def _resolve_pattern_section_gain_db(role: str, section: Optional[Dict[str, Any]]) -> float:
    section_name = _resolve_pattern_section_mix_key(section)
    energy = float((section or {}).get("energy") or 0.92)
    drum_density = float((section or {}).get("drum_density") or 1.0)
    melodic_density = float((section or {}).get("melodic_density") or 1.0)
    chop_density = float((section or {}).get("chop_density") or 1.0)
    bass_density = float((section or {}).get("bass_density") or 1.0)
    resolved_role = (_clean_text(role) or "sample").lower()

    role_section_gains = {
        "kick": {"intro": -1.2, "verse": 0.0, "pre_hook": 0.2, "hook": 0.4, "break": -1.7, "outro": -1.25},
        "snare": {"intro": -1.0, "verse": -0.05, "pre_hook": -0.1, "hook": 0.15, "break": -0.95, "outro": -0.8},
        "clap": {"intro": -1.45, "verse": -0.24, "pre_hook": 0.04, "hook": 0.3, "break": -1.3, "outro": -0.95},
        "hat": {"intro": -2.3, "verse": -0.28, "pre_hook": 0.62, "hook": 0.35, "break": -2.8, "outro": -1.9},
        "hihat": {"intro": -2.3, "verse": -0.28, "pre_hook": 0.62, "hook": 0.35, "break": -2.8, "outro": -1.9},
        "perc": {"intro": -1.8, "verse": -0.25, "pre_hook": 0.18, "hook": 0.28, "break": -1.9, "outro": -1.2},
        "bass": {"intro": -0.75, "verse": 0.0, "pre_hook": -0.18, "hook": 0.22, "break": -1.45, "outro": -0.9},
        "melody": {"intro": -1.0, "verse": -0.22, "pre_hook": -0.34, "hook": 0.22, "break": -1.8, "outro": -0.95},
        "lead": {"intro": -1.1, "verse": -0.18, "pre_hook": -0.28, "hook": 0.28, "break": -1.65, "outro": -0.88},
        "texture": {"intro": 0.2, "verse": -0.25, "pre_hook": -0.52, "hook": -0.4, "break": 0.42, "outro": 0.18},
        "chop": {"intro": -2.0, "verse": -0.32, "pre_hook": 0.48, "hook": 0.25, "break": -2.45, "outro": -1.85},
        "drum_bed": {"intro": -1.8, "verse": -0.25, "pre_hook": 0.22, "hook": 0.34, "break": -2.1, "outro": -1.45},
        "resample": {"intro": -2.4, "verse": -0.42, "pre_hook": 0.16, "hook": 0.24, "break": -2.0, "outro": -1.7},
    }
    gain_db = float((role_section_gains.get(resolved_role) or {}).get(section_name, 0.0))

    if resolved_role in {"kick", "snare", "clap", "hat", "hihat", "perc", "drum_bed"}:
        gain_db += (drum_density - 0.92) * 1.8
    elif resolved_role == "bass":
        gain_db += (bass_density - 0.94) * 1.5
    elif resolved_role in {"melody", "lead"}:
        gain_db += (melodic_density - 0.88) * 1.7
    elif resolved_role == "chop":
        gain_db += (chop_density - 0.84) * 1.7
    elif resolved_role == "texture":
        gain_db += (0.78 - energy) * 0.9
    elif resolved_role == "resample":
        gain_db += (chop_density - 0.84) * 1.2
    else:
        gain_db += (energy - 0.9) * 0.6

    return round(float(gain_db), 4)


def _apply_pattern_kick_bass_sidechain(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sample_rate: int,
) -> Optional[Dict[str, Any]]:
    kick_events: List[Dict[str, Any]] = []
    bass_layer_names: List[str] = []
    for layer_manifest in rendered_layers:
        role_name = _clean_text(layer_manifest.get("role")).lower()
        if role_name == "kick":
            kick_events.extend(list(layer_manifest.get("trigger_positions") or []))
        elif role_name == "bass":
            layer_name = _clean_text(layer_manifest.get("name"))
            if layer_name and layer_name in layer_renders:
                bass_layer_names.append(layer_name)

    if not kick_events or not bass_layer_names:
        return None

    section_target_gains = {
        "intro": 0.84,
        "verse": 0.76,
        "pre_hook": 0.81,
        "hook": 0.71,
        "break": 0.88,
        "outro": 0.85,
    }
    section_hold_ms = {
        "intro": 34.0,
        "verse": 44.0,
        "pre_hook": 30.0,
        "hook": 52.0,
        "break": 24.0,
        "outro": 34.0,
    }
    section_release_ms = {
        "intro": 96.0,
        "verse": 112.0,
        "pre_hook": 86.0,
        "hook": 126.0,
        "break": 74.0,
        "outro": 92.0,
    }
    total_samples = max(render.shape[0] for render in layer_renders.values())
    envelope = np.ones(total_samples, dtype=np.float32)
    sidechain_events: List[Dict[str, Any]] = []
    attack_samples = max(8, int(round(sample_rate * 0.004)))

    for event in kick_events:
        start_seconds = float(event.get("start_seconds") or 0.0)
        section_name = _clean_text(event.get("song_role") or event.get("section_name")).lower() or "verse"
        start_index = max(0, min(total_samples, int(round(start_seconds * sample_rate))))
        hold_samples = max(16, int(round((section_hold_ms.get(section_name, 42.0) / 1000.0) * sample_rate)))
        release_samples = max(24, int(round((section_release_ms.get(section_name, 108.0) / 1000.0) * sample_rate)))
        target_gain = float(section_target_gains.get(section_name, 0.76))
        attack_start = max(0, start_index - attack_samples)
        if start_index > attack_start:
            attack_curve = np.linspace(1.0, target_gain, start_index - attack_start, dtype=np.float32)
            envelope[attack_start:start_index] = np.minimum(envelope[attack_start:start_index], attack_curve)
        hold_end = min(total_samples, start_index + hold_samples)
        envelope[start_index:hold_end] = np.minimum(envelope[start_index:hold_end], target_gain)
        release_end = min(total_samples, hold_end + release_samples)
        if release_end > hold_end:
            release_curve = np.linspace(target_gain, 1.0, release_end - hold_end, dtype=np.float32)
            envelope[hold_end:release_end] = np.minimum(envelope[hold_end:release_end], release_curve)
        sidechain_events.append(
            {
                "start_seconds": round(float(start_seconds), 4),
                "section_name": section_name or None,
                "target_gain": round(float(target_gain), 4),
                "hold_ms": round(float(section_hold_ms.get(section_name, 42.0)), 2),
                "release_ms": round(float(section_release_ms.get(section_name, 108.0)), 2),
            }
        )

    smoothing_samples = max(16, int(round(sample_rate * 0.006)))
    if envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            envelope = np.convolve(envelope, kernel, mode="same").astype(np.float32)
    envelope = np.clip(envelope, 0.66, 1.0)

    for layer_name in bass_layer_names:
        render = layer_renders.get(layer_name)
        if render is None:
            continue
        usable = envelope[: render.shape[0]]
        layer_renders[layer_name] = render * usable[:, np.newaxis]

    return {
        "mode": "kick_bass_sidechain",
        "kick_events_count": len(kick_events),
        "bass_layers_count": len(bass_layer_names),
        "events_preview": sidechain_events[:48],
        "min_gain": round(float(np.min(envelope)) if envelope.size else 1.0, 4),
        "max_gain": round(float(np.max(envelope)) if envelope.size else 1.0, 4),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_pattern_drum_bus_glue(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sample_rate: int,
) -> Optional[Dict[str, Any]]:
    if np is None or not layer_renders:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    core_drum_layers = _select_drum_character_layers(
        role_by_name=role_by_name,
        available_layers=[name for name, role in role_by_name.items() if role in {"kick", "snare", "clap", "hat", "hihat", "perc", "drum_bed"} and name in layer_renders],
    )
    drum_layer_names: List[str] = list(core_drum_layers)
    for preferred_role in ("drum_variation", "drum_character", "drum_fill"):
        for layer_name, role_name in role_by_name.items():
            if role_name == preferred_role and layer_name in layer_renders and layer_name not in drum_layer_names:
                drum_layer_names.append(layer_name)
                break
    if len(drum_layer_names) < 2:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in drum_layer_names)
    dry_bus = np.zeros((total_samples, 2), dtype=np.float32)
    for name in drum_layer_names:
        render = layer_renders.get(name)
        if render is None or render.size == 0:
            continue
        dry_bus[: render.shape[0]] += render

    bus = dry_bus.copy()
    if signal is not None and bus.shape[0] > 2048:
        try:
            low_sos = signal.butter(2, [55.0, 155.0], btype="bandpass", fs=sample_rate, output="sos")
            body_sos = signal.butter(2, [180.0, 950.0], btype="bandpass", fs=sample_rate, output="sos")
            snap_sos = signal.butter(2, [1200.0, 5200.0], btype="bandpass", fs=sample_rate, output="sos")
            shaped = np.zeros_like(bus)
            mono = np.mean(bus, axis=1).astype(np.float32)
            low = signal.sosfiltfilt(low_sos, mono).astype(np.float32)
            body = signal.sosfiltfilt(body_sos, mono).astype(np.float32)
            snap = signal.sosfilt(snap_sos, mono).astype(np.float32)
            shaped_mono = ((low * 0.5) + (body * 0.36) + (snap * 0.14)).astype(np.float32)
            shaped[:, 0] = shaped_mono
            shaped[:, 1] = shaped_mono
            bus = (bus * 0.34) + shaped
        except Exception:
            pass

    abs_bus = np.mean(np.abs(bus), axis=1).astype(np.float32)
    if abs_bus.size:
        threshold = float(np.percentile(abs_bus, 72))
        if threshold > 0:
            gain = np.ones(abs_bus.shape[0], dtype=np.float32)
            over = abs_bus > threshold
            gain[over] = np.power(np.maximum(threshold / np.maximum(abs_bus[over], 1e-6), 0.0), 0.42).astype(np.float32)
            smoothing_samples = max(32, int(round(sample_rate * 0.008)))
            if gain.size > smoothing_samples:
                kernel = np.hanning(smoothing_samples).astype(np.float32)
                kernel_sum = float(np.sum(kernel))
                if kernel_sum > 0:
                    kernel /= kernel_sum
                    gain = np.convolve(gain, kernel, mode="same").astype(np.float32)
            bus *= np.clip(gain, 0.56, 1.08)[:, np.newaxis]

    parallel = np.tanh(bus * 1.72).astype(np.float32) * 0.18
    if signal is not None and parallel.shape[0] > 2048:
        try:
            hp_sos = signal.butter(2, 32.0, btype="highpass", fs=sample_rate, output="sos")
            lp_sos = signal.butter(2, 9800.0, btype="lowpass", fs=sample_rate, output="sos")
            mono = np.mean(parallel, axis=1).astype(np.float32)
            mono = signal.sosfilt(hp_sos, mono).astype(np.float32)
            mono = signal.sosfilt(lp_sos, mono).astype(np.float32)
            parallel[:, 0] = mono
            parallel[:, 1] = mono
        except Exception:
            pass
    parallel = np.clip(parallel, -0.42, 0.42).astype(np.float32)
    layer_name = "drum-glue-bus"
    layer_renders[layer_name] = parallel
    rendered_layers.append(
        {
            "name": layer_name,
            "role": "drum_bus",
            "pattern": None,
            "trigger_positions": [],
            "source_path": None,
            "copied_input_path": None,
            "render_mode": "parallel_drum_bus_glue",
        }
    )

    return {
        "mode": "parallel_drum_bus_glue",
        "profile": "boombap_parallel_sp1200_bus_v1",
        "drum_layers": drum_layer_names,
        "drum_layers_count": len(drum_layer_names),
        "parallel_layer_name": layer_name,
        "parallel_peak": round(float(np.max(np.abs(parallel))) if parallel.size else 0.0, 6),
        "dry_bus_peak": round(float(np.max(np.abs(dry_bus))) if dry_bus.size else 0.0, 6),
    }


def _apply_pattern_motif_sample_glue(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sample_rate: int,
) -> Optional[Dict[str, Any]]:
    if np is None or not layer_renders:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    lead_layers = [name for name, role in role_by_name.items() if role in {"melody", "lead"} and name in layer_renders]
    support_layers = [name for name, role in role_by_name.items() if role in {"chop", "texture", "resample"} and name in layer_renders]
    if not lead_layers or not support_layers:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in lead_layers + support_layers)
    lead_envelope = np.zeros(total_samples, dtype=np.float32)
    for name in lead_layers:
        render = layer_renders.get(name)
        if render is None or render.size == 0:
            continue
        mono = np.mean(np.abs(render), axis=1).astype(np.float32)
        lead_envelope[: mono.shape[0]] = np.maximum(lead_envelope[: mono.shape[0]], mono)
    if not np.any(lead_envelope > 0):
        return None

    smoothing_samples = max(32, int(round(sample_rate * 0.02)))
    if lead_envelope.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            lead_envelope = np.convolve(lead_envelope, kernel, mode="same").astype(np.float32)
    reference_level = float(np.percentile(lead_envelope, 86)) if np.any(lead_envelope > 0) else 0.0
    if reference_level <= 1e-6:
        return None
    lead_envelope = np.clip(lead_envelope / reference_level, 0.0, 1.0).astype(np.float32)

    duck_amounts = {
        "texture": 0.18,
        "chop": 0.12,
        "resample": 0.16,
    }
    events: List[Dict[str, Any]] = []
    for layer_name in support_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        role = role_by_name.get(layer_name, "")
        duck_amount = float(duck_amounts.get(role, 0.1))
        usable = lead_envelope[: render.shape[0]]
        gain = (1.0 - (usable * duck_amount)).astype(np.float32)
        if role == "resample":
            gain = np.clip(gain + 0.02, 0.78, 1.02).astype(np.float32)
        elif role == "texture":
            gain = np.clip(gain, 0.74, 1.0).astype(np.float32)
        else:
            gain = np.clip(gain, 0.8, 1.0).astype(np.float32)
        processed = render * gain[:, np.newaxis]
        if role in {"texture", "resample"}:
            mid = ((processed[:, 0] + processed[:, 1]) * 0.5).astype(np.float32)
            side = ((processed[:, 0] - processed[:, 1]) * 0.5).astype(np.float32)
            width_push = 1.0 + (usable * (0.04 if role == "texture" else 0.06))
            side *= width_push[: side.shape[0]]
            processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        events.append(
            {
                "layer_name": layer_name,
                "role": role,
                "duck_amount": round(float(duck_amount), 4),
                "min_gain": round(float(np.min(gain)) if gain.size else 1.0, 4),
                "max_gain": round(float(np.max(gain)) if gain.size else 1.0, 4),
            }
        )

    return {
        "mode": "motif_sample_glue",
        "profile": "motif_sample_glue_v1",
        "lead_layers": lead_layers,
        "support_layers": support_layers,
        "events": events,
        "events_count": len(events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_pattern_topline_signature(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    if np is None or signal is None or not layer_renders:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    candidate_layers = _select_sample_stack_focus_layers(
        role_by_name=role_by_name,
        available_layers=[
            name
            for name, role in role_by_name.items()
            if role in {"melody", "lead", "chop", "texture", "resample"} and name in layer_renders
        ],
    )
    lead_layers = [name for name in candidate_layers if role_by_name.get(name) in {"melody", "lead"}][:1]
    hook_layers = [
        name
        for name in candidate_layers
        if role_by_name.get(name) == "chop" and "hook" in _clean_text(name).lower()
    ][:1]
    resample_layers = [name for name in candidate_layers if role_by_name.get(name) == "resample"][:1]
    texture_layers = [name for name in candidate_layers if role_by_name.get(name) == "texture"][:1]
    if not lead_layers or not (hook_layers or resample_layers or texture_layers):
        return None

    source_layers = lead_layers + hook_layers + resample_layers + texture_layers
    total_samples = max(layer_renders[name].shape[0] for name in source_layers if name in layer_renders)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    smoothing_samples = max(96, int(round(sample_rate * 0.028)))
    nyquist_hz = sample_rate * 0.5

    def _smooth_envelope(values: "np.ndarray") -> "np.ndarray":
        if values.size <= smoothing_samples:
            return values.astype(np.float32)
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum <= 0:
            return values.astype(np.float32)
        kernel /= kernel_sum
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    def _layer_envelope(layer_names: List[str]) -> "np.ndarray":
        envelope = np.zeros(total_samples, dtype=np.float32)
        for layer_name in layer_names:
            render = layer_renders.get(layer_name)
            if render is None or render.size == 0:
                continue
            mono = np.mean(np.abs(render), axis=1).astype(np.float32)
            envelope[: mono.shape[0]] = np.maximum(envelope[: mono.shape[0]], mono)
        if not np.any(envelope > 0):
            return envelope
        envelope = _smooth_envelope(envelope)
        reference_level = float(np.percentile(envelope, 84)) if np.any(envelope > 0) else 0.0
        if reference_level <= 1e-6:
            return np.zeros_like(envelope)
        return np.clip(envelope / reference_level, 0.0, 1.0).astype(np.float32)

    lead_envelope = _layer_envelope(lead_layers)
    hook_envelope = _layer_envelope(hook_layers)
    if not np.any(lead_envelope > 0):
        return None

    section_focus = np.zeros(total_samples, dtype=np.float32)
    current_bar = 0
    section_focus_map = {
        "intro": 0.22,
        "verse": 0.36,
        "pre_hook": 0.72,
        "hook": 1.0,
        "break": 0.24,
        "outro": 0.18,
    }
    for section in list(sections or []):
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        start_index = min(total_samples, current_bar * samples_per_bar)
        end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
        if end_index > start_index:
            section_focus[start_index:end_index] = float(section_focus_map.get(section_name, 0.4))
        current_bar += bars
    section_focus = _smooth_envelope(section_focus)
    support_focus = np.maximum(section_focus, np.maximum(lead_envelope * 0.5, hook_envelope * 0.95)).astype(np.float32)
    support_focus = np.clip(support_focus, 0.0, 1.0).astype(np.float32)

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(40.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    lead_body_sos = _bandpass(260.0, 1280.0)
    lead_presence_sos = _bandpass(1800.0, 4200.0)
    hook_body_sos = _bandpass(320.0, 1700.0)
    hook_bite_sos = _bandpass(2100.0, 5200.0)
    resample_focus_sos = _bandpass(900.0, 3400.0)
    resample_air_sos = _bandpass(4200.0, 8400.0)
    texture_mud_sos = _bandpass(220.0, 720.0)
    texture_air_sos = _bandpass(3200.0, 7800.0)

    events: List[Dict[str, Any]] = []
    processed_layers = 0

    def _width_process(render: "np.ndarray", width_env: "np.ndarray") -> "np.ndarray":
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32)
        side *= width_env[: side.shape[0]]
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    def _apply_mono_band_mix(
        render: "np.ndarray",
        body_sos: Optional[Any],
        body_gain: "np.ndarray",
        presence_sos: Optional[Any] = None,
        presence_gain: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        processed = render.copy()
        mono = np.mean(processed, axis=1).astype(np.float32)
        adjustment = np.zeros(mono.shape[0], dtype=np.float32)
        if body_sos is not None:
            adjustment += signal.sosfiltfilt(body_sos, mono).astype(np.float32) * body_gain[: mono.shape[0]]
        if presence_sos is not None and presence_gain is not None:
            adjustment += signal.sosfiltfilt(presence_sos, mono).astype(np.float32) * presence_gain[: mono.shape[0]]
        processed[:, 0] += adjustment
        processed[:, 1] += adjustment
        return processed.astype(np.float32)

    for layer_name in lead_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0 or lead_body_sos is None or lead_presence_sos is None:
            continue
        layer_focus = np.clip(np.maximum(section_focus[: render.shape[0]], hook_envelope[: render.shape[0]]), 0.0, 1.0).astype(np.float32)
        body_lift = 0.012 + (layer_focus * 0.018)
        presence_tuck = -(0.012 + (hook_envelope[: render.shape[0]] * 0.034))
        processed = _apply_mono_band_mix(render, lead_body_sos, body_lift, lead_presence_sos, presence_tuck)
        processed = _width_process(processed, 1.0 - (layer_focus * 0.03))
        processed *= (0.997 + (section_focus[: processed.shape[0]] * 0.008))[:, np.newaxis]
        processed = np.tanh(processed * (1.0 + (layer_focus[:, np.newaxis] * 0.035))).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        processed_layers += 1
        events.append(
            {
                "layer_name": layer_name,
                "role": "lead",
                "body_lift_max": round(float(np.max(0.018 + (layer_focus * 0.03))), 4),
                "presence_tuck_max": round(float(np.max(0.012 + (hook_envelope[: layer_focus.shape[0]] * 0.034))), 4),
                "width_min": round(float(np.min(1.0 - (layer_focus * 0.05))), 4),
            }
        )

    for layer_name in hook_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0 or hook_body_sos is None or hook_bite_sos is None:
            continue
        layer_focus = np.clip(np.maximum(support_focus[: render.shape[0]], hook_envelope[: render.shape[0]]), 0.0, 1.0).astype(np.float32)
        processed = _apply_mono_band_mix(
            render,
            hook_body_sos,
            0.014 + (layer_focus * 0.022),
            hook_bite_sos,
            0.018 + (layer_focus * 0.036),
        )
        processed = _width_process(processed, 1.0 + (layer_focus * 0.14))
        processed = np.tanh(processed * (1.0 + (layer_focus[:, np.newaxis] * 0.055))).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        processed_layers += 1
        events.append(
            {
                "layer_name": layer_name,
                "role": "hook_chop",
                "bite_push_max": round(float(np.max(0.018 + (layer_focus * 0.036))), 4),
                "width_max": round(float(np.max(1.0 + (layer_focus * 0.14))), 4),
            }
        )

    for layer_name in resample_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0 or resample_focus_sos is None or resample_air_sos is None:
            continue
        layer_focus = np.clip(np.maximum(support_focus[: render.shape[0]], hook_envelope[: render.shape[0]] * 1.05), 0.0, 1.0).astype(np.float32)
        processed = _apply_mono_band_mix(
            render,
            resample_focus_sos,
            0.012 + (layer_focus * 0.028),
            resample_air_sos,
            0.002 + (layer_focus * 0.008),
        )
        processed = _width_process(processed, 1.0 + (layer_focus * 0.12))
        processed = np.tanh(processed * (1.0 + (layer_focus[:, np.newaxis] * 0.05))).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        processed_layers += 1
        events.append(
            {
                "layer_name": layer_name,
                "role": "resample",
                "focus_push_max": round(float(np.max(0.012 + (layer_focus * 0.028))), 4),
                "width_max": round(float(np.max(1.0 + (layer_focus * 0.16))), 4),
            }
        )

    for layer_name in texture_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0 or texture_mud_sos is None or texture_air_sos is None:
            continue
        layer_focus = np.clip(np.maximum(section_focus[: render.shape[0]] * 0.85, lead_envelope[: render.shape[0]] * 0.6), 0.0, 1.0).astype(np.float32)
        processed = _apply_mono_band_mix(
            render,
            texture_mud_sos,
            -(0.03 + (lead_envelope[: render.shape[0]] * 0.02)),
            texture_air_sos,
            0.004 + (layer_focus * 0.014),
        )
        processed = _width_process(processed, 1.0 + (layer_focus * 0.1))
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        processed_layers += 1
        events.append(
            {
                "layer_name": layer_name,
                "role": "texture",
                "mud_tuck_max": round(float(np.max(0.03 + (lead_envelope[: layer_focus.shape[0]] * 0.02))), 4),
                "air_push_max": round(float(np.max(0.004 + (layer_focus * 0.014))), 4),
                "width_max": round(float(np.max(1.0 + (layer_focus * 0.1))), 4),
            }
        )

    if processed_layers <= 0:
        return None

    return {
        "mode": "motif_topline_signature",
        "profile": "motif_topline_signature_v1",
        "lead_layers": lead_layers,
        "hook_layers": hook_layers,
        "resample_layers": resample_layers,
        "texture_layers": texture_layers,
        "candidate_layers": candidate_layers,
        "processed_layers_count": int(processed_layers),
        "events": events,
        "events_count": len(events),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_pattern_tonal_memory_bus(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    if np is None or signal is None or not layer_renders:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    candidate_layers = _select_representative_sample_layers(
        role_by_name=role_by_name,
        available_layers=[
            name
            for name, role in role_by_name.items()
            if role in {"melody", "lead", "chop", "texture", "resample"} and name in layer_renders
        ],
    )
    lead_layers = [name for name in candidate_layers if role_by_name.get(name) in {"melody", "lead"}][:2]
    hook_layers = [
        name
        for name in candidate_layers
        if role_by_name.get(name) == "chop" and "hook" in _clean_text(name).lower()
    ][:2]
    resample_layers = [name for name in candidate_layers if role_by_name.get(name) == "resample"][:1]
    texture_layers = [name for name in candidate_layers if role_by_name.get(name) == "texture"][:1]
    source_layers = lead_layers + hook_layers + resample_layers + texture_layers
    if len(source_layers) < 2:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in source_layers)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    smoothing_samples = max(128, int(round(sample_rate * 0.035)))
    nyquist_hz = sample_rate * 0.5

    def _smooth(values: "np.ndarray") -> "np.ndarray":
        if values.size <= smoothing_samples:
            return values.astype(np.float32)
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum <= 0:
            return values.astype(np.float32)
        kernel /= kernel_sum
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    def _envelope_for(layer_names: List[str]) -> "np.ndarray":
        envelope = np.zeros(total_samples, dtype=np.float32)
        for layer_name in layer_names:
            render = layer_renders.get(layer_name)
            if render is None or render.size == 0:
                continue
            mono = np.mean(np.abs(render), axis=1).astype(np.float32)
            envelope[: mono.shape[0]] = np.maximum(envelope[: mono.shape[0]], mono)
        if not np.any(envelope > 0):
            return envelope
        envelope = _smooth(envelope)
        reference_level = float(np.percentile(envelope, 84)) if np.any(envelope > 0) else 0.0
        if reference_level <= 1e-6:
            return np.zeros_like(envelope)
        return np.clip(envelope / reference_level, 0.0, 1.0).astype(np.float32)

    lead_envelope = _envelope_for(lead_layers)
    hook_envelope = _envelope_for(hook_layers)
    resample_envelope = _envelope_for(resample_layers)
    texture_envelope = _envelope_for(texture_layers)
    if not np.any(lead_envelope > 0):
        return None

    section_bloom = np.zeros(total_samples, dtype=np.float32)
    current_bar = 0
    section_bloom_map = {
        "intro": 0.16,
        "verse": 0.28,
        "pre_hook": 0.68,
        "hook": 1.0,
        "break": 0.24,
        "outro": 0.12,
    }
    for section in list(sections or []):
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        start_index = min(total_samples, current_bar * samples_per_bar)
        end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
        if end_index > start_index:
            section_bloom[start_index:end_index] = float(section_bloom_map.get(section_name, 0.34))
        current_bar += bars
    section_bloom = _smooth(section_bloom)

    bus = np.zeros((total_samples, 2), dtype=np.float32)
    role_gain_map = {
        "lead": 0.3,
        "hook": 0.36,
        "resample": 0.26,
        "texture": 0.16,
    }

    for layer_name in lead_layers:
        render = layer_renders.get(layer_name)
        if render is not None and render.size:
            bus[: render.shape[0]] += render * role_gain_map["lead"]
    for layer_name in hook_layers:
        render = layer_renders.get(layer_name)
        if render is not None and render.size:
            bus[: render.shape[0]] += render * role_gain_map["hook"]
    for layer_name in resample_layers:
        render = layer_renders.get(layer_name)
        if render is not None and render.size:
            bus[: render.shape[0]] += render * role_gain_map["resample"]
    for layer_name in texture_layers:
        render = layer_renders.get(layer_name)
        if render is not None and render.size:
            bus[: render.shape[0]] += render * role_gain_map["texture"]

    if not np.any(np.abs(bus) > 1e-7):
        return None

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(45.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    body_sos = _bandpass(240.0, 1680.0)
    presence_sos = _bandpass(1700.0, 5200.0)
    air_sos = _bandpass(4600.0, 9200.0)
    if body_sos is None or presence_sos is None or air_sos is None:
        return None

    focus_envelope = np.maximum(
        section_bloom,
        np.maximum(hook_envelope * 1.05, np.maximum(resample_envelope * 0.82, lead_envelope * 0.42)),
    ).astype(np.float32)
    focus_envelope = np.clip(focus_envelope, 0.0, 1.0).astype(np.float32)

    mono = np.mean(bus, axis=1).astype(np.float32)
    body_band = signal.sosfiltfilt(body_sos, mono).astype(np.float32)
    presence_band = signal.sosfiltfilt(presence_sos, mono).astype(np.float32)
    air_band = signal.sosfiltfilt(air_sos, mono).astype(np.float32)
    colored_mono = (
        (body_band * (0.52 + (focus_envelope * 0.1)))
        + (presence_band * (0.24 + (focus_envelope * 0.12)))
        + (air_band * (0.08 + (focus_envelope * 0.04)))
    ).astype(np.float32)
    colored = np.stack([colored_mono, colored_mono], axis=1).astype(np.float32)

    lead_pull = np.clip(0.92 - (lead_envelope * 0.12), 0.76, 0.94).astype(np.float32)
    texture_push = 1.0 + (texture_envelope * 0.08)
    tonal_bus = colored * (focus_envelope[:, np.newaxis] * 0.36 + 0.06)
    tonal_bus *= lead_pull[:, np.newaxis]
    tonal_bus *= texture_push[: tonal_bus.shape[0], np.newaxis]
    tonal_bus = np.tanh(tonal_bus * (1.12 + (focus_envelope[:, np.newaxis] * 0.22))).astype(np.float32)

    mid = ((tonal_bus[:, 0] + tonal_bus[:, 1]) * 0.5).astype(np.float32)
    side = ((tonal_bus[:, 0] - tonal_bus[:, 1]) * 0.5).astype(np.float32)
    side *= (1.04 + (focus_envelope * 0.24))
    tonal_bus = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
    tonal_bus = np.clip(tonal_bus, -0.34, 0.34).astype(np.float32)

    layer_name = "motif-tonal-memory"
    layer_renders[layer_name] = tonal_bus
    rendered_layers.append(
        {
            "name": layer_name,
            "role": "tonal_glue",
            "pattern": None,
            "trigger_positions": [],
            "source_path": None,
            "copied_input_path": None,
            "render_mode": "motif_tonal_memory_bus",
        }
    )

    return {
        "mode": "motif_tonal_memory_bus",
        "profile": "motif_tonal_memory_bus_v1",
        "candidate_layers": candidate_layers,
        "lead_layers": lead_layers,
        "hook_layers": hook_layers,
        "resample_layers": resample_layers,
        "texture_layers": texture_layers,
        "bus_layer_name": layer_name,
        "bus_peak": round(float(np.max(np.abs(tonal_bus))) if tonal_bus.size else 0.0, 6),
        "focus_peak": round(float(np.max(focus_envelope)) if focus_envelope.size else 0.0, 6),
        "section_bloom_peak": round(float(np.max(section_bloom)) if section_bloom.size else 0.0, 6),
        "events_count": int(len(source_layers)),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_pattern_ensemble_section_contrast(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or signal is None or not layer_renders or not resolved_sections:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    drum_roles = {"kick", "snare", "clap", "hat", "hihat", "perc", "drum_bed"}
    drum_width_roles = {"snare", "clap", "hat", "hihat", "perc", "drum_bed"}
    sample_roles = {"melody", "lead", "chop", "texture", "resample", "tonal_glue"}
    bass_layers = [name for name, role in role_by_name.items() if role == "bass" and name in layer_renders]
    selected_drum_layers = set(
        _select_drum_character_layers(
            role_by_name=role_by_name,
            available_layers=[name for name, role in role_by_name.items() if role in drum_roles and name in layer_renders],
        )
    )
    representative_sample_layers = _select_sample_stack_focus_layers(
        role_by_name=role_by_name,
        available_layers=[name for name, role in role_by_name.items() if role in sample_roles and name in layer_renders],
    )
    selected_sample_layers = set(representative_sample_layers[:3])
    focus_layer_names = list(selected_drum_layers) + bass_layers + list(selected_sample_layers)
    total_samples = max(layer_renders[name].shape[0] for name in focus_layer_names if name in layer_renders)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    smoothing_samples = max(128, int(round(sample_rate * 0.04)))
    nyquist_hz = sample_rate * 0.5

    drum_gain_db = np.zeros(total_samples, dtype=np.float32)
    drum_width = np.zeros(total_samples, dtype=np.float32)
    drum_punch = np.zeros(total_samples, dtype=np.float32)
    bass_gain_db = np.zeros(total_samples, dtype=np.float32)
    bass_weight = np.zeros(total_samples, dtype=np.float32)
    bass_harmonics = np.zeros(total_samples, dtype=np.float32)
    sample_gain_db = np.zeros(total_samples, dtype=np.float32)
    sample_width = np.zeros(total_samples, dtype=np.float32)
    sample_presence = np.zeros(total_samples, dtype=np.float32)
    sample_air = np.zeros(total_samples, dtype=np.float32)
    sample_mud_tuck = np.zeros(total_samples, dtype=np.float32)
    section_events: List[Dict[str, Any]] = []

    section_profile_map = {
        "intro": {
            "drum_gain_db": -0.24,
            "drum_width": 0.0,
            "drum_punch": 0.004,
            "bass_gain_db": -0.14,
            "bass_weight": 0.006,
            "bass_harmonics": 0.002,
            "sample_gain_db": -0.16,
            "sample_width": 0.02,
            "sample_presence": 0.004,
            "sample_air": 0.002,
            "sample_mud_tuck": 0.006,
        },
        "verse": {
            "drum_gain_db": 0.0,
            "drum_width": -0.01,
            "drum_punch": 0.012,
            "bass_gain_db": 0.0,
            "bass_weight": 0.012,
            "bass_harmonics": 0.004,
            "sample_gain_db": -0.18,
            "sample_width": -0.03,
            "sample_presence": 0.003,
            "sample_air": 0.0,
            "sample_mud_tuck": 0.01,
        },
        "pre_hook": {
            "drum_gain_db": 0.14,
            "drum_width": 0.02,
            "drum_punch": 0.02,
            "bass_gain_db": -0.06,
            "bass_weight": 0.008,
            "bass_harmonics": 0.01,
            "sample_gain_db": 0.06,
            "sample_width": 0.08,
            "sample_presence": 0.014,
            "sample_air": 0.008,
            "sample_mud_tuck": 0.015,
        },
        "hook": {
            "drum_gain_db": 0.28,
            "drum_width": 0.045,
            "drum_punch": 0.028,
            "bass_gain_db": 0.16,
            "bass_weight": 0.026,
            "bass_harmonics": 0.014,
            "sample_gain_db": 0.12,
            "sample_width": 0.12,
            "sample_presence": 0.022,
            "sample_air": 0.012,
            "sample_mud_tuck": 0.02,
        },
        "break": {
            "drum_gain_db": -0.52,
            "drum_width": -0.04,
            "drum_punch": -0.004,
            "bass_gain_db": -0.34,
            "bass_weight": -0.004,
            "bass_harmonics": 0.0,
            "sample_gain_db": -0.22,
            "sample_width": -0.01,
            "sample_presence": -0.003,
            "sample_air": -0.002,
            "sample_mud_tuck": 0.004,
        },
        "outro": {
            "drum_gain_db": -0.26,
            "drum_width": -0.01,
            "drum_punch": 0.002,
            "bass_gain_db": -0.2,
            "bass_weight": 0.004,
            "bass_harmonics": 0.002,
            "sample_gain_db": -0.12,
            "sample_width": 0.01,
            "sample_presence": 0.002,
            "sample_air": 0.002,
            "sample_mud_tuck": 0.008,
        },
    }

    current_bar = 0
    for section in resolved_sections:
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        start_index = min(total_samples, current_bar * samples_per_bar)
        end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
        current_bar += bars
        if end_index <= start_index:
            continue
        profile = dict(section_profile_map.get(section_name) or section_profile_map["verse"])
        energy = float((section or {}).get("energy") or 0.92)
        drum_density = float((section or {}).get("drum_density") or 0.92)
        melodic_density = float((section or {}).get("melodic_density") or 0.88)
        chop_density = float((section or {}).get("chop_density") or 0.84)
        bass_density = float((section or {}).get("bass_density") or 0.94)

        profile["drum_gain_db"] += (drum_density - 0.92) * 0.8
        profile["drum_punch"] += (drum_density - 0.92) * 0.016
        profile["bass_gain_db"] += (bass_density - 0.94) * 0.7
        profile["bass_weight"] += (bass_density - 0.94) * 0.012
        profile["sample_gain_db"] += (max(melodic_density, chop_density) - 0.88) * 0.7
        profile["sample_width"] += (chop_density - 0.84) * 0.08
        profile["sample_presence"] += (energy - 0.92) * 0.018
        profile["sample_air"] += (energy - 0.92) * 0.012
        profile["sample_mud_tuck"] += max(0.0, chop_density - 0.84) * 0.01

        drum_gain_db[start_index:end_index] = float(profile["drum_gain_db"])
        drum_width[start_index:end_index] = float(profile["drum_width"])
        drum_punch[start_index:end_index] = float(profile["drum_punch"])
        bass_gain_db[start_index:end_index] = float(profile["bass_gain_db"])
        bass_weight[start_index:end_index] = float(profile["bass_weight"])
        bass_harmonics[start_index:end_index] = float(profile["bass_harmonics"])
        sample_gain_db[start_index:end_index] = float(profile["sample_gain_db"])
        sample_width[start_index:end_index] = float(profile["sample_width"])
        sample_presence[start_index:end_index] = float(profile["sample_presence"])
        sample_air[start_index:end_index] = float(profile["sample_air"])
        sample_mud_tuck[start_index:end_index] = float(profile["sample_mud_tuck"])
        section_events.append(
            {
                "section_name": section_name,
                "bars": int(bars),
                "drum_gain_db": round(float(profile["drum_gain_db"]), 4),
                "bass_gain_db": round(float(profile["bass_gain_db"]), 4),
                "sample_gain_db": round(float(profile["sample_gain_db"]), 4),
                "sample_width": round(float(profile["sample_width"]), 4),
            }
        )

    def _smooth(values: "np.ndarray") -> "np.ndarray":
        if values.size <= smoothing_samples:
            return values.astype(np.float32)
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum <= 0:
            return values.astype(np.float32)
        kernel /= kernel_sum
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    drum_gain_db = _smooth(drum_gain_db)
    drum_width = _smooth(drum_width)
    drum_punch = _smooth(drum_punch)
    bass_gain_db = _smooth(bass_gain_db)
    bass_weight = _smooth(bass_weight)
    bass_harmonics = _smooth(bass_harmonics)
    sample_gain_db = _smooth(sample_gain_db)
    sample_width = _smooth(sample_width)
    sample_presence = _smooth(sample_presence)
    sample_air = _smooth(sample_air)
    sample_mud_tuck = _smooth(sample_mud_tuck)

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(40.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    drum_body_sos = _bandpass(68.0, 240.0)
    drum_attack_sos = _bandpass(1700.0, 5200.0)
    bass_body_sos = _bandpass(46.0, 180.0)
    bass_harmonic_sos = _bandpass(220.0, 920.0)
    sample_presence_sos = _bandpass(1450.0, 4600.0)
    sample_air_sos = _bandpass(4600.0, 8800.0)
    sample_mud_sos = _bandpass(220.0, 760.0)

    processed_layers = 0
    touched_roles: Dict[str, int] = {}

    def _apply_width(render: "np.ndarray", width_env: "np.ndarray") -> "np.ndarray":
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32)
        side *= width_env[: side.shape[0]]
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    processing_layers: List[str] = []
    seen_processing_layers: Set[str] = set()

    def _filter_zero_phase(sos: Optional[Any], mono: "np.ndarray") -> "np.ndarray":
        if sos is None:
            return np.zeros(mono.shape[0], dtype=np.float32)
        return signal.sosfiltfilt(sos, mono).astype(np.float32)

    def _filter_fast(sos: Optional[Any], mono: "np.ndarray") -> "np.ndarray":
        if sos is None:
            return np.zeros(mono.shape[0], dtype=np.float32)
        return signal.sosfilt(sos, mono).astype(np.float32)

    def _queue_layer(name: str) -> None:
        if name in layer_renders and name not in seen_processing_layers:
            processing_layers.append(name)
            seen_processing_layers.add(name)

    for layer_name in sorted(selected_drum_layers):
        _queue_layer(layer_name)
    for layer_name in bass_layers:
        _queue_layer(layer_name)
    for layer_name in representative_sample_layers[:3]:
        _queue_layer(layer_name)

    for layer_name in processing_layers:
        role_name = role_by_name.get(layer_name, "")
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        role = role_name or ""

        if role in drum_roles:
            if layer_name not in selected_drum_layers:
                continue
            gain_linear = np.power(10.0, np.clip(drum_gain_db[: render.shape[0]], -1.2, 0.8) / 20.0).astype(np.float32)
            processed = render * gain_linear[:, np.newaxis]
            if drum_body_sos is not None and drum_attack_sos is not None:
                body_scale = 1.15 if role == "kick" else 0.44 if role in {"snare", "clap"} else 0.18
                attack_scale = 0.22 if role == "kick" else 0.92 if role in {"snare", "clap"} else 0.7
                if role in {"kick", "snare", "clap"}:
                    layer_punch = drum_punch[: render.shape[0]]
                    mono = np.mean(processed, axis=1).astype(np.float32)
                    body = _filter_zero_phase(drum_body_sos, mono)
                    attack = _filter_fast(drum_attack_sos, mono)
                    adjustment = ((body * (layer_punch * body_scale)) + (attack * (layer_punch * attack_scale))).astype(np.float32)
                    processed[:, 0] += adjustment
                    processed[:, 1] += adjustment
                elif role in {"hat", "hihat", "perc"}:
                    layer_punch = drum_punch[: render.shape[0]]
                    mono = np.mean(processed, axis=1).astype(np.float32)
                    attack = _filter_fast(drum_attack_sos, mono)
                    adjustment = (attack * (layer_punch * attack_scale * 0.75)).astype(np.float32)
                    processed[:, 0] += adjustment
                    processed[:, 1] += adjustment
            if role in drum_width_roles:
                processed = _apply_width(processed, 1.0 + np.clip(drum_width[: render.shape[0]], -0.08, 0.14))
            layer_renders[layer_name] = np.clip(np.tanh(processed * 1.01), -1.0, 1.0).astype(np.float32)
            processed_layers += 1
            touched_roles[role] = int(touched_roles.get(role, 0)) + 1
            continue

        if role == "bass":
            gain_linear = np.power(10.0, np.clip(bass_gain_db[: render.shape[0]], -1.1, 0.7) / 20.0).astype(np.float32)
            processed = render * gain_linear[:, np.newaxis]
            if bass_body_sos is not None and bass_harmonic_sos is not None:
                layer_weight = bass_weight[: render.shape[0]]
                layer_harm = bass_harmonics[: render.shape[0]]
                mono = np.mean(processed, axis=1).astype(np.float32)
                low = _filter_zero_phase(bass_body_sos, mono)
                harmonic = _filter_fast(bass_harmonic_sos, mono)
                adjustment = ((low * layer_weight) + (harmonic * layer_harm)).astype(np.float32)
                processed[:, 0] += adjustment
                processed[:, 1] += adjustment
            mid = ((processed[:, 0] + processed[:, 1]) * 0.5).astype(np.float32)
            side = ((processed[:, 0] - processed[:, 1]) * 0.5).astype(np.float32) * 0.92
            processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
            layer_renders[layer_name] = np.clip(np.tanh(processed * 1.015), -1.0, 1.0).astype(np.float32)
            processed_layers += 1
            touched_roles[role] = int(touched_roles.get(role, 0)) + 1
            continue

        if role in sample_roles:
            if layer_name not in selected_sample_layers:
                continue
            role_sample_gain_db = sample_gain_db[: render.shape[0]].copy()
            role_width = sample_width[: render.shape[0]].copy()
            role_presence = sample_presence[: render.shape[0]].copy()
            role_air = sample_air[: render.shape[0]].copy()
            role_mud = sample_mud_tuck[: render.shape[0]].copy()

            lower_name = _clean_text(layer_name).lower()
            if role == "texture":
                role_sample_gain_db -= 0.06
                role_presence *= 0.82
                role_air *= 0.9
                role_mud *= 1.12
            elif role == "resample":
                role_sample_gain_db += 0.05
                role_presence *= 1.08
                role_width *= 1.06
            elif role == "tonal_glue":
                role_sample_gain_db += 0.08
                role_width *= 1.12
                role_air *= 1.1
            elif role == "chop" and "hook" in lower_name:
                role_sample_gain_db += 0.1
                role_width *= 1.1
                role_presence *= 1.14
            elif role in {"melody", "lead"}:
                role_width *= 0.92

            gain_linear = np.power(10.0, np.clip(role_sample_gain_db, -1.0, 0.9) / 20.0).astype(np.float32)
            processed = render * gain_linear[:, np.newaxis]
            if sample_presence_sos is not None and sample_air_sos is not None and sample_mud_sos is not None:
                mono = np.mean(processed, axis=1).astype(np.float32)
                adjustment = np.zeros(mono.shape[0], dtype=np.float32)
                if role in {"melody", "lead", "chop", "resample", "tonal_glue"}:
                    presence = _filter_fast(sample_presence_sos, mono)
                    adjustment += presence * role_presence
                if role in {"texture", "chop", "resample", "tonal_glue"}:
                    air = _filter_fast(sample_air_sos, mono)
                    adjustment += air * role_air
                if role in {"texture", "chop"}:
                    mud = _filter_zero_phase(sample_mud_sos, mono)
                    adjustment -= mud * role_mud
                adjustment = adjustment.astype(np.float32)
                processed[:, 0] += adjustment
                processed[:, 1] += adjustment
            processed = _apply_width(processed, 1.0 + np.clip(role_width, -0.08, 0.18))
            layer_renders[layer_name] = np.clip(np.tanh(processed * 1.01), -1.0, 1.0).astype(np.float32)
            processed_layers += 1
            touched_roles[role] = int(touched_roles.get(role, 0)) + 1

    if processed_layers <= 0:
        return None

    return {
        "mode": "ensemble_section_contrast",
        "profile": "boombap_ensemble_section_contrast_v1",
        "processed_layers_count": int(processed_layers),
        "selected_drum_layers": sorted(selected_drum_layers),
        "selected_sample_layers": sorted(selected_sample_layers),
        "touched_roles": touched_roles,
        "events": section_events,
        "events_count": len(section_events),
        "smoothing_samples": int(smoothing_samples),
        "max_drum_gain_db": round(float(np.max(drum_gain_db)) if drum_gain_db.size else 0.0, 4),
        "max_bass_gain_db": round(float(np.max(bass_gain_db)) if bass_gain_db.size else 0.0, 4),
        "max_sample_gain_db": round(float(np.max(sample_gain_db)) if sample_gain_db.size else 0.0, 4),
        "max_sample_width": round(float(np.max(sample_width)) if sample_width.size else 0.0, 4),
    }


def _apply_pattern_hook_memory_lift(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or signal is None or not layer_renders or not resolved_sections:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    candidate_layers = _select_sample_stack_focus_layers(
        role_by_name=role_by_name,
        available_layers=[
            name
            for name, role in role_by_name.items()
            if role in {"melody", "lead", "chop", "texture", "resample", "tonal_glue"} and name in layer_renders
        ],
    )
    lead_layers = [name for name in candidate_layers if role_by_name.get(name) in {"melody", "lead"}][:1]
    hook_layers = [
        name
        for name in candidate_layers
        if role_by_name.get(name) == "chop" and "hook" in _clean_text(name).lower()
    ][:1]
    resample_layers = [name for name in candidate_layers if role_by_name.get(name) == "resample"][:1]
    tonal_layers = [name for name in candidate_layers if role_by_name.get(name) == "tonal_glue"][:1]
    texture_layers = [name for name in candidate_layers if role_by_name.get(name) == "texture"][:1]
    source_layers = lead_layers + hook_layers + resample_layers + tonal_layers
    if not lead_layers or not (hook_layers or resample_layers or tonal_layers):
        return None

    total_samples = max(layer_renders[name].shape[0] for name in (source_layers + texture_layers) if name in layer_renders)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    smoothing_samples = max(96, int(round(sample_rate * 0.018)))
    hook_focus = np.zeros(total_samples, dtype=np.float32)
    hook_pulse = np.zeros(total_samples, dtype=np.float32)
    section_events: List[Dict[str, Any]] = []
    current_bar = 0

    for section in resolved_sections:
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        start_index = min(total_samples, current_bar * samples_per_bar)
        end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
        if end_index > start_index:
            if section_name == "pre_hook":
                hook_focus[start_index:end_index] = np.maximum(hook_focus[start_index:end_index], 0.56)
            elif section_name == "hook":
                hook_focus[start_index:end_index] = np.maximum(hook_focus[start_index:end_index], 1.0)
                for bar_offset in range(bars):
                    bar_start = start_index + (bar_offset * samples_per_bar)
                    for pulse_ratio, pulse_gain in ((0.0, 1.0), (0.5, 0.82)):
                        pulse_center = min(end_index - 1, bar_start + int(round(samples_per_bar * pulse_ratio)))
                        pulse_half = max(48, int(round(samples_per_bar * 0.08)))
                        pulse_start = max(start_index, pulse_center - pulse_half)
                        pulse_end = min(end_index, pulse_center + pulse_half)
                        if pulse_end > pulse_start:
                            pulse = np.hanning(max(3, pulse_end - pulse_start)).astype(np.float32)
                            hook_pulse[pulse_start:pulse_end] = np.maximum(
                                hook_pulse[pulse_start:pulse_end],
                                pulse * float(pulse_gain),
                            )
                section_events.append(
                    {
                        "section_name": section_name,
                        "bars": int(bars),
                        "bar_start": int(current_bar),
                        "pulse_points_per_bar": 2,
                    }
                )
        current_bar += bars

    if not np.any(hook_focus > 0):
        return None

    if hook_focus.size > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            hook_focus = np.convolve(hook_focus, kernel, mode="same").astype(np.float32)
            hook_pulse = np.convolve(hook_pulse, kernel, mode="same").astype(np.float32)
    hook_focus = np.clip(hook_focus, 0.0, 1.0).astype(np.float32)
    hook_pulse = np.clip(hook_pulse, 0.0, 1.0).astype(np.float32)
    hook_energy = np.clip(np.maximum(hook_focus * 0.62, hook_pulse), 0.0, 1.0).astype(np.float32)

    nyquist_hz = sample_rate * 0.5

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(40.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    lead_presence_sos = _bandpass(1650.0, 4600.0)
    lead_body_sos = _bandpass(260.0, 1350.0)
    hook_presence_sos = _bandpass(1900.0, 5400.0)
    resample_presence_sos = _bandpass(1300.0, 4200.0)
    texture_mud_sos = _bandpass(220.0, 760.0)
    bus_presence_sos = _bandpass(1350.0, 5200.0)
    bus_air_sos = _bandpass(4200.0, 8600.0)

    processed_layers = 0
    touched_layers: List[str] = []

    def _apply_width(render: "np.ndarray", width_env: "np.ndarray") -> "np.ndarray":
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32)
        side *= width_env[: side.shape[0]]
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    def _apply_mono_band_mix(
        render: "np.ndarray",
        primary_sos: Optional[Any],
        primary_gain: "np.ndarray",
        secondary_sos: Optional[Any] = None,
        secondary_gain: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        processed = render.copy()
        mono = np.mean(processed, axis=1).astype(np.float32)
        adjustment = np.zeros(mono.shape[0], dtype=np.float32)
        if primary_sos is not None:
            adjustment += signal.sosfiltfilt(primary_sos, mono).astype(np.float32) * primary_gain[: mono.shape[0]]
        if secondary_sos is not None and secondary_gain is not None:
            adjustment += signal.sosfiltfilt(secondary_sos, mono).astype(np.float32) * secondary_gain[: mono.shape[0]]
        processed[:, 0] += adjustment
        processed[:, 1] += adjustment
        return processed.astype(np.float32)

    for layer_name in lead_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        usable_focus = hook_focus[: render.shape[0]]
        usable_pulse = hook_pulse[: render.shape[0]]
        if lead_presence_sos is not None and lead_body_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                lead_body_sos,
                usable_focus * 0.008,
                lead_presence_sos,
                usable_pulse * 0.024,
            )
        else:
            processed = render.copy()
        gain = 1.0 + (usable_focus * 0.022) + (usable_pulse * 0.03)
        processed *= gain[:, np.newaxis]
        processed = _apply_width(processed, 1.0 + (usable_focus * 0.04) + (usable_pulse * 0.03))
        layer_renders[layer_name] = np.clip(np.tanh(processed * 1.01), -1.0, 1.0).astype(np.float32)
        processed_layers += 1
        touched_layers.append(layer_name)

    for layer_name in hook_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        usable_focus = hook_focus[: render.shape[0]]
        usable_pulse = hook_pulse[: render.shape[0]]
        if hook_presence_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                hook_presence_sos,
                (usable_focus * 0.022) + (usable_pulse * 0.038),
            )
        else:
            processed = render.copy()
        gain = 1.0 + (usable_focus * 0.028) + (usable_pulse * 0.042)
        processed *= gain[:, np.newaxis]
        processed = _apply_width(processed, 1.0 + (usable_focus * 0.08) + (usable_pulse * 0.045))
        layer_renders[layer_name] = np.clip(np.tanh(processed * 1.02), -1.0, 1.0).astype(np.float32)
        processed_layers += 1
        touched_layers.append(layer_name)

    for layer_name in resample_layers + tonal_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        usable_focus = hook_focus[: render.shape[0]]
        usable_pulse = hook_pulse[: render.shape[0]]
        if resample_presence_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                resample_presence_sos,
                (usable_focus * 0.016) + (usable_pulse * 0.024),
            )
        else:
            processed = render.copy()
        gain = 1.0 + (usable_focus * 0.02) + (usable_pulse * 0.024)
        processed *= gain[:, np.newaxis]
        processed = _apply_width(processed, 1.0 + (usable_focus * 0.06))
        layer_renders[layer_name] = np.clip(np.tanh(processed * 1.012), -1.0, 1.0).astype(np.float32)
        processed_layers += 1
        touched_layers.append(layer_name)

    for layer_name in texture_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        usable_focus = hook_focus[: render.shape[0]]
        usable_pulse = hook_pulse[: render.shape[0]]
        if texture_mud_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                texture_mud_sos,
                -((usable_focus * 0.012) + (usable_pulse * 0.014)),
            )
        else:
            processed = render.copy()
        gain = 1.0 - (usable_pulse * 0.026)
        processed *= gain[:, np.newaxis]
        processed = _apply_width(processed, 1.0 + (usable_focus * 0.03))
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0).astype(np.float32)
        processed_layers += 1
        touched_layers.append(layer_name)

    bus = np.zeros((total_samples, 2), dtype=np.float32)
    for layer_name in source_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        weight = 0.24 if layer_name in lead_layers else 0.3 if layer_name in hook_layers else 0.22 if layer_name in tonal_layers else 0.14
        bus[: render.shape[0]] += render * weight
    bus *= (hook_energy[:, np.newaxis] * 0.34)
    if np.any(np.abs(bus) > 1e-7) and bus_presence_sos is not None and bus_air_sos is not None:
        bus_mono = np.mean(bus, axis=1).astype(np.float32)
        presence = signal.sosfiltfilt(bus_presence_sos, bus_mono).astype(np.float32)
        air = signal.sosfiltfilt(bus_air_sos, bus_mono).astype(np.float32)
        bus_colored = ((bus_mono * 0.38) + (presence * 0.42) + (air * 0.1)).astype(np.float32)
        bus = np.stack([bus_colored, bus_colored], axis=1).astype(np.float32)
        bus = np.tanh(bus * (1.08 + (hook_energy[:, np.newaxis] * 0.18))).astype(np.float32)
        bus = _apply_width(bus, 1.0 + (hook_focus * 0.14) + (hook_pulse * 0.08))
        bus = np.clip(bus, -0.32, 0.32).astype(np.float32)
        layer_name = "hook-memory-bus"
        layer_renders[layer_name] = bus
        rendered_layers.append(
            {
                "name": layer_name,
                "role": "hook_memory",
                "pattern": None,
                "trigger_positions": [],
                "source_path": None,
                "copied_input_path": None,
                "render_mode": "hook_memory_lift",
            }
        )
    else:
        layer_name = ""

    if processed_layers <= 0 and not layer_name:
        return None

    return {
        "mode": "hook_memory_lift",
        "profile": "boombap_hook_memory_lift_v1",
        "candidate_layers": candidate_layers,
        "lead_layers": lead_layers,
        "hook_layers": hook_layers,
        "resample_layers": resample_layers,
        "tonal_layers": tonal_layers,
        "texture_layers": texture_layers,
        "bus_layer_name": layer_name or None,
        "processed_layers_count": int(processed_layers),
        "touched_layers": touched_layers[:24],
        "events": section_events,
        "events_count": len(section_events),
        "hook_focus_peak": round(float(np.max(hook_focus)) if hook_focus.size else 0.0, 6),
        "hook_pulse_peak": round(float(np.max(hook_pulse)) if hook_pulse.size else 0.0, 6),
        "smoothing_samples": int(smoothing_samples),
    }


def _apply_pattern_layer_section_mix(
    layer_render: "np.ndarray",
    role: str,
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if layer_render.size == 0 or not resolved_sections:
        return None
    resolved_role = (_clean_text(role) or "sample").lower()
    if resolved_role not in {"hat", "hihat", "perc", "melody", "lead", "texture", "chop", "clap", "snare", "drum_bed", "resample"}:
        return None

    total_samples = layer_render.shape[0]
    section_signature = tuple(
        (
            _clean_text((section or {}).get("name")).lower(),
            _clean_text((section or {}).get("song_role")).lower(),
            _clean_text((section or {}).get("section_family")).lower(),
            int((section or {}).get("bars") or 1),
        )
        for section in resolved_sections
    )
    cache_key = (
        resolved_role,
        int(sample_rate),
        round(float(tempo_bpm), 4),
        int(total_samples),
        section_signature,
    )
    cached_profile = _SECTION_MIX_PROFILE_CACHE.get(cache_key)
    if cached_profile is None:
        samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
        gain_db_envelope = np.zeros(total_samples, dtype=np.float32)
        width_envelope = np.zeros(total_samples, dtype=np.float32)
        current_bar = 0
        section_events: List[Dict[str, Any]] = []

        role_section_gains = {
            "hat": {"intro": -0.7, "verse": 0.0, "pre_hook": 0.55, "hook": 0.42, "break": -1.0, "outro": -0.55},
            "hihat": {"intro": -0.7, "verse": 0.0, "pre_hook": 0.55, "hook": 0.42, "break": -1.0, "outro": -0.55},
            "perc": {"intro": -0.95, "verse": -0.08, "pre_hook": 0.18, "hook": 0.26, "break": -1.1, "outro": -0.7},
            "snare": {"intro": -0.2, "verse": 0.0, "pre_hook": 0.08, "hook": 0.18, "break": -0.28, "outro": -0.18},
            "clap": {"intro": -0.38, "verse": 0.0, "pre_hook": 0.14, "hook": 0.24, "break": -0.3, "outro": -0.24},
            "melody": {"intro": -0.38, "verse": 0.0, "pre_hook": -0.12, "hook": 0.3, "break": -0.62, "outro": -0.32},
            "lead": {"intro": -0.3, "verse": 0.0, "pre_hook": 0.0, "hook": 0.26, "break": -0.55, "outro": -0.25},
            "texture": {"intro": 0.1, "verse": -0.08, "pre_hook": -0.28, "hook": 0.16, "break": -0.42, "outro": -0.1},
            "chop": {"intro": -0.5, "verse": 0.0, "pre_hook": 0.32, "hook": 0.28, "break": -0.72, "outro": -0.45},
            "drum_bed": {"intro": -0.6, "verse": 0.0, "pre_hook": 0.18, "hook": 0.28, "break": -0.75, "outro": -0.45},
            "resample": {"intro": -0.65, "verse": 0.0, "pre_hook": 0.2, "hook": 0.32, "break": -0.84, "outro": -0.55},
        }
        role_section_width = {
            "hat": {"intro": 0.0, "verse": 0.02, "pre_hook": 0.05, "hook": 0.08, "break": -0.03, "outro": 0.0},
            "hihat": {"intro": 0.0, "verse": 0.02, "pre_hook": 0.05, "hook": 0.08, "break": -0.03, "outro": 0.0},
            "perc": {"intro": 0.01, "verse": 0.03, "pre_hook": 0.07, "hook": 0.1, "break": -0.04, "outro": 0.01},
            "snare": {"intro": 0.0, "verse": 0.0, "pre_hook": 0.01, "hook": 0.03, "break": 0.0, "outro": 0.0},
            "clap": {"intro": 0.0, "verse": 0.02, "pre_hook": 0.05, "hook": 0.09, "break": 0.0, "outro": 0.01},
            "melody": {"intro": 0.02, "verse": 0.06, "pre_hook": 0.04, "hook": 0.15, "break": -0.02, "outro": 0.03},
            "lead": {"intro": 0.02, "verse": 0.05, "pre_hook": 0.05, "hook": 0.13, "break": -0.01, "outro": 0.02},
            "texture": {"intro": 0.03, "verse": 0.05, "pre_hook": 0.02, "hook": 0.11, "break": -0.04, "outro": 0.02},
            "chop": {"intro": 0.0, "verse": 0.03, "pre_hook": 0.08, "hook": 0.12, "break": -0.02, "outro": 0.0},
            "drum_bed": {"intro": 0.0, "verse": 0.02, "pre_hook": 0.04, "hook": 0.06, "break": -0.03, "outro": 0.0},
            "resample": {"intro": 0.0, "verse": 0.03, "pre_hook": 0.06, "hook": 0.1, "break": -0.02, "outro": 0.0},
        }

        for section in resolved_sections:
            section_name = _resolve_pattern_section_mix_key(section)
            bars = max(1, int((section or {}).get("bars") or 1))
            start_index = min(total_samples, current_bar * samples_per_bar)
            end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
            if end_index <= start_index:
                current_bar += bars
                continue
            gain_db = float((role_section_gains.get(resolved_role) or {}).get(section_name, 0.0))
            width_gain = float((role_section_width.get(resolved_role) or {}).get(section_name, 0.0))
            gain_db_envelope[start_index:end_index] += gain_db
            width_envelope[start_index:end_index] += width_gain
            section_events.append(
                {
                    "section_name": section_name,
                    "song_role": _clean_text((section or {}).get("song_role")).lower() or section_name,
                    "bars": int(bars),
                    "gain_db": round(gain_db, 4),
                    "width_gain": round(width_gain, 4),
                }
            )
            current_bar += bars

        smoothing_samples = max(64, int(round(sample_rate * 0.025)))
        if total_samples > smoothing_samples:
            kernel = np.hanning(smoothing_samples).astype(np.float32)
            kernel_sum = float(np.sum(kernel))
            if kernel_sum > 0:
                kernel /= kernel_sum
                gain_db_envelope = np.convolve(gain_db_envelope, kernel, mode="same").astype(np.float32)
                width_envelope = np.convolve(width_envelope, kernel, mode="same").astype(np.float32)

        gain_linear = np.power(10.0, np.clip(gain_db_envelope, -1.2, 0.8) / 20.0).astype(np.float32)
        cached_profile = {
            "gain_linear": gain_linear,
            "width_envelope": width_envelope.astype(np.float32),
            "section_events": section_events,
            "smoothing_samples": int(smoothing_samples),
            "max_gain_db": round(float(np.max(gain_db_envelope)) if gain_db_envelope.size else 0.0, 4),
            "min_gain_db": round(float(np.min(gain_db_envelope)) if gain_db_envelope.size else 0.0, 4),
            "max_width_gain": round(float(np.max(width_envelope)) if width_envelope.size else 0.0, 4),
            "min_width_gain": round(float(np.min(width_envelope)) if width_envelope.size else 0.0, 4),
        }
        _SECTION_MIX_PROFILE_CACHE[cache_key] = cached_profile

    gain_linear = cached_profile["gain_linear"]
    width_envelope = cached_profile["width_envelope"]
    section_events = cached_profile["section_events"]
    smoothing_samples = int(cached_profile["smoothing_samples"])

    if resolved_role in {"melody", "lead", "texture", "chop", "hat", "hihat", "perc", "clap"}:
        if float(np.max(np.abs(width_envelope))) > 1e-6:
            mid = ((layer_render[:, 0] + layer_render[:, 1]) * 0.5).astype(np.float32)
            side = ((layer_render[:, 0] - layer_render[:, 1]) * 0.5).astype(np.float32)
            side *= (1.0 + np.clip(width_envelope, -0.12, 0.22))
            processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
        else:
            processed = layer_render.copy()
    else:
        processed = layer_render.copy()
    processed *= gain_linear[:, np.newaxis]
    layer_render[:] = np.clip(processed, -1.0, 1.0)
    return {
        "mode": "section_mix",
        "role": resolved_role,
        "events": section_events,
        "smoothing_samples": int(smoothing_samples),
        "max_gain_db": cached_profile["max_gain_db"],
        "min_gain_db": cached_profile["min_gain_db"],
        "max_width_gain": cached_profile["max_width_gain"],
        "min_width_gain": cached_profile["min_width_gain"],
    }


def _resize_audio_to_length(audio: "np.ndarray", target_length: int) -> "np.ndarray":
    if np is None:
        return audio
    stereo = _ensure_stereo(audio).astype(np.float32)
    target = max(1, int(target_length))
    if stereo.shape[0] == target:
        return stereo.copy()
    if stereo.shape[0] <= 1:
        resized = np.zeros((target, 2), dtype=np.float32)
        resized[: stereo.shape[0]] = stereo[:1]
        return resized
    source_x = np.linspace(0.0, 1.0, stereo.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target, dtype=np.float32)
    left = np.interp(target_x, source_x, stereo[:, 0]).astype(np.float32)
    right = np.interp(target_x, source_x, stereo[:, 1]).astype(np.float32)
    return np.stack([left, right], axis=1).astype(np.float32)


def _gate_audio_tail(audio: "np.ndarray", active_ratio: float, floor_gain: float = 0.0) -> "np.ndarray":
    if np is None:
        return audio
    processed = _ensure_stereo(audio).astype(np.float32).copy()
    if processed.shape[0] <= 4:
        return processed
    active = max(1, min(processed.shape[0], int(round(processed.shape[0] * float(active_ratio)))))
    fade = min(max(16, processed.shape[0] // 20), max(1, processed.shape[0] - active))
    if active < processed.shape[0]:
        if fade > 0:
            end = min(processed.shape[0], active + fade)
            processed[active:end] *= np.linspace(1.0, float(floor_gain), end - active, dtype=np.float32)[:, np.newaxis]
        processed[active + fade :] *= float(floor_gain)
    return processed


def _build_chop_stutter(audio: "np.ndarray", repeats: int, slice_ratio: float, spacing_ratio: float) -> "np.ndarray":
    if np is None:
        return audio
    source = _ensure_stereo(audio).astype(np.float32)
    output = np.zeros_like(source)
    if source.shape[0] <= 16:
        return source.copy()
    slice_len = max(16, min(source.shape[0], int(round(source.shape[0] * float(slice_ratio)))))
    spacing = max(1, int(round(source.shape[0] * float(spacing_ratio))))
    chunk = source[:slice_len].copy()
    fade = min(max(8, slice_len // 18), max(1, slice_len // 4))
    if fade > 0:
        chunk[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, np.newaxis]
        chunk[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, np.newaxis]
    for repeat_index in range(max(1, int(repeats))):
        start = repeat_index * spacing
        end = min(output.shape[0], start + chunk.shape[0])
        if end <= start:
            continue
        gain = 1.0 - (repeat_index * 0.16)
        output[start:end] += chunk[: end - start] * max(0.45, gain)
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def _append_variation_transform(manifest: Dict[str, Any], transform_name: str) -> None:
    current = _clean_text(manifest.get("transform"))
    if current:
        manifest["transform"] = f"{current}+{transform_name}"
    else:
        manifest["transform"] = transform_name


def _resolve_pattern_boombap_trigger_groove(
    role: str,
    section: Optional[Dict[str, Any]],
    step_index: int,
    steps_per_bar: int,
    seconds_per_step: float,
    layer_variant: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    resolved_role = (_clean_text(role) or "sample").lower()
    if resolved_role not in {"kick", "snare", "clap", "hat", "hihat", "perc", "chop"}:
        return None
    resolved_variant = _clean_text(layer_variant).lower() or "default"
    song_role = _resolve_section_song_role(section)
    phase = int(step_index) % max(1, int(steps_per_bar))
    quarter = max(1, int(steps_per_bar) // 4)
    eighth = max(1, int(steps_per_bar) // 8)
    sixteenth = max(1, int(steps_per_bar) // 16)
    offset_ratio = 0.0
    gain_db = 0.0
    feel = "center"

    if resolved_role == "kick":
        if phase == 0:
            offset_ratio = -0.028
            gain_db = 0.18
            feel = "front_downbeat"
        elif phase in {quarter + eighth, (quarter * 3) + eighth, max(0, int(steps_per_bar) - eighth)}:
            offset_ratio = 0.035
            gain_db = -0.08
            feel = "late_syncopation"
        else:
            offset_ratio = -0.012 if phase % max(1, quarter) == 0 else 0.018
            feel = "pocket_kick"
    elif resolved_role in {"snare", "clap"}:
        backbeats = {quarter, quarter * 3}
        if phase in backbeats:
            offset_ratio = 0.052
            gain_db = 0.22 if song_role in {"hook", "pre_hook"} else 0.12
            feel = "late_backbeat"
        elif resolved_variant == "ghost":
            offset_ratio = -0.032 if phase < (quarter * 2) else 0.026
            gain_db = -0.34
            feel = "ghost_push_pull"
        else:
            offset_ratio = 0.024
            feel = "soft_late_hit"
    elif resolved_role in {"hat", "hihat"}:
        if phase % max(1, eighth * 2) == 0:
            offset_ratio = -0.012
            gain_db = -0.12
            feel = "tight_anchor_hat"
        elif phase % max(1, eighth) == 0:
            offset_ratio = 0.045
            gain_db = 0.08 if song_role in {"hook", "pre_hook"} else -0.02
            feel = "lazy_eighth_hat"
        else:
            offset_ratio = 0.022 if ((phase // max(1, sixteenth)) % 2) else -0.018
            gain_db = -0.24 if resolved_variant == "alt" else -0.12
            feel = "sixteenth_shuffle_hat"
    elif resolved_role == "perc":
        offset_ratio = 0.04 if phase >= (quarter * 2) else -0.018
        gain_db = -0.1 if song_role in {"intro", "outro"} else 0.05
        feel = "loose_perc"
    elif resolved_role == "chop":
        if song_role == "hook" and phase in {0, quarter * 2}:
            offset_ratio = -0.018
            gain_db = 0.2
            feel = "hook_chop_on_top"
        elif phase >= int(steps_per_bar) - max(1, eighth + sixteenth):
            offset_ratio = 0.048
            gain_db = -0.04
            feel = "late_pickup_chop"
        elif resolved_variant == "phrase":
            offset_ratio = 0.026
            gain_db = -0.12
            feel = "laid_back_phrase_chop"
        else:
            offset_ratio = 0.018
            feel = "laid_back_chop"

    if song_role == "break":
        offset_ratio *= 0.55
        gain_db -= 0.08
    elif song_role == "pre_hook":
        offset_ratio *= 0.88
    elif song_role == "hook":
        gain_db += 0.05

    offset_seconds = float(seconds_per_step) * float(max(-0.09, min(0.09, offset_ratio)))
    if abs(offset_seconds) < 1e-6 and abs(gain_db) < 1e-6:
        return None
    return {
        "profile": "boombap_mpc_pocket_v1",
        "feel": feel,
        "offset_seconds": round(float(offset_seconds), 5),
        "gain_db": round(float(gain_db), 4),
    }


def _apply_gain_envelope_segment(
    render: "np.ndarray",
    start_index: int,
    end_index: int,
    start_gain: float,
    end_gain: float,
) -> None:
    if np is None or render.size == 0:
        return
    start = max(0, min(render.shape[0], int(start_index)))
    end = max(start, min(render.shape[0], int(end_index)))
    if end <= start:
        return
    envelope = np.linspace(float(start_gain), float(end_gain), end - start, dtype=np.float32)
    render[start:end] *= envelope[:, np.newaxis]


def _apply_constant_gain_segment(
    render: "np.ndarray",
    start_index: int,
    end_index: int,
    gain: float,
) -> None:
    if np is None or render.size == 0:
        return
    start = max(0, min(render.shape[0], int(start_index)))
    end = max(start, min(render.shape[0], int(end_index)))
    if end <= start:
        return
    render[start:end] *= float(gain)


def _inject_reverse_transition_swell(
    render: "np.ndarray",
    boundary_index: int,
    duration_samples: int,
    gain: float,
) -> bool:
    if np is None or render.size == 0:
        return False
    duration = max(32, int(duration_samples))
    boundary = max(0, min(render.shape[0], int(boundary_index)))
    start = max(0, boundary - duration)
    end = min(render.shape[0], boundary + max(32, duration // 2))
    if boundary <= start or end <= boundary:
        return False
    source = render[boundary:end].copy()
    if source.shape[0] < 32:
        return False
    swell = np.flip(source, axis=0).copy()
    swell = _resize_audio_to_length(swell, boundary - start)
    fade = np.linspace(0.0, float(gain), swell.shape[0], dtype=np.float32)
    render[start:boundary] += swell * fade[:, np.newaxis]
    render[start:boundary] = np.clip(render[start:boundary], -1.0, 1.0)
    return True


def _apply_pattern_drum_turnarounds(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or not layer_renders or len(resolved_sections) < 2:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    drum_layer_names = _select_representative_drum_layers(
        role_by_name=role_by_name,
        available_layers=[
            name
            for name, role in role_by_name.items()
            if role in {"snare", "clap", "hat", "hihat", "perc", "kick"} and name in layer_renders
        ],
    )
    if not drum_layer_names:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in drum_layer_names)
    turnaround_bus = np.zeros((total_samples, 2), dtype=np.float32)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    eighth = max(32, int(round(samples_per_bar * 0.125)))
    sixteenth = max(24, int(round(samples_per_bar * 0.0625)))
    quarter = max(64, int(round(samples_per_bar * 0.25)))
    events: List[Dict[str, Any]] = []
    current_bar = 0

    for section_index, section in enumerate(resolved_sections[:-1]):
        bars = max(1, int((section or {}).get("bars") or 1))
        current_bar += bars
        next_section = resolved_sections[section_index + 1]
        next_role = _resolve_section_song_role(next_section)
        previous_role = _resolve_section_song_role(section)
        if next_role not in {"pre_hook", "hook", "break", "verse"}:
            continue
        boundary_index = min(total_samples, current_bar * samples_per_bar)
        if boundary_index <= 0 or boundary_index >= total_samples:
            continue

        if next_role == "hook":
            fill_length = quarter
            stutter_repeats = 4
            layer_gain_map = {"hat": 0.92, "hihat": 0.92, "perc": 0.76, "snare": 0.82, "clap": 0.78, "kick": 0.34}
            room_duck_roles = {"hat", "hihat", "perc"}
            transition_type = "hook_turnaround"
        elif next_role == "pre_hook":
            fill_length = max(eighth + sixteenth, int(round(quarter * 0.7)))
            stutter_repeats = 3
            layer_gain_map = {"hat": 0.84, "hihat": 0.84, "perc": 0.68, "snare": 0.7, "clap": 0.62, "kick": 0.18}
            room_duck_roles = {"hat", "hihat"}
            transition_type = "pre_hook_pickup_fill"
        elif next_role == "break":
            fill_length = max(eighth, int(round(quarter * 0.55)))
            stutter_repeats = 2
            layer_gain_map = {"hat": 0.58, "hihat": 0.58, "perc": 0.44, "snare": 0.78, "clap": 0.64, "kick": 0.22}
            room_duck_roles = {"hat", "hihat", "perc", "kick"}
            transition_type = "break_turnaround"
        else:
            fill_length = max(eighth, int(round(quarter * 0.5)))
            stutter_repeats = 2
            layer_gain_map = {"hat": 0.52, "hihat": 0.52, "perc": 0.34, "snare": 0.46, "clap": 0.38, "kick": 0.0}
            room_duck_roles = {"hat", "hihat"}
            transition_type = "verse_turnaround"

        start_index = max(0, boundary_index - fill_length)
        if start_index >= boundary_index:
            continue
        affected_layers: List[str] = []

        for layer_name in drum_layer_names:
            render = layer_renders.get(layer_name)
            if render is None or render.size == 0:
                continue
            role_name = role_by_name.get(layer_name, "")
            layer_gain = float(layer_gain_map.get(role_name, 0.0))
            if layer_gain <= 0.0:
                continue
            source_start = max(0, boundary_index - max(fill_length, eighth))
            source = render[source_start:boundary_index].copy()
            if source.shape[0] < 16 or float(np.max(np.abs(source))) < 0.01:
                continue

            if role_name in {"hat", "hihat"}:
                snippet = _build_chop_stutter(
                    source,
                    repeats=stutter_repeats,
                    slice_ratio=0.16 if next_role == "hook" else 0.2,
                    spacing_ratio=0.15 if next_role == "hook" else 0.18,
                )
            elif role_name == "perc":
                snippet = _build_chop_stutter(
                    source,
                    repeats=max(2, stutter_repeats - 1),
                    slice_ratio=0.24,
                    spacing_ratio=0.22,
                )
            elif role_name in {"snare", "clap"}:
                snippet = _build_chop_stutter(
                    source,
                    repeats=max(2, stutter_repeats - (0 if next_role == "hook" else 1)),
                    slice_ratio=0.26,
                    spacing_ratio=0.24,
                )
                snippet = _gate_audio_tail(snippet, active_ratio=0.82 if next_role == "hook" else 0.74, floor_gain=0.0)
            else:
                snippet = _gate_audio_tail(source, active_ratio=0.54, floor_gain=0.0)

            snippet = _resize_audio_to_length(snippet, boundary_index - start_index)
            fade = np.linspace(0.72, 1.0 if next_role == "hook" else 0.92, snippet.shape[0], dtype=np.float32)
            snippet *= fade[:, np.newaxis] * layer_gain
            turnaround_bus[start_index:boundary_index] += snippet
            if role_name in room_duck_roles:
                _apply_constant_gain_segment(render, start_index, boundary_index, 0.84 if next_role == "hook" else 0.8 if next_role == "pre_hook" else 0.76)
            affected_layers.append(layer_name)

        if affected_layers:
            events.append(
                {
                    "boundary_bar": int(current_bar),
                    "boundary_seconds": round(float(boundary_index / float(sample_rate)), 4),
                    "from_role": previous_role,
                    "to_role": next_role,
                    "transition_type": transition_type,
                    "fill_length_seconds": round(float((boundary_index - start_index) / float(sample_rate)), 4),
                    "affected_layers": affected_layers[:16],
                    "affected_layers_count": len(affected_layers),
                }
            )

    if not events or not np.any(np.abs(turnaround_bus) > 1e-7):
        return None

    turnaround_bus = np.tanh(turnaround_bus * 1.08).astype(np.float32)
    turnaround_bus = np.clip(turnaround_bus, -0.42, 0.42).astype(np.float32)
    layer_name = "drum-turnaround-bus"
    layer_renders[layer_name] = turnaround_bus
    rendered_layers.append(
        {
            "name": layer_name,
            "role": "drum_fill",
            "pattern": None,
            "trigger_positions": [],
            "source_path": None,
            "copied_input_path": None,
            "render_mode": "drum_turnarounds",
        }
    )

    return {
        "mode": "drum_turnarounds",
        "profile": "boombap_drum_turnarounds_v1",
        "bus_layer_name": layer_name,
        "selected_layers": drum_layer_names,
        "events": events,
        "events_count": len(events),
        "turnaround_peak": round(float(np.max(np.abs(turnaround_bus))) if turnaround_bus.size else 0.0, 6),
    }


def _apply_pattern_drum_section_evolution(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or not layer_renders or not resolved_sections:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    evolving_roles = {"kick", "snare", "clap", "hat", "hihat", "perc", "drum_bed"}
    evolving_layers = [name for name, role in role_by_name.items() if role in evolving_roles and name in layer_renders]
    if not evolving_layers:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in evolving_layers)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    quarter = max(64, int(round(samples_per_bar * 0.25)))
    eighth = max(32, int(round(samples_per_bar * 0.125)))
    sixteenth = max(16, int(round(samples_per_bar * 0.0625)))
    variation_bus = np.zeros((total_samples, 2), dtype=np.float32)
    processed_layers = 0
    bar_events: List[Dict[str, Any]] = []
    current_bar = 0

    def _bar_profile(section_role: str, phase: int) -> Dict[str, Dict[str, float]]:
        normalized_section = _clean_text(section_role).lower() or "verse"
        if normalized_section == "hook":
            if phase % 2 == 0:
                return {
                    "hat": {"gain": 1.04, "width": 1.08},
                    "hihat": {"gain": 1.04, "width": 1.08},
                    "perc": {"gain": 1.06, "width": 1.1},
                    "snare": {"gain": 1.03},
                    "clap": {"gain": 1.04, "width": 1.06},
                    "kick": {"gain": 1.02},
                    "drum_bed": {"gain": 1.03, "width": 1.03},
                }
            return {
                "hat": {"gain": 0.97, "width": 1.04},
                "hihat": {"gain": 0.97, "width": 1.04},
                "perc": {"gain": 1.02, "width": 1.06},
                "snare": {"gain": 1.05},
                "clap": {"gain": 1.06, "width": 1.08},
                "kick": {"gain": 1.0},
                "drum_bed": {"gain": 1.01, "width": 1.02},
            }
        if normalized_section == "pre_hook":
            return {
                "hat": {"gain": 1.05 if phase >= 1 else 0.98, "width": 1.08},
                "hihat": {"gain": 1.05 if phase >= 1 else 0.98, "width": 1.08},
                "perc": {"gain": 1.08, "width": 1.1},
                "snare": {"gain": 1.02},
                "clap": {"gain": 1.03, "width": 1.05},
                "kick": {"gain": 1.01},
                "drum_bed": {"gain": 1.02, "width": 1.02},
            }
        if normalized_section == "break":
            return {
                "hat": {"gain": 0.82, "width": 0.96},
                "hihat": {"gain": 0.82, "width": 0.96},
                "perc": {"gain": 0.84, "width": 0.98},
                "snare": {"gain": 0.94},
                "clap": {"gain": 0.9, "width": 0.98},
                "kick": {"gain": 0.92},
                "drum_bed": {"gain": 0.88, "width": 0.98},
            }
        if normalized_section == "outro":
            return {
                "hat": {"gain": 0.9, "width": 0.98},
                "hihat": {"gain": 0.9, "width": 0.98},
                "perc": {"gain": 0.92, "width": 1.0},
                "snare": {"gain": 0.96},
                "clap": {"gain": 0.94, "width": 1.0},
                "kick": {"gain": 0.95},
                "drum_bed": {"gain": 0.94, "width": 1.0},
            }
        verse_cycle = [
            {
                "hat": {"gain": 0.96, "width": 1.0},
                "hihat": {"gain": 0.96, "width": 1.0},
                "perc": {"gain": 0.94, "width": 1.0},
                "snare": {"gain": 1.0},
                "clap": {"gain": 0.98, "width": 1.0},
                "kick": {"gain": 1.01},
                "drum_bed": {"gain": 0.98, "width": 1.0},
            },
            {
                "hat": {"gain": 1.02, "width": 1.06},
                "hihat": {"gain": 1.02, "width": 1.06},
                "perc": {"gain": 1.04, "width": 1.08},
                "snare": {"gain": 0.99},
                "clap": {"gain": 1.0, "width": 1.03},
                "kick": {"gain": 1.0},
                "drum_bed": {"gain": 1.0, "width": 1.02},
            },
            {
                "hat": {"gain": 0.94, "width": 0.98},
                "hihat": {"gain": 0.94, "width": 0.98},
                "perc": {"gain": 0.98, "width": 1.02},
                "snare": {"gain": 1.03},
                "clap": {"gain": 1.04, "width": 1.04},
                "kick": {"gain": 1.0},
                "drum_bed": {"gain": 0.98, "width": 1.0},
            },
            {
                "hat": {"gain": 1.06, "width": 1.1},
                "hihat": {"gain": 1.06, "width": 1.1},
                "perc": {"gain": 1.08, "width": 1.12},
                "snare": {"gain": 1.04},
                "clap": {"gain": 1.05, "width": 1.06},
                "kick": {"gain": 1.02},
                "drum_bed": {"gain": 1.03, "width": 1.03},
            },
        ]
        return verse_cycle[phase % len(verse_cycle)]

    def _apply_width(render: "np.ndarray", width: float) -> "np.ndarray":
        if abs(width - 1.0) < 1e-4:
            return render
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32) * float(width)
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    for section in resolved_sections:
        section_role = _resolve_section_song_role(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        if bars <= 0:
            continue
        for section_bar_index in range(bars):
            bar_phase = int(section_bar_index) % (2 if section_role == "hook" else 4)
            bar_start = min(total_samples, (current_bar + section_bar_index) * samples_per_bar)
            bar_end = min(total_samples, bar_start + samples_per_bar)
            if bar_end <= bar_start:
                continue
            profile = _bar_profile(section_role, bar_phase)
            for layer_name in evolving_layers:
                render = layer_renders.get(layer_name)
                if render is None or render.size == 0 or bar_start >= render.shape[0]:
                    continue
                role_name = role_by_name.get(layer_name, "")
                segment_end = min(bar_end, render.shape[0])
                if segment_end <= bar_start:
                    continue
                role_profile = profile.get(role_name) or {}
                gain = float(role_profile.get("gain", 1.0))
                width = float(role_profile.get("width", 1.0))
                if abs(gain - 1.0) > 1e-4:
                    _apply_constant_gain_segment(render, bar_start, segment_end, gain)
                if abs(width - 1.0) > 1e-4 and role_name in {"hat", "hihat", "perc", "clap", "drum_bed"}:
                    render[bar_start:segment_end] = _apply_width(render[bar_start:segment_end].copy(), width)
                processed_layers += 1

            should_add_pickup = False
            if section_role == "hook" and bar_phase == 1:
                should_add_pickup = True
            elif section_role in {"verse", "pre_hook"} and bar_phase == 3:
                should_add_pickup = True
            if should_add_pickup:
                pickup_start = max(bar_start, bar_end - quarter)
                pickup_center = max(bar_start, bar_end - eighth)
                pickup_layers: List[str] = []
                for layer_name in evolving_layers:
                    render = layer_renders.get(layer_name)
                    if render is None or render.size == 0 or pickup_start >= render.shape[0]:
                        continue
                    role_name = role_by_name.get(layer_name, "")
                    if role_name not in {"hat", "hihat", "perc", "snare", "clap"}:
                        continue
                    source_end = min(bar_end, render.shape[0])
                    source = render[pickup_start:source_end].copy()
                    if source.shape[0] < 16 or float(np.max(np.abs(source))) < 0.008:
                        continue
                    if role_name in {"hat", "hihat"}:
                        snippet = _build_chop_stutter(
                            source,
                            repeats=3 if section_role == "hook" else 2,
                            slice_ratio=0.18 if section_role == "hook" else 0.22,
                            spacing_ratio=0.16 if section_role == "hook" else 0.2,
                        )
                        gain = 0.34 if section_role == "hook" else 0.24
                    elif role_name == "perc":
                        snippet = _build_chop_stutter(source, repeats=2, slice_ratio=0.24, spacing_ratio=0.22)
                        gain = 0.26 if section_role == "hook" else 0.18
                    else:
                        snippet = _build_chop_stutter(source, repeats=2, slice_ratio=0.28, spacing_ratio=0.24)
                        snippet = _gate_audio_tail(snippet, active_ratio=0.76, floor_gain=0.0)
                        gain = 0.22 if section_role == "hook" else 0.16
                    target_end = min(total_samples, pickup_start + snippet.shape[0])
                    if target_end <= pickup_start:
                        continue
                    faded = snippet[: target_end - pickup_start] * np.linspace(0.72, 1.0, target_end - pickup_start, dtype=np.float32)[:, np.newaxis]
                    variation_bus[pickup_start:target_end] += faded * gain
                    if role_name in {"hat", "hihat"}:
                        _apply_constant_gain_segment(render, pickup_start, min(render.shape[0], bar_end), 0.86 if section_role == "hook" else 0.9)
                    pickup_layers.append(layer_name)
                if pickup_layers:
                    bar_events.append(
                        {
                            "section_role": section_role,
                            "bar_index": int(current_bar + section_bar_index),
                            "bar_phase": int(bar_phase),
                            "event_type": "pickup_variation",
                            "pickup_center_seconds": round(float(pickup_center / float(sample_rate)), 4),
                            "layers": pickup_layers[:12],
                            "layers_count": len(pickup_layers),
                        }
                    )
        current_bar += bars

    if np.any(np.abs(variation_bus) > 1e-7):
        variation_bus = np.tanh(variation_bus * 1.06).astype(np.float32)
        variation_bus = np.clip(variation_bus, -0.32, 0.32).astype(np.float32)
        layer_name = "drum-evolution-bus"
        layer_renders[layer_name] = variation_bus
        rendered_layers.append(
            {
                "name": layer_name,
                "role": "drum_variation",
                "pattern": None,
                "trigger_positions": [],
                "source_path": None,
                "copied_input_path": None,
                "render_mode": "drum_section_evolution",
            }
        )
    else:
        layer_name = ""

    if processed_layers <= 0 and not layer_name:
        return None

    return {
        "mode": "drum_section_evolution",
        "profile": "boombap_drum_section_evolution_v1",
        "bus_layer_name": layer_name or None,
        "processed_layers_count": int(processed_layers),
        "events": bar_events,
        "events_count": len(bar_events),
        "variation_peak": round(float(np.max(np.abs(variation_bus))) if variation_bus.size else 0.0, 6),
    }


def _apply_pattern_drum_kit_character(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or signal is None or not layer_renders or not resolved_sections:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    drum_roles = {"kick", "snare", "clap", "hat", "hihat", "perc", "drum_bed", "drum_fill", "drum_variation"}
    drum_layers = _select_drum_character_layers(
        role_by_name=role_by_name,
        available_layers=[name for name, role in role_by_name.items() if role in drum_roles and name in layer_renders],
    )
    if not drum_layers:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in drum_layers)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    smoothing_samples = max(128, int(round(sample_rate * 0.035)))
    nyquist_hz = sample_rate * 0.5

    thump_env = np.zeros(total_samples, dtype=np.float32)
    snap_env = np.zeros(total_samples, dtype=np.float32)
    air_env = np.zeros(total_samples, dtype=np.float32)
    width_env = np.zeros(total_samples, dtype=np.float32)
    crunch_env = np.zeros(total_samples, dtype=np.float32)
    section_events: List[Dict[str, Any]] = []

    section_profile_map = {
        "intro": {"thump": 0.012, "snap": 0.006, "air": -0.004, "width": 0.0, "crunch": 0.01},
        "verse": {"thump": 0.02, "snap": 0.01, "air": -0.01, "width": -0.01, "crunch": 0.016},
        "pre_hook": {"thump": 0.016, "snap": 0.02, "air": 0.008, "width": 0.04, "crunch": 0.022},
        "hook": {"thump": 0.028, "snap": 0.028, "air": 0.015, "width": 0.08, "crunch": 0.032},
        "break": {"thump": -0.008, "snap": -0.006, "air": -0.01, "width": -0.02, "crunch": 0.008},
        "outro": {"thump": 0.006, "snap": 0.002, "air": -0.006, "width": 0.0, "crunch": 0.01},
    }

    current_bar = 0
    for section in resolved_sections:
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        start_index = min(total_samples, current_bar * samples_per_bar)
        end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
        current_bar += bars
        if end_index <= start_index:
            continue
        energy = float((section or {}).get("energy") or 0.92)
        drum_density = float((section or {}).get("drum_density") or 0.92)
        profile = dict(section_profile_map.get(section_name) or section_profile_map["verse"])
        profile["thump"] += (drum_density - 0.92) * 0.014
        profile["snap"] += (energy - 0.92) * 0.024
        profile["air"] += (energy - 0.92) * 0.018
        profile["width"] += (drum_density - 0.92) * 0.06
        profile["crunch"] += (drum_density - 0.92) * 0.022

        thump_env[start_index:end_index] = float(profile["thump"])
        snap_env[start_index:end_index] = float(profile["snap"])
        air_env[start_index:end_index] = float(profile["air"])
        width_env[start_index:end_index] = float(profile["width"])
        crunch_env[start_index:end_index] = float(profile["crunch"])
        section_events.append(
            {
                "section_name": section_name,
                "bars": int(bars),
                "thump": round(float(profile["thump"]), 4),
                "snap": round(float(profile["snap"]), 4),
                "air": round(float(profile["air"]), 4),
                "width": round(float(profile["width"]), 4),
                "crunch": round(float(profile["crunch"]), 4),
            }
        )

    def _smooth(values: "np.ndarray") -> "np.ndarray":
        if values.size <= smoothing_samples:
            return values.astype(np.float32)
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum <= 0:
            return values.astype(np.float32)
        kernel /= kernel_sum
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    thump_env = _smooth(thump_env)
    snap_env = _smooth(snap_env)
    air_env = _smooth(air_env)
    width_env = _smooth(width_env)
    crunch_env = _smooth(crunch_env)

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(40.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    kick_low_sos = _bandpass(46.0, 150.0)
    kick_click_sos = _bandpass(1600.0, 4200.0)
    snare_body_sos = _bandpass(180.0, 920.0)
    snare_snap_sos = _bandpass(1600.0, 5600.0)
    hat_air_sos = _bandpass(4600.0, 9800.0)
    perc_air_sos = _bandpass(2600.0, 8200.0)
    bed_body_sos = _bandpass(140.0, 880.0)
    bed_air_sos = _bandpass(3200.0, 8400.0)

    processed_layers = 0
    touched_roles: Dict[str, int] = {}
    character_bus = np.zeros((total_samples, 2), dtype=np.float32)

    def _apply_width(render: "np.ndarray", width: "np.ndarray") -> "np.ndarray":
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32)
        side *= (1.0 + width[: side.shape[0]])
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    def _apply_mono_band_mix(
        render: "np.ndarray",
        primary_sos: Optional[Any],
        primary_gain: "np.ndarray",
        secondary_sos: Optional[Any] = None,
        secondary_gain: Optional["np.ndarray"] = None,
        dry_mix_gain: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        processed = render.copy()
        mono = np.mean(processed, axis=1).astype(np.float32)
        adjustment = np.zeros(mono.shape[0], dtype=np.float32)
        if primary_sos is not None:
            adjustment += signal.sosfiltfilt(primary_sos, mono).astype(np.float32) * primary_gain[: mono.shape[0]]
        if secondary_sos is not None and secondary_gain is not None:
            adjustment += signal.sosfiltfilt(secondary_sos, mono).astype(np.float32) * secondary_gain[: mono.shape[0]]
        if dry_mix_gain is not None:
            adjustment += mono * dry_mix_gain[: mono.shape[0]]
        processed[:, 0] += adjustment
        processed[:, 1] += adjustment
        return processed.astype(np.float32)

    for layer_name in drum_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        role = role_by_name.get(layer_name, "")
        processed = render.copy()
        usable_thump = thump_env[: render.shape[0]]
        usable_snap = snap_env[: render.shape[0]]
        usable_air = air_env[: render.shape[0]]
        usable_width = width_env[: render.shape[0]]
        usable_crunch = crunch_env[: render.shape[0]]

        if role == "kick" and kick_low_sos is not None and kick_click_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                kick_low_sos,
                usable_thump * 1.22,
                kick_click_sos,
                usable_snap * 0.42,
            )
        elif role in {"snare", "clap", "drum_fill"} and snare_body_sos is not None and snare_snap_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                snare_body_sos,
                usable_thump * 0.48,
                snare_snap_sos,
                usable_snap * 1.04,
            )
            processed = _apply_width(processed, usable_width * 0.7)
        elif role in {"hat", "hihat"} and hat_air_sos is not None:
            processed = _apply_mono_band_mix(render, hat_air_sos, usable_air * 1.18)
            processed = _apply_width(processed, usable_width * 0.9)
        elif role == "perc" and perc_air_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                perc_air_sos,
                usable_air * 0.82,
                dry_mix_gain=usable_snap * 0.05,
            )
            processed = _apply_width(processed, usable_width * 0.82)
        elif role == "drum_variation":
            processed *= (1.0 + (usable_snap * 0.04))[:, np.newaxis]
            processed = _apply_width(processed, usable_width * 0.7)
        elif role == "drum_bed" and bed_body_sos is not None and bed_air_sos is not None:
            processed = _apply_mono_band_mix(
                render,
                bed_body_sos,
                usable_thump * 0.32,
                bed_air_sos,
                usable_air * 0.54,
            )
            processed = _apply_width(processed, usable_width * 0.55)

        drive = 1.0 + np.clip(usable_crunch, 0.0, 0.08)
        processed = np.tanh(processed * drive[:, np.newaxis]).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        character_bus[: processed.shape[0]] += processed * ((usable_crunch[:, np.newaxis] * 0.18) + 0.01)
        processed_layers += 1
        touched_roles[role] = int(touched_roles.get(role, 0)) + 1

    if processed_layers <= 0:
        return None

    character_bus = np.tanh(character_bus * 1.08).astype(np.float32)
    character_bus = np.clip(character_bus, -0.24, 0.24).astype(np.float32)
    bus_layer_name = "drum-kit-character-bus"
    layer_renders[bus_layer_name] = character_bus
    rendered_layers.append(
        {
            "name": bus_layer_name,
            "role": "drum_character",
            "pattern": None,
            "trigger_positions": [],
            "source_path": None,
            "copied_input_path": None,
            "render_mode": "drum_kit_character",
        }
    )

    return {
        "mode": "drum_kit_character",
        "profile": "boombap_drum_kit_character_v1",
        "bus_layer_name": bus_layer_name,
        "selected_layers": drum_layers,
        "processed_layers_count": int(processed_layers),
        "touched_roles": touched_roles,
        "events": section_events,
        "events_count": len(section_events),
        "character_bus_peak": round(float(np.max(np.abs(character_bus))) if character_bus.size else 0.0, 6),
    }


def _apply_pattern_sample_stack_groove_response(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sample_rate: int,
) -> Optional[Dict[str, Any]]:
    if np is None or signal is None or not layer_renders:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    available_layers = [name for name, role in role_by_name.items() if role in {"melody", "lead", "chop", "texture", "resample", "tonal_glue"} and name in layer_renders]
    sample_layers: List[str] = []
    seen_layers: Set[str] = set()

    def _add_first(predicate: Any) -> None:
        if len(sample_layers) >= 3:
            return
        for layer_name in available_layers:
            if layer_name in seen_layers:
                continue
            if predicate(layer_name):
                sample_layers.append(layer_name)
                seen_layers.add(layer_name)
                return

    _add_first(lambda name: role_by_name.get(name) in {"melody", "lead"})
    _add_first(lambda name: role_by_name.get(name) == "chop" and "hook" in _clean_text(name).lower())
    _add_first(lambda name: role_by_name.get(name) == "chop")
    _add_first(lambda name: role_by_name.get(name) == "texture")
    _add_first(lambda name: role_by_name.get(name) == "resample")
    _add_first(lambda name: role_by_name.get(name) == "tonal_glue")
    if not sample_layers:
        return None

    kick_events: List[Dict[str, Any]] = []
    backbeat_events: List[Dict[str, Any]] = []
    for layer_manifest in rendered_layers:
        role_name = _clean_text(layer_manifest.get("role")).lower()
        triggers = list(layer_manifest.get("trigger_positions") or [])
        if role_name == "kick":
            kick_events.extend(triggers)
        elif role_name in {"snare", "clap"}:
            backbeat_events.extend(triggers)

    if not kick_events and not backbeat_events:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in sample_layers)
    kick_duck_env = np.zeros(total_samples, dtype=np.float32)
    backbeat_push_env = np.zeros(total_samples, dtype=np.float32)
    backbeat_width_env = np.zeros(total_samples, dtype=np.float32)

    kick_hold_ms = {
        "intro": 28.0,
        "verse": 38.0,
        "pre_hook": 34.0,
        "hook": 42.0,
        "break": 24.0,
        "outro": 28.0,
    }
    kick_release_ms = {
        "intro": 74.0,
        "verse": 92.0,
        "pre_hook": 82.0,
        "hook": 98.0,
        "break": 62.0,
        "outro": 76.0,
    }
    kick_amount = {
        "intro": 0.08,
        "verse": 0.12,
        "pre_hook": 0.1,
        "hook": 0.11,
        "break": 0.06,
        "outro": 0.07,
    }
    backbeat_attack_ms = {
        "intro": 10.0,
        "verse": 12.0,
        "pre_hook": 9.0,
        "hook": 8.0,
        "break": 8.0,
        "outro": 10.0,
    }
    backbeat_hold_ms = {
        "intro": 26.0,
        "verse": 32.0,
        "pre_hook": 38.0,
        "hook": 44.0,
        "break": 22.0,
        "outro": 24.0,
    }
    backbeat_release_ms = {
        "intro": 72.0,
        "verse": 88.0,
        "pre_hook": 102.0,
        "hook": 116.0,
        "break": 58.0,
        "outro": 70.0,
    }
    backbeat_amount = {
        "intro": 0.04,
        "verse": 0.07,
        "pre_hook": 0.11,
        "hook": 0.14,
        "break": 0.03,
        "outro": 0.04,
    }
    backbeat_width = {
        "intro": 0.01,
        "verse": 0.03,
        "pre_hook": 0.06,
        "hook": 0.09,
        "break": 0.0,
        "outro": 0.01,
    }

    def _apply_peak_env(buffer: "np.ndarray", start_index: int, attack_samples: int, hold_samples: int, release_samples: int, peak_value: float) -> None:
        if start_index < 0 or start_index >= buffer.shape[0]:
            return
        attack_start = max(0, start_index - attack_samples)
        if start_index > attack_start:
            attack_curve = np.linspace(0.0, float(peak_value), start_index - attack_start, dtype=np.float32)
            buffer[attack_start:start_index] = np.maximum(buffer[attack_start:start_index], attack_curve)
        hold_end = min(buffer.shape[0], start_index + hold_samples)
        buffer[start_index:hold_end] = np.maximum(buffer[start_index:hold_end], float(peak_value))
        release_end = min(buffer.shape[0], hold_end + release_samples)
        if release_end > hold_end:
            release_curve = np.linspace(float(peak_value), 0.0, release_end - hold_end, dtype=np.float32)
            buffer[hold_end:release_end] = np.maximum(buffer[hold_end:release_end], release_curve)

    for event in kick_events:
        start_seconds = float(event.get("start_seconds") or 0.0)
        section_name = _clean_text(event.get("song_role") or event.get("section_name")).lower() or "verse"
        start_index = max(0, min(total_samples - 1, int(round(start_seconds * sample_rate))))
        _apply_peak_env(
            kick_duck_env,
            start_index=start_index,
            attack_samples=max(8, int(round(sample_rate * 0.0035))),
            hold_samples=max(12, int(round((kick_hold_ms.get(section_name, 38.0) / 1000.0) * sample_rate))),
            release_samples=max(16, int(round((kick_release_ms.get(section_name, 92.0) / 1000.0) * sample_rate))),
            peak_value=float(kick_amount.get(section_name, 0.1)),
        )

    for event in backbeat_events:
        start_seconds = float(event.get("start_seconds") or 0.0)
        section_name = _clean_text(event.get("song_role") or event.get("section_name")).lower() or "verse"
        start_index = max(0, min(total_samples - 1, int(round(start_seconds * sample_rate))))
        attack_samples = max(8, int(round((backbeat_attack_ms.get(section_name, 10.0) / 1000.0) * sample_rate)))
        hold_samples = max(12, int(round((backbeat_hold_ms.get(section_name, 32.0) / 1000.0) * sample_rate)))
        release_samples = max(16, int(round((backbeat_release_ms.get(section_name, 88.0) / 1000.0) * sample_rate)))
        _apply_peak_env(
            backbeat_push_env,
            start_index=start_index,
            attack_samples=attack_samples,
            hold_samples=hold_samples,
            release_samples=release_samples,
            peak_value=float(backbeat_amount.get(section_name, 0.08)),
        )
        _apply_peak_env(
            backbeat_width_env,
            start_index=start_index,
            attack_samples=attack_samples,
            hold_samples=hold_samples,
            release_samples=release_samples,
            peak_value=float(backbeat_width.get(section_name, 0.03)),
        )

    smoothing_samples = max(32, int(round(sample_rate * 0.01)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            kick_duck_env = np.convolve(kick_duck_env, kernel, mode="same").astype(np.float32)
            backbeat_push_env = np.convolve(backbeat_push_env, kernel, mode="same").astype(np.float32)
            backbeat_width_env = np.convolve(backbeat_width_env, kernel, mode="same").astype(np.float32)

    nyquist_hz = sample_rate * 0.5

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(40.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    body_sos = _bandpass(180.0, 760.0)
    presence_sos = _bandpass(1500.0, 4600.0)
    air_sos = _bandpass(4200.0, 8600.0)

    processed_layers = 0
    touched_roles: Dict[str, int] = {}
    groove_bus = np.zeros((total_samples, 2), dtype=np.float32)

    def _apply_width(render: "np.ndarray", width_env: "np.ndarray") -> "np.ndarray":
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32)
        side *= (1.0 + width_env[: side.shape[0]])
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    for layer_name in sample_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        role = role_by_name.get(layer_name, "")
        usable_kick = kick_duck_env[: render.shape[0]]
        usable_backbeat = backbeat_push_env[: render.shape[0]]
        usable_width = backbeat_width_env[: render.shape[0]]
        processed = render.copy()

        kick_duck_strength = 0.0
        backbeat_push_strength = 0.0
        air_push_strength = 0.0
        width_strength = 0.0
        if role in {"chop", "resample", "hook_memory"}:
            kick_duck_strength = 1.0
            backbeat_push_strength = 1.0
            air_push_strength = 0.85
            width_strength = 1.0
        elif role in {"melody", "lead"}:
            kick_duck_strength = 0.72
            backbeat_push_strength = 0.8
            air_push_strength = 0.0
            width_strength = 0.42
        elif role == "texture":
            kick_duck_strength = 0.92
            backbeat_push_strength = 0.42
            air_push_strength = 0.6
            width_strength = 0.75
        elif role == "tonal_glue":
            kick_duck_strength = 0.86
            backbeat_push_strength = 0.68
            air_push_strength = 0.72
            width_strength = 0.82

        gain = (1.0 - (usable_kick * (0.16 * kick_duck_strength))) + (usable_backbeat * (0.06 * backbeat_push_strength))
        gain = np.clip(gain, 0.72, 1.08).astype(np.float32)
        processed *= gain[:, np.newaxis]

        if body_sos is not None and presence_sos is not None and air_sos is not None:
            mono = np.mean(processed, axis=1).astype(np.float32)
            adjustment = np.zeros(mono.shape[0], dtype=np.float32)
            if role in {"chop", "resample", "tonal_glue"}:
                body = signal.sosfiltfilt(body_sos, mono).astype(np.float32)
                adjustment -= body * (usable_kick * (0.14 * kick_duck_strength))
            if role in {"melody", "lead", "chop", "resample", "tonal_glue"}:
                presence = signal.sosfiltfilt(presence_sos, mono).astype(np.float32)
                adjustment += presence * (usable_backbeat * (0.12 * backbeat_push_strength))
            if role in {"texture", "chop", "resample", "tonal_glue"} and air_push_strength > 0.0:
                air = signal.sosfiltfilt(air_sos, mono).astype(np.float32)
                adjustment += air * (usable_backbeat * (0.06 * air_push_strength))
            adjustment = adjustment.astype(np.float32)
            processed[:, 0] += adjustment
            processed[:, 1] += adjustment

        if width_strength > 0.0:
            processed = _apply_width(processed, usable_width * (0.95 * width_strength))
        processed = np.tanh(processed * (1.0 + (usable_backbeat[:, np.newaxis] * 0.08 * backbeat_push_strength))).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        groove_bus[: processed.shape[0]] += processed * ((usable_backbeat[:, np.newaxis] * 0.12 * max(0.5, backbeat_push_strength)) + 0.004)
        processed_layers += 1
        touched_roles[role] = int(touched_roles.get(role, 0)) + 1

    if processed_layers <= 0:
        return None

    groove_bus = np.tanh(groove_bus * 1.04).astype(np.float32)
    groove_bus = np.clip(groove_bus, -0.22, 0.22).astype(np.float32)
    bus_layer_name = "sample-groove-response-bus"
    layer_renders[bus_layer_name] = groove_bus
    rendered_layers.append(
        {
            "name": bus_layer_name,
            "role": "sample_groove",
            "pattern": None,
            "trigger_positions": [],
            "source_path": None,
            "copied_input_path": None,
            "render_mode": "sample_stack_groove_response",
        }
    )

    return {
        "mode": "sample_stack_groove_response",
        "profile": "boombap_sample_stack_groove_response_v1",
        "bus_layer_name": bus_layer_name,
        "selected_layers": sample_layers,
        "processed_layers_count": int(processed_layers),
        "kick_events_count": len(kick_events),
        "backbeat_events_count": len(backbeat_events),
        "touched_roles": touched_roles,
        "groove_bus_peak": round(float(np.max(np.abs(groove_bus))) if groove_bus.size else 0.0, 6),
        "min_gain": round(float(np.min(1.0 - (kick_duck_env * 0.16) + (backbeat_push_env * 0.06))) if total_samples else 1.0, 6),
        "max_backbeat_push": round(float(np.max(backbeat_push_env)) if backbeat_push_env.size else 0.0, 6),
    }


def _apply_pattern_sample_stack_section_dialogue(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or signal is None or not layer_renders or not resolved_sections:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    available_layers = [name for name, role in role_by_name.items() if role in {"melody", "lead", "chop", "texture", "resample", "tonal_glue", "sample_groove"} and name in layer_renders]
    sample_layers: List[str] = []
    seen_layers: Set[str] = set()

    def _add_first(predicate: Any) -> None:
        if len(sample_layers) >= 3:
            return
        for layer_name in available_layers:
            if layer_name in seen_layers:
                continue
            if predicate(layer_name):
                sample_layers.append(layer_name)
                seen_layers.add(layer_name)
                return

    _add_first(lambda name: role_by_name.get(name) in {"melody", "lead"})
    _add_first(lambda name: role_by_name.get(name) == "chop" and "hook" in _clean_text(name).lower())
    _add_first(lambda name: role_by_name.get(name) == "sample_groove")
    _add_first(lambda name: role_by_name.get(name) == "chop")
    _add_first(lambda name: role_by_name.get(name) == "texture")
    _add_first(lambda name: role_by_name.get(name) == "resample")
    _add_first(lambda name: role_by_name.get(name) == "tonal_glue")
    if not sample_layers:
        return None

    total_samples = max(layer_renders[name].shape[0] for name in sample_layers)
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    smoothing_samples = max(128, int(round(sample_rate * 0.022)))

    call_env = np.zeros(total_samples, dtype=np.float32)
    answer_env = np.zeros(total_samples, dtype=np.float32)
    width_env = np.zeros(total_samples, dtype=np.float32)
    dry_env = np.zeros(total_samples, dtype=np.float32)
    section_events: List[Dict[str, Any]] = []
    current_bar = 0

    for section in resolved_sections:
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        energy = float((section or {}).get("energy") or 0.92)
        chop_density = float((section or {}).get("chop_density") or 0.84)
        melodic_density = float((section or {}).get("melodic_density") or 0.88)
        for bar_index in range(bars):
            bar_start = min(total_samples, (current_bar + bar_index) * samples_per_bar)
            bar_end = min(total_samples, bar_start + samples_per_bar)
            if bar_end <= bar_start:
                continue
            half = max(bar_start + 1, min(bar_end, bar_start + (samples_per_bar // 2)))
            quarter = max(bar_start + 1, min(bar_end, bar_start + (samples_per_bar // 4)))
            three_quarter = max(half, min(bar_end, bar_start + ((samples_per_bar * 3) // 4)))
            phase = bar_index % (2 if section_name == "hook" else 4)

            if section_name == "verse":
                call_env[bar_start:half] = np.maximum(call_env[bar_start:half], 0.42 + ((melodic_density - 0.88) * 0.16))
                dry_env[half:bar_end] = np.maximum(dry_env[half:bar_end], 0.16 + (0.04 if phase in {1, 3} else 0.0))
                width_env[bar_start:bar_end] = np.maximum(width_env[bar_start:bar_end], -0.04 if phase in {0, 2} else 0.0)
            elif section_name == "pre_hook":
                call_env[bar_start:half] = np.maximum(call_env[bar_start:half], 0.34)
                answer_env[half:bar_end] = np.maximum(answer_env[half:bar_end], 0.4 + ((chop_density - 0.84) * 0.18))
                width_env[quarter:bar_end] = np.maximum(width_env[quarter:bar_end], 0.05)
                dry_env[bar_start:quarter] = np.maximum(dry_env[bar_start:quarter], 0.08)
            elif section_name == "hook":
                call_env[bar_start:quarter] = np.maximum(call_env[bar_start:quarter], 0.28 + ((energy - 0.92) * 0.16))
                answer_env[half:bar_end] = np.maximum(answer_env[half:bar_end], 0.58 + ((chop_density - 0.84) * 0.22))
                answer_env[three_quarter:bar_end] = np.maximum(answer_env[three_quarter:bar_end], 0.72)
                width_env[half:bar_end] = np.maximum(width_env[half:bar_end], 0.1 if phase == 0 else 0.14)
            elif section_name == "break":
                dry_env[bar_start:bar_end] = np.maximum(dry_env[bar_start:bar_end], 0.18)
                width_env[bar_start:bar_end] = np.maximum(width_env[bar_start:bar_end], -0.03)
            elif section_name == "outro":
                call_env[bar_start:half] = np.maximum(call_env[bar_start:half], 0.16)
                dry_env[half:bar_end] = np.maximum(dry_env[half:bar_end], 0.12)
            else:
                call_env[bar_start:half] = np.maximum(call_env[bar_start:half], 0.24)
                answer_env[half:bar_end] = np.maximum(answer_env[half:bar_end], 0.22)

        section_events.append(
            {
                "section_name": section_name,
                "bars": int(bars),
                "energy": round(float(energy), 4),
                "melodic_density": round(float(melodic_density), 4),
                "chop_density": round(float(chop_density), 4),
            }
        )
        current_bar += bars

    def _smooth(values: "np.ndarray") -> "np.ndarray":
        if values.size <= smoothing_samples:
            return values.astype(np.float32)
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum <= 0:
            return values.astype(np.float32)
        kernel /= kernel_sum
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    call_env = np.clip(_smooth(call_env), 0.0, 1.0).astype(np.float32)
    answer_env = np.clip(_smooth(answer_env), 0.0, 1.0).astype(np.float32)
    width_env = np.clip(_smooth(width_env), -0.08, 0.18).astype(np.float32)
    dry_env = np.clip(_smooth(dry_env), 0.0, 0.28).astype(np.float32)

    nyquist_hz = sample_rate * 0.5

    def _bandpass(low_hz: float, high_hz: float) -> Optional[Any]:
        low = max(40.0, float(low_hz))
        high = min(float(high_hz), nyquist_hz * 0.92)
        if high <= low + 60.0:
            return None
        return signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")

    body_sos = _bandpass(220.0, 980.0)
    presence_sos = _bandpass(1500.0, 4800.0)
    air_sos = _bandpass(4200.0, 8400.0)

    dialogue_bus = np.zeros((total_samples, 2), dtype=np.float32)
    processed_layers = 0
    touched_roles: Dict[str, int] = {}

    def _apply_width(render: "np.ndarray", env: "np.ndarray", scale: float) -> "np.ndarray":
        if scale <= 0:
            return render
        mid = ((render[:, 0] + render[:, 1]) * 0.5).astype(np.float32)
        side = ((render[:, 0] - render[:, 1]) * 0.5).astype(np.float32)
        side *= (1.0 + (env[: side.shape[0]] * scale))
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    def _filter_zero_phase(sos: Optional[Any], mono: "np.ndarray") -> "np.ndarray":
        if sos is None:
            return np.zeros(mono.shape[0], dtype=np.float32)
        return signal.sosfiltfilt(sos, mono).astype(np.float32)

    def _filter_fast(sos: Optional[Any], mono: "np.ndarray") -> "np.ndarray":
        if sos is None:
            return np.zeros(mono.shape[0], dtype=np.float32)
        return signal.sosfilt(sos, mono).astype(np.float32)

    for layer_name in sample_layers:
        render = layer_renders.get(layer_name)
        if render is None or render.size == 0:
            continue
        role = role_by_name.get(layer_name, "")
        processed = render.copy()
        usable_call = call_env[: render.shape[0]]
        usable_answer = answer_env[: render.shape[0]]
        usable_width = width_env[: render.shape[0]]
        usable_dry = dry_env[: render.shape[0]]

        call_gain = 0.0
        answer_gain = 0.0
        body_scale = 0.0
        presence_scale = 0.0
        air_scale = 0.0
        width_scale = 0.0
        dry_scale = 0.0

        if role in {"chop", "resample", "hook_memory"}:
            call_gain = 0.015
            answer_gain = 0.07
            body_scale = 0.0
            presence_scale = 0.08
            air_scale = 0.04
            width_scale = 0.85
            dry_scale = 0.65
        elif role in {"melody", "lead"}:
            call_gain = 0.04
            answer_gain = 0.035
            body_scale = 0.0
            presence_scale = 0.04
            air_scale = 0.0
            width_scale = 0.36
            dry_scale = 0.45
        elif role == "texture":
            call_gain = -0.02
            answer_gain = 0.025
            body_scale = 0.0
            presence_scale = 0.0
            air_scale = 0.02
            width_scale = 0.58
            dry_scale = 0.8
        elif role == "tonal_glue":
            call_gain = 0.0
            answer_gain = 0.04
            body_scale = 0.0
            presence_scale = 0.03
            air_scale = 0.015
            width_scale = 0.6
            dry_scale = 0.5
        elif role == "sample_groove":
            call_gain = 0.0
            answer_gain = 0.03
            body_scale = 0.0
            presence_scale = 0.0
            air_scale = 0.0
            width_scale = 0.32
            dry_scale = 0.0

        gain = (1.0 + (usable_call * call_gain) + (usable_answer * answer_gain) - (usable_dry * dry_scale))
        gain = np.clip(gain, 0.74, 1.12).astype(np.float32)
        processed *= gain[:, np.newaxis]

        if body_sos is not None and presence_sos is not None and air_sos is not None:
            mono = np.mean(processed, axis=1).astype(np.float32)
            adjustment = np.zeros(mono.shape[0], dtype=np.float32)
            if body_scale != 0.0 and role in {"melody", "lead", "chop", "resample", "texture", "tonal_glue"}:
                body = _filter_zero_phase(body_sos, mono)
                adjustment += body * ((usable_call * body_scale) + (usable_answer * (body_scale * 0.4)))
            if presence_scale != 0.0 and role in {"melody", "lead", "chop", "resample", "tonal_glue", "sample_groove"}:
                presence = _filter_fast(presence_sos, mono)
                adjustment += presence * ((usable_answer * presence_scale) - (usable_dry * (presence_scale * 0.22)))
            if air_scale != 0.0 and role in {"chop", "resample", "texture", "tonal_glue", "sample_groove"}:
                air = _filter_fast(air_sos, mono)
                adjustment += air * ((usable_answer * air_scale) - (usable_dry * (air_scale * 0.2)))
            adjustment = adjustment.astype(np.float32)
            processed[:, 0] += adjustment
            processed[:, 1] += adjustment

        if width_scale > 0.0:
            processed = _apply_width(processed, usable_width, width_scale)
        processed = np.tanh(processed * (1.0 + (usable_answer[:, np.newaxis] * 0.045))).astype(np.float32)
        layer_renders[layer_name] = np.clip(processed, -1.0, 1.0)
        dialogue_bus[: processed.shape[0]] += processed * (((usable_answer[:, np.newaxis] * 0.08) + (usable_call[:, np.newaxis] * 0.03)) * max(0.3, width_scale or 0.4))
        processed_layers += 1
        touched_roles[role] = int(touched_roles.get(role, 0)) + 1

    if processed_layers <= 0:
        return None

    dialogue_bus = np.tanh(dialogue_bus * 1.05).astype(np.float32)
    dialogue_bus = np.clip(dialogue_bus, -0.2, 0.2).astype(np.float32)
    bus_layer_name = "sample-section-dialogue-bus"
    layer_renders[bus_layer_name] = dialogue_bus
    rendered_layers.append(
        {
            "name": bus_layer_name,
            "role": "sample_dialogue",
            "pattern": None,
            "trigger_positions": [],
            "source_path": None,
            "copied_input_path": None,
            "render_mode": "sample_stack_section_dialogue",
        }
    )

    return {
        "mode": "sample_stack_section_dialogue",
        "profile": "boombap_sample_stack_section_dialogue_v1",
        "bus_layer_name": bus_layer_name,
        "selected_layers": sample_layers,
        "processed_layers_count": int(processed_layers),
        "touched_roles": touched_roles,
        "events": section_events,
        "events_count": len(section_events),
        "dialogue_bus_peak": round(float(np.max(np.abs(dialogue_bus))) if dialogue_bus.size else 0.0, 6),
        "call_peak": round(float(np.max(call_env)) if call_env.size else 0.0, 6),
        "answer_peak": round(float(np.max(answer_env)) if answer_env.size else 0.0, 6),
    }


def _apply_pattern_producer_transitions(
    layer_renders: Dict[str, "np.ndarray"],
    rendered_layers: List[Dict[str, Any]],
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if np is None or not layer_renders or len(resolved_sections) < 2:
        return None

    role_by_name = {
        _clean_text(layer.get("name")): _clean_text(layer.get("role")).lower()
        for layer in rendered_layers
        if _clean_text(layer.get("name"))
    }
    total_samples = max(render.shape[0] for render in layer_renders.values())
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    events: List[Dict[str, Any]] = []
    current_bar = 0

    high_detail_roles = {"hat", "hihat", "perc", "chop", "texture", "melody", "lead", "drum_bed", "resample"}
    harmonic_roles = {"bass", "melody", "lead", "chop", "texture", "resample"}
    drum_roles = {"kick", "snare", "clap", "hat", "hihat", "perc", "drum_bed"}

    for section_index, section in enumerate(resolved_sections[:-1]):
        bars = max(1, int((section or {}).get("bars") or 1))
        current_bar += bars
        next_section = resolved_sections[section_index + 1]
        previous_role = _resolve_section_song_role(section)
        next_role = _resolve_section_song_role(next_section)
        boundary_index = min(total_samples, current_bar * samples_per_bar)
        if boundary_index <= 0 or boundary_index >= total_samples:
            continue

        pre_quarter = max(64, int(round(samples_per_bar * 0.25)))
        pre_eighth = max(32, int(round(samples_per_bar * 0.125)))
        pre_sixteenth = max(16, int(round(samples_per_bar * 0.0625)))
        post_eighth = max(32, int(round(samples_per_bar * 0.125)))
        post_quarter = max(64, int(round(samples_per_bar * 0.25)))
        affected: List[str] = []
        transition_type = "section_glue"

        if next_role == "hook":
            transition_type = "hook_breath_lift"
            for layer_name, render in layer_renders.items():
                role = role_by_name.get(layer_name, "")
                if role in high_detail_roles:
                    _apply_gain_envelope_segment(render, boundary_index - pre_quarter, boundary_index - pre_sixteenth, 1.0, 0.72)
                    _apply_constant_gain_segment(render, boundary_index - pre_sixteenth, boundary_index, 0.24 if role in {"hat", "hihat", "chop"} else 0.42)
                    _apply_gain_envelope_segment(render, boundary_index, boundary_index + post_eighth, 1.08, 1.0)
                    affected.append(layer_name)
                elif role in {"kick", "snare", "clap"}:
                    _apply_gain_envelope_segment(render, boundary_index, boundary_index + post_eighth, 1.08, 1.0)
                    affected.append(layer_name)
                elif role == "bass":
                    _apply_constant_gain_segment(render, boundary_index - pre_sixteenth, boundary_index, 0.58)
                    _apply_gain_envelope_segment(render, boundary_index, boundary_index + post_quarter, 1.04, 1.0)
                    affected.append(layer_name)
            for layer_name, render in layer_renders.items():
                role = role_by_name.get(layer_name, "")
                if role in {"chop", "texture"} and _inject_reverse_transition_swell(render, boundary_index, pre_eighth, 0.18):
                    affected.append(f"{layer_name}:reverse_swell")
        elif next_role == "pre_hook":
            transition_type = "pre_hook_pickup"
            for layer_name, render in layer_renders.items():
                role = role_by_name.get(layer_name, "")
                if role in {"hat", "hihat", "perc", "chop"}:
                    _apply_gain_envelope_segment(render, boundary_index - pre_eighth, boundary_index, 0.9, 1.08)
                    _apply_gain_envelope_segment(render, boundary_index, boundary_index + post_eighth, 1.06, 1.0)
                    affected.append(layer_name)
                elif role in {"melody", "lead", "texture"}:
                    _apply_constant_gain_segment(render, boundary_index - pre_sixteenth, boundary_index, 0.64)
                    affected.append(layer_name)
        elif next_role == "break":
            transition_type = "break_drop"
            for layer_name, render in layer_renders.items():
                role = role_by_name.get(layer_name, "")
                if role in harmonic_roles:
                    _apply_gain_envelope_segment(render, boundary_index - pre_quarter, boundary_index, 1.0, 0.38)
                    _apply_constant_gain_segment(render, boundary_index, boundary_index + post_quarter, 0.42)
                    affected.append(layer_name)
                elif role in drum_roles:
                    _apply_constant_gain_segment(render, boundary_index - pre_sixteenth, boundary_index + post_eighth, 0.72)
                    affected.append(layer_name)
        elif next_role == "outro":
            transition_type = "outro_degrade"
            for layer_name, render in layer_renders.items():
                role = role_by_name.get(layer_name, "")
                if role in high_detail_roles:
                    _apply_gain_envelope_segment(render, boundary_index - pre_quarter, boundary_index + post_quarter, 0.95, 0.66)
                    affected.append(layer_name)
                elif role in {"kick", "bass"}:
                    _apply_gain_envelope_segment(render, boundary_index, boundary_index + post_quarter, 0.92, 0.78)
                    affected.append(layer_name)
        else:
            for layer_name, render in layer_renders.items():
                role = role_by_name.get(layer_name, "")
                if role in {"chop", "texture", "melody", "lead"}:
                    _apply_gain_envelope_segment(render, boundary_index - pre_sixteenth, boundary_index + post_eighth, 0.92, 1.0)
                    affected.append(layer_name)

        if affected:
            events.append(
                {
                    "boundary_bar": int(current_bar),
                    "boundary_seconds": round(float(boundary_index / float(sample_rate)), 4),
                    "from_role": previous_role,
                    "to_role": next_role,
                    "transition_type": transition_type,
                    "affected_layers": affected[:24],
                    "affected_layers_count": len(affected),
                }
            )

    if not events:
        return None
    return {
        "mode": "producer_arrangement_transitions",
        "profile": "boombap_section_moves_v1",
        "events": events,
        "events_count": len(events),
    }


def _render_pattern_section_trigger_variation(
    sample: "np.ndarray",
    role: str,
    section: Optional[Dict[str, Any]],
    sample_rate: int,
    step_index: int,
    layer_variant: Optional[str] = None,
    steps_per_bar: Optional[int] = None,
) -> tuple["np.ndarray", Optional[Dict[str, Any]]]:
    resolved_role = (_clean_text(role) or "sample").lower()
    resolved_variant = _clean_text(layer_variant).lower() or "default"
    if resolved_role not in {"texture", "chop", "melody", "lead", "kick", "snare", "clap", "hat", "hihat", "perc", "resample"}:
        return sample, None

    section_name = _resolve_pattern_section_mix_key(section)
    resolved_steps_per_bar = max(1, int(steps_per_bar or 16))
    phase = int(step_index) % resolved_steps_per_bar
    quarter = max(1, resolved_steps_per_bar // 4)
    eighth = max(1, resolved_steps_per_bar // 8)
    stereo = _ensure_stereo(sample).astype(np.float32)
    mono = stereo.mean(axis=1).astype(np.float32)
    processed = stereo.copy()
    variation_manifest: Dict[str, Any] = {
        "section_name": _clean_text((section or {}).get("name")) or section_name or None,
        "song_role": _clean_text((section or {}).get("song_role")).lower() or section_name or None,
        "role": resolved_role,
        "layer_variant": resolved_variant,
        "step_index": int(step_index),
    }

    if resolved_role == "texture":
        if section_name == "break":
            processed = np.flip(processed, axis=0).copy()
            variation_manifest["transform"] = "reverse_break"
        elif section_name == "hook":
            delayed = np.zeros_like(mono)
            delay_samples = max(16, int(round(sample_rate * 0.012)))
            if delay_samples < mono.shape[0]:
                delayed[delay_samples:] = mono[:-delay_samples]
            processed = np.stack([stereo[:, 0] + (delayed * 0.06), stereo[:, 1] - (delayed * 0.05)], axis=1).astype(np.float32)
            variation_manifest["transform"] = "section_spread"
        elif section_name == "pre_hook":
            transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
            processed += np.stack([transient * 0.04, -transient * 0.03], axis=1).astype(np.float32)
            variation_manifest["transform"] = "fill_shimmer"
    elif resolved_role == "chop":
        transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
        if resolved_variant == "phrase":
            if section_name == "break":
                processed = np.flip(processed, axis=0).copy()
                processed *= 0.72
                variation_manifest["transform"] = "phrase_reverse_break"
            elif section_name == "pre_hook":
                processed += np.stack([transient * 0.1, -transient * 0.08], axis=1).astype(np.float32)
                processed *= 0.92
                variation_manifest["transform"] = "phrase_lift"
            elif section_name == "hook":
                side = ((processed[:, 0] - processed[:, 1]) * 0.5).astype(np.float32) * 1.1
                mid = ((processed[:, 0] + processed[:, 1]) * 0.5).astype(np.float32)
                processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
                variation_manifest["transform"] = "phrase_hook_answer"
            elif section_name in {"intro", "outro"}:
                trim_index = max(1, int(round(processed.shape[0] * 0.62)))
                processed[trim_index:] *= np.linspace(1.0, 0.0, processed.shape[0] - trim_index, dtype=np.float32)[:, np.newaxis]
                processed *= 0.84
                variation_manifest["transform"] = "phrase_trimmed_edge"
            else:
                processed *= 0.9
                variation_manifest["transform"] = "phrase_call"
            if section_name in {"hook", "pre_hook"} and phase >= resolved_steps_per_bar - max(1, eighth + 1):
                stutter = _build_chop_stutter(processed, repeats=2, slice_ratio=0.42, spacing_ratio=0.36)
                processed = (processed * 0.42) + (stutter * 0.78)
                _append_variation_transform(variation_manifest, "phrase_pickup_stutter")
            elif section_name == "verse" and phase in {0, quarter * 2}:
                gated = _gate_audio_tail(processed, active_ratio=0.72, floor_gain=0.18)
                processed = (processed * 0.52) + (gated * 0.58)
                _append_variation_transform(variation_manifest, "phrase_dry_call_gate")
        elif section_name in {"hook", "pre_hook"}:
            processed += np.stack([transient * 0.16, -transient * 0.12], axis=1).astype(np.float32)
            variation_manifest["transform"] = "transient_boost"
        elif section_name == "break":
            processed = np.flip(processed, axis=0).copy()
            processed *= 0.88
            variation_manifest["transform"] = "reverse_break"
        elif section_name in {"intro", "outro"}:
            processed *= 0.92
            variation_manifest["transform"] = "trimmed_edge"
        if resolved_variant == "answer":
            gated = _gate_audio_tail(processed, active_ratio=0.56, floor_gain=0.0)
            processed = (processed * 0.46) + (gated * 0.82)
            if section_name in {"hook", "pre_hook"}:
                processed += np.stack([transient * 0.08, -transient * 0.06], axis=1).astype(np.float32)
            variation_manifest["transform"] = "answer_gate"
        elif resolved_variant == "hook":
            widened = _resize_audio_to_length(processed, max(8, int(round(processed.shape[0] * 0.92))))
            padded = np.zeros_like(processed)
            padded[: min(padded.shape[0], widened.shape[0])] = widened[: padded.shape[0]]
            side = ((padded[:, 0] - padded[:, 1]) * 0.5).astype(np.float32) * 1.18
            mid = ((padded[:, 0] + padded[:, 1]) * 0.5).astype(np.float32)
            processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
            processed *= 1.04
            variation_manifest["transform"] = "hook_bank_push"
        elif resolved_variant == "transition":
            stutter = _build_chop_stutter(processed, repeats=2, slice_ratio=0.26, spacing_ratio=0.18)
            processed = (processed * 0.38) + (stutter * 0.88)
            if section_name in {"pre_hook", "break", "outro"}:
                reversed_gate = _gate_audio_tail(np.flip(processed, axis=0).copy(), active_ratio=0.52, floor_gain=0.0)
                processed = (processed * 0.42) + (reversed_gate * 0.66)
            variation_manifest["transform"] = "transition_bank_stutter"
        elif resolved_variant != "phrase":
            if phase >= resolved_steps_per_bar - max(1, eighth + 1):
                processed = _build_chop_stutter(processed, repeats=3 if section_name in {"hook", "pre_hook"} else 2, slice_ratio=0.32, spacing_ratio=0.24)
                _append_variation_transform(variation_manifest, "mpc_pickup_stutter")
            elif resolved_variant == "alt" and phase in {max(0, quarter - 1), max(0, quarter * 3 - 1)}:
                reversed_gate = _gate_audio_tail(np.flip(processed, axis=0).copy(), active_ratio=0.58, floor_gain=0.0)
                processed = (processed * 0.44) + (reversed_gate * 0.74)
                _append_variation_transform(variation_manifest, "reverse_answer_gate")
            elif section_name == "hook" and phase == 0:
                sped = _resize_audio_to_length(processed, max(8, int(round(processed.shape[0] * 0.94))))
                padded = np.zeros_like(processed)
                padded[: min(padded.shape[0], sped.shape[0])] = sped[: padded.shape[0]]
                processed = (processed * 0.55) + (padded * 0.62)
                _append_variation_transform(variation_manifest, "mpc_pitchy_hook_hit")
    elif resolved_role == "resample":
        if section_name == "hook":
            widened = _resize_audio_to_length(processed, max(8, int(round(processed.shape[0] * 0.9))))
            padded = np.zeros_like(processed)
            padded[: min(padded.shape[0], widened.shape[0])] = widened[: padded.shape[0]]
            processed = (processed * 0.48) + (padded * 0.68)
            variation_manifest["transform"] = "resample_hook_push"
        elif section_name in {"pre_hook", "break"}:
            stutter = _build_chop_stutter(processed, repeats=2, slice_ratio=0.24, spacing_ratio=0.2)
            processed = (processed * 0.42) + (stutter * 0.78)
            variation_manifest["transform"] = "resample_transition_stutter"
        elif section_name == "outro":
            processed = np.flip(processed, axis=0).copy()
            processed *= 0.76
            variation_manifest["transform"] = "resample_outro_reverse"
        else:
            processed *= 0.92
            variation_manifest["transform"] = "resample_phrase"
    elif resolved_role in {"melody", "lead"}:
        if section_name == "break":
            processed *= 0.9
            variation_manifest["transform"] = "dry_break"
        elif section_name == "hook":
            side = ((processed[:, 0] - processed[:, 1]) * 0.5).astype(np.float32)
            mid = ((processed[:, 0] + processed[:, 1]) * 0.5).astype(np.float32)
            side *= 1.12
            processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
            variation_manifest["transform"] = "wider_phrase"
    else:
        mono = processed.mean(axis=1).astype(np.float32)
        transient = np.concatenate([[0.0], np.diff(mono)]).astype(np.float32)
        tail_ratio = 1.0
        gain_scale = 1.0
        width_scale = 1.0
        if resolved_variant == "ghost":
            tail_ratio = 0.52
            gain_scale *= 0.82
            width_scale *= 0.92
        if resolved_role in {"hat", "hihat"}:
            if section_name in {"intro", "outro"}:
                tail_ratio *= 0.72
                gain_scale *= 0.94
                variation_manifest["transform"] = "tight_hat"
            elif section_name in {"hook", "pre_hook"}:
                width_scale *= 1.12
                gain_scale *= 1.02 if step_index % 4 in {1, 3} else 0.98
                variation_manifest["transform"] = "shuffle_hat"
        elif resolved_role in {"snare", "clap"}:
            if section_name in {"hook", "pre_hook"}:
                processed[:, 0] += transient * 0.07
                processed[:, 1] -= transient * 0.06
                width_scale *= 1.08
                variation_manifest["transform"] = "crack_backbeat"
            elif section_name in {"intro", "outro"}:
                tail_ratio *= 0.78
                gain_scale *= 0.92
                variation_manifest["transform"] = "soft_backbeat"
            elif section_name == "break":
                tail_ratio *= 0.64
                variation_manifest["transform"] = "dry_break_hit"
        elif resolved_role == "kick":
            if section_name in {"hook", "pre_hook"}:
                processed[:, 0] += transient * 0.04
                processed[:, 1] += transient * 0.04
                gain_scale *= 1.02
                variation_manifest["transform"] = "punch_kick"
            elif section_name == "break":
                tail_ratio *= 0.8
                gain_scale *= 0.94
                variation_manifest["transform"] = "short_break_kick"
        elif resolved_role == "perc":
            if section_name in {"hook", "pre_hook"}:
                width_scale *= 1.1
                variation_manifest["transform"] = "open_perc"
            elif section_name in {"intro", "outro"}:
                gain_scale *= 0.9
                variation_manifest["transform"] = "soft_perc"

        if tail_ratio < 0.999:
            trim_index = max(1, min(processed.shape[0], int(round(processed.shape[0] * tail_ratio))))
            processed[trim_index:] *= np.linspace(1.0, 0.0, processed.shape[0] - trim_index, dtype=np.float32)[:, np.newaxis]
        if abs(width_scale - 1.0) > 1e-3:
            mid = ((processed[:, 0] + processed[:, 1]) * 0.5).astype(np.float32)
            side = ((processed[:, 0] - processed[:, 1]) * 0.5).astype(np.float32) * width_scale
            processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
        if abs(gain_scale - 1.0) > 1e-3:
            processed *= gain_scale

    processed = np.clip(processed, -1.0, 1.0).astype(np.float32)
    return processed, variation_manifest


def _apply_pattern_macro_mix(
    master: "np.ndarray",
    sections: Optional[Iterable[Dict[str, Any]]],
    sample_rate: int,
    tempo_bpm: float,
) -> Optional[Dict[str, Any]]:
    resolved_sections = list(sections or [])
    if master.size == 0 or not resolved_sections:
        return None

    total_samples = master.shape[0]
    samples_per_bar = max(1, int(round((240.0 / float(max(1.0, tempo_bpm))) * sample_rate)))
    gain_db_envelope = np.zeros(total_samples, dtype=np.float32)
    width_envelope = np.zeros(total_samples, dtype=np.float32)
    body_envelope = np.zeros(total_samples, dtype=np.float32)
    current_bar = 0
    events: List[Dict[str, Any]] = []

    section_gain_map = {
        "intro": -0.35,
        "verse": 0.0,
        "pre_hook": 0.16,
        "hook": 0.34,
        "break": -0.55,
        "outro": -0.28,
    }
    section_width_map = {
        "intro": 0.02,
        "verse": 0.05,
        "pre_hook": 0.08,
        "hook": 0.12,
        "break": -0.04,
        "outro": 0.03,
    }
    section_body_map = {
        "intro": -0.04,
        "verse": 0.0,
        "pre_hook": 0.03,
        "hook": 0.055,
        "break": -0.08,
        "outro": -0.03,
    }

    for section in resolved_sections:
        section_name = _resolve_pattern_section_mix_key(section)
        bars = max(1, int((section or {}).get("bars") or 1))
        start_index = min(total_samples, current_bar * samples_per_bar)
        end_index = min(total_samples, (current_bar + bars) * samples_per_bar)
        if end_index <= start_index:
            current_bar += bars
            continue
        gain_db = float(section_gain_map.get(section_name, 0.0))
        width_gain = float(section_width_map.get(section_name, 0.0))
        body_gain = float(section_body_map.get(section_name, 0.0))
        gain_db_envelope[start_index:end_index] += gain_db
        width_envelope[start_index:end_index] += width_gain
        body_envelope[start_index:end_index] += body_gain
        events.append(
            {
                "section_name": section_name,
                "song_role": _clean_text((section or {}).get("song_role")).lower() or section_name,
                "bars": int(bars),
                "gain_db": round(gain_db, 4),
                "width_gain": round(width_gain, 4),
                "body_gain": round(body_gain, 4),
            }
        )
        current_bar += bars

    smoothing_samples = max(128, int(round(sample_rate * 0.045)))
    if total_samples > smoothing_samples:
        kernel = np.hanning(smoothing_samples).astype(np.float32)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            kernel /= kernel_sum
            gain_db_envelope = np.convolve(gain_db_envelope, kernel, mode="same").astype(np.float32)
            width_envelope = np.convolve(width_envelope, kernel, mode="same").astype(np.float32)
            body_envelope = np.convolve(body_envelope, kernel, mode="same").astype(np.float32)

    gain_linear = np.power(10.0, np.clip(gain_db_envelope, -1.0, 0.7) / 20.0).astype(np.float32)
    mid = ((master[:, 0] + master[:, 1]) * 0.5).astype(np.float32)
    side = ((master[:, 0] - master[:, 1]) * 0.5).astype(np.float32)
    side *= (1.0 + np.clip(width_envelope, -0.12, 0.22))
    body_linear = np.power(10.0, np.clip(body_envelope, -0.12, 0.08)).astype(np.float32)
    mid *= body_linear
    processed = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
    processed *= gain_linear[:, np.newaxis]
    master[:] = np.clip(processed, -1.0, 1.0)
    return {
        "mode": "macro_mix",
        "events": events,
        "smoothing_samples": int(smoothing_samples),
        "max_gain_db": round(float(np.max(gain_db_envelope)) if gain_db_envelope.size else 0.0, 4),
        "min_gain_db": round(float(np.min(gain_db_envelope)) if gain_db_envelope.size else 0.0, 4),
        "max_width_gain": round(float(np.max(width_envelope)) if width_envelope.size else 0.0, 4),
        "min_width_gain": round(float(np.min(width_envelope)) if width_envelope.size else 0.0, 4),
        "max_body_gain": round(float(np.max(body_envelope)) if body_envelope.size else 0.0, 4),
        "min_body_gain": round(float(np.min(body_envelope)) if body_envelope.size else 0.0, 4),
    }


def _extract_style_tokens_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for key in ("genre", "style", "mood"):
        value = _clean_text(metadata.get(key))
        if value:
            tokens.append(value)

    for key in ("genres", "styles", "moods", "tags"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            tokens.extend(_normalize_str_list(value))
        else:
            text = _clean_text(value)
            if text:
                tokens.append(text)
    return _normalize_str_list(tokens)


def _build_external_reference_context(
    project_id: Optional[str],
    source_reference_ids: Optional[Iterable[str]],
) -> Dict[str, Any]:
    if not project_id:
        return {"references": [], "style_tokens": [], "tempos": [], "arrangement_notes": [], "mix_notes": []}

    from tools.audio_gen.project_paths import build_project_paths
    from tools.audio_gen.project_registry import load_reference_assets

    references = load_reference_assets(project_id, reference_ids=source_reference_ids)
    if not references:
        return {"references": [], "style_tokens": [], "tempos": [], "arrangement_notes": [], "mix_notes": []}

    project_paths = build_project_paths(project_id)
    contextual_references: List[Dict[str, Any]] = []
    style_tokens: List[str] = []
    tempos: List[float] = []
    arrangement_notes: List[str] = []
    mix_notes: List[str] = []

    for item in references:
        source_type = _clean_text(item.get("source_type"))
        if source_type not in {"youtube_url", "spotify_url"}:
            continue

        metadata = dict(item.get("metadata") or {})
        imported_metadata = {}
        imported_metadata_path = _clean_text(metadata.get("imported_metadata_path"))
        if imported_metadata_path:
            absolute_metadata_path = os.path.join(project_paths.project_dir, imported_metadata_path)
            imported_metadata = _safe_read_json(absolute_metadata_path)

        provider = _clean_text(item.get("provider"))
        resource_type = _clean_text(metadata.get("resource_type"))
        caption = _clean_text(item.get("caption"))
        provider_id = _clean_text(item.get("provider_id"))
        tags = _normalize_str_list(item.get("tags"))
        title = _clean_text(imported_metadata.get("title") or imported_metadata.get("name") or caption)
        artist = _clean_text(imported_metadata.get("artist"))
        channel = _clean_text(imported_metadata.get("channel"))
        style_tokens.extend(token for token in _extract_style_tokens_from_metadata(imported_metadata) if token)

        tempo_bpm = item.get("tempo_bpm")
        if tempo_bpm is not None:
            try:
                tempos.append(float(tempo_bpm))
            except Exception:
                pass

        display_source = f"{provider} {resource_type}".strip() or source_type
        note_bits = [display_source, title or caption or provider_id]
        if artist:
            note_bits.append(f"artist {artist}")
        if channel:
            note_bits.append(f"channel {channel}")
        arrangement_notes.append("Referencia externa: " + " | ".join(bit for bit in note_bits if bit))

        if tags:
            mix_notes.append(
                f"Considerar a referencia externa '{title or caption or provider_id}' com tags: {', '.join(tags)}."
            )

        contextual_references.append(
            {
                "reference_id": item.get("reference_id"),
                "source_type": source_type,
                "provider": provider,
                "provider_id": provider_id,
                "resource_type": resource_type,
                "caption": caption,
                "source_url": item.get("source_url"),
                "tags": tags,
                "tempo_bpm": item.get("tempo_bpm"),
                "key_signature": item.get("key_signature"),
                "imported_metadata": imported_metadata,
            }
        )

    return {
        "references": contextual_references,
        "style_tokens": _normalize_str_list(style_tokens),
        "tempos": tempos,
        "arrangement_notes": _normalize_str_list(arrangement_notes),
        "mix_notes": _normalize_str_list(mix_notes),
    }


def _summarize_external_reference_metadata(reference: Dict[str, Any], imported_metadata: Dict[str, Any]) -> str:
    provider = _clean_text(reference.get("provider")) or _clean_text(reference.get("source_type")) or "reference"
    resource_type = _clean_text((reference.get("metadata") or {}).get("resource_type"))
    caption = _clean_text(imported_metadata.get("title") or imported_metadata.get("name") or reference.get("caption"))
    artist = _clean_text(imported_metadata.get("artist"))
    channel = _clean_text(imported_metadata.get("channel"))
    style_tokens = _extract_style_tokens_from_metadata(imported_metadata)

    parts = [provider]
    if resource_type:
        parts.append(resource_type)
    if caption:
        parts.append(caption)
    if artist:
        parts.append(f"artist {artist}")
    if channel:
        parts.append(f"channel {channel}")
    if style_tokens:
        parts.append(f"tags {', '.join(style_tokens[:6])}")
    return "; ".join(parts)


def _parse_loudnorm_json(stderr_text: str) -> Optional[Dict[str, Any]]:
    matches = re.findall(r"(\{\s*\"input_i\".*?\})", stderr_text or "", re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def _loudnorm_value(stats: Optional[Dict[str, Any]], key: str, default: float) -> str:
    if not stats:
        return str(default)
    value = _clean_text(stats.get(key))
    if not value or value.lower() in {"nan", "-inf", "inf"}:
        return str(default)
    return value


def _parse_iso8601_duration_seconds(duration_text: str) -> Optional[float]:
    normalized = _clean_text(duration_text).upper()
    if not normalized or not normalized.startswith("P"):
        return None

    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        normalized,
    )
    if not match:
        return None

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    total_seconds = seconds + (minutes * 60) + (hours * 3600) + (days * 86400)
    return float(total_seconds)


def _fetch_youtube_video_metadata(
    source_url: str,
    api_key: Optional[str] = None,
    timeout_seconds: float = 15.0,
) -> Dict[str, Any]:
    from tools.audio_gen.project_registry import parse_reference_source_url

    parsed_reference = parse_reference_source_url(source_url)
    if parsed_reference.get("provider") != "youtube" or parsed_reference.get("resource_type") != "video":
        raise ValueError("A busca de metadata via YouTube Data API exige uma URL de video do YouTube.")

    resolved_api_key = _clean_text(api_key) or _clean_text(os.getenv("YOUTUBE_DATA_API_KEY"))
    if not resolved_api_key:
        raise RuntimeError(
            "A YouTube Data API precisa de uma API key. Informe `youtube_api_key` no request "
            "ou defina `YOUTUBE_DATA_API_KEY` no ambiente."
        )

    response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "snippet,contentDetails",
            "id": parsed_reference["provider_id"],
            "key": resolved_api_key,
        },
        timeout=float(timeout_seconds),
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    if not items:
        raise RuntimeError(f"Nenhum video foi retornado pela YouTube Data API para {parsed_reference['provider_id']}.")

    item = dict(items[0] or {})
    snippet = dict(item.get("snippet") or {})
    content_details = dict(item.get("contentDetails") or {})
    duration_iso8601 = _clean_text(content_details.get("duration"))
    duration_seconds = _parse_iso8601_duration_seconds(duration_iso8601)

    return {
        "title": _clean_text(snippet.get("title")),
        "description": _clean_text(snippet.get("description")),
        "channel": _clean_text(snippet.get("channelTitle")),
        "published_at": _clean_text(snippet.get("publishedAt")),
        "tags": _normalize_str_list(snippet.get("tags")),
        "duration_iso8601": duration_iso8601 or None,
        "duration_seconds": round(float(duration_seconds), 3) if duration_seconds is not None else None,
        "api_source": "youtube_data_api",
        "provider_id": parsed_reference["provider_id"],
        "resource_type": parsed_reference["resource_type"],
    }


@dataclass
class AnalysisRequest:
    audio_path: str
    label: str = ""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    source_reference_ids: List[str] = field(default_factory=list)
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptionRequest:
    audio_path: str
    model_path: Optional[str] = None
    language: str = "auto"
    output_format: str = "json"
    use_gpu: bool = True
    label: str = ""
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    source_reference_ids: List[str] = field(default_factory=list)
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SeparationRequest:
    audio_path: str
    target_stem: str = "vocals"
    model_name: str = "htdemucs"
    label: str = ""
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    source_reference_ids: List[str] = field(default_factory=list)
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeatBlueprintRequest:
    label: str
    reference_audio_paths: List[str]
    style_prompt: str = ""
    style_tags: List[str] = field(default_factory=list)
    target_bpm: Optional[float] = None
    bars: int = DEFAULT_BEAT_BARS
    steps_per_bar: int = DEFAULT_BEAT_STEPS_PER_BAR
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    source_reference_ids: List[str] = field(default_factory=list)
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalReferenceMetadataImportRequest:
    project_id: str
    projects_root: Optional[str] = None
    source_url: Optional[str] = None
    reference_id: Optional[str] = None
    imported_metadata: Dict[str, Any] = field(default_factory=dict)
    role: str = ""
    caption: str = ""
    origin: str = "external_link"
    tags: List[str] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    key_signature: Optional[str] = None
    merge_metadata: Dict[str, Any] = field(default_factory=dict)
    context_label: str = ""
    create_context: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoopbackCaptureRequest:
    project_id: str
    source_url: str
    label: str = ""
    duration_seconds: float = 30.0
    source_mode: str = "manual"
    device_name: str = ""
    open_source_url: bool = False
    launch_delay_seconds: float = 2.5
    sample_rate: int = 48000
    channels: int = DEFAULT_CHANNELS
    block_frames: int = 2048
    background_backend: str = "mpv"
    playback_device_name: str = ""
    background_start_delay_seconds: float = 2.0
    player_extra_args: List[str] = field(default_factory=list)
    role: str = "reference"
    reference_audio_mode: str = "instrumental"
    imported_metadata: Dict[str, Any] = field(default_factory=dict)
    reference_metadata: Dict[str, Any] = field(default_factory=dict)
    reference_tags: List[str] = field(default_factory=list)
    import_youtube_metadata: bool = True
    auto_duration_from_youtube_metadata: bool = True
    youtube_api_key: Optional[str] = None
    capture_padding_seconds: float = 0.75
    create_analysis: bool = True
    create_separation: bool = False
    create_working_set: bool = True
    working_set_label: str = ""
    working_set_description: str = ""
    create_blueprint: bool = True
    style_prompt: str = ""
    style_tags: List[str] = field(default_factory=list)
    target_bpm: Optional[float] = None
    bars: int = DEFAULT_BEAT_BARS
    steps_per_bar: int = DEFAULT_BEAT_STEPS_PER_BAR
    keep_capture: bool = False
    cleanup_after_flow: bool = True
    capture_ttl_hours: int = 24
    projects_root: Optional[str] = None
    run_id: Optional[str] = None
    debug_source_audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaptureCleanupRequest:
    project_id: str
    capture_ids: List[str] = field(default_factory=list)
    cleanup_expired_only: bool = False
    remove_promoted_audio: bool = False
    projects_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternBeatLayer:
    audio_path: str
    pattern: str
    name: str = ""
    role: str = "sample"
    gain_db: float = 0.0
    pan: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternBeatRequest:
    label: str
    tempo_bpm: float
    layers: List[PatternBeatLayer]
    bars: int = DEFAULT_BEAT_BARS
    steps_per_bar: int = DEFAULT_BEAT_STEPS_PER_BAR
    sample_rate: int = DEFAULT_SAMPLE_RATE
    swing: float = 0.0
    timing_jitter_ratio: float = 0.0
    velocity_jitter_db: float = 0.0
    seed: Optional[int] = None
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass
class BlueprintBeatRenderRequest:
    project_id: str
    label: str = ""
    blueprint_version_id: Optional[str] = None
    run_id: Optional[str] = None
    bars_override: Optional[int] = None
    steps_per_bar_override: Optional[int] = None
    target_duration_seconds: Optional[float] = None
    swing: Optional[float] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    creativity: float = 0.35
    fill_probability: float = 0.28
    timing_jitter_ratio: float = 0.04
    velocity_jitter_db: float = 1.5
    seed: Optional[int] = None
    tonal_anchor_key: Optional[str] = None
    tonal_anchor_scale: Optional[str] = None
    tonal_anchor_confidence: Optional[float] = None
    add_supporting_bass: bool = True
    add_melodic_motif: bool = True
    add_reference_texture: bool = True
    add_reference_chops: bool = True
    apply_creative_mix: bool = True
    create_master: bool = False
    master_target_lufs: float = DEFAULT_TARGET_LUFS
    master_true_peak_db: float = DEFAULT_TRUE_PEAK_DB
    persist_to_project: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MixTrackRequest:
    audio_path: str
    label: str = ""
    gain_db: float = 0.0
    start_ms: int = 0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    trim_start_ms: Optional[int] = None
    trim_end_ms: Optional[int] = None
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MixRequest:
    label: str
    tracks: List[MixTrackRequest]
    output_path: Optional[str] = None
    target_peak_db: float = DEFAULT_TARGET_PEAK_DB
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "tracks": [track.to_dict() for track in self.tracks],
        }


@dataclass
class MasteringRequest:
    input_audio_path: str
    label: str = ""
    output_path: Optional[str] = None
    target_lufs: float = DEFAULT_TARGET_LUFS
    loudness_range: float = DEFAULT_LOUDNESS_RANGE
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VocalFitRequest:
    instrumental_audio_path: str
    vocal_audio_path: str
    label: str = ""
    output_path: Optional[str] = None
    ducked_instrumental_path: Optional[str] = None
    prepared_vocal_path: Optional[str] = None
    auto_time_align: bool = True
    max_tempo_adjustment_ratio: float = 0.12
    auto_pitch_align: bool = True
    max_pitch_shift_semitones: int = 2
    vocal_gain_db: Optional[float] = None
    instrumental_gain_db: Optional[float] = None
    ducking_strength: float = 0.32
    output_duration_mode: str = "match_vocal"
    auto_extend_instrumental: bool = True
    instrumental_adaptation_profile: str = "auto"
    enable_vocal_transcription: bool = True
    target_peak_db: float = DEFAULT_TARGET_PEAK_DB
    create_master: bool = False
    master_target_lufs: float = DEFAULT_TARGET_LUFS
    master_true_peak_db: float = DEFAULT_TRUE_PEAK_DB
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    persist_to_project: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_audio(request: AnalysisRequest) -> Dict[str, Any]:
    ensure_runtime_ready("analysis")
    source_path = _resolve_existing_path(request.audio_path)
    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    copied_input_path = _copy_into_directory(source_path, run_dirs["inputs_dir"])

    probe = _ffprobe_audio_metadata(source_path)
    samples, sample_rate = _load_audio_array(
        source_path,
        sample_rate=int(request.sample_rate or DEFAULT_SAMPLE_RATE),
        channels=DEFAULT_CHANNELS,
    )
    analysis = _compute_audio_descriptors(samples, sample_rate)
    librosa_analysis = _compute_librosa_descriptors(source_path, sample_rate=22050)
    if librosa_analysis:
        analysis.update(librosa_analysis)
        if analysis.get("tempo_estimate_bpm") is None and librosa_analysis.get("tempo_estimate_bpm_librosa") is not None:
            analysis["tempo_estimate_bpm"] = librosa_analysis.get("tempo_estimate_bpm_librosa")

    label = _clean_text(request.label) or os.path.splitext(os.path.basename(source_path))[0]
    midi_path = os.path.join(
        run_dirs["analyses_dir"],
        f"{_slugify(label)}_melody.mid",
    )
    notes_path = os.path.join(
        run_dirs["analyses_dir"],
        f"{_slugify(label)}_melody_notes.json",
    )
    melodic_analysis = _extract_basic_pitch_data(
        source_path,
        output_midi_path=midi_path,
        output_notes_path=notes_path,
        midi_tempo=float(analysis.get("tempo_estimate_bpm") or 120.0),
    )
    if melodic_analysis:
        melodic_summary = dict(melodic_analysis.get("summary") or {})
        melodic_summary["notes_path"] = melodic_analysis.get("notes_path")
        melodic_summary["midi_path"] = melodic_analysis.get("midi_path")
        if analysis.get("duration_seconds"):
            melodic_summary = _summarize_basic_pitch_note_events(
                melodic_analysis.get("note_events") or [],
                float(analysis.get("duration_seconds") or 0.0),
            )
            melodic_summary["notes_path"] = melodic_analysis.get("notes_path")
            melodic_summary["midi_path"] = melodic_analysis.get("midi_path")
        analysis["melodic_summary"] = melodic_summary

    audio_stream = next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"),
        {},
    )
    analysis["codec_name"] = audio_stream.get("codec_name")
    analysis["bit_rate"] = audio_stream.get("bit_rate") or probe.get("format", {}).get("bit_rate")
    analysis["format_name"] = probe.get("format", {}).get("format_name")
    analysis["input_path"] = source_path

    analysis_path = os.path.join(
        run_dirs["analyses_dir"],
        f"{_slugify(label)}_analysis.json",
    )
    _write_json(analysis_path, analysis)

    project_context = None
    if request.persist_to_project and request.project_id:
        from tools.audio_gen.project_registry import bootstrap_project_storage

        project_paths = bootstrap_project_storage(request.project_id)
        project_analysis_path = os.path.join(
            project_paths.derived_analyses_dir,
            os.path.basename(analysis_path),
        )
        shutil.copy2(analysis_path, project_analysis_path)
        project_context = add_derived_context(
            project_id=request.project_id,
            context_type="analysis",
            label=label,
            summary=_summarize_analysis(analysis),
            relative_path=os.path.relpath(project_analysis_path, project_paths.project_dir),
            source_reference_ids=request.source_reference_ids,
            metadata={
                "artifact_path": analysis_path,
                "melody_midi_path": melodic_analysis.get("midi_path"),
                "melody_notes_path": melodic_analysis.get("notes_path"),
            },
        )

    manifest = {
        "request": request.to_dict(),
        "input_path": source_path,
        "copied_input_path": copied_input_path,
        "analysis_path": analysis_path,
        "analysis": analysis,
        "melody_midi_path": melodic_analysis.get("midi_path"),
        "melody_notes_path": melodic_analysis.get("notes_path"),
        "project_context": project_context,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "analysis_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def import_external_reference_metadata(request: ExternalReferenceMetadataImportRequest) -> Dict[str, Any]:
    if not _clean_text(request.project_id):
        raise ValueError("project_id deve ser informado para importar metadata de referencia externa.")

    imported_metadata = dict(request.imported_metadata or {})
    if not imported_metadata:
        raise ValueError("imported_metadata deve conter pelo menos um campo de metadata.")
    projects_root = _clean_text(request.projects_root) or DEFAULT_AUDIO_GEN_PROJECTS_DIR

    existing_reference = None
    if _clean_text(request.reference_id):
        existing_reference = find_reference_asset(
            request.project_id,
            reference_id=request.reference_id,
            projects_root=projects_root,
        )
        if existing_reference is None:
            raise ValueError(f"Referencia nao encontrada no projeto: {request.reference_id}")
    elif _clean_text(request.source_url):
        existing_reference = find_reference_asset(
            request.project_id,
            source_url=_clean_text(request.source_url),
            projects_root=projects_root,
        )

    if existing_reference is None:
        if not _clean_text(request.source_url):
            raise ValueError("source_url deve ser informado quando reference_id nao existir.")
        reference = add_reference_link(
            project_id=request.project_id,
            source_url=request.source_url,
            role=_clean_text(request.role) or "reference",
            caption=_clean_text(request.caption),
            origin=_clean_text(request.origin) or "external_link",
            tags=request.tags,
            tempo_bpm=request.tempo_bpm,
            key_signature=request.key_signature,
            imported_metadata=imported_metadata,
            metadata=request.merge_metadata,
            projects_root=projects_root,
        )
    else:
        from tools.audio_gen.project_paths import build_project_paths
        from tools.audio_gen.project_registry import parse_reference_source_url

        project_paths = build_project_paths(request.project_id, projects_root=projects_root)
        reference = dict(existing_reference)
        metadata = dict(reference.get("metadata") or {})
        imported_metadata_path = _clean_text(metadata.get("imported_metadata_path"))
        current_imported = {}
        if imported_metadata_path:
            current_imported = _safe_read_json(os.path.join(project_paths.project_dir, imported_metadata_path))

        merged_imported_metadata = dict(current_imported)
        merged_imported_metadata.update(imported_metadata)

        if not imported_metadata_path:
            parsed_reference = parse_reference_source_url(reference.get("source_url") or request.source_url or "")
            metadata_filename = (
                f"{_slugify(parsed_reference['provider'])}_{_slugify(parsed_reference['resource_type'])}_"
                f"{_slugify(parsed_reference['provider_id'])}_metadata.json"
            )
            stored_metadata_path = os.path.join(project_paths.references_text_dir, metadata_filename)
            imported_metadata_path = os.path.relpath(stored_metadata_path, project_paths.project_dir)
        else:
            stored_metadata_path = os.path.join(project_paths.project_dir, imported_metadata_path)

        _write_json(stored_metadata_path, merged_imported_metadata)

        merged_metadata = dict(metadata)
        merged_metadata.update(dict(request.merge_metadata or {}))
        merged_metadata["imported_metadata_path"] = imported_metadata_path
        merged_metadata["imported_metadata_keys"] = sorted(merged_imported_metadata.keys())

        merged_tags = _normalize_str_list(
            [
                *(reference.get("tags") or []),
                *(request.tags or []),
            ]
        )

        reference = update_reference_asset(
            request.project_id,
            reference["reference_id"],
            projects_root=projects_root,
            caption=_clean_text(request.caption) or reference.get("caption"),
            role=_clean_text(request.role) or reference.get("role"),
            origin=_clean_text(request.origin) or reference.get("origin"),
            relative_path=imported_metadata_path,
            tempo_bpm=request.tempo_bpm if request.tempo_bpm is not None else reference.get("tempo_bpm"),
            key_signature=_clean_text(request.key_signature) or reference.get("key_signature"),
            tags=merged_tags,
            metadata=merged_metadata,
        )
        imported_metadata = merged_imported_metadata

    project_context = None
    if request.create_context:
        context_label = _clean_text(request.context_label) or _clean_text(reference.get("caption")) or "external-reference"
        project_context = add_derived_context(
            project_id=request.project_id,
            context_type="external_reference_metadata",
            label=context_label,
            summary=_summarize_external_reference_metadata(reference, imported_metadata),
            relative_path=reference.get("relative_path"),
            source_reference_ids=[reference["reference_id"]],
            metadata={
                "provider": reference.get("provider"),
                "provider_id": reference.get("provider_id"),
                "source_url": reference.get("source_url"),
                "imported_metadata_keys": sorted(imported_metadata.keys()),
            },
            projects_root=projects_root,
        )

    return {
        "request": request.to_dict(),
        "reference": reference,
        "imported_metadata": imported_metadata,
        "project_context": project_context,
    }


def list_available_loopback_devices() -> Dict[str, Any]:
    devices = list_loopback_devices()
    default_device = next((item for item in devices if item.get("is_default")), None)
    return {
        "devices": devices,
        "default_device": default_device,
        "count": len(devices),
    }


def list_available_audio_routing_devices() -> Dict[str, Any]:
    return list_audio_routing_devices()


def _resolve_youtube_background_vb_devices() -> Dict[str, Optional[str]]:
    routing = list_audio_routing_devices()
    speakers = list(routing.get("speakers") or [])
    loopback_devices = list(routing.get("loopback_devices") or [])

    playback_speaker = next(
        (
            item
            for item in speakers
            if "vb-audio virtual cable" in _clean_text(item.get("name")).lower()
            and "cable input" in _clean_text(item.get("name")).lower()
        ),
        None,
    )
    capture_loopback = next(
        (
            item
            for item in loopback_devices
            if "vb-audio virtual cable" in _clean_text(item.get("name")).lower()
            and "cable output" in _clean_text(item.get("name")).lower()
        ),
        None,
    )

    if capture_loopback is None and playback_speaker is not None:
        capture_loopback = next(
            (
                item
                for item in loopback_devices
                if _clean_text(item.get("name")).lower() == _clean_text(playback_speaker.get("name")).lower()
            ),
            None,
        )

    playback_speaker_id = _clean_text((playback_speaker or {}).get("id"))
    playback_device_name = None
    if playback_speaker_id:
        normalized_id = playback_speaker_id.strip("{}")
        playback_device_name = f"wasapi/{{{normalized_id}}}"

    return {
        "playback_device_name": playback_device_name,
        "playback_device_label": _clean_text((playback_speaker or {}).get("name")) or None,
        "capture_device_name": _clean_text((capture_loopback or {}).get("name")) or None,
    }


def _select_instrumental_separation_reference(
    separation_manifest: Optional[Dict[str, Any]],
    project_id: str,
    projects_root: str,
) -> Optional[Dict[str, Any]]:
    if not separation_manifest:
        return None

    registered_references = list(separation_manifest.get("registered_references") or [])
    preferred_tokens = ("no_vocals", "instrumental", "accompaniment", "minus_vocals")
    for reference in registered_references:
        basename = os.path.basename(_clean_text(reference.get("relative_path"))).lower()
        role = _clean_text(reference.get("role")).lower()
        caption = _clean_text(reference.get("caption")).lower()
        haystack = " ".join([basename, role, caption])
        if any(token in haystack for token in preferred_tokens):
            return reference

    output_files = list(separation_manifest.get("output_files") or [])
    instrumental_output = next(
        (
            item
            for item in output_files
            if any(token in os.path.basename(_clean_text(item)).lower() for token in preferred_tokens)
        ),
        None,
    )
    if instrumental_output:
        instrumental_name = os.path.basename(_clean_text(instrumental_output)).lower()
        for reference in registered_references:
            if os.path.basename(_clean_text(reference.get("relative_path"))).lower() == instrumental_name:
                return reference

    for reference in registered_references:
        basename = os.path.basename(_clean_text(reference.get("relative_path"))).lower()
        role = _clean_text(reference.get("role")).lower()
        caption = _clean_text(reference.get("caption")).lower()
        haystack = " ".join([basename, role, caption])
        if "vocals" in haystack and "no_vocals" not in haystack:
            continue
        return reference

    return None


def _upsert_external_reference_for_capture(
    request: LoopbackCaptureRequest,
    imported_metadata: Dict[str, Any],
    projects_root: str,
) -> Dict[str, Any]:
    existing_reference = find_reference_asset(
        request.project_id,
        source_url=_clean_text(request.source_url),
        projects_root=projects_root,
    )

    if existing_reference is None:
        if imported_metadata:
            imported = import_external_reference_metadata(
                ExternalReferenceMetadataImportRequest(
                    project_id=request.project_id,
                    projects_root=projects_root,
                    source_url=request.source_url,
                    imported_metadata=dict(imported_metadata or {}),
                    role=request.role,
                    caption=request.label,
                    tags=list(request.reference_tags or []),
                    merge_metadata=dict(request.reference_metadata or {}),
                    create_context=False,
                )
            )
            return imported["reference"]

        return add_reference_link(
            project_id=request.project_id,
            source_url=request.source_url,
            role=request.role,
            caption=request.label,
            origin="external_link",
            tags=request.reference_tags,
            imported_metadata=imported_metadata,
            metadata=request.reference_metadata,
            projects_root=projects_root,
        )

    merged_tags = _normalize_str_list(
        [
            *(existing_reference.get("tags") or []),
            *(request.reference_tags or []),
        ]
    )
    merged_metadata = dict(existing_reference.get("metadata") or {})
    merged_metadata.update(dict(request.reference_metadata or {}))
    if imported_metadata:
        imported = import_external_reference_metadata(
            ExternalReferenceMetadataImportRequest(
                project_id=request.project_id,
                projects_root=projects_root,
                reference_id=existing_reference["reference_id"],
                imported_metadata=dict(imported_metadata or {}),
                role=request.role,
                caption=request.label,
                tags=merged_tags,
                merge_metadata=merged_metadata,
                create_context=False,
            )
        )
        return imported["reference"]

    return update_reference_asset(
        request.project_id,
        existing_reference["reference_id"],
        projects_root=projects_root,
        caption=_clean_text(request.label) or existing_reference.get("caption"),
        role=_clean_text(request.role) or existing_reference.get("role"),
        tags=merged_tags,
        metadata=merged_metadata,
    )


def _promote_capture_to_reference_audio(
    project_id: str,
    external_reference: Dict[str, Any],
    capture_session: Dict[str, Any],
    clean_capture_path: str,
    run_id: str,
    projects_root: str,
) -> Dict[str, Any]:
    from tools.audio_gen.project_registry import bootstrap_project_storage

    project_paths = bootstrap_project_storage(project_id, projects_root=projects_root)
    promoted_path = _copy_into_directory(clean_capture_path, project_paths.references_audio_dir)
    promoted_reference = add_reference_asset(
        project_id=project_id,
        source_type="audio",
        role="captured_reference",
        caption=_clean_text(capture_session.get("label")) or os.path.basename(promoted_path),
        relative_path=os.path.relpath(promoted_path, project_paths.project_dir),
        origin="wasapi_loopback_capture",
        tags=_normalize_str_list(
            [
                "captured-reference",
                external_reference.get("provider"),
                *(external_reference.get("tags") or []),
            ]
        ),
        tempo_bpm=external_reference.get("tempo_bpm"),
        key_signature=external_reference.get("key_signature"),
        source_run_id=run_id,
        metadata={
            "capture_id": capture_session.get("capture_id"),
            "linked_source_reference_id": external_reference.get("reference_id"),
            "source_url": external_reference.get("source_url"),
            "temporary_capture": True,
            "capture_cleanup_status": capture_session.get("cleanup_status"),
        },
        projects_root=projects_root,
    )
    link_external_reference_to_audio(
        project_id,
        external_reference["reference_id"],
        promoted_reference["reference_id"],
        capture_id=capture_session.get("capture_id"),
        projects_root=projects_root,
    )
    return update_capture_session(
        project_id,
        capture_session["capture_id"],
        projects_root=projects_root,
        promoted_reference_id=promoted_reference["reference_id"],
        status="promoted",
    )


def cleanup_loopback_capture_artifacts(request: CaptureCleanupRequest) -> Dict[str, Any]:
    projects_root = _clean_text(request.projects_root) or DEFAULT_AUDIO_GEN_PROJECTS_DIR
    sessions = load_capture_sessions(request.project_id, capture_ids=request.capture_ids, projects_root=projects_root)
    now = datetime.now()
    cleaned_sessions: List[Dict[str, Any]] = []

    for session in sessions:
        expires_at = _clean_text(session.get("expires_at"))
        is_expired = False
        if expires_at:
            try:
                is_expired = datetime.fromisoformat(expires_at) <= now
            except ValueError:
                is_expired = False
        if request.cleanup_expired_only and not is_expired:
            continue
        if session.get("keep_capture"):
            continue

        deleted_paths: List[str] = []
        for path in (session.get("raw_capture_path"), session.get("clean_capture_path")):
            if _delete_file_if_exists(path):
                deleted_paths.append(path)

        promoted_reference_id = _clean_text(session.get("promoted_reference_id"))
        promoted_reference = None
        if promoted_reference_id:
            promoted_reference = find_reference_asset(
                request.project_id,
                reference_id=promoted_reference_id,
                projects_root=projects_root,
            )
        if request.remove_promoted_audio and promoted_reference_id:
            if promoted_reference:
                relative_path = _clean_text(promoted_reference.get("relative_path"))
                if relative_path:
                    from tools.audio_gen.project_paths import build_project_paths

                    project_paths = build_project_paths(request.project_id, projects_root=projects_root)
                    absolute_path = os.path.join(project_paths.project_dir, relative_path)
                    if _delete_file_if_exists(absolute_path):
                        deleted_paths.append(absolute_path)
                update_reference_asset(
                    request.project_id,
                    promoted_reference_id,
                    projects_root=projects_root,
                    status="archived",
                    metadata={
                        **dict(promoted_reference.get("metadata") or {}),
                        "cleanup_removed_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
        elif promoted_reference:
            update_reference_asset(
                request.project_id,
                promoted_reference_id,
                projects_root=projects_root,
                metadata={
                    **dict(promoted_reference.get("metadata") or {}),
                    "capture_cleanup_status": "completed" if deleted_paths else "skipped",
                    "capture_cleanup_updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

        updated_session = update_capture_session(
            request.project_id,
            session["capture_id"],
            projects_root=projects_root,
            cleanup_status="completed" if deleted_paths else "skipped",
            status="expired" if request.cleanup_expired_only else session.get("status"),
            metadata={
                **dict(session.get("metadata") or {}),
                "deleted_paths": deleted_paths,
                "cleanup_request": request.to_dict(),
            },
        )
        cleaned_sessions.append(updated_session)

    return {
        "request": request.to_dict(),
        "cleaned_sessions": cleaned_sessions,
        "cleaned_count": len(cleaned_sessions),
    }


def capture_reference_from_loopback(request: LoopbackCaptureRequest) -> Dict[str, Any]:
    if not _clean_text(request.project_id):
        raise ValueError("project_id deve ser informado para capturar referencia via loopback.")
    if float(request.duration_seconds) <= 0:
        raise ValueError("duration_seconds deve ser maior que zero.")
    reference_audio_mode = _clean_text(request.reference_audio_mode).lower() or "instrumental"
    if reference_audio_mode not in {"instrumental", "song_remove_vocals"}:
        raise ValueError("reference_audio_mode deve ser `instrumental` ou `song_remove_vocals`.")

    projects_root = _clean_text(request.projects_root) or DEFAULT_AUDIO_GEN_PROJECTS_DIR
    from tools.audio_gen.project_registry import bootstrap_project_storage
    from tools.audio_gen.project_registry import parse_reference_source_url

    imported_metadata = dict(request.imported_metadata or {})
    youtube_api_metadata = None
    youtube_api_status = "not_applicable"
    background_playback = None
    vb_devices = None
    resolved_duration_seconds = float(request.duration_seconds)
    try:
        parsed_reference = parse_reference_source_url(request.source_url)
    except Exception:
        parsed_reference = {}

    if (
        parsed_reference.get("provider") == "youtube"
        and parsed_reference.get("resource_type") == "video"
        and (request.import_youtube_metadata or request.auto_duration_from_youtube_metadata)
    ):
        resolved_api_key = _clean_text(request.youtube_api_key) or _clean_text(os.getenv("YOUTUBE_DATA_API_KEY"))
        if resolved_api_key:
            youtube_api_metadata = _fetch_youtube_video_metadata(
                request.source_url,
                api_key=resolved_api_key,
            )
            youtube_api_status = "fetched"
            merged_youtube_metadata = dict(youtube_api_metadata)
            merged_youtube_metadata.update(imported_metadata)
            imported_metadata = merged_youtube_metadata

            if request.auto_duration_from_youtube_metadata and youtube_api_metadata.get("duration_seconds") is not None:
                resolved_duration_seconds = float(youtube_api_metadata["duration_seconds"]) + max(
                    0.0,
                    float(request.capture_padding_seconds or 0.0),
                )
        else:
            youtube_api_status = "missing_api_key"

    project_paths = bootstrap_project_storage(request.project_id, projects_root=projects_root)
    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    label = _clean_text(request.label) or f"loopback-{_slugify(request.source_url)}"
    raw_capture_path = os.path.join(
        run_dirs["captures_raw_dir"],
        f"{_slugify(label)}_loopback_raw.wav",
    )
    clean_capture_path = os.path.join(
        run_dirs["captures_clean_dir"],
        f"{_slugify(label)}_loopback_clean.wav",
    )

    external_reference = _upsert_external_reference_for_capture(
        request,
        imported_metadata=imported_metadata,
        projects_root=projects_root,
    )

    if _clean_text(request.debug_source_audio_path):
        debug_source_path = _resolve_existing_path(request.debug_source_audio_path)
        _normalize_audio_to_wav(
            debug_source_path,
            raw_capture_path,
            sample_rate=int(request.sample_rate or 48000),
            channels=int(request.channels or DEFAULT_CHANNELS),
        )
        capture_payload = {
            "output_path": raw_capture_path,
            "duration_seconds": round(float(resolved_duration_seconds), 4),
            "sample_rate": int(request.sample_rate or 48000),
            "channels": int(request.channels or DEFAULT_CHANNELS),
            "device_name": "debug-source-audio",
            "opened_url": False,
            "debug_source_audio_path": debug_source_path,
        }
    elif _clean_text(request.source_mode).lower() == "youtube_background":
        if parsed_reference.get("provider") != "youtube":
            raise ValueError("O modo `youtube_background` so suporta referencias `youtube_url`.")
        ensure_runtime_ready("loopback_capture")
        vb_devices = _resolve_youtube_background_vb_devices()
        if not vb_devices.get("playback_device_name") or not vb_devices.get("capture_device_name"):
            raise RuntimeError(
                "O modo `youtube_background` exige os devices do VB-CABLE. "
                "Nao foi possivel resolver `CABLE Input` para playback e `CABLE Output` para captura."
            )
        background_playback = launch_youtube_background_player(
            request.source_url,
            backend=request.background_backend,
            playback_device_name=vb_devices["playback_device_name"],
            duration_seconds=resolved_duration_seconds,
            extra_args=request.player_extra_args,
        )
        if background_playback.get("stream_info"):
            stream_info = dict(background_playback["stream_info"])
            imported_metadata = {
                **stream_info,
                **imported_metadata,
            }
            external_reference = _upsert_external_reference_for_capture(
                request,
                imported_metadata=imported_metadata,
                projects_root=projects_root,
            )

        try:
            if float(request.background_start_delay_seconds or 0.0) > 0:
                time.sleep(float(request.background_start_delay_seconds))
            capture_payload = capture_wasapi_loopback(
                output_path=raw_capture_path,
                duration_seconds=float(resolved_duration_seconds),
                device_name=vb_devices["capture_device_name"],
                sample_rate=int(request.sample_rate or 48000),
                channels=int(request.channels or DEFAULT_CHANNELS),
                block_frames=int(request.block_frames or 2048),
                open_url=None,
                launch_delay_seconds=0.0,
            )
        finally:
            background_stop = stop_background_player(background_playback.get("process"))
            background_playback["stop"] = background_stop
    else:
        ensure_runtime_ready("loopback_capture")
        capture_payload = capture_wasapi_loopback(
            output_path=raw_capture_path,
            duration_seconds=float(resolved_duration_seconds),
            device_name=request.device_name,
            sample_rate=int(request.sample_rate or 48000),
            channels=int(request.channels or DEFAULT_CHANNELS),
            block_frames=int(request.block_frames or 2048),
            open_url=request.source_url if request.open_source_url else None,
            launch_delay_seconds=float(request.launch_delay_seconds or 0.0),
        )

    cleaned_capture = _hygienize_audio_capture(
        raw_capture_path,
        clean_capture_path,
        sample_rate=int(request.sample_rate or 48000),
        channels=int(request.channels or DEFAULT_CHANNELS),
    )
    capture_session = add_capture_session(
        project_id=request.project_id,
        label=label,
        source_reference_id=external_reference["reference_id"],
        source_url=external_reference.get("source_url"),
        provider=external_reference.get("provider"),
        provider_id=external_reference.get("provider_id"),
        device_name=capture_payload.get("device_name"),
        source_run_id=run_dirs["run_id"],
        raw_capture_path=raw_capture_path,
        clean_capture_path=clean_capture_path,
        status="captured",
        cleanup_status="pending",
        keep_capture=bool(request.keep_capture),
        expires_at=_build_expiration_timestamp(int(request.capture_ttl_hours or 0)),
        notes="Captura de referencia via WASAPI loopback.",
        metadata={
            "capture_payload": capture_payload,
            "cleaned_capture": cleaned_capture,
            "reference_audio_mode": reference_audio_mode,
            "youtube_api_metadata": youtube_api_metadata,
            "youtube_api_status": youtube_api_status,
            "background_playback": {
                key: value
                for key, value in (background_playback or {}).items()
                if key != "process"
            },
            "vb_devices": vb_devices,
            "resolved_duration_seconds": round(float(resolved_duration_seconds), 4),
        },
        projects_root=projects_root,
    )

    promoted_capture_session = _promote_capture_to_reference_audio(
        request.project_id,
        external_reference,
        capture_session,
        clean_capture_path,
        run_dirs["run_id"],
        projects_root=projects_root,
    )
    promoted_reference = find_reference_asset(
        request.project_id,
        reference_id=promoted_capture_session.get("promoted_reference_id"),
        projects_root=projects_root,
    )

    separation_manifest = None
    selected_reference = promoted_reference
    selected_reference_reason = "captured_reference"
    should_remove_vocals = reference_audio_mode == "song_remove_vocals"
    should_run_separation = bool(promoted_reference and (request.create_separation or should_remove_vocals))
    if should_run_separation and promoted_reference:
        separation_manifest = separate_audio(
            SeparationRequest(
                audio_path=os.path.join(project_paths.project_dir, promoted_reference["relative_path"]),
                label=label,
                project_id=request.project_id,
                run_id=run_dirs["run_id"],
                source_reference_ids=[
                    external_reference["reference_id"],
                    promoted_reference["reference_id"],
                ],
                persist_to_project=True,
            )
        )
        if should_remove_vocals:
            instrumental_reference = _select_instrumental_separation_reference(
                separation_manifest,
                request.project_id,
                projects_root=projects_root,
            )
            if instrumental_reference is None:
                raise RuntimeError(
                    "A referencia foi marcada para remocao de voz, mas nenhum stem instrumental foi identificado."
                )
            selected_reference = instrumental_reference
            selected_reference_reason = "instrumental_stem_after_vocal_removal"

    source_reference_ids = _normalize_str_list(
        [
            external_reference["reference_id"],
            promoted_reference["reference_id"] if promoted_reference else None,
            selected_reference["reference_id"] if selected_reference else None,
        ]
    )

    analysis_manifest = None
    project_context = None
    if request.create_analysis and selected_reference:
        analysis_manifest = analyze_audio(
            AnalysisRequest(
                audio_path=os.path.join(project_paths.project_dir, selected_reference["relative_path"]),
                label=label,
                project_id=request.project_id,
                run_id=run_dirs["run_id"],
                source_reference_ids=source_reference_ids,
                persist_to_project=True,
            )
        )
        project_context = analysis_manifest.get("project_context")

    working_set = None
    working_set_reference_ids = [
        external_reference["reference_id"],
        promoted_reference["reference_id"] if promoted_reference else None,
        selected_reference["reference_id"] if selected_reference else None,
    ]
    derived_context_ids = [
        project_context.get("context_id") if project_context else None,
    ]
    if request.create_working_set:
        for registered_reference in (separation_manifest or {}).get("registered_references", []):
            working_set_reference_ids.append(registered_reference.get("reference_id"))
        working_set = add_working_set(
            project_id=request.project_id,
            label=_clean_text(request.working_set_label) or f"{label} working set",
            description=_clean_text(request.working_set_description)
            or "Working set criado a partir de captura loopback de referencia externa.",
            reference_ids=[item for item in working_set_reference_ids if item],
            capture_ids=[promoted_capture_session["capture_id"]],
            derived_context_ids=[item for item in derived_context_ids if item],
            metadata={
                "source_url": external_reference.get("source_url"),
                "capture_id": promoted_capture_session["capture_id"],
                "reference_audio_mode": reference_audio_mode,
                "selected_reference_id": selected_reference["reference_id"] if selected_reference else None,
                "selected_reference_reason": selected_reference_reason,
            },
            set_active=True,
            projects_root=projects_root,
        )

    blueprint_manifest = None
    if request.create_blueprint and selected_reference:
        blueprint_manifest = create_reference_beat_blueprint(
            BeatBlueprintRequest(
                label=label,
                reference_audio_paths=[os.path.join(project_paths.project_dir, selected_reference["relative_path"])],
                style_prompt=request.style_prompt,
                style_tags=list(request.style_tags or []),
                target_bpm=request.target_bpm,
                bars=int(request.bars or DEFAULT_BEAT_BARS),
                steps_per_bar=int(request.steps_per_bar or DEFAULT_BEAT_STEPS_PER_BAR),
                project_id=request.project_id,
                run_id=run_dirs["run_id"],
                source_reference_ids=source_reference_ids,
                persist_to_project=True,
            )
        )

    cleanup_manifest = None
    if request.cleanup_after_flow and not request.keep_capture:
        cleanup_manifest = cleanup_loopback_capture_artifacts(
            CaptureCleanupRequest(
                project_id=request.project_id,
                capture_ids=[promoted_capture_session["capture_id"]],
                cleanup_expired_only=False,
                remove_promoted_audio=False,
                projects_root=projects_root,
            )
        )
        promoted_capture_session = find_capture_session(
            request.project_id,
            capture_id=promoted_capture_session["capture_id"],
            projects_root=projects_root,
        ) or promoted_capture_session

    manifest = {
        "request": request.to_dict(),
        "external_reference": external_reference,
        "reference_audio_mode": reference_audio_mode,
        "youtube_api_metadata": youtube_api_metadata,
        "youtube_api_status": youtube_api_status,
        "background_playback": {
            key: value
            for key, value in (background_playback or {}).items()
            if key != "process"
        }
        if background_playback
        else None,
        "vb_devices": vb_devices,
        "resolved_duration_seconds": round(float(resolved_duration_seconds), 4),
        "capture_payload": capture_payload,
        "cleaned_capture": cleaned_capture,
        "capture_session": promoted_capture_session,
        "promoted_reference": promoted_reference,
        "selected_reference": selected_reference,
        "selected_reference_reason": selected_reference_reason,
        "analysis_manifest": analysis_manifest,
        "separation_manifest": separation_manifest,
        "working_set": working_set,
        "blueprint_manifest": blueprint_manifest,
        "cleanup_manifest": cleanup_manifest,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "loopback_reference_capture_manifest.json")
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    return manifest


def transcribe_audio(request: TranscriptionRequest) -> Dict[str, Any]:
    output_format = _clean_text(request.output_format).lower() or "json"
    if output_format not in {"text", "json", "srt"}:
        raise ValueError("output_format deve ser um de: text, json, srt.")

    status = ensure_runtime_ready(
        "transcription",
        whisper_model_path=request.model_path,
        models_root=DEFAULT_AUDIO_GEN_MODELS_DIR,
    )
    source_path = _resolve_existing_path(request.audio_path)
    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    copied_input_path = _copy_into_directory(source_path, run_dirs["inputs_dir"])

    model_path = status["resolved_whisper_model_path"]
    ffmpeg_path = status["ffmpeg_path"]
    label = _clean_text(request.label) or os.path.splitext(os.path.basename(source_path))[0]
    extension = {"text": ".txt", "json": ".json", "srt": ".srt"}[output_format]
    transcript_path = os.path.join(
        run_dirs["transcripts_dir"],
        f"{_slugify(label)}_transcript{extension}",
    )
    filter_model_path = _normalize_ffmpeg_filter_path(model_path)
    filter_transcript_path = _normalize_ffmpeg_filter_path(transcript_path)

    whisper_filter = (
        f"whisper=model={_escape_ffmpeg_filter_value(filter_model_path)}"
        f":language={_escape_ffmpeg_filter_value(_clean_text(request.language) or 'auto')}"
        f":destination={_escape_ffmpeg_filter_value(filter_transcript_path)}"
        f":format={output_format}"
        f":use_gpu={'true' if request.use_gpu else 'false'}"
    )
    result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            source_path,
            "-vn",
            "-af",
            whisper_filter,
            "-f",
            "null",
            _ffmpeg_null_output(),
        ]
    )
    if result.returncode != 0:
        _raise_command_error("ffmpeg whisper", result)
    if not os.path.exists(transcript_path):
        raise RuntimeError("A transcricao foi executada, mas o arquivo de saida nao foi encontrado.")

    raw_content = _read_text(transcript_path)
    parsed_content: Any = raw_content
    if output_format == "json":
        try:
            parsed_content = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_content = raw_content

    project_context = None
    if request.persist_to_project and request.project_id:
        from tools.audio_gen.project_registry import bootstrap_project_storage

        project_paths = bootstrap_project_storage(request.project_id)
        project_transcript_path = os.path.join(
            project_paths.derived_transcripts_dir,
            os.path.basename(transcript_path),
        )
        shutil.copy2(transcript_path, project_transcript_path)
        summary = raw_content.strip().replace("\n", " ")[:180]
        project_context = add_derived_context(
            project_id=request.project_id,
            context_type="transcription",
            label=label,
            summary=summary,
            relative_path=os.path.relpath(project_transcript_path, project_paths.project_dir),
            source_reference_ids=request.source_reference_ids,
            metadata={"artifact_path": transcript_path, "format": output_format},
        )

    manifest = {
        "request": request.to_dict(),
        "input_path": source_path,
        "copied_input_path": copied_input_path,
        "model_path": model_path,
        "transcript_path": transcript_path,
        "transcript": parsed_content,
        "project_context": project_context,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "transcription_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def separate_audio(request: SeparationRequest) -> Dict[str, Any]:
    status = ensure_runtime_ready("separation")
    source_path = _resolve_existing_path(request.audio_path)
    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    copied_input_path = _copy_into_directory(source_path, run_dirs["inputs_dir"])

    demucs_cli_path = status.get("demucs_cli_path")
    if demucs_cli_path:
        command = [
            demucs_cli_path,
            "--name",
            _clean_text(request.model_name) or "htdemucs",
            "--two-stems",
            _clean_text(request.target_stem) or "vocals",
            "-o",
            run_dirs["separated_dir"],
            source_path,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "demucs",
            "--name",
            _clean_text(request.model_name) or "htdemucs",
            "--two-stems",
            _clean_text(request.target_stem) or "vocals",
            "-o",
            run_dirs["separated_dir"],
            source_path,
        ]

    result = _run_command(command)
    if result.returncode != 0:
        _raise_command_error("demucs", result)

    output_files: List[str] = []
    for root, _, filenames in os.walk(run_dirs["separated_dir"]):
        for filename in filenames:
            if filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                output_files.append(os.path.join(root, filename))
    output_files.sort()

    registered_references: List[Dict[str, Any]] = []
    if request.persist_to_project and request.project_id:
        from tools.audio_gen.project_registry import bootstrap_project_storage

        project_paths = bootstrap_project_storage(request.project_id)
        for stem_path in output_files:
            copied_stem_path = _copy_into_directory(stem_path, project_paths.references_audio_dir)
            registered_references.append(
                add_reference_asset(
                    project_id=request.project_id,
                    source_type="audio",
                    role=f"separated_{_slugify(os.path.splitext(os.path.basename(stem_path))[0])}",
                    caption=os.path.basename(stem_path),
                    relative_path=os.path.relpath(copied_stem_path, project_paths.project_dir),
                    origin="separation_run",
                    tags=["separated", request.target_stem],
                    source_run_id=run_dirs["run_id"],
                    metadata={"artifact_path": stem_path},
                )
            )

    manifest = {
        "request": request.to_dict(),
        "input_path": source_path,
        "copied_input_path": copied_input_path,
        "output_files": output_files,
        "registered_references": registered_references,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "separation_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def create_reference_beat_blueprint(request: BeatBlueprintRequest) -> Dict[str, Any]:
    ensure_runtime_ready("analysis")
    if not request.reference_audio_paths:
        raise ValueError("Informe pelo menos um audio de referencia para montar o blueprint.")

    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    reference_summaries: List[Dict[str, Any]] = []
    tempos: List[float] = []
    inferred_roles: List[str] = []
    collected_moods: List[str] = []
    external_reference_context = _build_external_reference_context(
        request.project_id,
        request.source_reference_ids,
    )
    tempos.extend(external_reference_context["tempos"])

    for reference_path in request.reference_audio_paths:
        source_path = _resolve_existing_path(reference_path)
        samples, sample_rate = _load_audio_array(source_path, sample_rate=22050, channels=2)
        descriptors = _compute_audio_descriptors(samples, sample_rate)
        librosa_descriptors = _compute_librosa_descriptors(source_path, sample_rate=22050)
        if librosa_descriptors:
            descriptors.update(librosa_descriptors)
            if descriptors.get("tempo_estimate_bpm") is None and librosa_descriptors.get("tempo_estimate_bpm_librosa") is not None:
                descriptors["tempo_estimate_bpm"] = librosa_descriptors.get("tempo_estimate_bpm_librosa")
        role = _infer_reference_role_from_path(source_path)
        inferred_roles.append(role)
        if descriptors.get("tempo_estimate_bpm") is not None:
            tempos.append(float(descriptors["tempo_estimate_bpm"]))
        collected_moods.extend(descriptors.get("mood_tags") or [])
        reference_summaries.append(
            {
                "path": source_path,
                "role": role,
                "analysis": descriptors,
            }
        )

    target_bpm = float(request.target_bpm) if request.target_bpm is not None else (
        round(float(median(tempos)), 2) if tempos else 92.0
    )
    style_tags = _normalize_instrumental_style_tags(request.style_tags)
    for mood in collected_moods:
        if mood not in style_tags:
            style_tags.append(mood)
    for token in external_reference_context["style_tokens"]:
        if token not in style_tags:
            style_tags.append(token)

    roles_set = set(inferred_roles)
    recommended_layers = []
    for role, pattern in (
        ("kick", _resolve_default_blueprint_pattern("kick", style_tags)),
        ("snare", _resolve_default_blueprint_pattern("snare", style_tags)),
        ("clap", _resolve_default_blueprint_pattern("clap", style_tags)),
        ("hat", _resolve_default_blueprint_pattern("hat", style_tags)),
        ("bass", _resolve_default_blueprint_pattern("bass", style_tags)),
        ("melody", _resolve_default_blueprint_pattern("melody", style_tags)),
    ):
        if role in roles_set or role in {"kick", "snare", "hat"}:
            recommended_layers.append(
                {
                    "name": role,
                    "role": role,
                    "pattern": pattern,
                    "source_reference_ids": request.source_reference_ids,
                    "notes": "Camada sugerida automaticamente a partir das referencias analisadas.",
                    "metadata": {},
                }
            )

    arrangement_notes = [
        f"Target BPM sugerido: {target_bpm}",
        "Priorizar espaco para voz se o material de referencia parecer song-oriented.",
        "Usar o spectral centroid das referencias para decidir entre timbres mais escuros ou mais brilhantes.",
    ]
    arrangement_notes.extend(external_reference_context["arrangement_notes"])
    if request.style_prompt:
        arrangement_notes.append(f"Direcao textual: {request.style_prompt}")

    mix_notes = [
        "Deixar o kick e o bass compartilharem o low-end com sidechain ou recorte complementar.",
        "Evitar excesso de energia acima de 8 kHz se o material de referencia for dark/sub-heavy.",
    ]
    mix_notes.extend(external_reference_context["mix_notes"])

    blueprint = {
        "label": _clean_text(request.label),
        "summary": (
            f"Blueprint gerado a partir de {len(reference_summaries)} referencia(s), "
            f"com target BPM sugerido em {target_bpm}."
        ),
        "target_bpm": target_bpm,
        "bars": max(1, int(request.bars)),
        "steps_per_bar": max(1, int(request.steps_per_bar)),
        "style_tags": style_tags,
        "reference_summaries": reference_summaries,
        "external_reference_context": external_reference_context["references"],
        "recommended_layers": recommended_layers,
        "arrangement_notes": _normalize_str_list(arrangement_notes),
        "mix_notes": _normalize_str_list(mix_notes),
    }

    blueprint_artifact_path = os.path.join(
        run_dirs["beats_dir"],
        f"{_slugify(request.label)}_blueprint.json",
    )
    _write_json(blueprint_artifact_path, blueprint)

    project_blueprint = None
    if request.persist_to_project and request.project_id:
        from tools.audio_gen.project_registry import bootstrap_project_storage

        project_paths = bootstrap_project_storage(request.project_id)
        project_blueprint_path = os.path.join(
            project_paths.beat_blueprint_versions_dir,
            os.path.basename(blueprint_artifact_path),
        )
        shutil.copy2(blueprint_artifact_path, project_blueprint_path)
        project_blueprint = add_beat_blueprint(
            project_id=request.project_id,
            label=request.label,
            summary=blueprint["summary"],
            relative_path=os.path.relpath(project_blueprint_path, project_paths.project_dir),
            source_reference_ids=request.source_reference_ids,
            style_tags=style_tags,
            target_bpm=target_bpm,
            bars=request.bars,
            steps_per_bar=request.steps_per_bar,
            layers=recommended_layers,
            arrangement_notes=blueprint["arrangement_notes"],
            mix_notes=blueprint["mix_notes"],
            metadata={"artifact_path": blueprint_artifact_path},
        )

    manifest = {
        "request": request.to_dict(),
        "blueprint_artifact_path": blueprint_artifact_path,
        "blueprint": blueprint,
        "project_blueprint": project_blueprint,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "beat_blueprint_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def render_pattern_beat(request: PatternBeatRequest) -> Dict[str, Any]:
    ensure_runtime_ready("beat_rendering")
    if np is None:
        raise RuntimeError("NumPy nao esta disponivel no runtime atual.")
    if float(request.tempo_bpm) <= 0:
        raise ValueError("tempo_bpm deve ser maior que zero.")
    if not request.layers:
        raise ValueError("Informe pelo menos uma camada para renderizar o beat.")

    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    bars = max(1, int(request.bars))
    steps_per_bar = max(1, int(request.steps_per_bar))
    total_steps = bars * steps_per_bar
    seconds_per_bar = 240.0 / float(request.tempo_bpm)
    seconds_per_step = seconds_per_bar / steps_per_bar
    swing = max(0.0, min(0.45, float(request.swing)))
    timing_jitter_ratio = max(0.0, min(0.45, float(request.timing_jitter_ratio or 0.0)))
    velocity_jitter_db = max(0.0, min(9.0, float(request.velocity_jitter_db or 0.0)))
    jitter_seed = (
        int(request.seed)
        if request.seed is not None
        else (zlib.crc32(f"{request.label}:{request.tempo_bpm}:{bars}:{steps_per_bar}".encode("utf-8")) & 0xFFFFFFFF)
    )
    rng = np.random.default_rng(jitter_seed)
    render_started_at = time.perf_counter()
    stage_timings_ms: Dict[str, float] = {}

    def _record_stage_timing(stage_name: str, started_at: float) -> None:
        stage_timings_ms[stage_name] = round((time.perf_counter() - started_at) * 1000.0, 3)

    rendered_layers: List[Dict[str, Any]] = []
    longest_sample_length = 0
    prepared_layers = []
    stage_started_at = time.perf_counter()
    for layer in request.layers:
        source_path = _resolve_existing_path(layer.audio_path)
        copied_input_path = _copy_into_directory(source_path, run_dirs["inputs_dir"])
        sample_array, sample_rate = _load_audio_array(
            source_path,
            sample_rate=int(request.sample_rate or DEFAULT_SAMPLE_RATE),
            channels=DEFAULT_CHANNELS,
        )
        processed_sample = _apply_gain_and_pan(sample_array, gain_db=layer.gain_db, pan=layer.pan)
        longest_sample_length = max(longest_sample_length, processed_sample.shape[0])
        prepared_layers.append(
            {
                "layer": layer,
                "source_path": source_path,
                "copied_input_path": copied_input_path,
                "sample_rate": sample_rate,
                "sample": processed_sample,
                "pattern": _parse_pattern(layer.pattern, bars=bars, steps_per_bar=steps_per_bar),
            }
        )
    _record_stage_timing("prepare_layers", stage_started_at)

    base_total_samples = int(math.ceil((total_steps * seconds_per_step) * request.sample_rate))
    required_total_samples = base_total_samples
    for prepared in prepared_layers:
        pattern = prepared["pattern"]
        sample = prepared["sample"]
        active_steps = [index for index, is_active in enumerate(pattern) if is_active]
        if not active_steps:
            continue
        last_step_index = max(active_steps)
        last_start_seconds = last_step_index * seconds_per_step
        if swing > 0 and last_step_index % 2 == 1:
            last_start_seconds += seconds_per_step * swing * 0.5
        required_total_samples = max(
            required_total_samples,
            int(round(last_start_seconds * request.sample_rate)) + sample.shape[0],
        )

    groove_tail_margin = int(round(seconds_per_step * 0.12 * request.sample_rate))
    total_samples = max(required_total_samples + max(0, groove_tail_margin), base_total_samples)
    master = np.zeros((total_samples, 2), dtype=np.float32)
    layer_renders: Dict[str, "np.ndarray"] = {}

    stage_started_at = time.perf_counter()
    for prepared in prepared_layers:
        layer = prepared["layer"]
        pattern = prepared["pattern"]
        sample = prepared["sample"]
        sections = list((layer.metadata or {}).get("sections") or [])
        layer_render = np.zeros((total_samples, 2), dtype=np.float32)
        trigger_positions = []
        for step_index, active in enumerate(pattern):
            if not active:
                continue
            start_seconds = step_index * seconds_per_step
            if swing > 0 and step_index % 2 == 1:
                start_seconds += seconds_per_step * swing * 0.5
            timing_offset_seconds = 0.0
            if timing_jitter_ratio > 0:
                timing_offset_seconds = float(rng.uniform(-1.0, 1.0)) * seconds_per_step * timing_jitter_ratio
                start_seconds = max(0.0, start_seconds + timing_offset_seconds)
            triggered_sample = sample
            gain_offset_db = 0.0
            groove_manifest = None
            groove_offset_seconds = 0.0
            groove_gain_db = 0.0
            section_info = _resolve_pattern_section_for_step(
                step_index=step_index,
                steps_per_bar=steps_per_bar,
                sections=sections,
            )
            groove_manifest = _resolve_pattern_boombap_trigger_groove(
                role=layer.role,
                section=section_info,
                step_index=int(step_index),
                steps_per_bar=steps_per_bar,
                seconds_per_step=seconds_per_step,
                layer_variant=_clean_text((layer.metadata or {}).get("layer_variant")),
            )
            if groove_manifest:
                groove_offset_seconds = float(groove_manifest.get("offset_seconds") or 0.0)
                groove_gain_db = float(groove_manifest.get("gain_db") or 0.0)
                start_seconds = max(0.0, start_seconds + groove_offset_seconds)
            start_index = int(round(start_seconds * request.sample_rate))
            section_gain_db = _resolve_pattern_section_gain_db(layer.role, section_info)
            total_trigger_gain_db = float(section_gain_db) + float(groove_gain_db)
            trigger_variation_manifest = None
            if _clean_text(layer.role).lower() in {"texture", "chop", "melody", "lead", "kick", "snare", "clap", "hat", "hihat", "perc", "resample"}:
                triggered_sample, trigger_variation_manifest = _render_pattern_section_trigger_variation(
                    sample=triggered_sample,
                    role=layer.role,
                    section=section_info,
                    sample_rate=int(request.sample_rate),
                    step_index=int(step_index),
                    layer_variant=_clean_text((layer.metadata or {}).get("layer_variant")),
                    steps_per_bar=steps_per_bar,
                )
            if velocity_jitter_db > 0:
                gain_offset_db = float(rng.uniform(-velocity_jitter_db, velocity_jitter_db))
                total_trigger_gain_db += gain_offset_db
            if total_trigger_gain_db != 0.0:
                triggered_sample = triggered_sample * float(10 ** (total_trigger_gain_db / 20.0))
            end_index = min(master.shape[0], start_index + triggered_sample.shape[0])
            if end_index <= start_index:
                continue
            layer_render[start_index:end_index] += triggered_sample[: end_index - start_index]
            trigger_entry = {
                "step_index": step_index,
                "start_seconds": round(float(start_seconds), 4),
                "timing_offset_seconds": round(float(timing_offset_seconds), 4),
                "groove_offset_seconds": round(float(groove_offset_seconds), 5),
                "gain_offset_db": round(float(gain_offset_db), 4),
                "groove_gain_db": round(float(groove_gain_db), 4),
                "section_gain_db": round(float(section_gain_db), 4),
                "section_name": _clean_text((section_info or {}).get("name")) or None,
                "song_role": _clean_text((section_info or {}).get("song_role")).lower() or None,
                "section_family": _clean_text((section_info or {}).get("section_family")).lower() or None,
            }
            if groove_manifest:
                trigger_entry["groove"] = groove_manifest
            if trigger_variation_manifest:
                trigger_entry["section_variation"] = trigger_variation_manifest
            trigger_positions.append(trigger_entry)

        layer_name = layer.name or os.path.splitext(os.path.basename(layer.audio_path))[0]
        layer_renders[layer_name] = layer_render
        rendered_layers.append(
            {
                "name": layer_name,
                "role": layer.role,
                "pattern": layer.pattern,
                "trigger_positions": trigger_positions,
                "source_path": prepared["source_path"],
                "copied_input_path": prepared["copied_input_path"],
            }
        )
    _record_stage_timing("render_triggers", stage_started_at)

    section_mix_manifests: Dict[str, Dict[str, Any]] = {}
    stage_started_at = time.perf_counter()
    for prepared in prepared_layers:
        layer = prepared["layer"]
        layer_name = layer.name or os.path.splitext(os.path.basename(layer.audio_path))[0]
        layer_render = layer_renders.get(layer_name)
        if layer_render is None:
            continue
        section_mix_manifest = _apply_pattern_layer_section_mix(
            layer_render=layer_render,
            role=layer.role,
            sections=(layer.metadata or {}).get("sections"),
            sample_rate=int(request.sample_rate),
            tempo_bpm=float(request.tempo_bpm),
        )
        if section_mix_manifest:
            section_mix_manifests[layer_name] = section_mix_manifest
    _record_stage_timing("section_mix", stage_started_at)

    transition_sections = None
    for prepared in prepared_layers:
        candidate_sections = list(((prepared["layer"].metadata or {}).get("sections") or []))
        if candidate_sections:
            transition_sections = candidate_sections
            break
    stage_started_at = time.perf_counter()
    producer_transitions_manifest = _apply_pattern_producer_transitions(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("producer_transitions", stage_started_at)
    stage_started_at = time.perf_counter()
    motif_sample_glue_manifest = _apply_pattern_motif_sample_glue(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sample_rate=int(request.sample_rate),
    )
    _record_stage_timing("motif_sample_glue", stage_started_at)
    stage_started_at = time.perf_counter()
    topline_signature_manifest = _apply_pattern_topline_signature(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("topline_signature", stage_started_at)
    stage_started_at = time.perf_counter()
    tonal_memory_bus_manifest = _apply_pattern_tonal_memory_bus(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("tonal_memory_bus", stage_started_at)
    stage_started_at = time.perf_counter()
    ensemble_section_contrast_manifest = _apply_pattern_ensemble_section_contrast(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("ensemble_section_contrast", stage_started_at)
    stage_started_at = time.perf_counter()
    hook_memory_lift_manifest = _apply_pattern_hook_memory_lift(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("hook_memory_lift", stage_started_at)
    stage_started_at = time.perf_counter()
    drum_turnarounds_manifest = _apply_pattern_drum_turnarounds(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("drum_turnarounds", stage_started_at)
    stage_started_at = time.perf_counter()
    drum_section_evolution_manifest = _apply_pattern_drum_section_evolution(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("drum_section_evolution", stage_started_at)
    stage_started_at = time.perf_counter()
    drum_kit_character_manifest = _apply_pattern_drum_kit_character(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("drum_kit_character", stage_started_at)
    stage_started_at = time.perf_counter()
    sample_stack_groove_response_manifest = _apply_pattern_sample_stack_groove_response(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sample_rate=int(request.sample_rate),
    )
    _record_stage_timing("sample_stack_groove_response", stage_started_at)
    stage_started_at = time.perf_counter()
    sample_stack_section_dialogue_manifest = _apply_pattern_sample_stack_section_dialogue(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sections=transition_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("sample_stack_section_dialogue", stage_started_at)

    stage_started_at = time.perf_counter()
    kick_bass_sidechain_manifest = _apply_pattern_kick_bass_sidechain(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sample_rate=int(request.sample_rate),
    )
    _record_stage_timing("kick_bass_sidechain", stage_started_at)
    stage_started_at = time.perf_counter()
    drum_bus_glue_manifest = _apply_pattern_drum_bus_glue(
        layer_renders=layer_renders,
        rendered_layers=rendered_layers,
        sample_rate=int(request.sample_rate),
    )
    _record_stage_timing("drum_bus_glue", stage_started_at)
    stage_started_at = time.perf_counter()
    for layer_name, layer_render in layer_renders.items():
        master += layer_render
    _record_stage_timing("sum_layers", stage_started_at)

    macro_mix_sections = None
    for prepared in prepared_layers:
        candidate_sections = list(((prepared["layer"].metadata or {}).get("sections") or []))
        if candidate_sections:
            macro_mix_sections = candidate_sections
            break
    stage_started_at = time.perf_counter()
    macro_mix_manifest = _apply_pattern_macro_mix(
        master=master,
        sections=macro_mix_sections,
        sample_rate=int(request.sample_rate),
        tempo_bpm=float(request.tempo_bpm),
    )
    _record_stage_timing("macro_mix", stage_started_at)

    for rendered_layer in rendered_layers:
        layer_name = _clean_text(rendered_layer.get("name"))
        if layer_name and layer_name in section_mix_manifests:
            rendered_layer["section_mix_manifest"] = section_mix_manifests[layer_name]

    peak = float(np.max(np.abs(master))) if master.size else 0.0
    normalization_gain = 1.0
    if peak > 0.99:
        normalization_gain = 0.99 / peak
        master *= normalization_gain

    groove_summary: List[Dict[str, Any]] = []
    for rendered_layer in rendered_layers:
        triggers = list(rendered_layer.get("trigger_positions") or [])
        groove_triggers = [trigger for trigger in triggers if isinstance(trigger, dict) and trigger.get("groove")]
        transforms: List[str] = []
        feels: List[str] = []
        for trigger in groove_triggers:
            feel = _clean_text((trigger.get("groove") or {}).get("feel"))
            if feel and feel not in feels:
                feels.append(feel)
            transform = _clean_text(((trigger.get("section_variation") or {}).get("transform")))
            if transform and transform not in transforms:
                transforms.append(transform)
        if groove_triggers:
            groove_summary.append(
                {
                    "name": rendered_layer.get("name"),
                    "role": rendered_layer.get("role"),
                    "profile": "boombap_mpc_pocket_v1",
                    "trigger_count": len(triggers),
                    "groove_trigger_count": len(groove_triggers),
                    "feels": feels[:10],
                    "transforms": transforms[:10],
                }
            )

    output_path = os.path.join(
        run_dirs["beats_dir"],
        f"{_slugify(request.label)}_pattern_beat.wav",
    )
    stage_started_at = time.perf_counter()
    _write_audio_array_to_wav(master, output_path, sample_rate=request.sample_rate)
    _record_stage_timing("write_output_wav", stage_started_at)

    project_session = None
    if request.persist_to_project and request.project_id:
        project_session = add_mix_session(
            project_id=request.project_id,
            label=request.label,
            artifact_path=output_path,
            source_run_id=run_dirs["run_id"],
            target_peak_db=DEFAULT_TARGET_PEAK_DB,
            notes="Beat renderizado a partir de pads/padroes locais.",
            metadata={"session_type": "pattern_beat"},
        )

    manifest = {
        "request": request.to_dict(),
        "output_path": output_path,
        "rendered_layers": rendered_layers,
        "section_mix_manifests": section_mix_manifests,
        "producer_transitions_manifest": producer_transitions_manifest,
        "motif_sample_glue_manifest": motif_sample_glue_manifest,
        "topline_signature_manifest": topline_signature_manifest,
        "tonal_memory_bus_manifest": tonal_memory_bus_manifest,
        "ensemble_section_contrast_manifest": ensemble_section_contrast_manifest,
        "hook_memory_lift_manifest": hook_memory_lift_manifest,
        "drum_turnarounds_manifest": drum_turnarounds_manifest,
        "drum_section_evolution_manifest": drum_section_evolution_manifest,
        "drum_kit_character_manifest": drum_kit_character_manifest,
        "sample_stack_groove_response_manifest": sample_stack_groove_response_manifest,
        "sample_stack_section_dialogue_manifest": sample_stack_section_dialogue_manifest,
        "kick_bass_sidechain_manifest": kick_bass_sidechain_manifest,
        "drum_bus_glue_manifest": drum_bus_glue_manifest,
        "macro_mix_manifest": macro_mix_manifest,
        "render_timing": {
            "total_ms": round((time.perf_counter() - render_started_at) * 1000.0, 3),
            "stages_ms": stage_timings_ms,
            "layers_count": len(rendered_layers),
            "prepared_layers_count": len(prepared_layers),
        },
        "normalization_gain": round(float(normalization_gain), 6),
        "humanization": {
            "timing_jitter_ratio": round(float(timing_jitter_ratio), 4),
            "velocity_jitter_db": round(float(velocity_jitter_db), 4),
            "seed": int(jitter_seed),
            "groove_profile": "boombap_mpc_pocket_v1",
            "groove_summary": groove_summary,
        },
        "project_session": project_session,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "pattern_beat_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def render_blueprint_beat(request: BlueprintBeatRenderRequest) -> Dict[str, Any]:
    ensure_runtime_ready("beat_rendering")
    resolved_blueprint = _resolve_project_blueprint_payload(
        project_id=request.project_id,
        blueprint_version_id=request.blueprint_version_id,
    )
    blueprint_record = resolved_blueprint["blueprint_record"] or {}
    blueprint = dict(resolved_blueprint["blueprint"] or {})
    blueprint["style_tags"] = _normalize_instrumental_style_tags(
        blueprint.get("style_tags") or blueprint_record.get("style_tags") or []
    )
    tonal_anchor_key = _clean_text(request.tonal_anchor_key)
    tonal_anchor_scale = _clean_text(request.tonal_anchor_scale)
    tonal_anchor_confidence = request.tonal_anchor_confidence
    if tonal_anchor_key or tonal_anchor_scale or tonal_anchor_confidence is not None:
        blueprint["_render_overrides"] = {
            "tonal_anchor_key": tonal_anchor_key or None,
            "tonal_anchor_scale": tonal_anchor_scale or None,
            "tonal_anchor_confidence": tonal_anchor_confidence,
            "tonal_anchor_source": None,
        }

    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    resolved_label = (
        _clean_text(request.label)
        or _clean_text(blueprint.get("label"))
        or _clean_text(blueprint_record.get("label"))
        or "blueprint-beat"
    )
    minimum_bars = max(1, int(request.bars_override or blueprint.get("bars") or blueprint_record.get("bars") or 4))
    resolved_steps = max(
        1,
        int(request.steps_per_bar_override or blueprint.get("steps_per_bar") or blueprint_record.get("steps_per_bar") or 16),
    )
    resolved_bpm = float(blueprint.get("target_bpm") or blueprint_record.get("target_bpm") or 92.0)
    resolved_bars = _resolve_total_bars_for_duration(
        tempo_bpm=resolved_bpm,
        target_duration_seconds=request.target_duration_seconds,
        minimum_bars=minimum_bars,
    )
    resolved_swing = (
        round(float(request.swing), 3)
        if request.swing is not None
        else _infer_blueprint_swing(blueprint.get("style_tags") or blueprint_record.get("style_tags") or [], resolved_bpm)
    )
    resolved_seed = (
        int(request.seed)
        if request.seed is not None
        else (zlib.crc32(f"{request.project_id}:{resolved_label}:{resolved_bpm}:{resolved_bars}".encode("utf-8")) & 0xFFFFFFFF)
    )

    generated_samples_dir = os.path.join(run_dirs["beats_dir"], f"{_slugify(resolved_label)}_synthetic_kit")
    generated_layers = _build_pattern_layers_from_blueprint(
        blueprint=blueprint,
        generated_samples_dir=generated_samples_dir,
        sample_rate=int(request.sample_rate or DEFAULT_SAMPLE_RATE),
        total_bars=resolved_bars,
        steps_per_bar=resolved_steps,
        creativity=max(0.0, min(1.0, float(request.creativity))),
        fill_probability=max(0.0, min(1.0, float(request.fill_probability))),
        seed=resolved_seed,
        add_supporting_bass=bool(request.add_supporting_bass),
        add_melodic_motif=bool(request.add_melodic_motif),
        add_reference_texture=bool(request.add_reference_texture),
        add_reference_chops=bool(request.add_reference_chops),
    )

    pattern_manifest = render_pattern_beat(
        PatternBeatRequest(
            label=resolved_label,
            tempo_bpm=resolved_bpm,
            layers=generated_layers["layers"],
            bars=resolved_bars,
            steps_per_bar=resolved_steps,
            sample_rate=int(request.sample_rate or DEFAULT_SAMPLE_RATE),
            swing=resolved_swing,
            timing_jitter_ratio=max(0.0, min(0.45, float(request.timing_jitter_ratio))),
            velocity_jitter_db=max(0.0, min(9.0, float(request.velocity_jitter_db))),
            seed=resolved_seed,
            project_id=request.project_id,
            run_id=run_dirs["run_id"],
            persist_to_project=bool(request.persist_to_project),
        )
    )

    creative_mix_manifest = None
    render_output_path = pattern_manifest["output_path"]
    if request.apply_creative_mix:
        creative_mix_output_path = os.path.join(
            run_dirs["mixes_dir"],
            f"{_slugify(resolved_label)}_creative_mix.wav",
        )
        creative_mix_manifest = _render_creative_mix_bus(
            input_audio_path=pattern_manifest["output_path"],
            output_path=creative_mix_output_path,
            style_tags=blueprint.get("style_tags") or blueprint_record.get("style_tags") or [],
        )
        if request.persist_to_project:
            creative_mix_manifest["project_session"] = add_mix_session(
                project_id=request.project_id,
                label=f"{resolved_label}-creative-mix",
                artifact_path=creative_mix_output_path,
                source_run_id=run_dirs["run_id"],
                target_peak_db=DEFAULT_TARGET_PEAK_DB,
                notes="Creative mix bus aplicado ao autorender do blueprint.",
                metadata={"session_type": "blueprint_creative_mix"},
            )
        render_output_path = creative_mix_output_path

    master_manifest = None
    if request.create_master:
        master_manifest = master_audio(
            MasteringRequest(
                input_audio_path=render_output_path,
                label=f"{resolved_label}-master",
                target_lufs=float(request.master_target_lufs),
                true_peak_db=float(request.master_true_peak_db),
                project_id=request.project_id,
                run_id=run_dirs["run_id"],
                persist_to_project=bool(request.persist_to_project),
            )
        )

    manifest = {
        "request": request.to_dict(),
        "resolved_blueprint": {
            "version_id": blueprint_record.get("version_id"),
            "label": blueprint_record.get("label") or blueprint.get("label"),
            "artifact_path": resolved_blueprint.get("artifact_path"),
        },
        "resolved_render": {
            "label": resolved_label,
            "tempo_bpm": resolved_bpm,
            "bars": resolved_bars,
            "steps_per_bar": resolved_steps,
            "swing": resolved_swing,
            "sub_frequency_hz": generated_layers["sub_frequency_hz"],
            "target_duration_seconds": round(float(request.target_duration_seconds), 4)
            if request.target_duration_seconds is not None
            else None,
            "resolved_duration_seconds": round(float((240.0 / resolved_bpm) * resolved_bars), 4),
            "creativity": round(float(request.creativity), 4),
            "fill_probability": round(float(request.fill_probability), 4),
            "timing_jitter_ratio": round(float(request.timing_jitter_ratio), 4),
            "velocity_jitter_db": round(float(request.velocity_jitter_db), 4),
            "seed": int(resolved_seed),
            "tonal_anchor_key": tonal_anchor_key or None,
            "tonal_anchor_scale": tonal_anchor_scale or None,
            "tonal_anchor_confidence": round(float(tonal_anchor_confidence), 4)
            if tonal_anchor_confidence is not None
            else None,
        },
        "arrangement_sections": generated_layers["sections"],
        "pitch_context": generated_layers["pitch_context"],
        "synthesized_samples": generated_layers["synthesized_samples"],
        "pattern_manifest": pattern_manifest,
        "creative_mix_manifest": creative_mix_manifest,
        "render_output_path": render_output_path,
        "master_manifest": master_manifest,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "blueprint_render_manifest.json")
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    return manifest


def fit_vocal_to_beat(request: VocalFitRequest) -> Dict[str, Any]:
    ensure_runtime_ready("mix")
    instrumental_path = _resolve_existing_path(request.instrumental_audio_path)
    vocal_path = _resolve_existing_path(request.vocal_audio_path)
    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    ffmpeg_path = ensure_runtime_ready("mix")["ffmpeg_path"]

    label = _clean_text(request.label) or (
        f"{os.path.splitext(os.path.basename(vocal_path))[0]}-fit-{os.path.splitext(os.path.basename(instrumental_path))[0]}"
    )
    prepared_vocal_path = _clean_text(request.prepared_vocal_path) or os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_prepared_vocal.wav",
    )
    ducked_instrumental_path = _clean_text(request.ducked_instrumental_path) or os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_ducked_instrumental.wav",
    )
    output_path = _clean_text(request.output_path) or os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_vocal_fit_mix.wav",
    )

    instrumental_analysis = analyze_audio(
        AnalysisRequest(
            audio_path=instrumental_path,
            label=f"{label}-instrumental",
            project_id=request.project_id,
            run_id=run_dirs["run_id"],
            persist_to_project=False,
        )
    )
    vocal_analysis = analyze_audio(
        AnalysisRequest(
            audio_path=vocal_path,
            label=f"{label}-vocal",
            project_id=request.project_id,
            run_id=run_dirs["run_id"],
            persist_to_project=False,
        )
    )
    fit_plan = _build_vocal_fit_plan(instrumental_analysis, vocal_analysis, request)
    instrumental_duration_seconds = float(fit_plan.get("instrumental_duration_seconds") or _resolve_audio_duration_seconds(instrumental_path))
    vocal_duration_seconds = float(fit_plan.get("vocal_duration_seconds") or _resolve_audio_duration_seconds(vocal_path))
    adapted_instrumental_manifest = None
    vocal_aware_arrangement_manifest = None
    phrase_energy_manifest = None
    vocal_spectral_space_manifest = None
    vocal_stereo_space_manifest = None
    vocal_clarity_manifest = None
    vocal_phrase_level_manifest = None
    vocal_sibilance_manifest = None
    vocal_breath_control_manifest = None
    vocal_plosive_control_manifest = None
    vocal_harshness_control_manifest = None
    vocal_space_manifest = None
    instrumental_brightness_space_manifest = None
    instrumental_low_mid_space_manifest = None
    instrumental_presence_space_manifest = None
    instrumental_hotspot_micro_duck_manifest = None
    instrumental_transcript_line_support_manifest = None
    vocal_fit_glue_bus_manifest = None
    vocal_issue_hotspot_manifest = None
    phrase_blocks_manifest = None
    macrosections_manifest = None
    resolved_adaptation_profile = _normalize_instrumental_adaptation_profile(request.instrumental_adaptation_profile)
    vocal_render_profile = _apply_instrumental_adaptation_profile(
        _derive_vocal_fit_render_profile(vocal_analysis),
        resolved_adaptation_profile,
    )
    vocal_mix_profile = None

    if (
        bool(request.auto_extend_instrumental)
        and request.project_id
        and vocal_duration_seconds > instrumental_duration_seconds + 0.25
    ):
        adapted_instrumental_manifest = render_blueprint_beat(
            BlueprintBeatRenderRequest(
                project_id=request.project_id,
                label=f"{label}-instrumental-adapted",
                run_id=run_dirs["run_id"],
                target_duration_seconds=vocal_duration_seconds,
                creativity=float(vocal_render_profile["creativity"]),
                fill_probability=float(vocal_render_profile["fill_probability"]),
                timing_jitter_ratio=float(vocal_render_profile["timing_jitter_ratio"]),
                velocity_jitter_db=float(vocal_render_profile["velocity_jitter_db"]),
                tonal_anchor_key=_clean_text((fit_plan.get("vocal_key") or {}).get("key")) or None,
                tonal_anchor_scale=_clean_text((fit_plan.get("vocal_key") or {}).get("scale")) or None,
                tonal_anchor_confidence=(fit_plan.get("vocal_key") or {}).get("confidence"),
                add_supporting_bass=True,
                add_melodic_motif=bool(vocal_render_profile["add_melodic_motif"]),
                add_reference_texture=bool(vocal_render_profile["add_reference_texture"]),
                add_reference_chops=bool(vocal_render_profile["add_reference_chops"]),
                apply_creative_mix=True,
                create_master=False,
                persist_to_project=bool(request.persist_to_project),
            )
        )
        instrumental_path = _resolve_existing_path(
            _clean_text(adapted_instrumental_manifest.get("render_output_path"))
            or _clean_text((adapted_instrumental_manifest.get("pattern_manifest") or {}).get("output_path"))
            or instrumental_path
        )
        instrumental_analysis = analyze_audio(
            AnalysisRequest(
                audio_path=instrumental_path,
                label=f"{label}-instrumental-adapted-analysis",
                project_id=request.project_id,
                run_id=run_dirs["run_id"],
                persist_to_project=False,
            )
        )
        fit_plan = _build_vocal_fit_plan(instrumental_analysis, vocal_analysis, request)
        adapted_pitch_context = (adapted_instrumental_manifest.get("pitch_context") or {})
        if adapted_pitch_context:
            fit_plan = _apply_instrumental_tonal_override_to_fit_plan(
                fit_plan=fit_plan,
                pitch_context=adapted_pitch_context,
                request=request,
            )
        instrumental_duration_seconds = float(
            fit_plan.get("instrumental_duration_seconds") or _resolve_audio_duration_seconds(instrumental_path)
        )
        fit_plan.setdefault("recommendations", []).insert(
            0,
            "Instrumental original era menor que o vocal; o beat foi re-renderizado para cobrir a duracao do vocal.",
        )

    resolved_vocal_gain_db = (
        float(request.vocal_gain_db)
        if request.vocal_gain_db is not None
        else float(fit_plan.get("auto_vocal_gain_db") or 0.0)
    )
    resolved_instrumental_gain_db = (
        float(request.instrumental_gain_db)
        if request.instrumental_gain_db is not None
        else float(fit_plan.get("auto_instrumental_gain_db") or 0.0)
    )
    output_duration_mode = (_clean_text(request.output_duration_mode).lower() or "match_vocal")
    resolved_pitch_shift_semitones = (
        int(fit_plan.get("recommended_pitch_shift_semitones"))
        if fit_plan.get("should_pitch_align") and fit_plan.get("recommended_pitch_shift_semitones") is not None
        else 0
    )
    if bool(request.enable_vocal_transcription):
        vocal_transcription_manifest = _run_vocal_fit_transcription(
            vocal_path=vocal_path,
            label=label,
            project_id=request.project_id,
            run_id=run_dirs["run_id"],
        )
    else:
        vocal_transcription_manifest = {
            "transcription_status": "disabled",
            "transcription_attempts": [],
            "transcript": None,
            "transcript_path": None,
        }
    preliminary_vocal_transcript_line_manifest = _derive_vocal_transcript_line_events(
        transcript_manifest=vocal_transcription_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=None,
    )

    if adapted_instrumental_manifest and adapted_instrumental_manifest.get("pattern_manifest"):
        resolved_render = adapted_instrumental_manifest.get("resolved_render") or {}
        rearranged_output_path = os.path.join(
            run_dirs["mixes_dir"],
            f"{_slugify(label)}_vocal_aware_instrumental.wav",
        )
        vocal_arrangement = _build_vocal_aware_section_overrides(
            vocal_path=vocal_path,
            sample_rate=DEFAULT_SAMPLE_RATE,
            bars=int(resolved_render.get("bars") or 1),
            tempo_bpm=float(resolved_render.get("tempo_bpm") or fit_plan.get("instrumental_tempo_bpm") or 92.0),
            steps_per_bar=int(resolved_render.get("steps_per_bar") or 16),
            adaptation_profile=resolved_adaptation_profile,
            transcript_line_manifest=preliminary_vocal_transcript_line_manifest or {},
        )
        vocal_aware_arrangement_manifest = {
            **vocal_arrangement,
            **_render_vocal_aware_arrangement_mix(
                pattern_manifest=adapted_instrumental_manifest["pattern_manifest"],
                bar_overrides=vocal_arrangement["bar_overrides"],
                step_overrides=vocal_arrangement.get("step_overrides"),
                phrase_blocks=vocal_arrangement.get("phrase_blocks"),
                macrosections=vocal_arrangement.get("macrosections"),
                output_path=rearranged_output_path,
                sample_rate=DEFAULT_SAMPLE_RATE,
                bars=int(resolved_render.get("bars") or 1),
                steps_per_bar=int(resolved_render.get("steps_per_bar") or 16),
                tempo_bpm=float(resolved_render.get("tempo_bpm") or fit_plan.get("instrumental_tempo_bpm") or 92.0),
            ),
        }
        instrumental_path = rearranged_output_path
        phrase_blocks_manifest = vocal_arrangement.get("phrase_blocks") or {}
        macrosections_manifest = vocal_arrangement.get("macrosections") or {}

    if not phrase_blocks_manifest:
        estimated_bars = _resolve_total_bars_for_duration(
            tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
            target_duration_seconds=instrumental_duration_seconds,
            minimum_bars=1,
        )
        phrase_blocks_manifest = _detect_vocal_phrase_blocks(
            _compute_bar_activity_levels(
                vocal_path=vocal_path,
                sample_rate=DEFAULT_SAMPLE_RATE,
                bars=int(estimated_bars),
                tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
            )
        )
    if not macrosections_manifest:
        macrosections_manifest = _detect_vocal_macrosections(
            phrase_blocks=phrase_blocks_manifest or {},
            total_bars=int(
                _resolve_total_bars_for_duration(
                    tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
                    target_duration_seconds=instrumental_duration_seconds,
                    minimum_bars=1,
                )
            ),
            adaptation_profile=resolved_adaptation_profile,
            transcript_line_manifest=preliminary_vocal_transcript_line_manifest or {},
        )
    vocal_mix_profile = _derive_vocal_fit_mix_profile(
        vocal_analysis=vocal_analysis,
        instrumental_analysis=instrumental_analysis,
        phrase_blocks=phrase_blocks_manifest or {},
    )
    vocal_transcript_line_manifest = _derive_vocal_transcript_line_events(
        transcript_manifest=vocal_transcription_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=macrosections_manifest or {},
    )

    phrase_energy_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_phrase_energy_instrumental.wav",
    )
    phrase_energy_manifest = _apply_phrase_energy_automation(
        instrumental_path=instrumental_path,
        output_path=phrase_energy_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        macrosections=macrosections_manifest or {},
        transcript_line_manifest=vocal_transcript_line_manifest or {},
    )
    instrumental_path = phrase_energy_output_path

    macrosection_groove_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_macrosection_groove_instrumental.wav",
    )
    macrosection_groove_manifest = _apply_macrosection_groove_automation(
        instrumental_path=instrumental_path,
        output_path=macrosection_groove_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        macrosections=macrosections_manifest or {},
        transcript_line_manifest=vocal_transcript_line_manifest or {},
    )
    instrumental_path = macrosection_groove_output_path

    macrosection_harmonic_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_macrosection_harmonic_instrumental.wav",
    )
    macrosection_harmonic_manifest = _apply_macrosection_harmonic_support(
        instrumental_path=instrumental_path,
        output_path=macrosection_harmonic_output_path,
        vocal_key=fit_plan.get("vocal_key") or {},
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        macrosections=macrosections_manifest or {},
        transcript_line_manifest=vocal_transcript_line_manifest or {},
    )
    instrumental_path = macrosection_harmonic_output_path

    transcript_line_support_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_transcript_line_support_instrumental.wav",
    )
    instrumental_transcript_line_support_manifest = _apply_transcript_line_support(
        instrumental_path=instrumental_path,
        output_path=transcript_line_support_output_path,
        transcript_line_manifest=vocal_transcript_line_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    instrumental_path = transcript_line_support_output_path

    tempo_ratio = fit_plan.get("tempo_ratio")
    vocal_processors = ["aformat=channel_layouts=stereo"]
    if fit_plan.get("should_time_align") and tempo_ratio is not None:
        vocal_processors.append(f"atempo={float(tempo_ratio):.6f}")
    if resolved_pitch_shift_semitones:
        pitch_ratio = 2.0 ** (float(resolved_pitch_shift_semitones) / 12.0)
        vocal_processors.append(
            f"rubberband=pitch={pitch_ratio:.6f}:formant=preserved:pitchq=quality:transients=smooth"
        )
    vocal_processors.extend(
        [
            "highpass=f=90",
            "lowpass=f=14000",
            "acompressor=threshold=0.10:ratio=2.4:attack=8:release=110:makeup=1.2",
            f"volume={resolved_vocal_gain_db:.2f}dB",
        ]
    )
    vocal_prepare_result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            vocal_path,
            "-af",
            ",".join(vocal_processors),
            "-c:a",
            "pcm_s16le",
            prepared_vocal_path,
        ]
    )
    if vocal_prepare_result.returncode != 0:
        _raise_command_error("ffmpeg vocal prepare", vocal_prepare_result)

    vocal_clarity_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_contextual_vocal.wav",
    )
    vocal_clarity_manifest = _apply_contextual_vocal_clarity(
        vocal_path=prepared_vocal_path,
        output_path=vocal_clarity_output_path,
        vocal_analysis=vocal_analysis,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=macrosections_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    vocal_phrase_level_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_phrase_vocal.wav",
    )
    vocal_phrase_level_manifest = _apply_vocal_phrase_level_automation(
        vocal_path=vocal_clarity_output_path,
        output_path=vocal_phrase_level_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=macrosections_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    vocal_issue_hotspot_manifest = _detect_vocal_issue_hotspots(
        vocal_path=vocal_phrase_level_output_path,
        sample_rate=DEFAULT_SAMPLE_RATE,
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        phrase_blocks=phrase_blocks_manifest or {},
    )
    vocal_sibilance_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_sibilance_vocal.wav",
    )
    vocal_sibilance_manifest = _apply_contextual_vocal_sibilance_control(
        vocal_path=vocal_phrase_level_output_path,
        output_path=vocal_sibilance_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        macrosections=macrosections_manifest or {},
    )
    vocal_breath_control_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_breath_vocal.wav",
    )
    vocal_breath_control_manifest = _apply_vocal_breath_control(
        vocal_path=vocal_sibilance_output_path,
        output_path=vocal_breath_control_output_path,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=macrosections_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    vocal_plosive_control_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_plosive_vocal.wav",
    )
    vocal_plosive_control_manifest = _apply_vocal_plosive_control(
        vocal_path=vocal_breath_control_output_path,
        output_path=vocal_plosive_control_output_path,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=macrosections_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    vocal_harshness_control_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_harshness_vocal.wav",
    )
    vocal_harshness_control_manifest = _apply_vocal_harshness_control(
        vocal_path=vocal_plosive_control_output_path,
        output_path=vocal_harshness_control_output_path,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        macrosections=macrosections_manifest or {},
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    prepared_vocal_path = vocal_harshness_control_output_path

    ducking_strength = max(0.05, min(0.95, float(request.ducking_strength)))
    ducking_manifest = _apply_vocal_activity_ducking(
        instrumental_path=instrumental_path,
        vocal_path=prepared_vocal_path,
        output_path=ducked_instrumental_path,
        sample_rate=DEFAULT_SAMPLE_RATE,
        ducking_strength=ducking_strength,
        target_duration_seconds=vocal_duration_seconds if output_duration_mode == "match_vocal" else instrumental_duration_seconds,
        instrumental_gain_db=resolved_instrumental_gain_db,
        macrosections=macrosections_manifest or {},
    )
    spectral_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_spectral_space_instrumental.wav",
    )
    vocal_spectral_space_manifest = _apply_vocal_spectral_space(
        instrumental_path=ducked_instrumental_path,
        output_path=spectral_space_output_path,
        vocal_analysis=vocal_analysis,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    stereo_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_stereo_space_instrumental.wav",
    )
    vocal_stereo_space_manifest = _apply_vocal_stereo_space(
        instrumental_path=spectral_space_output_path,
        output_path=stereo_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    brightness_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_brightness_space_instrumental.wav",
    )
    instrumental_brightness_space_manifest = _apply_contextual_instrumental_brightness_space(
        instrumental_path=stereo_space_output_path,
        output_path=brightness_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        hotspot_manifest=vocal_issue_hotspot_manifest,
    )
    low_mid_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_low_mid_space_instrumental.wav",
    )
    instrumental_low_mid_space_manifest = _apply_contextual_instrumental_low_mid_space(
        instrumental_path=brightness_space_output_path,
        output_path=low_mid_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        hotspot_manifest=vocal_issue_hotspot_manifest,
    )
    presence_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_presence_space_instrumental.wav",
    )
    instrumental_presence_space_manifest = _apply_contextual_instrumental_presence_space(
        instrumental_path=low_mid_space_output_path,
        output_path=presence_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        hotspot_manifest=vocal_issue_hotspot_manifest,
    )
    instrumental_hotspot_micro_duck_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_hotspot_duck_instrumental.wav",
    )
    instrumental_hotspot_micro_duck_manifest = _apply_instrumental_hotspot_micro_duck(
        instrumental_path=presence_space_output_path,
        output_path=instrumental_hotspot_micro_duck_output_path,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    instrumental_transition_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_transition_space_instrumental.wav",
    )
    instrumental_transition_space_manifest = _apply_transition_space(
        audio_path=instrumental_hotspot_micro_duck_output_path,
        output_path=instrumental_transition_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        macrosections=macrosections_manifest or {},
        target_mode="instrumental",
        transcript_line_manifest=vocal_transcript_line_manifest or {},
    )
    ducked_instrumental_path = instrumental_transition_space_output_path

    vocal_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_space_vocal.wav",
    )
    vocal_space_manifest = _apply_contextual_vocal_space(
        vocal_path=prepared_vocal_path,
        output_path=vocal_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        mix_profile=vocal_mix_profile,
        macrosections=macrosections_manifest or {},
    )
    vocal_transition_space_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_transition_space_vocal.wav",
    )
    vocal_transition_space_manifest = _apply_transition_space(
        audio_path=vocal_space_output_path,
        output_path=vocal_transition_space_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        macrosections=macrosections_manifest or {},
        target_mode="vocal",
        transcript_line_manifest=vocal_transcript_line_manifest or {},
    )
    vocal_phrase_contrast_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_phrase_contrast_vocal.wav",
    )
    vocal_phrase_contrast_manifest = _apply_vocal_phrase_contrast(
        vocal_path=vocal_transition_space_output_path,
        output_path=vocal_phrase_contrast_output_path,
        phrase_blocks=phrase_blocks_manifest or {},
        tempo_bpm=float(fit_plan.get("instrumental_tempo_bpm") or 92.0),
        sample_rate=DEFAULT_SAMPLE_RATE,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        macrosections=macrosections_manifest or {},
        transcript_line_manifest=vocal_transcript_line_manifest or {},
    )
    prepared_vocal_path = vocal_phrase_contrast_output_path

    limiter_linear = max(0.1, min(1.0, 10 ** (float(request.target_peak_db) / 20.0)))
    if output_duration_mode == "match_vocal":
        target_duration_seconds = max(0.1, float(vocal_duration_seconds))
        mix_filter = (
            f"[0:a]aformat=channel_layouts=stereo,atrim=end={target_duration_seconds:.4f},asetpts=N/SR/TB[ducked];"
            f"[1:a]aformat=channel_layouts=stereo,atrim=end={target_duration_seconds:.4f},asetpts=N/SR/TB[voc];"
            "[ducked][voc]amix=inputs=2:duration=longest:normalize=0:weights='1 0.84',"
            "equalizer=f=2800:t=q:w=1.0:g=0.18,"
            "equalizer=f=4800:t=q:w=1.2:g=0.12,"
            f"alimiter=limit={limiter_linear:.4f}[out]"
        )
    else:
        mix_filter = (
            f"[0:a]aformat=channel_layouts=stereo,volume=0dB[ducked];"
            "[1:a]aformat=channel_layouts=stereo[voc];"
            "[ducked][voc]amix=inputs=2:duration=first:normalize=0:weights='1 0.84',"
            "equalizer=f=2800:t=q:w=1.0:g=0.18,"
            "equalizer=f=4800:t=q:w=1.2:g=0.12,"
            f"alimiter=limit={limiter_linear:.4f}[out]"
        )
    pre_glue_output_path = os.path.join(
        run_dirs["mixes_dir"],
        f"{_slugify(label)}_vocal_fit_mix_pre_glue.wav",
    )
    mix_result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            ducked_instrumental_path,
            "-i",
            prepared_vocal_path,
            "-filter_complex",
            mix_filter,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            pre_glue_output_path,
        ]
    )
    if mix_result.returncode != 0:
        _raise_command_error("ffmpeg vocal fit mix", mix_result)
    vocal_fit_glue_bus_manifest = _render_vocal_fit_glue_bus(
        input_audio_path=pre_glue_output_path,
        vocal_path=prepared_vocal_path,
        output_path=output_path,
        vocal_analysis=vocal_analysis,
        sample_rate=DEFAULT_SAMPLE_RATE,
        mix_profile=vocal_mix_profile,
        macrosections=macrosections_manifest or {},
    )

    project_mix_session = None
    if request.persist_to_project and request.project_id:
        project_mix_session = add_mix_session(
            project_id=request.project_id,
            label=label,
            artifact_path=output_path,
            source_run_id=run_dirs["run_id"],
            target_peak_db=request.target_peak_db,
            notes="Mix adaptado para encaixe de vocal do usuario sobre instrumental.",
            metadata={"session_type": "vocal_fit_mix", "fit_plan": fit_plan},
        )

    fit_context = None
    if request.persist_to_project and request.project_id:
        fit_context = add_derived_context(
            project_id=request.project_id,
            context_type="vocal_fit_plan",
            label=label,
            summary=" | ".join(fit_plan.get("recommendations") or []),
            relative_path=None,
            source_reference_ids=[],
            metadata=fit_plan,
        )

    master_manifest = None
    if request.create_master:
        master_manifest = master_audio(
            MasteringRequest(
                input_audio_path=output_path,
                label=f"{label}-master",
                target_lufs=float(request.master_target_lufs),
                true_peak_db=float(request.master_true_peak_db),
                project_id=request.project_id,
                run_id=run_dirs["run_id"],
                persist_to_project=bool(request.persist_to_project),
            )
        )

    manifest = {
        "request": request.to_dict(),
        "instrumental_analysis": instrumental_analysis,
        "vocal_analysis": vocal_analysis,
        "fit_plan": fit_plan,
        "adapted_instrumental_manifest": adapted_instrumental_manifest,
        "vocal_aware_arrangement_manifest": vocal_aware_arrangement_manifest,
        "vocal_render_profile": vocal_render_profile,
        "resolved_instrumental_adaptation_profile": resolved_adaptation_profile,
        "output_duration_mode": output_duration_mode,
        "resolved_pitch_shift_semitones": int(resolved_pitch_shift_semitones),
        "resolved_vocal_gain_db": round(float(resolved_vocal_gain_db), 4),
        "resolved_instrumental_gain_db": round(float(resolved_instrumental_gain_db), 4),
        "prepared_vocal_path": prepared_vocal_path,
        "ducked_instrumental_path": ducked_instrumental_path,
        "output_path": output_path,
        "vocal_prepare_filter": ",".join(vocal_processors),
        "phrase_blocks_manifest": phrase_blocks_manifest,
        "macrosections_manifest": macrosections_manifest,
        "vocal_transcription_manifest": vocal_transcription_manifest,
        "vocal_transcript_line_manifest": vocal_transcript_line_manifest,
        "vocal_mix_profile": vocal_mix_profile,
        "phrase_energy_manifest": phrase_energy_manifest,
        "macrosection_groove_manifest": macrosection_groove_manifest,
        "macrosection_harmonic_manifest": macrosection_harmonic_manifest,
        "instrumental_transcript_line_support_manifest": instrumental_transcript_line_support_manifest,
        "ducking_manifest": ducking_manifest,
        "vocal_phrase_level_manifest": vocal_phrase_level_manifest,
        "vocal_issue_hotspot_manifest": vocal_issue_hotspot_manifest,
        "vocal_clarity_manifest": vocal_clarity_manifest,
        "vocal_sibilance_manifest": vocal_sibilance_manifest,
        "vocal_breath_control_manifest": vocal_breath_control_manifest,
        "vocal_plosive_control_manifest": vocal_plosive_control_manifest,
        "vocal_harshness_control_manifest": vocal_harshness_control_manifest,
        "vocal_space_manifest": vocal_space_manifest,
        "vocal_spectral_space_manifest": vocal_spectral_space_manifest,
        "vocal_stereo_space_manifest": vocal_stereo_space_manifest,
        "vocal_phrase_contrast_manifest": vocal_phrase_contrast_manifest,
        "instrumental_brightness_space_manifest": instrumental_brightness_space_manifest,
        "instrumental_low_mid_space_manifest": instrumental_low_mid_space_manifest,
        "instrumental_presence_space_manifest": instrumental_presence_space_manifest,
        "instrumental_hotspot_micro_duck_manifest": instrumental_hotspot_micro_duck_manifest,
        "instrumental_transition_space_manifest": instrumental_transition_space_manifest,
        "vocal_fit_glue_bus_manifest": vocal_fit_glue_bus_manifest,
        "vocal_transition_space_manifest": vocal_transition_space_manifest,
        "mix_filter_graph": mix_filter,
        "project_mix_session": project_mix_session,
        "fit_context": fit_context,
        "master_manifest": master_manifest,
        "output_duration_seconds": _resolve_audio_duration_seconds(output_path),
    }
    manifest["ui_summary"] = _build_vocal_fit_ui_summary(
        fit_plan=fit_plan,
        vocal_mix_profile=vocal_mix_profile,
        phrase_blocks_manifest=phrase_blocks_manifest,
        macrosections_manifest=macrosections_manifest,
        hotspot_manifest=vocal_issue_hotspot_manifest,
        output_duration_seconds=manifest["output_duration_seconds"],
        output_path=output_path,
        prepared_vocal_path=prepared_vocal_path,
        ducked_instrumental_path=ducked_instrumental_path,
        adaptation_profile=resolved_adaptation_profile,
        transcription_status=_clean_text((vocal_transcription_manifest or {}).get("transcription_status")) or "disabled",
        resolved_pitch_shift_semitones=int(resolved_pitch_shift_semitones),
    )
    manifest_path = os.path.join(run_dirs["manifests_dir"], "vocal_fit_manifest.json")
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    return manifest


def mix_audio(request: MixRequest) -> Dict[str, Any]:
    status = ensure_runtime_ready("mix")
    ffmpeg_path = status["ffmpeg_path"]
    if not request.tracks:
        raise ValueError("Informe pelo menos uma track para mixagem.")

    active_tracks = [track for track in request.tracks if not track.mute]
    solo_tracks = [track for track in active_tracks if track.solo]
    if solo_tracks:
        active_tracks = solo_tracks
    if not active_tracks:
        raise ValueError("Nenhuma track ativa restou para mixagem.")

    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    input_args: List[str] = []
    filter_chains: List[str] = []
    amix_inputs: List[str] = []

    for index, track in enumerate(active_tracks):
        source_path = _resolve_existing_path(track.audio_path)
        _copy_into_directory(source_path, run_dirs["inputs_dir"])
        input_args.extend(["-i", source_path])

        processors: List[str] = ["aformat=channel_layouts=stereo"]
        trim_start = float(track.trim_start_ms or 0) / 1000.0
        trim_end = None if track.trim_end_ms is None else float(track.trim_end_ms) / 1000.0
        if trim_start or trim_end is not None:
            atrim = f"atrim=start={trim_start:.3f}"
            if trim_end is not None:
                atrim += f":end={trim_end:.3f}"
            processors.append(atrim)
            processors.append("asetpts=N/SR/TB")
        if track.highpass_hz:
            processors.append(f"highpass=f={float(track.highpass_hz):.2f}")
        if track.lowpass_hz:
            processors.append(f"lowpass=f={float(track.lowpass_hz):.2f}")
        if float(track.gain_db):
            processors.append(f"volume={float(track.gain_db):.2f}dB")
        if float(track.pan):
            left_gain, right_gain = _pan_gains(float(track.pan))
            processors.append(
                f"pan=stereo|c0={left_gain:.4f}*c0|c1={right_gain:.4f}*c1"
            )
        if int(track.start_ms or 0) > 0:
            delay_ms = int(track.start_ms)
            processors.append(f"adelay={delay_ms}|{delay_ms}")

        filter_chains.append(f"[{index}:a]{','.join(processors)}[a{index}]")
        amix_inputs.append(f"[a{index}]")

    limiter_linear = max(0.1, min(1.0, 10 ** (float(request.target_peak_db) / 20.0)))
    filter_chains.append(
        f"{''.join(amix_inputs)}amix=inputs={len(active_tracks)}:normalize=0:dropout_transition=0,"
        f"alimiter=limit={limiter_linear:.4f}[out]"
    )
    filter_graph = ";".join(filter_chains)

    output_path = _clean_text(request.output_path)
    if not output_path:
        output_path = os.path.join(
            run_dirs["mixes_dir"],
            f"{_slugify(request.label)}_mix.wav",
        )

    result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            *input_args,
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    if result.returncode != 0:
        _raise_command_error("ffmpeg mix", result)

    project_session = None
    if request.persist_to_project and request.project_id:
        project_session = add_mix_session(
            project_id=request.project_id,
            label=request.label,
            artifact_path=output_path,
            source_run_id=run_dirs["run_id"],
            target_peak_db=request.target_peak_db,
            notes="Mix renderizado pelo pipeline local de audio.",
        )

    manifest = {
        "request": request.to_dict(),
        "output_path": output_path,
        "filter_graph": filter_graph,
        "project_session": project_session,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "mix_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def master_audio(request: MasteringRequest) -> Dict[str, Any]:
    status = ensure_runtime_ready("master")
    ffmpeg_path = status["ffmpeg_path"]
    input_path = _resolve_existing_path(request.input_audio_path)
    run_dirs = _resolve_run_dir(project_id=request.project_id, run_id=request.run_id)
    _copy_into_directory(input_path, run_dirs["inputs_dir"])

    output_path = _clean_text(request.output_path)
    label = _clean_text(request.label) or os.path.splitext(os.path.basename(input_path))[0]
    if not output_path:
        output_path = os.path.join(
            run_dirs["masters_dir"],
            f"{_slugify(label)}_master.wav",
        )

    first_pass_filter = (
        f"loudnorm=I={float(request.target_lufs):.1f}:"
        f"LRA={float(request.loudness_range):.1f}:"
        f"TP={float(request.true_peak_db):.1f}:print_format=json"
    )
    analysis_result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-i",
            input_path,
            "-af",
            first_pass_filter,
            "-f",
            "null",
            _ffmpeg_null_output(),
        ]
    )
    if analysis_result.returncode != 0:
        _raise_command_error("ffmpeg loudnorm analysis", analysis_result)
    loudnorm_stats = _parse_loudnorm_json(analysis_result.stderr or "")

    second_pass_filter = (
        "highpass=f=25,"
        "lowpass=f=18000,"
        "acompressor=threshold=0.125:ratio=2.5:attack=15:release=180:makeup=1,"
        f"loudnorm=I={float(request.target_lufs):.1f}:"
        f"LRA={float(request.loudness_range):.1f}:"
        f"TP={float(request.true_peak_db):.1f}:"
        f"measured_I={_loudnorm_value(loudnorm_stats, 'input_i', request.target_lufs)}:"
        f"measured_LRA={_loudnorm_value(loudnorm_stats, 'input_lra', request.loudness_range)}:"
        f"measured_TP={_loudnorm_value(loudnorm_stats, 'input_tp', request.true_peak_db)}:"
        f"measured_thresh={_loudnorm_value(loudnorm_stats, 'input_thresh', -34.0)}:"
        f"offset={_loudnorm_value(loudnorm_stats, 'target_offset', 0.0)}:"
        "linear=true:print_format=summary,"
        f"alimiter=limit={max(0.1, min(1.0, 10 ** (float(request.true_peak_db) / 20.0))):.4f}"
    )
    render_result = _run_command(
        [
            ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            input_path,
            "-af",
            second_pass_filter,
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    if render_result.returncode != 0:
        _raise_command_error("ffmpeg master", render_result)

    project_session = None
    if request.persist_to_project and request.project_id:
        project_session = add_master_session(
            project_id=request.project_id,
            label=label,
            artifact_path=output_path,
            source_run_id=run_dirs["run_id"],
            target_lufs=request.target_lufs,
            true_peak_db=request.true_peak_db,
            notes="Master renderizado em duas etapas com loudnorm.",
            metadata={"loudnorm_stats": loudnorm_stats or {}},
        )

    manifest = {
        "request": request.to_dict(),
        "input_path": input_path,
        "output_path": output_path,
        "analysis_stderr": analysis_result.stderr,
        "loudnorm_stats": loudnorm_stats,
        "render_stderr": render_result.stderr,
        "project_session": project_session,
    }
    manifest_path = os.path.join(run_dirs["manifests_dir"], "master_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest
