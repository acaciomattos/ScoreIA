"""Runtime diagnostics for local audio generation and processing."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional

from tools.audio_gen.model_paths import DEFAULT_AUDIO_GEN_MODELS_DIR, build_model_paths


def _package_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _import_module(module_name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "ok": True,
            "module": module,
            "error": None,
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {
            "ok": False,
            "module": None,
            "error": str(exc),
            "version": None,
        }


def _module_spec_status(module_name: str, distribution_name: Optional[str] = None) -> Dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    version = _package_version(distribution_name or module_name)
    return {
        "ok": bool(spec),
        "module": None,
        "error": None if spec else f"O modulo `{module_name}` nao esta disponivel no ambiente atual.",
        "version": version,
    }


def _run_command(args: List[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "returncode": None,
        }
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": None,
        "returncode": result.returncode,
    }


def _escape_ffmpeg_filter_value(value: str) -> str:
    escaped = str(value).replace("\\", "/")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(",", "\\,")
    return escaped


def _normalize_ffmpeg_filter_path(path: str) -> str:
    normalized = str(path or "").strip()
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


def _validate_whisper_model(
    ffmpeg_path: Optional[str],
    model_path: Optional[str],
) -> Dict[str, Any]:
    if not ffmpeg_path or not model_path:
        return {"ok": None, "error": None}
    temp_path = None
    temp_handle = None
    try:
        temp_handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_path = temp_handle.name
        temp_handle.close()
        filter_model_path = _normalize_ffmpeg_filter_path(model_path)
        filter_transcript_path = _normalize_ffmpeg_filter_path(temp_path)
        whisper_filter = (
            f"whisper=model={_escape_ffmpeg_filter_value(filter_model_path)}"
            f":language=auto"
            f":destination={_escape_ffmpeg_filter_value(filter_transcript_path)}"
            f":format=json"
            f":use_gpu=false"
        )
        result = _run_command(
            [
                ffmpeg_path,
                "-hide_banner",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.05",
                "-af",
                whisper_filter,
                "-f",
                "null",
                "NUL" if os.name == "nt" else "/dev/null",
            ]
        )
        if result["ok"]:
            return {"ok": True, "error": None}
        error_text = (result.get("stderr") or result.get("stdout") or result.get("error") or "").strip()
        return {"ok": False, "error": error_text[:1200] or "Falha ao validar o modelo do whisper."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:1200]}
    finally:
        if temp_handle is not None:
            try:
                temp_handle.close()
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _discover_whisper_models(models_root: str = DEFAULT_AUDIO_GEN_MODELS_DIR) -> List[str]:
    paths = build_model_paths(models_root=models_root)
    if not os.path.exists(paths.whisper_dir):
        return []

    discovered: List[str] = []
    for entry in sorted(os.listdir(paths.whisper_dir)):
        lower_name = entry.lower()
        if not lower_name.endswith(".bin"):
            continue
        discovered.append(os.path.join(paths.whisper_dir, entry))
    return discovered


def _score_whisper_model_candidate(path: str) -> tuple[int, str]:
    name = os.path.basename(path).lower()
    if name == "ggml-base.bin":
        return (0, name)
    if name == "ggml-small.bin":
        return (1, name)
    if name == "ggml-medium.bin":
        return (2, name)
    if name == "ggml-large-v3.bin":
        return (3, name)
    if "base" in name:
        return (4, name)
    if "small" in name:
        return (5, name)
    if "medium" in name:
        return (6, name)
    if "large" in name:
        return (7, name)
    return (20, name)


def _resolve_whisper_model_path(
    whisper_model_path: Optional[str] = None,
    models_root: str = DEFAULT_AUDIO_GEN_MODELS_DIR,
) -> Optional[str]:
    explicit = str(whisper_model_path or "").strip()
    if explicit:
        candidate = os.path.abspath(explicit)
        return candidate if os.path.exists(candidate) else None

    discovered = sorted(_discover_whisper_models(models_root=models_root), key=_score_whisper_model_candidate)
    return os.path.abspath(discovered[0]) if discovered else None


def collect_runtime_status(
    whisper_model_path: Optional[str] = None,
    models_root: str = DEFAULT_AUDIO_GEN_MODELS_DIR,
) -> Dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    demucs_cli_path = shutil.which("demucs")
    mpv_path = shutil.which("mpv")
    ffplay_path = shutil.which("ffplay")

    ffmpeg_version = None
    ffprobe_version = None
    whisper_filter_available = False
    whisper_filter_error = None

    if ffmpeg_path:
        ffmpeg_version_result = _run_command([ffmpeg_path, "-version"])
        if ffmpeg_version_result["stdout"]:
            ffmpeg_version = ffmpeg_version_result["stdout"].splitlines()[0].strip()
        whisper_filter_result = _run_command([ffmpeg_path, "-hide_banner", "-h", "filter=whisper"])
        whisper_filter_available = whisper_filter_result["ok"] and (
            "Filter whisper" in whisper_filter_result["stdout"]
            or "Filter whisper" in whisper_filter_result["stderr"]
        )
        whisper_filter_error = whisper_filter_result["error"]

    if ffprobe_path:
        ffprobe_version_result = _run_command([ffprobe_path, "-version"])
        if ffprobe_version_result["stdout"]:
            ffprobe_version = ffprobe_version_result["stdout"].splitlines()[0].strip()

    numpy_import = _import_module("numpy")
    scipy_import = _import_module("scipy")
    torch_import = _import_module("torch")
    torchaudio_import = _import_module("torchaudio")
    demucs_import = _import_module("demucs")
    soundfile_import = _import_module("soundfile")
    librosa_status = _module_spec_status("librosa")
    basic_pitch_status = _module_spec_status("basic_pitch", "basic-pitch")
    pyloudnorm_status = _module_spec_status("pyloudnorm")
    matchering_status = _module_spec_status("matchering")
    soundcard_status = _module_spec_status("soundcard")
    yt_dlp_status = _module_spec_status("yt_dlp", "yt-dlp")

    discovered_whisper_models = _discover_whisper_models(models_root=models_root)
    resolved_whisper_model_path = _resolve_whisper_model_path(
        whisper_model_path=whisper_model_path,
        models_root=models_root,
    )
    whisper_model_validation = _validate_whisper_model(ffmpeg_path, resolved_whisper_model_path)
    whisper_model_compatible = whisper_model_validation.get("ok")
    whisper_model_validation_error = whisper_model_validation.get("error")

    capabilities = {
        "analysis": bool(ffmpeg_path and ffprobe_path and numpy_import["ok"] and scipy_import["ok"]),
        "analysis_enriched": bool(
            ffmpeg_path and ffprobe_path and numpy_import["ok"] and scipy_import["ok"] and librosa_status["ok"]
        ),
        "beat_rendering": bool(ffmpeg_path and numpy_import["ok"]),
        "mix": bool(ffmpeg_path),
        "master": bool(ffmpeg_path),
        "transcription": bool(
            ffmpeg_path
            and whisper_filter_available
            and resolved_whisper_model_path
            and whisper_model_compatible is not False
        ),
        "separation": bool(demucs_cli_path or demucs_import["ok"]),
        "melodic_analysis": bool(basic_pitch_status["ok"]),
        "loopback_capture": bool(platform.system() == "Windows" and soundcard_status["ok"]),
        "youtube_background_capture": bool(
            platform.system() == "Windows"
            and soundcard_status["ok"]
            and yt_dlp_status["ok"]
            and (mpv_path or ffplay_path)
        ),
    }

    issues: List[str] = []
    warnings: List[str] = []

    if not ffmpeg_path:
        issues.append("`ffmpeg` nao foi encontrado no PATH. Mixagem, masterizacao e transcricao ficam indisponiveis.")
    if not ffprobe_path:
        issues.append("`ffprobe` nao foi encontrado no PATH. A analise musical fica indisponivel.")
    if not numpy_import["ok"]:
        issues.append(f"O modulo `numpy` nao esta disponivel: {numpy_import['error']}")
    if not scipy_import["ok"]:
        issues.append(f"O modulo `scipy` nao esta disponivel: {scipy_import['error']}")
    if ffmpeg_path and not whisper_filter_available:
        warnings.append("O `ffmpeg` atual nao expoe o filtro `whisper`; a transcricao local nao vai funcionar nesta build.")
    if whisper_filter_available and not resolved_whisper_model_path:
        warnings.append(
            "Nenhum modelo `whisper.cpp` foi encontrado em `models/audio_gen/whisper/`; "
            "a transcricao precisa de um arquivo `.bin` local."
        )
    if whisper_filter_available and resolved_whisper_model_path and whisper_model_compatible is False:
        warnings.append(
            "O modelo `whisper.cpp` encontrado nao carregou corretamente no filtro `whisper` do ffmpeg; "
            "a transcricao local fica indisponivel ate trocar esse `.bin` por uma variante compativel."
        )
    if not demucs_cli_path and not demucs_import["ok"]:
        warnings.append(
            "Nem o CLI `demucs` nem o modulo Python `demucs` estao disponiveis; "
            "a separacao de stems fica preparada, mas ainda nao executavel nesta venv."
        )
    if torch_import["ok"] and "+cpu" in str(torch_import["version"] or ""):
        warnings.append("O PyTorch ativo e CPU-only; tarefas pesadas de audio podem ficar lentas.")
    if not torchaudio_import["ok"]:
        warnings.append("`torchaudio` nao esta ativo na venv atual. O baseline segue funcional porque usa `ffmpeg`.")
    if not soundfile_import["ok"]:
        warnings.append("`soundfile` nao esta ativo na venv atual. O baseline usa WAV/ffmpeg para contornar isso.")
    if not librosa_status["ok"]:
        warnings.append("`librosa` nao esta ativo na venv atual. A analise musical segue funcional, mas menos rica.")
    if not basic_pitch_status["ok"]:
        warnings.append("`basic-pitch` nao esta ativo na venv atual. A extracao melodica/MIDI fica indisponivel.")
    if not pyloudnorm_status["ok"]:
        warnings.append("`pyloudnorm` nao esta ativo na venv atual. A validacao de loudness fica limitada ao ffmpeg.")
    if not matchering_status["ok"]:
        warnings.append("`matchering` nao esta ativo na venv atual. O mastering por referencia fica indisponivel.")
    if platform.system() == "Windows" and not soundcard_status["ok"]:
        warnings.append(
            "`soundcard` nao esta ativo na venv atual. A captura WASAPI loopback continua prevista, "
            "mas precisa desse pacote opcional para gravar a saida do sistema."
        )
    if platform.system() == "Windows" and not yt_dlp_status["ok"]:
        warnings.append(
            "`yt-dlp` nao esta ativo na venv atual. O modo YouTube em background fica indisponivel."
        )
    if platform.system() == "Windows" and not mpv_path:
        warnings.append(
            "O executavel `mpv` nao esta no PATH. O caminho recomendado de reproducao isolada em device virtual "
            "fica indisponivel; so resta fallback assistido ou `ffplay` no device padrao."
        )

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_path": ffprobe_path,
        "ffprobe_version": ffprobe_version,
        "whisper_filter_available": whisper_filter_available,
        "whisper_filter_error": whisper_filter_error,
        "demucs_cli_path": demucs_cli_path,
        "mpv_path": mpv_path,
        "ffplay_path": ffplay_path,
        "models_root": models_root,
        "discovered_whisper_models": discovered_whisper_models,
        "resolved_whisper_model_path": resolved_whisper_model_path,
        "whisper_model_compatible": whisper_model_compatible,
        "whisper_model_validation_error": whisper_model_validation_error,
        "modules": {
            "numpy": {
                "ok": numpy_import["ok"],
                "version": numpy_import["version"] or _package_version("numpy"),
                "error": numpy_import["error"],
            },
            "scipy": {
                "ok": scipy_import["ok"],
                "version": scipy_import["version"] or _package_version("scipy"),
                "error": scipy_import["error"],
            },
            "torch": {
                "ok": torch_import["ok"],
                "version": torch_import["version"] or _package_version("torch"),
                "error": torch_import["error"],
            },
            "torchaudio": {
                "ok": torchaudio_import["ok"],
                "version": torchaudio_import["version"] or _package_version("torchaudio"),
                "error": torchaudio_import["error"],
            },
            "demucs": {
                "ok": demucs_import["ok"],
                "version": demucs_import["version"] or _package_version("demucs"),
                "error": demucs_import["error"],
            },
            "soundfile": {
                "ok": soundfile_import["ok"],
                "version": soundfile_import["version"] or _package_version("soundfile"),
                "error": soundfile_import["error"],
            },
            "librosa": {
                "ok": librosa_status["ok"],
                "version": librosa_status["version"],
                "error": librosa_status["error"],
            },
            "basic_pitch": {
                "ok": basic_pitch_status["ok"],
                "version": basic_pitch_status["version"],
                "error": basic_pitch_status["error"],
            },
            "pyloudnorm": {
                "ok": pyloudnorm_status["ok"],
                "version": pyloudnorm_status["version"],
                "error": pyloudnorm_status["error"],
            },
            "matchering": {
                "ok": matchering_status["ok"],
                "version": matchering_status["version"],
                "error": matchering_status["error"],
            },
            "soundcard": {
                "ok": soundcard_status["ok"],
                "version": soundcard_status["version"],
                "error": soundcard_status["error"],
            },
            "yt_dlp": {
                "ok": yt_dlp_status["ok"],
                "version": yt_dlp_status["version"],
                "error": yt_dlp_status["error"],
            },
        },
        "capabilities": capabilities,
        "issues": issues,
        "warnings": warnings,
        "ready_for_audio_baseline": bool(
            capabilities["analysis"] and capabilities["beat_rendering"] and capabilities["mix"] and capabilities["master"]
        ),
    }


def format_runtime_status(status: Dict[str, Any]) -> str:
    lines = [
        f"Python: {status['python_version']}",
        f"Platform: {status['platform']}",
        f"ffmpeg: {status.get('ffmpeg_version') or 'nao encontrado'}",
        f"ffprobe: {status.get('ffprobe_version') or 'nao encontrado'}",
        f"Whisper filter: {status.get('whisper_filter_available')}",
        f"Whisper model: {status.get('resolved_whisper_model_path') or 'nao encontrado'}",
        f"Whisper model compatible: {status.get('whisper_model_compatible')}",
        f"Demucs CLI: {status.get('demucs_cli_path') or 'nao encontrado'}",
        "Capabilities:",
    ]
    for capability_name in (
        "analysis",
        "analysis_enriched",
        "melodic_analysis",
        "beat_rendering",
        "mix",
        "master",
        "transcription",
        "separation",
        "loopback_capture",
        "youtube_background_capture",
    ):
        lines.append(f"- {capability_name}: {status['capabilities'].get(capability_name)}")

    if status.get("issues"):
        lines.append("Issues:")
        lines.extend(f"- {item}" for item in status["issues"])
    if status.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in status["warnings"])

    return "\n".join(lines)


def ensure_runtime_ready(
    required: str = "analysis",
    whisper_model_path: Optional[str] = None,
    models_root: str = DEFAULT_AUDIO_GEN_MODELS_DIR,
) -> Dict[str, Any]:
    status = collect_runtime_status(
        whisper_model_path=whisper_model_path,
        models_root=models_root,
    )

    capability_ready = status["capabilities"].get(required)
    if not capability_ready:
        raise RuntimeError(
            f"O runtime de audio nao esta pronto para '{required}'.\n{format_runtime_status(status)}"
        )
    return status
