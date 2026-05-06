"""YouTube background playback helpers for isolated loopback capture."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _optional_import_yt_dlp():
    try:
        from yt_dlp import YoutubeDL  # type: ignore[import-not-found]

        return YoutubeDL
    except Exception:
        return None


def collect_youtube_background_status() -> Dict[str, Any]:
    yt_dlp_cls = _optional_import_yt_dlp()
    mpv_path = shutil.which("mpv")
    ffplay_path = shutil.which("ffplay")
    return {
        "yt_dlp_available": bool(yt_dlp_cls),
        "mpv_path": mpv_path,
        "ffplay_path": ffplay_path,
        "capabilities": {
            "youtube_background_mpv": bool(yt_dlp_cls and mpv_path),
            "youtube_background_ffplay": bool(yt_dlp_cls and ffplay_path),
        },
    }


def ensure_youtube_background_ready(backend: str = "mpv") -> Dict[str, Any]:
    status = collect_youtube_background_status()
    normalized_backend = _clean_text(backend).lower() or "mpv"
    if not status["yt_dlp_available"]:
        raise RuntimeError(
            "O modo YouTube background precisa do pacote opcional `yt-dlp`. "
            "Instale na `.venv310` com `pip install yt-dlp`."
        )
    if normalized_backend == "mpv" and not status["mpv_path"]:
        raise RuntimeError(
            "O backend recomendado para YouTube em background precisa do executavel `mpv` no PATH."
        )
    if normalized_backend == "ffplay" and not status["ffplay_path"]:
        raise RuntimeError(
            "O backend `ffplay` nao foi encontrado no PATH."
        )
    if normalized_backend not in {"mpv", "ffplay"}:
        raise ValueError("backend deve ser `mpv` ou `ffplay`.")
    return status


def resolve_youtube_background_stream(
    source_url: str,
    format_selector: str = "bestaudio/best",
) -> Dict[str, Any]:
    YoutubeDL = _optional_import_yt_dlp()
    if YoutubeDL is None:
        raise RuntimeError(
            "A resolucao de stream do YouTube precisa do pacote opcional `yt-dlp`. "
            "Instale na `.venv310` com `pip install yt-dlp`."
        )

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": format_selector,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source_url, download=False)

    if info is None:
        raise RuntimeError("Nao foi possivel resolver o stream do YouTube.")

    if "entries" in info and info["entries"]:
        info = info["entries"][0]

    direct_url = _clean_text(info.get("url"))
    if not direct_url:
        raise RuntimeError("O yt-dlp nao retornou uma URL de stream reproduzivel.")

    http_headers = dict(info.get("http_headers") or {})
    return {
        "title": _clean_text(info.get("title")),
        "duration_seconds": info.get("duration"),
        "direct_url": direct_url,
        "webpage_url": _clean_text(info.get("webpage_url")) or _clean_text(source_url),
        "extractor": _clean_text(info.get("extractor")),
        "format_id": _clean_text(info.get("format_id")),
        "acodec": _clean_text(info.get("acodec")),
        "ext": _clean_text(info.get("ext")),
        "http_headers": http_headers,
    }


def _build_mpv_command(
    mpv_path: str,
    stream_info: Dict[str, Any],
    playback_device_name: str = "",
    duration_seconds: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    command = [
        mpv_path,
        "--no-video",
        "--force-window=no",
        "--no-terminal",
        "--really-quiet",
    ]
    user_agent = _clean_text((stream_info.get("http_headers") or {}).get("User-Agent"))
    if user_agent:
        command.append(f"--user-agent={user_agent}")
    if _clean_text(playback_device_name):
        command.append(f"--audio-device={_clean_text(playback_device_name)}")
    if duration_seconds is not None and float(duration_seconds) > 0:
        command.append(f"--length={float(duration_seconds):.3f}")
    if extra_args:
        command.extend([str(item) for item in extra_args if _clean_text(item)])
    command.append(stream_info["direct_url"])
    return command


def _build_ffplay_command(
    ffplay_path: str,
    stream_info: Dict[str, Any],
    duration_seconds: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    command = [
        ffplay_path,
        "-nodisp",
        "-autoexit",
        "-vn",
    ]
    user_agent = _clean_text((stream_info.get("http_headers") or {}).get("User-Agent"))
    if user_agent:
        command.extend(["-user_agent", user_agent])
    if duration_seconds is not None and float(duration_seconds) > 0:
        command.extend(["-t", f"{float(duration_seconds):.3f}"])
    if extra_args:
        command.extend([str(item) for item in extra_args if _clean_text(item)])
    command.append(stream_info["direct_url"])
    return command


def launch_youtube_background_player(
    source_url: str,
    backend: str = "mpv",
    playback_device_name: str = "",
    duration_seconds: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    status = ensure_youtube_background_ready(backend=backend)
    stream_info = resolve_youtube_background_stream(source_url)
    normalized_backend = _clean_text(backend).lower() or "mpv"

    if normalized_backend == "mpv":
        command = _build_mpv_command(
            status["mpv_path"],
            stream_info,
            playback_device_name=playback_device_name,
            duration_seconds=duration_seconds,
            extra_args=extra_args,
        )
    else:
        command = _build_ffplay_command(
            status["ffplay_path"],
            stream_info,
            duration_seconds=duration_seconds,
            extra_args=extra_args,
        )

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return {
        "backend": normalized_backend,
        "command": command,
        "stream_info": stream_info,
        "process": process,
        "pid": process.pid,
    }


def stop_background_player(process: Optional[subprocess.Popen[Any]], timeout_seconds: float = 5.0) -> Dict[str, Any]:
    if process is None:
        return {"stopped": False, "returncode": None}
    if process.poll() is not None:
        return {"stopped": True, "returncode": process.returncode}

    process.terminate()
    try:
        process.wait(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=float(timeout_seconds))
    return {"stopped": True, "returncode": process.returncode}
