"""Runtime diagnostics for local image generation."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata as importlib_metadata
from typing import Any, Dict, Optional

from tools.img_gen.image_models import DEFAULT_SD15_BASE_ALIAS, resolve_model_artifact


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
        }
    except Exception as exc:
        return {
            "ok": False,
            "module": None,
            "error": str(exc),
        }


def _major_version(version_text: Optional[str]) -> Optional[int]:
    text = str(version_text or "").strip()
    if not text:
        return None
    try:
        return int(text.split(".", 1)[0])
    except ValueError:
        return None


def collect_runtime_status(
    model_alias: str = DEFAULT_SD15_BASE_ALIAS,
    model_category: str = "checkpoints",
) -> Dict[str, Any]:
    versions = {
        "torch": _package_version("torch"),
        "diffusers": _package_version("diffusers"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "safetensors": _package_version("safetensors"),
        "huggingface_hub": _package_version("huggingface_hub"),
        "pillow": _package_version("Pillow"),
    }

    torch_import = _import_module("torch")
    diffusers_import = _import_module("diffusers")
    transformers_import = _import_module("transformers")

    cuda_available = False
    cuda_device_name = None
    torch_runtime_version = None
    torch_build_cuda = None
    torch_cpu_only = False
    if torch_import["ok"]:
        torch_module = torch_import["module"]
        torch_runtime_version = getattr(torch_module, "__version__", None)
        torch_build_cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
        torch_cpu_only = "+cpu" in str(torch_runtime_version or "")
        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            cuda_device_name = torch_module.cuda.get_device_name(0)

    model_artifact = resolve_model_artifact(model_category, alias=model_alias)

    issues = []
    warnings = []

    if not versions["diffusers"]:
        issues.append("A biblioteca 'diffusers' nao esta instalada na venv ativa.")
    elif not diffusers_import["ok"]:
        issues.append(
            "A biblioteca 'diffusers' esta instalada, mas falhou ao importar: "
            f"{diffusers_import['error']}"
        )

    if not versions["transformers"]:
        issues.append("A biblioteca 'transformers' nao esta instalada na venv ativa.")
    elif not transformers_import["ok"]:
        issues.append(
            "A biblioteca 'transformers' esta instalada, mas falhou ao importar: "
            f"{transformers_import['error']}"
        )

    if not versions["torch"]:
        issues.append("A biblioteca 'torch' nao esta instalada na venv ativa.")
    elif not torch_import["ok"]:
        issues.append(
            "A biblioteca 'torch' esta instalada, mas falhou ao importar: "
            f"{torch_import['error']}"
        )
    elif torch_cpu_only:
        issues.append(
            "O PyTorch instalado na venv ativa e CPU-only "
            f"({torch_runtime_version}). Para usar a RTX 2060, instale um build CUDA."
        )

    transformers_major = _major_version(versions["transformers"])
    if transformers_major and transformers_major >= 5:
        issues.append(
            "A versao instalada de transformers e 5.x. Para este runtime de imagem com "
            "torch 2.1.2 e diffusers 0.32.1, use uma versao 4.x de transformers."
        )

    if not model_artifact:
        issues.append(
            f"Nenhum checkpoint local registrado para o alias '{model_alias}'."
        )
    elif model_artifact.get("status") != "ready":
        issues.append(
            f"O checkpoint '{model_alias}' nao esta pronto para uso. "
            f"Status atual: {model_artifact.get('status')}."
        )

    if versions["diffusers"] == "0.37.0" and versions["torch"] == "2.1.2":
        warnings.append(
            "O ambiente atual usa diffusers 0.37.0 com torch 2.1.2; esse par falhou "
            "no import local e precisa de ajuste de versao antes da geracao."
        )

    if not cuda_available:
        warnings.append(
            "CUDA nao esta disponivel no runtime atual; a geracao local em CPU sera muito lenta."
        )

    ready_for_local_generation = not issues

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": versions,
        "torch_runtime_version": torch_runtime_version,
        "torch_build_cuda": torch_build_cuda,
        "torch_cpu_only": torch_cpu_only,
        "torch_import": {
            "ok": torch_import["ok"],
            "error": torch_import["error"],
        },
        "diffusers_import": {
            "ok": diffusers_import["ok"],
            "error": diffusers_import["error"],
        },
        "transformers_import": {
            "ok": transformers_import["ok"],
            "error": transformers_import["error"],
        },
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "model_alias": model_alias,
        "model_category": model_category,
        "model_artifact": model_artifact,
        "issues": issues,
        "warnings": warnings,
        "ready_for_local_generation": ready_for_local_generation,
    }


def format_runtime_status(status: Dict[str, Any]) -> str:
    lines = [
        f"Python: {status['python_version']}",
        f"Platform: {status['platform']}",
        f"Torch: {status.get('torch_runtime_version') or status['versions'].get('torch') or 'nao instalado'}",
        f"Diffusers: {status['versions'].get('diffusers') or 'nao instalado'}",
        f"Transformers: {status['versions'].get('transformers') or 'nao instalado'}",
        f"Accelerate: {status['versions'].get('accelerate') or 'nao instalado'}",
        f"CUDA disponivel: {status['cuda_available']}",
    ]
    if status.get("torch_build_cuda"):
        lines.append(f"Torch CUDA build: {status['torch_build_cuda']}")
    if status.get("cuda_device_name"):
        lines.append(f"GPU: {status['cuda_device_name']}")

    artifact = status.get("model_artifact") or {}
    if artifact:
        lines.append(
            f"Checkpoint: {artifact.get('alias')} ({artifact.get('status')}) -> {artifact.get('local_path')}"
        )
    else:
        lines.append("Checkpoint: nao registrado")

    if status.get("issues"):
        lines.append("Issues:")
        lines.extend(f"- {item}" for item in status["issues"])
    if status.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in status["warnings"])

    return "\n".join(lines)


def ensure_runtime_ready(
    model_alias: str = DEFAULT_SD15_BASE_ALIAS,
    model_category: str = "checkpoints",
) -> Dict[str, Any]:
    status = collect_runtime_status(
        model_alias=model_alias,
        model_category=model_category,
    )
    if not status["ready_for_local_generation"]:
        raise RuntimeError(format_runtime_status(status))
    return status
