import os
from datetime import datetime
from typing import Dict, Optional

DEFAULT_ML_ARTIFACTS_DIR = os.path.join("artifacts", "ml")


def criar_run_context(run_timestamp: Optional[str] = None) -> Dict[str, str]:
    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    return {
        "timestamp": timestamp,
        "run_id": f"run_{timestamp}",
    }


def resolver_caminhos_artefatos(
    run_context: Dict[str, str],
    artifacts_dir: str = DEFAULT_ML_ARTIFACTS_DIR,
) -> Dict[str, str]:
    run_dir = os.path.join(artifacts_dir, run_context["run_id"])

    return {
        "artifacts_dir": artifacts_dir,
        "run_dir": run_dir,
        "run_model_dir": os.path.join(run_dir, "model"),
        "run_preprocess_dir": os.path.join(run_dir, "preprocess"),
        "run_inference_dir": os.path.join(run_dir, "inference"),
        "run_metadata_dir": os.path.join(run_dir, "metadata"),
        "run_reports_dir": os.path.join(run_dir, "reports"),
        "run_scored_dir": os.path.join(run_dir, "scored_data"),
    }


def resolver_caminhos_artefatos_estagio(
    caminhos_root: Dict[str, str],
    stage_name: str,
) -> Dict[str, str]:
    run_dir = os.path.join(caminhos_root["run_dir"], stage_name)
    return {
        "artifacts_dir": caminhos_root["artifacts_dir"],
        "run_dir": run_dir,
        "run_model_dir": os.path.join(run_dir, "model"),
        "run_preprocess_dir": os.path.join(run_dir, "preprocess"),
        "run_inference_dir": os.path.join(run_dir, "inference"),
        "run_metadata_dir": os.path.join(run_dir, "metadata"),
        "run_reports_dir": os.path.join(run_dir, "reports"),
        "run_scored_dir": os.path.join(run_dir, "scored_data"),
    }


def garantir_diretorios_artefatos(caminhos: Dict[str, str]) -> None:
    keys = (
        "run_dir",
        "run_model_dir",
        "run_preprocess_dir",
        "run_inference_dir",
        "run_metadata_dir",
        "run_reports_dir",
        "run_scored_dir",
    )

    for key in keys:
        os.makedirs(caminhos[key], exist_ok=True)


def nome_artefato(prefixo: str, run_context: Dict[str, str], extensao: str) -> str:
    ext = extensao.lstrip(".")
    return f"{prefixo}_{run_context['timestamp']}.{ext}"
