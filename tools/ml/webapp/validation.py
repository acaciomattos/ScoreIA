from typing import Any, Dict, Optional

from tools.ml.data_pipeline import DataPipeline


def calcular_limites_stability_max_folds(
    n_months: int,
    oot_months: int,
) -> Optional[Dict[str, int]]:
    n_months = int(n_months or 0)
    oot_months = int(oot_months or 0)
    n_months_in_time = n_months - oot_months

    if n_months_in_time < 2:
        return None

    max_folds = int(n_months_in_time)
    return {
        "min": 2,
        "max": max_folds,
        "default": min(5, max_folds),
        "n_months_in_time": n_months_in_time,
    }


def validar_temporal_oot_web(
    caminho: str,
    formato: str,
    temporal_feature: Optional[str],
    oot_months_raw: str,
    stability_max_folds_raw: str = "",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "is_valid_for_training": True,
        "temporal_validation": None,
        "oot_months": 0,
        "erro_temporal": None,
        "alerta_temporal": None,
        "show_oot_field": bool(temporal_feature),
        "stability_fold_bounds": None,
        "stability_max_folds_resolved": None,
    }

    if not temporal_feature:
        return result

    try:
        checker = DataPipeline(caminho)
        checker.carregar(formato)
        temporal_validation = checker.validar_temporal_feature(temporal_feature)
    except Exception as exc:
        result["is_valid_for_training"] = False
        result["erro_temporal"] = str(exc)
        return result

    result["temporal_validation"] = temporal_validation

    if not temporal_validation.get("valid", False):
        result["is_valid_for_training"] = False
        result["erro_temporal"] = temporal_validation.get("error")
        return result

    n_meses = int(temporal_validation.get("n_months", 0))

    if n_meses < 2:
        result["is_valid_for_training"] = False
        result["erro_temporal"] = (
            f"A feature temporal '{temporal_feature}' possui apenas {n_meses} mes(es) distinto(s). "
            "Sao necessarios ao menos 2 meses para separar OOT."
        )
        return result

    if not oot_months_raw:
        result["is_valid_for_training"] = False
        result["alerta_temporal"] = (
            "Foi observado que o conjunto de dados inserido possui "
            f"{n_meses} meses diferentes. Quantos meses devem ser reservados "
            "como 'out-of-time' para analise de estabilidade?"
        )
        return result

    try:
        oot_months = int(oot_months_raw)
    except ValueError:
        result["is_valid_for_training"] = False
        result["erro_temporal"] = "O parametro 'oot_months' deve ser um numero inteiro."
        return result

    if oot_months < 1 or oot_months >= n_meses:
        result["is_valid_for_training"] = False
        result["erro_temporal"] = (
            f"Valor invalido para OOT: {oot_months}. "
            f"Use um valor entre 1 e {n_meses - 1}."
        )
        return result

    result["oot_months"] = oot_months
    result["stability_fold_bounds"] = calcular_limites_stability_max_folds(n_meses, oot_months)

    if stability_max_folds_raw:
        try:
            stability_max_folds = int(stability_max_folds_raw)
        except ValueError:
            result["is_valid_for_training"] = False
            result["erro_temporal"] = "O parametro 'stability_max_folds' deve ser um numero inteiro."
            return result

        bounds = result["stability_fold_bounds"]
        if bounds and not (bounds["min"] <= stability_max_folds <= bounds["max"]):
            result["is_valid_for_training"] = False
            result["erro_temporal"] = (
                f"Valor invalido para stability_max_folds: {stability_max_folds}. "
                f"Use um valor entre {bounds['min']} e {bounds['max']}."
            )
            return result

        result["stability_max_folds_resolved"] = stability_max_folds
    else:
        bounds = result["stability_fold_bounds"]
        if bounds:
            result["stability_max_folds_resolved"] = bounds["default"]

    return result
