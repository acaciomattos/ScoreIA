import json
import os
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, abort, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from tools.audio_gen import (
    AnalysisRequest,
    BlueprintBeatRenderRequest,
    ExternalReferenceMetadataImportRequest,
    LoopbackCaptureRequest,
    VocalFitRequest,
    add_reference_asset,
    add_reference_link,
    analyze_audio,
    capture_reference_from_loopback,
    collect_runtime_status,
    collect_youtube_background_status,
    create_project,
    create_reference_beat_blueprint,
    fit_vocal_to_beat,
    import_external_reference_metadata,
    list_available_audio_routing_devices,
    list_available_loopback_devices,
    list_projects,
    load_project_bundle,
    render_blueprint_beat,
    set_active_beat_blueprint,
    set_active_working_set,
)
from tools.audio_gen.audio_pipeline import BeatBlueprintRequest
from tools.audio_gen.project_paths import (
    DEFAULT_AUDIO_GEN_ARTIFACTS_DIR,
    DEFAULT_AUDIO_GEN_PROJECTS_DIR,
    build_project_artifact_paths,
    build_project_paths,
)


app = Flask(__name__)

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROJECT_TYPES = [
    ("music_production", "Producao musical"),
    ("reference_study", "Estudo de referencia"),
    ("vocal_fit", "Vocal fit"),
    ("sample_pack", "Sample pack"),
]
REFERENCE_ROLES = [
    ("reference", "Referencia"),
    ("sample_source", "Source sample"),
    ("texture_source", "Texture source"),
    ("vocal", "Vocal"),
    ("instrumental", "Instrumental"),
]
CAPTURE_SOURCE_MODES = [
    ("manual", "Manual / browser"),
    ("youtube_background", "YouTube background"),
]
REFERENCE_AUDIO_MODES = [
    ("instrumental", "Instrumental"),
    ("song_remove_vocals", "Musica com remocao de voz"),
]
INSTRUMENTAL_ADAPTATION_PROFILES = [
    ("auto", "Auto"),
    ("dry", "Dry"),
    ("spaced", "Spaced"),
    ("dense", "Dense"),
    ("hook_support", "Hook support"),
]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
TEXT_EXTENSIONS = {".json", ".txt", ".md"}


def _projects_root() -> str:
    return os.path.abspath(
        app.config.get("AUDIO_PROJECTS_ROOT")
        or os.path.join(APP_ROOT, DEFAULT_AUDIO_GEN_PROJECTS_DIR)
    )


def _artifacts_root() -> str:
    return os.path.abspath(
        app.config.get("AUDIO_ARTIFACTS_ROOT")
        or os.path.join(APP_ROOT, DEFAULT_AUDIO_GEN_ARTIFACTS_DIR)
    )


def _status_label(ok: Any) -> str:
    if ok:
        return "ok"
    return "missing"


def _runtime_status_for_view() -> Dict[str, Any]:
    status = collect_runtime_status(
        models_root=os.path.abspath(os.path.join(APP_ROOT, "models", "audio_gen")),
    )
    modules = status.get("modules", {})
    torch_status = modules.get("torch", {})
    demucs_status = modules.get("demucs", {})

    status["ffmpeg"] = {
        "status": _status_label(status.get("ffmpeg_path")),
        "path": status.get("ffmpeg_path"),
        "version": status.get("ffmpeg_version"),
    }
    status["ffprobe"] = {
        "status": _status_label(status.get("ffprobe_path")),
        "path": status.get("ffprobe_path"),
        "version": status.get("ffprobe_version"),
    }
    status["torch"] = {
        "status": _status_label(torch_status.get("ok")),
        "version": torch_status.get("version"),
        "error": torch_status.get("error"),
    }
    status["demucs"] = {
        "status": _status_label(status.get("demucs_cli_path") or demucs_status.get("ok")),
        "path": status.get("demucs_cli_path"),
        "version": demucs_status.get("version"),
        "error": demucs_status.get("error"),
    }
    return status


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_optional_identifier(value: Any) -> Optional[str]:
    text = _clean_text(value)
    if not text or text.lower() in {"none", "null", "undefined"}:
        return None
    return text


def _parse_csv(value: Any) -> List[str]:
    text = _clean_text(value)
    if not text:
        return []
    return [chunk.strip() for chunk in text.split(",") if chunk and chunk.strip()]


def _checkbox_enabled(field_name: str, default: bool = False) -> bool:
    if field_name not in request.form:
        return default
    return request.form.get(field_name) == "on"


def _parse_int(value: Any, field_name: str, default: Optional[int] = None, minimum: Optional[int] = None) -> int:
    text = _clean_text(value)
    if not text:
        if default is None:
            raise ValueError(f"O campo '{field_name}' deve ser informado.")
        parsed = default
    else:
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError(f"O campo '{field_name}' deve ser inteiro.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"O campo '{field_name}' deve ser maior ou igual a {minimum}.")
    return parsed


def _parse_float(value: Any, field_name: str, default: Optional[float] = None, minimum: Optional[float] = None) -> float:
    text = _clean_text(value)
    if not text:
        if default is None:
            raise ValueError(f"O campo '{field_name}' deve ser informado.")
        parsed = default
    else:
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"O campo '{field_name}' deve ser numerico.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"O campo '{field_name}' deve ser maior ou igual a {minimum}.")
    return parsed


def _parse_optional_float(value: Any) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    return float(text)


def _parse_optional_int(value: Any) -> Optional[int]:
    text = _clean_text(value)
    if not text:
        return None
    return int(text)


def _parse_optional_json(value: Any, field_name: str) -> Dict[str, Any]:
    text = _clean_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"O campo '{field_name}' deve conter JSON valido.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"O campo '{field_name}' deve conter um objeto JSON.")
    return parsed


def _resolve_path(path: str) -> str:
    candidate = _clean_text(path)
    if not candidate:
        raise ValueError("Caminho vazio.")
    if os.path.isabs(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(APP_ROOT, candidate))


def _is_safe_under_root(target_path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(target_path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _safe_root(storage: str) -> Optional[str]:
    normalized = _clean_text(storage).lower()
    if normalized == "projects":
        return _projects_root()
    if normalized == "artifacts":
        return _artifacts_root()
    return None


def _file_url(path: str) -> Optional[str]:
    if not path:
        return None
    try:
        absolute_path = _resolve_path(path)
    except ValueError:
        return None
    for storage, root in (("projects", _projects_root()), ("artifacts", _artifacts_root())):
        if _is_safe_under_root(absolute_path, root) and os.path.exists(absolute_path):
            relative_path = os.path.relpath(absolute_path, root).replace(os.sep, "/")
            return url_for("serve_storage_file", storage=storage, relative_path=relative_path)
    return None


def _copy_uploaded_file(file_storage: Any, target_dir: str, prefix: str) -> str:
    filename = secure_filename(getattr(file_storage, "filename", "") or "")
    if not filename:
        raise ValueError("Nenhum arquivo foi enviado.")
    os.makedirs(target_dir, exist_ok=True)
    stem, ext = os.path.splitext(filename)
    safe_name = f"{prefix}_{uuid.uuid4().hex[:8]}_{stem or 'arquivo'}{ext.lower()}"
    target_path = os.path.join(target_dir, safe_name)
    file_storage.save(target_path)
    return target_path


def _project_bundle(project_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not project_id:
        return None
    return load_project_bundle(project_id, projects_root=_projects_root())


def _list_project_audio_candidates(project_id: Optional[str], limit: int = 32) -> List[Dict[str, Any]]:
    if not project_id:
        return []
    paths = build_project_paths(project_id, projects_root=_projects_root())
    artifact_paths = build_project_artifact_paths(project_id, artifacts_root=_artifacts_root())
    candidates: List[Dict[str, Any]] = []

    def _scan(root: str, group: str, label_root: str) -> None:
        if not os.path.exists(root):
            return
        for current_root, _, files in os.walk(root):
            for filename in files:
                extension = os.path.splitext(filename)[1].lower()
                if extension not in AUDIO_EXTENSIONS:
                    continue
                absolute_path = os.path.join(current_root, filename)
                try:
                    modified_at = os.path.getmtime(absolute_path)
                except OSError:
                    modified_at = 0.0
                label = os.path.relpath(absolute_path, label_root).replace(os.sep, "/")
                candidates.append(
                    {
                        "group": group,
                        "label": label,
                        "path": absolute_path,
                        "url": _file_url(absolute_path),
                        "modified_at": modified_at,
                    }
                )

    _scan(paths.references_audio_dir, "project", paths.project_dir)
    _scan(artifact_paths.project_artifacts_dir, "artifacts", artifact_paths.project_artifacts_dir)
    candidates.sort(key=lambda item: (item.get("modified_at") or 0.0, item.get("label") or ""), reverse=True)
    return candidates[:limit]


def _bundle_counts(bundle: Optional[Dict[str, Any]]) -> Dict[str, int]:
    manifests = (bundle or {}).get("manifests") or {}
    return {
        "references": len((manifests.get("references") or {}).get("items", [])),
        "contexts": len((manifests.get("derived_context") or {}).get("items", [])),
        "blueprints": len((manifests.get("beat_blueprints") or {}).get("items", [])),
        "captures": len((manifests.get("capture_sessions") or {}).get("items", [])),
        "working_sets": len((manifests.get("working_sets") or {}).get("items", [])),
    }


def _collect_result_paths(payload: Any, limit: int = 14) -> List[Dict[str, Any]]:
    collected: List[str] = []
    seen = set()

    def _walk(value: Any) -> None:
        if len(collected) >= limit:
            return
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                _walk(child)
            return
        if not isinstance(value, str):
            return
        text = _clean_text(value)
        if not text:
            return
        extension = os.path.splitext(text)[1].lower()
        if extension not in AUDIO_EXTENSIONS.union(TEXT_EXTENSIONS):
            return
        try:
            absolute_path = _resolve_path(text)
        except ValueError:
            return
        if absolute_path in seen or not os.path.exists(absolute_path):
            return
        seen.add(absolute_path)
        collected.append(absolute_path)

    _walk(payload)
    return [
        {
            "path": item,
            "name": os.path.basename(item),
            "is_audio": os.path.splitext(item)[1].lower() in AUDIO_EXTENSIONS,
            "url": _file_url(item),
        }
        for item in collected
    ]


def _result_pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.get("/files/<storage>/<path:relative_path>")
def serve_storage_file(storage: str, relative_path: str):
    root = _safe_root(storage)
    if root is None:
        abort(404)
    normalized_relative_path = _clean_text(relative_path).replace("/", os.sep)
    target_path = os.path.abspath(os.path.join(root, normalized_relative_path))
    if not _is_safe_under_root(target_path, root) or not os.path.exists(target_path):
        abort(404)
    return send_file(target_path)


@app.route("/", methods=["GET", "POST"])
def index():
    success_message = None
    error_message = None
    action_result = None
    action_result_paths: List[Dict[str, Any]] = []
    action_result_json = None
    form_state = request.form.to_dict(flat=True) if request.method == "POST" else {}
    current_project_id = _clean_optional_identifier(
        request.args.get("project_id")
        or request.form.get("current_project_id")
        or request.form.get("project_id")
    )

    try:
        if request.method == "POST":
            action = _clean_text(request.form.get("action"))
            if action == "create_project":
                created = create_project(
                    name=_clean_text(request.form.get("project_name")),
                    project_type=_clean_text(request.form.get("project_type")) or "music_production",
                    description=_clean_text(request.form.get("project_description")),
                    tags=_parse_csv(request.form.get("project_tags")),
                    project_id=_clean_optional_identifier(request.form.get("project_id_override")),
                    projects_root=_projects_root(),
                )
                current_project_id = created["project"]["project_id"]
                action_result = created
                success_message = f"Projeto '{created['project']['name']}' criado."

            elif action == "upload_reference":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de enviar referencias.")
                reference_file = request.files.get("reference_file")
                project_paths = build_project_paths(current_project_id, projects_root=_projects_root())
                stored_path = _copy_uploaded_file(reference_file, project_paths.references_audio_dir, "reference")
                relative_path = os.path.relpath(stored_path, project_paths.project_dir)
                reference = add_reference_asset(
                    project_id=current_project_id,
                    source_type="audio",
                    role=_clean_text(request.form.get("reference_role")) or "reference",
                    caption=_clean_text(request.form.get("reference_caption")) or os.path.basename(stored_path),
                    relative_path=relative_path,
                    origin="audio_webapp_upload",
                    tags=_parse_csv(request.form.get("reference_tags")),
                    metadata={"uploaded_filename": getattr(reference_file, "filename", "")},
                    projects_root=_projects_root(),
                )
                analysis = None
                if _checkbox_enabled("create_reference_analysis", default=False):
                    analysis = analyze_audio(
                        AnalysisRequest(
                            audio_path=stored_path,
                            label=_clean_text(request.form.get("reference_analysis_label")) or reference["caption"],
                            project_id=current_project_id,
                            source_reference_ids=[reference["reference_id"]],
                            persist_to_project=True,
                        )
                    )
                action_result = {"reference": reference, "analysis": analysis}
                success_message = "Referencia local adicionada ao projeto."

            elif action == "register_link":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de registrar links.")
                source_url = _clean_text(request.form.get("source_url"))
                if not source_url:
                    raise ValueError("Informe a URL da referencia externa.")
                metadata_payload = _parse_optional_json(request.form.get("imported_metadata_json"), "imported_metadata_json")
                if metadata_payload:
                    action_result = import_external_reference_metadata(
                        ExternalReferenceMetadataImportRequest(
                            project_id=current_project_id,
                            projects_root=_projects_root(),
                            source_url=source_url,
                            imported_metadata=metadata_payload,
                            role=_clean_text(request.form.get("link_role")) or "reference",
                            caption=_clean_text(request.form.get("link_caption")),
                            tags=_parse_csv(request.form.get("link_tags")),
                            context_label=_clean_text(request.form.get("context_label")),
                            create_context=_checkbox_enabled("create_link_context", default=True),
                        )
                    )
                    success_message = "Link externo registrado com metadata importada."
                else:
                    reference = add_reference_link(
                        project_id=current_project_id,
                        source_url=source_url,
                        role=_clean_text(request.form.get("link_role")) or "reference",
                        caption=_clean_text(request.form.get("link_caption")),
                        tags=_parse_csv(request.form.get("link_tags")),
                        projects_root=_projects_root(),
                    )
                    action_result = {"reference": reference}
                    success_message = "Link externo registrado no projeto."

            elif action == "capture_loopback":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de iniciar captura.")
                action_result = capture_reference_from_loopback(
                    LoopbackCaptureRequest(
                        project_id=current_project_id,
                        source_url=_clean_text(request.form.get("capture_source_url")),
                        label=_clean_text(request.form.get("capture_label")),
                        duration_seconds=_parse_float(request.form.get("capture_duration_seconds"), "capture_duration_seconds", default=30.0, minimum=1.0),
                        source_mode=_clean_text(request.form.get("capture_source_mode")) or "manual",
                        device_name=_clean_text(request.form.get("capture_device_name")),
                        playback_device_name=_clean_text(request.form.get("capture_playback_device_name")),
                        role=_clean_text(request.form.get("capture_role")) or "reference",
                        reference_audio_mode=_clean_text(request.form.get("capture_reference_audio_mode")) or "instrumental",
                        create_analysis=_checkbox_enabled("capture_create_analysis", default=True),
                        create_separation=_checkbox_enabled("capture_create_separation", default=False),
                        create_working_set=_checkbox_enabled("capture_create_working_set", default=True),
                        create_blueprint=_checkbox_enabled("capture_create_blueprint", default=True),
                        style_prompt=_clean_text(request.form.get("capture_style_prompt")),
                        style_tags=_parse_csv(request.form.get("capture_style_tags")),
                        target_bpm=_parse_optional_float(request.form.get("capture_target_bpm")),
                        bars=_parse_int(request.form.get("capture_bars"), "capture_bars", default=16, minimum=1),
                        steps_per_bar=_parse_int(request.form.get("capture_steps_per_bar"), "capture_steps_per_bar", default=16, minimum=1),
                        keep_capture=_checkbox_enabled("capture_keep_capture", default=False),
                        cleanup_after_flow=_checkbox_enabled("capture_cleanup_after_flow", default=True),
                        projects_root=_projects_root(),
                    )
                )
                success_message = "Captura assistida concluida."

            elif action == "create_blueprint":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de criar blueprint.")
                bundle = _project_bundle(current_project_id) or {}
                selected_ids = [_clean_text(item) for item in request.form.getlist("blueprint_reference_ids") if _clean_text(item)]
                references_by_id = {
                    item.get("reference_id"): item
                    for item in ((bundle.get("manifests") or {}).get("references") or {}).get("items", [])
                }
                project_paths = build_project_paths(current_project_id, projects_root=_projects_root())
                reference_audio_paths: List[str] = []
                for reference_id in selected_ids:
                    item = references_by_id.get(reference_id)
                    relative_path = _clean_text((item or {}).get("relative_path"))
                    if not relative_path:
                        continue
                    absolute_path = os.path.abspath(os.path.join(project_paths.project_dir, relative_path))
                    if os.path.exists(absolute_path):
                        reference_audio_paths.append(absolute_path)
                if not reference_audio_paths:
                    raise ValueError("Selecione pelo menos uma referencia com audio local para montar o blueprint.")
                action_result = create_reference_beat_blueprint(
                    BeatBlueprintRequest(
                        label=_clean_text(request.form.get("blueprint_label")) or "blueprint-webapp",
                        reference_audio_paths=reference_audio_paths,
                        style_prompt=_clean_text(request.form.get("blueprint_style_prompt")),
                        style_tags=_parse_csv(request.form.get("blueprint_style_tags")),
                        target_bpm=_parse_optional_float(request.form.get("blueprint_target_bpm")),
                        bars=_parse_int(request.form.get("blueprint_bars"), "blueprint_bars", default=16, minimum=1),
                        steps_per_bar=_parse_int(request.form.get("blueprint_steps_per_bar"), "blueprint_steps_per_bar", default=16, minimum=1),
                        project_id=current_project_id,
                        source_reference_ids=selected_ids,
                        persist_to_project=True,
                    )
                )
                success_message = "Beat blueprint criado e persistido."

            elif action == "set_active_blueprint":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de ativar blueprint.")
                blueprint_version_id = _clean_text(request.form.get("blueprint_version_id"))
                if not blueprint_version_id:
                    raise ValueError("Escolha um blueprint para ativar.")
                action_result = set_active_beat_blueprint(
                    current_project_id,
                    blueprint_version_id,
                    projects_root=_projects_root(),
                )
                success_message = "Blueprint ativo atualizado."

            elif action == "set_active_working_set":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de ativar working set.")
                working_set_id = _clean_text(request.form.get("working_set_id"))
                if not working_set_id:
                    raise ValueError("Escolha um working set para ativar.")
                action_result = set_active_working_set(
                    current_project_id,
                    working_set_id,
                    projects_root=_projects_root(),
                )
                success_message = "Working set ativo atualizado."

            elif action == "render_blueprint":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de renderizar.")
                action_result = render_blueprint_beat(
                    BlueprintBeatRenderRequest(
                        project_id=current_project_id,
                        label=_clean_text(request.form.get("render_label")) or "blueprint-render-webapp",
                        blueprint_version_id=_clean_optional_identifier(request.form.get("render_blueprint_version_id")),
                        target_duration_seconds=_parse_optional_float(request.form.get("render_target_duration_seconds")),
                        bars_override=_parse_optional_int(request.form.get("render_bars_override")),
                        steps_per_bar_override=_parse_optional_int(request.form.get("render_steps_override")),
                        creativity=_parse_float(request.form.get("render_creativity"), "render_creativity", default=0.35, minimum=0.0),
                        fill_probability=_parse_float(request.form.get("render_fill_probability"), "render_fill_probability", default=0.28, minimum=0.0),
                        create_master=_checkbox_enabled("render_create_master", default=False),
                        persist_to_project=True,
                    )
                )
                success_message = "Render do blueprint concluido."

            elif action == "vocal_fit":
                if not current_project_id:
                    raise ValueError("Selecione um projeto antes de rodar vocal-fit.")
                instrumental_audio_path = _clean_text(request.form.get("vocal_fit_instrumental_audio_path"))
                if not instrumental_audio_path:
                    instrumental_audio_path = _clean_text(request.form.get("vocal_fit_instrumental_candidate"))
                if not instrumental_audio_path:
                    raise ValueError("Informe ou selecione o instrumental de entrada.")
                vocal_file = request.files.get("vocal_fit_vocal_file")
                project_paths = build_project_paths(current_project_id, projects_root=_projects_root())
                stored_vocal_path = _copy_uploaded_file(vocal_file, project_paths.references_audio_dir, "vocal")
                vocal_relative_path = os.path.relpath(stored_vocal_path, project_paths.project_dir)
                vocal_reference = add_reference_asset(
                    project_id=current_project_id,
                    source_type="audio",
                    role="vocal",
                    caption=_clean_text(request.form.get("vocal_fit_vocal_caption")) or os.path.basename(stored_vocal_path),
                    relative_path=vocal_relative_path,
                    origin="audio_webapp_vocal_fit",
                    tags=_parse_csv(request.form.get("vocal_fit_vocal_tags")),
                    metadata={"uploaded_filename": getattr(vocal_file, "filename", "")},
                    projects_root=_projects_root(),
                )
                action_result = fit_vocal_to_beat(
                    VocalFitRequest(
                        instrumental_audio_path=instrumental_audio_path,
                        vocal_audio_path=stored_vocal_path,
                        label=_clean_text(request.form.get("vocal_fit_label")) or "vocal-fit-webapp",
                        ducking_strength=_parse_float(request.form.get("vocal_fit_ducking_strength"), "vocal_fit_ducking_strength", default=0.32, minimum=0.0),
                        instrumental_adaptation_profile=_clean_text(request.form.get("vocal_fit_profile")) or "auto",
                        create_master=_checkbox_enabled("vocal_fit_create_master", default=False),
                        project_id=current_project_id,
                        persist_to_project=True,
                    )
                )
                action_result = {
                    "vocal_reference": vocal_reference,
                    "vocal_fit": action_result,
                }
                success_message = "Vocal-fit concluido."

            else:
                raise ValueError("Acao desconhecida.")

    except Exception as exc:
        error_message = str(exc)

    projects = list_projects(projects_root=_projects_root())
    if not current_project_id and projects:
        current_project_id = _clean_optional_identifier(projects[0].get("project_id"))

    bundle = None
    try:
        bundle = _project_bundle(current_project_id)
    except Exception as exc:
        if error_message is None:
            error_message = str(exc)
        current_project_id = None

    runtime_status = _runtime_status_for_view()
    youtube_background_status = collect_youtube_background_status()
    loopback_devices = list_available_loopback_devices()
    routing_devices = list_available_audio_routing_devices()
    project_audio_candidates = _list_project_audio_candidates(current_project_id)
    counts = _bundle_counts(bundle)

    if action_result is not None:
        action_result_paths = _collect_result_paths(action_result)
        action_result_json = _result_pretty_json(action_result)

    return render_template(
        "index.html",
        success_message=success_message,
        error_message=error_message,
        projects=projects,
        current_project_id=current_project_id,
        current_bundle=bundle,
        current_project=(bundle or {}).get("project"),
        counts=counts,
        runtime_status=runtime_status,
        youtube_background_status=youtube_background_status,
        loopback_devices=loopback_devices,
        routing_devices=routing_devices,
        project_audio_candidates=project_audio_candidates,
        action_result=action_result,
        action_result_paths=action_result_paths,
        action_result_json=action_result_json,
        form_state=form_state,
        project_types=PROJECT_TYPES,
        reference_roles=REFERENCE_ROLES,
        capture_source_modes=CAPTURE_SOURCE_MODES,
        reference_audio_modes=REFERENCE_AUDIO_MODES,
        instrumental_adaptation_profiles=INSTRUMENTAL_ADAPTATION_PROFILES,
        file_url=_file_url,
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("SCOREIA_AUDIO_WEBAPP_HOST", "127.0.0.1"),
        port=int(os.getenv("SCOREIA_AUDIO_WEBAPP_PORT", "5002")),
        debug=False,
        use_reloader=False,
    )

