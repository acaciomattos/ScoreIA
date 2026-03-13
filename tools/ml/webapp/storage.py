import os
import uuid
from datetime import datetime
from typing import Optional


WEB_UPLOADS_DIR = os.path.abspath(os.path.join("artifacts", "ml", "web_uploads"))


def _ensure_upload_dir() -> str:
    os.makedirs(WEB_UPLOADS_DIR, exist_ok=True)
    return WEB_UPLOADS_DIR


def _sanitize_ext(ext: str, formato: str) -> str:
    if ext:
        return ext.lower()
    return ".csv" if formato == "csv" else ".parquet"


def _is_valid_persisted_path(path: Optional[str]) -> bool:
    if not path:
        return False
    abs_path = os.path.abspath(path)
    uploads_root = _ensure_upload_dir()
    return abs_path.startswith(uploads_root) and os.path.exists(abs_path)


def persistir_upload_web(file_storage, formato: str) -> str:
    uploads_root = _ensure_upload_dir()
    original_name = getattr(file_storage, "filename", "") or "dataset"
    _, ext = os.path.splitext(original_name)
    ext = _sanitize_ext(ext, formato)

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"web_upload_{ts}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(uploads_root, file_name)
    file_storage.save(path)
    return path


def resolver_dataset_web(file_storage, persisted_dataset_path: Optional[str], formato: str) -> str:
    if file_storage is not None and getattr(file_storage, "filename", ""):
        return persistir_upload_web(file_storage, formato)

    if _is_valid_persisted_path(persisted_dataset_path):
        return os.path.abspath(persisted_dataset_path)

    raise ValueError("Nenhum dataset disponivel. Envie um arquivo ou reutilize um dataset previamente validado.")


def limpar_uploads_web() -> int:
    uploads_root = _ensure_upload_dir()
    removidos = 0

    for nome in os.listdir(uploads_root):
        path = os.path.join(uploads_root, nome)
        if not os.path.isfile(path):
            continue
        try:
            os.remove(path)
            removidos += 1
        except OSError:
            continue

    return removidos
