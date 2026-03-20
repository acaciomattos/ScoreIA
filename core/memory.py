import json
from pathlib import Path
import os
import uuid
from datetime import datetime

MEMORY_FILE = Path("memory/profile.json")
MAX_INTERACOES = 50


def carregar_memoria():
    if not os.path.exists(MEMORY_FILE):
        return {
            "memoria_episodica": []
        }
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_memoria(memoria):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)


def registrar_interacao(memoria, tipo, pergunta, resposta, tema=None, artefatos=None):
    interacao = {
        "id": str(uuid.uuid4()),
        "tipo": tipo or "geral",
        "tema": tema,
        "pergunta": pergunta,
        "resposta": resposta,
        "artefatos": artefatos or {},
        "timestamp": datetime.now().isoformat()
    }

    memoria.setdefault("memoria_episodica", []).append(interacao)
    memoria["memoria_episodica"] = memoria["memoria_episodica"][-MAX_INTERACOES:]


def obter_interacao_anterior(memoria, offset=1):
    epis = memoria.get("memoria_episodica", [])
    if len(epis) >= offset:
        return epis[-offset]
    return None


def buscar_por_tema(memoria, tema, limite=5):
    resultados = []
    for i in reversed(memoria.get("memoria_episodica", [])):
        if i.get("tema") == tema:
            resultados.append(i)
        if len(resultados) >= limite:
            break
    return resultados


def registrar_padrao_planejamento(memoria, evento):
    """
    Registra eventos estruturados de planejamento para aprendizado de padrões.
    """
    if "eventos_planejamento" not in memoria:
        memoria["eventos_planejamento"] = []

    memoria["eventos_planejamento"].append(...)



    memoria["padroes_planejamento"].append({
        "timestamp": evento.get("timestamp"),
        "acao": evento.get("acao"),              # incluir | remover | reordenar
        "categoria": evento.get("categoria"),    # estudo | administrativa | etc
        "tecnica": evento.get("tecnica"),
        "detalhes": evento.get("detalhes", {})
    })

    # limite de segurança
    memoria["padroes_planejamento"] = memoria["padroes_planejamento"][-100:]


