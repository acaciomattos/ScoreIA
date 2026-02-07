import json
from pathlib import Path
from datetime import datetime
import uuid

CAMINHO_MEMORIA = Path("memory/profile.json")

def carregar_memoria():
    if CAMINHO_MEMORIA.exists():
        with open(CAMINHO_MEMORIA, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_memoria(memoria):
    with open(CAMINHO_MEMORIA, "w", encoding="utf-8") as f:
        json.dump(memoria, f, indent=2, ensure_ascii=False)

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
    memoria["memoria_episodica"] = memoria["memoria_episodica"][-50:]

    
def obter_interacao_anterior(memoria, offset=1):
    """
    offset=1 → última
    offset=2 → anterior à anterior
    """
    memoria_ep = memoria.get("memoria_episodica", [])
    if len(memoria_ep) >= offset:
        return memoria_ep[-offset]
    return None


def buscar_por_tema(memoria, tema, limite=5):
    return [
        i for i in reversed(memoria.get("memoria_episodica", []))
        if i.get("tema") == tema
    ][:limite]



