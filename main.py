import os
import socket
import threading
import time
import webbrowser

from core.agent import Agente
from core.memory import registrar_interacao, salvar_memoria


WEBAPP_HOST = os.getenv("SCOREIA_WEBAPP_HOST", "127.0.0.1")
WEBAPP_PORT = int(os.getenv("SCOREIA_WEBAPP_PORT", "5000"))

agente = Agente()
_webapp_thread = None


def _webapp_url():
    return f"http://{WEBAPP_HOST}:{WEBAPP_PORT}"


def _is_webapp_running(host=WEBAPP_HOST, port=WEBAPP_PORT, timeout=0.3):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _abrir_interface_modelagem():
    global _webapp_thread

    url = _webapp_url()
    if _is_webapp_running():
        webbrowser.open(url)
        return {
            "mensagem": f"Interface de modelagem ja estava ativa em {url}.",
            "artefatos": {"url": url, "status": "already_running"},
        }

    from tools.ml.webapp.app import app

    def _run_webapp():
        app.run(
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
            debug=False,
            use_reloader=False,
        )

    _webapp_thread = threading.Thread(
        target=_run_webapp,
        name="scoreia-webapp",
        daemon=True,
    )
    _webapp_thread.start()

    for _ in range(30):
        if _is_webapp_running():
            break
        time.sleep(0.2)

    if _is_webapp_running():
        webbrowser.open(url)
        return {
            "mensagem": f"Interface de modelagem iniciada em {url}.",
            "artefatos": {"url": url, "status": "started"},
        }

    return {
        "mensagem": (
            f"A tentativa de iniciar a interface web foi disparada, mas a porta {WEBAPP_PORT} "
            f"ainda nao respondeu. Verifique manualmente em {url}."
        ),
        "artefatos": {"url": url, "status": "startup_pending"},
    }


def handle_interacao(agent, mensagem):
    resposta = None
    tipo_interacao = "comando_usuario"
    tema = None
    artefatos = None

    msg = mensagem.strip()
    msg_lower = msg.lower()

    if msg.startswith("estudar:"):
        tema = msg.replace("estudar:", "").strip()
        resposta = agent.estudar(tema)
        tipo_interacao = "estudo"

    elif msg.startswith("planejar:"):
        objetivo = msg.replace("planejar:", "").strip()
        resposta = agent.planejar(objetivo)
        tipo_interacao = "planejamento"
        tema = "organizacao pessoal"

    elif msg_lower.startswith("refazer planejamento"):
        tarefas = agent.refazer_planejamento()
        resposta = "Planejamento refatorado com base nos seus padroes."
        tipo_interacao = "refatoracao_planejamento"
        artefatos = {"tarefas": tarefas}

    elif msg_lower.startswith("exportar planejamento"):
        resposta = agent.exportar_planejamento_csv()
        tipo_interacao = "exportacao"

    elif msg_lower.startswith("exportar tarefas"):
        resposta = agent.exportar_tarefas_csv()
        tipo_interacao = "exportacao"

    elif msg_lower == "modelagem":
        abertura = _abrir_interface_modelagem()
        resposta = abertura["mensagem"]
        tipo_interacao = "modelagem"
        tema = "pipeline ml"
        artefatos = abertura["artefatos"]

    elif msg.startswith("remover tarefa"):
        n = int(msg.split()[-1])
        tarefa = agent.remover_tarefa(n)
        resposta = f"Tarefa removida: {tarefa['descricao']}"
        tipo_interacao = "revisao_tarefa"
        artefatos = {"tarefa_removida": tarefa}

    elif msg.startswith("reordenar tarefa"):
        _, _, origem, destino = msg.split()
        agent.reordenar_tarefa(int(origem), int(destino))
        resposta = "Tarefa reordenada."
        tipo_interacao = "revisao_tarefa"

    elif msg.startswith("incluir tarefa:"):
        desc = msg.replace("incluir tarefa:", "").strip()
        tarefa = agent.incluir_tarefa(desc)
        resposta = f"Tarefa incluida: {tarefa['descricao']}"
        tipo_interacao = "revisao_tarefa"
        artefatos = {"tarefa_incluida": tarefa}

    else:
        resposta = agent.executar(msg)
        tipo_interacao = "interacao_livre"

    registrar_interacao(
        memoria=agent.memoria,
        tipo=tipo_interacao,
        pergunta=mensagem,
        resposta=resposta,
        tema=tema,
        artefatos=artefatos,
    )

    agent.atualizar_padroes_planejamento()
    salvar_memoria(agent.memoria)
    return resposta


def run_cli():
    print("Comandos disponiveis:")
    print("  estudar: <tema>")
    print("  planejar: <objetivo>")
    print("  modelagem")
    print("  refazer planejamento")
    print("  exportar planejamento")
    print("  exportar tarefas")
    print("  remover tarefa <n>")
    print("  reordenar tarefa <origem> <destino>")
    print("  incluir tarefa: <descricao>")
    print("  sair\n")

    while True:
        entrada = input("Voce: ")

        if entrada.lower() == "sair":
            break

        resposta = handle_interacao(agente, entrada)
        print("\nAgente:\n", resposta, "\n")


if __name__ == "__main__":
    run_cli()
