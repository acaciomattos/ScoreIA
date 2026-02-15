from core.agent import Agente
from core.memory import registrar_interacao, salvar_memoria



agente = Agente()


def handle_interacao(agent, mensagem):
    resposta = None
    tipo_interacao = "comando_usuario"
    tema = None
    artefatos = None

    msg = mensagem.strip()

    # -----------------------------
    # ESTUDO
    # -----------------------------
    if msg.startswith("estudar:"):
        tema = msg.replace("estudar:", "").strip()
        resposta = agent.estudar(tema)
        tipo_interacao = "estudo"

    # -----------------------------
    # PLANEJAMENTO
    # -----------------------------
    elif msg.startswith("planejar:"):
        objetivo = msg.replace("planejar:", "").strip()
        resposta = agent.planejar(objetivo)
        tipo_interacao = "planejamento"
        tema = "organização pessoal"

    elif msg.lower().startswith("refazer planejamento"):
        tarefas = agent.refazer_planejamento()
        resposta = "Planejamento refatorado com base nos seus padrões."
        tipo_interacao = "refatoracao_planejamento"
        artefatos = {"tarefas": tarefas}

    # -----------------------------
    # EXPORTAÇÕES
    # -----------------------------
    elif msg.lower().startswith("exportar planejamento"):
        resposta = agent.exportar_planejamento_csv()
        tipo_interacao = "exportacao"

    elif msg.lower().startswith("exportar tarefas"):
        resposta = agent.exportar_tarefas_csv()
        tipo_interacao = "exportacao"

    # -----------------------------
    # REVISÃO DE TAREFAS
    # -----------------------------
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
        resposta = f"Tarefa incluída: {tarefa['descricao']}"
        tipo_interacao = "revisao_tarefa"
        artefatos = {"tarefa_incluida": tarefa}

    # -----------------------------
    # EXECUÇÃO GENÉRICA (pensar / executar)
    # -----------------------------
    else:
        resposta = agent.executar(msg)
        tipo_interacao = "interacao_livre"

    # -----------------------------
    # REGISTRO UNIFICADO
    # -----------------------------
    registrar_interacao(
        memoria=agent.memoria,
        tipo=tipo_interacao,
        pergunta=mensagem,
        resposta=resposta,
        tema=tema,
        artefatos=artefatos
    )

    # -----------------------------
    # APRENDIZADO DE PADRÕES
    # -----------------------------
    agent.atualizar_padroes_planejamento()

    # -----------------------------
    # PERSISTÊNCIA
    # -----------------------------
    salvar_memoria(agent.memoria)

    return resposta


# -----------------------------
# LOOP PRINCIPAL
# -----------------------------

print("Comandos disponíveis:")
print("  estudar: <tema>")
print("  planejar: <objetivo>")
print("  refazer planejamento")
print("  exportar planejamento")
print("  exportar tarefas")
print("  remover tarefa <n>")
print("  reordenar tarefa <origem> <destino>")
print("  incluir tarefa: <descrição>")
print("  sair\n")

while True:
    entrada = input("Você: ")

    if entrada.lower() == "sair":
        break

    resposta = handle_interacao(agente, entrada)

    print("\nAgente:\n", resposta, "\n")
