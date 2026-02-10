from core.agent import Agente

agente = Agente()

print("Comandos disponíveis:")
print("  estudar: <tema>")
print("  planejar: <objetivo>")
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

    if entrada.startswith("estudar:"):
        tema = entrada.replace("estudar:", "").strip()
        resposta = agente.estudar(tema)

    elif entrada.startswith("planejar:"):
        objetivo = entrada.replace("planejar:", "").strip()
        resposta = agente.planejar(objetivo)
        
    elif entrada.lower().startswith("exportar planejamento"):
        resposta = agente.exportar_planejamento_csv()

    elif entrada.lower().startswith("exportar tarefas"):
        resposta = agente.exportar_tarefas_csv()

    elif entrada.startswith("remover tarefa"):
        n = int(entrada.split()[-1])
        resposta = agente.remover_tarefa(n)

    elif entrada.startswith("reordenar tarefa"):
        _, _, o, d = entrada.split()
        resposta = agente.reordenar_tarefa(int(o), int(d))

    elif entrada.startswith("incluir tarefa:"):
        desc = entrada.replace("incluir tarefa:", "").strip()
        resposta = agente.incluir_tarefa(desc)

    else:
        resposta = agente.executar(entrada)

    print("\nAgente:\n", resposta, "\n")
