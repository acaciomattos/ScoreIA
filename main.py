from core.agent import Agente

agente = Agente()

print("Comandos disponíveis:")
print("  estudar: <tema>")
print("  sair\n")

while True:
    entrada = input("Você: ")

    if entrada.lower() == "sair":
        break

    if entrada.startswith("estudar:"):
        tema = entrada.replace("estudar:", "").strip()
        resposta = agente.estudar(tema)
    else:
        resposta = agente.pensar(entrada)

    print("\nAgente:\n", resposta, "\n")
