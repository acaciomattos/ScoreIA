import re

DIAS_SEMANA = [
    "segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"
]

PRIORIDADES = ["alta", "média", "baixa"]


def parsear_tarefas_planejamento(texto):
    tarefas = []
    dia_atual = None

    linhas = texto.splitlines()

    for linha in linhas:
        linha_limpa = linha.strip().lower()

        # Detecta dia da semana
        for dia in DIAS_SEMANA:
            if dia in linha_limpa:
                dia_atual = dia
                break

        # Detecta tarefas numeradas ou com hífen
        match = re.match(r"^(\d+\.|-)\s*(.+)", linha.strip())
        if match:
            descricao = match.group(2)

            prioridade = None
            for p in PRIORIDADES:
                if p in descricao.lower():
                    prioridade = p
                    descricao = re.sub(p, "", descricao, flags=re.IGNORECASE)
                    break

            tarefas.append({
                "descricao": descricao.strip(" ()-"),
                "prioridade": prioridade,
                "dia": dia_atual
            })

    return tarefas

def extrair_tarefas(texto):
    linhas = texto.split("\n")
    return [l.strip("- ").strip() for l in linhas if l.strip()]

