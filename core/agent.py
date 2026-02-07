import requests
from tools.study import buscar_referencias, extrair_texto
from core.memory import carregar_memoria, salvar_memoria, registrar_interacao

OLLAMA_URL = "http://localhost:11434/api/generate"


class Agente:
    def __init__(self, modelo="mistral"):
        self.modelo = modelo
        self.memoria = carregar_memoria()

    # =========================
    # Comunicação com o LLM
    # =========================
    def _chamar_llm(self, prompt):
        payload = {
            "model": self.modelo,
            "prompt": prompt,
            "stream": False
        }
        resposta = requests.post(OLLAMA_URL, json=payload)
        resposta.raise_for_status()
        return resposta.json()["response"]

    # =========================
    # Prompt base (governado pelo profile.json)
    # =========================
    def _base_prompt(self):
        p = self.memoria["preferencias"]
        pt = self.memoria["preferencias_tecnicas"]
        ec = self.memoria["estilo_comunicacao"]
        er = self.memoria["estrutura_resposta"]
        prof = self.memoria["profundidade"]

        return f"""
Você é um agente pessoal local, técnico e controlado.

IDIOMA:
- Idioma: {ec['idioma']}
- Tom: {ec['tom']}
- Uso de exemplos: {ec['uso_exemplos']}
- Uso de analogias: {ec['uso_analogias']}

ESTILO DE RESPOSTA:
- Nível de detalhe: {p['nivel_detalhe']}
- Formato: {p['formato_resposta']}
- Usar tópicos: {er['usar_topicos']}
- Usar passos numerados: {er['usar_passos_numerados']}
- Incluir resumo final: {er['resumo_final']}

PROFUNDIDADE:
- Nível: {prof['nivel']}
- Explicitar supostos: {prof['explicitar_supostos']}
- Alertar limitações: {prof['alertar_limitacoes']}

PREFERÊNCIAS TÉCNICAS:
- Linguagens principais: {pt['linguagem_principal']}
- Python: {pt['bibliotecas_preferidas_python']}
- R: {pt['bibliotecas_preferidas_r']}
- Padrão de código: {pt['padrao_codigo']}
- Nível de comentários: {pt['nivel_comentarios']}
- Estilo de documentação: {pt['estilo_documentacao']}

OBJETIVOS ATUAIS:
{self.memoria['objetivos']}

REGRAS:
- Não inferir dados sensíveis
- Não aprender preferências sem confirmação explícita
- Não tomar decisões críticas autonomamente
- Apenas recomendar, nunca decidir
"""

    # =========================
    # Registro seguro de histórico
    # =========================
    def _registrar_acao(self, descricao):
        if self.memoria["politica_memoria"]["registrar_interacoes"]:
            registrar_interacao(self.memoria, descricao)
            salvar_memoria(self.memoria)

    # =========================
    # Pensar / Conversar
    # =========================
    def pensar(self, mensagem):
        self._registrar_acao(f"pensar: {mensagem}")

        prompt = f"""
{self._base_prompt()}

SOLICITAÇÃO DO USUÁRIO:
{mensagem}

Responda respeitando todas as regras acima.
"""
        return self._chamar_llm(prompt)

    # =========================
    # Estudo guiado
    # =========================
    def estudar(self, tema):
        self._registrar_acao(f"estudar: {tema}")

        textos = []

        if self.memoria["restricoes"]["acessar_internet"]:
            refs = buscar_referencias(tema)

            for r in refs:
                try:
                    textos.append(extrair_texto(r["url"]))
                except Exception:
                    pass

        prompt = f"""
{self._base_prompt()}

TEMA DE ESTUDO:
{tema}

MATERIAIS DE REFERÊNCIA:
{textos}

TAREFAS:
1. Explique o tema de forma técnica
2. Dê exemplos práticos
3. Crie 5 exercícios progressivos
4. Aponte limitações e armadilhas comuns
5. Cite todas as referências utilizadas para construir a resposta, indicando o título e a URL de cada uma.

"""
        return self._chamar_llm(prompt)
