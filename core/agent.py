import requests
from tools.study import buscar_referencias, extrair_texto
from core.memory import (
    carregar_memoria,
    salvar_memoria,
    registrar_interacao,
    obter_interacao_anterior
)

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
        r = requests.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        return r.json()["response"]

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
- Não inventar contexto
- Não aprender preferências sem confirmação explícita
- Declarar suposições quando inferir
- Não tomar decisões críticas autonomamente
- Apenas recomendar, nunca decidir
"""



    # =========================
    # Inferência de referência
    # =========================
    def _inferir_interacao(self, mensagem, limite=10):
        """
        Se o usuário não especificar explicitamente a interação,
        tenta inferir pela compatibilidade + recência.
        """
        msg = mensagem.lower()
        epis = self.memoria.get("memoria_episodica", [])[-limite:]

        precisa_exercicio = "exercício" in msg
        precisa_codigo = "código" in msg or "script" in msg

        for interacao in reversed(epis):
            resposta = str(interacao.get("resposta", "")).lower()

            if precisa_exercicio and "exercício" in resposta:
                return interacao

            if precisa_codigo and ("exercício" in resposta or "exemplo" in resposta):
                return interacao

        return None

    # =========================
    # Resolução de contexto
    # =========================
    def _resolver_contexto(self, mensagem):
        contexto = []
        assumido = False

        if "anterior à anterior" in mensagem.lower():
            inter = obter_interacao_anterior(self.memoria, offset=2)
            if inter:
                contexto.append(inter)
                return contexto, False

        if "interação anterior" in mensagem.lower():
            inter = obter_interacao_anterior(self.memoria, offset=1)
            if inter:
                contexto.append(inter)
                return contexto, False

        # Inferência implícita
        inter = self._inferir_interacao(mensagem)
        if inter:
            contexto.append(inter)
            assumido = True

        return contexto, assumido

    # =========================
    # Entrada principal
    # =========================
    def executar(self, mensagem, tipo=None, tema=None):
        contexto, assumido = self._resolver_contexto(mensagem)

        prompt = f"""
{self._base_prompt()}

CONTEXTO RECUPERADO:
{contexto}

SOLICITAÇÃO ATUAL:
{mensagem}
"""

        resposta = self._chamar_llm(prompt)

        if assumido:
            resposta = (
                "⚠️ Assumi que você se refere à interação mais recente compatível.\n\n"
                + resposta
            )

        registrar_interacao(
            memoria=self.memoria,
            tipo=tipo or "pensar",
            pergunta=mensagem,
            resposta=resposta,
            tema=tema
        )

        salvar_memoria(self.memoria)
        return resposta


    # =========================
    # Estudo guiado
    # =========================
    def estudar(self, tema):

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
        resposta = self._chamar_llm(prompt)
        
            # REGISTRO DA INTERAÇÃO (PONTO-CHAVE)
        registrar_interacao(
            memoria=self.memoria,
            tipo="estudo",
            pergunta=f"estudar: {tema}",
            resposta=resposta,
            tema=tema
        )

        salvar_memoria(self.memoria)
        return resposta
