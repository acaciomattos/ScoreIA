import requests
from tools.study import buscar_referencias, extrair_texto
from core.memory import (
    carregar_memoria,
    salvar_memoria,
    registrar_interacao,
    obter_interacao_anterior,
    registrar_padrao_planejamento
)
import csv
import os
from datetime import datetime
from tools.planejamento_parser import parsear_tarefas_planejamento
import uuid
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy



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

    # =========================
    # Planejamento pessoal
    # =========================
    def planejar(self, objetivo, horizonte="semanal"):
        """
        Planejamento de vida pessoal.
        Exemplo de objetivo:
        - "organizar semana focando em estudo e saúde"
        """

        # Contexto por tema (se existir)
        contexto = []
        if objetivo:
            from core.memory import buscar_por_tema
            contexto = buscar_por_tema(self.memoria, tema="organização pessoal")

        prompt = f"""
{self._base_prompt()}

CONTEXTO RELEVANTE:
{contexto}

OBJETIVO DE PLANEJAMENTO:
{objetivo}

HORIZONTE:
{horizonte}

TAREFAS:
1. Crie um plano {horizonte} estruturado por dias
2. Divida o objetivo em tarefas acionáveis
3. Priorize tarefas considerando esforço x impacto
4. Aponte conflitos ou limitações realistas
5. Gere a saída em TEXTO CLARO (não JSON ainda)
"""

        resposta = self._chamar_llm(prompt)

        tarefas = parsear_tarefas_planejamento(resposta)

        # Artefatos iniciais (fase 4 ainda é sem parser rígido)
        artefatos = {
            "tipo": "planejamento",
            "horizonte": horizonte,
            "objetivo": objetivo,
            "conteudo_bruto": resposta,
            "tarefas": tarefas
        }

        registrar_interacao(
            memoria=self.memoria,
            tipo="planejamento",
            pergunta=f"planejar: {objetivo}",
            resposta=resposta,
            tema="organização pessoal",
            artefatos=artefatos
        )

        salvar_memoria(self.memoria)
        return resposta

    # =========================
    # Exportação de planejamentos
    # =========================
    def exportar_planejamento_csv(self, caminho="planejamento.csv", ultimo=True):
        """
        Exporta planejamento(s) para CSV.
        Por padrão exporta apenas o último planejamento.
        """

        planejamentos = [
            i for i in self.memoria.get("memoria_episodica", [])
            if i.get("tipo") == "planejamento"
        ]

        if not planejamentos:
            return "Nenhum planejamento encontrado para exportar."

        if ultimo:
            planejamentos = [planejamentos[-1]]

        # Garante diretório
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)

        with open(caminho, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Cabeçalho
            writer.writerow([
                "id_interacao",
                "timestamp",
                "objetivo",
                "horizonte",
                "conteudo"
            ])

            for p in planejamentos:
                artefatos = p.get("artefatos", {})

                writer.writerow([
                    p.get("id"),
                    p.get("timestamp"),
                    artefatos.get("objetivo"),
                    artefatos.get("horizonte"),
                    p.get("resposta")
                ])

        return f"Planejamento exportado com sucesso para '{caminho}'."

    # =========================
    # Exportação de tarefas
    # =========================
    def exportar_tarefas_csv(self, caminho="tarefas.csv", ultimo=True):
        """
        Exporta tarefas de planejamento para CSV.
        Por padrão exporta as tarefas do último planejamento.
        """

        planejamentos = [
            i for i in self.memoria.get("memoria_episodica", [])
            if i.get("tipo") == "planejamento"
        ]

        if not planejamentos:
            return "Nenhum planejamento encontrado."

        if ultimo:
            planejamentos = [planejamentos[-1]]

        tarefas = []
        for p in planejamentos:
            tarefas.extend(p.get("artefatos", {}).get("tarefas", []))

        if not tarefas:
            return "Nenhuma tarefa encontrada para exportar."

        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)

        with open(caminho, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["descricao", "prioridade", "dia"])

            for t in tarefas:
                writer.writerow([
                    t.get("descricao"),
                    t.get("prioridade"),
                    t.get("dia")
                ])

        return f"Tarefas exportadas com sucesso para '{caminho}'."
    
    # =========================
    # Exclusão de tarefas
    # =========================
    def remover_tarefa(self, indice):
        tarefa = self._obter_tarefas_ativas()[indice]
        tarefa["status"] = "removida"

        registrar_padrao_planejamento(self.memoria, {
            "acao": "remover",
            "categoria": tarefa["categoria"],
            "tecnica": tarefa["tecnica"]
        })

        return tarefa

    # =========================
    # Reordenação de tarefas
    # =========================
    def reordenar_tarefa(self, indice_origem, indice_destino):
        tarefas = self._obter_tarefas_ativas()
        tarefa = tarefas.pop(indice_origem)
        tarefas.insert(indice_destino, tarefa)

        registrar_padrao_planejamento(self.memoria, {
            "acao": "reordenar",
            "categoria": tarefa["categoria"],
            "tecnica": tarefa["tecnica"],
            "detalhes": {
                "from": indice_origem,
                "to": indice_destino
            }
        })

    # =========================
    # Criação de tarefas
    # =========================
    def _criar_tarefa(self, descricao, dia=None, origem="planejamento"):
        return {
            "id": str(uuid.uuid4()),
            "descricao": descricao,
            "dia": dia,
            "categoria": self._inferir_categoria(descricao),
            "tecnica": self._inferir_tecnica(descricao),
            "duracao_estimada": self._inferir_duracao(descricao),
            "prioridade": None,
            "origem": origem,
            "status": "ativa",
            "revisoes": []
        }

    # =========================
    # Inferindo categoria das tarefas
    # =========================
    def _inferir_categoria(self, descricao):
        d = descricao.lower()
        if any(k in d for k in ["estudar", "aprender", "ler"]):
            return "estudo"
        if any(k in d for k in ["email", "organizar", "reunião"]):
            return "administrativa"
        return "geral"

    # =========================
    # Inferindo técnica das tarefas
    # =========================
    def _inferir_tecnica(self, descricao):
        categoria = self._inferir_categoria(descricao)
        return categoria == "estudo"

    # =========================
    # Inferindo duração estimada das tarefas
    # =========================
    def _inferir_duracao(self, descricao):
        if self._inferir_tecnica(descricao):
            return self.memoria["padroes_planejamento"]["duracao_media"]["tecnica"]
        return self.memoria["padroes_planejamento"]["duracao_media"]["nao_tecnica"]

    # =========================
    # Inclusão de tarefas manualmente
    # =========================
    def incluir_tarefa(self, descricao, dia=None):
        tarefa = self._criar_tarefa(descricao, dia, origem="usuario")
        self.memoria["tarefas"].append(tarefa)

        registrar_padrao_planejamento(self.memoria, {
            "acao": "incluir",
            "categoria": tarefa["categoria"],
            "tecnica": tarefa["tecnica"]
        })

        return tarefa

    # =========================
    # Consulta de tarefas ativas
    # =========================
    def _obter_tarefas_ativas(self):
        return [t for t in self.memoria.get("tarefas", []) if t["status"] == "ativa"]

    # =========================
    # Atualização de padrões de planejamento com base em interações
    # =========================
    def atualizar_padroes_planejamento(self):
        eventos = self.memoria.get("eventos_planejamento", [])
        padroes = self.memoria["padroes_planejamento"]

        if not eventos:
            return

        reordenacoes_tecnicas = [
            e for e in eventos
            if e["acao"] == "reordenar" and e["tecnica"]
        ]

        if len(reordenacoes_tecnicas) >= 3:
            padroes["comportamento_revisao"]["reordena_tecnicas_para_topo"] = True

        remocoes_admin = [
            e for e in eventos
            if e["acao"] == "remover" and e["categoria"] == "administrativa"
        ]

        if len(remocoes_admin) >= 2:
            padroes["comportamento_revisao"]["remove_tarefas_administrativas"] = True

    # =========================
    # Refazer planejamento com base em padrões de revisão
    # =========================
    def refazer_planejamento(self):
        tarefas = self._obter_tarefas_ativas()
        padroes = self.memoria["padroes_planejamento"]

        # 1. Reordenar tarefas técnicas
        if padroes["comportamento_revisao"]["reordena_tecnicas_para_topo"]:
            tarefas.sort(key=lambda t: not t["tecnica"])

        # 2. Remover administrativas se padrão indicar
        if padroes["comportamento_revisao"]["remove_tarefas_administrativas"]:
            tarefas = [t for t in tarefas if t["categoria"] != "administrativa"]

        # 3. Recalcular prioridades
        for i, tarefa in enumerate(tarefas):
            tarefa["prioridade"] = i + 1
            tarefa["origem"] = "refatorado_por_padrao"

        self.memoria["tarefas"] = tarefas
        return tarefas


    def analisar_dados(self, config):

        from tools.data_pipeline import DataPipeline
        from tools.model_factory import ModelFactory
        from tools.metrics import (
            metricas_regressao,
            metricas_classificacao
        )
        from tools.optimization import otimizar_modelo_cv
        from core.memory import registrar_interacao, salvar_memoria

        from sklearn.model_selection import KFold, StratifiedKFold
        import numpy as np
        import torch

        # ==========================================================
        # 1. CARREGAMENTO
        # ==========================================================
        pipeline = DataPipeline(config["caminho"])
        pipeline.carregar(config["formato"])

        # ==========================================================
        # 2. HOLDOUT (ANTES DE QUALQUER FIT)
        # ==========================================================
        X_train_raw, X_test_raw, y_train, y_test = pipeline.split_holdout(
            target=config["target"],
            test_size=0.2,
            task=config["task"]
        )

        # ==========================================================
        # 3. FIT TRANSFORM APENAS NO TREINO
        # ==========================================================
        X_train = pipeline.fit_transform(
            X_train_raw,
            modelo=config["modelo"]
        )

        X_test = pipeline.transform(X_test_raw)

        input_dim = X_train.shape[1]

        # ==========================================================
        # 4. CRIAÇÃO DOS FOLDS (SOMENTE TREINO)
        # ==========================================================
        if config["task"] == "classification":
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

        folds = []

        for train_idx, val_idx in kf.split(X_train, y_train):
            folds.append((
                X_train[train_idx],
                X_train[val_idx],
                y_train[train_idx],
                y_train[val_idx]
            ))

        # ==========================================================
        # 5. OTIMIZAÇÃO (SOMENTE TREINO)
        # ==========================================================
        best_params = None
        modelo_final = config["modelo"]

        if config.get("otimizar", False):

            resultado_otimizacao = otimizar_modelo_cv(
                folds=folds,
                task=config["task"],
                modelo=config["modelo"],  # pode ser "lightgbm", "mlp" ou "auto"
                n_trials=config.get("n_trials", 30)
            )

            # Se modo auto
            if config["modelo"] == "auto":

                modelo_final = resultado_otimizacao["melhor_modelo"]
                best_params = resultado_otimizacao["resultado"]["best_params"]
                best_cv_score = resultado_otimizacao["resultado"]["best_score"]

            else:
                best_params = resultado_otimizacao["best_params"]
                best_cv_score = resultado_otimizacao["resultado"]["best_score"]

        # ==========================================================
        # 6. TREINO FINAL (SOMENTE TREINO)
        # ==========================================================
        if modelo_final == "lightgbm":

            model = ModelFactory.criar_lightgbm(
                task=config["task"],
                params=best_params
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

        elif modelo_final == "mlp":

            output_dim = 1 if config["task"] == "regression" else 1

            model = ModelFactory.criar_mlp(
                input_dim=input_dim,
                output_dim=output_dim,
                params=best_params
            )

            # Treino final completo (sem validação)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=best_params.get("lr", 1e-3) if best_params else 1e-3
            )

            criterion = (
                torch.nn.MSELoss()
                if config["task"] == "regression"
                else torch.nn.BCEWithLogitsLoss()
            )

            model.train()

            for epoch in range(200):
                optimizer.zero_grad()
                outputs = model(X_t).squeeze()
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()

            model.eval()

            with torch.no_grad():
                preds = model(
                    torch.tensor(X_test, dtype=torch.float32).to(device)
                ).cpu().numpy()

            y_pred = preds.squeeze()

        else:
            raise ValueError("Modelo final inválido.")

        # ==========================================================
        # 7. MÉTRICAS (APENAS NO TESTE)
        # ==========================================================
        if config["task"] == "regression":
            metricas = metricas_regressao(y_test, y_pred)
        else:
            metricas = metricas_classificacao(y_test, y_pred)

        # ==========================================================
        # 8. REGISTRO
        # ==========================================================
        registrar_interacao(
            memoria=self.memoria,
            tipo="analise_dados",
            pergunta=f"analise: {config['caminho']}",
            resposta=str(metricas),
            tema="análise de dados",
            artefatos={
                "metricas": metricas,
                "best_params": best_params,
                "modelo_final": modelo_final,
                "metadata": pipeline.metadata
            }
        )

        salvar_memoria(self.memoria)

        return {
            "metricas": metricas,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "modelo_final": modelo_final,
            "metadata": pipeline.metadata
        }
