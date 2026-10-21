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
                "Aviso: assumi que você se refere à interação mais recente compatível.\n\n"
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
        from core.memory import registrar_interacao, salvar_memoria
        from tools.ml.ml_pipeline_flow import (
            aplicar_pos_processamento_features_ml,
            aplicar_feature_selection_temporal_ml,
            carregar_pipeline_e_particionar,
            derivar_postprocessor_pruning_ml,
            derivar_postprocessor_reducao_ml,
            executar_estagio_modelagem_ml,
            inicializar_execucao_ml,
            montar_resumo_dados_ml,
        )

        run_context, caminhos_artefatos_root = inicializar_execucao_ml(config)

        dados_pipeline = carregar_pipeline_e_particionar(config)
        pipeline = dados_pipeline["pipeline"]
        split_info = dados_pipeline["split_info"]
        X_train_base = dados_pipeline["X_train_base"]
        X_test_base = dados_pipeline["X_test_base"]
        X_oot_base = dados_pipeline["X_oot_base"]
        y_train = dados_pipeline["y_train"]
        y_test = dados_pipeline["y_test"]
        y_oot = dados_pipeline["y_oot"]

        resultado_fs = aplicar_feature_selection_temporal_ml(
            config=config,
            pipeline=pipeline,
            split_info=split_info,
            X_train=X_train_base,
            X_test=X_test_base,
            X_oot=X_oot_base,
            y_train=y_train,
        )
        feature_selection_stability = resultado_fs["feature_selection_stability"]
        selected_feature_indices = resultado_fs["selected_feature_indices"]
        feature_names_model = list(getattr(pipeline, "feature_names_out", []))

        resultado_postprocessamento_full = aplicar_pos_processamento_features_ml(
            config=config,
            X_train_base=X_train_base,
            X_test_base=X_test_base,
            X_oot_base=X_oot_base,
            y_train=y_train,
            feature_names_model=feature_names_model,
            selected_feature_indices=selected_feature_indices,
        )
        postprocessor_full = resultado_postprocessamento_full["postprocessor"]
        pipeline.metadata["feature_selection_stability"] = feature_selection_stability
        pipeline.metadata["feature_postprocessing_full"] = resultado_postprocessamento_full["postprocessing_metadata"]

        resumo_dados = montar_resumo_dados_ml(
            split_info=split_info,
            y_train=y_train,
            y_test=y_test,
            y_oot=y_oot,
            feature_selection_stability=feature_selection_stability,
            temporal_feature=config.get("temporal_feature"),
        )
        resumo_dados["segment_strategy"] = config.get("segment_strategy", "global")
        resumo_dados["classification_balance_strategy"] = config.get("classification_balance_strategy", "none")
        resumo_dados["feature_postprocessing_full"] = resultado_postprocessamento_full["postprocessing_metadata"]

        def _public_stage_payload(stage_output):
            return {
                "stage_name": stage_output["stage_name"],
                "source_stage_name": stage_output["source_stage_name"],
                "summary": stage_output["summary"],
                "modelo_final": stage_output["modelo_final"],
                "best_params": stage_output["best_params"],
                "best_cv_score": stage_output["best_cv_score"],
                "metricas_treino": stage_output["metricas_treino"],
                "metricas": stage_output["metricas"],
                "metricas_oot": stage_output["metricas_oot"],
                "threshold_classificacao": stage_output["threshold_classificacao"],
                "selected_feature_names": stage_output["selected_feature_names"],
                "postprocessor_metadata": stage_output["postprocessor_metadata"],
                "caminho_interpretabilidade": stage_output["caminho_interpretabilidade"],
                "caminho_relatorio_html": stage_output["caminho_relatorio_html"],
                "caminho_dataset_scored": stage_output["caminho_dataset_scored"],
                "artefatos_exportados": stage_output["artefatos_exportados"],
                "caminhos_artefatos": stage_output["caminhos_artefatos"],
            }

        model_stage_outputs = {}

        estagio_full = executar_estagio_modelagem_ml(
            stage_name="model_full",
            config=config,
            pipeline=pipeline,
            split_info=split_info,
            X_train_base=X_train_base,
            X_test_base=X_test_base,
            X_oot_base=X_oot_base,
            y_train=y_train,
            y_test=y_test,
            y_oot=y_oot,
            postprocessor=postprocessor_full,
            run_context=run_context,
            caminhos_artefatos_root=caminhos_artefatos_root,
            resumo_dados_base=resumo_dados,
            feature_selection_stability=feature_selection_stability,
            stage_summary={
                "executed": True,
                "method": "baseline_full",
                "n_selected_features": int(len(postprocessor_full.feature_names_out)),
            },
        )
        model_stage_outputs["model_full"] = _public_stage_payload(estagio_full)

        estagio_final = estagio_full
        estagio_fonte_reducao = estagio_full

        pruning_result = derivar_postprocessor_pruning_ml(
            config=config,
            source_stage=estagio_full,
            X_train_base=X_train_base,
            y_train=y_train,
        )
        pruning_summary = {
            "executed": False,
            "reason": pruning_result.get("reason", "disabled"),
            "importance_profile": pruning_result.get("importance_profile"),
        }

        if pruning_result.get("executed"):
            estagio_prunned = executar_estagio_modelagem_ml(
                stage_name="model_prunned",
                config=config,
                pipeline=pipeline,
                split_info=split_info,
                X_train_base=X_train_base,
                X_test_base=X_test_base,
                X_oot_base=X_oot_base,
                y_train=y_train,
                y_test=y_test,
                y_oot=y_oot,
                postprocessor=pruning_result["postprocessor"],
                run_context=run_context,
                caminhos_artefatos_root=caminhos_artefatos_root,
                resumo_dados_base=resumo_dados,
                feature_selection_stability=feature_selection_stability,
                source_stage_name=estagio_full["stage_name"],
                stage_summary=pruning_result["summary"],
            )
            model_stage_outputs["model_prunned"] = _public_stage_payload(estagio_prunned)
            estagio_final = estagio_prunned
            estagio_fonte_reducao = estagio_prunned
            pruning_summary = pruning_result["summary"]

        reduction_result = derivar_postprocessor_reducao_ml(
            config=config,
            source_stage=estagio_fonte_reducao,
            X_train_base=X_train_base,
            y_train=y_train,
        )
        reduction_summary = {
            "executed": False,
            "reason": reduction_result.get("reason", "disabled"),
            "importance_profile": reduction_result.get("importance_profile"),
        }

        if reduction_result.get("executed"):
            estagio_reduced = executar_estagio_modelagem_ml(
                stage_name="model_reduced",
                config=config,
                pipeline=pipeline,
                split_info=split_info,
                X_train_base=X_train_base,
                X_test_base=X_test_base,
                X_oot_base=X_oot_base,
                y_train=y_train,
                y_test=y_test,
                y_oot=y_oot,
                postprocessor=reduction_result["postprocessor"],
                run_context=run_context,
                caminhos_artefatos_root=caminhos_artefatos_root,
                resumo_dados_base=resumo_dados,
                feature_selection_stability=feature_selection_stability,
                source_stage_name=estagio_fonte_reducao["stage_name"],
                stage_summary=reduction_result["summary"],
            )
            model_stage_outputs["model_reduced"] = _public_stage_payload(estagio_reduced)
            estagio_final = estagio_reduced
            reduction_summary = reduction_result["summary"]

        resultado = {
            "metricas_treino": estagio_final["metricas_treino"],
            "metricas": estagio_final["metricas"],
            "metricas_oot": estagio_final["metricas_oot"],
            "threshold_classificacao": estagio_final["threshold_classificacao"],
            "best_params": estagio_final["best_params"],
            "best_cv_score": estagio_final["best_cv_score"],
            "modelo_final": estagio_final["modelo_final"],
            "final_stage_name": estagio_final["stage_name"],
            "run_context": run_context,
            "caminhos_artefatos": caminhos_artefatos_root,
            "resumo_dados": estagio_final["resumo_dados"],
            "metadata": estagio_final["metadata"],
            "analise_tempo_segmento": estagio_final["analise_tempo_segmento"],
            "feature_selection_stability": feature_selection_stability,
            "selected_feature_names": estagio_final["selected_feature_names"],
            "postprocessor_metadata": estagio_final["postprocessor_metadata"],
            "pruning_summary": pruning_summary,
            "reduction_summary": reduction_summary,
            "interpretabilidade": estagio_final["interpretabilidade"],
            "learning_curve": estagio_final["learning_curve"],
            "caminho_interpretabilidade": estagio_final["caminho_interpretabilidade"],
            "caminho_interpretabilidade_run": estagio_final["caminho_interpretabilidade"],
            "caminho_relatorio_html": estagio_final["caminho_relatorio_html"],
            "caminho_relatorio_run": estagio_final["caminho_relatorio_html"],
            "caminho_dataset_scored": estagio_final["caminho_dataset_scored"],
            "artefatos_exportados": estagio_final["artefatos_exportados"],
            "model_stage_outputs": model_stage_outputs,
        }

        registrar_interacao(
            memoria=self.memoria,
            tipo="analise_dados",
            pergunta=f"analise: {config['caminho']}",
            resposta=str(estagio_final["metricas"]),
            tema="analise de dados",
            artefatos=resultado,
        )

        salvar_memoria(self.memoria)
        return resultado


