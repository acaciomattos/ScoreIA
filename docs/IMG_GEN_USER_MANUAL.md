# Manual de Uso - Modulo de Geracao de Imagens do ScoreIA

Documento atualizado em `2026-03-26`.

Este manual descreve o fluxo atual do `img_gen`, com foco em uso simples, qualidade de imagem e organizacao opcional por projeto.

## 1. O que o modulo faz hoje

O modulo `img_gen` permite:

- gerar imagens a partir de um prompt simples;
- refinar uma imagem existente com `img2img`;
- editar partes especificas com `inpainting`;
- organizar imagens por `project_id` quando voce quiser contexto persistente;
- reutilizar `visual_bible`, referencias e resultados aprovados dentro de um projeto.

Modos de execucao:

- `modo livre`: sem projeto, salvo em `artifacts/img_gen/_ad_hoc/`;
- `modo por projeto`: com projeto ativo, salvo em `artifacts/img_gen/<project_id>/` e relacionado ao contexto salvo em `projects/img_gen/<project_id>/`.

## 2. Pre-requisitos

Use a venv `.venv310`.

Dependencias principais do runtime:

- `torch`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `huggingface_hub`

Aliases padrao:

- checkpoint base: `sd15_runwayml_base`
- inpainting: `sd15_inpainting_base`

Pastas padrao:

- `models/img_gen/checkpoints/sd15_runwayml_base`
- `models/img_gen/inpainting/sd15_inpainting_base`

## 3. Como abrir a interface

Na raiz do projeto:

```powershell
.\.venv310\Scripts\python.exe .\main.py
```

Depois, no CLI do agente:

```text
gerar imagem
```

A interface abre em `http://127.0.0.1:5001`.

## 4. Como a interface esta organizada hoje

A aba principal e `Geracao`, e ela abre por padrao.

Ordem atual das abas:

- `1. Geracao`
- `2. Projeto`
- `3. Contexto`
- `4. Modelos`

Ideia geral:

- se voce so quer gerar uma imagem rapido, fique em `Geracao`;
- se quiser organizar contexto, abra ou crie um projeto em `Projeto`;
- se quiser montar `visual_bible`, referencias ou imports, use `Contexto`;
- `Modelos` serve para registrar checkpoints e pesos locais quando necessario.

## 5. Geracao mais simples possivel

Hoje ja e possivel gerar imagem sem selecionar projeto e sem preencher campos tecnicos.

Fluxo minimo:

1. Abra a aba `Geracao`.
2. Fique em `Rascunho`.
3. Escreva um prompt simples.
4. Clique em `Gerar imagem`.

Exemplo:

`uma biblioteca magica aconchegante, ilustracao detalhada, luz quente`

Se nenhum projeto estiver ativo, a execucao vai para `artifacts/img_gen/_ad_hoc/`.

## 6. Melhorias de qualidade implementadas

O runtime agora aplica automaticamente um preset de qualidade adequado para a sua maquina:

- scheduler `DPM++ 2M Karras` com `timestep_spacing=trailing`;
- reforco automatico de qualidade no prompt;
- negative prompt tecnico padrao para reduzir artefatos comuns;
- defaults diferentes para `txt2img`, `img2img` e `inpainting`;
- blur automatico na mascara do `inpainting` para melhorar a integracao da borda.

Essas melhorias valem tanto para uso pela webapp quanto para chamadas diretas em Python.

## 7. Defaults atuais recomendados para a RTX 2060 6 GB

### 7.1. Rascunho (`txt2img`)

- `512x512`
- `18 steps`
- `guidance scale = 7.5`
- `2 imagens por rodada`

### 7.2. Refino (`img2img`)

- `512x512`
- `strength = 0.30`
- `20 steps`
- `guidance scale = 7.2`
- `1 imagem por rodada`

### 7.3. Inpainting

- `512x512`
- `strength = 0.75`
- `24 steps`
- `guidance scale = 7.5`
- `1 imagem por rodada`
- `mask blur radius = 6.0`

## 8. Modos de geracao

### 8.1. Rascunho

Use quando voce quer criar uma imagem do zero a partir de texto.

Campos essenciais:

- `Prompt`
- `O que evitar` opcional

Mesmo com um prompt simples, o runtime acrescenta automaticamente um reforco de qualidade e um scheduler melhor.

### 8.2. Refino

Use quando voce ja tem uma imagem-base e quer ajustar composicao, luz, textura ou acabamento mantendo parte da estrutura original.

Campos essenciais:

- `Imagem-base`
- `Prompt`

### 8.3. Inpainting

Use quando voce quer trocar apenas uma parte da imagem.

Campos essenciais:

- `Imagem-base`
- `Mascara`
- `Prompt`

A mascara agora recebe um blur padrao no runtime para reduzir bordas artificiais.

### 8.4. Resultados

Mostra o ultimo resultado e o historico recente para reaproveitamento rapido.

## 9. Quando usar projeto

Projeto e opcional.

Faz sentido usar projeto quando voce quer:

- manter uma `visual_bible`;
- guardar referencias e imagens aprovadas;
- continuar um trabalho em varias sessoes;
- reaproveitar o mesmo universo visual em varias execucoes.

## 10. O que existe na aba Contexto

A aba `Contexto` foi simplificada para quatro blocos:

- `Visual Bible`
- `Referencias`
- `Imports`
- `Working Set e Biblioteca`

### 10.1. Visual Bible

Serve para registrar a identidade visual do projeto:

- resumo;
- `prompt_anchor` e `negative_prompt_anchor`;
- `style_profile`;
- personagens;
- ambientes;
- regras de composicao e continuidade.

Observacao: o fluxo novo deixou `props` fora da interface. Objetos podem ser descritos dentro do ambiente ou do prompt.

### 10.2. Referencias

Agora e possivel fazer upload direto de:

- imagens;
- textos;
- audios;
- videos.

### 10.3. Imports

A importacao entre projetos foi simplificada para uma selecao visual de imagens do projeto de origem.

### 10.4. Working Set e Biblioteca

Use `working_set` para ativar subconjuntos do projeto e a biblioteca para reaproveitar imagens em `Refino` e `Inpainting`.

## 11. Projeto de onboarding

O repositorio inclui um projeto-exemplo:

- `projects/img_gen/onboarding_story_demo/`

Ele ja traz:

- referencias;
- `visual_bible`;
- `working_set`;
- `promoted_assets`;
- uma run de exemplo em `artifacts/img_gen/onboarding_story_demo/run_onboarding_demo_001/`.

## 12. Onde os arquivos ficam salvos

### Projetos

- `projects/img_gen/<project_id>/project.json`
- `projects/img_gen/<project_id>/visual_bibles/manifest.json`
- `projects/img_gen/<project_id>/references/manifest.json`
- `projects/img_gen/<project_id>/promoted_assets/manifest.json`
- `projects/img_gen/<project_id>/working_sets/manifest.json`

### Artefatos

- `artifacts/img_gen/_ad_hoc/run_<timestamp>/...`
- `artifacts/img_gen/<project_id>/run_<timestamp>/...`

### Modelos

- `models/img_gen/registry.json`
- `models/img_gen/checkpoints/...`
- `models/img_gen/inpainting/...`
- `models/img_gen/controlnet/...`
- `models/img_gen/ip_adapter/...`

## 13. Troubleshooting rapido

### 13.1. `Projeto nao encontrado: None`

Ja foi corrigido. O fluxo simples nao exige projeto.

### 13.2. `Informe ip_adapter_image_path quando usar um ip_adapter_alias`

Ja foi corrigido no modo simples. `IP-Adapter` e `ControlNet` nao entram mais como defaults implicitos.

### 13.3. `O campo 'inpaint_mask_image_path' deve apontar para uma imagem`

Isso acontece quando nao ha mascara valida. A interface agora aceita upload direto de imagem-base e mascara.

### 13.4. A imagem nao seguiu bem a `visual_bible`

O sistema hoje reforca a `visual_bible` no prompt composto, mas consistencia perfeita ainda depende de referencia visual, refino e, em fases futuras, pesos adicionais quando fizer sentido.

## 14. Fluxo recomendado daqui para frente

Para uso diario:

1. gerar um `Rascunho`;
2. escolher a melhor imagem;
3. fazer `Refino`;
4. usar `Inpainting` quando precisar corrigir uma area especifica;
5. guardar as melhores imagens no projeto se elas forem reutilizaveis.
