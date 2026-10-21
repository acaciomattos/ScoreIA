# FASE 7 - Multimodal: Audio

Este documento consolida o baseline atual da fase 7 e a direcao tecnica
recomendada para o proximo ciclo de desenvolvimento.

## Objetivo

Construir o nucleo do ScoreIA para entender e manipular audio musical com foco em:

- transcricao;
- separacao de voz e instrumental;
- analise musical;
- criacao de beats por referencia;
- mixagem;
- masterizacao.

## Ambiente padrao

O ambiente padrao da fase 7 passa a ser:

- `.venv310`

Motivos:

- ja possui `torch`, `torchaudio`, `torchvision` com CUDA ativo;
- ja possui `demucs`, `soundfile` e dependencias importantes do pipeline multimodal;
- evita divergencia entre o Python global CPU-only e o runtime real do projeto.

Diretriz operacional:

- comandos de audio devem usar `.venv310/Scripts/python.exe` quando houver risco de cair no
  Python global;
- a fase 7 deve assumir `.venv310` como ambiente oficial do projeto ate que surja conflito real.

## Arquitetura adotada

A fase 7 segue o mesmo principio da fase 6:

- `tools/audio_gen/`: runtime, schemas e persistencia;
- `projects/audio_gen/<project_id>/`: estado duravel do projeto;
- `artifacts/audio_gen/<project_id>/run_<timestamp>/`: artefatos por execucao;
- `models/audio_gen/`: modelos e arquivos locais de apoio.

## Estado atual do baseline

Ja implementado:

- organizacao por `project_id`;
- registry e manifests de audio;
- diagnostico de runtime local;
- transcricao via `ffmpeg` + filtro `whisper`;
- separacao preparada para Demucs;
- analise musical com `ffprobe` + NumPy/SciPy;
- beat blueprint por referencias;
- captura assistida de referencias via `WASAPI` loopback;
- renderizacao de beats com patterns e samples locais;
- autorender de beat a partir do blueprint ativo com kit sintetico minimo local;
- mixagem local via `ffmpeg`;
- masterizacao local em duas etapas com `loudnorm`.

No ambiente atual com `.venv310`:

- `ffmpeg` e `ffprobe`: disponiveis;
- `torch` + CUDA: disponiveis;
- `torchaudio`: disponivel;
- `demucs`: disponivel;
- `soundfile`: disponivel.

Ainda pendente:

- refinamento da webapp para expor esse fluxo ponta a ponta com boa ergonomia.

## Stack recomendada

### Nucleo obrigatorio

- `ffmpeg` + `ffprobe`
  - base deterministica para conversao, render, mix e master;
- `whisper.cpp`
  - transcricao local alinhada ao pipeline atual;
- `Demucs`
  - separacao de voz/instrumental e stems;
- `librosa`
  - analise musical mais rica para ritmo, harmonia e espectro;
- `Basic Pitch`
  - extracao de notas/MIDI e apoio a referencias melodicas;
- `pyloudnorm`
  - validacao de loudness para mix/master;
- `Matchering`
  - mastering por referencia como camada adicional ao `loudnorm`.

### Recomendado

- `pedalboard`
  - efeitos e processamento em Python quando fizer sentido fora do `ffmpeg`.

### Experimental

- `AudioCraft / MusicGen`
  - camada criativa opcional para ideacao;
- `Essentia`
  - nao priorizar no Windows; considerar apenas se houver necessidade clara de MIR avancado.

## Instalacao e configuracao manual

Itens que ja estao atendidos na `.venv310`:

- `torch` com CUDA;
- `torchaudio`;
- `demucs`;
- `soundfile`.

Itens recomendados para instalar no ambiente padrao:

```powershell
.\.venv310\Scripts\python.exe -m pip install librosa basic-pitch pyloudnorm matchering
.\.venv310\Scripts\python.exe -m pip install pedalboard
.\.venv310\Scripts\python.exe -m pip install soundcard
```

Item manual obrigatorio:

- baixar pelo menos um modelo `whisper.cpp` e colocar em `models/audio_gen/whisper/`.

Configuracao manual opcional:

- definir `YOUTUBE_DATA_API_KEY` no ambiente para permitir importacao oficial de metadata
  e resolucao automatica da duracao de videos do YouTube no fluxo de captura loopback.
- instalar `yt-dlp` na `.venv310` para resolver streams do YouTube sem navegador principal:
  `.\.venv310\Scripts\python.exe -m pip install yt-dlp`
- instalar `mpv` no Windows e garantir `mpv.exe` no `PATH`;
- instalar/configurar um device virtual dedicado, se quisermos reproducao isolada em background
  sem poluir o output principal.

Itens a evitar por enquanto:

- `Essentia` no Windows nativo;
- stacks generativas pesadas no mesmo fluxo principal sem necessidade real.

## Politica para referencias externas

### Arquivos locais

Continuam sendo a fonte principal para:

- analise de audio real;
- transcricao;
- separacao;
- render, mix e master.

### Links do YouTube

Viavel incluir como referencia de contexto:

- titulo;
- canal;
- descricao;
- tags;
- duracao;
- playlist;
- anotacoes do usuario.

Nao devem ser tratados como fonte oficial de audio dentro do projeto.
Quando for necessario usar o audio como referencia sonora real, o caminho adotado sera:

- abrir a URL localmente;
- capturar temporariamente a saida do sistema via `WASAPI` loopback;
- promover a captura higienizada a uma referencia local do projeto;
- expurgar os WAVs temporarios depois do fluxo.

### Links do Spotify

Viavel incluir como referencia de contexto:

- track;
- artista;
- album;
- playlist;
- metadados e organizacao de repertorio.

Nao devem ser tratados como fonte oficial de audio para ingestao, cache ou treinamento.
O uso sonoro de referencia segue o mesmo caminho da captura local via `WASAPI` loopback.

## Proposta de ingestao de referencias

Adicionar ao fluxo da fase 7 suporte a:

- `source_type = local_file | youtube_url | spotify_url`

Campos sugeridos:

- `source_url`
- `provider`
- `provider_id`
- `title`
- `artist`
- `album`
- `duration_seconds`
- `notes`
- `import_metadata`

Uso pratico:

- `derived_context` passa a aceitar contexto vindo de metadata;
- `beat_blueprint` pode usar essas referencias como insumo de estilo;
- quando for necessaria analise de audio real, o projeto pode promover um audio local
  correspondente ou gerar uma captura temporaria via `WASAPI` loopback.

## Fluxo de captura de referencias externas

Fluxo adotado para YouTube/Spotify quando o audio precisar virar referencia sonora:

1. registrar a `youtube_url` ou `spotify_url` como referencia externa;
2. opcionalmente importar metadata dessa referencia;
3. abrir a URL localmente de forma assistida;
4. opcionalmente consultar a YouTube Data API para obter metadata e a duracao oficial do video;
5. iniciar captura `WASAPI` loopback no device de saida escolhido;
6. gravar um `.wav` bruto temporario em `artifacts/audio_gen/<project_id>/run_<timestamp>/captures/raw/`;
7. higienizar a captura e salvar um `.wav` limpo temporario em `captures/clean/`;
8. promover o `.wav` limpo para `projects/audio_gen/<project_id>/references/audio/`;
9. gerar `analysis`, `derived_context`, `working_set` e `beat_blueprint` a partir dessa referencia promovida;
10. manter somente o audio promovido e os artefatos finais;
11. expurgar os WAVs temporarios da captura, por fluxo imediato ou politica TTL.

### Caminho recomendado para YouTube-only

Para reduzir atrito e evitar abrir o browser principal do usuario, a fase 7 passa a priorizar
um modo `youtube_background`:

1. resolver a referencia `youtube_url` com metadata oficial;
2. resolver o stream reproduzivel com `yt-dlp`;
3. iniciar reproducao em background com `mpv`;
4. direcionar essa reproducao para um device dedicado, preferencialmente virtual;
5. capturar o loopback desse mesmo device;
6. promover o WAV limpo para `references/audio/`;
7. usar esse asset promovido no pipeline normal de `analysis`, `working_set` e `beat_blueprint`.

Observacoes:

- `mpv` e device virtual formam o caminho recomendado porque permitem isolamento melhor;
- `ffplay` pode servir como fallback, mas tende a depender do device padrao e nao e a trilha preferida;
- por enquanto, o modo background fica focado so em `youtube_url`;
- no `audio_gen`, quando `youtube_background` for usado, o pipeline passa a assumir automaticamente
  os devices do VB-CABLE, sem expor override manual na interface;
- a referencia do link do YouTube passa a aceitar um modo de conteudo:
  - `instrumental`
  - `song_remove_vocals`
  Se o usuario informar `song_remove_vocals`, o WAV capturado e separado e o stem instrumental
  vira a referencia principal para `analysis` e `beat_blueprint`.
- uma vez salvo o `beat_blueprint`, o pipeline ja consegue renderizar automaticamente um beat novo
  a partir dele, sintetizando localmente um kit minimo (`kick`, `snare`, `hat` e `bass` de suporte).
- o autorender agora aceita `target_duration_seconds` e monta secoes (`intro`, `main_a`,
  `fill`, `main_b`, `outro`) com humanizacao controlada por jitter de timing e variacao de ganho.
- o autorender passou a sintetizar tambem `bassline` com movimento e um `motivo` melodico leve,
  guiados pelo contexto tonal inferido da referencia quando disponivel.
- o autorender tambem pode derivar uma camada de `texture` a partir do proprio audio promovido
  da referencia, reduzindo a sensacao de timbre totalmente sintetico.
- o autorender agora tambem pode derivar `chops` ritmicos da referencia e aplicar um `creative mix`
  leve antes do master, aproximando o resultado final de uma pre-producao musical mais plausivel.
- o motor de criacao do instrumental agora tambem aumenta o contraste entre secoes:
  `intro`, `main_a`, `fill`, `main_b`, `break`, `main_c` e `outro` passam a influenciar densidade,
  ganhos por camada, progressao harmonica e atividade de chops, para o beat soar menos como um
  loop unico com pequenas variacoes.
- o render de pattern agora tambem aplica um `kick_bass_sidechain` interno antes do `creative mix`,
  melhorando o encaixe do low-end e dando mais respiracao ao kick e ao bass sem expor parametros
  tecnicos extras na interface.
- o gerador de `bass` e `melody` agora tambem responde melhor as bordas das secoes, ajustando
  escolhas de passos, duracoes e contorno de frase em entradas, fills, breaks e retornos, com
  limites de tessitura mais plausiveis para evitar saltos exagerados.
- as camadas derivadas da referencia tambem ficaram mais ricas: `texture` agora pode costurar mais
  de um segmento do audio promovido ao longo das secoes, e `chop` ganhou tratamento de transiente e
  largura para carregar mais identidade ritmica e timbrica.
- o render de pattern agora tambem aplica um `section_mix` leve em camadas selecionadas como
  `melody`, `texture`, `chop`, `hat` e `clap`, para que mudancas de secao afetem ganho e abertura
  antes mesmo do `creative mix`.
- alem disso, algumas camadas disparadas por trigger agora recebem microvariacoes por secao
  durante o proprio render, para que `chop`, `texture` e frases melodicas possam mudar de carater
  em `intro`, `fill`, `break`, `main_b` e `main_c`, e nao apenas de nivel.
- o beat combinado agora tambem passa por uma automacao `macro_mix` por secao, para que `intro`,
  `fill`, `main_b`, `break`, `main_c` e `outro` mudem energia, largura e corpo do render ja na
  saida do pattern, antes do `creative mix`.
- o render orientado por referencia agora tambem pode gerar uma segunda camada complementar de
  `chop`, usando outro recorte da referencia promovida para reduzir a sensacao de um unico slice
  repetido ao longo de toda a faixa.
- o motor de instrumental agora assume um perfil interno mais claro de hip-hop/rap/boom bap
  anos 90-2000, reforcando swing, escolhas harmonicas e padroes-base de bateria mesmo quando o
  blueprint de origem vier mais neutro.
- o kit local tambem ficou menos reto e mais "sampleado": os padroes default de kick/hat/bass
  ficaram mais sincopados, variantes `alt` passaram a entrar com mais intencao e uma camada
  `snare-ghost` pode ser adicionada automaticamente para dar mais bounce de boombap.
- os timbres sinteticos do kit tambem ficaram mais encorpados e menos limpos: kick e bass com mais
  thump, snare/clap com cauda mais dusty, hats menos brilhantes e a camada melodica um pouco mais
  "tecla/sample" do que beep sintetico puro.
- os one-shots de bateria agora tambem podem ser hibridizados com recortes curtos extraidos da
  propria referencia promovida, para que kick/snare/clap/hat mantenham a estrutura gerada pelo
  motor mas herdem mais ataque, textura e sujeira de source material real.
- esses hits de bateria tambem agora podem variar por secao durante o render, para que hook,
  pre-hook, break, intro e outro nao soem como repeticao do mesmo one-shot: kicks podem vir com
  mais punch, hats mais apertados ou mais shuffle e backbeats mais secos ou mais "crack".
- os chops de referencia agora tambem incluem uma camada mais longa `chop-phrase` em renders medios,
  usando outro recorte da referencia e um padrao de call/response por secao para reforcar identidade
  sample-based em verso, pre-hook, hook e outro.
- as faixas geradas de `bass` e `melody` agora seguem frases de quatro compassos mais proximas de
  boombap, com eventos marcados como `bass_call`, `bass_response`, `turnaround`, `melody_call` e
  `melody_answer`, reduzindo a sensacao de escala automatica e reforcando repeticao com variacao.
- o carregamento local de audio agora usa um fast path para WAV PCM interno que ja esta no sample
  rate e numero de canais esperados, evitando normalizacoes FFmpeg desnecessarias durante o render
  de patterns.
- a extracao de `texture`, `chop` e `perc` agora usa uma pontuacao interna de "crate digging",
  ranqueando candidatos por corpo, ataque, movimento ritmico, equilibrio espectral e penalidades
  de ruido/mud, em vez de escolher apenas o trecho mais alto ou transiente. Essa pontuacao fica
  registrada no manifest para auditoria.
- o render de patterns agora aplica um mapa interno `boombap_mpc_pocket_v1`, deslocando tempo e
  ganho por papel musical: kick, snare/clap, hats, percussoes e chops passam a ocupar um pocket
  menos reto, com resumo agregado no manifest para depuracao.
- o sequenciador de chops tambem passou a responder melhor aos papeis musicais (`verse`,
  `pre_hook`, `hook`) e a aplicar variacoes por trigger, como stutter de pickup, reverse answer
  gate e hook hit levemente pitchado, reforcando a sensacao de sample/chop MPC sem criar novos
  parametros no webapp.
- o render orientado por referencia agora tambem cria uma camada interna `drum-bed`, uma cama
  percussiva/room extraida da propria referencia e costurada por secao, para que a bateria soe
  menos isolada e mais sampleada.
- antes do `creative mix`, o render de pattern tambem adiciona um bus paralelo
  `boombap_parallel_sp1200_bus_v1`, misturando corpo comprimido/saturado de kick, snare, clap,
  hats, percussoes e drum-bed para trazer mais peso e cola ao instrumental.
- o baixo gerado agora usa um mapa dedicado de pocket boombap, com funcoes como `kick_lock`,
  `pocket_answer`, `muted_pickup`, `ghost_low`, `approach_up` e `turnaround`, criando respostas
  nos espacos da bateria e mantendo o registro mais grave/encorpado.
- o render de pattern agora tambem aplica movimentos internos de arranjo por fronteira de secao
  antes do sidechain e do drum bus: cola de entrada para verso, pickup de pre-hook, respiro/lift
  de hook com reverse swell, drops de break e degradacao de outro, para reduzir a sensacao de loop
  continuo.
- a melodia agora tambem guarda uma `motif memory` interna, reaproveitando um motivo central como
  `motif_anchor`, `motif_answer` e `motif_fragment` ao longo das secoes, em vez de gerar frases
  desconectadas a cada trecho.
- os chops derivados da referencia agora incluem bancos dedicados `answer`, `hook` e `transition`,
  permitindo que verso, refrão e bordas de secao puxem familias diferentes de recortes da mesma
  referencia.
- depois de gerar o material principal, o motor tambem pode criar uma camada `resample-motif`
  derivada do proprio `chop`/`melody`/`texture` ja renderizado, adicionando um comportamento mais
  proximo de self-sampling de produtor.
- o render agora tambem aplica um passe `motif_sample_glue_v1`, abrindo e duckando levemente
  `texture`, `chops` e `resample-motif` em volta da melodia principal para que o stack sampleado
  soe mais coeso.
- o render agora tambem adiciona `motif_tonal_memory_bus_v1`, uma cama tonal paralela e sensivel a
  secao derivada de `melody`, `hook chops`, `resample-motif` e `texture`, reforcando memoria de
  refrÃƒÂ£o e assinatura emocional sem abrir novos controles no webapp.
- o render agora tambem aplica `boombap_ensemble_section_contrast_v1`, um passe coletivo de
  contraste por secao para `drums`, `bass` e `sample stack`, deixando `verse`, `pre_hook`, `hook`,
  `break` e `outro` responderem como blocos musicais com mais intencao e menos cara de loop rico
  continuo.
- o render agora tambem aplica `boombap_hook_memory_lift_v1`, reforcando pulsos recorrentes do
  refrÃ£o em `melody`, `hook chops`, `resample`, `tonal glue` e numa camada paralela
  `hook-memory-bus`, para o hook soar mais lembravel e menos como apenas uma secao mais alta.
- a bateria agora tambem passa por `boombap_drum_turnarounds_v1`, criando fills curtos nas bordas
  de secao e uma camada paralela `drum-turnaround-bus` antes de entradas importantes como
  `pre_hook`, `hook`, `break` e retomadas, para o beat soar mais produzido a mao.
- o render agora tambem aplica `boombap_drum_section_evolution_v1`, variando `hat`, `perc`,
  `snare`, `clap` e `kick` dentro da propria secao e adicionando uma camada `drum-evolution-bus`
  para que verso e refrÃ£o continuem se movendo mesmo antes da proxima transicao.
- a bateria agora tambem passa por `boombap_drum_kit_character_v1`, mudando a cor de `kick`,
  `snare`, `clap`, hats, perc e `drum_bed` por secao e adicionando uma camada
  `drum-kit-character-bus`, para que `verse`, `pre_hook`, `hook` e `break` soem como o mesmo kit
  empurrado para direcoes diferentes.
- o render agora tambem aplica `boombap_sample_stack_groove_response_v1`, fazendo `melody`,
  `chops`, `texture`, `resample` e suporte tonal responderem com mais clareza ao movimento de
  `kick/backbeat`, alem de criar uma camada `sample-groove-response-bus` para o sample stack
  respirar com a bateria em vez de apenas ficar por cima dela.
- o sample stack agora tambem passa por `boombap_sample_stack_section_dialogue_v1`, criando um
  call/answer mais claro entre `verse`, `pre_hook` e `hook` e adicionando uma camada
  `sample-section-dialogue-bus`, para as camadas musicais soarem mais conversacionais entre secoes.
- depois dessa cola base, o render agora tambem aplica `motif_topline_signature_v1`, refinando a
  hierarquia do topo harmonico: `hook chops` ganham mais mordida/largura em momentos de refrÃ£o,
  `resample-motif` recebe lift harmonico mais intencional, `texture` fica menos embarrada e a
  melodia principal cede um pouco de pressao em upper-mids para o stack soar menos "camadas
  corretas em paralelo" e mais sampleado/com assinatura.
- como os bancos extras de chop aumentaram a complexidade do render, a fase 7 agora tambem usa
  cache do ranking base de candidatos de chop por referencia/variante, reduzindo o custo do stack
  mais rico sem mudar a selecao musical.
- o pipeline agora ja possui uma primeira etapa de `vocal-fit` para arquivos vocais fornecidos
  pelo usuario: analise comparativa, plano de encaixe, vocal preparado, instrumental com ducking,
  mix combinado e master opcional.
- no `vocal-fit`, o vocal passa a ser a referencia temporal principal: se o instrumental for menor,
  o beat pode ser re-renderizado para cobrir a duracao do vocal, e o ducking responde a atividade
  real da voz em vez de usar apenas uma configuracao global fixa.
- quando esse rerender adaptado acontece, o motor agora pode ancorar o contexto tonal do beat na
  tonalidade estimada do vocal, para que o instrumental ja nasca mais proximo do centro harmonico
  da voz antes da etapa final de mix.
- quando o beat e re-renderizado para `vocal-fit`, o pipeline agora tambem pode fazer um passe de
  `vocal-aware arrangement`, reduzindo camadas por papel e omitindo principalmente elementos
  texturais em barras com maior atividade vocal antes da etapa final de ducking e mix.
- esse passe `vocal-aware` agora tambem pode operar em resolucao mais fina por passo, suavizando
  hats, melodia e chops em trechos mais densos da voz sem precisar “apagar” barras inteiras.
- alem disso, o `vocal-aware` agora detecta blocos de frase da voz e passa a tratar entrada,
  sustentacao, saida e recuperacao como momentos diferentes do arranjo, para o beat “respirar”
  em trechos maiores em vez de reagir apenas barra a barra.
- esses blocos de frase agora tambem definem perfis de contraste como `dense_long` e `open_short`,
  para que trechos densos fiquem mais contidos e retornos abertos recuperem melhor melodia,
  chops e textura.
- quando esse caminho passa pelo rerender completo do beat, o `vocal-aware` tambem pode injetar
  pequenos acentos estruturais nas bordas das frases, como release snares e recovery downbeats,
  reutilizando as proprias camadas do instrumental renderizado.
- depois desse passe, o instrumental tambem pode receber uma etapa dedicada de `phrase energy
  automation`, suavizando ganho por estagio de frase antes do ducking e do espaco espectral.
- essa etapa agora tambem escreve microtransicoes de frase (`pre_entry`, `entry_attack`,
  `release_lift`, `recovery_return`) para que as bordas entre blocos vocais soem menos mecanicas.
- depois do ducking, o instrumental tambem pode passar por uma etapa leve de `spectral space`,
  abrindo um pequeno bolsao de presenca perto do centro espectral do vocal antes do mix final.
- essa etapa de `spectral space` agora tambem pode responder ao perfil das frases vocais, abrindo
  mais espaco em blocos densos e preservando mais conteudo medio/agudo em retornos mais abertos.
- depois disso, o instrumental tambem pode passar por uma etapa de `stereo space`, aliviando o
  centro e devolvendo largura de forma diferente entre frases densas e retornos mais abertos.
- o proprio vocal agora tambem pode passar por uma etapa contextual de `vocal clarity`, reforcando
  presenca, limpando low-mids e controlando melhor a imagem da voz conforme o perfil das frases.
- depois disso, o vocal tambem pode receber automacao fina de ganho por frase, para que entrada,
  sustentacao, saida e recuperacao sejam ajustadas sem depender de um vocal globalmente mais alto.
- o vocal agora tambem pode passar por controle contextual de sibilancia, enquanto o instrumental
  recebe um passe contextual de brilho, para tratar conflitos de altas frequencias de forma mais
  cirurgica do que um corte amplo unico.
- o pipeline agora tambem detecta hotspots reais do vocal, como picos curtos de sibilancia e
  respiracoes provaveis, e usa esses eventos para reforcar o tratamento de altas frequencias do
  vocal e do instrumental apenas onde fizer sentido.
- esses hotspots agora tambem dirigem um controle leve de respiracao no vocal e micro-ducks no
  instrumental, para abrir pequenos bolsos de mix exatamente em momentos problematicos.

Entidades novas do projeto:

- `capture_sessions/manifest.json`
  - rastreia `capture_id`, referencia externa de origem, device usado, caminhos temporarios,
    expurgo, referencia promovida e dados opcionais da YouTube Data API;
- `references/audio/`
  - passa a receber o audio promovido da captura;
- `working_sets`
  - podem agrupar `reference_ids`, `capture_ids` e `derived_context_ids` da mesma sessao.

## Diretriz de produto

Para a fase 7, o melhor resultado vem de um stack hibrido:

- IA para transcricao, separacao, extracao melodica e referencia musical;
- DSP deterministico para render, mix, master e parte do beat pipeline.

Nao tratar geracao musical open source como substituta do nucleo de producao.
Ela entra como camada criativa opcional.

## Backlog recomendado

### Prioridade 1

- definir `.venv310` como ambiente oficial do workspace;
- instalar `librosa`, `basic-pitch`, `pyloudnorm`, `matchering`;
- adicionar modelo `whisper.cpp` em `models/audio_gen/whisper/`;
- validar transcricao e separacao no pipeline real.

### Prioridade 2

- enriquecer a analise musical com `librosa`;
- adicionar extracao de notas/MIDI com `Basic Pitch`;
- evoluir `beat_blueprint` com contexto ritmico, harmonico e melodico.

### Prioridade 3

- suportar referencias `youtube_url` e `spotify_url` com metadata;
- persistir essas referencias no registry de audio;
- usar o contexto importado no blueprint e no working set.

### Prioridade 4

- adicionar captura `WASAPI` loopback para gerar referencias sonoras temporarias;
- higienizar a captura e promover o audio limpo para `references/audio/`;
- registrar `capture_sessions` e politica de expurgo;
- integrar o resultado automaticamente a `analysis`, `working_set` e `beat_blueprint`.

### Prioridade 5

- adicionar `youtube_background` com `yt-dlp + mpv`;
- expor listagem de devices de playback e capture para roteamento isolado;
- documentar o setup recomendado com device virtual;

### Prioridade 6

- criar webapp dedicada de audio nos moldes da fase 6;
- expor upload local e cadastro de links de referencia;
- expor escolha de device, captura assistida e reaproveitamento da referencia promovida.

### Prioridade 7

- avaliar `pedalboard` como complemento do pipeline;
- avaliar `AudioCraft / MusicGen` em trilha experimental separada;
- avaliar automacao futura de abertura do link e sincronizacao de inicio de captura.

## Estrutura prevista de projeto

```text
projects/audio_gen/<project_id>/
  project.json
  references/
    manifest.json
    audio/
    text/
  derived_context/
    manifest.json
    transcripts/
    analyses/
  beat_blueprints/
    manifest.json
    versions/
  mix_sessions/
    manifest.json
  masters/
    manifest.json
  capture_sessions/
    manifest.json
  working_sets/
    manifest.json
```

## Resumo executivo

- ambiente oficial: `.venv310`;
- stack principal: `ffmpeg`, `whisper.cpp`, `Demucs`, `librosa`, `Basic Pitch`,
  `pyloudnorm`, `Matchering`;
- YouTube e Spotify entram como referencias de metadata e contexto;
- quando necessario, a referencia sonora real pode vir de captura local via `WASAPI` loopback;
- o `vocal-fit` agora trata automaticamente hotspots de sibilancia, respiracao e plosivas,
  abrindo espaco espectral e de low-mid de forma contextual no instrumental;
- o `vocal-fit` tambem reage automaticamente a trechos agressivos de presenca/harshness,
  suavizando o vocal e abrindo espaco de presenca no instrumental sem novos parametros;
- o mix final do `vocal-fit` agora passa por um `glue bus` contextual, para dar mais cola,
  controle de low-mid e um acabamento mais coeso sem aumentar a complexidade da interface;
- a voz preparada agora tambem pode receber uma ambiencia curta e contextual por frase, para dar
  profundidade sem perder foco em trechos densos;
- o `vocal-fit` tambem resolve um perfil interno automatico de mix, para modular espaco e glue
  sem transformar essas decisoes em parametros visiveis no webapp;
- o `vocal-fit` agora aceita uma preferencia criativa opcional de adaptacao do instrumental:
  `auto`, `dry`, `spaced`, `dense` ou `hook_support`;
- o fluxo agora tambem detecta macrosecoes vocais e usa `intro`, `verse`, `hook`, `break` e
  `outro` para orientar melhor o rearranjo do instrumental adaptado;
- essas macrosecoes agora tambem influenciam energia, ducking e espaco do vocal/instrumental,
  deixando o mix respirar de forma diferente em `verse`, `hook`, `break` e `outro`;
- as macrosecoes agora tambem influenciam transicoes do arranjo e o `glue bus` final, para
  reforcar melhor retornos como `break -> hook`;
- essas mesmas macrosecoes agora tambem modulam o tratamento do proprio vocal, fazendo com que
  `vocal clarity`, automacao de nivel, controle de sibilancia, respiracoes, plosivas e harshness
  respondam de forma diferente entre `intro`, `verse`, `hook`, `break` e `outro`, sem virar
  parametros visiveis no webapp;
- o instrumental adaptado agora tambem pode receber uma etapa de `macrosection_groove_automation`
  depois de `phrase_energy`, aplicando pequenos `push/hold` por secao para que `hook`, `verse` e
  `break` soem mais intencionais sem depender de time-stretch agressivo;
- depois disso, o instrumental adaptado tambem pode passar por `macrosection_harmonic_support`,
  usando a tonalidade detectada da voz como ancora para reforcos leves de tonica/quinta em
  `hook`, `verse` e `transition`, e relaxando esse suporte em `break/outro`;
- o `vocal-fit` agora tambem pode aplicar `transition_space` curto no vocal e no instrumental,
  usando bordas de macrosecao e releases de frase para abrir largura/ambiencia de forma
  controlada em momentos como `break -> hook`, sem virar um wash grande de reverb;
- depois disso, o vocal ainda pode passar por `vocal_phrase_contrast`, que reforca entradas,
  trechos importantes e alguns hotspots de forma dinamica e localizada, sem depender de subir o
  volume global da voz;
- a trilha de transcricao textual para o `vocal-fit` agora roda por padrao quando existir um modelo
  `whisper.cpp` local compativel, priorizando o `ggml-base.bin` quando presente;
- a execucao de transcricao tenta `GPU -> CPU` e registra status/erros explicitamente no manifesto;
- o parser de transcript do `vocal-fit` agora aceita tanto JSON comum quanto o formato NDJSON
  emitido pelo filtro `whisper` do FFmpeg, convertendo timestamps em milissegundos para segundos
  antes de derivar `line_events`;
- esses `line_events` agora tambem alimentam um passe `transcript_line_support` no instrumental,
  para abrir um pouco mais de espaco em linhas densas e devolver respiracao musical nas saidas de
  frase;
- os mesmos `line_events` agora tambem dirigem pequenas transicoes de `transition_space` no vocal e
  no instrumental, aproximando retornos e finais de linha da performance real transcrita;
- esses `line_events` tambem passaram a ser classificados em perfis internos como `dense_lyric`,
  `emphatic_lyric`, `adlib` e `non_lyric_event`, para que tosses, tags e eventos entre colchetes
  nao sejam tratados como linhas principais da letra;
- esses perfis agora tambem interagem com as macrosecoes, permitindo que uma linha densa em
  `hook` puxe um comportamento diferente de uma linha densa em `verse`, enquanto eventos nao
  liricos ficam mais neutros;
- o construtor de arranjo vocal-aware agora tambem consome esses `line_events` antes do rerender do
  instrumental, gerando overrides estruturais por barra e por passo para servir melhor linhas
  densas, enfaticas, adlibs e eventos nao liricos;
- o passe `macrosection_harmonic_support` agora tambem usa esses `line_events`, deixando o suporte
  harmonico mais leve em linhas densas de `verse`, mais responsivo em `hook` e mais neutro perto de
  eventos nao liricos;
- o passe `macrosection_groove_automation` agora tambem usa esses `line_events`, para deixar o
  groove mais contido sob linhas densas de `verse`, mais responsivo em `hook` e quase neutro em
  eventos nao liricos;
- o passe `phrase_energy_automation` agora tambem usa esses `line_events`, para segurar melhor o
  beat em trechos liricos densos e devolver energia com mais intencao em `hook` e nas saidas de
  linha;
- o motor de criacao do instrumental agora tambem carrega `song_role` e `section_family` por
  secao, permitindo empurrar o arranjo para algo mais proximo de `verse/pre_hook/hook/break`
  sem quebrar a nomenclatura interna existente do renderer;
- o source material local do beat agora tambem pode auto-adicionar um `hat-alt` sintetico em
  beats mais longos, usando uma variante complementar de one-shot para dar mais personalidade ao
  kit sem depender de bancos externos;
- esses `song_role` agora tambem alimentam os passes internos de mix do beat, incluindo
  `section_mix`, `kick_bass_sidechain`, `trigger_variation` e `macro_mix`, para que `verse`,
  `pre_hook` e `hook` soem diferentes de fato dentro do render;
- a familia local de bateria agora tambem se estende para `kick-alt`, `clap-alt` e `snare-alt`
  em renders mais longos, adicionando mais variacao de source material antes mesmo de integrar um
  banco externo;
- evitar multiplicar ambientes sem necessidade real.
