Phase 7 audio-generation modules live in this package.

Current baseline:

- project-scoped state keyed by `project_id`;
- local runtime diagnostics for `ffmpeg`, Whisper-on-`ffmpeg`, NumPy/SciPy, and Demucs availability;
- transcription via the `ffmpeg` `whisper` filter when a local `whisper.cpp` model is available;
- musical analysis using `ffprobe` plus NumPy/SciPy descriptors;
- reference-guided beat blueprints and pattern-based beat rendering from local samples;
- assisted capture of system audio references via `WASAPI` loopback when the optional
  `soundcard` package is available;
- a YouTube-only background path, intended for isolated playback on a dedicated device,
  when optional playback helpers are installed;
- mix and master pipelines built on `ffmpeg` filter graphs.

Core modules:

- `model_paths.py`
  - canonical filesystem layout for `models/audio_gen`
- `project_paths.py`
  - canonical filesystem layout for `projects/audio_gen/<project_id>/...`
- `schemas.py`
  - base records for references, derived context, beat blueprints, mix sessions, masters,
    capture sessions, and working sets
- `project_registry.py`
  - create, list, load, update, and persist project-scoped manifests
- `runtime_env.py`
  - runtime diagnostics for local audio work
- `capture_loopback.py`
  - device listing, assisted URL opening, and `WASAPI` loopback capture helpers
- `youtube_background.py`
  - YouTube stream resolution via `yt-dlp` plus background playback launch for
    `mpv` or `ffplay`
- `audio_pipeline.py`
  - local runners for transcription, separation, analysis, loopback capture, beat rendering,
  mix, and master

Default environment:

- `.venv310`
  - the official runtime for phase 7 work;
  - expected to provide the CUDA-enabled PyTorch stack plus Demucs support.

Expected model layout:

```text
models/audio_gen/
  whisper/
    ggml-base.bin
    ggml-small.bin
  separation/
  cache/
```

Notes:

- Whisper transcription expects a `whisper.cpp` GGML model file under `models/audio_gen/whisper/`
  or an explicit `model_path`.
- Stem separation is wired for Demucs and will report clearly when the active environment
  does not provide the CLI or Python package yet.
- `WASAPI` loopback capture requires the optional package `soundcard` in the active
  environment.
- The recommended YouTube background path requires:
  - `yt-dlp` in the `.venv310`;
  - `mpv` available on `PATH`;
  - a dedicated playback device, ideally a virtual cable, when you want isolated capture
    without polluting the main output device.
- Mix/master and pattern beat rendering already work with local audio files as long as
  `ffmpeg` and `ffprobe` are available.
- `YouTube` and `Spotify` references should be treated as metadata/context inputs, not as
  official audio-ingestion sources for the pipeline.
- The project registry now supports structured `youtube_url` and `spotify_url` references via
  `add_reference_link(...)`, including URL normalization, `provider_id`, and optional metadata
  sidecar persistence under `references/text/`.
- The registry also supports `create_reference_links_working_set(...)` to register a batch of
  external references and bind them into a `working_set` in one step.
- The audio pipeline now supports an explicit metadata-import flow via
  `import_external_reference_metadata(...)`, which updates or creates a saved external reference
  and can emit a `derived_context` summary for the project.
- `create_reference_beat_blueprint(...)` now enriches the blueprint with external reference
  context when `source_reference_ids` point to saved YouTube/Spotify references in the project.
- `capture_reference_from_loopback(...)` now orchestrates:
  `URL -> loopback capture -> cleanup/hygienization -> promoted local reference -> analysis -> working set -> beat blueprint`,
  with optional expurgo of the temporary capture WAVs after the flow.
- For `youtube_url` video references, `capture_reference_from_loopback(...)` can optionally
  enrich metadata and resolve the target capture duration from the YouTube Data API when
  `youtube_api_key` is provided in the request or `YOUTUBE_DATA_API_KEY` exists in the environment.
- The same runner now supports `source_mode="youtube_background"` for a YouTube-only path:
  the URL is resolved with `yt-dlp`, playback is launched in the background, and loopback capture
  records the chosen device without opening the main browser workflow.
- In `youtube_background`, the pipeline now auto-selects the VB-CABLE devices internally:
  playback is routed to `CABLE Input` and capture is taken from the matching VB path, so the web UI
  does not need manual device parameters for this mode.
- `LoopbackCaptureRequest.reference_audio_mode` now distinguishes:
  - `instrumental`
  - `song_remove_vocals`
  When `song_remove_vocals` is used, the captured WAV is separated and the instrumental stem becomes
  the reference asset used for analysis and beat-blueprint generation.
- `render_blueprint_beat(...)` now turns the active project blueprint into an actual beat render:
  it synthesizes a minimal local kit (`kick`, `snare`, `hat`, and optional support `bass`), routes
  the result through the existing pattern renderer, and can optionally master the final WAV.
- `BlueprintBeatRenderRequest` now also supports:
  - `target_duration_seconds`
  - `creativity`
  - `fill_probability`
  - `timing_jitter_ratio`
  - `velocity_jitter_db`
  so the active blueprint can render into a longer arrangement with section-aware variation
  and controlled humanization instead of a short static loop.
- The same autorender path now synthesizes phrase tracks for `bass` and `melody`:
  a moving bassline and a lightweight melodic motif are derived from the reference pitch context
  when the blueprint does not already provide richer melodic layers.
- The autorender can also derive a low-level `texture` layer from the promoted reference audio,
  using a filtered loop taken from the source so the final beat is less purely synthetic.
- The same path can now derive a rhythmic `chop` layer from the promoted reference and route the
  rendered beat through a lightweight `creative mix` bus before optional mastering.
- The beat autorender now also pushes section contrast further:
  `intro`, `main_a`, `fill`, `main_b`, `break`, `main_c`, and `outro` can drive different layer
  gains, drum density, melodic density, bass density, harmonic movement, and chop density so the
  arrangement sounds more like distinct parts of one beat instead of one loop with small mutations.
- The pattern renderer now also applies an internal `kick_bass_sidechain` pass before the creative
  mix stage, helping the low-end breathe better and giving the hybrid synthetic/reference engine a
  more musical pocket without turning that behavior into another user-facing control.
- The phrase-track generator for `bass` and `melody` now also reacts to section borders:
  section entries, fills, breaks, and returns can reshape step choices, note lengths, and phrase
  contours while keeping notes inside a more plausible range for each role.
- Reference-derived `texture` and `chop` layers now also carry more identity:
  the texture renderer can stitch different source segments across sections instead of looping one
  static pad forever, and the chop renderer now adds a stronger transient/stereo treatment so those
  layers contribute more than just background glue.
- The pattern renderer now also applies a lightweight `section_mix` pass on top of selected layers
  such as `melody`, `texture`, `chop`, `hat`, and `clap`, so section changes can affect width and
  gain locally before the final creative mix stage.
- On top of that, selected trigger-based layers now also receive lightweight section-specific
  micro-variations while being rendered, so `chop`, `texture`, and melodic phrases can change
  character across `intro`, `fill`, `break`, `main_b`, and `main_c` instead of only changing
  level or pattern density.
- The combined beat render now also receives a lightweight `macro_mix` automation pass by section,
  so `intro`, `fill`, `main_b`, `break`, `main_c`, and `outro` can feel different in overall
  energy, width, and body before the final creative mix stage.
- Reference-driven chop rendering now also supports a complementary secondary chop layer, so the
  beat can carry two different chop flavors from the promoted reference instead of leaning on one
  repeated slice for the whole arrangement.
- The instrumental renderer now defaults to an internal 90s-2000s hip-hop / boom-bap profile,
  even when the source blueprint is sparse, so swing, progression bias, drum patterns, and mix
  choices start from a more authentic rap/boom-bap baseline instead of a neutral loop template.
- The local drum engine also gained dirtier boom-bap source behavior: default patterns are more
  syncopated, alternate layers are more purposeful, and a `snare-ghost` layer can be auto-added to
  give the groove more bounce and less straight-grid stiffness.
- The synthetic kit timbre was also pushed further toward dusty/sample-based color: kicks and bass
  carry more thump, snares/claps can leave a softer dusty tail, hats are less shiny, and melodic
  one-shots are now a little more key-like instead of purely clean synth bleeps.
- Drum one-shots can now also be hybridized with short transient/source slices extracted from the
  promoted reference audio, so kick/snare/clap/hat layers keep their generated structure while
  inheriting more real sample texture and attack from the source material.
- Those drum layers can now also receive section-aware trigger variation during rendering, so hits
  in `hook`, `pre_hook`, `break`, `intro`, and `outro` are not all exact clones of the same shot:
  kicks can punch harder, hats can tighten or shuffle, and backbeats can crack or dry out by role.
- Reference chops now also include a longer `chop-phrase` layer on medium-length renders. Unlike
  the short stab layers, this phrase layer uses a different source slice and a call/response pattern
  by section, giving the beat more sample-based identity across verse, pre-hook, hook, and outro.
- The generated `bass` and `melody` phrase tracks now use more boom-bap-oriented four-bar call /
  response behavior. Bass events are tagged as `bass_call`, `bass_response`, or `turnaround`, and
  melody events can mark `melody_call` / `melody_answer`, making the harmonic layer feel less like a
  straight scale walk and more like a repeated musical phrase with controlled variation.
- The local audio loader now fast-paths internal PCM WAV files that already match the render sample
  rate and channel count, avoiding unnecessary FFmpeg normalization during pattern rendering.
- Reference-derived `texture`, `chop`, and `perc` extraction now use an internal crate-digging
  score instead of only loudness/transient strength. Candidate slices are ranked by level, body,
  rhythmic motion, attack, spectral balance, and noise/mud penalties, and the chosen score is saved
  in render manifests for auditability.
- Pattern rendering now applies an internal `boombap_mpc_pocket_v1` groove map to drum, percussion,
  and chop triggers. Kicks, snares/claps, hats, percussion, and chops receive role-aware timing and
  gain offsets so the pocket feels less grid-straight while remaining deterministic.
- The reference chop sequencer also reacts more strongly to musical section roles such as `verse`,
  `pre_hook`, and `hook`, with pickup stutters, reverse answer gates, and hook-hit variations applied
  at trigger time instead of exposing extra UI controls.
- Reference-guided instrumental rendering now also adds an internal `drum-bed` layer on medium
  renders. It extracts a filtered rhythmic room/top bed from the promoted reference and stitches it
  across sections to make the kit feel less isolated and more sample-based.
- The pattern renderer also adds a parallel `boombap_parallel_sp1200_bus_v1` drum-glue bus, blending
  compressed/saturated body from kick, snare, clap, hats, percussion, and drum-bed layers before the
  creative mix stage.
- Generated bass phrases now use a dedicated boom-bap pocket map instead of a simple walking line:
  `kick_lock`, `pocket_answer`, `muted_pickup`, `ghost_low`, `approach_up`, and `turnaround` notes
  create more space around the drums while keeping the bass in a lower, heavier register.
- Pattern rendering now applies internal producer-style section moves before sidechain and drum-bus
  glue. Section boundaries can receive verse glue, pre-hook pickup, hook breath/lift with reverse
  swells, break drops, and outro degradation so the instrumental feels arranged rather than looped.
- Melody generation now keeps an internal motif memory: the engine chooses a recurring boom-bap
  motif anchor and reuses it as `motif_anchor`, `motif_answer`, and `motif_fragment` across intro,
  verse, hook, break, and outro instead of treating each section as a fresh unrelated phrase.
- Reference-derived chops now include dedicated banks for `answer`, `hook`, and `transition`, so
  the beat can reuse different slice families for response hits, hook identity, and section pickups.
- After generating the core material, the engine can also derive an internal `resample-motif` layer
  from its own chop/melody/texture output, adding a more producer-like self-sampled accent layer.
- The render now also applies an internal `motif_sample_glue_v1` pass, lightly ducking and widening
  `texture`, `chops`, and `resample-motif` around the active melodic motif so those layers feel
  stacked together instead of competing in the same space.
- After that glue stage, the beat renderer now also applies `motif_topline_signature_v1`, giving
  the melodic top-end a clearer hierarchy: `hook` chops get more bite and width in hook moments,
  `resample-motif` gains a more intentional harmonic lift, `texture` gets cleaner air support, and
  the lead melody tucks a bit of upper-mid pressure so the stack feels more sample-led and less
  like parallel correct layers.
- The renderer now also adds `motif_tonal_memory_bus_v1`, a subtle section-aware parallel tonal bed
  derived from `melody`, `hook` chops, `resample-motif`, and `texture`, reinforcing hook memory
  and emotional signature without exposing new controls in the webapp.
- The beat render now also applies `boombap_ensemble_section_contrast_v1`, a collective section
  contrast pass that makes drums, bass, and the sample stack respond together across `verse`,
  `pre_hook`, `hook`, `break`, and `outro` so the arrangement feels less like a rich continuous
  loop and more like a produced song with section intent.
- The renderer now also applies `boombap_hook_memory_lift_v1`, reinforcing recurring hook pulses
  across `melody`, `hook` chops, `resample`, `tonal glue`, and a subtle `hook-memory-bus` so the
  refrain is easier to remember and feels more like a deliberate chorus than just a louder section.
- The drum stage now also applies `boombap_drum_turnarounds_v1`, generating short section-boundary
  fill accents and a parallel `drum-turnaround-bus` before key entries like `pre_hook`, `hook`,
  `break`, and returning verse moments so the beat feels more hand-produced.
- The renderer now also applies `boombap_drum_section_evolution_v1`, varying hat/perc/snare/clap/
  kick behavior inside the same section and adding a subtle `drum-evolution-bus` so verses and
  hooks keep moving even before the next section boundary arrives.
- The drum stack now also applies `boombap_drum_kit_character_v1`, reshaping the color of kick,
  snare, clap, hats, percussion, and drum bed by section and adding a `drum-kit-character-bus` so
  `verse`, `pre_hook`, `hook`, and `break` feel like the same kit pushed in different directions.
- The renderer now also applies `boombap_sample_stack_groove_response_v1`, making `melody`,
  `chops`, `texture`, `resample`, and tonal support react more clearly to kick/backbeat motion and
  adding a `sample-groove-response-bus` so the sample stack breathes with the drums instead of
  sitting on top of them.
- The sample stack now also applies `boombap_sample_stack_section_dialogue_v1`, shaping a clearer
  internal call/answer relationship between `verse`, `pre_hook`, and `hook` and adding a
  `sample-section-dialogue-bus` so the musical layers feel more conversational across sections.
- The multi-bank chop path now reuses a cached base candidate ranking per reference/variant window,
  reducing render time for the richer boom-bap stack without changing the musical choices.
- `fit_vocal_to_beat(...)` now handles the first `vocal-fit` workflow for user-provided vocal files:
  it analyzes the beat and vocal, produces a fit plan, prepares the vocal, ducks the instrumental,
  renders the combined mix, and can optionally master the result.
- In `vocal-fit`, the vocal is now treated as the master timeline:
  when the instrumental is shorter, the active blueprint can be re-rendered to cover the vocal
  duration, and the ducking stage responds to vocal activity instead of applying a single global
  sidechain setting.
- When the beat is re-rendered for `vocal-fit`, the rerender can now anchor its pitch context to
  the vocal key/scale so the adapted instrumental starts closer to the harmonic center of the
  user-provided voice before the final mix.
- When the beat is re-rendered for `vocal-fit`, the pipeline can also produce a `vocal-aware`
  arrangement pass that reduces selected layers by role and only mutes texture-like elements on
  bars with stronger vocal activity
  before the final ducking and mix stage.
- The `vocal-aware` pass now also reacts at step-level resolution, so dense vocal phrases can
  locally soften hats/melody/chops without flattening the entire bar.
- The same `vocal-aware` stage now groups consecutive vocal bars into phrase blocks and applies
  entry / sustain / release / recovery behavior so the beat can breathe around larger vocal
  phrases instead of reacting only as isolated bars.
- Those phrase blocks now also drive contrast profiles such as `dense_long` and `open_short`,
  so dense vocal passages stay tighter while open returns can recover more melody, chops, and
  texture.
- When the beat passes through the full `vocal-aware` rerender path, that same stage can also
  inject small structural accents at phrase edges, such as release snares and recovery downbeats,
  using the rendered beat's own source layers.
- After the `vocal-aware` rearrangement, the instrumental can now go through a dedicated
  phrase-energy automation pass, smoothing gain by vocal phrase stage before the later ducking
  and spectral-space steps.
- That phrase-energy pass now also writes short transition ramps for `pre_entry`, `entry_attack`,
  `release_lift`, and `recovery_return`, so phrase boundaries feel less like hard gain steps.
- After ducking, the instrumental can also pass through a lightweight `spectral space` stage that
  carves a small presence pocket around the vocal centroid before the final mix.
- That spectral-space stage is now also phrase-aware: denser vocal blocks open a stronger pocket,
  while more open returns keep more mid/high content from the instrumental.
- After that, the instrumental can also pass through a contextual `stereo space` stage that
  slightly relaxes the center and recovers width differently across dense phrases and open
  returns.
- The vocal itself can now also pass through a contextual clarity stage after basic preparation,
  reinforcing presence, trimming low-mid buildup, and tightening the vocal image differently
  across dense phrases and more open returns.
- After that, the vocal can also receive phrase-level gain automation so entry / sustain /
  release / recovery moments are nudged differently without relying on a single louder vocal
  track across the whole song.
- The vocal can now also pass through contextual sibilance control, while the instrumental gets a
  matching contextual brightness-space pass, so high-frequency conflicts are managed more
  surgically than a single broad presence cut.
- The pipeline can now also detect vocal issue hotspots such as short sibilance peaks and likely
  breath regions, and use those hotspots to intensify the corrective high-frequency processing on
  both the vocal and the instrumental only where needed.
- Those detected breath and sibilance hotspots now also drive a light breath-control stage on the
  vocal and micro-duck moves on the instrumental, so the mix can open tiny pockets exactly around
  problematic moments instead of reacting only at the phrase level.
- The same hotspot detection now also identifies likely plosive hits, allowing the vocal to
  receive dedicated low-band plosive control while the instrumental briefly opens extra low-mid
  space around those impacts.
- The hotspot pass now also marks likely harshness regions in the vocal presence band, so the
  vocal can be smoothed locally while the instrumental opens extra presence space only around
  those aggressive moments.
- The final `vocal-fit` mix now also passes through a contextual glue bus, adding light shared
  compression, low-mid cleanup, presence lift, and gentle saturation so the result feels more
  cohesive without exposing extra controls in the UI.
- The prepared vocal can now also receive a contextual short-space pass, adding restrained
  phrase-aware depth that stays tighter in dense passages and opens a little more on release and
  recovery moments.
- The `vocal-fit` chain now also resolves an internal automatic mix profile, so space and glue can
  adapt as `tight`, `balanced`, or `open` behaviors without turning those controls into visible
  webapp parameters.
- `VocalFitRequest` can now also accept an optional high-level instrumental adaptation preference:
  `auto`, `dry`, `spaced`, `dense`, or `hook_support`. This is intended as a future creative UI
  option, not as a low-level technical control.
- The vocal-fit flow now also derives larger vocal macrosections such as `intro`, `verse`,
  `pre_hook`, `hook`, `break`, and `outro`, and uses them to steer the adapted instrumental
  arrangement across longer spans instead of only reacting bar-by-bar.
- Those macrosections now also feed phrase-energy, ducking, and vocal-space stages, so the mix can
  breathe differently in `verse`, `hook`, `break`, and `outro` without exposing new user-facing
  knobs.
- The same macrosection layer now also shapes arrangement transition accents and the final glue
  bus, so `break -> hook` and similar returns can feel more intentional in the finished mix.
- The vocal-side chain now also reacts to those macrosections: contextual clarity, phrase-level
  vocal automation, sibilance control, breath control, plosive control, and harshness control can
  all respond differently across `intro`, `verse`, `hook`, `break`, and `outro` without adding
  more webapp parameters.
- The adapted instrumental can now also pass through `macrosection_groove_automation` after the
  phrase-energy stage, adding subtle push/hold behavior by section so `hook`, `verse`, and
  `break` feel more intentional without changing the user-facing controls or forcing aggressive
  time-stretch.
- After that, the adapted instrumental can also receive `macrosection_harmonic_support`, which
  uses the detected vocal key as a tonal anchor and modulates light tonic/fifth reinforcement by
  macrosection so `hook`, `verse`, and `break` sit more naturally around the vocal's harmonic
  center.
- The vocal-fit chain can now also apply short `transition_space` passes to both the instrumental
  and the vocal, using macrosection borders and phrase releases to add controlled width/ambience
  around moments like `break -> hook` without turning the mix into a large reverb wash.
- After those transition-space passes, the vocal can also receive a dedicated
  `vocal_phrase_contrast` stage that adds subtle line-by-line dynamic emphasis for entries,
  important phrase regions, and selected hotspots without relying on a louder global vocal level.
- The `vocal-fit` flow now transcribes the user vocal by default whenever a compatible local
  `whisper.cpp` model is available, and it derives transcript-based line events from that result.
- The transcript parser now accepts both regular JSON payloads and the NDJSON-style one-object-per-
  line output emitted by the FFmpeg `whisper` filter, converting millisecond timestamps into
  seconds before building line events for downstream vocal-fit stages.
- Those transcript-derived line events now also feed a dedicated `transcript_line_support` pass on
  the instrumental side, so denser lyric lines can receive a little more space while line endings
  can breathe back into the beat more musically.
- The same line events now also steer short `transition_space` accents on both the instrumental and
  vocal paths, so line endings can help shape returns and transitions with more performance-aware
  timing.
- Transcript-derived line events are now also classified into lightweight internal profiles such as
  `dense_lyric`, `emphatic_lyric`, `adlib`, and `non_lyric_event`, so coughs, tags, and bracketed
  cues do not get treated like lead lyrical lines inside the vocal-fit chain.
- Those line profiles now also interact with macrosections, so a dense lyrical line in `hook` can
  trigger a different support/transition response than a dense line in `verse`, and non-lyric
  events can stay much more neutral.
- The vocal-aware arrangement builder now consumes those transcript line events before rerendering
  the adapted instrumental, so structural bar/step overrides can react to lyric density and
  non-lyric events instead of relying only on energy envelopes.
- The `macrosection_harmonic_support` stage now also consumes transcript line events, so harmonic
  reinforcement can stay lighter under dense lyrical lines in `verse`, return harder in `hook`, and
  remain more neutral around non-lyric events.
- The `macrosection_groove_automation` stage now also consumes transcript line events, so groove
  push/hold behavior can stay calmer under dense verse lines, return with more lift in hooks, and
  remain almost neutral around non-lyric events.
- The `phrase_energy_automation` stage now also consumes transcript line events, so dense lyrical
  passages can locally hold the beat in place while hook lines and returns can recover with more
  intention.
- The beat renderer now also annotates each section with a more musical `song_role` and
  `section_family` such as `verse`, `pre_hook`, `hook`, and `break`, while keeping the internal
  renderer names (`main_a`, `main_b`, `main_c`) stable for backward compatibility.
- The local synthetic drum source material can now also auto-add a complementary `hat-alt` layer
  on longer beats, using an alternate one-shot variant so the kit feels less monolithic without
  requiring an external sample bank.
- Those section roles now also feed the internal beat-mix passes (`section_mix`,
  `kick_bass_sidechain`, `trigger_variation`, and `macro_mix`), so `verse`, `pre_hook`, and
  `hook` already feel different in the rendered beat instead of being just labels in the manifest.
- The local synthetic drum family now also extends to `kick-alt`, `clap-alt`, and `snare-alt` on
  longer renders, giving the beat a second layer of source-material variation before any external
  sample-bank integration.
- The consolidated implementation roadmap lives in
  `docs/PHASE_7_MULTIMODAL_AUDIO_PLAN.md`.
