# Architecture and extension guide

The benchmark runs in Python and uses FFmpeg for audio conversion. The LFM3 profile
delegates ASR inference to a separate Llama.app process over loopback.

```text
JSONL manifest -> validate -> convert to mono 16 kHz
                                  |
                    +-------------+-------------+
                    |                           |
               ASR adapter                Diarization adapter
             text + timed tokens         speaker activity intervals
                    |                           |
                    +-------- attribution ------+
                                  |
                      speaker-labeled transcript
                                  |
               timing + optional reference scores -> JSON/CSV
```

- `schema.py`: strict Pydantic configuration, samples, tokens and segments. Unknown fields
  and invalid intervals fail instead of being ignored.
- `audio.py`: conversion, duration and input file hashing.
- `adapters.py`: Hugging Face checkpoint resolution and NeMo model boundaries.
- `alignment.py`: joins approximate token emission times with speaker activity.
- `metrics.py`: text normalization, CER/WER, and exact interval DER.
- `measurement.py`: synchronized phase timers and memory sampler.
- `runner.py`: experiment lifecycle, model reuse, warmups, repeats, provenance and outputs.
- `cli.py`: commands and display only.

The OpenThai and Typhoon ASR profiles use the same cache-aware NeMo streaming implementation, configured by
`models.toml`. Shared model code contains no hardcoded model names. The NeMo ASR decoder cache
and hypothesis are recreated per recording. The models are loaded once per model alias
and reused across clips. Both ASR and diarization remain loaded for the complete pipeline.
Benchmark-managed checkpoints use a fixed full Hugging Face revision. `--offline` refuses
network fetches; use `fetch` first when a checkpoint is missing. No copy is made to a second
torch cache. See [LFM3](lfm3.md) for the server-managed checkpoint and turn timestamps.

## Attribution limits

Token times come from encoder frame indices times the preprocessor stride and subsampling
factor. They are emission times, not forced word alignments. The timestamp matcher follows
the publisher's reference heuristic: use covering activity within 0.2 seconds, choosing the
longest segment on overlap; otherwise use the nearest interval within one second; otherwise
retain the previous speaker or `unknown`. Turns split at speaker changes or pauses over one
second. Display times are clamped to the waveform duration. Original token times are retained.
It does not isolate overlapping voices or infer real identities. A single transcript cannot
recover two simultaneous utterances. A detected speaker may have no attributed tokens.

Long recordings are currently decoded as one ASR stream. OpenThai can drop words after long
quiet pauses, as its model card documents. No automatic segmentation has been added: clipping
at pauses while preserving global speaker identities and token timestamps needs a separate,
tested policy. Include silence-heavy and long recordings in evaluation before adopting this
pipeline for an application. The same warning applies to the previous standalone script.

## Add another model

For a compatible prompt-conditioned, cache-aware RNNT checkpoint, add an alias with a pinned
revision under `[asr.<alias>]` in `models.toml`. Validate its language dictionary, attention
context support, token timestamp semantics, and decoding settings before comparison. The
current schema supports only the four tested latency settings. A different model family
requires an explicit adapter; do not assume all `.nemo` checkpoints fit this interface.

For a new family (for example Whisper):

1. Implement `ASRAdapter.transcribe(Path) -> Transcript` and expose a tokenizer with
   `ids_to_tokens` and `ids_to_text`, or introduce a separately validated alignment contract.
2. Produce trustworthy timestamps. Do not fabricate token times from uniform text splitting.
3. Add a typed configuration variant and explicit factory registration in `adapters.py`.
4. Add adapter contract tests and a real recording smoke test; document timestamp precision,
   language support, required dependencies, and latency semantics.

A different speaker detector implements `DiarizationAdapter.detect(Path) -> list[Segment]`. Add its typed configuration/factory and
validate automatic speaker counting, overlap handling and per-recording state reset. Keep
speaker detection mandatory; unsupported models and missing weights should fail explicitly.

The project is structured for application integration, but does not claim service readiness:
there is no request concurrency, job queue, upload validation endpoint, authentication,
streaming transport, timeout/cancellation policy, or application deployment configuration.

## Qwen adapter

`QwenASRConfig` is a separate discriminated configuration variant. It resolves a full
Hugging Face snapshot rather than a `.nemo` file. The runner performs diarization first
and passes activity to `ASRAdapter.transcribe(audio, activity)`. NeMo ignores that input
and returns timestamped tokens; Qwen returns typed speaker segments directly. The runner
uses those segments without running token attribution. See [Qwen](qwen.md).
