# AGENTS.md

Instructions for AI coding agents working in this repository.

## Purpose and user preferences

ASR Bench is a local evaluation project for comparing ASR pipelines with **mandatory
speaker diarization**, initially focused on Thai recordings. The user prefers accuracy
and readable, maintainable Python over low latency. Keep responsibilities in small modules.
The project is an evaluation harness and application integration starting point, not a
production HTTP service. Do not claim deployment readiness or measured accuracy without evidence.

## Environment and commands

Work from this repository root. Use `uv` and the project's isolated `.venv`; do not use
or modify the separate `~/projects/NeMo` checkout or its environment for this project.
Python 3.13 is specified by `.python-version` and `pyproject.toml`.

```bash
uv sync --locked
uv run asr-bench --help
uv run asr-bench models
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Use `uv run ruff format .` to apply formatting. Line length is 119. Consult the current
configuration rather than assuming defaults. Keep heavyweight model imports inside the
adapter/runtime code so tests and configuration inspection do not load models.

For dependency changes, use `uv add`/`uv remove` and review both `pyproject.toml` and
`uv.lock`. Never install ad hoc packages into `.venv` with pip and leave the lockfile stale.
NeMo is pinned to an official Git commit through `[tool.uv.sources]`. The official
`qwen-asr==0.0.6` runtime requires Transformers 4.57.6 and Hugging Face Hub <1; resolve
upgrades jointly and verify all affected adapters. Do not blindly upgrade those libraries.

## Architecture

Code lives in `src/asr_bench/`:

| Module | Responsibility |
|---|---|
| `schema.py` | Strict configuration, manifest, segment, token and transcript types |
| `audio.py` | FFmpeg conversion to mono 16 kHz WAV, duration and hashing |
| `adapters.py` | Checkpoint retrieval, model loading and model-specific inference |
| `alignment.py` | Assign NeMo token emission times to detected speaker intervals |
| `measurement.py` | Synchronized phase timers and sampled process/CUDA memory |
| `metrics.py` | Text normalization, CER/WER and interval-based DER |
| `runner.py` | Model lifecycle, mandatory diarization, repeats, warmup, scoring and outputs |
| `reporting.py` | Readable UTF-8 speaker transcript rendering from result dictionaries |
| `cli.py` | CLI arguments and concise result presentation |

`models.toml` maps CLI aliases to pinned Hugging Face repositories and checkpoints:

- `openthai`: `iapp/OpenThai2.0-ASR-Streaming`, `nemo_streaming` adapter.
- `typhoon`: `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b`, same adapter family.
- `qwen`: original `Qwen/Qwen3-ASR-0.6B`, `qwen_asr` adapter; not the `-hf` conversion.
- `[diarization]`: shared `nvidia/Nemotron-3-Diarization` speaker detector.

Keep model identities in configuration, not hardcoded in shared inference code. Add a
validated configuration variant and adapter factory branch for a new model family.
Unsupported settings should fail explicitly rather than silently falling back.

## Model and timestamp contracts

The runner performs diarization before ASR. ASR adapters accept
`transcribe(audio: Path, activity: list[Segment] | None) -> Transcript`.

- NeMo returns text plus timestamped tokens. `SpeakerAttributor` groups them into turns.
  These are approximate **emission times**, not forced word alignments. The default ASR
  latency is 1120 ms. Reset decoder/encoder streaming state for every recording.
- Qwen requires speaker activity, transcribes each detected interval in pieces of at most
  `max_segment_s`, and returns typed segments directly. Its `tokens` list is empty and
  `timestamp_method` is `diarization_turn`. Qwen's published forced aligner does not support
  Thai. Never invent token timestamps or pass Qwen through NeMo's token attribution path.
- Preserve `timestamp_method` in JSON and use the correct heading in readable transcripts.
- Speaker labels are anonymous, local to each recording. Do not force the expected speaker
  count into the detector or imply that labels identify real people across recordings.
- Diarization is not source separation. Overlapping Qwen intervals contain mixed audio and
  may produce duplicate text. NeMo emits one text stream. Document these limits accurately.
- OpenThai can omit speech after long quiet pauses when decoded as one stream. Pause
  segmentation is not currently implemented. Adding it requires preserving global speaker
  identity and correct time offsets, with tests.

## Offline operation and privacy

Inference always runs locally. Hugging Face provides checkpoint files, not hosted inference.
`--offline` means checkpoint resolution must use cached files at the **exact pinned revision**;
it does not enable a separate local inference mode. Do not upload recordings or transcripts,
add hosted inference, or introduce telemetry integrations without an explicit user request.
Reuse the Hugging Face cache; do not create duplicate torch-cache copies of checkpoints.
FFmpeg is provided through `imageio-ffmpeg` and is the only external program needed for audio
conversion. Keep inference in Python rather than launching other inference scripts.

## Evaluation integrity

- Human-reviewed `reference_text` is required for CER/WER; full-file `reference_segments`
  are required for DER. Never use a model prediction as ground truth.
- An omitted reference means unannotated. An empty reference means explicitly empty/silent.
  Missing or zero-denominator rates stay null; retain insertion/false-alarm counts.
- Current text policy: NFC, case/punctuation/numerals preserved, CER removes whitespace,
  WER uses Thai `newmm` tokenization on both sides. Record and document normalization changes;
  do not compare scores across different policies without disclosing the difference.
- DER uses optimal global speaker mapping, zero collar, includes overlap, and evaluates the
  full recording. It scores speaker activity, not which speaker was assigned each word.
- Aggregate by summing error counts and reference units, not by averaging clip percentages.
  Preserve scored-reference coverage so unannotated clips do not dilute error rates.
- Processing time includes ASR, diarization and attribution, excluding checkpoint retrieval,
  model loading, FFmpeg conversion, scoring and writing. Keep excluded phases separately timed.
  Synchronize CUDA at timing boundaries. Record warmups, repeats, hardware and thread count.
- RTF is processing time divided by audio duration. It is not live response latency.
  Memory is sampled process RSS / PyTorch CUDA allocation, not full system or driver memory.
- Keep audio hashes, manifest hash, source hashes, model revisions, package versions and
  effective settings in results. Update documentation when metric semantics change.

## Inputs and result files

JSONL manifests contain one sample per line; audio paths are relative to the manifest.
Sample IDs and model aliases must be safe filename components and sample IDs must be unique.
Reject invalid/nonfinite intervals, unknown configuration fields and invalid input explicitly.
Do not silently skip recordings on failure.

```bash
uv run asr-bench fetch --model qwen
uv run asr-bench run --manifest examples/local.jsonl \
  --model openthai --model typhoon --model qwen \
  --out results/new-comparison --offline --device cpu --threads 8
```

Use a **new output directory** for every run; preserve the refusal to overwrite an existing
experiment. Save per-recording JSON and a matching readable `.txt` with speaker/time turns
and full text, plus run metadata, measurements and JSON/CSV summaries. Report failures
explicitly and retain already-completed results. Do not discard unattributed text; preserve
unknown speakers where the adapter supports them. An empty diarizer result is not proof
that the recording was silent.

`data/`, `results/`, `.venv/` and `.env` are ignored. Keep private audio, transcripts, secrets
and checkpoint weights out of Git. The included local clip has no reviewed reference labels.

## Verification and documentation

For behavioral changes, add focused tests covering the new contract and important edge
cases. Unit tests must not download model weights. Mock model boundaries, not the behavior
being tested. Relevant existing tests cover mandatory diarization, schema validation,
reference scoring, speaker permutation/overlap, token preservation, native Qwen segments,
readable exports, output persistence and refusal to overwrite results.

Run unit tests, Ruff lint and formatting checks before reporting completion. Run real
model smoke tests when changing adapters or runtime dependencies; test other affected
model families after shared dependency changes. Use a new result directory and disclose
what was actually tested. Do not claim GPU validation from a CPU run, or accuracy from a
single unannotated clip. Documentation-only changes do not require model inference.

Read `README.md` and the relevant guide under `docs/` before changing a workflow:

- `architecture.md`: module boundaries and adapter extension.
- `metrics.md`: scoring definitions and comparison limits.
- `qwen.md`: official runtime, Thai timestamp limitation and diarized-turn strategy.
- `testing.md`: tests and troubleshooting.
- `validation.md`: historical checks and smoke-test observations; do not rewrite history
  to imply that earlier runs validated newer code.

Update affected documentation in the same change. Preserve user edits and existing results.
Do not commit, push, or publish unless requested.
