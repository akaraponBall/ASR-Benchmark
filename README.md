# ASR Bench

A local Python project for comparing speech recognition **with mandatory speaker detection**.
Compare OpenThai, Typhoon ASR, and Qwen3-ASR-0.6B, all paired with NVIDIA Nemotron-3-Diarization.
Audio and transcripts stay on your machine. Hugging Face supplies model files; NeMo/PyTorch
perform inference. No hosted inference API, account, or Langfuse service is required.

This is an offline evaluation tool and an application integration starting point. It is
not an HTTP service. Accuracy, speaker attribution, and overlapping speech still need
validation against your own reviewed recordings before production use.

## Setup

```bash
cd ~/projects/asr-bench
uv sync --locked
uv run asr-bench --help
uv run asr-bench models
```

Python 3.13 is pinned in `.python-version`. `uv.lock` pins the environment, and `models.toml`
pins model repository revisions. NeMo comes from a fixed official source commit, **not**
from your separate `~/projects/NeMo` checkout. The default PyTorch installation includes
CUDA libraries on Linux; CPU inference also works without a GPU. GPU runs require a
compatible NVIDIA driver. FFmpeg is bundled via `imageio-ffmpeg`; no system install needed.
Model weights are several GB and are shared through the existing Hugging Face cache.

## First run

The existing `testing_clip.m4a` has been copied into ignored `data/` and is listed in
`examples/local.jsonl`. It has no ground-truth labels. Model predictions are not references.

```bash
# OpenThai and the speaker model are already cached on this machine.
uv run asr-bench run --manifest examples/local.jsonl --model openthai \
  --out results/openthai-first --offline

# Download the second ASR model before comparing (one-time).
uv run asr-bench fetch --model openthai --model typhoon

# Identical audio, latency, diarizer and hardware for both models.
uv run asr-bench run --manifest examples/local.jsonl \
  --model openthai --model typhoon --out results/comparison-01 \
  --offline --device cpu --threads 8 --warmup 1 --repeat 3
```

Every run prints speaker-labeled turns and saves:

- `run.json`: pinned settings, package versions, hardware, normalization and manifest hash.
- `<model>--<sample>--<repeat>.json`: transcript, tokens, speaker turns, raw activity,
  input SHA-256, timing, memory, references (when supplied), and metric components.
- `<model>--<sample>--<repeat>.txt`: readable UTF-8 transcript with timestamped speaker
  turns and full text. Open it in an editor with word wrap enabled.
- `measurements.json`: per-recording measurements.
- `summary.json` and `summary.csv`: per-model comparison, including micro-averaged accuracy.
- `failure.json`: error and completed count if inference fails; already-written results remain.

Choose a **new output directory** each time; existing directories are refused. There is no
fallback to transcription without speakers. If no speech is detected, the activity list is
empty, text is retained with `unknown` speaker, and a warning is recorded.

## Qwen3-ASR

`--model qwen` selects `Qwen/Qwen3-ASR-0.6B`. It transcribes detected speaker turns,
since Qwen's published forced aligner does not support Thai. Its timeline uses speaker
interval boundaries rather than word/token emission times. See [Qwen setup and limitations](docs/qwen.md).

```bash
uv run asr-bench run --manifest examples/local.jsonl --model qwen \
  --out results/qwen-01 --offline
```

## Accuracy requires references

Add human-reviewed text and speaker intervals to a JSONL manifest. Paths are relative to
the manifest file, not your terminal's working directory. Each recording needs a unique ID.

```json
{"id":"example","audio":"../data/example.wav","reference_text":"สวัสดีครับ","reference_segments":[{"start":0.3,"end":1.8,"speaker":"person_A"}]}
```

This is a **schema illustration**, not ground truth for `testing_clip.m4a`. Omit optional
reference fields until they have been reviewed. `reference_segments: []` explicitly means
no speech; an omitted field means no annotation. Supply full-file annotations including
overlap. Predicted speaker IDs are anonymous and are optimally mapped to reference IDs
for scoring. Do not force the known speaker count into detection.

See [metrics](docs/metrics.md), [architecture and adding models](docs/architecture.md),
and [testing and troubleshooting](docs/testing.md).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

NeMo and model logs can be verbose. Redirect terminal output if needed; result files remain
structured UTF-8 JSON. Data, results and the virtual environment are ignored by Git. Keep
private recordings and human annotations out of source control.

## Sources

- [OpenThai model card and reference service](https://huggingface.co/iapp/OpenThai2.0-ASR-Streaming)
- [Typhoon streaming ASR model card](https://huggingface.co/typhoon-ai/typhoon-asr-streaming-nemotron-0.6b)
- [NVIDIA speaker diarization model](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [NeMo Speech source](https://github.com/NVIDIA-NeMo/Speech)
- [uv projects](https://docs.astral.sh/uv/guides/projects/)

The model cards specify OpenMDW-1.1; review the applicable model licenses for your application.
