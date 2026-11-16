# Initial validation — 2026-11-16

The environment was created using `uv sync --locked` with Python 3.13.18. NeMo is
installed from official commit `205cb6ae0beb5b51c5b279d951baa682a37e7069`, independently
of the original checkout. The lock resolved PyTorch 2.15.0; the exact package set is in
`uv.lock` and each run's `run.json`.

Checks completed:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
uv run --locked asr-bench models
uv run --locked asr-bench run --manifest examples/local.jsonl \
  --model openthai --model typhoon --out results/smoke-01 --offline --device cpu
```

All lint and format checks passed; **9 tests passed**. The smoke test used real cached
weights for both ASR models and the mandatory NVIDIA speaker model. No mock was used
for the model smoke test. Model download also completed for Typhoon.

| ASR | Speakers detected | Processing seconds | Real-time factor |
|---|---:|---:|---:|
| OpenThai | 2 | 20.79 | 0.420 |
| Typhoon | 2 | 16.96 | 0.342 |

Audio duration: 49.557 seconds; CPU, 8 threads, default 1120 ms ASR latency, offline
speaker detection, no warmup, one measured pass. These timings exclude checkpoint
retrieval, loading and conversion. They are smoke-test observations, not a statistically
reliable speed ranking. Both runs share the same speaker detector. There are no reviewed
reference labels for this clip, so CER, WER and DER were correctly left unscored.

Raw outputs are in ignored `results/smoke-01/`; they include full provenance and memory
measurements. GPU inference was not tested. No external service or application deployment
was performed. Long-recording pause segmentation and speaker-attributed text accuracy
metrics are not implemented; see the architecture and metrics guides.

## Qwen addition — 2026-11-16

Installed `qwen-asr==0.0.6` and resolved its pinned Transformers 4.57.6 runtime with
Hugging Face Hub 0.36.2 in `uv.lock`. Downloaded the original Qwen3-ASR-0.6B snapshot,
revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. The `qwen` profile uses detected
speaker-turn boundaries because the published forced aligner does not support Thai.

`uv run --locked pytest -q`: **15 passed**. Ruff lint and formatting checks passed.
Contract tests cover snapshot revision/offline behavior, Qwen mapping, per-turn audio
splitting, mandatory activity input, no fabricated token times, and direct segment
output through the runner and readable report.

Real inference completed for **Qwen, OpenThai and Typhoon** after the dependency update:

```bash
uv run --locked asr-bench run --manifest examples/local.jsonl \
  --model qwen --model openthai --model typhoon \
  --out results/qwen-smoke-01 --offline --device cpu
```

All three pipelines detected two speakers. Qwen processing took 47.98 seconds for the
49.557-second clip, excluding model loading and conversion. This is a single CPU smoke
test, not an accuracy claim. Each pipeline wrote JSON and readable speaker `.txt` output.
No reference accuracy scores were generated. GPU execution remains untested.

## Short Qwen intervals and clip3 comparison — 2026-11-16

The 1922.048-second local `testing_clip3.m4a` exposed a Qwen feature-extractor failure
on a 10 ms detected interval: reflected STFT padding required more samples. The adapter
now right-pads pieces shorter than one analysis window with silence, retaining the
recorded samples and original speaker boundaries. This does not change NeMo inference.
See [Qwen preprocessing](qwen.md).

After the fix, **23 tests passed**, including real feature-extractor checks for inputs
from one sample through normal turn lengths. Ruff lint and formatting checks passed.
Real Qwen inference completed on the full recording after the fix. The completed
OpenThai result predates this Qwen-only fix; Typhoon ran afterward.

| ASR | Processing seconds | RTF |
|---|---:|---:|
| OpenThai | 692.17 | 0.360 |
| Qwen | 1873.99 | 0.975 |
| Typhoon | 703.61 | 0.366 |

All runs used CPU, 8 threads, no warmup, one measured pass, offline pinned checkpoints,
and the same raw diarization activity (8 anonymous speaker labels). The recording has
no reviewed references; CER/WER/DER remain unscored. These are single-recording runtime
observations, not accuracy validation or a reliable general speed ranking.

Combined copied results, summaries and source-run provenance are in ignored
`results/testing_clip3-comparison-20261116-complete/`. The original failed attempt and
completed OpenThai output remain in `results/testing_clip3-comparison-20261116/`;
the successful Qwen/Typhoon retry is in `results/testing_clip3-comparison-20261116-retry/`.
