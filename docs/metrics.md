# Measurement policy

## What is measured

| Metric | Reference required | Meaning |
|---|---|---|
| CER | Reviewed transcript | Character insertions + deletions + substitutions / reference characters |
| WER | Reviewed transcript | Same error ratio after Thai `newmm` word segmentation |
| DER | Full-file speaker activity | Missed speech + false alarm + speaker confusion / reference speaker-seconds |
| Speaker count error | Speaker activity | Predicted count minus annotated count; does not measure turn correctness |
| Processing seconds | No | ASR + diarization + attribution, including adapter preprocessing |
| Real-time factor (RTF) | No | Processing seconds / recording duration; lower is faster |
| Phase seconds | No | Separate ASR, diarization and attribution durations |
| Peak process RSS | No | Process resident memory sampled every 20 ms, including loaded models |
| Peak CUDA allocation | No | Maximum PyTorch tensor allocation during the measured iteration |

All error rates are fractions in JSON/CSV (0.1 means 10%); the CLI displays percentages.
Rates can exceed 100%. Missing references yield absent per-item scores and null summary
rates. Empty references have no denominator: rate is null and insertion/false-alarm counts
are retained. Do not replace null with zero. Summary scores sum errors and denominators,
rather than averaging clip percentages. Repeats count as repeated observations. Summary
includes the number of scored observations so coverage is visible.

## Text normalization

Both sides receive Unicode NFC. CER removes all whitespace. Case, punctuation, numbers,
Thai digit forms, and filler words are preserved. WER uses PyThaiNLP's `newmm` tokenizer on
both sides rather than treating Thai text as a single space-delimited word. Version and
normalization policy are recorded with every run. Thai CER is typically easier to compare
across tokenizers. This normalization does **not** reproduce the publishers' leaderboard
normalization; local scores should not be compared directly with their published numbers.

## Diarization

DER uses exact interval boundaries, zero tolerance collar, includes overlapping speakers,
and evaluates the whole recording. A maximum-weight global speaker assignment makes the
score invariant to anonymous label names. Same-speaker overlapping annotations are merged
implicitly by active-speaker sets. Overlap contributes multiple reference speaker-seconds.
No reference speaker count is passed to the detector. Text assignment and raw speaker
activity are stored separately. DER scores activity, **not the accuracy of which text was
assigned to a speaker**. CER/WER ignore speaker labels. Review labeled turns manually;
speaker-attributed transcription metrics are a future extension, not a reported capability.

## Timing and memory interpretation

Checkpoint retrieval and model loading are timed separately from processing, as is FFmpeg
conversion. Processing timing synchronizes CUDA before and after each phase. It excludes
metric calculation, output writing, and warmup. Default warmup=0: the first inference may
include lazy initialization. For steady-state comparisons use `--warmup 1 --repeat 3` or
more; for first-request behavior use zero warmup. Loading is not a process-cold startup
measurement; Python imports and filesystem caches affect it. Warmup uses the same clip.

The ASR latency setting is 1120 ms by default. The offline diarizer uses about 30.4 seconds
of context latency. Neither is measured user-facing response time. This tool processes
finished files; RTF is not live-stream latency or time to first token.

RSS sampling may miss brief allocations and excludes FFmpeg child memory. Its absolute
value includes Python, both models, and retained allocator memory from earlier comparisons.
CUDA allocation excludes the driver and non-PyTorch allocations. Run each model in a fresh
CLI invocation when absolute memory isolation matters. Keep thread count, device, latency,
warmups and reference coverage identical. Do not run simultaneous benchmarks on the same
GPU or infer reliability from one clip. Stratify test recordings by noise, dialect, duration,
speaker count, silence, microphone distance and overlap. Keep a held-out test set separate
from clips used to tune settings.

## Comparing Qwen

Qwen processes diarized turns instead of a whole-file streaming transcript. Its reported
`timestamp_method` is `diarization_turn`; other profiles use `token_emission`. CER/WER
include any omissions from missed speech intervals and any duplicated overlap text. DER
still scores the shared speaker detector, not the text assigned to each speaker. See
[Qwen limitations](qwen.md). Qwen does not use the NeMo latency setting.
