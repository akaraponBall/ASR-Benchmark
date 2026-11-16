# Qwen3-ASR-0.6B

The `qwen` alias maps to **Qwen/Qwen3-ASR-0.6B**, revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. It uses the original checkpoint and
the official `qwen-asr==0.0.6` Transformers runtime, not the separate `-hf` conversion.
The checkpoint and processor files are downloaded into the shared Hugging Face cache.

```bash
uv run asr-bench models
uv run asr-bench fetch --model qwen
uv run asr-bench run --manifest examples/local.jsonl --model qwen \
  --out results/qwen-01 --offline
```

Speaker detection remains mandatory. Qwen supports Thai speech recognition, but its
published forced aligner supports 11 languages that do **not** include Thai. Consequently
this adapter does not claim word/token timestamps or use synthetic token alignment.

The pipeline is:

1. Run the shared NVIDIA diarizer over the full recording.
2. Cut each detected speaker interval into pieces of at most `max_segment_s` (default 25 s).
3. Transcribe each piece with Qwen, using `language="Thai"` and float32 on CPU or GPU.
4. Attach the detected speaker and original interval boundaries to each piece of text.

Very short pieces are right-padded with silence to one feature-extractor analysis window
(400 samples / 25 ms in the pinned runtime). This avoids reflected STFT padding failures
on tiny detected intervals. No recorded samples are discarded or neighboring audio added;
reported speaker boundaries remain unchanged. These tiny intervals may still produce
unreliable text and should be reviewed along with other diarization errors.

The result records `timestamp_method="diarization_turn"`. `tokens` is empty; the timeline
contains detected turn boundaries. NeMo results continue to use `token_emission` and
token-time attribution. Readable transcript headings identify the method.

Missed speech activity is excluded from Qwen's inputs, so diarization misses can become
transcription omissions. Adjacent same-speaker intervals are transcribed separately.
Overlapping intervals contain the original mixture; this does not separate voices and
may transcribe the same words under multiple speakers. No speaker count is forced.
A recording with no detected speech produces an empty transcript and activity list.
Do not interpret a turn's start/end as the exact time of its first/last word.

This pipeline differs from whole-file NeMo streaming. Compare overall application quality
and runtime, and retain the timestamp method in reports. Qwen has no 1120 ms latency
setting in this adapter. Processing time includes diarization and all per-turn ASR calls.

The official runtime requires Transformers 4.57.6 and Hugging Face Hub <1. The project
lockfile resolves these jointly with NeMo; `uv sync --locked` restores that environment.
It also installs the official runtime's dependencies (including demo libraries); this
project does not start a server. ASR loading uses a local snapshot and does not execute
remote custom code. `--offline` requires both Qwen and the diarization checkpoint cached.

Sources: [model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B),
[official runtime](https://github.com/QwenLM/Qwen3-ASR).
