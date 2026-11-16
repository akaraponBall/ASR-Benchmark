# Tests and troubleshooting

## Automated checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Unit tests do not download weights. They exercise edit counts, empty references, speaker
label permutation, overlap, false alarms, missed speech, same-speaker interval duplication,
unknown speakers, token preservation, schema validation, and manifest path handling. Runner
tests use fake adapters to validate mandatory diarization and output/scoring behavior.
A real end-to-end run uses the ignored local example and installed NeMo adapters.

## Common problems

- **No accuracy numbers:** add reviewed `reference_text` and/or `reference_segments`. Two
  detected speakers alone does not prove that their words were assigned correctly.
- **Model missing offline:** run `uv run asr-bench fetch --model <alias>` with network access.
  A checkpoint cached at another revision does not satisfy a pinned revision.
- **Unknown model or configuration typo:** `uv run asr-bench models`; unknown fields are
  rejected. Run commands from the project root or pass an absolute `--config` path.
- **Output directory exists:** choose another `--out`; results are never overwritten.
- **CUDA unavailable:** use `--device cpu`. `--device cuda` fails explicitly rather than
  silently switching hardware. Default `auto` records its selected device in `run.json`.
- **Out of memory:** CPU and GPU memory must fit both models at once. Use CPU or a larger
  device; reducing an unrelated CLI batch size is not supported here.
- **Incorrect words after silence:** see the long-recording limitation in the architecture
  guide. Increasing latency alone does not solve silence-related model errors.
- **Overlapping voices:** the diarizer may identify overlap while ASR returns one text stream.
  Source separation and overlapping-speech recognition are not implemented.
- **Strange speaker switch boundaries:** inspect raw `diarization` intervals and original
  `tokens` in each result. Attribution is approximate; DER scores activity independently.
- **Interrupted/failed run:** inspect `failure.json` and per-recording JSON; use a new output
  directory on rerun. Failures are not silently skipped or counted as successful results.

Dependency updates are deliberate: change the source/version, run `uv lock`, sync, run unit
checks and both real model smoke tests, then review metrics on a held-out annotated set.
Do not independently upgrade PyTorch or NeMo using pip inside this locked environment.
