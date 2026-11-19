# LFM3-Omni-2B through Llama.app

The `lfm3` profile uses `http://localhost:9931`. In WSL, if Linux loopback refuses
the connection, the benchmark uses Windows `curl.exe` to reach Windows loopback.
To check which models are available from the operating system running Llama.app:

```bash
curl http://localhost:9931/v1/models
```

For Llama.app on Windows, run `curl.exe http://127.0.0.1:9931/v1/models` from WSL.
Before running with `--offline`, cache the pinned diarizer with
`uv run asr-bench fetch --model lfm3`.

The benchmark selects the model when exactly one advertised ID contains the model name.
If Llama.app advertises a custom alias or multiple quantizations, set `model_id` in
`[asr.lfm3]` in `models.toml` to the intended ID. The exact advertised ID is saved in
`run.json`. Llama.app controls the checkpoint, quantization and server settings. Record
the server settings separately when comparing runs.

The script runs the shared diarizer first, slices speaker activity into chunks of at
most 25 seconds, and sends each chunk as base64 WAV to `/v1/chat/completions`. It adapts
the model card's system prompt to `Perform ASR in Thai.` and requests greedy text
generation. Set `language = "English"` for an English smoke test. The model card lists
Thai among text languages and omits it from the speech language list.

Transcript timestamps refer to the audio fed to the model for each speaker turn.
Overlapping intervals can produce duplicate text because each includes the mixed audio.
When the diarizer detects no speech, the script transcribes the recording in chunks
with the speaker marked `unknown`. Text from those chunks triggers the benchmark's
no-detected-speech warning.

ASR wall time includes local request and response handling, queueing and inference.
The device flag controls the benchmark's diarizer. Benchmark memory samples cover the
Python process and its PyTorch allocations. Llama.app memory and startup are outside
those measurements.

If audio is rejected, check that Llama.app has the multimodal model and its projector
available. If the server reports a generation limit, increase `max_tokens` or decrease
`max_segment_s` in `[asr.lfm3]`. Use a fresh output directory after a failed attempt.

Sources: [model card](https://huggingface.co/LiquidAI/LFM3-Omni-2B),
[GGUF card](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF),
[Llama.app API](https://llama.app/docs/api),
[llama.cpp audio request format](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
