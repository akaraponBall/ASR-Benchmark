"""Model boundaries. Imports and downloads happen only when an adapter is created."""

from pathlib import Path
from typing import Protocol

from .schema import ASRConfig, Checkpoint, DiarizationConfig, QwenASRConfig, Segment, Token, Transcript


class ASRAdapter(Protocol):
    tokenizer: object

    def transcribe(self, audio: Path, activity: list[Segment] | None = None) -> Transcript: ...


class DiarizationAdapter(Protocol):
    def detect(self, audio: Path) -> list[Segment]: ...


class NemoStreamingASR:
    def __init__(self, config: ASRConfig, checkpoint: str, device: str):
        from nemo.collections.asr.models import ASRModel

        self.model = ASRModel.restore_from(checkpoint, map_location=device).eval()
        self.model.encoder.set_default_att_context_size([56, config.latency_ms // 80 - 1])
        decoding = self.model.cfg.decoding.copy()
        decoding.strategy = "greedy_batch"
        decoding.greedy.max_symbols_per_step = config.max_symbols_per_step
        decoding.fused_batch_size = -1
        self.model.change_decoding_strategy(decoding)
        self.model.set_inference_prompt(config.language)
        self.model.decoding.set_strip_lang_tags(True)
        self.tokenizer = self.model.tokenizer
        self.frame_seconds = self.model.encoder.subsampling_factor * self.model.cfg.preprocessor.window_stride

    def transcribe(self, audio: Path, activity: list[Segment] | None = None) -> Transcript:
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        model = self.model
        buffer = CacheAwareStreamingAudioBuffer(model=model)
        buffer.append_audio_file(str(audio))
        channel, time_cache, lengths = model.encoder.get_initial_cache_state(batch_size=1)
        hypotheses, predictions = None, None
        with torch.inference_mode():
            for step, (chunk, chunk_lengths) in enumerate(buffer):
                predictions, texts, channel, time_cache, lengths, hypotheses = model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=channel,
                    cache_last_time=time_cache,
                    cache_last_channel_len=lengths,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=hypotheses,
                    previous_pred_out=predictions,
                    drop_extra_pre_encoded=0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded,
                    return_transcription=True,
                )
        if not hypotheses:
            raise RuntimeError("ASR returned no hypothesis")
        hypothesis = hypotheses[0]
        ids = [int(i) for i in hypothesis.y_sequence]
        timestamps = [float(t) * self.frame_seconds for t in hypothesis.timestamp]
        if len(ids) != len(timestamps):
            raise RuntimeError("token IDs and emission times are misaligned")
        text = texts[0] if isinstance(texts[0], str) else texts[0].text
        return Transcript(text=text, tokens=[Token(id=i, time=t) for i, t in zip(ids, timestamps, strict=True)])


class QwenASR:
    """Transcribe diarized turns: Qwen's forced aligner does not support Thai."""

    tokenizer = None

    def __init__(self, config: QwenASRConfig, checkpoint: str, device: str):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.config = config
        self.model = Qwen3ASRModel.from_pretrained(
            checkpoint,
            dtype=torch.float32,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=config.max_new_tokens,
            local_files_only=True,
        )

    def transcribe(self, audio: Path, activity: list[Segment] | None = None) -> Transcript:
        import numpy as np
        import soundfile as sf
        import torch

        if activity is None:
            raise ValueError("Qwen requires speaker detection before transcription")
        waveform, rate = sf.read(str(audio), dtype="float32")
        duration = len(waveform) / rate
        turns = []
        with torch.inference_mode():
            for segment in activity:
                end = min(segment.end, duration)
                start = segment.start
                while start < end:
                    stop = min(end, start + self.config.max_segment_s)
                    first, last = round(start * rate), round(stop * rate)
                    if last <= first:
                        raise ValueError("speaker interval is shorter than an audio sample")
                    piece = waveform[first:last]
                    # The feature extractor uses reflected STFT padding, which fails on
                    # very short turns. Supply at least one analysis window, retaining
                    # all recorded samples and the original diarization boundaries.
                    window_samples = self.model.processor.feature_extractor.n_fft
                    if len(piece) < window_samples:
                        piece = np.pad(piece, (0, window_samples - len(piece)))
                    outputs = self.model.transcribe(
                        audio=(piece, rate),
                        language=self.config.language,
                        return_time_stamps=False,
                    )
                    if len(outputs) != 1:
                        raise RuntimeError("Qwen returned an unexpected number of transcripts")
                    text = outputs[0].text.strip()
                    # Detect obvious output-limit exhaustion instead of silently accepting truncation.
                    if len(self.model.processor.tokenizer.encode(text)) >= self.config.max_new_tokens:
                        raise RuntimeError("Qwen reached max_new_tokens; increase the limit or shorten segments")
                    turns.append(Segment(start=start, end=stop, speaker=segment.speaker, text=text))
                    start = stop
        return Transcript(
            text=" ".join(t.text for t in turns if t.text),
            tokens=[],
            segments=turns,
            timestamp_method="diarization_turn",
        )


class NemoDiarizer:
    def __init__(self, config: DiarizationConfig, checkpoint: str, device: str):
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model = SortformerEncLabelModel.restore_from(checkpoint, map_location=device).eval()
        for name in ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_update_period", "spkcache_len"):
            setattr(self.model.sortformer_modules, name, getattr(config, name))

    def detect(self, audio: Path) -> list[Segment]:
        import soundfile as sf

        waveform, rate = sf.read(str(audio), dtype="float32")
        output = self.model.diarize(audio=[waveform], sample_rate=rate, batch_size=1, verbose=False)
        segments = []
        for line in output[0]:
            start, end, speaker = line.split()
            segments.append(Segment(start=float(start), end=float(end), speaker=speaker))
        return sorted(segments, key=lambda s: (s.start, s.end, s.speaker))


def resolve_checkpoint(config: Checkpoint | QwenASRConfig, offline: bool) -> str:
    from huggingface_hub import hf_hub_download, snapshot_download

    if isinstance(config, QwenASRConfig):
        return snapshot_download(
            config.repo,
            revision=config.revision,
            local_files_only=offline,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
    return hf_hub_download(config.repo, config.filename, revision=config.revision, local_files_only=offline)


def create_asr(config: ASRConfig | QwenASRConfig, path: str, device: str) -> ASRAdapter:
    if isinstance(config, QwenASRConfig):
        return QwenASR(config, path, device)
    if config.kind == "nemo_streaming":
        return NemoStreamingASR(config, path, device)
    raise ValueError(f"unsupported ASR adapter: {config.kind}")


def create_diarizer(config: DiarizationConfig, path: str, device: str) -> DiarizationAdapter:
    if config.kind == "nemo_sortformer":
        return NemoDiarizer(config, path, device)
    raise ValueError(f"unsupported diarization adapter: {config.kind}")
