from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from asr_bench.adapters import QwenASR, resolve_checkpoint
from asr_bench.schema import QwenASRConfig, Segment, Settings


def config():
    return QwenASRConfig(kind="qwen_asr", repo="Qwen/Qwen3-ASR-0.6B", revision="a" * 40, max_segment_s=2)


def test_qwen_transcribes_detected_turns_without_fabricated_tokens():
    adapter = QwenASR.__new__(QwenASR)
    adapter.config = config()
    adapter.model = Mock()
    adapter.model.processor.feature_extractor.n_fft = 400
    adapter.model.processor.tokenizer.encode.return_value = [1, 2]
    adapter.model.transcribe.return_value = [SimpleNamespace(text="สวัสดี")]
    activity = [Segment(start=1, end=6, speaker="speaker_1")]
    with patch("soundfile.read", return_value=(np.ones(16000 * 7, dtype=np.float32), 16000)):
        transcript = adapter.transcribe("audio.wav", activity)
    assert [(s.start, s.end) for s in transcript.segments] == [(1, 3), (3, 5), (5, 6)]
    assert all(s.speaker == "speaker_1" for s in transcript.segments)
    assert transcript.timestamp_method == "diarization_turn"
    assert transcript.tokens == []
    assert adapter.model.transcribe.call_count == 3
    assert adapter.model.transcribe.call_args.kwargs["return_time_stamps"] is False
    with pytest.raises(ValueError, match="requires speaker detection"):
        adapter.transcribe("audio.wav")


@pytest.mark.parametrize("samples", [1, 160, 200, 201, 399, 400, 401, 16000])
def test_qwen_short_turns_retain_audio_and_boundaries(samples):
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor()
    adapter = QwenASR.__new__(QwenASR)
    adapter.config = config()
    adapter.model = Mock()
    adapter.model.processor.feature_extractor = extractor
    adapter.model.processor.tokenizer.encode.return_value = [1]
    waveform = np.linspace(-0.5, 0.5, samples, dtype=np.float32)

    def infer(*, audio, **kwargs):
        piece, rate = audio
        np.testing.assert_array_equal(piece[:samples], waveform)
        np.testing.assert_array_equal(piece[samples:], 0)
        assert len(piece) == max(samples, extractor.n_fft)
        extractor(piece, sampling_rate=rate, padding=True, truncation=False, return_tensors="pt")
        return [SimpleNamespace(text="สวัสดี")]

    adapter.model.transcribe.side_effect = infer
    activity = [Segment(start=0, end=samples / 16000, speaker="speaker_1")]
    with patch("soundfile.read", return_value=(waveform, 16000)):
        transcript = adapter.transcribe("audio.wav", activity)
    assert len(transcript.segments) == 1
    assert transcript.segments[0].start == 0
    assert transcript.segments[0].end == samples / 16000
    assert transcript.segments[0].speaker == "speaker_1"
    assert transcript.tokens == []


def test_qwen_snapshot_resolution_honors_offline_and_revision():
    with patch("huggingface_hub.snapshot_download", return_value="/cache/qwen") as download:
        assert resolve_checkpoint(config(), True) == "/cache/qwen"
        assert download.call_args.kwargs["local_files_only"] is True
        assert download.call_args.kwargs["revision"] == "a" * 40


def test_qwen_mapping_is_a_separate_validated_adapter():
    from pathlib import Path

    settings = Settings.load(Path(__file__).parents[1] / "models.toml")
    assert isinstance(settings.asr["qwen"], QwenASRConfig)
    assert settings.asr["openthai"].kind == "nemo_streaming"
