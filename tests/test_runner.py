import json
from pathlib import Path
from unittest.mock import patch

import pytest

from asr_bench.runner import run_benchmark
from asr_bench.schema import Segment, Token, Transcript


class Tokenizer:
    def ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def ids_to_text(self, ids):
        return "ก"


class FakeASR:
    def __init__(self, native_turns=False):
        self.native_turns = native_turns
        self.tokenizer = None if native_turns else Tokenizer()

    def transcribe(self, audio, activity=None):
        if self.native_turns:
            assert activity is not None
            return Transcript(
                text="ก",
                tokens=[],
                segments=[Segment(start=0, end=1, speaker="speaker_0", text="ก")],
                timestamp_method="diarization_turn",
            )
        return Transcript(text="ก", tokens=[Token(id=2, time=0.5)])


class FakeDiarizer:
    def detect(self, audio):
        return [Segment(start=0, end=1, speaker="speaker_0")]


@pytest.mark.parametrize("native_turns", [False, True])
def test_complete_pipeline_with_references_and_no_overwrite(tmp_path, native_turns):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake audio")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "clip",
                "audio": "clip.wav",
                "reference_text": "ก",
                "reference_segments": [{"start": 0, "end": 1, "speaker": "person"}],
            }
        )
    )
    config = Path(__file__).parents[1] / "models.toml"
    output = tmp_path / "results"
    with (
        patch("asr_bench.runner.resolve_checkpoint", return_value="cached.nemo"),
        patch("asr_bench.runner.prepare_audio", return_value=1.0),
        patch("asr_bench.runner.create_asr", return_value=FakeASR(native_turns)),
        patch("asr_bench.runner.create_diarizer", return_value=FakeDiarizer()) as diarizer,
    ):
        summary = run_benchmark(manifest, config, ["openthai"], output, "cpu", 2, 0, True, 1)
        diarizer.assert_called_once()
        assert summary[0]["measurements"] == 2
        assert summary[0]["cer"] == summary[0]["der"] == 0
        result = json.loads((output / "openthai--clip--0.json").read_text())
        assert result["segments"][0]["speaker"] == "speaker_0"
        assert result["detected_speakers"] == 1
        for repeat in range(2):
            readable = (output / f"openthai--clip--{repeat}.txt").read_text(encoding="utf-8")
            expected = "[00:00:00.000 - 00:00:01.000]" if native_turns else "[00:00:00.500 - 00:00:00.580]"
            assert expected + " speaker_0" in readable
            assert "ก" in readable
            assert "Full transcription" in readable
        with pytest.raises(FileExistsError):
            run_benchmark(manifest, config, ["openthai"], output, "cpu", 1, 0, True, 1)
