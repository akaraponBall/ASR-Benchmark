import json

import pytest
from pydantic import ValidationError

from asr_bench.alignment import SpeakerAttributor
from asr_bench.metrics import diarization_scores, text_scores
from asr_bench.schema import Segment, Token, Transcript, load_manifest


class Tokenizer:
    def ids_to_tokens(self, ids):
        return ["<th-TH>" if i == 1 else str(i) for i in ids]

    def ids_to_text(self, ids):
        return " ".join(map(str, ids))


def seg(start, end, speaker):
    return Segment(start=start, end=end, speaker=speaker)


def test_text_scores_and_empty_reference():
    score = text_scores("ก ข", "ก ค")
    assert score["cer"]["rate"] == 0.5
    assert score["cer"]["substitutions"] == 1
    assert text_scores("", "ก")["cer"]["rate"] is None
    assert text_scores("", "ก")["cer"]["insertions"] == 1


def test_der_permutation_and_overlap():
    reference = [seg(0, 2, "a"), seg(1, 3, "b")]
    hypothesis = [seg(0, 2, "speaker_1"), seg(1, 3, "speaker_0")]
    assert diarization_scores(reference, hypothesis, 3)["rate"] == 0


def test_der_miss_false_alarm_and_confusion():
    reference = [seg(0, 2, "a"), seg(2, 4, "b")]
    hypothesis = [seg(0, 4, "x"), seg(4, 5, "y")]
    result = diarization_scores(reference, hypothesis, 5)
    assert result["confusion_s"] == 2
    assert result["false_alarm_s"] == 1
    assert result["rate"] == 0.75
    assert diarization_scores(reference, [], 5)["rate"] == 1
    assert diarization_scores([], hypothesis, 5)["rate"] is None


def test_der_duplicate_same_speaker_intervals_count_once():
    result = diarization_scores([seg(0, 2, "a"), seg(1, 3, "a")], [seg(0, 3, "x")], 3)
    assert result["reference_speaker_s"] == 3
    assert result["rate"] == 0


def test_attribution_preserves_tokens_and_unknown_speech():
    transcript = Transcript(text="2 3 4", tokens=[Token(id=i, time=t) for i, t in [(1, 0), (2, 0.5), (3, 2), (4, 4)]])
    turns = SpeakerAttributor().assign(transcript, Tokenizer(), [seg(0, 1, "a"), seg(1.5, 3, "b")], 5)
    assert [t.text for t in turns] == ["2", "3", "4"]
    assert [t.speaker for t in turns] == ["a", "b", "b"]
    unknown = SpeakerAttributor().assign(transcript, Tokenizer(), [], 5)
    assert all(t.speaker == "unknown" for t in unknown)
    assert " ".join(t.text for t in unknown) == "2 3 4"


def test_schema_rejects_invalid_intervals_and_unknown_fields():
    with pytest.raises(ValidationError):
        seg(2, 1, "a")
    with pytest.raises(ValidationError):
        Segment(start=0, end=float("nan"), speaker="a")
    with pytest.raises(ValidationError):
        Segment(start=0, end=1, speaker="a", ignored=True)


def test_manifest_relative_paths_duplicates_and_traversal_ids(tmp_path):
    (tmp_path / "clip.wav").touch()
    manifest = tmp_path / "manifest.jsonl"
    record = {"id": "clip", "audio": "clip.wav"}
    manifest.write_text(json.dumps(record) + "\n")
    assert load_manifest(manifest)[0].audio == tmp_path / "clip.wav"
    manifest.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(manifest)
    record["id"] = "../unsafe"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_manifest(manifest)
