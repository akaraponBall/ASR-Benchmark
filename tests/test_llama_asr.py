import base64
import io
import json
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from asr_bench.llama_asr import LlamaASR
from asr_bench.runner import run_benchmark
from asr_bench.schema import DiarizationConfig, LlamaASRConfig, Segment

MODEL_ID = "LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M"


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as file:
        file.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        file.writeframes(b"\x01\x00" * 16000 + b"\x02\x00" * 16000)
    return path


@pytest.fixture
def server(monkeypatch):
    connection = Mock()
    requests = []
    server = {"finish_reason": "stop", "requests": requests}

    def respond():
        method, path, *body = connection.request.call_args.args
        if method == "GET":
            assert path == "/v1/models"
            result = {"data": [{"id": "unrelated"}, {"id": MODEL_ID}]}
        else:
            assert method == "POST" and path == "/v1/chat/completions"
            requests.append(json.loads(body[0]))
            result = {"choices": [{"message": {"content": " สวัสดี "}, "finish_reason": server["finish_reason"]}]}
        return Mock(status=200, read=Mock(return_value=json.dumps(result).encode()))

    connection.getresponse.side_effect = respond
    factory = Mock(return_value=connection)
    monkeypatch.setattr("asr_bench.llama_asr.http.client.HTTPConnection", factory)
    yield server
    factory.assert_called_with("localhost", 9931, timeout=120)
    assert connection.close.call_count == len(requests) + 1


def test_sends_wav_chunks_and_preserves_speaker_times(audio, server):
    adapter = LlamaASR(LlamaASRConfig(kind="llama_audio", repo="LiquidAI/LFM3-Omni-2B", max_segment_s=0.5))
    result = adapter.transcribe(audio, [Segment(start=0.75, end=1.5, speaker="speaker_2")])

    assert result.timestamp_method == "diarization_turn"
    assert result.tokens == []
    assert result.text == "สวัสดี สวัสดี"
    assert [(turn.start, turn.end, turn.speaker) for turn in result.segments] == [
        (0.75, 1.25, "speaker_2"),
        (1.25, 1.5, "speaker_2"),
    ]
    for request, expected in zip(
        server["requests"], [b"\x01\x00" * 4000 + b"\x02\x00" * 4000, b"\x02\x00" * 4000], strict=True
    ):
        assert request["model"] == MODEL_ID
        system, user = request["messages"]
        assert system["content"] == "Perform ASR in Thai. Respond in text."
        part = user["content"][0]
        assert part["type"] == "input_audio"
        with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"]))) as chunk:
            assert (chunk.getframerate(), chunk.getnchannels(), chunk.getsampwidth()) == (16000, 1, 2)
            assert chunk.readframes(chunk.getnframes()) == expected


def test_empty_activity_preserves_text_and_incomplete_responses_fail(audio, server):
    adapter = LlamaASR(LlamaASRConfig(kind="llama_audio", repo="LiquidAI/LFM3-Omni-2B", model_id=MODEL_ID))
    with pytest.raises(ValueError, match="requires speaker detection"):
        adapter.transcribe(audio)
    result = adapter.transcribe(audio, [])
    assert result.segments == [Segment(start=0, end=2, speaker="unknown", text="สวัสดี")]
    server["finish_reason"] = "length"
    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        adapter.transcribe(audio, [])


def test_benchmark_lfm3_resolves_only_diarizer_and_records_server_id(audio, server, monkeypatch):
    manifest = audio.with_suffix(".jsonl")
    manifest.write_text(json.dumps({"id": "clip", "audio": audio.name}), encoding="utf-8")
    config = Path(__file__).parents[1] / "models.toml"
    out = audio.parent / "results"
    resolve = Mock(return_value="cached.nemo")
    diarizer = Mock()
    diarizer.detect.return_value = [Segment(start=0, end=1, speaker="speaker_0")]
    monkeypatch.setattr("asr_bench.runner.resolve_checkpoint", resolve)
    monkeypatch.setattr("asr_bench.runner.create_diarizer", Mock(return_value=diarizer))

    run_benchmark(manifest, config, ["lfm3"], out, "cpu", 1, 0, True, 1)

    resolve.assert_called_once()
    checkpoint, offline = resolve.call_args.args
    assert isinstance(checkpoint, DiarizationConfig) and offline
    diarizer.detect.assert_called_once()
    assert len(server["requests"]) == 1
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["served_model_ids"] == {"lfm3": MODEL_ID}
    assert "speaker_0\nสวัสดี" in (out / "lfm3--clip--0.txt").read_text(encoding="utf-8")


def test_wsl_connection_refusal_uses_windows_loopback(audio):
    responses = [
        Mock(stdout=json.dumps({"data": [{"id": MODEL_ID}]}).encode()),
        Mock(stdout=b'{"choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}]}'),
    ]
    with (
        patch("asr_bench.llama_asr.http.client.HTTPConnection") as connection,
        patch("asr_bench.llama_asr.platform.release", return_value="microsoft-standard-WSL2"),
        patch("asr_bench.llama_asr.shutil.which", return_value="/windows/curl.exe") as which,
        patch("asr_bench.llama_asr.subprocess.run", side_effect=responses) as curl,
    ):
        connection.return_value.request.side_effect = ConnectionRefusedError()
        adapter = LlamaASR(LlamaASRConfig(kind="llama_audio", repo="LiquidAI/LFM3-Omni-2B"))
        result = adapter.transcribe(audio, [])

    assert result.text == "Hello"
    connection.assert_called_once()
    which.assert_called_once_with("curl.exe")
    get, post = curl.call_args_list
    assert "http://127.0.0.1:9931/v1/models" in get.args[0]
    assert "http://127.0.0.1:9931/v1/chat/completions" in post.args[0]
    assert get.kwargs["input"] is None
    assert post.args[0][-2:] == ["--data-binary", "@-"]
    assert json.loads(post.kwargs["input"])["model"] == MODEL_ID
