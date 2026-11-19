"""Transcribe speaker intervals through the loopback Llama.app API."""

import base64
import http.client
import io
import json
import platform
import shutil
import subprocess
import wave
from pathlib import Path

from .schema import LlamaASRConfig, Segment, Transcript


class LlamaASR:
    tokenizer = None

    def __init__(self, config: LlamaASRConfig):
        self.config = config
        self._windows_curl = None
        try:
            models = self._request("/v1/models")
        except RuntimeError as exc:
            if not isinstance(exc.__cause__, ConnectionRefusedError) or "microsoft" not in platform.release().lower():
                raise
            self._windows_curl = shutil.which("curl.exe")
            if self._windows_curl is None:
                raise
            models = self._request("/v1/models")
        try:
            entries = models["data"]
            if not isinstance(entries, list):
                raise TypeError("expected a model list")
            ids = [entry["id"] for entry in entries]
            if not all(isinstance(model_id, str) and model_id for model_id in ids):
                raise TypeError("expected nonempty model IDs")
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Invalid /v1/models response from Llama.app") from exc
        if config.model_id is not None:
            matches = [model_id for model_id in ids if model_id == config.model_id]
        else:
            name = config.repo.rsplit("/", 1)[-1].casefold()
            matches = [model_id for model_id in ids if name in model_id.rsplit("/", 1)[-1].casefold()]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Llama.app model for {config.repo}. Set model_id from /v1/models: {ids}")
        self.model_id = matches[0]

    def _request(self, path: str, payload: dict | None = None) -> dict:
        if self._windows_curl is not None:
            return self._request_windows(path, payload)
        connection = http.client.HTTPConnection("localhost", self.config.port, timeout=self.config.timeout_s)
        try:
            if payload is None:
                connection.request("GET", path)
            else:
                connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(
                    f"Llama.app {path} returned HTTP {response.status} {response.reason}. "
                    "Check that the selected model is loaded with audio input enabled."
                )
            data = json.loads(response.read())
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise RuntimeError(
                f"Llama.app request failed at http://localhost:{self.config.port}{path}: {exc}"
            ) from exc
        finally:
            connection.close()
        return data

    def _request_windows(self, path: str, payload: dict | None) -> dict:
        url = f"http://127.0.0.1:{self.config.port}{path}"
        command = [self._windows_curl, "-q", "--noproxy", "*", "--silent", "--show-error", "--fail", url]
        body = None
        if payload is not None:
            command += ["--header", "Content-Type: application/json", "--data-binary", "@-"]
            body = json.dumps(payload).encode("utf-8")
        try:
            response = subprocess.run(
                command, input=body, capture_output=True, timeout=self.config.timeout_s, check=True
            )
            return json.loads(response.stdout)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Llama.app request via Windows failed at {url}: {error}") from exc
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise RuntimeError(f"Llama.app request via Windows failed at {url}: {exc}") from exc

    def _transcribe_piece(self, wav: bytes) -> str:
        result = self._request(
            "/v1/chat/completions",
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": f"Perform ASR in {self.config.language}. Respond in text."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"},
                            }
                        ],
                    },
                ],
                "temperature": 0,
                "reasoning_effort": "none",
                "max_tokens": self.config.max_tokens,
                "stream": False,
            },
        )
        try:
            choice = result["choices"][0]
            finish_reason = choice["finish_reason"]
            text = choice["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("Invalid chat completion from Llama.app") from exc
        if finish_reason != "stop":
            raise RuntimeError(f"Llama.app did not finish transcription (finish_reason={finish_reason!r})")
        if not isinstance(text, str):
            raise TypeError("Llama.app did not return text for transcription")
        return text.strip()

    def transcribe(self, audio: Path, activity: list[Segment] | None = None) -> Transcript:
        if activity is None:
            raise ValueError("Llama ASR requires speaker detection before transcription")
        turns = []
        with wave.open(str(audio), "rb") as source:
            rate = source.getframerate()
            frames = source.getnframes()
            chunk_frames = max(1, int(self.config.max_segment_s * rate))
            intervals = activity or [Segment(start=0, end=frames / rate, speaker="unknown")]
            for segment in intervals:
                start = round(segment.start * rate)
                end = min(round(segment.end * rate), frames)
                if end <= start:
                    raise ValueError("speaker interval contains no audio samples")
                source.setpos(start)
                for first in range(start, end, chunk_frames):
                    last = min(first + chunk_frames, end)
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as chunk:
                        chunk.setnchannels(source.getnchannels())
                        chunk.setsampwidth(source.getsampwidth())
                        chunk.setframerate(rate)
                        chunk.writeframes(source.readframes(last - first))
                    text = self._transcribe_piece(buffer.getvalue())
                    turns.append(Segment(start=first / rate, end=last / rate, speaker=segment.speaker, text=text))
        turns.sort(key=lambda turn: (turn.start, turn.end, turn.speaker))
        return Transcript(
            text=" ".join(turn.text for turn in turns if turn.text),
            tokens=[],
            segments=turns,
            timestamp_method="diarization_turn",
        )
