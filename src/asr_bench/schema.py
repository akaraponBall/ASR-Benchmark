"""Validated configuration and interchange types; no model imports."""

import math
import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Checkpoint(StrictModel):
    repo: str
    filename: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")


class ASRConfig(Checkpoint):
    kind: Literal["nemo_streaming"] = "nemo_streaming"
    language: str = "th-TH"
    latency_ms: Literal[80, 320, 560, 1120] = 1120
    max_symbols_per_step: int = Field(default=2, ge=1)


class QwenASRConfig(StrictModel):
    kind: Literal["qwen_asr"]
    repo: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    language: str = "Thai"
    max_new_tokens: int = Field(default=2048, ge=1)
    max_segment_s: float = Field(default=25, gt=0, le=30)


class LlamaASRConfig(StrictModel):
    kind: Literal["llama_audio"]
    repo: str = Field(min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    port: int = Field(default=9931, ge=1, le=65535)
    language: str = Field(default="Thai", min_length=1)
    max_segment_s: float = Field(default=25, ge=0.001, le=30)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_s: float = Field(default=120, gt=0)


ASRSettings = Annotated[ASRConfig | QwenASRConfig | LlamaASRConfig, Field(discriminator="kind")]


class DiarizationConfig(Checkpoint):
    kind: Literal["nemo_sortformer"] = "nemo_sortformer"
    chunk_len: int = Field(default=340, gt=0)
    chunk_right_context: int = Field(default=40, ge=0)
    fifo_len: int = Field(default=40, ge=0)
    spkcache_update_period: int = Field(default=300, gt=0)
    spkcache_len: int = Field(default=264, gt=0)


class Settings(StrictModel):
    asr: dict[str, ASRSettings]
    diarization: DiarizationConfig

    @model_validator(mode="after")
    def safe_aliases(self):
        if not self.asr:
            raise ValueError("at least one ASR model must be configured")
        for alias in self.asr:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", alias):
                raise ValueError(f"unsafe model alias: {alias!r}")
        return self

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))


class Segment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker: str = Field(min_length=1)
    text: str = ""

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        return self


class Token(StrictModel):
    id: int
    time: float = Field(ge=0)


class Transcript(StrictModel):
    text: str
    tokens: list[Token]
    segments: list[Segment] | None = None
    timestamp_method: Literal["token_emission", "diarization_turn"] = "token_emission"


class Sample(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    audio: Path
    reference_text: str | None = None
    reference_segments: list[Segment] | None = None


def load_manifest(path: Path) -> list[Sample]:
    samples = []
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = Sample.model_validate_json(line)
            sample.audio = (path.resolve().parent / sample.audio).resolve(strict=True)
            if sample.id in ids:
                raise ValueError(f"duplicate sample ID: {sample.id}")
            ids.add(sample.id)
            samples.append(sample)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError("manifest is empty")
    return samples


def validate_timeline(segments: list[Segment], duration: float):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("audio duration must be finite and positive")
    for segment in segments:
        if segment.end > duration + 0.001:
            raise ValueError(f"segment ending at {segment.end} exceeds audio duration {duration}")
