"""Normalize audio at the boundary; model adapters only receive mono 16 kHz WAV."""

import hashlib
import subprocess
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf


def prepare_audio(source: Path, destination: Path) -> float:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )
    info = sf.info(str(destination))
    if info.frames == 0 or info.samplerate != 16000 or info.channels != 1:
        raise ValueError("audio conversion did not produce nonempty mono 16 kHz audio")
    return info.duration


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
