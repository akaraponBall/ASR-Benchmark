"""Orchestrate adapters; persist raw outputs and scores separately from model logic."""

import csv
import gc
import importlib.metadata
import json
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from .adapters import create_asr, create_diarizer, resolve_checkpoint
from .alignment import SpeakerAttributor
from .audio import file_hash, prepare_audio
from .measurement import Measurement
from .metrics import NORMALIZATION, diarization_scores, text_scores
from .reporting import render_transcript
from .schema import Settings, load_manifest, validate_timeline


def run_benchmark(
    manifest: Path,
    config_path: Path,
    names: list[str],
    output: Path,
    device: str,
    repeats: int,
    warmup: int,
    offline: bool,
    threads: int,
):
    import torch

    settings = Settings.load(config_path)
    samples = load_manifest(manifest)
    if not names or len(set(names)) != len(names):
        raise ValueError("provide at least one model alias, without duplicates")
    for name in names:
        if name not in settings.asr:
            raise ValueError(f"unknown model alias {name!r}; available: {', '.join(settings.asr)}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier benchmark.
    packages = {}
    for package in (
        "asr-bench",
        "nemo-toolkit",
        "torch",
        "jiwer",
        "pythainlp",
        "huggingface-hub",
        "qwen-asr",
        "transformers",
    ):
        packages[package] = importlib.metadata.version(package)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cpu": platform.processor(),
        "threads": threads,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "packages": packages,
        "config": settings.model_dump(),
        "models": names,
        "manifest_sha256": file_hash(manifest),
        "warmup": warmup,
        "repeats": repeats,
        "normalization": NORMALIZATION,
        "offline": offline,
        "source_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    _write_json(output / "run.json", metadata)
    rows = []
    try:
        for name in names:
            print(f"Loading {name} + speaker detector...", flush=True)
            download_start = time.perf_counter()
            asr_path = resolve_checkpoint(settings.asr[name], offline)
            diar_path = resolve_checkpoint(settings.diarization, offline)
            download_s = time.perf_counter() - download_start
            load_start = time.perf_counter()
            asr = create_asr(settings.asr[name], asr_path, device)
            diarizer = create_diarizer(settings.diarization, diar_path, device)
            if device == "cuda":
                torch.cuda.synchronize()
            load_s = time.perf_counter() - load_start
            for sample in samples:
                with tempfile.TemporaryDirectory(prefix="asr-bench-") as directory:
                    audio = Path(directory) / "audio.wav"
                    conversion_start = time.perf_counter()
                    duration = prepare_audio(sample.audio, audio)
                    conversion_s = time.perf_counter() - conversion_start
                    validate_timeline(sample.reference_segments or [], duration)
                    audio_hash = file_hash(sample.audio)
                    for _ in range(warmup):
                        warm_activity = diarizer.detect(audio)
                        asr.transcribe(audio, warm_activity)
                    for repeat in range(repeats):
                        with Measurement(device) as measurement:
                            with measurement.phase("diarization"):
                                activity = diarizer.detect(audio)
                            with measurement.phase("asr"):
                                transcript = asr.transcribe(audio, activity)
                            with measurement.phase("attribution"):
                                turns = (
                                    transcript.segments
                                    if transcript.segments is not None
                                    else (SpeakerAttributor().assign(transcript, asr.tokenizer, activity, duration))
                                )
                        # Preserve raw speaker activity separately from attributed transcript turns.
                        scores = {}
                        if sample.reference_text is not None:
                            scores.update(text_scores(sample.reference_text, transcript.text))
                        if sample.reference_segments is not None:
                            scores["der"] = diarization_scores(sample.reference_segments, activity, duration)
                        processing_s = sum(measurement.timings.values())
                        row = {
                            "model": name,
                            "sample": sample.id,
                            "repeat": repeat,
                            "duration_s": duration,
                            "processing_s": processing_s,
                            "rtf": processing_s / duration,
                            "audio_conversion_s": conversion_s,
                            "checkpoint_resolve_s": download_s,
                            "model_load_s": load_s,
                            "measurement": measurement.report(),
                            "scores": scores,
                            "detected_speakers": len({s.speaker for s in activity}),
                            "unknown_turns": sum(s.speaker == "unknown" for s in turns),
                            "warnings": ["ASR produced text but diarizer detected no speech"]
                            if transcript.text and not activity
                            else [],
                        }
                        result = {
                            **row,
                            "audio": str(sample.audio),
                            "audio_sha256": audio_hash,
                            "text": transcript.text,
                            "timestamp_method": transcript.timestamp_method,
                            "tokens": [t.model_dump() for t in transcript.tokens],
                            "segments": [s.model_dump() for s in turns],
                            "diarization": [s.model_dump() for s in activity],
                            "reference_text": sample.reference_text,
                            "reference_segments": [s.model_dump() for s in sample.reference_segments]
                            if sample.reference_segments is not None
                            else None,
                        }
                        _write_json(output / f"{name}--{sample.id}--{repeat}.json", result)
                        transcript_path = output / f"{name}--{sample.id}--{repeat}.txt"
                        transcript_path.write_text(render_transcript(result), encoding="utf-8")
                        rows.append(row)
                        print(
                            f"{name}/{sample.id} repeat={repeat}: {processing_s:.2f}s, "
                            f"RTF={row['rtf']:.3f}, speakers={row['detected_speakers']}",
                            flush=True,
                        )
                        for turn in turns:
                            print(f"  [{turn.start:.2f}-{turn.end:.2f}] {turn.speaker}: {turn.text}")
            del asr, diarizer
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {"error": str(exc), "type": type(exc).__name__, "completed_measurements": len(rows)},
        )
        raise
    _write_json(output / "measurements.json", rows)
    summary = summarize(rows)
    _write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def summarize(rows):
    result = []
    for name in dict.fromkeys(row["model"] for row in rows):
        group = [r for r in rows if r["model"] == name]
        duration = sum(r["duration_s"] for r in group)
        summary = {
            "model": name,
            "measurements": len(group),
            "audio_s_including_repeats": duration,
            "processing_s": sum(r["processing_s"] for r in group),
            "rtf": sum(r["processing_s"] for r in group) / duration,
            "mean_processing_s": sum(r["processing_s"] for r in group) / len(group),
            "max_process_rss_mib": max(r["measurement"]["peak_process_rss_mib"] for r in group),
        }
        for metric, denominator in (
            ("cer", "reference_units"),
            ("wer", "reference_units"),
            ("der", "reference_speaker_s"),
        ):
            scores = [r["scores"][metric] for r in group if metric in r["scores"]]
            units = sum(s[denominator] for s in scores)
            errors = sum(
                s["errors"] if metric != "der" else s["missed_s"] + s["false_alarm_s"] + s["confusion_s"]
                for s in scores
            )
            summary[metric] = errors / units if units else None
            summary[f"{metric}_scored_measurements"] = len(scores)
        result.append(summary)
    return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
