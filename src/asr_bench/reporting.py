"""Readable transcript exports from saved benchmark results; no inference required."""


def render_transcript(result: dict) -> str:
    lines = [
        f"Model: {result['model']}",
        f"Recording: {result['sample']}",
        f"Audio: {result['audio']}",
        f"Detected speakers: {result['detected_speakers']}",
        "",
        (
            "Speaker timeline (detected speaker-turn boundaries)"
            if result.get("timestamp_method") == "diarization_turn"
            else "Speaker timeline (approximate emission times)"
        ),
        "",
    ]
    for segment in result["segments"]:
        lines.extend(
            [
                f"[{_timestamp(segment['start'])} - {_timestamp(segment['end'])}] {segment['speaker']}",
                segment["text"],
                "",
            ]
        )
    if not result["segments"]:
        lines.extend(["No speaker-labeled turns available.", ""])
    for warning in result.get("warnings", []):
        lines.extend([f"Warning: {warning}", ""])
    lines.extend(["Full transcription", "", result["text"], ""])
    return "\n".join(lines)


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"
