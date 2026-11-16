from asr_bench.reporting import render_transcript


def result(segments):
    return {
        "model": "test",
        "sample": "clip",
        "audio": "clip.m4a",
        "detected_speakers": 2,
        "segments": segments,
        "text": "สวัสดี ได้เลย",
    }


def test_readable_timeline_keeps_speakers_unicode_and_hour_timestamps():
    transcript = render_transcript(
        result(
            [
                {"start": 2.32, "end": 18.96, "speaker": "speaker_0", "text": "สวัสดี"},
                {"start": 3661.9996, "end": 3663.0, "speaker": "speaker_1", "text": "ได้เลย"},
            ]
        )
    )
    assert "[00:00:02.320 - 00:00:18.960] speaker_0\nสวัสดี" in transcript
    assert "[01:01:02.000 - 01:01:03.000] speaker_1\nได้เลย" in transcript
    assert "Full transcription\n\nสวัสดี ได้เลย" in transcript


def test_no_turns_preserves_full_text_and_warnings():
    data = result([])
    data["warnings"] = ["No speech activity detected"]
    transcript = render_transcript(data)
    assert "No speaker-labeled turns available." in transcript
    assert "Warning: No speech activity detected" in transcript
    assert "สวัสดี ได้เลย" in transcript
