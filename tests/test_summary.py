import pytest

from asr_bench.runner import summarize


def test_summary_uses_micro_average_and_reports_scored_coverage():
    rows = []
    for errors, units in ((1, 2), (1, 10), (0, 0)):
        scores = {"cer": {"errors": errors, "reference_units": units}} if units else {}
        rows.append(
            {
                "model": "example",
                "duration_s": 5,
                "processing_s": 1,
                "measurement": {"peak_process_rss_mib": 100},
                "scores": scores,
            }
        )
    result = summarize(rows)[0]
    assert result["cer"] == pytest.approx(2 / 12)
    assert result["cer_scored_measurements"] == 2
    assert result["wer"] is None
    assert result["der"] is None
    assert result["rtf"] == 0.2
