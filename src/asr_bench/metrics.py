"""Deterministic reference-based scores. Missing references produce no invented score."""

import itertools
import unicodedata

import jiwer
import numpy as np
from pythainlp.tokenize import word_tokenize
from scipy.optimize import linear_sum_assignment

from .schema import Segment, validate_timeline

NORMALIZATION = "NFC; case/punctuation/numerals preserved; CER removes whitespace; WER uses newmm"


def text_scores(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = (unicodedata.normalize("NFC", t) for t in (reference, hypothesis))
    chars = jiwer.process_characters("".join(reference.split()), "".join(hypothesis.split()))
    words = jiwer.process_words(
        " ".join(word_tokenize(reference, engine="newmm", keep_whitespace=False)),
        " ".join(word_tokenize(hypothesis, engine="newmm", keep_whitespace=False)),
    )
    result = {}
    for name, score in (("cer", chars), ("wer", words)):
        errors = score.substitutions + score.deletions + score.insertions
        units = score.hits + score.substitutions + score.deletions
        result[name] = {
            "rate": errors / units if units else None,
            "errors": errors,
            "reference_units": units,
            "insertions": score.insertions,
            "deletions": score.deletions,
            "substitutions": score.substitutions,
        }
    return result


def diarization_scores(reference: list[Segment], hypothesis: list[Segment], duration: float) -> dict:
    """DER with zero collar, overlap included, and optimal global speaker permutation.

    Evaluate the full file. Integrate over exact segment boundaries rather than
    quantized frames. Repeated overlapping intervals for one speaker count once.
    """
    validate_timeline(reference, duration)
    validate_timeline(hypothesis, duration)
    ref_names = sorted({s.speaker for s in reference})
    hyp_names = sorted({s.speaker for s in hypothesis})
    boundaries = sorted(
        {0.0, duration, *(s.start for s in reference + hypothesis), *(s.end for s in reference + hypothesis)}
    )
    spans = []
    overlap = np.zeros((len(ref_names), len(hyp_names)))
    for start, end in itertools.pairwise(boundaries):
        middle = (start + end) / 2
        refs = {s.speaker for s in reference if s.start <= middle < s.end}
        hyps = {s.speaker for s in hypothesis if s.start <= middle < s.end}
        spans.append((end - start, refs, hyps))
        for r in refs:
            for h in hyps:
                overlap[ref_names.index(r), hyp_names.index(h)] += end - start
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    mapping = {hyp_names[h]: ref_names[r] for r, h in zip(rows, cols, strict=True)}
    missed = false_alarm = confusion = reference_time = 0.0
    for seconds, refs, hyps in spans:
        correct = sum(mapping.get(h) in refs for h in hyps)
        reference_time += len(refs) * seconds
        missed += max(0, len(refs) - len(hyps)) * seconds
        false_alarm += max(0, len(hyps) - len(refs)) * seconds
        confusion += (min(len(refs), len(hyps)) - correct) * seconds
    return {
        "rate": (missed + false_alarm + confusion) / reference_time if reference_time else None,
        "missed_s": missed,
        "false_alarm_s": false_alarm,
        "confusion_s": confusion,
        "reference_speaker_s": reference_time,
        "collar_s": 0.0,
        "overlap_included": True,
        "speaker_mapping": mapping,
        "reference_speakers": len(ref_names),
        "predicted_speakers": len(hyp_names),
        "speaker_count_error": len(hyp_names) - len(ref_names),
    }
