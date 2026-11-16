"""Assign token emission times to anonymous speakers; this is not source separation."""

import re

from .schema import Segment, Transcript


class SpeakerAttributor:
    def __init__(self, tolerance: float = 0.2, nearest_limit: float = 1.0, pause: float = 1.0):
        self.tolerance = tolerance
        self.nearest_limit = nearest_limit
        self.pause = pause

    def assign(self, transcript: Transcript, tokenizer, activity: list[Segment], duration: float) -> list[Segment]:
        turns = []
        current = None
        previous = "unknown"
        for token in transcript.tokens:
            piece = tokenizer.ids_to_tokens([token.id])[0]
            if token.id == 0 or re.fullmatch(r"<[a-z]{2,3}(-[A-Za-z]{2,4})?>", piece):
                continue
            speaker = self._speaker_at(token.time, activity, previous)
            previous = speaker
            if current is None or current["speaker"] != speaker or token.time - current["end"] > self.pause:
                current = {"speaker": speaker, "start": token.time, "end": token.time, "ids": []}
                turns.append(current)
            current["ids"].append(token.id)
            current["end"] = token.time
        result = []
        for turn in turns:
            text = tokenizer.ids_to_text(turn["ids"]).strip()
            if text:
                # Latency can put token emission just beyond the waveform. Clamp for display only.
                start = min(turn["start"], max(0.0, duration - 0.001))
                end = min(duration, max(start + 0.001, turn["end"] + 0.08))
                result.append(Segment(start=start, end=end, speaker=turn["speaker"], text=text))
        return result

    def _speaker_at(self, time, activity, previous):
        covering = [s for s in activity if s.start - self.tolerance <= time <= s.end + self.tolerance]
        if covering:
            return max(covering, key=lambda s: s.end - s.start).speaker
        nearest = min(activity, key=lambda s: min(abs(time - s.start), abs(time - s.end)), default=None)
        if nearest and min(abs(time - nearest.start), abs(time - nearest.end)) <= self.nearest_limit:
            return nearest.speaker
        return previous
