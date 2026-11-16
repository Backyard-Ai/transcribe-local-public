"""Finds the passages a person should check against the audio."""

# Starting values, to be tuned on longer and harder recordings.
LOW_LOGPROB = -0.8
HIGH_COMPRESSION = 2.4
HIGH_NO_SPEECH = 0.6
LOW_WORD_PROBABILITY = 0.5

SENTENCE_ENDINGS = (".", "?", "!")
SENTENCE_PAUSE_SECONDS = 1.5
# Poor audio can come back with little punctuation, so sentences have a cap too.
SENTENCE_MAX_SECONDS = 20.0


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def split_sentences(segment: dict) -> list[dict]:
    """Break a segment up using its word timings; one segment can run for many sentences."""
    words = segment.get("words") or []
    if not words:
        return [{"start": segment["start"], "end": segment["end"], "text": segment["text"], "probability": None}]

    groups = []
    current = []
    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else None
        if (
            following is None
            or word["word"].strip().endswith(SENTENCE_ENDINGS)
            or following["start"] - word["end"] >= SENTENCE_PAUSE_SECONDS
            or word["end"] - current[0]["start"] >= SENTENCE_MAX_SECONDS
        ):
            groups.append(current)
            current = []
    return [
        {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": "".join(word["word"] for word in group).strip(),
            "probability": sum(word["probability"] for word in group) / len(group),
        }
        for group in groups
    ]


def segment_reasons(segment: dict, previous: dict | None = None) -> list[str]:
    """Reasons the model only reports for a whole segment, so they cover all of its sentences."""
    reasons = []
    if segment["avg_logprob"] < LOW_LOGPROB:
        reasons.append("low confidence")
    if segment["compression_ratio"] > HIGH_COMPRESSION:
        reasons.append("repetitive text")
    if segment["no_speech_prob"] > HIGH_NO_SPEECH:
        reasons.append("may not be speech")
    if previous and segment["text"] and _normalized(segment["text"]) == _normalized(previous["text"]):
        reasons.append("repeats previous segment")
    return reasons


def build_passages(segments: list[dict]) -> list[dict]:
    """Split the transcript into sentences, each with the reasons it needs checking, if any.

    Word confidence is judged per sentence: a segment can run for most of a minute,
    so averaging across it would hide a garbled sentence among clear ones.
    """
    passages = []
    previous = None
    for segment in segments:
        shared_reasons = segment_reasons(segment, previous)
        for sentence in split_sentences(segment):
            if not sentence["text"]:
                continue
            reasons = list(shared_reasons)
            if sentence["probability"] is not None and sentence["probability"] < LOW_WORD_PROBABILITY:
                reasons.append("uncertain words")
            passages.append({"start": sentence["start"], "end": sentence["end"], "text": sentence["text"], "reasons": reasons})
        previous = segment
    return passages
