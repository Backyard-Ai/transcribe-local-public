"""Finds audio that sounds like something but has no transcript.

This is independent of the transcription models, so it still catches speech
they drop. It reports passages to check; it can't tell speech from other sounds.
"""

from pathlib import Path

import numpy as np
import soundfile

WINDOW_SECONDS = 0.5
# Louder than the recording's own background by this much counts as "something".
ABOVE_BACKGROUND_DB = 10.0
# Quieter than this is treated as silence, even in a very clean recording.
MINIMUM_LEVEL_DB = -55.0
MIN_GAP_SECONDS = 1.0
BACKGROUND_PERCENTILE = 10


def _window_levels(wav_path: Path) -> list[float]:
    """Level of each window in dB, read in blocks so length doesn't matter."""
    levels = []
    with soundfile.SoundFile(wav_path) as file:
        size = int(WINDOW_SECONDS * file.samplerate)
        for block in file.blocks(blocksize=size, dtype="float32"):
            rms = float(np.sqrt(np.mean(block**2)))
            levels.append(20 * np.log10(rms + 1e-10))
    return levels


def _transcribed_spans(segments: list[dict]) -> list[tuple[float, float]]:
    spans = []
    for segment in segments:
        words = segment.get("words") or []
        if words:
            spans.extend((word["start"], word["end"]) for word in words)
        else:
            spans.append((segment["start"], segment["end"]))
    return spans


def find_untranscribed_audio(wav_path: Path, segments: list[dict]) -> list[tuple[float, float]]:
    """Return (start, end) passages louder than the background with no words."""
    levels = _window_levels(wav_path)
    if not levels:
        return []
    threshold = max(np.percentile(levels, BACKGROUND_PERCENTILE) + ABOVE_BACKGROUND_DB, MINIMUM_LEVEL_DB)

    covered = [False] * len(levels)
    for start, end in _transcribed_spans(segments):
        first = int(start / WINDOW_SECONDS)
        last = int(end / WINDOW_SECONDS)
        for index in range(first, min(last + 1, len(levels))):
            covered[index] = True

    gaps = []
    gap_start = None
    for index, level in enumerate(levels + [threshold - 1]):
        is_gap = index < len(levels) and level >= threshold and not covered[index]
        if is_gap and gap_start is None:
            gap_start = index
        elif not is_gap and gap_start is not None:
            start, end = gap_start * WINDOW_SECONDS, index * WINDOW_SECONDS
            if end - start >= MIN_GAP_SECONDS:
                gaps.append((start, end))
            gap_start = None
    return gaps
