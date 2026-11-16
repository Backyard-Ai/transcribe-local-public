"""Scores the app's transcripts against existing ones.

Prints numbers, timestamps and "Recording 1" style labels only, never words or file
names, so the report can be shared without sharing any content. The key linking
labels to files is saved next to the transcripts, on this computer.

Scores count differences between the two transcripts. An existing transcript can have
errors or tidied wording of its own, so a difference isn't always the app's mistake.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transcribe_local.audio import SUPPORTED_EXTENSIONS
from transcribe_local.output import format_time
from transcribe_local.pipeline import DATA_ROOT, DEFAULT_OUTPUT
from transcribe_local.safety import block_network, require_local

DEFAULT_INPUT = DATA_ROOT / "input"
KEY_NAME = "comparison_key.csv"

FILLERS = {"ah", "er", "erm", "hm", "hmm", "mhm", "mm", "mmm", "uh", "uhh", "uhm", "um", "umm"}
# A run of missed or added words this long is worth finding by ear.
LONG_RUN_WORDS = 10
# Counts this far apart usually mean the wrong pair of files, or an unfinished transcript.
SUSPICIOUS_LENGTH_RATIO = 1.5

_TIMESTAMP = re.compile(r"[\[(#]?\b\d{1,2}:\d{2}(?::\d{2})?(?:[.,:\-]\d+)?\b[\])#]?")
_SUBTITLE_ARROW = re.compile(r"-->")
_ROLE_ONLY_LINE = re.compile(
    r"^\s*(speaker|interviewer|interviewee|moderator|participant|respondent|facilitator)?\s*[a-z]?\d*\s*:?\s*$",
    re.IGNORECASE,
)
_SPEAKER_LABEL = re.compile(r"^\s*(?:[^\s:]+\s+){0,2}[^\s:]+\s*:\s+")
_APP_LINE = re.compile(r"^\[(\d+):(\d{2}):(\d{2})\]\s?(.*)$")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def words_in(text: str) -> list[str]:
    normalized = text.lower().replace("’", "'").replace("‘", "'")
    return _WORD.findall(normalized)


def reference_words(text: str) -> list[str]:
    """Words spoken in an existing transcript, without timestamps or speaker labels."""
    words = []
    for line in text.splitlines():
        line = _SUBTITLE_ARROW.sub(" ", _TIMESTAMP.sub(" ", line))
        if _ROLE_ONLY_LINE.match(line):
            continue
        words.extend(words_in(_SPEAKER_LABEL.sub("", line)))
    return words


def app_words(text: str) -> tuple[list[str], list[float]]:
    """Words in one of the app's .txt transcripts, each with the time of its line."""
    words, times = [], []
    for line in text.splitlines():
        match = _APP_LINE.match(line)
        if not match:
            continue
        hours, minutes, seconds, spoken = match.groups()
        start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        found = words_in(spoken)
        words.extend(found)
        times.extend([float(start)] * len(found))
    return words, times


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[str, int]]:
    """Word-by-word alignment as (operation, position in hypothesis). Exact, and fast enough for hours of speech."""
    vocabulary: dict[str, int] = {}
    ref = np.array([vocabulary.setdefault(word, len(vocabulary)) for word in reference], dtype=np.int32)
    hyp = np.array([vocabulary.setdefault(word, len(vocabulary)) for word in hypothesis], dtype=np.int32)
    n, m = len(ref), len(hyp)

    diagonal, deleted, inserted = 0, 1, 2
    moves = np.zeros((n + 1, m + 1), dtype=np.uint8)
    moves[0, 1:] = inserted
    moves[1:, 0] = deleted
    positions = np.arange(m + 1, dtype=np.int32)
    previous = positions.copy()
    for i in range(1, n + 1):
        substitute = previous[:-1] + (hyp != ref[i - 1])
        delete = previous[1:] + 1
        current = np.empty(m + 1, dtype=np.int32)
        current[0] = i
        current[1:] = np.minimum(substitute, delete)
        moves[i, 1:] = np.where(substitute <= delete, diagonal, deleted)
        # Inserting can chain across the row, which a running minimum handles in one pass.
        best = np.minimum.accumulate(current - positions) + positions
        moves[i, best < current] = inserted
        previous = best

    operations = []
    i, j = n, m
    while i > 0 or j > 0:
        move = moves[i, j]
        if move == diagonal:
            operations.append(("match" if ref[i - 1] == hyp[j - 1] else "wrong", j - 1))
            i, j = i - 1, j - 1
        elif move == deleted:
            operations.append(("missed", j))
            i -= 1
        else:
            operations.append(("added", j - 1))
            j -= 1
    operations.reverse()
    return operations


@dataclass
class Score:
    reference_words: int
    app_words: int
    wrong: int
    missed: int
    added: int
    accuracy_without_fillers: float
    long_missed: list[tuple[int, float]] = field(default_factory=list)
    long_added: list[tuple[int, float]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return _accuracy(self.wrong, self.missed, self.added, self.reference_words)


def _accuracy(wrong: int, missed: int, added: int, total: int) -> float:
    if not total:
        return 0.0
    return max(0.0, 1 - (wrong + missed + added) / total) * 100


def _counts(operations: list[tuple[str, int]]) -> tuple[int, int, int]:
    kinds = [kind for kind, _ in operations]
    return kinds.count("wrong"), kinds.count("missed"), kinds.count("added")


def _long_runs(operations: list[tuple[str, int]], kind: str, times: list[float]) -> list[tuple[int, float]]:
    runs = []
    length = 0
    first_position = 0
    for operation, position in operations + [("end", 0)]:
        if operation == kind:
            if not length:
                first_position = position
            length += 1
            continue
        if length >= LONG_RUN_WORDS:
            nearest = min(first_position, len(times) - 1)
            runs.append((length, times[nearest] if times else 0.0))
        length = 0
    return runs


def score(reference: list[str], hypothesis: list[str], times: list[float]) -> Score:
    operations = align(reference, hypothesis)
    wrong, missed, added = _counts(operations)
    reference_clean = [word for word in reference if word not in FILLERS]
    hypothesis_clean = [word for word in hypothesis if word not in FILLERS]
    clean = _counts(align(reference_clean, hypothesis_clean))
    return Score(
        reference_words=len(reference),
        app_words=len(hypothesis),
        wrong=wrong,
        missed=missed,
        added=added,
        accuracy_without_fillers=_accuracy(*clean, len(reference_clean)),
        long_missed=_long_runs(operations, "missed", times),
        long_added=_long_runs(operations, "added", times),
    )


def find_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each recording with the .txt beside it: same name, or the only one in its folder."""
    pairs = []
    for folder in sorted({path.parent for path in input_dir.rglob("*") if path.is_file()}):
        recordings = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        texts = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".txt")
        for recording in recordings:
            same_name = [text for text in texts if text.stem == recording.stem]
            if same_name:
                pairs.append((recording, same_name[0]))
            elif len(recordings) == 1 and len(texts) == 1:
                pairs.append((recording, texts[0]))
    return pairs


def find_app_transcript(output_dir: Path, stem: str) -> Path | None:
    """The newest transcript for this recording, including reruns saved as 'name (2).txt'."""
    pattern = re.compile(rf"^{re.escape(stem)}(?: \(\d+\))?\.txt$")
    matches = [path for path in output_dir.glob("*.txt") if pattern.match(path.name)]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _runs_text(runs: list[tuple[int, float]]) -> str:
    return ", ".join(f"{length} words near {format_time(time)}" for length, time in runs)


def report(input_dir: Path, output_dir: Path, out=None) -> int:
    out = out or sys.stdout
    require_local(input_dir, "recordings")
    require_local(output_dir, "transcripts")

    pairs = find_pairs(input_dir)
    if not pairs:
        print("No recordings with a .txt transcript beside them were found.", file=out)
        return 1

    key_rows = []
    scores = []
    not_ready = 0
    for number, (recording, text_path) in enumerate(pairs, start=1):
        label = f"Recording {number}"
        transcript = find_app_transcript(output_dir, recording.stem)
        key_rows.append([label, str(recording), str(text_path), str(transcript or "")])
        if transcript is None:
            not_ready += 1
            print(f"{label}: not transcribed yet", file=out)
            continue

        reference = reference_words(read_text(text_path))
        hypothesis, times = app_words(read_text(transcript))
        result = score(reference, hypothesis, times)
        scores.append(result)
        print(f"{label}: {result.reference_words} words in existing transcript, {result.app_words} from the app", file=out)
        print(f"  Accuracy: {result.accuracy:.1f}% ({result.accuracy_without_fillers:.1f}% ignoring um, uh and similar)", file=out)
        print(f"  Differences: {result.wrong} wrong, {result.missed} missed, {result.added} added", file=out)
        if result.long_missed:
            print(f"  Long missed stretches: {_runs_text(result.long_missed)}", file=out)
        if result.long_added:
            print(f"  Long added stretches: {_runs_text(result.long_added)}", file=out)
        smaller, larger = sorted((result.reference_words, result.app_words))
        if not smaller or larger / smaller > SUSPICIOUS_LENGTH_RATIO:
            print("  CHECK: word counts are far apart. These may not be the same recording.", file=out)

    if scores:
        total = sum(s.reference_words for s in scores)
        wrong, missed, added = (sum(getattr(s, name) for s in scores) for name in ("wrong", "missed", "added"))
        recordings = "1 recording" if len(scores) == 1 else f"{len(scores)} recordings"
        print(f"\nOverall, {recordings}, {total} words", file=out)
        print(f"  Accuracy: {_accuracy(wrong, missed, added, total):.1f}%", file=out)
        print(f"  Differences: {wrong} wrong, {missed} missed, {added} added", file=out)
    if not_ready:
        needs = "1 recording still needs" if not_ready == 1 else f"{not_ready} recordings still need"
        print(f"\n{needs} transcribing in the app.", file=out)

    output_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / KEY_NAME
    with open(key_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "recording", "existing transcript", "app transcript"])
        writer.writerows(key_rows)
    print(f"\nWhich file is which: {KEY_NAME}, in your transcripts folder. Don't share that file.", file=out)
    print("Everything above is safe to share: it contains no words or file names.", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare", description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    block_network()
    return report(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
