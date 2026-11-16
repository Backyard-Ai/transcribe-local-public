"""Writes finished transcripts and the list of passages to check."""

import csv
from pathlib import Path


def format_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def write_text(segments: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for segment in segments:
            file.write(f"[{format_time(segment['start'])}] {segment['text']}\n")


def write_review(flagged: list[dict], untranscribed: list[tuple[float, float]], path: Path) -> None:
    rows = [(passage["start"], passage["end"], "; ".join(passage["reasons"]), passage["text"]) for passage in flagged]
    rows += [(start, end, "sound with no transcript", "") for start, end in untranscribed]
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "end", "reasons", "text"])
        for start, end, reasons, text in sorted(rows, key=lambda row: row[0]):
            writer.writerow([format_time(start), format_time(end), reasons, text])
