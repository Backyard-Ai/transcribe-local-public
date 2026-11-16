"""Saves transcript segments as they finish, so long jobs can resume after a crash."""

import json
import os
from pathlib import Path


def load_segments(path: Path) -> list[dict]:
    """Read saved segments, dropping a last line cut off by a crash."""
    if not path.exists():
        return []
    segments = []
    valid_bytes = 0
    with open(path, "rb") as file:
        for line in file:
            try:
                segments.append(json.loads(line))
            except ValueError:
                break
            if not line.endswith(b"\n"):
                segments.pop()
                break
            valid_bytes += len(line)
    if valid_bytes != path.stat().st_size:
        with open(path, "r+b") as file:
            file.truncate(valid_bytes)
    return segments


class SegmentWriter:
    def __init__(self, path: Path):
        self._file = open(path, "a", encoding="utf-8", newline="\n")

    def append(self, segment: dict) -> None:
        self._file.write(json.dumps(segment, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
