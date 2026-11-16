"""Converts any audio or video file to the 16 kHz mono WAV the models expect."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000

SUPPORTED_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".amr", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".wma",
    ".m4v", ".mkv", ".mov", ".mp4", ".webm",
}


# Stops a console window flashing up each time the desktop window runs a tool.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class AudioError(Exception):
    """A recording couldn't be read or converted."""


def _find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    # winget adds its folder to PATH, but only for programs started after the install.
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for candidate in packages.glob(f"Gyan.FFmpeg_*/*/bin/{name}.exe"):
        return str(candidate)
    raise AudioError(f"{name} isn't installed or isn't on PATH.")


def recordings_in(folder: Path) -> list[Path]:
    """Every recording in a folder and the folders inside it."""
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    with open(source, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [_find_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe couldn't read {path.name}: {result.stderr.strip()}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError) as error:
        raise AudioError(f"{path.name} doesn't report a length.") from error


def convert_to_wav(source: Path, destination: Path) -> str | None:
    """Write a working copy; the original recording is never changed.

    Returns any read errors. ffmpeg skips damaged audio and still reports success,
    so those words would otherwise go missing without a trace.
    """
    if destination.exists():
        return None
    partial = destination.with_suffix(".part")
    command = [
        _find_tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        # Plain WAV stops at 4 GB (about 37 hours at this rate); RF64 has no such limit.
        "-rf64", "auto", "-f", "wav", str(partial),
    ]
    result = subprocess.run(command, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg couldn't convert {source.name}: {result.stderr.strip()}")
    partial.replace(destination)
    return result.stderr.strip() or None
