"""Takes one recording from audio file to finished transcript. Shared by the window and command line."""

import ctypes
import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from transcribe_local.audio import (
    SUPPORTED_EXTENSIONS,
    AudioError,
    convert_to_wav,
    duration_seconds,
    source_hash,
)
from transcribe_local.safety import block_network, require_local

REPO_ROOT = Path(__file__).resolve().parents[2]
# Written by scripts/setup.ps1 when a computer needs a different data folder. Not in Git.
LOCAL_SETTINGS = REPO_ROOT / "local_settings.json"
DESKTOP_DATA_ROOT = Path(r"E:\TranscriptionData")


def resolve_data_root(environment: dict, settings_path: Path, desktop_folder: Path, home: Path) -> Path:
    """Where recordings and transcripts live: a setting if there is one, else a sensible default."""
    if environment.get("TRANSCRIBE_LOCAL_DATA"):
        return Path(environment["TRANSCRIBE_LOCAL_DATA"])
    try:
        configured = json.loads(settings_path.read_text(encoding="utf-8")).get("data_folder")
    except (OSError, ValueError, AttributeError):
        configured = None
    if configured:
        return Path(configured)
    # Only an existing folder counts: on another computer, E: could be a USB stick.
    if desktop_folder.is_dir():
        return desktop_folder
    return home / "TranscriptionData"


DATA_ROOT = resolve_data_root(os.environ, LOCAL_SETTINGS, DESKTOP_DATA_ROOT, Path.home())
MODELS_ROOT = Path(os.environ.get("TRANSCRIBE_LOCAL_MODELS", REPO_ROOT / "models"))
DEFAULT_OUTPUT = DATA_ROOT / "output"
DEFAULT_WORK = DATA_ROOT / "work"
DEFAULT_MODEL = MODELS_ROOT / "faster-whisper-large-v3"

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class Cancelled(Exception):
    """The person stopped the job. What's saved so far is kept for next time."""


def keep_awake(enabled: bool) -> None:
    """Applies to the calling thread, so call it from the thread doing the work."""
    if os.name == "nt":
        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if enabled else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


class TimeLeft:
    """Estimates from the speed since the first segment, so setup time and resumed audio don't skew it."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._first = None

    def minutes(self, position: float, duration: float) -> float | None:
        now = self._clock()
        if self._first is None:
            self._first = (position, now)
            return None
        first_position, first_time = self._first
        done, elapsed = position - first_position, now - first_time
        if done <= 0 or elapsed <= 0:
            return None
        return max(0.0, duration - position) / (done / elapsed) / 60

    def restart(self) -> None:
        """Start measuring again, for when the speed changes, such as a switch to the processor."""
        self._first = None


def load_job(path: Path, new_job: dict) -> dict:
    """A damaged record only costs the job's history; its saved segments are separate."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return new_job


def save_job(path: Path, job: dict) -> None:
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(job, indent=2), encoding="utf-8")
    partial.replace(path)


def remove_working_audio(wav_path: Path) -> bool:
    """Delete the job's copy of the recording. Returns False if it couldn't be removed."""
    try:
        wav_path.unlink(missing_ok=True)
        wav_path.with_suffix(".part").unlink(missing_ok=True)
    except OSError:
        return False
    return True


def free_stem(output_dir: Path, stem: str) -> str:
    """Pick a name that won't overwrite an earlier transcript, which may have been edited."""
    candidate = stem
    number = 2
    while any((output_dir / f"{candidate}{suffix}").exists() for suffix in (".docx", ".txt", "_review.csv")):
        candidate = f"{stem} ({number})"
        number += 1
    return candidate


def process(
    source: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    work_dir: Path = DEFAULT_WORK,
    model_path: Path = DEFAULT_MODEL,
    on_stage: Callable[[str], None] = lambda _: None,
    on_progress: Callable[[float, float], None] = lambda position, duration: None,
    on_notice: Callable[[str], None] = lambda _: None,
) -> dict:
    """on_notice receives plain-language news the person should see, such as a switch to the processor."""
    block_network()

    from transcribe_local.checkpoint import load_segments
    from transcribe_local.coverage import find_untranscribed_audio
    from transcribe_local.engine import SETTINGS, transcribe
    from transcribe_local.maxqda import write_docx
    from transcribe_local.output import write_review, write_text
    from transcribe_local.review import build_passages

    require_local(source, "the recording")
    require_local(output_dir, "transcripts")
    require_local(work_dir, "working files")
    if not source.is_file():
        raise AudioError("The file isn't there anymore. It may have been moved or renamed.")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(f"{source.suffix} isn't a supported audio or video format.")
    if not model_path.exists():
        raise AudioError(f"The model isn't at {model_path}. Run scripts/download_models.py first.")

    on_stage("Reading the recording")
    digest = source_hash(source)
    job_key = hashlib.sha256((digest + json.dumps(SETTINGS, sort_keys=True)).encode()).hexdigest()[:8]
    job_dir = work_dir / f"{source.stem}-{job_key}"
    job_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = job_dir / "audio.wav"
    segments_path = job_dir / "segments.jsonl"
    job_path = job_dir / "job.json"

    job = load_job(job_path, {
        "source": source.name,
        "sha256": digest,
        "settings": SETTINGS,
        "created": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "runs": [],
    })

    on_stage("Converting the audio")
    read_errors = convert_to_wav(source, wav_path)
    if read_errors:
        job["read_errors"] = read_errors
    duration = duration_seconds(wav_path)
    job["duration"] = duration
    save_job(job_path, job)

    if not job["complete"]:
        on_stage("Finding speech and loading the model")
        run = transcribe(
            wav_path,
            segments_path,
            model_path,
            duration,
            lambda position: on_progress(position, duration),
            on_notice=on_notice,
        )
        job["runs"].append(run)
        job["complete"] = True
        save_job(job_path, job)

    on_stage("Writing the transcript")
    segments = load_segments(segments_path)
    flagged = [passage for passage in build_passages(segments) if passage["reasons"]]
    untranscribed = find_untranscribed_audio(wav_path, segments)
    stem = free_stem(output_dir, source.stem)
    write_docx(segments, output_dir / f"{stem}.docx")
    write_text(segments, output_dir / f"{stem}.txt")
    write_review(flagged, untranscribed, output_dir / f"{stem}_review.csv")
    # Kept until now because a stopped or failed job needs it to resume. A problem
    # here is reported rather than raised, since the transcript is already safe.
    working_audio_removed = remove_working_audio(wav_path)

    # Speed covers the last run only: a run that was stopped or crashed records no time.
    last_run = job["runs"][-1]
    return {
        "source": source.name,
        "transcript": str(output_dir / f"{stem}.docx"),
        "audio_minutes": round(duration / 60, 2),
        "last_run_minutes": round(last_run["seconds"] / 60, 2),
        "last_run_times_real_time": round((duration - last_run["resumed_from"]) / last_run["seconds"], 1),
        "segments": len(segments),
        "flagged_passages": len(flagged),
        "sound_with_no_transcript": len(untranscribed),
        "read_errors": job.get("read_errors"),
        "working_audio_removed": working_audio_removed,
        "last_run": last_run,
    }
