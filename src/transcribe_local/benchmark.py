"""Measures how fast this computer transcribes and how much graphics memory it uses.

Use non-sensitive audio only, such as a recording of yourself reading a book aloud.
The transcript is made in a temporary folder and deleted straight afterwards, and only
numbers are printed, never words or the file's name.

  .venv\\Scripts\\python -m transcribe_local.benchmark "C:\\path\\to\\sample.mp3"
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from transcribe_local import audio, engine
from transcribe_local.hardware import _NO_WINDOW, current_profile
from transcribe_local.pipeline import DEFAULT_MODEL
from transcribe_local.safety import block_network, require_local

SAMPLE_SECONDS = 0.5


def graphics_memory_used_gb() -> float | None:
    """Graphics memory in use by every program, from NVIDIA's driver tool. None if there's no card."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        result = subprocess.run(
            [tool, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW,
        )
        return float(result.stdout.strip().splitlines()[0]) / 1024
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


class MemoryPeak:
    """Reads graphics memory in the background while a job runs and keeps the highest reading."""

    def __init__(self, read: Callable[[], float | None] = graphics_memory_used_gb, interval: float = SAMPLE_SECONDS):
        self._read = read
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak: float | None = None

    def _sample(self) -> None:
        while True:
            reading = self._read()
            if reading is not None:
                self.peak = reading if self.peak is None else max(self.peak, reading)
            if self._stop.wait(self._interval):
                return

    def __enter__(self) -> "MemoryPeak":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def run_benchmark(
    source: Path,
    model_path: Path = DEFAULT_MODEL,
    read_memory: Callable[[], float | None] | None = None,
) -> dict:
    read_memory = read_memory or graphics_memory_used_gb
    block_network()
    require_local(source, "the sample recording")
    if not source.is_file():
        raise audio.AudioError("The sample recording isn't there.")
    if not model_path.exists():
        raise audio.AudioError(f"The model isn't at {model_path}. Run scripts/download_models.py first.")

    # A fresh folder each time, so nothing is resumed and the timing covers the whole sample.
    with tempfile.TemporaryDirectory(prefix="transcribe-local-benchmark-", ignore_cleanup_errors=True) as folder:
        wav = Path(folder) / "audio.wav"
        audio.convert_to_wav(source, wav)
        duration = audio.duration_seconds(wav)
        before = read_memory()
        with MemoryPeak(read_memory) as memory:
            run = engine.transcribe(wav, Path(folder) / "segments.jsonl", model_path, duration)
    # Windows can refuse to delete a file that's still open. Say so rather than leave it unnoticed.
    leftover = folder if Path(folder).exists() else None

    attempt = (run["device"], run["compute_type"], run["batch_size"])
    used = memory.peak - before if memory.peak is not None and before is not None else None
    return {
        "audio_minutes": round(duration / 60, 2),
        "minutes": round(run["seconds"] / 60, 2),
        "times_real_time": round(duration / run["seconds"], 1) if run["seconds"] else None,
        "settings": attempt,
        "first_choice_settings": attempt == current_profile().attempts[0],
        "memory_before_gb": None if before is None else round(before, 2),
        "memory_peak_gb": None if memory.peak is None else round(memory.peak, 2),
        "memory_used_by_job_gb": None if used is None else round(used, 2),
        "card_problem": bool(run.get("card_problem")),
        "leftover_folder": leftover,
    }


def report(result: dict) -> str:
    device, compute_type, batch_size = result["settings"]
    lines = [
        "Benchmark (numbers only)",
        f"  Audio: {result['audio_minutes']:.1f} min",
        f"  Time, including loading the model: {result['minutes']:.1f} min",
        f"  Speed: {result['times_real_time'] or 'too fast to measure'}x real time",
        f"  Settings used: {'graphics card' if device == 'cuda' else 'processor'}, {compute_type}, batch {batch_size}",
    ]
    if not result["first_choice_settings"]:
        lines.append("  NOTE: lighter settings than this computer's first choice were needed.")
    if result["card_problem"]:
        lines.append("  NOTE: the graphics card couldn't run the model, so the processor was used.")
    if result["memory_peak_gb"] is not None:
        lines.append(
            f"  Graphics memory: {result['memory_before_gb']:.1f} GB in use before, "
            f"{result['memory_peak_gb']:.1f} GB at the peak, so about {result['memory_used_by_job_gb']:.1f} GB for this job"
        )
    if result.get("leftover_folder"):
        lines.append(f"  WARNING: couldn't delete the temporary folder {result['leftover_folder']}. Delete it yourself.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", type=Path, help="a non-sensitive recording, a few minutes long")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args.sample, args.model)
    except audio.AudioError:
        # Its details can name the file, and this printout is meant to be shareable.
        print("Benchmark failed: the sample recording couldn't be read or converted.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    print(report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
