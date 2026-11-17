"""Runs Whisper and saves each finished segment. Where it runs comes from hardware.py."""

import gc
import os
import time
from collections.abc import Callable
from pathlib import Path

# The models load from a local folder, so nothing here should reach the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import soundfile  # noqa: E402

from transcribe_local.audio import SAMPLE_RATE  # noqa: E402
from transcribe_local.checkpoint import SegmentWriter, load_segments  # noqa: E402

MODEL_NAME = "faster-whisper-large-v3"

# Trailing audio shorter than this can't hold a word worth another pass.
MIN_REMAINING_SECONDS = 0.5

# The default threshold (0.5) silently dropped real speech in quiet passages.
# Stray noise that gets through is cheaper: it can be caught in review, while
# speech dropped here never reaches the transcript at all.
VAD_OPTIONS = {"threshold": 0.1, "min_silence_duration_ms": 2000, "speech_pad_ms": 400}

# Recorded with each job. Changing any of these starts a fresh transcript.
SETTINGS = {
    "model": MODEL_NAME,
    "language": "en",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "vad": VAD_OPTIONS,
}


# Words in an error that mean the graphics card itself couldn't run the model, for
# example a card newer than the transcription library, a broken driver install, or a
# card that can't use the lighter settings.
_CARD_FAILURE_SIGNS = ("cuda", "cudnn", "cublas", "kernel image", "nvrtc", ".dll", "target device")

# Why the graphics card couldn't be used, once it has failed. Later jobs in the same
# session go straight to the processor instead of failing on the card again.
_card_failure: str | None = None

CARD_FAILURE_NOTICE = (
    "The graphics card couldn't run the model, so the processor is being used instead. "
    "It's much slower. Updating the NVIDIA driver usually fixes this."
)


def _is_out_of_memory(error: Exception) -> bool:
    # The graphics libraries also report running out as CUBLAS_STATUS_ALLOC_FAILED
    # or CUDNN_STATUS_ALLOC_FAILED.
    message = str(error).lower()
    return isinstance(error, MemoryError) or any(
        sign in message for sign in ("out of memory", "bad_alloc", "alloc_failed")
    )


def is_card_failure(error: Exception, device: str) -> bool:
    """The card can't run the model at all. Running out of memory doesn't count: lighter settings fix that."""
    if device != "cuda" or _is_out_of_memory(error):
        return False
    return any(sign in str(error).lower() for sign in _CARD_FAILURE_SIGNS)


def load_model(model_path: Path, device: str, compute_type: str, cpu_threads: int):
    if device == "cuda":
        from transcribe_local.cuda_libraries import make_findable

        make_findable()
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(model_path),
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads if device == "cpu" else 0,
        local_files_only=True,
    )


def _segment_record(segment, offset: float) -> dict:
    return {
        "start": round(segment.start + offset, 3),
        "end": round(segment.end + offset, 3),
        "text": segment.text.strip(),
        "avg_logprob": round(segment.avg_logprob, 4),
        "compression_ratio": round(segment.compression_ratio, 4),
        "no_speech_prob": round(segment.no_speech_prob, 4),
        "words": [
            {
                "start": round(word.start + offset, 3),
                "end": round(word.end + offset, 3),
                "word": word.word,
                "probability": round(word.probability, 4),
            }
            for word in (segment.words or [])
        ],
    }


def _run_attempt(
    wav_path: Path,
    segments_path: Path,
    model_path: Path,
    attempt: tuple[str, str, int],
    cpu_threads: int,
    duration: float,
    on_progress: Callable[[float], None],
) -> None:
    from faster_whisper import BatchedInferencePipeline

    device, compute_type, batch_size = attempt
    saved = load_segments(segments_path)
    offset = saved[-1]["end"] if saved else 0.0
    if duration - offset < MIN_REMAINING_SECONDS:
        return

    # Only what's left of the recording is held in memory, at about 230 MB per hour.
    with soundfile.SoundFile(wav_path) as file:
        file.seek(int(offset * SAMPLE_RATE))
        audio = file.read(dtype="float32")

    model = load_model(model_path, device, compute_type, cpu_threads)
    segments, _ = BatchedInferencePipeline(model).transcribe(
        audio,
        language=SETTINGS["language"],
        batch_size=batch_size,
        beam_size=SETTINGS["beam_size"],
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=VAD_OPTIONS,
        condition_on_previous_text=SETTINGS["condition_on_previous_text"],
    )
    with SegmentWriter(segments_path) as writer:
        for segment in segments:
            record = _segment_record(segment, offset)
            writer.append(record)
            on_progress(record["end"])


def transcribe(
    wav_path: Path,
    segments_path: Path,
    model_path: Path,
    duration: float,
    on_progress: Callable[[float], None] = lambda _: None,
    attempts: tuple[tuple[str, str, int], ...] | None = None,
    cpu_threads: int | None = None,
    on_notice: Callable[[str], None] = lambda _: None,
) -> dict:
    """Transcribe to segments_path, resuming from whatever is already saved.

    Running out of memory moves on to the next, lighter attempt. A graphics card that
    can't run the model at all skips the remaining card attempts and goes to the
    processor, and on_notice is told so the person knows why it slowed down.
    """
    global _card_failure
    from transcribe_local.hardware import current_profile, default_cpu_threads

    if attempts is None:
        profile = current_profile()
        attempts = profile.attempts
        cpu_threads = cpu_threads or profile.cpu_threads
    cpu_threads = cpu_threads or default_cpu_threads()
    started = time.monotonic()
    saved = load_segments(segments_path)
    resumed_from = saved[-1]["end"] if saved else 0.0
    last_error = None
    card_problem = None
    for attempt in attempts:
        device, compute_type, batch_size = attempt
        if device == "cuda" and _card_failure is not None:
            if card_problem is None:
                card_problem = _card_failure
                on_notice(CARD_FAILURE_NOTICE)
            continue
        try:
            _run_attempt(wav_path, segments_path, model_path, attempt, cpu_threads, duration, on_progress)
        except (RuntimeError, MemoryError, OSError, ValueError) as error:
            # Keep only the text: the exception's traceback would hold the failed
            # model in graphics memory while the next attempt tries to load.
            if _is_out_of_memory(error):
                last_error = str(error)
            elif is_card_failure(error, device):
                _card_failure = card_problem = last_error = str(error)
                on_notice(CARD_FAILURE_NOTICE)
            else:
                raise
        else:
            run = {
                "device": device,
                "compute_type": compute_type,
                "batch_size": batch_size,
                "resumed_from": resumed_from,
                "seconds": round(time.monotonic() - started, 2),
            }
            if card_problem:
                run["card_problem"] = card_problem
            return run
        gc.collect()
    if card_problem and last_error in (None, card_problem):
        raise RuntimeError(f"The graphics card couldn't run the model: {card_problem}")
    raise RuntimeError(f"Ran out of memory at the lightest settings: {last_error}")
