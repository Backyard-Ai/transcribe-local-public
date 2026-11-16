import shutil
import subprocess
from types import SimpleNamespace

import faster_whisper
import numpy as np
import pytest
import soundfile

from transcribe_local import audio, engine
from transcribe_local.checkpoint import load_segments
from transcribe_local.engine import CARD_FAILURE_NOTICE, is_card_failure, transcribe

RATE = 16000
OUT_OF_MEMORY = "CUDA failed with error out of memory"
# What CTranslate2 reports when it has no code that runs on the card.
NO_KERNEL = "CUDA failed with error no kernel image is available for execution on the device"
CARD_THEN_PROCESSOR = (("cuda", "float16", 4), ("cuda", "int8_float16", 2), ("cpu", "int8", 1))


@pytest.fixture(autouse=True)
def card_not_failed_yet(monkeypatch):
    monkeypatch.setattr(engine, "_card_failure", None)


def fake_segment(start, end, text):
    return SimpleNamespace(
        start=start, end=end, text=text, avg_logprob=-0.2, compression_ratio=1.2, no_speech_prob=0.0, words=[]
    )


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "audio.wav"
    soundfile.write(path, np.zeros(60 * RATE, dtype="float32"), RATE)
    return path


class FakeModels:
    """Stands in for the real models and records what each attempt could see."""

    def __init__(self, monkeypatch, plan):
        self.plan = iter(plan)
        self.loaded = []
        self.live_at_load = []
        self.audio_lengths = []
        self.devices = []
        fakes = self

        class FakeWhisperModel:
            live = 0

            def __init__(self, path, device, compute_type, local_files_only, cpu_threads=0):
                assert local_files_only
                fakes.live_at_load.append(FakeWhisperModel.live)
                FakeWhisperModel.live += 1
                fakes.loaded.append(compute_type)
                fakes.devices.append((device, cpu_threads))

            def __del__(self):
                FakeWhisperModel.live -= 1

        class FakePipeline:
            def __init__(self, model):
                self.model = model

            def transcribe(self, audio_array, **options):
                fakes.audio_lengths.append(len(audio_array) / RATE)
                step = next(fakes.plan)

                def generate(model=self.model):
                    for item in step:
                        # Raised fresh each time: a stored exception would keep its
                        # traceback here, and the model alive with it.
                        if isinstance(item, str):
                            raise RuntimeError(item)
                        if item is MemoryError:
                            raise MemoryError()
                        yield item

                return generate(), None

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
        monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", FakePipeline)


ATTEMPTS = (("cpu", "float16", 8), ("cpu", "int8_float16", 2))


def test_out_of_memory_frees_the_model_before_trying_lighter_settings(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [[OUT_OF_MEMORY], [fake_segment(0.0, 59.9, "Done.")]])
    result = transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=ATTEMPTS)
    assert fakes.loaded == ["float16", "int8_float16"]
    assert fakes.live_at_load == [0, 0]
    assert result["compute_type"] == "int8_float16"


def test_graphics_card_falls_back_to_the_processor(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [[OUT_OF_MEMORY], [fake_segment(0.0, 59.9, "Done.")]])
    attempts = (("cuda", "int8_float16", 1), ("cpu", "int8", 1))
    result = transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=attempts, cpu_threads=6)
    assert fakes.devices == [("cuda", 0), ("cpu", 6)]
    assert result["device"] == "cpu"


def test_processor_running_out_of_memory_is_retried(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [[MemoryError], [fake_segment(0.0, 59.9, "Done.")]])
    result = transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=(("cpu", "int8", 4), ("cpu", "int8", 1)))
    assert result["batch_size"] == 1
    assert len(fakes.loaded) == 2


def test_retry_keeps_saved_segments_and_continues_from_them(monkeypatch, wav, tmp_path):
    fakes = FakeModels(
        monkeypatch,
        [
            [fake_segment(0.0, 20.0, "First."), OUT_OF_MEMORY],
            [fake_segment(1.0, 39.0, "Second.")],
        ],
    )
    path = tmp_path / "s.jsonl"
    transcribe(wav, path, tmp_path, 60.0, attempts=ATTEMPTS)
    assert fakes.audio_lengths == pytest.approx([60.0, 40.0])
    saved = load_segments(path)
    assert [s["text"] for s in saved] == ["First.", "Second."]
    # The second attempt's times are relative to where it resumed.
    assert (saved[1]["start"], saved[1]["end"]) == (21.0, 59.0)


def test_running_out_at_the_lightest_settings_is_reported(monkeypatch, wav, tmp_path):
    FakeModels(monkeypatch, [[OUT_OF_MEMORY], [OUT_OF_MEMORY]])
    with pytest.raises(RuntimeError, match="lightest settings"):
        transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=ATTEMPTS)


def test_other_errors_are_not_retried(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [["cublas64_12.dll is not found"]])
    with pytest.raises(RuntimeError, match="cublas"):
        transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=ATTEMPTS)
    assert fakes.loaded == ["float16"]


def test_stopping_keeps_what_was_saved(monkeypatch, wav, tmp_path):
    class Stop(Exception):
        pass

    def stop_after_first(position):
        raise Stop

    FakeModels(monkeypatch, [[fake_segment(0.0, 5.0, "Kept."), fake_segment(5.0, 9.0, "Never reached.")]])
    path = tmp_path / "s.jsonl"
    with pytest.raises(Stop):
        transcribe(wav, path, tmp_path, 60.0, on_progress=stop_after_first, attempts=ATTEMPTS)
    assert [s["text"] for s in load_segments(path)] == ["Kept."]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg isn't on PATH")
def test_damaged_recording_reports_read_errors(tmp_path):
    source = tmp_path / "source.mp3"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=20", str(source)],
        check=True,
    )
    clean = audio.convert_to_wav(source, tmp_path / "clean.wav")
    assert clean is None

    data = bytearray(source.read_bytes())
    middle = len(data) // 2
    data[middle:middle + 20000] = bytes(range(256)) * 78 + bytes(32)
    damaged = tmp_path / "damaged.mp3"
    damaged.write_bytes(data)
    assert audio.convert_to_wav(damaged, tmp_path / "damaged.wav")
    assert (tmp_path / "damaged.wav").exists()


def test_existing_conversion_is_reused(tmp_path):
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"already here")
    assert audio.convert_to_wav(tmp_path / "missing.m4a", destination) is None
    assert destination.read_bytes() == b"already here"


def test_card_that_cannot_run_the_model_goes_straight_to_the_processor(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [[NO_KERNEL], [fake_segment(0.0, 59.9, "Done.")]])
    notices = []
    result = transcribe(
        wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR, cpu_threads=6,
        on_notice=notices.append,
    )
    # The lighter card settings would fail the same way, so they're skipped.
    assert fakes.devices == [("cuda", 0), ("cpu", 6)]
    assert result["device"] == "cpu"
    assert "no kernel image" in result["card_problem"]
    assert notices == [CARD_FAILURE_NOTICE]


def test_later_jobs_skip_a_card_that_already_failed(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [[NO_KERNEL], [fake_segment(0.0, 59.9, "One.")], [fake_segment(0.0, 59.9, "Two.")]])
    transcribe(wav, tmp_path / "a.jsonl", tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR, cpu_threads=6)
    notices = []
    second = transcribe(
        wav, tmp_path / "b.jsonl", tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR, cpu_threads=6,
        on_notice=notices.append,
    )
    assert fakes.devices == [("cuda", 0), ("cpu", 6), ("cpu", 6)]
    assert "no kernel image" in second["card_problem"]
    assert notices == [CARD_FAILURE_NOTICE]


def test_card_failing_partway_keeps_its_segments_and_the_processor_continues(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [[fake_segment(0.0, 20.0, "Card."), NO_KERNEL], [fake_segment(1.0, 39.0, "Processor.")]])
    path = tmp_path / "s.jsonl"
    transcribe(wav, path, tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR)
    assert fakes.audio_lengths == pytest.approx([60.0, 40.0])
    assert [s["text"] for s in load_segments(path)] == ["Card.", "Processor."]


def test_card_failure_without_a_processor_attempt_is_reported(monkeypatch, wav, tmp_path):
    FakeModels(monkeypatch, [[NO_KERNEL]])
    with pytest.raises(RuntimeError, match="graphics card couldn't run the model"):
        transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=(("cuda", "float16", 4),))


def test_ordinary_errors_on_the_card_are_still_raised(monkeypatch, wav, tmp_path):
    fakes = FakeModels(monkeypatch, [["something unrelated broke"]])
    with pytest.raises(RuntimeError, match="unrelated"):
        transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR)
    assert fakes.devices == [("cuda", 0)]
    assert engine._card_failure is None


def test_successful_run_records_no_card_problem(monkeypatch, wav, tmp_path):
    FakeModels(monkeypatch, [[fake_segment(0.0, 59.9, "Done.")]])
    result = transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR)
    assert "card_problem" not in result


@pytest.mark.parametrize(
    "message, device, expected",
    [
        (NO_KERNEL, "cuda", True),
        ("cuDNN failed with status CUDNN_STATUS_EXECUTION_FAILED", "cuda", True),
        ("Library cublas64_12.dll is not found or cannot be loaded", "cuda", True),
        (
            "Requested int8_float16 compute type, but the target device or backend do not support "
            "efficient int8_float16 computation.",
            "cuda",
            True,
        ),
        ("something unrelated broke", "cuda", False),
        # Running out of memory is fixed by lighter settings, not by giving up on the card.
        (OUT_OF_MEMORY, "cuda", False),
        ("cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED", "cuda", False),
        ("cuDNN failed with status CUDNN_STATUS_ALLOC_FAILED", "cuda", False),
        # On the processor there's no card to give up on.
        ("Library cublas64_12.dll is not found or cannot be loaded", "cpu", False),
    ],
)
def test_card_failures_are_recognised(message, device, expected):
    assert is_card_failure(RuntimeError(message), device) is expected


def test_library_reporting_out_of_memory_tries_lighter_card_settings(monkeypatch, wav, tmp_path):
    fakes = FakeModels(
        monkeypatch, [["cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED"], [fake_segment(0.0, 59.9, "Done.")]]
    )
    result = transcribe(wav, tmp_path / "s.jsonl", tmp_path, 60.0, attempts=CARD_THEN_PROCESSOR, cpu_threads=6)
    assert fakes.devices == [("cuda", 0), ("cuda", 0)]
    assert "card_problem" not in result
    assert engine._card_failure is None
