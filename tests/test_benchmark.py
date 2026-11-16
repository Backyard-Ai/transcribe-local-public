import socket
import threading

import numpy as np
import pytest
import soundfile

from transcribe_local import audio, benchmark, engine
from transcribe_local.checkpoint import SegmentWriter
from transcribe_local.hardware import current_profile

RATE = 16000
SECRET = "Words that must never be printed."


@pytest.fixture
def sample(tmp_path, monkeypatch):
    # block_network() replaces these for the whole test session otherwise.
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    source = tmp_path / "private name.wav"
    soundfile.write(source, np.zeros(RATE, dtype="float32"), RATE)
    model = tmp_path / "model"
    model.mkdir()
    return source, model


@pytest.fixture
def fake_job(monkeypatch):
    seen = {}

    def convert(source, destination):
        destination.write_bytes(b"audio")

    def transcribe(wav_path, segments_path, model_path, duration):
        seen["wav"] = wav_path
        # Like a real job, keep running until the memory sampler has had its readings.
        seen.get("all_read", threading.Event()).wait(timeout=5)
        with SegmentWriter(segments_path) as writer:
            writer.append({"start": 0.0, "end": 60.0, "text": SECRET, "words": []})
        return {**seen.get("run", {"device": "cuda", "compute_type": "float16", "batch_size": 4}), "seconds": 4.0}

    monkeypatch.setattr(audio, "convert_to_wav", convert)
    monkeypatch.setattr(audio, "duration_seconds", lambda path: 120.0)
    monkeypatch.setattr(engine, "transcribe", transcribe)
    return seen


def readings(*values, all_read=None):
    """Returns each value in turn, then repeats the last, setting all_read once they're used up."""
    values = iter(values)
    last = [None]

    def read():
        try:
            last[0] = next(values)
        except StopIteration:
            if all_read:
                all_read.set()
        return last[0]

    return read


def test_benchmark_reports_speed_and_memory(sample, fake_job):
    source, model = sample
    fake_job["all_read"] = threading.Event()
    read = readings(1.0, 1.0, 5.5, 6.2, 3.0, all_read=fake_job["all_read"])
    result = benchmark.run_benchmark(source, model, read_memory=read)
    assert result["times_real_time"] == 30.0
    assert result["memory_before_gb"] == 1.0
    assert result["memory_peak_gb"] == 6.2
    assert result["memory_used_by_job_gb"] == pytest.approx(5.2)
    assert result["settings"] == ("cuda", "float16", 4)
    assert result["first_choice_settings"] == (("cuda", "float16", 4) == current_profile().attempts[0])


def test_benchmark_deletes_its_working_files(sample, fake_job):
    source, model = sample
    fake_job["all_read"] = threading.Event()
    fake_job["all_read"].set()
    benchmark.run_benchmark(source, model, read_memory=lambda: None)
    assert not fake_job["wav"].exists()
    assert not fake_job["wav"].parent.exists()


def test_benchmark_prints_numbers_only(sample, fake_job, monkeypatch, capsys):
    source, model = sample
    fake_job["all_read"] = threading.Event()
    fake_job["all_read"].set()
    monkeypatch.setattr(benchmark, "graphics_memory_used_gb", lambda: None)
    assert benchmark.main([str(source), "--model", str(model)]) == 0
    printed = capsys.readouterr()
    assert "30.0x real time" in printed.out
    assert SECRET not in printed.out + printed.err
    assert "private name" not in printed.out + printed.err


def test_report_mentions_fallbacks():
    text = benchmark.report({
        "audio_minutes": 5.0, "minutes": 1.0, "times_real_time": 5.0, "settings": ("cpu", "int8", 1),
        "first_choice_settings": False, "memory_before_gb": None, "memory_peak_gb": None,
        "memory_used_by_job_gb": None, "card_problem": True,
    })
    assert "processor, int8, batch 1" in text
    assert "lighter settings" in text
    assert "couldn't run the model" in text
    assert "Graphics memory" not in text


def test_unreadable_sample_is_reported_without_its_name(sample, monkeypatch, capsys):
    source, model = sample

    def broken(source, destination):
        raise audio.AudioError(f"ffmpeg couldn't convert {source.name}: {source}: Invalid data")

    monkeypatch.setattr(audio, "convert_to_wav", broken)
    assert benchmark.main([str(source), "--model", str(model)]) == 1
    printed = capsys.readouterr()
    assert "couldn't be read or converted" in printed.err
    assert "private name" not in printed.out + printed.err


def test_report_warns_about_a_leftover_folder():
    text = benchmark.report({
        "audio_minutes": 5.0, "minutes": 1.0, "times_real_time": 5.0, "settings": ("cuda", "float16", 4),
        "first_choice_settings": True, "memory_before_gb": None, "memory_peak_gb": None,
        "memory_used_by_job_gb": None, "card_problem": False, "leftover_folder": "C:\\Temp\\bench-1",
    })
    assert "couldn't delete the temporary folder" in text


def test_missing_sample_is_reported(tmp_path, capsys):
    assert benchmark.main([str(tmp_path / "missing.wav"), "--model", str(tmp_path)]) == 1
    assert "Benchmark failed" in capsys.readouterr().err


def test_memory_peak_keeps_the_highest_reading():
    with benchmark.MemoryPeak(readings(2.0, 7.5, 3.0), interval=0.001) as memory:
        while memory.peak is None or memory.peak < 7.5:
            pass
    assert memory.peak == 7.5


def test_no_card_means_no_memory_reading(monkeypatch):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: None)
    assert benchmark.graphics_memory_used_gb() is None
