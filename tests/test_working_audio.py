import os
import shutil
import socket

import numpy as np
import pytest
import soundfile

from transcribe_local import engine
from transcribe_local.checkpoint import SegmentWriter, load_segments
from transcribe_local.pipeline import Cancelled, process, remove_working_audio

RATE = 16000
SEGMENT = {
    "start": 0.0,
    "end": 2.9,
    "text": "Hello.",
    "avg_logprob": -0.2,
    "compression_ratio": 1.2,
    "no_speech_prob": 0.01,
    "words": [],
}

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg isn't on PATH")


@pytest.fixture
def job(tmp_path, monkeypatch):
    # process() turns on the network block, so restore the originals afterward.
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    source = tmp_path / "interview.wav"
    soundfile.write(source, np.zeros(3 * RATE, dtype="float32"), RATE)
    model = tmp_path / "model"
    model.mkdir()
    return {"source": source, "output": tmp_path / "out", "work": tmp_path / "work", "model": model}


def run(job):
    return process(job["source"], job["output"], job["work"], job["model"])


def fake_transcribe(calls):
    def transcribe(wav_path, segments_path, model_path, duration, on_progress, on_notice=None):
        calls.append(wav_path)
        assert wav_path.exists()
        with SegmentWriter(segments_path) as writer:
            writer.append(SEGMENT)
        return {"compute_type": "float16", "batch_size": 8, "resumed_from": 0.0, "seconds": 1.0}

    return transcribe


def working_files(job, pattern):
    return list(job["work"].glob(f"*/{pattern}"))


def test_working_audio_is_deleted_after_a_finished_job(job, monkeypatch):
    monkeypatch.setattr(engine, "transcribe", fake_transcribe([]))
    result = run(job)
    assert result["working_audio_removed"] is True
    assert working_files(job, "audio.wav") == []


def test_only_the_working_copy_is_deleted(job, monkeypatch):
    monkeypatch.setattr(engine, "transcribe", fake_transcribe([]))
    run(job)
    assert job["source"].exists()
    assert (job["output"] / "interview.docx").exists()
    assert (job["output"] / "interview_review.csv").exists()
    assert [s["text"] for s in load_segments(working_files(job, "segments.jsonl")[0])] == ["Hello."]


def test_stopped_job_keeps_working_audio_to_resume(job, monkeypatch):
    def stopped(*args, **kwargs):
        raise Cancelled

    monkeypatch.setattr(engine, "transcribe", stopped)
    with pytest.raises(Cancelled):
        run(job)
    assert len(working_files(job, "audio.wav")) == 1


def test_failed_job_keeps_working_audio_to_retry(job, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("something broke")

    monkeypatch.setattr(engine, "transcribe", failed)
    with pytest.raises(RuntimeError):
        run(job)
    assert len(working_files(job, "audio.wav")) == 1


def test_failed_export_keeps_working_audio(job, monkeypatch):
    from transcribe_local import maxqda

    def broken_export(segments, path):
        raise OSError("disk full")

    monkeypatch.setattr(engine, "transcribe", fake_transcribe([]))
    monkeypatch.setattr(maxqda, "write_docx", broken_export)
    with pytest.raises(OSError):
        run(job)
    assert len(working_files(job, "audio.wav")) == 1


def test_finished_job_runs_again_without_retranscribing(job, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "transcribe", fake_transcribe(calls))
    run(job)
    result = run(job)
    assert len(calls) == 1
    assert result["segments"] == 1
    assert (job["output"] / "interview (2).docx").exists()
    assert working_files(job, "audio.wav") == []


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to delete a file that's open")
def test_file_that_cannot_be_deleted_is_reported_not_raised(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"data")
    with open(wav, "rb"):
        # Windows won't delete a file that's open.
        assert remove_working_audio(wav) is False
    assert remove_working_audio(wav) is True
    assert not wav.exists()


def test_removing_also_clears_a_partial_conversion(tmp_path):
    wav = tmp_path / "audio.wav"
    partial = tmp_path / "audio.part"
    partial.write_bytes(b"data")
    assert remove_working_audio(wav) is True
    assert not partial.exists()
