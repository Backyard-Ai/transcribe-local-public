import numpy as np
import soundfile

from transcribe_local.coverage import find_untranscribed_audio
from transcribe_local.output import write_review

RATE = 16000


def write_wav(path, passages):
    """passages: (seconds, amplitude) in order, as noise at that amplitude."""
    rng = np.random.default_rng(0)
    audio = np.concatenate([rng.normal(0, amplitude, int(seconds * RATE)) for seconds, amplitude in passages])
    soundfile.write(path, audio.astype("float32"), RATE)


def transcript(start, end):
    return {"start": start, "end": end, "text": "x", "words": [{"start": start, "end": end, "word": "x", "probability": 1.0}]}


def test_loud_passage_without_words_is_reported(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, [(10, 0.001), (5, 0.1), (10, 0.001)])
    gaps = find_untranscribed_audio(wav, [])
    assert len(gaps) == 1
    assert gaps[0][0] == 10.0
    assert gaps[0][1] == 15.0


def test_loud_passage_with_words_is_not_reported(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, [(10, 0.001), (5, 0.1), (10, 0.001)])
    assert find_untranscribed_audio(wav, [transcript(10.0, 15.0)]) == []


def test_partly_transcribed_passage_reports_only_the_rest(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, [(10, 0.001), (10, 0.1), (10, 0.001)])
    gaps = find_untranscribed_audio(wav, [transcript(10.0, 14.9)])
    assert gaps == [(15.0, 20.0)]


def test_steady_background_is_not_reported(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, [(30, 0.01)])
    assert find_untranscribed_audio(wav, []) == []


def test_very_quiet_blip_in_clean_recording_is_not_reported(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, [(20, 0.00001), (3, 0.0001), (20, 0.00001)])
    assert find_untranscribed_audio(wav, []) == []


def test_brief_sound_is_not_reported(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, [(10, 0.001), (0.4, 0.1), (10, 0.001)])
    assert find_untranscribed_audio(wav, []) == []


def test_review_lists_everything_in_time_order(tmp_path):
    path = tmp_path / "review.csv"
    flagged = [{"start": 30.0, "end": 31.0, "text": "unclear", "reasons": ["low confidence"]}]
    write_review(flagged, [(5.0, 8.0), (40.0, 42.0)], path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[1:] == [
        "0:00:05,0:00:08,sound with no transcript,",
        "0:00:30,0:00:31,low confidence,unclear",
        "0:00:40,0:00:42,sound with no transcript,",
    ]
