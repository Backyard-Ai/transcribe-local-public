import json
from pathlib import Path

import pytest

from transcribe_local import safety
from transcribe_local.checkpoint import SegmentWriter, load_segments
from transcribe_local.output import format_time
from transcribe_local.review import build_passages


def segment(text="hello there", **overrides):
    return {
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "avg_logprob": -0.2,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.01,
        "words": [{"start": 0.0, "end": 1.0, "word": text, "probability": 0.9}],
        **overrides,
    }


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "segments.jsonl"
    with SegmentWriter(path) as writer:
        writer.append(segment("one"))
        writer.append(segment("two"))
    assert [s["text"] for s in load_segments(path)] == ["one", "two"]


def test_checkpoint_drops_line_cut_off_by_crash(tmp_path):
    path = tmp_path / "segments.jsonl"
    with SegmentWriter(path) as writer:
        writer.append(segment("one"))
    with open(path, "a", encoding="utf-8") as file:
        file.write('{"start": 1.0, "text": "cut o')
    assert [s["text"] for s in load_segments(path)] == ["one"]

    with SegmentWriter(path) as writer:
        writer.append(segment("two"))
    assert [s["text"] for s in load_segments(path)] == ["one", "two"]


def test_checkpoint_drops_complete_json_without_newline(tmp_path):
    path = tmp_path / "segments.jsonl"
    path.write_text(json.dumps(segment("one")) + "\n" + json.dumps(segment("two")), encoding="utf-8")
    assert [s["text"] for s in load_segments(path)] == ["one"]


def test_checkpoint_missing_file_is_empty(tmp_path):
    assert load_segments(tmp_path / "none.jsonl") == []


def test_confident_segment_is_not_flagged():
    assert build_passages([segment()])[0]["reasons"] == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"avg_logprob": -1.5}, "low confidence"),
        ({"compression_ratio": 3.0}, "repetitive text"),
        ({"no_speech_prob": 0.9}, "may not be speech"),
        ({"words": [{"start": 0, "end": 1, "word": "x", "probability": 0.1}]}, "uncertain words"),
    ],
)
def test_flag_reasons(overrides, reason):
    assert reason in build_passages([segment(**overrides)])[0]["reasons"]


def test_repeated_segment_is_flagged_ignoring_case():
    passages = build_passages([segment("Thank you."), segment("thank you.")])
    assert [p["reasons"] for p in passages] == [[], ["repeats previous segment"]]


def test_garbled_sentence_is_flagged_among_clear_ones():
    clear = [{"start": i, "end": i + 0.9, "word": " clear", "probability": 0.95} for i in range(30)]
    clear[-1]["word"] = " clear."
    garbled = [{"start": 30 + i, "end": 30.9 + i, "word": " mumble", "probability": 0.1} for i in range(3)]
    garbled[-1]["word"] = " mumble."
    words = clear + garbled
    whole_segment_average = sum(w["probability"] for w in words) / len(words)
    assert whole_segment_average > 0.5
    passages = build_passages([segment("text", start=0, end=33, words=words)])
    assert passages[-1]["text"] == "mumble mumble mumble."
    assert passages[-1]["reasons"] == ["uncertain words"]
    assert all(p["reasons"] == [] for p in passages[:-1])


def test_whole_segment_problem_covers_each_sentence():
    words = [
        {"start": 0, "end": 1, "word": " One.", "probability": 0.9},
        {"start": 1, "end": 2, "word": " Two.", "probability": 0.9},
    ]
    passages = build_passages([segment("One. Two.", words=words, compression_ratio=3.0)])
    assert [p["reasons"] for p in passages] == [["repetitive text"], ["repetitive text"]]


def test_blank_text_makes_no_passage():
    assert build_passages([segment("", words=[])]) == []


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "0:00:00"), (65.9, "0:01:05"), (3661, "1:01:01"), (45000, "12:30:00")],
)
def test_format_time(seconds, expected):
    assert format_time(seconds) == expected


def test_cloud_folders_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(safety, "_volume_label", lambda path: "")
    monkeypatch.setattr(safety, "_google_drive_roots", lambda: [tmp_path / "synced"])
    monkeypatch.setattr(safety, "_onedrive_roots", lambda: [])
    with pytest.raises(safety.CloudPathError):
        safety.require_local(tmp_path / "synced" / "interview.m4a", "the recording")
    safety.require_local(tmp_path / "local" / "interview.m4a", "the recording")


def test_google_drive_volume_is_refused(monkeypatch):
    monkeypatch.setattr(safety, "_volume_label", lambda path: "someone@example.com - Google Drive")
    monkeypatch.setattr(safety, "_google_drive_roots", lambda: [])
    monkeypatch.setattr(safety, "_onedrive_roots", lambda: [])
    with pytest.raises(safety.CloudPathError):
        safety.require_local(Path("G:/My Drive/interview.m4a"), "the recording")
