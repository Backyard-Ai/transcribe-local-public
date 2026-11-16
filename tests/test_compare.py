import io
import random

import pytest

from transcribe_local.compare import (
    KEY_NAME,
    align,
    app_words,
    find_app_transcript,
    find_pairs,
    read_text,
    reference_words,
    report,
    score,
)


def counts(reference, hypothesis):
    result = score(reference.split(), hypothesis.split(), [0.0] * len(hypothesis.split()))
    return result.wrong, result.missed, result.added


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        ("a b c", "a b c", (0, 0, 0)),
        ("a b c", "a x c", (1, 0, 0)),
        ("a b c", "a c", (0, 1, 0)),
        ("a c", "a b c", (0, 0, 1)),
        ("", "a b", (0, 0, 2)),
        ("a b", "", (0, 2, 0)),
        ("", "", (0, 0, 0)),
        ("a b c d e", "x y", (2, 3, 0)),
    ],
)
def test_counts_differences(reference, hypothesis, expected):
    assert counts(reference, hypothesis) == expected


def _edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def test_alignment_matches_a_plain_reference_implementation():
    rng = random.Random(7)
    for _ in range(300):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 25))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 25))]
        operations = align(a, b)
        kinds = [kind for kind, _ in operations]
        assert len(kinds) - kinds.count("match") == _edit_distance(a, b)
        assert kinds.count("match") + kinds.count("wrong") + kinds.count("missed") == len(a)
        assert kinds.count("match") + kinds.count("wrong") + kinds.count("added") == len(b)


def test_alignment_handles_hours_of_speech_quickly():
    rng = random.Random(1)
    words = [str(rng.randint(0, 500)) for _ in range(6000)]
    assert sum(kind == "match" for kind, _ in align(words, words)) == 6000


def test_accuracy_percentage():
    assert score("a b c d".split(), "a x c d".split(), [0.0] * 4).accuracy == 75.0


def test_accuracy_never_goes_below_zero():
    assert score(["a"], "x y z w".split(), [0.0] * 4).accuracy == 0.0


def test_fillers_can_be_ignored():
    result = score("well um i think".split(), "well i think".split(), [0.0] * 3)
    assert result.missed == 1
    assert result.accuracy == 75.0
    assert result.accuracy_without_fillers == 100.0


def test_long_missed_stretch_is_located():
    reference = "start " + "gone " * 12 + "end"
    result = score(reference.split(), ["start", "end"], [10.0, 300.0])
    assert result.long_missed == [(12, 300.0)]


def test_long_added_stretch_is_located():
    hypothesis = ["start"] + ["extra"] * 11 + ["end"]
    result = score(["start", "end"], hypothesis, [60.0] * 13)
    assert result.long_added == [(11, 60.0)]


def test_short_stretches_are_not_listed():
    result = score("a b c d".split(), ["a"], [0.0])
    assert result.long_missed == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello there.", ["hello", "there"]),
        ("Speaker 1: Hello there.", ["hello", "there"]),
        ("Dr. Lee: Hello there.", ["hello", "there"]),
        ("Lisa B.: Hello", ["hello"]),
        ("[00:01:23] Hello", ["hello"]),
        ("00:01:23 Hello", ["hello"]),
        ("#00:01:23-4# Hello", ["hello"]),
        ("Hello [0:00:10.5]", ["hello"]),
        ("Speaker 1 00:12:03\nHello", ["hello"]),
        ("Interviewer\nHello", ["hello"]),
        ("1\n00:00:01,000 --> 00:00:04,000\nHello", ["hello"]),
        ("I don’t know", ["i", "don't", "know"]),
        ("So here is the whole thing: it works", ["so", "here", "is", "the", "whole", "thing", "it", "works"]),
    ],
)
def test_existing_transcript_formats(text, expected):
    assert reference_words(text) == expected


def test_app_transcript_words_keep_their_times():
    words, times = app_words("[0:00:16] So I slid.\n[1:02:03] Looks good.\n")
    assert words == ["so", "i", "slid", "looks", "good"]
    assert times == [16.0, 16.0, 16.0, 3723.0, 3723.0]


@pytest.mark.parametrize(
    "encoded",
    ["café time".encode("utf-8"), "café time".encode("utf-8-sig"), "café time".encode("utf-16"), "café time".encode("cp1252")],
)
def test_common_text_encodings_are_read(tmp_path, encoded):
    path = tmp_path / "t.txt"
    path.write_bytes(encoded)
    assert read_text(path) == "café time"


def test_pairs_recording_with_only_text_in_its_folder(tmp_path):
    folder = tmp_path / "study"
    folder.mkdir()
    (folder / "audio.m4a").touch()
    (folder / "different name.txt").touch()
    assert find_pairs(tmp_path) == [(folder / "audio.m4a", folder / "different name.txt")]


def test_pairs_by_name_when_a_folder_has_several(tmp_path):
    for name in ("a.m4a", "a.txt", "b.mp3", "b.txt"):
        (tmp_path / name).touch()
    assert find_pairs(tmp_path) == [(tmp_path / "a.m4a", tmp_path / "a.txt"), (tmp_path / "b.mp3", tmp_path / "b.txt")]


def test_recording_without_a_transcript_is_skipped(tmp_path):
    (tmp_path / "alone.m4a").touch()
    assert find_pairs(tmp_path) == []


def test_ambiguous_folder_is_not_guessed(tmp_path):
    for name in ("a.m4a", "b.m4a", "notes.txt"):
        (tmp_path / name).touch()
    assert find_pairs(tmp_path) == []


def test_finds_newest_rerun_and_ignores_other_recordings(tmp_path):
    import os

    older = tmp_path / "a.txt"
    newer = tmp_path / "a (2).txt"
    other = tmp_path / "a b.txt"
    for path, when in ((older, 100), (newer, 200), (other, 300)):
        path.touch()
        os.utime(path, (when, when))
    assert find_app_transcript(tmp_path, "a") == newer
    assert find_app_transcript(tmp_path, "missing") is None


@pytest.fixture
def study(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    folder = input_dir / "SecretFolderName"
    folder.mkdir(parents=True)
    (folder / "SecretParticipant.m4a").touch()
    (folder / "SecretExisting.txt").write_text("Interviewer: my confidential zebra story", encoding="utf-8")
    (output_dir / "SecretParticipant.txt").write_text("[0:00:05] my confidential giraffe story\n", encoding="utf-8")
    second = input_dir / "SecondFolder"
    second.mkdir()
    (second / "SecretNotDoneYet.m4a").touch()
    (second / "SecretNotes.txt").write_text("words", encoding="utf-8")
    return input_dir, output_dir


def test_report_scores_each_recording(study):
    out = io.StringIO()
    assert report(*study, out=out) == 0
    text = out.getvalue()
    # Folders are taken in alphabetical order.
    assert "Recording 1: not transcribed yet" in text
    assert "Recording 2: 4 words in existing transcript, 4 from the app" in text
    assert "Accuracy: 75.0%" in text
    assert "1 wrong, 0 missed, 0 added" in text
    assert "1 recording still needs transcribing" in text


def test_report_never_shows_words_or_file_names(study):
    out = io.StringIO()
    report(*study, out=out)
    text = out.getvalue().lower()
    for private in ("zebra", "giraffe", "confidential", "story", "secret", "secondfolder"):
        assert private not in text


def test_key_links_labels_to_files_on_this_computer_only(study):
    input_dir, output_dir = study
    report(input_dir, output_dir, out=io.StringIO())
    key = (output_dir / KEY_NAME).read_text(encoding="utf-8-sig")
    assert "Recording 1" in key
    assert "SecretParticipant.m4a" in key


def test_report_warns_when_word_counts_are_far_apart(tmp_path):
    (tmp_path / "input" / "x").mkdir(parents=True)
    (tmp_path / "output").mkdir()
    (tmp_path / "input" / "x" / "a.m4a").touch()
    (tmp_path / "input" / "x" / "ref.txt").write_text("one two three four five six", encoding="utf-8")
    (tmp_path / "output" / "a.txt").write_text("[0:00:01] one two\n", encoding="utf-8")
    out = io.StringIO()
    report(tmp_path / "input", tmp_path / "output", out=out)
    assert "may not be the same recording" in out.getvalue()


def test_report_says_when_there_is_nothing_to_compare(tmp_path):
    (tmp_path / "input").mkdir()
    out = io.StringIO()
    assert report(tmp_path / "input", tmp_path / "output", out=out) == 1
    assert "No recordings" in out.getvalue()


def test_report_refuses_cloud_folders(study, monkeypatch):
    from transcribe_local import compare, safety

    def refuse(path, purpose):
        raise safety.CloudPathError("synced")

    monkeypatch.setattr(compare, "require_local", refuse)
    with pytest.raises(safety.CloudPathError):
        report(*study, out=io.StringIO())
