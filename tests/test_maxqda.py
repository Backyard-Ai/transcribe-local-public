import pytest
from docx import Document

from transcribe_local.maxqda import build_paragraphs, format_timestamp, write_docx
from transcribe_local.review import split_sentences


def segment(start, end, text="Words here.", **overrides):
    return {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": -0.2,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.01,
        "words": [],
        **overrides,
    }


def timed_words(*items):
    return [{"start": start, "end": end, "word": word, "probability": 0.9} for word, start, end in items]


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "[0:00:00.0]"),
        (10.5, "[0:00:10.5]"),
        (59.96, "[0:01:00.0]"),
        (3599.99, "[1:00:00.0]"),
        (45000.04, "[12:30:00.0]"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_segment_splits_into_sentences_with_their_own_times():
    words = timed_words((" Yes.", 0.0, 0.5), (" I", 0.6, 0.7), (" agree.", 0.7, 1.2))
    sentences = split_sentences(segment(0, 1.2, "Yes. I agree.", words=words))
    assert [(s["start"], s["end"], s["text"]) for s in sentences] == [
        (0.0, 0.5, "Yes."),
        (0.6, 1.2, "I agree."),
    ]


def test_long_pause_splits_unpunctuated_speech():
    words = timed_words((" well", 0.0, 0.5), (" so", 3.0, 3.5))
    assert [s["text"] for s in split_sentences(segment(0, 3.5, "well so", words=words))] == ["well", "so"]


def test_unpunctuated_speech_is_capped():
    words = timed_words(*[(" word", float(i), i + 0.9) for i in range(45)])
    sentences = split_sentences(segment(0, 45, "word " * 45, words=words))
    # A sentence ends with the word that reaches the cap, so it can run one word over.
    assert all(s["end"] - s["start"] <= 21.0 for s in sentences)
    assert len(sentences) > 1
    assert sum(len(s["text"].split()) for s in sentences) == 45


def test_no_words_are_lost_when_splitting():
    words = timed_words((" One.", 0.0, 1.0), (" Two", 1.1, 1.5), (" three?", 1.6, 2.0), (" Four", 5.0, 6.0))
    text = " ".join(s["text"] for s in split_sentences(segment(0, 6, "One. Two three? Four", words=words)))
    assert text == "One. Two three? Four"


def test_segment_without_word_times_is_kept_whole():
    sentence = split_sentences(segment(0, 2, "Hello there."))[0]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 2, "Hello there.")


def test_control_characters_do_not_break_the_export(tmp_path):
    path = tmp_path / "out.docx"
    write_docx([segment(0, 1, "Hello\x07 there\x00.")], path)
    assert Document(path).paragraphs[0].text == "Hello there. [0:00:01.0]"


def test_pause_starts_new_paragraph():
    assert len(build_paragraphs([segment(0, 1), segment(3, 4)])) == 2


def test_short_gap_stays_in_same_paragraph():
    assert len(build_paragraphs([segment(0, 1), segment(1.5, 2)])) == 1


def test_long_stretch_without_pauses_is_broken_up():
    groups = build_paragraphs([segment(i * 10, i * 10 + 10) for i in range(5)])
    assert [len(group) for group in groups] == [2, 2, 1]


def test_docx_has_no_speaker_labels_and_ends_paragraphs_with_timestamps(tmp_path):
    path = tmp_path / "out.docx"
    write_docx([segment(0, 1.5, "Hello."), segment(1.5, 3, "Again."), segment(9, 10, "Later.")], path)
    assert [p.text for p in Document(path).paragraphs] == ["Hello. Again. [0:00:03.0]", "Later. [0:00:10.0]"]


def test_flagged_passage_is_highlighted_and_timestamped_without_changing_words(tmp_path):
    path = tmp_path / "flagged.docx"
    write_docx([segment(0, 1, "Unclear.", avg_logprob=-2.0), segment(1, 2, "Clear.")], path)
    paragraph = Document(path).paragraphs[0]
    assert paragraph.text == "Unclear. [0:00:01.0] Clear. [0:00:02.0]"
    highlighted = [run.text for run in paragraph.runs if run.font.highlight_color is not None]
    assert highlighted == ["Unclear."]


def test_timestamps_never_go_backwards(tmp_path):
    # MAXQDA removes any timestamp earlier than the one before it.
    path = tmp_path / "order.docx"
    segments = [segment(i * 2.0, i * 2.0 + 1.9, avg_logprob=-2.0 if i % 2 else -0.2) for i in range(20)]
    write_docx(segments, path)
    text = " ".join(p.text for p in Document(path).paragraphs)
    stamps = [part.split("]")[0] for part in text.split("[")[1:]]
    seconds = [sum(float(n) * m for n, m in zip(s.split(":"), (3600, 60, 1))) for s in stamps]
    assert seconds == sorted(seconds)
