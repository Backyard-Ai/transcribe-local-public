"""Writes DOCX transcripts following MAXQDA's documented import rules.

From maxqda.com/help/import/transcripts:
- DOCX preserves formatting and is the recommended format.
- Timestamps belong at least at the end of each paragraph, and also inside a
  paragraph around passages that are hard to understand.

Transcripts carry no speaker labels. Automatic labels can be wrong, and a wrong
label is worse than none. Paragraph breaks don't imply a change of speaker.
"""

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX

from transcribe_local.review import build_passages

# Short paragraphs keep a timestamp close to every passage, and a pause is often
# where one person stops and another starts.
PARAGRAPH_PAUSE_SECONDS = 1.5
PARAGRAPH_MAX_SECONDS = 20.0

# DOCX can't store these, and one stray character from the model would otherwise
# fail the export on every retry.
_NOT_ALLOWED_IN_DOCX = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")


def format_timestamp(seconds: float) -> str:
    """MAXQDA's [h:mm:ss.x] format."""
    tenths = round(seconds * 10)
    return f"[{tenths // 36000}:{tenths % 36000 // 600:02d}:{tenths % 600 // 10:02d}.{tenths % 10}]"


def build_paragraphs(segments: list[dict]) -> list[list[dict]]:
    paragraphs = []
    current = []
    for passage in build_passages(segments):
        if current:
            paused = passage["start"] - current[-1]["end"] >= PARAGRAPH_PAUSE_SECONDS
            too_long = passage["start"] - current[0]["start"] >= PARAGRAPH_MAX_SECONDS
            if paused or too_long:
                paragraphs.append(current)
                current = []
        current.append(passage)
    if current:
        paragraphs.append(current)
    return paragraphs


def write_docx(segments: list[dict], path: Path) -> None:
    """Passages to check are highlighted, so the transcript's words stay untouched."""
    document = Document()
    for group in build_paragraphs(segments):
        paragraph = document.add_paragraph()
        for index, passage in enumerate(group):
            run = paragraph.add_run(_NOT_ALLOWED_IN_DOCX.sub("", passage["text"]))
            if passage["reasons"]:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            is_last = index == len(group) - 1
            if is_last or passage["reasons"]:
                paragraph.add_run(f" {format_timestamp(passage['end'])}")
            if not is_last:
                paragraph.add_run(" ")
    document.save(path)
