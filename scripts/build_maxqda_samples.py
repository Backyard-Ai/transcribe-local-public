"""Builds a MAXQDA import test file from a finished job.

One sentence is marked for review on purpose, whatever its real confidence, to
test how the highlight and the extra timestamp import. Never use this on a real recording.

Usage: build_maxqda_samples.py <segments.jsonl> <output folder>
"""

import sys
from pathlib import Path

from transcribe_local.checkpoint import load_segments
from transcribe_local.maxqda import write_docx
from transcribe_local.review import split_sentences

segments_path = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

sentences = [
    {**segment, **sentence, "words": []}
    for segment in load_segments(segments_path)
    for sentence in split_sentences(segment)
]
sentences[len(sentences) // 2]["avg_logprob"] = -2.0
write_docx(sentences, output / "TEST_timestamps_and_highlight.docx")
print(f"Wrote a test file with {len(sentences)} sentences to {output}")
