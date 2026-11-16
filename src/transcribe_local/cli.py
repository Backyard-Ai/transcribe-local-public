"""Command line entry point. Prints progress and counts, never transcript text."""

import argparse
import json
import sys
from pathlib import Path

from transcribe_local.pipeline import DEFAULT_MODEL, DEFAULT_OUTPUT, DEFAULT_WORK, TimeLeft, keep_awake, process


def _progress_printer(time_left: TimeLeft):
    def show(position: float, duration: float) -> None:
        remaining = time_left.minutes(position, duration)
        estimate = f"about {remaining:.0f} min left" if remaining is not None else "working out the time left"
        percent = min(100, position / duration * 100) if duration else 100
        print(f"\r  {percent:5.1f}%  {estimate}   ", end="", file=sys.stderr, flush=True)

    return show


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcribe-local", description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="recordings to transcribe, in order")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    failures = 0
    keep_awake(True)
    try:
        for source in args.files:
            print(f"{source.name}", file=sys.stderr)
            try:
                time_left = TimeLeft()

                def notice(text: str, time_left=time_left) -> None:
                    time_left.restart()
                    print(f"\n  NOTE: {text}", file=sys.stderr)

                result = process(
                    source,
                    args.output,
                    args.work,
                    args.model,
                    on_progress=_progress_printer(time_left),
                    on_notice=notice,
                )
                print(file=sys.stderr)
                print(json.dumps(result, indent=2))
            except Exception as error:  # One bad file shouldn't stop the rest of the queue.
                failures += 1
                print(f"  Failed: {error}", file=sys.stderr)
    finally:
        keep_awake(False)
    return 1 if failures else 0
