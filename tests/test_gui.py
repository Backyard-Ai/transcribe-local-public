import threading
import time
import tkinter as tk
from pathlib import Path

import pytest

from transcribe_local import gui
from transcribe_local.gui import App
from transcribe_local.pipeline import Cancelled

RESULT = {"audio_minutes": 60.0, "flagged_passages": 2, "sound_with_no_transcript": 1}


# Like the real program, tests use one toolkit instance for their whole life. Creating
# and destroying instances while worker threads run lets Python clean up an old
# window's objects from the wrong thread, which breaks the next window to open.
_apps_kept_alive = []


@pytest.fixture(scope="session")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def make_app(tk_root):
    apps = []

    def build(run_job, check_battery=lambda: False):
        window = tk.Toplevel(tk_root)
        window.withdraw()
        app = App(
            window, run_job=run_job, describe_computer=lambda: "Using: test computer", check_battery=check_battery
        )
        apps.append(app)
        _apps_kept_alive.append(app)
        return app

    yield build
    for app in apps:
        app.root.after_cancel(app.poll_job)
        app.root.after_cancel(app.power_job)
        app.root.destroy()


def pump(app, until, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.root.update()
        if until():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the window")


def finished_ok(path, output, on_stage, on_progress, on_notice=None):
    on_stage("Converting the audio")
    on_progress(30.0, 60.0)
    return RESULT


def test_job_runs_and_shows_a_summary(make_app):
    app = make_app(finished_ok)
    path = Path("interview.m4a")
    app.add_paths([path])
    app.start()
    pump(app, lambda: app.statuses[path].startswith("Done"))
    pump(app, lambda: not app.running)
    assert app.statuses[path] == "Done: 60 min of audio, 3 passages to check"
    assert str(app.start_button["state"]) == "disabled"


def test_short_recording_is_not_shown_as_zero_minutes(make_app):
    app = make_app(lambda path, output, on_stage, on_progress, on_notice=None: {**RESULT, "audio_minutes": 0.47, "flagged_passages": 0, "sound_with_no_transcript": 1})
    path = Path("clip.m4a")
    app.add_paths([path])
    app.start()
    pump(app, lambda: app.statuses[path].startswith("Done"))
    assert app.statuses[path] == "Done: under 1 min of audio, 1 passage to check"


def test_damaged_audio_is_called_out(make_app):
    app = make_app(lambda path, output, on_stage, on_progress, on_notice=None: {**RESULT, "read_errors": "Invalid data"})
    path = Path("damaged.mp3")
    app.add_paths([path])
    app.start()
    pump(app, lambda: app.statuses[path].startswith("Done"))
    assert "couldn't be read" in app.statuses[path]


def test_add_folder_finds_recordings_in_subfolders_and_skips_other_files(make_app, tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "top.mp3").touch()
    (tmp_path / "one" / "a.m4a").touch()
    (tmp_path / "one" / "a.txt").touch()
    (tmp_path / "two" / "b.M4A").touch()
    app = make_app(finished_ok)
    app.add_folder_path(tmp_path)
    assert sorted(path.name for path in app.files) == ["a.m4a", "b.M4A", "top.mp3"]
    assert app.stage.get() == "Added 3 recordings."


def test_add_folder_says_when_nothing_was_found(make_app, tmp_path):
    (tmp_path / "notes.txt").touch()
    app = make_app(finished_ok)
    app.add_folder_path(tmp_path)
    assert app.files == []
    assert app.stage.get() == "No recordings found in that folder."


def test_window_shows_which_hardware_it_uses(make_app):
    app = make_app(finished_ok)
    pump(app, lambda: app.computer.get() == "Using: test computer")


def test_battery_warning_shows_only_on_battery(make_app):
    from transcribe_local.hardware import BATTERY_WARNING

    assert make_app(finished_ok, check_battery=lambda: True).power.get() == BATTERY_WARNING
    assert make_app(finished_ok, check_battery=lambda: False).power.get() == ""
    assert make_app(finished_ok, check_battery=lambda: None).power.get() == ""


def test_failed_battery_check_does_not_stop_the_window(make_app):
    def broken():
        raise OSError("no power information")

    assert make_app(finished_ok, check_battery=broken).power.get() == ""


def test_switch_to_the_processor_is_shown(make_app):
    def run_job(path, output, on_stage, on_progress, on_notice=None):
        on_notice("The graphics card couldn't run the model.")
        return {**RESULT, "last_run": {"device": "cpu", "card_problem": "no kernel image"}}

    app = make_app(run_job)
    pump(app, lambda: app.computer.get() == "Using: test computer")
    path = Path("interview.m4a")
    app.add_paths([path])
    app.start()
    pump(app, lambda: app.statuses[path].startswith("Done") and not app.running)
    assert app.computer.get() == "Using: processor. The graphics card couldn't run the model."
    assert "made on the processor" in app.statuses[path]


def test_late_hardware_check_does_not_hide_a_switch_to_the_processor(make_app):
    app = make_app(finished_ok)
    pump(app, lambda: app.computer.get() == "Using: test computer")
    app.messages.put(("notice", None, "Using: processor. Card failed."))
    app.messages.put(("computer", None, "Using: the card"))
    pump(app, lambda: app.messages.empty())
    assert app.computer.get() == "Using: processor. Card failed."


def test_undeleted_working_audio_is_called_out(make_app):
    app = make_app(lambda path, output, on_stage, on_progress, on_notice=None: {**RESULT, "working_audio_removed": False})
    path = Path("locked.m4a")
    app.add_paths([path])
    app.start()
    pump(app, lambda: app.statuses[path].startswith("Done"))
    assert "couldn't delete the working audio copy" in app.statuses[path]


@pytest.mark.parametrize("folder", ["", "   ", "relative\\folder"])
def test_output_folder_must_be_a_full_path(make_app, folder):
    calls = []
    app = make_app(lambda path, output, on_stage, on_progress, on_notice=None: calls.append(output) or RESULT)
    app.add_paths([Path("x.m4a")])
    app.output.set(folder)
    app.start()
    assert not app.running
    assert calls == []
    assert "full folder path" in app.stage.get()


def test_window_retries_when_the_toolkit_fails_to_start(monkeypatch):
    attempts = []

    def flaky_tk():
        attempts.append(1)
        if len(attempts) < 3:
            raise tk.TclError("Can't find a usable init.tcl")
        return "window"

    monkeypatch.setattr(gui.tk, "Tk", flaky_tk)
    assert gui.open_window(delay_seconds=0) == "window"


def test_window_reports_failure_after_retries(monkeypatch):
    def broken_tk():
        raise tk.TclError("Can't find a usable init.tcl")

    monkeypatch.setattr(gui.tk, "Tk", broken_tk)
    with pytest.raises(tk.TclError):
        gui.open_window(delay_seconds=0)


def test_finish_message_mentions_failures(make_app):
    def run_job(path, output, on_stage, on_progress, on_notice=None):
        raise RuntimeError("could not read it")

    app = make_app(run_job)
    app.add_paths([Path("bad.m4a")])
    app.start()
    pump(app, lambda: not app.running and app.worker is None)
    assert app.stage.get() == "Finished, but 1 file failed. See the Status column."


def test_failed_file_does_not_stop_the_queue(make_app):
    def run_job(path, output, on_stage, on_progress, on_notice=None):
        if path.name == "bad.m4a":
            raise RuntimeError("could not read it")
        return RESULT

    app = make_app(run_job)
    bad, good = Path("bad.m4a"), Path("good.m4a")
    app.add_paths([bad, good])
    app.start()
    pump(app, lambda: app.statuses[good].startswith("Done"))
    assert app.statuses[bad] == "Failed: could not read it"


def test_unexpected_error_does_not_freeze_the_window(make_app):
    def run_job(path, output, on_stage, on_progress, on_notice=None):
        raise KeyError("surprise")

    app = make_app(run_job)
    path = Path("x.m4a")
    app.add_paths([path])
    app.start()
    pump(app, lambda: app.statuses[path].startswith("Failed") and not app.running)


def test_stop_cancels_and_failed_or_stopped_files_can_restart(make_app):
    started = threading.Event()

    def run_job(path, output, on_stage, on_progress, on_notice=None):
        started.set()
        while True:
            on_progress(10.0, 60.0)
            time.sleep(0.01)

    app = make_app(run_job)
    path = Path("long.m4a")
    app.add_paths([path])
    app.start()
    pump(app, started.is_set)
    app.stop()
    pump(app, lambda: not app.running and "resume" in app.statuses[path])
    assert str(app.start_button["state"]) == "normal"


def test_finished_files_are_not_run_again(make_app):
    calls = []

    def run_job(path, output, on_stage, on_progress, on_notice=None):
        calls.append(path.name)
        return RESULT

    app = make_app(run_job)
    first = Path("first.m4a")
    app.add_paths([first])
    app.start()
    pump(app, lambda: app.statuses[first].startswith("Done") and not app.running)
    second = Path("second.m4a")
    app.add_paths([second])
    app.start()
    pump(app, lambda: app.statuses[second].startswith("Done") and not app.running)
    assert calls == ["first.m4a", "second.m4a"]


def test_duplicate_files_are_added_once(make_app):
    app = make_app(finished_ok)
    app.add_paths([Path("a.m4a"), Path("a.m4a")])
    assert len(app.files) == 1


def test_cancelled_is_not_reported_as_failure(make_app):
    def run_job(path, output, on_stage, on_progress, on_notice=None):
        raise Cancelled

    app = make_app(run_job)
    path = Path("x.m4a")
    app.add_paths([path])
    app.start()
    pump(app, lambda: not app.running and "resume" in app.statuses[path])
    assert not app.statuses[path].startswith("Failed")
