"""The desktop window. Shows progress and counts, never transcript text."""

import ctypes
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from transcribe_local.audio import SUPPORTED_EXTENSIONS, recordings_in
from transcribe_local.pipeline import DEFAULT_OUTPUT, Cancelled, TimeLeft, keep_awake, process

POLL_MILLISECONDS = 100
# Plugging in or unplugging is noticed within this time.
POWER_CHECK_MILLISECONDS = 30_000


def _format_remaining(minutes: float) -> str:
    if minutes < 1:
        return "under a minute left"
    if minutes < 90:
        return f"about {minutes:.0f} min left"
    return f"about {minutes / 60:.1f} hours left"


def _summary(result: dict) -> str:
    to_check = result["flagged_passages"] + result["sound_with_no_transcript"]
    passages = "1 passage" if to_check == 1 else f"{to_check} passages"
    length = f"{result['audio_minutes']:.0f} min" if result["audio_minutes"] >= 1 else "under 1 min"
    summary = f"{length} of audio, {passages} to check"
    if result.get("read_errors"):
        summary += ". WARNING: part of the audio couldn't be read"
    if not result.get("working_audio_removed", True):
        summary += ". WARNING: couldn't delete the working audio copy"
    if result.get("last_run", {}).get("card_problem"):
        summary += ". NOTE: made on the processor, because the graphics card couldn't run the model"
    return summary


def _describe_computer() -> str:
    from transcribe_local.hardware import current_profile

    profile = current_profile()
    notes = [note for note in (profile.warning, profile.first_run_note) if note]
    return " ".join([f"Using: {profile.description}."] + notes)


def _check_battery() -> bool | None:
    from transcribe_local.hardware import on_battery

    return on_battery()


class App:
    def __init__(self, root: tk.Tk, run_job=process, describe_computer=_describe_computer, check_battery=_check_battery):
        self.root = root
        self.run_job = run_job
        self.check_battery = check_battery
        self.computer = tk.StringVar(value="Checking this computer…")
        self.power = tk.StringVar(value="")
        self.notice_shown = False
        self.files: list[Path] = []
        self.statuses: dict[Path, str] = {}
        self.messages: queue.Queue = queue.Queue()
        self.stop_requested = threading.Event()
        self.worker: threading.Thread | None = None
        self.output = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.stage = tk.StringVar(value="Add recordings to get started.")
        self.progress = tk.DoubleVar(value=0)

        root.title("Transcribe Local")
        root.geometry("820x520")
        root.minsize(560, 440)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._refresh()
        self._poll()
        self._check_power()
        # Checking the hardware loads the graphics libraries, which takes a few seconds,
        # so it runs alongside the window instead of delaying it.
        threading.Thread(target=self._check_computer, args=(describe_computer,), daemon=True).start()

    def _check_computer(self, describe_computer) -> None:
        try:
            text = describe_computer()
        except Exception as error:
            text = f"Couldn't check this computer: {error}"
        self.messages.put(("computer", None, text))

    def _check_power(self) -> None:
        from transcribe_local.hardware import BATTERY_WARNING

        try:
            self.power.set(BATTERY_WARNING if self.check_battery() else "")
        except Exception:  # A failed check shouldn't stop the window; it just shows nothing.
            self.power.set("")
        self.power_job = self.root.after(POWER_CHECK_MILLISECONDS, self._check_power)

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Recordings").grid(row=0, column=0, sticky="w")
        columns = ("file", "status")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        self.table.heading("file", text="File")
        self.table.heading("status", text="Status")
        self.table.column("file", width=250, anchor="w")
        self.table.column("status", width=520, anchor="w")
        self.table.grid(row=1, column=0, sticky="nsew", pady=(4, 6))

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky="w")
        self.add_button = ttk.Button(buttons, text="Add recordings…", command=self.add_files)
        self.add_button.pack(side="left")
        self.add_folder_button = ttk.Button(buttons, text="Add folder…", command=self.add_folder)
        self.add_folder_button.pack(side="left", padx=(6, 0))
        self.remove_button = ttk.Button(buttons, text="Remove selected", command=self.remove_selected)
        self.remove_button.pack(side="left", padx=6)

        ttk.Label(frame, text="Save transcripts to").grid(row=3, column=0, sticky="w", pady=(14, 2))
        folder = ttk.Frame(frame)
        folder.grid(row=4, column=0, sticky="ew")
        folder.columnconfigure(0, weight=1)
        self.output_entry = ttk.Entry(folder, textvariable=self.output)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        self.browse_button = ttk.Button(folder, text="Browse…", command=self.choose_output)
        self.browse_button.grid(row=0, column=1, padx=(6, 0))
        self.open_button = ttk.Button(folder, text="Open", command=self.open_output)
        self.open_button.grid(row=0, column=2, padx=(6, 0))

        ttk.Progressbar(frame, variable=self.progress, maximum=100).grid(row=5, column=0, sticky="ew", pady=(16, 4))
        ttk.Label(frame, textvariable=self.stage).grid(row=6, column=0, sticky="w")

        ttk.Label(frame, textvariable=self.computer, foreground="#555555", wraplength=520).grid(
            row=7, column=0, sticky="w", pady=(12, 0)
        )
        actions = ttk.Frame(frame)
        actions.grid(row=7, column=0, sticky="e", pady=(12, 0))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop)
        self.stop_button.pack(side="left")
        self.start_button = ttk.Button(actions, text="Start", command=self.start)
        self.start_button.pack(side="left", padx=(6, 0))
        ttk.Label(frame, textvariable=self.power, foreground="#9A3412", wraplength=520).grid(
            row=8, column=0, sticky="w", pady=(4, 0)
        )

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _refresh(self) -> None:
        self.table.delete(*self.table.get_children())
        for index, path in enumerate(self.files):
            self.table.insert("", "end", iid=str(index), values=(path.name, self.statuses.get(path, "Waiting")))
        idle = "disabled" if self.running else "normal"
        for widget in (self.add_button, self.add_folder_button, self.remove_button, self.browse_button, self.output_entry):
            widget.configure(state=idle)
        pending = any(not self.statuses.get(path, "").startswith("Done") for path in self.files)
        self.start_button.configure(state="normal" if pending and not self.running else "disabled")
        self.stop_button.configure(state="normal" if self.running else "disabled")

    def add_paths(self, paths) -> None:
        for path in map(Path, paths):
            if path not in self.files:
                self.files.append(path)
                self.statuses[path] = "Waiting"
        self._refresh()

    def add_files(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in sorted(SUPPORTED_EXTENSIONS))
        self.add_paths(
            filedialog.askopenfilenames(
                title="Choose recordings",
                filetypes=[("Audio and video", patterns), ("All files", "*.*")],
            )
        )

    def add_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose a folder of recordings")
        if chosen:
            self.add_folder_path(Path(chosen))

    def add_folder_path(self, folder: Path) -> None:
        found = recordings_in(folder)
        self.add_paths(found)
        self.stage.set(f"Added {len(found)} recordings." if found else "No recordings found in that folder.")

    def remove_selected(self) -> None:
        for index in sorted((int(item) for item in self.table.selection()), reverse=True):
            self.statuses.pop(self.files.pop(index), None)
        self._refresh()

    def choose_output(self) -> None:
        chosen = filedialog.askdirectory(title="Save transcripts to", initialdir=self.output.get())
        if chosen:
            self.output.set(chosen)

    def open_output(self) -> None:
        folder = Path(self.output.get())
        if folder.exists():
            os.startfile(folder)
        else:
            self.stage.set("That folder doesn't exist yet. It's created when the first transcript is saved.")

    def start(self) -> None:
        queued = [path for path in self.files if not self.statuses.get(path, "").startswith("Done")]
        if not queued or self.running:
            return
        output = Path(self.output.get().strip())
        if not output.is_absolute():
            self.stage.set("Choose a full folder path for saving transcripts, such as E:\\TranscriptionData\\output.")
            return
        self.stop_requested.clear()
        self.progress.set(0)
        self.worker = threading.Thread(target=self._work, args=(queued, output), daemon=True)
        self.worker.start()
        self._refresh()

    def stop(self) -> None:
        self.stop_requested.set()
        self.stage.set("Stopping. Everything finished so far is saved…")

    def close(self) -> None:
        if self.running and not messagebox.askyesno(
            "Stop transcribing?", "A recording is still being transcribed. Progress so far is saved, and you can resume it later.\n\nClose anyway?"
        ):
            return
        self.stop_requested.set()
        self.root.destroy()

    def _work(self, queued: list[Path], output: Path) -> None:
        keep_awake(True)
        try:
            for path in queued:
                if self.stop_requested.is_set():
                    break
                self._run_one(path, output)
        finally:
            keep_awake(False)
            self.messages.put(("finished", None, None))

    def _run_one(self, path: Path, output: Path) -> None:
        time_left = TimeLeft()

        def check_stop() -> None:
            if self.stop_requested.is_set():
                raise Cancelled

        def on_stage(text: str) -> None:
            check_stop()
            self.messages.put(("status", path, text))

        def on_progress(position: float, duration: float) -> None:
            check_stop()
            percent = min(100.0, position / duration * 100) if duration else 100.0
            self.messages.put(("progress", path, (percent, time_left.minutes(position, duration))))

        def on_notice(text: str) -> None:
            # The speed changes, so the time left is measured afresh.
            time_left.restart()
            self.messages.put(("notice", None, f"Using: processor. {text}"))

        self.messages.put(("status", path, "Starting"))
        try:
            result = self.run_job(path, output, on_stage=on_stage, on_progress=on_progress, on_notice=on_notice)
        except Cancelled:
            self.messages.put(("status", path, "Stopped. Start again to resume."))
        except Exception as error:  # A failed file shouldn't stop the queue or freeze the window.
            self.messages.put(("failed", path, str(error)))
        else:
            self.messages.put(("done", path, result))

    def _finished_message(self) -> str:
        if self.stop_requested.is_set():
            return "Stopped."
        failed = sum(status.startswith("Failed") for status in self.statuses.values())
        if failed:
            files = "1 file" if failed == 1 else f"{failed} files"
            return f"Finished, but {files} failed. See the Status column."
        return "Finished. Transcripts are in the folder above."

    def _poll(self) -> None:
        # Rescheduled even after an error, or the window would stop updating for good.
        try:
            self._handle_messages()
        finally:
            self.poll_job = self.root.after(POLL_MILLISECONDS, self._poll)

    def _handle_messages(self) -> None:
        changed = False
        while True:
            try:
                kind, path, detail = self.messages.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "computer":
                # The first check can finish after a job has already switched to the processor.
                if not self.notice_shown:
                    self.computer.set(detail)
            elif kind == "notice":
                self.notice_shown = True
                self.computer.set(detail)
            elif kind == "status":
                if detail == "Starting":
                    self.progress.set(0)
                self.statuses[path] = detail
                self.stage.set(f"{path.name}: {detail}")
            elif kind == "progress":
                percent, remaining = detail
                self.progress.set(percent)
                estimate = _format_remaining(remaining) if remaining is not None else "working out the time left"
                self.statuses[path] = f"Transcribing, {percent:.0f}%"
                self.stage.set(f"{path.name}: {percent:.0f}%, {estimate}")
            elif kind == "done":
                self.statuses[path] = f"Done: {_summary(detail)}"
                self.progress.set(100)
            elif kind == "failed":
                self.statuses[path] = f"Failed: {detail}"
            elif kind == "finished":
                self.worker = None
                self.stage.set(self._finished_message())
        if changed:
            self._refresh()


def open_window(attempts: int = 3, delay_seconds: float = 1.0) -> tk.Tk:
    """Starting the window toolkit has failed intermittently, and a retry gets past that."""
    for attempt in range(attempts):
        try:
            return tk.Tk()
        except tk.TclError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)


def _show_error(text: str) -> None:
    # The launcher has no console, so an error that isn't shown here is never seen.
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, text, "Transcribe Local", 0x10)
    else:
        print(text, file=sys.stderr)


def main() -> None:
    try:
        if os.name == "nt":
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        root = open_window()
        root.report_callback_exception = lambda kind, value, traceback: _show_error(f"Something went wrong:\n\n{value}")
        App(root)
        root.mainloop()
    except Exception as error:
        _show_error(f"Transcribe Local couldn't start:\n\n{error}")


if __name__ == "__main__":
    main()
