"""Checks the computer this runs on and picks transcription settings to suit it.

Every computer uses the same model, so transcripts are comparable wherever they were
made. Only how it runs changes: on an NVIDIA graphics card sized to its memory, or on
the processor, which works anywhere but is much slower.

Run `python -m transcribe_local.hardware` for a report, and add `--test` to also load
the model and run it briefly.
"""

import argparse
import ctypes
import functools
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

# Each attempt is (device, compute type, batch size). If one runs out of memory, the
# job resumes from its last saved segment with the next. The processor comes last on
# every graphics card profile: slow, but better than the job failing.
CPU_ATTEMPTS = (("cpu", "int8", 4), ("cpu", "int8", 2), ("cpu", "int8", 1))
LARGE_CARD_ATTEMPTS = (
    ("cuda", "float16", 8),
    ("cuda", "float16", 4),
    ("cuda", "int8_float16", 4),
    ("cuda", "int8_float16", 2),
    ("cuda", "int8_float16", 1),
) + CPU_ATTEMPTS[-1:]
MEDIUM_CARD_ATTEMPTS = LARGE_CARD_ATTEMPTS[1:]
SMALL_CARD_ATTEMPTS = LARGE_CARD_ATTEMPTS[2:]

# Graphics memory needed, in GB. Measured on the desktop's 12 GB card: up to 7.7 GB at
# full speed, and about 2.7 GB at the lightest card settings. The middle boundary is an
# estimate that hasn't been tested on a real card of that size.
LARGE_CARD_GB = 10.0
MEDIUM_CARD_GB = 6.5
SMALL_CARD_GB = 3.5

# The newest card generation the transcription library (CTranslate2 4.8.5 for Windows)
# has ready-made code for: (8, 6) is the desktop's RTX 30 series. Newer cards, such as
# the RTX 40 and 50 series, run the library's portable code instead, which the NVIDIA
# driver translates for the card the first time, so that first start is slow.
LIBRARY_NEWEST_CARD = (8, 6)

FIRST_RUN_NOTE = (
    "This graphics card is newer than the transcription library, so the first run after "
    "installing or updating the NVIDIA driver can take several minutes to start."
)
BATTERY_WARNING = (
    "This laptop is on battery. Plug it in: the graphics card is much slower on battery. "
    "Progress is saved if the battery runs out."
)

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@dataclass(frozen=True)
class Computer:
    system: str
    processor: str
    logical_cores: int
    memory_gb: float
    card_name: str | None = None
    card_memory_gb: float | None = None
    # Why an NVIDIA card that's present can't be used, if it can't.
    card_problem: str | None = None
    # The card's generation as (major, minor), e.g. (8, 6) for the RTX 30 series.
    card_generation: tuple[int, int] | None = None


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    attempts: tuple[tuple[str, str, int], ...]
    cpu_threads: int
    warning: str | None = None
    first_run_note: str | None = None

    @property
    def uses_card(self) -> bool:
        return self.attempts[0][0] == "cuda"


def _processor_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown processor"


def _memory_gb() -> float:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 1024**3
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, ValueError, OSError):
        return 0.0


def _nvidia_card_from_driver() -> str | None:
    """The card name reported by NVIDIA's driver tool, which works even when CUDA doesn't."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        result = subprocess.run(
            [tool, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    name = result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None
    return name


def detect_computer() -> Computer:
    card_name = card_memory = problem = generation = None
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            card_name, card_memory = properties.name, properties.total_memory / 1024**3
            generation = (properties.major, properties.minor)
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                problem = "the transcription library can't reach the graphics card"
    except Exception as error:  # A broken install shouldn't stop the processor from being used.
        problem = f"the graphics card libraries didn't load ({error})"

    if card_name is None:
        driver_card = _nvidia_card_from_driver()
        if driver_card:
            card_name = driver_card
            problem = problem or "an NVIDIA card is installed, but CUDA isn't working. Updating the NVIDIA driver usually fixes this"

    return Computer(
        system=platform.platform(),
        processor=_processor_name(),
        logical_cores=os.cpu_count() or 1,
        memory_gb=_memory_gb(),
        card_name=card_name,
        card_memory_gb=card_memory,
        card_problem=problem,
        card_generation=generation,
    )


def default_cpu_threads(logical_cores: int | None = None) -> int:
    # Most processors run two threads per core; using one per core is fastest for this work.
    return max(1, (logical_cores or os.cpu_count() or 2) // 2)


def choose_profile(computer: Computer, device_override: str | None = None) -> Profile:
    cpu_threads = default_cpu_threads(computer.logical_cores)
    card = computer.card_name
    memory = computer.card_memory_gb

    def processor(warning: str | None) -> Profile:
        return Profile(
            "processor",
            f"processor only ({computer.processor}): much slower than a graphics card",
            CPU_ATTEMPTS,
            cpu_threads,
            warning,
        )

    if device_override == "cpu":
        return processor("Set to use the processor only.")
    if card is None:
        return processor("No NVIDIA graphics card found, so transcription will be much slower.")
    if computer.card_problem or memory is None:
        return processor(f"{card} can't be used: {computer.card_problem or 'its memory size is unknown'}.")
    if memory < SMALL_CARD_GB:
        return processor(f"{card} has {memory:.0f} GB of memory, too little for the model, so the processor is used.")

    label = f"{card} ({memory:.0f} GB)"
    newer = computer.card_generation is not None and computer.card_generation > LIBRARY_NEWEST_CARD
    note = FIRST_RUN_NOTE if newer else None
    if memory >= LARGE_CARD_GB:
        return Profile("large card", f"{label}: full speed", LARGE_CARD_ATTEMPTS, cpu_threads, None, note)
    if memory >= MEDIUM_CARD_GB:
        return Profile("medium card", f"{label}: fast", MEDIUM_CARD_ATTEMPTS, cpu_threads, None, note)
    return Profile(
        "small card", f"{label}: lighter settings to fit its memory", SMALL_CARD_ATTEMPTS, cpu_threads, None, note
    )


@functools.lru_cache(maxsize=1)
def current_profile() -> Profile:
    override = os.environ.get("TRANSCRIBE_LOCAL_DEVICE", "").strip().lower() or None
    return choose_profile(detect_computer(), override)


def battery_from_status(ac_line: int, battery_flag: int) -> bool | None:
    """True on battery, False plugged in, None if there's no battery or Windows can't tell."""
    if battery_flag & 128 or ac_line == 255:  # 128: no battery. 255: unknown.
        return None
    return ac_line == 0


def on_battery() -> bool | None:
    if os.name != "nt":
        return None

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte),
            ("battery_percent", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
            ("seconds_left", ctypes.c_ulong), ("seconds_full", ctypes.c_ulong),
        ]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return battery_from_status(status.ac_line, status.battery_flag)


def is_windows_home() -> bool:
    """Windows Home has Device encryption instead of BitLocker's settings."""
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            return str(winreg.QueryValueEx(key, "EditionID")[0]).startswith("Core")
    except OSError:
        return False


# Windows' codes for a drive's encryption. Device encryption on Windows Home is
# BitLocker underneath, so it reports the same codes. "Waiting for activation" is
# common on new laptops: the drive is encrypted but the key isn't protected yet.
ENCRYPTION_STATES = {
    "0": "not available",
    "1": "on",
    "2": "OFF",
    "3": "turning on",
    "4": "turning off",
    "5": "SUSPENDED",
    "6": "on",
    "8": "WAITING FOR ACTIVATION, not protecting the drive yet",
}
ENCRYPTION_OK = ("on", "turning on", "not checked on this system")


def encryption_state(code: str) -> str:
    code = code.strip()
    if not code:
        return "not available"
    return ENCRYPTION_STATES.get(code, f"unknown (code {code})")


def encryption_advice(state: str, windows_home: bool) -> str | None:
    """What to do before adding confidential recordings, or None if the drive is protected."""
    if state in ENCRYPTION_OK:
        return None
    if state == "couldn't check":
        return (
            "Couldn't check drive encryption. Check it yourself in Settings before adding "
            "confidential recordings, or run this check again."
        )
    if state == "SUSPENDED":
        return (
            "Drive encryption is paused, usually for an update. Restart the computer, which "
            "normally resumes it, then run this check again before adding confidential recordings."
        )
    if windows_home:
        return (
            "Turn on Device encryption before adding confidential recordings: Settings > "
            "Privacy & security > Device encryption. If that page isn't there, this computer "
            "can't use it. docs/laptop-setup.md lists the other options."
        )
    return "Turn on BitLocker for this drive before adding confidential recordings."


def _encryption(folder: str) -> str:
    if os.name != "nt":
        return "not checked on this system"
    drive = os.path.splitdrive(os.path.abspath(folder))[0] + "\\"
    script = f"(New-Object -ComObject Shell.Application).NameSpace('{drive}').Self.ExtendedProperty('System.Volume.BitLockerProtection')"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "couldn't check"
    return encryption_state(result.stdout)


def _quick_test(profile: Profile, model_path) -> str:
    import time

    import numpy as np

    from transcribe_local.engine import load_model

    device, compute_type, _ = profile.attempts[0]
    started = time.monotonic()
    model = load_model(model_path, device, compute_type, profile.cpu_threads)
    segments, _ = model.transcribe(np.zeros(16000 * 3, dtype="float32"), language="en")
    list(segments)
    return f"passed: the model loaded and ran on the {'graphics card' if device == 'cuda' else 'processor'} in {time.monotonic() - started:.0f} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hardware", description=__doc__)
    parser.add_argument("--test", action="store_true", help="also load the model and run it briefly")
    args = parser.parse_args(argv)

    from transcribe_local.pipeline import DATA_ROOT, DEFAULT_MODEL
    from transcribe_local.safety import block_network, cloud_reason

    computer = detect_computer()
    profile = current_profile()
    card = (
        f"{computer.card_name}, {computer.card_memory_gb:.0f} GB" if computer.card_memory_gb
        else computer.card_name or "no NVIDIA card found"
    )
    print("This computer")
    print(f"  System: {computer.system}")
    print(f"  Processor: {computer.processor}, {computer.logical_cores} threads")
    print(f"  Memory: {computer.memory_gb:.0f} GB")
    print(f"  Graphics card: {card}")
    print(f"\nSettings: {profile.description}")
    for note in (profile.warning, profile.first_run_note):
        if note:
            print(f"  NOTE: {note}")
    if on_battery():
        print(f"  NOTE: {BATTERY_WARNING}")

    problems = 0
    print("\nChecks")
    synced = cloud_reason(DATA_ROOT)
    print(f"  Data folder: {DATA_ROOT} ({'PROBLEM: ' + synced if synced else 'local, not synced'})")
    problems += bool(synced)
    windows_home = is_windows_home()
    encryption = _encryption(str(DATA_ROOT))
    print(f"  Drive encryption ({'Device encryption' if windows_home else 'BitLocker'}): {encryption}")
    model_ok = DEFAULT_MODEL.exists()
    print(f"  Model: {'found' if model_ok else 'MISSING, run scripts/download_models.py'}")
    from transcribe_local.audio import AudioError, _find_tool

    try:
        _find_tool("ffmpeg")
        print("  ffmpeg: found")
    except AudioError:
        print("  ffmpeg: MISSING, install it with: winget install --id Gyan.FFmpeg --exact")
        problems += 1
    problems += not model_ok

    if args.test and model_ok:
        from transcribe_local.engine import is_card_failure

        block_network()
        if profile.first_run_note:
            print("  Quick test: running. The first time on this card can take several minutes…", flush=True)
        try:
            print(f"  Quick test: {_quick_test(profile, DEFAULT_MODEL)}")
        except Exception as error:
            print(f"  Quick test: FAILED, {error}")
            if is_card_failure(error, profile.attempts[0][0]):
                print(
                    "  The graphics card couldn't run the model. Update the NVIDIA driver and run this "
                    "check again. Until then, transcription falls back to the processor, which is much slower."
                )
            problems += 1
    advice = encryption_advice(encryption, windows_home)
    if advice:
        print(f"\n{advice}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
