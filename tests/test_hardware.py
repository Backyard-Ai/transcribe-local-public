import json

import pytest

from transcribe_local.hardware import (
    CPU_ATTEMPTS,
    FIRST_RUN_NOTE,
    LARGE_CARD_ATTEMPTS,
    Computer,
    battery_from_status,
    choose_profile,
    default_cpu_threads,
    encryption_advice,
    encryption_state,
)
from transcribe_local.pipeline import resolve_data_root


def computer(card=None, memory=None, problem=None, cores=24, generation=None):
    return Computer("Windows-11", "Test CPU", cores, 32.0, card, memory, problem, generation)


def test_desktop_card_keeps_the_settings_it_was_built_with():
    profile = choose_profile(computer("NVIDIA GeForce RTX 3060", 12.0))
    assert profile.name == "large card"
    assert profile.attempts[:5] == (
        ("cuda", "float16", 8),
        ("cuda", "float16", 4),
        ("cuda", "int8_float16", 4),
        ("cuda", "int8_float16", 2),
        ("cuda", "int8_float16", 1),
    )
    assert profile.warning is None


@pytest.mark.parametrize(
    "memory, name, first_attempt",
    [
        (24.0, "large card", ("cuda", "float16", 8)),
        (10.0, "large card", ("cuda", "float16", 8)),
        (8.0, "medium card", ("cuda", "float16", 4)),
        (6.5, "medium card", ("cuda", "float16", 4)),
        (6.0, "small card", ("cuda", "int8_float16", 4)),
        (4.0, "small card", ("cuda", "int8_float16", 4)),
    ],
)
def test_card_settings_follow_its_memory(memory, name, first_attempt):
    profile = choose_profile(computer("NVIDIA Laptop GPU", memory))
    assert profile.name == name
    assert profile.attempts[0] == first_attempt
    assert profile.uses_card


def test_every_card_profile_ends_with_the_processor():
    for memory in (4.0, 8.0, 12.0):
        assert choose_profile(computer("Card", memory)).attempts[-1][0] == "cpu"


@pytest.mark.parametrize(
    "card, memory, problem, warning_part",
    [
        (None, None, None, "No NVIDIA graphics card"),
        ("NVIDIA GeForce MX150", 2.0, None, "too little"),
        ("NVIDIA GeForce RTX 3050", None, "CUDA isn't working", "CUDA isn't working"),
    ],
)
def test_processor_is_used_when_no_card_can_be(card, memory, problem, warning_part):
    profile = choose_profile(computer(card, memory, problem))
    assert profile.name == "processor"
    assert profile.attempts == CPU_ATTEMPTS
    assert not profile.uses_card
    assert warning_part in profile.warning


def test_processor_can_be_chosen_on_purpose():
    profile = choose_profile(computer("NVIDIA GeForce RTX 3060", 12.0), device_override="cpu")
    assert profile.attempts == CPU_ATTEMPTS
    assert "processor only" in profile.warning


def test_large_card_attempts_match_the_desktop_profile():
    assert LARGE_CARD_ATTEMPTS[-1] == ("cpu", "int8", 1)


@pytest.mark.parametrize("cores, threads", [(24, 12), (8, 4), (1, 1), (2, 1)])
def test_processor_threads_are_one_per_core(cores, threads):
    assert default_cpu_threads(cores) == threads
    assert choose_profile(computer(cores=cores)).cpu_threads == threads


def test_data_folder_from_environment_comes_first(tmp_path):
    settings = tmp_path / "local_settings.json"
    settings.write_text(json.dumps({"data_folder": "D:\\Configured"}), encoding="utf-8")
    root = resolve_data_root({"TRANSCRIBE_LOCAL_DATA": "C:\\FromEnv"}, settings, tmp_path, tmp_path)
    assert str(root) == "C:\\FromEnv"


def test_data_folder_from_settings_file(tmp_path):
    settings = tmp_path / "local_settings.json"
    settings.write_text(json.dumps({"data_folder": "D:\\Configured"}), encoding="utf-8")
    assert str(resolve_data_root({}, settings, tmp_path / "missing", tmp_path)) == "D:\\Configured"


def test_desktop_folder_is_used_only_when_it_exists(tmp_path):
    desktop = tmp_path / "TranscriptionData"
    desktop.mkdir()
    assert resolve_data_root({}, tmp_path / "none.json", desktop, tmp_path / "home") == desktop


def test_other_computers_default_to_the_user_folder(tmp_path):
    home = tmp_path / "home"
    root = resolve_data_root({}, tmp_path / "none.json", tmp_path / "no E drive", home)
    assert root == home / "TranscriptionData"


@pytest.mark.parametrize("content", ["not json", "[]", '{"data_folder": ""}'])
def test_damaged_settings_file_falls_back_to_defaults(tmp_path, content):
    settings = tmp_path / "local_settings.json"
    settings.write_text(content, encoding="utf-8")
    home = tmp_path / "home"
    assert resolve_data_root({}, settings, tmp_path / "missing", home) == home / "TranscriptionData"


def test_laptop_rtx_5060_gets_the_medium_profile_and_a_first_run_note():
    # The owner's laptop: 8151 MB of graphics memory, a Blackwell (RTX 50 series) card.
    profile = choose_profile(computer("NVIDIA GeForce RTX 5060 Laptop GPU", 8151 / 1024, generation=(12, 0)))
    assert profile.name == "medium card"
    assert profile.attempts[0] == ("cuda", "float16", 4)
    assert profile.attempts[-1][0] == "cpu"
    assert profile.first_run_note == FIRST_RUN_NOTE


@pytest.mark.parametrize("generation", [(8, 6), (7, 5), None])
def test_cards_the_library_was_built_for_get_no_first_run_note(generation):
    profile = choose_profile(computer("NVIDIA GeForce RTX 3060", 12.0, generation=generation))
    assert profile.first_run_note is None


def test_processor_profile_has_no_first_run_note():
    profile = choose_profile(computer("NVIDIA GeForce RTX 5060", 8.0, generation=(12, 0)), device_override="cpu")
    assert profile.first_run_note is None


@pytest.mark.parametrize(
    "ac_line, battery_flag, expected",
    [(0, 1, True), (1, 8, False), (1, 128, None), (255, 1, None), (0, 128, None), (1, 128 | 8, None)],
)
def test_battery_status(ac_line, battery_flag, expected):
    assert battery_from_status(ac_line, battery_flag) is expected


@pytest.mark.parametrize(
    "code, state",
    [
        ("1", "on"),
        ("2", "OFF"),
        ("6", "on"),
        ("8\r\n", "WAITING FOR ACTIVATION, not protecting the drive yet"),
        ("", "not available"),
        ("0", "not available"),
        ("7", "unknown (code 7)"),
    ],
)
def test_encryption_codes(code, state):
    assert encryption_state(code) == state


@pytest.mark.parametrize("state", ["on", "turning on", "not checked on this system"])
def test_protected_drive_needs_no_advice(state):
    assert encryption_advice(state, windows_home=True) is None
    assert encryption_advice(state, windows_home=False) is None


def test_failed_or_paused_check_gets_its_own_advice():
    for windows_home in (True, False):
        assert "Couldn't check" in encryption_advice("couldn't check", windows_home)
        assert "Restart" in encryption_advice("SUSPENDED", windows_home)


@pytest.mark.parametrize("state", ["OFF", "not available", "WAITING FOR ACTIVATION, not protecting the drive yet"])
def test_unprotected_drive_gets_advice_for_its_windows_edition(state):
    assert "Device encryption" in encryption_advice(state, windows_home=True)
    assert "BitLocker" in encryption_advice(state, windows_home=False)


def test_driver_code_cache_is_large_enough_to_keep():
    import os

    import transcribe_local  # noqa: F401

    assert int(os.environ["CUDA_CACHE_MAXSIZE"]) >= 1024**3
