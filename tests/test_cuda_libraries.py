import os
import sys
import types

import pytest

from transcribe_local import cuda_libraries, hardware
from transcribe_local.hardware import parse_driver_report


@pytest.mark.parametrize(
    "report, expected",
    [
        ("NVIDIA GeForce RTX 5060 Laptop GPU, 8151, 12.0\n", ("NVIDIA GeForce RTX 5060 Laptop GPU", 8151 / 1024, (12, 0))),
        ("NVIDIA GeForce RTX 3060, 12288, 8.6\n", ("NVIDIA GeForce RTX 3060", 12.0, (8, 6))),
        # An older driver that can't report the card's generation.
        ("NVIDIA GeForce GTX 1080, 8192\n", ("NVIDIA GeForce GTX 1080", 8.0, None)),
        ("NVIDIA GeForce GTX 1080, 8192, [N/A]\n", ("NVIDIA GeForce GTX 1080", 8.0, None)),
        # Two cards: the first is used.
        ("Card A, 4096, 7.5\nCard B, 24576, 8.9\n", ("Card A", 4.0, (7, 5))),
    ],
)
def test_driver_report_is_read(report, expected):
    assert parse_driver_report(report) == expected


@pytest.mark.parametrize("report", ["", "\n\n", ", 8192, 8.6"])
def test_empty_driver_report_means_no_card(report):
    assert parse_driver_report(report) is None


def fake_nvidia_package(monkeypatch, tmp_path, names):
    root = tmp_path / "nvidia"
    for name in names:
        (root / name / "bin").mkdir(parents=True)
    (root / "not_a_library").mkdir(parents=True)
    package = types.ModuleType("nvidia")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "nvidia", package)
    return root


def test_library_folders_are_found(monkeypatch, tmp_path):
    root = fake_nvidia_package(monkeypatch, tmp_path, ["cublas", "cudnn", "cuda_runtime"])
    assert cuda_libraries.library_folders() == [root / "cublas" / "bin", root / "cuda_runtime" / "bin", root / "cudnn" / "bin"]


def test_no_libraries_without_the_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvidia", None)
    assert cuda_libraries.library_folders() == []


def test_making_findable_adds_each_folder_once(monkeypatch, tmp_path):
    root = fake_nvidia_package(monkeypatch, tmp_path, ["cublas", "cudnn"])
    added = []
    monkeypatch.setattr(cuda_libraries.os, "name", "nt")
    monkeypatch.setattr(cuda_libraries.os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setattr(cuda_libraries, "_made_findable", False)
    monkeypatch.setenv("PATH", "C:\\Windows")

    cuda_libraries.make_findable()
    cuda_libraries.make_findable()

    assert added == [str(root / "cublas" / "bin"), str(root / "cudnn" / "bin")]
    assert os.environ["PATH"].split(os.pathsep) == added + ["C:\\Windows"]


@pytest.mark.skipif(os.name != "nt", reason="the NVIDIA packages hold Windows libraries")
def test_installed_nvidia_libraries_are_where_the_program_looks():
    names = {path.name.lower() for folder in cuda_libraries.library_folders() for path in folder.glob("*.dll")}
    assert {"cublas64_12.dll", "cublaslt64_12.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudart64_12.dll"} <= names


class FakeLibrary:
    def __init__(self, count):
        self.count = count

    def get_cuda_device_count(self):
        return self.count


def detect_with(monkeypatch, card, folders, devices):
    monkeypatch.setattr(hardware, "_nvidia_card_from_driver", lambda: card)
    monkeypatch.setattr(cuda_libraries, "make_findable", lambda: folders)
    monkeypatch.setattr(hardware.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "ctranslate2", FakeLibrary(devices))
    return hardware.detect_computer()


CARD = ("NVIDIA GeForce RTX 5060 Laptop GPU", 8151 / 1024, (12, 0))


def test_working_card_is_detected(monkeypatch):
    found = detect_with(monkeypatch, CARD, ["cublas"], 1)
    assert (found.card_name, found.card_memory_gb, found.card_generation, found.card_problem) == (*CARD, None)


def test_missing_libraries_are_reported(monkeypatch):
    assert "setup.ps1" in detect_with(monkeypatch, CARD, [], 1).card_problem


def test_card_the_library_cant_reach_is_reported(monkeypatch):
    assert "driver" in detect_with(monkeypatch, CARD, ["cublas"], 0).card_problem


def test_no_card_means_no_problem(monkeypatch):
    found = detect_with(monkeypatch, None, [], 0)
    assert (found.card_name, found.card_problem) == (None, None)
