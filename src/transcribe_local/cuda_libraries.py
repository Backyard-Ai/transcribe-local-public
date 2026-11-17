"""Makes NVIDIA's graphics libraries findable for the transcription engine.

CTranslate2 runs the model on the graphics card with NVIDIA's cuBLAS and cuDNN libraries
from CUDA 12. Setup installs them from NVIDIA's own packages (nvidia-cublas-cu12,
nvidia-cudnn-cu12 and nvidia-cuda-runtime-cu12), which put them in
site-packages/nvidia/<name>/bin. Windows doesn't look there by itself.

Until version 0.1.1 the program installed PyTorch only to make these libraries findable.
The versions here match what PyTorch 2.11 (CUDA 12.8) supplied, so transcripts are
unchanged.
"""

import os
from pathlib import Path

_made_findable = False


def library_folders() -> list[Path]:
    """The folders holding NVIDIA's libraries, or none if the packages aren't installed."""
    try:
        import nvidia
    except ImportError:
        return []
    folders = []
    for root in getattr(nvidia, "__path__", []):
        folders.extend(folder for folder in sorted(Path(root).glob("*/bin")) if folder.is_dir())
    return folders


def make_findable() -> list[Path]:
    """Let Windows find the libraries. Safe to call more than once."""
    global _made_findable
    folders = library_folders()
    if os.name != "nt" or _made_findable or not folders:
        return folders
    for folder in folders:
        os.add_dll_directory(str(folder))
    # The libraries load further libraries of their own by name, which only searches PATH.
    # This changes PATH for this program only, not for Windows.
    os.environ["PATH"] = os.pathsep.join([*(str(folder) for folder in folders), os.environ.get("PATH", "")])
    _made_findable = True
    return folders
