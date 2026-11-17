# Third-party software

Transcribe Local's own code is under the MIT License (see `LICENSE`). It doesn't include any
other project's code. Setup downloads the pieces below onto your computer, each under its own
license. All of them allow free use, including in research. Most are open source; NVIDIA's
libraries are free to use but not open source.

## The speech model

- **Whisper large-v3** by OpenAI, MIT License, converted for faster-whisper by Systran
  (`Systran/faster-whisper-large-v3` on Hugging Face, MIT License).
  `scripts/download_models.py` fetches one fixed version of it.

## Programs you install separately

- **FFmpeg** (installed with `winget install --id Gyan.FFmpeg`). The build that command installs
  is licensed under the GPL. Transcribe Local only runs it as a separate program and doesn't
  include or change it, so the GPL doesn't apply to this code. If you ever bundle FFmpeg with
  this program, its license terms apply to what you distribute.
- **Python**, Python Software Foundation License.
- **NVIDIA graphics driver**, NVIDIA's own license, if you use a graphics card.

## Python packages (from `requirements.txt`)

| Package | License |
|---|---|
| faster-whisper, CTranslate2, onnxruntime, python-docx | MIT |
| NumPy, PyAV, soundfile, lxml | BSD |
| nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12 (NVIDIA's graphics libraries) | NVIDIA's own license (free to use, not open source) |
| huggingface_hub, tokenizers | Apache-2.0 |
| tqdm | MPL-2.0 and MIT |
| The rest (helper packages) | MIT, BSD, Apache-2.0 or PSF |

To see the license of every installed package:

```
.venv\Scripts\python -m pip show <package>
```
