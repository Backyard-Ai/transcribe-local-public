"""One-time model download. This is the only step that needs the internet."""

from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"

# One fixed version of the model, so every computer gets the same files and a later change
# upstream can't alter transcripts or slip in something unexpected. Change it on purpose only.
MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"

path = snapshot_download(
    "Systran/faster-whisper-large-v3",
    revision=MODEL_REVISION,
    local_dir=MODELS_ROOT / "faster-whisper-large-v3",
)
print(f"Saved to {path}")
