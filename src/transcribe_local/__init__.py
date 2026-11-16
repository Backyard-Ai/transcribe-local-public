"""Offline transcription of research interviews and focus groups for MAXQDA."""

import os

__version__ = "0.1.0"

# Cards newer than the transcription library (see hardware.LIBRARY_NEWEST_CARD) have
# its code translated by the NVIDIA driver on first use, and the driver keeps the result
# in a cache on disk. 1 GB keeps it from being squeezed out and redone each time. This
# must be set before any graphics library starts, which is why it's here. A value the
# person has set themselves is kept.
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(1024**3))
