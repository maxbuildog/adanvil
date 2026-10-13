"""Optional vision features: image and text embeddings, faces, product versions, framing.

Everything heavy runs in a separate environment (``adanvil vision-setup``) behind a persistent
worker; the base package only needs numpy and Pillow.
"""

from .client import (
    UNAVAILABLE,
    VisionUnavailableError,
    available,
    get_engine,
    status,
    use_engine,
)
from .engine import Face, FacePicture, FakeVisionEngine, VisionEngine
from .setup import vision_setup

__all__ = [
    "UNAVAILABLE",
    "Face",
    "FacePicture",
    "FakeVisionEngine",
    "VisionEngine",
    "VisionUnavailableError",
    "available",
    "get_engine",
    "status",
    "use_engine",
    "vision_setup",
]
