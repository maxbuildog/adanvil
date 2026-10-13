"""Detection and removal of burnt-in text from reference footage."""

from .core import (
    RemoveText,
    build_mask,
    detect_text_boxes,
    ffmpeg_filter_path,
    removal_filter,
    sample_frames,
)

__all__ = [
    "RemoveText",
    "build_mask",
    "detect_text_boxes",
    "ffmpeg_filter_path",
    "removal_filter",
    "sample_frames",
]
