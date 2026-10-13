"""Footage cataloguing and annotation helpers."""

from .annotate import annotate, entry_line, search
from .catalog import (
    Annotation,
    Catalog,
    Entry,
    EntrySpeech,
    Shot,
    entry_id,
    load_catalog,
    save_catalog,
)
from .index import (
    IndexResult,
    RefineResult,
    index_library,
    index_summary,
    pending_files,
    pending_speech,
    refine_speech,
    refine_targets,
)
from .overview import library_overview
from .vision_index import pending_vision
from .vision_search import ShotHit, search_mosaic, search_shots

__all__ = [
    "Annotation",
    "Catalog",
    "Entry",
    "EntrySpeech",
    "IndexResult",
    "RefineResult",
    "Shot",
    "ShotHit",
    "annotate",
    "entry_id",
    "entry_line",
    "index_library",
    "index_summary",
    "library_overview",
    "load_catalog",
    "pending_files",
    "pending_speech",
    "pending_vision",
    "refine_speech",
    "refine_targets",
    "save_catalog",
    "search",
    "search_mosaic",
    "search_shots",
]
