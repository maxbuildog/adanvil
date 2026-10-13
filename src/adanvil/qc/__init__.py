"""Automatic quality control for rendered advertisements."""

from .pair import PairError, compare_pair
from .report import Check, QCReport, qc_report
from .score import ScoreError, ScoreReport, ad_score

__all__ = [
    "Check",
    "PairError",
    "QCReport",
    "ScoreError",
    "ScoreReport",
    "ad_score",
    "compare_pair",
    "qc_report",
]
