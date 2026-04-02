"""Deviation analysis pipeline for comparing 3D-printed parts to CAD geometry."""

import os as _os

# Force single-threaded OpenMP to ensure deterministic RANSAC registration.
# Open3D's parallel RANSAC evaluates hypotheses across threads; thread scheduling
# differences cause different convergence points across runs even with the same
# seed.  Must be set before importing open3d (which initializes the thread pool).
_os.environ.setdefault("OMP_NUM_THREADS", "1")

from deviation_analysis.config import PipelineConfig, MethodSpec, ScanConfig
from deviation_analysis.deviation import DeviationMetrics

__all__ = [
    "PipelineConfig",
    "MethodSpec",
    "ScanConfig",
    "DeviationMetrics",
]
