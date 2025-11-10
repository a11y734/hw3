"""
Reusable helpers for the spam email classification project.

The package exposes preprocessing, feature extraction, model utilities,
visualizations, and persistence helpers that are shared by the CLI scripts
and the Streamlit application.
"""

from . import preprocessing, features, metrics, io_utils, visualizations, artifacts

__all__ = [
    "preprocessing",
    "features",
    "metrics",
    "io_utils",
    "visualizations",
    "artifacts",
]
