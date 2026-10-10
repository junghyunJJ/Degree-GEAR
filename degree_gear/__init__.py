"""Degree-GEAR: degree-preserving spatial transcriptomics clustering."""

import os

# Same import-time PyTorch allocator setting as the AAAI-27 code supplement.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .core import (
    DEFAULT_DEGREE_GEAR_CONFIG,
    DegreeGearConfig,
    EmbeddingConfig,
    GraphConfig,
    LeidenConfig,
    PreprocessingConfig,
    build_degree_gear_graph,
    embed_degree_gear,
    leiden_degree_gear,
    preprocess_degree_gear,
    run_degree_gear,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DEGREE_GEAR_CONFIG",
    "DegreeGearConfig",
    "EmbeddingConfig",
    "GraphConfig",
    "LeidenConfig",
    "PreprocessingConfig",
    "build_degree_gear_graph",
    "embed_degree_gear",
    "leiden_degree_gear",
    "preprocess_degree_gear",
    "run_degree_gear",
]
