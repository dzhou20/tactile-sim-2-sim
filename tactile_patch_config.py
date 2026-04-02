"""Compatibility wrapper for the new tactile package."""

from tactile.patch_config import (
    PatchSamplePoint,
    TactilePatch,
    TactilePatchRegistry,
    build_patch_sample_points,
)
from tactile.presets.umi import make_default_patch_registry, make_umi_finger1_inner_patch

__all__ = [
    "PatchSamplePoint",
    "TactilePatch",
    "TactilePatchRegistry",
    "build_patch_sample_points",
    "make_default_patch_registry",
    "make_umi_finger1_inner_patch",
]
