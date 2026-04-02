"""Tactile patch configuration primitives.

This module is intentionally independent from Isaac Sim so patch definitions can
be validated, tested, and reused without a simulator runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Tuple

Vector3 = Tuple[float, float, float]


def _vec_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: Vector3, scale: float) -> Vector3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def _vec_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_norm(v: Vector3) -> float:
    return sqrt(_vec_dot(v, v))


def _vec_normalize(v: Vector3) -> Vector3:
    norm = _vec_norm(v)
    if norm <= 1e-9:
        raise ValueError(f"Zero-length vector is not allowed: {v}")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _linspace(count: int, span: float) -> List[float]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count == 1:
        return [0.0]
    start = -0.5 * span
    step = span / float(count - 1)
    return [start + idx * step for idx in range(count)]


@dataclass(frozen=True)
class TactilePatch:
    """Rectangular tactile patch defined in a prim local frame."""

    name: str
    parent_prim: str
    origin: Vector3
    normal: Vector3
    axis_u: Vector3
    axis_v: Vector3
    size_u: float
    size_v: float
    rows: int
    cols: int
    sensor_radius: float
    offset: float = 0.0
    min_threshold: float = -1.0
    max_threshold: float = 100000.0
    sensor_period: float = 0.0
    name_prefix: str = "Tactile"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Patch name must not be empty")
        if not self.parent_prim:
            raise ValueError("parent_prim must not be empty")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"rows/cols must be positive, got {self.rows}x{self.cols}")
        if self.size_u < 0.0 or self.size_v < 0.0:
            raise ValueError(f"Patch size must be non-negative, got ({self.size_u}, {self.size_v})")
        if self.sensor_radius <= 0.0:
            raise ValueError(f"sensor_radius must be positive, got {self.sensor_radius}")
        if self.max_threshold < self.min_threshold:
            raise ValueError(
                "max_threshold must be greater than or equal to min_threshold, "
                f"got ({self.min_threshold}, {self.max_threshold})"
            )

        normal = _vec_normalize(self.normal)
        axis_u = _vec_normalize(self.axis_u)
        axis_v = _vec_normalize(self.axis_v)

        if abs(_vec_dot(normal, axis_u)) > 1e-5:
            raise ValueError("normal and axis_u must be orthogonal")
        if abs(_vec_dot(normal, axis_v)) > 1e-5:
            raise ValueError("normal and axis_v must be orthogonal")
        if abs(_vec_dot(axis_u, axis_v)) > 1e-5:
            raise ValueError("axis_u and axis_v must be orthogonal")

    def normalized_frame(self) -> "TactilePatch":
        self.validate()
        return TactilePatch(
            name=self.name,
            parent_prim=self.parent_prim,
            origin=self.origin,
            normal=_vec_normalize(self.normal),
            axis_u=_vec_normalize(self.axis_u),
            axis_v=_vec_normalize(self.axis_v),
            size_u=self.size_u,
            size_v=self.size_v,
            rows=self.rows,
            cols=self.cols,
            sensor_radius=self.sensor_radius,
            offset=self.offset,
            min_threshold=self.min_threshold,
            max_threshold=self.max_threshold,
            sensor_period=self.sensor_period,
            name_prefix=self.name_prefix,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class PatchSamplePoint:
    row: int
    col: int
    local_position: Vector3
    sensor_name: str


def build_patch_sample_points(patch: TactilePatch) -> List[PatchSamplePoint]:
    patch = patch.normalized_frame()
    offsets_u = _linspace(patch.rows, patch.size_u)
    offsets_v = _linspace(patch.cols, patch.size_v)
    base_origin = _vec_add(patch.origin, _vec_scale(patch.normal, patch.offset))

    points: List[PatchSamplePoint] = []
    for row, du in enumerate(offsets_u):
        for col, dv in enumerate(offsets_v):
            local_position = _vec_add(
                _vec_add(base_origin, _vec_scale(patch.axis_u, du)),
                _vec_scale(patch.axis_v, dv),
            )
            sensor_name = f"{patch.name_prefix}_{row}_{col}"
            points.append(
                PatchSamplePoint(
                    row=row,
                    col=col,
                    local_position=local_position,
                    sensor_name=sensor_name,
                )
            )
    return points


class TactilePatchRegistry:
    """Simple in-memory registry for named tactile patches."""

    def __init__(self, patches: Iterable[TactilePatch] | None = None):
        self._patches: Dict[str, TactilePatch] = {}
        if patches is not None:
            for patch in patches:
                self.add(patch)

    def add(self, patch: TactilePatch) -> None:
        patch.validate()
        if patch.name in self._patches:
            raise ValueError(f"Patch already exists: {patch.name}")
        self._patches[patch.name] = patch

    def get(self, name: str) -> TactilePatch:
        try:
            return self._patches[name]
        except KeyError as exc:
            raise KeyError(f"Unknown patch: {name}") from exc

    def all(self) -> List[TactilePatch]:
        return list(self._patches.values())

    def as_dict(self) -> Dict[str, TactilePatch]:
        return dict(self._patches)
