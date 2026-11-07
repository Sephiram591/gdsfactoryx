"""Differentiable complex transformation (KLayout DCplxTrans semantics).

A point ``p`` is transformed as ``disp + mag * R(rotation) @ M(mirror) @ p``
where ``M(True)`` mirrors at the x-axis (``y -> -y``) and ``R`` rotates
counter-clockwise by ``rotation`` degrees. ``mirror`` is a static python bool,
all other fields can be traced jax scalars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax

from gdsfactory._jax import (
    Array,
    asarray,
    cos_deg,
    maybe_float,
    sin_deg,
    to_float,
    xp,
)


@jax.tree_util.register_pytree_node_class
@dataclass
class Transform:
    x: Any = 0.0
    y: Any = 0.0
    rotation: Any = 0.0
    mirror: bool = False
    magnification: Any = 1.0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def tree_flatten(self) -> tuple[tuple[Any, ...], bool]:
        return (self.x, self.y, self.rotation, self.magnification), self.mirror

    @classmethod
    def tree_unflatten(cls, mirror: bool, children: tuple[Any, ...]) -> Transform:
        x, y, rotation, magnification = children
        return cls(x, y, rotation, mirror, magnification)

    @property
    def disp(self) -> Array:
        return xp.stack([asarray(self.x), asarray(self.y)])

    def copy(self) -> Transform:
        t = Transform(self.x, self.y, self.rotation, self.mirror, self.magnification)
        if "klayout_angle" in self._cache:
            t._cache["klayout_angle"] = self._cache["klayout_angle"]
        return t

    def klayout_angle(self) -> float:
        """KLayout's angle (degrees) of this transformation, as kfactory reads it for instance names.

        While the rotation is the one taken from a KLayout composition (``component._kl_value``),
        this is that composition's own angle. Otherwise it is ``to_klayout().angle``: rebuilding a
        KLayout transformation from an angle KLayout already produced would round-trip it through
        sin/cos once more and change its last bits.
        """
        recorded = self._cache.get("klayout_angle")
        if recorded is not None and recorded[0] == to_float(self.rotation):
            return float(recorded[1])
        return float(self.to_klayout().angle)

    def matrix(self) -> Array:
        c = cos_deg(self.rotation)
        s = sin_deg(self.rotation)
        m = asarray(self.magnification)
        if self.mirror:
            mat = xp.stack([xp.stack([c, s]), xp.stack([s, -c])])
        else:
            mat = xp.stack([xp.stack([c, -s]), xp.stack([s, c])])
        return m * mat

    def is_identity(self) -> bool:
        try:
            return (
                to_float(self.x) == 0
                and to_float(self.y) == 0
                and to_float(self.rotation) % 360 == 0
                and not self.mirror
                and to_float(self.magnification) == 1
                and not _any_tracer(self)
            )
        except Exception:
            return False

    def apply(self, points: Any) -> Array:
        """Transforms (N, 2) or (2,) points."""
        pts = asarray(points)
        if self.is_identity():
            return pts
        mat = self.matrix()
        out = pts @ mat.T
        return out + self.disp

    def apply_vector(self, vector: Any) -> Array:
        return asarray(vector) @ self.matrix().T

    def apply_angle(self, angle: Any) -> Any:
        """Transforms an orientation (degrees). Result normalized to [0, 360)."""
        a = -angle if self.mirror else angle
        out = self.rotation + a
        return _mod360(out)

    def __mul__(self, other: Transform) -> Transform:
        """Composition: (self * other).apply(p) == self.apply(other.apply(p))."""
        rot_other = -other.rotation if self.mirror else other.rotation
        disp = self.apply(other.disp)
        return Transform(
            x=disp[0],
            y=disp[1],
            rotation=_mod360(self.rotation + rot_other),
            mirror=self.mirror != other.mirror,
            magnification=self.magnification * other.magnification,
        )

    def inverted(self) -> Transform:
        inv_mag = 1.0 / self.magnification
        if self.mirror:
            rot = self.rotation
        else:
            rot = -self.rotation
        t = Transform(0.0, 0.0, _mod360(rot), self.mirror, inv_mag)
        d = t.apply(self.disp)
        t.x, t.y = -d[0], -d[1]
        return t

    def to_klayout(self) -> Any:
        import klayout.db as kdb

        return kdb.DCplxTrans(
            to_float(self.magnification),
            to_float(self.rotation),
            bool(self.mirror),
            to_float(self.x),
            to_float(self.y),
        )

    @classmethod
    def from_klayout(cls, t: Any) -> Transform:
        return cls(
            x=float(t.disp.x),
            y=float(t.disp.y),
            rotation=float(t.angle),
            mirror=bool(t.is_mirror()),
            magnification=float(t.mag),
        )


def _mod360(a: Any) -> Any:
    if isinstance(a, int | float):
        return float(a) % 360
    a = asarray(a)
    return xp.mod(a, 360.0)


def _any_tracer(t: Transform) -> bool:
    from gdsfactory._jax import is_tracer

    return any(is_tracer(v) for v in (t.x, t.y, t.rotation, t.magnification))


def rotation_matrix(angle: Any) -> Array:
    c = cos_deg(angle)
    s = sin_deg(angle)
    return xp.stack([xp.stack([c, -s]), xp.stack([s, c])])


def rotate_points(points: Any, angle: Any, center: Any = (0.0, 0.0)) -> Array:
    pts = asarray(points)
    center = asarray(center)
    return (pts - center) @ rotation_matrix(angle).T + center


def normalize_angle(a: Any) -> Any:
    a = maybe_float(a)
    return _mod360(a)


def manhattan_angle(a: Any) -> int | None:
    """Returns orientation as int in {0, 90, 180, 270} or None for off-grid."""
    v = to_float(a) % 360
    q = v / 90
    if abs(q - round(q)) < 1e-6:
        return int(round(q) % 4) * 90
    return None


__all__ = [
    "Transform",
    "manhattan_angle",
    "normalize_angle",
    "rotate_points",
    "rotation_matrix",
]
