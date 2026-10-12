"""Analytic cross-sections of a device's painted solids (viewer M3, spec R4, D4).

A section is a plane cut through the same analytic solids the 3D viewport draws
(spec D3), so the two viewports always agree (spec D4). The cut is done with
manifold3d on the manifolds :func:`gdsfactory.viewer.solids.analytic_solids`
already built, so the geometry is built once and sliced many times (spec R4's
slider).

The result is vector polygons: a plane through a closed manifold gives closed
rings, planar faces are cut exactly, and arcs carry the same 1 nm tessellation
as the 3D view (spec Q16). The solids are painted (spec Q13), so the rings are
disjoint and, with the ``background``, tile the section rectangle (spec D4).

Coordinates are measured from the box center, like the meshes (spec C5).

Only the ``xy`` plane slices directly (``Manifold.slice``); ``xz`` and ``yz``
rotate the manifold so the plane of interest becomes the XY plane, then map the
returned ``(a, b)`` back to the section's two axes. The mappings are pinned by
an asymmetric test (``tests/test_viewer_section.py``):

- ``xy``: slice at ``z``; ``(a, b) = (x, y)``.
- ``xz``: ``rotate([90, 0, 0])`` (``(x, y, z) -> (x, -z, y)``), slice at ``y``;
  ``(a, b) = (x, -z)``.
- ``yz``: ``rotate([0, -90, 0])`` (``(x, y, z) -> (-z, y, x)``), slice at ``x``;
  ``(a, b) = (-z, y)``.
"""

from __future__ import annotations

from typing import NamedTuple

import manifold3d as m3
import numpy as np

from gdsfactory.viewer.solids import SolidSet

__all__ = ["SectionSet", "section", "section_axis_names", "section_bounds", "section_normal_axis"]

#: The section planes the 2D viewport can show (spec R4).
PLANES = ("xy", "xz", "yz")

#: The rotated frame's slice height (its normal) for each plane, and the
#: ``(a, b) -> local (u, v)`` axis mapping. ``u``/``v`` are the section's own
#: axes: xy -> (x, y), xz -> (x, z), yz -> (z, y).
_ROPES = {"xy": (0.0, 0.0, 0.0), "xz": (90.0, 0.0, 0.0), "yz": (0.0, -90.0, 0.0)}


class SectionSet(NamedTuple):
    """One plane's cross-section: the plane, its position and the level rings.

    ``rings`` maps a level name to a tuple of ``(N, 2)`` float arrays (closed
    rings, in the local-origin frame). ``background`` is the box section minus
    every level. Rings may be outer or hole contours; the browser fills them
    even-odd, so orientation carries no meaning.
    """

    plane: str
    position: float
    rings: dict[str, tuple[np.ndarray, ...]]
    background: tuple[np.ndarray, ...]


def _rotated(man: m3.Manifold, plane: str) -> m3.Manifold:
    """The manifold oriented so the ``plane`` cut becomes an XY-plane cut."""
    return man.rotate(_ROPES[plane])


def _to_section_xy(rings: list[np.ndarray], plane: str, origin: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    """Maps a rotated slice's ``(a, b)`` rings to the section's local ``(u, v)``.

    ``origin`` is the box center; subtracting it gives the local-origin frame the
    meshes use (spec C5).
    """
    ox, oy, oz = origin
    out = []
    for ring in rings:
        a, b = ring[:, 0], ring[:, 1]
        if plane == "xy":
            u, v = a - ox, b - oy
        elif plane == "xz":
            u, v = a - ox, -b - oz
        else:  # yz
            u, v = -a - oz, b - oy
        out.append(np.column_stack([u, v]).astype(np.float64))
    return tuple(out)


def _slice(man: m3.Manifold, plane: str, position: float, origin: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    """Slices one manifold and returns its rings in the section's local frame."""
    cross = _rotated(man, plane).slice(float(position))
    if cross.is_empty():
        return ()
    return _to_section_xy([np.asarray(p) for p in cross.to_polygons()], plane, origin)


def section(solids: SolidSet, plane: str, position: float) -> SectionSet:
    """Cross-sections every painted level (and the background) at one plane.

    Args:
        solids: the manifolds from
            :func:`gdsfactory.viewer.solids.analytic_solids`.
        plane: ``"xy"``, ``"xz"`` or ``"yz"`` (spec R4).
        position: the plane's position (um) along its normal, in stack
            coordinates.

    Returns:
        A :class:`SectionSet` with one ring tuple per level name and one for
        ``background``.

    Raises:
        ValueError: on an unknown plane.
    """
    if plane not in PLANES:
        raise ValueError(f"Unknown plane {plane!r}; expected one of {PLANES}")
    origin = solids.origin
    rings = {name: _slice(man, plane, position, origin) for name, man in solids.painted.items()}
    return SectionSet(
        plane=plane,
        position=float(position),
        rings=rings,
        background=_slice(solids.background, plane, position, origin),
    )


def section_axis_names(plane: str) -> tuple[str, str]:
    """The stack-axis names the section's two axes ``(u, v)`` correspond to (spec R4)."""
    return {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("z", "y")}[plane]


def section_normal_axis(plane: str) -> str:
    """The stack-axis name the section's slider moves along (spec R4)."""
    return {"xy": "z", "xz": "y", "yz": "x"}[plane]


def section_bounds(size: tuple[float, float, float], plane: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """The section rectangle's local ``(u, v)`` bounds, centered on the origin (spec C5).

    ``u``/``v`` are the section's own axes (xy -> x, y; xz -> x, z; yz -> z, y),
    each the box extent along that axis.
    """
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (2, 1)}[plane]
    return (
        (-0.5 * size[axes[0]], 0.5 * size[axes[0]]),
        (-0.5 * size[axes[1]], 0.5 * size[axes[1]]),
    )
