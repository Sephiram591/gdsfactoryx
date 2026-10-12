"""Analytic 2D cross-sections of a :class:`StackDevice`'s levels (viewer M0).

This is the viewer's read-only geometry builder (spec D8). It reads only the
public data a ``StackDevice`` exposes before any simulation is set up --
``levels``, ``source_layers``, ``source_vertices``, ``source_sizes``,
``center``, ``z_center`` and ``partial_real_shape`` -- and never touches the
voxelizer, so it cannot conflict with changes to ``fdtdx_stack.py``.

The 2D region of a level at stack height ``z`` is the exact vector geometry the
model voxelizes (spec C2, D8): the merge of each source layer's polygons,
offset laterally by the level's wall shift, combined for derived layers with the
model's distance-field semantics (``or`` = min, ``and`` = max, ``not`` =
``max(a, -b)``), with each layer's sizing grow, and clipped to the device box.

A signed distance is offset laterally by ``s`` by taking the ``s``-level set of
the distance field, which is exactly a polygon offset of the layer's union. So
the region ``{sd + grow - shift < 0}`` is the union buffered by ``grow - shift``
(positive grows). Where the boundary moves outward, convex corners become
circular arcs; where it moves inward, convex corners stay sharp and concave
corners become arcs -- matching the model's round joins (spec Q17). Planar
faces are exact; the only approximation is the tessellation of the arcs, at a
fixed 1 nm chord tolerance (spec Q16).

``level_regions`` returns the unpainted region of every level (the D8 model
parity test uses this). ``slice_regions`` paints them by ``mesh_order`` and adds
the ``background`` region (spec Q13, M1 builds the matching solids).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, Polygon
from shapely.ops import unary_union

from gdsfactory.fdtdx_geometry import _edge_topology, split_t_junctions

__all__ = ["level_regions", "slice_regions"]

#: Fixed chord tolerance (um) for arc tessellation (spec Q16: 1 nm).
ARC_CHORD_TOL = 1e-3

_EMPTY = GeometryCollection()


def _quad_segs(radius: float, tol: float = ARC_CHORD_TOL) -> int:
    """Buffer segment count so a round join's arc is within ``tol`` of the true arc."""
    r = abs(radius)
    if r <= tol:
        return 1
    return max(1, math.ceil(math.pi / (2.0 * math.acos(1.0 - tol / r))))


def _buffer(geom: Any, radius: float) -> Any:
    """Grows (radius > 0) or shrinks (radius < 0) ``geom`` with round joins."""
    if geom.is_empty or radius == 0.0:
        return geom
    return geom.buffer(radius, quad_segs=_quad_segs(radius), join_style="round")


def _polygons(vertices: Any, sizes: Any) -> list[Any]:
    """Valid shapely polygons from one source layer's packed vertices (um)."""
    v = np.asarray(vertices, dtype=float)
    out: list[Any] = []
    start = 0
    for size in sizes:
        ring = v[start : start + int(size)]
        start += int(size)
        if ring.shape[0] < 3:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            out.append(poly)
        else:  # buffer(0) may split a self-touching polygon
            out.extend(g for g in poly.geoms if g.geom_type == "Polygon" and not g.is_empty)
    return out


def _layer_unions(device: Any) -> dict[int, Any]:
    """Exact union (merge) of each source layer's polygons, keyed by layer index.

    The model measures the distance to the union of a layer's polygons, so the
    merge is exact (spec D3 step 1). A layer with no polygons maps to empty.
    """
    unions: dict[int, Any] = {}
    for (_, index), vertices, sizes in zip(
        device.source_layers, device.source_vertices, device.source_sizes, strict=True
    ):
        polys = _polygons(vertices, sizes)
        unions[index] = unary_union(polys) if polys else _EMPTY
    return unions


def _box(device: Any) -> Any:
    """The device box's xy rectangle (um)."""
    shape = device.partial_real_shape
    if shape is None or None in tuple(shape):
        raise ValueError(f"StackDevice {device.name!r} has no box (partial_real_shape)")
    sx, sy = float(shape[0]) * 1e6, float(shape[1]) * 1e6
    cx, cy = device.center
    return Polygon(
        [(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)]
    )


def span(level: Any) -> tuple[float, float]:
    """Nominal (bottom, top) of a level in um (negative thickness grows down)."""
    z0, z1 = float(level.zmin), float(level.zmin) + float(level.thickness)
    return min(z0, z1), max(z0, z1)


def wall_shift(level: Any, z: float) -> float:
    """The level's lateral wall shift at height ``z`` (um); positive shrinks it.

    Reproduces the model's ``_wall_offset``: ``(z - z_ref) tan(angle) - bias -
    z_to_bias(z)`` with ``z_ref = zmin + width_to_z * thickness`` and
    ``z_to_bias`` interpolated at the fraction ``(z - zmin) / thickness``.
    """
    z_ref = level.zmin + level.width_to_z * level.thickness
    shift = (z - z_ref) * math.tan(math.radians(level.sidewall_angle)) - level.bias
    if level.z_to_bias is not None:
        zs, bs = level.z_to_bias
        shift -= float(np.interp((z - level.zmin) / level.thickness, zs, bs))
    return shift


def _expr_region(expr: tuple, unions: dict[int, Any], shift: float, box: Any) -> Any:
    """The level region ``{F + shift < 0}`` of one expression node at wall shift ``shift`` (um).

    ``or`` / ``and`` / ``xor`` are the union / intersection / symmetric
    difference of their operands with the same offset (spec C2). ``not`` is
    ``A - B``: ``{max(F_a, -F_b) + shift < 0}`` is ``{F_a + shift < 0}`` and
    ``{F_b - shift < 0}``, so the subtracted operand ``B`` is evaluated at the
    *flipped* offset ``-shift`` -- matching the model, which negates a
    subtracted literal's boundary with the offset's sign flipped. Each literal's
    sizing grow ``g`` buffers its layer's union by ``g - shift``; the node's own
    grow (the level's sizing) buffers the result. Fill literals are the whole
    plane, layers without polygons are empty.
    """
    kind = expr[0]
    if kind == "fill":
        return box  # {F_fill + shift < 0} is always true, for any shift
    if kind == "layer":
        union = unions.get(expr[1], _EMPTY)
        if union.is_empty:
            return _EMPTY
        return _buffer(union, float(expr[2]) - shift)
    a = _expr_region(expr[1], unions, shift, box)
    if kind == "not":
        b = _expr_region(expr[2], unions, -shift, box)
        region = a.difference(b)
    else:
        b = _expr_region(expr[2], unions, shift, box)
        if kind == "or":
            region = a.union(b)
        elif kind == "and":
            region = a.intersection(b)
        elif kind == "xor":
            region = a.symmetric_difference(b)
        else:
            raise ValueError(f"Unknown layer operation {kind!r}")
    grow = float(expr[3])
    return _buffer(region, grow) if grow else region


def _painted(device: Any, z: float) -> tuple[dict[int, Any], dict[str, Any], Any]:
    """Painted regions per level index, unpainted per level name, and background.

    Painting is by increasing ``mesh_order`` (lower wins), ties broken by the
    level's order in ``levels`` -- the model's ``sorted(..., key=(mesh_order, i))``
    (spec Q13). The background is the box minus everything painted.
    """
    unions = _layer_unions(device)
    box = _box(device)
    order = sorted(range(len(device.levels)), key=lambda i: (device.levels[i].mesh_order, i))
    painted: dict[int, Any] = {}
    unpainted: dict[str, Any] = {}
    covered = _EMPTY
    for i in order:
        level = device.levels[i]
        z_lo, z_hi = span(level)
        if z < z_lo or z > z_hi:  # outside its z span the model leaves the level empty
            unpainted[level.name] = _EMPTY
            painted[i] = _EMPTY
            continue
        region = _expr_region(level.expr, unions, wall_shift(level, z), box).intersection(box)
        unpainted[level.name] = region
        painted[i] = region.difference(covered)
        covered = covered.union(region)
    return painted, unpainted, box.difference(covered)


def level_regions(device: Any, z: float) -> dict[str, Any]:
    """Unpainted analytic region of every level at stack height ``z`` (um).

    Args:
        device: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
        z: stack height (um) of the xy slice.

    Returns:
        Level name -> shapely geometry (a polygon or multipolygon), clipped to
        the device box. A level that is empty there maps to an empty geometry.
    """
    return _painted(device, z)[1]


def slice_regions(device: Any, z: float) -> dict[str, Any]:
    """Painted regions at stack height ``z`` (um), plus the background.

    Each level's region is where it wins by ``mesh_order`` (spec Q13), so the
    returned geometries are disjoint and together with the ``"background"``
    entry they tile the device box.

    Args:
        device: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
        z: stack height (um) of the xy slice.

    Returns:
        Level name -> shapely geometry, and ``"background"`` for the box minus
        every level.
    """
    painted, _, background = _painted(device, z)
    out = {device.levels[i].name: painted[i] for i in range(len(device.levels))}
    out["background"] = background
    return out


def _layer_distances(device: Any, points: np.ndarray) -> dict[int, np.ndarray]:
    """Signed distance (um) from ``points`` to each source layer's polygon union.

    The model's own primitive (``fdtdx_geometry.signed_distance`` on the same
    T-junction-split edges the voxelizer uses), for the D8 parity test.
    """
    from gdsfactory.fdtdx_geometry import signed_distance

    points = np.asarray(points, dtype=float)
    out: dict[int, np.ndarray] = {}
    for (_, index), v0, sizes in zip(device.source_layers, device.source_vertices, device.source_sizes, strict=True):
        vertices = np.asarray(v0, dtype=float)
        tol = device._topology_tol(vertices)
        split, split_sizes = split_t_junctions(vertices, sizes, tol)
        v = vertices[split]
        nxt, ids, boundary = _edge_topology(v, split_sizes, tol)
        out[index] = np.asarray(signed_distance(points, v, v[nxt], ids, boundary))
    return out
