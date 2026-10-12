"""Analytic 3D solids of a ``StackDevice``'s levels, painted by ``mesh_order`` (viewer M1).

This builds the solids of spec D3 from the same data the M0 module
(:mod:`gdsfactory.viewer.geometry`) reads: the compiled ``levels``, the packed
``source_vertices`` of every source layer, and the device box. It matches the
geometry the model voxelizes exactly (spec C2), with no voxelization: a level at
height ``z`` is the level set of its lateral signed distance, offset by the
wall shift ``(z - z_ref) tan(angle) - bias - z_to_bias(z)`` with
``z_ref = zmin + width_to_z * thickness``.

Construction (spec D3 steps 2-7):

1. **Merge.** Each source layer's polygons are merged into their exact union
   (:func:`gdsfactory.viewer.geometry._layer_unions`).
2. **Split z.** Each level's ``[z_lo, z_hi]`` is split where the wall shift's
   slope changes (the ``z_to_bias`` knots) and where the offset crosses zero, so
   on every interval the offset ``r(z)`` is linear and of one sign.
3. **Prism or vertical walls.** A level with no sidewall angle and no
   ``z_to_bias`` is a prism of its 2D region (spec D3 step 3); a fill level is a
   slab of the box.
4. **Sloped walls.** Otherwise each literal is the union of a prism with the
   convex hull of every boundary edge's two end "stadiums" (spec D3 step 4),
   added where the level grows and subtracted where it shrinks. Planar faces are
   exact; the conical faces at vertices are tessellated to the fixed 1 nm chord
   tolerance (spec Q16). The union over edges reproduces the model's round joins
   at grown corners exactly, since for a point outside the union the distance to
   the union is the least distance to a boundary edge.
5. **Derived layers.** Expression nodes combine the literal solids with the
   model's booleans (``or`` = union, ``and`` = intersection, ``not`` = ``A - B``
   with ``B`` evaluated at the flipped offset, ``xor`` = symmetric difference).
6. **Fill levels.** The box slab between the level's z surfaces.
7. **Painting.** Every level's solid is clipped to the box and then has every
   higher-priority level (by ``mesh_order``, ties broken by the level's order)
   subtracted (spec Q13). The ``background`` material is the box minus all
   levels. The unpainted solid is kept too, for the "show full extent" toggle
   (spec R3).

The meshes are float32 positions measured from the box center (spec C5) and
uint32 triangle indices (spec D5). Nothing here is imported by
``import gdsfactory`` (spec C3): :mod:`gdsfactory.viewer` is opt-in.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, NamedTuple

import manifold3d as m3
import numpy as np
from shapely.geometry.polygon import orient

from gdsfactory.viewer.geometry import (
    ARC_CHORD_TOL,
    _box,
    _expr_region,
    _layer_unions,
    span,
    wall_shift,
)

__all__ = [
    "LevelSolid",
    "PortGlyph",
    "SolidMesh",
    "SolidSet",
    "analytic_solids",
    "background_mesh",
    "build",
    "level_solids",
]

_ROUND = m3.JoinType.Round
_EPS = 1e-9


class SolidMesh(NamedTuple):
    """One mesh: float32 positions (um, from the box center) and uint32 triangles."""

    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M, 3) uint32


class LevelSolid(NamedTuple):
    """One level's metadata and its two meshes (spec R1, R3, Q13)."""

    name: str
    material: str
    mesh_order: int
    fill: bool
    zmin: float
    thickness: float
    sidewall_angle: float
    bias: float
    width_to_z: float
    z_to_bias: tuple[tuple[float, ...], tuple[float, ...]] | None
    painted: SolidMesh
    full: SolidMesh


class PortGlyph(NamedTuple):
    """One port's metadata and the height its glyph is drawn at (spec R3, Q12).

    ``x``/``y``/``z`` are in the same local-origin frame as the level meshes
    (spec C5), so the browser draws the glyphs and the solids together.
    """

    name: str
    x: float
    y: float
    z: float
    orientation: float | None
    width: float
    layer: int


class SolidSet(NamedTuple):
    """The analytic manifolds behind :func:`build`, for tests and advanced use."""

    origin: tuple[float, float, float]
    size: tuple[float, float, float]
    full: dict[str, m3.Manifold]
    painted: dict[str, m3.Manifold]
    background: m3.Manifold
    box: m3.Manifold
    ports: tuple[PortGlyph, ...]


# ------------------------------------------------------------------ manifold helpers


def _circle_n(radius: float, tol: float = ARC_CHORD_TOL) -> int:
    """Segments per full circle so a round join's arc is within ``tol`` (spec Q16).

    A multiple of four, so a vertex lands on the edge's normal and its opposite
    (see :func:`_edge_hull`): the tangent there is then exact and the planar face
    next to the edge carries no tessellation error.
    """
    if radius <= tol:
        return 8
    n = math.ceil(math.pi / math.acos(max(-1.0, 1.0 - tol / radius)))
    return max(8, 4 * math.ceil(n / 4))


def _polys(geom: Any) -> list[Any]:
    """The polygons of a shapely Polygon / MultiPolygon / collection."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]


def _to_cs(geom: Any) -> m3.CrossSection:
    """A shapely region as a manifold cross-section (exteriors CCW, holes CW)."""
    contours = []
    for poly in _polys(geom):
        poly = orient(poly, 1.0)
        contours.append(np.asarray(poly.exterior.coords)[:-1])
        contours.extend(np.asarray(r.coords)[:-1] for r in poly.interiors)
    if not contours:
        return m3.CrossSection()
    return m3.CrossSection(contours)


def _offset_cs(cs: m3.CrossSection, delta: float) -> m3.CrossSection:
    """Grow (delta > 0) or shrink (delta < 0) a cross-section with round joins."""
    if cs.is_empty() or abs(delta) <= _EPS:
        return cs
    return cs.offset(delta, _ROUND, 2.0, _circle_n(abs(delta)))


def _prism(cs: m3.CrossSection, z0: float, z1: float) -> m3.Manifold:
    """Extrude a cross-section from ``z0`` to ``z1`` (a vertical-walled solid)."""
    if cs.is_empty() or z1 - z0 <= _EPS:
        return m3.Manifold()
    return m3.Manifold.extrude(cs, z1 - z0).translate((0.0, 0.0, z0))


def _union(a: m3.Manifold, b: m3.Manifold) -> m3.Manifold:
    if a.is_empty():
        return b
    if b.is_empty():
        return a
    return a + b


def _intersect(a: m3.Manifold, b: m3.Manifold) -> m3.Manifold:
    if a.is_empty() or b.is_empty():
        return m3.Manifold()
    return a ^ b


def _difference(a: m3.Manifold, b: m3.Manifold) -> m3.Manifold:
    if a.is_empty() or b.is_empty():
        return a
    return a - b


def _box_manifold(device: Any) -> m3.Manifold:
    """The device box as a solid (um), centered on the box center."""
    shape = device.partial_real_shape
    if shape is None or None in tuple(shape):
        raise ValueError(f"StackDevice {device.name!r} has no box (partial_real_shape)")
    sx, sy, sz = (float(v) * 1e6 for v in shape)
    cx, cy = device.center
    return m3.Manifold.cube((sx, sy, sz), center=True).translate((cx, cy, float(device.z_center)))


# ------------------------------------------------------------------ edge hulls


def _edges(cs: m3.CrossSection) -> list[tuple[np.ndarray, np.ndarray]]:
    """The boundary edges (um) of a cross-section, exterior and holes alike."""
    out = []
    for contour in cs.to_polygons():
        ring = np.asarray(contour, dtype=float)
        if ring.shape[0] < 2:
            continue
        closed = np.vstack([ring, ring[:1]]) if not np.allclose(ring[0], ring[-1]) else ring
        out.extend((closed[i], closed[i + 1]) for i in range(len(closed) - 1))
    return out


def _edge_hull(
    p0: np.ndarray, p1: np.ndarray, r0: float, r1: float, z0: float, z1: float
) -> m3.Manifold:
    """Convex hull of a boundary edge's stadiums at the two interval ends (spec D3 step 4).

    ``{(x, z) : dist(x, edge) <= |r(z)|}`` is convex for a linear ``r``; it is the
    convex hull of the edge grown by ``r0`` at ``z0`` and by ``r1`` at ``z1``.
    Each circle's vertices are offset to start on the edge's normal, so a vertex
    lies on the edge's normal and its opposite and the planar face along the edge
    carries no tessellation error (only the arcs do).
    """
    direction = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    angle = math.atan2(direction[1], direction[0])
    pts = []
    for p, r, z in ((p0, r0, z0), (p1, r0, z0), (p0, r1, z1), (p1, r1, z1)):
        if r <= ARC_CHORD_TOL:
            pts.append([float(p[0]), float(p[1]), float(z)])
        else:
            n = _circle_n(r)
            t = angle + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            pts.append(
                np.column_stack([p[0] + r * np.cos(t), p[1] + r * np.sin(t), np.full(n, z)])
            )
    points = np.vstack(pts)
    if points.shape[0] < 4:
        return m3.Manifold()
    return m3.Manifold.hull_points(points)


def _literal_sloped(
    cs: m3.CrossSection, r0: float, r1: float, z0: float, z1: float
) -> m3.Manifold:
    """One literal's solid over an interval where its offset is linear and one-signed.

    Growing (``r >= 0``): the prism of the union plus every edge's hull (spec D3
    step 4). Shrinking (``r <= 0``): the prism minus every edge's hull.
    """
    base = _prism(cs, z0, z1)
    hulls = m3.Manifold()
    for p0, p1 in _edges(cs):
        hulls = _union(hulls, _edge_hull(p0, p1, abs(r0), abs(r1), z0, z1))
    if r0 >= -_EPS and r1 >= -_EPS:
        return _union(base, hulls)
    if r0 <= _EPS and r1 <= _EPS:
        return _difference(base, hulls)
    raise ValueError(f"Offset changes sign on [{z0}, {z1}]: {r0} -> {r1}")


# ------------------------------------------------------------------ expression solids


def _literals(expr: tuple, sign: float = 1.0) -> list[tuple[int, float, float]]:
    """(layer index, grow, shift sign) of every non-fill literal of an expression.

    The subtracted operand of a ``not`` is evaluated at the flipped offset (spec
    C2), so its literal carries ``sign = -1``.
    """
    kind = expr[0]
    if kind == "fill":
        return []
    if kind == "layer":
        return [(int(expr[1]), float(expr[2]), sign)]
    if kind == "not":
        return _literals(expr[1], sign) + _literals(expr[2], -sign)
    return _literals(expr[1], sign) + _literals(expr[2], sign)


def _split_zs(level: Any, lo: float, hi: float) -> list[float]:
    """Heights splitting ``[lo, hi]`` so every literal's offset is linear and one-signed.

    Splits at the ``z_to_bias`` knots (where the offset's slope changes) and at
    every zero crossing of a literal's offset.
    """
    zs = [lo, hi]
    if level.z_to_bias is not None:
        zs += [level.zmin + f * level.thickness for f in level.z_to_bias[0]]
    knots = sorted({float(np.clip(z, lo, hi)) for z in zs})
    for _, grow, sign in _literals(level.expr):
        for a, b in itertools.pairwise(knots):
            ra = grow - sign * wall_shift(level, a)
            rb = grow - sign * wall_shift(level, b)
            if ra * rb < 0.0:
                knots.append(a + (b - a) * ra / (ra - rb))
        knots = sorted(set(knots))
    return [z for z in knots if lo - _EPS <= z <= hi + _EPS]


def _expr_solid(
    level: Any,
    expr: tuple,
    unions: dict[int, Any],
    union_cs: dict[int, m3.CrossSection],
    box_cs: m3.CrossSection,
    sign: float,
    z0: float,
    z1: float,
) -> m3.Manifold:
    """A level expression's solid over one interval (spec D3 steps 4-5)."""
    kind = expr[0]
    if kind == "fill":
        return _prism(box_cs, z0, z1)
    if kind == "layer":
        cs = union_cs.get(int(expr[1]))
        if cs is None or cs.is_empty():
            return m3.Manifold()
        grow = float(expr[2])
        r0 = grow - sign * wall_shift(level, z0)
        r1 = grow - sign * wall_shift(level, z1)
        if abs(r0 - r1) <= _EPS:
            return _prism(_offset_cs(cs, r0), z0, z1)
        return _literal_sloped(cs, r0, r1, z0, z1)
    grow = float(expr[3])
    if grow:
        raise NotImplementedError(
            f"Level {level.name!r} has a sloped wall and a sizing on a derived-layer "
            f"operation ({grow} um); only sizings inside the literals are supported"
        )
    a = _expr_solid(level, expr[1], unions, union_cs, box_cs, sign, z0, z1)
    b_sign = -sign if kind == "not" else sign
    b = _expr_solid(level, expr[2], unions, union_cs, box_cs, b_sign, z0, z1)
    if kind == "or":
        return _union(a, b)
    if kind == "and":
        return _intersect(a, b)
    if kind == "not":
        return _difference(a, b)
    if kind == "xor":
        return _union(_difference(a, b), _difference(b, a))
    raise ValueError(f"Unknown layer operation {kind!r}")


def _is_vertical(level: Any) -> bool:
    """Whether the level's wall shift is constant over its height (spec D3 step 3)."""
    return float(level.sidewall_angle) == 0.0 and level.z_to_bias is None


def _level_full(
    level: Any,
    unions: dict[int, Any],
    union_cs: dict[int, m3.CrossSection],
    box_poly: Any,
    box_cs: m3.CrossSection,
) -> m3.Manifold:
    """A level's full (unpainted) solid over its own z span, in the box (spec D3 steps 2-6)."""
    lo, hi = span(level)
    if hi - lo <= _EPS:
        return m3.Manifold()
    if _is_vertical(level):
        region = _expr_region(level.expr, unions, wall_shift(level, lo), box_poly)
        return _prism(_to_cs(region), lo, hi)
    solid = m3.Manifold()
    zs = _split_zs(level, lo, hi)
    for a, b in itertools.pairwise(zs):
        if b - a <= _EPS:
            continue
        solid = _union(solid, _expr_solid(level, level.expr, unions, union_cs, box_cs, 1.0, a, b))
    return solid


# ------------------------------------------------------------------ ports, painting, meshes


def _port_z(device: Any, layer: int) -> float:
    """The height a port's glyph is drawn at: the mid-height of its layer's level (spec Q12)."""
    from gdsfactory.fdtdx_stack import _expr_layers

    for level in device.levels:
        if layer in _expr_layers(level.expr):
            lo, hi = span(level)
            return 0.5 * (lo + hi)
    return float(device.z_center)


def _port_glyphs(device: Any, origin: tuple[float, float, float]) -> tuple[PortGlyph, ...]:
    """Port glyphs in the local-origin frame the meshes use (spec C5).

    ``origin`` is the box center the meshes are shifted by; the glyphs are
    drawn in the same ``content`` group as the meshes, so their coordinates
    must be local too.
    """
    ox, oy, oz = origin
    return tuple(
        PortGlyph(
            name=p.name,
            x=float(p.x) - ox,
            y=float(p.y) - oy,
            z=_port_z(device, int(p.layer)) - oz,
            orientation=None if p.orientation is None else float(p.orientation),
            width=float(p.width),
            layer=int(p.layer),
        )
        for p in device.ports
    )


def analytic_solids(device: Any) -> SolidSet:
    """Build the level solids (spec D3), their painting (Q13) and the background.

    Args:
        device: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.

    Returns:
        A :class:`SolidSet`: the full and painted manifold of every level, the
        ``background`` (box minus all levels), the box itself, the local origin
        and box size (um), and the port glyphs.
    """
    box_poly = _box(device)
    box_cs = _to_cs(box_poly)
    box_man = _box_manifold(device)
    unions = _layer_unions(device)
    union_cs = {index: _to_cs(geom) for index, geom in unions.items()}

    full = {
        level.name: _intersect(
            _level_full(level, unions, union_cs, box_poly, box_cs), box_man
        )
        for level in device.levels
    }

    order = sorted(range(len(device.levels)), key=lambda i: (device.levels[i].mesh_order, i))
    painted: dict[str, m3.Manifold] = {}
    covered = m3.Manifold()
    for i in order:
        name = device.levels[i].name
        painted[name] = _difference(full[name], covered)
        covered = _union(covered, full[name])
    background = _difference(box_man, covered)

    shape = device.partial_real_shape
    size = tuple(float(v) * 1e6 for v in shape)
    cx, cy = device.center
    origin = (float(cx), float(cy), float(device.z_center))
    return SolidSet(
        origin=origin,
        size=size,
        full=full,
        painted=painted,
        background=background,
        box=box_man,
        ports=_port_glyphs(device, origin),
    )


def _mesh(man: m3.Manifold, origin: tuple[float, float, float]) -> SolidMesh:
    """A manifold as a float32 local-origin mesh (spec C5, D5)."""
    mesh = man.to_mesh()
    vertices = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    vertices -= np.asarray(origin, dtype=np.float32)
    indices = np.asarray(mesh.tri_verts).astype(np.uint32)
    return SolidMesh(positions=vertices, indices=indices)


def level_solids(device: Any, solids: SolidSet) -> list[LevelSolid]:
    """Packages a :class:`SolidSet`'s manifolds into one :class:`LevelSolid` per level.

    Split out of :func:`build` so a caller that already holds the ``SolidSet``
    (the viewer's state, which slices the same manifolds for the 2D section)
    can reuse it instead of rebuilding the geometry (spec D3 risk: build once).

    Args:
        device: the :class:`~gdsfactory.fdtdx_stack.StackDevice` the solids were built from.
        solids: the manifolds from :func:`analytic_solids`.

    Returns:
        One :class:`LevelSolid` per ``device.levels``, in that order.
    """
    origin = solids.origin
    return [
        LevelSolid(
            name=level.name,
            material=level.material,
            mesh_order=int(level.mesh_order),
            fill=level.expr[0] == "fill",
            zmin=float(level.zmin),
            thickness=float(level.thickness),
            sidewall_angle=float(level.sidewall_angle),
            bias=float(level.bias),
            width_to_z=float(level.width_to_z),
            z_to_bias=level.z_to_bias,
            painted=_mesh(solids.painted[level.name], origin),
            full=_mesh(solids.full[level.name], origin),
        )
        for level in device.levels
    ]


def background_mesh(solids: SolidSet) -> SolidMesh:
    """The background mesh (box minus every level) of an already-built :class:`SolidSet`.

    Split out of :func:`build` alongside :func:`level_solids`, for callers that
    hold the ``SolidSet`` already.
    """
    return _mesh(solids.background, solids.origin)


def build(device: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], list[LevelSolid], SolidMesh, tuple[PortGlyph, ...]]:
    """Build the viewer's geometry for a device (spec M1): meshes, ports and the box.

    Args:
        device: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.

    Returns:
        ``(origin, size, levels, background, ports)``: the box center and size
        (um), one :class:`LevelSolid` per level (painted and full meshes), the
        background mesh, and the port glyphs.
    """
    solids = analytic_solids(device)
    levels = level_solids(device, solids)
    return solids.origin, solids.size, levels, background_mesh(solids), solids.ports
