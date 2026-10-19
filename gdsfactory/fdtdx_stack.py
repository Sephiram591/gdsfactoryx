"""Differentiable fdtdx object for a whole component with its LayerStack.

:class:`StackDevice` voxelizes every :class:`~gdsfactory.technology.LayerLevel` of
a LayerStack (logical and derived layers, ``mesh_order`` priorities, sidewall
angle, ``width_to_z``, ``bias``, ``z_to_bias``, background levels) into one
N-material region, differentiable with respect to the polygon vertices of every
source layer and to the bottom height (zmin), thickness and sidewall angle of
every level. The device
also builds its simulation volume (:meth:`StackDevice.get_simulation_volume`)
and places sources and detectors on the component's ports
(``StackDevice.create_<source or detector>_at``).

fdtdx ``Device`` objects only mix two materials and :func:`fdtdx.apply_params`
only knows about them, so the stack is applied with :func:`apply_stack`, which
writes the stack into the arrays and then calls ``fdtdx.apply_params`` (regular
devices and sources are updated as usual, after the stack).

Geometry (lengths in grid cells inside, gds/stack units are um):

- each source layer gets a 2D signed distance field (see
  :func:`gdsfactory.fdtdx_geometry.signed_distance`); a derived level (a
  boolean of layers, or one layer whose polygons overlap) gets the distance to
  the exact polygons of its boolean, built once as a recipe of copied layer
  vertices and crossings of layer edges and evaluated from the traced
  vertices (``derived_shapes="exact"``; :func:`derive_polygons`), or, with
  ``derived_shapes="composite"`` and where a level's sizing below its root
  cannot be drawn that way, the layers' distances combined with ``or = min``,
  ``and = max``, ``not = max(a, -b)``; layers that cover the
  whole wafer (``fill_layers``, default WAFER) and background levels are
  everywhere inside;
- a level at height z offsets its distance by
  ``(z - z_ref) * tan(sidewall_angle) - bias - z_to_bias(z)`` with
  ``z_ref = zmin + width_to_z * thickness``; along z each voxel is split at the
  (possibly rough) level surfaces, and on each part each level is the
  half-plane given by its distance and gradient at the voxel center, or, in
  columns where a polygon corner or a crossing of layer edges is within reach,
  the exact intersection / union of the nearby edges' offset half-planes, with
  round joins at grown corners, wherever that local model is certified
  (otherwise the half-plane; ``corner_lines``), offset for the part's mid
  height (so a surface cutting a voxel gets the sidewall at its own height);
  the levels are painted by increasing ``mesh_order`` (lower wins) by clipping
  the voxel square exactly; the background material gets the rest;
- :class:`TopRoughness` displaces the top surface of a level; dips are filled
  by the levels deposited on it (whose bottom follows) or by whatever level
  covers that height;
- the permittivity of a voxel is the Kottke subpixel average along the normal
  of the voxel's own model: the voxel's eps variance is split into each z
  sub-interval's lateral part, along its kernel-weighted interface flux
  (tilted out of the plane by the sidewall angle), and the vertical profile
  (along z, tilted by the roughness slopes), with the moments of all its
  ``xy_subsamples`` sub-squares added up; variance whose flux cancels
  inside the voxel falls back to one-sided differences of the mean eps_inf.
  So interfaces in neighbouring voxels do not mix, and films thinner than a
  voxel and 2D simulations work (``averaging="subpixel"``, the default; needs
  diagonally anisotropic arrays, which the device requests). Or the
  inverse-permittivity average of its materials (``averaging="inverse"``,
  fdtdx's convention). With 3-component arrays each E component is sampled at
  its own Yee position, with its own normal (``yee_staggered``). Electric
  conductivity is averaged linearly.

When it is placed, the device warns (:class:`StackDiagnosticWarning`) about
parts of a level and gaps narrower than about 1.5 cells, and, for levels drawn
with the composite model, about overlapping polygons on a layer that a level
shrinks (each polygon's distance is shrunk, not their union's);
:meth:`StackDevice.diagnose` returns the findings as data.

Dispersion (fdtdx >= 0.6.2): materials may carry a ``fdtdx.DispersionModel``
(Lorentz / Drude poles; ``Material.permittivity`` is then eps_inf). All stack
materials get the same pole slots (one per distinct (omega_0, gamma), see
:func:`gdsfactory.fdtdx_geometry.align_material_poles`), so fdtdx allocates
exactly those slots. In every voxel the slot constants c1, c2 are written and
the pole strengths are mixed, ``c3 = sum_m w_m c3_m``, which mixes the
susceptibilities exactly: ``eps(omega) = eps_inf_mix + sum_m w_m chi_m(omega)``
with ``eps_inf_mix`` from ``averaging``. (With ``"subpixel"`` the components
along an interface are then the exact arithmetic mean of eps(omega); fdtdx's
pole coefficients are isotropic, so the normal component keeps an arithmetic
mean of chi.) Each slot costs memory and time over the whole simulation, so
poles shared between materials share a slot.

Example:
    ```python
    import fdtdx, jax
    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import StackDevice, apply_stack, init_stack_params

    stack = StackDevice.from_layer_stack(
        my_circuit(gap=0.2), gf.gpdk.LAYER_STACK,
        materials={"si": fdtdx.Material(permittivity=12.1), "sio2": fdtdx.Material(permittivity=2.1)},
        background="sio2",
        included_layers=["core"],                 # z range of the box: the core ...
        extend_ports=(("o1", 1.0), ("o2", 1.0)),  # ... straight waveguides out of both ports ...
        boundary_offset=(0, 0, 1, -1, 1, -1),     # ... and 1 um more in y and z
        name="stack",
    )
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    for obj, c in (
        stack.create_mode_plane_source_at("o1", (3, 4), wave_character=wave, port_offset=-0.5),
        stack.create_mode_overlap_detector_at("o2", (3, 4), wave_characters=(wave,)),
    ):
        objects.append(obj)
        constraints += c
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    params = {**params, **init_stack_params(objects)}

    def loss(gap):
        p = {**params, "stack": {**params["stack"], **objects["stack"].pack(my_circuit(gap=gap))}}
        arrays2, objects2, _ = apply_stack(arrays, objects, p, key)
        ...
    ```
"""

import dataclasses
import math
import warnings
from collections.abc import Callable, Container, Sequence
from functools import partial
from typing import Any, Literal, NamedTuple, Self

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.dispersion import DispersionModel, Pole, compute_pole_coefficients
from fdtdx.materials import (
    compute_allowed_dispersive_coefficients,
    compute_allowed_electric_conductivities,
    compute_allowed_permittivities,
    compute_ordered_names,
)
from fdtdx.objects.detectors.diffractive import DiffractiveDetector
from fdtdx.objects.object import PositionConstraint
from fdtdx.objects.sources.source import HardConstantAmplitudePlanceSource
from fdtdx.objects.static_material.static import StaticMultiMaterialObject
from fdtdx.typing import INVALID_SLICE_TUPLE_3D

from gdsfactory._jax import has_tracers, to_numpy
from gdsfactory.fdtdx_timing import StepTimings, lap, record_timings, timed
from gdsfactory.fdtdx_geometry import (
    DerivedRecipe,
    _boundary_links,
    _edge_topology,
    align_material_poles,
    crossing_changes,
    derive_polygons,
    gaussian_random_field,
    nearest_edges,
    pack_polygons,
    recipe_rings,
    run_recipe,
    signed_distance,
    split_t_junctions,
    topology_changes,
    warn_reversible_losses,
)

Array = jax.Array
Averaging = Literal["inverse", "subpixel"]
PortDirection = Literal["in", "out"]  # into the component through the port, or out of it

__all__ = [
    "TopRoughness",
    "StackDevice",
    "StackDiagnostic",
    "StackDiagnosticWarning",
    "StackOptions",
    "apply_stack",
    "init_stack_params",
    "record_timings",
    "StepTimings",
]

_BIG = 1e6  # "infinitely" far, in cells
# um: from_layer_stack keeps the levels this close to the box (beyond
# level_margin), to cover fdtdx's rounding of the box to whole cells and the
# half cell sampled below it on grids up to ~130 nm; the levels that do not
# reach the placed box are dropped in place_on_grid
_SELECT_SLACK = 0.1
_ANISOTROPY_SENTINEL = "__diagonal_anisotropy__"
# offsets (cells) from a voxel center to fdtdx's Ex, Ey, Ez Yee positions:
# Ex at (i + 1/2, j, k), Ey at (i, j + 1/2, k), Ez at (i, j, k + 1/2)
_YEE_E_SHIFTS = ((0.0, -0.5, -0.5), (-0.5, 0.0, -0.5), (-0.5, -0.5, 0.0))
# corner model (see _corner_fill), lengths in sides of the lateral square:
_CORNER_TOL = 1e-3  # slack of the certification radii and of the corner columns
_PLANE_TOL = 1e-4  # half-planes closer than this are the same line
_CORNER_RAMP = 0.25  # width of the ramps that blend the corner model in
# width of the ramps of a sample's certificate for its voxel (several sub-samples per voxel): every
# literal constant or single over this much more than the reach; the corner columns reach this far
_VOXEL_RAMP = 0.05
_TIP_AREA = 1e-9  # round-join tips smaller than this (fraction of the square) are ignored
_MITRE_MIN = 0.05  # no tip correction where 1 + cos(angle between the edge normals) is below
# corner columns whose corner model is near the half-planes (both at the Gauss heights) take the seam
# between the half-planes at the mean offset (the ordinary columns along the edge) and at the Gauss
# heights (see _corner_fill), per level on the union of the levels painted up to it: it fades out
# smoothly as their gap reaches _STRAIGHT_WIDTH times the seam's own size, but at least _STRAIGHT_TOL
# (fractions)
_STRAIGHT_WIDTH = 0.5
_STRAIGHT_TOL = 1e-4
# cells: a level's part of a voxel layer shorter than this takes its sidewall at the part's middle
# (the exact mean over its pieces divides by the part's length)
_PART_TINY = 1e-6
# 3-point Gauss-Legendre (t, weight) on [0, 1]
_GL3 = ((0.5 - 0.5 * math.sqrt(0.6), 5 / 18), (0.5, 8 / 18), (0.5 + 0.5 * math.sqrt(0.6), 5 / 18))
# kernel flux per unit lateral eps variance below which the variance falls back to the FD normal
_KOTTKE_ETA = 0.25
# normalized eps variances (and squared jumps) below this are too small to divide by: they count as
# unexplained (and the vertical part goes along z) with no derivative of the ratios, which in float32
# would square the tiny denominator below the smallest normal number (an inf, then NaN gradients)
_KOTTKE_TINY = 1e-12
# um: a vertex this close to a port face is on it (default of _face_intervals;
# the device uses its topology tolerance, see _topology_tol)
_FACE_TOL = 1e-3
# polygon topology tolerance (see StackDevice.topology_tol): database units of
# the layout (snapping to its grid moves each vertex by up to half a unit per
# axis, so a vertex on a slanted edge can end up sqrt(2) units off it), plus
# float32 epsilons times the largest coordinate (the round-off of float32
# parameters moves a vertex off an edge by at most sqrt(2) of them)
_TOPOLOGY_DBU = 2.0
_TOPOLOGY_EPS = 2.0
# geometry diagnostics (see StackDevice.diagnose):
_MIN_FEATURE = 1.5  # cells (divided by xy_subsamples): narrower parts and gaps are reported
_CHECK_ANGLE = 90.0  # degrees: KLayout's ignore_angle; edges facing each other at a larger angle are never close
_NARROW_ANGLE = 45.0  # degrees: the two sides of one corner meeting at a larger angle are the corner, not a narrow part
_CHECK_DBU = 1e-4  # um: grid of the KLayout regions (coarser only where 32-bit coordinates need it)
_CHECK_HEIGHTS = 64  # most cross-sections checked per level
_ROUND_TOL = 0.01  # of the threshold: round joins are drawn to within it, and narrower places are round-off


# ---------------------------------------------------------------- static stack description


class _Level(NamedTuple):
    """A LayerLevel compiled to static data (lengths in um, angles in degrees)."""

    name: str
    expr: tuple  # layer expression, see _compile_layer
    zmin: float
    thickness: float
    sidewall_angle: float
    width_to_z: float
    bias: float
    z_to_bias: tuple[tuple[float, ...], tuple[float, ...]] | None
    mesh_order: int
    material: str


class _Port(NamedTuple):
    """A component port as static data (um, degrees; orientation None if unset)."""

    name: str
    x: float
    y: float
    orientation: float | None
    width: float
    layer: int


def _topology_tol(tol: float | None, vertices: Any) -> float:
    """Distance (um) within which the polygon topology of a layer's vertices is found.

    Vertices this close to an edge lie on it (T-junctions), this close to each
    other coincide (seams), and this close to a port face are on it. The same
    rule serves the device (built from ``source_vertices``), the topology check
    and :meth:`StackDevice.retopologize`, so what the check keeps a rebuild keeps.

    Args:
        tol: the device's ``topology_tol`` (um), or None for ``_TOPOLOGY_DBU``
            database units of the layout.
        vertices: (E, 2) vertices (um) of the layer.

    Returns:
        ``tol`` plus the float32 round-off of the vertices: ``_TOPOLOGY_EPS``
        float32 epsilons times their largest finite coordinate.
    """
    if tol is None:
        from gdsfactory import kcl

        tol = _TOPOLOGY_DBU * float(kcl.dbu)
    v = np.abs(np.asarray(vertices, dtype=float))
    largest = float(v[np.isfinite(v)].max(initial=0.0))
    return float(tol) + _TOPOLOGY_EPS * float(np.finfo(np.float32).eps) * largest


def _read_port(port: Any) -> _Port:
    """A component's port as static data (gradients stopped)."""
    from gdsfactory.pdk import get_layer

    orientation = port.orientation
    return _Port(
        name=str(port.name),
        x=float(to_numpy(port.x)),
        y=float(to_numpy(port.y)),
        orientation=None if orientation is None else float(to_numpy(orientation)),
        width=0.0 if port.width is None else float(to_numpy(port.width)),
        layer=int(get_layer(port.layer)),
    )


# A corner of a port extension on the port face, in a layer's packed vertices:
# (i, i) is vertex i, (i, j) is where edge i -> j crosses the face, (-1, -1) is
# the port's own corner (center -+ width / 2 along the face)
_FaceEnd = tuple[int, int]
_PORT_CORNER: _FaceEnd = (-1, -1)


def _face_frame(orientation: Any, xp: Any) -> tuple[Any, Any]:
    """Outward normal (along the port's orientation) and face tangent of a port."""
    angle = xp.deg2rad(orientation)
    return xp.stack([xp.cos(angle), xp.sin(angle)]), xp.stack([-xp.sin(angle), xp.cos(angle)])


def _face_point(end: _FaceEnd, side: float, v: Any, center: Any, width: Any, orientation: Any, xp: Any) -> Any:
    """A corner of an extension on the port face (see ``_FaceEnd``).

    Args:
        end: the corner.
        side: -1 for the first corner along the face tangent, +1 for the second
            (only used by the port's own corners).
        v: (E, 2) packed vertices of the layer.
        center: (2,) port center.
        width: port width.
        orientation: port orientation (degrees).
        xp: numpy or jax.numpy.

    Returns:
        The (2,) point.
    """
    n, t = _face_frame(orientation, xp)
    i, j = end
    if i < 0:
        return xp.asarray(center) + side * 0.5 * width * t
    if i == j:
        return v[i]
    a, b = v[i], v[j]
    return a + (xp.dot(n, xp.asarray(center) - a) / xp.dot(n, b - a)) * (b - a)


def _extension_rect(
    ends: tuple[_FaceEnd, _FaceEnd], v: Any, center: Any, width: Any, orientation: Any, length: float, xp: Any
) -> Any:
    """Corners (4, 2) of one extension, counter-clockwise.

    Corners 0 and 3 are on the port face (ordered along the face tangent), 1 and
    2 at the far end, ``length`` along the port's orientation.
    """
    n, _ = _face_frame(orientation, xp)
    a = _face_point(ends[0], -1.0, v, center, width, orientation, xp)
    b = _face_point(ends[1], 1.0, v, center, width, orientation, xp)
    return xp.stack([a, a + length * n, b + length * n, b])


def _face_intervals(
    port: _Port, vertices: np.ndarray, sizes: Sequence[int], tol: float = _FACE_TOL
) -> list[tuple[_FaceEnd, _FaceEnd]]:
    """Where a layer meets a port's face: the cross-section the port extension continues.

    The union of the layer's polygons (nonzero winding per polygon, so abutting
    and overlapping polygons merge) is sliced along a line parallel to the port
    face, just inside the component. Each covered interval that overlaps the port
    (its width along the face) is returned by its two ends on the face itself:
    a polygon vertex there, else the crossing of the polygon edge with the face,
    so the extension follows the vertices when they move.

    Args:
        port: the port (with an orientation).
        vertices: (E, 2) packed vertices (um) of the layer.
        sizes: vertices per polygon.
        tol: distance (um) within which a vertex is on the face and two crossings
            coincide.

    Returns:
        ``(first end, second end)`` per interval, ordered along the face tangent.
    """
    if not len(vertices):
        return []
    n, t = _face_frame(port.orientation, np)
    rel = np.asarray(vertices, dtype=float) - np.array([port.x, port.y])
    dn, dt = rel @ n, rel @ t  # dn < 0: inside the component
    starts = np.cumsum([0, *sizes[:-1]])
    nxt = np.arange(1, len(vertices) + 1)
    ids = np.repeat(np.arange(len(sizes)), sizes)
    for start, size in zip(starts, sizes, strict=True):
        nxt[start + size - 1] = start

    # crossings of the slice line dn = -2 tol, with their winding direction
    a = dn + 2 * tol
    b = a[nxt]
    edges = np.flatnonzero((a < 0) != (b < 0))
    if not edges.size:
        return []
    s = dt[edges] + (dt[nxt[edges]] - dt[edges]) * a[edges] / (a[edges] - b[edges])
    sign = np.where(b[edges] >= 0, 1, -1)

    def end(e: int) -> _FaceEnd:
        i, j = int(e), int(nxt[e])
        if abs(dn[i]) <= tol:
            return (i, i)
        if abs(dn[j]) <= tol:
            return (j, j)
        return (i, j)

    # sweep along the face; crossings at one position are applied together, so
    # seams between abutting polygons do not end an interval
    order = np.argsort(s, kind="stable")
    winding = np.zeros(len(sizes), dtype=int)
    intervals, opened = [], None
    k = 0
    while k < len(order):
        group = [order[k]]
        k += 1
        while k < len(order) and s[order[k]] - s[group[0]] <= tol:
            group.append(order[k])
            k += 1
        before = winding.copy()
        for g in group:
            winding[ids[edges[g]]] += sign[g]
        was, now = bool(before.any()), bool(winding.any())
        if was == now:
            continue
        # an edge of a polygon that entered (or left) the union here
        g = next(g for g in group if bool(before[ids[edges[g]]]) == was and bool(winding[ids[edges[g]]]) == now)
        if now:
            opened = (s[g], end(edges[g]))
        else:
            intervals.append((opened[0], s[g], opened[1], end(edges[g])))
    half = 0.5 * port.width
    return [(ea, eb) for s0, s1, ea, eb in intervals if s1 > -half + tol and s0 < half - tol]


def _port_extensions(
    polygons: dict[int, list[np.ndarray]],
    ports: dict[str, _Port],
    extended: Sequence[tuple[str, float, tuple[int, ...]]],
    tol: float | None = None,
) -> tuple[dict[int, list[tuple[str, float, _FaceEnd, _FaceEnd]]], dict[int, list[np.ndarray]]]:
    """The port extensions of every layer (see ``extend_ports`` of :meth:`StackDevice.from_layer_stack`).

    Args:
        polygons: static polygons (um) per layer index.
        ports: the component's ports by name.
        extended: (port, length, layers it may continue) per extended port.
        tol: the device's ``topology_tol``: where a layer meets a port face is
            found within the layer's topology tolerance (:func:`_topology_tol`).

    Returns:
        extensions: per layer, (port, length, face end a, face end b) of each
            extension, in the order of ``extended``, then along the face.
        rectangles: per layer, the (4, 2) corners (um) of each extension.
    """
    extensions: dict[int, list[tuple[str, float, _FaceEnd, _FaceEnd]]] = {}
    rectangles: dict[int, list[np.ndarray]] = {}
    for name, length, layers in extended:
        p = ports[name]
        for index in layers:
            polys = polygons.get(index, [])
            own = np.concatenate(polys) if polys else np.zeros((0, 2))
            ends = _face_intervals(p, own, [len(q) for q in polys], _topology_tol(tol, own))
            if index == p.layer and not ends:
                ends = [(_PORT_CORNER, _PORT_CORNER)]  # nothing drawn at the face: the port's width
            for pair in ends:
                extensions.setdefault(index, []).append((name, float(length), *pair))
                rectangles.setdefault(index, []).append(
                    _extension_rect(pair, own, (p.x, p.y), p.width, p.orientation, float(length), np)
                )
    return extensions, rectangles


def _face_spans(ends: Sequence[tuple[_FaceEnd, _FaceEnd]], v: np.ndarray, port: _Port) -> np.ndarray:
    """The (m, 2, 2) corners on the port face of extensions ``ends`` of the layer with own vertices ``v``.

    Non-finite where a face end is an edge that no longer crosses the face.
    """
    if not ends:
        return np.zeros((0, 2, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray([
            [_face_point(end, side, v, (port.x, port.y), port.width, port.orientation, np)
             for end, side in ((a, -1.0), (b, 1.0))]
            for a, b in ends
        ], dtype=float)


def _span(level: Any) -> tuple[float, float]:
    """Nominal (bottom, top) of a level in um.

    Args:
        level: anything with ``zmin`` and ``thickness`` (negative thickness grows down).

    Returns:
        (bottom, top) stack heights.
    """
    z0, z1 = float(level.zmin), float(level.zmin) + float(level.thickness)
    return min(z0, z1), max(z0, z1)


def _sizing(layer: Any) -> float:
    """Uniform grow of a Logical/DerivedLayer from its sizings.

    gdsfactory applies the sizings one after another (``region.sized(x, y,
    mode)``, zipped like gdsfactory zips them). A signed distance field can only
    shift by a constant, so the steps are summed. That equals the sequence only
    when every step grows or every step shrinks with one mode: a shrink then a
    grow (opening) removes features narrower than twice the shrink, which a sum
    cannot, so such sequences are rejected rather than drawn differently.

    Args:
        layer: the layer.

    Returns:
        The grow in um.

    Raises:
        NotImplementedError: for different x and y sizings, steps that both grow
            and shrink, or steps with different modes.
    """
    from gdsfactory import kcl

    steps = [
        (int(x), int(y), int(m))
        for x, y, m in zip(layer.sizings_xoffsets, layer.sizings_yoffsets, layer.sizings_modes, strict=False)
        if x or y
    ]
    if any(x != y for x, y, _ in steps):
        raise NotImplementedError(f"Anisotropic layer sizing {steps} (x != y) is not supported in {layer}")
    if any(x > 0 for x, *_ in steps) and any(x < 0 for x, *_ in steps):
        raise NotImplementedError(
            f"Layer sizing {[x for x, *_ in steps]} dbu of {layer} both grows and shrinks (an opening "
            "or closing), which is not a single offset; only all-grow or all-shrink sequences are supported"
        )
    if len({m for *_, m in steps}) > 1:
        raise NotImplementedError(f"Layer sizing of {layer} mixes modes {[m for *_, m in steps]}")
    return float(sum(x for x, *_ in steps)) * float(kcl.dbu)


def _compile_layer(layer: Any, fill_layers: set[int]) -> tuple:
    """Turns a LogicalLayer / DerivedLayer into a static expression tree.

    Nodes are ``("fill", grow)``, ``("layer", index, grow)`` and
    ``(op, left, right, grow)`` with op in and / or / xor / not.

    Args:
        layer: LogicalLayer, DerivedLayer or anything ``get_layer`` accepts.
        fill_layers: layer indices that cover the whole plane.

    Returns:
        The expression tree.
    """
    from gdsfactory.pdk import get_layer
    from gdsfactory.technology.layer_stack import DerivedLayer, LogicalLayer

    if isinstance(layer, LogicalLayer):
        index = int(get_layer(layer.layer))
        if index in fill_layers:
            return ("fill", _sizing(layer))
        return ("layer", index, _sizing(layer))
    if isinstance(layer, DerivedLayer):
        op = {"&": "and", "|": "or", "^": "xor", "-": "not"}.get(layer.operation, layer.operation)
        return (
            op,
            _compile_layer(layer.layer1, fill_layers),
            _compile_layer(layer.layer2, fill_layers),
            _sizing(layer),
        )
    return _compile_layer(LogicalLayer(layer=layer), fill_layers)


def _compile_level(level: Any, fill_layers: set[int]) -> tuple:
    """Expression tree of a LayerLevel, including background levels.

    Args:
        level: the LayerLevel.
        fill_layers: layer indices that cover the whole plane.

    Returns:
        The expression tree (see :func:`_compile_layer`).
    """
    from gdsfactory.pdk import get_layer

    if not level.background:
        return _compile_layer(level.layer, fill_layers)
    expr: tuple = ("fill", 0.0)
    for lay in level.background_exclude_layers:
        expr = ("not", expr, ("layer", int(get_layer(lay)), 0.0), 0.0)
    return expr


def _expr_layers(expr: tuple) -> set[int]:
    """Indices of all polygon layers an expression reads."""
    if expr[0] == "fill":
        return set()
    if expr[0] == "layer":
        return {expr[1]}
    return _expr_layers(expr[1]) | _expr_layers(expr[2])


def _expr_can_draw(expr: tuple, nonempty: set[int]) -> bool:
    """Whether an expression can be nonempty, given the layers that have polygons."""
    kind = expr[0]
    if kind == "fill":
        return True
    if kind == "layer":
        return expr[1] in nonempty
    if kind == "not":
        return _expr_can_draw(expr[1], nonempty)
    if kind == "and":
        return _expr_can_draw(expr[1], nonempty) and _expr_can_draw(expr[2], nonempty)
    return _expr_can_draw(expr[1], nonempty) or _expr_can_draw(expr[2], nonempty)


def _expr_positive_layers(expr: tuple) -> set[int]:
    """Indices of the polygon layers that add to an expression (not the subtracted ones)."""
    if expr[0] == "fill":
        return set()
    if expr[0] == "layer":
        return {expr[1]}
    if expr[0] == "not":
        return _expr_positive_layers(expr[1])
    return _expr_positive_layers(expr[1]) | _expr_positive_layers(expr[2])


Field = tuple[Array, Array, Array]  # signed distance (cells) and its in-plane gradient


def _eval_expr(expr: tuple, sdf: dict[int, Field], shape: tuple[int, int], res_um: float) -> Field:
    """Signed distance of a layer expression and its in-plane gradient.

    Args:
        expr: expression tree (see :func:`_compile_layer`).
        sdf: (distance, gx, gy) per source layer index, (nx, ny) each, in cells.
        shape: (nx, ny).
        res_um: grid resolution in um (converts grows to cells).

    Returns:
        (distance, gx, gy), negative inside; layers without polygons are far outside.
    """

    def where(cond: Array, u: Field, v: Field) -> Field:
        return tuple(jnp.where(cond, ui, vi) for ui, vi in zip(u, v, strict=True))  # type: ignore[return-value]

    def neg(u: Field) -> Field:
        return (-u[0], -u[1], -u[2])

    def lo(u: Field, v: Field) -> Field:  # union
        return where(u[0] <= v[0], u, v)

    def hi(u: Field, v: Field) -> Field:  # intersection
        return where(u[0] >= v[0], u, v)

    zero = jnp.zeros(shape)
    kind = expr[0]
    if kind == "fill":
        out, grow = (jnp.full(shape, -_BIG), zero, zero), expr[1]
    elif kind == "layer":
        out, grow = sdf.get(expr[1], (jnp.full(shape, _BIG), zero, zero)), expr[2]
    else:
        u = _eval_expr(expr[1], sdf, shape, res_um)
        v = _eval_expr(expr[2], sdf, shape, res_um)
        if kind == "or":
            out = lo(u, v)
        elif kind == "and":
            out = hi(u, v)
        elif kind == "not":
            out = hi(u, neg(v))
        elif kind == "xor":
            out = hi(lo(u, v), neg(hi(u, v)))
        else:
            raise ValueError(f"Unknown layer operation {kind!r}")
        grow = expr[3]
    return (out[0] - grow / res_um, out[1], out[2]) if grow else out


_TRUE: tuple = ("true",)
_FALSE: tuple = ("false",)


def _literals(
    expr: tuple, res_um: float, layers: Container[int]
) -> tuple[tuple, tuple[tuple[int, int, float], ...]]:
    """A level's region as a boolean tree of offset source-layer literals.

    The region of a level with lateral offset s (cells) is ``{F + s < 0}`` for
    the distance F of :func:`_eval_expr`. ``compile(node, a, b)`` is the region
    ``{F_node + a * s + b < 0}``: ``or`` / ``and`` are the union / intersection
    of their operands, the subtracted operands of ``not`` and ``xor`` are
    negated with the offset's sign flipped, and grows move b. A literal
    ``(i, a, b)`` is ``{sd_i + a * s + b < 0}`` for the distance sd_i of source
    layer i. Fill layers are true and layers without polygons false; both are
    folded away. Equal regions up to their boundaries.

    Args:
        expr: expression tree (see :func:`_compile_layer`).
        res_um: grid resolution in um (converts grows to cells).
        layers: indices of the source layers that have polygons.

    Returns:
        tree: ``("lit", k)``, ``("and", x, y)``, ``("or", x, y)``,
            ``("neg", x)``, ``("true",)`` or ``("false",)``; negations only
            wrap literals' subtrees, never a constant.
        literals: ``(layer index, a, b)`` per literal k, a = +-1, b in cells;
            equal literals share one k.
    """
    literals: list[tuple[int, int, float]] = []

    def lit(i: int, a: int, b: float) -> tuple:
        if (i, a, b) not in literals:
            literals.append((i, a, b))
        return ("lit", literals.index((i, a, b)))

    def both(x: tuple, y: tuple) -> tuple:
        if _FALSE in (x, y):
            return _FALSE
        return y if x == _TRUE else x if y == _TRUE else ("and", x, y)

    def either(x: tuple, y: tuple) -> tuple:
        if _TRUE in (x, y):
            return _TRUE
        return y if x == _FALSE else x if y == _FALSE else ("or", x, y)

    def neg(x: tuple) -> tuple:
        return _FALSE if x == _TRUE else _TRUE if x == _FALSE else ("neg", x)

    def compile_(node: tuple, a: int, b: float) -> tuple:
        kind = node[0]
        if kind == "fill":
            return _TRUE
        if kind == "layer":
            return lit(node[1], a, b - node[2] / res_um) if node[1] in layers else _FALSE
        u, v, grow = node[1], node[2], node[3]
        b = b - grow / res_um
        if kind == "or":
            return either(compile_(u, a, b), compile_(v, a, b))
        if kind == "and":
            return both(compile_(u, a, b), compile_(v, a, b))
        if kind == "not":
            return both(compile_(u, a, b), neg(compile_(v, -a, -b)))
        if kind == "xor":  # hi(lo(u, v), neg(hi(u, v)))
            inner = both(compile_(u, -a, -b), compile_(v, -a, -b))
            return both(either(compile_(u, a, b), compile_(v, a, b)), neg(inner))
        raise ValueError(f"Unknown layer operation {kind!r}")

    tree = compile_(expr, 1, 0.0)
    return tree, tuple(literals)


def _eval_bool(tree: tuple, values: Sequence[Any]) -> Any:
    """Evaluates a literal tree (see :func:`_literals`).

    Args:
        tree: the tree.
        values: boolean arrays, one per literal.

    Returns:
        The tree's value (a Python bool for a constant tree).
    """
    kind = tree[0]
    if kind == "lit":
        return values[tree[1]]
    if kind in ("true", "false"):
        return kind == "true"
    if kind == "neg":
        return ~_eval_bool(tree[1], values)
    x, y = _eval_bool(tree[1], values), _eval_bool(tree[2], values)
    return x & y if kind == "and" else x | y


def _unit_normal(gx: Array, gy: Array) -> Array:
    """Unit vectors (..., 2) along (gx, gy); (1, 0) where the gradient vanishes (no layer near)."""
    norm2 = gx**2 + gy**2
    ok = norm2 > 1e-12
    scale = jax.lax.rsqrt(jnp.where(ok, norm2, 1.0))
    return jnp.stack([jnp.where(ok, gx * scale, 1.0), jnp.where(ok, gy * scale, 0.0)], axis=-1)


def _layer_key(index: int) -> str:
    """Parameter name ``vertices/<layer name>`` (or the index if it has no name)."""
    from gdsfactory.pdk import get_layer_name

    try:
        return f"vertices/{get_layer_name(index)}"
    except Exception:
        return f"vertices/{index}"


def _clip(
    poly: Array, count: Array, n: Array, s: Array, attr: Array | None = None, new_attr: Array | None = None
) -> tuple[Array, ...]:
    """Clips convex polygons to the half-planes ``n . p + s >= 0``.

    With ``attr``, every vertex carries the attributes of its outgoing edge: a
    kept vertex keeps its own, a crossing that leaves the kept side starts an
    edge on the clipping line (``new_attr``), and a crossing that enters it
    continues the edge it lies on.

    Args:
        poly: (..., V, 2) vertices; the first ``count`` are valid, the rest repeat
            the last valid one.
        count: (...,) number of valid vertices.
        n: (..., 2) normals.
        s: (...,) offsets.
        attr: (..., V, A) attributes of the vertices' outgoing edges, or None.
        new_attr: (..., A) attributes of the edges on the clipping line.

    Returns:
        The clipped polygons (..., V, 2) in the same layout and their vertex
        counts; with ``attr`` also the clipped polygons' attributes (..., V, A).
    """
    num = poly.shape[-2]
    i = jnp.arange(num)
    valid = i < count[..., None]
    nxt = jnp.where(i + 1 < count[..., None], i + 1, 0)
    q = jnp.take_along_axis(poly, nxt[..., None], axis=-2)
    fp = jnp.sum(poly * n[..., None, :], axis=-1) + s[..., None]
    fq = jnp.sum(q * n[..., None, :], axis=-1) + s[..., None]
    keep = valid & (fp >= 0)
    cross = valid & ((fp >= 0) != (fq >= 0))
    # an edge (nearly) on the line has no well-defined crossing: take its kept
    # end, which changes no area and keeps the derivatives finite
    denom = fp - fq
    ok = cross & (jnp.abs(denom) > 16 * jnp.finfo(poly.dtype).eps)
    t = jnp.where(ok, fp / jnp.where(ok, denom, 1.0), jnp.where(fp >= 0, 0.0, 1.0))
    x = poly + t[..., None] * (q - poly)
    cand = jnp.stack([poly, x], axis=-2).reshape(*poly.shape[:-2], 2 * num, 2)
    if attr is not None:
        x_attr = jnp.where((fp >= 0)[..., None], new_attr[..., None, :], attr)
        cand_attr = jnp.stack([attr, x_attr], axis=-2).reshape(*poly.shape[:-2], 2 * num, attr.shape[-1])
        cand = jnp.concatenate([cand, cand_attr], axis=-1)  # one gather for both
    mask = jnp.stack([keep, cross], axis=-1).reshape(*poly.shape[:-2], 2 * num)
    cum = jnp.cumsum(mask, axis=-1)
    new_count = cum[..., -1]
    # slot j takes the (j+1)-th kept candidate; later slots repeat the last one
    # (an empty result still gathers in bounds: out-of-bounds gathers fill NaN,
    # which the masked area ignores but its derivative does not)
    j = jnp.minimum(i, jnp.maximum(new_count[..., None] - 1, 0))
    src = jnp.minimum(jnp.sum(cum[..., None, :] <= j[..., :, None], axis=-1), 2 * num - 1)
    out = jnp.take_along_axis(cand, src[..., None], axis=-2)
    if attr is None:
        return out, new_count
    return out[..., :2], new_count, out[..., 2:]


def _area(poly: Array, count: Array) -> Array:
    """Area of counter-clockwise polygons (shoelace).

    Args:
        poly: (..., V, 2) vertices, the first ``count`` valid.
        count: (...,) number of valid vertices.

    Returns:
        (...,) areas.
    """
    i = jnp.arange(poly.shape[-2])
    q = jnp.take_along_axis(poly, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    cross = poly[..., 0] * q[..., 1] - q[..., 0] * poly[..., 1]
    return 0.5 * jnp.sum(jnp.where(i < count[..., None], cross, 0.0), axis=-1)


def _moments(
    poly: Array, count: Array, attr: Array, kernel: tuple[float, float, float]
) -> tuple[Array, Array, Array]:
    """Area and kernel-weighted interface flux of counter-clockwise polygons.

    Every edge carries the attributes ``(n_x, n_y, b, ...)`` of the interface it
    lies on (:func:`_clip`), the plane ``n . p + b z + c = 0`` with a unit
    in-plane normal n, then further rates (``b`` times a lag, see
    :func:`_paint_layer`), or 0 on edges that are not interfaces. ``ell = du n_y - dv n_x``
    of an edge (du, dv) is then +-its length on its line (0 elsewhere, and no
    square root), and ``ell * (n, b)`` is the edge's area vector per unit
    height, pointing into the polygon (for either sign of the attributes).
    The kernel ``w = psi(ax + beta u) psi(ay + beta v)``, ``psi(x) = 1 - 4 x^2``,
    is integrated exactly along each edge (3-point Gauss-Legendre, degree 4).

    Args:
        poly: (..., V, 2) vertices, the first ``count`` valid.
        count: (...,) number of valid vertices.
        attr: (..., V, A) attributes of the vertices' outgoing edges, A >= 3.
        kernel: (ax, ay, beta): the kernel's center (floats or traced
            scalars) and its scale.

    Returns:
        area: (...,) areas (as :func:`_area`).
        flux: (..., 3) sum over the edges of ``ell * mean(w) * (n, b)``.
        rate: (..., A - 2) sum over the edges of ``ell * attr[2:]``: first the
            rate of change of the area with height when the planes move (p and
            z in one unit), then the further rates.
    """
    ax, ay, beta = kernel
    area = _area(poly, count)
    i = jnp.arange(poly.shape[-2])
    q = jnp.take_along_axis(poly, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    du, dv = q[..., 0] - poly[..., 0], q[..., 1] - poly[..., 1]
    ell = jnp.where(i < count[..., None], du * attr[..., 1] - dv * attr[..., 0], 0.0)
    omega = 0.0
    for t, weight in _GL3:
        xi = ax + beta * (poly[..., 0] + t * du)
        eta = ay + beta * (poly[..., 1] + t * dv)
        omega = omega + weight * (1 - 4 * xi**2) * (1 - 4 * eta**2)
    flux = jnp.sum((ell * omega)[..., None] * attr[..., :3], axis=-2)
    return area, flux, jnp.sum(ell[..., None] * attr[..., 2:], axis=-2)


def _cross(x: Array, y: Array) -> Array:
    """z component of the cross product of (..., 2) vectors."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _angle(x: Array, y: Array) -> Array:
    """Signed angle from x to y, (..., 2) vectors; 0 where |x| |y| < 1e-12."""
    ok = jnp.sum(x * x, axis=-1) * jnp.sum(y * y, axis=-1) >= 1e-24
    cr = jnp.where(ok, _cross(x, y), 0.0)
    dot = jnp.where(ok, jnp.sum(x * y, axis=-1), 1.0)
    return jnp.where(ok, jnp.arctan2(cr, dot), 0.0)


def _disk_area(poly: Array, count: Array, center: Array, r: Array) -> Array:
    """Area of counter-clockwise convex polygons inside disks.

    Sums, over the edges p -> q taken relative to the center, the signed area
    of the triangle (center, p, q) inside the disk: a circular sector where the
    edge is outside the circle, the triangle where it is inside.

    Args:
        poly: (..., V, 2) vertices, the first ``count`` valid.
        count: (...,) number of valid vertices.
        center: (..., 2) disk centers.
        r: (...,) radii.

    Returns:
        (...,) areas.
    """
    i = jnp.arange(poly.shape[-2])
    p = poly - center[..., None, :]
    q = jnp.take_along_axis(p, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    dd = q - p
    r2 = (r * r)[..., None]
    a = jnp.sum(dd * dd, axis=-1)
    b = jnp.sum(p * dd, axis=-1)
    disc = b * b - a * (jnp.sum(p * p, axis=-1) - r2)
    edge = a >= 1e-12  # padding repeats the last vertex
    hit = edge & (disc > 1e-12)  # the edge's line crosses the circle
    root = jnp.sqrt(jnp.where(hit, disc, 1.0))
    a = jnp.where(hit, a, 1.0)
    # the part of the edge inside the circle; without one, the whole edge is a sector
    t1 = jnp.where(hit, jnp.clip((-b - root) / a, 0.0, 1.0), 1.0)
    t2 = jnp.where(hit, jnp.clip((-b + root) / a, 0.0, 1.0), 1.0)
    p1 = p + t1[..., None] * dd
    p2 = p + t2[..., None] * dd
    area = 0.5 * r2 * (_angle(p, p1) + _angle(p2, q)) + 0.5 * _cross(p1, p2)
    keep = (i < count[..., None]) & edge & (r[..., None] >= 1e-9)
    return jnp.sum(jnp.where(keep, area, 0.0), axis=-1)


def _sidewall_shift(level: _Level, z: Array, zb: Array, t: Array, angle: Array, res_um: float) -> Array:
    """Lateral offset of a level's boundary at heights z.

    ``(z - z_ref) tan(angle) - bias - z_to_bias(z)`` with ``z_ref = zb +
    width_to_z * t``; ``z_to_bias`` is interpolated at the fraction ``(z - zb) / t``
    of the thickness (0 at zmin, 1 at zmin + thickness, also for negative
    thicknesses). It is added to the level's signed distance: positive shrinks
    the level.

    Args:
        level: the level (``width_to_z``, ``bias``, ``z_to_bias``).
        z: heights (cells).
        zb: the level's zmin (cells, same frame as z).
        t: the level's thickness (cells, negative grows down).
        angle: sidewall angle (degrees).
        res_um: grid resolution in um (converts the biases to cells).

    Returns:
        Offsets (cells) shaped like z, in z's dtype.
    """
    z_ref = zb + level.width_to_z * t
    grow = level.bias / res_um
    if level.z_to_bias is not None:
        zs, bs = level.z_to_bias
        grow = grow + jnp.interp((z - zb) / t, jnp.asarray(zs), jnp.asarray(bs)) / res_um
    return ((z - z_ref) * jnp.tan(jnp.deg2rad(angle)) - grow).astype(z.dtype)


def _paint_levels(
    poly: Array,
    count: Array,
    area0: Array,
    n: Array,
    o: Array,
    present: Array,
    is_fill: Sequence[bool],
    half: float,
    attr: Array | None = None,
    slope: Array | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[Array, ...]:
    """Paints levels in priority order into convex cells of the lateral square.

    Level j gets the part of the cell inside its half-plane ``n . p + o < 0``
    but outside every earlier present level, computed exactly by clipping the
    cell with the complements in priority order; fill levels take the whole
    rest of the cell; the background gets what is left. With ``attr``, every
    edge a clip creates carries its level's plane ``(n, slope)`` and each
    level's part also gets its interface moments (:func:`_moments`).

    Args:
        poly: (..., V, 2) cells (convex, counter-clockwise), the first
            ``count`` vertices valid; V must hold the vertices the clips add.
        count: (...,) number of valid vertices.
        area0: (...,) area of the cells over the square's.
        n: (..., K, 2) unit normals of the levels' half-planes.
        o: (..., K) their offsets (sides of the square).
        present: (..., K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the lateral square.
        attr: (..., V, 2 + R) the cells' edge attributes (see
            :func:`_moments`; 0 on edges that are not interfaces), or None:
            fractions only.
        slope: (..., K, R) z slope of each level's half-plane, then its
            further rates (with ``attr``).
        kernel: (ax, ay, beta) of :func:`_moments` (with ``attr``).

    Returns:
        visible: (..., K + 1) fractions of the square: the K levels, then the
            background.
        overflow: (...,) a clip produced more vertices than V.
        flux: (..., K, 3) with ``attr``: the kernel flux of each level's part.
        rate: (..., K, R) with ``attr``: its rates (see :func:`_moments`).
    """
    full_area = (2 * half) ** 2
    area = area0  # uncovered fraction so far
    overflow = jnp.zeros(count.shape, dtype=bool)
    visible = []
    moments = attr is not None
    if moments:
        _, flux, rate = _moments(poly, count, attr, kernel)  # of the uncovered part so far
        fluxes, rates = [], []
    for j, fill in enumerate(is_fill):
        here = present[..., j]
        if fill:
            visible.append(jnp.where(here, area, 0.0))
            area = jnp.where(here, 0.0, area)
            count = jnp.where(here, 0, count)
            if moments:
                fluxes.append(jnp.where(here[..., None], flux, 0.0))
                rates.append(jnp.where(here[..., None], rate, 0.0))
                flux = jnp.where(here[..., None], 0.0, flux)
                rate = jnp.where(here[..., None], 0.0, rate)
            continue
        # beyond the square's half-diagonal a line keeps or removes it whole
        s = jnp.clip(o[..., j], -2 * half, 2 * half)
        if moments:
            new_attr = jnp.concatenate([n[..., j, :], slope[..., j, :]], axis=-1)
            clipped, c_count, c_attr = _clip(poly, count, n[..., j, :], s, attr, new_attr)
            c_area, c_flux, c_rate = _moments(clipped, c_count, c_attr, kernel)
            c_area = c_area / full_area  # the part outside level j
            fluxes.append(jnp.where(here[..., None], flux - c_flux, 0.0))
            rates.append(jnp.where(here[..., None], rate - c_rate, 0.0))
            flux = jnp.where(here[..., None], c_flux, flux)
            rate = jnp.where(here[..., None], c_rate, rate)
            attr = jnp.where(here[..., None, None], c_attr, attr)
        else:
            clipped, c_count = _clip(poly, count, n[..., j, :], s)
            c_area = _area(clipped, c_count) / full_area  # the part outside level j
        visible.append(jnp.where(here, area - c_area, 0.0))
        overflow = overflow | (here & (c_count > poly.shape[-2]))
        poly = jnp.where(here[..., None, None], clipped, poly)
        count = jnp.where(here, c_count, count)
        area = jnp.where(here, c_area, area)
    visible.append(area)
    if moments:
        return jnp.stack(visible, axis=-1), overflow, jnp.stack(fluxes, axis=-2), jnp.stack(rates, axis=-2)
    return jnp.stack(visible, axis=-1), overflow


def _weigh_levels(x: Array, level_eps: Sequence[Sequence[float]], power: int = 1, axis: int = -1) -> Array:
    """Sums per-level values weighted by the levels' contrasts, per E component.

    Args:
        x: per-level values, levels along ``axis`` (K).
        level_eps: (C, K) static contrast of each level per E component.
        power: power of the contrasts.
        axis: the level axis of x; it becomes the component axis (C).

    Returns:
        ``sum_j level_eps[c, j] ** power * x_j`` per component c.
    """
    x = jnp.moveaxis(x, axis, -1)
    out = []
    for row in level_eps:
        terms = [float(e) ** power * x[..., j] for j, e in enumerate(row) if e != 0]
        out.append(sum(terms[1:], terms[0]) if terms else jnp.zeros_like(x[..., 0]))
    return jnp.moveaxis(jnp.stack(out, axis=-1), -1, axis)


@timed("normal_tensor")
def _normal_tensor(
    length: Array,
    bp: Array,
    flux: Array,
    mean: Array,
    square: Array,
    rate: Array,
    cell: float,
    kmass: float,
    slopes: Array | None = None,
) -> Array:
    """Squared-normal tensor of the voxels of one layer from their own model.

    The eps variance of a voxel (contrasts normalized by the eps range) splits
    exactly over its z sub-intervals s into a lateral part, ``sum_s len_s
    Var_s``, and a vertical one, the variance along z of the sub-intervals'
    means. The lateral variance of sub-interval s goes along its kernel-weighted
    interface flux ``F_s = int w grad(eps)`` (whose z component carries the
    sidewall tilt), with weight ``1 - beta``, where beta ramps from 0 to 1 as
    ``|F_s| / Var_s`` (both in sides of the lateral square) falls below
    ``_KOTTKE_ETA``: flux that cancels inside the sub-interval (a strip of one
    level between two of another, corner slivers) is unexplained variance,
    returned separately. The vertical variance is taken after removing the
    drift of the means between sub-intervals that tapered sidewalls cause (so
    only the surfaces' jumps count) and goes along z, or, with ``slopes``,
    along the jump-weighted mean of the rough surfaces' normals ``(-g, 1)``.
    The drift is that of sidewalls placed at each sub-interval's middle: a
    level painted at the middle of its own part of the voxel (see
    :func:`_level_walls`) gets its lag to the sub-interval's middle added to
    the means first (the rate's second channel), so a tilted wall whose part
    another level's surface cuts keeps its exact normal.
    The ratios are not differentiated below ``_KOTTKE_TINY`` (a sub-interval's
    variance there is unexplained, a vertical part goes along z), so the
    gradients stay finite in float32 at any sliver of any contrast.

    Args:
        length: (..., S) sub-interval lengths (cells; summing to one).
        bp: (..., S + 1) sorted sub-interval ends (cells).
        flux: (..., S, C, 3) kernel-weighted contrast flux per sub-interval
            (see :func:`_moments`).
        mean: (..., S, C) mean contrast per sub-interval.
        square: (..., S, C) mean squared contrast per sub-interval.
        rate: (..., S, C, 2) cell times the rate of change of the mean with
            height, and cell times the mean's change from moving each wall
            from its painted height to the sub-interval's middle (see
            :func:`_paint_layer`).
        cell: cells per side of the lateral square.
        kmass: kernel integral over the area the flux comes from, in units in
            which ``flux / kmass`` is per side of the kernel's square.
        slopes: (..., S - 1, 2) lateral gradient (cells per cell) of the surface
            at each inner end, or None (flat surfaces).

    Returns:
        (..., C, 4): the variance along x, y, z (variance times squared normal
        components, summed), then the unexplained variance.
    """
    v_lat = jnp.maximum(square - mean**2, 0.0)
    f = flux / kmass
    f2 = jnp.sum(f**2, axis=-1)
    low = f2 < (_KOTTKE_ETA * v_lat) ** 2
    # (|f| / (eta v))^2 from f / (eta v), whose derivative has (eta v)^-2 but no
    # higher power (f2 / lam2 would have lam2^-2, which float32 flushes to inf);
    # a tiny variance has no ratio: all unexplained (it adds at most its size)
    ramp = low & (v_lat > _KOTTKE_TINY)
    ratio = jnp.sum((f / (_KOTTKE_ETA * jnp.where(ramp, v_lat, 1.0))[..., None]) ** 2, axis=-1)
    beta = jnp.where(low, (1 - jnp.where(ramp, ratio, 0.0)) ** 2, 0.0)
    ok = f2 > 1e-12
    direction = jnp.where(ok[..., None], f**2 / jnp.where(ok, f2, 1.0)[..., None], 0.0)
    weight = length[..., None] * v_lat
    lateral = jnp.sum((weight * (1 - beta))[..., None] * direction, axis=-3)
    unexplained = jnp.sum(weight * beta, axis=-2)

    # the means without the drift of tapered lines between the middles of
    # neighbouring sub-intervals: only the surfaces' jumps are left
    c = bp[..., 1:-1]
    mid = 0.5 * (bp[..., 1:] + bp[..., :-1])
    u = rate[..., 0] / cell
    step = u[..., :-1, :] * (c - mid[..., :-1])[..., None] + u[..., 1:, :] * (mid[..., 1:] - c)[..., None]
    drift = jnp.concatenate([jnp.zeros_like(step[..., :1, :]), jnp.cumsum(step, axis=-2)], axis=-2)
    flat = mean + rate[..., 1] / cell - drift
    centered = flat - jnp.sum(length[..., None] * flat, axis=-2, keepdims=True)
    v_z = jnp.sum(length[..., None] * centered**2, axis=-2)
    if slopes is None:
        vertical = v_z[..., None] * jnp.asarray([0.0, 0.0, 1.0], dtype=v_z.dtype)
    else:
        # each surface's normal (-g, 1), weighted by its squared jump and by
        # 4 t (1 - t) at its height t in the voxel
        t = c - bp[..., :1]
        wj = (flat[..., 1:, :] - flat[..., :-1, :]) ** 2 * (4 * t * (1 - t))[..., None]  # (..., S - 1, C)
        g2 = slopes**2
        dirs = jnp.concatenate([g2, jnp.ones_like(g2[..., :1])], axis=-1) / (1 + jnp.sum(g2, axis=-1))[..., None]
        total = jnp.sum(wj, axis=-2)
        some = total > _KOTTKE_TINY  # (v_z is of the order of total: below, z is as good)
        tilt = jnp.sum(wj[..., None] * dirs[..., None, :], axis=-3) / jnp.where(some, total, 1.0)[..., None]
        tilt = jnp.where(some[..., None], tilt, jnp.asarray([0.0, 0.0, 1.0], dtype=tilt.dtype))
        vertical = v_z[..., None] * tilt
    return jnp.concatenate([lateral + vertical, unexplained[..., None]], axis=-1)


def _kernel_mass(kernel: tuple[float, float, float]) -> float:
    """Integral of the flux kernel (see :func:`_moments`) over the unit lateral square."""
    ax, ay, beta = kernel
    return (1 - 4 * (ax**2 + beta**2 / 12)) * (1 - 4 * (ay**2 + beta**2 / 12))


def _point_slopes(lows: Array, surface_slopes: Array | None) -> Array | None:
    """(nx, ny, 2K + 2, 2) surface slopes per point of :func:`_split_layer`: none at the layer's ends."""
    if surface_slopes is None:
        return None
    ends = jnp.zeros((*lows.shape[:-1], 2, 2), dtype=lows.dtype)
    return jnp.concatenate([ends, surface_slopes.astype(lows.dtype)], axis=-2)


def _split_layer(lows: Array, highs: Array, k: Array) -> tuple[Array, Array, Array, Array]:
    """The z sub-intervals of voxel layer k and the levels present on each.

    Args:
        lows: (..., K) bottom surface per level (cells).
        highs: (..., K) top surface per level (cells).
        k: the voxel layer, which spans [k, k + 1].

    Returns:
        perm: (..., 2K + 2) the sort of the points (the layer's ends, the
            bottoms, the tops), no gradient.
        bp: (..., S + 1) sorted sub-interval ends, S = 2K + 1 (cells).
        length: (..., S) sub-interval lengths.
        present: (..., S, K) whether each level is in each sub-interval.
    """
    num_levels = lows.shape[-1]
    lo = jnp.clip(lows, k, k + 1.0)
    hi = jnp.clip(highs, k, k + 1.0)
    ends = jnp.broadcast_to(jnp.asarray([k, k + 1.0], dtype=lows.dtype), (*lows.shape[:-1], 2))
    points = jnp.concatenate([ends, lo, hi], axis=-1)  # (..., 2K+2)
    # membership of sub-interval m (between sorted points m and m+1) comes from
    # the ranks of each level's surfaces in a stable sort, not from midpoints:
    # with tied surfaces every level keeps the sub-interval on its own side,
    # so moving a surface always changes that level's coverage
    perm = jnp.argsort(jax.lax.stop_gradient(points), axis=-1, stable=True)
    bp = jnp.take_along_axis(points, perm, axis=-1)
    rank = jnp.argsort(perm, axis=-1, stable=True)
    r_lo = rank[..., 2 : 2 + num_levels]
    r_hi = rank[..., 2 + num_levels :]
    length = bp[..., 1:] - bp[..., :-1]  # (..., S)
    m = jnp.arange(length.shape[-1])
    inside = (m >= r_lo[..., None]) & (m < r_hi[..., None])  # (..., K, S)
    # a level that ends strictly before this voxel layer (or starts strictly
    # after it) is clamped to a zero-length interval at one of its ends; it
    # is not there, and must not take the sub-intervals next to a surface
    # that moves onto that end (it would claim that surface's growth)
    reach = (jax.lax.stop_gradient(highs) >= k) & (jax.lax.stop_gradient(lows) <= k + 1.0)
    inside = inside & reach[..., None]
    return perm, bp, length, jnp.moveaxis(inside, -2, -1)


def _level_walls(
    split: tuple[Array, Array, Array, Array],
    lateral: Callable[[Array], Array],
    num_levels: int,
    knots: Array | None = None,
    with_slope: bool = False,
) -> tuple[Array, Array | None, Array]:
    """Each level's sidewall offset over its own part of a voxel layer, and its z slope.

    Level j fills [lo_j, hi_j] of the layer (its surfaces clamped to it). Its
    offset there is the exact mean of ``lateral`` over that part: the offset is
    linear between the level's ``knots`` (its z_to_bias knots), so the mean is
    the length-weighted midpoint values of the pieces between them. Only the
    level's own surfaces and knots enter: another level's surface cutting the
    part changes nothing. A straight edge (summed along its length) then gets
    its exact volume, also with z_to_bias. Parts shorter than ``_PART_TINY``
    take the offset at their middle.

    Args:
        split: the layer's sub-intervals (:func:`_split_layer`).
        lateral: maps heights (...,) to the (..., K) offsets of the levels (see
            :func:`_composite`).
        num_levels: K.
        knots: (K, M) heights (cells, the frame of the surfaces) where a
            level's offset may bend, or None: every offset is linear in z.
        with_slope: also return the mean z slope of each offset.

    Returns:
        (..., K) mean offsets; with ``with_slope``, (..., K) mean slopes
        (offset per cell of height), else None; and (..., K) the middles of
        the parts (cells).
    """
    perm, bp = split[:2]
    points = jnp.take_along_axis(bp, jnp.argsort(perm, axis=-1), axis=-1)  # the layer's ends, lows, highs
    lo = points[..., 2 : 2 + num_levels]
    hi = points[..., 2 + num_levels :]
    if knots is None:
        pieces = 0.5 * (lo + hi)[..., None]  # (..., K, 1)
        weights = jnp.ones_like(pieces)
    else:
        bends = jnp.clip(jnp.asarray(knots, dtype=lo.dtype), lo[..., None], hi[..., None])  # (..., K, M)
        ends = jnp.concatenate([lo[..., None], bends, hi[..., None]], axis=-1)
        ends = jnp.take_along_axis(ends, jnp.argsort(jax.lax.stop_gradient(ends), axis=-1), axis=-1)
        pieces = 0.5 * (ends[..., 1:] + ends[..., :-1])  # (..., K, M + 1) middles of the linear pieces
        length = hi - lo
        long = length > _PART_TINY
        weights = jnp.where(
            long[..., None],
            (ends[..., 1:] - ends[..., :-1]) / jnp.where(long, length, 1.0)[..., None],
            # a tiny part: its middle (the pieces all lie within _PART_TINY of it)
            jnp.full_like(pieces, 1.0 / pieces.shape[-1]),
        )
        pieces = jnp.where(long[..., None], pieces, 0.5 * (lo + hi)[..., None])

    def own(z: Array) -> Array:
        """(..., K, P) offset of each level at its own heights z (..., K, P)."""
        # lateral sees one axis of heights per column, as for the sub-intervals
        every = lateral(z.reshape(*z.shape[:-2], -1)).reshape(*z.shape, num_levels)  # (..., K, P, K)
        return jnp.moveaxis(jnp.diagonal(every, axis1=-3, axis2=-1), -1, -2)

    middle = 0.5 * (lo + hi)
    if not with_slope:
        return jnp.sum(weights * own(pieces), axis=-1), None, middle
    value, dvalue = jax.jvp(own, (pieces,), (jnp.ones_like(pieces),))
    return jnp.sum(weights * value, axis=-1), jnp.sum(weights * dvalue, axis=-1), middle


def _paint_layer(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    split: tuple[Array, Array, Array, Array],
    lateral: Callable[[Array], Array],
    half: float,
    level_eps: Sequence[Sequence[float]] | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
    cell: float = 1.0,
    point_slopes: Array | None = None,
    knots: Array | None = None,
) -> tuple[Array, ...]:
    """Half-plane painting of columns of one voxel layer, per z sub-interval (see :func:`_composite`).

    Args:
        offsets: (B1, B2, K) signed distance of each level (sides of the square).
        normals: (B1, B2, K, 2) unit in-plane normals.
        is_fill: (K,) static, whether a level fills the whole plane.
        split: the layer's sub-intervals (:func:`_split_layer`).
        lateral: the levels' sidewall offsets at heights (see :func:`_composite`).
        half: half side of the lateral square.
        level_eps: (C, K) static contrasts, or None: fractions only.
        kernel: the flux kernel (see :func:`_composite`).
        cell: cells per side of the lateral square.
        point_slopes: (B1, B2, 2K + 2, 2) :func:`_point_slopes`, or None (flat).
        knots: (K, M) heights where the levels' offsets bend (see
            :func:`_level_walls`), or None.

    Returns:
        visible (B1, B2, S, K + 1) fractions per sub-interval (not length
        weighted), length (B1, B2, S), present (B1, B2, S, K), shift (B1, B2, S,
        K) the levels' sidewall offsets; with ``level_eps`` also their rates
        (B1, B2, S, K, 2): the z slope b (cells per cell, 0 for fill levels)
        and ``b * lag``, with lag the height of the sub-interval's middle above
        the middle of the level's part (where its wall is painted); the
        contrast flux (B1, B2, S, C, 3) and rates (B1, B2, S, C, 2; see
        :func:`_normal_tensor`), the sub-interval ends (B1, B2, S + 1) and,
        with ``point_slopes``, the inner ends' surface slopes (B1, B2, S - 1,
        2).
    """
    perm, bp, length, present = split
    num_levels = len(is_fill)
    num_vertices = 4 + sum(not f for f in is_fill)
    square = jnp.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=bp.dtype)
    square = jnp.concatenate([square, jnp.repeat(square[-1:], num_vertices - 4, axis=0)])
    with_n = level_eps is not None
    # each level's sidewall over its own part of the layer (the same in every
    # sub-interval), which follows its (traced) surfaces; with level_eps also
    # its z slope (cells per cell; 0 for fill levels)
    batch = length.shape
    shift, slope, middle = _level_walls(split, lateral, num_levels, knots, with_n)
    shift = jnp.broadcast_to(shift[..., None, :], (*batch, num_levels))  # (B1, B2, S, K)
    if with_n:
        # and how far the sub-interval's middle is from where the wall is
        # painted: the normal's drift is that of walls at the middles
        lag = 0.5 * (bp[..., 1:] + bp[..., :-1])[..., None] - middle[..., None, :]  # (B1, B2, S, K)
        slope = cell * slope[..., None, :]
        slope = jnp.stack([jnp.broadcast_to(slope, (*batch, num_levels)), slope * lag], axis=-1)

    painted = _paint_levels(
        jnp.broadcast_to(square, (*batch, num_vertices, 2)),
        jnp.full(batch, 4, dtype=jnp.int32),
        jnp.ones(batch, dtype=bp.dtype),
        jnp.broadcast_to(normals[:, :, None], (*batch, num_levels, 2)),
        offsets[:, :, None, :] + shift,
        present,
        is_fill,
        half,
        # the square's sides are not interfaces
        *((jnp.zeros((*batch, num_vertices, 4), dtype=bp.dtype), slope, kernel) if with_n else ()),
    )
    parts = (painted[0], length, present, shift)
    if not with_n:
        return parts
    parts += (
        slope,
        _weigh_levels(painted[2], level_eps, axis=-2),
        _weigh_levels(painted[3], level_eps, axis=-2),
        bp,
    )
    if point_slopes is not None:
        parts += (jnp.take_along_axis(point_slopes, perm[..., 1:-1, None], axis=-2),)
    return parts


def _gauss_heights(split: tuple[Array, ...], lateral: Callable[[Array], Array], num_levels: int) -> Array:
    """Each level's sidewall offset at the two Gauss-Legendre heights of its own part of a voxel layer.

    The corner model paints there (:func:`_corner_fill`): near a corner of a
    sloped wall the area is quadratic in z, which the 2-point rule integrates
    exactly over the level's part (where no z_to_bias knot is inside it). As
    for :func:`_level_walls`, only the level's own surfaces enter: another
    level's surface that cuts the part changes nothing.

    Args:
        split: (B1, B2, ...) sub-intervals of :func:`_split_layer`.
        lateral: the levels' sidewall offsets at heights (see :func:`_composite`).
        num_levels: K.

    Returns:
        (B1, B2, S, K, 2) the offsets at the lower, then the upper height (the
        same in every sub-interval).
    """
    perm, bp = split[:2]
    points = jnp.take_along_axis(bp, jnp.argsort(perm, axis=-1), axis=-1)  # the layer's ends, lows, highs
    lo, hi = points[..., 2 : 2 + num_levels], points[..., 2 + num_levels :]
    z = 0.5 * (lo + hi)[..., None] + jnp.asarray([-0.5, 0.5], dtype=bp.dtype) / math.sqrt(3.0) * (hi - lo)[..., None]
    every = lateral(z.reshape(*z.shape[:-2], -1)).reshape(*z.shape, num_levels)  # (B1, B2, K, 2, K)
    own = jnp.moveaxis(jnp.diagonal(every, axis1=-3, axis2=-1), -1, -2)  # (B1, B2, K, 2)
    return jnp.broadcast_to(own[..., None, :, :], (*bp.shape[:-1], bp.shape[-1] - 1, num_levels, 2))


@timed("composite")
def _composite(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    lateral: Callable[[Array], Array],
    nz: int,
    half: float = 0.5,
    corner: "_CornerInputs | None" = None,
    level_eps: Sequence[Sequence[float]] | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
    cell: float = 1.0,
    surface_slopes: Array | None = None,
    knots: Array | None = None,
) -> Array | tuple[Array, Array]:
    """Volume fraction of each level (and the background) per voxel.

    Along z the voxel is split at the heights of all level surfaces (surfaces
    are clamped to the voxel, so a surface on a voxel boundary contributes half
    its derivative to each neighbor). On each sub-interval, the levels present
    there are painted in priority order. Laterally each level is a half-plane
    ``{p: n . p + offset + s_j < 0}`` in voxel coordinates, from its signed
    distance and gradient at the voxel center, moved by the mean s_j of its
    sidewall ``lateral`` over its own part of the voxel along z
    (:func:`_level_walls`; exact between the ``knots``, and independent of the
    other levels' surfaces). So a surface cutting the voxel gets the boundary
    at its own height, and a straight edge gets its exact volume summed along
    the edge, also with z_to_bias (per voxel, the area at s_j differs from its
    mean over z where the edge crosses a corner of the square within the
    voxel, by up to about 5e-3 of the voxel at 20 deg, but these errors cancel
    along the edge; corner columns use two heights, see below): level j gets
    the area of the voxel square
    inside it but outside every earlier present level, computed exactly by
    clipping the square with the complements in priority order. Levels that
    meet at a seam (complementary half-planes) therefore tile the voxel, nested
    levels nest, and slanted straight edges get exact areas. Fill levels cover
    the whole square. With ``corner``, the columns where a polygon corner or a
    crossing of layer edges is within reach are repainted with the exact local
    geometry (see :func:`_corner_fill`), in the voxel layers where a polygon
    level is present in one of them, with each level at the two Gauss heights
    of its own part of the voxel (:func:`_gauss_heights`), which integrate a
    sloped corner's quadratic area exactly, and with the half-planes' result
    where the edge is straight in the square (so a straight edge has no seam
    between corner and ordinary columns, see :func:`_corner_fill`).

    With ``level_eps``, the clips also carry each level's plane: its in-plane
    normal and its z slope ``b = d lateral / dz`` (so a sidewall angle tilts it
    out of the plane), and every voxel gets the squared-normal tensor of its
    own model (:func:`_normal_tensor`): its eps variance split into each
    sub-interval's lateral part, along the interface flux weighted by the
    kernel ``(1 - 4 xi^2) (1 - 4 eta^2)`` over the lateral square (which
    vanishes on the square's sides, so a line entering it changes nothing
    abruptly), and the vertical part along z (tilted by ``surface_slopes``).
    For several sub-squares per voxel, :func:`_composite_samples` adds up their
    moments before the tensor.

    Args:
        offsets: (nx, ny, K) signed distance of each level at the voxel
            centers (cells, negative inside); +-BIG for absent / fill.
        normals: (nx, ny, K, 2) unit in-plane gradient of each level's distance.
        is_fill: (K,) static, whether a level fills the whole plane.
        lows: (nx, ny, K) bottom surface per level (cells).
        highs: (nx, ny, K) top surface per level (cells).
        lateral: maps heights (..., ) (cells, voxel k spans [k, k + 1]) to the
            (..., K) offset of each level's boundary there, added to
            ``offsets`` (ignored for fill levels).
        nz: number of voxel layers.
        half: half side of the lateral square (cells); < 0.5 for sub-samples.
        corner: the corner columns of this sample grid, or None (one
            half-plane per level everywhere).
        level_eps: (C, K) static contrast of each level to the background per
            E component, normalized by the eps range, or None: fractions only.
        kernel: (ax, ay, beta): the flux kernel's coordinates ``ax + beta u``
            at the lateral square's points u (its side is 1).
        cell: cells per side of the lateral square.
        surface_slopes: (nx, ny, 2K, 2) lateral gradient (cells per cell) of
            each level's bottom then top surface, or None (flat).
        knots: (K, M) heights (cells) where the levels' offsets bend (their
            z_to_bias knots), or None: every offset is linear in z.

    Returns:
        (nx, ny, nz, K + 1): K levels then the background, summing to one;
        with ``level_eps`` also the (nx, ny, nz, C, 4) tensor of
        :func:`_normal_tensor`.
    """
    num_levels = lows.shape[-1]
    with_n = level_eps is not None
    point_slopes = _point_slopes(lows, surface_slopes) if with_n else None
    if with_n:
        kmass = _kernel_mass(kernel)

    @jax.checkpoint
    def one_slice(k: Array) -> Array:
        """Fractions (nx, ny, K + 1) of the voxel layer k."""
        split = _split_layer(lows, highs, k)
        parts = _paint_layer(
            offsets, normals, is_fill, split, lateral, half, level_eps, kernel, cell, point_slopes, knots
        )
        visible, length = parts[:2]  # (nx, ny, S, K + 1), (nx, ny, S)
        out = jnp.stack([jnp.sum(length * visible[..., j], axis=-1) for j in range(num_levels + 1)], axis=-1)
        if with_n:
            flux, rate, bp = parts[5:8]
            tensor = _normal_tensor(
                length,
                bp,
                flux,
                _weigh_levels(visible[..., :-1], level_eps),
                _weigh_levels(visible[..., :-1], level_eps, power=2),
                rate,
                cell,
                kmass,
                parts[8] if len(parts) > 8 else None,
            )
            out = (out, tensor)  # (nx, ny, K + 1), (nx, ny, C, 4)
        if corner is None:
            return out  # (nx, ny, K + 1)
        # the corner columns' sub-intervals, for the corner pass, with the
        # offsets at their Gauss heights
        cols = jnp.minimum(corner.cols, length.shape[0] * length.shape[1] - 1)
        gathered = [x.reshape(-1, *x.shape[2:])[cols][:, None] for x in split]
        parts = (*parts[:3], _gauss_heights(gathered, lateral, num_levels)[:, 0], *parts[4:])
        return out, tuple(x if i == 3 else x.reshape(-1, *x.shape[2:])[cols] for i, x in enumerate(parts))

    ks = jnp.arange(nz, dtype=lows.dtype)
    if corner is None:
        out = jax.lax.map(one_slice, ks)  # (nz, nx, ny, K+1)
        if with_n:
            return jnp.moveaxis(out[0], 0, 2), jnp.moveaxis(out[1], 0, 2)
        return jnp.moveaxis(out, 0, 2)
    out, columns = jax.lax.map(one_slice, ks)
    if with_n:
        out, tensor = out
    cols = jnp.minimum(corner.cols, offsets.shape[0] * offsets.shape[1] - 1)
    fill = _corner_pass(
        corner.layers,
        corner.taken,
        *columns[:4],
        columns[4:] if with_n else None,
        # the columns' half-planes, for the corner model's straight-edge part
        (offsets.reshape(-1, num_levels)[cols], normals.reshape(-1, num_levels, 2)[cols]),
        trees=corner.trees,
        literals=corner.literals,
        is_fill=tuple(is_fill),
        lines=corner.lines,
        sc=corner.sc,
        half=half,
        level_eps=None if level_eps is None else tuple(tuple(float(e) for e in row) for row in level_eps),
        kernel=tuple(float(x) for x in kernel),
        cell=float(cell),
    )  # (nz, N, K + 1)
    if with_n:
        fill, corner_tensor = fill
    out = out.reshape(nz, -1, num_levels + 1).at[:, corner.cols].set(fill, mode="drop").reshape(out.shape)
    if not with_n:
        return jnp.moveaxis(out, 0, 2)
    shape = tensor.shape
    tensor = tensor.reshape(nz, -1, *shape[3:]).at[:, corner.cols].set(corner_tensor, mode="drop").reshape(shape)
    return jnp.moveaxis(out, 0, 2), jnp.moveaxis(tensor, 0, 2)


def _lateral_samples(n_sub: int, edge_width: float) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
    """The lateral sub-samples of a voxel and their flux kernels.

    Sample (a, b) is the square ``edge_width / n_sub`` cells wide around the
    point (a, b) cells from the voxel center. Where these squares tile the
    voxel (``edge_width`` 1, or one sample), their kernels are the parts of one
    kernel over the voxel, which vanishes on its sides only. Otherwise (gaps or
    overlaps between the squares) each square gets its own kernel, which
    vanishes on its own sides: a line crossing a side that no other square
    continues would bring or take its flux at once.

    Args:
        n_sub: sub-samples per axis.
        edge_width: lateral width (cells) of the voxel square.

    Returns:
        (a, b, kernel) per sample, a-major; kernel is the (ax, ay, beta) of
        :func:`_moments` on the sample's square (side 1).
    """
    sub = [(k + 0.5) / n_sub - 0.5 for k in range(n_sub)]
    cell = edge_width / n_sub
    if n_sub > 1 and not math.isclose(edge_width, 1.0):
        return tuple((a, b, (0.0, 0.0, 1.0)) for a in sub for b in sub)
    span = (1 - 1 / n_sub) + cell  # side (cells) of the union of the squares
    return tuple((a, b, (a / span, b / span, cell / span)) for a in sub for b in sub)


@timed("composite_samples")
def _composite_samples(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    lateral: Callable[[Array], Array],
    nz: int,
    level_eps: Sequence[Sequence[float]],
    kernels: Sequence[tuple[float, float, float]],
    cell: float,
    corners: Sequence["_CornerInputs"] | None = None,
    surface_slopes: Array | None = None,
    half: float = 0.5,
    knots: Array | None = None,
) -> tuple[Array, Array | None]:
    """Fractions and normal tensor of voxels painted on several lateral sub-squares.

    Every sample (a square in the voxel, see :func:`_lateral_samples`) is
    painted as in :func:`_composite`, with its own distances, corner columns
    and kernel, on the same z sub-intervals (the surfaces are shared). The
    samples' moments per sub-interval are added up before
    :func:`_normal_tensor`: the voxel's lateral variance is the one over all its
    squares, also between them, and its flux is the sum of theirs. With one
    kernel over the voxel (squares that tile it), a line crossing the face
    between two squares takes its flux from one to the other, so the tensor is
    continuous there, and a plane on that face keeps its exact normal. (A
    tensor per square would miss the variance between the squares, and so give
    that plane none, and a vanishing sliver would bring its whole edge's flux
    into its square and turn that square's whole variance.) This continuity
    needs the squares' models to agree on the line: straight edges, and corner
    columns where the corner model is certified. Near a polygon corner painted
    with half-planes (``corner_lines`` 0, or beyond the corner model's
    certified reach), a square's half-plane can be another edge than the one
    that crosses into it from its neighbour, whose flux then comes at once:
    the normal jumps there, like the half-plane model's normal (and fill) where
    a voxel center crosses a corner's bisector.

    The corner columns of every sample are painted first (their half-planes,
    then :func:`_corner_pass` per sub-interval), one sample at a time with the
    kernel's center traced, so that all samples share one compiled pass; the
    voxel layers take them over before adding up. Every sample's corner model
    E is blended with its half-planes T by one weight per voxel and
    sub-interval: the least of the samples' certificates (``raw`` of
    :func:`_corner_fill`; 1 where a column is not a corner column of a sample,
    since its half-plane is exact there, and 0 where a sample's grid had no
    slot for it). Where it is > 0, every sample's E is exact, so the voxel's
    average ``W mean(E) + (1 - W) mean(T)`` is never farther from the exact
    fill than the half-planes' (with a weight per sample, the samples' T errors
    could cancel where some of them are made exact). The certificate ramps to
    1 within ``_VOXEL_RAMP`` of where the corner columns end (their radius is
    that much larger with sub-samples), so the weight stays continuous.

    Args:
        offsets: (P, nx, ny, K) each sample's signed distances (sides of its
            square; see :func:`_composite`).
        normals: (P, nx, ny, K, 2) each sample's unit normals.
        is_fill: (K,) static, whether a level fills the whole plane.
        lows: (nx, ny, K) bottom surface per level (cells).
        highs: (nx, ny, K) top surface per level (cells).
        lateral: see :func:`_composite` (sides of a sample's square).
        nz: number of voxel layers.
        level_eps: (C, K) static contrasts (see :func:`_composite`), or None:
            fractions only.
        kernels: (ax, ay, beta) of every sample (:func:`_lateral_samples`),
            all with the same beta.
        cell: cells per side of a sample's square.
        corners: every sample's corner columns (all with the same number of
            columns and the same static structure), or None: none.
        surface_slopes: see :func:`_composite`.
        half: half side of a sample's square.
        knots: see :func:`_composite`.

    Returns:
        (nx, ny, nz, K + 1) the samples' fractions, summed (not averaged), and
        the voxels' (nx, ny, nz, C, 4) tensor of :func:`_normal_tensor` (None
        without ``level_eps``).

    Raises:
        ValueError: kernels of different scales.
    """
    num_samples = offsets.shape[0]
    nx, ny, num_levels = lows.shape
    num = nx * ny
    with_n = level_eps is not None
    scales = {kernel[2] for kernel in kernels}
    if len(scales) != 1:
        raise ValueError(f"The samples' kernels have different scales {sorted(scales)}")
    # lengths and areas on a sample's square are beta and beta^2 of those on the
    # kernel's square: flux / kmass is then per side of the kernel's square
    scale = scales.pop()
    kmass = scale * sum(_kernel_mass(kernel) for kernel in kernels)
    point_slopes = _point_slopes(lows, surface_slopes) if with_n else None
    if with_n:
        level_eps = tuple(tuple(float(e) for e in row) for row in level_eps)
    ks = jnp.arange(nz, dtype=lows.dtype)
    dtype = lows.dtype

    def take(x: Array | None, cols: Array) -> Array | None:
        """(nx, ny, ...) -> (N, 1, ...) at the columns cols."""
        return None if x is None else x.reshape(num, *x.shape[2:])[cols][:, None]

    centers = jnp.asarray([kernel[:2] for kernel in kernels], dtype=dtype)  # (P, 2): ax, ay

    def sample_columns(inputs: tuple[Array, ...]) -> tuple[Array, ...]:
        """The corner columns of one sample per sub-interval: their half-planes, then the corner model."""
        o, n, center, cols, taken, layers = inputs
        gathered = tuple(take(x, cols) for x in (o, n, lows, highs, point_slopes))
        kernel = (center[0], center[1], scale)

        @jax.checkpoint
        def columns(k: Array) -> tuple[Array, ...]:
            """(N, ...) half-plane painting of the corner columns in voxel layer k."""
            go, gn, lo, hi, slopes = gathered
            split = _split_layer(lo, hi, k)
            parts = _paint_layer(go, gn, is_fill, split, lateral, half, level_eps, kernel, cell, slopes, knots)
            parts = (*parts[:3], _gauss_heights(split, lateral, num_levels), *parts[4:7])
            return tuple(x[:, 0] for x in parts)

        dense = jax.lax.map(columns, ks)
        first = corners[0]  # the static structure is the same in every sample
        return _corner_pass(
            layers,
            taken,
            *dense[:4],
            dense[4:] if with_n else None,
            (gathered[0][:, 0], gathered[1][:, 0]),  # the columns' half-planes
            center=center,
            trees=first.trees,
            literals=first.literals,
            is_fill=tuple(is_fill),
            lines=first.lines,
            sc=first.sc,
            half=half,
            level_eps=level_eps,
            kernel=(0.0, 0.0, float(scale)),
            cell=float(cell),
            per_interval=True,
        )  # (nz, N, S, K + 1), (nz, N, S) and (nz, N, S, C, 3), (nz, N, S, C, 2)

    repainted, corner_cols, dropped = (), None, None
    if corners:  # one sample at a time, so that their corner passes compile once
        corner_cols = jnp.stack([corner.cols for corner in corners])  # (P, N)
        repainted = jax.lax.map(
            sample_columns,
            (
                offsets,
                normals,
                centers,
                jnp.minimum(corner_cols, num - 1),
                jnp.stack([corner.taken for corner in corners]),
                jax.tree.map(lambda *x: jnp.stack(x), *(corner.layers for corner in corners)),
            ),
        )
        # per voxel layer, stacked over the samples: (nz, P, N, ...)
        repainted = tuple(jnp.moveaxis(x, 0, 1) for x in repainted)
        # columns that some sample needed but had no room for: half-planes only
        if all(corner.dropped is not None for corner in corners):
            dropped = jnp.any(jnp.stack([corner.dropped for corner in corners]), axis=0)  # (nx * ny,)
    num_s = 2 * num_levels + 1

    def put(x: Array, cols: Array, value: Array) -> Array:
        """x (nx, ny, ...) with the columns cols set to value (N, ...); unused slots (nx * ny) are dropped."""
        return x.reshape(num, *x.shape[2:]).at[cols].set(value, mode="drop").reshape(x.shape)

    @jax.checkpoint
    def one_slice(inputs: tuple[Array, tuple]) -> tuple[Array, Array | None]:
        """Summed fractions (nx, ny, K + 1) and tensor (nx, ny, C, 4) of the voxel layer k."""
        k, corner_layers = inputs
        split = _split_layer(lows, highs, k)
        perm, bp, length, _ = split
        if corners:
            # the voxel's certificate per sub-interval: the least of its samples'
            # (1 where a column is not a corner column of a sample: its
            # half-plane is exact there), 0 where a sample had no room for it
            certificate = jax.vmap(lambda c, v: jnp.ones((num, num_s), dtype=dtype).at[c].set(v, mode="drop"))(
                corner_cols, corner_layers[1]
            )
            voxel = jnp.min(certificate, axis=0)  # (nx * ny, S)
            if dropped is not None:
                voxel = jnp.where(dropped[:, None], 0.0, voxel)

        @jax.checkpoint
        def add(total: tuple[Array, ...], sample: tuple[Array, ...]) -> tuple[tuple[Array, ...], None]:
            """Adds one sample's fractions and moments (per sub-interval) to the voxels'."""
            o, n, center, *fix = sample
            kernel = (center[0], center[1], scale)
            parts = _paint_layer(o, n, is_fill, split, lateral, half, level_eps, kernel, cell, knots=knots)
            moments = (parts[0], parts[5], parts[6]) if with_n else parts[:1]  # fractions, flux, rate
            if fix:  # the corner model in this sample's corner columns, blended by the voxel's certificate
                cols, exact, _, *exact_moments = fix
                cs = jnp.minimum(cols, num - 1)
                w = voxel[cs]  # (N, S)
                blended = []
                for x, e in zip(moments, (exact, *exact_moments), strict=True):
                    t = x.reshape(num, *x.shape[2:])[cs]
                    ww = w.reshape(*w.shape, *([1] * (t.ndim - 2)))
                    blended.append(put(x, cols, ww * e + (1 - ww) * t))
                moments = tuple(blended)
            fill = jnp.stack([jnp.sum(length * moments[0][..., j], axis=-1) for j in range(num_levels + 1)], axis=-1)
            return tuple(x + y for x, y in zip(total, (fill, *moments), strict=True)), None

        # one sample at a time (as many small steps as separate samples would take)
        zeros = (
            jnp.zeros((nx, ny, num_levels + 1), dtype=dtype),
            jnp.zeros((nx, ny, num_s, num_levels + 1), dtype=dtype),
        )
        if with_n:
            num_c = len(level_eps)
            zeros += (
                jnp.zeros((nx, ny, num_s, num_c, 3), dtype=dtype),
                jnp.zeros((nx, ny, num_s, num_c, 2), dtype=dtype),
            )
        samples = (offsets, normals, centers) + ((corner_cols, *corner_layers) if corners else ())
        total, _ = jax.lax.scan(add, zeros, samples)
        if not with_n:
            return total[0], None
        fractions, visible, flux, rate = total
        slopes = None
        if point_slopes is not None:
            slopes = jnp.take_along_axis(point_slopes, perm[..., 1:-1, None], axis=-2)
        tensor = _normal_tensor(
            length,
            bp,
            flux,
            _weigh_levels(visible[..., :-1], level_eps) / num_samples,
            _weigh_levels(visible[..., :-1], level_eps, power=2) / num_samples,
            rate / num_samples,
            cell,
            kmass,
            slopes,
        )
        return fractions, tensor

    fractions, tensor = jax.lax.map(one_slice, (ks, repainted))
    fractions = jnp.moveaxis(fractions, 0, 2)
    return fractions, (None if tensor is None else jnp.moveaxis(tensor, 0, 2))


# ---------------------------------------------------------------- corner model


class _LayerGeometry(NamedTuple):
    """A source layer's polygons on the grid, after T-junction splitting."""

    #: (E, 2) vertices in cells (traced)
    vertices: Array
    #: (E,) end vertex of every edge
    nxt: np.ndarray
    #: (E,) polygon index of every edge
    ids: np.ndarray
    #: (E,) the edges that bound the union of the polygons
    boundary: np.ndarray
    #: (E,) start vertex of every edge with the filled side on its left (None
    #: without corner_lines)
    src: np.ndarray | None
    #: (E,) end vertex of every edge with the filled side on its left
    dst: np.ndarray | None
    #: (E,) the next boundary edge, -1 if none or ambiguous
    succ: np.ndarray | None


class _CornerPlan(NamedTuple):
    """Static literals of the levels for the corner model (see :func:`_literals`)."""

    #: per level (priority order): its literal tree, None for fill levels
    trees: tuple
    #: per level: (layer slot, a, b in cells) of each literal
    literals: tuple
    #: per layer slot: its position in ``StackDevice._slots()`` (source layers, then derived shapes)
    layers: tuple[int, ...]


class _LayerCorner(NamedTuple):
    """Two-edge local model of a source layer at the corner columns (see :func:`_layer_corner`).

    Lengths are in sides of the lateral square, relative to each column's
    sample point; the arrays are (N, ...), one row per column.
    """

    #: (N, 2, 3) half-planes ``n . p + lam < 0`` (outward n) of the two nearest
    #: edges; at a vertex the edge into it first
    planes: Array
    #: (N,) the layer is the intersection of the two half-planes, else their union
    op_and: Array
    #: (N,) the two edges meet at a vertex
    linked: Array
    #: (N,) that vertex is convex
    convex: Array
    #: (N, 2) that vertex
    vertex: Array
    #: (N,) distance to the second nearest edge (no gradient)
    d2s: Array
    #: (N,) radius within which the two edges are the whole boundary; -BIG
    #: where the model is invalid
    rpair: Array
    #: (N,) the layer's signed distance at the point
    sd: Array
    #: (N, 2) its unit gradient there
    normal: Array


class _CornerInputs(NamedTuple):
    """The corner columns of one sample grid (see :func:`_corner_fill`)."""

    #: (N,) flat column index (i * ny + j); nx * ny in unused slots
    cols: Array
    #: (N,) whether a slot holds a column
    taken: Array
    #: local model of every source layer the literals read, stacked (L, N, ...)
    layers: _LayerCorner
    #: per level (priority order): its literal tree, None for fill levels
    trees: tuple
    #: per level: (layer in ``layers``, a, b in cells) of each literal
    literals: tuple
    #: number of split lines per column
    lines: int
    #: cells -> sides of the lateral square
    sc: float
    #: (nx * ny,) the candidate columns that found no slot (they keep the
    #: half-planes), or None
    dropped: Array | None = None


class _Slot(NamedTuple):
    """One literal of one level in the corner columns; arrays (N, S) per column and sub-interval."""

    #: the level (priority order)
    level: int
    #: the literal's source layer (index into the stacked models)
    layer: int
    #: offset of the literal's boundary from its layer's (sides of the square)
    off: Array
    #: the literal's signed distance at the sample point
    val: Array
    #: the square is on one side of the literal's boundary: the value is ``val < 0``
    const: Array
    #: one edge in reach: the half-plane at the sample point, exact there
    single: Array
    #: a pair is the intersection of its planes, else their union
    op_and: Array
    #: (N, S, 3) planes [n, offset]: the single plane or the pair's first, then the pair's second
    planes: tuple[Array, Array]
    #: the planes are used and cut the square
    live: tuple[Array, Array]
    #: a pair at a vertex that the offset rounds (convex grown, reflex shrunk)
    rounded: Array


def _layer_corner(
    v: Array,
    src: np.ndarray,
    dst: np.ndarray,
    succ: np.ndarray,
    pid: np.ndarray,
    ids: Array,
    dist: Array,
    c: Array,
    sd: Array,
    normal: Array,
    sc: float,
) -> _LayerCorner:
    """Local model of a source layer from its two nearest boundary edges.

    Two edges that meet at a vertex bound the intersection (convex vertex) or
    the union (reflex vertex) of their half-planes; two edges that do not meet
    bound a strip (intersection) or a gap (union) of one polygon, or belong to
    two polygons of the union (union). The model is the layer up to ``rpair``
    from the point: the third nearest edge and the far ends of the two edges.
    It is invalid for degenerate edges and self-intersections, and where it
    disagrees with the layer's sign at the point. The classification has no
    gradient; the half-planes, the vertex and the radius follow the vertices.

    Args:
        v: (E, 2) the layer's vertices (cells, after T-junction splitting).
        src: (E,) static start vertex of every edge (filled side on its left).
        dst: (E,) static end vertex of every edge.
        succ: (E,) static next boundary edge (-1 if none).
        pid: (E,) static polygon index of every edge.
        ids: (N, 3) the three nearest boundary edges (:func:`nearest_edges`).
        dist: (N, 3) their distances (cells).
        c: (N, 2) the sample points (cells).
        sd: (N,) the layer's signed distance at the points (cells).
        normal: (N, 2) its unit gradient.
        sc: cells -> sides of the lateral square.

    Returns:
        The model, in sides of the lateral square relative to the points.
    """
    sg = jax.lax.stop_gradient
    src, dst, succ, pid = (jnp.asarray(x) for x in (src, dst, succ, pid))
    ids, dist = sg(ids), sg(dist)
    e1, e2, e3 = ids[:, 0], ids[:, 1], ids[:, 2]
    link12, link21 = succ[e1] == e2, succ[e2] == e1
    linked = link12 ^ link21  # both: a polygon of two edges
    ea = jnp.where(link12, e1, e2)  # at a vertex: the edge into it
    eb = jnp.where(link12, e2, e1)  # and the edge out of it

    def line(e: Array) -> tuple[Array, ...]:
        """Ends, unit direction, outward normal and scaled signed distance of the edges e."""
        p, q = v[src[e]], v[dst[e]]
        d = q - p
        l2 = jnp.sum(d * d, axis=-1)
        ok = sg(l2) > 1e-12
        u = d * jax.lax.rsqrt(jnp.where(ok, l2, 1.0))[:, None]
        n = jnp.stack([u[:, 1], -u[:, 0]], axis=-1)
        return p, q, u, n, sc * jnp.sum(n * (c - p), axis=-1), ok

    pa, qa, ua, na, lam_a, ok_a = line(ea)
    pb, qb, ub, nb, lam_b, ok_b = line(eb)
    convex = sg(_cross(ua, ub)) >= 0  # 180 deg (split T-junctions): coincident planes

    # two edges that do not meet: one polygon (a strip where the part of e1
    # nearest to the point is inside e2's half-plane, else a gap), or two
    s1, t1, s2, t2 = (sg(v[i[e]]) for i, e in ((src, e1), (dst, e1), (src, e2), (dst, e2)))
    d1, d2 = t1 - s1, t2 - s2
    f = jnp.sum((sg(c) - s1) * d1, axis=-1) / jnp.maximum(jnp.sum(d1 * d1, axis=-1), 1e-30)
    foot = s1 + jnp.clip(f, 0.0, 1.0)[:, None] * d1
    strip = jnp.sum(sg(jnp.where(link12[:, None], nb, na)) * (foot - s2), axis=-1) < 0
    same = pid[e1] == pid[e2]
    op_and = jnp.where(linked, convex, same & strip)
    crossing = (_cross(d1, s2 - s1) * _cross(d1, t2 - s1) < 0) & (_cross(d2, s1 - s2) * _cross(d2, t1 - s2) < 0)
    valid = ok_a & ok_b & (dist[:, 1] < jnp.inf) & ~(link12 & link21) & ~(~linked & same & crossing)
    # the model must give the layer's sign at the point
    la, lb, sdv = sg(lam_a), sg(lam_b), sg(sd)
    inside = jnp.where(op_and, (la < 0) & (lb < 0), (la < 0) | (lb < 0))
    valid = valid & ((inside == (sdv < 0)) | (jnp.abs(sdv) < 1e-4))

    def norm(x: Array) -> Array:
        """|x| of (N, 2) vectors, with a finite derivative at 0."""
        r2 = jnp.sum(x * x, axis=-1)
        ok = sg(r2) > 1e-24
        return jnp.where(ok, jnp.sqrt(jnp.where(ok, r2, 1.0)), 0.0)

    # the far ends of the two edges (a vertex they share is part of the model)
    far = jnp.minimum(norm(c - pa), norm(c - qb))
    far = jnp.where(linked, far, jnp.minimum(far, jnp.minimum(norm(c - qa), norm(c - pb))))
    p3, q3 = v[src[e3]], v[dst[e3]]
    d3 = q3 - p3
    l3 = jnp.sum(d3 * d3, axis=-1)
    long3 = sg(l3) > 1e-12  # a point-like edge: its start
    f3 = jnp.where(long3, jnp.sum((c - p3) * d3, axis=-1) / jnp.where(long3, l3, 1.0), 0.0)
    third = norm(c - p3 - jnp.clip(f3, 0.0, 1.0)[:, None] * d3)
    third = jnp.where(dist[:, 2] < jnp.inf, third, _BIG)
    return _LayerCorner(
        planes=jnp.stack([jnp.concatenate([na, lam_a[:, None]], -1), jnp.concatenate([nb, lam_b[:, None]], -1)], 1),
        op_and=op_and,
        linked=linked,
        convex=convex,
        vertex=sc * (qa - c),
        d2s=sc * jnp.minimum(dist[:, 1], _BIG),
        rpair=jnp.where(valid, sc * jnp.minimum(third, far) - 2 * _CORNER_TOL, -_BIG),
        sd=sc * sd,
        normal=normal,
    )


def _same_plane(p: Array, q: Array) -> Array:
    """+1 where half-planes (..., 3) are the same, -1 where complementary, else 0 (no gradient)."""
    p, q = jax.lax.stop_gradient(p), jax.lax.stop_gradient(q)
    same = jnp.max(jnp.abs(p - q), axis=-1) < _PLANE_TOL
    flipped = jnp.max(jnp.abs(p + q), axis=-1) < _PLANE_TOL
    return jnp.where(same, 1, jnp.where(flipped, -1, 0))


def _literal_value(slot: _Slot, first: Array, second: Array) -> Array:
    """A literal from whether points are inside its planes.

    Args:
        slot: the literal, (N, S) per column and sub-interval.
        first: (N, S, ...) whether the points are inside its first plane.
        second: (N, S, ...) the same for its second plane.

    Returns:
        (N, S, ...) whether the points are in the literal's region.
    """
    extra = (1,) * (first.ndim - slot.const.ndim)
    op_and, const, single = (x.reshape(x.shape + extra) for x in (slot.op_and, slot.const, slot.single))
    negative = (jax.lax.stop_gradient(slot.val) < 0).reshape(slot.val.shape + extra)
    return jnp.where(const, negative, jnp.where(single, first, jnp.where(op_and, first & second, first | second)))


def _assign(
    slots: Sequence[_Slot],
    trees: tuple,
    present: Array,
    is_fill: Sequence[bool],
    point: Array,
    flips: Array | None,
) -> Array:
    """First present level (priority order) whose literals' planes contain points (no gradient).

    Args:
        slots: the literals.
        trees: per level, its literal tree (None for fill levels).
        present: (N, S, K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        point: (N, S, T, 2) T points per column and sub-interval.
        flips: (N, S, T, num slots) where to negate each literal (or None).

    Returns:
        (N, S, T) level index, K (the background) where no level contains the point.
    """
    sg = jax.lax.stop_gradient
    shape = point.shape[:-1]
    level = jnp.full(shape, len(is_fill), dtype=jnp.int32)
    done = jnp.zeros(shape, dtype=bool)
    for j, fill in enumerate(is_fill):
        hit = jnp.broadcast_to(present[..., j, None], shape)
        if not fill:
            values = []
            for i, slot in enumerate(slots):
                if slot.level != j:
                    continue
                first, second = (
                    jnp.sum(sg(p[..., None, :2]) * point, axis=-1) + sg(p[..., None, 2]) < 0 for p in slot.planes
                )
                value = _literal_value(slot, first, second)
                values.append(value ^ flips[..., i] if flips is not None else value)
            hit = hit & jnp.broadcast_to(_eval_bool(trees[j], values), shape)
        level = jnp.where(~done & hit, j, level)
        done = done | hit
    return level


def _round_tips(
    slots: Sequence[_Slot],
    corner: _CornerInputs,
    exact: Array,
    present: Array,
    is_fill: Sequence[bool],
    half: float,
) -> tuple[Array, Array]:
    """Corrects the cells' mitred corners where the level set is round.

    A convex corner grown by g (or a reflex corner shrunk by g) is a circular
    arc of radius g around the vertex v, while the two offset half-planes meet
    at a mitre. They differ exactly in the tip ``Z = kite - disk``, with the kite
    ``v, v + g n_a, v + g (n_a + n_b) / (1 + n_a . n_b), v + g n_b`` (outward
    edge normals n_a, n_b; -g for a reflex corner): there the literal is the
    opposite of the mitred one. Where every other plane is on one side of the
    tip in the square, the mitred painting gave all of it to one level, and
    the literal negated gives it to another: the tip's area moves between the
    two. Literals of the same layer with the same offset share their tip.

    All literals are processed together, along a slot axis T.

    Args:
        slots: the literals.
        corner: the corner columns.
        exact: (N, S, K + 1) the cells' (mitred) fractions per sub-interval.
        present: (N, S, K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the lateral square.

    Returns:
        exact: the fractions with the tips moved.
        gate: (N, S) False where the correction does not apply: tips of two
            vertices, a tip cut by another plane, or a mitre sharper than
            ``_MITRE_MIN``.
    """
    sg = jax.lax.stop_gradient
    n, s = exact.shape[:2]
    num = len(slots)
    dtype = exact.dtype
    full_area = (2 * half) ** 2
    layer = np.array([slot.layer for slot in slots])

    def per_slot(x: Array) -> Array:
        """(L, N, ...) per layer -> (N, S, T, ...) per literal."""
        x = jnp.moveaxis(x[layer], 0, 1)[:, None]
        return jnp.broadcast_to(x, (n, s, num, *x.shape[3:]))

    m = jax.tree.map(per_slot, corner.layers)
    off = jnp.stack([slot.off for slot in slots], axis=-1)  # (N, S, T)
    rounded = jnp.stack([slot.rounded for slot in slots], axis=-1)
    na, nb = m.planes[..., 0, :2], m.planes[..., 1, :2]
    den = 1.0 + jnp.sum(na * nb, axis=-1)
    sharp = sg(den) <= _MITRE_MIN
    mitre = (na + nb) / jnp.where(sharp, 1.0, den)[..., None]
    g = jnp.abs(off)  # rounded literals have off != 0
    step = (jnp.where(m.convex, 1.0, -1.0) * g)[..., None]
    kite = jnp.stack([m.vertex, m.vertex + step * na, m.vertex + step * mitre, m.vertex + step * nb], axis=-2)
    count = jnp.full((n, s, num), 4, dtype=jnp.int32)
    ccw = sg(_area(kite, count)) >= 0
    kite = jnp.where(ccw[..., None, None], kite, kite[..., ::-1, :])
    poly = jnp.concatenate([kite, jnp.repeat(kite[..., -1:, :], 4, axis=-2)], axis=-2)
    for side in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # the kite in the square
        side = jnp.broadcast_to(jnp.asarray(side, dtype=dtype), (n, s, num, 2))
        poly, count = _clip(poly, count, side, jnp.full((n, s, num), half, dtype=dtype))
    zone = (_area(poly, count) - _disk_area(poly, count, m.vertex, g)) / full_area
    zone = jnp.where(rounded & ~sharp, jnp.maximum(zone, 0.0), 0.0)
    gate = ~jnp.any(rounded & sharp, axis=-1)

    # literals of one layer with one offset share their tip (same layer, so same
    # vertex); the first of them moves it
    tip = sg(zone) > _TIP_AREA
    member = (layer[:, None] == layer[None, :]) & (jnp.abs(sg(off)[..., :, None] - sg(off)[..., None, :]) < _PLANE_TOL)
    earlier = np.tril(np.ones((num, num), dtype=bool), -1)  # [t, u]: u before t
    first = tip & ~jnp.any(tip[..., None, :] & member & earlier, axis=-1)
    gate = gate & (jnp.sum(first, axis=-1) <= 1)

    # every other live plane must leave the tip whole
    valid = jnp.arange(poly.shape[-2]) < count[..., None]  # (N, S, T, 8)
    planes = sg(jnp.stack([p for slot in slots for p in slot.planes], axis=-2))  # (N, S, 2T, 3)
    live = jnp.stack([x for slot in slots for x in slot.live], axis=-1)  # (N, S, 2T)
    f = jnp.sum(planes[..., None, :, None, :2] * sg(poly)[..., None, :, :], axis=-1) + planes[..., None, :, None, 2]
    inside = valid[..., None, :]
    both = jnp.any(inside & (f > 1e-5), axis=-1) & jnp.any(inside & (f < -1e-5), axis=-1)  # (N, S, T, 2T)
    cut = jnp.any(live[..., None, :] & both & ~member[..., np.repeat(np.arange(num), 2)], axis=-1)
    gate = gate & ~jnp.any(first & cut, axis=-1)

    probe = jnp.sum(jnp.where(valid[..., None], sg(poly), 0.0), axis=-2) / jnp.maximum(count, 1)[..., None]
    mitred = _assign(slots, corner.trees, present, is_fill, probe, None)
    exact_level = _assign(slots, corner.trees, present, is_fill, probe, member)
    move = jnp.where(first, zone, 0.0)[..., None]
    num_out = len(is_fill) + 1
    moved = jax.nn.one_hot(exact_level, num_out, dtype=dtype) - jax.nn.one_hot(mitred, num_out, dtype=dtype)
    return exact + jnp.sum(move * moved, axis=-2), gate


@timed("corner_pass")
@partial(
    jax.jit,
    static_argnames=(
        "trees", "literals", "is_fill", "lines", "sc", "half", "level_eps", "kernel", "cell", "per_interval"
    ),
)
def _corner_pass(
    layers: _LayerCorner,
    taken: Array,
    dense: Array,
    length: Array,
    present: Array,
    shift: Array,
    moments: tuple[Array, ...] | None = None,
    planes: tuple[Array, Array] | None = None,
    *,
    center: Array | None = None,
    trees: tuple,
    literals: tuple,
    is_fill: tuple[bool, ...],
    lines: int,
    sc: float,
    half: float,
    level_eps: tuple[tuple[float, ...], ...] | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
    cell: float = 1.0,
    per_interval: bool = False,
) -> Array | tuple[Array, ...]:
    """The corner columns of every voxel layer, painted with :func:`_corner_fill`.

    A separate jitted pass with a static structure (not part of the dense scan),
    so its compiled code is shared by every call with the same shapes: sample
    grids, Yee positions, devices of the same stack. Voxel layers where no
    polygon level is in a corner column keep the half-plane result (exact
    there). Each voxel layer is rematerialized in the backward pass, so memory
    stays per layer. With ``per_interval``, the moments per sub-interval are
    returned instead of the tensor, for :func:`_composite_samples` to add up
    with the other sub-samples of the voxels.

    Args:
        layers: local model of every source layer the literals read, (L, N, ...).
        taken: (N,) whether a slot holds a column.
        dense: (nz, N, S, K + 1) half-plane fractions per sub-interval (not
            length weighted).
        length: (nz, N, S) sub-interval lengths.
        present: (nz, N, S, K) whether each level is in each sub-interval.
        shift: (nz, N, S, K, 2) lateral offset of each level at the two Gauss
            heights of its part of the voxel (sides of the square).
        moments: with ``level_eps``, the columns' rates of each level (nz,
            N, S, K, 2), the half-plane contrast flux (nz, N, S, C, 3) and rates
            (nz, N, S, C, 2) and, without ``per_interval``, the sub-interval ends
            (nz, N, S + 1) and, with rough surfaces, their slopes (nz, N, S - 1,
            2); see :func:`_paint_layer`.
        planes: (N, K) offsets and (N, K, 2) normals of the columns'
            half-planes (see :func:`_corner_fill`), or None.
        center: (2,) the kernel's (ax, ay), traced (so that the sub-samples of
            :func:`_composite_samples` share one compiled pass), or None: those
            of ``kernel``.
        trees: per level, its literal tree (None for fill levels).
        literals: per level, (layer in ``layers``, a, b in cells) of each literal.
        is_fill: (K,) whether a level fills the whole plane.
        lines: number of split lines.
        sc: cells -> sides of the lateral square.
        half: half side of the lateral square.
        level_eps: (C, K) contrasts of the levels (see :func:`_composite`), or
            None: fractions only.
        kernel: the flux kernel (see :func:`_composite`).
        cell: cells per side of the lateral square.
        per_interval: return the corner model per sub-interval, unblended,
            with each sample's certificate for its voxel (see
            :func:`_composite_samples`).

    Returns:
        (nz, N, K + 1) volume fractions of the levels, then the background;
        with ``level_eps`` also the (nz, N, C, 4) tensor of
        :func:`_normal_tensor`. With ``per_interval`` instead the corner
        model's fractions (nz, N, S, K + 1; not length weighted), the
        certificate for the voxel (nz, N, S; ``raw`` of :func:`_corner_fill`)
        and, with ``level_eps``, its contrast flux (nz, N, S, C, 3) and rates
        (nz, N, S, C, 2) per sub-interval.
    """
    corner = _CornerInputs(  # the columns are already gathered: no indices
        cols=None, taken=taken, layers=layers, trees=trees, literals=literals, lines=lines, sc=sc
    )
    polygons = np.logical_not(is_fill)
    with_n = level_eps is not None
    if center is not None:
        kernel = (center[0], center[1], kernel[2])
    if with_n:
        kmass = _kernel_mass(kernel)

    @jax.checkpoint
    def repaint(inputs: tuple[Array, ...]) -> Array:
        """(N, K + 1) fractions of one voxel layer from the corner model (and its moments)."""
        if with_n:
            return _corner_fill(
                corner, *inputs[:4], is_fill, half, (*inputs[4:7], level_eps, kernel), per_interval, planes
            )
        return _corner_fill(corner, *inputs[:4], is_fill, half, raw=per_interval, planes=planes)

    def half_planes(inputs: tuple[Array, ...]) -> Array:
        """(N, K + 1) fractions of one voxel layer from the half-planes (and their moments)."""
        dense, length = inputs[:2]
        if per_interval:  # exact: E = T, certified
            return (dense, jnp.ones(length.shape, dtype=length.dtype), *inputs[5:7])
        fill = jnp.stack([jnp.sum(length * dense[..., j], axis=-1) for j in range(dense.shape[-1])], axis=-1)
        return (fill, dense, *inputs[5:7]) if with_n else fill

    def one_slice(inputs: tuple[Array, ...]) -> Array:
        """(N, K + 1) fractions of one voxel layer: the corner model where a polygon level is (and the tensor)."""
        busy = jnp.any(inputs[2][..., polygons] & taken[:, None, None])
        out = jax.lax.cond(busy, repaint, half_planes, inputs)
        if not with_n or per_interval:
            return out
        fill, visible, flux, rate = out
        mean = _weigh_levels(visible[..., :-1], level_eps)
        square = _weigh_levels(visible[..., :-1], level_eps, power=2)
        slopes = inputs[8] if len(inputs) > 8 else None
        return fill, _normal_tensor(inputs[1], inputs[7], flux, mean, square, rate, cell, kmass, slopes)

    return jax.lax.map(one_slice, (dense, length, present, shift, *(moments or ())))


def _corner_fill(
    corner: _CornerInputs,
    dense: Array,
    length: Array,
    present: Array,
    shift: Array,
    is_fill: Sequence[bool],
    half: float,
    moments: tuple | None = None,
    raw: bool = False,
    planes: tuple[Array, Array] | None = None,
) -> Array | tuple[Array, ...]:
    """Paints the corner columns of one voxel layer with the exact local geometry.

    In each corner column and z sub-interval, every literal of every present
    level (:func:`_literals`) is constant (its boundary is farther than the
    square's half-diagonal: exact, since distances are 1-Lipschitz), single
    (one edge within reach: the half-plane at the sample point, exact there)
    or a pair (the two nearest edges of :func:`_layer_corner`, exact within
    ``rpair``). The distinct half-planes of the levels become up to
    ``corner.lines`` split lines plus one leftover plane per level; in each of
    the 2^M cells the lines cut, every level is then one half-plane, full or
    empty, and the cells are painted like the half-plane path. Grown convex
    and shrunk reflex corners are round (see :func:`_round_tips`).

    Every level is painted with its offsets at the two Gauss-Legendre heights
    of its own part of the voxel (``shift``, :func:`_gauss_heights`), and E is
    the mean of the two paintings: near a corner of a sloped wall the area is
    quadratic in z, so E is the exact volume of the 3D shape, as long as no
    vertex or edge crosses a corner or side of the square within the part and
    no z_to_bias knot is inside it. Where an edge is straight in the square,
    the two heights alone differ from the ordinary columns' T at the mean
    offset (by up to ``tan(angle) / (4 sqrt 3)`` of the voxel where the wall
    crosses a side of the square within the voxel), which would show as a
    seam along the edge. So with ``planes`` the columns' half-planes are also
    painted at the two heights (T_g), and the seam ``T - T_g`` is added to E
    level by level in priority order, on the union of the levels painted up
    to each (its fraction c_j, the sum of levels 0..j, which the later levels
    do not change): c_j becomes ``c_j(E) + v_j (c_j(T) - c_j(T_g))``, with v_j
    1 where ``c_j(E) = c_j(T_g)`` (that union's geometry in the square is the
    half-planes: a straight edge, a junction, a collinear vertex), so such an
    edge gets T exactly, and v_j ramping smoothly to 0 (``(1 - gap^2 /
    width^2)^2``, differentiated) as the gap ``|c_j(E) - c_j(T_g)|`` reaches
    a width of ``_STRAIGHT_WIDTH`` times the seam ``|c_j(T) - c_j(T_g)|`` (at
    least ``_STRAIGHT_TOL``). The unions are kept nested and in [0, 1] (c_j
    is clipped to [c_(j-1), 1]: a level keeps what the levels before it
    hold), and level j gets ``c_j - c_(j-1)``, the background the rest. So as
    a vertex bends off collinear or a corner enters the square, the fill
    moves from T to E at a rate of the order of the gap's own (no jump, and
    the gradient sees it), and only where the seam is nonzero (sloped walls)
    does v_j reach into real corners, by at most the seam (5e-4 of a voxel at
    the corners of ``test_offset_corners_get_exact_areas``). Where v_j = 0, E
    stays the two heights' mean, since T's half-plane is not the geometry
    there. A level's fraction thus depends on its own and the earlier levels'
    geometry only: a later level's corner or surface in the square changes
    nothing of it. The moments are mixed as ``v T + (1 - v) E``, with v the
    least v_j. Levels whose parts differ are paired height for height (lower
    with lower), as the half-planes pair the levels' mean offsets; another
    level's surface cutting a level's part changes nothing.

    The exact fractions E are blended with the half-plane ones T by a weight w
    that is 1 where every literal's model is certified (at both heights),
    ramps to 0 as a third feature comes within reach (and with the mode
    switches, so the fill stays continuous), and is 0 where a model is
    invalid, the planes need more lines or a tip cannot be corrected. Since E
    is exact wherever w > 0, the result ``w E + (1 - w) T`` is never farther
    from the level-set model than T, per sub-interval. With several
    sub-squares per voxel, :func:`_composite_samples` blends them all by the
    voxel's certificate instead (``raw``), so that this holds for their
    average. Limit: a hole's cut line or a T-junction that splits an outer
    edge near a corner makes the short piece a third feature there (w = 0 at
    that corner: the half-plane result).

    With ``moments``, the clips carry the planes as :func:`_composite` does;
    the edges that split cells carry their line's plane too, so where one
    material is on both sides their fluxes cancel and the cells' fluxes add up
    to the interfaces' (the round tips' area moves no flux). The flux and rate
    are blended like the fractions.

    Args:
        corner: the corner columns.
        dense: (N, S, K + 1) the columns' half-plane fractions per sub-interval
            (not length weighted).
        length: (N, S) sub-interval lengths.
        present: (N, S, K) whether each level is in each sub-interval.
        shift: (N, S, K, 2) lateral offset of each level at the lower and the
            upper Gauss height of its part of the voxel (sides of the square).
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the lateral square.
        moments: for the normal tensor, the rates of each level (N, S, K, 2;
            its z slope in cells per cell, then the slope times its lag, see
            :func:`_paint_layer`), the half-plane contrast flux (N, S, C, 3)
            and rates (N, S, C, 2), the static (C, K) contrasts and the kernel
            (see :func:`_composite`), or None.
        raw: return E and the certificate for the voxel instead of the blend:
            ``max(w, v)``, with v 1 where every literal of a present level is
            constant, or all but one, which is a single edge, each with
            ``_VOXEL_RAMP`` to spare, ramping to 0 within that margin (so
            ``v > 0`` only where ``w > 0``, and 1 where a column stops being a
            corner column of a sample grid with sub-samples).
        planes: the columns' half-planes: (N, K) offsets (sides of the square,
            without ``shift``) and (N, K, 2) unit normals, as painted into
            ``dense``; or None: E is the two heights' mean alone.

    Returns:
        (N, K + 1) the columns' fractions of the levels, then the background;
        with ``moments`` also the blended fractions per sub-interval (N, S,
        K + 1), contrast flux (N, S, C, 3) and rates (N, S, C, 2). With
        ``raw`` instead E's fractions per sub-interval (N, S, K + 1), the
        certificate (N, S) and, with ``moments``, E's flux and rates.
    """
    sg = jax.lax.stop_gradient
    with_n = moments is not None
    if with_n:
        slope, flux_t, rate_t, level_eps, kernel = moments
    n0 = length.shape[0]
    dtype = length.dtype
    # every level at the two Gauss heights of its part: the columns twice (the
    # second copy at the upper heights), painted together
    corner = corner._replace(
        taken=jnp.concatenate([corner.taken] * 2),
        layers=jax.tree.map(lambda x: jnp.concatenate([x, x], axis=1), corner.layers),
    )
    voxel_length = length
    if with_n:
        # the walls' rates there: the slope, and its lag to the sub-interval's
        # middle, from the middle of the level's part (slope[..., 1]) less the
        # height's distance from it (its part is where it is present)
        part = jnp.sum(jnp.where(present, length[..., None], 0.0), axis=-2)[:, None]  # (N, 1, K)
        b = slope[..., 0]
        slope = jnp.concatenate(
            [jnp.stack([b, slope[..., 1] + g * b * part / (2 * math.sqrt(3.0))], axis=-1) for g in (1.0, -1.0)]
        )
    length, present = (jnp.concatenate([x, x]) for x in (length, present))
    shift = jnp.concatenate([shift[..., 0], shift[..., 1]])
    n, s = length.shape
    rq = math.sqrt(2.0) * half  # the square's half-diagonal
    full_area = (2 * half) ** 2
    sc = corner.sc
    gauss_t = None
    if planes is not None:
        # the columns' half-planes at the same two heights: where E equals
        # them, the edge is straight in the square and E becomes T
        po, pn = (jnp.concatenate([x, x]) for x in planes)  # (2N, K), (2N, K, 2)
        num_t = 4 + sum(not f for f in is_fill)
        square_t = jnp.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=dtype)
        square_t = jnp.concatenate([square_t, jnp.repeat(square_t[-1:], num_t - 4, axis=0)])
        gauss_t = _paint_levels(
            jnp.broadcast_to(square_t, (n, s, num_t, 2)),
            jnp.full((n, s), 4, dtype=jnp.int32),
            jnp.ones((n, s), dtype=dtype),
            jnp.broadcast_to(pn[:, None], (n, s, *pn.shape[1:])),
            po[:, None, :] + shift,
            present,
            is_fill,
            half,
        )

    def ramp(x: Array) -> Array:
        return jnp.clip(x / _CORNER_RAMP, 0.0, 1.0)

    def per_column(x: Array) -> Array:
        """(N, ...) -> (N, S, ...)."""
        return jnp.broadcast_to(x[:, None], (n, s, *x.shape[1:]))

    def cuts(p: Array) -> Array:
        """Whether half-planes (..., 3) cut the square (no gradient)."""
        p = sg(p)
        return jnp.abs(p[..., 2]) < half * (jnp.abs(p[..., 0]) + jnp.abs(p[..., 1]))

    # every literal: constant, single edge or pair, and how far it is certified
    slots: list[_Slot] = []
    slot_slopes = []  # z slope of each literal's planes (they move with their level)
    weight = jnp.where(corner.taken[:, None], jnp.ones((n, s), dtype=dtype), 0.0)
    # and the certificate for the voxel: 1 where every literal is constant, or
    # all but one of each level, which is single, with _VOXEL_RAMP to spare;
    # it is > 0 only where the weight is
    voxel = weight
    for j, fill in enumerate(is_fill):
        if fill:
            continue
        here = present[..., j]
        constant, alone = [], []  # per literal: how surely constant, single
        for layer, a, b in corner.literals[j]:
            m = jax.tree.map(lambda x, layer=layer: x[layer], corner.layers)
            off = a * shift[..., j] + b * sc
            val = m.sd[:, None] + off
            reach = rq + jnp.abs(off) + _CORNER_TOL
            d2s = m.d2s[:, None]
            const = sg(jnp.abs(val)) >= rq
            single = ~const & (d2s > sg(reach))
            pair = ~const & ~single
            today = jnp.concatenate([per_column(m.normal), val[..., None]], axis=-1)
            pa, pb = (
                jnp.concatenate([per_column(m.planes[:, i, :2]), (m.planes[:, i, 2, None] + off)[..., None]], axis=-1)
                for i in (0, 1)
            )
            first = jnp.where(single[..., None], today, pa)
            # 1 where the model holds over the whole reach, ramping to 0 with the
            # radius; the switches between modes happen where both are exact
            cert = ramp(m.rpair[:, None] - reach)
            certified = jnp.where(d2s > reach, jnp.maximum(cert, ramp(d2s - reach)), cert)
            certified = jnp.where(jnp.abs(val) >= rq, jnp.maximum(certified, ramp(jnp.abs(val) - rq)), certified)
            weight = weight * jnp.where(here, certified, 1.0)
            constant.append(jnp.clip((jnp.abs(val) - rq) / _VOXEL_RAMP, 0.0, 1.0))
            alone.append(jnp.where(d2s > reach, jnp.clip((d2s - reach) / _VOXEL_RAMP, 0.0, 1.0), 0.0))
            if with_n:
                slot_slopes.append(a * slope[..., j, :])
            o = sg(off)
            slots.append(
                _Slot(
                    level=j,
                    layer=layer,
                    off=off,
                    val=val,
                    const=const,
                    single=single,
                    op_and=per_column(m.op_and),
                    planes=(first, pb),
                    live=(here & ~const & cuts(first), here & pair & cuts(pb)),
                    rounded=here & pair & m.linked[:, None] & jnp.where(m.convex[:, None], o < 0, o > 0),
                )
            )
        sure = math.prod(constant) + sum(
            g * math.prod(c for k, c in enumerate(constant) if k != i) for i, g in enumerate(alone)
        )
        voxel = voxel * jnp.where(here, jnp.minimum(sure, 1.0), 1.0)

    # distinct live planes: the first of each level is its leftover, the others
    # take the split lines (unused lines are far: their inside is empty)
    num_lines = corner.lines
    far = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 2 * half], dtype=dtype), (n, s, 3))
    lines, used = [far] * num_lines, [jnp.zeros((n, s), dtype=bool)] * num_lines
    left, has = [far] * len(is_fill), [jnp.zeros((n, s), dtype=bool)] * len(is_fill)
    lines_b = [jnp.zeros((n, s, 2), dtype=dtype)] * num_lines  # rates of the lines and leftovers
    left_b = [jnp.zeros((n, s, 2), dtype=dtype)] * len(is_fill)
    ok_lines = jnp.ones((n, s), dtype=bool)
    for i, slot in enumerate(slots):
        j = slot.level
        for plane, live in zip(slot.planes, slot.live, strict=True):
            seen = has[j] & (_same_plane(plane, left[j]) != 0)
            for k in range(num_lines):
                seen = seen | (used[k] & (_same_plane(plane, lines[k]) != 0))
            new = live & ~seen
            leftover = new & ~has[j]
            left[j] = jnp.where(leftover[..., None], plane, left[j])
            if with_n:
                left_b[j] = jnp.where(leftover[..., None], slot_slopes[i], left_b[j])
            has[j] = has[j] | leftover
            rest = new & ~leftover
            for k in range(num_lines):
                put = rest & ~used[k]
                lines[k] = jnp.where(put[..., None], plane, lines[k])
                if with_n:
                    lines_b[k] = jnp.where(put[..., None], slot_slopes[i], lines_b[k])
                used[k] = used[k] | put
                rest = rest & ~put
            ok_lines = ok_lines & ~rest

    # cell c is on the inside of line k if bit k of c is set
    num_poly = sum(not f for f in is_fill)
    num_vertices = 4 + num_lines + num_poly
    square = jnp.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=dtype)
    square = jnp.concatenate([square, jnp.repeat(square[-1:], num_vertices - 4, axis=0)])
    poly = jnp.broadcast_to(square, (n, s, 1, num_vertices, 2))  # (N, S, C, V, 2)
    count = jnp.full((n, s, 1), 4, dtype=jnp.int32)
    attr = jnp.zeros((n, s, 1, num_vertices, 4), dtype=dtype) if with_n else None
    for k in range(num_lines):  # the cells outside line k, then those inside
        cells = poly.shape[2]
        nk = jnp.broadcast_to(lines[k][..., None, :2], (n, s, cells, 2))
        ok = jnp.broadcast_to(jnp.clip(lines[k][..., None, 2], -2 * half, 2 * half), (n, s, cells))
        args = (
            jnp.concatenate([poly, poly], axis=2),
            jnp.concatenate([count, count], axis=2),
            jnp.concatenate([nk, -nk], axis=2),
            jnp.concatenate([ok, -ok], axis=2),
        )
        if with_n:  # both sides' new edges carry the line's plane
            line = jnp.concatenate([lines[k][..., :2], lines_b[k]], axis=-1)
            line = jnp.broadcast_to(line[..., None, :], (n, s, 2 * cells, 4))
            poly, count, attr = _clip(*args, jnp.concatenate([attr, attr], axis=2), line)
        else:
            poly, count = _clip(*args)
    num_cells = poly.shape[2]
    ok_clip = jnp.all(count <= num_vertices, axis=-1)

    # every plane's value on each side of its level's leftover (outside,
    # inside) in each cell: the side of the line or leftover it is, else its
    # constant value over the square
    planes = jnp.stack([p for slot in slots for p in slot.planes], axis=-2)  # (N, S, 2T, 3)
    live = jnp.stack([x for slot in slots for x in slot.live], axis=-1)  # (N, S, 2T)
    level_of = [slot.level for slot in slots for _ in slot.planes]
    own = jnp.stack([left[j] for j in level_of], axis=-2)
    own_has = jnp.stack([has[j] for j in level_of], axis=-1)
    on_left = jnp.where(own_has, _same_plane(planes, own), 0)[..., None, None]  # (N, S, 2T, 1, 1)
    on_line = _same_plane(planes[..., None, :], jnp.stack(lines, axis=-2)[..., None, :, :])  # (N, S, 2T, M)
    on_line = jnp.where(jnp.stack(used, axis=-1)[..., None, :], on_line, 0)
    negative = (sg(planes[..., 2]) < 0)[..., None, None]  # the value over the square if it does not cut it
    sides = jnp.asarray([[False], [True]])  # (2, 1)
    value = jnp.where(on_left == 1, sides, jnp.where(on_left == -1, ~sides, negative))
    for k in range(num_lines):
        bits = jnp.asarray([bool((c >> k) & 1) for c in range(num_cells)])  # (C,)
        on = on_line[..., k, None, None]
        value = jnp.where(on == 1, bits, jnp.where(on == -1, ~bits, value))
    value = jnp.where(live[..., None, None], value, negative)
    value = jnp.broadcast_to(value, (n, s, len(level_of), 2, num_cells))  # (N, S, 2T, 2, C)

    # each level in each cell: full, empty, or inside / outside its leftover
    unit = jnp.asarray([1.0, 0.0], dtype=dtype)
    normals, offsets, slopes = [], [], []
    for j, fill in enumerate(is_fill):
        if fill:
            normals.append(jnp.broadcast_to(unit, (n, s, num_cells, 2)))
            offsets.append(jnp.zeros((n, s, num_cells), dtype=dtype))
            slopes.append(jnp.zeros((n, s, num_cells, 2), dtype=dtype) if with_n else None)
            continue
        values = [
            _literal_value(slot, value[..., 2 * i, :, :], value[..., 2 * i + 1, :, :])
            for i, slot in enumerate(slots)
            if slot.level == j
        ]
        t = jnp.broadcast_to(_eval_bool(corner.trees[j], values), (n, s, 2, num_cells))
        outside, inside = t[..., 0, :], t[..., 1, :]
        lf = left[j][:, :, None, :]
        only_in, only_out = inside & ~outside, outside & ~inside
        normals.append(jnp.where(only_in[..., None], lf[..., :2], jnp.where(only_out[..., None], -lf[..., :2], unit)))
        offsets.append(
            jnp.where(only_in, lf[..., 2], jnp.where(only_out, -lf[..., 2], jnp.where(inside, -2 * half, 2 * half)))
        )
        if with_n:
            lb = left_b[j][:, :, None]
            slopes.append(jnp.where(only_in[..., None], lb, jnp.where(only_out[..., None], -lb, 0.0)))
    painted = _paint_levels(
        poly,
        count,
        _area(poly, count) / full_area,
        jnp.stack(normals, axis=-2),
        jnp.stack(offsets, axis=-1),
        jnp.broadcast_to(present[:, :, None, :], (n, s, num_cells, len(is_fill))),
        is_fill,
        half,
        *((attr, jnp.stack(slopes, axis=-2), kernel) if with_n else ()),
    )
    visible, overflow = painted[:2]
    exact = jnp.sum(visible, axis=2)  # (N, S, K + 1)
    # round joins only where some literal of this voxel layer is rounded
    exact, gate = jax.lax.cond(
        jnp.any(jnp.stack([slot.rounded for slot in slots])),
        lambda e: _round_tips(slots, corner, e, present, is_fill, half),
        lambda e: (e, jnp.ones((n, s), dtype=bool)),
        exact,
    )
    ok = ok_lines & ok_clip & ~jnp.any(overflow, axis=-1) & gate
    w = jnp.where(ok, weight, 0.0)
    if raw:  # the certificate for the voxel instead (> 0 only where w is)
        w = jnp.maximum(w, jnp.where(ok, voxel, 0.0))
    # the two heights: E is their mean, certified where both are
    def mean(x: Array) -> Array:
        """(2N, ...) -> (N, ...): the mean over the two heights."""
        return 0.5 * (x[:n0] + x[n0:])

    exact = mean(exact)  # (N, S, K + 1)
    w = jnp.minimum(w[:n0], w[n0:])  # (N, S)
    if gauss_t is not None:
        # level by level in priority order, on the union of the levels painted
        # up to j (its fraction c_j: the later levels change nothing of it, so
        # neither does their corner or surface in the square): where the
        # corner model's c_j is the half-planes' at both heights (the union's
        # geometry is straight in the square), c_j(E) + (c_j(T) - c_j(T_g)) is
        # T's, the mean offset of the ordinary columns along the edge. That
        # seam fades out smoothly (v_j = (1 - gap^2 / width^2)^2,
        # differentiated) as c_j(E) leaves c_j(T_g), over a width of
        # _STRAIGHT_WIDTH times the seam's size: the fill moves by at most
        # v' |seam| <= 1.54 / _STRAIGHT_WIDTH times the gap's own rate, and a
        # real corner (gap >= width) keeps E exactly. Vertical walls have no
        # seam: E there is untouched.
        tg = mean(gauss_t[0])
        zero = jnp.zeros_like(exact[..., 0])
        sums = (zero, zero, zero)  # c_j of E, T and T_g (in order, so an empty level adds exactly 0)
        below, moved = zero, zero  # c_(j-1) as painted, and what its seam added to it
        fractions, straight = [], []
        for j in range(len(is_fill)):
            sums = tuple(c + x[..., j] for c, x in zip(sums, (exact, dense, tg), strict=True))
            gap, seam = sums[0] - sums[2], sums[1] - sums[2]
            v = (1.0 - jnp.minimum(gap**2 / (_STRAIGHT_WIDTH**2 * seam**2 + _STRAIGHT_TOL**2), 1.0)) ** 2
            # the unions stay nested and in [0, 1]: c_j in [c_(j-1), 1], so a
            # level keeps what the levels before it hold (where clipped, the
            # gradient is the bound's; at a tie, the union's own)
            c = sums[0] + v * seam
            c = jnp.where(c > 1.0, 1.0, c)
            c = jnp.where(c < below, below, c)
            # level j: E_j plus what the seam added to c_j less what it added to
            # c_(j-1) (E_j itself where neither has a seam)
            add = c - sums[0]
            fractions.append(exact[..., j] + add - moved)
            straight.append(v)
            below, moved = c, add
        exact = jnp.stack([*fractions, exact[..., -1] - moved], axis=-1)  # (N, S, K + 1)
        straight = jnp.stack(straight, axis=-1)  # (N, S, K)
    if with_n:
        flux = _weigh_levels(jnp.sum(painted[2], axis=2), level_eps, axis=-2)  # (2N, S, C, 3)
        rate = _weigh_levels(jnp.sum(painted[3], axis=2), level_eps, axis=-2)  # (2N, S, C, 2)
        flux, rate = mean(flux), mean(rate)
        if gauss_t is not None:
            # one mix for all levels (the moments are summed over them): the
            # least of the levels' v, so T's only where every union is straight
            mix = jnp.min(straight, axis=-1)[..., None, None]  # (N, S, 1, 1)
            flux = mix * flux_t + (1 - mix) * flux
            rate = mix * rate_t + (1 - mix) * rate
    if raw:
        return (exact, w, flux, rate) if with_n else (exact, w)
    return _blend(w, exact, dense, voxel_length, *((flux, flux_t, rate, rate_t) if with_n else ()))


def _blend(w: Array, exact: Array, dense: Array, length: Array, *moments: Array) -> Array | tuple[Array, ...]:
    """``w E + (1 - w) T`` of the corner model E and the half-planes T (see :func:`_corner_fill`).

    Args:
        w: (N, S) the weight.
        exact: (N, S, K + 1) E's fractions per sub-interval.
        dense: (N, S, K + 1) T's.
        length: (N, S) sub-interval lengths.
        moments: optionally E's flux (N, S, C, 3), T's, E's rates (N, S, C,
            2) and T's.

    Returns:
        (N, K + 1) the blended fractions (length weighted); with ``moments``
        also the blended fractions, flux and rates per sub-interval.
    """
    vis = w[..., None] * exact + (1 - w[..., None]) * dense
    fill = jnp.stack([jnp.sum(length * vis[..., j], axis=-1) for j in range(vis.shape[-1])], axis=-1)
    if not moments:
        return fill
    flux, flux_t, rate, rate_t = moments
    w = w[..., None, None]
    return fill, vis, w * flux + (1 - w) * flux_t, w * rate + (1 - w) * rate_t


def _lateral_range(level: _Level, wall: tuple[Array, Array, Array], lo: Array, hi: Array, nz: int,
                   res_um: float) -> tuple[Array, Array]:
    """Smallest and largest sidewall offset (cells) of a level in the box (no gradient).

    The offset is piecewise linear in z (kinks at the z_to_bias knots), so its
    range over the heights the level covers is reached at their ends or a knot.

    Args:
        level: the level.
        wall: its (zmin, thickness, sidewall angle) as in :func:`_sidewall_shift`.
        lo: (nx, ny) its bottom surface (cells).
        hi: (nx, ny) its top surface (cells).
        nz: number of voxel layers.
        res_um: grid resolution in um.

    Returns:
        (min, max) of the offset.
    """
    sg = jax.lax.stop_gradient
    zb, t, angle = (sg(jnp.asarray(x, dtype=lo.dtype)) for x in wall)
    z_lo = jnp.clip(jnp.min(sg(lo)), 0.0, nz)
    z_hi = jnp.clip(jnp.max(sg(hi)), 0.0, nz)
    z = [z_lo, z_hi]
    if level.z_to_bias is not None:
        z += [jnp.clip(zb + f * t, z_lo, z_hi) for f in level.z_to_bias[0]]
    offsets = _sidewall_shift(level, jnp.stack(z), zb, t, angle, res_um)
    return jnp.min(offsets), jnp.max(offsets)


def _corner_candidates(plan: _CornerPlan, radii: Sequence[Sequence[Array]], dist: Sequence[Array]) -> Array:
    """The columns where the half-plane of some level may not be exact (no gradient).

    A literal is near a column if its layer's nearest boundary edge is within
    its radius (the lateral square's half-diagonal plus the largest offset of
    the literal in the box). Elsewhere a literal is constant over the square.
    A level whose only near literal has a single edge within the radius is
    exactly today's half-plane: the point lies in the interior Voronoi region
    of that edge (if its nearest point were a vertex, the edge continuing there
    would be as near), so the boundary there is that edge's line. The other
    columns are the candidates: two near literals of one level (a crossing of
    layer edges), or a near literal with a second edge within its radius (a
    corner, a strip or a gap).

    Args:
        plan: the levels' literals.
        radii: per level, per literal: the radius (cells).
        dist: per layer slot of the plan, (P, 3) distances (cells) to the
            nearest boundary edges (:func:`gdsfactory.fdtdx_geometry.nearest_edges`).

    Returns:
        (P,) whether each column is a candidate.
    """
    candidate = jnp.zeros(dist[0].shape[0], dtype=bool)
    for literals, rad in zip(plan.literals, radii, strict=True):
        near_count = jnp.zeros(candidate.shape, dtype=jnp.int32)
        for (layer, *_), r in zip(literals, rad, strict=True):
            near = dist[layer][:, 0] <= r
            near_count = near_count + near.astype(jnp.int32)
            candidate = candidate | (near & (dist[layer][:, 1] <= r))
        candidate = candidate | (near_count >= 2)
    return candidate


# ---------------------------------------------------------------- diagnostics (concrete, KLayout)


class StackDiagnostic(NamedTuple):
    """One kind of problem in one level's geometry, found by :meth:`StackDevice.diagnose` (lengths in um).

    Attributes:
        level: the level's name.
        kind: ``"width"``: parts of the level's shape narrower than
            ``threshold``; ``"gap"``: gaps in it narrower than ``threshold``;
            ``"overlap"``: overlapping polygons on layers the level shrinks.
        size: the narrowest width or gap found in the box (of places within
            1% of the threshold of the narrowest, the longest one); for
            ``"overlap"``, the largest shrink of the overlapping layers.
        cells: ``size`` in grid cells.
        threshold: the width or gap below which parts are reported (0 for
            ``"overlap"``).
        location: (x, y) gds point in the middle of that width or gap, in the
            box, or in the largest overlap.
        z: stack height of the cross-section with the narrowest width or gap,
            or where the shrink is largest.
        count: number of separate places found in the box.
        layers: for ``"overlap"``, the parameter keys (``vertices/<layer>``)
            of the layers whose polygons overlap; empty otherwise.
        message: the warning text.
    """

    level: str
    kind: Literal["width", "gap", "overlap"]
    size: float
    cells: float
    threshold: float
    location: tuple[float, float]
    z: float
    count: int
    layers: tuple[str, ...]
    message: str


class StackDiagnosticWarning(UserWarning):
    """A :class:`StackDiagnostic` reported when a :class:`StackDevice` is placed (see its ``diagnostics``)."""


def _wall_offset(level: _Level, z: np.ndarray) -> np.ndarray:
    """:func:`_sidewall_shift` in um at stack heights z (um), for the level's nominal zmin, thickness and angle."""
    z = np.asarray(z, dtype=float)
    z_ref = level.zmin + level.width_to_z * level.thickness
    grow = level.bias
    if level.z_to_bias is not None:
        zs, bs = level.z_to_bias
        grow = grow + np.interp((z - level.zmin) / level.thickness, zs, bs)
    return (z - z_ref) * math.tan(math.radians(level.sidewall_angle)) - grow


def _check_heights(level: _Level, lo: float, hi: float, step: float) -> np.ndarray:
    """Stack heights (um) of the cross-sections of a level that are checked.

    The level's offset is piecewise linear in z (kinks at the z_to_bias knots).
    The heights include ``lo``, ``hi`` and the knots between them, and are
    spaced so that the offset changes by at most ``step`` from one to the next
    (at most about ``_CHECK_HEIGHTS`` heights; all knots are kept).

    Args:
        level: the level.
        lo: lowest height (um).
        hi: highest height (um), above ``lo``.
        step: largest change of the offset (um) between neighbouring heights.

    Returns:
        The increasing heights.
    """
    knots = [lo, hi]
    if level.z_to_bias is not None:
        knots += [level.zmin + f * level.thickness for f in level.z_to_bias[0]]
    knots = np.unique(np.clip(knots, lo, hi))
    steps = np.maximum(1, np.ceil(np.abs(np.diff(_wall_offset(level, knots))) / step)).astype(int)
    if steps.sum() >= _CHECK_HEIGHTS:
        steps = np.maximum(1, steps * (_CHECK_HEIGHTS - 1) // steps.sum())
    pieces = [np.linspace(a, b, n + 1)[1:] for a, b, n in zip(knots[:-1], knots[1:], steps, strict=True)]
    return np.concatenate([knots[:1], *pieces])


def _kdb_region(polygons: Sequence[np.ndarray], dbu: float, merged: bool = True) -> Any:
    """A KLayout region of polygons (um) on a grid of ``dbu`` um; ``merged=False`` keeps their overlaps."""
    import klayout.db as kdb

    region = kdb.Region()
    region.merged_semantics = merged
    for p in polygons:
        q = np.rint(np.asarray(p, dtype=float) / dbu).astype(np.int64)
        region.insert(kdb.Polygon([kdb.Point(int(x), int(y)) for x, y in q]))
    return region


def _corners(region: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convex and reflex vertices of a merged KLayout region, with the arcs of their round joins.

    KLayout's hulls run clockwise and its holes counterclockwise, so the region
    is on the right of every contour and its convex vertices turn right. The
    round join of radius r at a convex vertex (where the region grows) or a
    reflex one (where it shrinks) is the arc of radius r around the vertex from
    the normal of its incoming edge to that of its outgoing edge, turning with
    the contour; it bulges up to ``r (1 - cos(turn / 2))`` beyond the chord
    between its ends, which is where KLayout's sizing in mode 0 cuts the corner.

    Args:
        region: the merged region.

    Returns:
        (convex, reflex): (n, 5) arrays of x, y (dbu), the angle of the join's
        first normal (outward at convex vertices, inward at reflex ones), the
        signed turn (radians) and ``1 - cos(turn / 2)``.
    """
    convex, reflex = [np.zeros((0, 5))], [np.zeros((0, 5))]
    for poly in region.each_merged():
        for contour in (poly.each_point_hull(), *(poly.each_point_hole(h) for h in range(poly.holes()))):
            c = np.array([(p.x, p.y) for p in contour], dtype=float)
            if len(c) < 3:
                continue
            e_in, e_out = c - np.roll(c, 1, axis=0), np.roll(c, -1, axis=0) - c
            cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
            turn = np.arctan2(cross, np.sum(e_in * e_out, axis=1))
            first = np.arctan2(e_in[:, 0], -e_in[:, 1]) + np.where(cross < 0, 0.0, math.pi)  # left or right normal
            rows = np.column_stack([c, first, turn, 1.0 - np.cos(0.5 * turn)])
            convex.append(rows[cross < 0])
            reflex.append(rows[cross > 0])
    return np.concatenate(convex), np.concatenate(reflex)


def _seeds(points: np.ndarray) -> Any:
    """A raw (unmerged) KLayout region of 2 x 2 dbu boxes centered on points ((n, >= 2) array, dbu)."""
    import klayout.db as kdb

    region = kdb.Region()
    region.merged_semantics = False
    for x, y in np.rint(points[:, :2]).astype(np.int64):
        region.insert(kdb.Box(int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1))
    return region


def _fans(points: np.ndarray, radius: float, tol: float) -> Any:
    """Round joins of a radius (dbu) at corners (rows of :func:`_corners`), as a KLayout region.

    Each join is the polygon from the corner to the ends of its arc and along
    the arc's circumscribed polygon, which is tangent to the two offset edges
    at their ends (it continues them without a step) and at most ``tol`` (dbu)
    outside the circle.
    """
    import klayout.db as kdb

    most = 2 * math.acos(1 / (1 + tol / radius))  # largest turn of one tangent piece
    fans = kdb.Region()
    templates: dict[float, Any] = {}  # the fan of a turn at the origin, its first normal along x
    for x, y, first, turn, _ in points:
        if turn not in templates:
            k = max(1, math.ceil(abs(turn) / most))
            step = turn / k
            angles = np.concatenate([[0.0], (np.arange(k) + 0.5) * step, [turn]])
            radii = np.full(k + 2, radius / math.cos(0.5 * step))
            radii[[0, -1]] = radius
            ends = zip(radii * np.cos(angles), radii * np.sin(angles), strict=True)
            templates[turn] = kdb.DPolygon([kdb.DPoint(0, 0), *(kdb.DPoint(float(u), float(w)) for u, w in ends)])
        fan = templates[turn].transformed(kdb.DCplxTrans(1.0, math.degrees(first), False, float(x), float(y)))
        fans.insert(fan.to_itype(1.0))
    return fans


def _round_offset(
    layer: Any,
    grow: int,
    corners: tuple[tuple[np.ndarray, Any], tuple[np.ndarray, Any]],
    near: Any,
    reach: int,
    tol: float,
) -> tuple[Any, np.ndarray]:
    """A layer's region offset the way the device draws it: by its signed distance, with round joins.

    KLayout's sizing in mode 0 cuts every corner between the ends of its two
    offset edges. That is the exact offset except at the round joins (convex
    corners where the layer grows, reflex ones where it shrinks): there the cut
    lies inside the join, and the join's fan (see :func:`_fans`) completes it.
    A corner keeps its cut where the join bulges at most ``tol`` beyond it, or
    where no edge but the corner's own two comes within ``reach + |grow|`` of
    the corner: then no width or gap involves the join, and the cut, like the
    join, meets the corner's two sides where they end (so their pair measures
    the same, see :func:`_narrow_places`).

    Args:
        layer: the layer's merged region (dbu).
        grow: offset (dbu, negative shrinks).
        corners: (vertices, seeds) of the layer's convex and of its reflex
            vertices (see :func:`_corners` and :func:`_seeds`).
        near: edges of every layer the checked shape is made of (raw).
        reach: the checks' distance plus the largest offset of those layers in
            the shape (dbu).
        tol: largest deviation (dbu) of the drawn joins from round ones.

    Returns:
        The offset region, and (m, 3) x, y and radius (dbu) of the joins drawn.
    """
    none = np.zeros((0, 3))
    if not grow:
        return layer, none
    cut = layer.sized(grow, grow, 0)
    points, seeds = corners[0] if grow > 0 else corners[1]
    r = abs(grow)
    big = points[:, 4] * r > tol
    if not big.any():
        return cut, none
    boxes = seeds.sized(reach + r)
    boxes.merged_semantics = False  # one box per corner, also where they overlap
    hits = [complex(c.x, c.y) for c in (p.bbox().center() for p in boxes.interacting(near, 3).each())]
    chosen = big & np.isin(np.rint(points[:, 0]) + 1j * np.rint(points[:, 1]), np.array(hits, dtype=complex))
    if not chosen.any():
        return cut, none
    fans = _fans(points[chosen], r, tol)
    joins = np.column_stack([points[chosen, :2], np.full(int(chosen.sum()), float(r))])
    return (cut | fans if grow > 0 else cut - fans), joins


def _tree_region(
    tree: tuple,
    literals: Sequence[tuple[int, int, float]],
    literal: Callable[[int, int], Any],
    s: float,
    universe: Any,
    dbu: float,
) -> Any:
    """A level's shape at lateral offset s, as a KLayout region (see :func:`_literals`).

    Args:
        tree: the level's literal tree.
        literals: (layer index, a, b) per literal, b in um.
        literal: the region of a layer (index) grown by an offset (dbu,
            negative shrinks).
        s: the level's offset (um, positive shrinks).
        universe: region of the plane the checks see (the complement of a
            literal is taken in it).
        dbu: grid of the regions (um).

    Returns:
        The region.
    """
    import klayout.db as kdb

    kind = tree[0]
    if kind == "lit":
        index, a, b = literals[tree[1]]
        return literal(index, round(-(a * s + b) / dbu))
    if kind in ("true", "false"):
        return universe if kind == "true" else kdb.Region()
    if kind == "neg":
        return universe - _tree_region(tree[1], literals, literal, s, universe, dbu)
    x = _tree_region(tree[1], literals, literal, s, universe, dbu)
    y = _tree_region(tree[2], literals, literal, s, universe, dbu)
    return x & y if kind == "and" else x | y


def _clip_segments(segments: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The parts of segments ((n, 2, 2) array) in the box ``|x| <= half`` (Liang-Barsky).

    Returns:
        The parts ((n, 2, 2)), and (n,) whether each one is in the open box (a
        part only on the box's boundary, along a face or at a point, is not).
    """
    p, d = segments[:, 0], segments[:, 1] - segments[:, 0]
    t0, t1 = np.zeros(len(p)), np.ones(len(p))
    inside = np.ones(len(p), dtype=bool)
    for k in range(2):
        for sign in (-1.0, 1.0):  # sign * (p + t d) <= half
            room, rate = half[k] - sign * p[:, k], sign * d[:, k]
            inside &= (rate != 0) | (room >= 0)
            t = room / np.where(rate == 0, 1.0, rate)
            t1 = np.where(rate > 0, np.minimum(t1, t), t1)
            t0 = np.where(rate < 0, np.maximum(t0, t), t0)
    parts = p[:, None] + np.stack([t0, t1], axis=1)[:, :, None] * d[:, None]
    inside &= (t0 <= t1) & np.all(np.abs(parts.mean(axis=1)) < half, axis=1)
    return parts, inside


def _closest_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distances between pairs of segments that do not cross ((n, 2, 2) arrays) and a closest point on each.

    Parallel segments are closest along the overlap of their projections: the
    points are then in its middle.

    Returns:
        (n,) distances, and (n, 2) points on ``a`` and on ``b``.
    """

    def foot(p: np.ndarray, s: np.ndarray) -> np.ndarray:
        """The points of segments s closest to points p."""
        d = s[:, 1] - s[:, 0]
        n = np.einsum("ij,ij->i", d, d)
        t = np.einsum("ij,ij->i", p - s[:, 0], d) / np.where(n > 0, n, 1.0)
        return s[:, 0] + np.clip(t, 0.0, 1.0)[:, None] * d

    ps = np.stack([a[:, 0], a[:, 1], foot(b[:, 0], a), foot(b[:, 1], a)], axis=1)
    qs = np.stack([foot(a[:, 0], b), foot(a[:, 1], b), b[:, 0], b[:, 1]], axis=1)
    rows = np.arange(len(a))
    k = np.argmin(np.linalg.norm(ps - qs, axis=2), axis=1)
    p, q = ps[rows, k], qs[rows, k]
    ua, ub = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    ua2 = np.einsum("ij,ij->i", ua, ua)
    lengths = np.sqrt(ua2 * np.einsum("ij,ij->i", ub, ub))
    t = np.einsum("nij,nj->ni", b - a[:, :1], ua) / np.where(ua2 > 0, ua2, 1.0)[:, None]  # b's ends along a
    lo, hi = np.maximum(t.min(axis=1), 0.0), np.minimum(t.max(axis=1), 1.0)
    parallel = (lengths > 0) & (np.abs(ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]) <= 1e-12 * lengths) & (lo <= hi)
    if parallel.any():
        middle = a[:, 0] + (0.5 * (lo + hi))[:, None] * ua
        p = np.where(parallel[:, None], middle, p)
        q = np.where(parallel[:, None], foot(middle, b), q)
    return np.linalg.norm(p - q, axis=1), p, q


def _on_joins(segments: np.ndarray, joins: np.ndarray, tol: float) -> np.ndarray:
    """Whether segments ((n, 2, 2), dbu) lie on round joins: their ends and middle on the arc of one of ``joins``.

    ``joins`` are (m, 3) x, y and radius (dbu); the arcs are drawn up to
    ``tol`` outside their circle.
    """
    out = np.zeros(len(segments), dtype=bool)
    if not len(joins) or not len(segments):
        return out
    slack = tol + 2.0
    largest = float(joins[:, 2].max())
    # a piece of a join is at most this long: only short segments can be on one
    short = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 2 * math.sqrt((2 * largest + slack) * slack) + 4
    if not short.any():
        return out
    by_x = joins[np.argsort(joins[:, 0])]
    which = np.flatnonzero(short)
    points = np.concatenate([segments[which], segments[which].mean(axis=1, keepdims=True)], axis=1)  # ends, middle
    lo = np.searchsorted(by_x[:, 0], points[:, 2, 0] - largest - slack, side="left")
    hi = np.searchsorted(by_x[:, 0], points[:, 2, 0] + largest + slack, side="right")
    for i, start, stop, p in zip(which, lo, hi, points, strict=True):
        if stop > start:
            j = by_x[start:stop]
            off = np.hypot(j[None, :, 0] - p[:, None, 0], j[None, :, 1] - p[:, None, 1]) - j[None, :, 2]
            out[i] = bool(np.any(np.all((off > -2.0) & (off < slack), axis=0)))
    return out


def _hull_key(poly: Any) -> tuple:
    """The points of a KLayout polygon's hull, sorted (identifies a polygon whatever its first point)."""
    return tuple(sorted((p.x, p.y) for p in poly.each_point_hull()))


def _joined(a: np.ndarray, b: np.ndarray, shape: Any) -> np.ndarray:
    """Whether the shape's boundary joins the two edges of each pair where they converge.

    The two sides of a corner are joined there: along the segment between
    their closest points (a cut corner, or a blunt tip), or by a chain bulging
    beyond it toward the point where their lines meet (a round corner; sides
    that touch, as at a sharp corner, are not narrow anyway). So are a part's
    side and the round join that ends the part (the end of a strip grown with
    round corners). A pair is therefore joined where the boundary passes
    through the middle of its closest points, or crosses the triangle they span
    with the point where the lines of the two edges meet. A corner pointing at
    an edge, two corners facing each other, or the side of a V-notch facing the
    edge it nearly cuts through, are not joined there.

    Args:
        a: the first edge of each pair ((n, 2, 2), dbu).
        b: the second edge of each pair.
        shape: the checked region.

    Returns:
        (n,) whether each pair is joined.
    """
    import klayout.db as kdb

    d, p, q = _closest_points(a, b)
    mid = np.rint(0.5 * (p + q)).astype(np.int64)
    half = np.clip(d // 4, 1, 2).astype(np.int64)  # the edges of the pair stay d / 2 away from the middle
    ua, ub = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    den = ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]
    w = b[:, 0] - a[:, 0]
    x = a[:, 0] + ((w[:, 0] * ub[:, 1] - w[:, 1] * ub[:, 0]) / np.where(den != 0, den, 1.0))[:, None] * ua
    sides = np.linalg.norm(np.stack([q - x, x - p, p - q], axis=1), axis=2)  # opposite p, q and x
    perimeter = sides.sum(axis=1)
    area = 0.5 * np.abs((q - p)[:, 0] * (x - p)[:, 1] - (q - p)[:, 1] * (x - p)[:, 0])
    # where the lines meet (within 32-bit coordinates), and the triangle is more than a dbu thick (its
    # inradius is 2 area / perimeter): moved in by one dbu toward its incenter
    thick = np.flatnonzero((den != 0) & (sides[:, 1] < 2**28) & (2 * area > 1.5 * perimeter))
    tri = np.stack([p[thick], q[thick], x[thick]], axis=1)
    center = np.einsum("ni,nij->nj", sides[thick], tri) / perimeter[thick, None]
    scale = 1.0 - perimeter[thick] / (2 * area[thick])
    tri = np.rint(center[:, None] + scale[:, None, None] * (tri - center[:, None]))
    mids, triangles = kdb.Region(), kdb.Region()
    mids.merged_semantics = triangles.merged_semantics = False
    keys = [[] for _ in range(len(a))]
    for i, ((u, v), h) in enumerate(zip(mid.tolist(), half.tolist(), strict=True)):
        box = kdb.Polygon(kdb.Box(u - h, v - h, u + h, v + h))
        mids.insert(box)
        keys[i].append(_hull_key(box))
    for i, corners in zip(thick.tolist(), tri.astype(np.int64).tolist(), strict=True):
        triangle = kdb.Polygon([kdb.Point(u, v) for u, v in corners])
        triangles.insert(triangle)
        keys[i].append(_hull_key(triangle))
    edges = shape.edges()
    hit = {_hull_key(poly) for poly in mids.interacting(edges).each()}
    hit |= {_hull_key(poly) for poly in triangles.interacting(edges).each()}
    return np.array([any(k in hit for k in key) for key in keys], dtype=bool)


def _narrow_places(
    pairs: Any, inside: tuple[int, int], limit: int, shape: Any, joins: np.ndarray, tol: float
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """The places of a width or space check that are narrow in the box.

    The box is ``|x| < inside[0]``, ``|y| < inside[1]`` (dbu). A pair of edges
    counts where at least one of them enters the box. It is measured between
    the parts of its edges in the box (or the whole edge that stays outside),
    so a gap that narrows outside the box gets its width at the box, and it is
    left out where that is ``limit`` or wider. Pairs of ``tol`` or less
    (touching edges, as at a sharp corner, or round-off of the grid and of the
    joins) are left out, and so are pairs whose edges the boundary joins where
    they converge (see :func:`_joined`) if they face each other at
    ``_NARROW_ANGLE`` or more (the two sides of a corner) or one of them is on
    a round join (the corner's own round shape). Sides joined at a smaller
    angle (a needle, a taper, a strip) are measured.

    Args:
        pairs: the KLayout edge pairs of the check.
        inside: half-sizes of the box (dbu).
        limit: the distance of the check (dbu).
        shape: the checked region.
        joins: (m, 3) x, y and radius (dbu) of the round joins in the shape.
        tol: largest deviation (dbu) of the joins from round ones (at least
            one dbu).

    Returns:
        Per narrow place: its width, a point in the middle of its narrowest
        part, and the lower-left and upper-right corners ((2, 2)) of its part in
        the box, all in dbu.
    """
    rows = [
        (e.x1, e.y1, e.x2, e.y2, f.x1, f.y1, f.x2, f.y2)
        for e, f in ((pair.first, pair.second) for pair in pairs.each() if pair.distance() > tol)
    ]
    if not rows:
        return []
    a, b = np.moveaxis(np.array(rows, dtype=float).reshape(-1, 2, 2, 2), 1, 0)
    half = np.asarray(inside, dtype=float)
    ca, in_a = _clip_segments(a, half)
    cb, in_b = _clip_segments(b, half)
    d, p, q = _closest_points(np.where(in_a[:, None, None], ca, a), np.where(in_b[:, None, None], cb, b))
    keep = (in_a | in_b) & (d < limit)
    middle = np.clip(0.5 * (p + q), -half, half)
    lo = np.minimum(middle, np.minimum(np.where(in_a[:, None], ca.min(axis=1), np.inf),
                                       np.where(in_b[:, None], cb.min(axis=1), np.inf)))
    hi = np.maximum(middle, np.maximum(np.where(in_a[:, None], ca.max(axis=1), -np.inf),
                                       np.where(in_b[:, None], cb.max(axis=1), -np.inf)))
    ua, ub = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    lengths = np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1)
    # facing each other at _NARROW_ANGLE or more, or one of them on a round join
    test = keep & (lengths > 0) & (-np.einsum("ij,ij->i", ua, ub) <= math.cos(math.radians(_NARROW_ANGLE)) * lengths)
    rest = np.flatnonzero(keep & ~test)
    if len(rest) and len(joins):
        test[rest[_on_joins(a[rest], joins, tol) | _on_joins(b[rest], joins, tol)]] = True
    if test.any():
        keep[np.flatnonzero(test)[_joined(a[test], b[test], shape)]] = False
    return [(float(d[i]), middle[i], np.array([lo[i], hi[i]])) for i in np.flatnonzero(keep)]


def _overlaps(polygons: Sequence[np.ndarray], dbu: float, box: Any) -> tuple[int, float, np.ndarray] | None:
    """Where polygons (um) overlap with positive area in a box (KLayout region).

    Overlaps of two dbu or thinner (round-off of abutting polygons) are left out.

    Returns:
        (number of separate overlaps, their area in um^2, (2,) center (um) of
        the largest one's bounding box), or None if they do not overlap.
    """
    overlap = (_kdb_region(polygons, dbu, merged=False).merged(False, 2) & box).sized(-1).sized(1)
    if overlap.is_empty():
        return None
    largest = max(overlap.each(), key=lambda p: p.area())
    c = largest.bbox().center()
    return overlap.count(), float(overlap.area()) * dbu**2, np.array([c.x, c.y], dtype=float) * dbu


# ---------------------------------------------------------------- derived shapes

_NO_LAYER = -(2**30)  # a layer index no slot has: read as empty


class _DerivedShape(NamedTuple):
    """The exact shape of a level's layer expression (see :func:`derive_polygons`), drawn as one layer slot."""

    #: the expression without the root's sizing (levels with equal ones share the shape)
    expr: tuple
    #: names of the levels that draw it
    levels: tuple[str, ...]
    #: layer indices of the pool (their vertices concatenated, in this order)
    layers: tuple[int, ...]
    #: the recipes of its parts (rings that touch at a point are in different
    #: parts): vertices are copies of pool vertices and crossings of pool edges
    recipes: tuple[DerivedRecipe, ...]
    #: distance (um) within which the recipe's nodes were found
    tol: float


def _root_grow(expr: tuple) -> float:
    """The sizing of an expression's root node (um)."""
    return float(expr[1] if expr[0] == "fill" else expr[2] if expr[0] == "layer" else expr[3])


def _with_root_grow(expr: tuple, grow: float) -> tuple:
    """The expression with its root node's sizing replaced."""
    if expr[0] == "fill":
        return ("fill", grow)
    if expr[0] == "layer":
        return ("layer", expr[1], grow)
    return (expr[0], expr[1], expr[2], grow)


def _inner_grows(expr: tuple) -> bool:
    """Whether any node below the root is sized."""

    def any_grow(node: tuple) -> bool:
        if node[0] in ("fill", "layer"):
            return bool(_root_grow(node))
        return bool(node[3]) or any_grow(node[1]) or any_grow(node[2])

    return expr[0] not in ("fill", "layer") and (any_grow(expr[1]) or any_grow(expr[2]))


def _slot_expr(expr: tuple, slots: dict[int, int]) -> tuple:
    """The (unsized) expression over pool slots for :func:`derive_polygons`; missing layers are empty."""
    kind = expr[0]
    if kind == "fill":
        return ("fill",)
    if kind == "layer":
        return ("layer", slots[expr[1]]) if expr[1] in slots else ("not", ("fill",), ("fill",))
    return (kind, _slot_expr(expr[1], slots), _slot_expr(expr[2], slots))


def _shapely_polygons(v: np.ndarray, sizes: Sequence[int]) -> Any:
    """The union (shapely, nonzero rule) of a layer's polygons."""
    import shapely

    polys = [shapely.make_valid(shapely.Polygon(p)) for p in np.split(np.asarray(v, dtype=float), np.cumsum(sizes)[:-1])
             if len(p) >= 3]
    return shapely.union_all(polys) if polys else shapely.Polygon()


def _union_overlaps(v: np.ndarray, sizes: Sequence[int], tol: float) -> bool:
    """Whether a layer's polygons overlap with positive area (more than ``tol`` wide)."""
    import shapely

    polys = [shapely.make_valid(shapely.Polygon(p)) for p in np.split(np.asarray(v, dtype=float), np.cumsum(sizes)[:-1])
             if len(p) >= 3]
    if len(polys) < 2:
        return False
    tree = shapely.STRtree(polys)
    ii, jj = tree.query(polys, predicate="intersects")
    for i, j in zip(ii, jj, strict=True):
        if i < j and not shapely.intersection(polys[i], polys[j]).buffer(-tol).is_empty:
            return True
    return False


def _part_index(shape: int, part: int) -> int:
    """Layer slot index of a part of a derived shape (negative, unlike gds layer indices)."""
    return -1 - 1024 * shape - part


def _shape_region(recipes: Sequence[DerivedRecipe], pool: np.ndarray) -> Any:
    """The bounded region (shapely) a derived shape's parts draw from pool vertices (before ``negate``)."""
    import shapely

    parts = [_recipe_region(r, run_recipe(r.kind, r.index, r.positions, pool, np)) for r in recipes if len(r.kind)]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def _recipe_region(recipe: DerivedRecipe, points: np.ndarray) -> Any:
    """The bounded region (shapely) a recipe's evaluated vertices draw (before ``negate``)."""
    import shapely

    region = shapely.Polygon()
    holes = []
    for ring in recipe_rings(recipe, points):
        if len(ring) < 3:
            continue
        poly = shapely.make_valid(shapely.Polygon(ring))
        if shapely.LinearRing(ring).is_ccw:
            region = region.union(poly)
        else:
            holes.append(poly)
    for h in holes:
        region = region.difference(h)
    return region


def _expr_region(expr: tuple, layers: dict[int, Any], universe: Any) -> Any:
    """The region (shapely) of an unsized expression on the layers' unions; missing layers are empty."""
    import shapely

    kind = expr[0]
    if kind == "fill":
        return universe
    if kind == "layer":
        return layers.get(expr[1], shapely.Polygon())
    x, y = _expr_region(expr[1], layers, universe), _expr_region(expr[2], layers, universe)
    return {"and": x.intersection, "or": x.union, "not": x.difference, "xor": x.symmetric_difference}[kind](y)


def _ring_area(ring: np.ndarray) -> float:
    """Signed area of a ring (positive counterclockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _derived_shapes(
    levels: Sequence[_Level],
    sources: Sequence[tuple[str, int]],
    vertices: Sequence[np.ndarray],
    sizes: Sequence[tuple[int, ...]],
    tol: Callable[[np.ndarray], float],
    name: str,
) -> tuple[_DerivedShape, ...]:
    """The exact shapes of the derived levels (see ``StackDevice.derived_shapes``).

    A level gets one when it is not a fill and its expression, with empty
    layers folded away, is more than one layer (or its complement), or one
    layer whose polygons overlap. Levels whose shape cannot be built (sizing
    below the expression's root, a corner without a formula, a recipe that
    does not reproduce the boolean) keep the boolean of offset layers, with a
    :class:`StackDiagnosticWarning`.

    Args:
        levels: the compiled levels.
        sources: (key, layer index) per source layer.
        vertices: their vertices (um).
        sizes: their vertices per polygon.
        tol: the topology tolerance (um) of a layer's vertices.
        name: the device's name (for warnings).

    Returns:
        The shapes; levels with equal unsized expressions share one.
    """
    import shapely

    by_index = {index: i for i, (_, index) in enumerate(sources)}
    nonempty = set(by_index)
    overlapping: dict[int, bool] = {}
    shapes: list[_DerivedShape] = []
    for lv in levels:
        if lv.expr[0] == "fill":
            continue
        key = _with_root_grow(lv.expr, 0.0)
        tree, literals = _literals(key, 1.0, nonempty)
        if tree in (_TRUE, _FALSE):
            continue
        single = tree[0] == "lit" or (tree[0] == "neg" and tree[1][0] == "lit")
        if single:
            index = literals[0][0]
            if index not in overlapping:
                i = by_index[index]
                overlapping[index] = _union_overlaps(vertices[i], sizes[i], tol(vertices[i]))
            if not overlapping[index]:
                continue  # one layer: its own distance is exact
        done = [k for k, s in enumerate(shapes) if s.expr == key]
        if done:
            shapes[done[0]] = shapes[done[0]]._replace(levels=(*shapes[done[0]].levels, lv.name))
            continue
        layers = tuple(sorted(_expr_layers(key) & nonempty))
        reason = ""
        if _inner_grows(lv.expr):
            reason = "it sizes a layer or an operand below the expression's root"
        else:
            slots = {index: s for s, index in enumerate(layers)}
            pool = [(vertices[by_index[index]], sizes[by_index[index]]) for index in layers]
            eps = max(tol(v) for v, _ in pool)
            try:
                recipes = derive_polygons(pool, _slot_expr(key, slots), eps)
            except ValueError as error:
                reason = str(error)
            else:
                # the recipe must reproduce the boolean (up to slivers of the tolerance)
                regions = {index: _shapely_polygons(*p) for index, p in zip(layers, pool, strict=True)}
                points = np.concatenate([v for v, _ in pool])
                lo, hi = points.min(axis=0) - 1.0, points.max(axis=0) + 1.0
                universe = shapely.box(*lo, *hi)
                drawn = _shape_region(recipes, points)
                if recipes[0].negate:
                    drawn = universe.difference(drawn)
                error = drawn.symmetric_difference(_expr_region(key, regions, universe))
                if not error.buffer(-2 * eps).is_empty:
                    reason = f"its polygons differ from the boolean by {error.area:.3g} um^2"
        if reason:
            warnings.warn(
                f"StackDevice {name!r}: level {lv.name!r} is drawn as the boolean of offset layers ({reason}); "
                "it may draw slivers where edges of its layers coincide",
                StackDiagnosticWarning,
                stacklevel=3,
            )
            continue
        shapes.append(_DerivedShape(expr=key, levels=(lv.name,), layers=layers, recipes=recipes, tol=eps))
    return tuple(shapes)


# ---------------------------------------------------------------- roughness


@dataclasses.dataclass(frozen=True)
class StackOptions:
    """Geometry features of a :class:`StackDevice` that cost time, all on by default.

    Each field is the StackDevice field of the same name; pass the options to
    :meth:`StackDevice.from_layer_stack` (``options=``), or the fields one by
    one as before. The defaults are the most accurate settings. The cost and
    accuracy benefit of each are in docs/fdtdx.md ("Cost of the geometry
    features"). ``StackOptions.fast()`` turns off the ones that are not needed
    for a quick look at a device.
    """

    #: "subpixel": Kottke averaging along the normal of each voxel's own model
    #: (needs 3-component arrays); "inverse": the mean of 1 / eps (fdtdx's convention)
    averaging: Averaging = "subpixel"
    #: normal of "subpixel" averaging: "voxel" (each voxel's own model) or "gradient"
    #: (finite differences of the mean eps; cheaper, up to 15-33 degrees off at slanted walls)
    subpixel_normal: Literal["voxel", "gradient"] = "voxel"
    #: sample each E component's materials at its own Yee position
    yee_staggered: bool = True
    #: split lines of the exact fill at polygon corners and edge crossings (0: off)
    corner_lines: int = 2
    #: lateral sub-samples per voxel and axis (cost ~ s^2)
    xy_subsamples: int = 1
    #: "exact" or "composite" drawing of derived levels (see ``StackDevice.derived_shapes``)
    derived_shapes: Literal["exact", "composite"] = "exact"
    #: warn about unresolved parts, gaps and shrinking overlaps when placed
    diagnostics: bool = True

    @classmethod
    def fast(cls) -> "StackOptions":
        """Every costly feature off: "inverse" averaging, no Yee staggering, no corner model or diagnostics."""
        return cls(averaging="inverse", yee_staggered=False, corner_lines=0, diagnostics=False)


@autoinit
class TopRoughness(TreeClass):
    """Gaussian roughness of the top surface of one level.

    Only the top surface of ``level`` is displaced. Where it dips below its
    nominal height the gap is filled by the surrounding layers: levels deposited
    on that surface (nominal bottom == the level's nominal top) follow it down
    when ``fill_above`` is set, and any other level covering that height (e.g. a
    cladding spanning it) fills it by ``mesh_order``. Where it rises, it
    displaces the material above.

    Attributes:
        level: name of the level whose top surface is rough.
        rms: standard deviation of the height offset (um).
        correlation_length: correlation length (um) of the Gaussian autocorrelation.
        fill_above: levels whose bottom sits on this surface follow it (no voids,
            no overlaps).
    """

    #: name of the level whose top surface is rough
    level: str = frozen_field()
    #: standard deviation of the height offset (um)
    rms: float = field()
    #: correlation length (um) of the Gaussian autocorrelation
    correlation_length: float = frozen_field(default=0.0)
    #: levels whose bottom sits on this surface follow it (no voids, no overlaps)
    fill_above: bool = frozen_field(default=True)

    def sample(self, shape: tuple[int, int], res_um: float, key: Array) -> Array:
        """Samples one rough surface.

        Args:
            shape: (nx, ny) of the device grid.
            res_um: grid resolution in um.
            key: PRNG key.

        Returns:
            (nx, ny) height offsets in cells.
        """
        f = gaussian_random_field(shape, self.correlation_length / res_um, key)
        return (self.rms / res_um) * f


# ---------------------------------------------------------------- device


@autoinit
class StackDevice(StaticMultiMaterialObject):
    """A component with its LayerStack as one differentiable fdtdx object.

    Build it with :meth:`from_layer_stack`, get the matching volume with
    :meth:`get_simulation_volume` and sources / detectors on the component's
    ports with the ``create_*_at`` methods. Merge :func:`init_stack_params` into
    the fdtdx params, update the vertices with :meth:`pack` and apply the device
    with :func:`apply_stack`. Parameters (all in stack units, um / degrees)::

        params[name] = {
            "vertices/<layer>": (E, 2),  # one per source layer with polygons
            "zmin/<level>": (),
            "thickness/<level>": (),
            "sidewall_angle/<level>": (),
        }

    The device box is overwritten on every :func:`apply_stack` (permittivity,
    conductivity and, in dispersive simulations, all pole coefficients):
    everything not covered by a level is ``background``. fdtdx Devices are
    applied afterwards and win where they overlap.

    The polygon topology (T-junctions, seams, polygon order and orientation,
    where the port extensions meet the polygons) is taken from the vertices the
    device is built with, within ``topology_tol``, and indexes any later
    vertices. :meth:`pack` checks that it still fits (:meth:`check_topology`)
    and :meth:`retopologize` takes it from new vertices.

    Attributes:
        levels: compiled levels in the box.
        source_layers: (parameter key, layer index) per source layer with polygons.
        source_vertices: initial packed vertices (um) per source layer; the
            static topology is computed from them.
        source_sizes: vertices per polygon, per source layer.
        background: material where no level is.
        center: gds point (um) at the xy center of the box.
        z_center: stack height (um) at the z center of the box.
        source_extensions: per source layer, (port, length, face end a,
            face end b) of the port extensions it ends with.
        extended_ports: (port, length, layers it may continue) per extended port.
        empty_layers: stack layers the levels in the box read that had no
            polygons, with those levels.
        topology_tol: distance (um) within which the polygon topology is found
            (None: 2 database units of the layout), plus float32 round-off.
        ports: the component's ports.
        roughness: top-surface roughness per level.
        noise_seed: roughness seed when no ``noise_key`` is passed.
        averaging: "subpixel" (Kottke, along the normal of each voxel's own
            model) or "inverse".
        subpixel_normal: "voxel" (the normal of each voxel's own model) or
            "gradient" (finite differences of the mean eps; cheaper, less exact).
        yee_staggered: sample each E component at its own Yee position.
        level_margin: how far (um) outside the box a level may start and still
            be drawn, so that zmin / thickness / roughness can move it in.
        edge_width: lateral width (cells) of the voxel square used for areas.
        xy_subsamples: lateral sub-samples per voxel and axis.
        edge_chunk: edges per step of the distance computation.
        corner_lines: split lines of the exact fill in columns near polygon
            corners and crossings of layer edges (0: one half-plane per level
            everywhere).
        corner_columns: columns per lateral sample grid painted with the corner
            model (None: sized when the device is placed; 0: none).
        diagnostics: warn when the device is placed about parts of a level
            and gaps narrower than the grid resolves, and overlapping polygons
            a level shrinks (:meth:`diagnose`).
        derived_shapes: "exact" draws each derived level as the offset of the
            exact polygons of its layer expression (built from
            ``source_vertices``, see ``derived_layers``); "composite" draws the
            min / max of its layers' offset distances.
        derived_layers: the exact shapes of the derived levels.
    """

    #: compiled levels (see :meth:`from_layer_stack`)
    levels: tuple[_Level, ...] = frozen_field()
    #: (param key, layer index) per source layer with polygons
    source_layers: tuple[tuple[str, int], ...] = frozen_field()
    #: initial packed vertices (um) per source layer; the static topology
    #: (T-junctions, seams, edge directions) is computed from them
    source_vertices: tuple[np.ndarray, ...] = frozen_field()
    #: vertices per polygon, per source layer
    source_sizes: tuple[tuple[int, ...], ...] = frozen_field()
    #: material filling everything that no level covers
    background: str = frozen_field()
    #: gds point (um) at the xy center of the device
    center: tuple[float, float] = frozen_field(default=(0.0, 0.0))
    #: stack height (um) at the z center of the device
    z_center: float = frozen_field(default=0.0)
    #: per source layer: (port, length, a, b) of each port extension, whose 4-vertex
    #: polygons end the layer's vertices; a / b are its corners on the port face:
    #: (i, i) copies the layer's vertex i, (i, j) is where edge i -> j crosses the
    #: face, (-1, -1) is the port's own corner
    source_extensions: tuple[tuple[tuple[str, float, _FaceEnd, _FaceEnd], ...], ...] = frozen_field(
        default=()
    )
    #: (port, length, layer indices it may continue) per port in ``extend_ports``,
    #: to find the extensions again from new polygons (empty: the layers that
    #: have extensions in ``source_extensions``)
    extended_ports: tuple[tuple[str, float, tuple[int, ...]], ...] = frozen_field(default=())
    #: (layer index, (level, nominal bottom, top) of each level in the box that reads
    #: it) per stack layer without polygons when the device was built: the levels
    #: read it as empty, and those that need it are not drawn, so polygons on it
    #: need a new device (:meth:`check_topology` reports them)
    empty_layers: tuple[tuple[int, tuple[tuple[str, float, float], ...]], ...] = frozen_field(default=())
    #: distance (um) within which a vertex lies on an edge of another polygon (a
    #: T-junction, split so that the partly shared edge is a seam), two vertices
    #: coincide (seams of abutting polygons) and a vertex is on a port face, when
    #: the polygon topology is found, by the device and by check_topology and
    #: retopologize alike; float32 round-off (2 epsilons times the largest
    #: coordinate) is added. None: 2 database units of the layout (gf.kcl.dbu),
    #: since snapping to its grid moves a vertex on a slanted edge up to sqrt(2)
    #: units off it. A vertex this close to an edge bends the edge through it;
    #: abutting edges whose ends are this close but apart are a seam too, and a
    #: voxel centered in the gap between them, if any, is filled as if far outside
    topology_tol: float | None = frozen_field(default=None)
    #: the component's ports, for the ``create_*_at`` methods
    ports: tuple[_Port, ...] = frozen_field(default=())
    #: top-surface roughness per level, sampled with ``noise_key``
    roughness: Sequence["TopRoughness"] = field(default=())
    #: seed used for the roughness when no ``noise_key`` is passed
    noise_seed: int = frozen_field(default=0)
    #: "subpixel" (Kottke, along the interface normal of each voxel's own model:
    #: the flux of its interfaces and its vertical profile) or "inverse" (fdtdx convention)
    averaging: Averaging = frozen_field(default="subpixel")
    #: normal of "subpixel" averaging: "voxel" (from each voxel's own sub-cell model:
    #: exact for straight interfaces, also slanted, thin and sloped ones) or
    #: "gradient" (from finite differences of the voxel-mean eps, as before the
    #: voxel model: errors up to 15-33 degrees at slanted walls, but cheaper)
    subpixel_normal: Literal["voxel", "gradient"] = frozen_field(default="voxel")
    #: with 3-component permittivity arrays (e.g. ``averaging="subpixel"``),
    #: sample each E component's materials at its own Yee position (half a cell
    #: apart); otherwise all components use the voxel center, a first-order error
    yee_staggered: bool = frozen_field(default=True)
    #: levels whose nominal span is within this distance (um) of the box are
    #: drawn, so that zmin / thickness / roughness can move them into it; levels
    #: in the box are drawn over its whole height and may move anywhere in it
    level_margin: float = frozen_field(default=0.0)
    #: lateral width of the voxel square used for the areas, in cells
    edge_width: float = frozen_field(default=1.0)
    #: in-plane sub-samples per voxel and axis: the lateral fill is averaged over
    #: s x s sub-squares, which makes it closer to the exact area fraction at
    #: polygon corners and narrow tips (cost ~ s^2); the subpixel normal comes
    #: from the moments of all of them together
    xy_subsamples: int = frozen_field(default=1)
    #: edges per step in the signed distance computation
    edge_chunk: int = frozen_field(default=64)
    #: in voxel columns near polygon corners and crossings of layer edges, paint
    #: the exact local geometry with up to this many split lines; 0 means one
    #: half-plane per level everywhere, as before (bit-identical). Cost grows
    #: about as 2^M + 1
    corner_lines: int = frozen_field(default=2)
    #: static number of columns per lateral sample grid painted with the corner
    #: model; None sizes it in place_on_grid. Candidates beyond it (raster
    #: order) keep the half-plane result
    corner_columns: int | None = frozen_field(default=None)
    #: when the device is placed, warn (StackDiagnosticWarning) about parts of a
    #: level and gaps narrower than the grid resolves, and about overlapping
    #: polygons a level shrinks; :meth:`diagnose` returns them as data
    diagnostics: bool = frozen_field(default=True)
    #: "exact": a derived level (and / or / not / xor of layers, or one layer whose
    #: polygons overlap) is drawn from the exact polygons of its boolean, built
    #: once as copies of layer vertices and crossings of layer edges, so its
    #: offset is the offset of that shape; "composite": from the min / max of its
    #: layers' distances, as before (slivers where edges of its layers coincide,
    #: sharp grown corners of an and, slits where an or of overlapping parts shrinks)
    derived_shapes: Literal["exact", "composite"] = frozen_field(default="exact")
    #: the exact shapes of the derived levels (with ``derived_shapes="exact"``),
    #: built from ``source_vertices``; part of the frozen topology (:meth:`check_topology`)
    derived_layers: tuple[_DerivedShape, ...] = frozen_field(default=())

    # ------------------------------------------------------------ construction

    @classmethod
    @timed("from_layer_stack")
    def from_layer_stack(
        cls,
        component: Any,
        layer_stack: Any,
        materials: dict[str, fdtdx.Material],
        background: str,
        *,
        included_layers: Sequence[str],
        extend_ports: Sequence[tuple[str, float] | tuple[str, float, Sequence[Any]]] = (),
        boundary_offset: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        levels: Sequence[str] | None = None,
        fill_layers: Sequence[Any] | None = None,
        options: StackOptions | None = None,
        **kwargs: Any,
    ) -> Self:
        """Builds the device and its box from a component and a LayerStack.

        ``extend_ports`` first adds a straight waveguide to each listed port. The
        box then starts as the xy bounding box of the polygons on the stack's
        layers (extensions included; all polygons if the levels only fill the
        plane) and the z range of ``included_layers``, and ``boundary_offset``
        moves its faces without changing the geometry. Every level that overlaps
        the final box (or comes within ``level_margin`` of it) is drawn, also
        levels outside ``included_layers``, unless no polygon is on its layers.
        Levels within 0.1 um more need a material too; those that do not reach
        the box once fdtdx has placed it on its grid are dropped then.

        Args:
            component: the component; polygons and ports are read with gradients
                stopped.
            layer_stack: a :class:`~gdsfactory.technology.LayerStack`.
            materials: fdtdx materials by ``LayerLevel.material``; needed for
                ``background`` and every level that is drawn.
            background: material (key of ``materials``) where no level is.
            included_layers: level names; the lowest bottom and the highest top
                among them are the initial z faces of the box.
            extend_ports: ``((port, length), ...)`` in um, or
                ``((port, length, layers), ...)``. Each port's cross-section is
                continued straight out of the component along the port's
                orientation (opposite to the direction light is injected through
                the port): every stack layer (or only ``layers``) whose polygons
                reach the port face, over a stretch that overlaps the port's
                width, gets a rectangle as wide as that stretch, so a rib
                waveguide continues with its slab and an etch over the port
                continues too. On the port's own layer, a port without polygons
                at its face gets a rectangle of the port's width. The extensions
                are polygons like the others: every level draws them with its
                thickness, sidewall angle and bias, and they follow the port and
                the polygons' edges when the component's parameters change.
            boundary_offset: (x+, x-, y+, y-, z+, z-) in um. Moves each face of the
                box along its axis, positive towards +x, +y or +z for both faces
                of an axis (so (1, -1, ...) grows the box by 1 um on each side
                in x). The geometry does not change: the box shows more or less
                of the same layer stack.
            levels: levels that may be drawn (default: every level with a
                material and nonzero thickness).
            fill_layers: layers that cover the whole plane (default: WAFER).
            options: the geometry features (:class:`StackOptions`; default: all
                on, the most accurate). Its fields are StackDevice fields, which
                can also be passed one by one in ``kwargs``, but not both.
            **kwargs: other StackDevice fields (``name``, ``roughness``,
                ``averaging``, ...). The box comes from the arguments above, so
                ``partial_real_shape``, ``partial_grid_shape``, ``center`` and
                ``z_center`` are not accepted.

        Returns:
            The device; its ``partial_real_shape`` is the box (fdtdx rounds it
            to whole cells around the box center).

        Raises:
            KeyError: unknown level or port names, or a level in the box (or the
                background) whose material is not in ``materials``.
            ValueError: invalid offsets or extensions (no orientation or width,
                length <= 0, a port layer no level draws), an empty box, a
                negative ``topology_tol``, or a component without polygons.
            TypeError: a box field passed in ``kwargs``.
        """
        from gdsfactory.pdk import get_layer

        if options is not None:
            given = sorted(set(kwargs) & {f.name for f in dataclasses.fields(StackOptions)})
            if given:
                raise TypeError(f"Pass {given} either in options or as arguments, not both")
            kwargs.update(dataclasses.asdict(options))
        for key in ("partial_real_shape", "partial_grid_shape", "center", "z_center"):
            if key in kwargs:
                raise TypeError(
                    f"from_layer_stack sets {key!r} from included_layers, extend_ports and boundary_offset"
                )
        offset = tuple(float(o) for o in boundary_offset)
        if len(offset) != 6:
            raise ValueError(f"boundary_offset must be 6 lengths (x+, x-, y+, y-, z+, z-), got {boundary_offset}")
        if isinstance(included_layers, str):
            included_layers = [included_layers]
        if not included_layers:
            raise ValueError("included_layers must name at least one level")
        unknown = [n for n in (*included_layers, *(levels or ())) if n not in layer_stack.layers]
        if unknown:
            raise KeyError(f"Unknown levels {unknown}; the stack has {sorted(layer_stack.layers)}")
        if background not in materials:
            raise KeyError(f"background {background!r} is not in materials")
        if fill_layers is None:
            fill_layers = [layer_stack_wafer_layer()]
        fills = {int(get_layer(lay)) for lay in fill_layers if lay is not None}

        # z faces: the included levels, then the offsets
        spans = [_span(layer_stack.layers[n]) for n in included_layers]
        z0 = min(lo for lo, _ in spans) + offset[5]
        z1 = max(hi for _, hi in spans) + offset[4]

        names = list(levels) if levels is not None else list(layer_stack.layers)
        candidates = []
        for name in names:
            level = layer_stack.layers[name]
            if level.material is None or level.thickness == 0:
                if levels is not None:
                    raise ValueError(f"Level {name!r} has no material or zero thickness")
                continue
            candidates.append((name, level, _compile_level(level, fills)))
        stack_layers = set().union(*(_expr_layers(expr) for *_, expr in candidates))
        lap("compile_levels")

        ports = [_read_port(port) for port in component.ports]

        # static polygons per layer (um), then the port extensions on their layers
        polygons = {
            index: [np.asarray(to_numpy(p), dtype=float).reshape(-1, 2) for p in polys]
            for index, polys in component.get_polygons_points().items()
        }
        lap("polygons")
        by_name = {p.name: p for p in ports}
        extended = []
        for name, length, *only in extend_ports:
            if name not in by_name:
                raise KeyError(f"Unknown port {name!r} in extend_ports; the component has {sorted(by_name)}")
            p = by_name[name]
            if p.orientation is None or p.width <= 0 or not float(length) > 0:
                raise ValueError(f"Cannot extend port {name!r}: needs an orientation, a width and a length > 0")
            if p.layer not in stack_layers:
                raise ValueError(f"Port {name!r} is on layer {p.layer}, which no level of the stack draws")
            layers = sorted(stack_layers)
            if only:
                layers = [index for index in layers if index in {int(get_layer(lay)) for lay in only[0]}]
            extended.append((str(name), float(length), tuple(layers)))
        topology_tol = kwargs.get("topology_tol")
        if topology_tol is not None and not float(topology_tol) >= 0:
            raise ValueError(f"topology_tol must be >= 0, got {topology_tol}")
        extensions, rectangles = _port_extensions(polygons, by_name, extended, topology_tol)
        lap("port_extensions")

        # xy faces: bounding box of the polygons on the stack's layers (any z;
        # all polygons if the levels only fill), then the offsets
        points = [q for index in stack_layers for q in (*polygons.get(index, []), *rectangles.get(index, []))]
        if not points:
            points = [q for polys in polygons.values() for q in polys]
        if not points:
            raise ValueError("The component has no polygons")
        points = np.concatenate(points)
        (bx0, by0), (bx1, by1) = points.min(axis=0), points.max(axis=0)
        x0, x1 = bx0 + offset[1], bx1 + offset[0]
        y0, y1 = by0 + offset[3], by1 + offset[2]
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ValueError(f"Empty box x [{x0}, {x1}], y [{y0}, {y1}], z [{z0}, {z1}] um")

        # levels just outside the box count too (see _SELECT_SLACK); the exact
        # selection needs the grid and happens in place_on_grid
        level_margin = float(kwargs.get("level_margin", 0.0))
        if level_margin < 0:
            raise ValueError(f"level_margin must be >= 0, got {level_margin}")
        margin = level_margin + _SELECT_SLACK
        nonempty = {index for index, polys in polygons.items() if polys} | set(rectangles)
        compiled = []
        empty: dict[int, list[tuple[str, float, float]]] = {}
        for name, level, expr in candidates:
            lo, hi = _span(level)
            if hi <= z0 - margin or lo >= z1 + margin:
                continue  # outside the box
            for index in sorted(_expr_layers(expr) - nonempty):
                empty.setdefault(index, []).append((name, float(lo), float(hi)))
            if not _expr_can_draw(expr, nonempty):
                continue  # no polygons on its layers (e.g. a metal level of a passive component)
            if level.material not in materials:
                raise KeyError(
                    f"Level {name!r} (z {lo:g} to {hi:g} um) is in the box but its material "
                    f"{level.material!r} is not in materials {sorted(materials)}; pass it, or "
                    "restrict levels=[...]"
                )
            bias = level.bias
            if isinstance(bias, tuple):
                if bias[0] != bias[1]:
                    raise NotImplementedError(f"Anisotropic bias {bias} is not supported")
                bias = bias[0]
            z_to_bias = None
            if level.z_to_bias is not None:
                z_to_bias = tuple(tuple(float(v) for v in vs) for vs in level.z_to_bias)
            compiled.append(
                _Level(
                    name=name,
                    expr=expr,
                    zmin=float(level.zmin),
                    thickness=float(level.thickness),
                    sidewall_angle=float(level.sidewall_angle),
                    width_to_z=float(level.width_to_z),
                    bias=float(bias or 0.0),
                    z_to_bias=z_to_bias,
                    mesh_order=int(level.mesh_order),
                    material=level.material,
                )
            )
        level_names = {lv.name for lv in compiled}
        for r in kwargs.get("roughness", ()):
            if r.level not in level_names:
                raise KeyError(f"roughness level {r.level!r} is not among {sorted(level_names)}")

        layer_ids = sorted(set().union(*(_expr_layers(lv.expr) for lv in compiled)))
        # extensions on layers no level in the box draws are dropped with them,
        # but the port's own layer must be drawn
        missing = [n for n, *_ in extend_ports if by_name[n].layer not in layer_ids]
        if missing:
            raise ValueError(f"No level in the box draws the layer of the extended ports {missing}")
        sources, verts, sizes, exts = [], [], [], []
        for index in layer_ids:
            polys = [*polygons.get(index, []), *rectangles.get(index, [])]
            if not polys:
                continue  # empty layer: constant "outside"
            sources.append((_layer_key(index), index))
            verts.append(np.concatenate(polys))
            sizes.append(tuple(len(q) for q in polys))
            exts.append(tuple(extensions.get(index, ())))

        used = {lv.material for lv in compiled} | {background}
        mats = {k: v for k, v in materials.items() if k in used}
        for k, m in mats.items():
            if m.is_magnetic or m.is_magnetically_conductive:
                raise NotImplementedError(f"Magnetic material {k!r} is not supported in a StackDevice")

        lap("select_levels")
        device = cls(
            materials=align_material_poles(mats),
            levels=tuple(compiled),
            source_layers=tuple(sources),
            source_vertices=tuple(verts),
            source_sizes=tuple(sizes),
            source_extensions=tuple(exts),
            extended_ports=tuple(extended),
            empty_layers=tuple((index, tuple(levels)) for index, levels in sorted(empty.items())),
            background=background,
            partial_real_shape=((x1 - x0) * 1e-6, (y1 - y0) * 1e-6, (z1 - z0) * 1e-6),
            center=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
            z_center=0.5 * (z0 + z1),
            ports=tuple(ports),
            **kwargs,
        )
        return device._with_derived_layers()

    @property
    def options(self) -> StackOptions:
        """The device's geometry features as a :class:`StackOptions`."""
        return StackOptions(**{f.name: getattr(self, f.name) for f in dataclasses.fields(StackOptions)})

    @timed("derived_shapes")
    def _with_derived_layers(self) -> Self:
        """The device with ``derived_layers`` built from its ``source_vertices`` (see ``derived_shapes``)."""
        shapes: tuple[_DerivedShape, ...] = ()
        if self.derived_shapes == "exact":
            shapes = _derived_shapes(
                self.levels, self.source_layers, self.source_vertices, self.source_sizes, self._topology_tol,
                self.name,
            )
        elif self.derived_shapes != "composite":
            raise ValueError(f"derived_shapes must be 'exact' or 'composite', got {self.derived_shapes!r}")
        return self.aset("derived_layers", shapes)

    def _slots(self) -> list[tuple[str, int]]:
        """(key, layer index) per layer slot: the source layers, then the derived shapes' nonempty parts."""
        return [*self.source_layers, *(
            (f"derived/{'+'.join(d.levels)}/{j}", _part_index(k, j))
            for k, d in enumerate(self.derived_layers) for j, r in enumerate(d.recipes) if len(r.kind)
        )]

    def _drawn_expr(self, level: _Level) -> tuple:
        """The expression a level is drawn with: the union of its derived shape's parts (offset by its root
        sizing), or its own.
        """
        for k, d in enumerate(self.derived_layers):
            if level.name in d.levels:
                grow = _root_grow(level.expr)
                parts = [("layer", _part_index(k, j), 0.0) for j, r in enumerate(d.recipes) if len(r.kind)]
                if not parts:
                    return ("fill", grow) if d.recipes[0].negate else ("layer", _NO_LAYER, 0.0)
                union = parts[0]
                for part in parts[1:]:
                    union = ("or", union, part, 0.0)
                if d.recipes[0].negate:
                    return ("not", ("fill", 0.0), union, grow)
                return _with_root_grow(union, grow)
        return level.expr

    # ------------------------------------------------------------ parameters

    def init_params(self, key: Array | None = None) -> dict[str, Array]:
        """Initial parameters: the component's vertices and the stack's thicknesses and angles.

        Args:
            key: unused (the parameters are deterministic).

        Returns:
            ``{"vertices/<layer>": (E, 2), "zmin/<level>": (), "thickness/<level>": (),
            "sidewall_angle/<level>": ()}``.
        """
        del key
        p: dict[str, Array] = {}
        for (k, _), v in zip(self.source_layers, self.source_vertices, strict=True):
            p[k] = jnp.asarray(v, dtype=jnp.float32)
        for lv in self.levels:
            p[f"zmin/{lv.name}"] = jnp.asarray(lv.zmin, dtype=jnp.float32)
            p[f"thickness/{lv.name}"] = jnp.asarray(lv.thickness, dtype=jnp.float32)
            p[f"sidewall_angle/{lv.name}"] = jnp.asarray(lv.sidewall_angle, dtype=jnp.float32)
        return p

    @timed("pack")
    def pack(self, component: Any, check: bool = True) -> dict[str, Array]:
        """Vertex parameters of a (possibly traced) component, port extensions included.

        Args:
            component: a component with the polygon topology of the one the
                device was built from (the same number of polygons and vertices
                per polygon on every source layer, see :meth:`check_topology`)
                and the extended ports.
            check: if the component is concrete (not traced), check that its
                vertices still fit the device's topology
                (:meth:`check_topology`). Traced components are never checked.

        Returns:
            ``{"vertices/<layer>": (E, 2)}``, differentiable with respect to the
            parameters the component was built from; the extensions follow their
            ports and the polygon edges they continue. Which vertices and edges
            an extension continues is fixed when the device is built.

        Raises:
            ValueError: if the number of polygons or vertices changed or, with
                ``check``, the topology no longer fits (:meth:`retopologize`
                gives a device for the new vertices) or the component has
                polygons on a stack layer the device was built without.
        """
        out = self._pack(component)
        if check and not has_tracers(out):
            self._raise_on_changes(out, component)
        return out

    @timed("pack_vertices")
    def _pack(self, component: Any) -> dict[str, Array]:
        """:meth:`pack` without the topology check."""
        polygons = component.get_polygons_points()
        out = {}
        extensions = self.source_extensions or tuple(() for _ in self.source_layers)
        for (k, index), sizes, exts in zip(self.source_layers, self.source_sizes, extensions, strict=True):
            polys = polygons.get(index, [])
            own = sizes[: len(sizes) - len(exts)]
            s = tuple(int(np.shape(q)[0]) for q in polys)
            if s != own:
                raise ValueError(
                    f"Polygon topology of {k} changed: device has {len(own)} polygons with sizes "
                    f"{own}, got {len(s)} with {s}. Build a device for the new polygons with "
                    "device.retopologize(component), or StackDevice.from_layer_stack."
                )
            v = pack_polygons(polys)[0] if polys else jnp.zeros((0, 2))
            rects = []
            for name, length, a, b in exts:
                # face corners on the polygons' own vertices and edges: an exact seam
                port = component.ports[name]
                center = jnp.stack([jnp.asarray(c, dtype=v.dtype) for c in port.center])
                width = jnp.asarray(port.width, dtype=v.dtype)
                orientation = jnp.asarray(port.orientation, dtype=v.dtype)
                rects.append(_extension_rect((a, b), v, center, width, orientation, length, jnp).astype(v.dtype))
            out[k] = jnp.concatenate([v, *rects]) if rects else v
        return out

    # ------------------------------------------------------------ topology

    @timed("check_topology")
    def check_topology(self, vertices: Any) -> None:
        """Checks that new vertices still fit the polygon topology the device was built with.

        Once, from the vertices it is built with (``source_vertices``), the
        device finds which polygon vertices lie inside edges of other polygons
        (T-junctions, split so that partly shared edges become seams), which
        edges are seams (abutting polygons, hole cut lines) and are left out of
        the distance, which way each polygon runs (for the corner model) and,
        for each port extension, where its layer meets the port face. Later
        vertices are indexed with all that, by polygon and vertex, so it must
        still fit them. Per source layer, this finds:

        - polygons that look reordered among the polygons of their size: one now
          lies where another one was, much closer than to its own place, in a
          cycle of such matches (identical polygons moving far, as in a
          stretched grating, do not count);
        - T-junctions that came apart (e.g. a taper moved off the side of an
          MMI, whose side would keep a bump through the taper's corners) and
          new ones;
        - seams that opened (e.g. abutting polygons moved apart: the gap would
          stay filled, and points inside it get wrong distances) and new ones;
        - polygons that now run the other way round (with ``corner_lines``);
        - per exact derived shape (``derived_layers``): where the polygons its
          recipe draws from the new vertices differ from the exact boolean of
          them by more than the tolerance (a part appeared or vanished, a seam
          opened), and crossings that left one of their two edges;
        - with a component: port extensions whose layer now meets the port face
          elsewhere (a vertex left the face, an edge no longer crosses it, or
          the layer reaches the face in more, fewer or other places), and
          polygons on a stack layer that had none when the device was built
          (``empty_layers``): the levels read it as empty and those that need
          it are not drawn, so only :meth:`from_layer_stack` builds a device
          for them.

        Relations are found with the device's topology tolerance
        (``topology_tol``, 2 database units by default, plus float32 round-off;
        computed from the new vertices for them), as :meth:`retopologize` and
        :meth:`from_layer_stack` find them: what held for the device's vertices
        counts as kept while it holds within it, and what holds within it now
        but did not is new. So layout snapping, round-off and moves that keep
        the topology (a taper sliding along an MMI side, abutting polygons
        moving together) pass, and when the check passes, a device built for
        the new vertices has the same T-junctions, seams and extensions: the
        device's edges bend through the new vertices by at most the tolerance.
        Traced vertices (inside ``jax.grad`` or ``jax.jit``) are not checked,
        so this costs nothing in a differentiated objective.

        Args:
            vertices: a component, packed like :meth:`pack` does (its ports
                locate the port faces), or device parameters with
                ``vertices/<layer>`` entries (um; other entries are ignored),
                such as the output of :meth:`pack` or ``params[name]``. Port
                extensions and new layers are only checked with a component.

        Raises:
            ValueError: if the topology no longer fits, naming each layer and what
                changed in it (:meth:`retopologize` gives a device for the new
                vertices), if the component has polygons on a layer the device
                was built without, or if the parameters have none of the
                device's layers, or another number of vertices.
        """
        component = None
        if hasattr(vertices, "get_polygons_points"):
            component, vertices = vertices, self._pack(vertices)
        else:
            keys = [k for k, _ in self.source_layers]
            vertices = {k: v for k, v in vertices.items() if k in keys}
            if not vertices:
                raise ValueError(
                    f"No vertices of StackDevice {self.name!r} in the parameters: pass params[{self.name!r}], "
                    f"the output of pack(), or a component (layers {keys})"
                )
        if not has_tracers(vertices):
            self._raise_on_changes(vertices, component)

    @timed("retopologize")
    def retopologize(self, vertices: Any) -> Self:
        """A copy of the device with the polygon topology of new vertices.

        When :meth:`pack` or :meth:`check_topology` reports that new vertices no
        longer fit the device's topology, this gives the device built for them:
        its ``source_vertices`` (and :meth:`init_params`) are the new vertices,
        so their T-junctions, seams, polygon order and orientations are found
        again, within the same tolerance as the check uses (``topology_tol``):
        whatever the check counts as kept, this keeps; the exact shapes of the
        derived levels (``derived_layers``) are built again from them too.
        Everything else is kept:
        the box, the levels and source layers, the materials, the other fields
        and the placement of a placed device, whose budget of corner columns
        only grows if the new vertices need more. Swap it into the placed
        objects and refresh its parameters, e.g. ``objects = objects.copy();
        objects[name] = objects[name].retopologize(c)`` and ``params[name] =
        {**params[name], **objects[name].pack(c)}``.

        Args:
            vertices: a component, read like :meth:`from_layer_stack` does: its
                polygons may also differ in number and size, the ports are read
                again and the port extensions are found again from its polygons
                and ports, on the layers they may continue. Or device parameters
                with every ``vertices/<layer>`` entry (um) and the device's
                number of vertices per layer: the polygon sizes and the
                extensions' face ends are kept (use a component when those
                change).

        Returns:
            The new device.

        Raises:
            TypeError: if the vertices are traced (inside ``jax.grad`` or
                ``jax.jit``).
            KeyError: if an extended port is not among the component's ports, or
                the parameters lack a layer's vertices.
            ValueError: if a source layer has no polygons left, the component
                has polygons on a stack layer the device was built without
                (``empty_layers``: only :meth:`from_layer_stack` adds the levels
                and source layers for them), or the parameters have another
                number of vertices.
        """
        if hasattr(vertices, "get_polygons_points"):
            component = vertices
            raw = component.get_polygons_points()
            if has_tracers(raw) or has_tracers([(p.center, p.width, p.orientation) for p in component.ports]):
                raise TypeError("retopologize needs concrete vertices; call it outside jax.grad and jax.jit")
            new = self._new_layers(raw)
            if new:
                raise ValueError(
                    f"StackDevice {self.name!r} cannot take the component's polygons on layers it was built "
                    "without:\n" + "\n".join(f"  - {p}" for p in new)
                    + "\nBuild a new device for the component with StackDevice.from_layer_stack."
                )
            polygons = {
                index: [np.asarray(to_numpy(p), dtype=float).reshape(-1, 2) for p in polys]
                for index, polys in raw.items()
            }
            ports = tuple(_read_port(port) for port in component.ports)
            by_name = {p.name: p for p in ports}
            extended = self._extended_ports()
            missing = [name for name, *_ in extended if name not in by_name]
            if missing:
                raise KeyError(f"The extended ports {missing} are not among the component's ports {sorted(by_name)}")
            drawn = {index for _, index in self.source_layers}
            extensions, rectangles = _port_extensions(
                polygons,
                by_name,
                [(name, length, tuple(i for i in ls if i in drawn)) for name, length, ls in extended],
                self.topology_tol,
            )
            verts, sizes, exts = [], [], []
            for k, index in self.source_layers:
                polys = [*polygons.get(index, []), *rectangles.get(index, [])]
                if not polys:
                    raise ValueError(
                        f"{k} has no polygons in the component; build a new device with StackDevice.from_layer_stack"
                    )
                verts.append(np.concatenate(polys))
                sizes.append(tuple(len(q) for q in polys))
                exts.append(tuple(extensions.get(index, ())))
            out = self.aset("source_vertices", tuple(verts))
            out = out.aset("source_sizes", tuple(sizes))
            out = out.aset("source_extensions", tuple(exts))
            out = out.aset("extended_ports", extended)
            out = out.aset("ports", ports)
        else:
            verts = []
            for (k, _), v0 in zip(self.source_layers, self.source_vertices, strict=True):
                if k not in vertices:
                    raise KeyError(f"retopologize needs every layer's vertices; {k!r} is missing")
                if has_tracers(vertices[k]):
                    raise TypeError("retopologize needs concrete vertices; call it outside jax.grad and jax.jit")
                v = np.asarray(vertices[k], dtype=float)
                if v.shape != np.shape(v0):
                    raise ValueError(
                        f"{k} has {v.shape[0]} vertices, the device {len(v0)}; pass a component to change the "
                        "polygons' sizes"
                    )
                verts.append(v)
            out = self.aset("source_vertices", tuple(verts))
        out = out._with_derived_layers()
        if out._grid_slice_tuple != INVALID_SLICE_TUPLE_3D and out.corner_lines > 0 and out.corner_columns:
            budget = out._corner_budget()
            if budget > out.corner_columns:
                out = out.aset("corner_columns", budget)
        return out

    def _extended_ports(self) -> tuple[tuple[str, float, tuple[int, ...]], ...]:
        """``extended_ports``, or for devices built without it, the layers their extensions are on."""
        if self.extended_ports or not self.source_extensions:
            return tuple(self.extended_ports)
        layers: dict[tuple[str, float], list[int]] = {}
        for (_, index), exts in zip(self.source_layers, self.source_extensions, strict=True):
            for name, length, *_ in exts:
                found = layers.setdefault((name, float(length)), [])
                if index not in found:
                    found.append(index)
        return tuple((name, length, tuple(found)) for (name, length), found in layers.items())

    def _topology_tol(self, vertices: Any) -> float:
        """The device's topology tolerance (um) for a layer's vertices (see ``topology_tol``)."""
        return _topology_tol(self.topology_tol, vertices)

    def _new_layers(self, polygons: dict[int, Sequence[Any]]) -> list[str]:
        """Stack layers the device was built without that now have polygons (see ``empty_layers``).

        Args:
            polygons: the component's polygons per layer index.

        Returns:
            One line per such layer, starting with the parameter key it would have.
        """
        drawn = {lv.name for lv in self.levels}
        found = []
        for index, readers in self.empty_layers:
            count = len(polygons.get(index, ()))
            if count:
                levels = ", ".join(
                    f"level {name!r} {'reads it as empty' if name in drawn else 'is not drawn'}" for name, *_ in readers
                )
                found.append(
                    f"{_layer_key(index)}: {count} polygon{'s' if count > 1 else ''} on a layer the device was "
                    f"built without ({levels})"
                )
        return found

    @timed("topology_check")
    def _raise_on_changes(self, vertices: dict[str, Any], component: Any) -> None:
        """Raises a ValueError that lists :meth:`_new_layers` and :meth:`_topology_changes`, if there are any."""
        new = [] if component is None else self._new_layers(component.get_polygons_points())
        problems = new + self._topology_changes(vertices, component)
        if not problems:
            return
        if new:
            fix = (
                "Build a new device for the component with StackDevice.from_layer_stack (retopologize keeps the "
                "levels and source layers)."
            )
        else:
            fix = (
                "Build a device for them with device.retopologize(component) (it keeps the box, the levels and "
                "the placement; then refresh its params), or pass vertices that keep the topology "
                "(pack(component, check=False) skips this check)."
            )
        raise ValueError(
            f"The vertices no longer fit the polygon topology StackDevice {self.name!r} was built with "
            "(lengths in um):\n" + "\n".join(f"  - {p}" for p in problems) + "\n" + fix
        )

    def _topology_changes(self, vertices: dict[str, Any], component: Any) -> list[str]:
        """What no longer fits in concrete vertices, per layer (see :meth:`check_topology`).

        The device's relations are those found within its topology tolerance
        for ``source_vertices``; the new vertices' are those found within the
        tolerance for them, as a device built for them would find them.

        Args:
            vertices: ``{"vertices/<layer>": (E, 2)}`` (um, concrete), port
                extensions included; layers that are missing are skipped.
            component: the component, to check the port extensions, or None.

        Returns:
            One line per change, starting with the layer's parameter key.
        """
        extensions = self.source_extensions or tuple(() for _ in self.source_layers)
        extended = self._extended_ports()
        ports = {}
        if component is not None:
            ports = {name: _read_port(component.ports[name]) for name, *_ in extended}
        problems = []
        for (k, index), v0, sizes, exts in zip(
            self.source_layers, self.source_vertices, self.source_sizes, extensions, strict=True
        ):
            if k not in vertices:
                continue
            v = np.asarray(vertices[k]).astype(float)
            if v.shape != np.shape(v0):
                problems.append(f"{k}: {v.shape[0]} vertices, the device has {len(v0)}")
                continue
            found = []
            own_sizes = sizes[: len(sizes) - len(exts)]
            if ports:
                own = v[: sum(own_sizes)]
                found += self._extension_changes(
                    index, own, own_sizes, exts, extended, ports, self._topology_tol(own)
                )
            if not np.isfinite(v).all():
                found.append(f"vertex {int(np.flatnonzero(~np.isfinite(v).all(axis=1))[0])} is not finite")
            else:
                found += topology_changes(
                    v0, v, sizes, self._topology_tol(v), self._topology_tol(v0), oriented=self.corner_lines > 0
                )
            problems += [f"{k}: {p}" for p in found]
        return problems + self._derived_changes(vertices)

    def _derived_changes(self, vertices: dict[str, Any]) -> list[str]:
        """What no longer fits the derived shapes' recipes in concrete vertices (see :meth:`check_topology`).

        Per shape: a crossing that left one of its two edges (beyond the
        topology tolerance of its ends), and where the recipe's polygons
        differ from the exact boolean of the new vertices by more than the
        tolerance (a part that appeared or vanished, a seam that opened).

        Args:
            vertices: ``{"vertices/<layer>": (E, 2)}`` (um, concrete); shapes
                with a missing layer are skipped.

        Returns:
            One line per change, starting with the levels that draw the shape.
        """
        import shapely

        keys = {index: k for k, index in self.source_layers}
        sizes = {index: n for (_, index), n in zip(self.source_layers, self.source_sizes, strict=True)}
        problems = []
        for d in self.derived_layers:
            if not all(keys[index] in vertices for index in d.layers):
                continue
            pool = [np.asarray(vertices[keys[index]], dtype=float) for index in d.layers]
            if any(v.shape != (sum(sizes[i]), 2) or not np.isfinite(v).all() for v, i in zip(pool, d.layers, strict=True)):
                continue  # reported per layer
            levels = ", ".join(repr(n) for n in d.levels)
            prefix = f"level {levels} (exact shape of {', '.join(keys[i] for i in d.layers)})"
            flat = np.concatenate(pool)
            tol = max(self._topology_tol(v) for v in pool)
            starts = np.cumsum([0, *(len(v) for v in pool)])
            for _, a, b, x in [c for r in d.recipes for c in crossing_changes(r, flat, tol)][:3]:
                layer = int(np.searchsorted(starts, a, side="right")) - 1
                problems.append(
                    f"{prefix}: the crossing of edge {a - starts[layer]}->{b - starts[layer]} of "
                    f"{keys[d.layers[layer]]} with another edge left that edge, now at ({x[0]:.6g}, {x[1]:.6g})"
                )
            lo, hi = flat.min(axis=0) - 1.0, flat.max(axis=0) + 1.0
            universe = shapely.box(*lo, *hi)
            drawn = _shape_region(d.recipes, flat)
            if d.recipes[0].negate:
                drawn = universe.difference(drawn)
            regions = {i: _shapely_polygons(v, sizes[i]) for i, v in zip(d.layers, pool, strict=True)}
            error = drawn.symmetric_difference(_expr_region(d.expr, regions, universe))
            if not error.buffer(-0.5 * tol).is_empty:
                circle = shapely.maximum_inscribed_circle(error.buffer(-0.5 * tol).buffer(0.5 * tol))
                c = np.asarray(circle.coords[0])
                width = 2 * float(circle.length)
                problems.append(
                    f"{prefix}: the shape it draws differs from the boolean of the new vertices by a part "
                    f"{width:.4g} um wide at ({c[0]:.6g}, {c[1]:.6g}): a part appeared or vanished, or a seam opened"
                )
        return problems

    @staticmethod
    def _extension_changes(
        index: int,
        own: np.ndarray,
        own_sizes: Sequence[int],
        exts: Sequence[tuple[str, float, _FaceEnd, _FaceEnd]],
        extended: Sequence[tuple[str, float, tuple[int, ...]]],
        ports: dict[str, _Port],
        tol: float,
    ) -> list[str]:
        """Port faces where a layer's polygons no longer meet the face as its extensions assume.

        The face corners of the device's extensions (from their frozen face ends,
        with the current vertices and ports) are compared with those found again
        (:func:`_face_intervals`, within ``tol`` as a device built now would find
        them); they may move by up to ``tol``. The corners set where each
        extension starts and how wide it is over its whole length, so a corner
        that moved along the face counts as much as one that moved off it.

        Args:
            index: the layer.
            own: (E, 2) current vertices (um) of its own polygons (no extensions).
            own_sizes: vertices per own polygon.
            exts: the device's extensions on the layer (``source_extensions``).
            extended: (port, length, layers it may continue) per extended port.
            ports: the component's current ports by name.
            tol: the layer's topology tolerance (um) for ``own``.

        Returns:
            One line per port whose extensions changed.
        """

        def spans(c: np.ndarray) -> str:
            if not len(c):
                return "no extension"
            return "extensions from " + ", ".join(f"({a[0]:.6g}, {a[1]:.6g}) to ({b[0]:.6g}, {b[1]:.6g})" for a, b in c)

        found = []
        for name, length, layers in extended:
            if index not in layers:
                continue
            port = ports[name]
            now = _face_intervals(port, own, own_sizes, tol)
            if index == port.layer and not now:
                now = [(_PORT_CORNER, _PORT_CORNER)]
            built = _face_spans([(a, b) for n, ln, a, b in exts if n == name and ln == length], own, port)
            seen = _face_spans(now, own, port)
            # non-finite where an edge no longer crosses the face: never within tol
            if built.shape == seen.shape and bool(np.all(np.linalg.norm(built - seen, axis=-1) <= tol)):
                continue
            found.append(
                f"its cross-section at port {name!r} changed: the device has {spans(built)} on the port face, "
                f"a device built now would have {spans(seen)}"
            )
        return found

    # ------------------------------------------------------------ diagnostics

    @timed("diagnose")
    def diagnose(self, resolution: float | None = None) -> list[StackDiagnostic]:
        """Geometry the grid does not resolve, and overlaps a level's offset does not draw as their union's.

        Every level but the fills is checked in the box, on the shape it draws
        from the device's polygons (``source_vertices``, port extensions
        included) with its layer expression, sizings, bias, z_to_bias and
        sidewall angle at its nominal zmin, thickness and angle:

        - ``"width"`` and ``"gap"``: parts of the shape, and gaps in it,
          narrower than 1.5 cells divided by ``xy_subsamples`` (sub-samples
          resolve thinner features), between any two edges that face each
          other at less than 90 degrees: also a corner pointing at an edge or
          at another corner, and a V-notch nearly cutting through a part. The
          two sides of one corner (sharp, cut, blunt or rounded) are the
          corner, not a narrow part, where they meet at 45 degrees or more;
          sides meeting at a smaller angle (a needle, a taper) are measured
          where they do not touch. The layers are offset as the device draws
          them, by their signed distance: with round joins where a layer grows
          at a convex corner or shrinks at a reflex one (drawn to within 1% of
          the threshold), and a round join is part of its corner, not a narrow
          part with the edges next to it. Where the level's boundary is
          offset, the shape changes with height, so it is checked at
          cross-sections over the level's height in the box, close enough that
          nothing narrows by more than half the threshold from one to the next
          (a part that vanishes, or a gap that closes, within the level is
          found too). Only the box counts: a width or gap is found where one
          of its two edges enters the box, and measured, and located, on the
          parts of its edges in the box. Port extensions that end at a box
          face are continued past it for the checks, so their artificial ends
          are not reported.
        - ``"overlap"`` (levels drawn with the composite model only; exact
          derived shapes draw the union): polygons of a source layer overlap
          with positive area (not only abut), and the level shrinks that layer somewhere in its
          height: a sidewall angle above ``z_ref``, a negative bias, z_to_bias
          or sizing, or a level that grows and subtracts the layer. A layer's
          distance is the minimum of its polygons' distances, so the level
          draws the union of the shrunk polygons, not their shrunk union,
          which can open slits where they overlap (where an overlap is
          narrower than about twice the shrink). Grows are exact.

        The checks run on concrete numpy data with KLayout (never tracers).
        :meth:`place_on_grid` warns with the message of every finding (a
        :class:`StackDiagnosticWarning`) unless ``diagnostics`` is False.

        Args:
            resolution: grid cell size (um); default: the placed device's.

        Returns:
            At most one finding per level and kind, in the order of ``levels``.

        Raises:
            ValueError: if the device is neither placed nor given a resolution,
                or has no box, or the resolution is not positive.
        """
        import klayout.db as kdb

        placed = self._grid_slice_tuple != INVALID_SLICE_TUPLE_3D
        if resolution is None:
            if not placed:
                raise ValueError(f"StackDevice {self.name!r} is not placed; pass the grid resolution (um)")
            resolution = self._res_um()
        cell = float(resolution)
        if not cell > 0:
            raise ValueError(f"resolution must be a positive length (um), got {resolution}")
        if placed:
            size = tuple(n * self._res_um() for n in self.grid_shape)
        elif self.partial_real_shape is None or None in tuple(self.partial_real_shape):
            raise ValueError(f"StackDevice {self.name!r} has no box (partial_real_shape)")
        else:
            size = tuple(float(s) * 1e6 for s in self.partial_real_shape)
        threshold = _MIN_FEATURE * cell / self.xy_subsamples
        z_box = (self.z_center - 0.5 * size[2], self.z_center + 0.5 * size[2])
        center = np.asarray(self.center, dtype=float)
        half = 0.5 * np.array(size[:2])

        # every level's literals (offsets in um) and the heights of its checked cross-sections
        source = {index: i for i, (_, index) in enumerate(self._slots())}
        plans = []
        reach = 0.0  # largest offset (um) of a literal
        for lv in self.levels:
            expr = self._drawn_expr(lv)
            if expr[0] == "fill":
                continue
            tree, literals = _literals(expr, 1.0, source)
            lo, hi = _span(lv)
            lo, hi = max(lo, z_box[0]), min(hi, z_box[1])
            if tree in (_TRUE, _FALSE) or not hi > lo:
                continue
            # a part narrows by twice the offset's change: at most half the threshold between heights
            heights = _check_heights(lv, lo, hi, 0.25 * threshold)
            offsets = _wall_offset(lv, heights)
            reach = max([reach, *(float(np.abs(a * offsets + b).max()) for _, a, b in literals)])
            plans.append((lv, tree, literals, heights, offsets))
        if not plans:
            return []

        # the polygons (um, from the device center); extensions that end near
        # or beyond a box face continue past it (their ends there are artificial)
        margin = threshold + reach + 2 * cell  # how far the plane the checks see reaches past the box
        extensions = self.source_extensions or tuple(() for _ in self.source_layers)
        polygons: dict[int, list[np.ndarray]] = {}
        for (_, index), v, sizes, exts in zip(
            self.source_layers, self.source_vertices, self.source_sizes, extensions, strict=True
        ):
            polys = np.split(np.asarray(v, dtype=float) - center, np.cumsum(sizes)[:-1])
            for rect in polys[len(polys) - len(exts) :]:
                along = rect[1] - rect[0]  # corners 0, 3 on the port face, 1, 2 at the end
                length = float(np.linalg.norm(along))
                if not length > 0:
                    continue
                n, far = along / length, 0.5 * (rect[1] + rect[2])
                axes = np.abs(n) > 1e-12
                exit_ = np.min((np.sign(n[axes]) * half[axes] - far[axes]) / n[axes])  # to the box face ahead
                if exit_ < margin:
                    rect[1:3] += (2 * float(half.sum()) + margin) * n
            polygons[index] = polys
        extent = max(float(np.abs(np.concatenate([np.concatenate(p) for p in polygons.values()])).max()),
                     *(half + margin))
        dbu = max(_CHECK_DBU, 4 * extent / 2**31)  # 32-bit coordinates, with room for the offsets

        def box_region(h: np.ndarray) -> Any:
            """The region |x| <= h[0], |y| <= h[1] (um)."""
            x, y = (round(float(c) / dbu) for c in h)
            return kdb.Region(kdb.Box(-x, -y, x, y))

        box, universe = box_region(half), box_region(half + margin)
        inside = (round(float(half[0]) / dbu), round(float(half[1]) / dbu))
        # the layers within reach of the universe decide its shapes (offsets reach at most `reach`)
        clip = box_region(half + 2 * margin)
        regions = {index: _kdb_region(polys, dbu) & clip for index, polys in polygons.items()}
        for k, d in enumerate(self.derived_layers):  # exact shapes, from the same (continued) polygons
            for j, r in enumerate(d.recipes):
                if len(r.kind):
                    pool = np.concatenate([np.concatenate(polygons[index]) for index in d.layers])
                    rings = recipe_rings(r, run_recipe(r.kind, r.index, r.positions, pool, np))
                    outer = [q for q in rings if len(q) >= 3 and _ring_area(q) > 0]
                    holes = [q for q in rings if len(q) >= 3 and _ring_area(q) < 0]
                    regions[_part_index(k, j)] = (_kdb_region(outer, dbu) - _kdb_region(holes, dbu)) & clip
        limit = round(threshold / dbu)
        tol = max(1.0, _ROUND_TOL * limit)  # dbu: round joins are drawn this close to round
        corners: dict[int, tuple[tuple[np.ndarray, Any], tuple[np.ndarray, Any]]] = {}
        keys = {index: k for k, index in self.source_layers}
        overlaps: dict[int, tuple[int, float, np.ndarray] | None] = {}
        n_sub = f" / xy_subsamples {self.xy_subsamples}" if self.xy_subsamples > 1 else ""
        off = " (diagnostics=False turns these warnings off)"

        findings = []
        for lv, tree, literals, heights, offsets in plans:
            prefix = f"StackDevice {self.name!r}, level {lv.name!r}"
            # the layers offset with round joins, as the device draws them (see _round_offset)
            near = kdb.Edges()
            for index in sorted({index for index, _, _ in literals}):
                near += regions[index].edges()
                if index not in corners:
                    corners[index] = tuple((v, _seeds(v)) for v in _corners(regions[index]))
            near.merged_semantics = False
            cache: dict[tuple[int, int, int], tuple[Any, np.ndarray]] = {}
            joins: list[np.ndarray] = []  # of the shape being built
            other = [0]  # largest offset (dbu) of the level's layers at the height being built

            def literal(
                index: int, grow: int, near: Any = near, cache: dict = cache, joins: list = joins, other: list = other
            ) -> Any:
                """A layer grown by ``grow`` dbu with round joins (recorded in ``joins``)."""
                key = (index, grow, other[0])
                if key not in cache:
                    # a width or gap at a join is within the limit of the join, which is within |grow| of
                    # its corner, and of another boundary, which is within other[0] of its layer's edges
                    cache[key] = _round_offset(
                        regions[index], grow, corners[index], near, limit + other[0] + math.ceil(tol) + 2, tol
                    )
                joins.append(cache[key][1])
                return cache[key][0]

            # parts and gaps narrower than the threshold, at every checked height
            found: dict[str, list[tuple[float, np.ndarray, np.ndarray, float]]] = {"width": [], "gap": []}
            seen = set()
            for z, s in zip(heights, offsets, strict=True):
                if round(s / dbu) in seen:
                    continue
                seen.add(round(s / dbu))
                joins.clear()
                other[0] = max(abs(round(-(a * s + b) / dbu)) for _, a, b in literals)
                shape = _tree_region(tree, literals, literal, float(s), universe, dbu)
                drawn = np.concatenate(joins) if joins else np.zeros((0, 3))
                for kind, check in (("width", shape.width_check), ("gap", shape.space_check)):
                    pairs = check(limit, ignore_angle=_CHECK_ANGLE)
                    places = _narrow_places(pairs, inside, limit, shape, drawn, tol)
                    found[kind] += [(d, c, e, float(z)) for d, c, e in places]
            for kind, places in found.items():
                if not places:
                    continue
                # the narrowest place, or the longest one within the joins' tolerance of it
                least = min(p[0] for p in places)
                d, c, _, z = max(
                    (p for p in places if p[0] <= least + tol), key=lambda p: float(np.hypot(*(p[2][1] - p[2][0])))
                )
                spots = kdb.Region()
                for _, _, e, _ in places:
                    (x0, y0), (x1, y1) = np.floor(e[0]).astype(np.int64), np.ceil(e[1]).astype(np.int64)
                    spots.insert(kdb.Box(int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1))
                count = spots.merged().count()
                size = d * dbu
                location = (float(center[0] + c[0] * dbu), float(center[1] + c[1] * dbu))
                what = ("part", "parts") if kind == "width" else ("gap", "gaps")
                findings.append(StackDiagnostic(
                    level=lv.name,
                    kind=kind,
                    size=size,
                    cells=size / cell,
                    threshold=threshold,
                    location=location,
                    z=z,
                    count=count,
                    layers=(),
                    message=(
                        f"{prefix}: {count} {what[count > 1]} narrower than {threshold:.4g} um "
                        f"({_MIN_FEATURE:g} cells{n_sub}); the narrowest is {size:.4g} um ({size / cell:.3g} "
                        f"cells) wide, at ({location[0]:.4f}, {location[1]:.4f}) um and z = {z:.4f} um. The grid "
                        f"does not resolve it: refine the grid, raise xy_subsamples, or widen the {what[0]}{off}"
                    ),
                ))

            # overlapping polygons on the layers the level shrinks
            shrink: dict[int, tuple[float, float]] = {}  # largest shrink (um) of each layer and its height
            for index, a, b in literals:
                if index < 0:
                    continue  # an exact shape: its distance is the offset of its union
                r = a * offsets + b
                k = int(np.argmax(r))
                if r[k] > dbu and r[k] > shrink.get(index, (0.0, 0.0))[0]:
                    shrink[index] = (float(r[k]), float(heights[k]))
            hits = {}
            for index in shrink:
                if index not in overlaps:
                    overlaps[index] = _overlaps(polygons[index], dbu, box)
                if overlaps[index] is not None:
                    hits[index] = overlaps[index]
            if hits:
                worst = max(hits, key=lambda i: shrink[i][0])
                size, z = shrink[worst]
                count = sum(h[0] for h in hits.values())
                area = sum(h[1] for h in hits.values())
                c = center + max(hits.values(), key=lambda h: h[1])[2]
                location = (float(c[0]), float(c[1]))
                layers = tuple(keys[i] for i in hits)
                findings.append(StackDiagnostic(
                    level=lv.name,
                    kind="overlap",
                    size=size,
                    cells=size / cell,
                    threshold=0.0,
                    location=location,
                    z=z,
                    count=count,
                    layers=layers,
                    message=(
                        f"{prefix} shrinks {', '.join(layers)} by up to {size:.4g} um ({size / cell:.3g} cells; "
                        f"sidewall angle, bias, z_to_bias or sizing), and {'its' if len(layers) == 1 else 'their'} "
                        f"polygons overlap at {count} {'place' if count == 1 else 'places'} in the box "
                        f"({area:.4g} um^2 in all; e.g. at ({location[0]:.4f}, {location[1]:.4f}) um). The level "
                        "draws the union of the shrunk polygons, not their shrunk union, which can open slits where "
                        "they overlap (where an overlap is narrower than about twice the shrink). Make the polygons "
                        f"abut instead of overlap, or merge them{off}"
                    ),
                ))
        return findings

    # ------------------------------------------------------------ geometry

    def _res_um(self) -> float:
        """Grid resolution in um."""
        return self._config.resolution * 1e6

    def interface_offsets(self, key: Array | None) -> list[Array]:
        """Samples every roughness entry.

        Args:
            key: PRNG key (default: ``PRNGKey(noise_seed)``).

        Returns:
            One (nx, ny) height offset in cells per entry of ``roughness``.
        """
        nx, ny, _ = self.grid_shape
        if key is None:
            key = jax.random.PRNGKey(self.noise_seed)
        return [
            r.sample((nx, ny), self._res_um(), jax.random.fold_in(key, i))
            for i, r in enumerate(self.roughness)
        ]

    @timed("layer_geometry")
    def _layer_geometry(self, params: dict[str, Array]) -> list[_LayerGeometry]:
        """Every source layer's polygons in grid cells, with their static edge structure.

        Partly shared junction edges are split first (static topology, from
        ``source_vertices`` within the topology tolerance, ``topology_tol``; see
        :meth:`check_topology`), so abutting polygons, such as a port extension
        and its waveguide or a taper on an MMI side, join without a seam.

        Args:
            params: device parameters (``vertices/<layer>`` in um).

        A derived shape (``derived_layers``) is one polygon whose vertices its
        recipe evaluates from its layers' vertices (:func:`run_recipe`); its
        zero-length joints and glue cuts are left out of the distance.

        Returns:
            One entry per layer slot (:meth:`_slots`: the source layers, then
            the nonempty derived shapes); the boundary links only with
            ``corner_lines``.
        """
        nx, ny, _ = self.grid_shape
        res_um = self._res_um()
        offset = np.array([nx / 2, ny / 2])
        center = np.asarray(self.center, dtype=float)
        out = []
        keys = {index: k for k, index in self.source_layers}
        for d in self.derived_layers:
            pool = None
            for r in d.recipes:
                if not len(r.kind):
                    continue
                if pool is None:
                    pool = jnp.concatenate(
                        [(jnp.asarray(params[keys[index]]) - center) / res_um + offset for index in d.layers]
                    )
                topology = (np.asarray(r.positions, dtype=float) - center) / res_um + offset
                v = run_recipe(r.kind, r.index, topology, pool)
                tol = d.tol / res_um
                sizes = (len(topology),)
                nxt, ids, boundary = _edge_topology(topology, sizes, tol)
                links = (None, None, None)
                if self.corner_lines > 0:
                    links = _boundary_links(topology, sizes, nxt, ids, boundary, tol)
                out.append(_LayerGeometry(v, nxt, ids, boundary, *links))
        derived = out
        out = []
        for (k, _), v0, sizes in zip(self.source_layers, self.source_vertices, self.source_sizes, strict=True):
            topology = (np.asarray(v0, dtype=float) - center) / res_um + offset
            tol = self._topology_tol(v0) / res_um  # cells
            # partly shared junction edges -> exactly shared seams (static)
            split, split_sizes = split_t_junctions(topology, sizes, tol)
            v = ((jnp.asarray(params[k]) - center) / res_um + offset)[split]
            nxt, ids, boundary = _edge_topology(topology[split], split_sizes, tol)
            links = (None, None, None)
            if self.corner_lines > 0:
                links = _boundary_links(topology[split], split_sizes, nxt, ids, boundary, tol)
            out.append(_LayerGeometry(v, nxt, ids, boundary, *links))
        return out + derived

    def _sample_points(self, shift: tuple[float, float] = (0.0, 0.0)) -> Array:
        """Lateral sample points (nx * ny, 2) in cells: the voxel centers moved by ``shift``, i-major."""
        nx, ny, _ = self.grid_shape
        xx, yy = jnp.meshgrid(
            jnp.arange(nx) + 0.5 + shift[0], jnp.arange(ny) + 0.5 + shift[1], indexing="ij"
        )
        return jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    @timed("source_sdfs")
    def _source_sdfs(
        self,
        params: dict[str, Array],
        shift: tuple[float, float] = (0.0, 0.0),
        geometry: Sequence[_LayerGeometry] | None = None,
    ) -> dict[int, Field]:
        """Signed distance of every source layer at the voxel centers.

        Args:
            params: device parameters (``vertices/<layer>`` in um).
            shift: offset (cells) of the sample points from the voxel centers.
            geometry: :meth:`_layer_geometry` of ``params``, if already known.

        Returns:
            (distance, gx, gy) per layer slot index (:meth:`_slots`) that a
            level is drawn with, (nx, ny) each, in cells, negative inside;
            (gx, gy) is the gradient with respect to the point.
        """
        nx, ny, _ = self.grid_shape
        if geometry is None:
            geometry = self._layer_geometry(params)
        read = set().union(*(_expr_layers(self._drawn_expr(lv)) for lv in self.levels))
        sdf = {}
        for (_, index), (v, nxt, ids, boundary, *_) in zip(self._slots(), geometry, strict=True):
            if index not in read:
                continue
            points = self._sample_points(shift).astype(v.dtype)

            def distance(q: Array, v: Array = v, nxt: np.ndarray = nxt, ids: np.ndarray = ids,
                         boundary: np.ndarray = boundary) -> Array:
                """Signed distance (cells) at the points q (P, 2)."""
                return signed_distance(q, v, v[nxt], ids, boundary, self.edge_chunk)

            # value and gradient with respect to the sample point (forward mode)
            tangents = jnp.stack([jnp.zeros_like(points).at[:, a].set(1.0) for a in (0, 1)])
            sd, grad = jax.vmap(
                lambda t, f=distance: jax.jvp(f, (points,), (t,)), out_axes=(None, 0)
            )(tangents)
            sdf[index] = (sd.reshape(nx, ny), grad[0].reshape(nx, ny), grad[1].reshape(nx, ny))
        return sdf

    @timed("level_surfaces")
    def _level_surfaces(
        self, p: dict[str, Array], noise_key: Array | None, apply_roughness: bool, z_shift: float
    ) -> tuple[list[tuple[_Level, tuple[Array, Array, Array] | None]], list[Array], list[Array]]:
        """The levels by priority, with their sidewalls and (rough) surfaces.

        Args:
            p: all device parameters.
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            z_shift: offset (cells) of the sampled voxels from the grid's voxels along z.

        Returns:
            active: (level, its (zmin, thickness, angle) for
                :func:`_sidewall_shift`, or None for fill levels) by increasing
                ``mesh_order``.
            lows: (nx, ny) bottom surface of each level (cells).
            highs: (nx, ny) top surface of each level (cells).
        """
        nx, ny, nz = self.grid_shape
        res_um = self._res_um()
        z0 = self.z_center - 0.5 * nz * res_um  # stack z at the device bottom
        offsets_z = self.interface_offsets(noise_key) if apply_roughness and self.roughness else []
        dtype = self._config.dtype
        order = sorted(range(len(self.levels)), key=lambda i: (self.levels[i].mesh_order, i))
        tops = {lv.name: max(lv.zmin, lv.zmin + lv.thickness) for lv in self.levels}
        active = []  # (level, (zmin, thickness, angle) of its sidewall or None for fills)
        lows, highs = [], []
        for i in order:
            lv = self.levels[i]
            # every level spans the whole box height: which voxels it fills
            # comes from its (traced) surfaces alone, so zmin, thickness and
            # roughness may move it anywhere in the box
            zb = (p[f"zmin/{lv.name}"] - z0) / res_um - z_shift
            t = p[f"thickness/{lv.name}"] / res_um
            angle = p[f"sidewall_angle/{lv.name}"]

            # surfaces (cells), with roughness shared per interface
            z_lo, z_hi = (zb, zb + t) if lv.thickness > 0 else (zb + t, zb)
            lo_nominal = lv.zmin if lv.thickness > 0 else lv.zmin + lv.thickness
            hi_nominal = lv.zmin + lv.thickness if lv.thickness > 0 else lv.zmin
            z_lo = jnp.broadcast_to(z_lo, (nx, ny))
            z_hi = jnp.broadcast_to(z_hi, (nx, ny))
            for r, dz in zip(self.roughness if offsets_z else (), offsets_z, strict=True):
                if r.level == lv.name:
                    z_hi = z_hi + dz
                elif r.fill_above and r.level in tops and abs(lo_nominal - tops[r.level]) < 1e-6:
                    z_lo = z_lo + dz

            # the sidewall (angle, bias, z_to_bias) is evaluated in _composite at
            # the height of each part of a voxel the level fills
            wall = None if self._drawn_expr(lv)[0] == "fill" else (zb, t, angle)
            active.append((lv, wall))
            lows.append(z_lo.astype(dtype))
            highs.append(z_hi.astype(dtype))
        return active, lows, highs

    @timed("corner_plan")
    def _corner_plan(
        self, active: Sequence[tuple[_Level, Any]], lows: Sequence[Array], highs: Sequence[Array]
    ) -> tuple[_CornerPlan | None, list[list[Array]]]:
        """The levels' literals for the corner model, and how far from a column each matters.

        Args:
            active: the levels by priority (:meth:`_level_surfaces`).
            lows: (nx, ny) bottom surface of each level (cells).
            highs: (nx, ny) top surface of each level (cells).

        Returns:
            plan: the literals, or None without ``corner_lines`` or when no level
                reads a source layer.
            radii: per level, per literal: the distance (cells, no gradient)
                within which its boundary can cut the lateral square somewhere
                in the box: the square's half-diagonal plus the largest offset
                of the literal over the heights the level covers.
        """
        if self.corner_lines <= 0:
            return None, []
        nz = self.grid_shape[2]
        res_um = self._res_um()
        # with sub-samples, _VOXEL_RAMP more: a sample's certificate for its voxel
        # is 1 where its columns end (see _composite_samples)
        reach = math.sqrt(2.0) * 0.5 + _CORNER_TOL + (_VOXEL_RAMP if self.xy_subsamples > 1 else 0.0)
        half_diagonal = reach * self.edge_width / self.xy_subsamples
        position = {index: i for i, (_, index) in enumerate(self._slots())}
        trees, literals, radii, layers = [], [], [], []
        for (lv, wall), lo, hi in zip(active, lows, highs, strict=True):
            if wall is None:
                trees.append(None)
                literals.append(())
                radii.append([])
                continue
            tree, lits = _literals(self._drawn_expr(lv), res_um, position)
            s_lo, s_hi = _lateral_range(lv, wall, lo, hi, nz, res_um)
            slots, rad = [], []
            for index, a, b in lits:
                if position[index] not in layers:
                    layers.append(position[index])
                slots.append((layers.index(position[index]), a, b))
                rad.append(half_diagonal + jnp.maximum(jnp.abs(a * s_lo + b), jnp.abs(a * s_hi + b)))
            trees.append(tree)
            literals.append(tuple(slots))
            radii.append(rad)
        if not layers:
            return None, []
        return _CornerPlan(trees=tuple(trees), literals=tuple(literals), layers=tuple(layers)), radii

    @timed("corner_capacity")
    def _corner_capacity(self, plan: _CornerPlan, geometry: Sequence[_LayerGeometry]) -> int:
        """Static number of corner columns per sample grid.

        ``corner_columns`` once the device is placed; otherwise a bound from the
        boundary vertices and the levels' nominal offsets, with a warning.

        Args:
            plan: the levels' literals.
            geometry: the source layers (:meth:`_layer_geometry`).

        Returns:
            The number of columns (at most nx * ny).
        """
        nx, ny, _ = self.grid_shape
        if self.corner_columns is not None:
            return max(0, min(nx * ny, int(self.corner_columns)))
        warnings.warn(
            f"StackDevice {self.name!r} has no corner_columns (it is set when the device is placed); "
            "using a static bound, which may be much larger than needed",
            stacklevel=3,
        )
        res_um = self._res_um()
        offsets = [0.0] * len(plan.layers)  # bound on the literals' offsets (cells), per layer slot
        for lv, literals in zip(sorted(self.levels, key=lambda lv: lv.mesh_order), plan.literals, strict=True):
            t = abs(lv.thickness) / res_um
            biases = [abs(lv.bias)] + [abs(b) for b in (lv.z_to_bias[1] if lv.z_to_bias else ())]
            wall = (1 + abs(lv.width_to_z)) * t * abs(math.tan(math.radians(lv.sidewall_angle)))
            for layer, _, b in literals:
                offsets[layer] = max(offsets[layer], abs(b) + wall + max(biases) / res_um)
        reach = math.sqrt(2.0) * 0.5 + (_VOXEL_RAMP if self.xy_subsamples > 1 else 0.0)
        half_diagonal = reach * self.edge_width / self.xy_subsamples
        bound = 64
        for layer, offset in zip(plan.layers, offsets, strict=True):
            vertices = int(np.count_nonzero(geometry[layer].boundary))
            bound += vertices * math.ceil(math.pi * (half_diagonal + offset + 1.5) ** 2)
        return min(nx * ny, bound)

    @timed("corner_inputs")
    def _corner_inputs(
        self,
        plan: _CornerPlan,
        radii: Sequence[Sequence[Array]],
        geometry: Sequence[_LayerGeometry],
        sdf: dict[int, Field],
        shift: tuple[float, float],
        capacity: int,
    ) -> _CornerInputs:
        """The corner columns of one lateral sample grid and their local models.

        Args:
            plan: the levels' literals.
            radii: per level, per literal, the radius of :meth:`_corner_plan`.
            geometry: the source layers (:meth:`_layer_geometry`).
            sdf: the source layers' distances on this grid (:meth:`_source_sdfs`).
            shift: offset (cells) of the grid's points from the voxel centers.
            capacity: number of corner columns (static).

        Returns:
            The corner columns for :func:`_composite`.
        """
        nx, ny, _ = self.grid_shape
        num = nx * ny
        sc = self.xy_subsamples / self.edge_width
        near = []
        for layer in plan.layers:
            v, _, _, boundary, src, dst, _ = geometry[layer]
            points = self._sample_points(shift).astype(v.dtype)
            near.append(nearest_edges(points, v[src], v[dst], boundary, self.edge_chunk))
        candidate = _corner_candidates(plan, radii, [d for d, _ in near])
        # the first candidates in raster order; unused slots hold nx * ny
        cols = jnp.nonzero(candidate, size=capacity, fill_value=num)[0]
        dropped = candidate & (jnp.cumsum(candidate) > capacity)
        cs = jnp.minimum(cols, num - 1)
        models = []
        for layer, (dist, ids) in zip(plan.layers, near, strict=True):
            v, _, pid, _, src, dst, succ = geometry[layer]
            sd, gx, gy = sdf[self._slots()[layer][1]]
            points = self._sample_points(shift).astype(v.dtype)
            normal = _unit_normal(gx, gy).reshape(num, 2)[cs]
            models.append(
                _layer_corner(v, src, dst, succ, pid, ids[cs], dist[cs], points[cs], sd.reshape(num)[cs], normal, sc)
            )
        dtype = self._config.dtype  # like the half-plane painting (the distances may be wider)
        layers = jax.tree.map(
            lambda *x: jnp.stack(x).astype(dtype) if jnp.issubdtype(x[0].dtype, jnp.floating) else jnp.stack(x), *models
        )
        return _CornerInputs(
            cols=cols,
            taken=cols < num,
            layers=layers,
            trees=plan.trees,
            literals=plan.literals,
            lines=self.corner_lines,
            sc=sc,
            dropped=dropped,
        )

    def _corner_candidate_count(self) -> int:
        """Most corner candidates of any lateral sample grid, at the initial parameters.

        Covers every grid :meth:`material_weights` samples for :meth:`write`
        and fdtdx's initial arrays: the voxel centers and the Yee positions,
        each with its xy sub-samples (the default roughness realization).

        Returns:
            The number of candidate columns.
        """
        n_sub = self.xy_subsamples
        sub = [(k + 0.5) / n_sub - 0.5 for k in range(n_sub)]
        shifts = [(0.0, 0.0, 0.0), *(_YEE_E_SHIFTS if self.yee_staggered else ())]

        @jax.jit
        def count(p: dict[str, Array], shift: Array) -> Array:
            """Candidates of the grid whose points are moved by shift (3,) from the voxel centers."""
            plan, radii = self._corner_plan(*self._level_surfaces(p, None, True, shift[2]))
            if plan is None:
                return jnp.zeros((), dtype=jnp.int32)
            geometry = self._layer_geometry(p)
            dist = []
            for layer in plan.layers:
                v, _, _, boundary, src, dst, _ = geometry[layer]
                points = self._sample_points((shift[0], shift[1])).astype(v.dtype)
                dist.append(nearest_edges(points, v[src], v[dst], boundary, self.edge_chunk)[0])
            return jnp.sum(_corner_candidates(plan, radii, dist), dtype=jnp.int32)

        p = self.init_params()
        return max(int(count(p, jnp.asarray([x + a, y + b, z]))) for x, y, z in shifts for a in sub for b in sub)

    def material_weights(
        self,
        params: dict[str, Array] | None = None,
        noise_key: Array | None = None,
        apply_roughness: bool = True,
        shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[list[str], Array]:
        """Volume fraction of every material in every voxel.

        Args:
            params: device parameters (defaults: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            shift: offset (cells) of the sampled voxels from the grid's voxels,
                e.g. to a Yee field position.

        Returns:
            names: material names in fdtdx order (``compute_ordered_names``).
            weights: (num_materials, nx, ny, nz), summing to one over materials.
        """
        return self._weights_and_normals(params, noise_key, apply_roughness, shift)[:2]

    @timed("weights")
    def _weights_and_normals(
        self,
        params: dict[str, Array] | None,
        noise_key: Array | None,
        apply_roughness: bool,
        shift: tuple[float, float, float],
        eps_cols: np.ndarray | None = None,
    ) -> tuple[list[str], Array, Array | None]:
        """Volume fractions of the materials and, with ``eps_cols``, the voxels' normal tensors.

        The tensor is :func:`_normal_tensor` of each voxel's own model for the
        permittivities ``eps_cols``, with contrasts normalized by their range
        over the levels' materials and the background. With lateral
        sub-samples, their moments are added up per z sub-interval first
        (:func:`_composite_samples`), so the variance between them counts; the
        flux kernel spans the union of their squares where these tile the
        voxel (see :func:`_lateral_samples`).

        Args:
            params: device parameters (defaults: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            shift: offset (cells) of the sampled voxels from the grid's voxels.
            eps_cols: (num_materials, C) static permittivity of every material
                (fdtdx order) for C E components, or None: fractions only.

        Returns:
            names: material names in fdtdx order.
            weights: (num_materials, nx, ny, nz), summing to one over materials.
            tensor: (C, 4, nx, ny, nz) variance along x, y, z and unexplained
                variance (see :func:`_normal_tensor`), or None without ``eps_cols``.
        """
        p = {**self.init_params(), **(params or {})}
        nx, ny, nz = self.grid_shape
        res_um = self._res_um()
        names = compute_ordered_names(self.materials)
        dtype = self._config.dtype
        active, lows, highs = self._level_surfaces(p, noise_key, apply_roughness, shift[2])
        if not active:
            w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype)
            tensor = None if eps_cols is None else jnp.zeros((eps_cols.shape[1], 4, nx, ny, nz), dtype=dtype)
            return names, w.at[names.index(self.background)].set(1.0), tensor
        is_fill = tuple(self._drawn_expr(lv)[0] == "fill" for lv, *_ in active)
        level_materials = [lv.material for lv, *_ in active]

        n_sub = self.xy_subsamples
        to_square = n_sub / self.edge_width  # cells -> sides of the lateral square

        def lateral(z: Array) -> Array:
            """(..., K) offsets (scaled cells) of the levels' boundaries at heights z (cells)."""
            return jnp.stack(
                [
                    jnp.zeros_like(z) if wall is None else to_square * _sidewall_shift(lv, z, *wall, res_um)
                    for lv, wall in active
                ],
                axis=-1,
            )

        # heights (cells) where a level's offset bends (its z_to_bias knots): each
        # level takes the exact mean of its offset over its part of a voxel
        bends = [[] if wall is None or lv.z_to_bias is None else [wall[0] + f * wall[1] for f in lv.z_to_bias[0]]
                 for lv, wall in active]
        knots = None
        if any(bends):
            # padded by repeating a level's last knot (a repeated knot, or one on a
            # straight offset, changes nothing)
            rows = [b or [0.0] for b in bends]
            width = max(len(r) for r in rows)
            knots = jnp.stack([jnp.stack([jnp.asarray(x, dtype=dtype) for x in r + r[-1:] * (width - len(r))])
                               for r in rows])  # (K, M)

        cell = self.edge_width / n_sub  # cells per side of a sub-sample's square
        normal_args = {}
        if eps_cols is not None:
            # each level's eps minus the background's, over the range of both (per component)
            eps_levels = np.asarray([eps_cols[names.index(m)] for m in level_materials], dtype=float)  # (K, C)
            eps_background = np.asarray(eps_cols[names.index(self.background)], dtype=float)
            spread = np.ptp(np.concatenate([eps_levels, eps_background[None]]), axis=0)
            spread = np.where(spread > 0, spread, 1.0)
            normal_args["level_eps"] = ((eps_levels - eps_background) / spread).T  # (C, K)
            normal_args["cell"] = cell
            if apply_roughness and self.roughness:  # the surfaces' lateral slopes (cells per cell)

                def slopes(z: Array) -> Array:
                    """(nx, ny, 2) gradient of a surface; 0 along an axis of one cell."""
                    return jnp.stack(
                        [jnp.gradient(z, axis=a) if z.shape[a] > 1 else jnp.zeros_like(z) for a in (0, 1)], axis=-1
                    )

                normal_args["surface_slopes"] = jnp.stack([slopes(z) for z in (*lows, *highs)], axis=-2)

        def half_planes(sdf: dict[int, Field]) -> tuple[Array, Array]:
            """(nx, ny, K) offsets (sides of the square) and (nx, ny, K, 2) normals of the levels, by priority."""
            offsets, normals = [], []
            for (lv, _), fill in zip(active, is_fill, strict=True):
                if fill:
                    offsets.append(jnp.full((nx, ny), -_BIG, dtype=dtype))
                    normals.append(jnp.zeros((nx, ny, 2), dtype=dtype))
                    continue
                sd, gx, gy = _eval_expr(self._drawn_expr(lv), sdf, (nx, ny), res_um)
                offsets.append((sd * to_square).astype(dtype))
                # (1, 0) where no layer is near (BIG distances, zero gradient)
                normals.append(_unit_normal(gx, gy).astype(dtype))
            return jnp.stack(offsets, axis=-1), jnp.stack(normals, axis=-2)

        surfaces = (jnp.stack(lows, axis=-1), jnp.stack(highs, axis=-1))  # (nx, ny, K) each
        geometry = self._layer_geometry(p)
        plan, radii = self._corner_plan(active, lows, highs)
        capacity = 0 if plan is None else self._corner_capacity(plan, geometry)
        # several sub-samples are added up per sub-interval (_composite_samples) with the tensor,
        # and with corner columns (their corner model is blended by the voxel's certificate)
        together = n_sub > 1 and (bool(normal_args) or capacity > 0)
        weights, tensor, samples = 0.0, None, []
        for a, b, kernel in _lateral_samples(n_sub, self.edge_width):
            grid = (shift[0] + a, shift[1] + b)
            sdf = self._source_sdfs(p, grid, geometry)
            corner = None
            if capacity:
                corner = self._corner_inputs(plan, radii, geometry, sdf, grid, capacity)
            if together:
                samples.append((*half_planes(sdf), corner, kernel))
                continue
            out = _composite(
                *half_planes(sdf), is_fill, *surfaces, lateral, nz, corner=corner, kernel=kernel, knots=knots,
                **normal_args,
            )
            if normal_args:  # one sub-sample
                out, tensor = out
            weights = weights + out
        if together:
            offsets, normals, corners, kernels = zip(*samples, strict=True)
            weights, tensor = _composite_samples(
                jnp.stack(offsets),
                jnp.stack(normals),
                is_fill,
                *surfaces,
                lateral,
                nz,
                normal_args.get("level_eps"),
                kernels,
                cell,
                corners if capacity else None,
                normal_args.get("surface_slopes"),
                knots=knots,
            )
        weights = weights / n_sub**2
        out = []
        for m in names:
            cols = [j for j, lm in enumerate(level_materials) if lm == m]
            if m == self.background:
                cols.append(len(level_materials))
            out.append(weights[..., cols].sum(axis=-1) if cols else jnp.zeros((nx, ny, nz), dtype))
        if tensor is not None:
            tensor = jnp.moveaxis(tensor, (3, 4), (0, 1))  # (C, 4, nx, ny, nz)
        return names, jnp.stack(out), tensor

    # ------------------------------------------------------------ fdtdx hooks

    @timed("place_on_grid")
    def place_on_grid(
        self, grid_slice_tuple: Any, config: SimulationConfig, key: Array
    ) -> Self:
        """fdtdx hook: aligns pole slots, requests anisotropic arrays (subpixel), sizes corner columns, diagnoses.

        Without ``corner_columns``, the corner model gets 1.5 times the most
        candidate columns of any sample grid at the initial parameters, plus 64,
        rounded up to a multiple of 64 (at most every column). With
        ``diagnostics``, every finding of :meth:`diagnose` at the grid's
        resolution is a :class:`StackDiagnosticWarning`.

        Args:
            grid_slice_tuple: the device's grid slices.
            config: simulation config.
            key: PRNG key.

        Returns:
            The placed device.
        """
        self = self._keep_reachable_levels(grid_slice_tuple, config)
        # aligned pole slots also for devices built without from_layer_stack;
        # fdtdx allocates the slots from the placed objects' materials
        mats = {k: v for k, v in self.materials.items() if k != _ANISOTROPY_SENTINEL}
        mats = _drop_empty_pole_slots(align_material_poles(mats))
        if self.averaging == "subpixel":
            # makes fdtdx allocate diagonally anisotropic arrays (never painted)
            mats[_ANISOTROPY_SENTINEL] = fdtdx.Material(permittivity=(1.0, 1.0, 2.0))
        self = self.aset("materials", mats)
        warn_reversible_losses(mats, config, self.name, conductivity=True)
        placed = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        lap("pole_alignment")
        if placed.corner_lines > 0 and placed.corner_columns is None:
            placed = placed.aset("corner_columns", placed._corner_budget())
        if placed.diagnostics:
            for finding in placed.diagnose():
                warnings.warn(finding.message, StackDiagnosticWarning, stacklevel=2)
        return placed

    @timed("corner_budget")
    def _corner_budget(self) -> int:
        """Corner columns for the initial parameters of a placed device (see :meth:`place_on_grid`)."""
        nx, ny, _ = self.grid_shape
        return min(nx * ny, 64 * math.ceil((1.5 * self._corner_candidate_count() + 64) / 64))

    @timed("keep_reachable_levels")
    def _keep_reachable_levels(self, grid_slice_tuple: Any, config: SimulationConfig) -> Self:
        """Drops the levels that cannot reach the placed box, with their materials and source layers.

        The box's exact height is known only now: fdtdx rounds it to whole cells
        around ``z_center``. A level is kept if its nominal span comes within
        ``level_margin`` of the box, or within half a cell below it, which the
        staggered Ex and Ey samples of the bottom voxels see. Materials of
        dropped levels are not allocated by fdtdx (e.g. no conductivity array
        for a metal level above the box).

        Args:
            grid_slice_tuple: the device's grid slices.
            config: simulation config.

        Returns:
            The device with the reachable levels only.
        """
        res_um = config.resolution * 1e6
        start, stop = grid_slice_tuple[2]
        half = 0.5 * (stop - start) * res_um
        bottom = self.z_center - half - 0.5 * res_um - self.level_margin
        top = self.z_center + half + self.level_margin
        levels = tuple(lv for lv in self.levels if _span(lv)[1] > bottom and _span(lv)[0] < top)
        # empty layers matter only to levels that reach the box
        empty = tuple(
            (index, reach)
            for index, readers in self.empty_layers
            if (reach := tuple((name, lo, hi) for name, lo, hi in readers if hi > bottom and lo < top))
        )
        if empty != tuple(self.empty_layers):
            self = self.aset("empty_layers", empty)
        if len(levels) == len(self.levels):
            return self
        layers = set().union(*(_expr_layers(lv.expr) for lv in levels))
        keep = [i for i, (_, index) in enumerate(self.source_layers) if index in layers]
        used = {lv.material for lv in levels} | {self.background, _ANISOTROPY_SENTINEL}
        out = self.aset("levels", levels)
        out = out.aset("materials", {k: m for k, m in self.materials.items() if k in used})
        for name in ("source_layers", "source_vertices", "source_sizes"):
            out = out.aset(name, tuple(getattr(self, name)[i] for i in keep))
        if self.source_extensions:
            out = out.aset("source_extensions", tuple(self.source_extensions[i] for i in keep))
        # the shapes of the kept levels (their pools are on the kept layers)
        names = {lv.name for lv in levels}
        shapes = tuple(
            d._replace(levels=tuple(n for n in d.levels if n in names))
            for d in self.derived_layers if any(n in names for n in d.levels)
        )
        return out.aset("derived_layers", shapes)

    def get_voxel_mask_for_shape(self) -> Array:
        """fdtdx hook: the device fills its whole box.

        Returns:
            (nx, ny, nz) all-True mask.
        """
        return jnp.ones(self.grid_shape, dtype=jnp.bool)

    def get_material_mapping(self) -> Array:
        """fdtdx hook: the nominal geometry rounded to the dominant material.

        Only used by fdtdx to paint the initial arrays; :func:`apply_stack`
        overwrites them with the mixed materials.

        Returns:
            (nx, ny, nz) material indices in fdtdx's order.
        """
        _, w = self.material_weights(apply_roughness=False)
        return jnp.argmax(w, axis=0).astype(jnp.int32)

    @timed("write")
    def write(
        self,
        arrays: Any,
        params: dict[str, Array] | None = None,
        noise_key: Array | None = None,
        apply_roughness: bool = True,
    ) -> Any:
        """Writes the mixed inverse permittivity, conductivity and pole coefficients.

        Args:
            arrays: fdtdx ArrayContainer.
            params: device parameters (default: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.

        Returns:
            The arrays with the device box overwritten.

        Raises:
            NotImplementedError: fully anisotropic (9-component) arrays.
            ValueError: unknown averaging, or "subpixel" with isotropic arrays.
        """
        n_comp = arrays.inv_permittivities.shape[0]
        if n_comp not in (1, 3):
            raise NotImplementedError("Fully anisotropic permittivity arrays are not supported")
        if self.averaging not in ("inverse", "subpixel"):
            raise ValueError(f"Unknown averaging {self.averaging!r}")
        if self.subpixel_normal not in ("voxel", "gradient"):
            raise ValueError(f"subpixel_normal must be 'voxel' or 'gradient', got {self.subpixel_normal!r}")
        if self.averaging == "subpixel" and n_comp != 3:
            raise ValueError(
                "averaging='subpixel' needs diagonally anisotropic arrays; build the "
                "device with from_layer_stack(..., averaging='subpixel')"
            )
        dtype = arrays.inv_permittivities.dtype
        iso = {"isotropic": n_comp == 1, "diagonally_anisotropic": n_comp == 3}
        eps_np = np.asarray(compute_allowed_permittivities(self.materials, **iso), dtype=float)  # (M, n_comp)
        eps = jnp.asarray(eps_np, dtype=dtype)
        sigma = None
        if arrays.electric_conductivity is not None:
            n_c = arrays.electric_conductivity.shape[0]
            sigma = jnp.asarray(
                compute_allowed_electric_conductivities(
                    self.materials, isotropic=n_c == 1, diagonally_anisotropic=n_c == 3
                ),
                dtype=dtype,
            )  # (M, n_c)
        # each E component sees the materials around its own Yee position;
        # components are processed one at a time to bound the memory
        shifts = _YEE_E_SHIFTS if n_comp == 3 and self.yee_staggered else ((0.0, 0.0, 0.0),)
        subpixel = self.averaging == "subpixel"
        voxel_normal = subpixel and self.subpixel_normal == "voxel"
        if subpixel:
            # the axes of one cell (2D simulations) get no normal component
            keep = jnp.asarray([float(n > 1) for n in self.grid_shape], dtype=dtype)[:, None, None, None]
            sentinel = np.asarray([m == _ANISOTROPY_SENTINEL for m in compute_ordered_names(self.materials)])
        inv, cond, w_cell = [], [], 0.0
        for c in range(n_comp):
            if c < len(shifts):
                if voxel_normal:  # the normals of the component's own voxels (all three if they share them)
                    cols = eps_np[:, [c]] if len(shifts) == n_comp else eps_np
                    _, w, tensors = self._weights_and_normals(params, noise_key, apply_roughness, shifts[c], cols)
                    tensors = tensors.astype(dtype)
                else:
                    w = self.material_weights(params, noise_key, apply_roughness, shifts[c])[1]
                w = w.astype(dtype)
                w_cell = w_cell + w / len(shifts)  # cell value for the isotropic quantities
                if sigma is not None and sigma.shape[1] == len(shifts):
                    cond.append(jnp.sum(w * sigma[:, c, None, None, None], axis=0))
            e = eps[:, c, None, None, None]
            if subpixel:
                # n2 <1/eps> + (1 - n2) / <eps> = (1 + n2 a) / <eps>, with the normal
                # of the voxel's own model (the unexplained variance takes the
                # one-sided differences of the mean eps), or only the latter
                # with subpixel_normal="gradient"
                mean_eps = jnp.sum(w * e, axis=0)
                fd = _normal_squared(mean_eps)[c]
                n2 = fd
                if voxel_normal:
                    tensor = tensors[0 if len(shifts) == n_comp else c]  # (4, nx, ny, nz)
                    along = tensor[:3] * keep
                    n2 = (along[c] + (tensor[3] + 1e-12) * fd) / (jnp.sum(along, axis=0) + tensor[3] + 1e-12)
                mean_inv = (1 + n2 * _anisotropy(w, eps_np[:, c], sentinel)) / mean_eps
            else:
                mean_inv = jnp.sum(w / e, axis=0)
            inv.append(lap(f"E{'xyz'[c]}" if n_comp == 3 else "E", mean_inv))
        arrays = arrays.at["inv_permittivities"].set(
            arrays.inv_permittivities.at[:, *self.grid_slice].set(jnp.stack(inv).astype(dtype))
        )
        lap("set_permittivity", arrays)
        if sigma is not None:
            if not cond:
                cond = [
                    jnp.sum(w_cell * sigma[:, k, None, None, None], axis=0)
                    for k in range(sigma.shape[1])
                ]
            cond = jnp.stack(cond) * self._config.resolution
            gradient_config = self._config.gradient_config
            if gradient_config is not None and gradient_config.method == "reversible":
                # fdtdx's reversible backward pass closes over the conductivity:
                # it gets no gradient there, and a traced one cannot be closed
                # over (place_on_grid warns that these gradients are inaccurate)
                cond = jax.lax.stop_gradient(cond)
            arrays = arrays.at["electric_conductivity"].set(
                arrays.electric_conductivity.at[:, *self.grid_slice].set(cond.astype(dtype))
            )
            lap("conductivity", arrays)
        if arrays.dispersive_c1 is not None:
            arrays = self._write_dispersion(arrays, w_cell)
        return arrays

    @timed("dispersion")
    def _write_dispersion(self, arrays: Any, w: Array) -> Any:
        """Writes the pole coefficients of the box.

        c1, c2 are the slot constants in every voxel, also where no material
        with that pole is (c3 = 0 keeps P = 0 there, and the adjoint of c3 sees
        the material's recurrence); ``c3 = sum_m w_m c3_m``. Slots that other
        objects need beyond the stack's own are zeroed in the box.

        Args:
            arrays: fdtdx ArrayContainer with dispersive arrays.
            w: (M, nx, ny, nz) material fractions in fdtdx's material order.

        Returns:
            The arrays with c1, c2, c3 and inv_c2 overwritten in the box.

        Raises:
            ValueError: if the arrays have fewer pole slots than the device needs.
        """
        n_slots = arrays.dispersive_c1.shape[0]
        dtype = arrays.dispersive_c1.dtype
        dt = self._config.time_step_duration
        slots = pole_slots(self.materials)
        n = len(slots)
        if n > n_slots:
            raise ValueError(
                f"{self.name!r} needs {n} pole slots but the arrays have {n_slots}; "
                "were the arrays initialized with this device?"
            )
        full = (n_slots, 1, *self.grid_shape)
        c1 = jnp.zeros(full, dtype=dtype)
        c2 = jnp.zeros(full, dtype=dtype)
        c3 = jnp.zeros(full, dtype=dtype)
        if n:
            a1, a2, _ = compute_pole_coefficients(slots, dt)
            _, _, a3 = compute_allowed_dispersive_coefficients(self.materials, dt, max_num_poles=n)
            shape = (n, 1, *self.grid_shape)

            def per_slot(a: np.ndarray) -> Array:
                """Broadcasts one coefficient per slot to the box."""
                return jnp.broadcast_to(jnp.asarray(a, dtype)[:, None, None, None, None], shape)

            c1 = c1.at[:n].set(per_slot(a1))
            c2 = c2.at[:n].set(per_slot(a2))
            mixed = jnp.einsum(  # full precision: GPUs default to TF32 for matmuls
                "ms,mxyz->sxyz",
                jnp.asarray(a3, dtype),
                w.astype(dtype),
                precision=jax.lax.Precision.HIGHEST,
            )
            c3 = c3.at[:n, 0].set(mixed)
        # exact reciprocal of the stored c2 (fdtdx's reverse-time step relies on it)
        inv_c2 = jnp.where(c2 == 0, 0.0, 1.0 / jnp.where(c2 == 0, 1.0, c2)).astype(dtype)
        for name, value in (
            ("dispersive_c1", c1),
            ("dispersive_c2", c2),
            ("dispersive_c3", c3),
            ("dispersive_inv_c2", inv_c2),
        ):
            current = getattr(arrays, name)
            arrays = arrays.at[name].set(current.at[:, :, *self.grid_slice].set(value))
        return arrays

    # ------------------------------------------------------------ simulation setup

    def get_simulation_volume(
        self, name: str = "volume", material: fdtdx.Material | None = None
    ) -> tuple[fdtdx.SimulationVolume, list[PositionConstraint]]:
        """Simulation volume that is exactly the device box.

        Args:
            name: name of the volume.
            material: material of the volume (default: the background); the
                device overwrites the whole box anyway.

        Returns:
            The volume and the constraint that centers this device in it.
        """
        volume = fdtdx.SimulationVolume(
            name=name,
            partial_real_shape=self.partial_real_shape,
            material=material or self.materials[self.background],
        )
        return volume, [self.place_at_center(volume)]

    def _port(self, port: str) -> _Port:
        """Looks up a port by name.

        Args:
            port: port name.

        Returns:
            The port.

        Raises:
            KeyError: if the component has no such port.
        """
        for p in self.ports:
            if p.name == port:
                return p
        raise KeyError(f"Unknown port {port!r}; {self.name!r} has ports {[p.name for p in self.ports]}")

    def _port_frame(
        self, port: str, level: str | None, port_offset: float
    ) -> tuple[_Port, int, int, list[float], float]:
        """Position and orientation of a port in the stack.

        Args:
            port: port name.
            level: level whose z span is used (default: the union of the levels
                that draw the port's layer).
            port_offset: shift (um) along the port's injection direction (into
                the component); negative moves out of it.

        Returns:
            The port, its normal axis (0 = x, 1 = y), its outward sign (+1 / -1),
            the center (x, y, z) in um (z: middle of the span) and the height of
            the span in um.

        Raises:
            KeyError: unknown port or level.
            ValueError: a port without orientation or not along x / y, or a port
                layer that no level draws.
        """
        p = self._port(port)
        if p.orientation is None:
            raise ValueError(f"Port {port!r} has no orientation")
        quarter = round(p.orientation / 90.0)
        if abs(p.orientation - 90.0 * quarter) > 1e-6:
            raise ValueError(f"Port {port!r} points at {p.orientation} deg; fdtdx planes need 0, 90, 180 or 270")
        quarter %= 4
        axis, sign = quarter % 2, (1 if quarter < 2 else -1)
        if level is not None:
            match = [lv for lv in self.levels if lv.name == level]
            if not match:
                raise KeyError(f"Level {level!r} is not in {self.name!r}: {[lv.name for lv in self.levels]}")
        else:
            match = [lv for lv in self.levels if p.layer in _expr_positive_layers(lv.expr)]
            if not match:
                raise ValueError(f"No level of {self.name!r} draws the layer of port {port!r}; pass level=...")
        spans = [_span(lv) for lv in match]
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        center = [p.x, p.y, 0.5 * (lo + hi)]
        center[axis] -= sign * port_offset  # sign: the port's outward orientation
        return p, axis, sign, center, hi - lo

    def _place_at(self, obj: Any, center: Sequence[float]) -> list[PositionConstraint]:
        """Constraint that puts the center of ``obj`` at a stack point.

        Args:
            obj: fdtdx object.
            center: (x, y, z) in um (gds xy, stack z).

        Returns:
            One position constraint relative to this device.
        """
        reference = (self.center[0], self.center[1], self.z_center)
        margins = tuple((c - r) * 1e-6 for c, r in zip(center, reference, strict=True))
        return [
            obj.place_relative_to(
                self, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0), margins=margins
            )
        ]

    def _at_port(
        self,
        cls: type,
        port: str,
        port_size_mult: Sequence[float],
        suffix: str,
        *,
        direction: PortDirection | None,
        level: str | None,
        port_offset: float,
        name: str | None,
        **fields: Any,
    ) -> tuple[Any, list[PositionConstraint]]:
        """Builds a plane source / detector on a port.

        Args:
            cls: fdtdx source or detector class.
            port: port name.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            suffix: default name is ``<port>_<suffix>``.
            direction: "in" / "out" relative to the component, or None for
                classes without a direction.
            level: see :meth:`_port_frame`.
            port_offset: see :meth:`_port_frame`.
            name: object name.
            **fields: other fields of ``cls``.

        Returns:
            The object and the constraints placing it.
        """
        p, axis, sign, center, height = self._port_frame(port, level, port_offset)
        if len(port_size_mult) != 2 or min(port_size_mult) <= 0:
            raise ValueError(f"port_size_mult must be two positive numbers, got {port_size_mult}")
        if p.width <= 0:
            raise ValueError(f"Port {port!r} has no width")
        real: list[float | None] = [None, None, None]
        grid: list[int | None] = [None, None, None]
        grid[axis] = 1  # one cell thick, normal to the port
        real[1 - axis] = port_size_mult[0] * p.width * 1e-6
        real[2] = port_size_mult[1] * height * 1e-6
        if direction is not None:
            fields["direction"] = _axis_direction(direction, sign)
        obj = cls(
            name=name or f"{port}_{suffix}",
            partial_real_shape=tuple(real),
            partial_grid_shape=tuple(grid),
            **fields,
        )
        return obj, self._place_at(obj, center)

    # ------------------------------------------------------------ sources on ports

    def create_mode_plane_source_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        mode_index: int = 0,
        filter_pol: Literal["te", "tm"] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.ModePlaneSource, list[PositionConstraint]]:
        """Waveguide mode source on a port.

        Args:
            port: port name; the plane is centered on it (in z on the middle of
                the port's level) and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            wave_character: wavelength / frequency of the mode.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            mode_index: mode order (0 = fundamental).
            filter_pol: only "te" or "tm" modes (default: all).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z (default:
                the levels that draw the port's layer).
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_mode_source``).
            **kwargs: other ModePlaneSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        fields = _given(temporal_profile=temporal_profile, filter_pol=filter_pol)
        return self._at_port(
            fdtdx.ModePlaneSource, port, port_size_mult, "mode_source", direction=direction, level=level,
            port_offset=port_offset, name=name, wave_character=wave_character, mode_index=mode_index, **fields, **kwargs,
        )

    def create_gaussian_plane_source_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        radius: float | None = None,
        std: float = 1 / 3,
        fixed_E_polarization_vector: tuple[float, float, float] | None = None,
        fixed_H_polarization_vector: tuple[float, float, float] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.GaussianPlaneSource, list[PositionConstraint]]:
        """Gaussian beam source on a port.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            wave_character: wavelength / frequency.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            radius: beam radius in um (default: half the smaller plane side).
            std: standard deviation of the Gaussian relative to ``radius``.
            fixed_E_polarization_vector: E polarization (global x, y, z).
            fixed_H_polarization_vector: H polarization (alternative to E).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_gaussian_source``).
            **kwargs: other GaussianPlaneSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        if radius is None:
            p, *_, height = self._port_frame(port, level, port_offset)
            radius = 0.5 * min(port_size_mult[0] * p.width, port_size_mult[1] * height)
        fields = _given(
            temporal_profile=temporal_profile,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
        )
        return self._at_port(
            fdtdx.GaussianPlaneSource, port, port_size_mult, "gaussian_source", direction=direction,
            level=level, port_offset=port_offset, name=name, wave_character=wave_character, radius=radius * 1e-6,
            std=std, **fields, **kwargs,
        )

    def create_uniform_plane_source_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        amplitude: float = 1.0,
        fixed_E_polarization_vector: tuple[float, float, float] | None = None,
        fixed_H_polarization_vector: tuple[float, float, float] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.UniformPlaneSource, list[PositionConstraint]]:
        """Uniform plane-wave source on a port.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            wave_character: wavelength / frequency.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            amplitude: field amplitude.
            fixed_E_polarization_vector: E polarization (global x, y, z).
            fixed_H_polarization_vector: H polarization (alternative to E).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_uniform_source``).
            **kwargs: other UniformPlaneSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        fields = _given(
            temporal_profile=temporal_profile,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
        )
        return self._at_port(
            fdtdx.UniformPlaneSource, port, port_size_mult, "uniform_source", direction=direction,
            level=level, port_offset=port_offset, name=name, wave_character=wave_character, amplitude=amplitude,
            **fields, **kwargs,
        )

    def create_hard_constant_amplitude_plane_source_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        amplitude: float = 1.0,
        fixed_E_polarization_vector: tuple[float, float, float] | None = None,
        fixed_H_polarization_vector: tuple[float, float, float] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[HardConstantAmplitudePlanceSource, list[PositionConstraint]]:
        """Hard (field-overwriting) constant-amplitude plane source on a port.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            wave_character: wavelength / frequency.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            amplitude: field amplitude.
            fixed_E_polarization_vector: E polarization (global x, y, z).
            fixed_H_polarization_vector: H polarization (alternative to E).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_hard_source``).
            **kwargs: other HardConstantAmplitudePlanceSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        fields = _given(
            temporal_profile=temporal_profile,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
        )
        return self._at_port(
            HardConstantAmplitudePlanceSource, port, port_size_mult, "hard_source", direction=direction,
            level=level, port_offset=port_offset, name=name, wave_character=wave_character, amplitude=amplitude,
            **fields, **kwargs,
        )

    def create_point_dipole_source_at(
        self,
        port: str,
        wave_character: fdtdx.WaveCharacter,
        polarization: int | Literal["x", "y", "z"],
        temporal_profile: fdtdx.TemporalProfile | None = None,
        azimuth_angle: float = 0.0,
        elevation_angle: float = 0.0,
        source_type: Literal["electric", "magnetic"] = "electric",
        amplitude: float = 1.0,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.PointDipoleSource, list[PositionConstraint]]:
        """Point dipole at a port (one cell; a point has no size to scale).

        Args:
            port: port name; the dipole sits at its center, in z at the middle
                of the port's level.
            wave_character: wavelength / frequency.
            polarization: dipole axis, 0 / 1 / 2 or "x" / "y" / "z".
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            azimuth_angle: rotation (degrees) about the vertical axis.
            elevation_angle: rotation (degrees) about the horizontal axis.
            source_type: "electric" or "magnetic".
            amplitude: dipole amplitude.
            level: level whose middle sets the dipole's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_dipole_source``).
            **kwargs: other PointDipoleSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        _, _, _, center, _ = self._port_frame(port, level, port_offset)
        axis = {"x": 0, "y": 1, "z": 2}.get(polarization, polarization)  # type: ignore[call-overload]
        source = fdtdx.PointDipoleSource(
            name=name or f"{port}_dipole_source",
            partial_grid_shape=(1, 1, 1),
            wave_character=wave_character,
            polarization=int(axis),
            azimuth_angle=azimuth_angle,
            elevation_angle=elevation_angle,
            source_type=source_type,
            amplitude=amplitude,
            **_given(temporal_profile=temporal_profile),
            **kwargs,
        )
        return source, self._place_at(source, center)

    # ------------------------------------------------------------ detectors on ports

    def create_mode_overlap_detector_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        wave_characters: Sequence[fdtdx.WaveCharacter],
        mode_index: int = 0,
        filter_pol: Literal["te", "tm"] | None = None,
        scaling_mode: Literal["continuous", "pulse"] | None = None,
        direction: PortDirection = "out",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.ModeOverlapDetector, list[PositionConstraint]]:
        """Mode overlap (mode amplitude) detector on a port.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            wave_characters: wavelengths / frequencies to detect.
            mode_index: mode order (0 = fundamental).
            filter_pol: only "te" or "tm" modes (default: all).
            scaling_mode: "continuous" or "pulse" phasor normalization (default:
                fdtdx's, "continuous"); use "pulse" with pulsed sources.
            direction: "out" measures the mode leaving the component, "in" the
                mode entering it.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_mode_detector``).
            **kwargs: other ModeOverlapDetector fields.

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        fields = _given(filter_pol=filter_pol, scaling_mode=scaling_mode)
        return self._at_port(
            fdtdx.ModeOverlapDetector, port, port_size_mult, "mode_detector", direction=direction,
            level=level, port_offset=port_offset, name=name, wave_characters=tuple(wave_characters),
            mode_index=mode_index, **fields, **kwargs,
        )

    def create_poynting_flux_detector_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        reduce_volume: bool = True,
        keep_all_components: bool = False,
        direction: PortDirection = "out",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.PoyntingFluxDetector, list[PositionConstraint]]:
        """Poynting flux (power vs time) detector on a port.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            reduce_volume: sum the flux over the plane.
            keep_all_components: keep all three flux components.
            direction: "out" counts power leaving the component as positive,
                "in" power entering it.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_flux_detector``).
            **kwargs: other PoyntingFluxDetector fields.

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        _, axis, *_ = self._port_frame(port, level, port_offset)
        return self._at_port(
            fdtdx.PoyntingFluxDetector, port, port_size_mult, "flux_detector", direction=direction,
            level=level, port_offset=port_offset, name=name, reduce_volume=reduce_volume,
            keep_all_components=keep_all_components, fixed_propagation_axis=axis, **kwargs,
        )

    def create_phasor_detector_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        wave_characters: Sequence[fdtdx.WaveCharacter],
        components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
        reduce_volume: bool = False,
        scaling_mode: Literal["continuous", "pulse"] | None = None,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.PhasorDetector, list[PositionConstraint]]:
        """Frequency-domain field (phasor) detector on a port plane.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            wave_characters: wavelengths / frequencies to record.
            components: field components (default: fdtdx's, all six).
            reduce_volume: sum over the plane.
            scaling_mode: "continuous" or "pulse" normalization (default:
                fdtdx's, "continuous").
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_phasor_detector``).
            **kwargs: other PhasorDetector fields.

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        fields = _given(components=None if components is None else tuple(components), scaling_mode=scaling_mode)
        return self._at_port(
            fdtdx.PhasorDetector, port, port_size_mult, "phasor_detector", direction=None, level=level,
            port_offset=port_offset, name=name, wave_characters=tuple(wave_characters), reduce_volume=reduce_volume,
            **fields, **kwargs,
        )

    def create_field_detector_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
        reduce_volume: bool = False,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.FieldDetector, list[PositionConstraint]]:
        """Time-domain field detector on a port plane.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            components: field components (default: fdtdx's, all six).
            reduce_volume: sum over the plane.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_field_detector``).
            **kwargs: other FieldDetector fields.

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        fields = _given(components=None if components is None else tuple(components))
        return self._at_port(
            fdtdx.FieldDetector, port, port_size_mult, "field_detector", direction=None, level=level,
            port_offset=port_offset, name=name, reduce_volume=reduce_volume, **fields, **kwargs,
        )

    def create_energy_detector_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        reduce_volume: bool = False,
        aggregate: str | None = None,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.EnergyDetector, list[PositionConstraint]]:
        """Electromagnetic energy density detector on a port plane.

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            reduce_volume: sum over the plane.
            aggregate: aggregation over time, e.g. "mean" (default: none).
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_energy_detector``).
            **kwargs: other EnergyDetector fields.

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        return self._at_port(
            fdtdx.EnergyDetector, port, port_size_mult, "energy_detector", direction=None, level=level,
            port_offset=port_offset, name=name, reduce_volume=reduce_volume, aggregate=aggregate, **kwargs,
        )

    def create_diffractive_detector_at(
        self,
        port: str,
        port_size_mult: Sequence[float],
        frequencies: Sequence[float],
        orders: Sequence[tuple[int, int]] = ((0, 0),),
        direction: PortDirection = "out",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[DiffractiveDetector, list[PositionConstraint]]:
        """Diffraction-order detector on a port plane (needs periodic boundaries).

        Args:
            port: port name; the plane is centered on it and normal to it.
            port_size_mult: (a, b): the plane is a * port width wide and
                b * level height tall.
            frequencies: frequencies (Hz) to analyze.
            orders: diffraction orders (m, n) to record.
            direction: "out" for orders leaving the component, "in" entering it.
            level: level whose height and middle set the plane's z.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_diffractive_detector``).
            **kwargs: other DiffractiveDetector fields.

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        return self._at_port(
            DiffractiveDetector, port, port_size_mult, "diffractive_detector", direction=direction,
            level=level, port_offset=port_offset, name=name, frequencies=tuple(frequencies), orders=tuple(orders),
            **kwargs,
        )


def _axis_direction(direction: PortDirection, outward: int) -> Literal["+", "-"]:
    """fdtdx direction ("+" / "-") of "in" / "out" at a port.

    Args:
        direction: "in" (into the component) or "out".
        outward: +1 if the port points towards +axis, else -1.

    Returns:
        "+" or "-" along the port's normal axis.

    Raises:
        ValueError: for anything but "in" / "out".
    """
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    return "+" if (outward > 0) == (direction == "out") else "-"


def _given(**kwargs: Any) -> dict[str, Any]:
    """The keyword arguments that are not None (None keeps fdtdx's default)."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _anisotropy(w: Array, e: np.ndarray, ignore: np.ndarray) -> Array:
    """``<eps> <1/eps> - 1`` of the voxels, as a sum over pairs of materials.

    ``a = 1/2 sum_lm w_l w_m (e_l - e_m)^2 / (e_l e_m)``, which equals
    ``<eps> <1/eps> - 1`` where the weights sum to one, and is exactly 0 (with
    a zero derivative) wherever only one material, or materials of one eps, are
    present: the Kottke average ``(1 + n2 a) / <eps>`` then does not depend on
    the normal.

    Args:
        w: (M, nx, ny, nz) material fractions.
        e: (M,) static permittivities.
        ignore: (M,) static, materials left out (never painted).

    Returns:
        (nx, ny, nz) the anisotropy a.
    """
    a = jnp.zeros(w.shape[1:], dtype=w.dtype)
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            g = (e[i] - e[j]) ** 2 / (e[i] * e[j])
            if g != 0 and not (ignore[i] or ignore[j]):
                a = a + float(g) * (w[i] * w[j])
    return a


def _normal_squared(q: Array) -> Array:
    """Squared components (3, nx, ny, nz) of the interface normal of ``q``.

    Uses the mean of the squared forward and backward differences, which (unlike
    central differences) does not vanish where a film thinner than a voxel makes
    ``q`` extremal, and is zero along axes of a single cell (2D simulations).
    The subpixel average uses it only for the variance whose interface flux
    cancels inside the voxel (see :func:`_normal_tensor`), where the voxel's
    own model gives no direction.

    Args:
        q: (nx, ny, nz) mean permittivity.

    Returns:
        (3, nx, ny, nz) squared normal components, summing to one (or zero
        where ``q`` is uniform).
    """
    g2 = []
    for axis in range(3):
        pad = [(1, 1) if a == axis else (0, 0) for a in range(3)]
        d = jnp.diff(jnp.pad(q, pad, mode="edge"), axis=axis)  # n + 1 differences
        n = q.shape[axis]
        forward = jax.lax.slice_in_dim(d, 1, n + 1, axis=axis)
        backward = jax.lax.slice_in_dim(d, 0, n, axis=axis)
        g2.append(0.5 * (forward**2 + backward**2))
    g2 = jnp.stack(g2)
    return g2 / (jnp.sum(g2, axis=0, keepdims=True) + 1e-12)


def _drop_empty_pole_slots(materials: dict[str, fdtdx.Material]) -> dict[str, fdtdx.Material]:
    """Removes the pole slots that no material uses (zero strength everywhere).

    Aligned materials keep a slot for every pole of every material they were
    aligned with; after levels are dropped, slots of their materials would cost
    memory and time in every voxel for nothing.

    Args:
        materials: materials after :func:`align_material_poles`.

    Returns:
        The materials with the used slots only (no dispersion if none is used).
    """
    poles = {k: m.dispersion.poles for k, m in materials.items() if m.dispersion is not None}
    if not poles:
        return materials
    num = len(next(iter(poles.values())))
    live = [s for s in range(num) if any(p[s].coupling_sq != 0 for p in poles.values())]
    if len(live) == num:
        return materials
    out = dict(materials)
    for k in poles:
        m = materials[k]
        out[k] = fdtdx.Material(
            permittivity=m.permittivity,
            permeability=m.permeability,
            electric_conductivity=m.electric_conductivity,
            magnetic_conductivity=m.magnetic_conductivity,
            dispersion=DispersionModel(poles=tuple(poles[k][s] for s in live)) if live else None,
        )
    return out


def pole_slots(materials: dict[str, fdtdx.Material]) -> tuple[Pole, ...]:
    """The shared pole slots of aligned materials.

    Args:
        materials: materials after :func:`align_material_poles`.

    Returns:
        The poles of the slots (empty if no material is dispersive).

    Raises:
        ValueError: if the materials' slots differ (not aligned).
    """
    slots: tuple[Pole, ...] | None = None
    for name, m in materials.items():
        if m.dispersion is None or m.dispersion.num_poles == 0:
            continue
        keys = [(p.omega_0, p.gamma) for p in m.dispersion.poles]
        if slots is None:
            slots = tuple(m.dispersion.poles)
        elif keys != [(p.omega_0, p.gamma) for p in slots]:
            raise ValueError(
                f"Material {name!r} has pole slots {keys} that differ from the other "
                "materials'; align them with align_material_poles"
            )
    return slots or ()


def layer_stack_wafer_layer() -> Any:
    """The WAFER layer of the active PDK.

    Returns:
        The layer, or None if the PDK has none.
    """
    from gdsfactory.pdk import get_layer

    try:
        return get_layer("WAFER")
    except Exception:
        return None


def stack_devices(objects: Any) -> list[StackDevice]:
    """All StackDevices of an fdtdx ObjectContainer."""
    return [o for o in objects.objects if isinstance(o, StackDevice)]


def init_stack_params(objects: Any) -> dict[str, dict[str, Array]]:
    """Initial parameters of all StackDevices.

    Args:
        objects: fdtdx ObjectContainer (from ``fdtdx.place_objects``).

    Returns:
        ``{device name: device params}``; merge it into fdtdx's params.
    """
    return {o.name: o.init_params() for o in stack_devices(objects)}


@timed("apply_stack")
def apply_stack(
    arrays: Any,
    objects: Any,
    params: dict[str, Any],
    key: Array,
    noise_key: Array | None = None,
    apply_roughness: bool = True,
    **transform_kwargs: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """``fdtdx.apply_params`` that also applies every :class:`StackDevice`.

    Stacks are written first, then ``fdtdx.apply_params`` applies the regular
    devices and updates the sources with the final permittivity and pole
    coefficients. ``transform_kwargs`` are forwarded to ``fdtdx.apply_params``.

    In dispersive simulations with mode or plane-wave sources, fdtdx 0.6.2
    builds the sources' dispersive filter with numpy, so this call (like
    ``fdtdx.apply_params``) cannot run inside ``jax.jit``; ``jax.grad`` works.

    Args:
        arrays: fdtdx ArrayContainer.
        objects: fdtdx ObjectContainer.
        params: fdtdx params including the stacks' (:func:`init_stack_params`).
        key: PRNG key for ``fdtdx.apply_params``.
        noise_key: PRNG key for the roughness (default: each device's seed).
        apply_roughness: if False, the nominal geometry.
        **transform_kwargs: forwarded to ``fdtdx.apply_params``.

    Returns:
        (arrays, objects, info) as ``fdtdx.apply_params`` returns them.
    """
    for o in stack_devices(objects):
        arrays = o.write(arrays, params.get(o.name), noise_key, apply_roughness)
    lap("write_stacks", arrays)
    device_params = {d.name: params[d.name] for d in objects.devices}
    out = fdtdx.apply_params(arrays, objects, device_params, key, **transform_kwargs)
    lap("apply_params", out[0])
    return out
