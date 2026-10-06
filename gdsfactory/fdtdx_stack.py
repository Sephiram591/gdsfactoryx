"""Differentiable fdtdx object for a whole component with its LayerStack.

:class:`StackDevice` voxelizes every :class:`~gdsfactory.technology.LayerLevel` of
a LayerStack (logical and derived layers, ``mesh_order`` priorities, sidewall
angle, ``width_to_z``, ``bias``, ``z_to_bias``, background levels) into one
N-material region, differentiable with respect to the polygon vertices of every
source layer and to the thickness and sidewall angle of every level. The device
also builds its simulation volume (:meth:`StackDevice.get_simulation_volume`)
and places sources and detectors on the component's ports
(``StackDevice.create_<source or detector>_at``).

fdtdx ``Device`` objects only mix two materials and :func:`fdtdx.apply_params`
only knows about them, so the stack is applied with :func:`apply_stack`, which
writes the stack into the arrays and then calls ``fdtdx.apply_params`` (regular
devices and sources are updated as usual, after the stack).

Geometry (lengths in grid cells inside, gds/stack units are um):

- each source layer gets a 2D signed distance field (see
  :func:`gdsfactory.fdtdx_geometry.signed_distance`); derived layers combine them
  with ``or = min``, ``and = max``, ``not = max(a, -b)``; layers that cover the
  whole wafer (``fill_layers``, default WAFER) and background levels are
  everywhere inside;
- a level at height z offsets its distance by
  ``(z - z_ref) * tan(sidewall_angle) - bias - z_to_bias(z)`` with
  ``z_ref = zmin + width_to_z * thickness``; in every voxel each level is the
  half-plane given by its distance and gradient at the voxel center, and the
  levels are painted by increasing ``mesh_order`` (lower wins) by clipping the
  voxel square exactly, and exactly along z between the (possibly rough) level
  surfaces; the background material gets the rest;
- :class:`TopRoughness` displaces the top surface of a level; dips are filled
  by the levels deposited on it (whose bottom follows) or by whatever level
  covers that height;
- the permittivity of a voxel is the Kottke subpixel average along the local
  interface normal, estimated from one-sided differences of the mean eps_inf so
  that films thinner than a voxel and 2D simulations work
  (``averaging="subpixel"``, the default; needs diagonally anisotropic arrays,
  which the device requests), or the inverse-permittivity average of its
  materials (``averaging="inverse"``, fdtdx's convention). With 3-component
  arrays each E component is sampled at its own Yee position
  (``yee_staggered``). Electric conductivity is averaged linearly.

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

import math
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, Self

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.dispersion import Pole, compute_pole_coefficients
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

from gdsfactory._jax import to_numpy
from gdsfactory.fdtdx_geometry import (
    _edge_topology,
    align_material_poles,
    gaussian_random_field,
    pack_polygons,
    signed_distance,
    split_t_junctions,
    warn_reversible_losses,
)

Array = jax.Array
Averaging = Literal["inverse", "subpixel"]
PortDirection = Literal["in", "out"]  # into the component through the port, or out of it

__all__ = [
    "TopRoughness",
    "StackDevice",
    "apply_stack",
    "init_stack_params",
]

_BIG = 1e6  # "infinitely" far, in cells
_SLAB_MARGIN = 0.05  # um, default z margin voxelized around each level
_ANISOTROPY_SENTINEL = "__diagonal_anisotropy__"
# offsets (cells) from a voxel center to fdtdx's Ex, Ey, Ez Yee positions:
# Ex at (i + 1/2, j, k), Ey at (i, j + 1/2, k), Ez at (i, j, k + 1/2)
_YEE_E_SHIFTS = ((0.0, -0.5, -0.5), (-0.5, 0.0, -0.5), (-0.5, -0.5, 0.0))


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


def _extension_frame(center: Any, width: Any, orientation: Any, length: float, xp: Any) -> Any:
    """Corners (4, 2) of a straight waveguide leaving a port, counter-clockwise.

    Corners 0 and 3 are on the port face, 1 and 2 at the far end, ``length``
    along the port's orientation.
    """
    angle = xp.deg2rad(orientation)
    n = xp.stack([xp.cos(angle), xp.sin(angle)])  # outward, along the port's orientation
    t = xp.stack([-xp.sin(angle), xp.cos(angle)])
    a = xp.asarray(center) - 0.5 * width * t
    b = xp.asarray(center) + 0.5 * width * t
    return xp.stack([a, a + length * n, b + length * n, b])


def _extension_corners(
    port: _Port, length: float, vertices: np.ndarray, tol: float = 1e-3
) -> tuple[np.ndarray, tuple[int, int]]:
    """Static corners of a port extension, with its face corners matched to polygon vertices.

    Args:
        port: the port.
        length: extension length (um).
        vertices: (E, 2) packed vertices (um) of the port's layer.
        tol: distance (um) within which a face corner is a polygon vertex.

    Returns:
        The (4, 2) corners (face corners replaced by the matched vertices, so the
        extension and the waveguide share their edge exactly) and the indices of
        the matched vertices for the two face corners (-1 if none).
    """
    corners = _extension_frame(np.array([port.x, port.y]), port.width, port.orientation, length, np)
    matched = []
    for face, far in ((0, 1), (3, 2)):
        i = -1
        if len(vertices):
            d = np.linalg.norm(vertices - corners[face], axis=1)
            if d.min() <= tol:
                i = int(np.argmin(d))
                corners[far] = vertices[i] + corners[far] - corners[face]
                corners[face] = vertices[i]
        matched.append(i)
    return corners, (matched[0], matched[1])


def _extension_corners_traced(port: Any, length: float) -> Array:
    """Corners (4, 2) of a port extension from a (possibly traced) port."""
    center = jnp.stack([jnp.asarray(c, dtype=float) for c in port.center])
    return _extension_frame(center, jnp.asarray(port.width, dtype=float), jnp.asarray(port.orientation, dtype=float), length, jnp)


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

    Args:
        layer: the layer.

    Returns:
        The grow in um.

    Raises:
        NotImplementedError: for different x and y sizings.
    """
    from gdsfactory import kcl

    xs = tuple(layer.sizings_xoffsets)
    ys = tuple(layer.sizings_yoffsets)
    if xs != ys:
        raise NotImplementedError(f"Anisotropic layer sizing {xs} != {ys} is not supported")
    return float(sum(xs)) * float(kcl.dbu)


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


def _layer_key(index: int) -> str:
    """Parameter name ``vertices/<layer name>`` (or the index if it has no name)."""
    from gdsfactory.pdk import get_layer_name

    try:
        return f"vertices/{get_layer_name(index)}"
    except Exception:
        return f"vertices/{index}"


def _clip(poly: Array, count: Array, n: Array, s: Array) -> tuple[Array, Array]:
    """Clips convex polygons to the half-planes ``n . p + s >= 0``.

    Args:
        poly: (..., V, 2) vertices; the first ``count`` are valid, the rest repeat
            the last valid one.
        count: (...,) number of valid vertices.
        n: (..., 2) normals.
        s: (...,) offsets.

    Returns:
        The clipped polygons (..., V, 2) in the same layout and their vertex counts.
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
    mask = jnp.stack([keep, cross], axis=-1).reshape(*poly.shape[:-2], 2 * num)
    cum = jnp.cumsum(mask, axis=-1)
    new_count = cum[..., -1]
    # slot j takes the (j+1)-th kept candidate; later slots repeat the last one
    # (an empty result still gathers in bounds: out-of-bounds gathers fill NaN,
    # which the masked area ignores but its derivative does not)
    j = jnp.minimum(i, jnp.maximum(new_count[..., None] - 1, 0))
    src = jnp.minimum(jnp.sum(cum[..., None, :] <= j[..., :, None], axis=-1), 2 * num - 1)
    return jnp.take_along_axis(cand, src[..., None], axis=-2), new_count


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


def _composite(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    half: float = 0.5,
) -> Array:
    """Volume fraction of each level (and the background) per voxel.

    Along z the voxel is split at the heights of all level surfaces (surfaces
    are clamped to the voxel, so a surface on a voxel boundary contributes half
    its derivative to each neighbor). On each sub-interval, the levels present
    there are painted in priority order. Laterally each level is a half-plane
    ``{p: n . p + offset < 0}`` in voxel coordinates (from its signed distance and
    gradient at the voxel center): level j gets the area of the voxel square
    inside it but outside every earlier present level, computed exactly by
    clipping the square with the complements in priority order. Levels that meet
    at a seam (complementary half-planes) therefore tile the voxel, nested levels
    nest, and slanted straight edges get exact areas. Fill levels cover the
    whole square.

    Args:
        offsets: (nx, ny, nz, K) signed distance of each level at the voxel
            centers (cells, negative inside); +-BIG for absent / fill.
        normals: (nx, ny, K, 2) unit in-plane gradient of each level's distance.
        is_fill: (K,) static, whether a level fills the whole plane.
        lows: (nx, ny, K) bottom surface per level (cells).
        highs: (nx, ny, K) top surface per level (cells).
        half: half side of the lateral square (cells); < 0.5 for sub-samples.

    Returns:
        (nx, ny, nz, K + 1): K levels then the background, summing to one.
    """
    nz = offsets.shape[2]
    num_levels = lows.shape[-1]
    num_poly = sum(not f for f in is_fill)
    num_vertices = 4 + num_poly
    square = jnp.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=lows.dtype)
    square = jnp.concatenate([square, jnp.repeat(square[-1:], num_vertices - 4, axis=0)])
    full_area = (2 * half) ** 2

    @jax.checkpoint
    def one_slice(inp: tuple[Array, Array]) -> Array:
        """Fractions (nx, ny, K + 1) of the voxel layer k from (k, offsets at k)."""
        k, off = inp  # off: (nx, ny, K)
        lo = jnp.clip(lows, k, k + 1.0)
        hi = jnp.clip(highs, k, k + 1.0)
        ends = jnp.broadcast_to(jnp.asarray([k, k + 1.0], dtype=lows.dtype), (*lows.shape[:2], 2))
        points = jnp.concatenate([ends, lo, hi], axis=-1)  # (nx, ny, 2K+2)
        # membership of sub-interval m (between sorted points m and m+1) comes from
        # the ranks of each level's surfaces in a stable sort, not from midpoints:
        # with tied surfaces every level keeps the sub-interval on its own side,
        # so moving a surface always changes that level's coverage
        perm = jnp.argsort(jax.lax.stop_gradient(points), axis=-1, stable=True)
        bp = jnp.take_along_axis(points, perm, axis=-1)
        rank = jnp.argsort(perm, axis=-1, stable=True)
        r_lo = rank[..., 2 : 2 + num_levels]
        r_hi = rank[..., 2 + num_levels :]
        length = bp[..., 1:] - bp[..., :-1]  # (nx, ny, S)
        m = jnp.arange(length.shape[-1])
        inside = (m >= r_lo[..., None]) & (m < r_hi[..., None])  # (nx, ny, K, S)

        batch = length.shape
        poly = jnp.broadcast_to(square, (*batch, num_vertices, 2))
        count = jnp.full(batch, 4, dtype=jnp.int32)
        area = jnp.ones(batch, dtype=lows.dtype)  # uncovered fraction so far
        visible = []
        for j in range(num_levels):
            present = inside[..., j, :]
            if is_fill[j]:
                visible.append(jnp.sum(length * jnp.where(present, area, 0.0), axis=-1))
                area = jnp.where(present, 0.0, area)
                count = jnp.where(present, 0, count)
                continue
            n = jnp.broadcast_to(normals[:, :, j, None, :], (*batch, 2))
            # beyond the square's half-diagonal a line keeps or removes it whole
            o = jnp.broadcast_to(jnp.clip(off[:, :, j, None], -2 * half, 2 * half), batch)
            clipped, c_count = _clip(poly, count, n, o)  # keep the part outside level j
            c_area = _area(clipped, c_count) / full_area
            visible.append(jnp.sum(length * jnp.where(present, area - c_area, 0.0), axis=-1))
            poly = jnp.where(present[..., None, None], clipped, poly)
            count = jnp.where(present, c_count, count)
            area = jnp.where(present, c_area, area)
        visible.append(jnp.sum(length * area, axis=-1))
        return jnp.stack(visible, axis=-1)  # (nx, ny, K + 1)

    ks = jnp.arange(nz, dtype=lows.dtype)
    out = jax.lax.map(one_slice, (ks, jnp.moveaxis(offsets, 2, 0)))  # (nz, nx, ny, K+1)
    return jnp.moveaxis(out, 0, 2)


# ---------------------------------------------------------------- roughness


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
            "thickness/<level>": (),
            "sidewall_angle/<level>": (),
        }

    The device box is overwritten on every :func:`apply_stack` (permittivity,
    conductivity and, in dispersive simulations, all pole coefficients):
    everything not covered by a level is ``background``. fdtdx Devices are
    applied afterwards and win where they overlap.

    Attributes:
        levels: compiled levels in the box.
        source_layers: (parameter key, layer index) per source layer with polygons.
        source_vertices: initial packed vertices (um) per source layer.
        source_sizes: vertices per polygon, per source layer.
        background: material where no level is.
        center: gds point (um) at the xy center of the box.
        z_center: stack height (um) at the z center of the box.
        source_extensions: per source layer, (port, length, face vertex a,
            face vertex b) of the port extensions it ends with.
        ports: the component's ports.
        roughness: top-surface roughness per level.
        noise_seed: roughness seed when no ``noise_key`` is passed.
        averaging: "subpixel" (Kottke) or "inverse".
        yee_staggered: sample each E component at its own Yee position.
        slab_margin: z margin (um) voxelized around each level.
        edge_width: lateral width (cells) of the voxel square used for areas.
        xy_subsamples: lateral sub-samples per voxel and axis.
        edge_chunk: edges per step of the distance computation.
    """

    #: compiled levels (see :meth:`from_layer_stack`)
    levels: tuple[_Level, ...] = frozen_field()
    #: (param key, layer index) per source layer with polygons
    source_layers: tuple[tuple[str, int], ...] = frozen_field()
    #: initial packed vertices (um) per source layer
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
    #: polygons end the layer's vertices; a / b index the layer's vertices that the
    #: extension's face corners copy (-1: taken from the port)
    source_extensions: tuple[tuple[tuple[str, float, int, int], ...], ...] = frozen_field(default=())
    #: the component's ports, for the ``create_*_at`` methods
    ports: tuple[_Port, ...] = frozen_field(default=())
    #: top-surface roughness per level, sampled with ``noise_key``
    roughness: Sequence["TopRoughness"] = field(default=())
    #: seed used for the roughness when no ``noise_key`` is passed
    noise_seed: int = frozen_field(default=0)
    #: "subpixel" (Kottke, along interface normals) or "inverse" (fdtdx convention)
    averaging: Averaging = frozen_field(default="subpixel")
    #: with 3-component permittivity arrays (e.g. ``averaging="subpixel"``),
    #: sample each E component's materials at its own Yee position (half a cell
    #: apart); otherwise all components use the voxel center, a first-order error
    yee_staggered: bool = frozen_field(default=True)
    #: z margin (um) around each level's nominal extent that is voxelized; must
    #: cover thickness changes and roughness
    slab_margin: float = frozen_field(default=_SLAB_MARGIN)
    #: lateral width of the voxel square used for the areas, in cells
    edge_width: float = frozen_field(default=1.0)
    #: in-plane sub-samples per voxel and axis: the lateral fill is averaged over
    #: s x s sub-squares, which makes it closer to the exact area fraction at
    #: polygon corners and narrow tips (cost ~ s^2)
    xy_subsamples: int = frozen_field(default=1)
    #: edges per step in the signed distance computation
    edge_chunk: int = frozen_field(default=64)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_layer_stack(
        cls,
        component: Any,
        layer_stack: Any,
        materials: dict[str, fdtdx.Material],
        background: str,
        *,
        included_layers: Sequence[str],
        extend_ports: Sequence[tuple[str, float]] = (),
        boundary_offset: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        levels: Sequence[str] | None = None,
        fill_layers: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Builds the device and its box from a component and a LayerStack.

        ``extend_ports`` first adds a straight waveguide to each listed port. The
        box then starts as the xy bounding box of the polygons on the stack's
        layers (extensions included; all polygons if the levels only fill the
        plane) and the z range of ``included_layers``, and ``boundary_offset``
        moves its faces without changing the geometry. Every level that overlaps
        the final box (or comes within ``slab_margin`` of it) is drawn, also
        levels outside ``included_layers``, unless no polygon is on its layers.

        Args:
            component: the component; polygons and ports are read with gradients
                stopped.
            layer_stack: a :class:`~gdsfactory.technology.LayerStack`.
            materials: fdtdx materials by ``LayerLevel.material``; needed for
                ``background`` and every level that is drawn.
            background: material (key of ``materials``) where no level is.
            included_layers: level names; the lowest bottom and the highest top
                among them are the initial z faces of the box.
            extend_ports: ``((port, length), ...)`` in um. Each port gets a
                straight waveguide of the port's width on the port's layer,
                leaving the component along the port's orientation (opposite to
                the direction light is injected through the port). It is a
                polygon like the others: every level drawing that layer draws it,
                with its thickness, sidewall angle and bias, and it follows the
                port when the component's parameters change.
            boundary_offset: (x+, x-, y+, y-, z+, z-) in um. Moves each face of the
                box along its axis, positive towards +x, +y or +z for both faces
                of an axis (so (1, -1, ...) grows the box by 1 um on each side
                in x). The geometry does not change: the box shows more or less
                of the same layer stack.
            levels: levels that may be drawn (default: every level with a
                material and nonzero thickness).
            fill_layers: layers that cover the whole plane (default: WAFER).
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
                length <= 0, a port layer no level draws), an empty box, or a
                component without polygons.
            TypeError: a box field passed in ``kwargs``.
        """
        from gdsfactory.pdk import get_layer

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

        ports = []
        for port in component.ports:
            orientation = port.orientation
            ports.append(
                _Port(
                    name=str(port.name),
                    x=float(to_numpy(port.x)),
                    y=float(to_numpy(port.y)),
                    orientation=None if orientation is None else float(to_numpy(orientation)),
                    width=0.0 if port.width is None else float(to_numpy(port.width)),
                    layer=int(get_layer(port.layer)),
                )
            )

        # static polygons per layer (um), then the port extensions on their layers
        polygons = {
            index: [np.asarray(to_numpy(p), dtype=float).reshape(-1, 2) for p in polys]
            for index, polys in component.get_polygons_points().items()
        }
        extensions: dict[int, list[tuple[str, float, int, int]]] = {}
        rectangles: dict[int, list[np.ndarray]] = {}
        by_name = {p.name: p for p in ports}
        for name, length in extend_ports:
            if name not in by_name:
                raise KeyError(f"Unknown port {name!r} in extend_ports; the component has {sorted(by_name)}")
            p = by_name[name]
            if p.orientation is None or p.width <= 0 or not float(length) > 0:
                raise ValueError(f"Cannot extend port {name!r}: needs an orientation, a width and a length > 0")
            if p.layer not in stack_layers:
                raise ValueError(f"Port {name!r} is on layer {p.layer}, which no level of the stack draws")
            own = np.concatenate(polygons[p.layer]) if polygons.get(p.layer) else np.zeros((0, 2))
            corners, (ia, ib) = _extension_corners(p, float(length), own)
            extensions.setdefault(p.layer, []).append((name, float(length), ia, ib))
            rectangles.setdefault(p.layer, []).append(corners)

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

        # levels just outside the box count too: fdtdx's Yee samples at the
        # bottom face average half a cell below it
        margin = float(kwargs.get("slab_margin", _SLAB_MARGIN))
        nonempty = {index for index, polys in polygons.items() if polys} | set(rectangles)
        compiled = []
        for name, level, expr in candidates:
            lo, hi = _span(level)
            if hi <= z0 - margin or lo >= z1 + margin:
                continue  # outside the box
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
        missing = [n for index, ext in extensions.items() if index not in layer_ids for n, *_ in ext]
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

        return cls(
            materials=align_material_poles(mats),
            levels=tuple(compiled),
            source_layers=tuple(sources),
            source_vertices=tuple(verts),
            source_sizes=tuple(sizes),
            source_extensions=tuple(exts),
            background=background,
            partial_real_shape=((x1 - x0) * 1e-6, (y1 - y0) * 1e-6, (z1 - z0) * 1e-6),
            center=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
            z_center=0.5 * (z0 + z1),
            ports=tuple(ports),
            **kwargs,
        )

    # ------------------------------------------------------------ parameters

    def init_params(self, key: Array | None = None) -> dict[str, Array]:
        """Initial parameters: the component's vertices and the stack's thicknesses and angles.

        Args:
            key: unused (the parameters are deterministic).

        Returns:
            ``{"vertices/<layer>": (E, 2), "thickness/<level>": (), "sidewall_angle/<level>": ()}``.
        """
        del key
        p: dict[str, Array] = {}
        for (k, _), v in zip(self.source_layers, self.source_vertices, strict=True):
            p[k] = jnp.asarray(v, dtype=jnp.float32)
        for lv in self.levels:
            p[f"thickness/{lv.name}"] = jnp.asarray(lv.thickness, dtype=jnp.float32)
            p[f"sidewall_angle/{lv.name}"] = jnp.asarray(lv.sidewall_angle, dtype=jnp.float32)
        return p

    def pack(self, component: Any) -> dict[str, Array]:
        """Vertex parameters of a (possibly traced) component, port extensions included.

        Args:
            component: a component with the same polygon topology (number of
                polygons and vertices per polygon on every source layer) and the
                extended ports.

        Returns:
            ``{"vertices/<layer>": (E, 2)}``, differentiable with respect to the
            parameters the component was built from; the extensions follow their
            ports.

        Raises:
            ValueError: if the topology changed (rebuild the device then).
        """
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
                    f"{own}, got {len(s)} with {s}. Rebuild the StackDevice."
                )
            v = pack_polygons(polys)[0] if polys else jnp.zeros((0, 2))
            rects = []
            for name, length, ia, ib in exts:
                port = component.ports[name]
                corners = _extension_corners_traced(port, length)
                if ia >= 0:  # the face corners copy the polygon's own vertices: an exact seam
                    corners = corners.at[0].set(v[ia]).at[1].set(v[ia] + corners[1] - corners[0])
                if ib >= 0:
                    corners = corners.at[3].set(v[ib]).at[2].set(v[ib] + corners[2] - corners[3])
                rects.append(corners.astype(v.dtype))
            out[k] = jnp.concatenate([v, *rects]) if rects else v
        return out

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

    def _source_sdfs(
        self, params: dict[str, Array], shift: tuple[float, float] = (0.0, 0.0)
    ) -> dict[int, Field]:
        """Signed distance of every source layer at the voxel centers.

        Partly shared junction edges are split first (static topology), so
        abutting polygons, such as a port extension and its waveguide, join
        without a seam.

        Args:
            params: device parameters (``vertices/<layer>`` in um).
            shift: offset (cells) of the sample points from the voxel centers.

        Returns:
            (distance, gx, gy) per source layer index, (nx, ny) each, in cells,
            negative inside; (gx, gy) is the gradient with respect to the point.
        """
        nx, ny, _ = self.grid_shape
        res_um = self._res_um()
        offset = np.array([nx / 2, ny / 2])
        center = np.asarray(self.center, dtype=float)
        xx, yy = jnp.meshgrid(
            jnp.arange(nx) + 0.5 + shift[0], jnp.arange(ny) + 0.5 + shift[1], indexing="ij"
        )
        sdf = {}
        for (k, index), v0, sizes in zip(
            self.source_layers, self.source_vertices, self.source_sizes, strict=True
        ):
            topology = (np.asarray(v0, dtype=float) - center) / res_um + offset
            # partly shared junction edges -> exactly shared seams (static)
            split, split_sizes = split_t_junctions(topology, sizes)
            v = ((jnp.asarray(params[k]) - center) / res_um + offset)[split]
            nxt, ids, boundary = _edge_topology(topology[split], split_sizes)
            points = jnp.stack([xx.ravel(), yy.ravel()], axis=1).astype(v.dtype)

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
        p = {**self.init_params(), **(params or {})}
        nx, ny, nz = self.grid_shape
        res_um = self._res_um()
        z0 = self.z_center - 0.5 * nz * res_um  # stack z at the device bottom
        offsets_z = self.interface_offsets(noise_key) if apply_roughness and self.roughness else []

        names = compute_ordered_names(self.materials)
        dtype = self._config.dtype
        margin = self.slab_margin / res_um
        order = sorted(range(len(self.levels)), key=lambda i: (self.levels[i].mesh_order, i))
        tops = {lv.name: max(lv.zmin, lv.zmin + lv.thickness) for lv in self.levels}
        active = []  # (level, k0, k1, lateral z offset or None for fills)
        lows, highs = [], []
        for i in order:
            lv = self.levels[i]
            zb = (lv.zmin - z0) / res_um - shift[2]
            t_nominal = lv.thickness / res_um
            k0 = max(0, math.floor(min(zb, zb + t_nominal) - margin))
            k1 = min(nz, math.ceil(max(zb, zb + t_nominal) + margin))
            if k0 >= k1:
                continue  # level outside the device
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
                elif r.fill_above and abs(lo_nominal - tops[r.level]) < 1e-6:
                    z_lo = z_lo + dz

            shift_z = None
            if lv.expr[0] != "fill":
                # lateral offset of the level's boundary at the voxel-center heights
                zc = jnp.arange(k0, k1) + 0.5
                z_ref = zb + lv.width_to_z * t
                grow = lv.bias / res_um
                if lv.z_to_bias is not None:
                    frac = (zc - zb) / t
                    zs, bs = lv.z_to_bias
                    grow = grow + jnp.interp(frac, jnp.asarray(zs), jnp.asarray(bs)) / res_um
                shift_z = jnp.broadcast_to((zc - z_ref) * jnp.tan(jnp.deg2rad(angle)) - grow, zc.shape)
            active.append((lv, k0, k1, shift_z))
            lows.append(z_lo.astype(dtype))
            highs.append(z_hi.astype(dtype))

        if not active:
            w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype)
            return names, w.at[names.index(self.background)].set(1.0)
        is_fill = tuple(lv.expr[0] == "fill" for lv, *_ in active)
        level_materials = [lv.material for lv, *_ in active]

        n_sub = self.xy_subsamples
        sub = [(k + 0.5) / n_sub - 0.5 for k in range(n_sub)]

        def fractions(sdf: dict[int, Field]) -> Array:
            """(nx, ny, nz, K + 1) fractions of the levels, then the background."""
            offsets, normals = [], []
            for (lv, k0, k1, shift_z), fill in zip(active, is_fill, strict=True):
                o = jnp.full((nx, ny, nz), _BIG, dtype=dtype)  # outside the slab: absent
                if fill:
                    offsets.append(o.at[:, :, k0:k1].set(-_BIG))
                    normals.append(jnp.zeros((nx, ny, 2), dtype=dtype))
                    continue
                sd, gx, gy = _eval_expr(lv.expr, sdf, (nx, ny), res_um)
                lateral = (sd[:, :, None] + shift_z[None, None, :]) * (n_sub / self.edge_width)
                offsets.append(o.at[:, :, k0:k1].set(lateral.astype(dtype)))
                norm2 = gx**2 + gy**2
                ok = norm2 > 1e-12  # zero where no layer is near (BIG distances)
                scale = jax.lax.rsqrt(jnp.where(ok, norm2, 1.0))
                normal = jnp.stack([jnp.where(ok, gx * scale, 1.0), jnp.where(ok, gy * scale, 0.0)], axis=-1)
                normals.append(normal.astype(dtype))
            return _composite(
                jnp.stack(offsets, axis=-1),  # (nx, ny, nz, K), priority order
                jnp.stack(normals, axis=-2),  # (nx, ny, K, 2)
                is_fill,
                jnp.stack(lows, axis=-1),  # (nx, ny, K)
                jnp.stack(highs, axis=-1),
            )

        weights = 0.0
        for a in sub:
            for b in sub:
                weights = weights + fractions(self._source_sdfs(p, (shift[0] + a, shift[1] + b)))
        weights = weights / n_sub**2
        out = []
        for m in names:
            cols = [j for j, lm in enumerate(level_materials) if lm == m]
            if m == self.background:
                cols.append(len(level_materials))
            out.append(weights[..., cols].sum(axis=-1) if cols else jnp.zeros((nx, ny, nz), dtype))
        return names, jnp.stack(out)

    # ------------------------------------------------------------ fdtdx hooks

    def place_on_grid(
        self, grid_slice_tuple: Any, config: SimulationConfig, key: Array
    ) -> Self:
        """fdtdx hook: aligns the pole slots and requests anisotropic arrays for subpixel averaging.

        Args:
            grid_slice_tuple: the device's grid slices.
            config: simulation config.
            key: PRNG key.

        Returns:
            The placed device.
        """
        # aligned pole slots also for devices built without from_layer_stack;
        # fdtdx allocates the slots from the placed objects' materials
        mats = {k: v for k, v in self.materials.items() if k != _ANISOTROPY_SENTINEL}
        mats = align_material_poles(mats)
        if self.averaging == "subpixel":
            # makes fdtdx allocate diagonally anisotropic arrays (never painted)
            mats[_ANISOTROPY_SENTINEL] = fdtdx.Material(permittivity=(1.0, 1.0, 2.0))
        self = self.aset("materials", mats)
        warn_reversible_losses(mats, config, self.name, conductivity=True)
        return super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)

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
        if self.averaging == "subpixel" and n_comp != 3:
            raise ValueError(
                "averaging='subpixel' needs diagonally anisotropic arrays; build the "
                "device with from_layer_stack(..., averaging='subpixel')"
            )
        dtype = arrays.inv_permittivities.dtype
        iso = {"isotropic": n_comp == 1, "diagonally_anisotropic": n_comp == 3}
        eps = jnp.asarray(compute_allowed_permittivities(self.materials, **iso), dtype=dtype)
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
        inv, cond, w_cell = [], [], 0.0
        for c in range(n_comp):
            if c < len(shifts):
                w = self.material_weights(params, noise_key, apply_roughness, shifts[c])[1]
                w = w.astype(dtype)
                w_cell = w_cell + w / len(shifts)  # cell value for the isotropic quantities
                if sigma is not None and sigma.shape[1] == len(shifts):
                    cond.append(jnp.sum(w * sigma[:, c, None, None, None], axis=0))
            e = eps[:, c, None, None, None]
            mean_inv = jnp.sum(w / e, axis=0)
            if self.averaging == "subpixel":
                mean_eps = jnp.sum(w * e, axis=0)
                n2 = _normal_squared(mean_eps)[c]
                mean_inv = n2 * mean_inv + (1 - n2) / mean_eps
            inv.append(mean_inv)
        arrays = arrays.at["inv_permittivities"].set(
            arrays.inv_permittivities.at[:, *self.grid_slice].set(jnp.stack(inv).astype(dtype))
        )
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
        if arrays.dispersive_c1 is not None:
            arrays = self._write_dispersion(arrays, w_cell)
        return arrays

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


def _normal_squared(q: Array) -> Array:
    """Squared components (3, nx, ny, nz) of the interface normal of ``q``.

    Uses the mean of the squared forward and backward differences, which (unlike
    central differences) does not vanish where a film thinner than a voxel makes
    ``q`` extremal, and is zero along axes of a single cell (2D simulations).

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
    device_params = {d.name: params[d.name] for d in objects.devices}
    return fdtdx.apply_params(arrays, objects, device_params, key, **transform_kwargs)
