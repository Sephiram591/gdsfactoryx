"""Shared geometry and dispersion helpers for :mod:`gdsfactory.fdtdx_stack`.

- :func:`pack_polygons` and :func:`signed_distance`: a pure-JAX (jit-safe)
  signed distance to the union of a layer's polygons, differentiable with
  respect to the vertices. Each polygon is filled by the nonzero winding rule,
  overlapping polygons merge without seams, and KLayout hole cut lines and seams
  of abutting polygons are not treated as boundaries. :func:`split_t_junctions`
  makes partly shared junction edges (a narrower taper on a waveguide, a taper
  on the side of an MMI) exactly shared, so they are seams too.
- :func:`gaussian_random_field`: Gaussian random fields for surface roughness.
- :func:`align_material_poles` / :class:`SlotPole`: give all materials the same
  dispersive pole slots, so that mixing materials in a voxel mixes only the pole
  strengths (exact susceptibility mixing in fdtdx >= 0.6.2).
- :func:`warn_reversible_losses`: warns when fdtdx's reversible gradients meet
  lossy materials.
"""

import math
import warnings
from collections.abc import Sequence
from typing import Any

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.dispersion import DispersionModel, Pole

Array = jax.Array

__all__ = [
    "SlotPole",
    "align_material_poles",
    "gaussian_random_field",
    "pack_polygons",
    "signed_distance",
    "split_t_junctions",
    "warn_reversible_losses",
]


# ---------------------------------------------------------------- polygons


def pack_polygons(polygons: Sequence[Any]) -> tuple[Array, tuple[int, ...]]:
    """Concatenates (N_i, 2) vertex arrays into one (sum N_i, 2) array.

    Returns:
        vertices: the packed (differentiable) vertices.
        sizes: number of vertices per polygon (the static topology).
    """
    polys = [jnp.asarray(p) for p in polygons]
    if not polys:
        raise ValueError("Need at least one polygon")
    for p in polys:
        if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
            raise ValueError(f"Polygons must be (N >= 3, 2) arrays, got {p.shape}")
    return jnp.concatenate(polys, axis=0), tuple(int(p.shape[0]) for p in polys)


def split_t_junctions(
    vertices: np.ndarray, sizes: Sequence[int], tol: float = 1e-6
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Inserts into every edge the polygon vertices that lie on it (T-junctions).

    Where a polygon corner lies inside another edge, e.g. a taper end in the
    middle of the side of an MMI, the two polygons share only part of that edge,
    so the shared part is not recognized as a seam. Splitting the long edge at
    the corner makes the shared part an exactly coincident (reversed) edge, which
    :func:`_edge_topology` then excludes from the distance. Inserted points lie
    on their edges, so the filled region does not change.

    Static (numpy, from concrete vertices); the inserted points copy vertices, so
    with ``vertices[index]`` they follow those vertices when they move.

    Args:
        vertices: (E, 2) packed polygon vertices.
        sizes: vertices per polygon.
        tol: distance tolerance (same units as ``vertices``).

    Returns:
        index: indices into ``vertices`` of the split polygons' vertices.
        sizes: vertices per split polygon.
    """
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=float)
    n = int(sum(sizes))
    starts = np.cumsum([0, *sizes[:-1]])
    nxt = np.arange(1, n + 1)
    for start, size in zip(starts, sizes, strict=True):
        nxt[start + size - 1] = start
    a, b = vertices, vertices[nxt]
    ab = b - a
    length = np.linalg.norm(ab, axis=1)
    tree = cKDTree(vertices)
    candidates = tree.query_ball_point(0.5 * (a + b), r=0.5 * length + tol)
    inserts: list[list[int]] = [[] for _ in range(n)]
    for e in range(n):
        if length[e] <= 2 * tol:
            continue
        idx = np.array([i for i in candidates[e] if i != e and i != nxt[e]], dtype=int)
        if idx.size == 0:
            continue
        ap = vertices[idx] - a[e]
        t = ap @ ab[e] / length[e] ** 2
        dist = np.abs(ap[:, 0] * ab[e, 1] - ap[:, 1] * ab[e, 0]) / length[e]
        inner = (dist < tol) & (t * length[e] > tol) & ((1 - t) * length[e] > tol)
        if not inner.any():
            continue
        # one inserted point per distinct position, ordered along the edge
        found = sorted(zip(t[inner], idx[inner], strict=True))
        kept: list[tuple[float, int]] = []
        for ti, i in found:
            if not kept or (ti - kept[-1][0]) * length[e] > tol:
                kept.append((ti, int(i)))
        inserts[e] = [i for _, i in kept]
    index, new_sizes = [], []
    for start, size in zip(starts, sizes, strict=True):
        poly = []
        for v in range(start, start + size):
            poly.append(v)
            poly.extend(inserts[v])
        index.extend(poly)
        new_sizes.append(len(poly))
    return np.asarray(index, dtype=int), tuple(new_sizes)


def _edge_topology(
    vertices: np.ndarray, sizes: Sequence[int], tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static edge structure: end index, polygon id and distance mask per edge.

    Edges that coincide with another edge whose polygon lies on the other side
    (KLayout hole cut lines, seams of abutting polygons, whatever the polygons'
    orientations) are excluded from the distance: they are not part of the
    boundary of the union. They are kept for the winding number.
    """
    n = int(sum(sizes))
    if vertices.shape != (n, 2):
        raise ValueError(f"vertices {vertices.shape} do not match polygon sizes (sum {n})")
    nxt = np.arange(1, n + 1)
    ids = np.empty(n, dtype=np.int32)
    start = 0
    for i, size in enumerate(sizes):
        nxt[start + size - 1] = start
        ids[start : start + size] = i
        start += size
    q = np.round(vertices / tol).astype(np.int64)
    a = [tuple(v) for v in q]
    b = [tuple(v) for v in q[nxt]]
    # direct every edge as if its polygon were counter-clockwise (filled side on
    # the left): an edge is then a seam if the reversed edge exists
    v = np.asarray(vertices, dtype=float)
    cross = v[:, 0] * v[nxt, 1] - v[nxt, 0] * v[:, 1]
    clockwise = np.bincount(ids, weights=cross, minlength=len(sizes))[ids] < 0
    directed = [(eb, ea) if cw else (ea, eb) for ea, eb, cw in zip(a, b, clockwise, strict=True)]
    edges = set(directed)
    boundary = np.array([ea != eb and (eb, ea) not in edges for ea, eb in directed], dtype=bool)
    return nxt, ids, boundary


def _chunk_layout(
    poly_ids: np.ndarray, boundary: np.ndarray, chunk: int
) -> dict[str, np.ndarray]:
    """Static per-chunk bookkeeping for streaming the edges polygon by polygon.

    Polygons are contiguous in the packed edges, so the polygons touched by a
    chunk get consecutive local ids 0..n_local-1. Only the first polygon of a
    chunk can continue from the previous chunk and only the last one can
    continue into the next one; that open polygon is carried in the scan.
    """
    num_edges = poly_ids.shape[0]
    steps = max(1, -(-num_edges // chunk))
    idx = np.arange(steps * chunk)
    valid = idx < num_edges
    pid = poly_ids[np.minimum(idx, num_edges - 1)].reshape(steps, chunk)
    first = pid[:, 0]
    lid = pid - first[:, None]
    valid = valid.reshape(steps, chunk)
    in_boundary = np.concatenate([boundary, np.zeros(steps * chunk - num_edges, bool)])
    starts = np.arange(steps) * chunk
    ends = np.minimum(starts + chunk, num_edges) - 1
    last = poly_ids[ends]
    return {
        "edge_index": np.minimum(idx, num_edges - 1).reshape(steps, chunk),
        "wind_lid": np.where(valid, lid, chunk).astype(np.int32),
        "dist_lid": np.where(valid & in_boundary.reshape(steps, chunk), lid, chunk).astype(
            np.int32
        ),
        "n_local": (last - first + 1).astype(np.int32),
        "cont_in": np.array([s > 0 and poly_ids[s - 1] == f for s, f in zip(starts, first)]),
        "cont_out": np.array(
            [e + 1 < num_edges and poly_ids[e + 1] == lp for e, lp in zip(ends, last)]
        ),
    }


def signed_distance(
    points: Array,
    a: Array,
    b: Array,
    poly_ids: np.ndarray,
    boundary: np.ndarray,
    chunk: int = 64,
) -> Array:
    """Signed distance (negative inside) from points to the union of polygons.

    Pure JAX, safe under ``jit``. Each polygon is filled by the nonzero winding
    rule (orientation does not matter) and the union distance is the minimum of
    the per-polygon signed distances, so overlapping polygons do not leave seams.

    The edges are streamed in chunks and every polygon is folded into the union
    as soon as its last edge has been seen, so memory is ~ (chunk + 6) * P no
    matter how many polygons there are; compute is ~ E * P.

    Where the nearest boundary point lies inside an edge (or the point lies on a
    vertex), the value is the signed distance to that edge's line (oriented with
    the polygon; averaged over tied edges), so the derivative stays correct on
    the boundary itself; near vertices it is sign * |p - v|.

    Args:
        points: (P, 2) query points.
        a: (E, 2) edge start points.
        b: (E, 2) edge end points.
        poly_ids: (E,) static polygon index of each edge (non-decreasing).
        boundary: (E,) static mask of the edges that count for the distance.
        chunk: edges processed per step.
    """
    if np.any(np.diff(poly_ids) < 0):
        raise ValueError("poly_ids must be sorted (edges of a polygon contiguous)")
    poly_ids = np.asarray(poly_ids)
    lay = _chunk_layout(poly_ids, np.asarray(boundary), chunk)
    # polygon orientation (+1 counter-clockwise), to sign the distance to edge lines
    num_polygons = int(poly_ids.max()) + 1 if poly_ids.size else 0
    cross = jax.lax.stop_gradient(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    orient = jnp.sign(jax.ops.segment_sum(cross, jnp.asarray(poly_ids), num_segments=num_polygons))
    orient = orient[jnp.asarray(poly_ids)][lay["edge_index"]]  # (steps, C)
    a = a[lay["edge_index"]]  # (steps, C, 2)
    b = b[lay["edge_index"]]
    segments = chunk + 1  # local ids 0..chunk-1, plus one discarded segment
    k = jnp.arange(segments)[:, None]
    px = points[None, :, 0]
    py = points[None, :, 1]
    inf = jnp.asarray(jnp.inf, dtype=points.dtype)

    def seg_sum(x: Array, ids: Array) -> Array:
        return jax.ops.segment_sum(x, ids, num_segments=segments)

    @jax.checkpoint
    def body(carry: tuple[Array, ...], inp: tuple[Array, ...]) -> tuple[Any, None]:
        sd, open_m, open_lin, open_ni, open_n, open_w = carry
        ac, bc, oc, wid, did, n_local, cont_in, cont_out = inp
        ab = bc - ac
        ap = points[None, :, :] - ac[:, None, :]  # (C, P, 2)
        denom = jnp.sum(ab * ab, axis=-1)[:, None]
        t = jnp.sum(ap * ab[:, None, :], axis=-1) / jnp.where(denom > 0, denom, 1.0)
        d = ap - jnp.clip(t, 0.0, 1.0)[..., None] * ab[:, None, :]
        d2e = jnp.sum(d * d, axis=-1)  # (C, P)
        # signed distance to the edge line (negative inside): smooth across the
        # edge, unlike sqrt(d2), whose derivative vanishes on the boundary
        length = jnp.sqrt(jnp.where(denom > 0, denom, 1.0))
        cr = ab[:, None, 0] * ap[..., 1] - ab[:, None, 1] * ap[..., 0]
        lin = -oc[:, None] * cr / length
        # a point on a vertex is on both adjacent edges: their lines give the
        # derivative there (|p - v| has none at p = v)
        interior = ((t > 0) & (t < 1)) | (jax.lax.stop_gradient(d2e) <= 1e-12)

        # nearest edge per polygon: min over this chunk, merged with the open polygon.
        # The minimum only selects edges (no gradient); the differentiable value is
        # rebuilt from the selected edges with sums. segment_min's own derivative
        # rules are 0/0 for empty segments, which poisons derivatives of
        # derivatives (e.g. reverse mode through the normals' forward mode).
        d2_val = jax.lax.stop_gradient(d2e)
        open_val = jax.lax.stop_gradient(open_m)
        m_val = jax.ops.segment_min(d2_val, did, num_segments=segments)
        m_val = m_val.at[0].set(jnp.where(cont_in, jnp.minimum(m_val[0], open_val), m_val[0]))
        sel = (did < chunk)[:, None] & (d2_val <= m_val[did])
        sel_i = sel & interior
        lin_sum = seg_sum(jnp.where(sel_i, lin, 0.0), did)
        d2_sum = seg_sum(jnp.where(sel, d2e, 0.0), did)
        n_int = seg_sum(sel_i.astype(jnp.int32), did)
        n_sel = seg_sum(sel.astype(jnp.int32), did)
        take_open = cont_in & (open_val <= m_val[0])
        lin_sum = lin_sum.at[0].add(jnp.where(take_open, open_lin, 0.0))
        d2_sum = d2_sum.at[0].add(jnp.where(take_open & (open_n > 0), open_m, 0.0) * open_n)
        n_int = n_int.at[0].add(jnp.where(take_open, open_ni, 0))
        n_sel = n_sel.at[0].add(jnp.where(take_open, open_n, 0))
        m = jnp.where(n_sel > 0, d2_sum / jnp.maximum(n_sel, 1), inf)

        # winding number (piecewise constant, no gradient)
        ac, bc = jax.lax.stop_gradient(ac), jax.lax.stop_gradient(bc)
        x1, y1 = ac[:, 0:1], ac[:, 1:2]
        x2, y2 = bc[:, 0:1], bc[:, 1:2]
        is_left = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
        up = (y1 <= py) & (y2 > py) & (is_left > 0)
        down = (y2 <= py) & (y1 > py) & (is_left < 0)
        w = seg_sum(up.astype(jnp.int32) - down.astype(jnp.int32), wid)
        w = w.at[0].add(jnp.where(cont_in, open_w, 0))

        # fold every polygon that ends in this chunk into the union
        last = n_local - 1
        closed = (k < n_local) & ~((k == last) & cont_out)
        dist = jnp.sqrt(m + 1e-12)
        sd_vertex = jnp.where(w != 0, -dist, dist)
        on_edge = (n_int == n_sel) & (n_sel > 0)
        sd_k = jnp.where(on_edge, lin_sum / jnp.maximum(n_int, 1), sd_vertex)
        sd = jnp.minimum(sd, jnp.min(jnp.where(closed, sd_k, inf), axis=0))
        keep = cont_out
        return (
            sd,
            jnp.where(keep, m[last], inf),
            jnp.where(keep, lin_sum[last], 0.0),
            jnp.where(keep, n_int[last], 0),
            jnp.where(keep, n_sel[last], 0),
            jnp.where(keep, w[last], 0),
        ), None

    n_points = points.shape[0]
    zeros_i = jnp.zeros((n_points,), dtype=jnp.int32)
    init = (
        jnp.full((n_points,), jnp.inf, dtype=points.dtype),
        jnp.full((n_points,), jnp.inf, dtype=points.dtype),
        jnp.zeros((n_points,), dtype=points.dtype),
        zeros_i,
        zeros_i,
        zeros_i,
    )
    xs = (
        a,
        b,
        orient.astype(points.dtype),
        jnp.asarray(lay["wind_lid"]),
        jnp.asarray(lay["dist_lid"]),
        jnp.asarray(lay["n_local"]),
        jnp.asarray(lay["cont_in"]),
        jnp.asarray(lay["cont_out"]),
    )
    (sd, *_), _ = jax.lax.scan(body, init, xs)
    return sd


# ---------------------------------------------------------------- roughness


def gaussian_random_field(
    shape: tuple[int, int], correlation_length: float, key: Array
) -> Array:
    """Unit-variance Gaussian random field with covariance exp(-r^2 / L^2).

    White noise filtered in Fourier space on a grid padded by 3 L on each side,
    so the field is not periodic over ``shape``. Lengths are in cells.
    """
    pad = math.ceil(3 * correlation_length)
    n = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    w = jax.random.normal(key, n)
    kx = 2 * jnp.pi * jnp.fft.fftfreq(n[0])
    ky = 2 * jnp.pi * jnp.fft.fftfreq(n[1])
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    h = jnp.exp(-k2 * correlation_length**2 / 8)  # sqrt of the power spectrum
    f = jnp.fft.ifft2(jnp.fft.fft2(w) * h).real / jnp.sqrt(jnp.mean(h**2))
    return f[pad : pad + shape[0], pad : pad + shape[1]]


# ---------------------------------------------------------------- dispersion


@autoinit
class SlotPole(Pole):
    """A pole given directly by its ADE constants (omega_0, gamma, K).

    Used by :func:`align_material_poles`: every material of a device gets the
    same ordered slots, and a material that lacks a pole has ``strength = 0``
    in that slot.
    """

    #: resonance angular frequency omega_0 (rad/s); 0 for a Drude pole
    resonance: float = frozen_field()
    #: damping rate gamma (rad/s)
    damping_rate: float = frozen_field()
    #: coupling K (rad^2/s^2): delta_epsilon * omega_0^2 (Lorentz), omega_p^2 (Drude)
    strength: float = frozen_field()

    @property
    def omega_0(self) -> float:
        return float(self.resonance)

    @property
    def gamma(self) -> float:
        return float(self.damping_rate)

    @property
    def coupling_sq(self) -> float:
        return float(self.strength)


def align_material_poles(materials: dict[str, fdtdx.Material]) -> dict[str, fdtdx.Material]:
    """Gives every material the same ordered pole slots.

    The slots are the distinct (omega_0, gamma) pairs of all poles of all
    materials, in order of first appearance; poles of one material that share
    (omega_0, gamma) are merged (their K add, which is exact). A material gets
    its own K in its slots and 0 elsewhere, so the discrete coefficients c1, c2
    of a slot are the same for every material and only c3 (proportional to K)
    differs. Mixing materials in a voxel then reduces to mixing c3, which mixes
    the susceptibilities exactly: ``chi = sum_m w_m chi_m``.

    Materials without poles stay unchanged when no material is dispersive.
    Permittivity (eps_inf), permeability and conductivities are kept, so
    fdtdx's material order does not change. Idempotent.
    """
    slots: list[tuple[float, float]] = []
    for m in materials.values():
        if m.dispersion is None:
            continue
        for pole in m.dispersion.poles:
            key = (float(pole.omega_0), float(pole.gamma))
            if key not in slots:
                slots.append(key)
    if not slots:
        return dict(materials)
    out = {}
    for name, m in materials.items():
        strength = [0.0] * len(slots)
        if m.dispersion is not None:
            for pole in m.dispersion.poles:
                strength[slots.index((float(pole.omega_0), float(pole.gamma)))] += float(
                    pole.coupling_sq
                )
        poles = tuple(
            SlotPole(resonance=w0, damping_rate=g, strength=k)
            for (w0, g), k in zip(slots, strength, strict=True)
        )
        out[name] = fdtdx.Material(
            permittivity=m.permittivity,
            permeability=m.permeability,
            electric_conductivity=m.electric_conductivity,
            magnetic_conductivity=m.magnetic_conductivity,
            dispersion=DispersionModel(poles=poles),
        )
    return out


def warn_reversible_losses(
    materials: dict[str, fdtdx.Material], config: SimulationConfig, name: str, conductivity: bool
) -> None:
    """Warns about lossy materials when gradients use fdtdx's reversible mode.

    The reversible backward pass runs the update backwards in time, which
    amplifies round-off in lossy voxels: by ~exp(gamma T / 2) for a damped
    Lorentz pole, ~exp(gamma T) for a Drude pole, and per step for a
    conductivity. Use ``GradientConfig(method="checkpointed")`` for those.
    """
    gradient_config = config.gradient_config
    if gradient_config is None or gradient_config.method != "reversible":
        return
    lossy = []
    for k, m in materials.items():
        if conductivity and m.is_electrically_conductive:
            lossy.append(f"{k} (conductivity)")
        if m.dispersion is not None:
            exponent = max(
                (
                    p.gamma * config.time * (1.0 if p.omega_0 == 0 else 0.5)
                    for p in m.dispersion.poles
                    if p.coupling_sq != 0
                ),
                default=0.0,
            )
            if exponent > 9.0:  # round-off amplified more than ~1e4 times
                lossy.append(f"{k} (pole damping: error growth ~exp({exponent:.0f}))")
    if lossy:
        warnings.warn(
            f"{name!r} contains lossy materials {lossy}; fdtdx's reversible gradients "
            "reconstruct the fields backwards in time, which amplifies round-off there. "
            'Use GradientConfig(method="checkpointed") for accurate gradients.',
            stacklevel=3,
        )
