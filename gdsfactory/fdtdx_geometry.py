"""Shared geometry and dispersion helpers for :mod:`gdsfactory.fdtdx_stack`.

- :func:`pack_polygons` and :func:`signed_distance`: a pure-JAX (jit-safe)
  signed distance to the union of a layer's polygons, differentiable with
  respect to the vertices. Each polygon is filled by the nonzero winding rule,
  overlapping polygons merge without seams, and KLayout hole cut lines and seams
  of abutting polygons are not treated as boundaries. :func:`split_t_junctions`
  makes partly shared junction edges (a narrower taper on a waveguide, a taper
  on the side of an MMI) exactly shared, so they are seams too.
  :func:`topology_changes` tells whether moved vertices still fit that static
  topology. :func:`nearest_edges` finds the few boundary edges nearest to each
  point (for the exact fill at polygon corners).
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
from functools import partial
from typing import Any

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.dispersion import DispersionModel, Pole

Array = jax.Array

# edges shorter than this fraction of the topology tolerance are points: they
# are left out of the distance (the tolerance itself would also drop the short
# edges of a finely sampled curve)
_POINT = 1e-3

__all__ = [
    "SlotPole",
    "align_material_poles",
    "gaussian_random_field",
    "nearest_edges",
    "pack_polygons",
    "signed_distance",
    "split_t_junctions",
    "topology_changes",
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
    with ``vertices[index]`` they follow those vertices when they move, which is
    right as long as they stay inside their edges (see :func:`topology_changes`).
    A vertex up to ``tol`` off an edge is inserted too: the edge then bends
    through it by at most ``tol``, where leaving it out would leave a slit
    between the polygons (layout snapping moves a vertex off a slanted edge it
    was on).

    Args:
        vertices: (E, 2) packed polygon vertices.
        sizes: vertices per polygon.
        tol: distance (same units as ``vertices``) within which a vertex lies on
            an edge, and within which it is at the edge's end (then the two
            polygons share that vertex instead).

    Returns:
        index: indices into ``vertices`` of the split polygons' vertices.
        sizes: vertices per split polygon.
    """
    return _apply_inserts(_t_junction_inserts(vertices, sizes, tol), sizes)


def _t_junction_inserts(vertices: np.ndarray, sizes: Sequence[int], tol: float = 1e-6) -> list[list[int]]:
    """The vertices that lie inside every edge (see :func:`split_t_junctions`).

    Args:
        vertices: (E, 2) packed polygon vertices.
        sizes: vertices per polygon.
        tol: distance tolerance (same units as ``vertices``, see
            :func:`split_t_junctions`).

    Returns:
        Per edge (by its start vertex), the indices of the vertices inside it,
        one per distinct position, ordered along the edge.
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
    return inserts


def _apply_inserts(inserts: Sequence[Sequence[int]], sizes: Sequence[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Index and sizes of the polygons with ``inserts[e]`` inserted after every vertex e."""
    starts = np.cumsum([0, *sizes[:-1]])
    index, new_sizes = [], []
    for start, size in zip(starts, sizes, strict=True):
        poly = []
        for v in range(start, start + size):
            poly.append(v)
            poly.extend(inserts[v])
        index.extend(poly)
        new_sizes.append(len(poly))
    return np.asarray(index, dtype=int), tuple(new_sizes)


def _directed(
    vertices: np.ndarray, nxt: np.ndarray, ids: np.ndarray, num_polygons: int
) -> tuple[np.ndarray, np.ndarray]:
    """Start and end vertex of every edge, directed with its polygon's filled side on the left.

    Args:
        vertices: (E, 2) packed polygon vertices.
        nxt: (E,) end vertex of every edge (polygon order).
        ids: (E,) polygon index of every edge.
        num_polygons: number of polygons.

    Returns:
        src: (E,) start vertex of every directed edge.
        dst: (E,) end vertex of every directed edge.
    """
    v = np.asarray(vertices, dtype=float)
    cross = v[:, 0] * v[nxt, 1] - v[nxt, 0] * v[:, 1]
    clockwise = np.bincount(ids, weights=cross, minlength=num_polygons)[ids] < 0
    edge = np.arange(len(v))
    return np.where(clockwise, nxt, edge), np.where(clockwise, edge, nxt)


def _seams(
    vertices: np.ndarray, nxt: np.ndarray, ids: np.ndarray, num_polygons: int, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed edges, the point-like ones, and the edge each other one abuts.

    Two edges abut (a seam: abutting polygons, a KLayout hole cut line, a
    zero-width spike) when, directed with their polygons' filled sides on the
    left, one runs back between the other's ends: each end within ``tol`` of
    the other's opposite end (pairwise, so near points do not chain). The
    edges must also point opposite ways, which only matters for edges up to
    about ``tol`` long, whose ends alone do not tell.

    Args:
        vertices: (E, 2) packed polygon vertices.
        nxt: (E,) end vertex of every edge (polygon order).
        ids: (E,) polygon index of every edge.
        num_polygons: number of polygons.
        tol: distance within which two vertices are the same point.

    Returns:
        src: (E,) start vertex of every directed edge.
        dst: (E,) end vertex of every directed edge.
        point: (E,) edges shorter than ``_POINT * tol``.
        partner: (E,) the lowest-index edge each other edge abuts, or -1.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=float)
    src, dst = _directed(v, nxt, ids, num_polygons)
    a, b = v[src], v[dst]
    d = b - a
    point = np.hypot(d[:, 0], d[:, 1]) <= _POINT * tol
    partner = np.full(len(v), -1, dtype=np.int64)
    live = np.flatnonzero(~point)
    if live.size < 2:
        return src, dst, point, partner
    # candidates within tol per coordinate, then each end within tol
    tree = cKDTree(np.concatenate([a[live], b[live]], axis=1))
    found = tree.query_ball_point(np.concatenate([b[live], a[live]], axis=1), r=tol, p=np.inf)
    counts = np.fromiter((len(f) for f in found), dtype=np.int64, count=len(found))
    if not counts.sum():
        return src, dst, point, partner
    e = np.repeat(live, counts)
    f = live[np.concatenate([np.asarray(x, dtype=np.int64) for x in found if len(x)])]
    ok = (
        (e != f)
        & (np.linalg.norm(a[f] - b[e], axis=1) <= tol)
        & (np.linalg.norm(b[f] - a[e], axis=1) <= tol)
        & (np.sum(d[e] * d[f], axis=1) < 0)
    )
    lowest = np.full(len(v), len(v), dtype=np.int64)
    np.minimum.at(lowest, e[ok], f[ok])
    partner = np.where(lowest < len(v), lowest, -1)
    return src, dst, point, partner


def _edge_topology(
    vertices: np.ndarray, sizes: Sequence[int], tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static edge structure: end index, polygon id and distance mask per edge.

    Edges that abut another edge whose polygon lies on the other side (KLayout
    hole cut lines, seams of abutting polygons, whatever the polygons'
    orientations; ends within ``tol``, see :func:`_seams`) and point-like edges
    are excluded from the distance: they are not part of the boundary of the
    union. They are kept for the winding number.

    Args:
        vertices: (E, 2) packed polygon vertices.
        sizes: vertices per polygon.
        tol: distance within which two vertices are the same point.

    Returns:
        nxt: (E,) end vertex of every edge.
        ids: (E,) polygon index of every edge.
        boundary: (E,) mask of the edges that bound the union.
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
    _, _, point, partner = _seams(vertices, nxt, ids, len(sizes), tol)
    return nxt, ids, ~point & (partner < 0)


def _boundary_links(
    vertices: np.ndarray,
    sizes: Sequence[int],
    nxt: np.ndarray,
    ids: np.ndarray,
    boundary: np.ndarray,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static boundary connectivity: directed ends and successor of every edge.

    Every edge is directed as in :func:`_edge_topology` (filled side on the
    left, whatever its polygon's orientation). The successor of a boundary edge
    is the boundary edge that starts where it ends, found by position, so links
    cross T-junction seams and hole cut lines: split abutting polygons link like
    the merged polygon. Edges starting at the very end point (within
    ``_POINT * tol``) are taken first; only without them, the one within
    ``tol``.

    Args:
        vertices: (E, 2) packed polygon vertices.
        sizes: vertices per polygon.
        nxt: (E,) end vertex of every edge (from :func:`_edge_topology`).
        ids: (E,) polygon index of every edge.
        boundary: (E,) mask of the edges that bound the union.
        tol: distance within which two vertices are the same point.

    Returns:
        src: (E,) start vertex of every edge.
        dst: (E,) end vertex of every edge.
        succ: (E,) the next boundary edge, or -1 for edges that are not on the
            boundary and where no boundary edge, or more than one (a pinch),
            starts at the end.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=float)
    src, dst = _directed(v, nxt, ids, len(sizes))
    succ = np.full(len(v), -1, dtype=np.int64)
    edges = np.flatnonzero(boundary)
    if not edges.size:
        return src, dst, succ
    tree = cKDTree(v[src[edges]])
    for e, near in zip(edges, tree.query_ball_point(v[dst[edges]], r=tol), strict=True):
        following = edges[np.asarray(near, dtype=np.int64)]
        following = following[following != e]
        gap = np.linalg.norm(v[src[following]] - v[dst[e]], axis=1)
        exact = following[gap <= _POINT * tol]
        if exact.size:
            following = exact
        if following.size == 1:
            succ[e] = following[0]
    return src, dst, succ


# ---------------------------------------------------------------- topology checks


def _xy(p: np.ndarray) -> str:
    """A point, for messages."""
    return f"({p[0]:.6g}, {p[1]:.6g})"


def _reordered_polygons(
    frozen: np.ndarray, current: np.ndarray, sizes: Sequence[int], tol: float, ratio: float
) -> list[list[tuple[int, int, float, float]]]:
    """Cycles of polygons that look reordered among the polygons of their size.

    Polygon ``a`` matches the frozen polygon ``b`` if the rms distance between
    their vertices (by index) is below ``ratio`` times its rms distance to its
    own frozen vertices, which must exceed ``tol``. Polygons that move far
    without being reordered, such as the teeth of a stretched grating, can match
    another polygon this way, but only a reordering closes a cycle (``a``
    matches ``b``, ..., which matches ``a``).

    Args:
        frozen: (E, 2) packed vertices before.
        current: (E, 2) packed vertices now.
        sizes: vertices per polygon.
        tol: rms distance up to which a polygon has not moved.
        ratio: how much closer the other polygon must be.

    Returns:
        Per cycle, (polygon, the frozen polygon it matches, rms distance to it,
        rms distance to its own) of each member.
    """
    from scipy.spatial import cKDTree

    starts = np.cumsum([0, *sizes[:-1]]).astype(int)
    match = np.arange(len(sizes))
    dist = np.zeros((len(sizes), 2))
    groups: dict[int, list[int]] = {}
    for p, size in enumerate(sizes):
        groups.setdefault(int(size), []).append(p)
    for size, members in groups.items():
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        rows = starts[idx][:, None] + np.arange(size)
        f, c = frozen[rows], current[rows]  # (G, N, 2)
        own = np.sqrt(np.mean(np.sum((c - f) ** 2, axis=-1), axis=-1))
        moved = np.flatnonzero(own > tol)
        if not moved.size:
            continue
        # an rms distance is at least the distance of the means: only polygons
        # whose centroids are this close can match
        near = cKDTree(f.mean(axis=1)).query_ball_point(c[moved].mean(axis=1), ratio * own[moved])
        for a, candidates in zip(moved, near, strict=True):
            others = [b for b in candidates if b != a]
            if not others:
                continue
            d = np.sqrt(np.mean(np.sum((c[a] - f[others]) ** 2, axis=-1), axis=-1))
            b = int(np.argmin(d))
            if d[b] < ratio * own[a]:
                match[idx[a]] = idx[others[b]]
                dist[idx[a]] = d[b], own[a]
    # cycles of the matches (each polygon matches one: a functional graph)
    cycles = []
    state = np.zeros(len(sizes), dtype=int)  # 0 unseen, 1 on the current path, 2 done
    for start in range(len(sizes)):
        path, p = [], start
        while state[p] == 0:
            state[p] = 1
            path.append(p)
            p = int(match[p])
        if state[p] == 1 and match[p] != p:
            cycle = path[path.index(p) :]
            cycles.append([(q, int(match[q]), float(dist[q, 0]), float(dist[q, 1])) for q in cycle])
        state[path] = 2
    return cycles


def topology_changes(
    frozen: np.ndarray,
    current: np.ndarray,
    sizes: Sequence[int],
    keep_tol: float,
    tol: float = 1e-6,
    oriented: bool = True,
    ratio: float = 0.25,
) -> list[str]:
    """What no longer fits when the static topology of ``frozen`` indexes ``current``.

    The topology of a layer's polygons is computed once, with numpy, from the
    vertices a device is built with: the T-junctions (:func:`split_t_junctions`),
    the seams left out of the distance (:func:`_edge_topology`) and the edges'
    directions (:func:`_boundary_links`). Later vertices, e.g. of the component
    rebuilt with other parameters, are indexed with it, which is right only while
    they keep it. This compares, by vertex index:

    - polygons that look reordered among the polygons of their size: one now
      lies where another one was (rms vertex distance below ``ratio`` times that
      to its own, in a cycle of such matches, so that polygons that move far
      without being reordered, e.g. a stretched grating, do not count). Reported
      alone, since everything else would follow from it;
    - polygons whose orientation flipped (with ``oriented``);
    - T-junctions that came apart (a vertex left the edge it was inside, or the
      vertices inside an edge changed their order) and new ones;
    - seams that opened (abutting edges parted, or a zero-length edge grew) and
      new ones (edges that now abut, running the other way).

    ``frozen``'s relations are those found within ``tol``, as the topology was
    computed; ``current``'s are those a topology computed from it within
    ``keep_tol`` would have. A relation of ``frozen`` is kept while it holds
    within ``keep_tol``, and one of ``current`` that ``frozen`` lacks is new.
    So round-off and moves that keep the topology (a taper end sliding along
    the side of an MMI, abutting polygons moving together) pass, and when
    nothing is reported, a topology computed from ``current`` with
    ``keep_tol`` has the same T-junctions and seams (edges up to ``keep_tol``
    long may count as points in one and not the other, which moves the
    boundary by at most that).

    Args:
        frozen: (E, 2) packed vertices the topology was computed from.
        current: (E, 2) packed vertices it is used with.
        sizes: vertices per polygon (of both).
        keep_tol: distance within which a relation of ``frozen`` still holds,
            and within which relations of ``current`` are found.
        tol: distance within which a vertex is on an edge or two vertices
            coincide, as when the topology was computed from ``frozen``.
        oriented: also report flipped orientations.
        ratio: how much closer a reordered polygon lies to another polygon's
            frozen vertices than to its own.

    Returns:
        One description per kind of change: its first case (vertices and
        polygons by index, lengths in the vertices' units) and how many more
        there are. Empty if the topology still fits.

    Raises:
        ValueError: if the vertices do not match ``sizes``.
    """
    frozen = np.asarray(frozen, dtype=float)
    current = np.asarray(current, dtype=float)
    n = int(sum(sizes))
    if frozen.shape != (n, 2) or current.shape != (n, 2):
        raise ValueError(f"vertices {frozen.shape} and {current.shape} do not match polygon sizes (sum {n})")
    if not n:
        return []
    found: dict[str, list[str]] = {}

    def report(kind: str, message: str) -> None:
        found.setdefault(kind, []).append(message)

    def summary() -> list[str]:
        return [m[0] + (f" (and {len(m) - 1} more like it)" if len(m) > 1 else "") for m in found.values()]

    def edge(polygon: int, i: int, j: int) -> str:
        return f"edge {i} -> {j} of polygon {polygon}"

    for cycle in _reordered_polygons(frozen, current, sizes, keep_tol, ratio):
        a, b, d_match, d_own = cycle[0]
        report(
            "reordered",
            f"polygons {', '.join(str(q) for q, *_ in cycle)} ({sizes[a]} vertices each) look reordered: "
            f"polygon {a} now lies where polygon {b} was (rms vertex distance {d_match:.3g}, and {d_own:.3g} "
            "to its own vertices)",
        )
    if found:
        return summary()

    nxt, ids, _ = _edge_topology(frozen, sizes, tol)
    if oriented:

        def twice_area(v: np.ndarray) -> np.ndarray:
            return np.bincount(ids, weights=v[:, 0] * v[nxt, 1] - v[nxt, 0] * v[:, 1], minlength=len(sizes))

        a_f, a_c = twice_area(frozen), twice_area(current)
        perimeter = np.bincount(ids, weights=np.linalg.norm(current[nxt] - current, axis=1), minlength=len(sizes))
        # both clearly nonzero: |area| / perimeter is about half the width
        flipped = (a_f * a_c < 0) & (np.abs(a_c) > 2 * keep_tol * perimeter)
        for p in np.flatnonzero(flipped):
            report(
                "orientation",
                f"polygon {p} now runs the other way round (signed area {a_f[p] / 2:.3g}, now {a_c[p] / 2:.3g})",
            )

    # T-junctions: the frozen ones still inside their edges, in order; no new ones
    inserts = _t_junction_inserts(frozen, sizes, tol)
    for e in (e for e in range(n) if inserts[e]):
        start, ab = current[e], current[nxt[e]] - current[e]
        length = float(np.hypot(*ab))
        last = -np.inf
        for i in inserts[e]:
            ap = current[i] - start
            if length > 0:
                along = float(ap @ ab) / length
                off = abs(float(ap[0] * ab[1] - ap[1] * ab[0])) / length
            else:
                along, off = 0.0, float(np.hypot(*ap))
            d = math.hypot(off, max(-along, along - length, 0.0))
            if d > keep_tol:
                report(
                    "junction left",
                    f"vertex {i} of polygon {ids[i]} was inside {edge(ids[e], e, nxt[e])} (a T-junction) and is "
                    f"now {d:.3g} away from it, at {_xy(current[i])}",
                )
            elif along < last - keep_tol:
                report("junction order", f"the vertices inside {edge(ids[e], e, nxt[e])} changed their order")
            last = max(last, along)
    now = _t_junction_inserts(current, sizes, keep_tol)
    for e in range(n):
        for i in now[e]:
            if all(np.hypot(*(current[i] - current[j])) > keep_tol for j in inserts[e]):
                report(
                    "new junction",
                    f"vertex {i} of polygon {ids[i]} now lies inside {edge(ids[e], e, nxt[e])}, at "
                    f"{_xy(current[i])} (a new T-junction)",
                )

    # seams, on the frozen split polygons: the frozen ones still closed, no new
    # ones (as found within tol in frozen and within keep_tol in current)
    index, split_sizes = _apply_inserts(inserts, sizes)
    vf, vc = frozen[index], current[index]
    s_nxt, s_ids, _ = _edge_topology(vf, split_sizes, tol)
    src, dst, point, partner = _seams(vf, s_nxt, s_ids, len(sizes), tol)
    _, _, point_now, partner_now = _seams(vc, s_nxt, s_ids, len(sizes), keep_tol)
    boundary, boundary_now = ~point & (partner < 0), ~point_now & (partner_now < 0)

    def split_edge(s: int) -> str:
        return edge(s_ids[s], index[s], index[s_nxt[s]])

    for s in np.flatnonzero(point):
        d = float(np.linalg.norm(vc[dst[s]] - vc[src[s]]))
        if d > keep_tol:
            report("grew", f"{split_edge(s)} had zero length (left out of the distance) and is now {d:.3g} long")
    reported = set()
    for s in np.flatnonzero(~boundary & ~point & boundary_now):
        f = int(partner[s])
        if (f, s) in reported:
            continue  # each pair once
        reported.add((s, f))
        gap = max(float(np.linalg.norm(vc[src[s]] - vc[dst[f]])), float(np.linalg.norm(vc[dst[s]] - vc[src[f]])))
        report("opened", f"the seam between {split_edge(s)} and {split_edge(f)} opened by {gap:.3g}, near {_xy(vc[s])}")
    new = boundary & ~boundary_now & ~point_now  # collapsed to a point: its distance is its end's
    for s in np.flatnonzero(new):
        f = int(partner_now[s])
        if s < f or not new[f]:  # each pair once
            report("new seam", f"{split_edge(s)} now abuts {split_edge(f)}, near {_xy(vc[s])} (a new seam)")
    return summary()


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
    sign_tol: float = 1e-3,
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
        sign_tol: distance (points' units) from an edge's line within which the
            line's side is trusted over the winding number (round-off on the
            boundary); beyond it the winding number signs the distance.
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
        lin_k = lin_sum / jnp.maximum(n_int, 1)
        # the edge line's side is the polygon's side only near its boundary: with
        # seam edges excluded, the nearest remaining edge can face away from a
        # point beyond the seam, so away from the line the winding sets the sign
        lin_val = jax.lax.stop_gradient(lin_k)
        agrees = ((lin_val < 0) == (w != 0)) | (jnp.abs(lin_val) <= sign_tol)
        sd_k = jnp.where(on_edge & agrees, lin_k, sd_vertex)
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


def nearest_edges(
    points: Array, a: Array, b: Array, boundary: np.ndarray, chunk: int = 64, k: int = 3
) -> tuple[Array, Array]:
    """The k nearest boundary edges of every point and their distances.

    Pure JAX, safe under ``jit``. The edges are streamed in chunks like
    :func:`signed_distance`, keeping the k nearest per point, so memory is
    ~ (chunk + k) * P. Equally near edges are ordered by edge index. Everything
    is under ``stop_gradient``: the result only selects edges, and callers
    rebuild differentiable distances from the selected edges.

    Args:
        points: (P, 2) query points.
        a: (E, 2) edge start points.
        b: (E, 2) edge end points.
        boundary: (E,) static mask of the edges that count.
        chunk: edges processed per step.
        k: edges kept per point.

    Returns:
        dist: (P, k) unsigned distances to the nearest boundary edges,
            ascending; inf where there are fewer than k boundary edges.
        ids: (P, k) int32 indices of those edges (0 where dist is inf).
    """
    points, a, b = (jax.lax.stop_gradient(x) for x in (points, a, b))
    lay = _chunk_layout(np.zeros(a.shape[0], dtype=np.int32), np.asarray(boundary, dtype=bool), chunk)
    counted = lay["dist_lid"] < chunk  # boundary edges, not padding
    return _nearest_scan(points, a, b, jnp.asarray(lay["edge_index"], dtype=jnp.int32), jnp.asarray(counted), k)


@partial(jax.jit, static_argnames=("k",))
def _nearest_scan(points: Array, a: Array, b: Array, index: Array, counted: Array, k: int) -> tuple[Array, Array]:
    """The scan of :func:`nearest_edges` over the edge chunks ``index`` (steps, C), boundary edges ``counted``.

    Jitted with the layout as arrays, so eager calls with the same shapes share
    the compiled scan.
    """
    inf = jnp.asarray(jnp.inf, dtype=points.dtype)

    def body(carry: tuple[Array, Array], inp: tuple[Array, ...]) -> tuple[Any, None]:
        d2_k, id_k = carry  # (k, P), nearest first
        ac, bc, idx, m = inp
        ab = bc - ac
        ap = points[None, :, :] - ac[:, None, :]  # (C, P, 2)
        denom = jnp.sum(ab * ab, axis=-1)[:, None]
        t = jnp.clip(jnp.sum(ap * ab[:, None, :], axis=-1) / jnp.where(denom > 0, denom, 1.0), 0.0, 1.0)
        d = ap - t[..., None] * ab[:, None, :]
        d2 = jnp.where(m[:, None], jnp.sum(d * d, axis=-1), inf)  # (C, P)
        # the carry first: top_k keeps the lower position on ties, i.e. the lower edge index
        d2_all = jnp.concatenate([d2_k, d2], axis=0)
        id_all = jnp.concatenate([id_k, jnp.broadcast_to(idx[:, None], d2.shape)], axis=0)
        neg, pos = jax.lax.top_k(-d2_all.T, k)  # (P, k)
        return (-neg.T, jnp.take_along_axis(id_all.T, pos, axis=1).T), None

    n_points = points.shape[0]
    init = (jnp.full((k, n_points), jnp.inf, dtype=points.dtype), jnp.zeros((k, n_points), dtype=jnp.int32))
    xs = (a[index], b[index], index, counted)
    (d2, ids), _ = jax.lax.scan(body, init, xs)
    d2, ids = d2.T, ids.T
    return jnp.sqrt(d2), jnp.where(jnp.isinf(d2), 0, ids)


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
