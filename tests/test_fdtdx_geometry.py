"""Tests for the shared fdtdx geometry and dispersion helpers used by the StackDevice."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")

import gdsfactory as gf  # noqa: E402
from gdsfactory._jax import to_numpy  # noqa: E402
from gdsfactory.fdtdx_geometry import (  # noqa: E402
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


@pytest.fixture(autouse=True)
def no_snapping() -> Iterator[None]:
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    yield
    gf.snap.SNAP_ENABLED = True


def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _grid_points(nx: int, ny: int) -> jax.Array:
    xx, yy = jnp.meshgrid(jnp.arange(nx) + 0.5, jnp.arange(ny) + 0.5, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1)


def _sdf(polys: list[np.ndarray], points: jax.Array, chunk: int) -> jax.Array:
    v, sizes = pack_polygons(polys)
    nxt, ids, boundary = _edge_topology(np.asarray(v), sizes)
    return signed_distance(points, v, v[nxt], ids, boundary, chunk)


def fill(polys, shape=(40, 30), topology: np.ndarray | None = None) -> jax.Array:
    """In-plane fill fraction per cell: the distance kernel with the StackDevice's ramp."""
    v, sizes = pack_polygons(polys)
    nxt, ids, boundary = _edge_topology(to_numpy(v) if topology is None else topology, sizes)
    sd = signed_distance(_grid_points(*shape).astype(v.dtype), v, v[nxt], ids, boundary)
    return jnp.clip(0.5 - sd, 0.0, 1.0).reshape(shape)


# ---------------------------------------------------------------- distance kernel


def test_rectangle_area() -> None:
    f = fill([rect(5.3, 4.6, 27.8, 19.1)])
    assert float(f.sum()) == pytest.approx(22.5 * 14.5, abs=4 * 0.25)  # corner voxels
    assert float(f.max()) == 1.0
    assert float(f.min()) == 0.0


def test_orientation_does_not_matter() -> None:
    r = rect(5.3, 4.6, 27.8, 19.1)
    np.testing.assert_allclose(fill([r]), fill([r[::-1]]))


def test_union_of_overlapping_polygons_has_no_seam() -> None:
    a = rect(5, 5, 25.5, 15)
    b = rect(15, 10, 35, 25)
    f = fill([a, b])
    union_area = 20.5 * 10 + 20 * 15 - 10.5 * 5
    assert float(f.sum()) == pytest.approx(union_area, abs=8 * 0.25)
    # voxel center (25.5, 12.5) lies on a's right edge, which is inside b
    assert float(f[25, 12]) == 1.0


def test_seam_does_not_flip_the_sign_beyond_it() -> None:
    """A short polygon abutting a long one: beyond the long one, the short one's far edge
    is its nearest remaining edge but faces away (it is outside, not 12 cells inside)."""
    short, long = rect(0, 0, 2, 30), rect(2, 0, 10, 30)  # seam at x = 2
    sd = _sdf([short, long], jnp.asarray([[12.5, 15.5], [5.5, 15.5]]), 64)
    np.testing.assert_allclose(sd, [2.5, -4.5], atol=1e-5)
    assert float(fill([short, long]).sum()) == pytest.approx(10 * 30, abs=1e-3)


def test_hole_with_cut_line() -> None:
    """KLayout resolved_holes() keyhole polygon: no artifact along the cut line."""
    outer = [[4, 4], [34, 4], [34, 26], [4, 26]]
    # hole [12, 26] x [10, 20], connected to the outer boundary at x=4, y=15.5
    keyhole = np.array(
        outer[:3]
        + [[4, 26], [4, 15.5], [12, 15.5], [12, 20], [26, 20], [26, 10], [12, 10], [12, 15.5], [4, 15.5]],
        dtype=float,
    )
    f = fill([keyhole])
    assert float(f.sum()) == pytest.approx(30 * 22 - 14 * 10, abs=8 * 0.25)
    assert float(f[8, 15]) == 1.0  # voxel center (8.5, 15.5) is on the cut line
    assert float(f[18, 15]) == 0.0  # inside the hole


def test_vertex_gradient_matches_finite_differences() -> None:
    def area(width: jax.Array) -> jax.Array:
        c = gf.components.straight(length=4.0, width=width)
        v, _ = pack_polygons(c.get_polygons_points(layer=(1, 0)))
        # um -> cells; the offsets keep edges off cell boundaries and corner bisectors
        # off cell centers, where the fill is not differentiable (AD and FD then differ)
        polys = [(v - jnp.array([-0.2171, -1.0383])) / 0.05]
        return fill(polys, shape=(90, 40)).sum()

    g = jax.grad(area)(0.5)
    h = 1e-3
    fd = (area(0.5 + h) - area(0.5 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-3)
    assert float(g) == pytest.approx(80 / 0.05, rel=0.02)  # length (cells) per um of width


@pytest.mark.parametrize("y_top", [20.5, 20.0, 20.3])
def test_gradient_with_edges_on_voxel_centers(y_top: float) -> None:
    """Edges through voxel centers (sd == 0 there) must keep their gradient."""

    def area(y):
        return fill([jnp.array([[5.5, 10.5], [30.5, 10.5], [30.5, y], [5.5, y]])]).sum()

    # the top edge is 25 cells long; the two corner voxels add up to +-1/2 cell each
    assert float(jax.grad(area)(y_top)) == pytest.approx(25.0, abs=2 * 0.5 + 1e-6)


@pytest.mark.parametrize("chunk", [3, 7, 64, 1000])
def test_streamed_union_matches_per_polygon_reference(chunk: int) -> None:
    """Chunks that split polygons and chunks that hold many polygons agree."""
    rng = np.random.default_rng(0)
    polys = []
    for _ in range(40):
        x0, y0 = rng.uniform(0, 50, 2)
        w, h = rng.uniform(1, 8, 2)
        polys.append(rect(x0, y0, x0 + w, y0 + h)[:: rng.choice([1, -1])])
    phi = np.linspace(0, 2 * np.pi, 157, endpoint=False)  # spans many chunks
    polys.insert(17, np.stack([30 + 9 * np.cos(phi), 25 + 9 * np.sin(phi)], axis=1))
    points = _grid_points(60, 60)
    reference = jnp.min(jnp.stack([_sdf([p], points, 64) for p in polys]), axis=0)
    np.testing.assert_allclose(_sdf(polys, points, chunk), reference, rtol=1e-6, atol=1e-6)


def _split_layer(polys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Vertices after T-junction splitting, with their edge topology (as the StackDevice builds them)."""
    v0, sizes = pack_polygons(polys)
    split, split_sizes = split_t_junctions(np.asarray(v0), sizes)
    v = np.asarray(v0)[split]
    nxt, ids, boundary = _edge_topology(v, split_sizes)
    return v, nxt, ids, boundary, split_sizes


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_nearest_edges_match_brute_force(chunk: int) -> None:
    """The k nearest boundary edges, streamed in chunks, are those of a brute-force search.

    With overlapping polygons, a seam of abutting polygons split at a T-junction
    (not a boundary) and a polygon of many edges.
    """
    rng = np.random.default_rng(1)
    corners, sizes = rng.uniform(0, 40, (12, 2)), rng.uniform(2, 9, (12, 2))
    polys = [rect(x, y, x + w, y + h) for (x, y), (w, h) in zip(corners, sizes)]
    polys += [rect(10, 50, 20, 56), rect(20, 51, 30, 55)]  # a taper on the side of a bar
    phi = np.linspace(0, 2 * np.pi, 75, endpoint=False)
    polys.append(np.stack([30 + 6 * np.cos(phi), 62 + 6 * np.sin(phi)], axis=1))
    v, nxt, _, boundary, _ = _split_layer(polys)
    assert (~boundary).sum() >= 2  # the seam
    points = rng.uniform(-5, 70, (600, 2))
    f32 = [jnp.asarray(x, dtype=jnp.float32) for x in (points, v, v[nxt])]
    dist, ids = jax.jit(lambda p, a, b: nearest_edges(p, a, b, boundary, chunk))(*f32)

    a, ab = v, v[nxt] - v
    t = np.clip(np.sum((points[:, None] - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)
    d = np.linalg.norm(points[:, None] - a - t[..., None] * ab, axis=-1)
    d[:, ~boundary] = np.inf
    order = np.argsort(d, axis=1, kind="stable")
    ranked = np.take_along_axis(d, order, axis=1)
    np.testing.assert_allclose(dist, ranked[:, :3], rtol=1e-6, atol=1e-5)
    untied = ranked[:, 3] - ranked[:, 2] > 1e-3
    assert untied.sum() > 300
    for p in np.flatnonzero(untied):
        assert set(np.asarray(ids[p]).tolist()) == set(order[p, :3].tolist())

    # fewer boundary edges than asked for: infinitely far, edge 0
    tri = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    dist, ids = nearest_edges(
        jnp.asarray([[1.0, 0.5]]), jnp.asarray(tri), jnp.asarray(tri[[1, 2, 0]]), np.ones(3, bool), chunk, k=5
    )
    np.testing.assert_allclose(dist[0, :3], [0.5, 1.0, 1.4], atol=1e-6)
    assert bool(jnp.all(jnp.isinf(dist[0, 3:]))) and ids[0, 3:].tolist() == [0, 0]


def test_boundary_links() -> None:
    """Successors follow the boundary across T-junction seams and cut lines and stop at pinches."""

    def links(polys: list[np.ndarray]) -> dict:
        """(start, end) of every boundary edge (filled side on the left) -> that of its successor."""
        v, nxt, ids, boundary, sizes = _split_layer(polys)
        src, dst, succ = _boundary_links(v, sizes, nxt, ids, boundary)

        def key(e: int) -> tuple:
            return tuple(v[src[e]].tolist()), tuple(v[dst[e]].tolist())

        return {key(e): key(succ[e]) if succ[e] >= 0 else None for e in np.flatnonzero(boundary)}

    mmi = rect(0, -0.75, 2, 0.75)
    taper = np.array([(2, -0.25), (4, -0.1), (4, 0.1), (2, 0.25)])
    merged = np.array([(0, -0.75), (2, -0.75), (2, -0.25), (4, -0.1), (4, 0.1), (2, 0.25), (2, 0.75), (0, 0.75)])
    whole = links([merged])
    assert len(whole) == 8 and None not in whole.values()
    assert all(edge[1] == nxt[0] for edge, nxt in whole.items())  # each edge continues where it ends
    assert links([mmi, taper]) == whole
    assert links([mmi, taper[::-1]]) == whole  # a clockwise taper
    assert links([merged[::-1]]) == whole

    # two squares touching at (1, 1): two edges end there and two start there
    pinch = links([np.array([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (1, 2), (1, 1), (0, 1)], dtype=float)])
    assert pinch[((1.0, 0.0), (1.0, 1.0))] is None and pinch[((1.0, 2.0), (1.0, 1.0))] is None
    assert pinch[((0.0, 0.0), (1.0, 0.0))] == ((1.0, 0.0), (1.0, 1.0))

    # a hole connected by a KLayout cut line: the outer and the hole outline close across the cut
    outer = [[4, 4], [34, 4], [34, 26], [4, 26]]
    keyhole = np.array(
        outer[:3] + [[4, 26], [4, 15.5], [12, 15.5], [12, 20], [26, 20], [26, 10], [12, 10], [12, 15.5], [4, 15.5]],
        dtype=float,
    )
    hole = links([keyhole])
    assert len(hole) == 10  # the cut lines are not boundary
    assert hole[((4.0, 26.0), (4.0, 15.5))] == ((4.0, 15.5), (4.0, 4.0))
    assert hole[((12.0, 10.0), (12.0, 15.5))] == ((12.0, 15.5), (12.0, 20.0))


# ---------------------------------------------------------------- topology checks


MMI = rect(0, -0.75, 2, 0.75)
TAPER = np.array([(2, -0.25), (4, -0.1), (4, 0.1), (2, 0.25)], dtype=float)


def _changes(before: list[np.ndarray], after: list[np.ndarray], keep_tol: float = 1e-9, **kwargs) -> list[str]:
    sizes = [len(p) for p in before]
    assert [len(p) for p in after] == sizes
    return topology_changes(np.concatenate(before), np.concatenate(after), sizes, keep_tol, **kwargs)


def _moved(polys: list[np.ndarray], degrees: float, shift: tuple[float, float]) -> list[np.ndarray]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [np.asarray(p, dtype=float) @ np.array([[c, s], [-s, c]]) + np.asarray(shift) for p in polys]


def _teeth(period: float, n: int = 12, start: int = 0) -> list[np.ndarray]:
    return [rect(k * period, 0, k * period + 0.2, 1) for k in range(start, start + n)]


def test_topology_changes_pass_moves_that_keep_the_topology() -> None:
    """Junctions sliding along their edges, rigid moves, round-off, far moves of identical polygons."""
    narrower = TAPER * [1.0, 0.8]  # the taper's end slides along the MMI side
    keyhole = np.array(
        [[4, 4], [34, 4], [34, 26], [4, 26], [4, 15.5], [12, 15.5], [12, 20], [26, 20], [26, 10], [12, 10],
         [12, 15.5], [4, 15.5]], dtype=float,
    )
    cases = {
        "taper on MMI": ([MMI, TAPER], [MMI, narrower]),
        "clockwise taper": ([MMI, TAPER[::-1]], [MMI + [0.1, 0], narrower[::-1] + [0.1, 0]]),
        "abutting squares": ([rect(0, 0, 1, 1), rect(1, 0, 2, 1)], [rect(0, 0.3, 1.2, 1.3), rect(1.2, 0.3, 2, 1.3)]),
        "keyhole": ([keyhole], [keyhole * 1.1]),
        "stretched grating": (_teeth(0.5), _teeth(0.65)),  # the last tooth moves 3.3 periods
        "shifted grating": (_teeth(0.5), _teeth(0.5, start=2)),  # every tooth where another was: no cycle
    }
    for name, (before, after) in cases.items():
        assert _changes(before, after) == [], name
        assert _changes(before, _moved(after, 33.0, (5.0, -2.0)), keep_tol=1e-9) == [], name
        # float32 round-off of a rotated layout: off its edges by ~1e-6
        rounded = [p.astype(np.float32) for p in _moved(after, 33.0, (5.0, -2.0))]
        assert _changes(before, rounded, keep_tol=1e-5) == [], name


CHANGES = {
    # the side of the MMI would keep a bump through the moved taper's corners
    "T-junction came apart": (
        [MMI, TAPER], [MMI, TAPER + [0.1, 0.0]],
        r"^vertex 4 of polygon 1 was inside edge 1 -> 2 of polygon 0 \(a T-junction\) and is now 0\.1 away from it, "
        r"at \(2\.1, -0\.25\) \(and 1 more like it\)$",
    ),
    "new T-junction": (
        [MMI, TAPER + [0.1, 0.0]], [MMI, TAPER], r"^vertex 4 of polygon 1 now lies inside edge 1 -> 2 of polygon 0",
    ),
    # two strips on the MMI side that swap places: the split side would zig-zag
    "T-junctions out of order": (
        [MMI, rect(2, -0.6, 2.5, -0.4), rect(2, 0.25, 3.2, 0.7)],
        [MMI, rect(2, 0.3, 2.5, 0.5), rect(2, -0.65, 3.2, -0.2)],
        r"^the vertices inside edge 1 -> 2 of polygon 0 changed their order",
    ),
    # the gap between them would stay filled
    "seam opened": (
        [rect(0, 0, 1, 1), rect(1, 0, 2, 1)], [rect(0, 0, 1, 1), rect(1.05, 0, 2.05, 1)],
        r"^the seam between edge 1 -> 2 of polygon 0 and edge 7 -> 4 of polygon 1 opened by 0\.05, near \(1, 0\)$",
    ),
    "new seam": (
        [rect(0, 0, 1, 1), rect(1.05, 0, 2.05, 1)], [rect(0, 0, 1, 1), rect(1, 0, 2, 1)],
        r"^edge 1 -> 2 of polygon 0 now abuts edge 7 -> 4 of polygon 1, near \(1, 0\) \(a new seam\)$",
    ),
    # a taper drawn with a zero-width tip, which then opens
    "zero-length edge grew": (
        [np.array([(0, -0.25), (2, 0), (2, 0), (0, 0.25)], dtype=float)],
        [np.array([(0, -0.25), (2, -0.05), (2, 0.05), (0, 0.25)], dtype=float)],
        r"^edge 1 -> 2 of polygon 0 had zero length \(left out of the distance\) and is now 0\.1 long$",
    ),
    "orientation flipped": (
        [rect(0, 0, 2, 1)], [rect(0, 0, 2, 1) * [-1, 1]],
        r"^polygon 0 now runs the other way round \(signed area 2, now -2\)$",
    ),
    "polygons swapped": (
        [rect(0, 0, 1, 1), rect(3, 0, 4, 2)], [rect(3, 0, 4, 2), rect(0, 0, 1, 1)],
        r"^polygons 0, 1 \(4 vertices each\) look reordered: polygon 0 now lies where polygon 1 was \(rms vertex "
        r"distance 0, and 3\.08 to its own vertices\)$",
    ),
    "identical polygons rotated in order": (
        _teeth(0.5, n=5), [*_teeth(0.5, n=4, start=1), *_teeth(0.5, n=1)],
        r"^polygons 0, 1, 2, 3, 4 \(4 vertices each\) look reordered",
    ),
}


@pytest.mark.parametrize("name", list(CHANGES))
def test_topology_changes_report_what_no_longer_fits(name: str) -> None:
    before, after, expected = CHANGES[name]
    found = _changes(before, after)
    assert len(found) == 1, found
    assert re.match(expected, found[0]), found[0]
    assert _changes(before, before) == []
    if name == "orientation flipped":
        assert _changes(before, after, oriented=False) == []


def _found(polys: list[np.ndarray], tol: float) -> tuple[tuple[int, ...], int]:
    """Split sizes and seam edges of a layer's topology computed within tol (as a device built for it)."""
    v, sizes = pack_polygons(polys)
    index, split = split_t_junctions(np.asarray(v), sizes, tol)
    _, _, boundary = _edge_topology(np.asarray(v)[index], split, tol)
    return split, int((~boundary).sum())


def test_junctions_and_seams_within_the_tolerance() -> None:
    """A corner up to tol off an edge is inserted into it and abutting edges whose ends are up to tol apart are
    a seam (pairwise: the short edges of a finely sampled curve are neither points nor seams)."""
    tol = 1e-3
    assert _found([MMI, TAPER + [0.6 * tol, 0.0]], tol) == ((6, 4), 2)  # the side bends through the corners
    assert _found([MMI, TAPER + [1.5 * tol, 0.0]], tol) == ((4, 4), 0)  # a gap
    assert _found([rect(0, 0, 1, 1), rect(1 + 0.6 * tol, 0.2 * tol, 2, 1)], tol) == ((4, 4), 2)
    assert _found([rect(0, 0, 1, 1), rect(1 + 1.5 * tol, 0.0, 2, 1)], tol) == ((4, 4), 0)
    phi = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = 0.4 * tol * 400 / (2 * np.pi) * np.stack([np.cos(phi), np.sin(phi)], axis=1)  # edges 0.4 tol long
    assert _found([circle], tol) == ((400,), 0)


@pytest.mark.parametrize(
    ("before", "after", "kept"),
    [
        ([MMI, TAPER], [MMI, TAPER + [0.9e-3, 0.0]], True),  # a T-junction kept within keep_tol
        ([MMI, TAPER], [MMI, TAPER + [1.2e-3, 0.0]], False),  # one that came apart
        ([MMI, TAPER + [1.2e-3, 0.0]], [MMI, TAPER + [0.9e-3, 0.0]], False),  # a new one
        ([rect(0, 0, 1, 1), rect(1, 0, 2, 1)], [rect(0, 0, 1, 1), rect(1.0009, 0, 2, 1)], True),  # a seam kept
        ([rect(0, 0, 1, 1), rect(1, 0, 2, 1)], [rect(0, 0, 1, 1), rect(1.0012, 0, 2, 1)], False),  # one opened
        ([rect(0, 0, 1, 1), rect(1.0012, 0, 2, 1)], [rect(0, 0, 1, 1), rect(1.0009, 0, 2, 1)], False),  # a new one
    ],
)
def test_topology_changes_agree_with_a_rebuild(before: list[np.ndarray], after: list[np.ndarray], kept: bool) -> None:
    """Nothing is reported exactly when a topology computed from the new vertices within keep_tol has the
    frozen one's T-junctions and seams, so a rebuild keeps whatever the check keeps."""
    keep_tol = 1e-3
    assert (_changes(before, after, keep_tol=keep_tol, tol=keep_tol) == []) == kept
    assert (_found(after, keep_tol) == _found(before, keep_tol)) == kept


def test_many_polygons_gradient() -> None:
    v0, sizes = pack_polygons([rect(i * 3.0 + 0.3, 2.0, i * 3.0 + 2.1, 9.0) for i in range(300)])
    topology = np.asarray(v0)

    def f(v):
        polys = [v[4 * i : 4 * i + 4] for i in range(300)]
        return jnp.sum(fill(polys, shape=(910, 12), topology=topology) * jnp.arange(910)[:, None])

    g = jax.grad(f)(v0)
    assert g.shape == v0.shape
    # moving a rectangle's right edge (vertices 1, 2) by dx adds 7 cells with
    # x-weight floor(x_edge); the y-edges sit on voxel boundaries (no corners)
    i = 123
    expected = 7.0 * math.floor(i * 3.0 + 2.1)
    assert float(g[4 * i + 1, 0] + g[4 * i + 2, 0]) == pytest.approx(expected, rel=1e-6)


def test_whole_layer_gradient_matches_finite_differences() -> None:
    """Every polygon of a coupler's layer (bends, straights, overlaps) at once, cell by cell.

    The fill is not differentiable on a measure-zero set (e.g. a cell center on a
    corner bisector), where AD returns one valid one-sided derivative and central
    differences their mean; allow a few such cells.
    """
    res = 0.05

    def build(gap):
        return gf.components.coupler(gap=gap, dx=6.0, dy=3.0, length=2.0)

    polys0 = build(0.3).get_polygons_points(layer=(1, 0))
    v0, sizes = pack_polygons(polys0)
    lo = to_numpy(v0).min(axis=0) - np.array([0.3137, 0.2711])
    shape = tuple(int(n) for n in np.ceil((to_numpy(v0).max(axis=0) + 0.3 - lo) / res))
    topology = (to_numpy(v0) - lo) / res

    def f(gap):
        polys = build(gap).get_polygons_points(layer=(1, 0))
        assert tuple(len(p) for p in polys) == sizes
        return fill([(p - lo) / res for p in polys], shape=shape, topology=topology)

    _, jvp = jax.jvp(f, (0.3,), (1.0,))
    h = 1e-5
    fd = np.asarray((f(0.3 + h) - f(0.3 - h)) / (2 * h))
    jvp = np.asarray(jvp)
    moving = np.abs(fd) > 1e-3 * np.abs(fd).max()
    disagree = np.abs(jvp - fd) > 1e-2 * np.abs(fd).max()
    assert moving.sum() > 300  # the edges of all polygons move with the gap
    assert disagree.sum() <= 3, np.argwhere(disagree)[:10]


def test_gaussian_random_field_statistics() -> None:
    lc = 6.0
    f = gaussian_random_field((400, 400), lc, jax.random.PRNGKey(1))
    assert float(f.mean()) == pytest.approx(0.0, abs=0.05)
    assert float(f.std()) == pytest.approx(1.0, rel=0.05)
    lag = int(lc)
    corr = float(jnp.mean(f[:-lag] * f[lag:]) / jnp.mean(f * f))
    assert corr == pytest.approx(math.exp(-1.0), abs=0.05)


# ---------------------------------------------------------------- dispersion

C0 = 299792458.0
W0_SI = 2 * math.pi * C0 / 0.3e-6  # Lorentz resonance (rad/s)
SI_DISP = fdtdx.Material(
    permittivity=7.0,
    dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=W0_SI, damping=0.0, delta_epsilon=4.9),)
    ),
)
AU_DISP = fdtdx.Material(
    permittivity=1.0,
    dispersion=fdtdx.DispersionModel(poles=(fdtdx.DrudePole(plasma_frequency=1.37e16, damping=1.0e14),)),
)
OMEGA = 2 * math.pi * C0 / 1.55e-6


def test_align_material_poles() -> None:
    two_lorentz = fdtdx.Material(
        permittivity=7.0,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(resonance_frequency=W0_SI, damping=0.0, delta_epsilon=2.0),
                fdtdx.LorentzPole(resonance_frequency=W0_SI, damping=0.0, delta_epsilon=2.9),
            )
        ),
    )
    mats = {"sio2": fdtdx.Material(permittivity=2.1), "si": two_lorentz, "au": AU_DISP}
    aligned = align_material_poles(mats)
    assert list(aligned) == list(mats)
    keys = {k: [(p.omega_0, p.gamma) for p in m.dispersion.poles] for k, m in aligned.items()}
    assert keys["sio2"] == keys["si"] == keys["au"] == [(W0_SI, 0.0), (0.0, 1.0e14)]
    assert [p.coupling_sq for p in aligned["sio2"].dispersion.poles] == [0.0, 0.0]
    assert aligned["si"].dispersion.poles[0].coupling_sq == pytest.approx(4.9 * W0_SI**2)
    assert aligned["au"].dispersion.poles[1].coupling_sq == pytest.approx(1.37e16**2)
    for k in mats:
        assert aligned[k].permittivity == mats[k].permittivity
        assert aligned[k].electric_conductivity == mats[k].electric_conductivity
    # same susceptibility as the original materials, and idempotent
    for k in ("si", "au"):
        assert complex(aligned[k].dispersion.susceptibility(OMEGA)) == pytest.approx(
            complex(mats[k].dispersion.susceptibility(OMEGA)), rel=1e-12
        )
    again = align_material_poles(aligned)
    assert [(p.omega_0, p.gamma, p.coupling_sq) for p in again["si"].dispersion.poles] == [
        (p.omega_0, p.gamma, p.coupling_sq) for p in aligned["si"].dispersion.poles
    ]
    plain = {"a": fdtdx.Material(), "b": fdtdx.Material(permittivity=4.0)}
    assert all(m.dispersion is None for m in align_material_poles(plain).values())


def test_reversible_loss_warning() -> None:
    import warnings
    from types import SimpleNamespace

    reversible = SimpleNamespace(gradient_config=SimpleNamespace(method="reversible"), time=1e-12)
    with pytest.warns(UserWarning, match="checkpointed"):
        warn_reversible_losses({"au": AU_DISP}, reversible, "d", conductivity=False)
    with pytest.warns(UserWarning, match="conductivity"):
        warn_reversible_losses({"m": fdtdx.Material(electric_conductivity=1e5)}, reversible, "d", True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_reversible_losses({"si": SI_DISP}, reversible, "d", conductivity=True)  # lossless
        checkpointed = SimpleNamespace(gradient_config=SimpleNamespace(method="checkpointed"), time=1e-12)
        warn_reversible_losses({"au": AU_DISP}, checkpointed, "d", conductivity=True)


@pytest.mark.parametrize("corner", ["straight", "right angle"])
def test_derivative_on_a_vertex(corner: str) -> None:
    """A point exactly on a vertex keeps the edges' derivative (|p - v| has none at p = v)."""
    p = jnp.array([[2.0, 2.0]])
    # the vertex (2, 2) is a 180 deg corner on the top edge, or the top-right corner
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [0.0, 2.0]]) if corner == "straight" \
        else rect(0.0, 0.0, 2.0, 2.0)
    i = 3 if corner == "straight" else 2

    def sd(dy):
        v = jnp.asarray(poly).at[i, 1].add(dy)
        nxt, ids, boundary = _edge_topology(poly, (len(poly),))
        return signed_distance(p, v, v[nxt], ids, boundary)[0]

    g = float(jax.grad(sd)(0.0))
    h = 1e-4
    left, right = (float(sd(0.0)) - float(sd(-h))) / h, (float(sd(h)) - float(sd(0.0))) / h
    assert float(sd(0.0)) == pytest.approx(0.0, abs=1e-6)
    assert min(left, right) - 1e-3 <= g <= max(left, right) + 1e-3  # a one-sided slope or between
    assert g < -0.4  # raising the vertex puts the point inside (was 0 before)


# ---------------------------------------------------------------- derived shapes


def _sq(x0, y0, x1, y1):
    return rect(x0, y0, x1, y1), (4,)


def _cat(*layers):
    return np.concatenate([v for v, _ in layers]), tuple(n for _, ns in layers for n in ns)


def _recipe_region(recipes, pool):
    """The union of the parts' regions (holes subtracted per part)."""
    shapely = pytest.importorskip("shapely")
    return shapely.union_all([_part_region(r, pool) for r in recipes])


def _part_region(recipe, pool):
    shapely = pytest.importorskip("shapely")
    rings = recipe_rings(recipe, run_recipe(recipe.kind, recipe.index, recipe.positions, pool, np))
    region = shapely.Polygon()
    for r in rings:
        if shapely.LinearRing(r).is_ccw:
            region = region.union(shapely.make_valid(shapely.Polygon(r)))
    for r in rings:
        if not shapely.LinearRing(r).is_ccw:
            region = region.difference(shapely.make_valid(shapely.Polygon(r)))
    return region


def _boolean_region(layers, expr, universe):
    shapely = pytest.importorskip("shapely")
    regions = []
    for v, sizes in layers:
        polys = np.split(np.asarray(v, dtype=float), np.cumsum(sizes)[:-1])
        regions.append(shapely.union_all([shapely.make_valid(shapely.Polygon(p)) for p in polys]))

    def ev(e):
        if e[0] == "fill":
            return universe
        if e[0] == "layer":
            return regions[e[1]]
        x, y = ev(e[1]), ev(e[2])
        return {"and": x.intersection, "or": x.union, "not": x.difference, "xor": x.symmetric_difference}[e[0]](y)

    return ev(expr)


def _star(rng, cx, cy, n, r0, r1):
    phi = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(r0, r1, n)
    return np.stack([cx + r * np.cos(phi), cy + r * np.sin(phi)], axis=1)


A, B = ("layer", 0), ("layer", 1)
DERIVED_CASES = {
    "abutting or": ([_sq(0, 0, 1, 0.5), _sq(1, 0, 2, 0.5)], ("or", A, B)),
    "hole": ([_sq(0, 0, 2, 1), _sq(0.5, 0.3, 1.5, 0.7)], ("not", A, B)),
    "and of crossing bars": ([_sq(0, 0.4, 2, 0.6), _sq(0.9, 0, 1.1, 1)], ("and", A, B)),
    "xor": ([_sq(0, 0, 1, 1), _sq(0.5, 0.5, 1.5, 1.5)], ("xor", A, B)),
    "complement": ([_sq(0, 0, 1, 1), _sq(2, 0, 3, 1)], ("not", ("not", ("fill",), A), B)),
    "same-layer overlap": ([_cat(_sq(0, 0, 1.01, 0.5), _sq(1.0, 0, 2, 0.5))], A),
    "seam and T-junction": ([(np.array([(0, -0.75), (2, -0.75), (2, 0.75), (0, 0.75), (2, -0.25), (4, -0.1),
                                        (4, 0.1), (2, 0.25)], float), (4, 4)), _sq(3, -1, 5, 1)], ("not", A, B)),
    "empty and": ([_sq(0, 0, 1, 1), _sq(2, 2, 3, 3)], ("and", A, B)),
    "coincident edges": ([_sq(0, 0, 1, 1), _sq(0, 0, 1, 0.5)], ("or", A, B)),
    "coincident end": ([_sq(0, -0.25, 2, 0.25), _sq(1.4, -0.5, 2, 0.5)], ("not", A, B)),
    "two holes": ([_sq(0, 0, 3, 1), _cat(_sq(0.5, 0.3, 1, 0.7), _sq(2, 0.3, 2.5, 0.7))], ("not", A, B)),
}


@pytest.mark.parametrize("case", list(DERIVED_CASES))
def test_derive_polygons_matches_shapely(case: str) -> None:
    """The recipe's polygons are the boolean's (holes, xor, complements, seams, coincident edges, empty)."""
    shapely = pytest.importorskip("shapely")
    layers, expr = DERIVED_CASES[case]
    recipes = derive_polygons(layers, expr, 1e-6)
    pool = np.concatenate([v for v, _ in layers])
    universe = shapely.box(-10, -10, 10, 10)
    region = _recipe_region(recipes, pool)
    if recipes[0].negate:
        region = universe.difference(region)
    assert region.symmetric_difference(_boolean_region(layers, expr, universe)).area < 1e-12
    assert recipes[0].negate == (case == "complement")
    assert (sum(len(r.kind) for r in recipes) == 0) == (case == "empty and")
    assert len(recipes) == (2 if case == "xor" else 1)  # the xor's parts touch at its two crossings
    for r in recipes[1:]:  # parts meet at points only
        assert _part_region(r, pool).intersection(_part_region(recipes[0], pool)).area == 0


def test_derive_polygons_of_random_stars() -> None:
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(1)
    universe = shapely.box(-10, -10, 10, 10)
    for _ in range(4):
        layers = [(_star(rng, 0.5, 0.5, 9, 0.25, 0.5), (9,)), (_star(rng, 0.7, 0.6, 7, 0.2, 0.4), (7,))]
        for op in ("and", "or", "not", "xor"):
            recipes = derive_polygons(layers, (op, A, B))
            region = _recipe_region(recipes, np.concatenate([v for v, _ in layers]))
            assert region.symmetric_difference(_boolean_region(layers, (op, A, B), universe)).area < 1e-12, op


def test_shared_crossings_evaluate_bit_identically() -> None:
    """WG - ETCH and WG & ETCH share their crossings: the same formulas give the same float32 numbers."""
    layers = [_sq(0, -0.25, 2, 0.25), _sq(1.4, -0.5, 2.3, 0.5)]
    pool = jnp.asarray(np.concatenate([v for v, _ in layers]) * 50.0 + 0.137, dtype=jnp.float32)
    (one,), (two,) = derive_polygons(layers, ("not", A, B)), derive_polygons(layers, ("and", A, B))
    a = run_recipe(one.kind, one.index, one.positions * 50.0 + 0.137, pool)
    b = run_recipe(two.kind, two.index, two.positions * 50.0 + 0.137, pool)
    shared = 0
    for i in np.flatnonzero(one.kind == 1):
        for j in np.flatnonzero((two.kind == 1) & (two.index == one.index[i]).all(axis=1)):
            assert np.array_equal(np.asarray(a[i]), np.asarray(b[j]))
            shared += 1
    assert shared == 2


def test_joint_keeps_a_seam_vertex_derivative_at_one_half() -> None:
    """Abutting rectangles on two layers, or: the seam vertex is one vertex of each layer, joined at zero length,
    so moving one layer's vertex changes the area as that layer's own polygon does (1/2, not 1)."""
    layers = [_sq(0, 0, 1, 0.5), _sq(1, 0, 2, 0.5)]
    (recipe,) = derive_polygons(layers, ("or", A, B))
    pool = jnp.asarray(np.concatenate([v for v, _ in layers]))

    def area(pool):
        p = run_recipe(recipe.kind, recipe.index, recipe.positions, pool)
        q = jnp.roll(p, -1, axis=0)
        return 0.5 * jnp.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])

    g = jax.grad(area)(pool)
    corner = int(np.flatnonzero((np.asarray(pool) == [1.0, 0.5]).all(axis=1))[0])  # the first layer's (1, 0.5)
    assert float(area(pool)) == pytest.approx(1.0)
    assert float(g[corner, 1]) == pytest.approx(0.5)


def test_glue_cuts_cancel_for_any_motion() -> None:
    """A waveguide with two etched holes is one polygon glued by cuts; moved at random, its winding still
    counts each hole and the outside once, and the cuts stay out of the distance."""
    layers = [_sq(0, 0, 3, 1), _cat(_sq(0.5, 0.3, 1, 0.7), _sq(2, 0.3, 2.5, 0.7))]
    (recipe,) = derive_polygons(layers, ("not", A, B))
    assert len(recipe.rings) == 3
    pool = np.concatenate([v for v, _ in layers])
    rng = np.random.default_rng(0)
    moved = pool + rng.uniform(-0.05, 0.05, pool.shape)
    points = run_recipe(recipe.kind, recipe.index, recipe.positions, moved, np)
    nxt, ids, boundary = _edge_topology(recipe.positions, (len(points),), 1e-6)
    sd = signed_distance(jnp.asarray([[1.5, 0.5], [0.75, 0.5], [2.25, 0.5], [4.0, 0.5]]), jnp.asarray(points),
                         jnp.asarray(points)[nxt], ids, boundary)
    assert [bool(x < 0) for x in sd] == [True, False, False, False]
    assert int((~boundary).sum()) >= 4  # the two cut pairs


def test_run_recipe_is_finite_at_degeneracies() -> None:
    """Parallel and zero-length lines give finite values and gradients; float32 keeps the motion's precision."""
    kind = np.array([1, 1, 0], np.int8)
    index = np.array([[0, 1, 2, 3], [0, 1, 4, 5], [2, 2, 2, 2]], np.int32)
    pool = jnp.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.0), (0.5, 0.0)])

    def f(pool):
        return run_recipe(kind, index, np.zeros((3, 2)), pool).sum()

    assert bool(jnp.isfinite(f(pool)))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(jax.grad(f)(pool)))
    # a crossing far from the origin, in float32: local frame
    far = np.array([(1000.0, 1000.0), (1010.0, 1000.3), (1004.0, 995.0), (1004.2, 1005.0)])
    (recipe,) = derive_polygons([(far[:2].tolist() + [(1010.0, 990.0)], (3,)), (far[2:].tolist() + [(990.0, 1000.0)],
                                                                            (3,))], ("and", A, B))
    pool64 = np.concatenate([far[:2], [(1010.0, 990.0)], far[2:], [(990.0, 1000.0)]])
    exact = run_recipe(recipe.kind, recipe.index, recipe.positions, pool64 + 1e-3, np)
    single = run_recipe(recipe.kind, recipe.index, recipe.positions, jnp.asarray(pool64 + 1e-3, jnp.float32))
    assert float(np.abs(np.asarray(single, dtype=float) - exact).max()) < 2e-4  # float32 of 1000: 6e-5


def test_crossing_changes_finds_crossings_that_left_their_edges() -> None:
    layers = [_sq(0, -0.25, 2, 0.25), _sq(1.4, -0.5, 2.0, 0.5)]
    (recipe,) = derive_polygons(layers, ("not", A, B))
    pool = np.concatenate([v for v, _ in layers])
    assert crossing_changes(recipe, pool, 1e-6) == []
    moved = pool.copy()
    moved[4:, 0] = np.where(moved[4:, 0] < 1.5, -0.1, moved[4:, 0])  # the etch starts before the waveguide
    assert len(crossing_changes(recipe, moved, 1e-6)) == 2
