"""Tests for the shared fdtdx geometry and dispersion helpers used by the StackDevice."""

from __future__ import annotations

import math
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")

import gdsfactory as gf  # noqa: E402
from gdsfactory._jax import to_numpy  # noqa: E402
from gdsfactory.fdtdx_geometry import (  # noqa: E402
    _edge_topology,
    align_material_poles,
    gaussian_random_field,
    pack_polygons,
    signed_distance,
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
