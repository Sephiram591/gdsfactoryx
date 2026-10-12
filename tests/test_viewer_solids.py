"""Tests for the viewer's analytic 3D solids (spec D3, M1).

M1's exit criteria: both D8 parity tests on F1-F3, R3's exact-width check on F2,
and the build time recorded (spec D3 risk). The model-parity test (D8 test 1)
checks that a point is inside a level's analytic solid exactly when the model's
signed distance there is negative; the solver-parity test (D8 test 2) checks the
solids against the voxelized ``material_weights``. Both exclude points within
the arc chord tolerance (1 nm, spec Q16) or near a voxel interface, where the
tessellation and the subpixel average respectively blur the boundary.

The fixture builders are M0's (``test_viewer_geometry``): ``f2_stack()`` and
``f3_stack()`` plus F1 from ``examples/fdtdx_elevator_coupler.py``.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import manifold3d as m3
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack, f2_stack, f3_stack  # noqa: E402

from gdsfactory.fdtdx_stack import StackDevice, init_stack_params  # noqa: E402
from gdsfactory.viewer import solids  # noqa: E402
from gdsfactory.viewer.geometry import _layer_distances, span  # noqa: E402

TOL = 1e-3  # um, the fixed arc chord tolerance (spec Q16)
BIG = 1e9  # um, "far outside" for a layer with no polygons
_FAR = np.array([1e4, 1e4, 1e4])  # a ray far enough to leave any fixture box

FIXTURES = {"F1": f1_stack, "F2": f2_stack, "F3": f3_stack}


def _inside(manifold: object, points: np.ndarray) -> np.ndarray:
    """Point-in-solid by parity of ray crossings (manifold's own ``ray_cast``)."""
    out = np.empty(len(points), dtype=bool)
    for i, p in enumerate(points):
        out[i] = len(manifold.ray_cast(tuple(p), tuple(p + _FAR))) % 2 == 1
    return out


def _model_shift(level: object, z: np.ndarray) -> np.ndarray:
    """The model's wall shift (um) at heights ``z`` (spec C2, ``_sidewall_shift``)."""
    z = np.asarray(z, dtype=float)
    z_ref = level.zmin + level.width_to_z * level.thickness
    shift = (z - z_ref) * math.tan(math.radians(level.sidewall_angle)) - level.bias
    if level.z_to_bias is not None:
        zs, bs = level.z_to_bias
        shift = shift - np.interp((z - level.zmin) / level.thickness, zs, bs)
    return shift


def _box_points(device: StackDevice, n: int, seed: int) -> np.ndarray:
    """``n`` uniform random points (um) inside the device box."""
    shape = device.partial_real_shape
    cx, cy = device.center
    cz = device.z_center
    sx, sy, sz = (float(v) * 1e6 for v in shape)
    rng = np.random.default_rng(seed)
    u = rng.random((n, 3))
    return np.column_stack(
        [cx - sx / 2 + u[:, 0] * sx, cy - sy / 2 + u[:, 1] * sy, cz - sz / 2 + u[:, 2] * sz]
    )


def _model_field(device: StackDevice, points: np.ndarray, level: object) -> np.ndarray:
    """The model's level signed distance (um): the layer field plus the wall shift.

    ``_eval_expr`` is M0's faithful copy of ``fdtdx_stack._eval_expr``; the layer
    distances come from the same ``fdtdx_geometry.signed_distance`` primitive the
    voxelizer uses.
    """
    from test_viewer_geometry import _eval_expr

    dist = _layer_distances(device, points[:, :2])
    return _eval_expr(level.expr, dist) + _model_shift(level, points[:, 2])


@pytest.mark.parametrize("name", ["F1", "F2", "F3"])
def test_model_parity(name: str) -> None:
    """D8 test 1 in 3D: inside the analytic solid iff the model's field is negative.

    Points within the arc chord tolerance of a boundary are excluded (spec Q16),
    as are points outside the level's own z span (the model's z surfaces).
    """
    device = FIXTURES[name]()
    solid_set = solids.analytic_solids(device)
    points = _box_points(device, 2000, seed=0)
    for level in device.levels:
        lo, hi = span(level)
        field = _model_field(device, points, level)
        in_span = (points[:, 2] >= lo) & (points[:, 2] <= hi)
        far = in_span & (np.abs(field) > TOL)
        inside_model = field < 0.0
        inside_solid = _inside(solid_set.full[level.name], points)
        mismatch = far & (inside_solid != inside_model)
        assert not mismatch.any(), (
            f"{level.name}: {int(mismatch.sum())}/{int(far.sum())} far points disagree "
            f"(worst |field| {np.max(np.abs(field[mismatch])):.3g} um)"
        )
        # outside its z span the model leaves the level empty
        assert not (inside_solid & ~in_span).any(), f"{level.name} extends past its z span"


def _place(device: StackDevice, resolution: float) -> tuple[dict, dict]:
    """Places a device in its own volume and returns fdtdx's objects and params."""
    config = fdtdx.SimulationConfig(
        time=10e-15, resolution=resolution * 1e-6, backend=jax.default_backend(), dtype=jnp.float32
    )
    volume, constraints = device.get_simulation_volume()
    objects, _, params, _, _ = fdtdx.place_objects(
        object_list=[volume, device], config=config, constraints=constraints, key=jax.random.PRNGKey(0)
    )
    return objects, {**params, **init_stack_params(objects)}


@pytest.mark.parametrize(("name", "resolution"), [("F1", 0.05), ("F3", 0.02)])
def test_solver_parity(name: str, resolution: float) -> None:
    """D8 test 2: a voxel the viewer paints as a level is that level's material in the solver.

    Voxels within one cell of an analytic boundary hold averaged fractions, so
    only interior voxels (every face neighbor painted the same) whose dominant
    material weight is 1 are checked. A voxel the viewer assigns to a level must
    then have that level's material weight 1 in ``material_weights`` (the
    reverse is not checked, since levels may share a material or the background).
    """
    device = FIXTURES[name]()
    objects, params = _place(device, resolution)
    placed = objects["stack"]
    nx, ny, nz = placed.grid_shape
    names, weights = placed.material_weights(params["stack"], apply_roughness=False)
    weights = np.asarray(weights)
    wmax = np.max(weights, axis=0)

    cx, cy = device.center
    cz = device.z_center
    xs = cx + (np.arange(nx) + 0.5 - nx / 2) * resolution
    ys = cy + (np.arange(ny) + 0.5 - ny / 2) * resolution
    zs = cz + (np.arange(nz) + 0.5 - nz / 2) * resolution
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    solid_set = solids.analytic_solids(device)
    label = np.full(len(points), -1, dtype=int)  # -1: background
    for i, level in enumerate(device.levels):
        label[_inside(solid_set.painted[level.name], points)] = i
    grid = label.reshape(nx, ny, nz)

    interior = wmax > 0.999
    for axis in range(3):
        for shift in (-1, 1):
            interior &= np.roll(grid, shift, axis=axis) == grid

    background_index = names.index(device.background)
    checked = 0
    for index in np.where(interior.ravel())[0]:
        level_index = label[index]
        material = background_index if level_index < 0 else names.index(device.levels[level_index].material)
        assert weights[material].ravel()[index] > 0.999, (
            f"voxel {points[index]} painted {level_index} is not material "
            f"{names[material]} (weight {weights[material].ravel()[index]})"
        )
        checked += 1
    assert checked > 0


def test_painted_solids_tile_the_box() -> None:
    """Painted levels are disjoint and, with the background, fill the box (spec Q13)."""
    for name in ("F1", "F2", "F3"):
        device = FIXTURES[name]()
        solid_set = solids.analytic_solids(device)
        painted = sum((m.volume() for m in solid_set.painted.values()), 0.0)
        total = painted + solid_set.background.volume()
        assert total == pytest.approx(solid_set.box.volume(), rel=1e-6), name
        # the levels are disjoint: the painted volumes sum to the union's volume
        union = None
        for m in solid_set.painted.values():
            union = m if union is None else union + m
        assert painted == pytest.approx(union.volume(), rel=1e-6), name


def _slab_width(manifold: object, x: float, z: float, depth: float = 1e-4) -> float:
    """The y extent of a solid's cross-section in a thin box about (x, z)."""
    slab = m3.Manifold.cube((0.02, 4.0, depth), center=True).translate((x, 0.0, z))
    box = (manifold ^ slab).bounding_box()  # (xmin, ymin, zmin, xmax, ymax, zmax)
    return float(box[4] - box[1])


def test_painting_follows_mesh_order() -> None:
    """A level is hollowed where a lower ``mesh_order`` wins in the same z band (spec Q13).

    F1's ``oxide`` fill (``mesh_order`` 9) is overlapped by ``lower`` (``mesh_order``
    1), and F3's ``box`` fill (``mesh_order`` 9) by ``slab`` (``mesh_order`` 2), so
    each fill's painted solid is smaller than its full one by exactly the winner's
    overlap.
    """
    for fixture, fill, winner in ((f1_stack, "oxide", "lower"), (f3_stack, "box", "slab")):
        solid_set = solids.analytic_solids(fixture())
        painted_fill = solid_set.painted[fill]
        full_fill = solid_set.full[fill]
        assert painted_fill.volume() < full_fill.volume(), fill
        removed = full_fill.volume() - painted_fill.volume()
        overlap = (full_fill ^ solid_set.full[winner]).volume()
        assert removed == pytest.approx(overlap, rel=1e-6), fill


def test_f2_top_width_is_exact() -> None:
    """F2's top face is narrower by exactly 2*thickness*tan(10 deg) (spec R3 AC).

    The straight runs along x, so the width is the y extent of a cross-section.
    Away from the tapered corners the sidewalls are planar, so the offset is
    exact (spec C2); a thin slab about the center of the guide measures it.
    """
    device = f2_stack()
    solid_set = solids.analytic_solids(device)
    core = solid_set.full["core"]
    center_x = float(device.center[0])
    width_bottom = _slab_width(core, center_x, 0.0)
    width_top = _slab_width(core, center_x, 0.22)
    assert width_bottom == pytest.approx(0.5, abs=1e-4)
    assert width_bottom - width_top == pytest.approx(2 * 0.22 * math.tan(math.radians(10.0)), abs=1e-4)


def test_f3_bias_and_z_to_bias_are_built() -> None:
    """F3's derived layer, ``bias`` and ``z_to_bias`` all contribute (spec D3 steps 2, 5)."""
    device = f3_stack()
    solid_set = solids.analytic_solids(device)
    core = solid_set.full["core"]
    slab = solid_set.full["slab"]
    assert core.volume() > 0.0 and slab.volume() > 0.0
    # core = WG - ETCH: the etch rectangle (x in [1.4, 2.0]) is removed
    assert _inside(core, np.array([[0.5, 0.0, 0.1]]))[0]  # inside the plain WG
    assert not _inside(core, np.array([[1.7, 0.0, 0.1]]))[0]  # inside the etch
    # slab's z_to_bias undercut: its offset grows with depth (z below its zmin),
    # so it narrows downward (the top at z = 0 is the widest)
    assert _slab_width(slab, 1.0, -0.2) < _slab_width(slab, 1.0, -0.05) < _slab_width(slab, 1.0, 0.0)


def test_build_time_is_recorded(capsys: pytest.CaptureFixture[str]) -> None:
    """The D3 risk: measure the build time on F1-F3 (spec M1 exit)."""
    for name in ("F1", "F2", "F3"):
        device = FIXTURES[name]()
        start = time.perf_counter()
        solids.build(device)
        elapsed = time.perf_counter() - start
        with capsys.disabled():
            print(f"\n{name} build: {elapsed * 1e3:.1f} ms")
        assert elapsed < 30.0, f"{name} build took {elapsed:.1f}s"
