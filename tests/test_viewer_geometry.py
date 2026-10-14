"""Model-parity tests for the viewer's analytic geometry builder (spec D8, M0).

M0's exit criterion: on fixtures F1-F3, a point in the box is inside a level's
analytic xy region exactly when the model's signed distance there (the
``fdtdx_geometry.signed_distance`` primitive plus the C2 offsets) is negative.
Points within the arc chord tolerance (1 nm, spec Q16) of a boundary are
excluded, since there the shapely arcs are tessellated.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")

import shapely  # noqa: E402

import gdsfactory as gf  # noqa: E402
from gdsfactory.fdtdx_stack import StackDevice  # noqa: E402
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer  # noqa: E402
from gdsfactory.viewer.geometry import (  # noqa: E402
    _layer_distances,
    level_regions,
    slice_regions,
    span,
    wall_shift,
)

gf.gpdk.PDK.activate()

L = gf.gpdk.LAYER
WG, SLAB, ETCH, WAFER = L.WG, L.SLAB150, L.DEEP_ETCH, L.WAFER

MATERIALS = {
    "air": fdtdx.Material(),
    "sio2": fdtdx.Material(permittivity=2.1),
    "si": fdtdx.Material(permittivity=12.0),
}

TOL = 1e-3  # um, the fixed arc chord tolerance (spec Q16)
BIG = 1e9  # um, "far outside" for a layer with no polygons

_ROOT = Path(__file__).resolve().parents[1]


def _load_example(name: str) -> object:
    """Imports an ``examples/*.py`` module without the examples dir being a package."""
    path = _ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def f1_stack() -> StackDevice:
    """F1: the elevator coupler (``examples/fdtdx_elevator_coupler.py``)."""
    return _load_example("fdtdx_elevator_coupler").build_stack(5.0)


def f2_stack() -> StackDevice:
    """F2: a straight waveguide with 10-degree sidewalls and a fill level."""
    c = gf.Component()
    ref = c.add_ref(gf.components.straight(length=2.0, width=0.5))
    c.add_ports(ref.ports)
    ls = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-0.3, thickness=0.3, material="sio2", mesh_order=9),
            "core": LayerLevel(
                layer=(1, 0), zmin=0.0, thickness=0.22, sidewall_angle=10.0, material="si", mesh_order=1
            ),
        }
    )
    return StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="stack",
        included_layers=["box", "core"],
        boundary_offset=(0.5, -0.5, 0.5, -0.5, 0.0, 0.0),
    )


def f3_stack() -> StackDevice:
    """F3: a derived layer (``WG - ETCH``) with ``bias`` and a ``z_to_bias`` undercut.

    The shape is the one in ``tests/test_fdtdx_stack.py`` (``circuit()``): a 2 um
    straight on ``WG``, a ``SLAB`` rectangle, and an ``ETCH`` rectangle over the end.
    """
    c = gf.Component()
    c.add_ref(gf.components.straight(length=2.0, width=0.5))
    c.add_polygon([(0.0, -0.4), (2.0, -0.4), (2.0, 0.4), (0.0, 0.4)], layer=SLAB)
    c.add_polygon([(1.4, -0.5), (2.0, -0.5), (2.0, 0.5), (1.4, 0.5)], layer=ETCH)
    ls = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-0.3, thickness=0.3, material="sio2", mesh_order=9),
            "core": LayerLevel(
                layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH),
                derived_layer=LogicalLayer(layer=WG),
                zmin=0.0, thickness=0.22, bias=0.02, material="si", mesh_order=1,
            ),
            "slab": LayerLevel(
                layer=SLAB, zmin=0.0, thickness=-0.2, material="air", mesh_order=2,
                z_to_bias=([0.0, 1.0], [0.0, -0.2]),
            ),
        }
    )
    return StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="stack",
        included_layers=["box", "core", "slab"],
        boundary_offset=(0.5, -0.5, 0.5, -0.5, 0.0, 0.0),
    )


FIXTURES = {"F1": f1_stack, "F2": f2_stack, "F3": f3_stack}


def _eval_expr(expr: tuple, dist: dict[int, np.ndarray]) -> np.ndarray:
    """The model's level signed distance (um) for an expression, before the wall shift.

    Mirrors ``fdtdx_stack._eval_expr``: ``or``/``and``/``not``/``xor`` are the
    min / max / ``max(a, -b)`` / xor combinations (spec C2), each literal's
    sizing grow subtracted.
    """
    kind = expr[0]
    if kind == "fill":
        return np.full(next(iter(dist.values())).shape, -BIG)
    if kind == "layer":
        return dist.get(expr[1], np.full(next(iter(dist.values())).shape, BIG)) - expr[2]
    a = _eval_expr(expr[1], dist)
    b = _eval_expr(expr[2], dist)
    if kind == "or":
        region = np.minimum(a, b)
    elif kind == "and":
        region = np.maximum(a, b)
    elif kind == "not":
        region = np.maximum(a, -b)
    elif kind == "xor":
        region = np.maximum(np.minimum(a, b), -np.maximum(a, b))
    else:
        raise ValueError(f"Unknown layer operation {kind!r}")
    return region - expr[3]


def _sample_points(device: StackDevice, n: int, seed: int) -> np.ndarray:
    """``n`` uniform random points strictly inside the device box (um)."""
    shape = device.partial_real_shape
    cx, cy = device.center
    sx, sy = float(shape[0]) * 1e6, float(shape[1]) * 1e6
    rng = np.random.default_rng(seed)
    u = rng.random((n, 2))
    return np.stack([cx - sx / 2 + u[:, 0] * sx, cy - sy / 2 + u[:, 1] * sy], axis=1)


def _check_parity(device: StackDevice, z: float, n: int = 4000, seed: int = 0) -> None:
    """Asserts the analytic xy region of every level matches the model at height ``z``.

    A level is empty outside its own z span (the model's z surfaces), so parity
    is only meaningful inside the span; there the model's lateral field
    (``signed_distance`` plus the C2 offsets) decides membership. Points within
    the arc chord tolerance of a boundary are excluded (spec Q16).
    """
    points = _sample_points(device, n, seed)
    dist = _layer_distances(device, points)
    regions = level_regions(device, z)
    for level in device.levels:
        region = regions[level.name]
        lo, hi = span(level)
        if not (lo <= z <= hi):
            assert region.is_empty, f"{level.name} is nonempty outside its z span at z={z}"
            continue
        field = _eval_expr(level.expr, dist) + wall_shift(level, z)
        inside_model = field < 0.0
        far = np.abs(field) > TOL  # away from the tessellated boundary
        inside_region = shapely.contains_xy(region, points[:, 0], points[:, 1])
        mismatch = far & (inside_region != inside_model)
        assert not mismatch.any(), (
            f"{level.name} at z={z}: {int(mismatch.sum())}/{int(far.sum())} far points disagree "
            f"(worst |field| {np.max(np.abs(field[mismatch])):.3g} um)"
        )


@pytest.mark.parametrize("name", ["F1", "F2", "F3"])
def test_model_parity(name: str) -> None:
    """D8 test 1, restricted to xy slices (M0's exit criterion)."""
    device = FIXTURES[name]()
    sz = float(device.partial_real_shape[2]) * 1e6
    lo, hi = device.z_center - sz / 2, device.z_center + sz / 2
    for i in range(7):  # heights across the box
        z = lo + (hi - lo) * (i + 0.5) / 7
        _check_parity(device, z, seed=i)


def test_slice_regions_tile_the_box() -> None:
    """Painted slices are disjoint and, with the background, cover the box (spec Q13).

    At z = 0 the core (mesh_order 1) wins over the box fill (mesh_order 9) where
    they overlap, so the fill is only the box minus the core and nothing is left
    for the background.
    """
    device = f2_stack()
    regions = slice_regions(device, 0.0)
    assert set(regions) == {"box", "core", "background"}
    total = shapely.union_all([g for g in regions.values() if not g.is_empty])
    from gdsfactory.viewer.geometry import _box

    box = _box(device)
    assert total.symmetric_difference(box).area < 1e-6
    assert not regions["core"].is_empty  # the waveguide
    assert not regions["box"].is_empty  # the fill, where the core is not
    assert regions["background"].is_empty  # the fill covers the rest of the box


def test_f2_top_width_is_exact() -> None:
    """The core's top face is narrower by exactly 2*thickness*tan(10 deg) (planar faces).

    The straight runs along x, so its width is the y extent (``bounds[3] - bounds[1]``).
    """
    device = f2_stack()
    bottom = level_regions(device, 0.0)["core"]
    top = level_regions(device, 0.22)["core"]
    w_bottom = bottom.bounds[3] - bottom.bounds[1]
    w_top = top.bounds[3] - top.bounds[1]
    assert abs(w_bottom - 0.5) < 1e-4
    assert abs((w_bottom - w_top) - 2 * 0.22 * math.tan(math.radians(10.0))) < 1e-4
