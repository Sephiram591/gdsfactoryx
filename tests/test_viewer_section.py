"""Tests for the viewer's 2D cross-section (spec R4, D4, M3).

M3's exit criterion is R4's acceptance criteria. The section is sliced in the
kernel (not the browser), so the geometry half is testable headlessly here: the
exact band edges through F1, F2's 10-degree sidewalls, the tiling of the section
rectangle, a cross-check against M0's analytic ``slice_regions``, the wire round
trip, the state's plane validation/clamping, and an asymmetric fixture that pins
the ``xz``/``yz`` axis mapping.

What is *not* provable here is the render itself -- the canvas draw, hover
hit-testing and the fps target (spec Q8); that is the user's marimo check.

The F1/F2/F3 fixtures come from ``test_viewer_geometry`` (M0).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack, f2_stack, f3_stack

from gdsfactory.viewer import payload, solids, state
from gdsfactory.viewer.section import (
    PLANES,
    section,
    section_axis_names,
    section_bounds,
    section_normal_axis,
)

FIXTURES = {"F1": f1_stack, "F2": f2_stack, "F3": f3_stack}


# --------------------------------------------------------------- point helpers


def _contains(points: np.ndarray, rings: tuple[np.ndarray, ...]) -> np.ndarray:
    """Even-odd point-in-rings for every point (the browser's fill rule, spec R4)."""
    inside = np.zeros(len(points), dtype=bool)
    for ring in rings:
        x, y = points[:, 0], points[:, 1]
        for i in range(len(ring)):
            xj, yj = ring[i]
            xk, yk = ring[(i + 1) % len(ring)]
            straddle = (yj > y) != (yk > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                crossing = x < (xk - xj) * (y - yj) / (yk - yj) + xj
            inside ^= straddle & crossing
    return inside


def _rect_points(bounds: tuple, n: int, seed: int) -> np.ndarray:
    """``n`` uniform points inside the section rectangle (local ``u, v``)."""
    (u0, u1), (v0, v1) = bounds
    rng = np.random.default_rng(seed)
    u = rng.random((n, 2))
    return np.column_stack([u0 + u[:, 0] * (u1 - u0), v0 + u[:, 1] * (v1 - v0)])


def _ring_bounds(rings: tuple[np.ndarray, ...]) -> tuple[float, float, float, float]:
    """The ``(umin, umax, vmin, vmax)`` of a level's rings."""
    all_points = np.vstack(rings)
    return (
        float(all_points[:, 0].min()),
        float(all_points[:, 0].max()),
        float(all_points[:, 1].min()),
        float(all_points[:, 1].max()),
    )


# ------------------------------------------------------------------ R4: F1 bands


def _scan_line(sliced: object, u0: float) -> list[tuple[float, float, str]]:
    """The exact band intervals along the vertical line ``u = u0`` (local frame).

    Collects every ring's crossing of the line (the exact edges), then classifies
    each interval between consecutive crossings by its midpoint, so the returned
    ``(v_lo, v_hi, owner)`` list is exact, not sampled.
    """
    crossings: list[float] = []

    def collect(rings: tuple[np.ndarray, ...]) -> None:
        for ring in rings:
            for i in range(len(ring)):
                (ua, va), (ub, vb) = ring[i], ring[(i + 1) % len(ring)]
                if (ua - u0) * (ub - u0) < 0:
                    crossings.append(float(va + (u0 - ua) / (ub - ua) * (vb - va)))

    collect(sliced.background)
    for rings in sliced.rings.values():
        collect(rings)
    levels = sorted({round(v, 9) for v in crossings})

    bands = []
    for lo, hi in zip(levels, levels[1:]):
        point = np.array([[u0, 0.5 * (lo + hi)]])
        if _contains(point, sliced.background)[0]:
            owner = "background"
        else:
            owner = next(
                (name for name, rings in sliced.rings.items() if rings and _contains(point, rings)[0]),
                "?",
            )
        bands.append((lo, hi, owner))
    return bands


def test_f1_xz_band_edges_are_exact() -> None:
    """R4 AC: an xz cut through F1's taper overlap gives the exact band edges (spec R4, D4).

    Both tapers overlap for local ``u`` in [-0.5, 0.5]; scanning the vertical
    line ``u = 0.25`` there (off any ring vertex) gives, in stack z: ``lower`` in
    [-0.22, -0.12], ``upper`` in [0, 0.2], ``oxide`` below z = 0 except where
    ``lower`` is, and air (background) above z = 0.2. Every face is planar, so
    the edges are exact (spec C2).
    """
    solid_set = solids.analytic_solids(f1_stack())
    sliced = section(solid_set, "xz", 0.0)
    origin_z = solid_set.origin[2]

    bands = [(lo + origin_z, hi + origin_z, owner) for lo, hi, owner in _scan_line(sliced, 0.25)]
    expected = [
        (-1.6, -0.22, "oxide"),
        (-0.22, -0.12, "lower"),
        (-0.12, 0.0, "oxide"),
        (0.0, 0.2, "upper"),
        (0.2, 1.4, "background"),
    ]
    assert [owner for _, _, owner in bands] == [owner for _, _, owner in expected]
    np.testing.assert_allclose(
        [(lo, hi) for lo, hi, _ in bands], [(lo, hi) for lo, hi, _ in expected], atol=1e-6
    )


def test_f1_xz_lower_taper_wins_over_the_oxide() -> None:
    """R4 AC: where ``lower`` is, it is painted, not the oxide underneath (spec Q13)."""
    sliced = section(solids.analytic_solids(f1_stack()), "xz", 0.0)
    # a point inside the lower taper's band (z = -0.17 -> v = -0.07) and inside
    # the oxide's x range is owned by ``lower``, never by ``oxide``
    point = np.array([[0.0, -0.07]])
    assert _contains(point, sliced.rings["lower"])[0]
    assert not _contains(point, sliced.rings["oxide"])[0]


# --------------------------------------------------------- R4: F2 10 deg walls


def test_f2_xz_sidewalls_are_ten_degrees() -> None:
    """R4 AC: F2's tapered sidewalls are straight lines exactly 10 deg from vertical (spec R3).

    The straight guide runs along x, so its yz cross-section is the trapezoid:
    the side is straight (planar face, spec C2), and the half-width shrinks by
    ``thickness * tan(10 deg)`` from bottom to top.
    """
    sliced = section(solids.analytic_solids(f2_stack()), "yz", 1.0)
    ring = sliced.rings["core"][0]
    # u is the local y, v the local z (z vertical, spec R4); the +y side is the
    # u > 0 chain
    side = ring[ring[:, 0] > 1e-9]
    side = side[np.argsort(side[:, 1])]  # order by z, bottom to top
    bottom, top = side[0], side[-1]
    half_bottom = abs(bottom[0])
    half_top = abs(top[0])
    assert half_bottom == pytest.approx(0.25, abs=1e-6)
    assert half_bottom - half_top == pytest.approx(0.22 * math.tan(math.radians(10.0)), abs=1e-6)
    # the side is straight: every vertex is collinear with the two ends
    slope = (top[0] - bottom[0]) / (top[1] - bottom[1])
    predicted = bottom[0] + slope * (side[:, 1] - bottom[1])
    np.testing.assert_allclose(side[:, 0], predicted, atol=1e-6)


# ------------------------------------------------------------ R4: tiling (Q13)


@pytest.mark.parametrize(
    ("name", "plane", "position", "seed"),
    [("F1", "xz", 0.0, 1), ("F2", "xz", 0.0, 2), ("F3", "xy", -0.04, 3)],
)
def test_sections_tile_the_rectangle(name: str, plane: str, position: float, seed: int) -> None:
    """R4/D4: a section's level rings plus the background tile the rectangle (spec Q13).

    Every point of the section rectangle is owned by exactly one level or by the
    background -- no gap and no overlap -- so the painted solids are disjoint and
    the background is their complement.
    """
    solid_set = solids.analytic_solids(FIXTURES[name]())
    sliced = section(solid_set, plane, position)
    bounds = section_bounds(solid_set.size, plane)
    points = _rect_points(bounds, 4000, seed=seed)
    membership = np.array(
        [_contains(points, sliced.background)]
        + [_contains(points, rings) for rings in sliced.rings.values() if rings]
    )
    covered = membership.any(axis=0)
    assert covered.all(), f"{name}/{plane}: {int((~covered).sum())} unowned points"
    assert (membership.sum(axis=0) == 1).all(), f"{name}/{plane}: overlapping rings"


def test_holes_are_not_painted() -> None:
    """R4: a level ring with a hole is filled even-odd, so the hole belongs to a lower level.

    F3's ``box`` fill is cut by ``slab`` at z = -0.04, so the box's xy ring has a
    hole; a point inside that hole is owned by ``slab``, not by ``box``.
    """
    solid_set = solids.analytic_solids(f3_stack())
    sliced = section(solid_set, "xy", -0.04)
    assert len(sliced.rings["box"]) >= 2, "expected a hole in the box fill"
    # the slab sits over x in [-0.96, 0.96] (local), y in [-0.36, 0.36]
    point = np.array([[0.0, 0.0]])
    assert not _contains(point, sliced.rings["box"])[0]
    assert _contains(point, sliced.rings["slab"])[0]


# ------------------------------------------------------ cross-check with M0


@pytest.mark.parametrize("name", ["F1", "F2", "F3"])
def test_xy_section_matches_m0_slice_regions(name: str) -> None:
    """R4: the xy section from manifold3d agrees with M0's analytic ``slice_regions``.

    Two independent code paths -- manifold3d's boolean slice and shapely's
    analytic regions -- must give the same ownership at the same z. Points within
    the arc chord tolerance of a boundary are excluded (spec Q16).
    """
    import shapely

    from gdsfactory.viewer.geometry import slice_regions

    device = FIXTURES[name]()
    solid_set = solids.analytic_solids(device)
    z = solid_set.origin[2] + 0.25 * solid_set.size[2]  # a slice inside the box
    sliced = section(solid_set, "xy", z)
    regions = slice_regions(device, z)

    bounds = section_bounds(solid_set.size, "xy")
    points = _rect_points(bounds, 3000, seed=7)
    # back to stack xy for shapely (the section frame is from the box center, C5)
    stack_points = points + np.array([solid_set.origin[0], solid_set.origin[1]])
    shapely_points = shapely.points(stack_points)

    for lname, rings in sliced.rings.items():
        manifold_in = _contains(points, rings) if rings else np.zeros(len(points), dtype=bool)
        region = regions[lname]
        if region.is_empty:
            assert not manifold_in.any(), f"{name}: {lname} sliced where M0 is empty"
            continue
        shapely_in = shapely.contains_xy(region, stack_points[:, 0], stack_points[:, 1])
        far = shapely.distance(region.boundary, shapely_points) > 1e-3
        mismatch = far & (manifold_in != shapely_in)
        assert not mismatch.any(), (
            f"{name}: {int(mismatch.sum())}/{int(far.sum())} points disagree for {lname}"
        )


# ---------------------------------------------------------------- wire format


def test_section_payload_round_trips() -> None:
    """R4/D5: the wire payload decodes back to the kernel's ``SectionSet``."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"plane": "xz", "plane_position": 0.0})
    wire = payload.build_section_payload(viewer_state)
    sliced = section(viewer_state.solids, "xz", 0.0)

    assert wire["plane"] == "xz"
    assert wire["position"] == pytest.approx(0.0)
    assert (wire["axis_u"], wire["axis_v"], wire["normal_axis"]) == ("x", "z", "y")
    # the browser labels the local rings in device coordinates (spec C5), so the
    # payload carries the box center the rings are measured from
    assert wire["origin"] == pytest.approx(list(viewer_state.origin))
    assert wire["bounds"] == [list(section_bounds(viewer_state.size, "xz")[0]),
                              list(section_bounds(viewer_state.size, "xz")[1])]
    for name, rings in sliced.rings.items():
        decoded = wire["rings"][name]
        assert len(decoded) == len(rings), name
        for got, expected in zip(decoded, rings):
            np.testing.assert_allclose(np.asarray(got), expected, atol=1e-9)
    for got, expected in zip(wire["background"], sliced.background):
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-9)


# ------------------------------------------------------------------ state (R4)


def test_plane_defaults_to_the_box_center() -> None:
    """R4: the section starts on the xy plane at the box center (spec C5)."""
    viewer_state = state.ViewerState(f1_stack())
    assert viewer_state.plane == "xy"
    lo, hi = viewer_state.plane_bounds()
    assert viewer_state.plane_position == pytest.approx(0.5 * (lo + hi))
    assert viewer_state.plane_position == pytest.approx(viewer_state.origin[2])


def test_plane_patch_switches_the_normal_axis() -> None:
    """R4: switching the plane moves the slider to that axis' box range (spec R4)."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"plane": "xz"})
    assert viewer_state.plane == "xz"
    lo, hi = viewer_state.plane_bounds()
    # xz's normal is y: F1's box is y in [-2, 2]
    assert (lo, hi) == pytest.approx((-2.0, 2.0))
    viewer_state.apply_patch({"plane": "yz"})
    assert viewer_state.plane_bounds() == pytest.approx((9.5 - 11.0, 9.5 + 11.0))


def test_plane_position_is_clamped_and_validated() -> None:
    """R4: the position is clamped to the box and non-finite values are refused."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"plane": "xz"})  # normal y, bounds [-2, 2]
    viewer_state.apply_patch({"plane_position": 100.0})
    assert viewer_state.plane_position == pytest.approx(2.0)
    viewer_state.apply_patch({"plane_position": -100.0})
    assert viewer_state.plane_position == pytest.approx(-2.0)
    with pytest.raises(ValueError, match="Non-finite"):
        viewer_state.apply_patch({"plane_position": float("nan")})
    with pytest.raises(ValueError, match="Unknown plane"):
        viewer_state.apply_patch({"plane": "qq"})


def test_each_plane_remembers_its_own_position() -> None:
    """R4: switching planes keeps each plane's last slider position."""
    viewer_state = state.ViewerState(f1_stack())
    assert viewer_state.plane == "xy"

    # move the xy plane off center, then visit the other two planes
    viewer_state.apply_patch({"plane_position": 1.0})
    assert viewer_state.plane_positions["xy"] == pytest.approx(1.0)

    viewer_state.apply_patch({"plane": "xz"})
    assert viewer_state.plane_position == pytest.approx(viewer_state.origin[1])
    viewer_state.apply_patch({"plane_position": -1.0})
    viewer_state.apply_patch({"plane": "yz"})
    assert viewer_state.plane_position == pytest.approx(viewer_state.origin[0])

    # returning to a plane restores where it was left, not the box center
    viewer_state.apply_patch({"plane": "xz"})
    assert viewer_state.plane_position == pytest.approx(-1.0)
    viewer_state.apply_patch({"plane": "xy"})
    assert viewer_state.plane_position == pytest.approx(1.0)


def test_state_carries_the_plane_fields() -> None:
    """R4: the wire state carries the plane, its position and the slider bounds."""
    viewer_state = state.ViewerState(f1_stack())
    wire = viewer_state.to_state()
    assert wire["plane"] == "xy"
    assert wire["plane_position"] == pytest.approx(viewer_state.plane_position)
    assert wire["plane_bounds"] == list(viewer_state.plane_bounds())


def test_widget_reslices_only_on_plane_changes() -> None:
    """D10/R4: the section is rebuilt only when the plane or its position changes."""
    from gdsfactory.viewer import widget

    viewer = widget.view(f1_stack())
    before = viewer.section
    # a style edit must not reslice (the meshes are not resent either)
    viewer._on_msg(None, {"event": "set_state", "patch": {"styles": {"upper": {"color": "#00ff00"}}}}, [])
    assert viewer.section is before
    # a plane change must reslice
    viewer._on_msg(None, {"event": "set_state", "patch": {"plane": "xz"}}, [])
    assert viewer.section["plane"] == "xz"
    assert viewer.section is not before


# ------------------------------------------------------------- axis mapping


def test_axis_mapping_is_asymmetric() -> None:
    """R4: the ``xz``/``yz`` mappings are pinned by an asymmetric fixture.

    A wrong mapping (e.g. swapping u and v, or a sign error) is invisible on a
    symmetric fixture, so this builds a bar that is long in x (3 um), narrow in
    y (1 um) and thin in z (0.2 um): the ``xy`` section is wide in x, the ``xz``
    section is wide in x and thin in z, and the ``yz`` section is narrow in y and
    thin in z (``z`` vertical). Each plane's ``(u, v)`` axes are the documented
    pair (spec R4).
    """
    import fdtdx

    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import StackDevice
    from gdsfactory.technology import LayerLevel, LayerStack

    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon([(0.0, -0.5), (3.0, -0.5), (3.0, 0.5), (0.0, 0.5)], layer=(1, 0))
    layer_stack = LayerStack(
        layers={"core": LayerLevel(layer=(1, 0), zmin=0.0, thickness=0.2, material="si", mesh_order=1)}
    )
    materials = {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=12.0)}
    device = StackDevice.from_layer_stack(
        component, layer_stack, materials, background="air", name="stack",
        included_layers=["core"], boundary_offset=(0.5, -0.5, 0.5, -0.5, 0.0, 0.0),
    )
    solid_set = solids.analytic_solids(device)

    for plane in PLANES:
        assert section_axis_names(plane) == {
            "xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")
        }[plane]
        assert section_normal_axis(plane) == {"xy": "z", "xz": "y", "yz": "x"}[plane]

    # xy through the bar: wide in x (u ~ [-1.5, 1.5]), narrow in y (v ~ [-0.5, 0.5])
    xy = _ring_bounds(section(solid_set, "xy", 0.1).rings["core"])
    assert (xy[1] - xy[0], xy[3] - xy[2]) == pytest.approx((3.0, 1.0), abs=1e-6)
    # xz: wide in x, thin in z (v ~ [-0.1, 0.1])
    xz = _ring_bounds(section(solid_set, "xz", 0.0).rings["core"])
    assert (xz[1] - xz[0], xz[3] - xz[2]) == pytest.approx((3.0, 0.2), abs=1e-6)
    # yz: narrow in y (u ~ [-0.5, 0.5]), thin in z (v ~ [-0.1, 0.1])
    yz = _ring_bounds(section(solid_set, "yz", 1.0).rings["core"])
    assert (yz[1] - yz[0], yz[3] - yz[2]) == pytest.approx((1.0, 0.2), abs=1e-6)


def test_unknown_plane_raises() -> None:
    """R4: slicing an unknown plane is refused, not silently mapped."""
    solid_set = solids.analytic_solids(f2_stack())
    with pytest.raises(ValueError, match="Unknown plane"):
        section(solid_set, "qq", 0.0)
