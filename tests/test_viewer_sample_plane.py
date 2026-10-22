"""A detector plane's glyph sits where fdtdx records its field.

fdtdx co-locates every detector component at ``(i·Δx, j·Δy, (k+½)·Δz)``, so a
plane normal to x or y is sampled on its cell's lower edge. The detector is
placed by anchoring that edge on the requested centre, and the glyph (drawn
before a run and adopted after one) is drawn at the same point.
"""

from __future__ import annotations

from typing import Any

import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("anywidget")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack  # noqa: E402

from gdsfactory.fdtdx_stack import detector_plane_own_positions  # noqa: E402
from gdsfactory.viewer import adopt, sim, widget  # noqa: E402


def test_own_positions_anchor_the_sampled_point() -> None:
    """The lower edge along an x or y normal, the centre otherwise."""
    assert detector_plane_own_positions(0) == (-1.0, 0.0, 0.0)
    assert detector_plane_own_positions(1) == (0.0, -1.0, 0.0)
    assert detector_plane_own_positions(2) == (0.0, 0.0, 0.0)


def test_sampled_coordinate_snaps_to_the_recorded_point() -> None:
    """x and y samples are on a cell's lower edge, z samples at its centre."""
    # 100 cells of 0.04 um in y: nodes on multiples of 0.04, the box centre a node
    assert sim.sampled_coordinate(1, 0.0, 4.0, 0.04) == pytest.approx(0.0)
    assert sim.sampled_coordinate(1, 0.01, 4.0, 0.04) == pytest.approx(0.0)
    assert sim.sampled_coordinate(1, 0.03, 4.0, 0.04) == pytest.approx(0.04)
    # halfway between two nodes: the lower index, as fdtdx's argmin takes it
    assert sim.sampled_coordinate(1, 0.02, 4.0, 0.04) == pytest.approx(0.0)
    # off the box clamps to the last cell's lower edge
    assert sim.sampled_coordinate(1, 99.0, 4.0, 0.04) == pytest.approx(1.96)
    # 75 cells of 0.04 um in z: an odd count puts the box centre on a cell centre
    assert sim.sampled_coordinate(2, 0.0, 3.0, 0.04) == pytest.approx(0.0)
    assert sim.sampled_coordinate(2, 0.03, 3.0, 0.04) == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("type_name", "expected"),
    [
        ("ModeOverlapDetector", "own_positions=detector_plane_own_positions(1)"),
        ("ModePlaneSource", "own_positions=(0, 0, 0)"),
    ],
)
def test_launch_code_anchors_an_absolute_plane_like_the_run(type_name: str, expected: str) -> None:
    """The exported tail places a detector plane the way the viewer's own run does."""
    viewer = widget.view(f1_stack())
    spec = sim.ObjectSpec(
        type=type_name,
        port="None",
        params={"name": "plane", "axis": 1, "center": [7.5, 0.0, -0.1], "size": [18.0, 3.0]},
    )
    lines = sim._absolute_lines(spec, sim.TYPES[type_name], viewer.settings, [1.55], viewer.stack)
    assert expected in "\n".join(lines)
    assert "detector_plane_own_positions" in sim._LAUNCH_PRELUDE


def _run(viewer: Any) -> Any:
    viewer.run()
    viewer._run_thread.join()
    assert viewer.last_error is None, viewer.last_error
    return viewer.last_result


@pytest.mark.parametrize(("plane", "axis"), [("yz", 0), ("xz", 1)])
def test_an_editor_glyph_sits_where_the_run_samples_the_plane(plane: str, axis: int) -> None:
    """The glyph drawn before a run is the placed cell's lower edge, near the request."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(
        None, {"event": "set_settings", "settings": {"resolution_um": 0.1, "pml_thickness": 4}}, []
    )
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    viewer._on_msg(None, {"event": "add_detector", "port": plane}, [])
    spec = next(s for s in viewer.specs if s.category == "detector")
    origin = tuple(viewer._viewer_state.origin)
    glyph = sim.spec_glyph(viewer.stack, spec, origin, viewer.settings)
    requested = float(viewer.stack._port_frame(plane, None, 0.0)[3][axis]) - origin[axis]

    result = _run(viewer)
    spacing = float(result.config.uniform_spacing())
    shape = tuple(int(v) for v in result.objects.volume.grid_shape)
    lower, _upper = adopt.grid_bounds_um(
        tuple(result.objects[spec.name].grid_slice), spacing, shape, origin
    )

    # the glyph is in the box-centred local frame; the grid bounds are in stack coordinates
    assert glyph["center"][axis] == pytest.approx(lower[axis] - origin[axis], abs=1e-6)
    # the nearest sampled point to the request, so within half a cell of it
    assert abs(glyph["center"][axis] - requested) <= 0.5 * spacing * 1e6 + 1e-6
