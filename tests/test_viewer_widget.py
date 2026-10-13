"""Tests for the viewer's shared state, wire format and app shell (spec M2).

M2's exit criteria are R1's, R2's and R3's acceptance criteria. The parts that
do not need a browser are tested here: the state model and its defaults, the
GDS-layer color resolution, the wire payload round trip, the patch handler, and
the spec C3 guard that ``import gdsfactory`` stays free of ``anywidget``.

The visual half -- the WebGL render, picking and highlight -- is the user's
marimo check (there is no browser on the compute node).

The F1/F2/F3 fixtures come from ``test_viewer_geometry`` (M0).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anywidget")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack, f2_stack, f3_stack

from gdsfactory.viewer import palette, payload, state, widget
from gdsfactory.viewer.solids import build

FIXTURES = {"F1": f1_stack, "F2": f2_stack, "F3": f3_stack}


# ------------------------------------------------------------------ R1: explorer


def test_f1_lists_levels_and_ports() -> None:
    """R1 AC: F1 lists 3 layers (oxide, lower, upper) and 2 ports (o1, o2)."""
    viewer_state = state.ViewerState(f1_stack())
    assert list(viewer_state.styles) == ["oxide", "lower", "upper"]
    assert [port.name for port in viewer_state.ports] == ["o1", "o2"]


def test_selection_round_trips() -> None:
    """R1 AC: selecting o2 is carried in the state the viewport reads."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"selection": {"kind": "port", "name": "o2"}})
    assert viewer_state.to_state()["selection"] == {"kind": "port", "name": "o2"}


def test_bad_selection_is_rejected() -> None:
    """A selection of an unknown item is refused, not stored."""
    viewer_state = state.ViewerState(f1_stack())
    with pytest.raises(ValueError, match="Unknown selection"):
        viewer_state.apply_patch({"selection": {"kind": "port", "name": "nope"}})
    assert viewer_state.selection is None


# ------------------------------------------------------------------ R2: colors


def test_default_opacities_follow_fill() -> None:
    """R2 AC: F1's oxide (a fill) starts at the fill opacity; the rest default."""
    viewer_state = state.ViewerState(f1_stack())
    assert viewer_state.styles["oxide"].opacity == state.FILL_OPACITY
    assert viewer_state.styles["oxide"].visible
    for name in ("lower", "upper"):
        assert viewer_state.styles[name].opacity == state.LEVEL_OPACITY


def test_default_colors_come_from_the_pdk() -> None:
    """R2: a level's default color is the PDK's layer_views fill color.

    F1's ``upper`` sits on a layer the active PDK colours, so its default must
    be that fill color, not a palette fallback. This is what fails if the PDK
    lookup silently falls back for every layer.
    """
    from gdsfactory.fdtdx_stack import _expr_layers
    from gdsfactory.pdk import get_layer_tuple, get_layer_views

    viewer_state = state.ViewerState(f1_stack())
    for style in viewer_state.styles.values():
        assert style.color.startswith("#")
        assert len(style.color) == 7

    upper = viewer_state.styles["upper"]
    layer = min(_expr_layers(viewer_state.stack.levels[2].expr))
    expected = get_layer_views().get_from_tuple(get_layer_tuple(layer)).fill_color.as_hex()
    assert upper.color == expected
    assert upper.color not in palette.FALLBACK


def test_color_edit_updates_the_state() -> None:
    """R2 AC: changing upper to red is carried in the state both viewports read."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"styles": {"upper": {"color": "#ff0000"}}})
    assert viewer_state.styles["upper"].color == "#ff0000"
    assert viewer_state.to_state()["styles"]["upper"]["color"] == "#ff0000"


def test_opacity_is_clamped() -> None:
    """R2: opacity is clamped to [0, 1]."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"styles": {"upper": {"opacity": 2.5}}})
    assert viewer_state.styles["upper"].opacity == 1.0
    viewer_state.apply_patch({"styles": {"upper": {"opacity": -1.0}}})
    assert viewer_state.styles["upper"].opacity == 0.0


def test_unknown_level_is_rejected() -> None:
    """A patch naming an unknown level raises and leaves the state unchanged."""
    viewer_state = state.ViewerState(f1_stack())
    with pytest.raises(ValueError, match="Unknown level"):
        viewer_state.apply_patch({"styles": {"nope": {"color": "#000000"}}})
    with pytest.raises(ValueError, match="Unknown state keys"):
        viewer_state.apply_patch({"nonsense": {}})


def test_derived_layer_color_resolves() -> None:
    """R2: F3's derived-layer level (WG - ETCH) still resolves a GDS color."""
    viewer_state = state.ViewerState(f3_stack())
    core = viewer_state.styles["core"]
    assert core.color.startswith("#")
    assert core.opacity == state.LEVEL_OPACITY


# ------------------------------------------------------------------ R3: visibility


def test_visibility_toggles_are_carried() -> None:
    """R3 AC: hiding oxide and toggling o1 are carried in the state."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch(
        {"styles": {"oxide": {"visible": False}}, "port_visible": {"o1": False}}
    )
    wire = viewer_state.to_state()
    assert wire["styles"]["oxide"]["visible"] is False
    assert wire["port_visible"]["o1"] is False
    assert wire["port_visible"]["o2"] is True


def test_show_full_toggle() -> None:
    """R3 (Should): the full-extent toggle is carried in the state."""
    viewer_state = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"show_full": True})
    assert viewer_state.to_state()["show_full"] is True


def test_port_fade_defaults_and_patches() -> None:
    """R4: the port fade's 1/e distance defaults to 0.5 um and is editable."""
    viewer_state = state.ViewerState(f1_stack())
    assert viewer_state.to_state()["port_fade_um"] == state.PORT_FADE_UM
    viewer_state.apply_patch({"port_fade_um": 1.25})
    assert viewer_state.to_state()["port_fade_um"] == 1.25
    with pytest.raises(ValueError):
        viewer_state.apply_patch({"port_fade_um": 0.0})


# ------------------------------------------------------------------ wire format


def test_payload_round_trips_to_the_meshes() -> None:
    """D5/C5: the wire buffers decode back to the built meshes exactly."""
    for name, fixture in FIXTURES.items():
        stack = fixture()
        viewer_state = state.ViewerState(stack)
        geometry, _ = payload.build_payload(viewer_state)
        for index, level in enumerate(viewer_state.levels):
            wire = geometry["levels"][index]
            assert wire["name"] == level.name
            for which in ("painted", "full"):
                source = getattr(level, which)
                mesh = wire[which]
                positions = np.frombuffer(mesh["positions"], dtype=np.float32)
                indices = np.frombuffer(mesh["indices"], dtype=np.uint32)
                assert positions.shape == (source.positions.size,), name
                assert indices.shape == (source.indices.size,), name
                assert mesh["count"] == source.positions.shape[0]
                assert mesh["tris"] == source.indices.shape[0]
                np.testing.assert_array_equal(positions, source.positions.ravel())
                np.testing.assert_array_equal(indices, source.indices.ravel())


def test_payload_matches_build() -> None:
    """D5: the payload's box and level set match ``solids.build`` directly."""
    stack = f1_stack()
    origin, size, levels, _, _ = build(stack)
    geometry, meta = payload.build_payload(state.ViewerState(stack))
    assert geometry["origin"] == list(origin)
    assert geometry["size"] == list(size)
    assert [level["name"] for level in geometry["levels"]] == [lv.name for lv in levels]
    assert [level["name"] for level in meta["levels"]] == [lv.name for lv in levels]
    assert meta["background"] == stack.background


def test_state_edit_carries_no_geometry() -> None:
    """D10: a style edit only re-sends the small state, never the meshes."""
    viewer_state = state.ViewerState(f1_stack())
    wire = viewer_state.to_state()
    assert "levels" not in wire
    assert "origin" not in wire
    assert set(wire) == {
        "styles",
        "port_visible",
        "selection",
        "active_viewport",
        "show_full",
        "plane",
        "plane_position",
        "plane_bounds",
        "port_fade_um",
    }


# ------------------------------------------------------------------ the widget


def test_view_returns_a_widget_over_the_stack() -> None:
    """D7: ``view(stack)`` returns a Viewer whose ``.stack`` is the device."""
    stack = f1_stack()
    viewer = widget.view(stack)
    assert viewer.stack is stack
    assert viewer.geometry["size"] == list(state.ViewerState(stack).size)
    assert viewer.state["styles"]["oxide"]["opacity"] == state.FILL_OPACITY


def test_widget_applies_a_patch_from_the_browser() -> None:
    """D10: the message handler applies a browser patch and re-syncs the state."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "set_state", "patch": {"styles": {"upper": {"color": "#00ff00"}}}}, [])
    assert viewer.state["styles"]["upper"]["color"] == "#00ff00"


def test_widget_ignores_a_bad_patch() -> None:
    """D10: a bad patch leaves the canonical state unchanged."""
    viewer = widget.view(f1_stack())
    before = dict(viewer.state["styles"])
    viewer._on_msg(None, {"event": "set_state", "patch": {"styles": {"nope": {}}}}, [])
    assert viewer.state["styles"] == before


def test_static_assets_exist() -> None:
    """The committed bundle anywidget loads exists and is non-empty (spec D6)."""
    static = Path(widget.__file__).parent / "static"
    for name in ("viewer.js", "viewer.css"):
        path = static / name
        assert path.is_file(), f"missing {name}; run `npm run build` in js/"
        assert path.stat().st_size > 0


# ------------------------------------------------------------------ C3 guard


def test_import_gdsfactory_does_not_pull_the_gui_deps() -> None:
    """C3: ``import gdsfactory`` must not import the viewer or its gui deps.

    ``anywidget`` and every ``gdsfactory.viewer`` module are viewer-owned, so
    their absence is the contract this guard protects (spec C3, D2).

    ``manifold3d`` is *not* asserted: it is already in ``sys.modules`` after a
    bare ``import gdsfactory``, but not through any viewer code. The chain is
    base ``gdsfactory/__init__`` -> ``export/__init__`` -> ``export/to_3d.py``
    -> ``trimesh`` -> ``trimesh/boolean.py``, which imports ``manifold3d`` in a
    ``try/except``. That import only succeeds because C6 puts ``manifold3d`` in
    the ``betterfdtd`` env; it is unrelated to this milestone.
    """
    code = (
        "import sys, gdsfactory; "
        "viewer = [m for m in sys.modules if m.startswith('gdsfactory.viewer')]; "
        "print('anywidget' in sys.modules, bool(viewer))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"
