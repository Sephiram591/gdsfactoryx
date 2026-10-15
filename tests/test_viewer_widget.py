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
        # M4 adds the setup and the run flag (spec R12, R13, D15)
        "sources",
        "detectors",
        "glyphs",
        "simulation",
        "running",
        # R15 adds the detector field display settings (spec R15, D16)
        "field_settings",
    }


# ------------------------------------------------- R13: boundary offset re-box


def test_boundary_offset_defaults_to_zero() -> None:
    """R13: the box starts at the geometry's own bounding box (no offset)."""
    viewer_state = state.ViewerState(f1_stack())
    assert viewer_state.settings.boundary_offset == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert viewer_state.to_state()["simulation"]["boundary_offset"] == [0.0] * 6


def test_boundary_offset_grows_the_box() -> None:
    """R13: a symmetric offset grows the box and keeps its center (spec R13)."""
    viewer_state = state.ViewerState(f1_stack())
    before = viewer_state.size
    viewer_state.apply_patch({"simulation": {"boundary_offset": [1, -1, 1, -1, 1, -1]}})
    assert viewer_state.size == (before[0] + 2, before[1] + 2, before[2] + 2)
    # symmetric offsets leave the center where it was
    assert viewer_state.origin == state.ViewerState(f1_stack()).origin
    assert viewer_state.geometry_changed is True
    assert viewer_state.revision == 1


def test_boundary_offset_moves_one_face() -> None:
    """R13: an asymmetric offset moves the box's center by half the offset."""
    viewer_state = state.ViewerState(f1_stack())
    base = state.ViewerState(f1_stack())
    viewer_state.apply_patch({"simulation": {"boundary_offset": [2, 0, 0, 0, 0, 0]}})
    assert viewer_state.size[0] == base.size[0] + 2
    assert viewer_state.origin[0] == base.origin[0] + 1
    assert viewer_state.size[1:] == base.size[1:]


def test_boundary_offset_clamps_the_plane_positions() -> None:
    """R13: a smaller box pulls the section sliders back inside it."""
    viewer_state = state.ViewerState(f1_stack())
    # the yz plane's normal is x, so an x-face offset moves its slider range
    lo, hi = viewer_state.plane_bounds("yz")
    viewer_state.apply_patch({"plane": "yz"})
    viewer_state.apply_patch({"plane_position": hi})
    assert viewer_state.plane_positions["yz"] == hi
    # moving the x+ face inward (a negative x+ offset) pulls the yz plane's
    # upper bound in, so the slider parked at the old max must be clamped
    viewer_state.apply_patch({"simulation": {"boundary_offset": [-3, 0, 0, 0, 0, 0]}})
    new_lo, new_hi = viewer_state.plane_bounds("yz")
    assert new_hi < hi
    assert viewer_state.plane_positions["yz"] <= new_hi


def test_boundary_offset_is_validated() -> None:
    """R13: a malformed or box-collapsing offset is refused, leaving the state."""
    viewer_state = state.ViewerState(f1_stack())
    before = viewer_state.size
    for bad in ([1, 2, 3], [0, 0, 0, 0, 0], ["x", 0, 0, 0, 0, 0], [-100, 100, 0, 0, 0, 0]):
        with pytest.raises(ValueError):
            viewer_state.apply_patch({"simulation": {"boundary_offset": bad}})
    assert viewer_state.size == before
    assert viewer_state.settings.boundary_offset == (0.0,) * 6


def test_boundary_offset_round_trips_through_the_wire() -> None:
    """R13: the offset survives to_wire/from_wire."""
    settings = state.SimSettings.from_wire({"boundary_offset": [1, -2, 3, -4, 5, -6]})
    assert settings.boundary_offset == (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
    assert settings.to_wire()["boundary_offset"] == [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]


def test_widget_resends_geometry_on_a_boundary_offset_edit() -> None:
    """R13: a boundary edit re-sends the meshes so the browser rebuilds (D5)."""
    viewer = widget.view(f1_stack())
    before = viewer.geometry["size"]
    viewer._on_msg(None, {"event": "set_settings", "settings": {"boundary_offset": [1, -1, 0, 0, 0, 0]}}, [])
    assert viewer.geometry["size"][0] == before[0] + 2
    assert viewer.state["simulation"]["boundary_offset"] == [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]


def test_widget_does_not_resend_geometry_for_a_plain_setting() -> None:
    """D5: a non-box setting edit leaves the meshes alone (only state re-sends)."""
    viewer = widget.view(f1_stack())
    before = viewer.geometry
    viewer._on_msg(None, {"event": "set_settings", "settings": {"resolution_um": 0.05}}, [])
    assert viewer.geometry is before
    assert viewer.state["simulation"]["resolution_um"] == 0.05


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


def test_run_installs_the_marimo_context_on_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: the run worker installs the caller's marimo context.

    Under marimo, a trait set only reaches the browser if the thread setting it
    has the runtime context installed; the context is thread-local, so the
    worker thread must install the one captured on the cell's thread. Without
    this the phase label and progress bar freeze on their last main-thread
    value (the run looks stuck on "placing…").
    """
    installed: list[bool] = []

    class FakeContext:
        def install(self):  # type: ignore[no-untyped-def]
            import contextlib

            @contextlib.contextmanager
            def _cm():  # type: ignore[no-untyped-def]
                installed.append(True)
                yield

            return _cm()

    monkeypatch.setattr(widget, "_marimo_runtime_context", lambda: FakeContext())
    viewer = widget.view(f1_stack())
    monkeypatch.setattr(widget, "_run_sim", lambda *a, **k: None)

    done: list[bool] = []
    viewer.run(on_done=lambda _r: done.append(True))
    viewer._run_thread.join(timeout=5)

    assert installed == [True], "the worker did not install the marimo context"
    assert done == [True], "on_done did not fire"


def test_widget_writes_a_saved_image(tmp_path: Path) -> None:
    """D14: a ``save_image`` message writes the PNG under ``save_dir``."""
    import base64

    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    payload = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
    viewer._on_msg(
        None,
        {"event": "save_image", "data": f"data:image/png;base64,{payload}", "filename": "shot.png"},
        [],
    )
    written = Path(viewer.save_result["path"])
    assert written.parent == tmp_path.resolve()
    assert written.name == "shot.png"
    assert written.read_bytes() == b"\x89PNG\r\n\x1a\nfake"


def test_saved_image_does_not_clobber(tmp_path: Path) -> None:
    """D14: a second save to the same name lands beside the first, not over it."""
    import base64

    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    payload = base64.b64encode(b"one").decode()
    viewer._on_msg(
        None,
        {"event": "save_image", "data": f"data:image/png;base64,{payload}", "filename": "shot.png"},
        [],
    )
    viewer._on_msg(
        None,
        {"event": "save_image", "data": f"data:image/png;base64,{payload}", "filename": "shot.png"},
        [],
    )
    assert Path(viewer.save_result["path"]).name == "shot-1.png"


def test_saved_image_reports_a_bad_payload(tmp_path: Path) -> None:
    """D14: a malformed data URL reports an error rather than raising."""
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    viewer._on_msg(None, {"event": "save_image", "data": "not-a-data-url", "filename": "x.png"}, [])
    assert "error" in viewer.save_result
    assert not (tmp_path / "x.png").exists()


def test_save_dir_defaults_to_cwd() -> None:
    """D14: with no ``save_dir`` the widget writes to the process directory."""
    viewer = widget.view(f1_stack())
    assert viewer.save_dir == str(Path.cwd().resolve())


def test_set_save_dir_from_the_browser(tmp_path: Path) -> None:
    """D14: a ``set_save_dir`` message retargets the folder and creates it."""
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    target = tmp_path / "shots" / "run1"
    viewer._on_msg(None, {"event": "set_save_dir", "dir": str(target)}, [])
    assert viewer.save_dir == str(target.resolve())
    assert viewer.save_result["folder"] == str(target.resolve())
    assert target.is_dir()


def test_set_save_dir_resolves_relative_under_the_current(tmp_path: Path) -> None:
    """D14: a relative folder nests under the existing save directory."""
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    viewer._on_msg(None, {"event": "set_save_dir", "dir": "shots"}, [])
    assert viewer.save_dir == str((tmp_path / "shots").resolve())


def test_set_save_dir_rejects_an_empty_path(tmp_path: Path) -> None:
    """D14: an empty folder is refused and leaves the target unchanged."""
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    viewer._on_msg(None, {"event": "set_save_dir", "dir": "  "}, [])
    assert viewer.save_dir == str(tmp_path.resolve())
    assert "error" in viewer.save_result


def test_saved_image_honors_the_new_folder(tmp_path: Path) -> None:
    """D14: after a retarget, the PNG lands in the new folder."""
    import base64

    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    viewer._on_msg(None, {"event": "set_save_dir", "dir": "shots"}, [])
    payload = base64.b64encode(b"png").decode()
    viewer._on_msg(
        None,
        {"event": "save_image", "data": f"data:image/png;base64,{payload}", "filename": "a.png"},
        [],
    )
    assert Path(viewer.save_result["path"]) == (tmp_path / "shots" / "a.png").resolve()


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
