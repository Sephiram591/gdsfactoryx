"""Tests for adopting a foreign completed simulation (spec R16, D18, R17, R18).

The kernel half of what can be tested without a browser:

- the grid→real conversion and the glyph it yields (D18);
- the lenient foreign-result validation (R16);
- ``load_result``: adoption, geometry rebuild, read-only adopted specs, the
  running guard, and garbage refused with the previous result untouched;
- the multi-frequency field payload (R17) and ``field_freq`` round-trip;
- a "None"-port spec end-to-end: glyph, placement through a real
  ``place_objects``, and parity with the port-frame path (R18).

The F1 fixture comes from ``test_viewer_geometry`` (M0); the real-run pattern
from ``test_viewer_fields.py``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("anywidget")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack  # noqa: E402

from gdsfactory.viewer import adopt, fields, sim, widget  # noqa: E402
from gdsfactory.viewer.payload import build_field_payload  # noqa: E402

COARSE = {"resolution_um": 0.1, "pml_thickness": 4}


def _coarse(viewer: Any) -> None:
    viewer._on_msg(None, {"event": "set_settings", "settings": COARSE}, [])


def _detector_viewer() -> Any:
    """A viewer with a source and one PhasorDetector on o1."""
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "o1",
         "spec": {"type": "PhasorDetector", "port": "o1",
                  "params": {"components": ["Ex", "Ey"]}}},
        [],
    )
    return viewer


def _run(viewer: Any) -> Any:
    viewer.run()
    viewer._run_thread.join()
    assert viewer.last_error is None, viewer.last_error
    return viewer.last_result


# ------------------------------------------------------- D18: glyph_from_grid


def test_glyph_from_grid_z_plane() -> None:
    """D18: a z-normal plane's glyph has the right center, extents and axis."""
    # volume 10x10x6 cells at 0.1 um; origin (box center) at (5, 5, 0.3)
    grid = (slice(2, 3), slice(0, 20), slice(0, 10))
    glyph = adopt.glyph_from_grid(grid, 0.1e-6, (10, 10, 6), (5.0, 5.0, 0.3),
                                  "d1", "det", "PhasorDetector", "detector")
    # corner = origin - 0.5*size = (4.5, 4.5, 0.0); the slice spans 0.2..0.3 um
    # from it along x, so the center sits at x = 4.75 in the local (mesh) frame
    assert glyph["axis"] == 0
    assert glyph["center"] == pytest.approx([4.75, 5.5, 0.5], abs=1e-9)
    assert glyph["u_half"] == pytest.approx(1.0, abs=1e-9)   # y half (in-plane min axis)
    assert glyph["v_half"] == pytest.approx(0.5, abs=1e-9)   # z half (in-plane max axis)
    assert glyph["port"] == "None"
    assert glyph["kind"] == "detector"


def test_glyph_from_grid_x_plane_has_axis_and_halves() -> None:
    """D18: the normal axis is the singleton; u/v are the in-plane halves."""
    grid = (slice(0, 20), slice(4, 5), slice(0, 12))
    glyph = adopt.glyph_from_grid(grid, 0.1e-6, (10, 10, 6), (5.0, 5.0, 0.3),
                                  "d1", "det", "PhasorDetector", "detector")
    assert glyph["axis"] == 1
    # u = x half = 1.0 um, v = z half = 0.6 um
    assert glyph["u_half"] == pytest.approx(1.0, abs=1e-9)
    assert glyph["v_half"] == pytest.approx(0.6, abs=1e-9)


def test_glyph_from_grid_volume_detector_is_a_point() -> None:
    """D18: a volume detector (no singleton axis) collapses to a point glyph."""
    grid = (slice(0, 20), slice(0, 20), slice(0, 6))
    glyph = adopt.glyph_from_grid(grid, 0.1e-6, (10, 10, 6), (5.0, 5.0, 0.3),
                                  "d1", "det", "PhasorDetector", "detector")
    assert glyph["u_half"] == 0.0
    assert glyph["v_half"] == 0.0


def test_adopt_result_builds_none_port_specs() -> None:
    """D18: one read-only spec per detector, named from the placed object."""
    viewer = _detector_viewer()
    result = _run(viewer)
    from gdsfactory.viewer.state import ViewerState

    fresh = ViewerState(viewer.stack)
    specs = adopt.adopt_result(fresh, result.objects, result.config)
    assert len(specs) == len(result.objects.detectors)
    for spec, obj in zip(specs, result.objects.detectors, strict=True):
        assert spec.port == "None"
        assert spec.params["name"] == obj.name
        assert spec.id in fresh.adopted
        assert spec.glyph is not None


# ----------------------------------------------- R16: validate_foreign_result


def test_validate_foreign_result_accepts_its_own_run() -> None:
    """R16: the widget's own result passes the lenient validation."""
    viewer = _detector_viewer()
    result = _run(viewer)
    fields.validate_foreign_result(result.objects, result.arrays, result.config,
                                   stack=viewer.stack)  # must not raise


def test_validate_foreign_result_refuses_missing_pieces() -> None:
    """R16: missing detector_states, detectors/volume or resolution is refused."""

    class _Arrays:
        pass

    class _Config:
        resolution = 1e-7

    viewer = _detector_viewer()
    result = _run(viewer)
    with pytest.raises(ValueError, match="detector_states"):
        fields.validate_foreign_result(result.objects, _Arrays(), result.config)
    with pytest.raises(ValueError, match="resolution"):
        fields.validate_foreign_result(result.objects, result.arrays, object())


def test_validate_foreign_result_refuses_a_mismatched_stack() -> None:
    """R16: a stack whose box differs from the result's grid by > 1 cell."""
    viewer = _detector_viewer()
    result = _run(viewer)
    # a re-boxed stack: 2 um narrower in x (20 cells at 0.1 um >> tolerance);
    # partial_real_shape is in meters, so 2 um is 2e-6
    from gdsfactory.viewer.state import _rebox_stack

    other = _rebox_stack(viewer.stack,
                         tuple(v - 2.0e-6 for v in viewer.stack.partial_real_shape),
                         (viewer.stack.center[0], viewer.stack.center[1]),
                         viewer.stack.z_center)
    with pytest.raises(ValueError, match="does not fit the stack"):
        fields.validate_foreign_result(result.objects, result.arrays, result.config,
                                       stack=other)


# ------------------------------------------------------- R16/D18: load_result


def test_load_result_same_stack_adopts_detectors() -> None:
    """R16: adoption adds one read-only spec per detector and re-shades."""
    viewer = _detector_viewer()
    result = _run(viewer)
    before_geometry = viewer.geometry
    viewer.load_result(*result)
    specs = [s for s in viewer.specs if s.category == "detector"]
    assert specs, "adoption must create detector specs"
    for spec, obj in zip(specs, result.objects.detectors, strict=True):
        assert spec.port == "None"
        assert spec.params["name"] == obj.name
    wire = viewer.state
    for entry in wire["detectors"]:
        assert entry["adopted"] is True
    # fields keyed by the adopted ids, matching the detector names
    names = {s.params["name"] for s in specs}
    assert names == set(result.arrays.detector_states)
    # geometry was re-sent (same stack, so the same box)
    assert viewer.geometry["size"] == before_geometry["size"]
    # the old run's specs are gone: only the adopted ones remain
    assert not any(s.port == "o1" and s.id not in [x.id for x in specs]
                   for s in viewer.specs)


def test_load_result_different_stack_rebuilds_geometry() -> None:
    """R16: a foreign stack's box becomes the viewer's geometry."""
    viewer = _detector_viewer()
    result = _run(viewer)
    from gdsfactory.viewer.state import _rebox_stack

    # one cell wider in x (0.1 um at the coarse settings) — within the
    # one-cell stack-match tolerance, but a different box
    other = _rebox_stack(viewer.stack,
                         (viewer.stack.partial_real_shape[0] + 1.0e-7,
                          *viewer.stack.partial_real_shape[1:]),
                         (viewer.stack.center[0], viewer.stack.center[1]),
                         viewer.stack.z_center)
    viewer.load_result(result.objects, result.arrays, result.config, stack=other)
    assert viewer.stack is other
    # geometry size is in um, partial_real_shape in meters
    assert viewer.geometry["size"] == pytest.approx(
        [float(v) * 1e6 for v in other.partial_real_shape], abs=1e-5)
    # the old specs were replaced by the adopted ones
    assert all(s.id in viewer._viewer_state.adopted for s in viewer.specs)


def test_adopted_specs_are_read_only() -> None:
    """R16: display edits pass; type/port/position edits and removal are refused."""
    viewer = _detector_viewer()
    result = _run(viewer)
    viewer.load_result(*result)
    spec = next(s for s in viewer.specs if s.category == "detector")

    # a display edit is accepted
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_transform": "logabs"}}},
        [],
    )
    assert viewer.state["field_settings"][spec.id]["field_transform"] == "logabs"

    # a setup edit is refused, and the state re-syncs unchanged
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "type": "EnergyDetector"}},
        [],
    )
    assert spec.type == "PhasorDetector"
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"port_offset": 0.2}}},
        [],
    )
    assert "port_offset" not in spec.params
    with pytest.raises(ValueError, match="cannot be removed"):
        viewer._viewer_state.remove_spec(spec.id)


def test_load_result_garbage_leaves_the_result_untouched() -> None:
    """R16 AC: a refused result raises and the previous traits stay."""
    viewer = _detector_viewer()
    result = _run(viewer)
    viewer.load_result(*result)
    before_fields = dict(viewer.fields)
    before_specs = list(viewer.specs)

    class _Arrays:
        pass  # no detector_states attribute at all: refused by R16's validation

    with pytest.raises(ValueError):
        viewer.load_result(result.objects, _Arrays(), result.config)
    assert viewer.last_result.objects is result.objects
    assert viewer.fields == before_fields
    assert viewer.specs == before_specs


def test_load_result_while_running_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """R16: the run lock holds — a load_result in flight raises RuntimeError."""
    viewer = _detector_viewer()
    result = _run(viewer)

    import threading

    import gdsfactory.viewer.widget as wmod

    # a fake run that blocks until released, so `running` stays true
    release = threading.Event()
    started = threading.Event()

    def fake(*a: Any, **k: Any) -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(wmod, "_run_sim", fake)
    viewer.run()
    assert started.wait(5)
    try:
        with pytest.raises(RuntimeError, match="in flight"):
            viewer.load_result(*result)
    finally:
        release.set()
        viewer._run_thread.join()
        viewer._viewer_state.running = False


def test_build_objects_skips_adopted_specs() -> None:
    """D18: an adopted spec is never re-created for a later run."""
    from fdtdx.objects.detectors.detector import Detector

    viewer = _detector_viewer()
    result = _run(viewer)
    viewer.load_result(*result)
    objects, config, constraints = viewer._prepare()
    names = {o.name for o in objects if isinstance(o, Detector)}
    foreign = {s.params["name"] for s in viewer.specs if s.id in viewer._viewer_state.adopted}
    assert foreign and not (names & foreign)


# ----------------------------------------------------- R17: frequency frames


def test_multi_frequency_payload_carries_all_frames() -> None:
    """R17: every frequency frame travels, with freqs and freq = 0."""
    viewer = _detector_viewer()
    # three frequencies on the detector
    spec = next(s for s in viewer.specs if s.category == "detector")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"wavelengths": [1.5, 1.55, 1.6]}}},
        [],
    )
    result = _run(viewer)
    spec = next(s for s in viewer.specs if s.category == "detector")
    entry = fields.field_payload(result, spec.name, spec.id)
    assert entry is not None
    assert len(entry["freqs"]) == 3
    assert entry["freq"] == 0
    rows, cols = entry["grid"]
    n = rows * cols * len(entry["components"]) * 3
    assert np.frombuffer(entry["real"], dtype=np.float32).size == n
    # the frequencies come from the wave characters: c / lambda
    c = 299792458.0
    assert entry["freqs"] == pytest.approx([c / (w * 1e-6) for w in (1.5, 1.55, 1.6)],
                                           rel=1e-3)


def test_field_freq_round_trips_through_the_patch_path() -> None:
    """R17: field_freq is a display param, patched like the others."""
    viewer = _detector_viewer()
    spec = next(s for s in viewer.specs if s.category == "detector")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_freq": 2}}},
        [],
    )
    settings = viewer.state["field_settings"][spec.id]
    assert settings["field_freq"] == 2
    assert spec.field_params()["field_freq"] == 2
    # an out-of-range or non-int stays through validation (clamped client-side)
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_freq": 0}}},
        [],
    )
    assert viewer.state["field_settings"][spec.id]["field_freq"] == 0


def test_single_frequency_payload_layout_is_unchanged() -> None:
    """R17: a single-frequency run's buffers are one frame, as before."""
    viewer = _detector_viewer()
    result = _run(viewer)
    spec = next(s for s in viewer.specs if s.category == "detector")
    entry = fields.field_payload(result, spec.name, spec.id)
    assert entry is not None
    rows, cols = entry["grid"]
    n = rows * cols * len(entry["components"])
    assert np.frombuffer(entry["real"], dtype=np.float32).size == n


# ------------------------------------------------------ R18: the "None" port


def test_none_port_spec_renders_an_absolute_glyph() -> None:
    """R18: a "None"-port spec's glyph sits where the params say.

    Glyphs are in the meshes' local frame (center minus the box origin,
    spec C5), so the wire center is the params' stack-frame center shifted.
    """
    viewer = _detector_viewer()
    origin = viewer._viewer_state.origin
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "o1",
         "spec": {"type": "PhasorDetector", "port": "None",
                  "params": {"axis": 2, "center": [5.0, 5.0, 0.3],
                             "size": [4.0, 1.0], "components": ["Ex"]}}},
        [],
    )
    spec = next(s for s in viewer.specs if s.port == "None")
    glyph = next(g for g in viewer.state["glyphs"] if g["id"] == spec.id)
    expected = [c - o for c, o in zip((5.0, 5.0, 0.3), origin, strict=True)]
    assert glyph["center"] == pytest.approx(expected, abs=1e-6)
    assert glyph["axis"] == 2
    assert glyph["u_half"] == pytest.approx(2.0, abs=1e-6)
    assert glyph["v_half"] == pytest.approx(0.5, abs=1e-6)
    # editing center moves the glyph
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"center": [6.0, 4.0, 0.3]}}},
        [],
    )
    glyph = next(g for g in viewer.state["glyphs"] if g["id"] == spec.id)
    expected = [c - o for c, o in zip((6.0, 4.0, 0.3), origin, strict=True)]
    assert glyph["center"] == pytest.approx(expected, abs=1e-6)


def test_none_port_spec_places_through_place_objects() -> None:
    """R18: an absolute spec's constraint resolves to the requested center."""
    import jax

    import fdtdx

    viewer = _detector_viewer()
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "o1",
         "spec": {"type": "PhasorDetector", "port": "None",
                  "params": {"axis": 0, "center": [6.0, 0.0, 0.2],
                             "size": [1.0, 0.5], "components": ["Ex"]}}},
        [],
    )
    spec = next(s for s in viewer.specs if s.port == "None")
    objects, config, constraints = viewer._prepare()
    placed = fdtdx.place_objects(objects, config, constraints,
                                 jax.random.PRNGKey(0))[0]
    obj = next(o for o in placed.detectors if o.name == spec.name)
    # the placement resolves in the mesh local frame: the volume corner sits at
    # origin - 0.5 * size (adopt.grid_bounds_um's same math), in um
    res_um = float(config.resolution) * 1e6
    volume = next(o for o in placed.objects if o is placed.volume)
    origin = viewer._viewer_state.origin
    size_um = [float(v) * 1e6 for v in volume.partial_real_shape]
    center = [
        0.5 * ((origin[i] - 0.5 * size_um[i] + obj.grid_slice[i].start * res_um)
               + (origin[i] - 0.5 * size_um[i] + obj.grid_slice[i].stop * res_um))
        for i in range(3)
    ]
    assert center == pytest.approx([6.0, 0.0, 0.2], abs=2.0 * res_um)


def test_absolute_placement_matches_the_port_path() -> None:
    """R18: a detector on a port and one placed absolutely at the same center
    land on the same grid cells (tolerance 1 cell)."""
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_detector", "port": "o1",
                          "spec": {"type": "PhasorDetector", "port": "o1"}}, [])
    port_spec = next(s for s in viewer.specs if s.port == "o1")
    glyph = next(g for g in viewer.state["glyphs"] if g["id"] == port_spec.id)

    # the same detector, placed absolutely at the port glyph's frame
    viewer2 = widget.view(f1_stack())
    _coarse(viewer2)
    size = [2 * glyph["u_half"], 2 * glyph["v_half"]]
    center = [c + o for c, o in zip(glyph["center"], viewer2._viewer_state.origin)]
    viewer2._on_msg(
        None,
        {"event": "add_detector", "port": "o1",
         "spec": {"type": "PhasorDetector", "port": "None",
                  "params": {"axis": glyph["axis"], "center": center, "size": size}}},
        [],
    )
    abs_spec = next(s for s in viewer2.specs if s.port == "None")

    def slice_of(viewer_obj: Any, spec: Any) -> tuple[int, ...]:
        import jax

        import fdtdx

        objects, config, constraints = viewer_obj._prepare()
        placed = fdtdx.place_objects(objects, config, constraints,
                                     jax.random.PRNGKey(0))[0]
        obj = next(o for o in placed.detectors if o.name == spec.params.get("name")
                   or o.name == f"{spec.port}_{spec.type}")
        return tuple(int(v) for sl in obj.grid_slice for v in (sl.start, sl.stop))

    a = slice_of(viewer, port_spec)
    b = slice_of(viewer2, abs_spec)
    assert all(abs(x - y) <= 1 for x, y in zip(a, b, strict=True))


def test_switching_none_back_to_a_port_restores_the_port_path() -> None:
    """R18: port None -> o1 restores the port-frame glyph and params."""
    viewer = _detector_viewer()
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "o1",
         "spec": {"type": "PhasorDetector", "port": "None",
                  "params": {"axis": 2, "center": [5.0, 5.0, 0.3],
                             "size": [4.0, 1.0]}}},
        [],
    )
    spec = next(s for s in viewer.specs if s.port == "None")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "port": "o1"}},
        [],
    )
    assert spec.port == "o1"
    glyph = next(g for g in viewer.state["glyphs"] if g["id"] == spec.id)
    # the port frame glyph is at the port, not at the old absolute center
    assert glyph["center"] != pytest.approx([5.0, 5.0, 0.3], abs=1e-6)


def test_synthetic_spec_round_trips_as_a_payload() -> None:
    """R16: an adopted spec's wire entry survives ObjectSpec.from_wire."""
    viewer = _detector_viewer()
    result = _run(viewer)
    viewer.load_result(*result)
    spec = next(s for s in viewer.specs if s.category == "detector")
    entry = next(e for e in viewer.state["detectors"] if e["id"] == spec.id)
    back = sim.ObjectSpec.from_wire(entry)
    assert back.type == spec.type
    assert back.port == spec.port
    assert back.params["name"] == spec.params["name"]
    assert back.params["center"] == spec.params["center"]
