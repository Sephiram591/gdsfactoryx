"""Tests for the detector field overlays (spec R15, D16, Q20-Q23).

The kernel half of R15 is what can be tested without a browser:

- the raw field payload a detector's result yields (R15, D16);
- that a display-setting edit is carried in the state and refused when bad
  (R15, D16);
- the strict ``show_result`` validation, which refuses a mismatched result and
  leaves the previous one untouched (Q23);
- the global ``log_range`` setting (Q22).

The ramp's alpha rules -- a signed transform transparent at the midpoint, an
``abs``/``logabs`` value transparent at the low end -- are pure browser code and
are covered by ``js/viewer/fieldramp.test.mjs``.

The F1 fixture comes from ``test_viewer_geometry`` (M0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("anywidget")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack  # noqa: E402

from gdsfactory.viewer import fields, state, widget  # noqa: E402
from gdsfactory.viewer.payload import build_field_payload  # noqa: E402


def _coarse(viewer: Any) -> None:
    """A cheap grid, so a real run fits in memory (100 nm, 4-cell PML)."""
    viewer._on_msg(
        None,
        {"event": "set_settings", "settings": {"resolution_um": 0.1, "pml_thickness": 4}},
        [],
    )


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
    """Runs the viewer's simulation synchronously and returns the result."""
    viewer.run()
    viewer._run_thread.join()
    assert viewer.last_error is None, viewer.last_error
    return viewer.last_result


# --------------------------------------------------------- R15: field payload


def test_field_payload_carries_the_raw_components() -> None:
    """R15/D16: the kernel sends the raw complex field, not a transformed one."""
    viewer = _detector_viewer()
    result = _run(viewer)
    spec = next(s for s in viewer.specs if s.category == "detector")

    entry = fields.field_payload(result, spec.name, spec.id)
    assert entry is not None
    assert entry["components"] == ["Ex", "Ey"]
    rows, cols = entry["grid"]
    real = np.frombuffer(entry["real"], dtype=np.float32)
    imag = np.frombuffer(entry["imag"], dtype=np.float32)
    # one plane per component, in the component-major layout the browser expects
    assert real.size == rows * cols * len(entry["components"])
    assert imag.size == real.size
    # a source drove the run, so the field is not all zero
    assert np.abs(real + 1j * imag).max() > 0


def test_field_payload_drops_the_normal_axis_wherever_it_sits() -> None:
    """R15: the grid is the two in-plane extents, for any plane orientation.

    A detector on the ``xz`` volume plane has its normal (y) as the *middle*
    grid axis -- the phasor is ``(1, 1, 6, nx, 1, nz)`` -- so taking the last
    two axes would drop the wide axis and texture a one-pixel sliver.
    """
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    viewer._on_msg(None, {"event": "add_detector", "port": "xz"}, [])
    result = _run(viewer)
    spec = next(s for s in viewer.specs if s.category == "detector")

    entry = fields.field_payload(result, spec.name, spec.id)
    assert entry is not None
    rows, cols = entry["grid"]
    # the xz plane is the full volume cross-section (18 x 3 um at 0.2 um), and
    # neither in-plane extent is a singleton
    assert rows > 1 and cols > 1
    real = np.frombuffer(entry["real"], dtype=np.float32)
    assert real.size == rows * cols * len(entry["components"])


def test_field_payload_is_none_for_a_detector_without_a_field() -> None:
    """R15 AC: a detector that recorded no field keeps its plain glyph."""
    viewer = _detector_viewer()
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    result = _run(viewer)
    # an EnergyDetector reduces its volume and records no in-plane grid
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": [s.id for s in viewer.specs if s.name == "o2_ModeOverlapDetector"][0],
                  "type": "EnergyDetector"}},
        [],
    )
    assert fields.field_payload(result, "o2_EnergyDetector", "x") is None


def test_build_field_payload_keys_by_spec_id() -> None:
    """R15: the payload is keyed by spec id, so the browser matches glyphs."""
    viewer = _detector_viewer()
    result = _run(viewer)
    payload = build_field_payload(viewer._viewer_state, result)
    detector_ids = {s.id for s in viewer.specs if s.category == "detector"}
    assert set(payload) == detector_ids
    # a source contributes no field
    assert all(s.category == "detector" for s in viewer.specs if s.id in payload)


# --------------------------------------------------- R15: display settings


def test_field_settings_default_and_travel_in_the_state() -> None:
    """R15: a detector's display settings are complete and ride the state."""
    viewer = _detector_viewer()
    spec = next(s for s in viewer.specs if s.category == "detector")
    settings = viewer.state["field_settings"][spec.id]
    assert settings["field_transform"] == "abs"
    assert settings["field_visible"] is True
    assert settings["field_opacity"] == 1.0
    assert settings["field_component"] is None


def test_a_display_edit_reaches_the_state() -> None:
    """R15 AC: a display edit is a state patch, with no kernel round-trip."""
    viewer = _detector_viewer()
    spec = next(s for s in viewer.specs if s.category == "detector")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_transform": "logabs",
                                            "field_component": "Ey",
                                            "field_opacity": 0.4}}},
        [],
    )
    settings = viewer.state["field_settings"][spec.id]
    assert settings["field_transform"] == "logabs"
    assert settings["field_component"] == "Ey"
    assert settings["field_opacity"] == 0.4


def test_a_bad_display_edit_is_refused() -> None:
    """R15: an unknown transform or an out-of-range opacity is refused."""
    viewer = _detector_viewer()
    spec = next(s for s in viewer.specs if s.category == "detector")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_transform": "nope"}}},
        [],
    )
    assert viewer.state["field_settings"][spec.id]["field_transform"] == "abs"
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_opacity": 5.0}}},
        [],
    )
    assert viewer.state["field_settings"][spec.id]["field_opacity"] == 1.0


def test_a_source_refuses_field_settings() -> None:
    """R15: only a detector has field display settings."""
    viewer = _detector_viewer()
    source = next(s for s in viewer.specs if s.category == "source")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "source",
         "spec": {"id": source.id, "params": {"field_transform": "abs"}}},
        [],
    )
    assert "field_transform" not in source.params


# ------------------------------------------------------- Q22: the log range


def test_log_range_defaults_and_round_trips() -> None:
    """Q22: ``log_range`` is one widget-wide setting, default 6 decades."""
    viewer = _detector_viewer()
    assert viewer.state["simulation"]["log_range"] == 6.0
    viewer._on_msg(None, {"event": "set_settings", "settings": {"log_range": 4.0}}, [])
    assert viewer.state["simulation"]["log_range"] == 4.0


def test_a_bad_log_range_is_refused() -> None:
    """Q22: a non-positive log range is refused, leaving the state unchanged."""
    viewer = _detector_viewer()
    viewer._on_msg(None, {"event": "set_settings", "settings": {"log_range": -1.0}}, [])
    assert viewer.state["simulation"]["log_range"] == 6.0


# ------------------------------------------------------ Q23: show_result


def test_show_result_adopts_a_matching_result() -> None:
    """Q23: a matching result becomes the current result and redraws overlays."""
    viewer = _detector_viewer()
    result = _run(viewer)
    # clear the fields, so the adoption is observable
    viewer.fields = {}
    viewer.show_result(result)
    assert viewer.last_result is result
    assert viewer.run_result["status"] == "done"
    assert viewer.fields, "show_result must publish the field payload"


def test_show_result_refuses_a_mismatched_result() -> None:
    """Q23 AC: a mismatched result raises and leaves the previous result alone."""
    viewer = _detector_viewer()
    result = _run(viewer)
    before_fields = dict(viewer.fields)
    before_result = viewer.last_result

    class _Cfg:
        resolution = 5e-7  # 0.5 µm, not the settings' 0.1 µm

    bad = type("Bad", (), {})()
    bad.arrays = result.arrays
    bad.objects = result.objects
    bad.config = _Cfg()

    with pytest.raises(ValueError):
        viewer.show_result(bad)
    assert viewer.last_result is before_result
    assert viewer.fields == before_fields


def test_show_result_refuses_a_result_missing_a_detector() -> None:
    """Q23: a result that does not hold every detector is refused."""
    viewer = _detector_viewer()
    result = _run(viewer)

    class _Arrays:
        detector_states: dict[str, Any] = {}

    bad = type("Bad", (), {})()
    bad.arrays = _Arrays()
    bad.objects = result.objects
    bad.config = result.config

    with pytest.raises(ValueError, match="no field for detector"):
        viewer.show_result(bad)


def test_validate_result_accepts_the_widgets_own_result() -> None:
    """Q23: a mode-1 result always validates against the setup that made it."""
    viewer = _detector_viewer()
    result = _run(viewer)
    fields.validate_result(viewer._viewer_state, result)  # must not raise


def test_the_state_carries_field_settings_for_detectors_only() -> None:
    """R15: ``to_state`` publishes one display block per detector."""
    viewer = _detector_viewer()
    blocks = viewer.state["field_settings"]
    detectors = [s for s in viewer.specs if s.category == "detector"]
    sources = [s for s in viewer.specs if s.category == "source"]
    assert set(blocks) == {s.id for s in detectors}
    assert all(s.id not in blocks for s in sources)


def test_display_settings_survive_a_rerun() -> None:
    """R15: the settings belong to the detector, not the result."""
    viewer = _detector_viewer()
    spec = next(s for s in viewer.specs if s.category == "detector")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"field_transform": "phase"}}},
        [],
    )
    _run(viewer)
    assert viewer.state["field_settings"][spec.id]["field_transform"] == "phase"
