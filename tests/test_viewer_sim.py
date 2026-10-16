"""Tests for the viewer's sources, detectors, settings and run (spec M4).

M4's exit criteria are R12's and R13's acceptance criteria plus R14 mode 2. The
parts that do not need a browser are tested here:

- the declarative spec model and the defaults that come from the settings (R12,
  R13, D11);
- that adding or editing a spec calls no ``create_*_at`` (D11);
- that retrieval produces ``objects``/``config``/``constraints`` that
  ``fdtdx.place_objects`` accepts (R12, R14 mode 2);
- that the PML thickness reaches the prepared constraints (R13);
- that a mode-1 run does not block and locks the setup while it runs (D15).

The visual half -- the 3D planes, the explorer rows and the double-click -- is
the user's marimo check (there is no browser on the compute node).

The F1 fixture comes from ``test_viewer_geometry`` (M0).
"""

from __future__ import annotations

import threading
import time
from typing import Any, ClassVar

import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("anywidget")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack  # noqa: E402

from gdsfactory.viewer import sim, state, widget  # noqa: E402


def _coarse(viewer: Any) -> None:
    """Sets a cheap grid, so ``place_objects`` does not allocate a huge volume.

    F1's box at the default 20 nm resolution needs ~33 GiB on an L40S; 100 nm
    with a 4-cell PML fits comfortably and still exercises placement.
    """
    viewer._on_msg(
        None,
        {"event": "set_settings", "settings": {"resolution_um": 0.1, "pml_thickness": 4}},
        [],
    )


# ------------------------------------------------------------------ R12: specs


def test_sources_and_detectors_start_empty() -> None:
    """R12: the Sources and Detectors tabs start empty."""
    viewer_state = state.ViewerState(f1_stack())
    assert viewer_state.specs == []
    wire = viewer_state.to_state()
    assert wire["sources"] == []
    assert wire["detectors"] == []
    assert wire["glyphs"] == []


def test_double_click_adds_a_mode_overlap_detector_on_the_port() -> None:
    """R12 AC: double-clicking o2 adds a ModeOverlapDetector row on o2.

    The gesture is the browser's; the kernel message is what this checks.
    """
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    (spec,) = viewer.specs
    assert spec.type == "ModeOverlapDetector"
    assert spec.category == "detector"
    assert spec.port == "o2"
    assert viewer.state["detectors"][0]["port"] == "o2"


def test_new_source_uses_the_default_type() -> None:
    """R12: the "New source" control adds a ModePlaneSource (Q18)."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    (spec,) = viewer.specs
    assert spec.type == "ModePlaneSource"
    assert spec.category == "source"


def test_glyph_sits_at_the_port_frame() -> None:
    """R12 AC: the added detector is drawn at o2, before any placement (D11)."""
    viewer_state = state.ViewerState(f1_stack())
    spec = viewer_state.add_spec(sim.default_detector_spec("o2", viewer_state.settings))
    (glyph,) = state.spec_glyphs(viewer_state)
    assert glyph["id"] == spec.id
    assert glyph["port"] == "o2"

    # the glyph center is the port frame's center, in the local (box-center)
    # frame the meshes use (spec C5, D11)
    _, _, _, center, _ = viewer_state.stack._port_frame("o2", None, 0.0)
    expected = [center[i] - viewer_state.origin[i] for i in range(3)]
    assert glyph["center"] == pytest.approx(expected)
    assert glyph["u_half"] > 0 and glyph["v_half"] > 0


def test_editing_port_offset_moves_the_glyph_without_creating(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12 AC: editing port_offset moves the plane, with no ``create_*_at`` call.

    This is D11's contract: an edit only touches the spec. ``create_*_at`` runs
    on retrieval, so the spy must stay empty across the edit.
    """
    viewer_state = state.ViewerState(f1_stack())
    calls = _spy_create(monkeypatch)
    spec = viewer_state.add_spec(sim.default_detector_spec("o2", viewer_state.settings))

    before = state.spec_glyphs(viewer_state)[0]["center"]
    viewer_state.update_spec({"id": spec.id, "params": {"port_offset": 0.5}})
    after = state.spec_glyphs(viewer_state)[0]["center"]

    assert calls == [], "an edit must not call create_*_at (spec D11)"
    assert before != after, "port_offset must move the glyph"


def test_adding_a_spec_does_not_create(monkeypatch: pytest.MonkeyPatch) -> None:
    """D11: adding a spec is declarative; nothing is created until retrieval."""
    viewer_state = state.ViewerState(f1_stack())
    calls = _spy_create(monkeypatch)
    viewer_state.add_spec(sim.default_source_spec("o1", viewer_state.settings))
    assert calls == []


def test_unknown_port_is_rejected() -> None:
    """R12: a spec on a port the device does not have is refused."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "nope"}, [])
    assert viewer.specs == []


def test_a_spec_can_be_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12: a source/detector has a visibility toggle, carried in the glyph.

    Visibility is display-only (the pre-placement toggle); it must not reach
    ``create_*_at``.
    """
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    (spec,) = viewer.specs
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector", "spec": {"id": spec.id, "params": {"visible": False}}},
        [],
    )
    (glyph,) = viewer.state["glyphs"]
    assert glyph["visible"] is False

    calls = _spy_create(monkeypatch)
    _ = viewer.objects
    assert calls == ["ModeOverlapDetector"], "a hidden object is still created on retrieval"


# ------------------------------------------------------------- R13: settings


def test_default_wavelength_feeds_a_new_detector() -> None:
    """R13 AC: the default wavelength changes a subsequently added detector."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "set_settings", "settings": {"wavelength_um": 1.31}}, [])
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    (spec,) = viewer.specs
    assert spec.params["wavelengths"] == [1.31]


def test_default_port_size_feeds_a_new_source() -> None:
    """R13/Q18: a new plane's size comes from the settings' port_size_mult."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "set_settings", "settings": {"port_size_mult": [2.0, 4.0]}}, [])
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    (spec,) = viewer.specs
    assert spec.params["port_size_mult"] == [2.0, 4.0]


def test_pml_thickness_reaches_the_constraints() -> None:
    """R13 AC: the PML thickness field is in the prepared constraints (D11)."""
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    objects, _, constraints = viewer.objects, viewer.config, viewer.constraints

    thicknesses = sorted(
        obj.partial_grid_shape[obj.axis]
        for obj in objects
        if type(obj).__name__ == "PerfectlyMatchedLayer"
    )
    assert thicknesses == [4] * 6
    assert constraints, "the boundaries contribute constraints"


def test_settings_edit_is_visible_in_the_state() -> None:
    """R13: the settings round-trip through the state the browser reads."""
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "set_settings", "settings": {"time_s": 1e-12}}, [])
    assert viewer.state["simulation"]["time_s"] == 1e-12


def test_a_bad_settings_edit_leaves_the_state_unchanged() -> None:
    """R13: a non-positive setting is refused, not stored."""
    viewer = widget.view(f1_stack())
    before = viewer.settings.resolution_um
    viewer._on_msg(None, {"event": "set_settings", "settings": {"resolution_um": -1.0}}, [])
    assert viewer.settings.resolution_um == before


# ------------------------------------------------------- R12/R14 mode 2: export


def test_retrieval_is_accepted_by_place_objects() -> None:
    """R12/R14 AC: the retrieval returns objects/constraints place_objects accepts.

    No FDTD run: placement only.
    """
    import jax

    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])

    objects, config, constraints = viewer.objects, viewer.config, viewer.constraints
    assert objects and constraints

    placed, _, _, _, _ = fdtdx.place_objects(objects, config, constraints, jax.random.PRNGKey(0))
    assert type(placed).__name__ == "ObjectContainer"


# --------------------------------------------------------- R12: volume planes


def test_plane_ports_are_offered_but_are_not_device_ports() -> None:
    """R12: xy/xz/yz are port choices, but not ports of the device."""
    stack = f1_stack()
    viewer_state = state.ViewerState(stack)
    # R1's AC still holds: the device's own ports are unchanged
    assert [port.name for port in viewer_state.ports] == ["o1", "o2"]
    assert {"xy", "xz", "yz"}.isdisjoint({p.name for p in stack.ports})
    # ... and they are offered to the explorer, after the device's ports
    meta = viewer_state.plane_ports
    assert [port.name for port in meta] == ["xy", "xz", "yz"]
    assert {"xy", "xz", "yz"} <= set(viewer_state.port_visible)


def test_plane_size_one_fills_the_simulation_plane() -> None:
    """R12/Q18: ``port_size_mult=(1, 1)`` on a plane fills the whole plane."""
    stack = f1_stack()
    size = tuple(float(s) * 1e6 for s in stack.partial_real_shape)
    for plane, (u, v) in (("yz", (1, 2)), ("xz", (0, 2)), ("xy", (0, 1))):
        _p, axis, sign, center, height = stack._port_frame(plane, None, 0.0)
        assert axis == {"yz": 0, "xz": 1, "xy": 2}[plane]
        assert sign == 1
        # centered in the volume
        assert center[0] == pytest.approx(stack.center[0])
        assert center[1] == pytest.approx(stack.center[1])
        assert center[2] == pytest.approx(stack.z_center)
        # (1, 1) spans the box's two in-plane extents
        assert height == pytest.approx(size[v])
        assert _p.width == pytest.approx(size[u])
        obj, _ = stack.create_uniform_plane_source_at(
            plane, (1.0, 1.0), wave_character=sim._wave_character(1.55), name=f"{plane}_s"
        )
        real = obj.partial_real_shape
        # the two in-plane extents are the box's; the normal axis is grid-shaped
        # (None in partial_real_shape, 1 in partial_grid_shape)
        assert float(real[u]) * 1e6 == pytest.approx(size[u])
        assert float(real[v]) * 1e6 == pytest.approx(size[v])
        assert real[axis] is None
        assert obj.partial_grid_shape[axis] == 1


def test_a_plane_honors_its_level() -> None:
    """R12: a plane's ``level`` sets its height (it is not ignored).

    The ``xy`` plane's normal is z, so the level *is* the plane's position: a
    level name puts it at that level's span middle, and ``z_center`` at the
    volume's mid-height. An ``xz`` / ``yz`` plane's z is in-plane, so the level
    moves its vertical center there instead.
    """
    from gdsfactory.fdtdx_stack import Z_CENTER_LEVEL

    stack = f1_stack()
    # the levels' span middles (F1: lower -0.17, upper +0.10, oxide -2.5)
    for level, expected in (("lower", -0.17), ("upper", 0.10)):
        _p, axis, _sign, center, _h = stack._port_frame("xy", level, 0.0)
        assert axis == 2
        assert center[2] == pytest.approx(expected), f"xy plane on {level}"

    # z_center (and no level) pin the volume's mid-height
    for level in (None, Z_CENTER_LEVEL):
        _p, _axis, _sign, center, _h = stack._port_frame("xy", level, 0.0)
        assert center[2] == pytest.approx(stack.z_center)

    # an xz plane's normal is y; its level moves its in-plane z center
    _p, axis, _sign, center, _h = stack._port_frame("xz", "upper", 0.0)
    assert axis == 1
    assert center[2] == pytest.approx(0.10)
    assert center[1] == pytest.approx(stack.center[1])

    # a level a port does not draw still resolves for a plane (no layer match)
    _p, _axis, _sign, center, _h = stack._port_frame("xy", "oxide", 0.0)
    assert center[2] == pytest.approx(-2.5)

    # an unknown level raises
    with pytest.raises(KeyError):
        stack._port_frame("xy", "nope", 0.0)


def test_z_center_keeps_a_ports_span_height() -> None:
    """R12: ``z_center`` on a *port* moves z but keeps the level's height."""
    from gdsfactory.fdtdx_stack import Z_CENTER_LEVEL

    stack = f1_stack()
    _p, _axis, _sign, center, height = stack._port_frame("o2", Z_CENTER_LEVEL, 0.0)
    assert center[2] == pytest.approx(stack.z_center)
    assert height == pytest.approx(0.1), "the lower level's height is unchanged"


def test_a_plane_source_places_like_a_port_source() -> None:
    """R12/R14: a source on a volume plane is accepted by ``place_objects``."""
    import jax

    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_source", "port": "xy"}, [])
    assert viewer.state["sources"][-1]["port"] == "xy"

    objects, config, constraints = viewer.objects, viewer.config, viewer.constraints
    placed, _, _, _, _ = fdtdx.place_objects(objects, config, constraints, jax.random.PRNGKey(0))
    assert type(placed).__name__ == "ObjectContainer"


def test_export_shares_one_build() -> None:
    """D11: ``objects``/``config``/``constraints`` come from one build.

    The constraint names must match the objects they were built with; a second
    independent build would leave them referring to objects from the first.
    """
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    assert viewer.objects is viewer.objects
    assert viewer.constraints is viewer.constraints


def test_a_setup_edit_rebuilds_the_export() -> None:
    """D11: an edit bumps the revision, so the next retrieval rebuilds."""
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    before = viewer.objects
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    assert viewer.objects is not before
    assert len(viewer.objects) > len(before)


# ------------------------------------------------------------- D15: background run


class _FakeResult:
    """A stand-in for :class:`sim.RunResult` with the shape the widget reads."""

    class _Arrays:
        detector_states: ClassVar[dict[str, Any]] = {"o2_ModeOverlapDetector": object()}

    arrays = _Arrays()


def _slow_run(monkeypatch: pytest.MonkeyPatch, seconds: float = 0.6) -> None:
    """Replaces the kernel's run with a slow stub, so D15 is testable."""

    def fake(
        stack: Any,
        viewer_state: Any,
        *,
        key: Any = None,
        progress: Any = None,
        phase: Any = None,
        cancel: Any = None,
        keep_partial: bool = False,
    ) -> Any:
        time.sleep(seconds)
        return _FakeResult()

    monkeypatch.setattr(widget, "_run_sim", fake)


def test_run_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: ``.run()`` returns at once; the widget stays interactive."""
    _slow_run(monkeypatch)
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])

    started = time.monotonic()
    viewer.run()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2, "run() must return immediately (spec D15)"
    assert viewer.running and viewer.state["running"] is True
    assert viewer.run_result == {"status": "running"}
    viewer._run_thread.join()


def test_setup_is_locked_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: the kernel refuses every setup edit while a run is in flight."""
    _slow_run(monkeypatch)
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    viewer.run()
    try:
        specs_before = list(viewer.specs)
        wavelength_before = viewer.settings.wavelength_um

        viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
        viewer._on_msg(None, {"event": "set_settings", "settings": {"wavelength_um": 9.9}}, [])
        viewer._on_msg(None, {"event": "set_state", "patch": {"simulation": {"wavelength_um": 9.9}}}, [])

        assert viewer.specs == specs_before
        assert viewer.settings.wavelength_um == wavelength_before
    finally:
        viewer._run_thread.join()


def test_read_only_edits_still_work_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: tab switches, the viewports and styling stay live while running."""
    _slow_run(monkeypatch)
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    viewer.run()
    try:
        viewer._on_msg(None, {"event": "set_state", "patch": {"plane": "yz"}}, [])
        viewer._on_msg(
            None, {"event": "set_state", "patch": {"styles": {"upper": {"visible": False}}}}, []
        )
        assert viewer.state["plane"] == "yz"
        assert viewer.state["styles"]["upper"]["visible"] is False
    finally:
        viewer._run_thread.join()


def test_setup_unlocks_after_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: the lock clears when the run finishes, and the result is recorded."""
    _slow_run(monkeypatch, seconds=0.05)
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "add_detector", "port": "o2"}, [])
    viewer.run()
    viewer._run_thread.join()

    assert viewer.running is False
    assert viewer.state["running"] is False
    assert viewer.run_result["status"] == "done"
    assert viewer.run_result["detectors"] == ["o2_ModeOverlapDetector"]
    assert isinstance(viewer.last_result, _FakeResult)

    # the setup accepts edits again
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    assert len(viewer.specs) == 2


def test_run_reports_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15/Q20: a failing run records the error and still unlocks the setup."""

    def boom(
        stack: Any,
        viewer_state: Any,
        *,
        key: Any = None,
        progress: Any = None,
        phase: Any = None,
        cancel: Any = None,
        keep_partial: bool = False,
    ) -> Any:
        raise RuntimeError("no GPU")

    monkeypatch.setattr(widget, "_run_sim", boom)
    viewer = widget.view(f1_stack())
    viewer.run()
    viewer._run_thread.join()

    assert viewer.running is False
    assert viewer.run_result == {"status": "error", "message": "no GPU"}
    assert isinstance(viewer.last_error, RuntimeError)


def test_a_second_run_is_refused_while_one_is_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: a run already in flight cannot be started again."""
    _slow_run(monkeypatch)
    viewer = widget.view(f1_stack())
    viewer.run()
    try:
        with pytest.raises(RuntimeError):
            viewer.run()
    finally:
        viewer._run_thread.join()


def test_the_run_message_starts_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """R14 mode 1: the browser's Run action reaches the kernel as a run."""
    _slow_run(monkeypatch, seconds=0.05)
    viewer = widget.view(f1_stack())
    viewer._on_msg(None, {"event": "run"}, [])
    assert viewer.running is True
    viewer._run_thread.join()
    assert viewer.run_result["status"] == "done"


def test_run_uses_a_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: the run happens off the kernel's main thread."""
    _slow_run(monkeypatch, seconds=0.05)
    viewer = widget.view(f1_stack())
    viewer.run()
    assert isinstance(viewer._run_thread, threading.Thread)
    assert viewer._run_thread is not threading.main_thread()
    viewer._run_thread.join()


def test_run_forwards_fdtdx_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15/R14: fdtdx's progress callback is mirrored into the ``progress`` trait."""
    seen: list[tuple[int, int]] = []

    def fake(
        stack: Any,
        viewer_state: Any,
        *,
        key: Any = None,
        progress: Any = None,
        phase: Any = None,
        cancel: Any = None,
        keep_partial: bool = False,
    ) -> Any:
        assert progress is not None, "the widget must pass a progress callback (spec D15)"
        progress(1, 4)
        seen.append((viewer.progress["step"], viewer.progress["total"]))
        progress(4, 4)
        return _FakeResult()

    monkeypatch.setattr(widget, "_run_sim", fake)
    viewer = widget.view(f1_stack())
    assert viewer.progress == {}
    viewer.run()
    viewer._run_thread.join()

    # the callback fired mid-run and was visible on the trait at that moment
    assert seen == [(1, 4)]
    # and it is cleared once the run finishes
    assert viewer.progress == {}


def test_sim_run_turns_off_fdtdx_tqdm_and_forwards_the_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D15: ``sim.run`` passes the callback to ``run_fdtd`` and silences tqdm."""
    import fdtdx

    captured: dict[str, Any] = {}

    def fake_run_fdtd(*, arrays: Any, objects: Any, config: Any, key: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return None, arrays

    monkeypatch.setattr(fdtdx, "run_fdtd", fake_run_fdtd)
    monkeypatch.setattr(sim, "build_objects", lambda _stack, _state: ([], object(), []))
    monkeypatch.setattr(fdtdx, "place_objects", lambda *_a, **_k: (None, None, {}, None, None))

    from gdsfactory import fdtdx_stack

    monkeypatch.setattr(
        fdtdx_stack, "apply_stack", lambda _arrays, _objects, _params, _key: (None, None, None)
    )
    monkeypatch.setattr(fdtdx_stack, "init_stack_params", lambda _objects: {})

    seen: list[tuple[int, int]] = []

    def cb(step: int, total: int) -> None:
        seen.append((step, total))

    sim.run(object(), object(), progress=cb)
    # fdtdx gets a wrapper (so the phase flip rides the same callback), which
    # forwards the fraction through to the caller's callback
    captured["progress_callback"](2, 5)
    assert seen == [(2, 5)]
    assert captured["show_progress"] is False


def test_sim_run_reports_each_phase_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: ``sim.run`` calls the ``phase`` callback placing → compiling → stepping."""
    import fdtdx

    phases: list[str] = []

    def fake_run_fdtd(*, arrays: Any, objects: Any, config: Any, key: Any, **kwargs: Any) -> Any:
        # the callback is the first time step, so "stepping" flips only here
        kwargs["progress_callback"](1, 10)
        return None, arrays

    monkeypatch.setattr(fdtdx, "run_fdtd", fake_run_fdtd)
    monkeypatch.setattr(sim, "build_objects", lambda _stack, _state: ([], object(), []))
    monkeypatch.setattr(fdtdx, "place_objects", lambda *_a, **_k: (None, None, {}, None, None))

    from gdsfactory import fdtdx_stack

    monkeypatch.setattr(
        fdtdx_stack, "apply_stack", lambda _arrays, _objects, _params, _key: (None, None, None)
    )
    monkeypatch.setattr(fdtdx_stack, "init_stack_params", lambda _objects: {})

    sim.run(object(), object(), phase=phases.append)
    assert phases == ["placing", "compiling", "stepping"]


def test_run_mirrors_the_phase_into_the_trait(monkeypatch: pytest.MonkeyPatch) -> None:
    """D15: the phase callback's names land on the ``phase`` trait and clear after."""
    seen: list[str] = []

    def fake(
        stack: Any,
        viewer_state: Any,
        *,
        key: Any = None,
        progress: Any = None,
        phase: Any = None,
        cancel: Any = None,
        keep_partial: bool = False,
    ) -> Any:
        assert phase is not None, "the widget must pass a phase callback (spec D15)"
        phase("compiling")
        seen.append(viewer.phase)
        phase("stepping")
        seen.append(viewer.phase)
        return _FakeResult()

    monkeypatch.setattr(widget, "_run_sim", fake)
    viewer = widget.view(f1_stack())
    viewer.run()
    viewer._run_thread.join()

    # the names were visible on the trait as the phases advanced...
    assert seen == ["compiling", "stepping"]
    # ...and it is cleared once the run finishes
    assert viewer.phase == ""


# --------------------------------------------------------- D17: stopping a run


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, *, on_step: Any = None) -> dict[str, Any]:
    """Stubs the placement pipeline, leaving ``run_fdtd``'s kwargs observable.

    Returns a dict that ``run_fdtd``'s replacement fills with the kwargs it was
    called with, so a test can assert on the stopping condition.
    """
    import fdtdx

    captured: dict[str, Any] = {}

    def fake_run_fdtd(*, arrays: Any, objects: Any, config: Any, key: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        if on_step is not None:
            on_step()
        return None, arrays

    monkeypatch.setattr(fdtdx, "run_fdtd", fake_run_fdtd)
    monkeypatch.setattr(sim, "build_objects", lambda _stack, _state: ([], _TimeConfig(), []))
    monkeypatch.setattr(
        fdtdx, "place_objects", lambda *_a, **_k: (None, None, {}, _TimeConfig(), None)
    )

    from gdsfactory import fdtdx_stack

    monkeypatch.setattr(
        fdtdx_stack, "apply_stack", lambda _arrays, _objects, _params, _key: (None, None, None)
    )
    monkeypatch.setattr(fdtdx_stack, "init_stack_params", lambda _objects: {})
    return captured


def test_cancel_condition_stops_at_the_flag() -> None:
    """D17: the condition keeps going until the flag is set, then stops."""
    import jax.numpy as jnp

    flag = threading.Event()
    condition = sim.make_cancel_condition(flag).setup((jnp.asarray(0), None), _TimeConfig(), None)

    def running(step: int) -> bool:
        return bool(condition((jnp.asarray(step, dtype=jnp.int32), None), _TimeConfig(), None))

    assert running(0) is True
    flag.set()
    assert running(1) is False


def test_cancel_condition_reproduces_the_time_limit() -> None:
    """D17: a custom condition replaces the default, so it must end the run itself."""
    import jax.numpy as jnp

    flag = threading.Event()
    condition = sim.make_cancel_condition(flag).setup((jnp.asarray(0), None), _TimeConfig(), None)

    def running(step: int) -> bool:
        return bool(condition((jnp.asarray(step, dtype=jnp.int32), None), _TimeConfig(), None))

    assert running(_TimeConfig.time_steps_total - 1) is True
    assert running(_TimeConfig.time_steps_total) is False


def test_run_passes_no_condition_when_not_cancellable(monkeypatch: pytest.MonkeyPatch) -> None:
    """D17: an ordinary run is unchanged — no condition, so no per-step host read."""
    captured = _stub_pipeline(monkeypatch)
    sim.run(object(), object())
    assert captured["stopping_condition"] is None


def test_run_passes_a_condition_when_cancellable(monkeypatch: pytest.MonkeyPatch) -> None:
    """D17: a cancellable run hands fdtdx the flag-reading condition."""
    captured = _stub_pipeline(monkeypatch)
    flag = threading.Event()
    sim.run(object(), object(), cancel=flag)
    condition = captured["stopping_condition"]
    assert condition is not None
    assert condition.cancel_flag is flag


def test_a_stop_during_placing_raises_at_the_next_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D17: a stop lands between phases, not only between time steps."""
    flag = threading.Event()

    def stop_during_build(_stack: Any, _state: Any) -> Any:
        flag.set()
        return [], object(), []

    monkeypatch.setattr(sim, "build_objects", stop_during_build)

    with pytest.raises(sim.RunStopped):
        sim.run(object(), object(), cancel=flag)


def test_a_discarded_stop_raises_and_a_kept_one_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """D17: ``keep_partial`` decides between raising and returning the arrays."""
    flag = threading.Event()
    captured = _stub_pipeline(monkeypatch, on_step=flag.set)

    with pytest.raises(sim.RunStopped):
        sim.run(object(), object(), cancel=flag)

    flag.clear()
    result = sim.run(object(), object(), cancel=flag, keep_partial=True)
    assert isinstance(result, sim.RunResult)


class _TimeConfig:
    """The two ``config`` fields the cancel condition reads (spec D17)."""

    time_steps_total = 10


# ------------------------------------------------------------------- helpers


def _spy_create(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every ``create_*_at`` call, via the one place that makes them.

    :func:`sim._create` is the single caller of ``stack.create_*_at`` (spec
    D11), so spying on it observes every creation without touching the device
    (``StackDevice`` is an immutable pytree).
    """
    calls: list[str] = []
    real = sim._create

    def wrapper(stack: Any, spec: Any, settings: Any, **kwargs: Any) -> Any:
        calls.append(spec.type)
        return real(stack, spec, settings, **kwargs)

    monkeypatch.setattr(sim, "_create", wrapper)
    return calls
