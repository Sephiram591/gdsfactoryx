"""The viewer's ``anywidget`` app shell (spec D2, D7, D9, D10, D15).

One widget owns the menubar, the explorer and both viewports (spec D9). It
holds the shared :class:`~gdsfactory.viewer.state.ViewerState` in the kernel and
syncs traits to the browser:

- ``geometry`` and ``meta`` — set once, from
  :func:`gdsfactory.viewer.payload.build_payload`; they carry the mesh buffers.
- ``state`` — the small JSON state, re-sent on every edit.
- ``section`` — the 2D cross-section rings (spec R4), rebuilt only when the
  plane or its position changes, so a color edit never reslices.

The browser edits the state by sending ``{"event": "set_state", "patch": ...}``;
:meth:`Viewer._on_msg` applies it to the kernel's state and syncs the canonical
result back (spec D10). A rejected patch re-syncs the state unchanged, so the
browser cannot diverge.

The save button normally downloads a PNG in the browser (spec D14). A per-viewer
"save to" preference can route it to the **kernel** instead: the browser sends
``{"event": "save_image", "data": <data URL>, "filename": ...}`` and the kernel
writes the file under :attr:`Viewer.save_dir` (the process working directory by
default), reporting the written path back in the ``save_result`` trait.

The three run modes (spec R14) hang off the handle:

- ``.objects`` / ``.config`` / ``.constraints`` — the mode-2 export, computed on
  access (spec D11), so they are properties, not stored attributes.
- ``.run()`` — mode 1: places and runs on a **background thread** and returns
  immediately (spec D15), so the widget stays interactive. The setup is locked
  while the run is in flight: ``state["running"]`` is set, and the kernel
  rejects setup edits in :meth:`_on_msg`. fdtdx's ``progress_callback`` is
  mirrored into the ``progress`` trait, which the browser draws as a bar.
- ``.launch_code()`` — mode 3, the script tail (spec D12). M5.
"""

from __future__ import annotations

import base64
import binascii
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anywidget
import traitlets

from gdsfactory.viewer.fields import validate_result
from gdsfactory.viewer.payload import build_field_payload, build_payload, build_section_payload
from gdsfactory.viewer.sim import RunStopped, build_objects
from gdsfactory.viewer.sim import run as _run_sim
from gdsfactory.viewer.state import ViewerState

__all__ = ["Viewer", "view"]

_STATIC = Path(__file__).parent / "static"

#: the messages the browser can send besides a state patch
_SETUP_EVENTS = frozenset({"add_source", "add_detector", "update_spec", "remove_spec", "set_settings"})


def _marimo_runtime_context() -> Any:
    """The running marimo runtime context, or ``None`` outside marimo.

    Under marimo, trait sync reaches the browser through a comm message that
    resolves the runtime context from a **thread-local**. Only the cell's own
    thread has it, so a trait set from a worker thread is silently dropped
    unless that thread installs the context first (see :meth:`Viewer.run`).
    The import is local so the widget still works under plain anywidget hosts
    (e.g. Jupyter), where no marimo context exists.
    """
    try:
        from marimo._runtime.context import safe_get_context
    except ImportError:
        return None
    return safe_get_context()


class Viewer(anywidget.AnyWidget):
    """The app-shell widget returned by :func:`view` (spec D7, D9)."""

    _esm = _STATIC / "viewer.js"
    _css = _STATIC / "viewer.css"

    #: set once: the mesh buffers and the box (spec D5, C5)
    geometry = traitlets.Dict(default_value={}).tag(sync=True)
    #: set once: the level and port metadata (spec D5)
    meta = traitlets.Dict(default_value={}).tag(sync=True)
    #: the shared state, re-sent on every edit (spec D10)
    state = traitlets.Dict(default_value={}).tag(sync=True)
    #: the 2D cross-section rings, rebuilt when the plane changes (spec R4)
    section = traitlets.Dict(default_value={}).tag(sync=True)
    #: the last run's outcome summary, for the browser to show (spec R14, Q20)
    run_result = traitlets.Dict(default_value={}).tag(sync=True)
    #: every detector's recorded field from the current result, keyed by spec id
    #: (spec R15, D16): the raw complex field as float32 buffers, so a transform
    #: or color edit re-shades in the browser with no kernel round-trip
    fields = traitlets.Dict(default_value={}).tag(sync=True)
    #: the in-flight run's progress as ``{"step": int, "total": int}`` (spec D15);
    #: ``{}`` when no run is in flight or the total is not yet known
    progress = traitlets.Dict(default_value={}).tag(sync=True)
    #: the in-flight run's phase — ``"placing"``, ``"compiling"`` or
    #: ``"stepping"`` (spec D15) — or ``""`` when no run is in flight
    phase = traitlets.Unicode(default_value="").tag(sync=True)
    #: whether the user has asked the in-flight run to stop (spec D17); the
    #: browser greys the button and shows "stopping…" while it is true
    stopping = traitlets.Bool(default_value=False).tag(sync=True)
    #: the last kernel-side save's outcome, for the browser to confirm (spec
    #: D14): ``{"path": str}`` on success, ``{"folder": str}`` when the save
    #: folder changed, or ``{"error": str}`` on failure
    save_result = traitlets.Dict(default_value={}).tag(sync=True)
    #: the absolute folder a server-side save writes into (spec D14). Settable
    #: from the browser, so the settings screen can offer a folder field.
    save_dir = traitlets.Unicode(default_value="").tag(sync=True)

    def __init__(self, stack: Any, save_dir: Path | str | None = None, **kwargs: Any) -> None:
        """Builds the device's geometry and prepares the widget.

        Args:
            stack: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
            save_dir: where the save button writes a PNG when the browser
                prefers a server-side save (spec D14). Defaults to the process
                working directory, resolved once at construction so a later
                ``os.chdir`` does not move the target.
            **kwargs: passed to :class:`anywidget.AnyWidget`.
        """
        self._viewer_state = ViewerState(stack)
        initial_save_dir = Path(save_dir).expanduser() if save_dir is not None else Path.cwd()
        self._run_result: Any = None
        self._run_error: BaseException | None = None
        self._run_thread: threading.Thread | None = None
        #: the stop flag for the in-flight run (spec D17); ``None`` when no run
        #: is in flight. Set by :meth:`stop` and read by the worker's condition.
        self._run_cancel: threading.Event | None = None
        #: the result in place before the current run started, so a *discarded*
        #: stop can leave ``last_result`` as it was rather than as ``None`` (D17)
        self._previous_result: Any = None
        #: cache of the prepared ``(objects, config, constraints)``, keyed by the
        #: state revision, so the three properties share one build (spec D11)
        self._prepared: tuple[int, tuple[list[Any], Any, list[Any]]] | None = None
        super().__init__(**kwargs)
        geometry, meta = build_payload(self._viewer_state)
        self.geometry = geometry
        self.meta = meta
        self.state = self._viewer_state.to_state()
        self.section = build_section_payload(self._viewer_state)
        self.save_dir = str(initial_save_dir.resolve())
        self.on_msg(self._on_msg)

    # ------------------------------------------------------------------ handle

    @property
    def stack(self) -> Any:
        """The :class:`~gdsfactory.fdtdx_stack.StackDevice` being viewed (spec D7)."""
        return self._viewer_state.stack

    @property
    def specs(self) -> list[Any]:
        """The source/detector specs (spec R12, D11)."""
        return self._viewer_state.specs

    @property
    def settings(self) -> Any:
        """The simulation settings (spec R13)."""
        return self._viewer_state.settings

    @property
    def running(self) -> bool:
        """Whether a mode-1 run is in flight (spec D15)."""
        return self._viewer_state.running

    def _prepare(self) -> tuple[list[Any], Any, list[Any]]:
        """The prepared objects, config and constraints (spec R14 mode 2, D11).

        Memoized on the state revision: the three properties (``objects``,
        ``config``, ``constraints``) share one build, so the constraints'
        object names always match the objects they were built with. A setup
        edit bumps the revision and forces a rebuild on next access.
        """
        revision = self._viewer_state.revision
        if self._prepared is None or self._prepared[0] != revision:
            self._prepared = (revision, build_objects(self.stack, self._viewer_state))
        return self._prepared[1]

    @property
    def objects(self) -> list[Any]:
        """The prepared simulation objects, computed on access (spec D11)."""
        return self._prepare()[0]

    @property
    def config(self) -> Any:
        """The prepared ``fdtdx.SimulationConfig``, computed on access (spec D11)."""
        return self._prepare()[1]

    @property
    def constraints(self) -> list[Any]:
        """The prepared placement constraints, computed on access (spec D11)."""
        return self._prepare()[2]

    def run(self, on_done: Callable[[Any], None] | None = None, key: Any = None) -> None:
        """Places and runs the simulation on a background thread (spec D15).

        Returns immediately; the widget stays interactive. While the run is in
        flight the setup is locked (``state['running']`` is true and setup edits
        are rejected). On completion ``run_result`` carries a summary and
        ``state['running']`` returns to false; detector results are on
        ``self.last_result.arrays.detector_states``.

        Args:
            on_done: called with the :class:`~gdsfactory.viewer.sim.RunResult`
                (or None on error) when the run finishes, on the worker thread.
            key: an optional ``jax`` PRNG key.

        Raises:
            RuntimeError: if a run is already in flight.
        """
        if self._viewer_state.running:
            raise RuntimeError("A run is already in flight")
        # snapshot the previous result before clearing, so a discarded stop can
        # restore it (spec D17)
        self._previous_result = self._run_result
        self._run_result = None
        self._run_error = None
        self._viewer_state.running = True
        self.state = self._viewer_state.to_state()
        self.run_result = {"status": "running"}
        self.progress = {}
        self.phase = "placing"
        self.stopping = False
        self._run_cancel = threading.Event()

        # capture the marimo runtime context here, on the cell's thread; the
        # worker must install it or its trait updates never reach the browser
        runtime_context = _marimo_runtime_context()

        def _on_progress(step: int, total: int) -> None:
            self.progress = {"step": int(step), "total": int(total)}

        def _on_phase(name: str) -> None:
            # once a stop is requested the phase must stay "stopping", so a later
            # "compiling" from the same run cannot overwrite it (spec D17)
            if self._run_cancel is not None and self._run_cancel.is_set():
                return
            self.phase = name

        def _work() -> None:
            if runtime_context is not None:
                with runtime_context.install():
                    _work_inner()
            else:
                _work_inner()

        def _work_inner() -> None:
            try:
                result = _run_sim(
                    self.stack,
                    self._viewer_state,
                    key=key,
                    progress=_on_progress,
                    phase=_on_phase,
                    cancel=self._run_cancel,
                    keep_partial=self._viewer_state.settings.keep_partial,
                )
                self._run_result = result
                self.run_result = {"status": "done", "detectors": _detector_names(result)}
                # the field buffers ride the same worker-thread context install
                self.fields = build_field_payload(self._viewer_state, result)
            except RunStopped:
                # a stopped run whose partial result is discarded: keep the
                # previous result and its overlays, report the stop (spec D17)
                self._run_result = self._previous_result
                self._run_error = None
                self.run_result = {"status": "stopped"}
            except BaseException as exc:
                self._run_error = exc
                self.run_result = {"status": "error", "message": str(exc)}
            finally:
                self._viewer_state.running = False
                self.state = self._viewer_state.to_state()
                self.progress = {}
                self.phase = ""
                self.stopping = False
                self._run_cancel = None
                if on_done is not None:
                    on_done(self._run_result)

        self._run_thread = threading.Thread(target=_work, name="viewer-run", daemon=True)
        self._run_thread.start()

    def stop(self) -> bool:
        """Asks the in-flight run to stop (spec D17).

        Sets the run's cancel flag; the worker notices it at the next step or
        phase boundary and unwinds. Idempotent, and a no-op when no run is in
        flight. Does not clear ``running`` — the worker's ``finally`` owns that,
        so the widget never claims a run has ended before it has.

        Returns:
            True if a run was in flight and the stop was requested, False
            otherwise.
        """
        cancel = self._run_cancel
        if cancel is None or not self._viewer_state.running:
            return False
        cancel.set()
        self.stopping = True
        return True

    @property
    def last_result(self) -> Any:
        """The last completed run's :class:`~gdsfactory.viewer.sim.RunResult` (spec Q20)."""
        return self._run_result

    @property
    def last_error(self) -> BaseException | None:
        """The last run's exception, if it failed (spec Q20)."""
        return self._run_error

    def show_result(self, result: Any) -> None:
        """Adopts a result the user produced elsewhere as the current result (R15).

        The common case is a user who ran the simulation under ``srun`` and
        wants to look at the fields in the widget: this adopts that result's
        detector arrays and redraws every overlay, with no re-run and no
        re-send of the geometry. It does **not** adopt the result's geometry or
        objects — the widget's own stack and specs stay authoritative.

        Args:
            result: a :class:`~gdsfactory.viewer.sim.RunResult`, or any object
                with ``.arrays.detector_states``, ``.objects`` and ``.config``.

        Raises:
            ValueError: if the result's detectors do not correspond to the
                widget's current setup by name, grid shape, grid spacing and
                plane position (spec Q23). The previous result and its overlays
                are left untouched.
        """
        validate_result(self._viewer_state, result)
        self._run_result = result
        self._run_error = None
        self.run_result = {"status": "done", "detectors": _detector_names(result)}
        self.fields = build_field_payload(self._viewer_state, result)

    # ------------------------------------------------------------------ messages

    def _on_msg(self, _widget: Any, content: dict[str, Any], _buffers: list) -> None:
        """Applies a browser message to the kernel state and re-syncs (spec D10).

        Handles a state ``patch`` (the general path) and the setup events
        (add/update/remove a source or detector, edit the settings). While a run
        is in flight the setup events and any setup-editing patch are refused
        (spec D15): the state is re-synced unchanged, so the browser cannot
        diverge from the kernel's authority.
        """
        event = content.get("event")
        if event == "run":
            # mode 1 from the widget's Run action (spec R14, D15); a second
            # request while a run is in flight is ignored, not an error
            try:
                self.run()
            except RuntimeError:
                pass
            return
        if event == "stop":
            # the Run button's stop toggle (spec D17): flag the in-flight run so
            # its worker unwinds at the next step or phase boundary
            self.stop()
            return
        if event == "save_image":
            # the browser's save button, routed to the kernel (spec D14): write
            # the PNG beside the notebook and report the path back
            self._save_image(content)
            return
        if event == "set_save_dir":
            # the settings screen's folder field (spec D14): point server-side
            # saves at a new directory, creating it if the write is legal
            self._set_save_dir(content.get("dir"))
            return
        before = (self._viewer_state.plane, self._viewer_state.plane_position)
        try:
            if event == "set_state":
                self._viewer_state.apply_patch(content.get("patch") or {})
            elif event in _SETUP_EVENTS:
                self._apply_setup_event(event, content)
            else:
                return
        except (ValueError, TypeError, KeyError):
            pass  # a bad or refused message leaves the state unchanged
        self.state = self._viewer_state.to_state()
        if self._viewer_state.geometry_changed:
            # a boundary_offset edit re-boxed the device (spec R13): re-send the
            # mesh buffers and metadata so the browser rebuilds its viewports
            self._viewer_state.geometry_changed = False
            geometry, meta = build_payload(self._viewer_state)
            self.geometry = geometry
            self.meta = meta
            self.section = build_section_payload(self._viewer_state)
            return
        after = (self._viewer_state.plane, self._viewer_state.plane_position)
        if after != before:
            self.section = build_section_payload(self._viewer_state)

    def _apply_setup_event(self, event: str, content: dict[str, Any]) -> None:
        """Turns a setup event into a state patch (spec R12, R13, D15).

        Refused while a run is in flight, so the message can only touch the
        setup when the lock is clear.
        """
        if self._viewer_state.running:
            raise ValueError("The setup is locked while a run is in flight")
        if event == "add_source":
            self._viewer_state.apply_patch({"sources": [{"op": "add", "port": content["port"],
                                                         **({"spec": content["spec"]} if "spec" in content else {})}]})
        elif event == "add_detector":
            self._viewer_state.apply_patch({"detectors": [{"op": "add", "port": content["port"],
                                                           **({"spec": content["spec"]} if "spec" in content else {})}]})
        elif event == "update_spec":
            key = "sources" if content["kind"] == "source" else "detectors"
            self._viewer_state.apply_patch({key: [{"op": "update", **content["spec"]}]})
        elif event == "remove_spec":
            key = "sources" if content["kind"] == "source" else "detectors"
            self._viewer_state.apply_patch({key: [{"op": "remove", "id": content["id"]}]})
        elif event == "set_settings":
            self._viewer_state.apply_patch({"simulation": content["settings"]})

    def _set_save_dir(self, folder: Any) -> None:
        """Points server-side saves at ``folder``, creating it (spec D14).

        The browser cannot browse the server's filesystem, so the folder comes
        in as a path string. A relative path resolves against the current
        :attr:`save_dir`, so a short name like ``shots`` nests under the
        existing target. The directory is created here — so a typo or an
        unwritable path is reported at once, not on the next save.
        """
        if not isinstance(folder, str) or not folder.strip():
            self.save_result = {"error": "enter a folder path"}
            return
        try:
            target = Path(folder).expanduser()
            if not target.is_absolute():
                target = Path(self.save_dir) / target
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.save_result = {"error": str(exc)}
        else:
            self.save_dir = str(target.resolve())
            self.save_result = {"folder": self.save_dir}

    def _save_image(self, content: dict[str, Any]) -> None:
        """Writes a browser PNG under :attr:`save_dir` (spec D14).

        The browser sends a ``data:image/png;base64,...`` URL and a filename;
        the file lands in ``save_dir`` (created if missing) and ``save_result``
        carries the absolute path, or an ``error`` string if the write failed.
        An existing file is kept: a ``-1``, ``-2``, … suffix is appended until
        the name is free, so repeated saves never clobber an earlier one.
        """
        data = content.get("data")
        filename = content.get("filename") or "viewer.png"
        try:
            raw = _decode_data_url(data)
            target = _free_path(Path(self.save_dir) / Path(filename).name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        except (ValueError, OSError, binascii.Error) as exc:
            self.save_result = {"error": str(exc)}
        else:
            self.save_result = {"path": str(target.resolve())}


def _decode_data_url(data: Any) -> bytes:
    """Decodes a base64 ``data:`` URL's payload into bytes (spec D14)."""
    if not isinstance(data, str) or not data.startswith("data:"):
        raise ValueError("expected a data URL")
    _, _, encoded = data.partition(",")
    return base64.b64decode(encoded, validate=True)


def _free_path(path: Path) -> Path:
    """Returns ``path``, or ``stem-1``, ``stem-2``, … if it already exists."""
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def _detector_names(result: Any) -> list[str]:
    """The names of the detectors in a run's result (spec R14, Q20)."""
    try:
        return sorted(result.arrays.detector_states)
    except (AttributeError, TypeError):
        return []


def view(stack: Any, save_dir: Path | str | None = None) -> Viewer:
    """Returns the viewer for a device (spec D7).

    Args:
        stack: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
        save_dir: where a server-side save writes its PNG (spec D14). Defaults
            to the process working directory.

    Returns:
        The :class:`Viewer` app-shell widget.
    """
    return Viewer(stack, save_dir=save_dir)
