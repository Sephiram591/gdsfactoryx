"""The viewer's shared state (spec R1, R2, D9, D10).

One :class:`ViewerState` owns everything the explorer and the viewports read:
the geometry built once from a :class:`~gdsfactory.fdtdx_stack.StackDevice` (via
:func:`gdsfactory.viewer.solids.build`), the per-layer color/opacity/visibility,
the per-port visibility, the selection and the active viewport.

The state is coordinated in Python (spec D10): the browser sends a patch, the
kernel applies it here and syncs the canonical state back. :meth:`apply_patch`
is a pure in-place mutation, so it is testable without a kernel or a browser.

This module does not import ``anywidget``: :class:`ViewerState` is plain Python,
and only :mod:`gdsfactory.viewer.widget` wraps it in a widget.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, NamedTuple

from gdsfactory.viewer.palette import fallback_color
from gdsfactory.viewer.sim import (
    _ABSOLUTE,
    _FIELD_PARAMS,
    FIELD_TRANSFORMS,
    TYPES,
    ObjectSpec,
    SimSettings,
    default_detector_spec,
    default_source_spec,
    spec_glyph,
)
from gdsfactory.viewer.solids import (
    LevelSolid,
    PortGlyph,
    SolidMesh,
    SolidSet,
    analytic_solids,
    background_mesh,
    level_solids,
    plane_ports,
)

__all__ = ["LayerStyle", "ViewerState", "rebox_stack", "spec_glyphs"]

#: Default opacity of a fill level and of every other level (spec R2, Q5).
FILL_OPACITY = 0.15
LEVEL_OPACITY = 0.9

#: Default plane fade as a fraction of the box size along the section's normal
#: (spec R4). The 1/e distance is this fraction times that size.
PLANE_FADE_FRAC = 0.05

#: The section planes the 2D viewport can show (spec R4).
PLANES = ("xy", "xz", "yz")

#: Every parameter name a source/detector spec may carry (spec R12): the common
#: fields, the absolute-placement keys (spec R18), plus each type's extras.
#: ``type`` and ``port`` are separate. The detector display settings (spec R15)
#: are included, so a detector spec may carry them; a source never uses them.
_SPEC_PARAM_KEYS = frozenset(
    {"port_offset", "level", "direction", "name", "wavelengths", "port_size_mult", "visible"}
    | set(_ABSOLUTE)
    | set(_FIELD_PARAMS)
    | {key for kind in TYPES.values() for key in kind.extra}
)


class LayerStyle(NamedTuple):
    """A level's color, opacity and visibility (spec R2)."""

    color: str  # "#rrggbb"
    opacity: float
    visible: bool


def _gds_layer(expr: tuple) -> int | None:
    """The expression's representative GDS layer index, or None for a fill level.

    A level's expression reads one or more layers; the lowest index is taken as
    its GDS layer, so a derived layer (e.g. ``WG - ETCH``) reports ``WG``. A fill
    level reads no polygon layer, so it has none.
    """
    from gdsfactory.fdtdx_stack import _expr_layers

    layers = _expr_layers(expr)
    return min(layers) if layers else None


def _default_color(layer: int | None, index: int) -> str:
    """The PDK fill color of ``layer``, or a fallback palette color (spec R2).

    ``layer`` is the kfactory layer index from ``_expr_layers``;
    ``get_layer_tuple`` maps it to the ``(layer, datatype)`` the active PDK's
    ``layer_views`` are keyed by. A layer the PDK does not colour raises
    ``ValueError`` and falls back to the palette.
    """
    if layer is not None:
        from gdsfactory.pdk import get_layer_tuple, get_layer_views

        try:
            view = get_layer_views().get_from_tuple(get_layer_tuple(layer))
            return view.fill_color.as_hex()
        except ValueError:
            pass
    return fallback_color(index)


class ViewerState:
    """Everything the viewer shows and how it is styled (spec R1, R2, D9, D10)."""

    def __init__(self, stack: Any, settings: SimSettings | None = None) -> None:
        """Builds the device's geometry once and sets the default styles.

        Args:
            stack: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
            settings: kernel-side simulation settings for the new state;
                defaults to fresh defaults (used by ``load_result``, R16).
                Their ``averaging`` is replaced by the device's own.
        """
        self.stack = stack
        #: the averaging the device was built with (spec R13): the settings
        #: start from it, and the launch code emits an override only when the
        #: viewer's choice differs from it
        self.built_averaging: str = str(getattr(stack, "averaging", SimSettings.averaging))
        #: the device as built and its box, kept so a ``boundary_offset`` edit
        #: can re-box the device without rebuilding the geometry from scratch
        #: (spec R13): the offset moves the faces off this box
        self._base_stack = stack
        self._base_box = (
            tuple(float(v) for v in stack.partial_real_shape),
            (float(stack.center[0]), float(stack.center[1])),
            float(stack.z_center),
        )
        #: set when a patch re-boxed the device, so the widget re-sends the
        #: ``geometry``/``meta`` traits and the browser rebuilds its meshes
        #: (spec D5, R13); the widget clears it after re-sending
        self.geometry_changed: bool = False
        solids = analytic_solids(stack)
        #: the manifolds behind the meshes; the 2D section slices these, so the
        #: geometry is built once and shared (spec D3 risk, R4)
        self.solids: SolidSet = solids
        self.origin: tuple[float, float, float] = solids.origin
        self.size: tuple[float, float, float] = solids.size
        self.levels: list[LevelSolid] = level_solids(stack, solids)
        self.background: SolidMesh = background_mesh(solids)
        self.ports: tuple[PortGlyph, ...] = solids.ports
        #: the simulation-volume plane ports (spec R12): extra port choices for
        #: sources/detectors, not ports of the device
        self.plane_ports: tuple[PortGlyph, ...] = plane_ports(stack, solids.origin)

        self.styles: dict[str, LayerStyle] = {
            level.name: LayerStyle(
                color=_default_color(_gds_layer(stack.levels[index].expr), index),
                opacity=FILL_OPACITY if level.fill else LEVEL_OPACITY,
                visible=True,
            )
            for index, level in enumerate(self.levels)
        }
        #: every port glyph's visibility, device ports and plane ports alike
        #: (spec R1, R12), so either can be toggled and either may carry a spec
        self.port_visible: dict[str, bool] = {
            port.name: True for port in (*self.ports, *self.plane_ports)
        }
        self.selection: dict[str, str] | None = None
        self.active_viewport: str = "3d"
        self.show_full: bool = False
        self.plane: str = "xy"
        #: the sources and detectors as declarative specs (spec R12, D11); the
        #: widget creates nothing until retrieval, so this is the whole setup
        self.specs: list[ObjectSpec] = []
        #: ids of specs adopted from a loaded foreign result (spec R16); they
        #: are read-only, skipped by ``build_objects`` and flagged in ``to_state``
        self.adopted: set[str] = set()
        #: the setup-wide simulation settings (spec R13); their averaging starts
        #: as the device's own, so the device and the settings agree
        self.settings = replace(
            settings if settings is not None else SimSettings(), averaging=self.built_averaging
        )
        #: true while a mode-1 run is in flight; the kernel rejects setup edits
        #: in this state (spec D15). Set by the widget's run worker.
        self.running: bool = False
        #: bumped on every setup change, so the widget can memoize the prepared
        #: objects/config/constraints and rebuild them only when they change
        #: (spec D11: ``.objects`` etc. are computed on access)
        self.revision: int = 0
        self._next_spec_id: int = 0
        #: each plane remembers where its slider was, so switching planes and
        #: back returns to the last position seen for that plane (spec R4)
        self.plane_positions: dict[str, float] = {
            plane: 0.5 * sum(self.plane_bounds(plane)) for plane in PLANES
        }
        #: the plane fade as a fraction of the box size along the section's
        #: normal; the 1/e distance is that fraction of the size (spec R4)
        self.plane_fade_frac: float = PLANE_FADE_FRAC
        #: which explorer tab the browser shows (spec R17): the kernel sends the
        #: active world's traits — the editor's, or the result's own geometry and
        #: state when the Result tab is active — so the render knows which set
        #: applies
        self.active_tab: str = "layout"

    # ------------------------------------------------------------------ section

    def rebox(self, boundary_offset: Any) -> None:
        """Re-boxes the device from a ``boundary_offset`` (spec R13).

        ``boundary_offset`` is ``(x+, x-, y+, y-, z+, z-)`` in µm, an offset
        from the box the device was **built** with. Each value moves its face
        along the axis, positive towards +x, +y or +z — the same convention
        :meth:`~gdsfactory.fdtdx_stack.StackDevice.from_layer_stack` uses, so
        ``(1, -1, 0, 0, 0, 0)`` grows the box by 1 µm on each x side. The
        geometry is not rebuilt — the same :class:`StackDevice` is
        reconstructed with the moved faces (the fields it bakes the box into
        are ``partial_real_shape``, ``center`` and ``z_center``), and the
        solids, levels, ports and plane ports are rebuilt from it, since
        growing the box shows more of the stack.

        The slider positions are clamped to the new bounds; a plane that now
        falls outside the box is pulled back in.

        Raises:
            ValueError: if the offset is not 6 finite numbers, or moves a face
                past its opposite (an empty box).
        """
        offset = tuple(float(v) for v in boundary_offset)
        if len(offset) != 6 or not all(math.isfinite(v) for v in offset):
            raise ValueError(f"boundary_offset must be 6 finite numbers, got {boundary_offset!r}")
        base_shape, base_center, base_z = self._base_box
        # each axis' +face takes the even offset and its -face the odd one, so
        # the box grows by ``offset[even] - offset[odd]`` and shifts by their
        # mean (the same faces ``from_layer_stack`` builds the box from). The
        # offset is in µm; ``partial_real_shape`` is in metres, the centers µm.
        shape = (
            base_shape[0] + (offset[0] - offset[1]) * 1e-6,
            base_shape[1] + (offset[2] - offset[3]) * 1e-6,
            base_shape[2] + (offset[4] - offset[5]) * 1e-6,
        )
        if min(shape) <= 0:
            raise ValueError(f"boundary_offset collapses the box to {shape} m")
        center = (
            base_center[0] + 0.5 * (offset[0] + offset[1]),
            base_center[1] + 0.5 * (offset[2] + offset[3]),
        )
        z_center = base_z + 0.5 * (offset[4] + offset[5])
        # the base keeps the averaging it was built with; the settings' wins
        stack = _with_averaging(
            _rebox_stack(self._base_stack, shape, center, z_center), self.settings.averaging
        )
        solids = analytic_solids(stack)
        self.stack = stack
        self.solids = solids
        self.origin = solids.origin
        self.size = solids.size
        self.levels = level_solids(stack, solids)
        self.background = background_mesh(solids)
        self.ports = solids.ports
        self.plane_ports = plane_ports(stack, solids.origin)
        # a re-boxed device may drop or add a port (a level no longer reaches
        # the box), so keep the visibility map in step with the glyphs
        names = {port.name for port in (*self.ports, *self.plane_ports)}
        self.port_visible = {name: self.port_visible.get(name, True) for name in names}
        # and the style map in step with the levels (a level outside the old
        # box appears; one no longer reaching it is dropped)
        level_names = {level.name for level in self.levels}
        self.styles = {name: s for name, s in self.styles.items() if name in level_names}
        for level in self.levels:
            if level.name not in self.styles:
                self.styles[level.name] = LayerStyle(
                    color="#cccccc", opacity=FILL_OPACITY if level.fill else LEVEL_OPACITY,
                    visible=True,
                )
        for plane in PLANES:
            lo, hi = self.plane_bounds(plane)
            self.plane_positions[plane] = min(hi, max(lo, self.plane_positions[plane]))
        self.geometry_changed = True

    def plane_bounds(self, plane: str | None = None) -> tuple[float, float]:
        """The section slider's ``(min, max)`` along ``plane``'s normal (spec R4).

        The range is the device box along that axis, from ``origin``/``size``
        (both measured from the box center, spec C5). ``plane`` defaults to the
        active plane.
        """
        axis = {"xy": 2, "xz": 1, "yz": 0}[plane or self.plane]
        half = 0.5 * self.size[axis]
        center = self.origin[axis]
        return center - half, center + half

    @property
    def plane_position(self) -> float:
        """The active plane's remembered slider position (spec R4)."""
        return self.plane_positions[self.plane]

    # ------------------------------------------------------------------ specs

    def _new_id(self, kind: str) -> str:
        """A fresh stable id for a new spec (spec R12)."""
        self._next_spec_id += 1
        return f"{kind}-{self._next_spec_id}"

    def add_spec(self, spec: ObjectSpec) -> ObjectSpec:
        """Adds a source/detector spec, assigning it an id (spec R12, D11).

        Raises:
            ValueError: if the spec names a port this device does not have
                ("None", absolute placement, is always accepted — spec R18).
        """
        if spec.port != "None" and spec.port not in self.port_visible:
            raise ValueError(f"Unknown port {spec.port!r}")
        if not spec.id:
            spec.id = self._new_id(spec.category)
        self.specs.append(spec)
        return spec

    def adopt_spec(self, spec: ObjectSpec) -> ObjectSpec:
        """Adds a read-only spec describing an adopted result object (R16).

        Unlike :meth:`add_spec` there is no port check — an adopted spec is
        always placed absolutely — and the spec's id is recorded in
        :attr:`adopted`, so setup edits to it are refused and
        ``build_objects`` skips it.

        Args:
            spec: the spec to adopt; its ``glyph`` may already be set.

        Returns:
            The spec, with an id assigned.
        """
        if not spec.id:
            spec.id = self._new_id(spec.category)
        self.adopted.add(spec.id)
        self.specs.append(spec)
        return spec

    def find_spec(self, spec_id: str) -> ObjectSpec:
        """The spec with ``spec_id`` (spec R12).

        Raises:
            ValueError: if no spec has that id.
        """
        for spec in self.specs:
            if spec.id == spec_id:
                return spec
        raise ValueError(f"Unknown source/detector id {spec_id!r}")

    def update_spec(self, data: dict[str, Any]) -> ObjectSpec:
        """Replaces one spec's type, port and/or parameters (spec R12).

        ``data`` is ``{"id", "type"?, "port"?, "params"?}``; omitted fields are
        kept. Parameters are merged, so a one-field edit does not drop the rest.

        An adopted spec (spec R16) accepts only its display params — the
        detector's field settings and ``visible`` — since it describes a live
        placed object; anything else raises ``ValueError``.

        Raises:
            ValueError: on an unknown type, port or parameter, or a
                non-display edit to an adopted spec.
        """
        spec = self.find_spec(str(data.get("id", "")))
        if spec.id in self.adopted:
            if "type" in data or "port" in data:
                raise ValueError(
                    f"{spec.category} {spec.name!r} comes from the loaded result and is read-only"
                )
            if "params" in data:
                bad = set(data["params"]) - set(_FIELD_PARAMS) - {"visible"}
                if bad:
                    raise ValueError(
                        f"detector {spec.name!r} comes from the loaded result; only its "
                        f"display settings are editable (refused: {sorted(bad)})"
                    )
        if "type" in data:
            type_name = str(data["type"])
            if type_name not in TYPES:
                raise ValueError(f"Unknown source/detector type {type_name!r}")
            spec.type = type_name
        if "port" in data:
            port = str(data["port"])
            if port != "None" and port not in self.port_visible:
                raise ValueError(f"Unknown port {port!r}")
            spec.port = port
        if "params" in data:
            unknown = set(data["params"]) - set(_SPEC_PARAM_KEYS)
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")
            _validate_field_params(spec, data["params"])
            spec.params.update(data["params"])
        return spec

    def remove_spec(self, spec_id: str) -> None:
        """Removes a source/detector spec (spec R12).

        Raises:
            ValueError: if the spec was adopted from the loaded result (R16).
        """
        spec = self.find_spec(spec_id)
        if spec.id in self.adopted:
            raise ValueError(
                f"detector {spec.name!r} comes from the loaded result and cannot be removed"
            )
        self.specs.remove(spec)
        if self.selection and self.selection.get("name") == spec_id:
            self.selection = None

    # ------------------------------------------------------------------ patching

    def apply_patch(self, patch: dict[str, Any]) -> None:
        """Applies one browser patch in place (spec D10).

        Recognised keys: ``styles`` (per-level partial ``{color, opacity,
        visible}``), ``port_visible``, ``selection``, ``active_viewport``,
        ``show_full``, ``plane``, ``plane_position``, and — from M4 — ``sources``,
        ``detectors`` and ``simulation`` (spec R12, R13). A ``plane_position``
        patch is stored against the active plane, so each plane keeps its own
        slider position. Unknown names or keys raise ``ValueError``; the widget
        handler catches it and re-syncs the canonical state, so a bad patch
        cannot corrupt the state.

        While ``running`` is set, the setup-editing keys (``sources``,
        ``detectors``, ``simulation``) are refused with ``ValueError`` so the
        widget re-syncs the unchanged setup (spec D15). Read-only keys still
        apply.

        Args:
            patch: the partial state the browser sent.

        Raises:
            ValueError: on an unknown level, port, key or value, or a
                setup edit while a run is in flight.
        """
        unknown = set(patch) - {
            "styles",
            "active_tab",
            "port_visible",
            "selection",
            "active_viewport",
            "show_full",
            "plane",
            "plane_position",
            "plane_fade_frac",
            "sources",
            "detectors",
            "simulation",
        }
        if unknown:
            raise ValueError(f"Unknown state keys: {sorted(unknown)}")

        if self.running and ({"sources", "detectors", "simulation"} & set(patch)):
            raise ValueError("The setup is locked while a run is in flight")

        if "active_tab" in patch:
            tab = patch["active_tab"]
            if tab not in ("layout", "sources", "simulation", "result"):
                raise ValueError(f"Unknown tab {tab!r}")
            self.active_tab = tab

        for name, style in patch.get("styles", {}).items():
            if name not in self.styles:
                raise ValueError(f"Unknown level {name!r}")
            current = self.styles[name]
            bad = set(style) - {"color", "opacity", "visible"}
            if bad:
                raise ValueError(f"Unknown style keys for {name!r}: {sorted(bad)}")
            self.styles[name] = LayerStyle(
                color=str(style.get("color", current.color)),
                opacity=_opacity(style.get("opacity", current.opacity)),
                visible=bool(style.get("visible", current.visible)),
            )

        for name, visible in patch.get("port_visible", {}).items():
            if name not in self.port_visible:
                raise ValueError(f"Unknown port {name!r}")
            self.port_visible[name] = bool(visible)

        if "selection" in patch:
            self.selection = _selection(patch["selection"], self)

        if "active_viewport" in patch:
            if patch["active_viewport"] not in ("3d", "2d"):
                raise ValueError(f"Unknown viewport {patch['active_viewport']!r}")
            self.active_viewport = patch["active_viewport"]

        if "show_full" in patch:
            self.show_full = bool(patch["show_full"])

        if "plane" in patch:
            if patch["plane"] not in PLANES:
                raise ValueError(f"Unknown plane {patch['plane']!r}")
            self.plane = patch["plane"]

        if "plane_position" in patch:
            position = float(patch["plane_position"])
            if not math.isfinite(position):
                raise ValueError(f"Non-finite plane position {patch['plane_position']!r}")
            lo, hi = self.plane_bounds()
            self.plane_positions[self.plane] = min(hi, max(lo, position))

        if "plane_fade_frac" in patch:
            fade = float(patch["plane_fade_frac"])
            if not math.isfinite(fade) or fade <= 0:
                raise ValueError(f"Non-positive plane fade {patch['plane_fade_frac']!r}")
            self.plane_fade_frac = fade

        if "sources" in patch:
            _apply_spec_patch(self, "source", patch["sources"])
        if "detectors" in patch:
            _apply_spec_patch(self, "detector", patch["detectors"])
        if "simulation" in patch:
            previous = self.settings.boundary_offset
            settings = SimSettings.from_wire({**self.settings.to_wire(), **patch["simulation"]})
            # re-box before committing the settings: a rejected offset (an empty
            # box) must leave both the settings and the geometry untouched
            if settings.boundary_offset != previous:
                self.rebox(settings.boundary_offset)
            self.settings = settings
            # the averaging is the device's own field (spec R13); the geometry
            # does not depend on it, so nothing is rebuilt
            self.stack = _with_averaging(self.stack, settings.averaging)
        if {"sources", "detectors", "simulation"} & set(patch):
            self.revision += 1

    # ------------------------------------------------------------------ wire

    def to_state(self) -> dict[str, Any]:
        """The small JSON state sent to the browser on every edit (spec D5)."""
        return {
            "styles": {
                name: {"color": s.color, "opacity": s.opacity, "visible": s.visible}
                for name, s in self.styles.items()
            },
            "active_tab": self.active_tab,
            "port_visible": dict(self.port_visible),
            "selection": self.selection,
            "active_viewport": self.active_viewport,
            "show_full": self.show_full,
            "plane": self.plane,
            "plane_position": self.plane_position,
            "plane_bounds": list(self.plane_bounds()),
            "plane_fade_frac": self.plane_fade_frac,
            "sources": [
                {**s.to_wire(), "adopted": s.id in self.adopted}
                for s in self.specs
                if s.category == "source"
            ],
            "detectors": [
                {**s.to_wire(), "adopted": s.id in self.adopted}
                for s in self.specs
                if s.category == "detector"
            ],
            "field_settings": {
                s.id: s.field_params() for s in self.specs if s.category == "detector"
            },
            "glyphs": spec_glyphs(self),
            "simulation": self.settings.to_wire(),
            "running": self.running,
        }


def rebox_stack(device: Any, boundary_offset: Any) -> Any:
    """A copy of ``device`` with the box moved by ``boundary_offset`` (spec R13).

    The public form of the re-box :meth:`ViewerState.rebox` applies to its own
    device: ``boundary_offset`` is ``(x+, x-, y+, y-, z+, z-)`` in µm off the
    box the device was built with. Also used by the launch-code emitter (spec
    R14 mode 3), so a viewer-edited offset is reproduced by the emitted script.

    Raises:
        ValueError: if the offset is not 6 finite numbers, or moves a face
            past its opposite (an empty box).
    """
    offset = tuple(float(v) for v in boundary_offset)
    if len(offset) != 6 or not all(math.isfinite(v) for v in offset):
        raise ValueError(f"boundary_offset must be 6 finite numbers, got {boundary_offset!r}")
    base_shape = tuple(float(v) for v in device.partial_real_shape)
    base_center = (float(device.center[0]), float(device.center[1]))
    base_z = float(device.z_center)
    shape = (
        base_shape[0] + (offset[0] - offset[1]) * 1e-6,
        base_shape[1] + (offset[2] - offset[3]) * 1e-6,
        base_shape[2] + (offset[4] - offset[5]) * 1e-6,
    )
    if min(shape) <= 0:
        raise ValueError(f"boundary_offset collapses the box to {shape} m")
    center = (
        base_center[0] + 0.5 * (offset[0] + offset[1]),
        base_center[1] + 0.5 * (offset[2] + offset[3]),
    )
    z_center = base_z + 0.5 * (offset[4] + offset[5])
    return _rebox_stack(device, shape, center, z_center)


def _rebox_stack(device: Any, shape: tuple[float, float, float], center: tuple[float, float], z_center: float) -> Any:
    """A copy of ``device`` with its box fields replaced (spec R13).

    :class:`~gdsfactory.fdtdx_stack.StackDevice` is an immutable pytreeclass
    tree, so the box fields cannot be set in place. Its generated ``__init__``
    takes every field as a keyword, though, so the device is reconstructed with
    each field read back from the original and the three box fields overridden.
    The polygons, levels and materials are shared, not rebuilt — only the box
    the viewer draws and clips to changes.
    """
    import inspect

    names = [p for p in inspect.signature(type(device).__init__).parameters if p != "self"]
    kwargs = {name: getattr(device, name) for name in names}
    kwargs["partial_real_shape"] = tuple(shape)
    kwargs["center"] = tuple(center)
    kwargs["z_center"] = float(z_center)
    return type(device)(**kwargs)


def _with_averaging(device: Any, averaging: str) -> Any:
    """``device`` with its ``averaging`` set to ``averaging`` (spec R13).

    The averaging only decides how the device paints the grid when it is
    placed, so the polygons, levels and box are shared, not rebuilt. A device
    that already averages that way (or has no ``averaging``) is returned as is.
    """
    if getattr(device, "averaging", averaging) == averaging:
        return device
    return device.aset("averaging", averaging)


def _opacity(value: Any) -> float:
    """A clamped opacity in [0, 1] (spec R2)."""
    return min(1.0, max(0.0, float(value)))


def _selection(selection: Any, state: ViewerState) -> dict[str, str] | None:
    """A validated selection ``{"kind", "name"}`` or None (spec R1, R12).

    ``kind`` is ``"level"``, ``"port"`` or ``"source"``/``"detector"``; for a
    source/detector the name is the spec's id.
    """
    if selection is None:
        return None
    if not isinstance(selection, dict) or set(selection) != {"kind", "name"}:
        raise ValueError(f"Bad selection {selection!r}")
    kind, name = selection["kind"], selection["name"]
    if kind == "level" and name in state.styles:
        return {"kind": "level", "name": name}
    if kind == "port" and name in state.port_visible:
        return {"kind": "port", "name": name}
    if kind in ("source", "detector") and any(s.id == name for s in state.specs):
        return {"kind": kind, "name": name}
    raise ValueError(f"Unknown selection {selection!r}")


def _validate_field_params(spec: ObjectSpec, params: dict[str, Any]) -> None:
    """Refuses a bad detector display setting (spec R15, D16).

    A source carries none of these, so any of them on a source is refused. A
    detector's ``field_transform`` must be one of the known transforms and
    ``field_opacity`` must be a number in [0, 1]; the colors are accepted as
    strings (the browser validates the CSS).
    """
    given = set(params) & set(_FIELD_PARAMS)
    if not given:
        return
    if spec.category != "detector":
        raise ValueError(f"a {spec.category} has no field display settings: {sorted(given)}")
    if "field_transform" in params and params["field_transform"] not in FIELD_TRANSFORMS:
        raise ValueError(
            f"unknown field transform {params['field_transform']!r}; "
            f"expected one of {sorted(FIELD_TRANSFORMS)}"
        )
    if "field_opacity" in params:
        opacity = float(params["field_opacity"])
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError(f"field_opacity must be in [0, 1], got {params['field_opacity']!r}")


def _apply_spec_patch(state: ViewerState, category: str, ops: Any) -> None:
    """Applies a list of source/detector operations (spec R12).

    Each op is ``{"op": "add"|"update"|"remove", ...}``: ``add`` takes a spec
    (or just a ``port``, using the default type), ``update`` takes ``{"id", ...}``
    and ``remove`` takes ``{"id"}``. An op of the wrong category is refused.
    """
    if not isinstance(ops, list):
        raise ValueError(f"{category} patch must be a list, got {ops!r}")
    for op in ops:
        if not isinstance(op, dict):
            raise ValueError(f"Bad {category} op {op!r}")
        action = op.get("op")
        if action == "add":
            spec = ObjectSpec.from_wire(op["spec"]) if "spec" in op else (
                default_source_spec(str(op["port"]), state.settings)
                if category == "source"
                else default_detector_spec(str(op["port"]), state.settings)
            )
            if spec.category != category:
                raise ValueError(f"A {spec.category} cannot be added as a {category}")
            state.add_spec(spec)
        elif action == "update":
            spec = state.find_spec(str(op.get("id", "")))
            if spec.category != category:
                raise ValueError(f"{spec.id!r} is a {spec.category}, not a {category}")
            state.update_spec(op)
        elif action == "remove":
            spec = state.find_spec(str(op.get("id", "")))
            if spec.category != category:
                raise ValueError(f"{spec.id!r} is a {spec.category}, not a {category}")
            state.remove_spec(spec.id)
        else:
            raise ValueError(f"Unknown {category} op {action!r}")


def spec_glyphs(state: ViewerState) -> list[dict[str, Any]]:
    """Every source/detector's glyph for the payload (spec R12, D11)."""
    return [
        spec_glyph(state.stack, spec, state.origin, state.settings) for spec in state.specs
    ]
