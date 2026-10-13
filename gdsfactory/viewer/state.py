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
from typing import Any, NamedTuple

from gdsfactory.viewer.palette import fallback_color
from gdsfactory.viewer.sim import (
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
)

__all__ = ["LayerStyle", "ViewerState", "spec_glyphs"]

#: Default opacity of a fill level and of every other level (spec R2, Q5).
FILL_OPACITY = 0.15
LEVEL_OPACITY = 0.9

#: The section planes the 2D viewport can show (spec R4).
PLANES = ("xy", "xz", "yz")

#: Every parameter name a source/detector spec may carry (spec R12): the common
#: fields plus each type's extras. ``type`` and ``port`` are separate.
_SPEC_PARAM_KEYS = frozenset(
    {"port_offset", "level", "direction", "name", "wavelengths", "port_size_mult", "visible"}
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

    def __init__(self, stack: Any) -> None:
        """Builds the device's geometry once and sets the default styles.

        Args:
            stack: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
        """
        self.stack = stack
        solids = analytic_solids(stack)
        #: the manifolds behind the meshes; the 2D section slices these, so the
        #: geometry is built once and shared (spec D3 risk, R4)
        self.solids: SolidSet = solids
        self.origin: tuple[float, float, float] = solids.origin
        self.size: tuple[float, float, float] = solids.size
        self.levels: list[LevelSolid] = level_solids(stack, solids)
        self.background: SolidMesh = background_mesh(solids)
        self.ports: tuple[PortGlyph, ...] = solids.ports

        self.styles: dict[str, LayerStyle] = {
            level.name: LayerStyle(
                color=_default_color(_gds_layer(stack.levels[index].expr), index),
                opacity=FILL_OPACITY if level.fill else LEVEL_OPACITY,
                visible=True,
            )
            for index, level in enumerate(self.levels)
        }
        self.port_visible: dict[str, bool] = {port.name: True for port in self.ports}
        self.selection: dict[str, str] | None = None
        self.active_viewport: str = "3d"
        self.show_full: bool = False
        self.plane: str = "xy"
        #: the sources and detectors as declarative specs (spec R12, D11); the
        #: widget creates nothing until retrieval, so this is the whole setup
        self.specs: list[ObjectSpec] = []
        #: the setup-wide simulation settings (spec R13)
        self.settings = SimSettings()
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

    # ------------------------------------------------------------------ section

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
            ValueError: if the spec names a port this device does not have.
        """
        if spec.port not in self.port_visible:
            raise ValueError(f"Unknown port {spec.port!r}")
        if not spec.id:
            spec.id = self._new_id(spec.category)
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
        """
        spec = self.find_spec(str(data.get("id", "")))
        if "type" in data:
            type_name = str(data["type"])
            if type_name not in TYPES:
                raise ValueError(f"Unknown source/detector type {type_name!r}")
            spec.type = type_name
        if "port" in data:
            port = str(data["port"])
            if port not in self.port_visible:
                raise ValueError(f"Unknown port {port!r}")
            spec.port = port
        if "params" in data:
            unknown = set(data["params"]) - set(_SPEC_PARAM_KEYS)
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")
            spec.params.update(data["params"])
        return spec

    def remove_spec(self, spec_id: str) -> None:
        """Removes a source/detector spec (spec R12)."""
        spec = self.find_spec(spec_id)
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
            "port_visible",
            "selection",
            "active_viewport",
            "show_full",
            "plane",
            "plane_position",
            "sources",
            "detectors",
            "simulation",
        }
        if unknown:
            raise ValueError(f"Unknown state keys: {sorted(unknown)}")

        if self.running and ({"sources", "detectors", "simulation"} & set(patch)):
            raise ValueError("The setup is locked while a run is in flight")

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

        if "sources" in patch:
            _apply_spec_patch(self, "source", patch["sources"])
        if "detectors" in patch:
            _apply_spec_patch(self, "detector", patch["detectors"])
        if "simulation" in patch:
            self.settings = SimSettings.from_wire({**self.settings.to_wire(), **patch["simulation"]})
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
            "port_visible": dict(self.port_visible),
            "selection": self.selection,
            "active_viewport": self.active_viewport,
            "show_full": self.show_full,
            "plane": self.plane,
            "plane_position": self.plane_position,
            "plane_bounds": list(self.plane_bounds()),
            "sources": [s.to_wire() for s in self.specs if s.category == "source"],
            "detectors": [s.to_wire() for s in self.specs if s.category == "detector"],
            "glyphs": spec_glyphs(self),
            "simulation": self.settings.to_wire(),
            "running": self.running,
        }


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
