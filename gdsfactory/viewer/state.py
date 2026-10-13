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
from gdsfactory.viewer.solids import (
    LevelSolid,
    PortGlyph,
    SolidMesh,
    SolidSet,
    analytic_solids,
    background_mesh,
    level_solids,
)

__all__ = ["LayerStyle", "ViewerState"]

#: Default opacity of a fill level and of every other level (spec R2, Q5).
FILL_OPACITY = 0.15
LEVEL_OPACITY = 0.9

#: The section planes the 2D viewport can show (spec R4).
PLANES = ("xy", "xz", "yz")


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

    # ------------------------------------------------------------------ patching

    def apply_patch(self, patch: dict[str, Any]) -> None:
        """Applies one browser patch in place (spec D10).

        Recognised keys: ``styles`` (per-level partial ``{color, opacity,
        visible}``), ``port_visible``, ``selection``, ``active_viewport``,
        ``show_full``, ``plane`` and ``plane_position``. A ``plane_position``
        patch is stored against the active plane, so each plane keeps its own
        slider position. Unknown names or keys raise ``ValueError``; the widget
        handler catches it and re-syncs the canonical state, so a bad patch
        cannot corrupt the state.

        Args:
            patch: the partial state the browser sent.

        Raises:
            ValueError: on an unknown level, port, key or value.
        """
        unknown = set(patch) - {
            "styles",
            "port_visible",
            "selection",
            "active_viewport",
            "show_full",
            "plane",
            "plane_position",
        }
        if unknown:
            raise ValueError(f"Unknown state keys: {sorted(unknown)}")

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
        }


def _opacity(value: Any) -> float:
    """A clamped opacity in [0, 1] (spec R2)."""
    return min(1.0, max(0.0, float(value)))


def _selection(selection: Any, state: ViewerState) -> dict[str, str] | None:
    """A validated selection ``{"kind": "level"|"port", "name": ...}`` or None."""
    if selection is None:
        return None
    if not isinstance(selection, dict) or set(selection) != {"kind", "name"}:
        raise ValueError(f"Bad selection {selection!r}")
    kind, name = selection["kind"], selection["name"]
    if kind == "level" and name in state.styles:
        return {"kind": "level", "name": name}
    if kind == "port" and name in state.port_visible:
        return {"kind": "port", "name": name}
    raise ValueError(f"Unknown selection {selection!r}")
