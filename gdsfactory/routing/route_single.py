"""`route_single` places a Manhattan route between two ports.

`route_single` only works for an individual routes. For routing groups of ports you need to use `route_bundle` instead

To make a route, you need to supply:

 - input port
 - output port
 - bend
 - straight
 - taper to taper to wider straights and reduce straight loss (Optional)

To generate a route:

 1. Generate the backbone of the route.
 This is a list of manhattan coordinates that the route would pass through
 if it used only sharp bends (right angles)

 2. Replace the corners by bend references
 (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends

"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, cast

import kfactory as kf

import gdsfactory as gf
from gdsfactory._jax import to_float
from gdsfactory.component import Component
from gdsfactory.routing._kf_router import ManhattanRoute, route_bundle_kf
from gdsfactory.config import CONF
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    LayerTransitions,
    Port,
    Step,
    WayPoints,
)


def route_single(
    component: Component,
    port1: Port,
    port2: Port,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    waypoints: WayPoints | None = None,
    steps: Sequence[Step] | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    auto_taper: bool = True,
    on_collision: Literal["error", "show_error", "warning"] | None = None,
    on_placer_error: Literal["error", "show_error", "warning"] | None = None,
    on_error: Literal["error"] | None = None,
    layer_transitions: LayerTransitions | None = None,
) -> ManhattanRoute:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.

    Args:
        component: to place the route into.
        port1: start port.
        port2: end port.
        cross_section: spec.
        layer: layer spec.
        bend: bend spec.
        straight: straight spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        waypoints: optional list of points to pass through.
        steps: optional list of steps to pass through.
            Each step is a dict with keys: x (absolute), y (absolute), dx (relative), dy (relative).
            Use x/y to set an absolute coordinate and dx/dy to shift relative to the current position.
        port_type: port type to route.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route in um. If None, defaults to cross_section.width.
        auto_taper: add auto tapers.
        on_collision: action to take on route collision. "error" raises an exception.
            "show_error" shows the error in klayout's marker database.
            "warning" emits a warning and falls back to error markers.
            None silently falls back to error markers. Defaults to CONF.on_collision.
        on_placer_error: action to take on placer error. Same options as on_collision.
            Defaults to CONF.on_placer_error.
        on_error: deprecated, use on_placer_error instead. Maps to on_placer_error.
        layer_transitions: dictionary of layer transitions to use for the routing when auto_taper=True.

    Example:
        ```python
        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.move((40, 20))
        gf.routing.route_single(c, mmi1.ports["o2"], mmi2.ports["o1"], radius=5, cross_section="strip")
        c.plot()
        ```
    """
    if on_error is not None:
        warnings.warn(
            "on_error is deprecated, use on_placer_error instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        on_placer_error = on_placer_error or (on_error if on_error == "error" else None)

    on_collision = on_collision or CONF.on_collision
    on_placer_error = on_placer_error or CONF.on_placer_error

    if cross_section is None and (layer is None or route_width is None):
        raise ValueError(
            f"Either {cross_section=} or {layer=} and route_width must be provided"
        )

    c = component
    p1 = port1
    p2 = port2
    port_type = port_type or p1.port_type

    if cross_section is None:
        cross_section = gf.cross_section.cross_section(
            layer=cast("LayerSpec", layer),
            width=cast("float", route_width),
            port_names=("e1", "e2") if port_type == "electrical" else ("o1", "o2"),
            port_types=(port_type, port_type),
        )

    if route_width:
        xs = gf.get_cross_section(cross_section, width=route_width)
    else:
        xs = gf.get_cross_section(cross_section)
    width = route_width or xs.width

    radius = radius or xs.radius
    bend90 = gf.get_component(bend, cross_section=xs, radius=radius, width=width)
    if auto_taper:
        p1 = add_auto_tapers(component, [p1], xs, layer_transitions)[0]
        p2 = add_auto_tapers(component, [p2], xs, layer_transitions)[0]

    def straight_(width: float, length: float, **kwargs: Any) -> gf.Component:
        xs = kwargs.pop("cross_section", cross_section)
        return gf.get_component(straight, length=length, cross_section=xs, **kwargs)

    if steps and waypoints:
        raise ValueError("Provide only one of steps or waypoints")

    waypoints_list = [] if waypoints is None else list(waypoints)

    if steps:
        x, y = p1.center
        for d in steps:
            if isinstance(d, dict):
                if not STEP_DIRECTIVES.issuperset(d):
                    raise ValueError(
                        f"Invalid step directives: {list(d.keys() - STEP_DIRECTIVES)}."
                        f"Valid directives are {list(STEP_DIRECTIVES)}"
                    )
                x = d.get("x", x) + d.get("dx", 0)
                y = d.get("y", y) + d.get("dy", 0)
            else:
                raise ValueError(
                    f"Invalid step {d!r}. Each step must be a dict with keys (x, y, dx, dy)."
                )
            waypoints_list.append((x, y))

    if waypoints_list and steps and len(waypoints_list) < 2:
        p = waypoints_list[-1]
        x, y = (p.x, p.y) if hasattr(p, "x") and hasattr(p, "y") else (p[0], p[1])
        x1, y1 = p1.center
        x2, y2 = p2.center
        orientation = p2.orientation
        orientation = None if orientation is None else round(to_float(orientation))
        if orientation is not None and int(orientation) in {0, 180}:
            yt = y1 + (y2 - y1) / 3
            ytt = y1 + 2 * (y2 - y1) / 3
            waypoints_list = [(x, yt), (x, ytt)]
        elif orientation is not None and int(orientation) in {90, 270}:
            xt = x1 + (x2 - x1) / 3
            xtt = x1 + 2 * (x2 - x1) / 3
            waypoints_list = [(xt, y), (xtt, y)]

    if waypoints_list:
        w = [
            (p.x, p.y) if hasattr(p, "x") and hasattr(p, "y") else (p[0], p[1])
            for p in waypoints_list
        ]
        if isinstance(waypoints_list[0], kf.kdb.DPoint):
            # like upstream: DPoint waypoints are the full backbone (incl. the ports)
            pts = w
        else:
            # place the route through exactly these points (start, waypoints, end)
            pts = [(p1.x, p1.y), *w, (p2.x, p2.y)]
        kf_on_placer_error = (
            "error" if on_placer_error == "warning" else on_placer_error
        )
        try:
            return place_manhattan_points(
                component,
                p1,
                p2,
                pts,
                straight_factory=straight_,
                bend90=bend90,
                port_type=port_type,
                allow_width_mismatch=allow_width_mismatch,
                route_width=width,
                radius=radius,
            )
        except Exception as e:
            if on_placer_error == "error":
                raise kf.routing.generic.PlacerError(
                    f"Error while trying to place route from {p1.name} to {p2.name} at"
                    f" points (um): {[(to_float(a), to_float(b)) for a, b in pts]}"
                ) from e
            if on_placer_error == "show_error":
                _show_placer_error(component, p1, p2, pts, route_width or p1.width, e)
                raise kf.routing.generic.PlacerError(
                    f"Error while trying to place route from {p1.name} to {p2.name}"
                ) from e
            if on_placer_error == "warning":
                gf.logger.error(f"Error in route_single: {e}")
                warnings.warn(f"Routing failed: {e}", stacklevel=2)
            _ = kf_on_placer_error
            return _error_path(component, p1, p2, pts, route_width or p1.width)

    else:
        kf_on_collision = "error" if on_collision == "warning" else on_collision
        kf_on_placer_error = (
            "error" if on_placer_error == "warning" else on_placer_error
        )
        try:
            return route_bundle_kf(
                component,
                [p1],
                [p2],
                router="optical",
                straight_factory=straight_,
                bend90=bend90,
                starts=start_straight_length,
                ends=end_straight_length,
                separation=0,
                place_port_type=port_type,
                allow_width_mismatch=allow_width_mismatch,
                route_width=route_width,
                radius=radius,
                on_collision=kf_on_collision,
                on_placer_error=kf_on_placer_error,
                obstacles=component,
            )[0]
        except Exception as e:
            if on_placer_error == "error" or on_collision == "error":
                raise

            if on_placer_error == "show_error" or on_collision == "show_error":
                raise

            if on_placer_error == "warning" or on_collision == "warning":
                gf.logger.error(f"Error in route_single: {e}")
                warnings.warn(f"Routing failed: {e}", stacklevel=2)

            route = route_bundle_kf(
                component,
                [p1],
                [p2],
                router="electrical",
                separation=0,
                starts=start_straight_length,
                ends=end_straight_length,
                on_collision=None,
                on_placer_error=None,
                route_width=width,
                place_layer=gf.CONF.layer_error_path,
            )
            return route[0]


def _show_placer_error(
    component: Component, p1: Port, p2: Port, pts: Sequence[Any], width: Any, e: Exception
) -> None:
    """Shows the failed route in KLayout's marker database (like upstream)."""
    import klayout.db as kdb

    db = kf.rdb.ReportDatabase("Route Placing Errors")
    cell = db.create_cell(component.name or "")
    cat = db.create_category(f"{p1.name} - {p2.name}")
    it = db.create_item(cell=cell, category=cat)
    pts_um = [(to_float(a), to_float(b)) for a, b in pts]
    it.add_value(
        f"Error while trying to place route from {p1.name} to {p2.name} at points (um): {pts_um}"
    )
    it.add_value(f"Exception: {e}")
    path = kdb.DPath([kdb.DPoint(x, y) for x, y in pts_um], to_float(width))
    it.add_value(path.polygon())
    component.show(lyrdb=db)


def _error_path(
    component: Component, p1: Port, p2: Port, pts: Sequence[Any], width: Any
) -> ManhattanRoute:
    """Draws the requested backbone on the error layer."""
    import klayout.db as kdb

    from gdsfactory._jax import asarray, jnp
    from gdsfactory.path import Path

    pts_arr = jnp.stack([jnp.stack([asarray(a), asarray(b)]) for a, b in pts])
    path = Path(pts_arr)
    w = asarray(width)
    q1 = path.centerpoint_offset_curve(pts_arr, w / 2, path.start_angle, path.end_angle)
    q2 = path.centerpoint_offset_curve(pts_arr, -w / 2, path.start_angle, path.end_angle)
    component.add_polygon(jnp.concatenate([q1, q2[::-1]]), layer=gf.CONF.layer_error_path)
    return ManhattanRoute(
        backbone=[kdb.Point(round(to_float(a) * 1000), round(to_float(b) * 1000)) for a, b in pts],
        backbone_um=pts_arr,
        start_port=p1,
        end_port=p2,
    )


def place_manhattan_points(
    component: Component,
    p1: Port,
    p2: Port,
    pts: Sequence[Any],
    straight_factory: Any,
    bend90: Component,
    port_type: str = "optical",
    allow_width_mismatch: bool = False,
    route_width: Any = None,
    radius: Any = None,
) -> ManhattanRoute:
    """Places bends and straights along a manhattan backbone (differentiable).

    Uses kfactory's ``place_manhattan`` on concrete values for the topology and the
    differentiable rebuild of :mod:`gdsfactory.routing._kf_router`.
    """
    from gdsfactory.routing._kf_router import place_manhattan_kf

    return place_manhattan_kf(
        component,
        p1,
        p2,
        pts,
        straight_factory=straight_factory,
        bend90=bend90,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        route_width=route_width,
        radius=radius,
    )


def route_single_electrical(
    component: Component,
    port1: Port,
    port2: Port,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> None:
    """Places a route between two electrical ports.

    Args:
        component: The cell to place the route in.
        port1: The first port.
        port2: The second port.
        start_straight_length: The length of the straight at the start of the route.
        end_straight_length: The length of the straight at the end of the route.
        layer: The layer of the route.
        width: The width of the route.
        cross_section: The cross section of the route.

    """
    xs = gf.get_cross_section(cross_section)
    layer = layer or xs.layer
    width = width or xs.width
    layer = gf.get_layer(layer)
    route_bundle_kf(
        component,
        [port1],
        [port2],
        router="electrical",
        separation=0,
        route_width=width,
        # upstream passes a layer only for tuple specs, which get_layer never returns,
        # so the wire is drawn on the layer of port1
        place_layer=None,
        starts=start_straight_length,
        ends=end_straight_length,
    )
