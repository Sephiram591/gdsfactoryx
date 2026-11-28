"""Route for electrical based on phidl.routing.route_quad."""

from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory._jax import asarray, safe_arctan2, to_float, xp
from gdsfactory.typings import Port


def _get_rotated_basis(
    theta: Any,
) -> tuple[Any, Any]:
    """Returns basis vectors rotated CCW by theta (in degrees)."""
    theta = xp.radians(theta)
    e1 = xp.stack([xp.cos(theta), xp.sin(theta)])
    e2 = xp.stack([-1 * xp.sin(theta), xp.cos(theta)])
    return e1, e2


def route_quad(
    component: gf.Component,
    port1: Port,
    port2: Port,
    width1: float | None = None,
    width2: float | None = None,
    layer: gf.typings.LayerSpec = "M1",
    manhattan_target_step: float | None = None,
) -> None:
    """Routes a basic quadrilateral polygon directly between two ports.

    Args:
        component: Component to add the route to.
        port1: Port to start route.
        port2 : Port objects to end route.
        width1: Width of quadrilateral at ports. If None, uses port widths.
        width2: Width of quadrilateral at ports. If None, uses port widths.
        layer: Layer to put the route on.
        manhattan_target_step: if not none, min step to manhattanize the polygon

    Example:
        ```python
        import gdsfactory as gf

        c = gf.Component()
        pad1 = c << gf.components.pad(size=(50, 50))
        pad2 = c << gf.components.pad(size=(10, 10))
        pad2.movex(100)
        pad2.movey(50)
        gf.routing.route_quad(
        c,
        pad1.ports["e2"],
        pad2.ports["e4"],
        width1=None,
        width2=None,
        )
        c.plot()
        ```
    """

    def get_port_edges(port: Port, width: Any) -> tuple[Any, Any]:
        _, e1 = _get_rotated_basis(port.orientation)
        center = asarray(port.center)
        pt1 = center + e1 * width / 2
        pt2 = center - e1 * width / 2
        return pt1, pt2

    if width1 is None:
        width1 = port1.width
    if width2 is None:
        width2 = port2.width
    vertices = xp.stack(
        list(get_port_edges(port1, width1) + get_port_edges(port2, width2))
    )
    center = xp.mean(vertices, axis=0)
    displacements = vertices - center
    # sort vertices by angle from center of quadrilateral to make convex polygon
    angles = safe_arctan2(displacements[:, 0], displacements[:, 1])
    sorted_idx = [
        idx for idx, _ in sorted(enumerate(angles), key=lambda x: to_float(x[1]))
    ]
    sorted_vertices = xp.stack([vertices[i] for i in sorted_idx])

    if manhattan_target_step is not None:
        component.add_polygon(
            sorted_vertices,
            layer=layer,
        )
    else:
        component.add_polygon(points=sorted_vertices, layer=layer)
