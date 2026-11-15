from __future__ import annotations

__all__ = ["ellipse"]

import gdsfactory as gf
from gdsfactory._jax import safe_sqrt, to_float, xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def ellipse(
    radii: tuple[float, float] = (10.0, 5.0),
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns ellipse component.

    Args:
        radii: Semimajor and semiminor axis lengths of the ellipse.
        angle_resolution: number of degrees per point.
        layer: Specific layer(s) to put polygon geometry on.

    The orientation of the ellipse is determined by the order of the radii variables;
    if the first element is larger, the ellipse will be horizontal and if the second
    element is larger, the ellipse will be vertical.
    """
    c = gf.Component()
    a = radii[0]
    b = radii[1]
    n_points = max(int(360 / to_float(angle_resolution)), 2) + 1
    t = xp.linspace(0, 360, n_points) * xp.pi / 180
    r = a * b / (safe_sqrt((b * xp.cos(t)) ** 2 + (a * xp.sin(t)) ** 2))
    xpts = r * xp.cos(t)
    ypts = r * xp.sin(t)
    c.add_polygon(points=xp.column_stack([xpts, ypts]), layer=layer)
    return c
