from __future__ import annotations

__all__ = ["circle"]

import gdsfactory as gf
from gdsfactory._jax import xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def circle(
    radius: float = 10.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Generate a circle geometry.

    Args:
        radius: of the circle.
        angle_resolution: number of degrees per point.
        layer: layer.
    """
    if radius <= 0:
        raise ValueError(f"radius={radius} must be > 0")
    c = Component()
    num_points = int(xp.round(360.0 / angle_resolution)) + 1
    theta = xp.deg2rad(xp.linspace(0, 360, num_points, endpoint=True))
    points = xp.stack((radius * xp.cos(theta), radius * xp.sin(theta)), axis=-1)
    c.add_polygon(points=points, layer=layer)
    return c
