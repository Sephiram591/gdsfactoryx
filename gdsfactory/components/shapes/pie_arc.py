from __future__ import annotations

__all__ = ["pie_arc"]

import gdsfactory as gf
from gdsfactory._jax import to_float, xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def pie_arc(
    radius: float = 10.0,
    radius_y: float | None = None,
    start_angle: float = 0.0,
    end_angle: float = 90.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a pie-shaped arc (sector/wedge), centered at origin.

    The shape is a closed polygon from the origin along two radial lines
    connected by an elliptical arc.

    Args:
        radius: x-radius (or uniform radius if radius_y is None).
        radius_y: y-radius for elliptical arc. Defaults to radius.
        start_angle: start angle in degrees.
        end_angle: end angle in degrees.
        angle_resolution: degrees per arc point.
        layer: layer spec.
    """
    if radius_y is None:
        radius_y = radius

    c = Component()
    sweep = end_angle - start_angle
    n_points = max(int(abs(to_float(sweep)) / to_float(angle_resolution)), 2)
    theta = xp.deg2rad(xp.linspace(start_angle, end_angle, n_points, endpoint=True))

    arc_points = xp.column_stack([radius * xp.cos(theta), radius_y * xp.sin(theta)])
    points = xp.concatenate(
        [xp.zeros((1, 2), dtype=arc_points.dtype), arc_points], axis=0
    )
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = pie_arc()
    c.show()
