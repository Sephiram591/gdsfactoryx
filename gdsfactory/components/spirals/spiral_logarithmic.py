from __future__ import annotations

__all__ = ["spiral_logarithmic"]

import numpy as np

import gdsfactory as gf
from gdsfactory._jax import to_float, xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_logarithmic(
    width: float = 0.5,
    n_turns: int = 4,
    a: float = 1.0,
    b: float = 0.1,
    angle_resolution: float = 2.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a logarithmic spiral: r = a * exp(b * theta).

    Args:
        width: width of the spiral trace.
        n_turns: number of turns.
        a: initial radius scaling factor.
        b: growth rate.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    c = Component()

    theta_max = n_turns * 2 * np.pi
    n_points = int(np.ceil(theta_max / np.radians(to_float(angle_resolution)))) + 1
    theta = xp.linspace(0, theta_max, n_points)

    r_center = a * xp.exp(b * theta)
    hw = width / 2.0

    # Tangent direction: dr/dtheta = a * b * exp(b * theta) = b * r
    dr = b * r_center
    tx = dr * xp.cos(theta) - r_center * xp.sin(theta)
    ty = dr * xp.sin(theta) + r_center * xp.cos(theta)
    t_len = xp.sqrt(tx**2 + ty**2)
    t_len = xp.where(t_len == 0, 1.0, t_len)
    nx = -ty / t_len
    ny = tx / t_len

    x_center = r_center * xp.cos(theta)
    y_center = r_center * xp.sin(theta)

    outer_x = x_center + nx * hw
    outer_y = y_center + ny * hw
    inner_x = x_center - nx * hw
    inner_y = y_center - ny * hw

    points_x = xp.concatenate([outer_x, inner_x[::-1]])
    points_y = xp.concatenate([outer_y, inner_y[::-1]])
    points = xp.stack((points_x, points_y), axis=-1)

    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = spiral_logarithmic()
    c.show()
