from __future__ import annotations

__all__ = ["spiral_fermat"]

import numpy as np

import gdsfactory as gf
from gdsfactory._jax import to_float, xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_fermat(
    width: float = 1.0,
    n_turns: int = 5,
    a: float = 5.0,
    angle_resolution: float = 2.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a Fermat spiral: r = a * sqrt(theta).

    Args:
        width: width of the spiral trace.
        n_turns: number of turns.
        a: scaling factor.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    c = Component()

    theta_max = n_turns * 2 * np.pi
    n_points = int(np.ceil(theta_max / np.radians(to_float(angle_resolution)))) + 1
    # Start slightly above zero to avoid division issues at theta=0.
    theta = xp.linspace(1e-6, theta_max, n_points)

    r_center = a * xp.sqrt(theta)
    hw = width / 2.0

    # Tangent direction: dr/dtheta = a / (2 * sqrt(theta))
    dr = a / (2 * xp.sqrt(theta))
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
    c = spiral_fermat()
    c.show()
