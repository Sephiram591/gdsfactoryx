from __future__ import annotations

__all__ = ["torus_wave"]

import gdsfactory as gf
from gdsfactory._jax import xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def torus_wave(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    amplitude: float = 0.5,
    n_oscillations: int = 8,
    in_phase: bool = True,
    angle_resolution: float = 1.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a torus (full ring) with sinusoidal boundary modulation.

    Inner and outer boundaries oscillate sinusoidally. When in_phase=True,
    both boundaries are modulated in phase; when False, they are pi/2 out
    of phase.

    Args:
        inner_radius: mean inner radius.
        outer_radius: mean outer radius.
        amplitude: amplitude of boundary oscillation.
        n_oscillations: number of oscillations around the boundary.
        in_phase: if True, inner and outer modulations are in phase.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    c = Component()
    n_points = int(xp.round(360.0 / angle_resolution)) + 1
    theta = xp.linspace(0, 2 * xp.pi, n_points, endpoint=True)

    r_outer = outer_radius + amplitude * xp.sin(n_oscillations * theta)
    phase_shift = 0 if in_phase else xp.pi / 2
    r_inner = inner_radius + amplitude * xp.sin(n_oscillations * theta + phase_shift)

    outer_x = r_outer * xp.cos(theta)
    outer_y = r_outer * xp.sin(theta)
    inner_x = r_inner[::-1] * xp.cos(theta[::-1])
    inner_y = r_inner[::-1] * xp.sin(theta[::-1])

    points = list(
        zip(
            xp.concatenate([outer_x, inner_x]),
            xp.concatenate([outer_y, inner_y]),
            strict=False,
        )
    )
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = torus_wave()
    c.show()
