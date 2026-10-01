from __future__ import annotations

__all__ = ["taper_parabolic"]

import gdsfactory as gf
from gdsfactory._jax import xp
from gdsfactory.path import transition_exponential
from gdsfactory.typings import LayerSpec

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper_parabolic(
    length: float = 20,
    width1: float = 0.5,
    width2: float = 5.0,
    exp: float = 0.5,
    npoints: int = 100,
    layer: LayerSpec = "WG",
) -> gf.Component:
    """Returns a parabolic_taper.

    Args:
        length: in um.
        width1: in um.
        width2: in um.
        exp: exponent.
        npoints: number of points.
        layer: layer spec.
    """
    x = xp.linspace(0, 1, npoints)
    y = transition_exponential(y1=width1, y2=width2, exp=exp)(x) / 2

    x = length * x
    points1 = xp.stack([x, y]).T
    points2 = xp.flipud(xp.stack([x, -y]).T)
    points = xp.concatenate([points1, points2])

    c = gf.Component()
    c.add_polygon(points, layer=layer)
    c.add_port(name="o1", center=(0, 0), width=width1, orientation=180, layer=layer)
    c.add_port(name="o2", center=(length, 0), width=width2, orientation=0, layer=layer)
    return c
