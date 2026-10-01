from __future__ import annotations

__all__ = ["fiducial_squares"]

import gdsfactory as gf
from gdsfactory._jax import xp
from gdsfactory.typings import Float2, LayerSpecs


@gf.cell_with_module_name(tags=["shapes"])
def fiducial_squares(
    layers: LayerSpecs = ("WG",), size: Float2 = (5, 5), offset: float = 0.14
) -> gf.Component:
    """Returns fiducials with two squares.

    Args:
        layers: list of layers to draw the squares.
        size: size of each square in um.
        offset: space between squares in x and y.
    """
    c = gf.Component()

    dx, dy = (xp.array(size) + xp.array([offset, offset])) / 2

    for layer in layers:
        r = c << gf.c.rectangle(size=size, layer=layer, centered=True)
        r.move((dx, dy))

    for layer in layers:
        r = c << gf.c.rectangle(size=size, layer=layer, centered=True)
        r.move((-dx, -dy))

    return c
