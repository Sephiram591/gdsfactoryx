"""Optimizes layout parameters with gradients (gdsfactoryx JAX backend).

1. Finds the vertical offset of an MMI so that the route connecting it has a
   target length (gradient flows through kfactory's router topology and the
   placed bends/straights).
2. Finds the width of a waveguide so that its rasterized fill factor in a
   window reaches a target (gradient flows through the soft rasterizer).
"""

from __future__ import annotations

from typing import Any

import jax

import gdsfactory as gf


def route_length(dy: Any) -> Any:
    c = gf.Component()
    m1 = c << gf.components.mmi1x2()
    m2 = c << gf.components.mmi1x2()
    m2.move((60.0, dy))
    route = gf.routing.route_single(
        c, m1.ports["o2"], m2.ports["o1"], radius=5, cross_section="strip"
    )
    return route.length * 1e-3  # um


def fill_factor(width: Any) -> Any:
    c = gf.components.straight(length=5, width=width)
    rho, _ = gf.rasterize.rasterize(
        c, layer=(1, 0), bbox=(0, -1.5, 5, 1.5), resolution=0.05
    )
    return rho.mean()


def optimize(
    f: Any, x0: float, target: float, lr: float, steps: int = 30
) -> tuple[float, float]:
    loss_and_grad = jax.value_and_grad(lambda x: (f(x) - target) ** 2)
    x = x0
    for _ in range(steps):
        _, g = loss_and_grad(x)
        x = x - lr * float(g)
    return x, float(f(x))


if __name__ == "__main__":
    gf.gpdk.PDK.activate()

    dy, length = optimize(route_length, x0=20.0, target=110.0, lr=0.4)
    print(f"dy = {dy:.4f} um -> route length {length:.4f} um (target 110)")

    width, ff = optimize(fill_factor, x0=0.5, target=0.4, lr=2.0)
    print(f"width = {width:.4f} um -> fill factor {ff:.4f} (target 0.4)")
