from __future__ import annotations

__all__ = ["array_polar"]

import math

from kfactory.conf import CheckInstances

import gdsfactory as gf
from gdsfactory._jax import to_float, xp
from gdsfactory.component import Component
from gdsfactory.transform import Transform
from gdsfactory.typings import ComponentSpec


@gf.cell(
    with_module_name=True,
    check_instances=CheckInstances.IGNORE,
    tags=["containers"],
)
def array_polar(
    component: ComponentSpec = "C",
    n_items: int = 6,
    radius: float = 50.0,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    rotate_items: bool = True,
    add_ports: bool = True,
) -> Component:
    """Returns a polar/circular array of components.

    Places component refs at equal angular intervals around a circle.

    Args:
        component: component to replicate.
        n_items: number of items in the array.
        radius: radius of the circle.
        start_angle: starting angle in degrees.
        end_angle: ending angle in degrees.
        rotate_items: if True, rotate each item to point radially outward.
        add_ports: add ports from each element.
    """
    c = Component()
    comp = gf.get_component(component)

    if abs(to_float(end_angle) - to_float(start_angle)) >= 360.0:
        angles = xp.linspace(start_angle, end_angle, n_items, endpoint=False)
    else:
        angles = xp.linspace(start_angle, end_angle, n_items, endpoint=True)

    for i, angle_deg in enumerate(angles):
        angle_rad = xp.radians(angle_deg)
        x = radius * xp.cos(angle_rad)
        y = radius * xp.sin(angle_rad)

        ref = c.add_ref(comp)
        if rotate_items:
            ref.rotate(angle_deg)
        ref.move((x, y))

        if add_ports and comp.ports:
            # upstream copies the ports with the instance's simple (integer)
            # transformation `ref.trans`: rotation floored to a multiple of 90 deg
            # and displacement on the dbu grid.
            t = ref.transform
            rot90 = 90 * (math.floor((to_float(t.rotation) % 360) / 90 + 1e-9) % 4)
            trans = Transform(
                gf.snap.snap_to_grid(t.x),
                gf.snap.snap_to_grid(t.y),
                rot90,
                t.mirror,
            )
            for port in comp.ports:
                name = f"{port.name}_{i + 1}"
                c.add_port(name, port=port.copy(trans))

    elec_ports = [p for p in c.ports if p.name and p.port_type == "electrical"]
    for p in elec_ports:
        c.create_pin(ports=[p], name=p.name)

    return c


if __name__ == "__main__":
    c = array_polar()
    c.show()
