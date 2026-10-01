from __future__ import annotations

__all__ = ["torus"]

import gdsfactory as gf
from gdsfactory._jax import xp
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def torus(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a torus (annular sector / ring sector) centered at origin.

    Args:
        inner_radius: inner radius.
        outer_radius: outer radius.
        start_angle: start angle in degrees.
        end_angle: end angle in degrees.
        angle_resolution: degrees per arc point.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    if inner_radius < 0:
        raise ValueError(f"inner_radius={inner_radius} must be >= 0")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be > inner_radius")

    c = Component()
    sweep = end_angle - start_angle
    n_points = max(int(abs(sweep) / angle_resolution), 2) + 1
    theta = xp.deg2rad(xp.linspace(start_angle, end_angle, n_points, endpoint=True))

    outer_x = outer_radius * xp.cos(theta)
    outer_y = outer_radius * xp.sin(theta)
    inner_x = inner_radius * xp.cos(theta[::-1])
    inner_y = inner_radius * xp.sin(theta[::-1])

    points = list(
        zip(
            xp.concatenate([outer_x, inner_x]),
            xp.concatenate([outer_y, inner_y]),
            strict=False,
        )
    )
    c.add_polygon(points, layer=layer)

    if port_type and abs(sweep) < 360:
        width = outer_radius - inner_radius
        mid_r = (inner_radius + outer_radius) / 2
        prefix = "o" if port_type == "optical" else "e"
        sa_rad = xp.deg2rad(start_angle)
        ea_rad = xp.deg2rad(end_angle)
        c.add_port(
            f"{prefix}1",
            center=(mid_r * xp.cos(sa_rad), mid_r * xp.sin(sa_rad)),
            width=width,
            orientation=start_angle + 90,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(mid_r * xp.cos(ea_rad), mid_r * xp.sin(ea_rad)),
            width=width,
            orientation=end_angle - 90,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()
    if port_type == "electrical":
        for port in c.ports:
            c.create_pin(ports=[port], name=port.name)
    return c


if __name__ == "__main__":
    c = torus()
    c.show()
