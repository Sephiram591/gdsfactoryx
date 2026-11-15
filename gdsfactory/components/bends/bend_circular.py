from __future__ import annotations

__all__ = ["bend_circular", "bend_circular180", "bend_circular_all_angle"]

import warnings
from functools import partial
from typing import Literal, overload

import numpy as np

import gdsfactory as gf
from gdsfactory._jax import (
    has_abstract_tracers,
    is_abstract_tracer,
    maybe_float,
    to_float,
    to_numpy,
    xp,
)
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.path import arc
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from .._schematic import bend_schematic


@overload
def _bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: Literal[False] = False,
    angular_step: float | None = None,
) -> Component: ...


@overload
def _bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: Literal[True] = True,
    angular_step: float | None = None,
) -> ComponentAllAngle: ...


def _bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: bool = False,
    angular_step: float | None = None,
) -> Component | ComponentAllAngle:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        all_angle: if True returns a ComponentAllAngle.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.

    ```text
                  o2
                  |
                 /
                /
        o1_____/
    ```
    """
    x = gf.get_cross_section(cross_section)
    radius = radius if radius is not None else x.radius
    assert radius is not None
    if layer is not None and width is not None:
        x = gf.get_cross_section(cross_section, layer=layer, width=width)
    elif layer is not None:
        x = gf.get_cross_section(cross_section, layer=layer)
    elif width is not None:
        x = gf.get_cross_section(cross_section, width=width)

    p = arc(radius=radius, angle=angle, npoints=npoints, angular_step=angular_step)
    c = p.extrude(x, all_angle=all_angle)

    c.info["length"] = maybe_float(snap_to_grid(p.length()))
    c.info["dy"] = maybe_float(abs(p.points[0][0] - p.points[-1][0]))
    c.info["radius"] = maybe_float(radius)
    c.info["width"] = width if width is not None else x.width
    angle_val = None if is_abstract_tracer(angle) else int(to_float(angle))
    top = None if angle_val in {180, -180, -90} else 0
    bottom = 0 if angle_val == -90 else None
    x.add_bbox(c, top=top, bottom=bottom)
    if not allow_min_radius_violation:
        x.validate_radius(radius)
    c.add_route_info(
        cross_section=x,
        length=c.info["length"],
        n_bend_90=abs(angle / 90.0),
        min_bend_radius=radius,
    )
    signed_radius = (
        xp.where(angle > 0, -radius, radius)
        if is_abstract_tracer(angle)
        else (-radius if to_float(angle) > 0 else radius)
    )
    if has_abstract_tracers((p.points, angle)):
        start_names = {s.port_names[0] for s in x.sections if s.port_names}
        end_names = {s.port_names[1] for s in x.sections if s.port_names}
        for prt in c.ports:
            if prt.name in start_names:
                prt.info["radius"] = signed_radius
            elif prt.name in end_names:
                prt.info["radius"] = -signed_radius
    else:
        for prt in c.ports:  # positive radius for outward left turning
            if np.allclose(to_numpy(prt.center_array), to_numpy(p.points[0])):
                prt.info["radius"] = signed_radius
            elif np.allclose(to_numpy(prt.center_array), to_numpy(p.points[-1])):
                prt.info["radius"] = -signed_radius
    return c


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    angular_step: float | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    if not is_abstract_tracer(angle) and abs(to_float(angle)) not in {90, 180}:
        warnings.warn(
            f"bend_circular angle should be 90 or 180. Got {angle}. Use bend_circular_all_angle instead.",
            UserWarning,
            stacklevel=3,
        )
    return _bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=False,
        angular_step=angular_step,
    )


@gf.vcell
def bend_circular_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    angular_step: float | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> ComponentAllAngle:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return _bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=True,
        angular_step=angular_step,
    )


bend_circular180 = partial(bend_circular, angle=180)
