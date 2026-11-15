from __future__ import annotations

__all__ = [
    "spiral_racetrack",
    "spiral_racetrack_fixed_length",
    "spiral_racetrack_heater_doped",
    "spiral_racetrack_heater_metal",
]

import jax
import numpy as np

import gdsfactory as gf
from gdsfactory._jax import (
    has_abstract_tracers,
    has_tracers,
    is_abstract_tracer,
    jnp,
    stop_gradient,
    to_float,
    xp,
)
from gdsfactory.component import Component
from gdsfactory.routing.route_bundle import route_bundle
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Floats,
    Port,
)

from .._schematic import spiral_schematic
from ..bends.bend_euler import bend_euler
from ..bends.bend_s import get_min_sbend_size
from ..waveguides.straight import straight


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack(
    min_radius: float | None = None,
    straight_length: float = 20.0,
    spacings: Floats = (2, 2, 3, 3, 2, 2),
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
    extra_90_deg_bend: bool = False,
    allow_min_radius_violation: bool = False,
) -> Component:
    """Returns Racetrack-Spiral.

    Args:
        min_radius: smallest radius in um.
        straight_length: length of the straight segments in um.
        spacings: space between the center of neighboring waveguides in um.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
        extra_90_deg_bend: if True, we add an additional straight + 90 degree bent at the output, so the output port is looking down.
        allow_min_radius_violation: if True, will allow the s-bend to have a smaller radius than the minimum radius.
    """
    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    min_radius = min_radius if min_radius is not None else xs.radius
    assert min_radius is not None

    _bend_s = gf.get_component(
        bend_s,
        size=(straight_length, -min_radius * 2 + 1 * spacings[0]),
        cross_section=cross_section_s or cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )
    bend_s_ref = c << _bend_s
    c.info["length"] = _bend_s.info["length"]

    ports: list[Port] = []
    for port in bend_s_ref.ports:
        for i in range(len(spacings)):
            _bend = gf.get_component(
                bend,
                angle=180,
                radius=min_radius + xp.sum(xp.asarray(spacings[:i])),
                cross_section=cross_section,
            )
            bend_ref = c << _bend
            bend_ref.connect("o1", port)

            _straight = gf.get_component(
                straight, length=straight_length, cross_section=cross_section
            )
            straight_ref = c << _straight
            straight_ref.connect("o1", bend_ref.ports["o2"])
            port = straight_ref.ports["o2"]

            c.info["length"] += _bend.info["length"] + _straight.info["length"]
        ports.append(port)

    c.add_port("o1", port=ports[0])

    if extra_90_deg_bend:
        bend_ref = c << gf.get_component(
            bend,
            angle=90,
            radius=min_radius + xp.sum(xp.asarray(spacings)),
            cross_section=cross_section,
        )
        bend_ref.connect("o1", ports[1])
        c.add_port("o2", port=bend_ref.ports["o2"])

    else:
        c.add_port("o2", port=ports[1])
    return c


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack_fixed_length(
    length: float = 1000,
    in_out_port_spacing: float = 150,
    n_straight_sections: int = 8,
    min_radius: float | None = None,
    min_spacing: float = 5.0,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = "bend_circular",
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
) -> Component:
    """Returns Racetrack-Spiral with a specified total length.

    The input and output ports are aligned in y. This class is meant to
    be used for generating interferometers with long waveguide lengths, where
    the most important parameter is the length difference between the arms.

    Args:
        length: total length of the spiral from input to output ports in um.
        in_out_port_spacing: spacing between input and output ports of the spiral in um.
        n_straight_sections: total number of straight sections for the racetrack spiral. Has to be even.
        min_radius: smallest radius in um.
        min_spacing: minimum center-center spacing between adjacent waveguides.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
    """
    c = gf.Component()

    xs_s_bend = cross_section_s or cross_section
    xs = gf.get_cross_section(xs_s_bend)
    min_radius = min_radius if min_radius is not None else xs.radius
    assert min_radius is not None

    if np.mod(n_straight_sections, 2) != 0:
        raise ValueError("The number of straight sections has to be even!")

    # get the length of the straight sections to achieve the required length
    spacings = (min_spacing,) * (n_straight_sections // 2)

    straight_length = _req_straight_len(
        length=length,
        in_out_port_spacing=in_out_port_spacing,
        min_radius=min_radius,
        spacings=spacings,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section_s_bend=xs_s_bend,
        cross_section=cross_section,
    )

    _spiral = spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=spacings,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=cross_section,
        cross_section_s=cross_section_s,
        extra_90_deg_bend=True,
    )

    spiral = c << _spiral
    c.info["length"] = _spiral.info["length"]
    c.info["straight_length"] = straight_length

    if (
        not is_abstract_tracer(spiral.ports["o1"].x)
        and not is_abstract_tracer(spiral.ports["o2"].x)
        and to_float(spiral.ports["o1"].x) > to_float(spiral.ports["o2"].x)
    ):
        spiral.mirror_x()

    # add a bit more to the spiral racetrack to make the in and out ports be aligned in y
    in_wg = c << gf.get_component(
        straight,
        length=spiral.ports["o1"].x - spiral.xmin,
        cross_section=cross_section,
    )
    if np.mod(n_straight_sections // 2, 2) == 1:
        in_wg.mirror_y()
    in_wg.connect("o1", spiral.ports["o1"])

    c.info["length"] += spiral.ports["o1"].x - spiral.xmin

    temp_component = Component()

    o2_temp = temp_component.add_port(
        name="o2_temp",
        center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
        orientation=180,
        cross_section=gf.get_cross_section(xs_s_bend),
    )

    routes = route_bundle(
        c,
        spiral.ports["o2"],
        o2_temp,
        straight=straight,
        bend=bend,
        cross_section=xs_s_bend,
        radius=min_radius,
    )

    c.add_port(
        "o2",
        center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
        orientation=0,
        cross_section=gf.get_cross_section(xs_s_bend),
    )
    c.add_port("o1", port=in_wg.ports["o2"])
    c.info["length"] += c.kcl.dbu * routes[0].length
    return c


def _spiral_total_length(
    str_len: float,
    in_out_port_spacing: float,
    min_radius: float,
    spacings: tuple[float, ...],
    bend: ComponentSpec,
    bend_s: ComponentSpec,
    cross_section: CrossSectionSpec,
    cross_section_s_bend: CrossSectionSpec,
    straight: ComponentSpec = straight,
) -> float:
    """Length from input to output port of the fixed-length spiral for a straight length."""
    _bend = gf.get_component(
        bend,
        angle=90,
        radius=min_radius,
        cross_section=cross_section_s_bend,
    )
    ports = list(_bend.ports)
    p1, p2 = ports[0], ports[1]
    tx = xp.abs(p1.x - p2.x)
    ty = xp.abs(p1.y - p2.y)
    bend_length = _bend.info.get("length", min_radius * np.pi / 2)

    _spiral = spiral_racetrack(
        min_radius=min_radius,
        straight_length=str_len,
        spacings=spacings,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=cross_section,
        cross_section_s=cross_section_s_bend,
        extra_90_deg_bend=True,
    )

    o1 = _spiral.ports["o1"]
    o2 = _spiral.ports["o2"]
    if has_abstract_tracers((o1.x, o2.x)):
        swap = o1.x > o2.x
        o1_x = xp.where(swap, -o1.x, o1.x)
        o2_x = xp.where(swap, -o2.x, o2.x)
        xmin = xp.where(swap, -_spiral.xmax, _spiral.xmin)
    elif to_float(o1.x) > to_float(o2.x):
        o1_x = -o1.x
        o2_x = -o2.x
        xmin = -_spiral.xmax
    else:
        o1_x = o1.x
        o2_x = o2.x
        xmin = _spiral.xmin

    total_length = _spiral.info["length"]
    total_length += o1_x - xmin

    target_x = o1_x + in_out_port_spacing
    target_y = o1.y
    dx = xp.abs(target_x - o2_x)
    dy = xp.abs(target_y - o2.y)
    route_length = dx - tx + dy - ty + bend_length
    return total_length + route_length  # type: ignore[no-any-return]


def _req_straight_len(
    length: float = 1000,
    in_out_port_spacing: float = 100,
    min_radius: float | None = None,
    spacings: Floats = (1.0, 1.0),
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s_bend: CrossSectionSpec = "strip",
    straight: ComponentSpec = straight,
) -> float:
    """Returns geometrical parameters to make a spiral of a given length.

    The straight length is interpolated linearly in a sweep of spiral lengths, as
    upstream. When the inputs are traced, the two bracketing sweep points are
    rebuilt with traced geometry and the interpolation formula is evaluated on
    them, so the result is differentiable in every input (with the bracket fixed).

    Args:
        length: total length of the spiral from input to output ports in um.
        in_out_port_spacing: spacing between input and output ports of the spiral in um.
        min_radius: smallest radius in um. Defaults to the radius of the cross-section.
        spacings: spacings between adjacent waveguides.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s_bend: s bend cross section
        straight: factory to generate the straight segments.
    """
    from scipy.interpolate import interp1d

    xs = gf.get_cross_section(cross_section)
    min_radius = min_radius if min_radius is not None else xs.radius
    assert min_radius is not None
    traced_args = (length, in_out_port_spacing, min_radius, *spacings)
    traced = has_tracers(traced_args)
    common = dict(
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=cross_section,
        cross_section_s_bend=cross_section_s_bend,
    )

    # Figure out the min straight for the spiral so that the inner
    # s bend has min radius within the bend radius of the waveguide
    min_straigth_length_t = get_min_sbend_size(
        (None, -min_radius * 2 + 1 * spacings[0]), cross_section_s_bend
    )

    n = 100

    def _bracket_concrete(
        length_f: float,
        in_out_f: float,
        min_r_f: float,
        *spacings_f: float,
    ) -> tuple[float, int, int]:
        min_s_len = to_float(
            get_min_sbend_size(
                (None, -min_r_f * 2 + 1 * spacings_f[0]), cross_section_s_bend
            )
        )
        if min_s_len > 0.8 * in_out_f:
            raise ValueError(
                "The maximum straight length makes the inner s bend too tight. Increase the in-out port spacing."
            )
        straight_lengths = np.linspace(min_s_len, 0.9 * in_out_f, n)
        concrete_kw = dict(
            in_out_port_spacing=in_out_f,
            min_radius=min_r_f,
            spacings=tuple(spacings_f),
            **common,
        )
        lens = [float(_spiral_total_length(s, **concrete_kw)) for s in straight_lengths]
        f = interp1d(lens, straight_lengths)
        s_val = float(f(length_f))
        order = np.argsort(lens, kind="mergesort")
        xs_sorted = np.asarray(lens)[order]
        hi = int(np.clip(np.searchsorted(xs_sorted, length_f), 1, n - 1))
        return s_val, int(order[hi - 1]), int(order[hi])

    if has_abstract_tracers(traced_args):
        i0, i1 = jax.pure_callback(
            lambda l_v, io_v, mr_v, *sp_v: tuple(
                np.asarray(idx, dtype=np.int32)
                for idx in _bracket_concrete(
                    float(l_v), float(io_v), float(mr_v), *(float(s) for s in sp_v)
                )[1:]
            ),
            (
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
            ),
            jax.lax.stop_gradient(jnp.asarray(length)),
            jax.lax.stop_gradient(jnp.asarray(in_out_port_spacing)),
            jax.lax.stop_gradient(jnp.asarray(min_radius)),
            *(jax.lax.stop_gradient(jnp.asarray(s)) for s in spacings),
            vmap_method="sequential",
        )
        a, b = min_straigth_length_t, 0.9 * in_out_port_spacing
        s0 = a + i0 * (b - a) / (n - 1)
        s1 = a + i1 * (b - a) / (n - 1)
        traced_kw = dict(
            in_out_port_spacing=in_out_port_spacing,
            min_radius=min_radius,
            spacings=tuple(spacings),
            **common,
        )
        l0 = _spiral_total_length(s0, **traced_kw)
        l1 = _spiral_total_length(s1, **traced_kw)
        dl = l1 - l0
        safe_dl = xp.where(xp.abs(dl) > 1e-30, dl, 1.0)
        return xp.where(  # type: ignore[no-any-return]
            xp.abs(dl) > 1e-30, s0 + (length - l0) * (s1 - s0) / safe_dl, s0
        )

    straight_length, i0, i1 = _bracket_concrete(
        to_float(length),
        to_float(in_out_port_spacing),
        to_float(min_radius),
        *(to_float(s) for s in spacings),
    )
    if not traced:
        return straight_length

    a, b = min_straigth_length_t, 0.9 * in_out_port_spacing
    s0 = a + i0 * (b - a) / (n - 1)
    s1 = a + i1 * (b - a) / (n - 1)
    traced_kw = dict(
        in_out_port_spacing=in_out_port_spacing,
        min_radius=min_radius,
        spacings=tuple(spacings),
        **common,
    )
    l0 = _spiral_total_length(s0, **traced_kw)
    l1 = _spiral_total_length(s1, **traced_kw)
    t = s0 + (length - l0) * (s1 - s0) / (l1 - l0)
    return straight_length + (t - stop_gradient(t))  # type: ignore[no-any-return]


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack_heater_metal(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "heater_metal",
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
) -> Component:
    """Returns spiral racetrack with a heater above.

    based on <https://doi.org/10.1364/OL.400230> .

    Args:
        min_radius: smallest radius. Defaults to the radius of the cross-section.
        straight_length: length of the straight segments.
        spacing: space between the center of neighboring waveguides.
        num: number of loops.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
        via_stack: via stack to connect the heater to the metal layer.
    """
    c = gf.Component()
    xs = gf.get_cross_section(waveguide_cross_section)
    min_radius = min_radius if min_radius is not None else (xs.radius or 0)
    assert (
        min_radius is not None
    )  # for type checkers: the `or 0` above never gives None

    spiral = c << spiral_racetrack(
        min_radius,
        straight_length,
        (spacing,) * num,
        straight,
        bend,
        bend_s,
        waveguide_cross_section,
    )

    heater_top = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_top.connect(
        "e1",
        spiral.ports["o1"].copy().copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_top.movey(spacing * num // 2)
    heater_bot = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_bot.connect(
        "e1",
        spiral.ports["o2"].copy().copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_bot.movey(-spacing * num // 2)

    heater_bend = c << gf.get_component(
        bend,
        angle=180,
        radius=min_radius + spacing * (num // 2 + 1),
        cross_section=heater_cross_section,
    )
    heater_bend.y = spiral.y
    heater_bend.x = spiral.x + min_radius + spacing * (num // 2 + 1)
    heater_top.connect("e1", heater_bend.ports["e1"])
    heater_bot.connect("e1", heater_bend.ports["e2"])

    c.add_ports(spiral.ports)

    if via_stack:
        via_stack = gf.get_component(via_stack)
        via_stack_top = c << via_stack
        via_stack_bot = c << via_stack
        via_stack_top.connect(
            "e3",
            heater_bot.ports["e2"],
            allow_layer_mismatch=True,
            allow_width_mismatch=True,
        )
        via_stack_bot.connect(
            "e3",
            heater_top.ports["e2"],
            allow_layer_mismatch=True,
            allow_width_mismatch=True,
        )

        p1 = via_stack_top.ports
        p2 = via_stack_bot.ports
        c.add_ports(p1, prefix="top_")
        c.add_ports(p2, prefix="bot_")

    else:
        c.add_port("e1", port=heater_bot["e2"])
        c.add_port("e2", port=heater_top["e2"])

    top_ports = [p for p in c.ports if p.name and p.name.startswith("top_")]
    bot_ports = [p for p in c.ports if p.name and p.name.startswith("bot_")]
    if top_ports:
        c.create_pin(ports=top_ports, name="top")
    if bot_ports:
        c.create_pin(ports=bot_ports, name="bot")
    e1_port = [p for p in c.ports if p.name == "e1"]
    e2_port = [p for p in c.ports if p.name == "e2"]
    if e1_port:
        c.create_pin(ports=e1_port, name="e1")
    if e2_port:
        c.create_pin(ports=e2_port, name="e2")

    return c


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack_heater_doped(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "npp",
) -> Component:
    """Returns spiral racetrack with a heater between the loops.

    based on <https://doi.org/10.1364/OL.400230> but with the heater between the loops.

    Args:
        min_radius: smallest radius in um. Defaults to the radius of the cross-section.
        straight_length: length of the straight segments in um.
        spacing: space between the center of neighboring waveguides in um.
        num: number.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
    """
    xs = gf.get_cross_section(waveguide_cross_section)
    min_radius = min_radius if min_radius is not None else (xs.radius or 0)

    c = gf.Component()

    spiral = c << spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=(spacing,) * (num // 2)
        + (spacing + 1,) * 2
        + (spacing,) * (num // 2 - 2),
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=waveguide_cross_section,
    )

    heater_straight = gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )

    heater_top = c << heater_straight
    heater_bot = c << heater_straight

    heater_bot.connect(
        "e1",
        spiral.ports["o1"].copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_bot.movey(-spacing * (num // 2 - 1))
    heater_top.connect(
        "e1",
        spiral.ports["o2"].copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_top.movey(spacing * (num // 2 - 1))

    c.add_ports(spiral.ports)
    c.add_ports(prefix="top_", ports=heater_top.ports)
    c.add_ports(prefix="bot_", ports=heater_bot.ports)

    top_ports = [p for p in c.ports if p.name and p.name.startswith("top_")]
    bot_ports = [p for p in c.ports if p.name and p.name.startswith("bot_")]
    if top_ports:
        c.create_pin(ports=top_ports, name="top")
    if bot_ports:
        c.create_pin(ports=bot_ports, name="bot")

    return c
