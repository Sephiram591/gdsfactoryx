"""A minimal implementation of Dubins paths for waveguide routing adapted for gdsFactory by Quentin Wach.

https://quentinwach.com/blog/2024/02/15/dubins-paths-for-waveguide-routing.html
"""

from typing import Any

import gdsfactory as gf
from gdsfactory._jax import (
    safe_arctan2,
    safe_hypot,
    safe_sqrt,
    stop_gradient,
    to_float,
    xp,
)
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.bends import bend_circular_all_angle
from gdsfactory.components.waveguides import straight_all_angle
from gdsfactory.typings import CrossSectionSpec, OpticalAllAngleRoute, Port


def route_dubins(
    component: Component,
    port1: Port,
    port2: Port,
    cross_section: CrossSectionSpec,
) -> OpticalAllAngleRoute:
    """Route between ports using Dubins paths with radius from cross-section.

    Piecewise-smooth differentiable with respect to the port positions,
    port orientations, and bend radius within each discrete Dubins word region
    (the choice of path type among LSL, RSR, LSR, RSL, RLR, LRL is made on
    concrete primal values).

    Args:
        component: component to add the route to.
        port1: input port.
        port2: output port.
        cross_section: cross-section.
    """
    # Get start position and orientation
    x1, y1 = port1.center
    angle1 = port1.orientation
    START = (x1, y1, angle1)  # Convert to um

    # Get end position and orientation
    x2, y2 = port2.center
    angle2 = port2.orientation
    angle2 = xp.mod(angle2 + 180, 360)  # Adjust for input connection
    END = (x2, y2, angle2)  # Convert to um

    xs = gf.get_cross_section(cross_section)
    # Find the Dubins path between ports using radius from cross-section
    path = dubins_path(start=START, end=END, cross_section=xs)  # Convert radius to um
    instances = place_dubins_path(component, xs, port1, solution=path)
    length = sum((xp.abs(seg_len) for _, seg_len, _ in path), start=xp.asarray(0.0))
    length_straights = sum(
        (xp.abs(seg_len) for mode, seg_len, _ in path if mode == "S"),
        start=xp.asarray(0.0),
    )

    backbone = [
        gf.kdb.DPoint(to_float(x1), to_float(y1)),
        gf.kdb.DPoint(to_float(x2), to_float(y2)),
    ]
    return OpticalAllAngleRoute(
        backbone=backbone,
        backbone_um=xp.stack([xp.stack([x1, y1]), xp.stack([x2, y2])]),
        start_port=port1,
        end_port=port2,
        length=length,
        length_straights=length_straights,
        instances=instances,
    )


def general_planner(
    planner: str, alpha: Any, beta: Any, d: Any
) -> tuple[list[Any], list[str], Any] | None:
    """Finds the optimal path between two points using various planning methods."""
    sa = xp.sin(alpha)
    sb = xp.sin(beta)
    ca = xp.cos(alpha)
    cb = xp.cos(beta)
    c_ab = xp.cos(alpha - beta)
    mode = list(planner)

    planner_uc = planner.upper()

    if planner_uc == "LSL":
        tmp0 = d + sa - sb
        p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
        if to_float(p_squared) < 0:
            return None
        tmp1 = safe_arctan2((cb - ca), tmp0)
        t = mod_to_pi(-alpha + tmp1)
        p = safe_sqrt(p_squared)
        q = mod_to_pi(beta - tmp1)

    elif planner_uc == "RSR":
        tmp0 = d - sa + sb
        p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
        if to_float(p_squared) < 0:
            return None
        tmp1 = safe_arctan2((ca - cb), tmp0)
        t = mod_to_pi(alpha - tmp1)
        p = safe_sqrt(p_squared)
        q = mod_to_pi(-beta + tmp1)

    elif planner_uc == "LSR":
        p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
        if to_float(p_squared) < 0:
            return None
        p = safe_sqrt(p_squared)
        tmp2 = safe_arctan2((-ca - cb), (d + sa + sb)) - safe_arctan2(-2.0, p)
        t = mod_to_pi(-alpha + tmp2)
        q = mod_to_pi(-mod_to_pi(beta) + tmp2)

    elif planner_uc == "RSL":
        p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
        if to_float(p_squared) < 0:
            return None
        p = safe_sqrt(p_squared)
        tmp2 = safe_arctan2((ca + cb), (d - sa - sb)) - safe_arctan2(2.0, p)
        t = mod_to_pi(alpha - tmp2)
        q = mod_to_pi(beta - tmp2)

    elif planner_uc == "RLR":
        tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
        if abs(to_float(tmp_rlr)) > 1.0:
            return None

        p = mod_to_pi(
            2 * xp.pi - xp.arccos(xp.clip(tmp_rlr, -1.0 + 1e-12, 1.0 - 1e-12))
        )
        t = mod_to_pi(alpha - safe_arctan2(ca - cb, d - sa + sb) + mod_to_pi(p / 2.0))
        q = mod_to_pi(alpha - beta - t + mod_to_pi(p))

    elif planner_uc == "LRL":
        tmp_lrl = (6.0 - d * d + 2 * c_ab + 2 * d * (-sa + sb)) / 8.0
        if abs(to_float(tmp_lrl)) > 1.0:
            return None
        p = mod_to_pi(
            2 * xp.pi - xp.arccos(xp.clip(tmp_lrl, -1.0 + 1e-12, 1.0 - 1e-12))
        )
        t = mod_to_pi(-alpha - safe_arctan2(ca - cb, d + sa - sb) + p / 2.0)
        q = mod_to_pi(mod_to_pi(beta) - alpha - t + mod_to_pi(p))

    else:
        raise ValueError(f"Invalid planner: {planner}")

    path = [t, p, q]

    for i in [0, 2]:
        if planner[i].islower():
            path[i] = (2 * xp.pi) - path[i]

    cost = sum(xp.abs(v) for v in path)

    return (path, mode, cost)


def dubins_path_length(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    xs: CrossSectionSpec,
) -> Any:
    """Calculate the length of a Dubins path in um."""
    path = dubins_path(start=start, end=end, cross_section=xs)
    return sum((xp.abs(seg_len) for _, seg_len, _ in path), start=xp.asarray(0.0))


def dubins_path(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    cross_section: CrossSectionSpec,
) -> list[tuple[str, float, float]]:
    """Finds the Dubins path between two points."""
    xs = gf.get_cross_section(cross_section)
    (sx, sy, syaw) = start  # Coordinates already in um
    (ex, ey, eyaw) = end  # Coordinates already in um

    # Convert angles to radians
    syaw = xp.radians(syaw)
    eyaw = xp.radians(eyaw)

    # Use radius in um
    c = xs.radius  # Already converted to um

    assert c is not None, "Cross-section radius is None"

    # Calculate relative end position
    ex = ex - sx
    ey = ey - sy

    # Transform to local coordinates
    lex = xp.cos(syaw) * ex + xp.sin(syaw) * ey
    ley = -xp.sin(syaw) * ex + xp.cos(syaw) * ey
    leyaw = eyaw - syaw

    # Calculate normalized distance
    D = safe_hypot(lex, ley)
    d = D / c  # Normalize by radius

    # Calculate angles for path planning
    theta = mod_to_pi(safe_arctan2(ley, lex))
    alpha = mod_to_pi(-theta)
    beta = mod_to_pi(leyaw - theta)

    # Find best path
    planners = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]
    bcost: Any = float("inf")
    bt, bp, bq, bmode = None, None, None, None

    for planner in planners:
        solution = general_planner(planner, alpha, beta, d)
        if solution is None:
            continue
        (path, mode, cost) = solution
        (t, p, q) = path
        if to_float(bcost) > to_float(cost):
            bt, bp, bq, bmode = t, p, q, mode
            bcost = cost

    assert bt is not None and bp is not None and bq is not None and bmode is not None

    # Return path segments with lengths in um
    return list(zip(bmode, [bt * c, bp * c, bq * c], [c] * 3, strict=False))


def mod_to_pi(angle: Any) -> Any:
    """Normalizes an angle to the range [0, 2*pi)."""
    return angle - 2.0 * xp.pi * stop_gradient(xp.floor(angle / (2.0 * xp.pi)))


def pi_to_pi(angle: Any) -> Any:
    """Constrains an angle to the range [-pi, pi)."""
    return angle - 2.0 * xp.pi * stop_gradient(
        xp.floor((angle + xp.pi) / (2.0 * xp.pi))
    )


def linear(
    start: tuple[float, float, float], end: tuple[float, float, float], steps: int
) -> tuple[list[float], list[float]]:
    """Creates a list of points on lines between a given start point and end point.

    start/end: [x, y, angle], the start/end point with given jaw angle.
    """
    start_x, start_y = start[0], start[1]
    dx = (end[0] - start_x) / steps
    dy = (end[1] - start_y) / steps

    # Preallocate output array
    x = [start_x + i * dx for i in range(steps + 1)]
    y = [start_y + i * dy for i in range(steps + 1)]
    return x, y


def arrow_orientation(angle: float) -> tuple[float, float]:
    """Returns x, y setoffs for a given angle to orient the arrows marking the yaw of the start and end points."""
    rad = xp.radians(angle)
    alpha_x = xp.cos(rad)
    alpha_y = xp.sin(rad)
    return alpha_x, alpha_y


def place_dubins_path(
    component: Component,
    xs: CrossSectionSpec,
    port1: Port,
    solution: list[tuple[str, float, float]],
) -> list[ComponentReference]:
    """Creates GDS component with Dubins path.

    Args:
        component: component to add the route to.
        xs: cross-section.
        port1: input port.
        solution: Dubins path solution.
    """
    c = component
    current_position = port1

    instances: list[ComponentReference] = []

    for mode, length, radius in solution:
        if mode == "L":
            # Length and radius are in um
            arc_angle = 180 * length / (xp.pi * radius)
            bend = c.add_ref_off_grid(
                bend_circular_all_angle(angle=arc_angle, cross_section=xs)
            )
            bend.connect("o1", current_position)
            current_position = bend.ports["o2"]
            instances.append(bend)

        elif mode == "R":
            arc_angle = -(180 * length / (xp.pi * radius))
            bend = c.add_ref_off_grid(
                bend_circular_all_angle(angle=arc_angle, cross_section=xs)
            )
            bend.connect("o1", current_position)
            current_position = bend.ports["o2"]
            instances.append(bend)

        elif mode == "S":
            straight = c.add_ref_off_grid(
                straight_all_angle(length=length, cross_section=xs)
            )
            straight.connect("o1", current_position)
            current_position = straight.ports["o2"]
            instances.append(straight)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    return instances
