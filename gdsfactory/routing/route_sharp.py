"""based on phidl.routing."""

from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory import typings
from gdsfactory._jax import asarray, to_float, xp
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection, Section
from gdsfactory.path import Path, transition
from gdsfactory.routing.route_quad import _get_rotated_basis
from gdsfactory.typings import CrossSectionSpec, LayerSpec


def path_straight(port1: typings.Port, port2: typings.Port) -> Path:
    """Return waypoint path between port1 and port2 in a straight line.

    Useful when ports point directly at each other.

    Args:
        port1: start port.
        port2: end port.

    """
    delta_orientation = round(
        abs(to_float(port1.orientation) - to_float(port2.orientation)) % 360, 3
    )
    e1, e2 = _get_rotated_basis(port1.orientation)
    displacement = asarray(port2.center) - asarray(port1.center)
    xrel = round(
        to_float(xp.dot(displacement, e1)), 3
    )  # relative position of port 2, forward/backward
    yrel = round(
        to_float(xp.dot(displacement, e2)), 3
    )  # relative position of port 2, left/right
    if (delta_orientation not in (0, 180, 360)) or (yrel != 0) or (xrel <= 0):
        raise ValueError("path_straight(): ports must point directly at each other.")
    return Path(xp.stack([asarray(port1.center), asarray(port2.center)]))


def path_L(port1: typings.Port, port2: typings.Port) -> Path:
    """Return waypoint path between port1 and port2 in an L shape.

    Useful when orthogonal ports can be directly connected with one turn.

    Args:
        port1: start port.
        port2: end port.

    """
    delta_orientation = round(
        abs((to_float(port1.orientation) - to_float(port2.orientation)) % 360), 3
    )
    if delta_orientation not in (90, 270):
        raise ValueError("path_L(): ports must be orthogonal.")
    e1, _e2 = _get_rotated_basis(port1.orientation)

    # assemble waypoints
    pt1 = asarray(port1.center)
    pt3 = asarray(port2.center)
    delta_vec = pt3 - pt1
    pt2 = pt1 + xp.dot(delta_vec, e1) * e1
    return Path(xp.stack([pt1, pt2, pt3]))


def path_U(port1: typings.Port, port2: typings.Port, length1: float = 200) -> Path:
    """Return waypoint path between port1 and port2 in a U shape.

    Useful when ports face the same direction or toward each other.

    Args:
        port1: start port.
        port2: end port.
        length1: Length of segment exiting port1. Should be larger than bend radius.

    """
    delta_orientation = round(
        abs((to_float(port1.orientation) - to_float(port2.orientation)) % 360), 3
    )
    if delta_orientation not in (0, 180, 360):
        raise ValueError("path_U(): ports must be parallel.")
    theta = xp.radians(port1.orientation)
    e1 = xp.stack([xp.cos(theta), xp.sin(theta)])
    e2 = xp.stack([-1 * xp.sin(theta), xp.cos(theta)])
    # assemble waypoints
    pt1 = asarray(port1.center)
    pt4 = asarray(port2.center)
    pt2 = pt1 + length1 * e1  # outward by length1 distance
    delta_vec = pt4 - pt2
    pt3 = pt2 + xp.dot(delta_vec, e2) * e2
    return Path(xp.stack([pt1, pt2, pt3, pt4]))


def path_J(
    port1: typings.Port, port2: typings.Port, length1: float = 200, length2: float = 200
) -> Path:
    """Return waypoint path between port1 and port2 in a J shape.

    Useful when  orthogonal ports cannot be connected directly with an L shape.

    Args:
        port1: start port.
        port2: end port.
        length1: Length of segment exiting port1. Should be larger than bend radius.
        length2: Length of segment exiting port2. Should be larger than bend radius.
    """
    delta_orientation = round(
        abs((to_float(port1.orientation) - to_float(port2.orientation)) % 360), 3
    )
    if delta_orientation not in (90, 270):
        raise ValueError("path_J(): ports must be orthogonal.")
    e1, _ = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble waypoints
    pt1 = asarray(port1.center)
    pt2 = pt1 + length1 * e1  # outward from port1 by length1
    pt5 = asarray(port2.center)
    pt4 = pt5 + length2 * e2  # outward from port2 by length2
    delta_vec = pt4 - pt2
    pt3 = pt2 + xp.dot(delta_vec, e2) * e2  # move orthogonally in e2 direction
    return Path(xp.stack([pt1, pt2, pt3, pt4, pt5]))


def path_C(
    port1: typings.Port,
    port2: typings.Port,
    length1: float = 100,
    left1: float = 100,
    length2: float = 100,
) -> Path:
    """Return waypoint path between port1 and port2 in a C shape. Useful when ports are parallel and face away from each other.

    Args:
        port1: start port.
        port2: end port.
        length1: Length of route segment coming out of port1. Should be larger than bend radius.
        left1: Length of route segment that turns left (or right if negative) from port1. Should be larger than twice the bend radius.
        length2: Length of route segment coming out of port2. Should be larger than bend radius.

    """
    delta_orientation = round(
        abs((to_float(port1.orientation) - to_float(port2.orientation)) % 360), 3
    )
    if delta_orientation not in (0, 180, 360):
        raise ValueError("path_C(): ports must be parallel.")
    e1, e_left = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble route points
    pt1 = asarray(port1.center)
    pt2 = pt1 + length1 * e1  # outward from port1 by length1
    pt3 = pt2 + left1 * e_left  # leftward by left1
    pt6 = asarray(port2.center)
    pt5 = pt6 + length2 * e2  # outward from port2 by length2
    delta_vec = pt5 - pt3
    pt4 = pt3 + xp.dot(delta_vec, e1) * e1  # move orthogonally in e1 direction
    return Path(xp.stack([pt1, pt2, pt3, pt4, pt5, pt6]))


def path_manhattan(port1: typings.Port, port2: typings.Port, radius: float) -> Path:
    """Return waypoint path between port1 and port2 using manhattan routing.

    Routing uses straight, L, U, J, or C waypoint path as needed.
    Ports must face orthogonal or parallel directions.

    Args:
        port1: start port.
        port2: end port.
        radius: Bend radius for 90 degree bend.
    """
    radius = radius + 0.1
    radius_f = to_float(radius)
    e1, e2 = _get_rotated_basis(port1.orientation)
    displacement = asarray(port2.center) - asarray(port1.center)
    xrel = xp.dot(
        displacement, e1
    )  # port2 position, forward(+)/backward(-) from port 1
    yrel = xp.dot(displacement, e2)  # port2 position, left(+)/right(-) from port1
    xrel_f = round(to_float(xrel), 3)
    yrel_f = round(to_float(yrel), 3)
    orel = round(
        abs((to_float(port2.orientation) - to_float(port1.orientation)) % 360), 3
    )  # relative orientation
    if orel not in (0, 90, 180, 270, 360):
        raise ValueError(
            "path_manhattan(): ports must face parallel or orthogonal directions."
        )
    if orel in (90, 270):
        # Orthogonal case
        if (
            (orel == 90 and yrel_f < -radius_f) or (orel == 270 and yrel_f > radius_f)
        ) and xrel_f > radius_f:
            pts = path_L(port1, port2)
        else:
            # Adjust length1 and length2 to ensure intermediate segments fit bend radius
            direction = -1 if (orel == 270) else 1
            length2 = (
                2 * radius - direction * yrel
                if (abs(radius_f + direction * yrel_f) < 2 * radius_f)
                else radius
            )
            length1 = (
                2 * radius + xrel if (abs(radius_f - xrel_f) < 2 * radius_f) else radius
            )
            pts = path_J(port1, port2, length1=length1, length2=length2)
    elif orel == 180 and yrel_f == 0 and xrel_f > 0:
        pts = path_straight(port1, port2)
    elif (orel == 180 and xrel_f <= 2 * radius_f) or (abs(yrel_f) < 2 * radius_f):
        # Adjust length1 and left1 to ensure intermediate segments fit bend radius
        left1 = (
            xp.abs(yrel) + 2 * radius if (abs(yrel_f) < 4 * radius_f) else 2 * radius
        )
        y_direction = -1 if (yrel_f < 0) else 1
        left1 = y_direction * left1
        length2 = radius
        x_direction = -1 if (orel == 180) else 1
        segmentx_length = abs(
            to_float(xrel) + x_direction * to_float(length2) - radius_f
        )
        length1 = (
            xrel + x_direction * length2 + 2 * radius
            if segmentx_length < 2 * radius_f
            else radius
        )

        pts = path_C(
            port1,
            port2,
            length1=length1,
            length2=length2,
            left1=left1,
        )
    else:
        # Adjust length1 to ensure segment comes out of port2
        length1 = radius + xrel if (orel == 0 and xrel_f > 0) else radius
        pts = path_U(port1, port2, length1=length1)
    return pts


def path_Z(
    port1: typings.Port, port2: typings.Port, length1: float = 100, length2: float = 100
) -> Path:
    """Return waypoint path between port1 and port2 in a Z shape.

    Ports can have any relative orientation.

    Args:
        port1: start port.
        port2: end port.
        length1: Length of route segment coming out of port1.
        length2: Length of route segment coming out of port2.

    """
    # get basis vectors in port directions
    e1, _ = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble route  points
    pt1 = asarray(port1.center)
    pt2 = pt1 + length1 * e1  # outward from port1 by length1
    pt4 = asarray(port2.center)
    pt3 = pt4 + length2 * e2  # outward from port2 by length2
    return Path(xp.stack([pt1, pt2, pt3, pt4]))


def path_V(port1: typings.Port, port2: typings.Port) -> Path:
    """Return waypoint path between port1 and port2 in a V shape.

    Useful when ports point to a single connecting point.

    Args:
        port1: Start port.
        port2: End port.
    """
    # Get basis vectors in port directions
    e1, _ = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)

    # Assemble route points
    pt1 = asarray(port1.center)
    pt3 = asarray(port2.center)

    # Solve for intersection
    e = xp.column_stack((e1, -1 * e2))
    distance = xp.linalg.solve(e, pt3 - pt1)[0]
    pt2 = distance * e1 + pt1
    return Path(xp.stack([pt1, pt2, pt3]))


def route_sharp(
    component: Component,
    port1: typings.Port,
    port2: typings.Port,
    width: float | None = None,
    path_type: str = "manhattan",
    manual_path: Path | None = None,
    layer: LayerSpec | None = None,
    cross_section: CrossSectionSpec | None = None,
    port_names: tuple[str, str] = ("o1", "o2"),
    **kwargs: Any,
) -> None:
    """Returns Component route between ports.

    Args:
        component: Component to add the route to.
        port1: start port.
        port2: end port.
        width: None, int, float, array-like[2], or CrossSection. \
                If None, the route linearly tapers between the widths the ports \
                If set to a single number (e.g. `width=1.7`): makes a fixed-width route \
                If set to a 2-element array (e.g. `width=[1.8,2.5]`): makes a route \
                whose width varies linearly from width[0] to width[1] \
                If set to a CrossSection: uses the CrossSection parameters for the route.
        path_type : {'manhattan', 'L', 'U', 'J', 'C', 'V', 'Z', 'straight', 'manual'}.
        manual_path: array-like[N][2] or Path Waypoint for  manual route.
        layer: Layer to put route on.
        cross_section: CrossSection to use for the route.
        port_names: Tuple of port names for the start and end of the route.
        kwargs: Keyword arguments passed to the waypoint path function.

    Method of waypoint path creation. Should be one of:

     - manhattan: automatic manhattan routing (see path_manhattan() ).
     - L: L-shaped path for orthogonal ports that can be directly connected.
     - U: U-shaped path for parallel or facing ports.
     - J: J-shaped path for orthogonal ports that cannot be directly connected.
     - C: C-shaped path for ports that face away from each other.
     - Z: Z-shaped path with three segments for ports at any angles.
     - V: V-shaped path with two segments for ports at any angles.
     - straight: straight path for ports that face each other.
     - manual: use an explicit waypoint path provided in manual_path.

    Example:
        ```python
        import gdsfactory as gf

        c = gf.Component()
        c1 = c << gf.components.pad(port_orientation=None)
        c2 = c << gf.components.pad(port_orientation=None)

        c2.movex(400)
        c2.movey(-200)

        gf.routing.route_sharp(c, c1.ports["e4"], c2.ports["e1"], path_type="L")
        c.plot()
        ```
    """
    if path_type == "C":
        p = path_C(port1, port2, **kwargs)
    elif path_type == "J":
        p = path_J(port1, port2, **kwargs)
    elif path_type == "L":
        p = path_L(port1, port2)
    elif path_type == "U":
        p = path_U(port1, port2, **kwargs)
    elif path_type == "V":
        p = path_V(port1, port2)
    elif path_type == "Z":
        p = path_Z(port1, port2, **kwargs)
    elif path_type == "manhattan":
        radius = max(port1.width, port2.width)
        p = path_manhattan(port1, port2, radius=radius)
    elif path_type == "manual":
        p = manual_path if isinstance(manual_path, Path) else Path(manual_path)
    elif path_type == "straight":
        p = path_straight(port1, port2)
    else:
        raise ValueError(
            f"route_sharp() received invalid path_type {path_type} not in "
            "{'manhattan', 'L', 'U', 'J', 'C', 'V', 'Z', 'straight', 'manual'}"
        )

    if cross_section is not None:
        cross_section = gf.get_cross_section(cross_section)
        d = p.extrude(cross_section=cross_section)
    elif width is None:
        layer = layer or port1.layer
        s1 = Section(
            width=port1.width,
            port_names=port_names,
            layer=layer,
        )
        s2 = Section(
            width=port2.width,
            port_names=port_names,
            layer=layer,
        )
        x1 = CrossSection(sections=(s1,))
        x2 = CrossSection(sections=(s2,))
        trans = transition(cross_section1=x1, cross_section2=x2, width_type="linear")
        d = p.extrude_transition(transition=trans)
    else:
        if layer is None:
            raise ValueError("layer is required for width")
        d = p.extrude(width=width, layer=layer)

    component << d
