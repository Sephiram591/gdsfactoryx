from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kfactory import kdb

from gdsfactory._jax import get_dbu, is_tracer, to_float
from gdsfactory.routing._dual import DNum
from gdsfactory.typings import Port


class RouteWarning(UserWarning):
    pass


def to_dbu(x: Any) -> DNum:
    """Um -> dbu like kfactory (``CplxTrans(dbu).inverted() * x``), keeping derivatives."""
    if isinstance(x, DNum):
        return x
    dbu = get_dbu()
    v = kdb.CplxTrans(dbu).inverted() * to_float(x)
    if is_tracer(x):
        return DNum.traced(x / dbu, v)
    return DNum(v)


def _port_from_kport(p: Any, center: Any = None, width: Any = None) -> Port:
    """Convert a kfactory KPort into a differentiable gdsfactory Port."""
    from gdsfactory.pdk import get_layer

    t = p.dcplx_trans
    li = p.layer_info
    if width is None:
        width = p.width * get_dbu() if isinstance(p.width, int) else p.width
    if center is None:
        center = (t.disp.x, t.disp.y)
    return Port(
        name=p.name,
        center=center,
        width=width,
        orientation=t.angle,
        layer=get_layer((li.layer, li.datatype)),
        port_type=p.port_type,
    )


def direction_ports_from_list_ports(
    optical_ports: Sequence[Port],
) -> dict[str, list[Port]]:
    """Returns a dict of WENS ports."""
    direction_ports: dict[str, list[Port]] = {x: [] for x in ["E", "N", "W", "S"]}
    for p in optical_ports:
        orientation = (to_float(p.orientation) + 360.0) % 360
        if orientation <= 45.0 or orientation >= 315:
            direction_ports["E"].append(p)
        elif orientation <= 135.0:
            direction_ports["N"].append(p)
        elif orientation <= 225.0:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            list_ports.sort(key=lambda p: to_float(p.y))

        if direction in ["S", "N"]:
            list_ports.sort(key=lambda p: to_float(p.x))

    return direction_ports


def check_ports_have_equal_spacing(list_ports: Sequence[Port]) -> float:
    """Returns port separation.

    Raises error if not constant.

    """
    if not isinstance(list_ports, list):
        raise ValueError(f"list_ports should be a list of ports, got {list_ports}")
    if not list_ports:
        raise ValueError("list_ports should not be empty")

    orientation = get_list_ports_angle(list_ports)
    if orientation in [0, 180]:
        xys = [to_float(p.y) for p in list_ports]
    else:
        xys = [to_float(p.x) for p in list_ports]

    seps = [round(abs(c2 - c1), 5) for c1, c2 in zip(xys[1:], xys[:-1], strict=False)]
    different_seps = set(seps)
    if len(different_seps) > 1:
        raise ValueError(f"Ports should have the same separation. Got {different_seps}")
    return float(seps[0])


def get_list_ports_angle(list_ports: Sequence[Port]) -> float | None:
    """Returns the orientation/angle (in degrees) of a list of ports."""
    if not list_ports:
        return None
    angles = [
        to_float(p.orientation) if p.orientation is not None else None
        for p in list_ports
    ]
    if len(set(angles)) > 1:
        raise ValueError(f"All port angles should be the same. Got {list_ports}")
    return angles[0]
