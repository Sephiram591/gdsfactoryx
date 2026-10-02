"""Differentiable front end of kfactory's manhattan bundle router.

Port (MIT License, Copyright (c) 2022 PsiQuantum Corp) of the preprocessing of
``kfactory.routing.optical.route_bundle`` / ``kfactory.routing.generic.route_bundle``
and of ``kfactory.routing.optical.path_length_match``, running
:func:`gdsfactory.routing._traced_manhattan.route_smart` on dual coordinates.

All inputs are converted to dbu exactly like kfactory (concrete values) while
keeping their JAX values, so the resulting backbone points are kfactory's
integers with the derivatives of the traced inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import klayout.db as kdb
import numpy as np
from kfactory.conf import logger
from kfactory.routing.steps import Step, Straight

from gdsfactory._jax import is_tracer, to_float
from gdsfactory.routing._dual import DNum, Point, Trans
from gdsfactory.routing._traced_manhattan import ManhattanRouter, route_smart

DBU = 1e-3


def to_dbu(x: Any) -> DNum:
    """um -> dbu like kfactory (``CplxTrans(dbu).inverted() * x``), keeping derivatives."""
    if isinstance(x, DNum):
        return x
    v = kdb.CplxTrans(DBU).inverted() * to_float(x)
    if is_tracer(x):
        return DNum.traced(x / DBU, v)
    return DNum(v)


def _steps(value: Any, n: int) -> list[list[Step]]:
    """kfactory's conversion of starts/ends (dbu numbers or Steps) to per-route Steps."""
    if value is None or (isinstance(value, list | tuple) and len(value) == 0):
        return [[] for _ in range(n)]
    if isinstance(value, DNum | int | float | np.number) or is_tracer(value) or (
        isinstance(value, np.ndarray | jax.Array) and np.ndim(value) == 0
    ):
        return [[Straight(dist=to_dbu(value))] for _ in range(n)]
    if isinstance(value, list):
        if isinstance(value[0], Step):
            return [value for _ in range(n)]
        if isinstance(value[0], list):
            return value
        return [[Straight(dist=to_dbu(s)) for s in value]] * n
    raise TypeError(f"Unsupported starts/ends {value!r}")


def path_length_match(
    routers: Sequence[ManhattanRouter],
    element: int = -1,
    loops: int = 1,
    loop_side: int = -1,
    loop_position: int = -1,
    path_length: Any = None,
) -> None:
    """Port of ``kfactory.routing.optical.path_length_match`` on dual coordinates."""
    from kfactory.routing.optical import LoopPosition, LoopSide

    loop_side = LoopSide(loop_side)
    loop_position = LoopPosition(loop_position)
    if path_length is None:
        path_length = max(router.path_length for router in routers)
    elif path_length < max(router.path_length for router in routers):
        path_length_ = max(router.path_length for router in routers)
        logger.warning(
            f"Requesting path length matching to {path_length!r}[dbu], but the minimal"
            f" possible path length is {path_length_!r}. Increasing to minimum."
        )
    path_length = DNum.of(path_length)
    if path_length % 2:
        path_length += 1
    if element is None:
        raise ValueError("Element to put path length matching must be defined")
    match loop_side:
        case LoopSide.center:
            loops += 1
    br = max(routers[0].bend90_radius, routers[0].width + routers[0].separation)
    br = DNum.of(br)
    for router in routers:
        length = router.path_length
        match loop_side:
            case LoopSide.left:
                loop_length = (path_length - length) // (loops * 2)
                pts = [
                    Point(0, 0),
                    Point(0, loop_length + 2 * br),
                    Point(2 * br, loop_length + 2 * br),
                    Point(2 * br, 0),
                ]
                for i in range(1, loops):
                    t = Trans(i * 4 * br, 0)
                    pts += [t * pt for pt in pts[:4]]
            case LoopSide.right:
                loop_length = (path_length - length) // (loops * 2)
                pts = [
                    Point(0, 0),
                    Point(0, -(loop_length + 2 * br)),
                    Point(2 * br, -(loop_length + 2 * br)),
                    Point(2 * br, 0),
                ]
                for i in range(1, loops):
                    t = Trans(i * 4 * br, 0)
                    pts += [t * pt for pt in pts[:4]]
            case LoopSide.center:
                loop_length = (path_length - length) // (loops * 2)
                lh1 = loop_length // 2
                lh2 = loop_length - lh1
                if lh1 > br:
                    lh1 -= br
                    lh2 += br
                pts = [Point(0, 0)]
                for i in range(loops):
                    pts.extend(
                        [
                            Point(4 * br * i, lh1 + 2 * br),
                            Point(4 * br * i + 2 * br, lh1 + 2 * br),
                            Point(4 * br * i + 2 * br, -(lh2)),
                            Point(4 * br * (i + 1), -(lh2)),
                        ]
                    )
                pts.extend(
                    [
                        Point(4 * br * (i + 1), 2 * br),
                        Point(4 * br * (i + 1) + 2 * br, 2 * br),
                        Point(4 * br * (i + 1) + 2 * br, 0),
                    ]
                )
            case _:
                raise ValueError(f"Argument side must be of any value of {LoopSide.__members__}")
        if element < -1:
            element_pts = router.start.pts[element - 1 : element + 1]
        elif element == -1:
            element_pts = router.start.pts[element - 1 :]
        else:
            element_pts = router.start.pts[element : element + 2]
        v = element_pts[1] - element_pts[0]
        if v.x != 0:
            d = 0 if v.x > 0 else 2
        elif v.y > 0:
            d = 1
        else:
            d = 3
        t = Trans(element_pts[0].to_v()) * Trans(d, False, 0, 0)
        match loop_position:
            case LoopPosition.start:
                if element == 0 or element == -len(router.start.pts):
                    t *= Trans(br, 0)
                else:
                    t *= Trans(2 * br, 0)
            case LoopPosition.center:
                t *= Trans(round((v.length() - pts[-1].x) / 2), 0)
            case LoopPosition.end:
                if element == 0 or element == -len(router.start.pts):
                    t *= Trans(round(v.length() - br - pts[-1].x // 2), 0)
                else:
                    t *= Trans(round(v.length() - 2 * br - pts[-1].x), 0)
            case _:
                raise ValueError(
                    f"Argument loop_position must be of any value of {LoopPosition.__members__}."
                )
        if length % 2:
            length += 1
        if loop_length * 2 * loops != path_length - length:
            l_diff = (path_length - length - loop_length * 2 * loops) // 2
            if loop_side == LoopSide.right:
                pts[1].y -= l_diff
                pts[2].y -= l_diff
            else:
                pts[1].y += l_diff
                pts[2].y += l_diff
        pts = [t * p for p in pts]
        if element < 0:
            router.start.pts[element:element] = pts
        else:
            router.start.pts[element + 1 : element + 1] = pts


def traced_route_bundle(
    start_ports: Sequence[Trans],
    end_ports: Sequence[Trans],
    widths: Sequence[Any],
    *,
    separation: Any,
    bend90_radius: Any,
    starts: Any = None,
    ends: Any = None,
    route_width: Any = None,
    sort_ports: bool = False,
    bbox_routing: str = "minimal",
    bboxes: Sequence[Any] | None = None,
    waypoints: Sequence[Point] | Trans | None = None,
    start_angles: Any = None,
    end_angles: Any = None,
    allow_sbend: bool = False,
    constraints: Sequence[Any] | None = None,
) -> list[ManhattanRouter]:
    """kfactory ``route_bundle`` (routing part only) on dual coordinates.

    Args:
        start_ports: start port transformations (dual, dbu).
        end_ports: end port transformations (dual, dbu).
        widths: port widths (dbu).
        separation: separation (um or DNum dbu).
        bend90_radius: bend radius (dbu, DNum).
        starts: start straights (um, list of um or Steps).
        ends: end straights.
        route_width: route width (um).
        sort_ports: sort ports.
        bbox_routing: "minimal" or "full".
        bboxes: obstacle boxes (dual or kdb boxes in dbu).
        waypoints: dual points (dbu) or a dual Trans.
        start_angles: overwrite start port angles (degrees).
        end_angles: overwrite end port angles.
        allow_sbend: allow sbends.
        constraints: kfactory constraints (PathLengthMatch supported).

    Returns:
        routers in kfactory's order (``router.start.pts`` is the backbone).
    """
    n = len(start_ports)
    if not n:
        return []
    angles = {0: 0, 90: 1, 180: 2, 270: 3}
    start_ports = list(start_ports)
    end_ports = list(end_ports)
    if start_angles is not None:
        sa = start_angles if isinstance(start_angles, list) else [start_angles] * n
        start_ports = [p * Trans(angles[a] - p.angle, False, 0, 0) for a, p in zip(sa, start_ports, strict=True)]
    if end_angles is not None:
        ea = end_angles if isinstance(end_angles, list) else [end_angles] * n
        end_ports = [p * Trans(angles[a] - p.angle, False, 0, 0) for a, p in zip(ea, end_ports, strict=True)]
    if route_width:
        rw = route_width if isinstance(route_width, list) else [route_width] * n
        widths = [to_dbu(w) for w in rw]
    routers = route_smart(
        start_ports=start_ports,
        end_ports=end_ports,
        widths=list(widths),
        starts=_steps(starts, n),
        ends=_steps(ends, n),
        bend90_radius=bend90_radius,
        separation=to_dbu(separation),
        sort_ports=sort_ports,
        bbox_routing=bbox_routing,
        bboxes=list(bboxes or []),
        waypoints=waypoints,
        allow_sbend=allow_sbend,
    )
    for c in constraints or []:
        from kfactory.schematic import PathLengthMatch

        if isinstance(c, PathLengthMatch):
            if c.all or len(c.route_names) == 1:
                path_length_match(
                    routers=routers,
                    element=c.element,
                    loops=c.loops,
                    loop_side=c.loop_side,
                    loop_position=c.loop_position,
                    path_length=c.length,
                )
        else:
            raise NotImplementedError(
                f"Constraint {type(c).__name__} is not supported for traced routing"
            )
    return routers


__all__ = ["path_length_match", "to_dbu", "traced_route_bundle"]
