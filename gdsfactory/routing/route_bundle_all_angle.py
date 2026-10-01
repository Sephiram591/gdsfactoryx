"""All-angle bundle routing (differentiable).

The backbone of every route (where to bend, by which angle) is computed with
kfactory's all-angle algorithm on concrete values; derivatives of the backbone
points with respect to the traced inputs (port positions, backbone points,
separation) are obtained by re-running that algorithm on perturbed inputs. The
bends and straights are then placed natively with traced geometry, so lengths,
bend angles and positions are differentiable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory._jax import asarray, is_tracer, to_float, to_numpy, xp
from gdsfactory.component import Component, ComponentReference
from gdsfactory.routing._kf_router import ManhattanRoute
from gdsfactory.typings import (
    CellAllAngleSpec,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    Port,
)

OpticalAllAngleRoute = ManhattanRoute
MIN_ALL_ANGLE_ROUTES_POINTS = 3


def _tangent(x: Any) -> Any:
    import jax

    return x - jax.lax.stop_gradient(x) if is_tracer(x) else 0.0


def _kf_backbones(
    ports1: list[dict[str, float]],
    ports2: list[dict[str, float]],
    backbone: list[tuple[float, float]],
    separation: list[float],
    bend_func: Callable[..., Component],
    bend_ports: tuple[str, str],
) -> list[np.ndarray]:
    """Backbone points of every route with kfactory's all-angle algorithm (concrete)."""
    import klayout.db as kdb
    from kfactory.routing.aa import optical as aa

    from gdsfactory.routing._kf_router import mirror

    m = mirror()
    cell = m.new_cell("aa")
    bends: dict[tuple[float, float], Any] = {}

    def bend_factory(width: float, angle: float) -> Any:
        key = (round(width, 9), round(angle, 9))
        if key not in bends:
            bends[key] = m.export(bend_func(width=width, angle=angle))
        return bends[key]

    def kport(d: dict[str, float], name: str) -> Any:
        return cell.create_port(
            name=name,
            dcplx_trans=kdb.DCplxTrans(1, d["orientation"], False, d["x"], d["y"]),
            width=round(d["width"] * 1000) / 1000,
            layer=m.layer(d["layer"]),
            port_type="optical",
        )

    sps = [kport(d, f"s{i}") for i, d in enumerate(ports1)]
    eps = [kport(d, f"e{i}") for i, d in enumerate(ports2)]
    out: list[np.ndarray] = []
    if backbone:
        pts_list = aa.backbone2bundle(
            backbone=[kdb.DPoint(x, y) for x, y in backbone],
            port_widths=[p.dwidth for p in sps],
            spacings=separation,
        )
        for ps, pe, pts in zip(sps, eps, pts_list, strict=False):
            pts_ = pts
            v_start = pts_[0] - pts_[1]
            v_end = pts_[-1] - pts_[-2]
            psb = ps.copy()
            psb.dcplx_trans = kdb.DCplxTrans(
                1, float(np.rad2deg(np.arctan2(v_start.y, v_start.x))), False, pts_[0].to_v()
            )
            peb = pe.copy()
            peb.dcplx_trans = kdb.DCplxTrans(
                1, float(np.rad2deg(np.arctan2(v_end.y, v_end.x))), False, pts_[-1].to_v()
            )
            pts_ = aa._get_connection_between_ports(
                port_start=ps,
                port_end=psb,
                bend_factory=bend_factory,
                bend_ports=bend_ports,
                backbone=pts_,
            )
            pts_.reverse()
            pts_ = aa._get_connection_between_ports(
                port_start=pe,
                port_end=peb,
                bend_factory=bend_factory,
                backbone=pts_,
                bend_ports=bend_ports,
            )
            pts_.reverse()
            out.append(np.array([[p.x, p.y] for p in pts_]))
    else:
        for ps, pe in zip(sps, eps, strict=False):
            pts_ = aa._get_connection_between_ports(
                port_start=ps,
                port_end=pe,
                bend_factory=bend_factory,
                bend_ports=bend_ports,
                backbone=[],
            )
            pts_.append(pe.dcplx_trans.disp.to_p())
            out.append(np.array([[p.x, p.y] for p in pts_]))
    cell.delete()
    return out


def _effective_radius(p1: Port, p2: Port) -> Any:
    """Distance from p1 to the crossing of the port axes (bends are symmetric)."""
    d1 = p1.direction
    d2 = p2.direction
    mat = xp.stack([d1, -d2], axis=1)
    rhs = p2.center_array - p1.center_array
    det = to_float(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    if abs(det) < 1e-12:
        return float("inf")
    st = xp.linalg.solve(mat, rhs)
    return xp.abs(st[0])


def _angle_deg(v: Any) -> Any:
    return xp.rad2deg(xp.arctan2(v[1], v[0]))


def place_all_angle_route(
    c: Component,
    width: Any,
    backbone: Any,
    straight_func: Callable[..., Component],
    bend_func: Callable[..., Component],
    bend_ports: tuple[str, str] = ("o1", "o2"),
    straight_ports: tuple[str, str] = ("o1", "o2"),
    tolerance: float = 0.1,
    angle_tolerance: float = 0.0001,
) -> ManhattanRoute:
    """Places bends and straights along an all-angle backbone (differentiable).

    Port of kfactory's ``routing.aa.optical.route`` with traced geometry.
    """
    import klayout.db as kdb

    pts = asarray(backbone)
    n = pts.shape[0]
    if n < MIN_ALL_ANGLE_ROUTES_POINTS:
        raise ValueError("All angle routes with less than 3 points are not supported.")

    bend90 = bend_func(width=width, angle=90)
    layer = bend90.ports[bend_ports[0]].layer
    start_v = pts[1] - pts[0]
    end_v = pts[-1] - pts[-2]
    start_port = Port(
        name="o1",
        center=pts[0],
        width=width,
        orientation=_angle_deg(start_v),
        layer=layer,
    )
    end_port = Port(
        name="o1",
        center=pts[-1],
        width=width,
        orientation=xp.mod(_angle_deg(end_v) + 180, 360),
        layer=layer,
    )
    dbu = 1e-3
    old_pt = pts[0]
    pt = pts[1]
    start_offset: Any = 0.0
    effective_radius: Any = 0.0
    _port = start_port
    insts: list[ComponentReference] = []
    length = xp.linalg.norm(pt - old_pt)
    length_straights: Any = 0.0

    for k in range(2, n):
        new_pt = pts[k]
        s_v = pt - old_pt
        e_v = new_pt - pt
        length = length + xp.linalg.norm(e_v)
        a = xp.mod(_angle_deg(e_v) - _angle_deg(s_v) + 180, 360) - 180
        a_f = to_float(a)
        if abs(a_f) >= angle_tolerance:
            bend = bend_func(width=width, angle=xp.abs(a))
            p1, p2 = (bend.ports[_p] for _p in bend_ports)
            effective_radius = _effective_radius(p1, p2)
            if to_float(xp.linalg.norm(pt - old_pt) - effective_radius - start_offset) < -(
                dbu * tolerance
            ):
                raise ValueError(
                    f"Not enough space to place bends at points {[to_numpy(old_pt), to_numpy(pt)]}."
                )
        else:
            effective_radius = 0.0
            a_f = 0.0
            bend = None

        straight_len = xp.linalg.norm(pt - old_pt) - effective_radius - start_offset
        if to_float(straight_len) > 0:
            s = c.add_ref(straight_func(width=width, length=straight_len))
            length_straights = length_straights + straight_len
            s.connect(straight_ports[0], _port, allow_width_mismatch=True)
            _port = s.ports[straight_ports[1]]
            insts.append(s)
        if a_f != 0 and bend is not None:
            b = c.add_ref(bend)
            if a_f < 0:
                b.connect(bend_ports[1], _port, allow_width_mismatch=True)
                _port = b.ports[bend_ports[0]]
            else:
                b.connect(bend_ports[0], _port, allow_width_mismatch=True)
                _port = b.ports[bend_ports[1]]
            insts.append(b)
        start_offset = effective_radius
        old_pt = pt
        pt = new_pt

    straight_len = xp.linalg.norm(pt - old_pt) - effective_radius
    if to_float(straight_len) < -(dbu * tolerance):
        raise ValueError(
            f"Not enough space to place bends at points {[to_numpy(old_pt), to_numpy(pt)]}."
        )
    if to_float(straight_len) > 0:
        s = c.add_ref(straight_func(width=width, length=straight_len))
        length_straights = length_straights + straight_len
        s.connect(straight_ports[0], _port, allow_width_mismatch=True)
        insts.append(s)

    pts_np = to_numpy(pts)
    return ManhattanRoute(
        backbone=[kdb.DPoint(float(x), float(y)) for x, y in pts_np],
        backbone_um=pts,
        start_port=start_port,
        end_port=end_port,
        instances=insts,
        length=length,
        length_straights=length_straights,
    )


def route_bundle_all_angle(
    component: ComponentSpec,
    ports1: list[Port],
    ports2: list[Port],
    backbone: Coordinates | None = None,
    separation: list[float] | float = 3.0,
    straight: CellAllAngleSpec = "straight_all_angle",
    bend: CellAllAngleSpec = "bend_euler_all_angle",
    bend_ports: tuple[str, str] = ("o1", "o2"),
    straight_ports: tuple[str, str] = ("o1", "o2"),
    cross_section: CrossSectionSpec | None = None,
) -> list[ManhattanRoute]:
    """Route a bundle of ports to another bundle of ports with non manhattan ports.

    Differentiable with respect to port positions, backbone points and separation.

    Args:
        component: to add the routing.
        ports1: list of start ports to connect.
        ports2: list of end ports to connect.
        backbone: list of points to connect the ports.
        separation: list of spacings.
        straight: function to create straights.
        bend: function to create bends.
        bend_ports: tuple of ports to connect the bends.
        straight_ports: tuple of ports to connect the straights.
        cross_section: cross_section to use. Overrides the  cross_section.
    """
    if cross_section:
        straight_func = gf.get_cell(straight, cross_section=cross_section)
        bend_func = gf.get_cell(bend, cross_section=cross_section)
    else:
        straight_func = gf.get_cell(straight)
        bend_func = gf.get_cell(bend)

    c = gf.get_component(component)
    ports1 = list(ports1)
    ports2 = list(ports2)
    backbone_pts: list[Sequence[Any]] = [
        (p.x, p.y) if hasattr(p, "x") and hasattr(p, "y") else (p[0], p[1])
        for p in (backbone or [])
    ]
    seps: list[Any] = (
        list(separation)
        if isinstance(separation, list | tuple)
        else [separation] * len(ports1)
    )

    def pdict(p: Port) -> dict[str, Any]:
        return {
            "x": to_float(p.x),
            "y": to_float(p.y),
            "orientation": to_float(p.orientation),
            "width": to_float(p.width),
            "layer": p.layer,
        }

    base1 = [pdict(p) for p in ports1]
    base2 = [pdict(p) for p in ports2]
    base_bb = [(to_float(x), to_float(y)) for x, y in backbone_pts]
    base_sep = [to_float(s) for s in seps]

    def run(b1: Any, b2: Any, bb: Any, sep: Any) -> list[np.ndarray]:
        return _kf_backbones(b1, b2, bb, sep, bend_func, bend_ports)

    pts_base = run(base1, base2, base_bb, base_sep)

    # traced inputs -> jacobian of the backbone points by central differences
    traced: list[tuple[Any, Callable[[float], tuple[Any, Any, Any, Any]]]] = []

    def perturbed(kind: str, i: int, axis: str | int) -> Callable[[float], tuple[Any, Any, Any, Any]]:
        def f(h: float) -> tuple[Any, Any, Any, Any]:
            b1 = [dict(d) for d in base1]
            b2 = [dict(d) for d in base2]
            bb = [list(p) for p in base_bb]
            sep = list(base_sep)
            if kind == "p1":
                b1[i][axis] += h  # type: ignore[index]
            elif kind == "p2":
                b2[i][axis] += h  # type: ignore[index]
            elif kind == "bb":
                bb[i][axis] += h  # type: ignore[index]
            else:
                sep[i] += h
            return b1, b2, [tuple(p) for p in bb], sep

        return f

    for i, p in enumerate(ports1):
        for axis in ("x", "y"):
            if is_tracer(getattr(p, axis)):
                traced.append((getattr(p, axis), perturbed("p1", i, axis)))
    for i, p in enumerate(ports2):
        for axis in ("x", "y"):
            if is_tracer(getattr(p, axis)):
                traced.append((getattr(p, axis), perturbed("p2", i, axis)))
    for i, pt in enumerate(backbone_pts):
        for j in range(2):
            if is_tracer(pt[j]):
                traced.append((pt[j], perturbed("bb", i, j)))
    for i, s in enumerate(seps):
        if is_tracer(s):
            traced.append((s, perturbed("sep", i, 0)))

    pts_traced: list[Any] = [asarray(p) for p in pts_base]
    h = 1e-4
    for value, f in traced:
        plus = run(*f(+h))
        minus = run(*f(-h))
        for r in range(len(pts_base)):
            if plus[r].shape == pts_base[r].shape and minus[r].shape == pts_base[r].shape:
                d = (plus[r] - minus[r]) / (2 * h)
                pts_traced[r] = pts_traced[r] + asarray(d) * _tangent(value)

    routes = []
    for p1, pts in zip(ports1, pts_traced, strict=False):
        # the start/end points are exactly the (traced) port centers
        routes.append(
            place_all_angle_route(
                c,
                p1.width,
                pts,
                straight_func=straight_func,
                bend_func=bend_func,
                bend_ports=bend_ports,
                straight_ports=straight_ports,
            )
        )
    return routes


__all__ = ["OpticalAllAngleRoute", "place_all_angle_route", "route_bundle_all_angle"]
