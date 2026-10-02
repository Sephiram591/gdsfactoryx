"""All-angle bundle routing (differentiable).

The backbone of every route is computed with kfactory's all-angle algorithm
(``kfactory.routing.aa.optical``, MIT License, Copyright (c) 2022 PsiQuantum
Corp) and, in parallel, by a traced port of the same algorithm:

- bundle offsets (``backbone2bundle``) and port/bundle connections are vector
  geometry (offset edges, line intersections) written with ``xp``,
- the connection angle that kfactory finds with ``scipy.optimize.minimize_scalar``
  keeps kfactory's value; its derivative comes from the implicit function
  theorem applied to the optimality condition of the same objective
  (``jax.grad`` of the objective), so it is exact.

The points take kfactory's values (straight-through) and the derivatives of
the traced port, which is checked to agree with kfactory to 1e-6 um. The bends
and straights are then placed with traced geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

import gdsfactory as gf
from gdsfactory._jax import asarray, has_tracers, is_tracer, stop_gradient, to_float, to_numpy, xp
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
_POINT_TOLERANCE = 1e-6  # um, traced port vs kfactory


# ---------------------------------------------------------------------------
# traced geometry helpers
# ---------------------------------------------------------------------------
def _vec(x: Any, y: Any) -> Any:
    return xp.stack([asarray(x), asarray(y)])


def _dir(angle_deg: Any) -> Any:
    a = xp.deg2rad(asarray(angle_deg))
    return xp.stack([xp.cos(a), xp.sin(a)])


def _cut(p1: Any, d1: Any, p2: Any, d2: Any) -> Any:
    """Intersection of the lines p1 + s d1 and p2 + t d2 (None if parallel)."""
    det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
    if abs(to_float(det)) < 1e-14:
        return None
    rhs = p2 - p1
    s = (rhs[0] * (-d2[1]) - (-d2[0]) * rhs[1]) / det
    return p1 + s * d1


def _shifted(p1: Any, p2: Any, d: Any) -> tuple[Any, Any]:
    """Edge p1->p2 shifted by d to its left (KLayout DEdge.shifted)."""
    v = p2 - p1
    n = xp.stack([-v[1], v[0]]) / xp.sqrt(xp.sum(v**2))
    return p1 + d * n, p2 + d * n


def _fdiv(a: Any, b: Any) -> float:
    """Python float floor division as kfactory does it on um values.

    Piecewise constant, so its derivative is zero. (Only the centering of
    a bundle is floored. The pitch between routes, pw + spacing, stays
    continuous because kfactory adds back the remainders.)
    """
    return to_float(a) // to_float(b)


class _P:
    """Traced port frame (center, orientation in degrees, width)."""

    def __init__(self, center: Any, angle: Any, width: Any) -> None:
        self.c = asarray(center)
        self.a = angle
        self.w = width

    def dir(self) -> Any:
        return _dir(self.a)


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


def _kf_port(cell: Any, m: Any, p: _P, name: str, layer: Any) -> Any:
    import klayout.db as kdb

    return cell.create_port(
        name=name,
        dcplx_trans=kdb.DCplxTrans(1, to_float(p.a), False, to_float(p.c[0]), to_float(p.c[1])),
        width=round(to_float(p.w) * 1000) / 1000,
        layer=m.layer(layer),
        port_type="optical",
    )


class _Router:
    """Runs kfactory's all-angle backbone algorithm and its traced port side by side."""

    def __init__(
        self, cell: Any, bend_func: Callable[..., Component], bend_ports: tuple[str, str], layer: Any
    ) -> None:
        from gdsfactory.routing._kf_router import mirror

        self.m = mirror()
        self.cell = cell
        self.bend_func = bend_func
        self.bend_ports = bend_ports
        self.layer = layer
        self._bends: dict[tuple[float, float], Any] = {}
        self._n = 0

    def kf_bend(self, width: float, angle: float) -> Any:
        key = (round(width, 9), round(angle, 9))
        if key not in self._bends:
            self._bends[key] = self.m.export(self.bend_func(width=width, angle=angle))
        return self._bends[key]

    def kport(self, p: _P) -> Any:
        self._n += 1
        return _kf_port(self.cell, self.m, p, f"p{self._n}", self.layer)

    # ------------------------------------------------------- traced pieces
    def eff_radius(self, width: Any, angle: Any) -> Any:
        bend = self.bend_func(width=width, angle=angle)
        p1, p2 = (bend.ports[n] for n in self.bend_ports)
        return _effective_radius(p1, p2)

    def partial_route(self, theta: Any, ps: _P, pe: _P) -> tuple[Any, Any, Any]:
        """Traced kfactory ``_get_partial_route2`` (signed r2)."""
        bend_angle = xp.mod(180 - theta + ps.a, 180)
        radius = self.eff_radius(ps.w, xp.abs(bend_angle))
        rp = ps.c + radius * ps.dir()
        xe = _cut(rp, _dir(theta), pe.c, pe.dir())
        if xe is None:
            return rp, None, None
        bend2_angle = xp.abs(xp.mod(-theta + pe.a + 180, 360) - 180)
        er2 = self.eff_radius(ps.w, bend2_angle)
        r2 = xp.sqrt(xp.sum((xe - pe.c) ** 2)) - er2
        return rp, xe, r2

    def optimal_angle(self, kps: Any, kpe: Any, ps: _P, pe: _P) -> Any:
        """kfactory's optimized angle (value) with its implicit derivative."""
        import kfactory.routing.aa.optical as aa
        import klayout.db as kdb
        from scipy.optimize import minimize_scalar

        p0 = kdb.DPoint(0, 0)
        p1 = kdb.DPoint(1, 0)

        def obj(angle: float) -> float:
            return aa._get_partial_route(
                angle=angle,
                bend_factory=self.kf_bend,
                bend_ports=self.bend_ports,
                start_port=kps,
                end_port=kpe,
                _p0=p0,
                _p1=p1,
            )[2]

        theta = float(minimize_scalar(obj, bounds=(-180, 180)).x)
        traced = any(is_tracer(v) for v in (ps.c, ps.a, ps.w, pe.c, pe.a, pe.w))
        if not traced:
            return theta

        # kfactory minimizes |r2|; at its solution r2(theta, inputs) = 0, so
        # dtheta/dinputs = -(dr2/dinputs) / (dr2/dtheta)  (implicit function theorem)
        def r2(th: Any, ps_: _P, pe_: _P) -> Any:
            return self.partial_route(th, ps_, pe_)[2]

        concrete = lambda q: _P(to_numpy(q.c), to_float(q.a), to_float(q.w))  # noqa: E731
        psc, pec = concrete(ps), concrete(pe)
        if r2(theta, psc, pec) is None:
            return theta
        dr2 = float(jax.grad(lambda th: r2(th, psc, pec))(theta))
        if abs(dr2) < 1e-12:
            return theta
        r = r2(theta, ps, pe)
        return theta - (r - stop_gradient(r)) / dr2

    def connection(self, ps: _P, pe: _P, backbone: list[Any]) -> list[Any]:
        """Traced kfactory ``_get_connection_between_ports``."""
        import kfactory.routing.aa.optical as aa
        import klayout.db as kdb

        kps, kpe = self.kport(ps), self.kport(pe)
        ts, te = kps.dcplx_trans, kpe.dcplx_trans
        p0, p1 = kdb.DPoint(0, 0), kdb.DPoint(1, 0)
        xing_k = kdb.DEdge(ts * p0, ts * p1).cut_point(kdb.DEdge(te * p0, te * p1))
        if xing_k is not None:
            vx = ts.inverted() * xing_k
            vb = te.inverted() * xing_k
            if vx.x > 0 and vb.x > 0:
                xing = _cut(ps.c, ps.dir(), pe.c, pe.dir())
                return [ps.c, xing, *backbone]
        theta = self.optimal_angle(kps, kpe, ps, pe)
        rp, xe, _ = self.partial_route(theta, ps, pe)
        kv = aa._get_partial_route2(
            angle=to_float(theta),
            bend_factory=self.kf_bend,
            bend_ports=self.bend_ports,
            start_port=kps,
            end_port=kpe,
            _p0=p0,
            _p1=p1,
        )
        if xe is None or not isinstance(kv, tuple):
            raise RuntimeError(f"Cannot find an automatic route from {kps} to bundle port {kpe}")
        return [ps.c, rp, xe, *backbone[1:]]


def _backbone2bundle(backbone: list[Any], widths: list[Any], spacings: list[Any]) -> list[list[Any]]:
    """Traced kfactory ``backbone2bundle``."""
    edges = list(zip(backbone[:-1], backbone[1:], strict=True))
    width = sum(widths) + sum(spacings)
    x = _fdiv(-width, 2)
    out = []
    for pw, spacing in zip(widths, spacings, strict=False):
        x = x + _fdiv(pw, 2) + _fdiv(spacing, 2)
        e1 = _shifted(*edges[0], -x)
        pts = [e1[0]]
        for e in edges[1:]:
            e2 = _shifted(*e, -x)
            pts.append(_cut(e2[0], e2[1] - e2[0], e1[0], e1[1] - e1[0]))
            e1 = e2
        pts.append(e1[1])
        x = x + spacing - _fdiv(spacing, 2) + pw - _fdiv(pw, 2)
        out.append(pts)
    return out


def _backbones(
    ports1: list[Port],
    ports2: list[Port],
    backbone: list[tuple[Any, Any]],
    separation: list[Any],
    bend_func: Callable[..., Component],
    bend_ports: tuple[str, str],
) -> list[Any]:
    """Backbone points of every route: kfactory values with traced derivatives."""
    import kfactory.routing.aa.optical as aa
    import klayout.db as kdb

    from gdsfactory.routing._kf_router import _CHECK_TRACED_ROUTER, mirror

    layer = ports1[0].layer
    p1s = [_P(p.center_array, p.orientation, p.width) for p in ports1]
    p2s = [_P(p.center_array, p.orientation, p.width) for p in ports2]

    traced = _CHECK_TRACED_ROUTER or has_tracers(
        ([(p.c, p.a, p.w) for p in (*p1s, *p2s)], backbone, list(separation))
    )
    if not traced:
        w = to_float(p1s[0].w)
        traced = bend_func(width=w, angle=90).has_tracers()

    with mirror().scratch() as cell:
        r = _Router(cell, bend_func, bend_ports, layer)
        # kfactory's own computation (concrete values)
        sps = [r.kport(_P(to_numpy(p.c), to_float(p.a), to_float(p.w))) for p in p1s]
        eps = [r.kport(_P(to_numpy(p.c), to_float(p.a), to_float(p.w))) for p in p2s]
        kf_routes: list[np.ndarray] = []
        if backbone:
            pts_list = aa.backbone2bundle(
                backbone=[kdb.DPoint(to_float(x), to_float(y)) for x, y in backbone],
                port_widths=[p.dwidth for p in sps],
                spacings=[to_float(s) for s in separation],
            )
            for ps, pe, pts in zip(sps, eps, pts_list, strict=False):
                v_start = pts[0] - pts[1]
                v_end = pts[-1] - pts[-2]
                psb = ps.copy()
                psb.dcplx_trans = kdb.DCplxTrans(
                    1, float(np.rad2deg(np.arctan2(v_start.y, v_start.x))), False, pts[0].to_v()
                )
                peb = pe.copy()
                peb.dcplx_trans = kdb.DCplxTrans(
                    1, float(np.rad2deg(np.arctan2(v_end.y, v_end.x))), False, pts[-1].to_v()
                )
                pts_ = aa._get_connection_between_ports(
                    port_start=ps, port_end=psb, bend_factory=r.kf_bend, bend_ports=bend_ports, backbone=pts
                )
                pts_.reverse()
                pts_ = aa._get_connection_between_ports(
                    port_start=pe, port_end=peb, bend_factory=r.kf_bend, backbone=pts_, bend_ports=bend_ports
                )
                pts_.reverse()
                kf_routes.append(np.array([[p.x, p.y] for p in pts_]))
        else:
            for ps, pe in zip(sps, eps, strict=False):
                pts_ = aa._get_connection_between_ports(
                    port_start=ps, port_end=pe, bend_factory=r.kf_bend, bend_ports=bend_ports, backbone=[]
                )
                pts_.append(pe.dcplx_trans.disp.to_p())
                kf_routes.append(np.array([[p.x, p.y] for p in pts_]))
        if not traced:
            return [asarray(kr) for kr in kf_routes]
        traced_routes: list[list[Any]] = []
        if backbone:
            bb = [_vec(x, y) for x, y in backbone]
            bundles = _backbone2bundle(bb, [p.w for p in p1s], separation)
            for ps, pe, pts in zip(p1s, p2s, bundles, strict=False):
                v_start = pts[0] - pts[1]
                v_end = pts[-1] - pts[-2]
                psb = _P(pts[0], xp.rad2deg(xp.arctan2(v_start[1], v_start[0])), ps.w)
                peb = _P(pts[-1], xp.rad2deg(xp.arctan2(v_end[1], v_end[0])), pe.w)
                pts_ = r.connection(ps, psb, list(pts))
                pts_.reverse()
                pts_ = r.connection(pe, peb, pts_)
                pts_.reverse()
                traced_routes.append(pts_)
        else:
            for ps, pe in zip(p1s, p2s, strict=False):
                pts_ = r.connection(ps, pe, [])
                pts_.append(pe.c)
                traced_routes.append(pts_)


    out = []
    for i, (tr, kr) in enumerate(zip(traced_routes, kf_routes, strict=True)):
        t = xp.stack([asarray(p) for p in tr])
        tn = to_numpy(t)
        if tn.shape != kr.shape or np.max(np.abs(tn - kr)) > _POINT_TOLERANCE:
            raise RuntimeError(
                f"Traced all-angle router diverged from kfactory for route {i}: "
                f"{tn.tolist()} != {kr.tolist()}. Please report this as a bug."
            )
        # kfactory's values, traced derivatives
        if is_tracer(t):
            out.append(t + stop_gradient(asarray(kr) - t))
        else:
            out.append(asarray(kr))
    return out


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

    Differentiable with respect to port positions, orientations and widths,
    backbone points and separation.

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
    backbone_pts: list[tuple[Any, Any]] = [
        (p.x, p.y) if hasattr(p, "x") and hasattr(p, "y") else (p[0], p[1])
        for p in (backbone or [])
    ]
    seps: list[Any] = (
        list(separation)
        if isinstance(separation, list | tuple)
        else [separation] * len(ports1)
    )
    pts_list = _backbones(ports1, ports2, backbone_pts, seps, bend_func, bend_ports)
    return [
        place_all_angle_route(
            c,
            p1.width,
            pts,
            straight_func=straight_func,
            bend_func=bend_func,
            bend_ports=bend_ports,
            straight_ports=straight_ports,
        )
        for p1, pts in zip(ports1, pts_list, strict=False)
    ]


__all__ = ["OpticalAllAngleRoute", "place_all_angle_route", "route_bundle_all_angle"]
