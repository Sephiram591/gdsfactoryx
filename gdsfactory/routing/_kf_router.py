"""Differentiable manhattan routing on top of kfactory's routers.

kfactory's routers (``route_smart`` + placers) decide the routes (corners,
bundling, tapers, path length matching loops, ...) on concrete values in a
private "mirror" KLayout layout, and their placed instances are rebuilt as
references of the corresponding (possibly traced) gdsfactoryx Components.

Gradients:
    When any input carries a JAX tracer, the backbones are recomputed with
    :mod:`gdsfactory.routing._traced_manhattan`, a port of kfactory's router on
    dual numbers: the routing decisions are kfactory's (checked to the dbu),
    and every corner coordinate is a traced function of the inputs (port
    positions, bend radius, separation, start/end straights, waypoints and
    port bounding boxes). The instances of every route are then re-chained
    with their traced geometry, the straights absorbing the change of segment
    length. Gradients are exact within a routing topology; they are undefined
    where the router switches topology.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from gdsfactory._jax import Array, asarray, is_tracer, jnp, to_float, to_numpy, xp
from gdsfactory._ports import Port
from gdsfactory.transform import Transform

if TYPE_CHECKING:
    import kfactory as kf

    from gdsfactory.component import Component, ComponentReference

DBU = 1e-3

# Debug switch: run (and check against kfactory) the traced router on every route.
_CHECK_TRACED_ROUTER = bool(__import__("os").environ.get("GDSFACTORYX_CHECK_TRACED_ROUTER"))


def _sg(x: Any) -> Any:
    return jax.lax.stop_gradient(x)


def _tangent(x: Any) -> Any:
    """Zero-valued quantity carrying the derivative of x."""
    if is_tracer(x):
        return x - _sg(x)
    return 0.0


# ---------------------------------------------------------------------------
# Route result
# ---------------------------------------------------------------------------
@dataclass
class ManhattanRoute:
    """A placed route (kfactory compatible attributes, lengths in dbu).

    Attributes:
        backbone: backbone corner points (kdb.Point, dbu, concrete).
        backbone_um: (N, 2) differentiable backbone points in um.
        start_port: port of the route at its start (facing the start port).
        end_port: port of the route at its end (facing the end port).
        instances: references placed for this route, from start to end.
        n_bend90: number of 90 degree bends.
        n_taper: number of tapers.
        bend90_radius: bend radius (dbu).
        taper_length: taper length (dbu).
        length_straights: total length of the straights (dbu, differentiable).
        length: route length (dbu, differentiable).
        polygons: polygons placed directly (electrical wires), per layer.
    """

    backbone: list[Any]
    start_port: Port
    end_port: Port
    instances: list[ComponentReference] = field(default_factory=list)
    n_bend90: int = 0
    n_taper: int = 0
    bend90_radius: Any = 0
    taper_length: Any = 0
    length_straights: Any = 0
    length: Any = 0
    backbone_um: Any = None
    polygons: dict[Any, list[Any]] = field(default_factory=dict)

    @property
    def length_backbone(self) -> Any:
        """Length of the backbone in dbu (differentiable)."""
        pts = self.backbone_um
        if pts is None:
            pts = asarray([[p.x * DBU, p.y * DBU] for p in self.backbone])
        d = xp.diff(pts, axis=0)
        return xp.sum(xp.sqrt(xp.sum(d**2, axis=1))) / DBU

    @property
    def length_um(self) -> Any:
        return self.length * DBU


OpticalManhattanRoute = ManhattanRoute


# ---------------------------------------------------------------------------
# Mirror layout
# ---------------------------------------------------------------------------
class _Mirror:
    """Private KLayout layout used to run kfactory's routers on concrete values."""

    _counter = itertools.count()

    def __init__(self) -> None:
        import kfactory as kf

        self.kcl = kf.KCLayout(f"gdsfactoryx_router_{next(self._counter)}")
        self.kcl.layout.dbu = DBU
        self.cache: dict[int, Any] = {}
        self.keep: dict[int, Any] = {}
        self.by_index: dict[int, Any] = {}
        self.tree_cache: dict[int, tuple[Any, Any]] = {}
        self._pool: list[Any] = []

    def layer(self, layer: Any) -> int:
        from gdsfactory.pdk import get_layer_info

        return int(self.kcl.layout.layer(get_layer_info(layer)))

    def export(self, component: Component) -> Any:
        """Concrete DKCell copy of component in the mirror layout.

        Locked (cached) components and their children are exported once and
        reused; unlocked components are re-exported on every call.
        """
        from gdsfactory.klayout_bridge import to_kfactory

        key = id(component)
        if component.locked and key in self.cache and self.keep.get(key) is component:
            return self.cache[key]
        kc = to_kfactory(
            component,
            kcl=self.kcl,
            unique_prefix=f"m{next(self._counter)}_",
            cache=self.tree_cache,
        )
        if component.locked:
            self.cache[key] = kc
            self.keep[key] = component
        self.by_index[kc.cell_index()] = component
        return kc

    def new_cell(self, name: str = "route") -> Any:
        import kfactory as kf

        return kf.DKCell(name=f"{name}_{next(self._counter)}", kcl=self.kcl)

    @contextlib.contextmanager
    def scratch(self) -> Iterator[Any]:
        """A cleared scratch cell, returned to a pool afterwards.

        Deleting cells makes kfactory rebuild its cell index, which made every
        routing call O(number of cells); reusing cleared cells avoids that.
        Nested routing calls (e.g. from a factory) get their own cell.
        """
        cell = self._pool.pop() if self._pool else self.new_cell()
        try:
            yield cell
        finally:
            cell.kdb_cell.clear()
            cell.ports.clear()
            self._pool.append(cell)

    def port(self, cell: Any, p: Port, name: str | None = None) -> Any:
        import klayout.db as kdb

        orientation = to_float(p.orientation) if p.orientation is not None else 0.0
        return cell.create_port(
            name=name or p.name or "p",
            dcplx_trans=kdb.DCplxTrans(
                1, round(orientation, 9) % 360, False, to_float(p.x), to_float(p.y)
            ),
            width=round(to_float(p.width) / DBU) * DBU,
            layer=self.layer(p.layer),
            port_type=p.port_type,
        )


_MIRROR: _Mirror | None = None


def mirror() -> _Mirror:
    global _MIRROR
    if _MIRROR is None:
        _MIRROR = _Mirror()
    return _MIRROR


# ---------------------------------------------------------------------------
# kfactory hook
# ---------------------------------------------------------------------------
@dataclass
class _Record:
    pts: list[Any]
    start: Any
    end: Any


class _Hook:
    """Wraps kfactory's generic route_bundle to record the placed backbones."""

    def __init__(self) -> None:
        self.records: list[_Record] = []
        self.start_ports: list[Any] = []
        self.end_ports: list[Any] = []
        self.bend90_cell: Any = None

    @contextlib.contextmanager
    def installed(self) -> Iterator[_Hook]:
        import kfactory.routing.electrical as kfe
        import kfactory.routing.optical as kfo

        originals = (kfo.route_bundle_generic, kfe.route_bundle_generic)
        hook = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            placer = kwargs["placer_function"]

            def recording_placer(c: Any, p1: Any, p2: Any, pts: Any, **kw: Any) -> Any:
                hook.records.append(_Record(list(pts), p1, p2))
                return placer(c, p1, p2, pts, **kw)

            kwargs["placer_function"] = recording_placer
            return originals[0](*args, **kwargs)

        kfo.route_bundle_generic = wrapped
        kfe.route_bundle_generic = wrapped
        try:
            yield self
        finally:
            kfo.route_bundle_generic, kfe.route_bundle_generic = originals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def route_bundle_kf(
    component: Component,
    ports1: Sequence[Port],
    ports2: Sequence[Port],
    *,
    router: str = "optical",
    separation: Any = 3.0,
    bend90: Component | None = None,
    straight_factory: Callable[..., Component] | None = None,
    taper: Component | None = None,
    sbend_factory: Callable[..., Component] | None = None,
    starts: Any = None,
    ends: Any = None,
    waypoints: Any = None,
    route_width: Any = None,
    place_layer: Any = None,
    radius: Any = None,
    on_collision: Any = None,
    on_placer_error: Any = None,
    collision_check_layers: Any = None,
    bboxes: Any = None,
    obstacles: Component | None = None,
    **kf_kwargs: Any,
) -> list[ManhattanRoute]:
    """Routes ports1 -> ports2 with kfactory and places differentiable references.

    Args:
        component: component to place the routes into.
        ports1: start ports.
        ports2: end ports.
        router: "optical" (bends + straights) or "electrical" (wires).
        separation: center to center separation (um).
        bend90: 90 degree bend component (optical).
        straight_factory: ``f(width=..., length=...) -> Component`` (optical).
        taper: optional taper component placed on long straights.
        sbend_factory: ``f(offset=..., length=..., width=...) -> Component``.
        starts: minimal straight length after the start ports (um, list or Steps).
        ends: minimal straight length before the end ports.
        waypoints: list of (x, y) points.
        route_width: route width (um).
        place_layer: layer of electrical wires.
        radius: traced radius of the bend (only used for derivatives).
        on_collision: kfactory collision handling.
        on_placer_error: kfactory placer error handling.
        collision_check_layers: layers to check collisions on.
        bboxes: list of boxes (um) to route around.
        obstacles: component whose geometry is exported for collision checks.
        kf_kwargs: forwarded to kfactory's route_bundle.
    """
    import kfactory as kf
    import klayout.db as kdb

    m = mirror()
    ports1 = list(ports1)
    ports2 = list(ports2)
    if not ports1:
        return []

    # ---------------------------------------------------------------- inputs
    def port_dict(p: Port) -> dict[str, Any]:
        return {
            "name": p.name,
            "x": to_float(p.x),
            "y": to_float(p.y),
            "orientation": to_float(p.orientation) if p.orientation is not None else 0.0,
            "width": to_float(p.width),
            "layer": p.layer,
            "port_type": p.port_type,
        }

    def num(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, list | tuple):
            return [to_float(x) if _is_number(x) else x for x in v]
        return to_float(v) if _is_number(v) else v

    base = {
        "start_ports": [port_dict(p) for p in ports1],
        "end_ports": [port_dict(p) for p in ports2],
        "separation": to_float(separation),
        "starts": num(starts),
        "ends": num(ends),
        "waypoints": None
        if waypoints is None
        else [[to_float(p[0]), to_float(p[1])] for p in waypoints],
        "route_width": num(route_width),
    }

    traced = (
        _CHECK_TRACED_ROUTER
        or any(_port_traced(p) for p in (*ports1, *ports2))
        or _has_tracers_any(separation, starts, ends, waypoints, route_width)
        or any(isinstance(b, PortBox) and b.traced for b in (bboxes or []))
        or _has_traced_geometry(bend90, taper)
    )

    # --------------------------------------------------- concrete components
    bend90_concrete = _concrete_component(bend90) if bend90 is not None else None
    taper_concrete = _concrete_component(taper) if taper is not None else None
    straight_cells: dict[int, tuple[float, float]] = {}
    sbend_cells: dict[int, tuple[float, float, float]] = {}
    special: dict[str, int] = {}

    def kf_straight(width: float, length: float) -> Any:
        assert straight_factory is not None
        comp = straight_factory(width=width, length=length)
        kc = m.export(comp)
        straight_cells[kc.cell_index()] = (width, length)
        return kc

    def run(cell: Any, kwargs: dict[str, Any], with_obstacles: bool) -> tuple[Any, _Hook]:
        if with_obstacles:
            cell.create_inst(m.export(obstacles))
        sp = [
            m.port(cell, _port_from_dict(d), name=f"s{i}")
            for i, d in enumerate(kwargs["start_ports"])
        ]
        ep = [
            m.port(cell, _port_from_dict(d), name=f"e{i}")
            for i, d in enumerate(kwargs["end_ports"])
        ]
        wps = None
        if kwargs["waypoints"] is not None:
            wps = [kdb.DPoint(x, y) for x, y in kwargs["waypoints"]]
        hook = _Hook()
        hook.start_ports = sp
        hook.end_ports = ep
        common = dict(
            separation=kwargs["separation"],
            starts=kwargs["starts"],
            ends=kwargs["ends"],
            waypoints=wps,
            route_width=kwargs["route_width"],
            bboxes=[_dbox(b.dbox if isinstance(b, PortBox) else b) for b in (bboxes or [])],
            on_collision=on_collision_run,
            on_placer_error=on_placer_error,
            collision_check_layers=None
            if not collision_check_layers
            else [m.kcl.layout.get_info(m.layer(lay)) for lay in collision_check_layers],
            **kf_kwargs,
        )
        with hook.installed():
            if router == "optical":
                assert bend90_concrete is not None
                kb = m.export(bend90_concrete)
                kt = m.export(taper_concrete) if taper_concrete is not None else None
                special["bend"] = kb.cell_index()
                if kt is not None:
                    special["taper"] = kt.cell_index()
                hook.bend90_cell = kb
                sb = None
                if sbend_factory is not None:

                    def sb(c: Any, offset: float, length: float, width: float) -> Any:
                        comp = sbend_factory(offset=offset, length=length, width=width)
                        kc = m.export(comp)
                        sbend_cells[kc.cell_index()] = (offset, length, width)
                        inst = c << kc
                        return kf.DInstanceGroup(insts=[inst], ports=list(inst.ports))

                routes = kf.routing.optical.route_bundle(
                    cell,
                    sp,
                    ep,
                    straight_factory=kf_straight,
                    bend90_cell=kb,
                    taper_cell=kt,
                    sbend_factory=sb,
                    **common,
                )
            else:
                layer_info = (
                    m.kcl.layout.get_info(m.layer(place_layer)) if place_layer is not None else None
                )
                routes = kf.routing.electrical.route_bundle(
                    cell, sp, ep, place_layer=layer_info, **common
                )
        return routes, hook

    # The rest of the layout only matters for displaying a collision
    # ("show_error"): check without it, and export it only if a collision
    # is found (kfactory's check itself looks at the new routes only).
    on_collision_run = on_collision
    lazy_obstacles = obstacles is not None and on_collision is not None
    if lazy_obstacles:
        on_collision_run = "error"
    with m.scratch() as cell:
        try:
            routes_kf, hook = run(cell, base, with_obstacles=False)
        except RuntimeError as e:
            if not (
                lazy_obstacles
                and on_collision == "show_error"
                and "collision" in str(e).lower()
            ):
                raise
            cell.kdb_cell.clear()
            cell.ports.clear()
            on_collision_run = on_collision
            run(cell, base, with_obstacles=True)  # shows the error and raises
            raise
        # ------------------------------------------------------------ backbones
        base_bbs = [np.asarray([[p.x * DBU, p.y * DBU] for p in r.pts], dtype=float) for r in hook.records]
        route_records = list(hook.records)
        traced_bbs: list[Array] | None = None
        if traced:
            traced_bbs = _traced_backbones(
                hook,
                ports1,
                ports2,
                router=router,
                bend90=bend90,
                separation=separation,
                starts=starts,
                ends=ends,
                waypoints=waypoints,
                route_width=route_width,
                bboxes=bboxes,
                sbend=sbend_factory is not None,
                kf_kwargs=kf_kwargs,
            )

        def traced_backbone(i: int) -> Array:
            return traced_bbs[i] if traced_bbs is not None else asarray(base_bbs[i])

        # ------------------------------------------------------- rebuild routes
        out: list[ManhattanRoute] = []
        start_lookup = {_key(p): p for p in ports1}
        end_lookup = {_key(p): p for p in ports2}
        for i, (rk, rec) in enumerate(zip(routes_kf, route_records, strict=False)):
            bb = traced_backbone(i)
            p_start = start_lookup.get(_kkey(rec.start)) or ports1[min(i, len(ports1) - 1)]
            p_end = end_lookup.get(_kkey(rec.end)) or ports2[min(i, len(ports2) - 1)]
            if router == "optical":
                route = _rebuild_optical(
                    component,
                    rk,
                    bb,
                    p_start,
                    p_end,
                    bend90=bend90,
                    taper=taper,
                    straight_factory=straight_factory,
                    straight_cells=straight_cells,
                    sbend_cells=sbend_cells,
                    sbend_factory=sbend_factory,
                    traced=traced,
                    route_width=route_width,
                    special=special,
                )
            else:
                route = _rebuild_electrical(
                    component, rk, bb, p_start, p_end, route_width=route_width, place_layer=place_layer
                )
            out.append(route)
        return out


def place_manhattan_kf(
    component: Component,
    p1: Port,
    p2: Port,
    pts: Sequence[Any],
    *,
    straight_factory: Callable[..., Component],
    bend90: Component,
    port_type: str = "optical",
    allow_width_mismatch: bool = False,
    route_width: Any = None,
    radius: Any = None,
) -> ManhattanRoute:
    """Places a route along explicit backbone points (start, corners..., end).

    The topology is placed by kfactory's ``place_manhattan`` on concrete values,
    the backbone points themselves are used as (traced) corners.
    """
    import klayout.db as kdb
    from kfactory.routing.optical import place_manhattan

    m = mirror()
    with m.scratch() as cell:
        kp1 = m.port(cell, p1, name="s0")
        kp2 = m.port(cell, p2, name="e0")
        bend90_concrete = _concrete_component(bend90)
        kb = m.export(bend90_concrete)
        straight_cells: dict[int, tuple[float, float]] = {}

        def kf_straight(width: int, length: int, **kw: Any) -> Any:
            comp = straight_factory(width=width * DBU, length=length * DBU)
            kc = m.export(comp)
            straight_cells[kc.cell_index()] = (width * DBU, length * DBU)
            return m.kcl[kc.cell_index()]

        pts_dbu = [kdb.Point(round(to_float(x) / DBU), round(to_float(y) / DBU)) for x, y in pts]
        rk = place_manhattan(
            m.kcl[cell.cell_index()],
            p1=m.kcl[cell.cell_index()].ports[kp1.name],
            p2=m.kcl[cell.cell_index()].ports[kp2.name],
            straight_factory=kf_straight,
            bend90_cell=m.kcl[kb.cell_index()],
            pts=pts_dbu,
            port_type=port_type,
            allow_width_mismatch=allow_width_mismatch,
            route_width=round(to_float(route_width) / DBU) if route_width is not None else None,
        )
        backbone = xp.stack([xp.stack([asarray(x), asarray(y)]) for x, y in pts])
        traced = is_tracer(backbone) or _has_traced_geometry(bend90) or is_tracer(p1.center_array) or is_tracer(p2.center_array)
        route = _rebuild_optical(
            component,
            rk,
            backbone,
            p1,
            p2,
            bend90=bend90,
            taper=None,
            straight_factory=straight_factory,
            straight_cells=straight_cells,
            sbend_cells={},
            sbend_factory=None,
            traced=traced,
            route_width=route_width,
            special={"bend": kb.cell_index()},
        )
        return route


def to_kf_port(p: Port) -> Any:
    """Concrete integer kfactory Port (in the private mirror layout)."""
    m = mirror()
    cell = m.new_cell("port")
    dp = m.port(cell, p)
    return m.kcl[cell.cell_index()].ports[dp.name]


def _dbox(b: Any) -> Any:
    """kdb.DBox from a DBox, Box, (l, b, r, t) tuple or ((l, b), (r, t))."""
    import klayout.db as kdb

    if isinstance(b, kdb.DBox):
        return b
    if isinstance(b, kdb.Box):
        return b.to_dtype(DBU)
    if all(hasattr(b, a) for a in ("left", "bottom", "right", "top")):
        return kdb.DBox(to_float(b.left), to_float(b.bottom), to_float(b.right), to_float(b.top))
    vals = list(b)
    if len(vals) == 2:
        (x0, y0), (x1, y1) = vals
        return kdb.DBox(to_float(x0), to_float(y0), to_float(x1), to_float(y1))
    return kdb.DBox(*map(to_float, vals))


class PortBox:
    """Bounding box of (possibly traced) points, used as a routing obstacle.

    kfactory sees the concrete ``kdb.DBox``; the traced router sees a dual box
    whose corners carry the derivatives of the points.
    """

    def __init__(self, points: Sequence[Any]) -> None:
        import klayout.db as kdb

        self.points = [(p[0], p[1]) for p in points]
        self.dbox = kdb.DBox()
        for x, y in self.points:
            self.dbox += kdb.DPoint(to_float(x), to_float(y))

    @property
    def traced(self) -> bool:
        return any(is_tracer(c) for p in self.points for c in p)

    def dual(self) -> Any:
        from gdsfactory.routing._dual import Box, Point
        from gdsfactory.routing._traced_router import to_dbu

        b = Box()
        for x, y in self.points:
            b += Point(to_dbu(x), to_dbu(y))
        return b


def _port_traced(p: Port) -> bool:
    return any(is_tracer(v) for v in (p.x, p.y, p.width))


def _has_tracers_any(*values: Any) -> bool:
    from gdsfactory._jax import has_tracers

    return any(has_tracers(v) for v in values if v is not None)


def _dual_port(kp: Any, p: Port) -> Any:
    """Dual Trans of a kfactory port (exact kfactory integers + traced position)."""
    from gdsfactory.routing._dual import DNum, Trans

    t = kp.trans
    x = DNum.traced(p.x / DBU if is_tracer(p.x) else None, t.disp.x)
    y = DNum.traced(p.y / DBU if is_tracer(p.y) else None, t.disp.y)
    return Trans(t.angle, t.is_mirror(), x, y)


def _traced_bend_radius(bend90: Component | None, kb: Any, port_type: str) -> Any:
    """kfactory's get_radius of the bend, with the derivative of the traced bend."""
    from kfactory.routing.generic import get_radius

    from gdsfactory.routing._dual import DNum

    if bend90 is None or kb is None:
        return DNum(0)
    v = get_radius(kb.ports.filter(port_type=port_type))
    ports = [p for p in bend90.ports if p.port_type == port_type]
    if not bend90.has_tracers() or len(ports) != 2:
        return DNum(v)
    p1, p2 = ports
    if manhattan_index(p1) == manhattan_index(p2):
        r = xp.sqrt(xp.sum((p1.center_array - p2.center_array) ** 2)) / DBU
    else:
        c = _virtual_corner(p1, p2)
        r1 = xp.sqrt(xp.sum((p1.center_array - c) ** 2))
        r2 = xp.sqrt(xp.sum((p2.center_array - c) ** 2))
        r = (r1 if to_float(r1) >= to_float(r2) else r2) / DBU
    return DNum.traced(r, v)


def manhattan_index(p: Port) -> int:
    return round(to_float(p.orientation) / 90) % 4


def _traced_backbones(
    hook: _Hook,
    ports1: list[Port],
    ports2: list[Port],
    *,
    router: str,
    bend90: Component | None,
    separation: Any,
    starts: Any,
    ends: Any,
    waypoints: Any,
    route_width: Any,
    bboxes: Any,
    sbend: bool,
    kf_kwargs: dict[str, Any],
) -> list[Array]:
    """Backbones from the traced port of kfactory's router (checked against kfactory)."""
    from gdsfactory.routing._dual import Box, DNum, Point
    from gdsfactory.routing._traced_router import to_dbu, traced_route_bundle

    port_type = kf_kwargs.get("place_port_type", "optical")
    start_ts = [_dual_port(kp, p) for kp, p in zip(hook.start_ports, ports1, strict=True)]
    end_ts = [_dual_port(kp, p) for kp, p in zip(hook.end_ports, ports2, strict=True)]
    widths = [
        DNum.traced(p.width / DBU if is_tracer(p.width) else None, kp.width)
        for kp, p in zip(hook.start_ports, ports1, strict=True)
    ]
    if router == "optical":
        bend90_radius = _traced_bend_radius(bend90, getattr(hook, "bend90_cell", None), port_type)
    else:
        bend90_radius = DNum(0)
    wps = None
    if waypoints is not None:
        wps = [Point(to_dbu(x), to_dbu(y)) for x, y in waypoints]
    dual_boxes = []
    for b in bboxes or []:
        if isinstance(b, PortBox):
            dual_boxes.append(b.dual())
        else:
            dual_boxes.append(Box(_dbox(b).to_itype(DBU)))
    routers = traced_route_bundle(
        start_ts,
        end_ts,
        widths,
        separation=separation,
        bend90_radius=bend90_radius,
        starts=starts,
        ends=ends,
        route_width=route_width,
        sort_ports=kf_kwargs.get("sort_ports", False),
        bbox_routing=kf_kwargs.get("bbox_routing", "minimal"),
        bboxes=dual_boxes,
        waypoints=wps,
        start_angles=kf_kwargs.get("start_angles"),
        end_angles=kf_kwargs.get("end_angles"),
        allow_sbend=sbend,
        constraints=kf_kwargs.get("constraints"),
    )
    out: list[Array] = []
    for i, (r, rec) in enumerate(zip(routers, hook.records, strict=True)):
        pts = r.start.pts
        concrete = [(int(p.x.v), int(p.y.v)) for p in pts]
        expected = [(p.x, p.y) for p in rec.pts]
        if concrete != expected:
            raise RuntimeError(
                "Traced manhattan router diverged from kfactory for route "
                f"{i}: {concrete} != {expected}. Please report this as a bug."
            )
        out.append(
            xp.stack(
                [xp.stack([asarray(p.x.tv()) * DBU, asarray(p.y.tv()) * DBU]) for p in pts]
            )
        )
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) or is_tracer(v) or type(v).__module__.startswith("jax")


def _port_from_dict(d: dict[str, Any]) -> Port:
    return Port(
        name=d["name"],
        center=(d["x"], d["y"]),
        width=d["width"],
        orientation=d["orientation"],
        layer=d["layer"],
        port_type=d["port_type"],
    )


def _key(p: Port) -> tuple[int, int]:
    return (round(to_float(p.x) / DBU), round(to_float(p.y) / DBU))


def _kkey(p: Any) -> tuple[int, int]:
    t = p.dcplx_trans
    return (round(t.disp.x / DBU), round(t.disp.y / DBU))


def _has_traced_geometry(*components: Component | None) -> bool:
    return any(c is not None and c.has_tracers() for c in components)


_CONCRETE_CACHE: dict[int, tuple[Any, Any]] = {}


def _concrete_component(c: Component) -> Component:
    """Concrete (tracer-free) version of a component, for the kfactory mirror."""
    if not c.has_tracers():
        return c
    key = id(c)
    if key in _CONCRETE_CACHE and _CONCRETE_CACHE[key][0] is c:
        return _CONCRETE_CACHE[key][1]
    cc = _strip(c, {})
    _CONCRETE_CACHE[key] = (c, cc)
    return cc


def _strip(c: Component, memo: dict[int, Component]) -> Component:
    from gdsfactory.component import Component as _C
    from gdsfactory.component import ComponentReference

    if id(c) in memo:
        return memo[id(c)]
    n = _C(name=c.name)
    n.polygons = {k: [asarray(to_numpy(p)) for p in v] for k, v in c.polygons.items()}
    n.labels = list(c.labels)
    for r in c.insts:
        t = r.transform
        ref = ComponentReference(
            _strip(r.cell, memo),
            Transform(to_float(t.x), to_float(t.y), to_float(t.rotation), t.mirror, to_float(t.magnification)),
            r._name,
            n,
            r.na,
            r.nb,
            asarray(to_numpy(r.a)),
            asarray(to_numpy(r.b)),
        )
        n.insts.append(ref)
    for p in c.ports:
        q = p.copy()
        q.center = asarray(to_numpy(p.center_array))
        q.orientation = to_float(p.orientation) if p.orientation is not None else None
        q.width = to_float(p.width)
        n.ports.append(q)
    n.info = c.info
    n.settings = c.settings
    n.locked = True
    memo[id(c)] = n
    return n


# ---------------------------------------------------------------------------
# rebuilding
# ---------------------------------------------------------------------------
def _local_ports(cell: Component, port_type: str | None = None) -> list[Port]:
    ports = [p for p in cell.ports if p.orientation is not None]
    if port_type is not None:
        typed = [p for p in ports if p.port_type == port_type]
        if len(typed) >= 2:
            ports = typed
    return ports


def _virtual_corner(a: Port, b: Port) -> Array:
    """Intersection of the axes of two (perpendicular) ports, local coordinates."""
    da = a.direction
    db = b.direction
    # solve a + s*da = b + t*db
    mat = xp.stack([da, -db], axis=1)
    rhs = b.center_array - a.center_array
    st = xp.linalg.solve(mat, rhs)
    return a.center_array + st[0] * da


def _rebuild_optical(
    component: Component,
    rk: Any,
    backbone: Array,
    p_start: Port,
    p_end: Port,
    *,
    bend90: Component | None,
    taper: Component | None,
    straight_factory: Callable[..., Component] | None,
    straight_cells: dict[int, tuple[float, float]],
    sbend_cells: dict[int, tuple[float, float, float]],
    sbend_factory: Callable[..., Component] | None,
    traced: bool,
    route_width: Any,
    special: dict[str, int] | None = None,
) -> ManhattanRoute:
    """Places our references for a kfactory route; adds tangents if traced."""
    special = special or {}
    m = mirror()
    elements: list[dict[str, Any]] = []
    for inst in rk.instances:
        ci = inst.cell.cell_index()
        conc_t = Transform.from_klayout(inst.dcplx_trans)
        if ci in straight_cells:
            width, length = straight_cells[ci]
            elements.append(dict(kind="straight", t=conc_t, width=width, length=length))
        elif bend90 is not None and ci == special.get("bend"):
            elements.append(dict(kind="rigid", t=conc_t, comp=bend90, bend=True))
        elif taper is not None and ci == special.get("taper"):
            elements.append(dict(kind="rigid", t=conc_t, comp=taper, bend=False))
        elif ci in sbend_cells:
            offset, length, width = sbend_cells[ci]
            assert sbend_factory is not None
            elements.append(dict(kind="rigid", t=conc_t, comp=m.by_index[ci], bend=False))
        elif ci in m.by_index:
            elements.append(dict(kind="rigid", t=conc_t, comp=m.by_index[ci], bend=False))
        else:
            from gdsfactory.klayout_bridge import from_kfactory

            elements.append(dict(kind="rigid", t=conc_t, comp=from_kfactory(inst.cell), bend=False))

    width_traced = route_width if is_tracer(route_width) else None

    # concrete straight components
    for e in elements:
        if e["kind"] == "straight":
            assert straight_factory is not None
            e["comp"] = straight_factory(width=e["width"], length=e["length"])

    if traced:
        _add_tangents(elements, backbone, p_start, p_end, straight_factory, width_traced, rk)

    refs = []
    for e in elements:
        ref = component.add_ref(e["comp"])
        ref.transform = e["t"]
        refs.append(ref)

    length_kf = float(rk.length)
    length_straights_kf = float(rk.length_straights)
    length = length_kf
    length_straights = length_straights_kf
    if traced:
        dl = 0.0
        dls = 0.0
        for e in elements:
            if e["kind"] == "straight":
                dls = dls + _tangent(e["comp"].info.get("length", 0.0))
            else:
                dl = dl + _tangent(e["comp"].info.get("length", 0.0))
        length = length_kf + (dl + dls) / DBU
        length_straights = length_straights_kf + dls / DBU

    sp = _port_from_kport(rk.start_port)
    ep = _port_from_kport(rk.end_port)
    return ManhattanRoute(
        backbone=list(rk.backbone),
        backbone_um=backbone,
        start_port=sp,
        end_port=ep,
        instances=refs,
        n_bend90=rk.n_bend90,
        n_taper=rk.n_taper,
        bend90_radius=rk.bend90_radius,
        taper_length=rk.taper_length,
        length_straights=length_straights,
        length=length,
    )


def _port_from_kport(p: Any) -> Port:
    from gdsfactory.pdk import get_layer

    t = p.dcplx_trans
    li = p.layer_info
    width = p.width * DBU if isinstance(p.width, int) else p.width
    return Port(
        name=p.name,
        center=(t.disp.x, t.disp.y),
        width=width,
        orientation=t.angle,
        layer=get_layer((li.layer, li.datatype)),
        port_type=p.port_type,
    )


def _add_tangents(
    elements: list[dict[str, Any]],
    backbone: Array,
    p_start: Port,
    p_end: Port,
    straight_factory: Callable[..., Component] | None,
    width_traced: Any,
    rk: Any,
) -> None:
    """Chains the elements with traced geometry and adds tangents to transforms/lengths."""
    port_type = p_start.port_type
    # local in/out ports (traced) and concrete transformed positions
    cur = p_start.center_array  # traced
    cur_conc = to_numpy(p_start.center_array)
    corners = backbone  # traced (N, 2)
    corners_np = to_numpy(corners)
    seg_index = 0

    # group elements per segment: bends terminate a segment
    segments: list[list[dict[str, Any]]] = [[]]
    for e in elements:
        segments[-1].append(e)
        if e.get("bend"):
            segments.append([])

    for s_idx, seg in enumerate(segments):
        if not seg and s_idx == len(segments) - 1:
            break
        # direction of this segment (concrete)
        a = corners_np[seg_index]
        b = corners_np[min(seg_index + 1, len(corners_np) - 1)]
        u_np = b - a
        nrm = np.linalg.norm(u_np)
        u = asarray(u_np / nrm) if nrm > 0 else asarray([1.0, 0.0])

        # identify in/out local ports for each element (concrete matching)
        for e in seg:
            comp = e["comp"]
            lp = _local_ports(comp, port_type)
            t: Transform = e["t"]
            pos = [to_numpy(t.apply(q.center_array)) for q in lp]
            d = [np.linalg.norm(pp - cur_conc) for pp in pos]
            i_in = int(np.argmin(d))
            others = [k for k in range(len(lp)) if k != i_in]
            i_out = max(others, key=lambda k: np.linalg.norm(pos[k] - pos[i_in])) if others else i_in
            e["in"] = lp[i_in]
            e["out"] = lp[i_out]
            cur_conc = pos[i_out]

        # target position for the end of this segment
        last = seg[-1] if seg else None
        if last is not None and last.get("bend"):
            v_local = _virtual_corner(last["in"], last["out"])
            t = last["t"]
            corner_tr = corners[seg_index + 1]
            target = corner_tr + t.apply_vector(last["in"].center_array - v_local)
            body = seg[:-1]
        else:
            target = p_end.center_array
            body = seg

        # traced lengths of rigid elements along u; concrete straight lengths
        rigid_len = 0.0
        for e in body:
            if e["kind"] == "rigid":
                rigid_len = rigid_len + xp.dot(e["t"].apply_vector(e["out"].center_array - e["in"].center_array), u)
        straights = [e for e in body if e["kind"] == "straight"]
        conc_len = sum(e["length"] for e in straights)
        required = xp.dot(target - cur, u)
        residual = required - rigid_len - conc_len
        residual = residual - _sg(residual)  # tangent only
        if straights:
            longest = max(straights, key=lambda e: e["length"])
            for e in straights:
                extra = residual if e is longest else 0.0
                length_tr = e["length"] + extra
                width_tr = width_traced if width_traced is not None else e["width"]
                if is_tracer(length_tr) or is_tracer(width_tr):
                    assert straight_factory is not None
                    e["comp"] = straight_factory(width=width_tr, length=length_tr)
                    lp = _local_ports(e["comp"], port_type)
                    # keep in/out mapping by name
                    e["in"] = next(q for q in lp if q.name == e["in"].name)
                    e["out"] = next(q for q in lp if q.name == e["out"].name)

        # chain placement with traced geometry
        for e in seg:
            t: Transform = e["t"]
            trans_tr = cur - t.apply_vector(e["in"].center_array)
            nt = t.copy()
            nt.x = t.x + _tangent(trans_tr[0])
            nt.y = t.y + _tangent(trans_tr[1])
            e["t"] = nt
            cur = trans_tr + t.apply_vector(e["out"].center_array)
        seg_index += 1


def _rebuild_electrical(
    component: Component,
    rk: Any,
    backbone: Array,
    p_start: Port,
    p_end: Port,
    route_width: Any,
    place_layer: Any,
) -> ManhattanRoute:
    """Draws the wire of an electrical route as a (differentiable) polygon."""
    from gdsfactory.path import Path

    width = route_width if route_width is not None else p_start.width
    pts = backbone
    if pts.shape[0] >= 2:
        path = Path(pts)
        w = asarray(width)
        p1 = path.centerpoint_offset_curve(pts, w / 2, path.start_angle, path.end_angle)
        p2 = path.centerpoint_offset_curve(pts, -w / 2, path.start_angle, path.end_angle)
        poly = xp.concatenate([p1, p2[::-1]])
        layer = place_layer if place_layer is not None else p_start.layer
        component.add_polygon(poly, layer=layer)
    sp = _port_from_kport(rk.start_port)
    ep = _port_from_kport(rk.end_port)
    d = xp.diff(pts, axis=0)
    length = xp.sum(xp.sqrt(xp.sum(d**2, axis=1))) / DBU
    return ManhattanRoute(
        backbone=list(rk.backbone),
        backbone_um=pts,
        start_port=sp,
        end_port=ep,
        instances=[],
        length=length,
        length_straights=length,
    )


__all__ = [
    "ManhattanRoute",
    "OpticalManhattanRoute",
    "PortBox",
    "mirror",
    "place_manhattan_kf",
    "route_bundle_kf",
    "to_kf_port",
]
