"""Gradient tests for the differentiable (JAX) backend.

Every test compares ``jax.grad`` (or ``jax.jvp``) of a scalar built from a layout
with central finite differences of the same function. Grid snapping is disabled
during the checks so that finite differences see the unsnapped geometry (the
gradients themselves are straight-through and do not depend on snapping).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads as jax_check_grads

import gdsfactory as gf
from gdsfactory._jax import to_float
from gdsfactory.component import polygon_area
from gdsfactory.routing._kf_router import PortBox


@pytest.fixture(autouse=True)
def no_snapping() -> Iterator[None]:
    gf.snap.SNAP_ENABLED = False
    yield
    gf.snap.SNAP_ENABLED = True


def summary(c: gf.Component) -> Any:
    """Smooth scalar summary of a layout: areas, vertices, ports and bbox."""
    s: Any = 0.0
    for layer, polys in c.get_polygons_points().items():
        for p in polys:
            s = s + (layer % 5 + 1) * jnp.abs(polygon_area(p))
            s = s + 1e-2 * jnp.sum(0.3 * p[:, 0] + 0.7 * p[:, 1])
    for i, port in enumerate(c.ports):
        s = s + (i + 1) * (0.3 * port.x + 0.7 * port.y)
    b = c.dbbox()
    return s + 0.11 * b.left + 0.13 * b.right + 0.17 * b.top + 0.19 * b.bottom


def check_grad(
    f: Callable[[Any], Any], x0: float, rtol: float = 1e-4, h: float | None = None
) -> None:
    """Compares jax.grad with central differences.

    Routing topology is computed by kfactory on the 1 nm grid, so routing checks
    use steps of several nm (``h``).
    """
    g = float(jax.grad(f)(x0))
    h = h or max(abs(x0), 1.0) * 1e-5
    fd = (to_float(f(x0 + h)) - to_float(f(x0 - h))) / (2 * h)
    assert np.isclose(g, fd, rtol=rtol, atol=1e-6), f"grad={g} fd={fd}"


@pytest.mark.parametrize(
    ("factory", "param", "value"),
    [
        ("straight", "length", 10.0),
        ("straight", "width", 0.5),
        ("bend_circular", "radius", 10.0),
        ("bend_euler", "radius", 10.0),
        ("bend_euler", "p", 0.5),
        ("taper", "length", 10.0),
        ("taper", "width2", 1.0),
        ("mmi1x2", "length_mmi", 5.5),
        ("mmi1x2", "width_mmi", 2.5),
        ("coupler", "length", 20.0),
        ("ring_single", "length_x", 4.0),
        ("mzi", "delta_length", 10.0),
    ],
)
def test_component_gradients(factory: str, param: str, value: float) -> None:
    f = gf.get_active_pdk().cells[factory]
    check_grad(lambda v: summary(f(**{param: v})), value)


def test_reference_connect_and_move_gradients() -> None:
    def f(length: Any) -> Any:
        c = gf.Component()
        s1 = c << gf.components.straight(length=length)
        b = c << gf.components.bend_euler(radius=5)
        b.connect("o1", s1.ports["o2"])
        s2 = c << gf.components.straight(length=2 * length)
        s2.connect("o1", b.ports["o2"])
        s2.rotate(15, center=(0, 0))
        s2.move((length, 0.0))
        return summary(c) + s2.ports["o2"].x * s2.ports["o2"].y

    check_grad(f, 7.0)


def test_port_position_matches_length() -> None:
    def f(length: Any) -> Any:
        return gf.components.straight(length=length).ports["o2"].x

    assert float(jax.grad(f)(3.0)) == pytest.approx(1.0)


def test_route_single_gradients() -> None:
    def f(dy: Any) -> Any:
        c = gf.Component()
        m1 = c << gf.components.mmi1x2()
        m2 = c << gf.components.mmi1x2()
        m2.move((40, dy))
        route = gf.routing.route_single(
            c, m1.ports["o2"], m2.ports["o1"], radius=5, cross_section="strip"
        )
        return summary(c) + route.length * 1e-3

    check_grad(f, 20.0, h=0.01)


@pytest.mark.parametrize("name", ["pitch", "dy", "separation"])
def test_route_bundle_gradients(name: str) -> None:
    def f(v: Any) -> Any:
        kwargs = {"pitch": 10.0, "dy": 200.0, "separation": 5.0}
        kwargs[name] = v
        c = gf.Component()
        p1 = [
            gf.Port(
                f"t{i}", center=(10.0 * i, 0.0), width=0.5, orientation=90, layer=(1, 0)
            )
            for i in range(4)
        ]
        p2 = [
            gf.Port(
                f"b{i}",
                center=(100 + i * kwargs["pitch"], kwargs["dy"]),
                width=0.5,
                orientation=270,
                layer=(1, 0),
            )
            for i in range(4)
        ]
        routes = gf.routing.route_bundle(
            c, p1, p2, cross_section="strip", separation=kwargs["separation"]
        )
        return summary(c) + sum(r.length for r in routes) * 1e-3

    check_grad(f, {"pitch": 10.0, "dy": 200.0, "separation": 5.0}[name], h=0.01)


def _all_angle_route(case: str, v: Any) -> Any:
    c = gf.ComponentAllAngle()
    a = c.add_ref_off_grid(gf.components.straight(length=5))
    b = c.add_ref_off_grid(gf.components.straight(length=5))
    if case == "crossing":  # the port axes cross: no optimization
        b.rotate(30.0)
        b.move((80.0, v))
        routes = gf.routing.route_bundle_all_angle(c, [a.ports["o2"]], [b.ports["o1"]])
    elif case == "optimized":  # kfactory solves for the connection angle
        b.rotate(190.0)
        b.move((80.0, v))
        routes = gf.routing.route_bundle_all_angle(c, [a.ports["o2"]], [b.ports["o2"]])
    else:  # backbone + separation
        a2 = c.add_ref_off_grid(gf.components.straight(length=5))
        a2.move((0, -5))
        b2 = c.add_ref_off_grid(gf.components.straight(length=5))
        b2.move((0, -5))
        for r in (b, b2):
            r.rotate(10.0)
            r.move((140.0, 80.0))
        routes = gf.routing.route_bundle_all_angle(
            c,
            [a.ports["o2"], a2.ports["o2"]],
            [b.ports["o1"], b2.ports["o1"]],
            backbone=[(37.0, -2.5), (54.0, 48.0), (100.0, 64.0)],
            separation=v,
        )
    return sum(r.length_um for r in routes) * 1e-3 + sum(
        jnp.sum(r.backbone_um[1]) for r in routes
    )


@pytest.mark.parametrize(
    ("case", "x0", "rtol"),
    [("crossing", 40.0, 1e-5), ("optimized", 30.0, 1e-3), ("backbone", 3.0, 1e-4)],
)
def test_route_bundle_all_angle_gradients(case: str, x0: float, rtol: float) -> None:
    # "optimized": the value is scipy's bounded minimizer (xatol 1e-5), the
    # derivative its implicit derivative, so FD carries the optimizer's noise.
    check_grad(lambda v: _all_angle_route(case, v), x0, rtol=rtol, h=0.01)


def test_route_bundle_all_angle_length_um() -> None:
    from gdsfactory.routing.route_astar import get_route_bend_count
    from gdsfactory.routing.route_dubins import dubins_path_length, route_dubins
    from gdsfactory.typings import OpticalAllAngleRoute

    def length_fn(dx: Any) -> Any:
        c = gf.ComponentAllAngle()
        a = c.add_ref_off_grid(gf.components.straight(length=5))
        b = c.add_ref_off_grid(gf.components.straight(length=5))
        b.move((5.0 + dx, 0.0))
        routes = gf.routing.route_bundle_all_angle(c, [a.ports["o2"]], [b.ports["o1"]])
        assert isinstance(routes[0], OpticalAllAngleRoute)
        assert routes[0].length == routes[0].length_um
        assert get_route_bend_count(routes[0]) == 100
        return routes[0].length_um

    val, g = jax.value_and_grad(length_fn)(40.0)
    assert float(val) == pytest.approx(40.0, rel=1e-6)
    assert float(g) == pytest.approx(1.0, rel=1e-5)

    def dubins_fn(dy: Any) -> Any:
        c = gf.ComponentAllAngle()
        a = c.add_ref_off_grid(gf.components.straight(length=5))
        b = c.add_ref_off_grid(gf.components.straight(length=5))
        b.rotate(90.0)
        b.move((30.0, dy))
        r = route_dubins(c, a.ports["o2"], b.ports["o1"], cross_section="strip")
        assert isinstance(r, OpticalAllAngleRoute)
        assert get_route_bend_count(r) == 100
        return r.length_um

    check_grad(dubins_fn, 20.0, rtol=1e-4, h=0.01)
    # Curve length must exceed Euclidean chord distance (~sqrt(25^2 + 20^2) = 32.0156)
    assert float(dubins_fn(20.0)) > 33.0
    xs = gf.get_cross_section("strip")
    assert float(dubins_path_length((5.0, 0.0, 0.0), (30.0, 20.0, 90.0), xs)) == pytest.approx(
        float(dubins_fn(20.0)), rel=1e-6
    )


def test_spiral_fixed_length_gradient() -> None:
    def f(length: Any) -> Any:
        c = gf.components.spiral_racetrack_fixed_length(length=length)
        return c.info["straight_length"] + c.xsize * 0.01

    check_grad(f, 1000.0, h=0.01)


def test_forward_mode() -> None:
    def f(radius: Any) -> Any:
        return gf.components.bend_euler(radius=radius).ports["o2"].y

    primal, tangent = jax.jvp(f, (10.0,), (1.0,))
    assert float(primal) == pytest.approx(10.0)
    assert float(tangent) == pytest.approx(1.0, rel=1e-6)


def test_export_with_traced_values(tmp_path: Any) -> None:
    """Components built with tracers can still be exported (concrete values)."""
    paths = []

    def f(length: Any) -> Any:
        c = gf.components.straight(length=length)
        paths.append(c.write_gds(tmp_path / "traced.gds"))
        return c.ports["o2"].x

    jax.grad(f)(4.0)
    c = gf.import_gds(paths[0])
    assert c.dxsize == pytest.approx(4.0)


def test_cache_is_bypassed_when_traced() -> None:
    gf.snap.SNAP_ENABLED = True  # unsnapped cells are never cached
    c1 = gf.components.straight(length=10.0)
    assert gf.components.straight(length=10.0) is c1

    def f(length: Any) -> Any:
        c = gf.components.straight(length=length)
        assert c is not c1
        return c.ports["o2"].x

    jax.grad(f)(10.0)


def _build_primitive(name: str, x: Any) -> gf.Component:
    if name == "straight":
        return gf.components.straight(length=x[0], width=x[1])
    if name == "bend_circular":
        return gf.components.bend_circular(radius=x[0], width=x[1])
    if name == "bend_euler":
        return gf.components.bend_euler(radius=x[0], width=x[1], p=x[2])
    if name == "taper":
        return gf.components.taper(length=x[0], width1=x[1], width2=x[2])
    if name == "extrude_path":
        pts = jnp.array(
            [[0.0, 0.0], [x[0], 0.3 * x[1]], [x[0] + 15.0, x[1]]],
            dtype=x.dtype,
        )
        poly = gf.functions.extrude_path(
            pts, width=x[2], with_manhattan_facing_angles=False
        )
        c = gf.Component()
        c.add_polygon(poly, layer=(1, 0))
        return c
    if name == "smooth":
        pts = jnp.array(
            [[0.0, 0.0], [x[0], 0.0], [x[0], x[1]], [x[0] + 20.0, x[1]]],
            dtype=x.dtype,
        )
        return gf.path.smooth(pts, radius=x[2], bend=gf.path.euler).extrude("strip")
    if name == "bend_s_offset":
        return gf.components.bend_s_offset(offset=x[0], radius=10.0)
    raise ValueError(f"Unknown primitive {name!r}")


_FLOAT32_CASES = [
    ("straight", [10.0, 0.5]),
    ("bend_circular", [10.0, 0.5]),
    ("bend_euler", [10.0, 0.5, 0.5]),
    ("taper", [10.0, 0.5, 2.0]),
    ("extrude_path", [15.0, 8.0, 0.6]),
    ("smooth", [20.0, 20.0, 5.0]),
    ("bend_s_offset", [5.0]),
]


def _assert_float32_grad(
    name: str, init: list[float], ref_grad: Any, *, check_val_dtype: bool
) -> None:
    def _loss(v: Any) -> Any:
        return summary(_build_primitive(name, v))

    x32 = jnp.array(init, dtype=jnp.float32)
    val, grad = jax.value_and_grad(_loss)(x32)
    if check_val_dtype:
        assert jnp.asarray(val).dtype == jnp.float32
    assert grad.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(grad)))
    assert not bool(jnp.any(jnp.isnan(grad)))
    assert np.all(np.abs(np.asarray(grad)) > 1e-4)
    np.testing.assert_allclose(np.asarray(grad), ref_grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(("name", "init"), _FLOAT32_CASES)
def test_float32_gradients(name: str, init: list[float]) -> None:
    def _loss(v: Any) -> Any:
        return summary(_build_primitive(name, v))

    ref_grad = np.asarray(jax.grad(_loss)(jnp.array(init, dtype=jnp.float64)))
    _assert_float32_grad(name, init, ref_grad, check_val_dtype=False)


@pytest.mark.parametrize(("name", "init"), _FLOAT32_CASES)
def test_float32_gradients_x64_disabled(name: str, init: list[float]) -> None:
    def _loss(v: Any) -> Any:
        return summary(_build_primitive(name, v))

    ref_grad = np.asarray(jax.grad(_loss)(jnp.array(init, dtype=jnp.float64)))
    with jax.enable_x64(False):
        _assert_float32_grad(name, init, ref_grad, check_val_dtype=True)


@pytest.mark.parametrize(
    ("name", "init"),
    [
        ("straight", [10.0, 0.5]),
        ("bend_circular", [10.0, 0.5]),
        ("bend_euler", [10.0, 0.5, 0.5]),
        ("taper", [10.0, 0.5, 2.0]),
        ("extrude_path", [15.0, 8.0, 0.6]),
        ("smooth", [20.0, 20.0, 5.0]),
    ],
)
def test_primitives_jacfwd_eq_jacrev_and_check_grads(
    name: str, init: list[float]
) -> None:
    x0 = jnp.array(init, dtype=jnp.float64)

    def _scalar(v: Any) -> Any:
        return summary(_build_primitive(name, v))

    def _vector(v: Any) -> Any:
        c = _build_primitive(name, v)
        return jnp.stack([summary(c), c.dxmax, c.dymax, c.xsize, c.ysize])

    jf = jax.jacfwd(_vector)(x0)
    jr = jax.jacrev(_vector)(x0)
    assert np.all(np.isfinite(np.asarray(jf)))
    assert float(jnp.linalg.norm(jf)) > 0.0
    np.testing.assert_allclose(np.asarray(jf), np.asarray(jr), rtol=1e-7, atol=1e-7)
    jax_check_grads(
        _scalar, (x0,), order=1, modes=("fwd", "rev"), eps=1e-5, rtol=1e-4, atol=1e-4
    )


def _assert_finite_fwd_rev(fn: Callable[[Any], Any], x0: float) -> tuple[float, float]:
    g_rev = jax.grad(fn)(x0)
    g_fwd = jax.jacfwd(fn)(x0)
    assert np.isfinite(float(g_rev))
    assert np.isfinite(float(g_fwd))
    np.testing.assert_allclose(float(g_fwd), float(g_rev), rtol=1e-7, atol=1e-9)
    return float(g_rev), float(g_fwd)


def test_boundary_zero_angle_gradients() -> None:
    def _arc_o2x(a: Any) -> Any:
        return gf.path.arc(radius=10.0, angle=a).extrude("strip").ports["o2"].x

    def _arc_summary_a(a: Any) -> Any:
        return summary(gf.path.arc(radius=10.0, angle=a).extrude("strip"))

    def _arc_summary_r(r: Any) -> Any:
        return summary(gf.path.arc(radius=r, angle=0.0).extrude("strip"))

    g_arc_x, _ = _assert_finite_fwd_rev(_arc_o2x, 0.0)
    assert g_arc_x == pytest.approx(10.0 * np.pi / 180.0, rel=1e-6)
    h_arc = 1e-5
    fd_arc_right = (to_float(_arc_o2x(h_arc)) - to_float(_arc_o2x(0.0))) / h_arc
    assert g_arc_x == pytest.approx(fd_arc_right, rel=1e-4, abs=1e-4)
    _assert_finite_fwd_rev(_arc_summary_a, 0.0)
    _assert_finite_fwd_rev(_arc_summary_r, 10.0)

    for step in (None, 1.0):

        def _euler_o2x(a: Any, *, angular_step: float | None = step) -> Any:
            return gf.components.bend_euler(
                radius=10.0, angle=a, angular_step=angular_step
            ).ports["o2"].x

        def _euler_summary_a(a: Any, *, angular_step: float | None = step) -> Any:
            return summary(
                gf.components.bend_euler(
                    radius=10.0, angle=a, angular_step=angular_step
                )
            )

        def _euler_summary_r(r: Any, *, angular_step: float | None = step) -> Any:
            return summary(
                gf.components.bend_euler(
                    radius=r, angle=0.0, angular_step=angular_step
                )
            )

        g_eu_x, _ = _assert_finite_fwd_rev(_euler_o2x, 0.0)
        _assert_finite_fwd_rev(_euler_summary_a, 0.0)
        _assert_finite_fwd_rev(_euler_summary_r, 10.0)
        h = 1e-5
        fd_right = (to_float(_euler_o2x(h)) - to_float(_euler_o2x(0.0))) / h
        assert g_eu_x == pytest.approx(fd_right, rel=1e-3, abs=1e-4)


def test_boundary_180_angle_gradients() -> None:
    def _smooth_no_bbox(c: gf.Component) -> Any:
        s: Any = 0.0
        for layer, polys in c.get_polygons_points().items():
            for p in polys:
                s = s + (layer % 5 + 1) * jnp.abs(polygon_area(p))
                s = s + 1e-2 * jnp.sum(0.3 * p[:, 0] + 0.7 * p[:, 1])
        for i, port in enumerate(c.ports):
            s = s + (i + 1) * (0.3 * port.x + 0.7 * port.y)
        return s

    def _arc_a(a: Any) -> Any:
        return summary(gf.path.arc(radius=10.0, angle=a, npoints=181).extrude("strip"))

    def _arc_a_smooth(a: Any) -> Any:
        return _smooth_no_bbox(
            gf.path.arc(radius=10.0, angle=a, npoints=181).extrude("strip")
        )

    def _arc_r(r: Any) -> Any:
        return summary(gf.path.arc(radius=r, angle=180.0).extrude("strip"))

    def _arc_r_o2y(r: Any) -> Any:
        return gf.path.arc(radius=r, angle=180.0).extrude("strip").ports["o2"].y

    g_arc_a, _ = _assert_finite_fwd_rev(_arc_a, 180.0)
    assert abs(g_arc_a) > 1e-4
    check_grad(_arc_a_smooth, 180.0)
    _assert_finite_fwd_rev(_arc_r, 10.0)
    check_grad(_arc_r, 10.2)
    assert float(jax.grad(_arc_r_o2y)(10.0)) == pytest.approx(2.0, rel=1e-6)

    for step in (None, 1.0):

        def _euler_a(a: Any, *, angular_step: float | None = step) -> Any:
            return summary(
                gf.components.bend_euler(
                    radius=10.0, angle=a, angular_step=angular_step
                )
            )

        def _euler_a_smooth(a: Any) -> Any:
            return _smooth_no_bbox(
                gf.components.bend_euler(radius=10.0, angle=a, npoints=181)
            )

        def _euler_r(r: Any, *, angular_step: float | None = step) -> Any:
            return summary(
                gf.components.bend_euler(
                    radius=r, angle=180.0, angular_step=angular_step
                )
            )

        def _euler_r_o2y(r: Any, *, angular_step: float | None = step) -> Any:
            return gf.components.bend_euler(
                radius=r, angle=180.0, angular_step=angular_step
            ).ports["o2"].y

        g_ea, _ = _assert_finite_fwd_rev(_euler_a, 180.0)
        g_er, _ = _assert_finite_fwd_rev(_euler_r, 10.0)
        assert abs(g_ea) > 1e-4
        assert abs(g_er) > 1e-4
        check_grad(_euler_a_smooth, 180.0)
        check_grad(_euler_r, 10.2)
        assert float(jax.grad(_euler_r_o2y)(10.0)) == pytest.approx(2.0, rel=1e-5)


def test_smooth_touching_endpoint_and_taper_parabolic() -> None:
    def _touching_smooth(dx: Any) -> Any:
        pts = jnp.array([[0.0, 0.0], [0.0, 10.0], [dx, 10.0]])
        c = gf.path.smooth(pts, radius=10.0, bend=gf.path.arc).extrude("strip")
        return summary(c)

    g_smooth, _ = _assert_finite_fwd_rev(_touching_smooth, 20.0)
    assert abs(g_smooth) > 1e-4
    check_grad(_touching_smooth, 20.0)

    def _parabolic_len(length: Any) -> Any:
        return summary(gf.components.taper_parabolic(length=length))

    def _parabolic_w2(width2: Any) -> Any:
        return summary(gf.components.taper_parabolic(width2=width2))

    def _parabolic_exp(exp: Any) -> Any:
        return summary(gf.components.taper_parabolic(exp=exp))

    check_grad(_parabolic_len, 20.0)
    check_grad(_parabolic_w2, 5.0)
    check_grad(_parabolic_exp, 0.5)


def test_routing_parameter_sweeps() -> None:
    def _single_radius(r: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(60.0, 60.0), width=0.5, orientation=90, layer=(1, 0))
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            radius=r,
            bend=partial(gf.components.bend_euler, npoints=101),
        )
        # Check all unsnapped bend/straight geometry via summary(c); route.length
        # derives from kfactory's integer-DBU area even when SNAP_ENABLED=False.
        return summary(c)

    def _bundle_radius(r: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(60.0, 60.0), width=0.5, orientation=90, layer=(1, 0))
        gf.routing.route_bundle(
            c,
            [p1],
            [p2],
            cross_section="strip",
            radius=r,
            bend=partial(gf.components.bend_euler, npoints=101),
        )
        return summary(c)

    def _single_width_route(w: Any) -> tuple[Any, gf.Component]:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(60.0, 40.0), width=0.5, orientation=180, layer=(1, 0))
        route = gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            route_width=w,
            auto_taper=False,
            allow_width_mismatch=True,
        )
        return route.length, c

    def _bundle_width_route(w: Any) -> tuple[Any, gf.Component]:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(60.0, 40.0), width=0.5, orientation=180, layer=(1, 0))
        routes = gf.routing.route_bundle(
            c,
            [p1],
            [p2],
            cross_section="strip",
            route_width=w,
            auto_taper=False,
            allow_width_mismatch=True,
        )
        return routes[0].length, c

    def _single_width(w: Any) -> Any:
        return summary(_single_width_route(w)[1])

    def _bundle_width(w: Any) -> Any:
        return summary(_bundle_width_route(w)[1])

    def _single_width_length(w: Any) -> Any:
        return _single_width_route(w)[0]

    def _bundle_width_length(w: Any) -> Any:
        return _bundle_width_route(w)[0]

    def _run_starts(s: Any, *, single: bool) -> tuple[Any, gf.Component]:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=90, layer=(1, 0))
        p2 = gf.Port("o2", center=(40.0, 0.0), width=0.5, orientation=90, layer=(1, 0))
        if single:
            route = gf.routing.route_single(
                c, p1, p2, cross_section="strip", radius=5.0, start_straight_length=s
            )
            return route.length, c
        routes = gf.routing.route_bundle(
            c, [p1], [p2], cross_section="strip", radius=5.0, start_straight_length=s
        )
        return routes[0].length, c

    def _single_starts_length(s: Any) -> Any:
        return _run_starts(s, single=True)[0]

    def _bundle_starts_length(s: Any) -> Any:
        return _run_starts(s, single=False)[0]

    def _bundle_starts_summary(s: Any) -> Any:
        length, c = _run_starts(s, single=False)
        return summary(c) + length * 1e-3

    def _bundle_portbox(y_top: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(100.0, 0.0), width=0.5, orientation=180, layer=(1, 0))
        obstacle = PortBox([(30.0, -25.0), (70.0, y_top)])
        routes = gf.routing.route_bundle(
            c,
            [p1],
            [p2],
            cross_section="strip",
            radius=5.0,
            bboxes=[obstacle],
        )
        return summary(c) + routes[0].length * 1e-3

    def _single_sbend_dy(dy: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(50.0, dy), width=0.5, orientation=180, layer=(1, 0))
        gf.routing.route_single_sbend(c, p1, p2, cross_section="strip")
        return summary(c)

    def _single_sbend_start(s: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(60.0, 8.0), width=0.5, orientation=180, layer=(1, 0))
        gf.routing.route_single_sbend(
            c, p1, p2, cross_section="strip", start_straight_length=s
        )
        return summary(c)

    def _single_sbend_width(w: Any) -> Any:
        c = gf.Component()
        xs = gf.CrossSection(
            sections=(
                gf.Section(width=w, offset=0.0, layer=(1, 0), port_names=("o1", "o2")),
            )
        )
        p1 = gf.Port("o1", center=(0.0, 0.0), width=w, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(50.0, 6.0), width=w, orientation=180, layer=(1, 0))
        gf.routing.route_single_sbend(
            c,
            p1,
            p2,
            bend_s=partial(gf.components.bend_s, width=w),
            cross_section=xs,
            allow_width_mismatch=True,
        )
        return summary(c)

    def _bundle_sbend_dy(dy: Any) -> Any:
        c = gf.Component()
        p1s = [
            gf.Port("a1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0)),
            gf.Port("a2", center=(0.0, 20.0), width=0.5, orientation=0, layer=(1, 0)),
        ]
        p2s = [
            gf.Port("b1", center=(60.0, dy), width=0.5, orientation=180, layer=(1, 0)),
            gf.Port(
                "b2", center=(60.0, 20.0 + dy), width=0.5, orientation=180, layer=(1, 0)
            ),
        ]
        gf.routing.route_bundle_sbend(
            c, p1s, p2s, cross_section="strip", auto_taper=False
        )
        return summary(c)

    def _bundle_sbend_start(s: Any) -> Any:
        c = gf.Component()
        p1s = [
            gf.Port("a1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0)),
            gf.Port("a2", center=(0.0, 20.0), width=0.5, orientation=0, layer=(1, 0)),
        ]
        p2s = [
            gf.Port("b1", center=(60.0, 8.0), width=0.5, orientation=180, layer=(1, 0)),
            gf.Port(
                "b2", center=(60.0, 28.0), width=0.5, orientation=180, layer=(1, 0)
            ),
        ]
        gf.routing.route_bundle_sbend(
            c,
            p1s,
            p2s,
            cross_section="strip",
            auto_taper=False,
            start_straight_length=s,
        )
        return summary(c)

    check_grad(_single_radius, 8.0, h=0.01)
    check_grad(_bundle_radius, 8.0, h=0.01)
    check_grad(_single_width, 0.6, h=0.004)
    check_grad(_bundle_width, 0.6, h=0.004)
    assert float(jax.grad(_single_width_length)(0.6)) == pytest.approx(0.0, abs=1e-6)
    assert float(jax.grad(_bundle_width_length)(0.6)) == pytest.approx(0.0, abs=1e-6)
    assert float(jax.grad(_single_starts_length)(20.0)) == pytest.approx(
        2000.0, rel=1e-5
    )
    assert float(jax.grad(_bundle_starts_length)(20.0)) == pytest.approx(
        2000.0, rel=1e-5
    )
    check_grad(_bundle_starts_summary, 20.0, h=0.01)
    check_grad(_bundle_portbox, 12.0, h=0.01)
    check_grad(_single_sbend_dy, 8.0, h=0.01)
    check_grad(_single_sbend_start, 5.0, h=0.01)
    check_grad(_single_sbend_width, 0.6, h=0.005)
    check_grad(_bundle_sbend_dy, 10.0, h=0.01)
    check_grad(_bundle_sbend_start, 5.0, h=0.01)


def _build_egfx_component(name: str, v: Any) -> gf.Component:
    if name == "circle":
        return gf.components.circle(radius=v)
    if name == "ellipse":
        return gf.components.ellipse(radii=(v, 5.0))
    if name == "torus":
        return gf.components.torus(inner_radius=v, outer_radius=10.0)
    if name == "pie_arc":
        return gf.components.pie_arc(radius=v)
    if name == "via":
        return gf.components.via(size=(v, 0.7))
    if name == "coupler_adiabatic":
        return gf.components.coupler_adiabatic(length1=v)
    if name == "delay_snake":
        return gf.components.delay_snake(length=v)
    if name == "delay_snake2":
        return gf.components.delay_snake2(length=v)
    if name == "array":
        return gf.components.array(column_pitch=v)
    if name == "snspd":
        return gf.components.snspd(size=(v, 8.0))
    if name == "ytron_round":
        return gf.components.ytron_round(source_length=v)
    if name == "optimal_step":
        return gf.components.optimal_step(start_width=v, end_width=22.0, num_pts=10)
    if name == "bend_s_offset":
        return gf.components.bend_s_offset(offset=v, radius=10.0)
    if name == "spiral_archimedes":
        return gf.components.spiral_archimedes(width=v)
    if name == "spiral_fermat":
        return gf.components.spiral_fermat(width=v)
    if name == "spiral_logarithmic":
        return gf.components.spiral_logarithmic(width=v)
    if name == "taper_adiabatic":
        return gf.components.taper_adiabatic(width1=v, width2=5.0)
    if name == "taper_parabolic":
        return gf.components.taper_parabolic(length=v)
    raise ValueError(f"Unknown component {name!r}")


@pytest.mark.parametrize(
    ("name", "x0"),
    [
        ("circle", 10.0),
        ("ellipse", 10.0),
        ("torus", 5.0),
        ("pie_arc", 10.0),
        ("via", 0.7),
        ("coupler_adiabatic", 20.0),
        ("delay_snake", 1600.0),
        ("delay_snake2", 1600.0),
        ("array", 150.0),
        ("snspd", 10.0),
        ("ytron_round", 500.0),
        ("optimal_step", 10.0),
        ("bend_s_offset", 5.0),
        ("spiral_archimedes", 1.0),
        ("spiral_fermat", 1.0),
        ("spiral_logarithmic", 0.5),
        ("taper_adiabatic", 0.5),
        ("taper_parabolic", 20.0),
    ],
)
def test_egfx_component_ad_smoke(name: str, x0: float) -> None:
    def _loss(v: Any) -> Any:
        return summary(_build_egfx_component(name, v))

    g = float(jax.grad(_loss)(x0))
    assert np.isfinite(g)
    assert abs(g) > 1e-6
    if name == "optimal_step":
        # Concrete calls keep SciPy's default fminbound xtol=1e-5 for GDS golden parity
        # (~1.9e-3 slope bias on xmax); the traced path solves to xtol=1e-11.
        check_grad(_loss, x0, rtol=3e-3, h=1e-3)
        check_grad(lambda v: jax.jvp(_loss, (v,), (0.0,))[0], x0)
    else:
        check_grad(_loss, x0)
