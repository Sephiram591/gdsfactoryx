"""Gradient tests for the differentiable (JAX) backend.

Every test compares ``jax.grad`` (or ``jax.jvp``) of a scalar built from a layout
with central finite differences of the same function. Grid snapping is disabled
during the checks so that finite differences see the unsnapped geometry (the
gradients themselves are straight-through and do not depend on snapping).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory._jax import to_float
from gdsfactory.component import polygon_area


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
            gf.Port(f"t{i}", center=(10.0 * i, 0.0), width=0.5, orientation=90, layer=(1, 0))
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
    return sum(r.length for r in routes) * 1e-3 + sum(
        jnp.sum(r.backbone_um[1]) for r in routes
    )


@pytest.mark.parametrize(
    "case,x0,rtol",
    [("crossing", 40.0, 1e-5), ("optimized", 30.0, 1e-3), ("backbone", 3.0, 1e-4)],
)
def test_route_bundle_all_angle_gradients(case: str, x0: float, rtol: float) -> None:
    # "optimized": the value is scipy's bounded minimizer (xatol 1e-5), the
    # derivative its implicit derivative, so FD carries the optimizer's noise.
    check_grad(lambda v: _all_angle_route(case, v), x0, rtol=rtol, h=0.01)


def test_spiral_fixed_length_gradient() -> None:
    def f(length: Any) -> Any:
        c = gf.components.spiral_racetrack_fixed_length(length=length)
        return c.info["straight_length"] + c.xsize * 0.01

    check_grad(f, 1000.0, h=0.01)


def test_rasterize_gradient() -> None:
    def f(width: Any) -> Any:
        c = gf.components.straight(length=5, width=width)
        rho, _ = gf.rasterize.rasterize(
            c, layer=(1, 0), bbox=(0.013, -1.0, 4.987, 1.0), resolution=0.05
        )
        return rho.mean()

    check_grad(f, 0.5, rtol=1e-3)


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
