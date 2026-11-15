"""``jax.jit`` of gdsfactoryx geometry: the same results as eager, bit for bit.

Grid snapping stays enabled here: it is the step that used to need concrete values. Every check runs on each
available JAX backend (CPU, and the GPU when there is one), since compiled code can fuse operations differently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory._jax import DBU, primal, snap_dbu
from gdsfactory.component import polygon_area


def _devices() -> list[Any]:
    devices = [jax.devices("cpu")[0]]
    try:
        devices += jax.devices("gpu")[:1]
    except RuntimeError:
        pass
    return devices


@pytest.fixture(params=_devices(), ids=lambda d: d.platform)
def device(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(autouse=True)
def snapping() -> Iterator[None]:
    enabled = gf.snap.SNAP_ENABLED
    gf.snap.SNAP_ENABLED = True
    yield
    gf.snap.SNAP_ENABLED = enabled


def snap_dbu_reference(x: Any) -> Any:
    """``snap_dbu`` of jax values before jit support (numpy on the concrete value), kept as the oracle."""
    inv = 1.0 / DBU
    v = np.asarray(primal(x), dtype=np.float64) * inv
    snapped = np.sign(v) * np.floor(np.abs(v) + 0.5) * DBU
    return x + jax.lax.stop_gradient(jnp.asarray(snapped) - x)


def bits(x: Any) -> np.ndarray:
    a = np.asarray(x)
    assert a.dtype == np.float64, a.dtype
    return a.view(np.int64)


def coordinates() -> np.ndarray:
    """(N, 2) um: half-dbu ties (rounded away from zero), one ulp around them, random, small and special values."""
    rng = np.random.default_rng(0)
    ties = (rng.integers(-(10**7), 10**7, size=500) + 0.5) * DBU
    near = ties + rng.choice([-1.0, 1.0], size=500) * np.spacing(ties)
    rand = rng.uniform(-1e4, 1e4, size=2000)
    small = rng.uniform(-2e-3, 2e-3, size=500)
    special = np.array(
        [0.0, -0.0, DBU, -DBU, 0.5 * DBU, -0.5 * DBU, 1.5 * DBU, -1.5 * DBU, 1e5, -1e5]
    )
    return np.concatenate([ties, near, rand, small, special]).reshape(-1, 2)


def test_snap_jit_matches_eager_bitwise(device: Any) -> None:
    x = jax.device_put(jnp.asarray(coordinates()), device)
    expected = bits(snap_dbu_reference(x))
    def _snap(v: Any) -> Any:
        return snap_dbu(v)

    np.testing.assert_array_equal(bits(snap_dbu(x)), expected)
    np.testing.assert_array_equal(bits(jax.jit(_snap)(x)), expected)
    primal_out, tangent = jax.jvp(snap_dbu, (x,), (jnp.ones_like(x),))
    np.testing.assert_array_equal(bits(primal_out), expected)
    np.testing.assert_array_equal(np.asarray(tangent), 1.0)
    jit_primal, jit_tangent = jax.jit(
        lambda v: jax.jvp(snap_dbu, (v,), (jnp.ones_like(v),))
    )(x)
    np.testing.assert_array_equal(bits(jit_primal), expected)
    np.testing.assert_array_equal(np.asarray(jit_tangent), 1.0)


def test_snap_gradient_is_identity_under_jit(device: Any) -> None:
    x = jax.device_put(jnp.asarray(coordinates()), device)
    w = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=x.shape)), device
    )
    grad = jax.jit(jax.grad(lambda v: jnp.sum(w * snap_dbu(v))))(x)
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(w))


def test_snap_float32_input_is_promoted_like_before(device: Any) -> None:
    x = jax.device_put(jnp.asarray(coordinates().astype(np.float32)), device)
    expected = bits(snap_dbu_reference(x))
    def _snap(v: Any) -> Any:
        return snap_dbu(v)

    np.testing.assert_array_equal(bits(snap_dbu(x)), expected)
    np.testing.assert_array_equal(bits(jax.jit(_snap)(x)), expected)


def test_snap_python_and_numpy_values_unchanged() -> None:
    values = coordinates()
    inv = 1.0 / DBU  # KLayout multiplies by 1/dbu (dividing by dbu rounds differently)
    expected = np.sign(values * inv) * np.floor(np.abs(values * inv) + 0.5) * DBU
    np.testing.assert_array_equal(bits(snap_dbu(values)), bits(expected))
    for v in values.ravel()[:50]:
        assert isinstance(snap_dbu(float(v)), float)
        assert snap_dbu(float(v)) == float(
            np.sign(v * inv) * np.floor(abs(v * inv) + 0.5) * DBU
        )


def test_snap_disabled_returns_input_under_jit(device: Any) -> None:
    """SNAP_ENABLED is read when the function is traced (a new function here, not a cached compile)."""
    def _snap(v: Any) -> Any:
        return snap_dbu(v)

    x = jax.device_put(jnp.asarray(coordinates()), device)
    gf.snap.SNAP_ENABLED = False
    np.testing.assert_array_equal(bits(jax.jit(_snap)(x)), bits(x))


def build(p: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """Traced polygon vertices, port centers and widths, and a reference moved by a traced offset."""
    w, dx, h = p[0], p[1], p[2]
    child = gf.Component()
    child.add_polygon(jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]) * h, layer=(2, 0))
    c = gf.Component()
    xs = jnp.array([0.0, 10.0, 10.0, 0.0]) + jnp.array([0.0, dx, dx, 0.0])
    c.add_polygon(
        jnp.stack([xs, jnp.stack([-w / 2, -w / 2, w / 2, w / 2])], axis=1), layer=(1, 0)
    )
    c.add_port(name="o1", center=(0.0, 0.0), width=w, orientation=180, layer=(1, 0))
    c.add_port(
        name="o2",
        center=jnp.stack([10.0 + dx, 0.0 * dx]),
        width=w,
        orientation=0,
        layer=(1, 0),
    )
    c.add_ref(child).move((dx, 2.0 * dx))
    polys = c.get_polygons_points()
    return (
        [q for k in sorted(polys) for q in polys[k]],
        [port.center_array for port in c.ports],
        [jnp.asarray(port.width) for port in c.ports],
    )


def test_component_under_jit_matches_eager_bitwise(device: Any) -> None:
    p = jax.device_put(jnp.array([0.5004999, 0.12345, 1.9876543]), device)
    eager = jax.tree.leaves(build(p))
    jitted = jax.tree.leaves(jax.jit(build)(p))
    assert len(eager) == len(jitted) == 2 + 2 + 2
    for a, b in zip(eager, jitted, strict=True):
        np.testing.assert_array_equal(bits(b), bits(a))


def test_component_gradient_under_jit_matches_eager(device: Any) -> None:
    def scalar(p: Any) -> Any:
        polys, centers, widths = build(p)
        return (
            sum(jnp.sum(q * q) for q in polys)
            + sum(jnp.sum(c) for c in centers)
            + sum(widths)
        )

    p = jax.device_put(jnp.array([0.5004999, 0.12345, 1.9876543]), device)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.grad(scalar))(p)),
        np.asarray(jax.grad(scalar)(p)),
        rtol=1e-12,
        atol=0,
    )


def test_klayout_operation_still_raises_under_jit() -> None:
    def merged_area(w: Any) -> Any:
        c = gf.Component()
        c.add_polygon(
            jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) * w,
            layer=(1, 0),
        )
        return sum(
            jnp.sum(q) for q in c.get_polygons_points(merge=True)[gf.get_layer((1, 0))]
        )

    with pytest.raises(TypeError):
        jax.jit(merged_area)(jnp.asarray(1.5))


def _component_leaves(c: gf.Component) -> tuple[list[Any], list[Any], list[Any]]:
    polys = c.get_polygons_points()
    return (
        [q for k in sorted(polys) for q in polys[k]],
        [port.center_array for port in c.ports],
        [jnp.asarray(port.width) for port in c.ports],
    )


def _component_loss(c: gf.Component) -> Any:
    polys, centers, widths = _component_leaves(c)
    loss: Any = jnp.asarray(0.0)
    for q in polys:
        loss = loss + jnp.sum(q * q)
    for center in centers:
        loss = loss + jnp.sum(center * center)
    for w in widths:
        loss = loss + w * w
    return loss


def _assert_jit_matches_eager(fn: Callable[[Any], Any], x: Any) -> None:
    eager = jax.tree.leaves(fn(x))
    jitted = jax.tree.leaves(jax.jit(fn)(x))
    assert len(eager) == len(jitted) > 0
    for a, b in zip(eager, jitted, strict=True):
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        assert b_np.shape == a_np.shape
        np.testing.assert_allclose(b_np, a_np, rtol=1e-12, atol=1e-12)


def _assert_jit_grad_matches_eager(
    loss_fn: Callable[[Any], Any], x: Any, *, rtol: float = 1e-12
) -> None:
    eager_val = loss_fn(x)
    eager_grad = jax.grad(loss_fn)(x)
    jit_grad = jax.grad(jax.jit(loss_fn))(x)
    vg_val, vg_grad = jax.jit(jax.value_and_grad(loss_fn))(x)

    assert np.all(np.isfinite(np.asarray(jit_grad)))
    assert float(jnp.linalg.norm(jit_grad)) > 0.0
    np.testing.assert_allclose(
        np.asarray(jit_grad), np.asarray(eager_grad), rtol=rtol, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(vg_grad), np.asarray(eager_grad), rtol=rtol, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(vg_val), np.asarray(eager_val), rtol=rtol, atol=0
    )


def _assert_jit_component(
    build_fn: Callable[[Any], gf.Component], x: Any, *, rtol: float = 1e-12
) -> None:
    def _leaves(v: Any) -> tuple[list[Any], list[Any], list[Any]]:
        return _component_leaves(build_fn(v))

    def _loss(v: Any) -> Any:
        return _component_loss(build_fn(v))

    _assert_jit_matches_eager(_leaves, x)
    _assert_jit_grad_matches_eager(_loss, x, rtol=rtol)


def _straight_x_loss(length: Any) -> Any:
    c = gf.components.straight(length=length)
    xs_sum = sum(
        jnp.sum(q[:, 0]) for polys in c.get_polygons_points().values() for q in polys
    )
    return 0.5 * c.ports["o2"].x + 0.25 * xs_sum


def test_snap_toggle_straight_through_grad_under_jit(device: Any) -> None:
    """Fresh local functions re-trace SNAP_ENABLED and keep ST grads in both modes."""
    x_len = jax.device_put(jnp.asarray(10.0004), device)
    x_vec = jax.device_put(jnp.array([1.0004, -2.0006, 0.5004]), device)
    weights = jax.device_put(jnp.array([1.5, -0.75, 2.25]), device)
    weights_np = np.asarray(weights)

    gf.snap.SNAP_ENABLED = True

    def _snap_loss_on(v: Any) -> Any:
        return jnp.sum(weights * snap_dbu(v))

    def _cell_loss_on(length: Any) -> Any:
        return _straight_x_loss(length)

    snap_val_on, snap_grad_on = jax.jit(jax.value_and_grad(_snap_loss_on))(x_vec)
    np.testing.assert_allclose(
        np.asarray(snap_val_on),
        np.sum(weights_np * np.array([1.0, -2.001, 0.5])),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(np.asarray(snap_grad_on), weights_np, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(jax.grad(jax.jit(_snap_loss_on))(x_vec)),
        weights_np,
        rtol=1e-12,
        atol=0,
    )

    cell_val_on, cell_grad_on = jax.jit(jax.value_and_grad(_cell_loss_on))(x_len)
    np.testing.assert_allclose(np.asarray(cell_val_on), 10.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(cell_grad_on), 1.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(jax.grad(jax.jit(_cell_loss_on))(x_len)), 1.0, rtol=1e-12, atol=0
    )

    gf.snap.SNAP_ENABLED = False

    def _snap_loss_off(v: Any) -> Any:
        return jnp.sum(weights * snap_dbu(v))

    def _cell_loss_off(length: Any) -> Any:
        return _straight_x_loss(length)

    snap_val_off, snap_grad_off = jax.jit(jax.value_and_grad(_snap_loss_off))(x_vec)
    np.testing.assert_allclose(
        np.asarray(snap_val_off),
        np.sum(weights_np * np.asarray(x_vec)),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(snap_grad_off), weights_np, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(jax.jit(_snap_loss_off))(x_vec)),
        weights_np,
        rtol=1e-12,
        atol=0,
    )

    cell_val_off, cell_grad_off = jax.jit(jax.value_and_grad(_cell_loss_off))(x_len)
    np.testing.assert_allclose(np.asarray(cell_val_off), 10.0004, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(cell_grad_off), 1.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(jax.grad(jax.jit(_cell_loss_off))(x_len)), 1.0, rtol=1e-12, atol=0
    )


@gf.cell
def _hier_wg_chain(length: Any, radius: Any, width: Any) -> gf.Component:
    c = gf.Component()
    xs = gf.CrossSection(
        sections=(
            gf.Section(
                width=width, offset=0.0, layer=(1, 0), port_names=("o1", "o2")
            ),
        )
    )
    s1 = c << gf.components.straight(length=length, cross_section=xs)
    b = c << gf.components.bend_circular(
        radius=radius, npoints=64, cross_section=xs, allow_min_radius_violation=True
    )
    b.connect("o1", s1.ports["o2"])
    s2 = c << gf.components.straight(length=0.5 * length, cross_section=xs)
    s2.connect("o1", b.ports["o2"])
    c.add_port("o1", port=s1.ports["o1"])
    c.add_port("o2", port=s2.ports["o2"])
    return c


def test_hierarchical_cell_connect_and_flatten_under_jit(device: Any) -> None:
    p = jax.device_put(jnp.array([7.5004, 8.1234, 0.5502]), device)

    def _build_unflat(v: Any) -> gf.Component:
        return _hier_wg_chain(v[0], v[1], v[2])

    def _build_flat(v: Any) -> gf.Component:
        c_flat = _hier_wg_chain(v[0], v[1], v[2]).copy().flatten(merge=False)
        assert not c_flat.insts
        return c_flat

    def _canonicalize_poly(pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if area2 < 0.0:
            pts = pts[::-1]
        idx = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
        return np.roll(pts, -idx, axis=0)

    def _flat_state(v: Any) -> tuple[dict[int, list[Any]], list[tuple[Any, Any, Any]]]:
        c_flat = _build_flat(v)
        polys = {lay: list(c_flat.polygons[lay]) for lay in sorted(c_flat.polygons)}
        ports = [(port.center, port.orientation, port.width) for port in c_flat.ports]
        return polys, ports

    _assert_jit_component(_build_unflat, p)

    # flatten(merge=False) dispatches to KLayout flatten_exact eagerly (CW vertex
    # order) vs _flat_polygons_klayout under jit (CCW); compare canonicalized polygons.
    eager_polys, eager_ports = _flat_state(p)
    jit_polys, jit_ports = jax.jit(_flat_state)(p)
    assert list(eager_polys) == list(jit_polys)
    for lay in eager_polys:
        e_sorted = sorted(
            (_canonicalize_poly(np.asarray(q)) for q in eager_polys[lay]),
            key=lambda q: (len(q), round(float(q[0, 0]), 6), round(float(q[0, 1]), 6)),
        )
        j_sorted = sorted(
            (_canonicalize_poly(np.asarray(q)) for q in jit_polys[lay]),
            key=lambda q: (len(q), round(float(q[0, 0]), 6), round(float(q[0, 1]), 6)),
        )
        assert len(e_sorted) == len(j_sorted)
        for ep, jp in zip(e_sorted, j_sorted, strict=True):
            np.testing.assert_allclose(jp, ep, rtol=1e-12, atol=1e-12)
    for a, b in zip(jax.tree.leaves(eager_ports), jax.tree.leaves(jit_ports), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-12)

    eager_vg = jax.value_and_grad(lambda v: _component_loss(_build_flat(v)))(p)
    jit_vg = jax.jit(jax.value_and_grad(lambda v: _component_loss(_build_flat(v))))(p)
    for a, b in zip(jax.tree.leaves(eager_vg), jax.tree.leaves(jit_vg), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-12)

    # KLayout flatten(merge=True) on concrete floats preserves total area and ports.
    def _flat_invariants(v: Any) -> tuple[Any, list[Any], list[Any]]:
        c_flat = _hier_wg_chain(v[0], v[1], v[2]).copy().flatten()
        polys, centers, widths = _component_leaves(c_flat)
        area = sum(jnp.abs(polygon_area(q)) for q in polys)
        return area, centers, widths

    jit_area, jit_centers, jit_widths = jax.jit(_flat_invariants)(p)
    conc_area, conc_centers, conc_widths = _flat_invariants((7.5004, 8.1234, 0.5502))
    np.testing.assert_allclose(
        np.asarray(jit_area), float(conc_area), rtol=1e-4, atol=1e-4
    )
    for jc, cc in zip(jit_centers, conc_centers, strict=True):
        np.testing.assert_allclose(np.asarray(jc), np.asarray(cc), rtol=1e-4, atol=1e-3)
    for jw, cw in zip(jit_widths, conc_widths, strict=True):
        np.testing.assert_allclose(float(jw), float(cw), rtol=1e-4, atol=1e-3)


def test_cross_section_and_extrude_under_jit(device: Any) -> None:
    p_xs = jax.device_put(jnp.array([0.55, 0.4, 9.5]), device)

    def _build_xs_extrude(v: Any) -> gf.Component:
        w, margin, radius = v[0], v[1], v[2]
        xs = gf.CrossSection(
            sections=(
                gf.Section(width=w, offset=0.0, layer=(1, 0), port_names=("o1", "o2")),
                gf.Section(width=w + margin, offset=0.0, layer=(2, 0)),
            )
        )
        return gf.path.arc(radius=radius, angle=90.0, npoints=64).extrude(xs)

    _assert_jit_component(_build_xs_extrude, p_xs)

    p_ep = jax.device_put(jnp.array([15.0, 8.0, 0.6]), device)

    def _build_extrude_path(v: Any) -> gf.Component:
        dx, dy, width = v[0], v[1], v[2]
        pts = jnp.array([[0.0, 0.0], [dx, 0.3 * dy], [dx + 15.0, dy]])
        poly = gf.functions.extrude_path(
            pts, width=width, with_manhattan_facing_angles=False
        )
        c = gf.Component()
        c.add_polygon(poly, layer=(1, 0))
        return c

    _assert_jit_component(_build_extrude_path, p_ep)


def _build_jit_component(name: str, v: Any) -> gf.Component:
    if name == "bend_euler":
        return gf.components.bend_euler(radius=v[0], width=v[1], p=0.5)
    if name == "bend_circular":
        return gf.components.bend_circular(radius=v[0], width=v[1])
    if name == "taper":
        return gf.components.taper(length=v[0], width1=v[1], width2=v[2])
    if name == "taper_adiabatic":
        return gf.components.taper_adiabatic(width1=v[0], width2=v[1], alpha=v[2])
    raise ValueError(f"Unknown test component {name!r}")


@pytest.mark.parametrize(
    ("name", "init"),
    [
        ("bend_euler", [10.0, 0.5]),
        ("bend_circular", [10.0, 0.5]),
        ("taper", [12.0, 0.5, 2.2]),
        ("taper_adiabatic", [0.5, 3.0, 1.0]),
    ],
)
def test_components_under_jit_and_grad(
    device: Any, name: str, init: list[float]
) -> None:
    p = jax.device_put(jnp.array(init), device)

    def _build(v: Any) -> gf.Component:
        return _build_jit_component(name, v)

    _assert_jit_component(_build, p)


def test_smooth_under_jit_and_grad(device: Any) -> None:
    p = jax.device_put(jnp.array([20.0, 20.0, 5.0]), device)

    def _smooth_outputs(v: Any) -> tuple[Any, tuple[list[Any], list[Any], list[Any]]]:
        dx, dy, radius = v[0], v[1], v[2]
        pts = jnp.array([[0.0, 0.0], [dx, 0.0], [dx, dy], [dx + 20.0, dy]])
        path = gf.path.smooth(pts, radius=radius, bend=gf.path.euler, npoints=64)
        return path.points, _component_leaves(path.extrude("strip"))

    def _smooth_loss(v: Any) -> Any:
        dx, dy, radius = v[0], v[1], v[2]
        pts = jnp.array([[0.0, 0.0], [dx, 0.0], [dx, dy], [dx + 20.0, dy]])
        path = gf.path.smooth(pts, radius=radius, bend=gf.path.euler, npoints=64)
        return jnp.sum(path.points * path.points) + _component_loss(
            path.extrude("strip")
        )

    _assert_jit_matches_eager(_smooth_outputs, p)
    _assert_jit_grad_matches_eager(_smooth_loss, p)


def test_routing_under_jit(device: Any) -> None:
    w0 = jax.device_put(jnp.asarray(0.55), device)

    def _build_single_sbend(w: Any) -> gf.Component:
        c = gf.Component()
        xs = gf.CrossSection(
            sections=(
                gf.Section(width=w, offset=0.0, layer=(1, 0), port_names=("o1", "o2")),
            )
        )
        p1 = gf.Port("o1", center=(0.0, 0.0), width=w, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(40.0, 4.0), width=w, orientation=180, layer=(1, 0))
        gf.routing.route_single_sbend(
            c,
            p1,
            p2,
            bend_s=partial(gf.components.bend_s, cross_section=xs),
            cross_section=xs,
            allow_width_mismatch=True,
        )
        return c

    def _build_bundle_sbend(w: Any) -> gf.Component:
        c = gf.Component()
        xs = gf.CrossSection(
            sections=(
                gf.Section(width=w, offset=0.0, layer=(1, 0), port_names=("o1", "o2")),
            )
        )
        p1s = [
            gf.Port("a1", center=(0.0, 0.0), width=w, orientation=0, layer=(1, 0)),
            gf.Port("a2", center=(0.0, 15.0), width=w, orientation=0, layer=(1, 0)),
        ]
        p2s = [
            gf.Port("b1", center=(45.0, 4.0), width=w, orientation=180, layer=(1, 0)),
            gf.Port("b2", center=(45.0, 19.0), width=w, orientation=180, layer=(1, 0)),
        ]
        gf.routing.route_bundle_sbend(
            c,
            p1s,
            p2s,
            cross_section=xs,
            auto_taper=False,
            allow_width_mismatch=True,
        )
        return c

    _assert_jit_component(_build_single_sbend, w0)
    _assert_jit_component(_build_bundle_sbend, w0)

    # KFactory Manhattan routers and bezier curvature solver require concrete port coords under jit.
    dy0 = jax.device_put(jnp.asarray(20.0), device)

    def _traced_route_single(dy: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(40.0, dy), width=0.5, orientation=180, layer=(1, 0))
        return gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            on_placer_error="error",
            on_collision="error",
        ).length

    def _traced_route_bundle(dy: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(40.0, dy), width=0.5, orientation=180, layer=(1, 0))
        return gf.routing.route_bundle(
            c,
            [p1],
            [p2],
            cross_section="strip",
            on_placer_error="error",
            on_collision="error",
        )[0].length

    def _traced_sbend_coords(dy: Any) -> Any:
        c = gf.Component()
        p1 = gf.Port("o1", center=(0.0, 0.0), width=0.5, orientation=0, layer=(1, 0))
        p2 = gf.Port("o2", center=(40.0, dy), width=0.5, orientation=180, layer=(1, 0))
        gf.routing.route_single_sbend(c, p1, p2, cross_section="strip")
        return _component_loss(c)

    with pytest.raises(TypeError):
        jax.jit(_traced_route_single)(dy0)
    with pytest.raises(TypeError):
        jax.jit(_traced_route_bundle)(dy0)
    with pytest.raises(TypeError):
        jax.jit(_traced_sbend_coords)(dy0)
