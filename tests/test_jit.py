"""``jax.jit`` of gdsfactoryx geometry: the same results as eager, bit for bit.

Grid snapping stays enabled here: it is the step that used to need concrete values. Every check runs on each
available JAX backend (CPU, and the GPU when there is one), since compiled code can fuse operations differently.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory._jax import DBU, primal, snap_dbu


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
    np.testing.assert_array_equal(bits(snap_dbu(x)), expected)
    np.testing.assert_array_equal(bits(jax.jit(snap_dbu)(x)), expected)
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
    np.testing.assert_array_equal(bits(snap_dbu(x)), expected)
    np.testing.assert_array_equal(bits(jax.jit(snap_dbu)(x)), expected)


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
    x = jax.device_put(jnp.asarray(coordinates()), device)
    gf.snap.SNAP_ENABLED = False
    np.testing.assert_array_equal(bits(jax.jit(lambda v: snap_dbu(v))(x)), bits(x))


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
