import jax
import jax.numpy as jnp
import numpy as np

import gdsfactoryx as gfx

BOUNDS = [[-6.0, -6.0], [6.0, 6.0]]
SHAPE = (96, 96)
PIXEL = (12.0 / 96) ** 2


def square(x0, y0, s):
    return jnp.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]])


def test_square_exact_coverage():
    d = gfx.rasterize([square(0.1, 0.2, 1.0)], x_edges=jnp.arange(0.0, 2.01, 0.5),
                      y_edges=jnp.arange(0.0, 2.01, 0.5))
    expected = np.zeros((4, 4))
    # x coverage per column: [0.4, 0.5, 0.1, 0] / 0.5 ; y: [0.3, 0.5, 0.2, 0] / 0.5
    cx = np.array([0.4, 0.5, 0.1, 0.0]) / 0.5
    cy = np.array([0.3, 0.5, 0.2, 0.0]) / 0.5
    expected = cx[:, None] * cy[None, :]
    np.testing.assert_allclose(d, expected, atol=1e-12)


def test_orientation_independent():
    p = square(-1.3, 0.7, 2.1)
    xe = ye = jnp.linspace(-3, 3, 25)
    np.testing.assert_allclose(
        gfx.rasterize([p], x_edges=xe, y_edges=ye),
        gfx.rasterize([p[::-1]], x_edges=xe, y_edges=ye),
        atol=1e-12,
    )


def test_raster_area_matches_polygon_area_and_grad():
    def total(r):
        g = gfx.components.ring(radius=r, width=0.5)
        return gfx.rasterize(g, "WG", bounds=BOUNDS, shape=SHAPE).sum() * PIXEL

    def area(r):
        return gfx.components.ring(radius=r, width=0.5).area("WG")

    r = jnp.array(4.0)
    np.testing.assert_allclose(total(r), area(r), rtol=1e-10)
    np.testing.assert_allclose(jax.grad(total)(r), jax.grad(area)(r), rtol=1e-8)


def test_vertex_gradient_matches_finite_difference():
    xe = ye = jnp.linspace(-2, 2, 17)
    w = jax.random.normal(jax.random.PRNGKey(0), (16, 16))
    tri = jnp.array([[-1.1, -0.9], [1.3, -0.2], [0.1, 1.4]])

    def loss(p):
        return jnp.sum(w * gfx.rasterize([p], x_edges=xe, y_edges=ye))

    g = jax.grad(loss)(tri)
    h = 1e-6
    num = np.zeros_like(tri)
    for i in range(3):
        for j in range(2):
            e = jnp.zeros_like(tri).at[i, j].set(h)
            num[i, j] = (loss(tri + e) - loss(tri - e)) / (2 * h)
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-8)


def test_jit_rasterize():
    f = jax.jit(lambda p: gfx.rasterize([p], bounds=BOUNDS, shape=SHAPE).sum())
    np.testing.assert_allclose(f(square(0, 0, 1.0)) * PIXEL, 1.0, rtol=1e-10)
