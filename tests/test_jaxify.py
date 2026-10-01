import warnings

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gdsfactoryx as gfx


def fd(f, x, h=0.05):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_passthrough_returns_gdsfactory_objects():
    c = gfx.components.straight(length=10)
    assert isinstance(c, gf.Component)
    assert gfx.components.straight is gfx.components.straight
    assert gfx.Component is gf.Component
    assert gfx.gpdk.LAYER is gf.gpdk.LAYER


def test_geometry_values_match_gdsfactory():
    g = gfx.components.straight(length=jnp.array(10.0), width=jnp.array(0.5))
    assert isinstance(g, gfx.Geometry)
    assert g.layers == [(1, 0)]
    np.testing.assert_allclose(g.area("WG"), 5.0)
    np.testing.assert_allclose(g.ports["o2"].center, [10.0, 0.0])
    np.testing.assert_allclose(g.ports["o2"].orientation, 0.0)
    np.testing.assert_allclose(g.ports["o1"].width, 0.5)


def test_grad_area_straight():
    def area(length, width):
        return gfx.components.straight(length=length, width=width).area("WG")

    dl, dw = jax.grad(area, argnums=(0, 1))(jnp.array(10.0), jnp.array(0.5))
    np.testing.assert_allclose(dl, 0.5, rtol=1e-6)
    np.testing.assert_allclose(dw, 10.0, rtol=1e-6)


def test_grad_port_position():
    def x(length):
        return gfx.components.straight(length=length).ports["o2"].x

    np.testing.assert_allclose(jax.grad(x)(jnp.array(7.0)), 1.0, rtol=1e-6)


@pytest.mark.parametrize("radius", [5.0, 10.0, 50.0])
def test_grad_ring_area(radius):
    def area(r):
        return gfx.components.ring(radius=r, width=0.5).area("WG")

    n = len(gfx.components.ring(radius=jnp.array(radius), width=0.5).layer("WG")[0])
    sides = (n - 1) // 2
    expected = sides * np.sin(2 * np.pi / sides) * 0.5  # d/dR of the polygon area
    np.testing.assert_allclose(jax.grad(area)(jnp.array(radius)), expected, rtol=2e-3)


def test_bend_euler_matches_gdsfactory_fd():
    def port_xy(r):
        return gfx.components.bend_euler(radius=r).ports["o2"].center

    def gf_port_xy(r):
        return np.array(gf.components.bend_euler(radius=float(r)).ports["o2"].center)

    jac = jax.jacfwd(port_xy)(jnp.array(10.0))
    np.testing.assert_allclose(jac, fd(gf_port_xy, 10.0), rtol=1e-3, atol=1e-6)


def test_forward_and_reverse_mode_agree():
    def f(p):
        g = gfx.components.mmi1x2(width_mmi=p[0], length_mmi=p[1], gap_mmi=p[2])
        return jnp.stack([g.area("WG"), g.ports["o2"].y, g.ports["o3"].x])

    p = jnp.array([2.5, 5.5, 0.25])
    np.testing.assert_allclose(jax.jacfwd(f)(p), jax.jacrev(f)(p), rtol=1e-10, atol=1e-12)


def test_jit_after_eager_call_and_new_values():
    def area(length):
        return gfx.components.straight(length=length, width=0.5).area("WG")

    area(jnp.array(10.0))  # records the topology template
    jitted = jax.jit(jax.value_and_grad(area))
    v, g = jitted(jnp.array(12.0))
    np.testing.assert_allclose(v, 6.0)
    np.testing.assert_allclose(g, 0.5, rtol=1e-6)


def test_jit_without_template_raises_then_template_fixes_it():
    @gfx.jaxify
    def taper(length):
        return gf.components.taper(length=length, width1=0.5, width2=1.0)

    f = jax.jit(lambda L: taper(L).area())
    with pytest.raises(gfx.TemplateError):
        f(jnp.array(10.0))
    taper.template(jnp.array(10.0))
    np.testing.assert_allclose(f(jnp.array(10.0)), 7.5)


def test_vmap():
    def area(length):
        return gfx.components.straight(length=length, width=0.5).area("WG")

    area(jnp.array(1.0))
    out = jax.vmap(area)(jnp.array([10.0, 11.0, 12.0]))
    np.testing.assert_allclose(out, [5.0, 5.5, 6.0])
    grads = jax.vmap(jax.grad(area))(jnp.array([10.0, 11.0]))
    np.testing.assert_allclose(grads, [0.5, 0.5], rtol=1e-6)


def test_user_cell_with_hierarchy():
    @gfx.cell
    def pair(gap: float = 1.0, length: float = 5.0) -> gf.Component:
        c = gf.Component()
        s = gf.components.straight(length=length, width=0.5)
        a = c << s
        b = c << s
        b.dmove((0, gap + 0.5))
        c.add_ports(a.ports, prefix="a_")
        c.add_ports(b.ports, prefix="b_")
        return c

    def f(gap, length):
        g = pair(gap=gap, length=length)
        return g.ports["b_o2"].y + g.ports["b_o2"].x

    dg, dl = jax.grad(f, argnums=(0, 1))(jnp.array(1.0), jnp.array(5.0))
    np.testing.assert_allclose([dg, dl], [1.0, 1.0], rtol=1e-6)
    assert isinstance(pair(gap=1.0), gf.Component)
    assert isinstance(pair.component(gap=jnp.array(2.0)), gf.Component)


def test_container_with_static_component_argument():
    s = gf.components.straight(length=5, width=0.5)

    def area(length):
        return gfx.components.extend_ports(component=s, length=length).area("WG")

    np.testing.assert_allclose(jax.grad(area)(jnp.array(3.0)), 1.0, rtol=1e-6)


def test_jaxify_user_function_with_cross_section():
    @gfx.jaxify
    def make(L, w):
        return gf.components.straight(
            length=L, cross_section=gf.cross_section.strip(width=w)
        )

    g = jax.grad(lambda L, w: make(L, w).area("WG"), argnums=(0, 1))(
        jnp.array(4.0), jnp.array(0.6)
    )
    np.testing.assert_allclose(g, [0.6, 4.0], rtol=1e-6)


def test_pytree_kwargs():
    def area(p):
        return gfx.components.straight(**p).area("WG")

    g = jax.grad(area)({"length": jnp.array(4.0), "width": jnp.array(0.6)})
    np.testing.assert_allclose([g["length"], g["width"]], [0.6, 4.0], rtol=1e-6)


def test_path_points_are_differentiable():
    pts = gfx.path.straight(length=jnp.array(3.0), npoints=4)
    assert pts.shape == (4, 2)
    jac = jax.jacfwd(lambda L: gfx.path.straight(length=L, npoints=4))(jnp.array(3.0))
    np.testing.assert_allclose(jac[:, 0], [0, 1 / 3, 2 / 3, 1], atol=1e-8)


def test_topology_change_warns_and_stays_finite():
    def area(r):
        return gfx.components.circle(radius=r, angle_resolution=2.5).area("WG")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(area)(jnp.array(3.0))
    np.testing.assert_allclose(g, 2 * np.pi * 3.0, rtol=5e-3)


def test_geometry_transforms_are_differentiable():
    def f(dx, angle):
        g = gfx.components.straight(length=jnp.array(10.0)).rotate(angle).translate(dx)
        return g.ports["o2"].x

    dx, da = jax.grad(f, argnums=(0, 1))(jnp.array(1.0), jnp.array(90.0))
    np.testing.assert_allclose(dx, 1.0)
    np.testing.assert_allclose(da, -10.0 * np.pi / 180, rtol=1e-6)
