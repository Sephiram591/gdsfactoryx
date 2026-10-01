"""Recover a ring's radius and width from a target density map.

Gradients flow from a pixel-space loss, through the exact rasterizer, through
the real gdsfactory `ring` geometry, back to its parameters.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import gdsfactoryx as gfx  # noqa: E402

gfx.activate_pdk(dbu=1e-5)  # fine grid -> accurate finite-difference gradients

BOUNDS = [[-7.0, -7.0], [7.0, 7.0]]
SHAPE = (140, 140)


def density(params):
    ring = gfx.components.ring(radius=params["radius"], width=params["width"])
    return gfx.rasterize(ring, "WG", bounds=BOUNDS, shape=SHAPE)


target = density({"radius": jnp.array(5.3), "width": jnp.array(0.65)})


def loss(params):
    return jnp.mean((density(params) - target) ** 2)


params = {"radius": jnp.array(5.0), "width": jnp.array(0.5)}
value_and_grad = jax.value_and_grad(loss)
lr = {"radius": 2.0, "width": 0.5}
for step in range(60):
    value, grads = value_and_grad(params)
    # normalized gradient descent: robust to the scale of the pixel loss
    norm = jnp.sqrt(sum(g**2 for g in grads.values()))
    decay = 0.05 * 0.93**step
    params = {k: params[k] - lr[k] * decay * grads[k] / norm for k in params}
    if step % 10 == 0:
        print(f"step {step:3d} loss {value:.3e} " +
              " ".join(f"{k}={float(v):.4f}" for k, v in params.items()))

print("final", {k: round(float(v), 3) for k, v in params.items()})

# The optimized design as a regular gdsfactory Component:
component = gfx.components.ring.component(**params)
print(component.name, component.dbbox())
