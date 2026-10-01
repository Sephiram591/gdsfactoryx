# gdsfactoryx

A JAX-differentiable wrapper around [gdsfactory](https://github.com/gdsfactory/gdsfactory).

`gdsfactoryx` does not reimplement any geometry. It runs the real, unmodified
gdsfactory (pinned to `gdsfactory==9.51.0` from PyPI). Every gdsfactory function
is wrapped so it accepts `jax.Array` arguments and works with `jax.grad`,
`jax.jacfwd`/`jacrev`, `jax.jit` and `jax.vmap`.

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import gdsfactoryx as gfx

gfx.activate_pdk(dbu=1e-5)          # generic PDK on a fine grid (see "Accuracy")

gfx.components.straight(length=10)  # plain floats -> a normal gf.Component

geom = gfx.components.ring(radius=jnp.array(5.0), width=0.5)  # -> gfx.Geometry
geom.layer("WG")                    # list of (N, 2) vertex arrays (um)
gfx.components.straight(length=jnp.array(3.0)).ports["o2"].center  # ports too

area = lambda r: gfx.components.ring(radius=r, width=0.5).area("WG")
jax.grad(area)(jnp.array(5.0))      # ~ 2*pi*w
```

## What gets wrapped

`gfx` mirrors the `gdsfactory` namespace (`gfx.components`, `gfx.containers`,
`gfx.path`, `gfx.routing`, `gfx.cross_section`, `gfx.functions`, `gfx.samples`,
plus any other `gf.<name>`):

| gdsfactory attribute | in `gdsfactoryx` |
| --- | --- |
| functions / factories | `JaxifiedFunction` (same signature) |
| submodules | wrapped views of the module |
| classes, constants (`Component`, `Port`, `gpdk.LAYER`, ...) | re-exported unchanged |

A wrapped function called with **no** floating-point `jax.Array` arguments just
calls gdsfactory and returns its native result, so `gfx` is a drop-in for
`gf`. When **any** argument, at any depth of the argument pytree, is a
floating JAX array or tracer, the function returns a differentiable result:

| gdsfactory returns | differentiable result |
| --- | --- |
| `Component` / `ComponentReference` | `gfx.Geometry` (polygons per `(layer, datatype)` + `PortGeometry` per port) |
| `gf.Path` | `(N, 2)` array of points |
| pytree of numbers | same pytree of arrays |

### Your own cells

Write ordinary gdsfactory code. The function receives Python floats wherever the
caller passed JAX arrays, so hierarchy, references, routing and booleans all work:

```python
import gdsfactory as gf

@gfx.cell                     # = gf.cell + gfx.jaxify
def coupler(gap: float = 0.2, length: float = 10.0) -> gf.Component:
    c = gf.Component()
    s = gf.components.straight(length=length)
    a, b = c << s, c << s
    b.dmove((0, gap + 0.5))
    c.add_ports(a.ports, prefix="a_")
    c.add_ports(b.ports, prefix="b_")
    return c

jax.grad(lambda g: coupler(gap=g).ports["b_o1"].y)(jnp.array(0.2))  # 1.0
coupler.component(gap=jnp.array(0.3))   # real gf.Component, e.g. to write GDS
```

`gfx.jaxify(fn)` wraps any function (decorator or call, with options
`step=`, `merge=`, `layers=`).

### Rasterization

`gfx.rasterize(geometry, layer, bounds=[[x0, y0], [x1, y1]], shape=(nx, ny))`
returns the **exact** area fraction of each pixel covered by the polygons,
computed in pure JAX with Green's theorem. It's differentiable with respect to
the vertices, so gdsfactory parameters can drive JAX-based FDTD/FDFD solvers
or pixel losses. See [examples/fit_ring.py](examples/fit_ring.py), which
recovers a ring's radius and width from a target density map.

`Geometry` also offers differentiable `area`, `bbox`, `translate`, `rotate`
and `+` (union).

## How it works

```
params (jax) ──► pure_callback ──► real gdsfactory ──► polygons/ports (numpy) ──► Geometry
                 custom_jvp: J = central finite differences of the aligned geometry
```

* **Forward:** the wrapped call runs gdsfactory inside `jax.pure_callback`.
  Results are memoized per argument value.
* **Derivative:** a `jax.custom_jvp` rule computes the Jacobian with central
  finite differences (2 extra gdsfactory evaluations per scalar input) and returns
  `J @ tangent`. That's linear in the tangent, so reverse mode (`grad`, `vjp`,
  `jacrev`) works as well as forward mode.
* **Alignment:** polygons are matched between evaluations by centroid
  (Hungarian assignment), and vertices by the best cyclic shift. If the vertex
  count changes (e.g. arcs whose point count depends on the radius), each
  reference vertex follows the closest point on the perturbed boundary, which
  still gives the correct normal motion for area or raster losses. A
  `gfx.TopologyWarning` is emitted when this happens.
* **jit / vmap:** output shapes have to be known when tracing. They're taken
  from the last eager call with the same static arguments, or recorded explicitly
  with `fn.template(example_args...)`. Otherwise you get a `gfx.TemplateError`.

## Accuracy

gdsfactory snaps every vertex to the database unit (1 nm by default). That
rounding adds noise to finite differences: dA/dR of a 10 µm ring is about 4% off
at 1 nm. `gfx.activate_pdk(pdk, dbu=1e-5)` activates a copy of any PDK on a
0.01 nm grid (still ±21 mm of 32-bit coordinate range), which brings the error
below 0.1%. `dbu=1e-6` brings it to ~0.002%, with ±2.1 mm of range. It must be
called before any cell is built, and you get a `gfx.GradientAccuracyWarning` if
you take gradients on a coarse grid. Generate the final GDS for fabrication in a
session that uses the foundry dbu.

The finite-difference step is `gfx.settings.fd_step` (default 0.02, in the
units of the argument), or per function via `jaxify(fn, step=...)`. Geometry
that is linear in a parameter (lengths, widths, offsets, radii) is exact for any
step.

## Limitations

* First-order derivatives only (no Hessians through gdsfactory).
* Each gradient costs `2n + 1` gdsfactory evaluations for `n` scalar inputs.
* Every perturbed evaluation creates new cells in the session's layout. Call
  `gfx.clear_cache()` (gdsfactory's) between long optimization runs if memory
  grows.
* Only floating JAX arrays are differentiable. Integers, strings and objects
  (cross-sections, components passed to containers) are treated as static.

## Install

```bash
module load miniforge
conda create -n gdsfactoryx-wrap python=3.12
conda activate gdsfactoryx-wrap
pip install -e ".[cuda,dev]"
pytest
```

## License

MIT, same as gdsfactory (see [LICENSE](LICENSE)).
