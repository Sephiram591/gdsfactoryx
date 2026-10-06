# gdsfactoryx

gdsfactoryx is a port of [gdsfactory](https://github.com/gdsfactory/gdsfactory)
with a differentiable [JAX](https://jax.readthedocs.io) geometry backend.
Polygons, ports, references and routes are float64 arrays, so any scalar
computed from a layout can be differentiated with `jax.grad` with respect to
the parameters of the cell functions that built it. Layouts can be voxelized
directly into an [fdtdx](https://github.com/ymahlau/fdtdx) FDTD simulation, with
gradients flowing from the simulation back to the layout parameters.

```python
import jax
import gdsfactory as gf

gf.gpdk.PDK.activate()
jax.grad(lambda r: gf.components.ring_single(radius=r).dxsize)(10.0)
```

!!! note "Basic gdsfactory usage"
    gdsfactoryx keeps gdsfactory's API: you still `import gdsfactory as gf`, and
    cells, PDKs, routing, YAML schematics and GDS export work as upstream. For
    everything that is not specific to this port, use the
    [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/).
    These docs cover only what gdsfactoryx adds or changes.

## Installation

gdsfactoryx is not on PyPI. Install it from GitHub; the `fdtdx` extra adds the
FDTD integration:

```bash
pip install "gdsfactory[fdtdx] @ git+https://github.com/Sephiram591/gdsfactoryx.git"
```

The distribution is still called `gdsfactory`, so it replaces an upstream
gdsfactory installed in the same environment. Use a separate environment if you
need both. For GPU simulations, install a CUDA-enabled `jax` as described in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

For development:

```bash
git clone https://github.com/Sephiram591/gdsfactoryx.git
cd gdsfactoryx
pip install -e ".[dev,fdtdx]"
```

## Contents

- [Differentiable layouts](differentiable.md): what is differentiable, how the
  JAX backend works, its caveats and the API differences from upstream
  gdsfactory.
- [FDTD with fdtdx](fdtdx.md): simulating a component and its `LayerStack`
  with fdtdx through `StackDevice`, and optimizing layouts with gradients of
  simulation results.
- [API reference](api.md): `gdsfactory.fdtdx_stack` and
  `gdsfactory.fdtdx_geometry`.
