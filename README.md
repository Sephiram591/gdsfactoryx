# gdsfactoryx

[![tests](https://github.com/Sephiram591/gdsfactoryx/actions/workflows/test_code.yml/badge.svg)](https://github.com/Sephiram591/gdsfactoryx/actions/workflows/test_code.yml)
[![MIT](https://img.shields.io/github/license/Sephiram591/gdsfactoryx)](https://choosealicense.com/licenses/mit/)

**gdsfactoryx** (`9.51.0.post3`) is the JAX-differentiable port of upstream
[GDSFactory](https://github.com/gdsfactory/gdsfactory) (`v9.51.0`), built to power
[bifrost](https://github.com/Sephiram591/bifrost) (end-to-end differentiable 3D FDTD photonics simulation with
`fdtdx`, `LayerStack`-to-3D permittivity voxelization, and an interactive 3D viewer).

In `gdsfactoryx`, polygons, ports, references, paths, cross-sections, and routes use a pure-Python/JAX array backend
(`gdsfactory._jax.xp`), so layout geometry can be differentiated with `jax.grad` and compiled under fixed-topology
`jax.jit` (and `@bf.jit` in `bifrost`) with respect to component parameters. `klayout.db` is kept for concrete GDS/OASIS
I/O, viewing, and discrete boolean/DRC operations. The public API remains gdsfactory's: you still
`import gdsfactory as gf`.

- **Documentation**: `gdsfactoryx` keeps upstream `gdsfactory` APIs; see the
  [GDSFactory docs](https://gdsfactory.github.io/gdsfactory/) for general layout/PDK usage.
- **FDTD simulation & 3D viewer**: live in [bifrost](https://github.com/Sephiram591/bifrost) (`import bifrost as bf`),
  which consumes `gdsfactoryx` `Component`s and `LayerStack` definitions directly.

```python
import jax
import gdsfactory as gf

gf.gpdk.PDK.activate()
jax.grad(lambda r: gf.components.ring_single(radius=r).dxsize)(10.0)
```

## Installation

`gdsfactoryx` provides the same top-level `gdsfactory` import package as upstream
[gdsfactory](https://github.com/gdsfactory/gdsfactory), so the two distributions cannot share an environment. Remove
upstream `gdsfactory` first, then install `gdsfactoryx` with `uv` or `pip`:

```bash
# With uv
uv pip uninstall gdsfactory
uv pip install gdsfactoryx

# Or with pip
pip uninstall -y gdsfactory
pip install gdsfactoryx
```

To install the latest development version from GitHub:

```bash
uv pip install "gdsfactoryx @ git+https://github.com/Sephiram591/gdsfactoryx.git"
# or: pip install "gdsfactoryx @ git+https://github.com/Sephiram591/gdsfactoryx.git"
```

For local development from a clone:

```bash
git clone https://github.com/Sephiram591/gdsfactoryx.git
cd gdsfactoryx
uv sync --all-extras
```

If both distributions end up installed, `import gdsfactory` raises an `ImportError`. Fix it by uninstalling both and
reinstalling `gdsfactoryx`:

```bash
pip uninstall -y gdsfactory gdsfactoryx && pip install gdsfactoryx
```

Packages that declare a dependency on upstream `gdsfactory` (for example `gplugins`) will reinstall upstream `gdsfactory`
and break the environment unless installed without dependency resolution. Install such packages with `--no-deps` and
install their remaining dependencies manually:

```bash
pip install --no-deps gplugins
```

## Differentiability, `SNAP_ENABLED`, and `jax.jit`

Reverse mode (`jax.grad`, `jax.vjp`) and forward mode (`jax.jvp`, `jax.jacfwd`) work out of the box. Differentiation
runs **eagerly** by default: wrap your objective in `jax.grad`, and component factories keep normal Python control flow
on concrete primal values. Without tracers, geometry executes at NumPy speed.

**Fixed-topology `jax.jit` and `@bf.jit`**: `jax.jit` (and `bifrost`'s `@bf.jit`) works when the layout's discrete
structure—vertex counts, Python `if`/loop branches, cell names, and routing topology—does not depend on traced values.
Discrete choices and KLayout-backed operations require concrete values and raise a `TypeError` /
`ConcretizationTypeError` inside `jax.jit`; run those eagerly or keep structural parameters static.

**Straight-through DBU grid snapping (`gf.snap.SNAP_ENABLED`)**: Manufacturing-grid snapping (`1 nm` / `gf.kcl.dbu`)
uses straight-through rounding (primal coordinates snapped to the DBU grid, identity gradient in AD) both eagerly and
under `jax.jit`. Set `gf.snap.SNAP_ENABLED = False` when validating analytical gradients against finite differences.
Inside `jax.jit`, `SNAP_ENABLED` is read at trace time—changing it does not alter an already-compiled function.

| Quantity | How |
| --- | --- |
| polygon vertices | `component.get_polygons_points()` returns `(N, 2)` arrays |
| ports | `port.x`, `port.y`, `port.center_array`, `port.width`, `port.orientation` |
| references | `ref.transform`, `ref.move/rotate/mirror/connect`, `ref.xmin = ...` |
| bounding boxes and areas | `component.dbbox()`, `xmin/xmax/ymin/ymax/xsize/ysize/center`, `component.area(layer)` |
| paths and extrusion | `gf.path.arc/euler/straight/smooth/spiral_archimedean`, `Path.length()`, `gf.path.extrude` |
| routes | `route_single`, `route_bundle`, `route_bundle_all_angle`, `route_dubins`: `route.length`, `route.length_um`, element positions |
| layer stacks | `LayerStack`, `LayerLevel`, `DerivedLayer`, `LogicalLayer` (consumed by `bifrost` for 3D FDTD voxelization) |

Notes:

- Discrete choices (number of points, `if` branches, cell names, routing topology) use concrete primal values, so the
  gradient is exact for the chosen topology and zero across a discrete switch.
- `@gf.cell` calls with tracer arguments bypass the cell cache.
- KLayout is used only for discrete operations (I/O, `show`/`plot`, booleans, sizing, DRC fixes, fill, merged areas);
  gradients stop there. `gf.boolean`, `offset`, `over_under`, `fix_width`, `fix_spacing`, `trim`, `fill`,
  `get_polygons(merge=True)`, and A* routing are not differentiable.
- `component.area(layer)` computes the merged area as upstream; its derivative is that of the summed polygon areas,
  exact for non-overlapping polygons.

### Differences from upstream API

- `Component` and `ComponentReference` are `gdsfactoryx` classes, not `kfactory` `DKCell`/`DInstance`; use
  `component.to_kfactory()` for a concrete `kfactory` cell.
- `component.get_polygons()` returns KLayout polygons as upstream; `get_polygons_points()` returns differentiable arrays.
- `port.center` is an `(x, y)` tuple; use `port.center_array` (or `port.xy`) for array math.
- Manhattan routes are `gdsfactory.routing.ManhattanRoute` objects with `kfactory`'s attributes plus `backbone_um`
  (differentiable); all-angle and Dubins routes return `OpticalAllAngleRoute` objects with `length` / `length_um` in µm.

## Upstream & License

`gdsfactoryx` is derived from [gdsfactory/gdsfactory](https://github.com/gdsfactory/gdsfactory) (`v9.51.0`) under the
MIT license (see [LICENSE](LICENSE)). For 3D electromagnetic simulation, `LayerStack` meshing, and visualization, see
[bifrost](https://github.com/Sephiram591/bifrost).
