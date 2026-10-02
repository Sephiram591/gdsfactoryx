# Differentiable layouts (gdsfactoryx)

gdsfactoryx replaces gdsfactory's KLayout/kfactory integer-grid geometry backend
by a float64 [JAX](https://jax.readthedocs.io) backend. Polygons, ports,
references and routes are JAX-compatible arrays, so any scalar computed from a
layout can be differentiated with respect to the float parameters of the cell
functions that produced it:

```python
import jax
import gdsfactory as gf

gf.gpdk.PDK.activate()


def ring_footprint(radius):
    c = gf.components.ring_single(radius=radius)
    return c.dxsize * c.dysize


jax.grad(ring_footprint)(10.0)
```

## What is differentiable

| Quantity | How |
| --- | --- |
| polygon vertices | `component.get_polygons_points()` returns `(N, 2)` arrays |
| ports | `port.x`, `port.y`, `port.center_array`, `port.width`, `port.orientation` |
| references | `ref.transform`, `ref.move/rotate/mirror/connect`, `ref.xmin = ...` |
| bounding boxes | `component.dbbox()`, `xmin/xmax/ymin/ymax/xsize/ysize/center` |
| areas | `component.area(layer)` (see below), `component.area_unmerged(layer)` |
| paths | `gf.path.arc/euler/straight/smooth/spiral_archimedean`, `Path.length()` |
| extrusion | `gf.path.extrude`, `extrude_transition`, all cross-section fields |
| routes | `route_single`, `route_bundle` (+ electrical), `route_bundle_all_angle`, `route_dubins`: straight lengths, element positions, `route.length` |
| rasterization | `gf.rasterize.rasterize(component, layer, resolution)` |

Both reverse mode (`jax.grad`, `jax.vjp`) and forward mode (`jax.jvp`,
`jax.jacfwd`) work. Differentiation is **eager**: wrap your objective in
`jax.grad` but not in `jax.jit`. Cell functions keep their Python control flow,
which needs concrete values for decisions like the number of points of a bend.

## How it works

- **Arrays**: geometry is stored as float64 arrays. A small numpy-compatible
  namespace (`gdsfactory._jax.xp`) dispatches to `jax.numpy` only when a value
  carries a tracer, and uses plain numpy otherwise. Ordinary use without
  `jax.grad` therefore runs at numpy speed.
- **Discrete choices** such as the number of points, `if` branches, cell names and
  routing topology use the concrete primal values (`gdsfactory._jax.to_float`).
  The gradient is the derivative of the layout with the discrete choices held
  fixed.
- **Grid snapping** (1 nm) happens where kfactory/KLayout snap: when polygons
  are inserted, when instances are placed, for manhattan ports, and in
  `gf.snap.snap_to_grid`. It uses straight-through rounding: the value is
  snapped and the gradient is the identity. Set `gf.snap.SNAP_ENABLED = False`
  to compare against finite differences on unsnapped geometry.
- **Cell cache**: `@gf.cell` caches components by their arguments. Calls whose
  arguments contain tracers bypass the cache, so every gradient evaluation
  rebuilds the traced cells.
- **Routing**: route gradients come from autodiff, never from finite
  differences.
  - *Manhattan* (`route_single`, `route_bundle`): kfactory's manhattan router
    (`gdsfactory/routing/_traced_manhattan.py`) is ported onto dual numbers. Each
    coordinate holds kfactory's exact integer, which drives every comparison and
    branch, plus a JAX value that carries the derivative. The router therefore
    makes the same decisions as kfactory (corners, bundle ordering, path-length
    loops), and the corners are differentiable. Every route is checked against
    kfactory's output. Placed straights absorb the change of each segment length.
  - *All-angle* (`route_bundle_all_angle`): bundle offsets and port connections
    come from a traced port of kfactory's geometry. Where kfactory finds the
    connection angle with `scipy.optimize.minimize_scalar`, the value is
    kfactory's and the derivative comes from the implicit function theorem on the
    optimum (`r2(theta, inputs) = 0`).
  - The derivative is exact for the chosen topology. Changing a corner count or
    bundle order is a discrete jump, and its gradient is zero.
- **Swept cells**: `spiral_racetrack_fixed_length` interpolates a sweep the way
  upstream does. The two bracketing sweep points are rebuilt with traced
  geometry, so the straight length is differentiable in the length, the port
  spacing, the radius and the spacings.
- **KLayout** is only used for inherently discrete operations: GDS/OASIS I/O,
  `show`/`plot`, booleans, sizing/offset, DRC fixes, fill, merged areas. Values
  crossing into KLayout are concrete (gradients stop there).

## Caveats

- `component.area(layer)` returns the merged area, like upstream gdsfactory. Its
  derivative is the derivative of the sum of the polygon areas, which is exact
  when polygons on that layer don't overlap.
- Booleans (`gf.boolean`), `offset`, `over_under`, `fix_width`, `fix_spacing`,
  `trim`, `fill`, `get_polygons(merge=True)` and A* routing are not
  differentiable.
- Gradients are piecewise: they are exact within a fixed topology (number of
  points, chosen route) and undefined at the discrete switching points.
- `jax.jit` isn't supported.

## API differences from upstream gdsfactory

- `Component` and `ComponentReference` are gdsfactoryx classes, no longer
  kfactory `DKCell`/`DInstance` objects. Use `component.to_kfactory()` to get a
  (concrete) kfactory cell.
- `ComponentAllAngle` is a virtual cell, as upstream: shapes and references may
  be off-grid and are materialized on export like kfactory's `VInstance`.
- `component.get_polygons()` returns KLayout polygons, as upstream does.
  `component.get_polygons_points()` returns the differentiable arrays.
- `port.center` is an `(x, y)` tuple, as upstream. Use `port.center_array` (or
  `port.xy`) for array math.
- Routes are `gdsfactory.routing.ManhattanRoute` objects with the same
  attributes as kfactory's (`length` and `length_straights` in dbu,
  `backbone` in dbu), plus `backbone_um` (differentiable).
