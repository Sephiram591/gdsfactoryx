"""Junction closure gate, part 3: the production closure on a device's junction voxels against the reference.

Places StackDevices (averaging="subpixel_full") whose polygons have corners of several kinds (reflex, rotated
convex, acute) with vertical walls through the whole box (2.5D: no surface in any voxel), picks junction voxels
and compares, at each, through the production code path (``_interfaces`` -> ``_junction_inputs`` -> sub-cells ->
``periodic_cell_tensor``, blended with the laminate by omega as ``_complete_tensor`` does):

- the closure C (the voxel's own model on 4 x 4 x 4 sub-cells, laminate sub-cells, periodic Q1 cell problem)
  against the reference periodic cell tensor of the exact polygon in the voxel (reference_fem: interface-fitted,
  refined): C's discretization error;
- C, the laminate along the faces' leading direction and the blend actually written against the reference
  embedded tensor (the voxel at the center of a 5 x 5 patch of the exact polygon, Dirichlet, averaged over the
  voxel): the model (boundary-condition) error;
- with the corner model (``corner_lines=2``, the default) and with half-planes only (``corner_lines=0``).

Run (CPU is enough; jax float32 as in production):
    python benchmarks/fdtdx_subpixel/validate_closure.py [out.json]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference_fem import converged_2d  # noqa: E402

RES = 0.02
PER_SHAPE = 12


def rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def rotated_square(deg: float, half: float) -> list[tuple[float, float]]:
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return [tuple(float(v) for v in r @ np.array(p)) for p in [(-half, -half), (half, -half), (half, half), (-half, half)]]


SHAPES = {
    "L shape": [(-0.5, -0.4), (0.5, -0.4), (0.5, -0.1), (-0.15, -0.1), (-0.15, 0.4), (-0.5, 0.4)],
    "rotated square 30": rotated_square(30.0, 0.3),
    "triangle acute": [(-0.5, -0.3), (0.5, -0.3), (-0.1, 0.35)],
}
# the box is built on the polygon's bounding box: shifted against it by a fraction of a cell, so that the
# bounding box's corners (and the other vertices) lie inside voxels, not on their sides
SHIFT = (0.0073, 0.0041)


def device_junctions(polygon, corner_lines: int, picked=None):
    """Places the device; returns its junction voxels at mid height with omega, laminate, closure, residual."""
    import fdtdx
    import jax
    import jax.numpy as jnp

    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import _JUNCTION_HIGH, _JUNCTION_LOW, StackDevice, init_stack_params
    from gdsfactory.fdtdx_subpixel import from_six, junction_weight, laminate, periodic_cell_tensor
    from gdsfactory.technology import LayerLevel, LayerStack

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    mats = {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=12.0)}
    c = gf.Component()
    c.add_polygon(polygon, layer=gf.gpdk.LAYER.WG)
    ls = LayerStack(layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.4, material="si",
                                                mesh_order=1)})
    d = StackDevice.from_layer_stack(
        c, ls, mats, background="air", name="s", included_layers=["core"],
        boundary_offset=(0.1 + SHIFT[0], -0.1 + SHIFT[0], 0.1 + SHIFT[1], -0.1 + SHIFT[1], -0.1, 0.1),
        averaging="subpixel_full", corner_lines=corner_lines,
        diagnostics=False,
    )
    config = fdtdx.SimulationConfig(time=1e-15, grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6),
                                    dtype=jnp.float32)
    volume, constraints = d.get_simulation_volume()
    objects, _, params, _, _ = fdtdx.place_objects(
        object_list=[volume, d], config=config, constraints=constraints, key=jax.random.PRNGKey(0)
    )
    params = {**params, **init_stack_params(objects)}
    dev = objects["s"]
    nx, ny, nz = dev.grid_shape
    g = dev._interfaces(params["s"], None, True, (0.0, 0.0, 0.0))  # the voxels themselves
    names, eps_table = dev._material_table()
    omega, normal = junction_weight(from_six(g.direction), _JUNCTION_LOW, _JUNCTION_HIGH)
    omega, normal = np.asarray(omega), np.asarray(normal)
    k = nz // 2
    if picked is None:
        cand = np.argwhere(omega[:, :, k] > 0)
        picked = [tuple(int(v) for v in p) for p in cand[:: max(1, len(cand) // PER_SHAPE)][:PER_SHAPE]]
    if not picked:
        raise ValueError("no junction voxels at mid height")
    voxels = jnp.asarray([(i * ny + j) * nz + k for i, j in picked], dtype=jnp.int32)
    inputs, cells_of = dev._junction_inputs(g, voxels)
    tensors, residuals = jax.jit(lambda a: periodic_cell_tensor(cells_of(a)[1]))(inputs)
    f = np.moveaxis(np.asarray(g.weights), 0, -1)
    out = []
    for n_, (i, j) in enumerate(picked):
        lam = np.asarray(laminate(eps_table, jnp.asarray(f[i, j, k]), jnp.asarray(normal[i, j, k])), dtype=float)
        closure = np.asarray(tensors[n_], dtype=float)
        w = float(omega[i, j, k])
        out.append(dict(voxel=(i, j), omega=w, laminate=lam, closure=closure, written=(1 - w) * lam + w * closure,
                        residual=float(residuals[n_]),
                        f_si=float(f[i, j, k][names.index("si")]),
                        origin=(dev.center[0] + (i - nx / 2) * RES, dev.center[1] + (j - ny / 2) * RES)))
    return picked, out


def references(polygon, origin):
    """Periodic cell tensor and embedded (5 x 5 patch, Dirichlet) tensor of the exact polygon in a voxel."""
    import shapely
    import shapely.affinity
    from shapely.geometry import box

    eps = np.stack([np.eye(3), 12.0 * np.eye(3)])
    poly = shapely.Polygon(polygon)
    x0, y0 = origin

    def regions(size: int):
        lo = (size - 1) / 2  # stack -> patch units, the voxel at [lo, lo + 1]^2
        big = shapely.affinity.affine_transform(poly, [1 / RES, 0, 0, 1 / RES, -x0 / RES + lo, -y0 / RES + lo])
        square = box(0, 0, size, size)
        parts = [(p, 1) for p in shapely.get_parts(big.intersection(square)) if p.area > 1e-12]
        return parts + [(p, 0) for p in shapely.get_parts(square.difference(big)) if p.area > 1e-12]

    periodic, _, hist = converged_2d(regions(1), eps, "periodic", n=8, levels=3)
    embedded, _, hist_e = converged_2d(regions(5), eps, "dirichlet", n=8, levels=2, size=5, window=(2.0, 3.0))
    return periodic, rel(hist[-1], hist[-2]), embedded, rel(hist_e[-1], hist_e[-2])


def main(out: str | None) -> None:
    import jax

    print("versions:", dict(jax=jax.__version__, backend=jax.default_backend()), flush=True)
    rows = []
    t0 = time.time()
    for name, polygon in SHAPES.items():
        picked, with_corners = device_junctions(polygon, 2)
        _, half_planes = device_junctions(polygon, 0, picked)
        for a, b in zip(with_corners, half_planes, strict=True):
            periodic, per_change, embedded, emb_change = references(polygon, a["origin"])
            row = dict(shape=name, voxel=a["voxel"], f_si=round(a["f_si"], 4), ref_change=per_change,
                       embedded_change=emb_change, periodic_ref_vs_embedded=rel(periodic, embedded))
            for tag, r in (("corner", a), ("halfplane", b)):
                row.update({
                    f"{tag}_omega": round(r["omega"], 3), f"{tag}_residual": r["residual"],
                    f"{tag}_closure_vs_periodic_ref": rel(r["closure"], periodic),
                    f"{tag}_closure_vs_embedded": rel(r["closure"], embedded),
                    f"{tag}_laminate_vs_embedded": rel(r["laminate"], embedded),
                    f"{tag}_written_vs_embedded": rel(r["written"], embedded),
                })
            rows.append(row)
            print({k_: (round(x, 4) if isinstance(x, float) else x) for k_, x in row.items()}, flush=True)
            if out:
                Path(out).write_text(json.dumps(rows, indent=1))
    keys = [k for k in rows[0] if k.endswith(("_vs_periodic_ref", "_vs_embedded"))]
    for key in keys:
        values = np.asarray([r[key] for r in rows])
        print(f"{key:40s} median {np.median(values):.4f}  p90 {np.quantile(values, 0.9):.4f}  max {values.max():.4f}")
    print(f"elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
