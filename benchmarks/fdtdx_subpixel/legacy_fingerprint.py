"""Fingerprint of the legacy voxelization paths, to check a refactor changes no number (run with two checkouts).

Writes, for a few devices (corners of every kind, sub-samples, the "subpixel" tensor), the material weights and
the written inverse permittivity to an .npz; compare two runs with ``--compare a.npz b.npz``.

Run: PYTHONPATH=<checkout> python benchmarks/fdtdx_subpixel/legacy_fingerprint.py out.npz (not python -I, which
ignores PYTHONPATH; it prints which gdsfactory it imported)
"""

from __future__ import annotations

import sys

import numpy as np


def main(out: str) -> None:
    import fdtdx
    import jax
    import jax.numpy as jnp

    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import StackDevice, apply_stack, init_stack_params
    from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

    print("gdsfactory from", gf.__file__, flush=True)
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    layer = gf.gpdk.LAYER
    mats = {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=12.0), "sio2": fdtdx.Material(permittivity=2.1)}
    c = gf.Component()
    c.add_polygon([(0, 0), (1.2, 0), (1.2, 0.4), (0.7, 0.4), (0.5, 0.9), (0, 0.6)], layer=layer.WG)
    c.add_polygon([(0.3, -0.2), (1.0, -0.2), (1.0, 0.15), (0.3, 0.15)], layer=layer.SLAB150)
    c.add_polygon([(0.8, 0.1), (1.4, 0.3), (1.0, 0.7)], layer=layer.DEEP_ETCH)
    ls = LayerStack(
        layers={
            "core": LayerLevel(
                layer=LogicalLayer(layer=layer.WG) - LogicalLayer(layer=layer.DEEP_ETCH),
                derived_layer=LogicalLayer(layer=layer.WG), zmin=0.0, thickness=0.22,
                material="si", mesh_order=2, sidewall_angle=10.0,
            ),
            "slab": LayerLevel(layer=layer.SLAB150, zmin=0.0, thickness=0.09, material="sio2", mesh_order=3),
        }
    )
    results = {}
    for name, kwargs in {
        "default": {},
        "sub2": {"xy_subsamples": 2},
        "inverse": {"averaging": "inverse"},
        "nocorner": {"corner_lines": 0},
    }.items():
        d = StackDevice.from_layer_stack(
            c, ls, mats, background="air", name="s", included_layers=["core"],
            boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.1, -0.1), **kwargs,
        )
        config = fdtdx.SimulationConfig(time=1e-15, grid=fdtdx.core.grid.UniformGrid(spacing=0.02e-6), dtype=jnp.float32)
        volume, constraints = d.get_simulation_volume()
        objects, arrays, params, _, _ = fdtdx.place_objects(
            object_list=[volume, d], config=config, constraints=constraints, key=jax.random.PRNGKey(0)
        )
        params = {**params, **init_stack_params(objects)}
        placed = objects["s"]
        _, w = placed.material_weights(params["s"])
        arrays2, _, _ = apply_stack(arrays, objects, params, jax.random.PRNGKey(0))
        results[f"{name}_w"] = np.asarray(w)
        results[f"{name}_inv"] = np.asarray(arrays2.inv_permittivities[:, *placed.grid_slice])
        print(name, placed.grid_shape, placed.corner_columns, flush=True)
    np.savez(out, **results)


def compare(a: str, b: str) -> None:
    x, y = np.load(a), np.load(b)
    same = True
    for k in sorted(x.files):
        eq = np.array_equal(x[k], y[k])
        diff = float(np.abs(x[k] - y[k]).max()) if x[k].shape == y[k].shape else float("nan")
        print(f"{k}: identical={eq} max|diff|={diff:.3g}")
        same &= eq
    print("ALL IDENTICAL" if same else "DIFFERENT")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
