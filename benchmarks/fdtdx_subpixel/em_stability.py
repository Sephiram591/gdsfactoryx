"""EM validation: long-time stability of averaging="subpixel_full" in a closed lossless box.

fdtdx couples the off-diagonal entries of the inverse permittivity by averaging neighboring field components,
which is not symmetric where the tensor varies in space; with averaged interface voxels (rows from different
Yee positions, junction closures) a closed lossless domain is where growth would show. The device: a straight
waveguide with sloped sidewalls, a slab, a deep etch, a rotated square pillar and a pad, every material a
different lossless full tensor; the box is periodic on all six faces. A short dipole pulse excites it; the total
energy (fdtdx's EnergyDetector over the whole volume) is recorded every ``--every`` steps for ``--time``. Stable:
after the pulse the energy stays bounded (the collocated energy of a staggered scheme oscillates a little, it
does not grow).

Run on a GPU node from the checkout:
    python benchmarks/fdtdx_subpixel/em_stability.py out.json [--time 2e-12] [--every 1000] [--averaging subpixel_full]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

RES = 0.02  # um


def rotation(axis: int, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    i, j = [k for k in range(3) if k != axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def tensors() -> dict[str, np.ndarray]:
    si = rotation(2, 25.0) @ np.diag([9.0, 12.0, 14.0]) @ rotation(2, 25.0).T
    si[0, 2] = si[2, 0] = 0.8
    return {
        "air": np.eye(3),
        "sio2": rotation(0, 20.0) @ np.diag([2.1, 2.6, 3.2]) @ rotation(0, 20.0).T,
        "si": si,
        "pad": rotation(1, 35.0) @ np.diag([4.0, 6.0, 8.0]) @ rotation(1, 35.0).T,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--time", type=float, default=2e-12, help="simulated time (s)")
    parser.add_argument("--every", type=int, default=1000, help="steps between energy records")
    parser.add_argument("--averaging", default="subpixel_full")
    parser.add_argument("--diagonal", action="store_true", help="the tensors' diagonals only (legacy baseline)")
    args = parser.parse_args()

    import fdtdx
    import jax
    import jax.numpy as jnp

    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import StackDevice, apply_stack, init_stack_params
    from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    L = gf.gpdk.LAYER
    eps = tensors()
    if args.diagonal:
        eps = {k: np.diag(np.diag(v)) for k, v in eps.items()}
    mats = {k: fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row) for row in v)) for k, v in eps.items()}
    c = gf.Component()
    c.add_ref(gf.components.straight(length=2.0, width=0.5))
    c.add_polygon([(0.0, -0.4), (2.0, -0.4), (2.0, 0.4), (0.0, 0.4)], layer=L.SLAB150)
    c.add_polygon([(1.4, -0.5), (2.0, -0.5), (2.0, 0.5), (1.4, 0.5)], layer=L.DEEP_ETCH)
    c.add_polygon([(0.2, 0.6), (0.6, 0.6), (0.6, 0.7), (0.2, 0.7)], layer=L.M1)
    a = math.radians(30.0)
    pillar = [(0.9 + 0.15 * (math.cos(a) * x - math.sin(a) * y), -0.75 + 0.15 * (math.sin(a) * x + math.cos(a) * y))
              for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    c.add_polygon(pillar, layer=L.WG)
    ls = LayerStack(layers={
        "box": LayerLevel(layer=L.WAFER, zmin=-0.3, thickness=0.3, material="sio2", mesh_order=9),
        "core": LayerLevel(layer=LogicalLayer(layer=L.WG) - LogicalLayer(layer=L.DEEP_ETCH),
                           derived_layer=LogicalLayer(layer=L.WG), zmin=0.0, thickness=0.22, material="si",
                           mesh_order=2, sidewall_angle=10.0, width_to_z=0.5),
        "slab": LayerLevel(layer=L.SLAB150, zmin=0.0, thickness=0.08, material="si", mesh_order=3),
        "clad": LayerLevel(layer=L.WAFER, zmin=0.0, thickness=0.5, material="sio2", mesh_order=10),
        "pad": LayerLevel(layer=L.M1, zmin=0.4, thickness=0.1, material="pad", mesh_order=1),
    })
    device = StackDevice.from_layer_stack(
        c, ls, mats, background="air", name="stack", included_layers=list(ls.layers),
        boundary_offset=(0.3, -0.3, 0.3, -0.3, 0.1, 0.0), averaging=args.averaging, diagnostics=False,
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    config = fdtdx.SimulationConfig(time=args.time, grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6),
                                    dtype=jnp.float32, backend=jax.default_backend())
    steps = config.time_steps_total
    volume, volume_constraints = device.get_simulation_volume(material=mats["air"])  # exactly the device box
    bound = fdtdx.BoundaryConfig(**{f"boundary_type_{s}{ax}": "periodic" for s in ("min", "max") for ax in "xyz"})
    boundaries, b_constraints = fdtdx.boundary_objects_from_config(bound, volume)
    profile = fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(wavelength=5 * 1.55e-6), center_wave=wave)
    source = fdtdx.PointDipoleSource(name="dipole", partial_grid_shape=(1, 1, 1), wave_character=wave,
                                     temporal_profile=profile, polarization=1)
    detector = fdtdx.EnergyDetector(name="energy", reduce_volume=True, plot=False,
                                    switch=fdtdx.OnOffSwitch(fixed_on_time_steps=list(range(0, steps, args.every))))
    constraints = [
        *b_constraints, *volume_constraints, detector.same_size(volume), detector.place_at_center(volume),
        source.place_relative_to(volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0),
                                 margins=(-0.4e-6, 0.0, 0.0)),
    ]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device, source, detector, *boundaries.values()], config=config,
        constraints=constraints, key=key,
    )
    params = {**params, **init_stack_params(objects)}
    t0 = time.time()
    arrays, objects, info = apply_stack(arrays, objects, params, key)
    status = {k: float(v) for k, v in info["stack_status"].get("stack", {}).items()}
    _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    energy = np.asarray(state.detector_states["energy"]["energy"]).reshape(-1)
    seconds = time.time() - t0
    recorded = list(range(0, steps, args.every))[: len(energy)]
    # after the pulse (the source's envelope is negligible after 12 sigma)
    sigma_t = 1.0 / (2 * math.pi * (3e8 / (5 * 1.55e-6)))
    after = [i for i, s in enumerate(recorded) if s * config.time_step_duration > 12 * sigma_t]
    tail = energy[after]
    out = dict(averaging=args.averaging, diagonal=args.diagonal, steps=steps, every=args.every,
               grid=list(objects["stack"].grid_shape), seconds=seconds, status=status,
               energy=energy.tolist(), steps_recorded=recorded,
               first_after_pulse=float(tail[0]), max_over_first=float(tail.max() / tail[0]),
               min_over_first=float(tail.min() / tail[0]), last_over_first=float(tail[-1] / tail[0]),
               finite=bool(np.all(np.isfinite(energy))))
    print({k: v for k, v in out.items() if k not in ("energy", "steps_recorded")}, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
