"""EM validation: corner scattering, a 2D grating of rotated rods (fdtdx StackDevice against meep).

A grating in the xy plane, invariant along z (one cell, periodic): rectangles ROD rotated by ``angle``, period
PERIOD along x (< wavelength: only the zeroth order propagates; the first orders' guided resonances lie below
1.2 um), a plane wave along +y with E along x (in plane: it crosses the oblique walls and the corners). The
zeroth-order transmission tau (the plane average of E over a period, divided by the run without rods on the same
grid; translation invariant, e^{-i w t} as both codes' DFTs) is recorded for E_x and E_z (the z coupling of
anisotropic rods converts polarization).

Rod materials (``--kind``): ``scalar`` (eps 12) or ``aniso`` (a rotated uniaxial tensor with an xz entry).

* ``fdtdx``: resolutions x averagings x angles x grid offsets (the rod moved by fractions of a cell);
  ``--tag`` names the gdsfactory checkout on the path (e.g. the baseline's averaging="subpixel_full").
* ``meep``: the same structure with meep's subpixel smoothing (anisotropic Kottke averaging), at resolutions
  (pixels/um); the finest two give a second-order Richardson estimate, the independent reference.

Run:
    python em_grating.py fdtdx out.json --res 40 20 10 5 --averaging subpixel_full subpixel --kind scalar
    python em_grating.py meep out.json --px 50 100 200 400 --kind scalar      (CPU)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

WAVELENGTH = 1.55  # um
PERIOD = 0.6
ROD = (0.3, 0.2)
BOX_Y = 0.4  # um: the device box spans y in [-BOX_Y, BOX_Y]
SOURCE_Y = -0.8
TRANS_Y = 0.8
PML = 30  # cells (fdtdx)


def rotation_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rod_tensor(kind: str) -> np.ndarray:
    if kind == "scalar":
        return 12.0 * np.eye(3)
    eps = rotation_z(25.0) @ np.diag([9.0, 12.0, 14.0]) @ rotation_z(25.0).T
    eps[0, 2] = eps[2, 0] = 0.8
    return eps


def rod_polygon(angle: float, center: tuple[float, float]) -> list[tuple[float, float]]:
    a = math.radians(angle)
    hx, hy = ROD[0] / 2, ROD[1] / 2
    return [(center[0] + math.cos(a) * x - math.sin(a) * y, center[1] + math.sin(a) * x + math.cos(a) * y)
            for x, y in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def fdtdx_run(res_nm: float, kind: str | None, averaging: str, angle: float, offset: tuple[float, float]) -> np.ndarray:
    """(2,) complex: plane-averaged (Ex, Ez) phasors at the transmission plane; ``kind=None``: no rods."""
    import fdtdx
    import jax
    import jax.numpy as jnp

    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import StackDevice, apply_stack, init_stack_params
    from gdsfactory.technology import LayerLevel, LayerStack

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    res = res_nm * 1e-3
    eps = rod_tensor(kind) if kind else np.eye(3)
    mats = {"air": fdtdx.Material(),
            "rod": fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row) for row in eps))}
    poly = rod_polygon(angle, (offset[0] * res, offset[1] * res))
    c = gf.Component()
    c.add_polygon(poly, layer=gf.gpdk.LAYER.WG)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    ls = LayerStack(layers={"rod": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=-1.0, thickness=2.0, material="rod",
                                               mesh_order=1)})
    device = StackDevice.from_layer_stack(
        c, ls, mats, background="air", name="stack", included_layers=["rod"],
        boundary_offset=(PERIOD / 2 - max(xs), -PERIOD / 2 - min(xs), BOX_Y - max(ys), -BOX_Y - min(ys),
                         res / 2 - 1.0, 1.0 - res / 2),
        averaging=averaging, diagnostics=False,
    )
    interior = 2 * (abs(SOURCE_Y) + 0.4)
    cells_y = 2 * math.ceil((interior / res + 2 * PML) / 2)
    config = fdtdx.SimulationConfig(time=140e-15, grid=fdtdx.core.grid.UniformGrid(spacing=res * 1e-6),
                                    dtype=jnp.float32, backend=jax.default_backend())
    volume = fdtdx.SimulationVolume(partial_real_shape=(PERIOD * 1e-6, cells_y * res * 1e-6, res * 1e-6),
                                    material=mats["air"])
    bound = fdtdx.BoundaryConfig(
        boundary_type_minx="periodic", boundary_type_maxx="periodic", boundary_type_minz="periodic",
        boundary_type_maxz="periodic", thickness_grid_miny=PML, thickness_grid_maxy=PML,
    )
    boundaries, b_constraints = fdtdx.boundary_objects_from_config(bound, volume)
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    profile = fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(wavelength=10 * WAVELENGTH * 1e-6),
                                         center_wave=wave)
    source = fdtdx.UniformPlaneSource(name="source", partial_grid_shape=(None, 1, None), direction="+",
                                      wave_character=wave, temporal_profile=profile,
                                      fixed_E_polarization_vector=(1.0, 0.0, 0.0))
    detector = fdtdx.PhasorDetector(name="trans", partial_grid_shape=(None, 1, None), wave_characters=(wave,),
                                    components=("Ex", "Ez"), reduce_volume=True, scaling_mode="pulse")

    def at(obj, y):
        return obj.place_relative_to(volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0),
                                     margins=(0, y * 1e-6, 0))

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device, source, detector, *boundaries.values()], config=config,
        constraints=[*b_constraints, at(device, 0.0), at(source, SOURCE_Y), at(detector, TRANS_Y)], key=key,
    )
    params = {**params, **init_stack_params(objects)}
    arrays, objects, info = apply_stack(arrays, objects, params, key)
    _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    status = info.get("stack_status", {}).get("stack")
    if status:
        print("  status", {k: float(v) for k, v in status.items()}, flush=True)
    return np.asarray(state.detector_states["trans"]["phasor"]).reshape(-1)[:2]


def meep_run(px: int, kind: str | None, angle: float) -> np.ndarray:
    """(2,) complex: period-averaged (Ex, Ez) DFT fields at the transmission plane; ``kind=None``: no rods."""
    import meep as mp

    dpml = 1.0
    sy = 2 * (abs(SOURCE_Y) + 0.4) + 2 * dpml
    fcen = 1 / WAVELENGTH
    geometry = []
    if kind:
        e = rod_tensor(kind)
        medium = mp.Medium(epsilon_diag=mp.Vector3(e[0, 0], e[1, 1], e[2, 2]),
                           epsilon_offdiag=mp.Vector3(e[0, 1], e[0, 2], e[1, 2]))
        geometry = [mp.Prism([mp.Vector3(x, y) for x, y in rod_polygon(angle, (0.0, 0.0))], height=mp.inf,
                             material=medium)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(PERIOD, sy, 0), boundary_layers=[mp.PML(dpml, direction=mp.Y)], geometry=geometry,
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2 * fcen), component=mp.Ex,
                           center=mp.Vector3(0, SOURCE_Y), size=mp.Vector3(PERIOD, 0))],
        resolution=px, k_point=mp.Vector3(), force_all_components=True,
    )
    dft = sim.add_dft_fields([mp.Ex, mp.Ez], fcen, 0, 1, center=mp.Vector3(0, TRANS_Y), size=mp.Vector3(PERIOD, 0))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, TRANS_Y), 1e-10))
    return np.array([np.mean(sim.get_dft_array(dft, c, 0)) for c in (mp.Ex, mp.Ez)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code", choices=["fdtdx", "meep"])
    parser.add_argument("out")
    parser.add_argument("--res", type=float, nargs="+", default=[40.0, 20.0, 10.0, 5.0], help="fdtdx (nm)")
    parser.add_argument("--px", type=int, nargs="+", default=[50, 100, 200, 400], help="meep (pixels/um)")
    parser.add_argument("--averaging", nargs="+", default=["subpixel_full"])
    parser.add_argument("--kind", default="scalar", choices=["scalar", "aniso"])
    parser.add_argument("--angles", type=float, nargs="+", default=[30.0, 45.0])
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.0, 0.3, 0.5, 0.6, 0.2],
                        help="pairs (x, y) in cells")
    parser.add_argument("--tag", default="worktree")
    args = parser.parse_args()
    offsets = list(zip(args.offsets[::2], args.offsets[1::2], strict=True))
    rows = []
    t0 = time.time()
    if args.code == "fdtdx":
        import jax

        import gdsfactory

        print("versions:", dict(jax=jax.__version__, gdsfactory=gdsfactory.__file__, backend=jax.default_backend()),
              flush=True)
        for res in args.res:
            empty = fdtdx_run(res, None, "subpixel", 0.0, (0.0, 0.0))
            for averaging in args.averaging:
                for angle in args.angles:
                    for offset in offsets:
                        tau = fdtdx_run(res, args.kind, averaging, angle, offset) / empty[0]
                        rows.append(dict(code="fdtdx", tag=args.tag, kind=args.kind, averaging=averaging, res_nm=res,
                                         angle=angle, offset=offset, tau=[[float(v.real), float(v.imag)] for v in tau]))
                        print(f"{args.tag} {averaging:14s} res {res:5.1f} angle {angle:4.1f} offset {offset}: "
                              f"tau_x {tau[0]:.5f} tau_z {tau[1]:.5f}  {time.time() - t0:.0f} s", flush=True)
                        Path(args.out).write_text(json.dumps(rows, indent=1))
    else:
        import meep as mp

        print("versions:", dict(meep=mp.__version__), flush=True)
        for px in args.px:
            empty = meep_run(px, None, 0.0)
            for angle in args.angles:
                tau = meep_run(px, args.kind, angle) / empty[0]
                rows.append(dict(code="meep", kind=args.kind, px=px, angle=angle,
                                 tau=[[float(v.real), float(v.imag)] for v in tau]))
                print(f"meep px {px} angle {angle}: tau_x {tau[0]:.6f} tau_z {tau[1]:.6f}  {time.time() - t0:.0f} s",
                      flush=True)
                Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
