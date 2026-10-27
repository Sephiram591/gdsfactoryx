"""EM validation: an anisotropic slab in fdtdx against the Berreman transfer matrix (berreman.py).

A 1D problem in a 3D run: 4 x 4 cells periodic in x and y, PML in z. A slab (a StackDevice level on the whole
wafer, its z faces at chosen fractions of a cell) sits in air; a TFSF plane-wave Gaussian pulse (x or y
polarized) is launched along +z below it; plane-averaged phasors of (Ex, Ey) are recorded above the slab
(transmission, two planes) and below the source (reflection). Each run is divided by a run without the slab on
the same grid, which gives the Jones matrices at the center wavelength, compared with berreman.jones:

* ``t_FDTD = t exp(-i k d)`` (the slab replaces a path of length d in air), all four entries, so polarization
  conversion is checked too; ``|r|`` entry by entry; energy ``sum_i |r_ij|^2 + |t_ij|^2 = 1`` per incident
  polarization j (lossless, air on both sides).
* the phase convention of fdtdx's phasors (e^{+i w t} sums, so e^{-i w t} fields, as in berreman.py) and the
  numerical dispersion of air are measured with the two transmission planes of the run without the slab.

Cases (``--cases``):
    full                rotated uniaxial tensor with an xz entry, averaging="subpixel_full"
    staircase           the same, faces snapped to the nearest cell boundary (what voxel-center sampling
                        paints), averaging="subpixel_full" (all voxels homogeneous): the first-order baseline
    diag-<averaging>    diagonal tensor diag(9, 12, 14) with averaging "inverse", "subpixel" or "subpixel_full"

``--grad`` also checks d|t_xx|^2/dd and d|t_yx|^2/dd (d = the slab's thickness parameter) through the FDTD
(jax.grad, checkpointed) against central finite differences of the same FDTD and against the transfer matrix.

Run on a GPU node from the checkout:
    python benchmarks/fdtdx_subpixel/em_slab.py out.json --res 40 20 10 5 [--cases ...] [--offsets ...] [--grad]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from berreman import jones

WAVELENGTH = 1.55  # um
THICKNESS = 0.137  # um: not a multiple of any grid spacing used
BOX = (-0.32, 0.64)  # um: the device box in stack z, whole cells at 40, 20, 10, 5 nm; the slab starts near 0
PML = 30  # cells
SECOND_PLANE = 0.2  # um above the transmission plane


def rotation_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def full_tensor() -> np.ndarray:
    eps = rotation_z(25.0) @ np.diag([9.0, 12.0, 14.0]) @ rotation_z(25.0).T
    eps[0, 2] = eps[2, 0] = 0.8
    return eps


DIAG = np.diag([9.0, 12.0, 14.0])


def case_setup(case: str) -> tuple[np.ndarray, str, bool]:
    """(tensor, averaging, staircase) of a case."""
    if case == "full":
        return full_tensor(), "subpixel_full", False
    if case == "staircase":
        return full_tensor(), "subpixel_full", True
    if case.startswith("diag-"):
        return DIAG, case[len("diag-"):], False
    raise ValueError(f"unknown case {case!r}")


def slab_faces(offset: float, res: float, staircase: bool) -> tuple[float, float]:
    """(bottom, thickness) in um: bottom at ``offset`` cells above z = 0 (a cell boundary)."""
    z0, z1 = offset * res, offset * res + THICKNESS
    if staircase:
        z0, z1 = round(z0 / res) * res, round(z1 / res) * res
    return z0, z1 - z0


class Slab:
    """One placed simulation (a resolution, a case, an offset, a polarization); ``None`` tensor = no slab."""

    def __init__(self, res_nm: float, eps: np.ndarray | None, averaging: str, z0: float, thickness: float,
                 pol: int, gradient: bool = False, backward: bool = False) -> None:
        import fdtdx
        import jax
        import jax.numpy as jnp

        import gdsfactory as gf
        from gdsfactory.fdtdx_stack import StackDevice, init_stack_params
        from gdsfactory.technology import LayerLevel, LayerStack

        gf.gpdk.PDK.activate()
        gf.snap.SNAP_ENABLED = False
        res = res_nm * 1e-3
        lateral = 4 * res
        slab = fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row) for row in
                                                 (np.eye(3) if eps is None else eps)))
        mats = {"air": fdtdx.Material(), "slab": slab}
        half = lateral / 2
        c = gf.Component()
        c.add_polygon([(-half, -half), (half, -half), (half, half), (-half, half)], layer=gf.gpdk.LAYER.WG)
        ls = LayerStack(layers={"slab": LayerLevel(layer=gf.gpdk.LAYER.WAFER, zmin=z0, thickness=thickness,
                                                    material="slab", mesh_order=1)})
        self.device = StackDevice.from_layer_stack(
            c, ls, mats, background="air", name="stack", included_layers=["slab"],
            boundary_offset=(0.0, 0.0, 0.0, 0.0, BOX[1] - (z0 + thickness), BOX[0] - z0),
            averaging=averaging, diagnostics=False,
        )
        # layout in um from the volume's center: the box (its center at z_dev), the source 0.4 below it, the
        # transmission planes 0.4 and 0.6 above it, the reflection plane 0.3 below the source; backward: the
        # same mirrored about the box's center, the wave going down
        z_dev = 0.5 * (BOX[0] + BOX[1])
        sign = -1.0 if backward else 1.0
        z_src = z_dev - sign * (0.5 * (BOX[1] - BOX[0]) + 0.4)
        z_t = z_dev + sign * (0.5 * (BOX[1] - BOX[0]) + 0.4)
        z_t2 = z_t + sign * SECOND_PLANE
        z_r = z_src - sign * 0.3
        # the same volume both ways
        reach = 0.5 * (BOX[1] - BOX[0]) + 0.4 + max(SECOND_PLANE, 0.3)
        interior = 2 * (abs(z_dev) + reach) + 0.2
        cells = 2 * math.ceil((interior / res + 2 * PML) / 2)
        kwargs = {}
        if gradient:
            kwargs["gradient_config"] = fdtdx.GradientConfig(method="checkpointed", num_checkpoints=64)
        config = fdtdx.SimulationConfig(
            time=140e-15, grid=fdtdx.core.grid.UniformGrid(spacing=res * 1e-6), dtype=jnp.float32,
            backend=jax.default_backend(), **kwargs,
        )
        volume = fdtdx.SimulationVolume(partial_real_shape=(lateral * 1e-6, lateral * 1e-6, cells * res * 1e-6),
                                        material=mats["air"])
        bound = fdtdx.BoundaryConfig(
            boundary_type_minx="periodic", boundary_type_maxx="periodic", boundary_type_miny="periodic",
            boundary_type_maxy="periodic", thickness_grid_minz=PML, thickness_grid_maxz=PML,
        )
        boundaries, b_constraints = fdtdx.boundary_objects_from_config(bound, volume)
        wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
        profile = fdtdx.GaussianPulseProfile(
            spectral_width=fdtdx.WaveCharacter(wavelength=10 * WAVELENGTH * 1e-6), center_wave=wave
        )
        source = fdtdx.UniformPlaneSource(
            name="source", partial_grid_shape=(None, None, 1), direction="-" if backward else "+", wave_character=wave,
            temporal_profile=profile, fixed_E_polarization_vector=(1.0, 0.0, 0.0) if pol == 0 else (0.0, 1.0, 0.0),
        )

        def plane(name):
            return fdtdx.PhasorDetector(
                name=name, partial_grid_shape=(None, None, 1), wave_characters=(wave,), components=("Ex", "Ey"),
                reduce_volume=True, scaling_mode="pulse",
            )

        planes = {"trans": z_t, "trans2": z_t2, "refl": z_r}
        detectors = [plane(n) for n in planes]

        def at(obj, z):
            return obj.place_relative_to(volume, axes=(0, 1, 2), own_positions=(0, 0, 0),
                                         other_positions=(0, 0, 0), margins=(0, 0, z * 1e-6))

        constraints = [*b_constraints, at(self.device, z_dev), at(source, z_src),
                       *(at(d, planes[d.name]) for d in detectors)]
        self.key = jax.random.PRNGKey(0)
        self.objects, self.arrays, params, self.config, _ = fdtdx.place_objects(
            object_list=[volume, self.device, source, *detectors, *boundaries.values()], config=config,
            constraints=constraints, key=self.key,
        )
        self.params = {**params, **init_stack_params(self.objects)}

    def phasors(self, stack_params=None):
        """(3, 2) complex (jax): rows trans, trans2, refl; columns Ex, Ey (plane averages)."""
        import fdtdx
        import jax.numpy as jnp

        from gdsfactory.fdtdx_stack import apply_stack

        params = self.params if stack_params is None else {**self.params, "stack": stack_params}
        arrays, objects, _ = apply_stack(self.arrays, self.objects, params, self.key)
        _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=self.config, key=self.key,
                                  show_progress=False)
        rows = [jnp.reshape(state.detector_states[n]["phasor"], (-1,))[:2] for n in ("trans", "trans2", "refl")]
        return jnp.stack(rows)


def jones_from_runs(with_slab: list[np.ndarray], without: list[np.ndarray]):
    """T (2, 2), |r| (2, 2) from the per-polarization phasors of runs with and without the slab."""
    t = np.zeros((2, 2), dtype=complex)
    r_mag = np.zeros((2, 2))
    for pol in (0, 1):
        inc = without[pol][0, pol]  # the incident wave at the transmission plane
        t[:, pol] = with_slab[pol][0] / inc
        r_mag[:, pol] = np.abs(with_slab[pol][2] / inc)
    return t, r_mag


def reference(thickness: float, eps: np.ndarray):
    r, t = jones([(eps, thickness)], 1.0, 1.0, WAVELENGTH)
    return r, t * np.exp(-2j * np.pi / WAVELENGTH * thickness)


def sweep(args) -> list[dict]:
    rows = []
    t_start = time.time()
    k0 = 2 * math.pi / WAVELENGTH
    for res in args.res:
        air = {}
        for backward in (False, True) if args.reciprocity else (False,):
            air[backward] = [np.asarray(Slab(res, None, "subpixel_full", 0.0, THICKNESS, pol, backward=backward)
                                        .phasors()) for pol in (0, 1)]
        # convention and dispersion: phase of trans2 / trans in air (e^{-i w t} fields give +k L)
        ratio = air[False][0][1, 0] / air[False][0][0, 0]
        k_num = float(np.angle(ratio)) / SECOND_PLANE
        print(f"res {res} nm: air phase k_num / k0 = {k_num / k0:.6f}, |ratio| - 1 = {abs(ratio) - 1:.2e}",
              flush=True)
        for case in args.cases:
            eps, averaging, staircase = case_setup(case)
            for offset in args.offsets:
                z0, thickness = slab_faces(offset, res * 1e-3, staircase)
                runs = [np.asarray(Slab(res, eps, averaging, z0, thickness, pol).phasors()) for pol in (0, 1)]
                t, r_mag = jones_from_runs(runs, air[False])
                r_ref, t_ref = reference(THICKNESS, eps)  # always the true slab
                energy = np.abs((np.abs(t) ** 2 + r_mag**2).sum(axis=0) - 1).max()
                row = dict(
                    case=case, averaging=averaging, res_nm=res, offset=offset, k_num_over_k0=k_num / k0,
                    t_err=float(np.abs(t - t_ref).max()), r_err=float(np.abs(r_mag - np.abs(r_ref)).max()),
                    energy_err=float(energy), t_yx=float(abs(t[1, 0])), t_yx_ref=float(abs(t_ref[1, 0])),
                    # a reciprocal slab with a symmetric reduced tensor has a symmetric Jones t
                    t_asymmetry=float(abs(t[0, 1] - t[1, 0])),
                    t=[[[float(v.real), float(v.imag)] for v in r] for r in t],
                    t_ref=[[[float(v.real), float(v.imag)] for v in r] for r in t_ref],
                )
                if args.reciprocity:
                    back = [np.asarray(Slab(res, eps, averaging, z0, thickness, pol, backward=True).phasors())
                            for pol in (0, 1)]
                    t_back, _ = jones_from_runs(back, air[True])
                    row["reciprocity_err"] = float(np.abs(t_back - t.T).max())
                    row["t_back"] = [[[float(v.real), float(v.imag)] for v in r] for r in t_back]
                rows.append(row)
                print(f"{case:22s} res {res:5.1f} offset {offset:.2f}: t_err {row['t_err']:.3e} "
                      f"r_err {row['r_err']:.3e} energy {row['energy_err']:.1e} |t_yx| {row['t_yx']:.5f} "
                      f"(ref {row['t_yx_ref']:.5f}) asym {row['t_asymmetry']:.1e}"
                      + (f" recip {row['reciprocity_err']:.1e}" if args.reciprocity else "")
                      + f"  {time.time() - t_start:.0f} s", flush=True)
                Path(args.out).write_text(json.dumps(rows, indent=1))
    return rows


def gradient_check(res: float, offset: float, step: float) -> dict:
    """d|t_xx|^2/dd and d|t_yx|^2/dd: jax.grad through the FDTD, central differences, transfer matrix."""
    import jax
    import jax.numpy as jnp

    eps = full_tensor()
    z0, thickness = slab_faces(offset, res * 1e-3, False)
    inc = complex(np.asarray(Slab(res, None, "subpixel_full", 0.0, THICKNESS, 0).phasors())[0, 0])
    sim = Slab(res, eps, "subpixel_full", z0, thickness, 0, gradient=True)
    key = "thickness/slab"
    assert key in sim.params["stack"], sorted(sim.params["stack"])

    def powers(d):
        ph = sim.phasors({**sim.params["stack"], key: d})
        return jnp.abs(ph[0] / inc) ** 2  # (|t_xx|^2, |t_yx|^2)

    d0 = jnp.asarray(thickness, jnp.float32)
    t0 = time.time()
    value, vjp = jax.vjp(powers, d0)
    grads = [float(vjp(jnp.eye(2, dtype=value.dtype)[i])[0]) for i in range(2)]
    t_grad = time.time() - t0
    fd = (np.asarray(powers(d0 + step)) - np.asarray(powers(d0 - step))) / (2 * step)

    def tmm(d):
        _, t = jones([(eps, d)], 1.0, 1.0, WAVELENGTH)
        return np.abs(np.array([t[0, 0], t[1, 0]])) ** 2

    h = 1e-6
    ref = (tmm(thickness + h) - tmm(thickness - h)) / (2 * h)
    out = dict(res_nm=res, offset=offset, step=step, value=np.asarray(value).tolist(), grad=grads,
               fd=fd.tolist(), tmm=ref.tolist(), tmm_value=tmm(thickness).tolist(), seconds=t_grad,
               grad_vs_fd=[abs(g - f) / max(abs(f), 1e-12) for g, f in zip(grads, fd)],
               grad_vs_tmm=[abs(g - f) / max(abs(f), 1e-12) for g, f in zip(grads, ref)])
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--res", type=float, nargs="+", default=[40.0, 20.0, 10.0, 5.0])
    parser.add_argument("--cases", nargs="+", default=["full", "staircase", "diag-inverse", "diag-subpixel",
                                                         "diag-subpixel_full"])
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--grad", action="store_true", help="only the gradient check (at each --res, first offset)")
    parser.add_argument("--reciprocity", action="store_true", help="also run each case downward: t_back = t^T")
    parser.add_argument("--step", type=float, default=0.002, help="finite-difference step (um)")
    args = parser.parse_args()
    import fdtdx
    import jax

    import gdsfactory

    print("versions:", dict(jax=jax.__version__, fdtdx=getattr(fdtdx, "__version__", "?"),
                            gdsfactory=gdsfactory.__file__, backend=jax.default_backend()), flush=True)
    if args.grad:
        rows = [gradient_check(res, args.offsets[0], args.step) for res in args.res]
        Path(args.out).write_text(json.dumps(rows, indent=1))
    else:
        sweep(args)


if __name__ == "__main__":
    main()
