"""MEEP reference for fdtdx_elevator_coupler.py: same component, LayerStack, source and monitors.

1. Export the gdsfactory geometry once (one process; imports gdsfactory/fdtdx):
       python examples/meep_elevator_coupler.py export
2. Run MEEP for one overlap (MPI; imports only meep/numpy):
       mpirun -np 48 python examples/meep_elevator_coupler.py run 1.00
"""

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "output"
OVERLAPS = np.linspace(1, 6, 4)


def export(resolution: float) -> None:
    """Writes the component's polygons and the LayerStack levels to JSON, one file per overlap."""
    sys.path.insert(0, str(Path(__file__).parent))
    import fdtdx_elevator_coupler as ex

    import gdsfactory as gf

    OUT.mkdir(exist_ok=True)
    wafer = gf.get_layer("WAFER")
    for overlap in OVERLAPS:
        # the polygons the fdtdx StackDevice draws: the component plus its port extensions
        stack = ex.build_stack(overlap)
        polygons = {
            index: np.split(np.asarray(v), np.cumsum(sizes)[:-1])
            for (_, index), v, sizes in zip(stack.source_layers, stack.source_vertices, stack.source_sizes, strict=True)
        }
        levels = []
        for name, lv in ex.LAYER_STACK.layers.items():
            index = gf.get_layer(lv.layer.layer)
            levels.append({
                "name": name,
                "zmin": lv.zmin,
                "thickness": lv.thickness,
                "mesh_order": lv.mesh_order,
                "index": float(np.sqrt(ex.MATERIALS[lv.material].permittivity[0])),
                "polygons": None if index == wafer else [np.asarray(p).tolist() for p in polygons.get(index, [])],
            })
        geometry = {
            "levels": levels,
            "domain": ex.domain(overlap),
            "wavelength": ex.WAVELENGTH,
            "pml": ex.PML,
            "resolution": resolution,
            "background_index": float(np.sqrt(ex.MATERIALS["air"].permittivity[0])),
        }
        (OUT / f"geometry_{overlap:.2f}.json").write_text(json.dumps(geometry))


def run(overlap: float, symmetry: bool = True, resolution: float | None = None, tag: str = "meep") -> None:
    import meep as mp

    g = json.loads((OUT / f"geometry_{overlap:.2f}.json").read_text())
    (x0, x1), (y0, y1), (z0, z1) = g["domain"]
    res, pml, f0 = resolution or g["resolution"], g["pml"], 1 / g["wavelength"]
    cell = mp.Vector3(x1 - x0, y1 - y0, z1 - z0)
    center = mp.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)

    geometry = []  # MEEP: later objects win, and lower mesh_order has priority
    for lv in sorted(g["levels"], key=lambda lv: -lv["mesh_order"]):
        medium = mp.Medium(index=lv["index"])
        if lv["polygons"] is None:  # wafer-wide level
            zc = lv["zmin"] + lv["thickness"] / 2
            size = mp.Vector3(mp.inf, mp.inf, abs(lv["thickness"]))
            geometry.append(mp.Block(center=mp.Vector3(center.x, center.y, zc), size=size, material=medium))
        else:
            for p in lv["polygons"]:
                vertices = [mp.Vector3(x, y, lv["zmin"]) for x, y in p]
                geometry.append(mp.Prism(vertices, height=lv["thickness"], material=medium))

    plane = mp.Vector3(0, cell.y - 2 * pml, cell.z - 2 * pml)
    at_x = lambda x: mp.Vector3(x, center.y, center.z)  # noqa: E731
    # same Gaussian envelope as fdtdx: sigma_t = 1 / (2 pi * 0.05 f0)
    pulse = mp.GaussianSource(f0, fwidth=2 * np.pi * 0.05 * f0)
    parity = mp.ODD_Y  # TE0 (E_y even about y = 0) is odd as a vector field
    source = mp.EigenModeSource(
        pulse, center=at_x(x0 + pml + 0.2), size=plane, eig_band=1, eig_parity=parity, eig_match_freq=True
    )
    sim = mp.Simulation(
        cell_size=cell,
        geometry_center=center,
        geometry=geometry,
        default_material=mp.Medium(index=g["background_index"]),
        sources=[source],
        resolution=1 / res,
        boundary_layers=[mp.PML(pml)],
        symmetries=[mp.Mirror(mp.Y, phase=-1)] if symmetry else [],
    )
    mode_in = sim.add_mode_monitor(f0, 0, 1, mp.ModeRegion(center=at_x(x0 + pml + 0.5), size=plane))
    mode_out = sim.add_mode_monitor(f0, 0, 1, mp.ModeRegion(center=at_x(x1 - pml - 0.5), size=plane))
    slice_center, slice_size = mp.Vector3(center.x, 0, center.z), mp.Vector3(cell.x, 0, cell.z)
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], f0, 0, 1, center=slice_center, size=slice_size)
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-5))

    a_in = sim.get_eigenmode_coefficients(mode_in, [1], eig_parity=parity).alpha[0, 0, 0]
    a_out = sim.get_eigenmode_coefficients(mode_out, [1], eig_parity=parity).alpha[0, 0, 0]
    E2 = sum(np.abs(np.squeeze(sim.get_dft_array(dft, c, 0))) ** 2 for c in (mp.Ex, mp.Ey, mp.Ez))
    x, _, z, _ = sim.get_array_metadata(dft_cell=dft)
    eps = np.squeeze(sim.get_array(center=slice_center, size=slice_size, component=mp.Dielectric))
    ex_, _, ez_, _ = sim.get_array_metadata(center=slice_center, size=slice_size)
    if mp.am_master():
        T = float(abs(a_out / a_in) ** 2)
        print(f"MEEP overlap {overlap:.2f} um, resolution {res * 1e3:.1f} nm: T = {T:.4f}", flush=True)
        suffix = "" if symmetry else "_nosym"
        np.savez(
            OUT / f"{tag}_overlap_{overlap:.2f}{suffix}.npz",
            T=T, x=np.asarray(x), z=np.asarray(z), E2=E2, eps=eps, eps_x=np.asarray(ex_), eps_z=np.asarray(ez_),
        )


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(float(sys.argv[2]) if len(sys.argv) > 2 else 0.02)
    else:
        args = sys.argv[3:]
        res = float(args[args.index("--res") + 1]) if "--res" in args else None
        tag = args[args.index("--tag") + 1] if "--tag" in args else "meep"
        run(float(sys.argv[2]), symmetry="--nosym" not in args, resolution=res, tag=tag)
