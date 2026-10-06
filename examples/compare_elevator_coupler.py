"""Compares the fdtdx and MEEP elevator-coupler sweeps: transmissions, field maps, geometry.

    python examples/compare_elevator_coupler.py            # reads examples/output/*.npz
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

OUT = Path(__file__).parent / "output"
OVERLAPS = np.linspace(1, 6, 4)
Z_VIEW = (-0.7, 0.6)  # um, the part of the xz plane shown


def load(prefix: str, overlap: float) -> dict:
    d = dict(np.load(OUT / f"{prefix}_overlap_{overlap:.2f}.npz"))
    d["E2"] = np.squeeze(d["E2"])
    d["eps"] = np.squeeze(d["eps"])
    # normalize to the input mode: peak |E|^2 in the straight input section,
    # between the source plane (x = -0.8) and the start of the taper (x = 0)
    x = d["x"]
    d["E2"] = d["E2"] / d["E2"][(x > -0.6) & (x < -0.1)].max()
    return d


def on_grid(d: dict, x: np.ndarray, z: np.ndarray, key: str = "E2") -> np.ndarray:
    xs, zs = (d["eps_x"], d["eps_z"]) if key == "eps" and "eps_x" in d else (d["x"], d["z"])
    f = RegularGridInterpolator((np.asarray(xs), np.asarray(zs)), d[key], bounds_error=False, fill_value=None)
    X, Z = np.meshgrid(x, z, indexing="ij")
    return np.clip(f((X, Z)), 0, None) if key == "E2" else f((X, Z))


def guide_power(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """|E|^2 integrated over the upper (z > -0.06) and lower (z < -0.06) guide regions."""
    z = d["z"]
    upper = d["E2"][:, (z > -0.06) & (z < 0.45)].sum(axis=1)
    lower = d["E2"][:, (z > -0.45) & (z <= -0.06)].sum(axis=1)
    return upper, lower


def main(prefixes: tuple[str, str] = ("fdtdx", "meep")) -> None:
    rows = [o for o in OVERLAPS if all((OUT / f"{p}_overlap_{o:.2f}.npz").exists() for p in prefixes)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(18, 2.2 * len(rows)), squeeze=False)
    report = []
    for row, overlap in zip(axes, rows):
        a, b = (load(p, overlap) for p in prefixes)
        x, z = a["x"], a["z"]
        keep = (z >= Z_VIEW[0]) & (z <= Z_VIEW[1])
        interior = (x > -0.6) & (x < x.max() - 0.5)  # after the source, before the PML
        b_on_a = on_grid(b, x, z)
        err = np.linalg.norm((a["E2"] - b_on_a)[interior][:, keep]) / np.linalg.norm(b_on_a[interior][:, keep])
        report.append((overlap, float(a["T"]), float(b["T"]), float(err)))
        extent = (x.min(), x.max(), Z_VIEW[0], Z_VIEW[1])
        style = dict(extent=extent, origin="lower", aspect="auto")
        row[0].imshow(a["eps"][:, keep].T, cmap="Greys", **style)
        row[0].contour(x, z[keep], on_grid(b, x, z[keep], "eps").T, levels=[3.0], colors="r", linewidths=0.6)
        row[0].set_title(f"overlap {overlap:.2f} um: eps (fdtdx, grey) / MEEP n=1.73 contour (red)", fontsize=9)
        for ax, d, name in ((row[1], a["E2"], prefixes[0]), (row[2], b_on_a, prefixes[1])):
            ax.imshow(np.sqrt(d[:, keep]).T, cmap="inferno", vmin=0, vmax=1.2, **style)
            ax.set_title(f"{name}: |E| (y = 0), T = {(a if ax is row[1] else b)['T']:.3f}", fontsize=9)
        for ax in row:
            ax.set(ylabel="z (um)", xlabel="x (um)")
    fig.tight_layout()
    fig.savefig(OUT / "elevator_coupler_fields.png", dpi=130)

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(13, 4))
    o, ta, tb, _ = np.array(report).T
    ax_t.plot(o, ta, "o-", label=prefixes[0])
    ax_t.plot(o, tb, "s--", label=prefixes[1])
    ax_t.set(xlabel="taper overlap (um)", ylabel="TE0 transmission upper -> lower", ylim=(0, 1))
    ax_t.legend()
    for overlap, color in zip(rows, plt.cm.viridis(np.linspace(0, 0.9, len(rows)))):
        a, b = (load(p, overlap) for p in prefixes)
        for d, ls in ((a, "-"), (b, "--")):
            up, lo = guide_power(d)
            ax_p.plot(d["x"], lo / (up + lo), ls, color=color, label=f"{overlap:.2f} um" if ls == "-" else None)
    ax_p.set(xlabel="x (um)", ylabel="fraction of |E|^2 in the lower guide", ylim=(0, 1))
    ax_p.set_title(f"solid: {prefixes[0]}, dashed: {prefixes[1]}", fontsize=9)
    ax_p.legend(title="overlap", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "elevator_coupler_transmission.png", dpi=130)

    print(f"{'overlap':>8} {'T ' + prefixes[0]:>10} {'T ' + prefixes[1]:>10} {'|E|^2 rel. diff':>16}")
    for overlap, t_a, t_b, err in report:
        print(f"{overlap:8.2f} {t_a:10.4f} {t_b:10.4f} {err:16.3f}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:3]) if len(sys.argv) > 2 else ("fdtdx", "meep"))
