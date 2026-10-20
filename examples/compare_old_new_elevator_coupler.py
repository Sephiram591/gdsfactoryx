"""Compares the elevator-coupler fdtdx results from the old voxelization, the new one, and MEEP.

    python examples/compare_old_new_elevator_coupler.py [examples_dir]     # CPU only

The old sweep is the archive ``output_old_voxelization_2026-10-05``; new fdtdx and MEEP are read
from ``output``, both under ``examples_dir`` (default: this directory). Same normalization and
interpolation as ``compare_elevator_coupler.py``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_elevator_coupler import OVERLAPS, Z_VIEW, guide_power, on_grid


def load(folder: Path, prefix: str, overlap: float) -> dict:
    d = dict(np.load(folder / f"{prefix}_overlap_{overlap:.2f}.npz"))
    d["E2"] = np.squeeze(d["E2"])
    d["eps"] = np.squeeze(d["eps"])
    x = d["x"]
    d["E2"] = d["E2"] / d["E2"][(x > -0.6) & (x < -0.1)].max()  # input-mode peak, as in the other script
    return d


def main(base: Path) -> None:
    out, old = base / "output", base / "output_old_voxelization_2026-10-05"
    cases = (("old", old, "fdtdx"), ("new", out, "fdtdx"), ("meep", out, "meep"))
    rows, report = [], []
    for overlap in OVERLAPS:
        data = {n: load(f, p, overlap) for n, f, p in cases}
        rows.append((overlap, data))
        meep = data["meep"]
        errs = {}
        for name in ("old", "new"):
            d = data[name]
            x, z = d["x"], d["z"]
            keep = (z >= Z_VIEW[0]) & (z <= Z_VIEW[1])
            interior = (x > -0.6) & (x < x.max() - 0.5)
            ref = on_grid(meep, x, z)
            errs[name] = float(
                np.linalg.norm((d["E2"] - ref)[interior][:, keep]) / np.linalg.norm(ref[interior][:, keep])
            )
        t_old, t_new, t_meep = (float(data[n]["T"]) for n in ("old", "new", "meep"))
        report.append((overlap, t_old, t_new, t_meep, abs(t_new - t_meep), abs(t_old - t_meep), errs["old"], errs["new"]))

    print(f"{'overlap':>8} {'T old':>8} {'T new':>8} {'T meep':>8} {'|new-meep|':>11} {'|old-meep|':>11} "
          f"{'E2 old':>8} {'E2 new':>8}")
    for r in report:
        print(f"{r[0]:8.2f} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:11.4f} {r[5]:11.4f} {r[6]:8.3f} {r[7]:8.3f}")

    # fields: old / new / meep |E| and the |E|^2 differences to meep, on the new grid
    fig, axes = plt.subplots(len(rows), 5, figsize=(26, 2.2 * len(rows)), squeeze=False)
    for row, (overlap, data) in zip(axes, rows):
        x, z = data["new"]["x"], data["new"]["z"]
        keep = (z >= Z_VIEW[0]) & (z <= Z_VIEW[1])
        style = dict(extent=(x.min(), x.max(), Z_VIEW[0], Z_VIEW[1]), origin="lower", aspect="auto")
        maps = {
            "old": on_grid(data["old"], x, z),
            "new": data["new"]["E2"],
            "meep": on_grid(data["meep"], x, z),
        }
        for ax, name in zip(row[:3], ("old", "new", "meep")):
            ax.imshow(np.sqrt(maps[name][:, keep]).T, cmap="inferno", vmin=0, vmax=1.2, **style)
            ax.set_title(f"{overlap:.2f} um, {name}: |E| (y = 0), T = {float(data[name]['T']):.3f}", fontsize=9)
        for ax, name in zip(row[3:], ("old", "new")):
            im = ax.imshow((maps[name] - maps["meep"])[:, keep].T, cmap="RdBu_r", vmin=-0.3, vmax=0.3, **style)
            ax.set_title(f"|E|^2 {name} - meep", fontsize=9)
        for ax in row:
            ax.set(ylabel="z (um)", xlabel="x (um)")
    fig.colorbar(im, ax=axes[:, 3:], shrink=0.6, label="|E|^2 difference")
    fig.savefig(out / "elevator_old_vs_new_fields.png", dpi=110, bbox_inches="tight")

    fig, (ax_t, ax_e, ax_p) = plt.subplots(1, 3, figsize=(18, 4))
    o, t_old, t_new, t_meep, _, _, _, _ = np.array(report).T
    ax_t.plot(o, t_old, "o-", label="fdtdx old")
    ax_t.plot(o, t_new, "s-", label="fdtdx new")
    ax_t.plot(o, t_meep, "^--", color="k", label="meep")
    ax_t.set(xlabel="taper overlap (um)", ylabel="TE0 transmission upper -> lower", ylim=(0, 1))
    ax_t.legend()
    ax_e.plot(o, abs(t_old - t_meep), "o-", label="fdtdx old")
    ax_e.plot(o, abs(t_new - t_meep), "s-", label="fdtdx new")
    ax_e.set(xlabel="taper overlap (um)", ylabel="|T - T meep|")
    ax_e.legend()
    for (overlap, data), color in zip(rows, plt.cm.viridis(np.linspace(0, 0.9, len(rows)))):
        for name, ls in (("old", ":"), ("new", "-"), ("meep", "--")):
            up, lo = guide_power(data[name])
            ax_p.plot(data[name]["x"], lo / (up + lo), ls, color=color,
                      label=f"{overlap:.2f} um" if name == "new" else None)
    ax_p.set(xlabel="x (um)", ylabel="fraction of |E|^2 in the lower guide", ylim=(0, 1))
    ax_p.set_title("dotted: old, solid: new, dashed: meep", fontsize=9)
    ax_p.legend(title="overlap", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "elevator_old_vs_new_transmission.png", dpi=130)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
