"""Junction closure gate, part 1: boundary-condition and patch-size bias of a voxel's cell problem.

Reference FEM only (numpy / scipy / shapely). Checks the reference itself (a homogeneous voxel; layers on the
grid, where the periodic cell problem must equal Kottke's laminate), then measures the apparent tensor of the
central voxel of a P x P patch (P = 1: the voxel alone) whose geometry continues the voxel's (the same
half-planes), for periodic, affine Dirichlet and mean-field ("mean") conditions on the patch:

- planar interfaces at oblique angles, where the exact answer is the laminate: the bias of each condition;
- corners, where no local answer is exact: the spread between conditions and its trend with the patch size.

Relative errors are |A - B|_F / |B|_F. Run (CPU, no JAX):

    python -I benchmarks/fdtdx_subpixel/study_boundary_conditions.py [out.json]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference_fem import converged_2d, half_plane, kottke_frame, paint

SI, AIR = 12.0, 1.0
EPS = np.stack([AIR * np.eye(3), SI * np.eye(3)])


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def through(n, point, offset=0.0):
    """Half-plane {p : n . (p - point) - offset < 0}."""
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    return half_plane(n, -(n @ np.asarray(point, dtype=float)) - offset)


def voxel_fraction(levels, size: int) -> float:
    lo = (size - 1) / 2
    from shapely.geometry import box

    vox = box(lo, lo, lo + 1, lo + 1)
    return sum(r.intersection(vox).area for r, m in paint(levels, 0, size) if m == 1)


def run_case(name, make_levels, kottke=None, sizes=(1, 3, 5), bcs=("periodic", "dirichlet", "mean"), n=8, levels=3):
    rows = []
    for size in sizes:
        c = np.array([size / 2, size / 2])
        lv = make_levels(c)
        regions = paint(lv, 0, size)
        window = None if size == 1 else ((size - 1) / 2, (size + 1) / 2)
        for bc in bcs:
            if bc == "mean" and size > 1:
                continue  # the mean-field condition is defined on the voxel alone here
            best, _, hist = converged_2d(regions, EPS, bc, n=n, levels=levels, size=size, window=window)
            row = dict(case=name, size=size, bc=bc, change=rel(hist[-1], hist[-2]),
                       xx=round(float(best[0, 0]), 4), yy=round(float(best[1, 1]), 4), xy=round(float(best[0, 1]), 4))
            if kottke is not None:
                row["err_vs_kottke"] = rel(best, kottke)
            rows.append(row)
    return rows


def main(out: str | None) -> None:
    rows = []
    t0 = time.time()

    # 1. the reference itself
    best, _, _ = converged_2d(paint([], 1), EPS, "periodic", n=4, levels=2)
    rows.append(dict(case="homogeneous", bc="periodic", err=rel(best, EPS[1])))
    k = kottke_frame(EPS, np.array([0.7, 0.3]), np.array([1.0, 0, 0]))
    for bc in ("periodic", "dirichlet", "mean"):
        best, _, hist = converged_2d(paint([(half_plane([1, 0], -0.3), 1)], 0), EPS, bc, n=8, levels=3)
        rows.append(dict(case="aligned x<0.3", bc=bc, err_vs_kottke=rel(best, k), change=rel(hist[-1], hist[-2])))

    # 2. planar interfaces: oblique, through the voxel (exact answer: the laminate)
    for degrees in (0.0, 15.0, 30.0, 45.0):
        for offset in (0.0, 0.25):
            a = math.radians(degrees)
            n = np.array([math.cos(a), math.sin(a)])
            f = voxel_fraction([(through(n, (0.5 + offset * n[0], 0.5 + offset * n[1])), 1)], 1)
            k = kottke_frame(EPS, np.array([1 - f, f]), np.array([n[0], n[1], 0.0]))
            rows += run_case(
                f"plane {degrees:g} deg, f={f:.3f}",
                lambda c, n=n, offset=offset: [(through(n, c + offset * n), 1)],
                kottke=k,
            )

    # 3. corners (no exact local answer): spread between conditions and trend with the patch
    corners = {
        "convex 90 at center": lambda c: [(through([1, 0], c).intersection(through([0, 1], c)), 1)],
        "convex 90 at (-0.2,-0.2)": lambda c: [(through([1, 0], c - 0.2).intersection(through([0, 1], c - 0.2)), 1)],
        "reflex 270 at center": lambda c: [(through([1, 0], c).union(through([0, 1], c)), 1)],
        "convex 90 rotated 30": lambda c: [(through([math.cos(0.5236), math.sin(0.5236)], c).intersection(
            through([-math.sin(0.5236), math.cos(0.5236)], c)), 1)],
    }
    for name, make in corners.items():
        rows += run_case(name, make)

    for r in rows:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print(f"elapsed {time.time() - t0:.1f} s")
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
