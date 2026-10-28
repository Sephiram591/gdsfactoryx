"""Junction closure gate, part 2: the production closure against the reference (JAX + reference FEM).

The production closure (``gdsfactory.fdtdx_subpixel.periodic_cell_tensor``) solves the voxel's periodic cell
problem with trilinear elements on n x n sub-cells (2.5D here: one sub-cell in z), each with a constant tensor
from its own material fractions:

- ``voigt``: the arithmetic mean of the sub-cell's materials;
- ``kottke``: the laminate (:func:`laminate`) along the sub-cell's own interface direction (its faces'
  length-weighted principal direction; the arithmetic mean where it has no face).

Measured against the reference FEM (interface-fitted, refined):

1. discretization error: closure(n) vs the reference periodic voxel tensor, n = 2, 4, 6, 8;
2. modeling error: vs the embedded tensor (central voxel of a 5 x 5 patch, Dirichlet), which is the laminate
   for planar interfaces;
3. the transition families: a corner entering the voxel, and a second wall turning parallel to the first,
   with the blend ``(1 - omega) laminate(n*) + omega closure`` (omega from the face directions).

Run on a GPU node: python -I benchmarks/fdtdx_subpixel/study_closure.py [out.json]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import shapely
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from reference_fem import converged_2d, half_plane, paint

from gdsfactory.fdtdx_subpixel import junction_weight, laminate, periodic_cell_tensor

jax.config.update("jax_enable_x64", True)

SI, AIR = 12.0, 1.0
EPS = np.stack([AIR * np.eye(3), SI * np.eye(3)])


def rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def through(n, point, offset=0.0):
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    return half_plane(n, -(n @ np.asarray(point, dtype=float)) - offset)


def interfaces(regions):
    """Segments between regions of different materials: list of (LineString, a, b)."""
    out = []
    for i, (r, a) in enumerate(regions):
        for s, b in regions[i + 1 :]:
            if a == b:
                continue
            shared = r.boundary.intersection(s.boundary)
            for g in shapely.get_parts(shared):
                if g.geom_type == "LineString" and g.length > 1e-12:
                    out.append((g, a, b))
    return out


def subcell_tensors(regions, n: int, kind: str) -> np.ndarray:
    """(n, n, 1, 3, 3) sub-cell tensors of the unit square's regions."""
    faces = interfaces(regions)
    out = np.zeros((n, n, 1, 3, 3))
    for i in range(n):
        for j in range(n):
            sq = box(i / n, j / n, (i + 1) / n, (j + 1) / n)
            f = np.zeros(len(EPS))
            for r, m in regions:
                f[m] += r.intersection(sq).area
            f /= f.sum()
            d = np.zeros((3, 3))
            for seg, _, _ in faces:
                part = seg.intersection(sq)
                for g in shapely.get_parts(part):
                    if g.geom_type != "LineString" or g.length < 1e-14:
                        continue
                    (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
                    t = np.array([x1 - x0, y1 - y0]) / g.length
                    nrm = np.array([-t[1], t[0], 0.0])
                    d += g.length * np.outer(nrm, nrm)
            if kind == "kottke" and np.trace(d) > 0:
                _, v = np.linalg.eigh(d)
                out[i, j, 0] = np.asarray(laminate(EPS, jnp.asarray(f), jnp.asarray(v[:, -1])))
            else:
                out[i, j, 0] = np.einsum("m,mij->ij", f, EPS)
    return out


def closure(regions, n: int, kind: str) -> np.ndarray:
    t, _ = periodic_cell_tensor(jnp.asarray(subcell_tensors(regions, n, kind)))
    return np.asarray(t)


def voxel_direction_tensor(regions, tau: float = 0.25) -> np.ndarray:
    """Face direction tensor with face-local thickness factors (the production weighting):
    each face: length * c^2 * phi(min of the two adjacent regions' area / face length), phi(t) = min(1, t/tau).
    """
    d = np.zeros((3, 3))
    for i, (r, a) in enumerate(regions):
        for s, b in regions[i + 1 :]:
            if a == b:
                continue
            shared = r.boundary.intersection(s.boundary)
            length = sum(g.length for g in shapely.get_parts(shared) if g.geom_type == "LineString")
            if length < 1e-12:
                continue
            thick = min(r.area, s.area) / length
            psi = min(1.0, thick / tau)
            c2 = (np.linalg.norm(EPS[a] - EPS[b]) / max(np.linalg.norm(e) for e in EPS)) ** 2
            for g in shapely.get_parts(shared):
                if g.geom_type != "LineString" or g.length < 1e-14:
                    continue
                (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
                tvec = np.array([x1 - x0, y1 - y0]) / g.length
                nrm = np.array([-tvec[1], tvec[0], 0.0])
                d += g.length * c2 * psi * np.outer(nrm, nrm)
    return d


def fractions(regions) -> np.ndarray:
    f = np.zeros(len(EPS))
    for r, m in regions:
        f[m] += r.area
    return f / f.sum()


def embedded(make, size=5):
    c = np.array([size / 2, size / 2])
    regions = paint(make(c), 0, size)
    window = ((size - 1) / 2, (size + 1) / 2)
    best, _, _ = converged_2d(regions, EPS, "dirichlet", n=8, levels=2, size=size, window=window)
    return best


def main(out: str | None) -> None:
    t0 = time.time()
    rows = []
    cases = {
        "plane 30 deg f=0.5": lambda c: [(through([math.cos(0.5236), math.sin(0.5236)], c), 1)],
        "convex 90 at center": lambda c: [(through([1, 0], c).intersection(through([0, 1], c)), 1)],
        "reflex 270 at center": lambda c: [(through([1, 0], c).union(through([0, 1], c)), 1)],
        "convex 90 rotated 30": lambda c: [(through([math.cos(0.5236), math.sin(0.5236)], c).intersection(
            through([-math.sin(0.5236), math.cos(0.5236)], c)), 1)],
        "acute 60 at center": lambda c: [(through([0, 1], c).intersection(
            through([math.sin(math.pi / 3), -math.cos(math.pi / 3)], c)), 1)],
        "strip w=0.3 at 20 deg": lambda c: [(through([math.cos(0.349), math.sin(0.349)], c, 0.15).difference(
            through([math.cos(0.349), math.sin(0.349)], c, -0.15)), 1)],
    }
    # 1, 2: discretization and modeling error at fixed geometries
    for name, make in cases.items():
        regions = paint(make(np.array([0.5, 0.5])), 0)
        ref_per, _, hist = converged_2d(regions, EPS, "periodic", n=8, levels=3)
        emb = embedded(make)
        row = dict(case=name, ref_change=rel(hist[-1], hist[-2]), embedded_vs_periodic_ref=rel(ref_per, emb))
        for kind in ("voigt", "kottke"):
            for n in (2, 4, 6, 8):
                c = closure(regions, n, kind)
                row[f"{kind}{n}_vs_ref"] = rel(c, ref_per)
                if n == 8:
                    row[f"{kind}{n}_vs_embedded"] = rel(c, emb)
        d = voxel_direction_tensor(regions)
        omega, nstar = junction_weight(jnp.asarray(d), 0.02, 0.2)
        lam = np.asarray(laminate(EPS, jnp.asarray(fractions(regions)), nstar))
        blend = (1 - float(omega)) * lam + float(omega) * closure(regions, 6, "kottke")
        row.update(omega=float(omega), laminate_vs_embedded=rel(lam, emb), blend6_vs_embedded=rel(blend, emb))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    # 3. transition families
    families = {
        # Si = {x > cx, y > 0.5} (voxel coordinates): planar (the top half) for cx < 0; the corner enters
        # through the left side at cx = 0, crosses the voxel and leaves it at cx = 1
        "corner entering": [
            (cx, lambda c, cx=cx: [(through([-1, 0], (c[0] - 0.5 + cx, 0)).intersection(through([0, -1], c)), 1)])
            for cx in (-0.1, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
        ],
        # Si = {x < 0.5} and {n2 . (p - c) > -0.2}: a convex corner at 90 deg that closes into a strip
        # 0.3 < x < 0.5 (two parallel walls, planar) as the second wall turns parallel to the first
        "second wall turning": [
            (deg, lambda c, deg=deg: [(through([1, 0], c).intersection(
                through([-math.cos(math.radians(deg)), -math.sin(math.radians(deg))], c, 0.2)), 1)])
            for deg in (90.0, 60.0, 30.0, 15.0, 5.0, 1.0, 0.0)
        ],
    }
    for fam, members in families.items():
        for x, make in members:
            regions = paint(make(np.array([0.5, 0.5])), 0)
            emb = embedded(make)
            d = voxel_direction_tensor(regions)
            omega, nstar = junction_weight(jnp.asarray(d), 0.02, 0.2)
            f = fractions(regions)
            lam = np.asarray(laminate(EPS, jnp.asarray(f), nstar))
            clo = closure(regions, 6, "kottke")
            blend = (1 - float(omega)) * lam + float(omega) * clo
            row = dict(family=fam, x=x, f_si=float(f[1]), omega=float(omega), xx=round(float(blend[0, 0]), 4),
                       yy=round(float(blend[1, 1]), 4), xy=round(float(blend[0, 1]), 4),
                       blend_vs_embedded=rel(blend, emb), laminate_vs_embedded=rel(lam, emb),
                       closure_vs_embedded=rel(clo, emb))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    print(f"elapsed {time.time() - t0:.1f} s")
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
