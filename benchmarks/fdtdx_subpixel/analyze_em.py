"""Tables of the EM validation runs: errors per case and resolution (worst over grid offsets), observed orders.

    python analyze_em.py slab out_slab.json [more.json ...]
    python analyze_em.py grating meep.json fdtdx.json [more fdtdx.json ...]

Slab: against the Berreman reference (exact). Grating: against meep's sequence, extrapolated with its own
observed order from the finest three resolutions (Richardson with that order); the spread of fdtdx over grid
offsets is reported separately (an exact method has none).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows += json.loads(open(p).read())
    return rows


def order(e_coarse: float, e_fine: float, ratio: float = 2.0) -> float:
    return math.log(e_coarse / e_fine) / math.log(ratio) if e_coarse > 0 and e_fine > 0 else float("nan")


def slab(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["case"], r["res_nm"])].append(r)
    cases = sorted({c for c, _ in by})
    keys = ["t_err", "r_err", "energy_err", "t_asymmetry", "reciprocity_err"]
    for case in cases:
        print(f"\n{case}")
        print(f"{'res nm':>7} " + " ".join(f"{k:>16}" for k in keys) + f" {'|t_yx| (ref)':>18}")
        prev = None
        for res in sorted({res for c, res in by if c == case}, reverse=True):
            group = by[(case, res)]
            worst = {k: max((g[k] for g in group if k in g), default=float("nan")) for k in keys}
            conv = f"  order {order(prev, worst['t_err']):.2f}" if prev else ""
            ref = group[0]["t_yx_ref"]
            spread = max(g["t_yx"] for g in group) - min(g["t_yx"] for g in group)
            print(f"{res:7.1f} " + " ".join(f"{worst[k]:16.3e}" for k in keys)
                  + f" {group[0]['t_yx']:.5f} ({ref:.5f}) spread {spread:.1e}{conv}")
            prev = worst["t_err"]


def grating(meep_rows: list[dict], rows: list[dict]) -> None:
    for kind in sorted({r["kind"] for r in meep_rows}):
        for angle in sorted({r["angle"] for r in meep_rows if r["kind"] == kind}):
            seq = sorted((r for r in meep_rows if r["kind"] == kind and r["angle"] == angle), key=lambda r: r["px"])
            taus = [np.array([complex(*v) for v in r["tau"]]) for r in seq]
            pxs = [r["px"] for r in seq]
            print(f"\n{kind} rods, {angle} deg; meep: " + ", ".join(f"{p}: {t[0]:.5f}/{t[1]:.5f}" for p, t in zip(pxs, taus, strict=True)))
            ref = taus[-1]
            if len(taus) >= 3:
                d1, d2 = np.abs(taus[-2] - taus[-3]).max(), np.abs(taus[-1] - taus[-2]).max()
                p = order(d1, d2)
                p_used = p if np.isfinite(p) and 0.5 < p < 4 else 1.0
                ref = taus[-1] + (taus[-1] - taus[-2]) / (2**p_used - 1)
                print(f"  meep observed order {p:.2f}; extrapolated {ref[0]:.5f}/{ref[1]:.5f}; "
                      f"last step {d2:.2e} (reference uncertainty about {abs(d2) / (2**p_used - 1):.1e})")
            group = [r for r in rows if r["kind"] == kind and r["angle"] == angle]
            for tag, averaging in sorted({(r.get("tag", ""), r["averaging"]) for r in group}):
                prev = None
                line = []
                for res in sorted({r["res_nm"] for r in group}, reverse=True):
                    vals = [np.array([complex(*v) for v in r["tau"]]) for r in group
                            if r["res_nm"] == res and r["averaging"] == averaging and r.get("tag", "") == tag]
                    if not vals:
                        continue
                    err = max(np.abs(v - ref).max() for v in vals)
                    spread = max(np.abs(v - vals[0]).max() for v in vals)
                    conv = f" (order {order(prev, err):.2f})" if prev else ""
                    line.append(f"{res:g} nm {err:.2e} spread {spread:.1e}{conv}")
                    prev = err
                print(f"  {tag:10s} {averaging:14s}: " + "; ".join(line))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "slab":
        slab(load(sys.argv[2:]))
    else:
        grating(load(sys.argv[2:3]), load(sys.argv[3:]))
