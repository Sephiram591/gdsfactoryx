"""Cost of writing a StackDevice: averaging="subpixel_full" against "subpixel" (compile, warm forward and
backward, peak GPU memory, junction counts), on the test device of tests/test_fdtdx_stack.py (a waveguide with
sloped walls, a slab, a deep etch and a pad) at its resolution (130 x 90 x 45) and at twice it.

Run on a GPU node from the checkout:
    python benchmarks/fdtdx_subpixel/bench_write.py out.json [--res 0.02 0.01] [--averaging subpixel subpixel_full]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))


def peak_bytes() -> int | None:
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def timed(fn, *args, repeat: int = 5) -> tuple[float, float]:
    """(first call incl. compile, median of warm calls), seconds, results waited for."""
    import jax

    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    warm = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        warm.append(time.perf_counter() - t0)
    return first, statistics.median(warm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--res", type=float, nargs="+", default=[0.02, 0.01])
    parser.add_argument("--averaging", nargs="+", default=["subpixel", "subpixel_full"])
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp

    import test_fdtdx_stack as T

    T.gf.gpdk.PDK.activate()
    T.gf.snap.SNAP_ENABLED = False
    rows = []
    print("versions:", dict(jax=jax.__version__, device=str(jax.devices()[0])), flush=True)
    for res in args.res:
        T.RES = res
        for averaging in args.averaging:
            t0 = time.perf_counter()
            d = T.build(averaging=averaging)
            objects, arrays, params = T.place(d)
            placed = time.perf_counter() - t0
            d = objects["stack"]
            p = params["stack"]
            def written(q, d=d, arrays=arrays):
                return d.write(arrays, q).inv_permittivities[:, *d.grid_slice]

            write = jax.jit(written)
            fwd_first, fwd = timed(write, p)
            weights = jnp.ones_like(write(p))
            grad = jax.jit(jax.grad(lambda q, weights=weights, written=written: jnp.sum(weights * written(q))))
            bwd_first, bwd = timed(grad, p)
            status = d.interface_status(p) if averaging == "subpixel_full" else {}
            row = dict(res_um=res, grid=list(d.grid_shape), averaging=averaging, place_s=placed,
                       write_first_s=fwd_first, write_s=fwd, grad_first_s=bwd_first, grad_s=bwd,
                       peak_bytes=peak_bytes(), status=status)
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
