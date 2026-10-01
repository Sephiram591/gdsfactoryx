"""Differentiable, anti-aliased rasterization of polygons (pure JAX).

Each pixel value is the *exact* area fraction of the pixel covered by the
polygons, computed edge by edge with Green's theorem:

    area(P ∩ B) = -∮_{∂P} g_B(y) dx,   g_B(y) = clip(y, y0, y1) - y0,

with x restricted to the pixel column. The result is piecewise smooth in the
vertex coordinates, so gradients flow from pixel densities (e.g. a permittivity
map fed to a JAX FDTD/FDFD solver) back to gdsfactory parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gdsfactoryx.geometry import Geometry, polygon_signed_area


def _antiderivative(y: Array, lo: Array, hi: Array) -> Array:
    """G(y) = ∫_{-inf}^{y} (clip(s, lo, hi) - lo) ds."""
    span = hi - lo
    below = jnp.zeros_like(y)
    inside = 0.5 * (y - lo) ** 2
    above = 0.5 * span**2 + span * (y - hi)
    return jnp.where(y < lo, below, jnp.where(y > hi, above, inside))


def _edges(polygons: Sequence[Array]) -> Array:
    """(E, 5) array of [x0, y0, x1, y1, orientation_sign] for all edges."""
    rows = []
    for points in polygons:
        sign = jax.lax.stop_gradient(jnp.sign(polygon_signed_area(points)))
        nxt = jnp.roll(points, -1, axis=0)
        s = jnp.broadcast_to(sign, (points.shape[0], 1))
        rows.append(jnp.concatenate([points, nxt, s], axis=1))
    return jnp.concatenate(rows, axis=0)


def coverage(
    polygons: Sequence[Array],
    x_edges: Array,
    y_edges: Array,
    *,
    chunk: int = 128,
) -> Array:
    """Exact covered area fraction per pixel, shape (len(x_edges)-1, len(y_edges)-1).

    Args:
        polygons: (N_i, 2) vertex arrays (any orientation).
        x_edges: increasing pixel boundaries along x.
        y_edges: increasing pixel boundaries along y.
        chunk: edges processed per step (bounds memory to chunk*nx*ny).
    """
    x_edges = jnp.asarray(x_edges)
    y_edges = jnp.asarray(y_edges)
    nx, ny = x_edges.shape[0] - 1, y_edges.shape[0] - 1
    if not polygons:
        return jnp.zeros((nx, ny), dtype=x_edges.dtype)

    edges = _edges(polygons)
    dtype = jnp.result_type(edges, x_edges)
    edges = edges.astype(dtype)
    n_edges = edges.shape[0]
    n_chunks = -(-n_edges // chunk)
    # Zero-length padding edges contribute exactly zero.
    edges = jnp.pad(edges, ((0, n_chunks * chunk - n_edges), (0, 0)))
    edges = edges.reshape(n_chunks, chunk, 5)

    bx0, bx1 = x_edges[:-1], x_edges[1:]
    by0, by1 = y_edges[:-1], y_edges[1:]
    eps = 1e-6 * jnp.min(by1 - by0)

    def body(acc: Array, e: Array) -> tuple[Array, None]:
        x0, y0, x1, y1, sign = (e[:, i : i + 1] for i in range(5))
        dx = x1 - x0
        slope = (y1 - y0) / jnp.where(dx == 0, 1.0, dx)
        # Restrict the edge to each pixel column (chunk, nx).
        xa = jnp.clip(x0, bx0, bx1)
        xb = jnp.clip(x1, bx0, bx1)
        ya = y0 + (xa - x0) * slope
        yb = y0 + (xb - x0) * slope
        ya, yb = ya[..., None], yb[..., None]  # (chunk, nx, 1)
        lo, hi = by0[None, None, :], by1[None, None, :]
        d = yb - ya
        small = jnp.abs(d) < eps
        safe_d = jnp.where(small, 1.0, d)
        mean_secant = (_antiderivative(yb, lo, hi) - _antiderivative(ya, lo, hi)) / (
            safe_d
        )
        mid = 0.5 * (ya + yb)
        mean_mid = jnp.clip(mid, lo, hi) - lo
        mean_g = jnp.where(small, mean_mid, mean_secant)
        integral = (xb - xa)[..., None] * mean_g  # (chunk, nx, ny)
        return acc - jnp.sum(sign[..., None] * integral, axis=0), None

    area, _ = jax.lax.scan(body, jnp.zeros((nx, ny), dtype=dtype), edges)
    pixel_area = (bx1 - bx0)[:, None] * (by1 - by0)[None, :]
    return area / pixel_area


def grid_edges(bounds: Any, shape: tuple[int, int]) -> tuple[Array, Array]:
    """Pixel edges for bounds [[xmin, ymin], [xmax, ymax]] and shape (nx, ny)."""
    (xmin, ymin), (xmax, ymax) = bounds
    nx, ny = shape
    return jnp.linspace(xmin, xmax, nx + 1), jnp.linspace(ymin, ymax, ny + 1)


def rasterize(
    geometry: Geometry | Sequence[Array],
    layer: Any = None,
    *,
    bounds: Any = None,
    shape: tuple[int, int] | None = None,
    x_edges: Array | None = None,
    y_edges: Array | None = None,
    clip: bool = False,
    chunk: int = 128,
) -> Array:
    """Rasterizes a Geometry layer (or a list of polygons) into a density map.

    Specify the grid either with `bounds` + `shape` (nx, ny) or explicit
    `x_edges` / `y_edges`. Returns an (nx, ny) array of covered area fractions
    in [0, 1] (sums of overlapping polygons may exceed 1 unless `clip=True`;
    jaxified geometry is merged per layer by default, so this does not happen).
    """
    if isinstance(geometry, Geometry):
        polygons = (
            geometry.layer(layer)
            if layer is not None
            else [p for polys in geometry.polygons.values() for p in polys]
        )
    else:
        polygons = list(geometry)

    if x_edges is None or y_edges is None:
        if bounds is None or shape is None:
            raise ValueError("Pass either bounds and shape, or x_edges and y_edges")
        x_edges, y_edges = grid_edges(bounds, shape)

    density = coverage(polygons, x_edges, y_edges, chunk=chunk)
    return jnp.clip(density, 0.0, 1.0) if clip else density
