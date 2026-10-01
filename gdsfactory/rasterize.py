"""Differentiable rasterization of Components.

Turns the (traced) polygons of a Component into a smooth density map on a
regular grid, so that any pixel-based objective (e.g. a JAX FDFD/FDTD solver,
a lithography model or a fill-factor target) can be differentiated with
respect to the parameters of the component functions.

The density of a polygon at a point p is ``sigmoid(-sd(p) / smoothing)`` where
``sd`` is the signed distance to the polygon boundary (negative inside). Several
polygons are combined with a soft union ``1 - prod(1 - rho_i)``.

Example:
    ```python
    import jax
    import gdsfactory as gf

    gf.gpdk.PDK.activate()

    def fill(width):
        c = gf.components.straight(length=5, width=width)
        rho, (x, y) = gf.rasterize.rasterize(c, layer=(1, 0), resolution=0.02)
        return rho.mean()

    jax.grad(fill)(0.5)
    ```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from gdsfactory._jax import Array, asarray, jnp, to_float, to_numpy

if TYPE_CHECKING:
    from gdsfactory.component import Component
    from gdsfactory.typings import LayerSpec


def _segment_distance(points: Array, a: Array, b: Array) -> Array:
    """Distance from points (P, 2) to segments a->b (E, 2): returns (P, E)."""
    ab = b - a  # (E, 2)
    ap = points[:, None, :] - a[None, :, :]  # (P, E, 2)
    denom = jnp.sum(ab * ab, axis=-1)  # (E,)
    t = jnp.sum(ap * ab[None], axis=-1) / jnp.where(denom > 0, denom, 1.0)
    t = jnp.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = points[:, None, :] - proj
    return jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)


def _inside(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point in polygon test (concrete, non-differentiable)."""
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x1 = poly[:, 0][None, :]
    y1 = poly[:, 1][None, :]
    x2 = np.roll(poly[:, 0], -1)[None, :]
    y2 = np.roll(poly[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    crossings = np.sum(cond & (x < xint), axis=1)
    return (crossings % 2) == 1


def signed_distance(points: Array, polygon: Array, chunk: int = 65536) -> Array:
    """Signed distance (negative inside) from points (P, 2) to a polygon (N, 2)."""
    a = polygon
    b = jnp.roll(polygon, -1, axis=0)
    inside = _inside(to_numpy(points), to_numpy(polygon))
    out = []
    for start in range(0, points.shape[0], chunk):
        pts = points[start : start + chunk]
        d = jnp.min(_segment_distance(pts, a, b), axis=1)
        out.append(d)
    dist = jnp.concatenate(out) if out else jnp.zeros((0,))
    return jnp.where(inside, -dist, dist)


def rasterize_polygons(
    polygons: Sequence[Array],
    x: Array,
    y: Array,
    smoothing: float | None = None,
) -> Array:
    """Soft density of a list of polygons on the grid (x, y): returns (len(y), len(x))."""
    x = asarray(x)
    y = asarray(y)
    dx = float(np.min(np.abs(np.diff(to_numpy(x))))) if x.shape[0] > 1 else 1.0
    dy = float(np.min(np.abs(np.diff(to_numpy(y))))) if y.shape[0] > 1 else 1.0
    beta = smoothing if smoothing is not None else 0.5 * min(dx, dy)
    xx, yy = jnp.meshgrid(x, y)
    pts = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    empty = jnp.ones(pts.shape[0])
    margin = 8 * beta
    for poly in polygons:
        poly = asarray(poly)
        pmin = to_numpy(jnp.min(poly, axis=0)) - margin
        pmax = to_numpy(jnp.max(poly, axis=0)) + margin
        pts_np = to_numpy(pts)
        mask = np.all((pts_np >= pmin) & (pts_np <= pmax), axis=1)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            continue
        sd = signed_distance(pts[idx], poly)
        rho = jax.nn.sigmoid(-sd / beta)
        empty = empty.at[idx].set(empty[idx] * (1 - rho))
    return (1 - empty).reshape(yy.shape)


def rasterize(
    component: Component,
    layer: LayerSpec | None = None,
    resolution: float = 0.02,
    bbox: Sequence[float] | None = None,
    x: Array | None = None,
    y: Array | None = None,
    smoothing: float | None = None,
    padding: float = 0.0,
) -> tuple[Array, tuple[Array, Array]]:
    """Differentiable rasterization of a Component.

    Args:
        component: to rasterize.
        layer: layer to rasterize. None rasterizes all layers together.
        resolution: pixel size (um), used when x/y are not given.
        bbox: (xmin, ymin, xmax, ymax) of the grid. Defaults to the (concrete) bbox.
        x: grid x coordinates (pixel centers). Overrides resolution/bbox.
        y: grid y coordinates (pixel centers).
        smoothing: edge smoothing length (um). Defaults to half a pixel.
        padding: padding around the bbox (um).

    Returns:
        density: (ny, nx) array in [0, 1].
        (x, y): the grid coordinates.
    """
    from gdsfactory.component import _layer_key

    polys_by_layer = component.get_polygons_points()
    if layer is not None:
        polys = polys_by_layer.get(_layer_key(layer), [])
    else:
        polys = [p for ps in polys_by_layer.values() for p in ps]

    if x is None or y is None:
        if bbox is None:
            b = component.dbbox(layer)
            bbox = (to_float(b.left), to_float(b.bottom), to_float(b.right), to_float(b.top))
        xmin, ymin, xmax, ymax = (to_float(v) for v in bbox)
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
        nx = max(1, round((xmax - xmin) / resolution))
        ny = max(1, round((ymax - ymin) / resolution))
        x = jnp.asarray(xmin + (np.arange(nx) + 0.5) * resolution)
        y = jnp.asarray(ymin + (np.arange(ny) + 0.5) * resolution)
    density = rasterize_polygons(polys, x, y, smoothing=smoothing)
    return density, (x, y)


__all__ = ["rasterize", "rasterize_polygons", "signed_distance"]
