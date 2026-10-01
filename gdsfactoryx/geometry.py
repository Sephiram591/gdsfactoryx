"""JAX pytrees holding differentiable gdsfactory geometry."""

from __future__ import annotations

from typing import Any, NamedTuple

import gdsfactory as gf
import jax.numpy as jnp
from jax import Array

LayerKey = tuple[int, int]


class PortGeometry(NamedTuple):
    """Differentiable port: center (2,), orientation (degrees) and width (um)."""

    center: Array
    orientation: Array
    width: Array

    @property
    def x(self) -> Array:
        return self.center[0]

    @property
    def y(self) -> Array:
        return self.center[1]


def layer_key(layer: Any) -> LayerKey:
    """Resolves any gdsfactory LayerSpec ("WG", (1, 0), LAYER.WG, ...) to a tuple."""
    if (
        isinstance(layer, tuple)
        and len(layer) == 2
        and all(isinstance(v, int) for v in layer)
    ):
        return layer
    return tuple(int(v) for v in gf.get_layer_tuple(layer))


def polygon_signed_area(points: Array) -> Array:
    """Shoelace signed area of a closed polygon given as (N, 2) vertices."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y)


class Geometry(NamedTuple):
    """Differentiable snapshot of a gdsfactory Component.

    polygons: {(layer, datatype): [(N_i, 2) vertex arrays in um]}
    ports: {port name: PortGeometry}

    Polygons are merged per layer (unless `merge=False` was requested) and holes
    are encoded as simple polygons with cut lines, exactly as gdsfactory's
    `get_polygons_points` returns them.
    """

    polygons: dict[LayerKey, list[Array]]
    ports: dict[str, PortGeometry]

    @property
    def layers(self) -> list[LayerKey]:
        return list(self.polygons)

    def layer(self, layer: Any) -> list[Array]:
        """Returns the polygons on `layer` (any LayerSpec); [] if it is empty."""
        return self.polygons.get(layer_key(layer), [])

    def area(self, layer: Any = None) -> Array:
        """Total polygon area (um^2) on `layer`, or summed over all layers."""
        layers = self.layers if layer is None else [layer_key(layer)]
        total = jnp.asarray(0.0)
        for key in layers:
            for points in self.polygons.get(key, []):
                total = total + jnp.abs(polygon_signed_area(points))
        return total

    def bbox(self, layer: Any = None) -> Array:
        """[[xmin, ymin], [xmax, ymax]] of the polygons on `layer` (or all)."""
        layers = self.layers if layer is None else [layer_key(layer)]
        points = [p for key in layers for p in self.polygons.get(key, [])]
        if not points:
            raise ValueError(f"No polygons on {layers}")
        allpts = jnp.concatenate(points, axis=0)
        return jnp.stack([allpts.min(axis=0), allpts.max(axis=0)])

    def translate(self, dx: Any = 0.0, dy: Any = 0.0) -> Geometry:
        """Returns a copy moved by (dx, dy); dx and dy may be traced."""
        offset = jnp.stack([jnp.asarray(dx), jnp.asarray(dy)])
        return Geometry(
            polygons={k: [p + offset for p in v] for k, v in self.polygons.items()},
            ports={
                n: p._replace(center=p.center + offset) for n, p in self.ports.items()
            },
        )

    def rotate(self, angle: Any, center: Any = (0.0, 0.0)) -> Geometry:
        """Returns a copy rotated by `angle` degrees about `center`."""
        theta = jnp.deg2rad(angle)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]])
        center = jnp.asarray(center)

        def tf(points: Array) -> Array:
            return (points - center) @ rot.T + center

        return Geometry(
            polygons={k: [tf(p) for p in v] for k, v in self.polygons.items()},
            ports={
                n: PortGeometry(
                    center=tf(p.center[None])[0],
                    orientation=jnp.mod(p.orientation + angle, 360.0),
                    width=p.width,
                )
                for n, p in self.ports.items()
            },
        )

    def __add__(self, other: Geometry) -> Geometry:
        """Union of two geometries (polygons concatenated, ports must not clash)."""
        polygons = {k: list(v) for k, v in self.polygons.items()}
        for k, v in other.polygons.items():
            polygons.setdefault(k, []).extend(v)
        clash = set(self.ports) & set(other.ports)
        if clash:
            raise ValueError(f"Port names clash: {sorted(clash)}")
        return Geometry(polygons=polygons, ports={**self.ports, **other.ports})
