"""Host-side (numpy) extraction of gdsfactory results and topology alignment.

Everything in this module runs outside of JAX tracing. It converts the result of
an ordinary gdsfactory call into a "raw" numpy representation, describes its
structure with a hashable :class:`Spec`, and maps a raw result onto a reference
structure so that finite-difference evaluations line up vertex by vertex.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import gdsfactory as gf
import jax
import numpy as np
from scipy.optimize import linear_sum_assignment

LayerKey = tuple[int, int]

# Per-port slots in the flat vector: center x, center y, orientation, width.
PORT_SIZE = 4


class TopologyWarning(UserWarning):
    """Raised when perturbed geometry does not have the reference topology."""


@dataclass
class RawGeometry:
    polygons: dict[LayerKey, list[np.ndarray]]
    ports: dict[str, np.ndarray]  # name -> [x, y, orientation, width]


@dataclass
class RawPath:
    points: np.ndarray


@dataclass
class RawTree:
    treedef: Any
    leaves: list[np.ndarray]


Raw = RawGeometry | RawPath | RawTree


@dataclass(frozen=True)
class Spec:
    """Hashable description of the shape of a flattened result."""

    kind: str  # "geometry" | "path" | "tree"
    layers: tuple[tuple[LayerKey, tuple[int, ...]], ...] = ()
    ports: tuple[str, ...] = ()
    npoints: int = 0
    treedef: Any = None
    shapes: tuple[tuple[int, ...], ...] = ()

    @property
    def size(self) -> int:
        if self.kind == "geometry":
            nverts = sum(sum(counts) for _, counts in self.layers)
            return 2 * nverts + PORT_SIZE * len(self.ports)
        if self.kind == "path":
            return 2 * self.npoints
        return int(sum(np.prod(s, dtype=int) for s in self.shapes))

    def angle_mask(self) -> np.ndarray:
        """Boolean mask of the flat entries that are angles in degrees."""
        mask = np.zeros(self.size, dtype=bool)
        if self.kind == "geometry":
            start = self.size - PORT_SIZE * len(self.ports)
            mask[start + 2 :: PORT_SIZE] = True
        return mask


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def _is_component_like(obj: Any) -> bool:
    return isinstance(obj, gf.Component | gf.ComponentReference) or (
        hasattr(obj, "kcl") and hasattr(obj, "ports") and hasattr(obj, "bbox")
    )


def _polygon_points(polygon: Any, dbu: float) -> np.ndarray:
    simple = polygon.to_simple_polygon().to_dtype(dbu)
    return np.array([(p.x, p.y) for p in simple.each_point()], dtype=np.float64)


def _centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def to_raw(result: Any, merge: bool = True, layers: Any = None) -> Raw:
    """Converts the result of a gdsfactory call into a raw numpy representation."""
    if _is_component_like(result):
        dbu = result.kcl.dbu
        polygons_by_layer = gf.functions.get_polygons(
            result, merge=merge, by="tuple", layers=layers
        )
        polygons: dict[LayerKey, list[np.ndarray]] = {}
        for key, polys in polygons_by_layer.items():
            points = [_polygon_points(p, dbu) for p in polys]
            points = [p for p in points if len(p) >= 3]
            if points:
                # Deterministic order: sort by centroid.
                points.sort(key=lambda p: tuple(np.round(_centroid(p), 6)))
                polygons[tuple(int(v) for v in key)] = points
        ports = {}
        for port in result.ports:
            x, y = port.center
            orientation = port.orientation if port.orientation is not None else 0.0
            ports[port.name] = np.array(
                [x, y, orientation, port.width], dtype=np.float64
            )
        return RawGeometry(polygons=polygons, ports=dict(sorted(ports.items())))

    if isinstance(result, gf.Path):
        return RawPath(points=np.asarray(result.points, dtype=np.float64))

    leaves, treedef = jax.tree_util.tree_flatten(result)
    try:
        arrays = [np.asarray(leaf, dtype=np.float64) for leaf in leaves]
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"gdsfactoryx can only differentiate functions returning a Component, "
            f"ComponentReference, Path, or a pytree of numbers; got {type(result)}"
        ) from e
    return RawTree(treedef=treedef, leaves=arrays)


def spec_of(raw: Raw) -> Spec:
    if isinstance(raw, RawGeometry):
        layers = tuple(
            (key, tuple(len(p) for p in polys)) for key, polys in raw.polygons.items()
        )
        return Spec(kind="geometry", layers=layers, ports=tuple(raw.ports))
    if isinstance(raw, RawPath):
        return Spec(kind="path", npoints=len(raw.points))
    return Spec(
        kind="tree",
        treedef=raw.treedef,
        shapes=tuple(leaf.shape for leaf in raw.leaves),
    )


def flatten(raw: Raw) -> np.ndarray:
    """Flattens an aligned raw result into a 1D vector following its Spec."""
    if isinstance(raw, RawGeometry):
        parts = [p.ravel() for polys in raw.polygons.values() for p in polys]
        parts += [raw.ports[name] for name in raw.ports]
        return np.concatenate(parts) if parts else np.zeros(0)
    if isinstance(raw, RawPath):
        return raw.points.ravel()
    parts = [leaf.ravel() for leaf in raw.leaves]
    return np.concatenate(parts) if parts else np.zeros(0)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        warnings.warn(message, TopologyWarning, stacklevel=4)


def _cyclic_align(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rolls `points` (same length as `ref`) to best match `ref` vertex by vertex."""
    # argmin_s sum |points[i + s] - ref[i]|^2 == argmax_s of the circular
    # cross-correlation, computed with FFTs.
    fp = np.fft.rfft(points, axis=0)
    fr = np.fft.rfft(ref, axis=0)
    corr = np.fft.irfft(np.conj(fr) * fp, n=len(ref), axis=0).sum(axis=1)
    shift = int(np.argmax(corr))
    return np.roll(points, -shift, axis=0)


def _closest_points(
    queries: np.ndarray, polyline: np.ndarray, closed: bool
) -> np.ndarray:
    """Closest point on a polyline for every query point (chunked)."""
    a = polyline
    b = np.roll(polyline, -1, axis=0) if closed else polyline[1:]
    a = a if closed else a[:-1]
    ab = b - a
    denom = np.maximum((ab**2).sum(axis=1), 1e-30)
    out = np.empty_like(queries)
    chunk = max(1, 2_000_000 // max(len(a), 1))
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk, None, :]
        t = np.clip(((q - a) * ab).sum(axis=2) / denom, 0.0, 1.0)
        proj = a + t[..., None] * ab
        idx = np.argmin(((q - proj) ** 2).sum(axis=2), axis=1)
        out[start : start + chunk] = proj[np.arange(len(idx)), idx]
    return out


def _match_vertices(points: np.ndarray, ref: np.ndarray, closed: bool) -> np.ndarray:
    if len(points) == len(ref):
        return _cyclic_align(points, ref) if closed else points
    _warn_once(
        "Vertex count changed between evaluations; using closest-point "
        "correspondence. Gradients only capture normal motion of the boundary. "
        "Fix the discretization (e.g. pass `npoints`) for exact correspondence."
    )
    return _closest_points(ref, points, closed=closed)


def align(raw: Raw, spec: Spec, ref: Raw) -> Raw:
    """Maps `raw` onto the structure of `ref` (which must follow `spec`)."""
    if spec.kind == "tree":
        assert isinstance(raw, RawTree)
        shapes = tuple(leaf.shape for leaf in raw.leaves)
        if raw.treedef != spec.treedef or shapes != spec.shapes:
            raise ValueError(
                "Output structure changed between evaluations "
                f"({spec.treedef}, {spec.shapes}) -> ({raw.treedef}, {shapes})."
            )
        return raw

    if spec.kind == "path":
        assert isinstance(raw, RawPath) and isinstance(ref, RawPath)
        return RawPath(points=_match_vertices(raw.points, ref.points, closed=False))

    assert isinstance(raw, RawGeometry) and isinstance(ref, RawGeometry)
    polygons: dict[LayerKey, list[np.ndarray]] = {}
    for key, ref_polys in ref.polygons.items():
        new_polys = raw.polygons.get(key, [])
        aligned = list(ref_polys)  # unmatched polygons keep reference coords
        if len(new_polys) != len(ref_polys):
            _warn_once(
                f"Number of polygons on layer {key} changed between evaluations "
                f"({len(ref_polys)} -> {len(new_polys)}); unmatched polygons get "
                "zero gradient."
            )
        if new_polys:
            ref_c = np.array([_centroid(p) for p in ref_polys])
            new_c = np.array([_centroid(p) for p in new_polys])
            cost = ((ref_c[:, None, :] - new_c[None, :, :]) ** 2).sum(axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                aligned[r] = _match_vertices(new_polys[c], ref_polys[r], closed=True)
        polygons[key] = aligned

    ports = {}
    for name in spec.ports:
        if name not in raw.ports:
            raise ValueError(f"Port {name!r} disappeared between evaluations.")
        ports[name] = raw.ports[name]
    return RawGeometry(polygons=polygons, ports=ports)
