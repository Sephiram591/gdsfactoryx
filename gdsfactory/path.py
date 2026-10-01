"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component
The CrossSection defines the layer numbers, widths and offsets

All path points are float64 ``jax.numpy`` arrays: every function here is
differentiable (in eager mode) with respect to its float arguments (radius,
angle, length, widths, offsets, waypoints, ...). Discrete choices (number of
points, branches) are made on the concrete primal values.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar, cast, overload

import jax
import numpy as np
import numpy.typing as npt

import gdsfactory.typings  # noqa: F401  (import order: typings before cross_section)
from gdsfactory._cell import cell
from gdsfactory._jax import (
    Array,
    asarray,
    is_tracer,
    jnp,
    xp,
    xset,
    round_st,
    stop_gradient,
    to_float,
    to_numpy,
)
from gdsfactory.component import Box, Component, ComponentAllAngle
from gdsfactory.cross_section import (
    CrossSection,
    Section,
    Transition,
    TransitionAsymmetric,
)
from gdsfactory.pdk import get_layer_name
from gdsfactory.transform import Transform, rotate_points
from gdsfactory.typings import (
    AngleInDegrees,
    AnyComponent,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    WidthTypes,
)


def _simplify(points: Array, tolerance: float) -> Array:
    """Simplifies a polyline (shapely topology-preserving Douglas-Peucker).

    The kept vertices are chosen on the concrete values; the result is a
    differentiable selection of the input points.
    """
    import shapely.geometry as sg

    pts_np = to_numpy(points)
    ls = sg.LineString(pts_np)
    simple = np.asarray(ls.simplify(tolerance=tolerance).coords)
    # map simplified coordinates back to indices of the original points
    idx: list[int] = []
    j = 0
    for q in simple:
        while j < len(pts_np) and not np.array_equal(pts_np[j], q):
            j += 1
        if j == len(pts_np):
            return asarray(simple)
        idx.append(j)
        j += 1
    return asarray(points)[np.asarray(idx)]


def reflect_points(
    points: Any,
    p1: Any = (0, 0),
    p2: Any = (1, 0),
) -> Array:
    """Reflects points across the line formed by p1 and p2.

    ``points`` may be input as either single points [1,2] or array-like[N][2],
    and will return in kind.
    """
    pts = asarray(points)
    return_single_point = pts.ndim == 1
    pts = xp.atleast_2d(pts)
    p1_array = asarray(p1)
    p2_array = asarray(p2)

    line_vec = p2_array - p1_array
    line_vec_norm = xp.sum(line_vec**2)
    proj = xp.sum(line_vec * (pts - p1_array), axis=-1, keepdims=True)
    reflected = 2 * (p1_array + (p2_array - p1_array) * proj / line_vec_norm) - pts
    return reflected[0] if return_single_point else reflected


def _angle_deg(dy: Any, dx: Any) -> Any:
    return xp.arctan2(dy, dx) / xp.pi * 180


def _mod360(a: Any) -> Any:
    if is_tracer(a) or isinstance(a, jax.Array):
        return xp.mod(a, 360)
    return float(np.mod(a, 360))


def _is_points_like(path: Any) -> bool:
    if isinstance(path, jax.Array | np.ndarray) or is_tracer(path):
        return path.ndim == 2 and path.shape[1] == 2
    if isinstance(path, list | tuple) and path and not isinstance(path[0], Path):
        first = path[0]
        if isinstance(first, list | tuple | np.ndarray | jax.Array):
            return len(first) > 0 and not isinstance(first[0], Path | list | tuple)
        return False
    return False


class Path:
    """You can extrude a Path with a CrossSection to create a Component.

    Parameters:
        path: array-like[N][2], Path, or list of Paths.
    """

    def __init__(
        self,
        path: Any = None,
        start_angle: float | None = None,
        end_angle: float | None = None,
    ) -> None:
        self.points: Array = asarray([[0.0, 0.0]])
        self.start_angle: Any = 0.0
        self.end_angle: Any = 0.0
        self.info: dict[str, Any] = {}
        if path is not None:
            if isinstance(path, Path):
                self.points = path.points
                self.start_angle = path.start_angle
                self.end_angle = path.end_angle
            elif _is_points_like(path):
                self.points = _as_points(path)
                if len(self.points) > 1:
                    d1 = self.points[1] - self.points[0]
                    self.start_angle = _angle_deg(d1[1], d1[0])
                    d2 = self.points[-1] - self.points[-2]
                    self.end_angle = _angle_deg(d2[1], d2[0])
            elif isinstance(path, list | tuple) and len(path) > 0:
                self.append(list(path))
            else:
                raise ValueError(
                    "Path() the `path` argument must be either blank, a path Object, "
                    "an array-like[N][2] list of points, or a list of these"
                )
        if start_angle is not None:
            self.start_angle = _mod360(start_angle)
        if end_angle is not None:
            self.end_angle = _mod360(end_angle)

    def __repr__(self) -> str:
        return (
            f"Path(start_angle={to_float(self.start_angle)}, "
            f"end_angle={to_float(self.end_angle)}, "
            f"points={to_numpy(self.points)})"
        )

    def __len__(self) -> int:
        return len(self.points)

    def __iadd__(self, path_or_points: Any) -> Path:
        return self.append(path_or_points)

    def __add__(self, path: Any) -> Path:
        new = self.copy()
        return new.append(path)

    # ------------------------------------------------------------ geometry
    def transform(self, trans: Any, /) -> Path:
        """Transforms the Path in place (Transform or KLayout transformation)."""
        if not isinstance(trans, Transform):
            import klayout.db as kdb

            if isinstance(trans, kdb.DTrans):
                trans = kdb.DCplxTrans(trans)
            elif isinstance(trans, kdb.Trans):
                trans = kdb.DCplxTrans(trans.to_dtype(1e-3))
            elif isinstance(trans, kdb.ICplxTrans):
                trans = trans.to_itrans(1e-3)
            trans = Transform.from_klayout(trans)
        self.points = trans.apply(self.points)
        self.start_angle = trans.apply_angle(self.start_angle)
        self.end_angle = trans.apply_angle(self.end_angle)
        return self

    def dbbox(self, layer: int | None = None) -> Box:
        mn = xp.min(self.points, axis=0)
        mx = xp.max(self.points, axis=0)
        return Box(mn[0], mn[1], mx[0], mx[1])

    ibbox = dbbox
    bbox = dbbox

    @property
    def kcl(self) -> Any:
        import gdsfactory as gf

        return gf.kcl

    def bbox_np(self) -> npt.NDArray[np.float64]:
        pts = to_numpy(self.points)
        return np.array([pts.min(axis=0), pts.max(axis=0)], dtype=np.float64)

    @property
    def xmin(self) -> Any:
        return xp.min(self.points[:, 0])

    @property
    def xmax(self) -> Any:
        return xp.max(self.points[:, 0])

    @property
    def ymin(self) -> Any:
        return xp.min(self.points[:, 1])

    @property
    def ymax(self) -> Any:
        return xp.max(self.points[:, 1])

    @property
    def x(self) -> Any:
        return (self.xmin + self.xmax) / 2

    @property
    def y(self) -> Any:
        return (self.ymin + self.ymax) / 2

    @property
    def center(self) -> Array:
        return xp.stack([self.x, self.y])

    @property
    def xsize(self) -> Any:
        return self.xmax - self.xmin

    @property
    def ysize(self) -> Any:
        return self.ymax - self.ymin

    def move(self, *args: Any) -> Path:
        """Moves: move((dx, dy)), move(dx, dy) or move(origin, destination)."""
        from gdsfactory._ports import _move_args

        dx, dy = _move_args(args)
        self.points = self.points + xp.stack([asarray(dx), asarray(dy)])
        return self

    dmove = move

    def movex(self, dx: Any) -> Path:
        return self.move(dx, 0.0)

    def movey(self, dy: Any) -> Path:
        return self.move(0.0, dy)

    def rotate(self, angle: Any = 45, center: Any = (0.0, 0.0)) -> Path:
        """Rotates the path (degrees, counter-clockwise) around center."""
        if not is_tracer(angle) and float(angle) == 0:
            return self
        self.points = rotate_points(self.points, angle, center)
        self.start_angle = _mod360(self.start_angle + angle)
        self.end_angle = _mod360(self.end_angle + angle)
        return self

    drotate = rotate

    def append(self, path: Any) -> Path:
        """Attach Path to the end of this Path.

        The input path automatically rotates and translates such that it continues
        smoothly from the previous segment.

        Args:
            path: Path, array-like[N][2], or list of Paths. The input path that will be appended.
        """
        if isinstance(path, Path):
            start_angle = path.start_angle
            end_angle = path.end_angle
            points = path.points
        elif _is_points_like(path):
            points = _as_points(path)
            start_angle, end_angle = 0.0, 0.0
            if len(points) > 1:
                d1 = points[1] - points[0]
                start_angle = _angle_deg(d1[1], d1[0])
                d2 = points[-1] - points[-2]
                end_angle = _angle_deg(d2[1], d2[0])
        elif isinstance(path, list | tuple):
            for p in path:
                self.append(p)
            return self
        else:
            raise ValueError(
                "Path.append() the `path` argument must be either "
                "a Path object, an array-like[N][2] list of points, or a list of these"
            )

        rot = self.end_angle - start_angle
        if is_tracer(rot) or to_float(rot) != 0:
            points = rotate_points(points, angle=rot)
        points = points + (self.points[-1, :] - points[0, :])
        self.end_angle = _mod360(end_angle + self.end_angle - start_angle)
        self.points = xp.concatenate([self.points, points[1:]])
        return self

    def offset(self, offset: Any = 0) -> Path:
        """Offsets Path so that it follows the Path centerline plus an offset.

        The offset can either be a fixed value, or a function
        of the form my_offset(t) where t goes from 0->1

        Args:
            offset: int or float, callable. Magnitude of the offset
        """
        if callable(offset):
            lengths = _cumulative_lengths(self.points)
            points = self.centerpoint_offset_curve(
                self.points,
                offset_distance=offset(lengths / lengths[-1]),
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            tol = 1e-6
            ds = tol / lengths[-1]
            ny1 = offset(ds) - offset(0)
            start_angle = _angle_deg(-ny1, tol) + self.start_angle
            ny2 = offset(1) - offset(1 - ds)
            end_angle = _angle_deg(-ny2, tol) + self.end_angle
        elif not is_tracer(offset) and to_float(offset) == 0:
            return self
        else:
            points = self.centerpoint_offset_curve(
                self.points,
                offset_distance=offset,
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            start_angle = self.start_angle
            end_angle = self.end_angle

        self.points = points
        self.start_angle = start_angle
        self.end_angle = end_angle
        return self

    def centerpoint_offset_curve(
        self,
        points: Any,
        offset_distance: Any,
        start_angle: Any = None,
        end_angle: Any = None,
    ) -> Array:
        """Creates a offset curve computing the centerpoint offset of x and y points."""
        points = asarray(points)
        cos_mid, sin_mid, sin_half = _compute_offset_directions(points)
        return _offset_curve_from_directions(
            points,
            offset_distance,
            cos_mid,
            sin_mid,
            sin_half,
            start_angle=start_angle,
            end_angle=end_angle,
        )

    def _parametric_offset_curve(
        self,
        points: Any,
        offset_distance: Any,
        start_angle: Any = None,
        end_angle: Any = None,
    ) -> Array:
        """Creates a parametric offset by using gradient of the supplied x and y points."""
        points = asarray(points)
        x = points[:, 0]
        y = points[:, 1]
        dxdt = xp.gradient(x)
        dydt = xp.gradient(y)
        if start_angle is not None:
            dxdt = xset(dxdt, (0), xp.cos(start_angle * xp.pi / 180))
            dydt = xset(dydt, (0), xp.sin(start_angle * xp.pi / 180))
        if end_angle is not None:
            dxdt = xset(dxdt, (-1), xp.cos(end_angle * xp.pi / 180))
            dydt = xset(dydt, (-1), xp.sin(end_angle * xp.pi / 180))
        norm = xp.sqrt(dxdt**2 + dydt**2)
        x_offset = x + offset_distance * dydt / norm
        y_offset = y - offset_distance * dxdt / norm
        return xp.stack([x_offset, y_offset]).T

    def length(self) -> Any:
        """Return cumulative length (rounded to 1e-3 um, straight-through gradient)."""
        return round_st(self.length_exact(), 3)

    def length_exact(self) -> Any:
        d = xp.diff(self.points, axis=0)
        return xp.sum(xp.sqrt(xp.sum(d**2, axis=1)))

    def curvature(self) -> tuple[Array, Array]:
        """Calculates Path curvature (numerically).

        Returns:
            s: array-like[N] The arc-length of the Path
            K: array-like[N] The curvature of the Path
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        dx = xp.diff(x)
        dy = xp.diff(y)
        ds = xp.sqrt(dx**2 + dy**2)
        s = xp.cumsum(ds)
        theta = xp.unwrap(xp.arctan2(dy, dx))

        match len(ds):
            case 0 | 1:
                k = asarray([np.inf])
            case 2:
                k = xp.nan_to_num(_gradient(theta, s, edge_order=1), nan=np.inf)
            case _:
                k = _gradient(theta, s, edge_order=2)
        return s, k

    def __hash__(self) -> int:
        return self.hash_geometry()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return False
        return (
            np.array_equal(to_numpy(self.points), to_numpy(other.points))
            and to_float(self.start_angle) == to_float(other.start_angle)
            and to_float(self.end_angle) == to_float(other.end_angle)
        )

    def hash_geometry(self, precision: float = 1e-4) -> int:
        """Computes an SHA1 hash of the points and start/end angles."""
        magic_offset = 0.17048614
        final_hash = hashlib.sha1()
        adjusted_points = (
            ((to_numpy(self.points) / precision) + magic_offset).round().astype(np.int64)
        )
        final_hash.update(adjusted_points.tobytes())
        adjusted_angles = np.array([to_float(self.start_angle), to_float(self.end_angle)])
        adjusted_angles = (
            ((adjusted_angles / precision) + magic_offset).round().astype(np.int64)
        )
        final_hash.update(adjusted_angles.tobytes())
        return int.from_bytes(final_hash.digest(), byteorder="big")

    def plot(self) -> None:
        """Plot path in matplotlib."""
        import matplotlib.pyplot as plt

        pts = to_numpy(self.points)
        plt.plot(pts[:, 0], pts[:, 1])
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: bool = False,
        register_cross_section: bool = False,
    ) -> AnyComponent:
        """Returns Component by extruding a Path with a CrossSection.

        Args:
            cross_section: to extrude.
            layer: optional layer.
            width: optional width in um.
            simplify: Tolerance value for the simplification algorithm.
            all_angle: kept for API compatibility (all components support any angle).
            register_cross_section: if True, the cross_section factory is registered in the active PDK.
        """
        return extrude(
            p=self,
            cross_section=cross_section,
            layer=layer,
            width=width,
            simplify=simplify,
            all_angle=all_angle,
            register_cross_section=register_cross_section,
        )

    def extrude_transition(
        self,
        transition: Transition | TransitionAsymmetric,
        all_angle: bool = False,
    ) -> AnyComponent:
        """Extrudes a path along a transition."""
        return extrude_transition(p=self, transition=transition, all_angle=all_angle)

    def copy(self) -> Path:
        """Returns a copy of the Path."""
        p = Path()
        p.info = self.info.copy()
        p.points = self.points
        p.start_angle = self.start_angle
        p.end_angle = self.end_angle
        return p

    def mirror(self, p1: Any = (0, 1), p2: Any = (0, 0)) -> Path:
        """Mirrors the Path across the line formed between the two specified points."""
        self.points = reflect_points(self.points, p1, p2)
        p1a = asarray(p1)
        p2a = asarray(p2)
        angle = _angle_deg(p2a[1] - p1a[1], p2a[0] - p1a[0])
        if self.start_angle is not None:
            self.start_angle = _mod360(2 * angle - self.start_angle)
        if self.end_angle is not None:
            self.end_angle = _mod360(2 * angle - self.end_angle)
        return self

    dmirror = mirror

    def mirror_x(self, x: Any = 0.0) -> Path:
        return self.mirror((x, 1), (x, 0))

    def mirror_y(self, y: Any = 0.0) -> Path:
        return self.mirror((1, y), (0, y))

    def invert(self) -> Path:
        """Inverts the Path by reversing the order of its points."""
        self.points = self.points[::-1]
        self.start_angle, self.end_angle = (
            _mod360(self.end_angle + 180),
            _mod360(self.start_angle + 180),
        )
        return self


def _as_points(path: Any) -> Array:
    if isinstance(path, jax.Array | np.ndarray) or is_tracer(path):
        return asarray(path)
    from gdsfactory._jax import points_array

    if any(len(p) != 2 for p in path):
        raise ValueError("Path points must be (x, y) pairs")
    return points_array(path)


def _cumulative_lengths(points: Array) -> Array:
    d = xp.diff(points, axis=0)
    lengths = xp.cumsum(xp.sqrt(xp.sum(d**2, axis=1)))
    return xp.concatenate([xp.zeros(1), lengths])


def _gradient(f: Array, x: Array, edge_order: int = 1) -> Array:
    """jnp version of np.gradient(f, x) for non-uniform 1D spacing."""
    n = f.shape[0]
    if n < 2:
        return xp.full_like(f, xp.inf)
    out = xp.zeros_like(f)
    dx1 = x[1:-1] - x[:-2]
    dx2 = x[2:] - x[1:-1]
    a = -(dx2) / (dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    out = xset(out, slice(1, -1), a * f[:-2] + b * f[1:-1] + c * f[2:])
    if edge_order == 1 or n < 3:
        out = xset(out, (0), (f[1] - f[0]) / (x[1] - x[0]))
        out = xset(out, (-1), (f[-1] - f[-2]) / (x[-1] - x[-2]))
    else:
        dx1 = x[1] - x[0]
        dx2 = x[2] - x[1]
        a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
        b = (dx1 + dx2) / (dx1 * dx2)
        c = -dx1 / (dx2 * (dx1 + dx2))
        out = xset(out, (0), a * f[0] + b * f[1] + c * f[2])
        dx1 = x[-2] - x[-3]
        dx2 = x[-1] - x[-2]
        a = dx2 / (dx1 * (dx1 + dx2))
        b = -(dx2 + dx1) / (dx1 * dx2)
        c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
        out = xset(out, (-1), a * f[-3] + b * f[-2] + c * f[-1])
    return out


PathFactory = Callable[..., Path]
T = TypeVar("T", float, npt.NDArray[np.floating[Any]])


def _sinusoidal_transition(y1: Any, y2: Any) -> Callable[[Any], Any]:
    dy = y2 - y1

    def sine(t: Any) -> Any:
        return y1 + (1 - xp.cos(xp.pi * t)) * dy / 2

    return sine


def _parabolic_transition(y1: Any, y2: Any) -> Callable[[Any], Any]:
    dy = y2 - y1

    def parabolic(t: Any) -> Any:
        return y1 + xp.sqrt(t) * dy

    return parabolic


def _linear_transition(y1: Any, y2: Any) -> Callable[[Any], Any]:
    dy = y2 - y1

    def linear(t: Any) -> Any:
        return y1 + t * dy

    return linear


def transition_exponential(y1: Any, y2: Any, exp: float = 0.5) -> Callable[[Any], Any]:
    """Returns the function for an exponential transition.

    Args:
        y1: start width in um.
        y2: end width in um.
        exp: exponent.
    """
    return lambda t: y1 + (y2 - y1) * t**exp


adiabatic_polyfit_TE1550SOI_220nm = np.array(
    [
        1.02478963e-09,
        -8.65556534e-08,
        3.32415694e-06,
        -7.68408985e-05,
        1.19282177e-03,
        -1.31366332e-02,
        1.05721429e-01,
        -6.31057637e-01,
        2.80689677e00,
        -9.26867694e00,
        2.24535191e01,
        -3.90664800e01,
        4.71899278e01,
        -3.74726005e01,
        1.77381560e01,
        -1.12666286e00,
    ]
)


def transition_adiabatic(
    w1: float,
    w2: float,
    neff_w: Callable[[Any], Any],
    wavelength: float = 1.55,
    alpha: float = 1,
    max_length: float = 200,
    num_points_ODE: int = 2000,
) -> tuple[Array, Array]:
    """Returns the points for an optimal adiabatic transition for well-guided modes.

    The ODE dw/dx = alpha * wavelength / (neff(w) * w) is integrated with
    ``jax.experimental.ode.odeint``, so the result is differentiable with
    respect to w1, w2, wavelength and alpha (``neff_w`` must be jax-traceable).

    Args:
        w1: start width in um.
        w2: end width in um.
        neff_w: a callable that returns the effective index as a function of width.
        wavelength: wavelength, in same units as widths.
        alpha: parameter that scales the rate of width change.
        max_length: maximum length in um.
        num_points_ODE: number of samplings points for the ODE solve.

    References:
        [1] Burns, W. K., et al. "Optical waveguide parabolic coupling horns."
            Appl. Phys. Lett., vol. 30, no. 1, 1 Jan. 1977, pp. 28-30, doi:10.1063/1.89199.
        [2] Fu, Yunfei, et al. "Efficient adiabatic silicon-on-insulator waveguide taper."
            Photonics Res., vol. 2, no. 3, 1 June 2014, pp. A41-A44, doi:10.1364/PRJ.2.000A41.
    """
    from jax.experimental.ode import odeint

    def dWdx(w: Any, x: Any, wavelength: Any, alpha: Any) -> Any:
        return alpha * wavelength / (neff_w(w) * w)

    if to_float(w2) < to_float(w1):
        wmin, wmax, order = w2, w1, -1
    else:
        wmin, wmax, order = w1, w2, 1

    x = xp.linspace(0, max_length, num_points_ODE)
    sol = odeint(dWdx, asarray(wmin), x, asarray(wavelength), asarray(alpha))
    mask = to_numpy(sol) < to_float(wmax)
    xs = x[mask]
    ws = sol[mask]
    return xs, ws[::order]


def transition(
    cross_section1: CrossSectionSpec,
    cross_section2: CrossSectionSpec,
    width_type: WidthTypes | Callable[[float, float, float], float] = "sine",
    offset_type: WidthTypes | Callable[[float, float, float], float] = "sine",
) -> Transition:
    """Returns a smoothly-transitioning between two CrossSections.

    Only cross-sectional elements that have the `name` (as in X.add(..., name = 'wg') )
    parameter specified in both input CrosSections will be created.
    Port names will be cloned from the input CrossSections in reverse.

    Args:
        cross_section1: First CrossSection.
        cross_section2: Second CrossSection.
        width_type: 'sine', 'parabolic', 'linear' or Callable.
        offset_type: 'sine', 'parabolic', 'linear' or Callable.
    """
    from gdsfactory.pdk import get_cross_section, get_layer

    X1 = get_cross_section(cross_section1)
    X2 = get_cross_section(cross_section2)

    layers1 = {get_layer(section.layer) for section in X1.sections}
    layers2 = {get_layer(section.layer) for section in X2.sections}
    layers1.add(get_layer(X1.layer))
    layers2.add(get_layer(X2.layer))

    if not layers1.intersection(layers2):
        raise ValueError(
            f"transition() found no common layers X1 {layers1} and X2 {layers2}"
        )

    return Transition(
        cross_section1=X1,
        cross_section2=X2,
        width_type=width_type,
        offset_type=offset_type,
    )


def transition_asymmetric(
    cross_section1: CrossSectionSpec,
    cross_section2: CrossSectionSpec,
    width_type1: WidthTypes | Callable[[float, float, float], float] = "sine",
    width_type2: WidthTypes | Callable[[float, float, float], float] = "sine",
    offset_type1: WidthTypes | Callable[[float, float, float], float] = "sine",
    offset_type2: WidthTypes | Callable[[float, float, float], float] = "sine",
) -> TransitionAsymmetric:
    """Returns a smoothly-transitioning object between two CrossSections with asymmetric transitions."""
    from gdsfactory.pdk import get_cross_section, get_layer

    X1 = get_cross_section(cross_section1)
    X2 = get_cross_section(cross_section2)

    layers1 = {get_layer(section.layer) for section in X1.sections}
    layers2 = {get_layer(section.layer) for section in X2.sections}
    layers1.add(get_layer(X1.layer))
    layers2.add(get_layer(X2.layer))

    if not layers1.intersection(layers2):
        raise ValueError(
            f"transition_asymmetric() found no common layers X1 {layers1} and X2 {layers2}"
        )

    return TransitionAsymmetric(
        cross_section1=X1,
        cross_section2=X2,
        width_type1=width_type1,
        width_type2=width_type2,
        offset_type1=offset_type1,
        offset_type2=offset_type2,
    )


@cell
def along_path(
    p: Path,
    component: ComponentSpec,
    spacing: float,
    padding: float,
) -> Component:
    """Returns Component containing many copies of `component` along `p`.

    Places as many copies of `component` along each segment of `p` as possible
    under the given constraints. `spacing` is always followed precisely, but
    actual `padding` may exceed the provided value to place components evenly.

    Args:
        p: Path to place components along.
        component: Component to repeat along the path. The unrotated version of \
                this object should be oriented for placement on a horizontal line.
        spacing: distance between component placements.
        padding: minimum distance from the path start to the first component.
    """
    from gdsfactory.pdk import get_component

    component = get_component(component)

    length = p.length()
    number = (to_float(length) - 2 * to_float(padding)) // to_float(spacing) + 1

    c = Component()

    cum_dist = 0.0
    next_component = (length - (number - 1) * spacing) / 2
    stop = length - next_component

    pts = p.points
    for i in range(len(pts) - 1):
        start_pt = pts[i]
        segment_vector = pts[i + 1] - start_pt
        segment_length = xp.sqrt(xp.sum(segment_vector**2))
        unit_vector = segment_vector / segment_length
        angle = xp.rad2deg(xp.arctan2(segment_vector[1], segment_vector[0]))

        while to_float(next_component) <= to_float(cum_dist + segment_length) and to_float(
            next_component
        ) <= to_float(stop):
            added_dist = next_component - cum_dist
            offset = added_dist * unit_vector
            component_ref = c << component
            component_ref.rotate(angle).move(start_pt + offset)
            next_component = next_component + spacing
        cum_dist = cum_dist + segment_length

    return c


def _get_named_sections(sections: tuple[Section, ...]) -> dict[str, Section]:
    named_sections: dict[str, Section] = {}
    for section in sections:
        if section.skip_transition:
            continue
        name = section.name or get_layer_name(section.layer)
        if name in named_sections:
            raise ValueError(
                f"Duplicate name or layer '{name}' of section used for cross-section in transition. Cross-sections with multiple Sections for a single layer must have unique names for each section"
            )
        named_sections[name] = section
    return named_sections


def _is_implicit_section_name(name: str) -> bool:
    """Return whether a section name was assigned by gdsfactory."""
    return name == "_default" or (
        len(name) == 10
        and name.startswith("s_")
        and all(character in "0123456789abcdef" for character in name[2:])
    )


def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: bool = False,
    register_cross_section: bool = False,
) -> AnyComponent:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using any CrossSection returning a Component
    The CrossSection defines the layer numbers, widths and offsets

    Args:
        p: a path is a list of points (arc, straight, euler).
        cross_section: to extrude.
        layer: optional layer to extrude.
        width: optional width to extrude.
        simplify: Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting polygon \
                by more than the value listed here will be removed.
        all_angle: kept for API compatibility.
        register_cross_section: if True, registers the cross-section factory \
            used for extrusion in the global cross-section registry.
    """
    from gdsfactory.pdk import get_cross_section, get_layer

    if (cross_section is None) == (layer is None):
        raise ValueError("Provide exactly one of 'cross_section' or 'layer'")
    if layer is not None and width is None:
        raise ValueError("When providing 'layer', 'width' must also be provided")

    if cross_section is not None:
        x = (
            get_cross_section(cross_section, width=width)
            if width is not None
            else get_cross_section(cross_section)
        )
    else:
        s = Section(
            width=cast("float", width),
            layer=cast("LayerSpec", layer),
            port_names=("o1", "o2"),
            port_types=("optical", "optical"),
        )
        x = get_cross_section(CrossSection(sections=(s,)))

    c = ComponentAllAngle() if all_angle else Component()
    path_length = p.length()

    layer = get_layer(layer or x.layer)
    _dir_cache: tuple[Array, Array, Array] | None = None
    points = p.points
    start_angle = p.start_angle
    end_angle = p.end_angle

    for section in x.sections:
        p_sec = p.copy()
        port_names = section.port_names
        port_types = section.port_types
        hidden = section.hidden

        offset_value: Any = section.offset
        width_value: Any = section.width
        width_function = section.width_function
        offset_function = section.offset_function
        layer = get_layer(section.layer)

        insets = section.insets
        has_insets = bool(insets) and tuple(to_float(i) for i in insets) != (0, 0)  # type: ignore[union-attr]
        path_changed = has_insets

        if has_insets:
            assert insets is not None
            trimmed = _inset_path(p_sec, insets)
            if trimmed is None:
                warnings.warn(
                    f"Cannot apply delay to Section '{section.name}', delay results in points outside "
                    f"of original path.",
                    stacklevel=3,
                )
                continue
            p_sec = trimmed

        if callable(offset_function):
            p_sec.offset(offset_function)
            path_changed = True
            offset_value = 0
        end_angle = p_sec.end_angle
        start_angle = p_sec.start_angle
        points = p_sec.points
        if callable(width_function):
            lengths = _cumulative_lengths(p_sec.points)
            width_value = width_function(lengths / lengths[-1])

        assert width_value is not None

        dy1 = offset_value + width_value / 2
        dy2 = offset_value - width_value / 2

        if path_changed:
            cos_mid, sin_mid, sin_half = _compute_offset_directions(points)
        elif _dir_cache is not None:
            cos_mid, sin_mid, sin_half = _dir_cache
        else:
            cos_mid, sin_mid, sin_half = _compute_offset_directions(points)
            _dir_cache = (cos_mid, sin_mid, sin_half)

        points1, points2 = _apply_offsets(
            points,
            dy1,
            dy2,
            cos_mid,
            sin_mid,
            sin_half,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        if isinstance(simplify, bool):
            raise ValueError("simplify argument must be a number (e.g. 1e-3) or None")

        with_simplify = section.simplify or simplify

        if with_simplify:
            points1 = _simplify(points1, tolerance=with_simplify)
            points2 = _simplify(points2, tolerance=with_simplify)

        points_poly = xp.concatenate([points1, points2[::-1, :]])
        section_length = p_sec.length() if path_changed else path_length

        if not hidden and to_float(section_length) > 1e-3:
            c.add_polygon(points_poly, layer=layer)

        scalar_width = xp.ndim(width_value) == 0
        if port_names[0]:
            port_width = width_value if scalar_width else width_value[0]
            port_orientation = _mod360(p_sec.start_angle + 180)
            center = (points1[0] + points2[0]) / 2
            c.add_port(
                name=port_names[0],
                layer=layer,
                port_type=port_types[0],
                width=port_width,
                orientation=port_orientation,
                center=center,
                cross_section=x,
                register_cross_section=register_cross_section,
            )
        if port_names[1]:
            port_width = width_value if scalar_width else width_value[-1]
            port_orientation = _mod360(p_sec.end_angle)
            center = (points1[-1] + points2[-1]) / 2
            c.add_port(
                name=port_names[1],
                layer=layer,
                port_type=port_types[1],
                width=port_width,
                center=center,
                orientation=port_orientation,
                cross_section=x,
                register_cross_section=register_cross_section,
            )

    c.info["length"] = path_length

    for via in x.components_along_path:
        if via.offset:
            points_offset = p.centerpoint_offset_curve(
                points,
                offset_distance=via.offset,
                start_angle=start_angle,
                end_angle=end_angle,
            )
            _p = Path(points_offset)
        else:
            _p = p
        _ = c << along_path(
            p=_p, component=via.component, spacing=via.spacing, padding=via.padding
        )
    return c


def _inset_path(p_sec: Path, insets: tuple[Any, Any]) -> Path | None:
    """Trims the start/end of a path by insets (um). Differentiable in insets and points."""
    p_pts = p_sec.points
    seg = xp.diff(p_pts, axis=0)
    seg_len = xp.sqrt(xp.sum(seg**2, axis=1))
    fwd = xp.cumsum(seg_len)
    rev = xp.cumsum(seg_len[::-1])
    fwd_np = to_numpy(fwd)
    rev_np = to_numpy(rev)
    i0 = to_float(insets[0])
    i1 = to_float(insets[1])
    if i0 > fwd_np[-1] and i1 > fwd_np[-1]:
        return None

    start_diff_idx = int(np.argwhere(fwd_np >= i0)[0, 0])
    reversed_stop_diff_idx = int(np.argwhere(rev_np >= i1)[0, 0])
    stop_diff_idx = (len(seg) - 1) - reversed_stop_diff_idx

    v_start = -seg[start_diff_idx]
    v_stop = seg[stop_diff_idx]
    v_start_direction = v_start / xp.linalg.norm(v_start)
    v_stop_direction = v_stop / xp.linalg.norm(v_stop)

    start_inset_remainder = fwd[start_diff_idx] - insets[0]
    stop_inset_remainder = rev[reversed_stop_diff_idx] - insets[1]

    new_start_point = v_start_direction * start_inset_remainder + p_pts[start_diff_idx + 1]
    new_stop_point = v_stop_direction * stop_inset_remainder + p_pts[stop_diff_idx]

    trimmed = xp.concatenate(
        [
            new_start_point[None, :],
            p_pts[start_diff_idx + 1 : stop_diff_idx + 1],
            new_stop_point[None, :],
        ]
    )
    tn = to_numpy(trimmed)
    keep = np.concatenate(([True], np.any(np.diff(tn, axis=0) != 0, axis=1)))
    trimmed = trimmed[np.nonzero(keep)[0]]
    return Path(
        trimmed,
        start_angle=p_sec.start_angle if i0 == 0 else None,
        end_angle=p_sec.end_angle if i1 == 0 else None,
    )


def extrude_transition(
    p: Path,
    transition: Transition | TransitionAsymmetric,
    all_angle: bool = False,
) -> AnyComponent:
    """Extrudes a path along a transition, allowing different transition methods for the upper and lower edges.

    Args:
        p: Path to extrude.
        transition: Transition or TransitionAsymmetric object describing the cross-sections and default transition types.
        all_angle: kept for API compatibility.

    Returns:
        Component: The extruded component with the specified transition methods for each edge.
    """
    from gdsfactory.pdk import get_cross_section, get_layer

    c = ComponentAllAngle() if all_angle else Component()

    if not isinstance(transition, Transition | TransitionAsymmetric):
        raise TypeError(
            f"Expected Transition or TransitionAsymmetric, got {type(transition).__name__}"
        )

    x1 = get_cross_section(transition.cross_section1)
    x2 = get_cross_section(transition.cross_section2)
    if isinstance(transition, TransitionAsymmetric):
        width_type1 = transition.width_type1
        width_type2 = transition.width_type2
        offset_type1 = transition.offset_type1
        offset_type2 = transition.offset_type2
    else:
        width_type1 = width_type2 = transition.width_type
        offset_type1 = offset_type2 = transition.offset_type

    named_sections1 = _get_named_sections(x1.sections)
    named_sections2 = _get_named_sections(x2.sections)

    names1 = list(named_sections1.keys())
    names2 = list(named_sections2.keys())

    common_sections = set(names1).intersection(names2)
    if not common_sections and len(names1) == len(names2) == 1:
        name1, name2 = names1[0], names2[0]
        section1 = named_sections1[name1]
        section2 = named_sections2[name2]
        if (
            _is_implicit_section_name(name1)
            and _is_implicit_section_name(name2)
            and get_layer(section1.layer) == get_layer(section2.layer)
        ):
            named_sections2[name1] = named_sections2.pop(name2)
            common_sections = {name1}
    if not common_sections:
        raise ValueError(
            f"transition() found no common section names X1 {names1} and X2 {names2}"
        )

    lengths_abs = _cumulative_lengths(p.points)
    path_length = round_st(lengths_abs[-1], 3)
    lengths = lengths_abs / lengths_abs[-1]

    def _make(kind: Any, v1: Any, v2: Any) -> Callable[[Any], Any]:
        if kind == "linear":
            return _linear_transition(v1, v2)
        if kind == "sine":
            return _sinusoidal_transition(v1, v2)
        if kind == "parabolic":
            return _parabolic_transition(v1, v2)
        if callable(kind):
            return lambda t: kind(t, v1, v2)
        raise NotImplementedError

    # deterministic order (set iteration order is arbitrary)
    for section_name in [n for n in names1 if n in common_sections]:
        section1 = named_sections1[section_name]
        section2 = named_sections2[section_name]
        port_names = section1.port_names
        port_types = section1.port_types

        offset1, offset2 = section1.offset, section2.offset
        width1, width2 = section1.width, section2.width

        offset_func1 = _make(offset_type1, offset1, offset2)
        width_func1 = _make(width_type1, width1, width2)
        offset_func2 = _make(offset_type2, offset1, offset2)
        width_func2 = _make(width_type2, width1, width2)

        layer1 = get_layer(section1.layer)
        layer2 = get_layer(section2.layer)
        if layer1 != layer2:
            hidden = True
            layers = [layer1, layer2]
            layer = layer1
        else:
            hidden = section1.hidden
            layer = layer1
            layers = [layer, layer]

        end_angle = p.end_angle
        start_angle = p.start_angle
        points = p.points
        width_value1 = width_func1(lengths)
        offset_value1 = offset_func1(lengths)
        width_value2 = width_func2(lengths)
        offset_value2 = offset_func2(lengths)

        points1 = p.centerpoint_offset_curve(
            points,
            offset_distance=offset_value1 + width_value1 / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        points2 = p.centerpoint_offset_curve(
            points,
            offset_distance=offset_value2 - width_value2 / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )

        if section1.simplify is not None and section2.simplify is not None:
            tolerance = min([section1.simplify, section2.simplify])
            points1 = _simplify(points1, tolerance=tolerance)
            points2 = _simplify(points2, tolerance=tolerance)

        points_poly = xp.concatenate([points1, points2[::-1, :]])

        if not hidden and to_float(path_length) > 1e-3:
            c.add_polygon(points_poly, layer=layer)

        offset_arr = xp.broadcast_to(asarray(offset_value1), lengths.shape)
        if port_names[0] is not None:
            center = p.centerpoint_offset_curve(
                points[:2],
                offset_distance=offset_arr[:2],
                start_angle=start_angle,
                end_angle=None,
            )[0]
            c.add_port(
                name=port_names[0],
                layer=get_layer(layers[0]),
                port_type=port_types[0],
                width=width1,
                orientation=_mod360(p.start_angle + 180),
                center=center,
                cross_section=x1,
            )
        if port_names[1] is not None:
            center = p.centerpoint_offset_curve(
                points[-2:],
                offset_distance=offset_arr[-2:],
                start_angle=None,
                end_angle=end_angle,
            )[-1]
            c.add_port(
                name=port_names[1],
                layer=get_layer(layers[1]),
                port_type=port_types[1],
                width=width2,
                center=center,
                orientation=_mod360(p.end_angle),
                cross_section=x2,
            )

    c.info["length"] = round_st(p.length_exact(), 3)
    return c


def _compute_offset_directions(points: Array) -> tuple[Array, Array, Array]:
    """Pre-compute direction vectors for centerpoint offset curves.

    Returns (cos_theta_mid, sin_theta_mid, sin_half_dtheta_int).
    """
    points = asarray(points)
    dx = xp.diff(points[:, 0])
    dy = xp.diff(points[:, 1])
    theta = xp.unwrap(xp.arctan2(dy, dx))
    theta = xp.concatenate([theta[:1], theta, theta[-1:]])
    theta_mid = (xp.pi + theta[1:] + theta[:-1]) / 2
    dtheta_int = xp.pi + theta[:-1] - theta[1:]
    sin_half = xp.sin(dtheta_int / 2)
    return xp.cos(theta_mid), xp.sin(theta_mid), sin_half


def _offset_curve_from_directions(
    points: Array,
    offset_distance: Any,
    cos_theta_mid: Array,
    sin_theta_mid: Array,
    sin_half_dtheta_int: Array,
    start_angle: Any = None,
    end_angle: Any = None,
) -> Array:
    """Single offset curve from pre-computed direction vectors."""
    offset_array = xp.broadcast_to(
        asarray(offset_distance) / sin_half_dtheta_int, sin_half_dtheta_int.shape
    )
    new_x = points[:, 0] - offset_array * cos_theta_mid
    new_y = points[:, 1] - offset_array * sin_theta_mid
    new_points = xp.stack([new_x, new_y], axis=1)

    if start_angle is not None:
        sa = start_angle * xp.pi / 180
        first = points[0, :] + xp.stack(
            [xp.sin(sa) * offset_array[0], -xp.cos(sa) * offset_array[0]]
        )
        new_points = xset(new_points, (0, slice(None)), first)

    if end_angle is not None:
        ea = end_angle * xp.pi / 180
        last = points[-1, :] + xp.stack(
            [xp.sin(ea) * offset_array[-1], -xp.cos(ea) * offset_array[-1]]
        )
        new_points = xset(new_points, (-1, slice(None)), last)

    return new_points


def _apply_offsets(
    points: Array,
    offset_distance1: Any,
    offset_distance2: Any,
    cos_theta_mid: Array,
    sin_theta_mid: Array,
    sin_half_dtheta_int: Array,
    start_angle: Any = None,
    end_angle: Any = None,
) -> tuple[Array, Array]:
    return (
        _offset_curve_from_directions(
            points,
            offset_distance1,
            cos_theta_mid,
            sin_theta_mid,
            sin_half_dtheta_int,
            start_angle=start_angle,
            end_angle=end_angle,
        ),
        _offset_curve_from_directions(
            points,
            offset_distance2,
            cos_theta_mid,
            sin_theta_mid,
            sin_half_dtheta_int,
            start_angle=start_angle,
            end_angle=end_angle,
        ),
    )


def _rotated_delta(point: Any, center: Any, orientation: AngleInDegrees) -> Array:
    """Gets the rotated distance of a point from a center."""
    ca = xp.cos(orientation * xp.pi / 180)
    sa = xp.sin(orientation * xp.pi / 180)
    rot_mat = xp.stack([xp.stack([ca, -sa]), xp.stack([sa, ca])])
    delta = asarray(point) - asarray(center)
    return delta @ rot_mat


def _cut_path_with_ray(
    start_point: Any,
    start_angle: float | None,
    end_point: Any,
    end_angle: float | None,
    path: Any,
) -> Array:
    """Cuts or extends a path given a point and angle to project (non-differentiable)."""
    import shapely.geometry as sg
    import shapely.ops

    path = to_numpy(path)
    start_point = to_numpy(start_point)
    end_point = to_numpy(end_point)
    far_distance = 10000

    path_cmp = np.copy(path)
    dp = path[0] - path[1]
    path_cmp[0] += far_distance / np.sqrt(np.sum(dp**2)) * dp
    dp = path[-1] - path[-2]
    path_cmp[-1] += far_distance / np.sqrt(np.sum(dp**2)) * dp

    intersections = [sg.Point(path[0]), sg.Point(path[-1])]
    distances: list[float] = []
    ls = sg.LineString(path_cmp)
    for i, angle, point in [(0, start_angle, start_point), (1, end_angle, end_point)]:
        if angle:
            angle_rad = np.deg2rad(to_float(angle))
            d_far = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * far_distance
            ls_ray = sg.LineString([point - d_far, point + d_far])
            intersection = ls.intersection(ls_ray)
            if not isinstance(intersection, sg.Point):
                if not isinstance(intersection, sg.MultiPoint):
                    raise ValueError(
                        f"Expected intersection to be a point, but got {intersection}"
                    )
                _, intersection = shapely.ops.nearest_points(sg.Point(point), intersection)
            intersections[i] = intersection
        else:
            intersection = intersections[i]
        distances.append(ls.project(intersection))
    points = [np.array(intersections[0].coords[0])]
    points.extend(
        np.array(point)
        for point in path[1:-1]
        if distances[0] < ls.project(sg.Point(point)) < distances[1]
    )
    points.append(np.array(intersections[1].coords[0]))
    return asarray(np.array(points))


# Floor on the angular resolution of an auto-computed bend, in degrees per point.
_MAX_DEG_PER_BEND_POINT = 5.0


def _bend_npoints_floor(angle: float) -> int:
    """Minimum points for an auto-computed bend so it stays a curve, not a chord."""
    return math.ceil(abs(angle) / _MAX_DEG_PER_BEND_POINT) + 1


def arc(
    radius: float | None = 10.0,
    angle: float = 90,
    npoints: int | None = None,
    start_angle: float = -90,
    angular_step: float | None = None,
) -> Path:
    """Returns a radial arc.

    Args:
        radius: minimum radius of curvature.
        angle: total angle of the curve.
        npoints: Number of points used per 360 degrees. Defaults to pdk.bend_points_distance.
        start_angle: initial angle of the curve for drawing, default -90 degrees.
        angular_step: If provided, determines the angular step (in degrees) between points. \
                This overrides npoints calculation.
    """
    from gdsfactory.pdk import get_active_pdk

    PDK = get_active_pdk()

    if radius is None or (not is_tracer(radius) and not radius):
        raise ValueError("arc() requires a radius argument")

    if npoints is not None and angular_step is not None:
        raise ValueError(
            "arc() requires either npoints or angular_step, not both. "
            "Use angular_step for angular discretization."
        )

    angle_f = to_float(angle)
    if angular_step is not None:
        npoints = math.ceil(abs(angle_f / to_float(angular_step))) + 1
    elif not npoints:
        npoints = int(abs(angle_f) / 360 * to_float(radius) / PDK.bend_points_distance / 2)
        npoints = max(npoints, _bend_npoints_floor(angle_f), 2)
    else:
        npoints = max(int(npoints), 2)

    t = xp.linspace(
        start_angle * xp.pi / 180, (angle + start_angle) * xp.pi / 180, npoints
    )
    x = radius * xp.cos(t)
    y = radius * (xp.sin(t) + 1)
    points = xp.stack([x, y]).T * np.sign(angle_f)

    path = Path()
    path.points = points
    path.start_angle = start_angle + 90
    path.end_angle = start_angle + angle + 90
    return path


_SQRT_HALF_PI: float = float(np.sqrt(np.pi / 2))
_SQRT_2_OVER_PI: float = float(np.sqrt(2 / np.pi))


def _scipy_fresnel(z: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    from scipy.special import fresnel

    s, c = fresnel(np.asarray(z, dtype=np.float64))
    return np.asarray(s, dtype=np.float64), np.asarray(c, dtype=np.float64)


@jax.custom_jvp
def fresnel(z: Array) -> tuple[Array, Array]:
    """Differentiable Fresnel integrals (S(z), C(z)) = scipy.special.fresnel(z)."""
    shape = jax.ShapeDtypeStruct(jnp.shape(z), jnp.float64)
    return jax.pure_callback(_scipy_fresnel, (shape, shape), z, vmap_method="expand_dims")


@fresnel.defjvp
def _fresnel_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Any, Any]:
    (z,) = primals
    (dz,) = tangents
    s, c = fresnel(z)
    arg = xp.pi * z**2 / 2
    return (s, c), (xp.sin(arg) * dz, xp.cos(arg) * dz)


def _fresnel_xy(t: Array) -> Array:
    """[x, y] of the normalised clothoid for parameter values t."""
    z = t * _SQRT_2_OVER_PI
    if isinstance(z, jax.Array) or is_tracer(z):
        sin_fresnel, cos_fresnel = fresnel(z)
    else:
        sin_fresnel, cos_fresnel = _scipy_fresnel(z)
    return xp.stack([cos_fresnel * _SQRT_HALF_PI, sin_fresnel * _SQRT_HALF_PI])


def _fresnel(R0: float, s: Any, num_pts: int, n_iter: int = 8) -> Array:
    """Clothoid points with uniform arc-length sampling."""
    t = xp.linspace(0, s / (np.sqrt(2) * R0), num_pts)
    return np.sqrt(2) * R0 * _fresnel_xy(t)


def _fresnel_angular(R0: float, s: Any, num_pts: int, n_iter: int = 8) -> Array:
    """Clothoid points with uniform angular sampling."""
    t_max = s / (np.sqrt(2) * R0)
    theta_max = t_max**2 / 2
    thetas = xp.linspace(0, theta_max, num_pts)
    t = xp.sqrt(2 * thetas)
    return np.sqrt(2) * R0 * _fresnel_xy(t)


def euler(
    radius: float = 10,
    angle: float = 90,
    p: float = 0.5,
    use_eff: bool = False,
    npoints: int | None = None,
    angular_step: float | None = None,
) -> Path:
    """Returns an euler bend that adiabatically transitions from straight to curved.

    `radius` is the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc).
    If p < 1.0, will create a "partial euler" curve as described in Vogelbacher et. al.
    https://dx.doi.org/10.1364/oe.27.031394

    Args:
        radius: minimum radius of curvature.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        use_eff: If False: `radius` is the minimum radius of curvature of the bend. \
                If True: The curve will be scaled such that the endpoints match an \
                arc with parameters `radius` and `angle`.
        npoints: Number of points used per 360 degrees.
        angular_step: If provided, determines the angular step (in degrees) between points. \
                This overrides npoints calculation.
    """
    from gdsfactory.pdk import get_active_pdk

    if angular_step is not None and npoints is not None:
        raise ValueError(
            "euler() requires either npoints or angular_step, not both. "
            "Use angular_step for angular discretization."
        )

    if radius is None or (not is_tracer(radius) and not radius):
        raise ValueError("euler() requires a radius argument")

    p_f = to_float(p)
    if (p_f < 0) or (p_f > 1):
        raise ValueError(f"euler requires argument `p` be between 0 and 1. Got {p}")
    if p_f == 0:
        path = arc(radius, angle, npoints=npoints, angular_step=angular_step)
        path.info["Reff"] = radius
        path.info["Rmin"] = radius
        return path

    angle_f = to_float(angle)
    if angle_f < 0:
        mirror = True
        angle = -angle
        angle_f = -angle_f
    else:
        mirror = False

    R0 = 1
    alpha = xp.radians(asarray(angle))
    sp = R0 * xp.sqrt(p * alpha)
    is_small_angle = abs(angle_f) <= 1e-6
    Rp = R0 / xp.sqrt(p * alpha) if not is_small_angle else asarray(np.inf)

    pdk = get_active_pdk()
    if angular_step is not None:
        step = to_float(angular_step)
        euler_angle = p_f * angle_f / 2
        arc_angle = (1 - p_f) * angle_f / 2
        num_pts_euler = max(2, math.ceil(euler_angle / step))
        num_pts_arc = max(2, math.ceil(arc_angle / step) + 1)
        npoints = 2 * num_pts_euler + num_pts_arc - 2
    else:
        if not npoints:
            npoints = abs(int(angle_f / 360 * to_float(radius) / pdk.bend_points_distance / 2))
            npoints = max(npoints, _bend_npoints_floor(angle_f), 2)
        else:
            npoints = max(int(npoints), 2)
        num_pts_euler = int(np.round(2 * p_f / (p_f + 1) * npoints))
        num_pts_arc = npoints - num_pts_euler

    if npoints <= 2:
        num_pts_euler = 0
        num_pts_arc = 2

    if num_pts_euler > 0:
        if angular_step is not None:
            xbend1, ybend1 = _fresnel_angular(R0, sp, num_pts_euler)
        else:
            xbend1, ybend1 = _fresnel(R0, sp, num_pts_euler)
        x_p, y_p = xbend1[-1], ybend1[-1]
        sinc_quarter = xp.sinc(p * alpha / (4 * xp.pi))
        dx = x_p - sp / 2 * sinc_quarter * xp.cos(p * alpha / 4)
        dy = y_p - (sp**3 / 8) * sinc_quarter**2
    else:
        xbend1 = ybend1 = xp.zeros((0,))
        dx = 0.0
        dy = 0.0

    if not is_small_angle:
        if angular_step is not None:
            arc_angle_section = alpha * (1 - p) / 2
            theta = xp.linspace(0, arc_angle_section, num_pts_arc)
            arc_angles = theta + p * alpha / 2
        else:
            arc_angles = xp.linspace(p * alpha / 2, alpha / 2, num_pts_arc)
        xbend2 = Rp * xp.sin(arc_angles) + dx
        ybend2 = Rp * (1 - xp.cos(arc_angles)) + dy
    else:
        xbend2 = xp.zeros(num_pts_arc) + dx
        ybend2 = xp.zeros(num_pts_arc) + dy

    x = xp.concatenate([xbend1, xbend2[1:]])
    y = xp.concatenate([ybend1, ybend2[1:]])
    points1 = xp.stack([x, y]).T
    points2 = xp.flipud(xp.stack([x, -y]).T)

    points2 = rotate_points(points2, angle - 180)
    points2 = points2 - points2[0, :] + points1[-1, :]

    points = xp.concatenate([points1[: -1 if len(points1) > 1 else None], points2])

    start_angle = 0.0
    end_angle = start_angle + angle

    if is_small_angle:
        Reff: Any = np.inf
        Rmin: Any = np.inf
        scale: Any = 0.0
    else:
        dyy = xp.tan(xp.radians(end_angle - 90)) * points[-1][0]
        Reff = points[-1][1] - dyy
        Rmin = Rp
        if abs(180 - angle_f) < 1e-3:
            Reff = points[-1][1] / 2
        scale = radius / Reff if use_eff else radius / Rmin

    points = points * scale

    path = Path()
    path.points = points
    path.start_angle = start_angle
    path.end_angle = end_angle
    path.info["Reff"] = Reff * scale if not is_small_angle else np.inf
    path.info["Rmin"] = Rmin * scale if not is_small_angle else np.inf
    if mirror:
        path.mirror((1, 0))
    return path


def _find_root_in_range(
    equation: Callable[[float], float],
    variable_range: tuple[float, float],
) -> tuple[float, float]:
    """Find a root of `equation` within the given range using Brent's method."""
    from scipy import optimize

    var_lo, var_hi = variable_range
    try:
        root = float(optimize.brentq(equation, var_lo, var_hi, xtol=1e-9, maxiter=500))
    except ValueError:
        x_vals = np.linspace(var_lo, var_hi, 1000)
        residuals = [equation(x) for x in x_vals]
        sign_changes = np.where(np.diff(np.sign(residuals)))[0]
        if len(sign_changes) == 0:
            raise RuntimeError(
                f"No root found in [{var_lo}, {var_hi}]. "
                "Verify that the constraint changes sign within the bracket."
            )
        root = float(
            optimize.brentq(
                equation, x_vals[sign_changes[0]], x_vals[sign_changes[0] + 1]
            )
        )
    return root, float(equation(root))


def _topic_theta_of_s(s: float, Rc: float, theta_p: float) -> float:
    """Accumulated angle along the TOP spiral section."""
    return float((4 * Rc * theta_p * s**3 - s**4) / (16 * Rc**4 * theta_p**3))


def _topic_compute_x0_y0(Rc: float, theta_p: float) -> tuple[float, float]:
    from scipy import integrate

    s_max = 2 * Rc * theta_p
    x0_int, _ = integrate.quad(
        lambda s: np.cos(_topic_theta_of_s(s, Rc, theta_p)), 0, s_max
    )
    y0_int, _ = integrate.quad(
        lambda s: np.sin(_topic_theta_of_s(s, Rc, theta_p)), 0, s_max
    )
    x0 = x0_int - Rc * np.sin(theta_p)
    y0 = y0_int + Rc * np.cos(theta_p)
    return x0, y0


def _topic_compute_top_coordinates(
    Rc: float, theta_p: float, n_points: int = 300
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    from scipy import integrate

    l_max = 2 * Rc * theta_p
    l_vals = np.linspace(0, l_max, n_points)
    theta_vals = np.array([_topic_theta_of_s(s, Rc, theta_p) for s in l_vals])
    x_top = integrate.cumulative_trapezoid(np.cos(theta_vals), l_vals, initial=0)
    y_top = integrate.cumulative_trapezoid(np.sin(theta_vals), l_vals, initial=0)
    return x_top, y_top


def topic(
    radius: float = 10.0, angle: float = 90.0, p: float = 0.1, npoints: int = 100
) -> Path:
    """Returns a Third Order Polynomial Interconnected Circular (TOPIC) bend.

    See https://arxiv.org/html/2411.15025v1.

    The geometry is computed for a unit radius and scaled, so it is exactly
    differentiable with respect to ``radius``; ``angle`` and ``p`` are
    treated as constants (the inner root solve is not differentiated).

    Args:
        radius: radius at the start and end of bend.
        angle: total angle of the curve in degrees.
        p: transition fraction in [0, 0.5).
        npoints: Number of points used per 360 degrees.
    """
    p_f = to_float(p)
    angle_f = to_float(angle)
    if p_f < 0.0 or p_f >= 0.5:
        raise ValueError(
            "The angle of bend during the transition from the TOP segment to the circular is p*angle . "
            "topic() requires the transition angle to be between 0 (circular bend) and 0.5*angle . "
        )
    if abs(angle_f) <= 1e-6:
        raise ValueError("The bend's total angle should be larger than 1e-6.")
    if p_f < 1e-4:
        topic_path = arc(radius=radius, angle=angle, npoints=npoints)
        topic_path.end_angle = angle
        topic_path.info["Rmin"] = radius
        return topic_path

    unit = 1.0
    theta_t = np.radians(angle_f)
    theta_p = p_f * theta_t

    def constraint(Rc: float) -> float:
        if Rc <= 0:
            return 1e9
        x0, y0 = _topic_compute_x0_y0(Rc, theta_p)
        return float(x0 * np.cos(theta_t / 2) + (y0 - unit) * np.sin(theta_t / 2))

    Rc, _ = _find_root_in_range(constraint, (1e-3 * unit, unit))
    x0, y0 = _topic_compute_x0_y0(Rc, theta_p)

    n_points_circ = int(npoints * (Rc * (theta_t - 2 * theta_p)) / (unit * theta_t))
    n_points_top = max(2, (npoints - n_points_circ) // 2)
    if 2 * n_points_top + n_points_circ == npoints - 1:
        n_points_circ += 1

    x_top, y_top = _topic_compute_top_coordinates(Rc, theta_p, n_points=n_points_top)
    theta_list = np.linspace(theta_p, theta_t - theta_p, n_points_circ + 2)
    x_arc = np.array([x0 + Rc * np.sin(theta) for theta in theta_list[1:-1]])
    y_arc = np.array([y0 - Rc * np.cos(theta) for theta in theta_list[1:-1]])

    dist = np.sqrt(x_top**2 + (unit - y_top) ** 2)
    thetas = np.arcsin(np.clip(x_top / dist, -1, 1))
    x_top_prime = (dist * np.sin(theta_t - thetas))[::-1]
    y_top_prime = (unit - dist * np.cos(theta_t - thetas))[::-1]

    x_all = np.concatenate([x_top, x_arc, x_top_prime])
    y_all = np.concatenate([y_top, y_arc, y_top_prime])
    points = np.column_stack((x_all, y_all))

    topic_path = Path()
    topic_path.points = asarray(points) * radius
    topic_path.end_angle = angle
    topic_path.info["Rmin"] = Rc * radius
    return topic_path


def straight(length: float = 10.0, npoints: int = 2) -> Path:
    """Returns a straight path.

    For transitions you should increase have at least 100 points

    Args:
        length: of straight.
        npoints: number of points.
    """
    if to_float(length) < 0:
        raise ValueError(f"length = {length} needs to be > 0")
    x = xp.linspace(0, length, npoints)
    y = x * 0
    points = xp.stack([x, y]).T

    p = Path()
    p.append(points)
    return p


def spiral_archimedean(
    min_bend_radius: float, separation: float, number_of_loops: float, npoints: int
) -> Path:
    """Returns an Archimedean spiral.

    Args:
        min_bend_radius: Inner radius of the spiral.
        separation: Half the radial separation between loops in um.
        number_of_loops: number of loops.
        npoints: number of Points.
    """
    theta = xp.linspace(0, number_of_loops * 2 * xp.pi, int(npoints))
    points = (separation / xp.pi * theta + min_bend_radius)[:, None] * xp.stack(
        (xp.sin(theta), xp.cos(theta)), axis=1
    )
    return Path(points)


def _compute_segments(points: Any) -> tuple[Array, Array, Array, Array, Array]:
    points = asarray(points)
    normals = xp.diff(points, axis=0)
    norms = xp.linalg.norm(normals, axis=1)

    tol = 1e-6
    if np.any(to_numpy(norms) < tol):
        warnings.warn(
            "Zero-length segments (duplicate consecutive points)",
            RuntimeWarning,
            stacklevel=3,
        )

    normals = (normals.T / norms).T
    dx = xp.diff(points[:, 0])
    dy = xp.diff(points[:, 1])
    ds = xp.sqrt(dx**2 + dy**2)
    theta = xp.degrees(xp.arctan2(dy, dx))
    dtheta = xp.diff(theta)
    dtheta = dtheta - 360 * xp.floor(stop_gradient((dtheta + 180) / 360))
    return points, normals, ds, theta, dtheta


def smooth(
    points: Any,
    radius: float = 4.0,
    bend: PathFactory = euler,
    **kwargs: Any,
) -> Path:
    """Returns a smooth Path from a series of waypoints.

    Args:
        points: array-like[N][2] List of waypoints for the path to follow.
        radius: radius of curvature, passed to `bend`.
        bend: bend function that returns a path that round corners.
        kwargs: Extra keyword arguments that will be passed to `bend`.
    """
    if isinstance(points, Path):
        points = points.points

    points, normals, ds, theta, dtheta = _compute_segments(points)
    dtheta_np = to_numpy(dtheta)
    colinear = np.concatenate([[False], np.abs(dtheta_np) < 1e-6, [False]])
    if np.any(colinear):
        points, normals, ds, theta, dtheta = _compute_segments(
            points[np.nonzero(~colinear)[0], :]
        )
        dtheta_np = to_numpy(dtheta)

    if np.any(np.abs(np.abs(dtheta_np) - 180) < 1e-6):
        raise ValueError(
            "smooth() received points which double-back on themselves"
            "--turns cannot be computed when going forwards then exactly backwards."
        )

    paths: list[Path] = []
    radii: list[Any] = []
    for i in range(len(dtheta_np)):
        dt = dtheta[i]
        P = bend(radius=radius, angle=dt, **kwargs)
        chord = xp.linalg.norm(P.points[-1, :] - P.points[0, :])
        r = xp.abs((chord / 2) / xp.sin(xp.radians(dt / 2)))
        radii.append(r)
        paths.append(P)

    if radii:
        d = xp.abs(xp.stack(radii) / xp.tan(xp.radians(180 - dtheta) / 2))
    else:
        d = xp.zeros((0,))
    encroachment = xp.concatenate([xp.zeros(1), d]) + xp.concatenate([d, xp.zeros(1)])
    if np.any(to_numpy(encroachment) > to_numpy(ds) + 1e-12):
        raise ValueError(
            "smooth(): Not enough distance between points to to fit curves."
            "Try reducing the radius or spacing the points out farther"
        )
    p1 = points[1:-1, :] - normals[:-1, :] * d[:, None]

    new_points: list[Array] = [points[0:1, :]]
    for n in range(len(dtheta_np)):
        p = paths[n]
        p.rotate(theta[n])
        p.move(p1[n])
        new_points.append(p.points)
    new_points.append(points[-1:, :])
    new_points_np = xp.concatenate(new_points)

    path = Path()
    path.append(new_points_np)
    path.rotate(theta[0])
    path.move(points[0, :])
    path.start_angle = theta[0]
    path.end_angle = theta[-1]
    return path


__all__ = [
    "Path",
    "along_path",
    "arc",
    "euler",
    "extrude",
    "extrude_transition",
    "fresnel",
    "smooth",
    "spiral_archimedean",
    "straight",
    "topic",
    "transition",
    "transition_adiabatic",
]

_ = (Sequence, overload, Literal)
