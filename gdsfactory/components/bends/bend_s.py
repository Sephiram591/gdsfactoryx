from __future__ import annotations

__all__ = ["bend_s", "bend_s_offset", "bezier"]

import math
import warnings
from typing import Any

import jax
import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory._jax import (
    _jax_float_dtype,
    has_abstract_tracers,
    has_tracers,
    is_abstract_tracer,
    jnp,
    maybe_float,
    stop_gradient,
    to_float,
    xp,
)
from gdsfactory.component import Component
from gdsfactory.config import ErrorType
from gdsfactory.functions import angles_deg, curvature, snap_angle
from gdsfactory.typings import Coordinate, Coordinates, CrossSectionSpec, Size

from .._schematic import sbend_schematic


def bezier_curve(
    t: npt.NDArray[np.floating[Any]], control_points: Coordinates
) -> npt.NDArray[np.floating[Any]]:
    """Returns bezier coordinates.

    Args:
        t: 1D array of points varying between 0 and 1.
        control_points: for the bezier curve.
    """
    from scipy.special import binom

    xs = 0.0
    ys = 0.0
    n = len(control_points) - 1
    for k in range(n + 1):
        ank = binom(n, k) * (1 - t) ** (n - k) * t**k
        xs = xs + ank * control_points[k][0]
        ys = ys + ank * control_points[k][1]

    return xp.column_stack([xs, ys])  # type: ignore[no-any-return]


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bezier(
    control_points: Coordinates = ((0.0, 0.0), (5.0, 0.0), (5.0, 1.8), (10.0, 1.8)),
    npoints: int = 201,
    with_manhattan_facing_angles: bool = True,
    start_angle: int | None = None,
    end_angle: int | None = None,
    cross_section: CrossSectionSpec = "strip",
    bend_radius_error_type: ErrorType | None = None,
    allow_min_radius_violation: bool = False,
    width: float | None = None,
) -> Component:
    """Returns Bezier bend.

    Args:
        control_points: list of points.
        npoints: number of points varying between 0 and 1.
        with_manhattan_facing_angles: bool.
        start_angle: optional start angle in deg.
        end_angle: optional end angle in deg.
        cross_section: spec.
        bend_radius_error_type: error type.
        allow_min_radius_violation: bool.
        width: width to use. Defaults to cross_section.width.
    """
    if width is not None:
        xs = gf.get_cross_section(cross_section, width=width)
    else:
        xs = gf.get_cross_section(cross_section)

    t = np.linspace(0, 1, npoints)
    path_points = bezier_curve(t, control_points)
    path = gf.Path(path_points)

    if with_manhattan_facing_angles:
        path.start_angle = start_angle or snap_angle(path.start_angle)
        path.end_angle = end_angle or snap_angle(path.end_angle)

    c = path.extrude(xs)
    curv = curvature(path_points, t)
    length = path.length()
    if to_float(xp.max(xp.abs(curv))) == 0:
        min_bend_radius = np.inf
    else:
        min_bend_radius = maybe_float(gf.snap.snap_to_grid(1 / xp.max(xp.abs(curv))))

    c.info["length"] = maybe_float(length)
    c.info["min_bend_radius"] = min_bend_radius
    c.info["start_angle"] = maybe_float(path.start_angle)
    c.info["end_angle"] = maybe_float(path.end_angle)
    c.add_route_info(
        cross_section=xs,
        length=c.info["length"],
        n_bend_s=1,
        min_bend_radius=min_bend_radius,
    )

    if not allow_min_radius_violation:
        xs.validate_radius(min_bend_radius, bend_radius_error_type)

    xs.add_bbox(c)
    return c


def find_min_curv_bezier_control_points(
    start_point: Coordinate,
    end_point: Coordinate,
    start_angle: float,
    end_angle: float,
    npoints: int = 201,
    alpha: float = 0.05,
    nb_pts: int = 2,
) -> Coordinates:
    """Returns bezier control points that minimize curvature.

    Args:
        start_point: start point.
        end_point: end point.
        start_angle: start angle in deg.
        end_angle: end angle in deg.
        npoints: number of points varying between 0 and 1.
        alpha: weight for angle mismatch.
        nb_pts: number of control points.
    """
    from scipy.optimize import minimize

    # the optimizer runs on concrete values (the optimum is not differentiated)
    start_point_f = (to_float(start_point[0]), to_float(start_point[1]))
    end_point_f = (to_float(end_point[0]), to_float(end_point[1]))
    start_angle = to_float(start_angle)
    end_angle = to_float(end_angle)
    alpha = to_float(alpha)
    t = np.linspace(0, 1, npoints)

    def array_1d_to_cpts(a: npt.NDArray[np.float64]) -> list[tuple[float, float]]:
        xs = a[::2]
        ys = a[1::2]
        return list(zip(xs, ys, strict=False))

    def objective_func(p: npt.NDArray[np.float64]) -> float:
        """Minimize  max curvaturea and negligible start angle and end angle mismatch."""
        ps = array_1d_to_cpts(p)
        control_points = [start_point_f, *ps, end_point_f]
        path_points = bezier_curve(t, control_points)

        max_curv = np.max(np.abs(np.asarray(curvature(path_points, t))))

        angles = np.asarray(angles_deg(path_points))
        dstart_angle = abs(angles[0] - start_angle)
        dend_angle = abs(angles[-2] - end_angle)
        angle_mismatch = dstart_angle + dend_angle
        return float(angle_mismatch * alpha + max_curv)

    x0, y0 = start_point_f
    xn, yn = end_point_f

    initial_guess: list[float] = []
    for i in range(nb_pts):
        x = (i + 1) * (x0 + xn) / nb_pts
        y = (i + 1) * (y0 + yn) / nb_pts
        initial_guess += [x, y]

    # initial_guess = [(x0 + xn) / 2, y0, (x0 + xn) / 2, yn]
    res = minimize(objective_func, initial_guess, method="Nelder-Mead")
    p = res.x
    points = [start_point] + array_1d_to_cpts(p) + [end_point]
    return tuple(points)


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_s(
    size: Size = (11.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    width: float | None = None,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        allow_min_radius_violation: bool.
        width: width to use. Defaults to cross_section.width.

    """
    dx, dy = size

    if not is_abstract_tracer(dy) and to_float(dy) == 0:
        return gf.components.straight(
            length=dx, cross_section=cross_section, width=width
        )

    return bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


def _get_euler_sbend_angle_middle_length_from_jog(
    jog: float, radius: float, p: float = 1, use_eff: bool = False
) -> tuple[float, float]:
    """Compute the Euler bend angle (in degrees) and middle straight length for an S-bend.

    using SciPy to numerically solve for the bend angle required to achieve half the jog.

    The vertical displacement for an Euler bend of angle θ (in radians) is given by:
      displacement(θ) = radius * sqrt(pi * θ) * S( sqrt(2θ/pi) )
    where S() is the Fresnel sine integral.

    The S-bend consists of two symmetric Euler bends. If the jog is less than twice the displacement
    of a full 90° Euler bend, the angle is computed such that one Euler bend gives a displacement of jog/2.
    Otherwise, a full 90° Euler bend is used and the extra required offset is added as a straight section.

    Args:
        jog: The vertical displacement of the S-bend.
        radius: The radius of the Euler bend.
        p: proportion of the curve that is an Euler curve.
        use_eff: if True, use effective radius.

    Returns:
      tuple: (angle_deg, middle_length) where:
          - angle_deg is the Euler bend angle in degrees.
          - middle_length is the length of the straight segment between the Euler bends.
    """
    from scipy import optimize

    def euler_displacement(theta: Any, r: Any = radius, p_val: Any = p) -> Any:
        curve = gf.path.euler(radius=r, angle=theta, use_eff=use_eff, p=p_val)
        return curve.ysize

    def _solve_sub_jog(j_val: Any, r_val: Any, p_val: Any) -> Any:
        def _solve_concrete(j_v: Any, r_v: Any, p_v: Any) -> float:
            res = optimize.minimize_scalar(
                lambda theta: float(
                    (to_float(euler_displacement(theta, r_v, p_v)) - float(j_v)) ** 2
                ),
                bounds=(1.0, 90.0),
                method="bounded",
            )
            return float(res.x)

        if not has_tracers((j_val, r_val, p_val)):
            return _solve_concrete(to_float(j_val), to_float(r_val), to_float(p_val))

        float_dtypes = [
            jnp.asarray(v).dtype
            for v in (j_val, r_val, p_val)
            if isinstance(v, jax.Array | np.ndarray)
            and np.issubdtype(jnp.asarray(v).dtype, np.floating)
        ]
        out_dtype = jnp.result_type(*float_dtypes) if float_dtypes else _jax_float_dtype()

        @jax.custom_jvp
        def _solve(j_in: Any, r_in: Any, p_in: Any) -> Any:
            def _cb(j_cb: Any, r_cb: Any, p_cb: Any) -> np.ndarray:
                return np.asarray(
                    _solve_concrete(float(j_cb), float(r_cb), float(p_cb)),
                    dtype=np.dtype(out_dtype),
                )

            return jax.pure_callback(
                _cb,
                jax.ShapeDtypeStruct((), out_dtype),
                j_in,
                r_in,
                p_in,
                vmap_method="sequential",
            )

        @_solve.defjvp
        def _solve_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
            j_p, r_p, p_p = primals
            j_t, r_t, p_t = tangents
            theta_star = _solve(j_p, r_p, p_p)

            def _res_of_params(j_in: Any, r_in: Any, p_in: Any) -> Any:
                return jnp.asarray(
                    euler_displacement(theta_star, r_in, p_in) - j_in, dtype=out_dtype
                )

            def _res_of_theta(th_in: Any) -> Any:
                return jnp.asarray(
                    euler_displacement(th_in, r_p, p_p) - j_p, dtype=out_dtype
                )

            _, df_dp_dot = jax.jvp(_res_of_params, (j_p, r_p, p_p), (j_t, r_t, p_t))
            _, df_dtheta = jax.jvp(
                _res_of_theta, (theta_star,), (jnp.ones_like(theta_star),)
            )
            nonzero = jnp.abs(df_dtheta) > 1e-30
            safe_denom = jnp.where(nonzero, df_dtheta, jnp.ones_like(df_dtheta))
            theta_dot = jnp.where(
                nonzero, -df_dp_dot / safe_denom, jnp.zeros_like(theta_star)
            )
            return theta_star, jnp.asarray(theta_dot, dtype=out_dtype)

        return _solve(
            jnp.asarray(j_val, dtype=out_dtype),
            jnp.asarray(r_val, dtype=out_dtype),
            jnp.asarray(p_val, dtype=out_dtype),
        )

    dy_full = euler_displacement(theta=90)

    if has_abstract_tracers((jog, dy_full)):
        angle_sub = _solve_sub_jog(jog, radius, p)
        angle_deg = xp.where(jog <= dy_full, angle_sub, 90.0)
        middle_length = xp.where(jog <= dy_full, 0.0, 2 * jog - 2 * dy_full)
    elif to_float(jog) <= to_float(dy_full):
        angle_deg = _solve_sub_jog(jog, radius, p)
        middle_length = 0.0
    else:
        angle_deg = 90.0
        middle_length = 2 * jog - 2 * dy_full

    return angle_deg, middle_length


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_s_offset(
    offset: float = 40.0,
    radius: float | None = 10.0,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    with_euler: bool | None = None,
    p: float = 1,
    with_arc_floorplan: bool = False,
    npoints: int | None = None,
    angular_step: float | None = None,
) -> gf.Component:
    """Return S bend made of two bends with a straight section.

    Args:
        offset: in um.
        radius: in um. if None, uses cross_section_radius.
        cross_section: spec.
        width: width to use. Defaults to cross_section.width.
        with_euler: deprecated, use p=0 for circular arc instead.
        p: 1 means standard Euler bend. 0 means circular arc.
        with_arc_floorplan: if True the size of the bend will be adjusted to match an arc bend with the specified radius. If False: `radius` is the minimum radius of curvature.
        npoints: number of points.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
    """
    if with_euler is not None:
        warnings.warn(
            "with_euler is deprecated. Use p=0 for circular arc instead. And p=1 for euler bend.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not with_euler:
            p = 0

    if width is not None:
        xs = gf.get_cross_section(cross_section, width=width)
    else:
        xs = gf.get_cross_section(cross_section)

    radius = radius if radius is not None else xs.radius
    assert radius is not None, "radius cannot be None"

    xs.validate_radius(radius)
    angle, middle_length = _get_euler_sbend_angle_middle_length_from_jog(
        jog=abs(offset) / 2, radius=radius, p=p, use_eff=with_arc_floorplan
    )
    angle = (
        xp.where(offset >= 0, angle, -angle)
        if is_abstract_tracer(offset)
        else (angle if math.copysign(1, to_float(offset)) > 0 else -angle)
    )
    path = gf.path.euler(
        radius=radius,
        angle=+angle,
        p=p,
        use_eff=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
    )
    if is_abstract_tracer(middle_length):
        path += gf.path.straight(
            length=xp.where(middle_length > 1e-6, middle_length, 1e-6)
        )
    elif to_float(middle_length) > 1e-6:
        path += gf.path.straight(length=middle_length)
    path += gf.path.euler(
        radius=radius,
        angle=-angle,
        p=p,
        use_eff=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
    )

    return gf.path.extrude(path, cross_section=xs)


def get_min_sbend_size(
    size: tuple[float | None, float | None] = (None, 10.0),
    cross_section: CrossSectionSpec = "strip",
    num_points: int = 100,
) -> float:
    """Returns the minimum sbend size to comply with bend radius requirements.

    Args:
        size: in x and y direction. One of them is None, which is the size we need to figure out.
        cross_section: spec.
        num_points: number of points to iterate over between max_size and 0.1 * max_size.
    """
    size_list = list(size)
    cross_section_f = gf.get_cross_section(cross_section)

    if size_list[0] is None:
        ind = 0
        known_s = size_list[1]
    elif size_list[1] is None:
        ind = 1
        known_s = size_list[0]
    else:
        raise ValueError("One of the two elements in size has to be None")

    min_radius = cross_section_f.radius

    if min_radius is None:
        raise ValueError("The min radius for the specified layer is not known!")

    assert known_s is not None

    # Guess sizes, iterate over them until we cannot achieve the min radius
    # the max size corresponds to an ellipsoid
    # the sweep is discrete: run it on concrete values
    def _find_k_concrete(known_f: float, min_r_f: float) -> int:
        sz = [known_f, known_f]
        max_s = float(np.sqrt(np.abs(min_r_f * known_f)) * 2.5)
        sizes = np.linspace(max_s, 0.1 * max_s, num_points)
        t = np.linspace(0, 1, 201)
        for idx, s in enumerate(sizes):
            sz[ind] = float(s)
            dx, dy = sz
            control_points = ((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy))
            path_points = bezier_curve(t, control_points)
            curv = curvature(path_points, t)
            min_bend_radius = 1.0 / max(np.abs(curv))
            if min_bend_radius < min_r_f:
                return idx
        return -1

    if has_abstract_tracers((known_s, min_radius)):
        k = jax.pure_callback(
            lambda kv, rv: np.asarray(
                _find_k_concrete(float(kv), float(rv)), dtype=np.int32
            ),
            jax.ShapeDtypeStruct((), jnp.int32),
            jnp.asarray(known_s),
            jnp.asarray(min_radius),
            vmap_method="sequential",
        )
        traced = (
            xp.sqrt(xp.abs(min_radius * known_s))
            * 2.5
            * (1.0 - 0.9 * k / (num_points - 1))
        )
        return xp.where(k >= 0, traced, jnp.inf)  # type: ignore[no-any-return]

    known_f = to_float(known_s)
    min_r_f = to_float(min_radius)
    k = _find_k_concrete(known_f, min_r_f)
    if k < 0:
        return np.inf

    max_size = float(np.sqrt(np.abs(min_r_f * known_f)) * 2.5)
    min_size = float(np.linspace(max_size, 0.1 * max_size, num_points)[k])
    if has_tracers((known_s, min_radius)):
        traced = xp.sqrt(xp.abs(min_radius * known_s)) * 2.5
        traced = traced * (1 - 0.9 * k / (num_points - 1))
        return min_size + (traced - stop_gradient(traced))  # type: ignore[no-any-return]
    return min_size
