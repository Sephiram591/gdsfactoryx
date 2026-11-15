from __future__ import annotations

__all__ = ["optimal_step"]

from typing import Any

import jax
import numpy as np

import gdsfactory as gf
from gdsfactory._jax import (
    _jax_float_dtype,
    has_abstract_tracers,
    has_tracers,
    jnp,
    to_float,
    to_numpy,
    xp,
)
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["superconductors"])
def optimal_step(
    start_width: float = 10,
    end_width: float = 22,
    num_pts: int = 50,
    width_tol: float = 1e-3,
    anticrowding_factor: float = 1.2,
    symmetric: bool = False,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns an optimally-rounded step geometry.

    Args:
        start_width: Width of the connector on the left end of the step.
        end_width: Width of the connector on the right end of the step.
        num_pts: number of points comprising the entire step geometry.
        width_tol: Point at which to terminate the calculation of the optimal step
        anticrowding_factor: Factor to reduce current crowding by elongating
            the structure and reducing the curvature
        symmetric: If True, adds a mirrored copy of the step across the x-axis to the
            geometry and adjusts the width of the ports.
        layer: layer spec to put polygon geometry on.

    based on phidl.geometry
    Optimal structure from <https://doi.org/10.1103/PhysRevB.84.174510>
    Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
    superconducting nanocircuits. Physical Review B, 84(17), 1-27.
    """

    def step_points(eta: Any, W: Any, a: Any) -> tuple[Any, Any]:
        """Returns step points.

        Returns points from a unit semicircle in the w (= u + iv) plane to
        the optimal curve in the zeta (= x + iy) plane which transitions
        a wire from a width of 'W' to a width of 'a'
        eta takes value 0 to pi
        """
        gamma = (a**2 + W**2) / (a**2 - W**2)
        w = xp.exp(1j * eta)
        zeta = (
            4
            * 1j
            / np.pi
            * (
                W * xp.arctan(xp.sqrt((w - gamma) / (gamma + 1)))
                + a * xp.arctan(xp.sqrt((gamma - 1) / (w - gamma)))
            )
        )
        return xp.real(zeta), xp.imag(zeta)

    def invert_step_point(
        x_desired: Any = -10,
        y_desired: Any | None = None,
        W: Any = 1,
        a: Any = 2,
    ) -> tuple[Any, Any]:
        """Finds the eta associated with x_desired or y_desired along the optimal curve."""
        from scipy.optimize import fminbound

        idx = 0 if y_desired is None else 1
        target = y_desired if y_desired is not None else x_desired

        def _solve_concrete(t_v: float, w_v: float, a_v: float) -> float:
            def fh(eta: float) -> float:
                pt = step_points(eta, W=w_v + 0j, a=a_v + 0j)
                return float((to_float(pt[idx]) - t_v) ** 2)

            return float(fminbound(fh, 0.0, np.pi))

        if not has_tracers((target, W, a)):
            found_eta = _solve_concrete(to_float(target), to_float(W), to_float(a))
            return step_points(found_eta, W=W + 0j, a=a + 0j)

        # Differentiable eta via the implicit function theorem inside XLA
        float_dtypes = [
            jnp.asarray(v).dtype
            for v in (target, W, a)
            if isinstance(v, jax.Array | np.ndarray)
            and np.issubdtype(jnp.asarray(v).dtype, np.floating)
        ]
        out_dtype = jnp.result_type(*float_dtypes) if float_dtypes else _jax_float_dtype()

        @jax.custom_jvp
        def _solve(t_in: Any, w_in: Any, a_in: Any) -> Any:
            def _cb(t_cb: Any, w_cb: Any, a_cb: Any) -> np.ndarray:
                return np.asarray(
                    _solve_concrete(float(t_cb), float(w_cb), float(a_cb)),
                    dtype=np.dtype(out_dtype),
                )

            return jax.pure_callback(
                _cb,
                jax.ShapeDtypeStruct((), out_dtype),
                t_in,
                w_in,
                a_in,
                vmap_method="sequential",
            )

        @_solve.defjvp
        def _solve_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
            t_p, w_p, a_p = primals
            t_t, w_t, a_t = tangents
            eta_star = _solve(t_p, w_p, a_p)

            def _res_of_params(t_in: Any, w_in: Any, a_in: Any) -> Any:
                return jnp.asarray(
                    step_points(eta_star, W=w_in + 0j, a=a_in + 0j)[idx] - t_in,
                    dtype=out_dtype,
                )

            def _res_of_eta(e_in: Any) -> Any:
                return jnp.asarray(
                    step_points(e_in, W=w_p + 0j, a=a_p + 0j)[idx] - t_p,
                    dtype=out_dtype,
                )

            _, df_dp_dot = jax.jvp(_res_of_params, (t_p, w_p, a_p), (t_t, w_t, a_t))
            _, df_deta = jax.jvp(_res_of_eta, (eta_star,), (jnp.ones_like(eta_star),))
            nonzero = jnp.abs(df_deta) > 1e-30
            safe_denom = jnp.where(nonzero, df_deta, jnp.ones_like(df_deta))
            eta_dot = jnp.where(
                nonzero, -df_dp_dot / safe_denom, jnp.zeros_like(eta_star)
            )
            return eta_star, jnp.asarray(eta_dot, dtype=out_dtype)

        eta = _solve(
            jnp.asarray(target, dtype=out_dtype),
            jnp.asarray(W, dtype=out_dtype),
            jnp.asarray(a, dtype=out_dtype),
        )
        return step_points(eta, W=W + 0j, a=a + 0j)

    if to_float(start_width) > to_float(end_width):
        reverse = True
        start_width, end_width = end_width, start_width
    else:
        reverse = False

    D = Component()
    xpts: list[Any] = []
    ypts: list[Any] = []
    if to_float(start_width) == to_float(end_width):  # Just return a square
        if symmetric:
            ypts = [
                -start_width / 2,
                start_width / 2,
                start_width / 2,
                -start_width / 2,
            ]
            xpts = [0, 0, start_width, start_width]
        if not symmetric:
            ypts = [0, start_width, start_width, 0]
            xpts = [0, 0, start_width, start_width]
        D.info["num_squares"] = 1
    else:
        xmin, _ = invert_step_point(
            y_desired=start_width * (1 + width_tol), W=start_width, a=end_width
        )
        xmax, _ = invert_step_point(
            y_desired=end_width * (1 - width_tol), W=start_width, a=end_width
        )

        xpts = list(xp.linspace(xmin, xmax, num_pts))
        for x in xpts:
            x, y = invert_step_point(x_desired=x, W=start_width, a=end_width)
            ypts.append(y)

        ypts[-1] = end_width
        ypts[0] = start_width
        x_curve = xp.asarray(xpts)
        y_curve = xp.asarray(ypts)

        if not symmetric:
            xpts.append(xpts[-1])
            ypts.append(0)
            xpts.append(xpts[0])
            ypts.append(0)
        else:
            xpts += list(xpts[::-1])
            ypts += [-y for y in ypts[::-1]]
            xpts = [x / 2 for x in xpts]
            ypts = [y / 2 for y in ypts]

        # anticrowding_factor stretches the wire out; a stretched wire is a
        # gentler transition, so there's less chance of current crowding if
        # the fabrication isn't perfect but as a result, the wire isn't as
        # short as it could be
        xpts = [x * anticrowding_factor for x in xpts]

        if reverse:
            xpts = [-x for x in xpts]
            start_width, end_width = end_width, start_width

        if has_abstract_tracers((x_curve, y_curve)):
            D.info["num_squares"] = xp.sum(
                xp.diff(x_curve) / ((y_curve[:-1] + y_curve[1:]) / 2)
            )
        else:
            y_num_sq = to_numpy(y_curve)
            x_num_sq = to_numpy(x_curve)
            D.info["num_squares"] = float(
                np.round(
                    np.sum(np.diff(x_num_sq) / ((y_num_sq[:-1] + y_num_sq[1:]) / 2)), 3
                )
            )

    xpts_arr = xp.asarray(xpts)
    ypts_arr = xp.asarray(ypts)
    D.add_polygon(xp.column_stack([xpts_arr, ypts_arr]), layer=layer)
    port_type = "electrical"
    xmin_pt = xp.min(xpts_arr)
    xmax_pt = xp.max(xpts_arr)
    if not symmetric:
        D.add_port(
            name="e1",
            center=(xmin_pt, start_width / 2),
            width=start_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        D.add_port(
            name="e2",
            center=(xmax_pt, end_width / 2),
            width=end_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
    if symmetric:
        D.add_port(
            name="e1",
            center=(xmin_pt, 0),
            width=start_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        D.add_port(
            name="e2",
            center=(xmax_pt, 0),
            width=end_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )

    for p in list(D.ports):
        if p.name and p.port_type == "electrical":
            D.create_pin(ports=[p], name=p.name)

    return D
