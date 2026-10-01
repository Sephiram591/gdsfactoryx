from __future__ import annotations

__all__ = ["optimal_step"]

import numpy as np

import gdsfactory as gf
from gdsfactory._jax import has_tracers, stop_gradient, to_float, xp
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

    def step_points(eta: float, W: complex, a: complex) -> tuple[float, float]:
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
        x_desired: float = -10,
        y_desired: float | None = None,
        W: float = 1,
        a: float = 2,
    ) -> tuple[float, float]:
        """Finds the eta associated with x_desired or y_desired along the optimal curve."""

        W0, a0 = to_float(W), to_float(a)
        x0 = None if x_desired is None else to_float(x_desired)
        y0 = None if y_desired is None else to_float(y_desired)

        def fh(eta: float) -> float:
            guessed_x, guessed_y = step_points(eta, W=W0 + 0j, a=a0 + 0j)
            if y0 is None:
                return (guessed_x - x0) ** 2  # Error relative to x_desired
            return (guessed_y - y0) ** 2  # Error relative to y_desired

        from scipy.optimize import fminbound

        # Minimize error to find optimal eta
        found_eta = fminbound(fh, 0, np.pi)
        if not has_tracers([x_desired, y_desired, W, a]):
            return step_points(found_eta, W=W + 0j, a=a + 0j)

        # differentiable eta via the implicit function theorem
        # (value is the minimizer found above, gradient is d eta / d params)
        import jax

        idx = 0 if y_desired is None else 1
        target = y_desired if y_desired is not None else x_desired
        g_eta = float(
            jax.grad(lambda e: step_points(e, W=W0 + 0j, a=a0 + 0j)[idx])(
                float(found_eta)
            )
        )
        residual = target - step_points(found_eta, W=W + 0j, a=a + 0j)[idx]
        eta = found_eta + (residual - stop_gradient(residual)) / g_eta
        return step_points(eta, W=W + 0j, a=a + 0j)

    if start_width > end_width:
        reverse = True
        start_width, end_width = end_width, start_width
    else:
        reverse = False

    D = Component()
    xpts: list[float] = []
    ypts: list[float] = []
    if start_width == end_width:  # Just return a square
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
        y_num_sq = np.array([to_float(y) for y in ypts])
        x_num_sq = np.array([to_float(x) for x in xpts])

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

        D.info["num_squares"] = float(
            np.round(
                np.sum(np.diff(x_num_sq) / ((y_num_sq[:-1] + y_num_sq[1:]) / 2)), 3
            )
        )

    D.add_polygon(list(zip(xpts, ypts, strict=False)), layer=layer)
    port_type = "electrical"
    if not symmetric:
        D.add_port(
            name="e1",
            center=(min(xpts), start_width / 2),
            width=start_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        D.add_port(
            name="e2",
            center=(max(xpts), end_width / 2),
            width=end_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
    if symmetric:
        D.add_port(
            name="e1",
            center=(min(xpts), 0),
            width=start_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        D.add_port(
            name="e2",
            center=(max(xpts), 0),
            width=end_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )

    for p in list(D.ports):
        if p.name and p.port_type == "electrical":
            D.create_pin(ports=[p], name=p.name)

    return D
