"""JAX helpers shared by the differentiable geometry backend.

gdsfactoryx stores every coordinate (polygon vertices, port centers, reference
transforms, ...) as ``jax.numpy`` float64 arrays so that any scalar derived
from a layout can be differentiated with :func:`jax.grad` with respect to the
float parameters of the component functions that produced it.

Differentiation is supported in *eager* mode (``jax.grad`` / ``jax.jvp`` /
``jax.vjp`` without ``jax.jit``): component functions keep their ordinary
Python control flow, which needs the primal values to be concrete.
Non-differentiable operations (boolean, offset, GDS export, ...) operate on the
concrete primal values and stop gradients.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

Array = jax.Array


def is_tracer(x: Any) -> bool:
    """Returns True if x is a JAX tracer (i.e. carries derivative information)."""
    return isinstance(x, jax.core.Tracer)


def has_tracers(tree: Any) -> bool:
    """Returns True if any leaf of a pytree (or nested dict/list/tuple) is a tracer."""
    if is_tracer(tree):
        return True
    if isinstance(tree, dict):
        return any(has_tracers(v) for v in tree.values())
    if isinstance(tree, list | tuple | set | frozenset):
        return any(has_tracers(v) for v in tree)
    if hasattr(tree, "__dict__") and not isinstance(tree, type):
        # pydantic models / dataclasses (e.g. CrossSection, Section)
        try:
            values = vars(tree)
        except TypeError:
            return False
        return any(is_tracer(v) for v in values.values()) or any(
            has_tracers(v)
            for v in values.values()
            if isinstance(v, list | tuple | dict)
        )
    return False


def primal(x: Any) -> Any:
    """Strips all JAX transformation levels and returns the concrete value.

    Works for tracers created by ``jax.grad``/``jax.jvp``/``jax.vjp`` in eager
    mode. Raises for abstract tracers (e.g. inside ``jax.jit``).
    """
    while is_tracer(x):
        if hasattr(x, "primal"):
            x = x.primal
        elif hasattr(x, "val"):
            x = x.val
        else:
            try:
                return jax.core.concrete_or_error(np.asarray, x)
            except Exception as e:  # pragma: no cover
                raise TypeError(
                    f"Cannot get a concrete value from {type(x).__name__}. "
                    "gdsfactoryx supports eager jax.grad/jvp/vjp, not jax.jit."
                ) from e
    return x


def to_numpy(x: Any, dtype: Any = np.float64) -> npt.NDArray[Any]:
    """Returns a concrete numpy array (gradients are stopped)."""
    return np.asarray(primal(x), dtype=dtype)


def to_float(x: Any) -> float:
    """Returns a concrete python float (gradients are stopped)."""
    return float(np.asarray(primal(x)))


def maybe_float(x: Any) -> Any:
    """Returns a python float for concrete numbers, keeps tracers untouched."""
    if is_tracer(x):
        return x
    if isinstance(x, jax.Array | np.ndarray | np.generic):
        if np.ndim(x) == 0:
            return float(x)
        return x
    return x


def asarray(x: Any) -> Array:
    """Returns a float64 jax array."""
    return jnp.asarray(x, dtype=jnp.float64)


def points_array(points: Any) -> Array:
    """Returns an (N, 2) float64 jax array from a sequence of points."""
    if isinstance(points, jax.Array | np.ndarray):
        arr = jnp.asarray(points, dtype=jnp.float64)
    else:
        pts = list(points)
        if pts and any(is_tracer(c) for p in pts for c in _iter_coords(p)):
            arr = jnp.stack([jnp.stack([asarray(p[0]), asarray(p[1])]) for p in pts])
        else:
            arr = jnp.asarray(
                np.asarray([[to_float(p[0]), to_float(p[1])] for p in pts]),
                dtype=jnp.float64,
            )
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(f"Expected (N, 2) points, got shape {arr.shape}")
    return arr


def _iter_coords(p: Any) -> tuple[Any, ...]:
    if isinstance(p, jax.Array | np.ndarray):
        return (p,)
    if hasattr(p, "x") and hasattr(p, "y"):
        return (p.x, p.y)
    return tuple(p)


def float_or_array(x: Any) -> Any:
    """Pydantic before-validator: keeps jax values, converts the rest to float."""
    if x is None:
        return None
    if is_tracer(x) or isinstance(x, jax.Array):
        return x
    if isinstance(x, np.ndarray) and x.ndim > 0:
        return x
    return float(x)


def round_st(x: Any, decimals: int = 0, step: float | None = None) -> Any:
    """Straight-through rounding: the value is rounded, the gradient is identity.

    Args:
        x: value (python number, numpy or jax array, tracer).
        decimals: number of decimals (ignored if step is given).
        step: round to multiples of step.
    """
    if not is_tracer(x) and not isinstance(x, jax.Array):
        if step is not None:
            return np.round(np.asarray(x) / step) * step if np.ndim(x) else float(
                np.round(x / step) * step
            )
        return np.round(x, decimals) if np.ndim(x) else float(np.round(x, decimals))
    xa = asarray(x)
    if step is not None:
        rounded = jnp.round(jax.lax.stop_gradient(xa) / step) * step
    else:
        rounded = jnp.round(jax.lax.stop_gradient(xa), decimals)
    return xa + jax.lax.stop_gradient(rounded - xa)


def stop_gradient_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda v: jax.lax.stop_gradient(v) if isinstance(v, jax.Array) else v, tree
    )


def smooth_abs(x: Array, eps: float = 0.0) -> Array:
    return jnp.sqrt(x * x + eps) if eps else jnp.abs(x)


def deg2rad(angle: Any) -> Any:
    return angle * (jnp.pi / 180.0)


def cos_deg(angle: Any) -> Any:
    """Exact cosine for multiples of 90 degrees (keeps gradients)."""
    return _trig_deg(angle, jnp.cos, (1.0, 0.0, -1.0, 0.0))


def sin_deg(angle: Any) -> Any:
    """Exact sine for multiples of 90 degrees (keeps gradients)."""
    return _trig_deg(angle, jnp.sin, (0.0, 1.0, 0.0, -1.0))


def _trig_deg(
    angle: Any, f: Callable[[Any], Any], exact: tuple[float, float, float, float]
) -> Any:
    a = primal(angle)
    a_f = float(np.asarray(a))
    q = a_f / 90.0
    value = f(deg2rad(asarray(angle)))
    if abs(q - round(q)) < 1e-12:
        # exact value with the derivative of the smooth function
        exact_v = exact[round(q) % 4]
        return value - jax.lax.stop_gradient(value) + exact_v
    return value


__all__ = [
    "Array",
    "asarray",
    "cos_deg",
    "deg2rad",
    "float_or_array",
    "has_tracers",
    "is_tracer",
    "jnp",
    "maybe_float",
    "points_array",
    "primal",
    "sin_deg",
    "to_float",
    "to_numpy",
]
