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


def _is_jax(x: Any) -> bool:
    return isinstance(x, jax.Array | jax.core.Tracer)


def _any_jax(args: Any, kwargs: Any) -> bool:
    for a in (*args, *kwargs.values()):
        if _is_jax(a):
            return True
        if isinstance(a, list | tuple):
            for b in a:
                if _is_jax(b):
                    return True
                if isinstance(b, list | tuple) and any(_is_jax(v) for v in b):
                    return True
    return False


class _DualNamespace:
    """numpy-compatible namespace that only uses jax.numpy when needed.

    ``xp.f(*args)`` calls ``jax.numpy.f`` if any argument is a jax array or a
    tracer (so derivatives flow), and plain ``numpy.f`` otherwise (fast, no
    dispatch overhead). Results computed from concrete inputs are numpy arrays.
    """

    def __init__(self, jmod: Any, nmod: Any) -> None:
        self._jmod = jmod
        self._nmod = nmod

    def __getattr__(self, name: str) -> Any:
        jf = getattr(self._jmod, name)
        nf = getattr(self._nmod, name, None)
        if name == "linalg":
            ns = _DualNamespace(jf, nf)
            setattr(self, name, ns)
            return ns
        if nf is None:
            setattr(self, name, jf)
            return jf
        if not callable(jf) or isinstance(jf, type):
            setattr(self, name, nf)
            return nf

        def f(*args: Any, **kwargs: Any) -> Any:
            if _any_jax(args, kwargs):
                return jf(*args, **kwargs)
            return nf(*args, **kwargs)

        f.__name__ = name
        f.__doc__ = getattr(nf, "__doc__", None)
        setattr(self, name, f)
        return f


xp: Any = _DualNamespace(jnp, np)


def xset(a: Any, index: Any, value: Any) -> Any:
    """Functional ``a[index] = value`` for numpy and jax arrays."""
    if _is_jax(a) or _is_jax(value):
        return jnp.asarray(a).at[index].set(value)
    out = np.array(a, dtype=np.float64, copy=True)
    out[index] = value
    return out


def xmul(a: Any, index: Any, value: Any) -> Any:
    """Functional ``a[index] *= value`` for numpy and jax arrays."""
    if _is_jax(a) or _is_jax(value):
        return jnp.asarray(a).at[index].multiply(value)
    out = np.array(a, dtype=np.float64, copy=True)
    out[index] *= value
    return out


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


def asarray(x: Any) -> Any:
    """Returns a float64 array: numpy for concrete values, jax if traced.

    numpy results are read-only (geometry is immutable, like jax arrays).
    """
    if _is_jax(x) or (isinstance(x, list | tuple) and _any_jax(x, {})):
        if isinstance(x, list | tuple) and not _is_jax(x):
            return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in x]) if x else jnp.zeros((0,))
        return jnp.asarray(x, dtype=jnp.float64)
    arr = np.array(x, dtype=np.float64)
    arr.flags.writeable = False
    return arr


def points_array(points: Any) -> Array:
    """Returns an (N, 2) float64 array (numpy, or jax if traced) from points."""
    if isinstance(points, jax.Array | np.ndarray):
        arr = asarray(points)
    else:
        pts = list(points)
        if pts and any(_is_jax(c) for p in pts for c in _iter_coords(p)):
            arr = jnp.stack(
                [jnp.stack([jnp.asarray(p[0], dtype=jnp.float64), jnp.asarray(p[1], dtype=jnp.float64)]) for p in pts]
            )
        else:
            arr = asarray([[to_float(p[0]), to_float(p[1])] for p in pts])
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


DBU = 1e-3


def snap_dbu(x: Any) -> Any:
    """Snaps to the 1 nm grid exactly like KLayout.

    KLayout converts um to dbu by multiplying with 1/dbu and rounds half away
    from zero; the floating point arithmetic is reproduced so that values on
    half-dbu boundaries round the same way as upstream.
    Straight-through for traced values (value snapped, gradient identity).
    Disabled when ``gdsfactory.snap.SNAP_ENABLED`` is False.
    """
    from gdsfactory import snap

    if not snap.SNAP_ENABLED:
        return x
    inv = 1.0 / DBU
    if _is_jax(x):
        v = np.asarray(primal(x), dtype=np.float64) * inv
        snapped = np.sign(v) * np.floor(np.abs(v) + 0.5) * DBU
        return x + jax.lax.stop_gradient(jnp.asarray(snapped) - x)
    if isinstance(x, int | float):
        v = x * inv
        return float(np.sign(v) * np.floor(abs(v) + 0.5) * DBU)
    v = np.asarray(x, dtype=np.float64) * inv
    out = np.sign(v) * np.floor(np.abs(v) + 0.5) * DBU
    out.flags.writeable = False
    return out


def stop_gradient(x: Any) -> Any:
    """jax.lax.stop_gradient that leaves numpy/python values untouched."""
    return jax.lax.stop_gradient(x) if _is_jax(x) else x


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
    return _trig_deg(angle, xp.cos, (1.0, 0.0, -1.0, 0.0))


def sin_deg(angle: Any) -> Any:
    """Exact sine for multiples of 90 degrees (keeps gradients)."""
    return _trig_deg(angle, xp.sin, (0.0, 1.0, 0.0, -1.0))


def _trig_deg(
    angle: Any, f: Callable[[Any], Any], exact: tuple[float, float, float, float]
) -> Any:
    a = primal(angle)
    a_f = float(np.asarray(a))
    q = a_f / 90.0
    is_exact = abs(q - round(q)) < 1e-12
    if not _is_jax(angle):
        if is_exact:
            return exact[round(q) % 4]
        return float(f(np.deg2rad(a_f)))
    value = f(deg2rad(jnp.asarray(angle, dtype=jnp.float64)))
    if is_exact:
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
    "xmul",
    "xp",
    "xset",
]
