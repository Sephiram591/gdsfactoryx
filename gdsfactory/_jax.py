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

Geometry whose structure does not depend on the traced values (polygons with a
fixed number of vertices, ports, reference transforms; grid snapping included)
also works inside ``jax.jit``, with the same results as eager. Operations that
need concrete values (discrete choices, KLayout operations) raise there.
"""

from __future__ import annotations

import functools
import os
import types
from collections.abc import Callable
from typing import Any, TypeAlias

import jax

if os.environ.get("JAX_ENABLE_X64", "1").lower() not in ("0", "false", "no"):
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# An explicit alias, so type checkers accept Array in annotations. Not a PEP 695 `type`
# statement (UP040): other modules use Array at runtime, where it must be the class.
Array: TypeAlias = jax.Array  # noqa: UP040


def _jax_float_dtype() -> Any:
    """Returns float64 when JAX x64 mode is enabled, else float32."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _is_jax(x: Any) -> bool:
    return isinstance(x, jax.Array | jax.core.Tracer)


def _has_jax(x: Any) -> bool:
    if _is_jax(x):
        return True
    if isinstance(x, list | tuple):
        return any(_has_jax(v) for v in x)
    if isinstance(x, dict):
        return any(_has_jax(v) for v in x.values())
    return False


def _any_jax(args: Any, kwargs: Any) -> bool:
    return any(_has_jax(a) for a in args) or any(
        _has_jax(v) for v in kwargs.values()
    )


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


def is_concrete_tracer(x: Any) -> bool:
    """Returns True if x is a tracer with a concrete primal (eager grad/jvp)."""
    if not is_tracer(x):
        return False
    try:
        primal(x)
    except TypeError:
        return False
    else:
        return True


def is_abstract_tracer(x: Any) -> bool:
    """Returns True if x is an abstract tracer (e.g. inside jax.jit/vmap)."""
    return is_tracer(x) and not is_concrete_tracer(x)


def _walk_has(tree: Any, pred: Callable[[Any], bool], visited: dict[int, Any]) -> bool:
    if tree is None or isinstance(
        tree, bool | int | float | complex | str | bytes | np.generic | type | types.ModuleType
    ):
        return False
    if isinstance(tree, np.ndarray) and tree.dtype != object:
        return False
    if pred(tree):
        return True
    if isinstance(tree, jax.Array):
        return False
    obj_id = id(tree)
    if obj_id in visited:
        return False
    visited[obj_id] = tree

    if pred is is_tracer:
        ht = getattr(tree, "has_tracers", None)
        if callable(ht):
            return bool(ht())

    if isinstance(tree, dict):
        return any(
            _walk_has(k, pred, visited) or _walk_has(v, pred, visited)
            for k, v in tree.items()
        )
    if isinstance(tree, list | tuple | set | frozenset):
        return any(_walk_has(v, pred, visited) for v in tree)
    if isinstance(tree, functools.partial):
        return (
            _walk_has(tree.func, pred, visited)
            or _walk_has(tree.args, pred, visited)
            or _walk_has(tree.keywords, pred, visited)
        )
    if isinstance(tree, types.MethodType):
        return _walk_has(tree.__self__, pred, visited) or _walk_has(
            tree.__func__, pred, visited
        )
    if callable(tree):
        closure = getattr(tree, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if _walk_has(val, pred, visited):
                    return True
        if _walk_has(getattr(tree, "__defaults__", None), pred, visited):
            return True
        if _walk_has(getattr(tree, "__kwdefaults__", None), pred, visited):
            return True

    if not isinstance(tree, dict | list | tuple | set | frozenset):
        leaves = jax.tree_util.tree_leaves(tree)
        if not (len(leaves) == 1 and leaves[0] is tree):
            if any(_walk_has(leaf, pred, visited) for leaf in leaves):
                return True

    try:
        d = vars(tree)
    except TypeError:
        d = None
    if d is not None and any(_walk_has(v, pred, visited) for v in d.values()):
        return True

    for cls in type(tree).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            try:
                val = object.__getattribute__(tree, slot)
            except AttributeError:
                continue
            if _walk_has(val, pred, visited):
                return True
    return False


def has_tracers(tree: Any) -> bool:
    """Returns True if any reachable leaf of tree is a JAX tracer."""
    return _walk_has(tree, is_tracer, {})


def has_abstract_tracers(tree: Any) -> bool:
    """Returns True if any reachable leaf of tree is an abstract JAX tracer."""
    return _walk_has(tree, is_abstract_tracer, {})


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
                    "This operation needs concrete values (a discrete choice such as a "
                    "number of points or an `if` on a value, or a KLayout operation). "
                    "Under jax.jit, gdsfactoryx supports geometry whose structure does "
                    "not depend on the traced values; use eager jax.grad/jvp/vjp otherwise."
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
        if x.ndim == 0:
            return float(x)
        return x
    return x


def _to_jnp_recursive(x: Any, dtype: Any) -> Any:
    if _is_jax(x):
        return jnp.asarray(x, dtype=dtype)
    if isinstance(x, list | tuple):
        if not x:
            return jnp.zeros((0,), dtype=dtype)
        return jnp.stack([_to_jnp_recursive(v, dtype) for v in x])
    return jnp.asarray(x, dtype=dtype)


def asarray(x: Any) -> Any:
    """Returns a float array: numpy for concrete values, jax if traced.

    numpy results are read-only (geometry is immutable, like jax arrays).
    """
    if _has_jax(x):
        return _to_jnp_recursive(x, _jax_float_dtype())
    arr = np.array(x, dtype=np.float64)
    arr.flags.writeable = False
    return arr


def _xy(p: Any) -> tuple[Any, Any]:
    if hasattr(p, "x") and hasattr(p, "y"):
        return p.x, p.y
    return p[0], p[1]


def points_array(points: Any) -> Array:
    """Returns an (N, 2) float array (numpy, or jax if traced) from points."""
    if isinstance(points, jax.Array | np.ndarray):
        arr = asarray(points)
    else:
        pts = list(points)
        if pts and any(_has_jax(c) for p in pts for c in _iter_coords(p)):
            dtype = _jax_float_dtype()
            rows = []
            for p in pts:
                if _is_jax(p) or isinstance(p, np.ndarray):
                    rows.append(jnp.asarray(p, dtype=dtype))
                else:
                    px, py = _xy(p)
                    rows.append(
                        jnp.stack(
                            [jnp.asarray(px, dtype=dtype), jnp.asarray(py, dtype=dtype)]
                        )
                    )
            arr = jnp.stack(rows)
        else:
            coords = []
            for p in pts:
                px, py = _xy(p)
                coords.append([to_float(px), to_float(py)])
            arr = asarray(coords)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(f"Expected (N, 2) points, got shape {arr.shape}")
    return arr


def _iter_coords(p: Any) -> tuple[Any, ...]:
    if _is_jax(p) or isinstance(p, np.ndarray):
        return (p,)
    if hasattr(p, "x") and hasattr(p, "y"):
        return (p.x, p.y)
    return tuple(p)


def float_or_array(x: Any) -> Any:
    """Pydantic before-validator: keeps jax values/sequences, converts scalars to float."""
    if x is None:
        return None
    if is_tracer(x) or isinstance(x, jax.Array):
        return x
    if isinstance(x, np.ndarray) and x.ndim > 0:
        return x
    if isinstance(x, list | tuple):
        if _has_jax(x):
            return asarray(x)
        return np.asarray(x, dtype=np.float64)
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
            return (
                np.round(np.asarray(x) / step) * step
                if np.ndim(x)
                else float(np.round(x / step) * step)
            )
        return np.round(x, decimals) if np.ndim(x) else float(np.round(x, decimals))
    xa = asarray(x)
    if step is not None:
        rounded = jnp.round(jax.lax.stop_gradient(xa) / step) * step
    else:
        rounded = jnp.round(jax.lax.stop_gradient(xa), decimals)
    return xa + jax.lax.stop_gradient(rounded - xa)


DBU = 1e-3


def get_dbu() -> float:
    """Returns the active KLayout database unit in um."""
    import kfactory as kf

    return float(kf.kcl.dbu)


def snap_dbu(x: Any, dbu: float | None = None) -> Any:
    """Snaps to the active DBU grid exactly like KLayout.

    KLayout converts um to dbu by multiplying with 1/dbu and rounds half away
    from zero; the floating point arithmetic is reproduced so that values on
    half-dbu boundaries round the same way as upstream.
    Straight-through for jax values (value snapped, gradient identity). jax
    values are snapped with jax.numpy, by the same operations as numpy values,
    so this also works on abstract tracers (inside ``jax.jit``).
    Disabled when ``gdsfactory.snap.SNAP_ENABLED`` is False.
    """
    from gdsfactory import snap

    if not snap.SNAP_ENABLED:
        return x
    unit = get_dbu() if dbu is None else float(dbu)
    inv = 1.0 / unit
    if _has_jax(x):
        xa = asarray(x)
        v = jax.lax.stop_gradient(xa) * inv
        snapped = jnp.sign(v) * jnp.floor(jnp.abs(v) + 0.5) * unit
        return xa + jax.lax.stop_gradient(snapped - xa)
    if isinstance(x, int | float):
        v = x * inv
        return float(np.sign(v) * np.floor(abs(v) + 0.5) * unit)
    v = np.asarray(x, dtype=np.float64) * inv
    out = np.sign(v) * np.floor(np.abs(v) + 0.5) * unit
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


def safe_sqrt(x: Any) -> Any:
    """Differentiable sqrt that returns 0 with 0 gradient at x <= 0."""
    if _has_jax(x):
        xa = asarray(x)
        pos = xa > 0.0
        safe_x = jnp.where(pos, xa, 1.0)
        return jnp.where(pos, jnp.sqrt(safe_x), 0.0)
    xa = np.asarray(x, dtype=np.float64)
    out = np.sqrt(np.maximum(xa, 0.0))
    return float(out) if np.ndim(x) == 0 else out


def safe_hypot(dx: Any, dy: Any) -> Any:
    """Differentiable hypot that returns 0 with 0 gradient at (0, 0)."""
    if _has_jax(dx) or _has_jax(dy):
        dx_a = asarray(dx)
        dy_a = asarray(dy)
        m = jnp.maximum(jnp.abs(dx_a), jnp.abs(dy_a))
        nz = m > 0.0
        safe_m = jnp.where(nz, m, 1.0)
        nx = dx_a / safe_m
        ny = dy_a / safe_m
        sq = jnp.where(nz, nx * nx + ny * ny, 1.0)
        return jnp.where(nz, safe_m * jnp.sqrt(sq), 0.0)
    out = np.hypot(np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64))
    return float(out) if np.ndim(dx) == 0 and np.ndim(dy) == 0 else out


def safe_arctan2(y: Any, x: Any) -> Any:
    """Differentiable arctan2 that avoids 0/0 NaN gradients at (0, 0)."""
    if _has_jax(y) or _has_jax(x):
        ya = asarray(y)
        xa = asarray(x)
        nz = (ya != 0.0) | (xa != 0.0)
        safe_x = jnp.where(nz, xa, 1.0)
        return jnp.where(nz, jnp.arctan2(ya, safe_x), 0.0)
    out = np.arctan2(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
    return float(out) if np.ndim(y) == 0 and np.ndim(x) == 0 else out


def deg2rad(angle: Any) -> Any:
    return angle * (jnp.pi / 180.0)


def cos_deg(angle: Any) -> Any:
    """Exact cosine for multiples of 90 degrees (keeps gradients)."""
    return _trig_deg(angle, jnp.cos, np.cos, (1.0, 0.0, -1.0, 0.0))


def sin_deg(angle: Any) -> Any:
    """Exact sine for multiples of 90 degrees (keeps gradients)."""
    return _trig_deg(angle, jnp.sin, np.sin, (0.0, 1.0, 0.0, -1.0))


def _trig_deg(
    angle: Any,
    jf: Callable[[Any], Any],
    nf: Callable[[Any], Any],
    exact: tuple[float, float, float, float],
) -> Any:
    if _has_jax(angle):
        dtype = _jax_float_dtype()
        a = jnp.asarray(angle, dtype=dtype)
        value = jf(deg2rad(a))
        q = jax.lax.stop_gradient(a) / 90.0
        q_round = jnp.round(q)
        is_exact = jnp.abs(q - q_round) < 1e-12
        idx = jnp.mod(q_round.astype(jnp.int32), 4)
        exact_v = jnp.asarray(exact, dtype=dtype)[idx]
        st_exact = value - jax.lax.stop_gradient(value) + exact_v
        return jnp.where(is_exact, st_exact, value)
    a_arr = np.asarray(angle, dtype=np.float64)
    q = a_arr / 90.0
    q_round = np.round(q)
    is_exact = np.abs(q - q_round) < 1e-12
    idx = np.mod(q_round.astype(np.int64), 4)
    exact_v = np.asarray(exact, dtype=np.float64)[idx]
    out = np.where(is_exact, exact_v, nf(np.deg2rad(a_arr)))
    return float(out) if a_arr.ndim == 0 else out


def _brentq_1d(
    residual_fn: Any,
    a: Any,
    b: Any,
    *params: Any,
    xtol: float = 1e-12,
) -> Any:
    """Solve 1D ``residual_fn(x, *params) = 0`` via ``brentq`` with implicit JVP.

    Runs ``scipy.optimize.brentq`` in float64 on concrete inputs (directly when
    untraced, or via ``jax.pure_callback`` under JAX tracing) and differentiates
    using the implicit function theorem ``dx/dp = -(df/dp) / (df/dx)``.
    """
    from scipy.optimize import brentq

    def _solve_concrete(a_val: Any, b_val: Any, *p_vals: Any) -> float:
        a_f = float(np.asarray(a_val, dtype=np.float64))
        b_f = float(np.asarray(b_val, dtype=np.float64))
        p_concrete = tuple(
            float(np.asarray(p, dtype=np.float64))
            if np.ndim(p) == 0
            else np.asarray(p, dtype=np.float64)
            for p in p_vals
        )
        return float(
            brentq(lambda x: float(residual_fn(x, *p_concrete)), a_f, b_f, xtol=xtol)
        )

    float_dtypes = [
        jnp.asarray(v).dtype
        for v in (a, b, *params)
        if isinstance(v, jax.Array | np.ndarray)
        and np.issubdtype(jnp.asarray(v).dtype, np.floating)
    ]
    out_dtype = jnp.result_type(*float_dtypes) if float_dtypes else _jax_float_dtype()

    if not has_tracers((a, b, *params)):
        has_jax_array = any(isinstance(v, jax.Array) for v in (a, b, *params))
        root = _solve_concrete(a, b, *params)
        return jnp.asarray(root, dtype=out_dtype) if has_jax_array else root

    a_arr = jnp.asarray(a, dtype=out_dtype)
    b_arr = jnp.asarray(b, dtype=out_dtype)
    p_arrs = tuple(jnp.asarray(p) for p in params)

    @jax.custom_jvp
    def _solve(a_in: Any, b_in: Any, *ps: Any) -> Any:
        def _cb(a_cb: Any, b_cb: Any, *ps_cb: Any) -> np.ndarray:
            return np.asarray(
                _solve_concrete(a_cb, b_cb, *ps_cb), dtype=np.dtype(out_dtype)
            )

        return jax.pure_callback(
            _cb,
            jax.ShapeDtypeStruct((), out_dtype),
            a_in,
            b_in,
            *ps,
            vmap_method="sequential",
        )

    @_solve.defjvp
    def _solve_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
        a_p, b_p, *ps_p = primals
        _a_t, _b_t, *ps_t = tangents
        x_star = _solve(a_p, b_p, *ps_p)

        def _res_of_ps(*ps_inner: Any) -> Any:
            return jnp.asarray(residual_fn(x_star, *ps_inner), dtype=out_dtype)

        def _res_of_x(x_inner: Any) -> Any:
            return jnp.asarray(residual_fn(x_inner, *ps_p), dtype=out_dtype)

        _, df_dp_dot = jax.jvp(_res_of_ps, tuple(ps_p), tuple(ps_t))
        _, df_dx = jax.jvp(_res_of_x, (x_star,), (jnp.ones_like(x_star),))
        nonzero = jnp.abs(df_dx) > 1e-30
        safe_denom = jnp.where(nonzero, df_dx, jnp.ones_like(df_dx))
        x_dot = jnp.where(nonzero, -df_dp_dot / safe_denom, jnp.zeros_like(x_star))
        return x_star, jnp.asarray(x_dot, dtype=out_dtype)

    return _solve(a_arr, b_arr, *p_arrs)


__all__ = [
    "DBU",
    "Array",
    "asarray",
    "cos_deg",
    "deg2rad",
    "float_or_array",
    "get_dbu",
    "has_abstract_tracers",
    "has_tracers",
    "is_abstract_tracer",
    "is_concrete_tracer",
    "is_tracer",
    "jnp",
    "maybe_float",
    "points_array",
    "primal",
    "safe_arctan2",
    "safe_hypot",
    "safe_sqrt",
    "sin_deg",
    "snap_dbu",
    "to_float",
    "to_numpy",
    "xmul",
    "xp",
    "xset",
]
