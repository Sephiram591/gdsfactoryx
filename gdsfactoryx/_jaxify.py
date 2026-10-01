"""`jaxify`: make any gdsfactory function differentiable with JAX.

The wrapped function runs the *real* gdsfactory code (so the geometry is
bit-for-bit what gdsfactory produces) inside a `jax.pure_callback`, and exposes
its derivative through a `jax.custom_jvp` rule whose Jacobian is computed by
central finite differences of the aligned geometry. Because the tangent output
is linear in the input tangents, both forward mode (`jvp`, `jacfwd`) and reverse
mode (`grad`, `vjp`, `jacrev`) work, as do `jit` and `vmap`.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as _jax_core

from gdsfactoryx._extract import Raw, Spec, align, flatten, spec_of, to_raw
from gdsfactoryx._pdk import check_dbu
from gdsfactoryx.geometry import Geometry, PortGeometry


@dataclass
class Settings:
    """Global defaults for differentiable evaluation.

    fd_step: central finite-difference step (in the units of the perturbed
        argument, um for lengths). gdsfactory snaps geometry to the database
        unit (1 nm by default), so the per-vertex derivative error is bounded by
        roughly dbu / (2 * fd_step); geometry that is linear in the argument
        (lengths, widths, offsets, radii) is exact for any step.
    cache_size: number of gdsfactory evaluations memoized per wrapped function.
    """

    fd_step: float = 0.02
    cache_size: int = 512


settings = Settings()


class TemplateError(RuntimeError):
    """Raised when the output topology is unknown under jit/vmap."""


def _is_dynamic(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating)


def _concrete(x: jax.Array) -> np.ndarray | None:
    """Concrete value of `x` if available (eager, grad, jacfwd), else None."""
    value = _jax_core.to_concrete_value(x)
    return None if value is None else np.asarray(value, dtype=np.float64)


def _hashable(obj: Any) -> bool:
    try:
        hash(obj)
    except TypeError:
        return False
    return True


@dataclass(frozen=True, eq=False)
class _Call:
    """One call signature: static arguments plus the layout of dynamic ones."""

    wrapper: JaxifiedFunction
    treedef: Any
    static: tuple[Any, ...]
    dyn_idx: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]

    @functools.cached_property
    def key(self) -> Any:
        key = (self.treedef, self.static, self.dyn_idx, self.shapes)
        return key if _hashable(key) else None

    @property
    def size(self) -> int:
        return int(sum(np.prod(s, dtype=int) for s in self.shapes))

    def arguments(self, x: np.ndarray) -> tuple[tuple[Any, ...], dict[str, Any]]:
        leaves = list(self.static)
        offset = 0
        for i, shape in zip(self.dyn_idx, self.shapes, strict=True):
            n = int(np.prod(shape, dtype=int))
            value = x[offset : offset + n]
            leaves[i] = float(value[0]) if shape == () else value.reshape(shape)
            offset += n
        args, kwargs = jax.tree_util.tree_unflatten(self.treedef, leaves)
        return args, kwargs

    def evaluate(self, x: np.ndarray) -> Raw:
        x = np.asarray(x, dtype=np.float64)
        w = self.wrapper
        cache_key = None if self.key is None else (self.key, x.tobytes())
        if cache_key is not None and cache_key in w._cache:
            w._cache.move_to_end(cache_key)
            return w._cache[cache_key]
        args, kwargs = self.arguments(x)
        raw = to_raw(w.func(*args, **kwargs), merge=w.merge, layers=w.layers)
        if cache_key is not None:
            w._cache[cache_key] = raw
            while len(w._cache) > settings.cache_size:
                w._cache.popitem(last=False)
        return raw


def _unflatten(flat: jax.Array, spec: Spec) -> Any:
    if spec.kind == "path":
        return flat.reshape(spec.npoints, 2)
    if spec.kind == "tree":
        leaves, offset = [], 0
        for shape in spec.shapes:
            n = int(np.prod(shape, dtype=int))
            leaves.append(flat[offset : offset + n].reshape(shape))
            offset += n
        return jax.tree_util.tree_unflatten(spec.treedef, leaves)

    polygons, offset = {}, 0
    for key, counts in spec.layers:
        polys = []
        for n in counts:
            polys.append(flat[offset : offset + 2 * n].reshape(n, 2))
            offset += 2 * n
        polygons[key] = polys
    ports = {}
    for name in spec.ports:
        v = flat[offset : offset + 4]
        ports[name] = PortGeometry(center=v[:2], orientation=v[2], width=v[3])
        offset += 4
    return Geometry(polygons=polygons, ports=ports)


class JaxifiedFunction:
    """A gdsfactory function that accepts JAX arrays and is differentiable.

    Calling it with no floating-point `jax.Array` arguments simply calls the
    original function and returns its native result (e.g. a `gf.Component`).
    When any argument (at any depth of the argument pytree) is a floating
    `jax.Array` or tracer, it instead returns a differentiable result:

    - Component / ComponentReference -> :class:`gdsfactoryx.Geometry`
    - gf.Path -> (N, 2) array of path points
    - pytree of numbers -> same pytree of arrays

    The wrapped function receives Python floats (or numpy arrays) wherever the
    caller passed JAX arrays, so it can be any ordinary gdsfactory code.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        step: float | None = None,
        merge: bool = True,
        layers: Any = None,
    ) -> None:
        functools.update_wrapper(self, func)
        self.func = func
        self.step = step
        self.merge = merge
        self.layers = layers
        self._templates: dict[Any, tuple[Spec, Raw]] = {}
        self._fns: dict[Any, Callable[[jax.Array], jax.Array]] = {}
        self._cache: OrderedDict[Any, Raw] = OrderedDict()

    def __repr__(self) -> str:
        name = getattr(self.func, "__qualname__", repr(self.func))
        return f"<jaxified {name}>"

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        return self if obj is None else functools.partial(self, obj)

    def with_options(self, **options: Any) -> JaxifiedFunction:
        """Returns a copy with different `step`, `merge` or `layers`."""
        kwargs = {"step": self.step, "merge": self.merge, "layers": self.layers}
        kwargs.update(options)
        return JaxifiedFunction(self.func, **kwargs)

    # -- public helpers -----------------------------------------------------

    def component(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the original function with JAX arrays converted to floats.

        Use it to get the real `gf.Component` (to write GDS, plot, route...)
        at the current (e.g. optimized) parameter values.
        """
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        leaves = [
            (float(np.asarray(v)) if np.ndim(v) == 0 else np.asarray(v))
            if _is_dynamic(v)
            else v
            for v in leaves
        ]
        args, kwargs = jax.tree_util.tree_unflatten(treedef, leaves)
        return self.func(*args, **kwargs)

    def template(self, *args: Any, **kwargs: Any) -> Spec:
        """Records the output topology at concrete arguments.

        Needed before calling under `jit`/`vmap` if the function has not been
        called eagerly with the same static arguments yet.
        """
        call, x, _ = self._prepare(args, kwargs)
        if call is None:
            raise TemplateError("template() needs at least one floating jax.Array")
        value = _concrete(x)
        if value is None:
            raise TemplateError("template() needs concrete (non-traced) arguments")
        return self._template(call, value)[0]

    # -- implementation -----------------------------------------------------

    def _prepare(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[_Call, jax.Array, Any]:
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        dyn_idx = tuple(i for i, leaf in enumerate(leaves) if _is_dynamic(leaf))
        if not dyn_idx:
            return None, None, None  # type: ignore[return-value]
        dyn = [leaves[i] for i in dyn_idx]
        dtype = jnp.result_type(*dyn)
        static = tuple(None if i in dyn_idx else v for i, v in enumerate(leaves))
        call = _Call(
            wrapper=self,
            treedef=treedef,
            static=static,
            dyn_idx=dyn_idx,
            shapes=tuple(tuple(d.shape) for d in dyn),
        )
        x = jnp.concatenate([jnp.ravel(d).astype(dtype) for d in dyn])
        return call, x, dtype

    def _template(self, call: _Call, value: np.ndarray) -> tuple[Spec, Raw]:
        raw = call.evaluate(value)
        template = (spec_of(raw), raw)
        if call.key is not None:
            self._templates[call.key] = template
        return template

    def _step(self) -> float:
        return float(self.step if self.step is not None else settings.fd_step)

    def _build(
        self, call: _Call, spec: Spec, ref: Raw, dtype: Any
    ) -> Callable[[jax.Array], jax.Array]:
        size, n = spec.size, call.size
        angle = spec.angle_mask()

        def forward_host(x: np.ndarray) -> np.ndarray:
            raw = align(call.evaluate(x), spec, ref)
            return flatten(raw).astype(dtype)

        def jacobian_host(x: np.ndarray) -> np.ndarray:
            check_dbu()
            x = np.asarray(x, dtype=np.float64)
            base = align(call.evaluate(x), spec, ref)
            h = self._step()
            jac = np.empty((size, n))
            for i in range(n):
                dx = np.zeros(n)
                dx[i] = h
                plus = flatten(align(call.evaluate(x + dx), spec, base))
                minus = flatten(align(call.evaluate(x - dx), spec, base))
                diff = plus - minus
                diff[angle] = (diff[angle] + 180.0) % 360.0 - 180.0
                jac[:, i] = diff / (2 * h)
            return jac.astype(dtype)

        out_shape = jax.ShapeDtypeStruct((size,), dtype)
        jac_shape = jax.ShapeDtypeStruct((size, n), dtype)

        @jax.custom_jvp
        def fn(x: jax.Array) -> jax.Array:
            return jax.pure_callback(
                forward_host, out_shape, x, vmap_method="sequential"
            )

        @fn.defjvp
        def fn_jvp(
            primals: tuple[jax.Array], tangents: tuple[jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            (x,), (t,) = primals, tangents
            jac = jax.pure_callback(
                jacobian_host, jac_shape, x, vmap_method="sequential"
            )
            return fn(x), jac @ t.astype(dtype)

        return fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call, x, dtype = self._prepare(args, kwargs)
        if call is None:
            return self.func(*args, **kwargs)

        value = _concrete(x)
        if value is not None:
            spec, ref = self._template(call, value)
        elif call.key is not None and call.key in self._templates:
            spec, ref = self._templates[call.key]
        else:
            name = getattr(self.func, "__name__", "function")
            raise TemplateError(
                f"The output topology of {name} is unknown under jit/vmap. Call it "
                f"once eagerly with the same static arguments, or call "
                f"`{name}.template(...)` with example values first."
            )

        fn_key = None if call.key is None else (call.key, spec, np.dtype(dtype))
        fn = self._fns.get(fn_key) if fn_key is not None else None
        if fn is None:
            fn = self._build(call, spec, ref, dtype)
            if fn_key is not None:
                self._fns[fn_key] = fn
        return _unflatten(fn(x), spec)


def jaxify(
    func: Callable[..., Any] | None = None,
    *,
    step: float | None = None,
    merge: bool = True,
    layers: Any = None,
) -> Any:
    """Wraps a gdsfactory function so it accepts JAX arrays and is differentiable.

    Usable directly (`jaxify(gf.components.ring_single)`) or as a decorator
    (`@jaxify` / `@jaxify(step=0.01)`).

    Args:
        func: any function returning a Component, ComponentReference, Path or a
            pytree of numbers.
        step: finite-difference step; defaults to `gdsfactoryx.settings.fd_step`.
        merge: merge polygons per layer before extraction (recommended for
            rasterization, avoids double counting overlaps).
        layers: restrict extraction to these layers (default: all non-empty).
    """
    if func is None:
        return functools.partial(jaxify, step=step, merge=merge, layers=layers)
    if isinstance(func, JaxifiedFunction):
        return func.with_options(step=step, merge=merge, layers=layers)
    return JaxifiedFunction(func, step=step, merge=merge, layers=layers)


def cell(
    func: Callable[..., Any] | None = None,
    *,
    step: float | None = None,
    merge: bool = True,
    layers: Any = None,
    **cell_kwargs: Any,
) -> Any:
    """`gf.cell` + `jaxify`: a cached, differentiable component factory.

    The decorated function is written with ordinary gdsfactory code; it receives
    Python floats where the caller passed JAX arrays.
    """
    import gdsfactory as gf

    if func is None:
        return functools.partial(
            cell, step=step, merge=merge, layers=layers, **cell_kwargs
        )
    return JaxifiedFunction(
        gf.cell(func, **cell_kwargs), step=step, merge=merge, layers=layers
    )
