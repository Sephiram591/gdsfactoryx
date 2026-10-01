"""Cell decorator for the differentiable backend.

``@gf.cell`` turns a function returning a Component into a cached factory:

- the bound arguments are stored in ``component.settings``
- the component gets a deterministic name from function name + parameters
- calls with identical (concrete) arguments return the same locked Component

When any argument carries a JAX tracer (e.g. inside ``jax.grad``), the cache is
bypassed so derivative information flows through the geometry.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, cast

from gdsfactory._jax import has_tracers

if TYPE_CHECKING:
    from gdsfactory.component import Component

ComponentParams = ParamSpec("ComponentParams")

MAX_NAME_LENGTH = 99
_CACHES: list[dict[Any, Any]] = []
factories: dict[str, Callable[..., Any]] = {}
"""Registry of cell functions (name -> decorated function)."""


class ComponentFunc(Protocol[ComponentParams]):
    __name__: str

    def __call__(
        self, *args: ComponentParams.args, **kwargs: ComponentParams.kwargs
    ) -> Component: ...


def clean_name(name: str) -> str:
    """Ensures that gds cells are composed of [a-zA-Z0-9_\\-]."""
    from kfactory.serialization import clean_name as _kf_clean_name

    return _kf_clean_name(name)


def _module_basename(func: Callable[..., Any]) -> str:
    name, module = func.__name__, func.__module__
    return clean_name(name if module == "__main__" else f"{name}_{module}")


def _concrete(value: Any) -> Any:
    """Replaces tracers / jax arrays by concrete python values (for naming)."""
    import numpy as np

    from gdsfactory._jax import is_tracer, primal

    if is_tracer(value):
        value = primal(value)
    if type(value).__module__.startswith("jax"):
        arr = np.asarray(value)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    if isinstance(value, dict):
        return {k: _concrete(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_concrete(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_concrete(v) for v in value)
    return value


def get_cell_name(cell_type: str, max_cellname_length: int | None = None, **kwargs: Any) -> str:
    """Returns a cell name from the function name and its parameters.

    Same algorithm as kfactory (so names match upstream gdsfactory).
    """
    from kfactory.serialization import get_cell_name as _kf_get_cell_name

    params = {k: _concrete(v) for k, v in kwargs.items()}
    return _kf_get_cell_name(cell_type, max_cellname_length=max_cellname_length, **params)


def _metadata(value: Any) -> Any:
    """Settings value as kfactory stores it (functions -> names); keeps jax values."""
    from kfactory.serialization import convert_metadata_type

    if has_tracers(value) or type(value).__module__.startswith("jax"):
        return value
    if isinstance(value, dict):
        return {k: _metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_metadata(v) for v in value)
    try:
        return convert_metadata_type(value)
    except Exception:
        return value


def _freeze(value: Any) -> Any:
    """Hashable key for caching (only called without tracers)."""
    import pydantic

    from gdsfactory.serialization import clean_value_json

    if isinstance(value, pydantic.BaseModel):
        # exact field values (clean_value_json rounds floats)
        try:
            return ("__model__", type(value).__name__, _freeze(dict(value)))
        except Exception:
            return ("__id__", id(value))

    try:
        hash(value)
        if not isinstance(value, float | int | str | bool | type(None) | tuple | frozenset):
            # functions / partials / components: hash by identity or serialization
            if callable(value) or hasattr(value, "name"):
                try:
                    return ("__json__", repr(clean_value_json(value)))
                except Exception:
                    return ("__id__", id(value))
        if isinstance(value, tuple):
            return tuple(_freeze(v) for v in value)
        return value
    except TypeError:
        pass
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    try:
        return ("__json__", repr(clean_value_json(value)))
    except Exception:
        return ("__id__", id(value))


def clear_cache() -> None:
    for c in _CACHES:
        c.clear()


def cached_cells() -> list[Any]:
    """Components currently stored in the cell caches."""
    return [c for cache in _CACHES for c in cache.values()]


def cell(
    _func: Callable[..., Any] | None = None,
    /,
    *,
    set_settings: bool = True,
    set_name: bool = True,
    check_ports: bool = True,
    check_instances: Any = None,
    snap_ports: bool = True,
    add_port_layers: bool = True,
    cache: dict[Any, Any] | None = None,
    basename: str | None = None,
    drop_params: list[str] | None = None,
    register_factory: bool = True,
    overwrite_existing: bool | None = None,
    layout_cache: bool | None = None,
    info: dict[str, Any] | None = None,
    post_process: Iterable[Callable[[Component], None]] | None = None,
    debug_names: bool | None = None,
    tags: list[str] | None = None,
    with_module_name: bool = False,
    lvs_equivalent_ports: list[list[str]] | None = None,
    ports: Any = None,
    schematic_function: Callable[..., Any] | None = None,
    output_type: Any = None,
) -> Any:
    """Decorator to convert a function into a (cached, differentiable) Component factory."""
    drop = set(drop_params if drop_params is not None else ["self", "cls"])
    post = list(post_process or [])

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        _cache: dict[Any, Any] = cache if cache is not None else {}
        _CACHES.append(_cache)
        _basename = basename or (_module_basename(func) if with_module_name else None)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Component:
            from gdsfactory.component import Component

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = {k: v for k, v in bound.arguments.items() if k not in drop}
            # flatten **kwargs parameters
            for name, p in sig.parameters.items():
                if p.kind is inspect.Parameter.VAR_KEYWORD and name in params:
                    params.update(params.pop(name))
            traced = has_tracers(params)
            key = None
            if not traced:
                try:
                    key = _freeze(params)
                    hash(key)
                except Exception:
                    key = None
                if key is not None and key in _cache:
                    return cast("Component", _cache[key])

            c = func(*bound.args, **bound.kwargs)
            if not isinstance(c, Component):
                raise TypeError(
                    f"{func.__name__} must return a Component, got {type(c)}"
                )
            if c.locked:
                # returned a cached cell from another factory: wrap a copy
                c = c.copy()

            function_name = func.__name__
            if set_name:
                c._name = get_cell_name(_basename or function_name, **params)
            if set_settings:
                c.settings = type(c.settings)(
                    {k: _metadata(v) for k, v in params.items()}
                )
                c.function_name = function_name
                c.basename = _basename
                c.module = func.__module__
            if info:
                c.info.update(info)
            if lvs_equivalent_ports:
                c.lvs_equivalent_ports = lvs_equivalent_ports
            for pp in post:
                pp(c)
            if schematic_function is not None:
                c._schematic_function = schematic_function  # type: ignore[attr-defined]
            c.locked = True
            if key is not None:
                _cache[key] = c
            return c

        wrapper.is_gf_cell = True  # type: ignore[attr-defined]
        wrapper.cache = _cache  # type: ignore[attr-defined]
        wrapper.tags = tags or []  # type: ignore[attr-defined]
        if schematic_function is not None:
            wrapper.schematic_function = schematic_function  # type: ignore[attr-defined]

        def get_schematic(*args: Any, **kwargs: Any) -> Any:
            if schematic_function is None:
                raise ValueError(f"{func.__name__} has no schematic_function")
            return schematic_function(*args, **kwargs)

        wrapper.get_schematic = get_schematic  # type: ignore[attr-defined]
        if register_factory:
            factories[_basename or func.__name__] = wrapper
        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


def vcell(_func: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
    """Like ``cell`` (all components support any angle in gdsfactoryx)."""
    kwargs = {k: v for k, v in kwargs.items() if k in _CELL_KWARGS_EARLY}

    def mark(f: Callable[..., Any]) -> Callable[..., Any]:
        f.is_gf_vcell = True  # type: ignore[attr-defined]
        return f

    if _func is not None:
        return mark(cell(_func, **kwargs))
    return lambda f: mark(cell(**kwargs)(f))


_CELL_KWARGS_EARLY = set(inspect.signature(cell).parameters) - {"_func"}


def cell_with_module_name(_func: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
    """Decorator like ``cell`` but with ``with_module_name=True`` by default."""
    return cell(_func, with_module_name=True, **kwargs)


def schematic_cell(_func: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
    """Builds a Component from a function returning a schematic/netlist dict."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def build(*args: Any, **kw: Any) -> Component:
            from gdsfactory.read.from_yaml import from_yaml

            schematic = func(*args, **kw)
            netlist = schematic.model_dump() if hasattr(schematic, "model_dump") else schematic
            return from_yaml(netlist)

        return cell(build, **{k: v for k, v in kwargs.items() if k in _CELL_KWARGS})

    if _func is not None:
        return decorator(_func)
    return decorator


_CELL_KWARGS = set(inspect.signature(cell).parameters) - {"_func"}

__all__ = [
    "cell",
    "cell_with_module_name",
    "clean_name",
    "cached_cells",
    "factories",
    "clear_cache",
    "get_cell_name",
    "schematic_cell",
    "vcell",
]
