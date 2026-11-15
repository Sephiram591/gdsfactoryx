"""Cell decorator for the differentiable backend.

``@gf.cell`` turns a function returning a Component into a cached factory:

- the bound arguments are stored in ``component.settings``
- the component gets a deterministic name from function name + parameters
- calls with identical (concrete) arguments return the same locked Component

When any argument carries a JAX tracer (e.g. inside ``jax.grad``), the cache is
bypassed so derivative information flows through the geometry.
"""

from __future__ import annotations

import copy as _copy
import functools
import inspect
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, cast

from gdsfactory._jax import has_abstract_tracers, has_tracers, snap_dbu
from gdsfactory.transform import manhattan_angle

if TYPE_CHECKING:
    from gdsfactory.component import Component

ComponentParams = ParamSpec("ComponentParams")

MAX_NAME_LENGTH = 99
_CACHES: list[dict[Any, Any]] = []
_BY_NAME: dict[str, Any] = {}
factories: dict[str, Callable[..., Any]] = {}
"""Registry of cell functions (name -> decorated function)."""
_SNAP_PORTS_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "_SNAP_PORTS_OVERRIDE", default=None
)


class ComponentFunc(Protocol[ComponentParams]):
    __name__: str

    def __call__(
        self, *args: ComponentParams.args, **kwargs: ComponentParams.kwargs
    ) -> Component: ...


def clean_name(name: str) -> str:
    r"""Ensures that gds cells are composed of [a-zA-Z0-9_\-]."""
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
        # kfactory sorts nested dict keys when naming
        return {k: _concrete(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_concrete(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_concrete(v) for v in value)
    return value


def get_cell_name(
    cell_type: str, max_cellname_length: int | None = None, **kwargs: Any
) -> str:
    """Returns a cell name from the function name and its parameters.

    Same algorithm as kfactory (so names match upstream gdsfactory).
    """
    from kfactory.serialization import get_cell_name as _kf_get_cell_name

    params = {k: _concrete(v) for k, v in kwargs.items()}
    return _kf_get_cell_name(
        cell_type, max_cellname_length=max_cellname_length, **params
    )


def _layout_cache_enabled() -> bool:
    from gdsfactory.config import CONF

    return bool(getattr(CONF, "cell_layout_cache", False))


def _metadata(value: Any) -> Any:
    """Settings value as kfactory stores it (functions -> names); keeps jax values."""
    from kfactory.serialization import convert_metadata_type

    if isinstance(value, dict):
        return {k: _metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_metadata(v) for v in value)
    if has_tracers(value) or type(value).__module__.startswith("jax"):
        return value
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
        if not isinstance(
            value, float | int | str | bool | type(None) | tuple | frozenset
        ):
            # functions / partials / components: hash by identity or serialization
            if callable(value) or hasattr(value, "name"):
                try:
                    return ("__json__", repr(clean_value_json(value)))
                except Exception:
                    return ("__id__", id(value))
        if isinstance(value, tuple):
            return tuple(_freeze(v) for v in value)
    except TypeError:
        pass
    else:
        return value
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
    _BY_NAME.clear()
    from gdsfactory.routing._kf_router import _clear_routing_mirror

    _clear_routing_mirror()


def cached_cells() -> list[Any]:
    """Components currently stored in the cell caches."""
    return [c for cache in _CACHES for c in cache.values()]


def _check_ports(c: Component) -> None:
    """Raises on duplicate port names, as kfactory's ``@cell`` does (``check_ports=False`` allows them)."""
    counts: dict[str | None, int] = {}
    for port in c.ports:
        counts[port.name] = counts.get(port.name, 0) + 1
    duplicates = [(name, n) for name, n in counts.items() if n > 1]
    if duplicates:
        raise ValueError(
            "Found duplicate port names: "
            + ", ".join(f"{name}: {n}" for name, n in duplicates)
            + " If this intentional, please pass `check_ports=False` to the @cell decorator"
        )


def cell(
    _func: Callable[..., Any] | None = None,
    /,
    *,
    set_settings: bool = True,
    add_settings: bool | None = None,
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
    validate_with_pydantic: bool = False,
) -> Any:
    """Decorator to convert a function into a (cached, differentiable) Component factory."""
    drop = set(drop_params if drop_params is not None else ["self", "cls"])
    post = [post_process] if callable(post_process) else list(post_process or [])
    should_set_settings = set_settings if add_settings is None else add_settings

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        _cache: dict[Any, Any] = cache if cache is not None else {}
        _CACHES.append(_cache)
        _basename = basename or (_module_basename(func) if with_module_name else None)
        _validated_func: Callable[..., Any] | None = None

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Component:
            nonlocal _validated_func
            from gdsfactory.component import Component

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = {k: v for k, v in bound.arguments.items() if k not in drop}
            # flatten **kwargs parameters
            for name, p in sig.parameters.items():
                if p.kind is inspect.Parameter.VAR_KEYWORD and name in params:
                    params.update(params.pop(name))
            # unsnapped cells (gf.snap.SNAP_ENABLED = False) never enter the cache
            from gdsfactory import snap as _snap

            traced = has_tracers(params) or not _snap.SNAP_ENABLED
            key = None
            if not traced:
                try:
                    key = _freeze(params)
                    hash(key)
                except Exception:
                    key = None
                if key is not None and key in _cache:
                    return cast("Component", _cache[key])
                use_layout_cache = (
                    _layout_cache_enabled() if layout_cache is None else layout_cache
                )
                if set_name and use_layout_cache:
                    # like upstream (CONF.cell_layout_cache): reuse an existing cell
                    # with the same name (e.g. a function passed by its name)
                    name = get_cell_name(_basename or func.__name__, **params)
                    existing = _BY_NAME.get(name)
                    if existing is not None:
                        if key is not None:
                            _cache[key] = existing
                        return cast("Component", existing)

            _snap_token = _SNAP_PORTS_OVERRIDE.set(snap_ports)
            try:
                if validate_with_pydantic and not traced:
                    if _validated_func is None:
                        import pydantic

                        _validated_func = pydantic.validate_call(
                            config=pydantic.ConfigDict(arbitrary_types_allowed=True)
                        )(func)
                    c = _validated_func(*bound.args, **bound.kwargs)
                else:
                    c = func(*bound.args, **bound.kwargs)
            finally:
                _SNAP_PORTS_OVERRIDE.reset(_snap_token)
            if not isinstance(c, Component):
                raise TypeError(
                    f"{func.__name__} must return a Component, got {type(c)}"
                )
            if c.locked:
                # returned a cached cell from another factory: wrap a copy
                c = c.copy()

            function_name = func.__name__
            if set_name:
                if has_abstract_tracers(params):
                    c._name = f"{_basename or function_name}_traced"
                else:
                    c._name = get_cell_name(_basename or function_name, **params)
            if should_set_settings:
                c.settings = type(c.settings)(
                    dict(params)
                    if traced
                    else {k: _metadata(v) for k, v in params.items()}
                )
                c.function_name = function_name
                c.basename = _basename
                c.module = func.__module__
            if info:
                c.info.update(_copy.deepcopy(info))
            if tags:
                c.tags = list(tags)
            if lvs_equivalent_ports:
                c.lvs_equivalent_ports = lvs_equivalent_ports
            if not traced:
                if snap_ports and not c._virtual:
                    for port in c.ports:
                        if (
                            port._orientation is None
                            or manhattan_angle(port._orientation) is not None
                        ):
                            port._center = snap_dbu(port._center)
                            if port._width is not None:
                                port._width = snap_dbu(port._width)
                for pp in post:
                    pp(c)
                if check_ports:
                    _check_ports(c)
            if schematic_function is not None:
                c._schematic_function = schematic_function  # type: ignore[attr-defined]
            c.locked = True
            if key is not None:
                _cache[key] = c
                if set_name:
                    _BY_NAME.setdefault(c.name, c)
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


def cell_with_module_name(
    _func: Callable[..., Any] | None = None, /, **kwargs: Any
) -> Any:
    """Decorator like ``cell`` but with ``with_module_name=True`` by default."""
    return cell(_func, with_module_name=True, **kwargs)


def schematic_cell(_func: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
    """Builds a Component from a function returning a schematic/netlist dict."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def build(*args: Any, **kw: Any) -> Component:
            from gdsfactory.read.from_yaml import from_yaml

            schematic = func(*args, **kw)
            netlist = (
                schematic.model_dump()
                if hasattr(schematic, "model_dump")
                else schematic
            )
            return from_yaml(netlist)

        return cell(build, **{k: v for k, v in kwargs.items() if k in _CELL_KWARGS})

    if _func is not None:
        return decorator(_func)
    return decorator


_CELL_KWARGS = set(inspect.signature(cell).parameters) - {"_func"}

__all__ = [
    "cached_cells",
    "cell",
    "cell_with_module_name",
    "clean_name",
    "clear_cache",
    "factories",
    "get_cell_name",
    "schematic_cell",
    "vcell",
]
