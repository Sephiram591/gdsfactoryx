"""Mirror gdsfactory modules, wrapping their functions with `jaxify`."""

from __future__ import annotations

import importlib
import inspect
import types
from typing import Any

from gdsfactoryx._jaxify import JaxifiedFunction

_modules: dict[str, WrappedModule] = {}
_functions: dict[int, JaxifiedFunction] = {}


def _is_gdsfactory_module(obj: Any) -> bool:
    return isinstance(obj, types.ModuleType) and (
        obj.__name__ == "gdsfactory" or obj.__name__.startswith("gdsfactory.")
    )


def wrap(obj: Any) -> Any:
    """Wraps a gdsfactory attribute.

    - plain functions and functools.partials -> JaxifiedFunction (cached)
    - gdsfactory submodules -> WrappedModule
    - classes, constants, and everything else -> returned unchanged
    """
    if isinstance(obj, JaxifiedFunction) or inspect.isclass(obj):
        return obj
    if _is_gdsfactory_module(obj):
        if obj.__name__ not in _modules:
            _modules[obj.__name__] = WrappedModule(obj)
        return _modules[obj.__name__]
    if callable(obj) and not isinstance(obj, types.BuiltinFunctionType):
        key = id(obj)
        cached = _functions.get(key)
        if cached is None or cached.func is not obj:
            cached = JaxifiedFunction(obj)
            _functions[key] = cached
        return cached
    return obj


class WrappedModule(types.ModuleType):
    """A view of a gdsfactory module whose functions accept JAX arrays."""

    def __init__(self, module: types.ModuleType) -> None:
        name = "gdsfactoryx" + module.__name__[len("gdsfactory") :]
        super().__init__(name, module.__doc__)
        self.__wrapped_module__ = module

    def __getattr__(self, name: str) -> Any:
        module = self.__wrapped_module__
        try:
            attr = getattr(module, name)
        except AttributeError:
            # Submodules that have not been imported yet.
            try:
                attr = importlib.import_module(f"{module.__name__}.{name}")
            except ImportError:
                raise AttributeError(
                    f"module {self.__name__!r} has no attribute {name!r}"
                ) from None
        return wrap(attr)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self.__wrapped_module__)))

    def __repr__(self) -> str:
        return f"<gdsfactoryx wrapper of {self.__wrapped_module__.__name__!r}>"


def module_getattr(module_name: str) -> Any:
    """Returns a PEP 562 `__getattr__` that forwards to a wrapped gdsfactory module."""

    def __getattr__(name: str) -> Any:
        module = importlib.import_module(module_name)
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(wrap(module), name)

    return __getattr__


def module_dir(module_name: str) -> Any:
    def __dir__() -> list[str]:
        return sorted(dir(importlib.import_module(module_name)))

    return __dir__
