"""gdsfactoryx: a JAX-differentiable wrapper around gdsfactory.

`gdsfactoryx` mirrors the `gdsfactory` namespace. Every function is wrapped so
that it accepts `jax.Array` arguments:

    import jax, jax.numpy as jnp
    import gdsfactoryx as gfx

    gfx.components.straight(length=10)             # -> gf.Component (unchanged)
    geom = gfx.components.straight(length=jnp.array(10.0))  # -> gfx.Geometry

    def loss(length):
        return gfx.components.straight(length=length).area("WG")

    jax.grad(loss)(jnp.array(10.0))                # -> 0.5 (the width)

Classes, constants and non-function attributes (Component, Port, LAYER, ...)
are re-exported unchanged, so `gfx` can be used as a drop-in for `gf`.
"""

from __future__ import annotations

from typing import Any

import gdsfactory as _gf

from gdsfactoryx import (
    components,
    containers,
    cross_section,
    functions,
    path,
    raster,
    routing,
    samples,
)
from gdsfactoryx._extract import TopologyWarning
from gdsfactoryx._jaxify import (
    JaxifiedFunction,
    Settings,
    TemplateError,
    cell,
    jaxify,
    settings,
)
from gdsfactoryx._pdk import GradientAccuracyWarning, activate_pdk
from gdsfactoryx._wrap import WrappedModule, wrap
from gdsfactoryx.geometry import Geometry, PortGeometry, layer_key
from gdsfactoryx.raster import coverage, grid_edges, rasterize

__version__ = "0.1.0"
gdsfactory_version = _gf.__version__

__all__ = [
    "GradientAccuracyWarning",
    "Geometry",
    "JaxifiedFunction",
    "PortGeometry",
    "Settings",
    "TemplateError",
    "TopologyWarning",
    "WrappedModule",
    "activate_pdk",
    "cell",
    "components",
    "containers",
    "coverage",
    "cross_section",
    "functions",
    "grid_edges",
    "jaxify",
    "layer_key",
    "path",
    "raster",
    "rasterize",
    "routing",
    "samples",
    "settings",
    "wrap",
]


def __getattr__(name: str) -> Any:
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    try:
        attr = getattr(_gf, name)
    except AttributeError:
        raise AttributeError(f"module 'gdsfactoryx' has no attribute {name!r}") from None
    return wrap(attr)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_gf)))
