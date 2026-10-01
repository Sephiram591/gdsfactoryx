from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from gdsfactory import klayout_bridge as kb
from gdsfactory.component import Component, ComponentReference, boolean_operations

if TYPE_CHECKING:
    from gdsfactory.typings import ComponentOrReference, LayerSpec


def boolean(
    A: ComponentOrReference,
    B: ComponentOrReference,
    operation: Literal["or", "|", "not", "-", "^", "xor", "&", "and", "A-B"],
    layer: LayerSpec,
    layer1: LayerSpec | None = None,
    layer2: LayerSpec | None = None,
) -> Component:
    """Performs boolean operations between 2 Component or Instance objects.

    The `operation` parameter specifies the type of boolean operation to perform.
    Supported operations include {'not', 'and', 'or', 'xor', '-', '&', '|', '^'}:

    - `'|'` is equivalent to `'or'`
    - `'-'` is equivalent to `'not'`
    - `'&'` is equivalent to `'and'`
    - `'^'` is equivalent to `'xor'`

    Args:
        A: Component(/Reference) or list of Component(/References).
        B: Component(/Reference) or list of Component(/References).
        operation: {'not', 'and', 'or', 'xor', '-', '&', '|', '^'}.
        layer: Specific layer to put polygon geometry on.
        layer1: Specific layer to get polygons.
        layer2: Specific layer to get polygons.

    Returns: Component with polygon(s) of the boolean operations between
      the 2 input Components performed.

    Example:
        ```python
        import gdsfactory as gf

        c = gf.Component()
        c1 = c << gf.components.circle(radius=10)
        c2 = c << gf.components.circle(radius=9)
        c2.movex(5)

        c = gf.boolean(c1, c2, operation="xor")
        c.plot()
        ```
    """
    from gdsfactory import get_layer

    if operation not in boolean_operations:
        raise ValueError(
            f"Boolean operation {operation} not supported. Choose from {list(boolean_operations.keys())}"
        )

    c = Component()
    layer1 = layer1 or layer
    layer2 = layer2 or layer

    layer_index1 = get_layer(layer1)
    layer_index2 = get_layer(layer2)
    layer_index = get_layer(layer)

    ar = kb.arrays_to_region(_get_polygons(A, layer_index1))
    br = kb.arrays_to_region(_get_polygons(B, layer_index2))
    region = boolean_operations[operation](ar, br)
    for points in kb.region_to_arrays(region):
        c.add_polygon(points, layer=layer_index)

    return c


def _get_polygons(
    obj: ComponentOrReference, layer_index: int
) -> list[npt.NDArray[np.floating[Any]]]:
    """Returns the flattened polygons of a Component or reference on a layer."""
    if isinstance(obj, ComponentReference):
        polys = obj.get_polygons_points(layer=layer_index)
    else:
        polys = obj.get_polygons_points(layers=[layer_index])
    return list(polys.get(int(layer_index), []))
