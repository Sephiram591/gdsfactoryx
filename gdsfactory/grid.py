"""pack a list of components into a grid.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest
from typing import Any, Literal


import gdsfactory as gf
from gdsfactory._jax import to_float, xp
from gdsfactory.component import Component, ComponentReference
from gdsfactory.transform import Transform
from gdsfactory.typings import Anchor, ComponentSpec, ComponentSpecs, Float2, Spacing


def _align_offset(bbox: Any, align_x: str, align_y: str) -> tuple[Any, Any]:
    match align_x:
        case "xmin":
            x = -bbox.left
        case "xmax":
            x = -bbox.right
        case "center":
            x = -bbox.center()[0]
        case _:
            x = 0.0
    match align_y:
        case "ymin":
            y = -bbox.bottom
        case "ymax":
            y = -bbox.top
        case "center":
            y = -bbox.center()[1]
        case _:
            y = 0.0
    return x, y


def _or(prev: Any, default: Any) -> Any:
    """Python ``prev or default`` (as used by kfactory) on concrete values."""
    return prev if prev is not None and to_float(prev) != 0 else default


def _spacing_xy(spacing: Any) -> tuple[Any, Any]:
    if isinstance(spacing, tuple | list):
        return spacing[0], spacing[1]
    return spacing, spacing


def _create_insts(
    target: Component,
    kcells: Sequence[Component | None],
    rotation: float,
    mirror: bool,
) -> list[ComponentReference | None]:
    insts: list[ComponentReference | None] = []
    for kcell in kcells:
        if kcell is None:
            insts.append(None)
            continue
        inst = target.add_ref(kcell)
        inst.transform = Transform(0.0, 0.0, rotation, mirror)
        insts.append(inst)
    return insts


def _flatten_kcells(kcells: Any) -> tuple[bool, list[Any]]:
    is_1d = kcells[0] is None or isinstance(kcells[0], Component)
    if is_1d:
        return True, list(kcells)
    return False, [kcell for array in kcells for kcell in array]


def _grid(
    target: Component,
    kcells: Any,
    spacing: Any,
    shape: tuple[int, int] | None = None,
    align_x: str = "center",
    align_y: str = "center",
    rotation: float = 0,
    mirror: bool = False,
) -> list[ComponentReference]:
    """Port of kfactory.grid (same placement), differentiable w.r.t. bboxes/spacing."""
    spacing_x, spacing_y = _spacing_xy(spacing)
    is_1d, flat = _flatten_kcells(kcells)

    if shape is None:
        kcell_array = [flat] if is_1d else [list(a) for a in kcells]
        insts = [_create_insts(target, array, rotation, mirror) for array in kcell_array]
        bboxes = [
            [None if inst is None else inst.dbbox() for inst in array]
            for array in insts
        ]
        w = max(
            max(0 if bbox is None else bbox.width() + spacing_x for bbox in box_array)
            for box_array in bboxes
        )
        h = max(
            max(0 if bbox is None else bbox.height() + spacing_y for bbox in box_array)
            for box_array in bboxes
        )
        x0: Any = 0.0
        y0: Any = 0.0
        for array, bbox_array in zip(insts, bboxes, strict=False):
            y0 += h - h / 2
            for bbox, inst in zip(bbox_array, array, strict=False):
                x0 += w - w / 2
                if bbox is not None and inst is not None:
                    x, y = _align_offset(bbox, align_x, align_y)
                    inst.transform_by(Transform(x0 + x, y0 + y))
                x0 += w / 2
            y0 += h / 2
            x0 = 0.0
        return [inst for array in insts for inst in array if inst is not None]

    if len(flat) > shape[0] * shape[1]:
        raise ValueError(
            f"Shape container size {shape[0] * shape[1]=!r} must be bigger "
            f"than the number of kcells {len(flat)}"
        )
    x0 = 0.0
    y0 = 0.0
    _insts = _create_insts(target, flat, rotation, mirror)
    shape_bboxes = [None if inst is None else inst.dbbox() for inst in _insts]
    w = max(0 if box is None else box.width() for box in shape_bboxes) + spacing_x
    h = max(0 if box is None else box.height() for box in shape_bboxes) + spacing_y
    for i, (inst, bbox) in enumerate(zip(_insts, shape_bboxes, strict=False)):
        i_x = i % shape[1]
        if i_x == 0:
            y0 += h - h / 2
            x0 = 0.0
        else:
            x0 += w - w / 2
        if bbox is not None and inst is not None:
            x, y = _align_offset(bbox, align_x, align_y)
            inst.transform_by(Transform(x0 + x, y0 + y))
        if i_x == shape[1] - 1:
            y0 += h / 2
            x0 = 0.0
        else:
            x0 += w / 2
    return [inst for inst in _insts if inst is not None]


def _flexgrid(
    target: Component,
    kcells: Any,
    spacing: Any,
    shape: tuple[int, int] | None = None,
    align_x: str = "center",
    align_y: str = "center",
    rotation: float = 0,
    mirror: bool = False,
) -> list[ComponentReference]:
    """Port of kfactory.flexgrid (same placement), differentiable w.r.t. bboxes/spacing."""
    spacing_x, spacing_y = _spacing_xy(spacing)
    is_1d, flat = _flatten_kcells(kcells)
    xmin: dict[int, Any] = {}
    ymin: dict[int, Any] = {}
    ymax: dict[int, Any] = {}
    xmax: dict[int, Any] = {}

    def _align_and_measure(inst: ComponentReference, i_x: int, i_y: int) -> None:
        x, y = _align_offset(inst.dbbox(), align_x, align_y)
        inst.transform_by(Transform(x, y))
        bbox_ = inst.dbbox()
        xmin[i_x] = xp.minimum(_or(xmin.get(i_x), bbox_.left), bbox_.left - spacing_x)
        xmax[i_x] = xp.maximum(_or(xmax.get(i_x), bbox_.right), bbox_.right)
        ymin[i_y] = xp.minimum(
            _or(ymin.get(i_y), bbox_.bottom), bbox_.bottom - spacing_y
        )
        ymax[i_y] = xp.maximum(_or(ymax.get(i_y), bbox_.top), bbox_.top)

    x0: Any = 0.0
    y0: Any = 0.0
    if shape is None:
        kcell_array = [flat] if is_1d else [list(a) for a in kcells]
        insts = [_create_insts(target, array, rotation, mirror) for array in kcell_array]
        for i_y, array in enumerate(insts):
            for i_x, inst in enumerate(array):
                if inst is not None:
                    _align_and_measure(inst, i_x, i_y)
        for i_y, array in enumerate(insts):
            y0 -= ymin.get(i_y, 0)
            for i_x, inst in enumerate(array):
                x0 -= xmin.get(i_x, 0)
                if inst is not None:
                    inst.transform_by(Transform(x0, y0))
                x0 += xmax.get(i_x, 0)
            y0 += ymax.get(i_y, 0)
            x0 = 0.0
        return [inst for array in insts for inst in array if inst is not None]

    if len(flat) > shape[0] * shape[1]:
        raise ValueError(
            f"Shape container size {shape[0] * shape[1]=} must be bigger "
            f"than the number of kcells {len(flat)}"
        )
    _insts = _create_insts(target, flat, rotation, mirror)
    for i, inst in enumerate(_insts):
        if inst is not None:
            _align_and_measure(inst, i % shape[1], i // shape[1])
    for i, inst in enumerate(_insts):
        i_x = i % shape[1]
        i_y = i // shape[1]
        if i_x == 0:
            y0 -= ymin.get(i_y, 0)
            x0 = 0.0
        else:
            x0 -= xmin.get(i_x, 0)
        if inst is not None:
            inst.transform_by(Transform(x0, y0))
        if i_x == shape[1] - 1:
            y0 += ymax.get(i_y, 0)
            x0 = 0.0
        else:
            x0 += xmax.get(i_x, 0)
    return [inst for inst in _insts if inst is not None]


def grid(
    components: ComponentSpecs = ("rectangle", "triangle"),
    spacing: Spacing | float = (5.0, 5.0),
    shape: tuple[int, int] | None = None,
    align_x: Literal["origin", "xmin", "xmax", "center"] = "center",
    align_y: Literal["origin", "ymin", "ymax", "center"] = "center",
    rotation: int = 0,
    mirror: bool = False,
    flex: bool = False,
) -> Component:
    """Returns Component with a 1D or 2D grid of components.

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D).
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width or a single float.
        shape: x, y shape of the grid (see np.reshape). \
                If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: x alignment along (origin, xmin, xmax, center).
        align_y: y alignment along (origin, ymin, ymax, center).
        rotation: for each component in degrees.
        mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.
        flex: use minimal row height and column width where possible.

    Returns:
        Component containing components grid.

    Example:
        ```python
        import gdsfactory as gf

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.grid(
        components,
        shape=(1, len(components)),
        rotation=0,
        mirror=False,
        spacing=(100, 100),
        )
        c.plot()
        ```
    """
    c = gf.Component()
    grid_func = _flexgrid if flex else _grid
    instances = grid_func(
        c,
        kcells=[gf.get_component(component) for component in components],
        shape=shape,
        spacing=spacing,
        align_x=align_x,
        align_y=align_y,
        rotation=rotation,
        mirror=mirror,
    )
    for i, instance in enumerate(instances):
        c.add_ports(instance.ports, prefix=f"{i}_")
    return c


def grid_with_text(
    components: Sequence[ComponentSpec] = ("rectangle", "triangle"),
    text_prefix: str = "",
    text_offsets: Sequence[Float2] | None = None,
    text_anchors: Sequence[Anchor] | None = None,
    text_mirror: bool = False,
    text_rotation: int = 0,
    text: ComponentSpec | None = "text_rectangular",
    spacing: Spacing | float = (5.0, 5.0),
    shape: tuple[int, int] | None = None,
    align_x: Literal["origin", "xmin", "xmax", "center"] = "center",
    align_y: Literal["origin", "ymin", "ymax", "center"] = "center",
    rotation: int = 0,
    mirror: bool = False,
    labels: Sequence[str] | None = None,
    flex: bool = False,
) -> Component:
    """Returns Component with 1D or 2D grid of components with text labels.

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D).
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_offsets: relative to component anchor. Defaults to center.
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        text_mirror: if True mirrors text.
        text_rotation: Optional text rotation.
        text: function to add text labels.
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width.
        shape: x, y shape of the grid (see np.reshape).
        align_x: x alignment along (origin, xmin, xmax, center).
        align_y: y alignment along (origin, ymin, ymax, center).
        rotation: for each component in degrees.
        mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.
        labels: list of labels for each component.
        flex: use minimal row height and column width where possible.


    Example:
        ```python
        import gdsfactory as gf

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.grid_with_text(
        components,
        shape=(1, len(components)),
        rotation=0,
        mirror=False,
        spacing=(100, 100),
        text_offsets=((0, 100), (0, -100)),
        text_anchors=("nc", "sc"),
        )
        c.plot()
        ```
    """
    component_list = [gf.get_component(component) for component in components]
    text_offsets = text_offsets or ((0, 0),)
    text_anchors = text_anchors or ("center",)
    labels_not_none: list[str | None] = (
        list(labels) if labels else [None] * len(component_list)
    )

    if len(labels_not_none) != len(component_list):
        raise ValueError(
            f"Number of labels {len(labels_not_none)} must match number of components {len(component_list)}"
        )

    c = gf.Component()
    grid_func = _flexgrid if flex else _grid
    instances = grid_func(
        c,
        kcells=component_list,
        shape=shape,
        spacing=spacing,
        align_x=align_x,
        align_y=align_y,
        rotation=rotation,
        mirror=mirror,
    )
    for i, instance in enumerate(instances):
        c.add_ports(instance.ports, prefix=f"{i}_")
        text_string = labels_not_none[i] or f"{text_prefix}_{i}"

        if text:
            for text_offset, text_anchor in zip_longest(text_offsets, text_anchors):
                t = c << gf.get_component(text, text=text_string)
                if text_mirror:
                    t.dmirror()
                if text_rotation:
                    t.rotate(text_rotation)
                size_info = instance.dsize_info
                text_offset = text_offset or (0, 0)
                text_anchor = text_anchor or "center"
                o = xp.asarray(text_offset, dtype=xp.float64)
                d = xp.asarray(getattr(size_info, text_anchor))
                od = o + d
                t.move((od[0], od[1]))
    return c
