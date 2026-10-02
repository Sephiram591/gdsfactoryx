"""Differentiable Component: a canvas of polygons, references, ports and labels.

All coordinates are float64 ``jax.numpy`` arrays (um). There is no grid
snapping inside the geometry kernel; coordinates are rounded to the database
unit only when a Component is exported (``write_gds``, ``show``, ``plot``,
``to_kfactory``). Any scalar computed from a Component (port positions, bounding
boxes, areas, polygon vertices, rasterized masks, ...) can be differentiated
with :func:`jax.grad` with respect to the float parameters that produced it.
"""

from __future__ import annotations

import copy as _copy
import itertools
import weakref
import pathlib
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self, TypeAlias

import numpy as np
import numpy.typing as npt

from gdsfactory._jax import (
    Array,
    asarray,
    has_tracers,
    is_tracer,
    jnp,
    xp,
    maybe_float,
    points_array,
    snap_dbu,
    stop_gradient,
    to_float,
    to_numpy,
)
from gdsfactory._ports import Pin, Pins, Port, PortInfo, Ports
from gdsfactory.config import CONF, GDSDIR_TEMP
from gdsfactory.serialization import clean_value_json
from gdsfactory.transform import Transform, manhattan_angle

if TYPE_CHECKING:
    import klayout.db as kdb
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gdsfactory.cross_section import CrossSection, CrossSectionSpec
    from gdsfactory.typings import Layer, LayerSpec, LayerSpecs, PathType


from kfactory.exceptions import LockedError as _KFLockedError  # noqa: E402


class LockedError(_KFLockedError):
    """Raised when modifying a locked (cached) Component."""

    def __init__(self, component: Any) -> None:
        AttributeError.__init__(
            self,
            f"Component {getattr(component, 'name', component)!r} is locked "
            "(it was returned by a cell function and may be cached). "
            "Use `component.copy()` to get a modifiable copy."
        )


class AddPortError(ValueError):
    """Error raised when adding a port fails."""


class Info(dict[str, Any]):
    """Attribute-accessible dict used for Component.info and Component.settings.

    Like kfactory's pydantic settings, iterating yields ``(name, value)`` pairs.
    """

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        return iter(self.items())

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def model_dump(self, exclude_none: bool = False, **kwargs: Any) -> dict[str, Any]:
        if exclude_none:
            return {k: v for k, v in self.items() if v is not None}
        return dict(self)

    def model_copy(self, deep: bool = False, update: dict[str, Any] | None = None) -> Info:
        new = Info(_copy.deepcopy(dict(self)) if deep else dict(self))
        if update:
            new.update(update)
        return new

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        dict.update(self, *args, **kwargs)


class Label:
    __slots__ = ("layer", "position", "text")

    def __init__(self, text: str, position: Any, layer: Any) -> None:
        self.text = text
        self.position = asarray(position)
        self.layer = layer

    def transformed(self, t: Transform) -> Label:
        return Label(self.text, t.apply(self.position), self.layer)

    @property
    def x(self) -> Any:
        return self.position[0]

    @property
    def y(self) -> Any:
        return self.position[1]

    @property
    def string(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Label({self.text!r}, ({to_float(self.x)}, {to_float(self.y)}), {self.layer})"


class Point2(tuple):  # type: ignore[type-arg]
    """(x, y) tuple with ``.x`` / ``.y`` accessors (like kdb.DPoint)."""

    __slots__ = ()

    def __new__(cls, x: Any, y: Any) -> Point2:
        return super().__new__(cls, (x, y))

    @property
    def x(self) -> Any:
        return self[0]

    @property
    def y(self) -> Any:
        return self[1]


class Box:
    """Differentiable axis-aligned box with KLayout DBox-like accessors."""

    __slots__ = ("bottom", "left", "right", "top")

    def __init__(self, left: Any, bottom: Any, right: Any, top: Any) -> None:
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

    @classmethod
    def empty_box(cls) -> Box:
        return cls(0.0, 0.0, 0.0, 0.0)

    def empty(self) -> bool:
        return to_float(self.right) < to_float(self.left)

    def width(self) -> Any:
        return self.right - self.left

    def height(self) -> Any:
        return self.top - self.bottom

    def center(self) -> Point2:
        # same floating point formula as KLayout's DBox::center
        return Point2(
            self.left + (self.right - self.left) / 2,
            self.bottom + (self.top - self.bottom) / 2,
        )

    @property
    def p1(self) -> Point2:
        return Point2(self.left, self.bottom)

    @property
    def p2(self) -> Point2:
        return Point2(self.right, self.top)

    def to_array(self) -> Array:
        return xp.stack(
            [
                xp.stack([asarray(self.left), asarray(self.bottom)]),
                xp.stack([asarray(self.right), asarray(self.top)]),
            ]
        )

    def inside(self, other: Box) -> bool:
        return (
            to_float(self.left) >= to_float(other.left)
            and to_float(self.right) <= to_float(other.right)
            and to_float(self.bottom) >= to_float(other.bottom)
            and to_float(self.top) <= to_float(other.top)
        )

    def enlarged(self, dx: Any, dy: Any | None = None) -> Box:
        dy = dx if dy is None else dy
        return Box(self.left - dx, self.bottom - dy, self.right + dx, self.top + dy)

    def __eq__(self, other: object) -> bool:
        if not all(hasattr(other, a) for a in ("left", "bottom", "right", "top")):
            return NotImplemented
        mine = (self.left, self.bottom, self.right, self.top)
        theirs = (other.left, other.bottom, other.right, other.top)  # type: ignore[attr-defined]
        return all(abs(to_float(a) - to_float(b)) < 1e-9 for a, b in zip(mine, theirs, strict=True))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Box({to_float(self.left):.6g}, {to_float(self.bottom):.6g}, "
            f"{to_float(self.right):.6g}, {to_float(self.top):.6g})"
        )


def _box_from_points(pts: Array) -> Box:
    mn = xp.min(pts, axis=0)
    mx = xp.max(pts, axis=0)
    return Box(mn[0], mn[1], mx[0], mx[1])


def _layer_key(layer: Any) -> int:
    from gdsfactory.pdk import get_layer

    return int(get_layer(layer))


# --------------------------------------------------------------------------
# geometry mixin shared by Component and ComponentReference
# --------------------------------------------------------------------------
class _BBoxMixin:
    def dbbox(self, layer: Any = None) -> Box:  # pragma: no cover - abstract
        raise NotImplementedError

    def bbox(self, layer: Any = None) -> Box:
        return self.dbbox(layer)

    def bbox_np(self) -> npt.NDArray[np.float64]:
        b = self.dbbox()
        return np.array(
            [[to_float(b.left), to_float(b.bottom)], [to_float(b.right), to_float(b.top)]]
        )

    def bbox_array(self) -> Array:
        return self.dbbox().to_array()

    @property
    def xmin(self) -> Any:
        return self.dbbox().left

    @property
    def xmax(self) -> Any:
        return self.dbbox().right

    @property
    def ymin(self) -> Any:
        return self.dbbox().bottom

    @property
    def ymax(self) -> Any:
        return self.dbbox().top

    @property
    def x(self) -> Any:
        b = self.dbbox()
        return b.left + (b.right - b.left) / 2  # KLayout's DBox::center formula

    @property
    def y(self) -> Any:
        b = self.dbbox()
        return b.bottom + (b.top - b.bottom) / 2

    @property
    def center(self) -> tuple[Any, Any]:
        c = self.dbbox().center()
        return (c[0], c[1])

    @property
    def xsize(self) -> Any:
        b = self.dbbox()
        return b.right - b.left

    @property
    def ysize(self) -> Any:
        b = self.dbbox()
        return b.top - b.bottom

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(self.dbbox())

    dsize_info = size_info
    dxmin = xmin
    dxmax = xmax
    dymin = ymin
    dymax = ymax
    dx = x
    dy = y
    dcenter = center
    dxsize = xsize
    dysize = ysize


class SizeInfo:
    def __init__(self, box: Box) -> None:
        self._b = box

    def _bf(self) -> tuple[float, float, float, float]:
        b = self._b
        return (to_float(b.left), to_float(b.bottom), to_float(b.right), to_float(b.top))

    @property
    def west(self) -> Any:
        return self._b.left

    @property
    def east(self) -> Any:
        return self._b.right

    @property
    def south(self) -> Any:
        return self._b.bottom

    @property
    def north(self) -> Any:
        return self._b.top

    @property
    def width(self) -> Any:
        return self._b.width()

    @property
    def height(self) -> Any:
        return self._b.height()

    @property
    def center(self) -> tuple[Any, Any]:
        c = self._b.center()
        return (c[0], c[1])

    @property
    def sw(self) -> Array:
        return xp.stack([asarray(self.west), asarray(self.south)])

    @property
    def nw(self) -> Array:
        return xp.stack([asarray(self.west), asarray(self.north)])

    @property
    def se(self) -> Array:
        return xp.stack([asarray(self.east), asarray(self.south)])

    @property
    def ne(self) -> Array:
        return xp.stack([asarray(self.east), asarray(self.north)])

    @property
    def cw(self) -> Array:
        return xp.stack([asarray(self.west), self.center[1]])

    @property
    def ce(self) -> Array:
        return xp.stack([asarray(self.east), self.center[1]])

    @property
    def sc(self) -> Array:
        return xp.stack([self.center[0], asarray(self.south)])

    @property
    def nc(self) -> Array:
        return xp.stack([self.center[0], asarray(self.north)])

    @property
    def cc(self) -> Array:
        return self.center


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------
class ReferencePorts(Ports):
    """Ports of a reference, computed from the referenced cell's ports."""

    def __init__(self, ref: ComponentReference) -> None:
        self._ref = ref

    @property
    def _ports(self) -> list[Port]:  # type: ignore[override]
        return self._ref._get_ports()

    def __getitem__(self, key: Any) -> Port:
        if isinstance(key, tuple):
            name, *idx = key
            ia = idx[0] if idx else 0
            ib = idx[1] if len(idx) > 1 else 0
            p = self._ref.cell.ports[name]
            return p.copy(self._ref._array_transform(ia, ib))
        return super().__getitem__(key)

    def copy(self, trans: Transform | None = None) -> Ports:
        return Ports(p.copy(trans) for p in self._ports)

    def each_by_array_coord(self) -> Iterator[tuple[int, int, Port]]:
        for ia in range(self._ref.na):
            for ib in range(self._ref.nb):
                for p in self._ref.cell.ports:
                    yield ia, ib, p.copy(self._ref._array_transform(ia, ib))


class ComponentReference(_BBoxMixin):
    """Placement of a Component inside another Component (with optional array)."""

    def __init__(
        self,
        cell: Component,
        transform: Transform | None = None,
        name: str | None = None,
        parent: Component | None = None,
        na: int = 1,
        nb: int = 1,
        a: Any = (0.0, 0.0),
        b: Any = (0.0, 0.0),
    ) -> None:
        self.cell = cell
        self.transform = transform or Transform()
        self._name = name
        self.parent_cell = parent
        self.na = int(na)
        self.nb = int(nb)
        self.a = asarray(a)
        self.b = asarray(b)
        self._ports = ReferencePorts(self)
        # all-angle ("virtual") references: upstream flattens them into polygons,
        # so their bbox is tight; regular references use the transformed cell bbox
        self.virtual = False

    # ------------------------------------------------------------ naming
    @property
    def name(self) -> str:
        """Instance name (kfactory default: cell name + position in dbu + angle)."""
        if self._name is not None:
            return self._name
        t = self.transform
        x = round(to_float(t.x) * 1000)
        y = round(to_float(t.y) * 1000)
        name = f"{self.cell.name}_{x}_{y}"
        # KLayout's own angle (atan2 of the stored sin/cos), like kfactory
        angle = t.to_klayout().angle
        if angle != 0:
            if float(angle).is_integer():
                name += f"_A{int(angle)}"
            else:
                name += f"_A{str(float(angle)).replace('.', 'p')}"
        if t.mirror:
            name += "_M"
        return name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def is_named(self) -> bool:
        return self._name is not None

    @property
    def parent_component(self) -> Component | None:
        return self.parent_cell

    @property
    def component(self) -> Component:
        return self.cell

    @property
    def ports(self) -> ReferencePorts:
        return self._ports

    def __getitem__(self, key: Any) -> Port:
        return self._ports[key]

    @property
    def pins(self) -> Pins:
        return Pins(pin.copy(self.transform) for pin in self.cell.pins)

    def __contains__(self, key: str) -> bool:
        return key in self._ports

    def is_regular_array(self) -> bool:
        return self.na > 1 or self.nb > 1

    @property
    def columns(self) -> int:
        return self.na

    @property
    def rows(self) -> int:
        return self.nb

    # ---------------------------------------------------------- transforms
    @property
    def transform(self) -> Transform:
        return self._transform

    @transform.setter
    def transform(self, t: Transform) -> None:
        # like KLayout: instance displacements are stored on the dbu grid, except
        # for all-angle (virtual) references, which keep float positions
        if getattr(self, "virtual", False):
            self._transform = t
            return
        self._transform = Transform(
            snap_dbu(t.x), snap_dbu(t.y), t.rotation, t.mirror, t.magnification
        )

    def _array_transform(self, ia: int = 0, ib: int = 0) -> Transform:
        if ia == 0 and ib == 0:
            return self.transform
        off = ia * self.a + ib * self.b
        t = self.transform.copy()
        t.x = t.x + off[0]
        t.y = t.y + off[1]
        return t

    def array_transform(self, ia: int = 0, ib: int = 0) -> Transform:
        """Transform of array element (ia, ib)."""
        return self._array_transform(ia, ib)

    def array_transforms(self) -> list[Transform]:
        return [
            self._array_transform(ia, ib) for ia in range(self.na) for ib in range(self.nb)
        ]

    def _get_ports(self) -> list[Port]:
        t = self.transform
        if self.virtual:
            return [p.copy(t, on_grid=False) for p in self.cell.ports]
        return [p.copy(t) for p in self.cell.ports]

    @property
    def dcplx_trans(self) -> Any:
        return self.transform.to_klayout()

    @dcplx_trans.setter
    def dcplx_trans(self, value: Any) -> None:
        self.transform = Transform.from_klayout(value) if not isinstance(value, Transform) else value

    @property
    def trans(self) -> Any:
        """Integer (dbu) simple transformation, like kfactory's ``Instance.trans``."""
        return self.transform.to_klayout().s_trans().to_itype(1e-3)

    @trans.setter
    def trans(self, value: Any) -> None:
        import klayout.db as kdb

        if isinstance(value, kdb.Trans):
            value = kdb.DCplxTrans(value.to_dtype(1e-3))
        elif isinstance(value, kdb.DTrans):
            value = kdb.DCplxTrans(value)
        self.dcplx_trans = value

    @property
    def dtrans(self) -> Any:
        """Simple transformation in um (kdb.DTrans)."""
        return self.transform.to_klayout().s_trans()

    @dtrans.setter
    def dtrans(self, value: Any) -> None:
        self.trans = value

    @property
    def magnification(self) -> Any:
        return self.transform.magnification

    @magnification.setter
    def magnification(self, value: Any) -> None:
        self.transform.magnification = value

    @property
    def rotation(self) -> Any:
        return self.transform.rotation

    @property
    def is_mirrored(self) -> bool:
        return self.transform.mirror

    def transform_by(self, t: Transform) -> Self:
        """Applies t after the current transform (in the parent's frame)."""
        self.transform = _kl_value(
            t * self.transform, t.to_klayout() * self.transform.to_klayout()
        )
        self.a = t.apply_vector(self.a)
        self.b = t.apply_vector(self.b)
        return self

    def move(self, *args: Any) -> Self:
        """Moves the reference: move((dx, dy)), move(dx, dy), move(origin, destination)."""
        from gdsfactory._ports import _move_args

        dx, dy = _move_args(args)
        t = self.transform.copy()
        t.x = t.x + dx
        t.y = t.y + dy
        self.transform = t
        return self

    dmove = move

    def movex(self, *args: Any) -> Self:
        if len(args) == 1:
            return self.move(args[0], 0.0)
        origin, destination = args
        return self.move(destination - origin, 0.0)

    def movey(self, *args: Any) -> Self:
        if len(args) == 1:
            return self.move(0.0, args[0])
        origin, destination = args
        return self.move(0.0, destination - origin)

    dmovex = movex
    dmovey = movey

    def rotate(self, angle: Any, center: Any = None) -> Self:
        """Rotates (degrees, counter-clockwise) around center (default origin)."""
        if isinstance(center, Port):
            center = center.center_array
        elif isinstance(center, str):
            center = self.ports[center].center
        c = asarray((0.0, 0.0) if center is None else center)
        t = Transform(c[0], c[1]) * Transform(0.0, 0.0, angle) * Transform(-c[0], -c[1])
        return self.transform_by(t)

    drotate = rotate

    def mirror(self, p1: Any = (0.0, 1.0), p2: Any = (0.0, 0.0)) -> Self:
        """Mirrors across the line through p1 and p2."""
        if isinstance(p1, Port):
            p1 = p1.center_array
        if isinstance(p2, Port):
            p2 = p2.center_array
        p1 = asarray(p1)
        p2 = asarray(p2)
        d = p2 - p1
        theta = xp.degrees(xp.arctan2(d[1], d[0]))
        # mirror across line at angle theta through p1: T(p1) R(theta) M R(-theta) T(-p1)
        t = (
            Transform(p1[0], p1[1])
            * Transform(0.0, 0.0, 2 * theta, mirror=True)
            * Transform(-p1[0], -p1[1])
        )
        return self.transform_by(t)

    dmirror = mirror

    def mirror_x(self, x: Any = 0.0) -> Self:
        """Mirrors across the vertical line at x (flips x)."""
        return self.mirror((x, 0.0), (x, 1.0))

    def mirror_y(self, y: Any = 0.0) -> Self:
        """Mirrors across the horizontal line at y (flips y)."""
        t = Transform(0.0, y) * Transform(0.0, 0.0, 0.0, mirror=True) * Transform(0.0, -asarray(y))
        return self.transform_by(t)

    dmirror_x = mirror_x
    dmirror_y = mirror_y

    def connect(
        self,
        port: str | Port,
        other: Port | ComponentReference,
        other_port_name: str | None = None,
        *,
        mirror: bool = False,
        allow_width_mismatch: bool = False,
        allow_layer_mismatch: bool = False,
        allow_type_mismatch: bool = False,
        use_mirror: bool = False,
        use_angle: bool = True,
    ) -> Self:
        """Places the reference so that ``port`` faces ``other`` (aligned, opposite)."""
        if isinstance(other, ComponentReference):
            if other_port_name is None:
                raise ValueError("other_port_name is required when connecting to a reference")
            op = other.ports[other_port_name]
        elif isinstance(other, Port):
            op = other
        else:
            raise TypeError(f"Cannot connect to {type(other)}")

        if isinstance(port, Port):
            # a port of this reference (possibly of an array element): map it back
            # into the cell frame (kfactory semantics)
            local = port.copy(self.transform.inverted())
        elif isinstance(port, tuple):
            local = self.ports[port].copy(self.transform.inverted())
        else:
            local = self.cell.ports[port]

        if not allow_width_mismatch and round(to_float(local.width) * 1000) != round(
            to_float(op.width) * 1000
        ):
            raise PortWidthMismatchError(
                f"Width mismatch {self.cell.name}:{local.name} ({to_float(local.width)}) "
                f"!= {op.name} ({to_float(op.width)})"
            )
        if not allow_layer_mismatch and local.layer != op.layer:
            raise PortLayerMismatchError(
                f"Layer mismatch {self.cell.name}:{local.name} ({local.layer}) "
                f"!= {op.name} ({op.layer})"
            )
        if not allow_type_mismatch and local.port_type != op.port_type:
            raise PortTypeMismatchError(
                f"Type mismatch {self.cell.name}:{local.name} ({local.port_type}) "
                f"!= {op.name} ({op.port_type})"
            )

        # kfactory semantics: T = op_trans * conn * p_trans^-1 (magnification reset).
        # With use_mirror=False the other port's mirror is ignored and the
        # current mirror of this reference is kept (xor'ed with `mirror`).
        if not use_angle:
            self.transform = Transform(op.x - local.x, op.y - local.y)
            return self
        op_mirror = op.mirror if use_mirror else False
        conn_mirror = mirror if use_mirror else (mirror != self.transform.mirror)
        op_t = Transform(op.x, op.y, op.orientation, op_mirror, 1.0)
        conn = Transform(0.0, 0.0, 180.0, conn_mirror, 1.0)
        p_t = Transform(local.x, local.y, local.orientation, local.mirror, 1.0)
        self.transform = _kl_value(
            op_t * conn * p_t.inverted(),
            op_t.to_klayout() * conn.to_klayout() * p_t.to_klayout().inverted(),
        )
        return self

    # -------------------------------------------------------------- geometry
    def _bbox_points(self, t: Transform, layer: Any = None) -> Array | None:
        if self.virtual:
            return self.cell._transformed_bbox_points(t, layer)
        return self.cell._transformed_bbox_corners(t, layer)

    def dbbox(self, layer: Any = None) -> Box:
        """Bounding box of the reference (cell bbox corners transformed, like kfactory)."""
        boxes = []
        for t in self.array_transforms():
            pts = self._bbox_points(t, layer)
            if pts is not None:
                boxes.append(pts)
        if not boxes:
            return Box.empty_box()
        return _box_from_points(xp.concatenate(boxes))

    def ibbox(self, layer: Any = None) -> Box:
        return self.dbbox(layer)

    def _set_position(self, attr: str, value: Any) -> None:
        current = getattr(self, attr)
        if isinstance(value, Port):
            value = value.x if attr in {"x", "xmin", "xmax"} else value.y
        delta = value - current
        if attr in {"x", "xmin", "xmax"}:
            self.move(delta, 0.0)
        else:
            self.move(0.0, delta)

    @_BBoxMixin.xmin.setter  # type: ignore[attr-defined, untyped-decorator]
    def xmin(self, value: Any) -> None:
        self._set_position("xmin", value)

    @_BBoxMixin.xmax.setter  # type: ignore[attr-defined, untyped-decorator]
    def xmax(self, value: Any) -> None:
        self._set_position("xmax", value)

    @_BBoxMixin.ymin.setter  # type: ignore[attr-defined, untyped-decorator]
    def ymin(self, value: Any) -> None:
        self._set_position("ymin", value)

    @_BBoxMixin.ymax.setter  # type: ignore[attr-defined, untyped-decorator]
    def ymax(self, value: Any) -> None:
        self._set_position("ymax", value)

    @_BBoxMixin.x.setter  # type: ignore[attr-defined, untyped-decorator]
    def x(self, value: Any) -> None:
        self._set_position("x", value)

    @_BBoxMixin.y.setter  # type: ignore[attr-defined, untyped-decorator]
    def y(self, value: Any) -> None:
        self._set_position("y", value)

    @_BBoxMixin.center.setter  # type: ignore[attr-defined, untyped-decorator]
    def center(self, value: Any) -> None:
        if isinstance(value, Port):
            value = value.center
        value = asarray(value)
        c = self.center
        self.move(value[0] - c[0], value[1] - c[1])

    dxmin = xmin
    dxmax = xmax
    dymin = ymin
    dymax = ymax
    dx = x
    dy = y
    dcenter = center

    def get_polygons_points(self, layer: Any = None) -> dict[int, list[Array]]:
        """Differentiable (N, 2) point arrays of the referenced geometry, per layer."""
        return self.get_polygons(layer)

    def get_polygons(self, layer: Any = None) -> dict[int, list[Array]]:
        out: dict[int, list[Array]] = {}
        for t in self.array_transforms():
            for lay, polys in self.cell._flat_polygons(t, layer).items():
                out.setdefault(lay, []).extend(polys)
        return out

    def flatten(self, levels: int | None = None) -> None:
        """Flattens this reference into its parent cell."""
        parent = self.parent_cell
        if parent is None:
            raise ValueError("Reference has no parent cell")
        parent._check_unlocked()
        for t in self.array_transforms():
            for lay, polys in self.cell._flat_polygons(t).items():
                parent.polygons.setdefault(lay, []).extend(polys)
            for lab in self.cell._flat_labels(t):
                parent.labels.append(lab)
        parent.insts.remove(self)

    def copy(self) -> ComponentReference:
        return ComponentReference(
            self.cell, self.transform.copy(), None, None, self.na, self.nb, self.a, self.b
        )

    def __repr__(self) -> str:
        t = self.transform
        try:
            return (
                f"ComponentReference({self.cell.name!r}, x={to_float(t.x):.4g}, "
                f"y={to_float(t.y):.4g}, rotation={to_float(t.rotation):.4g}, "
                f"mirror={t.mirror})"
            )
        except Exception:
            return f"ComponentReference({self.cell.name!r})"

    # kfactory compatibility
    @property
    def instance(self) -> ComponentReference:
        return self

    @property
    def kcl(self) -> Any:
        from gdsfactory import kcl

        return kcl

    @property
    def size(self) -> int:
        return self.na * self.nb


Instance = ComponentReference
DInstance = ComponentReference
VInstance = ComponentReference


class PortWidthMismatchError(ValueError):
    pass


class PortLayerMismatchError(ValueError):
    pass


class PortTypeMismatchError(ValueError):
    pass


class Instances:
    """List of references, also accessible by name."""

    def __init__(self, parent: Component) -> None:
        self._parent = parent
        self._insts: list[ComponentReference] = []

    def __iter__(self) -> Iterator[ComponentReference]:
        return iter(list(self._insts))

    def __len__(self) -> int:
        return len(self._insts)

    def __getitem__(self, key: str | int) -> ComponentReference:
        if isinstance(key, int):
            return self._insts[key]
        for r in self._insts:
            if r.name == key:
                return r
        raise KeyError(f"{key!r} not in {[r.name for r in self._insts]}")

    def __contains__(self, item: str | ComponentReference) -> bool:
        if isinstance(item, ComponentReference):
            return any(r is item for r in self._insts)
        return any(r.name == item for r in self._insts)

    def append(self, ref: ComponentReference) -> None:
        self._insts.append(ref)

    def remove(self, ref: ComponentReference) -> None:
        self._insts = [r for r in self._insts if r is not ref]

    def __delitem__(self, key: str | int | ComponentReference) -> None:
        if isinstance(key, ComponentReference):
            self.remove(key)
        else:
            self.remove(self[key])

    def clear(self) -> None:
        self._insts.clear()

    def keys(self) -> list[str]:
        return [r.name for r in self._insts]

    def items(self) -> list[tuple[str, ComponentReference]]:
        return [(r.name, r) for r in self._insts]

    def values(self) -> list[ComponentReference]:
        return list(self._insts)

    def __repr__(self) -> str:
        return f"Instances({self.keys()})"


# --------------------------------------------------------------------------
# Component
# --------------------------------------------------------------------------
_unnamed_counter = itertools.count()
_LIVE_COMPONENTS: weakref.WeakSet[Component] = weakref.WeakSet()


def live_components() -> list[Component]:
    """All Components that are still alive (for layer bookkeeping)."""
    return list(_LIVE_COMPONENTS)


def remap_layer_index(old: int, new: int) -> None:
    """Moves geometry of all live components from layer index old to new."""
    for c in live_components():
        if old in c.polygons:
            c.polygons.setdefault(new, []).extend(c.polygons.pop(old))
        if old in c._paths:
            c._paths.setdefault(new, []).extend(c._paths.pop(old))
        for lab in c.labels:
            if lab.layer == old:
                lab.layer = new
        for p in c.ports:
            if int(p.layer) == old:
                from gdsfactory._ports import _layer_enum

                p.layer = _layer_enum(new)


class Component(_BBoxMixin):
    """Canvas where you add polygons, references and ports.

    (``ComponentAllAngle`` subclasses keep shapes and ports off-grid, like
    kfactory's virtual cells.)

    - stores settings that you use to build the component
    - stores info that you want to use
    - polygons, ports and references are differentiable jax arrays
    - can write to GDS/OASIS and show in KLayout (via an export to kfactory)
    """

    _virtual: bool = False

    def __new__(cls, name: str | None = None, base: Any = None, **kwargs: Any) -> Any:
        if isinstance(base, Component):
            # kfactory idiom ``Component(base=other.base)``: same cell
            return base
        return super().__new__(cls)

    def __init__(self, name: str | None = None, base: Any = None, **kwargs: Any) -> None:
        if isinstance(base, Component):
            return
        self._name = name or f"Unnamed_{next(_unnamed_counter)}"
        _LIVE_COMPONENTS.add(self)
        self.polygons: dict[int, list[Array]] = {}
        self.labels: list[Label] = []
        self._paths: dict[int, list[tuple[Array, float]]] = {}
        self.insts = Instances(self)
        self.ports = Ports()
        self.info = Info()
        self.settings = Info()
        self.function_name: str | None = None
        self.basename: str | None = None
        self.module: str | None = None
        self.locked = False
        self.routes: dict[str, Any] = {}
        self.pins = Pins()
        self.schematic: Any = None
        self.child: Component | None = None
        self._has_tracers_cache: bool | None = None
        self.lvs_equivalent_ports: list[list[str]] | None = None
        self.vinsts: tuple[ComponentReference, ...] = ()
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    # --------------------------------------------------------------- naming
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._check_unlocked()
        self._name = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, ports={self.ports.keys()})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _check_unlocked(self) -> None:
        if self.locked:
            raise LockedError(self)

    def lock(self) -> None:
        self.locked = True

    @property
    def kcl(self) -> Any:
        from gdsfactory import kcl

        return kcl

    @property
    def base(self) -> Component:
        """kfactory compatibility: the component itself."""
        return self

    @property
    def factory_name(self) -> str:
        """Name of the factory (cell function) that created this component."""
        name = self.basename or self.function_name
        if name is None:
            raise ValueError(f"Component {self.name!r} was not created by a cell function")
        return name

    @property
    def references(self) -> list[ComponentReference]:
        return list(self.insts)

    def called_cells(self) -> list[Component]:
        """All descendant components (unique, children first)."""
        seen: dict[int, Component] = {}

        def visit(c: Component) -> None:
            for r in c.insts:
                if id(r.cell) not in seen:
                    visit(r.cell)
                    seen[id(r.cell)] = r.cell

        visit(self)
        return list(seen.values())

    def has_tracers(self) -> bool:
        """True if any coordinate carries derivative information."""
        for polys in self.polygons.values():
            if any(is_tracer(p) for p in polys):
                return True
        for p in self.ports:
            if is_tracer(p.center_array) or is_tracer(p.orientation) or is_tracer(p.width):
                return True
        for r in self.insts:
            t = r.transform
            if any(is_tracer(v) for v in (t.x, t.y, t.rotation, t.magnification)):
                return True
            if r.cell.has_tracers():
                return True
        return False

    # ------------------------------------------------------------- polygons
    def add_polygon(self, points: Any, layer: LayerSpec) -> Any:
        """Adds a polygon. points: (N, 2) array-like, klayout polygon/box/region."""
        self._check_unlocked()
        key = _layer_key(layer)
        for arr in _to_point_arrays(points):
            if arr.shape[0] < 3:
                continue
            self.polygons.setdefault(key, []).append(arr if self._virtual else snap_dbu(arr))
        return None

    def add_polygons(self, polygons: Iterable[Any], layer: LayerSpec) -> None:
        for p in polygons:
            self.add_polygon(p, layer)

    def add_box(self, box: Box | Sequence[float], layer: LayerSpec) -> None:
        if isinstance(box, Box):
            l, b, r, t = box.left, box.bottom, box.right, box.top
        else:
            l, b, r, t = box
        self.add_polygon(_rect_points(l, b, r, t), layer)

    def shapes(self, layer: LayerSpec) -> _ShapesProxy:
        return _ShapesProxy(self, _layer_key(layer))

    def add_label(
        self,
        text: str = "hello",
        position: Any = (0.0, 0.0),
        layer: LayerSpec = "TEXT",
    ) -> Label:
        self._check_unlocked()
        if hasattr(position, "x") and hasattr(position, "y") and not isinstance(position, Port):
            position = (position.x, position.y)
        if isinstance(position, Port):
            position = position.center
        lab = Label(str(text), asarray(position), _layer_key(layer))
        self.labels.append(lab)
        return lab

    # ---------------------------------------------------------------- ports
    def add_port(
        self,
        name: str | None = None,
        *,
        port: Port | None = None,
        center: Any = None,
        width: Any = None,
        orientation: Any = None,
        layer: LayerSpec | None = None,
        port_type: str | None = None,
        keep_mirror: bool = False,
        cross_section: CrossSectionSpec | None = None,
        register_cross_section: bool = False,
    ) -> Port:
        """Adds a Port to the Component."""
        self._check_unlocked()
        from gdsfactory.pdk import get_active_pdk, get_cross_section, get_layer

        info: dict[str, Any] = {}
        mirror = False
        if port is not None:
            center = center if center is not None else port.center
            width = width if width is not None else port.width
            orientation = orientation if orientation is not None else port.orientation
            layer = layer if layer is not None else port.layer
            port_type = port_type if port_type is not None else port.port_type
            name = name if name is not None else port.name
            info = _copy.deepcopy(dict(port.info))
            mirror = port.mirror if keep_mirror else False
            _xs = port.info.get("cross_section")
            _xs_is_registered = isinstance(_xs, str) and _xs in get_active_pdk().cross_sections
            if cross_section is None and _xs_is_registered:
                cross_section = _xs

        xs_name = None
        xs = None
        if cross_section:
            xs = get_cross_section(cross_section)
            xs_name = xs.name
            if layer is None:
                layer = xs.layer
            if width is None:
                width = xs.width

        if port_type is None:
            port_type = "optical"
        if orientation is None:
            orientation = 0

        if port_type not in CONF.port_types:
            warnings.warn(
                f"Port type {port_type} not in {CONF.port_types}. "
                "Please add it to the port_types list in the config gf.CONF.port_types.",
                stacklevel=3,
            )

        if layer is None:
            raise AddPortError("Must specify layer or cross_section")
        if width is None:
            raise AddPortError("Must specify width or cross_section")
        if center is None:
            raise AddPortError("Must specify center or port")

        _port = Port(
            name=name,
            center=center,
            width=width,
            orientation=orientation,
            layer=get_layer(layer),
            port_type=port_type,
            info=info,
            mirror=mirror,
            on_grid=not self._virtual,
        )
        if xs_name and xs is not None:
            _port.info["cross_section"] = xs_name
            if len(getattr(xs, "sections", ())) > 1:
                try:
                    _port.info["cross_section_settings"] = clean_value_json(
                        xs, serialization_max_digits=15
                    )
                except Exception:
                    pass
            if register_cross_section:
                pdk = get_active_pdk()
                if xs_name not in pdk.cross_sections:
                    pdk.register_cross_sections(**{xs_name: lambda: xs})
        self.ports.append(_port)
        return _port

    def add_ports(
        self,
        ports: Iterable[Port] | Ports | dict[str, Port],
        prefix: str = "",
        suffix: str = "",
        **kwargs: Any,
    ) -> None:
        """Adds a list or dict of ports (copied)."""
        self._check_unlocked()
        if isinstance(ports, dict):
            items = list(ports.items())
        else:
            items = [(p.name, p) for p in ports]
        for name, p in items:
            new_name = f"{prefix}{name}{suffix}" if name is not None else None
            self.add_port(name=new_name, port=p, **kwargs)

    def get_ports_list(self, **kwargs: Any) -> list[Port]:
        from gdsfactory.port import select_ports

        return select_ports(ports=self.ports, **kwargs)

    def get_ports_dict(self, **kwargs: Any) -> dict[str, Port]:
        return {p.name: p for p in self.get_ports_list(**kwargs)}  # type: ignore[misc]

    def pprint_ports(self, **kwargs: Any) -> None:
        from gdsfactory.port import pprint_ports

        pprint_ports(self.get_ports_list(**kwargs))

    def auto_rename_ports(
        self, rename_func: Callable[..., None] | None = None, **kwargs: Any
    ) -> None:
        """Renames ports clockwise per port type (o1, o2, ... / e1, e2, ...).

        Without arguments this is kfactory's ``rename_clockwise_multi`` (the
        upstream default); with keyword arguments it is
        :func:`gdsfactory.port.auto_rename_ports`.
        """
        self._check_unlocked()
        if rename_func is not None:
            rename_func(self.ports, **kwargs)
            return
        if kwargs:
            from gdsfactory.port import auto_rename_ports

            auto_rename_ports(self, **kwargs)
            return
        for port_type, prefix in (("optical", "o"), ("electrical", "e")):
            _rename_clockwise(
                [p for p in self.ports if p.port_type == port_type], prefix=prefix
            )

    def create_port(self, **kwargs: Any) -> Port:
        self._check_unlocked()
        name = kwargs.pop("name", None)
        if "dcplx_trans" in kwargs:
            t = kwargs.pop("dcplx_trans")
            kwargs["center"] = (t.disp.x, t.disp.y)
            kwargs["orientation"] = t.angle
        if "trans" in kwargs:
            t = kwargs.pop("trans")
            kwargs["center"] = (t.disp.x * 1e-3, t.disp.y * 1e-3)
            kwargs["orientation"] = t.angle * 90
        return self.add_port(name=name, **kwargs)

    def create_pin(
        self,
        *,
        name: str | None = None,
        ports: Sequence[Port],
        pin_type: str = "DC",
        info: dict[str, Any] | None = None,
    ) -> Pin:
        """Registers a logical pin made of existing ports of this component."""
        self._check_unlocked()
        own = []
        for p in ports:
            match = next((q for q in self.ports if q is p or q.name == p.name), None)
            if match is None:
                raise ValueError(f"Port {p.name!r} is not a port of {self.name!r}")
            own.append(match)
        pin = Pin(name, own, pin_type, info)
        self.pins.append(pin)
        return pin

    def remove_port(self, name: str) -> None:
        self._check_unlocked()
        self.ports.remove(name)

    # ----------------------------------------------------------- references
    def add_ref(
        self,
        component: Component,
        name: str | None = None,
        columns: int = 1,
        rows: int = 1,
        column_pitch: Any = 0.0,
        row_pitch: Any = 0.0,
    ) -> ComponentReference:
        """Adds a reference (instance) to a Component."""
        if not isinstance(component, Component):
            raise ValueError(f"Expected a Component, got {type(component)}")
        self._check_unlocked()
        if rows > 1 and to_float(row_pitch) == 0:
            raise ValueError(f"rows = {rows} > 1 require {row_pitch=} > 0")
        if columns > 1 and to_float(column_pitch) == 0:
            raise ValueError(f"columns = {columns} > 1 require {column_pitch} > 0")
        if name is not None and name in self.insts:
            raise ValueError(f"Reference {name!r} already exists in {self.name!r}")
        ref = ComponentReference(
            component,
            Transform(),
            name=name,
            parent=self,
            na=columns,
            nb=rows,
            a=xp.stack([asarray(column_pitch), asarray(0.0)]),
            b=xp.stack([asarray(0.0), asarray(row_pitch)]),
        )
        if self._virtual:
            # references in virtual (all-angle) cells are virtual (kfactory VInstance)
            ref.virtual = True
        self.insts.append(ref)
        return ref

    def add_ref_off_grid(
        self, component: Component, name: str | None = None
    ) -> ComponentReference:
        """Adds an all-angle reference (upstream: a virtual instance)."""
        ref = self.add_ref(component, name=name)
        ref.virtual = True
        return ref

    create_vinst = add_ref_off_grid

    def create_inst(self, component: Component, **kwargs: Any) -> ComponentReference:
        na = kwargs.pop("na", 1)
        nb = kwargs.pop("nb", 1)
        a = kwargs.pop("a", (0.0, 0.0))
        b = kwargs.pop("b", (0.0, 0.0))
        ref = self.add_ref(component, **kwargs)
        ref.na, ref.nb = na, nb
        ref.a = _vec(a)
        ref.b = _vec(b)
        return ref

    def __lshift__(self, component: Component) -> ComponentReference:
        return self.add_ref(component)

    def add(self, instances: Iterable[ComponentReference] | ComponentReference) -> None:
        self._check_unlocked()
        refs = [instances] if isinstance(instances, ComponentReference) else list(instances)
        for r in refs:
            r.parent_cell = self
            self.insts.append(r)

    def absorb(self, reference: ComponentReference) -> Self:
        self._check_unlocked()
        if reference not in self.insts:
            raise ValueError(
                "The reference you asked to absorb does not exist in this Component."
            )
        reference.flatten()
        return self

    def remove(self, items: Any) -> None:
        self._check_unlocked()
        items = items if isinstance(items, list | tuple) else [items]
        for item in items:
            if isinstance(item, ComponentReference):
                self.insts.remove(item)
            elif isinstance(item, Port):
                self.ports.remove(item.name)  # type: ignore[arg-type]

    @property
    def named_references(self) -> dict[str, ComponentReference]:
        return {r.name: r for r in self.insts}

    # ------------------------------------------------------------ geometry
    def _flat_polygons(
        self, t: Transform | None = None, layer: Any = None
    ) -> dict[int, list[Array]]:
        """All polygons (recursively) transformed by t."""
        t = t or Transform()
        lay_key = _layer_key(layer) if layer is not None else None
        out: dict[int, list[Array]] = {}
        for lay, polys in self.polygons.items():
            if lay_key is not None and lay != lay_key:
                continue
            if t.is_identity():
                out.setdefault(lay, []).extend(polys)
            else:
                out.setdefault(lay, []).extend(t.apply(p) for p in polys)
        for r in self.insts:
            for rt in r.array_transforms():
                for lay, polys in r.cell._flat_polygons(t * rt, layer).items():
                    out.setdefault(lay, []).extend(polys)
        return out

    def _flat_polygons_klayout(
        self, t: Transform | None = None, kt: Any = None
    ) -> dict[int, list[Array]]:
        """Flattened polygons rounded exactly like KLayout's flatten.

        KLayout transforms the integer (dbu) shapes of each child with the
        product of the instances' ICplxTrans and rounds once. The value
        reproduces that; the derivative is the one of the float transformation
        (straight-through).
        """
        import klayout.db as kdb

        t = t or Transform()
        kt = kt if kt is not None else kdb.ICplxTrans()
        out: dict[int, list[Array]] = {}
        for lay, polys in self.polygons.items():
            out.setdefault(lay, []).extend(_klayout_apply(t, kt, p) for p in polys)
        for r in self.insts:
            for rt in r.array_transforms():
                krt = kdb.ICplxTrans(rt.to_klayout(), 1e-3)
                for lay, polys in r.cell._flat_polygons_klayout(t * rt, kt * krt).items():
                    out.setdefault(lay, []).extend(polys)
        return out

    def _flat_labels(self, t: Transform | None = None) -> list[Label]:
        t = t or Transform()
        out = [lab.transformed(t) for lab in self.labels]
        for r in self.insts:
            for rt in r.array_transforms():
                out.extend(r.cell._flat_labels(t * rt))
        return out

    def _transformed_bbox_points(self, t: Transform, layer: Any = None) -> Array | None:
        """Points whose bounding box is the bbox of this cell transformed by t."""
        # tight: exact for manhattan transforms (bbox corners), polygon points otherwise
        if self._is_empty(layer):
            return None
        if manhattan_angle(t.rotation) is not None:
            b = self.dbbox(layer)
            return t.apply(_rect_points(b.left, b.bottom, b.right, b.top))
        lay_key = _layer_key(layer) if layer is not None else None
        pts = [
            t.apply(p)
            for k, ps in self.polygons.items()
            if lay_key is None or k == lay_key
            for p in ps
        ]
        for r in self.insts:
            for rt in r.array_transforms():
                rp = r._bbox_points(t * rt, layer)
                if rp is not None:
                    pts.append(rp)
        return xp.concatenate(pts) if pts else None

    def _transformed_bbox_corners(self, t: Transform, layer: Any = None) -> Array | None:
        """Cell bbox corners transformed by t (kfactory's loose instance bbox).

        Rounded like KLayout's ``Box.transformed(ICplxTrans)`` (integer corners).
        """
        if self._is_empty(layer):
            return None
        b = self.dbbox(layer)
        return _klayout_apply(t, None, _rect_points(b.left, b.bottom, b.right, b.top))

    def _is_empty(self, layer: Any = None) -> bool:
        if layer is None:
            if any(self.polygons.values()):
                return False
        elif self.polygons.get(_layer_key(layer)):
            return False
        return all(r.cell._is_empty(layer) for r in self.insts)

    def dbbox(self, layer: Any = None) -> Box:
        """Bounding box (differentiable)."""
        pts: list[Array] = []
        lay_key = _layer_key(layer) if layer is not None else None
        for lay, polys in self.polygons.items():
            if lay_key is not None and lay != lay_key:
                continue
            pts.extend(polys)
        for r in self.insts:
            for rt in r.array_transforms():
                rp = r._bbox_points(rt, layer)
                if rp is not None:
                    pts.append(rp)
        if not pts:
            return Box.empty_box()
        return _box_from_points(xp.concatenate(pts))

    ibbox = dbbox

    def get_polygons(
        self,
        merge: bool = False,
        by: str = "index",
        layers: LayerSpecs | None = None,
        smooth: float | None = None,
        layer: LayerSpec | None = None,
    ) -> Any:
        """Returns KLayout polygons per layer (concrete, like upstream gdsfactory).

        Use :meth:`get_polygons_points` for differentiable (N, 2) point arrays.

        Args:
            merge: if True merges the polygons.
            by: the format of the dict key: "index", "name" or "tuple".
            layers: list of layers to get polygons from. Defaults to all layers.
            smooth: if set, smooths merged polygons.
            layer: if given, returns the list of kdb.DPolygon on that layer (um).
        """
        from gdsfactory import klayout_bridge as kb

        if layer is not None:
            key = _layer_key(layer)
            return [kb.array_to_kdpolygon(p) for p in self._flat_polygons(layer=layer).get(key, [])]
        polys = self._polygons_by(by=by, layers=layers)
        out: dict[Any, list[Any]] = {}
        for k, ps in polys.items():
            if merge or smooth:
                r = kb.merge_arrays(ps, smooth=smooth)
                out[k] = list(r.each())
            else:
                out[k] = [kb.array_to_kpolygon(p) for p in ps]
        return out

    def _polygons_by(
        self, by: str = "index", layers: LayerSpecs | None = None
    ) -> dict[Any, list[Array]]:
        from gdsfactory.pdk import get_layer_name, get_layer_tuple

        polys = self._flat_polygons()
        if layers is not None:
            keys = {_layer_key(lay) for lay in layers}
            polys = {k: v for k, v in polys.items() if k in keys}
        polys = {k: v for k, v in polys.items() if v}
        if by == "index":
            return dict(polys)
        if by == "name":
            return {get_layer_name(k): v for k, v in polys.items()}
        if by == "tuple":
            return {get_layer_tuple(k): v for k, v in polys.items()}
        raise ValueError(f"by={by!r} must be 'index', 'name' or 'tuple'")

    def get_polygons_points(
        self,
        merge: bool = False,
        scale: float | None = None,
        by: str = "index",
        layers: LayerSpecs | None = None,
        layer: LayerSpec | None = None,
    ) -> Any:
        """Returns (N, 2) point arrays per layer (differentiable unless merge=True).

        Args:
            merge: if True merges the polygons (KLayout, non-differentiable).
            scale: optional scaling factor for the points.
            by: the format of the dict key: "index", "name" or "tuple".
            layers: list of layers to get polygons from. Defaults to all layers.
            layer: if given, returns the list of arrays on that layer.
        """
        from gdsfactory import klayout_bridge as kb

        if layer is not None:
            ps = self._flat_polygons(layer=layer).get(_layer_key(layer), [])
            if merge:
                ps = kb.region_to_arrays(kb.merge_arrays(ps))
            return [p * scale for p in ps] if scale else list(ps)
        polys = self._polygons_by(by=by, layers=layers)
        if merge:
            polys = {k: kb.region_to_arrays(kb.merge_arrays(v)) for k, v in polys.items()}
        if scale:
            return {k: [p * scale for p in v] for k, v in polys.items()}
        return polys

    def get_labels(self, layer: LayerSpec | None = None, recursive: bool = True) -> list[Label]:
        labels = self._flat_labels() if recursive else list(self.labels)
        if layer is None:
            return labels
        key = _layer_key(layer)
        return [lab for lab in labels if lab.layer == key]

    def area(self, layer: LayerSpec | None = None) -> Any:
        """Area of a layer in um2 (overlaps merged, like upstream).

        The value is KLayout's merged area; the derivative is the derivative of
        the sum of the polygon areas (exact when polygons on the layer do not
        overlap). Use :meth:`area_unmerged` for a fully differentiable sum.
        """
        from gdsfactory import klayout_bridge as kb

        if layer is None:
            return sum(self.area(lay) for lay in self.layers)
        polys = self._flat_polygons(layer=layer).get(_layer_key(layer), [])
        if not polys:
            return 0.0
        merged = float(kb.merge_arrays(polys).area()) * 1e-6
        if any(is_tracer(p) for p in polys):
            total = sum(xp.abs(polygon_area(p)) for p in polys)
            return merged + (total - stop_gradient(total))
        return merged

    def area_unmerged(self, layer: LayerSpec | None = None) -> Any:
        """Sum of polygon areas (differentiable; overlaps are double counted)."""
        polys = self._flat_polygons(layer=layer)
        total: Any = 0.0
        for ps in polys.values():
            for p in ps:
                total = total + xp.abs(polygon_area(p))
        return total

    @property
    def layers(self) -> list[Layer]:
        from gdsfactory.pdk import get_layer_tuple

        keys = sorted({k for k, v in self._flat_polygons().items() if v})
        return [get_layer_tuple(k) for k in keys]

    def begin_shapes_rec(self, layer: Any) -> Any:
        """KLayout recursive shape iterator over an export of this component (concrete)."""
        from gdsfactory.pdk import get_layer_info

        kc = self.to_kfactory()
        return kc.kdb_cell.begin_shapes_rec(kc.kcl.layout.layer(get_layer_info(layer)))

    def get_region(self, layer: LayerSpec, merge: bool = False, smooth: float | None = None) -> kdb.Region:
        from gdsfactory import klayout_bridge as kb

        polys = self._flat_polygons(layer=layer).get(_layer_key(layer), [])
        r = kb.arrays_to_region(polys)
        if merge:
            r.merge()
        if smooth:
            r = r.smoothed(int(smooth * 1e3))
        return r

    # --------------------------------------------------------------- edits
    def flatten(self, merge: bool = True) -> Self:
        """Flattens all references into this component (in place).

        Without tracers this is exactly kfactory's ``KCell.flatten`` (KLayout
        flatten, then merge per layer when ``merge``). With tracers the polygons
        are transformed differentiably and not merged (same union of geometry,
        values within 1 nm of the untraced result).
        """
        self._check_unlocked()
        from gdsfactory import snap

        if snap.SNAP_ENABLED and not self.has_tracers():
            from gdsfactory.klayout_bridge import flatten_exact

            self.polygons, self.labels = flatten_exact(self, merge=merge)
            self.insts.clear()
            return self
        polys = self._flat_polygons_klayout()
        self.labels = self._flat_labels()
        self.insts.clear()
        self.polygons = polys
        return self

    def copy(self) -> Component:
        """Returns an unlocked shallow copy (children are shared, geometry copied)."""
        c = object.__new__(self.__class__)
        c.__dict__.update(self.__dict__)
        _LIVE_COMPONENTS.add(c)
        c._name = f"{self.name}_copy" if self.locked else self.name
        c.polygons = {k: list(v) for k, v in self.polygons.items()}
        c.labels = list(self.labels)
        c._paths = {k: list(v) for k, v in self._paths.items()}
        c.insts = Instances(c)
        for r in self.insts:
            nr = ComponentReference(
                r.cell, r.transform.copy(), r._name, c, r.na, r.nb, r.a, r.b
            )
            nr.virtual = r.virtual
            c.insts.append(nr)
        c.ports = Ports(p.copy() for p in self.ports)
        c.pins = Pins(
            Pin(
                pin.name,
                [c.ports[p.name] for p in pin.ports if p.name in c.ports],
                pin.pin_type,
                dict(pin.info),
            )
            for pin in self.pins
        )
        c.info = self.info.model_copy(deep=False)
        c.settings = self.settings.model_copy(deep=False)
        c.routes = dict(self.routes)
        c.vinsts = ()
        c.locked = False
        return c

    def dup(self, new_name: str | None = None) -> Self:
        c = self.copy()
        if new_name:
            c._name = new_name
        return c  # type: ignore[return-value]

    def transformed(self, t: Transform) -> Component:
        """Returns a new flat-ish component: this component referenced with transform t."""
        c = Component()
        ref = c.add_ref(self)
        ref.transform = t
        c.add_ports(ref.ports)
        return c

    def copy_child_info(self, component: Component) -> None:
        self._check_unlocked()
        for k, v in component.info.items():
            if k not in self.info:
                self.info[k] = v

    def add_route_info(
        self,
        cross_section: CrossSection | str,
        length: Any,
        length_eff: Any = None,
        taper: bool = False,
        **kwargs: Any,
    ) -> None:
        from gdsfactory.pdk import get_active_pdk

        self._check_unlocked()
        pdk = get_active_pdk()
        length_eff = length if length_eff is None else length_eff
        xs_name = (
            cross_section
            if isinstance(cross_section, str)
            else pdk.get_cross_section_name(cross_section)
        )
        info = self.info
        if taper:
            info[f"route_info_{xs_name}_taper_length"] = length
        info["route_info_type"] = xs_name
        info["route_info_length"] = length_eff
        info["route_info_weight"] = length_eff
        info[f"route_info_{xs_name}_length"] = length_eff
        for key, value in kwargs.items():
            info[f"route_info_{key}"] = value

    def extract(self, layers: LayerSpecs, recursive: bool = True) -> Component:
        """Returns a new flat Component with only the given layers."""
        keys = {_layer_key(lay) for lay in layers}
        c = Component()
        src = self._flat_polygons() if recursive else self.polygons
        for k, polys in src.items():
            if k in keys:
                c.polygons[k] = list(polys)
        c.add_ports(self.ports)
        return c

    def _cells_for(self, recursive: bool) -> list[Component]:
        return [self, *self.called_cells()] if recursive else [self]

    def _layer_op(self, recursive: bool, op: Callable[[Component], None]) -> Self:
        """Applies op to self (and all descendant cells if recursive), in place.

        Like upstream, recursive operations modify the referenced cells too
        (temporarily unlocking cached cells).
        """
        if not recursive:
            self._check_unlocked()
            op(self)
            return self
        for c in self._cells_for(True):
            was_locked = c.locked
            c.locked = False
            try:
                op(c)
            finally:
                c.locked = was_locked
        return self

    def remove_layers(
        self,
        layers: LayerSpecs,
        recursive: bool = True,
    ) -> Self:
        """Removes layers (in place; recursive also removes them from child cells)."""
        keys = {_layer_key(lay) for lay in layers}

        def op(c: Component) -> None:
            c.polygons = {k: v for k, v in c.polygons.items() if k not in keys}
            c.labels = [lab for lab in c.labels if lab.layer not in keys]
            c._paths = {k: v for k, v in c._paths.items() if k not in keys}

        return self._layer_op(recursive, op)

    def copy_layers(
        self,
        layer_map: dict[LayerSpec, LayerSpec],
        recursive: bool = False,
    ) -> Self:
        """Copies polygons from source to destination layers (in place)."""
        pairs = [(_layer_key(k), _layer_key(v)) for k, v in layer_map.items()]

        def op(c: Component) -> None:
            for src, dst in pairs:
                c.polygons.setdefault(dst, []).extend(list(c.polygons.get(src, [])))

        return self._layer_op(recursive, op)

    def remap_layers(
        self,
        layer_map: dict[LayerSpec, LayerSpec],
        recursive: bool = False,
    ) -> Self:
        """Moves polygons, labels and ports from source to destination layers."""
        remap = {_layer_key(k): _layer_key(v) for k, v in layer_map.items()}

        def op(c: Component) -> None:
            new: dict[int, list[Array]] = {}
            for k, polys in c.polygons.items():
                new.setdefault(remap.get(k, k), []).extend(polys)
            c.polygons = new
            for lab in c.labels:
                lab.layer = remap.get(lab.layer, lab.layer)
            for p in c.ports:
                if int(p.layer) in remap:
                    from gdsfactory.pdk import get_layer

                    p.layer = get_layer(remap[int(p.layer)])

        return self._layer_op(recursive, op)

    def offset(
        self,
        layer: LayerSpec,
        distance: float,
        flatten: bool = False,
        corner_mode: int = 2,
    ) -> None:
        """Grows/shrinks polygons on layer by distance (KLayout, non-differentiable)."""
        self._check_unlocked()
        from gdsfactory import klayout_bridge as kb

        if flatten:
            self.flatten()
        key = _layer_key(layer)
        polys = self._flat_polygons(layer=layer).get(key, [])
        d = round(to_float(distance) * 1e3)
        r = kb.arrays_to_region(polys)
        r = r.sized(d, d, corner_mode)
        self._replace_layer(key, kb.region_to_arrays(r))

    def over_under(self, layer: LayerSpec, distance: float = 1.0) -> None:
        """Grows then shrinks polygons (removes small gaps; KLayout, non-differentiable)."""
        self._check_unlocked()
        from gdsfactory import klayout_bridge as kb

        key = _layer_key(layer)
        polys = self._flat_polygons(layer=layer).get(key, [])
        d = round(to_float(distance) * 1e3)
        r = kb.arrays_to_region(polys).sized(d).sized(-d)
        self._replace_layer(key, kb.region_to_arrays(r))

    def _replace_layer(self, key: int, polys: list[Array]) -> None:
        """Replaces the (flattened) content of one layer, keeping the other layers."""
        if self.insts:
            for r in list(self.insts):
                if not r.cell._is_empty(key):
                    # move the other layers of the reference into this cell, drop this layer
                    for t in r.array_transforms():
                        for lay, ps in r.cell._flat_polygons(t).items():
                            if lay != key:
                                self.polygons.setdefault(lay, []).extend(ps)
                        self.labels.extend(r.cell._flat_labels(t))
                    self.insts.remove(r)
        self.polygons[key] = list(polys)

    def trim(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        flatten: bool = False,
    ) -> None:
        """Clips the component to a box (KLayout, non-differentiable; flattens)."""
        self._check_unlocked()
        from gdsfactory import klayout_bridge as kb

        b = self.dbbox()
        if (
            to_float(b.left) >= left
            and to_float(b.right) <= right
            and to_float(b.bottom) >= bottom
            and to_float(b.top) <= top
        ):
            return
        import klayout.db as kdb

        box = kdb.Region(kdb.Box(round(left * 1e3), round(bottom * 1e3), round(right * 1e3), round(top * 1e3)))
        polys = self._flat_polygons()
        labels = [
            lab
            for lab in self._flat_labels()
            if left <= to_float(lab.x) <= right and bottom <= to_float(lab.y) <= top
        ]
        self.insts.clear()
        self.polygons = {
            k: kb.region_to_arrays(kb.arrays_to_region(v) & box) for k, v in polys.items()
        }
        self.labels = labels

    def fix_width(
        self,
        layer: LayerSpec,
        min_width: float = 0.2,
        n_threads: int | None = None,
        tile_size: tuple[float, float] | None = None,
        overlap: int = 1,
        smooth: int | None = None,
        flatten: bool = True,
    ) -> None:
        """Fixes min width of a layer (KLayout minkowski, non-differentiable)."""
        from kfactory.utils.violations import fix_width_minkowski_tiled

        from gdsfactory.pdk import get_layer_info

        self._check_unlocked()
        key = _layer_key(layer)
        kc = self._export_layer(layer)
        fix = fix_width_minkowski_tiled(
            kc.to_itype(),
            min_width=round(to_float(min_width) * 1e3),
            ref=get_layer_info(layer),
            n_threads=n_threads,
            tile_size=tile_size,
            overlap=overlap,
            smooth=smooth,
        )
        from gdsfactory import klayout_bridge as kb

        self._replace_layer(key, kb.region_to_arrays(fix))

    def fix_spacing(
        self,
        layer: LayerSpec,
        min_space: float = 0.2,
        size_bias: float = 0.0,
        smooth_factor: float = 0.05,
    ) -> None:
        """Fixes min spacing of a layer (KLayout, non-differentiable).

        Like upstream, the fixed geometry is added to the layer.
        """
        from kfactory.utils.violations import fix_spacing_tiled

        from gdsfactory import klayout_bridge as kb
        from gdsfactory.pdk import get_layer_info

        self._check_unlocked()
        key = _layer_key(layer)
        kc = self._export_layer(layer)
        fix = fix_spacing_tiled(
            kc.to_itype(),
            min_space=round(to_float(min_space) * 1e3),
            layer=get_layer_info(layer),
            smooth_factor=smooth_factor,
        )
        if size_bias:
            d = round(to_float(size_bias) * 1e3)
            fix = fix.sized(+d).sized(-d)
        self.polygons.setdefault(key, []).extend(kb.region_to_arrays(fix))

    def _export_layer(self, layer: LayerSpec) -> Any:
        """Concrete kfactory cell with the flattened polygons of one layer."""
        from gdsfactory.klayout_bridge import to_kfactory

        tmp = Component(name=f"{self.name}_layer")
        key = _layer_key(layer)
        tmp.polygons = {key: list(self._flat_polygons(layer=layer).get(key, []))}
        return to_kfactory(tmp, add_ports=False)

    def fill(
        self,
        fill_cell: Any,
        fill_layers: Iterable[tuple[LayerSpec, float]] = (),
        fill_regions: Iterable[tuple[Any, float]] = (),
        exclude_layers: Iterable[tuple[LayerSpec, float]] = (),
        exclude_regions: Iterable[tuple[Any, float]] = (),
        n_threads: int | None = None,
        tile_size: tuple[float, float] | None = None,
        row_step: Any = None,
        col_step: Any = None,
        x_space: float = 0,
        y_space: float = 0,
        tile_border: tuple[float, float] = (20, 20),
        multi: bool = False,
    ) -> None:
        """Fills regions with fill_cell (KLayout tiled fill, non-differentiable).

        The fill cells are added as references of ``fill_cell``.
        """
        import kfactory as kf
        from kfactory.utils.fill import fill_tiled

        from gdsfactory.klayout_bridge import to_kfactory
        from gdsfactory.pdk import get_component, get_layer_info
        from gdsfactory.transform import Transform

        self._check_unlocked()
        fill_comp = get_component(fill_cell)
        kcl = kf.KCLayout(f"gdsfactoryx_fill_{next(_unnamed_counter)}")
        kcl.layout.dbu = 1e-3
        kc = to_kfactory(self, kcl=kcl, add_ports=False)
        kfill = to_kfactory(fill_comp, kcl=kcl, add_ports=False, unique_prefix="fill_")
        before = {id(i) for i in kc.kdb_cell.each_inst()}
        n_before = kc.kdb_cell.child_instances()
        fill_tiled(
            kc,
            fill_cell=kfill,
            fill_layers=[(get_layer_info(lay), int(sp)) for lay, sp in fill_layers],
            fill_regions=[(r, int(sp)) for r, sp in fill_regions],
            exclude_layers=[(get_layer_info(lay), int(sp)) for lay, sp in exclude_layers],
            exclude_regions=[(r, int(sp)) for r, sp in exclude_regions],
            n_threads=n_threads,
            tile_size=tile_size,
            row_step=row_step,
            col_step=col_step,
            x_space=x_space,
            y_space=y_space,
            tile_border=tile_border,
            multi=multi,
        )
        _ = before
        for i, inst in enumerate(kc.kdb_cell.each_inst()):
            if i < n_before or inst.cell_index != kfill.cell_index():
                continue
            ca = inst.dcell_inst
            ref = self.add_ref(fill_comp)
            ref.transform = Transform.from_klayout(inst.dcplx_trans)
            if ca.is_regular_array():
                ref.na, ref.nb = ca.na, ca.nb
                ref.a = asarray([ca.a.x, ca.a.y])
                ref.b = asarray([ca.b.x, ca.b.y])

    def child_cells(self) -> int:
        """Number of distinct cells referenced directly by this cell."""
        return len({id(r.cell) for r in self.insts})

    def delete(self) -> None:
        """Kept for kfactory compatibility (components are garbage collected)."""
        self.insts.clear()
        self.polygons = {}
        self.labels = []

    def get_boxes(self, layer: LayerSpec, recursive: bool = True) -> list[Any]:
        """Axis-aligned rectangles on a layer as kdb.DBox (concrete)."""
        import klayout.db as kdb

        key = _layer_key(layer)
        polys = (
            self._flat_polygons(layer=layer).get(key, [])
            if recursive
            else self.polygons.get(key, [])
        )
        boxes = []
        for p in polys:
            q = to_numpy(p)
            if q.shape[0] == 4:
                xs = set(np.round(q[:, 0], 9))
                ys = set(np.round(q[:, 1], 9))
                if len(xs) == 2 and len(ys) == 2:
                    boxes.append(kdb.DBox(min(xs), min(ys), max(xs), max(ys)))
        return boxes

    def get_paths(self, layer: LayerSpec, recursive: bool = True) -> list[Any]:
        """Paths inserted through ``shapes(layer).insert(DPath)`` as kdb.DPath."""
        import klayout.db as kdb

        key = _layer_key(layer)
        out = []
        items: list[tuple[Transform, Component]] = [(Transform(), self)]
        if recursive:
            stack: list[tuple[Transform, Component]] = [(Transform(), self)]
            items = []
            while stack:
                t, c = stack.pop()
                items.append((t, c))
                for r in c.insts:
                    for rt in r.array_transforms():
                        stack.append((t * rt, r.cell))
        for t, c in items:
            for pts, width in c._paths.get(key, []):
                q = to_numpy(t.apply(pts))
                out.append(kdb.DPath([kdb.DPoint(float(x), float(y)) for x, y in q], float(width)))
        return out

    # --------------------------------------------------------- transforms
    def move(self, *args: Any) -> Self:
        """Moves all geometry and ports in place (differentiable)."""
        from gdsfactory._ports import _move_args

        self._check_unlocked()
        dx, dy = _move_args(args)
        self.transform(Transform(dx, dy))
        return self

    def _set_position(self, attr: str, value: Any) -> None:
        if isinstance(value, Port):
            value = value.x if attr in {"x", "xmin", "xmax"} else value.y
        delta = value - getattr(self, attr)
        if attr in {"x", "xmin", "xmax"}:
            self.move(delta, 0.0)
        else:
            self.move(0.0, delta)

    @_BBoxMixin.xmin.setter  # type: ignore[attr-defined, untyped-decorator]
    def xmin(self, value: Any) -> None:
        self._set_position("xmin", value)

    @_BBoxMixin.xmax.setter  # type: ignore[attr-defined, untyped-decorator]
    def xmax(self, value: Any) -> None:
        self._set_position("xmax", value)

    @_BBoxMixin.ymin.setter  # type: ignore[attr-defined, untyped-decorator]
    def ymin(self, value: Any) -> None:
        self._set_position("ymin", value)

    @_BBoxMixin.ymax.setter  # type: ignore[attr-defined, untyped-decorator]
    def ymax(self, value: Any) -> None:
        self._set_position("ymax", value)

    @_BBoxMixin.x.setter  # type: ignore[attr-defined, untyped-decorator]
    def x(self, value: Any) -> None:
        self._set_position("x", value)

    @_BBoxMixin.y.setter  # type: ignore[attr-defined, untyped-decorator]
    def y(self, value: Any) -> None:
        self._set_position("y", value)

    @_BBoxMixin.center.setter  # type: ignore[attr-defined, untyped-decorator]
    def center(self, value: Any) -> None:
        if isinstance(value, Port):
            value = value.center
        value = asarray(value)
        c = self.center
        self.move(value[0] - c[0], value[1] - c[1])

    dxmin = xmin
    dxmax = xmax
    dymin = ymin
    dymax = ymax
    dx = x
    dy = y
    dcenter = center

    def movex(self, dx: Any) -> Self:
        return self.move(dx, 0.0)

    def movey(self, dy: Any) -> Self:
        return self.move(0.0, dy)

    dmove = move
    dmovex = movex
    dmovey = movey

    def transform(self, t: Transform) -> Self:
        self._check_unlocked()
        self.polygons = {k: [t.apply(p) for p in v] for k, v in self.polygons.items()}
        self.labels = [lab.transformed(t) for lab in self.labels]
        for r in self.insts:
            r.transform_by(t)
        for p in self.ports:
            p.apply_transform(t)
        return self

    def rotate(self, angle: Any, center: Any = (0.0, 0.0)) -> Self:
        c = asarray(center)
        return self.transform(
            Transform(c[0], c[1]) * Transform(0.0, 0.0, angle) * Transform(-c[0], -c[1])
        )

    def mirror_x(self, x: Any = 0.0) -> Self:
        x = asarray(x)
        return self.transform(
            Transform(x, 0.0) * Transform(0.0, 0.0, 180.0, mirror=True) * Transform(-x, 0.0)
        )

    def mirror_y(self, y: Any = 0.0) -> Self:
        y = asarray(y)
        return self.transform(
            Transform(0.0, y) * Transform(0.0, 0.0, 0.0, mirror=True) * Transform(0.0, -y)
        )

    def __getitem__(self, key: str) -> Port:
        return self.ports[key]

    def __contains__(self, key: str) -> bool:
        return key in self.ports

    # -------------------------------------------------------------- export
    def to_kfactory(self, kcl: Any = None) -> Any:
        """Exports to a (non-differentiable) kfactory DKCell."""
        from gdsfactory.klayout_bridge import to_kfactory

        return to_kfactory(self, kcl=kcl)

    def write_gds(
        self,
        gdspath: PathType | None = None,
        gdsdir: PathType | None = None,
        save_options: Any = None,
        with_metadata: bool = True,
        exclude_layers: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> pathlib.Path:
        """Writes the component to GDS (coordinates rounded to 1 nm)."""
        from gdsfactory.klayout_bridge import write

        if gdspath and gdsdir:
            warnings.warn("gdspath and gdsdir have both been specified.", stacklevel=2)
        gdsdir = pathlib.Path(gdsdir or GDSDIR_TEMP)
        gdspath = pathlib.Path(gdspath or gdsdir / f"{self.name}.gds")
        gdspath.parent.mkdir(parents=True, exist_ok=True)
        write(
            self,
            gdspath,
            save_options=save_options,
            with_metadata=with_metadata,
            exclude_layers=exclude_layers,
        )
        return gdspath

    def write(self, filename: PathType, **kwargs: Any) -> None:
        self.write_gds(gdspath=filename, **kwargs)

    def write_oas(self, gdspath: PathType | None = None, **kwargs: Any) -> pathlib.Path:
        gdspath = gdspath or pathlib.Path(GDSDIR_TEMP) / f"{self.name}.oas"
        return self.write_gds(gdspath=gdspath, **kwargs)

    def show(self, **kwargs: Any) -> None:
        """Shows in KLayout (requires klive)."""
        from gdsfactory.klayout_bridge import show

        show(self, **kwargs)

    def plot(
        self,
        lyrdb: Any = None,
        display_type: Any = None,
        *,
        show_labels: bool = True,
        show_ruler: bool = True,
        pixel_buffer_options: Any = None,
        return_fig: bool = False,
        ax: Axes | None = None,
        show_ports: bool = True,
    ) -> Figure | None:
        """Plots the component with KLayout's renderer (concrete values)."""
        from gdsfactory.plot import plot_component

        return plot_component(
            self,
            ax=ax,
            show_labels=show_labels,
            show_ruler=show_ruler,
            show_ports=show_ports,
            return_fig=return_fig,
            pixel_buffer_options=pixel_buffer_options,
        )

    def plot_klayout(self, **kwargs: Any) -> Any:
        return self.to_kfactory().plot(**kwargs)

    def to_dict(self, with_ports: bool = False) -> dict[str, Any]:
        """Returns a dictionary representation of the Component."""
        from gdsfactory.port import to_dict

        d: dict[str, Any] = {
            "name": self.name,
            "info": self.info.model_dump(exclude_none=True),
            "settings": self.settings.model_dump(exclude_none=True),
        }
        if with_ports:
            d["ports"] = {
                port.name: to_dict(port) for port in self.ports if port.name is not None
            }
        res = clean_value_json(d)
        assert isinstance(res, dict)
        return res

    def get_netlist(self, recursive: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Returns the netlist (recursive=True: netlists of all hierarchy levels)."""
        from gdsfactory.get_netlist import function_namer, get_netlist, get_netlist_recursive

        if kwargs.get("component_namer") is None:
            kwargs["component_namer"] = function_namer
        if recursive:
            return get_netlist_recursive(self, **kwargs)
        kwargs.pop("netlist_namer", None)
        return get_netlist(self, **kwargs)

    def get_netlist_recursive(self, **kwargs: Any) -> dict[str, Any]:
        from gdsfactory.get_netlist import get_netlist_recursive

        return get_netlist_recursive(self, **kwargs)

    def write_netlist(self, netlist: dict[str, Any], filepath: str | pathlib.Path | None = None) -> str:
        import yaml

        netlist = clean_value_json(netlist)
        yaml_string = yaml.dump(netlist)
        if filepath:
            pathlib.Path(filepath).write_text(yaml_string)
        return yaml_string

    def to_3d(self, *args: Any, **kwargs: Any) -> Any:
        from gdsfactory.export.to_3d import to_3d

        return to_3d(self, *args, **kwargs)

    def to_array(self, *args: Any, **kwargs: Any) -> Array:
        """Differentiable rasterization (see gdsfactory.rasterize.rasterize)."""
        from gdsfactory.rasterize import rasterize

        return rasterize(self, *args, **kwargs)

    def has_ports(self) -> bool:
        return len(self.ports) > 0

    # kfactory compat helpers
    def each_inst(self) -> Iterator[ComponentReference]:
        return iter(self.insts)

    @property
    def kdb_cell(self) -> Any:
        return self.to_kfactory().kdb_cell

    def is_library_cell(self) -> bool:
        return False


class ComponentAllAngle(Component):
    """All-angle component (kfactory's VKCell): shapes and ports stay off-grid.

    References inside it, and references to it added with ``add_ref_off_grid``,
    are virtual: they are materialized on export like upstream (the residual
    non-manhattan transformation is baked into a copy of the cell).
    """

    _virtual = True

ComponentBase = Component
AnyComponent: TypeAlias = Component


class _ShapesProxy:
    """kfactory ``cell.shapes(layer)`` emulation (KLayout objects, concrete).

    The differentiable geometry lives in ``component.polygons``; this proxy
    converts on the fly.
    """

    def __init__(self, c: Component, layer: int) -> None:
        self._c = c
        self._layer = layer

    def insert(self, shape: Any) -> Any:
        import klayout.db as kdb

        self._c._check_unlocked()
        if isinstance(shape, kdb.DText | kdb.Text):
            t = shape if isinstance(shape, kdb.DText) else shape.to_dtype(1e-3)
            self._c.labels.append(Label(t.string, (t.x, t.y), self._layer))
            return shape
        if isinstance(shape, kdb.Texts):
            for t in shape.each():
                self.insert(t)
            return shape
        if isinstance(shape, kdb.DPath | kdb.Path):
            dpath = shape if isinstance(shape, kdb.DPath) else shape.to_dtype(1e-3)
            pts = asarray([[p.x, p.y] for p in dpath.each_point()])
            self._c._paths.setdefault(self._layer, []).append((pts, dpath.width))
        for arr in _to_point_arrays(shape):
            if arr.shape[0] >= 3:
                self._c.polygons.setdefault(self._layer, []).append(
                    arr if self._c._virtual else snap_dbu(arr)
                )
        return shape

    def each(self, *args: Any) -> Iterator[Any]:
        import klayout.db as kdb

        from gdsfactory.klayout_bridge import array_to_kdpolygon

        for p in self._c.polygons.get(self._layer, []):
            yield array_to_kdpolygon(p)
        for lab in self._c.labels:
            if lab.layer == self._layer:
                yield kdb.DText(lab.text, to_float(lab.x), to_float(lab.y))

    def __iter__(self) -> Iterator[Any]:
        return self.each()

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        n_labels = sum(1 for lab in self._c.labels if lab.layer == self._layer)
        return len(self._c.polygons.get(self._layer, [])) + n_labels

    def is_empty(self) -> bool:
        return self.size() == 0

    def clear(self) -> None:
        self._c._check_unlocked()
        self._c.polygons.pop(self._layer, None)
        self._c.labels = [lab for lab in self._c.labels if lab.layer != self._layer]
        self._c._paths.pop(self._layer, None)

    def bbox(self) -> Any:
        import klayout.db as kdb

        b = kdb.DBox()
        for p in self.each():
            b += p.bbox()
        return b

def _kl_value(t: Transform, kt: Any) -> Transform:
    """t with values taken from KLayout's composition kt (straight-through).

    Reproduces KLayout's floating point arithmetic (same rounding as upstream)
    while keeping the derivatives of the differentiable composition t.
    """
    rot = t.rotation
    if manhattan_angle(rot) is not None:
        # kfactory uses exact integer transformations for manhattan placements
        return t
    r = to_float(rot)
    a = kt.angle
    a_adj = r + ((a - r + 180) % 360 - 180)

    def st(value: Any, target: float) -> Any:
        if is_tracer(value) or isinstance(value, jax_array_types()):
            return value + stop_gradient(target - value)
        return target

    return Transform(
        st(t.x, kt.disp.x),
        st(t.y, kt.disp.y),
        st(rot, a_adj) if abs(a_adj - r) > 0 else rot,
        t.mirror,
        t.magnification,
    )


def _klayout_apply(t: Transform, kt: Any, points: Array) -> Array:
    """t.apply(points) with the value computed by KLayout's ICplxTrans kt (dbu).

    Used where upstream geometry goes through KLayout's integer transformation
    (flatten, instance bounding boxes) so that rounding matches exactly.
    """
    import klayout.db as kdb

    from gdsfactory import snap

    if t.is_identity():
        return points
    exact = t.apply(points)
    if not snap.SNAP_ENABLED:
        return exact
    if kt is None:
        kt = kdb.ICplxTrans(t.to_klayout(), 1e-3)
    q = np.round(to_numpy(points) * 1000.0).astype(np.int64)
    if kt.is_ortho() and not kt.is_mag() and float(kt.disp.x).is_integer() and float(kt.disp.y).is_integer():
        # integer arithmetic: exact
        st = kt.s_trans()
        res = [st.trans(kdb.Point(int(x), int(y))) for x, y in q]
    else:
        res = [kt.trans(kdb.Point(int(x), int(y))) for x, y in q]
    rounded = np.array([[p.x, p.y] for p in res], dtype=np.float64) * 1e-3
    if is_tracer(exact) or isinstance(exact, jax_array_types()):
        return exact + stop_gradient(asarray(rounded) - exact)
    rounded.flags.writeable = False
    return rounded


def jax_array_types() -> tuple[type, ...]:
    import jax

    return (jax.Array,)


def _rename_clockwise(ports: list[Port], prefix: str = "o", start: int = 1) -> None:
    """kfactory ``rename_clockwise`` (sorts on the integer port transformation)."""

    def sort_key(port: Port) -> tuple[int, int, int]:
        t = port.trans
        angle = {2: 0, 1: 1, 0: 2}.get(t.angle, 3)
        dir_1 = 1 if angle < 2 else -1
        dir_2 = -1 if t.angle < 2 else 1
        key_1 = dir_1 * (t.disp.x if angle % 2 else t.disp.y)
        key_2 = dir_2 * (t.disp.y if angle % 2 else t.disp.x)
        return angle, key_1, key_2

    for i, p in enumerate(sorted(ports, key=sort_key), start=start):
        p.name = f"{prefix}{i}"


def polygon_area(points: Array) -> Any:
    """Signed shoelace area (differentiable)."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * xp.sum(x * xp.roll(y, -1) - xp.roll(x, -1) * y)


def _rect_points(left: Any, bottom: Any, right: Any, top: Any) -> Array:
    l, b, r, t = (asarray(v) for v in (left, bottom, right, top))
    return xp.stack(
        [xp.stack([l, b]), xp.stack([l, t]), xp.stack([r, t]), xp.stack([r, b])]
    )


def _vec(v: Any) -> Array:
    if hasattr(v, "x") and hasattr(v, "y"):
        return xp.stack([asarray(v.x), asarray(v.y)])
    return asarray(v)


def _to_point_arrays(points: Any) -> list[Array]:
    """Converts any polygon-like input into a list of (N, 2) arrays."""
    if isinstance(points, Box):
        return [_rect_points(points.left, points.bottom, points.right, points.top)]
    mod = type(points).__module__
    if mod.startswith("klayout") or mod.startswith("pya"):
        from gdsfactory.klayout_bridge import klayout_shape_to_arrays

        return klayout_shape_to_arrays(points)
    if isinstance(points, xp.ndarray | np.ndarray) or is_tracer(points):
        return [points_array(points)]
    pts = list(points)
    if not pts:
        return []
    first = pts[0]
    if hasattr(first, "x") and hasattr(first, "y") and type(first).__module__.startswith("klayout"):
        return [asarray([[q.x, q.y] for q in pts])]
    return [points_array(pts)]


# ---------------------------------------------------------------------------
# KLayout region helpers (non-differentiable boolean / sizing operations)
# ---------------------------------------------------------------------------
def ensure_tuple_of_tuples(points: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(points, np.ndarray) or isinstance(points, xp.ndarray) or is_tracer(points):
        return tuple(map(tuple, to_numpy(points).tolist()))
    if isinstance(points, list) and points and isinstance(points[0], np.ndarray | list):
        return tuple(tuple(point) for point in points)
    return points  # type: ignore[no-any-return]


def points_to_polygon(points: Any) -> Any:
    """Returns a KLayout DPolygon (concrete values) from points."""
    import klayout.db as kdb

    if isinstance(points, kdb.Polygon | kdb.DPolygon | kdb.DSimplePolygon | kdb.Region):
        return points
    pts = to_numpy(asarray(ensure_tuple_of_tuples(points)))
    return kdb.DPolygon([kdb.DPoint(float(x), float(y)) for x, y in pts])


def size(region: Any, offset: float, dbu: float = 1e3) -> Any:
    return region.dup().size(int(offset * dbu))


def boolean_or(region1: Any, region2: Any) -> Any:
    return (region1.__or__(region2)).merge()


def boolean_not(region1: Any, region2: Any) -> Any:
    return region1 - region2


def boolean_xor(region1: Any, region2: Any) -> Any:
    return region1 ^ region2


def boolean_and(region1: Any, region2: Any) -> Any:
    return region1 & region2


def copy(region: Any) -> Any:
    return region.dup()


boolean_operations = {
    "or": boolean_or,
    "|": boolean_or,
    "not": boolean_not,
    "-": boolean_not,
    "^": boolean_xor,
    "xor": boolean_xor,
    "&": boolean_and,
    "and": boolean_and,
    "A-B": boolean_not,
}


def container(
    component: Any,
    function: Callable[..., None] | None = None,
    **kwargs: Any,
) -> Component:
    """Returns new component with a component reference.

    Args:
        component: to add to container.
        function: function to apply to component.
        kwargs: keyword arguments to pass to function.
    """
    import gdsfactory as gf

    component = gf.get_component(component)
    c = Component()
    cref = c << component
    c.add_ports(cref.ports)
    if function:
        function(c, **kwargs)
    c.copy_child_info(component)
    return c


def _stop(x: Any) -> Any:
    return to_numpy(x)


__all__ = [
    "AddPortError",
    "Box",
    "Component",
    "ComponentAllAngle",
    "ComponentBase",
    "ComponentReference",
    "Info",
    "Instance",
    "Label",
    "LockedError",
    "container",
    "polygon_area",
]

_ = (has_tracers, maybe_float, PortInfo)
