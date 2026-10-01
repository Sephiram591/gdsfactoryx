"""Differentiable Port and Ports containers.

A Port stores its center, orientation (degrees) and width as (possibly traced)
jax scalars, so port positions can be differentiated with respect to the
parameters that produced them.
"""

from __future__ import annotations

import copy as _copy
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from gdsfactory._jax import (
    Array,
    asarray,
    cos_deg,
    is_tracer,
    jnp,
    xp,
    maybe_float,
    sin_deg,
    snap_dbu,
    to_float,
)
from gdsfactory.transform import Transform, manhattan_angle


class PortInfo(dict[str, Any]):
    """dict with kfactory-compatible helpers."""

    def model_copy(self, deep: bool = False) -> PortInfo:
        return PortInfo(_copy.deepcopy(dict(self)) if deep else dict(self))

    def model_dump(self) -> dict[str, Any]:
        return dict(self)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


class Port:
    """Differentiable port.

    Args:
        name: port name.
        center: (x, y) in um.
        width: port width in um.
        orientation: in degrees (any angle allowed).
        layer: layer index (LayerEnum or int).
        port_type: optical, electrical, placement, ...
        info: additional metadata.
    """

    __slots__ = (
        "_center",
        "_orientation",
        "_width",
        "info",
        "layer",
        "mirror",
        "name",
        "port_type",
        "__dict__",  # allows extra attributes (e.g. name_original)
    )

    def __init__(
        self,
        name: str | None = None,
        center: Any = (0.0, 0.0),
        width: Any = 0.5,
        orientation: Any = 0.0,
        layer: Any = 0,
        port_type: str = "optical",
        info: dict[str, Any] | None = None,
        mirror: bool = False,
        cross_section: Any = None,
    ) -> None:
        self.name = name
        self._orientation = _angle(orientation)
        self._center = self._snap_center(_point(center))
        self._width = snap_dbu(maybe_float(width)) if width is not None else None
        self.layer = _layer_enum(layer)
        self.port_type = port_type
        self.info = PortInfo(info or {})
        self.mirror = mirror
        if cross_section is not None:
            xs_name = getattr(cross_section, "name", cross_section)
            self.info["cross_section"] = xs_name

    # ------------------------------------------------------------------ props
    def _snap_center(self, c: Any) -> Any:
        """Manhattan ports live on the dbu grid (kfactory integer ports)."""
        if self._orientation is None or manhattan_angle(self._orientation) is not None:
            return snap_dbu(c)
        return c

    @property
    def center(self) -> tuple[Any, Any]:
        """(x, y) tuple (entries are floats/numpy scalars, or tracers when traced)."""
        return (self._center[0], self._center[1])

    @property
    def center_array(self) -> Array:
        """Center as a (2,) array (numpy, or jax when traced)."""
        return self._center

    xy = center_array

    @center.setter
    def center(self, value: Any) -> None:
        self._center = self._snap_center(_point(value))

    @property
    def x(self) -> Any:
        return self._center[0]

    @x.setter
    def x(self, value: Any) -> None:
        self._center = self._snap_center(xp.stack([asarray(value), self._center[1]]))

    @property
    def y(self) -> Any:
        return self._center[1]

    @y.setter
    def y(self, value: Any) -> None:
        self._center = self._snap_center(xp.stack([self._center[0], asarray(value)]))

    @property
    def orientation(self) -> Any:
        return self._orientation

    @orientation.setter
    def orientation(self, value: Any) -> None:
        self._orientation = _angle(value)

    @property
    def width(self) -> Any:
        return self._width

    @width.setter
    def width(self, value: Any) -> None:
        self._width = snap_dbu(maybe_float(value))

    @property
    def iwidth(self) -> int:
        """Width in database units (1 nm)."""
        return round(to_float(self._width) * 1000)

    @property
    def icenter(self) -> tuple[int, int]:
        return round(to_float(self.x) * 1000), round(to_float(self.y) * 1000)

    # kfactory compatibility aliases
    dcenter = center
    dx = x
    dy = y
    dwidth = width

    @property
    def angle(self) -> int:
        """Manhattan orientation index (0: east, 1: north, 2: west, 3: south)."""
        a = manhattan_angle(self._orientation)
        if a is None:
            raise ValueError(
                f"Port {self.name!r} orientation {self.orientation} is not manhattan"
            )
        return a // 90

    dangle = orientation

    @property
    def is_manhattan(self) -> bool:
        return manhattan_angle(self._orientation) is not None

    @property
    def cross_section(self) -> Any:
        """Cross-section of the port (object with ``.name`` and ``.width``).

        The PDK CrossSection named in ``info["cross_section"]`` if registered,
        otherwise a minimal description derived from layer and width.
        """
        from types import SimpleNamespace

        name = self.info.get("cross_section")
        if isinstance(name, str):
            try:
                from gdsfactory.pdk import get_cross_section

                return get_cross_section(name)
            except Exception:
                return SimpleNamespace(name=name, width=self.width, layer=self.layer)
        if name is not None and hasattr(name, "name"):
            return name
        try:
            from gdsfactory.pdk import get_layer_name

            layer_name = get_layer_name(self.layer)
        except Exception:
            layer_name = str(self.layer)
        auto = f"{layer_name}_{round(to_float(self.width) * 1000)}"
        return SimpleNamespace(name=auto, width=self.width, layer=self.layer)

    @property
    def layer_info(self) -> Any:
        from gdsfactory.pdk import get_layer_info

        return get_layer_info(self.layer)

    @property
    def direction(self) -> Array:
        """Unit vector pointing out of the port."""
        return xp.stack([cos_deg(self._orientation), sin_deg(self._orientation)])

    @property
    def normal(self) -> Array:
        """Unit vector along the port face (90 deg ccw from the direction)."""
        return xp.stack([-sin_deg(self._orientation), cos_deg(self._orientation)])

    @property
    def endpoints(self) -> Array:
        """The two corners of the port face."""
        half = asarray(self._width) / 2
        return xp.stack(
            [self._center - half * self.normal, self._center + half * self.normal]
        )

    @property
    def transform(self) -> Transform:
        return Transform(self.x, self.y, self._orientation, self.mirror, 1.0)

    @property
    def dcplx_trans(self) -> Any:
        return self.transform.to_klayout()

    @property
    def trans(self) -> Any:
        import klayout.db as kdb

        from gdsfactory.config import CONF

        dbu = 1e-3
        _ = CONF
        a = manhattan_angle(self._orientation) or 0
        return kdb.Trans(
            a // 90,
            self.mirror,
            round(to_float(self.x) / dbu),
            round(to_float(self.y) / dbu),
        )

    # -------------------------------------------------------------- methods
    def copy(
        self,
        trans: Transform | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Port:
        p = Port.__new__(Port)
        p.name = self.name if name is None else name
        p._center = self._center
        p._width = self._width
        p._orientation = self._orientation
        p.layer = self.layer
        p.port_type = self.port_type
        p.info = PortInfo(_copy.deepcopy(dict(self.info)))
        p.mirror = self.mirror
        for k, v in kwargs.items():
            setattr(p, k, v)
        if trans is not None:
            p.apply_transform(trans)
        return p

    def copy_polar(
        self, d: float = 0, d_orth: float = 0, angle: float = 2, mirror: bool = False
    ) -> Port:
        """Returns a copy moved d along the orientation and d_orth orthogonal.

        ``angle`` is the extra rotation in multiples of 90 degrees (2 flips it).
        """
        p = self.copy()
        p.center = self._center + d * self.direction + d_orth * self.normal
        p.orientation = self._orientation + 90 * angle
        p.mirror = self.mirror != mirror
        return p

    def apply_transform(self, t: Transform) -> Port:
        self._orientation = t.apply_angle(self._orientation)
        self._center = self._snap_center(t.apply(self._center))
        if not isinstance(t.magnification, int | float) or t.magnification != 1:
            self._width = self._width * t.magnification
        self.mirror = self.mirror != t.mirror
        return self

    def transformed(self, t: Transform) -> Port:
        return self.copy().apply_transform(t)

    def move(self, *args: Any) -> Port:
        """Moves in place: move((dx, dy)) or move(dx, dy) or move(origin, dest)."""
        dx, dy = _move_args(args)
        self._center = self._snap_center(self._center + xp.stack([asarray(dx), asarray(dy)]))
        return self

    def moved(self, *args: Any) -> Port:
        return self.copy().move(*args)

    dmove = move

    def rotate(self, angle: Any, center: Any = None) -> Port:
        c = self._center if center is None else asarray(center)
        t = Transform(0.0, 0.0, angle)
        self._orientation = t.apply_angle(self._orientation)
        self._center = self._snap_center(t.apply(self._center - c) + c)
        return self

    def flip(self) -> Port:
        self._orientation = xp.mod(asarray(self._orientation) + 180.0, 360.0)
        return self

    def flipped(self) -> Port:
        return self.copy().flip()

    def to_itype(self) -> Port:
        return self

    def to_dtype(self) -> Port:
        return self

    def to_dict(self) -> dict[str, Any]:
        from gdsfactory.pdk import get_layer_tuple

        try:
            layer: Any = get_layer_tuple(self.layer)
        except Exception:
            layer = self.layer
        return {
            "name": self.name,
            "center": [to_float(self.x), to_float(self.y)],
            "width": to_float(self.width),
            "orientation": to_float(self.orientation),
            "layer": layer,
            "port_type": self.port_type,
        }

    def is_close(self, other: Port, tol: float = 1e-3) -> bool:
        return (
            abs(to_float(self.x) - to_float(other.x)) < tol
            and abs(to_float(self.y) - to_float(other.y)) < tol
        )

    def __repr__(self) -> str:
        try:
            return (
                f"Port(name={self.name!r}, center=({to_float(self.x):.4g}, "
                f"{to_float(self.y):.4g}), width={to_float(self.width):.4g}, "
                f"orientation={to_float(self.orientation):.4g}, layer={self.layer}, "
                f"port_type={self.port_type!r})"
            )
        except Exception:
            return f"Port(name={self.name!r})"

    def print(self) -> None:
        print(repr(self))  # noqa: T201

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Port):
            return NotImplemented
        return (
            self.name == other.name
            and self.layer == other.layer
            and self.port_type == other.port_type
            and abs(to_float(self.width) - to_float(other.width)) < 1e-9
            and self.is_close(other, tol=1e-9)
            and abs(
                (to_float(self.orientation) - to_float(other.orientation) + 180) % 360
                - 180
            )
            < 1e-9
        )

    def __hash__(self) -> int:
        return id(self)


class Ports:
    """Ordered collection of Ports, indexable by name or position."""

    def __init__(self, ports: Iterable[Port] | None = None) -> None:
        self._ports: list[Port] = []
        for p in ports or []:
            self._ports.append(p)

    # -------------------------------------------------------------- container
    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports)

    def __len__(self) -> int:
        return len(self._ports)

    def __bool__(self) -> bool:
        return bool(self._ports)

    def __contains__(self, item: str | Port) -> bool:
        if isinstance(item, Port):
            return any(p is item for p in self._ports)
        return any(p.name == item for p in self._ports)

    def __getitem__(self, key: str | int | tuple[Any, ...]) -> Port:
        if isinstance(key, int):
            return self._ports[key]
        if isinstance(key, tuple):
            key = key[0]
        for p in self._ports:
            if p.name == key:
                return p
        raise KeyError(
            f"{key!r} not in {[p.name for p in self._ports]}",
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return [p.name for p in self._ports]  # type: ignore[misc]

    def values(self) -> list[Port]:
        return list(self._ports)

    def items(self) -> list[tuple[str, Port]]:
        return [(p.name, p) for p in self._ports]  # type: ignore[misc]

    def copy(self, trans: Transform | None = None) -> Ports:
        return Ports(p.copy(trans) for p in self._ports)

    def clear(self) -> None:
        self._ports.clear()

    @property
    def ports(self) -> list[Port]:
        return self._ports

    # ------------------------------------------------------------------ edits
    def add_port(
        self, port: Port, name: str | None = None, keep_mirror: bool = True
    ) -> Port:
        p = port.copy()
        if name is not None:
            p.name = name
        if not keep_mirror:
            p.mirror = False
        self._ports.append(p)
        return p

    def append(self, port: Port) -> None:
        self._ports.append(port)

    def add_ports(
        self,
        ports: Iterable[Port],
        prefix: str = "",
        suffix: str = "",
        keep_mirror: bool = True,
    ) -> None:
        for p in ports:
            name = f"{prefix}{p.name}{suffix}"
            self.add_port(p, name=name, keep_mirror=keep_mirror)

    def create_port(
        self,
        *,
        name: str | None = None,
        width: Any = None,
        layer: Any = None,
        port_type: str = "optical",
        center: Any = None,
        orientation: Any = None,
        dcplx_trans: Any = None,
        trans: Any = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Port:
        if dcplx_trans is not None:
            center = (dcplx_trans.disp.x, dcplx_trans.disp.y)
            orientation = dcplx_trans.angle
        if trans is not None:
            center = (trans.disp.x * 1e-3, trans.disp.y * 1e-3)
            orientation = trans.angle * 90
        p = Port(
            name=name,
            center=center if center is not None else (0, 0),
            width=width,
            orientation=orientation or 0,
            layer=layer,
            port_type=port_type,
            info=info,
        )
        self._ports.append(p)
        return p

    def remove(self, name: str) -> None:
        self._ports = [p for p in self._ports if p.name != name]

    def sort(self, key: Callable[[Port], Any] | None = None, reverse: bool = False) -> None:
        self._ports.sort(key=key or (lambda p: p.name), reverse=reverse)  # type: ignore[arg-type]

    def reverse(self) -> None:
        self._ports.reverse()

    def update(self, other: dict[str, Port] | Ports) -> None:
        items = other.items()
        for name, p in items:
            self.remove(name)
            q = p.copy()
            q.name = name
            self._ports.append(q)

    # ---------------------------------------------------------------- queries
    def filter(
        self,
        angle: int | None = None,
        orientation: float | None = None,
        layer: Any = None,
        port_type: str | None = None,
        regex: str | None = None,
    ) -> list[Port]:
        return filter_ports(
            self._ports,
            angle=angle,
            orientation=orientation,
            layer=layer,
            port_type=port_type,
            regex=regex,
        )

    def print(self, unit: str | None = None) -> None:
        from gdsfactory.port import pprint_ports

        pprint_ports(self)

    def __repr__(self) -> str:
        return f"Ports({[p.name for p in self._ports]})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ports):
            other_list = list(other)
        elif isinstance(other, list):
            other_list = other
        else:
            return NotImplemented
        return len(other_list) == len(self._ports) and all(
            a == b for a, b in zip(self._ports, other_list, strict=True)
        )

    __hash__ = None  # type: ignore[assignment]


class Pin:
    """Logical group of ports (e.g. an electrical net with several ports)."""

    def __init__(
        self,
        name: str | None,
        ports: Sequence[Port],
        pin_type: str = "DC",
        info: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.ports = list(ports)
        self.pin_type = pin_type
        self.info = PortInfo(info or {})

    def copy(self, trans: Transform | None = None) -> Pin:
        return Pin(self.name, [p.copy(trans) for p in self.ports], self.pin_type, dict(self.info))

    def __repr__(self) -> str:
        return f"Pin({self.name!r}, ports={[p.name for p in self.ports]}, pin_type={self.pin_type!r})"


class Pins(list[Pin]):
    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            for p in self:
                if p.name == key:
                    return p
            raise KeyError(key)
        return super().__getitem__(key)


def filter_ports(
    ports: Iterable[Port],
    angle: int | None = None,
    orientation: float | None = None,
    layer: Any = None,
    port_type: str | None = None,
    regex: str | None = None,
) -> list[Port]:
    out = list(ports)
    if regex:
        pattern = re.compile(regex)
        out = [p for p in out if p.name and pattern.match(p.name)]
    if layer is not None:
        from gdsfactory.pdk import get_layer

        lay = get_layer(layer)
        out = [p for p in out if p.layer == lay]
    if port_type:
        out = [p for p in out if p.port_type == port_type]
    if angle is not None:
        out = [p for p in out if manhattan_angle(p.orientation) == (angle % 4) * 90]
    if orientation is not None:
        o = float(orientation) % 360
        out = [
            p
            for p in out
            if abs((to_float(p.orientation) - o + 180) % 360 - 180) < 1e-6
        ]
    return out


def _layer_enum(layer: Any) -> Any:
    """Layer index as the PDK LayerEnum member when one exists (kfactory semantics)."""
    if isinstance(layer, int) and not hasattr(layer, "layer"):
        try:
            import kfactory as kf

            return kf.kcl.layers(layer)  # type: ignore[call-arg]
        except Exception:
            return layer
    return layer


def _point(value: Any) -> Array:
    if hasattr(value, "x") and hasattr(value, "y") and not isinstance(value, Port):
        value = (value.x, value.y)
    if isinstance(value, Port):
        return value.center_array
    if is_tracer(value) or hasattr(value, "shape"):
        arr = asarray(value)
    elif isinstance(value, Sequence) and any(is_tracer(v) for v in value):
        arr = xp.stack([asarray(value[0]), asarray(value[1])])
    else:
        arr = asarray(np.asarray([to_float(value[0]), to_float(value[1])]))
    if arr.shape != (2,):
        raise ValueError(f"Port center must be (x, y), got shape {arr.shape}")
    return arr


def _angle(value: Any) -> Any:
    if value is None:
        return None
    if is_tracer(value) or hasattr(value, "shape"):
        return xp.mod(asarray(value), 360.0)
    return float(value) % 360


def _move_args(args: tuple[Any, ...]) -> tuple[Any, Any]:
    if len(args) == 1:
        v = args[0]
        if hasattr(v, "x") and hasattr(v, "y"):
            return v.x, v.y
        return v[0], v[1]
    if len(args) == 2:
        a, b = args
        if _is_point(a) and _is_point(b):
            pa = _point(a)
            pb = _point(b)
            return pb[0] - pa[0], pb[1] - pa[1]
        return a, b
    raise TypeError(f"move takes 1 or 2 arguments, got {len(args)}")


def _is_point(v: Any) -> bool:
    if isinstance(v, Port):
        return True
    if hasattr(v, "shape"):
        return tuple(v.shape) == (2,)
    return isinstance(v, Sequence) and not isinstance(v, str) and len(v) == 2


__all__ = ["Pin", "Pins", "Port", "PortInfo", "Ports", "filter_ports"]
