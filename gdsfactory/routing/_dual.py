"""Dual integer geometry types for tracing kfactory's manhattan router with JAX.

Every coordinate is a :class:`DNum`: the exact integer (dbu) that kfactory's
router computes (``v``) and, optionally, a JAX value (``t``) whose primal is
always exactly ``v`` and which carries the derivative with respect to the
traced inputs.

All comparisons, hashing, ordering and branching use ``v``, so code written for
``klayout.db`` integer types makes exactly the same decisions as kfactory.
Arithmetic propagates ``t``; integer-only operations (floor division, rounding,
truncation) are straight-through (value as kfactory, derivative of the real
valued operation).

The :class:`Point`, :class:`Vector`, :class:`Trans`, :class:`Box` and
:class:`Edge` classes mirror the subset of the ``klayout.db`` API used by
kfactory's routing code.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import klayout.db as kdb
import numpy as np


def _is_jax(x: Any) -> bool:
    return isinstance(x, jax.Array | jax.core.Tracer)


def _st(t: Any, v: float) -> Any:
    """Value ``v`` with the derivative of ``t`` (straight-through)."""
    return t + jax.lax.stop_gradient(v - t)


class DNum:
    """Exact integer (or float) value ``v`` with an optional JAX value ``t``."""

    __slots__ = ("t", "v")

    def __init__(self, v: Any, t: Any = None) -> None:
        if isinstance(v, DNum):
            v, t = v.v, v.t
        self.v = v
        self.t = t

    # ---------------------------------------------------------- helpers
    @staticmethod
    def of(x: Any) -> DNum:
        return x if isinstance(x, DNum) else DNum(x)

    @staticmethod
    def traced(t: Any, v: Any) -> DNum:
        """DNum with value v whose JAX value has primal v and derivative of t."""
        if t is None or not _is_jax(t):
            return DNum(v)
        return DNum(v, _st(t, v))

    def tv(self) -> Any:
        """JAX value if traced, else the concrete value."""
        return self.v if self.t is None else self.t

    @property
    def is_traced(self) -> bool:
        return self.t is not None

    def _bin(self, other: Any, op: Any) -> DNum:
        o = DNum.of(other)
        v = op(self.v, o.v)
        if self.t is None and o.t is None:
            return DNum(v)
        return DNum.traced(op(self.tv(), o.tv()), v)

    # ------------------------------------------------------- arithmetic
    def __add__(self, other: Any) -> DNum:
        return self._bin(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other: Any) -> DNum:
        return self._bin(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> DNum:
        return DNum.of(other)._bin(self, lambda a, b: a - b)

    def __mul__(self, other: Any) -> DNum:
        return self._bin(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __neg__(self) -> DNum:
        return DNum(-self.v, None if self.t is None else -self.t)

    def __pos__(self) -> DNum:
        return self

    def __abs__(self) -> DNum:
        return DNum(abs(self.v), None if self.t is None else jax.numpy.abs(self.t))

    def __floordiv__(self, other: Any) -> DNum:
        o = DNum.of(other)
        v = self.v // o.v
        if self.t is None and o.t is None:
            return DNum(v)
        return DNum.traced(self.tv() / o.tv(), v)

    def __rfloordiv__(self, other: Any) -> DNum:
        return DNum.of(other).__floordiv__(self)

    def __truediv__(self, other: Any) -> DNum:
        return self._bin(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> DNum:
        return DNum.of(other)._bin(self, lambda a, b: a / b)

    def __mod__(self, other: Any) -> DNum:
        o = DNum.of(other)
        v = self.v % o.v
        # x % n has derivative 1 w.r.t. x (piecewise)
        if self.t is None:
            return DNum(v)
        return DNum.traced(self.t, v)

    def __round__(self, ndigits: int | None = None) -> DNum:
        v = round(self.v, ndigits) if ndigits is not None else round(self.v)
        return DNum.traced(self.t, v)

    def __trunc__(self) -> DNum:
        return DNum.traced(self.t, math.trunc(self.v))

    def __floor__(self) -> DNum:
        return DNum.traced(self.t, math.floor(self.v))

    def __ceil__(self) -> DNum:
        return DNum.traced(self.t, math.ceil(self.v))

    # ------------------------------------------------------ comparisons
    def __eq__(self, other: object) -> bool:
        if isinstance(other, DNum):
            return bool(self.v == other.v)
        if isinstance(other, int | float | np.integer | np.floating):
            return bool(self.v == other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __lt__(self, other: Any) -> bool:
        return bool(self.v < DNum.of(other).v)

    def __le__(self, other: Any) -> bool:
        return bool(self.v <= DNum.of(other).v)

    def __gt__(self, other: Any) -> bool:
        return bool(self.v > DNum.of(other).v)

    def __ge__(self, other: Any) -> bool:
        return bool(self.v >= DNum.of(other).v)

    def __hash__(self) -> int:
        return hash(self.v)

    def __bool__(self) -> bool:
        return bool(self.v)

    def __int__(self) -> int:
        return int(self.v)

    def __index__(self) -> int:
        return int(self.v)

    def __float__(self) -> float:
        return float(self.v)

    def __repr__(self) -> str:
        return f"{self.v}{'*' if self.t is not None else ''}"

    __str__ = __repr__


def dint(x: Any) -> DNum:
    """int() truncation keeping derivative (straight-through)."""
    x = DNum.of(x)
    return DNum.traced(x.t, int(x.v))


def dmax(*args: Any, key: Any = None) -> Any:
    """max() that keeps the selected DNum (and thus its derivative)."""
    return max(*args, key=key) if key is not None else max(*args)


# ---------------------------------------------------------------------------
# Points and vectors
# ---------------------------------------------------------------------------
class _XY:
    __slots__ = ("_x", "_y")

    def __init__(self, x: Any = 0, y: Any = 0) -> None:
        if isinstance(x, _XY) and y == 0:
            x, y = x.x, x.y
        elif isinstance(x, kdb.Point | kdb.Vector):
            x, y = x.x, x.y
        self._x = DNum.of(x)
        self._y = DNum.of(y)

    @property
    def x(self) -> DNum:
        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        self._x = DNum.of(value)

    @property
    def y(self) -> DNum:
        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        self._y = DNum.of(value)

    def _key(self) -> tuple[Any, Any]:
        return (self._x.v, self._y.v)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))

    def __repr__(self) -> str:
        return f"{self._x},{self._y}"

    def to_s(self) -> str:
        return f"{self._x.v},{self._y.v}"

    __str__ = to_s

    def __iter__(self) -> Any:
        return iter((self._x, self._y))


class Vector(_XY):
    """Dual kdb.Vector."""

    def to_kdb(self) -> kdb.Vector:
        return kdb.Vector(int(self._x.v), int(self._y.v))

    def dup(self) -> Vector:
        return Vector(self._x, self._y)

    def to_p(self) -> Point:
        return Point(self._x, self._y)

    def to_v(self) -> Vector:
        return self

    def __add__(self, other: Any) -> Any:
        if isinstance(other, Point):
            return Point(self._x + other.x, self._y + other.y)
        return Vector(self._x + other.x, self._y + other.y)

    def __sub__(self, other: Any) -> Vector:
        return Vector(self._x - other.x, self._y - other.y)

    def __neg__(self) -> Vector:
        return Vector(-self._x, -self._y)

    def __mul__(self, f: Any) -> Vector:
        return Vector(self._x * f, self._y * f)

    __rmul__ = __mul__

    def __iadd__(self, other: Any) -> Vector:
        self._x = self._x + other.x
        self._y = self._y + other.y
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vector | kdb.Vector):
            ox = other.x.v if isinstance(other.x, DNum) else other.x
            oy = other.y.v if isinstance(other.y, DNum) else other.y
            return self._key() == (ox, oy)
        return NotImplemented

    __hash__ = _XY.__hash__

    def length(self) -> DNum:
        v = math.hypot(self._x.v, self._y.v)
        if not (self._x.is_traced or self._y.is_traced) or v == 0:
            return DNum(v)
        t = jax.numpy.sqrt(self._x.tv() ** 2 + self._y.tv() ** 2)
        return DNum.traced(t, v)

    abs = length

    def sq_length(self) -> DNum:
        return self._x * self._x + self._y * self._y

    def __lt__(self, other: Any) -> bool:
        return self.to_kdb() < _concrete(other)


class Point(_XY):
    """Dual kdb.Point."""

    def to_kdb(self) -> kdb.Point:
        return kdb.Point(int(self._x.v), int(self._y.v))

    def dup(self) -> Point:
        return Point(self._x, self._y)

    def to_v(self) -> Vector:
        return Vector(self._x, self._y)

    def to_p(self) -> Point:
        return self

    def __add__(self, other: Any) -> Point:
        return Point(self._x + other.x, self._y + other.y)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Vector):
            return Point(self._x - other.x, self._y - other.y)
        return Vector(self._x - other.x, self._y - other.y)

    def __neg__(self) -> Point:
        return Point(-self._x, -self._y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Point | kdb.Point):
            ox = other.x.v if isinstance(other.x, DNum) else other.x
            oy = other.y.v if isinstance(other.y, DNum) else other.y
            return self._key() == (ox, oy)
        return NotImplemented

    __hash__ = _XY.__hash__

    def __lt__(self, other: Any) -> bool:
        return self.to_kdb() < _concrete(other)

    def distance(self, other: Point) -> DNum:
        return (self - other).length()


def _concrete(o: Any) -> Any:
    return o.to_kdb() if hasattr(o, "to_kdb") else o


# ---------------------------------------------------------------------------
# Simple transformation (90 degree rotations, mirror, integer displacement)
# ---------------------------------------------------------------------------
def _rot(k: int, x: DNum, y: DNum) -> tuple[DNum, DNum]:
    k %= 4
    if k == 0:
        return x, y
    if k == 1:
        return -y, x
    if k == 2:
        return -x, -y
    return y, -x


class Trans:
    """Dual kdb.Trans: p' = disp + R(rot) M(mirror) p."""

    __slots__ = ("_disp", "_mirror", "_rot")

    R0: Trans
    R90: Trans
    R180: Trans
    R270: Trans
    M0: Trans
    M45: Trans
    M90: Trans
    M135: Trans

    def __init__(self, *args: Any) -> None:
        rot, mirror, disp = 0, False, Vector(0, 0)
        if len(args) == 0:
            pass
        elif len(args) == 1:
            a = args[0]
            if isinstance(a, Trans):
                rot, mirror, disp = a._rot, a._mirror, a._disp.dup()
            elif isinstance(a, kdb.Trans):
                rot, mirror, disp = a.angle, a.is_mirror(), Vector(a.disp.x, a.disp.y)
            elif isinstance(a, _XY | kdb.Vector | kdb.Point):
                disp = Vector(a.x, a.y)
            else:
                raise TypeError(f"Trans({a!r})")
        elif len(args) == 2:
            a, b = args
            disp = Vector(a, b)
        elif len(args) == 3:
            rot, mirror, v = args
            disp = Vector(v.x, v.y)
        elif len(args) == 4:
            rot, mirror, x, y = args
            disp = Vector(x, y)
        else:
            raise TypeError(f"Trans{args!r}")
        self._rot = int(rot) % 4
        self._mirror = bool(mirror)
        self._disp = disp

    @classmethod
    def from_kdb(cls, t: kdb.Trans) -> Trans:
        return cls(t.angle, t.is_mirror(), t.disp.x, t.disp.y)

    def to_kdb(self) -> kdb.Trans:
        return kdb.Trans(self._rot, self._mirror, int(self._disp.x.v), int(self._disp.y.v))

    # ------------------------------------------------------ properties
    @property
    def angle(self) -> int:
        return self._rot

    @angle.setter
    def angle(self, value: int) -> None:
        self._rot = int(value) % 4

    @property
    def rot(self) -> int:
        return self._rot + (4 if self._mirror else 0)

    @property
    def mirror(self) -> bool:
        return self._mirror

    @mirror.setter
    def mirror(self, value: bool) -> None:
        self._mirror = bool(value)

    def is_mirror(self) -> bool:
        return self._mirror

    @property
    def disp(self) -> Vector:
        return self._disp

    @disp.setter
    def disp(self, v: Any) -> None:
        self._disp = Vector(v.x, v.y)

    def dup(self) -> Trans:
        return Trans(self._rot, self._mirror, self._disp.x, self._disp.y)

    def assign(self, other: Trans) -> None:
        self._rot, self._mirror, self._disp = other._rot, other._mirror, other._disp.dup()

    # --------------------------------------------------------- algebra
    def _apply(self, x: DNum, y: DNum) -> tuple[DNum, DNum]:
        if self._mirror:
            y = -y
        return _rot(self._rot, x, y)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, Trans):
            r2 = -other._rot if self._mirror else other._rot
            dx, dy = self._apply(other._disp.x, other._disp.y)
            return Trans(
                (self._rot + r2) % 4,
                self._mirror != other._mirror,
                self._disp.x + dx,
                self._disp.y + dy,
            )
        if isinstance(other, kdb.Trans):
            return self * Trans.from_kdb(other)
        if isinstance(other, Point | kdb.Point):
            x, y = self._apply(DNum.of(other.x), DNum.of(other.y))
            return Point(x + self._disp.x, y + self._disp.y)
        if isinstance(other, Vector | kdb.Vector):
            x, y = self._apply(DNum.of(other.x), DNum.of(other.y))
            return Vector(x, y)
        if isinstance(other, Box):
            return other.transformed(self)
        return NotImplemented

    def __imul__(self, other: Trans) -> Trans:
        self.assign(self * other)
        return self

    def trans(self, p: Any) -> Any:
        return self * p

    def inverted(self) -> Trans:
        # p = M^-1 R^-1 (q - d)
        r = self._rot if self._mirror else (-self._rot) % 4
        t = Trans(r, self._mirror, 0, 0)
        d = t * self._disp
        return Trans(r, self._mirror, -d.x, -d.y)

    def invert(self) -> Trans:
        self.assign(self.inverted())
        return self

    # ------------------------------------------------- identity / order
    def _key(self) -> tuple[Any, ...]:
        return (self._rot, self._mirror, self._disp.x.v, self._disp.y.v)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Trans):
            return self._key() == other._key()
        if isinstance(other, kdb.Trans):
            return self.to_kdb() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._key())

    def __lt__(self, other: Any) -> bool:
        return self.to_kdb() < _concrete(other)

    def to_s(self) -> str:
        return self.to_kdb().to_s()

    __str__ = to_s

    def __repr__(self) -> str:
        return f"Trans({self._rot}, {self._mirror}, {self._disp!r})"

    def to_dtype(self, dbu: float) -> kdb.DTrans:
        return self.to_kdb().to_dtype(dbu)


Trans.R0 = Trans(0, False, 0, 0)
Trans.R90 = Trans(1, False, 0, 0)
Trans.R180 = Trans(2, False, 0, 0)
Trans.R270 = Trans(3, False, 0, 0)
Trans.M0 = Trans(0, True, 0, 0)
Trans.M45 = Trans(1, True, 0, 0)
Trans.M90 = Trans(2, True, 0, 0)
Trans.M135 = Trans(3, True, 0, 0)


# ---------------------------------------------------------------------------
# Boxes and edges
# ---------------------------------------------------------------------------
class Box:
    """Dual kdb.Box (empty boxes behave like KLayout's)."""

    __slots__ = ("_b", "_empty", "_l", "_r", "_t")

    def __init__(self, *args: Any) -> None:
        self._empty = False
        if len(args) == 0:
            self._empty = True
            self._l = self._b = DNum(1)
            self._r = self._t = DNum(-1)
            return
        if len(args) == 1:
            a = args[0]
            if isinstance(a, Box):
                self._empty = a._empty
                self._l, self._b, self._r, self._t = a._l, a._b, a._r, a._t
                return
            if isinstance(a, kdb.Box):
                if a.empty():
                    self.__init__()  # type: ignore[misc]
                    return
                self._l, self._b, self._r, self._t = (
                    DNum(a.left),
                    DNum(a.bottom),
                    DNum(a.right),
                    DNum(a.top),
                )
                return
            # square box of size a centered at the origin
            w = DNum.of(a)
            h = w // 2
            self._l, self._b, self._r, self._t = -h, -h, w - h, w - h
            return
        if len(args) == 2:
            p1, p2 = args
            self._set(DNum.of(p1.x), DNum.of(p1.y), DNum.of(p2.x), DNum.of(p2.y))
            return
        if len(args) == 4:
            self._set(*(DNum.of(a) for a in args))
            return
        raise TypeError(f"Box{args!r}")

    def _set(self, l: DNum, b: DNum, r: DNum, t: DNum) -> None:
        self._l, self._r = (l, r) if l <= r else (r, l)
        self._b, self._t = (b, t) if b <= t else (t, b)

    @classmethod
    def _from(cls, l: DNum, b: DNum, r: DNum, t: DNum) -> Box:
        box = cls.__new__(cls)
        box._empty = False
        box._l, box._b, box._r, box._t = l, b, r, t
        return box

    def to_kdb(self) -> kdb.Box:
        if self._empty:
            return kdb.Box()
        return kdb.Box(int(self._l.v), int(self._b.v), int(self._r.v), int(self._t.v))

    def dup(self) -> Box:
        return Box(self)

    def empty(self) -> bool:
        return self._empty

    @property
    def left(self) -> DNum:
        return self._l

    @property
    def right(self) -> DNum:
        return self._r

    @property
    def bottom(self) -> DNum:
        return self._b

    @property
    def top(self) -> DNum:
        return self._t

    @property
    def p1(self) -> Point:
        return Point(self._l, self._b)

    @property
    def p2(self) -> Point:
        return Point(self._r, self._t)

    def width(self) -> DNum:
        return self._r - self._l

    def height(self) -> DNum:
        return self._t - self._b

    def center(self) -> Point:
        c = self.to_kdb().center()  # KLayout's integer rounding
        return Point(
            DNum.traced((self._l.tv() + self._r.tv()) / 2 if (self._l.is_traced or self._r.is_traced) else None, c.x),
            DNum.traced((self._b.tv() + self._t.tv()) / 2 if (self._b.is_traced or self._t.is_traced) else None, c.y),
        )

    def contains(self, p: Any) -> bool:
        return self.to_kdb().contains(_concrete(p) if not isinstance(p, kdb.Point) else p)

    def inside(self, other: Any) -> bool:
        return self.to_kdb().inside(_concrete(other))

    def overlaps(self, other: Any) -> bool:
        return self.to_kdb().overlaps(_concrete(other))

    def touches(self, other: Any) -> bool:
        return self.to_kdb().touches(_concrete(other))

    # -------------------------------------------------------- algebra
    def __add__(self, other: Any) -> Box:
        if isinstance(other, kdb.Box):
            other = Box(other)
        if isinstance(other, Point | kdb.Point):
            other = Box._from(DNum.of(other.x), DNum.of(other.y), DNum.of(other.x), DNum.of(other.y))
        if isinstance(other, kdb.DBox):
            raise TypeError("cannot add DBox to an integer Box")
        if other.empty():
            return self.dup()
        if self._empty:
            return other.dup()
        return Box._from(
            min(self._l, other._l),
            min(self._b, other._b),
            max(self._r, other._r),
            max(self._t, other._t),
        )

    def __iadd__(self, other: Any) -> Box:
        res = self + other
        self._empty, self._l, self._b, self._r, self._t = (
            res._empty,
            res._l,
            res._b,
            res._r,
            res._t,
        )
        return self

    def __and__(self, other: Any) -> Box:
        if isinstance(other, kdb.Box):
            other = Box(other)
        k = self.to_kdb() & other.to_kdb()
        if k.empty():
            return Box()
        return Box._from(
            max(self._l, other._l),
            max(self._b, other._b),
            min(self._r, other._r),
            min(self._t, other._t),
        )

    def enlarged(self, *args: Any) -> Box:
        if self._empty:
            return Box()
        if len(args) == 1:
            a = args[0]
            if isinstance(a, _XY):
                dx, dy = a.x, a.y
            else:
                dx = dy = DNum.of(a)
        else:
            dx, dy = DNum.of(args[0]), DNum.of(args[1])
        return Box._from(self._l - dx, self._b - dy, self._r + dx, self._t + dy)

    def enlarge(self, *args: Any) -> Box:
        res = self.enlarged(*args)
        self._empty, self._l, self._b, self._r, self._t = (
            res._empty,
            res._l,
            res._b,
            res._r,
            res._t,
        )
        return self

    def moved(self, v: Any) -> Box:
        if self._empty:
            return Box()
        return Box._from(self._l + v.x, self._b + v.y, self._r + v.x, self._t + v.y)

    def transformed(self, t: Trans) -> Box:
        if self._empty:
            return Box()
        p1 = t * Point(self._l, self._b)
        p2 = t * Point(self._r, self._t)
        return Box(p1, p2)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Box | kdb.Box):
            return self.to_kdb() == _concrete(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.to_kdb().to_s())

    def __lt__(self, other: Any) -> bool:
        return self.to_kdb() < _concrete(other)

    def __repr__(self) -> str:
        return f"Box({self.to_kdb().to_s()})"

    def to_s(self) -> str:
        return self.to_kdb().to_s()


class Edge:
    """Dual kdb.Edge (only what the router needs)."""

    __slots__ = ("p1", "p2")

    def __init__(self, p1: Any, p2: Any) -> None:
        self.p1 = Point(p1.x, p1.y)
        self.p2 = Point(p2.x, p2.y)

    def to_kdb(self) -> kdb.Edge:
        return kdb.Edge(self.p1.to_kdb(), self.p2.to_kdb())

    def shifted(self, d: Any) -> Edge:
        """Shifts perpendicular to the edge (positive: to the left), manhattan only."""
        v = self.p2 - self.p1
        if v.x == 0 and v.y == 0:
            return Edge(self.p1, self.p2)
        if v.y == 0:
            s = 1 if v.x > 0 else -1
            off = Vector(0, d * s)
        elif v.x == 0:
            s = 1 if v.y > 0 else -1
            off = Vector(-(d * s), 0)
        else:
            # non-manhattan: KLayout rounding
            k = self.to_kdb().shifted(int(DNum.of(d).v))
            return Edge(Point(k.p1.x, k.p1.y), Point(k.p2.x, k.p2.y))
        return Edge(self.p1 + off, self.p2 + off)

    def contains(self, p: Any) -> bool:
        return self.to_kdb().contains(_concrete(p))


def to_kdb_points(pts: list[Any]) -> list[kdb.Point]:
    return [p.to_kdb() if hasattr(p, "to_kdb") else p for p in pts]


__all__ = ["Box", "DNum", "Edge", "Point", "Trans", "Vector", "dint", "to_kdb_points"]
