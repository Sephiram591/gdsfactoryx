"""Opt-in wall-clock timing of the steps that build a :class:`~gdsfactory.fdtdx_stack.StackDevice`.

Off by default. When off, every timed call costs one global lookup and adds
nothing to traced code. Turn it on with :func:`record_timings`::

    with record_timings() as timings:
        device = StackDevice.from_layer_stack(...)
        objects, arrays, *_ = fdtdx.place_objects(...)
        arrays, *_ = apply_stack(arrays, objects, params, key)
    print(timings.table())

Steps nest (``write/weights/source_sdfs``). Each record is classified:

- **traced:** the step ran while JAX traced a function for ``jax.jit`` (or
  ``lax.scan`` and the like): its wall time is tracing time, which is part of
  compiling.
- **eager:** the step ran on concrete values, also under eager autodiff
  (``jax.grad`` or ``jax.value_and_grad`` without ``jit``, whose tracers carry
  concrete primals). Its outputs (their primals) are waited for
  (``jax.block_until_ready``) at the step's end, outside any trace, so the
  wall time includes the device work. JAX compile events (tracing, lowering
  and XLA compilation, from ``jax.monitoring``) that happen while the step is
  open are its compile time; the rest is run time. The backward pass of an
  eager ``jax.grad`` runs after the steps returned and belongs to no step: time
  the whole call with :func:`timed_step` to see it.

Work outside every step is not lost: compile events that fire while no step is
open are summed in :attr:`StepTimings.outside_compile` (lowering and XLA
compilation of jitted callers), and the timing plugin reports each test's time
outside the steps.

Timing never changes results or gradients: it only reads the clock, waits for
eager outputs, and listens to JAX's monitoring events.
"""

from __future__ import annotations

import contextlib
import functools
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import jax

__all__ = ["StepRecord", "StepTimings", "record_timings", "timed", "timed_step", "lap"]

_F = TypeVar("_F", bound=Callable[..., Any])

#: JAX monitoring events that count as compile time (their intervals are merged, so nested ones count once)
COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)


@dataclass
class StepRecord:
    """Accumulated time of one step (one key of :attr:`StepTimings.steps`)."""

    #: calls of the step
    calls: int = 0
    #: calls made while JAX traced (their wall time is tracing time)
    traced_calls: int = 0
    #: total wall time (s), eager and traced calls
    wall: float = 0.0
    #: wall time (s) of the traced calls
    traced: float = 0.0
    #: JAX compile time (s) inside the eager calls (merged compile events)
    compile: float = 0.0

    @property
    def run(self) -> float:
        """Eager wall time minus the compile time inside it (s)."""
        return max(self.wall - self.traced - self.compile, 0.0)

    def add(self, other: StepRecord) -> None:
        """Adds another record's times and counts to this one."""
        self.calls += other.calls
        self.traced_calls += other.traced_calls
        self.wall += other.wall
        self.traced += other.traced
        self.compile += other.compile

    def as_dict(self) -> dict[str, float]:
        """The record as a JSON-ready dict (run included)."""
        return {
            "calls": self.calls,
            "traced_calls": self.traced_calls,
            "wall": self.wall,
            "traced": self.traced,
            "compile": self.compile,
            "run": self.run,
        }


@dataclass
class _Open:
    """A step being timed: its key, start time, compile intervals and laps."""

    key: str
    start: float
    lap: float
    intervals: list[tuple[float, float]] = field(default_factory=list)


class StepTimings:
    """Records the wall, compile and run time of named steps (see the module docstring)."""

    def __init__(self) -> None:
        """Creates an empty recorder."""
        #: per step key ("parent/child"), the accumulated record
        self.steps: dict[str, StepRecord] = {}
        #: compile events (start, end) that fired while no step was open
        self.outside: list[tuple[float, float]] = []
        self._stack: list[_Open] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------ recording

    def _on_event(self, event: str, duration: float, **_: Any) -> None:
        """jax.monitoring listener: a compile interval for every open step."""
        if event not in COMPILE_EVENTS:
            return
        end = time.perf_counter()
        if not self._stack:
            with self._lock:
                self.outside.append((end - duration, end))
            return
        with self._lock:
            for frame in self._stack:
                frame.intervals.append((end - duration, end))

    def open(self, name: str) -> _Open:
        """Starts a step nested in the innermost open one."""
        key = f"{self._stack[-1].key}/{name}" if self._stack else name
        now = time.perf_counter()
        frame = _Open(key, now, now)
        self._stack.append(frame)
        return frame

    def close(self, frame: _Open, value: Any = None) -> None:
        """Ends a step: waits for its eager ``value`` and records the time."""
        traced = _wait(value)
        end = time.perf_counter()
        if frame in self._stack:  # also drops steps an exception left open inside it
            del self._stack[self._stack.index(frame):]
        self._record(frame.key, frame.start, end, frame.intervals, traced)

    def lap(self, name: str, value: Any = None) -> None:
        """Records the time since the innermost step's start or last lap as its sub-step ``<step>.<name>``."""
        if not self._stack:
            return
        frame = self._stack[-1]
        traced = _wait(value)
        end = time.perf_counter()
        start, frame.lap = frame.lap, end
        inside = [(a, b) for a, b in frame.intervals if b > start]
        own = frame.key.rsplit("/", 1)[-1]
        self._record(f"{frame.key}/{own}.{name}", start, end, inside, traced)

    def _record(self, key: str, start: float, end: float, intervals: list[tuple[float, float]], traced: bool) -> None:
        """Adds one call of ``key`` to its record."""
        rec = self.steps.setdefault(key, StepRecord())
        rec.calls += 1
        rec.wall += end - start
        if traced:
            rec.traced_calls += 1
            rec.traced += end - start
        else:
            rec.compile += min(_merged_length(intervals, start, end), end - start)

    @property
    def outside_compile(self) -> float:
        """Compile time (s) outside every step: lowering and XLA compilation of jitted callers (merged events)."""
        if not self.outside:
            return 0.0
        return _merged_length(self.outside, min(a for a, _ in self.outside), max(b for _, b in self.outside))

    # ------------------------------------------------------------ reporting

    def by_leaf(self) -> dict[str, StepRecord]:
        """Records summed per step name, whatever outer steps called it.

        ``apply_stack/write/weights`` and ``write/weights`` both count as
        ``weights``; a lap ``name`` of step ``step`` is ``step.name``.
        """
        out: dict[str, StepRecord] = {}
        for key, rec in self.steps.items():
            out.setdefault(key.rsplit("/", 1)[-1], StepRecord()).add(rec)
        return out

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Every step key's record as a JSON-ready dict."""
        return {k: v.as_dict() for k, v in sorted(self.steps.items())}

    def table(self, leaf: bool = True) -> str:
        """A text table of the steps (by leaf name with ``leaf``), slowest first."""
        rows = self.by_leaf() if leaf else self.steps
        return format_table(rows)


def format_table(rows: dict[str, StepRecord], title: str = "step") -> str:
    """A text table of records: calls, traced calls, wall, traced, compile, run (s), slowest first."""
    width = max([len(title), *(len(k) for k in rows)])
    head = f"{title:<{width}}  {'calls':>6} {'traced':>6} {'wall s':>9} {'trace s':>9} {'compile s':>9} {'run s':>9}"
    lines = [head, "-" * len(head)]
    for key, r in sorted(rows.items(), key=lambda kv: -kv[1].wall):
        lines.append(
            f"{key:<{width}}  {r.calls:>6d} {r.traced_calls:>6d} {r.wall:>9.2f} {r.traced:>9.2f} "
            f"{r.compile:>9.2f} {r.run:>9.2f}"
        )
    return "\n".join(lines)


def _merged_length(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    """Length of the union of ``intervals`` clipped to [start, end]."""
    total, reach = 0.0, start
    for a, b in sorted(intervals):
        a, b = max(a, reach), min(b, end)
        if b > a:
            total += b - a
            reach = b
    return total


def _wait(value: Any) -> bool:
    """Waits for the eager arrays in ``value``; True if it holds jit tracers or JAX is tracing for jit.

    Nothing is waited for under a jit trace, so no host sync enters traced code.
    Tracers of eager autodiff carry concrete primals, which are waited for.
    """
    leaves = jax.tree_util.tree_leaves(value)
    if _jit_tracing() or any(_is_jit_tracer(x) for x in leaves):
        return True
    arrays = [x for x in map(_concrete, leaves) if isinstance(x, jax.Array)]
    if arrays:
        jax.block_until_ready(arrays)
    return False


def _concrete(x: Any) -> Any:
    """The concrete primal under eager-autodiff tracers (x itself otherwise)."""
    while isinstance(x, jax.core.Tracer) and hasattr(x, "primal"):
        x = x.primal
    return x


def _is_jit_tracer(x: Any) -> bool:
    """True for a tracer without a concrete primal (jit / scan tracing)."""
    return isinstance(_concrete(x), jax.core.Tracer)


def _jit_tracing() -> bool:
    """True while JAX traces a function into a jaxpr (jit, scan, ...), also under autodiff of it.

    Eager autodiff alone (``jax.grad`` without ``jit``) is not tracing: its steps run on concrete values.
    """
    try:
        from jax._src import core

        trace = core.trace_ctx.trace
        while trace is not None:
            if type(trace).__name__ == "DynamicJaxprTrace":
                return True
            trace = getattr(trace, "parent_trace", None)
        return False
    except Exception:  # pragma: no cover - private API changed
        return False


#: the active recorder, or None (timing off)
_ACTIVE: StepTimings | None = None


@contextlib.contextmanager
def record_timings(timings: StepTimings | None = None) -> Iterator[StepTimings]:
    """Turns step timing on inside the block.

    Args:
        timings: recorder to add to (default: a new one).

    Yields:
        The recorder; its :attr:`StepTimings.steps` fill as the steps run.
    """
    global _ACTIVE
    previous = _ACTIVE
    timings = timings if timings is not None else StepTimings()
    _ACTIVE = timings
    listener = timings._on_event
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield timings
    finally:
        _ACTIVE = previous
        with contextlib.suppress(Exception):
            from jax._src import monitoring

            monitoring.unregister_event_duration_listener(listener)


def active() -> StepTimings | None:
    """The active recorder, or None when timing is off."""
    return _ACTIVE


class _Step:
    """Context manager of :func:`timed_step`; ``done(value)`` names the outputs to wait for."""

    __slots__ = ("_frame", "_timings", "_value")

    def __init__(self, timings: StepTimings, name: str) -> None:
        self._timings = timings
        self._frame = timings.open(name)
        self._value: Any = None

    def done(self, value: Any) -> Any:
        """Marks ``value`` as the step's output (waited for when eager) and returns it."""
        self._value = value
        return value

    def __enter__(self) -> _Step:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._timings.close(self._frame, self._value if exc[0] is None else None)


class _Off:
    """The no-op step used while timing is off."""

    __slots__ = ()

    def done(self, value: Any) -> Any:
        return value

    def __enter__(self) -> _Off:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


_OFF = _Off()


def timed_step(name: str) -> _Step | _Off:
    """A context manager timing the step ``name`` (a no-op when timing is off)."""
    timings = _ACTIVE
    return _OFF if timings is None else _Step(timings, name)


def lap(name: str, value: Any = None) -> Any:
    """Records the time since the enclosing step's start or last lap as sub-step ``name``.

    Args:
        name: sub-step name.
        value: the sub-step's outputs, waited for when eager.

    Returns:
        ``value``.
    """
    timings = _ACTIVE
    if timings is not None:
        timings.lap(name, value)
    return value


def timed(name: str) -> Callable[[_F], _F]:
    """Decorator timing every call of a function as the step ``name`` (its output is waited for when eager)."""

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timings = _ACTIVE
            if timings is None:
                return fn(*args, **kwargs)
            frame = timings.open(name)
            out = None
            try:
                out = fn(*args, **kwargs)
            finally:
                timings.close(frame, out)
            return out

        return wrapper  # type: ignore[return-value]

    return decorate
