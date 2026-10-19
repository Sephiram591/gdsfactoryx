"""Pytest plugin timing the StackDevice geometry steps and the FDTD solve per test.

Off unless ``FDTDX_TIMINGS=<name>`` is set in the environment or
``--fdtdx-timings=<name>`` is passed (``tests/conftest.py`` registers it).
Then every test runs inside :func:`gdsfactory.fdtdx_timing.record_timings`;
``fdtdx.place_objects`` and ``fdtdx.run_fdtd`` are timed as the steps
``place_objects`` and ``fdtd_solve``. At the end of the session it prints the
per-step table (wall, compile, run; see :mod:`gdsfactory.fdtdx_timing`) and the
slowest tests with their steps, and writes everything as JSON to
``.scratch/fdtdx_work/timings/<name>.json`` (or to ``<name>`` itself if it ends
in ``.json``).

Time outside the steps is reported too: per test (``outside``: the test's
duration less its top-level steps, and ``outside_compile``: the JAX compile
events that fired while no step was open, i.e. lowering and compilation of
jitted callers) and for the session. The steps alone are lower bounds of what a
feature costs.

Under pytest-xdist only the workers record (the controller runs no tests); each
writes ``<name>.<worker id>.json`` (e.g. ``<name>.gw0.json``), to be merged by
the reader.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

import pytest

ENV = "FDTDX_TIMINGS"
TOP_TESTS = 15  # slowest tests listed with their steps
TOP_STEPS = 6  # steps listed per test


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds ``--fdtdx-timings=<name>``."""
    parser.addoption(
        "--fdtdx-timings",
        default=None,
        help="time the StackDevice geometry steps and the FDTD solve; write "
        ".scratch/fdtdx_work/timings/<name>.json (or env FDTDX_TIMINGS)",
    )


class FdtdxTimingPlugin:
    """Records per-test step timings and reports them at the end of the session."""

    def __init__(self, name: str, root: pathlib.Path, worker: str | None = None) -> None:
        """Creates the plugin writing ``<root>/.scratch/fdtdx_work/timings/<name>.json`` (``<name>.<worker>.json``)."""
        from gdsfactory.fdtdx_timing import StepTimings

        path = pathlib.Path(name)
        tag = f".{worker}" if worker else ""
        self.path = (
            path.with_suffix(f"{tag}.json")
            if path.suffix == ".json"
            else root / ".scratch/fdtdx_work/timings" / f"{name}{tag}.json"
        )
        self.name = name
        self.session = StepTimings()
        self.tests: dict[str, dict[str, Any]] = {}
        #: over all tests: their duration, the part inside top-level steps, the compile outside every step
        self.total = {"tests": 0.0, "in_steps": 0.0, "outside_compile": 0.0}
        self.start = time.perf_counter()
        self._patched: list[tuple[Any, str, Any]] = []

    # ------------------------------------------------------------ hooks

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        """Wraps ``fdtdx.place_objects`` and ``fdtdx.run_fdtd`` in timed steps."""
        import fdtdx

        from gdsfactory.fdtdx_timing import timed

        for attr, step in (("place_objects", "place_objects"), ("run_fdtd", "fdtd_solve")):
            fn = getattr(fdtdx, attr, None)
            if fn is not None:
                self._patched.append((fdtdx, attr, fn))
                setattr(fdtdx, attr, timed(step)(fn))
        self.start = time.perf_counter()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> Any:
        """Runs one test (setup, call, teardown) with its own recorder."""
        from gdsfactory.fdtdx_timing import StepTimings, record_timings

        timings = StepTimings()
        t0 = time.perf_counter()
        with record_timings(timings):
            yield
        duration = time.perf_counter() - t0
        for key, rec in timings.steps.items():
            self.session.steps.setdefault(key, type(rec)()).add(rec)
        in_steps = sum(rec.wall for key, rec in timings.steps.items() if "/" not in key)
        self.total["tests"] += duration
        self.total["in_steps"] += in_steps
        self.total["outside_compile"] += timings.outside_compile
        if timings.steps or timings.outside:
            self.tests[item.nodeid] = {
                "duration": duration,
                "outside": max(duration - in_steps, 0.0),
                "outside_compile": timings.outside_compile,
                "steps": {k: v.as_dict() for k, v in timings.by_leaf().items()},
                "keys": timings.as_dict(),
            }

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Restores fdtdx and writes the JSON file."""
        for module, attr, fn in self._patched:
            setattr(module, attr, fn)
        wall = time.perf_counter() - self.start
        data = {
            "name": self.name,
            "suite_wall": wall,
            "total": {**self.total, "outside": max(self.total["tests"] - self.total["in_steps"], 0.0)},
            "by_step": {k: v.as_dict() for k, v in sorted(self.session.by_leaf().items())},
            "by_key": self.session.as_dict(),
            "tests": self.tests,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1))
        self.suite_wall = wall

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Prints the per-step table and the slowest tests' steps."""
        from gdsfactory.fdtdx_timing import StepRecord, format_table

        tr = terminalreporter
        tr.section(f"fdtdx step timings ({self.name})")
        tr.write_line(f"suite wall {getattr(self, 'suite_wall', 0.0):.1f} s; JSON {self.path}")
        tr.write_line("by step (inclusive; nested steps also count in their callers):")
        tr.write_line(format_table(self.session.by_leaf()))
        outside = max(self.total["tests"] - self.total["in_steps"], 0.0)
        tr.write_line(
            f"tests {self.total['tests']:.1f} s: {self.total['in_steps']:.1f} s in top-level steps, {outside:.1f} s "
            f"outside them (of which compile outside steps: {self.total['outside_compile']:.1f} s)"
        )
        tr.write_line("")
        tr.write_line(f"slowest {TOP_TESTS} tests with their slowest {TOP_STEPS} steps:")
        slow = sorted(self.tests.items(), key=lambda kv: -kv[1]["duration"])[:TOP_TESTS]
        for nodeid, t in slow:
            tr.write_line(
                f"{t['duration']:8.2f} s  {nodeid}  (outside steps {t['outside']:.2f} s, "
                f"compile there {t['outside_compile']:.2f} s)"
            )
            steps = sorted(t["steps"].items(), key=lambda kv: -kv[1]["wall"])[:TOP_STEPS]
            for key, r in steps:
                rec = StepRecord(**{k: r[k] for k in ("calls", "traced_calls", "wall", "traced", "compile")})
                tr.write_line(
                    f"           {key:<32} wall {rec.wall:7.2f}  trace {rec.traced:7.2f}  "
                    f"compile {rec.compile:7.2f}  run {rec.run:7.2f}  ({rec.calls} calls)"
                )


def configure(config: pytest.Config) -> None:
    """Registers the plugin if ``--fdtdx-timings`` or ``FDTDX_TIMINGS`` names a run.

    Under pytest-xdist the controller runs no tests and does not register; each
    worker registers and writes its own file.
    """
    name = config.getoption("--fdtdx-timings", default=None) or os.environ.get(ENV)
    if not name:
        return
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None and getattr(config.option, "numprocesses", None):
        return  # xdist controller
    worker = None if workerinput is None else workerinput["workerid"]
    config.pluginmanager.register(FdtdxTimingPlugin(name, pathlib.Path(config.rootpath), worker), "fdtdx-timings")
