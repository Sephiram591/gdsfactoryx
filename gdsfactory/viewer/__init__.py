"""The gdsfactory viewer (spec ``design/viewer-spec.md``).

The v1 deliverable is an ``anywidget`` (spec D2), returned by ``view(stack)``
(spec D7). The analytic geometry builder builds the 2D level regions
(:mod:`gdsfactory.viewer.geometry`, milestone M0, spec D8) and the 3D level
solids, their ``mesh_order`` painting and the port glyphs
(:mod:`gdsfactory.viewer.solids`, milestone M1, spec D3). The explorer and the
3D viewport are M2 (spec R1-R3): :mod:`gdsfactory.viewer.state` holds the shared
state, :mod:`gdsfactory.viewer.payload` the wire format and
:mod:`gdsfactory.viewer.widget` the app shell. The sources, detectors and the
run are M4 (spec R12-R14, D11, D15): :mod:`gdsfactory.viewer.sim` holds the
declarative specs and the preparation path, and ``widget`` exposes
``.objects``/``.config``/``.constraints`` (mode 2) and ``.run()`` (mode 1, on a
background thread with the setup locked while it runs).

Nothing here is imported by ``import gdsfactory`` (spec C3); the extra
dependencies (``manifold3d``, ``anywidget``) are the ``[gui]`` extra (spec C6).
``view`` is reached lazily, so importing this package does not import them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["view"]


def __getattr__(name: str) -> Any:
    """Lazily exposes :func:`gdsfactory.viewer.widget.view` (spec C3)."""
    if name == "view":
        from gdsfactory.viewer.widget import view

        return view
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
