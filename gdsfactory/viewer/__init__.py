"""The gdsfactory viewer (spec ``design/viewer-spec.md``).

The v1 deliverable is an ``anywidget`` (spec D2), returned by ``view(stack)``
(spec D7). The analytic geometry builder exists so far: :mod:`gdsfactory.viewer.geometry`
builds the 2D level regions (milestone M0, spec D8) and
:mod:`gdsfactory.viewer.solids` builds the 3D level solids, their ``mesh_order``
painting and the port glyphs (milestone M1, spec D3). The widget itself is M2
onwards. Nothing here is imported by ``import gdsfactory`` (spec C3); the extra
dependencies (``manifold3d``, ``anywidget``) are the ``[gui]`` extra (spec C6).
"""

from __future__ import annotations

__all__: list[str] = []
