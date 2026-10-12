"""Fallback layer colors for the viewer (spec R2).

A layer's color defaults to the active PDK's ``layer_views`` fill color (spec
R2). When the PDK has no view for the layer, or no PDK is active, a color is
taken from :data:`FALLBACK`, cycled by the layer's position in the stack so a
device's layers stay distinguishable.
"""

from __future__ import annotations

__all__ = ["FALLBACK", "fallback_color"]

#: Distinct, medium-saturation hex colors for layers the PDK does not color.
FALLBACK: tuple[str, ...] = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
)


def fallback_color(index: int) -> str:
    """A stable fallback hex color for the ``index``-th layer in the stack."""
    return FALLBACK[index % len(FALLBACK)]
