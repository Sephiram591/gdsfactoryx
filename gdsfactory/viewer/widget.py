"""The viewer's ``anywidget`` app shell (spec D2, D7, D9, D10).

One widget owns the menubar, the explorer and the 3D viewport (spec D9). It
holds the shared :class:`~gdsfactory.viewer.state.ViewerState` in the kernel and
syncs three traits to the browser:

- ``geometry`` and ``meta`` — set once, from
  :func:`gdsfactory.viewer.payload.build_payload`; they carry the mesh buffers.
- ``state`` — the small JSON state, re-sent on every edit.

The browser edits the state by sending ``{"event": "set_state", "patch": ...}``;
:meth:`Viewer._on_msg` applies it to the kernel's state and syncs the canonical
result back (spec D10). A rejected patch re-syncs the state unchanged, so the
browser cannot diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anywidget
import traitlets

from gdsfactory.viewer.payload import build_payload
from gdsfactory.viewer.state import ViewerState

__all__ = ["Viewer", "view"]

_STATIC = Path(__file__).parent / "static"


class Viewer(anywidget.AnyWidget):
    """The app-shell widget returned by :func:`view` (spec D7, D9)."""

    _esm = _STATIC / "viewer.js"
    _css = _STATIC / "viewer.css"

    #: set once: the mesh buffers and the box (spec D5, C5)
    geometry = traitlets.Dict(default_value={}).tag(sync=True)
    #: set once: the level and port metadata (spec D5)
    meta = traitlets.Dict(default_value={}).tag(sync=True)
    #: the shared state, re-sent on every edit (spec D10)
    state = traitlets.Dict(default_value={}).tag(sync=True)

    def __init__(self, stack: Any, **kwargs: Any) -> None:
        """Builds the device's geometry and prepares the widget.

        Args:
            stack: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.
            **kwargs: passed to :class:`anywidget.AnyWidget`.
        """
        self._viewer_state = ViewerState(stack)
        super().__init__(**kwargs)
        geometry, meta = build_payload(self._viewer_state)
        self.geometry = geometry
        self.meta = meta
        self.state = self._viewer_state.to_state()
        self.on_msg(self._on_msg)

    @property
    def stack(self) -> Any:
        """The :class:`~gdsfactory.fdtdx_stack.StackDevice` being viewed (spec D7)."""
        return self._viewer_state.stack

    def _on_msg(self, _widget: Any, content: dict[str, Any], _buffers: list) -> None:
        """Applies a browser patch to the kernel state and re-syncs it (spec D10)."""
        if content.get("event") != "set_state":
            return
        try:
            self._viewer_state.apply_patch(content.get("patch") or {})
        except (ValueError, TypeError):
            pass  # a bad patch leaves the state unchanged
        self.state = self._viewer_state.to_state()


def view(stack: Any) -> Viewer:
    """Returns the viewer for a device (spec D7).

    Args:
        stack: a :class:`~gdsfactory.fdtdx_stack.StackDevice`.

    Returns:
        The :class:`Viewer` app-shell widget.
    """
    return Viewer(stack)
