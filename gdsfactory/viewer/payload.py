"""The wire payload the browser receives (spec D5, C5).

The geometry is built once, by :class:`~gdsfactory.viewer.state.ViewerState`,
and split into two payloads so a style or selection edit never resends meshes:

- **geometry** carries the meshes as contiguous ``memoryview`` buffers.
  ipywidgets extracts those from nested trait values and ships them as binary
  comm buffers, so the float32 positions (measured from the box center, spec C5)
  keep their nm precision with no base64 or JSON cost.
- **meta** carries the level and port metadata as plain JSON.

The ``count`` / ``tris`` integers travel beside each buffer so the browser can
rebuild the typed arrays without a length probe.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gdsfactory.viewer.solids import SolidMesh
from gdsfactory.viewer.state import ViewerState

__all__ = ["build_payload"]


def _mesh_payload(mesh: SolidMesh) -> dict[str, Any]:
    """One mesh as a JSON header plus two contiguous binary buffers."""
    positions = np.ascontiguousarray(mesh.positions, dtype=np.float32)
    indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
    return {
        "positions": positions.data,
        "count": int(positions.shape[0]),
        "indices": indices.data,
        "tris": int(indices.shape[0]),
    }


def _level_meta(level: Any) -> dict[str, Any]:
    """A level's non-geometry metadata for the explorer (spec R1)."""
    return {
        "name": level.name,
        "material": level.material,
        "mesh_order": int(level.mesh_order),
        "fill": bool(level.fill),
        "zmin": float(level.zmin),
        "thickness": float(level.thickness),
        "sidewall_angle": float(level.sidewall_angle),
        "bias": float(level.bias),
        "width_to_z": float(level.width_to_z),
    }


def _port_meta(port: Any) -> dict[str, Any]:
    """A port's metadata for the explorer (spec R1)."""
    return {
        "name": port.name,
        "x": float(port.x),
        "y": float(port.y),
        "z": float(port.z),
        "orientation": None if port.orientation is None else float(port.orientation),
        "width": float(port.width),
        "layer": int(port.layer),
    }


def build_payload(viewer_state: ViewerState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Builds the geometry and metadata payloads from an already-built state.

    The geometry is not rebuilt here: :class:`ViewerState` owns the single build,
    and this reads it, so the meshes the browser renders are exactly the ones the
    kernel's state holds (spec D5, D10).

    Args:
        viewer_state: the state built once from a device.

    Returns:
        ``(geometry, meta)``: the meshes as binary buffers plus the box, and the
        level/port metadata as JSON.
    """
    geometry = {
        "origin": [float(v) for v in viewer_state.origin],
        "size": [float(v) for v in viewer_state.size],
        "levels": [
            {
                "name": level.name,
                "painted": _mesh_payload(level.painted),
                "full": _mesh_payload(level.full),
            }
            for level in viewer_state.levels
        ],
        "background": _mesh_payload(viewer_state.background),
    }
    meta = {
        "levels": [_level_meta(level) for level in viewer_state.levels],
        "ports": [_port_meta(port) for port in viewer_state.ports],
        "background": viewer_state.stack.background,
    }
    return geometry, meta
