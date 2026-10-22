"""Adopting a foreign completed simulation into the viewer (spec R16, D18).

A result from arbitrary user code — the user's own ``fdtdx`` volume, detectors
and source, placed by :func:`fdtdx.place_objects` and run by
:func:`fdtdx.run_fdtd` — reaches the viewer as the three pieces fdtdx hands
out: ``(objects, arrays, config)``. Nothing about it is keyed to the viewer's
specs, so adoption rebuilds the display from the placed objects themselves:

- every detector in ``objects.detectors`` becomes a read-only "None"-port spec
  (absolute placement, spec R18) whose glyph is derived from the detector's
  ``grid_slice`` and the result's resolution;
- every source in ``objects.sources`` becomes the same kind of read-only spec,
  so the Result tab can list it with a visibility toggle (spec R17);
- only phasor-family detectors (a ``"phasor"`` key in
  ``arrays.detector_states[name]``) shade; the rest are listed with plain
  glyphs.

The grid→real conversion relies on the fdtdx invariant that the volume's grid
slices start at 0 (``fdtdx.fdtd.initialization`` anchors them), so every
detector's ``grid_slice`` indices are offsets from the volume corner in grid
points.
"""

from __future__ import annotations

from typing import Any

from gdsfactory.viewer.sim import TYPES, ObjectSpec, _IN_PLANE

__all__ = ["adopt_result", "glyph_from_grid"]


def grid_bounds_um(
    grid_slice: tuple[slice, slice, slice],
    resolution_m: float,
    volume_grid_shape: tuple[int, int, int],
    origin_um: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    """A grid slice's low/high corners in µm, in the meshes' local frame.

    Grid axis ``i`` spans ``grid_slice[i].start * res`` to
    ``grid_slice[i].stop * res`` from the volume corner; ``origin_um`` is the
    box center the meshes are shifted by (spec C5), so the volume corner sits
    at ``origin - 0.5 * size`` in that frame.
    """
    res_um = float(resolution_m) * 1e6
    corner = [origin_um[i] - 0.5 * volume_grid_shape[i] * res_um for i in range(3)]
    lo = [corner[i] + grid_slice[i].start * res_um for i in range(3)]
    hi = [corner[i] + grid_slice[i].stop * res_um for i in range(3)]
    return lo, hi


def _sampled_center(
    lo: list[float], hi: list[float], axis: int, category: str, plane: bool
) -> list[float]:
    """A placed object's center, with a detector plane moved onto its samples.

    fdtdx records a detector's x and y samples on a cell's lower edge and z
    samples at its centre, so along a plane's normal the cell's lower edge (x, y)
    or middle (z) is where the field is, not the middle of the cell.
    """
    center = [0.5 * (l + h) for l, h in zip(lo, hi, strict=True)]
    if plane and category == "detector" and axis in (0, 1):
        center[axis] = lo[axis]
    return center


def glyph_from_grid(
    grid_slice: tuple[slice, slice, slice],
    resolution_m: float,
    volume_grid_shape: tuple[int, int, int],
    origin_um: tuple[float, float, float],
    spec_id: str,
    name: str,
    type_name: str,
    category: str,
) -> dict[str, Any]:
    """A detector's wire glyph from its placed ``grid_slice`` (spec D18).

    The same wire shape :func:`~gdsfactory.viewer.sim.spec_glyph` produces, so
    the browser cannot tell an adopted glyph from a computed one. A plane
    detector (exactly one 1-wide grid axis) becomes a plane glyph with its
    normal axis and in-plane half-extents; a volume/line/point detector
    becomes a point glyph at its centroid.
    """
    lo, hi = grid_bounds_um(grid_slice, resolution_m, volume_grid_shape, origin_um)
    half = [0.5 * (h - l) for l, h in zip(lo, hi, strict=True)]
    widths = [int(grid_slice[i].stop - grid_slice[i].start) for i in range(3)]
    singletons = [i for i in range(3) if widths[i] == 1]

    u_half = v_half = 0.0
    axis = singletons[0] if len(singletons) == 1 else 2
    center = _sampled_center(lo, hi, axis, category, len(singletons) == 1)
    if len(singletons) == 1:
        u_axis, v_axis = _IN_PLANE[axis]
        u_half, v_half = half[u_axis], half[v_axis]
    return {
        "id": spec_id,
        "name": name,
        "kind": category,
        "type": type_name,
        "port": "None",
        "center": center,
        "axis": axis,
        "u_half": u_half,
        "v_half": v_half,
        "direction": None,
        "visible": True,
    }


def _spec_type_for(obj: Any, category: str) -> str:
    """The viewer type name for a placed fdtdx source or detector (spec D18)."""
    type_name = type(obj).__name__
    if type_name in TYPES and TYPES[type_name].category == category:
        return type_name
    # a subclass the viewer does not name: fall back to the category's first
    # type, so the spec stays listable (a phasor-family detector still shades
    # through its "phasor" detector state, not its type name)
    for kind in TYPES.values():
        if kind.category == category:
            return kind.name
    return type_name


def _absolute_params(
    grid_slice: tuple[slice, slice, slice],
    resolution_m: float,
    volume_grid_shape: tuple[int, int, int],
    origin_um: tuple[float, float, float],
    category: str = "source",
) -> dict[str, Any]:
    """The ``axis``/``center``/``size`` params describing a placement (R18)."""
    lo, hi = grid_bounds_um(grid_slice, resolution_m, volume_grid_shape, origin_um)
    widths = [int(grid_slice[i].stop - grid_slice[i].start) for i in range(3)]
    singletons = [i for i in range(3) if widths[i] == 1]
    axis = singletons[0] if len(singletons) == 1 else 2
    center = _sampled_center(lo, hi, axis, category, len(singletons) == 1)
    u_axis, v_axis = _IN_PLANE[axis]
    return {
        "axis": axis,
        "center": center,
        "size": [hi[u_axis] - lo[u_axis], hi[v_axis] - lo[v_axis]],
    }


def _config_spacing(config: Any) -> float | None:
    """A config's grid spacing in meters, or None (fdtdx ≥ 0.6 or older)."""
    if hasattr(config, "uniform_spacing"):
        try:
            return float(config.uniform_spacing())
        except ValueError:
            return None
    spacing = getattr(getattr(config, "grid", None), "spacing", None)
    if spacing is None:
        spacing = getattr(config, "resolution", None)
    return None if spacing is None else float(spacing)


def adopt_result(state: Any, objects: Any, config: Any) -> list[ObjectSpec]:
    """Creates one read-only "None"-port spec per source and detector (R16).

    Names are read from the placed objects — fdtdx auto-generates
    ``Object_{n}`` names, never assume. The specs describe the placement; they
    are never re-created by ``build_objects`` (they live in ``state.adopted``).

    Args:
        state: the fresh :class:`~gdsfactory.viewer.state.ViewerState` to add
            the specs to.
        objects: the result's placed ``ObjectContainer``.
        config: the result's ``SimulationConfig`` (for the resolution).

    Returns:
        The adopted specs, sources first, in container order within each kind.
    """
    resolution_m = _config_spacing(config)
    if resolution_m is None:
        raise ValueError("the result needs a positive grid spacing (.config.grid.spacing)")
    volume = objects.volume
    volume_grid_shape = tuple(int(v) for v in volume.grid_shape)
    origin_um = tuple(float(v) for v in state.origin)
    volume_slice = volume.grid_slice
    anchored = all(getattr(volume_slice[i], "start", 0) == 0 for i in range(3))

    specs: list[ObjectSpec] = []
    for category, placed in (
        ("source", getattr(objects, "sources", ())),
        ("detector", objects.detectors),
    ):
        for obj in placed:
            specs.append(_adopt_object(
                state, obj, category, anchored, resolution_m, volume_grid_shape, origin_um,
            ))
    return specs


def _adopt_object(
    state: Any,
    obj: Any,
    category: str,
    anchored: bool,
    resolution_m: float,
    volume_grid_shape: tuple[int, int, int],
    origin_um: tuple[float, float, float],
) -> ObjectSpec:
    """One placed object's read-only "None"-port spec (spec R16, D18)."""
    name = str(obj.name)
    type_name = _spec_type_for(obj, category)
    params: dict[str, Any] = {"name": name, "visible": True}
    # the recorded components, so the default shading component (|E|, or the
    # first one when no E was recorded) matches what the result holds
    components = getattr(obj, "components", None)
    if components and "components" in TYPES[type_name].extra:
        params["components"] = [str(c) for c in components]
    spec = ObjectSpec(type=type_name, port="None", params=params)
    if anchored:
        try:
            params.update(_absolute_params(
                tuple(obj.grid_slice), resolution_m, volume_grid_shape, origin_um, category,
            ))
            spec.glyph = glyph_from_grid(
                tuple(obj.grid_slice), resolution_m, volume_grid_shape, origin_um,
                spec_id="", name=name, type_name=type_name, category=category,
            )
        except (AttributeError, TypeError, ValueError):
            # no usable grid_slice (or not yet placed): list it without a glyph
            # rather than refusing the whole result
            spec.glyph = None
    state.adopt_spec(spec)
    return spec
