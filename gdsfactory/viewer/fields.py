"""Detector field extraction and result validation (spec R15, D16, Q20-Q23).

A detector that recorded a field carries it on the last run's
``result.arrays.detector_states[name]`` in one of two layouts, both a stack of
**frames** shaped ``(num_frames, num_components, *grid_shape)`` (spec R15):

- a phasor-family detector's complex ``"phasor"``, one frame per recorded
  frequency (``frame_kind == "freq"``);
- a ``FieldDetector``'s real ``"fields"``, one frame per recorded time step
  (``frame_kind == "time"``), which the browser can play back and export as a
  video.

For a plane detector the grid shape has the detector's normal axis as a
singleton and the two in-plane axes last, already in the order the glyph's
``(u, v)`` basis uses (spec D11), so the trailing two dims can be handed to a
texture with no transpose.

The browser applies the display transform (``real``, ``imag``, ``abs``,
``phase``, ``logabs``), the colormap, the transparency and the opacity, so the
kernel sends the **raw** field — the real and imaginary parts of a phasor, the
real samples of a time-domain recording, as float32 buffers — and a transform
or opacity edit re-shades with no kernel work beyond the state patch (spec
R15's AC, D16).

:func:`validate_result` is the strict check ``viewer.show_result`` makes before
adopting a user-supplied result (spec Q23): the result's detectors must match the
widget's current setup by name, grid shape, grid spacing and plane position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["TRANSFORMS", "field_payload", "validate_foreign_result", "validate_result"]

#: The display transforms the browser applies to a component (spec R15).
TRANSFORMS = ("real", "imag", "abs", "phase", "logabs")

#: Where a recorded sample sits along each stack axis, in grid cells from the
#: cell's centre. fdtdx co-locates all six components at the E_z Yee point
#: ``(i·Δx, j·Δy, (k+½)·Δz)`` (``fdtdx.core.physics.curl.interpolate_fields``):
#: x and y samples lie on the cell's lower edge, z samples at its centre. The
#: browser draws every sample as a cell, so it shifts the raster by this much.
_SAMPLE_SHIFT = (-0.5, -0.5, 0.0)

#: The number of grid points a grid extent may differ by before a result is
#: refused as "recorded on a different device" (spec Q23).
_GRID_TOL = 1

#: The relative tolerance on a grid's physical spacing, in µm (spec Q23).
_SPACING_RTOL = 1e-6

#: The most float32 bytes one detector's frames may take on the wire. A
#: ``FieldDetector`` stores one plane per recorded step, which can run to
#: gigabytes; above this the frames are thinned by a stride (every k-th one is
#: sent) and the payload says so (``frames_recorded``, ``frame_stride``).
_MAX_FIELD_BYTES = 256 * 1024 * 1024


def _grid_ndim(result: Any, name: str) -> int:
    """The number of grid axes a detector's field has (3 for a plane detector).

    Read from the placed object's ``grid_slice`` — one slice per grid axis — so
    the phasor's leading frame axes (frequencies and latent time steps) can be
    told apart from its trailing grid axes without guessing.
    """
    try:
        grid_slice = result.objects[name].grid_slice
    except (AttributeError, KeyError, TypeError):
        return 3
    try:
        return len(grid_slice)
    except TypeError:
        return 3


def _detector_frames(result: Any, name: str) -> tuple[Any, str] | None:
    """The frames a detector recorded, as ``((frames, components, *grid), kind)``.

    A phasor's raw array is ``(latent, num_freqs, num_components, *grid_shape)``
    — the leading axis is the latent time step, a singleton for the phasor
    family (``_num_latent_time_steps`` is 1). Only that one axis is peeled: the
    frequency axis is kept, so a multi-frequency detector's every frame reaches
    the browser (``kind == "freq"``). A ``FieldDetector``'s real ``"fields"``
    array is already ``(recorded_steps, num_components, *grid_shape)``
    (``kind == "time"``).

    The array is returned as the detector state holds it (a device array, not
    yet copied to the host), so a caller can thin the frames before the copy. A
    detector that recorded no field, or whose array does not fit those shapes
    (e.g. a volume-reduced one), yields None: it keeps its plain glyph.
    """
    try:
        state = result.arrays.detector_states[name]
    except (AttributeError, KeyError, TypeError):
        return None
    if not hasattr(state, "get"):
        return None
    grid_ndim = _grid_ndim(result, name)
    phasor = state.get("phasor")
    if phasor is not None:
        if not np.iscomplexobj(phasor):
            return None
        # peel the single latent time step axis, keeping (num_freqs, components, *grid)
        if phasor.ndim > grid_ndim + 2 and phasor.shape[0] == 1:
            phasor = phasor[0]
        return (phasor, "freq") if phasor.ndim == grid_ndim + 2 else None
    recorded = state.get("fields")
    if recorded is not None and not np.iscomplexobj(recorded) and recorded.ndim == grid_ndim + 2:
        return recorded, "time"
    return None


def _detector_steps(result: Any, name: str, num_frames: int) -> list[int] | None:
    """The solver steps a time-domain detector recorded at, one per frame.

    Read from the placed detector's on-mask (``_is_on_at_time_step_arr``, one
    flag per simulation step, set by its ``switch``), so a ``record_interval``
    shows up as the spacing between frames. None when the mask is missing or
    its count disagrees with the frames.
    """
    try:
        on = np.asarray(result.objects[name]._is_on_at_time_step_arr)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    steps = np.flatnonzero(on)
    return [int(s) for s in steps] if steps.size == num_frames else None


def _solve_clock(result: Any) -> tuple[float, int] | None:
    """The run's ``(time_step_duration, time_steps_total)``, or None."""
    try:
        return float(result.config.time_step_duration), int(result.config.time_steps_total)
    except (AttributeError, TypeError, ValueError):
        return None


def _detector_components(result: Any, name: str) -> list[str]:
    """The component names a placed detector recorded, in order (spec R15)."""
    try:
        obj = result.objects[name]
    except (AttributeError, KeyError, TypeError):
        return []
    components = getattr(obj, "components", None)
    return [str(c) for c in components] if components else []


def _detector_freqs(result: Any, name: str, num_freqs: int) -> list[float]:
    """The frequencies a phasor detector recorded at, in Hz (spec R17).

    Read from the placed detector's ``wave_characters`` (each
    :class:`~fdtdx.core.wavelength.WaveCharacter` exposes ``get_frequency()``).
    Falls back to frame indices when the object exposes none, or to as many
    entries as frames when the counts disagree.
    """
    try:
        obj = result.objects[name]
        wcs = list(obj.wave_characters)
    except (AttributeError, KeyError, TypeError):
        return list(range(num_freqs))
    try:
        freqs = [float(wc.get_frequency()) for wc in wcs]
    except (AttributeError, TypeError, ValueError):
        return list(range(num_freqs))
    return freqs if len(freqs) == num_freqs else list(range(num_freqs))


def field_payload(result: Any, name: str, spec_id: str) -> dict[str, Any] | None:
    """One detector's raw field for the browser (spec R15, D16, R17).

    Sends every recorded component of every frame as contiguous float32
    buffers shaped ``(num_frames, components, v, u)``, so the browser can apply
    any transform and pick or play any frame without a further kernel
    round-trip. A phasor sends its real and imaginary parts, one frame per
    recorded frequency; a ``FieldDetector`` sends its real samples only (no
    ``imag``), one frame per recorded time step. A single-frequency run sends
    one frame, laid out exactly as before the frame axis existed. Frames over
    ``_MAX_FIELD_BYTES`` are thinned by a stride before they leave the device.
    Returns None when the detector recorded no field, so it keeps its plain
    glyph.

    Args:
        result: the :class:`~gdsfactory.viewer.sim.RunResult` to read.
        name: the detector's name.
        spec_id: the spec id, so the browser matches the field to its glyph.

    Returns:
        ``{"id", "name", "components", "grid", "shift", "frame_kind", "freq",
        "frames_recorded", "frame_stride", "real"}`` plus ``"freqs"`` (Hz) and
        ``"imag"`` for a phasor, or ``"times"`` (s) for a time-domain
        recording — with ``"steps"`` (the solver step of each frame),
        ``"time_step"`` (s) and ``"steps_total"`` when the result carries its
        clock — with the buffers as ``memoryview``s, or None. ``shift`` is the
        ``(u, v)`` offset of a sample from its cell's centre, in cells
        (``-0.5`` on x and y, ``0`` on z); ``freq`` is the initially selected
        frame.
    """
    found = _detector_frames(result, name)
    if found is None or found[0].ndim < 4:
        # no field, or a volume-reduced detector with no in-plane grid to draw
        return None
    frames, kind = found
    complex_valued = kind == "freq"
    recorded = int(frames.shape[0])
    # thin the frames on the device, before the host copy, when they would not
    # fit the wire budget (float32, twice over for a complex field)
    frame_bytes = int(np.prod(frames.shape[1:])) * 4 * (2 if complex_valued else 1)
    stride = max(1, -(-recorded * frame_bytes // _MAX_FIELD_BYTES))
    phasor = np.asarray(frames[::stride] if stride > 1 else frames)
    components = _detector_components(result, name)
    if not components or phasor.shape[1] != len(components):
        # fall back to generic names so the field is still plottable
        components = [f"c{i}" for i in range(phasor.shape[1])]
    # the two in-plane axes, with the detector's normal axis (a singleton)
    # dropped. The normal is not always last — an xz detector's is the middle
    # axis — so the singleton is found by value, not position. The axes run in
    # stack order (x, y, z), which is (u, v) for every plane: the glyph's u is
    # the lower of the two in-plane stack axes and v the higher (spec D11).
    in_plane = [i for i in range(2, phasor.ndim) if phasor.shape[i] > 1]
    if len(in_plane) != 2:
        return None
    # drop the normal axis, leaving (num_freqs, components, u, v) in stack order
    plane = np.squeeze(phasor, axis=tuple(i for i in range(2, phasor.ndim) if i not in in_plane))
    # The browser textures a plane row-major with the rows as v and the columns
    # as u (the same order the 2D blit and the 3D ``DataTexture`` use), so the
    # field is transposed to (num_freqs, components, v, u). Sending it in stack
    # order would swap the axes of any non-square plane — a 450x75 xz plane
    # would read as 75x450.
    ordered = np.ascontiguousarray(np.swapaxes(plane, 2, 3))
    real = np.ascontiguousarray(ordered.real, dtype=np.float32)
    entry: dict[str, Any] = {
        "id": spec_id,
        "name": name,
        "components": components,
        "grid": [int(ordered.shape[2]), int(ordered.shape[3])],
        # (u, v) sample offsets in cells, in the same stack-axis order as the grid
        "shift": [_SAMPLE_SHIFT[i - 2] for i in in_plane],
        "frame_kind": kind,
        "freq": 0,
        "frames_recorded": recorded,
        "frame_stride": stride,
        "real": real.data,
    }
    if complex_valued:
        entry["freqs"] = _detector_freqs(result, name, recorded)[::stride]
        entry["imag"] = np.ascontiguousarray(ordered.imag, dtype=np.float32).data
    else:
        steps = _detector_steps(result, name, recorded)
        clock = _solve_clock(result)
        if steps is not None and clock is not None:
            # the solver steps the frames were recorded at, and the solve's own
            # clock, so the browser can play several detectors on one timeline
            step, total = clock
            entry["steps"] = steps[::stride]
            entry["times"] = [s * step for s in entry["steps"]]
            entry["time_step"] = step
            entry["steps_total"] = total
        else:
            # no on-mask or clock: the frames are ordered but not timed
            entry["times"] = [float(i) for i in range(recorded)][::stride]
    return entry


def validate_result(state: Any, result: Any) -> None:
    """Refuses a result that does not match the widget's current setup (Q23).

    Strict: every detector the widget holds must appear in the result with the
    same grid shape, the same grid spacing and a plane at the same place, and
    the result must hold no detectors the widget does not. Raises ``ValueError``
    with a reason on the first mismatch, so the caller can leave the previous
    result and its overlays untouched.

    Args:
        state: the :class:`~gdsfactory.viewer.state.ViewerState`.
        result: the candidate result.

    Raises:
        ValueError: if the result does not correspond to the current setup.
    """
    try:
        detector_states = result.arrays.detector_states
    except AttributeError as exc:
        raise ValueError(
            "show_result needs an object with .arrays.detector_states, .objects and .config"
        ) from exc

    expected = {spec.name: spec for spec in state.specs if spec.category == "detector"}
    found = set(detector_states)
    missing = sorted(set(expected) - found)
    if missing:
        raise ValueError(
            f"the result has no field for detector(s) {missing}; it does not match "
            "the current setup"
        )
    extra = sorted(found - set(expected))
    if extra:
        raise ValueError(
            f"the result holds detector(s) {extra} that the setup does not; it was "
            "recorded on a different device"
        )

    resolution = _result_resolution(result)
    if resolution is None:
        raise ValueError("the result has no grid resolution (.config.grid.spacing)")

    for name, spec in expected.items():
        found = _detector_frames(result, name)
        if found is None:
            continue  # no field recorded: nothing to mis-shade
        frames = found[0]
        components = _detector_components(result, name)
        if components and frames.shape[1] != len(components):
            raise ValueError(
                f"detector {name!r}: the result has {frames.shape[1]} components but "
                f"the detector records {len(components)}"
            )
        grid = tuple(int(v) for v in frames.shape[2:])
        if not _grid_matches(state, spec, grid, resolution):
            raise ValueError(
                f"detector {name!r}: the result's {grid} grid does not match the "
                "current setup; it was recorded on a different device"
            )

def _result_resolution(result: Any) -> float | None:
    """The result's grid spacing in µm, or None (spec Q23).

    fdtdx ≥ 0.6 carries the spacing on the config (any grid type); older
    versions on ``config.resolution`` (in meters, like the spacing).
    """
    config = getattr(result, "config", None)
    spacing = _config_spacing(config)
    return None if spacing is None else spacing * 1e6  # meters -> µm


def _grid_matches(state: Any, spec: Any, grid: tuple[int, ...], resolution_um: float) -> bool:
    """Whether a recorded grid fits the spec's current plane (spec Q23).

    Checks the spacing against the settings and each in-plane extent against the
    detector's plane size from its port frame — the same frame the glyph uses
    (spec D11) — so a result recorded at another resolution, or on a detector
    whose plane has since been resized, is refused.
    """
    settings = state.settings
    if not np.isclose(resolution_um, settings.resolution_um, rtol=_SPACING_RTOL):
        return False
    from gdsfactory.viewer.sim import TYPES, spec_glyph

    kind = TYPES[spec.type]
    if not kind.plane:
        return True
    glyph = spec_glyph(state.stack, spec, state.origin, settings)
    if glyph.get("error"):
        return False
    # the two in-plane extents, in grid points at the settings resolution
    in_plane = [g for g in grid if g > 1]
    expected = sorted(
        int(round(2 * half / resolution_um)) for half in (glyph["u_half"], glyph["v_half"])
    )
    return len(in_plane) == 2 and all(
        abs(a - b) <= _GRID_TOL for a, b in zip(sorted(in_plane), expected)
    )


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


def validate_foreign_result(
    objects: Any, arrays: Any, config: Any, stack: Any = None
) -> None:
    """Accepts-or-refuses a result from arbitrary user code (spec R16, D18).

    Deliberately lenient — the goal is to display a foreign result, not to gate
    it — but a result that could never shade (or whose grid disagrees with the
    stack it is drawn on) is refused with a reason instead of silently
    rendering nothing. The strict name-based :func:`validate_result` is a
    different, tighter contract for :meth:`Viewer.show_result` and stays as is.

    Args:
        objects: the result's placed ``ObjectContainer`` (needs ``.detectors``
            and ``.volume``).
        arrays: the result's arrays (needs ``.detector_states``).
        config: the result's ``SimulationConfig`` (needs a positive grid
            spacing).
        stack: the :class:`~gdsfactory.fdtdx_stack.StackDevice` the viewer will
            rebuild its geometry from, if known.

    Raises:
        ValueError: if any required piece is missing, a phasor-family detector
            would never shade, or the stack's box disagrees with the result's
            grid.
    """
    detector_states = getattr(arrays, "detector_states", None)
    if not hasattr(detector_states, "get"):
        raise ValueError("the result needs .arrays.detector_states")
    detectors = getattr(objects, "detectors", None)
    volume = getattr(objects, "volume", None)
    if detectors is None or volume is None:
        raise ValueError("the result needs .objects.detectors and .objects.volume")
    resolution = _config_spacing(config)
    if resolution is None or not resolution > 0:
        raise ValueError("the result needs a positive grid resolution (.config.grid.spacing)")

    for obj in detectors:
        state = detector_states.get(obj.name) if hasattr(detector_states, "get") else None
        phasor = state.get("phasor") if hasattr(state, "get") else None
        recorded = state.get("fields") if hasattr(state, "get") else None
        widths = [int(obj.grid_slice[i].stop - obj.grid_slice[i].start) for i in range(3)]
        is_plane = sum(1 for w in widths if w > 1) == 2
        if phasor is not None and np.iscomplexobj(phasor):
            if not is_plane:
                raise ValueError(
                    f"detector {obj.name!r} records a phasor but is not a plane "
                    f"(grid widths {widths}); its field could never shade"
                )
            shape = tuple(phasor.shape)
        elif recorded is not None and is_plane and len(recorded.shape) == 5:
            # a time-domain plane recording shades too; a volume or volume-reduced
            # one is listed but never shaded, so it has nothing to check
            shape = tuple(recorded.shape)
        else:
            continue  # a plain detector: listed, never shaded — nothing to check
        if shape[-3:] != tuple(widths):
            raise ValueError(
                f"detector {obj.name!r}: field grid shape {shape[-3:]} does not "
                f"match the placed detector's {tuple(widths)}"
            )

    if stack is not None:
        real_shape_m = tuple(float(v) for v in stack.partial_real_shape)
        grid_shape = tuple(int(v) for v in volume.grid_shape)
        for i in range(3):
            grid_m = grid_shape[i] * float(resolution)
            if abs(grid_m - real_shape_m[i]) > float(resolution):
                raise ValueError(
                    f"the result's grid extent along axis {i} ({grid_shape[i]} cells "
                    f"at {float(resolution)} m) does not fit the stack's box "
                    f"({real_shape_m[i]:g} m, tolerance one cell); pass the stack "
                    "the result was recorded on"
                )
