"""Detector field extraction and result validation (spec R15, D16, Q20-Q23).

A detector that recorded a field carries it on the last run's
``result.arrays.detector_states[name]["phasor"]`` as a complex64 array shaped
``(num_frequencies, num_components, *grid_shape)`` (spec R15). For a plane
detector the grid shape has the detector's normal axis as a singleton and the
two in-plane axes last, already in the order the glyph's ``(u, v)`` basis uses
(spec D11), so the trailing two dims can be handed to a texture with no
transpose.

The browser applies the display transform (``real``, ``imag``, ``abs``,
``phase``, ``logabs``), the colormap, the transparency and the opacity, so the
kernel sends the **raw** complex field — its real and imaginary parts as float32
buffers — and a transform or opacity edit re-shades with no kernel work beyond
the state patch (spec R15's AC, D16).

:func:`validate_result` is the strict check ``viewer.show_result`` makes before
adopting a user-supplied result (spec Q23): the result's detectors must match the
widget's current setup by name, grid shape, grid spacing and plane position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["TRANSFORMS", "field_payload", "validate_result"]

#: The display transforms the browser applies to a component (spec R15).
TRANSFORMS = ("real", "imag", "abs", "phase", "logabs")

#: The number of grid points a grid extent may differ by before a result is
#: refused as "recorded on a different device" (spec Q23).
_GRID_TOL = 1

#: The relative tolerance on a grid's physical spacing, in µm (spec Q23).
_SPACING_RTOL = 1e-6


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


def _detector_phasor(result: Any, name: str) -> np.ndarray | None:
    """The complex field a detector recorded, as ``(num_components, *grid)``.

    The raw array is ``(..., num_components, *grid_shape)`` — the leading axes
    are the run's frequencies and latent time steps, both singletons for the
    widget's single-frequency CW runs, so they are peeled off and the first
    frequency/latent frame is used. A detector that recorded no field, or whose
    array does not fit that shape, yields None: it keeps its plain glyph.
    """
    try:
        state = result.arrays.detector_states[name]
    except (AttributeError, KeyError, TypeError):
        return None
    phasor = state.get("phasor") if hasattr(state, "get") else None
    if phasor is None:
        return None
    array = np.asarray(phasor)
    if not np.iscomplexobj(array):
        return None
    grid_ndim = _grid_ndim(result, name)
    # peel the leading frame axes down to (num_components, *grid_shape)
    while array.ndim > grid_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != grid_ndim + 1:
        return None
    return array


def _detector_components(result: Any, name: str) -> list[str]:
    """The component names a placed detector recorded, in order (spec R15)."""
    try:
        obj = result.objects[name]
    except (AttributeError, KeyError, TypeError):
        return []
    components = getattr(obj, "components", None)
    return [str(c) for c in components] if components else []


def field_payload(result: Any, name: str, spec_id: str) -> dict[str, Any] | None:
    """One detector's raw field for the browser (spec R15, D16).

    Sends the real and imaginary parts of every recorded component as
    contiguous float32 buffers, so the browser can apply any transform without a
    further kernel round-trip. Returns None when the detector recorded no field,
    so it keeps its plain glyph.

    Args:
        result: the :class:`~gdsfactory.viewer.sim.RunResult` to read.
        name: the detector's name.
        spec_id: the spec id, so the browser matches the field to its glyph.

    Returns:
        ``{"id", "name", "components", "grid", "real", "imag"}`` with the two
        buffers as ``memoryview``s, or None.
    """
    phasor = _detector_phasor(result, name)
    if phasor is None or phasor.ndim < 3:
        # no field, or a volume-reduced detector with no in-plane grid to draw
        return None
    components = _detector_components(result, name)
    if not components or phasor.shape[0] != len(components):
        # fall back to generic names so the field is still plottable
        components = [f"c{i}" for i in range(phasor.shape[0])]
    # the two in-plane axes, with the detector's normal axis (a singleton)
    # dropped. The normal is not always last — an xz detector's is the middle
    # axis — so the singleton is found by value, not position. The axes run in
    # stack order (x, y, z), which is (u, v) for every plane: the glyph's u is
    # the lower of the two in-plane stack axes and v the higher (spec D11).
    in_plane = [i for i in range(1, phasor.ndim) if phasor.shape[i] > 1]
    if len(in_plane) != 2:
        return None
    # drop the normal axis, leaving (components, u, v) in stack order
    plane = np.squeeze(phasor, axis=tuple(i for i in range(1, phasor.ndim) if i not in in_plane))
    # The browser textures a plane row-major with the rows as v and the columns
    # as u (the same order the 2D blit and the 3D ``DataTexture`` use), so the
    # field is transposed to (components, v, u). Sending it in stack order would
    # swap the axes of any non-square plane — a 450x75 xz plane would read as
    # 75x450.
    ordered = np.ascontiguousarray(np.swapaxes(plane, 1, 2))
    real = np.ascontiguousarray(ordered.real, dtype=np.float32)
    imag = np.ascontiguousarray(ordered.imag, dtype=np.float32)
    return {
        "id": spec_id,
        "name": name,
        "components": components,
        "grid": [int(ordered.shape[1]), int(ordered.shape[2])],
        "real": real.data,
        "imag": imag.data,
    }


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
        raise ValueError("the result has no grid resolution (.config.resolution)")

    for name, spec in expected.items():
        phasor = _detector_phasor(result, name)
        if phasor is None:
            continue  # no field recorded: nothing to mis-shade
        components = _detector_components(result, name)
        if components and phasor.shape[0] != len(components):
            raise ValueError(
                f"detector {name!r}: the result has {phasor.shape[0]} components but "
                f"the detector records {len(components)}"
            )
        grid = tuple(int(v) for v in phasor.shape[1:])
        if not _grid_matches(state, spec, grid, resolution):
            raise ValueError(
                f"detector {name!r}: the result's {grid} grid does not match the "
                "current setup; it was recorded on a different device"
            )

def _result_resolution(result: Any) -> float | None:
    """The result's grid spacing in µm, or None (spec Q23)."""
    config = getattr(result, "config", None)
    resolution = getattr(config, "resolution", None)
    if resolution is None:
        return None
    return float(resolution) * 1e6  # meters -> µm


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
