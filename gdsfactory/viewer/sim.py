"""Sources, detectors and the run (spec R12, R13, R14, D11, D15).

The viewer holds sources and detectors as **declarative specs** (spec D11): an
:class:`ObjectSpec` is a type name, a port and a dict of editable parameters.
Nothing is created until the user retrieves the prepared objects, at which point
:func:`build_objects` calls the device's ``create_*_at`` once per spec and
returns ``(objects, config, constraints)`` for ``fdtdx.place_objects`` (spec R14
mode 2). :func:`run` extends that to mode 1: place and run in the kernel.

This module is plain Python (no ``anywidget``): :class:`ViewerState` owns the
specs, and the widget wraps it. ``fdtdx`` and ``jax`` are imported lazily, so
importing this module does not import the simulation stack (spec C3).

Glyphs are computed from a spec and the port frame (spec D11): the same frame
``create_*_at`` uses, so the pre-placement viewport matches what will be placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

__all__ = [
    "TYPES",
    "ObjectSpec",
    "RunResult",
    "SimSettings",
    "build_objects",
    "default_detector_spec",
    "default_source_spec",
    "export",
    "run",
    "spec_glyph",
]


# --------------------------------------------------------------------- registry

@dataclass(frozen=True)
class ObjectType:
    """One creatable source or detector type (spec R12).

    ``method`` is the :class:`~gdsfactory.fdtdx_stack.StackDevice` factory;
    ``plane`` marks types sized by ``port_size_mult`` (all but the point dipole);
    ``wave`` is how many wave characters the type takes (``"one"``, ``"many"``
    or ``"none"``); ``direction`` marks types with an ``in``/``out`` field;
    ``extra`` lists the additional editable parameter names it accepts.
    """

    name: str
    category: str  # "source" | "detector"
    method: str
    plane: bool = True
    wave: str = "one"
    direction: bool = False
    extra: tuple[str, ...] = ()


#: The creatable types (spec R12). The defaults named in the spec — a
#: ``ModePlaneSource`` and a ``ModeOverlapDetector`` — are the first of each.
TYPES: dict[str, ObjectType] = {
    t.name: t
    for t in (
        ObjectType("ModePlaneSource", "source", "create_mode_plane_source_at", direction=True,
                   extra=("mode_index", "filter_pol")),
        ObjectType("GaussianPlaneSource", "source", "create_gaussian_plane_source_at", direction=True),
        ObjectType("UniformPlaneSource", "source", "create_uniform_plane_source_at", direction=True),
        ObjectType("HardConstantAmplitudePlanceSource", "source",
                   "create_hard_constant_amplitude_plane_source_at", direction=True),
        ObjectType("PointDipoleSource", "source", "create_point_dipole_source_at", plane=False,
                   extra=("polarization", "source_type")),
        ObjectType("ModeOverlapDetector", "detector", "create_mode_overlap_detector_at", wave="many",
                   direction=True, extra=("mode_index", "filter_pol", "scaling_mode")),
        ObjectType("PoyntingFluxDetector", "detector", "create_poynting_flux_detector_at", wave="none",
                   direction=True, extra=("reduce_volume",)),
        ObjectType("PhasorDetector", "detector", "create_phasor_detector_at", wave="many",
                   extra=("components", "reduce_volume", "scaling_mode")),
        ObjectType("FieldDetector", "detector", "create_field_detector_at", wave="none",
                   extra=("components", "reduce_volume")),
        ObjectType("EnergyDetector", "detector", "create_energy_detector_at", wave="none",
                   extra=("reduce_volume", "aggregate")),
    )
}

#: The default type added by each gesture (spec R12, Q18).
DEFAULT_SOURCE = "ModePlaneSource"
DEFAULT_DETECTOR = "ModeOverlapDetector"


# ---------------------------------------------------------------------- models

#: The editable parameters every spec carries (spec R12). ``type`` and ``port``
#: are first-class; the rest live in ``params`` and are validated per type.
#: ``visible`` is display-only — the viewports' pre-placement toggle (R12) —
#: and is never passed to ``create_*_at``.
_COMMON = ("port_offset", "level", "direction", "name", "wavelengths", "port_size_mult", "visible")


@dataclass
class ObjectSpec:
    """A source or detector as data, not a live fdtdx object (spec D11).

    Attributes:
        type: a key of :data:`TYPES`.
        port: the port name the object sits on.
        params: the editable parameters (``port_offset``, ``port_size_mult``,
            ``level``, ``direction``, ``wavelengths`` and any type-specific
            extras). Absent keys fall back to the settings (spec R13).
        id: a stable identity for update/remove/selection. Assigned by
            :class:`~gdsfactory.viewer.state.ViewerState`; empty until then.
    """

    type: str
    port: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    @property
    def category(self) -> str:
        """``"source"`` or ``"detector"``."""
        return TYPES[self.type].category

    @property
    def name(self) -> str:
        """The object's name (``params['name']`` or ``<port>_<type>``)."""
        return str(self.params.get("name") or f"{self.port}_{self.type}")

    def to_wire(self) -> dict[str, Any]:
        """The spec as JSON for the browser (spec D5)."""
        return {"id": self.id, "type": self.type, "port": self.port, "params": dict(self.params)}

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> ObjectSpec:
        """Rebuilds a spec from a browser patch, validating the type and port."""
        type_name = str(data.get("type", ""))
        if type_name not in TYPES:
            raise ValueError(f"Unknown source/detector type {type_name!r}")
        port = str(data.get("port", ""))
        if not port:
            raise ValueError("A source/detector needs a port")
        params = dict(data.get("params") or {})
        unknown = set(params) - set(_COMMON) - set(TYPES[type_name].extra)
        if unknown:
            raise ValueError(f"Unknown parameters for {type_name!r}: {sorted(unknown)}")
        return cls(type=type_name, port=port, params=params, id=str(data.get("id") or ""))


@dataclass
class SimSettings:
    """The setup-wide simulation settings (spec R13).

    Attributes:
        wavelength_um: the default wavelength, in µm, for new sources/detectors.
        pulse_width_frac: the Gaussian pulse's spectral width as a fraction of
            the center frequency; ``0`` means a single-frequency (CW) run.
        port_size_mult: ``(a, b)`` — a new plane is ``a`` port widths wide and
            ``b`` level heights tall.
        pml_thickness: the PML thickness, in grid cells.
        resolution_um: the grid resolution, in µm.
        time_s: the simulation time, in seconds.
    """

    wavelength_um: float = 1.55
    pulse_width_frac: float = 0.0
    port_size_mult: tuple[float, float] = (3.0, 6.0)
    pml_thickness: int = 10
    resolution_um: float = 0.02
    time_s: float = 400e-15

    def to_wire(self) -> dict[str, Any]:
        """The settings as JSON for the browser (spec R13)."""
        return {
            "wavelength_um": float(self.wavelength_um),
            "pulse_width_frac": float(self.pulse_width_frac),
            "port_size_mult": [float(self.port_size_mult[0]), float(self.port_size_mult[1])],
            "pml_thickness": int(self.pml_thickness),
            "resolution_um": float(self.resolution_um),
            "time_s": float(self.time_s),
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SimSettings:
        """Rebuilds settings from a browser patch, validating each field."""
        defaults = cls()
        mult = data.get("port_size_mult", defaults.port_size_mult)
        if not (isinstance(mult, (list, tuple)) and len(mult) == 2 and min(mult) > 0):
            raise ValueError(f"port_size_mult must be two positive numbers, got {mult!r}")
        wavelength = float(data.get("wavelength_um", defaults.wavelength_um))
        resolution = float(data.get("resolution_um", defaults.resolution_um))
        time_s = float(data.get("time_s", defaults.time_s))
        pml = int(data.get("pml_thickness", defaults.pml_thickness))
        if wavelength <= 0 or resolution <= 0 or time_s <= 0 or pml < 0:
            raise ValueError("Settings must be positive (wavelength, resolution, time) or non-negative (PML)")
        return cls(
            wavelength_um=wavelength,
            pulse_width_frac=float(data.get("pulse_width_frac", defaults.pulse_width_frac)),
            port_size_mult=(float(mult[0]), float(mult[1])),
            pml_thickness=pml,
            resolution_um=resolution,
            time_s=time_s,
        )


class RunResult(NamedTuple):
    """The outcome of a mode-1 run (spec R14, Q20).

    ``arrays`` is the fdtdx :class:`ArrayContainer`; detector results are read
    from ``arrays.detector_states`` keyed by detector name. ``objects`` is the
    placed :class:`ObjectContainer`, so ``objects[name].compute_overlap(...)``
    works as in the reference example.
    """

    objects: Any
    arrays: Any
    config: Any


# ------------------------------------------------------------------- defaults

def default_source_spec(port: str, settings: SimSettings) -> ObjectSpec:
    """The default source added by the "New source" control (spec R12)."""
    return ObjectSpec(DEFAULT_SOURCE, port, _default_params(DEFAULT_SOURCE, port, settings))


def default_detector_spec(port: str, settings: SimSettings) -> ObjectSpec:
    """The default detector added by a 3D port double-click (spec R12, Q18)."""
    return ObjectSpec(DEFAULT_DETECTOR, port, _default_params(DEFAULT_DETECTOR, port, settings))


def _default_params(type_name: str, port: str, settings: SimSettings) -> dict[str, Any]:
    """The starting parameters for a new spec, taken from the settings (spec Q18).

    A spec on a simulation-volume plane (spec R12) defaults to
    ``port_size_mult=(1, 1)``: the plane's width and height are already the
    volume's in-plane extents, so a port's multiplier would overflow the volume.
    """
    kind = TYPES[type_name]
    params: dict[str, Any] = {"port_offset": 0.0, "level": None, "visible": True}
    if kind.plane:
        from gdsfactory.fdtdx_stack import PLANE_PORTS

        params["port_size_mult"] = [1.0, 1.0] if port in PLANE_PORTS else list(settings.port_size_mult)
    if kind.wave != "none":
        params["wavelengths"] = [settings.wavelength_um]
    if kind.direction:
        params["direction"] = "in" if kind.category == "source" else "out"
    return params


# ------------------------------------------------------------------- building

def _wave_character(wavelength_um: float) -> Any:
    """One ``fdtdx.WaveCharacter`` for a wavelength in µm."""
    import fdtdx

    return fdtdx.WaveCharacter(wavelength=wavelength_um * 1e-6)


def _wave_characters(spec: ObjectSpec, settings: SimSettings) -> tuple[Any, ...]:
    """The spec's wavelengths, or the settings default (spec R13, Q18)."""
    wavelengths = spec.params.get("wavelengths") or [settings.wavelength_um]
    return tuple(_wave_character(float(w)) for w in wavelengths)


def _temporal_profile(settings: SimSettings, wave: Any) -> Any | None:
    """A Gaussian pulse from the settings, or None for a single frequency."""
    import fdtdx

    if settings.pulse_width_frac <= 0:
        return None
    return fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(frequency=settings.pulse_width_frac * wave.get_frequency()),
    )


def _create(stack: Any, spec: ObjectSpec, settings: SimSettings) -> tuple[Any, list[Any]]:
    """Calls the device's ``create_*_at`` for one spec (spec D11).

    Builds the keyword arguments from the spec's parameters, falling back to the
    settings for anything the spec leaves unset. Every type is created through
    its ``create_<object>_at`` factory, never by hand-constructing an fdtdx
    object, so the port frame, plane sizing and constraints come for free.
    """
    kind = TYPES[spec.type]
    method = getattr(stack, kind.method)
    params = spec.params
    kwargs: dict[str, Any] = {
        "port": spec.port,
        "port_offset": float(params.get("port_offset", 0.0)),
        "name": spec.name,
    }
    if params.get("level") is not None:
        kwargs["level"] = params["level"]
    if kind.plane:
        mult = params.get("port_size_mult") or list(settings.port_size_mult)
        kwargs["port_size_mult"] = (float(mult[0]), float(mult[1]))

    waves = _wave_characters(spec, settings)
    wave = waves[0] if waves else _wave_character(settings.wavelength_um)
    if kind.wave == "one":
        kwargs["wave_character"] = wave
        kwargs["temporal_profile"] = _temporal_profile(settings, wave)
    elif kind.wave == "many":
        kwargs["wave_characters"] = waves

    if kind.direction:
        kwargs["direction"] = params.get("direction", "in" if kind.category == "source" else "out")

    for key in kind.extra:
        if key in params:
            kwargs[key] = params[key]
    # pulse sources/detectors need the matching scaling mode
    if kind.category == "detector" and settings.pulse_width_frac > 0 and "scaling_mode" in kind.extra:
        kwargs.setdefault("scaling_mode", "pulse")
    return method(**kwargs)


def build_objects(stack: Any, state: Any) -> tuple[list[Any], Any, list[Any]]:
    """The prepared objects, config and constraints (spec R14 mode 2, D11).

    Creates the simulation volume, the device, every source/detector from its
    spec (via ``create_*_at``), and the PML boundaries from the settings. This
    is what ``Viewer.objects`` / ``Viewer.config`` / ``Viewer.constraints``
    return, and what :func:`run` places.

    Args:
        stack: the :class:`~gdsfactory.fdtdx_stack.StackDevice`.
        state: a :class:`~gdsfactory.viewer.state.ViewerState` holding the specs
            and settings.

    Returns:
        ``(objects, config, constraints)`` for ``fdtdx.place_objects``.
    """
    import fdtdx

    settings: SimSettings = state.settings
    volume, constraints = stack.get_simulation_volume()
    objects: list[Any] = [volume, stack]
    constraints = list(constraints)

    for spec in state.specs:
        obj, cons = _create(stack, spec, settings)
        objects.append(obj)
        constraints += cons

    config = fdtdx.SimulationConfig(
        time=settings.time_s, resolution=settings.resolution_um * 1e-6
    )
    pml = fdtdx.BoundaryConfig.from_uniform_bound(thickness=settings.pml_thickness)
    boundaries, cons = fdtdx.boundary_objects_from_config(pml, volume)
    objects += list(boundaries.values())
    constraints += cons
    return objects, config, constraints


def export(stack: Any, state: Any) -> tuple[list[Any], Any, list[Any]]:
    """Alias of :func:`build_objects` for the mode-2 retrieval (spec R14)."""
    return build_objects(stack, state)


def run(
    stack: Any, state: Any, *, key: Any = None, progress: Any = None, phase: Any = None
) -> RunResult:
    """Places and runs the simulation in the kernel (spec R14 mode 1).

    Blocks until the run finishes; callers that must not block (the widget's
    background worker, spec D15) run this on a thread. Returns a
    :class:`RunResult`; detector results are on ``result.arrays.detector_states``.

    The run has three phases, and only the last reports a fraction. Placement
    and JAX compilation dominate the wall time on a large volume, and fdtdx
    reports no progress for either, so ``phase`` tells the caller which one it
    is in:

    ``"placing"``
        building the objects and placing them on the grid;
    ``"compiling"``
        inside ``run_fdtd`` before the first time step (mostly XLA compile);
    ``"stepping"``
        time-stepping, where ``progress`` fires.

    Args:
        stack: the :class:`~gdsfactory.fdtdx_stack.StackDevice`.
        state: the :class:`~gdsfactory.viewer.state.ViewerState`.
        key: an optional ``jax`` PRNG key (default: ``PRNGKey(0)``).
        progress: an optional ``callable(step, total)`` fdtdx calls as the run
            advances (spec D15). When given, fdtdx's own tqdm bar is turned off,
            since the caller is drawing the progress instead.
        phase: an optional ``callable(name)`` called once per phase above.
    """
    import fdtdx
    import jax

    from gdsfactory.fdtdx_stack import apply_stack, init_stack_params

    def _phase(name: str) -> None:
        if phase is not None:
            phase(name)

    _phase("placing")
    objects, config, constraints = build_objects(stack, state)
    key = jax.random.PRNGKey(0) if key is None else key
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, objects, _ = apply_stack(arrays, objects, {**params, **init_stack_params(objects)}, key)

    # the first time step implies compilation is done, so the phase flips there
    stepping = False

    def _progress(step: int, total: int) -> None:
        nonlocal stepping
        if not stepping:
            stepping = True
            _phase("stepping")
        if progress is not None:
            progress(step, total)

    # the wrapper is installed when either callback needs it: ``phase`` needs it
    # to see the flip into "stepping" even when nobody draws a fraction
    use_callback = progress is not None or phase is not None

    _phase("compiling")
    _, arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=progress is None,
        progress_callback=_progress if use_callback else None,
    )
    return RunResult(objects=objects, arrays=arrays, config=config)


# --------------------------------------------------------------------- glyphs

def spec_glyph(stack: Any, spec: ObjectSpec, origin: tuple[float, float, float],
               settings: SimSettings) -> dict[str, Any]:
    """A source/detector's glyph, in the local frame the meshes use (spec D11).

    Reads the port frame — the same one ``create_*_at`` uses — so the drawn
    plane matches the placed one. A plane glyph is a center, a normal axis and
    two half-extents; a point (dipole) has zero half-extents.

    Args:
        stack: the device.
        spec: the source/detector spec.
        origin: the box center the meshes are shifted by (spec C5).
        settings: the settings supplying any parameter the spec leaves unset.

    Returns:
        ``{"id", "name", "kind", "type", "port", "center", "axis", "u_half",
        "v_half", "direction"}`` with the center in the local frame. ``id`` is
        the spec's identity, so the browser can match a selection to a glyph
        (two specs may share a name, spec R12).
    """
    kind = TYPES[spec.type]
    try:
        p, axis, _sign, center, height = stack._port_frame(
            spec.port, spec.params.get("level"), float(spec.params.get("port_offset", 0.0))
        )
    except (KeyError, ValueError):
        return _empty_glyph(spec)

    if kind.plane:
        mult = spec.params.get("port_size_mult") or list(settings.port_size_mult)
        u_half = 0.5 * float(mult[0]) * float(p.width)
        v_half = 0.5 * float(mult[1]) * float(height)
    else:
        u_half = v_half = 0.0

    ox, oy, oz = origin
    return {
        "id": spec.id,
        "name": spec.name,
        "kind": spec.category,
        "type": spec.type,
        "port": spec.port,
        "center": [float(center[0]) - ox, float(center[1]) - oy, float(center[2]) - oz],
        "axis": int(axis),
        "u_half": u_half,
        "v_half": v_half,
        "direction": spec.params.get("direction"),
        "visible": bool(spec.params.get("visible", True)),
    }


def _empty_glyph(spec: ObjectSpec) -> dict[str, Any]:
    """A degenerate glyph for a spec whose port frame cannot be resolved yet."""
    return {
        "id": spec.id,
        "name": spec.name,
        "kind": spec.category,
        "type": spec.type,
        "port": spec.port,
        "center": [0.0, 0.0, 0.0],
        "axis": 0,
        "u_half": 0.0,
        "v_half": 0.0,
        "direction": spec.params.get("direction"),
        "visible": bool(spec.params.get("visible", True)),
        "error": True,
    }
