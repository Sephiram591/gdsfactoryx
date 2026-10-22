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

import functools
import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

__all__ = [
    "AVERAGINGS",
    "TYPES",
    "CancelCondition",
    "ObjectSpec",
    "RunResult",
    "RunStopped",
    "SimSettings",
    "build_objects",
    "default_detector_spec",
    "default_source_spec",
    "export",
    "launch_code",
    "run",
    "spec_glyph",
    "validate_sampling",
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
                   direction=True, extra=("mode_index", "filter_pol", "scaling_mode", "dft_subsample")),
        ObjectType("PoyntingFluxDetector", "detector", "create_poynting_flux_detector_at", wave="none",
                   direction=True, extra=("reduce_volume", "record_interval")),
        ObjectType("PhasorDetector", "detector", "create_phasor_detector_at", wave="many",
                   extra=("components", "reduce_volume", "scaling_mode", "dft_subsample")),
        ObjectType("FieldDetector", "detector", "create_field_detector_at", wave="none",
                   extra=("components", "reduce_volume", "record_interval")),
        ObjectType("EnergyDetector", "detector", "create_energy_detector_at", wave="none",
                   extra=("reduce_volume", "aggregate", "record_interval")),
    )
}

#: The temporal-downsampling extras (spec R12). ``record_interval`` is the
#: viewer's name for a time-domain detector's recording stride — it becomes
#: fdtdx's ``switch=OnOffSwitch(interval=n)``, so only every n-th step is
#: stored — and ``dft_subsample`` is fdtdx's own phasor-DFT stride: a positive
#: integer, or ``"auto"`` to derive it from the highest recorded frequency.
_SAMPLING = ("record_interval", "dft_subsample")


def validate_sampling(params: dict[str, Any]) -> None:
    """Refuses a bad temporal-downsampling parameter (spec R12).

    ``record_interval`` must be a whole number of at least 1; ``dft_subsample``
    the same, or ``"auto"``. ``None`` means unset (fdtdx's every-step default).

    Raises:
        ValueError: on a value fdtdx would reject.
    """
    for key in _SAMPLING:
        value = params.get(key)
        if value is None or (key == "dft_subsample" and value == "auto"):
            continue
        whole = isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer()
        if not whole or value < 1:
            expected = "a whole number >= 1" + (' or "auto"' if key == "dft_subsample" else "")
            raise ValueError(f"{key} must be {expected}, got {value!r}")

#: The default type added by each gesture (spec R12, Q18).
DEFAULT_SOURCE = "ModePlaneSource"
DEFAULT_DETECTOR = "ModeOverlapDetector"


# ---------------------------------------------------------------------- models

#: The editable parameters every spec carries (spec R12). ``type`` and ``port``
#: are first-class; the rest live in ``params`` and are validated per type.
#: ``visible`` is display-only — the viewports' pre-placement toggle (R12) —
#: and is never passed to ``create_*_at``.
_COMMON = ("port_offset", "level", "direction", "name", "wavelengths", "port_size_mult", "visible")

#: The absolute-placement parameters a spec with ``port == "None"`` carries
#: (spec R18): the plane's normal axis (0=x, 1=y, 2=z), the center in µm in
#: stack coordinates, and the in-plane sizes in µm (plane types only). Never
#: passed to ``create_*_at`` — they feed :func:`_create_absolute`.
_ABSOLUTE = ("axis", "center", "size")

#: The detector display parameters (spec R15, D16). Display-only, like
#: ``visible``: they style the field overlay and are never passed to
#: ``create_*_at``. Only detectors carry them.
_FIELD_PARAMS = (
    "field_component",
    "field_transform",
    "field_color_low",
    "field_color_high",
    "field_opacity",
    "field_visible",
    "field_freq",
    "field_outline",
    "field_range",
)

#: The display transforms a detector's field can take (spec R15).
FIELD_TRANSFORMS = ("real", "imag", "abs", "phase", "logabs")

#: The default field-overlay colors (spec D16): a cool blue and a warm red.
FIELD_COLOR_LOW = "#3b6fd6"
FIELD_COLOR_HIGH = "#d64b3b"

#: The pseudo-component for the total electric-field magnitude, ``|E|`` (spec
#: R15) — the browser's ``TOTAL_COMPONENT`` — and a detector's default one.
TOTAL_COMPONENT = "|E|"


def _default_component(params: dict[str, Any]) -> str | None:
    """The component a detector is shaded with until one is picked (spec R15).

    ``|E|`` when the detector records an E component — fdtdx records all six
    unless ``components`` says otherwise — else its first recorded component.
    """
    components = params.get("components")
    if components is None:
        return TOTAL_COMPONENT
    names = [str(c) for c in ([components] if isinstance(components, str) else components)]
    if any(name.startswith("E") for name in names):
        return TOTAL_COMPONENT
    return names[0] if names else None


@dataclass
class ObjectSpec:
    """A source or detector as data, not a live fdtdx object (spec D11).

    Attributes:
        type: a key of :data:`TYPES`.
        port: the port name the object sits on, or ``"None"`` for absolute
            placement (spec R18).
        params: the editable parameters (``port_offset``, ``port_size_mult``,
            ``level``, ``direction``, ``wavelengths``, the absolute-placement
            keys and any type-specific extras). Absent keys fall back to the
            settings (spec R13).
        id: a stable identity for update/remove/selection. Assigned by
            :class:`~gdsfactory.viewer.state.ViewerState`; empty until then.
        glyph: an override for the wire glyph (spec D18). Set on adopted
            foreign detectors, whose glyph is derived from their placed
            ``grid_slice`` rather than a port frame; never sent through
            ``to_wire`` — the wire glyph travels in ``state["glyphs"]``.
    """

    type: str
    port: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    glyph: dict[str, Any] | None = None

    @property
    def category(self) -> str:
        """``"source"`` or ``"detector"``."""
        return TYPES[self.type].category

    @property
    def name(self) -> str:
        """The object's name (``params['name']`` or ``<port>_<type>``)."""
        return str(self.params.get("name") or f"{self.port}_{self.type}")

    @property
    def absolute(self) -> bool:
        """Whether this spec is placed absolutely (``port == "None"``, R18)."""
        return self.port == "None"

    def field_params(self) -> dict[str, Any]:
        """The detector's display settings with defaults filled in (spec R15).

        Only a detector has these; a source returns an empty dict. Absent keys
        fall back to the defaults, so the browser always gets a complete set.
        An unpicked component defaults to ``|E|`` (see
        :func:`_default_component`); one set to None — "(none)" picked — stays
        None. The outline defaults off.
        """
        if self.category != "detector":
            return {}
        params = self.params
        return {
            "field_component": params["field_component"] if "field_component" in params
            else _default_component(params),
            "field_transform": str(self.params.get("field_transform") or "abs"),
            "field_color_low": str(self.params.get("field_color_low") or FIELD_COLOR_LOW),
            "field_color_high": str(self.params.get("field_color_high") or FIELD_COLOR_HIGH),
            "field_opacity": _field_opacity(self.params.get("field_opacity", 1.0)),
            "field_visible": bool(self.params.get("field_visible", True)),
            "field_freq": int(self.params.get("field_freq", 0)),
            "field_outline": bool(self.params.get("field_outline", False)),
            # the colour scale's fixed [min, max], in the transform's units; None
            # means the browser spans the whole recording (every frame)
            "field_range": _field_range(self.params.get("field_range")),
        }

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
        allowed = set(_COMMON) - ({"port_offset", "level", "port_size_mult"} if port == "None" else set())
        if port == "None":
            allowed |= set(_ABSOLUTE)
        unknown = set(params) - allowed - set(TYPES[type_name].extra)
        if unknown:
            raise ValueError(f"Unknown parameters for {type_name!r}: {sorted(unknown)}")
        _validate_absolute(port, params)
        validate_sampling(params)
        return cls(type=type_name, port=port, params=params, id=str(data.get("id") or ""))


#: the permittivity averagings a :class:`SimSettings` accepts (spec R13): the
#: values of :data:`gdsfactory.fdtdx_stack.Averaging`, repeated here so that
#: importing this module does not import the simulation stack (spec C3)
AVERAGINGS: tuple[str, ...] = ("inverse", "subpixel", "subpixel_full")


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
        boundary_offset: ``(x+, x-, y+, y-, z+, z-)`` in µm — how far the
            simulation box's faces move off the geometry's bounding box (the
            ``StackDevice`` box, spec R13). Each value moves its face along the
            axis, positive towards +x, +y or +z, so ``(1, -1, 0, 0, 0, 0)``
            grows the box by 1 µm on each x side; the geometry itself does not
            change.
    """

    wavelength_um: float = 1.55
    pulse_width_frac: float = 0.0
    port_size_mult: tuple[float, float] = (3.0, 6.0)
    pml_thickness: int = 10
    resolution_um: float = 0.02
    time_s: float = 400e-15
    boundary_offset: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    #: the number of decades below a plane's maximum that ``logabs`` floors at
    #: (spec D16, Q22). One value for every detector's overlay; the bottom of the
    #: ``logabs`` color ramp, so a true zero maps to the floor, not to ``-inf``.
    log_range: float = 6.0
    #: how the device averages the permittivity at material interfaces (spec
    #: R13), one of :data:`AVERAGINGS`: ``"subpixel"`` (Kottke, the diagonal of
    #: the averaged tensor), ``"subpixel_full"`` (the full tensor, with the
    #: off-diagonal entries oblique and sloped walls need; about 5x the time
    #: per step and 1.8x the memory) or ``"inverse"`` (fdtdx's isotropic
    #: inverse average). The viewer seeds it from the device's own
    #: ``averaging`` and places the device with it.
    averaging: str = "subpixel"
    #: whether a run the user stops keeps its partial result (spec D17). When
    #: False (the default) a stopped run is discarded and the previous result
    #: stays on screen; when True its detectors' partial fields are shown.
    keep_partial: bool = False

    def to_wire(self) -> dict[str, Any]:
        """The settings as JSON for the browser (spec R13)."""
        return {
            "wavelength_um": float(self.wavelength_um),
            "pulse_width_frac": float(self.pulse_width_frac),
            "port_size_mult": [float(self.port_size_mult[0]), float(self.port_size_mult[1])],
            "pml_thickness": int(self.pml_thickness),
            "resolution_um": float(self.resolution_um),
            "time_s": float(self.time_s),
            "boundary_offset": [float(v) for v in self.boundary_offset],
            "log_range": float(self.log_range),
            "averaging": str(self.averaging),
            "keep_partial": bool(self.keep_partial),
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
        offset = data.get("boundary_offset", defaults.boundary_offset)
        if not (isinstance(offset, (list, tuple)) and len(offset) == 6):
            raise ValueError(f"boundary_offset must be 6 numbers (x+, x-, y+, y-, z+, z-), got {offset!r}")
        offset = tuple(float(v) for v in offset)
        if not all(math.isfinite(v) for v in offset):
            raise ValueError(f"boundary_offset must be finite, got {offset!r}")
        log_range = float(data.get("log_range", defaults.log_range))
        if not math.isfinite(log_range) or log_range <= 0:
            raise ValueError(f"log_range must be a positive number of decades, got {log_range!r}")
        averaging = str(data.get("averaging", defaults.averaging))
        if averaging not in AVERAGINGS:
            raise ValueError(f"averaging must be one of {AVERAGINGS}, got {averaging!r}")
        return cls(
            wavelength_um=wavelength,
            pulse_width_frac=float(data.get("pulse_width_frac", defaults.pulse_width_frac)),
            port_size_mult=(float(mult[0]), float(mult[1])),
            pml_thickness=pml,
            resolution_um=resolution,
            time_s=time_s,
            boundary_offset=offset,
            log_range=log_range,
            averaging=averaging,
            keep_partial=bool(data.get("keep_partial", defaults.keep_partial)),
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


class RunStopped(Exception):
    """A mode-1 run ended because the user asked it to stop (spec D17).

    Raised when a cancelled run's partial result is being *discarded* (the
    default). When the settings say to keep the partial run, :func:`run` returns
    its :class:`RunResult` instead, and the widget adopts the partial fields.
    """


def _stopping_condition_type() -> Any:
    """The fdtdx ``StoppingCondition`` base, imported lazily (spec C3).

    ``fdtdx`` is an optional, heavy import, so the condition's base class is
    resolved only when a cancellable run is actually set up.
    """
    from fdtdx.fdtd.stop_conditions import StoppingCondition

    return StoppingCondition


@functools.lru_cache(maxsize=1)
def _cancel_condition_class() -> Any:
    """The ``CancelCondition`` class, built once (spec D17).

    Defining it lazily keeps ``fdtdx`` off this module's import path (spec C3);
    caching the class keeps its identity stable across runs, so JAX sees the same
    traced computation and does not recompile.
    """
    import jax
    import jax.numpy as jnp
    from fdtdx.core.jax.pytrees import autoinit, frozen_field

    base = _stopping_condition_type()

    @autoinit
    class CancelCondition(base):  # type: ignore[misc, valid-type]
        """Time limit plus a cooperative host-flag stop (spec D17)."""

        max_steps: int = frozen_field(default=0)
        cancel_flag: Any = frozen_field(default=None)

        def setup(self, state: Any, config: Any, objects: Any) -> Any:
            return self.aset("max_steps", config.time_steps_total, create_new_ok=True)

        def _validate(self, state: Any, config: Any, objects: Any) -> None:
            pass

        def __call__(self, state: Any, config: Any, objects: Any) -> jax.Array:
            curr, _ = state

            def _host_read(step: Any) -> Any:
                flag_ = self.cancel_flag
                return bool(flag_ is not None and flag_.is_set())

            stop = jax.experimental.io_callback(
                _host_read,
                result_shape_dtypes=jax.ShapeDtypeStruct((), jnp.bool_),
                step=curr,
                ordered=True,
            )
            return jnp.logical_and(self.max_steps > curr, jnp.logical_not(stop))

    return CancelCondition


def make_cancel_condition(flag: Any) -> Any:
    """A ``StoppingCondition`` that ends the loop when ``flag`` is set (spec D17).

    The loop's own condition is the only place a stop can be applied *exactly*
    and still return the arrays: raising out of the ``progress_callback``
    unwinds ``run_fdtd`` before it returns, so no partial result survives. The
    condition therefore reproduces ``TimeStepCondition``'s time limit itself
    (a custom condition *replaces* the default, so the run would otherwise never
    end) and ANDs in a per-step read of the host flag via ``io_callback``. fdtdx
    only ever uses ``io_callback`` in the loop's body; using it in the condition
    is legal and makes the stop land on the exact requested step.

    Args:
        flag: a ``threading.Event`` (or anything with ``is_set()``); it is read
            on the host, never traced, so it may be any Python object.

    Returns:
        A ``StoppingCondition`` instance, ready to pass to ``fdtdx.run_fdtd``.
    """
    return _cancel_condition_class()(cancel_flag=flag)


# ------------------------------------------------------------------- defaults

def _field_opacity(value: Any) -> float:
    """A clamped field-overlay opacity in [0, 1] (spec R15)."""
    return min(1.0, max(0.0, float(value)))


def _field_range(value: Any) -> list[float] | None:
    """A detector's fixed colour-scale ``[min, max]``, or None for auto (spec R15).

    Raises:
        ValueError: unless ``value`` is None or two finite numbers, min < max.
    """
    if value is None:
        return None
    try:
        low, high = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field_range must be [min, max] or null, got {value!r}") from exc
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(f"field_range must be two finite numbers with min < max, got {value!r}")
    return [low, high]


def _validate_absolute(port: str, params: dict[str, Any]) -> None:
    """Checks a spec's absolute-placement params (spec R18).

    Only invoked for ``port == "None"``. ``axis`` must be 0, 1 or 2;
    ``center`` three finite floats (µm); ``size`` two positive floats (µm).
    """
    if port != "None":
        return
    axis = params.get("axis")
    if not (isinstance(axis, int) and not isinstance(axis, bool) and axis in (0, 1, 2)):
        raise ValueError(f"axis must be 0 (x), 1 (y) or 2 (z), got {axis!r}")
    center = params.get("center")
    if not (isinstance(center, (list, tuple)) and len(center) == 3
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in center)):
        raise ValueError(f"center must be three finite numbers (x, y, z in um), got {center!r}")
    size = params.get("size")
    if size is not None and not (
        isinstance(size, (list, tuple)) and len(size) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in size)
    ):
        raise ValueError(f"size must be two positive numbers (u, v in um), got {size!r}")


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


def _create_absolute(spec: ObjectSpec, settings: SimSettings,
                     volume: Any | None, stack: Any) -> tuple[Any, list[Any]]:
    """Builds a source/detector at an absolute position (spec R18).

    The same object construction ``StackDevice._at_port`` does — a plane is one
    grid cell thick along its normal axis and sized by ``partial_real_shape`` —
    and the placement constraint references the stack the way
    ``StackDevice._place_at`` does: margins from the box center (µm, stack
    coordinates) to the spec's ``center``, so absolute and port placement land
    in the same frame. The wave and extra-params handling matches
    :func:`_create`.
    """
    import fdtdx

    if volume is None:
        raise ValueError(f"spec {spec.name!r}: absolute placement needs the simulation volume")
    kind = TYPES[spec.type]
    params = spec.params
    axis = int(params.get("axis", 2))
    center = tuple(float(v) for v in params["center"])  # validated by from_wire
    cls = getattr(fdtdx, spec.type)

    real: list[float | None] = [None, None, None]
    grid: list[int | None] = [None, None, None]
    if kind.plane:
        size = params.get("size")
        if not size:
            raise ValueError(f"spec {spec.name!r}: a plane needs a size (u, v in um)")
        u_axis, v_axis = _IN_PLANE[axis]
        grid[axis] = 1
        real[u_axis] = float(size[0]) * 1e-6
        real[v_axis] = float(size[1]) * 1e-6

    fields: dict[str, Any] = {"name": spec.name}
    waves = _wave_characters(spec, settings)
    wave = waves[0] if waves else _wave_character(settings.wavelength_um)
    if kind.wave == "one":
        fields["wave_character"] = wave
        profile = _temporal_profile(settings, wave)
        if profile is not None:
            fields["temporal_profile"] = profile
    elif kind.wave == "many":
        fields["wave_characters"] = waves
    if kind.direction:
        # an absolute plane's direction is its normal axis' orientation: a
        # source "in" along +axis launches towards +axis (spec R18)
        fields["direction"] = "+" if params.get("direction", "in") == "in" else "-"
    fields.update(_extra_kwargs(kind, params))
    if kind.category == "detector" and settings.pulse_width_frac > 0 and "scaling_mode" in kind.extra:
        fields.setdefault("scaling_mode", "pulse")
    obj = cls(partial_real_shape=tuple(real), partial_grid_shape=tuple(grid), **fields)

    reference = (float(stack.center[0]), float(stack.center[1]), float(stack.z_center))
    margins = tuple((c - r) * 1e-6 for c, r in zip(center, reference, strict=True))
    # a detector plane is anchored by the point fdtdx samples it at, like
    # ``StackDevice._at_port`` does (see ``detector_plane_own_positions``)
    from gdsfactory.fdtdx_stack import detector_plane_own_positions

    own = detector_plane_own_positions(axis) if kind.plane and kind.category == "detector" else (0, 0, 0)
    cons = [
        obj.place_relative_to(
            stack, axes=(0, 1, 2), own_positions=own, other_positions=(0, 0, 0),
            margins=margins,
        )
    ]
    return obj, cons


def _create(stack: Any, spec: ObjectSpec, settings: SimSettings,
            volume: Any | None = None) -> tuple[Any, list[Any]]:
    """Calls the device's ``create_*_at`` for one spec (spec D11).

    Builds the keyword arguments from the spec's parameters, falling back to the
    settings for anything the spec leaves unset. Every type is created through
    its ``create_<object>_at`` factory, never by hand-constructing an fdtdx
    object, so the port frame, plane sizing and constraints come for free.
    """
    kind = TYPES[spec.type]
    if spec.absolute:
        return _create_absolute(spec, settings, volume, stack)
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

    kwargs.update(_extra_kwargs(kind, params))
    # pulse sources/detectors need the matching scaling mode
    if kind.category == "detector" and settings.pulse_width_frac > 0 and "scaling_mode" in kind.extra:
        kwargs.setdefault("scaling_mode", "pulse")
    return method(**kwargs)


def _extra_kwargs(kind: ObjectType, params: dict[str, Any]) -> dict[str, Any]:
    """The fdtdx keyword arguments for a spec's type extras (spec R12).

    Every extra passes through by name except the temporal downsampling
    (:data:`_SAMPLING`): ``record_interval`` becomes fdtdx's
    ``switch=OnOffSwitch(interval=n)``, and a whole-number ``dft_subsample`` is
    passed as an ``int``. A value of 1 (or unset) keeps fdtdx's every-step
    default, so nothing is passed. :func:`_extra_args` emits the same as source.
    """
    kwargs: dict[str, Any] = {}
    for key in kind.extra:
        value = params.get(key)
        if key not in params or (key in _SAMPLING and value is None):
            continue
        if key == "record_interval":
            if int(value) > 1:
                import fdtdx

                kwargs["switch"] = fdtdx.OnOffSwitch(interval=int(value))
        elif key == "dft_subsample":
            kwargs[key] = value if value == "auto" else int(value)
        else:
            kwargs[key] = value
    return kwargs


def _extra_args(kind: ObjectType, params: dict[str, Any]) -> list[str]:
    """``key=value`` source text for a spec's type extras (spec R14 mode 3).

    The emitted twin of :func:`_extra_kwargs`, so the launch code builds the
    same objects ``build_objects`` does.
    """
    args: list[str] = []
    for key in kind.extra:
        value = params.get(key)
        if key not in params or (key in _SAMPLING and value is None):
            continue
        if key == "record_interval":
            if int(value) > 1:
                args.append(f"switch=fdtdx.OnOffSwitch(interval={int(value)})")
        elif key == "dft_subsample":
            args.append(f"dft_subsample={value if value == 'auto' else int(value)!r}")
        else:
            args.append(f"{key}={_repr(value)}")
    return args


def _pml_coeffs_in_config_dtype(objects: Any, config: Any) -> Any:
    """Re-casts the PMLs' CPML coefficients to ``config.dtype`` (fdtdx 0.6.3).

    Importing gdsfactory enables jax x64 globally (``gdsfactory._jax``), and
    fdtdx's ``place_on_grid`` then derives ``pml_a/b``/``inv_kappa`` as float64
    regardless of ``config.dtype`` — so the CPML update ``b * psi + a * d``
    promotes the float32 psi fields and ``eqxi.while_loop`` rejects the body's
    changed output structure. Casting the coefficients back keeps the loop's
    tree structure stable.
    """
    import jax.numpy as jnp

    from fdtdx.fdtd.container import ObjectContainer

    if objects is None or getattr(objects, "object_list", None) is None:
        return objects
    out = []
    for obj in objects.object_list:
        if type(obj).__name__ == "PerfectlyMatchedLayer":
            for name in ("pml_a_E", "pml_b_E", "inv_kappa_E", "pml_a_H", "pml_b_H", "inv_kappa_H"):
                value = getattr(obj, name, None)
                if value is not None:
                    obj = obj.aset(name, value.astype(config.dtype))
        out.append(obj)
    return ObjectContainer(object_list=out, volume_idx=objects.volume_idx)


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
        if spec.id in getattr(state, "adopted", ()):
            continue  # adopted foreign detectors are already placed (D18)
        obj, cons = _create(stack, spec, settings, volume=volume)
        objects.append(obj)
        constraints += cons

    config = fdtdx.SimulationConfig(
        time=settings.time_s,
        grid=fdtdx.core.grid.UniformGrid(spacing=settings.resolution_um * 1e-6),
    )
    pml = fdtdx.BoundaryConfig.from_uniform_bound(thickness=settings.pml_thickness)
    boundaries, cons = fdtdx.boundary_objects_from_config(pml, volume)
    objects += list(boundaries.values())
    constraints += cons
    return objects, config, constraints


def export(stack: Any, state: Any) -> tuple[list[Any], Any, list[Any]]:
    """Alias of :func:`build_objects` for the mode-2 retrieval (spec R14)."""
    return build_objects(stack, state)


# ------------------------------------------------------------------- mode 3

def _num(value: Any) -> str:
    """A float as source text; whole numbers kept integer-shaped (``1``, not ``1.0``)."""
    f = float(value)
    return str(int(f)) if f.is_integer() and abs(f) < 1e16 else repr(f)


def _seq(values: Any) -> str:
    """A tuple literal from a sequence of numbers."""
    return "(" + ", ".join(_num(v) for v in values) + ",)"


def _repr(value: Any) -> str:
    """Source text for a parameter value (spec R12 extras)."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_repr(v) for v in value) + "]"
    return repr(value)


_LAUNCH_HEADER = """\
# --- viewer launch code (spec R14 mode 3) ------------------------------------
# Assumes `stack` (the StackDevice) is in scope. Appending this tail to the
# script that defines it builds the same setup the viewer shows. Run it on a
# GPU allocation."""


def _waves_tuple(wavelengths: Any) -> str:
    """Source text for the ``wave_characters`` tuple of a spec's wavelengths."""
    return "(" + ", ".join(f"_wave({w})" for w in wavelengths) + ",)"


def _absolute_lines(spec: ObjectSpec, kind: ObjectType, settings: SimSettings,
                    wavelengths: Any, stack: Any) -> list[str]:
    """The emitted construction for an absolutely-placed spec (spec R18).

    Mirrors :func:`_create_absolute`: the object is hand-constructed — no
    ``create_*_at`` covers absolute placement — and placed relative to the
    stack with margins from the stack's box center to the spec's center.
    """
    axis = int(spec.params.get("axis", 2))
    center = [float(v) for v in spec.params["center"]]
    lines = [f"# {spec.name}: absolute placement (no create_*_at for it)"]
    lines.append(f"{spec.name} = fdtdx.{spec.type}(")
    if kind.plane:
        size = spec.params.get("size") or (0.0, 0.0)
        real: list[str | None] = ["None", "None", "None"]
        grid = ["None", "None", "None"]
        u_axis, v_axis = _IN_PLANE[axis]
        grid[axis] = "1"
        real[u_axis] = f"{_num(size[0])} * 1e-6"
        real[v_axis] = f"{_num(size[1])} * 1e-6"
        lines.append(f"    partial_real_shape=({real[0]}, {real[1]}, {real[2]}),")
        lines.append(f"    partial_grid_shape=({grid[0]}, {grid[1]}, {grid[2]}),")
    else:
        lines.append("    partial_grid_shape=(1, 1, 1),")
    lines.append(f"    name={spec.name!r},")
    if kind.wave == "one":
        lines.append(f"    wave_character=_wave({wavelengths[0]}),")
        if settings.pulse_width_frac > 0:
            lines.append(f"    temporal_profile=_pulse(_wave({wavelengths[0]})),")
    elif kind.wave == "many":
        lines.append(f"    wave_characters={_waves_tuple(wavelengths)},")
    if kind.direction:
        # an absolute plane's direction is its normal axis' orientation (R18)
        char = "+" if spec.params.get("direction", "in") == "in" else "-"
        lines.append(f"    direction={char!r},")
    lines.extend(f"    {arg}," for arg in _extra_args(kind, spec.params))
    if kind.category == "detector" and settings.pulse_width_frac > 0 and "scaling_mode" in kind.extra:
        lines.append("    scaling_mode='pulse',")
    lines.append(")")
    lines.append(f"cons_{spec.name} = [{spec.name}.place_relative_to(")
    # a detector plane is anchored on its sampled point, as ``_create_absolute`` does
    own = f"detector_plane_own_positions({axis})" if kind.plane and kind.category == "detector" else "(0, 0, 0)"
    lines.append(f"    stack, axes=(0, 1, 2), own_positions={own}, other_positions=(0, 0, 0),")
    lines.append("    margins=(")
    lines.append(f"        ({_num(center[0])} - float(stack.center[0])) * 1e-6,")
    lines.append(f"        ({_num(center[1])} - float(stack.center[1])) * 1e-6,")
    lines.append(f"        ({_num(center[2])} - float(stack.z_center)) * 1e-6,")
    lines.append("    ),")
    lines.append(")]")
    return lines


_LAUNCH_PRELUDE = """\

# --- helpers the viewer's run uses -------------------------------------------
import fdtdx
import jax

from gdsfactory.fdtdx_stack import apply_stack, detector_plane_own_positions, init_stack_params


def _wave(wavelength_um):
    return fdtdx.WaveCharacter(wavelength=wavelength_um * 1e-6)


def _pulse(wave):
    return fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(
            frequency={pwf} * wave.get_frequency()
        ),
    )


def _pml_coeffs_in_config_dtype(objects, config):
    # fdtdx 0.6.3: gdsfactory enables jax x64, so the PML coefficients come
    # out float64 regardless of config.dtype; the CPML update then promotes
    # the float32 psi fields and eqxi.while_loop rejects the changed tree.
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ObjectContainer

    out = []
    for obj in objects.object_list:
        if type(obj).__name__ == "PerfectlyMatchedLayer":
            for name in ("pml_a_E", "pml_b_E", "inv_kappa_E",
                         "pml_a_H", "pml_b_H", "inv_kappa_H"):
                value = getattr(obj, name, None)
                if value is not None:
                    obj = obj.aset(name, value.astype(config.dtype))
        out.append(obj)
    return ObjectContainer(object_list=out, volume_idx=objects.volume_idx)
"""


def launch_code(stack: Any, state: Any) -> str:
    """The script tail that rebuilds the viewer's setup (spec R14 mode 3, D12).

    A pure string build — no ``create_*_at`` is called here (spec D11), no
    object is constructed — so it is safe to refresh on every setup edit. The
    emitted code mirrors :func:`build_objects` line for line: the simulation
    volume, every source/detector through ``stack.create_<object>_at``, the
    ``SimulationConfig`` and the PML boundaries, then the
    place/apply/run sequence :func:`run` uses — including the fdtdx 0.6.3
    PML dtype re-cast, without which ``run_fdtd`` rejects the loop body.

    Appending the result to a script that defines ``stack`` (the
    :class:`~gdsfactory.fdtdx_stack.StackDevice`) builds the same object list
    mode 2 returns. A non-zero ``boundary_offset`` is emitted as a
    ``rebox_stack`` call reproducing the viewer's re-box, and an ``averaging``
    other than the one the device was built with as an ``aset`` (spec R13).

    Args:
        stack: the :class:`~gdsfactory.fdtdx_stack.StackDevice` the code names
            (only its ``center``/``z_center`` are read, for absolute specs).
        state: a :class:`~gdsfactory.viewer.state.ViewerState` holding the
            specs and settings.

    Returns:
        The copyable Python source.
    """
    settings: SimSettings = state.settings
    lines: list[str] = [_LAUNCH_HEADER]
    # the helpers the tail calls at import time must be defined before use
    lines.append(_LAUNCH_PRELUDE.replace("{pwf}", _num(settings.pulse_width_frac)))

    # a viewer-edited boundary_offset re-boxed the device (spec R13): the
    # emitted code must reproduce that box before anything is placed on it
    if tuple(settings.boundary_offset) != (0.0,) * 6:
        lines.append("from gdsfactory.viewer.state import rebox_stack")
        lines.append(f"stack = rebox_stack(stack, {_seq(settings.boundary_offset)})")
        lines.append("")

    # an averaging picked in the viewer replaces the one the device was built
    # with (spec R13): the emitted script places the device with the same one
    if settings.averaging != getattr(state, "built_averaging", settings.averaging):
        lines.append(f"stack = stack.aset(\"averaging\", {_repr(settings.averaging)})")
        lines.append("")

    # the volume and the device — the head of build_objects' object list
    lines.append("volume, constraints = stack.get_simulation_volume()")
    lines.append("objects = [volume, stack]")

    # one call per spec, in spec order, with the same fallbacks _create applies
    for spec in state.specs:
        if spec.id in getattr(state, "adopted", ()):
            continue  # adopted foreign detectors are already placed (D18)
        kind = TYPES[spec.type]
        if kind.wave != "none":
            wavelengths = spec.params.get("wavelengths") or [settings.wavelength_um]
        else:
            wavelengths = None
        if spec.absolute:
            # R18: no create_*_at exists for absolute placement, so the tail
            # constructs the object directly, as _create_absolute does
            lines.extend(_absolute_lines(spec, kind, settings, wavelengths, stack))
            lines.append(f"objects.append({spec.name})")
            lines.append(f"constraints += cons_{spec.name}")
            continue
        args = [f"port={spec.port!r}"]
        if kind.plane:
            mult = spec.params.get("port_size_mult") or list(settings.port_size_mult)
            args.append(f"port_size_mult={_seq(mult)}")
        if kind.wave == "one":
            args.append(f"wave_character=_wave({wavelengths[0]})")
            if settings.pulse_width_frac > 0:
                args.append(f"temporal_profile=_pulse(_wave({wavelengths[0]}))")
        elif kind.wave == "many":
            args.append("wave_characters=" + _waves_tuple(wavelengths))
        if kind.direction:
            args.append(f"direction={spec.params.get('direction', 'in' if kind.category == 'source' else 'out')!r}")
        args.extend(_extra_args(kind, spec.params))
        if kind.category == "detector" and settings.pulse_width_frac > 0 and "scaling_mode" in kind.extra:
            args.append("scaling_mode='pulse'")
        if spec.params.get("level") is not None:
            args.append(f"level={spec.params['level']!r}")
        args.append(f"port_offset={_num(spec.params.get('port_offset', 0.0))}")
        args.append(f"name={spec.name!r}")
        lines.append(f"{spec.name}, cons_{spec.name} = stack.{kind.method}(")
        for arg in args:
            lines.append(f"    {arg},")
        lines.append(")")
        lines.append(f"objects.append({spec.name})")
        lines.append(f"constraints += cons_{spec.name}")

    # the config and the PML boundaries, as build_objects builds them
    lines.append("config = fdtdx.SimulationConfig(")
    lines.append(f"    time={_num(settings.time_s)},")
    lines.append("    grid=fdtdx.core.grid.UniformGrid(spacing="
                 f"{_num(settings.resolution_um)} * 1e-6),")
    lines.append(")")
    lines.append(f"pml = fdtdx.BoundaryConfig.from_uniform_bound(thickness={int(settings.pml_thickness)})")
    lines.append("boundaries, cons = fdtdx.boundary_objects_from_config(pml, volume)")
    lines.append("objects += list(boundaries.values())")
    lines.append("constraints += cons")

    # the run sequence, exactly as sim.run does it; the marker lets a reader
    # (or a test) cut the tail after building, before the FDTD run
    lines.append("# --- run: delete below to only build the setup -------------------------------")
    lines.append("key = jax.random.PRNGKey(0)")
    lines.append("objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)")
    lines.append("objects = _pml_coeffs_in_config_dtype(objects, config)")
    lines.append("arrays, objects, _ = apply_stack(arrays, objects, {**params, **init_stack_params(objects)}, key)")
    lines.append("_, arrays = fdtdx.run_fdtd(")
    lines.append("    arrays=arrays,")
    lines.append("    objects=objects,")
    lines.append("    config=config,")
    lines.append("    key=key,")
    lines.append("    show_progress=True,")
    lines.append(")")
    return "\n".join(lines) + "\n"


def run(
    stack: Any,
    state: Any,
    *,
    key: Any = None,
    progress: Any = None,
    phase: Any = None,
    cancel: Any = None,
    keep_partial: bool = False,
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
        cancel: an optional ``threading.Event``; when set, the run stops at the
            next step (spec D17). Stopping through the loop's own condition
            means ``run_fdtd`` returns normally, so the detectors' partial
            readings survive — see ``keep_partial``.
        keep_partial: when True, a cancelled run returns its partial
            :class:`RunResult` instead of raising :class:`RunStopped`. When
            False (the default), a cancelled run raises :class:`RunStopped`, so
            the caller discards it.

    Raises:
        RunStopped: a cancelled run whose partial result is not kept.
    """
    import fdtdx
    import jax

    from gdsfactory.fdtdx_stack import apply_stack, init_stack_params

    def _check_cancel() -> None:
        # a stop can land between phases too, not just between steps (spec D17)
        if cancel is not None and cancel.is_set():
            raise RunStopped("the run was stopped")

    def _phase(name: str) -> None:
        if phase is not None:
            phase(name)

    _check_cancel()
    _phase("placing")
    objects, config, constraints = build_objects(stack, state)
    _check_cancel()
    key = jax.random.PRNGKey(0) if key is None else key
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    objects = _pml_coeffs_in_config_dtype(objects, config)
    arrays, objects, _ = apply_stack(arrays, objects, {**params, **init_stack_params(objects)}, key)
    _check_cancel()

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

    stopping_condition = None
    if cancel is not None:
        stopping_condition = make_cancel_condition(cancel).setup(
            (jax.numpy.asarray(0, dtype=jax.numpy.int32), arrays), config, objects
        )

    _check_cancel()
    _phase("compiling")
    _, arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=progress is None,
        progress_callback=_progress if use_callback else None,
        stopping_condition=stopping_condition,
    )
    # a cancelled run exits the loop early, so the arrays hold the detectors'
    # partial readings; honour ``keep_partial`` (spec D17)
    if cancel is not None and cancel.is_set() and not keep_partial:
        raise RunStopped("the run was stopped")
    return RunResult(objects=objects, arrays=arrays, config=config)


# --------------------------------------------------------------------- glyphs

#: The in-plane axes of each normal axis (x→y,z; y→x,z; z→x,y). Shared with
#: ``glyph_from_grid`` and the absolute-glyph math so u is always the lower
#: in-plane stack axis and v the higher (spec D11, R18).
_IN_PLANE = ((1, 2), (0, 2), (0, 1))


def sampled_coordinate(axis: int, coord_um: float, box_um: float, resolution_um: float) -> float:
    """Where a detector plane's samples fall along its normal axis, in µm.

    fdtdx records x and y samples on a cell's lower edge and z samples at its
    centre, and places a one-cell plane in the cell whose sampled point is
    nearest the target (``detector_plane_own_positions``), the lower index
    winning a tie. This replays that choice on the grid the box resolves to, so
    a glyph drawn here sits where the field is recorded.

    Args:
        axis: the plane's normal axis (0 = x, 1 = y, 2 = z).
        coord_um: the target along ``axis``, in µm from the box center.
        box_um: the box extent along ``axis``, in µm.
        resolution_um: the grid spacing, in µm.

    Returns:
        The sampled coordinate along ``axis``, in µm from the box center.
    """
    cells = max(1, math.ceil(box_um / resolution_um - 1e-6))
    # in cells from the box's lower face; the nearest index, a tie going down
    x = coord_um / resolution_um + cells / 2
    if axis == 2:
        index = math.ceil(x - 1.0 - 1e-9)  # cell centres sit at k + 1/2
        index = min(max(index, 0), cells - 1)
        return (index + 0.5 - cells / 2) * resolution_um
    index = min(max(math.ceil(x - 0.5 - 1e-9), 0), cells - 1)  # lower edges sit at k
    return (index - cells / 2) * resolution_um


def _snap_to_samples(glyph: dict[str, Any], spec: ObjectSpec, stack: Any, settings: SimSettings) -> None:
    """Moves a detector plane glyph's normal coordinate onto its sample point, in place."""
    kind = TYPES[spec.type]
    shape = getattr(stack, "partial_real_shape", None)
    if not (kind.plane and kind.category == "detector") or shape is None or None in tuple(shape):
        return
    axis = glyph["axis"]
    glyph["center"][axis] = sampled_coordinate(
        axis, glyph["center"][axis], float(shape[axis]) * 1e6, float(settings.resolution_um)
    )


def _absolute_glyph(spec: ObjectSpec, origin: tuple[float, float, float]) -> dict[str, Any]:
    """The wire glyph of an absolutely-placed spec (spec R18).

    Reads ``axis``/``center``/``size`` (µm, stack coordinates — the same frame
    the box center the meshes are shifted by uses) instead of a port frame.
    A point type (no size) has zero half-extents. Shared with
    :func:`~gdsfactory.viewer.adopt.glyph_from_grid`, whose grid→µm conversion
    feeds these same numbers.
    """
    kind = TYPES[spec.type]
    params = spec.params
    axis = int(params.get("axis", 2))
    center = [float(v) for v in params.get("center") or (0.0, 0.0, 0.0)]
    size = params.get("size") or (0.0, 0.0)
    if kind.plane and len(size) == 2:
        u_axis, v_axis = _IN_PLANE[axis]
        u_half, v_half = 0.5 * float(size[0]), 0.5 * float(size[1])
    else:
        u_half = v_half = 0.0
    ox, oy, oz = origin
    return {
        "id": spec.id,
        "name": spec.name,
        "kind": spec.category,
        "type": spec.type,
        "port": spec.port,
        "center": [center[0] - ox, center[1] - oy, center[2] - oz],
        "axis": axis,
        "u_half": u_half,
        "v_half": v_half,
        "direction": params.get("direction"),
        "visible": bool(params.get("visible", True)),
    }


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
    if spec.absolute:
        glyph = _absolute_glyph(spec, origin)
        if spec.glyph is None:  # an adopted spec's center is already a placed sample point
            _snap_to_samples(glyph, spec, stack, settings)
        return glyph
    if spec.glyph is not None:
        return {**spec.glyph}
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
    glyph = {
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
    _snap_to_samples(glyph, spec, stack, settings)
    return glyph


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
