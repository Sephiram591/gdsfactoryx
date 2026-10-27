"""Tests for the differentiable fdtdx LayerStack device."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")

import gdsfactory as gf  # noqa: E402
from fdtdx.dispersion import susceptibility_from_coefficients  # noqa: E402

from gdsfactory.fdtdx_geometry import _edge_topology, split_t_junctions  # noqa: E402
from gdsfactory.fdtdx_stack import (  # noqa: E402
    _YEE_E_SHIFTS,
    StackDevice,
    StackDiagnostic,
    StackDiagnosticWarning,
    TopRoughness,
    _area,
    _clip,
    _composite,
    _composite_samples,
    _corners,
    _disk_area,
    _eval_bool,
    _eval_expr,
    _extension_rect,
    _face_intervals,
    _lateral_samples,
    _literals,
    _moments,
    _normal_squared,
    _normal_tensor,
    _Port,
    _round_offset,
    _seeds,
    _sizing,
    apply_stack,
    init_stack_params,
)
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer  # noqa: E402

RES = 0.02  # um
L = gf.gpdk.LAYER
WG, SLAB, ETCH, WAFER, METAL = L.WG, L.SLAB150, L.DEEP_ETCH, L.WAFER, L.M1

MATERIALS = {
    "air": fdtdx.Material(),
    "sio2": fdtdx.Material(permittivity=2.1),
    "si": fdtdx.Material(permittivity=12.0),
    "metal": fdtdx.Material(permittivity=1.0, electric_conductivity=1e5),
}


@pytest.fixture(autouse=True)
def pdk() -> Iterator[None]:
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    yield
    gf.snap.SNAP_ENABLED = True


def circuit(width=0.5, etch_x=1.4) -> gf.Component:
    """2 um straight on WG, slab rectangle, deep etch over the end, metal pad."""
    c = gf.Component()
    c.add_ref(gf.components.straight(length=2.0, width=width))
    c.add_polygon([(0.0, -0.4), (2.0, -0.4), (2.0, 0.4), (0.0, 0.4)], layer=SLAB)
    c.add_polygon([(etch_x, -0.5), (2.0, -0.5), (2.0, 0.5), (etch_x, 0.5)], layer=ETCH)
    c.add_polygon([(0.2, 0.6), (0.6, 0.6), (0.6, 0.7), (0.2, 0.7)], layer=METAL)
    return c


def stack(**core) -> LayerStack:
    return LayerStack(
        layers={
            "box": LayerLevel(layer=WAFER, zmin=-0.3, thickness=0.3, material="sio2", mesh_order=9),
            "core": LayerLevel(
                layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH),
                derived_layer=LogicalLayer(layer=WG),
                zmin=0.0,
                thickness=0.22,
                material="si",
                mesh_order=2,
                **({"sidewall_angle": 10.0, "width_to_z": 0.5} | core),
            ),
            "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.08, material="si", mesh_order=3),
            "clad": LayerLevel(layer=WAFER, zmin=0.0, thickness=0.5, material="sio2", mesh_order=10),
            "pad": LayerLevel(layer=METAL, zmin=0.4, thickness=0.1, material="metal", mesh_order=1),
        }
    )


def place(device: StackDevice, extra=()):
    config = fdtdx.SimulationConfig(
        time=10e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6),
        backend=jax.default_backend(),
        dtype=jnp.float32,
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(3.0e-6, 2.0e-6, 1.0e-6))
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device, *extra],
        config=config,
        constraints=[device.place_at_center(volume), *(o.place_at_center(volume) for o in extra)],
        key=jax.random.PRNGKey(0),
    )
    params = {**params, **init_stack_params(objects)}
    return objects, arrays, params


def box(ls: LayerStack, z_range=(-0.3, 0.6), padding=0.3) -> dict:
    """from_layer_stack box arguments: all levels, then offsets to z_range and an xy padding (um)."""
    spans = [sorted((lv.zmin, lv.zmin + lv.thickness)) for lv in ls.layers.values()]
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    return dict(
        included_layers=list(ls.layers),
        boundary_offset=(padding, -padding, padding, -padding, z_range[1] - hi, z_range[0] - lo),
    )


def build(c=None, ls=None, materials=None, z_range=(-0.3, 0.6), padding=0.3, **kwargs) -> StackDevice:
    ls = ls or stack()
    if "included_layers" not in kwargs:
        kwargs.update(box(ls, z_range, padding))
    return StackDevice.from_layer_stack(
        c or circuit(),
        ls,
        materials or MATERIALS,
        background="air",
        name="stack",
        **kwargs,
    )


def cell(device, x, y, z):
    """Voxel index of stack point (x, y, z) in um."""
    nx, ny, nz = device.grid_shape
    cx, cy = device.center
    return (
        int(math.floor((x - cx) / RES + nx / 2)),
        int(math.floor((y - cy) / RES + ny / 2)),
        int(math.floor((z - device.z_center) / RES + nz / 2)),
    )


def test_params_and_shape() -> None:
    objects, _, params = place(build())
    d = objects["stack"]
    assert d.grid_shape == (130, 90, 45)  # bbox 2 x 1.2 um + 0.3 padding, z 0.9 um
    assert set(params["stack"]) == {
        "vertices/WG", "vertices/SLAB150", "vertices/DEEP_ETCH", "vertices/M1",
        *(
            f"{k}/{lv}"
            for k in ("zmin", "thickness", "sidewall_angle")
            for lv in ("box", "core", "slab", "clad", "pad")
        ),
    }


def test_materials_priority_and_derived_layers() -> None:
    objects, _, params = place(build())
    d = objects["stack"]
    names, w = d.material_weights(params["stack"])
    w = {n: w[i] for i, n in enumerate(names)}
    np.testing.assert_allclose(sum(w.values()), 1.0, atol=1e-6)

    def material_at(x, y, z):
        i = cell(d, x, y, z)
        return max(w, key=lambda n: float(w[n][i]))

    assert material_at(1.0, 0.0, -0.15) == "sio2"  # box
    assert material_at(0.7, 0.0, 0.15) == "si"  # core beats clad (mesh_order 2 < 10)
    assert material_at(0.7, 0.3, 0.04) == "si"  # slab outside the core
    assert material_at(0.7, 0.3, 0.15) == "sio2"  # cladding above the slab
    assert material_at(1.7, 0.0, 0.15) == "sio2"  # WG - DEEP_ETCH removes the core
    assert material_at(1.7, 0.0, 0.04) == "si"  # ... but not the slab
    assert material_at(0.4, 0.65, 0.45) == "metal"
    assert material_at(1.0, 0.0, 0.55) == "air"  # above the cladding: background


def test_core_volume_with_sidewall() -> None:
    """width_to_z=0.5 with a linear taper keeps the drawn cross-section area."""
    ls = LayerStack(layers={"core": stack().layers["core"]})
    objects, _, params = place(build(ls=ls))
    names, w = objects["stack"].material_weights(params["stack"])
    si = float(w[names.index("si")].sum()) * RES**3
    # 1.4 um of core survives the etch, 0.5 um wide, 0.22 um thick
    assert si == pytest.approx(1.4 * 0.5 * 0.22, rel=0.02)


def test_negative_thickness_and_z_to_bias() -> None:
    """gpdk-style undercut: extends below zmin and shrinks with depth."""
    ls = LayerStack(
        layers={
            "box": LayerLevel(layer=WAFER, zmin=-0.3, thickness=0.3, material="sio2", mesh_order=9),
            "undercut": LayerLevel(
                layer=SLAB, zmin=0.0, thickness=-0.2, material="air", mesh_order=1,
                z_to_bias=([0.0, 1.0], [0.0, -0.2]),
            ),
        }
    )
    objects, _, params = place(build(ls=ls))
    d = objects["stack"]
    names, w = d.material_weights(params["stack"])
    air = w[names.index("air")]
    i, j, k = cell(d, 1.0, 0.0, -0.1)
    assert float(air[i, j, k]) == pytest.approx(1.0)
    # shrinks by 0.2 um per side over the full depth: half way down the half width is 0.3
    edge_mid = cell(d, 1.0, 0.35, -0.1)
    assert float(air[edge_mid]) < 0.5
    z = np.arange(10) * RES + RES / 2  # 10 cells from 0 down to -0.2
    expected = sum((2.0 - 2 * 0.2 * zz / 0.2) * (0.8 - 2 * 0.2 * zz / 0.2) for zz in z) * RES
    below = air[:, :, : cell(d, 0, 0, 0.0)[2]].sum() * RES**3
    assert float(below) == pytest.approx(expected, rel=0.03)


def _weights(objects, params, noise=True):
    d = objects["stack"]
    names, w = d.material_weights(params["stack"], noise_key=jax.random.PRNGKey(2), apply_roughness=noise)
    return d, {n: w[i] for i, n in enumerate(names)}


def test_top_roughness_filled_by_layers_deposited_on_it() -> None:
    """Box top is rough; clad, core and slab sit on it and follow it down."""
    objects, _, params = place(build(roughness=[TopRoughness(level="box", rms=0.01, correlation_length=0.08)]))
    d, w = _weights(objects, params)
    _, w0 = _weights(objects, params, noise=False)
    k_top = cell(d, 0, 0, 0.45)[2]
    assert float(w["air"][:, :, :k_top].max()) < 1e-6  # no voids
    # where only box and cladding meet (x < 0, outside the silicon) both are oxide
    i_out = cell(d, -0.05, 0, 0)[0]
    assert float(jnp.abs(w["sio2"] - w0["sio2"])[:i_out, :, :k_top].max()) < 1e-6
    assert float(jnp.abs(w["si"] - w0["si"]).max()) > 0.1  # the core bottom moved
    k0 = cell(d, 0, 0, 0.0)[2]
    assert float(w["si"][:, :, k0 - 2 : k0 + 2].sum()) == pytest.approx(
        float(w0["si"][:, :, k0 - 2 : k0 + 2].sum()), rel=0.05
    )  # zero-mean roughness: same silicon on average


def test_top_roughness_without_fill_leaves_dips_to_background() -> None:
    rough = [TopRoughness(level="box", rms=0.01, correlation_length=0.08, fill_above=False)]
    objects, _, params = place(build(roughness=rough))
    d, w = _weights(objects, params)
    k0 = cell(d, 0, 0, 0.0)[2]
    assert float(w["air"][:, :, k0 - 2 : k0].max()) > 0.1  # dips below z=0: nothing fills them


def test_core_top_roughness_is_filled_by_cladding() -> None:
    objects, _, params = place(build(roughness=[TopRoughness(level="core", rms=0.01, correlation_length=0.05)]))
    d, w = _weights(objects, params)
    _, w0 = _weights(objects, params, noise=False)
    k_top = cell(d, 0, 0, 0.45)[2]
    assert float(w["air"][:, :, :k_top].max()) < 1e-6
    assert float(jnp.abs(w["si"] - w0["si"]).max()) > 0.1
    # only the top surface moved: the bottom of the core is untouched
    k0 = cell(d, 0, 0, 0.0)[2]
    np.testing.assert_allclose(w["si"][:, :, : k0 + 3], w0["si"][:, :, : k0 + 3], atol=1e-6)


def test_unknown_roughness_level() -> None:
    with pytest.raises(KeyError, match="roughness level"):
        build(roughness=[TopRoughness(level="nope", rms=0.01)])


def test_gradients_match_finite_differences() -> None:
    objects, _, params = place(build())
    d = objects["stack"]
    si_index = d.material_weights(params["stack"])[0].index("si")

    def si_volume(width, thickness):
        p = {**params["stack"], **d.pack(circuit(width=width)), "thickness/core": thickness}
        return d.material_weights(p)[1][si_index].sum()

    gw, gt = jax.grad(si_volume, argnums=(0, 1))(0.5, 0.22)
    h = 1e-3
    fdw = (si_volume(0.5 + h, 0.22) - si_volume(0.5 - h, 0.22)) / (2 * h)
    fdt = (si_volume(0.5, 0.22 + h) - si_volume(0.5, 0.22 - h)) / (2 * h)
    assert float(gw) == pytest.approx(float(fdw), rel=5e-3)
    assert float(gt) == pytest.approx(float(fdt), rel=5e-3)
    assert float(gw) > 0 and float(gt) > 0


def test_zmin_moves_the_level() -> None:
    objects, _, params = place(build())
    d = objects["stack"]
    metal = d.material_weights(params["stack"])[0].index("metal")
    nz = d.grid_shape[2]
    zc = jnp.arange(nz) + 0.5  # voxel-center heights (cells)

    def moment(zmin):
        """Metal volume and its first z moment (cells)."""
        w = d.material_weights({**params["stack"], "zmin/pad": zmin})[1][metal]
        return w.sum(), (w * zc).sum()

    volume, m0 = moment(0.4)
    # a rigid shift: the volume stays, the moment moves by volume * dz
    v1, m1 = moment(0.42)
    assert float(v1) == pytest.approx(float(volume), rel=1e-4)
    assert float(m1 - m0) == pytest.approx(float(volume) * 0.02 / RES, rel=1e-3)
    g = jax.grad(lambda z: moment(z)[1])(0.4)
    h = 1e-3
    fd = (moment(0.4 + h)[1] - moment(0.4 - h)[1]) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=5e-3)
    assert float(g) == pytest.approx(float(volume) / RES, rel=1e-3)


@pytest.mark.parametrize(("level", "key", "value"), [("core", "thickness", 0.30), ("pad", "zmin", 0.25)])
def test_large_moves_match_the_moved_stack(level, key, value) -> None:
    """Moving a level by 80-150 nm through its parameters = building the moved stack."""
    moved = stack()
    moved.layers[level] = moved.layers[level].model_copy(update={key: value})
    objects, _, params = place(build())
    reference, _, ref_params = place(build(ls=moved))
    d, r = objects["stack"], reference["stack"]
    assert d.grid_shape == r.grid_shape and d.z_center == r.z_center
    names, w = d.material_weights({**params["stack"], f"{key}/{level}": value})
    ref_names, ref = r.material_weights(ref_params["stack"])
    assert names == ref_names
    np.testing.assert_allclose(np.asarray(w), np.asarray(ref), atol=2e-5)


def test_levels_that_miss_the_placed_box_are_dropped() -> None:
    # box top at ~0.35 um: the pad (0.4 to 0.5 um) is within the selection slack
    # but misses the grid, so it is dropped with its material, layer and pole slot
    objects, arrays, params = place(build(materials=DISP_MATERIALS, z_range=(-0.3, 0.35)))
    d = objects["stack"]
    assert [lv.name for lv in d.levels] == ["box", "core", "slab", "clad"]
    assert "metal" not in d.materials
    assert "zmin/pad" not in params["stack"] and "vertices/M1" not in params["stack"]
    assert arrays.electric_conductivity is None
    assert arrays.dispersive_c1.shape[0] == 1  # only Si's Lorentz pole, no Drude slot for the pad

    # with level_margin the pad is kept and can be moved into the box
    objects, arrays, params = place(build(z_range=(-0.3, 0.35), level_margin=0.1))
    d = objects["stack"]
    assert "pad" in [lv.name for lv in d.levels]
    assert arrays.electric_conductivity is not None
    names, w = d.material_weights(params["stack"])
    assert float(w[names.index("metal")].sum()) == 0.0
    names, w = d.material_weights({**params["stack"], "zmin/pad": 0.3})
    assert float(w[names.index("metal")].sum()) > 0.0

    with pytest.raises(ValueError, match="level_margin"):
        build(level_margin=-0.1)


@pytest.mark.parametrize("averaging", ["inverse", "subpixel"])
def test_apply_stack(averaging: str) -> None:
    objects, arrays, params = place(build(averaging=averaging, roughness=[TopRoughness(level="core", rms=0.005, correlation_length=0.05)]))
    d = objects["stack"]
    assert arrays.inv_permittivities.shape[0] == (3 if averaging == "subpixel" else 1)
    assert arrays.electric_conductivity is not None  # metal level

    @jax.jit
    def run(vertices, noise_key):
        p = {**params, "stack": {**params["stack"], **vertices}}
        arrays2, _, _ = apply_stack(arrays, objects, p, jax.random.PRNGKey(0), noise_key=noise_key)
        return arrays2

    def eps_sum(width):
        a = run(d.pack(circuit(width=width)), jax.random.PRNGKey(1))
        return jnp.sum(1.0 / a.inv_permittivities)

    a = run(d.pack(circuit()), jax.random.PRNGKey(1))
    inv = a.inv_permittivities[:, *d.grid_slice]
    assert float(inv.min()) >= 1 / 12.0 - 1e-6
    assert float(inv.max()) <= 1.0 + 1e-6
    assert float(a.electric_conductivity[:, *d.grid_slice].max()) > 0
    assert float(jax.grad(eps_sum)(0.5)) > 0


def test_gpdk_layer_stack_smoke() -> None:
    ls = gf.gpdk.LAYER_STACK
    mats = {lv.material: fdtdx.Material(permittivity=4.0) for lv in ls.layers.values() if lv.material}
    mats["air"] = fdtdx.Material()
    c = gf.components.straight(length=2.0)
    device = StackDevice.from_layer_stack(
        c, ls, mats, background="air", name="stack",
        included_layers=["core"], boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.28, -0.4),
    )
    objects, _, params = place(device)
    names, w = objects["stack"].material_weights(params["stack"])
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-6)
    assert "vertices/WG" in params["stack"]


# ---------------------------------------------------------------- dispersion

C0 = 299792458.0
W0_SI = 2 * math.pi * C0 / 0.3e-6
OMEGA = 2 * math.pi * C0 / 1.55e-6
SI_DISP = fdtdx.Material(
    permittivity=7.0,
    dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=W0_SI, damping=0.0, delta_epsilon=4.9),)
    ),
)
AU_DISP = fdtdx.Material(
    permittivity=1.0,
    dispersion=fdtdx.DispersionModel(poles=(fdtdx.DrudePole(plasma_frequency=1.37e16, damping=1.0e14),)),
)
DISP_MATERIALS = {**MATERIALS, "si": SI_DISP, "metal": AU_DISP}


def _box(arrays, device, name):
    a = getattr(arrays, name)
    return a[:, *device.grid_slice] if a.ndim == 4 else a[:, :, *device.grid_slice]


def _apply(objects, arrays, params, **kwargs):
    arrays2, _, _ = apply_stack(arrays, objects, params, jax.random.PRNGKey(0), **kwargs)
    return arrays2


@pytest.mark.parametrize("averaging", ["inverse", "subpixel"])
def test_dispersive_stack_mixes_susceptibilities(averaging: str) -> None:
    objects, arrays, params = place(build(materials=DISP_MATERIALS, averaging=averaging))
    d = objects["stack"]
    assert arrays.dispersive_c1.shape[0] == 2  # one Lorentz (si) + one Drude (metal) slot
    arrays2 = _apply(objects, arrays, params)
    if averaging == "subpixel":  # pole strengths use the mean of the three Yee positions
        shifts = ((0.0, -0.5, -0.5), (-0.5, 0.0, -0.5), (-0.5, -0.5, 0.0))
        names = d.material_weights(params["stack"])[0]
        w = sum(d.material_weights(params["stack"], shift=s)[1] for s in shifts) / 3
    else:
        names, w = d.material_weights(params["stack"])
    w = {n: np.asarray(w[i]) for i, n in enumerate(names)}
    assert float(((w["si"] > 0.05) & (w["si"] < 0.95)).sum()) > 100
    assert float(w["metal"].max()) == pytest.approx(1.0)  # the pad

    dt = d._config.time_step_duration
    c1, c2, c3 = (_box(arrays2, d, f"dispersive_c{i}")[:, 0] for i in (1, 2, 3))
    # slot constants everywhere in the box
    for slot, mat in enumerate((SI_DISP, AU_DISP)):
        a1, a2, a3 = fdtdx.compute_pole_coefficients(mat.dispersion.poles, dt)
        np.testing.assert_allclose(c1[slot], a1[0], rtol=1e-6)
        np.testing.assert_allclose(c2[slot], a2[0], rtol=1e-6)
        expected = w["si"] if slot == 0 else w["metal"]
        np.testing.assert_allclose(c3[slot], expected * a3[0], rtol=1e-5, atol=1e-7 * a3[0])

    # reconstructed susceptibility = sum_m w_m chi_m (what sources see)
    chi = np.asarray(susceptibility_from_coefficients(c1, c2, c3, OMEGA, dt))
    expected = w["si"] * complex(SI_DISP.dispersion.susceptibility(OMEGA)) + w["metal"] * complex(
        AU_DISP.dispersion.susceptibility(OMEGA)
    )
    np.testing.assert_allclose(chi.real, expected.real, rtol=2e-3, atol=1e-3)
    np.testing.assert_allclose(chi.imag, expected.imag, rtol=2e-3, atol=1e-3)

    inv = _box(arrays2, d, "inv_permittivities")
    assert bool(jnp.all(jnp.isfinite(inv)))
    assert float(inv.min()) >= 1 / 7.0 - 1e-6 and float(inv.max()) <= 1.0 + 1e-6


def test_dispersive_stack_gradient_and_roughness() -> None:
    rough = [TopRoughness(level="core", rms=0.01, correlation_length=0.05)]
    objects, arrays, params = place(build(materials=DISP_MATERIALS, roughness=rough))
    d = objects["stack"]

    def c3_si(width, noise_key=None):
        p = {**params, "stack": {**params["stack"], **d.pack(circuit(width=width))}}
        return jnp.sum(_box(_apply(objects, arrays, p, noise_key=noise_key), d, "dispersive_c3")[0])

    # width 0.5 puts the corners exactly on Yee sample points, where the derivative is
    # a kink (any subgradient is valid); 0.5137 is a generic point with a unique one
    w0 = 0.5137
    g = jax.grad(c3_si)(w0)
    h = 1e-3
    fd = (c3_si(w0 + h) - c3_si(w0 - h)) / (2 * h)
    assert float(g) > 0
    assert float(g) == pytest.approx(float(fd), rel=5e-3)
    a = _box(_apply(objects, arrays, params, noise_key=jax.random.PRNGKey(1)), d, "dispersive_c3")
    b = _box(_apply(objects, arrays, params, noise_key=jax.random.PRNGKey(2)), d, "dispersive_c3")
    assert float(jnp.abs(a - b).max()) > 0  # roughness moves the silicon's poles too


def test_non_dispersive_stack_clears_its_box() -> None:
    """Another object makes the simulation dispersive; the stack box has no poles."""
    gold = fdtdx.UniformMaterialObject(partial_real_shape=(0.6e-6, 0.6e-6, 0.6e-6), material=AU_DISP)
    objects, arrays, params = place(build(), extra=[gold])
    d = objects["stack"]
    assert arrays.dispersive_c1 is not None
    assert float(jnp.abs(_box(arrays, d, "dispersive_c3")).max()) > 0  # gold painted at init
    arrays2 = _apply(objects, arrays, params)
    for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3"):
        assert float(jnp.abs(_box(arrays2, d, name)).max()) == 0.0


def test_eps_inf_tie_subpixel_is_finite() -> None:
    """Air and a Drude metal both have eps_inf = 1 (the normal does not matter there)."""
    objects, arrays, params = place(build(materials=DISP_MATERIALS, averaging="subpixel"))
    inv = _box(_apply(objects, arrays, params), objects["stack"], "inv_permittivities")
    assert bool(jnp.all(jnp.isfinite(inv)))


def _film_stack() -> LayerStack:
    """Si / 10 nm SiO2 film / Si, the film inside the voxel z in [0, 20] nm."""
    return LayerStack(
        layers={
            "below": LayerLevel(layer=WAFER, zmin=-0.3, thickness=0.305, material="si", mesh_order=2),
            "film": LayerLevel(layer=WAFER, zmin=0.005, thickness=0.01, material="sio2", mesh_order=1),
            "above": LayerLevel(layer=WAFER, zmin=0.015, thickness=0.285, material="si", mesh_order=2),
        }
    )


def test_subpixel_film_thinner_than_a_voxel() -> None:
    """The film's normal must not vanish: z gets <1/eps>, x and y get 1/<eps>."""
    device = build(ls=_film_stack(), averaging="subpixel", z_range=(-0.3, 0.3), yee_staggered=False)
    objects, arrays, params = place(device)
    d = objects["stack"]
    inv = _box(_apply(objects, arrays, params), d, "inv_permittivities")
    i, j, k = d.grid_shape[0] // 2, d.grid_shape[1] // 2, cell(d, 0, 0, 0.01)[2]
    assert float(inv[2, i, j, k]) == pytest.approx(0.5 / 2.1 + 0.5 / 12.0, rel=1e-4)
    assert float(inv[0, i, j, k]) == pytest.approx(1 / (0.5 * 2.1 + 0.5 * 12.0), rel=1e-4)
    assert float(inv[1, i, j, k]) == pytest.approx(1 / (0.5 * 2.1 + 0.5 * 12.0), rel=1e-4)


def test_yee_staggered_sampling() -> None:
    """Ez sits at the voxel's z center, Ex and Ey half a cell lower (fdtdx's Yee grid).

    The 10 nm film fills [5, 15] nm. Ez's dual cell is the voxel [0, 20] nm (film
    fraction 1/2); Ex's and Ey's are [-10, 10] nm (film fraction 1/4).
    """
    objects, arrays, params = place(build(ls=_film_stack(), averaging="subpixel", z_range=(-0.3, 0.3)))
    d = objects["stack"]
    inv = _box(_apply(objects, arrays, params), d, "inv_permittivities")
    i, j, k = d.grid_shape[0] // 2, d.grid_shape[1] // 2, cell(d, 0, 0, 0.01)[2]
    assert float(inv[2, i, j, k]) == pytest.approx(0.5 / 2.1 + 0.5 / 12.0, rel=1e-4)
    for c in (0, 1):  # the film is a z-interface: tangential components get 1/<eps>
        assert float(inv[c, i, j, k]) == pytest.approx(1 / (0.25 * 2.1 + 0.75 * 12.0), rel=1e-4)


def test_subpixel_in_a_2d_simulation() -> None:
    """One axis of a single cell: no crash, and that axis is a parallel (arithmetic) average."""
    # x [-0.3, 2.3], y one cell around 0 (bbox y [-0.5, 0.7]), z [-0.3, 0.6] (levels [-0.3, 0.5])
    device = build(
        averaging="subpixel",
        included_layers=list(stack().layers),
        boundary_offset=(0.3, -0.3, RES / 2 - 0.7, -RES / 2 + 0.5, 0.1, 0.0),
    )
    config = fdtdx.SimulationConfig(
        time=10e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6),
        backend=jax.default_backend(),
        dtype=jnp.float32,
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(3.0e-6, RES * 1e-6, 1.0e-6))
    objects, arrays, params, _, _ = fdtdx.place_objects(
        object_list=[volume, device],
        config=config,
        constraints=[device.place_at_center(volume)],
        key=jax.random.PRNGKey(0),
    )
    params = {**params, **init_stack_params(objects)}
    d = objects["stack"]
    assert d.grid_shape[1] == 1
    inv = _box(_apply(objects, arrays, params), d, "inv_permittivities")
    assert bool(jnp.all(jnp.isfinite(inv)))
    names, w = d.material_weights(params["stack"], shift=(-0.5, 0.0, -0.5))  # Ey's Yee position
    eps_of = {"air": 1.0, "sio2": 2.1, "si": 12.0, "metal": 1.0}
    eps = jnp.asarray([eps_of.get(n, 1.0) for n in names])  # the anisotropy sentinel has w = 0
    np.testing.assert_allclose(inv[1], 1 / jnp.einsum("m,mxyz->xyz", eps, w), rtol=1e-5)


def test_sampling_shift_moves_every_level() -> None:
    """A z shift moves all levels, also fill levels processed after polygon levels.

    The cladding (fill, last in mesh order) ends at z = 0.5 um, a voxel boundary;
    half a voxel lower, that voxel is half cladding.
    """
    objects, _, params = place(build())
    d = objects["stack"]
    names, w0 = d.material_weights(params["stack"])
    _, w1 = d.material_weights(params["stack"], shift=(0.0, 0.0, -0.5))
    sio2 = names.index("sio2")
    k = cell(d, 0, 0, 0.51)[2]  # voxel [0.5, 0.52] um
    assert float(w0[sio2, 2, 2, k]) == pytest.approx(0.0, abs=1e-6)
    assert float(w1[sio2, 2, 2, k]) == pytest.approx(0.5, abs=1e-6)
    k = cell(d, 0, 0, -0.29)[2]  # voxel [-0.3, -0.28]: the box bottom is the device bottom
    assert float(w1[sio2, 2, 2, k]) == pytest.approx(0.5, abs=1e-6)


def test_xy_subsamples() -> None:
    """Sub-sampling keeps straight edges exact and fractions normalized."""
    ls = LayerStack(layers={"core": stack().layers["core"]})
    volumes = []
    for n in (1, 2):
        objects, _, params = place(build(ls=ls, xy_subsamples=n))
        names, w = objects["stack"].material_weights(params["stack"])
        np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-6)
        volumes.append(float(w[names.index("si")].sum()) * RES**3)
    assert volumes[1] == pytest.approx(volumes[0], rel=5e-3)
    assert volumes[1] == pytest.approx(1.4 * 0.5 * 0.22, rel=0.02)


# ---------------------------------------------------------------- seams and junctions


def _weights_of(component, ls, materials, size, center, z_center=0.14, **kwargs):
    """Material fractions of a StackDevice that fills a volume of `size` (um) around (center, z_center).

    ``kwargs`` are other StackDevice fields.
    """
    objects, _, params = _placed(component, ls, materials, size, center, z_center, **kwargs)
    d = objects["s"]
    names, w = d.material_weights(params["s"])
    return d, {n: np.asarray(w[i]) for i, n in enumerate(names)}


def _placed(component, ls, materials, size, center, z_center=0.14, **kwargs):
    """A StackDevice "s" that fills a volume of `size` (um) around (center, z_center): objects, arrays, params.

    ``kwargs`` are other StackDevice fields.
    """
    shape = tuple(1e-6 * s for s in size)
    pts = np.concatenate([np.asarray(p) for polys in component.get_polygons_points().values() for p in polys])
    (bx0, by0), (bx1, by1) = pts.min(axis=0), pts.max(axis=0)
    spans = [sorted((lv.zmin, lv.zmin + lv.thickness)) for lv in ls.layers.values()]
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    (cx, cy), (sx, sy, sz) = center, size
    device = StackDevice.from_layer_stack(
        component, ls, materials, background="air", name="s", included_layers=list(ls.layers),
        boundary_offset=(cx + sx / 2 - bx1, cx - sx / 2 - bx0, cy + sy / 2 - by1, cy - sy / 2 - by0,
                         z_center + sz / 2 - hi, z_center - sz / 2 - lo),
        **kwargs,
    )
    config = fdtdx.SimulationConfig(
        time=1e-15, grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6), dtype=jnp.float32
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=shape)
    objects, arrays, params, _, _ = fdtdx.place_objects(
        object_list=[volume, device], config=config,
        constraints=[device.place_at_center(volume)], key=jax.random.PRNGKey(0),
    )
    return objects, arrays, {**params, **init_stack_params(objects)}


def _polygons(*polys) -> gf.Component:
    c = gf.Component()
    for poly in polys:
        c.add_polygon(poly, layer=WG)
    return c


JUNCTIONS = {
    # a taper starting 20 nm narrower than the straight it continues
    "narrower taper": (
        [[(0, -0.25), (2, -0.25), (2, 0.25), (0, 0.25)], [(2, -0.24), (4, -0.1), (4, 0.1), (2, 0.24)]],
        [(0, -0.25), (2, -0.25), (2, -0.24), (4, -0.1), (4, 0.1), (2, 0.24), (2, 0.25), (0, 0.25)],
    ),
    # a taper on the side of an MMI body
    "taper on MMI": (
        [[(0, -0.75), (2, -0.75), (2, 0.75), (0, 0.75)], [(2, -0.25), (4, -0.1), (4, 0.1), (2, 0.25)]],
        [(0, -0.75), (2, -0.75), (2, -0.25), (4, -0.1), (4, 0.1), (2, 0.25), (2, 0.75), (0, 0.75)],
    ),
    # the same with a clockwise taper (abutting polygons of opposite orientations)
    "clockwise taper on MMI": (
        [[(0, -0.75), (2, -0.75), (2, 0.75), (0, 0.75)], [(2, 0.25), (4, 0.1), (4, -0.1), (2, -0.25)]],
        [(0, -0.75), (2, -0.75), (2, -0.25), (4, -0.1), (4, 0.1), (2, 0.25), (2, 0.75), (0, 0.75)],
    ),
    # a crossing: arms abutting the middle of a bar from both sides
    "crossing": (
        [
            [(0, -0.25), (4, -0.25), (4, 0.25), (0, 0.25)],
            [(1.75, 0.25), (2.25, 0.25), (2.25, 1.25), (1.75, 1.25)],
            [(1.75, -1.25), (2.25, -1.25), (2.25, -0.25), (1.75, -0.25)],
        ],
        [(0, -0.25), (1.75, -0.25), (1.75, -1.25), (2.25, -1.25), (2.25, -0.25), (4, -0.25),
         (4, 0.25), (2.25, 0.25), (2.25, 1.25), (1.75, 1.25), (1.75, 0.25), (0, 0.25)],
    ),
}


@pytest.mark.parametrize("name", list(JUNCTIONS))
def test_partial_junctions_do_not_open_under_a_sidewall_angle(name: str) -> None:
    """Abutting polygons that share only part of an edge equal the merged polygon."""
    pieces, merged = JUNCTIONS[name]
    ls = LayerStack(layers={"core": LayerLevel(
        layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1, sidewall_angle=30.0, width_to_z=0.0,
    )})
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"]}
    size = (4.4, 2.8, 0.3)
    _, split = _weights_of(_polygons(*pieces), ls, mats, size, center=(2.0, 0.0))
    _, whole = _weights_of(_polygons(merged), ls, mats, size, center=(2.0, 0.0))
    np.testing.assert_allclose(split["si"], whole["si"], atol=1e-4)
    assert float(split["si"].max()) == pytest.approx(1.0)


def _seam_stack(material_b: str) -> LayerStack:
    return LayerStack(layers={
        "a": LayerLevel(layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH),
                        derived_layer=LogicalLayer(layer=WG), zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        "b": LayerLevel(layer=LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH),
                        derived_layer=LogicalLayer(layer=WG), zmin=0.0, thickness=0.22, material=material_b, mesh_order=2),
    })


@pytest.mark.parametrize("material_b", ["si", "sio2"])
def test_levels_meeting_at_a_seam_do_not_leak(material_b: str) -> None:
    """WG - ETCH next to WG & ETCH: the seam voxel is covered, split by area."""
    wg = [(0, -0.5), (4, -0.5), (4, 0.5), (0, 0.5)]
    etch = [(2.01, -1.0), (5, -1.0), (5, 1.0), (2.01, 1.0)]  # seam through voxel centers
    c = gf.Component()
    c.add_polygon(wg, layer=WG)
    c.add_polygon(etch, layer=ETCH)
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}
    d, w = _weights_of(c, _seam_stack(material_b), mats, (4.4, 1.4, 0.3), center=(2.0, 0.0))
    i, j, k = cell(d, 2.01, 0.0, 0.11)
    covered = w["si"][i, j, k] + w.get("sio2", np.zeros_like(w["si"]))[i, j, k]
    assert float(w["air"][i, j, k]) == pytest.approx(0.0, abs=1e-6)  # no background leak
    assert float(covered) == pytest.approx(1.0, abs=1e-6)
    if material_b != "si":
        assert float(w["si"][i, j, k]) == pytest.approx(0.5, abs=1e-6)


def test_slanted_edges_get_exact_areas() -> None:
    """Fractions of voxels cut by one straight slanted edge equal the exact areas."""
    shapely = pytest.importorskip("shapely.geometry")
    angle = math.radians(17.0)
    u, v = np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])
    origin = np.array([0.31, -0.22])
    strip = [tuple(origin + a * u + b * v) for a, b in ((0, 0), (3.5, 0), (3.5, 0.7), (0, 0.7))]
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1)})
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"]}
    d, w = _weights_of(_polygons(strip), ls, mats, (4.4, 2.4, 0.3), center=(2.0, 0.6))
    poly = shapely.Polygon(strip)
    nx, ny, _ = d.grid_shape
    k = cell(d, 0, 0, 0.11)[2]
    x0, y0 = 2.0 - nx * RES / 2, 0.6 - ny * RES / 2
    errors = []
    for i in range(nx):
        for j in range(ny):
            box = shapely.box(x0 + i * RES, y0 + j * RES, x0 + (i + 1) * RES, y0 + (j + 1) * RES)
            exact = poly.intersection(box).area / RES**2
            near_corner = min(box.centroid.distance(shapely.Point(p)) for p in strip) < 2 * RES
            if 0 < exact < 1 and not near_corner:
                errors.append(abs(float(w["si"][i, j, k]) - exact))
    assert len(errors) > 400
    assert max(errors) < 1e-4


# a strip with its x edges off the voxel boundaries, on a level whose bottom (5 nm)
# and top (214 nm) cut the voxels [0, 20] and [200, 220] nm at 1/4 and 7/10
STRIP = [(0.313, -0.6), (1.287, -0.6), (1.287, 0.6), (0.313, 0.6)]
TAPER_Z = (0.005, 0.209)  # zmin, thickness (um)
TAPERS = {
    "sidewall": {"sidewall_angle": 40.0, "width_to_z": 0.5},
    # z_to_bias kink at 207.7 nm: between the middle of the top part (207 nm) and the voxel center (210 nm)
    "z_to_bias kink": {"sidewall_angle": 15.0, "z_to_bias": ((0.0, 0.97, 1.0), (0.0, 0.0, -0.03))},
}


def _tapered_strip(taper: dict, size=(1.6, 1.4, 0.3), z_center=0.11):
    ls = LayerStack(layers={"core": LayerLevel(
        layer=WG, zmin=TAPER_Z[0], thickness=TAPER_Z[1], material="si", mesh_order=1, **taper,
    )})
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"]}
    return _weights_of(_polygons(STRIP), ls, mats, size, center=(0.8, 0.0), z_center=z_center)


@pytest.mark.parametrize("name", list(TAPERS))
def test_sidewall_at_the_height_of_each_sub_interval(name: str) -> None:
    """A surface cutting a voxel gets the sidewall's mean over the level's part of it.

    Along the row y = 0 the strip's edges are straight lines in x, so the fill of
    voxel (i, k) is ``L * clip(1/2 - d_i - s, 0, 1)`` with the level's part [lo, hi]
    of the voxel, L = hi - lo, s the mean offset over it and d_i the distance to the
    edges. For a straight sidewall s is the offset at the middle z_m of the part; the
    voxel center would put the edge (z_c - z_m) tan(angle) off, here 0.125 and 0.15
    tan(angle) cells in the bottom and top voxels. Summed along the row the fill is
    linear in s, so the volume is exact (many thin voxels), also with a z_to_bias
    kink inside the part (review finding A2: the offset at z_m alone missed the
    ramp above the kink, 0.34 cells per side here).
    """
    taper = TAPERS[name]
    d, w = _tapered_strip(taper)
    nx, _, nz = d.grid_shape
    j = cell(d, 0.8, 0.0, 0.0)[1]
    z0 = d.z_center - nz * RES / 2  # stack z of the device bottom
    zb, t = (TAPER_Z[0] - z0) / RES, TAPER_Z[1] / RES  # cells
    xc = (d.center[0] - nx * RES / 2) / RES + np.arange(nx) + 0.5
    dist = np.maximum(STRIP[0][0] / RES - xc, xc - STRIP[1][0] / RES)  # signed distance, cells
    zs, bs = taper.get("z_to_bias", ((0.0, 1.0), (0.0, 0.0)))
    kinks = [zb + f * t for f in zs]

    def shift(z):
        """Lateral offset (cells) of the edges at heights z (cells)."""
        z_ref = zb + taper.get("width_to_z", 0.0) * t
        return (z - z_ref) * math.tan(math.radians(taper["sidewall_angle"])) - np.interp((z - zb) / t, zs, bs) / RES

    cut = kinked = 0
    for k in range(nz):
        lo, hi = max(k, zb), min(k + 1.0, zb + t)
        if hi <= lo:
            np.testing.assert_allclose(w["si"][:, j, k], 0.0, atol=1e-6)
            continue
        cut += hi - lo < 1
        kinked += any(lo < z < hi for z in kinks)
        zq = lo + (np.arange(4000) + 0.5) * (hi - lo) / 4000
        expected = (hi - lo) * np.clip(0.5 - dist - shift(zq).mean(), 0.0, 1.0)
        np.testing.assert_allclose(w["si"][:, j, k], expected, atol=2e-5, err_msg=f"k={k}")
        exact = (hi - lo) * np.clip(0.5 - dist[:, None] - shift(zq), 0.0, 1.0).mean(axis=1)
        assert float(w["si"][:, j, k].sum()) == pytest.approx(exact.sum(), abs=2e-4), f"k={k}"
    assert cut == 2  # the bottom and top surfaces cut a voxel each
    assert kinked == (name == "z_to_bias kink")  # the kink is inside the top voxel's part


def test_sidewall_gradients_where_surfaces_cut_voxels() -> None:
    """The middle of a cut voxel's part follows the surfaces: gradients match finite differences.

    Also with the surfaces on voxel boundaries (ties), in a single voxel layer
    (a 2D simulation) cut by the top surface, and with a z_to_bias kink inside
    the top voxel's part (the mean over the pieces between the kinks).
    """
    d, _ = _tapered_strip(TAPERS["sidewall"])
    flat, _ = _tapered_strip(TAPERS["sidewall"], size=(1.6, 1.4, RES), z_center=0.214)
    kinked, _ = _tapered_strip(TAPERS["z_to_bias kink"])
    assert flat.grid_shape[2] == 1

    def moment(device, zmin, thickness, angle):
        """First z moment (cells) of the silicon."""
        p = {"zmin/core": zmin, "thickness/core": thickness, "sidewall_angle/core": angle}
        names, w = device.material_weights(p)
        return (w[names.index("si")] * (jnp.arange(device.grid_shape[2]) + 0.5)).sum()

    for device, angle in ((d, 40.0), (flat, 40.0), (kinked, 15.0)):
        x0 = (*TAPER_Z, angle)
        grad = jax.grad(lambda *x, device=device: moment(device, *x), argnums=(0, 1, 2))
        g = grad(*x0)
        assert all(np.isfinite(float(x)) for x in g)
        for a, h in enumerate((1e-3, 1e-3, 0.1)):
            up, down = list(x0), list(x0)
            up[a] += h
            down[a] -= h
            fd = (moment(device, *up) - moment(device, *down)) / (2 * h)
            assert float(g[a]) == pytest.approx(float(fd), rel=5e-3), (device.grid_shape, a)
    # surfaces on voxel boundaries
    g = jax.grad(lambda *x: moment(d, *x), argnums=(0, 1, 2))(0.0, 0.2, 40.0)
    assert all(bool(jnp.isfinite(v)) for v in g)


def test_sidewall_ignores_other_levels_surfaces() -> None:
    """A level's sidewall comes from its own surfaces only (review finding I2-R1).

    The offset is the mean over the level's own part of each voxel, so a lower-priority cap
    whose bottom cuts the core's voxels changes nothing of the core: the silicon is the same
    wherever the cap starts (before, sampling the offset between every level's surfaces moved
    0.26 cells of silicon), and its derivative with respect to the cap's zmin is zero.
    """
    ls = LayerStack(layers={
        "core": LayerLevel(layer=WG, zmin=TAPER_Z[0], thickness=TAPER_Z[1], material="si", mesh_order=1,
                           **TAPERS["sidewall"]),
        "cap": LayerLevel(layer=WG, zmin=0.1061, thickness=0.2, material="sio2", mesh_order=2),
    })
    mats = {n: MATERIALS[n] for n in ("air", "si", "sio2")}
    objects, _, params = _placed(_polygons(STRIP), ls, mats, (1.6, 1.4, 0.3), (0.8, 0.0), z_center=0.11)
    d, p = objects["s"], params["s"]

    def silicon(cap_zmin):
        names, w = d.material_weights({**p, "zmin/cap": cap_zmin})
        return w[names.index("si")]

    far = np.asarray(silicon(0.5))  # the cap above the box
    assert far.sum() > 1e3
    for cap_zmin in (0.1061, 0.1153):  # its bottom inside a voxel the core fills
        np.testing.assert_allclose(np.asarray(silicon(cap_zmin)), far, atol=1e-6, err_msg=str(cap_zmin))
    grad = jax.grad(lambda c: (silicon(c) * (jnp.arange(d.grid_shape[2]) + 0.5)).sum())(0.1061)
    assert abs(float(grad)) < 1e-3


# ---------------------------------------------------------------- corners

CORE_ONLY = {"air": MATERIALS["air"], "si": MATERIALS["si"]}


def _core(**level) -> LayerStack:
    """A 0.22 um core on WG."""
    core = LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1, **level)
    return LayerStack(layers={"core": core})


def _rotated(points, degrees: float, origin) -> list[tuple[float, float]]:
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return [tuple(origin + r @ (np.asarray(p, dtype=float) - origin)) for p in points]


def _area_fractions(device, region) -> np.ndarray:
    """(nx, ny) fraction of every voxel's square inside a shapely region."""
    shapely = pytest.importorskip("shapely")
    nx, ny, _ = device.grid_shape
    x0, y0 = device.center[0] - nx * RES / 2, device.center[1] - ny * RES / 2
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    voxels = shapely.box(x0 + i * RES, y0 + j * RES, x0 + (i + 1) * RES, y0 + (j + 1) * RES)
    return shapely.area(shapely.intersection(region, voxels)) / RES**2


# non-convex, 17 deg off the grid, edges of 11 cells or more; in a (1.7, 1.5) um box around (0.6, 0.45)
CORNERS = _rotated(
    [(0.0, 0.0), (1.2, 0.0), (1.2, 0.75), (0.6, 0.75), (0.6, 0.375), (0.225, 0.375), (0.225, 0.9), (0.0, 0.9)],
    17.0,
    np.array([0.6137, 0.4429]),
)
# in a (1.2, 1.1) um box around (0.55, 0.45)
TRIANGLE = [(0.1137, 0.0712), (0.9871, 0.2043), (0.3519, 0.8164)]


def test_literals_match_eval_expr() -> None:
    """A level's literal tree is the sign of its distance plus an offset.

    With or, and, not, xor, grows, fill and missing layers.
    """
    rng = np.random.default_rng(0)
    shape = (30, 20)
    zero = jnp.zeros(shape)
    sdf = {i: (jnp.asarray(rng.normal(0.0, 2.0, shape), dtype=jnp.float32), zero, zero) for i in (1, 2, 3)}
    exprs = [
        ("or", ("layer", 1, 0.01), ("layer", 2, 0.0), -0.02),
        ("and", ("layer", 1, 0.0), ("not", ("fill", 0.0), ("layer", 3, 0.02), 0.0), 0.01),
        ("not", ("layer", 1, -0.01), ("xor", ("layer", 2, 0.0), ("layer", 3, 0.03), 0.01), 0.0),
        ("xor", ("layer", 1, 0.0), ("layer", 9, 0.0), 0.0),  # layer 9 has no polygons
        ("and", ("layer", 9, 0.0), ("layer", 2, 0.0), 0.0),
        ("or", ("fill", 0.0), ("layer", 2, 0.0), 0.0),
    ]
    for expr in exprs:
        tree, literals = _literals(expr, RES, sdf)
        assert all(i in sdf for i, *_ in literals)
        for s in (-1.3, 0.0, 0.7):  # lateral offset (cells)
            f = _eval_expr(expr, sdf, shape, RES)[0] + s
            inside = jnp.broadcast_to(_eval_bool(tree, [sdf[i][0] + a * s + b < 0 for i, a, b in literals]), shape)
            assert bool(jnp.all((inside == (f < 0)) | (jnp.abs(f) < 1e-4))), (expr, s)
    assert _literals(exprs[4], RES, sdf)[0] == ("false",)
    assert _literals(exprs[5], RES, sdf)[0] == ("true",)


def test_convex_corner_at_voxel_center_fills_a_quarter() -> None:
    """A right-angle corner on a voxel center fills a quarter of it; one half-plane per level fills half."""
    square = [(0.01, 0.01), (0.61, 0.01), (0.61, 0.41), (0.01, 0.41)]
    for lines, expected in ((2, 0.25), (0, 0.5)):
        d, w = _weights_of(
            _polygons(square), _core(), CORE_ONLY, (1.0, 0.8, 0.3), center=(0.3, 0.2), z_center=0.11, corner_lines=lines
        )
        i, j, k = cell(d, 0.01, 0.01, 0.11)
        around = w["si"][i - 1 : i + 2, j - 1 : j + 2, k]
        np.testing.assert_allclose(around, [[0, 0, 0], [0, expected, 0.5], [0, 0.5, 1]], atol=1e-5)


def test_reflex_corner_fills_three_quarters() -> None:
    """The inner corner of an L on a voxel center fills three quarters of it."""
    ell = [(0.01, 0.01), (0.61, 0.01), (0.61, 0.21), (0.21, 0.21), (0.21, 0.61), (0.01, 0.61)]
    d, w = _weights_of(_polygons(ell), _core(), CORE_ONLY, (1.0, 1.0, 0.3), center=(0.3, 0.3), z_center=0.11)
    i, j, k = cell(d, 0.21, 0.21, 0.11)
    assert float(w["si"][i, j, k]) == pytest.approx(0.75, abs=1e-5)


def test_polygon_corners_get_exact_areas() -> None:
    """Every voxel of a rotated non-convex polygon, at its corners too, gets its exact area."""
    shapely = pytest.importorskip("shapely")
    d, w = _weights_of(_polygons(CORNERS), _core(), CORE_ONLY, (1.7, 1.5, 0.3), center=(0.6, 0.45), z_center=0.11)
    k = cell(d, 0, 0, 0.11)[2]
    exact = _area_fractions(d, shapely.Polygon(CORNERS))
    assert ((exact > 0) & (exact < 1)).sum() > 300
    np.testing.assert_allclose(w["si"][:, :, k], exact, atol=1e-4)


@pytest.mark.parametrize("bias", [0.01, -0.01])
@pytest.mark.parametrize("width_to_z", [0.0, 1.0])
@pytest.mark.parametrize("angle", [20.0, -20.0])
def test_offset_corners_get_exact_areas(angle: float, width_to_z: float, bias: float) -> None:
    """Sidewalls and biases move the corners: sharp where the offset keeps them, round where it rounds them.

    The level is ``{F + s(z) < 0}``, the polygon grown by -s (round joins: grown convex
    and shrunk reflex corners are circular arcs). Every voxel lies between the exact area
    at the middle of the voxel (half-plane columns, and corner columns where the edge is
    straight in the square, so a straight edge has no seam) and the mean of the exact
    areas at the two Gauss heights (corner columns at a corner), which is the 3D volume
    where a corner's area is quadratic in z, to 1e-3: the seam between the two fades out
    smoothly as a corner enters a square (measured 5e-4; it was 1e-4 with the near
    discontinuous switch of review R3-findings-1). Against the 3D volume (a z-quadrature),
    the total error is at most 0.98 of the cross-section's at the middle (0.93 to 0.972 here).
    """
    shapely = pytest.importorskip("shapely")
    ls = _core(sidewall_angle=angle, width_to_z=width_to_z, bias=bias)
    d, w = _weights_of(_polygons(CORNERS), ls, CORE_ONLY, (1.7, 1.5, 0.3), center=(0.6, 0.45), z_center=0.11)
    nz = d.grid_shape[2]
    for z in (0.03, 0.11, 0.19):  # offsets from -4 to 4 cells
        k = cell(d, 0, 0, z)[2]
        z_lo = d.z_center + (k - nz / 2) * RES

        def area(t, z_lo=z_lo):
            """Exact fill at the fraction t of the voxel's height."""
            s = (z_lo + t * RES - width_to_z * 0.22) * math.tan(math.radians(angle)) - bias
            return _area_fractions(d, shapely.Polygon(CORNERS).buffer(-s, quad_segs=256))

        mid = area(0.5)
        gauss = 0.5 * (area(0.5 - 0.5 / math.sqrt(3)) + area(0.5 + 0.5 / math.sqrt(3)))
        got = w["si"][:, :, k]
        off = np.maximum(np.minimum(mid, gauss) - got, got - np.maximum(mid, gauss))
        assert float(off.max()) <= 1e-3, f"z={z}: {float(off.max()):.2e}"
        volume = sum(area((q + 0.5) / 16) for q in range(16)) / 16
        assert np.abs(got - volume).sum() <= 0.98 * np.abs(mid - volume).sum(), z


def test_sloped_corner_gets_its_volume() -> None:
    """A right-angle corner of a 20 deg wall, on a voxel center halfway up: the corner model integrates
    its area, quadratic in z, exactly (the cross-section at the middle misses tan^2 / 12 of the voxel)."""
    shapely = pytest.importorskip("shapely")
    square = [(0.01, 0.01), (0.61, 0.01), (0.61, 0.41), (0.01, 0.41)]
    ls = _core(sidewall_angle=20.0, width_to_z=0.5)
    d, w = _weights_of(_polygons(square), ls, CORE_ONLY, (1.0, 0.8, 0.3), center=(0.3, 0.2), z_center=0.11)
    i, j, k = cell(d, 0.01, 0.01, 0.11)
    z_lo = d.z_center + (k - d.grid_shape[2] / 2) * RES

    def area(t):
        """Exact fill at the fraction t of the voxel's height (the wall is at the drawn square at z = 0.11)."""
        s = (z_lo + t * RES - 0.11) * math.tan(math.radians(20.0))
        return _area_fractions(d, shapely.Polygon(square).buffer(-s, quad_segs=256))

    volume, mid = sum(area((q + 0.5) / 64) for q in range(64)) / 64, area(0.5)
    around = (slice(i - 1, i + 2), slice(j - 1, j + 2))
    assert abs(float(mid[i, j] - volume[i, j])) > 5e-3  # the cross-section's error at the corner
    np.testing.assert_allclose(w["si"][:, :, k][around], volume[around], atol=1e-4)


def _sweep(d, params, vertices, probe):
    """Fill of si in the voxels ``probe`` as coordinate ``axis`` of the WG vertices ``rows`` moves to x.

    Args:
        d: the placed StackDevice.
        params: its parameters.
        vertices: (rows, axis).
        probe: index of the voxels into the (nx, ny, nz) fill.

    Returns:
        The jitted fill f(x) and its jvp (per um), with the default corner model.
    """
    p0 = dict(params)
    v0 = jnp.asarray(p0["vertices/WG"], jnp.float32)
    si = d.material_weights(p0)[0].index("si")
    rows, axis = vertices

    def fill(x):
        v = v0.at[rows, axis].set(x)
        return d.material_weights({**p0, "vertices/WG": v})[1][si][probe]

    return jax.jit(fill), jax.jit(lambda x: jax.jvp(fill, (x,), (jnp.ones_like(x),))[1])


@pytest.mark.parametrize("angle", [10.0, 20.0])
def test_bent_vertex_on_a_sloped_wall_has_no_seam(angle: float) -> None:
    """A vertex on a sloped wall bent off collinear by 1e-3 .. 0.15 cells: the fill moves smoothly from
    the ordinary columns' T to the corner model, AD sees it, and no seam opens (review R3-findings-1:
    a 1e-4-wide gradient-stopped switch made the fill jump by up to 0.045 with FD 7/cell vs jvp 0.2)."""
    c = _polygons([(-0.3, -0.3), (0.01, -0.301), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)])
    ls = _core(sidewall_angle=angle, width_to_z=0.5)
    objects, _, params = _placed(c, ls, CORE_ONLY, (1.0, 0.8, 0.3), (0.0, 0.0), z_center=0.11, diagnostics=False)
    d = objects["s"]
    v0 = np.asarray(params["s"]["vertices/WG"])
    m = int(np.argmin(np.abs(v0[:, 0] - 0.01) + np.abs(v0[:, 1] + 0.301)))
    i, j, k = cell(d, 0.01, -0.3, 0.11)  # the wall crosses a side of the square in voxel layer k
    f, jvp = _sweep(d, params["s"], ([m], 1), (i, slice(j - 1, j + 1), k))
    h = 1e-4 * RES
    for delta in (1.5e-3, 5e-3, 1.5e-2, 5e-2, 0.15, -5e-3, -5e-2):
        x = jnp.float32(-0.3 + delta * RES)
        w = np.asarray(f(x))
        below = w[0] if delta > 0 else 1.0 - w[1]  # the voxel the wall leaves alone: 0 (or 1 above)
        assert below <= 2e-3, (delta, w)
        fd = (np.asarray(f(x + h)) - np.asarray(f(x - h))) / (2 * h) * RES
        np.testing.assert_allclose(np.asarray(jvp(x)) * RES, fd, atol=0.05, err_msg=str(delta))


@pytest.mark.parametrize("angle", [10.0, 20.0])
def test_sloped_corner_entering_a_voxel_is_smooth(angle: float) -> None:
    """A convex corner of a sloped wall crosses a side of a voxel's square (the wall crosses that side
    within the voxel layer): the fill's slope stays O(1)/cell and matches AD (review R3-findings-1: it
    reached 6.6/cell, jvp -0.004).

    From 0.15 cells before the side to 0.05 cells past it: the corner first enters the square at the
    upper Gauss height (tan(angle) / (2 sqrt 3) = 0.05 or 0.1 cells before the side), where the seam
    fades out, at most (1 + 1.54 / _STRAIGHT_WIDTH) times the corner's own rate of about 0.47/cell
    (measured 1.94/cell at 10 deg); past the side the slope is that rate. The fill only kinks (its
    slope jumps, its value does not) where the corner crosses the side at a Gauss height, like the
    half-planes' at the middle (the grid-line tie): AD is compared with the slopes between those ties.
    """
    c = _polygons([(-0.2, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.2, 0.3)])
    ls = _core(sidewall_angle=angle, width_to_z=0.5)
    objects, _, params = _placed(c, ls, CORE_ONLY, (1.0, 0.8, 0.3), (0.0, 0.0), z_center=0.11, diagnostics=False)
    d = objects["s"]
    v0 = np.asarray(params["s"]["vertices/WG"])
    left = np.flatnonzero(np.abs(v0[:, 0] + 0.2) < 1e-6).tolist()
    i, j, k = cell(d, -0.19, -0.29, 0.11)
    f, jvp = _sweep(d, params["s"], (left, 0), (i, slice(j - 1, j + 1), k))
    ts = np.concatenate([np.arange(-150, 0), np.arange(1, 51)]) * 1e-3  # cells from the side, off the tie at 0
    xs = -0.2 + RES * ts
    vals = np.array([np.asarray(f(jnp.float32(x))) for x in xs])
    jvps = np.array([np.asarray(jvp(jnp.float32(x))) for x in xs]) * RES
    steps = np.flatnonzero(np.diff(ts) < 1.5e-3)  # neighbours on the same side of the tie
    slope = (vals[steps + 1] - vals[steps]) / (xs[steps + 1] - xs[steps])[:, None] * RES
    assert np.abs(slope).max() <= 2.5, np.abs(slope).max()
    assert np.abs(slope[ts[steps] > 0]).max() <= 0.6  # past the side: the corner's own rate
    tie = math.tan(math.radians(angle)) / (2 * math.sqrt(3))  # cells: the corner on the side at a Gauss height
    smooth = np.all([(ts[steps] > t) | (ts[steps + 1] < t) for t in (-tie, tie)], axis=0)  # steps not across one
    assert (~smooth).sum() == 1  # the upper height's, before the side
    np.testing.assert_allclose(slope[smooth], 0.5 * (jvps[steps] + jvps[steps + 1])[smooth], atol=0.02)


def test_later_levels_corner_leaves_a_sloped_edge_alone() -> None:
    """A lower-priority level's corners in the squares along a sloped straight edge change nothing of it.

    They make those columns corner columns, where the core's 20 deg wall crosses a side of the square
    within the voxel layer: the core keeps the ordinary columns' mean-offset fill there (no seam along
    its edge) and gets no gradient from the other level's vertices, since the straight-edge seam is
    faded per level, on the union of the levels painted up to it. (A fade shared by all levels, as in
    the first fix of review R3-findings-1, and the 1e-4 switch before it, let the other level's corner
    hold the edge at its two Gauss heights there: tan(20 deg) / (4 sqrt 3) = 0.05 off.)
    """
    core = [(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)]  # its bottom edge on a grid line
    pad = [(-0.0157, -0.42), (0.1843, -0.42), (0.1843, -0.3063), (-0.0157, -0.3063)]  # top corners below it
    ls = LayerStack(layers={
        "core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1,
                           sidewall_angle=20.0, width_to_z=0.5),
        "pad": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.22, material="sio2", mesh_order=2),
    })
    mats = {n: MATERIALS[n] for n in ("air", "si", "sio2")}
    c = _polygons(core)
    c.add_polygon(pad, layer=SLAB)
    where = dict(size=(1.0, 1.0, 0.3), center=(0.0, 0.0), z_center=0.11, diagnostics=False)
    objects, _, params = _placed(c, ls, mats, **where)
    d, p = objects["s"], params["s"]
    names, w = d.material_weights(p)
    alone = _weights_of(_polygons(core), _core(sidewall_angle=20.0, width_to_z=0.5), CORE_ONLY, **where)[1]
    si = np.asarray(w[names.index("si")])
    i, j, k = cell(d, -0.01, -0.31, 0.11)  # the pad's corner, in the row below the core's edge
    assert float(alone["si"][i, j, k]) == pytest.approx(0.0, abs=1e-5)  # the mean offset is on the side
    np.testing.assert_allclose(si, alone["si"], atol=1e-5)
    assert float(np.asarray(w).min()) >= -1e-6
    np.testing.assert_allclose(np.asarray(w).sum(axis=0), 1.0, atol=1e-5)
    grad = jax.grad(lambda v: d.material_weights({**p, "vertices/SLAB150": v})[1][names.index("si")].sum())(
        p["vertices/SLAB150"]
    )
    assert float(jnp.abs(grad).max()) * RES <= 1e-4


def test_derived_layer_corners_get_exact_areas() -> None:
    """WG - ETCH next to WG & ETCH, with a slanted etch edge through a WG corner: exact, no leak at the seam."""
    shapely = pytest.importorskip("shapely")
    wg = _rotated([(0.0, -0.3), (0.9, -0.3), (0.9, 0.3), (0.0, 0.3)], 8.0, np.array([0.45, 0.0]))
    corner, along = np.asarray(wg[1]), np.array([math.cos(math.radians(70)), math.sin(math.radians(70))])
    p0, p1 = corner - along, corner + along
    etch = [tuple(p0), tuple(p1), tuple(p1 + [1.0, 0.0]), tuple(p0 + [1.0, 0.0])]
    c = gf.Component()
    c.add_polygon(wg, layer=WG)
    c.add_polygon(etch, layer=ETCH)
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}
    d, w = _weights_of(c, _seam_stack("sio2"), mats, (1.3, 1.0, 0.3), center=(0.5, 0.0), z_center=0.11)
    k = cell(d, 0, 0, 0.11)[2]
    a, b = shapely.Polygon(wg), shapely.Polygon(etch)
    for name, region in (("si", a.difference(b)), ("sio2", a.intersection(b))):
        np.testing.assert_allclose(w[name][:, :, k], _area_fractions(d, region), atol=1e-4, err_msg=name)
    np.testing.assert_allclose(w["air"][:, :, k], 1 - _area_fractions(d, a), atol=1e-4)


def test_vertex_outside_the_voxel() -> None:
    """A 20 deg tip whose vertex is just outside a voxel that both its edges cross."""
    shapely = pytest.importorskip("shapely")
    half = math.tan(math.radians(10.0))
    tip = [(0.298, 0.01), (1.298, 0.01 - half), (1.298, 0.01 + half)]  # 0.1 cells left of the voxel [0.30, 0.32]
    fills = {}
    for lines in (2, 0):
        d, w = _weights_of(
            _polygons(tip), _core(), CORE_ONLY, (1.4, 0.6, 0.3), center=(0.8, 0.0), z_center=0.11, corner_lines=lines
        )
        fills[lines] = w["si"][:, :, cell(d, 0, 0, 0.11)[2]]
    i, j, _ = cell(d, 0.31, 0.01, 0.11)
    exact = _area_fractions(d, shapely.Polygon(tip))
    assert exact[i, j] == pytest.approx(2 * half * 0.6, abs=1e-6)
    np.testing.assert_allclose(fills[2], exact, atol=1e-4)
    assert abs(float(fills[0][i, j]) - exact[i, j]) > 0.1  # one half-plane is far off


def test_corner_model_never_worse() -> None:
    """Random stars in WG - ETCH and WG | ETCH levels with biases: in every voxel, for every
    material, the fill is at least as close to the exact one as with one half-plane per level."""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(1)

    def star(cx, cy, n, r0, r1):
        phi = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(r0, r1, n)
        return [(cx + ri * math.cos(p), cy + ri * math.sin(p)) for ri, p in zip(r, phi)]

    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}
    for _ in range(2):
        a, b = star(0.5, 0.5, 7, 0.25, 0.5), star(0.7, 0.6, 6, 0.2, 0.4)
        c = gf.Component()
        c.add_polygon(a, layer=WG)
        c.add_polygon(b, layer=ETCH)
        b1, b2 = (float(x) for x in rng.uniform(-0.02, 0.02, 2))
        derived = {"derived_layer": LogicalLayer(layer=WG), "zmin": 0.0, "thickness": 0.22}
        ls = LayerStack(layers={
            "one": LayerLevel(layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), material="si", mesh_order=1,
                              bias=b1, **derived),
            "two": LayerLevel(layer=LogicalLayer(layer=WG) | LogicalLayer(layer=ETCH), material="sio2", mesh_order=2,
                              bias=b2, **derived),
        })
        pa, pb = shapely.Polygon(a), shapely.Polygon(b)
        for mode in ("composite", "exact"):
            box_ = dict(size=(1.4, 1.4, 0.3), center=(0.6, 0.55), z_center=0.11, derived_shapes=mode)
            d, new = _weights_of(c, ls, mats, **box_)
            _, old = _weights_of(c, ls, mats, **box_, corner_lines=0)
            k = cell(d, 0, 0, 0.11)[2]
            if mode == "composite":  # the level sets {F - bias < 0}: max(F_a, -F_b) and min(F_a, F_b)
                one = pa.buffer(b1, quad_segs=256).difference(pb.buffer(-b1, quad_segs=256))
                two = pa.buffer(b2, quad_segs=256).union(pb.buffer(b2, quad_segs=256)).difference(one)
            else:  # the offsets of the exact shapes (round joins)
                one = pa.difference(pb).buffer(b1, quad_segs=256)
                two = pa.union(pb).buffer(b2, quad_segs=256).difference(one)
            f_one, f_two = _area_fractions(d, one), _area_fractions(d, two)
            total_new = total_old = 0.0
            for name, exact in (("si", f_one), ("sio2", f_two), ("air", 1 - f_one - f_two)):
                e_new, e_old = np.abs(new[name][:, :, k] - exact), np.abs(old[name][:, :, k] - exact)
                assert float((e_new - e_old).max()) <= 1e-5, (mode, name)
                total_new, total_old = total_new + e_new.sum(), total_old + e_old.sum()
            # exact shapes: tips at crossings that two levels share keep the half-plane (fewer certified)
            assert total_new < (0.2 if mode == "composite" else 0.5) * total_old, mode


def test_corner_model_never_worse_for_the_voxel_average() -> None:
    """With xy_subsamples=2 the corner model is blended by one certificate per voxel, the least of its
    sub-squares': the voxel's average stays at least as close to the exact fill as with half-planes.
    (With a weight per sub-square, one made exact while another kept its half-plane once moved this
    voxel average 6.8e-3 further off: their half-plane errors had cancelled.)"""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(100)

    def star(cx, cy, n, r0, r1):
        phi = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(r0, r1, n)
        return [(cx + ri * math.cos(p), cy + ri * math.sin(p)) for ri, p in zip(r, phi)]

    wgs = [star(rng.uniform(0.4, 0.8), rng.uniform(0.4, 0.7), int(rng.integers(5, 9)), 0.15, 0.45)
           for _ in range(int(rng.integers(1, 3)))]
    etch = star(0.7, 0.6, int(rng.integers(4, 7)), 0.15, 0.4)
    c = gf.Component()
    for p in wgs:
        c.add_polygon(p, layer=WG)
    c.add_polygon(etch, layer=ETCH)
    bias = 0.0024648867736126907
    derived = {"derived_layer": LogicalLayer(layer=WG), "zmin": 0.0, "thickness": 0.22}
    ls = LayerStack(layers={
        "one": LayerLevel(layer=LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH), material="si", mesh_order=1,
                          **derived),
        "two": LayerLevel(layer=LogicalLayer(layer=WG), material="sio2", mesh_order=2, bias=bias, **derived),
    })
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}
    d, new = _weights_of(c, ls, mats, (1.5, 1.4, 0.3), center=(0.6, 0.55), z_center=0.11, xy_subsamples=2)
    _, old = _weights_of(c, ls, mats, (1.5, 1.4, 0.3), center=(0.6, 0.55), z_center=0.11, xy_subsamples=2,
                         corner_lines=0)
    nx, ny, nz = d.grid_shape
    k = nz // 2
    one = shapely.union_all([shapely.Polygon(p) for p in wgs]).intersection(shapely.Polygon(etch))
    two = shapely.union_all([shapely.Polygon(p).buffer(bias, quad_segs=256) for p in wgs]).difference(one)
    x0, y0 = d.center[0] - nx * RES / 2, d.center[1] - ny * RES / 2
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    exact = {"si": 0.0, "sio2": 0.0}
    for a in (0.25, 0.75):  # the sub-squares, half a cell wide
        for b in (0.25, 0.75):
            sq = shapely.box(x0 + (i + a - 0.25) * RES, y0 + (j + b - 0.25) * RES, x0 + (i + a + 0.25) * RES,
                             y0 + (j + b + 0.25) * RES)
            for name, region in (("si", one), ("sio2", two)):
                exact[name] = exact[name] + shapely.area(shapely.intersection(region, sq)) / (0.5 * RES) ** 2 / 4
    exact["air"] = 1 - exact["si"] - exact["sio2"]
    gain = 0.0
    for name, f in exact.items():
        e_new, e_old = np.abs(new[name][:, :, k] - f), np.abs(old[name][:, :, k] - f)
        assert float((e_new - e_old).max()) <= 1e-5, name
        gain += float(e_old.sum() - e_new.sum())
    assert gain > 0.2  # the corner model still helps (0.45 summed over the layer)


def test_corner_lines_zero_is_unchanged() -> None:
    """corner_lines=0 is one half-plane per level; the corner model changes only columns near corners."""
    rectangle = _rotated([(0.0, 0.0), (0.9, 0.0), (0.9, 0.5), (0.0, 0.5)], 17.0, np.array([0.4511, 0.2493]))
    ls = _core(sidewall_angle=10.0, width_to_z=0.5)
    d, w = _weights_of(_polygons(rectangle), ls, CORE_ONLY, (1.3, 1.0, 0.3), center=(0.45, 0.25), z_center=0.11)
    d0, w0 = _weights_of(
        _polygons(rectangle), ls, CORE_ONLY, (1.3, 1.0, 0.3), center=(0.45, 0.25), z_center=0.11, corner_lines=0
    )
    assert d0.corner_columns is None and d.corner_columns > 0
    changed = np.any(w["si"] != w0["si"], axis=2)
    assert changed.sum() >= 4
    # within reach of both edges of a corner: the square's half-diagonal and the
    # largest sidewall offset (0.11 um tan(10 deg)) from each, so sqrt(2) times that from the vertex
    reach = math.sqrt(2.0) * (math.sqrt(0.5) * RES + 0.11 * math.tan(math.radians(10.0))) + 1e-3
    nx, ny, _ = d.grid_shape
    x = d.center[0] + (np.arange(nx) + 0.5 - nx / 2) * RES
    y = d.center[1] + (np.arange(ny) + 0.5 - ny / 2) * RES
    near = np.zeros((nx, ny), dtype=bool)
    for cx, cy in rectangle:
        near |= np.hypot(x[:, None] - cx, y[None, :] - cy) <= reach
    assert not (changed & ~near).any()  # elsewhere bit-identical


@pytest.mark.parametrize("angle", [15.0, -15.0])  # shrinks the corners (sharp) / grows them (round)
def test_corner_gradients_match_finite_differences(angle: float) -> None:
    """Silicon moment of a triangle w.r.t. a vertex, the sidewall angle and the thickness."""
    d, _ = _weights_of(
        _polygons(TRIANGLE), _core(sidewall_angle=angle, width_to_z=0.0), CORE_ONLY, (1.2, 1.1, 0.3),
        center=(0.55, 0.45), z_center=0.11,
    )
    v0 = d.init_params()["vertices/WG"]
    x = jnp.arange(d.grid_shape[0])[:, None, None] + 0.5

    def moment(dx, angle, thickness):
        p = {"vertices/WG": v0.at[1, 0].add(dx), "sidewall_angle/core": angle, "thickness/core": thickness}
        names, w = d.material_weights(p)
        return (w[names.index("si")] * x).sum()

    x0 = (0.0, angle, 0.2137)  # the top surface cuts a voxel
    g = jax.grad(moment, argnums=(0, 1, 2))(*x0)
    for a, h in enumerate((1e-3, 0.1, 1e-3)):
        up, down = list(x0), list(x0)
        up[a] += h
        down[a] -= h
        fd = (moment(*up) - moment(*down)) / (2 * h)
        assert float(g[a]) == pytest.approx(float(fd), rel=5e-3), a


def _degenerate_corners():
    """Component, two-level stack and materials with corners on voxel centers and corners, edges on voxel
    boundaries, 180 and 1 deg vertices and a corner shared by two levels (place around (0.5, 0.3) um)."""
    c = gf.Component()
    spike = 0.55 * math.tan(math.radians(0.5))
    for poly in (
        [(0.01, 0.01), (0.31, 0.01), (0.31, 0.21), (0.01, 0.21)],  # vertices on voxel centers
        [(0.40, 0.0), (0.70, 0.0), (0.70, 0.20), (0.40, 0.20)],  # on voxel corners, edges on voxel boundaries
        [(0.0, 0.40), (0.15, 0.40), (0.30, 0.40), (0.30, 0.60), (0.0, 0.60)],  # a 180 deg vertex
        [(0.45, 0.50), (1.0, 0.50 - spike), (1.0, 0.50 + spike)],  # a 1 deg tip
    ):
        c.add_polygon(poly, layer=WG)
    c.add_polygon([(0.31, 0.21), (0.51, 0.21), (0.51, 0.31), (0.31, 0.31)], layer=SLAB)  # a shared corner
    ls = LayerStack(layers={
        "core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1, sidewall_angle=10.0,
                           width_to_z=0.5),
        "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.1, material="sio2", mesh_order=2),
    })
    return c, ls, {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}


def test_corner_gradients_are_finite_at_degeneracies() -> None:
    """Corners on voxel centers and corners, edges on voxel boundaries, 180 and 1 deg vertices, a corner
    shared by two levels; also in a 2D grid and with sub-samples on a narrower square."""
    c, ls, mats = _degenerate_corners()
    core = LayerStack(layers={"core": ls.layers["core"]})
    for stack_, size, center, kwargs in (
        (ls, (1.2, 0.8, 0.3), (0.5, 0.3), {}),
        (core, (1.2, RES, 0.3), (0.5, 0.01), {}),  # one cell along y, through the corners on voxel centers
        (core, (1.2, 0.8, 0.3), (0.5, 0.3), {"xy_subsamples": 2, "edge_width": 0.8}),
    ):
        d, _ = _weights_of(c, stack_, mats, size, center=center, z_center=0.11, **kwargs)

        def loss(p, d=d):
            w = d.material_weights(p)[1]
            return (w * jax.random.uniform(jax.random.PRNGKey(0), w.shape)).sum()

        g = jax.grad(loss)(d.init_params())
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g)), d.grid_shape


def test_disk_area_matches_shapely() -> None:
    """Area of convex polygons in disks: generic, the center on a vertex, tangent, all and nothing."""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(4)
    polys, counts, centers, radii, expected = [], [], [], [], []
    for _ in range(8):
        hull = shapely.MultiPoint(rng.uniform(-1, 1, (6, 2))).convex_hull
        pts = np.asarray(shapely.geometry.polygon.orient(hull, 1.0).exterior.coords)[:-1]
        edge = pts[1] - pts[0]
        normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)  # outward
        inner = pts.mean(axis=0)
        for center, r in (
            (rng.uniform(-1, 1, 2), rng.uniform(0.2, 1.0)),
            (pts[0], 0.3),  # the center on a vertex
            (inner, float(normal @ (pts[0] - inner))),  # tangent to an edge from inside
            (pts[0] + 0.5 * edge + 0.4 * normal, 0.4),  # tangent from outside
            (inner, 5.0),  # the whole polygon
            (inner + 5.0, 0.5),  # nothing
        ):
            polys.append(np.concatenate([pts, np.repeat(pts[-1:], 8 - len(pts), axis=0)]))
            counts.append(len(pts))
            centers.append(center)
            radii.append(r)
            disk = shapely.Point(center).buffer(r, quad_segs=4096)
            expected.append(shapely.Polygon(pts).intersection(disk).area)
    poly, center, r = (jnp.asarray(np.asarray(x), dtype=jnp.float32) for x in (polys, centers, radii))
    count = jnp.asarray(counts, dtype=jnp.int32)
    np.testing.assert_allclose(_disk_area(poly, count, center, r), expected, atol=2e-5)
    g = jax.grad(lambda *x: _disk_area(x[0], count, x[1], x[2]).sum(), argnums=(0, 1, 2))(poly, center, r)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)


def test_corner_budget() -> None:
    """Placement sizes corner_columns from the candidates; a smaller budget repaints at most that many columns."""
    args = (_polygons(CORNERS), _core(), CORE_ONLY, (1.7, 1.5, 0.3))
    d, _ = _weights_of(*args, center=(0.6, 0.45), z_center=0.11)
    nx, ny, _ = d.grid_shape
    assert 0 < d._corner_candidate_count() <= d.corner_columns <= nx * ny
    _, w8 = _weights_of(*args, center=(0.6, 0.45), z_center=0.11, corner_columns=8)
    _, w0 = _weights_of(*args, center=(0.6, 0.45), z_center=0.11, corner_lines=0)
    changed = np.any(w8["si"] != w0["si"], axis=2)
    assert 0 < changed.sum() <= 8
    np.testing.assert_allclose(w8["si"] + w8["air"], 1.0, atol=1e-6)


def test_corners_where_a_surface_cuts_the_voxel() -> None:
    """A tapered level whose top surface cuts the voxels: half-planes at the middle of the level's part, the
    corner model at the part's two Gauss heights.

    Every voxel lies between the two (the corner model integrates the corners' areas,
    quadratic in the offset, exactly; the midpoint rule would miss ``L ds^2 A'' / 24``
    there, with the offset change ds = 0.7 tan(30 deg) cells over the part and A'' of
    order 2 cot(26 deg) at the triangle's sharpest corner, a few hundredths; where a
    corner just enters a square, the seam between the two fades out smoothly, so such a
    voxel is in between: up to 2.8e-3 from both, 8e-7 outside the range). Both
    match many thin voxels up to the residual where an edge crosses a corner of the
    square within the part (straight edges are linear in the offset elsewhere).
    """
    shapely = pytest.importorskip("shapely")
    ls = LayerStack(layers={"core": LayerLevel(
        layer=WG, zmin=TAPER_Z[0], thickness=TAPER_Z[1], material="si", mesh_order=1, sidewall_angle=30.0,
        width_to_z=0.5,
    )})
    d, w = _weights_of(_polygons(TRIANGLE), ls, CORE_ONLY, (1.2, 1.1, 0.3), center=(0.55, 0.45), z_center=0.11)
    k = cell(d, 0, 0, 0.21)[2]  # [0.20, 0.22] um, cut at 0.7 by the top
    z0, top = d.z_center + (k - d.grid_shape[2] / 2) * RES, sum(TAPER_Z)
    z_ref = TAPER_Z[0] + 0.5 * TAPER_Z[1]

    def fill(z):
        offset = (z - z_ref) * math.tan(math.radians(30.0))
        return _area_fractions(d, shapely.Polygon(TRIANGLE).buffer(-offset, quad_segs=256))

    part = (top - z0) / RES
    mid = part * fill(0.5 * (z0 + top))
    gauss = 0.5 * part * sum(fill(0.5 * (z0 + top) + g * (top - z0) / (2 * math.sqrt(3))) for g in (-1, 1))
    got = w["si"][:, :, k]
    off = np.maximum(np.minimum(mid, gauss) - got, got - np.maximum(mid, gauss))
    assert float(off.max()) <= 1e-4
    thin = part * np.mean([fill(z) for z in z0 + (np.arange(40) + 0.5) * (top - z0) / 40], axis=0)
    np.testing.assert_allclose(w["si"][:, :, k], thin, atol=0.04)
    assert float(w["si"][:, :, k].sum()) == pytest.approx(float(thin.sum()), abs=0.1)  # ~0.05 from the corners


# ---------------------------------------------------------------- Kottke normals


def test_clip_carries_edge_attributes() -> None:
    """Every vertex carries its outgoing edge's attributes: kept vertices their own, the crossing that
    leaves the kept side the clipping line's, the crossing that enters it those of the edge it lies on."""
    square = jnp.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
    attr = jnp.asarray([[1.0], [2.0], [3.0], [4.0], [4.0]])
    n, s = jnp.asarray([0.6, 0.8]), jnp.asarray(0.05)
    poly, count = _clip(square, jnp.asarray(4), n, s)
    clipped, clipped_count, clipped_attr = _clip(square, jnp.asarray(4), n, s, attr, jnp.asarray([9.0]))
    np.testing.assert_array_equal(clipped, poly)
    assert int(clipped_count) == int(count) == 4
    # enters on edge 1 at y = -0.4375 and leaves on edge 3 at y = 0.3125
    np.testing.assert_allclose(clipped[:4], [[0.5, -0.4375], [0.5, 0.5], [-0.5, 0.5], [-0.5, 0.3125]], atol=1e-6)
    np.testing.assert_array_equal(clipped_attr[:4, 0], [2.0, 3.0, 4.0, 9.0])


def test_moments_match_quadrature() -> None:
    """Area, kernel flux and rate of squares clipped by up to three random planes (also along a side,
    through two vertices, nothing left and nothing removed) match dense quadrature along the edges,
    for two kernels; their gradients are finite."""
    rng = np.random.default_rng(2)
    num = 64
    planes = []  # (n_x, n_y, offset, z slope) per trial and clip
    for _ in range(3):
        phi = rng.uniform(0, 2 * np.pi, num)
        plane = np.stack([np.cos(phi), np.sin(phi), rng.uniform(-0.8, 0.8, num), rng.normal(size=num)], axis=-1)
        plane[rng.uniform(size=num) < 0.3, 2] = 2.0  # removes nothing
        planes.append(plane)
    planes = np.stack(planes, axis=1)
    planes[0, 0] = [1.0, 0.0, 0.5, 0.3]  # along the side x = -1/2
    planes[1, 0] = [math.sqrt(0.5), math.sqrt(0.5), 0.0, -0.2]  # through two vertices
    planes[2, 0] = [0.6, 0.8, -2.0, 1.0]  # nothing left
    planes[3] = [0.0, 1.0, 2.0, 0.0]  # the whole square
    planes = jnp.asarray(planes, dtype=jnp.float32)
    square = jnp.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]] + [[-0.5, 0.5]] * 3)

    def clipped(planes):
        poly, count = jnp.broadcast_to(square, (num, 7, 2)), jnp.full((num,), 4)
        attr = jnp.zeros((num, 7, 3))  # the square's sides are not interfaces
        for k in range(3):
            n, s, b = planes[:, k, :2], planes[:, k, 2], planes[:, k, 3:]
            poly, count, attr = _clip(poly, count, n, s, attr, jnp.concatenate([n, b], axis=-1))
        return poly, count, attr

    poly, count, attr = clipped(planes)
    assert int(count[2]) == 0 and int(count[3]) == 4
    t = (np.arange(4000) + 0.5) / 4000
    for kernel in ((0.0, 0.0, 1.0), (-0.25, 0.25, 0.5)):
        area, flux, rate = _moments(poly, count, attr, kernel)
        np.testing.assert_array_equal(area, _area(poly, count))
        ax, ay, beta = kernel
        for i in range(num):
            p, a = (np.asarray(x[i, : int(count[i])], dtype=float) for x in (poly, attr))
            d = np.roll(p, -1, axis=0) - p
            ell = d[:, 0] * a[:, 1] - d[:, 1] * a[:, 0]
            points = p[:, None] + t[None, :, None] * d[:, None]
            w = (1 - 4 * (ax + beta * points[..., 0]) ** 2) * (1 - 4 * (ay + beta * points[..., 1]) ** 2)
            np.testing.assert_allclose(flux[i], (ell * w.mean(axis=1)) @ a, atol=1e-4)
            np.testing.assert_allclose(rate[i, 0], ell @ a[:, 2], atol=1e-5)

    def loss(planes):
        area, flux, rate = _moments(*clipped(planes), (-0.25, 0.25, 0.5))
        return jnp.sum(area + flux[:, 0] - 0.7 * flux[:, 1] + 1.1 * flux[:, 2] + rate[:, 0])

    assert bool(jnp.all(jnp.isfinite(jax.grad(loss)(planes))))


def _one_plane(phi, offset, tilt, n_sub=1, width=1.0, split=None):
    """Fractions (B, K + 1) and normal tensors (B, 4) of B voxels, each cut by one plane.

    Voxel b's level is ``{n . p + offset + tilt (z - 1/2) < 0}`` with n = (cos phi, sin phi), p from
    the voxel center and z in [0, 1] (cells), painted on n_sub x n_sub sub-squares of a lateral
    square ``width`` cells wide, as material_weights does (with sub-samples, their moments are added
    up: _composite_samples). With ``split``, a fill level of the background's eps covers the voxel
    above that height: it changes nothing but the sub-intervals. The fractions must equal those
    without the tensor bit for bit.
    """
    phi, offset, tilt = (jnp.asarray(x, dtype=jnp.float32) for x in (phi, offset, tilt))
    normal = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)
    is_fill = (False,) if split is None else (False, True)
    shape = (phi.shape[0], 1, len(is_fill))
    lows = jnp.full(shape, -10.0) if split is None else jnp.full(shape, -10.0).at[..., 1].set(split)
    highs = jnp.full(shape, 10.0)
    normals = jnp.broadcast_to(normal[:, None, None], (*shape, 2))
    level_eps = np.asarray([[1.0, 0.0][: len(is_fill)]])
    cell = width / n_sub

    def lateral(z):
        return jnp.stack([tilt[:, None, None] / cell * (z - 0.5), jnp.zeros_like(z)][: len(is_fill)], axis=-1)

    samples, fractions = [], 0.0
    for a, b, kernel in _lateral_samples(n_sub, width):
        off = (offset + normal[:, 0] * a + normal[:, 1] * b) / cell  # at the sub-square's center
        offsets = jnp.stack([off, jnp.full_like(off, -1e6)][: len(is_fill)], axis=-1)[:, None]
        args = (offsets, normals, is_fill, lows, highs, lateral, 1)
        fractions = fractions + _composite(*args)
        samples.append((offsets, kernel))
    if n_sub == 1:
        f, t = _composite(*args, level_eps=level_eps, kernel=kernel, cell=cell)
    else:
        offsets, kernels = zip(*samples, strict=True)
        f, t = _composite_samples(
            jnp.stack(offsets), jnp.stack([normals] * n_sub**2), is_fill, lows, highs, lateral, 1, level_eps,
            kernels, cell,
        )
    np.testing.assert_array_equal(f, fractions)
    return np.asarray(f[:, 0, 0]) / n_sub**2, np.asarray(t[:, 0, 0, 0])


def test_composite_normal_of_one_planar_interface() -> None:
    """A voxel cut by one plane (slanted in the plane, offset, tilted by a sidewall) gets its exact normal.

    Also with sub-samples and a wider lateral square, and with a surface that splits the voxel at
    another height: the wall keeps one offset over its whole part (the mean, _level_walls), and the
    normal's drift between the parts is that of the wall at each part's middle (the lag rate), so
    tilted walls stay exact too (they were 0.086 off before that).
    """
    rng = np.random.default_rng(5)
    num = 48
    phi = rng.uniform(0, 2 * np.pi, num)
    offset = rng.uniform(-0.6, 0.6, num)
    tilt = np.tan(np.radians(rng.uniform(-40, 40, num))) * (rng.uniform(size=num) < 0.75)

    def n2(t):
        return t[:, :3] / np.maximum(t[:, :3].sum(axis=-1, keepdims=True), 1e-30)

    def expected(tilt):
        return np.stack([np.cos(phi) ** 2, np.sin(phi) ** 2, tilt**2], axis=-1) / (1 + tilt[:, None] ** 2)

    for n_sub, width in ((1, 1.0), (2, 1.0), (1, 1.5), (2, 1.5)):
        f, t = _one_plane(phi, offset, tilt, n_sub, width)
        cut = np.minimum(f[:, 0], 1 - f[:, 0]) > 0.01
        assert cut.sum() > 25
        np.testing.assert_allclose(n2(t)[cut], expected(tilt)[cut], atol=1e-5, err_msg=f"{n_sub} {width}")
        assert float(t[cut, 3].max()) <= 1e-3 * float(t[cut, :3].sum(axis=-1).min())  # no fallback
    b30 = math.tan(math.radians(30.0)) * np.sign(rng.uniform(-1, 1, num))
    for split, n_sub in ((0.3, 1), (0.71, 1), (0.3, 2)):
        for slope in (0 * tilt, b30, tilt):
            f, t = _one_plane(phi, 0.5 * offset, slope, n_sub, split=split)
            cut = np.minimum(f[:, 0], 1 - f[:, 0]) > 0.01
            assert cut.sum() > 25
            np.testing.assert_allclose(n2(t)[cut], expected(slope)[cut], atol=1e-5, err_msg=f"{split} {n_sub}")


def _voxel(levels, eps, surface_slopes=None):
    """Squared normal (3,) and tensor (4,) of one voxel z in [0, 1] (cells) holding ``levels`` by priority.

    ``levels``: dicts of ``fill`` (default False), ``lo``, ``hi`` (cells) and, for polygon levels,
    ``offset``, ``phi`` and ``tilt`` (as in _one_plane); ``eps``: per level, then the background.
    """
    is_fill = tuple(lv.get("fill", False) for lv in levels)

    def get(key, default):
        return jnp.asarray([[[lv.get(key, default) for lv in levels]]], dtype=jnp.float32)

    phi, tilt = get("phi", 0.0), get("tilt", 0.0)
    e = np.asarray(eps, dtype=float)
    level_eps = ((e[:-1] - e[-1]) / (np.ptp(e) or 1.0))[None]
    slopes = None if surface_slopes is None else jnp.asarray(surface_slopes, dtype=jnp.float32)[None, None]

    def lateral(z):
        return tilt[:, :, None, :] * (z[..., None] - 0.5)

    normals = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)
    args = (get("offset", 0.0), normals, is_fill, get("lo", -10.0), get("hi", 10.0), lateral, 1)
    _, tensor = _composite(*args, level_eps=level_eps, surface_slopes=slopes)
    t = np.asarray(tensor[0, 0, 0, 0], dtype=float)
    return t[:3] / (t[:3].sum() + t[3] + 1e-30), t


def test_composite_normal_films_ties_continuity() -> None:
    """Films (centred or not) and surfaces are averaged along z, with no fallback; an eps tie has no
    tensor; the normal is continuous as a film vanishes and as a level's bottom crosses another's top;
    rough surfaces tilt the vertical part along their normals."""
    for lo, hi in ((0.25, 0.75), (0.1, 0.4)):
        n2, t = _voxel([{"fill": True, "lo": lo, "hi": hi}, {"fill": True}], [2.1, 12.0, 1.0])
        np.testing.assert_allclose(n2, [0.0, 0.0, 1.0], atol=1e-6)
        assert t[3] <= 1e-6 * t[2]
    n2, _ = _voxel([{"fill": True, "hi": 0.97}, {"fill": True}], [12.0, 2.1, 1.0])
    np.testing.assert_allclose(n2, [0.0, 0.0, 1.0], atol=1e-6)
    _, t = _voxel([{"offset": 0.1, "phi": 0.7}], [1.0, 1.0])
    assert not t.any()

    wall = {"offset": 0.1, "phi": 0.0}
    n2_wall, _ = _voxel([wall, {"fill": True}], [12.0, 2.1, 1.0])
    np.testing.assert_allclose(n2_wall, [1.0, 0.0, 0.0], atol=1e-6)
    n2, _ = _voxel([wall, {"fill": True, "lo": 0.3, "hi": 0.3001}, {"fill": True}], [12.0, 4.0, 2.1, 1.0])
    assert n2[2] < 1e-3

    reference = None
    for dz in (-1e-6, 0.0, 1e-6):  # the core's bottom on the box's top
        core = {"offset": 0.1, "phi": 0.4, "tilt": 0.2, "lo": 0.4 + dz, "hi": 5.0}
        n2, _ = _voxel([core, {"fill": True, "hi": 0.4}, {"fill": True}], [12.0, 2.1, 1.0, 1.0])
        reference = n2 if reference is None else reference
        np.testing.assert_allclose(n2, reference, atol=1e-4)

    for g in (0.3, -1.2):  # a rough top with slope g along x: slopes of lo_0, lo_1, hi_0, hi_1
        slopes = np.zeros((4, 2))
        slopes[2, 0] = g
        n2, _ = _voxel([{"fill": True, "hi": 0.4}, {"fill": True}], [12.0, 2.1, 1.0], surface_slopes=slopes)
        np.testing.assert_allclose(n2, np.asarray([g**2, 0.0, 1.0]) / (1 + g**2), atol=1e-6)


def _mean_eps(names, w):
    """<1/eps> and <eps> (numpy) of material fractions w, with eps 1 for every material but si (12)."""
    e = np.asarray([12.0 if m == "si" else 1.0 for m in names])
    w = np.asarray(w, dtype=float)
    return np.einsum("m,mxyz->xyz", 1 / e, w), np.einsum("m,mxyz->xyz", e, w)


@pytest.mark.parametrize("sidewall", [0.0, 15.0])
def test_kottke_normal_of_a_planar_wall(sidewall: float) -> None:
    """Voxels cut only by a long strip's wall (17 deg off the grid, leaning by the sidewall angle) get
    the Kottke average along its exact normal ``(-sin phi, cos phi, tan(angle))``, with each E
    component sampled at its own Yee position."""
    angle = math.radians(17.0)
    u, v = np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])
    origin = np.array([0.31, -0.22])
    strip = [tuple(origin + a * u + b * v) for a, b in ((0, 0), (2.0, 0), (2.0, 0.7), (0, 0.7))]
    objects, arrays, params = _placed(
        _polygons(strip), _core(sidewall_angle=sidewall, width_to_z=0.5), CORE_ONLY, (2.4, 1.6, 0.3),
        center=(1.16, 0.41), z_center=0.11,
    )
    d = objects["s"]
    inv = np.asarray(_box(_apply(objects, arrays, params), d, "inv_permittivities"))
    nx, ny, nz = d.grid_shape
    tan = math.tan(math.radians(sidewall))
    n2 = np.array([v[0] ** 2, v[1] ** 2, tan**2]) / (1 + tan**2)
    x = d.center[0] + (np.arange(nx) + 0.5 - nx / 2) * RES
    y = d.center[1] + (np.arange(ny) + 0.5 - ny / 2) * RES
    along = (x[:, None] - origin[0]) * u[0] + (y[None, :] - origin[1]) * u[1]  # um along the strip
    z0 = d.z_center - nz * RES / 2
    surfaces = [-z0 / RES, (0.22 - z0) / RES]  # the core's bottom and top (cells)
    checked = 0
    for c, shift in enumerate(_YEE_E_SHIFTS):
        names, w = d.material_weights(params["s"], shift=shift)
        si = np.asarray(w[names.index("si")])
        clear = [not any(k + shift[2] < s < k + 1 + shift[2] for s in surfaces) for k in range(nz)]
        cut = (si > 0.05) & (si < 0.95) & ((along > 0.3) & (along < 1.7))[:, :, None] & np.asarray(clear)
        mean_inv, mean_eps = _mean_eps(names, w)
        expected = n2[c] * mean_inv + (1 - n2[c]) / mean_eps
        np.testing.assert_allclose(inv[c][cut], expected[cut], rtol=1e-4, err_msg=f"component {c}")
        checked += cut.sum()
    assert checked > 500


def test_kottke_normal_ignores_neighbouring_interfaces() -> None:
    """A rib: the wall voxel above the slab top is averaged along x only, the slab top beside the wall
    along z only. One-sided differences of the mean eps across neighbouring voxels mix the two."""
    c = gf.Component()
    c.add_polygon([(0.313, -0.6), (1.5, -0.6), (1.5, 0.6), (0.313, 0.6)], layer=WG)  # wall off the grid
    c.add_polygon([(-0.5, -0.8), (1.9, -0.8), (1.9, 0.8), (-0.5, 0.8)], layer=SLAB)
    ls = LayerStack(layers={
        "core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.09, material="si", mesh_order=2),
    })
    objects, arrays, params = _placed(
        c, ls, CORE_ONLY, (1.2, 1.0, 0.3), center=(0.4, 0.0), z_center=0.11, yee_staggered=False
    )
    d = objects["s"]
    inv = np.asarray(_box(_apply(objects, arrays, params), d, "inv_permittivities"))
    names, w = d.material_weights(params["s"])
    si = np.asarray(w[names.index("si")])
    mean_inv, mean_eps = _mean_eps(names, w)
    wall = cell(d, 0.31, 0.0, 0.11)  # x [0.30, 0.32] holds the wall, z [0.10, 0.12] is above the slab
    top = cell(d, 0.29, 0.0, 0.09)  # x [0.28, 0.30] is beside the wall, z [0.08, 0.10] holds the slab top
    assert float(si[wall]) == pytest.approx(0.35, abs=1e-4)
    assert float(si[top]) == pytest.approx(0.5, abs=1e-4)
    for voxel, normal in ((wall, 0), (top, 2)):
        for a in range(3):
            expected = mean_inv[voxel] if a == normal else 1 / mean_eps[voxel]
            assert float(inv[(a, *voxel)]) == pytest.approx(float(expected), rel=1e-4), (voxel, a)
    fd = np.asarray(_normal_squared(jnp.asarray(mean_eps, dtype=jnp.float32)))
    assert float(fd[(0, *wall)]) < 0.95 and float(fd[(2, *top)]) < 0.95


def test_kottke_normal_at_a_corner_is_continuous() -> None:
    """Inside a convex corner, one half-plane per level takes the nearer edge, so its normal jumps from
    x to y as the voxel center crosses the corner's bisector; the corner model's normal comes from
    both edges in the voxel (symmetric on the bisector) and moves continuously."""
    def rectangle(dx):
        """Corner (0.296 + dx, 0.216) um, near the voxel center (0.29, 0.21): 0.3 cells from both edges."""
        return _polygons([(-0.3, -0.2), (0.296 + dx, -0.2), (0.296 + dx, 0.216), (-0.3, 0.216)])

    def inv_near_the_corner(lines):
        """inv_permittivities (3,) of the voxel for the corner 2e-4 cells left and right of the bisector."""
        objects, arrays, params = _placed(
            rectangle(0.0), _core(), CORE_ONLY, (1.0, 0.8, 0.3), center=(0.0, 0.0), z_center=0.11,
            yee_staggered=False, corner_lines=lines,
        )
        d = objects["s"]

        @jax.jit
        def inv(p):
            return _box(_apply(objects, arrays, {**params, "s": p}), d, "inv_permittivities")

        voxel = (slice(None), *cell(d, 0.29, 0.21, 0.11))
        return [np.asarray(inv({**params["s"], **d.pack(rectangle(dx))})[voxel]) for dx in (-4e-6, 4e-6)]

    for lines in (2, 0):
        before, after = inv_near_the_corner(lines)
        if lines:
            np.testing.assert_allclose(before, after, atol=1e-3)
            assert float(before[0]) == pytest.approx(float(before[1]), abs=1e-3)
            assert float(before[0]) > float(before[2]) + 0.1  # a mix of x and y, not z
        else:  # harmonic along x, then along y
            assert float(before[0]) > float(after[0]) + 0.1 and float(after[1]) > float(before[1]) + 0.1


def test_subpixel_gradients_match_finite_differences() -> None:
    """A fixed random weighting of the subpixel inverse permittivity: its gradients with respect to the
    waveguide width and the core's thickness, sidewall angle and zmin match central differences.

    The voxels that an interface enters or leaves within a step are left out of the weighting: an
    inverse permittivity's derivative jumps there, and the sloped walls, at a different position in
    every voxel layer, always come close to a voxel face of some component's Yee grid.
    """
    objects, arrays, params = place(build(z_range=(-0.1, 0.3), padding=0.1))
    d = objects["stack"]
    names = ("thickness/core", "sidewall_angle/core", "zmin/core")
    si = d.material_weights(params["stack"])[0].index("si")

    def stack_params(x):
        return {**params["stack"], **d.pack(circuit(width=x[0])), **dict(zip(names, x[1:], strict=True))}

    @jax.jit
    def inv(p):
        return _box(_apply(objects, arrays, {**params, "stack": p}), d, "inv_permittivities")

    @jax.jit
    def whole(p):
        """Voxels all or nothing silicon, at each component's Yee position."""
        f = jnp.stack([d.material_weights(p, shift=shift)[1][si] for shift in _YEE_E_SHIFTS])
        return (f < 1e-6) | (f > 1 - 1e-6)

    x0 = (0.5137, 0.2137, 10.0, 0.004)  # generic: the core's surfaces cut voxels, its corners are off Yee points
    steps = (1e-3, 5e-4, 0.1, 5e-4)
    points = []
    for a, h in enumerate(steps):
        for sign in (1, -1):
            x = list(x0)
            x[a] += sign * h
            points.append(x)
    reference = whole(stack_params(x0))
    keep = jnp.all(jnp.stack([whole(stack_params(x)) == reference for x in points]), axis=0)
    assert float(keep.mean()) > 0.99
    weight = jax.random.uniform(jax.random.PRNGKey(7), (3, *d.grid_shape)) * keep
    base = inv(stack_params(x0))

    def loss(*x):
        return jnp.sum(weight * (inv(stack_params(x)) - base))  # small terms: small float32 sums

    g = jax.grad(loss, argnums=(0, 1, 2, 3))(*x0)
    for a, h in enumerate(steps):
        fd = (loss(*points[2 * a]) - loss(*points[2 * a + 1])) / (2 * h)
        assert bool(jnp.isfinite(g[a])) and float(g[a]) != 0.0, a
        assert float(g[a]) == pytest.approx(float(fd), rel=5e-3), (a, float(g[a]), float(fd))


def test_subpixel_gradients_finite_at_ties() -> None:
    """Finite gradients of the subpixel average at degenerate points: corners on Yee points (width
    0.5), tied surfaces (the core, slab and cladding bottoms on the box top), a level of zero
    thickness on a voxel face, and an eps_inf tie (air and the Drude metal)."""
    objects, arrays, params = place(build(materials=DISP_MATERIALS, z_range=(-0.1, 0.52), padding=0.1))
    d = objects["stack"]
    assert "pad" in [lv.name for lv in d.levels]
    r = jax.random.uniform(jax.random.PRNGKey(8), (3, *d.grid_shape))

    def loss(p):
        return jnp.sum(r * _box(_apply(objects, arrays, {**params, "stack": p}), d, "inv_permittivities"))

    g = jax.jit(jax.grad(loss))({**params["stack"], "thickness/slab": 0.0})  # both slab surfaces on z = 0
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g))
    assert float(jnp.abs(g["thickness/core"])) > 0


def test_normal_tensor_gradients_finite_at_low_contrast_slivers() -> None:
    """Float32 gradients of the normal tensor are finite at corner slivers of levels of low contrast
    (normalized |de| 0.1 and 0.02: variances below 1e-9), where the ramp's ratio once had a derivative
    lam2^-2 that flushed to inf, and at a tiny jump between rough surfaces next to a voxel face."""
    rng = np.random.default_rng(0)
    num = 4096
    lows, highs = jnp.full((num, 1, 2), -10.0), jnp.full((num, 1, 2), 10.0)
    for de in (1.0, 0.1, 0.02):
        eps = np.asarray([1 + 11 * de, 12.0, 1.0])  # the second level, absent, sets the range
        phi = rng.uniform(0, 2 * np.pi, (num, 2))
        normals = jnp.asarray(np.stack([np.cos(phi), np.sin(phi)], axis=-1)[:, None], dtype=jnp.float32)
        corner = (np.abs(np.cos(phi[:, 0])) + np.abs(np.sin(phi[:, 0]))) / 2
        depth = 10 ** rng.uniform(-8, -1, num)  # cells: a sliver at a corner of the square
        offsets = jnp.asarray(np.stack([corner - depth, np.full(num, 5.0)], axis=-1)[:, None], dtype=jnp.float32)
        weights = jnp.asarray(rng.uniform(size=(num, 4)), dtype=jnp.float32)

        def loss(offsets, normals=normals, eps=eps, weights=weights):
            _, t = _composite(
                offsets, normals, (False, False), lows, highs, lambda z: jnp.zeros((*z.shape, 2)), 1,
                level_eps=((eps[:-1] - eps[-1]) / np.ptp(eps))[None],
            )
            return jnp.sum(weights * t[:, 0, 0, 0])

        g = jax.jit(jax.grad(loss))(offsets)
        assert bool(jnp.all(jnp.isfinite(g))), (de, int(jnp.sum(~jnp.isfinite(g))))

    def vertical(x):
        """Tensor of one voxel split at height x[0] (cells) into contrasts x[1] and 0, under a rough surface."""
        mean = jnp.stack([x[1], 0.0])[None, :, None]
        return _normal_tensor(
            jnp.stack([x[0], 1 - x[0]])[None], jnp.stack([0.0, x[0], 1.0])[None], jnp.zeros((1, 2, 1, 3)), mean,
            mean**2, jnp.zeros((1, 2, 1, 2)), 1.0, 4 / 9, jnp.asarray([[[0.5, 0.0]]]),
        )

    g = jax.jit(jax.grad(lambda x: jnp.sum(vertical(x))))(jnp.asarray([2e-7, 2e-7], dtype=jnp.float32))
    assert bool(jnp.all(jnp.isfinite(g)))


def _sub_squares(normals, offsets, eps, n_sub=2, width=1.0):
    """Fractions (B, K + 1) and tensors (B, 4) of B z-uniform voxels painted on n_sub x n_sub sub-squares.

    Level j of voxel b is ``{n_bj . p + offsets_bj < 0}`` (p in cells from the voxel center), by
    priority; ``eps``: per level, then the background; the sub-squares are ``width / n_sub`` cells wide.
    Also returns the levels' normalized contrasts.
    """
    normals, offsets = jnp.asarray(normals, dtype=jnp.float32), jnp.asarray(offsets, dtype=jnp.float32)
    num, k = offsets.shape
    e = np.asarray(eps, dtype=float)
    level_eps = ((e[:-1] - e[-1]) / np.ptp(e))[None]
    cell = width / n_sub
    samples = _lateral_samples(n_sub, width)
    f, t = _composite_samples(
        jnp.stack([((offsets + normals[..., 0] * a + normals[..., 1] * b) / cell)[:, None] for a, b, _ in samples]),
        jnp.broadcast_to(normals[:, None], (len(samples), num, 1, k, 2)),
        (False,) * k,
        jnp.full((num, 1, k), -10.0),
        jnp.full((num, 1, k), 10.0),
        lambda z: jnp.zeros((*z.shape, k)),
        1,
        level_eps,
        [kernel for *_, kernel in samples],
        cell,
    )
    return f[:, 0, 0] / len(samples), t[:, 0, 0, 0], level_eps[0]


def test_sub_squares_are_added_up_before_the_normal() -> None:
    """With sub-samples, the voxel's moments are added up over its sub-squares before the normal, so no
    variance is dropped (also not between the sub-squares): a plane on the face between two sub-squares
    (each uniform) gets its exact normal, and a line crossing that face beside another interface moves
    the normal continuously, with derivatives that match differences there. (Tensors per sub-square gave
    that plane none, and a vanishing sliver's whole edge turned its sub-square's variance.) Sub-squares
    that do not tile the voxel (gaps, overlaps) each have their own kernel: continuous across their sides."""
    def total_variance(f, contrasts):
        e = np.append(contrasts, 0.0)  # the background's contrast is 0
        return f @ e**2 - (f @ e) ** 2

    x0 = np.asarray([-1e-4, 0.0, 1e-4, 0.3137])  # the plane x = x0 (cells), si on the left
    f, t, contrasts = _sub_squares(np.broadcast_to([1.0, 0.0], (4, 1, 2)), -x0[:, None], [12.0, 1.0])
    np.testing.assert_allclose(f[:, 0], 0.5 + x0, atol=1e-6)
    np.testing.assert_allclose(t[:, 0], total_variance(f, contrasts), rtol=1e-5)  # all of it along x
    assert float(np.abs(t[:, 1:]).max()) < 1e-7

    # si {x < x0} beside a slab {y < 0.3} (eps 4): x0 crosses the face x = 0 between the sub-squares
    def two_levels(x0):
        normals = jnp.broadcast_to(jnp.asarray([[1.0, 0.0], [0.0, 1.0]]), (*x0.shape, 2, 2))
        offsets = jnp.stack([-x0, jnp.full_like(x0, -0.3)], axis=-1)
        return _sub_squares(normals, offsets, [12.0, 4.0, 1.0])

    x0 = jnp.asarray([-1e-4, 0.0, 1e-4], dtype=jnp.float32)
    f, t, contrasts = two_levels(x0)
    np.testing.assert_allclose(t.sum(axis=-1), total_variance(f, contrasts), rtol=1e-5)
    n2 = t[:, :3] / t[:, :3].sum(axis=-1, keepdims=True)
    assert 0.01 < float(n2[1, 1]) < 0.5  # the slab's edge counts too
    np.testing.assert_allclose(n2[0], n2[2], atol=1e-3)
    _, jvp = jax.jvp(lambda x: two_levels(x)[1], (x0,), (jnp.ones_like(x0),))
    h = 2e-3
    fd = (np.asarray(two_levels(x0 + h)[1]) - np.asarray(two_levels(x0 - h)[1])) / (2 * h)
    assert bool(jnp.all(jnp.isfinite(jvp)))
    np.testing.assert_allclose(jvp, fd, atol=2e-3 * float(np.abs(fd).max()))

    # sub-squares 0.4 cells wide (a gap between them) and 0.75 cells wide (overlapping): x0 crosses a
    # side of one of them, at the gap or inside the other one
    for width, side in ((0.8, 0.05), (1.5, 0.125)):
        x0 = jnp.asarray([side - 1e-4, side + 1e-4], dtype=jnp.float32)
        normals = jnp.broadcast_to(jnp.asarray([[1.0, 0.0], [0.0, 1.0]]), (2, 2, 2))
        f, t, contrasts = _sub_squares(normals, jnp.stack([-x0, jnp.full_like(x0, -0.3)], axis=-1), [12.0, 4.0, 1.0],
                                       width=width)
        np.testing.assert_allclose(t.sum(axis=-1), total_variance(f, contrasts), rtol=1e-5)
        n2 = t[:, :3] / t[:, :3].sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(n2[0], n2[1], atol=2e-3, err_msg=str(width))


def test_kottke_normal_with_sub_samples_on_their_faces() -> None:
    """xy_subsamples=2, with a rib wall exactly on the face between the sub-squares of its voxels (each
    sub-square uniform): the wall voxel above the slab top is averaged along x only, and the derivatives of
    the inverse permittivity with respect to the wall position and the core thickness match central
    differences in every voxel. (With tensors per sub-square the voxel lost the variance between them and
    fell back to differences of the mean eps across its neighbours, and its derivatives blew up.)"""
    ls = LayerStack(layers={
        "core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2137, material="si", mesh_order=1),
        "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.0937, material="si", mesh_order=2),
    })

    def rib(x):
        c = gf.Component()
        c.add_polygon([(x, -0.6), (1.5, -0.6), (1.5, 0.6), (x, 0.6)], layer=WG)
        c.add_polygon([(-0.5, -0.8), (1.9, -0.8), (1.9, 0.8), (-0.5, 0.8)], layer=SLAB)
        return c

    x0, t0 = 0.31, 0.2137  # the wall on voxel centers, the surfaces off voxel faces
    objects, arrays, params = _placed(
        rib(x0), ls, CORE_ONLY, (0.6, 0.2, 0.3), center=(0.4, 0.0), z_center=0.1137, yee_staggered=False,
        xy_subsamples=2,
    )
    d = objects["s"]
    packed = d.pack(rib(x0))
    key = next(k for k, v in packed.items() if np.isclose(np.asarray(v)[:, 0], x0).any())
    moving = jnp.asarray(np.isclose(np.asarray(packed[key])[:, 0], x0), dtype=jnp.float32)

    def stack_params(x, t):
        return {**params["s"], **packed, key: jnp.asarray(packed[key]).at[:, 0].add(moving * (x - x0)),
                "thickness/core": t}

    @jax.jit
    def inv(x, t):
        return _box(_apply(objects, arrays, {**params, "s": stack_params(x, t)}), d, "inv_permittivities")

    names, w = d.material_weights(stack_params(x0, t0))
    mean_inv, mean_eps = _mean_eps(names, w)
    wall = cell(d, x0, 0.0, 0.1137)  # z [0.1037, 0.1237] is above the slab top
    assert float(w[names.index("si")][wall]) == pytest.approx(0.5, abs=1e-5)
    base = inv(x0, t0)
    for a in range(3):
        expected = mean_inv[wall] if a == 0 else 1 / mean_eps[wall]
        assert float(base[(a, *wall)]) == pytest.approx(float(expected), rel=1e-4), a
    fd = np.asarray(_normal_squared(jnp.asarray(mean_eps, dtype=jnp.float32)))
    assert float(fd[(0, *wall)]) < 0.95

    h = 1e-4  # um: the wall stays inside its voxels, the top inside its voxel layer
    for tangent in ((1.0, 0.0), (0.0, 1.0)):
        _, jvp = jax.jvp(inv, (jnp.float32(x0), jnp.float32(t0)), tuple(jnp.float32(x) for x in tangent))
        up, down = (inv(x0 + s * tangent[0] * h, t0 + s * tangent[1] * h) for s in (1, -1))
        fd = (np.asarray(up) - np.asarray(down)) / (2 * h)
        assert bool(jnp.all(jnp.isfinite(jvp))), tangent
        np.testing.assert_allclose(jvp, fd, atol=1e-3 * float(np.abs(fd).max()), err_msg=str(tangent))


def test_kottke_normal_is_continuous_across_sub_squares() -> None:
    """xy_subsamples=2: as the core's edge crosses the face between the sub-squares of a voxel that also
    holds the slab's edge, the inverse permittivity moves continuously, with and without the corner model.
    (With tensors per sub-square it jumped by up to 80% there.)"""
    ls = LayerStack(layers={
        "core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.22, material="sio2", mesh_order=2),
    })

    def device(dx):
        """Core x < 0.29 + dx, slab y < 0.216 (um): 0.3 cells from the voxel center (0.29, 0.21)."""
        c = gf.Component()
        c.add_polygon([(-0.3, -0.3), (0.29 + dx, -0.3), (0.29 + dx, 0.3), (-0.3, 0.3)], layer=WG)
        c.add_polygon([(-0.4, -0.3), (0.4, -0.3), (0.4, 0.216), (-0.4, 0.216)], layer=SLAB)
        return c

    for lines in (2, 0):
        objects, arrays, params = _placed(
            device(0.0), ls, {k: MATERIALS[k] for k in ("air", "sio2", "si")}, (1.0, 0.8, 0.3), center=(0.0, 0.0),
            z_center=0.11, yee_staggered=False, corner_lines=lines, xy_subsamples=2, diagnostics=False,
        )
        d = objects["s"]

        @jax.jit
        def inv(p, objects=objects, arrays=arrays, params=params, d=d):
            return _box(_apply(objects, arrays, {**params, "s": p}), d, "inv_permittivities")

        voxel = (slice(None), *cell(d, 0.29, 0.21, 0.11))
        before, after = (
            np.asarray(inv({**params["s"], **d.pack(device(dx), check=False)})[voxel]) for dx in (-4e-6, 4e-6)
        )
        np.testing.assert_allclose(before, after, atol=1e-3, err_msg=f"corner_lines {lines}")
        assert float(before[0]) > float(before[2]) + 0.1  # mostly along x


def test_sub_samples_with_corner_columns() -> None:
    """xy_subsamples=2 with corner columns at degenerate corners (on voxel centers and corners, edges on
    voxel boundaries and on the faces between sub-squares, a corner shared by two levels): the fractions
    painted with the normal tensor (the sub-samples' corner passes in one loop, with traced kernel centers)
    equal material_weights, the tensor is finite, and so are the subpixel gradients."""
    c, ls, mats = _degenerate_corners()
    objects, arrays, params = _placed(
        c, ls, mats, (1.2, 0.8, 0.3), center=(0.5, 0.3), z_center=0.11, xy_subsamples=2, yee_staggered=False,
        diagnostics=False,
    )
    d = objects["s"]
    assert d.corner_columns > 0
    names, w = d.material_weights(params["s"])
    eps = np.asarray([[12.0 if m == "si" else 2.1 if m == "sio2" else 1.0] for m in names])
    _, w_n, tensor = d._weights_and_normals(params["s"], None, True, (0.0, 0.0, 0.0), eps)
    np.testing.assert_allclose(w_n, w, atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(tensor))) and float(tensor[0, :3].sum()) > 0
    r = jax.random.uniform(jax.random.PRNGKey(0), (3, *d.grid_shape))

    def loss(p):
        return jnp.sum(r * _box(_apply(objects, arrays, {**params, "s": p}), d, "inv_permittivities"))

    g = jax.jit(jax.grad(loss))(params["s"])
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g))
    assert float(jnp.abs(g["thickness/core"])) > 0


# ---------------------------------------------------------------- box, port extensions, volume, ports


def _waveguide(width: float = 0.5, clockwise: bool = False) -> gf.Component:
    """2 um waveguide drawn as one polygon in either orientation, with ports o1 / o2."""
    c = gf.Component()
    pts = [(0.0, -width / 2), (2.0, -width / 2), (2.0, width / 2), (0.0, width / 2)]
    c.add_polygon(pts[::-1] if clockwise else pts, layer=WG)
    c.add_port("o1", center=(0.0, 0.0), width=width, orientation=180, layer=WG)
    c.add_port("o2", center=(2.0, 0.0), width=width, orientation=0, layer=WG)
    return c


def _straight(rotation: float, width: float = 0.5) -> gf.Component:
    c = gf.Component()
    ref = c << gf.components.straight(length=2.0, width=width)
    ref.rotate(rotation)
    c.add_ports(ref.ports)
    return c


def _place_own_volume(device, extra=()):
    """Places a device in its own volume (get_simulation_volume) with extra (object, constraints) pairs."""
    config = fdtdx.SimulationConfig(
        time=10e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6),
        backend=jax.default_backend(),
        dtype=jnp.float32,
    )
    volume, constraints = device.get_simulation_volume()
    objs = [volume, device]
    for obj, cons in extra:
        objs.append(obj)
        constraints += cons
    objects, arrays, params, _, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=constraints, key=jax.random.PRNGKey(0)
    )
    return objects, arrays, {**params, **init_stack_params(objects)}


def _named(objects, params):
    names, w = objects["stack"].material_weights(params["stack"])
    return {n: np.asarray(w[i]) for i, n in enumerate(names)}


def test_box_from_included_layers_and_offsets() -> None:
    d = build(included_layers=["core"], boundary_offset=(0.1, -0.1, 0.2, 0.0, 0.08, -0.3))
    # bbox x [0, 2], y [-0.5, 0.7]; core z [0, 0.22]
    x0, x1, y0, y1, z0, z1 = -0.1, 2.1, -0.5, 0.9, -0.3, 0.3
    np.testing.assert_allclose(np.array(d.partial_real_shape) * 1e6, [x1 - x0, y1 - y0, z1 - z0], atol=1e-9)
    np.testing.assert_allclose(d.center, [(x0 + x1) / 2, (y0 + y1) / 2])
    assert d.z_center == pytest.approx((z0 + z1) / 2)
    assert {lv.name for lv in d.levels} == {"box", "core", "slab", "clad"}  # the pad (z 0.4) is outside
    assert d.averaging == "subpixel"  # the default


def test_box_errors() -> None:
    no_metal = {k: v for k, v in MATERIALS.items() if k != "metal"}
    build(materials=no_metal, included_layers=["core"])  # the pad is outside: no material needed
    with pytest.raises(KeyError, match="'metal' is not in materials"):
        build(materials=no_metal, included_layers=["pad"])
    with pytest.raises(KeyError, match="Unknown levels"):
        build(included_layers=["nope"])
    with pytest.raises(TypeError, match="z_center"):
        build(z_center=0.1)
    with pytest.raises(ValueError, match="Empty box"):
        build(included_layers=["core"], boundary_offset=(0, 0, 0, 0, -0.3, 0))
    with pytest.raises(KeyError, match="Unknown port"):
        build(_straight(0), included_layers=["core"], extend_ports=(("nope", 1.0),))
    with pytest.raises(ValueError, match="length > 0"):
        build(_straight(0), included_layers=["core"], extend_ports=(("o1", 0.0),))


def test_boundary_offset_shows_more_of_the_same_stack() -> None:
    """A bigger box keeps the voxels it had, extends the WAFER levels and adds levels it now reaches."""
    small = _named(*_place_own_volume(build(included_layers=["core"], boundary_offset=(0,) * 6))[::2])
    big = _named(*_place_own_volume(
        build(included_layers=["core"], boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.3, -0.2))
    )[::2])
    n = round(0.2 / RES)  # x [0, 2] -> [-0.2, 2.2], y [-0.5, 0.7] -> [-0.7, 0.9], z [0, 0.22] -> [-0.2, 0.52]
    for m, w in small.items():
        np.testing.assert_allclose(big[m][n:-n, n:-n, n : n + w.shape[2]], w, atol=1e-5, err_msg=m)
    assert set(big) - set(small) == {"metal"}  # the pad level, outside the core's z range
    def at(x, y, z):  # voxel of the big box containing (x, y, z) um
        return int((x + 0.2) / RES), int((y + 0.7) / RES), int((z + 0.2) / RES)

    assert float(big["sio2"][:, :, at(0, 0, -0.1)[2]].min()) == pytest.approx(1.0)  # buried oxide (WAFER)
    assert float(big["sio2"][at(-0.19, -0.69, 0.35)]) == pytest.approx(1.0)  # cladding beyond the bbox
    assert float(big["air"][at(-0.19, -0.69, 0.51)]) == pytest.approx(1.0)  # above it: background
    assert float(big["metal"][at(0.41, 0.66, 0.45)]) == pytest.approx(1.0)  # the pad


@pytest.mark.parametrize("clockwise", [False, True])
def test_port_extensions_continue_the_waveguide(clockwise: bool) -> None:
    """Straight waveguides out of the ports: same sidewalls, no seam, they follow the ports."""
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "clad")})  # core: 10 deg sidewall
    device = build(
        _waveguide(clockwise=clockwise), ls, included_layers=["box", "clad"],
        extend_ports=(("o1", 0.3), ("o2", 0.5)), boundary_offset=(0.1, 0.0, 0.5, -0.5, 0.0, 0.0),
    )
    np.testing.assert_allclose(np.array(device.partial_real_shape[0]) * 1e6, 2.8 + 0.1)  # x [-0.3, 2.6]
    objects, _, params = place(device)
    d = objects["stack"]
    w = _named(objects, params)
    i_mid = cell(d, 1.01, 0, 0)[0]
    for x in (-0.25, -0.01, 0.01, 1.99, 2.01, 2.45):  # extensions, both junctions
        i = cell(d, x, 0, 0)[0]
        for m in w:
            np.testing.assert_allclose(w[m][i], w[m][i_mid], atol=1e-5, err_msg=f"{m} at x={x}")
    # the far end is a sidewall too: it leans back towards the top (width_to_z = 0.5)
    i_end, (_, j, k_low) = cell(d, 2.49, 0, 0)[0], cell(d, 0, 0, 0.01)
    k_high = cell(d, 0, 0, 0.21)[2]
    assert float(w["si"][i_end, j, k_high]) < float(w["si"][i_end, j, k_low]) - 0.3
    assert float(w["si"][cell(d, 2.55, 0, 0)[0]].max()) == 0.0  # nothing beyond it

    si = list(w).index("si")

    def si_volume(width):
        p = {**params["stack"], **d.pack(_waveguide(width, clockwise))}
        return d.material_weights(p)[1][si].sum()

    w0 = 0.5137  # off the grid: a unique derivative
    g = jax.grad(si_volume)(w0)
    h = 1e-3
    fd = (si_volume(w0 + h) - si_volume(w0 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=5e-3)
    # the 0.8 um of extensions follow the port width: (2 + 0.8) um x 0.22 um / RES^3
    assert float(g) * RES**3 == pytest.approx(2.8 * 0.22, rel=0.02)


def _rib(slab_width: float = 3.0, clockwise: bool = False) -> gf.Component:
    """2 um rib waveguide: a 0.5 um core on WG and a wider slab, both ending at the ports."""
    c = gf.Component()
    for layer, width in ((WG, 0.5), (SLAB, slab_width)):
        pts = [(0.0, -width / 2), (2.0, -width / 2), (2.0, width / 2), (0.0, width / 2)]
        c.add_polygon(pts[::-1] if clockwise else pts, layer=layer)
    c.add_port("o1", center=(0.0, 0.0), width=0.5, orientation=180, layer=WG)
    c.add_port("o2", center=(2.0, 0.0), width=0.5, orientation=0, layer=WG)
    return c


def _rib_device(extend_ports, slab_width: float = 3.0, clockwise: bool = False) -> StackDevice:
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "slab", "clad")})
    return build(
        _rib(slab_width, clockwise), ls, included_layers=["box", "clad"],
        extend_ports=extend_ports, boundary_offset=(0.1, 0.0, 0.3, -0.3, 0.0, 0.0),
    )


@pytest.mark.parametrize("clockwise", [False, True])
def test_port_extensions_continue_the_whole_cross_section(clockwise: bool) -> None:
    """A rib waveguide continues with its slab, which follows the slab width."""
    device = _rib_device((("o1", 0.3), ("o2", 0.5)), clockwise=clockwise)
    objects, _, params = _place_own_volume(device)
    d = objects["stack"]
    w = _named(objects, params)
    i_mid = cell(d, 1.01, 0, 0)[0]
    for x in (-0.25, -0.01, 0.01, 1.99, 2.01, 2.45):  # extensions, both junctions
        i = cell(d, x, 0, 0)[0]
        for m in w:
            np.testing.assert_allclose(w[m][i], w[m][i_mid], atol=1e-5, err_msg=f"{m} at x={x}")
    assert float(w["si"][cell(d, -0.25, 1.0, 0.04)]) == pytest.approx(1.0)  # slab beyond the core
    assert float(w["si"][cell(d, 2.55, 0, 0)[0]].max()) == 0.0  # nothing beyond the extensions

    si = list(w).index("si")

    def si_volume(slab_width):
        p = {**params["stack"], **d.pack(_rib(slab_width, clockwise))}
        return d.material_weights(p)[1][si].sum()

    w0 = 3.0137  # off the grid: a unique derivative
    g = jax.grad(si_volume)(w0)
    h = 1e-3
    fd = (si_volume(w0 + h) - si_volume(w0 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=5e-3)
    # the slab's 0.8 um of extensions follow its width: (2 + 0.8) um x 0.08 um / RES^3
    assert float(g) * RES**3 == pytest.approx(2.8 * 0.08, rel=0.02)


def test_port_extensions_on_listed_layers_only() -> None:
    device = _rib_device((("o1", 0.3, [WG]),))
    layers = {k for (k, _), ext in zip(device.source_layers, device.source_extensions, strict=True) if ext}
    assert layers == {"vertices/WG"}
    objects, _, params = _place_own_volume(device)
    d = objects["stack"]
    w = _named(objects, params)
    assert float(w["si"][cell(d, -0.25, 0.0, 0.1)]) == pytest.approx(1.0)  # the core continues
    assert float(w["si"][cell(d, -0.25, 1.0, 0.04)]) == 0.0  # the slab does not


def _port(orientation: float = 0.0, width: float = 0.5) -> _Port:
    return _Port(name="o", x=0.0, y=0.0, orientation=orientation, width=width, layer=0)


def _layer(*polys) -> tuple[np.ndarray, list[int]]:
    return np.concatenate([np.asarray(p, dtype=float) for p in polys]), [len(p) for p in polys]


def test_face_intervals_merge_seams_and_skip_other_waveguides() -> None:
    """Abutting polygons (either orientation) are one stretch; a guide beside the port is not extended."""
    upper = [(-1, 0), (0, 0), (0, 1), (-1, 1)]
    lower = [(-1, -1), (-1, 0), (0, 0), (0, -1)]  # clockwise
    beside = [(-1, 2), (0, 2), (0, 3), (-1, 3)]
    v, sizes = _layer(upper, lower, beside)
    ends = _face_intervals(_port(), v, sizes)
    assert len(ends) == 1
    assert all(e[0] == e[1] for e in ends[0])  # both corners are vertices on the face
    rect = _extension_rect(ends[0], v, (0.0, 0.0), 0.5, 0.0, 0.4, np)
    np.testing.assert_allclose(rect, [(0, -1), (0.4, -1), (0.4, 1), (0, 1)], atol=1e-12)
    # the port on the other face (pointing to -x) sees the same stretch, mirrored
    ends = _face_intervals(_Port(name="o", x=-1.0, y=0.0, orientation=180.0, width=0.5, layer=0), v, sizes)
    rect = _extension_rect(ends[0], v, (-1.0, 0.0), 0.5, 180.0, 0.4, np)
    np.testing.assert_allclose(rect, [(-1, 1), (-1.4, 1), (-1.4, -1), (-1, -1)], atol=1e-12)


def test_face_intervals_on_edges_follow_the_vertices() -> None:
    """A polygon crossing the face (no vertex there) is cut where its edges cross it."""
    v, sizes = _layer([(-1.0, -1.0), (1.0, 0.0), (-1.0, 1.0)])  # tip beyond the face x = 0
    ends = _face_intervals(_port(), v, sizes)
    assert len(ends) == 1 and all(e[0] != e[1] for e in ends[0])
    rect = _extension_rect(ends[0], v, (0.0, 0.0), 0.5, 0.0, 0.4, np)
    np.testing.assert_allclose(rect, [(0, -0.5), (0.4, -0.5), (0.4, 0.5), (0, 0.5)], atol=1e-12)

    def corners(v, center):
        return _extension_rect(ends[0], v, center, 0.5, 0.0, 0.4, jnp)

    jv, jc = jax.jacfwd(corners, argnums=(0, 1))(jnp.asarray(v), jnp.zeros(2))
    h = 1e-6
    for k in range(v.size):
        dv = np.zeros(v.size)
        dv[k] = h
        dv = dv.reshape(v.shape)
        fd = (_extension_rect(ends[0], v + dv, (0.0, 0.0), 0.5, 0.0, 0.4, np)
              - _extension_rect(ends[0], v - dv, (0.0, 0.0), 0.5, 0.0, 0.4, np)) / (2 * h)
        np.testing.assert_allclose(np.asarray(jv).reshape(4, 2, -1)[..., k], fd, atol=1e-4)
    # moving the port along its normal slides the corners along the edges
    np.testing.assert_allclose(np.asarray(jc)[:, :, 0], [(1, 0.5), (1, 0.5), (1, -0.5), (1, -0.5)], atol=1e-5)


def test_get_simulation_volume_fits_the_device() -> None:
    objects, arrays, _ = _place_own_volume(build())
    assert objects["volume"].grid_shape == objects["stack"].grid_shape
    assert arrays.inv_permittivities.shape[0] == 3  # subpixel averaging by default


@pytest.mark.parametrize("rotation", [0, 90])
def test_sources_and_detectors_on_ports(rotation: float) -> None:
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "clad")})
    along = rotation == 0
    d = StackDevice.from_layer_stack(
        _straight(rotation), ls, MATERIALS, background="air", name="stack", included_layers=["core"],
        extend_ports=(("o1", 0.4), ("o2", 0.4)),
        boundary_offset=((0, 0, 0.6, -0.6) if along else (0.6, -0.6, 0, 0)) + (0.4, -0.3),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pol = {"fixed_E_polarization_vector": (0.0, 0.0, 1.0)}  # transverse to both port axes
    made = [
        d.create_mode_plane_source_at("o1", (2, 3), wave_character=wave),
        d.create_gaussian_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=0.1, **pol),
        d.create_uniform_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=0.2, **pol),
        d.create_hard_constant_amplitude_plane_source_at(
            "o1", (2, 3), wave_character=wave, port_offset=0.3, **pol
        ),
        d.create_point_dipole_source_at("o1", wave_character=wave, polarization="y", port_offset=0.4),
        d.create_mode_overlap_detector_at("o2", (2, 3), wave_characters=(wave,)),
        d.create_poynting_flux_detector_at("o2", (2, 3), port_offset=0.1),
        d.create_phasor_detector_at("o2", (2, 3), wave_characters=(wave,), port_offset=0.2),
        d.create_field_detector_at("o2", (2, 3), port_offset=0.3),
        d.create_energy_detector_at("o2", (2, 3), port_offset=0.4),
        d.create_diffractive_detector_at("o2", (2, 3), frequencies=(wave.get_frequency(),), port_offset=0.5),
    ]
    objects, _, _ = _place_own_volume(d, made)
    dev = objects["stack"]
    axis = 0 if along else 1
    ports = {p.name: p for p in dev.ports}

    def center_um(obj):  # center of a placed object in stack coordinates
        s = obj.grid_slice_tuple
        mid = [0.5 * (a + b) for a, b in s]
        return [
            dev.center[0] + (mid[0] - dev.grid_shape[0] / 2) * RES,
            dev.center[1] + (mid[1] - dev.grid_shape[1] / 2) * RES,
            dev.z_center + (mid[2] - dev.grid_shape[2] / 2) * RES,
        ]

    for obj, _ in made:
        o = objects[obj.name]
        port = ports[obj.name[:2]]
        expected = [port.x, port.y, 0.11]  # on the port, in z at the core middle
        got = center_um(o)
        assert abs(got[1 - axis] - expected[1 - axis]) <= RES and abs(got[2] - expected[2]) <= RES
        if "mode" in obj.name:  # the others are offset along the port normal
            assert abs(got[axis] - expected[axis]) <= RES
        if "dipole" in obj.name:
            assert o.grid_shape == (1, 1, 1)
            continue
        assert o.grid_shape[axis] == 1
        assert o.grid_shape[1 - axis] == round(2 * 0.5 / RES)
        assert o.grid_shape[2] == round(3 * 0.22 / RES)
    # o1 points out of the component towards -axis, o2 towards +axis: both launch / measure towards +
    assert objects["o1_mode_source"].direction == "+"
    assert objects["o2_mode_detector"].direction == "+"
    assert objects["o2_flux_detector"].direction == "+"
    assert objects["o2_flux_detector"].fixed_propagation_axis == axis
    out = d.create_mode_overlap_detector_at("o1", (2, 3), wave_characters=(wave,), name="reflection")[0]
    assert out.direction == "-"  # leaving through o1: towards -axis
    # port_offset > 0 moves along the injection direction (into the component), < 0 into the extension
    for port_offset in (0.2, -0.2):
        src, cons = d.create_uniform_plane_source_at(
            "o1", (2, 3), wave_character=wave, port_offset=port_offset, name="s", **pol
        )
        placed, _, _ = _place_own_volume(d, [(src, cons)])
        p = ports["o1"]
        inward = -np.array([np.cos(np.radians(p.orientation)), np.sin(np.radians(p.orientation))])
        got = center_um(placed["s"])[:2]
        np.testing.assert_allclose(got, np.array([p.x, p.y]) + port_offset * inward, atol=RES)


def test_port_errors() -> None:
    d = StackDevice.from_layer_stack(
        _straight(0), stack(), MATERIALS, background="air", name="stack", included_layers=["core"]
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    with pytest.raises(KeyError, match="Unknown port"):
        d.create_mode_plane_source_at("nope", (2, 3), wave_character=wave)
    with pytest.raises(KeyError, match="Level 'nope'"):
        d.create_mode_plane_source_at("o1", (2, 3), wave_character=wave, level="nope")
    with pytest.raises(ValueError, match="direction"):
        d.create_mode_plane_source_at("o1", (2, 3), wave_character=wave, direction="+")
    src, _ = d.create_mode_plane_source_at("o1", (2, 3), wave_character=wave, level="clad")
    assert src.partial_real_shape[2] == pytest.approx(3 * 0.5e-6)  # the cladding's height


def test_levels_without_polygons_need_no_material() -> None:
    """Levels whose layers have no polygons are skipped; fill (WAFER) levels always need a material."""
    ls = gf.gpdk.LAYER_STACK
    c = gf.components.straight(length=2.0)
    mats = {"si": MATERIALS["si"], "sio2": MATERIALS["sio2"], "air": MATERIALS["air"]}
    # z [-0.5, 2.0] reaches the etch, slab, nitride, ge, via, metal and heater levels: no polygons there
    d = StackDevice.from_layer_stack(
        c, ls, mats, background="air", included_layers=["core"], boundary_offset=(0, 0, 0, 0, 1.78, -0.5)
    )
    assert {lv.name for lv in d.levels} == {"box", "core", "clad"}
    with pytest.raises(KeyError, match="Level 'box'"):  # WAFER level: fills the plane without polygons
        StackDevice.from_layer_stack(
            c, ls, {k: v for k, v in mats.items() if k != "sio2"}, background="air",
            included_layers=["core"], boundary_offset=(0, 0, 0, 0, 0, -0.5),
        )


def test_sizing_sums_one_signed_sequence() -> None:
    wg = LogicalLayer(layer=WG)
    assert _sizing(wg) == 0.0
    assert _sizing(wg.sized(50).sized(30)) == pytest.approx(0.08)  # dbu = 1 nm
    assert _sizing(wg.sized((-20, -30))) == pytest.approx(-0.05)
    assert _sizing((wg - LogicalLayer(layer=ETCH)).sized(40, mode=3)) == pytest.approx(0.04)


@pytest.mark.parametrize(
    "sized, match",
    [
        (lambda la: la.sized((-100, 100)), "both grows and shrinks"),  # opening
        (lambda la: la.sized(100).sized(-100), "both grows and shrinks"),  # closing
        (lambda la: la.sized(100, 50), "Anisotropic"),
        (lambda la: la.sized((50, 50), mode=(2, 3)), "mixes modes"),
    ],
)
def test_sizing_rejects_what_a_sum_cannot_draw(sized, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _sizing(sized(LogicalLayer(layer=WG)))


def test_from_layer_stack_rejects_an_opening() -> None:
    ls = stack()
    core = ls.layers["core"]
    ls.layers["core"] = core.model_copy(update={"layer": LogicalLayer(layer=WG).sized((-100, 100))})
    with pytest.raises(NotImplementedError, match="both grows and shrinks"):
        build(ls=ls)


# ---------------------------------------------------------------- topology checks


def _unplaced(component, **kwargs) -> StackDevice:
    """A core-only device for the component, not placed (enough for pack and the topology checks)."""
    return StackDevice.from_layer_stack(
        component, _core(), CORE_ONLY, background="air", name="s", included_layers=["core"], **kwargs
    )


def _mmi_and_taper(dx: float = 0.0, scale: float = 1.0) -> gf.Component:
    """JUNCTIONS' taper on the side of an MMI, the taper moved by dx along x and its width scaled."""
    mmi, taper = JUNCTIONS["taper on MMI"][0]
    return _polygons(mmi, [(x + dx, y * scale) for x, y in taper])


def test_a_taper_moved_off_an_mmi_side() -> None:
    """With the T-junction frozen, the MMI side would bulge through the moved taper's corners and fill the
    gap: pack raises, and retopologize gives the device of the moved taper, with the gap."""
    size, center = (4.4, 1.8, 0.3), (2.0, 0.0)
    objects, _, params = _placed(_mmi_and_taper(), _core(), CORE_ONLY, size, center, z_center=0.11)
    d = objects["s"]
    d.pack(_mmi_and_taper(scale=0.8))  # the taper's end slides along the side: the topology holds
    moved = _mmi_and_taper(dx=0.1)
    with pytest.raises(
        ValueError,
        match=r"vertices/WG: vertex \d+ of polygon 1 was inside edge \d+ -> \d+ of polygon 0 \(a T-junction\) "
        r"and is now 0\.1 away",
    ):
        d.pack(moved)
    stale_params = {**params["s"], **d.pack(moved, check=False)}
    with pytest.raises(ValueError, match="T-junction"):
        d.check_topology(stale_params)
    si = d.material_weights(params["s"])[0].index("si")
    stale = np.asarray(d.material_weights(stale_params)[1][si])

    fixed_device = d.retopologize(moved)  # placed like d
    fixed_device.check_topology(moved)
    fixed = np.asarray(fixed_device.material_weights({**params["s"], **fixed_device.pack(moved)})[1][si])
    _, fresh = _weights_of(moved, _core(), CORE_ONLY, size, center, z_center=0.11)  # float32 vertices
    np.testing.assert_allclose(fixed, fresh["si"], atol=2e-5)
    gap = cell(d, 2.05, 0.0, 0.11)
    assert float(fixed[gap]) == 0.0
    assert float(stale[gap]) == pytest.approx(1.0)


def _squares(gap: float, dy: float = 0.0) -> gf.Component:
    """Two 1 um squares abutting at x = 1, the second moved by gap along x, both moved by dy along y."""
    return _polygons(
        [(0, dy), (1, dy), (1, 1 + dy), (0, 1 + dy)],
        [(1 + gap, dy), (2 + gap, dy), (2 + gap, 1 + dy), (1 + gap, 1 + dy)],
    )


def test_abutting_polygons_moved_apart() -> None:
    """With the seam frozen, the edges between them would stay out of the distance and the gap filled."""
    d = _unplaced(_squares(0.0))
    d.pack(_squares(0.0, dy=0.3))  # moving together keeps the seam
    d.check_topology(d.init_params())  # float32 round-off
    with pytest.raises(ValueError, match=r"vertices/WG: the seam between .* opened by 0\.05"):
        d.pack(_squares(0.05))
    with pytest.raises(ValueError, match="a new seam"):
        _unplaced(_squares(0.05)).pack(_squares(0.0))
    # vertices optimized directly: checked and retopologized from the parameters
    apart = d.pack(_squares(0.05), check=False)
    with pytest.raises(ValueError, match="opened"):
        d.check_topology(apart)
    d.retopologize(apart).check_topology(apart)
    with pytest.raises(ValueError, match="No vertices"):
        d.check_topology({"zmin/core": 0.0})


def test_polygons_listed_in_another_order() -> None:
    """Polygons of the same size listed the other way round would swap their topology; identical polygons
    moving far (a stretched grating) are not a reordering."""
    a = [(0, 0), (1, 0), (1, 0.5), (0, 0.5)]
    b = [(0, 1), (2, 1), (2, 1.4), (0, 1.4)]
    d = _unplaced(_polygons(a, b))
    with pytest.raises(ValueError, match=r"vertices/WG: polygons 0, 1 \(4 vertices each\) look reordered"):
        d.pack(_polygons(b, a))
    d.pack(_polygons(b, a), check=False)
    swapped = d.retopologize(_polygons(b, a))
    swapped.pack(_polygons(b, a))
    np.testing.assert_allclose(swapped.source_vertices[0], np.concatenate([b, a]))

    def grating(period: float) -> gf.Component:
        x = [k * period for k in range(12)]
        return _polygons(*([(x0, 0), (x0 + 0.2, 0), (x0 + 0.2, 1), (x0, 1)] for x0 in x))

    _unplaced(grating(0.5)).pack(grating(0.62))  # the last tooth moves 2.6 periods


@pytest.mark.parametrize("name", list(JUNCTIONS))
def test_moves_that_keep_the_topology_pass(name: str) -> None:
    """Rigid moves keep every junction; traced components are never checked, so a reordering under
    jax.grad passes (the check costs nothing in an objective)."""
    pieces, _ = JUNCTIONS[name]
    d = _unplaced(_polygons(*pieces))
    d.check_topology(d.init_params())  # float32 round-off
    d.pack(_polygons(*(_rotated(p, 25.0, np.array([0.3, -0.2])) for p in pieces)))
    reordered = pieces[::-1]
    with pytest.raises(ValueError, match="look reordered"):
        d.pack(_polygons(*reordered))

    def total(shift):
        return jnp.sum(d.pack(_polygons(*(jnp.asarray(p, dtype=float) + shift for p in reordered)))["vertices/WG"])

    assert float(jax.grad(total)(0.0)) == pytest.approx(2 * sum(len(p) for p in pieces))


def _waveguide_to(end: float, port_x: float, width: float = 0.5) -> gf.Component:
    """A waveguide on WG from x = 0 to end, with its output port o2 at port_x."""
    c = gf.Component()
    c.add_polygon([(0.0, -width / 2), (end, -width / 2), (end, width / 2), (0.0, width / 2)], layer=WG)
    c.add_port("o1", center=(0.0, 0.0), width=width, orientation=180, layer=WG)
    c.add_port("o2", center=(port_x, 0.0), width=width, orientation=0, layer=WG)
    return c


def _rib_to(slab_end: float) -> gf.Component:
    """A 2 um rib whose 3 um slab ends at slab_end, ports on the core."""
    c = _waveguide_to(2.0, 2.0)
    c.add_polygon([(0.0, -1.5), (slab_end, -1.5), (slab_end, 1.5), (0.0, 1.5)], layer=SLAB)
    return c


def test_port_extensions_must_still_meet_the_port_face() -> None:
    """An extension continues the layer where it met the port face; when it meets the face elsewhere, pack
    raises, and retopologize finds the extensions again, on the layers they may continue."""
    d = _unplaced(_waveguide_to(2.0, 2.0), extend_ports=(("o2", 0.5),))
    d.pack(_waveguide_to(2.3, 2.3, width=0.45))  # the end and the port move together: the extension follows
    retreated = _waveguide_to(1.9, 2.0)  # the end left the port face
    with pytest.raises(
        ValueError,
        match=r"vertices/WG: its cross-section at port 'o2' changed: the device has extensions from "
        r"\(1\.9, -0\.25\) to \(1\.9, 0\.25\) on the port face, a device built now would have extensions from "
        r"\(2, -0\.25\) to \(2, 0\.25\)",
    ):
        d.pack(retreated)
    rebuilt = d.retopologize(retreated)
    assert rebuilt.source_extensions == ((("o2", 0.5, (-1, -1), (-1, -1)),),)  # no polygon there: the port's width
    assert rebuilt.ports == d.ports
    rebuilt.pack(retreated)

    # a slab that reaches the face later gets an extension too, unless the port only extends the core
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "slab", "clad")})
    kwargs = dict(included_layers=["box", "clad"], boundary_offset=(0.1, 0.0, 0.3, -0.3, 0.0, 0.0))
    d = build(_rib_to(1.9), ls, extend_ports=(("o2", 0.5),), **kwargs)
    with pytest.raises(
        ValueError, match=r"vertices/SLAB\w*: its cross-section at port 'o2' changed: the device has no extension"
    ):
        d.pack(_rib_to(2.0))
    rebuilt = d.retopologize(_rib_to(2.0))
    rebuilt.pack(_rib_to(2.0))
    assert sorted(len(s) for s in d.source_sizes) == [1, 2]  # the core's extension
    assert [len(s) for s in rebuilt.source_sizes] == [2, 2]  # and the slab's
    build(_rib_to(1.9), ls, extend_ports=(("o2", 0.5, [WG]),), **kwargs).pack(_rib_to(2.0))


def test_port_face_changes_under_10_nm() -> None:
    """A waveguide end that retreats from an extended port face by a few nm (a fraction of a cell) is caught:
    the device's extension would bridge the gap a device built now leaves open. Within the topology tolerance
    (2 nm) both bridge it."""
    d = _unplaced(_waveguide_to(2.0, 2.0), extend_ports=(("o2", 0.5),))
    for retreat in (0.004, 0.008):
        with pytest.raises(ValueError, match=r"cross-section at port 'o2' changed"):
            d.pack(_waveguide_to(2.0 - retreat, 2.0))
    nearly = _waveguide_to(1.999, 2.0)
    d.pack(nearly)
    assert d.retopologize(nearly).source_extensions == d.source_extensions


def _junctions(device: StackDevice) -> tuple[tuple[int, ...], int]:
    """Split sizes and seam edges of an unplaced device's first layer, as it finds them (in um)."""
    v = np.asarray(device.source_vertices[0], dtype=float)
    tol = device._topology_tol(v)
    index, sizes = split_t_junctions(v, device.source_sizes[0], tol)
    _, _, boundary = _edge_topology(v[index], sizes, tol)
    return sizes, int((~boundary).sum())


@pytest.mark.parametrize("name", list(JUNCTIONS))
def test_topology_with_layout_snapping(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """gdsfactory snaps to 1 nm by default, which moves a rotated component's junction corners up to ~1.4 nm off
    the slanted edges they lie on: they keep the topology (2 database units), and a device built for the
    rotated component keeps the junctions."""
    monkeypatch.setattr(gf.snap, "SNAP_ENABLED", True)
    pieces, _ = JUNCTIONS[name]
    d = _unplaced(_polygons(*pieces))
    for angle in (10.0, 25.0, 45.0):
        rotated = _polygons(*(_rotated(p, angle, np.array([0.3, -0.2])) for p in pieces))
        d.pack(rotated)
        assert _junctions(d.retopologize(rotated)) == _junctions(d), angle
        assert _junctions(_unplaced(rotated)) == _junctions(d), angle


def test_retopologize_keeps_what_the_check_keeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Junction corners within the topology tolerance of an MMI side pass the check, and the device built for
    them (retopologize, from the component or the parameters, or from_layer_stack) keeps the junctions: it
    equals the merged polygon through the moved corners. The MMI side lies inside voxels, where a dropped
    junction leaves a slit (fill off by ~0.3). Also with the 1 nm snapping of a rotated component."""
    pieces, merged = JUNCTIONS["taper on MMI"]
    size, center = (4.4, 1.8, 0.3), (2.007, 0.003)
    objects, _, params = _placed(_polygons(*pieces), _core(), CORE_ONLY, size, center, z_center=0.11)
    d = objects["s"]
    si = d.material_weights(params["s"])[0].index("si")

    def fill(device, component):
        return np.asarray(device.material_weights({**params["s"], **device.pack(component)})[1][si])

    def shifted(points, dx):  # the taper's end corners moved by dx along x
        return [(x + dx, y) if x == 2 and abs(y) == 0.25 else (x, y) for x, y in points]

    near = _polygons(*(shifted(p, 1e-3) for p in pieces))
    d.check_topology(near)
    _, merged_near = _weights_of(_polygons(shifted(merged, 1e-3)), _core(), CORE_ONLY, size, center, z_center=0.11)
    vertices = {k: np.asarray(v) for k, v in d.pack(near).items()}
    for device in (d, d.retopologize(near), d.retopologize(vertices)):
        geometry = device._layer_geometry(device.init_params())[0]
        assert np.bincount(geometry.ids).tolist() == [6, 4] and int((~geometry.boundary).sum()) == 2
        np.testing.assert_allclose(fill(device, near), merged_near["si"], atol=2e-5)
    _, fresh = _weights_of(near, _core(), CORE_ONLY, size, center, z_center=0.11)
    np.testing.assert_allclose(fresh["si"], merged_near["si"], atol=2e-5)
    with pytest.raises(ValueError, match="T-junction"):  # 3 nm: a gap the device would bridge
        d.check_topology(_polygons(*(shifted(p, 3e-3) for p in pieces)))

    monkeypatch.setattr(gf.snap, "SNAP_ENABLED", True)
    rotated = _polygons(*(_rotated(p, 25.0, np.array([2.0, 0.0])) for p in pieces))
    d.check_topology(rotated)
    _, merged_rotated = _weights_of(
        _polygons(_rotated(merged, 25.0, np.array([2.0, 0.0]))), _core(), CORE_ONLY, size, center, z_center=0.11
    )
    np.testing.assert_allclose(fill(d.retopologize(rotated), rotated), merged_rotated["si"], atol=2e-5)


def test_topology_tolerance_far_from_the_origin() -> None:
    """5000 um from the origin, float32 parameters are off their edges by up to ~1e-3 um: they pass, while a
    taper moved off an MMI side by 4 nm is caught."""
    mmi, taper = JUNCTIONS["taper on MMI"][0]

    def far(dx: float = 0.0) -> gf.Component:
        return _polygons([(x + 5000.0, y) for x, y in mmi], [(x + 5000.0 + dx, y) for x, y in taper])

    d = _unplaced(far())
    eps = float(np.finfo(np.float32).eps)
    assert d._topology_tol(d.source_vertices[0]) == pytest.approx(2e-3 + 2 * eps * 5004.0)
    d.check_topology(d.init_params())
    for dx in (0.004, 0.008):
        moved = {k: np.asarray(v, dtype=np.float32) for k, v in d.pack(far(dx), check=False).items()}
        with pytest.raises(ValueError, match="T-junction"):
            d.check_topology(moved)
    assert _unplaced(far(), topology_tol=0.01)._topology_tol(np.zeros((1, 2))) == 0.01
    with pytest.raises(ValueError, match="topology_tol"):
        _unplaced(far(), topology_tol=-1.0)


def test_polygons_on_a_layer_the_device_was_built_without() -> None:
    """A device built from a strip waveguide has no slab level and its core reads the etch layer as empty, so
    slab or etch polygons later need a new device: pack and check_topology report them and retopologize
    raises."""
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "slab", "clad")})
    kwargs = dict(included_layers=["box", "clad"], boundary_offset=(0.1, 0.0, 0.3, -0.3, 0.0, 0.0))
    d = build(_waveguide_to(2.0, 2.0), ls, **kwargs)
    assert [lv.name for lv in d.levels] == ["box", "core", "clad"]
    with pytest.raises(
        ValueError, match=r"vertices/SLAB\w*: 1 polygon on a layer the device was built without \(level 'slab' is "
        r"not drawn\)(.|\n)*StackDevice.from_layer_stack"
    ):
        d.pack(_rib_to(2.0))
    with pytest.raises(ValueError, match=r"level 'slab' is not drawn"):
        d.check_topology(_rib_to(2.0))
    with pytest.raises(ValueError, match=r"level 'slab' is not drawn(.|\n)*from_layer_stack"):
        d.retopologize(_rib_to(2.0))
    etched = _waveguide_to(2.0, 2.0)
    etched.add_polygon(_rect(1.0, -0.5, 1.2, 0.5), layer=ETCH)
    with pytest.raises(ValueError, match=r"vertices/\w*ETCH\w*: 1 polygon .*\(level 'core' reads it as empty\)"):
        d.pack(etched)
    d.check_topology(d.pack(_rib_to(2.0), check=False))  # parameters carry no layers: not checked
    assert [lv.name for lv in build(_rib_to(2.0), ls, **kwargs).levels] == ["box", "core", "slab", "clad"]
    # a slab within 0.1 um above the box is kept until placement, which finds that it cannot reach the box
    high = LayerStack(layers={
        **ls.layers, "slab": LayerLevel(layer=SLAB, zmin=0.55, thickness=0.03, material="si", mesh_order=3)
    })
    unplaced = build(_waveguide_to(2.0, 2.0), high, **kwargs)
    assert {name for _, readers in unplaced.empty_layers for name, *_ in readers} == {"core", "slab"}
    objects, _, _ = place(unplaced)
    assert {name for _, readers in objects["stack"].empty_layers for name, *_ in readers} == {"core"}
    objects["stack"].pack(_rib_to(2.0))


# ---------------------------------------------------------------- geometry diagnostics


def _rect(x0, y0, x1, y1) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _diagnose(
    component, resolution=RES, xy_subsamples=1, ls=None, derived_shapes="exact", **level
) -> dict[str, StackDiagnostic]:
    """Findings by kind of an unplaced device for the component (a core on WG with ``level``, or ``ls``)."""
    device = StackDevice.from_layer_stack(
        component, ls or _core(**level), CORE_ONLY, background="air", name="s", included_layers=["core"],
        boundary_offset=(0.3, -0.3, 0.3, -0.3, 0.0, 0.0), xy_subsamples=xy_subsamples, derived_shapes=derived_shapes,
    )
    found = device.diagnose(resolution)
    assert len({f.kind for f in found}) == len(found)  # one per level and kind
    return {f.kind: f for f in found}


# two waveguides 1 cell apart, and a line 1 cell wide
GAP_AND_LINE = [_rect(0, 0, 2, 0.5), _rect(0, 0.52, 2, 1.0), _rect(0, 1.3, 2, 1.32)]


def test_sub_cell_gap_and_line_are_diagnosed() -> None:
    found = _diagnose(_polygons(*GAP_AND_LINE))
    assert set(found) == {"width", "gap"}
    for kind, y in (("width", 1.31), ("gap", 0.51)):
        f = found[kind]
        assert (f.level, f.count, f.layers) == ("core", 1, ())
        assert f.size == pytest.approx(0.02, abs=1e-6)
        assert f.cells == pytest.approx(1.0, abs=1e-4)
        assert f.threshold == pytest.approx(1.5 * RES)
        assert f.location == pytest.approx((1.0, y), abs=1e-6)
        assert 0.0 <= f.z <= 0.22
    assert re.match(
        r"StackDevice 's', level 'core': 1 gap narrower than 0\.03 um \(1\.5 cells\); the narrowest is 0\.02 um "
        r"\(1 cells\) wide, at \(1\.0000, 0\.5100\) um",
        found["gap"].message,
    )
    assert "1 part narrower" in found["width"].message
    assert _diagnose(_polygons(*GAP_AND_LINE), resolution=0.01) == {}  # resolved on a finer grid


def test_diagnostics_scale_with_xy_subsamples() -> None:
    """Sub-samples resolve thinner features: the threshold is 1.5 cells / xy_subsamples."""
    assert _diagnose(_polygons(*GAP_AND_LINE), xy_subsamples=2) == {}  # 1 cell > 0.75
    half_cell = [_rect(0, 0, 2, 0.5), _rect(0, 0.51, 2, 1.0), _rect(0, 1.3, 2, 1.31)]
    found = _diagnose(_polygons(*half_cell), xy_subsamples=2)
    assert set(found) == {"width", "gap"}
    for f in found.values():
        assert f.threshold == pytest.approx(0.75 * RES)
        assert f.cells == pytest.approx(0.5, abs=1e-4)
        assert "(1.5 cells / xy_subsamples 2)" in f.message


def test_diagnostics_follow_the_sidewalls() -> None:
    """A line narrowing below the threshold towards its top, or vanishing, and a gap closing towards its
    bottom are found at their narrowest cross-section."""
    lines = _polygons(_rect(0, 0, 2, 0.06), _rect(0, 1, 2, 2))  # 3 cells at the bottom, and a wide block
    angle = math.degrees(math.atan(0.02 / 0.22))  # each side moves in by 0.02 um up to the top
    f = _diagnose(lines, sidewall_angle=angle, width_to_z=0.0)
    assert set(f) == {"width"}
    assert f["width"].size == pytest.approx(0.02, abs=1e-4)
    assert f["width"].z == pytest.approx(0.22)
    assert f["width"].location[1] == pytest.approx(0.03, abs=1e-6)
    # vanishes at z = 0.16: the last cross-section before is within half the threshold
    f = _diagnose(lines, sidewall_angle=2 * angle, width_to_z=0.0)["width"]
    assert 0 < f.size <= 0.5 * f.threshold + 1e-6
    assert 0.1 < f.z < 0.17
    assert _diagnose(lines, width_to_z=0.0) == {}
    # z_to_bias grows a 3-cell gap by 0.03 um on each side at the bottom: it closes there
    gap = _polygons(_rect(0, 0, 2, 0.5), _rect(0, 0.56, 2, 1.0))
    f = _diagnose(gap, z_to_bias=((0.0, 0.5, 1.0), (0.03, 0.0, 0.0)))
    assert set(f) == {"gap"}
    assert 0 < f["gap"].size <= 0.5 * f["gap"].threshold + 1e-6
    assert f["gap"].z < 0.11


def test_resolved_designs_are_not_diagnosed() -> None:
    """A rib waveguide continued through both ports, with fill levels, and placed: no findings or warnings."""
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "slab", "clad")})
    kwargs = dict(included_layers=["box", "clad"], boundary_offset=(0.1, 0.0, 0.3, -0.3, 0.0, 0.0))
    device = build(_rib(), ls, extend_ports=(("o1", 0.5), ("o2", 0.5)), corner_lines=0, **kwargs)
    assert device.diagnose(RES) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error", StackDiagnosticWarning)
        objects, _, _ = _place_own_volume(device)
    assert objects["stack"].diagnose() == []


def _etched_end(extend: float, beyond: float, derived_shapes: str = "exact") -> StackDevice:
    """Unplaced core (WG - ETCH, 10 deg sidewalls) of a waveguide whose end is etched up to its port o2.

    ``extend``: length of o2's extension (0: none); ``beyond``: how far (um) the box reaches past the end.
    """
    c = _waveguide()
    c.add_polygon(_rect(1.4, -0.5, 2.0, 0.5), layer=ETCH)
    return StackDevice.from_layer_stack(
        c, LayerStack(layers={"core": stack().layers["core"]}), CORE_ONLY, background="air", name="s",
        included_layers=["core"], extend_ports=(("o2", extend),) if extend else (),
        boundary_offset=(beyond, 0.0, 0.3, -0.3, 0.0, 0.0), derived_shapes=derived_shapes,
    )


def test_artificial_ends_of_port_extensions_are_not_diagnosed() -> None:
    """With the composite model, where the waveguide and the etch over it end together, the core (grown below
    z_ref by its sidewall angle) draws a sliver twice the grow wide: found where they end in the box, not at
    the end of port extensions at a box face. The exact derived shape draws no sliver there."""
    for extend, beyond, end in ((0.0, 0.3, 2.0), (0.5, 0.3, 2.5)):
        f = _etched_end(extend, beyond, "composite").diagnose(RES)
        assert [x.kind for x in f] == ["width"]
        assert f[0].location == pytest.approx((end, 0.0), abs=1e-3)
        assert f[0].z < 0.11  # the core grows below z_ref
        assert _etched_end(extend, beyond).diagnose(RES) == []
    assert _etched_end(0.5, 0.0, "composite").diagnose(RES) == []


OVERLAP = [_rect(0, 0, 1.01, 0.5), _rect(1.0, 0, 2, 0.5)]  # overlapping by 0.01 um
ABUT = [_rect(0, 0, 1.0, 0.5), _rect(1.0, 0, 2, 0.5)]


@pytest.mark.parametrize(
    ("level", "shrink", "z"),
    [
        (dict(sidewall_angle=10.0, width_to_z=0.5), 0.11 * math.tan(math.radians(10.0)), 0.22),
        (dict(bias=-0.01), 0.01, None),
    ],
)
def test_overlapping_polygons_on_a_shrinking_level(level: dict, shrink: float, z: float | None) -> None:
    """With the composite model, each polygon is shrunk, not their union: overlapping polygons are found, abutting
    ones are not. The exact model draws their shrunk union: nothing to report."""
    assert _diagnose(_polygons(*OVERLAP), **level) == {}
    f = _diagnose(_polygons(*OVERLAP), derived_shapes="composite", **level)["overlap"]
    assert (f.level, f.count, f.layers, f.threshold) == ("core", 1, ("vertices/WG",), 0.0)
    assert f.size == pytest.approx(shrink, rel=1e-6)
    assert f.cells == pytest.approx(shrink / RES, rel=1e-6)
    assert f.location == pytest.approx((1.005, 0.25), abs=1e-6)
    if z is not None:
        assert f.z == pytest.approx(z)
    assert re.match(
        rf"StackDevice 's', level 'core' shrinks vertices/WG by up to {shrink:.4g} um .* and its polygons overlap "
        r"at 1 place in the box \(0\.005 um\^2 in all; e\.g\. at \(1\.0050, 0\.2500\) um\)",
        f.message,
    )
    assert _diagnose(_polygons(*ABUT), **level) == {}


def test_overlapping_polygons_are_exact_where_a_level_grows() -> None:
    """The union of grown polygons is their grown union; a growing level shrinks the layers it subtracts."""
    for level in ({}, dict(bias=0.01), dict(sidewall_angle=-10.0, width_to_z=0.0)):
        assert _diagnose(_polygons(*OVERLAP), **level) == {}
    c = _polygons(_rect(0, 0, 2, 1))
    for poly in ([(0.5, 0.4), (1.01, 0.4), (1.01, 0.6), (0.5, 0.6)], _rect(1.0, 0.4, 1.5, 0.6)):
        c.add_polygon(poly, layer=ETCH)
    core = LayerLevel(
        layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), derived_layer=LogicalLayer(layer=WG), zmin=0.0,
        thickness=0.22, material="si", mesh_order=1, bias=0.01,
    )
    f = _diagnose(c, ls=LayerStack(layers={"core": core}), derived_shapes="composite")
    assert set(f) == {"overlap"}
    assert re.fullmatch(r"vertices/DEEP_ETCH\w*", f["overlap"].layers[0])
    assert f["overlap"].size == pytest.approx(0.01)
    assert _diagnose(c, ls=LayerStack(layers={"core": core})) == {}  # the exact shape: the etch's union


def test_placement_warns_about_unresolved_geometry() -> None:
    """One StackDiagnosticWarning per level and kind when the device is placed; diagnostics=False silences them."""
    args = (_polygons(*GAP_AND_LINE), _core(), CORE_ONLY, (2.4, 1.8, 0.3), (1.0, 0.66))
    with pytest.warns(StackDiagnosticWarning) as record:
        objects, _, _ = _placed(*args, z_center=0.11, corner_lines=0)
    messages = [str(w.message) for w in record if issubclass(w.category, StackDiagnosticWarning)]
    assert len(messages) == 2
    assert sorted(messages) == sorted(f.message for f in objects["s"].diagnose())
    assert {f.kind for f in objects["s"].diagnose()} == {"width", "gap"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", StackDiagnosticWarning)
        objects, _, _ = _placed(*args, z_center=0.11, corner_lines=0, diagnostics=False)
    assert len(objects["s"].diagnose()) == 2  # still available as data
    unplaced = _unplaced(_polygons(*GAP_AND_LINE))
    with pytest.raises(ValueError, match="not placed"):
        unplaced.diagnose()
    with pytest.raises(ValueError, match="positive"):
        unplaced.diagnose(0.0)


def _tip_above(degrees: float, gap: float = 0.005) -> list[tuple[float, float]]:
    """A triangle whose tip of ``degrees`` points down at the top (y = 0.5) of a waveguide, ``gap`` above it."""
    w = 0.3 * math.tan(math.radians(degrees / 2))
    return [(1.0, 0.5 + gap), (1.0 + w, 0.8 + gap), (1.0 - w, 0.8 + gap)]


def test_corners_and_notches_near_edges_are_diagnosed() -> None:
    """A corner of 90 degrees or less pointing at an edge, and a V-notch nearly cutting through a part, are
    found: only the two sides of one corner are the corner, not a narrow part (review finding I4-1)."""
    for degrees in (20.0, 60.0, 90.0, 120.0):
        f = _diagnose(_polygons(_rect(0, 0, 2, 0.5), _tip_above(degrees)))
        assert set(f) == {"gap"}, degrees
        assert f["gap"].size == pytest.approx(0.005, abs=1e-5)
        assert f["gap"].location == pytest.approx((1.0, 0.5025), abs=1e-3)
    square = _rotated(_rect(0.9, 0.505, 1.1, 0.705), 45.0, np.array([0.9, 0.505]))
    square = [(x, y - min(y for _, y in square) + 0.505) for x, y in square]  # its corner 0.005 above the edge
    assert _diagnose(_polygons(_rect(0, 0, 2, 0.5), square))["gap"].size == pytest.approx(0.005, abs=1e-5)
    for opening in (30.0, 60.0, 90.0):
        w = 0.3 * math.tan(math.radians(opening / 2))
        f = _diagnose(_polygons([(0, 0), (2, 0), (2, 0.5), (1.0 + w, 0.5), (1.0, 0.005), (1.0 - w, 0.5), (0, 0.5)]))
        assert set(f) == {"width"}, opening
        assert f["width"].size == pytest.approx(0.005, abs=1e-5)
        assert f["width"].location == pytest.approx((1.0, 0.0025), abs=1e-3)
    # resolved shapes with sharp corners of 30 to 82 degrees, also rounded by a sidewall angle, are not reported
    parallelogram = [(0, 0), (1, 0), (1.3, 0.5), (0.3, 0.5)]
    facet = [(0, 0), (2, 0), (2 + 0.5 / math.tan(math.radians(82)), 0.5), (0, 0.5)]
    assert _diagnose(_polygons([(0, 0), (1.0, 0), (0, math.tan(math.radians(30)))])) == {}
    for shape in (parallelogram, facet):
        assert _diagnose(_polygons(shape)) == {}
        assert _diagnose(_polygons(shape), sidewall_angle=10.0, width_to_z=0.5) == {}


def _etched_holes(holes, bias: float) -> dict[str, StackDiagnostic]:
    """Findings of a WG - ETCH level with ``bias`` on a large WG block with etch holes."""
    c = _polygons(_rect(-0.5, -0.5, 2.5, 1.5))
    for hole in holes:
        c.add_polygon(hole, layer=ETCH)
    core = LayerLevel(
        layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), derived_layer=LogicalLayer(layer=WG), zmin=0.0,
        thickness=0.22, material="si", mesh_order=1, bias=bias,
    )
    return _diagnose(c, ls=LayerStack(layers={"core": core}))


def test_diagnostics_draw_round_joins() -> None:
    """Offsets have the device's round joins, not KLayout's square corners (review finding I4-2): corners facing
    each other diagonally are their distance minus twice the grow apart, and a strip with round ends keeps its
    width."""
    for gap, expected in ((0.08, None), (0.065, 0.025)):
        s = gap / math.sqrt(2)
        grown = _diagnose(_polygons(_rect(0, 0, 1.0, 0.5), _rect(1.0 + s, 0.5 + s, 2.0, 1.0)), bias=0.02)
        # the holes grow where the level shrinks: the silicon between their corners narrows
        shrunk = _etched_holes([_rect(0.2, 0.2, 1.0, 0.5), _rect(1.0 + s, 0.5 + s, 1.8, 0.8)], -0.02)
        if expected is None:  # square corners: a gap and a part of 0.0234 um
            assert grown == {} and shrunk == {}
            continue
        assert set(grown) == {"gap"} and set(shrunk) == {"width"}
        for f in (grown["gap"], shrunk["width"]):
            assert f.size == pytest.approx(expected, abs=8e-4)  # square corners: 0.0084 um
            assert f.location == pytest.approx((1.0 + 0.5 * s, 0.5 + 0.5 * s), abs=2e-3)
    # a line 0.01 um wide grown by 0.005 um: 0.02 um wide in its middle, also next to its round ends
    f = _diagnose(_polygons(_rect(0.2, 0.5, 1.8, 0.51)), bias=0.005)
    assert set(f) == {"width"}
    assert f["width"].size == pytest.approx(0.02, abs=1e-6)
    assert f["width"].location == pytest.approx((1.0, 0.505), abs=1e-6)


def test_round_offsets_match_the_signed_distance() -> None:
    """The diagnostics offset a layer by its signed distance (round joins at grown convex and shrunk reflex
    corners), to within the joins' tolerance; KLayout's own sizing is not that offset."""
    kdb = pytest.importorskip("klayout.db")

    def region(*polys):
        r = kdb.Region()
        for p in polys:
            r.insert(kdb.Polygon([kdb.Point(round(x * 1e4), round(y * 1e4)) for x, y in p]))
        return r.merged()

    def disk(r: int, n: int = 180):  # within 0.1 dbu of the circle
        return kdb.Polygon([kdb.Point(round(r * math.cos(2 * math.pi * k / n)), round(r * math.sin(2 * math.pi * k / n)))
                            for k in range(n)])

    def reference(r, g: int):
        if g > 0:
            return r.minkowski_sum(disk(g)).merged()
        outer = kdb.Region(r.bbox().enlarged(1000 - 4 * g, 1000 - 4 * g))
        return outer - (outer - r).minkowski_sum(disk(-g)).merged()

    def within(a, b, s: int) -> bool:
        return (a - b.sized(s)).is_empty() and (b - a.sized(s)).is_empty()

    shapes = [
        region([(0, 0), (1, 0), (1, 0.3), (0.3, 0.3), (0.3, 1), (0, 1)]),  # convex and reflex corners
        region([(0, 0), (1.0, 0), (0, math.tan(math.radians(30)))]),  # acute corners
        region(_rect(0, 0, 1, 1)) - region(_rect(0.2, 0.2, 0.4, 0.4), [(0.6, 0.6), (0.8, 0.6), (0.6, 0.65)]),
        region(*[_rect(i * 0.03, 0, i * 0.03 + 0.012, 0.3) for i in range(10)], _rect(0, -0.1, 0.3, 0)),  # a comb
        region(_rotated(_rect(0, 0, 0.8, 0.3), 17.0, np.array([0.4, 0.15]))),
    ]
    for r in shapes:
        corners = tuple((v, _seeds(v)) for v in _corners(r))
        for g in (30, 200, 600, -30, -200, -600):  # dbu of 0.1 nm
            offset, _ = _round_offset(r, g, corners, r.edges(), 10**8, 3.0)  # every join drawn
            assert within(offset, reference(r, g), 5), g
    assert not within(shapes[0].sized(200), reference(shapes[0], 200), 5)  # KLayout's square corners


def test_joins_only_where_another_edge_is_near(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drawing round joins only at corners with another edge within reach finds what drawing all of them finds."""
    import gdsfactory.fdtdx_stack as module

    rng = np.random.default_rng(5)

    def star(cx, cy, n, r0, r1):
        phi = np.sort(rng.uniform(0, 2 * np.pi, n))
        return [(cx + r * math.cos(p), cy + r * math.sin(p)) for r, p in zip(rng.uniform(r0, r1, n), phi)]

    every = module._round_offset
    for expr, level in (
        (LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), dict(sidewall_angle=-20.0, width_to_z=0.5)),
        (LogicalLayer(layer=WG) | LogicalLayer(layer=ETCH), dict(bias=-0.012)),
        (LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH), dict(sidewall_angle=15.0, width_to_z=0.5)),
        (LogicalLayer(layer=WG), dict(bias=0.018)),
    ):
        c = _polygons(star(0.5, 0.5, 7, 0.25, 0.5), *(star(x, y, 5, 0.02, 0.12) for x, y in rng.uniform(0.1, 1.1, (3, 2))))
        c.add_polygon(star(0.7, 0.6, 6, 0.2, 0.4), layer=ETCH)
        core = LayerLevel(layer=expr, derived_layer=LogicalLayer(layer=WG), zmin=0.0, thickness=0.22, material="si",
                          mesh_order=1, **level)
        ls = LayerStack(layers={"core": core})
        found = {k: (round(f.size, 4), f.count) for k, f in _diagnose(c, ls=ls).items()}
        with monkeypatch.context() as m:
            m.setattr(module, "_round_offset", lambda *args: every(*args[:4], 10**8, args[5]))
            assert {k: (round(f.size, 4), f.count) for k, f in _diagnose(c, ls=ls).items()} == found


def test_narrow_places_outside_the_box_are_not_diagnosed() -> None:
    """A gap whose edges stay outside the box, near one of its corners, is not reported, although the edges'
    bounding boxes overlap the box (review finding I4-3); the same gap in the box is."""

    def strips(center) -> list[list[tuple[float, float]]]:
        """Two 2 um long 45 degree strips 0.01 um apart around a point."""
        along, across = np.array([1.0, -1.0]) / math.sqrt(2), np.array([1.0, 1.0]) / math.sqrt(2)
        return [[tuple(p + d) for d in (-along, along, along + 0.2 * across, -along + 0.2 * across)]
                for p in (np.asarray(center), np.asarray(center) + 0.21 * across)]

    anchors = (_rect(-1, -1, -0.99, -0.99), _rect(2.99, 2.99, 3, 3))  # with the offsets, the box is [0.3, 1.7]^2
    # the gap 0.35 um past the box's corner, or across the box (measured on its part in the box)
    for center, expected in (((1.806, 1.806), None), ((0.6, 0.6), (0.745, 0.745))):
        device = StackDevice.from_layer_stack(
            _polygons(*anchors, *strips(center)), _core(), CORE_ONLY, background="air", name="s",
            included_layers=["core"], boundary_offset=(-1.3, 1.3, -1.3, 1.3, 0.0, 0.0),
        )
        found = device.diagnose(RES)
        if expected is None:
            assert found == []
        else:
            assert [f.kind for f in found] == ["gap"]
            assert found[0].size == pytest.approx(0.01, abs=1e-4)  # vertices on the 0.1 nm grid of the checks
            assert found[0].location == pytest.approx(expected, abs=5e-3)


def test_narrow_places_are_measured_in_the_box() -> None:
    """A gap that narrows toward a point outside the box gets its width and its location at the box, not the
    narrowest width of the whole edges (review finding I4-4)."""
    c = _polygons(_rect(0, 0, 2, 0.5), [(0, 0.505), (2, 0.525), (2, 1.0), (0, 1.0)])  # 0.005 + 0.01 x wide
    for x_offsets, size, x in (((0.0, 1.0), 0.015, 1.0), ((0.3, -0.3), 0.005, 0.0)):  # box x in [1, 2], [-0.3, 2.3]
        device = StackDevice.from_layer_stack(
            c, _core(), CORE_ONLY, background="air", name="s", included_layers=["core"],
            boundary_offset=(*x_offsets, 0.3, -0.3, 0.0, 0.0),
        )
        (f,) = device.diagnose(RES)
        assert f.kind == "gap"
        assert f.size == pytest.approx(size, abs=1e-6)
        assert f.location == pytest.approx((x, 0.5025 + 0.005 * x), abs=1e-6)  # the middle of the gap there


# ---------------------------------------------------------------- exact derived shapes


def _core_of(expr, **level) -> LayerStack:
    """A 0.22 um si core drawing a layer expression."""
    core = LayerLevel(layer=expr, derived_layer=LogicalLayer(layer=WG), zmin=0.0, thickness=0.22, material="si",
                      mesh_order=1, **level)
    return LayerStack(layers={"core": core})


def _two_layers(wg, etch, etch_layer=ETCH) -> gf.Component:
    c = gf.Component()
    for poly in wg:
        c.add_polygon(poly, layer=WG)
    for poly in etch:
        c.add_polygon(poly, layer=etch_layer)
    return c


def _true_volume(device, region_at, z0=0.0, z1=0.22, n=8) -> np.ndarray:
    """(nx, ny, nz) exact volume fractions of a region that depends on the height (shapely, n heights per voxel)."""
    shapely = pytest.importorskip("shapely")
    nx, ny, nz = device.grid_shape
    x0, y0 = device.center[0] - nx * RES / 2, device.center[1] - ny * RES / 2
    zb = device.z_center - nz * RES / 2
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    voxels = shapely.box(x0 + i * RES, y0 + j * RES, x0 + (i + 1) * RES, y0 + (j + 1) * RES)
    out = np.zeros((nx, ny, nz))
    for k in range(nz):
        a, b = max(zb + k * RES, z0), min(zb + (k + 1) * RES, z1)
        for q in range(n if b > a else 0):
            z = a + (q + 0.5) * (b - a) / n
            out[:, :, k] += (b - a) / n / RES * shapely.area(shapely.intersection(region_at(z), voxels)) / RES**2
    return out


def test_derived_level_has_no_sliver_where_edges_coincide() -> None:
    """WG - ETCH where the etch ends with the waveguide (x = 2.0): the exact shape draws nothing there at any
    height, matches the true offset of WG - ETCH everywhere and is not diagnosed; the composite model draws
    a sliver 2|offset| wide below z_ref (up to 0.88 si) and reports it."""
    shapely = pytest.importorskip("shapely")
    objects, _, params = place(build())
    d = objects["stack"]
    assert [s.levels for s in d.derived_layers] == [("core",)]  # kept by _keep_reachable_levels
    assert d.diagnose() == []
    names, w = d.material_weights(params["stack"])
    si = np.asarray(w[names.index("si")])
    i0, i1 = cell(d, 1.99, 0, 0)[0], cell(d, 2.015, 0, 0)[0]
    above_slab = [k for k in range(d.grid_shape[2]) if 0.08 < d.z_center + (k + 0.5 - d.grid_shape[2] / 2) * RES < 0.22]
    assert float(np.abs(si[i0 : i1 + 1][:, :, above_slab]).max()) == 0.0

    # core only: against the true offset of WG - ETCH (sidewall 10 deg about z_ref = 0.11)
    core_only = LayerStack(layers={"core": stack().layers["core"]})
    objects, _, params = place(build(ls=core_only, materials=CORE_ONLY))
    d = objects["stack"]
    names, w = d.material_weights(params["stack"])
    si = np.asarray(w[names.index("si")])
    shape = shapely.box(0.0, -0.25, 2.0, 0.25).difference(shapely.box(1.4, -0.5, 2.0, 0.5))
    tan = math.tan(math.radians(10.0))
    truth = _true_volume(d, lambda z: shape.buffer(-(z - 0.11) * tan, quad_segs=64))
    assert float(np.abs(si - truth).max()) < 0.03
    assert float(np.abs(si[cell(d, 1.99, 0, 0)[0] : cell(d, 2.015, 0, 0)[0] + 1]).max()) == 0.0

    objects, _, params = place(build(ls=core_only, materials=CORE_ONLY, derived_shapes="composite"))
    d = objects["stack"]
    assert d.derived_layers == ()
    names, w = d.material_weights(params["stack"])
    si = np.asarray(w[names.index("si")])
    assert float(si[cell(d, 1.99, 0, 0)[0], cell(d, 0, 0, 0)[1], cell(d, 0, 0, 0.01)[2]]) > 0.85
    assert [f.kind for f in d.diagnose()] == ["width"]


@pytest.mark.parametrize("delta", [0.005, 0.01, -0.005])
def test_near_coincident_edges_draw_no_sliver(delta: float) -> None:
    """The etch ends 5-10 nm past the waveguide's end: nothing is left there; 5 nm before it: the real strip,
    grown below z_ref by the sidewall, is drawn."""
    c = _two_layers([_rect(0, -0.25, 2.0, 0.25)], [_rect(1.4, -0.5, 2.0 + delta, 0.5)])
    d, w = _weights_of(c, _core_of(LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), sidewall_angle=10.0,
                                   width_to_z=0.5), CORE_ONLY, (2.6, 1.0, 0.3), center=(1.0, 0.0), z_center=0.11)
    si = w["si"]
    strip = si[cell(d, 1.97, 0, 0)[0] : cell(d, 2.03, 0, 0)[0] + 1]
    if delta > 0:
        assert float(strip.max()) <= 1e-5
    else:  # 5 nm grown by up to 17.6 nm on each side at the bottom
        assert float(strip[:, cell(d, 2.0, 0, 0)[1], cell(d, 0, 0, 0.01)[2]].sum()) > 0.9


def test_grown_and_has_round_corners() -> None:
    """The and of two crossing bars, grown and shrunk by a bias: the offset of their intersection (round joins
    when grown), not the intersection of the grown bars (sharp corners)."""
    shapely = pytest.importorskip("shapely")
    c = _two_layers([_rect(0.0, 0.4, 2.0, 0.6)], [_rect(0.9, 0.0, 1.1, 1.0)])
    for bias in (0.03, -0.03):
        errors = {}
        for mode in ("exact", "composite"):
            d, w = _weights_of(c, _core_of(LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH), bias=bias), CORE_ONLY,
                               (1.0, 1.0, 0.3), center=(1.0, 0.5), z_center=0.11, derived_shapes=mode)
            exact = _area_fractions(d, shapely.box(0.9, 0.4, 1.1, 0.6).buffer(bias, quad_segs=256, join_style="round"))
            errors[mode] = float(np.abs(w["si"][:, :, cell(d, 0, 0, 0.11)[2]] - exact).max())
        assert errors["exact"] < 1e-3, (bias, errors)
        if bias > 0:
            assert errors["composite"] > 0.1  # the sharp corners of the grown bars' intersection


@pytest.mark.parametrize("overlap", [0.0, 0.02])
def test_or_of_abutting_layers_shrinks_as_one(overlap: float) -> None:
    """WG | SLAB of a rectangle split between the two layers, abutting or overlapping, shrunk by a bias: one
    rectangle, no slit where they meet (review finding I4-5); the composite model opens one."""
    c = _two_layers([_rect(0.0, 0.0, 1.0 + overlap, 0.5)], [_rect(1.0, 0.0, 2.0, 0.5)], etch_layer=SLAB)
    for bias in (-0.02, -0.005):
        rows = {}
        for mode in ("exact", "composite"):
            ls = _core_of(LogicalLayer(layer=WG) | LogicalLayer(layer=SLAB), bias=bias)
            d, w = _weights_of(c, ls, CORE_ONLY, (2.4, 0.9, 0.3), center=(1.0, 0.25), z_center=0.11,
                               derived_shapes=mode)
            j, k = cell(d, 0, 0.25, 0)[1], cell(d, 0, 0, 0.11)[2]
            rows[mode] = w["si"][cell(d, 0.1, 0, 0)[0] : cell(d, 1.9, 0, 0)[0], j, k]
        assert float(np.abs(rows["exact"] - 1.0).max()) < 1e-5, (overlap, bias)
        if overlap < -2 * bias and bias < -0.01:  # a slit of a voxel or more
            assert float(rows["composite"].min()) < 0.5, (overlap, bias)
        assert _diagnose(c, ls=ls) == {}


def test_overlapping_polygons_shrink_as_their_union() -> None:
    """Overlapping polygons on one layer, shrunk: their union shrinks (no slit); the composite model opens a slit."""
    c = _polygons(*OVERLAP)
    rows = {}
    for mode in ("exact", "composite"):
        d, w = _weights_of(c, _core(bias=-0.01), CORE_ONLY, (2.4, 0.9, 0.3), center=(1.0, 0.25), z_center=0.11,
                           derived_shapes=mode)
        rows[mode] = w["si"][cell(d, 0.1, 0, 0)[0] : cell(d, 1.9, 0, 0)[0], cell(d, 0, 0.25, 0)[1],
                             cell(d, 0, 0, 0.11)[2]]
        assert len(d.derived_layers) == (mode == "exact")
    assert float(np.abs(rows["exact"] - 1.0).max()) < 1e-5
    assert float(rows["composite"].min()) < 0.6


@pytest.mark.parametrize(
    "op", ["hole", "xor", "background", "empty and"],
)
def test_derived_holes_xor_and_background(op: str) -> None:
    """A hole etched into a waveguide, an xor, everything but two layers (drawn as a complement) and an and of
    disjoint layers (nothing): the fill is the offset of the exact shape."""
    shapely = pytest.importorskip("shapely")
    wg, etch = LogicalLayer(layer=WG), LogicalLayer(layer=ETCH)
    a = shapely.box(0.0, 0.0, 1.2, 0.8)
    b = {"hole": shapely.box(0.4, 0.3, 0.8, 0.5), "xor": shapely.box(0.6, 0.4, 1.6, 1.0),
         "background": shapely.box(0.6, 0.4, 1.6, 1.0), "empty and": shapely.box(1.4, 0.0, 1.8, 0.6)}[op]
    expr, shape = {
        "hole": (wg - etch, a.difference(b)),
        "xor": (wg ^ etch, a.symmetric_difference(b)),
        "background": (LogicalLayer(layer=WAFER) - wg - etch, shapely.box(-5, -5, 5, 5).difference(a.union(b))),
        "empty and": (wg & etch, shapely.Polygon()),
    }[op]
    c = _two_layers([list(a.exterior.coords)[:-1]], [list(b.exterior.coords)[:-1]])
    d, w = _weights_of(c, _core_of(expr, bias=0.01), CORE_ONLY, (2.2, 1.6, 0.3), center=(0.8, 0.5), z_center=0.11)
    (shape_,) = d.derived_layers
    assert shape_.recipes[0].negate == (op == "background")
    assert (sum(len(r.kind) for r in shape_.recipes) == 0) == (op == "empty and")
    assert len(shape_.recipes) == (2 if op == "xor" else 1)  # an xor's two parts touch at the crossings
    exact = _area_fractions(d, shape.buffer(0.01, quad_segs=256))
    assert float(np.abs(w["si"][:, :, cell(d, 0, 0, 0.11)[2]] - exact).max()) < 1e-3


def test_sizing_of_the_root_is_the_offset_of_the_shape() -> None:
    """Sizing the whole expression offsets the exact shape (round joins); sizing below the root cannot be drawn
    exactly: that level falls back to the boolean of offset layers, with a warning."""
    shapely = pytest.importorskip("shapely")
    c = _two_layers([_rect(0.0, 0.4, 2.0, 0.6)], [_rect(0.9, 0.0, 1.1, 1.0)])
    expr = (LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH)).sized(30)
    d, w = _weights_of(c, _core_of(expr), CORE_ONLY, (1.0, 1.0, 0.3), center=(1.0, 0.5), z_center=0.11)
    assert len(d.derived_layers) == 1
    exact = _area_fractions(d, shapely.box(0.9, 0.4, 1.1, 0.6).buffer(0.03, quad_segs=256))
    assert float(np.abs(w["si"][:, :, cell(d, 0, 0, 0.11)[2]] - exact).max()) < 1e-3

    inner = LogicalLayer(layer=WG).sized(30) & LogicalLayer(layer=ETCH)
    with pytest.warns(StackDiagnosticWarning, match="level 'core' is drawn as the boolean of offset layers"):
        d, w = _weights_of(c, _core_of(inner), CORE_ONLY, (1.0, 1.0, 0.3), center=(1.0, 0.5), z_center=0.11)
    assert d.derived_layers == ()
    _, ref = _weights_of(c, _core_of(inner), CORE_ONLY, (1.0, 1.0, 0.3), center=(1.0, 0.5), z_center=0.11,
                         derived_shapes="composite")
    assert np.array_equal(w["si"], ref["si"])


def test_derived_gradients_match_finite_differences() -> None:
    """Gradients through the exact shape's recipe: the etch's start, the waveguide's width, the sidewall angle
    and the thickness, against finite differences; finite at the coincidence x = 2.0 (and 1 nm off it)."""
    core_only = LayerStack(layers={"core": stack().layers["core"]})
    objects, _, params = place(build(ls=core_only, materials=CORE_ONLY))
    d = objects["stack"]
    si_index = d.material_weights(params["stack"])[0].index("si")

    def si_volume(width, etch_x, angle, thickness, etch_end=2.0):
        c = circuit(width=width, etch_x=etch_x)
        p = {**params["stack"], **d.pack(c, check=False), "sidewall_angle/core": angle, "thickness/core": thickness}
        v = p["vertices/DEEP_ETCH"]  # the etch's end (x = 2.0, with the waveguide's)
        p["vertices/DEEP_ETCH"] = v.at[:, 0].set(jnp.where(v[:, 0] > 1.9, etch_end, v[:, 0]))
        return d.material_weights(p)[1][si_index].sum()

    x0 = (0.5, 1.4, 10.0, 0.22)
    g = jax.grad(si_volume, argnums=(0, 1, 2, 3))(*x0)
    for a, h in enumerate((1e-3, 1e-3, 0.1, 1e-3)):
        up, down = list(x0), list(x0)
        up[a] += h
        down[a] -= h
        fd = (si_volume(*up) - si_volume(*down)) / (2 * h)
        assert float(g[a]) == pytest.approx(float(fd), rel=5e-3), a
    assert float(g[0]) > 0 and float(g[1]) > 0 and float(g[3]) > 0
    for end in (2.0, 2.001, 1.999):
        ge = jax.grad(si_volume, argnums=4)(*x0, end)
        assert bool(jnp.isfinite(ge)), end


def test_derived_gradients_are_finite_at_degeneracies() -> None:
    """Exact shapes with a crossing at 0.1 deg, an empty and, float32, sub-samples, a 2D grid and Yee shifts."""
    spike = 1.0 * math.tan(math.radians(0.1))
    c = _two_layers([_rect(0.0, 0.0, 1.0, 0.5), [(0.0, 0.6), (1.0, 0.6 + spike), (1.0, 0.7)]],
                    [[(0.0, 0.6 - 0.001), (1.0, 0.6 + spike + 0.001), (1.0, 0.55), (0.0, 0.55)],
                     _rect(0.5, -0.1, 1.2, 0.2), _rect(3.0, 3.0, 3.1, 3.1)])
    for expr in (LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH)):
        for size, center, kwargs in (
            ((1.4, 1.0, 0.3), (0.5, 0.35), {}),
            ((1.4, RES, 0.3), (0.5, 0.25), {}),  # a 2D grid
            ((1.4, 1.0, 0.3), (0.5, 0.35), {"xy_subsamples": 2}),
        ):
            d, _ = _weights_of(c, _core_of(expr, sidewall_angle=10.0, width_to_z=0.5), CORE_ONLY, size,
                               center=center, z_center=0.11, **kwargs)

            def loss(p, d=d):
                w = d.material_weights(p, shift=_YEE_E_SHIFTS[0])[1]
                return (w * jax.random.uniform(jax.random.PRNGKey(0), w.shape)).sum()

            g = jax.grad(loss)(d.init_params())
            assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g)), (expr, kwargs)


def test_check_topology_reports_derived_shape_changes() -> None:
    """A waveguide growing past the etch's end adds a part the frozen shape lacks; the etch sliding off the
    waveguide moves a crossing off its edge. retopologize builds the shape again."""
    device = build(ls=LayerStack(layers={"core": stack().layers["core"]}), materials=CORE_ONLY)
    device.check_topology(circuit())
    longer = gf.Component()
    longer.add_ref(gf.components.straight(length=2.05, width=0.5))
    longer.add_polygon(_rect(1.4, -0.5, 2.0, 0.5), layer=ETCH)
    with pytest.raises(ValueError, match=r"level 'core' \(exact shape of vertices/WG, vertices/DEEP_ETCH\): the "
                                         r"shape it draws differs from the boolean of the new vertices by a part "
                                         r"0\.0[45]\d* um wide at \(2\.0[23]"):
        device.check_topology(device._pack(longer))
    with pytest.raises(ValueError, match="left that edge"):
        device.check_topology(device._pack(circuit(etch_x=-0.1)))
    new = device.retopologize(device._pack(longer))
    new.check_topology(new._pack(longer))
    (shape,) = new.derived_layers
    (recipe,) = shape.recipes
    assert int(recipe.kind[recipe.ring_of >= 0].sum()) == 4  # both etch edges cross both waveguide sides


def test_composite_mode_keeps_plain_levels() -> None:
    """Plain levels (one layer without overlaps, fills) are drawn the same in both modes; derived_shapes is
    validated."""
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "slab", "clad")})
    weights = {}
    for mode in ("exact", "composite"):
        objects, _, params = place(build(ls=ls, derived_shapes=mode))
        assert objects["stack"].derived_layers == ()
        weights[mode] = np.asarray(objects["stack"].material_weights(params["stack"])[1])
    assert np.array_equal(weights["exact"], weights["composite"])
    with pytest.raises(ValueError, match="derived_shapes"):
        build(ls=ls, derived_shapes="round")


# ---------------------------------------------------------------- step timings


def test_step_timings_change_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timing on gives bit-identical values and gradients and records every geometry step."""
    from gdsfactory import fdtdx_timing
    from gdsfactory.fdtdx_stack import record_timings

    monkeypatch.setattr(fdtdx_timing, "_ACTIVE", None)  # off, also under the timing plugin

    def run(objects, arrays, params):
        d = objects["stack"]

        def loss(width):
            p = {**params, "stack": {**params["stack"], **d.pack(circuit(width=width))}}
            a, _, _ = apply_stack(arrays, objects, p, jax.random.PRNGKey(0))
            return jnp.sum(1.0 / a.inv_permittivities)

        value, grad = jax.value_and_grad(loss)(0.5)  # eager autodiff: write and its steps run on concrete primals
        weights = d.material_weights(params["stack"])[1]  # eager
        jitted = jax.jit(lambda p: d.material_weights(p)[1])(params["stack"])
        return [np.asarray(x) for x in (value, grad, weights, jitted)]

    kwargs = dict(averaging="subpixel", corner_lines=2)
    off = run(*place(build(**kwargs)))
    assert fdtdx_timing.active() is None
    with record_timings() as timings:
        on = run(*place(build(**kwargs)))
        d = build(**kwargs)
        d.pack(circuit(width=0.55))  # concrete: topology check
    assert fdtdx_timing.active() is None
    for a, b in zip(off, on, strict=True):
        assert np.array_equal(a, b)
    steps = timings.by_leaf()
    expected = {
        "from_layer_stack", "from_layer_stack.compile_levels", "from_layer_stack.polygons",
        "from_layer_stack.port_extensions", "from_layer_stack.select_levels", "derived_shapes",
        "place_on_grid", "keep_reachable_levels", "place_on_grid.pole_alignment", "corner_budget", "diagnose",
        "pack", "pack_vertices", "topology_check", "apply_stack", "apply_stack.write_stacks",
        "apply_stack.apply_params", "write", "write.Ex", "write.Ey", "write.Ez", "write.set_permittivity",
        "weights", "level_surfaces", "layer_geometry", "source_sdfs", "corner_plan",
    }
    assert expected <= set(steps), sorted(expected - set(steps))
    assert {"composite", "composite_samples"} & set(steps)
    assert "normal_tensor" in steps
    # inside jax.value_and_grad without jit: eager autodiff, not tracing (R3-timing-2)
    assert steps["write"].calls > 0 and steps["write"].traced_calls == 0
    top = timings.steps["weights"]  # material_weights called directly (not under write)
    assert top.traced_calls >= 1 and top.calls > top.traced_calls  # under jax.jit, and eager
    assert all(r.wall >= 0 and r.compile >= 0 and r.run >= 0 for r in timings.steps.values())
    assert "weights" in timings.table()


def test_step_timings_merge_compile_intervals() -> None:
    """Nested or overlapping compile events count once, clipped to the step."""
    from gdsfactory.fdtdx_timing import _merged_length

    assert _merged_length([(1.0, 2.0), (1.5, 1.8), (1.9, 3.0)], 0.0, 2.5) == pytest.approx(1.5)
    assert _merged_length([(0.0, 1.0)], 2.0, 3.0) == 0.0
    assert _merged_length([], 0.0, 1.0) == 0.0


def test_step_timings_tell_jit_tracing_from_eager_autodiff() -> None:
    """Steps under jit are traced; under eager jax.grad they are eager (compile events counted, outputs
    waited for); compile events outside every step are summed apart (R3-timing-2, -3)."""
    from gdsfactory.fdtdx_timing import StepTimings, record_timings, timed_step

    def work(x, name):
        with timed_step(name) as step:
            return step.done(jnp.sin(x) * 2.0)

    timings = StepTimings()
    with record_timings(timings):
        jax.jit(lambda x: work(x, "in_jit").sum())(jnp.arange(7.0))  # traced step; jit's own compile is outside
        jax.grad(lambda x: work(x, "in_grad").sum())(jnp.arange(5.0))  # eager autodiff
        work(jnp.arange(3.0), "in_eager")
        with timed_step("whole_grad") as step:  # the caller's boundary sees the backward pass
            step.done(jax.value_and_grad(lambda x: work(x, "in_grad2").sum())(jnp.arange(4.0)))
    assert timings.steps["in_jit"].traced_calls == 1
    assert timings.steps["in_grad"].traced_calls == 0 and timings.steps["in_grad"].calls == 1
    assert timings.steps["in_eager"].traced_calls == 0
    assert timings.steps["whole_grad/in_grad2"].traced_calls == 0
    assert timings.steps["whole_grad"].wall >= timings.steps["whole_grad/in_grad2"].wall
    assert timings.outside_compile > 0.0  # the jit's lowering and XLA compile belong to no step


def test_timing_plugin_skips_the_xdist_controller_and_tags_workers(tmp_path) -> None:
    """Under xdist only workers record, each into its own file (the controller used to overwrite theirs)."""
    import importlib.util
    import pathlib
    import types

    spec = importlib.util.spec_from_file_location(
        "fdtdx_timing_plugin_test", pathlib.Path(__file__).with_name("fdtdx_timing_plugin.py")
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    class Manager:
        registered: list = []

        def register(self, obj, name):
            self.registered.append(obj)

    def config(**attrs):
        manager = Manager()
        manager.registered = []
        return types.SimpleNamespace(
            getoption=lambda *a, **k: "x", pluginmanager=manager, rootpath=tmp_path, **attrs
        ), manager

    cfg, manager = config(option=types.SimpleNamespace(numprocesses=4))  # controller
    plugin.configure(cfg)
    assert manager.registered == []
    cfg, manager = config(option=types.SimpleNamespace(numprocesses=None))  # plain pytest
    plugin.configure(cfg)
    assert [p.path.name for p in manager.registered] == ["x.json"]
    cfg, manager = config(option=types.SimpleNamespace(numprocesses=None), workerinput={"workerid": "gw1"})
    plugin.configure(cfg)
    assert [p.path.name for p in manager.registered] == ["x.gw1.json"]


def test_stack_options() -> None:
    """options= sets the feature fields, device.options reads them back, and old keywords still work."""
    from gdsfactory.fdtdx_stack import StackOptions

    default = build()
    assert default.options == StackOptions()
    fast = build(options=StackOptions.fast())
    assert fast.options == StackOptions.fast()
    assert (fast.averaging, fast.yee_staggered, fast.corner_lines, fast.diagnostics) == ("inverse", False, 0, False)
    assert build(corner_lines=0).options == StackOptions(corner_lines=0)  # keywords as before
    assert build(options=StackOptions(corner_lines=0)).options == StackOptions(corner_lines=0)
    with pytest.raises(TypeError, match="corner_lines"):
        build(options=StackOptions(), corner_lines=0)


def test_subpixel_normal_gradient() -> None:
    """subpixel_normal="gradient" (finite-difference Kottke normal) is valid, differentiable and differs from "voxel"."""
    out = {}
    for normal in ("gradient", "voxel"):
        objects, arrays, params = place(build(subpixel_normal=normal))
        d = objects["stack"]

        def eps_sum(t, d=d, arrays=arrays, params=params):
            a = d.write(arrays, {**params["stack"], "thickness/core": t})
            return jnp.sum(1.0 / a.inv_permittivities[:, *d.grid_slice]), a.inv_permittivities[:, *d.grid_slice]

        (_, inv), g = jax.value_and_grad(eps_sum, has_aux=True)(0.22)
        assert float(inv.min()) >= 1 / 12.0 - 1e-6 and float(inv.max()) <= 1.0 + 1e-6
        assert np.isfinite(float(g))
        out[normal] = np.asarray(inv)
    assert np.max(np.abs(out["gradient"] - out["voxel"])) > 1e-3  # the normals differ at the slanted walls
    objects, arrays, params = place(build(subpixel_normal="none"))
    with pytest.raises(ValueError, match="subpixel_normal"):
        objects["stack"].write(arrays, params["stack"])


# ---------------------------------------------------------------- derived shapes: model fixes (round 3, 2.5-fix-model)


def _exact_device(component, expr, **level) -> StackDevice:
    """A StackDevice drawing a layer expression (exact derived shapes), the box 0.3 um around the polygons."""
    return StackDevice.from_layer_stack(
        component, _core_of(expr, **level), CORE_ONLY, background="air", name="s", included_layers=["core"],
        boundary_offset=(0.3, -0.3, 0.3, -0.3, 0.0, 0.0),
    )


def test_a_far_collinear_edge_is_not_a_crossing_partner() -> None:
    """A joint of abutting layers is built from the edges that touch it: a polygon listed first with an edge on
    the same line far away does not take its place, and the device passes its own check."""
    wg = [_rect(0.0, 2.0, 1.02, 2.5), _rect(0.0, 0.0, 1.02, 0.5)]
    c = _two_layers(wg, [_rect(1.0, 0.0, 2.0, 0.5)], etch_layer=SLAB)
    d = _exact_device(c, LogicalLayer(layer=WG) | LogicalLayer(layer=SLAB), bias=-0.02)
    assert len(d.derived_layers) == 1
    d.pack(c, check=True)
    d.check_topology(d._pack(c))
    d.retopologize(c).pack(c, check=True)


def test_a_joint_that_opens_into_a_step_is_reported() -> None:
    """WG | SLAB with collinear tops: moving WG's top off SLAB's makes the joint a step, which the distance
    leaves out; check_topology and pack(check=True) report it (a device built on the moved polygons is fine)."""
    c = _two_layers([_rect(0.0, 0.0, 1.0, 0.5)], [_rect(1.0, 0.0, 2.0, 0.5)], etch_layer=SLAB)
    d = _exact_device(c, LogicalLayer(layer=WG) | LogicalLayer(layer=SLAB), bias=-0.02)
    d.check_topology(d._pack(c))
    for top in (0.54, 0.6):
        moved = _two_layers([_rect(0.0, 0.0, 1.0, top)], [_rect(1.0, 0.0, 2.0, 0.5)], etch_layer=SLAB)
        with pytest.raises(ValueError, match="joint of two layers' collinear edges opened into a step"):
            d.check_topology(d._pack(moved))
        with pytest.raises(ValueError, match="opened into a step"):
            d.pack(moved, check=True)
        new = d.retopologize(moved)
        new.check_topology(new._pack(moved))
    # a small move (below the tolerance) is not a step
    nudged = _two_layers([_rect(0.0, 0.0, 1.0, 0.5 + 1e-4)], [_rect(1.0, 0.0, 2.0, 0.5)], etch_layer=SLAB)
    d.check_topology(d._pack(nudged))


@pytest.mark.parametrize("folded_first", [True, False])
def test_folded_subtrees_do_not_change_the_overlap_test(folded_first: bool) -> None:
    """(ETCH & M1) | WG with no M1 polygons: the folded ETCH subtree leaves its literal before WG's; the
    overlapping WG polygons still get their union as exact shape (no slit between them at a bias)."""
    folded = LogicalLayer(layer=ETCH) & LogicalLayer(layer=METAL)
    wg = LogicalLayer(layer=WG)
    expr = folded | wg if folded_first else wg | folded
    c = _two_layers(OVERLAP, [_rect(0.2, 0.1, 0.3, 0.2)])
    d, w = _weights_of(c, _core_of(expr, bias=-0.01), CORE_ONLY, (2.4, 0.9, 0.3), center=(1.0, 0.25), z_center=0.11)
    assert len(d.derived_layers) == 1
    row = w["si"][cell(d, 0.1, 0, 0)[0] : cell(d, 1.9, 0, 0)[0], cell(d, 0, 0.25, 0)[1], cell(d, 0, 0, 0.11)[2]]
    assert float(row.min()) > 0.999


def test_the_build_and_the_check_share_one_error_width() -> None:
    """A strip 3.8 nm wide that node clustering drops: the build accepts it (below twice the tolerance in
    width) and the check on the unchanged component accepts it too."""
    for gap in (0.0019, 0.0025, 0.0038):
        c = _two_layers([_rect(0.0, 0.0, 2.0, 0.5)], [_rect(1.4, -0.5, 2.0 - gap, 1.0)])
        for tri in ([(2.0 - gap / 2, 0.5), (2.5, 1.2), (1.5, 1.2)], [(2.0 - gap / 2, 0.0), (1.5, -0.7), (2.5, -0.7)]):
            c.add_polygon(tri, layer=SLAB)
        expr = LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH) - LogicalLayer(layer=SLAB)
        d = _exact_device(c, expr)
        d.pack(c, check=True)


def test_winding_numbers_match_the_dense_count_and_scale() -> None:
    """The y-chunked winding numbers equal the dense count, and 27k edges take seconds (the dense count took 20 s)."""
    import time

    from gdsfactory.fdtdx_geometry import _winding_numbers

    rng = np.random.default_rng(0)
    a, b = rng.uniform(-1, 1, (300, 2)), rng.uniform(-1, 1, (300, 2))
    pts = rng.uniform(-1.2, 1.2, (3000, 2))
    px, py = pts[:, None, 0], pts[:, None, 1]
    x1, y1, x2, y2 = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
    left = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
    dense = ((y1 <= py) & (y2 > py) & (left > 0)).sum(1) - ((y2 <= py) & (y1 > py) & (left < 0)).sum(1)
    assert np.array_equal(_winding_numbers(pts, a, b, chunk=256), dense)

    n = 27200
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t)], axis=1)
    probes = rng.uniform(-1.1, 1.1, (n, 2))
    start = time.perf_counter()
    inside = _winding_numbers(probes, ring, np.roll(ring, -1, axis=0))
    assert time.perf_counter() - start < 10.0
    assert np.mean((inside != 0) != (np.linalg.norm(probes, axis=1) < 1.0)) < 1e-3  # polygon vs circle: edge cases only


def test_exact_mode_at_zero_bias_is_close_to_composite() -> None:
    """Known limit (R3-model-4): where two corners of one derived level share a voxel (here the tip of the
    etch lies on the waveguide's edge) the exact shape's corner model keeps the half-plane, while the composite
    model, with one literal per layer, is exact. At bias 0 both draw the same shape; the excess is bounded here
    so a regression shows (measured: totals 0.62 vs 0.35, worst voxel 0.256 vs 0.000)."""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(1)

    def star(cx, cy, n, r0, r1):
        phi = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(r0, r1, n)
        return [(cx + ri * math.cos(p), cy + ri * math.sin(p)) for ri, p in zip(r, phi)]

    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}
    for _ in range(2):
        a, b = star(0.5, 0.5, 7, 0.25, 0.5), star(0.7, 0.6, 6, 0.2, 0.4)
        rng.uniform(-0.02, 0.02, 2)
    c = gf.Component()
    c.add_polygon(a, layer=WG)
    c.add_polygon(b, layer=ETCH)
    derived = {"derived_layer": LogicalLayer(layer=WG), "zmin": 0.0, "thickness": 0.22}
    ls = LayerStack(layers={
        "one": LayerLevel(layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), material="si", mesh_order=1,
                          **derived),
        "two": LayerLevel(layer=LogicalLayer(layer=WG) | LogicalLayer(layer=ETCH), material="sio2", mesh_order=2,
                          **derived),
    })
    pa, pb = shapely.Polygon(a), shapely.Polygon(b)
    one = pa.difference(pb)
    two = pa.union(pb).difference(one)
    errors = {}
    for mode in ("composite", "exact"):
        d, w = _weights_of(c, ls, mats, (1.4, 1.4, 0.3), center=(0.6, 0.55), z_center=0.11, derived_shapes=mode)
        f_one, f_two = _area_fractions(d, one), _area_fractions(d, two)
        k = cell(d, 0, 0, 0.11)[2]
        errors[mode] = sum(
            np.abs(w[name][:, :, k] - f) for name, f in (("si", f_one), ("sio2", f_two), ("air", 1 - f_one - f_two))
        )
    assert errors["composite"].sum() < 0.5
    assert errors["exact"].sum() < 2.0 * errors["composite"].sum()
    assert float((errors["exact"] - errors["composite"]).max()) < 0.3


@pytest.mark.parametrize("bias_b", [0.005, 0.0])
def test_levels_sharing_a_crossing_round_their_own_tips(bias_b: float) -> None:
    """WG - ETCH (si, bias 5 nm) and WG & ETCH (sio2) cross at voxel centres: with equal offsets each derived
    shape rounds its own tip at the shared crossing, in its own cone (R3-accuracy-3: the voxel fell back to the
    half-plane, 0.20 off); the si fill matches the round-join offset of WG - ETCH."""
    shapely = pytest.importorskip("shapely")

    def cells(pts):
        return [(RES * x, RES * y) for x, y in pts]

    wg = cells([(-45.5, -13.5), (-29.5, -13.5), (-29.5, -1.5), (-45.5, -1.5)])
    et = cells([(-35.5, -9.5), (-24.5, -9.5), (-24.5, 3.5), (-35.5, 3.5)])
    c = _two_layers([wg], [et])
    derived = {"derived_layer": LogicalLayer(layer=WG), "zmin": 0.0, "thickness": 0.22}
    ls = LayerStack(layers={
        "a": LayerLevel(layer=LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH), material="si", mesh_order=1,
                        bias=0.005, **derived),
        "b": LayerLevel(layer=LogicalLayer(layer=WG) & LogicalLayer(layer=ETCH), material="sio2", mesh_order=2,
                        bias=bias_b, **derived),
    })
    mats = {"air": MATERIALS["air"], "si": MATERIALS["si"], "sio2": MATERIALS["sio2"]}
    region = shapely.Polygon(wg).difference(shapely.Polygon(et)).buffer(0.005, quad_segs=256)
    errors = {}
    for mode in ("exact", "composite"):
        d, w = _weights_of(c, ls, mats, (0.8, 0.8, 0.3), center=(-0.62, -0.1), z_center=0.11, derived_shapes=mode)
        k = cell(d, 0, 0, 0.11)[2]
        errors[mode] = np.abs(w["si"][:, :, k] - _area_fractions(d, region))
    assert float(errors["exact"].max()) < 1e-3, float(errors["exact"].max())
    assert float((errors["exact"] - errors["composite"]).max()) < 1e-4


def test_placement_paints_its_mapping_without_the_corner_model() -> None:
    """fdtdx's material mapping (the initial painting, overwritten by apply_stack) skips the corner pass, which
    otherwise compiled and autotuned on its own at the first placement (R3-accuracy-2: 42 s at 260x180x90)."""
    from gdsfactory.fdtdx_stack import record_timings

    with record_timings() as timings:
        objects, arrays, params = place(build())
    assert "corner_pass" not in timings.by_leaf()
    assert "weights" in timings.by_leaf()
    d = objects["stack"]
    assert d.corner_lines == 2  # the device itself keeps its corner model
    names, w = d.material_weights(params["stack"], apply_roughness=False)
    agree = np.asarray(jnp.argmax(w, axis=0) == d.get_material_mapping())
    assert agree.mean() > 0.99  # only corner voxels may round differently


# ---------------------------------------------------------------- the FDTD solve (timed by the timing plugin)


def test_fdtd_solve_forward_and_gradient() -> None:
    """A small solve through a StackDevice: forward, and a jax.grad of the detected energy with respect to the
    core thickness. Both run inside timed steps (``fdtd_solve_forward``, ``fdtd_solve_gradient``: eager calls
    whose outputs are waited for, compile and backward pass included), which the timing plugin reports."""
    from gdsfactory.fdtdx_timing import timed_step

    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "clad")})
    d = StackDevice.from_layer_stack(
        _straight(0.0, width=0.5), ls, MATERIALS, background="air", name="stack", included_layers=["core"],
        boundary_offset=(0.0, 0.0, 0.3, -0.3, 0.3, -0.2),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    source, c1 = d.create_point_dipole_source_at("o1", wave_character=wave, polarization="y", port_offset=0.2)
    probe, c2 = d.create_energy_detector_at("o2", (2, 3), reduce_volume=True, port_offset=0.2)
    volume, constraints = d.get_simulation_volume()
    config = fdtdx.SimulationConfig(
        time=40e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=RES * 1e-6),
        backend=jax.default_backend(),
        dtype=jnp.float32,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=8),
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, d, source, probe], config=config, constraints=[*constraints, *c1, *c2], key=key
    )
    params = {**params, **init_stack_params(objects)}

    def energy(thickness):
        p = {**params, "stack": {**params["stack"], "thickness/core": thickness}}
        new, objs, _ = apply_stack(arrays, objects, p, key)
        _, final = fdtdx.run_fdtd(arrays=new, objects=objs, config=config, key=key, show_progress=False)
        return 1e24 * jnp.sum(final.detector_states[probe.name]["energy"])  # SI energies are ~1e-25 J

    with timed_step("fdtd_solve_forward") as step:
        value = step.done(energy(0.22))
    assert np.isfinite(float(value)) and float(value) > 0
    with timed_step("fdtd_solve_gradient") as step:
        value, grad = step.done(jax.value_and_grad(energy)(0.22))
    assert np.isfinite(float(grad)) and float(grad) != 0.0


def test_mode_objects_on_a_stack_see_the_stack_permittivity() -> None:
    """A mode source and a mode detector on a stack's port solve their modes on the permittivity apply_stack
    wrote: fdtdx (0.6.3) applies objects that overlap none of its own Devices only in place_objects, on the
    dominant-material painting. A second call with the same parameters reuses them; a thicker core solves again."""
    ls = LayerStack(layers={k: stack().layers[k] for k in ("box", "core", "clad")})
    d = StackDevice.from_layer_stack(
        _straight(0.0), ls, MATERIALS, background="air", name="stack", included_layers=["core"],
        extend_ports=(("o1", 0.4),), boundary_offset=(0, 0, 0.6, -0.6, 0.4, -0.3),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    made = [
        d.create_mode_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=-0.2),
        d.create_mode_overlap_detector_at("o1", (2, 3), wave_characters=(wave,), port_offset=0.2),
    ]
    source, detector = (obj.name for obj, _ in made)
    objects, arrays, params = _place_own_volume(d, made)
    key = jax.random.PRNGKey(0)
    placed = np.asarray(objects[source]._inv_permittivity)  # what place_objects solved the mode on

    arrays, objects2, info = apply_stack(arrays, objects, params, key)
    src = objects2[source]
    np.testing.assert_array_equal(np.asarray(src._inv_permittivity), np.asarray(arrays.inv_permittivities[:, *src.grid_slice]))
    assert not np.array_equal(placed, np.asarray(src._inv_permittivity))  # the painting is not the stack
    assert sorted(info["stack_reapplied"]) == sorted([source, detector]) and info["stack_reused"] == []

    _, objects3, info = apply_stack(arrays, objects2, params, key)
    assert info["stack_reapplied"] == [] and sorted(info["stack_reused"]) == sorted([source, detector])

    thicker = {**params, "stack": {**params["stack"], "thickness/core": 0.24}}
    arrays4, objects4, info = apply_stack(arrays, objects3, thicker, key)
    assert sorted(info["stack_reapplied"]) == sorted([source, detector])
    src = objects4[source]
    np.testing.assert_array_equal(np.asarray(src._inv_permittivity), np.asarray(arrays4.inv_permittivities[:, *src.grid_slice]))


# ---------------------------------------------------------------- averaging="subpixel_full" (complete tensors)


def _rotation_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _tensor_material(eps: np.ndarray) -> fdtdx.Material:
    return fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row) for row in eps))


# anisotropic test materials: a rotated uniaxial crystal (full tensor) and a diagonal one
ANISO = _rotation_z(25.0) @ np.diag([9.0, 12.0, 14.0]) @ _rotation_z(25.0).T
ANISO[0, 2] = ANISO[2, 0] = 0.8  # also coupling z
DIAG = np.diag([2.1, 2.6, 3.2])
ANISO_MATERIALS = {"air": _tensor_material(DIAG), "si": _tensor_material(ANISO)}


def _full(polygon, materials=CORE_ONLY, size=(2.2, 1.6, 0.3), z_center=0.11, **kwargs):
    """A placed "subpixel_full" device of a polygon on the core: device, written box (9, nx, ny, nz), arrays, params."""
    objects, arrays, params = _placed(
        _polygons(polygon), _core(), materials, size, (0.0, 0.0), z_center=z_center, averaging="subpixel_full", **kwargs
    )
    written = np.asarray(_box(_apply(objects, arrays, params), objects["s"], "inv_permittivities"))
    return objects["s"], written, arrays, params


def _laminate64(eps: np.ndarray, f: np.ndarray, n: np.ndarray) -> np.ndarray:
    from gdsfactory.fdtdx_subpixel import laminate

    with jax.enable_x64(True):
        return np.asarray(laminate(np.asarray(eps), jnp.asarray(f, jnp.float64), jnp.asarray(n, jnp.float64)))


def _yee_point(d, c, i, j, k):
    """Stack point (um) of E_c of voxel (i, j, k)."""
    nx, ny, nz = d.grid_shape
    sx, sy, sz = _YEE_E_SHIFTS[c]
    return (
        d.center[0] + (i + 0.5 - nx / 2 + sx) * RES,
        d.center[1] + (j + 0.5 - ny / 2 + sy) * RES,
        d.z_center + (k + 0.5 - nz / 2 + sz) * RES,
    )


ROW = ((0, 1, 2), (3, 4, 5), (6, 7, 8))  # fdtdx's row-major entries of each E component's row


def test_homogeneous_anisotropic_box_writes_fdtdxs_own_inverse() -> None:
    """A box of one fully anisotropic material (rotated principal axes, off-diagonal entries coupling z) gets, at
    every Yee position, the rows of the material's inverse: fdtdx's own (its volume of the same material, inverted
    in float32 by fdtdx) and numpy's float64 inverse, in fdtdx's row-major order."""
    mats = {"air": _tensor_material(ANISO), "si": _tensor_material(ANISO)}
    objects, arrays, params = _placed(
        _polygons(_rect(-0.5, -0.3, 0.5, 0.3)), _core(), mats, (1.2, 0.8, 0.3), (0.0, 0.0), z_center=0.11,
        averaging="subpixel_full",
    )
    d = objects["s"]
    assert arrays.inv_permittivities.shape[0] == 9
    written = np.asarray(_box(_apply(objects, arrays, params), d, "inv_permittivities"))
    want = np.linalg.inv(ANISO).reshape(9)
    np.testing.assert_allclose(written.reshape(9, -1).T, np.broadcast_to(want, (written[0].size, 9)), rtol=2e-6, atol=1e-7)
    # fdtdx's own inverse of the same material: place_objects painted the box with it (its dominant material)
    fdtdx_inv = np.asarray(_box(arrays, d, "inv_permittivities"))
    np.testing.assert_allclose(written, fdtdx_inv, rtol=2e-6, atol=1e-7)


def test_material_contract_rejects_what_it_cannot_write() -> None:
    """Off-diagonal tensors need "subpixel_full" (the other averagings would drop them); non-symmetric, indefinite
    and non-finite tensors, and the reserved placeholder name, are rejected when the device is built."""
    poly = _polygons(_rect(-0.5, -0.3, 0.5, 0.3))
    with pytest.raises(ValueError, match="subpixel_full"):
        StackDevice.from_layer_stack(poly, _core(), ANISO_MATERIALS, background="air", included_layers=["core"])
    bad = np.diag([2.0, 3.0, 4.0])
    bad[0, 1] = 0.5
    with pytest.raises(ValueError, match="non-symmetric"):
        StackDevice.from_layer_stack(
            poly, _core(), {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=tuple(map(tuple, bad)))},
            background="air", included_layers=["core"], averaging="subpixel_full",
        )
    with pytest.raises(ValueError, match="reserved"):
        StackDevice.from_layer_stack(
            poly, _core(), {**CORE_ONLY, "__diagonal_anisotropy__": fdtdx.Material()}, background="air",
            included_layers=["core"],
        )
    # diagonal anisotropy stays allowed with "subpixel" (each E component takes its diagonal entry)
    StackDevice.from_layer_stack(
        poly, _core(), {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=(11.0, 12.0, 13.0))},
        background="air", included_layers=["core"],
    )


PLANAR_FULL_TOL = 3e-6  # float32: measured 1.2e-6 on scalar materials


@pytest.mark.parametrize(("materials", "degrees"), [(CORE_ONLY, 30.0), (ANISO_MATERIALS, 30.0), (ANISO_MATERIALS, 0.0)])
def test_planar_walls_get_kottkes_anisotropic_laminate_in_every_entry(materials, degrees: float) -> None:
    """At every Yee position on a straight wall (away from corners and from the core's top and bottom), each E
    component's row is the row of the inverse of Kottke's laminate of the two materials' full tensors along the
    wall's normal, with the exact fill fraction of the voxel square: all three entries, for isotropic, diagonal
    and fully anisotropic materials, oblique and on the grid."""
    shapely = pytest.importorskip("shapely")
    rect = _rotated(_rect(-0.8, -0.3, 0.8, 0.3), degrees, np.zeros(2))
    d, full, _, _ = _full(rect, materials)
    poly = shapely.Polygon(next(iter(_polygons(rect).get_polygons_points().values()))[0])
    from gdsfactory.fdtdx_subpixel import material_tensor

    eps = np.stack([material_tensor(materials["air"]), material_tensor(materials["si"])])
    th = math.radians(degrees)
    c_, s_ = math.cos(th), math.sin(th)
    nx, ny, nz = d.grid_shape
    k = cell(d, 0, 0, 0.11)[2]
    errors = []
    for c in range(3):
        for i in range(nx):
            for j in range(ny):
                x, y, z = _yee_point(d, c, i, j, k)
                f = poly.intersection(shapely.box(x - RES / 2, y - RES / 2, x + RES / 2, y + RES / 2)).area / RES**2
                if not 1e-3 < f < 1 - 1e-3:
                    continue
                u, v = x * c_ + y * s_, -x * s_ + y * c_
                if abs(u) > 0.65 and abs(v) > 0.15 or abs(u) <= 0.65 and abs(v) <= 0.2:
                    continue  # near a corner, or not near a wall
                n = np.array([c_, s_, 0.0]) if abs(u) > 0.65 else np.array([-s_, c_, 0.0])
                want = np.linalg.inv(_laminate64(eps, np.array([1 - f, f]), n))[c]
                errors.append(float(np.abs(full[list(ROW[c]), i, j, k] - want).max()))
    assert len(errors) > 150
    assert max(errors) < PLANAR_FULL_TOL, max(errors)


def test_films_thinner_than_a_voxel_are_laminates_of_their_three_materials() -> None:
    """Horizontal layers of three materials, one an 8 nm film inside a voxel (two opposing faces, each seen):
    the voxel's tensor is the laminate along z of its exact fractions (Ez, Ex rows; fully anisotropic materials)."""
    eps_c = _rotation_z(-40.0) @ np.diag([4.0, 5.0, 7.0]) @ _rotation_z(-40.0).T
    mats = {**ANISO_MATERIALS, "c": _tensor_material(eps_c)}
    # the film [0.111, 0.119] lies inside one voxel at the Ez positions ([0.11, 0.13]) and the Ex ones ([0.10, 0.12])
    layers = {
        "a": LayerLevel(layer=WAFER, zmin=-0.1, thickness=0.211, material="air", mesh_order=3),
        "film": LayerLevel(layer=WAFER, zmin=0.111, thickness=0.008, material="c", mesh_order=1),
        "b": LayerLevel(layer=WAFER, zmin=0.119, thickness=0.2, material="si", mesh_order=2),
    }
    ls = LayerStack(layers=layers)
    objects, arrays, params = _placed(
        _polygons(_rect(-0.3, -0.3, 0.3, 0.3)), ls, mats, (0.4, 0.4, 0.2), (0.0, 0.0), z_center=0.11,
        averaging="subpixel_full",
    )
    d = objects["s"]
    full = np.asarray(_box(_apply(objects, arrays, params), d, "inv_permittivities"))
    from gdsfactory.fdtdx_subpixel import material_tensor

    eps = np.stack([material_tensor(mats[m]) for m in ("air", "c", "si")])
    nx, ny, nz = d.grid_shape
    for c in (0, 2):
        z0 = _yee_point(d, c, 0, 0, 0)[2] - RES / 2
        k = int(math.floor((0.115 - z0) / RES))  # the voxel holding the film
        lo, hi = z0 + k * RES, z0 + (k + 1) * RES
        f = np.array([
            max(0.0, min(0.111, hi) - lo), max(0.0, min(0.119, hi) - max(0.111, lo)), max(0.0, hi - max(0.119, lo))
        ]) / RES
        assert f[1] == pytest.approx(0.4, abs=1e-6)
        want = np.linalg.inv(_laminate64(eps, f, np.array([0.0, 0.0, 1.0])))[c]
        np.testing.assert_allclose(full[list(ROW[c]), nx // 2, ny // 2, k], want, atol=3e-6)


@pytest.mark.parametrize("xy_subsamples", [1, 2])
def test_junction_voxels_are_symmetric_positive_definite_and_within_the_wiener_bounds(xy_subsamples: int) -> None:
    """On a device with corners of every kind (derived etch, slab, sloped walls), every voxel's effective tensor at
    every Yee position is symmetric positive definite and between the harmonic and arithmetic means of its
    materials (in the matrix sense); the junction voxels are the closure's, with a converged solve. Also with two
    sample squares per voxel, whose junction sub-cells are painted with half-planes and made to hold the
    voxel's materials."""
    d = build(averaging="subpixel_full", xy_subsamples=xy_subsamples)
    objects, arrays, params = place(d)
    d = objects["stack"]
    from gdsfactory.fdtdx_subpixel import from_six

    names, eps = d._material_table()
    inv_eps = np.linalg.inv(eps)
    worst = 0.0
    for shift in _YEE_E_SHIFTS:
        g = d._interfaces(params["stack"], None, True, shift)
        inv6, status = d._complete_tensor(g, d.junction_voxels)
        assert int(status["junctions"]) > 50 and int(status["junction_overflow"]) == 0
        assert float(status["closure_residual"]) < 1e-5
        inv = np.asarray(from_six(inv6)).reshape(-1, 3, 3)
        w = np.moveaxis(np.asarray(g.weights), 0, -1).reshape(-1, len(names))
        mixed = w.max(axis=1) < 1 - 1e-6
        e = np.linalg.inv(inv[mixed].astype(np.float64))
        arithmetic = np.einsum("vm,mij->vij", w[mixed], eps)
        harmonic = np.linalg.inv(np.einsum("vm,mij->vij", w[mixed], inv_eps))
        scale = np.abs(arithmetic).max(axis=(1, 2))
        assert np.linalg.eigvalsh(e).min() > 0
        worst = max(
            worst,
            float((-np.linalg.eigvalsh(arithmetic - e).min(axis=1) / scale).max()),
            float((-np.linalg.eigvalsh(e - harmonic).min(axis=1) / scale).max()),
        )
    assert worst < 1e-5, worst


def test_a_corner_entering_a_voxel_changes_its_tensor_continuously() -> None:
    """A rectangle's corner moved across a voxel's side in shrinking steps: the written entries change by
    amounts that shrink with the step (no jump where the second face appears, nor where the closure takes over
    from the laminate)."""
    x_side = 0.3  # a voxel side (an integer number of cells from the box center)

    def rect(x1: float) -> list:
        return _rect(-0.8 + 0.01, -0.3 + 0.01, x1, 0.3 + 0.01)

    objects, arrays, params = _placed(
        _polygons(rect(x_side + 0.004)), _core(), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0), z_center=0.11,
        averaging="subpixel_full",
    )
    d = objects["s"]
    written = jax.jit(lambda p: d.write(arrays, p).inv_permittivities[:, *d.grid_slice])
    changes = []
    for step in (4e-3, 4e-4, 4e-5):
        values = [np.asarray(written({**params["s"], **d.pack(_polygons(rect(x_side + dx)))})) for dx in (-step, step)]
        changes.append(float(np.abs(values[1] - values[0]).max()))
    assert changes[1] < 0.3 * changes[0] and changes[2] < 0.3 * changes[1], changes


def test_capacity_failures_raise_or_warn() -> None:
    """Junction voxels beyond junction_voxels are a coverage failure: an error where the result is concrete,
    NaN under jit (so it cannot be accepted), or, with interface_failure="warn", a warning and the laminate."""
    rect = _rotated(_rect(-0.8, -0.3, 0.8, 0.3), 30.0, np.zeros(2))
    objects, arrays, params = _placed(
        _polygons(rect), _core(), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0), z_center=0.11, averaging="subpixel_full",
    )
    d = objects["s"].aset("junction_voxels", 8)
    status = d.interface_status(params["s"])
    assert status["junction_overflow"] > 0 and status["junction_voxels"] == 8
    with pytest.raises(ValueError, match="junction_voxels"):
        d.write(arrays, params["s"])
    nan = jax.jit(lambda p: d.write(arrays, p).inv_permittivities[:, *d.grid_slice])(params["s"])
    assert bool(jnp.all(jnp.isnan(nan)))
    lenient = d.aset("interface_failure", "warn")
    with pytest.warns(StackDiagnosticWarning, match="junction_voxels"):
        out = lenient.write(arrays, params["s"])
    assert bool(jnp.all(jnp.isfinite(out.inv_permittivities)))


def test_uncertified_corner_columns_are_reported() -> None:
    """Two corners closer than the corner model resolves (rectangles half a voxel apart): their columns are
    painted partly with one half-plane per level. That is reported (interface_status, the write's status, a
    diagnostic when placed), not a failure; a lone rotated rectangle has none."""
    two = [_rect(-0.5, -0.3, 0.0137, 0.3), _rect(0.0237, -0.3, 0.5, 0.3)]
    with pytest.warns(StackDiagnosticWarning, match="not certified"):
        objects, arrays, params = _placed(
            _polygons(*two), _core(), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0), z_center=0.11,
            averaging="subpixel_full",
        )
    d = objects["s"]
    status = d.interface_status(params["s"])
    assert status["corner_uncertified"] > 0 and status["junction_overflow"] == 0
    _, written = d.write_with_status(arrays, params["s"])
    assert int(written["corner_uncertified"]) == status["corner_uncertified"] and not bool(written["failed"])
    rect = _rotated(_rect(-0.8, -0.3, 0.8, 0.3), 30.0, np.zeros(2))
    objects, _, params = _placed(
        _polygons(rect), _core(), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0), z_center=0.11, averaging="subpixel_full",
    )
    assert objects["s"].interface_status(params["s"])["corner_uncertified"] == 0


def test_subpixel_full_needs_nine_components() -> None:
    """Writing "subpixel_full" into 3-component arrays (a device built for "subpixel") is an error that says how to
    build it."""
    objects, arrays, params = _placed(
        _polygons(_rect(-0.8, -0.3, 0.8, 0.3)), _core(), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0), z_center=0.11
    )
    d = objects["s"].aset("averaging", "subpixel_full")
    with pytest.raises(ValueError, match="9-component"):
        d.write(arrays, params["s"])


def test_subpixel_full_couples_z_at_sloped_sidewalls() -> None:
    """A 10 degree sidewall tilts the wall's normal out of the plane: the xz entries are nonzero along walls normal
    to x, with the sign of n_x n_z (opposite on the two sides)."""
    d, full, _, _ = _full(_rect(-0.8, -0.3, 0.8, 0.3), size=(2.2, 1.6, 0.3))
    assert float(np.abs(full[[2, 6]]).max()) < 1e-6  # vertical walls: no z coupling
    objects, arrays, params = _placed(
        _polygons(_rect(-0.8, -0.3, 0.8, 0.3)), _core(sidewall_angle=10.0), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0),
        z_center=0.11, averaging="subpixel_full",
    )
    d = objects["s"]
    full = np.asarray(_box(_apply(objects, arrays, params), d, "inv_permittivities"))
    k = cell(d, 0, 0, 0.11)[2]
    j = cell(d, 0, 0, 0)[1]
    i_left, i_right = cell(d, -0.8, 0, 0)[0], cell(d, 0.8, 0, 0)[0]
    left = full[2, i_left - 1 : i_left + 2, j, k]
    right = full[2, i_right - 1 : i_right + 2, j, k]
    assert float(np.abs(left).max()) > 1e-3 and float(np.abs(right).max()) > 1e-3
    assert np.sign(left[np.argmax(np.abs(left))]) == -np.sign(right[np.argmax(np.abs(right))])


def test_subpixel_full_gradients_match_finite_differences() -> None:
    """Directional derivatives of the written tensor (all nine entries) in a vertex of an oblique rectangle
    (moving a corner: junction voxels), its thickness and its sidewall angle: the reverse-mode gradient of the
    entries weighted by their own forward-mode derivative (``|J v|^2``, so no cancellation) matches central
    differences. The steps are small against the junction weight's transition (a corner entering a voxel
    reaches the closure within about 0.03 cells) and the geometry is off coincidences with the grid: a surface
    on a voxel face, or on a face of the closure's sub-cells (quarters of a voxel), is a kink, where the
    derivative is one of the sides' (or their mean) and central differences are first order."""
    rect = _rotated(_rect(-0.8, -0.3, 0.8, 0.3), 30.0, np.asarray([0.0137, -0.0071]))
    core = LayerLevel(layer=WG, zmin=0.0063, thickness=0.2071, material="si", mesh_order=1, sidewall_angle=10.0)
    objects, arrays, params = _placed(
        _polygons(rect), LayerStack(layers={"core": core}), CORE_ONLY, (2.2, 1.6, 0.3), (0.0, 0.0), z_center=0.11,
        averaging="subpixel_full",
    )
    d = objects["s"]

    @jax.jit
    def written(p):
        return d.write(arrays, p).inv_permittivities[:, *d.grid_slice]

    @jax.jit
    def loss(p, weights):
        return jnp.sum(weights * written(p))

    p0 = params["s"]
    key = "vertices/WG"
    directions = {
        key: jnp.zeros_like(p0[key]).at[2].set(jnp.asarray([0.6, 0.8])),  # one corner, obliquely
        "thickness/core": jnp.ones_like(p0["thickness/core"]),
        "sidewall_angle/core": jnp.ones_like(p0["sidewall_angle/core"]),
    }
    # central differences of the vertex direction: -2.0 % at 1e-4 um, -0.12 % at 3e-5, -0.05 % at 1e-5
    steps = {key: 2e-5, "thickness/core": 2e-5, "sidewall_angle/core": 0.005}
    for name, direction in directions.items():
        tangent = {k: jnp.zeros_like(v) for k, v in p0.items()}
        tangent[name] = direction
        weights = jax.jvp(written, (p0,), (tangent,))[1]
        ad = float(jnp.sum(jax.grad(loss)(p0, weights)[name] * direction))
        h = steps[name]
        plus = {**p0, name: p0[name] + h * direction}
        minus = {**p0, name: p0[name] - h * direction}
        fd = (float(loss(plus, weights)) - float(loss(minus, weights))) / (2 * h)
        assert ad > 0 and abs(ad - float(jnp.sum(weights**2))) <= 1e-3 * ad, (name, ad)  # reverse = forward
        assert abs(ad - fd) <= 0.01 * abs(fd), (name, ad, fd)
