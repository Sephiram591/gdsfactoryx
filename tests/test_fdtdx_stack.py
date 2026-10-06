"""Tests for the differentiable fdtdx LayerStack device."""

from __future__ import annotations

import math
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")

import gdsfactory as gf  # noqa: E402
from fdtdx.dispersion import susceptibility_from_coefficients  # noqa: E402

from gdsfactory.fdtdx_stack import (  # noqa: E402
    StackDevice,
    TopRoughness,
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
        time=10e-15, resolution=RES * 1e-6, backend=jax.default_backend(), dtype=jnp.float32
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
        *(f"{k}/{lv}" for k in ("thickness", "sidewall_angle") for lv in ("box", "core", "slab", "clad", "pad")),
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
    # slot constants everywhere in the box, exact reciprocal for the reverse step
    for slot, mat in enumerate((SI_DISP, AU_DISP)):
        a1, a2, a3 = fdtdx.compute_pole_coefficients(mat.dispersion.poles, dt)
        np.testing.assert_allclose(c1[slot], a1[0], rtol=1e-6)
        np.testing.assert_allclose(c2[slot], a2[0], rtol=1e-6)
        expected = w["si"] if slot == 0 else w["metal"]
        np.testing.assert_allclose(c3[slot], expected * a3[0], rtol=1e-5, atol=1e-7 * a3[0])
    np.testing.assert_allclose(_box(arrays2, d, "dispersive_inv_c2")[:, 0], 1.0 / c2, rtol=1e-6)

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
    for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3", "dispersive_inv_c2"):
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
        time=10e-15, resolution=RES * 1e-6, backend=jax.default_backend(), dtype=jnp.float32
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


def _weights_of(component, ls, materials, size, center, z_center=0.14):
    """Material fractions of a StackDevice that fills a volume of `size` (um) around (center, z_center)."""
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
    )
    config = fdtdx.SimulationConfig(time=1e-15, resolution=RES * 1e-6, dtype=jnp.float32)
    volume = fdtdx.SimulationVolume(partial_real_shape=shape)
    objects, _, _, _, _ = fdtdx.place_objects(
        object_list=[volume, device], config=config,
        constraints=[device.place_at_center(volume)], key=jax.random.PRNGKey(0),
    )
    d = objects["s"]
    names, w = d.material_weights(init_stack_params(objects)["s"])
    return d, {n: np.asarray(w[i]) for i, n in enumerate(names)}


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
        time=10e-15, resolution=RES * 1e-6, backend=jax.default_backend(), dtype=jnp.float32
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
    made = [
        d.create_mode_plane_source_at("o1", (2, 3), wave_character=wave),
        d.create_gaussian_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=0.1),
        d.create_uniform_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=0.2),
        d.create_hard_constant_amplitude_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=0.3),
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
        src, cons = d.create_uniform_plane_source_at("o1", (2, 3), wave_character=wave, port_offset=port_offset, name="s")
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
