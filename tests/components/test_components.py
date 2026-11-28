from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.difftest import difftest
from gdsfactory.get_factories import get_cells
from gdsfactory.serialization import clean_value_json

cells = get_cells([gf.components])

pad_array_layer = partial(gf.c.pad_array, layer="M1")
pad_array_size = partial(gf.c.pad_array, layer="M1", size=(100, 100))
taper_with_trenches = partial(gf.c.taper, cross_section="rib_with_trenches")

bend_euler_angular_resolution = partial(
    gf.components.bend_euler, radius=10, angle=90, angular_step=20
)

cells.update(
    {
        "pad_array_layer": pad_array_layer,
        "pad_array_size": pad_array_size,
        "taper_with_trenches": taper_with_trenches,
        "bend_euler_angular_resolution": bend_euler_angular_resolution,
    }
)


skip_test = {
    "coupler_bent",
    "component_sequence",
    "extend_ports_list",
    "grating_coupler_elliptical_lumerical_etch70",
    "ring_double_pn",
    "straight_heater_doped_strip",  # TODO: fix this
    "straight_piecewise",
    "text_freetype",
    "version_stamp",
    "die_frame_phix",
    "dbr_tapered",
    "taper_hecken",
}
cells_to_test = set(cells.keys()) - skip_test

# 1-DBU rounding sliver in _req_straight_len vs upstream's C++ kfactory reference.
_SLIVER_TOLERANT_CELLS = {"spiral_racetrack_fixed_length"}

# Upstream's gdsfactory-test-data reference used odeint with discrete sol < wmax
# truncation on a 0.1 um x-grid (stopping early at width 4.994 um instead of 5.0 um);
# gdsfactoryx integrates continuously to w2 for exact endpoint width and AD/JIT parity.
# Kept out of skip_test so test_settings[taper_adiabatic] still runs.
_SKIP_GDS_DIFFTEST = {"taper_adiabatic"}


default_container_arguments: dict[str, dict[str, Any]] = dict(
    bbox=dict(component="mmi1x2", layer="SLAB90"),
    pack_doe=dict(doe="mmi1x2", settings=dict(length_mmi=(100, 200))),
    pack_doe_grid=dict(doe="mmi1x2", settings=dict(length_mmi=(100, 200))),
    add_fiber_array_optical_south_electrical_north=dict(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal="metal_routing",
        pad_pitch=100,
    ),
)


@pytest.fixture(params=cells_to_test)
def component_name(request: pytest.FixtureRequest) -> Any:
    return request.param


def get_component_with_defaults(name: str) -> gf.Component:
    """Get a component, applying default arguments if specified."""
    if name in default_container_arguments:
        return cells[name](**default_container_arguments[name])
    return cells[name]()


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    if component_name in _SKIP_GDS_DIFFTEST:
        pytest.skip(
            f"{component_name} uses continuous quadrature to w2 (upstream reference truncated early)"
        )
    component = get_component_with_defaults(component_name)
    difftest(
        component=component,
        test_name=component_name,
        dirpath=PATH.gds_ref,
        ignore_sliver_differences=component_name in _SLIVER_TOLERANT_CELLS,
    )


def test_taper_adiabatic_geometry() -> None:
    """Verify taper_adiabatic reaches exact endpoint widths and custom length."""
    c = gf.components.taper_adiabatic()
    o1 = c.ports["o1"]
    o2 = c.ports["o2"]
    assert float(o1.x) == pytest.approx(0.0)
    assert float(o1.y) == pytest.approx(0.0)
    assert float(o1.width) == pytest.approx(0.5)
    assert float(o1.orientation) == pytest.approx(180.0)
    length = float(o2.x)
    assert 0.0 < length < 200.0
    assert float(o2.y) == pytest.approx(0.0)
    assert float(o2.width) == pytest.approx(5.0)
    assert float(o2.orientation) == pytest.approx(0.0)

    polys = c.get_polygons_points(layer=(1, 0))
    assert len(polys) == 1
    pts = np.asarray(polys[0])
    at_start = pts[np.abs(pts[:, 0]) < 1e-6]
    at_end = pts[np.abs(pts[:, 0] - length) < 1e-6]
    assert len(at_start) == 2
    assert len(at_end) == 2
    assert float(at_start[:, 1].max() - at_start[:, 1].min()) == pytest.approx(
        0.5, abs=2e-3
    )
    assert float(at_end[:, 1].max() - at_end[:, 1].min()) == pytest.approx(
        5.0, abs=2e-3
    )

    c_custom = gf.components.taper_adiabatic(length=30.0)
    assert float(c_custom.ports["o2"].x) == pytest.approx(30.0, abs=2e-3)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = get_component_with_defaults(component_name)
    data_regression.check(clean_value_json(component.to_dict()))
