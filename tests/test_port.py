from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.klayout_bridge import export_gds


def test_get_ports_sort_clockwise() -> None:
    """Test that the ports are sorted clockwise.

    .. code::

            3   4
            |___|_
        2 -|      |- 5
           |      |
        1 -|______|- 6
            |   |
            8   7
    """
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    p = gf.port.get_ports_list(ref, sort_ports=True, clockwise=True)
    p1 = p[0]
    p8 = p[-1]

    assert p1.name == "o1", p1.name
    assert p1.orientation == 180, p1.orientation
    assert p8.name == "o8", p8.name
    assert p8.orientation == 270, p8.orientation


def test_get_ports_sort_counter_clockwise() -> None:
    """Test that the ports are sorted counter clockwise.

    .. code::

            4   3
            |___|_
        5 -|      |- 2
           |      |
        6 -|______|- 1
            |   |
            7   8

    """
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    p = gf.port.get_ports_list(ref, sort_ports=True, clockwise=False)
    p1 = p[0]
    p8 = p[-1]
    assert p1.name == "o6", p1.name
    assert p1.orientation == 0, p1.orientation
    assert p8.name == "o7", p8.name
    assert p8.orientation == 270, p8.orientation


def test_map_ports_layer_to_orientation() -> None:
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    ports = {p.name: p for p in ref.ports if p.name is not None}
    ports_map = gf.port.map_ports_layer_to_orientation(ports)
    assert ports_map == {
        "1_0_E0": "o6",
        "1_0_E1": "o5",
        "1_0_N0": "o3",
        "1_0_N1": "o4",
        "1_0_S0": "o8",
        "1_0_S1": "o7",
        "1_0_W0": "o1",
        "1_0_W1": "o2",
    }


@pytest.mark.parametrize("port_type", ["electrical", "optical", "placement"])
def test_rename_ports(port_type: str, data_regression: DataRegressionFixture) -> None:
    c = gf.components.nxn(port_type=port_type)
    data_regression.check(c.to_dict())


def test_port() -> None:
    p = gf.Port(name="foo", orientation=359, center=(0, 0), width=5, layer=0)
    assert p.orientation == 359


def test_select_ports_layer() -> None:
    ports = [
        gf.Port(
            name="p1", center=(0, 0), width=1, orientation=0, layer=gf.get_layer((1, 0))
        ),
        gf.Port(
            name="p2", center=(1, 0), width=1, orientation=0, layer=gf.get_layer((2, 0))
        ),
    ]

    selected = gf.port.select_ports(ports, layer=(1, 0))

    assert [port.name for port in selected] == ["p1"]


def test_select_ports_layers_excluded() -> None:
    ports = [
        gf.Port(
            name="p1", center=(0, 0), width=1, orientation=0, layer=gf.get_layer((1, 0))
        ),
        gf.Port(
            name="p2", center=(1, 0), width=1, orientation=0, layer=gf.get_layer((2, 0))
        ),
        gf.Port(
            name="p3", center=(2, 0), width=1, orientation=0, layer=gf.get_layer((3, 0))
        ),
    ]

    selected = gf.port.select_ports(ports, layers_excluded=[(2, 0), (3, 0)])

    assert [port.name for port in selected] == ["p1"]


def test_add_port_preserves_info() -> None:
    """Component.add_port(port=...) must keep the source port's info (#4554)."""
    gf.gpdk.PDK.activate()

    child = gf.Component()
    child.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    child.add_port(
        name="signal",
        center=(1, 0.5),
        width=1.0,
        orientation=0,
        layer=(1, 0),
    )
    child.ports["signal"].info["my_extra"] = "value"
    child.ports["signal"].info["tags"] = ["a", "b"]

    parent = gf.Component()
    parent.add_ref(child)
    promoted = parent.add_port(port=child.ports["signal"])

    assert promoted.info["my_extra"] == "value"
    assert promoted.info["tags"] == ["a", "b"]

    # mutable info values are deep-copied, not shared with the source port
    promoted.info["tags"].append("c")
    assert child.ports["signal"].info["tags"] == ["a", "b"]


def test_port_renaming_helpers_name_original() -> None:
    gf.gpdk.PDK.activate()

    p = gf.Port(name="o1", center=(0.0, 0.0), width=0.5, orientation=0.0, layer=1)
    assert p.name_original == "o1"
    p.name = "renamed"
    assert p.name_original == "o1"
    assert p.copy(name="copied").name_original == "o1"

    c = gf.Component()
    for i, (cx, cy, orient) in enumerate(
        [
            (0.0, 2.0, 180.0),
            (0.0, 4.0, 180.0),
            (2.0, 6.0, 90.0),
            (4.0, 6.0, 90.0),
            (6.0, 4.0, 0.0),
            (6.0, 2.0, 0.0),
            (4.0, 0.0, 270.0),
            (2.0, 0.0, 270.0),
        ],
        start=1,
    ):
        c.add_port(
            name=f"p{i}", center=(cx, cy), width=0.5, orientation=orient, layer=(1, 0)
        )
    assert all(p.name_original == p.name for p in c.ports)

    c_orient = c.copy()
    gf.port.rename_ports_by_orientation(c_orient)
    assert any(p.name.startswith("o") for p in c_orient.ports if p.name is not None)

    c_cw = c.copy()
    gf.port.auto_rename_ports(c_cw)
    assert len(c_cw.ports) == 8

    c_ccw = c.copy()
    gf.port.auto_rename_ports_counter_clockwise(c_ccw)
    assert len(c_ccw.ports) == 8

    c_side = c.copy()
    gf.port.auto_rename_ports_orientation(c_side)
    assert len(c_side.ports) == 8

    c_layer_orient = c.copy()
    gf.port.auto_rename_ports_layer_orientation(c_layer_orient)
    assert all(p.name_original is not None for p in c_layer_orient.ports)

    ports_dict = {p.name: p for p in c.ports if p.name is not None}
    cw_map = gf.port.map_ports_to_orientation_cw(ports_dict)
    ccw_map = gf.port.map_ports_to_orientation_ccw(ports_dict)
    assert set(cw_map.values()) == set(ports_dict)
    assert set(ccw_map.values()) == set(ports_dict)


def test_port_to_dict_json_serializable() -> None:
    gf.gpdk.PDK.activate()

    p = gf.Port(
        name="o1",
        center=(1.25, -3.5),
        width=0.5,
        orientation=90.0,
        layer=(1, 0),
    )
    d1 = p.to_dict()
    d2 = gf.port.to_dict(p)
    assert json.loads(json.dumps(d1))["name"] == "o1"
    assert json.loads(json.dumps(d2))["center"] == [1.25, -3.5]

    p_none = gf.Port(
        name="e1",
        center=(2.0, 4.0),
        width=None,
        orientation=None,
        layer=0,
        port_type="electrical",
    )
    d_none_1 = json.loads(json.dumps(p_none.to_dict()))
    d_none_2 = json.loads(json.dumps(gf.port.to_dict(p_none)))
    assert d_none_1["orientation"] is None
    assert d_none_1["width"] is None
    assert d_none_2["orientation"] is None
    assert d_none_2["width"] is None


def test_port_layer_zero_not_falsy() -> None:
    gf.gpdk.PDK.activate()

    ports = gf.port.port_array(cross_section="strip", layer=0, n=2)
    assert all(int(p.layer) == 0 for p in ports)

    c = gf.Component()
    c.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=0)
    c.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=(1, 0))
    markers = gf.port.read_port_markers(c, layers=[0])
    assert 0 in markers.polygons
    assert int(gf.get_layer((1, 0))) not in markers.polygons


def test_port_membership_component_and_reference() -> None:
    gf.gpdk.PDK.activate()

    sub = gf.components.straight(length=10)
    assert sub.ports["o1"] in sub.ports

    top = gf.Component()
    ref = top << sub
    p_o1 = ref.ports["o1"]
    assert p_o1 in ref.ports
    assert p_o1.copy() in ref.ports


def test_port_none_orientation_and_width(tmp_path: Path) -> None:
    gf.gpdk.PDK.activate()

    p = gf.Port(
        name="pad1",
        center=(10.0, 20.0),
        width=None,
        orientation=None,
        layer=0,
        port_type="electrical",
    )
    assert p.orientation is None
    assert p.width is None
    assert np.isclose(float(p.dx), 10.0)
    assert np.isclose(float(p.dy), 20.0)
    assert not p.is_manhattan

    p_rot = p.copy().rotate(45.0)
    assert p_rot.orientation is None
    p_flip = p.flipped()
    assert p_flip.orientation is None
    p_moved = p.moved(1.0, 2.0)
    assert np.isclose(float(p_moved.dx), 11.0)
    assert np.isclose(float(p_moved.dy), 22.0)
    assert p_moved.orientation is None

    with pytest.raises(ValueError):
        _ = p.angle
    with pytest.raises(ValueError):
        _ = p.direction
    with pytest.raises(ValueError):
        p.copy_polar(d=1.0)

    c = gf.Component()
    c.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=(1, 0))
    c.add_port(port=p)
    assert gf.port.get_ports_facing(c.ports, "E") == []
    out_gds = export_gds(
        c,
        tmp_path / "none_port.gds",
        max_vertex_count=4096,
        with_metadata=False,
    )
    assert out_gds.exists()


def test_port_copy_polar_snaps_summed_center_and_traces() -> None:
    p = gf.Port(
        name="o1",
        center=(0.0004, 0.0),
        width=0.5,
        orientation=45.0,
    )
    moved = p.copy_polar(d=0.0004 * np.sqrt(2), angle=-0.5)
    assert np.isclose(float(moved.orientation), 0.0)
    assert np.isclose(float(moved.x), 0.001, atol=1e-9)
    assert np.isclose(float(moved.y), 0.0, atol=1e-9)

    def end_x(d: jax.Array) -> jax.Array:
        return (
            gf.Port(
                name="o1",
                center=(0.0, 0.0),
                width=0.5,
                orientation=0.0,
            )
            .copy_polar(d=d, angle=0.0)
            .x
        )

    grad_val = jax.grad(end_x)(jax.numpy.asarray(1.0))
    assert np.isclose(float(grad_val), 1.0)
