from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.get_netlist import (
    PortCenterMatcher,
    SmartPortMatcher,
    _is_virtual_instance,
    get_netlist_recursive,
    legacy_namer,
)


def test_netlist_simple() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=2)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect("o1", c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 2


def test_netlist_simple_width_mismatch_throws_error() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=1)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect("o1", c1.ports["o2"], allow_width_mismatch=True)
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    with pytest.warns(UserWarning):
        c.get_netlist()


def test_netlist_complex() -> None:
    c = gf.components.ring_single()
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 6, len(netlist["instances"])


def test_get_netlist_cell_array() -> None:
    rows = 3
    component_to_array = gf.components.straight(length=10)
    c = gf.components.array(
        component_to_array,
        column_pitch=100,
        columns=1,
        rows=rows,
        add_ports=True,
    )
    n = c.get_netlist(on_multi_connect="ignore", instance_namer=legacy_namer)
    n_ports_expected = 2 * rows
    assert len(c.ports) == n_ports_expected, (
        f"Expected {n_ports_expected} ports on component. Got {len(c.ports)}"
    )
    assert len(n["instances"]) == 1, (
        f"Expected only one instance for array. Got {len(n['instances'])}"
    )
    inst_name = c.insts[0].name
    assert len(n["ports"]) == n_ports_expected, (
        f"Expected {n_ports_expected} ports in netlist. Got {len(n['ports'])}"
    )
    for ib in range(rows):
        for port in component_to_array.ports:
            expected_port_name = f"{port.name}_{ib + 1}_1"
            expected_lower_port_name = f"{inst_name}<0.{ib}>,{port.name}"
            assert expected_port_name in n["ports"]
            assert n["ports"][expected_port_name] == expected_lower_port_name

    inst = next(iter(n["instances"].values()))
    n_rows = inst["array"]["rows"]
    n_columns = inst["array"]["columns"]
    assert n_rows == rows and n_columns == 1, (
        f"Expected {n_rows=}={rows} and {n_columns=}=1"
    )


def test_get_netlist_cell_array_no_ports() -> None:
    rows = 3
    c = gf.components.array(
        gf.components.straight(length=10),
        columns=1,
        column_pitch=100,
        rows=rows,
        add_ports=False,
    )
    n = c.get_netlist(on_multi_connect="ignore")
    assert len(c.ports) == 0, (
        f"Expected no ports on component with add_ports=False. Got {len(c.ports)}"
    )
    assert len(n["ports"]) == 0, (
        f"Expected no ports in netlist with add_ports=False. Got {len(n['ports'])}"
    )
    assert len(n["instances"]) == 1, (
        f"Expected only one instance for array. Got {len(n['instances'])}"
    )
    inst = next(iter(n["instances"].values()))
    assert inst["array"]["columns"] == 1 and inst["array"]["rows"] == rows


def test_get_netlist_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.dmovey(-100)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_promoted() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.dmovey(-100)
    c.add_port("t1", port=i1.ports["o2"])
    c.add_port("t2", port=i2.ports["o1"])
    c.add_port("t3", port=i3.ports["o1"])
    c.add_port("t4", port=i3.ports["o2"])
    netlist = c.get_netlist()
    links = netlist["nets"]
    ports = netlist["ports"]
    expected_ports = {"t1": "i1,o2", "t2": "i2,o1", "t3": "i3,o1", "t4": "i3,o2"}

    assert len(links) == 1
    assert ports == expected_ports
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    assert "warnings" not in netlist


def test_get_netlist_close_enough_fails() -> None:
    """Move connection 1nm outwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovex(1)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 0


def test_get_netlist_close_enough_orthogonal_fails() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovey(0.001)
    netlist = c.get_netlist(port_matcher=SmartPortMatcher(position_tolerance_dbu=0.0))
    links = netlist["nets"]
    assert len(links) == 0


def test_get_netlist_close_enough_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.rotate(angle=0.01)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_throws_error_bad_rotation() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.rotate(90)
    with pytest.warns(UserWarning):
        c.get_netlist()


def test_get_netlist_tiny() -> None:
    c = gf.Component()
    cc = gf.components.straight(length=0.002)
    i1 = c.add_ref(cc, "i1")
    i2 = c.add_ref(cc, "i2")
    i3 = c.add_ref(cc, "i3")
    i4 = c.add_ref(cc, "i4")

    i2.connect("o2", i1.ports["o1"])
    i3.connect("o2", i2.ports["o1"])
    i4.connect("o2", i3.ports["o1"])

    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 3
    # cpairs = list(connections.items())
    # extracted_port_pair = set(cpairs[0])
    # expected_port_pair = {'i2,o2', 'i1,o1'}
    # assert extracted_port_pair == expected_port_pair


def test_get_netlist_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.rotate(35)
    i2.connect("o2", i1.ports["o1"])

    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i3 = c.add_ref(gf.components.wire_straight(), "i3")
    i2.connect("e2", i1.ports["e1"])
    i3.dmovey(-100)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_rotated_joint() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i2.connect("e2", i1.ports["e1"])
    i2.rotate(45)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_different_widths() -> None:
    """Move connection 1nm inwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(width=1, cross_section="metal1"), "i1")
    i2 = c.add_ref(gf.components.straight(width=10, cross_section="metal1"), "i2")
    i2.connect("e2", i1.ports["e1"], allow_width_mismatch=True)
    netlist = c.get_netlist(port_matcher=PortCenterMatcher())
    links = netlist["nets"]
    assert len(links) == 1, len(links)
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_transformed() -> None:
    rotation_value = 35
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.rotate(rotation_value)
    i2.connect("o2", i1.ports["o1"])

    # perform the initial sanity checks on the netlist
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1, len(links)
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_virtual_insts() -> None:
    """Test that virtual instances are included in the netlist."""
    c = gf.Component()
    cross_section = "strip"
    i1 = c.add_ref_off_grid(
        gf.components.straight(length=10, cross_section=cross_section)
    )
    i2 = c.add_ref_off_grid(
        gf.components.straight(length=9, cross_section=cross_section)
    )
    bend = c.add_ref_off_grid(
        gf.components.bend_euler_all_angle(angle=90, cross_section=cross_section),
    )
    bend.connect("o1", i1.ports["o2"])
    i2.connect("o1", bend.ports["o2"])
    c.add_port("o1", port=i1.ports["o1"])
    c.add_port("o2", port=i2.ports["o2"])
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 3, (
        f"Expected 3 instances in netlist. Got {len(netlist['instances'])}"
    )
    assert len(netlist["nets"]) == 2, (
        f"Expected 2 nets in netlist. Got {len(netlist['nets'])}"
    )
    assert len(netlist["ports"]) == 2, (
        f"Expected 2 ports in netlist. Got {len(netlist['ports'])}"
    )


def test_get_netlist_virtual_cell() -> None:
    """Test that get_netlist works with virtual cells."""
    c = gf.ComponentAllAngle()
    cross_section = "strip"
    i1 = c.add_ref_off_grid(
        gf.components.straight(length=10, cross_section=cross_section)
    )
    i2 = c.add_ref_off_grid(
        gf.components.straight(length=9, cross_section=cross_section)
    )
    bend = c.add_ref_off_grid(
        gf.components.bend_euler_all_angle(angle=33, cross_section=cross_section),
    )
    bend.connect("o1", i1.ports["o2"])
    i2.connect("o1", bend.ports["o2"])
    c.add_port("o1", port=i1.ports["o1"])
    c.add_port("o2", port=i2.ports["o2"])

    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 3, (
        f"Expected 3 instances in netlist. Got {len(netlist['instances'])}"
    )
    assert len(netlist["nets"]) == 2, (
        f"Expected 2 nets in netlist. Got {len(netlist['nets'])}"
    )
    assert len(netlist["ports"]) == 2, (
        f"Expected 2 ports in netlist. Got {len(netlist['ports'])}"
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_get_netlist_array_roundtrip(rotation: int, mirror: bool) -> None:
    """Test that get_netlist -> from_yaml round-trip is idempotent for arrays.

    The array pitch values in the netlist should be in the local (pre-transform)
    frame, so that from_yaml can reconstruct the same component and produce
    the same netlist on re-serialization.
    """
    c1 = gf.Component()
    ref = c1.add_ref(gf.get_component("straight"), rows=8, columns=1, row_pitch=100)
    if mirror:
        ref.dmirror()
    if rotation:
        ref.rotate(rotation)
    n1 = c1.get_netlist()

    c2 = gf.read.from_yaml(n1)
    n2 = c2.get_netlist()

    inst1 = next(iter(n1["instances"].values()))
    inst2 = next(iter(n2["instances"].values()))
    assert inst1["array"] == inst2["array"], (
        f"Array config changed after round-trip with {rotation=}, {mirror=}.\n"
        f"  Before: {inst1['array']}\n"
        f"  After:  {inst2['array']}"
    )

    pl1 = next(iter(n1["placements"].values()))
    pl2 = next(iter(n2["placements"].values()))
    assert pl1 == pl2, (
        f"Placement changed after round-trip with {rotation=}, {mirror=}.\n"
        f"  Before: {pl1}\n"
        f"  After:  {pl2}"
    )


def test_is_virtual_instance_all_angle_and_non_orthogonal() -> None:
    gf.gpdk.PDK.activate()

    c = gf.Component()
    ortho = c.add_ref(gf.components.straight(length=5), name="ortho")
    off_grid = c.add_ref_off_grid(gf.components.straight(length=5), name="off_grid")
    all_angle = c.add_ref(
        gf.components.bend_euler_all_angle(angle=33), name="all_angle"
    )
    rotated = c.add_ref(gf.components.straight(length=5), name="rotated")
    rotated.rotate(35)

    aa_parent = gf.ComponentAllAngle()
    in_aa = aa_parent.add_ref(gf.components.straight(length=5), name="in_aa")

    assert not _is_virtual_instance(ortho)
    assert not _is_virtual_instance(rotated)
    assert _is_virtual_instance(off_grid)
    assert _is_virtual_instance(all_angle)
    assert _is_virtual_instance(in_aa)


def test_get_netlist_all_angle_virtual_flags_and_recursive() -> None:
    gf.gpdk.PDK.activate()

    sub = gf.ComponentAllAngle()
    s1 = sub.add_ref(gf.components.straight(length=5), name="s1")
    bend = sub.add_ref(gf.components.bend_euler_all_angle(angle=33), name="bend")
    bend.connect("o1", s1.ports["o2"])
    sub.add_port("o1", port=s1.ports["o1"])
    sub.add_port("o2", port=bend.ports["o2"])

    top = gf.Component()
    sub_ref = top.add_ref(sub, name="sub_ref")
    s_end = top.add_ref_off_grid(gf.components.straight(length=4), name="s_end")
    s_end.connect("o1", sub_ref.ports["o2"])
    top.add_port("o1", port=sub_ref.ports["o1"])
    top.add_port("o2", port=s_end.ports["o2"])

    flat = top.get_netlist()
    assert flat["instances"]["sub_ref"]["virtual"] is True
    assert flat["instances"]["s_end"]["virtual"] is True
    assert flat["placements"]["s_end"]["rotation"] % 360 == pytest.approx(33.0)
    assert len(flat["nets"]) == 1
    assert flat["ports"] == {"o1": "sub_ref,o1", "o2": "s_end,o2"}

    rec = get_netlist_recursive(top)
    assert top.name in rec
    sub_net = rec[rec[top.name]["instances"]["sub_ref"]["component"]]
    assert sub_net["instances"]["bend"]["virtual"] is True
    assert len(sub_net["nets"]) == 1
