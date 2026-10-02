from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import kfactory as kf
import numpy as np
import numpy.typing as npt
from numpy import float64

import gdsfactory as gf
from gdsfactory._jax import round_st, to_float, xp, xset

if TYPE_CHECKING:
    from gdsfactory.component import Component, ComponentReference
    from gdsfactory.typings import LayerSpec, LayerSpecs

kdb = kf.kdb

RAD2DEG = 180.0 / np.pi
DEG2RAD = 1 / RAD2DEG


def move_port_to_zero(
    component: Component, port_name: str = "o1", mirror: bool = False
) -> gf.Component:
    """Return a container that contains a reference to the original component.

    The new component has port_name in (0, 0).

    Args:
        component: to move the port to (0, 0).
        port_name: to move to (0, 0).
        mirror: if True, mirrors the component.
    """
    port_names = [p.name for p in component.ports]
    if port_name not in port_names:
        raise ValueError(f"port_name = {port_name!r} not in {port_names}")

    c = gf.Component()
    ref = c << component
    if mirror:
        ref.dmirror()

    movement = ref.ports[port_name].center
    ref.move((-movement[0], -movement[1]))
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


def get_layers(component: Component) -> list[tuple[int, int]]:
    """Returns the layers of a component.

    Args:
        component: to get the layers from.
    """
    return [tuple(layer) for layer in component.layers]  # type: ignore[misc]


def extract(
    component: Component,
    layers: LayerSpecs,
    recursive: bool = True,
) -> Component:
    """Extracts a list of layers and adds them to a new Component.

    Args:
        component: to extract the layers from.
        layers: list of layers to extract.
        recursive: if True, extracts the shapes recursively.
    """
    from gdsfactory.pdk import get_layer_tuple

    c = gf.Component()

    layer_tuples = [get_layer_tuple(layer) for layer in layers]
    component_layers = get_layers(component)

    for layer_tuple in layer_tuples:
        if layer_tuple not in component_layers:
            warnings.warn(
                f"Layer {layer_tuple} not found in component {component.name!r} layers. {component_layers}",
                stacklevel=3,
            )

    layers_found = [lt for lt in component_layers if lt in layer_tuples]
    src = component.extract(layers=layers_found, recursive=recursive)
    for k, polys in src.polygons.items():
        c.polygons[k] = list(polys)

    return c


def move_to_center(component: Component, dx: float = 0, dy: float = 0) -> gf.Component:
    """Moves the component to the center of the bounding box."""
    c = component
    b = c.dbbox()
    c.move((-(b.left + b.right) / 2 + dx, -(b.bottom + b.top) / 2 + dy))
    return c


def move_port(
    component: Component, port_name: str, dx: float = 0, dy: float = 0
) -> gf.Component:
    """Moves the component port to a specific location.

    Warning: This function modifies the component in-place.

    Args:
        component: to move the port.
        port_name: to move.
        dx: to move the port.
        dy: to move the port.
    """
    c = component
    c.move((-c.ports[port_name].x + dx, -c.ports[port_name].y + dy))
    return c


type GetPolygonsResult = "dict[LayerSpec, list[npt.NDArray[np.floating[Any]]]]"


def get_polygons(
    component_or_instance: Component | ComponentReference,
    merge: bool = False,
    by: Literal["index", "name", "tuple"] = "index",
    layers: LayerSpecs | None = None,
    smooth: float | None = None,
) -> GetPolygonsResult:
    """Returns a dict of polygons ((N, 2) point arrays in um) per layer.

    Args:
        component_or_instance: to extract the polygons.
        merge: if True, merges the polygons.
        by: the format of the resulting keys in the dictionary ('index', 'name', 'tuple').
        layers: list of layer specs to extract the polygons from. If None, extracts all layers.
        smooth: if True, smooths the polygons.
    """
    from gdsfactory.pdk import get_layer, get_layer_name, get_layer_tuple

    if by == "index":
        get_key: Callable[[LayerSpec], LayerSpec] = get_layer
    elif by == "name":
        get_key = get_layer_name
    elif by == "tuple":
        get_key = get_layer_tuple
    else:
        raise ValueError("argument 'by' should be 'index' | 'name' | 'tuple'")

    from gdsfactory import klayout_bridge as kb

    polygons: GetPolygonsResult = {}

    c = component_or_instance
    if isinstance(c, gf.Component):
        flat = c.get_polygons_points(by="index", layers=layers)
    else:
        t = c.transform
        flat = {
            k: [t.apply(p) for p in v]
            for k, v in c.cell.get_polygons_points(by="index", layers=layers).items()
        }
    if layers is None:
        layers = sorted(k for k, v in flat.items() if v)

    layer_indexes = [get_layer(layer) for layer in layers]

    for layer_index in layer_indexes:
        layer_key = get_key(layer_index)
        if layer_key not in polygons:
            polygons[layer_key] = []
        polys = flat.get(int(layer_index), [])
        if smooth or merge:
            # non-differentiable: goes through a KLayout region
            r = kb.arrays_to_region(polys)
            if smooth:
                r.smooth(round(smooth / gf.kcl.dbu))
            if merge:
                r.merge()
            polys = kb.region_to_arrays(r)
        polygons[layer_key].extend(polys)
    return polygons


def get_polygons_points(
    component_or_instance: Component | ComponentReference,
    merge: bool = False,
    scale: float | None = None,
    by: Literal["index", "name", "tuple"] = "index",
    layers: LayerSpecs | None = None,
) -> dict[int | str | tuple[int, int], list[npt.NDArray[np.floating[Any]]]]:
    """Returns a dict with list of points per layer.

    Args:
        component_or_instance: to extract the polygons.
        merge: if True, merges the polygons.
        scale: if not None, scales the points.
        by: the format of the resulting keys in the dictionary ('index', 'name', 'tuple').
        layers: list of layer specs to extract the polygons from. If None, extracts all layers.
    """
    polygons_dict = get_polygons(
        component_or_instance=component_or_instance, merge=merge, by=by, layers=layers
    )
    scale = scale or 1
    return {
        layer: [scale * xp.asarray(polygon) for polygon in polygons]
        for layer, polygons in polygons_dict.items()
    }


def get_point_inside(
    component_or_instance: Component | ComponentReference, layer: LayerSpec
) -> npt.NDArray[np.floating[Any]]:
    """Returns a point inside the component or instance.

    Args:
        component_or_instance: to find a point inside.
        layer: to find a point inside.
    """
    layer = gf.get_layer(layer)
    return xp.asarray(
        get_polygons_points(component_or_instance, layers=[layer])[layer][0][0]
    )


def sign_shape(pts: npt.NDArray[np.floating[Any]]) -> float:
    pts = xp.asarray(pts)
    pts2 = xp.roll(pts, 1, axis=0)
    dx = pts2[:, 0] - pts[:, 0]
    y = pts2[:, 1] + pts[:, 1]
    return to_float(xp.sign((dx * y).sum()))


def area(pts: npt.NDArray[np.floating[Any]]) -> float:
    """Returns the area."""
    pts = xp.asarray(pts)
    pts2 = xp.roll(pts, 1, axis=0)
    dx = pts2[:, 0] - pts[:, 0]
    y = pts2[:, 1] + pts[:, 1]
    return xp.sum(dx * y) / 2  # type: ignore[return-value]


def centered_diff(a: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    a = xp.asarray(a)
    d = (xp.roll(a, -1, axis=0) - xp.roll(a, 1, axis=0)) / 2
    return d[1:-1]  # type: ignore[return-value]


def centered_diff2(a: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    a = xp.asarray(a)
    d = (xp.roll(a, -1, axis=0) - a) - (a - xp.roll(a, 1, axis=0))
    return d[1:-1]  # type: ignore[return-value]


def curvature(
    points: npt.NDArray[np.floating[Any]], t: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    """Args are the points and the tangents at each point.

        points : numpy.array shape (n, 2)
        t: numpy.array of size n

    Return:
        The curvature at each point.

    Computes the curvature at every point excluding the first and last point.

    For a planar curve parametrized as P(t) = (x(t), y(t)), the curvature is given
    by (x' y'' - x'' y' ) / (x' **2 + y' **2)**(3/2)

    """
    # Use centered difference for derivative
    dt = centered_diff(t)
    dp = centered_diff(points)
    dp2 = centered_diff2(points)

    dx = dp[:, 0] / dt
    dy = dp[:, 1] / dt

    dx2 = dp2[:, 0] / dt**2
    dy2 = dp2[:, 1] / dt**2

    res = (dx * dy2 - dx2 * dy) / (dx**2 + dy**2) ** (3 / 2)
    return res  # type: ignore[no-any-return]


def radius_of_curvature(
    points: npt.NDArray[np.floating[Any]], t: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    return 1 / curvature(points, t)


def path_length(points: npt.NDArray[np.floating[Any]]) -> float:
    """Returns: The path length.

    Args:
        points: With shape (N, 2) representing N points with coordinates x, y.
    """
    points = xp.asarray(points)
    dpts = points[1:, :] - points[:-1, :]
    _d = dpts**2
    return xp.sum(xp.sqrt(_d[:, 0] + _d[:, 1]))  # type: ignore[return-value]


def snap_angle(a: float) -> float:
    """Returns angle snapped along manhattan angle (0, 90, 180, 270).

    a: angle in deg
    Return angle snapped along manhattan angle
    """
    a = to_float(a) % 360
    if -45 < a < 45:
        return 0
    if 45 < a < 135:
        return 90
    if 135 < a < 225:
        return 180
    if 225 < a < 315:
        return 270
    return 0


def angles_rad(pts: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Returns the angles (radians) of the connection between each point and the next."""
    pts = xp.asarray(pts)
    _pts = xp.roll(pts, -1, 0)
    return xp.arctan2(_pts[:, 1] - pts[:, 1], _pts[:, 0] - pts[:, 0])  # type: ignore[return-value]


def angles_deg(pts: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Returns the angles (degrees) of the connection between each point and the next."""
    return angles_rad(pts) * RAD2DEG


def extrude_path(
    points: npt.NDArray[np.floating[Any]],
    width: float,
    with_manhattan_facing_angles: bool = True,
    spike_length: float64 | int | float = 0,
    start_angle: int | None = None,
    end_angle: int | None = None,
    grid: float | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Extrude a path of `width` along a curve defined by `points`.

    Args:
        points: numpy 2D array of shape (N, 2).
        width: of the path to extrude.
        with_manhattan_facing_angles: snaps to manhattan angles.
        spike_length: in um.
        start_angle: in degrees.
        end_angle: in degrees.
        grid: in um.

    Returns:
        numpy 2D array of shape (2*N, 2).
    """
    grid = grid or gf.kcl.dbu

    assert grid is not None

    if isinstance(points, list):
        points = xp.stack([xp.stack([p[0], p[1]]) for p in points], axis=0)
    points = xp.asarray(points, dtype=xp.float64)

    a = angles_deg(points)
    if with_manhattan_facing_angles:
        _start_angle = snap_angle(a[0] + 180)
        _end_angle = snap_angle(a[-2])
    else:
        _start_angle = a[0] + 180
        _end_angle = a[-2]

    start_angle_ = start_angle if start_angle is not None else _start_angle
    end_angle_ = end_angle if end_angle is not None else _end_angle

    assert start_angle_ is not None
    assert end_angle_ is not None

    a2 = angles_rad(points) * 0.5
    a1 = xp.roll(a2, 1)

    a2 = xset(a2, -1, end_angle_ * DEG2RAD - a2[-2])
    a1 = xset(a1, 0, start_angle_ * DEG2RAD - a1[1])

    a_plus = a2 + a1
    cos_a_min = xp.cos(a2 - a1)
    offsets = xp.column_stack((-xp.sin(a_plus) / cos_a_min, xp.cos(a_plus) / cos_a_min)) * (
        0.5 * width
    )

    points_back = xp.flipud(points - offsets)
    if to_float(spike_length) != 0:
        d = spike_length
        a_start = start_angle_ * DEG2RAD
        a_end = end_angle_ * DEG2RAD
        p_start_spike = points[0] + d * xp.array([[xp.cos(a_start), xp.sin(a_start)]])
        p_end_spike = points[-1] + d * xp.array([[xp.cos(a_end), xp.sin(a_end)]])

        pts = xp.vstack((p_start_spike, points + offsets, p_end_spike, points_back))
    else:
        pts = xp.vstack((points + offsets, points_back))

    from gdsfactory import snap

    if not snap.SNAP_ENABLED:
        return pts  # type: ignore[no-any-return]
    return round_st(pts, step=grid)  # type: ignore[no-any-return]


def trim(
    component: Component,
    domain: Sequence[tuple[float, float]],
    flatten: bool = False,
) -> gf.Component:
    """Trim a component by another geometry, preserving the component's layers and ports.

    Useful to get a smaller component from a larger one for simulation.

    Args:
        component: Component(/Reference).
        domain: list of array-like[N][2] representing the boundary of the component to keep.
        flatten: if True, flattens the component.

    Returns: New component with layers (and possibly ports) of the component restricted to the domain.

    Example:
        ```python
        import gdsfactory as gf
        c = gf.components.straight_pin(length=10)
        trimmed_c = gf.functions.trim(component=c, domain=[[0, -5], [0, 5], [5, 5], [5, -5]])
        trimmed_c.plot()
        ```
    """
    dummy = gf.Component()
    dummy.add_polygon(domain, layer=(1, 0))
    dbbox = dummy.dbbox()
    left, bottom, right, top = dbbox.left, dbbox.bottom, dbbox.right, dbbox.top
    from gdsfactory import klayout_bridge as kb

    # Clipping is a (non-differentiable) KLayout boolean: the component is
    # flattened and every layer is AND-ed with the domain bounding box.
    component.flatten()
    box = gf.kdb.Box(
        round(to_float(left) / gf.kcl.dbu),
        round(to_float(bottom) / gf.kcl.dbu),
        round(to_float(right) / gf.kcl.dbu),
        round(to_float(top) / gf.kcl.dbu),
    )
    clip = gf.kdb.Region(box)
    component.polygons = {
        k: kb.region_to_arrays(kb.arrays_to_region(v) & clip)
        for k, v in component.polygons.items()
    }
    return component


@gf.cell
def rotate(component: Component, angle: float) -> gf.Component:
    """Rotate a component by an angle in degrees.

    Args:
        component: to rotate.
        angle: in increments of 90°.

    Returns: Rotated component.
    """
    c = gf.Component()
    component = gf.get_component(component)
    ref = c.add_ref(component)
    ref.rotate(angle=angle)
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


rotate90 = partial(rotate, angle=90)
rotate180 = partial(rotate, angle=180)
rotate270 = partial(rotate, angle=270)


@gf.cell
def mirror(component: Component, x_mirror: bool = True) -> gf.Component:
    """Rotate a component by an angle in degrees.

    Args:
        component: to rotate.
        x_mirror: if True, mirrors the component along the x-axis.

    Returns: Rotated component.
    """
    c = gf.Component()
    component = gf.get_component(component)
    ref = c.add_ref(component)
    if x_mirror:
        ref.mirror_x()
    else:
        ref.mirror_y()
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


def remove_shapes_near_exclusion(
    c: gf.Component,
    target_layer: LayerSpec,
    exclusion_layer: LayerSpec,
    *,
    margin: float = 2.0,
    remove_entire_shapes: bool = True,
    flatten: bool = True,
) -> gf.Component:
    """Remove shapes on target_layer that interact with exclusion_layer.

    Args:
        c: Component to modify.
        target_layer: Layer containing shapes to potentially remove.
        exclusion_layer: Layer defining exclusion zones.
        margin: Exclusion margin/halo in microns (default 2.0).
        remove_entire_shapes: If True, removes entire shapes that touch the
            exclusion zone. If False, only clips the overlapping portions.
        flatten: If True, flattens the component before processing.

    Returns:
        Modified component with shapes removed/clipped.
    """
    from gdsfactory import klayout_bridge as kb

    if flatten:
        c.flatten()

    # Convert margin to database units
    margin_dbu = round(to_float(margin) / gf.kcl.dbu)

    # Get the exclusion region and expand it
    exclusion_region = c.get_region(exclusion_layer)
    halo_region = exclusion_region.sized(margin_dbu)

    # Get target shapes (non-recursive, like the cell's own shapes)
    target_layer_kdb = int(gf.get_layer(target_layer))
    target_region = kb.arrays_to_region(c.polygons.get(target_layer_kdb, []))

    if remove_entire_shapes:
        # Remove entire shapes that interact with the exclusion halo
        # A shape "interacts" if it has any overlap with the halo
        overlapping = target_region.overlapping(halo_region)
        cleaned_region = target_region - overlapping
    else:
        # Just clip/subtract the overlapping portions
        cleaned_region = target_region - halo_region

    # Clear target layer and add cleaned geometry
    c.polygons[target_layer_kdb] = kb.region_to_arrays(cleaned_region)
    return c
