"""Bridge between the differentiable geometry kernel and KLayout / kfactory.

KLayout is only used for operations that are inherently discrete:
GDS/OASIS I/O, viewing (klive), boolean / sizing operations, DRC and LVS.
Everything that crosses this bridge is converted to concrete numpy values
(gradients are stopped) and rounded to the 1 nm database unit.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np

from gdsfactory._jax import Array, asarray, to_float, to_numpy

if TYPE_CHECKING:
    import klayout.db as kdb

    from gdsfactory.component import Component

DBU = 1e-3


def _kdb() -> Any:
    import klayout.db as kdb

    return kdb


def array_to_kpolygon(points: Any) -> kdb.Polygon:
    """(N, 2) um array -> integer kdb.Polygon (rounded to 1 nm)."""
    kdb = _kdb()
    pts = np.round(to_numpy(points) / DBU).astype(np.int64)
    return kdb.Polygon([kdb.Point(int(x), int(y)) for x, y in pts])


def array_to_kdpolygon(points: Any) -> kdb.DPolygon:
    kdb = _kdb()
    pts = to_numpy(points)
    return kdb.DPolygon([kdb.DPoint(float(x), float(y)) for x, y in pts])


def arrays_to_region(polys: list[Any]) -> kdb.Region:
    kdb = _kdb()
    r = kdb.Region()
    for p in polys:
        r.insert(array_to_kpolygon(p))
    return r


def merge_arrays(polys: list[Any], smooth: float | None = None) -> kdb.Region:
    r = arrays_to_region(polys)
    r.merge()
    if smooth:
        r = r.smoothed(round(smooth / DBU))
    return r


def kpolygon_to_array(poly: Any) -> Array:
    """kdb.Polygon / DPolygon / SimplePolygon -> (N, 2) um jax array (holes resolved)."""
    kdb = _kdb()
    if isinstance(poly, kdb.Polygon | kdb.SimplePolygon):
        if isinstance(poly, kdb.Polygon) and poly.holes():
            poly = poly.resolved_holes()
        pts = [(p.x * DBU, p.y * DBU) for p in poly.each_point_hull()] if isinstance(
            poly, kdb.Polygon
        ) else [(p.x * DBU, p.y * DBU) for p in poly.each_point()]
    elif isinstance(poly, kdb.DPolygon | kdb.DSimplePolygon):
        if isinstance(poly, kdb.DPolygon) and poly.holes():
            poly = poly.resolved_holes()
        pts = [(p.x, p.y) for p in poly.each_point_hull()] if isinstance(
            poly, kdb.DPolygon
        ) else [(p.x, p.y) for p in poly.each_point()]
    else:
        raise TypeError(f"Unsupported polygon type {type(poly)}")
    return asarray(np.asarray(pts, dtype=np.float64))


def region_to_arrays(region: kdb.Region) -> list[Array]:
    return [kpolygon_to_array(p) for p in region.each()]


def klayout_shape_to_arrays(shape: Any) -> list[Array]:
    kdb = _kdb()
    if isinstance(shape, kdb.Region):
        return region_to_arrays(shape)
    if isinstance(shape, kdb.Box):
        return [kpolygon_to_array(kdb.Polygon(shape))]
    if isinstance(shape, kdb.DBox):
        return [kpolygon_to_array(kdb.DPolygon(shape))]
    if isinstance(shape, kdb.Path):
        return [kpolygon_to_array(shape.polygon())]
    if isinstance(shape, kdb.DPath):
        return [kpolygon_to_array(shape.polygon())]
    if isinstance(shape, kdb.Polygon | kdb.DPolygon | kdb.SimplePolygon | kdb.DSimplePolygon):
        return [kpolygon_to_array(shape)]
    if isinstance(shape, kdb.Shape):
        if shape.is_box() or shape.is_polygon() or shape.is_path() or shape.is_simple_polygon():
            return [kpolygon_to_array(shape.dpolygon)]
        return []
    raise TypeError(f"Unsupported klayout shape {type(shape)}")


# ---------------------------------------------------------------- export
def _unique_names(component: Component) -> dict[int, str]:
    """Assigns unique GDS cell names (cells with equal names but different ids)."""
    cells = [*component.called_cells(), component]
    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for c in cells:
        base = c.name
        if base in used:
            used[base] += 1
            name = f"{base}${used[base]}"
        else:
            used[base] = 0
            name = base
        names[id(c)] = name
    return names


def to_kfactory(
    component: Component,
    kcl: Any = None,
    add_ports: bool = True,
    exclude_layers: Any = None,
) -> Any:
    """Converts a (differentiable) Component tree to a kfactory DKCell (concrete)."""
    import kfactory as kf

    kdb = _kdb()
    from gdsfactory.pdk import get_layer_info

    if kcl is None:
        kcl = kf.KCLayout(f"gdsfactoryx_export_{id(component)}")
    names = _unique_names(component)
    excluded: set[int] = set()
    if exclude_layers:
        from gdsfactory.component import _layer_key

        excluded = {_layer_key(lay) for lay in exclude_layers}

    built: dict[int, Any] = {}

    def layer_index(layer: int) -> int:
        info = get_layer_info(layer)
        return kcl.layout.layer(info)

    def build(c: Component) -> Any:
        if id(c) in built:
            return built[id(c)]
        kc = kf.DKCell(name=names[id(c)], kcl=kcl)
        for lay, polys in c.polygons.items():
            if lay in excluded:
                continue
            li = layer_index(lay)
            shapes = kc.kdb_cell.shapes(li)
            for p in polys:
                shapes.insert(array_to_kpolygon(p))
        for lab in c.labels:
            if lab.layer in excluded:
                continue
            li = layer_index(lab.layer)
            kc.kdb_cell.shapes(li).insert(
                kdb.DText(lab.text, to_float(lab.x), to_float(lab.y))
            )
        for r in c.insts:
            child = build(r.cell)
            t = r.transform.to_klayout()
            if r.na > 1 or r.nb > 1:
                a = to_numpy(r.a)
                b = to_numpy(r.b)
                inst = kdb.DCellInstArray(
                    child.cell_index(),
                    t,
                    kdb.DVector(float(a[0]), float(a[1])),
                    kdb.DVector(float(b[0]), float(b[1])),
                    r.na,
                    r.nb,
                )
            else:
                inst = kdb.DCellInstArray(child.cell_index(), t)
            kinst = kc.kdb_cell.insert(inst)
            if r.is_named:
                kinst.set_property("name", r.name)
        if add_ports:
            for p in c.ports:
                orientation = to_float(p.orientation) if p.orientation is not None else 0.0
                trans = kdb.DCplxTrans(1, orientation, False, to_float(p.x), to_float(p.y))
                try:
                    kc.create_port(
                        name=p.name,
                        dwidth=round(to_float(p.width) / DBU) * DBU,
                        layer=layer_index(p.layer),
                        port_type=p.port_type,
                        dcplx_trans=trans,
                    )
                except Exception:
                    pass
        try:
            kc.info.update(_clean(dict(c.info)))
        except Exception:
            pass
        built[id(c)] = kc
        return kc

    return build(component)


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    from gdsfactory.serialization import clean_value_json

    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            cv = clean_value_json(v)
        except Exception:
            continue
        if isinstance(cv, int | float | str | bool) or cv is None:
            out[k] = cv
        elif isinstance(cv, list | dict):
            out[k] = cv
    return out


def write(
    component: Component,
    gdspath: str | pathlib.Path,
    save_options: Any = None,
    with_metadata: bool = True,
    exclude_layers: Any = None,
) -> pathlib.Path:
    kdb = _kdb()
    kc = to_kfactory(component, exclude_layers=exclude_layers)
    gdspath = pathlib.Path(gdspath)
    opts = save_options or kdb.SaveLayoutOptions()
    if not with_metadata:
        opts.write_context_info = False
    kc.kcl.layout.write(str(gdspath), opts)
    return gdspath


def show(component: Component, **kwargs: Any) -> None:
    import kfactory as kf

    kc = to_kfactory(component)
    kf.show(kc, **kwargs)


def _kport_to_kwargs(p: Any, dbu: float) -> dict[str, Any]:
    t = p.dcplx_trans
    width = getattr(p, "dwidth", None)
    if width is None:
        width = p.width * dbu if isinstance(p.width, int) else p.width
    li = p.layer_info
    return dict(
        name=p.name,
        center=(t.disp.x, t.disp.y),
        orientation=t.angle,
        width=width,
        layer=(li.layer, li.datatype),
        port_type=p.port_type,
    )


def from_kfactory(kcell: Any, cache: dict[int, Component] | None = None) -> Component:
    """Converts a kfactory/klayout cell hierarchy into a Component (constants).

    Ports, info and settings of every kfactory cell in the hierarchy are kept.
    """
    from gdsfactory.component import Component, Info
    from gdsfactory.pdk import get_layer
    from gdsfactory.transform import Transform

    cache = {} if cache is None else cache
    kcl = getattr(kcell, "kcl", None)
    layout = kcl.layout if kcl is not None else kcell.layout()
    kdb_cell = kcell.kdb_cell if hasattr(kcell, "kdb_cell") else kcell
    dbu = layout.dbu

    def convert(cell: Any) -> Component:
        if cell.cell_index() in cache:
            return cache[cell.cell_index()]
        c = Component(name=cell.name)
        for li in layout.layer_indexes():
            info = layout.get_info(li)
            try:
                lay = int(get_layer((info.layer, info.datatype)))
            except Exception:
                continue
            for shape in cell.shapes(li).each():
                if shape.is_text():
                    t = shape.dtext
                    c.labels.append(_label(t.string, t.x, t.y, lay))
                elif (
                    shape.is_box()
                    or shape.is_polygon()
                    or shape.is_path()
                    or shape.is_simple_polygon()
                ):
                    c.polygons.setdefault(lay, []).append(
                        kpolygon_to_array(shape.polygon)
                    )
        for inst in cell.each_inst():
            child = convert(layout.cell(inst.cell_index))
            ca = inst.dcell_inst
            ref = c.add_ref(child)
            ref.transform = Transform.from_klayout(ca.complex_trans())
            if ca.is_regular_array():
                ref.na, ref.nb = ca.na, ca.nb
                ref.a = asarray([ca.a.x, ca.a.y])
                ref.b = asarray([ca.b.x, ca.b.y])
            name = inst.property("name")
            if name:
                ref.name = str(name)
        if kcl is not None:
            try:
                kc = kcl[cell.cell_index()]
                for p in kc.ports:
                    try:
                        c.add_port(**_kport_to_kwargs(p, dbu))
                    except Exception:
                        pass
                c.info = Info(dict(kc.info))
                c.settings = Info(dict(kc.settings))
                fn = getattr(kc, "function_name", None)
                if fn:
                    c.function_name = fn
            except Exception:
                pass
        c.locked = True
        cache[cell.cell_index()] = c
        return c

    return convert(kdb_cell)


def _label(text: str, x: float, y: float, layer: int) -> Any:
    from gdsfactory.component import Label

    return Label(text, (x, y), layer)


__all__ = [
    "array_to_kpolygon",
    "arrays_to_region",
    "from_kfactory",
    "kpolygon_to_array",
    "merge_arrays",
    "region_to_arrays",
    "show",
    "to_kfactory",
    "write",
]
