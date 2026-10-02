"""Bridge between the differentiable geometry kernel and KLayout / kfactory.

KLayout is only used for operations that are inherently discrete:
GDS/OASIS I/O, viewing (klive), boolean / sizing operations, DRC and LVS.
Everything that crosses this bridge is converted to concrete numpy values
(gradients are stopped) and rounded to the 1 nm database unit.
"""

from __future__ import annotations

import contextlib
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
    v = to_numpy(points) * (1.0 / DBU)
    pts = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int64)  # KLayout rounding
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


_EXPORT_COUNTER = __import__("itertools").count()


def to_kfactory(
    component: Component,
    kcl: Any = None,
    add_ports: bool = True,
    exclude_layers: Any = None,
    unique_prefix: str = "",
    cache: dict[int, tuple[Any, Any]] | None = None,
) -> Any:
    """Converts a (differentiable) Component tree to a kfactory DKCell (concrete).

    Args:
        component: to convert.
        kcl: target KCLayout. Defaults to a new private layout.
        add_ports: export ports.
        exclude_layers: layers not to export.
        unique_prefix: prefix for all cell names (to avoid name clashes in kcl).
        cache: optional ``{id(component): (component, kcell)}`` reused across calls
            for locked components (exports into the same kcl).
    """
    import kfactory as kf

    kdb = _kdb()
    from gdsfactory.pdk import get_layer_info

    if kcl is None:
        kcl = kf.KCLayout(f"gdsfactoryx_export_{next(_EXPORT_COUNTER)}")
        kcl.layout.dbu = DBU
    names = {k: unique_prefix + v for k, v in _unique_names(component).items()}
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
        if cache is not None and c.locked and not excluded:
            hit = cache.get(id(c))
            if hit is not None and hit[0] is c:
                built[id(c)] = hit[1]
                return hit[1]
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
            if r.virtual:
                insert_virtual(kc, r, kdb.DCplxTrans())
                continue
            t = r.transform.to_klayout()
            child = build(r.cell)
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
                from kfactory.conf import PROPID

                kinst.set_property(PROPID.NAME, r.name)
        if add_ports:
            for p in c.ports:
                orientation = to_float(p.orientation) if p.orientation is not None else 0.0
                trans = kdb.DCplxTrans(1, orientation, False, to_float(p.x), to_float(p.y))
                kp = kc.create_port(
                    name=p.name,
                    width=round(to_float(p.width) / DBU) * DBU,
                    layer=layer_index(p.layer),
                    port_type=p.port_type,
                    dcplx_trans=trans,
                )
                with contextlib.suppress(Exception):
                    kp.info.update(_clean(dict(p.info)))
        with contextlib.suppress(Exception):
            kc.info.update(_clean(dict(c.info)))
        with contextlib.suppress(Exception):
            kc.settings = kf.KCellSettings(**_clean(dict(c.settings)))
        if c.function_name:
            kc.function_name = c.function_name
        if c.basename:
            kc.basename = c.basename
        built[id(c)] = kc
        if cache is not None and c.locked and not excluded:
            cache[id(c)] = (c, kc)
        return kc

    virtual_built: dict[tuple[int, int], Any] = {}

    def insert_virtual(parent: Any, r: Any, trans: Any) -> None:
        """kfactory ``VInstance.insert_into``: manhattan base transform for the
        instance, the residual transformation baked into a copy of the cell."""
        trans_ = trans * r.transform.to_klayout()
        base = kdb.DCplxTrans(kdb.ICplxTrans(trans_, DBU).s_trans().to_dtype(DBU))
        residual = base.inverted() * trans_
        if not r.cell._virtual and residual == kdb.DCplxTrans():
            child = build(r.cell)
        else:
            child = build_virtual(r.cell, residual)
        if r.na > 1 or r.nb > 1:
            a = to_numpy(r.a)
            b = to_numpy(r.b)
            inst = kdb.DCellInstArray(
                child.cell_index(),
                kdb.DCplxTrans(),
                kdb.DVector(float(a[0]), float(a[1])),
                kdb.DVector(float(b[0]), float(b[1])),
                r.na,
                r.nb,
            )
        else:
            inst = kdb.DCellInstArray(child.cell_index(), kdb.DCplxTrans())
        kinst = parent.kdb_cell.insert(inst)
        kinst.transform(base)
        if r.is_named:
            from kfactory.conf import PROPID

            kinst.set_property(PROPID.NAME, r.name)

    def build_virtual(c: Component, residual: Any) -> Any:
        key = (id(c), residual.hash())
        if key in virtual_built:
            return virtual_built[key]
        from gdsfactory.transform import Transform

        name = names.get(id(c), c.name)
        if residual != kdb.DCplxTrans():
            name += f"_{residual.hash():x}"
        kc = kf.DKCell(name=name, kcl=kcl)
        rt = Transform.from_klayout(residual)
        if c._virtual:
            # VKCell: float shapes transformed, then rounded once; child
            # instances inserted recursively
            for lay, polys in c.polygons.items():
                if lay in excluded or not polys:
                    continue
                shapes = kc.kdb_cell.shapes(layer_index(lay))
                for poly in polys:
                    shapes.insert(array_to_kdpolygon(poly).transformed(residual).to_itype(DBU))
            for lab in c.labels:
                if lab.layer not in excluded:
                    kc.kdb_cell.shapes(layer_index(lab.layer)).insert(
                        kdb.DText(lab.text, to_float(lab.x), to_float(lab.y)).transformed(residual)
                    )
            for r in c.insts:
                insert_virtual(kc, r, residual)
            for li in kcl.layer_indexes():
                shapes = kc.kdb_cell.shapes(li)
                if not shapes.is_empty():
                    region = kdb.Region(shapes)
                    region.merge()
                    shapes.clear()
                    shapes.insert(region)
        else:
            # KCell: flatten + merge on the grid, then KLayout transforms the
            # integer shapes
            itrans = kdb.ICplxTrans(residual, DBU)
            for lay, polys in c._flat_polygons_klayout().items():
                if lay in excluded or not polys:
                    continue
                region = kdb.Region()
                for poly in polys:
                    region.insert(array_to_kpolygon(poly))
                region.merge()
                kc.kdb_cell.shapes(layer_index(lay)).insert(region.transformed(itrans))
            for lab in c._flat_labels(rt):
                if lab.layer not in excluded:
                    kc.kdb_cell.shapes(layer_index(lab.layer)).insert(
                        kdb.DText(lab.text, to_float(lab.x), to_float(lab.y))
                    )
        if add_ports:
            for p in c.ports:
                q = p.copy(rt, on_grid=False)
                orientation = to_float(q.orientation) if q.orientation is not None else 0.0
                kp = kc.create_port(
                    name=q.name,
                    width=round(to_float(q.width) / DBU) * DBU,
                    layer=layer_index(q.layer),
                    port_type=q.port_type,
                    dcplx_trans=kdb.DCplxTrans(1, orientation, False, to_float(q.x), to_float(q.y)),
                )
                with contextlib.suppress(Exception):
                    kp.info.update(_clean(dict(q.info)))
        with contextlib.suppress(Exception):
            kc.info.update(_clean(dict(c.info)))
        with contextlib.suppress(Exception):
            kc.settings = kf.KCellSettings(**_clean(dict(c.settings)))
        if c.function_name:
            kc.function_name = c.function_name
        if c.basename:
            kc.basename = c.basename
        virtual_built[key] = kc
        return kc

    return build(component)


def flatten_exact(component: Component, merge: bool = True) -> tuple[dict[int, list[Array]], list[Any]]:
    """Flattens a (concrete) component exactly like kfactory's ``KCell.flatten``.

    Virtual references are inserted with ``insert_into_flat`` semantics (float
    shapes transformed then rounded once), regular references are flattened by
    KLayout (integer shapes and ICplxTrans), then shapes are merged per layer.

    Returns:
        polygons per layer index and labels.
    """
    import kfactory as kf

    from gdsfactory.component import Component as _C
    from gdsfactory.component import Label
    from gdsfactory.pdk import get_layer, get_layer_info

    kdb = _kdb()
    kcl = kf.KCLayout(f"gdsfactoryx_flatten_{next(_EXPORT_COUNTER)}")
    kcl.layout.dbu = DBU
    shell = _C(name=f"{component.name}_flat")
    shell.polygons = dict(component.polygons)
    shell.labels = list(component.labels)
    for r in component.insts:
        if not r.virtual:
            shell.insts.append(r)
    kc = to_kfactory(shell, kcl=kcl, add_ports=False)
    cache: dict[int, tuple[Any, Any]] = {}

    def layer_index(layer: int) -> int:
        return kcl.layout.layer(get_layer_info(layer))

    def insert_flat(r: Any, trans: Any) -> None:
        trans_ = trans * r.transform.to_klayout()
        transforms = [trans_]
        if r.na > 1 or r.nb > 1:
            a, b = to_numpy(r.a), to_numpy(r.b)
            transforms = [
                kdb.DCplxTrans(float(ia * a[0] + ib * b[0]), float(ia * a[1] + ib * b[1])) * trans_
                for ia in range(r.na)
                for ib in range(r.nb)
            ]
        for tr in transforms:
            if r.cell._virtual:
                for lay, polys in r.cell.polygons.items():
                    shapes = kc.kdb_cell.shapes(layer_index(lay))
                    for poly in polys:
                        shapes.insert(array_to_kdpolygon(poly).transformed(tr).to_itype(DBU))
                for lab in r.cell.labels:
                    kc.kdb_cell.shapes(layer_index(lab.layer)).insert(
                        kdb.DText(lab.text, to_float(lab.x), to_float(lab.y)).transformed(tr)
                    )
                for rr in r.cell.insts:
                    insert_flat(rr, tr)
            else:
                child = to_kfactory(r.cell, kcl=kcl, add_ports=False, cache=cache, unique_prefix="f_")
                for li in kcl.layer_indexes():
                    reg = kdb.Region(child.kdb_cell.begin_shapes_rec(li))
                    if not reg.is_empty():
                        reg.transform(kdb.ICplxTrans(tr, DBU))
                        kc.kdb_cell.shapes(li).insert(reg)

    for r in component.insts:
        if r.virtual:
            insert_flat(r, kdb.DCplxTrans())
    kc.kdb_cell.flatten(False)
    polys: dict[int, list[Array]] = {}
    labels: list[Any] = []
    for li in kcl.layout.layer_indexes():
        info = kcl.layout.get_info(li)
        try:
            lay = int(get_layer((info.layer, info.datatype)))
        except Exception:
            continue
        shapes = kc.kdb_cell.shapes(li)
        reg = kdb.Region(shapes)
        if merge:
            reg.merge()
        out = region_to_arrays(reg)
        if out:
            polys[lay] = out
        for sh in shapes.each(kdb.Shapes.STexts):
            t = sh.dtext
            labels.append(Label(t.string, (t.x, t.y), lay))
    return polys, labels


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
    kc = to_kfactory(component, exclude_layers=exclude_layers)
    gdspath = pathlib.Path(gdspath)
    opts = save_options
    if opts is None:
        from kfactory.utilities import save_layout_options

        opts = save_layout_options()
    if not with_metadata:
        opts.write_context_info = False
    kc.write(str(gdspath), save_options=opts, set_meta_data=with_metadata)
    return gdspath


def show(component: Any, **kwargs: Any) -> None:
    """Shows a Component (or a GDS path / kfactory cell) in KLayout via klive."""
    import kfactory as kf

    from gdsfactory.component import Component as _Component

    if isinstance(component, _Component):
        component = to_kfactory(component)
    kf.show(component, **kwargs)


def _kport_to_kwargs(p: Any, dbu: float) -> dict[str, Any]:
    t = p.dcplx_trans
    width = getattr(p, "dwidth", None)
    if width is None:
        width = p.width * dbu if isinstance(p.width, int) else p.width
    li = p.layer_info
    info: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        info = dict(p.info)
    with contextlib.suppress(Exception):
        xs = p.cross_section
        if xs is not None and getattr(xs, "name", None):
            import kfactory as kf

            # canonical name if import_gds remapped a conflicting cross-section
            canonical = kf.kcl.cross_sections.cross_sections.get(xs.name, xs)
            info["cross_section"] = canonical.name
    return dict(
        info=info,
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
            ref.transform = Transform.from_klayout(inst.dcplx_trans)
            if ca.is_regular_array():
                ref.na, ref.nb = ca.na, ca.nb
                ref.a = asarray([ca.a.x, ca.a.y])
                ref.b = asarray([ca.b.x, ca.b.y])
            from kfactory.conf import PROPID

            name = inst.property(PROPID.NAME)
            if name:
                ref.name = str(name)
        if kcl is not None:
            try:
                kc = kcl[cell.cell_index()]
                for p in kc.ports:
                    try:
                        kw = _kport_to_kwargs(p, dbu)
                        pinfo = kw.pop("info")
                        newp = c.add_port(**kw)
                        newp.info.update(pinfo)
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
