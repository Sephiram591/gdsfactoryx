"""Plotting (concrete values) for differentiable Components."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gdsfactory.component import Component


def plot_component(
    component: Component,
    ax: Axes | None = None,
    show_labels: bool = True,
    show_ruler: bool = True,
    show_ports: bool = True,
    return_fig: bool = False,
    pixel_buffer_options: dict[str, Any] | None = None,
) -> Figure | None:
    """Plots a Component with KLayout's renderer (like upstream gdsfactory).

    The component is exported to a private KLayout layout (gradients stopped,
    coordinates rounded to 1 nm) and rendered with the PDK layer views.
    """
    import klayout.lay as lay
    import matplotlib.pyplot as plt

    from gdsfactory.config import GDSDIR_TEMP
    from gdsfactory.klayout_bridge import to_kfactory
    from gdsfactory.pdk import get_layer_views

    kc = to_kfactory(component, add_ports=show_ports)
    lyp_path = GDSDIR_TEMP / f"layer_properties-{uuid4().hex}.lyp"
    GDSDIR_TEMP.mkdir(parents=True, exist_ok=True)
    layer_views = get_layer_views()
    try:
        layer_views.to_lyp(filepath=lyp_path)
        layout_view = lay.LayoutView()
        cell_view_index = layout_view.create_layout(True)
        layout_view.active_cellview_index = cell_view_index
        cell_view = layout_view.cellview(cell_view_index)
        layout = cell_view.layout()
        layout.assign(kc.kcl.layout)
        cell_view.cell = layout.cell(kc.name)
        layout_view.max_hier()
        layout_view.load_layer_props(str(lyp_path))
    finally:
        lyp_path.unlink(missing_ok=True)

    layout_view.add_missing_layers()
    layout_view.zoom_fit()
    layout_view.set_config("text-visible", "true" if show_labels else "false")
    layout_view.set_config("grid-show-ruler", "true" if show_ruler else "false")

    pixel_buffer = layout_view.get_pixels_with_options(
        **cast(
            dict[str, Any],
            ({"width": 800, "height": 600} | (pixel_buffer_options or {})),
        )
    )
    with BytesIO(pixel_buffer.to_png_data()) as f:
        img_array = plt.imread(f)

    dpi = 80
    if ax is not None:
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(
            figsize=(img_array.shape[1] / dpi, img_array.shape[0] / dpi), dpi=dpi
        )
    ax.imshow(img_array)
    ax.axis("off")
    ax.set_position((0, 0, 1, 1))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    return fig if return_fig else None


def plot_polygons(
    component: Component,
    ax: Axes | None = None,
    alpha: float = 0.6,
) -> Axes:
    """Plots the (flattened) polygons with matplotlib, one color per layer."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    from gdsfactory._jax import to_numpy

    if ax is None:
        _, ax = plt.subplots()
    cmap = plt.get_cmap("tab20")
    for i, (_layer, polys) in enumerate(sorted(component.get_polygons_points().items())):
        for p in polys:
            ax.add_patch(MplPolygon(to_numpy(p), closed=True, alpha=alpha, color=cmap(i % 20)))
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax


__all__ = ["plot_component", "plot_polygons"]
