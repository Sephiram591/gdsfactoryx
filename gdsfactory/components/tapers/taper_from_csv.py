"""Adiabatic tapers from CSV files."""

from __future__ import annotations

__all__ = [
    "taper_0p5_to_3_l36",
    "taper_from_csv",
    "taper_w10_l100",
    "taper_w10_l150",
    "taper_w10_l200",
    "taper_w11_l200",
    "taper_w12_l200",
]

import pathlib
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory._jax import xp
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec

from .._schematic import taper_schematic

data = pathlib.Path(__file__).parent / "csv_data"


def _symmetric_polygon(
    xs: npt.NDArray[np.float64],
    half_widths: Any,
) -> Any:
    """Build a symmetric closed taper polygon from x coordinates and half-widths."""
    x_pts = xp.concatenate([xs, xs[::-1]])
    y_pts = xp.concatenate([half_widths, -half_widths[::-1]])
    return xp.column_stack([x_pts, y_pts])


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper_from_csv(
    filepath: Path = data / "taper_strip_0p5_3_36.csv",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns taper from CSV file.

    Args:
        filepath: for CSV file.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    import pandas as pd

    # round_trip: correctly rounded floats. pandas' default parser can be one ulp off, which moves half-width ties
    # (e.g. 1.4495 um) to the other nanometre in the np.round below.
    taper_data = pd.read_csv(filepath, float_precision="round_trip")
    xs: npt.NDArray[np.float64] = np.asarray(taper_data["x"].values, dtype=np.float64) * 1e6
    ys: npt.NDArray[np.float64] = np.round(taper_data["width"].values * 1e6 / 2.0, 3)

    x = gf.get_cross_section(cross_section)
    layer = x.layer

    c = gf.Component()
    c.add_polygon(_symmetric_polygon(xs, ys), layer=layer)

    for section in x.sections[1:]:
        ys_trench = ys + section.width
        c.add_polygon(_symmetric_polygon(xs, ys_trench), layer=section.layer)

    c.add_port(
        name="o1",
        center=(xs[0], 0),
        width=2 * ys[0],
        orientation=180,
        layer=layer,
        cross_section=x,
    )
    c.add_port(
        name="o2",
        center=(xs[-1], 0),
        width=2 * ys[-1],
        orientation=0,
        layer=layer,
        cross_section=x,
    )
    x.add_bbox(c)
    return c


taper_0p5_to_3_l36 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_3_36.csv")
taper_w10_l100 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_100.csv")
taper_w10_l150 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_150.csv")
taper_w10_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_200.csv")
taper_w11_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_11_200.csv")
taper_w12_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_12_200.csv")
