"""PDK activation with a fine database unit for accurate gradients."""

from __future__ import annotations

import warnings
from typing import Any

import gdsfactory as gf
import kfactory as kf

#: Above this dbu (um), snapping noise noticeably degrades finite differences.
COARSE_DBU = 1e-4

_warned = False


class GradientAccuracyWarning(UserWarning):
    """Gradients are being taken on a coarse database grid."""


def activate_pdk(pdk: Any = None, dbu: float = 1e-5) -> gf.Pdk:
    """Activates `pdk` (default: generic PDK) with a fine database unit.

    gdsfactory snaps every vertex to the database grid (1 nm by default), which
    adds noise to finite-difference gradients (~4% on dA/dR of a 10 um ring at
    1 nm). With `dbu=1e-5` (0.01 nm) the error drops below ~0.1%. KLayout uses
    32-bit coordinates, so the usable extent is about +/- 2**31 * dbu
    (21 mm at 1e-5, 2.1 mm at 1e-6).

    Must be called before any cell is built (gdsfactory enforces this). Write
    final GDS files for fabrication from a session using the foundry dbu.
    """
    if pdk is None:
        pdk = gf.gpdk.get_generic_pdk()
    pdk = pdk.model_copy(update={"dbu": dbu})
    pdk.activate()
    return pdk


def check_dbu() -> None:
    """Warns once if the active layout's dbu is too coarse for gradients."""
    global _warned
    if _warned or kf.kcl.dbu <= COARSE_DBU:
        return
    _warned = True
    warnings.warn(
        f"Taking gradients with dbu={kf.kcl.dbu} um: gdsfactory snaps vertices to "
        "this grid, so finite-difference gradients can be off by several percent. "
        "Call `gdsfactoryx.activate_pdk(pdk, dbu=1e-5)` before building cells for "
        "accurate gradients.",
        GradientAccuracyWarning,
        stacklevel=3,
    )
