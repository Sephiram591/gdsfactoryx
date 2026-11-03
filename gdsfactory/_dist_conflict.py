"""Fail fast when upstream ``gdsfactory`` and ``gdsfactoryx`` are installed together.

Both distributions ship the same top-level ``gdsfactory`` package, and pip does not
stop one from overwriting the other's files. An environment with both installed
holds an unpredictable mix of the two, so importing it raises instead of
misbehaving later.

The check only fires when *both* distributions are installed. Running gdsfactoryx
from a source checkout (``PYTHONPATH``) is unaffected, because the ``gdsfactoryx``
distribution is then not installed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution


def _is_installed(name: str) -> bool:
    try:
        distribution(name)
    except PackageNotFoundError:
        return False
    return True


def check_no_conflicting_distribution() -> None:
    """Raise ``ImportError`` if upstream gdsfactory is installed next to gdsfactoryx."""
    if _is_installed("gdsfactory") and _is_installed("gdsfactoryx"):
        raise ImportError(
            "Both the upstream 'gdsfactory' distribution and 'gdsfactoryx' are installed. "
            "They provide the same 'gdsfactory' package and overwrite each other's files. "
            "Fix: pip uninstall -y gdsfactory gdsfactoryx && pip install gdsfactoryx"
        )


check_no_conflicting_distribution()
