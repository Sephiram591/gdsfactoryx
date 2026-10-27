"""Berreman 4 x 4 transfer matrix for layers of anisotropic media at normal incidence (numpy, independent of JAX).

Fields depend on z only, time e^{-i omega t}; H is scaled by the vacuum impedance. With E_z eliminated by
D_z = 0 (``E_z = -(eps_zx E_x + eps_zy E_y) / eps_zz``), the tangential fields u = (E_x, H_y, E_y, H_x) obey
``du/dz = i k0 M u`` with

    M = [[0, 1, 0, 0], [e_xx, 0, e_xy, 0], [0, 0, 0, -1], [-e_yx, 0, -e_yy, 0]],
    e_ab = eps_ab - eps_az eps_zb / eps_zz,

so a layer of thickness d maps u(0) to expm(i k0 M d) u(0). Between isotropic half-spaces of index n1
(incidence) and n3 this gives the Jones matrices of reflection and transmission, including polarization
conversion.

Run ``python berreman.py`` for the self-checks (isotropic slab against Fabry-Perot; a rotated uniaxial slab against
its decoupled eigenpolarizations).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import expm


def layer_matrix(eps: np.ndarray, k0d: float) -> np.ndarray:
    """expm(i k0 M d) of one layer (4, 4)."""
    eps = np.asarray(eps, dtype=complex)
    e = eps[:2, :2] - np.outer(eps[:2, 2], eps[2, :2]) / eps[2, 2]
    m = np.array(
        [[0, 1, 0, 0], [e[0, 0], 0, e[0, 1], 0], [0, 0, 0, -1], [-e[1, 0], 0, -e[1, 1], 0]], dtype=complex
    )
    return expm(1j * k0d * m)


def jones(layers: Sequence[tuple[np.ndarray, float]], n1: float, n3: float, wavelength: float):
    """Reflection and transmission Jones matrices of a stack (from z = 0 to its end) between two half-spaces.

    Args:
        layers: (eps (3, 3), thickness) in order of increasing z (same length unit as the wavelength).
        n1: index of the incidence half-space (z < 0).
        n3: index of the exit half-space.
        wavelength: vacuum wavelength.

    Returns:
        r (2, 2) and t (2, 2): column j is the response to an incident wave polarized along x (j = 0) or y (1),
        amplitudes of (E_x, E_y) of the reflected wave at z = 0 and of the transmitted wave at the stack's end.
    """
    k0 = 2 * np.pi / wavelength
    p = np.eye(4, dtype=complex)
    for eps, d in layers:
        p = layer_matrix(eps, k0 * d) @ p
    # basis of plane waves in an isotropic medium of index n: u = (E_x, H_y, E_y, H_x)
    def fwd(n):
        return np.array([[1, 0], [n, 0], [0, 1], [0, -n]], dtype=complex)

    def bwd(n):
        return np.array([[1, 0], [-n, 0], [0, 1], [0, n]], dtype=complex)

    r = np.zeros((2, 2), dtype=complex)
    t = np.zeros((2, 2), dtype=complex)
    for j in range(2):
        inc = fwd(n1)[:, j]
        # p (inc + B1 r) = F3 t  ->  [p B1, -F3] [r; t] = -p inc
        a = np.concatenate([p @ bwd(n1), -fwd(n3)], axis=1)
        sol = np.linalg.solve(a, -p @ inc)
        r[:, j], t[:, j] = sol[:2], sol[2:]
    return r, t


def _fabry_perot(n1, n2, n3, d, wavelength):
    k = 2 * np.pi / wavelength * n2 * d
    r12, r23 = (n1 - n2) / (n1 + n2), (n2 - n3) / (n2 + n3)
    t12, t23 = 2 * n1 / (n1 + n2), 2 * n2 / (n2 + n3)
    t = t12 * t23 * np.exp(1j * k) / (1 + r12 * r23 * np.exp(2j * k))
    r = (r12 + r23 * np.exp(2j * k)) / (1 + r12 * r23 * np.exp(2j * k))
    return r, t


def self_check() -> None:
    n1, n2, n3, d, lam = 1.0, 3.4, 1.45, 0.137, 1.55
    r, t = jones([(n2**2 * np.eye(3), d)], n1, n3, lam)
    rf, tf = _fabry_perot(n1, n2, n3, d, lam)
    assert abs(r[0, 0] - rf) < 1e-12 and abs(t[0, 0] - tf) < 1e-12 and abs(t[1, 1] - tf) < 1e-12
    assert abs(r[0, 1]) < 1e-14 and abs(t[0, 1]) < 1e-14
    # energy: |r|^2 + n3/n1 |t|^2 = 1 for a lossless slab
    assert abs(abs(r[0, 0]) ** 2 + n3 / n1 * abs(t[0, 0]) ** 2 - 1) < 1e-12
    # a uniaxial slab rotated by phi about z: in its own axes the polarizations decouple
    phi = 0.4
    rot = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    eps = rot @ np.diag([2.25, 4.0, 3.0]) @ rot.T
    r, t = jones([(eps, d)], n1, n1, lam)
    ra, ta = _fabry_perot(n1, 1.5, n1, d, lam)
    rb, tb = _fabry_perot(n1, 2.0, n1, d, lam)
    want = rot[:2, :2] @ np.diag([ta, tb]) @ rot[:2, :2].T
    assert np.abs(t - want).max() < 1e-12
    # z coupling (eps_xz) changes the effective e_xx only: compare with the reduced tensor
    eps_z = np.array([[4.0, 0.0, 0.5], [0.0, 3.0, 0.0], [0.5, 0.0, 2.0]])
    r1, t1 = jones([(eps_z, d)], n1, n1, lam)
    r2, t2 = jones([(np.diag([4.0 - 0.25 / 2.0, 3.0, 2.0]), d)], n1, n1, lam)
    assert np.abs(t1 - t2).max() < 1e-12
    print("berreman self-check passed")


if __name__ == "__main__":
    self_check()
