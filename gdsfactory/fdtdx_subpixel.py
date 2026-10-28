"""Material permittivity tensors and their subpixel averages for :mod:`gdsfactory.fdtdx_stack`.

Pure matrix routines, independent of the stack's geometry:

- the material tensor contract: every material's relative permittivity as a
  real, symmetric, positive-definite 3 x 3 matrix in simulation coordinates
  (fdtdx's row-major 9-tuple), checked once (:func:`material_tensors`);
- the planar (laminate) average of anisotropic materials of Kottke,
  Farjadpour and Johnson, Phys. Rev. E 77, 036611 (2008), Section VI
  (:func:`laminate`), written with the projectors onto the interface normal
  and onto its plane, so that it needs no tangent frame;
- the junction closure: the effective tensor of a voxel's own arrangement of
  materials from the periodic cell problem ``div(A (e_i + grad phi_i)) = 0``
  on a grid of sub-cells (:func:`periodic_cell_tensor`), and the weight that
  blends it with the laminate where a voxel's interfaces are close to one
  plane (:func:`junction_weight`).

Tensors are (..., 3, 3) arrays. Rows and columns are x, y, z of the
simulation; nothing here rotates a material: a crystal whose principal axes are
rotated by R relative to the simulation axes has the tensor ``R @ eps @ R.T``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

# full float32 precision in contractions (GPUs default to TF32 for them: 10 mantissa bits)
_HIGHEST = jax.lax.Precision.HIGHEST
# size of an interface direction tensor (its largest entry, its trace) below which it has no direction (a
# full face weighs ~1); its square stays a normal float32, which the derivatives of ratios with it need
_NO_DIRECTION = 1e-15

#: largest |eps - eps^T| accepted, relative to the largest |eps| entry
SYMMETRY_RTOL = 1e-6
#: smallest eigenvalue accepted, relative to the largest one
DEFINITE_RTOL = 1e-6


# ---------------------------------------------------------------- material tensor contract


def material_tensor(material: Any) -> np.ndarray:
    """The relative permittivity of an fdtdx Material as a (3, 3) matrix in simulation coordinates.

    fdtdx stores a scalar, a 3-tuple (the diagonal) and a nested 3 x 3 tuple
    alike as one row-major 9-tuple (xx, xy, xz, yx, yy, yz, zx, zy, zz).

    Args:
        material: an ``fdtdx.Material``.

    Returns:
        (3, 3) float64 array.

    Raises:
        ValueError: if the permittivity is not 9 numbers.
    """
    eps = np.asarray(material.permittivity, dtype=float)
    if eps.shape != (9,):
        raise ValueError(f"Expected fdtdx's 9-component permittivity, got shape {eps.shape}")
    return eps.reshape(3, 3)


def is_diagonal(eps: np.ndarray) -> bool:
    """Whether a (3, 3) tensor has no off-diagonal entry (exactly, as fdtdx decides its array layout)."""
    return not np.any(eps[~np.eye(3, dtype=bool)])


def check_permittivity(eps: np.ndarray, name: str) -> None:
    """Raises unless a permittivity tensor is finite, symmetric and positive definite.

    The tolerances are relative to the tensor's own scale (``SYMMETRY_RTOL``,
    ``DEFINITE_RTOL``). Invalid input is rejected, never repaired: eigenvalues
    are not clipped.

    Args:
        eps: (3, 3) tensor.
        name: the material's name, for the message.

    Raises:
        ValueError: with the reason and how to fix it.
    """
    if not np.all(np.isfinite(eps)):
        raise ValueError(f"Material {name!r} has a non-finite permittivity {eps.tolist()}")
    scale = float(np.abs(eps).max())
    asymmetry = float(np.abs(eps - eps.T).max())
    if not asymmetry <= SYMMETRY_RTOL * scale:
        raise ValueError(
            f"Material {name!r} has a non-symmetric permittivity {eps.tolist()} (largest |eps - eps^T| = "
            f"{asymmetry:.3g}). A StackDevice needs the real, symmetric permittivity of a lossless dielectric "
            "(losses go into electric_conductivity or a DispersionModel); a tensor whose principal axes are "
            "rotated by R is R @ eps @ R.T"
        )
    eigenvalues = np.linalg.eigvalsh(0.5 * (eps + eps.T))
    if not eigenvalues[0] > DEFINITE_RTOL * abs(eigenvalues[-1]):
        raise ValueError(
            f"Material {name!r} has a permittivity that is not positive definite (eigenvalues "
            f"{eigenvalues.tolist()}). Explicit FDTD is unstable for it; a material with Re(eps) < 0 in band "
            "needs eps_inf >= 1 and a DispersionModel. Eigenvalues are not clipped"
        )


def material_tensors(materials: Mapping[str, Any], names: Sequence[str], ignore: Sequence[str] = ()) -> np.ndarray:
    """The (M, 3, 3) permittivity tensors of materials in a given order, each checked.

    Args:
        materials: fdtdx Materials by name.
        names: the order (e.g. fdtdx's ``compute_ordered_names``).
        ignore: names not checked (internal placeholders), returned as the identity.

    Returns:
        (M, 3, 3) float64 array.

    Raises:
        ValueError: see :func:`check_permittivity`.
    """
    out = []
    for name in names:
        if name in ignore:
            out.append(np.eye(3))
            continue
        eps = material_tensor(materials[name])
        check_permittivity(eps, name)
        out.append(0.5 * (eps + eps.T))  # exactly symmetric
    return np.stack(out) if out else np.zeros((0, 3, 3))


def tensor_contrast(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Rotation-invariant contrast of two tensors: ``|a - b|_F / scale``."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / scale)


def contrast_matrix(eps: np.ndarray) -> np.ndarray:
    """(M, M) pairwise contrasts :func:`tensor_contrast`, scaled by the largest Frobenius norm of the materials.

    Zero exactly between materials with equal tensors, so that an interface
    between them is not an interface.
    """
    if len(eps) == 0:
        return np.zeros((0, 0))
    scale = float(max(np.linalg.norm(e) for e in eps))
    diff = eps[:, None] - eps[None, :]
    return np.sqrt(np.sum(diff**2, axis=(-2, -1))) / scale


# ---------------------------------------------------------------- matrix helpers


def symmetric_inverse(a: Array) -> Array:
    """Inverse of (..., 3, 3) symmetric matrices from their adjugate (exact symmetry, cheap derivatives)."""
    a00, a01, a02 = a[..., 0, 0], a[..., 0, 1], a[..., 0, 2]
    a11, a12, a22 = a[..., 1, 1], a[..., 1, 2], a[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    rows = [[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2) / det[..., None, None]


def to_six(a: Array) -> Array:
    """(..., 3, 3) symmetric -> (..., 6): xx, yy, zz, xy, xz, yz."""
    return jnp.stack([a[..., 0, 0], a[..., 1, 1], a[..., 2, 2], a[..., 0, 1], a[..., 0, 2], a[..., 1, 2]], axis=-1)


def from_six(s: Array) -> Array:
    """(..., 6) xx, yy, zz, xy, xz, yz -> (..., 3, 3) symmetric."""
    xx, yy, zz, xy, xz, yz = (s[..., i] for i in range(6))
    rows = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


# ---------------------------------------------------------------- planar (laminate) average


def laminate(eps: Any, fractions: Array, normal: Array) -> Array:
    """Effective permittivity of parallel layers of anisotropic materials (Kottke's planar average).

    In a frame Q whose first axis is the interface normal n, the continuous
    components (normal D, tangential E) and the averaged ones (normal E,
    tangential D) are related by ``tau(Q^T eps Q)``, so for layers of volume
    fractions f_m::

        eps_eff = Q tau^-1(sum_m f_m tau(Q^T eps_m Q)) Q^T

    with, for a local tensor [[a, b^T], [b, C]] (a = n.eps n, b its normal /
    tangential coupling, C its tangential block), ``tau = [[-1/a, b^T/a],
    [b/a, C - b b^T/a]]`` and its inverse ``[[-1/t, -s^T/t], [-s/t, G - s
    s^T/t]]`` for ``tau = [[t, s^T], [s, G]]``. Here a, b and C are taken with
    the projectors ``n n^T`` and ``I - n n^T`` directly in simulation
    coordinates, so the result does not depend on a choice of tangent axes,
    nor on the sign of n. For isotropic materials it is the scalar formula,
    ``eps_eff^-1 = <1/eps> n n^T + (I - n n^T) / <eps>``. The order of the
    layers does not enter, so it is exact for any number of parallel layers.

    Args:
        eps: (M, 3, 3) symmetric positive-definite tensors (static numpy or traced).
        fractions: (..., M) volume fractions; normalized to sum to one.
        normal: (..., 3) interface normal (any length; sign does not matter).

    Returns:
        (..., 3, 3) the effective permittivity, symmetric positive definite.
    """
    dtype = fractions.dtype
    eps = jnp.asarray(eps, dtype=dtype)
    num = eps.shape[0]
    f = fractions / jnp.sum(fractions, axis=-1, keepdims=True)
    n2 = jnp.sum(normal * normal, axis=-1, keepdims=True)
    n = normal * jax.lax.rsqrt(jnp.where(n2 > 0, n2, 1.0))
    n = jnp.where(n2 > 0, n, jnp.asarray([0.0, 0.0, 1.0], dtype=dtype))
    # sum_m f_m tau_m, part by part; materials one at a time (M is small and static), so that no
    # (..., M, 3, 3) array is formed over a whole grid
    t = jnp.zeros(f.shape[:-1], dtype=dtype)  # -sum f / a
    s = jnp.zeros((*f.shape[:-1], 3), dtype=dtype)  # sum f b / a
    bb = jnp.zeros((*f.shape[:-1], 3, 3), dtype=dtype)  # sum f b b^T / a
    mean = jnp.zeros((*f.shape[:-1], 3, 3), dtype=dtype)  # sum f eps (its tangential block is sum f C)
    for m in range(num):
        en = jnp.einsum("ij,...j->...i", eps[m], n, precision=_HIGHEST)  # eps n
        a = jnp.sum(en * n, axis=-1)  # n . eps n > 0
        b = en - a[..., None] * n  # (I - n n^T) eps n
        w = f[..., m] / a
        t = t - w
        s = s + w[..., None] * b
        bb = bb + w[..., None, None] * b[..., :, None] * b[..., None, :]
        mean = mean + f[..., m, None, None] * eps[m]
    # tangential block of the mean: (I - n n^T) mean (I - n n^T)
    mn = jnp.einsum("...ij,...j->...i", mean, n, precision=_HIGHEST)
    tangential = (
        mean
        - n[..., :, None] * mn[..., None, :]
        - mn[..., :, None] * n[..., None, :]
        + jnp.sum(mn * n, axis=-1)[..., None, None] * n[..., :, None] * n[..., None, :]
    )
    gamma = tangential - bb  # sum f (C - b b^T / a)
    # tau^-1: a_eff = -1/t, b_eff = -s/t, C_eff = gamma - s s^T / t
    a_eff = -1.0 / t
    b_eff = -s / t[..., None]
    c_eff = gamma - s[..., :, None] * s[..., None, :] / t[..., None, None]
    nn = n[..., :, None] * n[..., None, :]
    return a_eff[..., None, None] * nn + n[..., :, None] * b_eff[..., None, :] + b_eff[..., :, None] * n[..., None, :] + c_eff


# ---------------------------------------------------------------- junction closure


def _q1_gauss_gradients(sizes: tuple[float, float, float]) -> np.ndarray:
    """(8, 8, 3) gradients of the trilinear basis functions at the 2 x 2 x 2 Gauss points of a brick.

    [q, a, c]: point q = 4 q_x + 2 q_y + q_z, local node a = 4 a_x + 2 a_y + a_z, component c.
    """
    g = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    h = np.asarray(sizes, dtype=float)
    out = np.zeros((8, 8, 3))
    for q in range(8):
        xi = [g[(q >> 2) & 1], g[(q >> 1) & 1], g[q & 1]]
        for a in range(8):
            corner = [(a >> 2) & 1, (a >> 1) & 1, a & 1]
            for c in range(3):
                value = 1.0 / h[c]
                for k in range(3):
                    if k == c:
                        value *= -1.0 if corner[k] == 0 else 1.0
                    else:
                        value *= (1 - xi[k]) if corner[k] == 0 else xi[k]
                out[q, a, c] = value
    return out


def _q1_reference(sizes: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear (Q1) brick of the given sides: stiffness blocks and gradient integrals.

    Returns:
        stiffness: (3, 3, 8, 8) ``int_e d_c N_a d_d N_b``, local node a = 4 a_x + 2 a_y + a_z.
        gradient: (8, 3) ``int_e grad N_a``.
    """
    corners = np.array([[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)])
    h = np.asarray(sizes, dtype=float)
    vol = float(np.prod(h))
    mass = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])  # int l_a l_b on [0, 1]
    slope = np.array([-1.0, 1.0])  # l_a'
    stiffness = np.zeros((3, 3, 8, 8))
    for c in range(3):
        for d in range(3):
            for a in range(8):
                for b in range(8):
                    value = vol / (h[c] * h[d])
                    for k in range(3):
                        ak, bk = corners[a, k], corners[b, k]
                        if k == c and k == d:
                            value *= slope[ak] * slope[bk]
                        elif k == c:
                            value *= slope[ak] * 0.5
                        elif k == d:
                            value *= slope[bk] * 0.5
                        else:
                            value *= mass[ak, bk]
                    stiffness[c, d, a, b] = value
    gradient = np.zeros((8, 3))
    for a in range(8):
        for c in range(3):
            gradient[a, c] = vol / h[c] * slope[corners[a, c]] * 0.25
    return stiffness, gradient


def _periodic_assembly(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Global node of every element's 8 local nodes on a periodic grid of sub-cells.

    Returns:
        nodes: (E, 8) global node index (E = n1 n2 n3, i-major), local node 4 a_x + 2 a_y + a_z.
        offsets: the same grid's node count, (n1 n2 n3,).
    """
    n1, n2, n3 = shape
    i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    nodes = np.stack(
        [((i + ax) % n1) * n2 * n3 + ((j + ay) % n2) * n3 + (k + az) % n3 for ax in (0, 1) for ay in (0, 1) for az in (0, 1)],
        axis=-1,
    )
    return nodes, np.arange(n1 * n2 * n3)


def periodic_cell_tensor(coefficients: Array) -> tuple[Array, Array]:
    """Effective tensor of a periodic unit cell made of constant-tensor sub-cells.

    Solves, for the three unit fields e_i, the cell problem ``div(A (e_i +
    grad phi_i)) = 0`` with periodic phi, by trilinear (Q1) finite elements on
    the grid of sub-cells (every element has its sub-cell's constant tensor):
    one stiffness matrix K per cell, factorized once (Cholesky) for the three
    right-hand sides. It returns the energy bilinear form of the computed
    fields, ``A_eff_ij = <(e_i + grad phi_i) . A (e_j + grad phi_j)>``,
    integrated exactly (2 x 2 x 2 Gauss points per element): a sum of the
    elements' positive semidefinite contributions, so symmetric and positive
    semidefinite by construction, and, as the energy of admissible fields, not
    below the exact cell tensor and second order in the solve's error (the
    equal ``<A> - F^T K^-1 F`` would be first order, and lose about the
    condition number of K in float32). The mean field is e_i (phi is
    periodic), so this is the cell's homogenized tensor, between the harmonic
    and arithmetic means of A in the matrix sense. Its derivative is the one
    of the linear solve (implicit), not of an iteration.

    Args:
        coefficients: (..., n1, n2, n3, 3, 3) symmetric positive-definite
            tensor of every sub-cell; the cell is the unit cube, sub-cells
            1/n1 x 1/n2 x 1/n3 (an axis of one sub-cell is constant along it).

    Returns:
        tensor: (..., 3, 3) the effective tensor.
        residual: (...,) the solve's normwise backward error ``|K phi + F| / (|K| |phi| + |F|)`` (Frobenius
            norms; NaN where the factorization failed).
    """
    *batch, n1, n2, n3, _, _ = coefficients.shape
    dtype = coefficients.dtype
    num = n1 * n2 * n3
    stiffness, gradient = _q1_reference((1.0 / n1, 1.0 / n2, 1.0 / n3))
    nodes, _ = _periodic_assembly((n1, n2, n3))
    a = coefficients.reshape(*batch, num, 3, 3)
    # element matrices (..., E, 8, 8) = sum_cd A_cd S_cd, and loads (..., E, 8, 3) = int grad N_a . A e_i
    ke = jnp.einsum("...ecd,cdab->...eab", a, jnp.asarray(stiffness, dtype=dtype), precision=_HIGHEST)
    fe = jnp.einsum("ac,...eci->...eai", jnp.asarray(gradient, dtype=dtype), a, precision=_HIGHEST)
    rows = np.broadcast_to(nodes[:, :, None], (num, 8, 8))
    cols = np.broadcast_to(nodes[:, None, :], (num, 8, 8))
    k = jnp.zeros((*batch, num, num), dtype=dtype).at[..., rows, cols].add(ke)
    f = jnp.zeros((*batch, num, 3), dtype=dtype).at[..., nodes, :].add(fe)
    mean = jnp.mean(a, axis=-3)
    if num == 1:
        return mean, jnp.zeros(batch, dtype=dtype)
    # phi is defined up to a constant: fix it at node 0
    kr, fr = k[..., 1:, 1:], f[..., 1:, :]
    with jax.default_matmul_precision("highest"):
        chol = jnp.linalg.cholesky(kr)
        y = jax.scipy.linalg.solve_triangular(chol, fr, lower=True)
        phi = -jax.scipy.linalg.solve_triangular(jnp.swapaxes(chol, -1, -2), y, lower=False)
    # the fields e_i + grad phi_i at every element's Gauss points (..., E, 8 points, 3 fields, 3 components)
    full_phi = jnp.concatenate([jnp.zeros_like(phi[..., :1, :]), phi], axis=-2)  # (..., num, 3)
    grads = jnp.asarray(_q1_gauss_gradients((1.0 / n1, 1.0 / n2, 1.0 / n3)), dtype=dtype)  # (8, 8, 3)
    nodal = full_phi[..., nodes, :]  # (..., E, 8 nodes, 3 fields)
    fields = jnp.eye(3, dtype=dtype) + jnp.einsum("qac,...eai->...eqic", grads, nodal, precision=_HIGHEST)
    flux = jnp.einsum("...ecd,...eqjd->...eqjc", a, fields, precision=_HIGHEST)  # A (e_j + grad phi_j)
    tensor = jnp.einsum("...eqic,...eqjc->...ij", fields, flux, precision=_HIGHEST) / (num * 8)
    # the solve's normwise backward error |K phi + F| / (|K| |phi| + |F|) (NaN where the factorization failed)
    misfit = jnp.sqrt(jnp.sum((jnp.matmul(kr, phi, precision=_HIGHEST) + fr) ** 2, axis=(-2, -1)))
    scale = jnp.sqrt(jnp.sum(kr**2, axis=(-2, -1)) * jnp.sum(phi**2, axis=(-2, -1))) + jnp.sqrt(
        jnp.sum(fr**2, axis=(-2, -1))
    )
    residual = jnp.where(scale > 0, misfit / jnp.where(scale > 0, scale, 1.0), 0.0)
    return 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2)), jax.lax.stop_gradient(residual)


# ---------------------------------------------------------------- classification


def principal_direction(d: Array, squarings: int = 4) -> tuple[Array, Array]:
    """The leading eigenvector and eigenvalue of (..., 3, 3) positive semidefinite matrices.

    By repeated squaring (the columns of ``d^(2^squarings)``), which unlike an
    eigendecomposition has finite derivatives at repeated eigenvalues. The
    error of the direction is of order ``(lambda_2 / lambda_1)^(2^squarings)``:
    exact for rank one, and below 1e-9 where ``lambda_2 <= lambda_1 / 4``.

    Args:
        d: (..., 3, 3) symmetric positive semidefinite.
        squarings: number of squarings.

    Returns:
        direction: (..., 3) unit vector (sign arbitrary); e_z where d vanishes (entries below 1e-15).
        value: (...,) its Rayleigh quotient.
    """
    scale = jnp.max(jnp.abs(d), axis=(-2, -1), keepdims=True)
    ok = scale > _NO_DIRECTION
    # The direction does not depend on d's scale (it is homogeneous of degree 0), so the normalizations
    # carry no derivative: differentiating them would only add terms that cancel, through d / m^2, whose
    # m^2 underflows in float32 for small d and makes 0 * inf = NaN. Where d is (numerically) zero the
    # identity is squared instead. For a symmetric p with largest entry 1, p p has a diagonal entry >= 1.
    p = jnp.where(ok, d / jax.lax.stop_gradient(jnp.where(ok, scale, 1.0)), jnp.eye(3, dtype=d.dtype))
    for _ in range(squarings):
        p = jnp.matmul(p, p, precision=_HIGHEST)
        p = p / jax.lax.stop_gradient(jnp.max(jnp.abs(p), axis=(-2, -1), keepdims=True))
    norms = jnp.sum(p * p, axis=-2)  # (..., 3) squared column norms
    column = jnp.argmax(jax.lax.stop_gradient(norms), axis=-1)
    v = jnp.take_along_axis(p, column[..., None, None], axis=-1)[..., 0]
    v2 = jnp.sum(v * v, axis=-1, keepdims=True)
    good = ok[..., 0] & (v2 > 0)
    v = jnp.where(good, v * jax.lax.rsqrt(jnp.where(good, v2, 1.0)), jnp.asarray([0.0, 0.0, 1.0], dtype=d.dtype))
    value = jnp.einsum("...i,...ij,...j->...", v, d, v, precision=_HIGHEST)
    return v, value


def junction_weight(d: Array, low: float, high: float) -> tuple[Array, Array]:
    """Weight of the junction closure for a voxel with interface direction tensor d, and its planar normal.

    ``d = sum_faces w n n^T`` of a voxel's interfaces; its spread off its leading
    direction, ``r = 1 - lambda_1 / tr d`` (0 where all faces are parallel, up
    to 2/3), sets the weight ``omega``: 0 for r <= low (the laminate along the
    leading direction is used), 1 for r >= high (the closure), a smooth step
    in between. Only where omega < 1 is the direction used, and there the
    leading eigenvalue is separated (r < high <= 1/3), so it is well defined
    and differentiable; where r >= high its derivative is cut.

    Args:
        d: (..., 3, 3) direction tensors.
        low: r below which the voxel is planar.
        high: r above which it is a junction (at most 1/3).

    Returns:
        omega: (...,) in [0, 1].
        normal: (..., 3) the leading direction.
    """
    normal, value = principal_direction(d)
    trace = jnp.trace(d, axis1=-2, axis2=-1)
    some = trace > _NO_DIRECTION
    r = jnp.where(some, 1.0 - value / jnp.where(some, trace, 1.0), 0.0)
    x = jnp.clip((r - low) / (high - low), 0.0, 1.0)
    omega = x * x * (3.0 - 2.0 * x)
    normal = jnp.where((r < high)[..., None], normal, jax.lax.stop_gradient(normal))
    return omega, normal
