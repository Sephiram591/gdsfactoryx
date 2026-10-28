"""Independent finite-element reference for the effective permittivity of one voxel (numpy / scipy / shapely).

Solves the electrostatic cell problem ``div(A(x) (E0 + grad phi)) = 0`` for the three unit fields E0 = e_i
on a voxel (or on a patch of voxels around it) whose materials are given as polygons, with conforming,
interface-fitted linear elements, and returns the voxel's energy-consistent apparent tensor

    A_app = G^-T W G^-1,  W_ij = <E_i . A E_j>,  G_ci = <E_i>_c   (averages over the voxel)

which for the voxel alone (whose mean fields every boundary condition below keeps at e_i) is the energy
bilinear form. Normal displacement continuity is the weak form's natural interface condition; tangential E is
continuous because phi is. Boundary conditions on the mesh's square / cube:

- ``"periodic"``: phi periodic (the homogenized tensor of the periodic medium the square generates);
- ``"dirichlet"``: phi = 0 (affine boundary data; for the voxel alone an upper bound);
- ``"mean"``: phi free, with the mean field constrained to E0 (uniform traction; for the voxel alone a lower
  bound).

The geometry is 2.5D (polygons in x, y, uniform in z: fields and tensors are 3D, d/dz = 0) or prismatic 3D
(a polygon partition per z interval, linear wedge elements). Nothing here uses JAX or gdsfactory, so it is an
independent check of :mod:`gdsfactoryx.fdtdx_subpixel`-style closures and of the stack's voxel geometry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.prepared import prep

ROUND = 12  # decimals used to merge nodes


# ---------------------------------------------------------------- meshes


@dataclasses.dataclass
class Mesh2D:
    """Interface-fitted triangulation of the square [0, size]^2: nodes (N, 2), triangles (T, 3), material (T,)."""

    nodes: np.ndarray
    triangles: np.ndarray
    material: np.ndarray
    size: float = 1.0


def _boundary_with_points(points: Sequence[tuple[float, float]], size: float) -> LineString:
    """The square's boundary as one ring through its corners and the given boundary points (sorted)."""
    pts = {(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)}
    pts |= {(round(float(x), ROUND), round(float(y), ROUND)) for x, y in points}

    def key(p: tuple[float, float]) -> float:  # counterclockwise position along the boundary
        x, y = p
        if abs(y) < 1e-12:
            return x
        if abs(x - size) < 1e-12:
            return size + y
        if abs(y - size) < 1e-12:
            return 3 * size - x
        return 4 * size - y

    ring = sorted(pts, key=key)
    return LineString([*ring, ring[0]])


def mesh_2d(regions: Sequence[tuple[Polygon, int]], n: int, periodic: bool = True, size: int = 1) -> Mesh2D:
    """Triangulates the square [0, size]^2 fitted to the regions' boundaries and to a grid of spacing 1/n.

    Args:
        regions: (polygon, material) pairs partitioning the square (overlaps resolved by the caller).
        n: grid lines per unit length (the voxels' sides, at integers, are grid lines).
        periodic: make the boundary nodes of opposite sides match (images of every boundary point).
        size: side of the square in voxels.

    Returns:
        The mesh.
    """
    square = box(0.0, 0.0, size, size)
    lines = [r.boundary.intersection(square) for r, _ in regions]
    grid = [LineString([(k / n, 0), (k / n, size)]) for k in range(1, n * size)]
    grid += [LineString([(0, k / n), (size, k / n)]) for k in range(1, n * size)]
    inner = shapely.unary_union([*lines, *grid])
    # points where interfaces meet the boundary (and their images on the opposite side)
    hits = shapely.intersection(inner, square.boundary)
    points = [] if hits.is_empty else [tuple(c) for g in shapely.get_parts(hits) for c in g.coords]
    if periodic:
        images = []
        for x, y in points:
            if abs(x) < 1e-12 or abs(x - size) < 1e-12:
                images.append((size - round(x), y))
            if abs(y) < 1e-12 or abs(y - size) < 1e-12:
                images.append((x, size - round(y)))
        points += images
    edges = shapely.unary_union([inner, _boundary_with_points(points, float(size))])
    cells = shapely.get_parts(shapely.polygonize(shapely.get_parts(edges)))
    tris, mats = [], []
    region_list = [(prep(r), r, m) for r, m in regions]
    for cell in cells:
        if cell.area < 1e-16:
            continue
        probe = cell.representative_point()
        material = next((m for pr, _, m in region_list if pr.contains(probe)), None)
        if material is None:  # on a boundary (round-off): the region with the largest overlap
            material = max(region_list, key=lambda t: t[1].intersection(cell).area)[2]
        for t in shapely.get_parts(shapely.constrained_delaunay_triangles(cell)):
            tris.append(np.asarray(t.exterior.coords)[:3])
            mats.append(material)
    tris = np.round(np.asarray(tris), ROUND)
    d1, d2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    good = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) > 1e-20  # degenerate after rounding: dropped
    tris, mats = tris[good], np.asarray(mats)[good]
    nodes, index = np.unique(tris.reshape(-1, 2), axis=0, return_inverse=True)
    return Mesh2D(nodes=nodes, triangles=index.reshape(-1, 3), material=mats, size=float(size))


def refine(mesh: Mesh2D) -> Mesh2D:
    """Uniform 1:4 refinement (edge midpoints); stays conforming and interface-fitted."""
    t = mesh.triangles
    p = mesh.nodes
    mids = np.round(0.5 * (p[t[:, [0, 1, 2]]] + p[t[:, [1, 2, 0]]]), ROUND)  # (T, 3, 2)
    allp = np.vstack([p, mids.reshape(-1, 2)])
    nodes, inv = np.unique(np.round(allp, ROUND), axis=0, return_inverse=True)
    inv = inv.ravel()
    old = inv[: len(p)]
    m = inv[len(p) :].reshape(-1, 3)
    a, b, c = old[t[:, 0]], old[t[:, 1]], old[t[:, 2]]
    ab, bc_, ca = m[:, 0], m[:, 1], m[:, 2]
    tris = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc_), (ca, bc_, c), (ab, bc_, ca))])
    return Mesh2D(nodes=nodes, triangles=tris, material=np.tile(mesh.material, 4), size=mesh.size)


# ---------------------------------------------------------------- finite elements


def _p1_gradients(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Areas (T,) and gradients (T, 3, 2) of the barycentric coordinates of linear triangles."""
    p = nodes[triangles]  # (T, 3, 2)
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    # rows of the inverse of [d1 d2]: grad lambda_1 = (d2_y, -d2_x) / det, grad lambda_2 = (-d1_y, d1_x) / det
    g1 = np.stack([d2[:, 1], -d2[:, 0]], -1) / det[:, None]
    g2 = np.stack([-d1[:, 1], d1[:, 0]], -1) / det[:, None]
    return 0.5 * np.abs(det), np.stack([-g1 - g2, g1, g2], axis=1)


def _periodic_map(nodes: np.ndarray, size: float) -> np.ndarray:
    """Index of every node after identifying opposite faces of the square / cube [0, size]^d."""
    key = np.round(nodes, 10)
    key = np.where(np.abs(key - size) < 1e-9, 0.0, key)
    _, rep = np.unique(key, axis=0, return_inverse=True)
    return rep.ravel()


def _assemble(gradients, weights, tensors, elements, num: int, plane: int):
    """Stiffness (num, num) and loads (num, 3) of linear elements with constant tensors.

    gradients: (T, k, plane) basis gradients (plane = 2: in-plane, d/dz = 0; 3: full); weights: (T,)
    volumes; tensors: (T, 3, 3); elements: (T, k) node indices.
    """
    a = tensors[:, :plane, :plane]
    ke = np.einsum("t,tai,tij,tbj->tab", weights, gradients, a, gradients)
    fe = np.einsum("t,tai,tij->taj", weights, gradients, tensors[:, :plane, :])  # (T, k, 3): grad N . A e_j
    k = elements.shape[1]
    rows = np.repeat(elements, k, axis=1).ravel()
    cols = np.tile(elements, (1, k)).ravel()
    stiff = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(num, num)).tocsr()
    load = np.zeros((num, 3))
    np.add.at(load, elements.ravel(), fe.reshape(-1, 3))
    return stiff, load


def _potentials(stiff, load, bc: str, boundary: np.ndarray | None, constraint) -> np.ndarray:
    """Phi (num, 3) of the three unit fields for the boundary condition bc."""
    num = stiff.shape[0]
    phi = np.zeros((num, 3))
    if bc == "periodic":
        keep = np.arange(1, num)  # phi defined up to a constant
        phi[keep] = spla.spsolve(stiff[keep][:, keep].tocsc(), -load[keep])
    elif bc == "dirichlet":
        keep = np.setdiff1d(np.arange(num), boundary)
        phi[keep] = spla.spsolve(stiff[keep][:, keep].tocsc(), -load[keep])
    elif bc == "mean":
        # minimize the energy with int grad phi = 0 (C phi = 0), phi up to a constant (fix node 0)
        keep = np.arange(1, num)
        c = constraint[:, keep]
        kkt = sp.bmat([[stiff[keep][:, keep], c.T], [c, None]]).tocsc()
        rhs = np.vstack([-load[keep], np.zeros((c.shape[0], 3))])
        phi[keep] = spla.spsolve(kkt, rhs)[: len(keep)]
    else:
        raise ValueError(bc)
    return phi


def _apparent(fields: np.ndarray, tensors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """G^-T W G^-1 from the fields (T, excitation, component) of the elements of a voxel."""
    vol = weights.sum()
    g = np.einsum("t,tic->ci", weights, fields) / vol  # column i: mean field of excitation i
    w = np.einsum("t,tia,tab,tjb->ij", weights, fields, tensors, fields) / vol
    gi = np.linalg.inv(g)
    out = gi.T @ w @ gi
    return 0.5 * (out + out.T)


def _bc_data(nodes: np.ndarray, size: float, elements: np.ndarray, bc: str, dims: int):
    """Element connectivity, node count, boundary nodes and the mean-field constraint's builder for bc."""
    num = len(nodes)
    boundary = None
    if bc == "periodic":
        rep = _periodic_map(nodes, size)
        elements = rep[elements]
        num = rep.max() + 1
    elif bc == "dirichlet":
        boundary = np.flatnonzero((np.abs(nodes) < 1e-12).any(1) | (np.abs(nodes - size) < 1e-12).any(1))
    return elements, num, boundary


def cell_tensor_2d(mesh: Mesh2D, eps: np.ndarray, bc: str = "periodic", window: tuple[float, float] | None = None) -> np.ndarray:
    """Apparent tensor (3, 3) of a 2.5D voxel (uniform in z) for the boundary condition bc.

    Args:
        mesh: the mesh of the voxel or of a patch around it.
        eps: (M, 3, 3) material tensors.
        bc: "periodic", "dirichlet" or "mean" on the mesh's square.
        window: (lo, hi): the voxel [lo, hi]^2 inside a patch whose apparent tensor is returned (its sides
            must be mesh lines); None: the whole square.
    """
    area, grads = _p1_gradients(mesh.nodes, mesh.triangles)
    tensors = eps[mesh.material]
    elements, num, boundary = _bc_data(mesh.nodes, mesh.size, mesh.triangles, bc, 2)
    stiff, load = _assemble(grads, area, tensors, elements, num, 2)
    constraint = None
    if bc == "mean":
        c = np.zeros((2, num))
        np.add.at(c.T, elements.ravel(), (area[:, None, None] * grads).reshape(-1, 2))
        constraint = sp.csr_matrix(c)
    phi = _potentials(stiff, load, bc, boundary, constraint)
    gphi = np.einsum("tak,tai->tik", grads, phi[elements])  # (T, excitation, 2)
    fields = np.broadcast_to(np.eye(3), (len(area), 3, 3)).copy()
    fields[:, :, :2] += gphi
    sel = np.ones(len(area), dtype=bool)
    if window is not None:
        c = mesh.nodes[mesh.triangles].mean(axis=1)
        sel = np.all((c > window[0]) & (c < window[1]), axis=1)
    return _apparent(fields[sel], tensors[sel], area[sel])


def converged_2d(regions, eps, bc="periodic", n=16, levels=3, size=1, window=None):
    """Tensor on successive uniform refinements, a Richardson estimate and the sequence.

    Returns:
        best: the finest tensor; estimate: Richardson extrapolation assuming O(h^2) (energy of a smooth
        solution with fitted interfaces; corners converge slower); history: the tensors per level.
    """
    mesh = mesh_2d(regions, n, periodic=(bc == "periodic"), size=size)
    history = [cell_tensor_2d(mesh, eps, bc, window)]
    for _ in range(levels - 1):
        mesh = refine(mesh)
        history.append(cell_tensor_2d(mesh, eps, bc, window))
    est = history[-1] + (history[-1] - history[-2]) / 3.0 if len(history) > 1 else history[-1]
    return history[-1], est, history


# ---------------------------------------------------------------- prismatic 3D


def cell_tensor_prisms(
    layers: Sequence[tuple[float, float, Sequence[tuple[Polygon, int]]]],
    eps: np.ndarray,
    n: int,
    nz: int,
    bc: str = "periodic",
    refinements: int = 0,
) -> np.ndarray:
    """Apparent tensor of a voxel made of z intervals, each a polygon partition of the unit square.

    The lateral mesh is fitted to the union of all intervals' boundaries, so wedge (triangle x interval)
    elements with linear-in-triangle, linear-in-z basis functions conform everywhere; each interval gets at
    least one element layer, ``nz`` per unit height. Integrals use 6-point quadrature (exact for a wedge's
    stiffness with a constant tensor).

    Args:
        layers: (z0, z1, regions) covering [0, 1] in order.
        eps: (M, 3, 3) tensors.
        n: lateral grid lines per unit length.
        nz: element layers per unit height (at least one per interval).
        bc: see :func:`cell_tensor_2d`.
        refinements: uniform lateral refinements of the fitted mesh.
    """
    all_regions = [(r, 0) for _, _, regs in layers for r, _ in regs]
    mesh = mesh_2d(all_regions, n, periodic=(bc == "periodic"))
    for _ in range(refinements):
        mesh = refine(mesh)
    nodes2, tris = mesh.nodes, mesh.triangles
    pts = shapely.points(nodes2[tris].mean(axis=1))
    zs, lmat = [], []  # element layers and the material of every triangle in each
    for z0, z1, regs in layers:
        mat = np.full(len(tris), -1)
        for r, m in regs:
            inside = shapely.contains(r, pts) | shapely.touches(r, pts)
            mat = np.where((mat < 0) & inside, m, mat)
        if (mat < 0).any():
            raise ValueError("A triangle is in no region")
        k = max(1, int(np.ceil(nz * (z1 - z0) - 1e-9)))
        for j in range(k):
            zs.append((z0 + (z1 - z0) * j / k, z0 + (z1 - z0) * (j + 1) / k))
            lmat.append(mat)
    zgrid = np.array([zs[0][0]] + [b for _, b in zs])
    num2 = len(nodes2)
    nodes = np.concatenate([np.column_stack([nodes2, np.full(num2, z)]) for z in zgrid])
    lam = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])  # edge midpoints
    gz = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
    area, g2 = _p1_gradients(nodes2, tris)
    grads, weights, elements, tensors = [], [], [], []
    for li in range(len(zgrid) - 1):
        h = zgrid[li + 1] - zgrid[li]
        el = np.concatenate([tris + li * num2, tris + (li + 1) * num2], axis=1)  # (T, 6)
        for lq in lam:
            for zq in gz:
                # basis: lambda_a (1 - zeta) at the bottom nodes, lambda_a zeta at the top ones
                bot = np.concatenate([g2 * (1 - zq), np.broadcast_to(-lq[None, :, None] / h, (len(tris), 3, 1))], axis=2)
                top = np.concatenate([g2 * zq, np.broadcast_to(lq[None, :, None] / h, (len(tris), 3, 1))], axis=2)
                grads.append(np.concatenate([bot, top], axis=1))  # (T, 6, 3)
                weights.append(area * h / 6.0)
                elements.append(el)
                tensors.append(eps[lmat[li]])
    grads, weights = np.concatenate(grads), np.concatenate(weights)
    elements, tensors = np.concatenate(elements), np.concatenate(tensors)
    conn, num, boundary = _bc_data(nodes, 1.0, elements, bc, 3)
    stiff, load = _assemble(grads, weights, tensors, conn, num, 3)
    constraint = None
    if bc == "mean":
        c = np.zeros((3, num))
        np.add.at(c.T, conn.ravel(), (weights[:, None, None] * grads).reshape(-1, 3))
        constraint = sp.csr_matrix(c)
    phi = _potentials(stiff, load, bc, boundary, constraint)
    fields = np.eye(3)[None] + np.einsum("tak,tai->tik", grads, phi[conn])
    return _apparent(fields, tensors, weights)


# ---------------------------------------------------------------- analytic laminate (the paper's frame form)


def kottke_frame(eps: np.ndarray, f: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Kottke's planar average in an explicit frame [n, t1, t2] (Phys. Rev. E 77, 036611, Section VI)."""
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    seed = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = seed - (seed @ n) * n
    t1 /= np.linalg.norm(t1)
    q = np.stack([n, t1, np.cross(n, t1)], axis=1)
    tau = np.zeros((3, 3))
    for fm, e in zip(np.asarray(f) / np.sum(f), eps, strict=True):
        loc = q.T @ e @ q
        t = np.empty((3, 3))
        t[0, 0] = -1 / loc[0, 0]
        t[0, 1:] = loc[0, 1:] / loc[0, 0]
        t[1:, 0] = loc[1:, 0] / loc[0, 0]
        t[1:, 1:] = loc[1:, 1:] - np.outer(loc[1:, 0], loc[0, 1:]) / loc[0, 0]
        tau += fm * t
    out = np.empty((3, 3))
    out[0, 0] = -1 / tau[0, 0]
    out[0, 1:] = -tau[0, 1:] / tau[0, 0]
    out[1:, 0] = -tau[1:, 0] / tau[0, 0]
    out[1:, 1:] = tau[1:, 1:] - np.outer(tau[1:, 0], tau[0, 1:]) / tau[0, 0]
    return q @ out @ q.T


# ---------------------------------------------------------------- geometry helpers


def half_plane(normal: Sequence[float], offset: float, big: float = 50.0) -> Polygon:
    """{p : n . p + offset < 0} (n normalized first), as a big polygon."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]])
    p0 = -offset * n  # the point of the line nearest the origin
    pts = [p0 + big * t, p0 - big * t, p0 - big * t - 2 * big * n, p0 + big * t - 2 * big * n]
    return Polygon(pts)


def paint(levels: Sequence[tuple[Polygon, int]], background: int, size: float = 1.0) -> list[tuple[Polygon, int]]:
    """Regions of the square [0, size]^2 for levels painted in priority order (earlier wins), then the background."""
    left = box(0.0, 0.0, size, size)
    out = []
    for shape, m in levels:
        part = left.intersection(shape)
        if part.area > 1e-14:
            out.append((part, m))
        left = left.difference(shape)
    if left.area > 1e-14:
        out.append((left, background))
    clean = []
    for g, m in out:
        for p in shapely.get_parts(g):
            if isinstance(p, Polygon) and p.area > 1e-14:
                clean.append((p, m))
    return clean
