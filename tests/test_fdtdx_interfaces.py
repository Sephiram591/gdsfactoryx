"""Tests for the interface records of averaging="subpixel_full": painting moments, faces and the junction closure.

The voxel geometry (priority painting of half-planes in a square) is checked against independent polygon
intersections (shapely).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon, box  # noqa: E402

from gdsfactory.fdtdx_stack import (  # noqa: E402
    _interface_parts,
    _line_faces,
    _mean_shift,
    _paint_levels,
    _paint_moments,
    _split_layer,
    _straight_seams,
)
from gdsfactory.fdtdx_subpixel import from_six, junction_weight  # noqa: E402


def enable_x64():
    return jax.enable_x64(True)


def half_plane(n, o, big=50.0) -> Polygon:
    """{p : n . p + o < 0} for a unit n, as a big polygon."""
    n = np.asarray(n, dtype=float)
    t = np.array([-n[1], n[0]])
    p0 = -o * n
    return Polygon([p0 + big * t, p0 - big * t, p0 - big * t - 2 * big * n, p0 + big * t - 2 * big * n])


def shapely_parts(n, o, present, is_fill, half):
    """Priority painting of the square [-half, half]^2: the parts of the levels, then the background."""
    left = box(-half, -half, half, half)
    parts = []
    for j, fill in enumerate(is_fill):
        if not present[j]:
            parts.append(Polygon())
            continue
        shape = box(-10, -10, 10, 10) if fill else half_plane(n[j], np.clip(o[j], -2 * half, 2 * half))
        parts.append(left.intersection(shape))
        left = left.difference(shape)
    parts.append(left)
    return parts


def random_config(rng, num, fill_every=0):
    angles = rng.uniform(0, 2 * math.pi, num)
    n = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    o = rng.uniform(-0.6, 0.6, num)
    is_fill = tuple(bool(fill_every and (j % fill_every == fill_every - 1)) for j in range(num))
    present = rng.uniform(size=num) > 0.2
    return n, o, present, is_fill


def _square(half, num_vertices):
    sq = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return np.concatenate([sq, np.repeat(sq[-1:], num_vertices - 4, axis=0)])


@pytest.mark.parametrize("seed", range(8))
def test_paint_moments_match_the_painting_and_shapely(seed: int) -> None:
    """The fractions are _paint_levels' numbers (bit for bit), the first moments those of the shapely parts."""
    rng = np.random.default_rng(seed)
    num = 4
    n, o, present, is_fill = random_config(rng, num, fill_every=3)
    half = 0.5
    nv = 4 + sum(not f for f in is_fill)
    with enable_x64():
        args = (jnp.asarray(_square(half, nv)), jnp.asarray(4), jnp.asarray(n), jnp.asarray(o), jnp.asarray(present))
        visible, first, _ = _paint_moments(*args, is_fill, half)
        legacy, _ = _paint_levels(args[0], args[1], jnp.asarray(1.0), args[2], args[3], args[4], is_fill, half)
    assert np.array_equal(np.asarray(visible), np.asarray(legacy))
    parts = shapely_parts(n, o, present, is_fill, half)
    for j, part in enumerate(parts):
        assert abs(float(visible[j]) - part.area) < 1e-12
        if part.area > 1e-12:
            c = part.centroid
            np.testing.assert_allclose(np.asarray(first[j]), [c.x * part.area, c.y * part.area], atol=1e-12)


def shapely_interface(a: Polygon, b: Polygon, eps: float = 1e-10) -> float:
    """Length of the boundary two parts share: a's boundary within eps of b, and the other way round, averaged
    (robust to the round-off of separately clipped polygons).
    """
    if a.is_empty or b.is_empty:
        return 0.0
    return 0.5 * (a.boundary.intersection(b.buffer(eps)).length + b.boundary.intersection(a.buffer(eps)).length)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("center", [(0.0, 0.0), (0.2, -0.15)])
def test_line_faces_match_the_shapely_interfaces(seed: int, center) -> None:
    """Every interface of the painting, with the parts on both sides: the face (j, k) on level j's line is the
    boundary shared by the parts of levels j and k (or the background), in the full square and in a
    sub-square.
    """
    rng = np.random.default_rng(100 + seed)
    num = 5
    n, o, present, is_fill = random_config(rng, num, fill_every=4 if seed % 2 else 0)
    half = 0.5 if center == (0.0, 0.0) else 0.2
    c = np.asarray(center)
    # the painting in the sub-square's own frame (offsets moved to its center)
    o_local = o + n @ c
    parts = shapely_parts(n, o_local, present, is_fill, half)
    with enable_x64():
        faces = {
            (j, k): float(length)
            for j, k, length in _line_faces(
                jnp.asarray(n), jnp.asarray(np.clip(o_local, -2 * half, 2 * half)), jnp.asarray(present), is_fill,
                jnp.zeros(2), half,
            )
        }
    for (j, k), length in faces.items():
        want = shapely_interface(parts[j], parts[k])
        assert abs(length - want) < 1e-8, (j, k, length, want)
    # nothing else touches: pairs not yielded share no boundary of positive length
    for j in range(num + 1):
        for k in range(j + 1, num + 1):
            if (j, k) not in faces:
                assert shapely_interface(parts[j], parts[k]) < 1e-8, (j, k)


def test_coincident_lines_are_seams_or_nested() -> None:
    """Complementary half-planes on one line meet along the whole line (a seam); equal half-planes nest (the
    later one has no part there); exactly on the line, as fdtdx paints it.
    """
    n = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    o = np.array([-0.1, 0.1, -0.1])  # level 0: x < 0.1; level 1: x > 0.1; level 2: x < 0.1 (hidden)
    with enable_x64():
        faces = {
            (j, k): float(v)
            for j, k, v in _line_faces(jnp.asarray(n), jnp.asarray(o), jnp.ones(3, bool), (False,) * 3, jnp.zeros(2), 0.5)
        }
    assert faces[(0, 1)] == pytest.approx(1.0)
    assert faces[(0, 2)] == 0.0 and faces[(0, 3)] == 0.0
    assert all(v == 0.0 for (j, _), v in faces.items() if j > 0)


def _parts(levels, eps_index, slopes=None, lows=None, highs=None):
    """_interface_parts of one column for half-planes ((n, o), ...) over a full voxel layer."""
    num = len(levels)
    n = jnp.asarray([lv[0] for lv in levels], dtype=jnp.float64)
    o = jnp.asarray([lv[1] for lv in levels], dtype=jnp.float64)
    lows = jnp.asarray(lows if lows is not None else [-1.0] * num, dtype=jnp.float64)
    highs = jnp.asarray(highs if highs is not None else [2.0] * num, dtype=jnp.float64)
    split = _split_layer(lows, highs, jnp.asarray(0.0, jnp.float64))
    eps = np.array([[12.0, 2.1, 1.0][i] for i in eps_index] + [1.0])
    vectors = np.zeros((num + 1, 6))
    vectors[:, :3] = eps[:, None] / np.sqrt(3) / 12.0
    contrast2 = (eps[:, None] - eps[None, :]) ** 2 * 3 / (12.0 * np.sqrt(3)) ** 2
    return split, _interface_parts(
        o, n, (False,) * num, split, lambda z: jnp.zeros((*z.shape, num), z.dtype), 0.5, None, 1.0, vectors, contrast2
    )


def test_direction_tensor_is_rank_one_for_parallel_faces_and_spread_for_a_corner() -> None:
    """A strip between two parallel walls of different levels (a thin film) is planar (its opposing faces both
    count, with the same normal); a corner of two levels is not.
    """
    with enable_x64():
        a = 0.3
        nrm = [math.cos(a), math.sin(a)]
        anti = [-math.cos(a), -math.sin(a)]
        # oxide on n.p > 0.1 first, then Si on n.p > -0.05: a Si film -0.05 < n.p < 0.1 between oxide and air
        split, (visible, tensor, _, _, _) = _parts([(anti, 0.1), (anti, -0.05)], [1, 0])
        assert float(jnp.sum(split[2][:, None] * visible, axis=0)[1]) == pytest.approx(0.15, abs=0.02)
        d = jnp.sum(split[2][:, None] * tensor, axis=0)
        omega, normal = junction_weight(from_six(d), 0.002, 0.02)
        assert float(omega) == 0.0 and abs(abs(float(normal[0]) * nrm[0] + float(normal[1]) * nrm[1]) - 1) < 1e-12
        assert float(jnp.trace(from_six(d))) > 0
        split, (_, tensor, _, _, _) = _parts([([1.0, 0.0], 0.1), ([0.0, 1.0], 0.1)], [0, 0])
        d = jnp.sum(split[2][:, None] * tensor, axis=0)
        assert float(junction_weight(from_six(d), 0.002, 0.02)[0]) == 1.0


def test_mean_shift_matches_the_target_with_its_implicit_derivative() -> None:
    """One shift for all values, each clipped to [lower, 1]: their mean is the target, also where some clip,
    and the shift's derivative is the implicit one (central differences of the bisection)."""
    rng = np.random.default_rng(3)
    with enable_x64():
        raw = jnp.asarray(rng.uniform(-0.2, 1.2, size=(5, 16)))
        lower = jnp.asarray(np.minimum(rng.uniform(0.0, 0.3, size=(5, 16)), 1.0))
        target = jnp.asarray(rng.uniform(0.35, 0.8, size=(5, 1)))
        s = _mean_shift(raw, lower, target, iterations=60)
        mean = jnp.mean(jnp.clip(raw + s, lower, 1.0), axis=-1, keepdims=True)
        assert float(jnp.abs(mean - target).max()) < 1e-12

        def shift_of(t):
            return jnp.sum(_mean_shift(raw + t * jnp.sin(raw), lower, target + 0.3 * t, iterations=60))

        grad = float(jax.grad(shift_of)(0.0))
        h = 1e-6
        fd = float((shift_of(h) - shift_of(-h)) / (2 * h))
        assert abs(grad - fd) < 1e-6 * max(1.0, abs(fd))


def test_sub_square_seams_add_up_to_the_voxel() -> None:
    """Sub-squares given the voxel's seams and unions: their fractions average to the voxel's exactly, also
    where their own seams would clip (a corner model E near 1 with a positive seam), and stay in [0, 1]."""
    rng = np.random.default_rng(4)
    q, k = 16, 3
    with enable_x64():
        def fractions(size):
            x = rng.uniform(0.0, 1.0, size=(*size, k + 1)) ** 3
            return x / x.sum(axis=-1, keepdims=True)

        e = fractions((2, 1, q))
        t = np.clip(e + rng.normal(0.0, 0.02, size=e.shape), 0.0, None)
        t /= t.sum(axis=-1, keepdims=True)
        tg = np.clip(e + rng.normal(0.0, 0.02, size=e.shape), 0.0, None)
        tg /= tg.sum(axis=-1, keepdims=True)
        voxel, straight = _straight_seams(*(jnp.asarray(x.mean(axis=2)) for x in (e, t, tg)), (False,) * k)
        unions = jnp.cumsum(voxel[..., :-1], axis=-1)[:, :, None, :]
        subs, _ = _straight_seams(
            *(jnp.asarray(x) for x in (e, t, tg)), (False,) * k, straight[:, :, None], unions
        )
        assert float(jnp.abs(subs.mean(axis=2) - voxel).max()) < 1e-6
        assert float(subs.min()) > -1e-9 and float(subs.max()) < 1 + 1e-9
        assert float(jnp.abs(subs.sum(axis=-1) - 1).max()) < 1e-12


def test_a_surface_on_a_voxel_face_moves_the_sub_intervals_consistently() -> None:
    """A top surface exactly on the face between voxel layers 12 and 13 (a tie): in each layer the sub-interval
    lengths still add up to the layer (their derivatives add up to zero), and the level's coverage of the two
    layers moves by one cell per cell of the surface, half in each (before, the layer's end sorted inside the
    tie and the lengths of layer 12 grew with the surface)."""
    with enable_x64():
        def coverage(high, k):
            _, _, length, present = _split_layer(jnp.asarray([2.0]), jnp.reshape(high, (1,)), jnp.asarray(k, float))
            return jnp.sum(length), jnp.sum(length * present[..., 0])

        for k in (12.0, 13.0):
            d_total, d_level = jax.jacobian(lambda h, k=k: coverage(h, k))(13.0)
            assert float(d_total) == 0.0
            assert float(d_level) == pytest.approx(0.5)


def test_matched_sub_cells_hold_the_voxels_materials() -> None:
    """Sub-cell fractions (two levels of one material, a third, the background) made to average to a voxel's
    material fractions: exact means per material, still fractions (in [0, 1], summing to one), homogeneous
    sub-cells of the leading level untouched where nothing is missing, and no change where they already agree."""
    from gdsfactory.fdtdx_stack import _match_fractions

    rng = np.random.default_rng(5)
    level_material = (2, 2, 0, 1)  # levels 0 and 1 are material 2; the background is material 1
    with enable_x64():
        x = rng.uniform(0.0, 1.0, size=(3, 64, 4)) ** 4
        f = jnp.asarray(x / x.sum(axis=-1, keepdims=True))
        onehot = np.zeros((4, 3))
        onehot[np.arange(4), level_material] = 1
        own = np.asarray(f.mean(axis=1)) @ onehot
        assert float(jnp.abs(_match_fractions(f, jnp.asarray(own), level_material) - f).max()) < 1e-9
        target = np.clip(own + rng.normal(0.0, 0.03, size=own.shape), 0.01, None)
        target /= target.sum(axis=-1, keepdims=True)
        g = _match_fractions(f, jnp.asarray(target), level_material)
        assert np.abs(np.asarray(g.mean(axis=1)) @ onehot - target).max() < 1e-9
        assert float(g.min()) > -1e-12 and float(g.max()) < 1 + 1e-12
        assert float(jnp.abs(g.sum(axis=-1) - 1).max()) < 1e-12
