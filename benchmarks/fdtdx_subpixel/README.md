# Validation of `averaging="subpixel_full"` (full tensors and junction closure)

Drivers and their results (`results/`) for the StackDevice tensor and corner work. Every driver prints its
versions (jax, fdtdx, the gdsfactory it imported) and uses fixed seeds; GPU runs were on an NVIDIA L40S
(jax 0.11.2, fdtdx 0.6.3, float32 simulations), CPU references with numpy/scipy/shapely and meep.

| Driver | What it checks | Result |
|---|---|---|
| `reference_fem.py` | independent P1 finite-element homogenization of polygon partitions (periodic, Dirichlet, mean-field boundary conditions; interface-fitted, refined) | the reference for the closure studies |
| `study_boundary_conditions.py` | which cell problem a voxel tensor should solve, against the voxel embedded in a 5 x 5 patch | voxel-alone periodic: exact for grid-aligned laminates; Dirichlet and mean-field biased (16 %, 38 %) |
| `study_closure.py` | sub-cell models for the closure (Voigt, laminate sub-cells; 2 to 8 sub-cells) | laminate sub-cells converge 3 to 5x faster; thresholds r = 0.002 to 0.02 |
| `validate_closure.py` | the production closure on junction voxels of placed devices (an L, a rotated square, an acute triangle) against the reference | closure vs embedded: 5.1 % median, 14.6 % p90; laminate along one normal: 31 %, 42 %; closure vs exact periodic: 2.2 % median |
| `legacy_fingerprint.py` | the legacy averagings' weights and tensors, bitwise, against the branch point (deterministic GPU ops) | all identical |
| `berreman.py` | 4 x 4 transfer matrix for anisotropic layers at normal incidence (self-checks against Fabry-Perot and decoupled eigenpolarizations) | reference for `em_slab.py` |
| `em_slab.py` | a 137 nm slab of a rotated uniaxial crystal with an xz entry in fdtdx vs Berreman, 40 to 5 nm, three grid offsets; staircase and diagonal baselines; reciprocity; `--grad` gradient through the solve | see below |
| `em_grating.py` | corner scattering: a grating of rotated rods (scalar and anisotropic) in fdtdx vs meep's subpixel smoothing | see below |
| `em_stability.py` | a closed periodic box with full tensors in every material and junctions everywhere, 52450 steps | energy within 1 % after the pulse, no growth |
| `bench_write.py` | cost of `write` and its gradient against `"subpixel"` | see below |
| `analyze_em.py` | tables and observed orders of the EM runs | |

## Anisotropic slab (`results/slab.json`)

Error of the Jones transmission against Berreman, the worst of three slab positions against the grid:

| averaging / tensor | 40 nm | 20 nm | 10 nm | 5 nm | order |
|---|---|---|---|---|---|
| `subpixel_full`, rotated + xz (polarization conversion 0.035) | 1.3e-2 | 5.0e-3 | 9.9e-4 | 1.9e-4 | 2.4 |
| faces snapped to the cells (staircase) | 0.12 | 3.6e-2 | 1.7e-2 | 9.2e-3 | 0.9 |
| `subpixel_full`, diagonal | 1.4e-2 | 5.4e-3 | 1.1e-3 | 2.0e-4 | 2.4 |
| `subpixel`, diagonal | same as above | | | | |
| `inverse`, diagonal | 6.9e-2 | 3.6e-2 | 2.6e-2 | 1.3e-2 | 1.0 |

|r|^2 + |t|^2 = 1 within 2e-5; t symmetric within 6e-8; the backward run's t is t^T within 4e-6.

Gradient through the whole solve (`results/slab_grad.json`), d|t_xx|^2/d(thickness) and d|t_yx|^2/d(thickness):
jax against central differences of the simulation 8e-4 and 1e-4 (20 nm), 1e-4 and 3e-4 (10 nm). Against the
transfer matrix's derivative they are 84 % and 43 % off at 20 nm, 28 % and 14 % at 10 nm: the discretized
response is piecewise smooth in an interface's position (`results/slab_scan_20.0_subpixel_full.json`: within
one voxel the slope is that of the averaged voxel, with a kink where the interface crosses a sample voxel's
side), while |t|^2 itself is within 0.6 %.

## Write cost (`results/bench*.json`, L40S)

| device | averaging | write | gradient | peak memory |
|---|---|---|---|---|
| 130 x 90 x 45 | `subpixel` | 0.096 s | 0.70 s | 1.3 GB |
| 130 x 90 x 45 | `subpixel_full` (1514 junction voxels) | 0.21 s | 0.96 s | 2.5 GB |
| 260 x 180 x 90 | `subpixel` | 1.4 s | 7.9 s | 5.2 GB |
| 260 x 180 x 90 | `subpixel_full` (3049 junction voxels) | 0.85 s | 5.3 s | 17 GB |

First calls compile: about 60 s for the write and 270 s for its gradient with `subpixel_full`.

## Corner scattering: grating of rotated rods (`results/grating_*.json`)

Zeroth-order transmission of a grating of 0.3 x 0.2 um rods rotated by 30 degrees (period 0.6 um, in-plane E,
1.55 um), against meep's subpixel smoothing at 50 to 800 pixels/um extrapolated with its observed order
(about 1.1; the reference is uncertain to about 2e-3). Worst of three rod positions against the grid:

| rods, averaging | 40 nm | 20 nm | 10 nm | 5 nm |
|---|---|---|---|---|
| eps 12, `subpixel_full` | 3.2e-3 | 1.6e-3 | 7.7e-4 | 6.5e-4 |
| eps 12, previous `subpixel_full` (one averaged normal per voxel) | 7.5e-3 | 4.4e-3 | 2.0e-3 | 7.4e-4 |
| eps 12, `subpixel` (diagonal) | 2.7e-2 | 1.8e-2 | 9.8e-3 | 5.0e-3 |
| eps 12, `inverse` | 9.4e-2 | 5.6e-2 | 3.1e-2 | 1.7e-2 |
| rotated uniaxial crystal with xz entry, `subpixel_full` | 4.2e-3 | 1.8e-3 | 3.8e-4 | 4.2e-4 |

For the crystal rods, fdtdx's own extrapolation (changes 1.2e-3, 5.3e-4, 1.0e-4 between successive
resolutions) agrees with meep's within 3e-4. At 45 degrees meep is not yet in its asymptotic range (uncertain
to about 5e-3); both `subpixel_full` versions are within that from 20 nm on.

## Stability (`results/stability.json`)

A closed periodic box (130 x 113 x 45 at 20 nm: a waveguide with 10 degree sidewalls, a slab, a deep etch, a
rotated pillar and a pad; every material a different full tensor; 2555 junction voxels over the Yee
positions), excited by a dipole pulse: over 52450 steps the total energy after the pulse stays between 1.000
and 1.010 of its first value (last: 1.003).
