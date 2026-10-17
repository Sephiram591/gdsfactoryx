# gdsfactoryx viewer: spec

**Status:** drafting · **Owner:** floresh2 · **Last updated:** 2026-10-14

How to use this file: we workshop it, then distill only the **Accepted** items
into an implementation brief for an agent. IDs (G#, C#, R#, D#, Q#, F#, M#) are
stable: never renumber, only add. Item status is one of *Proposed*, *Accepted*,
*Rejected*, *Deferred*. Requirement priority is *Must* (v1), *Should* or *Later*.

---

## 1. Goals

- **G1.** Inspect a component *exactly as fdtdx will simulate it*: its
  LayerStack levels in 3D, built from the `StackDevice` geometry model, and its
  ports.
- **G2.** Look inside the device with axis-aligned cross-sections of the
  `StackDevice`.
- **G3.** Work on Engaging: Python runs on a compute node, the UI runs in the
  laptop's browser.

## 2. Non-goals (v1)

- Editing geometry in the GUI. Python stays the source of truth.
- Full-chip GDS viewing (KLayout / kweb already do this).
- The solver's voxelized / subpixel-averaged view (deferred to R7).
- Any sampled or voxelized geometry in the viewer itself (C2).
- _TBD_

## 3. Workflows

- **W1. Pre-flight check.** After building a `StackDevice`, confirm that its
  levels, ports and port extensions are where I expect before setting up a
  simulation.
- **W2. Stack inspection.** Scrub an xz cross-section through the elevator
  coupler (F1) to check layer overlap, sidewall angles, `zmin` and `thickness`.
- **W3.** _TBD_ (parameter sweeps? gradient overlays? optimization playback? See Q9.)

## 4. Constraints

- **C1. Engaging.** Geometry evaluation and JAX work run on a compute node
  (`srun` on `mit_normal_gpu`, or an OnDemand session), never on the login node.
  The browser does all rendering; the server sends meshes.
- **C2. Exact, analytic geometry** (from D3, D4, D8). 3D solids and
  cross-sections are vector geometry computed from the polygon vertices. They
  use no voxels, no sampled fields, no marching cubes and no per-pixel
  evaluation. They equal the continuous geometry that `StackDevice` voxelizes
  (`docs/fdtdx.md`, "Geometry model"):
  - per-layer signed distances to the union of the layer's polygons, combined
    for derived layers (`or` = min, `and` = max, `not` = `max(a, -b)`), with
    each layer's sizing grow
  - the lateral offset `(z - z_ref)·tan(sidewall_angle) - bias - z_to_bias(z)`,
    with `z_ref = zmin + width_to_z·thickness`
  - the level's z surfaces
  - painting by `mesh_order`, with `background` filling the rest

  In vertex terms, a level's boundary at height `z` is the exact polygon offset
  of its layers by `r(z)`: the level set of the signed distance. Where a face
  grows outward, convex corners become circular arcs. Where it shrinks inward,
  convex corners stay sharp and concave corners become arcs (Q17). Planar faces
  are exact. The only approximation is the tessellation of those arcs, with a
  fixed 1 nm chord tolerance (Q16).

  Out of scope: subpixel permittivity averaging (R7) and surface roughness
  (Q14). `gdsfactory/export/to_3d.py` ignores sidewalls and is not used.
- **C3. Dependencies.** New dependencies go behind an optional `[gui]` extra.
  `import gdsfactory` must not import them.
- **C6. Development environment.** Implementing agents use the **`betterfdtd`**
  conda environment (`conda run -n betterfdtd python ...`) and add needed
  packages to it. It already has gdsfactory, fdtdx, shapely, scipy, jax, trimesh
  and toolz; `manifold3d` and `anywidget` (the `[gui]` extra, C3) still need
  installing there.
- **C4. Core code.** Changes to `component.py`, `fdtdx_stack.py`,
  `fdtdx_geometry.py` or `routing/` need a decision entry here first. Viewer
  code reads their public APIs.
- **C5. Units.** Coordinates are in µm. The wire format uses float32 measured
  from a local origin (the box center), because absolute float32 loses nm
  precision at chip scale.
- **C7. fdtdx 0.6.3.** The viewer tracks fdtdx **0.6.3** (`betterfdtd` env).
  Since 0.6.2: `SimulationConfig` takes `grid=UniformGrid(spacing=...)` (no
  `resolution=`), `config.uniform_spacing()` returns the grid spacing (works on
  uniform and rectilinear grids; a non-uniform grid raises `ValueError`),
  `place_objects` returns a 5-tuple `(objects, arrays, params, config, info)`,
  `run_fdtd` returns `(timestep, arrays)`, and PML coefficients are derived
  during `place_on_grid`. Importing gdsfactory enables jax x64 globally, so
  fdtdx derives the PMLs' CPML coefficients as float64; `sim.run` re-casts
  them to `config.dtype` (otherwise the CPML update promotes the float32 psi
  fields and `eqxi.while_loop` rejects the changed structure).

## 5. UI layout (sketch)

```
┌────────────────────────┬────────────────────────────────────────────┐
│ EXPLORER               │ 3D VIEWPORT (perspective)                  │
│ ▾ Layers               │                                            │
│   [x] ■ oxide    15%   │                                            │
│   [x] ■ lower    90%   │                                            │
│   [x] ■ upper    90%   │                                            │
│ ▾ Ports                │                                            │
│   [x] o1               ├────────────────────────────────────────────┤
│   [x] o2               │ 2D VIEWPORT              [ xy | xz | yz ]  │
│                        │                                            │
├────────────────────────┤                                            │
│ PROPERTIES             │                                            │
│ (selected item)        │                                            │
│                        │ y ━━━━━━━━━━━━●━━━━━━━━━━━ 0.000 µm [____] │
└────────────────────────┴────────────────────────────────────────────┘
```

The arrangement of the two viewports is open (Q4); both live in one app-shell
widget with a menubar that switches between them (D9).

---

## 6. Requirements

Acceptance criteria (AC) use fixture F1 unless stated otherwise.

### R1. Model explorer (left panel): *Must*

A left panel with **three tabs** (D13):

1. **Layout / Ports** — the **Layers** and **Ports** sections.
2. **Sources / Detectors** — the **Sources** and **Detectors** sections.
3. **Simulation** — the global settings (R13).

Sources and Detectors (R12) start empty; the Simulation tab (R13) holds the
global settings.

- **Layers:** one row per level drawn by the `StackDevice`, with a visibility
  toggle, color swatch and opacity (R2). The properties pane shows the GDS layer,
  material, `zmin`, `thickness`, `sidewall_angle`, `bias`, `mesh_order` and
  whether it is a fill level.
- **Ports:** one row per `StackDevice` port, with a visibility toggle. The
  properties pane shows the name, center, orientation, width and layer. The
  simulation-volume plane choices (R12) are **not** listed here: they are not
  ports of the layout, only a placement choice in a source/detector's port
  dropdown.
- Selecting an item highlights it in both viewports.
- Groups have a visibility toggle that applies to all their children.

AC:
- F1 lists 3 layers (`oxide`, `lower`, `upper`) and 2 ports (`o1`, `o2`).
- Selecting `o2` highlights its glyph in the 3D viewport.

### R2. Layer colors and transparency: *Must*

- Each layer has an assignable color and opacity (0 to 1), set from its row in
  the explorer.
- **Default color:** the active PDK's `layer_views` fill color, or a fallback
  palette when the layer has none.
- **Default opacity (proposed, Q5):** 0.15 for fill levels, 0.9 for the rest.
- **Fill level:** a level whose layer is one of the `StackDevice` `fill_layers`
  (default `WAFER`), or a level with `background=True`.
- One color/opacity per layer drives both viewports.
- Persistence of edits: Q6.

AC:
- In F1, `oxide` starts at the fill opacity, and `lower` and `upper` at the default.
- Changing `upper` to red updates both viewports (via the shared local state,
  D10; a kernel round trip is acceptable here).

### R3. 3D perspective viewport: *Must*

- Perspective camera with orbit, pan, zoom and a "fit to view" control.
- Levels are rendered as solids built from the geometry model (C2, D3).
- Ports are drawn as glyphs: an arrow along the orientation plus a bar for the
  width, at the port's level height (Q12, Q15).
- Visibility can be toggled per layer and per port.
- The simulation region is outlined and ticked: the box edges are drawn in one
  outline color, and each axis carries tick marks with numeric labels at nice
  step values, plus an x/y/z name label at the middle of that axis (D14).
- *Should:*
  - z-exaggeration, since sub-µm films on 10–100 µm devices are hard to see at 1:1
  - the current 2D slice plane drawn in 3D
  - a settings screen (a gear beside Fit) that replaces the viewport and holds
    the rotation style, its sensitivity, the full-extent toggle and the
    screenshot scope (D14)
  - a "show full extent (unpainted)" toggle, per layer or global, to reveal a
    level that a higher-`mesh_order` level covers everywhere (Q13)
  - a **camera** button (a screenshot icon) that writes a PNG (D14). It saves
    the **active viewport** by default; the settings' screenshot scope makes it
    save the **whole widget** instead, and the settings' save-to row makes it
    write to the **browser** (download) or the **kernel** (a file beside the
    notebook, into a folder the save-folder field sets). The scope and the save
    target are per-viewer preferences kept in the browser (localStorage), not
    kernel state; the folder itself is kernel state (`save_dir`).

AC:
- Hiding `oxide` reveals the `lower` and `upper` SiN levels and their vertical offset.
- Toggling `o1` hides only that port's glyph.
- The solids pass D8's parity tests on F1, F2 and F3.
- F2 (10° sidewalls): the top face is narrower than the bottom by exactly
  `2·thickness·tan(10°)` in total width (planar faces, so no tessellation error).

### R4. 2D cross-section viewport: *Must*

- A three-way toggle between the **xy**, **xz** and **yz** planes.
- A slider sets the plane position along its normal, with a numeric entry box.
  Its range is the `StackDevice` box along that axis (after `boundary_offset`).
  Each plane remembers its own position, so switching planes and back returns to
  where that plane was left rather than resetting to the box center (D14).
  Step and snapping: Q8.
- The view shows the cross-section of the geometry model (C2, D4) as vector
  polygons: regions colored by layer (R2), painted by `mesh_order`, with the
  `background` material where no level is.
- Zoom and pan happen in the browser with no new request, and edges stay sharp
  at any zoom.
- Equal aspect by default.
- The section rectangle is outlined and ticked like the 3D region, with each
  axis named at the middle of its edge (D14).
- *Should:*
  - vertical exaggeration in xz and yz, driven by the 3D viewport's z multiplier
    so one slider serves both (D14)
  - outlines of ports that cut the plane
  - a hover readout of coordinates and the level/material under the cursor
  - a "save image" button that writes the canvas to a PNG (D14)
  - a **plane fade** setting: the gaussian 1/e distance with which ports,
    source/detector glyphs and field overlays fade with distance from the
    section plane, as a **fraction of the box size along the section's normal**
    (`plane_fade_frac`, default 0.05). The same fraction serves every plane,
    since each plane's normal size scales it.

AC:
- An xz section through the overlap region of F1 shows:
  - `lower` at z ∈ [−0.22, −0.12] µm
  - `upper` at z ∈ [0, 0.2] µm
  - oxide below z = 0 except where `lower` is
  - air above

  These band edges are exact (planar faces).
- At any zoom, a section edge lies on the analytic boundary (within the arc
  tolerance (1 nm, Q16) on arcs; exactly elsewhere).
- Dragging the slider updates the section at ≥ _N_ fps on F1 (target: Q8).
- An xz section of F2 shows trapezoids whose sides are straight lines at exactly
  10° from vertical.

### R12. Sources and detectors: *Must*

The explorer gains **Sources** and **Detectors** tabs. They start empty and
grow as the user adds objects. This is the authoring surface for a simulation
setup; it is not a run launcher (see §2).

- **Double-click to add.** In the **3D viewport only**, double-clicking a port
  glyph adds a default detector (a `ModeOverlapDetector`) on that port. Adding
  a source is explorer-driven (a "New source" control on the Sources tab).
- **Volume planes as port choices.** The port list also offers three virtual
  ports — `xy`, `xz` and `yz` — for the plane through the **center of the
  simulation volume** parallel to those faces. They are *not* ports of the
  device (`StackDevice.ports` is untouched): a source or detector may be placed
  on one exactly as on a port. A plane's width and height are the volume's own
  in-plane extents, so `port_size_mult = (1, 1)` fills the whole simulation
  plane. A plane defaults to `(1, 1)` for this reason — a port's default
  multiplier would overflow the volume.
- **Parameters editable in the explorer.** Each added object is a row whose
  properties are editable: the port, the object type, `port_offset`,
  `port_size_mult`, the wavelength(s), `direction`, `level` and `name`.
  Changing `port_offset` moves the object's glyph in the viewports live.
- **`level` sets the object's height, on a plane as well as a port.** A level
  name puts the plane at the **middle** of that level's z span (a port uses the
  union of the levels that draw its layer; a plane uses the one level named).
  The `level` list also offers **`z_center`**, the simulation volume's own
  mid-height: on a plane it pins the plane to the volume's center rather than a
  level's span, and on a port it moves the plane's z to the volume's center
  while keeping the level's height. The empty choice is `z_center` for a plane
  and the layer union for a port. On the `xy` plane, whose normal is z, `level`
  is the plane's position; on `xz` / `yz` it is the plane's in-plane vertical
  center.
- **Declarative, not eager.** The widget stores each object as a *spec* (its
  type, port and parameters). It does **not** call `stack.create_*_at` when the
  object is added or edited. `create_*_at` is called only when the user
  **retrieves** the prepared objects (e.g. `viewer.objects` /
  `viewer.constraints`, or a "Get objects" action), which runs the creation once
  and returns the `(object, constraints)` pairs for `fdtdx.place_objects`.
- **Pre-placement display.** Objects are drawn from their spec and the port
  frame (D8), before any placement, so the viewport is a pre-flight view (W1).
  Translucent planes/boxes in 3D, outlines in 2D, with visibility toggles.
- Defaults for the two required `create_mode_overlap_detector_at` arguments
  (`port_size_mult`, `wave_characters`) come from the Simulation tab (R13, Q18).

AC:
- On F1, double-clicking port `o2` in the 3D viewport adds a `ModeOverlapDetector`
  row on the Detectors tab, drawn at `o2`.
- Editing that row's `port_offset` moves its plane in the 3D viewport with no
  `create_*_at` call.
- Calling the retrieval method on the widget returns an object and constraints
  that `fdtdx.place_objects` accepts (checked in a test, no FDTD run).
- The port list offers `xy`, `xz` and `yz`; `StackDevice.ports` is unchanged
  (checked in a test). Adding a source on `xy` with the default size yields a
  plane spanning the whole simulation volume, and `place_objects` accepts it.
- A plane's `level` moves it: `level=lower` on F1 puts the `xy` plane at
  z = −0.17 µm and `level=upper` at +0.10 µm; `z_center` and the empty choice
  put it at the volume's mid-height (checked in a test).
- In 2D, a source/detector glyph whose plane is **parallel** to the section
  fades with the same gaussian as the ports (R4), and one whose plane **crosses**
  the section is drawn only where the section's position lies within the glyph's
  own extent along the section normal — a mode plane source on `o1` does not
  appear at every z in the `xy` view.

### R13. Simulation settings: *Must*

An explorer tab holding the settings that are shared by the whole setup, as
editable fields:

- **`wave_character`** — the default wavelength(s) used when a source or
  detector is added without one (R12, Q18).
- **`port_size_mult`** — the default plane size multiplier for new
  sources/detectors (R12, Q18).
- **PML / boundary** — the boundary thickness (e.g. the `BoundaryConfig` the
  run would use; `fdtdx.BoundaryConfig.from_uniform_bound(thickness=...)`).
- **`boundary_offset`** — the box's six face offsets, `(x+, x-, y+, y-, z+, z-)`
  in µm, as three signed pairs (x, y, z). Each value moves its face of the
  `StackDevice` box along the axis, positive towards +x, +y or +z, so
  `(1, -1, …)` grows the box by 1 µm on each x side. The geometry does not
  change; the viewports re-box live (see the AC below). A box that would
  collapse (a face past its opposite) is refused.
- **Simulation config** — the fields that affect the grid, notably resolution
  and time (`fdtdx.SimulationConfig`). These also feed Q8's slider step and
  Q18's derived defaults. A numeric field shows its value in display units and
  converts only when it is sent to the kernel (the time field shows fs while
  the kernel stores s), so the text the user typed is never rewritten by a
  unit round-trip.

Edits are local state (D10/D11); nothing is placed or run (§2).

AC:
- Setting the default wavelength on the Simulation tab changes the
  `wave_characters` of a subsequently added detector.
- The PML thickness field, when set, is included in the prepared constraints on
  retrieval (D11).
- Editing `boundary_offset` re-boxes the device in the kernel and re-sends the
  `geometry`/`meta` payloads, so both viewports (3D meshes and the 2D
  cross-section) update live, without a run; the section slider range follows
  the new box and a slider parked outside it is clamped back in.

### R14. Run the simulation, three ways: *Must*

From the same prepared setup (R12, R13), the widget offers three ways to run,
increasing in separation from the session:

1. **In the widget.** A "Run" action places and runs in the kernel the widget
   already occupies (C1: on the compute node) and returns the results. It
   **blocks** the worker thread until done, and the same button **stops** it
   while it is in flight (D17). It **does** report progress:
   fdtdx's `run_fdtd` is driven with a `progress_callback`, and the widget shows
   a progress bar (D15) beside a label naming the current phase. The kernel's
   allocation must be sized for the run — a
   run that needs a GPU must be in a GPU allocation, which the notebook session
   must already be.
2. **Export.** The widget returns the prepared `objects`, `config` and
   `constraints` as Python values, for the user to place and run themselves
   (`fdtdx.place_objects`, `apply_stack`, `fdtdx.run_fdtd`). This is D11's
   retrieval extended with the `config`.
3. **Return launch code.** The widget returns Python code (a string, or written
   to a file) that creates the sources, detectors, boundaries, `config` and
   `constraints` from the current setup, places them and runs the simulation.
   The user appends it to their own script, which already defines `stack`, and
   launches it under `srun`/`sbatch`. The code assumes `stack` is in scope and
   does not rebuild the device. Design and open questions: D12.

The three share one preparation path; only the final step differs. Modes 1 and 2
need the setup to be reproducible in-process; mode 3 emits code that the user
runs alongside their own device-building code.

AC:
- Mode 2's returned `objects`/`config`/`constraints` are accepted by
  `fdtdx.place_objects` in a test (no FDTD run).
- Mode 3's returned code, appended to a script that defines `stack`, builds the
  same object list as mode 2 (compared in a test; the script need not run the
  FDTD in CI).
- Mode 1's run is off-thread (`.run()` returns before the run finishes), locks
  the setup while in flight, and mirrors fdtdx's `progress_callback` into the
  `progress` trait, along with the run's phase into the `phase` trait (both
  checked in a test).
- Mode 1's run can be stopped (D17): a stop ends the run, unlocks the setup, and
  either discards the partial result (the default) or keeps it, according to the
  settings (checked in a test).

### R15. Detector field overlays: *Must* (promoted from R8, 2026-10-14)

A detector that recorded a field can shade its plane in both viewports from the
last result. The overlay is drawn **on the detector's own plane**, in the plane's
(u, v) frame, so it lines up with the glyph the pre-placement view already draws
(R12). A detector with no result yet, or whose result has no field, keeps its
plain translucent glyph; once a field is shown, the glyph's translucent fill is
**replaced** by the field (the detector's outline still marks its extent).

- **Where the field comes from.** The last run's
  `result.arrays.detector_states[name]` (R14 mode 1), or a result the user
  supplies (see *Supplying a result* below). Each detector's own array is shaded
  on its own plane — no cross-detector interpolation.
- **The payload's axis order is (v, u).** `field_payload` drops the detector
  plane's normal axis (whichever axis has extent 1) and sends the remaining two
  transposed so the buffer is **row-major in (v, u)** — v is the plane's own
  second axis (the vertical screen axis: `z` for an `xz` plane) and u its first
  (`x`). `grid` is `[rows = len(v), cols = len(u)]`, so a wide xz plane sends
  `[75, 450]` for a 450×75 phasor. Both viewports index the buffer row-major in
  (v, u), and the 2D blit flips the raster so v row 0 lands at the low-v screen
  edge; a u-major payload would render the field transposed.
- **Per-detector display settings, editable in the explorer.** Each detector row
  on the **Detectors** tab gains a **display** group, one value each:
  - **`field_component`** — which component of the recorded field to plot, from
    the detector's own `components` (a `PhasorDetector` records the subset the
    user asked for; a `ModeOverlapDetector` always records all six `Ex, Ey, Ez,
    Hx, Hy, Hz`). Default: the first component the detector recorded. The list
    also offers **`|E|`**, a pseudo-component: the total electric-field
    magnitude, `sqrt(|Ex|² + |Ey|² + |Ez|²)`, computed from the recorded `E`
    components (it is offered only when the result recorded any). It is derived
    in the browser, not sent as a component of its own, and it composes with
    every transform and ramp like any other component.
  - **`field_transform`** — one of **`real`, `imag`, `abs`, `phase`, `logabs`**.
    Default `abs`. `logabs` is `log10(abs(...))` (see *Log and the value floor*).
  - **`field_color_low`** / **`field_color_high`** — the two endpoint colors of
    the field's map, chosen per detector. The map depends on the transform (see
    *The colormap*): the signed transforms (`real`, `imag`, `phase`) use a
    **color → transparent → color** ramp (`field_color_low` at the low end,
    transparent at the midpoint, `field_color_high` at the high end); the
    non-negative transforms (`abs`, `logabs`) use a **transparent → color** ramp
    (`field_color_high` fading up from the low end). Defaults: a cool blue and a
    warm red.
  - **`field_opacity`** — the overlay's overall opacity, 0–1. Default 1. It
    multiplies the whole overlay's alpha, so the user can fade it without
    touching the map's shape.
  - **`field_visible`** — whether the overlay is drawn at all, separate from the
    detector glyph's own visibility (R12). Default on once a result exists.
- **Display settings are per-detector, per-session.** They live in the detector
  spec's params alongside `port_offset` and the rest, are carried on the wire
  like the other params, and are **never** passed to `create_*_at` (they are
  display-only, like `visible`, R12). They survive a re-run: they belong to the
  detector, not to the result.
- **One view-wide field setting.** `log_range` (default 6 decades) is the bottom
  of the `logabs` color ramp, shared by every detector's overlay (Q22). It is a
  view setting, not a detector field, and lives with the other view-wide controls
  rather than in a detector row. See *Log and the value floor* (D16).
- **The colormap is the only source of transparency** (D16). Alpha comes from the
  map's transparent middle, **not** from the field's intensity: there is no
  separate magnitude-scaled alpha. See *The colormap* below.
- **Two viewports.**
  - **3D:** the detector's plane rectangle (already built, `PlaneGlyphs`) carries
    a texture of the transformed field, with a small colorbar beside the
    viewport naming the transform and component.
  - **2D:** when the section plane is **parallel** to the detector's plane, the
    field is drawn over the detector's (u, v) rectangle in the section canvas,
    under the region outline and ticks, and it **fades with the same gaussian as
    the ports** (R4): the overlay's alpha falls off with the section plane's
    distance from the detector's plane, over `plane_fade_frac`, so a field fades in
    as its plane is approached rather than popping in. A detector whose plane is
    perpendicular to the section is not shaded in 2D (its field is a plane, and
    there is nothing to show off that plane); the 3D viewport still shows it.
- **Scale is per-plane** (Q21, resolved 2026-10-14). The range that maps onto the
  colormap is the transformed value's own range over that detector's plane, so
  two detectors with very different amplitudes are each readable. The colorbar
  shows the plane's own range. There is no shared scale across detectors.

**Supplying a result.** The widget exposes a setter — `viewer.show_result(result)`
— that takes a result the user produced themselves (a `RunResult`, R14, or any
object with the same shape: `.arrays.detector_states`, `.objects`, `.config`) and
makes it the widget's current result, exactly as a mode-1 run would (R14). After
it returns, every detector's overlay redraws from it, with no re-run and no
re-send of the geometry.

- **What it accepts.** The last result of a run the user launched outside the
  widget (R14 mode 3, or their own script): the common case is a user who ran
  under `srun` and wants to look at the fields in the widget.
- **What it validates.** The result's detectors must correspond to the widget's
  current setup — by name, grid shape, grid spacing and plane position (Q23) —
  or the setter raises (see the AC below). It does **not** re-place or re-run
  anything; it only adopts the arrays.
- **What it does not do.** It does not adopt the result's geometry or its
  `objects` as the widget's setup — the widget's own stack and specs stay
  authoritative. A result recorded on a different device is refused rather than
  silently mis-shaded.

AC:
- With a result present (from a mode-1 run or `show_result`), a detector's plane
  is shaded in 3D from `result.arrays.detector_states[name]`, and the colorbar
  names the component and transform.
- Changing a detector's `field_component` / `field_transform` /
  `field_color_low` / `field_color_high` / `field_opacity` re-shades its plane
  live, with no re-run and no kernel round-trip beyond the state patch (D10).
- Selecting `|E|` shades the plane with `sqrt(|Ex|² + |Ey|² + |Ez|²)` from the
  detector's recorded `E` components, through the same transforms and ramps as a
  real component (checked in a test); it is offered only when the result
  recorded at least one `E` component.
- The payload's buffer is row-major in (v, u) with `grid = [len(v), len(u)]`, so
  an xz plane's `grid` is taller-than-wide only if the plane is (checked in a
  test): a transposed payload would draw the field rotated.
- The overlay's alpha comes from the map alone, not from the field's intensity:
  a signed transform's value at the plane's midpoint is transparent at full
  `field_opacity`, and an `abs`/`logabs` value at the plane's low end is
  transparent (checked in a test).
- Switching the 2D section to a detector's plane shades the field there, faded
  by the same gaussian as the ports (R4); a section perpendicular to the
  detector's plane does not shade it.
- While a field is shown, the detector's translucent fill is gone (its outline
  remains); turning `field_visible` off restores the plain glyph.
- `show_result` with a result whose detectors match the setup — name, grid
  shape, grid spacing and plane position (Q23) — makes it the current result and
  redraws the overlays; a mismatched result raises and leaves the previous result
  and the overlays unchanged (checked in a test).
- A detector with no result, and a result with no field for a detector, both
  leave that detector's plain glyph (no blank texture, no error).

### Later candidates (not v1 unless promoted; see Q9)

- **R5.** Parameter explorer: sliders generated from a cell's signature, with a
  rebuild on change.
- **R6.** Shape-sensitivity overlay (`jax.jacfwd` of unmerged polygon vertices)
  and `dJ/dp` bars per parameter.
- **R7.** Solver view: voxelized permittivity / material weights from the placed
  fdtdx grid, as an alternative 2D mode (D4b).
- **R8.** Field overlays from fdtdx detectors. *Promoted 2026-10-14:* built as
  **R15**, in M4. (F1's `field` detector records the xz plane through the taper
  axis; R15 shades any detector's plane from the field it recorded.)
- **R9.** Optimization history playback (scrub logged parameters per iteration).
- **R10.** Full-chip 2D via `kweb` (already the `cad` extra).
- **R11.** Simulation volume display (PML thickness is set on the Simulation
  tab, R13):
  - translucent volume box in 3D and outline in 2D, with visibility toggles
  - open: also show the PML boundaries in the viewports?

### R16. Adopting a foreign completed result: *Must* (2026-10-16)

A user (or agent) with a **completed** fdtdx simulation — their own
`StackDevice`, source and detectors created directly with fdtdx, never through
the viewer's placeholders — can display it in the viewer via a kernel-side API,
without re-authoring the setup:

- **`Viewer.load_result(objects, arrays, config, stack=None, *, settings=None)`.**
  The three pieces fdtdx hands out (`place_objects` returns the objects/config,
  `run_fdtd` the arrays) pass straight through; the widget's own `RunResult`
  passes as `load_result(*result)`. `show_result` keeps its stricter,
  name-matched contract unchanged.
- **Lenient validation.** Refused only what can never render: missing
  `arrays.detector_states`, missing `objects.detectors`/`objects.volume`, a
  non-positive `config.resolution`, a phasor detector that is not a plane, a
  phasor whose grid shape disagrees with its placement, or a `stack` whose box
  disagrees with the result's grid by more than one cell. Everything else is
  accepted — the goal is to display, not to gate.
- **Two worlds (R17).** The result never replaces the editor's setup: it
  installs a **result world** — a separate `ViewerState` built from `stack`
  (default: the stack it was built from) with its own adopted specs, styles and
  cached geometry — that only the Result tab shows. The editor's stack, specs
  and level styles stay authoritative and untouched.
- **Phasor-family overlays only.** Detectors with a `"phasor"` state key
  (`PhasorDetector`, `ModeOverlapDetector`) shade (R15); every other detector
  is listed with a plain glyph and is selectable but never shaded.
- **Run lock.** `load_result` while a run is in flight raises `RuntimeError`
  (D15). A refused result raises `ValueError` and leaves the previous result,
  specs and overlays untouched.
- **After adoption a later `run()` is effectively fresh**: the adopted
  detectors are read-only descriptions of the foreign result and are excluded
  from `build_objects` (D18).

AC:
- A foreign result whose stack differs from the viewer's is loaded by passing
  `stack=...`; switching to the Result tab shows the new box and the adopted
  specs, while the editor views keep the viewer's own stack and specs
  (checked in a test and kernel-side against a live foreign result,
  2026-10-16).
- A result missing `detector_states` raises and leaves the previous result and
  overlays unchanged (checked in a test).
- `load_result` during a run raises `RuntimeError` (checked in a test).
- Adopted specs round-trip through the wire as valid `SpecPayload`s (checked in
  a test).

### R17. Result tab with a frequency slider: *Must* (2026-10-16)

The explorer gains a fourth tab, **Result**, that holds each detector's field
*plotting* options (the `field_*` group) for every detector in the last result
— the widget's own runs and adopted foreign results alike — instead of the
Sources/Detectors tab, which keeps authoring params only.

- **Two worlds, one wire.** A result installs its own `ViewerState` (the
  *result world*) beside the editor's; the active tab routes which world the
  single `geometry`/`meta`/`section`/`state` traits carry. The Result tab
  shows the result world (its geometry, its level styles, the adopted specs);
  every other tab shows the editor's, unchanged. On a tab switch the whole set
  is re-sent, `state` first so a rebuild sees the world's styles.
- **Shading only on the Result tab.** The field payload rides along whenever a
  result exists, but the browser shades only while the Result tab is active —
  the editor's views never show field overlays.
- **Frequency slider.** A multi-frequency detector (a `PhasorDetector` with
  several `wave_characters`) gets a slider selecting which frequency frame is
  shaded, with a readout of the frequency in THz. The kernel sends **all**
  frames in the payload (`freqs` in Hz plus the full `(num_freqs, v, u)`
  buffers); the browser slices locally, so a slider move re-shades with no
  kernel round-trip. Single-frequency detectors get no slider.
- The controls write the same `field_*` params (plus the new `field_freq` and
  `field_outline`), so shading, the colorbar and persistence work unchanged; a
  single detector yields the same UI with one section, and no result shows a
  short note.
- **Collapsible sections (2026-10-16).** Each detector's section is a
  `gfx-spec` card like the Sources/Detectors rows: a twisty collapses the
  controls (view-only state, kept across re-renders), and the section header
  carries the visibility checkbox **unlabeled before the name** — it writes the
  spec's `visible` param, so unchecking hides the whole detector (glyph, 2D
  shading and colorbar entry), not just the overlay. A `field_outline`
  checkbox (default on) toggles the outline drawn around the plane while it is
  shaded.
- **Sources section.** The result world's adopted sources are listed under a
  Sources heading — one read-only row each, an unlabeled visibility checkbox
  writing `visible` on the adopted source's spec, then its name. Adoption now
  covers `objects.sources` as well as detectors (the same "None"-port spec,
  glyph from the placed `grid_slice`, no field params).
- **Layers section.** The Result tab opens with its own **Layers** section —
  one compact row per shaded detector, styled like the editor's Layers tab:
  show checkbox, color swatch (the ramp's high color) and opacity slider. It
  styles the *result overlays* only and shares the same `field_visible` /
  `field_color_high` / `field_opacity` params as the per-detector section
  below it, so the two views stay in sync through the kernel round-trip. The
  editor's level colors (the Layers tab on Layout/Ports) are untouched.

AC:
- A multi-frequency payload carries every frame, `freqs` from the detector's
  wave characters, and `freq = 0` by default (checked in a test).
- `field_freq` round-trips through the spec patch path like the other
  display params (checked in a test).
- A single-frequency payload's buffers are unchanged in layout — one frame
  (checked in a test; the whole R15 field suite guards this).
- A Layers-row patch (`field_visible`/`field_color_high`/`field_opacity`)
  round-trips on an adopted spec too — display params are allowed there
  (checked kernel-side against a live adopted result, 2026-10-16).
- A tab switch re-sends the active world's `geometry`/`meta`/`section`/`state`:
  Result carries the result world's box and adopted specs, the editor tabs the
  viewer's own (checked kernel-side against a live foreign result with a
  different box, 2026-10-16).
- Field shading appears only while the Result tab is active (checked kernel-side:
  the payload is published on any tab, the browser's `applyFields` gates on
  `state.active_tab === "result"`).
- A Result-tab display edit (field params on a detector, `visible` on a source)
  is routed to the result world while its tab is active, and to the editor's
  setup otherwise — the same update_spec event serves both tabs (checked in a
  test: `test_result_tab_routes_source_visibility_to_the_result_world`).
- Adoption covers sources: one adopted spec per placed source with a
  grid-derived glyph and no field params (checked in
  `test_adopt_result_adopts_sources_with_visibility`).

### R18. "None" port: absolute placement: *Must* (2026-10-16)

A source/detector's port select gains a **"None"** choice that replaces the
port-dependent controls with absolute placement params: `axis` (the plane's
normal, 0/1/2 = x/y/z), `center` (x, y, z in µm, stack coordinates) and `size`
(in-plane extents in µm; plane types only). Point types (dipole) take `center`
only.

- The placement constraint mirrors `StackDevice._place_at`: margins from the
  stack box center to `center`, so absolute and port-frame placement land in
  the same frame and resolve through the same `place_objects`.
- **Mode sources/detectors work on any plane** — the mode solve runs on the
  permittivity cross-section the placed plane cuts (`mode.py` →
  `compute_mode`), not on a port. Caveat: a plane cutting nothing guided yields
  radiation modes (`mode_index`/`filter_pol` index those) — a user-facing note,
  not a restriction.
- Adopted foreign detectors use the same representation, read-only (D18), so
  the two features share one path.

AC:
- A "None"-port detector's glyph sits where the params say (in the meshes'
  local frame) and moves when `center` is edited (checked in a test).
- The constraint resolves through a real `place_objects` to the requested
  center, within grid snapping (checked in a test).
- A port-placed detector and an absolute one at the same center land on the
  same grid cells within one cell (checked in a test).
- Switching back to a real port restores the port-frame path (checked in a
  test).

---

## 7. Decisions

### D1. Browser UI with a Python backend on the compute node: *Accepted* (2026-10-12)

- **Options:** (a) web frontend + Python backend; (b) Qt desktop app; (c) KLayout plugin.
- **Decision:** (a). The UI runs in a **Jupyter or marimo notebook** in the
  browser; the Python side runs on the compute node. Accepted (2026-10-12).
- **Why:** C1. It needs no X forwarding, WebGL renders on the laptop, and the
  Python side keeps direct access to JAX and fdtdx.
- **Host:** the notebook host itself is D2 (`anywidget`).

### D2. Host: `anywidget` notebook widget: *Accepted* (2026-10-12, Q1)

- **Decision:** v1 is an `anywidget`. The Jupyter kernel is the backend. It
  works in JupyterLab, VS Code and marimo. One JS frontend and one message
  protocol.
- **Superseded:** the earlier proposal of a standalone FastAPI + websocket app
  (D2b) is dropped from v1. It served the same bundle, so it could return later
  if the notebook layout proves too cramped, but it is not a v1 deliverable.
- **Concern:** the explorer plus two viewports is an app-shaped layout and may
  be cramped in a notebook cell. The viewer is one app-shell widget (D9), so it
  cannot be split across cells; the fallback if it is too cramped is to give it
  its own tab or a full-width output area.
- **Resolves Q1.**

### D3. 3D solid generation: *Accepted* (2026-10-07, revised the same day)

**Decision:** solids are built analytically from the polygon vertices and match
`StackDevice`'s geometry model exactly (C2). No voxelization or sampled fields.

- (a) Mitred vertex offset (move each vertex along its corner bisector):
  *Rejected*.
  - It doesn't match the model at corners. In the model, corners that point
    the way the boundary moves become arcs: convex corners when growing,
    concave corners when shrinking. At a right-angle corner the mitre tip is
    `(√2 − 1)·r` ≈ `0.41·r` off. For example, with `r = 0.22·tan 10°` ≈ 39 nm
    it is off by about 16 nm. See Q17.
  - It breaks when edges collapse while shrinking.
- (b) Marching cubes on the sampled signed field: *Rejected* (2026-10-07).
  It samples the geometry, which the owner ruled out.
- (c) Reuse `export/to_3d.py`: *Rejected* (no sidewalls).
- (d) Analytic offset solids built with 3D booleans: *Accepted*.

Construction (*Proposed*; the implementer may refine it):
1. **Merge.** Per source layer, merge the polygons into their exact union
   (shapely `unary_union`), because the model measures distance to the union's
   boundary.
2. **Split z.** Per level, split `[z_lo, z_hi]` at `z_ref` (where the offset
   changes sign) and at the `z_to_bias` breakpoints. On each interval the offset
   `r(z)` is linear and has one sign.
3. **Prism or vertical walls.** If `r` is constant on the interval (vertical
   walls, the common case), the solid is a prism of the 2D offset polygon. Use
   shapely `buffer` with round joins, which is exact up to arc tessellation.
4. **Sloped walls.** Otherwise, for each boundary edge `e`, the set
   `{(x, z) : dist(x, e) ≤ |r(z)|}` is convex. It is the convex hull of the two
   end "stadiums" (the segment grown by `|r|` at each end of the interval).
   Its planar faces are exact, and its conical faces at the vertices are
   tessellated.
   - Growing: `prism(P) ∪ ⋃ hull_e`.
   - Shrinking: `prism(P) − ⋃ hull_e`.

   This handles features that merge or vanish along z without tracking topology.
5. **Derived layers.** Combine node solids with booleans that follow C2's
   min/max semantics (`or` = ∪, `and` = ∩, `not` = A minus B at the opposite
   threshold). Each node's sizing grow adds to its offset.
6. **Fill levels.** A fill level is the box slab between its z surfaces.
7. **Painting.** Each level's solid is where it wins by `mesh_order` (Q13,
   resolved: painted): intersect it with the box, then subtract every
   higher-priority level. A level that loses everywhere has an empty mesh; the
   unpainted "show full extent" toggle (R3, *Should*) shows its full shape.

- **Library:** `manifold3d` for the booleans (robust, guaranteed-manifold
  output, pip wheels). It is new, so it goes in the `[gui]` extra (C3).
  scipy's `ConvexHull` and shapely are already dependencies.
- **Risk:** the edge-hull count is the number of boundary edges times the
  number of z intervals. Long curved waveguides have thousands of edges.
  Vertical-wall levels skip step 4 entirely. Measure the time on F1–F3 in M1.

### D4. Cross-section computation: *Accepted* (2026-10-07, revised the same day)

**Decision:** cross-sections are analytic vector polygons: slices of the same
painted solids that D3 builds. Sections need no simulation set up and stay sharp
at any zoom.

- (a) Per-pixel evaluation of the signed field at screen resolution:
  *Rejected* (2026-10-07). It samples the geometry.
- (b) Solver grid, slicing `material_weights` on the placed fdtdx grid:
  *Deferred* to R7.
- (c) Slices of the analytic solids: *Accepted*.

Implementation notes (*Proposed*):
- **Slicing gives sharp, painted sections.** A plane through a closed mesh gives
  closed polygons. Planar faces are cut exactly, and arcs carry the same
  tessellation as the 3D view, so the 2D and 3D viewports always agree. The
  painted solids are disjoint, so the sections are already painted.
- **Slice in the browser.** Intersect the plane with the meshes the client
  already holds, stitch the segments into loops, and fill them. The slider
  then needs no round trip, and zoom and pan are local. Server-side slicing
  (`trimesh.section`) is the fallback if this is too slow.
- **Background.** The `background` region is the box section minus all levels.
- **Hover readout.** The hover readout (R4) is a point-in-polygon test on the
  section polygons.

### D5. Wire format: *Proposed*

- **Geometry:** one closed mesh per painted level, as binary arrays (float32
  positions per C5, uint32 indices), plus the level/port metadata as JSON.
- **Cross-sections:** computed in the browser from the meshes (D4), so they add
  nothing to the wire.
- Colors and opacity are applied by the client; edits are coordinated through the
  shared local state (D10).

### D6. Frontend stack: *Proposed*

- **Proposal:** TypeScript + three.js, bundled with esbuild into one ES module
  (required by `anywidget`).
- **UI framework:** plain DOM / Lit unless the explorer's complexity justifies
  React.

### D7. Inputs and entry point: *Accepted* (2026-10-12)

- **Module and function:** `gdsfactory.viewer`, with `view(stack: StackDevice)`
  as the entry point. A subpackage matches the house layout (D8 also assumes
  `gdsfactory/viewer/`), and there is no existing `view`/`viewer` in the package
  to collide with. The module path namespaces the short function name, so a
  future full-chip or schematic viewer can live elsewhere.
- **Returned handle:** `view()` returns the app-shell widget (D9) — call it the
  **`Viewer`** — which carries the whole setup and the three run modes:
  - `.stack` — the `StackDevice`.
  - `.specs` — the source/detector specs (R12).
  - `.settings` — the simulation settings (R13).
  - `.objects`, `.config`, `.constraints` — the export (R14 mode 2). These are
    **computed on access**, calling `create_*_at` per spec (D11), so they are
    properties, not stored attributes.
  - `.launch_code()` — returns the tail snippet as a string (R14 mode 3, D12).
  - `.run()` — places and runs in the kernel (R14 mode 1).
- **Later:** a `view(component, layer_stack)` convenience that builds the
  `StackDevice` itself (it needs material names and a background).

### D8. Analytic geometry builder, read-only on `StackDevice`: *Accepted* (2026-10-12)

(Revised 2026-10-07. The earlier proposal, a grid-independent field evaluator
added to `fdtdx_stack.py`, is withdrawn: the analytic approach doesn't need it.)

- **What:** a viewer-side module, `gdsfactory/viewer/geometry.py`, that builds
  D3's solids from data `StackDevice` already exposes before any simulation is
  set up:
  - `levels`: the compiled levels, with `expr`, `zmin`, `thickness`,
    `sidewall_angle`, `width_to_z`, `bias`, `z_to_bias`, `mesh_order`, `material`
  - `source_layers`, `source_vertices` and `source_sizes`: the packed polygon
    vertices in gds µm
  - `background`, `ports`
  - `center`, `z_center` and `partial_real_shape`: the box, in µm, µm and m
- **No core changes (C4).** It reads only those existing attributes, so it does
  not touch `fdtdx_stack.py` and cannot conflict with changes to it.
- **One weak point:** `levels` holds the private `_Level` type. If its fields
  change, the builder breaks; the parity tests catch that. Making the type
  public could come later.
- **Consistency comes from tests**, since the viewer doesn't share code with the
  voxelizer:
  1. **Model parity.** At random points in the box, a point is inside a level's
     analytic solid exactly when the model's field is negative there. The test
     computes that field with `fdtdx_geometry.signed_distance` (which takes
     arbitrary query points) plus the C2 offsets. Points within the arc
     tolerance (1 nm, Q16) of a boundary are excluded.
  2. **Solver parity.** Place F1 and F3 on an fdtdx grid in the test only.
     Every voxel center more than one cell from an analytic boundary lies in
     the level whose `material_weights(apply_roughness=False)` weight is 1.
     Voxels near interfaces hold averaged fractions, so they are excluded.
     (Needs F3, which is still to be built.)
- **Inherited limit, must be reported:** layer sizings that both grow and shrink
  are rejected by `_sizing`. The builder detects this and raises, rather than
  silently mis-rendering the level.

### D9. One app-shell widget owns the menubar and both viewports: *Accepted* (2026-10-12, revised the same day)

- **Decision:** a single `anywidget` owns the menubar and both viewports: the 3D
  viewport and the 2D cross-section viewport. It renders both viewports' canvases
  in its own DOM, in a flexbox layout, and switches between them from the
  menubar.
- **Why:** the menubar is trivial to build and style when the shell owns it, and
  a draggable splitter (Q4) is easy in a single flexbox. It also avoids the
  embedding problem: one widget can lay out two canvases directly, whereas
  separate widgets would each need their own DOM root and something external to
  place them.
- **Reverses** the earlier same-day decision that the viewports were separate
  widgets (which would have needed stock ipywidgets containers to arrange them).
  The two viewports are now regions of one widget, not widgets themselves.
- **Shared state:** geometry (D5), layer/port metadata, colors and selection live
  in one place inside the shell, so R2 color edits and R1 selection update both
  viewport canvases. Coordination is through the local Python environment (D10),
  so an edit may take a kernel round trip. The menubar sets
  `active_viewport = "3d" | "2d"`; only one viewport is drawn at a time.
- **Cost:** the two viewports are no longer independently usable — you cannot
  mount just the 2D view on its own, and everything shares one widget root.
- **Relates to Q4** (arrangement) and **Q11** (selection).

### D10. The `StackDevice` and the widgets share one Python environment: *Accepted* (2026-10-12)

- **Decision:** the `StackDevice` is a local object in the same Python process as
  the widgets. `view(stack)` reads it directly; there is no separate backend
  service and no serialized model protocol.
- **Coordination is in Python.** The explorer and the app-shell widget share
  selection and layer state through this local environment (the kernel), not
  through browser-side `jslink`. A selection or color change therefore crosses
  to the kernel and back.
- **Consequence:** R1's selection and R2's color edits may take a kernel round
  trip. Their ACs no longer promise otherwise — see R2's AC and D5.
- **What it does not change:** C1 still holds. The geometry evaluation and JAX
  work run on the compute node; the browser still renders. "Local" means the
  model is in the kernel's process, not that the UI is in the kernel.
- **Rejected alternative:** browser-side `jslink` between separate widgets
  (which would keep the edits off the kernel). Not used; coordination is in
  Python.
- **Fallback:** if kernel round trips make interaction feel laggy, the cheapest
  fix is to move only the view state (colors, selection) back to a browser-side
  link, while the geometry build stays in the kernel.

### D11. Sources and detectors are declarative specs; `create_*_at` runs on retrieval: *Accepted* (2026-10-12)

- **Decision:** the widget holds sources and detectors as *specs* — a type, a
  port, and the editable parameters (R12) — not as live fdtdx objects. It calls
  `stack.create_*_at` only when the user retrieves the prepared objects and
  constraints, and then once per spec.
- **Why:** `create_*_at` is a Python call on the kernel (D10). Deferring it keeps
  editing cheap and interactive (a `port_offset` change redraws the glyph from
  the spec, with no `create_*_at` and no fdtdx object churn), and it means the
  geometry the viewport draws pre-placement is derived from the same spec the
  retrieval will use — no divergence.
- **Retrieval:** exposes the accumulated `objects`, `config` and `constraints`
  for `fdtdx.place_objects` (this is R14 mode 2). Placement and running are R14;
  the widget's own run is mode 1.
- **Pre-placement display:** glyphs are drawn from the spec and the port frame
  (D8), so the viewport is a pre-flight view (W1). `port_offset` moves the glyph.
- **Gesture:** double-click a port glyph **in the 3D viewport** to add a default
  detector; sources are added from the explorer. (2D picking and the double-click
  gesture in 2D are Q11.)
- **C2 unaffected:** these objects are fdtdx boxes and planes, analytic by
  construction; they are not `StackDevice` geometry and are never voxelized.

### D12. Three run modes; the third returns the script's tail: *Accepted* (2026-10-12)

The widget can run the simulation three ways (R14): in the widget, by export, or
by returning launch code. All three come from one prepared setup, so the design
question is only how the setup travels to the run.

- **Mode 1 (in the widget).** Run in the kernel the widget occupies (D10), on the
  compute node (C1). Blocks the worker thread; the Run button toggles to Stop
  while it is in flight (D17). Progress is streamed from fdtdx's
  `progress_callback` (D15), along with the run's phase.
- **Mode 2 (export).** Return the prepared `objects`, `config` and `constraints`
  as Python values.
- **Mode 3 (launch code).** Return a self-contained snippet of Python the user
  appends to their own script and runs under `srun`/`sbatch`.

**Mode 3: emit the tail, not the whole script.** The user already has the code
that builds the `StackDevice` — that is their script's head. So the widget does
not reconstruct the device or record a build recipe. It emits only the *tail*:

1. create each source/detector from its spec (`stack.create_*_at`, D11);
2. create the boundary/PML objects from the R13 settings;
3. build the `SimulationConfig` and the `constraints`;
4. `fdtdx.place_objects`, `apply_stack`, `fdtdx.run_fdtd`.

The snippet **assumes `stack` is in scope** and uses it directly, so it is
compact and exact — no serialization, no recipe replay, no component-capture
problem. It is readable: it reads like the tail of
`examples/fdtdx_elevator_coupler.py`.

- **Must use `create_<object>_at`.** Every source and detector in the snippet is
  created through `stack.create_<object>_at` (as in D11), never by
  hand-constructing an fdtdx object. This is what makes the snippet inherit the
  port frame, plane sizing and constraints for free, and keeps it consistent with
  modes 1 and 2.

- **Assumption:** `stack` and the `fdtdx`/`jax` imports the snippet uses are
  available where it runs. The widget names what it assumes at the top of the
  snippet (a comment or an assertion), so a missing name fails clearly.
- **Deliverable:** a code string (copyable from the widget) and/or a written
  `.py` file. Whether it writes results to a file is Q19.
- **Superseded:** the earlier (a)-vs-(b) exploration (rebuild-from-recipe vs
  serialize-the-data) is withdrawn — both are moot once the user supplies the
  device code.

**Open questions (Q19).** What the snippet includes (does it write results to a
file, and in what format?), and whether the widget also offers a full standalone
script (device build included) as a convenience.

**Milestone:** modes 1 and 2 are M4; mode 3 (the tail emitter) is M5.

### D13. The explorer is three tabs, not one scrolling tree: *Accepted* (2026-10-12)

- **Decision:** the explorer's five sections (R1) are grouped into **three tabs**
  in one panel: **Layout / Ports** (Layers + Ports), **Sources / Detectors**
  (Sources + Detectors) and **Simulation**. A tab bar switches between them; only
  the active tab's sections are shown. The properties pane stays below the tab
  body, outside the tabs, so it is visible whichever tab is active.
- **Why:** the five sections in one column push the properties pane off-screen
  and mix per-device inspection (layers, ports) with authoring surfaces that are
  empty until M4 (sources, detectors, simulation). Grouping by task keeps the
  panel short and gives R12/R13's "tabs" wording a concrete layout.
- **Intent (do not revert to a single tree):** the three tabs are a deliberate
  layout choice, not a placeholder. R1's earlier "a tree with five sections"
  wording is superseded by this grouping; the sections themselves are unchanged.
- **Implemented in M2** (`js/viewer/explorer.ts`, `style.css`); the Sources,
  Detectors and Simulation tabs hold M4 placeholders until R12/R13 land.
- **Relates to R1, R12, R13.**

### D14. Region ticks, the settings screen, and per-plane section state: *Accepted* (2026-10-13)

Three related viewport additions made while implementing M3.

- **Region ticks.** Both viewports outline the simulation region in one light
  color (no per-axis colors) and tick each axis with marks and numeric labels at
  nice 1/2/5×10^n steps, plus an x/y/z **name** label at the middle of that
  axis. This replaces R3's earlier "axis triad and scale bar" *Should*: the
  outline and its ticks carry the same information, and the triad's separate
  lines duplicated the box edges. In 3D the ticks are drawn in the content
  group, so they follow the box when the z multiplier changes; sprite labels
  keep their size under that scale.
- **Settings screen.** A gear button beside **Fit** replaces the viewport with a
  settings screen holding: the **rotation style** (Trackball — the default —
  Arcball or Orbit), a **rotation sensitivity** slider, and the **full extent**
  toggle, which moves here from the menubar. 3D/2D/Fit close the screen. The
  rotation style is a browser-side choice (D10): switching rebuilds the three.js
  control over the same canvas, keeping the camera pose.
- **Section state.** Each 2D plane remembers its own slider position
  (`plane_positions` in `ViewerState`), so switching planes restores where that
  plane was left. The z multiplier is **shared** between the two viewports — one
  menubar slider drives both — and it does **not** affect the xy plane, whose
  vertical axis is y, not z.
- **Save image.** A save button (a camera glyph, the screenshot icon) writes a
  PNG of the active viewport: the 3D renderer's canvas or the 2D canvas. The
  settings screen's **screenshot scope** switches it to capture the **whole
  widget** instead — the menubar, explorer and viewport together — via
  `html2canvas`, with the viewport canvas as a fallback if that capture fails.
  The widget root lives in a shadow root in marimo, which html2canvas cannot
  clone, so the capture first copies the shell into the light DOM with three
  fixes: the widget's own `gfx-` rules are re-inserted inside the clone (they
  are `adoptedStyleSheets` the clone does not inherit); each `<canvas>` is
  replaced by an `<img>` of its current pixels (the 3D renderer keeps its
  drawing buffer via `preserveDrawingBuffer`, so `toDataURL` is not blank); and
  the clone is laid out on screen at the origin behind the page
  (`z-index: -1`), since `foreignObjectRendering` only paints elements the
  browser actually lays out. The capture uses html2canvas's
  `foreignObjectRendering` mode because the host notebook's own stylesheets
  share the document and html2canvas 1.4.1's color parser throws on the modern
  color functions they use (`color()`, `oklch()`, `color-mix()`,
  `light-dark()`), aborting the capture — the browser-native path sidesteps that
  parser, and an `onclone` hook clears the document's `html`/`body` backgrounds,
  which `parseBackgroundColor` reads before that path. The scope defaults to the
  viewport alone and is a per-viewer preference in the browser (localStorage),
  not kernel state.

  A second settings row, **save to**, chooses where the image lands. The default
  is **browser**: the PNG downloads as before. **Server** hands the image to the
  kernel instead — the browser sends the data URL in a `save_image` message, and
  the kernel writes the file under the `save_dir` trait (the process working
  directory, or whatever `view(stack, save_dir=…)` was given), appending a
  `-1`, `-2`, … suffix rather than overwriting an existing file. The written
  path is synced back in the `save_result` trait and shown as a brief message
  over the widget, so a server write is not silent. Like the scope, this is a
  per-viewer preference in localStorage, not kernel state.

  In server mode a **save folder** field sets that directory. A browser cannot
  browse the kernel's filesystem, so the field is a path string: the kernel
  resolves it (a relative path nests under the current folder), creates it, and
  syncs the resolved absolute path back into the `save_dir` trait, which the
  field reflects — so a bad or unwritable path is reported at once, not on the
  next save. `save_dir` is the source of truth, so a kernel-side change to it is
  picked up too.
- **Implemented in M3** (`js/viewer/scene.ts`, `section.ts`, `index.ts`;
  `gdsfactory/viewer/state.py`).
- **Relates to R3, R4, D9, D10, Q8.**

---

### D15. The run streams progress on a background thread: *Accepted* (2026-10-13)

Mode 1 (R14) runs the simulation in the widget's own kernel. Two things follow
from that, both settled here.

- **Off the kernel's main thread.** `.run()` returns immediately and the
  simulation runs on a daemon thread, so the widget stays interactive while it
  works. The setup is **locked** for the duration — `state['running']` is true
  and the kernel rejects setup edits (`sources`, `detectors`, `simulation`),
  since the run has already captured its `objects`/`config`/`constraints`. Read-
  only edits (viewport, styling, tab switches, plane position) stay live, and so
  does **collapsing and expanding a source/detector row** — the twisty is view
  state, not a setup edit.
- **Progress is streamed.** fdtdx's `run_fdtd` takes a `progress_callback(step,
  total)`, so the widget passes one and mirrors it into a synced `progress`
  trait (`{"step", "total"}`, `{}` when idle). The browser draws it as a
  **progress bar** in the menubar, beside Run: determinate once `total` is
  known, an indeterminate sweep before that. Passing a callback also turns
  fdtdx's own tqdm bar **off** (`show_progress=False`), so the two do not
  compete for the same output.
- **The phase is named.** A fraction alone is misleading here: placement and JAX
  compilation dominate a run's wall time and fdtdx reports no progress for
  either, so the bar sits in its indeterminate sweep for most of it. `run`
  therefore also takes a `phase` callback and calls it with `"placing"` (build
  and place the objects), `"compiling"` (inside `run_fdtd` before the first time
  step), then `"stepping"` (time-stepping, where `progress` fires — flipped from
  inside the same callback on its first call). The widget mirrors it into a
  synced `phase` trait (`""` when idle) and the browser shows it as a short
  label **to the left of the bar**.
- **The run can be stopped.** Run **becomes** Stop while a run is in flight (the
  icon and title swap in place). Pressing it sets a cancel flag the worker
  watches; the button disables and the phase label reads **"stopping…"** until
  the worker unwinds and clears `running`. The stop lands at the next time step
  or, if the run is still placing or compiling, at the next phase boundary — see
  D17 for the mechanism and the partial-result setting.
- **Implemented in M4** (`gdsfactory/viewer/widget.py`, `sim.py`;
  `js/viewer/index.ts`, `style.css`).
- **Relates to R14, D10, D12, D17, Q20.**

---

### D16. The field overlay is a two-color map with a transparent end: *Accepted* (2026-10-14)

R15 draws a detector's field on its plane. Two questions were open: what the
**color** ramp is, and what the second channel, **alpha**, should encode.

- **Decision — color and alpha both come from one map.** The overlay maps the
  transformed component onto a ramp between two user-chosen colors,
  `field_color_low` and `field_color_high`, with a **transparent** region. The
  ramp's shape depends on whether the transform is signed:

  - **Signed transforms** (`real`, `imag`, `phase`) get a **color → transparent →
    color** ramp: `field_color_low` at the low end of the plane's range,
    transparent at its midpoint, `field_color_high` at the high end. The midpoint
    is the natural zero of a signed quantity, so a near-zero region reads as
    "nothing here" and the two signs take the two colors.

    ```
    t(u, v)  = (value(u, v) - min) / (max - min)      # the plane's own range
    color    = mix(low, high, t)                      # low at t=0, high at t=1
    alpha    = |2t - 1|                               # 0 at the middle, 1 at each end
    ```

  - **Non-negative transforms** (`abs`, `logabs`) get a **transparent → color**
    ramp: transparent at the low end of the plane's range and `field_color_high`
    at the high end (a single hue fading up from nothing), since there is no
    signed midpoint to punch out.

    ```
    t(u, v)  = (value(u, v) - min) / (max - min)
    color    = mix(transparent, high, t)
    alpha    = t                                      # 0 at the low end, 1 at the high end
    ```

  The `field_opacity` setting multiplies the whole map's alpha, so the user can
  fade the overlay without changing its shape.
- **Alpha is the map, not the intensity.** This is the load-bearing change: the
  overlay's transparency is **not** a separate function of `|field|`. An earlier
  draft scaled alpha by the local magnitude (bright field opaque, weak field
  faint); that is dropped. The only thing that decides a cell's alpha is where
  its *transformed value* sits on the plane's range — a cell at the midpoint is
  transparent no matter how large the field is there, and a cell at either end is
  fully opaque no matter how small. Color and alpha read from the same number, so
  the picture never disagrees with itself.
- **Why this beats a magnitude alpha.** A magnitude-scaled alpha encodes a second
  quantity (how much field) on top of the color (the value), and the two fight:
  a strong field at the range's midpoint would be drawn faint, and a weak field
  at an end would be drawn solid, which reads as a rendering bug. With one map,
  the colorbar tells the whole story — position on the ramp is color *and*
  opacity at once.
- **Per-plane normalization** (Q21, resolved). `min` and `max` are over that
  detector's plane only (R15), so the ramp always spans the plane's own range and
  two detectors with very different amplitudes are each readable. Comparing
  amplitudes *between* detectors is the colorbar's job. There is no shared scale.
- **`logabs` and the ramp.** `logabs` is a transform on the *value* the ramp is
  applied to: `log10(|field|)`, over the plane's own `log10(|field|)` range. Its
  values are non-negative, so the low end of the ramp is the field's smallest
  magnitude and the high end its largest; the transparent middle is the geometric
  midpoint of the plane's log range. Because `log10(0)` is `−∞`, `logabs` floors
  at a **global** dynamic range (Q22, resolved; default 6 decades) below the
  plane's maximum, so a true zero maps to the floor rather than to `−∞`.
- **The fill is replaced, not overlaid.** Once a detector shows a field, its
  plain translucent plane fill (R12) is dropped — the field is what fills the
  rectangle. The detector's outline still marks its extent, and turning
  `field_visible` off restores the plain translucent glyph.
- **Which ramp a transform gets.** `real`, `imag` and `phase` are signed, so
  they take the color→transparent→color ramp and their zero reads as the
  transparent middle. `abs` and `logabs` are non-negative, so they take the
  transparent→color ramp: a single hue (`field_color_high`) fading up from the
  low end of the plane's range. `field_color_low` is unused by the
  non-negative transforms (the transparent end stands in for it).
- **2D fades like the ports.** In the section canvas the overlay's alpha is
  multiplied by the same gaussian the ports use (R4): `exp(-½ (d / fade)²)` with
  `d` the section plane's distance from the detector's plane and `fade` the
  1/e distance derived from the state's `plane_fade_frac`. A field therefore
  fades in as its plane is approached,
  matching how a port outline fades — one fade rule for the 2D viewport.
- **What alpha does not do.** It is not depth cueing and not a selection
  highlight; those are separate. The detector glyph's own opacity (R12) is
  superseded for a detector showing a field (above) and unchanged otherwise.

**Implemented in M4** (`js/viewer/section.ts`, `glyphs.ts`, `index.ts`;
`gdsfactory/viewer/fields.py`, `payload.py`). **Relates to R15, R12, R4, Q21, Q22.**

---

### D17. Stopping is a cooperative loop-condition exit: *Accepted* (2026-10-15)

D15 streams progress from inside the run; the natural place to *abort* one is the
same callback, by raising from it. That works — the raise propagates out of the
traced JAX loop — but it **cannot produce a partial result**: the raise unwinds
the loop *before* `run_fdtd` returns, so no arrays come back. Since the user
wants a setting to *keep* a stopped run's partial fields (below), the stop goes
through fdtdx's own loop condition instead.

- **Decision — stop via `stopping_condition`, not `progress_callback`.** fdtdx's
  `run_fdtd(..., stopping_condition=...)` takes a `StoppingCondition` whose
  `__call__(state, config, objects)` returns whether to keep looping. That is the
  `eqxi.while_loop`'s own condition, so ending it is a **normal loop exit**:
  `run_fdtd` returns the arrays, and the detectors' readings survive up to the
  stopping step. This is what makes `keep_partial` meaningful.
- **Exact, at one host read per step.** `CancelCondition` ANDs in
  `jax.experimental.io_callback(lambda step: flag.is_set(), ...)` with
  `ordered=True`. fdtdx only ever uses `io_callback` in the loop's *body*; using
  it in the *condition* is legal and makes the stop land on the exact requested
  step, with no coarse polling latency. **Verified by probe** on CPU (a flag set
  mid-run at step 5 stopped the loop at step 12; pre-set flags at 0 and 7 stopped
  at those exact steps). **Benchmarked** on F1 on an L40S (10 490 steps, best of
  3): the subclass itself costs nothing (−0.0%), and the per-step host read costs
  **+0.6%** (271.5 s → 272.1 s), so no coarse-modulus gating is needed.
- **A custom condition replaces the default, so it reproduces the time limit.**
  `setup()` fills `max_steps` from `config.time_steps_total` (as fdtdx's own
  `EnergyThresholdCondition` does) and `__call__` returns
  `max_steps > curr and not stop`. Without this the run would never end.
  `run_fdtd` raises `NotImplementedError` if a custom condition is combined with
  `config.gradient_config`; the viewer's config leaves it unset, so this is safe.
- **Stops between phases too.** The condition covers `stepping`. For `placing`
  and `compiling`, `sim.run` checks the flag at each boundary inside placement
  (top of the run, after `build_objects`, after `apply_stack`) and just before
  `run_fdtd`, raising `RunStopped`. So a stop lands promptly in any phase.
- **The flag is a `threading.Event`, read on the host.** It is a frozen field on
  the condition and is never traced, so it may be any Python object with
  `is_set()`. `_cancel_condition_class()` builds and caches the class once, so
  JAX sees the same traced computation across runs and does not recompile.
- **`RunStopped` is the discard contract.** `sim.run(..., cancel=flag,
  keep_partial=False)` raises `RunStopped` after `run_fdtd` returns; the widget
  restores the *previous* `last_result` (snapshotted before the run cleared it),
  so its field overlays stay on screen, and reports `{"status": "stopped"}`.
  With `keep_partial=True` the run returns its `RunResult` normally and the
  widget adopts the partial fields as a `"done"` result.
- **The stop is a trait, and the button is a toggle.** A synced `stopping` bool
  drives the button: disabled and reading "stopping…" while true. `stop()` is
  idempotent, a no-op when no run is in flight, and never clears `running` — the
  worker's `finally` owns that, so the widget never claims a run has ended before
  it has. A new run gets a fresh `Event`.
- **The partial-result setting** is `SimSettings.keep_partial` (default `False`),
  exposed in the Simulation tab as "keep partial run".
- **Implemented in M4** (`gdsfactory/viewer/sim.py`, `widget.py`;
  `js/viewer/index.ts`, `explorer.ts`, `wire.ts`).
- **Relates to R14, D12, D15, R13.**

---

### D18. Adoption rebuilds the ViewerState from the result: *Accepted* (2026-10-16)

How `load_result` (R16) turns a foreign result into display state.

- **A result world per load, never a swap.** The viewer never merges foreign
  detectors into its live specs, and never replaces its editor state either:
  the result gets a separate `ViewerState` built from the result's `stack`
  (default: its own), with fresh default `SimSettings` unless the caller passes
  `settings=`. The result world carries its own cached geometry/section
  payloads; the editor's setup, geometry and level styles are untouched and
  come back untouched when the user leaves the Result tab (R17). The editor's
  `_prepare` cache is keyed per world, so a stale cache cannot survive the
  swap either way.
- **One read-only "None"-port spec per detector.** Every detector in
  `objects.detectors` (names read from the placed objects — fdtdx may
  auto-generate) gets a spec with `port == "None"` (R18) whose `axis`/`center`/
  `size` params describe the placement, in stack µm. The ids live in
  `state.adopted`:
  - display edits (`field_*`, `visible`) pass and patch the display settings;
  - type/port/position edits and removal are refused with `ValueError`, and
    the refused patch re-syncs unchanged (the D10 contract).
- **The glyph is the rendered truth.** An adopted spec's glyph is derived from
  its placed `grid_slice` and the result's resolution (`adopt.glyph_from_grid`),
  not re-derived from the params — so a detector whose real extent differs from
  an axis-aligned reading still draws right. The conversion relies on the fdtdx
  invariant that the volume's grid slices start at 0; if that does not hold the
  spec falls back to a point glyph at the grid centroid.
- **Sub-cell accuracy.** Fields live at Yee half-cell offsets per component;
  the adopted rectangle can be half a cell off the true support. Accepted —
  below the resolution the grid draws at.
- **All frames travel.** A multi-frequency phasor's payload carries the full
  `(num_freqs, v, u)` float32 stacks (R17); a handful of frames is fine, a
  100-frame detector is ~30 MB per 450×75 plane. Accepted for this cut.
- **Implemented in M4** (`gdsfactory/viewer/adopt.py`, `fields.py`,
  `state.py`, `sim.py`, `widget.py`; `js/viewer/explorer.ts`, `wire.ts`,
  `glyphs.ts`, `section.ts`, `index.ts`).
- **Relates to R15, R16, R17, R18.**

---

## 8. Open questions

- **Q1.** *Resolved 2026-10-12:* an `anywidget` is the v1 host (D2). The
  standalone browser app is dropped from v1.
- **Q2.** *Resolved 2026-10-07:* the 3D solids must match the simulated geometry
  exactly (D3).
- **Q3.** *Resolved 2026-10-07:* the geometric section only (D4a). The
  solver-grid view stays R7.
- **Q4.** Viewport arrangement: one app-shell widget (D9) owns both viewports and
  the menubar, which shows one viewport at a time. Open: side by side instead of
  switched? A resizable splitter? Maximize one?
- **Q5.** Default opacities: fill 0.15 / others 0.9? Is the `background`
  material drawn at all in 3D (probably not) and in 2D (probably yes, faint)?
- **Q6.** Persist color/opacity edits?
  - session only
  - a per-project JSON/YAML file
  - written back to the PDK's `layer_views`
- **Q7.** *Resolved 2026-10-07:* v1 shows ports only. Sources and detectors
  moved to R12 (and, as of 2026-10-12, promoted into v1).
- **Q8.** Slider behavior:
  - step: fixed µm, or the FDTD resolution when known?
  - snapping to level interfaces and port centers?
  - latency / fps target?
- **Q9.** Which of R5–R11 should shape the v1 architecture now, even if built later?
- **Q10.** Should the explorer list only `StackDevice` levels, or also GDS layers
  with no level (labels, pins), shown in 2D only?
- **Q11.** Click-to-select in the viewports (picking) in v1, or explorer-only
  selection? The 3D double-click-to-add gesture (R12, D11) is decided; still
  open: selection picking in either viewport, and whether double-click does
  anything in 2D.
- **Q12.** At what z are port glyphs drawn: the mid-height of the port layer's
  level? What if that layer has no level?
- **Q13.** *Resolved 2026-10-12:* painted. Each level's solid is where it wins by
  `mesh_order`, matching what is simulated (F1's oxide is hollow where `lower`
  sits) and matching D4's sliced, painted sections. A level that loses everywhere
  to a higher-priority one has an empty mesh; the unpainted "show full extent"
  toggle (R3, *Should*) is the escape hatch. D4 slices the painted solids.
- **Q14.** Surface roughness: show only the nominal geometry in v1 (proposed)?
  Roughness is a random field sampled on the fdtdx grid, so it has no analytic
  vertex form and doesn't fit C2. A later option is to draw it as a displacement
  of the top faces.
- **Q15.** Ports with extensions (`extend_ports`): draw the glyph at the
  component's port or at the end of the extension? The extension is part of the
  layer geometry either way.
- **Q16.** *Resolved 2026-10-12:* a fixed 1 nm chord error for arc tessellation.
  Not relative to the box size and not a user setting in v1.
- **Q17.** *Resolved 2026-10-07:* keep the model's exact corners (C2). Mitred
  corners are rejected (D3a). Background:
  - Exact corners are arcs centered on the original vertex with radius `r(z)`:
    convex corners where the boundary grows, concave corners where it shrinks.
  - With `LayerLevel` defaults (`width_to_z = 0`, positive `sidewall_angle`, no
    bias), levels only shrink upward. Convex corners then agree, and only
    concave corners (junctions, notches) differ.
  - Mitred corners differ from what is simulated by `r·(1/cos(θ/2) − 1)` at a
    turn of θ: `0.41·r` at 90°, negligible on finely polygonized bends (D3a).
- **Q18.** Defaults for a newly added source/detector (R12): the default
  `port_size_mult` and default `wave_characters` for a double-click-added
  `ModeOverlapDetector`. *Resolved 2026-10-12:* they come from the Simulation tab
  (R13). Still open: what their initial values are before the user edits them.
- **Q19.** Mode 3 launch code (D12): what the emitted snippet includes, and
  whether it writes results to a file (and in what format). *Resolved
  2026-10-12:* the widget emits only the script's tail and assumes `stack` is in
  scope; no device rebuild or serialization.
- **Q20.** *Resolved 2026-10-14:* the widget's result surface (R14, R15). A
  run's outcome is the `RunResult` on `viewer.last_result` (the fdtdx
  `objects`/`arrays`/`config`), and `viewer.last_error` holds its exception;
  detector fields are read from `arrays.detector_states[name]`. R15 adds
  `viewer.show_result(result)`, which adopts a result the user produced
  elsewhere under the same surface. The `run_result`/`progress`/`phase` traits
  are the browser's summary of a mode-1 run, not the data.
- **Q21.** *Resolved 2026-10-14:* **per-plane.** Each detector's plane is
  normalized to its own maximum (R15, D16), so two planes with very different
  amplitudes are each readable and a newly added detector needs no rescaling of
  the others. The colorbar carries each plane's own range; comparing amplitudes
  *between* detectors is the user's read of the two colorbars, not a shared
  scale. There is no cross-detector shared-scale mode.
- **Q22.** *Resolved 2026-10-14:* **one global setting.** The `logabs` value
  floor (D16) is a single number for the whole widget, not a per-detector field.
  It is a color-ramp floor — the bottom of the `logabs` scale, `log10(max) −
  range` — so one value reads consistently across every detector's plane and the
  user sets it once. The control lives with the other view-wide settings, not in
  a detector row. v1 default: 6 decades.
- **Q23.** *Resolved 2026-10-14:* **strict.** `show_result` (R15) checks the
  result's detectors against the widget's current setup by **name, per-component
  grid shape, grid spacing, and the plane's position** — not just name and
  shape. A result recorded on a re-boxed device (R13) has the same detector
  names and array shapes but a shifted grid, so a shape-only check would silently
  mis-place the field; the position check refuses it instead. The setter raises
  on any mismatch and leaves the previous result and overlays untouched.

---

## 9. Milestones

- **M0. Lateral offsets (D8, no UI).** Read a `StackDevice`; produce each
  level's exact 2D polygons at a given height `z` (merge, offset, derived-layer
  booleans).
  - Exit: model parity (D8 test 1) restricted to xy slices on F1–F3.
- **M1. Solids and painting (no UI).** D3 steps 2–7. Output: per-level painted
  meshes plus levels, ports and box.
  - Exit: both D8 parity tests on F1–F3, R3's exact-width check on F2, and the
    build time recorded (D3 risk).
- **M2. Explorer and 3D viewport.** R1, R2, R3. Exit: their AC.
- **M3. Cross-section.** R4 (D4), plus D14's region ticks, settings screen,
  per-plane section state and save-image button. Exit: R4's AC.
- **M4. Sources and detectors, running, and field overlays.** R12, R13, D11;
  R14 modes 1 and 2; R15 (promoted from R8, D16); R16–R18 (adoption, the
  Result tab, absolute placement — D18).
  Exit: R12 and R13 AC, R14's mode-2 AC, and R15's AC.
- **M5. Launch code and polish.** R14 mode 3 (the tail emitter), then *Should*
  items from R1–R4 and R12.

## 10. Test fixtures

- **F1.** Elevator coupler: `examples/fdtdx_elevator_coupler.py`,
  `build_stack(overlap=5.0)`. No simulation objects are needed in v1.
  - levels `oxide` (WAFER fill, `mesh_order` 9), `lower` and `upper` (SiN,
    `mesh_order` 1), with `background="air"`
  - ports `o1` and `o2`, each with a 1 µm extension
  - box: y ∈ [−2, 2] µm and z ∈ [−1.6, 1.4] µm (the example's `domain()`); the x
    span is `2 * LENGTH - overlap + 2 * (LEAD + EXTENSION)`, so 18 µm at the
    5 µm overlap (22 µm at 1 µm), centered at x = 7.5 µm (9.5 µm at 1 µm)
- **F2.** Straight waveguide, 10° sidewalls, with a fill level. Builder `f2_stack()`.
  - component: `gf.components.straight(length=2.0, width=0.5)` on layer `(1,0)`;
    ports `o1`/`o2` at the ends, no extensions
  - levels:
    - `box`: `LayerLevel(layer="WAFER", zmin=-0.3, thickness=0.3, material="sio2",
      mesh_order=9)` — a fill level
    - `core`: `LayerLevel(layer=(1,0), zmin=0.0, thickness=0.22,
      sidewall_angle=10.0, material="si", mesh_order=1)`
  - `background="air"`
  - exercises: D3 step 4 (sloped walls, the only fixture), `mesh_order` painting
    (`core` dips into the `box` fill), and R3/R4's exact-angle checks
  - assertions (exact — planar faces):
    - top face width = `0.5 − 2·0.22·tan(10°)` = 0.4224 µm; bottom 0.5 µm
      (narrower by exactly `2·thickness·tan(10°)`)
    - xz section is a trapezoid with sides at exactly 10° from vertical
    - D8 model parity (test 1)
- **F3.** A component with a derived layer, plus `bias` and `z_to_bias`. Builder
  `f3_stack()`. _To assemble_ (the shape already exists in
  `tests/test_fdtdx_stack.py`: `circuit()`, `stack()`).
  - component: a 2 µm straight on `WG`, a `SLAB` rectangle, an `ETCH` rectangle
    over the end (the test's `circuit()`)
  - levels:
    - `box`: `LayerLevel(layer="WAFER", zmin=-0.3, thickness=0.3, material="sio2",
      mesh_order=9)` — a fill level
    - `core`: `LayerLevel(layer=LogicalLayer(WG) - LogicalLayer(ETCH),
      derived_layer=LogicalLayer(WG), zmin=0.0, thickness=0.22, bias=0.02,
      material="si", mesh_order=1)` — derived layer + `bias`
    - `slab`: `LayerLevel(layer=SLAB, zmin=0.0, thickness=-0.2, material="air",
      mesh_order=2, z_to_bias=([0.0, 1.0], [0.0, -0.2]))` — `z_to_bias` undercut
  - `background="air"`
  - exercises: D3 step 5 (derived-layer booleans: the `not` in `WG − ETCH`),
    `bias`, and D3 step 2's extra z-splits at the `z_to_bias` breakpoints
  - assertions: D8 model parity (test 1) and solver parity (test 2) — F3 must be
    placeable on an fdtdx grid
- **Fixture builders** live in the viewer's test module (`f2_stack()`, `f3_stack()`,
  importable by the parity tests); F1 is reused by importing `build_stack` from
  the example.

## 11. References

- `docs/fdtdx.md`: workflow and the "Geometry model" section
- `gdsfactory/fdtdx_stack.py`:
  - `StackDevice` and its attributes (D8), `from_layer_stack`
  - `_Level` and `_Port` (compiled level and port data)
  - `_compile_layer` / `_compile_level` (expression nodes, fill / background
    handling) and `_sizing` (layer grows)
  - `material_weights` (reference for the solver-parity test, and for the
    z-offset formula the viewer must reproduce)
- `gdsfactory/fdtdx_geometry.py`: `signed_distance` (takes arbitrary query
  points; used by the model-parity test)
- Libraries: shapely (`unary_union`, `buffer` with round joins), scipy
  (`ConvexHull`), `manifold3d` (3D booleans, new, `[gui]` extra)
- `gdsfactory/technology/layer_stack.py`: `LayerLevel` fields
- `gdsfactory/export/to_3d.py`: existing trimesh export (not used, see C2)

## Changelog

- 2026-10-07: Skeleton created from the initial viewer discussion and the
  requested UI (explorer, layer colors/opacity, 3D viewport, 2D cross-sections).
- 2026-10-07:
  - Accepted D3 (solids from the geometry model, exact match) and D4a
    (geometric cross-section at screen resolution).
  - v1 shows ports only; sources and detectors moved to R11.
  - Resolved Q2, Q3 and Q7.
  - Added D8 (grid-independent evaluator) and Q13–Q16.
- 2026-10-07: Analytic geometry from vertices, no voxelization.
  - Revised D3: analytic offset solids with 3D booleans replace marching cubes.
  - Revised D4: sections are slices of those solids, replacing per-pixel
    evaluation.
  - Revised D5: cross-sections are computed in the browser.
  - Revised D8: a read-only builder replaces changes to `fdtdx_stack.py`, and
    consistency is checked by parity tests.
  - Updated C2, R3/R4 AC, Q13/Q14/Q16 and milestones; added Q17.
- 2026-10-07: Corrected the corner rule in D3a and Q17. Concave corners round
  when shrinking, which is the default case. The 16 nm example no longer cites
  F2, whose rectangle corners are convex and match either way.
- 2026-10-07: Resolved Q17: exact corners (arcs as in the model), not mitred.
- 2026-10-12:
  - Accepted D2: `anywidget` is the v1 host; the standalone app is dropped from
    v1. Resolved Q1.
  - Added D9: one app-shell widget owns the menubar and both viewports, switched
    by the menubar. (Revised the same day from an earlier proposal of separate
    viewport widgets.)
  - Added D10: the `StackDevice` is local to the widgets' Python process, and
    explorer/viewport state is coordinated in Python rather than browser-side
    `jslink`. Softened the R2/R1 "no server round trip" AC and the D5 note.
  - Accepted D1: the UI runs in a Jupyter or marimo notebook (browser), Python on
    the compute node.
  - Promoted sources and detectors into v1 as R12 (D11): the explorer gains
    Sources and Detectors tabs, double-click a 3D port glyph to add a default
    detector, parameters editable in the explorer. `create_*_at` runs only on
    retrieval, so the widget stores declarative specs. Split R11 to volume/PML
    only; added Q18; updated D7 and the milestones.
  - Added R13: a Simulation explorer tab for global settings (wave_character,
    port_size_mult, PML/boundary, simulation config). This is where R12's
    defaults come from (Q18 resolved to R13); R11 is now volume display only.
  - Added R14 and D12: the widget can run the sim three ways — in the widget
    (mode 1), by exporting objects/config/constraints (mode 2), or by returning
    launch code for srun/sbatch (mode 3). Reversed the "prepare, don't run"
    non-goal; run-from-widget blocks and does not stream.
  - Mode 3 settled (D12, Q19): the widget emits only the *tail* of the user's
    script — object creation, placement and the run — and assumes `stack` is
    already in scope. The rebuild-from-recipe vs serialize-the-data question is
    withdrawn.
  - Resolved Q13: solids are painted (each level only where it wins by
    `mesh_order`). Added the unpainted "show full extent" toggle as an R3
    *Should*. Fixed Q7's stale R11 reference.
  - Resolved Q16: a fixed 1 nm chord tolerance for arc tessellation (C2, R4 AC).
  - Accepted D7: module `gdsfactory.viewer`, function `view(stack)`, returning a
    `Viewer` handle (`.stack`, `.specs`, `.settings`, `.objects`/`.config`/
    `.constraints` as computed properties, `.launch_code()`, `.run()`).
  - Accepted D8: the read-only builder at `gdsfactory/viewer/geometry.py`,
    reading existing `StackDevice` attributes, consistency via the two parity
    tests, grow-and-shrink sizings reported rather than silently dropped.
  - Added C6: implementing agents use the `betterfdtd` conda env and add needed
    packages to it (`manifold3d`, `anywidget` still missing there).
  - Defined F2 and F3 (§10): F2 is a straight waveguide with 10° sidewalls plus a
    `box` fill level; F3 is a derived layer (`WG − ETCH`) with `bias` and a
    `z_to_bias` undercut. Both have fills (as F1 does).
  - Added D13: the explorer is three tabs (Layout/Ports, Sources/Detectors,
    Simulation), superseding R1's single five-section tree. Implemented in M2;
    intentional, not a placeholder.
- 2026-10-13: M3 viewport polish; added D14.
  - Region ticks (both viewports): one outline color, tick marks and numeric
    labels at nice steps, an x/y/z name at each axis' midpoint. Replaced R3's
    "axis triad and scale bar" *Should* with the ticked outline.
  - A settings screen (gear beside Fit): rotation style (Trackball default),
    rotation sensitivity, and the full-extent toggle moved off the menubar.
  - R4: each plane remembers its own slider position; the z multiplier is shared
    with the 3D viewport and does not affect the xy plane.
  - Save-image button for the active viewport; added to R3 and R4 *Should*s.
  - Updated R3/R4, D14 and the M3 milestone.
- 2026-10-13 (later): added D15, streaming the mode-1 run's progress.
  - Mode 1 runs on a background thread with the setup locked (D15), and mirrors
    fdtdx's `progress_callback(step, total)` into a synced `progress` trait.
  - The menubar shows a **progress bar** in place of the old "running…" text:
    determinate once the total is known, an indeterminate sweep before that.
  - Narrowed §2's non-goal: mode 1 is still not cancellable, but it now reports
    progress. Updated R14 (prose and AC) and D12.
  - Added a **phase label** to the left of the bar: `run` calls a `phase`
    callback with `placing` → `compiling` → `stepping`, mirrored into a synced
    `phase` trait. Placement and JAX compilation dominate the wall time and
    report no fraction, so the label says which phase the indeterminate bar is
    covering. Updated D15 and R14.
- 2026-10-13 (later still): menubar polish and screenshot scope.
  - The save button's glyph is a **camera** (the screenshot icon), and a new
    settings row, **screenshot scope**, saves the **whole widget** instead of the
    active viewport. It defaults to the viewport and lives in the browser
    (localStorage). Updated R3 and D14.
  - The redundant "gdsfactory viewer" title text is gone from the menubar.
  - Menubar order is now: 3D, 2D, Fit, the shared **z multiplier**, then a
    spacer, then the run's phase label and progress bar, Run, Save and Settings.
  - Fixed the whole-widget capture, which had been saving the viewport: the
    shell lives in a shadow root html2canvas cannot clone, so it is copied into
    the light DOM with its `gfx-` rules, its canvases frozen to images, and laid
    out on screen behind the page. Verified end-to-end in marimo. Updated D14.
  - Added a **save to** setting: the camera button can write its PNG in the
    browser (download, the default) or hand it to the kernel, which writes it
    under `Viewer.save_dir` (`view(stack, save_dir=…)`) and syncs the path back
    in a new `save_result` trait, shown as a brief message over the widget.
    Updated R3 and D14.
  - Added a **save folder** field for server-side saves: the kernel resolves and
    creates the path, and syncs it back through the now-synced `save_dir` trait.
    Updated D14.
- 2026-10-14: `boundary_offset` as a live Simulation setting (R13).
  - The Simulation tab gains a **boundary offset** row: three signed pairs (x, y, z),
    each a `+`/`−` number, editing `boundary_offset` `(x+, x-, y+, y-, z+, z-)`.
    A value moves its face of the `StackDevice` box along the axis, positive
    towards +x/+y/+z, without changing the geometry.
  - Editing it re-boxes the device in the kernel: the `StackDevice` is
    reconstructed with the moved faces (`partial_real_shape`/`center`/
    `z_center`), and the solids, levels, ports and plane ports are rebuilt from
    it. The widget then re-sends the `geometry` and `meta` traits (D5), which
    the browser watches (`change:geometry`/`change:meta`) to rebuild the 3D
    meshes, the port glyphs, the scene bounds and the 2D section live. This is
    the first setting that re-sends the mesh payloads rather than only the small
    state (D10).
  - The section slider range follows the new box; a slider parked outside a
    shrunk box is clamped back in. A box that would collapse is refused.
  - Updated R13 (the field and its AC).
- 2026-10-14: F1's fixture overlap raised from 1 µm to 5 µm (§10). The shorter
  overlap shortens the box (x 22 µm → 18 µm), so the spec's F1 box note and the
  yz slider-range test now derive from the stack rather than a fixed value.
- 2026-10-14: field overlays promoted into M4 (R8 → R15, D16).
  - **R15** shades a detector's plane from the field it recorded, in both
    viewports. Per-detector display settings on the Detectors tab: the field
    component (from the detector's own `components`), the transform (`real`,
    `imag`, `abs`, `phase`, `logabs`), the two endpoint colors, the overlay
    opacity and a visibility toggle. Display settings are display-only spec
    params (D11), so they never reach `create_*_at` and survive a re-run.
  - **`viewer.show_result(result)`** adopts a result the user produced
    elsewhere (their own `srun` run) as the widget's current result, redrawing
    the overlays with no re-run. It validates the result's detectors against the
    setup and raises on a mismatch (Q23).
  - **D16** fixes the overlay's color and transparency: a **color → transparent
    → color** ramp between two per-detector endpoint colors (`field_color_low`,
    `field_color_high`), transparent at the plane's range midpoint. Alpha is the
    map alone — **not** a separate magnitude-scaled alpha. `logabs` is a
    transform on the ramp's input, not an alpha mode; it floors at the global
    `log_range` (Q22). The detector's translucent fill is replaced by the field
    while it is shown, and in 2D the overlay fades with the same gaussian as the
    ports (R4).
  - **Q20** (referenced since D15 but never written) now defines the widget's
    result surface: `last_result`, `last_error`, and `show_result`.
  - M4's exit criterion gains R15's AC. R8 is superseded; Q9's promotion of R8
    is answered.
- 2026-10-14: Q21 and Q23 resolved.
  - **Q21 — per-plane scale.** R15 normalizes each detector's plane to its own
    maximum; no cross-detector shared scale. A new detector needs no rescaling
    of the others, and the colorbar carries each plane's own range.
  - **Q23 — strict `show_result` validation.** It checks detector names, grid
    shape, grid spacing and the plane's position, not name and shape alone, so a
    result recorded on a re-boxed device (R13) is refused rather than mis-placed.
  - **Q22 — the `logabs` floor is a global setting.** One widget-wide number for
    the bottom of the `logabs` color ramp (default 6 decades), not a per-detector
    field, so the scale reads consistently across every detector's plane. It
    lives with the other view-wide settings, not in a detector row.
- 2026-10-14: D16 revised — the field overlay is a two-color map, and 2D fades
  like the ports.
  - The per-detector colormap is a ramp between two user-chosen colors
    (`field_color_low`, `field_color_high`) with a transparent region: the signed
    transforms (`real`, `imag`, `phase`) get color → transparent → color
    (transparent at the plane's range midpoint), the non-negative transforms
    (`abs`, `logabs`) get transparent → color (a single hue fading up from the
    low end). This replaces the named-colormap setting.
  - **Alpha is the map alone.** The earlier magnitude-scaled alpha
    (`|field| / max|field|`) is dropped: transparency no longer tracks intensity,
    only where the transformed value sits on the plane's range.
  - A detector showing a field **loses its translucent fill** — the field fills
    the rectangle, the outline remains.
  - The 2D overlay **fades with the same gaussian as the ports** (R4), so a
    field fades in as its plane is approached.
- 2026-10-14: R15 gains `|E|` and pins the payload's axis order.
  - **`|E|`** joins `field_component` as a pseudo-component —
    `sqrt(|Ex|² + |Ey|² + |Ez|²)` over the detector's recorded `E` components,
    derived in the browser. It is offered only when the result recorded any `E`
    component, and composes with every transform and ramp like a real component.
  - The field payload's buffer is **row-major in (v, u)** with
    `grid = [len(v), len(u)]`; `field_payload` drops the plane's normal axis and
    transposes the rest. Both viewports read it that way and the 2D blit flips
    the raster, so an xz plane's field no longer renders transposed.
- 2026-10-15: plane `level`, `z_center`, and two 2D glyph fixes.
  - **A plane honors its `level`** (it was ignored): the level's span middle
    sets the plane's height — the plane's position for `xy`, its in-plane
    vertical center for `xz` / `yz`. R12's earlier "a plane has no meaningful
    `level`" is reversed.
  - **`z_center` joins the `level` list**: it pins an object to the simulation
    volume's mid-height, on a plane or a port. On a port it keeps the level's
    height and only moves z.
  - **A crossing glyph is culled outside its own extent** along the section
    normal, so a mode plane source on `o1` no longer appears at every z in the
    `xy` view.
  - **A parallel glyph fades** with the same gaussian as the ports (R4) rather
    than popping in at a fixed coincidence tolerance.
- 2026-10-15: the plane fade and 2D draw-order fixes.
  - **"Port fade" is now "plane fade"** and is a **fraction, not a distance**:
    `plane_fade_frac` (default 0.05) scales the box size along the section's
    normal to give the gaussian 1/e distance, so the same setting suits every
    simulation size and plane. It governs ports, source/detector glyphs and
    field overlays alike (R4).
  - **A crossing glyph fades too**: a plane seen edge-on (a mode-plane source
    on `o1` in the `xy` view) fades with the same gaussian instead of popping
    in solid.
  - **The region outline is drawn under the glyphs**, so a full-volume `xy`
    detector is no longer hidden by the simulation boundary in the `xy` view.
- 2026-10-15: the source/detector collapse arrow stays live during a run.
  - `setLocked` disabled the **first** button in a spec row, which is the
    collapse twisty rather than the remove button, so collapsing and expanding
    a source/detector row froze while a run was in flight. The lock now targets
    the remove button directly (D15, R12).
- 2026-10-15: the 2D field blit's vertical mirror, and the stop button.
  - The 2D detector field was drawn with a **negative** `drawImage` height, which
    Chromium does not treat as a flip: it normalises the rect and draws the raster
    unflipped, so row 0 (the low-`v` row) landed at the high-`v` screen edge and
    every field in the 2D panels was mirrored vertically. The blit now translates
    to the rect's low-`v` corner and flips `y` in the transform, drawing with a
    positive height. 3D was unaffected (`DataTexture.flipY = false` with
    `PlaneGeometry`'s `uv.y = 0` at the low-`v` edge). Guarded by
    `js/viewer/sectionblit.test.mjs` (R15, D16).
  - Added **D17**: a mode-1 run can be stopped, via fdtdx's `stopping_condition`
    rather than by raising from `progress_callback` — only a normal loop exit
    returns the arrays, which is what lets a stopped run keep its partial fields.
    The Run button toggles to Stop in place, greys out and reads "stopping…" until
    the worker unwinds, and the stop lands at the next time step or phase
    boundary. Added the `keep_partial` simulation setting (default: discard, and
    the previous result's overlays stay on screen) and the `stopping` trait.
    Dropped §2's "Cancelling a run from the widget" non-goal, and updated R14
    mode 1, its AC, D12 and D15 (R13, R14, D15).

- 2026-10-16: adopting a foreign completed result. `Viewer.load_result(
  objects, arrays, config, stack=None)` accepts the three pieces fdtdx hands
  out from arbitrary user code and displays them (R16): a **result world** —
  a separate state built from the result's stack — is installed beside the
  editor's, every detector becomes a read-only "None"-port spec with a
  `grid_slice`-derived glyph, phasor-family detectors shade, and lenient
  validation refuses only what could never render (D18). The explorer's new
  **Result** tab holds each result detector's plotting options, and
  multi-frequency detectors get a frequency slider that re-shades client-side —
  the kernel sends all `(num_freqs, v, u)` frames in the payload (R17). A
  source/detector's port select gains a **"None"** choice for absolute
  placement (`axis`/`center`/`size` in stack µm), placed through the same
  margin frame as `_place_at`; mode sources/detectors work on any plane, not
  only ports (R18). The Result tab also opens with a **Layers** section — one
  row per shaded detector with show/color/opacity for the result overlays,
  independent of the editor's level Layers (R17). Later the same day: the
  two-world split — the Result tab shows the result world's geometry and
  styles while every other tab keeps the editor's (loading a result leaves the
  editor untouched), and field shading applies only while the Result tab is
  active (R17, D18). fdtdx 0.6.3 compatibility: `SimulationConfig(grid=...)`,
  `uniform_spacing()`, the 5-tuple `place_objects`, and the PML CPML
  coefficient dtype re-cast (C7). Later still: adoption covers the result's
  **sources** too — one read-only "None"-port spec each, listed in the Result
  tab's new Sources section with an unlabeled visibility checkbox (R17); the
  detector sections became collapsible `gfx-spec` cards with the show checkbox
  unlabeled in the header beside the name, and a `field_outline` display param
  (default on) toggles the plane's outline while shaded (R17). A Result-tab
  display edit routes to the result world by the active tab rather than by
  id-matching, so same-named editor specs cannot be misrouted. The header
  checkbox writes the spec's `visible` param — unchecking hides the whole
  detector (glyph, 2D shading, colorbar), not just the overlay.
