# gdsfactoryx viewer: spec

**Status:** drafting · **Owner:** floresh2 · **Last updated:** 2026-10-12

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
- Cancelling a run from the widget; a run started from the widget (R14 mode 1)
  blocks until it returns. It *does* show a progress bar (D15), but it is not
  cancellable.
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
    the rotation style, its sensitivity and the full-extent toggle (D14)
  - a "show full extent (unpainted)" toggle, per layer or global, to reveal a
    level that a higher-`mesh_order` level covers everywhere (Q13)
  - a "save image" button that writes the active viewport to a PNG (D14)

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
  A spec on a plane has no meaningful `level` (a plane spans the whole volume),
  so `level` is ignored for it.
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

### R13. Simulation settings: *Must*

An explorer tab holding the settings that are shared by the whole setup, as
editable fields:

- **`wave_character`** — the default wavelength(s) used when a source or
  detector is added without one (R12, Q18).
- **`port_size_mult`** — the default plane size multiplier for new
  sources/detectors (R12, Q18).
- **PML / boundary** — the boundary thickness (e.g. the `BoundaryConfig` the
  run would use; `fdtdx.BoundaryConfig.from_uniform_bound(thickness=...)`).
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

### R14. Run the simulation, three ways: *Must*

From the same prepared setup (R12, R13), the widget offers three ways to run,
increasing in separation from the session:

1. **In the widget.** A "Run" action places and runs in the kernel the widget
   already occupies (C1: on the compute node) and returns the results. It
   **blocks** until done; there is no cancel (§2). It **does** report progress:
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

### Later candidates (not v1 unless promoted; see Q9)

- **R5.** Parameter explorer: sliders generated from a cell's signature, with a
  rebuild on change.
- **R6.** Shape-sensitivity overlay (`jax.jacfwd` of unmerged polygon vertices)
  and `dJ/dp` bars per parameter.
- **R7.** Solver view: voxelized permittivity / material weights from the placed
  fdtdx grid, as an alternative 2D mode (D4b).
- **R8.** Field overlays from fdtdx detectors (F1's `field` detector records the
  xz plane through the taper axis).
- **R9.** Optimization history playback (scrub logged parameters per iteration).
- **R10.** Full-chip 2D via `kweb` (already the `cad` extra).
- **R11.** Simulation volume display (PML thickness is set on the Simulation
  tab, R13):
  - translucent volume box in 3D and outline in 2D, with visibility toggles
  - open: also show the PML boundaries in the viewports?

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
  compute node (C1). Blocks; no cancel. Progress is streamed from fdtdx's
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
- **Save image.** A save button (floppy-disk glyph) writes the active viewport
  to a PNG: the 3D renderer's canvas or the 2D canvas.
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
  only edits (viewport, styling, tab switches, plane position) stay live.
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
- **Implemented in M4** (`gdsfactory/viewer/widget.py`, `sim.py`;
  `js/viewer/index.ts`, `style.css`).
- **Relates to R14, D10, D12, Q20.**

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
- **M4. Sources and detectors and running.** R12, R13, D11; R14 modes 1 and 2.
  Exit: R12 and R13 AC, and R14's mode-2 AC.
- **M5. Launch code and polish.** R14 mode 3 (the tail emitter), then *Should*
  items from R1–R4 and R12.

## 10. Test fixtures

- **F1.** Elevator coupler: `examples/fdtdx_elevator_coupler.py`,
  `build_stack(overlap=1.0)`. No simulation objects are needed in v1.
  - levels `oxide` (WAFER fill, `mesh_order` 9), `lower` and `upper` (SiN,
    `mesh_order` 1), with `background="air"`
  - ports `o1` and `o2`, each with a 1 µm extension
  - box: y ∈ [−2, 2] µm and z ∈ [−1.6, 1.4] µm (the example's `domain()`); x is
    the polygons' extent, extensions included
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
