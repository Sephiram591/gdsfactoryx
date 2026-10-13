import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # gdsfactory is installed editable against the main checkout, which has no
    # gdsfactory/viewer yet. Put this worktree first so the viewer resolves here,
    # then the tests dir for the F1/F2/F3 fixtures.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    sys.path.insert(0, str(root))
    from test_viewer_geometry import f1_stack, f2_stack, f3_stack

    from gdsfactory.viewer import view

    return f1_stack, f2_stack, f3_stack, mo, view


@app.cell
def _(mo):
    mo.md("""
    # gdsfactory viewer - M4

    Sources, detectors and the run (spec R12, R13, R14, D11, D15).

    **Add a detector:** double-click a port glyph in the 3D viewport. It lands on
    the **Sources / Detectors** tab as a `ModeOverlapDetector` row, drawn as a
    translucent plane at the port (before anything is placed, spec D11).

    **Add a source:** click the **+** beside the Sources header.

    **Volume planes:** the port dropdown also offers `xy`, `xz` and `yz` - the plane through the **center of the simulation volume** parallel to those faces. They are not device ports. A plane's width and height are the volume's own in-plane extents, so `port_size_mult = (1, 1)` (the default for a plane) fills the whole simulation plane.

    **Edit a row:** change the type, port, `port_offset` (the plane moves live),
    `port_size_mult`, the wavelength, `level`, `name`, or a type-specific extra.
    None of this calls `create_*_at` — the specs are declarative.

    **Simulation tab:** the default wavelength and plane size feed newly added
    objects (Q18); the PML thickness and the grid settings feed retrieval.

    **Run (spec R14 mode 1, D15):** press **Run** in the menubar. The run starts
    on a background thread, so the widget stays live — orbit, switch tabs, scrub
    the section — while the **setup is locked**: the Sources/Detectors and
    Simulation tabs are disabled and the kernel refuses setup edits. A **progress bar** beside Run tracks
    it (fdtdx's `progress_callback`); there is no cancel (spec §2), so the run ends on its own and unlocks.

    **Export (spec R14 mode 2):** `viewer.objects`, `viewer.config` and
    `viewer.constraints` are computed on access, ready for
    `fdtdx.place_objects`.
    """)
    return


@app.cell
def _(mo):
    fixture = mo.ui.dropdown(options=["F1", "F2", "F3"], value="F1", label="fixture")
    fixture
    return (fixture,)


@app.cell
def _(f1_stack, f2_stack, f3_stack, fixture, view):
    builders = {"F1": f1_stack, "F2": f2_stack, "F3": f3_stack}
    stack = builders[fixture.value]()
    viewer = view(stack)
    return (viewer,)


@app.cell
def _(viewer):
    viewer
    return


@app.cell
def _(mo):
    mo.md("""
    **Export (spec R14 mode 2).** The prepared objects, config and constraints.
    F1's box at the default 20 nm resolution is very large; set a coarser
    resolution on the Simulation tab before placing.
    """)
    return


@app.cell
def _(viewer):
    viewer.specs, viewer.settings
    return


@app.cell
def _(viewer):
    # mode 2: the retrieval; place with fdtdx.place_objects(objects, config,
    # constraints, key) when you are ready (no run happens here)
    viewer.objects, viewer.config, viewer.constraints
    return


if __name__ == "__main__":
    app.run()
