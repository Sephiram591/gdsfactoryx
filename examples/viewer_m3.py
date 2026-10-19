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
    # gdsfactory viewer - M3

    The 2D cross-section viewport (spec R4, D4).

    Switch the menubar to **2D** and check:
    - **Plane toggle:** xy / xz / yz. On F1's xz you should see the two SiN tapers
      (`lower` z -0.22..-0.12, `upper` z 0..0.2) with the oxide below and air
      above. On F2's xz the core's sides taper at 10 degrees.
    - **Position slider:** drag it through the box; the section follows. The
      numeric box accepts an exact value.
    - **Shoulds:** the `z x` slider exaggerates the vertical axis (xz/yz); hover
      the canvas for an axis/level readout; the ports show as short yellow bars
      where the plane cuts them (click one to select it).
    - **Zoom/pan:** wheel to zoom, drag to pan. Vector polygons stay sharp.
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


if __name__ == "__main__":
    app.run()
