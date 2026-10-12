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
    # gdsfactory viewer - M2

    Explorer, layer colors and the 3D viewport (spec R1, R2, R3).

    Check the acceptance criteria:
    - **R1:** the Layers tree lists 3 layers (`oxide`, `lower`, `upper`) and Ports
      lists 2 (`o1`, `o2`); clicking `o2` highlights its glyph in the viewport.
    - **R2:** `oxide` starts faint (fill opacity 0.15), `lower`/`upper` solid;
      set `upper` to red and the viewport follows.
    - **R3:** uncheck `oxide` and the `lower`/`upper` SiN levels appear with their
      vertical offset; uncheck `o1` and only that glyph goes.
    - **Shoulds:** the z-exaggeration slider, the axis triad and scale bar, and
      the per-layer "show full extent" toggle.
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
