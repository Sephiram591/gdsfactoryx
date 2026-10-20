import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import fdtdx
    import marimo as mo

    import gdsfactory as gf
    from gdsfactory.fdtdx_stack import StackDevice
    from gdsfactory.technology import LayerLevel, LayerStack
    from gdsfactory.viewer import view
    import matplotlib.pyplot as plt

    gf.gpdk.PDK.activate()
    return LayerLevel, LayerStack, StackDevice, fdtdx, gf, view


@app.cell
def _():
    # Elevator coupler: an upper SiN taper (wide -> narrow) hands TE0 down to a lower
    # SiN taper (narrow -> wide) whose tip overlaps it. Lengths in um.
    WAVELENGTH = 0.619
    N_SIN, N_OXIDE = 2.0, 1.457
    UPPER, LOWER = (34, 0), (35, 0)
    LENGTH = 10.0  # each taper
    W_MAX, W_MIN = 0.65, 0.05  # wide end / tip
    OVERLAP = 3.0  # where the two tapers lie on top of each other
    LEAD = 0.5  # straight sections that carry the ports
    return LEAD, LENGTH, LOWER, N_OXIDE, N_SIN, OVERLAP, UPPER, W_MAX, W_MIN


@app.cell
def _(LEAD, LENGTH, LOWER, OVERLAP, UPPER, W_MAX, W_MIN, gf):
    @gf.cell
    def elevator_coupler(
        length: float = LENGTH,
        w_max: float = W_MAX,
        w_min: float = W_MIN,
        overlap: float = OVERLAP,
        lead: float = LEAD,
    ) -> gf.Component:
        """Upper taper (o1, on UPPER) over lower taper (o2, on LOWER), overlapping by ``overlap``."""
        c = gf.Component()
        a, b = w_max / 2, w_min / 2
        x0 = length - overlap  # lower taper starts here
        x1 = x0 + length  # lower taper's wide end

        c.add_polygon(
            [(-lead, -a), (0, -a), (length, -b), (length, b), (0, a), (-lead, a)], layer=UPPER
        )
        c.add_polygon(
            [(x0, -b), (x1, -a), (x1 + lead, -a), (x1 + lead, a), (x1, a), (x0, b)], layer=LOWER
        )
        c.add_port("o1", center=(-lead, 0), width=w_max, orientation=180, layer=UPPER)
        c.add_port("o2", center=(x1 + lead, 0), width=w_max, orientation=0, layer=LOWER)
        return c


    coupler = elevator_coupler()
    coupler.plot(return_fig=True, show_ports=True)
    return (coupler,)


@app.cell
def _(
    LOWER,
    LayerLevel,
    LayerStack,
    N_OXIDE,
    N_SIN,
    StackDevice,
    UPPER,
    W_MAX,
    coupler,
    fdtdx,
):
    LAYER_STACK = LayerStack(
        layers={
            "oxide": LayerLevel(layer="WAFER", zmin=-5.0, thickness=5.0, material="oxide", mesh_order=9),
            "lower": LayerLevel(layer=LOWER, zmin=-0.22, thickness=0.1, material="sin", mesh_order=1),
            "upper": LayerLevel(layer=UPPER, zmin=0.0, thickness=0.2, material="sin", mesh_order=1),
        }
    )
    MATERIALS = {
        "air": fdtdx.Material(),
        "oxide": fdtdx.Material(permittivity=N_OXIDE**2),
        "sin": fdtdx.Material(permittivity=N_SIN**2),
    }

    stack = StackDevice.from_layer_stack(
        coupler, LAYER_STACK, MATERIALS, background="air", name="stack",
        included_layers=["lower", "upper"],
        extend_ports=(("o1", 1.0), ("o2", 1.0)),
        boundary_offset=(0.0, 0.0, 2.0 - W_MAX / 2, -2.0 + W_MAX / 2, 1.4 - 0.2, -1.6 + 0.22),
    )
    return (stack,)


@app.cell
def _(stack, view):
    viewer = view(stack)
    viewer
    return


if __name__ == "__main__":
    app.run()
