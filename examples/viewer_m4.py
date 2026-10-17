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
    # then the tests dir for the F1/F2/F3 fixtures. The kernel may already have
    # gdsfactory imported from the main checkout (via the .pth), so evict it —
    # otherwise `gdsfactory.viewer` searches the old package's __path__ and fails.
    root = Path("/orcd/home/002/floresh2/file_system/git/gdsfactoryx/.claude/worktrees/viewer-m4")
    sys.path.insert(0, str(root / "tests"))
    sys.path.insert(0, str(root))
    for _name in list(sys.modules):
        if _name == "gdsfactory" or _name.startswith("gdsfactory."):
            del sys.modules[_name]
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
    it (fdtdx's `progress_callback`), labelled with the current phase: `placing` and `compiling` (JAX) report
    no fraction and dominate the wall time, so the bar sweeps while the label says which one it is in, then
    `stepping` gives a real fraction. There is no cancel (spec §2), so the run ends on its own and unlocks.

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
def _():
    return


@app.cell
def field_monitor(mo, viewer):

    import matplotlib.pyplot as plt
    import numpy as np


    def _field_on_detector(result, name):
        """|E|^2 on a detector's plane, with um axes from the viewer box."""
        vs = viewer._viewer_state
        res = result.config.resolution * 1e6  # um per cell
        gs = result.objects[name].grid_slice
        x0 = vs.origin[0] - 0.5 * vs.size[0]
        z0 = vs.origin[2] - 0.5 * vs.size[2]
        xs = x0 + (np.arange(gs[0].start, gs[0].stop) + 0.5) * res
        zs = z0 + (np.arange(gs[2].start, gs[2].stop) + 0.5) * res
        # phasor is (1, 1, 6, nx, ny, nz): components Ex,Ey,Ez,Hx,Hy,Hz
        phasor = np.asarray(result.arrays.detector_states[name]["phasor"])
        e = phasor[0, 0, 0:3, :, 0, :]
        return xs, zs, np.sum(np.abs(e) ** 2, axis=0)


    _result = viewer.last_result
    if _result is None:
        _out = mo.md("_No run yet - press **Run** in the viewer, then re-run this cell._")
    else:
        _names = sorted(_result.arrays.detector_states)
        _fig, _axes = plt.subplots(1, len(_names), figsize=(6 * len(_names), 2.6), squeeze=False)
        for _ax, _name in zip(_axes[0], _names):
            _xs, _zs, _e2 = _field_on_detector(_result, _name)
            _im = _ax.imshow(
                _e2.T, origin="lower", aspect="auto",
                extent=(_xs[0], _xs[-1], _zs[0], _zs[-1]), cmap="inferno",
            )
            _ax.set(title=f"{_name}: |E|^2 (y = 0)", xlabel="x (um)", ylabel="z (um)")
            _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
        _fig.tight_layout()
        _out = _fig
    _out
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


@app.cell(hide_code=True)
def foreign_sim(f1_stack):

    import jax
    import jax.numpy as jnp

    import fdtdx
    from fdtdx.core.wavelength import WaveCharacter

    key = jax.random.PRNGKey(42)

    # F1 (the elevator coupler fixture): tapered StackDevice uploaded as a foreign
    # result. o1 mode source at 619 nm + xz-plane phasor detector through the taper.
    foreign_stack = f1_stack()
    print("F1 box (m):", tuple(float(s) for s in foreign_stack.partial_real_shape))

    foreign_config = fdtdx.SimulationConfig(
        grid=fdtdx.core.grid.UniformGrid(spacing=0.1e-6),
        time=300e-15,
        dtype=jnp.float32,
    )
    foreign_volume, _volume_constraints = foreign_stack.get_simulation_volume()

    _wave = WaveCharacter(wavelength=0.619e-6)
    _pulse = fdtdx.GaussianPulseProfile(
        center_wave=_wave,
        spectral_width=fdtdx.WaveCharacter(frequency=0.05 * _wave.get_frequency()),
    )
    foreign_source, _src_constraints = foreign_stack.create_mode_plane_source_at(
        "o1", (4.5, 8), wave_character=_wave, temporal_profile=_pulse,
        port_offset=-0.3, name="source", mode_index=0, filter_pol="te",
    )
    foreign_field = fdtdx.PhasorDetector(
        name="field", wave_characters=(_wave,), components=("Ex", "Ey", "Ez"),
        scaling_mode="pulse", partial_grid_shape=(None, 1, None),
    )
    foreign_constraints = [
        *_volume_constraints, *_src_constraints,
        foreign_field.same_size(foreign_volume, axes=(0, 2)),
        foreign_field.place_at_center(foreign_volume),
    ]

    foreign_boundaries, _bnd_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=4),
        foreign_volume,
    )
    foreign_constraints.extend(_bnd_constraints)

    foreign_objects, foreign_arrays, foreign_params, foreign_config, _ = fdtdx.place_objects(
        object_list=[foreign_volume, foreign_stack, foreign_source, foreign_field,
                     *foreign_boundaries.values()],
        config=foreign_config,
        constraints=foreign_constraints,
        key=key,
    )
    print("grid:", foreign_objects.volume.grid_shape)
    print("detectors:", [(o.name, o.grid_slice) for o in foreign_objects.detectors])
    return (
        fdtdx,
        foreign_arrays,
        foreign_config,
        foreign_objects,
        foreign_params,
        foreign_stack,
        key,
    )


@app.cell(hide_code=True)
def foreign_apply2(
    fdtdx,
    foreign_arrays,
    foreign_config,
    foreign_objects,
    foreign_params,
    key,
):

    from fdtdx.fdtd.container import ObjectContainer

    from gdsfactory.fdtdx_stack import apply_stack, init_stack_params

    def _cast_pml_coeffs(objs, cfg):
        """Re-cast CPML coefficients to the config dtype (jax x64 poisoning fix).

        gdsfactory enables jax x64 globally, so fdtdx derives the PML's a/b/inv_kappa
        as float64 and the CPML update promotes the float32 psi fields — while_loop
        then rejects the changed structure. Cast them back, like sim.py does.
        """
        import jax.numpy as jnp
        out = []
        for obj in objs.object_list:
            if type(obj).__name__ == "PerfectlyMatchedLayer":
                for name in ("pml_a_E", "pml_b_E", "inv_kappa_E", "pml_a_H", "pml_b_H", "inv_kappa_H"):
                    val = getattr(obj, name, None)
                    if val is not None:
                        obj = obj.aset(name, val.astype(cfg.dtype))
            out.append(obj)
        return ObjectContainer(object_list=out, volume_idx=objs.volume_idx)

    foreign_all_params = {**foreign_params, **init_stack_params(foreign_objects)}
    fa2, fo2, _ = apply_stack(foreign_arrays, foreign_objects, foreign_all_params, key)
    fo2 = _cast_pml_coeffs(fo2, foreign_config)
    foreign_state2 = fdtdx.run_fdtd(
        arrays=fa2,
        objects=fo2,
        config=foreign_config,
        key=key,
        show_progress=False,
    )
    print("run done:", type(foreign_state2[1]).__name__)
    return (foreign_state2,)


@app.cell(hide_code=True)
def foreign_adopt2(
    foreign_config,
    foreign_objects,
    foreign_stack,
    foreign_state2,
    viewer,
):

    viewer.load_result(
        foreign_objects, foreign_state2[1], foreign_config,
        stack=foreign_stack,
    )
    print("adopted:", viewer._result_state.adopted)
    print("result specs:", [(s.name, s.port) for s in viewer._result_state.specs])
    print("editor stack:", viewer._viewer_state.stack.partial_real_shape)
    print("result stack:", viewer._result_state.stack.partial_real_shape)
    return


@app.cell(hide_code=True)
def push_bundle():

    import gdsfactory.viewer.widget as W
    W.Viewer.__dict__["_esm"]._contents = None
    print("cache cleared")
    return


@app.cell
def demo_r17(f1_stack, view):

    # Demo (spec R17): F1, o1 source at 619 nm, xz phasor detector.
    # The Result tab shows this run's geometry + detector shading; the editor
    # views keep F1's own stack and are never shaded.
    demo_viewer = view(f1_stack())
    demo_viewer._on_msg(None, {
        "event": "set_settings",
        "settings": {"wavelength_um": 0.619, "resolution_um": 0.2, "time_s": 2e-13},
    }, [])
    demo_viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    demo_viewer._on_msg(None, {"event": "add_detector", "port": "xz"}, [])
    print("specs:", [(s.id, s.name, s.port, s.params.get("wavelengths")) for s in demo_viewer._viewer_state.specs])
    return (demo_viewer,)


@app.cell
def demo_run(demo_viewer):

    demo_viewer.run()
    demo_viewer._run_thread.join(timeout=600)
    print("run_result:", demo_viewer.run_result)
    print("last_error:", demo_viewer.last_error)
    rs = demo_viewer._result_state
    print("result world:", rs is not None)
    if rs is not None:
        print("adopted:", rs.adopted)
        print("result specs:", [(s.id, s.name, s.port) for s in rs.specs])
    print("fields keys:", list(demo_viewer.fields))
    print("editor stack:", demo_viewer._viewer_state.stack.partial_real_shape)
    return


@app.cell
def demo_tabs(demo_viewer):

    # Two-world tab check (spec R17): switch the tab, then compare what each
    # world sends on the wire.
    demo_viewer._on_msg(None, {"event": "set_state", "patch": {"active_tab": "result"}}, [])
    res_geom_faces = len(demo_viewer.geometry.get("faces", []))
    res_state_tab = demo_viewer.state.get("active_tab")
    demo_viewer._on_msg(None, {"event": "set_state", "patch": {"active_tab": "sources"}}, [])
    ed_geom_faces = len(demo_viewer.geometry.get("faces", []))
    ed_state_tab = demo_viewer.state.get("active_tab")
    print("result tab: faces", res_geom_faces, "state.active_tab", res_state_tab)
    print("editor tab: faces", ed_geom_faces, "state.active_tab", ed_state_tab)
    print("editor stack still F1:", demo_viewer._viewer_state.stack.partial_real_shape)
    return


@app.cell(hide_code=True)
def demo_tab2(demo_viewer):

    g = None  # scratchpad-local
    demo_viewer._viewer_state.apply_patch({"active_tab": "result"})
    print("editor state active_tab:", demo_viewer._viewer_state.active_tab)
    print("active is result world:", demo_viewer._active_state() is demo_viewer._result_state)
    print("geometry keys:", sorted(demo_viewer.geometry.keys())[:8])
    print("geometry faces:", len(demo_viewer.geometry.get("faces", [])))
    print("state active_tab:", demo_viewer.state.get("active_tab"))
    return


@app.cell(hide_code=True)
def demo_tab3(demo_viewer):

    # geometry values after the result-tab swap
    print("editor stack size um:", [float(s) * 1e6 for s in demo_viewer._viewer_state.stack.partial_real_shape])
    print("result stack size um:", [float(s) * 1e6 for s in demo_viewer._result_state.stack.partial_real_shape])
    print("wire geometry size:", demo_viewer.geometry.get("size"), "origin:", demo_viewer.geometry.get("origin"))
    print("wire levels:", [lv.get("name") if isinstance(lv, dict) else lv for lv in demo_viewer.geometry.get("levels", [])])
    print("wire state active_tab:", demo_viewer.state.get("active_tab"))
    print("fields keys still:", list(demo_viewer.fields))
    return


@app.cell(hide_code=True)
def demo_tab4(
    demo_viewer,
    f1_stack,
    foreign_config,
    foreign_objects,
    foreign_stack,
    foreign_state2,
    view,
):

    # The demo ran on F1 (same stack in both worlds), so the geometry matches.
    # The real two-world check: load the FOREIGN result into demo_viewer, then
    # compare worlds again — the editor must keep F1 while Result shows 6x3x2.
    foreign_v = view(f1_stack())
    foreign_v.load_result(
        foreign_objects, foreign_state2[1], foreign_config, stack=foreign_stack,
    )
    print("demo editor stack um:", [float(s) * 1e6 for s in demo_viewer._viewer_state.stack.partial_real_shape])
    print("foreign_v editor stack um:", [float(s) * 1e6 for s in foreign_v._viewer_state.stack.partial_real_shape])
    print("foreign_v result stack um:", [float(s) * 1e6 for s in foreign_v._result_state.stack.partial_real_shape])
    foreign_v._viewer_state.apply_patch({"active_tab": "result"})
    print("after result tab, wire size:", foreign_v.geometry.get("size"))
    foreign_v._viewer_state.apply_patch({"active_tab": "sources"})
    print("after sources tab, wire size:", foreign_v.geometry.get("size"))
    return (foreign_v,)


@app.cell(hide_code=True)
def demo_tab5(foreign_v):

    # Did load_result's _sync_active(force=True) send the result world's geometry,
    # and did the later tab switch overwrite it with the editor's?
    print("foreign_v result tab now:", foreign_v._viewer_state.active_tab)
    print("foreign_v active world is result:", foreign_v._active_state() is foreign_v._result_state)
    print("wire geometry size:", foreign_v.geometry.get("size"))
    print("result world cached geometry size:", foreign_v._result_geometry.get("size"))
    print("editor world cached geometry size:", (foreign_v._editor_geometry or {}).get("size"))
    return


@app.cell(hide_code=True)
def demo_tab6(foreign_v):

    # demo_tab4 set the tab to "sources" at the end — set it back to "result"
    # and watch the wire swap.
    foreign_v._viewer_state.apply_patch({"active_tab": "result"})
    print("active is result:", foreign_v._active_state() is foreign_v._result_state)
    print("wire geometry size after ->result:", foreign_v.geometry.get("size"))
    foreign_v._viewer_state.apply_patch({"active_tab": "layout"})
    print("wire geometry size after ->layout:", foreign_v.geometry.get("size"))
    return


@app.cell(hide_code=True)
def demo_tab7(foreign_v):

    # apply_patch on the state directly does NOT go through _on_msg, so the
    # traits were never re-sent. Exercise the real path: _on_msg with set_state.
    foreign_v._on_msg(None, {"event": "set_state", "patch": {"active_tab": "result"}}, [])
    print("tab:", foreign_v._viewer_state.active_tab)
    print("wire size after _on_msg ->result:", foreign_v.geometry.get("size"))
    foreign_v._on_msg(None, {"event": "set_state", "patch": {"active_tab": "layout"}}, [])
    print("wire size after _on_msg ->layout:", foreign_v.geometry.get("size"))
    foreign_v._on_msg(None, {"event": "set_state", "patch": {"active_tab": "result"}}, [])
    print("wire size after _on_msg ->result again:", foreign_v.geometry.get("size"))
    return


if __name__ == "__main__":
    app.run()
