"""Tests for multi-frame detector playback and video export.

A ``FieldDetector``'s time-domain recording is shaded, played and saved as a
video like a phasor's frequency frames. The kernel half is tested here:

- the recording reaches the browser as frames with their times, real-only,
  thinned to the wire budget, and a foreign one is validated like a phasor;
- the temporal downsampling a detector row exposes — ``record_interval``
  (fdtdx's ``switch``) and ``dft_subsample`` — in placement and in the launch
  code;
- the fixed colour-scale setting (``field_range``);
- :func:`wrap_video` and the ``save_video`` message: the browser's encoded
  stream becomes an MP4, saved under ``save_dir`` or sent back to download.

Playback, the whole-recording range and the IVF bytes are browser code, covered
by ``js/viewer/fieldramp.test.mjs`` and ``js/viewer/video.test.mjs``.

The F1 fixture comes from ``test_viewer_geometry`` (M0).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
pytest.importorskip("anywidget")
pytest.importorskip("manifold3d")

from test_viewer_geometry import f1_stack  # noqa: E402

from gdsfactory.viewer import fields, sim, widget  # noqa: E402


def _coarse(viewer: Any) -> None:
    """A cheap grid, so a real run fits in memory (100 nm, 4-cell PML)."""
    viewer._on_msg(
        None,
        {"event": "set_settings", "settings": {"resolution_um": 0.1, "pml_thickness": 4}},
        [],
    )


def _field_viewer(**params: Any) -> Any:
    """A viewer with a source on o1 and a FieldDetector recording Ey on xz."""
    viewer = widget.view(f1_stack())
    _coarse(viewer)
    viewer._on_msg(None, {"event": "add_source", "port": "o1"}, [])
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "xz",
         "spec": {"type": "FieldDetector", "port": "xz",
                  "params": {"components": ["Ey"], **params}}},
        [],
    )
    return viewer


def _add_phasor(viewer: Any, **params: Any) -> None:
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "o2",
         "spec": {"type": "PhasorDetector", "port": "o2", "params": params}},
        [],
    )


def _run(viewer: Any) -> Any:
    """Runs the viewer's simulation synchronously and returns the result."""
    viewer.run()
    viewer._run_thread.join()
    assert viewer.last_error is None, viewer.last_error
    return viewer.last_result


def _fake_result(frames: np.ndarray, on: np.ndarray, step: float = 1e-17) -> Any:
    """A time-domain detector ``d`` on a 4x3 plane (y normal), as a result."""
    obj = SimpleNamespace(
        grid_slice=(slice(0, 4), slice(0, 1), slice(0, 3)),
        components=("Ex", "Ey"),
        _is_on_at_time_step_arr=on,
    )
    return SimpleNamespace(
        objects={"d": obj},
        arrays=SimpleNamespace(detector_states={"d": {"fields": frames}}),
        config=SimpleNamespace(time_step_duration=step, time_steps_total=len(on)),
    )


# ---------------------------------------------------- time-domain field payload


def test_a_field_detector_records_frames_the_browser_can_play() -> None:
    """Every recorded step is a frame, real-only, stamped with its time."""
    viewer = _field_viewer(record_interval=50)
    result = _run(viewer)
    spec = next(s for s in viewer.specs if s.category == "detector")

    entry = fields.field_payload(result, spec.name, spec.id)
    assert entry is not None
    assert entry["frame_kind"] == "time"
    assert "imag" not in entry and "freqs" not in entry
    total = int(result.config.time_steps_total)
    frames = len(entry["times"])
    # fdtdx records step t when t % interval == 0
    assert entry["frames_recorded"] == frames == -(-total // 50)
    assert entry["frame_stride"] == 1
    assert np.allclose(np.diff(entry["times"]), 50 * float(result.config.time_step_duration))
    # the solver steps and the solve's clock, for playing detectors together
    assert entry["steps"] == list(range(0, total, 50))
    assert entry["steps_total"] == total
    assert entry["time_step"] == pytest.approx(float(result.config.time_step_duration))
    rows, cols = entry["grid"]
    real = np.frombuffer(entry["real"], dtype=np.float32)
    assert real.size == frames * rows * cols
    # a source drove the run, so some frame is not all zero
    assert np.abs(real).max() > 0


def test_frames_over_the_wire_budget_are_thinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above the budget every k-th frame is sent, with its own time."""
    frames = np.random.default_rng(0).standard_normal((10, 2, 4, 1, 3)).astype(np.float32)
    on = np.zeros(20, dtype=bool)
    on[::2] = True  # recorded every 2nd step
    monkeypatch.setattr(fields, "_MAX_FIELD_BYTES", 400)  # a frame is 2*4*3*4 = 96 B
    entry = fields.field_payload(_fake_result(frames, on), "d", "id")

    assert entry["frame_stride"] == 3
    assert entry["frames_recorded"] == 10
    assert entry["times"] == pytest.approx([0.0, 6e-17, 12e-17, 18e-17])
    assert entry["steps"] == [0, 6, 12, 18]
    assert entry["steps_total"] == 20
    # (frames, components, v, u): the y-normal plane's u is x (4), v is z (3)
    assert entry["grid"] == [3, 4]
    sent = np.frombuffer(entry["real"], dtype=np.float32).reshape(4, 2, 3, 4)
    assert np.array_equal(sent, np.swapaxes(frames[::3, :, :, 0, :], 2, 3))


def test_an_untimed_recording_is_ordered_but_not_timed() -> None:
    """Without the on-mask or the solve's clock, frames get indices and no steps."""
    frames = np.zeros((3, 2, 4, 1, 3), dtype=np.float32)
    result = _fake_result(frames, np.ones(3, dtype=bool))
    del result.objects["d"]._is_on_at_time_step_arr
    entry = fields.field_payload(result, "d", "id")
    assert entry["times"] == [0.0, 1.0, 2.0]
    assert "steps" not in entry and "steps_total" not in entry


def test_a_phasor_still_sends_frequency_frames() -> None:
    """A complex phasor keeps its imaginary part and its frame kind."""
    phasor = np.ones((1, 3, 2, 4, 1, 3), dtype=np.complex64) * (1 + 2j)
    result = _fake_result(phasor, np.ones(1, dtype=bool))
    result.arrays.detector_states["d"] = {"phasor": phasor}
    entry = fields.field_payload(result, "d", "id")
    assert entry["frame_kind"] == "freq"
    assert entry["freqs"] == [0, 1, 2]  # no wave characters: frame indices
    assert np.frombuffer(entry["imag"], dtype=np.float32).max() == 2


def test_a_foreign_time_domain_plane_must_match_its_detector() -> None:
    """A loaded recording whose grid disagrees with its detector is refused."""
    obj = SimpleNamespace(name="d", grid_slice=(slice(0, 4), slice(0, 1), slice(0, 3)))
    objects = SimpleNamespace(detectors=[obj], volume=SimpleNamespace(grid_shape=(10, 10, 10)))
    config = SimpleNamespace(uniform_spacing=lambda: 1e-7)
    good = np.zeros((5, 1, 4, 1, 3), dtype=np.float32)
    fields.validate_foreign_result(objects, SimpleNamespace(detector_states={"d": {"fields": good}}), config)
    bad = np.zeros((5, 1, 4, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="field grid shape"):
        fields.validate_foreign_result(objects, SimpleNamespace(detector_states={"d": {"fields": bad}}), config)


# --------------------------------------------------- temporal downsampling


@pytest.mark.parametrize(
    ("key", "value"),
    [("record_interval", 0), ("record_interval", 2.5), ("record_interval", "4"),
     ("dft_subsample", -1), ("dft_subsample", "fast")],
)
def test_bad_temporal_downsampling_is_refused(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=key):
        sim.validate_sampling({key: value})


def test_good_temporal_downsampling_is_accepted() -> None:
    sim.validate_sampling({"record_interval": 4, "dft_subsample": "auto"})
    sim.validate_sampling({"record_interval": 4.0, "dft_subsample": 3})
    sim.validate_sampling({"record_interval": None})


def test_each_detector_type_offers_the_downsampling_it_has() -> None:
    """Time-domain types record every n-th step; phasors subsample their DFT."""
    for name in ("FieldDetector", "EnergyDetector", "PoyntingFluxDetector"):
        assert "record_interval" in sim.TYPES[name].extra
        assert "dft_subsample" not in sim.TYPES[name].extra
    for name in ("PhasorDetector", "ModeOverlapDetector"):
        assert "dft_subsample" in sim.TYPES[name].extra
        assert "record_interval" not in sim.TYPES[name].extra


def test_record_interval_becomes_fdtdx_switch() -> None:
    kind = sim.TYPES["FieldDetector"]
    assert sim._extra_kwargs(kind, {"record_interval": 1}) == {}
    kwargs = sim._extra_kwargs(kind, {"record_interval": 4.0, "components": ["Ey"]})
    assert kwargs["switch"].interval == 4
    assert kwargs["components"] == ["Ey"]
    assert sim._extra_args(kind, {"record_interval": 4}) == ["switch=fdtdx.OnOffSwitch(interval=4)"]
    phasor = sim.TYPES["PhasorDetector"]
    assert sim._extra_kwargs(phasor, {"dft_subsample": 3.0}) == {"dft_subsample": 3}
    assert sim._extra_kwargs(phasor, {"dft_subsample": "auto"}) == {"dft_subsample": "auto"}


def test_a_bad_downsampling_edit_is_refused_in_the_state() -> None:
    viewer = _field_viewer(record_interval=4)
    spec = next(s for s in viewer.specs if s.category == "detector")
    viewer._on_msg(
        None,
        {"event": "update_spec", "kind": "detector",
         "spec": {"id": spec.id, "params": {"record_interval": 0}}},
        [],
    )
    assert spec.params["record_interval"] == 4


def test_temporal_downsampling_reaches_the_placed_detectors() -> None:
    """The placed FieldDetector stores every 4th step; the phasor subsamples."""
    import jax

    viewer = _field_viewer(record_interval=4)
    _add_phasor(viewer, dft_subsample="auto")
    objects, config, constraints = viewer.objects, viewer.config, viewer.constraints
    field = next(o for o in objects if type(o).__name__ == "FieldDetector")
    phasor = next(o for o in objects if type(o).__name__ == "PhasorDetector")
    assert field.switch.interval == 4
    assert phasor.dft_subsample == "auto"

    placed, _, _, config, _ = fdtdx.place_objects(objects, config, constraints, jax.random.PRNGKey(0))
    total = int(config.time_steps_total)
    assert placed[field.name]._num_time_steps_on == -(-total // 4)


def test_launch_code_carries_the_temporal_downsampling() -> None:
    """The emitted tail builds the same downsampled detectors, port or absolute."""
    viewer = _field_viewer(record_interval=4)
    _add_phasor(viewer, dft_subsample="auto")
    viewer._on_msg(
        None,
        {"event": "add_detector", "port": "o1",
         "spec": {"type": "FieldDetector", "port": "None",
                  "params": {"axis": 2, "center": [5.0, 5.0, 0.3], "size": [4.0, 1.0],
                             "components": ["Ex"], "record_interval": 3, "name": "flat"}}},
        [],
    )
    code = viewer.launch_code
    assert "switch=fdtdx.OnOffSwitch(interval=4)" in code
    assert "switch=fdtdx.OnOffSwitch(interval=3)," in code
    assert "dft_subsample='auto'" in code

    namespace: dict[str, Any] = {"stack": viewer.stack}
    exec(compile(code.split("# --- run:")[0], "<launch_code>", "exec"), namespace)
    built = {o.name: o for o in namespace["objects"] if "Detector" in type(o).__name__}
    expected = {o.name: o for o in viewer.objects if "Detector" in type(o).__name__}
    assert set(built) == set(expected)
    for name, obj in expected.items():
        assert built[name].switch.interval == obj.switch.interval
        assert getattr(built[name], "dft_subsample", None) == getattr(obj, "dft_subsample", None)


# ------------------------------------------------------- fixed colour scale


def test_the_colour_scale_is_auto_until_fixed() -> None:
    viewer = _field_viewer()
    spec = next(s for s in viewer.specs if s.category == "detector")

    def edit(value: Any) -> None:
        viewer._on_msg(
            None,
            {"event": "update_spec", "kind": "detector",
             "spec": {"id": spec.id, "params": {"field_range": value}}},
            [],
        )

    def current() -> Any:
        return viewer.state["field_settings"][spec.id]["field_range"]

    assert current() is None
    edit([-1, 2])
    assert current() == [-1.0, 2.0]
    for bad in ([2, 1], [0, float("inf")], ["a", 1], [1]):
        edit(bad)
        assert current() == [-1.0, 2.0], f"{bad!r} must be refused"
    edit(None)
    assert current() is None


# --------------------------------------------------------------- video export

def _ffmpeg() -> str:
    """The ffmpeg binary ``imageio-ffmpeg`` ships (the one ``wrap_video`` uses)."""
    return pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()


def _encode(tmp_path: Path, codec: list[str], fmt: str, name: str, frames: int = 12) -> bytes:
    """A small test-pattern stream from the bundled ffmpeg, as a browser would send."""
    out = tmp_path / name
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
         "-frames:v", str(frames), *codec, "-f", fmt, str(out)],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


def _probe(path: Path) -> tuple[int, str]:
    """The video's frame count and its ffmpeg stream description.

    The count is the number of packets ``framemd5`` lists (one line per frame,
    after its ``#`` header): ffmpeg's ``frame=`` progress line is not printed
    when its stderr is not a terminal.
    """
    listed = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    frames = sum(1 for line in listed.stdout.splitlines() if line and not line.startswith("#"))
    info = subprocess.run([_ffmpeg(), "-i", str(path)], capture_output=True, text=True, check=False)
    stream = next((line for line in info.stderr.splitlines() if "Video:" in line), "")
    return frames, stream


def test_an_h264_stream_is_wrapped_into_an_mp4(tmp_path: Path) -> None:
    stream = _encode(tmp_path, ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "h264", "in.h264")
    mp4 = widget.wrap_video(stream, "h264", 10)
    out = tmp_path / "out.mp4"
    out.write_bytes(mp4)
    assert mp4[4:8] == b"ftyp"
    frames, description = _probe(out)
    assert frames == 12
    assert "h264" in description


def test_a_vp9_stream_is_transcoded_to_h264(tmp_path: Path) -> None:
    try:
        stream = _encode(tmp_path, ["-c:v", "libvpx-vp9", "-b:v", "200k"], "ivf", "in.ivf")
    except subprocess.CalledProcessError:
        pytest.skip("this ffmpeg build has no VP9 encoder to make the test stream")
    mp4 = widget.wrap_video(stream, "vp9", 10)
    out = tmp_path / "out.mp4"
    out.write_bytes(mp4)
    frames, description = _probe(out)
    assert frames == 12
    assert "h264" in description


def test_wrap_video_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="codec"):
        widget.wrap_video(b"", "theora", 10)
    with pytest.raises(ValueError, match="frame rate"):
        widget.wrap_video(b"", "h264", 0)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        widget.wrap_video(b"not a video", "h264", 10)


def test_save_video_writes_under_the_save_folder(tmp_path: Path) -> None:
    """Save to: server — the MP4 lands in save_dir, never clobbering."""
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    stream = _encode(tmp_path, ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "h264", "in.h264")
    message = {"event": "save_video", "filename": "d-Ey-real", "codec": "h264", "fps": 10,
               "target": "server"}
    viewer._on_msg(None, message, [memoryview(stream)])
    first = Path(viewer.save_result["path"])
    assert first == (tmp_path / "d-Ey-real.mp4").resolve()
    assert first.read_bytes()[4:8] == b"ftyp"
    viewer._on_msg(None, message, [memoryview(stream)])
    assert Path(viewer.save_result["path"]).name == "d-Ey-real-1.mp4"


def test_save_video_for_the_browser_sends_the_mp4_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save to: browser — the MP4 goes back as a video_ready message."""
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    sent: list[tuple[Any, Any]] = []
    monkeypatch.setattr(viewer, "send", lambda content, buffers=None: sent.append((content, buffers)))
    stream = _encode(tmp_path, ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "h264", "in.h264")
    viewer._on_msg(
        None,
        {"event": "save_video", "filename": "d", "codec": "h264", "fps": 10, "target": "browser"},
        [memoryview(stream)],
    )
    assert len(sent) == 1
    content, buffers = sent[0]
    assert content == {"event": "video_ready", "filename": "d.mp4"}
    assert bytes(buffers[0])[4:8] == b"ftyp"
    assert not list(tmp_path.glob("*.mp4")), "a browser save writes nothing server-side"


def test_a_failed_video_save_reports_an_error(tmp_path: Path) -> None:
    viewer = widget.view(f1_stack(), save_dir=tmp_path)
    message = {"event": "save_video", "filename": "d", "codec": "h264", "fps": 10, "target": "server"}
    viewer._on_msg(None, message, [])
    assert "no video data" in viewer.save_result["error"]
    viewer._on_msg(None, message, [memoryview(b"not a video")])
    assert "ffmpeg" in viewer.save_result["error"]
