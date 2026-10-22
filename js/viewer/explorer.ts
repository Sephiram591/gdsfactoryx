// The model explorer: Layers, Ports, Sources, Detectors, Simulation, plus a
// properties pane (spec R1, R2, R12, R13).
//
// Plain DOM, no framework (spec D6). Rows are built once and updated in place:
// the state round-trips through the kernel on every edit, and rebuilding the
// tree would tear down a slider or color picker mid-drag. Each edit is reported
// as a patch; the kernel owns the canonical state (spec D10).
//
// The Sources/Detectors and Simulation tabs author the simulation setup (spec
// R12, R13). They edit specs and settings, never live fdtdx objects (spec D11).
// While a mode-1 run is in flight every setup control is disabled and a note
// says why (spec D15): the kernel refuses setup edits regardless, so the
// disable is convenience, not the guarantee.
import type {
  FieldPayload,
  LevelMeta,
  MetaPayload,
  PortMeta,
  SimSettingsPayload,
  SpecPayload,
  StatePayload,
  TypeMeta,
} from "./wire";
import { TOTAL_COMPONENT, frameCount, shadeRange } from "./fieldramp";

//: The ``level`` value naming the simulation volume's own mid-height, offered
//: alongside the level names (spec R12). Mirrors ``Z_CENTER_LEVEL`` in
//: ``gdsfactory.fdtdx_stack``.
export const Z_CENTER_LEVEL = "z_center";

export interface ExplorerCallbacks {
  onPatch(patch: Record<string, unknown>): void;
  /** the active explorer tab, so the render picks the right style set (R17) */
  onActiveTab(tab: "layout" | "sources" | "simulation" | "result"): void;
  onSelect(
    selection: { kind: "level" | "port" | "source" | "detector"; name: string } | null,
  ): void;
  /** adds a default source on a port (spec R12, the "New source" control) */
  onAddSource(port: string): void;
  /** adds a default detector on a port (spec R12, the Detectors "+" action) */
  onAddDetector(port: string): void;
  /** edits one spec's type, port and/or parameters (spec R12) */
  onUpdateSpec(
    kind: "source" | "detector",
    id: string,
    change: { type?: string; port?: string; params?: Record<string, unknown> },
  ): void;
  /** removes a spec (spec R12) */
  onRemoveSpec(kind: "source" | "detector", id: string): void;
  /** applies a partial edit to the simulation settings (spec R13) */
  onSettings(fields: Record<string, unknown>): void;
  /** starts or stops a multi-frame detector's playback (client-side) */
  onPlay(id: string, playing: boolean, fps: number): void;
  /** renders every frame of a multi-frame detector and saves it as a video */
  onSaveVideo(id: string, fps: number): void;
  /** starts or stops playing every time-domain detector on one timeline */
  onPlayAll(playing: boolean, fps: number): void;
  /** records every time-domain detector together as one video */
  onRecordAll(fps: number): void;
}

//: What the Result tab's play/record-all controls describe: the shared
//: timeline's length and kind (see `timeline.ts`), or null with no
//: time-domain detector.
export interface GroupInfo {
  frames: number;
  detectors: number;
  kind: "shared" | "solve" | "index";
}

//: The playback rows of a multi-frame detector's Result card: play / save
//: video, and the frame rate on a row of its own.
interface PlaybackRow {
  rows: HTMLElement[];
  play: HTMLButtonElement;
  fps: HTMLInputElement;
  save: HTMLButtonElement;
  playing: boolean;
}

interface LevelRow {
  root: HTMLElement;
  visible: HTMLInputElement;
  swatch: HTMLInputElement;
  opacity: HTMLInputElement;
}

interface PortRow {
  root: HTMLElement;
  visible: HTMLInputElement;
}

/** One Result-tab Layers row: a stack level's show/color/opacity there (R17). */
interface ResultLayerRow {
  root: HTMLElement;
  visible: HTMLInputElement;
  swatch: HTMLInputElement;
  opacity: HTMLInputElement;
}

//: One editable control in a spec or settings form: the element to disable, a
//: writer that pushes the kernel's value back without stealing focus, and a
//: disable hook (a pair control is two inputs behind one element).
interface Control {
  element: HTMLElement;
  write(value: unknown): void;
  setDisabled(value: boolean): void;
}

interface SpecRow {
  root: HTMLElement;
  /** the type the row's fields were built for; a change rebuilds them */
  type: string;
  /** the port the row's fields were built for; "None" has its own set (R18) */
  port: string;
  /** the collapse arrow; its label tracks the row's collapsed state */
  twisty: HTMLButtonElement;
  collapsed: boolean;
  name: HTMLElement;
  visible: HTMLInputElement;
  /** the remove button, disabled while a run is in flight (spec D15) */
  remove: HTMLButtonElement;
  fields: Map<string, Control>;
  /** whether the spec came from a loaded result (read-only, spec R16) */
  adopted: boolean;
}

const TABS = ["Layout / Ports", "Sources / Detectors", "Simulation", "Result"];

//: The order the type selector lists each category's types in (spec R12).
const SOURCE_TYPES = ["ModePlaneSource", "GaussianPlaneSource", "UniformPlaneSource",
  "HardConstantAmplitudePlanceSource", "PointDipoleSource"];
const DETECTOR_TYPES = ["ModeOverlapDetector", "PoyntingFluxDetector", "PhasorDetector",
  "FieldDetector", "EnergyDetector"];

//: Tooltips for the Sources / Detectors tab: what each type does, and what each
//: field of a row sets. Native `title`s, like the rest of the viewer's controls.
const TYPE_HINTS: Record<string, string> = {
  ModePlaneSource: "injects the waveguide mode solved on the plane",
  GaussianPlaneSource: "a Gaussian beam launched from the plane",
  UniformPlaneSource: "a uniform-amplitude plane wave launched from the plane",
  HardConstantAmplitudePlanceSource: "a hard source: forces a constant amplitude on the plane",
  PointDipoleSource: "a point dipole: a single-cell source with no plane",
  ModeOverlapDetector: "the overlap of the field with a waveguide mode (the mode amplitude)",
  PoyntingFluxDetector: "the power flowing through the plane (Poynting flux)",
  PhasorDetector: "the frequency-domain field (phasor) on the plane",
  FieldDetector: "the raw E and H fields recorded on the plane",
  EnergyDetector: "the electromagnetic energy on the plane or in its volume",
};
const FIELD_HINTS: Record<string, string> = {
  port: "the device port the plane is centered on and normal to; xy / xz / yz are planes "
    + "through the simulation box center; None places it by axis, center and size instead",
  type: "the fdtdx object type",
  axis: "the plane's normal axis: 0 = x (a yz plane), 1 = y (an xz plane), 2 = z (an xy plane)",
  center: "the plane's center (x, y, z) in µm, in stack coordinates",
  size: "the plane's two in-plane sizes in µm, along its lower then higher stack axes",
  direction: "which way it faces relative to the component: in = into the component through "
    + "the port, out = out of it",
  port_offset: "shift along the port's injection direction in µm; negative moves the plane "
    + "out of the component, e.g. into a port extension",
  port_size_mult: "the plane's width and height as multiples of the port width and the level "
    + "height",
  level: "the level whose height and middle set the plane's z; (auto) uses the z span of the "
    + "levels that draw the port's layer, or the box's mid-height for a box plane; z_center "
    + "pins it to the box's mid-height",
  wavelengths: "wavelengths in µm, comma separated",
  name: "the object's name; blank uses <port>_<type>",
  mode_index: "the mode order: 0 is the fundamental mode",
  filter_pol: "keep only te or tm modes; blank keeps all",
  scaling_mode: "how the phasor is normalized: continuous or pulse",
  reduce_volume: "reduce the recorded volume to a single value instead of a field grid",
  components: "the field components recorded, e.g. Ex, Hy",
  aggregate: "how the energy is combined over the detector",
  polarization: "the dipole's polarization axis",
  source_type: "the fdtdx point-dipole source type",
  record_interval: "temporal downsampling: record every n-th time step (blank or 1 = every "
    + "step); a FieldDetector stores one plane per recorded step, so this divides its memory "
    + "and the number of video frames by n",
  dft_subsample: "temporal downsampling: feed the phasor's DFT every n-th time step, rescaled "
    + "to match every-step recording; auto picks n from the highest recorded frequency. "
    + "Saves time, not memory",
};
const VISIBLE_HINT = "show or hide this in the viewports (display only; the simulation is unchanged)";

//: A video-camera glyph for a detector card's "save video" button, drawn like
//: the menubar's icons (15 px, stroked in the text color).
const VIDEO_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<polygon points="23 7 16 12 23 17 23 7"/>'
  + '<rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>';
const SAVE_VIDEO_HINT = "save video: render every frame of the active viewport (3D or 2D) as "
  + "an MP4; the settings screen's \"Save to\" picks download or server";

/** The tooltip for a row's field: a source launches, a detector measures. */
function specFieldHint(key: string, kind: "source" | "detector"): string | undefined {
  if (key === "direction") {
    return kind === "source"
      ? "which way it launches: in = into the component through the port, out = out of it"
      : "which wave it measures: out = leaving the component through the port, in = entering it";
  }
  return FIELD_HINTS[key];
}

/** The tooltip for a port option: a box plane, a device port, or none (absolute). */
function portHint(name: string): string {
  if (name === "None") return "no port: place it by axis, center and size in stack coordinates";
  const planes: Record<string, string> = {
    xy: "the plane through the box center normal to z",
    xz: "the plane through the box center normal to y",
    yz: "the plane through the box center normal to x",
  };
  return planes[name] ?? `centered on the device port ${name} and normal to it`;
}

//: The transforms a detector's field can take (spec R15); the first three are
//: signed, the last two non-negative, which picks the ramp's shape (spec D16).
const FIELD_TRANSFORMS = ["real", "imag", "abs", "phase", "logabs"];

export class Explorer {
  readonly root: HTMLElement;
  private readonly properties: HTMLElement;
  private readonly panels: HTMLElement[];
  private readonly tabButtons: HTMLButtonElement[] = [];
  private readonly levelRows = new Map<string, LevelRow>();
  private readonly portRows = new Map<string, PortRow>();
  private readonly sourceRows = new Map<string, SpecRow>();
  private readonly detectorRows = new Map<string, SpecRow>();
  private readonly sourceList: HTMLElement;
  private readonly detectorList: HTMLElement;
  private readonly sourceEmpty: HTMLElement;
  private readonly detectorEmpty: HTMLElement;
  private readonly sourceAdd: HTMLButtonElement;
  private readonly detectorAdd: HTMLButtonElement;
  private readonly simFields = new Map<string, Control>();
  private readonly simNote: HTMLElement;
  private readonly layoutPanel: HTMLElement;
  private readonly resultPanel: HTMLElement;
  private readonly resultNote: HTMLElement;
  private readonly resultSections: HTMLElement;
  /** the Result tab's source rows (read-only, visibility-only), keyed by id */
  private readonly resultSourceList: HTMLElement;
  private resultSourceRows = new Map<string, { root: HTMLElement; visible: HTMLInputElement; name: HTMLElement }>();
  /** collapse state per detector section, kept across rebuilds (like the spec rows) */
  private resultCollapsed = new Set<string>();
  /** the last result's per-detector field payloads, keyed by spec id (R15, R17) */
  private fields: Record<string, FieldPayload> = {};
  /** the Result tab's per-detector display controls, keyed by spec id (R17) */
  private resultControls = new Map<string, Map<string, Control>>();
  /**
   * The Result tab's Layers-style rows (one per stack level), keyed by level
   * name — color and opacity of the *result world's* render while the Result
   * tab is active, independent of the editor's Layers tab (spec R17).
   */
  private resultLayers = new Map<string, ResultLayerRow>();
  private readonly resultLayersRoot: HTMLElement;
  private readonly resultLayersList: HTMLElement;
  /** each multi-frame detector's play / fps / save-video row, keyed by spec id */
  private resultPlayback = new Map<string, PlaybackRow>();
  /** play / record every time-domain detector together, beside the Detectors header */
  private readonly groupActions: HTMLElement;
  private readonly groupPlay: HTMLButtonElement;
  private readonly groupRecord: HTMLButtonElement;
  private readonly groupRow: HTMLElement;
  private readonly groupFps: HTMLInputElement;
  private readonly groupReadout: HTMLElement;
  private groupPlaying = false;
  private locked = false;

  constructor(
    private meta: MetaPayload,
    private readonly callbacks: ExplorerCallbacks,
  ) {
    this.root = document.createElement("div");
    this.root.className = "gfx-explorer";
    this.properties = document.createElement("div");
    this.properties.className = "gfx-properties";

    this.sourceList = document.createElement("div");
    this.sourceList.className = "gfx-spec-list";
    this.detectorList = document.createElement("div");
    this.detectorList.className = "gfx-spec-list";
    this.sourceEmpty = emptyNote("None yet. Use + to add one.");
    this.detectorEmpty = emptyNote("None yet. Use +, or double-click a port in 3D.");
    this.sourceAdd = this.buildAdd("source");
    this.detectorAdd = this.buildAdd("detector");

    const sim = this.buildSimForm();
    this.simNote = sim.note;

    this.resultSections = document.createElement("div");
    this.resultSourceList = document.createElement("div");
    this.resultSourceList.className = "gfx-spec-list";
    this.resultNote = emptyNote(
      "No result loaded. Run the simulation (or call load_result) and each "
      + "detector's plotting options appear here.",
    );
    this.resultLayersList = document.createElement("div");
    this.resultLayersRoot = this.section("Layers", [this.resultLayersList]);

    // play / record every time-domain detector together (beside the Detectors
    // header), with their shared frame rate and the timeline's time on a row
    // under it; all hidden until a result has a time-domain detector
    this.groupPlay = document.createElement("button");
    this.groupPlay.type = "button";
    this.groupPlay.className = "gfx-section-add";
    this.groupRecord = document.createElement("button");
    this.groupRecord.type = "button";
    this.groupRecord.className = "gfx-section-add";
    this.groupRecord.innerHTML = VIDEO_ICON;
    this.groupRecord.setAttribute("aria-label", "record all time-domain detectors");
    this.groupActions = document.createElement("span");
    this.groupActions.className = "gfx-section-actions";
    this.groupActions.append(this.groupPlay, this.groupRecord);
    this.groupFps = document.createElement("input");
    this.groupFps.type = "number";
    this.groupFps.className = "gfx-numeric";
    this.groupFps.min = "1";
    this.groupFps.max = "120";
    this.groupFps.step = "1";
    this.groupFps.value = "20";
    this.groupReadout = document.createElement("span");
    this.groupReadout.className = "gfx-label";
    this.groupRow = fieldRow("fps", this.groupFps,
      "frames per second for playing or recording every time-domain detector together");
    this.groupRow.appendChild(this.groupReadout);
    const groupRate = (): number =>
      Math.min(120, Math.max(1, Math.round(Number(this.groupFps.value) || 20)));
    this.groupPlay.addEventListener("click", () =>
      this.callbacks.onPlayAll(!this.groupPlaying, groupRate()));
    this.groupFps.addEventListener("change", () => {
      if (this.groupPlaying) this.callbacks.onPlayAll(true, groupRate());
    });
    this.groupRecord.addEventListener("click", () => this.callbacks.onRecordAll(groupRate()));
    this.setGroupPlaying(false);
    this.setGroup(null);

    this.layoutPanel = this.panel([]);
    this.resultPanel = this.panel([
      this.resultNote, this.resultLayersRoot,
      this.section("Sources", [this.resultSourceList]),
      // one card per detector with a field, under its own header like the
      // Layers and Sources sections above
      this.section("Detectors", [this.groupRow, this.resultSections], this.groupActions),
    ]);
    this.panels = [
      this.layoutPanel,
      this.panel([
        this.section("Sources", [this.sourceEmpty, this.sourceList], this.sourceAdd),
        this.section("Detectors", [this.detectorEmpty, this.detectorList], this.detectorAdd),
      ]),
      this.panel([this.section("Simulation", sim.rows)]),
      this.resultPanel,
    ];
    this.buildLayout();

    const tabbar = this.buildTabBar();
    const body = document.createElement("div");
    body.className = "gfx-tabbody";
    body.append(...this.panels);

    this.root.append(tabbar, body, this.properties);
    this.setTab(0);
  }

  /** (Re)builds the Layers and Ports sections from the current metadata. */
  private buildLayout(): void {
    this.layoutPanel.innerHTML = "";
    this.levelRows.clear();
    this.portRows.clear();
    this.layoutPanel.append(
      this.section("Layers", this.meta.levels.map((level) => this.buildLevelRow(level))),
      // only device ports here; the simulation-volume planes (spec R12) are a
      // source/detector placement choice, not a port of the layout
      this.section(
        "Ports",
        this.meta.ports.filter((port) => port.plane === null).map((port) => this.buildPortRow(port)),
      ),
    );
  }

  /**
   * Swaps the level/port metadata and rebuilds the layout rows (spec R13).
   *
   * A boundary-offset edit re-boxes the device, which can add a level that was
   * outside the old box or drop one the smaller box no longer reaches, so the
   * Layers and Ports lists must be rebuilt from the new metadata. The spec
   * rows and the settings controls are untouched (a boundary edit is what
   * triggers this, so no control the user is dragging is torn down).
   */
  setMeta(meta: MetaPayload): void {
    this.meta = meta;
    this.buildLayout();
  }

  /**
   * Records the last result's per-detector field payloads (spec R15, R17).
   *
   * The Result tab is rebuilt from these — one section per detector that has a
   * payload — and the frame slider's range is each payload's frame count (its
   * recorded frequencies or time steps). Called before :meth:`render`, which
   * writes the rows.
   */
  setFields(fields: Record<string, FieldPayload>): void {
    this.fields = fields ?? {};
    this.renderResult();
  }

  /** Moves a detector's frame slider and readout without an edit (playback). */
  showFrame(id: string, frame: number): void {
    this.resultControls.get(id)?.get("field_freq")?.write(frame);
  }

  /** Shows a detector's play button as playing (pause) or stopped (play). */
  setPlaying(id: string, playing: boolean): void {
    const row = this.resultPlayback.get(id);
    if (!row) return;
    row.playing = playing;
    row.play.textContent = playing ? "❚❚" : "▶";
    row.play.title = playing ? "pause" : "play the recorded frames";
  }

  /**
   * Shows or hides play / record all (they need a time-domain detector) and
   * says in their tooltips what the shared timeline is.
   */
  setGroup(info: GroupInfo | null): void {
    const available = info !== null;
    this.groupActions.hidden = !available;
    this.groupRow.hidden = !available;
    if (!info) return;
    const what = info.detectors === 1 ? "the time-domain detector" : `all ${info.detectors} time-domain detectors`;
    const timeline = info.kind === "solve"
      ? `their time grids differ, so every step of the solve (${info.frames} frames); each `
        + "shows its last recorded frame, and is transparent until it first records"
      : `${info.frames} frames`;
    this.groupPlay.dataset.hint = `play ${what} together: ${timeline}`;
    this.groupRecord.title = `record ${what} together as one MP4 of the active viewport: ${timeline}; `
      + "the settings screen's \"Save to\" picks download or server";
    this.setGroupPlaying(this.groupPlaying);
  }

  /** Shows play all as playing (pause) or stopped (play). */
  setGroupPlaying(playing: boolean): void {
    this.groupPlaying = playing;
    this.groupPlay.textContent = playing ? "❚❚" : "▶";
    this.groupPlay.title = playing ? "pause" : this.groupPlay.dataset.hint ?? "play all";
  }

  /** Greys play / record all while their video is being saved. */
  setGroupSaving(saving: boolean): void {
    this.groupPlay.disabled = saving;
    this.groupRecord.disabled = saving;
  }

  /** The shared timeline's position: its time in fs, or its frame index. */
  showGroupTime(text: string): void {
    this.groupReadout.textContent = text;
  }

  /** Greys a detector's playback row while its video is being saved. */
  setSaving(id: string, saving: boolean): void {
    const row = this.resultPlayback.get(id);
    if (!row) return;
    row.save.disabled = saving;
    row.play.disabled = saving;
    row.save.classList.toggle("gfx-active", saving);
    row.save.title = saving ? "saving the video…" : SAVE_VIDEO_HINT;
  }

  render(state: StatePayload): void {
    for (const level of this.meta.levels) {
      const row = this.levelRows.get(level.name);
      const style = state.styles[level.name];
      if (!row || !style) continue;
      if (row.visible.checked !== style.visible) row.visible.checked = style.visible;
      if (row.swatch.value !== style.color) row.swatch.value = style.color;
      const opacity = String(style.opacity);
      if (row.opacity.value !== opacity) row.opacity.value = opacity;
      row.opacity.title = `opacity ${style.opacity.toFixed(2)}`;
    }
    for (const port of this.meta.ports) {
      const row = this.portRows.get(port.name);
      if (!row) continue;
      const visible = state.port_visible[port.name] ?? true;
      if (row.visible.checked !== visible) row.visible.checked = visible;
    }

    this.renderSpecs(state.sources, "source", this.sourceRows, this.sourceList, this.sourceEmpty);
    this.renderSpecs(
      state.detectors, "detector", this.detectorRows, this.detectorList, this.detectorEmpty,
    );
    this.renderSettings(state.simulation);
    this.setLocked(state.running);
    this.renderResult(state);

    for (const row of this.levelRows.values()) row.root.classList.remove("gfx-selected");
    for (const row of this.portRows.values()) row.root.classList.remove("gfx-selected");
    for (const row of this.sourceRows.values()) row.root.classList.remove("gfx-selected");
    for (const row of this.detectorRows.values()) row.root.classList.remove("gfx-selected");
    const selected = state.selection;
    if (selected) this.row(selected.kind, selected.name)?.classList.add("gfx-selected");
    this.renderProperties(state);
  }

  // ------------------------------------------------------------- layout/ports

  private buildLevelRow(level: LevelMeta): HTMLElement {
    const root = document.createElement("div");
    root.className = "gfx-row";
    root.dataset.kind = "level";
    root.dataset.name = level.name;

    const visible = checkbox(true, (value) =>
      this.callbacks.onPatch({ styles: { [level.name]: { visible: value } } }),
    );

    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.className = "gfx-swatch";
    swatch.addEventListener("input", () =>
      this.callbacks.onPatch({ styles: { [level.name]: { color: swatch.value } } }),
    );

    const opacity = document.createElement("input");
    opacity.type = "range";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.01";
    opacity.className = "gfx-opacity";
    let pending: number | null = null;
    const flush = (): void => {
      if (pending !== null) {
        this.callbacks.onPatch({ styles: { [level.name]: { opacity: pending } } });
        pending = null;
      }
    };
    opacity.addEventListener("input", () => {
      pending = Number(opacity.value);
      if (window.requestAnimationFrame) window.requestAnimationFrame(flush);
      else flush();
    });

    root.append(visible, swatch, opacity, label(level.name));
    root.addEventListener("click", (event) => {
      if ((event.target as HTMLElement).tagName === "INPUT") return;
      this.callbacks.onSelect({ kind: "level", name: level.name });
    });
    this.levelRows.set(level.name, { root, visible, swatch, opacity });
    return root;
  }

  private buildPortRow(port: PortMeta): HTMLElement {
    const root = document.createElement("div");
    root.className = "gfx-row";
    root.dataset.kind = "port";
    root.dataset.name = port.name;
    const visible = checkbox(true, (value) =>
      this.callbacks.onPatch({ port_visible: { [port.name]: value } }),
    );
    root.append(visible, label(port.name));
    root.addEventListener("click", (event) => {
      if ((event.target as HTMLElement).tagName === "INPUT") return;
      this.callbacks.onSelect({ kind: "port", name: port.name });
    });
    this.portRows.set(port.name, { root, visible });
    return root;
  }

  // --------------------------------------------------------- sources/detectors

  /** The "+" action that adds a default source or detector (spec R12, Q18). */
  private buildAdd(kind: "source" | "detector"): HTMLButtonElement {
    const plus = document.createElement("button");
    plus.type = "button";
    plus.className = "gfx-section-add";
    plus.textContent = "+";
    plus.title = `add a ${kind} with the default type${kind === "detector" ? " on the last port" : " on the first port"}`;
    plus.addEventListener("click", () => {
      // one click adds a row: no picker. A source defaults to the first port and
      // a detector to the last, so the pair lands on opposite ends of the device
      // (spec R12, Q18); the row's port field still lets you move it after.
      if (this.locked) return;
      const ports = this.meta.ports;
      const port = ports.length > 1 && kind === "detector" ? ports[ports.length - 1] : ports[0];
      if (!port) return;
      if (kind === "source") this.callbacks.onAddSource(port.name);
      else this.callbacks.onAddDetector(port.name);
    });
    return plus;
  }

  private renderSpecs(
    specs: SpecPayload[],
    kind: "source" | "detector",
    rows: Map<string, SpecRow>,
    list: HTMLElement,
    empty: HTMLElement,
  ): void {
    const seen = new Set<string>();
    for (const spec of specs) {
      seen.add(spec.id);
      const type = this.meta.types[spec.type];
      if (!type) continue;
      let row = rows.get(spec.id);
      if (!row) {
        row = this.buildSpecRow(spec, type, kind);
        rows.set(spec.id, row);
        list.appendChild(row.root);
      } else if (row.type !== spec.type || row.port !== spec.port) {
        // the type and port select which fields exist, so a change rebuilds
        // them; keep the row's collapse state across the rebuild
        const rebuilt = this.buildSpecRow(spec, type, kind, row.collapsed);
        row.root.replaceWith(rebuilt.root);
        rows.set(spec.id, rebuilt);
        row = rebuilt;
      }
      this.fillSpecRow(row, spec);
    }
    for (const [id, row] of rows) {
      if (seen.has(id)) continue;
      row.root.remove();
      rows.delete(id);
    }
    empty.hidden = specs.length > 0;
  }

  private buildSpecRow(
    spec: SpecPayload,
    type: TypeMeta,
    kind: "source" | "detector",
    collapsed = false,
  ): SpecRow {
    const root = document.createElement("div");
    root.className = "gfx-spec";
    root.dataset.kind = kind;
    root.dataset.name = spec.id;

    const header = document.createElement("div");
    header.className = "gfx-spec-header";
    // the collapse arrow (spec R12): hides the row's fields, leaving the header
    // so a long setup stays navigable
    const twisty = document.createElement("button");
    twisty.type = "button";
    twisty.className = "gfx-twisty";
    // the pre-placement visibility toggle (spec R12); display-only, so it never
    // reaches create_*_at
    const visible = checkbox(true, (value) =>
      this.callbacks.onUpdateSpec(kind, spec.id, { params: { visible: value } }),
    );
    visible.title = VISIBLE_HINT;
    const name = document.createElement("span");
    name.className = "gfx-label";
    name.title = `this ${kind}'s name; click the row to select it in the viewports`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "gfx-button gfx-remove";
    remove.title = `remove this ${kind}`;
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      if (this.locked) return;
      this.callbacks.onRemoveSpec(kind, spec.id);
    });
    header.append(twisty, visible, name, remove);
    header.addEventListener("click", (event) => {
      if ((event.target as HTMLElement).closest("button") || (event.target as HTMLElement).tagName === "INPUT") return;
      this.callbacks.onSelect({ kind, name: spec.id });
    });

    const fields = document.createElement("div");
    fields.className = "gfx-spec-fields";
    const controls = new Map<string, Control>();
    const update = (change: { type?: string; port?: string; params?: Record<string, unknown> }): void =>
      this.callbacks.onUpdateSpec(kind, spec.id, change);
    const setParam = (key: string, value: unknown): void => update({ params: { [key]: value } });

    // port (spec R12). "None" (spec R18) places the object absolutely: the
    // port-frame rows below are replaced by axis/center/size.
    controls.set("port", this.selectControl(
      ["None", ...this.meta.ports.map((p) => p.name)], spec.port,
      (value) => update({ port: value }), portHint,
    ));
    // type (spec R12)
    const typeNames = kind === "source" ? SOURCE_TYPES : DETECTOR_TYPES;
    controls.set("type", this.selectControl(
      typeNames.filter((n) => this.meta.types[n]), spec.type,
      (value) => update({ type: value }), (name) => TYPE_HINTS[name],
    ));
    // absolute placement (spec R18): only when the spec's port is "None". A
    // point type (dipole) takes a center but no axis/size.
    if (spec.port === "None") {
      if (type.plane) {
        controls.set("axis", this.selectControl(
          ["0", "1", "2"], "2",
          (value) => setParam("axis", Number(value)),
          (name) => ({ "0": "x: a yz plane", "1": "y: an xz plane", "2": "z: an xy plane" })[name],
        ));
      }
      controls.set("center", this.tripleControl(
        (value) => setParam("center", value),
      ));
      if (type.plane) {
        controls.set("size", this.pairControl(
          (value) => setParam("size", value),
        ));
      }
    } else {
      // direction (spec R12)
      if (type.direction) {
        controls.set("direction", this.selectControl(
          ["in", "out"], "in",
          (value) => setParam("direction", value),
          (name) => ({
            in: "into the component through the port",
            out: "out of the component through the port",
          })[name],
        ));
      }
      // port_offset (spec R12): live, so the plane moves as it is edited
      controls.set("port_offset", this.numberControl(
        0, 0.01, (value) => setParam("port_offset", value), true,
      ));
      // port_size_mult (spec R12)
      if (type.plane) {
        controls.set("port_size_mult", this.pairControl(
          (value) => setParam("port_size_mult", value),
        ));
      }
      // level (spec R12). The empty choice is the port's own level union (a port)
      // or the volume's mid-height (a plane); z_center pins either to the
      // volume's mid-height explicitly, so a plane can be moved off a level.
      if (this.meta.levels.length > 0) {
        controls.set("level", this.selectControl(
          ["", ...this.meta.levels.map((l) => l.name), Z_CENTER_LEVEL], "",
          (value) => setParam("level", value || null),
          (name) => name === ""
            ? "the z span of the levels that draw the port's layer, or the box's mid-height "
              + "for a box plane"
            : name === Z_CENTER_LEVEL
              ? "the box's mid-height"
              : `center the plane on the z span of level ${name}`,
        ));
      }
    }
    // wavelength(s) (spec R12)
    if (type.wave !== "none") {
      controls.set("wavelengths", this.textControl(
        (value) => setParam("wavelengths", parseWavelengths(value)),
      ));
    }
    // name (spec R12)
    controls.set("name", this.textControl((value) => setParam("name", value)));
    // the type's extra parameters, edited generically (spec R12)
    for (const extra of type.extra) {
      controls.set(extra, this.textControl((value) => setParam(extra, parseValue(value))));
    }

    for (const [key, control] of controls) {
      const hint = specFieldHint(key, kind);
      // on the control too, so hovering the box explains it as well as the label;
      // a control that already titles its own boxes (the center) keeps them
      if (hint && !control.element.title) control.element.title = hint;
      fields.appendChild(fieldRow(key, control.element, hint));
    }

    const row: SpecRow = {
      root, type: spec.type, port: spec.port, twisty, collapsed, name, visible, remove,
      fields: controls, adopted: Boolean(spec.adopted),
    };
    // collapsing is view-only, so it lives in the row, not the kernel state: a
    // state round-trip must not expand a row the user collapsed
    twisty.addEventListener("click", (event) => {
      event.stopPropagation();
      row.collapsed = !row.collapsed;
      this.applyCollapsed(row);
    });
    root.append(header, fields);
    this.applyCollapsed(row);
    return row;
  }

  /** Shows or hides a spec row's fields to match its collapsed flag (spec R12). */
  private applyCollapsed(row: SpecRow): void {
    row.root.classList.toggle("gfx-collapsed", row.collapsed);
    row.twisty.textContent = row.collapsed ? "▶" : "▼";
    row.twisty.title = row.collapsed ? "expand" : "collapse";
    row.twisty.setAttribute("aria-expanded", String(!row.collapsed));
  }

  private fillSpecRow(row: SpecRow, spec: SpecPayload): void {
    // the kernel names the object `<port>_<type>` unless params.name overrides
    // it (spec R12); the row shows the same name the placed object will carry
    const name = spec.params.name;
    row.name.textContent = typeof name === "string" && name ? name : `${spec.port}_${spec.type}`;
    const visible = spec.params.visible ?? true;
    if (row.visible.checked !== Boolean(visible)) row.visible.checked = Boolean(visible);
    const values: Record<string, unknown> = {
      port: spec.port,
      type: spec.type,
      ...spec.params,
    };
    for (const [key, control] of row.fields) {
      if (key === "port_size_mult") continue; // written by its pair control
      if (key === "wavelengths") {
        control.write(formatWavelengths(spec.params.wavelengths));
        continue;
      }
      if (key === "level") {
        control.write(spec.params.level ?? "");
        continue;
      }
      if (key === "center") {
        control.write(spec.params.center);
        continue;
      }
      control.write(values[key]);
    }
    // an adopted spec is read-only except its display settings (spec R16): the
    // Result tab owns those, so the setup controls grey out and remove hides
    row.remove.hidden = row.adopted;
    for (const control of row.fields.values()) control.setDisabled(this.locked || row.adopted);
    if (!this.locked) row.remove.disabled = false;
  }

  // ------------------------------------------------------------------ settings

  private buildSimForm(): { rows: HTMLElement[]; note: HTMLElement } {
    const note = emptyNote(
      "A run streams its progress here; press the run button again to stop it "
      + "(spec R14, D17).",
    );
    const fields = document.createElement("div");
    fields.className = "gfx-spec-fields";
    // short labels; the descriptor and units ride along as tooltips (spec R13)
    const add = (key: string, labelText: string, control: Control, hint: string): void => {
      this.simFields.set(key, control);
      fields.appendChild(fieldRow(labelText, control.element, hint));
    };
    add("wavelength_um", "wavelength", this.numberControl(
      1.55, 0.01, (value) => this.callbacks.onSettings({ wavelength_um: value }), true,
    ), "default wavelength for a new source or detector, in µm");
    add("pulse_width_frac", "pulse width", this.numberControl(
      0, 0.01, (value) => this.callbacks.onSettings({ pulse_width_frac: value }), true,
    ), "Gaussian pulse spectral width, as a fraction of the center frequency; 0 = continuous wave");
    add("port_size_mult", "plane size", this.pairControl(
      (value) => this.callbacks.onSettings({ port_size_mult: value }),
    ), "a new plane is (a) port widths wide and (b) level heights tall");
    add("pml_thickness", "PML", this.numberControl(
      10, 1, (value) => this.callbacks.onSettings({ pml_thickness: Math.round(value) }), true,
    ), "perfectly-matched-layer thickness, in grid cells");
    add("resolution_um", "resolution", this.numberControl(
      0.02, 0.001, (value) => this.callbacks.onSettings({ resolution_um: value }), true,
    ), "grid resolution, in µm");
    // how the device paints the grid at material interfaces (spec R13): the
    // names are StackDevice's own `averaging` values, so the export reads alike
    add("averaging", "averaging", this.selectControl(
      ["subpixel", "subpixel_full", "inverse"], "subpixel",
      (value) => this.callbacks.onSettings({ averaging: value }),
    ), "permittivity averaging at material interfaces: subpixel = Kottke, the diagonal "
      + "of the averaged tensor; subpixel_full = the full tensor, accurate at oblique and "
      + "sloped walls (~5x the time per step, ~1.8x the memory); inverse = fdtdx's "
      + "isotropic inverse average");
    // the box is in fs; the kernel stores seconds (scale = s per fs), so the
    // typed text survives the round-trip (spec R13)
    add("time_fs", "time", this.numberControl(
      400, 10, (value) => this.callbacks.onSettings({ time_s: value }), true, 1e-15,
    ), "simulation time, in fs");
    // the box's faces, as an offset from the geometry's bounding box (spec R13):
    // three signed pairs (x, y, z), each `+`/`−` moving that face along its axis
    add("boundary_offset", "boundary offset", this.boundaryControl(
      (value) => this.callbacks.onSettings({ boundary_offset: value }),
    ), "move each box face off the geometry's bounding box, in µm (positive = towards +axis); the viewports update live");
    // the logabs ramp's floor, one number for the whole viewer (spec R15, Q22)
    add("log_range", "log range", this.numberControl(
      6, 1, (value) => this.callbacks.onSettings({ log_range: value }), true,
    ), "the logabs color ramp's floor, in decades below the plane's maximum");
    add("keep_partial", "keep partial run", this.boolControl(
      false, (value) => this.callbacks.onSettings({ keep_partial: value }),
    ), "when a stopped run keeps the fields its detectors recorded so far, "
      + "instead of discarding them (spec D17)");
    return { rows: [fields, note], note };
  }

  private renderSettings(settings: SimSettingsPayload): void {
    const values: Record<string, unknown> = {
      wavelength_um: settings.wavelength_um,
      pulse_width_frac: settings.pulse_width_frac,
      pml_thickness: settings.pml_thickness,
      resolution_um: settings.resolution_um,
      // the control's write() divides by its scale, so pass kernel units here
      time_fs: settings.time_s,
      log_range: settings.log_range,
      averaging: settings.averaging,
      keep_partial: settings.keep_partial,
    };
    for (const [key, control] of this.simFields) {
      if (key === "port_size_mult" || key === "boundary_offset") continue;
      control.write(values[key]);
    }
    this.simFields.get("port_size_mult")?.write(settings.port_size_mult);
    this.simFields.get("boundary_offset")?.write(settings.boundary_offset);
  }

  /** Enables or disables every setup control (spec D15). */
  private setLocked(locked: boolean): void {
    this.locked = locked;
    const rows = [...this.sourceRows.values(), ...this.detectorRows.values()];
    for (const row of rows) {
      for (const control of row.fields.values()) control.setDisabled(locked || row.adopted);
      row.visible.disabled = locked;
      row.remove.disabled = locked || row.adopted;
      row.remove.hidden = row.adopted;
    }
    for (const control of this.simFields.values()) control.setDisabled(locked);
    this.sourceAdd.disabled = locked;
    this.detectorAdd.disabled = locked;
    this.simNote.hidden = !locked;
    if (locked) this.simNote.textContent = "The setup is locked while a run is in flight.";
  }

  // ------------------------------------------------------------------ result tab

  /**
   * (Re)builds and fills the Result tab (spec R17).
   *
   * One section per detector with an entry in the last `fields` payload — the
   * widget's own runs and adopted foreign results alike — holding the detector's
   * field-display controls and, for a multi-frequency detector, the frequency
   * slider. A Layers section above them holds one row per stack level styling
   * the device while this tab is active. Rebuilt only when the payload or
   * metadata changes; the per-render fill writes values in place so a slider
   * drag is never torn down.
   */
  private renderResult(state?: StatePayload): void {
    const ids = Object.keys(this.fields);
    this.resultNote.hidden = ids.length > 0;

    // the result world's sources (spec R17): a visibility-only row each,
    // reconciled like the spec rows so a collapse is never torn down
    const specs = state?.sources ?? [];
    const seen = new Set<string>();
    for (const source of specs) {
      seen.add(source.id);
      let row = this.resultSourceRows.get(source.id);
      if (!row) {
        row = this.buildResultSourceRow(source);
        this.resultSourceList.appendChild(row.root);
        this.resultSourceRows.set(source.id, row);
      }
      const name = source.params.name;
      const text = typeof name === "string" && name ? name : `${source.port}_${source.type}`;
      if (row.name.textContent !== text) row.name.textContent = text;
      const visible = source.params.visible ?? true;
      if (row.visible.checked !== Boolean(visible)) row.visible.checked = Boolean(visible);
    }
    for (const [id, row] of this.resultSourceRows) {
      if (seen.has(id)) continue;
      row.root.remove();
      this.resultSourceRows.delete(id);
    }

    // rebuild the sections when the detector set changed; a rebuilt section
    // keeps its collapse state (view-only, like the spec rows)
    const current = new Set(ids);
    for (const [id, controls] of [...this.resultControls]) {
      if (current.has(id)) continue;
      controls.get("section")?.element.remove();
      this.resultControls.delete(id);
      this.resultCollapsed.delete(id);
      this.resultPlayback.delete(id);
    }
    for (const id of ids) {
      if (this.resultControls.has(id)) continue;
      const section = this.buildResultSection(id);
      this.resultSections.appendChild(section);
    }

    // the Layers rows follow the level metadata (a re-box can drop or add one)
    const levelNames = new Set(this.meta.levels.map((level) => level.name));
    for (const [name, row] of [...this.resultLayers]) {
      if (levelNames.has(name)) continue;
      row.root.remove();
      this.resultLayers.delete(name);
    }
    for (const level of this.meta.levels) {
      if (this.resultLayers.has(level.name)) continue;
      this.resultLayersList.appendChild(this.buildResultLayerRow(level.name));
    }
    if (!state) return;
    const settings = state.field_settings ?? {};
    for (const [id, controls] of this.resultControls) {
      const s = settings[id];
      if (!s) continue;
      // the component select's options are the components the result recorded
      // plus |E|, the total electric-field magnitude (spec R15)
      const entry = this.fields[id];
      const components = entry?.components ?? [];
      const componentSelect = controls.get("field_component")?.element as HTMLSelectElement;
      const hasE = components.some((name) => name.startsWith("E"));
      const options = ["", ...components, ...(hasE ? [TOTAL_COMPONENT] : [])];
      if (componentSelect.options.length !== options.length) {
        componentSelect.replaceChildren(
          ...options.map((value) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = value || "(none)";
            return option;
          }),
        );
      }
      // the frame slider and the playback row only exist for a multi-frame
      // detector (several frequencies, or a time-domain recording), and the
      // slider's range is its frame count (spec R17)
      const frames = entry ? frameCount(entry) : 1;
      const freq = controls.get("field_freq");
      if (freq) {
        const row = freq.element.parentElement;
        if (row) {
          row.hidden = frames <= 1;
          const rowLabel = row.firstElementChild;
          const text = entry?.frame_kind === "time" ? "time" : "frequency";
          if (rowLabel && rowLabel.textContent !== text) rowLabel.textContent = text;
        }
        if (frames > 1) freq.element.setAttribute("max", String(frames - 1));
      }
      const playback = this.resultPlayback.get(id);
      for (const row of playback?.rows ?? []) row.hidden = frames <= 1;
      for (const [key, control] of controls) {
        if (key === "section" || key === "field_range") continue;
        control.write((s as unknown as Record<string, unknown>)[key]);
      }
      // the scale boxes show the range in effect: the fixed one, or the whole
      // recording's when it is auto (spec R15)
      const scale = controls.get("field_range");
      if (scale && entry && s.field_component) {
        scale.write({ range: shadeRange(entry, s, state.simulation?.log_range ?? 6), fixed: Boolean(s.field_range) });
      }
      // the low color is unused by a non-negative transform (spec D16)
      const nonNegative = s.field_transform === "abs" || s.field_transform === "logabs";
      const low = controls.get("field_color_low");
      if (low) {
        low.element.classList.toggle("gfx-unused", nonNegative);
        low.element.title = nonNegative ? "unused by abs/logabs" : "";
      }
    }
    // the Layers rows show the style set currently on the wire (spec R17):
    // the result world's styles while the Result tab is active
    if (state) {
      // the header checkboxes follow the detector specs' `visible` param, not
      // the field settings — unchecking hides the whole detector (glyph too)
      const specs = new Set((state.detectors ?? []).map((d) => d.id));
      for (const [id, controls] of this.resultControls) {
        if (!specs.has(id)) continue;
        const spec = (state.detectors ?? []).find((d) => d.id === id);
        const visible = Boolean(spec?.params.visible ?? true);
        const header = controls.get("field_visible");
        if (header && header.element instanceof HTMLInputElement
            && header.element.checked !== visible) {
          header.element.checked = visible;
        }
      }
      for (const [name, row] of this.resultLayers) {
        const style = state.styles?.[name];
        if (!style) continue;
        if (row.visible.checked !== style.visible) row.visible.checked = style.visible;
        if (row.swatch.value !== style.color) row.swatch.value = style.color;
        const opacity = String(style.opacity);
        if (row.opacity.value !== opacity) row.opacity.value = opacity;
        row.opacity.title = `opacity ${style.opacity.toFixed(2)}`;
      }
    }
  }

  /**
   * One Layers-style row in the Result tab: a stack level as show / color /
   * opacity, the same controls the editor's Layers tab offers (spec R17). It
   * patches `styles`, which the kernel routes to the result world's own state
   * while this tab is active — so the editor's level colors are untouched.
   */
  private buildResultLayerRow(name: string): HTMLElement {
    const root = document.createElement("div");
    root.className = "gfx-row";
    root.dataset.kind = "level";
    root.dataset.name = name;

    const patch = (style: Record<string, unknown>): void =>
      this.callbacks.onPatch({ styles: { [name]: style } });

    const visible = checkbox(true, (value) => patch({ visible: value }));
    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.className = "gfx-swatch";
    swatch.addEventListener("input", () => patch({ color: swatch.value }));
    const opacity = document.createElement("input");
    opacity.type = "range";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.01";
    opacity.className = "gfx-opacity";
    let pending: number | null = null;
    const flush = (): void => {
      if (pending !== null) {
        patch({ opacity: pending });
        pending = null;
      }
    };
    opacity.addEventListener("input", () => {
      pending = Number(opacity.value);
      if (window.requestAnimationFrame) window.requestAnimationFrame(flush);
      else flush();
    });

    root.append(visible, swatch, opacity, label(name));
    this.resultLayers.set(name, { root, visible, swatch, opacity });
    return root;
  }

  /**
   * One result source's row (spec R17): an unlabeled visibility checkbox, then
   * the name — read-only, since the row describes a placed object.
   */
  private buildResultSourceRow(source: SpecPayload): { root: HTMLElement; visible: HTMLInputElement; name: HTMLElement } {
    const root = document.createElement("div");
    root.className = "gfx-row";
    root.dataset.kind = "source";
    root.dataset.name = source.id;
    const visible = checkbox(true, (value) =>
      this.callbacks.onUpdateSpec("source", source.id, { params: { visible: value } }),
    );
    visible.disabled = this.locked;
    const name = label("");
    root.append(visible, name);
    return { root, visible, name };
  }

  /** One detector's display controls in the Result tab (spec R17). */
  private buildResultSection(id: string): HTMLElement {
    const entry = this.fields[id];
    const section = document.createElement("div");
    section.className = "gfx-spec gfx-field-group";
    section.dataset.kind = "detector";
    section.dataset.name = id;

    const header = document.createElement("div");
    header.className = "gfx-spec-header";
    // the collapse arrow (spec R12): hides the row's fields, leaving the header
    const twisty = document.createElement("button");
    twisty.type = "button";
    twisty.className = "gfx-twisty";
    // the visibility toggle sits unlabeled in the header, before the name
    // (spec R17): it hides the whole detector — glyph and shading — via the
    // spec's `visible` param, the same one the Sources/Detectors rows write
    const headerVisible = document.createElement("input");
    headerVisible.type = "checkbox";
    headerVisible.title = "show this detector";
    const name = label(entry.name);
    header.append(twisty, headerVisible, name);

    let collapsed = this.resultCollapsed.has(id);
    const applyCollapsed = (): void => {
      section.classList.toggle("gfx-collapsed", collapsed);
      twisty.textContent = collapsed ? "▶" : "▼";
      twisty.title = collapsed ? "expand" : "collapse";
      twisty.setAttribute("aria-expanded", String(!collapsed));
    };
    twisty.addEventListener("click", (event) => {
      event.stopPropagation();
      const next = !this.resultCollapsed.has(id);
      if (next) this.resultCollapsed.add(id);
      else this.resultCollapsed.delete(id);
      collapsed = next;
      applyCollapsed();
    });
    applyCollapsed();
    // the header checkbox and the visibility row write the same param
    headerVisible.addEventListener("change", () =>
      this.setParam(id, "visible", headerVisible.checked));

    const fields = document.createElement("div");
    fields.className = "gfx-spec-fields";
    const controls = new Map<string, Control>();

    // a fixed colour scale is in the units of one component and transform, so
    // changing either returns the scale to auto (the whole recording's range)
    const component = this.selectControl([], "", (value) =>
      this.setParams(id, { field_component: value || null, field_range: null }));
    controls.set("field_component", component);
    fields.appendChild(fieldRow("component", component.element,
      "which recorded field component to shade the plane with"));
    const transform = this.selectControl(
      [...FIELD_TRANSFORMS], "abs",
      (value) => this.setParams(id, { field_transform: value, field_range: null }),
    );
    controls.set("field_transform", transform);
    fields.appendChild(fieldRow("transform", transform.element,
      "real/imag/phase are signed (two-color ramp); abs/logabs are non-negative"));
    const low = this.colorControl((value) => this.setParam(id, "field_color_low", value));
    controls.set("field_color_low", low);
    fields.appendChild(fieldRow("low color", low.element,
      "the ramp's low-end color (signed transforms only)"));
    const high = this.colorControl((value) => this.setParam(id, "field_color_high", value));
    controls.set("field_color_high", high);
    fields.appendChild(fieldRow("high color", high.element, "the ramp's high-end color"));
    const opacity = this.numberControl(1, 0.05, (value) => this.setParam(id, "field_opacity", value), true);
    controls.set("field_opacity", opacity);
    fields.appendChild(fieldRow("opacity", opacity.element, "the overlay's overall opacity, 0-1"));
    // "show" lives unlabeled in the header (spec R17); it writes the spec's
    // `visible` param, synced from the detector's spec payload (not the field
    // settings), so unchecking hides the glyph as well as the shading
    controls.set("field_visible", {
      element: headerVisible,
      write(value) {
        if (headerVisible.checked !== Boolean(value)) headerVisible.checked = Boolean(value);
      },
      setDisabled(value) {
        headerVisible.disabled = value;
      },
    });
    // the outline toggle (spec R17): the border drawn around the plane while it
    // is shaded, so the extent stays visible under a busy field
    const outline = checkbox(true, (value) => this.setParam(id, "field_outline", value));
    controls.set("field_outline", {
      element: outline,
      write(value) {
        if (outline.checked !== Boolean(value)) outline.checked = Boolean(value);
      },
      setDisabled(value) {
        outline.disabled = value;
      },
    });
    fields.appendChild(fieldRow("outline", outline, "draw the plane's outline while shaded"));
    fields.appendChild(this.buildScaleRow(id, controls));
    // the frame slider (spec R17): client-side slicing of the multi-frame
    // buffers — one frame per recorded frequency, or per recorded time step —
    // so a move re-shades with no kernel round-trip
    const freqLabel = document.createElement("span");
    freqLabel.className = "gfx-label";
    const freq = document.createElement("input");
    freq.type = "range";
    freq.min = "0";
    freq.step = "1";
    freq.className = "gfx-opacity";
    freq.addEventListener("input", () => {
      // scrubbing takes over from playback
      const playback = this.resultPlayback.get(id);
      if (playback?.playing) this.callbacks.onPlay(id, false, Number(playback.fps.value));
      this.setParam(id, "field_freq", Number(freq.value));
    });
    controls.set("field_freq", {
      element: freq,
      write: (value) => {
        const raw = Math.floor(Number(value ?? 0));
        const frame = Math.max(0, raw);
        if (document.activeElement !== freq && freq.value !== String(frame)) freq.value = String(frame);
        // a negative frame: playing all together, this detector has not
        // recorded yet at the shared timeline's time (it is transparent)
        freqLabel.textContent = raw < 0 ? "not recorded yet" : frameReadout(this.fields[id], frame);
        freqLabel.title = raw < 0 ? "" : frameNote(this.fields[id], frame);
      },
      setDisabled(value) {
        freq.disabled = value;
      },
    });
    const freqRow = fieldRow("frequency", freq,
      "which recorded frame to shade: a frequency of a phasor, a time step of a FieldDetector");
    freqRow.appendChild(freqLabel);
    fields.appendChild(freqRow);
    fields.append(...this.buildPlaybackRows(id));

    section.append(header, fields);
    controls.set("section", { element: section, write() {}, setDisabled() {} });
    this.resultControls.set(id, controls);
    return section;
  }

  private setParam(id: string, key: string, value: unknown): void {
    this.setParams(id, { [key]: value });
  }

  private setParams(id: string, params: Record<string, unknown>): void {
    this.callbacks.onUpdateSpec("detector", id, { params });
  }

  /**
   * The colour-scale row (spec R15): min and max boxes and an "auto" button.
   *
   * The boxes show the range every frame is shaded on — by default the whole
   * recording's min and max, so playback and a saved video keep one scale.
   * Editing either box fixes the scale (`field_range`); auto returns it to the
   * recording's range. The row is registered as the `field_range` control, and
   * written with the range in effect, not the raw setting.
   */
  private buildScaleRow(id: string, controls: Map<string, Control>): HTMLElement {
    const row = document.createElement("div");
    row.className = "gfx-field";
    const name = document.createElement("span");
    name.className = "gfx-field-name";
    name.textContent = "scale";
    name.title = "the colour scale every frame is shaded on, in the transform's units; "
      + "auto spans the whole recording's min and max";
    const box = (title: string): HTMLInputElement => {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "gfx-numeric";
      input.title = title;
      return input;
    };
    const min = box("the value at the ramp's low end");
    const max = box("the value at the ramp's high end");
    const auto = document.createElement("button");
    auto.type = "button";
    auto.className = "gfx-button";
    auto.textContent = "auto";
    auto.title = "span the whole recording's min and max (every frame)";
    const fix = (): void => {
      const low = Number(min.value);
      const high = Number(max.value);
      if (min.value !== "" && max.value !== "" && Number.isFinite(low) && Number.isFinite(high) && high > low) {
        this.setParam(id, "field_range", [low, high]);
      }
    };
    min.addEventListener("change", fix);
    max.addEventListener("change", fix);
    auto.addEventListener("click", () => this.setParam(id, "field_range", null));
    row.append(name, min, max, auto);
    controls.set("field_range", {
      element: row,
      write(value) {
        const { range, fixed } = value as { range: [number, number]; fixed: boolean };
        const shown = range.map((v) => String(Number(v.toPrecision(4))));
        if (document.activeElement !== min && min.value !== shown[0]) min.value = shown[0];
        if (document.activeElement !== max && max.value !== shown[1]) max.value = shown[1];
        auto.disabled = !fixed;
      },
      setDisabled(value) {
        min.disabled = value;
        max.disabled = value;
        auto.disabled = value;
      },
    });
    return row;
  }

  /**
   * The playback rows of a multi-frame detector: play/pause and "save video"
   * — which renders every frame of the active viewport and saves it as an MP4
   * (the "Save to" setting picks download or server) — then the frame rate,
   * on a row of its own.
   */
  private buildPlaybackRows(id: string): HTMLElement[] {
    const root = document.createElement("div");
    root.className = "gfx-field";
    const name = document.createElement("span");
    name.className = "gfx-field-name";
    name.textContent = "play";
    name.title = "play the recorded frames in the viewports, or save them as a video";
    const play = document.createElement("button");
    play.type = "button";
    play.className = "gfx-button";
    const fps = document.createElement("input");
    fps.type = "number";
    fps.className = "gfx-numeric";
    fps.min = "1";
    fps.max = "120";
    fps.step = "1";
    fps.value = "20";
    fps.title = "frames per second, for playback and the saved video";
    const rate = (): number => Math.min(120, Math.max(1, Math.round(Number(fps.value) || 20)));
    const save = document.createElement("button");
    save.type = "button";
    save.className = "gfx-button gfx-icon";
    save.innerHTML = VIDEO_ICON;
    save.title = SAVE_VIDEO_HINT;
    save.setAttribute("aria-label", "save video");
    root.append(name, play, save);
    const fpsRow = fieldRow("fps", fps, "frames per second, for playback and the saved video");
    const row: PlaybackRow = { rows: [root, fpsRow], play, fps, save, playing: false };
    play.addEventListener("click", () => this.callbacks.onPlay(id, !row.playing, rate()));
    fps.addEventListener("change", () => {
      if (row.playing) this.callbacks.onPlay(id, true, rate());
    });
    save.addEventListener("click", () => this.callbacks.onSaveVideo(id, rate()));
    this.resultPlayback.set(id, row);
    this.setPlaying(id, false);
    return row.rows;
  }

  // ------------------------------------------------------------------- controls

  private selectControl(
    options: string[],
    initial: string,
    onChange: (value: string) => void,
    optionHint?: (name: string) => string | undefined,
  ): Control {
    const select = document.createElement("select");
    select.className = "gfx-select";
    for (const name of options) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name || "(auto)";
      const hint = optionHint?.(name);
      if (hint) option.title = hint;
      select.appendChild(option);
    }
    select.value = initial;
    select.addEventListener("change", () => onChange(select.value));
    return {
      element: select,
      write(value) {
        const text = value === null || value === undefined ? "" : String(value);
        if (document.activeElement !== select && select.value !== text) select.value = text;
      },
      setDisabled(value) {
        select.disabled = value;
      },
    };
  }

  //: a numeric field in **display units**, converted to the kernel's units only
  //: on emit (`scale` = kernel units per display unit). The box therefore holds
  //: exactly what was typed; the round-trip through a small unit (the time field
  //: shows fs but stores s) would otherwise echo back as `999.9999999999999`.
  private numberControl(
    initial: number,
    step: number,
    onChange: (value: number) => void,
    live = false,
    scale = 1,
  ): Control {
    const input = document.createElement("input");
    input.type = "number";
    input.step = String(step);
    input.className = "gfx-numeric";
    input.value = String(initial);
    const emit = (text: string): void => {
      const value = Number(text);
      if (Number.isFinite(value)) onChange(value * scale);
    };
    if (live) coalesce(input, emit);
    else input.addEventListener("change", () => emit(input.value));
    return {
      element: input,
      write(value) {
        const shown =
          value === null || value === undefined ? "" : String(displayNumber(Number(value) / scale));
        if (document.activeElement !== input && input.value !== shown) input.value = shown;
      },
      setDisabled(value) {
        input.disabled = value;
      },
    };
  }

  /** A checkbox (spec R13): a boolean simulation setting. */
  private boolControl(initial: boolean, onChange: (value: boolean) => void): Control {
    const input = checkbox(initial, onChange);
    return {
      element: input,
      write(value) {
        const next = Boolean(value);
        if (input.checked !== next) input.checked = next;
      },
      setDisabled(value) {
        input.disabled = value;
      },
    };
  }

  /** A color input (spec R15): one endpoint of a detector's field ramp. */
  private colorControl(onChange: (value: string) => void): Control {
    const input = document.createElement("input");
    input.type = "color";
    input.className = "gfx-swatch";
    input.addEventListener("input", () => onChange(input.value));
    return {
      element: input,
      write(value) {
        const text = value === null || value === undefined ? "" : String(value);
        if (document.activeElement !== input && input.value !== text) input.value = text;
      },
      setDisabled(value) {
        input.disabled = value;
      },
    };
  }

  private textControl(onChange: (value: string) => void): Control {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "gfx-text";
    input.addEventListener("change", () => onChange(input.value));
    return {
      element: input,
      write(value) {
        const text = value === null || value === undefined ? "" : String(value);
        if (document.activeElement !== input && input.value !== text) input.value = text;
      },
      setDisabled(value) {
        input.disabled = value;
      },
    };
  }

  /** A two-number control (port_size_mult, spec R12, R13). */
  private pairControl(onChange: (value: [number, number]) => void): Control {
    const host = document.createElement("span");
    host.className = "gfx-pair";
    const a = document.createElement("input");
    a.type = "number";
    a.step = "0.1";
    a.className = "gfx-numeric";
    const b = document.createElement("input");
    b.type = "number";
    b.step = "0.1";
    b.className = "gfx-numeric";
    const emit = (): void => {
      const first = Number(a.value);
      const second = Number(b.value);
      if (Number.isFinite(first) && Number.isFinite(second) && first > 0 && second > 0) {
        onChange([first, second]);
      }
    };
    a.addEventListener("change", emit);
    b.addEventListener("change", emit);
    host.append(a, b);
    return {
      element: host,
      write(value) {
        const pair = Array.isArray(value) ? (value as number[]) : [0, 0];
        if (document.activeElement !== a && a.value !== String(pair[0])) a.value = String(pair[0]);
        if (document.activeElement !== b && b.value !== String(pair[1])) b.value = String(pair[1]);
      },
      setDisabled(value) {
        a.disabled = value;
        b.disabled = value;
      },
    };
  }

  /**
   * A three-number control — an absolute placement's `[x, y, z]` center in µm
   * (spec R18). Unlike the pair control the values may be any finite number
   * (the center can sit on a negative coordinate).
   */
  private tripleControl(onChange: (value: [number, number, number]) => void): Control {
    const host = document.createElement("span");
    host.className = "gfx-triple";
    const axes = ["x", "y", "z"] as const;
    const inputs: HTMLInputElement[] = [];
    const emit = (): void => {
      const values = inputs.map((input) => Number(input.value));
      if (values.every((value) => Number.isFinite(value))) {
        onChange(values as [number, number, number]);
      }
    };
    for (const axis of axes) {
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.1";
      input.className = "gfx-numeric";
      input.title = `${axis} center (µm)`;
      input.addEventListener("change", emit);
      inputs.push(input);
      host.appendChild(input);
    }
    return {
      element: host,
      write(value) {
        const triple = Array.isArray(value) ? (value as number[]) : [0, 0, 0];
        inputs.forEach((input, index) => {
          const shown = String(displayNumber(Number(triple[index] ?? 0)));
          if (document.activeElement !== input && input.value !== shown) input.value = shown;
        });
      },
      setDisabled(value) {
        for (const input of inputs) input.disabled = value;
      },
    };
  }

  /**
   * The box's six face offsets as three signed pairs (spec R13).
   *
   * Each axis is a `+`/`−` input pair, so the six values read as x+, x-, y+, y-,
   * z+, z- — the order `boundary_offset` takes. Each value moves its face along
   * the axis, positive towards +x, +y or +z (so `(1, -1, …)` grows the box on
   * each x side). Like the other settings controls it is live (`input` events,
   * coalesced), so dragging a value re-boxes the device and updates both
   * viewports without waiting for a blur.
   */
  private boundaryControl(
    onChange: (value: [number, number, number, number, number, number]) => void,
  ): Control {
    const host = document.createElement("span");
    host.className = "gfx-triple";
    const axes = ["x", "y", "z"] as const;
    const inputs: HTMLInputElement[] = [];
    const emit = (): void => {
      const values = inputs.map((input) => Number(input.value));
      if (values.every((value) => Number.isFinite(value))) {
        onChange(values as [number, number, number, number, number, number]);
      }
    };
    for (const axis of axes) {
      const group = document.createElement("span");
      group.className = "gfx-triple-axis";
      group.title = `the ${axis} box faces`;
      const name = document.createElement("span");
      name.className = "gfx-triple-name";
      name.textContent = axis;
      group.appendChild(name);
      for (const sign of ["+", "−"] as const) {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "0.1";
        input.className = "gfx-numeric";
        input.title = `${axis}${sign} face offset (µm, positive = towards +${axis})`;
        coalesce(input, emit);
        inputs.push(input);
        group.appendChild(input);
      }
      host.appendChild(group);
    }
    return {
      element: host,
      write(value) {
        const offset = Array.isArray(value) ? (value as number[]) : [0, 0, 0, 0, 0, 0];
        inputs.forEach((input, index) => {
          const shown = String(displayNumber(Number(offset[index] ?? 0)));
          if (document.activeElement !== input && input.value !== shown) input.value = shown;
        });
      },
      setDisabled(value) {
        for (const input of inputs) input.disabled = value;
      },
    };
  }

  // ------------------------------------------------------------------- plumbing
  private panel(children: HTMLElement[]): HTMLElement {
    const panel = document.createElement("div");
    panel.className = "gfx-tabpanel";
    panel.append(...children);
    return panel;
  }

  private buildTabBar(): HTMLElement {
    const bar = document.createElement("div");
    bar.className = "gfx-tabbar";
    TABS.forEach((name, index) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "gfx-tab";
      tab.textContent = name;
      tab.addEventListener("click", () => this.setTab(index));
      this.tabButtons.push(tab);
      bar.appendChild(tab);
    });
    return bar;
  }

  private setTab(index: number): void {
    this.panels.forEach((panel, i) => (panel.hidden = i !== index));
    this.tabButtons.forEach((tab, i) => tab.classList.toggle("gfx-active", i === index));
    const tabs = ["layout", "sources", "simulation", "result"] as const;
    this.callbacks.onActiveTab(tabs[index]);
  }

  private section(title: string, rows: HTMLElement[], add?: HTMLElement): HTMLElement {
    const section = document.createElement("div");
    section.className = "gfx-section";
    const header = document.createElement("div");
    header.className = "gfx-section-header";
    const name = document.createElement("span");
    name.textContent = title;
    header.appendChild(name);
    // the "+" adds a source or detector straight from its own section (spec R12);
    // the Result tab's Detectors header holds play / record all instead
    if (add) header.appendChild(add);
    section.appendChild(header);
    for (const row of rows) section.appendChild(row);
    return section;
  }

  private row(kind: string, name: string): HTMLElement | undefined {
    return this.root.querySelector<HTMLElement>(`[data-kind="${kind}"][data-name="${name}"]`) ?? undefined;
  }

  private renderProperties(state: StatePayload): void {
    const selected = state.selection;
    if (!selected) {
      this.properties.textContent = "Select a layer, port, source or detector.";
      return;
    }
    const fields =
      selected.kind === "level"
        ? levelProperties(this.meta.levels.find((l) => l.name === selected.name))
        : selected.kind === "port"
          ? portProperties(this.meta.ports.find((p) => p.name === selected.name))
          : specProperties(
              [...state.sources, ...state.detectors].find((s) => s.id === selected.name),
              this.meta.types,
            );
    const kindName = { level: "Layer", port: "Port", source: "Source", detector: "Detector" }[selected.kind];
    this.properties.replaceChildren(
      title(`${kindName}: ${selected.name}`),
      ...fields.map(([key, value]) => row(key, value)),
    );
  }
}

// ---------------------------------------------------------------------- helpers

/**
 * Formats a number for a display-unit field, absorbing the float noise a unit
 * round-trip leaves behind (`1e-12 / 1e-15` is `999.9999999999999`, not `1000`).
 * A value within a relative `1e-9` of a nearby integer is shown as that integer;
 * everything else keeps its natural `String()` form.
 */
function displayNumber(value: number): number | string {
  if (!Number.isFinite(value)) return "";
  const rounded = Math.round(value);
  const tolerance = Math.max(Math.abs(value), 1) * 1e-9;
  return Math.abs(value - rounded) <= tolerance ? rounded : value;
}

/** Reports `input` events at most once per frame, with the latest value. */
function coalesce(input: HTMLInputElement, handler: (value: string) => void): void {
  let frame = 0;
  let pending: string | null = null;
  const flush = (): void => {
    frame = 0;
    if (pending === null) return;
    const value = pending;
    pending = null;
    handler(value);
  };
  input.addEventListener("input", () => {
    pending = input.value;
    if (frame === 0) frame = requestAnimationFrame(flush);
  });
}

/** A comma-separated wavelength list, or undefined when unparseable (spec R12). */
function parseWavelengths(text: string): number[] | undefined {
  const parts = text.split(",").map((part) => part.trim()).filter((part) => part.length > 0);
  const values = parts.map(Number);
  if (values.length === 0 || values.some((value) => !Number.isFinite(value) || value <= 0)) return undefined;
  return values;
}

function formatWavelengths(value: unknown): string {
  return Array.isArray(value) ? (value as number[]).join(", ") : "";
}

/** A loose scalar parse for a type's extra parameters (spec R12). */
function parseValue(text: string): unknown {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (/^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) return Number(trimmed);
  if (trimmed.includes(",")) {
    return trimmed.split(",").map((part) => parseValue(part));
  }
  return trimmed;
}

function levelProperties(level: LevelMeta | undefined): [string, string][] {
  if (!level) return [];
  return [
    ["material", level.material],
    ["mesh_order", String(level.mesh_order)],
    ["fill", level.fill ? "yes" : "no"],
    ["zmin", `${level.zmin.toFixed(4)} µm`],
    ["thickness", `${level.thickness.toFixed(4)} µm`],
    ["sidewall_angle", `${level.sidewall_angle}°`],
    ["bias", `${level.bias.toFixed(4)} µm`],
  ];
}

function portProperties(port: PortMeta | undefined): [string, string][] {
  if (!port) return [];
  return [
    ["center", `(${port.x.toFixed(3)}, ${port.y.toFixed(3)}, ${port.z.toFixed(3)}) µm`],
    ["orientation", port.orientation === null ? "unset" : `${port.orientation}°`],
    ["width", `${port.width.toFixed(3)} µm`],
    ["layer", String(port.layer)],
  ];
}

function specProperties(spec: SpecPayload | undefined, types: Record<string, TypeMeta>): [string, string][] {
  if (!spec) return [];
  const params = spec.params;
  const rows: [string, string][] = [
    ["type", spec.type],
    ["port", spec.port],
    ["category", types[spec.type]?.category ?? "-"],
  ];
  for (const [key, value] of Object.entries(params)) {
    rows.push([key, Array.isArray(value) ? (value as unknown[]).join(", ") : String(value)]);
  }
  return rows;
}

/** A frame's readout: its time in fs, its frequency in THz, or its index. */
function frameReadout(field: FieldPayload | undefined, frame: number): string {
  if (field?.frame_kind === "time") {
    const seconds = field.times?.[frame];
    return seconds !== undefined ? `${(seconds * 1e15).toFixed(1)} fs` : `frame ${frame}`;
  }
  const hz = field?.freqs?.[frame];
  return hz !== undefined ? `${(hz / 1e12).toFixed(4)} THz` : `frame ${frame}`;
}

/** The frame readout's tooltip: position, and any thinning the kernel applied. */
function frameNote(field: FieldPayload | undefined, frame: number): string {
  if (!field) return "";
  const count = frameCount(field);
  const stride = field.frame_stride ?? 1;
  const thinned = stride > 1
    ? `; 1 in ${stride} of the ${field.frames_recorded ?? count * stride} recorded frames is shown`
    : "";
  return `frame ${frame + 1} of ${count}${thinned}`;
}

function fieldRow(name: string, element: HTMLElement, hint?: string): HTMLElement {
  const row = document.createElement("label");
  row.className = "gfx-field";
  const span = document.createElement("span");
  span.className = "gfx-field-name";
  span.textContent = name;
  // the descriptor and units live in the tooltip, so the label column stays
  // narrow enough to read (spec R13)
  if (hint) span.title = hint;
  row.append(span, element);
  return row;
}

function checkbox(checked: boolean, onChange: (value: boolean) => void): HTMLInputElement {
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  return input;
}

function label(text: string): HTMLElement {
  const span = document.createElement("span");
  span.className = "gfx-label";
  span.textContent = text;
  return span;
}

function title(text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = "gfx-properties-title";
  div.textContent = text;
  return div;
}

function row(key: string, value: string): HTMLElement {
  const div = document.createElement("div");
  div.className = "gfx-property";
  div.textContent = `${key}: ${value}`;
  return div;
}

function emptyNote(text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = "gfx-note";
  div.textContent = text;
  return div;
}
