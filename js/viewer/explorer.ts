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
  LevelMeta,
  MetaPayload,
  PortMeta,
  SimSettingsPayload,
  SpecPayload,
  StatePayload,
  TypeMeta,
} from "./wire";

export interface ExplorerCallbacks {
  onPatch(patch: Record<string, unknown>): void;
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
  /** the collapse arrow; its label tracks the row's collapsed state */
  twisty: HTMLButtonElement;
  collapsed: boolean;
  name: HTMLElement;
  visible: HTMLInputElement;
  fields: Map<string, Control>;
}

//: A section's "add" control: a port picker and an add button, hidden until the
//: section header's "+" reveals it (spec R12).
interface AddRow {
  root: HTMLElement;
  select: HTMLSelectElement;
  button: HTMLButtonElement;
  open: boolean;
}

const TABS = ["Layout / Ports", "Sources / Detectors", "Simulation"];

//: The order the type selector lists each category's types in (spec R12).
const SOURCE_TYPES = ["ModePlaneSource", "GaussianPlaneSource", "UniformPlaneSource",
  "HardConstantAmplitudePlanceSource", "PointDipoleSource"];
const DETECTOR_TYPES = ["ModeOverlapDetector", "PoyntingFluxDetector", "PhasorDetector",
  "FieldDetector", "EnergyDetector"];

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
  private readonly sourceAdd: AddRow;
  private readonly detectorAdd: AddRow;
  private readonly simFields = new Map<string, Control>();
  private readonly simNote: HTMLElement;
  private locked = false;

  constructor(
    private readonly meta: MetaPayload,
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
    this.sourceEmpty = emptyNote("No sources yet.");
    this.detectorEmpty = emptyNote("None yet. Use +, or double-click a port in 3D.");
    this.sourceAdd = this.buildAddRow("source");
    this.detectorAdd = this.buildAddRow("detector");

    const sim = this.buildSimForm();
    this.simNote = sim.note;

    this.panels = [
      this.panel([
        this.section("Layers", this.meta.levels.map((level) => this.buildLevelRow(level))),
        this.section("Ports", this.meta.ports.map((port) => this.buildPortRow(port))),
      ]),
      this.panel([
        this.section("Sources", [this.sourceAdd.root, this.sourceEmpty, this.sourceList],
          this.sourceAdd),
        this.section("Detectors", [this.detectorAdd.root, this.detectorEmpty, this.detectorList],
          this.detectorAdd),
      ]),
      this.panel([this.section("Simulation", sim.rows)]),
    ];

    const tabbar = this.buildTabBar();
    const body = document.createElement("div");
    body.className = "gfx-tabbody";
    body.append(...this.panels);

    this.root.append(tabbar, body, this.properties);
    this.setTab(0);
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
    this.renderSpecs(state.detectors, "detector", this.detectorRows, this.detectorList, this.detectorEmpty);
    this.renderSettings(state.simulation);
    this.setLocked(state.running);

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

  /** A port picker plus an add button, revealed by the section's "+" (spec R12). */
  private buildAddRow(kind: "source" | "detector"): AddRow {
    const root = document.createElement("div");
    root.className = "gfx-row gfx-add";
    root.hidden = true;
    const select = document.createElement("select");
    select.className = "gfx-select gfx-add-port";
    select.title = `port to add a ${kind} on`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "gfx-button";
    button.textContent = "Add";
    button.addEventListener("click", () => {
      if (this.locked || !select.value) return;
      if (kind === "source") this.callbacks.onAddSource(select.value);
      else this.callbacks.onAddDetector(select.value);
    });
    root.append(select, button);
    return { root, select, button, open: false };
  }

  /** Shows or hides a section's add row (the section header's "+", spec R12). */
  private toggleAdd(add: AddRow): void {
    add.open = !add.open;
    add.root.hidden = !add.open;
    if (add.open) {
      this.fillPortChoices(add.select);
      add.select.focus();
    }
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
      } else if (row.type !== spec.type) {
        // the type selects which fields exist, so a type change rebuilds them;
        // keep the row's collapse state across the rebuild
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
    const name = document.createElement("span");
    name.className = "gfx-label";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "gfx-button gfx-remove";
    remove.title = "remove";
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

    // port (spec R12)
    controls.set("port", this.selectControl(
      this.meta.ports.map((p) => p.name), spec.port,
      (value) => update({ port: value }),
    ));
    // type (spec R12)
    const typeNames = kind === "source" ? SOURCE_TYPES : DETECTOR_TYPES;
    controls.set("type", this.selectControl(
      typeNames.filter((n) => this.meta.types[n]), spec.type,
      (value) => update({ type: value }),
    ));
    // direction (spec R12)
    if (type.direction) {
      controls.set("direction", this.selectControl(
        ["in", "out"], "in",
        (value) => setParam("direction", value),
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
    // wavelength(s) (spec R12)
    if (type.wave !== "none") {
      controls.set("wavelengths", this.textControl(
        (value) => setParam("wavelengths", parseWavelengths(value)),
      ));
    }
    // level (spec R12)
    if (this.meta.levels.length > 0) {
      controls.set("level", this.selectControl(
        ["", ...this.meta.levels.map((l) => l.name)], "",
        (value) => setParam("level", value || null),
      ));
    }
    // name (spec R12)
    controls.set("name", this.textControl((value) => setParam("name", value)));
    // the type's extra parameters, edited generically (spec R12)
    for (const extra of type.extra) {
      controls.set(extra, this.textControl((value) => setParam(extra, parseValue(value))));
    }

    for (const [key, control] of controls) {
      fields.appendChild(fieldRow(key, control.element));
    }

    const row: SpecRow = { root, type: spec.type, twisty, collapsed, name, visible, fields: controls };
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
      control.write(values[key]);
    }
  }

  // ------------------------------------------------------------------ settings

  private buildSimForm(): { rows: HTMLElement[]; note: HTMLElement } {
    const note = emptyNote("No progress or cancel; the run ends on its own (§2).");
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
    add("time_fs", "time", this.numberControl(
      400, 10, (value) => this.callbacks.onSettings({ time_s: value * 1e-15 }), true,
    ), "simulation time, in fs");
    return { rows: [fields, note], note };
  }

  private renderSettings(settings: SimSettingsPayload): void {
    const values: Record<string, unknown> = {
      wavelength_um: settings.wavelength_um,
      pulse_width_frac: settings.pulse_width_frac,
      pml_thickness: settings.pml_thickness,
      resolution_um: settings.resolution_um,
      time_fs: settings.time_s / 1e-15,
    };
    for (const [key, control] of this.simFields) {
      if (key === "port_size_mult") continue;
      control.write(values[key]);
    }
    this.simFields.get("port_size_mult")?.write(settings.port_size_mult);
  }

  /** Enables or disables every setup control (spec D15). */
  private setLocked(locked: boolean): void {
    this.locked = locked;
    const rows = [...this.sourceRows.values(), ...this.detectorRows.values()];
    for (const row of rows) {
      for (const control of row.fields.values()) control.setDisabled(locked);
      row.visible.disabled = locked;
      const remove = row.root.querySelector("button");
      if (remove) (remove as HTMLButtonElement).disabled = locked;
    }
    for (const control of this.simFields.values()) control.setDisabled(locked);
    for (const add of [this.sourceAdd, this.detectorAdd]) {
      add.select.disabled = locked;
      add.button.disabled = locked;
    }
    this.simNote.hidden = !locked;
    if (locked) this.simNote.textContent = "The setup is locked while a run is in flight.";
  }

  /** Fills an add row's port list from the device's ports (spec R12). */
  private fillPortChoices(select: HTMLSelectElement): void {
    if (select.options.length > 0) return;
    for (const port of this.meta.ports) {
      const option = document.createElement("option");
      option.value = port.name;
      option.textContent = port.name;
      select.appendChild(option);
    }
  }

  // ------------------------------------------------------------------- controls

  private selectControl(
    options: string[],
    initial: string,
    onChange: (value: string) => void,
  ): Control {
    const select = document.createElement("select");
    select.className = "gfx-select";
    for (const name of options) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name || "(auto)";
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

  private numberControl(
    initial: number,
    step: number,
    onChange: (value: number) => void,
    live = false,
  ): Control {
    const input = document.createElement("input");
    input.type = "number";
    input.step = String(step);
    input.className = "gfx-numeric";
    input.value = String(initial);
    const emit = (text: string): void => {
      const value = Number(text);
      if (Number.isFinite(value)) onChange(value);
    };
    if (live) coalesce(input, emit);
    else input.addEventListener("change", () => emit(input.value));
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
    // the port lists only need filling once, and the tab may have been hidden
    if (index === 1) {
      this.fillPortChoices(this.sourceAdd.select);
      this.fillPortChoices(this.detectorAdd.select);
    }
  }

  private section(title: string, rows: HTMLElement[], add?: AddRow): HTMLElement {
    const section = document.createElement("div");
    section.className = "gfx-section";
    const header = document.createElement("div");
    header.className = "gfx-section-header";
    const name = document.createElement("span");
    name.textContent = title;
    header.appendChild(name);
    // the "+" reveals the section's add row, so a new source or detector can be
    // added from its own section rather than only the "New source" control (R12)
    if (add) {
      const plus = document.createElement("button");
      plus.type = "button";
      plus.className = "gfx-section-add";
      plus.textContent = "+";
      plus.title = `add a ${title.toLowerCase().replace(/s$/, "")}`;
      plus.addEventListener("click", () => {
        if (this.locked) return;
        this.toggleAdd(add);
      });
      header.appendChild(plus);
    }
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
