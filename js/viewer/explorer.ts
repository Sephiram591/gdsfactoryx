// The model explorer: Layers, Ports, Sources, Detectors, Simulation, plus a
// properties pane (spec R1, R2).
//
// Plain DOM, no framework (spec D6). Rows are built once and updated in place:
// the state round-trips through the kernel on every edit, and rebuilding the
// tree would tear down a slider or color picker mid-drag. Each edit is reported
// as a patch; the kernel owns the canonical state (spec D10).
import type { LevelMeta, MetaPayload, PortMeta, StatePayload } from "./wire";

export interface ExplorerCallbacks {
  onPatch(patch: Record<string, unknown>): void;
  onSelect(selection: { kind: "level" | "port"; name: string } | null): void;
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

const TABS = ["Layout / Ports", "Sources / Detectors", "Simulation"];

export class Explorer {
  readonly root: HTMLElement;
  private readonly properties: HTMLElement;
  private readonly panels: HTMLElement[];
  private readonly tabButtons: HTMLButtonElement[] = [];
  private readonly levelRows = new Map<string, LevelRow>();
  private readonly portRows = new Map<string, PortRow>();

  constructor(
    private readonly meta: MetaPayload,
    private readonly callbacks: ExplorerCallbacks,
  ) {
    this.root = document.createElement("div");
    this.root.className = "gfx-explorer";
    this.properties = document.createElement("div");
    this.properties.className = "gfx-properties";

    this.panels = [
      this.panel([
        this.section("Layers", this.meta.levels.map((level) => this.buildLevelRow(level))),
        this.section("Ports", this.meta.ports.map((port) => this.buildPortRow(port))),
      ]),
      this.panel([
        this.section("Sources", [emptyNote("No sources (R12 - M4).")]),
        this.section("Detectors", [emptyNote("No detectors (R12 - M4).")]),
      ]),
      this.panel([this.section("Simulation", [emptyNote("Settings (R13 - M4).")])]),
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

    for (const row of this.levelRows.values()) row.root.classList.remove("gfx-selected");
    for (const row of this.portRows.values()) row.root.classList.remove("gfx-selected");
    const selected = state.selection;
    if (selected) this.row(selected.kind, selected.name)?.classList.add("gfx-selected");
    this.renderProperties(state);
  }

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
  }

  private section(title: string, rows: HTMLElement[]): HTMLElement {
    const section = document.createElement("div");
    section.className = "gfx-section";
    const header = document.createElement("div");
    header.className = "gfx-section-header";
    header.textContent = title;
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
      this.properties.textContent = "Select a layer or port.";
      return;
    }
    const fields =
      selected.kind === "level"
        ? levelProperties(this.meta.levels.find((l) => l.name === selected.name))
        : portProperties(this.meta.ports.find((p) => p.name === selected.name));
    this.properties.replaceChildren(
      title(`${selected.kind === "level" ? "Layer" : "Port"}: ${selected.name}`),
      ...fields.map(([key, value]) => row(key, value)),
    );
  }
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
