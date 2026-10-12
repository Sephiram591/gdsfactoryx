// The app-shell anywidget frontend (spec D2, D6, D9).
//
// One widget root holds the menubar, the explorer and the viewport region
// (spec D9). The 2D viewport is M3, so it is a disabled placeholder here.
import "./style.css";

import { Explorer } from "./explorer";
import { PortGlyphs } from "./glyphs";
import { LevelMeshes } from "./mesh";
import { Scene3D } from "./scene";
import type {
  GeometryPayload,
  MetaPayload,
  StatePayload,
} from "./wire";

interface AnyModel {
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  save_changes(): void;
  send(content: Record<string, unknown>, callbacks?: unknown): void;
  on(event: string, handler: () => void): void;
}

function render({ model, el }: { model: AnyModel; el: HTMLElement }): () => void {
  const geometry = model.get("geometry") as GeometryPayload;
  const meta = model.get("meta") as MetaPayload;

  el.classList.add("gfx-shell");
  el.innerHTML = "";

  // -- menubar (spec D9): 2D is a placeholder until M3 ----------------------
  const menubar = document.createElement("div");
  menubar.className = "gfx-menubar";
  const title = document.createElement("span");
  title.className = "gfx-title";
  title.textContent = "gdsfactory viewer";
  const to3d = button("3D", true);
  const to2d = button("2D", false);
  to2d.disabled = true;
  to2d.title = "Cross-section (R4) - M3";
  const fit = button("Fit", false);
  const exaggeration = document.createElement("label");
  exaggeration.className = "gfx-control";
  exaggeration.textContent = "z ×";
  const exaggerationInput = document.createElement("input");
  exaggerationInput.type = "range";
  exaggerationInput.min = "1";
  exaggerationInput.max = "20";
  exaggerationInput.step = "0.5";
  exaggerationInput.value = "1";
  exaggeration.appendChild(exaggerationInput);
  const fullToggle = document.createElement("label");
  fullToggle.className = "gfx-control";
  fullToggle.title = "show full extent (unpainted)";
  const fullInput = document.createElement("input");
  fullInput.type = "checkbox";
  fullToggle.append(fullInput, document.createTextNode(" full extent"));
  menubar.append(title, to3d, to2d, fit, fullToggle, exaggeration);

  // -- explorer -------------------------------------------------------------
  const explorer = new Explorer(meta, {
    onPatch: (patch) => sendPatch(model, patch),
    onSelect: (selection) => sendPatch(model, { selection }),
  });

  // -- viewport -------------------------------------------------------------
  const viewport = document.createElement("div");
  viewport.className = "gfx-viewport";

  el.append(menubar, explorer.root, viewport);

  const scene = new Scene3D(viewport);
  scene.setBounds(geometry.size);

  const levels = new Map<string, LevelMeshes>();
  for (const level of geometry.levels) {
    const style = currentStyles(model)[level.name];
    const meshes = new LevelMeshes(level, style?.color ?? "#cccccc", style?.opacity ?? 0.9);
    levels.set(level.name, meshes);
    scene.content.add(meshes.group);
  }
  const glyphs = new PortGlyphs(meta.ports);
  scene.content.add(glyphs.group);

  fit.addEventListener("click", () => scene.fit());
  exaggerationInput.addEventListener("input", () =>
    scene.setZExaggeration(Number(exaggerationInput.value)),
  );
  fullInput.addEventListener("change", () =>
    sendPatch(model, { show_full: fullInput.checked }),
  );

  // -- state sync -----------------------------------------------------------
  const applyState = (): void => {
    const state = model.get("state") as StatePayload;
    for (const [name, meshes] of levels) {
      const style = state.styles[name];
      if (!style) continue;
      meshes.setColor(style.color);
      meshes.setOpacity(style.opacity);
      meshes.setVisible(style.visible);
      meshes.setShowFull(state.show_full);
      meshes.setSelected(state.selection?.kind === "level" && state.selection.name === name);
    }
    for (const port of meta.ports) {
      glyphs.setVisible(port.name, state.port_visible[port.name] ?? true);
    }
    glyphs.setSelected(state.selection?.kind === "port" ? state.selection.name : null);
    fullInput.checked = state.show_full;
    explorer.render(state);
  };

  model.on("change:state", applyState);
  applyState();

  return () => {
    scene.dispose();
    for (const meshes of levels.values()) meshes.dispose();
    glyphs.dispose();
  };
}

function sendPatch(model: AnyModel, patch: Record<string, unknown>): void {
  model.send({ event: "set_state", patch });
}

function currentStyles(model: AnyModel): Record<string, { color: string; opacity: number }> {
  const state = model.get("state") as StatePayload | undefined;
  return state?.styles ?? {};
}

function button(text: string, active: boolean): HTMLButtonElement {
  const element = document.createElement("button");
  element.type = "button";
  element.className = active ? "gfx-button gfx-active" : "gfx-button";
  element.textContent = text;
  return element;
}

export default { render };
