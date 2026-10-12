// The app-shell anywidget frontend (spec D2, D6, D9).
//
// One widget root holds the menubar, the explorer and the viewport region
// (spec D9). The viewport region holds both the 3D stack (three.js) and the 2D
// cross-section (Canvas2D, spec R4); ``active_viewport`` picks which is shown.
import "./style.css";

import { Explorer } from "./explorer";
import { PortGlyphs } from "./glyphs";
import { LevelMeshes } from "./mesh";
import { Scene3D } from "./scene";
import { Section2D } from "./section";
import type {
  GeometryPayload,
  MetaPayload,
  SectionPayload,
  StatePayload,
} from "./wire";

interface AnyModel {
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  save_changes(): void;
  send(content: Record<string, unknown>, callbacks?: unknown): void;
  on(event: string, handler: () => void): void;
}

const PLANES = ["xy", "xz", "yz"] as const;

function render({ model, el }: { model: AnyModel; el: HTMLElement }): () => void {
  const geometry = model.get("geometry") as GeometryPayload;
  const meta = model.get("meta") as MetaPayload;

  el.classList.add("gfx-shell");
  el.innerHTML = "";

  // -- menubar (spec D9) ----------------------------------------------------
  const menubar = document.createElement("div");
  menubar.className = "gfx-menubar";
  const title = document.createElement("span");
  title.className = "gfx-title";
  title.textContent = "gdsfactory viewer";
  const to3d = button("3D", true);
  const to2d = button("2D", false);
  const fit = button("Fit", false);
  const fullToggle = document.createElement("label");
  fullToggle.className = "gfx-control";
  fullToggle.title = "show full extent (unpainted)";
  const fullInput = document.createElement("input");
  fullInput.type = "checkbox";
  fullToggle.append(fullInput, document.createTextNode(" full extent"));
  const exaggeration = document.createElement("label");
  exaggeration.className = "gfx-control";
  exaggeration.title = "vertical exaggeration (3D)";
  exaggeration.textContent = "z ×";
  const exaggerationInput = document.createElement("input");
  exaggerationInput.type = "range";
  exaggerationInput.min = "1";
  exaggerationInput.max = "20";
  exaggerationInput.step = "0.5";
  exaggerationInput.value = "1";
  exaggeration.appendChild(exaggerationInput);
  menubar.append(title, to3d, to2d, fit, fullToggle, exaggeration);

  // -- section controls (2D only, spec R4) ---------------------------------
  const sectionControls = document.createElement("div");
  sectionControls.className = "gfx-section-controls";
  sectionControls.hidden = true;

  const planeToggle = document.createElement("div");
  planeToggle.className = "gfx-plane-toggle";
  const planeButtons = new Map<string, HTMLButtonElement>();
  for (const plane of PLANES) {
    const planeButton = button(plane, plane === "xy");
    planeButton.addEventListener("click", () => sendPatch(model, { plane }));
    planeButtons.set(plane, planeButton);
    planeToggle.appendChild(planeButton);
  }

  const position = document.createElement("label");
  position.className = "gfx-control";
  const positionSlider = document.createElement("input");
  positionSlider.type = "range";
  positionSlider.step = "0.01";
  const positionBox = document.createElement("input");
  positionBox.type = "number";
  positionBox.step = "0.01";
  positionBox.className = "gfx-numeric";
  position.append(positionSlider, positionBox);

  const sectionExaggeration = document.createElement("label");
  sectionExaggeration.className = "gfx-control";
  sectionExaggeration.title = "vertical exaggeration (2D)";
  sectionExaggeration.textContent = "z ×";
  const sectionExaggerationInput = document.createElement("input");
  sectionExaggerationInput.type = "range";
  sectionExaggerationInput.min = "1";
  sectionExaggerationInput.max = "20";
  sectionExaggerationInput.step = "0.5";
  sectionExaggerationInput.value = "1";
  sectionExaggeration.appendChild(sectionExaggerationInput);

  sectionControls.append(planeToggle, position, sectionExaggeration);

  // -- explorer -------------------------------------------------------------
  const explorer = new Explorer(meta, {
    onPatch: (patch) => sendPatch(model, patch),
    onSelect: (selection) => sendPatch(model, { selection }),
  });

  // -- viewport -------------------------------------------------------------
  const viewport = document.createElement("div");
  viewport.className = "gfx-viewport";
  const sceneRoot = document.createElement("div");
  sceneRoot.className = "gfx-viewport-3d";
  viewport.appendChild(sceneRoot);

  el.append(menubar, sectionControls, explorer.root, viewport);

  const scene = new Scene3D(sceneRoot);
  scene.setBounds(geometry.size);
  const section = new Section2D(meta, (name) => sendPatch(model, {
    selection: name ? { kind: "port", name } : null,
  }));
  viewport.appendChild(section.root);

  const levels = new Map<string, LevelMeshes>();
  for (const level of geometry.levels) {
    const style = currentStyles(model)[level.name];
    const meshes = new LevelMeshes(level, style?.color ?? "#cccccc", style?.opacity ?? 0.9);
    levels.set(level.name, meshes);
    scene.content.add(meshes.group);
  }
  const glyphs = new PortGlyphs(meta.ports);
  scene.content.add(glyphs.group);

  fit.addEventListener("click", () => {
    if (currentState(model)?.active_viewport === "2d") section.fit();
    else scene.fit();
  });
  exaggerationInput.addEventListener("input", () =>
    scene.setZExaggeration(Number(exaggerationInput.value)),
  );
  sectionExaggerationInput.addEventListener("input", () =>
    section.setExaggeration(Number(sectionExaggerationInput.value)),
  );
  fullInput.addEventListener("change", () =>
    sendPatch(model, { show_full: fullInput.checked }),
  );
  to3d.addEventListener("click", () => sendPatch(model, { active_viewport: "3d" }));
  to2d.addEventListener("click", () => sendPatch(model, { active_viewport: "2d" }));

  // The slider drags faster than the kernel can round trip (spec D10), so
  // coalesce to one patch per frame, keeping the latest value (the pattern
  // explorer.ts uses for opacity).
  let pendingPosition: number | null = null;
  let positionFrame = 0;
  const flushPosition = (): void => {
    positionFrame = 0;
    if (pendingPosition === null) return;
    sendPatch(model, { plane_position: pendingPosition });
    pendingPosition = null;
  };
  const queuePosition = (value: number): void => {
    pendingPosition = value;
    if (positionFrame === 0) positionFrame = requestAnimationFrame(flushPosition);
  };
  positionSlider.addEventListener("input", () => {
    positionBox.value = positionSlider.value;
    queuePosition(Number(positionSlider.value));
  });
  positionBox.addEventListener("change", () => {
    const value = Number(positionBox.value);
    positionSlider.value = String(value);
    queuePosition(value);
  });

  // -- state sync -----------------------------------------------------------
  const applyState = (): void => {
    const state = model.get("state") as StatePayload;
    const is2d = state.active_viewport === "2d";
    to3d.classList.toggle("gfx-active", !is2d);
    to2d.classList.toggle("gfx-active", is2d);
    sceneRoot.hidden = is2d;
    section.root.hidden = !is2d;
    sectionControls.hidden = !is2d;
    for (const [plane, planeButton] of planeButtons) {
      planeButton.classList.toggle("gfx-active", plane === state.plane);
    }
    const [lo, hi] = state.plane_bounds;
    positionSlider.min = String(lo);
    positionSlider.max = String(hi);
    if (document.activeElement !== positionBox) positionBox.value = String(state.plane_position);
    if (document.activeElement !== positionSlider) positionSlider.value = String(state.plane_position);

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
    section.setState(state);
    explorer.render(state);
  };

  const applySection = (): void => {
    const payload = model.get("section") as SectionPayload | undefined;
    if (payload && Object.keys(payload).length > 0) section.setPayload(payload);
  };

  model.on("change:state", applyState);
  model.on("change:section", applySection);
  applySection();
  applyState();

  return () => {
    if (positionFrame !== 0) cancelAnimationFrame(positionFrame);
    section.dispose();
    scene.dispose();
    for (const meshes of levels.values()) meshes.dispose();
    glyphs.dispose();
  };
}

function sendPatch(model: AnyModel, patch: Record<string, unknown>): void {
  model.send({ event: "set_state", patch });
}

function currentState(model: AnyModel): StatePayload | undefined {
  return model.get("state") as StatePayload | undefined;
}

function currentStyles(model: AnyModel): Record<string, { color: string; opacity: number }> {
  return currentState(model)?.styles ?? {};
}

function button(text: string, active: boolean): HTMLButtonElement {
  const element = document.createElement("button");
  element.type = "button";
  element.className = active ? "gfx-button gfx-active" : "gfx-button";
  element.textContent = text;
  return element;
}

export default { render };
