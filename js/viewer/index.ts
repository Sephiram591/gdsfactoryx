// The app-shell anywidget frontend (spec D2, D6, D9).
//
// One widget root holds the menubar, the explorer and the viewport region
// (spec D9). The viewport region holds both the 3D stack (three.js) and the 2D
// cross-section (Canvas2D, spec R4); ``active_viewport`` picks which is shown.
import "./style.css";

import * as THREE from "three";

import { Explorer } from "./explorer";
import { PlaneGlyphs, PortGlyphs } from "./glyphs";
import { LevelMeshes } from "./mesh";
import { Scene3D, CONTROL_KINDS, type ControlKind } from "./scene";
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

//: Human labels for the rotation styles the settings screen offers (spec R3).
const CONTROL_LABELS: Record<ControlKind, string> = {
  trackball: "Trackball (free roll)",
  arcball: "Arcball (gizmo)",
  orbit: "Orbit (turntable)",
};

//: A floppy-disk glyph for the save-image button (spec R3, R4).
const SAVE_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>' +
  '<polyline points="17 21 17 13 7 13 7 21"/>' +
  '<polyline points="7 3 7 8 15 8"/></svg>';

//: A gear glyph for the settings button (spec R3).
const GEAR_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<circle cx="12" cy="12" r="3"/>' +
  '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';

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
  const run = button("Run", false);
  run.title = "run the simulation in the kernel (spec R14 mode 1)";
  const runNote = document.createElement("span");
  runNote.className = "gfx-run-note";
  runNote.hidden = true;
  runNote.textContent = "running…";
  const settings = document.createElement("button");
  settings.type = "button";
  settings.className = "gfx-button gfx-settings";
  settings.title = "viewer settings";
  settings.innerHTML = GEAR_ICON;
  const exaggeration = document.createElement("label");
  exaggeration.className = "gfx-control";
  exaggeration.title = "vertical exaggeration (both viewports)";
  exaggeration.textContent = "z ×";
  const exaggerationInput = document.createElement("input");
  exaggerationInput.type = "range";
  exaggerationInput.min = "1";
  exaggerationInput.max = "20";
  exaggerationInput.step = "0.5";
  exaggerationInput.value = "1";
  exaggeration.appendChild(exaggerationInput);
  const save = document.createElement("button");
  save.type = "button";
  save.className = "gfx-button gfx-save";
  save.title = "save the current viewport as a PNG";
  save.innerHTML = SAVE_ICON;
  menubar.append(title, to3d, to2d, fit, run, runNote, settings, exaggeration, save);

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

  // the z multiplier is shared with the 3D view; it lives in the menubar only
  sectionControls.append(planeToggle, position);

  // -- explorer -------------------------------------------------------------
  const explorer = new Explorer(meta, {
    onPatch: (patch) => sendPatch(model, patch),
    onSelect: (selection) => sendPatch(model, { selection }),
    onAddSource: (port) => send(model, { event: "add_source", port }),
    onUpdateSpec: (kind, id, change) =>
      send(model, { event: "update_spec", kind, spec: { id, ...change } }),
    onRemoveSpec: (kind, id) => send(model, { event: "remove_spec", kind, id }),
    onSettings: (settings) => send(model, { event: "set_settings", settings }),
  });

  // -- viewport -------------------------------------------------------------
  const viewport = document.createElement("div");
  viewport.className = "gfx-viewport";
  const sceneRoot = document.createElement("div");
  sceneRoot.className = "gfx-viewport-3d";
  viewport.appendChild(sceneRoot);

  // -- settings screen (replaces the viewport, spec R3) ---------------------
  const settingsPanel = document.createElement("div");
  settingsPanel.className = "gfx-settings-panel";
  settingsPanel.hidden = true;
  const settingsHeading = document.createElement("div");
  settingsHeading.className = "gfx-settings-heading";
  settingsHeading.textContent = "Viewer settings";
  const controlRow = document.createElement("label");
  controlRow.className = "gfx-settings-row";
  const controlLabel = document.createElement("span");
  controlLabel.textContent = "Rotation style";
  const controlSelect = document.createElement("select");
  controlSelect.className = "gfx-select";
  for (const kind of CONTROL_KINDS) {
    const option = document.createElement("option");
    option.value = kind;
    option.textContent = CONTROL_LABELS[kind];
    controlSelect.appendChild(option);
  }
  controlRow.append(controlLabel, controlSelect);

  const sensitivityRow = document.createElement("label");
  sensitivityRow.className = "gfx-settings-row";
  const sensitivityLabel = document.createElement("span");
  sensitivityLabel.textContent = "Rotation sensitivity";
  const sensitivityInput = document.createElement("input");
  sensitivityInput.type = "range";
  sensitivityInput.className = "gfx-sensitivity";
  sensitivityInput.min = "0.2";
  sensitivityInput.max = "3";
  sensitivityInput.step = "0.1";
  const sensitivityValue = document.createElement("span");
  sensitivityValue.className = "gfx-settings-value";
  sensitivityRow.append(sensitivityLabel, sensitivityInput, sensitivityValue);

  const fullRow = document.createElement("label");
  fullRow.className = "gfx-settings-row";
  fullRow.title = "show full extent (unpainted)";
  const fullLabel = document.createElement("span");
  fullLabel.textContent = "Full extent";
  const fullInput = document.createElement("input");
  fullInput.type = "checkbox";
  const fullHint = document.createElement("span");
  fullHint.className = "gfx-settings-value";
  fullHint.textContent = "unpainted";
  fullRow.append(fullLabel, fullInput, fullHint);

  // the 2D port fade's 1/e distance (spec R4): larger keeps ports visible across
  // more planes, smaller restricts them to their own plane
  const fadeRow = document.createElement("label");
  fadeRow.className = "gfx-settings-row";
  fadeRow.title = "2D port fade: distance (µm) at which a port's outline falls to 1/e";
  const fadeLabel = document.createElement("span");
  fadeLabel.textContent = "Port fade (µm)";
  const fadeInput = document.createElement("input");
  fadeInput.type = "range";
  fadeInput.className = "gfx-fade";
  fadeInput.min = "0.05";
  fadeInput.max = "5";
  fadeInput.step = "0.05";
  fadeInput.value = "0.5";
  const fadeValue = document.createElement("span");
  fadeValue.className = "gfx-settings-value";
  fadeRow.append(fadeLabel, fadeInput, fadeValue);

  settingsPanel.append(settingsHeading, controlRow, sensitivityRow, fadeRow, fullRow);
  viewport.appendChild(settingsPanel);

  el.append(menubar, sectionControls, explorer.root, viewport);

  const scene = new Scene3D(sceneRoot);
  scene.setBounds(geometry.size, geometry.origin);
  controlSelect.value = scene.control; // the scene's default (spec R3)
  sensitivityInput.value = String(scene.sensitivity);
  sensitivityValue.textContent = `${scene.sensitivity.toFixed(1)}×`;
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
  // source/detector planes, drawn from their specs before anything is placed
  // (spec R12, D11); rebuilt on every setup edit
  const planes = new PlaneGlyphs();
  scene.content.add(planes.group);

  // -- run (spec R14 mode 1, D15) -------------------------------------------
  run.addEventListener("click", () => {
    if (run.disabled) return;
    send(model, { event: "run" });
  });

  // -- double-click a port in 3D to add a detector (spec R12) ---------------
  const raycaster = new THREE.Raycaster();
  scene.renderer.domElement.addEventListener("dblclick", (event) => {
    if (currentState(model)?.running) return;
    const rect = scene.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, scene.camera);
    const port = glyphs.pick(raycaster);
    if (port) send(model, { event: "add_detector", port });
  });

  fit.addEventListener("click", () => {
    closeSettings(); // switching the view or fitting returns to the viewport
    if (currentState(model)?.active_viewport === "2d") section.fit();
    else scene.fit();
  });

  // -- settings screen wiring (spec R3) -------------------------------------
  let settingsOpen = false;
  // the 3D/2D/Fit buttons leave the settings screen and return to the viewport
  const closeSettings = (): void => {
    if (!settingsOpen) return;
    settingsOpen = false;
    applySettingsVisibility();
  };
  const applySettingsVisibility = (): void => {
    settingsPanel.hidden = !settingsOpen;
    settings.classList.toggle("gfx-active", settingsOpen);
    // the settings screen replaces the viewport, so hide whichever viewport is
    // active (3D renderer or 2D canvas) while it is open
    const is2d = currentState(model)?.active_viewport === "2d";
    sceneRoot.hidden = settingsOpen || is2d;
    section.root.hidden = settingsOpen || !is2d;
    sectionControls.hidden = settingsOpen || !is2d;
    scene.setViewportVisible(!settingsOpen);
  };
  settings.addEventListener("click", () => {
    settingsOpen = !settingsOpen;
    applySettingsVisibility();
  });
  controlSelect.addEventListener("change", () => {
    scene.setControl(controlSelect.value as ControlKind);
  });
  sensitivityInput.addEventListener("input", () => {
    const value = Number(sensitivityInput.value);
    scene.setSensitivity(value);
    sensitivityValue.textContent = `${value.toFixed(1)}×`;
  });
  // one z multiplier drives both viewports (spec R3, R4)
  exaggerationInput.addEventListener("input", () => {
    const factor = Number(exaggerationInput.value);
    scene.setZExaggeration(factor);
    section.setExaggeration(factor);
  });
  fullInput.addEventListener("change", () =>
    sendPatch(model, { show_full: fullInput.checked }),
  );
  fadeInput.addEventListener("input", () => {
    const value = Number(fadeInput.value);
    fadeValue.textContent = `${value.toFixed(2)} µm`;
    sendPatch(model, { port_fade_um: value });
  });
  to3d.addEventListener("click", () => {
    closeSettings();
    sendPatch(model, { active_viewport: "3d" });
  });
  to2d.addEventListener("click", () => {
    closeSettings();
    sendPatch(model, { active_viewport: "2d" });
  });
  save.addEventListener("click", () => {
    const is2d = currentState(model)?.active_viewport === "2d";
    download(is2d ? section.toPng() : scene.toPng(), is2d ? "cross-section.png" : "view-3d.png");
  });

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
    // the settings screen replaces whichever viewport is active (spec R3)
    sceneRoot.hidden = settingsOpen || is2d;
    section.root.hidden = settingsOpen || !is2d;
    sectionControls.hidden = settingsOpen || !is2d;
    scene.setViewportVisible(!settingsOpen);
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
    // the source/detector planes, from the specs (spec R12, D11); selection
    // matches on the spec id, since two specs may share a name
    planes.rebuild(state.glyphs ?? []);
    for (const glyph of state.glyphs ?? []) planes.setVisible(glyph.id, glyph.visible);
    planes.setSelected(
      state.selection && (state.selection.kind === "source" || state.selection.kind === "detector")
        ? state.selection.name
        : null,
    );
    // while a mode-1 run is in flight the setup is frozen (spec D15); the
    // explorer disables its own controls, and Run is disabled here
    run.disabled = state.running;
    runNote.hidden = !state.running;
    fullInput.checked = state.show_full;
    if (document.activeElement !== fadeInput) fadeInput.value = String(state.port_fade_um);
    fadeValue.textContent = `${state.port_fade_um.toFixed(2)} µm`;
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
    planes.dispose();
  };
}

function sendPatch(model: AnyModel, patch: Record<string, unknown>): void {
  model.send({ event: "set_state", patch });
}

/** Sends a setup event (spec R12, R13, R14, D15). */
function send(model: AnyModel, content: Record<string, unknown>): void {
  model.send(content);
}

/** Triggers a browser download of a data-URL image. */
function download(dataUrl: string, filename: string): void {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.click();
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
