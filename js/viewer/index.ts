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

//: Human labels for the run's phases (spec D15). Placement and JAX compilation
//: report no fraction, so naming the phase is the only feedback they can give.
const PHASE_LABELS: Record<string, string> = {
  placing: "placing…",
  compiling: "compiling…",
  stepping: "stepping…",
};

//: Human labels for the rotation styles the settings screen offers (spec R3).
const CONTROL_LABELS: Record<ControlKind, string> = {
  trackball: "Trackball (free roll)",
  arcball: "Arcball (gizmo)",
  orbit: "Orbit (turntable)",
};

//: A camera glyph for the screenshot button (spec R3, R4): it saves an image
//: of the active viewport, or of the whole widget, per the settings toggle.
const SAVE_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>' +
  '<circle cx="12" cy="13" r="4"/></svg>';

//: A gear glyph for the settings button (spec R3).
const GEAR_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<circle cx="12" cy="12" r="3"/>' +
  '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';

//: A "fit to view" glyph for the Fit button (four arrows pointing outward).
const FIT_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>' +
  '<polyline points="21 15 21 21 15 21"/><polyline points="3 9 3 3 9 3"/>' +
  '<line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>' +
  '<line x1="21" y1="21" x2="14" y2="14"/><line x1="3" y1="3" x2="10" y2="10"/></svg>';

//: A play triangle for the Run button (spec R14 mode 1).
const RUN_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" ' +
  'stroke="currentColor" stroke-width="1" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M7 4.5v15l13-7.5z"/></svg>';

function render({ model, el }: { model: AnyModel; el: HTMLElement }): () => void {
  const geometry = model.get("geometry") as GeometryPayload;
  const meta = model.get("meta") as MetaPayload;

  el.classList.add("gfx-shell");
  el.innerHTML = "";

  // -- menubar (spec D9) ----------------------------------------------------
  const menubar = document.createElement("div");
  menubar.className = "gfx-menubar";
  const to3d = button("3D", true);
  const to2d = button("2D", false);
  const fit = iconButton(FIT_ICON, "fit the view to the device");
  const run = iconButton(RUN_ICON, "run the simulation in the kernel (spec R14 mode 1)");
  // the run's phase label and progress bar (spec D15): hidden until a run
  // starts; the label names the phase, the bar fills from the kernel's
  // `progress` trait (indeterminate until the step total is known)
  const runPhase = document.createElement("span");
  runPhase.className = "gfx-run-phase";
  runPhase.hidden = true;
  const runProgress = document.createElement("div");
  runProgress.className = "gfx-run-progress";
  runProgress.hidden = true;
  runProgress.setAttribute("role", "progressbar");
  runProgress.setAttribute("aria-label", "simulation progress");
  const runProgressFill = document.createElement("div");
  runProgressFill.className = "gfx-run-progress-fill";
  runProgress.append(runProgressFill);
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
  // the view controls sit left (3D, 2D, Fit, then the shared z multiplier);
  // the spacer starts the right-hand group, which reads left to right: the
  // run's phase label and progress bar, then Run, Save and Settings
  const spacer = document.createElement("div");
  spacer.className = "gfx-spacer";
  menubar.append(to3d, to2d, fit, exaggeration, spacer, runPhase, runProgress, run, save, settings);

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
    onAddDetector: (port) => send(model, { event: "add_detector", port }),
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

  // screenshot scope (spec R3, D14): the camera button saves just the active
  // viewport by default, or the whole widget when this is checked. It is a
  // per-viewer preference, kept in localStorage (it never reaches the kernel).
  const shotRow = document.createElement("label");
  shotRow.className = "gfx-settings-row";
  shotRow.title = "what the camera button saves";
  const shotLabel = document.createElement("span");
  shotLabel.textContent = "Screenshot";
  const shotInput = document.createElement("input");
  shotInput.type = "checkbox";
  shotInput.checked = readShotWhole();
  const shotHint = document.createElement("span");
  shotHint.className = "gfx-settings-value";
  shotHint.textContent = shotInput.checked ? "whole widget" : "viewport";
  shotInput.addEventListener("change", () => {
    shotHint.textContent = shotInput.checked ? "whole widget" : "viewport";
    writeShotWhole(shotInput.checked);
  });
  shotRow.append(shotLabel, shotInput, shotHint);

  // save target (spec R3, D14): the camera button downloads the PNG in the
  // browser by default, or hands it to the kernel to write beside the notebook.
  // A per-viewer preference, kept in localStorage like the scope above.
  const saveRow = document.createElement("label");
  saveRow.className = "gfx-settings-row";
  saveRow.title = "where the camera button writes";
  const saveLabel = document.createElement("span");
  saveLabel.textContent = "Save to";
  const saveSelect = document.createElement("select");
  saveSelect.className = "gfx-select";
  for (const [value, text] of [
    ["browser", "browser (download)"],
    ["server", "server (kernel)"],
  ] as const) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    saveSelect.appendChild(option);
  }
  saveSelect.value = readSaveTarget();
  saveSelect.addEventListener("change", () => {
    writeSaveTarget(saveSelect.value as SaveTarget);
    applySaveFolderVisibility();
  });
  saveRow.append(saveLabel, saveSelect);

  // the folder a server-side save writes into (spec D14). A browser cannot
  // browse the kernel's filesystem, so this is a path string the kernel
  // validates and creates; it shows only when saving to the server.
  const folderRow = document.createElement("label");
  folderRow.className = "gfx-settings-row";
  folderRow.title = "folder a server-side save writes into";
  const folderLabel = document.createElement("span");
  folderLabel.textContent = "Save folder";
  const folderInput = document.createElement("input");
  folderInput.type = "text";
  folderInput.className = "gfx-text";
  folderInput.value = (model.get("save_dir") as string | undefined) ?? "";
  folderInput.addEventListener("change", () =>
    send(model, { event: "set_save_dir", dir: folderInput.value }),
  );
  folderRow.append(folderLabel, folderInput);
  // the folder field is only meaningful for a server-side save
  function applySaveFolderVisibility(): void {
    folderRow.hidden = saveSelect.value !== "server";
  }

  settingsPanel.append(
    settingsHeading,
    controlRow,
    sensitivityRow,
    fadeRow,
    fullRow,
    shotRow,
    saveRow,
    folderRow,
  );
  viewport.appendChild(settingsPanel);

  // a transient message over the shell (spec D14): confirms a kernel-side save
  // or reports a failure, so a server write is not silent
  const toast = document.createElement("div");
  toast.className = "gfx-toast";
  toast.hidden = true;

  el.append(menubar, sectionControls, explorer.root, viewport, toast);

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
    const viewportPng = is2d ? section.toPng() : scene.toPng();
    // the whole widget is a heavier capture (rasterizing the shell), so build
    // it first and deliver the result once it is ready
    const shot: Promise<[string, string]> = readShotWhole()
      ? captureShell(el, viewportPng).then(
          (png): [string, string] => [png, is2d ? "viewer-2d.png" : "viewer-3d.png"],
        )
      : Promise.resolve([viewportPng, is2d ? "cross-section.png" : "view-3d.png"]);
    shot.then(([png, filename]) => {
      if (readSaveTarget() === "server") send(model, { event: "save_image", data: png, filename });
      else download(png, filename);
    });
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
    runPhase.hidden = !state.running;
    runProgress.hidden = !state.running;
    if (!state.running) {
      applyPhase();
      applyProgress();
    }
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

  // the run's phase (spec D15): name it left of the bar, so placement and
  // compilation — which report no fraction — are not a bare pulsing bar
  const applyPhase = (): void => {
    const name = (model.get("phase") as string | undefined) ?? "";
    runPhase.textContent = PHASE_LABELS[name] ?? "";
  };

  // the run's progress (spec D15): fill the bar from the kernel's `progress`
  // trait; until the total is known the bar is indeterminate (the CSS sweep)
  const applyProgress = (): void => {
    const payload = model.get("progress") as { step?: number; total?: number } | undefined;
    const step = payload?.step ?? 0;
    const total = payload?.total ?? 0;
    if (total > 0) {
      runProgressFill.style.width = `${Math.min(100, (100 * step) / total).toFixed(1)}%`;
      runProgress.classList.remove("gfx-run-progress-indeterminate");
      runProgress.setAttribute("aria-valuenow", String(step));
      runProgress.setAttribute("aria-valuemax", String(total));
    } else {
      runProgressFill.style.width = "100%";
      runProgress.classList.add("gfx-run-progress-indeterminate");
      runProgress.removeAttribute("aria-valuenow");
      runProgress.removeAttribute("aria-valuemax");
    }
  };

  // a kernel-side save's outcome (spec D14): confirm the written path, or show
  // the error; the toast clears itself after a few seconds
  let toastTimer = 0;
  const showToast = (message: string): void => {
    toast.textContent = message;
    toast.hidden = false;
    if (toastTimer !== 0) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toast.hidden = true;
      toastTimer = 0;
    }, 4000);
  };
  const applySaveResult = (): void => {
    const payload = model.get("save_result") as
      | { path?: string; folder?: string; error?: string }
      | undefined;
    if (!payload || Object.keys(payload).length === 0) return;
    if (payload.error) showToast(`save failed: ${payload.error}`);
    else if (payload.folder) showToast(`save folder: ${payload.folder}`);
    else if (payload.path) showToast(`saved ${payload.path}`);
  };

  // the kernel is the authority on the save folder (it resolves and creates
  // it), so reflect its value back into the field (spec D14)
  const applySaveDir = (): void => {
    const folder = model.get("save_dir") as string | undefined;
    if (typeof folder === "string" && document.activeElement !== folderInput) {
      folderInput.value = folder;
    }
  };

  model.on("change:state", applyState);
  model.on("change:progress", applyProgress);
  model.on("change:phase", applyPhase);
  model.on("change:section", applySection);
  model.on("change:save_result", applySaveResult);
  model.on("change:save_dir", applySaveDir);
  applySection();
  applyState();
  applyProgress();
  applyPhase();
  applySaveDir();
  applySaveFolderVisibility();

  return () => {
    if (positionFrame !== 0) cancelAnimationFrame(positionFrame);
    if (toastTimer !== 0) window.clearTimeout(toastTimer);
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

//: the localStorage key for the screenshot scope (spec R3, D14); a per-viewer
//: preference, so it stays in the browser and never reaches the kernel
const SHOT_WHOLE_KEY = "gfx-shot-whole";

/** Whether the camera button should capture the whole widget (default: no). */
function readShotWhole(): boolean {
  try {
    return window.localStorage.getItem(SHOT_WHOLE_KEY) === "1";
  } catch {
    return false; // a blocked/absent store just means the default
  }
}

function writeShotWhole(whole: boolean): void {
  try {
    window.localStorage.setItem(SHOT_WHOLE_KEY, whole ? "1" : "0");
  } catch {
    // storage unavailable (private window, blocked) — the setting is session-only
  }
}

//: the localStorage key for where the save button writes (spec R3, D14):
//: "browser" downloads the PNG, "server" sends it to the kernel to write
const SAVE_TO_KEY = "gfx-save-to";

type SaveTarget = "browser" | "server";

/** Whether the camera button downloads or writes server-side (default: browser). */
function readSaveTarget(): SaveTarget {
  try {
    return window.localStorage.getItem(SAVE_TO_KEY) === "server" ? "server" : "browser";
  } catch {
    return "browser"; // a blocked/absent store just means the default
  }
}

function writeSaveTarget(target: SaveTarget): void {
  try {
    window.localStorage.setItem(SAVE_TO_KEY, target);
  } catch {
    // storage unavailable (private window, blocked) — the setting is session-only
  }
}

/**
 * The widget's own rules, as CSS text, read from wherever the host scoped them.
 *
 * A light-DOM clone is unstyled: in marimo the widget lives in a shadow root
 * and its rules are `adoptedStyleSheets`, which the clone does not inherit. So
 * the capture carries the rules along and keeps only the ones that mention a
 * `gfx-` class, leaving the host's global styles behind.
 */
function shellStyleText(element: HTMLElement): string {
  const root = element.getRootNode() as Document | ShadowRoot;
  const parts: string[] = [];
  const adopted = (root as ShadowRoot).adoptedStyleSheets;
  if (adopted) {
    for (const sheet of adopted) {
      try {
        parts.push(Array.from(sheet.cssRules).map((rule) => rule.cssText).join("\n"));
      } catch {
        // a cross-origin sheet has no readable rules; skip it
      }
    }
  }
  for (const node of root.querySelectorAll?.("style") ?? []) {
    parts.push(node.textContent ?? "");
  }
  return parts.filter((text) => text.includes("gfx-")).join("\n");
}

/**
 * Rasterizes the whole widget shell into one PNG data URL (spec D14).
 *
 * html2canvas can neither clone a shadow root nor read a `<canvas>` through a
 * clone, so the shell is first copied into the light DOM with three fixes:
 *
 * - the widget's `gfx-` rules are re-inserted inside the clone (see
 *   `shellStyleText`), since the clone is otherwise unstyled;
 * - each `<canvas>` is replaced by an `<img>` of its current pixels — the
 *   three.js canvas keeps its drawing buffer (see `Scene3D`) and the section
 *   canvas is a 2D surface, so both `toDataURL` cleanly;
 * - the clone is laid out on screen at the origin behind the page
 *   (`z-index: -1`): `foreignObjectRendering` hands rasterization to the
 *   browser, which only paints elements it actually lays out, so an off-screen
 *   holder captures empty.
 *
 * The host's stylesheets share the document, and html2canvas 1.4.1's color
 * parser throws on the modern color functions they use (`color()`, `oklch()`,
 * `color-mix()`, `light-dark()`). `parseBackgroundColor` parses the document's
 * `html`/`body` backgrounds before the render branch, so `onclone` clears them;
 * every other color is parsed by the canvas renderer, which
 * `foreignObjectRendering` replaces with the browser's own rasterizer.
 *
 * If html2canvas is unavailable or still throws, the viewport image is
 * returned instead, so the button always produces something.
 */
async function captureShell(shell: HTMLElement, fallbackPng: string): Promise<string> {
  let holder: HTMLDivElement | null = null;
  try {
    const { default: html2canvas } = await import("html2canvas");
    const rect = shell.getBoundingClientRect();

    const clone = shell.cloneNode(true) as HTMLElement;
    const style = document.createElement("style");
    style.textContent = shellStyleText(shell);
    clone.insertBefore(style, clone.firstChild);

    const sources = Array.from(shell.querySelectorAll("canvas"));
    const targets = Array.from(clone.querySelectorAll("canvas"));
    for (let i = 0; i < sources.length; i++) {
      const image = document.createElement("img");
      try {
        image.src = sources[i].toDataURL("image/png");
      } catch {
        continue; // tainted canvas: leave it to render blank rather than abort
      }
      const bounds = sources[i].getBoundingClientRect();
      image.style.width = `${Math.round(bounds.width)}px`;
      image.style.height = `${Math.round(bounds.height)}px`;
      targets[i].replaceWith(image);
    }

    holder = document.createElement("div");
    holder.style.cssText =
      `position:fixed;left:0;top:0;width:${Math.round(rect.width)}px;z-index:-1;pointer-events:none;`;
    holder.appendChild(clone);
    document.body.appendChild(holder);

    // let layout settle and the frozen images decode before rasterizing
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
    await Promise.all(
      Array.from(clone.querySelectorAll("img")).map((image) =>
        image.complete
          ? Promise.resolve()
          : new Promise((resolve) => {
              image.addEventListener("load", resolve, { once: true });
              image.addEventListener("error", resolve, { once: true });
            }),
      ),
    );

    const canvas = await html2canvas(clone, {
      backgroundColor: null,
      // the shell is dark; keep its own colors rather than html2canvas's white
      useCORS: true,
      foreignObjectRendering: true,
      // the document backgrounds are not part of the shell being captured, so
      // clearing them on the clone loses nothing and keeps the parse above from
      // reading the host's modern-syntax background (see the note above)
      onclone: (doc: Document) => {
        doc.documentElement.style.backgroundColor = "transparent";
        if (doc.body) doc.body.style.backgroundColor = "transparent";
      },
    });
    return canvas.toDataURL("image/png");
  } catch (error) {
    console.warn("whole-widget capture failed; saving the viewport instead", error);
    return fallbackPng;
  } finally {
    holder?.remove();
  }
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

//: A menubar button whose label is a glyph, with the text in its tooltip (spec D9).
function iconButton(icon: string, title: string): HTMLButtonElement {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "gfx-button gfx-icon";
  element.title = title;
  element.innerHTML = icon;
  return element;
}

export default { render };
