// The app-shell anywidget frontend (spec D2, D6, D9).
//
// One widget root holds the menubar, the explorer and the viewport region
// (spec D9). The viewport region holds both the 3D stack (three.js) and the 2D
// cross-section (Canvas2D, spec R4); ``active_viewport`` picks which is shown.
import "./style.css";

import * as THREE from "three";

import { Explorer } from "./explorer";
import { componentData, parseColor, rampColor, transformValues, valueRange } from "./fieldramp";
import { PlaneGlyphs, PortGlyphs } from "./glyphs";
import { LevelMeshes } from "./mesh";
import { Scene3D, CONTROL_KINDS, type ControlKind } from "./scene";
import { Section2D } from "./section";
import type {
  FieldPayload,
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
//: ``stopping`` is a client-side state: the run is unwinding after a stop (D17).
const PHASE_LABELS: Record<string, string> = {
  placing: "placing…",
  compiling: "compiling…",
  stepping: "stepping…",
  stopping: "stopping…",
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

//: A "copy" glyph (two overlapping rectangles) for the clipboard button: it
//: copies the same image the save button captures, into the browser clipboard.
const COPY_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>' +
  '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

//: An "export code" glyph for the launch-code button (spec R14 mode 3, D12):
//: a box with the code chevrons `</>` inside — the code-like indicator the
//: milestone asks for. It copies the kernel's launch code to the clipboard.
const CODE_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="2" y="3" width="20" height="18" rx="2" ry="2"/>' +
  '<polyline points="9 10 6.5 12 9 14"/>' +
  '<polyline points="15 10 17.5 12 15 14"/></svg>';

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

//: A filled square for the Stop button: the Run button becomes this while a run
//: is in flight (spec D17).
const STOP_ICON =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" ' +
  'stroke="currentColor" stroke-width="1" stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>';

const RUN_TITLE = "run the simulation in the kernel (spec R14 mode 1)";
const STOP_TITLE = "stop the running simulation (spec D17)";

function render({ model, el }: { model: AnyModel; el: HTMLElement }): () => void {
  let geometry = model.get("geometry") as GeometryPayload;
  let meta = model.get("meta") as MetaPayload;

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
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "gfx-button gfx-copy";
  copy.title = "copy the current viewport image to the clipboard";
  copy.innerHTML = COPY_ICON;
  const exportCode = document.createElement("button");
  exportCode.type = "button";
  exportCode.className = "gfx-button gfx-export-code";
  exportCode.title = "copy the simulation launch code to the clipboard";
  exportCode.innerHTML = CODE_ICON;
  // the view controls sit left (3D, 2D, Fit, then the shared z multiplier);
  // the spacer starts the right-hand group, which reads left to right: the
  // run's phase label and progress bar, then Run, Save, Copy, Export code
  // and Settings
  const spacer = document.createElement("div");
  spacer.className = "gfx-spacer";
  menubar.append(to3d, to2d, fit, exaggeration, spacer, runPhase, runProgress, run, save, copy, exportCode, settings);

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
    onActiveTab: (tab) => {
      // the render applies `result_styles` while the Result tab is active
      // (spec R17); patching it re-syncs the state, which re-renders
      sendPatch(model, { active_tab: tab });
    },
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

  // the 2D plane fade (spec R4): a fraction of the box size along the section's
  // normal; larger keeps planes and ports visible across more slices, smaller
  // restricts them to their own
  const fadeRow = document.createElement("label");
  fadeRow.className = "gfx-settings-row";
  fadeRow.title = "Plane fade: 1/e distance as a fraction of the box size along the plane's normal";
  const fadeLabel = document.createElement("span");
  fadeLabel.textContent = "Plane fade";
  const fadeInput = document.createElement("input");
  fadeInput.type = "range";
  fadeInput.className = "gfx-fade";
  fadeInput.min = "0.01";
  fadeInput.max = "0.25";
  fadeInput.step = "0.01";
  fadeInput.value = "0.05";
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
  controlSelect.value = scene.control; // the scene's default (spec R3)
  sensitivityInput.value = String(scene.sensitivity);
  sensitivityValue.textContent = `${scene.sensitivity.toFixed(1)}×`;
  const section = new Section2D(meta, (name) => sendPatch(model, {
    selection: name ? { kind: "port", name } : null,
  }));
  viewport.appendChild(section.root);

  // the field colorbar (spec R15): one per detector showing a field, over the 3D
  // viewport. It names the component and transform and shows the plane's own
  // range, so a shaded plane can be read (spec R15's AC).
  const colorbar = document.createElement("div");
  colorbar.className = "gfx-colorbar";
  colorbar.hidden = true;
  sceneRoot.appendChild(colorbar);

  // -- 3D content (spec D5, R13) --------------------------------------------
  // The levels, port glyphs and source/detector planes are built from the
  // payload's meshes and metadata. A boundary-offset edit re-boxes the device
  // and the kernel re-sends `geometry`/`meta`, so the whole content group is
  // rebuilt in place; the scene and the 2D section are kept (the section just
  // gets the new metadata).
  const content = scene.content;
  let levels = new Map<string, LevelMeshes>();
  let glyphs = new PortGlyphs(meta.ports);
  // source/detector planes, drawn from their specs before anything is placed
  // (spec R12, D11); rebuilt on every setup edit
  const planes = new PlaneGlyphs();

  const disposeContent = (): void => {
    for (const meshes of levels.values()) {
      meshes.dispose();
      // detach as well as dispose: `dispose()` frees the GPU buffers but the
      // group stays in the scene, so a re-box would leave the old, differently
      // sized meshes drawn over the new ones
      content.remove(meshes.group);
    }
    glyphs.dispose();
    content.remove(glyphs.group);
  };

  const rebuildContent = (): void => {
    disposeContent();
    levels = new Map<string, LevelMeshes>();
    for (const level of geometry.levels) {
      const style = currentStyles(model)[level.name];
      const meshes = new LevelMeshes(level, style?.color ?? "#cccccc", style?.opacity ?? 0.9);
      levels.set(level.name, meshes);
      content.add(meshes.group);
    }
    glyphs = new PortGlyphs(meta.ports);
    content.add(glyphs.group);
    // re-append so the plane glyphs stay after the meshes (a rebuild adds the
    // new meshes and glyphs to the end, which would otherwise cover them)
    content.add(planes.group);
    // a re-boxed device invalidates the glyphs, so re-shade from the last
    // result's fields (spec R15); the payload is keyed by spec id, not glyph
    applyFields();
    scene.setBounds(geometry.size, geometry.origin);
  };
  rebuildContent();

  // -- run / stop (spec R14 mode 1, D15, D17) --------------------------------
  // one toggle button: idle it starts a run, running it stops one. While the
  // stop is unwinding, the button disables and the phase reads "stopping…".
  run.addEventListener("click", () => {
    if (run.disabled) return;
    if (currentState(model)?.running) send(model, { event: "stop" });
    else send(model, { event: "run" });
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
    fadeValue.textContent = `${value.toFixed(2)}×`;
    sendPatch(model, { plane_fade_frac: value });
  });
  to3d.addEventListener("click", () => {
    closeSettings();
    sendPatch(model, { active_viewport: "3d" });
  });
  to2d.addEventListener("click", () => {
    closeSettings();
    sendPatch(model, { active_viewport: "2d" });
  });
  // the two capture buttons share one shot: the active viewport, or the whole
  // widget when the settings toggle says so (the widget is the heavier capture,
  // so the result is delivered once it is ready)
  const captureShot = (): Promise<[string, string]> => {
    const is2d = currentState(model)?.active_viewport === "2d";
    const viewportPng = is2d ? section.toPng() : scene.toPng();
    return readShotWhole()
      ? captureShell(el, viewportPng).then(
          (png): [string, string] => [png, is2d ? "viewer-2d.png" : "viewer-3d.png"],
        )
      : Promise.resolve([viewportPng, is2d ? "cross-section.png" : "view-3d.png"]);
  };
  save.addEventListener("click", () => {
    captureShot().then(([png, filename]) => {
      if (readSaveTarget() === "server") send(model, { event: "save_image", data: png, filename });
      else download(png, filename);
    });
  });
  // the clipboard write must be permitted as a user gesture, so it starts
  // directly in the click handler; only the (possibly slow) capture is async
  copy.addEventListener("click", () => {
    let write: Promise<void>;
    try {
      write = navigator.clipboard.write([]);
    } catch {
      showToast("clipboard is not available here");
      return;
    }
    captureShot().then(([png]) =>
      write
        .then(() => fetch(png))
        .then((r) => r.blob())
        .then((blob) =>
          navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]),
        )
        .then(() => showToast("copied image to the clipboard"))
        .catch(() => showToast("copy failed: clipboard blocked or unsupported")),
    );
  });
  // the launch code rides the `launch_code` trait, refreshed on every setup
  // edit (spec D12), so the copy is synchronous — no kernel round trip — and
  // keeps the user-gesture permission the clipboard API requires
  exportCode.addEventListener("click", () => {
    const code = String(model.get("launch_code") || "");
    if (!code) {
      showToast("no launch code yet");
      return;
    }
    navigator.clipboard
      .writeText(code)
      .then(() => showToast("copied launch code to the clipboard"))
      .catch(() => showToast("copy failed: clipboard blocked or unsupported"));
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
  // the last result's detector fields (spec R15, D16): hand them to the 3D
  // planes, the explorer's component selects and the 2D section. Called from
  // applyState (so a display-setting edit re-shades) and on change:fields (a
  // new result).
  function applyFields(): void {
    const fields = (model.get("fields") as Record<string, FieldPayload> | undefined) ?? {};
    const state = model.get("state") as StatePayload;
    const settings = state.field_settings ?? {};
    const logRange = state.simulation?.log_range ?? 6;
    // fields shade only while the Result tab is active (spec R17): the editor
    // views show the setup, never an overlay
    const shading = state.active_tab === "result";
    for (const glyph of state.glyphs ?? []) {
      if (glyph.kind !== "detector") continue;
      planes.setField(
        glyph.id, shading ? fields[glyph.id] ?? null : null, settings[glyph.id], logRange,
      );
    }
    explorer.setFields(fields);
    section.setFields(shading ? fields : {});
    renderColorbar(state, fields, shading);
  }

  /**
   * Draws the colorbar for the detector showing a field (spec R15).
   *
   * The bar is a canvas ramp beside the 3D viewport, labelled with the
   * detector's component, transform and the plane's own value range. Only one is
   * shown — the first detector with a field — so the overlay stays readable when
   * several detectors are shaded.
   */
  function renderColorbar(
    state: StatePayload,
    fields: Record<string, FieldPayload>,
    shading: boolean,
  ): void {
    if (!shading) {
      colorbar.hidden = true;
      return;
    }
    const settings = state.field_settings ?? {};
    const logRange = state.simulation?.log_range ?? 6;
    const entry = (state.glyphs ?? []).find((glyph) => {
      if (glyph.kind !== "detector" || !glyph.visible) return false;
      const s = settings[glyph.id];
      const field = fields[glyph.id];
      return Boolean(s?.field_visible && field && s.field_component);
    });
    if (!entry) {
      colorbar.hidden = true;
      return;
    }
    const s = settings[entry.id];
    const field = fields[entry.id];
    const data = componentData(field, s.field_component!, s.field_freq ?? 0);
    if (!data) {
      colorbar.hidden = true;
      return;
    }
    const values = transformValues(data.real, data.imag, s.field_transform);
    const range = valueRange(values, s.field_transform, logRange);
    const low = parseColor(s.field_color_low);
    const high = parseColor(s.field_color_high);
    const steps = 64;
    const pixels: string[] = [];
    for (let i = 0; i < steps; i += 1) {
      const [r, g, b, a] = rampColor(i / (steps - 1), s.field_transform, low, high);
      // the transparent end is shown against the panel, so the bar reads as the
      // overlay does (spec D16)
      pixels.push(`rgba(${r},${g},${b},${a.toFixed(3)}) ${(i / steps) * 100}%`);
    }
    colorbar.hidden = false;
    colorbar.replaceChildren(
      colorbarTitle(`${entry.name} · ${s.field_component} · ${s.field_transform}`),
      colorbarRamp(pixels.join(",")),
      colorbarScale(range[0], range[1]),
    );
  }

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

    // the kernel sends the active world's geometry and styles (spec R17): the
    // Result tab carries the result's own, the other tabs the editor's — so the
    // styles here are already the ones this view shows
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
    // the last result's detector fields shade their planes (spec R15, D16); the
    // explorer names each detector's recorded components and the section draws
    // the field where it is parallel to a detector's plane
    applyFields();
    // while a mode-1 run is in flight the setup is frozen (spec D15); the
    // explorer disables its own controls, and Run is disabled here
    applyRunControls();
    fullInput.checked = state.show_full;
    if (document.activeElement !== fadeInput) fadeInput.value = String(state.plane_fade_frac);
    fadeValue.textContent = `${state.plane_fade_frac.toFixed(2)}×`;
    section.setState(state);
    explorer.render(state);
  };

  const applySection = (): void => {
    const payload = model.get("section") as SectionPayload | undefined;
    if (payload && Object.keys(payload).length > 0) section.setPayload(payload);
  };

  // a re-boxed device (spec R13): the kernel re-sends `geometry` and `meta`
  // when a boundary-offset edit moves the box, so rebuild the 3D content, hand
  // the section and explorer the new metadata, and re-apply the state
  const applyGeometry = (): void => {
    geometry = model.get("geometry") as GeometryPayload;
    meta = model.get("meta") as MetaPayload;
    rebuildContent();
    section.setMeta(meta);
    explorer.setMeta(meta);
    applySection();
    applyState();
  };

  // the run's phase (spec D15): name it left of the bar, so placement and
  // compilation — which report no fraction — are not a bare pulsing bar
  const applyPhase = (): void => {
    const name = (model.get("phase") as string | undefined) ?? "";
    runPhase.textContent = PHASE_LABELS[name] ?? "";
  };

  // the Run/Stop toggle (spec D15, D17): idle the button starts a run; while one
  // is in flight it becomes Stop; once the stop is requested it disables and the
  // phase reads "stopping…" until the worker unwinds. The phase label and bar
  // show only while a run is in flight.
  const applyRunControls = (): void => {
    const state = currentState(model);
    const running = state?.running ?? false;
    const stopping = (model.get("stopping") as boolean | undefined) ?? false;
    run.innerHTML = running ? STOP_ICON : RUN_ICON;
    run.title = running ? STOP_TITLE : RUN_TITLE;
    run.disabled = running && stopping;
    runPhase.hidden = !running;
    runProgress.hidden = !running;
    if (running && stopping) {
      runPhase.textContent = PHASE_LABELS.stopping;
      runProgress.hidden = true;
    } else if (!running) {
      applyPhase();
      applyProgress();
    }
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
  model.on("change:fields", applyFields);
  model.on("change:progress", applyProgress);
  model.on("change:phase", applyPhase);
  model.on("change:stopping", applyRunControls);
  model.on("change:section", applySection);
  model.on("change:geometry", applyGeometry);
  model.on("change:meta", applyGeometry);
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
    disposeContent();
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

//: computed properties the inlined copy leaves out: `all` would reset the rest,
//: `d` belongs to SVG paths, and `content` would revive pseudo-elements (the
//: list html2canvas uses, so the capture matches what it produced)
const UNCOPIED_PROPERTIES = new Set(["all", "d", "content"]);

/**
 * Copies `node`'s subtree with each element's computed style written inline.
 *
 * An SVG `foreignObject` image renders in a document of its own, where neither
 * the widget's shadow-scoped rules nor anything inherited from the host page
 * applies, so every element carries its full computed style. Reading it from
 * the live shell needs no stylesheet copy and no second layout. Each `<canvas>`
 * becomes an `<img>` of its current pixels: the three.js canvas keeps its
 * drawing buffer (see `Scene3D`) and the section canvas is a 2D surface, so
 * both `toDataURL` cleanly; a tainted one is left as an empty canvas. Form
 * controls keep their live state (see `copyFormState`).
 */
function inlineStyledClone(node: Node): Node | null {
  if (node.nodeType === Node.TEXT_NODE) return document.createTextNode((node as Text).data);
  if (node.nodeType !== Node.ELEMENT_NODE) return node.cloneNode(false);
  const element = node as Element;
  if (element instanceof HTMLScriptElement || element instanceof HTMLStyleElement) return null;
  if (!(element instanceof HTMLElement || element instanceof SVGElement)) {
    return element.cloneNode(false);
  }

  let clone: Element = element.cloneNode(false) as Element;
  if (element instanceof HTMLCanvasElement) {
    try {
      const image = document.createElement("img");
      image.src = element.toDataURL("image/png");
      clone = image;
    } catch {
      // tainted canvas: keep the empty clone rather than abort the capture
    }
  }
  const textValue = copyFormState(element, clone);

  const style = getComputedStyle(element);
  const declarations: string[] = [];
  for (let i = style.length - 1; i >= 0; i--) {
    const property = style.item(i);
    // custom properties (marimo's design tokens alone are hundreds per
    // element) only feed `var()`, which the computed values already resolve
    if (property.startsWith("--") || UNCOPIED_PROPERTIES.has(property)) continue;
    declarations.push(`${property}:${style.getPropertyValue(property)}`);
  }
  clone.setAttribute("style", declarations.join(";"));

  if (textValue !== null) {
    clone.textContent = textValue;
    return clone;
  }
  for (let child = element.firstChild; child; child = child.nextSibling) {
    const copy = inlineStyledClone(child);
    if (copy) clone.appendChild(copy);
  }
  return clone;
}

/**
 * Writes a form control's live state into `clone`'s attributes.
 *
 * The image is drawn from serialized markup, which carries attributes only. A
 * control's current state is a property (`value`, `checked`, `selected`) that
 * the attribute merely seeds, so without this a checked box would draw
 * unchecked and a slider or swatch at its default. Returns a `<textarea>`'s
 * value, which is its text content rather than an attribute, else null.
 */
function copyFormState(element: Element, clone: Element): string | null {
  if (element instanceof HTMLInputElement) {
    if (element.type === "checkbox" || element.type === "radio") {
      clone.toggleAttribute("checked", element.checked);
    } else if (element.type !== "file") {
      clone.setAttribute("value", element.value);
    }
  } else if (element instanceof HTMLOptionElement) {
    clone.toggleAttribute("selected", element.selected);
  } else if (element instanceof HTMLTextAreaElement) {
    return element.value;
  }
  return null;
}

/** Loads a serialized SVG as an image, ready to draw. */
function loadSvg(svg: SVGSVGElement): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src =
      "data:image/svg+xml;charset=utf-8," +
      encodeURIComponent(new XMLSerializer().serializeToString(svg));
  });
}

/**
 * Rasterizes the whole widget shell into one PNG data URL (spec D14).
 *
 * The shell is copied with its computed styles inline (`inlineStyledClone`),
 * wrapped in an SVG `foreignObject` and drawn into a canvas, so the browser's
 * own renderer paints it, colors and all. This is the pipeline html2canvas
 * runs with `foreignObjectRendering`, and the geometry below (canvas size, the
 * `foreignObject` offset, the scale) is its, so the output is unchanged. Doing
 * it here keeps the walk to the shell: html2canvas clones and style-copies the
 * whole host document into an iframe first, which in marimo is seconds of work
 * for a shell that takes milliseconds.
 *
 * If the render fails, the viewport image is returned instead, so the button
 * always produces something.
 */
async function captureShell(shell: HTMLElement, fallbackPng: string): Promise<string> {
  try {
    const rect = shell.getBoundingClientRect();
    const width = Math.ceil(rect.width);
    const height = Math.ceil(rect.height);
    const scale = window.devicePixelRatio || 1;

    const xmlns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(xmlns, "svg");
    svg.setAttribute("width", String(width * scale));
    svg.setAttribute("height", String(height * scale));
    const foreignObject = document.createElementNS(xmlns, "foreignObject");
    foreignObject.setAttribute("width", "100%");
    foreignObject.setAttribute("height", "100%");
    foreignObject.setAttribute("x", String(scale));
    foreignObject.setAttribute("y", String(scale));
    foreignObject.setAttribute("externalResourcesRequired", "true");
    foreignObject.appendChild(inlineStyledClone(shell)!);
    svg.appendChild(foreignObject);
    const image = await loadSvg(svg);

    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(width * scale);
    canvas.height = Math.floor(height * scale);
    const context = canvas.getContext("2d")!;
    context.scale(scale, scale);
    context.drawImage(image, 0, 0);
    return canvas.toDataURL("image/png");
  } catch (error) {
    console.warn("whole-widget capture failed; saving the viewport instead", error);
    return fallbackPng;
  }
}

function currentState(model: AnyModel): StatePayload | undefined {
  return model.get("state") as StatePayload | undefined;
}

// The field colorbar's three parts (spec R15): a title naming the detector,
// component and transform; the ramp itself; and the plane's own value range.
function colorbarTitle(text: string): HTMLElement {
  const element = document.createElement("div");
  element.className = "gfx-colorbar-title";
  element.textContent = text;
  return element;
}

function colorbarRamp(gradient: string): HTMLElement {
  const element = document.createElement("div");
  element.className = "gfx-colorbar-ramp";
  // a checkerboard shows through the transparent middle, so the alpha reads
  element.style.backgroundImage = `linear-gradient(to right, ${gradient}), ${CHECKER}`;
  return element;
}

function colorbarScale(lo: number, hi: number): HTMLElement {
  const element = document.createElement("div");
  element.className = "gfx-colorbar-scale";
  const low = document.createElement("span");
  low.textContent = formatValue(lo);
  const high = document.createElement("span");
  high.textContent = formatValue(hi);
  element.append(low, high);
  return element;
}

/** A compact number for the colorbar's ends (spec R15). */
function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(2);
  return value.toPrecision(3);
}

//: a CSS checkerboard, so a transparent ramp end reads as transparent
const CHECKER =
  "repeating-conic-gradient(#2a2a30 0% 25%, #3a3a42 0% 50%) 50% / 12px 12px";

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
