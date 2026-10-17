# Speed up the whole-widget screenshot in the gdsfactoryx viewer

## Where to work
- Repo: `gdsfactoryx`, worktree `/orcd/home/002/floresh2/file_system/git/gdsfactoryx/.claude/worktrees/viewer-m4`, branch `worktree-viewer-m4`. Run everything from that directory.
- Frontend entry point: `js/viewer/index.ts`. Build: `cd js && ./node_modules/.bin/tsc --noEmit && npm run build` (esbuild → `gdsfactory/viewer/static/viewer.js`).
- Python env for tooling: conda env `betterfdtd` (`/home/floresh2/.conda/envs/betterfdtd/bin/python`), which also has Playwright + cached Chromium.

## What the widget is
A `gdsfactoryx` viewer is an `anywidget.AnyWidget` (three.js 3D view + a 2D canvas cross-section) rendered inside marimo. The Python kernel is on a compute node; the browser talks to it over the anywidget websocket (`model.send(...)` down, synced traitlets up). A camera/save button captures a PNG.

## The problem
The save button has two scopes (a settings toggle):
- **Viewport-only** — `scene.toPng()` (three.js `WebGLRenderer.domElement.toDataURL`, `js/viewer/scene.ts:405`) or `section.toPng()` (2D canvas, `js/viewer/section.ts:504`).
- **Whole widget** — rasterizes the entire widget shell (menubar + settings + both viewports + toasts) into one PNG via `captureShell` (`js/viewer/index.ts:735-802`).

Measured server-side save latency (click → websocket → kernel write → path echoed back), two independent runs, Chromium, 1500×1000 viewport:
- **Viewport-only: ~33 ms** (range 31–35 ms) per image.
- **Whole widget: ~3.7–4.5 s** (range 3.6–4.7 s) per image — **~110–135× slower**.

The cost is almost entirely client-side rasterization: a browser-target whole-widget capture (no kernel round-trip) measured 4.27 s median, essentially the same as the server total, so transport + kernel write are only a few ms. **Goal: cut the whole-widget capture time substantially (target: well under 1 s) without changing the captured image's content or quality.**

## How `captureShell` works today (and why it's slow)
It clones the shadow-DOM shell into the light DOM and hands the clone to `html2canvas@1.4.1`, because html2canvas cannot clone a shadow root. The steps, all load-bearing:
1. `shellStyleText(shell)` (index.ts:690) reads the shadow root's 13 `adoptedStyleSheets` + `<style>` nodes and keeps only rules containing `gfx-`, re-inserted as a `<style>` in the clone (the clone is otherwise unstyled).
2. Each `<canvas>` in the clone is replaced by an `<img>` of the live canvas's `toDataURL("image/png")` (three.js uses `preserveDrawingBuffer: true`, `scene.ts:182`; the 2D canvas is a plain surface).
3. The clone goes in a `position:fixed;left:0;top:0;width:<rect.width>px;z-index:-1` holder. It must be laid out on screen: `foreignObjectRendering: true` hands rasterization to the browser, which only paints elements it actually lays out — an off-screen holder captures 0 pixels.
4. `await` rAF×2 + all `<img>` decode, then `html2canvas(clone, { backgroundColor:null, useCORS:true, foreignObjectRendering:true, onclone })`. The `onclone` clears `html`/`body` backgrounds because html2canvas 1.4.1's color parser throws on the host's modern color functions (`color()`, `oklch()`, `color-mix()`, `light-dark()`).
5. Fallback to the viewport PNG if html2canvas throws.

Suspected hot spots to investigate (verify, don't assume): html2canvas's own DOM walk/clone and its per-element style parsing even in `foreignObjectRendering` mode; the full-resolution re-raster of both canvases (three.js canvas is `devicePixelRatio`-scaled, `scene.ts:183`) inside a large `foreignObject`; PNG encode of the final canvas; the two forced rAFs + image-decode waits.

## Directions worth evaluating
- Replace html2canvas with a lighter capture path: `html-to-image` (`toCanvas`/`toPng`, uses native SVG `foreignObject` with no DOM walk), or hand-roll an SVG-`foreignObject` → `Image` → canvas render (the widget's CSS is already extractable via `shellStyleText`; the canvases are already frozen to data-URL images). Compare quality and time.
- Skip the DOM clone entirely: compose the output directly on a canvas — draw the two viewport images at their measured positions plus the menubar/settings chrome drawn with Canvas2D or overlaid SVG.
- Reduce work: raster at CSS pixels rather than `devicePixelRatio`, cap the three.js render target size for capture, or cache the menubar/settings layer between saves (only the viewports change).
- Check whether the forced rAF×2 + image-load waits can be shortened (e.g. `img.decode()`).

## Hard constraints (do not regress)
- **No screenshots of the machine.** Verify rendered state by reading code and driving the runtime: Playwright in the `betterfdtd` env, pixel checks via `getImageData`/`evaluate` (this respects the no-screenshot rule), and unit tests. If a screenshot is truly needed, ask first.
- **Same output.** The whole-widget PNG must still contain the full shell (menubar, controls, both viewports) with correct dark styling and the live canvas pixels — the current output is verified good; a faster path that changes what's captured is not a win.
- Keep the viewport-only path at its current ~33 ms and behavior.
- The host is a GPU node (L40S); the marimo server runs headless on port 2731 from `examples/viewer_m4.py` (restart with `/home/floresh2/.conda/envs/betterfdtd/bin/marimo edit examples/viewer_m4.py --headless --no-token --port 2731`). Driving the live widget: open the notebook in Playwright, "Take over session" if prompted, `page.wait_for_selector(".gfx-shell")`, then pierce the shadow root via `document.querySelector('marimo-anywidget').shadowRoot`.
- When re-running notebook cells after a JS rebuild, the kernel caches `Viewer._esm`; `importlib.reload(gdsfactory.viewer.widget)` is required to pick up a new bundle.
- Viewer tests: `PYTHONPATH=.:tests` with the `betterfdtd` python. Existing suites in `tests/test_viewer_*.py` must stay green.

## Deliverable
A change to the capture path that measurably reduces whole-widget save time (report median over ≥5 samples before/after, same harness), with the captured image unchanged. Include the timing method you used so the numbers are reproducible.
