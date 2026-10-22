// The 2D cross-section viewport (spec R4, D4).
//
// The kernel slices the analytic solids and sends vector polygons, so the
// section is sharp at any zoom and always agrees with the 3D view. Here the
// rings are filled with Canvas2D's even-odd rule (holes come for free) and
// zoom/pan are a screen transform applied on redraw, so nothing is resampled.
import { componentData, parseColor, rampColor, shadeRange, transformValues } from "./fieldramp";
import { REGION_COLOR, formatTick, niceStep, tickValues } from "./ticks";
import type {
  FieldPayload,
  FieldSettings,
  GlyphPayload,
  MetaPayload,
  PortMeta,
  Ring,
  SectionPayload,
  StatePayload,
} from "./wire";

const BG_COLOR = "#15151a";
const EDGE_COLOR = "#000000";
const PORT_COLOR = "#ffcc33";
const SELECTED_COLOR = "#66ff66";
//: sources are warm, detectors cool, matching the 3D planes (spec R12)
const SOURCE_COLOR = "#ff8844";
const DETECTOR_COLOR = "#44bbff";
//: tick label font size, in screen pixels
const TICK_LABEL_PX = 11;
const TICK_PX = 5 * (2 / 3);
//: gap between a tick mark and its label, in screen pixels: the original 4 px
//: relation, pulled 3x closer
const LABEL_GAP = (4 * (2 / 3)) / 3;
//: the axis name is drawn larger than the tick values it sits among
const AXIS_NAME_PX = 16;

//: An outline is not drawn beyond this many 1/e distances from its plane
//: (the gaussian has decayed to ~1% there, so it reads as gone, spec R4).
const PLANE_FADE_CUTOFF = 3;
//: A box outline (the plane lies in the port's face) is filled at this fraction
//: of the outline's own alpha, so the face reads as a translucent plate.
const PORT_FILL_ALPHA = 0.25;
//: Half-extent (µm) of the marker a source/detector glyph shows when it has no
//: plane of its own (a point dipole), and the tolerance within which a
//: parallel glyph plane is treated as coincident with the section (spec D11).
const GLYPH_CUT_HALF = 0.5;

interface View {
  //: screen pixels per µm
  scale: number;
  //: pan offset in screen pixels
  dx: number;
  dy: number;
}

/** The draw parameters that put a field raster upright over its plane (spec R15).
 *
 * The raster is row-major in (v, u), so its row 0 is the plane's low-v edge and
 * must land at ``y1`` (the low-v screen edge). ``drawImage`` does not flip on a
 * negative height, so the flip is a transform: translate to the low-v corner and
 * negate y. Pure and exported so the raster-to-screen mapping can be tested
 * without a canvas.
 */
export function fieldBlit(
  x0: number,
  y1: number,
  x1: number,
  y0: number,
): { translate: [number, number]; scale: [number, number]; dest: [number, number, number, number] } {
  return { translate: [x0, y1], scale: [1, -1], dest: [0, 0, x1 - x0, y1 - y0] };
}

export class Section2D {
  readonly root: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly readout: HTMLElement;
  private payload: SectionPayload | null = null;
  private state: StatePayload | null = null;
  //: the last result's per-detector fields (spec R15); empty until a run lands
  private fields: Record<string, FieldPayload> = {};
  //: frames shown in place of a detector's stored `field_freq` while it plays
  //: or is exported as a video, so each tick re-shades without the kernel
  private readonly frameOverrides = new Map<string, number>();
  //: a scratch canvas for rasterizing one field before it is drawn (spec R15)
  private scratch: HTMLCanvasElement | null = null;
  private exaggeration = 1;
  //: the box center (spec C5) of the section's two axes, for labelling local
  //: coordinates in device (stack) coordinates, like the slider
  private originU = 0;
  private originV = 0;
  private view: View = { scale: 1, dx: 0, dy: 0 };
  //: set once the user zooms or pans; until then the view re-fits on every
  //: resize and payload change (the container starts hidden at zero size)
  private userAdjusted = false;
  private readonly resizeObserver: ResizeObserver;

  constructor(
    private meta: MetaPayload,
    private readonly onPortSelect: (name: string | null) => void,
  ) {
    this.root = document.createElement("div");
    this.root.className = "gfx-cross-section";
    this.canvas = document.createElement("canvas");
    this.canvas.className = "gfx-section-canvas";
    this.readout = document.createElement("div");
    this.readout.className = "gfx-readout";
    this.root.append(this.canvas, this.readout);
    const context = this.canvas.getContext("2d");
    if (!context) throw new Error("2D canvas context unavailable");
    this.context = context;

    this.canvas.addEventListener("mousemove", (event) => this.onMove(event));
    this.canvas.addEventListener("mouseleave", () => (this.readout.textContent = ""));
    this.canvas.addEventListener("click", (event) => this.onClick(event));
    this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    this.canvas.addEventListener("mousedown", (event) => this.onPanStart(event));
    this.resizeObserver = new ResizeObserver(() => {
      if (!this.userAdjusted) this.fit();
      else this.draw();
    });
    this.resizeObserver.observe(this.root);
  }

  setPayload(payload: SectionPayload): void {
    this.payload = payload;
    // the section's (u, v) axes are a per-plane pair of stack axes (spec R4),
    // so pick the box center's matching components for labelling
    const [ox, oy, oz] = payload.origin;
    const origins: Record<SectionPayload["plane"], [number, number]> = {
      xy: [ox, oy],
      xz: [ox, oz],
      yz: [oy, oz],
    };
    [this.originU, this.originV] = origins[payload.plane];
    this.fit(); // refit on a new plane or position, unless the user has zoomed
    this.draw();
  }

  setState(state: StatePayload): void {
    this.state = state;
    this.draw();
  }

  /** The gaussian 1/e distance (µm) for the plane fade (spec R4).
   *
   * The state's fraction applies to the box size along the section's normal,
   * taken from the state's `plane_bounds` (the slider's own range).
   */
  private fade(): number {
    const state = this.state;
    if (!state) return 0;
    const size = state.plane_bounds[1] - state.plane_bounds[0];
    return state.plane_fade_frac * size;
  }

  /** The last result's per-detector fields (spec R15); re-draws the overlay. */
  setFields(fields: Record<string, FieldPayload>): void {
    this.fields = fields ?? {};
    this.draw();
  }

  /** Shows `frame` of a detector instead of its stored one (null: stop); re-draws. */
  setFrame(id: string, frame: number | null): void {
    if (frame === null) this.frameOverrides.delete(id);
    else this.frameOverrides.set(id, frame);
    this.draw();
  }

  /** Draws now and returns the canvas, so a video frame captures this state. */
  captureCanvas(): HTMLCanvasElement {
    this.draw();
    return this.canvas;
  }

  /**
   * Swaps the level/port metadata (spec R13).
   *
   * A boundary-offset edit re-boxes the device, which changes each level's
   * clipped extent and can move a port glyph, so the section must draw from the
   * new metadata. The payload that follows re-fits the view.
   */
  setMeta(meta: MetaPayload): void {
    this.meta = meta;
  }

  setExaggeration(factor: number): void {
    this.exaggeration = factor;
    if (this.userAdjusted) this.draw();
    else this.fit();
  }

  /** The factor actually applied to the v axis.
   *
   * The z multiplier is vertical only: it stretches v when v is z (xz, yz) and
   * does nothing on xy, where v is y (spec R4).
   */
  private get vExaggeration(): number {
    return this.payload?.axis_v === "z" ? this.exaggeration : 1;
  }

  /** Frames the section rectangle in the canvas with equal aspect (spec R4). */
  fit(): void {
    const payload = this.payload;
    if (!payload) return;
    const { width, height } = this.size();
    const [[u0, u1], [v0, v1]] = payload.bounds;
    const uSpan = Math.max(u1 - u0, 1e-6);
    const vSpan = Math.max((v1 - v0) * this.vExaggeration, 1e-6);
    const margin = 24;
    const scale = Math.min((width - 2 * margin) / uSpan, (height - 2 * margin) / vSpan);
    this.view = { scale, dx: 0, dy: 0 };
    this.userAdjusted = false;
    this.draw();
  }

  dispose(): void {
    this.resizeObserver.disconnect();
  }

  // ------------------------------------------------------------------ drawing

  private size(): { width: number; height: number } {
    return {
      width: Math.max(this.root.clientWidth, 1),
      height: Math.max(this.root.clientHeight, 1),
    };
  }

  /** Local (u, v) -> screen pixels. v is flipped (v up) and exaggerated. */
  private project(u: number, v: number): [number, number] {
    const { width, height } = this.size();
    const cx = width / 2 + this.view.dx;
    const cy = height / 2 + this.view.dy;
    return [cx + u * this.view.scale, cy - v * this.view.scale * this.vExaggeration];
  }

  private unproject(px: number, py: number): [number, number] {
    const { width, height } = this.size();
    const cx = width / 2 + this.view.dx;
    const cy = height / 2 + this.view.dy;
    return [(px - cx) / this.view.scale, (cy - py) / (this.view.scale * this.vExaggeration)];
  }

  private path(rings: Ring[]): Path2D {
    const path = new Path2D();
    for (const ring of rings) {
      if (ring.length < 3) continue;
      const [x0, y0] = this.project(ring[0][0], ring[0][1]);
      path.moveTo(x0, y0);
      for (let i = 1; i < ring.length; i++) {
        const [x, y] = this.project(ring[i][0], ring[i][1]);
        path.lineTo(x, y);
      }
      path.closePath();
    }
    return path;
  }

  private draw(): void {
    const { width, height } = this.size();
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== width * dpr || this.canvas.height !== height * dpr) {
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
    const ctx = this.context;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, width, height);

    const payload = this.payload;
    const state = this.state;
    if (!payload || !state) return;

    // levels, in mesh_order so a higher-priority level draws last (spec Q13)
    const ordered = [...this.meta.levels].sort((a, b) => a.mesh_order - b.mesh_order);
    for (const level of ordered) {
      const style = state.styles[level.name];
      const rings = payload.rings[level.name];
      if (!style || !style.visible || !rings || rings.length === 0) continue;
      ctx.globalAlpha = style.opacity;
      const path = this.path(rings);
      ctx.fillStyle = style.color;
      ctx.fill(path, "evenodd");
      ctx.globalAlpha = 1;
      ctx.strokeStyle = EDGE_COLOR;
      ctx.lineWidth = 1;
      ctx.stroke(path);
    }

    this.drawPorts(ctx, payload);
    this.drawFields(ctx, payload);
    this.drawRegion(ctx, payload);
    this.drawGlyphs(ctx, payload);
    this.drawTicks(ctx, payload);
  }

  /**
   * Shades a detector's plane with its field, where the section is parallel to
   * it (spec R15, D16).
   *
   * The field is a raster on the detector's own (u, v) rectangle, drawn under
   * the outline and ticks. A section perpendicular to the detector's plane
   * shows nothing (its field is a plane, and there is nothing off it). The
   * overlay fades with the same gaussian as the ports (spec R4), so a field
   * fades in as its plane is approached.
   */
  private drawFields(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const state = this.state;
    if (!state) return;
    const [ox, oy, oz] = payload.origin;
    const origin = [ox, oy, oz];
    const normalStack = { xy: 2, xz: 1, yz: 0 }[payload.plane];
    const fade = this.fade();
    for (const glyph of state.glyphs ?? []) {
      if (!glyph.visible || glyph.kind !== "detector") continue;
      if (glyph.u_half <= 0 || glyph.v_half <= 0) continue;
      // only a detector whose plane is parallel to the section can be shaded
      if (glyph.axis !== normalStack) continue;
      const settings = state.field_settings?.[glyph.id];
      const field = this.fields[glyph.id];
      if (!settings || !settings.field_visible || !field || !settings.field_component) continue;
      const frame = this.frameOverrides.get(glyph.id) ?? settings.field_freq ?? 0;
      const data = componentData(field, settings.field_component, frame);
      if (!data) continue;
      // the gaussian the ports use, from the section plane to the detector's
      // plane (both in the local frame) (spec D16, R4)
      const distance = glyph.center[glyph.axis] - (payload.position - origin[normalStack]);
      if (Math.abs(distance) > PLANE_FADE_CUTOFF * fade) continue;
      const alpha = Math.exp(-0.5 * (distance / fade) ** 2);
      if (alpha < 0.01) continue;
      const values = transformValues(data.real, data.imag, settings.field_transform);
      const range = shadeRange(field, settings, state.simulation?.log_range ?? 6);
      this.blitField(ctx, glyph, field, values, range, settings, alpha);
    }
  }

  /** Paints a detector's raster field over its (u, v) rectangle (spec R15). */
  private blitField(
    ctx: CanvasRenderingContext2D,
    glyph: GlyphPayload,
    field: FieldPayload,
    values: Float32Array,
    range: [number, number],
    settings: FieldSettings,
    alpha: number,
  ): void {
    const [rows, cols] = field.grid;
    const image = this.fieldCanvas(cols, rows);
    const pixels = image.getContext("2d")!.createImageData(cols, rows);
    const span = range[1] - range[0];
    const low = parseColor(settings.field_color_low);
    const high = parseColor(settings.field_color_high);
    for (let i = 0; i < values.length; i += 1) {
      const t = span > 0 ? (values[i] - range[0]) / span : 0;
      const [r, g, b, a] = rampColor(t, settings.field_transform, low, high);
      pixels.data[i * 4] = r;
      pixels.data[i * 4 + 1] = g;
      pixels.data[i * 4 + 2] = b;
      pixels.data[i * 4 + 3] = Math.round(a * 255);
    }
    image.getContext("2d")!.putImageData(pixels, 0, 0);

    // the rectangle's two corners in the section's local (u, v). The raster is
    // row-major with rows as v and columns as u (spec R15), so raster row 0 is
    // the plane's low-v edge and must land at ``y1`` (the low-v screen edge).
    // ``drawImage`` does NOT flip on a negative height (Chromium normalises the
    // rectangle), so the flip is done with a transform instead: translate to the
    // low-v corner and negate y, then draw with a positive height.
    const [au, av] = { xy: [0, 1], xz: [0, 2], yz: [1, 2] }[this.payload!.plane];
    // a sample is drawn as a cell centred on where it was recorded, which is
    // not always the cell's centre (``field.shift``, in cells)
    const [shiftU, shiftV] = field.shift ?? [0, 0];
    const cu = glyph.center[au] + (shiftU * 2 * glyph.u_half) / cols;
    const cv = glyph.center[av] + (shiftV * 2 * glyph.v_half) / rows;
    const [x0, y1] = this.project(cu - glyph.u_half, cv - glyph.v_half);
    const [x1, y0] = this.project(cu + glyph.u_half, cv + glyph.v_half);
    const blit = fieldBlit(x0, y1, x1, y0);
    ctx.save();
    ctx.globalAlpha = alpha * settings.field_opacity;
    // nearest-neighbour, so each recorded sample stays a visible block
    ctx.imageSmoothingEnabled = false;
    ctx.translate(blit.translate[0], blit.translate[1]);
    ctx.scale(blit.scale[0], blit.scale[1]);
    ctx.drawImage(image, blit.dest[0], blit.dest[1], blit.dest[2], blit.dest[3]);
    ctx.restore();
  }

  private fieldCanvas(cols: number, rows: number): HTMLCanvasElement {
    if (!this.scratch || this.scratch.width !== cols || this.scratch.height !== rows) {
      this.scratch = document.createElement("canvas");
      this.scratch.width = cols;
      this.scratch.height = rows;
    }
    return this.scratch;
  }

  /**
   * Outlines each source/detector where the section plane cuts it (spec R12).
   *
   * A glyph is an axis-aligned rectangle on a port frame (spec D11). The
   * section shows its intersection with the section plane: a segment when the
   * two planes cross, or the whole rectangle when they are parallel and
   * coincide. A dipole (zero extents) draws a small marker instead.
   */
  private drawGlyphs(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const state = this.state;
    if (!state) return;
    const [ox, oy, oz] = payload.origin;
    const origin = [ox, oy, oz];
    const normalStack = { xy: 2, xz: 1, yz: 0 }[payload.plane];
    const fade = this.fade();
    for (const glyph of state.glyphs ?? []) {
      if (!glyph.visible) continue;
      const cut = this.glyphCut(glyph, payload);
      if (!cut) continue;
      // any glyph fades with the same gaussian as the ports (spec R4): a
      // parallel plane fades in as its plane is approached, and a plane seen
      // edge-on (a crossing glyph) is solid while the section cuts it and fades
      // out over the last `fade` µm before the glyph's edge along the section
      // normal — neither pops in or out
      let alpha: number;
      if (glyph.axis === normalStack) {
        const distance = glyph.center[normalStack] - (payload.position - origin[normalStack]);
        alpha = Math.exp(-0.5 * (distance / fade) ** 2);
      } else {
        const d = Math.abs(glyph.center[normalStack] - (payload.position - origin[normalStack]));
        const e = this.glyphExtent(glyph, normalStack);
        // inside the extent minus one fade distance the line is solid; the last
        // fade distance is the gaussian's outer tail (0.01 at the edge, like the
        // ports' 3-cutoff); a patch narrower than the fade fades across it all
        const x = fade < e ? Math.max(0, (d - (e - fade)) / fade) : d / Math.max(e, 1e-9);
        alpha = x >= 1 ? 0 : Math.exp(-0.5 * (3 * x) ** 2);
      }
      if (alpha < 0.01) continue;
      const selected =
        (state.selection?.kind === "source" || state.selection?.kind === "detector") &&
        state.selection.name === glyph.id;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = selected ? SELECTED_COLOR : glyph.kind === "source" ? SOURCE_COLOR : DETECTOR_COLOR;
      ctx.lineWidth = 2;
      if (cut.kind === "segment") {
        const [x0, y0] = this.project(cut.a[0], cut.a[1]);
        const [x1, y1] = this.project(cut.b[0], cut.b[1]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        if (glyph.u_half === 0 && glyph.v_half === 0) {
          // a dipole has no plane; mark the point at the cut's midpoint
          ctx.beginPath();
          ctx.arc((x0 + x1) / 2, (y0 + y1) / 2, 3, 0, 2 * Math.PI);
          ctx.stroke();
        }
      } else {
        const [x0, y0] = this.project(cut.u0, cut.v0);
        const [x1, y1] = this.project(cut.u1, cut.v1);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
    ctx.globalAlpha = 1;
  }

  /**
   * A source/detector's cut in the section's local (u, v), or null.
   *
   * The glyph's center is already local (spec C5) and its `axis` is the plane
   * normal; the absolute plane position is brought into the local frame, as
   * :meth:`portCut` does for a port. The two planes either cross in a segment
   * or are parallel (then shown only where they coincide).
   */
  /** A glyph's half-extent along a stack axis (spec D11).
   *
   * The glyph's own normal is not a spanning axis; a dipole (no plane) has a
   * marker's half-extent on every axis.
   */
  private glyphExtent(glyph: GlyphPayload, axis: number): number {
    if (glyph.u_half === 0 && glyph.v_half === 0) return GLYPH_CUT_HALF;
    const spans = [0, 1, 2].filter((a) => a !== glyph.axis);
    return axis === spans[0] ? glyph.u_half : glyph.v_half;
  }

  private glyphCut(
    glyph: GlyphPayload,
    payload: SectionPayload,
  ):
    | { kind: "segment"; a: [number, number]; b: [number, number] }
    | { kind: "rect"; u0: number; u1: number; v0: number; v1: number }
    | null {
    const half = GLYPH_CUT_HALF;
    const [ox, oy, oz] = payload.origin;
    const origin = [ox, oy, oz];
    const center = glyph.center;
    // the section's two in-plane stack axes and its normal
    const [au, av] = { xy: [0, 1], xz: [0, 2], yz: [1, 2] }[payload.plane];
    const normalStack = { xy: 2, xz: 1, yz: 0 }[payload.plane];
    // the glyph's two spanning axes, ascending (matching the payload's spans)
    const spans = [0, 1, 2].filter((axis) => axis !== glyph.axis);
    const extent = (axis: number): number => {
      if (glyph.u_half === 0 && glyph.v_half === 0) return half;
      return axis === spans[0] ? glyph.u_half : glyph.v_half;
    };

    if (glyph.axis === normalStack) {
      // parallel planes: show the rectangle only where the two coincide
      if (
        Math.abs(center[glyph.axis] - (payload.position - origin[normalStack])) >
        PLANE_FADE_CUTOFF * this.fade()
      ) {
        return null;
      }
      return {
        kind: "rect",
        u0: center[au] - extent(au),
        u1: center[au] + extent(au),
        v0: center[av] - extent(av),
        v1: center[av] + extent(av),
      };
    }

    // the planes cross: the glyph normal is one of the section's in-plane axes
    // and fixes that coordinate; the segment runs along the other in-plane axis,
    // which is the glyph spanning axis that lies in the section. The glyph also
    // spans the section's normal axis, so it is cut only while the section's
    // position lies within that extent — otherwise the glyph is off the plane
    // and must not be drawn (a mode plane source does not fill all z, spec R12).
    if (Math.abs(center[normalStack] - (payload.position - origin[normalStack])) > extent(normalStack)) {
      return null;
    }
    const fixed = glyph.axis;
    const vary = fixed === au ? av : au;
    const reach = extent(vary);
    if (fixed === au) {
      return { kind: "segment", a: [center[au], center[av] - reach], b: [center[au], center[av] + reach] };
    }
    return { kind: "segment", a: [center[au] - reach, center[av]], b: [center[au] + reach, center[av]] };
  }

  private drawPorts(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const state = this.state;
    if (!state) return;
    const fade = this.fade();
    for (const port of this.meta.ports) {
      if (!(state.port_visible[port.name] ?? true)) continue;
      const distance = this.portDistance(port, payload);
      const alpha = Math.exp(-0.5 * (distance / fade) ** 2);
      if (alpha < 0.01) continue;
      const outline = this.portOutline(port, payload);
      if (!outline) continue;
      const selected = state.selection?.kind === "port" && state.selection.name === port.name;
      const color = selected ? SELECTED_COLOR : PORT_COLOR;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      const path = new Path2D();
      const [x0, y0] = this.project(outline[0][0], outline[0][1]);
      path.moveTo(x0, y0);
      for (let i = 1; i < outline.length; i++) {
        const [x, y] = this.project(outline[i][0], outline[i][1]);
        path.lineTo(x, y);
      }
      path.closePath();
      // a box outline (the plane lies in the port's face) gets a translucent
      // fill so the face reads as a plate; the bar case stays an open line
      if (outline.length === 4) {
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha * PORT_FILL_ALPHA;
        ctx.fill(path);
        ctx.globalAlpha = alpha;
      }
      ctx.stroke(path);
      ctx.globalAlpha = 1;
    }
  }

  /** The distance from the section plane to a port's plane, along its normal (µm).
   *
   * Port metadata is local, like the rings (spec C5), while the plane position
   * is absolute (stack) coordinates, so the plane is brought into the local
   * frame before the distance is measured.
   */
  private portDistance(port: PortMeta, payload: SectionPayload): number {
    const [ox, oy, oz] = payload.origin;
    if (payload.plane === "xy") return port.z - (payload.position - oz);
    if (payload.plane === "xz") return port.y - (payload.position - oy);
    return port.x - (payload.position - ox);
  }

  /** A port's outline in local (u, v), or null when the plane misses it.
   *
   * A port is a flat face: it spans its width, perpendicular to its orientation
   * (spec R3), and it lies in the plane of its own orientation axis. Where the
   * section plane cuts across that face (the port's orientation is not the
   * plane's normal) the outline is the width bar, as before. Where the plane's
   * normal *is* the port's orientation the plane lies in the port's own plane,
   * so the cut is the whole face and the outline is a box: the width across the
   * port and the layer's thickness along the orientation (spec R4).
   */
  private portOutline(port: PortMeta, payload: SectionPayload): [number, number][] | null {
    // a simulation-volume plane port (spec R12) has no orientation and spans the
    // whole volume; its width bar in 2D would be meaningless, so it is drawn
    // only in 3D (as the plane's rectangle)
    if (port.plane !== null) return null;
    const half = PLANE_FADE_CUTOFF * this.fade();
    const [ox, oy, oz] = payload.origin;
    const angle = ((port.orientation ?? 0) * Math.PI) / 180;
    // the port's normal (orientation) as a stack-axis unit vector; a port with no
    // orientation is treated as +x, matching the 3D glyph (spec R3)
    const nx = Math.cos(angle);
    const ny = Math.sin(angle);
    // the port face spans its width across the orientation (transverse) and the
    // layer's thickness vertically; both half-extents, so the face box is
    // width x height (spec R4)
    const hw = port.width / 2;
    const hh = port.height / 2;
    if (payload.plane === "xy") {
      // the plane's normal is z; a port is never z-facing, so it always cuts the
      // face and the outline is the width bar
      if (Math.abs(port.z - (payload.position - oz)) > half) return null;
      return [
        [port.x - hw * ny, port.y + hw * nx],
        [port.x + hw * ny, port.y - hw * nx],
      ];
    }
    if (payload.plane === "xz") {
      // the plane's normal is y; a y-facing port lies in it and shows its face,
      // so the cut is the face box: width across x, thickness along z
      if (Math.abs(port.y - (payload.position - oy)) > half) return null;
      if (Math.abs(ny) > 1e-9) {
        return [
          [port.x - hw * ny, port.z - hh],
          [port.x + hw * ny, port.z - hh],
          [port.x + hw * ny, port.z + hh],
          [port.x - hw * ny, port.z + hh],
        ];
      }
      return [
        [port.x, port.z - hw],
        [port.x, port.z + hw],
      ];
    }
    // yz: u = y, v = z (z vertical, spec R4); the plane's normal is x, so an
    // x-facing port lies in it and shows its face, so the cut is the face box:
    // width across y, thickness along z
    if (Math.abs(port.x - (payload.position - ox)) > half) return null;
    if (Math.abs(nx) > 1e-9) {
      return [
        [port.y - hw * nx, port.z - hh],
        [port.y + hw * nx, port.z - hh],
        [port.y + hw * nx, port.z + hh],
        [port.y - hw * nx, port.z + hh],
      ];
    }
    return [
      [port.y, port.z - hw],
      [port.y, port.z + hw],
    ];
  }

  /** Outlines the simulation region (the section rectangle, spec R4). */
  private drawRegion(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const [[u0, u1], [v0, v1]] = payload.bounds;
    const [x0, y0] = this.project(u0, v0);
    const [x1, y1] = this.project(u1, v1);
    ctx.strokeStyle = REGION_COLOR;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }

  /** Tick marks and labels for the two section axes (spec R4). */
  private drawTicks(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const [[u0, u1], [v0, v1]] = payload.bounds;
    const step = niceStep(Math.max(u1 - u0, v1 - v0));
    ctx.lineWidth = 1;
    ctx.font = `${TICK_LABEL_PX}px system-ui, sans-serif`;
    // u ticks run along the bottom edge, v ticks up the left edge; both stay
    // the plain outline color (spec R3)
    ctx.strokeStyle = REGION_COLOR;
    ctx.fillStyle = REGION_COLOR;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.beginPath();
    for (const au of tickValues(u0 + this.originU, u1 + this.originU, step)) {
      const [x, y] = this.project(au - this.originU, v0);
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + TICK_PX);
    }
    ctx.stroke();
    // ticks sit at nice device coordinates (local + box center, spec C5)
    for (const au of tickValues(u0 + this.originU, u1 + this.originU, step)) {
      const [x, y] = this.project(au - this.originU, v0);
      ctx.fillText(formatTick(au), x, y + TICK_PX + LABEL_GAP);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.beginPath();
    for (const av of tickValues(v0 + this.originV, v1 + this.originV, step)) {
      const [x, y] = this.project(u0, av - this.originV);
      ctx.moveTo(x, y);
      ctx.lineTo(x - TICK_PX, y);
    }
    ctx.stroke();
    for (const av of tickValues(v0 + this.originV, v1 + this.originV, step)) {
      const [x, y] = this.project(u0, av - this.originV);
      ctx.fillText(formatTick(av), x - TICK_PX - LABEL_GAP, y);
    }
    // name each axis at the middle of its ticked edge, in a larger face and a
    // little further out than the tick labels so the two never collide (a tick
    // often sits at the midpoint too)
    const nameGap = (TICK_PX + LABEL_GAP + AXIS_NAME_PX) * (2 / 3);
    ctx.font = `${AXIS_NAME_PX}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const [ux, uy] = this.project((u0 + u1) / 2, v0);
    ctx.fillText(payload.axis_u, ux, uy + nameGap);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const [vx, vy] = this.project(u0, (v0 + v1) / 2);
    ctx.fillText(payload.axis_v, vx - nameGap, vy);
  }

  /** The current frame as a PNG data URL (spec R4, "save image"). */
  toPng(): string {
    return this.canvas.toDataURL("image/png");
  }

  // ------------------------------------------------------------------ hover

  private onMove(event: MouseEvent): void {
    const payload = this.payload;
    const state = this.state;
    if (!payload || !state) return;
    const rect = this.canvas.getBoundingClientRect();
    const [u, v] = this.unproject(event.clientX - rect.left, event.clientY - rect.top);
    const level = this.levelAt(u, v);
    const name = level ?? (this.inBackground(u, v) ? "background" : "-");
    // the section is in local coordinates; label it in device coordinates (spec C5)
    this.readout.textContent =
      `${payload.axis_u}=${(u + this.originU).toFixed(3)} ` +
      `${payload.axis_v}=${(v + this.originV).toFixed(3)} µm · ${name}`;
  }

  private onClick(event: MouseEvent): void {
    const payload = this.payload;
    const state = this.state;
    if (!payload || !state) return;
    const rect = this.canvas.getBoundingClientRect();
    const [u, v] = this.unproject(event.clientX - rect.left, event.clientY - rect.top);
    for (const port of this.meta.ports) {
      const outline = this.portOutline(port, payload);
      if (!outline) continue;
      const tol = 6 / this.view.scale;
      for (let i = 0; i < outline.length; i++) {
        const a = outline[i];
        const b = outline[(i + 1) % outline.length];
        if (this.nearSegment(u, v, a, b, tol)) {
          this.onPortSelect(port.name);
          return;
        }
      }
    }
    this.onPortSelect(null);
  }

  /** The topmost visible level containing (u, v) by even-odd, or null. */
  private levelAt(u: number, v: number): string | null {
    const payload = this.payload;
    const state = this.state;
    if (!payload || !state) return null;
    const ordered = [...this.meta.levels].sort((a, b) => b.mesh_order - a.mesh_order);
    for (const level of ordered) {
      const style = state.styles[level.name];
      const rings = payload.rings[level.name];
      if (!style?.visible || !rings) continue;
      if (this.containsEvenOdd(u, v, rings)) return level.name;
    }
    return null;
  }

  private inBackground(u: number, v: number): boolean {
    return this.payload ? this.containsEvenOdd(u, v, this.payload.background) : false;
  }

  private containsEvenOdd(u: number, v: number, rings: Ring[]): boolean {
    let inside = false;
    for (const ring of rings) {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        if (yi > v !== yj > v && u < ((xj - xi) * (v - yi)) / (yj - yi) + xi) inside = !inside;
      }
    }
    return inside;
  }

  private nearSegment(u: number, v: number, a: [number, number], b: [number, number], tol: number): boolean {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const lengthSq = dx * dx + dy * dy;
    const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((u - a[0]) * dx + (v - a[1]) * dy) / lengthSq));
    const px = a[0] + t * dx;
    const py = a[1] + t * dy;
    return Math.hypot(u - px, v - py) <= tol;
  }

  // ------------------------------------------------------------------ zoom/pan

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const [u, v] = this.unproject(mx, my);
    const factor = Math.exp(-event.deltaY * 0.001);
    this.view.scale *= factor;
    // keep the point under the cursor fixed
    const [px, py] = this.project(u, v);
    this.view.dx += mx - px;
    this.view.dy += my - py;
    this.userAdjusted = true;
    this.draw();
  }

  private onPanStart(event: MouseEvent): void {
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startDx = this.view.dx;
    const startDy = this.view.dy;
    const move = (moveEvent: MouseEvent): void => {
      this.view.dx = startDx + (moveEvent.clientX - startX);
      this.view.dy = startDy + (moveEvent.clientY - startY);
      this.userAdjusted = true;
      this.draw();
    };
    const up = (): void => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }
}
