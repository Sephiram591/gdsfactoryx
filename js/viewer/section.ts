// The 2D cross-section viewport (spec R4, D4).
//
// The kernel slices the analytic solids and sends vector polygons, so the
// section is sharp at any zoom and always agrees with the 3D view. Here the
// rings are filled with Canvas2D's even-odd rule (holes come for free) and
// zoom/pan are a screen transform applied on redraw, so nothing is resampled.
import type { MetaPayload, PortMeta, Ring, SectionPayload, StatePayload } from "./wire";

const BG_COLOR = "#15151a";
const EDGE_COLOR = "#000000";
const PORT_COLOR = "#ffcc33";
const SELECTED_COLOR = "#66ff66";

//: How far a port's outline reaches either side of the plane it cuts (µm).
const PORT_CUT_HALF = 0.5;

interface View {
  //: screen pixels per µm
  scale: number;
  //: pan offset in screen pixels
  dx: number;
  dy: number;
}

export class Section2D {
  readonly root: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly readout: HTMLElement;
  private payload: SectionPayload | null = null;
  private state: StatePayload | null = null;
  private exaggeration = 1;
  private view: View = { scale: 1, dx: 0, dy: 0 };
  //: set once the user zooms or pans; until then the view re-fits on every
  //: resize and payload change (the container starts hidden at zero size)
  private userAdjusted = false;
  private readonly resizeObserver: ResizeObserver;

  constructor(
    private readonly meta: MetaPayload,
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
    this.fit(); // refit on a new plane or position, unless the user has zoomed
    this.draw();
  }

  setState(state: StatePayload): void {
    this.state = state;
    this.draw();
  }

  setExaggeration(factor: number): void {
    this.exaggeration = factor;
    if (this.userAdjusted) this.draw();
    else this.fit();
  }

  /** Frames the section rectangle in the canvas with equal aspect (spec R4). */
  fit(): void {
    const payload = this.payload;
    if (!payload) return;
    const { width, height } = this.size();
    const [[u0, u1], [v0, v1]] = payload.bounds;
    const uSpan = Math.max(u1 - u0, 1e-6);
    const vSpan = Math.max((v1 - v0) * this.exaggeration, 1e-6);
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
    return [cx + u * this.view.scale, cy - v * this.view.scale * this.exaggeration];
  }

  private unproject(px: number, py: number): [number, number] {
    const { width, height } = this.size();
    const cx = width / 2 + this.view.dx;
    const cy = height / 2 + this.view.dy;
    return [(px - cx) / this.view.scale, (cy - py) / (this.view.scale * this.exaggeration)];
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
    this.drawAxes(ctx, payload);
  }

  private drawPorts(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const state = this.state;
    if (!state) return;
    for (const port of this.meta.ports) {
      if (!(state.port_visible[port.name] ?? true)) continue;
      const cut = this.portCut(port, payload);
      if (!cut) continue;
      const selected = state.selection?.kind === "port" && state.selection.name === port.name;
      ctx.strokeStyle = selected ? SELECTED_COLOR : PORT_COLOR;
      ctx.lineWidth = 2;
      const [a, b] = cut;
      const [x0, y0] = this.project(a[0], a[1]);
      const [x1, y1] = this.project(b[0], b[1]);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
  }

  /** A port's outline where the section plane cuts it, in (u, v), or null. */
  private portCut(port: PortMeta, payload: SectionPayload): [[number, number], [number, number]] | null {
    const half = PORT_CUT_HALF;
    if (payload.plane === "xy") {
      if (Math.abs(port.z - payload.position) > half) return null;
      // the bar runs across the port, perpendicular to its orientation
      const angle = ((port.orientation ?? 0) * Math.PI) / 180;
      const w = port.width / 2;
      return [
        [port.x - w * Math.sin(angle), port.y + w * Math.cos(angle)],
        [port.x + w * Math.sin(angle), port.y - w * Math.cos(angle)],
      ];
    }
    if (payload.plane === "xz") {
      if (Math.abs(port.y - payload.position) > half) return null;
      return [
        [port.x, port.z - port.width / 2],
        [port.x, port.z + port.width / 2],
      ];
    }
    // yz
    if (Math.abs(port.x - payload.position) > half) return null;
    return [
      [port.z - port.width / 2, port.y],
      [port.z + port.width / 2, port.y],
    ];
  }

  private drawAxes(ctx: CanvasRenderingContext2D, payload: SectionPayload): void {
    const [[u0, u1], [v0, v1]] = payload.bounds;
    ctx.strokeStyle = "rgba(180,180,200,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const [ax, ay] = this.project(u0, 0);
    const [bx, by] = this.project(u1, 0);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    const [cx, cy] = this.project(0, v0);
    const [dx, dy] = this.project(0, v1);
    ctx.moveTo(cx, cy);
    ctx.lineTo(dx, dy);
    ctx.stroke();
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
    this.readout.textContent = `${payload.axis_u}=${u.toFixed(3)} ${payload.axis_v}=${v.toFixed(3)} µm · ${name}`;
  }

  private onClick(event: MouseEvent): void {
    const payload = this.payload;
    const state = this.state;
    if (!payload || !state) return;
    const rect = this.canvas.getBoundingClientRect();
    const [u, v] = this.unproject(event.clientX - rect.left, event.clientY - rect.top);
    for (const port of this.meta.ports) {
      const cut = this.portCut(port, payload);
      if (!cut) continue;
      if (this.nearSegment(u, v, cut[0], cut[1], 6 / this.view.scale)) {
        this.onPortSelect(port.name);
        return;
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
