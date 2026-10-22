// The wire format the kernel sends (spec D5, C5).
//
// Meshes travel as contiguous binary buffers: ipywidgets extracts them from the
// trait value and delivers them as DataView, so positions are decoded straight
// into a Float32Array with no base64 or JSON cost, at float32 precision measured
// from the box center (spec C5).

export interface MeshPayload {
  positions: DataView | ArrayBuffer;
  count: number;
  indices: DataView | ArrayBuffer;
  tris: number;
}

export interface LevelPayload {
  name: string;
  painted: MeshPayload;
  full: MeshPayload;
}

export interface GeometryPayload {
  origin: [number, number, number];
  size: [number, number, number];
  levels: LevelPayload[];
  background: MeshPayload;
}

export interface LevelMeta {
  name: string;
  material: string;
  mesh_order: number;
  fill: boolean;
  zmin: number;
  thickness: number;
  sidewall_angle: number;
  bias: number;
  width_to_z: number;
}

export interface PortMeta {
  name: string;
  x: number;
  y: number;
  z: number;
  orientation: number | null;
  width: number;
  //: the port layer's level thickness (µm); the 2D in-plane box depth (spec R4)
  height: number;
  layer: number;
  //: the normal axis (0=x, 1=y, 2=z) of a simulation-volume plane port, or null
  //: for a device port; a plane port is drawn as a rectangle (spec R12)
  plane: number | null;
}

export interface MetaPayload {
  levels: LevelMeta[];
  ports: PortMeta[];
  background: string;
  //: the source/detector types the explorer can add (spec R12)
  types: Record<string, TypeMeta>;
}

//: One creatable source/detector type (spec R12): its category, whether it is
//: sized by `port_size_mult`, how many wavelengths it takes, whether it has an
//: `in`/`out` direction, and the extra editable parameter names it accepts.
export interface TypeMeta {
  category: "source" | "detector";
  plane: boolean;
  wave: "one" | "many" | "none";
  direction: boolean;
  extra: string[];
}

export interface LayerStyle {
  color: string;
  opacity: number;
  visible: boolean;
}

export interface Selection {
  kind: "level" | "port" | "source" | "detector";
  name: string;
}

//: A source/detector as data, not a live object (spec D11). `id` is stable;
//: `params` holds the editable fields (absent ones fall back to the settings).
//: A spec with `port: "None"` is placed absolutely (axis/center/size params,
//: spec R18); one with `adopted: true` came from a loaded result and is
//: read-only except its display settings (spec R16).
export interface SpecPayload {
  id: string;
  type: string;
  port: string;
  params: Record<string, unknown>;
  adopted?: boolean;
}

//: A source/detector plane glyph, in the local frame the meshes use (spec C5,
//: R12). `axis` is the plane normal (0=x, 1=y, 2=z); `u_half`/`v_half` are the
//: two half-extents.
export interface GlyphPayload {
  //: the spec's id; two specs may share a name, so selection matches on this
  id: string;
  name: string;
  kind: "source" | "detector";
  type: string;
  port: string;
  center: [number, number, number];
  axis: number;
  u_half: number;
  v_half: number;
  direction: string | null;
  //: the pre-placement visibility toggle (spec R12)
  visible: boolean;
  error?: boolean;
}

//: The setup-wide simulation settings (spec R13).
export interface SimSettingsPayload {
  wavelength_um: number;
  pulse_width_frac: number;
  port_size_mult: [number, number];
  pml_thickness: number;
  resolution_um: number;
  time_s: number;
  //: `(x+, x-, y+, y-, z+, z-)` in µm — how far the box's faces move off the
  //: geometry's bounding box (spec R13). The geometry does not change.
  boundary_offset: [number, number, number, number, number, number];
  //: the bottom of the `logabs` color ramp, in decades below the plane's
  //: maximum (spec R15, Q22)
  log_range: number;
  //: how the device averages the permittivity at material interfaces (spec
  //: R13): the diagonal Kottke tensor, the full tensor (oblique and sloped
  //: walls; ~5x the time per step) or fdtdx's isotropic inverse average
  averaging: Averaging;
  //: whether a stopped run keeps its partial detector fields instead of
  //: discarding them (spec D17)
  keep_partial: boolean;
}

//: The permittivity averagings the Simulation tab offers (spec R13).
export type Averaging = "subpixel" | "subpixel_full" | "inverse";

//: One detector's field from the last result (spec R15, D16, R17). The float32
//: buffers are frequency-major stacks: frequency `f` occupies
//: `components.length * grid[0] * grid[1]` values starting at
//: `f * components.length * grid[0] * grid[1]`, and within a frequency,
//: component `c` occupies `grid[0]*grid[1]` values at
//: `(f * components.length + c) * grid[0] * grid[1]`.
export interface FieldPayload {
  //: the spec's id (matches a `SpecPayload.id`)
  id: string;
  name: string;
  components: string[];
  //: `[rows, cols]` of one component plane
  grid: [number, number];
  //: where a sample sits along the plane's `(u, v)` axes, in cells from the
  //: cell's centre (-0.5 on x and y, 0 on z); absent means cell-centred
  shift?: [number, number];
  //: the recorded frequencies in Hz (spec R17); frame `i` of the buffers is
  //: `freqs[i]`
  freqs?: number[];
  //: the initially selected frequency frame
  freq?: number;
  real: DataView | ArrayBuffer;
  imag: DataView | ArrayBuffer;
}

//: A detector's field display settings (spec R15, D16). `component` is null
//: until the user picks one, or when the result recorded none.
export interface FieldSettings {
  field_component: string | null;
  field_transform: "real" | "imag" | "abs" | "phase" | "logabs";
  field_color_low: string;
  field_color_high: string;
  field_opacity: number;
  field_visible: boolean;
  //: the selected frequency frame of a multi-frequency detector (spec R17)
  field_freq?: number;
  //: whether shading draws the plane's outline (spec R17); default true
  field_outline?: boolean;
}

export interface StatePayload {
  styles: Record<string, LayerStyle>;
  //: which explorer tab the browser shows; the kernel sends the active world's
  //: geometry/state for it (the Result tab shows the result's own, spec R17)
  active_tab: "layout" | "sources" | "simulation" | "result";
  port_visible: Record<string, boolean>;
  selection: Selection | null;
  active_viewport: "3d" | "2d";
  show_full: boolean;
  plane: "xy" | "xz" | "yz";
  plane_position: number;
  plane_bounds: [number, number];
  //: gaussian 1/e distance for the 2D plane fade, as a fraction of the box
  //: size along the section's normal (spec R4)
  plane_fade_frac: number;
  sources: SpecPayload[];
  detectors: SpecPayload[];
  glyphs: GlyphPayload[];
  //: per-detector field display settings, keyed by spec id (spec R15)
  field_settings: Record<string, FieldSettings>;
  simulation: SimSettingsPayload;
  //: true while a mode-1 run is in flight; the setup is locked (spec D15)
  running: boolean;
}

// The 2D cross-section payload (spec R4). Rings are `[u, v]` pairs in the
// section's local frame (from the box center, spec C5); a level's rings may be
// outer or hole contours and are filled even-odd.
export type Ring = [number, number][];

export interface SectionPayload {
  plane: "xy" | "xz" | "yz";
  position: number;
  axis_u: string;
  axis_v: string;
  normal_axis: string;
  //: the box center in stack coordinates (spec C5); rings/bounds are local, so
  //: labels and the hover readout add this to read in device coordinates
  origin: [number, number, number];
  bounds: [[number, number], [number, number]];
  rings: Record<string, Ring[]>;
  background: Ring[];
}

// A buffer may arrive as DataView (ipywidgets) or ArrayBuffer. A typed-array
// view over a DataView's backing buffer requires 4-byte alignment at its
// byteOffset, which ipywidgets does not guarantee, so copy the bytes out first
// when the offset is misaligned. Copying is cheap next to the mesh transfer.
function bytesOf(buffer: DataView | ArrayBuffer): { source: ArrayBuffer; offset: number } {
  return buffer instanceof ArrayBuffer
    ? { source: buffer, offset: 0 }
    : { source: buffer.buffer as ArrayBuffer, offset: buffer.byteOffset };
}

function asFloat32(buffer: DataView | ArrayBuffer, count: number): Float32Array {
  const { source, offset } = bytesOf(buffer);
  if (offset % 4 === 0) return new Float32Array(source, offset, count);
  return new Float32Array(source.slice(offset, offset + count * 4));
}

function asUint32(buffer: DataView | ArrayBuffer, count: number): Uint32Array {
  const { source, offset } = bytesOf(buffer);
  if (offset % 4 === 0) return new Uint32Array(source, offset, count);
  return new Uint32Array(source.slice(offset, offset + count * 4));
}

export function decodePositions(mesh: MeshPayload): Float32Array {
  return asFloat32(mesh.positions, mesh.count * 3);
}

export function decodeIndices(mesh: MeshPayload): Uint32Array {
  return asUint32(mesh.indices, mesh.tris * 3);
}
