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
}

export interface MetaPayload {
  levels: LevelMeta[];
  ports: PortMeta[];
  background: string;
}

export interface LayerStyle {
  color: string;
  opacity: number;
  visible: boolean;
}

export interface Selection {
  kind: "level" | "port";
  name: string;
}

export interface StatePayload {
  styles: Record<string, LayerStyle>;
  port_visible: Record<string, boolean>;
  selection: Selection | null;
  active_viewport: "3d" | "2d";
  show_full: boolean;
  plane: "xy" | "xz" | "yz";
  plane_position: number;
  plane_bounds: [number, number];
  //: gaussian 1/e distance (µm) for the 2D port fade vs. the plane's distance
  //: from the port (spec R4)
  port_fade_um: number;
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
