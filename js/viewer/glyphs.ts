// Port glyphs: an arrow along the port orientation plus a bar for its width
// (spec R3, Q12, Q15). Source/detector glyphs: a translucent plane at the
// object's port frame, before anything is placed (spec R12, D11). A detector
// with a result can shade its plane with the field it recorded (spec R15, D16).
import * as THREE from "three";

import {
  componentData,
  parseColor,
  rampTexture,
  transformValues,
  valueRange,
} from "./fieldramp";
import type { FieldPayload, FieldSettings, GlyphPayload, PortMeta } from "./wire";

const GLYPH_COLOR = 0xffcc33;
const SELECTED_COLOR = 0x66ff66;
//: sources are warm, detectors cool, so a glance tells them apart (spec R12)
const SOURCE_COLOR = 0xff8844;
const DETECTOR_COLOR = 0x44bbff;
const ERROR_COLOR = 0xff4444;
//: how opaque a source/detector plane is: readable over the solids, still see-through
const PLANE_OPACITY = 0.35;
//: a dipole glyph has no plane; draw a small sphere instead, of this radius (µm)
const POINT_RADIUS = 0.15;
//: the two unit vectors spanning a plane whose normal is the given axis (0=x,
//: 1=y, 2=z), in the order the wire's u_half / v_half expect
const PLANE_SPAN: [THREE.Vector3, THREE.Vector3][] = [
  [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)], // normal x: u=y, v=z
  [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1)], // normal y: u=x, v=z
  [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0)], // normal z: u=x, v=y
];

export class PortGlyphs {
  readonly group = new THREE.Group();
  private readonly items = new Map<string, { group: THREE.Group; material: THREE.MeshBasicMaterial }>();

  constructor(ports: PortMeta[]) {
    // a simulation-volume plane port (spec R12) draws nothing on its own: it is
    // only a dropdown choice, and the plane shows up once a source/detector is
    // placed on it (PlaneGlyphs, spec D11)
    for (const port of ports) {
      if (port.plane !== null) continue;
      this.items.set(port.name, this.build(port));
    }
  }

  private build(port: PortMeta): { group: THREE.Group; material: THREE.MeshBasicMaterial } {
    const group = new THREE.Group();
    const material = new THREE.MeshBasicMaterial({ color: GLYPH_COLOR });
    // ports are drawn at their own z; the group is placed at the port center
    group.position.set(port.x, port.y, port.z);

    const angle = ((port.orientation ?? 0) * Math.PI) / 180;
    const arrow = new THREE.Group();
    arrow.rotation.z = angle;

    // the width bar runs across the port, perpendicular to the arrow (the port
    // width spans y before the arrow group's rotation)
    const bar = new THREE.Mesh(new THREE.BoxGeometry(port.width * 0.06, port.width, port.width * 0.06), material);
    arrow.add(bar);

    // the arrow points along the orientation
    const shaft = new THREE.Mesh(
      new THREE.BoxGeometry(port.width * 0.6, port.width * 0.04, port.width * 0.04),
      material,
    );
    shaft.position.x = port.width * 0.3;
    arrow.add(shaft);
    const head = new THREE.Mesh(new THREE.ConeGeometry(port.width * 0.12, port.width * 0.25, 16), material);
    head.rotation.z = -Math.PI / 2;
    head.position.x = port.width * 0.72;
    arrow.add(head);

    group.add(arrow);
    this.group.add(group);
    return { group, material };
  }

  setVisible(name: string, visible: boolean): void {
    const item = this.items.get(name);
    if (item) item.group.visible = visible;
  }

  setSelected(name: string | null): void {
    for (const [key, item] of this.items) {
      item.material.color.setHex(key === name ? SELECTED_COLOR : GLYPH_COLOR);
    }
  }

  /** The port whose glyph a ray hits first, or null (spec R12, double-click). */
  pick(raycaster: THREE.Raycaster): string | null {
    const hits = raycaster.intersectObjects(this.group.children, true);
    if (hits.length === 0) return null;
    // the hit mesh sits under the port's group; walk up to that group
    let node: THREE.Object3D | null = hits[0].object;
    while (node && node.parent !== this.group) node = node.parent;
    if (!node) return null;
    for (const [name, item] of this.items) if (item.group === node) return name;
    return null;
  }

  dispose(): void {
    for (const item of this.items.values()) {
      item.group.traverse((object) => {
        if (object instanceof THREE.Mesh) object.geometry.dispose();
      });
      item.material.dispose();
    }
  }
}

//: A source/detector glyph as the viewport holds it: the group to toggle, the
//: material to recolor, and the group holding the plane (a dipole has none, so
//: a click can add a plane to it or leave the sphere).
interface PlaneItem {
  group: THREE.Group;
  material: THREE.MeshBasicMaterial;
  /** the plane half-width along u; 0 for a dipole */
  uHalf: number;
  /** the plane half-height along v; 0 for a dipole */
  vHalf: number;
  /** the plane normal axis (0=x, 1=y, 2=z) */
  axis: number;
  /** the textured plane mesh, when the glyph has one (spec R15) */
  plane: THREE.Mesh | null;
  /** the plane's field material, once a field has been shown (spec R15) */
  fieldMaterial: THREE.MeshBasicMaterial | null;
  /** the outline drawn while a field replaces the fill (spec R15, D16) */
  outline: THREE.LineLoop | null;
  /** the field texture, so a re-shade can update it in place (spec R15) */
  texture: THREE.DataTexture | null;
}

/**
 * The source/detector plane glyphs (spec R12, D11).
 *
 * Drawn from the spec's port frame, so the viewport is a pre-flight view: the
 * planes show where the objects will be placed, before any `create_*_at` call.
 * `rebuild` replaces the whole set on every setup edit (the glyph geometry
 * depends on the port frame, which an edit can move), which is cheap next to
 * the mesh payload.
 */
export class PlaneGlyphs {
  readonly group = new THREE.Group();
  private readonly items = new Map<string, PlaneItem>();

  /** Rebuilds every glyph from the state's glyph payload (spec R12, D11). */
  rebuild(glyphs: GlyphPayload[]): void {
    this.clear();
    for (const glyph of glyphs) this.items.set(glyph.id, this.build(glyph));
  }

  private build(glyph: GlyphPayload): PlaneItem {
    const group = new THREE.Group();
    const baseColor = glyph.error ? ERROR_COLOR : glyph.kind === "source" ? SOURCE_COLOR : DETECTOR_COLOR;
    const material = new THREE.MeshBasicMaterial({
      color: baseColor,
      transparent: true,
      opacity: PLANE_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    material.userData.baseColor = baseColor;
    group.position.set(glyph.center[0], glyph.center[1], glyph.center[2]);

    if (glyph.u_half > 0 && glyph.v_half > 0) {
      // a plane rectangle, laid out on its normal axis by u/v span vectors
      const [u, v] = PLANE_SPAN[glyph.axis] ?? PLANE_SPAN[0];
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2 * glyph.u_half, 2 * glyph.v_half), material);
      const normal = u.clone().cross(v);
      const basis = new THREE.Matrix4().makeBasis(u, v, normal);
      mesh.quaternion.setFromRotationMatrix(basis);
      group.add(mesh);
      // the outline that marks the plane's extent once a field replaces the
      // translucent fill (spec R15, D16); hidden until a field is shown
      const outline = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-glyph.u_half, -glyph.v_half, 0),
          new THREE.Vector3(glyph.u_half, -glyph.v_half, 0),
          new THREE.Vector3(glyph.u_half, glyph.v_half, 0),
          new THREE.Vector3(-glyph.u_half, glyph.v_half, 0),
        ]),
        new THREE.LineBasicMaterial({ color: baseColor, transparent: true, opacity: 0.9 }),
      );
      outline.quaternion.copy(mesh.quaternion);
      outline.visible = false;
      group.add(outline);
      this.group.add(group);
      return {
        group, material, uHalf: glyph.u_half, vHalf: glyph.v_half, axis: glyph.axis,
        plane: mesh, fieldMaterial: null, outline, texture: null,
      };
    }

    // a dipole (or an unresolvable port): a small marker instead of a plane
    group.add(new THREE.Mesh(new THREE.SphereGeometry(POINT_RADIUS, 12, 8), material));
    this.group.add(group);
    return {
      group, material, uHalf: 0, vHalf: 0, axis: glyph.axis,
      plane: null, fieldMaterial: null, outline: null, texture: null,
    };
  }

  /**
   * Shades a detector's plane with the field it recorded (spec R15, D16).
   *
   * The transformed component becomes a texture on the plane; the plain
   * translucent fill is replaced (the outline marks the extent instead). A null
   * `field` — no result, no component, or `field_visible` off — restores the
   * plain glyph.
   */
  setField(id: string, field: FieldPayload | null, settings: FieldSettings, logRange: number): void {
    const item = this.items.get(id);
    if (!item || !item.plane) return;
    const component = settings.field_component;
    // the selected frequency frame, client-side so a slider move re-shades with
    // no kernel round-trip (spec R17)
    const frame = settings.field_freq ?? 0;
    const data = field && component ? componentData(field, component, frame) : null;
    if (!field || !data || !settings.field_visible) {
      this.clearField(item);
      return;
    }
    const values = transformValues(data.real, data.imag, settings.field_transform);
    const range = valueRange(values, settings.field_transform, logRange);
    const low = parseColor(settings.field_color_low);
    const high = parseColor(settings.field_color_high);
    const bytes = rampTexture(values, range, settings.field_transform, low, high, settings.field_opacity);

    const [rows, cols] = field.grid;
    if (!item.texture || item.texture.image.width !== cols || item.texture.image.height !== rows) {
      item.texture?.dispose();
      item.texture = new THREE.DataTexture(
        new Uint8Array(cols * rows * 4), cols, rows, THREE.RGBAFormat,
      );
      // nearest-neighbour, so each recorded sample stays a visible block
      item.texture.minFilter = THREE.NearestFilter;
      item.texture.magFilter = THREE.NearestFilter;
      item.texture.colorSpace = THREE.SRGBColorSpace;
    }
    (item.texture.image.data as Uint8Array).set(bytes);
    item.texture.needsUpdate = true;
    // each sample is a cell centred where it was recorded, which is not always
    // the cell's centre (``field.shift``, in cells): slide the textured plane
    // along its own span vectors to match
    const [shiftU, shiftV] = field.shift ?? [0, 0];
    const [u, v] = PLANE_SPAN[item.axis] ?? PLANE_SPAN[0];
    item.plane.position
      .copy(u).multiplyScalar((shiftU * 2 * item.uHalf) / cols)
      .addScaledVector(v, (shiftV * 2 * item.vHalf) / rows);

    if (!item.fieldMaterial) {
      item.fieldMaterial = new THREE.MeshBasicMaterial({
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      item.plane.material = item.fieldMaterial;
    }
    item.fieldMaterial.map = item.texture;
    item.fieldMaterial.opacity = 1;
    item.fieldMaterial.needsUpdate = true;
    if (item.outline) item.outline.visible = settings.field_outline !== false;
    // the plain translucent fill is replaced by the field (spec R15)
    item.material.visible = false;
  }

  private clearField(item: PlaneItem): void {
    if (item.fieldMaterial && item.plane) {
      item.plane.material = item.material;
      item.fieldMaterial.dispose();
      item.fieldMaterial = null;
    }
    item.texture?.dispose();
    item.texture = null;
    item.plane?.position.set(0, 0, 0);
    if (item.outline) item.outline.visible = false;
    item.material.visible = true;
  }

  setVisible(id: string, visible: boolean): void {
    const item = this.items.get(id);
    if (item) item.group.visible = visible;
  }

  setSelected(id: string | null): void {
    for (const [key, item] of this.items) {
      const color = key === id ? SELECTED_COLOR : item.material.userData.baseColor;
      item.material.color.setHex(color);
      if (item.outline) (item.outline.material as THREE.LineBasicMaterial).color.setHex(color);
    }
  }

  dispose(): void {
    this.clear();
  }

  private clear(): void {
    for (const item of this.items.values()) {
      item.texture?.dispose();
      item.fieldMaterial?.dispose();
      item.group.traverse((object) => {
        if (object instanceof THREE.Mesh) object.geometry.dispose();
        if (object instanceof THREE.LineLoop) {
          object.geometry.dispose();
          (object.material as THREE.Material).dispose();
        }
      });
      item.material.dispose();
    }
    this.items.clear();
    this.group.clear();
  }
}
