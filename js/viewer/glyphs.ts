// Port glyphs: an arrow along the port orientation plus a bar for its width
// (spec R3, Q12, Q15). Source/detector glyphs: a translucent plane at the
// object's port frame, before anything is placed (spec R12, D11).
import * as THREE from "three";

import type { GlyphPayload, PortMeta } from "./wire";

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
      this.group.add(group);
      return { group, material, uHalf: glyph.u_half, vHalf: glyph.v_half, axis: glyph.axis };
    }

    // a dipole (or an unresolvable port): a small marker instead of a plane
    group.add(new THREE.Mesh(new THREE.SphereGeometry(POINT_RADIUS, 12, 8), material));
    this.group.add(group);
    return { group, material, uHalf: 0, vHalf: 0, axis: glyph.axis };
  }

  setVisible(id: string, visible: boolean): void {
    const item = this.items.get(id);
    if (item) item.group.visible = visible;
  }

  setSelected(id: string | null): void {
    for (const [key, item] of this.items) {
      if (key === id) item.material.color.setHex(SELECTED_COLOR);
      else item.material.color.setHex(item.material.userData.baseColor);
    }
  }

  dispose(): void {
    this.clear();
  }

  private clear(): void {
    for (const item of this.items.values()) {
      item.group.traverse((object) => {
        if (object instanceof THREE.Mesh) object.geometry.dispose();
      });
      item.material.dispose();
    }
    this.items.clear();
    this.group.clear();
  }
}
