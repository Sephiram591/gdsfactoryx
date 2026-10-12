// Port glyphs: an arrow along the port orientation plus a bar for its width
// (spec R3, Q12, Q15).
import * as THREE from "three";

import type { PortMeta } from "./wire";

const GLYPH_COLOR = 0xffcc33;
const SELECTED_COLOR = 0x66ff66;

export class PortGlyphs {
  readonly group = new THREE.Group();
  private readonly items = new Map<string, { group: THREE.Group; material: THREE.MeshBasicMaterial }>();

  constructor(ports: PortMeta[]) {
    for (const port of ports) this.items.set(port.name, this.build(port));
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

  dispose(): void {
    for (const item of this.items.values()) {
      item.group.traverse((object) => {
        if (object instanceof THREE.Mesh) object.geometry.dispose();
      });
      item.material.dispose();
    }
  }
}
