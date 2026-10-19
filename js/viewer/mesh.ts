// Level solids as three.js meshes (spec R3, Q13).
//
// Each level has two prebuilt geometries: `painted` (where it wins by
// `mesh_order`) and `full` (unpainted). The "show full extent" toggle swaps
// between them; a color or opacity edit only touches the material, so no
// geometry is rebuilt (spec D10).
import * as THREE from "three";

import { decodeIndices, decodePositions, type LevelPayload } from "./wire";

function geometryFrom(mesh: LevelPayload["painted"]): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(decodePositions(mesh), 3),
  );
  geometry.setIndex(new THREE.BufferAttribute(decodeIndices(mesh), 1));
  geometry.computeVertexNormals();
  return geometry;
}

export class LevelMeshes {
  readonly group = new THREE.Group();
  private readonly painted: THREE.BufferGeometry;
  private readonly full: THREE.BufferGeometry;
  private readonly material: THREE.MeshPhongMaterial;
  readonly mesh: THREE.Mesh;
  private showingFull = false;

  constructor(level: LevelPayload, color: string, opacity: number) {
    this.painted = geometryFrom(level.painted);
    this.full = geometryFrom(level.full);
    this.material = new THREE.MeshPhongMaterial({
      color: new THREE.Color(color),
      transparent: opacity < 1,
      opacity,
      // translucent layers must not occlude each other, or the stack reads
      // inside-out; opaque ones keep depth writes
      depthWrite: opacity >= 1,
      side: THREE.DoubleSide,
      shininess: 30,
    });
    this.mesh = new THREE.Mesh(this.painted, this.material);
    this.group.add(this.mesh);
  }

  setColor(color: string): void {
    this.material.color = new THREE.Color(color);
  }

  setOpacity(opacity: number): void {
    this.material.opacity = opacity;
    this.material.transparent = opacity < 1;
    this.material.depthWrite = opacity >= 1;
    this.material.needsUpdate = true;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setShowFull(showFull: boolean): void {
    if (showFull === this.showingFull) return;
    this.showingFull = showFull;
    this.mesh.geometry = showFull ? this.full : this.painted;
  }

  setSelected(selected: boolean): void {
    this.material.emissive = new THREE.Color(selected ? 0x333311 : 0x000000);
  }

  dispose(): void {
    this.painted.dispose();
    this.full.dispose();
    this.material.dispose();
  }
}
