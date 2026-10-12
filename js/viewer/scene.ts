// The 3D viewport: perspective camera, orbit/pan/zoom, fit-to-view and the
// axis triad + scale bar (spec R3).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class Scene3D {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  //: everything whose z is exaggerated lives under here
  readonly content = new THREE.Group();
  private readonly axes: THREE.AxesHelper;
  private readonly scaleBar: THREE.Group;
  private readonly resizeObserver: ResizeObserver;
  private target = new THREE.Vector3(0, 0, 0);
  private radius = 1;
  private zExaggeration = 1;

  constructor(private readonly container: HTMLElement) {
    this.scene.background = new THREE.Color(0x1b1b1f);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 2.0));
    const key = new THREE.DirectionalLight(0xffffff, 2.5);
    key.position.set(3, 5, 4);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 1.0);
    fill.position.set(-4, -2, -3);
    this.scene.add(fill);

    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1e5);
    this.camera.position.set(3, 2, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(this.content);
    this.axes = new THREE.AxesHelper(1);
    for (const material of asArray(this.axes.material)) material.depthTest = false;
    this.axes.renderOrder = 10;
    this.scene.add(this.axes);
    this.scaleBar = new THREE.Group();
    this.scene.add(this.scaleBar);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.renderLoop();
  }

  /** Frames the given box (µm, local origin) and sets up the triad + scale bar. */
  setBounds(size: [number, number, number]): void {
    this.target.set(0, 0, 0);
    this.radius = 0.5 * Math.hypot(size[0], size[1], size[2]);
    const span = Math.max(size[0], size[1], size[2]);
    this.axes.scale.setScalar(0.25 * span);
    this.axes.position.set(-size[0] / 2, -size[1] / 2, -size[2] / 2);
    this.buildScaleBar(span, size);
    this.fit();
  }

  /** Frames the whole scene (spec R3, "fit to view"). */
  fit(): void {
    const distance = this.radius / Math.sin((this.camera.fov * Math.PI) / 360);
    const direction = new THREE.Vector3(1, 0.8, 1).normalize();
    this.camera.position.copy(this.target).addScaledVector(direction, distance);
    this.camera.near = Math.max(distance / 1000, 1e-3);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(this.target);
    this.controls.update();
  }

  /** Scales the z of everything under `content` (spec R3, z-exaggeration). */
  setZExaggeration(factor: number): void {
    this.zExaggeration = factor;
    this.content.scale.set(1, 1, factor);
  }

  get exaggeration(): number {
    return this.zExaggeration;
  }

  private buildScaleBar(span: number, size: [number, number, number]): void {
    this.scaleBar.clear();
    const length = niceLength(span / 4);
    const y = -size[1] / 2;
    const z = -size[2] / 2;
    const x0 = -size[0] / 2;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x0, y, z),
      new THREE.Vector3(x0 + length, y, z),
    ]);
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0xdddddd, depthTest: false }),
    );
    line.renderOrder = 10;
    this.scaleBar.add(line);
  }

  private resize(): void {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private renderLoop = (): void => {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.renderLoop);
  };

  dispose(): void {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}

function asArray<T>(value: T | T[]): T[] {
  return Array.isArray(value) ? value : [value];
}

/** Rounds a length to 1/2/5 × 10ⁿ so the scale bar reads cleanly. */
function niceLength(value: number): number {
  if (value <= 0) return 1;
  const power = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / power;
  const step = normalized < 1.5 ? 1 : normalized < 3.5 ? 2 : normalized < 7.5 ? 5 : 10;
  return step * power;
}
