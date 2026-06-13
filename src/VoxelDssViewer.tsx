import { useEffect, useRef } from "react";
import {
  AmbientLight,
  Box3,
  BufferGeometry,
  Color,
  DirectionalLight,
  Float32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type VoxelSidecar = {
  format: string;
  dims: [number, number, number];
  cells: number[];
  voxelCount?: number;
  pitch?: number;
};

type VoxelCell = {
  x: number;
  y: number;
  z: number;
};

type FaceDef = {
  normal: Vector3;
  neighbor: [number, number, number];
  corners: [number, number, number][];
};

const DSS_KERNEL_RADIUS = 2;
const DSS_AO_RADIUS = 2;
const DSS_AO_INTENSITY = 0.72;
const MIN_AO = 0.18;
const BASE_COLOR = new Color("#d7c2aa");
const ACCENT_COLOR = new Color("#80ddd2");

const FACES: FaceDef[] = [
  {
    normal: new Vector3(1, 0, 0),
    neighbor: [1, 0, 0],
    corners: [
      [0.5, -0.5, -0.5],
      [0.5, 0.5, -0.5],
      [0.5, 0.5, 0.5],
      [0.5, -0.5, 0.5]
    ]
  },
  {
    normal: new Vector3(-1, 0, 0),
    neighbor: [-1, 0, 0],
    corners: [
      [-0.5, -0.5, 0.5],
      [-0.5, 0.5, 0.5],
      [-0.5, 0.5, -0.5],
      [-0.5, -0.5, -0.5]
    ]
  },
  {
    normal: new Vector3(0, 1, 0),
    neighbor: [0, 1, 0],
    corners: [
      [-0.5, 0.5, -0.5],
      [-0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
      [0.5, 0.5, -0.5]
    ]
  },
  {
    normal: new Vector3(0, -1, 0),
    neighbor: [0, -1, 0],
    corners: [
      [-0.5, -0.5, 0.5],
      [-0.5, -0.5, -0.5],
      [0.5, -0.5, -0.5],
      [0.5, -0.5, 0.5]
    ]
  },
  {
    normal: new Vector3(0, 0, 1),
    neighbor: [0, 0, 1],
    corners: [
      [0.5, -0.5, 0.5],
      [0.5, 0.5, 0.5],
      [-0.5, 0.5, 0.5],
      [-0.5, -0.5, 0.5]
    ]
  },
  {
    normal: new Vector3(0, 0, -1),
    neighbor: [0, 0, -1],
    corners: [
      [-0.5, -0.5, -0.5],
      [-0.5, 0.5, -0.5],
      [0.5, 0.5, -0.5],
      [0.5, -0.5, -0.5]
    ]
  }
];

class SparseVoxelField {
  readonly dims: { x: number; y: number; z: number };
  readonly cells: VoxelCell[] = [];
  private readonly occupied = new Set<number>();

  constructor(sidecar: VoxelSidecar) {
    this.dims = {
      x: sidecar.dims[0],
      y: sidecar.dims[1],
      z: sidecar.dims[2]
    };
    for (let index = 0; index + 2 < sidecar.cells.length; index += 3) {
      const x = sidecar.cells[index];
      const y = sidecar.cells[index + 1];
      const z = sidecar.cells[index + 2];
      if (!this.inBounds(x, y, z)) continue;
      const key = this.key(x, y, z);
      if (this.occupied.has(key)) continue;
      this.occupied.add(key);
      this.cells.push({ x, y, z });
    }
  }

  key(x: number, y: number, z: number) {
    return x * this.dims.y * this.dims.z + y * this.dims.z + z;
  }

  inBounds(x: number, y: number, z: number) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.dims.x && y < this.dims.y && z < this.dims.z;
  }

  isSolid(x: number, y: number, z: number) {
    return this.inBounds(x, y, z) && this.occupied.has(this.key(x, y, z));
  }
}

class DssField {
  private readonly normals = new Map<number, Vector3>();
  private readonly occlusion = new Map<number, number>();

  constructor(private readonly field: SparseVoxelField) {}

  normal(x: number, y: number, z: number) {
    const key = this.field.key(x, y, z);
    const cached = this.normals.get(key);
    if (cached) return cached;
    const normal = this.computeGradientNormal(x, y, z);
    if (normal.lengthSq() < 1e-4) {
      normal.set(
        x + 0.5 - this.field.dims.x / 2,
        y + 0.5 - this.field.dims.y / 2,
        z + 0.5 - this.field.dims.z / 2
      );
      if (normal.lengthSq() < 1e-4) normal.set(0, 1, 0);
    }
    normal.normalize();
    this.normals.set(key, normal);
    return normal;
  }

  vertexNormal(cell: VoxelCell, corner: [number, number, number], fallback: Vector3) {
    const sx = corner[0] > 0 ? 1 : -1;
    const sy = corner[1] > 0 ? 1 : -1;
    const sz = corner[2] > 0 ? 1 : -1;
    const normal = new Vector3();
    let total = 0;
    for (const ox of [0, sx]) {
      for (const oy of [0, sy]) {
        for (const oz of [0, sz]) {
          const x = cell.x + ox;
          const y = cell.y + oy;
          const z = cell.z + oz;
          if (!this.field.isSolid(x, y, z)) continue;
          const weight = ox === 0 && oy === 0 && oz === 0 ? 1.5 : 1;
          normal.addScaledVector(this.normal(x, y, z), weight);
          total += weight;
        }
      }
    }
    return total > 0 && normal.lengthSq() >= 1e-4 ? normal.normalize() : fallback.clone();
  }

  ao(x: number, y: number, z: number) {
    const key = this.field.key(x, y, z);
    const cached = this.occlusion.get(key);
    if (cached !== undefined) return cached;
    const normal = this.normal(x, y, z);
    let occluded = 0;
    let total = 0;
    for (let dx = -DSS_AO_RADIUS; dx <= DSS_AO_RADIUS; dx++) {
      for (let dy = -DSS_AO_RADIUS; dy <= DSS_AO_RADIUS; dy++) {
        for (let dz = -DSS_AO_RADIUS; dz <= DSS_AO_RADIUS; dz++) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
          const alignment = (dx * normal.x + dy * normal.y + dz * normal.z) / length;
          if (alignment <= 0) continue;
          const weight = gaussianWeight(dx, dy, dz, DSS_AO_RADIUS) * alignment;
          total += weight;
          if (this.field.isSolid(x + dx, y + dy, z + dz)) occluded += weight;
        }
      }
    }
    const value = total <= 1e-6 ? 1 : Math.max(MIN_AO, 1 - DSS_AO_INTENSITY * (occluded / total));
    this.occlusion.set(key, value);
    return value;
  }

  vertexAO(cell: VoxelCell, corner: [number, number, number], fallback: number) {
    const sx = corner[0] > 0 ? 1 : -1;
    const sy = corner[1] > 0 ? 1 : -1;
    const sz = corner[2] > 0 ? 1 : -1;
    let value = 0;
    let total = 0;
    for (const ox of [0, sx]) {
      for (const oy of [0, sy]) {
        for (const oz of [0, sz]) {
          const x = cell.x + ox;
          const y = cell.y + oy;
          const z = cell.z + oz;
          if (!this.field.isSolid(x, y, z)) continue;
          const weight = ox === 0 && oy === 0 && oz === 0 ? 1.5 : 1;
          value += this.ao(x, y, z) * weight;
          total += weight;
        }
      }
    }
    return total > 0 ? value / total : fallback;
  }

  private smoothedDensity(x: number, y: number, z: number, axisToIgnore: number) {
    let sum = 0;
    let total = 0;
    for (let dx = -DSS_KERNEL_RADIUS; dx <= DSS_KERNEL_RADIUS; dx++) {
      for (let dy = -DSS_KERNEL_RADIUS; dy <= DSS_KERNEL_RADIUS; dy++) {
        for (let dz = -DSS_KERNEL_RADIUS; dz <= DSS_KERNEL_RADIUS; dz++) {
          const wx = axisToIgnore === 0 ? 0 : dx;
          const wy = axisToIgnore === 1 ? 0 : dy;
          const wz = axisToIgnore === 2 ? 0 : dz;
          const weight = gaussianWeight(wx, wy, wz, DSS_KERNEL_RADIUS);
          sum += this.field.isSolid(x + dx, y + dy, z + dz) ? weight : 0;
          total += weight;
        }
      }
    }
    return total > 0 ? sum / total : 0;
  }

  private computeGradientNormal(x: number, y: number, z: number) {
    const radius = DSS_KERNEL_RADIUS;
    return new Vector3(
      this.smoothedDensity(x - radius, y, z, 0) - this.smoothedDensity(x + radius, y, z, 0),
      this.smoothedDensity(x, y - radius, z, 1) - this.smoothedDensity(x, y + radius, z, 1),
      this.smoothedDensity(x, y, z - radius, 2) - this.smoothedDensity(x, y, z + radius, 2)
    );
  }
}

function gaussianWeight(dx: number, dy: number, dz: number, radius: number) {
  const distance2 = dx * dx + dy * dy + dz * dz;
  const sigma = Math.max(0.75, radius * 0.65);
  return Math.exp(-distance2 / (2 * sigma * sigma));
}

function buildDssGeometry(sidecar: VoxelSidecar) {
  const field = new SparseVoxelField(sidecar);
  const dss = new DssField(field);
  const positions: number[] = [];
  const normals: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  let vertexIndex = 0;

  for (const cell of field.cells) {
    const baseNormal = dss.normal(cell.x, cell.y, cell.z);
    const baseAO = dss.ao(cell.x, cell.y, cell.z);
    const colorBlend = cell.y / Math.max(1, field.dims.y - 1);
    const color = BASE_COLOR.clone().lerp(ACCENT_COLOR, Math.max(0, Math.min(0.28, colorBlend * 0.28)));

    for (const face of FACES) {
      if (field.isSolid(cell.x + face.neighbor[0], cell.y + face.neighbor[1], cell.z + face.neighbor[2])) {
        continue;
      }
      for (const corner of face.corners) {
        const normal = dss.vertexNormal(cell, corner, baseNormal);
        const ao = dss.vertexAO(cell, corner, baseAO);
        positions.push(cell.x + 0.5 + corner[0], cell.y + 0.5 + corner[1], cell.z + 0.5 + corner[2]);
        normals.push(normal.x, normal.y, normal.z);
        colors.push(color.r * ao, color.g * ao, color.b * ao);
      }
      indices.push(vertexIndex, vertexIndex + 1, vertexIndex + 2, vertexIndex, vertexIndex + 2, vertexIndex + 3);
      vertexIndex += 4;
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new Float32BufferAttribute(normals, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  geometry.center();
  return geometry;
}

export function parseVoxelSidecar(raw: string): VoxelSidecar | null {
  const parsed = JSON.parse(raw) as Partial<VoxelSidecar>;
  if (!Array.isArray(parsed.dims) || parsed.dims.length !== 3) return null;
  if (!Array.isArray(parsed.cells) || parsed.cells.length < 3 || parsed.cells.length % 3 !== 0) return null;
  const dims = parsed.dims.map((value) => Number(value));
  if (dims.some((value) => !Number.isInteger(value) || value < 1 || value > 512)) return null;
  if (parsed.cells.some((value) => !Number.isInteger(value))) return null;
  return {
    format: typeof parsed.format === "string" ? parsed.format : "ih3d.voxel-dss.v1",
    dims: [dims[0], dims[1], dims[2]],
    cells: parsed.cells,
    voxelCount: typeof parsed.voxelCount === "number" ? parsed.voxelCount : parsed.cells.length / 3,
    pitch: typeof parsed.pitch === "number" ? parsed.pitch : undefined
  };
}

type VoxelDssViewerProps = {
  sidecar: VoxelSidecar;
  autoRotate: boolean;
};

export function VoxelDssViewer({ sidecar, autoRotate }: VoxelDssViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new Scene();
    scene.background = new Color("#070807");

    const camera = new PerspectiveCamera(42, 1, 0.1, 10000);
    const renderer = new WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor("#070807", 1);
    renderer.domElement.className = "dss-canvas";
    host.appendChild(renderer.domElement);

    const geometry = buildDssGeometry(sidecar);
    const material = new MeshStandardMaterial({
      metalness: 0.04,
      roughness: 0.82,
      vertexColors: true
    });
    const mesh = new Mesh(geometry, material);
    scene.add(mesh);

    scene.add(new AmbientLight("#f2ddc8", 1.65));
    const keyLight = new DirectionalLight("#fff3df", 3.1);
    keyLight.position.set(2.4, 3.2, 4.6);
    scene.add(keyLight);
    const rimLight = new DirectionalLight("#77e2d7", 1.45);
    rimLight.position.set(-3.2, 1.5, -2.2);
    scene.add(rimLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 2.4;
    controls.target.set(0, 0, 0);

    const box = new Box3().setFromObject(mesh);
    const size = box.getSize(new Vector3());
    const largestSide = Math.max(1, size.x, size.y, size.z);
    camera.near = Math.max(0.01, largestSide / 200);
    camera.far = largestSide * 20;
    camera.position.set(largestSide * 0.8, largestSide * 0.62, largestSide * 1.35);
    camera.updateProjectionMatrix();
    controls.update();

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    resize();

    let frame = 0;
    const render = () => {
      controls.autoRotate = autoRotate;
      controls.update();
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(render);
    };
    render();

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [autoRotate, sidecar]);

  return <div ref={hostRef} className="dss-viewer" />;
}
