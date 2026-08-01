import { BufferAttribute, type BufferGeometry } from "three";

function positionKey(x: number, y: number, z: number, tolerance: number) {
  return `${Math.round(x / tolerance)},${Math.round(y / tolerance)},${Math.round(z / tolerance)}`;
}

export function createSmoothNormalsGeometry(source: BufferGeometry) {
  const geometry = source.clone();
  const positions = geometry.getAttribute("position");
  if (!positions || positions.itemSize < 3 || positions.count < 3) return geometry;

  geometry.computeBoundingBox();
  const bounds = geometry.boundingBox;
  const span = bounds
    ? Math.max(
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z
      )
    : 1;
  const tolerance = Math.max(span * 1e-6, 1e-9);
  const keys = new Array<string>(positions.count);
  const accumulated = new Map<string, [number, number, number]>();

  for (let vertex = 0; vertex < positions.count; vertex += 1) {
    keys[vertex] = positionKey(
      positions.getX(vertex),
      positions.getY(vertex),
      positions.getZ(vertex),
      tolerance
    );
  }

  const addFace = (a: number, b: number, c: number) => {
    const ax = positions.getX(a);
    const ay = positions.getY(a);
    const az = positions.getZ(a);
    const abx = positions.getX(b) - ax;
    const aby = positions.getY(b) - ay;
    const abz = positions.getZ(b) - az;
    const acx = positions.getX(c) - ax;
    const acy = positions.getY(c) - ay;
    const acz = positions.getZ(c) - az;
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;

    for (const vertex of [a, b, c]) {
      const key = keys[vertex];
      const normal = accumulated.get(key);
      if (normal) {
        normal[0] += nx;
        normal[1] += ny;
        normal[2] += nz;
      } else {
        accumulated.set(key, [nx, ny, nz]);
      }
    }
  };

  const indices = geometry.getIndex();
  if (indices) {
    for (let offset = 0; offset + 2 < indices.count; offset += 3) {
      addFace(indices.getX(offset), indices.getX(offset + 1), indices.getX(offset + 2));
    }
  } else {
    for (let offset = 0; offset + 2 < positions.count; offset += 3) {
      addFace(offset, offset + 1, offset + 2);
    }
  }

  const normals = new Float32Array(positions.count * 3);
  for (let vertex = 0; vertex < positions.count; vertex += 1) {
    const normal = accumulated.get(keys[vertex]) ?? [0, 1, 0];
    const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
    normals[vertex * 3] = normal[0] / length;
    normals[vertex * 3 + 1] = normal[1] / length;
    normals[vertex * 3 + 2] = normal[2] / length;
  }
  geometry.setAttribute("normal", new BufferAttribute(normals, 3));
  return geometry;
}
