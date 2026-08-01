import { useEffect, useRef } from "react";
import {
  AmbientLight,
  Box3,
  Color,
  DirectionalLight,
  Material,
  Mesh,
  Object3D,
  PerspectiveCamera,
  Scene,
  Texture,
  Vector3,
  WebGLRenderer
} from "three";
import { MTLLoader } from "three/examples/jsm/loaders/MTLLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createSmoothNormalsGeometry } from "./smoothNormals";

type SurfaceMaterial = Material & {
  color?: Color;
  map?: Texture | null;
};

type SurfaceSnapshot = {
  material: SurfaceMaterial;
  color: Color | null;
  map: Texture | null;
};

type ObjPreviewViewerProps = {
  modelUrl: string;
  autoRotate: boolean;
  smoothNormals: boolean;
  appearance: "textured" | "clay";
  onTextureStatus: (status: { materialCount: number; textureCount: number }) => void;
  onError: (message: string | null) => void;
};

function materialList(material: Material | Material[]) {
  return Array.isArray(material) ? material : [material];
}

function collectSurfaceSnapshots(object: Object3D) {
  const snapshots: SurfaceSnapshot[] = [];
  const seen = new Set<Material>();
  object.traverse((node) => {
    if (!(node instanceof Mesh)) return;
    for (const material of materialList(node.material)) {
      if (seen.has(material)) continue;
      seen.add(material);
      const surface = material as SurfaceMaterial;
      snapshots.push({
        material: surface,
        color: surface.color?.clone() ?? null,
        map: surface.map ?? null
      });
    }
  });
  return snapshots;
}

function applyAppearance(snapshots: SurfaceSnapshot[], appearance: "textured" | "clay") {
  for (const snapshot of snapshots) {
    if (appearance === "clay") {
      snapshot.material.map = null;
      snapshot.material.color?.set("#d8c7b4");
    } else {
      snapshot.material.map = snapshot.map;
      if (snapshot.color && snapshot.material.color) {
        snapshot.material.color.copy(snapshot.color);
      }
    }
    snapshot.material.needsUpdate = true;
  }
}

export function ObjPreviewViewer({
  modelUrl,
  autoRotate,
  smoothNormals,
  appearance,
  onTextureStatus,
  onError
}: ObjPreviewViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appearanceRef = useRef(appearance);
  const autoRotateRef = useRef(autoRotate);
  const smoothNormalsRef = useRef(smoothNormals);
  const snapshotsRef = useRef<SurfaceSnapshot[]>([]);
  const meshGeometriesRef = useRef<
    Array<{ mesh: Mesh; original: Mesh["geometry"]; smooth: Mesh["geometry"] }>
  >([]);

  useEffect(() => {
    appearanceRef.current = appearance;
    applyAppearance(snapshotsRef.current, appearance);
  }, [appearance]);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    smoothNormalsRef.current = smoothNormals;
    for (const entry of meshGeometriesRef.current) {
      entry.mesh.geometry = smoothNormals ? entry.smooth : entry.original;
    }
  }, [smoothNormals]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    let frame = 0;
    let loadedObject: Object3D | null = null;
    snapshotsRef.current = [];
    meshGeometriesRef.current = [];
    onTextureStatus({ materialCount: 0, textureCount: 0 });
    onError(null);

    const scene = new Scene();
    scene.background = null;
    const camera = new PerspectiveCamera(32, 1, 0.01, 1000);
    const renderer = new WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = "srgb";
    renderer.domElement.className = "obj-preview-canvas";
    host.appendChild(renderer.domElement);

    scene.add(new AmbientLight("#fff5e8", 2.25));
    const keyLight = new DirectionalLight("#fff3df", 3.2);
    keyLight.position.set(2.5, 3.5, 4.8);
    scene.add(keyLight);
    const rimLight = new DirectionalLight("#78dfd5", 1.1);
    rimLight.position.set(-3, 1.8, -2.4);
    scene.add(rimLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotateSpeed = 2.4;
    controls.target.set(0, 0, 0);

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

    const showObject = (object: Object3D) => {
      if (disposed) return;
      loadedObject = object;
      const box = new Box3().setFromObject(object);
      const center = box.getCenter(new Vector3());
      const size = box.getSize(new Vector3());
      object.position.sub(center);
      scene.add(object);

      object.traverse((node) => {
        if (!(node instanceof Mesh)) return;
        const entry = {
          mesh: node,
          original: node.geometry,
          smooth: createSmoothNormalsGeometry(node.geometry)
        };
        node.geometry = smoothNormalsRef.current ? entry.smooth : entry.original;
        meshGeometriesRef.current.push(entry);
      });

      const largestSide = Math.max(0.01, size.x, size.y, size.z);
      camera.near = Math.max(0.001, largestSide / 200);
      camera.far = Math.max(100, largestSide * 30);
      camera.position.set(largestSide * 0.9, largestSide * 0.65, largestSide * 1.55);
      camera.updateProjectionMatrix();
      controls.update();

      const snapshots = collectSurfaceSnapshots(object);
      snapshotsRef.current = snapshots;
      const textures = new Set(snapshots.map((snapshot) => snapshot.map).filter(Boolean));
      onTextureStatus({
        materialCount: snapshots.length,
        textureCount: textures.size
      });
      applyAppearance(snapshots, appearanceRef.current);
    };

    const loadObject = (materials?: MTLLoader.MaterialCreator) => {
      const loader = new OBJLoader();
      if (materials) loader.setMaterials(materials);
      loader.load(
        modelUrl,
        showObject,
        undefined,
        (error) => {
          if (!disposed) onError(`OBJ preview failed: ${String(error)}`);
        }
      );
    };

    const modelWithoutQuery = modelUrl.split("?", 1)[0];
    const materialUrl = modelWithoutQuery.replace(/\.obj$/i, ".mtl");
    const resourcePath = modelWithoutQuery.slice(0, modelWithoutQuery.lastIndexOf("/") + 1);
    const materialLoader = new MTLLoader();
    materialLoader.setResourcePath(resourcePath);
    materialLoader.load(
      materialUrl,
      (materials) => {
        materials.preload();
        loadObject(materials);
      },
      undefined,
      () => loadObject()
    );

    const render = () => {
      controls.autoRotate = autoRotateRef.current;
      controls.update();
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(render);
    };
    render();

    return () => {
      disposed = true;
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      controls.dispose();
      const materials = new Set<Material>();
      const textures = new Set<Texture>();
      loadedObject?.traverse((node) => {
        if (!(node instanceof Mesh)) return;
        for (const material of materialList(node.material)) {
          materials.add(material);
          const map = (material as SurfaceMaterial).map;
          if (map) textures.add(map);
        }
      });
      for (const entry of meshGeometriesRef.current) {
        entry.original.dispose();
        entry.smooth.dispose();
      }
      for (const snapshot of snapshotsRef.current) {
        if (snapshot.map) textures.add(snapshot.map);
      }
      for (const texture of textures) texture.dispose();
      for (const material of materials) material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      snapshotsRef.current = [];
      meshGeometriesRef.current = [];
    };
  }, [modelUrl, onError, onTextureStatus]);

  return <div ref={hostRef} className="obj-preview-viewer" />;
}
