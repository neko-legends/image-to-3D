import { useEffect, useRef } from "react";
import {
  ACESFilmicToneMapping,
  AmbientLight,
  BackSide,
  Box3,
  Color,
  DirectionalLight,
  HemisphereLight,
  Material,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  SRGBColorSpace,
  Texture,
  Vector3,
  WebGLRenderer
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { createSmoothNormalsGeometry } from "./smoothNormals";

export type GltfPreviewAppearance = "textured" | "clay" | "cel";

type SurfaceMaterial = Material & {
  color?: Color;
  map?: Texture | null;
  alphaMap?: Texture | null;
  aoMap?: Texture | null;
  bumpMap?: Texture | null;
  emissiveMap?: Texture | null;
  envMap?: Texture | null;
  lightMap?: Texture | null;
  metalness?: number;
  metalnessMap?: Texture | null;
  normalMap?: Texture | null;
  roughness?: number;
  roughnessMap?: Texture | null;
};

type MeshMaterials = {
  mesh: Mesh;
  textured: Material | Material[];
  clay: Material | Material[];
  cel: Material | Material[];
};

type MeshGeometries = {
  mesh: Mesh;
  outline: Mesh;
  original: Mesh["geometry"];
  smooth: Mesh["geometry"];
};

type GltfPreviewViewerProps = {
  modelUrl: string;
  autoRotate: boolean;
  smoothNormals: boolean;
  appearance: GltfPreviewAppearance;
  onTextureStatus: (status: { materialCount: number; textureCount: number }) => void;
  onError: (message: string | null) => void;
};

const TEXTURE_KEYS = [
  "map",
  "alphaMap",
  "aoMap",
  "bumpMap",
  "emissiveMap",
  "envMap",
  "lightMap",
  "metalnessMap",
  "normalMap",
  "roughnessMap"
] as const;

const CEL_VERTEX_SHADER = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDirection;

  void main() {
    vUv = uv;
    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewDirection = normalize(-viewPosition.xyz);
    gl_Position = projectionMatrix * viewPosition;
  }
`;

const CEL_FRAGMENT_SHADER = `
  uniform vec3 baseColor;
  uniform sampler2D colorMap;
  uniform bool hasColorMap;
  uniform float opacity;
  uniform float alphaTestValue;

  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDirection;

  void main() {
    vec4 texel = hasColorMap ? texture2D(colorMap, vUv) : vec4(1.0);
    float alpha = texel.a * opacity;
    if (alpha < alphaTestValue) discard;

    vec3 normal = normalize(vNormal);
    vec3 lightDirection = normalize(vec3(0.38, 0.72, 0.58));
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float lightBand =
      diffuse > 0.72 ? 1.0 :
      diffuse > 0.38 ? 0.82 :
      diffuse > 0.08 ? 0.64 : 0.46;
    float rim = pow(1.0 - max(dot(normal, normalize(vViewDirection)), 0.0), 3.0);
    vec3 albedo = texel.rgb * baseColor;
    vec3 litColor = albedo * lightBand;
    litColor += rim * vec3(0.16, 0.28, 0.30);

    gl_FragColor = vec4(litColor, alpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

const OUTLINE_VERTEX_SHADER = `
  uniform float outlineWidth;

  void main() {
    vec3 expandedPosition = position + normalize(normal) * outlineWidth;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(expandedPosition, 1.0);
  }
`;

const OUTLINE_FRAGMENT_SHADER = `
  uniform vec3 outlineColor;

  void main() {
    gl_FragColor = vec4(outlineColor, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

function materialList(material: Material | Material[]) {
  return Array.isArray(material) ? material : [material];
}

function mapMaterials(
  material: Material | Material[],
  create: (surface: SurfaceMaterial) => Material
) {
  const mapped = materialList(material).map((item) => create(item as SurfaceMaterial));
  return Array.isArray(material) ? mapped : mapped[0];
}

function matteTexturedMaterial(surface: SurfaceMaterial) {
  surface.metalness = 0.02;
  surface.roughness = 0.88;
  surface.metalnessMap = null;
  surface.roughnessMap = null;
  surface.needsUpdate = true;
  return surface;
}

function clayMaterial(surface: SurfaceMaterial) {
  return new MeshStandardMaterial({
    color: "#d8c7b4",
    metalness: 0,
    roughness: 0.92,
    opacity: surface.opacity,
    transparent: surface.transparent,
    alphaTest: surface.alphaTest,
    side: surface.side,
    vertexColors: "vertexColors" in surface && Boolean(surface.vertexColors)
  });
}

function celMaterial(surface: SurfaceMaterial) {
  return new ShaderMaterial({
    uniforms: {
      baseColor: { value: surface.color?.clone() ?? new Color("#ffffff") },
      colorMap: { value: surface.map ?? null },
      hasColorMap: { value: Boolean(surface.map) },
      opacity: { value: surface.opacity },
      alphaTestValue: { value: surface.alphaTest }
    },
    vertexShader: CEL_VERTEX_SHADER,
    fragmentShader: CEL_FRAGMENT_SHADER,
    transparent: surface.transparent,
    side: surface.side,
    toneMapped: true
  });
}

function outlineMaterial(geometry: Mesh["geometry"]) {
  geometry.computeBoundingSphere();
  const radius = geometry.boundingSphere?.radius ?? 1;
  return new ShaderMaterial({
    uniforms: {
      outlineWidth: { value: Math.max(radius * 0.012, 0.0001) },
      outlineColor: { value: new Color("#160f18") }
    },
    vertexShader: OUTLINE_VERTEX_SHADER,
    fragmentShader: OUTLINE_FRAGMENT_SHADER,
    side: BackSide,
    toneMapped: true
  });
}

function collectPreviewMaterials(object: Object3D) {
  const meshMaterials: MeshMaterials[] = [];
  const materials = new Set<Material>();
  const textures = new Set<Texture>();
  const generatedMaterials = new Set<Material>();

  object.traverse((node) => {
    if (!(node instanceof Mesh)) return;
    const original = node.material;
    for (const material of materialList(original)) {
      materials.add(material);
      const surface = material as SurfaceMaterial;
      for (const key of TEXTURE_KEYS) {
        const texture = surface[key];
        if (texture) textures.add(texture);
      }
    }

    const textured = mapMaterials(original, matteTexturedMaterial);
    const clay = mapMaterials(original, (surface) => {
      const material = clayMaterial(surface);
      generatedMaterials.add(material);
      return material;
    });
    const cel = mapMaterials(original, (surface) => {
      const material = celMaterial(surface);
      generatedMaterials.add(material);
      return material;
    });
    meshMaterials.push({ mesh: node, textured, clay, cel });
  });

  return { meshMaterials, materials, textures, generatedMaterials };
}

function applyAppearance(
  meshMaterials: MeshMaterials[],
  meshGeometries: MeshGeometries[],
  appearance: GltfPreviewAppearance
) {
  for (const entry of meshMaterials) {
    entry.mesh.material = entry[appearance];
  }
  for (const entry of meshGeometries) {
    entry.outline.visible = appearance === "cel";
  }
}

function applySmoothNormals(
  meshGeometries: MeshGeometries[],
  smoothNormals: boolean,
  appearance: GltfPreviewAppearance
) {
  const useSmoothGeometry = smoothNormals || appearance === "cel";
  for (const entry of meshGeometries) {
    entry.mesh.geometry = useSmoothGeometry ? entry.smooth : entry.original;
    entry.outline.geometry = entry.smooth;
  }
}

export function GltfPreviewViewer({
  modelUrl,
  autoRotate,
  smoothNormals,
  appearance,
  onTextureStatus,
  onError
}: GltfPreviewViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appearanceRef = useRef(appearance);
  const autoRotateRef = useRef(autoRotate);
  const smoothNormalsRef = useRef(smoothNormals);
  const meshMaterialsRef = useRef<MeshMaterials[]>([]);
  const meshGeometriesRef = useRef<MeshGeometries[]>([]);

  useEffect(() => {
    appearanceRef.current = appearance;
    applyAppearance(meshMaterialsRef.current, meshGeometriesRef.current, appearance);
    applySmoothNormals(meshGeometriesRef.current, smoothNormalsRef.current, appearance);
  }, [appearance]);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    smoothNormalsRef.current = smoothNormals;
    applySmoothNormals(meshGeometriesRef.current, smoothNormals, appearanceRef.current);
  }, [smoothNormals]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    let frame = 0;
    let loadedObject: Object3D | null = null;
    let sourceMaterials = new Set<Material>();
    let sourceTextures = new Set<Texture>();
    let generatedMaterials = new Set<Material>();
    meshMaterialsRef.current = [];
    meshGeometriesRef.current = [];
    onTextureStatus({ materialCount: 0, textureCount: 0 });
    onError(null);

    const scene = new Scene();
    scene.background = null;
    const camera = new PerspectiveCamera(32, 1, 0.01, 1000);
    const renderer = new WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.82;
    renderer.domElement.className = "gltf-preview-canvas";
    host.appendChild(renderer.domElement);

    scene.add(new AmbientLight("#fff4e7", 0.7));
    scene.add(new HemisphereLight("#d8fff9", "#2b1e19", 1.45));
    const keyLight = new DirectionalLight("#fff0dc", 2.2);
    keyLight.position.set(3.4, 4.8, 5.2);
    scene.add(keyLight);
    const fillLight = new DirectionalLight("#72d9d0", 0.55);
    fillLight.position.set(-4, 1.8, 2);
    scene.add(fillLight);

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

    new GLTFLoader().load(
      modelUrl,
      (gltf) => {
        if (disposed) return;
        loadedObject = gltf.scene;
        const box = new Box3().setFromObject(gltf.scene);
        const center = box.getCenter(new Vector3());
        const size = box.getSize(new Vector3());
        gltf.scene.position.sub(center);
        scene.add(gltf.scene);

        const largestSide = Math.max(0.01, size.x, size.y, size.z);
        camera.near = Math.max(0.001, largestSide / 200);
        camera.far = Math.max(100, largestSide * 30);
        camera.position.set(largestSide * 0.9, largestSide * 0.62, largestSide * 1.55);
        camera.updateProjectionMatrix();
        controls.update();

        const collected = collectPreviewMaterials(gltf.scene);
        const meshGeometries: MeshGeometries[] = [];
        for (const entry of collected.meshMaterials) {
          const node = entry.mesh;
          if (!node.geometry.getAttribute("normal")) {
            node.geometry.computeVertexNormals();
          }
          const smooth = createSmoothNormalsGeometry(node.geometry);
          const outlineSurface = outlineMaterial(smooth);
          const outline = new Mesh(smooth, outlineSurface);
          outline.visible = false;
          outline.frustumCulled = false;
          node.add(outline);
          collected.generatedMaterials.add(outlineSurface);
          meshGeometries.push({
            mesh: node,
            outline,
            original: node.geometry,
            smooth
          });
        }
        meshMaterialsRef.current = collected.meshMaterials;
        meshGeometriesRef.current = meshGeometries;
        sourceMaterials = collected.materials;
        sourceTextures = collected.textures;
        generatedMaterials = collected.generatedMaterials;
        onTextureStatus({
          materialCount: sourceMaterials.size,
          textureCount: sourceTextures.size
        });
        applyAppearance(collected.meshMaterials, meshGeometries, appearanceRef.current);
        applySmoothNormals(meshGeometries, smoothNormalsRef.current, appearanceRef.current);
      },
      undefined,
      (error) => {
        if (!disposed) onError(`GLB preview failed: ${String(error)}`);
      }
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
      const geometries = new Set<Mesh["geometry"]>();
      for (const entry of meshGeometriesRef.current) {
        geometries.add(entry.original);
        geometries.add(entry.smooth);
      }
      for (const geometry of geometries) geometry.dispose();
      for (const texture of sourceTextures) texture.dispose();
      for (const material of sourceMaterials) material.dispose();
      for (const material of generatedMaterials) material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      meshMaterialsRef.current = [];
      meshGeometriesRef.current = [];
    };
  }, [modelUrl, onError, onTextureStatus]);

  return <div ref={hostRef} className="gltf-preview-viewer" />;
}
