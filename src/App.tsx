import "@google/model-viewer";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow, PhysicalSize } from "@tauri-apps/api/window";
import {
  Box,
  CheckCircle2,
  Clock3,
  Cpu,
  Download,
  FolderOpen,
  Image,
  Layers3,
  Loader2,
  PackageOpen,
  Play,
  Power,
  RefreshCw,
  RotateCcw,
  Settings2,
  Square,
  Sun,
  Trash2,
  TriangleAlert,
  WandSparkles,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parseVoxelSidecar, VoxelDssViewer, type VoxelSidecar } from "./VoxelDssViewer";

type PipelineMode = "meshTexture" | "meshOnly" | "textureOnly";
type ModelBackend = "hunyuan" | "trellis2";
type SeedMode = "fixed" | "randomize";
type MeshDetail = "low" | "medium" | "high" | "custom";
type TrellisPreset = "preview" | "standard" | "quality" | "highpoly" | "custom";
type QueueStatus = "pending" | "running" | "done" | "error" | "canceled";
type ActiveAction = "setup" | "generate" | null;

type RuntimeStatus = {
  pythonFound: boolean;
  pythonCommand: string | null;
  appDataDir: string;
  workerPath: string;
  envDir: string;
  envPython: string;
  envReady: boolean;
  sourceDir: string;
  sourceReady: boolean;
  modelDir: string;
  modelsReady: boolean;
  selectedBackend: ModelBackend;
  anyModelsReady: boolean;
  hunyuanEnvReady: boolean;
  hunyuanSourceReady: boolean;
  hunyuanModelsReady: boolean;
  trellisEnvReady: boolean;
  trellisSourceReady: boolean;
  trellisModelsReady: boolean;
  shapeModelReady: boolean;
  paintModelReady: boolean;
  textureWheelsReady: boolean;
  realesrganReady: boolean;
  envRuntimeProfile: string;
  nvidiaGpuFound: boolean;
  outputDir: string;
  jobsDir: string;
};

type GpuVram = {
  index: string;
  name: string;
  computeCap: string;
  memoryUsedMb: number;
  memoryTotalMb: number;
};

type VramStatus = {
  available: boolean;
  gpus: GpuVram[];
  message: string;
};

type AgentServerStatus = {
  enabled: boolean;
  port: number;
  url: string;
  openapiUrl: string;
  busy: boolean;
  activeJobId: string | null;
  message: string;
};

type ImageTo3DOptions = {
  modelBackend: ModelBackend;
  mode: PipelineMode;
  cudaDevice: string;
  runtimeProfile: string;
  outputDir: string;
  outputName: string;
  meshInputPath: string;
  recursive: boolean;
  overwrite: boolean;
  meshDetail: MeshDetail;
  meshSteps: number;
  meshGuidance: number;
  meshSeed: number;
  meshSeedMode: SeedMode;
  boxV: number;
  octreeResolution: number;
  numChunks: number;
  mcLevel: number;
  mcAlgo: string;
  enableFlashVdm: boolean;
  compileModels: boolean;
  forceOffload: boolean;
  removeFloaters: boolean;
  removeDegenerateFaces: boolean;
  reduceFaces: boolean;
  maxFaces: number;
  smoothNormals: boolean;
  voxelize: boolean;
  voxelResolution: number;
  textureViewSize: number;
  textureSteps: number;
  textureGuidance: number;
  textureSize: number;
  textureSeed: number;
  textureSeedMode: SeedMode;
  unwrapMesh: boolean;
  cameraAzimuths: string;
  cameraElevations: string;
  viewWeights: string;
  orthoScale: number;
  trellisPreset: TrellisPreset;
  trellisAutocrop: boolean;
  trellisForceOpaque: boolean;
  trellisAtlasPadding: number;
  trellisResolution: number;
  trellisSeed: number;
  trellisSeedMode: SeedMode;
  trellisSparseSteps: number;
  trellisSparseGuidance: number;
  trellisSparseRescale: number;
  trellisSparseRescaleT: number;
  trellisShapeSteps: number;
  trellisShapeGuidance: number;
  trellisShapeRescale: number;
  trellisShapeRescaleT: number;
  trellisTextureSteps: number;
  trellisTextureGuidance: number;
  trellisTextureRescale: number;
  trellisTextureRescaleT: number;
  trellisDecimationTarget: number;
  trellisTextureSize: number;
  trellisRemesh: boolean;
  trellisExportWebp: boolean;
};

type QueueItem = {
  path: string;
  status: QueueStatus;
  message?: string;
  outputHistory?: string[];
};

type WorkerEvent = {
  type:
    | "setup"
    | "daemon_ready"
    | "daemon_exit"
    | "model"
    | "runtime_ready"
    | "download_progress"
    | "stage_start"
    | "stage_done"
    | "stage_timing"
    | "mesh_ready"
    | "image_start"
    | "image_done"
    | "image_error"
    | "done"
    | "error"
    | "canceled";
  message?: string;
  name?: string;
  path?: string;
  output?: string;
  outputs?: string[];
  bytesDownloaded?: number;
  bytesTotal?: number | null;
  itemsDownloaded?: number;
  itemsTotal?: number | null;
  percent?: number;
  failedCount?: number;
  totalCount?: number;
  elapsedSeconds?: number;
};

type PreviewImages = {
  input?: string;
  error?: string;
};

const APP_VERSION = "v26.7.29";
const SETTINGS_STORAGE_KEY = "imageTo3D.settings.v2";
const PREVIOUS_SETTINGS_STORAGE_KEY = "imageTo3D.settings.v1";
const LEGACY_SETTINGS_STORAGE_KEY = "imageToHunyuan3D.settings.v3";
const OLDEST_SETTINGS_STORAGE_KEY = "imageToHunyuan3D.settings.v2";
const AGENT_STORAGE_KEY = "imageTo3D.agentControlEnabled.v1";
const LEGACY_AGENT_STORAGE_KEY = "imageToHunyuan3D.agentControlEnabled.v1";
const AGENT_PORT_STORAGE_KEY = "imageTo3D.agentApiPort.v1";
const LEGACY_AGENT_PORT_STORAGE_KEY = "imageToHunyuan3D.agentApiPort.v1";
const DEFAULT_AGENT_API_PORT = 17333;

const backendLabels: Record<ModelBackend, string> = {
  hunyuan: "Hunyuan3D 2.1",
  trellis2: "TRELLIS.2 4B"
};

type JobHistoryItem = {
  id: string;
  backend: ModelBackend;
  preset: string;
  status: string;
  message?: string | null;
  startedAt?: string | null;
  endedAt?: string | null;
  elapsedSeconds?: number | null;
  inputs: string[];
  outputs: string[];
  failedCount?: number | null;
  totalCount?: number | null;
};

type PresetAverage = {
  averageSeconds: number;
  samples: number;
};

type JobHistoryResponse = {
  jobs: JobHistoryItem[];
  presetAverages: Record<string, PresetAverage>;
};

const meshDetailPresets: Record<
  Exclude<MeshDetail, "custom">,
  Pick<ImageTo3DOptions, "meshSteps" | "octreeResolution" | "numChunks" | "maxFaces">
> = {
  low: {
    meshSteps: 24,
    octreeResolution: 256,
    numChunks: 6000,
    maxFaces: 5000
  },
  medium: {
    meshSteps: 50,
    octreeResolution: 384,
    numChunks: 8000,
    maxFaces: 20000
  },
  high: {
    meshSteps: 80,
    octreeResolution: 512,
    numChunks: 12000,
    maxFaces: 50000
  }
};

const meshDetailLabels: Record<MeshDetail, string> = {
  low: "Low poly (5k)",
  medium: "Medium (20k)",
  high: "High (50k)",
  custom: "Custom"
};

const trellisPresetLabels: Record<TrellisPreset, string> = {
  preview: "Preview / game-ready",
  standard: "Standard",
  quality: "Quality",
  highpoly: "High-poly",
  custom: "Custom"
};

const trellisPresets: Record<
  Exclude<TrellisPreset, "custom">,
  Pick<
    ImageTo3DOptions,
    | "trellisResolution"
    | "trellisSparseSteps"
    | "trellisShapeSteps"
    | "trellisTextureSteps"
    | "trellisTextureGuidance"
    | "trellisDecimationTarget"
    | "trellisTextureSize"
    | "trellisRemesh"
  >
> = {
  preview: {
    trellisResolution: 512,
    trellisSparseSteps: 12,
    trellisShapeSteps: 12,
    trellisTextureSteps: 12,
    trellisTextureGuidance: 3,
    trellisDecimationTarget: 20000,
    trellisTextureSize: 1024,
    trellisRemesh: false
  },
  standard: {
    trellisResolution: 1024,
    trellisSparseSteps: 12,
    trellisShapeSteps: 12,
    trellisTextureSteps: 12,
    trellisTextureGuidance: 7.5,
    trellisDecimationTarget: 100000,
    trellisTextureSize: 2048,
    trellisRemesh: false
  },
  quality: {
    trellisResolution: 1024,
    trellisSparseSteps: 25,
    trellisShapeSteps: 25,
    trellisTextureSteps: 25,
    trellisTextureGuidance: 7.5,
    trellisDecimationTarget: 200000,
    trellisTextureSize: 2048,
    trellisRemesh: false
  },
  highpoly: {
    trellisResolution: 1024,
    trellisSparseSteps: 25,
    trellisShapeSteps: 25,
    trellisTextureSteps: 25,
    trellisTextureGuidance: 7.5,
    trellisDecimationTarget: 500000,
    trellisTextureSize: 4096,
    trellisRemesh: false
  }
};

const defaultOptions: ImageTo3DOptions = {
  modelBackend: "hunyuan",
  mode: "meshTexture",
  cudaDevice: "auto",
  runtimeProfile: "cu124",
  outputDir: "",
  outputName: "",
  meshInputPath: "",
  recursive: true,
  overwrite: false,
  meshDetail: "low",
  meshSteps: 24,
  meshGuidance: 5,
  meshSeed: 2,
  meshSeedMode: "randomize",
  boxV: 1.01,
  octreeResolution: 256,
  numChunks: 6000,
  mcLevel: 0,
  mcAlgo: "mc",
  enableFlashVdm: true,
  compileModels: false,
  forceOffload: false,
  removeFloaters: true,
  removeDegenerateFaces: true,
  reduceFaces: true,
  maxFaces: 5000,
  smoothNormals: true,
  voxelize: false,
  voxelResolution: 64,
  textureViewSize: 512,
  textureSteps: 15,
  textureGuidance: 3,
  textureSize: 2048,
  textureSeed: 802344256,
  textureSeedMode: "randomize",
  unwrapMesh: true,
  cameraAzimuths: "0, 90, 180, 270, 0, 180",
  cameraElevations: "0, 0, 0, 0, 90, -90",
  viewWeights: "1, 0.1, 0.5, 0.1, 0.05, 0.05",
  orthoScale: 1.1,
  trellisPreset: "standard",
  trellisAutocrop: true,
  trellisForceOpaque: true,
  trellisAtlasPadding: 12,
  trellisResolution: 1024,
  trellisSeed: 12345,
  trellisSeedMode: "randomize",
  trellisSparseSteps: 12,
  trellisSparseGuidance: 7.5,
  trellisSparseRescale: 0.7,
  trellisSparseRescaleT: 5,
  trellisShapeSteps: 12,
  trellisShapeGuidance: 7.5,
  trellisShapeRescale: 0.5,
  trellisShapeRescaleT: 3,
  trellisTextureSteps: 12,
  trellisTextureGuidance: 7.5,
  trellisTextureRescale: 0.5,
  trellisTextureRescaleT: 3,
  trellisDecimationTarget: 100000,
  trellisTextureSize: 2048,
  trellisRemesh: false,
  trellisExportWebp: true
};

const isTauriRuntime = () => "__TAURI_INTERNALS__" in window || "__TAURI__" in window;
const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const isModelFile = (path: string) => /\.(glb|gltf|obj|ply)$/i.test(path);
const isPreviewableModelFile = (path: string) => /\.(glb|gltf)$/i.test(path);
const isPreviewableImage = (path: string) => /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(path);
const isTemporaryJobOutput = (path: string) => /[\\/]jobs[\\/]/i.test(path);

function normalizeAgentPort(value: string) {
  const port = Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

function coerceBoolean(value: unknown, fallback: boolean) {
  return typeof value === "boolean" ? value : fallback;
}

function coerceNumber(value: unknown, fallback: number, min: number, max: number) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(min, Math.min(max, value))
    : fallback;
}

function coerceString(value: unknown, fallback: string) {
  return typeof value === "string" ? value : fallback;
}

function coerceMode(value: unknown, fallback: PipelineMode): PipelineMode {
  return value === "meshTexture" || value === "meshOnly" || value === "textureOnly"
    ? value
    : fallback;
}

function coerceModelBackend(value: unknown, fallback: ModelBackend): ModelBackend {
  return value === "trellis2" || value === "hunyuan" ? value : fallback;
}

function coerceRuntimeProfile(value: unknown, fallback: string) {
  return value === "cu124" || value === "cu128" ? value : fallback;
}

function coerceSeedMode(value: unknown, fallback: SeedMode): SeedMode {
  return value === "fixed" || value === "randomize" ? value : fallback;
}

function coerceMeshDetail(value: unknown, fallback: MeshDetail): MeshDetail {
  return value === "low" || value === "medium" || value === "high" || value === "custom" ? value : fallback;
}

function coerceTrellisPreset(value: unknown, fallback: TrellisPreset): TrellisPreset {
  return value === "preview" ||
    value === "standard" ||
    value === "quality" ||
    value === "highpoly" ||
    value === "custom"
    ? value
    : fallback;
}

function detailFromMeshSettings(
  options: Pick<ImageTo3DOptions, "meshSteps" | "octreeResolution" | "numChunks" | "maxFaces">
): MeshDetail {
  const match = Object.entries(meshDetailPresets).find(
    ([, preset]) =>
      preset.meshSteps === options.meshSteps &&
      preset.octreeResolution === options.octreeResolution &&
      preset.numChunks === options.numChunks &&
      preset.maxFaces === options.maxFaces
  );
  return (match?.[0] as MeshDetail | undefined) ?? "custom";
}

function applyGeometryPreset(options: ImageTo3DOptions, meshDetail: MeshDetail): ImageTo3DOptions {
  return meshDetail === "custom"
    ? { ...options, meshDetail }
    : { ...options, meshDetail, ...meshDetailPresets[meshDetail] };
}

function applyTrellisPreset(options: ImageTo3DOptions, trellisPreset: TrellisPreset): ImageTo3DOptions {
  return trellisPreset === "custom"
    ? { ...options, trellisPreset }
    : { ...options, trellisPreset, ...trellisPresets[trellisPreset] };
}

function withCustomTrellisSettings(
  options: ImageTo3DOptions,
  changes: Partial<ImageTo3DOptions>
): ImageTo3DOptions {
  return { ...options, ...changes, trellisPreset: "custom" };
}

function loadSavedOptions(): ImageTo3DOptions {
  if (typeof window === "undefined") return defaultOptions;
  try {
    const saved = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    const previousSaved = saved === null ? window.localStorage.getItem(PREVIOUS_SETTINGS_STORAGE_KEY) : null;
    const savedFromPreviousSettingsKey = saved === null && previousSaved !== null;
    const savedFromLegacyKey = saved === null && previousSaved === null;
    const rawSaved =
      saved ??
      previousSaved ??
      window.localStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY) ??
      window.localStorage.getItem(OLDEST_SETTINGS_STORAGE_KEY);
    if (!rawSaved) return defaultOptions;
    const parsed = JSON.parse(rawSaved) as Partial<ImageTo3DOptions>;
    const meshSteps = coerceNumber(parsed.meshSteps, defaultOptions.meshSteps, 1, 200);
    const octreeResolution = coerceNumber(parsed.octreeResolution, defaultOptions.octreeResolution, 64, 2048);
    const numChunks = coerceNumber(parsed.numChunks, defaultOptions.numChunks, 1024, 999999);
    const maxFaces = coerceNumber(parsed.maxFaces, defaultOptions.maxFaces, 1000, 200000);
    const trellisDecimationTarget = coerceNumber(
      parsed.trellisDecimationTarget,
      defaultOptions.trellisDecimationTarget,
      1000,
      5000000
    );
    const inferredMeshDetail = detailFromMeshSettings({
      meshSteps,
      octreeResolution,
      numChunks,
      maxFaces
    });
    const loadedOptions: ImageTo3DOptions = {
      modelBackend: coerceModelBackend(parsed.modelBackend, defaultOptions.modelBackend),
      mode: coerceMode(parsed.mode, defaultOptions.mode),
      cudaDevice: coerceString(parsed.cudaDevice, defaultOptions.cudaDevice),
      runtimeProfile: coerceRuntimeProfile(parsed.runtimeProfile, defaultOptions.runtimeProfile),
      outputDir: coerceString(parsed.outputDir, defaultOptions.outputDir),
      outputName: coerceString(parsed.outputName, defaultOptions.outputName),
      meshInputPath: coerceString(parsed.meshInputPath, defaultOptions.meshInputPath),
      recursive: coerceBoolean(parsed.recursive, defaultOptions.recursive),
      overwrite: coerceBoolean(parsed.overwrite, defaultOptions.overwrite),
      meshDetail: coerceMeshDetail(parsed.meshDetail, inferredMeshDetail),
      meshSteps,
      meshGuidance: coerceNumber(parsed.meshGuidance, defaultOptions.meshGuidance, 0, 30),
      meshSeed: coerceNumber(parsed.meshSeed, defaultOptions.meshSeed, 0, 999999999),
      meshSeedMode: coerceSeedMode(parsed.meshSeedMode, defaultOptions.meshSeedMode),
      boxV: coerceNumber(parsed.boxV, defaultOptions.boxV, 0.1, 5),
      octreeResolution,
      numChunks,
      mcLevel: coerceNumber(parsed.mcLevel, defaultOptions.mcLevel, -10, 10),
      mcAlgo: coerceString(parsed.mcAlgo, defaultOptions.mcAlgo),
      enableFlashVdm: coerceBoolean(parsed.enableFlashVdm, defaultOptions.enableFlashVdm),
      compileModels: coerceBoolean(parsed.compileModels, defaultOptions.compileModels),
      forceOffload: coerceBoolean(parsed.forceOffload, defaultOptions.forceOffload),
      removeFloaters: coerceBoolean(parsed.removeFloaters, defaultOptions.removeFloaters),
      removeDegenerateFaces: coerceBoolean(
        parsed.removeDegenerateFaces,
        defaultOptions.removeDegenerateFaces
      ),
      reduceFaces: coerceBoolean(parsed.reduceFaces, defaultOptions.reduceFaces),
      maxFaces,
      smoothNormals: coerceBoolean(parsed.smoothNormals, defaultOptions.smoothNormals),
      voxelize: coerceBoolean(parsed.voxelize, defaultOptions.voxelize),
      voxelResolution: coerceNumber(parsed.voxelResolution, defaultOptions.voxelResolution, 8, 512),
      textureViewSize: coerceNumber(parsed.textureViewSize, defaultOptions.textureViewSize, 256, 2048),
      textureSteps: savedFromLegacyKey && parsed.textureSteps === 10
        ? defaultOptions.textureSteps
        : coerceNumber(parsed.textureSteps, defaultOptions.textureSteps, 1, 100),
      textureGuidance: coerceNumber(parsed.textureGuidance, defaultOptions.textureGuidance, 0, 30),
      textureSize: coerceNumber(parsed.textureSize, defaultOptions.textureSize, 512, 8192),
      textureSeed: coerceNumber(parsed.textureSeed, defaultOptions.textureSeed, 0, 999999999),
      textureSeedMode: coerceSeedMode(parsed.textureSeedMode, defaultOptions.textureSeedMode),
      unwrapMesh: coerceBoolean(parsed.unwrapMesh, defaultOptions.unwrapMesh),
      cameraAzimuths: coerceString(parsed.cameraAzimuths, defaultOptions.cameraAzimuths),
      cameraElevations: coerceString(parsed.cameraElevations, defaultOptions.cameraElevations),
      viewWeights: coerceString(parsed.viewWeights, defaultOptions.viewWeights),
      orthoScale: coerceNumber(parsed.orthoScale, defaultOptions.orthoScale, 0.1, 5),
      trellisPreset: coerceTrellisPreset(parsed.trellisPreset, defaultOptions.trellisPreset),
      trellisAutocrop: coerceBoolean(parsed.trellisAutocrop, defaultOptions.trellisAutocrop),
      trellisForceOpaque: coerceBoolean(parsed.trellisForceOpaque, defaultOptions.trellisForceOpaque),
      trellisAtlasPadding: coerceNumber(
        parsed.trellisAtlasPadding,
        defaultOptions.trellisAtlasPadding,
        0,
        64
      ),
      trellisResolution: coerceNumber(parsed.trellisResolution, defaultOptions.trellisResolution, 512, 1536),
      trellisSeed: coerceNumber(parsed.trellisSeed, defaultOptions.trellisSeed, 0, 999999999),
      trellisSeedMode: coerceSeedMode(parsed.trellisSeedMode, defaultOptions.trellisSeedMode),
      trellisSparseSteps: coerceNumber(parsed.trellisSparseSteps, defaultOptions.trellisSparseSteps, 1, 100),
      trellisSparseGuidance: coerceNumber(parsed.trellisSparseGuidance, defaultOptions.trellisSparseGuidance, 0, 30),
      trellisSparseRescale: coerceNumber(parsed.trellisSparseRescale, defaultOptions.trellisSparseRescale, 0, 2),
      trellisSparseRescaleT: coerceNumber(parsed.trellisSparseRescaleT, defaultOptions.trellisSparseRescaleT, 0, 10),
      trellisShapeSteps: coerceNumber(parsed.trellisShapeSteps, defaultOptions.trellisShapeSteps, 1, 100),
      trellisShapeGuidance: coerceNumber(parsed.trellisShapeGuidance, defaultOptions.trellisShapeGuidance, 0, 30),
      trellisShapeRescale: coerceNumber(parsed.trellisShapeRescale, defaultOptions.trellisShapeRescale, 0, 2),
      trellisShapeRescaleT: coerceNumber(parsed.trellisShapeRescaleT, defaultOptions.trellisShapeRescaleT, 0, 10),
      trellisTextureSteps: coerceNumber(parsed.trellisTextureSteps, defaultOptions.trellisTextureSteps, 1, 100),
      trellisTextureGuidance: coerceNumber(parsed.trellisTextureGuidance, defaultOptions.trellisTextureGuidance, 0, 30),
      trellisTextureRescale: coerceNumber(parsed.trellisTextureRescale, defaultOptions.trellisTextureRescale, 0, 2),
      trellisTextureRescaleT: coerceNumber(parsed.trellisTextureRescaleT, defaultOptions.trellisTextureRescaleT, 0, 10),
      trellisDecimationTarget,
      trellisTextureSize: coerceNumber(parsed.trellisTextureSize, defaultOptions.trellisTextureSize, 512, 8192),
      trellisRemesh: coerceBoolean(parsed.trellisRemesh, defaultOptions.trellisRemesh),
      trellisExportWebp: coerceBoolean(parsed.trellisExportWebp, defaultOptions.trellisExportWebp)
    };
    let migratedOptions = loadedOptions;
    if (
      savedFromPreviousSettingsKey &&
      (parsed.meshDetail === undefined || parsed.meshDetail === "medium") &&
      meshSteps === 50 &&
      octreeResolution === 384 &&
      numChunks === 8000 &&
      maxFaces === 15000 &&
      trellisDecimationTarget === 50000
    ) {
      migratedOptions = applyGeometryPreset(migratedOptions, "low");
    }
    if (parsed.trellisPreset === undefined) {
      migratedOptions = applyTrellisPreset(migratedOptions, defaultOptions.trellisPreset);
    }
    return migratedOptions;
  } catch {
    return defaultOptions;
  }
}

function formatVram(vram: VramStatus | null) {
  if (!vram?.available || vram.gpus.length === 0) {
    return { value: "NVIDIA", label: "not detected" };
  }
  const gpu = vram.gpus[0];
  return {
    value: `${Math.round((gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024)} GB`,
    label: `${gpu.name} free`
  };
}

function gpuOptionLabel(gpu: GpuVram) {
  return `${gpu.index}: ${gpu.name} sm_${gpu.computeCap.replace(".", "")}`;
}

function runtimeProfileLabel(profile: string) {
  if (profile === "cu128") return "CUDA 12.8 / RTX 50";
  if (profile === "cu124") return "CUDA 12.4 / stable";
  return profile || "not installed";
}

function gpuComputeCapability(gpu: GpuVram) {
  const value = Number(gpu.computeCap);
  return Number.isFinite(value) ? value : 0;
}

function isRtx50Gpu(gpu: GpuVram) {
  return gpuComputeCapability(gpu) >= 12 || /\bRTX\s*50|5090|5080|5070/i.test(gpu.name);
}

function selectedCudaGpu(vram: VramStatus | null, cudaDevice: string) {
  if (!vram?.available) return null;
  if (cudaDevice === "auto") return vram.gpus.find(isRtx50Gpu) ?? null;
  return vram.gpus.find((gpu) => gpu.index === cudaDevice) ?? null;
}

function statusIcon(status: QueueStatus) {
  if (status === "done") return <CheckCircle2 size={18} />;
  if (status === "error" || status === "canceled") return <XCircle size={18} />;
  if (status === "running") return <Loader2 className="spin" size={18} />;
  return <TriangleAlert size={18} />;
}

function parseOutputPaths(output?: string, outputs?: string[]) {
  if (outputs?.length) return outputs;
  return (
    output
      ?.split(";")
      .map((path) => path.trim())
      .filter(Boolean) ?? []
  );
}

function choosePreviewModel(paths?: string[]) {
  const models = paths?.filter(isPreviewableModelFile) ?? [];
  return models.find((path) => !isTemporaryJobOutput(path)) ?? models[0];
}

function orderOutputPaths(paths: string[]) {
  return [...paths].sort((left, right) => {
    const leftTemp = isTemporaryJobOutput(left) ? 1 : 0;
    const rightTemp = isTemporaryJobOutput(right) ? 1 : 0;
    if (leftTemp !== rightTemp) return leftTemp - rightTemp;
    const leftModel = isModelFile(left) ? 0 : 1;
    const rightModel = isModelFile(right) ? 0 : 1;
    return leftModel - rightModel;
  });
}

function runtimeReady(runtime: RuntimeStatus | null, options?: ImageTo3DOptions) {
  const profileReady = !options || runtime?.envRuntimeProfile === options.runtimeProfile;
  const backend = options?.modelBackend ?? runtime?.selectedBackend ?? "hunyuan";
  const backendReady = backend === "trellis2"
    ? runtime?.trellisEnvReady && runtime.trellisSourceReady && runtime.trellisModelsReady
    : runtime?.hunyuanEnvReady && runtime.hunyuanSourceReady && runtime.hunyuanModelsReady;
  const textureReady = backend !== "hunyuan" || !options || options.mode === "meshOnly"
    ? true
    : runtime?.textureWheelsReady && runtime.realesrganReady;
  return Boolean(runtime?.envReady && profileReady && backendReady && textureReady);
}

function runtimeNeedsSetup(runtime: RuntimeStatus | null, options?: ImageTo3DOptions) {
  return Boolean(runtime && !runtimeReady(runtime, options));
}

function backendRuntimeReady(runtime: RuntimeStatus | null, backend: ModelBackend) {
  return backend === "trellis2"
    ? Boolean(runtime?.trellisEnvReady && runtime.trellisSourceReady && runtime.trellisModelsReady)
    : Boolean(runtime?.hunyuanEnvReady && runtime.hunyuanSourceReady && runtime.hunyuanModelsReady);
}

function formatBytes(value?: number | null) {
  if (!value || value <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || size >= 10 ? 0 : 1;
  return `${size.toFixed(digits)} ${units[unit]}`;
}

function formatElapsedSeconds(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "not measured";
  const seconds = Math.round(value);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  }
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function formatHistoryDate(value?: string | null) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatWorkerMessage(payload: WorkerEvent) {
  if (payload.message && isTerminalProgressLine(payload.message)) return undefined;
  if (payload.type !== "download_progress") return cleanWorkerMessage(payload.message);
  const label = payload.message || (payload.name ? `Downloading ${payload.name}` : "Downloading");
  if (typeof payload.percent === "number" && payload.itemsTotal) {
    return `${label}: ${payload.percent}% (${payload.itemsDownloaded ?? 0} of ${payload.itemsTotal} files)`;
  }
  if (typeof payload.percent === "number" && typeof payload.itemsDownloaded === "number") {
    return `${label}: ${payload.percent}% (${payload.itemsDownloaded} files)`;
  }
  const downloaded = formatBytes(payload.bytesDownloaded);
  const total = formatBytes(payload.bytesTotal);
  if (typeof payload.percent === "number" && total) {
    return `${label}: ${payload.percent}% (${downloaded} of ${total})`;
  }
  if (typeof payload.percent === "number") {
    return `${label}: ${payload.percent}%`;
  }
  if (downloaded) {
    return `${label}: ${downloaded}`;
  }
  return label;
}

function cleanWorkerMessage(message?: string) {
  const stripped = message
    ?.replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, "")
    .replace(/\r/g, " ")
    .trim();
  return stripped || undefined;
}

function isTerminalProgressLine(message: string) {
  const stripped = message.replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, "");
  if (stripped.includes("\r") && (stripped.includes("%") || stripped.includes("|"))) return true;
  const compact = stripped.replace(/\s+/g, " ").trim();
  if (!compact) return true;
  const hasProgressShape = /\b\d+%\b|\b\d+\/\d+\b|\[[0-9:<>,.?it/s ]+\]/.test(compact) || compact.includes("|");
  return (
    hasProgressShape &&
    /^(Sampling|Simplifying|Building Sparse Grid|Adding clusters|Gathering results|xatlas:)/.test(compact)
  );
}

function App() {
  const [options, setOptions] = useState<ImageTo3DOptions>(loadSavedOptions);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [vram, setVram] = useState<VramStatus | null>(null);
  const [agentControlEnabled, setAgentControlEnabled] = useState(
    () => (window.localStorage.getItem(AGENT_STORAGE_KEY) ?? window.localStorage.getItem(LEGACY_AGENT_STORAGE_KEY)) === "1"
  );
  const [agentPort, setAgentPort] = useState(
    () => window.localStorage.getItem(AGENT_PORT_STORAGE_KEY) ?? window.localStorage.getItem(LEGACY_AGENT_PORT_STORAGE_KEY) ?? String(DEFAULT_AGENT_API_PORT)
  );
  const [agentStatus, setAgentStatus] = useState<AgentServerStatus | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const queueRef = useRef<QueueItem[]>([]);
  const activeActionRef = useRef<ActiveAction>(null);
  const setupPromptSeenRef = useRef(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [previewModelPath, setPreviewModelPath] = useState<string | null>(null);
  const [previewImages, setPreviewImages] = useState<PreviewImages>({});
  const [voxelSidecar, setVoxelSidecar] = useState<VoxelSidecar | null>(null);
  const [voxelPreviewError, setVoxelPreviewError] = useState<string | null>(null);
  const [dssPreviewEnabled, setDssPreviewEnabled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showSetupPrompt, setShowSetupPrompt] = useState(false);
  const [headline, setHeadline] = useState("Prepare runtime or drop files");
  const [log, setLog] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [jobHistory, setJobHistory] = useState<JobHistoryResponse>({
    jobs: [],
    presetAverages: {}
  });

  const vramDisplay = useMemo(() => formatVram(vram), [vram]);
  const rtx50RecommendationGpu = useMemo(
    () => selectedCudaGpu(vram, options.cudaDevice),
    [options.cudaDevice, vram]
  );
  const shouldRecommendCu128 = Boolean(rtx50RecommendationGpu && options.runtimeProfile !== "cu128");
  const selectedBackendLabel = backendLabels[options.modelBackend];
  const isHunyuanBackend = options.modelBackend === "hunyuan";
  const isTrellisBackend = options.modelBackend === "trellis2";
  const anyBackendInstalled = Boolean(runtime?.anyModelsReady || runtime?.hunyuanModelsReady || runtime?.trellisModelsReady);
  const completeCount = useMemo(() => queue.filter((item) => item.status === "done").length, [queue]);
  const errorCount = useMemo(() => queue.filter((item) => item.status === "error").length, [queue]);
  const selectedPresetAverage = isTrellisBackend
    ? jobHistory.presetAverages[`trellis2:${options.trellisPreset}`]
    : undefined;
  const selectedItem = useMemo(
    () => queue.find((item) => item.path === selectedPath) ?? queue[0],
    [queue, selectedPath]
  );
  const selectedModel = useMemo(
    () => previewModelPath ?? choosePreviewModel(selectedItem?.outputHistory),
    [previewModelPath, selectedItem]
  );
  const modelUrl = useMemo(
    () => (selectedModel && isTauriRuntime() ? convertFileSrc(selectedModel) : ""),
    [selectedModel]
  );
  const previewHint = selectedModel
    ? fileName(selectedModel)
    : busy
      ? "The mesh preview appears before texturing finishes"
      : "Drop a GLB to preview";
  const canUseDssPreview = Boolean(voxelSidecar);
  const showDssPreview = Boolean(modelUrl && dssPreviewEnabled && voxelSidecar);
  const canStart = useMemo(() => {
    if (busy || queue.length === 0) return false;
    if (options.modelBackend === "hunyuan" && options.mode === "textureOnly" && !options.meshInputPath) return false;
    return true;
  }, [busy, options.meshInputPath, options.mode, options.modelBackend, queue.length]);

  const pushLog = useCallback((line: string) => {
    setLog((current) => [line, ...current].slice(0, 120));
  }, []);

  const refreshRuntime = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const status = await invoke<RuntimeStatus>("check_runtime", {
      runtimeProfile: options.runtimeProfile,
      modelBackend: options.modelBackend
    });
    setRuntime(status);
    return status;
  }, [options.modelBackend, options.runtimeProfile]);

  const refreshVram = useCallback(async () => {
    if (!isTauriRuntime()) return;
    setVram(await invoke<VramStatus>("get_vram_status"));
  }, []);

  const refreshJobHistory = useCallback(async () => {
    if (!isTauriRuntime()) return;
    setJobHistory(await invoke<JobHistoryResponse>("list_job_history"));
  }, []);

  const refreshAgentStatus = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const status = await invoke<AgentServerStatus>("get_agent_server_status");
    setAgentStatus(status);
    if (status.enabled || !agentControlEnabled) {
      const nextPort = String(status.port || DEFAULT_AGENT_API_PORT);
      setAgentPort(nextPort);
      window.localStorage.setItem(AGENT_PORT_STORAGE_KEY, nextPort);
    }
  }, [agentControlEnabled]);

  const toggleAgentControl = useCallback(
    async (enabled: boolean) => {
      const port = normalizeAgentPort(agentPort);
      if (enabled && port === null) {
        pushLog("Choose an Agent API port between 1 and 65535.");
        return;
      }
      setAgentControlEnabled(enabled);
      window.localStorage.setItem(AGENT_STORAGE_KEY, enabled ? "1" : "0");
      if (port !== null) {
        window.localStorage.setItem(AGENT_PORT_STORAGE_KEY, String(port));
      }
      if (!isTauriRuntime()) return;
      try {
        const status = await invoke<AgentServerStatus>("set_agent_server_enabled", {
          enabled,
          port: port ?? agentStatus?.port ?? DEFAULT_AGENT_API_PORT
        });
        setAgentStatus(status);
        setAgentPort(String(status.port));
        pushLog(status.message);
      } catch (error) {
        setAgentControlEnabled(false);
        window.localStorage.setItem(AGENT_STORAGE_KEY, "0");
        pushLog(String(error));
      }
    },
    [agentPort, agentStatus?.port, pushLog]
  );

  const applyAgentPort = useCallback(async () => {
    const port = normalizeAgentPort(agentPort);
    if (port === null) {
      pushLog("Choose an Agent API port between 1 and 65535.");
      return;
    }
    window.localStorage.setItem(AGENT_PORT_STORAGE_KEY, String(port));
    if (!isTauriRuntime() || !agentControlEnabled) return;
    try {
      const status = await invoke<AgentServerStatus>("set_agent_server_enabled", {
        enabled: true,
        port
      });
      setAgentStatus(status);
      setAgentPort(String(status.port));
      pushLog(`Agent API moved to ${status.url}`);
    } catch (error) {
      pushLog(String(error));
    }
  }, [agentControlEnabled, agentPort, pushLog]);

  const addInputs = useCallback(
    async (paths: string[]) => {
      if (!isTauriRuntime() || paths.length === 0) return;
      const images = await invoke<string[]>("resolve_inputs", {
        paths,
        recursive: options.recursive
      });
      setQueue((current) => {
        const existing = new Set(current.map((item) => item.path));
        const additions = images
          .filter((path) => !existing.has(path))
          .map((path) => ({ path, status: "pending" as QueueStatus }));
        return [...current, ...additions];
      });
      if (!selectedPath && images.length > 0) {
        setSelectedPath(images[0]);
      }
      setHeadline(images.length === 1 ? `Queued ${fileName(images[0])}` : `Queued ${images.length} images`);
    },
    [options.recursive, selectedPath]
  );

  const previewModel = useCallback(
    (path: string, source: "dropped" | "selected" | "generated" = "selected") => {
      setPreviewModelPath(path);
      setHeadline(`Previewing ${fileName(path)}`);
      if (source === "dropped") {
        pushLog(`Previewing dropped model: ${path}`);
      }
    },
    [pushLog]
  );

  const handleDroppedPaths = useCallback(
    async (paths: string[]) => {
      if (!paths.length) return;
      const previewableModels = paths.filter(isPreviewableModelFile);
      const inputPaths = paths.filter((path) => !isPreviewableModelFile(path));

      if (previewableModels.length > 0) {
        previewModel(previewableModels[0], "dropped");
      }
      if (inputPaths.length > 0) {
        await addInputs(inputPaths);
      }
    },
    [addInputs, previewModel]
  );

  useEffect(() => {
    queueRef.current = queue;
  }, [queue]);

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(options));
  }, [options]);

  useEffect(() => {
    const updateScale = async () => {
      const applyScale = (width: number, height: number) => {
        const scale = Math.min(width / 1920, height / 1080);
        document.documentElement.style.setProperty("--ui-scale", String(scale));
      };
      if (isTauriRuntime()) {
        const size = await getCurrentWindow().innerSize();
        applyScale(size.width, size.height);
        const unlisten = await getCurrentWindow().onResized((event) => {
          const size = event.payload as PhysicalSize;
          applyScale(size.width, size.height);
        });
        return unlisten;
      }
      applyScale(window.innerWidth, window.innerHeight);
      const onResize = () => applyScale(window.innerWidth, window.innerHeight);
      window.addEventListener("resize", onResize);
      return () => window.removeEventListener("resize", onResize);
    };
    let cleanup: (() => void) | undefined;
    void updateScale().then((unlisten) => {
      cleanup = unlisten;
    });
    return () => cleanup?.();
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void refreshRuntime().catch((error) => pushLog(`Runtime check failed: ${String(error)}`));
    void refreshVram().catch((error) => pushLog(`VRAM check failed: ${String(error)}`));
    void refreshAgentStatus().catch((error) => pushLog(`Agent API check failed: ${String(error)}`));
    void refreshJobHistory().catch((error) => pushLog(`Job history check failed: ${String(error)}`));
  }, [pushLog, refreshAgentStatus, refreshJobHistory, refreshRuntime, refreshVram]);

  useEffect(() => {
    if (!isTauriRuntime() || !agentControlEnabled) return;
    const port = normalizeAgentPort(agentPort) ?? DEFAULT_AGENT_API_PORT;
    void invoke<AgentServerStatus>("set_agent_server_enabled", { enabled: true, port })
      .then((status) => {
        setAgentStatus(status);
        setAgentPort(String(status.port));
      })
      .catch((error) => {
        setAgentControlEnabled(false);
        window.localStorage.setItem(AGENT_STORAGE_KEY, "0");
        pushLog(`Agent API failed to start: ${String(error)}`);
      });
  }, [agentControlEnabled, pushLog]);

  useEffect(() => {
    if (!runtime || busy) return;
    if (anyBackendInstalled || setupPromptSeenRef.current) return;
    setupPromptSeenRef.current = true;
    setShowSetupPrompt(true);
  }, [anyBackendInstalled, busy, options.modelBackend, runtime]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    let cleanup: (() => void) | undefined;
    void getCurrentWindow()
      .onDragDropEvent((event: any) => {
        if (event.payload.type === "enter" || event.payload.type === "over") {
          setDragging(true);
        }
        if (event.payload.type === "leave") {
          setDragging(false);
        }
        if (event.payload.type === "drop") {
          setDragging(false);
          void handleDroppedPaths(event.payload.paths ?? []);
        }
      })
      .then((unlisten) => {
        cleanup = unlisten;
      });
    return () => cleanup?.();
  }, [handleDroppedPaths]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    let cleanup: (() => void) | undefined;
    void listen<WorkerEvent>("hunyuan-worker-event", (event) => {
      const payload = event.payload;
      const workerMessage = formatWorkerMessage(payload);
      if (workerMessage) {
        pushLog(workerMessage);
      }
      if (payload.type === "download_progress" && workerMessage) {
        setHeadline(workerMessage);
      }
      if (payload.type === "image_start" && payload.path) {
        setHeadline(workerMessage ?? `Processing ${fileName(payload.path)}`);
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path ? { ...item, status: "running", message: "Running" } : item
          )
        );
      }
      if ((payload.type === "stage_start" || payload.type === "model" || payload.type === "setup") && workerMessage) {
        setHeadline(workerMessage);
      }
      if (payload.type === "mesh_ready" && payload.path) {
        const preview = choosePreviewModel(parseOutputPaths(payload.output, payload.outputs));
        if (preview) {
          previewModel(preview, "generated");
        }
        setHeadline(workerMessage ?? `Mesh ready for ${fileName(payload.path)}. Generating texture...`);
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? { ...item, status: "running", message: "Mesh ready · generating texture" }
              : item
          )
        );
        setSelectedPath(payload.path);
      }
      if (payload.type === "image_done" && payload.path) {
        const outputs = orderOutputPaths(parseOutputPaths(payload.output, payload.outputs));
        const preview = choosePreviewModel(outputs);
        if (preview) {
          previewModel(preview, "generated");
        }
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? {
                  ...item,
                  status: "done",
                  message: workerMessage ?? payload.message ?? "Saved",
                  outputHistory: [...outputs, ...(item.outputHistory ?? [])]
                }
              : item
          )
        );
        setSelectedPath(payload.path);
      }
      if (payload.type === "image_error" && payload.path) {
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? { ...item, status: "error", message: workerMessage ?? payload.message ?? "Error" }
              : item
          )
        );
      }
      if (payload.type === "done" || payload.type === "runtime_ready") {
        setBusy(false);
        setCanceling(false);
        setHeadline(activeActionRef.current === "setup" ? "Runtime ready" : workerMessage ?? payload.message ?? "Batch complete");
        activeActionRef.current = null;
        void refreshRuntime().catch(() => undefined);
        void refreshVram().catch(() => undefined);
        void refreshJobHistory().catch(() => undefined);
      }
      if (payload.type === "error" || payload.type === "canceled") {
        setBusy(false);
        setCanceling(false);
        activeActionRef.current = null;
        setHeadline(payload.type === "canceled" ? "Canceled" : workerMessage ?? "Worker error");
        void refreshJobHistory().catch(() => undefined);
      }
    }).then((unlisten) => {
      cleanup = unlisten;
    });
    return () => cleanup?.();
  }, [previewModel, pushLog, refreshJobHistory, refreshRuntime, refreshVram]);

  useEffect(() => {
    if (!selectedItem || !isTauriRuntime()) {
      setPreviewImages({});
      return;
    }
    let canceled = false;
    setPreviewImages({});
    const loadPreview = async () => {
      try {
        const input = isPreviewableImage(selectedItem.path)
          ? await invoke<string>("preview_image_data_url", { path: selectedItem.path })
          : undefined;
        if (!canceled) setPreviewImages({ input });
      } catch (error) {
        if (!canceled) setPreviewImages({ error: String(error) });
      }
    };
    void loadPreview();
    return () => {
      canceled = true;
    };
  }, [selectedItem]);

  useEffect(() => {
    setVoxelSidecar(null);
    setVoxelPreviewError(null);
    if (!selectedModel || !isTauriRuntime()) {
      setDssPreviewEnabled(false);
      return;
    }
    let canceled = false;
    const loadSidecar = async () => {
      try {
        const raw = await invoke<string | null>("read_voxel_sidecar", { modelPath: selectedModel });
        if (canceled) return;
        if (!raw) {
          setDssPreviewEnabled(false);
          return;
        }
        const parsed = parseVoxelSidecar(raw);
        if (!parsed) {
          setVoxelPreviewError("Voxel preview data could not be read.");
          setDssPreviewEnabled(false);
          return;
        }
        setVoxelSidecar(parsed);
      } catch (error) {
        if (!canceled) {
          setVoxelPreviewError(String(error));
          setDssPreviewEnabled(false);
        }
      }
    };
    void loadSidecar();
    return () => {
      canceled = true;
    };
  }, [selectedModel]);

  const chooseInputs = useCallback(async () => {
    const selected = await openDialog({
      multiple: true,
      filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"] }]
    });
    if (Array.isArray(selected)) {
      await addInputs(selected);
    } else if (typeof selected === "string") {
      await addInputs([selected]);
    }
  }, [addInputs]);

  const chooseFolder = useCallback(async () => {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected === "string") {
      await addInputs([selected]);
    }
  }, [addInputs]);

  const chooseOutputDir = useCallback(async () => {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected === "string") {
      setOptions((current) => ({ ...current, outputDir: selected }));
    }
  }, []);

  const resetPythonRuntime = useCallback(async () => {
    if (!isTauriRuntime() || busy) return;
    const confirmed = window.confirm(
      "Reset the selected Python runtime? Models and outputs stay untouched. The next Prepare will reinstall dependencies for the selected PyTorch runtime."
    );
    if (!confirmed) return;
    try {
      const status = await invoke<RuntimeStatus>("reset_python_runtime", {
        runtimeProfile: options.runtimeProfile,
        modelBackend: options.modelBackend
      });
      setRuntime(status);
      pushLog(`${runtimeProfileLabel(options.runtimeProfile)} runtime reset. Choose Prepare to reinstall it.`);
      setHeadline("Python runtime reset");
    } catch (error) {
      pushLog(String(error));
    }
  }, [busy, options.modelBackend, options.runtimeProfile, pushLog]);

  const unloadModels = useCallback(async () => {
    if (!isTauriRuntime() || busy) return;
    try {
      const status = await invoke<RuntimeStatus>("unload_models", {
        runtimeProfile: options.runtimeProfile,
        modelBackend: options.modelBackend
      });
      setRuntime(status);
      void refreshVram().catch(() => undefined);
      setHeadline("Loaded models unloaded");
      pushLog(`Unloaded warm ${selectedBackendLabel} models. The next generation will reload them.`);
    } catch (error) {
      pushLog(String(error));
    }
  }, [busy, options.modelBackend, options.runtimeProfile, pushLog, refreshVram, selectedBackendLabel]);

  const chooseMeshInput = useCallback(async () => {
    const selected = await openDialog({
      multiple: false,
      filters: [{ name: "3D Mesh", extensions: ["glb", "gltf", "obj", "ply"] }]
    });
    if (typeof selected === "string") {
      setOptions((current) => ({ ...current, meshInputPath: selected }));
      if (isPreviewableModelFile(selected)) {
        previewModel(selected, "selected");
      }
    }
  }, [previewModel]);

  const reset = useCallback(() => {
    setQueue([]);
    setSelectedPath(null);
    setPreviewModelPath(null);
    setPreviewImages({});
    setHeadline("Prepare runtime or drop files");
  }, []);

  const startRuntimeSetup = useCallback(async () => {
    if (!isTauriRuntime() || busy) return;
    setShowSetupPrompt(false);
    setBusy(true);
    setCanceling(false);
    activeActionRef.current = "setup";
    setHeadline(`Preparing ${selectedBackendLabel}`);
    try {
      await invoke("prepare_runtime", { options });
    } catch (error) {
      setBusy(false);
      activeActionRef.current = null;
      setHeadline("Could not prepare runtime");
      pushLog(String(error));
    }
  }, [busy, options, pushLog, selectedBackendLabel]);

  const start = useCallback(async () => {
    if (!canStart || !isTauriRuntime()) return;
    if (!runtimeReady(runtime, options)) {
      setShowSetupPrompt(true);
      pushLog("Setup is required before generation.");
      return;
    }
    setBusy(true);
    setCanceling(false);
    activeActionRef.current = "generate";
    const startingMessage = runtimeReady(runtime, options)
      ? `Loading ${selectedBackendLabel} model`
      : "Preparing runtime first";
    setQueue((current) =>
      current.map((item, index) => ({
        ...item,
        status: index === 0 ? "running" : "pending",
        message: index === 0 ? startingMessage : undefined
      }))
    );
    setHeadline(startingMessage);
    try {
      const jobOptions = {
        ...options,
        outputDir: options.outputDir.trim() || runtime?.outputDir || ""
      };
      await invoke("start_generation_job", {
        paths: queueRef.current.map((item) => item.path),
        options: jobOptions
      });
    } catch (error) {
      setBusy(false);
      activeActionRef.current = null;
      setQueue((current) =>
        current.map((item) =>
          item.status === "running" ? { ...item, status: "pending", message: undefined } : item
        )
      );
      setHeadline("Could not start job");
      pushLog(String(error));
    }
  }, [canStart, options, pushLog, runtime, selectedBackendLabel]);

  const cancelActiveJob = useCallback(async () => {
    if (!isTauriRuntime()) return;
    setCanceling(true);
    try {
      await invoke("cancel_active_job");
    } catch (error) {
      setCanceling(false);
      pushLog(String(error));
    }
  }, [pushLog]);

  const openContainingFolder = useCallback(
    async (path: string) => {
      if (!isTauriRuntime()) return;
      try {
        await invoke("open_containing_folder", { path });
      } catch (error) {
        pushLog(String(error));
      }
    },
    [pushLog]
  );

  const openDirectPath = useCallback(
    async (path: string) => {
      if (!isTauriRuntime()) return;
      try {
        await invoke("open_path", { path });
      } catch (error) {
        pushLog(String(error));
      }
    },
    [pushLog]
  );

  const openOutputs = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const customOutput = options.outputDir.trim();
    try {
      if (customOutput) {
        await invoke("open_path", { path: customOutput });
      } else {
        await invoke("open_output_folder");
      }
    } catch (error) {
      pushLog(String(error));
    }
  }, [options.outputDir, pushLog]);

  return (
    <div className="scale-stage">
      <main className="app-shell">
        <aside className="sidebar">
          <section className="brand">
            <div className="brand-heading">
              <h1>Image To 3D</h1>
              <span className="app-version">{APP_VERSION}</span>
            </div>
            <p>Standalone local image-to-3D workstation</p>
          </section>

          <section className="panel">
            <div className="panel-title">
              <Settings2 size={16} />
              Pipeline
            </div>
            <label className="field">
              <span>Model</span>
              <select
                value={options.modelBackend}
                disabled={busy}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    modelBackend: event.target.value as ModelBackend,
                    mode: event.target.value === "trellis2" ? "meshTexture" : current.mode
                  }))
                }
              >
                <option value="hunyuan">Hunyuan3D 2.1 mesh/PBR</option>
                <option value="trellis2">TRELLIS.2 4B O-Voxel/PBR</option>
              </select>
            </label>
            {isHunyuanBackend ? (
            <div className="mode-tabs" role="tablist" aria-label="Pipeline mode">
              {[
                ["meshTexture", "Mesh + Texture"],
                ["meshOnly", "Mesh"],
                ["textureOnly", "Texture"]
              ].map(([value, label]) => (
                <button
                  key={value}
                  className={options.mode === value ? "active" : ""}
                  onClick={() => setOptions((current) => ({ ...current, mode: value as PipelineMode }))}
                >
                  {label}
                </button>
              ))}
            </div>
            ) : (
              <div className="backend-note">Single-image GLB with PBR materials</div>
            )}

            {isHunyuanBackend && options.mode === "textureOnly" ? (
              <div className="path-row">
                <label className="field">
                  <span>Source mesh</span>
                  <input
                    value={options.meshInputPath}
                    onChange={(event) =>
                      setOptions((current) => ({ ...current, meshInputPath: event.target.value }))
                    }
                  />
                </label>
                <button className="secondary icon small-icon" onClick={chooseMeshInput} title="Choose source mesh">
                  <PackageOpen size={16} />
                </button>
              </div>
            ) : null}

            <div className="path-row">
              <label className="field">
                <span>Output folder</span>
                <input
                  value={options.outputDir}
                  placeholder={runtime?.outputDir ?? "App outputs"}
                  onChange={(event) => setOptions((current) => ({ ...current, outputDir: event.target.value }))}
                />
              </label>
              <button className="secondary icon small-icon" onClick={chooseOutputDir} title="Choose output folder">
                <FolderOpen size={16} />
              </button>
            </div>

            <label className="field">
              <span>Output name</span>
              <input
                value={options.outputName}
                placeholder="Use image name"
                onChange={(event) => setOptions((current) => ({ ...current, outputName: event.target.value }))}
              />
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={options.recursive}
                onChange={(event) => setOptions((current) => ({ ...current, recursive: event.target.checked }))}
              />
              <span>Scan folders recursively</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.overwrite}
                onChange={(event) => setOptions((current) => ({ ...current, overwrite: event.target.checked }))}
              />
              <span>Overwrite outputs</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={agentControlEnabled}
                onChange={(event) => void toggleAgentControl(event.target.checked)}
              />
              <span>Agent API</span>
            </label>
            <label className="field agent-port-field">
              <span>API port</span>
              <input
                type="number"
                min="1"
                max="65535"
                value={agentPort}
                onChange={(event) => setAgentPort(event.target.value)}
                onBlur={() => void applyAgentPort()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.currentTarget.blur();
                  }
                }}
              />
            </label>
            {agentStatus?.enabled ? (
              <div className="agent-status">
                <span>{agentStatus.url}</span>
                <small>OpenAPI: {agentStatus.openapiUrl}</small>
              </div>
            ) : null}
          </section>

          {isHunyuanBackend ? (
          <>
          <section className="panel">
            <div className="panel-title">
              <Box size={16} />
              Mesh
            </div>
            <label className="field">
              <span>Geometry</span>
              <select
                value={options.meshDetail}
                disabled={options.mode === "textureOnly"}
                onChange={(event) => {
                  const meshDetail = event.target.value as MeshDetail;
                  setOptions((current) => applyGeometryPreset(current, meshDetail));
                }}
              >
                <option value="low">{meshDetailLabels.low}</option>
                <option value="medium">{meshDetailLabels.medium}</option>
                <option value="high">{meshDetailLabels.high}</option>
                <option value="custom">{meshDetailLabels.custom}</option>
              </select>
            </label>
            <div className="split-fields">
              <label className="field">
                <span>Steps</span>
                <input
                  type="number"
                  min="1"
                  max="200"
                  value={options.meshSteps}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      meshDetail: "custom",
                      meshSteps: Number(event.target.value) || current.meshSteps
                    }))
                  }
                />
              </label>
              <label className="field">
                <span>CFG</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.1"
                  value={options.meshGuidance}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      meshGuidance: Number(event.target.value) || current.meshGuidance
                    }))
                  }
                />
              </label>
            </div>
            <div className="split-fields">
              <label className="field">
                <span>Octree</span>
                <input
                  type="number"
                  min="64"
                  max="2048"
                  value={options.octreeResolution}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      meshDetail: "custom",
                      octreeResolution: Number(event.target.value) || current.octreeResolution
                    }))
                  }
                />
              </label>
              <label className="field">
                <span>Faces</span>
                <input
                  type="number"
                  min="1000"
                  max="200000"
                  step="1000"
                  value={options.maxFaces}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      meshDetail: "custom",
                      maxFaces: Number(event.target.value) || current.maxFaces
                    }))
                  }
                />
              </label>
            </div>
            <label className="field">
              <span>Chunks</span>
              <input
                type="number"
                min="1024"
                max="999999"
                value={options.numChunks}
                disabled={options.mode === "textureOnly"}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    meshDetail: "custom",
                    numChunks: Number(event.target.value) || current.numChunks
                  }))
                }
              />
            </label>
            <div className="seed-row">
              <label className="field">
                <span>Seed</span>
                <input
                  type="number"
                  value={options.meshSeed}
                  disabled={options.mode === "textureOnly" || options.meshSeedMode === "randomize"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, meshSeed: Number(event.target.value) || current.meshSeed }))
                  }
                />
              </label>
              <label className="field">
                <span>Mode</span>
                <select
                  value={options.meshSeedMode}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, meshSeedMode: event.target.value as SeedMode }))
                  }
                >
                  <option value="randomize">Random</option>
                  <option value="fixed">Fixed</option>
                </select>
              </label>
            </div>
            <label className="field">
              <span>Surface</span>
              <select
                value={options.mcAlgo}
                disabled={options.mode === "textureOnly"}
                onChange={(event) => setOptions((current) => ({ ...current, mcAlgo: event.target.value }))}
              >
                <option value="mc">Marching Cubes</option>
                <option value="dmc">DMC</option>
              </select>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.enableFlashVdm}
                disabled={options.mode === "textureOnly"}
                onChange={(event) => setOptions((current) => ({ ...current, enableFlashVdm: event.target.checked }))}
              />
              <span>Fast VAE mesh decode</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.compileModels}
                disabled={options.mode === "textureOnly"}
                onChange={(event) => setOptions((current) => ({ ...current, compileModels: event.target.checked }))}
              />
              <span>Compile shape model</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.voxelize}
                disabled={options.mode === "textureOnly"}
                onChange={(event) => setOptions((current) => ({ ...current, voxelize: event.target.checked }))}
              />
              <span>Voxel style mesh</span>
            </label>
            {options.voxelize ? (
              <label className="field">
                <span>Voxel resolution</span>
                <input
                  type="number"
                  min="8"
                  max="512"
                  value={options.voxelResolution}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      voxelResolution: Number(event.target.value) || current.voxelResolution
                    }))
                  }
                />
              </label>
            ) : null}
          </section>

          <section className="panel">
            <div className="panel-title">
              <WandSparkles size={16} />
              Texture
            </div>
            <div className="split-fields">
              <label className="field">
                <span>View resolution</span>
                <input
                  type="number"
                  min="256"
                  max="2048"
                  value={options.textureViewSize}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      textureViewSize: Number(event.target.value) || current.textureViewSize
                    }))
                  }
                />
              </label>
              <label className="field">
                <span>Final texture</span>
                <input
                  type="number"
                  min="512"
                  max="8192"
                  value={options.textureSize}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      textureSize: Number(event.target.value) || current.textureSize
                    }))
                  }
                />
              </label>
            </div>
            <small className="texture-resolution-help">
              View resolution is processed for every camera (six by default). 1024 uses 4× the
              pixels—and much more VRAM/time—than the recommended 512. Final texture controls the
              exported atlas size; 2048 is recommended.
            </small>
            <div className="split-fields">
              <label className="field">
                <span>Steps</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={options.textureSteps}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      textureSteps: Number(event.target.value) || current.textureSteps
                    }))
                  }
                />
              </label>
              <label className="field">
                <span>CFG</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.1"
                  value={options.textureGuidance}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({
                      ...current,
                      textureGuidance: Number(event.target.value) || current.textureGuidance
                    }))
                  }
                />
              </label>
            </div>
            <div className="seed-row">
              <label className="field">
                <span>Seed</span>
                <input
                  type="number"
                  value={options.textureSeed}
                  disabled={options.mode === "meshOnly" || options.textureSeedMode === "randomize"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, textureSeed: Number(event.target.value) || current.textureSeed }))
                  }
                />
              </label>
              <label className="field">
                <span>Mode</span>
                <select
                  value={options.textureSeedMode}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, textureSeedMode: event.target.value as SeedMode }))
                  }
                >
                  <option value="randomize">Random</option>
                  <option value="fixed">Fixed</option>
                </select>
              </label>
            </div>
            <div className="split-fields">
              <label className="field">
                <span>Views</span>
                <input
                  value={options.cameraAzimuths}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) => setOptions((current) => ({ ...current, cameraAzimuths: event.target.value }))}
                />
              </label>
              <label className="field">
                <span>Ortho</span>
                <input
                  type="number"
                  min="0.1"
                  max="5"
                  step="0.1"
                  value={options.orthoScale}
                  disabled={options.mode === "meshOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, orthoScale: Number(event.target.value) || current.orthoScale }))
                  }
                />
              </label>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.unwrapMesh}
                disabled={options.mode === "meshOnly"}
                onChange={(event) => setOptions((current) => ({ ...current, unwrapMesh: event.target.checked }))}
              />
              <span>UV unwrap before texture</span>
            </label>
          </section>
          </>
          ) : (
          <section className="panel">
            <div className="panel-title">
              <Box size={16} />
              TRELLIS.2
            </div>
            <label className="field">
              <span>Preset</span>
              <select
                value={options.trellisPreset}
                onChange={(event) =>
                  setOptions((current) =>
                    applyTrellisPreset(current, event.target.value as TrellisPreset)
                  )
                }
              >
                <option value="preview">{trellisPresetLabels.preview}</option>
                <option value="standard">{trellisPresetLabels.standard}</option>
                <option value="quality">{trellisPresetLabels.quality}</option>
                <option value="highpoly">{trellisPresetLabels.highpoly}</option>
                <option value="custom">{trellisPresetLabels.custom}</option>
              </select>
              <small className="preset-meta">
                {selectedPresetAverage
                  ? `Measured average ${formatElapsedSeconds(selectedPresetAverage.averageSeconds)} per asset across ${selectedPresetAverage.samples} run${selectedPresetAverage.samples === 1 ? "" : "s"}`
                  : "A measured average will appear after the first completed run."}
              </small>
            </label>
            <label className="field">
              <span>Resolution</span>
              <select
                value={options.trellisResolution}
                onChange={(event) =>
                  setOptions((current) =>
                    withCustomTrellisSettings(current, {
                      trellisResolution: Number(event.target.value) || 1024
                    })
                  )
                }
              >
                <option value={512}>512</option>
                <option value={1024}>1024 cascade</option>
                <option value={1536}>1536 cascade</option>
              </select>
            </label>
            <div className="seed-row">
              <label className="field">
                <span>Seed</span>
                <input
                  type="number"
                  value={options.trellisSeed}
                  disabled={options.trellisSeedMode === "randomize"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, trellisSeed: Number(event.target.value) || current.trellisSeed }))
                  }
                />
              </label>
              <label className="field">
                <span>Mode</span>
                <select
                  value={options.trellisSeedMode}
                  onChange={(event) => setOptions((current) => ({ ...current, trellisSeedMode: event.target.value as SeedMode }))}
                >
                  <option value="randomize">Random</option>
                  <option value="fixed">Fixed</option>
                </select>
              </label>
            </div>
            <div className="split-fields">
              <label className="field">
                <span>Sparse steps</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={options.trellisSparseSteps}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisSparseSteps: Number(event.target.value) || current.trellisSparseSteps
                      })
                    )
                  }
                />
              </label>
              <label className="field">
                <span>Sparse CFG</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.1"
                  value={options.trellisSparseGuidance}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisSparseGuidance: Number(event.target.value) || current.trellisSparseGuidance
                      })
                    )
                  }
                />
              </label>
            </div>
            <div className="split-fields">
              <label className="field">
                <span>Shape steps</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={options.trellisShapeSteps}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisShapeSteps: Number(event.target.value) || current.trellisShapeSteps
                      })
                    )
                  }
                />
              </label>
              <label className="field">
                <span>Shape CFG</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.1"
                  value={options.trellisShapeGuidance}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisShapeGuidance: Number(event.target.value) || current.trellisShapeGuidance
                      })
                    )
                  }
                />
              </label>
            </div>
            <div className="split-fields">
              <label className="field">
                <span>Texture steps</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={options.trellisTextureSteps}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisTextureSteps: Number(event.target.value) || current.trellisTextureSteps
                      })
                    )
                  }
                />
              </label>
              <label className="field">
                <span>Texture CFG</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.1"
                  value={options.trellisTextureGuidance}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisTextureGuidance: Number(event.target.value) || current.trellisTextureGuidance
                      })
                    )
                  }
                />
              </label>
            </div>
            <div className="split-fields">
              <label className="field">
                <span>Faces</span>
                <input
                  type="number"
                  min="1000"
                  max="5000000"
                  step="1000"
                  value={options.trellisDecimationTarget}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisDecimationTarget:
                          Number(event.target.value) || current.trellisDecimationTarget
                      })
                    )
                  }
                />
              </label>
              <label className="field">
                <span>Final texture</span>
                <input
                  type="number"
                  min="512"
                  max="8192"
                  value={options.trellisTextureSize}
                  onChange={(event) =>
                    setOptions((current) =>
                      withCustomTrellisSettings(current, {
                        trellisTextureSize: Number(event.target.value) || current.trellisTextureSize
                      })
                    )
                  }
                />
              </label>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.trellisRemesh}
                onChange={(event) =>
                  setOptions((current) =>
                    withCustomTrellisSettings(current, {
                      trellisRemesh: event.target.checked
                    })
                  )
                }
              />
              <span>Remesh GLB export</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.trellisAutocrop}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    trellisAutocrop: event.target.checked
                  }))
                }
              />
              <span>Crop subject before TRELLIS resize</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.trellisForceOpaque}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    trellisForceOpaque: event.target.checked
                  }))
                }
              />
              <span>Normalize materials as opaque</span>
            </label>
            <label className="field">
              <span>UV atlas padding</span>
              <input
                type="number"
                min="0"
                max="64"
                value={options.trellisAtlasPadding}
                onChange={(event) =>
                  setOptions((current) => ({
                    ...current,
                    trellisAtlasPadding:
                      Number(event.target.value) || (event.target.value === "0" ? 0 : current.trellisAtlasPadding)
                  }))
                }
              />
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={options.trellisExportWebp}
                onChange={(event) => setOptions((current) => ({ ...current, trellisExportWebp: event.target.checked }))}
              />
              <span>Use WebP textures</span>
            </label>
          </section>
          )}

        </aside>

        <section className="workspace">
          <header className="topbar">
            <div>
              <p className="eyebrow">{selectedBackendLabel}</p>
              <h2>{headline}</h2>
            </div>
            <div className="actions">
              <button className="secondary" onClick={chooseInputs}>
                <Image size={17} />
                Images
              </button>
              <button className="secondary" onClick={chooseFolder}>
                <FolderOpen size={17} />
                Folder
              </button>
              <button
                className={`secondary ${showHistory ? "active" : ""}`}
                onClick={() => setShowHistory((visible) => !visible)}
              >
                <Clock3 size={17} />
                {showHistory ? "Queue" : "History"}
              </button>
              <button className="secondary icon" onClick={reset} title="Clear queue">
                <RotateCcw size={17} />
              </button>
              <button className="primary" disabled={!canStart} onClick={() => void start()}>
                {busy && activeActionRef.current === "generate" ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
                Generate
              </button>
              <button className="secondary" disabled={!busy || canceling} onClick={cancelActiveJob}>
                {canceling ? <Loader2 className="spin" size={17} /> : <Square size={17} />}
                Cancel
              </button>
            </div>
          </header>

          <div className="stats">
            <div>
              <span>{queue.length}</span>
              queued
            </div>
            <div>
              <span>{completeCount}</span>
              saved
            </div>
            <div>
              <span>{errorCount}</span>
              failed
            </div>
            <div className="gpu-stat">
              <span className="vram-value">{vramDisplay.value}</span>
              <small>{vramDisplay.label}</small>
            </div>
          </div>

          <div className="media-grid">
            <section className={`input-panel ${dragging ? "dragging" : ""}`}>
              <div className="input-panel-copy">
                <Download size={24} />
                <div>
                  <h3>Drop images, folders, or GLBs</h3>
                  <p>
                    {selectedItem
                      ? fileName(selectedItem.path)
                      : selectedModel
                        ? fileName(selectedModel)
                        : "No input selected"}
                  </p>
                </div>
              </div>
              <div className="input-thumb">
                {previewImages.input ? (
                  <img src={previewImages.input} alt="" />
                ) : (
                  <div className="preview-empty compact">{selectedItem ? "Loading" : "Input"}</div>
                )}
              </div>
              <div className="input-panel-note">
                <p>
                  {isTrellisBackend
                    ? "Textured GLBs are generated by the app-managed TRELLIS.2 runtime."
                    : `${options.mode === "meshOnly" ? "Mesh GLBs" : "Textured GLBs"} are generated by the app-managed Hunyuan runtime.`}
                </p>
              </div>
            </section>

            <section className="preview-panel">
              <div className="preview-head">
                <div className="preview-title">
                  <strong>3D Preview</strong>
                  <small>{previewHint}</small>
                </div>
                {selectedModel ? (
                  <button
                    className={`secondary preview-toggle ${showDssPreview ? "active" : ""}`}
                    disabled={!canUseDssPreview}
                    onClick={() => setDssPreviewEnabled((enabled) => !enabled)}
                    title={canUseDssPreview ? "Toggle DSS lighting" : "DSS lighting is available for voxel-style outputs"}
                  >
                    <Sun size={15} />
                    DSS
                  </button>
                ) : null}
              </div>
              <div className="model-frame">
                {showDssPreview && voxelSidecar ? (
                  <VoxelDssViewer sidecar={voxelSidecar} autoRotate={!busy} />
                ) : modelUrl ? (
                  <model-viewer
                    key={modelUrl}
                    src={modelUrl}
                    camera-controls
                    auto-rotate={!busy}
                    rotation-per-second="28deg"
                    shadow-intensity="0.8"
                    exposure="1"
                  />
                ) : (
                  <div className="preview-empty">
                    <Layers3 size={30} />
                    {busy ? "Generating preview" : "Drop a GLB here"}
                  </div>
                )}
              </div>
              {previewImages.error || voxelPreviewError ? (
                <small className="preview-error">{previewImages.error ?? voxelPreviewError}</small>
              ) : null}
            </section>
          </div>

          <section className="queue">
            {showHistory ? (
              <>
                <div className="queue-header">
                  <span>Run</span>
                  <span>Status</span>
                  <span>Outputs</span>
                </div>
                {jobHistory.jobs.length === 0 ? (
                  <div className="empty">
                    <Clock3 size={18} />
                    Completed runs will appear here.
                  </div>
                ) : (
                  jobHistory.jobs.map((job) => (
                    <div className={`queue-row ${job.status}`} key={job.id}>
                      <div className="queue-input">
                        <strong>
                          {backendLabels[job.backend] ?? job.backend} ·{" "}
                          {trellisPresetLabels[job.preset as TrellisPreset] ?? job.preset}
                        </strong>
                        <small>
                          {formatHistoryDate(job.startedAt)} ·{" "}
                          {fileName(job.inputs[0] ?? job.id)}
                        </small>
                      </div>
                      <div className="status">
                        <Clock3 size={18} />
                        <div>
                          {job.status}
                          <small>{formatElapsedSeconds(job.elapsedSeconds)}</small>
                        </div>
                      </div>
                      <div className="output-history">
                        {job.outputs.length > 0 ? (
                          <>
                            <div className="output-files">
                              {job.outputs.map((path) => (
                                <button
                                  className={isModelFile(path) ? "output-chip model" : "output-chip"}
                                  key={path}
                                  title={path}
                                  onClick={() => {
                                    if (isPreviewableModelFile(path)) {
                                      previewModel(path, "selected");
                                    } else {
                                      void openDirectPath(path);
                                    }
                                  }}
                                >
                                  {isModelFile(path) ? <Layers3 size={14} /> : <PackageOpen size={14} />}
                                  {fileName(path)}
                                </button>
                              ))}
                            </div>
                            <button
                              className="secondary icon output-folder"
                              title="Open containing folder"
                              onClick={() => void openContainingFolder(job.outputs[0])}
                            >
                              <FolderOpen size={17} />
                            </button>
                          </>
                        ) : (
                          <span className="output-empty">{job.message ?? "No outputs"}</span>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </>
            ) : (
              <>
                <div className="queue-header">
                  <span>Input</span>
                  <span>Status</span>
                  <span>Outputs</span>
                </div>
                {queue.length === 0 ? (
                  <div className="empty">
                    <TriangleAlert size={18} />
                    No images queued.
                  </div>
                ) : (
                  queue.map((item) => (
                    <div
                      className={`queue-row ${item.status} ${selectedItem?.path === item.path ? "selected" : ""}`}
                      key={item.path}
                      onClick={() => {
                        setSelectedPath(item.path);
                        setPreviewModelPath(null);
                      }}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="queue-input">
                        <strong>{fileName(item.path)}</strong>
                        <small>{item.path}</small>
                      </div>
                      <div className="status">
                        {statusIcon(item.status)}
                        {item.message ?? item.status}
                      </div>
                      <div className="output-history">
                        {(item.outputHistory?.length ?? 0) > 0 ? (
                          <>
                            <div className="output-files">
                              {item.outputHistory?.map((path) => (
                                <button
                                  className={isModelFile(path) ? "output-chip model" : "output-chip"}
                                  key={path}
                                  title={isPreviewableModelFile(path) ? "Preview in 3D viewer" : path}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    if (isPreviewableModelFile(path)) {
                                      previewModel(path, "selected");
                                      return;
                                    }
                                    void openDirectPath(path);
                                  }}
                                >
                                  {isModelFile(path) ? <Layers3 size={14} /> : <PackageOpen size={14} />}
                                  {fileName(path)}
                                </button>
                              ))}
                            </div>
                            <button
                              className="secondary icon output-folder"
                              title="Open containing folder"
                              onClick={(event) => {
                                event.stopPropagation();
                                const first = item.outputHistory?.[0];
                                if (first) void openContainingFolder(first);
                              }}
                            >
                              <FolderOpen size={17} />
                            </button>
                          </>
                        ) : (
                          <span className="output-empty">Pending</span>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </>
            )}
          </section>

          <section className="runtime workspace-runtime">
            <div className="workspace-runtime-main">
              <div className="panel-title">
                <Cpu size={16} />
                Runtime
                <button
                  className="secondary icon mini-refresh"
                  title="Refresh runtime"
                  onClick={() => {
                    void refreshRuntime();
                    void refreshVram();
                  }}
                >
                  <RefreshCw size={14} />
                </button>
              </div>
              <div className="runtime-controls">
                <label className="field">
                  <span>CUDA GPU</span>
                  <select
                    value={options.cudaDevice}
                    disabled={busy}
                    onChange={(event) => setOptions((current) => ({ ...current, cudaDevice: event.target.value }))}
                  >
                    <option value="auto">Auto choose compatible GPU</option>
                    {vram?.gpus.map((gpu) => (
                      <option key={gpu.index} value={gpu.index}>
                        {gpuOptionLabel(gpu)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>PyTorch runtime</span>
                  <select
                    value={options.runtimeProfile}
                    disabled={busy}
                    onChange={(event) =>
                      setOptions((current) => ({ ...current, runtimeProfile: event.target.value }))
                    }
                  >
                    <option value="cu124">CUDA 12.4 / stable 30-40 series</option>
                    <option value="cu128">CUDA 12.8 / RTX 50 Blackwell</option>
                  </select>
                </label>
              </div>
              {shouldRecommendCu128 && rtx50RecommendationGpu ? (
                <div className="runtime-recommendation">
                  <span>
                    {rtx50RecommendationGpu.name} detected. CUDA 12.8 is recommended for RTX 50 GPUs.
                  </span>
                  <button
                    className="secondary mini-action"
                    disabled={busy}
                    onClick={() => setOptions((current) => ({ ...current, runtimeProfile: "cu128" }))}
                  >
                    Use CUDA 12.8
                  </button>
                </div>
              ) : null}
              <div className="runtime-status">
                <p className={backendRuntimeReady(runtime, options.modelBackend) ? "ok" : "warn"}>
                  Model: {selectedBackendLabel}
                </p>
                <p className={runtime?.nvidiaGpuFound ? "ok" : "warn"}>
                  NVIDIA: {runtime?.nvidiaGpuFound ? "detected" : "not detected"}
                </p>
                <p className={runtime?.envRuntimeProfile === options.runtimeProfile ? "ok" : "warn"}>
                  PyTorch: {runtimeProfileLabel(runtime?.envRuntimeProfile ?? "not installed")}
                </p>
                <p className={runtime?.pythonFound ? "ok" : "warn"}>
                  Python: {runtime?.pythonCommand ?? "missing"}
                </p>
                <p className={runtime?.envReady ? "ok" : "warn"}>
                  App env: {runtime?.envReady ? "ready" : "setup needed"}
                </p>
                <p className={runtime?.sourceReady ? "ok" : "warn"}>
                  Source: {runtime?.sourceReady ? "ready" : "not downloaded"}
                </p>
                <p className={runtime?.modelsReady ? "ok" : "warn"}>
                  Models: {runtime?.modelsReady ? "ready" : "not downloaded"}
                </p>
                {isHunyuanBackend ? (
                  <p className={runtime?.textureWheelsReady && runtime?.realesrganReady ? "ok" : "warn"}>
                    Texture: {runtime?.textureWheelsReady && runtime?.realesrganReady ? "ready" : "setup needed"}
                  </p>
                ) : null}
                <p className={runtime?.hunyuanModelsReady ? "ok" : "warn"}>
                  Hunyuan: {runtime?.hunyuanModelsReady ? "installed" : "not installed"}
                </p>
                <p className={runtime?.trellisModelsReady ? "ok" : "warn"}>
                  TRELLIS.2: {runtime?.trellisModelsReady ? "installed" : "not installed"}
                </p>
              </div>
            </div>
            <div className="runtime-actions">
              <button className="primary runtime-button" disabled={busy} onClick={() => setShowSetupPrompt(true)}>
                {busy && activeActionRef.current === "setup" ? <Loader2 className="spin" size={16} /> : <Download size={16} />}
                Prepare
              </button>
              <button
                className="secondary runtime-button"
                onClick={() => void invoke("open_runtime_folder").catch(pushLog)}
              >
                <FolderOpen size={16} />
                Runtime
              </button>
              <button
                className="secondary runtime-button"
                onClick={() => void openOutputs()}
              >
                <PackageOpen size={16} />
                Outputs
              </button>
              <button
                className="secondary runtime-button"
                disabled={busy}
                onClick={() => void unloadModels()}
              >
                <Power size={16} />
                Unload
              </button>
              <button
                className="secondary runtime-button"
                disabled={busy}
                onClick={() => void resetPythonRuntime()}
              >
                <Trash2 size={16} />
                Reset Env
              </button>
            </div>
          </section>

          <section className="log">
            {log.length === 0 ? <p>Runtime messages will appear here.</p> : log.map((line, index) => <p key={`${line}-${index}`}>{line}</p>)}
          </section>
        </section>
      </main>

      {showSetupPrompt ? (
        <div className="modal-backdrop" role="presentation">
          <section className="setup-modal" role="dialog" aria-modal="true" aria-labelledby="setup-title">
            <div className="setup-modal-icon">
              <Download size={28} />
            </div>
            <div className="setup-modal-copy">
              <p className="eyebrow">One-time setup</p>
              <h2 id="setup-title">Image To 3D needs local models</h2>
              <p>
                Before it can generate 3D models, the app needs to create a local runtime and download the selected
                model files. Nothing starts downloading until you choose setup.
              </p>
              <div className="setup-runtime-options">
                <label className="field">
                  <span>Model</span>
                  <select
                    value={options.modelBackend}
                    disabled={busy}
                    onChange={(event) =>
                      setOptions((current) => ({
                        ...current,
                        modelBackend: event.target.value as ModelBackend,
                        mode: event.target.value === "trellis2" ? "meshTexture" : current.mode
                      }))
                    }
                  >
                    <option value="hunyuan">Hunyuan3D 2.1</option>
                    <option value="trellis2">TRELLIS.2 4B</option>
                  </select>
                </label>
                <label className="field">
                  <span>CUDA GPU</span>
                  <select
                    value={options.cudaDevice}
                    disabled={busy}
                    onChange={(event) => setOptions((current) => ({ ...current, cudaDevice: event.target.value }))}
                  >
                    <option value="auto">Auto choose compatible GPU</option>
                    {vram?.gpus.map((gpu) => (
                      <option key={gpu.index} value={gpu.index}>
                        {gpuOptionLabel(gpu)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>PyTorch runtime</span>
                  <select
                    value={options.runtimeProfile}
                    disabled={busy}
                    onChange={(event) =>
                      setOptions((current) => ({ ...current, runtimeProfile: event.target.value }))
                    }
                  >
                    <option value="cu124">CUDA 12.4 / stable 30-40 series</option>
                    <option value="cu128">CUDA 12.8 / RTX 50 Blackwell</option>
                  </select>
                </label>
              </div>
              {shouldRecommendCu128 && rtx50RecommendationGpu ? (
                <div className="runtime-recommendation setup-recommendation">
                  <span>
                    {rtx50RecommendationGpu.name} detected. CUDA 12.8 is recommended for RTX 50 GPUs.
                  </span>
                  <button
                    className="secondary mini-action"
                    disabled={busy}
                    onClick={() => setOptions((current) => ({ ...current, runtimeProfile: "cu128" }))}
                  >
                    Use CUDA 12.8
                  </button>
                </div>
              ) : null}
              <div className="setup-checks">
                <span className={runtime?.pythonFound ? "ok" : "warn"}>
                  Python: {runtime?.pythonFound ? "ready" : "needed"}
                </span>
                <span className={runtime?.envReady ? "ok" : "warn"}>
                  App runtime: {runtime?.envReady ? "ready" : "not set up"}
                </span>
                <span className={runtime?.sourceReady ? "ok" : "warn"}>
                  {selectedBackendLabel} source: {runtime?.sourceReady ? "ready" : "download needed"}
                </span>
                <span className={runtime?.modelsReady ? "ok" : "warn"}>
                  {selectedBackendLabel} models: {runtime?.modelsReady ? "ready" : "download needed"}
                </span>
                {isHunyuanBackend ? (
                <span className={runtime?.textureWheelsReady && runtime?.realesrganReady ? "ok" : "warn"}>
                  Texture tools: {runtime?.textureWheelsReady && runtime?.realesrganReady ? "ready" : "setup needed"}
                </span>
                ) : null}
                <span className={runtime?.nvidiaGpuFound ? "ok" : "warn"}>
                  NVIDIA GPU: {runtime?.nvidiaGpuFound ? "detected" : "not detected"}
                </span>
              </div>
              <p className="setup-note">
                This is a large first-run download and install. TRELLIS.2 currently follows Microsoft&apos;s Linux/CUDA
                runtime requirements and may need build tools for native extensions.
              </p>
            </div>
            <div className="setup-actions">
              <button className="secondary" onClick={() => setShowSetupPrompt(false)}>
                Not now
              </button>
              <button
                className="primary"
                disabled={busy || runtime?.pythonFound === false}
                onClick={() => void startRuntimeSetup()}
              >
                {busy && activeActionRef.current === "setup" ? <Loader2 className="spin" size={17} /> : <Download size={17} />}
                Set up now
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

export default App;
