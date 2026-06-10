import "@google/model-viewer";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow, PhysicalSize } from "@tauri-apps/api/window";
import {
  Box,
  CheckCircle2,
  Cpu,
  Download,
  FolderOpen,
  Image,
  Layers3,
  Loader2,
  PackageOpen,
  Play,
  RefreshCw,
  RotateCcw,
  Settings2,
  Square,
  Trash2,
  TriangleAlert,
  WandSparkles,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type PipelineMode = "meshTexture" | "meshOnly" | "textureOnly";
type SeedMode = "fixed" | "randomize";
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
  url: string;
  openapiUrl: string;
  busy: boolean;
  activeJobId: string | null;
  message: string;
};

type HunyuanOptions = {
  mode: PipelineMode;
  cudaDevice: string;
  runtimeProfile: string;
  outputDir: string;
  outputName: string;
  meshInputPath: string;
  recursive: boolean;
  overwrite: boolean;
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
  forceOffload: boolean;
  removeFloaters: boolean;
  removeDegenerateFaces: boolean;
  reduceFaces: boolean;
  maxFaces: number;
  smoothNormals: boolean;
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
    | "model"
    | "runtime_ready"
    | "download_progress"
    | "stage_start"
    | "stage_done"
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
};

type PreviewImages = {
  input?: string;
  error?: string;
};

const APP_VERSION = "v26.6.15";
const SETTINGS_STORAGE_KEY = "imageToHunyuan3D.settings.v2";
const AGENT_STORAGE_KEY = "imageToHunyuan3D.agentControlEnabled.v1";

const defaultOptions: HunyuanOptions = {
  mode: "meshTexture",
  cudaDevice: "auto",
  runtimeProfile: "cu124",
  outputDir: "",
  outputName: "",
  meshInputPath: "",
  recursive: true,
  overwrite: false,
  meshSteps: 50,
  meshGuidance: 5,
  meshSeed: 2,
  meshSeedMode: "randomize",
  boxV: 1.01,
  octreeResolution: 384,
  numChunks: 8000,
  mcLevel: 0,
  mcAlgo: "mc",
  enableFlashVdm: false,
  forceOffload: false,
  removeFloaters: true,
  removeDegenerateFaces: true,
  reduceFaces: true,
  maxFaces: 15000,
  smoothNormals: true,
  textureViewSize: 512,
  textureSteps: 10,
  textureGuidance: 3,
  textureSize: 4096,
  textureSeed: 802344256,
  textureSeedMode: "randomize",
  unwrapMesh: true,
  cameraAzimuths: "0, 90, 180, 270, 0, 180",
  cameraElevations: "0, 0, 0, 0, 90, -90",
  viewWeights: "1, 0.1, 0.5, 0.1, 0.05, 0.05",
  orthoScale: 1.1
};

const isTauriRuntime = () => "__TAURI_INTERNALS__" in window || "__TAURI__" in window;
const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const isModelFile = (path: string) => /\.(glb|gltf|obj|ply)$/i.test(path);
const isPreviewableImage = (path: string) => /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(path);
const isTemporaryJobOutput = (path: string) => /[\\/]jobs[\\/]/i.test(path);

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

function coerceRuntimeProfile(value: unknown, fallback: string) {
  return value === "cu124" || value === "cu128" ? value : fallback;
}

function coerceSeedMode(value: unknown, fallback: SeedMode): SeedMode {
  return value === "fixed" || value === "randomize" ? value : fallback;
}

function loadSavedOptions(): HunyuanOptions {
  if (typeof window === "undefined") return defaultOptions;
  try {
    const saved = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!saved) return defaultOptions;
    const parsed = JSON.parse(saved) as Partial<HunyuanOptions>;
    return {
      mode: coerceMode(parsed.mode, defaultOptions.mode),
      cudaDevice: coerceString(parsed.cudaDevice, defaultOptions.cudaDevice),
      runtimeProfile: coerceRuntimeProfile(parsed.runtimeProfile, defaultOptions.runtimeProfile),
      outputDir: coerceString(parsed.outputDir, defaultOptions.outputDir),
      outputName: coerceString(parsed.outputName, defaultOptions.outputName),
      meshInputPath: coerceString(parsed.meshInputPath, defaultOptions.meshInputPath),
      recursive: coerceBoolean(parsed.recursive, defaultOptions.recursive),
      overwrite: coerceBoolean(parsed.overwrite, defaultOptions.overwrite),
      meshSteps: coerceNumber(parsed.meshSteps, defaultOptions.meshSteps, 1, 200),
      meshGuidance: coerceNumber(parsed.meshGuidance, defaultOptions.meshGuidance, 0, 30),
      meshSeed: coerceNumber(parsed.meshSeed, defaultOptions.meshSeed, 0, 999999999),
      meshSeedMode: coerceSeedMode(parsed.meshSeedMode, defaultOptions.meshSeedMode),
      boxV: coerceNumber(parsed.boxV, defaultOptions.boxV, 0.1, 5),
      octreeResolution: coerceNumber(parsed.octreeResolution, defaultOptions.octreeResolution, 64, 2048),
      numChunks: coerceNumber(parsed.numChunks, defaultOptions.numChunks, 1024, 999999),
      mcLevel: coerceNumber(parsed.mcLevel, defaultOptions.mcLevel, -10, 10),
      mcAlgo: coerceString(parsed.mcAlgo, defaultOptions.mcAlgo),
      enableFlashVdm: coerceBoolean(parsed.enableFlashVdm, defaultOptions.enableFlashVdm),
      forceOffload: coerceBoolean(parsed.forceOffload, defaultOptions.forceOffload),
      removeFloaters: coerceBoolean(parsed.removeFloaters, defaultOptions.removeFloaters),
      removeDegenerateFaces: coerceBoolean(
        parsed.removeDegenerateFaces,
        defaultOptions.removeDegenerateFaces
      ),
      reduceFaces: coerceBoolean(parsed.reduceFaces, defaultOptions.reduceFaces),
      maxFaces: coerceNumber(parsed.maxFaces, defaultOptions.maxFaces, 1000, 200000),
      smoothNormals: coerceBoolean(parsed.smoothNormals, defaultOptions.smoothNormals),
      textureViewSize: coerceNumber(parsed.textureViewSize, defaultOptions.textureViewSize, 256, 2048),
      textureSteps: coerceNumber(parsed.textureSteps, defaultOptions.textureSteps, 1, 100),
      textureGuidance: coerceNumber(parsed.textureGuidance, defaultOptions.textureGuidance, 0, 30),
      textureSize: coerceNumber(parsed.textureSize, defaultOptions.textureSize, 512, 8192),
      textureSeed: coerceNumber(parsed.textureSeed, defaultOptions.textureSeed, 0, 999999999),
      textureSeedMode: coerceSeedMode(parsed.textureSeedMode, defaultOptions.textureSeedMode),
      unwrapMesh: coerceBoolean(parsed.unwrapMesh, defaultOptions.unwrapMesh),
      cameraAzimuths: coerceString(parsed.cameraAzimuths, defaultOptions.cameraAzimuths),
      cameraElevations: coerceString(parsed.cameraElevations, defaultOptions.cameraElevations),
      viewWeights: coerceString(parsed.viewWeights, defaultOptions.viewWeights),
      orthoScale: coerceNumber(parsed.orthoScale, defaultOptions.orthoScale, 0.1, 5)
    };
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
  const models = paths?.filter(isModelFile) ?? [];
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

function runtimeReady(runtime: RuntimeStatus | null, options?: HunyuanOptions) {
  const profileReady = !options || runtime?.envRuntimeProfile === options.runtimeProfile;
  return Boolean(runtime?.envReady && runtime.sourceReady && runtime.modelsReady && profileReady);
}

function runtimeNeedsSetup(runtime: RuntimeStatus | null, options?: HunyuanOptions) {
  return Boolean(runtime && !runtimeReady(runtime, options));
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

function formatWorkerMessage(payload: WorkerEvent) {
  if (payload.type !== "download_progress") return payload.message;
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

function App() {
  const [options, setOptions] = useState<HunyuanOptions>(loadSavedOptions);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [vram, setVram] = useState<VramStatus | null>(null);
  const [agentControlEnabled, setAgentControlEnabled] = useState(
    () => window.localStorage.getItem(AGENT_STORAGE_KEY) === "1"
  );
  const [agentStatus, setAgentStatus] = useState<AgentServerStatus | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const queueRef = useRef<QueueItem[]>([]);
  const activeActionRef = useRef<ActiveAction>(null);
  const setupPromptSeenRef = useRef(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [previewImages, setPreviewImages] = useState<PreviewImages>({});
  const [busy, setBusy] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showSetupPrompt, setShowSetupPrompt] = useState(false);
  const [headline, setHeadline] = useState("Prepare runtime or drop images");
  const [log, setLog] = useState<string[]>([]);

  const vramDisplay = useMemo(() => formatVram(vram), [vram]);
  const completeCount = useMemo(() => queue.filter((item) => item.status === "done").length, [queue]);
  const errorCount = useMemo(() => queue.filter((item) => item.status === "error").length, [queue]);
  const selectedItem = useMemo(
    () => queue.find((item) => item.path === selectedPath) ?? queue[0],
    [queue, selectedPath]
  );
  const selectedModel = useMemo(
    () => choosePreviewModel(selectedItem?.outputHistory),
    [selectedItem]
  );
  const modelUrl = useMemo(
    () => (selectedModel && isTauriRuntime() ? convertFileSrc(selectedModel) : ""),
    [selectedModel]
  );
  const canStart = useMemo(() => {
    if (busy || queue.length === 0) return false;
    if (options.mode === "textureOnly" && !options.meshInputPath) return false;
    return true;
  }, [busy, options.meshInputPath, options.mode, queue.length]);

  const pushLog = useCallback((line: string) => {
    setLog((current) => [line, ...current].slice(0, 120));
  }, []);

  const refreshRuntime = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const status = await invoke<RuntimeStatus>("check_runtime");
    setRuntime(status);
    return status;
  }, []);

  const refreshVram = useCallback(async () => {
    if (!isTauriRuntime()) return;
    setVram(await invoke<VramStatus>("get_vram_status"));
  }, []);

  const refreshAgentStatus = useCallback(async () => {
    if (!isTauriRuntime()) return;
    setAgentStatus(await invoke<AgentServerStatus>("get_agent_server_status"));
  }, []);

  const toggleAgentControl = useCallback(
    async (enabled: boolean) => {
      setAgentControlEnabled(enabled);
      window.localStorage.setItem(AGENT_STORAGE_KEY, enabled ? "1" : "0");
      if (!isTauriRuntime()) return;
      try {
        const status = await invoke<AgentServerStatus>("set_agent_server_enabled", { enabled });
        setAgentStatus(status);
        pushLog(status.message);
      } catch (error) {
        setAgentControlEnabled(false);
        window.localStorage.setItem(AGENT_STORAGE_KEY, "0");
        pushLog(String(error));
      }
    },
    [pushLog]
  );

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
  }, [pushLog, refreshAgentStatus, refreshRuntime, refreshVram]);

  useEffect(() => {
    if (!isTauriRuntime() || !agentControlEnabled) return;
    void invoke<AgentServerStatus>("set_agent_server_enabled", { enabled: true })
      .then(setAgentStatus)
      .catch((error) => {
        setAgentControlEnabled(false);
        window.localStorage.setItem(AGENT_STORAGE_KEY, "0");
        pushLog(`Agent API failed to start: ${String(error)}`);
      });
  }, [agentControlEnabled, pushLog]);

  useEffect(() => {
    if (!runtimeNeedsSetup(runtime, options) || busy || setupPromptSeenRef.current) return;
    setupPromptSeenRef.current = true;
    setShowSetupPrompt(true);
  }, [busy, options, runtime]);

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
          void addInputs(event.payload.paths ?? []);
        }
      })
      .then((unlisten) => {
        cleanup = unlisten;
      });
    return () => cleanup?.();
  }, [addInputs]);

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
        setHeadline(payload.message ?? `Processing ${fileName(payload.path)}`);
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path ? { ...item, status: "running", message: "Running" } : item
          )
        );
      }
      if (payload.type === "stage_start" || payload.type === "model" || payload.type === "setup") {
        setHeadline(payload.message ?? "Working");
      }
      if (payload.type === "image_done" && payload.path) {
        const outputs = orderOutputPaths(parseOutputPaths(payload.output, payload.outputs));
        setQueue((current) =>
          current.map((item) =>
            item.path === payload.path
              ? {
                  ...item,
                  status: "done",
                  message: "Saved",
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
              ? { ...item, status: "error", message: payload.message ?? "Error" }
              : item
          )
        );
      }
      if (payload.type === "done" || payload.type === "runtime_ready") {
        setBusy(false);
        setCanceling(false);
        setHeadline(activeActionRef.current === "setup" ? "Runtime ready" : payload.message ?? "Batch complete");
        activeActionRef.current = null;
        void refreshRuntime().catch(() => undefined);
        void refreshVram().catch(() => undefined);
      }
      if (payload.type === "error" || payload.type === "canceled") {
        setBusy(false);
        setCanceling(false);
        activeActionRef.current = null;
        setHeadline(payload.type === "canceled" ? "Canceled" : "Worker error");
      }
    }).then((unlisten) => {
      cleanup = unlisten;
    });
    return () => cleanup?.();
  }, [pushLog, refreshRuntime, refreshVram]);

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
      "Reset the app Python environment? Models stay downloaded. The next Prepare will reinstall dependencies for the selected PyTorch runtime."
    );
    if (!confirmed) return;
    try {
      const status = await invoke<RuntimeStatus>("reset_python_runtime");
      setRuntime(status);
      pushLog("Python runtime reset. Choose Prepare to reinstall.");
      setHeadline("Python runtime reset");
    } catch (error) {
      pushLog(String(error));
    }
  }, [busy, pushLog]);

  const chooseMeshInput = useCallback(async () => {
    const selected = await openDialog({
      multiple: false,
      filters: [{ name: "3D Mesh", extensions: ["glb", "gltf", "obj", "ply"] }]
    });
    if (typeof selected === "string") {
      setOptions((current) => ({ ...current, meshInputPath: selected }));
    }
  }, []);

  const reset = useCallback(() => {
    setQueue([]);
    setSelectedPath(null);
    setPreviewImages({});
    setHeadline("Prepare runtime or drop images");
  }, []);

  const startRuntimeSetup = useCallback(async () => {
    if (!isTauriRuntime() || busy) return;
    setShowSetupPrompt(false);
    setBusy(true);
    setCanceling(false);
    activeActionRef.current = "setup";
    setHeadline("Preparing standalone runtime");
    try {
      await invoke("prepare_runtime", { options });
    } catch (error) {
      setBusy(false);
      activeActionRef.current = null;
      setHeadline("Could not prepare runtime");
      pushLog(String(error));
    }
  }, [busy, options, pushLog]);

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
    setQueue((current) => current.map((item) => ({ ...item, status: "pending", message: undefined })));
    setHeadline(runtimeReady(runtime, options) ? "Starting Hunyuan job" : "Preparing runtime first");
    try {
      const jobOptions = {
        ...options,
        outputDir: options.outputDir.trim() || runtime?.outputDir || ""
      };
      await invoke("start_hunyuan_job", {
        paths: queueRef.current.map((item) => item.path),
        options: jobOptions
      });
    } catch (error) {
      setBusy(false);
      activeActionRef.current = null;
      setHeadline("Could not start job");
      pushLog(String(error));
    }
  }, [canStart, options, pushLog, runtime]);

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
            <h1>Image To Hunyuan 3D</h1>
            <p>Standalone local Hunyuan3D 2.1 workstation</p>
          </section>

          <section className="panel">
            <div className="panel-title">
              <Settings2 size={16} />
              Pipeline
            </div>
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

            {options.mode === "textureOnly" ? (
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
            {agentStatus?.enabled ? (
              <div className="agent-status">
                <span>{agentStatus.url}</span>
                <small>OpenAPI: {agentStatus.openapiUrl}</small>
              </div>
            ) : null}
          </section>

          <section className="panel">
            <div className="panel-title">
              <Box size={16} />
              Mesh
            </div>
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
                    setOptions((current) => ({ ...current, meshSteps: Number(event.target.value) || current.meshSteps }))
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
                      octreeResolution: Number(event.target.value) || current.octreeResolution
                    }))
                  }
                />
              </label>
              <label className="field">
                <span>Chunks</span>
                <input
                  type="number"
                  min="1024"
                  max="999999"
                  value={options.numChunks}
                  disabled={options.mode === "textureOnly"}
                  onChange={(event) =>
                    setOptions((current) => ({ ...current, numChunks: Number(event.target.value) || current.numChunks }))
                  }
                />
              </label>
            </div>
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
          </section>

          <section className="panel">
            <div className="panel-title">
              <WandSparkles size={16} />
              Texture
            </div>
            <div className="split-fields">
              <label className="field">
                <span>View</span>
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
                <span>Texture</span>
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

          <div className="app-version">{APP_VERSION}</div>
        </aside>

        <section className="workspace">
          <header className="topbar">
            <div>
              <p className="eyebrow">Standalone NVIDIA pipeline</p>
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
                  <h3>Drop images or folders</h3>
                  <p>{selectedItem ? fileName(selectedItem.path) : "No input selected"}</p>
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
                  {options.mode === "meshOnly" ? "Mesh GLBs" : "Textured GLBs"} are generated by the app-managed Hunyuan runtime.
                </p>
              </div>
            </section>

            <section className="preview-panel">
              <div className="preview-head">
                <strong>3D Preview</strong>
                <small>{selectedModel ? fileName(selectedModel) : "No GLB selected"}</small>
              </div>
              <div className="model-frame">
                {modelUrl ? (
                  <model-viewer
                    key={modelUrl}
                    src={modelUrl}
                    camera-controls
                    auto-rotate
                    rotation-per-second="28deg"
                    shadow-intensity="0.8"
                    exposure="1"
                  />
                ) : (
                  <div className="preview-empty">
                    <Layers3 size={30} />
                    No GLB yet
                  </div>
                )}
              </div>
              {previewImages.error ? <small className="preview-error">{previewImages.error}</small> : null}
            </section>
          </div>

          <section className="queue">
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
                  onClick={() => setSelectedPath(item.path)}
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
                              title={path}
                              onClick={(event) => {
                                event.stopPropagation();
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
              <div className="runtime-status">
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
                <p className={runtime?.textureWheelsReady && runtime?.realesrganReady ? "ok" : "warn"}>
                  Texture: {runtime?.textureWheelsReady && runtime?.realesrganReady ? "ready" : "setup needed"}
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
              <h2 id="setup-title">Image To Hunyuan 3D needs local models</h2>
              <p>
                Before it can generate 3D models, the app needs to create its own local runtime and download the
                Hunyuan3D model files. Nothing starts downloading until you choose setup.
              </p>
              <div className="setup-runtime-options">
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
              <div className="setup-checks">
                <span className={runtime?.pythonFound ? "ok" : "warn"}>
                  Python: {runtime?.pythonFound ? "ready" : "needed"}
                </span>
                <span className={runtime?.envReady ? "ok" : "warn"}>
                  App runtime: {runtime?.envReady ? "ready" : "not set up"}
                </span>
                <span className={runtime?.sourceReady ? "ok" : "warn"}>
                  Hunyuan source: {runtime?.sourceReady ? "ready" : "download needed"}
                </span>
                <span className={runtime?.modelsReady ? "ok" : "warn"}>
                  Models: {runtime?.modelsReady ? "ready" : "download needed"}
                </span>
                <span className={runtime?.textureWheelsReady && runtime?.realesrganReady ? "ok" : "warn"}>
                  Texture tools: {runtime?.textureWheelsReady && runtime?.realesrganReady ? "ready" : "setup needed"}
                </span>
                <span className={runtime?.nvidiaGpuFound ? "ok" : "warn"}>
                  NVIDIA GPU: {runtime?.nvidiaGpuFound ? "detected" : "not detected"}
                </span>
              </div>
              <p className="setup-note">
                This is a large first-run download and install. It can take a while, and the app is designed for NVIDIA
                GPUs.
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
