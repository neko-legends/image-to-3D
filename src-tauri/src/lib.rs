use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env, fs,
    io::{BufRead, BufReader, Cursor, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;
use uuid::Uuid;

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

mod autoremesher;
mod child_env;
mod neko_store;
use autoremesher::{ActiveRemeshState, RemeshOptions, RemeshResult, RemesherStatus};
use neko_store::NekoStore;

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "jfif", "jpf", "jp2", "j2k", "webp", "bmp", "tif", "tiff",
];
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const EMBEDDED_WORKER: &str = include_str!("../../python/hunyuan_worker.py");
const ENV_MARKER: &str = ".ih3d-env-v4-ready";
const HUNYUAN_SOURCE_MARKER: &str = ".ih3d-source-v1-ready";
const HUNYUAN_MODEL_MARKER: &str = ".ih3d-models-v1-ready";
const TRELLIS_SOURCE_MARKER: &str = ".i3d-trellis-source-v1-ready";
const TRELLIS_MODEL_MARKER: &str = ".i3d-trellis-models-v3-ready";
const HUNYUAN_ENV_MARKER: &str = ".i3d-hunyuan-env-v2-ready";
const TRELLIS_ENV_MARKER: &str = ".i3d-trellis-env-v3-ready";
const WHEEL_MARKER: &str = ".ih3d-texture-wheels-v1-ready";
const DEFAULT_PYTORCH_PROFILE: &str = "cu124";
const DEFAULT_AGENT_API_PORT: u16 = 17333;
const AGENT_APP_ID: &str = "image-to-3d";
const AGENT_APP_NAME: &str = "ImageTo3D";
const AGENT_API_BIND_ADDRESS: &str = "127.0.0.1";
const AGENT_API_REGISTRY_FILE: &str = "agent-api-registry.json";
const DEFAULT_WINDOW_WIDTH: u32 = 1920;
const DEFAULT_WINDOW_HEIGHT: u32 = 1080;
const MIN_WINDOW_WIDTH: u32 = 640;
const MIN_WINDOW_HEIGHT: u32 = 360;
const LANDSCAPE_WINDOW_WIDTH_RATIO: f64 = 0.5;
const PORTRAIT_WINDOW_WIDTH_RATIO: f64 = 0.9;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStatus {
    python_found: bool,
    python_command: Option<String>,
    app_data_dir: String,
    worker_path: String,
    env_dir: String,
    env_python: String,
    env_ready: bool,
    source_dir: String,
    source_ready: bool,
    model_dir: String,
    models_ready: bool,
    selected_backend: String,
    any_models_ready: bool,
    hunyuan_env_ready: bool,
    hunyuan_source_ready: bool,
    hunyuan_models_ready: bool,
    trellis_env_ready: bool,
    trellis_source_ready: bool,
    trellis_models_ready: bool,
    shape_model_ready: bool,
    paint_model_ready: bool,
    texture_wheels_ready: bool,
    realesrgan_ready: bool,
    env_runtime_profile: String,
    nvidia_gpu_found: bool,
    output_dir: String,
    jobs_dir: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuVram {
    index: String,
    name: String,
    compute_cap: String,
    memory_used_mb: u64,
    memory_total_mb: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VramStatus {
    available: bool,
    gpus: Vec<GpuVram>,
    message: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct HunyuanOptions {
    #[serde(default = "default_model_backend")]
    model_backend: String,
    mode: String,
    cuda_device: String,
    runtime_profile: String,
    output_dir: String,
    output_name: String,
    mesh_input_path: String,
    #[serde(default)]
    model_dir: String,
    #[serde(default)]
    env_dir: String,
    #[serde(default)]
    runtime_dir: String,
    recursive: bool,
    overwrite: bool,
    mesh_detail: String,
    mesh_steps: u32,
    mesh_guidance: f32,
    mesh_seed: u64,
    mesh_seed_mode: String,
    box_v: f32,
    octree_resolution: u32,
    num_chunks: u32,
    mc_level: f32,
    mc_algo: String,
    enable_flash_vdm: bool,
    compile_models: bool,
    force_offload: bool,
    remove_floaters: bool,
    remove_degenerate_faces: bool,
    reduce_faces: bool,
    max_faces: u32,
    smooth_normals: bool,
    #[serde(default)]
    voxelize: bool,
    #[serde(default = "default_voxel_resolution")]
    voxel_resolution: u32,
    texture_view_size: u32,
    texture_steps: u32,
    texture_guidance: f32,
    texture_size: u32,
    texture_seed: u64,
    texture_seed_mode: String,
    unwrap_mesh: bool,
    camera_azimuths: String,
    camera_elevations: String,
    view_weights: String,
    ortho_scale: f32,
    #[serde(default = "default_trellis_preset")]
    trellis_preset: String,
    #[serde(default = "default_true")]
    trellis_autocrop: bool,
    #[serde(default = "default_true")]
    trellis_force_opaque: bool,
    #[serde(default = "default_trellis_atlas_padding")]
    trellis_atlas_padding: u32,
    #[serde(default = "default_trellis_resolution")]
    trellis_resolution: u32,
    #[serde(default = "default_trellis_seed")]
    trellis_seed: u64,
    #[serde(default = "default_randomize_seed_mode")]
    trellis_seed_mode: String,
    #[serde(default = "default_trellis_sparse_steps")]
    trellis_sparse_steps: u32,
    #[serde(default = "default_trellis_sparse_guidance")]
    trellis_sparse_guidance: f32,
    #[serde(default = "default_trellis_sparse_rescale")]
    trellis_sparse_rescale: f32,
    #[serde(default = "default_trellis_sparse_rescale_t")]
    trellis_sparse_rescale_t: f32,
    #[serde(default = "default_trellis_shape_steps")]
    trellis_shape_steps: u32,
    #[serde(default = "default_trellis_shape_guidance")]
    trellis_shape_guidance: f32,
    #[serde(default = "default_trellis_shape_rescale")]
    trellis_shape_rescale: f32,
    #[serde(default = "default_trellis_shape_rescale_t")]
    trellis_shape_rescale_t: f32,
    #[serde(default = "default_trellis_texture_steps")]
    trellis_texture_steps: u32,
    #[serde(default = "default_trellis_texture_guidance")]
    trellis_texture_guidance: f32,
    #[serde(default = "default_trellis_texture_rescale")]
    trellis_texture_rescale: f32,
    #[serde(default = "default_trellis_texture_rescale_t")]
    trellis_texture_rescale_t: f32,
    #[serde(default = "default_trellis_decimation_target")]
    trellis_decimation_target: u32,
    #[serde(default = "default_trellis_texture_size")]
    trellis_texture_size: u32,
    #[serde(default)]
    trellis_remesh: bool,
    #[serde(default = "default_true")]
    trellis_export_webp: bool,
}

impl Default for HunyuanOptions {
    /// Structural default. Used to build a minimal options value (e.g. for
    /// `check_runtime`) where only a subset of fields matter. Individual command
    /// paths always deserialize a full options object from the frontend, so this
    /// never shadows real user settings.
    fn default() -> Self {
        HunyuanOptions {
            model_backend: default_model_backend(),
            mode: "meshTexture".to_string(),
            cuda_device: "auto".to_string(),
            runtime_profile: DEFAULT_PYTORCH_PROFILE.to_string(),
            output_dir: String::new(),
            output_name: String::new(),
            mesh_input_path: String::new(),
            model_dir: String::new(),
            env_dir: String::new(),
            runtime_dir: String::new(),
            recursive: false,
            overwrite: false,
            mesh_detail: "medium".to_string(),
            mesh_steps: 0,
            mesh_guidance: 0.0,
            mesh_seed: 0,
            mesh_seed_mode: "fixed".to_string(),
            box_v: 0.0,
            octree_resolution: 0,
            num_chunks: 0,
            mc_level: 0.0,
            mc_algo: String::new(),
            enable_flash_vdm: false,
            compile_models: false,
            force_offload: false,
            remove_floaters: false,
            remove_degenerate_faces: false,
            reduce_faces: false,
            max_faces: 0,
            smooth_normals: false,
            voxelize: false,
            voxel_resolution: default_voxel_resolution(),
            texture_view_size: 0,
            texture_steps: 0,
            texture_guidance: 0.0,
            texture_size: 0,
            texture_seed: 0,
            texture_seed_mode: "fixed".to_string(),
            unwrap_mesh: false,
            camera_azimuths: String::new(),
            camera_elevations: String::new(),
            view_weights: String::new(),
            ortho_scale: 0.0,
            trellis_preset: default_trellis_preset(),
            trellis_autocrop: true,
            trellis_force_opaque: true,
            trellis_atlas_padding: default_trellis_atlas_padding(),
            trellis_resolution: default_trellis_resolution(),
            trellis_seed: default_trellis_seed(),
            trellis_seed_mode: default_randomize_seed_mode(),
            trellis_sparse_steps: default_trellis_sparse_steps(),
            trellis_sparse_guidance: default_trellis_sparse_guidance(),
            trellis_sparse_rescale: default_trellis_sparse_rescale(),
            trellis_sparse_rescale_t: default_trellis_sparse_rescale_t(),
            trellis_shape_steps: default_trellis_shape_steps(),
            trellis_shape_guidance: default_trellis_shape_guidance(),
            trellis_shape_rescale: default_trellis_shape_rescale(),
            trellis_shape_rescale_t: default_trellis_shape_rescale_t(),
            trellis_texture_steps: default_trellis_texture_steps(),
            trellis_texture_guidance: default_trellis_texture_guidance(),
            trellis_texture_rescale: default_trellis_texture_rescale(),
            trellis_texture_rescale_t: default_trellis_texture_rescale_t(),
            trellis_decimation_target: default_trellis_decimation_target(),
            trellis_texture_size: default_trellis_texture_size(),
            trellis_remesh: false,
            trellis_export_webp: true,
        }
    }
}

#[derive(Debug, Serialize)]
struct WorkerJob {
    action: String,
    #[serde(rename = "job_id")]
    job_id: String,
    paths: Vec<String>,
    options: HunyuanOptions,
    #[serde(rename = "app_data_dir")]
    app_data_dir: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentJobRequest {
    paths: Option<Vec<String>>,
    options: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentPathRequest {
    path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentApiAutostartConfig {
    enabled: Option<bool>,
    port: Option<u16>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentServerStatus {
    enabled: bool,
    port: u16,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
    active_job: Option<JobSummary>,
    last_job: Option<JobSummary>,
    message: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AgentApiRegistryEntry {
    app_id: String,
    app_name: String,
    default_port: u16,
    bind_address: String,
    port: u16,
    enabled: bool,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
    last_seen: Option<String>,
    note: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentApiRegistry {
    updated_at: String,
    apps: Vec<AgentApiRegistryEntry>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobSummary {
    id: String,
    status: String,
    message: String,
    outputs: Vec<String>,
    failed_count: Option<u64>,
    total_count: Option<u64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PresetAverage {
    average_seconds: f64,
    samples: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobHistoryResponse {
    jobs: Vec<serde_json::Value>,
    preset_averages: BTreeMap<String, PresetAverage>,
}

#[derive(Default)]
struct JobStateInner {
    active: Option<ActiveJob>,
    last_job: Option<JobSummary>,
}

#[derive(Clone, Default)]
struct ActiveJobState {
    inner: Arc<Mutex<JobStateInner>>,
}

struct ActiveJob {
    id: String,
    pid: u32,
    cancel_requested: bool,
    summary: JobSummary,
}

impl ActiveJobState {
    fn is_busy(&self) -> Result<bool, String> {
        self.inner
            .lock()
            .map(|state| state.active.is_some())
            .map_err(|_| "Unable to lock active job state.".to_string())
    }

    fn start(&self, id: String, pid: u32) -> Result<(), String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        if state.active.is_some() {
            return Err("Another job is already running.".to_string());
        }
        state.active = Some(ActiveJob {
            summary: JobSummary {
                id: id.clone(),
                status: "running".to_string(),
                message: "Job started.".to_string(),
                outputs: Vec::new(),
                failed_count: None,
                total_count: None,
            },
            id,
            pid,
            cancel_requested: false,
        });
        Ok(())
    }

    fn request_cancel(&self) -> Result<u32, String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = state.active.as_mut() else {
            return Err("No active job to cancel.".to_string());
        };
        job.cancel_requested = true;
        job.summary.status = "canceling".to_string();
        job.summary.message = "Cancel requested.".to_string();
        Ok(job.pid)
    }

    fn finish(&self, id: &str) -> Result<bool, String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = state.active.as_ref() else {
            return Ok(false);
        };
        if job.id != id {
            return Ok(false);
        }
        let mut job = state
            .active
            .take()
            .ok_or_else(|| "Unable to finish active job.".to_string())?;
        let canceled = job.cancel_requested;
        if canceled {
            job.summary.status = "canceled".to_string();
            job.summary.message = "Job canceled.".to_string();
        } else if matches!(job.summary.status.as_str(), "running" | "canceling") {
            job.summary.status = "done".to_string();
        }
        state.last_job = Some(job.summary);
        Ok(canceled)
    }

    fn finish_current(&self) -> Result<Option<bool>, String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(mut job) = state.active.take() else {
            return Ok(None);
        };
        if job.cancel_requested {
            job.summary.status = "canceled".to_string();
            job.summary.message = "Job canceled.".to_string();
        } else if matches!(job.summary.status.as_str(), "running" | "canceling") {
            job.summary.status = "done".to_string();
        }
        let canceled = job.cancel_requested;
        state.last_job = Some(job.summary);
        Ok(Some(canceled))
    }

    fn finish_if_pid(&self, pid: u32) -> Result<Option<bool>, String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = state.active.as_ref() else {
            return Ok(None);
        };
        if job.pid != pid {
            return Ok(None);
        }
        let mut job = state
            .active
            .take()
            .ok_or_else(|| "Unable to finish active job.".to_string())?;
        if job.cancel_requested {
            job.summary.status = "canceled".to_string();
            job.summary.message = "Job canceled.".to_string();
        } else if matches!(job.summary.status.as_str(), "running" | "canceling") {
            job.summary.status = "done".to_string();
        }
        let canceled = job.cancel_requested;
        state.last_job = Some(job.summary);
        Ok(Some(canceled))
    }

    fn record_event(&self, value: &serde_json::Value) -> Result<(), String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        if let Some(job) = state.active.as_mut() {
            apply_worker_event_to_summary(&mut job.summary, value);
        }
        Ok(())
    }

    fn record_last_event(&self, value: &serde_json::Value) -> Result<(), String> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        if let Some(summary) = state.last_job.as_mut() {
            apply_worker_event_to_summary(summary, value);
        }
        Ok(())
    }

    fn active_job_summary(&self) -> Result<Option<JobSummary>, String> {
        self.inner
            .lock()
            .map(|state| state.active.as_ref().map(|job| job.summary.clone()))
            .map_err(|_| "Unable to lock active job state.".to_string())
    }

    fn last_job_summary(&self) -> Result<Option<JobSummary>, String> {
        self.inner
            .lock()
            .map(|state| state.last_job.clone())
            .map_err(|_| "Unable to lock active job state.".to_string())
    }
}

fn apply_worker_event_to_summary(summary: &mut JobSummary, value: &serde_json::Value) {
    let event_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if let Some(message) = value.get("message").and_then(serde_json::Value::as_str) {
        summary.message = message.to_string();
    }
    if let Some(outputs) = value.get("outputs").and_then(serde_json::Value::as_array) {
        for output in outputs.iter().filter_map(serde_json::Value::as_str) {
            let output = output.to_string();
            if !summary.outputs.contains(&output) {
                summary.outputs.push(output);
            }
        }
    }
    if let Some(failed_count) = value.get("failedCount").and_then(serde_json::Value::as_u64) {
        summary.failed_count = Some(failed_count);
    }
    if let Some(total_count) = value.get("totalCount").and_then(serde_json::Value::as_u64) {
        summary.total_count = Some(total_count);
    }
    match event_type {
        "done" => {
            summary.status = if summary.failed_count.unwrap_or(0) > 0 {
                "error".to_string()
            } else {
                "done".to_string()
            };
        }
        "error" => summary.status = "error".to_string(),
        "canceled" => summary.status = "canceled".to_string(),
        _ => {}
    }
}

#[derive(Clone, Default)]
struct ManagedWorkerState {
    inner: Arc<Mutex<Option<ManagedWorker>>>,
}

#[derive(Clone)]
struct ManagedWorker {
    pid: u32,
    model_backend: String,
    runtime_profile: String,
    cuda_device: String,
    stdin: Arc<Mutex<ChildStdin>>,
}

impl ManagedWorkerState {
    fn get_matching(
        &self,
        model_backend: &str,
        runtime_profile: &str,
        cuda_device: &str,
    ) -> Result<Option<ManagedWorker>, String> {
        self.inner
            .lock()
            .map(|worker| {
                worker.as_ref().and_then(|worker| {
                    if worker.model_backend == model_backend
                        && worker.runtime_profile == runtime_profile
                        && worker.cuda_device == cuda_device
                    {
                        Some(worker.clone())
                    } else {
                        None
                    }
                })
            })
            .map_err(|_| "Unable to lock managed worker state.".to_string())
    }

    fn set(&self, worker: ManagedWorker) -> Result<(), String> {
        self.inner
            .lock()
            .map(|mut current| *current = Some(worker))
            .map_err(|_| "Unable to lock managed worker state.".to_string())
    }

    fn clear_if_pid(&self, pid: u32) -> Result<(), String> {
        self.inner
            .lock()
            .map(|mut worker| {
                if worker.as_ref().is_some_and(|worker| worker.pid == pid) {
                    *worker = None;
                }
            })
            .map_err(|_| "Unable to lock managed worker state.".to_string())
    }

    fn stop(&self) -> Result<(), String> {
        let worker = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock managed worker state.".to_string())?
            .take();
        if let Some(worker) = worker {
            let _ = kill_process_tree(worker.pid);
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct AgentServerState {
    inner: Arc<Mutex<AgentServerControl>>,
}

#[derive(Default)]
struct AgentServerControl {
    enabled: bool,
    port: u16,
    stop: Option<Arc<AtomicBool>>,
}

impl AgentServerControl {
    fn port(&self) -> u16 {
        if self.port == 0 {
            read_registered_agent_api_port().unwrap_or(DEFAULT_AGENT_API_PORT)
        } else {
            self.port
        }
    }
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(dir) = portable_app_data_dir() {
        if fs::create_dir_all(&dir).is_ok() {
            return Ok(dir);
        }
    }

    let resolved = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Unable to resolve app data folder: {error}"))?;
    let dir = legacy_app_data_dir()
        .filter(|legacy| legacy.exists() && !resolved.exists())
        .unwrap_or(resolved);
    fs::create_dir_all(&dir)
        .map_err(|error| format!("Unable to create app data folder: {error}"))?;
    Ok(dir)
}

/// Folder the app was installed/launched from. Inside an AppImage the
/// executable lives in a read-only (or throwaway extracted) mount, so use the
/// folder holding the `.AppImage` file instead, which is the Linux equivalent
/// of the Windows portable exe folder.
fn app_install_dir() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    if let Some(appimage) = env::var_os("APPIMAGE").map(PathBuf::from) {
        if let Some(parent) = appimage.parent().filter(|parent| parent.is_dir()) {
            return Some(parent.to_path_buf());
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
}

fn portable_app_data_dir() -> Option<PathBuf> {
    let exe_dir = app_install_dir()?;
    let has_portable_layout = exe_dir.join("python").join("hunyuan_worker.py").is_file()
        || exe_dir
            .join("output")
            .join("agent-api-autostart.json")
            .is_file()
        || exe_dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("portable"));
    if has_portable_layout {
        Some(exe_dir.join("data"))
    } else {
        None
    }
}

fn legacy_app_data_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .map(|base| base.join("public.image.to.hunyuan3d"))
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
            .map(|base| base.join("public.image.to.hunyuan3d"))
    }
}

/// App identifier used for the shared NekoLegendsAI store.
const NEKO_APP_SLUG: &str = "image-to-3d";

/// Resolve the shared NekoLegendsAI store, hinting the default location to the
/// executable's drive so a portable install on `D:` lands the store at
/// `D:\NekoLegendsAI` (next to the app, on the roomy data drive).
fn neko_store(app: &AppHandle) -> Result<NekoStore, String> {
    let app_data = app_data_dir(app)?;
    let exe_dir = app_install_dir();
    Ok(NekoStore::resolve_with_volume_hint(
        NEKO_APP_SLUG,
        &app_data,
        exe_dir.as_deref(),
    ))
}

fn env_python_path(env_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        env_dir.join("Scripts").join("python.exe")
    } else {
        env_dir.join("bin").join("python")
    }
}

fn normalize_runtime_profile(profile: &str) -> String {
    match profile.trim() {
        "cu128" => "cu128".to_string(),
        "cu124" => "cu124".to_string(),
        _ => DEFAULT_PYTORCH_PROFILE.to_string(),
    }
}

/// Resolve the venv dir honoring user/store options. Callers that already have
/// `&HunyuanOptions` should call this; legacy callers without options fall back
/// to the options-less `env_dir_for_profile` below.
fn env_dir_for_profile_opts(
    app: &AppHandle,
    options: &HunyuanOptions,
    profile: &str,
) -> Result<PathBuf, String> {
    resolve_env_dir(app, options, &normalize_runtime_profile(profile))
}

fn env_dir_for_profile(app: &AppHandle, profile: &str) -> Result<PathBuf, String> {
    env_dir_for_profile_opts(app, &HunyuanOptions::default(), profile)
}

/// Resolve the runtime (source tree) dir honoring user/store options.
fn runtime_dir_opts(app: &AppHandle, options: &HunyuanOptions) -> Result<PathBuf, String> {
    resolve_runtime_dir(app, options)
}

fn hunyuan_source_dir_opts(app: &AppHandle, options: &HunyuanOptions) -> Result<PathBuf, String> {
    Ok(runtime_dir_opts(app, options)?.join("Hunyuan3D-2.1"))
}

fn trellis_source_dir_opts(app: &AppHandle, options: &HunyuanOptions) -> Result<PathBuf, String> {
    Ok(runtime_dir_opts(app, options)?.join("TRELLIS.2"))
}

/// Resolve the directory used to store downloaded model files.
///
/// Priority:
///   1. A user-chosen path (from the setup UI) wins outright, so a user can
///      relocate the multi-GB model cache to a data drive and keep it stable
///      across app version upgrades.
///   2. A pre-existing app-local `data/models` folder that actually contains a
///      real download is auto-migrated into the shared NekoLegendsAI store
///      (once), so legacy users don't re-download.
///   3. Otherwise the shared NekoLegendsAI store is used — it lives outside the
///      version-specific app folder and persists across upgrades.
///
/// The previous logic preferred any existing `data/models`, but the publish
/// script created that folder empty on every release, so the shared store was
/// bypassed and models were re-downloaded for each new version.
fn resolve_model_dir(app: &AppHandle, options: &HunyuanOptions) -> Result<PathBuf, String> {
    let requested = options.model_dir.trim();
    if !requested.is_empty() {
        let path = PathBuf::from(requested);
        fs::create_dir_all(&path)
            .map_err(|error| format!("Unable to create model folder {requested}: {error}"))?;
        return Ok(path);
    }

    let app_data = app_data_dir(app)?;
    let legacy = app_data.join("models");
    let store_models = neko_store(app)?.models_dir;

    // Auto-migrate a real legacy download into the shared store (once).
    // Only migrate when the legacy folder holds an actual download — not the
    // empty placeholder the publish script leaves behind — and the store target
    // does not already contain models.
    if legacy_models_present(&legacy) && !target_has_models(&store_models) {
        if let Err(error) = migrate_legacy_models(&legacy, &store_models) {
            eprintln!("Model migration to shared store failed: {error}");
            // Fall back to the legacy folder so this run still works.
            return Ok(legacy);
        }
    }

    Ok(store_models)
}

/// True when a folder contains a real model download (a readiness marker or a
/// populated Hugging Face hub cache). Empty placeholder folders return false.
fn legacy_models_present(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    if dir.join(HUNYUAN_MODEL_MARKER).is_file() || dir.join(TRELLIS_MODEL_MARKER).is_file() {
        return true;
    }
    let hub = dir.join("huggingface").join("hub");
    match fs::read_dir(&hub) {
        Ok(entries) => entries.filter_map(Result::ok).any(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| name.starts_with("models--"))
                .unwrap_or(false)
        }),
        Err(_) => false,
    }
}

/// True when a target folder already holds downloaded models.
fn target_has_models(dir: &Path) -> bool {
    legacy_models_present(dir)
}

/// Move the contents of a legacy model folder into the shared store, leaving the
/// legacy folder empty. Any file that already exists at the target is left in
/// place (the legacy copy is removed) so a re-run after a partial migration is
/// safe.
fn migrate_legacy_models(legacy: &Path, target: &Path) -> Result<(), String> {
    fs::create_dir_all(target)
        .map_err(|error| format!("Unable to create shared model folder: {error}"))?;

    let entries = fs::read_dir(legacy)
        .map_err(|error| format!("Unable to read legacy model folder: {error}"))?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let from = entry.path();
        let to = target.join(&name);
        if from.is_dir() {
            // Merge directory trees (e.g. huggingface/hub) recursively.
            merge_dir(&from, &to)
                .map_err(|error| format!("Failed merging {}: {error}", from.display()))?;
            let _ = fs::remove_dir_all(&from);
        } else if to.exists() {
            let _ = fs::remove_file(&from);
        } else if let Err(error) = fs::rename(&from, &to) {
            // Cross-volume rename can fail; fall back to copy + remove.
            fs::copy(&from, &to)
                .map_err(|error| format!("Failed copying {}: {error}", from.display()))?;
            let _ = fs::remove_file(&from);
            eprintln!("Renamed {error}; copied {} instead.", from.display());
        }
    }
    Ok(())
}

/// Recursively copy `from` into `to` without overwriting existing files.
fn merge_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let path = entry.path();
        let dest = to.join(entry.file_name());
        if path.is_dir() {
            merge_dir(&path, &dest)?;
        } else if !dest.exists() {
            fs::copy(&path, &dest)?;
        }
    }
    Ok(())
}

/// Move a directory tree to a new location, merging contents and tolerating a
/// cross-volume source (where `rename` fails). Leaves `from` removed on success.
fn move_dir(from: &Path, to: &Path) -> Result<(), String> {
    if !from.is_dir() {
        return Ok(());
    }
    merge_dir(from, to).map_err(|error| format!("Failed merging {}: {error}", from.display()))?;
    let _ = fs::remove_dir_all(from);
    Ok(())
}

/// Resolve the directory used to store the Python virtual environment.
///
/// Mirrors `resolve_model_dir`: a user override wins; otherwise the shared
/// NekoLegendsAI store is used so the venv persists across app version upgrades.
/// A pre-existing version-local venv that holds a real install is auto-migrated
/// into the store once.
fn resolve_env_dir(app: &AppHandle, options: &HunyuanOptions, profile: &str) -> Result<PathBuf, String> {
    let requested = options.env_dir.trim();
    if !requested.is_empty() {
        let path = PathBuf::from(requested).join(format!("python-env-{profile}"));
        fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))
            .map_err(|error| format!("Unable to create env folder: {error}"))?;
        return Ok(path);
    }

    let app_data = app_data_dir(app)?;
    let legacy = app_data.join(format!("python-env-{profile}"));
    let store_env = neko_store(app)?.env_dir.join(format!("python-env-{profile}"));

    // Auto-migrate a real legacy venv into the shared store (once).
    if legacy.is_dir() && legacy.join(ENV_MARKER).is_file() && !store_env.join(ENV_MARKER).is_file() {
        if let Err(error) = move_dir(&legacy, &store_env) {
            eprintln!("Venv migration to shared store failed: {error}");
            return Ok(legacy);
        }
    }

    Ok(store_env)
}

/// Resolve the directory used to store the Hunyuan/TRELLIS source trees (and
/// their build artifacts). Mirrors `resolve_model_dir`: a user override wins,
/// otherwise the shared store is used, with one-time auto-migration of a real
/// legacy source tree.
fn resolve_runtime_dir(app: &AppHandle, options: &HunyuanOptions) -> Result<PathBuf, String> {
    let requested = options.runtime_dir.trim();
    if !requested.is_empty() {
        let path = PathBuf::from(requested);
        fs::create_dir_all(&path)
            .map_err(|error| format!("Unable to create runtime folder: {error}"))?;
        return Ok(path);
    }

    let app_data = app_data_dir(app)?;
    let legacy = app_data.join("runtime");
    let store_runtime = neko_store(app)?.root.join("runtimes").join(NEKO_APP_SLUG);

    // Auto-migrate a real legacy source tree into the shared store (once).
    let legacy_has_source = legacy.join("Hunyuan3D-2.1").join(HUNYUAN_SOURCE_MARKER).is_file()
        || legacy.join("TRELLIS.2").join(TRELLIS_SOURCE_MARKER).is_file();
    let store_has_source = store_runtime.join("Hunyuan3D-2.1").join(HUNYUAN_SOURCE_MARKER).is_file()
        || store_runtime.join("TRELLIS.2").join(TRELLIS_SOURCE_MARKER).is_file();
    if legacy_has_source && !store_has_source {
        if let Err(error) = move_dir(&legacy, &store_runtime) {
            eprintln!("Runtime source migration to shared store failed: {error}");
            return Ok(legacy);
        }
    }

    Ok(store_runtime)
}

fn default_output_dir() -> Result<PathBuf, String> {
    let app_dir = app_install_dir().ok_or_else(|| "Unable to resolve app folder.".to_string())?;
    Ok(app_dir.join("output"))
}

fn output_dir(_app: &AppHandle) -> Result<PathBuf, String> {
    let dir = default_output_dir()?;
    fs::create_dir_all(&dir).map_err(|error| format!("Unable to create output folder: {error}"))?;
    Ok(dir)
}

fn env_runtime_profile(env: &Path) -> String {
    let marker = env.join(ENV_MARKER);
    let Ok(content) = fs::read_to_string(marker) else {
        return "not installed".to_string();
    };
    let trimmed = content.trim();
    if trimmed == "ok" {
        return "cu124".to_string();
    }
    for line in trimmed.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == "pytorch" {
            return value.trim().to_string();
        }
    }
    "unknown".to_string()
}

fn env_marker_value(env: &Path, wanted_key: &str) -> Option<String> {
    let content = fs::read_to_string(env.join(ENV_MARKER)).ok()?;
    for line in content.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == wanted_key {
            return Some(value.trim().to_string());
        }
    }
    None
}

fn env_marker_flavor(env: &Path) -> String {
    env_marker_value(env, "runtime").unwrap_or_else(|| {
        if env.join(ENV_MARKER).is_file() {
            "legacy-full".to_string()
        } else {
            "not installed".to_string()
        }
    })
}

fn normalize_model_backend(value: &str) -> String {
    match value.trim() {
        "trellis2" | "trellis" => "trellis2".to_string(),
        "hunyuan" | "" => "hunyuan".to_string(),
        _ => "hunyuan".to_string(),
    }
}

fn hunyuan_env_ready(env: &Path) -> bool {
    env.join(HUNYUAN_ENV_MARKER).is_file()
        || (env.join(ENV_MARKER).is_file() && env_marker_flavor(env) == "legacy-full")
}

fn trellis_env_ready(env: &Path) -> bool {
    env.join(TRELLIS_ENV_MARKER).is_file()
}

fn default_voxel_resolution() -> u32 {
    64
}

fn default_model_backend() -> String {
    "hunyuan".to_string()
}

fn default_randomize_seed_mode() -> String {
    "randomize".to_string()
}

fn default_true() -> bool {
    true
}

fn default_trellis_preset() -> String {
    "standard".to_string()
}

fn default_trellis_atlas_padding() -> u32 {
    12
}

fn default_trellis_resolution() -> u32 {
    1024
}

fn default_trellis_seed() -> u64 {
    12345
}

fn default_trellis_sparse_steps() -> u32 {
    12
}

fn default_trellis_sparse_guidance() -> f32 {
    7.5
}

fn default_trellis_sparse_rescale() -> f32 {
    0.7
}

fn default_trellis_sparse_rescale_t() -> f32 {
    5.0
}

fn default_trellis_shape_steps() -> u32 {
    12
}

fn default_trellis_shape_guidance() -> f32 {
    7.5
}

fn default_trellis_shape_rescale() -> f32 {
    0.5
}

fn default_trellis_shape_rescale_t() -> f32 {
    3.0
}

fn default_trellis_texture_steps() -> u32 {
    12
}

fn default_trellis_texture_guidance() -> f32 {
    3.0
}

fn default_trellis_texture_rescale() -> f32 {
    0.5
}

fn default_trellis_texture_rescale_t() -> f32 {
    3.0
}

fn default_trellis_decimation_target() -> u32 {
    100_000
}

fn default_trellis_texture_size() -> u32 {
    2048
}

fn default_hunyuan_options() -> HunyuanOptions {
    HunyuanOptions {
        model_backend: "hunyuan".to_string(),
        mode: "meshTexture".to_string(),
        cuda_device: "auto".to_string(),
        runtime_profile: "cu124".to_string(),
        output_dir: String::new(),
        output_name: String::new(),
        mesh_input_path: String::new(),
        model_dir: String::new(),
        env_dir: String::new(),
        runtime_dir: String::new(),
        recursive: true,
        overwrite: false,
        mesh_detail: "low".to_string(),
        mesh_steps: 24,
        mesh_guidance: 5.0,
        mesh_seed: 2,
        mesh_seed_mode: "randomize".to_string(),
        box_v: 1.01,
        octree_resolution: 256,
        num_chunks: 6000,
        mc_level: 0.0,
        mc_algo: "mc".to_string(),
        enable_flash_vdm: true,
        compile_models: false,
        force_offload: false,
        remove_floaters: true,
        remove_degenerate_faces: true,
        reduce_faces: true,
        max_faces: 5000,
        smooth_normals: true,
        voxelize: false,
        voxel_resolution: 64,
        texture_view_size: 512,
        texture_steps: 15,
        texture_guidance: 3.0,
        texture_size: 2048,
        texture_seed: 802344256,
        texture_seed_mode: "randomize".to_string(),
        unwrap_mesh: true,
        camera_azimuths: "0, 90, 180, 270, 0, 180".to_string(),
        camera_elevations: "0, 0, 0, 0, 90, -90".to_string(),
        view_weights: "1, 0.1, 0.5, 0.1, 0.05, 0.05".to_string(),
        ortho_scale: 1.1,
        trellis_preset: default_trellis_preset(),
        trellis_autocrop: true,
        trellis_force_opaque: true,
        trellis_atlas_padding: default_trellis_atlas_padding(),
        trellis_resolution: default_trellis_resolution(),
        trellis_seed: default_trellis_seed(),
        trellis_seed_mode: default_randomize_seed_mode(),
        trellis_sparse_steps: default_trellis_sparse_steps(),
        trellis_sparse_guidance: default_trellis_sparse_guidance(),
        trellis_sparse_rescale: default_trellis_sparse_rescale(),
        trellis_sparse_rescale_t: default_trellis_sparse_rescale_t(),
        trellis_shape_steps: default_trellis_shape_steps(),
        trellis_shape_guidance: default_trellis_shape_guidance(),
        trellis_shape_rescale: default_trellis_shape_rescale(),
        trellis_shape_rescale_t: default_trellis_shape_rescale_t(),
        trellis_texture_steps: default_trellis_texture_steps(),
        trellis_texture_guidance: default_trellis_texture_guidance(),
        trellis_texture_rescale: default_trellis_texture_rescale(),
        trellis_texture_rescale_t: default_trellis_texture_rescale_t(),
        trellis_decimation_target: default_trellis_decimation_target(),
        trellis_texture_size: default_trellis_texture_size(),
        trellis_remesh: false,
        trellis_export_webp: true,
    }
}

fn apply_mesh_detail_preset(options: &mut HunyuanOptions) {
    match options.mesh_detail.as_str() {
        "low" => {
            options.mesh_steps = 24;
            options.octree_resolution = 256;
            options.num_chunks = 6000;
            options.max_faces = 5_000;
        }
        "medium" => {
            options.mesh_steps = 50;
            options.octree_resolution = 384;
            options.num_chunks = 8000;
            options.max_faces = 20_000;
        }
        "high" => {
            options.mesh_steps = 80;
            options.octree_resolution = 512;
            options.num_chunks = 12000;
            options.max_faces = 50_000;
        }
        "custom" => {}
        _ => {
            options.mesh_detail = "custom".to_string();
        }
    }
}

fn apply_trellis_preset(options: &mut HunyuanOptions) {
    match options.trellis_preset.as_str() {
        "preview" => {
            options.trellis_resolution = 512;
            options.trellis_sparse_steps = 12;
            options.trellis_shape_steps = 12;
            options.trellis_texture_steps = 12;
            options.trellis_texture_guidance = 3.0;
            options.trellis_decimation_target = 20_000;
            options.trellis_texture_size = 1024;
            options.trellis_remesh = false;
        }
        "standard" => {
            options.trellis_resolution = 1024;
            options.trellis_sparse_steps = 12;
            options.trellis_shape_steps = 12;
            options.trellis_texture_steps = 12;
            options.trellis_texture_guidance = 7.5;
            options.trellis_decimation_target = 100_000;
            options.trellis_texture_size = 2048;
            options.trellis_remesh = false;
        }
        "quality" => {
            options.trellis_resolution = 1024;
            options.trellis_sparse_steps = 25;
            options.trellis_shape_steps = 25;
            options.trellis_texture_steps = 25;
            options.trellis_texture_guidance = 7.5;
            options.trellis_decimation_target = 200_000;
            options.trellis_texture_size = 2048;
            options.trellis_remesh = false;
        }
        "highpoly" => {
            options.trellis_resolution = 1024;
            options.trellis_sparse_steps = 25;
            options.trellis_shape_steps = 25;
            options.trellis_texture_steps = 25;
            options.trellis_texture_guidance = 7.5;
            options.trellis_decimation_target = 500_000;
            options.trellis_texture_size = 4096;
            options.trellis_remesh = false;
        }
        "custom" => {}
        _ => options.trellis_preset = "custom".to_string(),
    }
}

fn normalize_options_for_job(options: &mut HunyuanOptions) -> Result<(), String> {
    options.model_backend = normalize_model_backend(&options.model_backend);
    apply_mesh_detail_preset(options);
    if options.model_backend == "trellis2" {
        apply_trellis_preset(options);
        options.mode = "meshTexture".to_string();
    }
    if options.output_dir.trim().is_empty() {
        options.output_dir = default_output_dir()?.display().to_string();
    } else {
        let output_dir = PathBuf::from(options.output_dir.trim());
        if output_dir.is_relative() {
            options.output_dir = default_output_dir()?
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(output_dir)
                .display()
                .to_string();
        }
    }
    Ok(())
}

fn jobs_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app_data_dir(app)?.join("jobs");
    fs::create_dir_all(&dir).map_err(|error| format!("Unable to create jobs folder: {error}"))?;
    Ok(dir)
}

#[tauri::command]
fn list_job_history(app: AppHandle) -> Result<JobHistoryResponse, String> {
    let history_dir = jobs_dir(&app)?.join("history");
    if !history_dir.is_dir() {
        return Ok(JobHistoryResponse {
            jobs: Vec::new(),
            preset_averages: BTreeMap::new(),
        });
    }

    let mut jobs = Vec::new();
    let mut samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let entries = fs::read_dir(&history_dir)
        .map_err(|error| format!("Unable to read job history: {error}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        let backend = value
            .get("backend")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let preset = value
            .get("preset")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("custom");
        let status = value
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        if status == "done" {
            if let Some(elapsed) = value
                .get("elapsedSeconds")
                .and_then(serde_json::Value::as_f64)
            {
                let completed_items = value
                    .get("totalCount")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1)
                    .max(1) as f64;
                samples
                    .entry(format!("{backend}:{preset}"))
                    .or_default()
                    .push(elapsed / completed_items);
            }
        }
        jobs.push(serde_json::json!({
            "id": value.get("id").cloned().unwrap_or(serde_json::Value::Null),
            "backend": backend,
            "preset": preset,
            "status": status,
            "message": value.get("message").cloned().unwrap_or(serde_json::Value::Null),
            "startedAt": value.get("startedAt").cloned().unwrap_or(serde_json::Value::Null),
            "endedAt": value.get("endedAt").cloned().unwrap_or(serde_json::Value::Null),
            "elapsedSeconds": value.get("elapsedSeconds").cloned().unwrap_or(serde_json::Value::Null),
            "inputs": value.get("inputs").cloned().unwrap_or_else(|| serde_json::json!([])),
            "outputs": value.get("outputs").cloned().unwrap_or_else(|| serde_json::json!([])),
            "failedCount": value.get("failedCount").cloned().unwrap_or(serde_json::Value::Null),
            "totalCount": value.get("totalCount").cloned().unwrap_or(serde_json::Value::Null),
        }));
    }
    jobs.sort_by(|left, right| {
        let left_started = left
            .get("startedAt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let right_started = right
            .get("startedAt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        right_started.cmp(left_started)
    });
    jobs.truncate(100);

    let preset_averages = samples
        .into_iter()
        .filter_map(|(key, values)| {
            if values.is_empty() {
                return None;
            }
            let samples = values.len();
            let average_seconds = values.iter().sum::<f64>() / samples as f64;
            Some((
                key,
                PresetAverage {
                    average_seconds,
                    samples,
                },
            ))
        })
        .collect();
    Ok(JobHistoryResponse {
        jobs,
        preset_averages,
    })
}

fn mark_job_manifest_status(
    app: &AppHandle,
    job_id: &str,
    status: &str,
    message: &str,
) -> Result<(), String> {
    let manifest_path = jobs_dir(app)?
        .join("history")
        .join(format!("{job_id}.json"));
    if !manifest_path.is_file() {
        return Ok(());
    }
    let raw = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("Unable to read job manifest: {error}"))?;
    let mut manifest = serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|error| format!("Unable to parse job manifest: {error}"))?;
    if let Some(object) = manifest.as_object_mut() {
        object.insert(
            "status".to_string(),
            serde_json::Value::String(status.to_string()),
        );
        object.insert(
            "message".to_string(),
            serde_json::Value::String(message.to_string()),
        );
    }
    let encoded = serde_json::to_string_pretty(&manifest)
        .map_err(|error| format!("Unable to serialize job manifest: {error}"))?;
    fs::write(&manifest_path, encoded)
        .map_err(|error| format!("Unable to update job manifest: {error}"))
}

fn worker_path(app: &AppHandle) -> Result<PathBuf, String> {
    let exe_dir = app_install_dir();
    let resource_dir = app.path().resource_dir().ok();
    let candidates = [
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("python")
            .join("hunyuan_worker.py"),
        exe_dir
            .unwrap_or_else(|| PathBuf::from("."))
            .join("python")
            .join("hunyuan_worker.py"),
        resource_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("_up_")
            .join("python")
            .join("hunyuan_worker.py"),
        resource_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("python")
            .join("hunyuan_worker.py"),
        resource_dir
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hunyuan_worker.py"),
    ];

    if let Some(path) = candidates.into_iter().find(|path| path.exists()) {
        return Ok(path);
    }

    write_embedded_worker(app)
}

fn write_embedded_worker(app: &AppHandle) -> Result<PathBuf, String> {
    let worker_dir = app_data_dir(app)?.join("python");
    fs::create_dir_all(&worker_dir)
        .map_err(|error| format!("Unable to create embedded worker folder: {error}"))?;
    let worker = worker_dir.join("hunyuan_worker.py");
    let needs_write = fs::read_to_string(&worker)
        .map(|content| content != EMBEDDED_WORKER)
        .unwrap_or(true);
    if needs_write {
        if let Err(error) = fs::write(&worker, EMBEDDED_WORKER) {
            let fallback = worker_dir.join("hunyuan_worker_embedded.py");
            fs::write(&fallback, EMBEDDED_WORKER).map_err(|fallback_error| {
                format!(
                    "Unable to write embedded worker: {error}; fallback copy failed: {fallback_error}"
                )
            })?;
            return Ok(fallback);
        }
    }
    Ok(worker)
}

fn parse_python_version(output: &str) -> Option<(u32, u32, u32)> {
    let version = output
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

fn supported_python_version(version: (u32, u32, u32)) -> bool {
    version.0 == 3 && (10..=12).contains(&version.1)
}

fn hide_command_window(command: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
    // Every helper process goes through here; keep AppImage paths out of it.
    child_env::strip_appimage_env(command);
}

fn command_from_python_spec(spec: &str) -> Command {
    let mut parts = spec.split_whitespace();
    let executable = parts.next().unwrap_or("python");
    let mut command = Command::new(executable);
    for arg in parts {
        command.arg(arg);
    }
    hide_command_window(&mut command);
    command
}

fn python_version(spec: &str) -> Option<(u32, u32, u32)> {
    let mut command = command_from_python_spec(spec);
    command.arg("--version");
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_python_version(&combined)
}

fn find_python() -> Option<String> {
    let candidates = if cfg!(target_os = "windows") {
        vec![
            "py -3.10", "py -3.11", "py -3.12", "py -3", "python", "python3",
        ]
    } else {
        // pymeshlab==2022.2.post3 (pinned by the worker) has Linux wheels only
        // up to Python 3.11, so a uv-managed 3.11 beats a system 3.12.
        vec!["python3.11", "python3.10", "python3", "python"]
    };
    candidates
        .into_iter()
        .map(str::to_string)
        .find(|candidate| {
            python_version(candidate).is_some_and(|version| {
                supported_python_version(version) && (cfg!(target_os = "windows") || version.1 <= 11)
            })
        })
        .or_else(find_uv_python)
}

/// Version range the worker's Linux dependency pins support, in `uv` request
/// syntax.
#[cfg(not(target_os = "windows"))]
const UV_PYTHON_REQUEST: &str = ">=3.10,<3.12";
/// Version `uv` installs when no supported interpreter exists.
#[cfg(not(target_os = "windows"))]
const UV_PYTHON_INSTALL: &str = "3.11";

/// Rolling distros (Arch/Omarchy) ship only a newer system Python than the
/// worker supports. When `uv` is installed, use (or install) a uv-managed
/// interpreter instead.
#[cfg(not(target_os = "windows"))]
fn find_uv_python() -> Option<String> {
    let mut command = Command::new("uv");
    command.args(["python", "find", "--no-project", UV_PYTHON_REQUEST]);
    hide_command_window(&mut command);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    // `command_from_python_spec` splits on whitespace.
    if path.is_empty() || path.contains(char::is_whitespace) {
        return None;
    }
    python_version(&path)
        .is_some_and(supported_python_version)
        .then_some(path)
}

#[cfg(target_os = "windows")]
fn find_uv_python() -> Option<String> {
    None
}

#[cfg(not(target_os = "windows"))]
fn uv_available() -> bool {
    let mut command = Command::new("uv");
    command.arg("--version");
    hide_command_window(&mut command);
    matches!(command.output(), Ok(output) if output.status.success())
}

#[cfg(target_os = "windows")]
fn uv_available() -> bool {
    false
}

/// Python used to run setup. On Linux/macOS, installs a uv-managed Python
/// when no supported interpreter is present.
fn find_or_install_python() -> Result<String, String> {
    if let Some(python) = find_python() {
        return Ok(python);
    }
    #[cfg(not(target_os = "windows"))]
    if uv_available() {
        let mut command = Command::new("uv");
        command.args(["python", "install", UV_PYTHON_INSTALL]);
        hide_command_window(&mut command);
        let output = command
            .output()
            .map_err(|error| format!("Unable to run uv to install Python {UV_PYTHON_INSTALL}: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "uv could not install Python {UV_PYTHON_INSTALL}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        return find_uv_python().ok_or_else(|| {
            format!("uv installed Python {UV_PYTHON_INSTALL}, but it could not be located.")
        });
    }
    Err(if cfg!(target_os = "windows") {
        "Python 3.10, 3.11, or 3.12 was not found. Install Python and try again.".to_string()
    } else {
        "Python 3.10 or 3.11 was not found. Install one (or install uv so Image To 3D can \
         fetch Python 3.11 itself) and try again."
            .to_string()
    })
}

fn configure_worker_command(command: &mut Command) {
    hide_command_window(command);
    command
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8:replace")
        .env("PYTHONLEGACYWINDOWSSTDIO", "0")
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .env("TQDM_DISABLE", "1")
        .env_remove("TQDM_ASCII");
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn configure_managed_worker_env(
    command: &mut Command,
    app_data: &Path,
    env_dir: &Path,
    hunyuan_source: &Path,
    trellis_source: &Path,
    models: &Path,
    runtime_dir: &Path,
) {
    configure_worker_storage_env(command, models);
    command
        .env("I3D_IN_MANAGED_ENV", "1")
        .env("I3D_HUNYUAN_SOURCE_DIR", hunyuan_source)
        .env("I3D_TRELLIS_SOURCE_DIR", trellis_source)
        .env("I3D_APP_DATA_DIR", app_data)
        .env("I3D_ENV_DIR", env_dir)
        .env("I3D_RUNTIME_DIR", runtime_dir)
        .env("IH3D_IN_MANAGED_ENV", "1")
        .env("IH3D_SOURCE_DIR", hunyuan_source)
        .env("IH3D_APP_DATA_DIR", app_data)
        .env("XFORMERS_FORCE_DISABLE_TRITON", "1");
}

fn configure_worker_storage_env(command: &mut Command, models: &Path) {
    command
        .env("I3D_MODEL_DIR", models)
        .env("IH3D_MODEL_DIR", models)
        .env("HF_HOME", models.join("huggingface"))
        .env("HF_HUB_CACHE", models.join("huggingface").join("hub"))
        .env(
            "TRANSFORMERS_CACHE",
            models.join("huggingface").join("transformers"),
        );

    // Route pip/uv/torch caches to the shared NekoLegendsAI store so PyTorch and
    // wheel downloads are reused across every app in the suite. `models` already
    // points at the store's models dir, so its parent is the store root. When the
    // legacy app-local models path is in use, caches simply live beside it — still
    // correct, just app-local.
    if let Some(store_root) = models.parent() {
        let cache = store_root.join("cache");
        command
            .env("PIP_CACHE_DIR", cache.join("pip"))
            .env("UV_CACHE_DIR", cache.join("uv"))
            .env("TORCH_HOME", cache.join("torch"));
    }
}

fn kill_process_tree(pid: u32) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T", "/F"]);
        command
    };

    #[cfg(not(target_os = "windows"))]
    let mut command = {
        let mut command = Command::new("kill");
        command.args(["-TERM", &format!("-{pid}")]);
        command
    };

    hide_command_window(&mut command);
    match command.output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => Err(format!(
            "Unable to cancel job: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(error) => Err(format!("Unable to cancel job: {error}")),
    }
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(ext))
        })
        .unwrap_or(false)
}

fn collect_images(path: &Path, recursive: bool, output: &mut Vec<String>) {
    if path.is_file() {
        if is_image_file(path) {
            output.push(path.display().to_string());
        }
        return;
    }
    if !path.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_file() && is_image_file(&entry_path) {
            output.push(entry_path.display().to_string());
        } else if recursive && entry_path.is_dir() {
            collect_images(&entry_path, recursive, output);
        }
    }
}

fn image_mime_type(path: &Path) -> Result<&'static str, String> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => Ok("image/png"),
        Some("jpg") | Some("jpeg") | Some("jpe") | Some("jfif") => Ok("image/jpeg"),
        Some("webp") => Ok("image/webp"),
        Some("bmp") => Ok("image/bmp"),
        Some("tif") | Some("tiff") => Ok("image/tiff"),
        Some(extension) => Err(format!("Unsupported preview image extension: {extension}")),
        None => Err("Preview image has no file extension.".to_string()),
    }
}

fn nvidia_smi_available() -> bool {
    let mut command = Command::new("nvidia-smi");
    command.args(["--query-gpu=name", "--format=csv,noheader"]);
    hide_command_window(&mut command);
    matches!(command.output(), Ok(output) if output.status.success())
}

fn model_cache_has_snapshot(model_root: &Path, repo: &str) -> bool {
    let repo_dir = model_root
        .join("huggingface")
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let Ok(entries) = fs::read_dir(snapshots) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry.path().is_dir()
            && fs::read_dir(entry.path())
                .map(|mut children| children.any(|child| child.is_ok()))
                .unwrap_or(false)
    })
}

fn runtime_ready_for_generation(app: &AppHandle, options: &HunyuanOptions) -> Result<(), String> {
    let requested_profile = normalize_runtime_profile(&options.runtime_profile);
    let model_backend = normalize_model_backend(&options.model_backend);
    let env = env_dir_for_profile_opts(app, options, &requested_profile)?;
    let hunyuan_source = hunyuan_source_dir_opts(app, options)?;
    let trellis_source = trellis_source_dir_opts(app, options)?;
    let models = resolve_model_dir(app, options)?;
    let env_python = env_python_path(&env);
    if !env_python.is_file() || !env.join(ENV_MARKER).is_file() {
        return Err("Setup is needed before generation. Choose Set up now first.".to_string());
    }
    let installed_profile = env_runtime_profile(&env);
    if installed_profile != "unknown" && installed_profile != requested_profile {
        return Err(format!(
            "The selected app runtime is installed for {installed_profile}, but this job is set to {requested_profile}. Choose Prepare to install the selected PyTorch profile."
        ));
    }
    if model_backend == "trellis2" {
        if !trellis_env_ready(&env) {
            return Err(
                "TRELLIS.2 runtime setup is not finished yet. Choose Set up now first.".to_string(),
            );
        }
        if !trellis_source.join(TRELLIS_SOURCE_MARKER).is_file() {
            return Err(
                "TRELLIS.2 source setup is not finished yet. Choose Set up now first.".to_string(),
            );
        }
        if !models.join(TRELLIS_MODEL_MARKER).is_file() {
            return Err(
                "TRELLIS.2 model downloads are not finished yet. Choose Set up now first."
                    .to_string(),
            );
        }
        return Ok(());
    }
    if !hunyuan_env_ready(&env) {
        return Err(
            "Hunyuan3D runtime setup is not finished yet. Choose Set up now first.".to_string(),
        );
    }
    if !hunyuan_source.join(HUNYUAN_SOURCE_MARKER).is_file() {
        return Err(
            "Hunyuan3D source setup is not finished yet. Choose Set up now first.".to_string(),
        );
    }
    if !models.join(HUNYUAN_MODEL_MARKER).is_file()
        && !model_cache_has_snapshot(&models, "tencent/Hunyuan3D-2.1")
    {
        return Err(
            "Hunyuan3D model downloads are not finished yet. Choose Set up now first.".to_string(),
        );
    }
    if options.mode != "meshOnly" {
        if !env.join(WHEEL_MARKER).is_file() {
            return Err("Texture tools are not set up yet. Choose Set up now first.".to_string());
        }
        if !hunyuan_source
            .join("hy3dpaint")
            .join("ckpt")
            .join("RealESRGAN_x4plus.pth")
            .is_file()
        {
            return Err("Texture setup is not finished yet. Choose Set up now first.".to_string());
        }
    }
    Ok(())
}

#[tauri::command]
fn check_runtime(
    app: AppHandle,
    runtime_profile: Option<String>,
    model_backend: Option<String>,
    model_dir: Option<String>,
    env_dir: Option<String>,
    runtime_dir: Option<String>,
) -> Result<RuntimeStatus, String> {
    let requested_profile = normalize_runtime_profile(
        runtime_profile
            .as_deref()
            .unwrap_or(DEFAULT_PYTORCH_PROFILE),
    );
    let selected_backend = normalize_model_backend(model_backend.as_deref().unwrap_or("hunyuan"));
    let app_data = app_data_dir(&app)?;
    let worker = worker_path(&app)?;
    let resolved_options = HunyuanOptions {
        model_dir: model_dir.unwrap_or_default(),
        env_dir: env_dir.unwrap_or_default(),
        runtime_dir: runtime_dir.unwrap_or_default(),
        ..Default::default()
    };
    let env = env_dir_for_profile_opts(&app, &resolved_options, &requested_profile)?;
    let env_python = env_python_path(&env);
    let hunyuan_source = hunyuan_source_dir_opts(&app, &resolved_options)?;
    let trellis_source = trellis_source_dir_opts(&app, &resolved_options)?;
    let source = if selected_backend == "trellis2" {
        trellis_source.clone()
    } else {
        hunyuan_source.clone()
    };
    let models = resolve_model_dir(&app, &resolved_options)?;
    let outputs = default_output_dir()?;
    let jobs = jobs_dir(&app)?;
    let python_command = find_python();
    let tencent_ready = model_cache_has_snapshot(&models, "tencent/Hunyuan3D-2.1");
    let hunyuan_env_ready = hunyuan_env_ready(&env);
    let trellis_env_ready = trellis_env_ready(&env);
    let hunyuan_source_ready = hunyuan_source.join(HUNYUAN_SOURCE_MARKER).is_file();
    let trellis_source_ready = trellis_source.join(TRELLIS_SOURCE_MARKER).is_file();
    let hunyuan_models_ready = models.join(HUNYUAN_MODEL_MARKER).is_file() || tencent_ready;
    let trellis_models_ready = models.join(TRELLIS_MODEL_MARKER).is_file();
    let selected_models_ready = if selected_backend == "trellis2" {
        trellis_models_ready
    } else {
        hunyuan_models_ready
    };
    let selected_source_ready = if selected_backend == "trellis2" {
        trellis_source_ready
    } else {
        hunyuan_source_ready
    };

    Ok(RuntimeStatus {
        python_found: python_command.is_some() || uv_available(),
        python_command: python_command.or_else(|| {
            uv_available().then(|| "uv (installs Python 3.11 during setup)".to_string())
        }),
        app_data_dir: app_data.display().to_string(),
        worker_path: worker.display().to_string(),
        env_dir: env.display().to_string(),
        env_python: env_python.display().to_string(),
        env_ready: env_python.is_file() && env.join(ENV_MARKER).is_file(),
        source_dir: source.display().to_string(),
        source_ready: selected_source_ready,
        model_dir: models.display().to_string(),
        models_ready: selected_models_ready,
        selected_backend,
        any_models_ready: hunyuan_models_ready || trellis_models_ready,
        hunyuan_env_ready,
        hunyuan_source_ready,
        hunyuan_models_ready,
        trellis_env_ready,
        trellis_source_ready,
        trellis_models_ready,
        shape_model_ready: tencent_ready,
        paint_model_ready: tencent_ready
            && model_cache_has_snapshot(&models, "facebook/dinov2-giant"),
        texture_wheels_ready: env.join(WHEEL_MARKER).is_file(),
        realesrgan_ready: hunyuan_source
            .join("hy3dpaint")
            .join("ckpt")
            .join("RealESRGAN_x4plus.pth")
            .is_file(),
        env_runtime_profile: env_runtime_profile(&env),
        nvidia_gpu_found: nvidia_smi_available(),
        output_dir: outputs.display().to_string(),
        jobs_dir: jobs.display().to_string(),
    })
}

#[tauri::command]
fn get_vram_status() -> VramStatus {
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=index,name,compute_cap,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    hide_command_window(&mut command);

    let output = match command.output() {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            return VramStatus {
                available: false,
                gpus: Vec::new(),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            };
        }
        Err(error) => {
            return VramStatus {
                available: false,
                gpus: Vec::new(),
                message: format!("VRAM unavailable: {error}"),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let gpus = stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<_> = line.split(',').map(str::trim).collect();
            if parts.len() < 5 {
                return None;
            }
            Some(GpuVram {
                index: parts[0].to_string(),
                name: parts[1].to_string(),
                compute_cap: parts[2].to_string(),
                memory_used_mb: parts[3].parse().ok()?,
                memory_total_mb: parts[4].parse().ok()?,
            })
        })
        .collect::<Vec<_>>();

    VramStatus {
        available: !gpus.is_empty(),
        message: if gpus.is_empty() {
            "No NVIDIA GPUs reported.".to_string()
        } else {
            "OK".to_string()
        },
        gpus,
    }
}

#[tauri::command]
fn resolve_inputs(paths: Vec<String>, recursive: bool) -> Vec<String> {
    let mut output = Vec::new();
    for path in paths {
        collect_images(Path::new(&path), recursive, &mut output);
    }
    output.sort();
    output.dedup();
    output
}

#[tauri::command]
fn preview_image_data_url(path: String) -> Result<String, String> {
    let image_path = PathBuf::from(path);
    if !image_path.is_file() {
        return Err("Preview image was not found.".to_string());
    }
    if let Ok(data_url) = preview_thumbnail_data_url(&image_path) {
        return Ok(data_url);
    }

    let mime_type = image_mime_type(&image_path)?;
    let bytes =
        fs::read(&image_path).map_err(|error| format!("Unable to read preview image: {error}"))?;
    Ok(format!(
        "data:{mime_type};base64,{}",
        general_purpose::STANDARD.encode(bytes)
    ))
}

#[tauri::command]
fn get_autoremesher_status(app: AppHandle) -> Result<RemesherStatus, String> {
    Ok(autoremesher::status(&app_data_dir(&app)?))
}

#[tauri::command]
async fn install_autoremesher(app: AppHandle) -> Result<RemesherStatus, String> {
    let app_data = app_data_dir(&app)?;
    tauri::async_runtime::spawn_blocking(move || autoremesher::install(&app_data))
        .await
        .map_err(|error| format!("AutoRemesher installer task failed: {error}"))?
}

#[tauri::command]
async fn run_autoremesher(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    active_remesh: State<'_, ActiveRemeshState>,
    options: RemeshOptions,
) -> Result<RemeshResult, String> {
    if active_jobs.is_busy()? {
        return Err("Wait for the current generation job to finish before remeshing.".to_string());
    }
    let app_data = app_data_dir(&app)?;
    let remesh_state = active_remesh.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        autoremesher::run(&app_data, &remesh_state, options)
    })
    .await
    .map_err(|error| format!("AutoRemesher task failed: {error}"))?
}

/// Agent API remesh runs happen on a background thread (the agent server is
/// single-threaded); this holds whether one is running and its last outcome.
static AGENT_REMESH_RUN: Mutex<(bool, Option<serde_json::Value>)> = Mutex::new((false, None));

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentRemeshRequest {
    input_path: String,
    target_quads: Option<u32>,
    edge_scaling: Option<f64>,
    sharp_edge: Option<f64>,
    smooth_normal: Option<f64>,
    adaptivity: Option<f64>,
    anisotropy: Option<f64>,
}

impl AgentRemeshRequest {
    /// Missing fields take the UI defaults.
    fn into_options(self) -> RemeshOptions {
        RemeshOptions {
            input_path: self.input_path,
            target_quads: self.target_quads.unwrap_or(50_000),
            edge_scaling: self.edge_scaling.unwrap_or(1.0),
            sharp_edge: self.sharp_edge.unwrap_or(90.0),
            smooth_normal: self.smooth_normal.unwrap_or(0.0),
            adaptivity: self.adaptivity.unwrap_or(1.0),
            anisotropy: self.anisotropy.unwrap_or(1.0),
        }
    }
}

fn agent_remesh_status(app: &AppHandle) -> Result<serde_json::Value, String> {
    let status = autoremesher::status(&app_data_dir(app)?);
    let (running, last) = AGENT_REMESH_RUN
        .lock()
        .map(|run| run.clone())
        .map_err(|_| "Unable to lock AutoRemesher agent state.".to_string())?;
    let mut value = serde_json::to_value(status).map_err(|error| error.to_string())?;
    if let Some(object) = value.as_object_mut() {
        object.insert("running".to_string(), serde_json::Value::Bool(running));
        object.insert("lastRun".to_string(), last.unwrap_or(serde_json::Value::Null));
    }
    Ok(value)
}

fn start_agent_remesh(
    app: &AppHandle,
    active_jobs: &ActiveJobState,
    body: &[u8],
) -> Result<serde_json::Value, String> {
    if active_jobs.is_busy()? {
        return Err("Wait for the current generation job to finish before remeshing.".to_string());
    }
    let request: AgentRemeshRequest = serde_json::from_slice(body)
        .map_err(|error| format!("Invalid JSON request (inputPath is required): {error}"))?;
    let options = request.into_options();
    let app_data = app_data_dir(app)?;
    let remesh_state = app.state::<ActiveRemeshState>().inner().clone();
    {
        let mut run = AGENT_REMESH_RUN
            .lock()
            .map_err(|_| "Unable to lock AutoRemesher agent state.".to_string())?;
        if run.0 {
            return Err("Another remesh is already running.".to_string());
        }
        *run = (true, None);
    }
    let input_path = options.input_path.clone();
    thread::spawn(move || {
        let outcome = match autoremesher::run(&app_data, &remesh_state, options) {
            Ok(result) => serde_json::json!({ "ok": true, "inputPath": input_path, "result": result }),
            Err(error) => serde_json::json!({ "ok": false, "inputPath": input_path, "error": error }),
        };
        if let Ok(mut run) = AGENT_REMESH_RUN.lock() {
            *run = (false, Some(outcome));
        }
    });
    Ok(serde_json::json!({ "ok": true, "started": true }))
}

#[tauri::command]
fn cancel_autoremesher(active_remesh: State<'_, ActiveRemeshState>) -> Result<(), String> {
    active_remesh.cancel()
}

fn preview_thumbnail_data_url(image_path: &Path) -> Result<String, String> {
    let image = image::ImageReader::open(image_path)
        .map_err(|error| format!("Unable to open preview image: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("Unable to detect preview format: {error}"))?
        .decode()
        .map_err(|error| format!("Unable to decode preview image: {error}"))?;
    let thumbnail = image.thumbnail(512, 512).to_rgb8();
    let mut encoded = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 84);
    encoder
        .encode_image(&thumbnail)
        .map_err(|error| format!("Unable to encode preview thumbnail: {error}"))?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        general_purpose::STANDARD.encode(encoded.into_inner())
    ))
}

fn voxel_sidecar_path(model_path: &Path) -> PathBuf {
    model_path.with_extension("ih3dvox.json")
}

#[tauri::command]
fn read_voxel_sidecar(model_path: String) -> Result<Option<String>, String> {
    let model_path = PathBuf::from(model_path);
    if !model_path.is_file() {
        return Ok(None);
    }
    let sidecar_path = voxel_sidecar_path(&model_path);
    if !sidecar_path.is_file() {
        return Ok(None);
    }
    let metadata = fs::metadata(&sidecar_path)
        .map_err(|error| format!("Unable to read voxel preview metadata: {error}"))?;
    if metadata.len() > 64 * 1024 * 1024 {
        return Err("Voxel preview data is too large to load.".to_string());
    }
    fs::read_to_string(&sidecar_path)
        .map(Some)
        .map_err(|error| format!("Unable to read voxel preview data: {error}"))
}

#[tauri::command]
fn open_path(app: AppHandle, path: String) -> Result<(), String> {
    let requested_path = PathBuf::from(path);
    if !requested_path.exists() {
        return Err(format!("Path does not exist: {}", requested_path.display()));
    }
    app.opener()
        .open_path(requested_path.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open path: {error}"))
}

#[tauri::command]
fn open_containing_folder(app: AppHandle, path: String) -> Result<(), String> {
    let requested_path = PathBuf::from(path);
    let folder = if requested_path.is_dir() {
        requested_path
    } else {
        requested_path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "Unable to resolve containing folder.".to_string())?
    };
    if !folder.is_dir() {
        return Err(format!("Folder does not exist: {}", folder.display()));
    }
    app.opener()
        .open_path(folder.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open folder: {error}"))
}

#[tauri::command]
fn open_runtime_folder(app: AppHandle) -> Result<(), String> {
    let folder = app_data_dir(&app)?;
    app.opener()
        .open_path(folder.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open runtime folder: {error}"))
}

#[tauri::command]
fn open_output_folder(app: AppHandle) -> Result<(), String> {
    let folder = output_dir(&app)?;
    app.opener()
        .open_path(folder.display().to_string(), None::<&str>)
        .map_err(|error| format!("Unable to open output folder: {error}"))
}

fn ensure_managed_worker(
    app: &AppHandle,
    active_jobs: &ActiveJobState,
    managed_workers: &ManagedWorkerState,
    options: &HunyuanOptions,
) -> Result<ManagedWorker, String> {
    let runtime_profile = normalize_runtime_profile(&options.runtime_profile);
    let model_backend = normalize_model_backend(&options.model_backend);
    let cuda_device = options.cuda_device.clone();
    if let Some(worker) =
        managed_workers.get_matching(&model_backend, &runtime_profile, &cuda_device)?
    {
        return Ok(worker);
    }
    managed_workers.stop()?;

    let env = env_dir_for_profile_opts(app, options, &runtime_profile)?;
    let worker_path = worker_path(app)?;
    let app_data = app_data_dir(app)?;
    let mut command = Command::new(env_python_path(&env));
    configure_worker_command(&mut command);
    configure_managed_worker_env(
        &mut command,
        &app_data,
        &env,
        &hunyuan_source_dir_opts(app, options)?,
        &trellis_source_dir_opts(app, options)?,
        &resolve_model_dir(app, options)?,
        &runtime_dir_opts(app, options)?,
    );
    command
        .arg(worker_path)
        .arg("--daemon")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start managed Image To 3D worker: {error}"))?;
    let pid = child.id();
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Unable to open managed worker stdin.".to_string())?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let worker = ManagedWorker {
        pid,
        model_backend,
        runtime_profile,
        cuda_device,
        stdin: Arc::new(Mutex::new(stdin)),
    };
    managed_workers.set(worker.clone())?;

    let app_for_stdout = app.clone();
    let app_for_stderr = app.clone();
    let app_for_wait = app.clone();
    let active_jobs_for_stdout = active_jobs.clone();
    let active_jobs_for_wait = active_jobs.clone();
    let managed_workers_for_wait = managed_workers.clone();
    thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let event_type = value
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    let completes_job = matches!(event_type, "done" | "error" | "canceled");
                    let _ = active_jobs_for_stdout.record_event(&value);
                    let _ = app_for_stdout.emit("hunyuan-worker-event", value);
                    if completes_job {
                        let _ = active_jobs_for_stdout.finish_current();
                    }
                } else {
                    let value = serde_json::json!({ "type": "setup", "message": line });
                    let _ = active_jobs_for_stdout.record_event(&value);
                    let _ = app_for_stdout.emit("hunyuan-worker-event", value);
                }
            }
        }
    });

    thread::spawn(move || {
        if let Some(stderr) = stderr {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_for_stderr.emit(
                    "hunyuan-worker-event",
                    serde_json::json!({ "type": "setup", "message": line }),
                );
            }
        }
    });

    thread::spawn(move || {
        let status = child.wait();
        let _ = managed_workers_for_wait.clear_if_pid(pid);
        let active = active_jobs_for_wait.finish_if_pid(pid).unwrap_or(None);
        if let Some(was_canceled) = active {
            if was_canceled {
                let value = serde_json::json!({ "type": "canceled", "message": "Job canceled." });
                let _ = active_jobs_for_wait.record_last_event(&value);
                let _ = app_for_wait.emit("hunyuan-worker-event", value);
            } else if !matches!(status, Ok(status) if status.success()) {
                let value = serde_json::json!({ "type": "error", "message": "Managed Image To 3D worker exited with an error." });
                let _ = active_jobs_for_wait.record_last_event(&value);
                let _ = app_for_wait.emit("hunyuan-worker-event", value);
            }
        }
    });

    Ok(worker)
}

fn start_worker_job_core(
    app: AppHandle,
    active_jobs: ActiveJobState,
    managed_workers: ManagedWorkerState,
    action: &str,
    paths: Vec<String>,
    mut options: HunyuanOptions,
) -> Result<String, String> {
    if active_jobs.is_busy()? {
        return Err("Another job is already running.".to_string());
    }
    if action == "generate" && paths.is_empty() {
        return Err("No input images were provided.".to_string());
    }
    let requested_backend = normalize_model_backend(&options.model_backend);
    if action == "generate"
        && requested_backend != "trellis2"
        && options.mode == "textureOnly"
        && options.mesh_input_path.trim().is_empty()
    {
        return Err("Texture-only mode needs a source GLB mesh.".to_string());
    }

    let worker = worker_path(&app)?;
    let app_data = app_data_dir(&app)?;
    let jobs = jobs_dir(&app)?;
    normalize_options_for_job(&mut options)?;
    let requested_profile = normalize_runtime_profile(&options.runtime_profile);

    let job_id = Uuid::new_v4().to_string();
    let job_path = jobs.join(format!("{job_id}.json"));
    let job = WorkerJob {
        action: action.to_string(),
        job_id: job_id.clone(),
        paths,
        options,
        app_data_dir: app_data.display().to_string(),
    };
    let job_json = serde_json::to_string_pretty(&job)
        .map_err(|error| format!("Unable to serialize job: {error}"))?;
    fs::write(&job_path, job_json).map_err(|error| format!("Unable to write job: {error}"))?;

    if action == "generate" {
        let worker = ensure_managed_worker(&app, &active_jobs, &managed_workers, &job.options)?;
        active_jobs.start(job_id.clone(), worker.pid)?;
        let request = serde_json::json!({ "jobPath": job_path.display().to_string() }).to_string();
        let mut stdin = worker
            .stdin
            .lock()
            .map_err(|_| "Unable to lock managed worker stdin.".to_string())?;
        if let Err(error) = writeln!(stdin, "{request}") {
            let _ = active_jobs.finish(&job_id);
            let _ = managed_workers.clear_if_pid(worker.pid);
            let _ = kill_process_tree(worker.pid);
            return Err(format!("Unable to send job to managed worker: {error}"));
        }
        return Ok(job_id);
    }

    let mut command = if action == "generate" {
        let env = env_dir_for_profile_opts(&app, &job.options, &requested_profile)?;
        Command::new(env_python_path(&env))
    } else {
        let python = find_or_install_python()?;
        command_from_python_spec(&python)
    };
    configure_worker_command(&mut command);
    if action == "generate" {
        let env = env_dir_for_profile_opts(&app, &job.options, &requested_profile)?;
        configure_managed_worker_env(
            &mut command,
            &app_data,
            &env,
            &hunyuan_source_dir_opts(&app, &job.options)?,
            &trellis_source_dir_opts(&app, &job.options)?,
            &resolve_model_dir(&app, &job.options)?,
            &runtime_dir_opts(&app, &job.options)?,
        );
    } else {
        configure_worker_storage_env(&mut command, &resolve_model_dir(&app, &job.options)?);
        command
            .env("I3D_ENV_DIR", env_dir_for_profile_opts(&app, &job.options, &requested_profile)?)
            .env("I3D_RUNTIME_DIR", runtime_dir_opts(&app, &job.options)?);
    }
    command
        .arg(worker)
        .arg("--job")
        .arg(job_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start Image To 3D worker: {error}"))?;
    let child_pid = child.id();
    if let Err(error) = active_jobs.start(job_id.clone(), child_pid) {
        let _ = kill_process_tree(child_pid);
        return Err(error);
    }

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_for_stdout = app.clone();
    let app_for_stderr = app.clone();
    let app_for_wait = app.clone();
    let active_jobs_for_stdout = active_jobs.clone();
    let active_jobs_for_wait = active_jobs.clone();
    let job_id_for_wait = job_id.clone();

    thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = active_jobs_for_stdout.record_event(&value);
                    let _ = app_for_stdout.emit("hunyuan-worker-event", value);
                } else {
                    let value = serde_json::json!({ "type": "setup", "message": line });
                    let _ = active_jobs_for_stdout.record_event(&value);
                    let _ = app_for_stdout.emit("hunyuan-worker-event", value);
                }
            }
        }
    });

    thread::spawn(move || {
        if let Some(stderr) = stderr {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_for_stderr.emit(
                    "hunyuan-worker-event",
                    serde_json::json!({ "type": "setup", "message": line }),
                );
            }
        }
    });

    thread::spawn(move || {
        let status = child.wait();
        let was_canceled = active_jobs_for_wait
            .finish(&job_id_for_wait)
            .unwrap_or(false);
        if was_canceled {
            let value = serde_json::json!({
                "type": "canceled",
                "message": "Job canceled."
            });
            let _ = active_jobs_for_wait.record_last_event(&value);
            let _ = app_for_wait.emit("hunyuan-worker-event", value);
        } else if !matches!(status, Ok(status) if status.success()) {
            let value = serde_json::json!({
                "type": "error",
                "message": "Image To 3D worker exited with an error."
            });
            let _ = active_jobs_for_wait.record_last_event(&value);
            let _ = app_for_wait.emit("hunyuan-worker-event", value);
        }
    });

    Ok(job_id)
}

fn start_worker_job(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    action: &str,
    paths: Vec<String>,
    options: HunyuanOptions,
) -> Result<String, String> {
    start_worker_job_core(
        app,
        active_jobs.inner().clone(),
        managed_workers.inner().clone(),
        action,
        paths,
        options,
    )
}

#[tauri::command]
fn prepare_runtime(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    options: HunyuanOptions,
) -> Result<String, String> {
    if active_jobs.is_busy()? {
        return Err("Wait for the current job to finish before preparing the runtime.".to_string());
    }
    managed_workers.stop()?;
    start_worker_job(
        app,
        active_jobs,
        managed_workers,
        "setup",
        Vec::new(),
        options,
    )
}

#[tauri::command]
fn start_generation_job(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    paths: Vec<String>,
    options: HunyuanOptions,
) -> Result<String, String> {
    runtime_ready_for_generation(&app, &options)?;
    start_worker_job(
        app,
        active_jobs,
        managed_workers,
        "generate",
        paths,
        options,
    )
}

#[tauri::command]
fn cancel_active_job(app: AppHandle, active_jobs: State<'_, ActiveJobState>) -> Result<(), String> {
    let job_id = active_jobs.active_job_summary()?.map(|summary| summary.id);
    let pid = active_jobs.request_cancel()?;
    if let Some(job_id) = job_id {
        let _ = mark_job_manifest_status(&app, &job_id, "canceled", "Job canceled.");
    }
    kill_process_tree(pid)
}

#[tauri::command]
fn reset_python_runtime(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    runtime_profile: Option<String>,
    model_backend: Option<String>,
) -> Result<RuntimeStatus, String> {
    if active_jobs.is_busy()? {
        return Err("Wait for the current job to finish before resetting the runtime.".to_string());
    }
    managed_workers.stop()?;
    let requested_profile = normalize_runtime_profile(
        runtime_profile
            .as_deref()
            .unwrap_or(DEFAULT_PYTORCH_PROFILE),
    );
    let env = env_dir_for_profile(&app, &requested_profile)?;
    if env.exists() {
        fs::remove_dir_all(&env)
            .map_err(|error| format!("Unable to reset Python runtime: {error}"))?;
    }
    check_runtime(app, Some(requested_profile), model_backend, None, None, None)
}

#[tauri::command]
fn unload_models(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    runtime_profile: Option<String>,
    model_backend: Option<String>,
) -> Result<RuntimeStatus, String> {
    if active_jobs.is_busy()? {
        return Err("Wait for the current job to finish before unloading models.".to_string());
    }
    managed_workers.stop()?;
    check_runtime(app, runtime_profile, model_backend, None, None, None)
}

struct HttpRequest {
    method: String,
    path: String,
    query: Option<String>,
    body: Vec<u8>,
}

fn validate_agent_api_port(port: Option<u16>) -> Result<u16, String> {
    let port = port
        .or_else(read_registered_agent_api_port)
        .unwrap_or(DEFAULT_AGENT_API_PORT);
    if port == 0 {
        return Err("Agent API port must be between 1 and 65535.".to_string());
    }
    Ok(port)
}

fn agent_api_url(port: u16) -> String {
    format!("http://{AGENT_API_BIND_ADDRESS}:{port}")
}

fn timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

fn shared_neko_legends_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library").join("Application Support"))
    } else {
        env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }?;
    Some(base.join("NekoLegends"))
}

fn agent_api_registry_path() -> Option<PathBuf> {
    Some(shared_neko_legends_dir()?.join(AGENT_API_REGISTRY_FILE))
}

fn read_agent_api_registry() -> AgentApiRegistry {
    let updated_at = timestamp_string();
    let Some(path) = agent_api_registry_path() else {
        return AgentApiRegistry {
            updated_at,
            apps: Vec::new(),
        };
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or(AgentApiRegistry {
            updated_at,
            apps: Vec::new(),
        })
}

fn read_registered_agent_api_port() -> Option<u16> {
    read_agent_api_registry()
        .apps
        .into_iter()
        .find(|entry| entry.app_id == AGENT_APP_ID)
        .map(|entry| entry.port)
        .filter(|port| *port > 0)
}

fn publish_agent_api_status(status: &AgentServerStatus) {
    let Some(path) = agent_api_registry_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut registry = read_agent_api_registry();
    let updated_at = timestamp_string();
    let entry = AgentApiRegistryEntry {
        app_id: AGENT_APP_ID.to_string(),
        app_name: AGENT_APP_NAME.to_string(),
        default_port: DEFAULT_AGENT_API_PORT,
        bind_address: AGENT_API_BIND_ADDRESS.to_string(),
        port: status.port,
        enabled: status.enabled,
        url: status.url.clone(),
        openapi_url: status.openapi_url.clone(),
        busy: status.busy,
        active_job_id: status.active_job_id.clone(),
        last_seen: Some(updated_at.clone()),
        note: Some("Local Agent API.".to_string()),
    };
    if let Some(existing) = registry
        .apps
        .iter_mut()
        .find(|entry| entry.app_id == AGENT_APP_ID)
    {
        *existing = entry;
    } else {
        registry.apps.push(entry);
    }
    registry.updated_at = updated_at;
    if let Ok(raw) = serde_json::to_string_pretty(&registry) {
        let _ = fs::write(path, raw);
    }
}

fn agent_status_from(
    agent_state: &AgentServerState,
    active_jobs: &ActiveJobState,
) -> Result<AgentServerStatus, String> {
    let (enabled, port) = {
        let control = agent_state
            .inner
            .lock()
            .map_err(|_| "Unable to lock agent server state.".to_string())?;
        (control.enabled, control.port())
    };
    let active_job = active_jobs.active_job_summary()?;
    let active_job_id = active_job.as_ref().map(|job| job.id.clone());
    let last_job = active_jobs.last_job_summary()?;
    let status = AgentServerStatus {
        enabled,
        port,
        url: agent_api_url(port),
        openapi_url: format!("{}/openapi.json", agent_api_url(port)),
        busy: active_job_id.is_some(),
        active_job_id,
        active_job,
        last_job,
        message: if enabled {
            "Agent API is enabled.".to_string()
        } else {
            "Agent API is off.".to_string()
        },
    };
    publish_agent_api_status(&status);
    Ok(status)
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("Unable to set read timeout: {error}"))?;
    let mut data = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut expected_len: Option<usize> = None;

    loop {
        let bytes_read = stream
            .read(&mut buffer)
            .map_err(|error| format!("Unable to read agent request: {error}"))?;
        if bytes_read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..bytes_read]);
        if let Some(header_end) = find_header_end(&data) {
            if expected_len.is_none() {
                let headers = String::from_utf8_lossy(&data[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                expected_len = Some(header_end + 4 + content_length);
            }
            if expected_len.is_some_and(|len| data.len() >= len) {
                break;
            }
        }
        if data.len() > 2 * 1024 * 1024 {
            return Err("Agent request is too large.".to_string());
        }
    }

    let header_end = find_header_end(&data).ok_or_else(|| "Invalid HTTP request.".to_string())?;
    let headers = String::from_utf8_lossy(&data[..header_end]);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| "Invalid HTTP request line.".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/").to_string();
    let (path, query) = match raw_path.split_once('?') {
        Some((path, query)) => (path.to_string(), Some(query.to_string())),
        None => (raw_path, None),
    };
    let body_start = header_end + 4;
    let body = if body_start <= data.len() {
        data[body_start..].to_vec()
    } else {
        Vec::new()
    };
    Ok(HttpRequest {
        method,
        path,
        query,
        body,
    })
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_json_response(
    stream: &mut TcpStream,
    status: &str,
    payload: serde_json::Value,
) -> Result<(), String> {
    let body = serde_json::to_vec(&payload)
        .map_err(|error| format!("Unable to serialize agent response: {error}"))?;
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|error| format!("Unable to write agent response: {error}"))
}

fn write_empty_response(stream: &mut TcpStream, status: &str) -> Result<(), String> {
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(headers.as_bytes())
        .map_err(|error| format!("Unable to write agent response: {error}"))
}

fn agent_openapi(port: u16) -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Image To 3D Agent API",
            "version": env!("CARGO_PKG_VERSION")
        },
        "servers": [{ "url": agent_api_url(port) }],
        "paths": {
            "/health": { "get": { "summary": "Check API status" } },
            "/status": { "get": { "summary": "Check active job status" } },
            "/history": { "get": { "summary": "List retained job history and measured preset averages" } },
            "/runtime": { "get": { "summary": "Check local runtime readiness" } },
            "/vram": { "get": { "summary": "Check NVIDIA VRAM" } },
            "/prepare": { "post": { "summary": "Prepare the standalone runtime" } },
            "/reset-runtime": { "post": { "summary": "Reset the managed Python runtime" } },
            "/unload-models": { "post": { "summary": "Stop the warm worker and release loaded models" } },
            "/open-output-folder": { "post": { "summary": "Open the default output folder" } },
            "/open-runtime-folder": { "post": { "summary": "Open the app runtime folder" } },
            "/open-path": { "post": { "summary": "Open a local file or folder" } },
            "/open-containing-folder": { "post": { "summary": "Open the containing folder for a local path" } },
            "/generate": { "post": { "summary": "Start an Image To 3D generation job" } },
            "/cancel": { "post": { "summary": "Cancel the active job" } },
            "/autoremesher": { "get": { "summary": "AutoRemesher install status, whether a remesh is running, and the last remesh result" } },
            "/autoremesher/install": { "post": { "summary": "Download, verify and install the managed AutoRemesher build" } },
            "/autoremesher/run": { "post": { "summary": "Start a remesh: {inputPath (OBJ/GLB/GLTF), targetQuads?, edgeScaling?, sharpEdge?, smoothNormal?, adaptivity?, anisotropy?}; poll GET /autoremesher" } },
            "/autoremesher/cancel": { "post": { "summary": "Cancel the running remesh" } }
        }
    })
}

fn parse_agent_job_request(body: &[u8]) -> Result<AgentJobRequest, String> {
    if body.is_empty() {
        return Ok(AgentJobRequest {
            paths: None,
            options: None,
        });
    }
    serde_json::from_slice(body).map_err(|error| format!("Invalid JSON request: {error}"))
}

fn parse_agent_path_request(body: &[u8]) -> Result<AgentPathRequest, String> {
    if body.is_empty() {
        return Err("Request body must include a path.".to_string());
    }
    let request: AgentPathRequest =
        serde_json::from_slice(body).map_err(|error| format!("Invalid JSON request: {error}"))?;
    if request.path.trim().is_empty() {
        return Err("Request path cannot be empty.".to_string());
    }
    Ok(request)
}

fn query_value(query: Option<&str>, name: &str) -> Option<String> {
    query.and_then(|query| {
        query.split('&').find_map(|part| {
            let (key, value) = part.split_once('=')?;
            if key == name && !value.trim().is_empty() {
                Some(value.trim().to_string())
            } else {
                None
            }
        })
    })
}

fn agent_options_from_request(
    options: Option<serde_json::Value>,
) -> Result<HunyuanOptions, String> {
    let mut merged = serde_json::to_value(default_hunyuan_options())
        .map_err(|error| format!("Unable to build default options: {error}"))?;
    let Some(options) = options else {
        return serde_json::from_value(merged)
            .map_err(|error| format!("Unable to read default options: {error}"));
    };
    if options.is_null() {
        return serde_json::from_value(merged)
            .map_err(|error| format!("Unable to read default options: {error}"));
    }
    let overrides = options
        .as_object()
        .ok_or_else(|| "Agent options must be a JSON object.".to_string())?;
    let base = merged
        .as_object_mut()
        .ok_or_else(|| "Unable to merge default options.".to_string())?;
    let mesh_numbers_overridden = ["meshSteps", "octreeResolution", "numChunks", "maxFaces"]
        .iter()
        .any(|key| overrides.contains_key(*key));
    if mesh_numbers_overridden && !overrides.contains_key("meshDetail") {
        base.insert(
            "meshDetail".to_string(),
            serde_json::Value::String("custom".to_string()),
        );
    }
    let trellis_settings_overridden = [
        "trellisResolution",
        "trellisSparseSteps",
        "trellisShapeSteps",
        "trellisTextureSteps",
        "trellisTextureGuidance",
        "trellisDecimationTarget",
        "trellisTextureSize",
        "trellisRemesh",
    ]
    .iter()
    .any(|key| overrides.contains_key(*key));
    if trellis_settings_overridden && !overrides.contains_key("trellisPreset") {
        base.insert(
            "trellisPreset".to_string(),
            serde_json::Value::String("custom".to_string()),
        );
    }
    for (key, value) in overrides {
        base.insert(key.clone(), value.clone());
    }
    serde_json::from_value(merged).map_err(|error| format!("Invalid agent options: {error}"))
}

fn handle_agent_route(
    request: HttpRequest,
    app: &AppHandle,
    active_jobs: &ActiveJobState,
    managed_workers: &ManagedWorkerState,
    agent_state: &AgentServerState,
) -> Result<serde_json::Value, String> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => Ok(serde_json::json!({
            "ok": true,
            "service": "Image To 3D",
            "version": env!("CARGO_PKG_VERSION"),
            "url": agent_status_from(agent_state, active_jobs)?.url
        })),
        ("GET", "/openapi.json") => Ok(agent_openapi(
            agent_status_from(agent_state, active_jobs)?.port,
        )),
        ("GET", "/status") => {
            let status = agent_status_from(agent_state, active_jobs)?;
            serde_json::to_value(status).map_err(|error| error.to_string())
        }
        ("GET", "/history") => {
            serde_json::to_value(list_job_history(app.clone())?).map_err(|error| error.to_string())
        }
        ("GET", "/runtime") => {
            let runtime_profile = query_value(request.query.as_deref(), "runtimeProfile");
            let model_backend = query_value(request.query.as_deref(), "modelBackend");
            let model_dir = query_value(request.query.as_deref(), "modelDir");
            let env_dir = query_value(request.query.as_deref(), "envDir");
            let runtime_dir = query_value(request.query.as_deref(), "runtimeDir");
            serde_json::to_value(check_runtime(app.clone(), runtime_profile, model_backend, model_dir, env_dir, runtime_dir)?)
                .map_err(|error| error.to_string())
        }
        ("GET", "/vram") => {
            serde_json::to_value(get_vram_status()).map_err(|error| error.to_string())
        }
        ("POST", "/prepare") => {
            if active_jobs.is_busy()? {
                return Err(
                    "Wait for the current job to finish before preparing the runtime.".to_string(),
                );
            }
            let request = parse_agent_job_request(&request.body)?;
            let options = agent_options_from_request(request.options)?;
            managed_workers.stop()?;
            let job_id = start_worker_job_core(
                app.clone(),
                active_jobs.clone(),
                managed_workers.clone(),
                "setup",
                Vec::new(),
                options,
            )?;
            Ok(serde_json::json!({ "ok": true, "jobId": job_id }))
        }
        ("POST", "/generate") => {
            let request = parse_agent_job_request(&request.body)?;
            let paths = request.paths.unwrap_or_default();
            let options = agent_options_from_request(request.options)?;
            runtime_ready_for_generation(app, &options)?;
            let job_id = start_worker_job_core(
                app.clone(),
                active_jobs.clone(),
                managed_workers.clone(),
                "generate",
                paths,
                options,
            )?;
            Ok(serde_json::json!({ "ok": true, "jobId": job_id }))
        }
        ("POST", "/reset-runtime") => {
            if active_jobs.is_busy()? {
                return Err(
                    "Wait for the current job to finish before resetting the runtime.".to_string(),
                );
            }
            managed_workers.stop()?;
            let request = parse_agent_job_request(&request.body)?;
            let options = agent_options_from_request(request.options)?;
            let runtime_profile = normalize_runtime_profile(&options.runtime_profile);
            let env = env_dir_for_profile_opts(app, &options, &runtime_profile)?;
            if env.exists() {
                fs::remove_dir_all(&env)
                    .map_err(|error| format!("Unable to reset Python runtime: {error}"))?;
            }
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/unload-models") => {
            if active_jobs.is_busy()? {
                return Err(
                    "Wait for the current job to finish before unloading models.".to_string(),
                );
            }
            managed_workers.stop()?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/open-output-folder") => {
            open_output_folder(app.clone())?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/open-runtime-folder") => {
            open_runtime_folder(app.clone())?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/open-path") => {
            let request = parse_agent_path_request(&request.body)?;
            open_path(app.clone(), request.path)?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/open-containing-folder") => {
            let request = parse_agent_path_request(&request.body)?;
            open_containing_folder(app.clone(), request.path)?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("POST", "/cancel") => {
            let pid = active_jobs.request_cancel()?;
            kill_process_tree(pid)?;
            Ok(serde_json::json!({ "ok": true }))
        }
        ("GET", "/autoremesher") => agent_remesh_status(app),
        ("POST", "/autoremesher/install") => {
            let status = autoremesher::install(&app_data_dir(app)?)?;
            serde_json::to_value(status).map_err(|error| error.to_string())
        }
        ("POST", "/autoremesher/run") => start_agent_remesh(app, active_jobs, &request.body),
        ("POST", "/autoremesher/cancel") => {
            app.state::<ActiveRemeshState>().cancel()?;
            Ok(serde_json::json!({ "ok": true }))
        }
        _ => Err(format!(
            "No agent endpoint for {} {}",
            request.method, request.path
        )),
    }
}

fn handle_agent_stream(
    mut stream: TcpStream,
    app: &AppHandle,
    active_jobs: &ActiveJobState,
    managed_workers: &ManagedWorkerState,
    agent_state: &AgentServerState,
) {
    let result = read_http_request(&mut stream).and_then(|request| {
        if request.method == "OPTIONS" {
            return write_empty_response(&mut stream, "204 No Content");
        }
        match handle_agent_route(request, app, active_jobs, managed_workers, agent_state) {
            Ok(payload) => write_json_response(&mut stream, "200 OK", payload),
            Err(error) => write_json_response(
                &mut stream,
                "400 Bad Request",
                serde_json::json!({ "ok": false, "error": error }),
            ),
        }
    });
    if let Err(error) = result {
        let _ = write_json_response(
            &mut stream,
            "400 Bad Request",
            serde_json::json!({ "ok": false, "error": error }),
        );
    }
}

fn run_agent_server(
    listener: TcpListener,
    app: AppHandle,
    active_jobs: ActiveJobState,
    managed_workers: ManagedWorkerState,
    agent_state: AgentServerState,
    stop: Arc<AtomicBool>,
) {
    let _ = listener.set_nonblocking(true);
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                handle_agent_stream(stream, &app, &active_jobs, &managed_workers, &agent_state)
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(80));
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(150));
            }
        }
    }
}

#[tauri::command]
fn get_agent_server_status(
    agent_state: State<'_, AgentServerState>,
    active_jobs: State<'_, ActiveJobState>,
) -> Result<AgentServerStatus, String> {
    agent_status_from(agent_state.inner(), active_jobs.inner())
}

fn set_agent_server_enabled_inner(
    app: AppHandle,
    agent_state: &AgentServerState,
    active_jobs: &ActiveJobState,
    managed_workers: &ManagedWorkerState,
    enabled: bool,
    port: Option<u16>,
) -> Result<AgentServerStatus, String> {
    let port = validate_agent_api_port(port)?;
    {
        let mut control = agent_state
            .inner
            .lock()
            .map_err(|_| "Unable to lock agent server state.".to_string())?;

        if control.enabled && (!enabled || control.port() != port) {
            if let Some(stop) = control.stop.take() {
                stop.store(true, Ordering::SeqCst);
            }
            control.enabled = false;
        }
        control.port = port;

        if enabled && !control.enabled {
            let listener = TcpListener::bind(("127.0.0.1", port))
                .map_err(|error| format!("Unable to start Agent API: {error}"))?;
            let stop = Arc::new(AtomicBool::new(false));
            thread::spawn({
                let app = app.clone();
                let active_jobs = active_jobs.clone();
                let managed_workers = managed_workers.clone();
                let agent_state = agent_state.clone();
                let stop = stop.clone();
                move || {
                    run_agent_server(
                        listener,
                        app,
                        active_jobs,
                        managed_workers,
                        agent_state,
                        stop,
                    )
                }
            });
            control.enabled = true;
            control.stop = Some(stop);
        }
    }

    agent_status_from(agent_state, active_jobs)
}

#[tauri::command]
fn set_agent_server_enabled(
    app: AppHandle,
    agent_state: State<'_, AgentServerState>,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    enabled: bool,
    port: Option<u16>,
) -> Result<AgentServerStatus, String> {
    set_agent_server_enabled_inner(
        app,
        agent_state.inner(),
        active_jobs.inner(),
        managed_workers.inner(),
        enabled,
        port,
    )
}

fn agent_api_autostart_config_path() -> Option<PathBuf> {
    app_install_dir().map(|dir| dir.join("output").join("agent-api-autostart.json"))
}

fn read_agent_api_autostart_config() -> Option<AgentApiAutostartConfig> {
    let path = agent_api_autostart_config_path()?;
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
}

fn agent_api_autostart_requested() -> bool {
    if read_agent_api_autostart_config()
        .and_then(|config| config.enabled)
        .unwrap_or(false)
    {
        return true;
    }
    std::env::var("I3D_AGENT_API")
        .or_else(|_| std::env::var("IH3D_AGENT_API"))
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn agent_api_autostart_port() -> Option<u16> {
    if let Some(port) = read_agent_api_autostart_config()
        .and_then(|config| config.port)
        .filter(|port| *port > 0)
    {
        return Some(port);
    }
    std::env::var("I3D_AGENT_API_PORT")
        .or_else(|_| std::env::var("IH3D_AGENT_API_PORT"))
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port > 0)
}

fn env_truthy(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn agent_api_headless_requested() -> bool {
    env::args().any(|arg| arg == "--headless" || arg == "--serve-agent-api")
        || env_truthy("NEKO_AGENT_HEADLESS")
        || env_truthy("I3D_HEADLESS")
        || env_truthy("IH3D_HEADLESS")
}

/// Explicit launch port: `--agent-api-port`/`--agent-port`, then the port env
/// vars. This beats the autostart config and the port saved in the shared
/// agent-api-registry.json (see `validate_agent_api_port`).
fn agent_api_cli_port() -> Option<u16> {
    agent_api_port_from_args(env::args().skip(1)).or_else(|| {
        env::var("I3D_AGENT_API_PORT")
            .or_else(|_| env::var("IH3D_AGENT_API_PORT"))
            .or_else(|_| env::var("NEKO_AGENT_API_PORT"))
            .ok()
            .and_then(|value| value.trim().parse::<u16>().ok())
            .filter(|port| *port > 0)
    })
}

fn agent_api_port_from_args(mut args: impl Iterator<Item = String>) -> Option<u16> {
    while let Some(arg) = args.next() {
        if arg == "--agent-api-port" || arg == "--agent-port" {
            return args.next().and_then(|value| value.parse::<u16>().ok());
        }
        if let Some(value) = arg
            .strip_prefix("--agent-api-port=")
            .or_else(|| arg.strip_prefix("--agent-port="))
        {
            return value.parse::<u16>().ok();
        }
    }
    None
}

fn hide_main_window(app: &tauri::App) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

fn clamp_window_dimension(value: u32, min: u32, monitor_max: u32) -> u32 {
    let max = monitor_max.max(1);
    value.max(min.min(max)).min(max)
}

fn initial_window_size(monitor_width: u32, monitor_height: u32) -> tauri::PhysicalSize<u32> {
    let width_ratio = if monitor_width < monitor_height {
        PORTRAIT_WINDOW_WIDTH_RATIO
    } else {
        LANDSCAPE_WINDOW_WIDTH_RATIO
    };
    let aspect = DEFAULT_WINDOW_HEIGHT as f64 / DEFAULT_WINDOW_WIDTH as f64;
    let mut width = (monitor_width as f64 * width_ratio).round();
    let mut height = (width * aspect).round();
    let max_height = (monitor_height as f64 * 0.9).round().max(1.0);

    if height > max_height {
        height = max_height;
        width = (height / aspect).round();
    }

    tauri::PhysicalSize::new(width.max(1.0) as u32, height.max(1.0) as u32)
}

fn apply_initial_window_size(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let monitor_size = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten())
        .map(|monitor| *monitor.size());
    let monitor_width = monitor_size
        .as_ref()
        .map(|size| size.width)
        .unwrap_or(DEFAULT_WINDOW_WIDTH);
    let monitor_height = monitor_size
        .as_ref()
        .map(|size| size.height)
        .unwrap_or(DEFAULT_WINDOW_HEIGHT);
    let size = initial_window_size(monitor_width, monitor_height);
    let size = tauri::PhysicalSize::new(
        clamp_window_dimension(size.width, MIN_WINDOW_WIDTH, monitor_width),
        clamp_window_dimension(size.height, MIN_WINDOW_HEIGHT, monitor_height),
    );

    if let Err(error) = window.set_size(tauri::Size::Physical(size)) {
        eprintln!("Failed to initialize window size: {error}");
    }
    let _ = window.center();
}

pub fn run() {
    tauri::Builder::default()
        .manage(ActiveJobState::default())
        .manage(ManagedWorkerState::default())
        .manage(ActiveRemeshState::default())
        .manage(AgentServerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let headless = agent_api_headless_requested();
            if headless {
                hide_main_window(app);
            } else {
                apply_initial_window_size(app);
            }
            if agent_api_autostart_requested() || headless {
                let handle = app.handle().clone();
                let agent_state = app.state::<AgentServerState>().inner().clone();
                let active_jobs = app.state::<ActiveJobState>().inner().clone();
                let managed_workers = app.state::<ManagedWorkerState>().inner().clone();
                let port = agent_api_cli_port().or_else(agent_api_autostart_port);
                if let Err(error) = set_agent_server_enabled_inner(
                    handle,
                    &agent_state,
                    &active_jobs,
                    &managed_workers,
                    true,
                    port,
                ) {
                    eprintln!("Unable to start Agent API: {error}");
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let managed_workers = window.state::<ManagedWorkerState>();
                let _ = managed_workers.stop();
                let active_remesh = window.state::<ActiveRemeshState>();
                let _ = active_remesh.cancel();
            }
        })
        .invoke_handler(tauri::generate_handler![
            cancel_active_job,
            cancel_autoremesher,
            check_runtime,
            get_agent_server_status,
            get_autoremesher_status,
            get_vram_status,
            install_autoremesher,
            list_job_history,
            open_containing_folder,
            open_output_folder,
            open_path,
            open_runtime_folder,
            prepare_runtime,
            preview_image_data_url,
            read_voxel_sidecar,
            reset_python_runtime,
            resolve_inputs,
            run_autoremesher,
            set_agent_server_enabled,
            start_generation_job,
            unload_models
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}

#[cfg(test)]
mod agent_port_tests {
    use super::*;

    fn args(values: &[&str]) -> impl Iterator<Item = String> {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn parses_agent_port_flags() {
        assert_eq!(agent_api_port_from_args(args(&["--headless", "--agent-api-port", "17555"])), Some(17555));
        assert_eq!(agent_api_port_from_args(args(&["--agent-api-port=17556"])), Some(17556));
        assert_eq!(agent_api_port_from_args(args(&["--agent-port", "17557"])), Some(17557));
        assert_eq!(agent_api_port_from_args(args(&["--headless"])), None);
    }

    #[test]
    fn explicit_port_beats_registry() {
        // Whatever the registry holds, an explicit port is used as-is.
        assert_eq!(validate_agent_api_port(Some(17555)), Ok(17555));
        assert!(validate_agent_api_port(Some(0)).is_err());
    }
}
