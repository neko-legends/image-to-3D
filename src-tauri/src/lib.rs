use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::{
    env,
    fs,
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

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "jfif", "jpf", "jp2", "j2k", "webp", "bmp", "tif", "tiff",
];
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const EMBEDDED_WORKER: &str = include_str!("../../python/hunyuan_worker.py");
const ENV_MARKER: &str = ".ih3d-env-v4-ready";
const SOURCE_MARKER: &str = ".ih3d-source-v1-ready";
const MODEL_MARKER: &str = ".ih3d-models-v1-ready";
const WHEEL_MARKER: &str = ".ih3d-texture-wheels-v1-ready";
const DEFAULT_PYTORCH_PROFILE: &str = "cu124";
const DEFAULT_AGENT_API_PORT: u16 = 17333;
const AGENT_APP_ID: &str = "image-to-hunyuan-3d";
const AGENT_APP_NAME: &str = "ImageToHunyuan3D";
const AGENT_API_BIND_ADDRESS: &str = "127.0.0.1";
const AGENT_API_REGISTRY_FILE: &str = "agent-api-registry.json";

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
    mode: String,
    cuda_device: String,
    runtime_profile: String,
    output_dir: String,
    output_name: String,
    mesh_input_path: String,
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentServerStatus {
    enabled: bool,
    port: u16,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
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

#[derive(Clone, Default)]
struct ActiveJobState {
    inner: Arc<Mutex<Option<ActiveJob>>>,
}

struct ActiveJob {
    id: String,
    pid: u32,
    cancel_requested: bool,
}

impl ActiveJobState {
    fn is_busy(&self) -> Result<bool, String> {
        self.inner
            .lock()
            .map(|job| job.is_some())
            .map_err(|_| "Unable to lock active job state.".to_string())
    }

    fn start(&self, id: String, pid: u32) -> Result<(), String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        if active.is_some() {
            return Err("Another job is already running.".to_string());
        }
        *active = Some(ActiveJob {
            id,
            pid,
            cancel_requested: false,
        });
        Ok(())
    }

    fn request_cancel(&self) -> Result<u32, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.as_mut() else {
            return Err("No active job to cancel.".to_string());
        };
        job.cancel_requested = true;
        Ok(job.pid)
    }

    fn finish(&self, id: &str) -> Result<bool, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.as_ref() else {
            return Ok(false);
        };
        if job.id != id {
            return Ok(false);
        }
        let canceled = job.cancel_requested;
        *active = None;
        Ok(canceled)
    }

    fn finish_current(&self) -> Result<Option<bool>, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.take() else {
            return Ok(None);
        };
        Ok(Some(job.cancel_requested))
    }

    fn finish_if_pid(&self, pid: u32) -> Result<Option<bool>, String> {
        let mut active = self
            .inner
            .lock()
            .map_err(|_| "Unable to lock active job state.".to_string())?;
        let Some(job) = active.as_ref() else {
            return Ok(None);
        };
        if job.pid != pid {
            return Ok(None);
        }
        let canceled = job.cancel_requested;
        *active = None;
        Ok(Some(canceled))
    }

    fn active_job_id(&self) -> Result<Option<String>, String> {
        self.inner
            .lock()
            .map(|job| job.as_ref().map(|job| job.id.clone()))
            .map_err(|_| "Unable to lock active job state.".to_string())
    }
}

#[derive(Clone, Default)]
struct ManagedWorkerState {
    inner: Arc<Mutex<Option<ManagedWorker>>>,
}

#[derive(Clone)]
struct ManagedWorker {
    pid: u32,
    runtime_profile: String,
    cuda_device: String,
    stdin: Arc<Mutex<ChildStdin>>,
}

impl ManagedWorkerState {
    fn get_matching(
        &self,
        runtime_profile: &str,
        cuda_device: &str,
    ) -> Result<Option<ManagedWorker>, String> {
        self.inner
            .lock()
            .map(|worker| {
                worker.as_ref().and_then(|worker| {
                    if worker.runtime_profile == runtime_profile
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
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Unable to resolve app data folder: {error}"))?;
    fs::create_dir_all(&dir)
        .map_err(|error| format!("Unable to create app data folder: {error}"))?;
    Ok(dir)
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

fn env_dir_for_profile(app: &AppHandle, profile: &str) -> Result<PathBuf, String> {
    let app_data = app_data_dir(app)?;
    let profile = normalize_runtime_profile(profile);
    let profiled_env = app_data.join(format!("python-env-{profile}"));
    if profiled_env.exists() {
        return Ok(profiled_env);
    }

    let legacy_env = app_data.join("python-env");
    if legacy_env.exists() && env_runtime_profile(&legacy_env) == profile {
        return Ok(legacy_env);
    }

    Ok(profiled_env)
}

fn runtime_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("runtime"))
}

fn source_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(runtime_dir(app)?.join("Hunyuan3D-2.1"))
}

fn model_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("models"))
}

fn default_output_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe()
        .map_err(|error| format!("Unable to resolve app location: {error}"))?;
    let app_dir = exe
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Unable to resolve app folder.".to_string())?;
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

fn default_voxel_resolution() -> u32 {
    64
}

fn default_hunyuan_options() -> HunyuanOptions {
    HunyuanOptions {
        mode: "meshTexture".to_string(),
        cuda_device: "auto".to_string(),
        runtime_profile: "cu124".to_string(),
        output_dir: String::new(),
        output_name: String::new(),
        mesh_input_path: String::new(),
        recursive: true,
        overwrite: false,
        mesh_detail: "medium".to_string(),
        mesh_steps: 50,
        mesh_guidance: 5.0,
        mesh_seed: 2,
        mesh_seed_mode: "randomize".to_string(),
        box_v: 1.01,
        octree_resolution: 384,
        num_chunks: 8000,
        mc_level: 0.0,
        mc_algo: "mc".to_string(),
        enable_flash_vdm: true,
        compile_models: false,
        force_offload: false,
        remove_floaters: true,
        remove_degenerate_faces: true,
        reduce_faces: true,
        max_faces: 15000,
        smooth_normals: true,
        voxelize: false,
        voxel_resolution: 64,
        texture_view_size: 512,
        texture_steps: 15,
        texture_guidance: 3.0,
        texture_size: 4096,
        texture_seed: 802344256,
        texture_seed_mode: "randomize".to_string(),
        unwrap_mesh: true,
        camera_azimuths: "0, 90, 180, 270, 0, 180".to_string(),
        camera_elevations: "0, 0, 0, 0, 90, -90".to_string(),
        view_weights: "1, 0.1, 0.5, 0.1, 0.05, 0.05".to_string(),
        ortho_scale: 1.1,
    }
}

fn apply_mesh_detail_preset(options: &mut HunyuanOptions) {
    match options.mesh_detail.as_str() {
        "low" => {
            options.mesh_steps = 24;
            options.octree_resolution = 256;
            options.num_chunks = 6000;
        }
        "medium" => {
            options.mesh_steps = 50;
            options.octree_resolution = 384;
            options.num_chunks = 8000;
        }
        "high" => {
            options.mesh_steps = 80;
            options.octree_resolution = 512;
            options.num_chunks = 12000;
        }
        "custom" => {}
        _ => {
            options.mesh_detail = "custom".to_string();
        }
    }
}

fn normalize_options_for_job(options: &mut HunyuanOptions) -> Result<(), String> {
    apply_mesh_detail_preset(options);
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

fn worker_path(app: &AppHandle) -> Result<PathBuf, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));
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
        fs::write(&worker, EMBEDDED_WORKER)
            .map_err(|error| format!("Unable to write embedded worker: {error}"))?;
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
        vec![
            "python3.10",
            "python3.11",
            "python3.12",
            "python3",
            "python",
        ]
    };
    candidates
        .into_iter()
        .map(str::to_string)
        .find(|candidate| python_version(candidate).is_some_and(supported_python_version))
}

fn configure_worker_command(command: &mut Command) {
    hide_command_window(command);
    command
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8:replace")
        .env("PYTHONLEGACYWINDOWSSTDIO", "0")
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .env_remove("TQDM_ASCII");
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn configure_managed_worker_env(
    command: &mut Command,
    app_data: &Path,
    source: &Path,
    models: &Path,
) {
    command
        .env("IH3D_IN_MANAGED_ENV", "1")
        .env("IH3D_SOURCE_DIR", source)
        .env("IH3D_APP_DATA_DIR", app_data)
        .env("HF_HOME", models.join("huggingface"))
        .env("HF_HUB_CACHE", models.join("huggingface").join("hub"))
        .env(
            "TRANSFORMERS_CACHE",
            models.join("huggingface").join("transformers"),
        )
        .env("XFORMERS_FORCE_DISABLE_TRITON", "1");
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
    let env = env_dir_for_profile(app, &requested_profile)?;
    let source = source_dir(app)?;
    let models = model_dir(app)?;
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
    if !source.join(SOURCE_MARKER).is_file() {
        return Err(
            "Hunyuan source setup is not finished yet. Choose Set up now first.".to_string(),
        );
    }
    if !models.join(MODEL_MARKER).is_file() {
        return Err("Model downloads are not finished yet. Choose Set up now first.".to_string());
    }
    if options.mode != "meshOnly" {
        if !env.join(WHEEL_MARKER).is_file() {
            return Err("Texture tools are not set up yet. Choose Set up now first.".to_string());
        }
        if !source
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
fn check_runtime(app: AppHandle, runtime_profile: Option<String>) -> Result<RuntimeStatus, String> {
    let requested_profile = normalize_runtime_profile(
        runtime_profile
            .as_deref()
            .unwrap_or(DEFAULT_PYTORCH_PROFILE),
    );
    let app_data = app_data_dir(&app)?;
    let worker = worker_path(&app)?;
    let env = env_dir_for_profile(&app, &requested_profile)?;
    let env_python = env_python_path(&env);
    let source = source_dir(&app)?;
    let models = model_dir(&app)?;
    let outputs = default_output_dir()?;
    let jobs = jobs_dir(&app)?;
    let tencent_ready = model_cache_has_snapshot(&models, "tencent/Hunyuan3D-2.1");

    Ok(RuntimeStatus {
        python_found: find_python().is_some(),
        python_command: find_python(),
        app_data_dir: app_data.display().to_string(),
        worker_path: worker.display().to_string(),
        env_dir: env.display().to_string(),
        env_python: env_python.display().to_string(),
        env_ready: env_python.is_file() && env.join(ENV_MARKER).is_file(),
        source_dir: source.display().to_string(),
        source_ready: source.join(SOURCE_MARKER).is_file(),
        model_dir: models.display().to_string(),
        models_ready: models.join(MODEL_MARKER).is_file(),
        shape_model_ready: tencent_ready,
        paint_model_ready: tencent_ready
            && model_cache_has_snapshot(&models, "facebook/dinov2-giant"),
        texture_wheels_ready: env.join(WHEEL_MARKER).is_file(),
        realesrgan_ready: source
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
    let cuda_device = options.cuda_device.clone();
    if let Some(worker) = managed_workers.get_matching(&runtime_profile, &cuda_device)? {
        return Ok(worker);
    }
    managed_workers.stop()?;

    let env = env_dir_for_profile(app, &runtime_profile)?;
    let worker_path = worker_path(app)?;
    let app_data = app_data_dir(app)?;
    let mut command = Command::new(env_python_path(&env));
    configure_worker_command(&mut command);
    configure_managed_worker_env(&mut command, &app_data, &source_dir(app)?, &model_dir(app)?);
    command
        .arg(worker_path)
        .arg("--daemon")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start managed Hunyuan worker: {error}"))?;
    let pid = child.id();
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Unable to open managed worker stdin.".to_string())?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let worker = ManagedWorker {
        pid,
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
                    let _ = app_for_stdout.emit("hunyuan-worker-event", value);
                    if completes_job {
                        let _ = active_jobs_for_stdout.finish_current();
                    }
                } else {
                    let _ = app_for_stdout.emit(
                        "hunyuan-worker-event",
                        serde_json::json!({ "type": "setup", "message": line }),
                    );
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
                let _ = app_for_wait.emit(
                    "hunyuan-worker-event",
                    serde_json::json!({ "type": "canceled", "message": "Job canceled." }),
                );
            } else if !matches!(status, Ok(status) if status.success()) {
                let _ = app_for_wait.emit(
                    "hunyuan-worker-event",
                    serde_json::json!({ "type": "error", "message": "Managed Hunyuan worker exited with an error." }),
                );
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
    if action == "generate"
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
        let env = env_dir_for_profile(&app, &requested_profile)?;
        Command::new(env_python_path(&env))
    } else {
        let python = find_python().ok_or_else(|| {
            "Python 3.10, 3.11, or 3.12 was not found. Install Python and try again.".to_string()
        })?;
        command_from_python_spec(&python)
    };
    configure_worker_command(&mut command);
    if action == "generate" {
        configure_managed_worker_env(
            &mut command,
            &app_data,
            &source_dir(&app)?,
            &model_dir(&app)?,
        );
    }
    command
        .arg(worker)
        .arg("--job")
        .arg(job_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start Hunyuan worker: {error}"))?;
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
    let active_jobs_for_wait = active_jobs.clone();
    let job_id_for_wait = job_id.clone();

    thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = app_for_stdout.emit("hunyuan-worker-event", value);
                } else {
                    let _ = app_for_stdout.emit(
                        "hunyuan-worker-event",
                        serde_json::json!({ "type": "setup", "message": line }),
                    );
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
            let _ = app_for_wait.emit(
                "hunyuan-worker-event",
                serde_json::json!({
                    "type": "canceled",
                    "message": "Job canceled."
                }),
            );
        } else if !matches!(status, Ok(status) if status.success()) {
            let _ = app_for_wait.emit(
                "hunyuan-worker-event",
                serde_json::json!({
                    "type": "error",
                    "message": "Hunyuan worker exited with an error."
                }),
            );
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
fn start_hunyuan_job(
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
fn cancel_active_job(active_jobs: State<'_, ActiveJobState>) -> Result<(), String> {
    let pid = active_jobs.request_cancel()?;
    kill_process_tree(pid)
}

#[tauri::command]
fn reset_python_runtime(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    runtime_profile: Option<String>,
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
    check_runtime(app, Some(requested_profile))
}

#[tauri::command]
fn unload_models(
    app: AppHandle,
    active_jobs: State<'_, ActiveJobState>,
    managed_workers: State<'_, ManagedWorkerState>,
    runtime_profile: Option<String>,
) -> Result<RuntimeStatus, String> {
    if active_jobs.is_busy()? {
        return Err("Wait for the current job to finish before unloading models.".to_string());
    }
    managed_workers.stop()?;
    check_runtime(app, runtime_profile)
}

struct HttpRequest {
    method: String,
    path: String,
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
    let active_job_id = active_jobs.active_job_id()?;
    let status = AgentServerStatus {
        enabled,
        port,
        url: agent_api_url(port),
        openapi_url: format!("{}/openapi.json", agent_api_url(port)),
        busy: active_job_id.is_some(),
        active_job_id,
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
    let path = raw_path.split('?').next().unwrap_or("/").to_string();
    let body_start = header_end + 4;
    let body = if body_start <= data.len() {
        data[body_start..].to_vec()
    } else {
        Vec::new()
    };
    Ok(HttpRequest { method, path, body })
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
            "title": "Image To Hunyuan 3D Agent API",
            "version": env!("CARGO_PKG_VERSION")
        },
        "servers": [{ "url": agent_api_url(port) }],
        "paths": {
            "/health": { "get": { "summary": "Check API status" } },
            "/status": { "get": { "summary": "Check active job status" } },
            "/runtime": { "get": { "summary": "Check local runtime readiness" } },
            "/vram": { "get": { "summary": "Check NVIDIA VRAM" } },
            "/prepare": { "post": { "summary": "Prepare the standalone runtime" } },
            "/reset-runtime": { "post": { "summary": "Reset the managed Python runtime" } },
            "/unload-models": { "post": { "summary": "Stop the warm worker and release loaded models" } },
            "/generate": { "post": { "summary": "Start a Hunyuan generation job" } },
            "/cancel": { "post": { "summary": "Cancel the active job" } }
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
    let mesh_numbers_overridden = ["meshSteps", "octreeResolution", "numChunks"]
        .iter()
        .any(|key| overrides.contains_key(*key));
    if mesh_numbers_overridden && !overrides.contains_key("meshDetail") {
        base.insert(
            "meshDetail".to_string(),
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
            "service": "Image To Hunyuan 3D",
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
        ("GET", "/runtime") => serde_json::to_value(check_runtime(app.clone(), None)?)
            .map_err(|error| error.to_string()),
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
            let env = env_dir_for_profile(app, &runtime_profile)?;
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
        ("POST", "/cancel") => {
            let pid = active_jobs.request_cancel()?;
            kill_process_tree(pid)?;
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

fn agent_api_autostart_requested() -> bool {
    std::env::var("IH3D_AGENT_API")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn agent_api_autostart_port() -> Option<u16> {
    std::env::var("IH3D_AGENT_API_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port > 0)
}

pub fn run() {
    tauri::Builder::default()
        .manage(ActiveJobState::default())
        .manage(ManagedWorkerState::default())
        .manage(AgentServerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            if agent_api_autostart_requested() {
                let handle = app.handle().clone();
                let agent_state = app.state::<AgentServerState>().inner().clone();
                let active_jobs = app.state::<ActiveJobState>().inner().clone();
                let managed_workers = app.state::<ManagedWorkerState>().inner().clone();
                if let Err(error) = set_agent_server_enabled_inner(
                    handle,
                    &agent_state,
                    &active_jobs,
                    &managed_workers,
                    true,
                    agent_api_autostart_port(),
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
            }
        })
        .invoke_handler(tauri::generate_handler![
            cancel_active_job,
            check_runtime,
            get_agent_server_status,
            get_vram_status,
            open_containing_folder,
            open_output_folder,
            open_path,
            open_runtime_folder,
            prepare_runtime,
            preview_image_data_url,
            read_voxel_sidecar,
            reset_python_runtime,
            resolve_inputs,
            set_agent_server_enabled,
            start_hunyuan_job,
            unload_models
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
