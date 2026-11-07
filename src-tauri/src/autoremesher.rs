use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
};
use uuid::Uuid;

#[cfg(unix)]
use std::os::unix::{fs::PermissionsExt, process::CommandExt as _};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

const VERSION: &str = "1.1.0";
#[cfg_attr(target_os = "linux", allow(dead_code))]
const WINDOWS_ARCHIVE_URL: &str = "https://github.com/huxingyi/autoremesher/releases/download/1.1.0/autoremesher-1.1.0-win32-x86_64.zip";
#[cfg_attr(target_os = "linux", allow(dead_code))]
const WINDOWS_ARCHIVE_SHA256: &str =
    "49a34610c3de1b4d784461c8b0380f8dade3837e6d39fc4154640bdc93a13202";
#[cfg(target_os = "linux")]
const LINUX_APPIMAGE_URL: &str =
    "https://github.com/huxingyi/autoremesher/releases/download/1.1.0/autoremesher-1.1.0.AppImage";
#[cfg(target_os = "linux")]
const LINUX_APPIMAGE_SHA256: &str =
    "e92426bfbdb7a2e9f33a93e133af8ed325c4cd8b07f7746ff149f5a3d820ed6b";
#[cfg(target_os = "linux")]
const LINUX_APPIMAGE_NAME: &str = "autoremesher-1.1.0.AppImage";
/// Platforms with an upstream AutoRemesher 1.1.0 build the app can manage:
/// the Windows zip and the x86_64 Linux AppImage.
const MANAGED_DOWNLOAD_SUPPORTED: bool = cfg!(target_os = "windows")
    || cfg!(all(target_os = "linux", target_arch = "x86_64"));
const UNSUPPORTED_MESSAGE: &str =
    "The managed AutoRemesher download is available in the Windows and x86_64 Linux builds.";
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemesherStatus {
    pub supported: bool,
    pub installed: bool,
    pub version: String,
    pub executable_path: Option<String>,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemeshOptions {
    pub input_path: String,
    pub target_quads: u32,
    pub edge_scaling: f64,
    pub sharp_edge: f64,
    pub smooth_normal: f64,
    pub adaptivity: f64,
    pub anisotropy: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemeshResult {
    pub output_path: String,
    pub report_path: String,
    pub message: String,
}

#[derive(Clone, Default)]
pub struct ActiveRemeshState {
    pid: Arc<Mutex<Option<u32>>>,
}

impl ActiveRemeshState {
    fn start(&self, pid: u32) -> Result<(), String> {
        let mut current = self
            .pid
            .lock()
            .map_err(|_| "Unable to lock AutoRemesher state.".to_string())?;
        if current.is_some() {
            return Err("Another remesh is already running.".to_string());
        }
        *current = Some(pid);
        Ok(())
    }

    fn finish(&self, pid: u32) {
        if let Ok(mut current) = self.pid.lock() {
            if *current == Some(pid) {
                *current = None;
            }
        }
    }

    pub fn cancel(&self) -> Result<(), String> {
        let pid = self
            .pid
            .lock()
            .map_err(|_| "Unable to lock AutoRemesher state.".to_string())?
            .take()
            .ok_or_else(|| "No remesh is currently running.".to_string())?;
        kill_process_tree(pid)
    }
}

fn tool_dir(app_data: &Path) -> PathBuf {
    app_data.join("tools").join("autoremesher").join(VERSION)
}

fn find_file_named(root: &Path, file_name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(file_name))
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file_named(&path, file_name) {
                return Some(found);
            }
        }
    }
    None
}

fn executable_path(app_data: &Path) -> Option<PathBuf> {
    let root = tool_dir(app_data);
    if cfg!(target_os = "windows") {
        find_file_named(&root, "autoremesher.exe")
    } else {
        #[cfg(target_os = "linux")]
        {
            let appimage = root.join(LINUX_APPIMAGE_NAME);
            if appimage.is_file() {
                return Some(appimage);
            }
        }
        find_file_named(&root, "autoremesher")
    }
}

pub fn status(app_data: &Path) -> RemesherStatus {
    let supported = MANAGED_DOWNLOAD_SUPPORTED;
    let executable = supported.then(|| executable_path(app_data)).flatten();
    RemesherStatus {
        supported,
        installed: executable.is_some(),
        version: VERSION.to_string(),
        executable_path: executable.as_ref().map(|path| path.display().to_string()),
        message: if !supported {
            UNSUPPORTED_MESSAGE.to_string()
        } else if executable.is_some() {
            format!("AutoRemesher {VERSION} is ready.")
        } else {
            format!("AutoRemesher {VERSION} is not installed yet.")
        },
    }
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let mut file = File::open(path)
        .map_err(|error| format!("Unable to verify AutoRemesher download: {error}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("Unable to verify AutoRemesher download: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "AutoRemesher download checksum did not match (expected {expected}, got {actual})."
        ))
    }
}

#[cfg_attr(target_os = "linux", allow(dead_code))]
fn extract_zip(archive_path: &Path, destination: &Path) -> Result<(), String> {
    let archive_file = File::open(archive_path)
        .map_err(|error| format!("Unable to open AutoRemesher archive: {error}"))?;
    let mut archive = zip::ZipArchive::new(archive_file)
        .map_err(|error| format!("Unable to read AutoRemesher archive: {error}"))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("Unable to read AutoRemesher archive entry: {error}"))?;
        let relative = entry
            .enclosed_name()
            .ok_or_else(|| "AutoRemesher archive contains an unsafe path.".to_string())?;
        let target = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&target)
                .map_err(|error| format!("Unable to create AutoRemesher folder: {error}"))?;
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Unable to create AutoRemesher folder: {error}"))?;
        }
        let mut output = File::create(&target)
            .map_err(|error| format!("Unable to extract AutoRemesher: {error}"))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|error| format!("Unable to extract AutoRemesher: {error}"))?;
    }
    Ok(())
}

pub fn install(app_data: &Path) -> Result<RemesherStatus, String> {
    if !MANAGED_DOWNLOAD_SUPPORTED {
        return Err(UNSUPPORTED_MESSAGE.to_string());
    }
    if executable_path(app_data).is_some() {
        return Ok(status(app_data));
    }
    #[cfg(target_os = "linux")]
    {
        install_linux_appimage(app_data)
    }
    #[cfg(not(target_os = "linux"))]
    {
        install_windows_archive(app_data)
    }
}

fn download(url: &str, destination: &Path) -> Result<(), String> {
    let response = ureq::get(url)
        .set("User-Agent", "NekoLegends-ImageTo3D")
        .call()
        .map_err(|error| format!("Unable to download AutoRemesher: {error}"))?;
    let mut reader = response.into_reader();
    let mut output = File::create(destination)
        .map_err(|error| format!("Unable to create AutoRemesher download: {error}"))?;
    std::io::copy(&mut reader, &mut output)
        .map_err(|error| format!("Unable to save AutoRemesher download: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("Unable to save AutoRemesher download: {error}"))
}

/// Linux: the upstream AppImage carries the same headless CLI as the Windows
/// build. It is stored as-is and run with APPIMAGE_EXTRACT_AND_RUN so it does
/// not need FUSE 2.
#[cfg(target_os = "linux")]
fn install_linux_appimage(app_data: &Path) -> Result<RemesherStatus, String> {
    let root = tool_dir(app_data);
    fs::create_dir_all(&root)
        .map_err(|error| format!("Unable to create AutoRemesher folder: {error}"))?;
    let partial = root.join("autoremesher.download");
    download(LINUX_APPIMAGE_URL, &partial)?;
    if let Err(error) = verify_sha256(&partial, LINUX_APPIMAGE_SHA256) {
        let _ = fs::remove_file(&partial);
        return Err(error);
    }
    fs::set_permissions(&partial, fs::Permissions::from_mode(0o755))
        .map_err(|error| format!("Unable to mark AutoRemesher executable: {error}"))?;
    fs::rename(&partial, root.join(LINUX_APPIMAGE_NAME))
        .map_err(|error| format!("Unable to install AutoRemesher: {error}"))?;
    Ok(status(app_data))
}

#[cfg(not(target_os = "linux"))]
fn install_windows_archive(app_data: &Path) -> Result<RemesherStatus, String> {

    let root = tool_dir(app_data);
    fs::create_dir_all(&root)
        .map_err(|error| format!("Unable to create AutoRemesher folder: {error}"))?;
    let download = root.join("autoremesher.download");
    self::download(WINDOWS_ARCHIVE_URL, &download)?;
    verify_sha256(&download, WINDOWS_ARCHIVE_SHA256)?;
    extract_zip(&download, &root)?;

    // The 1.1.0 Windows release archive contains a second zip with the actual
    // executable and Qt runtime, so unpack that layer when present.
    if executable_path(app_data).is_none() {
        let nested =
            find_file_named(&root, "autoremesher-1.1.0-win32-x86_64.zip").ok_or_else(|| {
                "AutoRemesher executable was not present in the release archive.".to_string()
            })?;
        extract_zip(&nested, &root)?;
        let _ = fs::remove_file(nested);
    }
    let _ = fs::remove_file(download);
    if executable_path(app_data).is_none() {
        return Err("AutoRemesher installation finished without an executable.".to_string());
    }
    Ok(status(app_data))
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn row_major_matrix(node: &gltf::Node<'_>) -> [[f32; 4]; 4] {
    let column_major = node.transform().matrix();
    let mut matrix = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            matrix[row][column] = column_major[column][row];
        }
    }
    matrix
}

fn multiply_matrix(left: [[f32; 4]; 4], right: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            result[row][column] = (0..4)
                .map(|index| left[row][index] * right[index][column])
                .sum();
        }
    }
    result
}

fn transform_position(matrix: [[f32; 4]; 4], position: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = position;
    [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]
}

fn triangle_indices(mode: gltf::mesh::Mode, indices: &[u32]) -> Result<Vec<[u32; 3]>, String> {
    match mode {
        gltf::mesh::Mode::Triangles => Ok(indices
            .chunks_exact(3)
            .map(|face| [face[0], face[1], face[2]])
            .collect()),
        gltf::mesh::Mode::TriangleStrip => Ok(indices
            .windows(3)
            .enumerate()
            .map(|(index, face)| {
                if index % 2 == 0 {
                    [face[0], face[1], face[2]]
                } else {
                    [face[1], face[0], face[2]]
                }
            })
            .collect()),
        gltf::mesh::Mode::TriangleFan => Ok(indices
            .get(1..)
            .unwrap_or_default()
            .windows(2)
            .map(|face| [indices[0], face[0], face[1]])
            .collect()),
        _ => Err("AutoRemesher conversion only supports triangle mesh primitives.".to_string()),
    }
}

fn write_node_geometry(
    node: gltf::Node<'_>,
    parent_transform: [[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    output: &mut BufWriter<File>,
    vertex_offset: &mut u32,
    mesh_index: &mut u32,
) -> Result<(), String> {
    let transform = multiply_matrix(parent_transform, row_major_matrix(&node));
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions = reader
                .read_positions()
                .ok_or_else(|| "A glTF mesh primitive has no positions.".to_string())?
                .map(|position| transform_position(transform, position))
                .collect::<Vec<_>>();
            let indices = reader
                .read_indices()
                .map(|values| values.into_u32().collect::<Vec<_>>())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            let faces = triangle_indices(primitive.mode(), &indices)?;
            writeln!(output, "o mesh_{mesh_index}")
                .map_err(|error| format!("Unable to write temporary OBJ: {error}"))?;
            for [x, y, z] in &positions {
                writeln!(output, "v {x} {y} {z}")
                    .map_err(|error| format!("Unable to write temporary OBJ: {error}"))?;
            }
            for [a, b, c] in faces {
                writeln!(
                    output,
                    "f {} {} {}",
                    vertex_offset.saturating_add(a).saturating_add(1),
                    vertex_offset.saturating_add(b).saturating_add(1),
                    vertex_offset.saturating_add(c).saturating_add(1)
                )
                .map_err(|error| format!("Unable to write temporary OBJ: {error}"))?;
            }
            *vertex_offset = vertex_offset.saturating_add(positions.len() as u32);
            *mesh_index = mesh_index.saturating_add(1);
        }
    }
    for child in node.children() {
        write_node_geometry(child, transform, buffers, output, vertex_offset, mesh_index)?;
    }
    Ok(())
}

fn convert_gltf_to_obj(input: &Path, output_path: &Path) -> Result<(), String> {
    let gltf = gltf::Gltf::open(input)
        .map_err(|error| format!("Unable to open model for remeshing: {error}"))?;
    let buffers = gltf::import_buffers(&gltf.document, input.parent(), gltf.blob)
        .map_err(|error| format!("Unable to load model geometry for remeshing: {error}"))?;
    let scene = gltf
        .document
        .default_scene()
        .or_else(|| gltf.document.scenes().next())
        .ok_or_else(|| "The model does not contain a glTF scene.".to_string())?;
    let file = File::create(output_path)
        .map_err(|error| format!("Unable to create temporary OBJ: {error}"))?;
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "# Temporary geometry exported by Image To 3D for AutoRemesher"
    )
    .map_err(|error| format!("Unable to write temporary OBJ: {error}"))?;
    let mut vertex_offset = 0_u32;
    let mut mesh_index = 0_u32;
    for node in scene.nodes() {
        write_node_geometry(
            node,
            identity_matrix(),
            &buffers,
            &mut output,
            &mut vertex_offset,
            &mut mesh_index,
        )?;
    }
    output
        .flush()
        .map_err(|error| format!("Unable to finish temporary OBJ: {error}"))?;
    if vertex_offset == 0 {
        return Err("The model does not contain remeshable triangle geometry.".to_string());
    }
    Ok(())
}

fn unique_output_path(input: &Path) -> Result<PathBuf, String> {
    let parent = input
        .parent()
        .ok_or_else(|| "The input model has no containing folder.".to_string())?;
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "The input model name is invalid.".to_string())?;
    let initial = parent.join(format!("{stem}_remeshed.obj"));
    if !initial.exists() && !initial.with_extension("report.txt").exists() {
        return Ok(initial);
    }
    for suffix in 2..10_000 {
        let candidate = parent.join(format!("{stem}_remeshed_{suffix}.obj"));
        if !candidate.exists() && !candidate.with_extension("report.txt").exists() {
            return Ok(candidate);
        }
    }
    Err("Unable to choose a unique remeshed output name.".to_string())
}

fn validate_options(options: &RemeshOptions) -> Result<(), String> {
    if !(100..=10_000_000).contains(&options.target_quads) {
        return Err("Target quads must be between 100 and 10,000,000.".to_string());
    }
    let ranges = [
        ("Edge scaling", options.edge_scaling, 1.0, 4.0),
        ("Sharp edge", options.sharp_edge, 30.0, 180.0),
        ("Smooth normal", options.smooth_normal, 0.0, 180.0),
        ("Adaptivity", options.adaptivity, 0.0, 1.0),
        ("Anisotropy", options.anisotropy, 0.0, 1.0),
    ];
    for (name, value, minimum, maximum) in ranges {
        if !value.is_finite() || value < minimum || value > maximum {
            return Err(format!("{name} must be between {minimum} and {maximum}."));
        }
    }
    Ok(())
}

pub fn run(
    app_data: &Path,
    state: &ActiveRemeshState,
    options: RemeshOptions,
) -> Result<RemeshResult, String> {
    validate_options(&options)?;
    let executable = executable_path(app_data)
        .ok_or_else(|| "Install AutoRemesher before running it.".to_string())?;
    let input = PathBuf::from(options.input_path.trim());
    if !input.is_file() {
        return Err(format!("Input model was not found: {}", input.display()));
    }
    let extension = input
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "obj" | "glb" | "gltf") {
        return Err("AutoRemesher accepts OBJ, GLB, or GLTF models in Image To 3D.".to_string());
    }

    let job_dir = app_data
        .join("tools")
        .join("autoremesher")
        .join("jobs")
        .join(Uuid::new_v4().to_string());
    fs::create_dir_all(&job_dir)
        .map_err(|error| format!("Unable to create remesh job folder: {error}"))?;
    let remesh_input = if extension == "obj" {
        input.clone()
    } else {
        let converted = job_dir.join("input.obj");
        if let Err(error) = convert_gltf_to_obj(&input, &converted) {
            let _ = fs::remove_file(&converted);
            let _ = fs::remove_dir(&job_dir);
            return Err(error);
        }
        converted
    };
    let output_path = unique_output_path(&input)?;
    let report_path = output_path.with_extension("report.txt");

    let mut command = Command::new(&executable);
    command
        .current_dir(executable.parent().unwrap_or_else(|| Path::new(".")))
        .arg("--input")
        .arg(&remesh_input)
        .arg("--output")
        .arg(&output_path)
        .arg("--report")
        .arg(&report_path)
        .arg("--target-quads")
        .arg(options.target_quads.to_string())
        .arg("--edge-scaling")
        .arg(options.edge_scaling.to_string())
        .arg("--sharp-edge")
        .arg(options.sharp_edge.to_string())
        .arg("--smooth-normal")
        .arg(options.smooth_normal.to_string())
        .arg("--adaptivity")
        .arg(options.adaptivity.to_string())
        .arg("--anisotropy")
        .arg(options.anisotropy.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    #[cfg(target_os = "linux")]
    {
        // No FUSE 2 needed; the headless run never shows a window, and the
        // bundled Qt 5 has no Wayland plugin.
        command
            .env("APPIMAGE_EXTRACT_AND_RUN", "1")
            .env("QT_QPA_PLATFORM", "offscreen");
    }
    crate::child_env::strip_appimage_env(&mut command);
    // Own process group so cancel (`kill -TERM -<pid>`) reaches the AppImage
    // runtime's child too.
    #[cfg(unix)]
    command.process_group(0);

    let child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            if remesh_input != input {
                let _ = fs::remove_file(&remesh_input);
            }
            let _ = fs::remove_dir(&job_dir);
            return Err(format!("Unable to start AutoRemesher: {error}"));
        }
    };
    let pid = child.id();
    if let Err(error) = state.start(pid) {
        let _ = kill_process_tree(pid);
        if remesh_input != input {
            let _ = fs::remove_file(&remesh_input);
        }
        let _ = fs::remove_dir(&job_dir);
        return Err(error);
    }
    let result = child.wait_with_output();
    state.finish(pid);
    if remesh_input != input {
        let _ = fs::remove_file(&remesh_input);
    }
    let _ = fs::remove_dir(&job_dir);
    let result = result.map_err(|error| format!("Unable to wait for AutoRemesher: {error}"))?;
    if !result.status.success() || !output_path.is_file() {
        let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(if detail.is_empty() {
            "AutoRemesher did not produce an output model.".to_string()
        } else {
            format!("AutoRemesher failed: {detail}")
        });
    }
    let stdout = String::from_utf8_lossy(&result.stdout);
    let summary = stdout
        .lines()
        .find(|line| line.starts_with("Quads:"))
        .unwrap_or("Remeshing complete")
        .to_string();
    Ok(RemeshResult {
        output_path: output_path.display().to_string(),
        report_path: report_path.display().to_string(),
        message: summary,
    })
}

#[cfg(target_os = "windows")]
fn kill_process_tree(pid: u32) -> Result<(), String> {
    let output = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|error| format!("Unable to cancel AutoRemesher: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(unix)]
fn kill_process_tree(pid: u32) -> Result<(), String> {
    let status = Command::new("kill")
        .args(["-TERM", &format!("-{pid}")])
        .status()
        .map_err(|error| format!("Unable to cancel AutoRemesher: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("Unable to cancel AutoRemesher process group.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose, Engine as _};

    #[test]
    fn transforms_positions_with_row_major_matrix() {
        let transform = [
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 3.0, 0.0, 6.0],
            [0.0, 0.0, 4.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            transform_position(transform, [1.0, 2.0, 3.0]),
            [7.0, 12.0, 19.0]
        );
    }

    #[test]
    fn triangle_strip_alternates_winding() {
        assert_eq!(
            triangle_indices(gltf::mesh::Mode::TriangleStrip, &[0, 1, 2, 3]).unwrap(),
            vec![[0, 1, 2], [2, 1, 3]]
        );
    }

    #[test]
    fn converts_gltf_geometry_and_node_transform_to_obj() {
        let temp = std::env::temp_dir().join(format!("image-to-3d-remesh-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp).unwrap();
        let input = temp.join("triangle.gltf");
        let output = temp.join("triangle.obj");
        let mut buffer = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0_u16, 1, 2] {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        buffer.extend_from_slice(&[0, 0]);
        let encoded = general_purpose::STANDARD.encode(buffer);
        let document = serde_json::json!({
            "asset": { "version": "2.0" },
            "buffers": [{ "uri": format!("data:application/octet-stream;base64,{encoded}"), "byteLength": 44 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0] },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] }],
            "nodes": [{ "mesh": 0, "translation": [2, 3, 4] }],
            "scenes": [{ "nodes": [0] }],
            "scene": 0
        });
        fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
        convert_gltf_to_obj(&input, &output).unwrap();
        let obj = fs::read_to_string(&output).unwrap();
        assert!(obj.contains("v 2 3 4"));
        assert!(obj.contains("v 3 3 4"));
        assert!(obj.contains("f 1 2 3"));
        fs::remove_dir_all(temp).unwrap();
    }
}
