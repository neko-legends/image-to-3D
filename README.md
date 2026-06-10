# Image To Hunyuan 3D

Image To Hunyuan 3D is a local desktop app for turning images into Hunyuan3D
2.1 meshes and textured GLBs without opening ComfyUI.

The app follows the same runtime idea as `D:\forPublic\DepthMapAIGenerator`:
the repository does not contain or commit a `.venv`. On first preparation, the
desktop app creates its own Python environment in app data, downloads the
Hunyuan source/runtime pieces, and keeps model weights outside this project
folder.

Launch is intentionally fast and quiet. The app checks what is already present
and shows a setup prompt when downloads are needed. It does not download models
or install packages until the user chooses setup.

## NVIDIA GPU Note

This app is designed for NVIDIA GPUs. Hunyuan3D 2.1 is a large CUDA workload;
CPU generation is not a supported target.

The upstream Hunyuan3D 2.1 notes list approximate VRAM needs of 10 GB for
shape generation, 21 GB for texture generation, and 29 GB for shape plus
texture. Smaller cards may work only with lower settings or future low-VRAM
work.

## Runtime Shape

Managed app data contains:

- `python-env` - app-owned Python environment.
- `runtime\Hunyuan3D-2.1` - downloaded Hunyuan source.
- `runtime\ComfyUI-Hunyuan3d-2-1-wheels` - Windows texture wheels used only as
  native wheel sources.
- `models` - Hugging Face model cache.
- `jobs` - per-run job files and temporary meshes.

Generated GLBs default to an `output` folder beside the running app executable.
If the user chooses a different output folder, the app saves and reuses that
path.

When the user chooses setup, first-run preparation downloads/installs:

- Hunyuan source from `Tencent-Hunyuan/Hunyuan3D-2.1`.
- PyTorch CUDA 12.4 wheels for Python 3.10-3.12.
- Hunyuan Python dependencies needed for direct inference.
- Prebuilt Windows texture wheels from `visualbruno/ComfyUI-Hunyuan3d-2-1`.
- Model weights from `tencent/Hunyuan3D-2.1`.
- Texture conditioning from `facebook/dinov2-giant`.
- `RealESRGAN_x4plus.pth` for texture enhancement.

The app installs a curated inference set rather than the full upstream
`requirements.txt`; demo/API/training packages such as Gradio, FastAPI, Blender
Python, Cupy, and Deepspeed are skipped.

ComfyUI is not required to run this app.

## Current Scope

- Queue image files or folders.
- Run Mesh + Texture, Mesh only, or Texture only.
- Texture-only mode accepts an existing `.glb`, `.gltf`, `.obj`, or `.ply`.
- Prepare an app-managed standalone runtime.
- Detect Python, runtime source, model readiness, texture wheels, RealESRGAN,
  NVIDIA GPU presence, and VRAM.
- Preview input images and generated GLBs in the app.
- Open the managed runtime folder and output folder.
- Optional local Agent API for automation while the GUI is running.
- Headless command runner for no-GUI generation.

## Agent And Headless Control

The GUI has an `Agent API` checkbox in the Pipeline panel. When enabled, the app
starts a local-only HTTP API at:

```text
http://127.0.0.1:17333
```

Useful endpoints:

- `GET /health`
- `GET /openapi.json`
- `GET /runtime`
- `GET /vram`
- `POST /prepare`
- `POST /generate`
- `POST /cancel`

Example agent request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:17333/generate -ContentType 'application/json' -Body (@{
  paths = @('D:\images\chair.png')
  options = @{
    mode = 'meshOnly'
    outputDir = 'D:\models'
    meshSteps = 12
    octreeResolution = 256
  }
} | ConvertTo-Json -Depth 6)
```

For no-GUI runs, use the headless runner:

```powershell
py -3 python\ih3d_headless.py --fast-preview D:\images\chair.png
```

The headless runner uses the same app-managed runtime and model cache as the GUI.
It streams JSON worker events to stdout for automation.

## Important Texture Note

The supplied ComfyUI workflow, `Jun-2.1_TextureUpscaleWithAuraSR.json`, was used
as a reference for the intended product flow: mesh, multiview texture, upscale,
bake, inpaint, GLB.

The standalone worker currently uses Tencent's direct Hunyuan3D 2.1 Python API
for shape and PBR texture generation. AuraSR-specific parity is tracked as a
follow-up because the standalone app should not depend on ComfyUI custom node
loading.

## Development

Prerequisites:

- Node.js 20.19+ or 22.12+
- Rust toolchain for Tauri builds
- Python 3.10, 3.11, or 3.12 available to the app
- NVIDIA GPU drivers with `nvidia-smi`

Install JavaScript dependencies:

```powershell
npm install
```

Run the frontend only:

```powershell
npm run dev
```

Run the desktop app:

```powershell
npm run tauri:dev
```

Run without the GUI:

```powershell
npm run headless -- --fast-preview D:\images\chair.png
```

Build release artifacts:

```powershell
npm run tauri:build
```

Publish the current build to the standardized public folders:

```powershell
npm run release:publish
```

Build and publish in one step:

```powershell
npm run release:build
```

Public release files are copied to:

- `release\portable` - standalone executable.
- `release\installer` - installer executable.

## Git Hygiene

Do not commit generated builds, local dependencies, app-managed Python
environments, downloaded source archives, model weights, copied GLBs, job files,
or installer files.

The `.gitignore` covers:

- `node_modules`
- `dist`
- `release`
- `src-tauri/target`
- local Python caches
- generated mesh files
- app runtime/model/output folders
- installer files

## Roadmap

- Run a small real Hunyuan job on the target NVIDIA machine.
- Add low-VRAM presets once they are validated.
- Add richer texture controls where Tencent's direct API exposes them.
- Add AuraSR-style optional view upscaling without requiring ComfyUI.
- Add retained job history with thumbnails and turntable previews.
