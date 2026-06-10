# Image To Hunyuan 3D

Image To Hunyuan 3D turns images into Hunyuan3D 2.1 meshes and textured GLBs
without opening ComfyUI.

The app is designed for NVIDIA GPUs. Hunyuan3D 2.1 is a large CUDA workload;
CPU generation is not a supported target.

## For Humans

Use the app like a normal desktop tool.

1. Run the portable app:

   ```text
   release\portable\image-to-hunyuan-3d.exe
   ```

   Or run the installer:

   ```text
   release\installer\Image To Hunyuan 3D_26.6.20_x64-setup.exe
   ```

2. On first launch, the app checks what is already installed. If setup is
   needed, it asks before downloading models or installing packages.

3. Add an image or a folder of images. You can also drop a `.glb` or `.gltf`
   file into the window to preview it without starting a generation job.

4. Pick the CUDA GPU and PyTorch runtime in the Runtime panel if you have more
   than one NVIDIA GPU.

   - `CUDA 12.4 / stable` is the default and is known to work on cards like the
     RTX 3090.
   - `CUDA 12.8 / RTX 50 Blackwell` is intended for RTX 50-series cards such as
     the RTX 5090.
   - `Auto choose compatible GPU` uses the best compatible card for the
     installed runtime.

5. Choose a mode:

   - `Mesh + Texture` creates a textured GLB.
   - `Mesh only` creates a faster mesh preview GLB.
   - `Texture only` textures an existing `.glb`, `.gltf`, `.obj`, or `.ply`.

6. Pick mesh detail:

   - `Low` is faster for previews.
   - `Medium` is the default balance.
   - `High` spends more time on a denser mesh.

7. Generated GLBs go into an `output` folder beside the running app by default.
   If you choose a different output folder, the app remembers it.

ComfyUI is not required. The app manages its own Python runtime, Hunyuan source,
model cache, and texture tools in app data.

If you want an AI agent to control the app while the GUI is open, enable the
`Agent API` checkbox in the Pipeline panel. Most people can leave it off.

## For AI Agents

This repo is a Tauri 2 + React desktop app with a Python worker. Do not create
or commit a project `.venv`; the app creates its managed runtime under app data.

Managed app data contains:

- `python-env*` - app-owned Python environments for selected PyTorch runtime
  profiles.
- `runtime\Hunyuan3D-2.1` - downloaded Hunyuan source.
- `runtime\ComfyUI-Hunyuan3d-2-1-wheels` - Windows texture wheel source.
- `models` - Hugging Face model cache.
- `jobs` - per-run job files and temporary meshes.

First-run preparation downloads/installs:

- Hunyuan source from `Tencent-Hunyuan/Hunyuan3D-2.1`.
- PyTorch CUDA wheels for Python 3.10-3.12.
- Curated Hunyuan inference dependencies.
- Prebuilt Windows texture wheels from `visualbruno/ComfyUI-Hunyuan3d-2-1`.
- Model weights from `tencent/Hunyuan3D-2.1`.
- Texture conditioning from `facebook/dinov2-giant`.
- `RealESRGAN_x4plus.pth` for texture enhancement.

The app intentionally skips full upstream demo/API/training dependencies such as
Gradio, FastAPI, Blender Python, Cupy, and Deepspeed.

Runtime profiles:

- `cu124` installs `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`,
  `torchaudio==2.5.1+cu124`. This is the known-good default for RTX 30/40-series
  cards and falls back to compatible GPUs when a newer card is not supported.
- `cu128` installs `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128`,
  `torchaudio==2.7.1+cu128`. This is intended for RTX 50 / Blackwell GPUs.

Choosing a different CUDA GPU does not delete files or reinstall packages; the
worker only points CUDA at the selected card. Changing the PyTorch runtime
profile and choosing `Prepare` installs or reuses the managed Python environment
for that profile. Runtime profiles can live side by side, and `Reset Env`
removes only the selected Python runtime environment; models and outputs stay
untouched.

### Build And Release

Prerequisites:

- Node.js 20.19+ or 22.12+
- Rust toolchain for Tauri builds
- Python 3.10, 3.11, or 3.12 available to the app
- NVIDIA GPU drivers with `nvidia-smi`

Install dependencies:

```powershell
npm install
```

Run the desktop app in development:

```powershell
npm run tauri:dev
```

Build and publish the standardized local release folders:

```powershell
npm run release:build
```

Release outputs:

- `release\portable\image-to-hunyuan-3d.exe`
- `release\installer\Image To Hunyuan 3D_26.6.20_x64-setup.exe`

### Headless Runner

Use the headless runner when the GUI should not be part of the automation flow:

```powershell
py -3 python\ih3d_headless.py --fast-preview D:\images\chair.png
```

Equivalent npm wrapper:

```powershell
npm run headless -- --fast-preview D:\images\chair.png
```

The runner uses the same managed runtime and model cache as the GUI. It streams
JSON worker events to stdout.

Useful flags:

- `--fast-preview` uses faster mesh-only settings.
- `--cuda-device 1` selects a specific NVIDIA GPU index.
- `--runtime-profile cu128` installs/uses the CUDA 12.8 RTX 50-series profile.
- `--mode meshTexture` creates mesh and texture.
- `--mode meshOnly` creates a faster mesh preview.
- `--mode textureOnly --mesh-input-path D:\mesh\source.glb` textures an existing
  mesh.
- `--output-dir D:\models` chooses the output folder.
- `--output-name chair-test` chooses the output basename.
- `--overwrite` replaces an existing output with the same name.

### Local Agent API

The GUI exposes a local-only Agent API when the user enables `Agent API` in the
Pipeline panel. The same API can be started on app launch by setting:

```powershell
$env:IH3D_AGENT_API = '1'
.\release\portable\image-to-hunyuan-3d.exe
```

The default port is `17333`. Change it in the Pipeline panel, or set
`IH3D_AGENT_API_PORT` when using environment auto-start.

Default base URL:

```text
http://127.0.0.1:17333
```

Endpoints:

- `GET /health`
- `GET /openapi.json`
- `GET /status`
- `GET /runtime`
- `GET /vram`
- `POST /prepare`
- `POST /reset-runtime`
- `POST /generate`
- `POST /cancel`

The `/generate` endpoint accepts the same important settings the UI exposes.
Agents may send partial options; missing values are filled from app defaults.

Mode values:

- `meshTexture` - Mesh + Texture.
- `meshOnly` - Mesh only.
- `textureOnly` - Texture only.

Common options:

- `cudaDevice`
- `runtimeProfile`
- `outputDir`
- `outputName`
- `meshInputPath`
- `overwrite`
- `meshDetail`
- `meshSteps`
- `meshGuidance`
- `meshSeed`
- `meshSeedMode`
- `octreeResolution`
- `numChunks`
- `textureViewSize`
- `textureSteps`
- `textureGuidance`
- `textureSize`
- `textureSeed`
- `textureSeedMode`

Example fast mesh preview request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:17333/generate -ContentType 'application/json' -Body (@{
  paths = @('D:\images\chair.png')
  options = @{
    mode = 'meshOnly'
    cudaDevice = 'auto'
    runtimeProfile = 'cu124'
    outputDir = 'D:\models'
    outputName = 'chair-preview'
    overwrite = $true
    meshDetail = 'low'
  }
} | ConvertTo-Json -Depth 6)
```

Example full mesh + texture request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:17333/generate -ContentType 'application/json' -Body (@{
  paths = @('D:\images\chair.png')
  options = @{
    mode = 'meshTexture'
    cudaDevice = '0'
    runtimeProfile = 'cu128'
    outputDir = 'D:\models'
    outputName = 'chair-textured'
  }
} | ConvertTo-Json -Depth 6)
```

### Git Hygiene

Do not commit generated builds, local dependencies, app-managed Python
environments, downloaded source archives, model weights, copied GLBs, job files,
test output, or installer files.

The `.gitignore` covers:

- `node_modules`
- `dist`
- `release`
- `src-tauri/target`
- local Python caches
- generated mesh/model files
- app runtime/model/output folders
- test input/output folders
- installer files

## Texture Note

The supplied ComfyUI workflow, `Jun-2.1_TextureUpscaleWithAuraSR.json`, was used
as a reference for the intended product flow: mesh, multiview texture, upscale,
bake, inpaint, GLB.

The standalone worker currently uses Tencent's direct Hunyuan3D 2.1 Python API
for shape and PBR texture generation. AuraSR-specific parity is tracked as a
follow-up because the standalone app should not depend on ComfyUI custom node
loading.

## Roadmap

- Add explicit `/capabilities` and `/settings` endpoints for richer agent
  discovery.
- Add low-VRAM presets once they are validated.
- Add richer texture controls where Tencent's direct API exposes them.
- Add AuraSR-style optional view upscaling without requiring ComfyUI.
- Add retained job history with thumbnails and turntable previews.
