# Performance Optimization Spec — ImageToHunyuan3D

**Target environment:** Windows 11, NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB GDDR7), cu128 runtime profile (torch 2.7.1+cu128).

**Scope:** Original findings plus implementation status. The optimization passes from this spec are now implemented in guarded form. Runtime validation on the target Windows/RTX 5090 machine is still required for actual timing numbers and for confirming CUDA-provider availability in ONNX Runtime.

**Implemented status:**
- Persistent managed Python worker daemon for generation, fed by Rust over stdin, with cached shape/paint pipelines reused across jobs.
- Direct venv launch for hot generation path; setup still uses system Python to create/update the venv.
- Stage timing events, TF32/cuDNN tuning, FlashVDM default-on wiring, rembg session reuse, GPU ONNX Runtime preference, thumbnail downscaling, and preview auto-rotate pause while busy.
- Upstream source patches for texture steps/guidance/seed, camera elevations/view weights, batched RealESRGAN enhancement with fallback, and standalone PBR GLB export.
- `compileModels` option exposed as an off-by-default guarded torch.compile path for the shape denoiser.

**Original baseline architecture (for reference):**
- Tauri (Rust) spawns system Python → `python/hunyuan_worker.py` re-invokes itself inside a managed venv (`reinvoke_inside_env`, `hunyuan_worker.py:772-786`).
- Per job: shape pipeline `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained` (`hunyuan_worker.py:1001`) + paint pipeline `Hunyuan3DPaintPipeline` (`hunyuan_worker.py:1016`) are loaded fresh, used for all images in the batch, then the process exits.
- Upstream Hunyuan3D-2.1 source is downloaded to app-data and patched at setup time (`patch_hunyuan_source`, `hunyuan_worker.py:556-621`) — the patching mechanism already exists and is the natural vehicle for several optimizations below.

---

## 1. High-impact, zero-quality-risk optimizations

### 1.1 Persistent worker process (eliminate per-job model reload)

**Problem.** Every "Generate" click spawns a fresh Python process which must:
1. Launch system Python, then re-launch venv Python (double interpreter startup, `hunyuan_worker.py:772-786`).
2. Import torch/transformers/diffusers cold (~5–15 s on Windows; Windows process spawn + DLL loading of `torch_cuda.dll`/cuDNN is notoriously slower than Linux).
3. Re-load from disk: DiT (~7.4 GB fp16 ckpt), shape VAE (~0.65 GB), paint UNet, DINOv2-giant (~4.5 GB), RealESRGAN. Checkpoint deserialization plus `.to("cuda")` transfer typically costs 30–90 s per job depending on disk and page cache.
4. Re-initialize CUDA context (~2–5 s).

On a 5090 the actual mesh diffusion (50 steps) may run in ~10–20 s, so **model loading can dominate total wall time for single-image jobs** — likely 60–80 % of the click-to-result latency after the first run.

**Proposal.** Convert the worker into a resident daemon:
- Keep the current job-JSON + JSON-lines-event protocol, but instead of one process per job, have the Rust side (`start_worker_job_core`, `lib.rs:870-990`) start the managed-env Python once and feed it jobs over stdin (or a localhost socket / named pipe). The process holds `shape_pipeline` and `paint_pipeline` resident on the GPU between jobs.
- Idle policy: configurable keep-alive (e.g. unload after N minutes idle, or a UI toggle "Keep models loaded"). 32 GB VRAM on the 5090 comfortably fits both pipelines resident (~16–20 GB total with DINOv2-giant).
- Invalidate/reload only when job options that affect construction change: `runtimeProfile`, `cudaDevice`, texture `max_num_view`/`resolution`/`textureSize` (these are baked into `Hunyuan3DPaintConfig` at `hunyuan_worker.py:1008-1016`), and `forceOffload`.
- Cancel handling must change: today cancel is `taskkill /T /F` (`lib.rs:557-563`). The daemon needs cooperative cancellation (a cancel message checked between pipeline stages, plus an interrupt callback via the pipeline `callback` kwarg between diffusion steps), with `taskkill` retained as the hard-kill fallback that simply costs a reload.
- The Agent API (`lib.rs:1357-1376`) benefits automatically — repeated `/generate` calls hit a warm process.

**Expected gain:** 30–90 s saved per job after the first; near-zero overhead for back-to-back generations.
**Quality risk:** none. **Effort:** medium-high (protocol + lifecycle + cancel semantics). This is the single largest win available.

**Lower-effort interim variant:** keep process-per-job but skip the double-spawn when the env is already prepared — have Rust detect the venv marker and invoke the venv python directly with `IH3D_IN_MANAGED_ENV=1`, eliminating the outer interpreter, its imports, and the `nvidia-smi`/`torch.cuda.get_arch_list()` subprocess probes (`hunyuan_worker.py:262-286`) on the hot path (cache the probe result in a file keyed by driver version). Saves ~3–8 s/job. Effort: low.

### 1.2 Enable FlashVDM volume decoding (option exists but is dead)

**Problem.** The `enableFlashVdm` option is plumbed through the entire stack — UI state (`App.tsx:203`), Rust defaults (`lib.rs:101` area), headless defaults (`ih3d_headless.py:70`) — but the worker **never reads it** and never calls `pipeline.enable_flashvdm(...)`. The VAE therefore uses `VanillaVolumeDecoder`, which evaluates the implicit field at **every** point of the dense `(octree_resolution+1)^3` grid through the cross-attention geo-decoder. At the default 384³ that is ~57 M query points; at 512³ it is ~135 M.

Upstream 2.1 ships `FlashVDMVolumeDecoding` and `HierarchicalVolumeDecoding` (`hy3dshape/models/autoencoders/volume_decoders.py:280,185`) which sparsely refine only near the surface — typically a **2–5× speedup of the latents→mesh stage** (which is a large fraction of shape time at high octree resolutions, often longer than the diffusion itself at 512³).

**Caveats discovered (must handle):**
- Upstream's `enable_flashvdm(replace_vae=True)` maps model names to Hunyuan3D-**2.0** turbo VAEs (`pipelines.py`, `turbo_vae_mapping`) and the 2.1 repo name `Hunyuan3D-2.1` is *not in the map* — so calling it naively with `replace_vae=True` silently keeps the 2.1 VAE (fine) but the cross-attention processor swap and `enable_flashvdm_decoder` path (`model.py:218-227`) still apply. **Correct integration:** call `pipeline.enable_flashvdm(enabled=True, replace_vae=False, mc_algo=<user mc_algo>)`, or set `pipeline.vae.enable_flashvdm_decoder(...)` directly via the existing source-patch mechanism.
- The adaptive top-k KV selection is mathematically an approximation of full attention. With `adaptive_kv_selection=False` it uses plain windowed evaluation ("flash" processor) with no approximation — recommend exposing `enableFlashVdm` as default-on with the exact (non-adaptive) mode, and a tooltip noting the adaptive mode as "fastest". Validate with an A/B mesh diff (Hausdorff distance on a fixed seed) before flipping the default.
- Alternatively/additionally, `HierarchicalVolumeDecoding` is exact-ish (coarse-to-fine with dilation) and a drop-in win for high octree resolutions; worth benchmarking both.

**Expected gain:** 2–5× on the volume-decode stage; at 384³ this typically saves 10–30 s, more at 512³.
**Quality risk:** none in non-adaptive/hierarchical mode; near-zero (validated) in adaptive mode. **Effort:** low — wire one boolean through `run_generation` (`hunyuan_worker.py:1036`).

### 1.3 TF32 + matmul precision + cuDNN autotuning (free Blackwell throughput)

**Problem.** No `torch.backends` configuration exists anywhere in the repo. Defaults in torch 2.7: `allow_tf32=False` for matmul, cuDNN TF32 off, `cudnn.benchmark=False`.

**Proposal.** At managed-worker startup (before pipeline load), set:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
```
- The DiT/VAE/paint UNet run in fp16 (upstream defaults: `pipelines.py from_pretrained dtype=torch.float16`, paint `torch_dtype=torch.float16`, RealESRGAN `half=True`), so TF32 mostly affects the fp32 *fragments*: DINOv2 preprocessing paths, marching-cubes prep (`grid_logits.float()`), schedulers, RealESRGAN pre/post, any layer-norm/softmax accumulations in fp32. Gains are modest but real and strictly free on quality at these tolerances (TF32 has 10-bit mantissa; inputs are fp16 anyway).
- `cudnn.benchmark=True` helps the conv-heavy RealESRGAN x4 upsampler, which runs **2 passes per view** (albedo + MR, `textureGenPipeline.py` enhance loop) = 12–18 RealESRGAN invocations per textured asset at fixed shapes — ideal autotune conditions.

**Expected gain:** 5–15 % on texture stage, small on shape stage. **Quality risk:** negligible (industry-standard inference settings). **Effort:** trivial (one env-guarded block in `run_managed_job`, `hunyuan_worker.py:1113`).

### 1.4 Cache the rembg ONNX session and move it off CPU

**Problem.** Two separate issues in `load_condition_image` (`hunyuan_worker.py:926-936`):
1. `BackgroundRemover()` is constructed **per image** — upstream's class creates a fresh `rembg.new_session()` (u2net ONNX, ~170 MB) each time (`hy3dshape/rembg.py`). Session construction + first-inference warmup is ~2–6 s per image on CPU, multiplied across batch items.
2. `onnxruntime==1.16.3` is the **CPU package** (`hunyuan_worker.py:102`), so u2net inference itself runs on CPU (~1–3 s/image) while a 5090 sits idle.

**Proposal.**
- Hoist a single `BackgroundRemover` instance to batch scope in `run_generation` (lazy-init on first non-alpha image, reused for all). Trivial, pure win.
- Replace `onnxruntime` with `onnxruntime-gpu` in `HUNYUAN_RUNTIME_PACKAGES` and pass CUDA providers to `new_session` (requires bypassing upstream's `BackgroundRemover` with a small local wrapper: `new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`). Verify `onnxruntime-gpu` wheel compat with CUDA 12.8 on Windows (ORT ≥ 1.19 supports CUDA 12; pin needs updating from 1.16.3 regardless). Fallback to CPU provider must be silent.
  - Note: this changes the env, so bump `ENV_MARKER` (`hunyuan_worker.py:45`) to trigger reinstall.

**Expected gain:** 2–8 s per batch image without alpha. **Quality risk:** none (same model). **Effort:** low (instance hoist) / medium (GPU ORT, dependency validation).

### 1.5 `torch.compile` on the DiT denoiser (and optionally paint UNet)

**Problem.** No compilation anywhere. The shape DiT runs 50 forward passes (×2 batch for CFG) per mesh; the paint UNet runs 15 UniPC steps over 6–9 views ×2 (CFG). The upstream pipeline even exposes a ready-made hook: `Hunyuan3DDiTPipeline.compile()` (`pipelines.py`) compiles vae/model/conditioner.

**Proposal.** Compile **only the DiT denoiser** (`shape_pipeline.model = torch.compile(shape_pipeline.model, mode="reduce-overhead", dynamic=False)`), not the VAE (volume decoder query shapes vary with chunking → recompile storms) and not the conditioner (runs once per image).

**Windows-specific caveats (important):**
- torch.compile/Inductor on Windows requires a working MSVC toolchain and has historically been flaky; Triton on Windows is only semi-supported (`triton-windows` community wheels work with torch 2.7 but are not installed here, and the env sets `XFORMERS_FORCE_DISABLE_TRITON=1`, `hunyuan_worker.py:767`).
- Compile cost (~1–4 min first time) only amortizes with the **persistent worker (1.1)**; with process-per-job it is a net loss unless `torch.compiler.save_cache_artifacts`/Inductor FX cache (`TORCHINDUCTOR_CACHE_DIR` pointed at app-data) is configured and proven to survive restarts on Windows.
- Therefore: ship this **behind an off-by-default `compileModels` option**, dependent on 1.1, with automatic fallback (catch compile errors → eager). Treat as a stretch goal; benchmark honestly (expected 10–30 % on the diffusion loop, i.e. a few seconds at 50 steps).

**Quality risk:** none (numerically equivalent within normal tolerance). **Effort:** medium-high due to Windows toolchain validation.

### 1.6 Stop re-encoding the condition image needlessly / cache cond across CFG

Two micro-items in the shape path, both upstream behaviors fixable via the existing source-patch mechanism — only worth doing while touching it anyway:
- `encode_cond` runs DINO conditioning then `unconditional_embedding` each call; fine. But `prepare_image` re-runs full PIL → tensor preprocessing per image; negligible. **Skip — listed to document it was considered.**
- The real one: `enable_pbar=False` is already passed (good). No action.

---

## 2. Texture-stage findings (the dominant cost after fix 1.1)

On a 5090, after shape generation is optimized, the paint pipeline becomes the long pole (typically 1–3 min/asset). Breakdown of stages from `textureGenPipeline.py __call__`: UV unwrap (xatlas, CPU) → multiview render (custom rasterizer) → multiview diffusion (15 UniPC steps, 512² × 6–9 views × 2 outputs) → RealESRGAN ×4 on 12–18 images → bake at `render_size=2048` to `texture_size=4096` → inpaint (CPU `mesh_inpaint_processor`) → save + GLB convert.

### 2.1 RealESRGAN: batch the enhance calls and keep autotune (no quality change)

The enhance loop upscales each albedo and MR view **sequentially** through `RealESRGANer.enhance` (`textureGenPipeline.py`). Each call is a 512→2048 ×4 SR pass; 12–18 calls/asset. With fixed input shapes and `cudnn.benchmark=True` (1.3) the per-call kernels are autotuned once. A further win: patch `imageSuperNet` to stack the views into one batched tensor pass (RRDBNet is fully convolutional; batching 6–9 views uses ~6 GB more VRAM — fine on 32 GB) instead of N separate `enhance` calls with their numpy↔GPU round-trips. Expected: 1.5–3× on the SR stage (often 10–30 s saved). Quality: bit-identical per-image. Effort: low-medium (source patch).

### 2.2 DINOv2-giant: confirm fp16 and consider distilled alternative — **rejected for quality**

DINOv2 is already cast to fp16 upstream (`multiview_utils.py: Dino_v2(...).to(torch.float16)`). Swapping to a smaller DINOv2 variant would change conditioning quality → out of scope per the no-compromise constraint. No action.

### 2.3 Dead texture knobs: align UI with reality (correctness, perf-adjacent)

`textureSteps` (default UI 10) and `textureGuidance` (3.0) are **never used** — upstream hardcodes 15 UniPC steps and CFG 3.0 (`multiview_utils.py: infer_steps_dict`, `guidance_scale=3.0`). `textureSeed` is also dead (upstream seeds 0 always). `unwrapMesh`, `orthoScale`, `cameraElevations`, `viewWeights` are likewise unread by the worker. Either wire them through (steps/guidance are easy: pass into `multiviewDiffusionNet.forward_one` via a source patch) or remove them from the UI. Not a speed item per se, but any future "fewer texture steps" preset is impossible until this plumbing exists, and users currently believe 10 steps are used when 15 actually run.

### 2.4 Texture bake/inpaint are CPU-bound stages

`mesh_inpaint_processor` (prebuilt native wheel) and xatlas UV unwrap run on CPU single-threaded. These are upstream native components; replacing them is high-effort/low-certainty. **Recommendation: instrument first** (see §5) — if profiling on the 5090 box shows unwrap/inpaint > 15 % of texture time, revisit. No action now.

---

## 3. App-layer optimizations (Rust/TS)

### 3.1 Downscale the input-preview data URL

`preview_image_data_url` (`lib.rs:810-822`) base64-encodes the **entire original file** and ships it over Tauri IPC for a small thumbnail; a 20 MB PNG becomes a ~27 MB JSON string, decoded on the JS side, for every queue item added. Fix: decode + downscale to ≤512 px JPEG/WebP in Rust (`image` crate) before encoding, or serve via `convertFileSrc` asset protocol like the GLB preview already does (`App.tsx:469`). Saves UI jank and memory on large/batch drops. Quality: thumbnail only — generation always reads the original path. Effort: low.

### 3.2 `<model-viewer>` auto-rotate burns GPU during generation

The preview viewer has `auto-rotate` + continuous rendering (`App.tsx:1356-1364`) and WebView2 compositing competes (mildly) with CUDA for the same GPU while a job runs. Consider pausing auto-rotate (or the whole viewer) while `state == running`. Minor, but free. Effort: trivial.

### 3.3 Agent API accept loop

The poll loop sleeps 80 ms between non-blocking accepts (`lib.rs:1357-1376`) — adds ≤80 ms latency per request. Irrelevant next to multi-second jobs. **No action; documented as considered.**

---

## 4. Explicitly rejected / opt-in only (quality trade-offs)

These are *not* recommended as defaults under the no-compromise constraint, but should be documented as user-facing choices:

| Idea | Why rejected as default |
|---|---|
| Reduce mesh steps below 50 / octree below 384 | Direct quality loss; already exposed via presets (low/medium/high, `lib.rs:375-397`) and `--fast-preview` |
| FlashVDM adaptive top-k as default | Approximation of attention; opt-in until A/B validated (§1.2) |
| Replace 2.1 VAE with 2.0 turbo VAE | Different model → different geometry; upstream mapping doesn't even include 2.1 |
| Reduce texture views below 6 / view size below 512 | Direct texture quality loss; already user-configurable |
| fp8 / int8 quantization of DiT on Blackwell | Real speed potential on sm_120 but unvalidated quality for this model; research project, not an optimization |
| `enable_model_cpu_offload` by default | It's a *slowdown* for VRAM savings; correctly off by default (`hunyuan_worker.py:1002`). With 32 GB, recommend hiding/discouraging `forceOffload` when detected VRAM ≥ 24 GB |

---

## 5. Instrumentation prerequisite (do this first)

Before implementing 1.5/2.1/2.4, add per-stage timing events to the worker (the `emit` infrastructure already exists, `hunyuan_worker.py:145-149`): cold-start (interpreter+imports), model-load (shape / paint separately), rembg, cond-encode, diffusion loop, volume decode, mesh export, UV unwrap, multiview render, multiview diffusion, SR, bake, inpaint, GLB convert. Emit `elapsedSeconds` per stage (pattern already exists for `image_done`, `hunyuan_worker.py:1085-1092`). This is cheap, ships value to users (visible stage timing), and converts every estimate in this spec into a measurement on the actual 5090/Windows target.

---

## 6. Recommended implementation order

| # | Item | Gain (est.) | Quality risk | Effort |
|---|---|---|---|---|
| 0 | §5 stage instrumentation | — (enables measurement) | none | low |
| 1 | §1.3 TF32/matmul/cudnn.benchmark | 5–15 % texture, small shape | negligible | trivial |
| 2 | §1.4a rembg session hoist | 2–6 s per non-alpha image | none | trivial |
| 3 | §1.2 FlashVDM/hierarchical decode (non-adaptive default) | 10–30 s+ per mesh | none (validated) | low |
| 4 | §1.1-interim skip outer interpreter + cache GPU probes | 3–8 s per job | none | low |
| 5 | §3.1 thumbnail downscale | UI responsiveness | none | low |
| 6 | §1.1 persistent worker daemon | 30–90 s per job | none | med-high |
| 7 | §1.4b onnxruntime-gpu | 1–3 s per image | none | medium |
| 8 | §2.1 batched RealESRGAN | 10–30 s per textured asset | none | medium |
| 9 | §2.3 wire/remove dead texture knobs | correctness | none | low |
| 10 | §1.5 torch.compile (opt-in, needs #6) | 10–30 % of diffusion loop | none | high (Windows) |

Combined realistic effect for a single mesh+texture job on the target machine: **from roughly "1–2 min overhead + 2–4 min compute" down to "~0 overhead + 1.5–3 min compute"** for warm jobs, with zero change to output quality at default settings.
