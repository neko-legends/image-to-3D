# Image To Hunyuan 3D Spec

## Goal

Create a dedicated local desktop app for Image to Hunyuan3D generation that
feels like the other `D:\forPublic` AI tools while running as a standalone
app-managed runtime instead of a ComfyUI controller.

## Product Shape

- Name: Image To Hunyuan 3D
- Icon mark: `IH`
- Theme: dark local-workstation UI, orange DepthMap lineage, teal 3D accent
- Primary user: local NVIDIA GPU user
- Primary output: textured GLB

## Runtime Strategy

Use the DepthMapAIGenerator pattern:

- No repository `.venv`.
- Tauri launches a lightweight bootstrap Python worker.
- The worker creates an app-managed virtual environment under app data.
- The worker downloads Hunyuan source, native texture wheels, and model weights
  into app data only after the user confirms setup.
- The worker installs a curated inference dependency set, not upstream demo/API
  packages such as Gradio, FastAPI, Blender Python, Cupy, or Deepspeed.
- Generation runs through Tencent's direct Hunyuan3D 2.1 Python APIs.
- ComfyUI is not required.

## Workflow Strategy

The app supports three modes:

- Mesh + Texture: image to mesh, then PBR texture and GLB export.
- Mesh: image to untextured GLB.
- Texture: existing mesh plus source image to textured GLB.

The supplied `Jun-2.1_TextureUpscaleWithAuraSR.json` remains a reference for the
desired high-quality texture flow. Standalone v1 uses Hunyuan's direct PBR
texture pipeline; AuraSR-specific standalone parity is a follow-up.

## Big TODO

1. Scaffold from DepthMapAIGenerator without copying build outputs. Done.
2. Rename app metadata, package names, ports, and Tauri identifiers. Done.
3. Generate proper IH icon assets. Done.
4. Replace the temporary ComfyUI bridge with a standalone worker. Done.
5. Add app-managed Python environment setup. Done.
6. Add Hunyuan source download. Done.
7. Add CUDA PyTorch and Hunyuan dependency installation. Done.
8. Add model downloads for Hunyuan3D 2.1, DINOv2, and RealESRGAN. Done.
9. Add Windows texture wheel installation. Done.
10. Add runtime status for Python, env, source, models, texture wheels, GPU, and VRAM. Done.
11. Add first-run setup prompt before any large downloads. Done.
12. Block generation until setup is complete. Done.
13. Add queue UI for images and folders. Done.
14. Add Mesh + Texture, Mesh, and Texture modes. Done.
15. Add texture-only source mesh picker. Done.
16. Add mesh controls: steps, guidance, seed, octree, chunks, surface mode. Done.
17. Add texture controls: view size, texture size, views, UV unwrap. Done.
18. Add default root release output folder and optional saved custom output folder. Done.
19. Add GLB preview in the app. Done.
20. Add README with NVIDIA/runtime/.venv guidance. Done.
21. Add `.gitignore` for generated models, outputs, builds, and environments. Done.
22. Add local Agent API toggle for automation while the GUI is running. Done.
23. Add no-GUI headless generation runner. Done.
24. Run Python syntax check.
25. Run TypeScript build.
26. Run Rust/Tauri compile check.
27. Smoke-test the UI in a browser/Tauri shell.
28. Run a small real Hunyuan job when the GPU is available and the user is ready.
29. Add low-VRAM presets after real-job testing.
30. Add AuraSR-style standalone view upscaling after dependency pins are validated.

## Acceptance Criteria

- The app builds with `npm run build`.
- The Tauri backend compiles.
- Runtime status reports app-managed env/source/model readiness and VRAM.
- The app launches without downloading large files.
- The app prompts before setup downloads or installs anything.
- The app can prepare a standalone runtime without ComfyUI after confirmation.
- Image/folder queueing works.
- A job can be submitted to the standalone worker.
- Final GLB paths appear in the queue.
- GLB preview renders when a generated model is selected.
- Agent API can be enabled from the GUI and exposes `/openapi.json`.
- Headless generation can run without the GUI.
- README clearly says the app is designed for NVIDIA GPUs.
- README clearly says no repo `.venv` is needed.
