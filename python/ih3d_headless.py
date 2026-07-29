from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


APP_ID = "public.image.to3d"
LEGACY_APP_ID = "public.image.to.hunyuan3d"
MESH_DETAIL_PRESETS = {
    "low": {
        "meshSteps": 24,
        "octreeResolution": 256,
        "numChunks": 6000,
        "maxFaces": 5000,
    },
    "medium": {
        "meshSteps": 50,
        "octreeResolution": 384,
        "numChunks": 8000,
        "maxFaces": 20000,
    },
    "high": {
        "meshSteps": 80,
        "octreeResolution": 512,
        "numChunks": 12000,
        "maxFaces": 50000,
    },
}
TRELLIS_PRESETS = {
    "preview": {
        "trellisResolution": 512,
        "trellisSparseSteps": 12,
        "trellisShapeSteps": 12,
        "trellisTextureSteps": 12,
        "trellisTextureGuidance": 3.0,
        "trellisDecimationTarget": 20000,
        "trellisTextureSize": 1024,
        "trellisRemesh": False,
    },
    "standard": {
        "trellisResolution": 1024,
        "trellisSparseSteps": 12,
        "trellisShapeSteps": 12,
        "trellisTextureSteps": 12,
        "trellisTextureGuidance": 7.5,
        "trellisDecimationTarget": 100000,
        "trellisTextureSize": 2048,
        "trellisRemesh": False,
    },
    "quality": {
        "trellisResolution": 1024,
        "trellisSparseSteps": 25,
        "trellisShapeSteps": 25,
        "trellisTextureSteps": 25,
        "trellisTextureGuidance": 7.5,
        "trellisDecimationTarget": 200000,
        "trellisTextureSize": 2048,
        "trellisRemesh": False,
    },
    "highpoly": {
        "trellisResolution": 1024,
        "trellisSparseSteps": 25,
        "trellisShapeSteps": 25,
        "trellisTextureSteps": 25,
        "trellisTextureGuidance": 7.5,
        "trellisDecimationTarget": 500000,
        "trellisTextureSize": 4096,
        "trellisRemesh": False,
    },
}


def configure_text_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_text_streams()
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
os.environ.pop("TQDM_ASCII", None)


def default_app_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        modern = base / APP_ID
        legacy = base / LEGACY_APP_ID
        return legacy if legacy.exists() and not modern.exists() else modern
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    modern = base / APP_ID
    legacy = base / LEGACY_APP_ID
    return legacy if legacy.exists() and not modern.exists() else modern


def default_output_dir(root: Path) -> Path:
    portable_exe = root / "release" / "portable" / "image-to-3d.exe"
    if portable_exe.exists():
        return portable_exe.parent / "output"
    return root / "output"


def default_options(root: Path) -> dict[str, object]:
    return {
        "modelBackend": "hunyuan",
        "mode": "meshTexture",
        "cudaDevice": "auto",
        "runtimeProfile": "cu124",
        "outputDir": str(default_output_dir(root)),
        "outputName": "",
        "meshInputPath": "",
        "recursive": True,
        "overwrite": False,
        "meshDetail": "low",
        "meshSteps": 24,
        "meshGuidance": 5.0,
        "meshSeed": 2,
        "meshSeedMode": "randomize",
        "boxV": 1.01,
        "octreeResolution": 256,
        "numChunks": 6000,
        "mcLevel": 0.0,
        "mcAlgo": "mc",
        "enableFlashVdm": True,
        "compileModels": False,
        "forceOffload": False,
        "removeFloaters": True,
        "removeDegenerateFaces": True,
        "reduceFaces": True,
        "maxFaces": 5000,
        "smoothNormals": True,
        "voxelize": False,
        "voxelResolution": 64,
        "textureViewSize": 512,
        "textureSteps": 15,
        "textureGuidance": 3.0,
        "textureSize": 2048,
        "textureSeed": 802344256,
        "textureSeedMode": "randomize",
        "unwrapMesh": True,
        "cameraAzimuths": "0, 90, 180, 270, 0, 180",
        "cameraElevations": "0, 0, 0, 0, 90, -90",
        "viewWeights": "1, 0.1, 0.5, 0.1, 0.05, 0.05",
        "orthoScale": 1.1,
        "trellisPreset": "standard",
        "trellisAutocrop": True,
        "trellisForceOpaque": True,
        "trellisAtlasPadding": 12,
        "trellisResolution": 1024,
        "trellisSeed": 12345,
        "trellisSeedMode": "randomize",
        "trellisSparseSteps": 12,
        "trellisSparseGuidance": 7.5,
        "trellisSparseRescale": 0.7,
        "trellisSparseRescaleT": 5.0,
        "trellisShapeSteps": 12,
        "trellisShapeGuidance": 7.5,
        "trellisShapeRescale": 0.5,
        "trellisShapeRescaleT": 3.0,
        "trellisTextureSteps": 12,
        "trellisTextureGuidance": 7.5,
        "trellisTextureRescale": 0.5,
        "trellisTextureRescaleT": 3.0,
        "trellisDecimationTarget": 100000,
        "trellisTextureSize": 2048,
        "trellisRemesh": False,
        "trellisExportWebp": True,
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Image To 3D without the GUI.")
    parser.add_argument("inputs", nargs="*", help="Input image paths for generation.")
    parser.add_argument("--action", choices=["generate", "setup"], default="generate")
    parser.add_argument("--model-backend", choices=["hunyuan", "trellis2"], default=None)
    parser.add_argument("--mode", choices=["meshTexture", "meshOnly", "textureOnly"], default=None)
    parser.add_argument("--cuda-device", default=None, help="CUDA GPU index to use, or auto.")
    parser.add_argument(
        "--runtime-profile",
        choices=["cu124", "cu128"],
        default=None,
        help="PyTorch CUDA runtime profile. cu128 is intended for RTX 50 / Blackwell GPUs.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--mesh-input-path", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mesh-detail", choices=["low", "medium", "high", "custom"], default=None)
    parser.add_argument(
        "--trellis-preset",
        choices=["preview", "standard", "quality", "highpoly", "custom"],
        default=None,
    )
    parser.add_argument("--mesh-steps", type=int, default=None)
    parser.add_argument("--mesh-guidance", type=float, default=None)
    parser.add_argument("--mesh-seed", type=int, default=None)
    parser.add_argument("--mesh-seed-mode", choices=["fixed", "randomize"], default=None)
    parser.add_argument("--octree-resolution", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--max-faces", type=int, default=None)
    parser.add_argument("--trellis-decimation-target", type=int, default=None)
    parser.add_argument("--texture-view-size", type=int, default=None)
    parser.add_argument("--texture-size", type=int, default=None)
    parser.add_argument("--voxelize", action="store_true", help="Export a blocky voxel-style mesh instead of a smooth mesh.")
    parser.add_argument("--voxel-resolution", type=int, default=None, help="Voxel grid resolution along the longest axis (8-512).")
    parser.add_argument("--fast-preview", action="store_true", help="Use faster mesh-only preview settings.")
    parser.add_argument("--app-data-dir", type=Path, default=None)
    return parser


def build_options(args: argparse.Namespace, root: Path) -> dict[str, object]:
    options = default_options(root)
    if args.fast_preview:
        options.update(
            {
                "mode": "meshOnly",
                "meshDetail": "custom",
                "meshSteps": 12,
                "octreeResolution": 256,
                "numChunks": 6000,
                "maxFaces": 5000,
            }
        )
        options["trellisPreset"] = "preview"
        options.update(TRELLIS_PRESETS["preview"])
    if args.mesh_detail:
        options["meshDetail"] = args.mesh_detail
        if args.mesh_detail in MESH_DETAIL_PRESETS:
            options.update(MESH_DETAIL_PRESETS[args.mesh_detail])
    if args.trellis_preset:
        options["trellisPreset"] = args.trellis_preset
        if args.trellis_preset in TRELLIS_PRESETS:
            options.update(TRELLIS_PRESETS[args.trellis_preset])

    overrides = {
        "mode": args.mode,
        "modelBackend": args.model_backend,
        "cudaDevice": args.cuda_device,
        "runtimeProfile": args.runtime_profile,
        "outputDir": str(Path(args.output_dir).resolve()) if args.output_dir else None,
        "outputName": args.output_name,
        "meshInputPath": str(Path(args.mesh_input_path).resolve()) if args.mesh_input_path else None,
        "overwrite": args.overwrite if args.overwrite else None,
        "voxelize": args.voxelize if args.voxelize else None,
        "voxelResolution": args.voxel_resolution,
        "meshGuidance": args.mesh_guidance,
        "meshSeed": args.mesh_seed,
        "meshSeedMode": args.mesh_seed_mode,
        "textureViewSize": args.texture_view_size,
        "textureSize": args.texture_size,
        "trellisDecimationTarget": args.trellis_decimation_target,
    }
    for key, value in overrides.items():
        if value is not None:
            options[key] = value
    mesh_overrides = {
        "meshSteps": args.mesh_steps,
        "octreeResolution": args.octree_resolution,
        "numChunks": args.num_chunks,
        "maxFaces": args.max_faces,
    }
    if any(value is not None for value in mesh_overrides.values()):
        options["meshDetail"] = "custom"
        for key, value in mesh_overrides.items():
            if value is not None:
                options[key] = value
    if args.trellis_decimation_target is not None:
        options["trellisPreset"] = "custom"
    return options


def main() -> int:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    worker = root / "python" / "hunyuan_worker.py"
    app_data_dir = args.app_data_dir or default_app_data_dir()
    jobs_dir = app_data_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "generate" and not args.inputs:
        print("No input images were provided.", file=sys.stderr)
        return 2

    job_id = f"headless-{uuid.uuid4()}"
    job_path = jobs_dir / f"{job_id}.json"
    job = {
        "action": args.action,
        "job_id": job_id,
        "paths": [str(Path(path).resolve()) for path in args.inputs],
        "options": build_options(args, root),
        "app_data_dir": str(app_data_dir),
    }
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, str(worker), "--job", str(job_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
