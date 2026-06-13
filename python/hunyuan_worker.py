from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import venv
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


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


HUNYUAN_REPO = "Tencent-Hunyuan/Hunyuan3D-2.1"
HUNYUAN_SOURCE_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/archive/refs/heads/main.zip"
WHEEL_SOURCE_URL = "https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1/archive/refs/heads/main.zip"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_REPO = "tencent/Hunyuan3D-2.1"
DINO_REPO = "facebook/dinov2-giant"
ENV_MARKER = ".ih3d-env-v4-ready"
SOURCE_MARKER = ".ih3d-source-v1-ready"
MODEL_MARKER = ".ih3d-models-v1-ready"
WHEEL_MARKER = ".ih3d-texture-wheels-v1-ready"
VOXEL_DSS_SIDECAR_MAX_CELLS = 300_000
BOOTSTRAP_PACKAGES = ["pip>=24.3", "setuptools<81", "wheel>=0.43"]
DEFAULT_PYTORCH_PROFILE = "cu124"
PYTORCH_RUNTIME_PROFILES = {
    "cu124": {
        "label": "PyTorch CUDA 12.4",
        "index_url": "https://download.pytorch.org/whl/cu124",
        "packages": [
            "torch==2.5.1+cu124",
            "torchvision==0.20.1+cu124",
            "torchaudio==2.5.1+cu124",
        ],
    },
    "cu128": {
        "label": "PyTorch CUDA 12.8",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "packages": [
            "torch==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "torchaudio==2.7.1+cu128",
        ],
    },
}
COMPAT_PACKAGES = [
    "numpy==1.26.4",
    "requests>=2.32.0",
    "pillow>=10.4.0",
]
HUNYUAN_RUNTIME_PACKAGES = [
    "ninja==1.11.1.1",
    "pybind11==2.13.4",
    "transformers==4.46.0",
    "diffusers==0.30.0",
    "accelerate==1.1.1",
    "pytorch-lightning==1.9.5",
    "huggingface-hub==0.30.2",
    "safetensors==0.4.4",
    "scipy==1.14.1",
    "einops==0.8.0",
    "opencv-python==4.10.0.84",
    "imageio==2.36.0",
    "scikit-image==0.24.0",
    "rembg==2.0.65",
    "realesrgan==0.3.0",
    "basicsr==1.4.2",
    "trimesh==4.4.7",
    "pymeshlab==2022.2.post3",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "omegaconf==2.3.0",
    "pyyaml==6.0.2",
    "configargparse==1.7",
    "tqdm==4.66.5",
    "psutil==6.0.0",
    "onnxruntime-gpu>=1.20.0",
    "torchmetrics==1.6.0",
    "timm",
    "torchdiffeq",
]
MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply"}
PYTHON_VERSION_RE = re.compile(r"Python\s+(\d+)\.(\d+)\.(\d+)")


def normalize_pytorch_profile(value: Any) -> str:
    profile = str(value or DEFAULT_PYTORCH_PROFILE).strip()
    return profile if profile in PYTORCH_RUNTIME_PROFILES else DEFAULT_PYTORCH_PROFILE


def env_marker_profile(marker: Path) -> str | None:
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if text == "ok":
        return "cu124"
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "pytorch":
            return normalize_pytorch_profile(value)
    return None


def env_dir_for_profile(app_data_dir: Path, pytorch_profile: Any = None) -> Path:
    profile = normalize_pytorch_profile(pytorch_profile)
    profiled_env = app_data_dir / f"python-env-{profile}"
    if profiled_env.exists():
        return profiled_env

    legacy_env = app_data_dir / "python-env"
    if legacy_env.exists() and env_marker_profile(legacy_env / ENV_MARKER) == profile:
        return legacy_env

    return profiled_env


def emit(event_type: str, message: str | None = None, **payload: Any) -> None:
    event: dict[str, Any] = {"type": event_type}
    if message:
        event["message"] = message
    event.update(payload)
    print(json.dumps(event, ensure_ascii=True), flush=True)


def emit_timing(stage: str, started: float, path: str | None = None) -> None:
    elapsed_seconds = time.perf_counter() - started
    payload: dict[str, Any] = {
        "stage": stage,
        "elapsedSeconds": round(elapsed_seconds, 3),
    }
    if path:
        payload["path"] = path
    emit("stage_timing", f"{stage} finished in {format_elapsed(elapsed_seconds)}", **payload)


def app_paths(app_data_dir: Path, pytorch_profile: Any = None) -> dict[str, Path]:
    runtime_dir = app_data_dir / "runtime"
    return {
        "app_data": app_data_dir,
        "env_dir": env_dir_for_profile(app_data_dir, pytorch_profile),
        "runtime_dir": runtime_dir,
        "source_dir": runtime_dir / "Hunyuan3D-2.1",
        "wheel_source_dir": runtime_dir / "ComfyUI-Hunyuan3d-2-1-wheels",
        "model_dir": app_data_dir / "models",
        "output_dir": app_data_dir / "outputs",
        "jobs_dir": app_data_dir / "jobs",
    }


def venv_python(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def parse_python_version(text: str) -> tuple[int, int, int] | None:
    match = PYTHON_VERSION_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def python_version(python_path: Path) -> tuple[int, int, int] | None:
    try:
        output = subprocess.check_output(
            [str(python_path), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None
    return parse_python_version(output)


def python_can_import(python_path: Path, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def supported_python_version(version: tuple[int, int, int] | None) -> bool:
    return bool(version and version[0] == 3 and 10 <= version[1] <= 12)


def parse_compute_cap(value: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)(?:\.(\d+))?", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def parse_memory_mb(value: str) -> int:
    match = re.search(r"(\d+)", value.replace(",", ""))
    return int(match.group(1)) if match else 0


def nvidia_gpus() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        cap = parse_compute_cap(parts[2])
        if cap is None:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "capability": cap,
                "memory_mb": parse_memory_mb(parts[3]),
            }
        )
    return gpus


def cap_label(capability: tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def torch_supported_cuda_caps() -> list[tuple[int, int]]:
    try:
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps(torch.cuda.get_arch_list()))",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        arches = json.loads(output)
    except Exception:
        return []
    caps = []
    for arch in arches:
        if not arch.startswith("sm_"):
            continue
        digits = arch.removeprefix("sm_")
        if len(digits) < 2 or not digits.isdigit():
            continue
        caps.append((int(digits[:-1]), int(digits[-1])))
    return caps


def cuda_cap_is_supported(device_cap: tuple[int, int], supported_caps: list[tuple[int, int]]) -> bool:
    if device_cap in supported_caps:
        return True
    device_major, device_minor = device_cap
    return any(major == device_major and minor <= device_minor for major, minor in supported_caps)


def normalize_cuda_device(value: Any) -> str:
    device = str(value or "auto").strip()
    return device if device else "auto"


def configure_cuda_device(cuda_device: Any = "auto") -> None:
    preferred_device = normalize_cuda_device(cuda_device)

    gpus = nvidia_gpus()
    if not gpus:
        return

    supported_caps = torch_supported_cuda_caps()
    if not supported_caps:
        return

    if preferred_device != "auto":
        selected = next((gpu for gpu in gpus if gpu["index"] == preferred_device), None)
        if selected is None:
            detected = ", ".join(f"{gpu['index']}: {gpu['name']}" for gpu in gpus)
            raise RuntimeError(f"Selected CUDA GPU {preferred_device} was not found. Detected GPUs: {detected}.")
        if not cuda_cap_is_supported(selected["capability"], supported_caps):
            supported = ", ".join(cap_label(cap) for cap in supported_caps)
            raise RuntimeError(
                f"Selected GPU {selected['name']} {cap_label(selected['capability'])} is not supported by this "
                f"PyTorch runtime. Supported by this runtime: {supported}. Choose another GPU or reinstall with "
                "the CUDA 12.8 / RTX 50 runtime profile."
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected["index"]
        emit("setup", f"Using selected CUDA GPU {selected['index']}: {selected['name']}")
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        emit("setup", f"Using CUDA device override: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return

    default_gpu = gpus[0]
    if cuda_cap_is_supported(default_gpu["capability"], supported_caps):
        emit("setup", f"Using {default_gpu['name']} for CUDA")
        return

    compatible = [gpu for gpu in gpus if cuda_cap_is_supported(gpu["capability"], supported_caps)]
    if not compatible:
        supported = ", ".join(cap_label(cap) for cap in supported_caps)
        detected = ", ".join(f"{gpu['name']} {cap_label(gpu['capability'])}" for gpu in gpus)
        raise RuntimeError(
            "The installed PyTorch CUDA runtime does not support the detected NVIDIA GPU. "
            f"Detected: {detected}. Supported by this runtime: {supported}."
        )

    selected = max(compatible, key=lambda gpu: gpu["memory_mb"])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = selected["index"]
    emit(
        "setup",
        f"Using {selected['name']} for CUDA because {default_gpu['name']} is not supported by this PyTorch runtime",
    )


def sanitize_installer_line(line: str) -> str | None:
    if line.startswith("Collecting "):
        package = line.removeprefix("Collecting ").split()[0]
        return f"Resolving dependency: {package}"
    if line.startswith("Downloading "):
        return "Downloading dependency packages"
    if line.startswith("Installing collected packages"):
        return "Installing dependency packages"
    if line.startswith("Successfully installed"):
        return "Dependency install step finished"
    if "already satisfied" in line.lower():
        return None
    if line.startswith("ERROR:") or " error:" in line.lower() or "failed" in line.lower():
        return line
    return None


def run_checked(command: list[str], message: str, cwd: Path | None = None) -> None:
    emit("setup", f"{message}. This can take several minutes.")
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    tail: list[str] = []
    error_lines: list[str] = []
    last_status: str | None = None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        tail = tail[-80:]
        lowered = line.lower()
        if line.startswith("ERROR:") or " error:" in lowered or "failed" in lowered:
            error_lines.append(line)
            error_lines = error_lines[-20:]
        status = sanitize_installer_line(line)
        if status and status != last_status:
            emit("setup", status)
            last_status = status
    code = process.wait()
    if code != 0:
        sections = []
        if error_lines:
            sections.append("Error lines:\n" + "\n".join(error_lines))
        sections.append("Last installer output:\n" + ("\n".join(tail) if tail else "No installer output was captured."))
        details = "\n\n".join(sections)
        raise RuntimeError(f"{message} failed.\n\n{details}")


def emit_download_progress(name: str, downloaded: int, total: int | None, message: str | None = None) -> None:
    percent = None
    if total and total > 0:
        percent = min(100.0, round(downloaded / total * 100, 1))
    emit(
        "download_progress",
        message or f"Downloading {name}",
        name=name,
        bytesDownloaded=downloaded,
        bytesTotal=total,
        percent=percent,
    )


def emit_item_progress(name: str, current: int, total: int | None, message: str | None = None) -> None:
    percent = None
    if total and total > 0:
        percent = min(100.0, round(current / total * 100, 1))
    emit(
        "download_progress",
        message or f"Downloading {name}",
        name=name,
        itemsDownloaded=current,
        itemsTotal=total,
        percent=percent,
    )


class SilentProgressWriter:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def hf_snapshot_progress_class(label: str):
    from tqdm.auto import tqdm

    class HfSnapshotProgress(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._ih3d_label = label
            self._ih3d_last_percent: float | None = None
            self._ih3d_last_current: int | None = None
            self._ih3d_last_total: int | None = None
            self._ih3d_last_emit = 0.0
            kwargs["file"] = SilentProgressWriter()
            kwargs["ascii"] = True
            kwargs["dynamic_ncols"] = False
            kwargs.setdefault("mininterval", 0.5)
            super().__init__(*args, **kwargs)
            self._emit_progress(force=True)

        def update(self, n: float = 1) -> bool | None:
            result = super().update(n)
            self._emit_progress()
            return result

        def close(self) -> None:
            self._emit_progress(force=True)
            super().close()

        def _emit_progress(self, force: bool = False) -> None:
            total = int(self.total or 0)
            current = int(self.n or 0)
            percent = round(current / total * 100, 1) if total > 0 else None
            now = time.monotonic()
            if (
                current == self._ih3d_last_current
                and total == self._ih3d_last_total
                and (force or percent == self._ih3d_last_percent or now - self._ih3d_last_emit < 1.0)
            ):
                return
            self._ih3d_last_percent = percent
            self._ih3d_last_current = current
            self._ih3d_last_total = total
            self._ih3d_last_emit = now
            emit_item_progress(self._ih3d_label, current, total or None)

    return HfSnapshotProgress


def download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ImageToHunyuan3D/26.6.13"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            temp_path = destination.with_suffix(destination.suffix + ".tmp")
            downloaded = 0
            next_emit = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_emit:
                        emit_download_progress(label, downloaded, total)
                        next_emit = downloaded + max((total or 100_000_000) // 100, 8 * 1024 * 1024)
            temp_path.replace(destination)
            emit_download_progress(label, downloaded, total or downloaded, f"Downloaded {label}")
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not download {label}: {error}") from error


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                normalized = Path(info.filename)
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise RuntimeError(f"Refusing unsafe zip entry: {info.filename}")
            archive.extractall(temp_dir)
        roots = [path for path in temp_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Unexpected archive shape in {zip_path}")
        shutil.move(str(roots[0]), str(destination))
    return destination


def ensure_source(paths: dict[str, Path]) -> Path:
    source_dir = paths["source_dir"]
    marker = source_dir / SOURCE_MARKER
    if marker.exists():
        return source_dir

    emit("setup", f"Downloading Hunyuan3D source: {HUNYUAN_REPO}")
    archive = paths["runtime_dir"] / "Hunyuan3D-2.1-main.zip"
    download_file(HUNYUAN_SOURCE_URL, archive, "Hunyuan3D source")
    safe_extract_zip(archive, source_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return source_dir


def patch_hunyuan_source(source_dir: Path) -> None:
    rembg_source = source_dir / "hy3dshape" / "hy3dshape" / "rembg.py"
    if rembg_source.is_file():
        content = rembg_source.read_text(encoding="utf-8")
        original = content
        if "def preferred_onnx_providers" not in content:
            content = content.replace(
                "from rembg import remove, new_session\n",
                "from rembg import remove, new_session\n"
                "\n"
                "\n"
                "def preferred_onnx_providers():\n"
                "    try:\n"
                "        import onnxruntime as ort\n"
                "        available = ort.get_available_providers()\n"
                "    except Exception:\n"
                "        return None\n"
                "    if 'CUDAExecutionProvider' in available:\n"
                "        return ['CUDAExecutionProvider', 'CPUExecutionProvider']\n"
                "    return None\n",
            )
        if "providers = preferred_onnx_providers()" not in content:
            content = content.replace(
                "    def __init__(self):\n        self.session = new_session()",
                "    def __init__(self):\n"
                "        providers = preferred_onnx_providers()\n"
                "        if providers:\n"
                "            try:\n"
                "                self.session = new_session(providers=providers)\n"
                "                return\n"
                "            except Exception:\n"
                "                pass\n"
                "        self.session = new_session()",
            )
        if content != original:
            rembg_source.write_text(content, encoding="utf-8")
            emit("setup", "Patched background remover to prefer ONNX CUDA")

    image_super_utils = source_dir / "hy3dpaint" / "utils" / "image_super_utils.py"
    if image_super_utils.is_file():
        content = image_super_utils.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "import numpy as np\nfrom PIL import Image",
            "import numpy as np\nimport torch\nfrom torch.nn import functional as F\nfrom PIL import Image",
        )
        content = content.replace(
            "import numpy as np\nimport torch\nfrom PIL import Image",
            "import numpy as np\nimport torch\nfrom torch.nn import functional as F\nfrom PIL import Image",
        )
        if "def enhance_batch" not in content:
            content = content.replace(
                "    def __call__(self, image):\n"
                "        output, _ = self.upsampler.enhance(np.array(image))\n"
                "        output = Image.fromarray(output)\n"
                "        return output",
                "    def __call__(self, image):\n"
                "        output, _ = self.upsampler.enhance(np.array(image))\n"
                "        output = Image.fromarray(output)\n"
                "        return output\n"
                "\n"
                "    def enhance_batch(self, images):\n"
                "        if len(images) <= 1:\n"
                "            return [self(image) for image in images]\n"
                "        try:\n"
                "            if self.upsampler.tile_size > 0:\n"
                "                return [self(image) for image in images]\n"
                "            tensors = []\n"
                "            for image in images:\n"
                "                array = np.array(image).astype(np.float32)\n"
                "                if len(array.shape) != 3 or array.shape[2] != 3:\n"
                "                    return [self(image) for image in images]\n"
                "                max_range = 65535 if np.max(array) > 256 else 255\n"
                "                array = array / max_range\n"
                "                array = array[:, :, [2, 1, 0]]\n"
                "                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()\n"
                "                tensors.append(tensor)\n"
                "            batch = torch.stack(tensors, dim=0).to(self.upsampler.device)\n"
                "            if self.upsampler.half:\n"
                "                batch = batch.half()\n"
                "            pre_pad = getattr(self.upsampler, \"pre_pad\", 0)\n"
                "            if pre_pad != 0:\n"
                "                batch = F.pad(batch, (0, pre_pad, 0, pre_pad), \"reflect\")\n"
                "            mod_scale = None\n"
                "            if self.upsampler.scale == 2:\n"
                "                mod_scale = 2\n"
                "            elif self.upsampler.scale == 1:\n"
                "                mod_scale = 4\n"
                "            mod_pad_h = 0\n"
                "            mod_pad_w = 0\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = batch.size()\n"
                "                if h % mod_scale != 0:\n"
                "                    mod_pad_h = mod_scale - h % mod_scale\n"
                "                if w % mod_scale != 0:\n"
                "                    mod_pad_w = mod_scale - w % mod_scale\n"
                "                batch = F.pad(batch, (0, mod_pad_w, 0, mod_pad_h), \"reflect\")\n"
                "            with torch.no_grad():\n"
                "                output = self.upsampler.model(batch)\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - mod_pad_h * self.upsampler.scale, 0:w - mod_pad_w * self.upsampler.scale]\n"
                "            if pre_pad != 0:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - pre_pad * self.upsampler.scale, 0:w - pre_pad * self.upsampler.scale]\n"
                "            output = output.float().cpu().clamp_(0, 1).numpy()\n"
                "            results = []\n"
                "            for image_output in output:\n"
                "                image_output = np.transpose(image_output[[2, 1, 0], :, :], (1, 2, 0))\n"
                "                image_output = (image_output * 255.0).round().astype(np.uint8)\n"
                "                results.append(Image.fromarray(image_output))\n"
                "            return results\n"
                "        except Exception:\n"
                "            return [self(image) for image in images]",
            )
        else:
            content = content.replace(
                "        try:\n"
                "            tensors = []\n"
                "            for image in images:\n"
                "                array = np.array(image).astype(np.float32) / 255.0\n"
                "                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()\n"
                "                tensors.append(tensor)\n"
                "            batch = torch.stack(tensors, dim=0).to(self.upsampler.device)\n"
                "            if self.upsampler.half:\n"
                "                batch = batch.half()\n"
                "            with torch.no_grad():\n"
                "                output = self.upsampler.model(batch)\n"
                "            output = output.float().cpu().clamp_(0, 1).numpy()\n"
                "            results = []\n"
                "            for image_output in output:\n"
                "                image_output = np.transpose(image_output, (1, 2, 0))\n"
                "                image_output = (image_output * 255.0).round().astype(np.uint8)\n"
                "                results.append(Image.fromarray(image_output))\n"
                "            return results\n",
                "        try:\n"
                "            if self.upsampler.tile_size > 0:\n"
                "                return [self(image) for image in images]\n"
                "            tensors = []\n"
                "            for image in images:\n"
                "                array = np.array(image).astype(np.float32)\n"
                "                if len(array.shape) != 3 or array.shape[2] != 3:\n"
                "                    return [self(image) for image in images]\n"
                "                max_range = 65535 if np.max(array) > 256 else 255\n"
                "                array = array / max_range\n"
                "                array = array[:, :, [2, 1, 0]]\n"
                "                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()\n"
                "                tensors.append(tensor)\n"
                "            batch = torch.stack(tensors, dim=0).to(self.upsampler.device)\n"
                "            if self.upsampler.half:\n"
                "                batch = batch.half()\n"
                "            pre_pad = getattr(self.upsampler, \"pre_pad\", 0)\n"
                "            if pre_pad != 0:\n"
                "                batch = F.pad(batch, (0, pre_pad, 0, pre_pad), \"reflect\")\n"
                "            mod_scale = None\n"
                "            if self.upsampler.scale == 2:\n"
                "                mod_scale = 2\n"
                "            elif self.upsampler.scale == 1:\n"
                "                mod_scale = 4\n"
                "            mod_pad_h = 0\n"
                "            mod_pad_w = 0\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = batch.size()\n"
                "                if h % mod_scale != 0:\n"
                "                    mod_pad_h = mod_scale - h % mod_scale\n"
                "                if w % mod_scale != 0:\n"
                "                    mod_pad_w = mod_scale - w % mod_scale\n"
                "                batch = F.pad(batch, (0, mod_pad_w, 0, mod_pad_h), \"reflect\")\n"
                "            with torch.no_grad():\n"
                "                output = self.upsampler.model(batch)\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - mod_pad_h * self.upsampler.scale, 0:w - mod_pad_w * self.upsampler.scale]\n"
                "            if pre_pad != 0:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - pre_pad * self.upsampler.scale, 0:w - pre_pad * self.upsampler.scale]\n"
                "            output = output.float().cpu().clamp_(0, 1).numpy()\n"
                "            results = []\n"
                "            for image_output in output:\n"
                "                image_output = np.transpose(image_output[[2, 1, 0], :, :], (1, 2, 0))\n"
                "                image_output = (image_output * 255.0).round().astype(np.uint8)\n"
                "                results.append(Image.fromarray(image_output))\n"
                "            return results\n",
            )
        if content != original:
            image_super_utils.write_text(content, encoding="utf-8")
            emit("setup", "Patched RealESRGAN enhancer for batched views")

    multiview_utils = source_dir / "hy3dpaint" / "utils" / "multiview_utils.py"
    if multiview_utils.is_file():
        content = multiview_utils.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):\n"
            "        pils = self.forward_one(\n"
            "            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input\n"
            "        )",
            "    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, num_inference_steps=None, guidance_scale=None, seed=0):\n"
            "        pils = self.forward_one(\n"
            "            images,\n"
            "            conditions,\n"
            "            prompt=prompt,\n"
            "            custom_view_size=custom_view_size,\n"
            "            resize_input=resize_input,\n"
            "            num_inference_steps=num_inference_steps,\n"
            "            guidance_scale=guidance_scale,\n"
            "            seed=seed,\n"
            "        )",
        )
        content = content.replace(
            "    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):\n"
            "        self.seed_everything(0)",
            "    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, num_inference_steps=None, guidance_scale=None, seed=0):\n"
            "        self.seed_everything(seed)",
        )
        content = content.replace(
            "        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))",
            "        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(seed))",
        )
        content = content.replace(
            "        mvd_image = self.pipeline(\n"
            "            input_images[0:1],\n"
            "            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],\n"
            "            prompt=prompt,\n"
            "            sync_condition=sync_condition,\n"
            "            guidance_scale=3.0,\n"
            "            **kwargs,\n"
            "        ).images",
            "        steps = num_inference_steps or infer_steps_dict[self.pipeline.scheduler.__class__.__name__]\n"
            "        guidance = 3.0 if guidance_scale is None else guidance_scale\n"
            "        mvd_image = self.pipeline(\n"
            "            input_images[0:1],\n"
            "            num_inference_steps=steps,\n"
            "            prompt=prompt,\n"
            "            sync_condition=sync_condition,\n"
            "            guidance_scale=guidance,\n"
            "            **kwargs,\n"
            "        ).images",
        )
        if content != original:
            multiview_utils.write_text(content, encoding="utf-8")
            emit("setup", "Patched texture diffusion options")

    mesh_utils = source_dir / "hy3dpaint" / "DifferentiableRenderer" / "mesh_utils.py"
    if mesh_utils.is_file():
        content = mesh_utils.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "import cv2\nimport bpy\nimport math",
            "import cv2\ntry:\n    import bpy\nexcept ImportError:\n    bpy = None\nimport math",
        )
        if content != original:
            mesh_utils.write_text(content, encoding="utf-8")
            emit("setup", "Patched Hunyuan mesh IO for standalone texture loading")

    mesh_render = source_dir / "hy3dpaint" / "DifferentiableRenderer" / "MeshRender.py"
    if mesh_render.is_file():
        content = mesh_render.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "try:\n"
            "    from .mesh_inpaint_processor import meshVerticeInpaint  # , meshVerticeColor\n"
            "except:\n"
            '    print("InPaint Function CAN NOT BE Imported!!!")',
            "try:\n"
            "    from .mesh_inpaint_processor import meshVerticeInpaint  # , meshVerticeColor\n"
            "except Exception:\n"
            "    try:\n"
            "        from mesh_inpaint_processor import meshVerticeInpaint\n"
            "    except Exception:\n"
            "        meshVerticeInpaint = None\n"
            '        print("InPaint Function CAN NOT BE Imported!!!")',
        )
        content = content.replace(
            "        if vertex_inpaint:\n"
            "            vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()\n"
            "            texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)",
            "        if vertex_inpaint and meshVerticeInpaint is not None:\n"
            "            vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()\n"
            "            texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)",
        )
        if content != original:
            mesh_render.write_text(content, encoding="utf-8")
            emit("setup", "Patched Hunyuan texture inpaint import for standalone wheels")

    texture_pipeline = source_dir / "hy3dpaint" / "textureGenPipeline.py"
    if not texture_pipeline.is_file():
        return
    content = texture_pipeline.read_text(encoding="utf-8")
    original = content
    if "_ih3d_emit_texture_timing" not in content:
        content = content.replace(
            "import os\nimport torch",
            "import os\nimport json\nimport time\nimport torch",
            1,
        )
        content = content.replace(
            "diffusers_logging.set_verbosity(50)\n",
            "diffusers_logging.set_verbosity(50)\n"
            "\n"
            "\n"
            "def _ih3d_format_elapsed(seconds):\n"
            "    if seconds < 60:\n"
            "        return f\"{seconds:.1f}s\"\n"
            "    minutes, remaining = divmod(int(seconds), 60)\n"
            "    hours, minutes = divmod(minutes, 60)\n"
            "    if hours:\n"
            "        return f\"{hours}h {minutes:02d}m {remaining:02d}s\"\n"
            "    return f\"{minutes}m {remaining:02d}s\"\n"
            "\n"
            "\n"
            "def _ih3d_emit_texture_timing(config, stage, started):\n"
            "    try:\n"
            "        elapsed = time.perf_counter() - started\n"
            "        payload = {\n"
            "            \"type\": \"stage_timing\",\n"
            "            \"message\": f\"{stage} finished in {_ih3d_format_elapsed(elapsed)}\",\n"
            "            \"stage\": stage,\n"
            "            \"elapsedSeconds\": round(elapsed, 3),\n"
            "        }\n"
            "        event_path = getattr(config, \"ih3d_event_path\", None)\n"
            "        if event_path:\n"
            "            payload[\"path\"] = event_path\n"
            "        print(json.dumps(payload, ensure_ascii=True), flush=True)\n"
            "    except Exception:\n"
            "        pass\n",
            1,
        )
    content = content.replace(
        "from DifferentiableRenderer.mesh_utils import convert_obj_to_glb",
        "from convert_utils import create_glb_with_pbr_materials",
    )
    content = content.replace(
        "        # Process mesh\n"
        "        path = os.path.dirname(mesh_path)\n"
        "        if use_remesh:\n"
        "            processed_mesh_path = os.path.join(path, \"white_mesh_remesh.obj\")\n"
        "            remesh_mesh(mesh_path, processed_mesh_path)\n"
        "        else:\n"
        "            processed_mesh_path = mesh_path\n",
        "        # Process mesh\n"
        "        path = os.path.dirname(mesh_path)\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        if use_remesh:\n"
        "            processed_mesh_path = os.path.join(path, \"white_mesh_remesh.obj\")\n"
        "            remesh_mesh(mesh_path, processed_mesh_path)\n"
        "        else:\n"
        "            processed_mesh_path = mesh_path\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture mesh prep\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        # Load mesh\n"
        "        mesh = trimesh.load(processed_mesh_path)\n"
        "        mesh = mesh_uv_wrap(mesh)\n"
        "        self.render.load_mesh(mesh=mesh)\n",
        "        # Load mesh\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        mesh = trimesh.load(processed_mesh_path)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture mesh load\", _ih3d_stage_started)\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        mesh = mesh_uv_wrap(mesh)\n"
        "        self.render.load_mesh(mesh=mesh)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture UV unwrap\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        ########### View Selection #########\n"
        "        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(\n"
        "            self.config.candidate_camera_elevs,\n"
        "            self.config.candidate_camera_azims,\n"
        "            self.config.candidate_view_weights,\n"
        "            self.config.max_selected_view_num,\n"
        "        )\n"
        "\n"
        "        normal_maps = self.view_processor.render_normal_multiview(\n"
        "            selected_camera_elevs, selected_camera_azims, use_abs_coor=True\n"
        "        )\n"
        "        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)\n",
        "        ########### View Selection #########\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(\n"
        "            self.config.candidate_camera_elevs,\n"
        "            self.config.candidate_camera_azims,\n"
        "            self.config.candidate_view_weights,\n"
        "            self.config.max_selected_view_num,\n"
        "        )\n"
        "\n"
        "        normal_maps = self.view_processor.render_normal_multiview(\n"
        "            selected_camera_elevs, selected_camera_azims, use_abs_coor=True\n"
        "        )\n"
        "        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture view render\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        multiviews_pbr = self.models[\"multiview_model\"](\n"
        "            image_style,\n"
        "            normal_maps + position_maps,\n"
        "            prompt=image_caption,\n"
        "            custom_view_size=self.config.resolution,\n"
        "            resize_input=True,\n"
        "        )",
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        multiviews_pbr = self.models[\"multiview_model\"](\n"
        "            image_style,\n"
        "            normal_maps + position_maps,\n"
        "            prompt=image_caption,\n"
        "            custom_view_size=self.config.resolution,\n"
        "            resize_input=True,\n"
        "            num_inference_steps=getattr(self.config, \"texture_steps\", None),\n"
        "            guidance_scale=getattr(self.config, \"texture_guidance\", None),\n"
        "            seed=getattr(self.config, \"texture_seed\", 0),\n"
        "        )\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture multiview diffusion\", _ih3d_stage_started)",
    )
    if "Texture multiview diffusion" not in content:
        content = content.replace(
            "        multiviews_pbr = self.models[\"multiview_model\"](\n"
            "            image_style,\n"
            "            normal_maps + position_maps,\n"
            "            prompt=image_caption,\n"
            "            custom_view_size=self.config.resolution,\n"
            "            resize_input=True,\n"
            "            num_inference_steps=getattr(self.config, \"texture_steps\", None),\n"
            "            guidance_scale=getattr(self.config, \"texture_guidance\", None),\n"
            "            seed=getattr(self.config, \"texture_seed\", 0),\n"
            "        )",
            "        _ih3d_stage_started = time.perf_counter()\n"
            "        multiviews_pbr = self.models[\"multiview_model\"](\n"
            "            image_style,\n"
            "            normal_maps + position_maps,\n"
            "            prompt=image_caption,\n"
            "            custom_view_size=self.config.resolution,\n"
            "            resize_input=True,\n"
            "            num_inference_steps=getattr(self.config, \"texture_steps\", None),\n"
            "            guidance_scale=getattr(self.config, \"texture_guidance\", None),\n"
            "            seed=getattr(self.config, \"texture_seed\", 0),\n"
            "        )\n"
            "        _ih3d_emit_texture_timing(self.config, \"Texture multiview diffusion\", _ih3d_stage_started)",
        )
    content = content.replace(
        "        for i in range(len(enhance_images[\"albedo\"])):\n"
        "            enhance_images[\"albedo\"][i] = self.models[\"super_model\"](enhance_images[\"albedo\"][i])\n"
        "            enhance_images[\"mr\"][i] = self.models[\"super_model\"](enhance_images[\"mr\"][i])",
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        super_model = self.models[\"super_model\"]\n"
        "        if hasattr(super_model, \"enhance_batch\"):\n"
        "            enhance_images[\"albedo\"] = super_model.enhance_batch(enhance_images[\"albedo\"])\n"
        "            enhance_images[\"mr\"] = super_model.enhance_batch(enhance_images[\"mr\"])\n"
        "        else:\n"
        "            for i in range(len(enhance_images[\"albedo\"])):\n"
        "                enhance_images[\"albedo\"][i] = super_model(enhance_images[\"albedo\"][i])\n"
        "                enhance_images[\"mr\"][i] = super_model(enhance_images[\"mr\"][i])\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture RealESRGAN enhance\", _ih3d_stage_started)",
    )
    if "Texture RealESRGAN enhance" not in content:
        content = content.replace(
            "        super_model = self.models[\"super_model\"]\n"
            "        if hasattr(super_model, \"enhance_batch\"):\n"
            "            enhance_images[\"albedo\"] = super_model.enhance_batch(enhance_images[\"albedo\"])\n"
            "            enhance_images[\"mr\"] = super_model.enhance_batch(enhance_images[\"mr\"])\n"
            "        else:\n"
            "            for i in range(len(enhance_images[\"albedo\"])):\n"
            "                enhance_images[\"albedo\"][i] = super_model(enhance_images[\"albedo\"][i])\n"
            "                enhance_images[\"mr\"][i] = super_model(enhance_images[\"mr\"][i])",
            "        _ih3d_stage_started = time.perf_counter()\n"
            "        super_model = self.models[\"super_model\"]\n"
            "        if hasattr(super_model, \"enhance_batch\"):\n"
            "            enhance_images[\"albedo\"] = super_model.enhance_batch(enhance_images[\"albedo\"])\n"
            "            enhance_images[\"mr\"] = super_model.enhance_batch(enhance_images[\"mr\"])\n"
            "        else:\n"
            "            for i in range(len(enhance_images[\"albedo\"])):\n"
            "                enhance_images[\"albedo\"][i] = super_model(enhance_images[\"albedo\"][i])\n"
            "                enhance_images[\"mr\"][i] = super_model(enhance_images[\"mr\"][i])\n"
            "        _ih3d_emit_texture_timing(self.config, \"Texture RealESRGAN enhance\", _ih3d_stage_started)",
        )
    content = content.replace(
        "        ###########  Bake  ##########\n"
        "        for i in range(len(enhance_images)):\n"
        "            enhance_images[\"albedo\"][i] = enhance_images[\"albedo\"][i].resize(\n"
        "                (self.config.render_size, self.config.render_size)\n"
        "            )\n"
        "            enhance_images[\"mr\"][i] = enhance_images[\"mr\"][i].resize((self.config.render_size, self.config.render_size))\n"
        "        texture, mask = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"albedo\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n"
        "        texture_mr, mask_mr = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"mr\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n",
        "        ###########  Bake  ##########\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        for i in range(len(enhance_images)):\n"
        "            enhance_images[\"albedo\"][i] = enhance_images[\"albedo\"][i].resize(\n"
        "                (self.config.render_size, self.config.render_size)\n"
        "            )\n"
        "            enhance_images[\"mr\"][i] = enhance_images[\"mr\"][i].resize((self.config.render_size, self.config.render_size))\n"
        "        texture, mask = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"albedo\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n"
        "        texture_mr, mask_mr = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"mr\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture bake\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        ##########  inpaint  ###########\n"
        "        texture = self.view_processor.texture_inpaint(texture, mask_np)\n"
        "        self.render.set_texture(texture, force_set=True)\n"
        "        if \"mr\" in enhance_images:\n"
        "            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)\n"
        "            self.render.set_texture_mr(texture_mr)\n",
        "        ##########  inpaint  ###########\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        texture = self.view_processor.texture_inpaint(texture, mask_np)\n"
        "        self.render.set_texture(texture, force_set=True)\n"
        "        if \"mr\" in enhance_images:\n"
        "            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)\n"
        "            self.render.set_texture_mr(texture_mr)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture inpaint\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        self.render.save_mesh(output_mesh_path, downsample=True)\n"
        "\n"
        "        if save_glb:\n",
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        self.render.save_mesh(output_mesh_path, downsample=True)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture OBJ export\", _ih3d_stage_started)\n"
        "\n"
        "        if save_glb:\n",
    )
    content = content.replace(
        '            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))\n'
        '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
        '            textures = {\n'
        '                "albedo": output_mesh_path.replace(".obj", ".jpg"),\n'
        '                "metallic": output_mesh_path.replace(".obj", "_metallic.jpg"),\n'
        '                "roughness": output_mesh_path.replace(".obj", "_roughness.jpg"),\n'
        '            }\n'
        '            _ih3d_stage_started = time.perf_counter()\n'
        '            create_glb_with_pbr_materials(output_mesh_path, textures, output_mesh_path.replace(".obj", ".glb"))\n'
        '            _ih3d_emit_texture_timing(self.config, "Texture GLB export", _ih3d_stage_started)\n'
        '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
    )
    if "Texture GLB export" not in content:
        content = content.replace(
            '            textures = {\n'
            '                "albedo": output_mesh_path.replace(".obj", ".jpg"),\n'
            '                "metallic": output_mesh_path.replace(".obj", "_metallic.jpg"),\n'
            '                "roughness": output_mesh_path.replace(".obj", "_roughness.jpg"),\n'
            '            }\n'
            '            create_glb_with_pbr_materials(output_mesh_path, textures, output_mesh_path.replace(".obj", ".glb"))\n'
            '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
            '            textures = {\n'
            '                "albedo": output_mesh_path.replace(".obj", ".jpg"),\n'
            '                "metallic": output_mesh_path.replace(".obj", "_metallic.jpg"),\n'
            '                "roughness": output_mesh_path.replace(".obj", "_roughness.jpg"),\n'
            '            }\n'
            '            _ih3d_stage_started = time.perf_counter()\n'
            '            create_glb_with_pbr_materials(output_mesh_path, textures, output_mesh_path.replace(".obj", ".glb"))\n'
            '            _ih3d_emit_texture_timing(self.config, "Texture GLB export", _ih3d_stage_started)\n'
            '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
        )
    if content != original:
        texture_pipeline.write_text(content, encoding="utf-8")
        emit("setup", "Patched Hunyuan texture export for standalone GLB output")


def ensure_wheel_source(paths: dict[str, Path]) -> Path:
    wheel_source_dir = paths["wheel_source_dir"]
    if (wheel_source_dir / "hy3dpaint").is_dir():
        return wheel_source_dir
    emit("setup", "Downloading Windows texture wheels")
    archive = paths["runtime_dir"] / "ComfyUI-Hunyuan3d-2-1-main.zip"
    download_file(WHEEL_SOURCE_URL, archive, "texture wheels")
    safe_extract_zip(archive, wheel_source_dir)
    return wheel_source_dir


def matching_wheel(dist_dir: Path, package_prefix: str, version: tuple[int, int, int]) -> Path | None:
    tag = f"cp{version[0]}{version[1]}"
    candidates = sorted(dist_dir.glob(f"{package_prefix}*{tag}*win_amd64.whl"))
    return candidates[0] if candidates else None


def install_texture_wheels(python_path: Path, paths: dict[str, Path]) -> None:
    marker = paths["env_dir"] / WHEEL_MARKER
    if marker.exists():
        return
    if platform.system() != "Windows":
        emit(
            "setup",
            "Texture native wheels are prebuilt for Windows. Non-Windows setup will use source builds if available.",
        )
        marker.write_text("non-windows\n", encoding="utf-8")
        return

    version = python_version(python_path)
    if not supported_python_version(version):
        raise RuntimeError("Texture wheels require Python 3.10, 3.11, or 3.12.")
    assert version is not None

    wheel_source = ensure_wheel_source(paths)
    raster = matching_wheel(
        wheel_source / "hy3dpaint" / "custom_rasterizer" / "dist",
        "custom_rasterizer",
        version,
    )
    painter = matching_wheel(
        wheel_source / "hy3dpaint" / "DifferentiableRenderer" / "dist",
        "mesh_inpaint_processor",
        version,
    )

    missing = []
    if raster is None:
        missing.append("custom_rasterizer")
    if painter is None:
        missing.append("mesh_inpaint_processor")
    if missing:
        tag = f"cp{version[0]}{version[1]}"
        raise RuntimeError(
            f"Missing prebuilt texture wheel(s) for {tag}: {', '.join(missing)}. "
            "Install Python 3.10, 3.11, or 3.12 and run Prepare again."
        )

    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", str(raster)], "Installing texture rasterizer wheel")
    run_checked(pip + ["install", "--upgrade", str(painter)], "Installing texture inpaint wheel")
    marker.write_text("ok\n", encoding="utf-8")


def ensure_pkg_resources(python_path: Path) -> None:
    if python_can_import(python_path, "pkg_resources"):
        return
    pip = [str(python_path), "-m", "pip"]
    run_checked(
        pip + ["install", "--upgrade", "setuptools<81"],
        "Installing setuptools compatibility package",
    )


def ensure_env(paths: dict[str, Path], source_dir: Path, pytorch_profile: str) -> Path:
    env_dir = paths["env_dir"]
    python_path = venv_python(env_dir)
    marker = env_dir / ENV_MARKER
    if python_path.exists() and not supported_python_version(python_version(python_path)):
        emit("setup", "Recreating app Python environment with a supported Python version")
        shutil.rmtree(env_dir, ignore_errors=True)
        python_path = venv_python(env_dir)

    installed_profile = env_marker_profile(marker)
    if python_path.exists() and marker.exists() and installed_profile != pytorch_profile:
        old_label = PYTORCH_RUNTIME_PROFILES.get(installed_profile or "", {}).get("label", "unknown PyTorch runtime")
        new_label = PYTORCH_RUNTIME_PROFILES[pytorch_profile]["label"]
        emit("setup", f"Recreating app Python environment: {old_label} -> {new_label}")
        shutil.rmtree(env_dir, ignore_errors=True)
        python_path = venv_python(env_dir)

    if not python_path.exists():
        emit("setup", f"Creating app Python environment: {env_dir}")
        venv.EnvBuilder(with_pip=True).create(env_dir)

    if not supported_python_version(python_version(python_path)):
        raise RuntimeError("Image To Hunyuan 3D requires Python 3.10, 3.11, or 3.12.")

    if marker.exists():
        ensure_pkg_resources(python_path)
        return python_path

    profile = PYTORCH_RUNTIME_PROFILES[pytorch_profile]
    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", *BOOTSTRAP_PACKAGES], "Updating pip")
    run_checked(
        pip + ["install", "--upgrade", *profile["packages"], "--index-url", str(profile["index_url"])],
        f"Installing NVIDIA {profile['label']} runtime",
    )
    run_checked(pip + ["install", "--upgrade", *COMPAT_PACKAGES], "Installing shared runtime packages")
    run_checked(
        pip + ["install", "--upgrade", *HUNYUAN_RUNTIME_PACKAGES],
        "Installing Hunyuan inference dependencies",
    )
    run_checked(pip + ["install", "--upgrade", "onnxruntime-gpu>=1.20.0"], "Installing GPU ONNX Runtime")
    run_checked(pip + ["install", "--upgrade", "numpy==1.26.4"], "Pinning compatible NumPy")
    install_texture_wheels(python_path, paths)
    ensure_pkg_resources(python_path)
    marker.write_text(f"pytorch={pytorch_profile}\n", encoding="utf-8")
    return python_path


def prepare_managed_environment(app_data_dir: Path, pytorch_profile: str) -> tuple[Path, dict[str, Path]]:
    paths = app_paths(app_data_dir, pytorch_profile)
    for key in ("runtime_dir", "model_dir", "output_dir", "jobs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    source_dir = ensure_source(paths)
    patch_hunyuan_source(source_dir)
    python_path = ensure_env(paths, source_dir, pytorch_profile)
    return python_path, paths


def managed_env_vars(app_data_dir: Path, paths: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["IH3D_IN_MANAGED_ENV"] = "1"
    env["IH3D_SOURCE_DIR"] = str(paths["source_dir"])
    env["HF_HOME"] = str(paths["model_dir"] / "huggingface")
    env["HF_HUB_CACHE"] = str(paths["model_dir"] / "huggingface" / "hub")
    env["TRANSFORMERS_CACHE"] = str(paths["model_dir"] / "huggingface" / "transformers")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env.pop("TQDM_ASCII", None)
    env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    env["IH3D_APP_DATA_DIR"] = str(app_data_dir)
    return env


def reinvoke_inside_env(python_path: Path, job_path: Path, app_data_dir: Path, paths: dict[str, Path]) -> int:
    command = [str(python_path), str(Path(__file__).resolve()), "--job", str(job_path)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=managed_env_vars(app_data_dir, paths),
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def ensure_realesrgan(paths: dict[str, Path]) -> Path:
    ckpt = paths["source_dir"] / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if ckpt.is_file():
        return ckpt
    download_file(REALESRGAN_URL, ckpt, "RealESRGAN texture enhancer")
    return ckpt


def ensure_models(paths: dict[str, Path], include_texture: bool = True) -> None:
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    marker = paths["model_dir"] / MODEL_MARKER
    realesrgan = paths["source_dir"] / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if marker.exists() and (not include_texture or realesrgan.is_file()):
        return
    ensure_realesrgan(paths)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    emit("model", "Downloading Hunyuan3D 2.1 model bundle")
    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[
            "README.md",
            "hunyuan3d-dit-v2-1/*",
            "hunyuan3d-vae-v2-1/*",
            "hunyuan3d-paintpbr-v2-1/*",
        ],
        local_files_only=False,
        tqdm_class=hf_snapshot_progress_class("Hunyuan3D model files"),
    )
    if include_texture:
        emit("model", "Downloading DINOv2 texture conditioning model")
        snapshot_download(
            repo_id=DINO_REPO,
            local_files_only=False,
            tqdm_class=hf_snapshot_progress_class("DINOv2 model files"),
        )
    marker.write_text("ok\n", encoding="utf-8")


def slugify(value: str, fallback: str = "asset") -> str:
    stem = Path(value).stem if value else fallback
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (slug or fallback)[:72]


def unique_target(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output name for {path}")


def as_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def as_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def as_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"


def seed_value(options: dict[str, Any], key: str, mode_key: str) -> int:
    if str(options.get(mode_key, "fixed")) == "randomize":
        return random.randint(0, 999_999_999)
    return as_int(options.get(key), 1, 0, 999_999_999)


def parse_number_list(value: str, default: list[float]) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except Exception:
        return default
    return parsed or default


def texture_view_count(options: dict[str, Any]) -> int:
    azimuths = parse_number_list(str(options.get("cameraAzimuths", "")), [0, 90, 180, 270, 0, 180])
    return max(6, min(9, len(azimuths)))


def texture_seed_value(options: dict[str, Any]) -> int:
    return seed_value(options, "textureSeed", "textureSeedMode") % (2**32)


def apply_texture_options(config: Any, options: dict[str, Any]) -> None:
    config.texture_steps = as_int(options.get("textureSteps"), 15, 1, 100)
    config.texture_guidance = as_float(options.get("textureGuidance"), 3.0, 0, 30)
    config.texture_seed = texture_seed_value(options)
    azimuths = parse_number_list(str(options.get("cameraAzimuths", "")), [0, 90, 180, 270, 0, 180])
    elevations = parse_number_list(str(options.get("cameraElevations", "")), [0, 0, 0, 0, 90, -90])
    weights = parse_number_list(str(options.get("viewWeights", "")), [1, 0.1, 0.5, 0.1, 0.05, 0.05])
    count = min(len(azimuths), len(elevations), len(weights))
    if count >= 1:
        config.candidate_camera_azims = azimuths[:count]
        config.candidate_camera_elevs = elevations[:count]
        config.candidate_view_weights = weights[:count]


def shape_cache_key(options: dict[str, Any]) -> tuple[Any, ...]:
    return (
        MODEL_REPO,
        as_bool(options.get("enableFlashVdm"), True),
        str(options.get("mcAlgo", "mc")) or "mc",
        as_bool(options.get("forceOffload"), False),
        as_bool(options.get("compileModels"), False),
    )


def paint_cache_key(options: dict[str, Any]) -> tuple[Any, ...]:
    return (
        texture_view_count(options),
        as_int(options.get("textureViewSize"), 512, 256, 2048),
        as_int(options.get("textureSize"), 4096, 512, 8192),
    )


def maybe_compile_shape_pipeline(shape_pipeline: Any, torch: Any) -> None:
    if getattr(shape_pipeline, "_ih3d_compiled", False):
        return
    started = time.perf_counter()
    try:
        if hasattr(shape_pipeline, "model"):
            shape_pipeline.model = torch.compile(shape_pipeline.model, mode="reduce-overhead", dynamic=False)
            shape_pipeline._ih3d_compiled = True
            emit_timing("Shape model compile", started)
    except Exception as error:
        emit("setup", f"Torch compile skipped: {error}")


def output_base_name(options: dict[str, Any], input_path: Path, index: int, total: int) -> str:
    requested = str(options.get("outputName", "")).strip()
    if not requested:
        return slugify(input_path.name)
    base = slugify(requested)
    return f"{base}_{index + 1:03d}" if total > 1 else base


def image_has_alpha(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        if image.mode != "RGBA":
            return False
        alpha = image.getchannel("A")
        return alpha.getextrema()[0] < 255


def configure_torch_performance(torch: Any) -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        emit("setup", "Enabled TF32 matmul and cuDNN autotuning")
    except Exception as error:
        emit("setup", f"Torch performance tuning skipped: {error}")


def load_condition_image(path: Path, rembg: Any | None = None):
    from PIL import Image

    image = Image.open(path)
    if image.mode == "RGBA" and image_has_alpha(path):
        return image.convert("RGBA"), rembg
    from hy3dshape.rembg import BackgroundRemover

    started = time.perf_counter()
    emit("stage_start", f"Removing background for {path.name}", path=str(path))
    if rembg is None:
        rembg = BackgroundRemover()
    result = rembg(image.convert("RGB"))
    emit_timing("Background removal", started, path=str(path))
    return result, rembg


def find_existing_output(*paths: Path) -> list[str]:
    output = []
    for path in paths:
        if path.is_file():
            output.append(str(path.resolve()))
    return output


def cleanup_job_scratch(jobs_dir: Path, keep: Path | None = None) -> None:
    if not jobs_dir.is_dir():
        return
    keep_resolved = keep.resolve() if keep else None
    for entry in jobs_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            if keep_resolved and entry.resolve() == keep_resolved:
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except Exception:
            continue


PIPELINE_CACHE: dict[str, Any] = {
    "shape_key": None,
    "shape_pipeline": None,
    "paint_key": None,
    "paint_pipeline": None,
    "background_remover": None,
    "torch_configured": False,
}


def voxel_sidecar_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".ih3dvox.json")


def write_voxel_sidecar(sidecar: dict[str, Any], mesh_path: Path) -> Path:
    sidecar_path = voxel_sidecar_path(mesh_path)
    payload = {
        **sidecar,
        "modelFile": mesh_path.name,
    }
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return sidecar_path


def voxelize_mesh(mesh: Any, resolution: int) -> tuple[Any, dict[str, Any] | None, int]:
    """Convert a smooth mesh into a blocky voxel-style mesh.

    The mesh is sampled into a voxel grid whose longest side is `resolution`
    voxels, then rebuilt as axis-aligned cubes (Minecraft-style)."""
    extents = getattr(mesh, "extents", None)
    if extents is None or max(extents) <= 0:
        raise RuntimeError("Mesh has no volume to voxelize.")
    pitch = float(max(extents)) / float(max(8, resolution))
    voxel_grid = mesh.voxelized(pitch)
    try:
        voxel_grid = voxel_grid.fill()
    except Exception:
        # Filling the interior is optional; a hollow shell still looks correct.
        pass
    sparse_indices = voxel_grid.sparse_indices
    voxel_count = int(len(sparse_indices))
    sidecar = None
    if voxel_count <= VOXEL_DSS_SIDECAR_MAX_CELLS:
        dims = [int(value) for value in voxel_grid.shape]
        sidecar = {
            "format": "ih3d.voxel-dss.v1",
            "dims": dims,
            "cells": [int(value) for row in sparse_indices for value in row],
            "voxelCount": voxel_count,
            "resolution": int(resolution),
            "pitch": float(pitch),
        }
    voxel_mesh = voxel_grid.as_boxes()
    # Preserve the original placement so downstream texturing lines up.
    voxel_mesh.apply_translation(mesh.bounds.mean(axis=0) - voxel_mesh.bounds.mean(axis=0))
    return voxel_mesh, sidecar, voxel_count


def run_generation(job: dict[str, Any], paths: dict[str, Path]) -> None:
    source_dir = paths["source_dir"]
    launch_cwd = Path.cwd()
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(source_dir / "hy3dshape"))
    sys.path.insert(0, str(source_dir / "hy3dpaint"))
    os.chdir(source_dir)

    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception as error:
        emit("setup", f"torchvision compatibility fix skipped: {error}")

    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    if not PIPELINE_CACHE.get("torch_configured"):
        configure_torch_performance(torch)
        PIPELINE_CACHE["torch_configured"] = True

    options = job.get("options", {})
    mode = str(options.get("mode", "meshTexture"))
    overwrite = as_bool(options.get("overwrite"), False)
    requested_output = Path(str(options.get("outputDir", "")).strip() or paths["output_dir"])
    if not requested_output.is_absolute():
        requested_output = (launch_cwd / requested_output).resolve()
    requested_output.mkdir(parents=True, exist_ok=True)
    cleanup_job_scratch(paths["jobs_dir"])
    run_dir = paths["jobs_dir"] / job.get("job_id", f"job-{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in job.get("paths", [])]
    total = len(input_paths)
    failed_count = 0
    batch_started = time.perf_counter()

    try:
        shape_pipeline = None
        if mode != "textureOnly":
            key = shape_cache_key(options)
            shape_pipeline = PIPELINE_CACHE.get("shape_pipeline") if PIPELINE_CACHE.get("shape_key") == key else None
            if shape_pipeline is None:
                model_started = time.perf_counter()
                emit("model", "Loading Hunyuan3D shape model on CUDA")
                shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_REPO)
                PIPELINE_CACHE["shape_pipeline"] = shape_pipeline
                PIPELINE_CACHE["shape_key"] = key
                emit_timing("Shape model load", model_started)
            else:
                emit("model", "Reusing loaded Hunyuan3D shape model")
            if as_bool(options.get("enableFlashVdm"), False):
                flash_started = time.perf_counter()
                if getattr(shape_pipeline, "_ih3d_flashvdm_enabled", False):
                    pass
                elif hasattr(shape_pipeline, "enable_flashvdm"):
                    try:
                        shape_pipeline.enable_flashvdm(
                            enabled=True,
                            adaptive_kv_selection=False,
                            mc_algo=str(options.get("mcAlgo", "mc")) or "mc",
                            replace_vae=False,
                        )
                        shape_pipeline._ih3d_flashvdm_enabled = True
                        emit_timing("FlashVDM setup", flash_started)
                    except Exception as error:
                        emit("setup", f"FlashVDM setup skipped: {error}")
                else:
                    emit("setup", "FlashVDM setup skipped: upstream pipeline does not expose enable_flashvdm")
            if as_bool(options.get("forceOffload"), False) and hasattr(shape_pipeline, "enable_model_cpu_offload"):
                if not getattr(shape_pipeline, "_ih3d_cpu_offload_enabled", False):
                    shape_pipeline.enable_model_cpu_offload()
                    shape_pipeline._ih3d_cpu_offload_enabled = True
            if as_bool(options.get("compileModels"), False):
                maybe_compile_shape_pipeline(shape_pipeline, torch)

        paint_pipeline = None
        if mode != "meshOnly":
            key = paint_cache_key(options)
            paint_pipeline = PIPELINE_CACHE.get("paint_pipeline") if PIPELINE_CACHE.get("paint_key") == key else None
            if paint_pipeline is None:
                model_started = time.perf_counter()
                emit("model", "Loading Hunyuan3D PBR texture model on CUDA")
                max_num_view = texture_view_count(options)
                resolution = as_int(options.get("textureViewSize"), 512, 256, 2048)
                config = Hunyuan3DPaintConfig(max_num_view, resolution)
                config.realesrgan_ckpt_path = str(source_dir / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
                config.multiview_cfg_path = str(source_dir / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
                config.custom_pipeline = str(source_dir / "hy3dpaint" / "hunyuanpaintpbr")
                if hasattr(config, "texture_size"):
                    config.texture_size = as_int(options.get("textureSize"), getattr(config, "texture_size"), 512, 8192)
                apply_texture_options(config, options)
                paint_pipeline = Hunyuan3DPaintPipeline(config)
                PIPELINE_CACHE["paint_pipeline"] = paint_pipeline
                PIPELINE_CACHE["paint_key"] = key
                emit_timing("Texture model load", model_started)
            else:
                apply_texture_options(paint_pipeline.config, options)
                emit("model", "Reusing loaded Hunyuan3D PBR texture model")

        background_remover = PIPELINE_CACHE.get("background_remover")
        for index, image_path in enumerate(input_paths):
            image_started = time.perf_counter()
            emit("image_start", f"Processing {image_path.name}", path=str(image_path))
            try:
                if not image_path.is_file():
                    raise RuntimeError(f"Input image was not found: {image_path}")

                base = output_base_name(options, image_path, index, total)
                final_mesh_path = unique_target(requested_output / f"{base}.glb", overwrite)
                intermediate_mesh = run_dir / f"{base}_mesh.glb"
                outputs: list[str] = []
                voxel_sidecar: dict[str, Any] | None = None

                if mode != "textureOnly":
                    assert shape_pipeline is not None
                    image, background_remover = load_condition_image(image_path, background_remover)
                    PIPELINE_CACHE["background_remover"] = background_remover
                    seed = seed_value(options, "meshSeed", "meshSeedMode")
                    generator = torch.Generator(device="cuda").manual_seed(seed % (2**32))
                    stage_started = time.perf_counter()
                    emit("stage_start", f"Generating mesh for {image_path.name}", path=str(image_path))
                    meshes = shape_pipeline(
                        image=image,
                        num_inference_steps=as_int(options.get("meshSteps"), 50, 1, 200),
                        guidance_scale=as_float(options.get("meshGuidance"), 5.0, 0, 30),
                        generator=generator,
                        box_v=as_float(options.get("boxV"), 1.01, 0.1, 5),
                        octree_resolution=as_int(options.get("octreeResolution"), 384, 64, 2048),
                        mc_level=as_float(options.get("mcLevel"), 0.0, -10, 10),
                        mc_algo=str(options.get("mcAlgo", "mc")) or None,
                        num_chunks=as_int(options.get("numChunks"), 8000, 1024, 999999),
                        enable_pbar=False,
                    )
                    emit_timing("Mesh generation", stage_started, path=str(image_path))
                    mesh = meshes[0]
                    if as_bool(options.get("voxelize"), False):
                        voxel_started = time.perf_counter()
                        emit("stage_start", f"Voxelizing mesh for {image_path.name}", path=str(image_path))
                        mesh, voxel_sidecar, voxel_count = voxelize_mesh(
                            mesh,
                            as_int(options.get("voxelResolution"), 64, 8, 512),
                        )
                        if voxel_sidecar is None:
                            emit(
                                "setup",
                                f"DSS preview skipped for {image_path.name}: {voxel_count:,} voxels is too large.",
                                path=str(image_path),
                            )
                        emit_timing("Mesh voxelize", voxel_started, path=str(image_path))
                    export_started = time.perf_counter()
                    mesh.export(intermediate_mesh)
                    emit_timing("Mesh export", export_started, path=str(image_path))
                    emit("stage_done", f"Mesh generated for {image_path.name}", path=str(image_path))

                if mode == "meshOnly":
                    shutil.move(str(intermediate_mesh), str(final_mesh_path))
                    outputs.append(str(final_mesh_path.resolve()))
                else:
                    assert paint_pipeline is not None
                    mesh_source = (
                        Path(str(options.get("meshInputPath", "")).strip())
                        if mode == "textureOnly"
                        else intermediate_mesh
                    )
                    if not mesh_source.is_file():
                        raise RuntimeError(f"Source mesh was not found: {mesh_source}")
                    output_obj = final_mesh_path.with_suffix(".obj")
                    stage_started = time.perf_counter()
                    emit("stage_start", f"Generating PBR texture for {image_path.name}", path=str(image_path))
                    paint_pipeline.config.ih3d_event_path = str(image_path)
                    result = paint_pipeline(
                        mesh_path=str(mesh_source),
                        image_path=str(image_path),
                        output_mesh_path=str(output_obj),
                        use_remesh=False,
                        save_glb=True,
                    )
                    emit_timing("Texture generation", stage_started, path=str(image_path))
                    result_path = Path(str(result)) if result else output_obj
                    glb_candidate = output_obj.with_suffix(".glb")
                    if glb_candidate.exists() and glb_candidate != final_mesh_path:
                        if final_mesh_path.exists() and overwrite:
                            final_mesh_path.unlink()
                        shutil.move(str(glb_candidate), str(final_mesh_path))
                    elif result_path.suffix.lower() == ".glb" and result_path.exists() and result_path != final_mesh_path:
                        if final_mesh_path.exists() and overwrite:
                            final_mesh_path.unlink()
                        shutil.move(str(result_path), str(final_mesh_path))
                    outputs.extend(find_existing_output(final_mesh_path, output_obj, output_obj.with_suffix(".jpg")))

                if voxel_sidecar and final_mesh_path.exists():
                    write_voxel_sidecar(voxel_sidecar, final_mesh_path)

                elapsed_seconds = time.perf_counter() - image_started
                emit(
                    "image_done",
                    f"Saved output in {format_elapsed(elapsed_seconds)}",
                    path=str(image_path),
                    outputs=outputs,
                    elapsedSeconds=round(elapsed_seconds, 3),
                )
            except Exception as error:
                failed_count += 1
                elapsed_seconds = time.perf_counter() - image_started
                emit(
                    "image_error",
                    f"{error} after {format_elapsed(elapsed_seconds)}",
                    path=str(image_path),
                    elapsedSeconds=round(elapsed_seconds, 3),
                    traceback=traceback.format_exc(),
                )

        batch_elapsed = format_elapsed(time.perf_counter() - batch_started)
        if failed_count:
            emit("done", f"Batch finished with errors in {batch_elapsed}.", failedCount=failed_count, totalCount=total)
        else:
            emit("done", f"Batch finished in {batch_elapsed}.", failedCount=0, totalCount=total)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        os.chdir(launch_cwd)


def run_managed_job(job: dict[str, Any]) -> None:
    app_data_dir = Path(job["app_data_dir"])
    options = job.get("options", {})
    pytorch_profile = normalize_pytorch_profile(options.get("runtimeProfile"))
    paths = app_paths(app_data_dir, pytorch_profile)
    for key in ("runtime_dir", "model_dir", "output_dir", "jobs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    action = str(job.get("action", "generate"))
    include_texture = action == "setup" or str(options.get("mode", "meshTexture")) != "meshOnly"
    if paths["source_dir"].is_dir():
        patch_hunyuan_source(paths["source_dir"])
    configure_cuda_device(options.get("cudaDevice", "auto"))
    ensure_models(paths, include_texture=include_texture)

    if action == "setup":
        emit("runtime_ready", "Standalone Hunyuan runtime is ready.")
        emit("done", "Standalone Hunyuan runtime is ready.")
        return

    run_generation(job, paths)


def run_outer(job_path: Path) -> int:
    with job_path.open("r", encoding="utf-8") as handle:
        job = json.load(handle)
    app_data_dir = Path(job["app_data_dir"])
    pytorch_profile = normalize_pytorch_profile(job.get("options", {}).get("runtimeProfile"))
    python_path, paths = prepare_managed_environment(app_data_dir, pytorch_profile)
    return reinvoke_inside_env(python_path, job_path, app_data_dir, paths)


def run_daemon() -> int:
    if os.environ.get("IH3D_IN_MANAGED_ENV") != "1":
        emit("error", "Managed worker daemon must be launched inside the app Python environment.")
        return 1
    emit("daemon_ready", "Managed Hunyuan worker is ready.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__shutdown__":
            emit("daemon_exit", "Managed Hunyuan worker is shutting down.")
            return 0
        try:
            request = json.loads(line)
            job_path_value = request.get("jobPath") or request.get("job_path")
            if not job_path_value:
                raise RuntimeError("Daemon request did not include jobPath.")
            job_path = Path(str(job_path_value))
            with job_path.open("r", encoding="utf-8") as handle:
                job = json.load(handle)
            run_managed_job(job)
        except KeyboardInterrupt:
            emit("canceled", "Job canceled.")
        except Exception as error:
            emit("error", str(error), traceback=traceback.format_exc())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()

    try:
        if args.daemon:
            return run_daemon()
        if args.job is None:
            parser.error("--job is required unless --daemon is used")
        if os.environ.get("IH3D_IN_MANAGED_ENV") == "1":
            with args.job.open("r", encoding="utf-8") as handle:
                job = json.load(handle)
            run_managed_job(job)
            return 0
        return run_outer(args.job)
    except KeyboardInterrupt:
        emit("canceled", "Job canceled.")
        return 130
    except Exception as error:
        emit("error", str(error), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
