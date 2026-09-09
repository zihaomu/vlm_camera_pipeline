"""Strict YOLO26 ONNX inference through Ultralytics and ONNX Runtime MIGraphX."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .realtime_pipeline import Detection, DetectorOutput

EXPECTED_YOLO26X_ONNX_SHA256 = (
    "88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a"
)
EXPECTED_ULTRALYTICS_COMMIT = "34e213ca3ece4c18962f5bb922ec74da0c474d24"
EXPECTED_ULTRALYTICS_PATCH_SHA256 = (
    "580502ff7b83c8131e3cc963c8e153dd8082915551361c149c72e3291477522e"
)
MIGRAPHX_PROVIDER = "MIGraphXExecutionProvider"


class MIGraphXBackendError(RuntimeError):
    """The requested strict MIGraphX execution contract was not met."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_revision(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _git_patch_sha256(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=5,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if untracked.stdout.strip():
        return "untracked-files-present"
    return hashlib.sha256(result.stdout).hexdigest()


def _native_package_version(package: str) -> str:
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _literal_metadata(value: str, *, field: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise MIGraphXBackendError(f"invalid ONNX {field} metadata: {value!r}") from exc


def _model_contract(model_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise MIGraphXBackendError(
            f"expected one ONNX input/output, got {len(inputs)}/{len(outputs)}"
        )

    def tensor_shape(value: Any) -> list[int | str]:
        return [
            int(dimension.dim_value) if dimension.dim_value else str(dimension.dim_param)
            for dimension in value.type.tensor_type.shape.dim
        ]

    input_shape = tensor_shape(inputs[0])
    output_shape = tensor_shape(outputs[0])
    if input_shape != [1, 3, 640, 640]:
        raise MIGraphXBackendError(f"unexpected YOLO26 input shape: {input_shape}")
    if output_shape != [1, 300, 6]:
        raise MIGraphXBackendError(f"unexpected YOLO26 output shape: {output_shape}")
    if metadata.get("task") != "detect" or metadata.get("end2end") != "True":
        raise MIGraphXBackendError(
            "YOLO26 ONNX must declare task=detect and end2end=True; "
            f"got task={metadata.get('task')!r}, end2end={metadata.get('end2end')!r}"
        )
    export_args = _literal_metadata(metadata.get("args", "{}"), field="args")
    if not isinstance(export_args, dict) or export_args.get("nms") is not False:
        raise MIGraphXBackendError(
            "expected the locked end-to-end export metadata to declare nms=False"
        )
    names = _literal_metadata(metadata.get("names", "{}"), field="names")
    if not isinstance(names, dict) or not names:
        raise MIGraphXBackendError("ONNX class-name metadata is missing or invalid")
    return {
        "input_name": inputs[0].name,
        "input_shape": input_shape,
        "output_name": outputs[0].name,
        "output_shape": output_shape,
        "metadata": metadata,
        "names": {int(key): str(value) for key, value in names.items()},
    }


def _write_cache_identity(cache_dir: Path, identity: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "identity.json"
    if identity_path.is_file():
        try:
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MIGraphXBackendError(f"invalid cache identity: {identity_path}") from exc
        if existing != identity:
            raise MIGraphXBackendError(
                "MIGraphX cache identity does not match this runtime; choose a new "
                f"--migraphx-cache-dir instead of reusing {cache_dir}"
            )
        return
    unidentified_cache_files = sorted(cache_dir.glob("*.mxr"))
    if unidentified_cache_files:
        raise MIGraphXBackendError(
            "refusing an unidentified MIGraphX cache; choose a new --migraphx-cache-dir "
            f"instead of reusing {cache_dir}"
        )
    temporary = identity_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(identity_path)


class MIGraphXCameraDetector:
    """Run a fixed-shape YOLO26 ONNX model on gfx1151 without CPU EP fallback."""

    backend_name = "onnxruntime-migraphx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        cache_dir: str | Path,
        ultralytics_repository: str | Path,
        device: str = "cuda:0",
        image_size: int = 640,
        confidence: float = 0.5,
        iou: float = 0.45,
        half: bool = True,
    ) -> None:
        model_file = Path(model_path).resolve()
        if not model_file.is_file():
            raise FileNotFoundError(f"model not found: {model_file}")
        if model_file.suffix.lower() != ".onnx":
            raise ValueError(f"MIGraphX requires an ONNX model, got: {model_file}")
        if image_size != 640:
            raise ValueError("the locked YOLO26x ONNX graph requires --imgsz 640")
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= iou <= 1.0:
            raise ValueError("confidence and IoU must be between 0 and 1")
        if not half:
            raise ValueError("the accelerated profile requires MIGraphX FP16")
        if not device.startswith("cuda:"):
            raise ValueError("ROCm PyTorch exposes AMD GPUs through a cuda:N device string")
        try:
            device_id = int(device.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid GPU device: {device!r}") from exc

        os.environ["ULTRALYTICS_MIGRAPHX_STRICT"] = "1"

        import onnxruntime as ort
        import torch
        import ultralytics
        from ultralytics import YOLO

        if MIGRAPHX_PROVIDER not in ort.get_available_providers():
            raise MIGraphXBackendError(
                f"{MIGRAPHX_PROVIDER} is unavailable; found {ort.get_available_providers()}. "
                "Run scripts/setup_migraphx_gfx1151.sh and do not fall back to CPU."
            )
        if not torch.cuda.is_available():
            raise MIGraphXBackendError("ROCm GPU is unavailable; refusing CPU fallback")
        properties = torch.cuda.get_device_properties(device_id)
        architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
        if architecture != "gfx1151":
            raise MIGraphXBackendError(
                f"expected gfx1151, got {architecture!r}; refusing to reuse target cache"
            )

        self._ort = ort
        self._torch = torch
        self._model = YOLO(str(model_file), task="detect")
        self._backend: Any | None = None
        self.model_path = model_file
        self.model_sha256 = _sha256(model_file)
        if self.model_sha256 != EXPECTED_YOLO26X_ONNX_SHA256:
            raise MIGraphXBackendError(
                "YOLO26x ONNX SHA-256 mismatch: "
                f"expected {EXPECTED_YOLO26X_ONNX_SHA256}, got {self.model_sha256}"
            )
        self.contract = _model_contract(model_file)
        self.cache_dir = Path(cache_dir).resolve()
        self.ultralytics_repository = Path(ultralytics_repository).resolve()
        self.ultralytics_commit = _git_revision(self.ultralytics_repository)
        if self.ultralytics_commit != EXPECTED_ULTRALYTICS_COMMIT:
            raise MIGraphXBackendError(
                "unexpected local Ultralytics revision: "
                f"expected {EXPECTED_ULTRALYTICS_COMMIT}, got {self.ultralytics_commit}"
            )
        self.ultralytics_patch_sha256 = _git_patch_sha256(self.ultralytics_repository)
        if self.ultralytics_patch_sha256 != EXPECTED_ULTRALYTICS_PATCH_SHA256:
            raise MIGraphXBackendError(
                "Ultralytics strict I/O Binding patch mismatch: "
                f"expected {EXPECTED_ULTRALYTICS_PATCH_SHA256}, "
                f"got {self.ultralytics_patch_sha256}"
            )
        self.device = device
        self.device_id = device_id
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.half = half
        self.gpu_architecture = architecture
        self.gpu_name = torch.cuda.get_device_name(device_id)
        self._strict_preflight_complete = False
        self._strict_session_seconds: float | None = None
        self._strict_first_run_ms: float | None = None
        self._identity = {
            "schema_version": 1,
            "backend": self.backend_name,
            "model_sha256": self.model_sha256,
            "gpu_arch": self.gpu_architecture,
            "gpu_name": self.gpu_name,
            "rocm": str(torch.version.hip),
            "migraphx": _native_package_version("migraphx"),
            "onnxruntime": ort.__version__,
            "onnxruntime_migraphx_wheel": importlib.metadata.version("onnxruntime-migraphx"),
            "ultralytics": ultralytics.__version__,
            "ultralytics_branch": "add-onnx-migraphx-backend",
            "ultralytics_commit": self.ultralytics_commit,
            "ultralytics_patch_sha256": self.ultralytics_patch_sha256,
            "provider": MIGRAPHX_PROVIDER,
            "migraphx_fp16": True,
            "input_shape": self.contract["input_shape"],
            "output_shape": self.contract["output_shape"],
        }
        _write_cache_identity(self.cache_dir, self._identity)
        os.environ["ULTRALYTICS_MIGRAPHX_CACHE_DIR"] = str(self.cache_dir)

    def _provider_options(self) -> dict[str, str]:
        return {
            "device_id": str(self.device_id),
            "migraphx_fp16_enable": "1",
            "migraphx_model_cache_dir": str(self.cache_dir),
        }

    def _strict_preflight(self) -> None:
        if self._strict_preflight_complete:
            return
        options = self._ort.SessionOptions()
        options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        started = time.perf_counter()
        session = self._ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=[(MIGRAPHX_PROVIDER, self._provider_options())],
        )
        self._strict_session_seconds = time.perf_counter() - started
        if not session.get_providers() or session.get_providers()[0] != MIGRAPHX_PROVIDER:
            raise MIGraphXBackendError(
                f"strict preflight selected unexpected providers: {session.get_providers()}"
            )
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        if list(input_info.shape) != [1, 3, 640, 640] or list(output_info.shape) != [1, 300, 6]:
            raise MIGraphXBackendError(
                f"unexpected ORT I/O shapes: {input_info.shape} -> {output_info.shape}"
            )
        input_tensor = self._torch.zeros(
            (1, 3, 640, 640), dtype=self._torch.float32, device=self.device
        )
        output_tensor = self._torch.empty(
            (1, 300, 6), dtype=self._torch.float32, device=self.device
        )
        io_binding = session.io_binding()
        io_binding.bind_input(
            name=input_info.name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=output_info.name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )
        self._torch.cuda.synchronize(self.device_id)
        started = time.perf_counter()
        session.run_with_iobinding(io_binding)
        self._torch.cuda.synchronize(self.device_id)
        self._strict_first_run_ms = (time.perf_counter() - started) * 1000.0
        if tuple(output_tensor.shape) != (1, 300, 6):
            raise MIGraphXBackendError(f"unexpected strict preflight output: {output_tensor.shape}")
        provider_options = session.get_provider_options().get(MIGRAPHX_PROVIDER, {})
        if provider_options.get("migraphx_fp16_enable") != "1":
            raise MIGraphXBackendError("MIGraphX provider did not enable FP16")
        self._strict_preflight_complete = True
        del input_tensor, output_tensor, io_binding, session

    def _verify_ultralytics_backend(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        auto_backend = getattr(predictor, "model", None)
        backend = getattr(auto_backend, "backend", None)
        if getattr(auto_backend, "format", None) != "onnx" or backend is None:
            raise MIGraphXBackendError("Ultralytics did not construct its ONNX backend")
        providers = list(getattr(backend, "providers", []))
        if not providers or providers[0] != MIGRAPHX_PROVIDER:
            raise MIGraphXBackendError(
                "Ultralytics did not select MIGraphX first: "
                f"{providers}"
            )
        if not getattr(backend, "migraphx_strict", False):
            raise MIGraphXBackendError("Ultralytics strict MIGraphX mode is disabled")
        session_options = backend.session.get_session_options()
        if (
            session_options.get_session_config_entry("session.disable_cpu_ep_fallback")
            != "1"
        ):
            raise MIGraphXBackendError("actual Ultralytics session allows CPU EP fallback")
        if not getattr(backend, "migraphx_fp16", False):
            raise MIGraphXBackendError("Ultralytics MIGraphX FP16 flag is disabled")
        if not getattr(backend, "use_io_binding", False):
            raise MIGraphXBackendError("Ultralytics MIGraphX GPU I/O Binding is disabled")
        bindings = list(getattr(backend, "bindings", []))
        if not bindings or any(not tensor.is_cuda for tensor in bindings):
            raise MIGraphXBackendError("Ultralytics output bindings are not GPU tensors")
        if not getattr(auto_backend, "end2end", False):
            raise MIGraphXBackendError("Ultralytics lost the YOLO26 end2end metadata")
        self._backend = backend

    def warmup(self, frame_bgr: np.ndarray, iterations: int = 2) -> None:
        print(
            "[yolo] stage=strict_preflight provider=MIGraphXExecutionProvider "
            "fp16=1 cpu_ep_fallback=disabled",
            flush=True,
        )
        self._strict_preflight()
        for _ in range(max(1, iterations)):
            self.detect_bgr(frame_bgr)
        self._verify_ultralytics_backend()
        print(
            f"[yolo] stage=ready provider={self._backend.provider} "
            f"io_binding={str(bool(self._backend.use_io_binding)).lower()} "
            f"cache={self.cache_dir}",
            flush=True,
        )

    def detect_bgr(self, frame_bgr: np.ndarray) -> DetectorOutput:
        if not self._strict_preflight_complete:
            self._strict_preflight()
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 uint8 array")
        self._torch.cuda.synchronize(self.device_id)
        results = self._model.predict(
            source=frame_bgr,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            half=self.half,
            verbose=False,
        )
        self._torch.cuda.synchronize(self.device_id)
        self._verify_ultralytics_backend()
        if len(results) != 1:
            raise MIGraphXBackendError(f"expected one Ultralytics result, got {len(results)}")
        result = results[0]
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            coordinates = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            names = result.names
            for xyxy, score, class_id in zip(coordinates, confidences, classes):
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = (
                    str(names.get(int(class_id), class_id))
                    if isinstance(names, dict)
                    else str(names[int(class_id)])
                )
                detections.append(
                    Detection(x1, y1, x2, y2, float(score), int(class_id), label)
                )
        speed = getattr(result, "speed", {}) or {}
        return DetectorOutput(
            detections=tuple(detections),
            preprocess_ms=_optional_float(speed.get("preprocess")),
            inference_ms=_optional_float(speed.get("inference")),
            postprocess_ms=_optional_float(speed.get("postprocess")),
            # The locked end-to-end YOLO26 output only needs confidence filtering.
            external_nms_ms=None,
        )

    def runtime_info(self) -> dict[str, Any]:
        backend = self._backend
        provider_options = (
            backend.provider_options.get(MIGRAPHX_PROVIDER, {})
            if backend is not None
            else {}
        )
        cache_files = sorted(path.name for path in self.cache_dir.glob("*.mxr"))
        return {
            "backend": self.backend_name,
            "loaded": backend is not None,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "gpu_arch": self.gpu_architecture,
            "provider": getattr(backend, "provider", None),
            "providers": list(getattr(backend, "providers", [])),
            "provider_options": dict(provider_options),
            "cpu_ep_fallback_disabled": True,
            "cpu_ep_registered_by_ort": bool(
                backend is not None
                and "CPUExecutionProvider" in getattr(backend, "providers", [])
            ),
            "strict_preflight_passed": self._strict_preflight_complete,
            "strict_session_seconds": _optional_float(self._strict_session_seconds),
            "strict_first_run_ms": _optional_float(self._strict_first_run_ms),
            "migraphx_fp16": bool(getattr(backend, "migraphx_fp16", False)),
            "io_binding": bool(getattr(backend, "use_io_binding", False)),
            "input_residency": "host-copy",
            "output_residency": "host-copy",
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "model_end2end": True,
            "external_nms": False,
            "input_shape": self.contract["input_shape"],
            "output_shape": self.contract["output_shape"],
            "cache_dir": str(self.cache_dir),
            "cache_files": cache_files,
            "ultralytics_branch": "add-onnx-migraphx-backend",
            "ultralytics_commit": self.ultralytics_commit,
            "ultralytics_patch_sha256": self.ultralytics_patch_sha256,
            "onnxruntime": self._ort.__version__,
        }


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return round(number, 3) if math.isfinite(number) else None
