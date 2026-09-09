"""GPU-resident YOLO26 preprocessing, MIGraphX inference, and detection leases."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .migraphx_detector import (
    EXPECTED_ULTRALYTICS_COMMIT,
    EXPECTED_ULTRALYTICS_PATCH_SHA256,
    EXPECTED_YOLO26X_ONNX_SHA256,
    MIGRAPHX_PROVIDER,
    MIGraphXBackendError,
    _git_patch_sha256,
    _git_revision,
    _model_contract,
    _native_package_version,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Geometry needed to map fixed-size YOLO output back to the source frame."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    gain: float


class GpuDetectionLease:
    """A non-copyable lease over one fixed-address GPU detection buffer."""

    __slots__ = (
        "_owner",
        "_released",
        "frame_id",
        "generation",
        "inference_ms",
        "ready_event",
        "tensor",
        "transform",
    )

    def __init__(
        self,
        owner: _DetectionSlot,
        *,
        frame_id: int,
        inference_ms: float,
        transform: LetterboxTransform,
    ) -> None:
        self._owner = owner
        self._released = False
        self.frame_id = frame_id
        self.generation = owner.generation
        self.inference_ms = inference_ms
        self.ready_event = owner.ready_event
        self.tensor = owner.tensor
        self.transform = transform

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._owner.release(self.generation)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __copy__(self) -> None:
        raise TypeError("GpuDetectionLease is non-copyable")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("GpuDetectionLease is non-copyable")

    def __del__(self) -> None:
        self.release()


class _DetectionSlot:
    __slots__ = ("generation", "in_use", "lock", "ready_event", "tensor")

    def __init__(self, torch: Any, device: str) -> None:
        self.tensor = torch.empty((300, 6), dtype=torch.float32, device=device)
        self.ready_event = torch.cuda.Event(enable_timing=False, blocking=True)
        self.generation = 0
        self.in_use = False
        self.lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self.lock:
            if self.in_use:
                return False
            self.in_use = True
            self.generation += 1
            return True

    def release(self, generation: int) -> None:
        with self.lock:
            if self.generation == generation:
                self.in_use = False


class _InputSlot:
    __slots__ = ("generation", "in_use", "lock", "ready_event", "tensor")

    def __init__(self, torch: Any, device: str) -> None:
        self.tensor = torch.empty((1, 3, 640, 640), dtype=torch.float32, device=device)
        self.ready_event = torch.cuda.Event(enable_timing=False, blocking=True)
        self.generation = 0
        self.in_use = False
        self.lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self.lock:
            if self.in_use:
                return False
            self.in_use = True
            self.generation += 1
            return True

    def release(self, generation: int) -> None:
        with self.lock:
            if self.generation == generation:
                self.in_use = False


class PreparedYoloFrame:
    """A non-copyable lease over fixed input and detection slots."""

    __slots__ = (
        "_detection_slot",
        "_detection_slot_generation",
        "_input_slot",
        "_input_slot_generation",
        "_lock",
        "_owner",
        "_released",
        "frame_id",
        "generation",
        "preprocess_ms",
        "transform",
    )

    def __init__(
        self,
        owner: StrictMIGraphXYolo,
        input_slot: _InputSlot,
        detection_slot: _DetectionSlot,
        *,
        frame_id: int,
        generation: int,
        preprocess_ms: float,
        transform: LetterboxTransform,
    ) -> None:
        self._lock = threading.Lock()
        self._owner = owner
        self._input_slot = input_slot
        self._input_slot_generation = input_slot.generation
        self._released = False
        self._detection_slot = detection_slot
        self._detection_slot_generation = detection_slot.generation
        self.frame_id = frame_id
        self.generation = generation
        self.preprocess_ms = preprocess_ms
        self.transform = transform

    def _consume(self, owner: StrictMIGraphXYolo) -> tuple[_InputSlot, _DetectionSlot]:
        if owner is not self._owner:
            raise ValueError("prepared YOLO frame belongs to a different runner")
        with self._lock:
            if self._released:
                raise RuntimeError("prepared YOLO frame was already consumed or released")
            self._released = True
            return self._input_slot, self._detection_slot

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._input_slot.release(self._input_slot_generation)
        self._detection_slot.release(self._detection_slot_generation)
        self._owner._finish_prepared(self.generation, cancelled=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __copy__(self) -> None:
        raise TypeError("PreparedYoloFrame is non-copyable")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("PreparedYoloFrame is non-copyable")

    def __del__(self) -> None:
        self.release()


class _HipYoloKernels:
    def __init__(self, library: Path) -> None:
        if not library.is_file():
            raise FileNotFoundError(
                f"zero-copy HIP kernel library not found: {library}; "
                "run scripts/setup_zerocopy_kernels_gfx1151.sh"
            )
        self.path = library.resolve()
        self._library = ctypes.CDLL(str(self.path))
        self._library.vlm_camera_zerocopy_last_error.argtypes = []
        self._library.vlm_camera_zerocopy_last_error.restype = ctypes.c_char_p
        self._library.vlm_camera_yolo_letterbox_rgb8_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.vlm_camera_yolo_letterbox_rgb8_f32.restype = ctypes.c_int
        self._library.vlm_camera_yolo_unletterbox_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._library.vlm_camera_yolo_unletterbox_f32.restype = ctypes.c_int

    def _raise(self, operation: str, status: int) -> None:
        detail = self._library.vlm_camera_zerocopy_last_error().decode("utf-8", "replace")
        raise MIGraphXBackendError(f"{operation} failed with status {status}: {detail}")

    def preprocess(
        self,
        *,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool,
        destination_pointer: int,
        target_width: int,
        target_height: int,
        source_ready_event: int,
        stream_pointer: int,
    ) -> LetterboxTransform:
        gain = ctypes.c_float()
        resized_width = ctypes.c_int()
        resized_height = ctypes.c_int()
        pad_left = ctypes.c_int()
        pad_top = ctypes.c_int()
        status = self._library.vlm_camera_yolo_letterbox_rgb8_f32(
            ctypes.c_void_p(source_pointer),
            source_pitch,
            source_width,
            source_height,
            int(source_is_bgr),
            ctypes.c_void_p(destination_pointer),
            target_width,
            target_height,
            ctypes.c_void_p(source_ready_event),
            ctypes.c_void_p(stream_pointer),
            ctypes.byref(gain),
            ctypes.byref(resized_width),
            ctypes.byref(resized_height),
            ctypes.byref(pad_left),
            ctypes.byref(pad_top),
        )
        if status:
            self._raise("YOLO letterbox", status)
        return LetterboxTransform(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            resized_width=resized_width.value,
            resized_height=resized_height.value,
            pad_left=pad_left.value,
            pad_top=pad_top.value,
            gain=float(gain.value),
        )

    def postprocess(
        self,
        *,
        detection_pointer: int,
        detection_count: int,
        confidence: float,
        transform: LetterboxTransform,
        stream_pointer: int,
    ) -> None:
        status = self._library.vlm_camera_yolo_unletterbox_f32(
            ctypes.c_void_p(detection_pointer),
            detection_count,
            confidence,
            transform.source_width,
            transform.source_height,
            transform.gain,
            transform.pad_left,
            transform.pad_top,
            ctypes.c_void_p(stream_pointer),
        )
        if status:
            self._raise("YOLO detection postprocess", status)


class StrictMIGraphXYolo:
    """Fixed-pool YOLO26 runner whose model I/O and image hot path stay on GPU 0."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        cache_dir: str | Path,
        ultralytics_repository: str | Path,
        kernel_library: str | Path,
        confidence: float = 0.5,
        device: str = "cuda:0",
        input_pool_size: int = 2,
        output_pool_size: int = 2,
    ) -> None:
        if device != "cuda:0":
            raise ValueError("the locked zero-copy path currently supports only cuda:0/ROCm0")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if input_pool_size < 2:
            raise ValueError("input_pool_size must be at least 2")
        if output_pool_size < 2:
            raise ValueError("output_pool_size must be at least 2")

        workspace = Path(__file__).resolve().parents[1]
        os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / ".cache"))
        os.environ.setdefault("TORCH_HOME", str(workspace / ".cache" / "torch"))
        os.environ.setdefault("MPLCONFIGDIR", str(workspace / ".cache" / "matplotlib"))
        os.environ.setdefault("HF_HOME", str(workspace / ".cache" / "huggingface"))
        os.environ["ULTRALYTICS_MIGRAPHX_STRICT"] = "1"

        import onnxruntime as ort
        import torch
        from ultralytics.nn.autobackend import AutoBackend

        if MIGRAPHX_PROVIDER not in ort.get_available_providers():
            raise MIGraphXBackendError("ONNX Runtime MIGraphXExecutionProvider is unavailable")
        if not torch.cuda.is_available():
            raise MIGraphXBackendError("ROCm GPU is unavailable; refusing CPU fallback")
        architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
        if architecture != "gfx1151":
            raise MIGraphXBackendError(f"expected gfx1151, got {architecture!r}")

        self.model_path = Path(model_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.ultralytics_repository = Path(ultralytics_repository).resolve()
        if _sha256(self.model_path) != EXPECTED_YOLO26X_ONNX_SHA256:
            raise MIGraphXBackendError("YOLO26x ONNX SHA-256 mismatch")
        if _git_revision(self.ultralytics_repository) != EXPECTED_ULTRALYTICS_COMMIT:
            raise MIGraphXBackendError("Ultralytics base commit mismatch")
        patch_sha = _git_patch_sha256(self.ultralytics_repository)
        if patch_sha != EXPECTED_ULTRALYTICS_PATCH_SHA256:
            raise MIGraphXBackendError(f"Ultralytics strict patch mismatch: {patch_sha}")
        self.contract = _model_contract(self.model_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ULTRALYTICS_MIGRAPHX_CACHE_DIR"] = str(self.cache_dir)

        self._torch = torch
        self._ort = ort
        self._kernels = _HipYoloKernels(Path(kernel_library))
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._output_stream = torch.cuda.Stream(device=0)
        self._input_slots = [_InputSlot(torch, device) for _ in range(input_pool_size)]
        # Kept as a stable diagnostic compatibility handle. Sequential callers
        # always acquire slot zero first; production uses the full fixed pool.
        self.input_tensor = self._input_slots[0].tensor
        self._input_pointer = self.input_tensor.data_ptr()
        self._slots = [_DetectionSlot(torch, device) for _ in range(output_pool_size)]

        self._model = AutoBackend(
            model=str(self.model_path),
            device=torch.device(device),
            fp16=True,
            verbose=False,
        ).eval()
        self._backend = self._model.backend
        self._verify_backend()
        self._binding_pointer = self._backend.bindings[0].data_ptr()
        self._warmup_complete = False
        self._runs = 0
        self._dropped_no_output_slot = 0
        self._dropped_input_busy = 0
        self._prepared_frames = 0
        self._cancelled_prepared_frames = 0
        self._prepared_generation = 0
        self._prepared_active: set[int] = set()
        self.gpu_architecture = architecture
        self.gpu_name = torch.cuda.get_device_name(0)
        self.confidence = confidence
        self.device = device

    def _verify_backend(self) -> None:
        backend = self._backend
        if backend.provider != MIGRAPHX_PROVIDER:
            raise MIGraphXBackendError(f"unexpected provider: {backend.provider}")
        if not backend.migraphx_strict or not backend.migraphx_fp16:
            raise MIGraphXBackendError("strict MIGraphX FP16 mode is not active")
        if not backend.use_io_binding:
            raise MIGraphXBackendError("MIGraphX I/O Binding is not active")
        if (
            backend.session.get_session_options().get_session_config_entry(
                "session.disable_cpu_ep_fallback"
            )
            != "1"
        ):
            raise MIGraphXBackendError("CPU EP fallback is not disabled")
        if len(backend.bindings) != 1:
            raise MIGraphXBackendError(f"expected one output binding, got {len(backend.bindings)}")
        output = backend.bindings[0]
        if not output.is_cuda or output.dtype != self._torch.float32:
            raise MIGraphXBackendError("bound output is not a GPU float32 tensor")
        if tuple(output.shape) != (1, 300, 6):
            raise MIGraphXBackendError(f"unexpected bound output shape: {tuple(output.shape)}")

    def _acquire_slot(self) -> _DetectionSlot | None:
        for slot in self._slots:
            if slot.try_acquire():
                return slot
        self._dropped_no_output_slot += 1
        return None

    def _acquire_input_slot(self) -> _InputSlot | None:
        for slot in self._input_slots:
            if slot.try_acquire():
                return slot
        self._dropped_input_busy += 1
        return None

    def warmup(self, iterations: int = 2) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        with self._state_lock:
            if self._prepared_active:
                raise RuntimeError("cannot warm up while a prepared YOLO frame is outstanding")
        with self._run_lock:
            stream = self._torch.cuda.current_stream(0)
            for slot in self._input_slots:
                slot.tensor.zero_()
                slot.ready_event.record(stream)
            stream.synchronize()
            # Exercise every fixed pointer at least once so pointer rebinding is
            # verified before the camera starts.
            for index in range(max(iterations, len(self._input_slots))):
                output = self._model(self._input_slots[index % len(self._input_slots)].tensor)
                if output.data_ptr() != self._binding_pointer:
                    raise MIGraphXBackendError("ORT output binding pointer changed during warm-up")
            self._warmup_complete = True

    def _finish_prepared(self, generation: int, *, cancelled: bool) -> None:
        with self._state_lock:
            if generation not in self._prepared_active:
                return
            self._prepared_active.remove(generation)
            if cancelled:
                self._cancelled_prepared_frames += 1

    def prepare_rgb8_pointer(
        self,
        *,
        frame_id: int,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool = False,
        source_ready_event: int = 0,
    ) -> PreparedYoloFrame | None:
        """Snapshot a GPU RGB frame into the stable ORT input without running inference."""
        if source_pointer <= 0:
            raise ValueError("source_pointer must be a non-zero GPU address")
        if not self._warmup_complete:
            raise RuntimeError("YOLO runner must be warmed up before preparing frames")
        with self._state_lock:
            input_slot = self._acquire_input_slot()
            if input_slot is None:
                return None
            detection_slot = self._acquire_slot()
            if detection_slot is None:
                input_slot.release(input_slot.generation)
                return None
            self._prepared_generation += 1
            generation = self._prepared_generation
            self._prepared_active.add(generation)

        started = time.perf_counter()
        try:
            input_pointer = input_slot.tensor.data_ptr()
            if input_pointer not in {slot.tensor.data_ptr() for slot in self._input_slots}:
                raise MIGraphXBackendError("fixed YOLO input pool pointer changed")
            stream = self._torch.cuda.current_stream(0)
            transform = self._kernels.preprocess(
                source_pointer=source_pointer,
                source_pitch=source_pitch,
                source_width=source_width,
                source_height=source_height,
                source_is_bgr=source_is_bgr,
                destination_pointer=input_pointer,
                target_width=640,
                target_height=640,
                source_ready_event=source_ready_event,
                stream_pointer=stream.cuda_stream,
            )
            input_slot.ready_event.record(stream)
            # ORT MIGraphX 1.23.2 has no user-stream hook. Wait for this input only;
            # the caller may release/requeue the camera slot as soon as this returns.
            input_slot.ready_event.synchronize()
            preprocess_ms = (time.perf_counter() - started) * 1000.0
            self._prepared_frames += 1
            return PreparedYoloFrame(
                self,
                input_slot,
                detection_slot,
                frame_id=frame_id,
                generation=generation,
                preprocess_ms=preprocess_ms,
                transform=transform,
            )
        except BaseException:
            input_slot.release(input_slot.generation)
            detection_slot.release(detection_slot.generation)
            self._finish_prepared(generation, cancelled=True)
            raise

    def infer_prepared(self, prepared: PreparedYoloFrame) -> GpuDetectionLease:
        """Consume a prepared input; safe to call from the dedicated YOLO worker thread."""
        input_slot, detection_slot = prepared._consume(self)
        try:
            with self._run_lock, self._torch.cuda.device(0):
                started = time.perf_counter()
                raw = self._model(input_slot.tensor)
                inference_ms = (time.perf_counter() - started) * 1000.0
                # run_with_iobinding is synchronous; the fixed input slot can be
                # reused as soon as ORT returns, while output postprocessing continues.
                input_slot.release(prepared._input_slot_generation)
                if raw.data_ptr() != self._binding_pointer:
                    raise MIGraphXBackendError("ORT output binding pointer changed")
                with self._torch.cuda.stream(self._output_stream):
                    detection_slot.tensor.copy_(raw[0], non_blocking=True)
                    self._kernels.postprocess(
                        detection_pointer=detection_slot.tensor.data_ptr(),
                        detection_count=detection_slot.tensor.shape[0],
                        confidence=self.confidence,
                        transform=prepared.transform,
                        stream_pointer=self._output_stream.cuda_stream,
                    )
                    detection_slot.ready_event.record(self._output_stream)
                # The fixed ORT output binding may be overwritten by the next run. A
                # per-slot event wait (not a device-wide sync) closes that reuse hazard.
                detection_slot.ready_event.synchronize()
                self._runs += 1
                return GpuDetectionLease(
                    detection_slot,
                    frame_id=prepared.frame_id,
                    inference_ms=inference_ms,
                    transform=prepared.transform,
                )
        except BaseException:
            detection_slot.release(prepared._detection_slot_generation)
            raise
        finally:
            input_slot.release(prepared._input_slot_generation)
            self._finish_prepared(prepared.generation, cancelled=False)

    def infer_rgb8_pointer(
        self,
        *,
        frame_id: int,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool = False,
        source_ready_event: int = 0,
    ) -> GpuDetectionLease | None:
        """Prepare and synchronously infer one GPU-resident packed RGB/BGR8 frame."""
        prepared = self.prepare_rgb8_pointer(
            frame_id=frame_id,
            source_pointer=source_pointer,
            source_pitch=source_pitch,
            source_width=source_width,
            source_height=source_height,
            source_is_bgr=source_is_bgr,
            source_ready_event=source_ready_event,
        )
        if prepared is None:
            return None
        return self.infer_prepared(prepared)

    def runtime_info(self) -> dict[str, Any]:
        cache_files = sorted(path.name for path in self.cache_dir.glob("*.mxr"))
        return {
            "schema_version": 1,
            "backend": "ultralytics-onnxruntime-migraphx-strict-iobinding",
            "provider": self._backend.provider,
            "providers_registered": list(self._backend.providers),
            "cpu_ep_fallback_disabled": True,
            "migraphx_fp16": bool(self._backend.migraphx_fp16),
            "io_binding": bool(self._backend.use_io_binding),
            "input_device": "gpu:0",
            "output_device": "gpu:0",
            "input_pointer": self._input_pointer,
            "input_pool_pointers": [slot.tensor.data_ptr() for slot in self._input_slots],
            "input_pool_size": len(self._input_slots),
            "ort_output_pointer": self._binding_pointer,
            "output_pool_pointers": [slot.tensor.data_ptr() for slot in self._slots],
            "output_pool_size": len(self._slots),
            "input_shape": list(self.input_tensor.shape),
            "output_shape": [1, 300, 6],
            "input_dtype": str(self.input_tensor.dtype),
            "output_dtype": str(self._backend.bindings[0].dtype),
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "tensor_h2d_bytes": 0,
            "tensor_d2h_bytes": 0,
            "tensor_d2d_bytes_per_run": 300 * 6 * 4,
            "hot_path_numpy": False,
            "global_device_synchronize": False,
            "input_event_wait": True,
            "source_event_wait_supported": True,
            "split_prepare_infer_api": True,
            "background_inference_supported": True,
            "fixed_input_outstanding_limit": len(self._input_slots),
            "newest_pending_snapshot_supported": True,
            "output_event_synchronized_before_ort_reuse": True,
            "runs": self._runs,
            "prepared_frames": self._prepared_frames,
            "cancelled_prepared_frames": self._cancelled_prepared_frames,
            "dropped_input_busy": self._dropped_input_busy,
            "dropped_no_output_slot": self._dropped_no_output_slot,
            "warmup_complete": self._warmup_complete,
            "gpu_arch": self.gpu_architecture,
            "gpu_name": self.gpu_name,
            "model_sha256": EXPECTED_YOLO26X_ONNX_SHA256,
            "ultralytics_commit": EXPECTED_ULTRALYTICS_COMMIT,
            "ultralytics_patch_sha256": EXPECTED_ULTRALYTICS_PATCH_SHA256,
            "kernel_library": str(self._kernels.path),
            "kernel_library_sha256": hashlib.sha256(self._kernels.path.read_bytes()).hexdigest(),
            "migraphx": _native_package_version("migraphx"),
            "onnxruntime": self._ort.__version__,
            "cache_dir": str(self.cache_dir),
            "cache_files": cache_files,
        }


def hip_kernel_code_objects(library: str | Path) -> list[str]:
    """Return target labels embedded in the repo-local HIP library."""
    result = subprocess.run(
        ["roc-obj-ls", str(Path(library).resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return sorted(
        {
            line.split()[1]
            for line in result.stdout.splitlines()
            if len(line.split()) >= 2 and line.split()[1].startswith("hipv4-")
        }
    )
