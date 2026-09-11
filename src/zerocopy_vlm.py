"""Strict GPU-resident Qwen3-VL preprocessing and llama.cpp HIP IPC v2 client."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .vlm import (
    EXPECTED_HIP_CODE_OBJECT,
    EXPECTED_MMPROJ_SHA256,
    EXPECTED_MODEL_SHA256,
    LlamaCppConfig,
    LlamaCppVlm,
    VlmRequestError,
    VlmStartupError,
    _request_json,
    _sha256,
)

EXPECTED_LLAMA_IPC_COMMIT = "be84695622f7be5c307d726379ddd745993669b8"
LLAMA_IPC_PROTOCOL = "vlm-hip-ipc-v2"
NORMALIZATION = "qwen3-vl-mean-0.5-std-0.5"
RESIZE_MODE = "smart_resize-pad_ceil-bilinear-align_corners"
PATCH_SIZE = 16
MERGE_SIZE = 2
ROCWMMA_CMAKE_ENTRY = "GGML_HIP_ROCWMMA_FATTN:BOOL=ON"
ROCWMMA_KERNEL_MARKER = "ggml_cuda_flash_attn_ext_wmma_f16"
SINGLE_SENTENCE_GBNF = (
    'root ::= [A-Z] [A-Za-z0-9\'-]* (" " [A-Za-z0-9] [A-Za-z0-9\'-]*){0,15} "."'
)


class ZeroCopyVlmError(RuntimeError):
    """The strict VLM device-input contract was violated."""


@dataclass(frozen=True, slots=True)
class QwenPreprocessGeometry:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    image_max_tokens: int

    @property
    def allocation_bytes(self) -> int:
        return self.target_width * self.target_height * 3 * 4


class _HipIpcBytes(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


class _HipUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class _HipRuntime:
    EVENT_DISABLE_TIMING = 0x2
    EVENT_INTERPROCESS = 0x4
    HOST_MALLOC_DEFAULT = 0
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, device_id: int = 0) -> None:
        self._library = ctypes.CDLL("libamdhip64.so")
        self._library.hipGetErrorString.argtypes = [ctypes.c_int]
        self._library.hipGetErrorString.restype = ctypes.c_char_p
        self._library.hipSetDevice.argtypes = [ctypes.c_int]
        self._library.hipSetDevice.restype = ctypes.c_int
        self._library.hipDeviceGetUuid.argtypes = [ctypes.POINTER(_HipUuid), ctypes.c_int]
        self._library.hipDeviceGetUuid.restype = ctypes.c_int
        self._library.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._library.hipMalloc.restype = ctypes.c_int
        self._library.hipFree.argtypes = [ctypes.c_void_p]
        self._library.hipFree.restype = ctypes.c_int
        self._library.hipHostMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._library.hipHostMalloc.restype = ctypes.c_int
        self._library.hipHostFree.argtypes = [ctypes.c_void_p]
        self._library.hipHostFree.restype = ctypes.c_int
        self._library.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._library.hipMemcpyAsync.restype = ctypes.c_int
        self._library.hipEventCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self._library.hipEventCreateWithFlags.restype = ctypes.c_int
        self._library.hipEventDestroy.argtypes = [ctypes.c_void_p]
        self._library.hipEventDestroy.restype = ctypes.c_int
        self._library.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._library.hipEventRecord.restype = ctypes.c_int
        self._library.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        self._library.hipEventSynchronize.restype = ctypes.c_int
        self._library.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_HipIpcBytes),
            ctypes.c_void_p,
        ]
        self._library.hipIpcGetMemHandle.restype = ctypes.c_int
        self._library.hipIpcGetEventHandle.argtypes = [
            ctypes.POINTER(_HipIpcBytes),
            ctypes.c_void_p,
        ]
        self._library.hipIpcGetEventHandle.restype = ctypes.c_int
        self.device_id = device_id
        self._check(self._library.hipSetDevice(device_id), "hipSetDevice")
        uuid = _HipUuid()
        self._check(
            self._library.hipDeviceGetUuid(ctypes.byref(uuid), device_id), "hipDeviceGetUuid"
        )
        self.device_uuid = bytes(uuid.bytes).hex()

    def _check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        value = self._library.hipGetErrorString(status)
        detail = value.decode("utf-8", "replace") if value else f"status {status}"
        raise ZeroCopyVlmError(f"{operation} failed: {detail}")

    def allocate(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(self._library.hipMalloc(ctypes.byref(pointer), size), "hipMalloc")
        if not pointer.value:
            raise ZeroCopyVlmError("hipMalloc returned a null pointer")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self._check(self._library.hipFree(ctypes.c_void_p(pointer)), "hipFree")

    def host_allocate(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(
            self._library.hipHostMalloc(
                ctypes.byref(pointer), size, self.HOST_MALLOC_DEFAULT
            ),
            "hipHostMalloc",
        )
        if not pointer.value:
            raise ZeroCopyVlmError("hipHostMalloc returned a null pointer")
        return int(pointer.value)

    def host_free(self, pointer: int) -> None:
        self._check(self._library.hipHostFree(ctypes.c_void_p(pointer)), "hipHostFree")

    def copy_device_to_host_async(
        self, destination: int, source: int, size: int, stream: int
    ) -> None:
        self._check(
            self._library.hipMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                size,
                self.MEMCPY_DEVICE_TO_HOST,
                ctypes.c_void_p(stream),
            ),
            "hipMemcpyAsync(Hybrid control metadata D2H)",
        )

    def copy_host_to_device_async(
        self, destination: int, source: int, size: int, stream: int
    ) -> None:
        self._check(
            self._library.hipMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                size,
                self.MEMCPY_HOST_TO_DEVICE,
                ctypes.c_void_p(stream),
            ),
            "hipMemcpyAsync(static UI resource H2D)",
        )

    def create_interprocess_event(self) -> int:
        event = ctypes.c_void_p()
        flags = self.EVENT_DISABLE_TIMING | self.EVENT_INTERPROCESS
        self._check(
            self._library.hipEventCreateWithFlags(ctypes.byref(event), flags),
            "hipEventCreateWithFlags",
        )
        if not event.value:
            raise ZeroCopyVlmError("hipEventCreateWithFlags returned a null event")
        return int(event.value)

    def create_local_event(self) -> int:
        event = ctypes.c_void_p()
        self._check(
            self._library.hipEventCreateWithFlags(
                ctypes.byref(event), self.EVENT_DISABLE_TIMING
            ),
            "hipEventCreateWithFlags(local)",
        )
        if not event.value:
            raise ZeroCopyVlmError("hipEventCreateWithFlags returned a null local event")
        return int(event.value)

    def destroy_event(self, event: int) -> None:
        self._check(self._library.hipEventDestroy(ctypes.c_void_p(event)), "hipEventDestroy")

    def record_event(self, event: int, stream: int) -> None:
        self._check(
            self._library.hipEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream)),
            "hipEventRecord",
        )

    def synchronize_event(self, event: int) -> None:
        self._check(
            self._library.hipEventSynchronize(ctypes.c_void_p(event)),
            "hipEventSynchronize(local)",
        )

    @staticmethod
    def _handle_bytes(handle: _HipIpcBytes) -> bytes:
        return ctypes.string_at(ctypes.byref(handle), ctypes.sizeof(handle))

    def memory_handle(self, pointer: int) -> bytes:
        handle = _HipIpcBytes()
        self._check(
            self._library.hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(pointer)),
            "hipIpcGetMemHandle",
        )
        return self._handle_bytes(handle)

    def event_handle(self, event: int) -> bytes:
        handle = _HipIpcBytes()
        self._check(
            self._library.hipIpcGetEventHandle(ctypes.byref(handle), ctypes.c_void_p(event)),
            "hipIpcGetEventHandle",
        )
        return self._handle_bytes(handle)


class _HipVlmKernels:
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
        self._library.vlm_camera_qwen3_vl_smart_resize.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.vlm_camera_qwen3_vl_smart_resize.restype = ctypes.c_int
        self._library.vlm_camera_qwen3_vl_preprocess_rgb8_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.vlm_camera_qwen3_vl_preprocess_rgb8_f32.restype = ctypes.c_int

    def _raise(self, operation: str, status: int) -> None:
        detail = self._library.vlm_camera_zerocopy_last_error().decode("utf-8", "replace")
        raise ZeroCopyVlmError(f"{operation} failed with status {status}: {detail}")

    def target_size(
        self, source_width: int, source_height: int, image_max_tokens: int
    ) -> tuple[int, int]:
        width = ctypes.c_int()
        height = ctypes.c_int()
        status = self._library.vlm_camera_qwen3_vl_smart_resize(
            source_width,
            source_height,
            image_max_tokens,
            ctypes.byref(width),
            ctypes.byref(height),
        )
        if status:
            self._raise("Qwen3-VL smart resize", status)
        return width.value, height.value

    def preprocess(
        self,
        *,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool,
        destination_pointer: int,
        destination_width: int,
        destination_height: int,
        image_max_tokens: int,
        source_ready_event: int,
        stream: int,
    ) -> QwenPreprocessGeometry:
        resized_width = ctypes.c_int()
        resized_height = ctypes.c_int()
        pad_left = ctypes.c_int()
        pad_top = ctypes.c_int()
        status = self._library.vlm_camera_qwen3_vl_preprocess_rgb8_f32(
            ctypes.c_void_p(source_pointer),
            source_pitch,
            source_width,
            source_height,
            int(source_is_bgr),
            ctypes.c_void_p(destination_pointer),
            destination_width,
            destination_height,
            image_max_tokens,
            ctypes.c_void_p(source_ready_event),
            ctypes.c_void_p(stream),
            ctypes.byref(resized_width),
            ctypes.byref(resized_height),
            ctypes.byref(pad_left),
            ctypes.byref(pad_top),
        )
        if status:
            self._raise("Qwen3-VL GPU preprocess", status)
        return QwenPreprocessGeometry(
            source_width=source_width,
            source_height=source_height,
            target_width=destination_width,
            target_height=destination_height,
            resized_width=resized_width.value,
            resized_height=resized_height.value,
            pad_left=pad_left.value,
            pad_top=pad_top.value,
            image_max_tokens=image_max_tokens,
        )


class _IpcSlot:
    __slots__ = (
        "base_event",
        "event",
        "event_handle",
        "generation",
        "in_use",
        "lock",
        "memory_handle",
        "pointer",
        "runtime",
        "size",
    )

    def __init__(self, runtime: _HipRuntime, size: int) -> None:
        self.runtime = runtime
        self.size = size
        self.pointer = runtime.allocate(size)
        self.base_event = 0
        self.event = 0
        try:
            self.base_event = runtime.create_local_event()
            self.event = runtime.create_interprocess_event()
            self.memory_handle = runtime.memory_handle(self.pointer)
            self.event_handle = runtime.event_handle(self.event)
        except BaseException:
            if self.event:
                runtime.destroy_event(self.event)
            if self.base_event:
                runtime.destroy_event(self.base_event)
            runtime.free(self.pointer)
            raise
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

    def close(self) -> None:
        with self.lock:
            if self.in_use:
                raise ZeroCopyVlmError("cannot close an in-use VLM IPC slot")
            base_event = self.base_event
            event = self.event
            pointer = self.pointer
            self.base_event = 0
            self.event = 0
            self.pointer = 0
        if event:
            self.runtime.destroy_event(event)
        if base_event:
            self.runtime.destroy_event(base_event)
        if pointer:
            self.runtime.free(pointer)


class HipIpcImageLease:
    """Non-copyable lease keeping one exportable HIP allocation alive through HTTP completion."""

    __slots__ = ("_final_ready", "_released", "_slot", "generation", "geometry")

    def __init__(
        self,
        slot: _IpcSlot,
        geometry: QwenPreprocessGeometry,
        *,
        final_ready: bool,
    ) -> None:
        self._slot = slot
        self._released = False
        self._final_ready = final_ready
        self.generation = slot.generation
        self.geometry = geometry

    @property
    def device_pointer(self) -> int:
        return self._slot.pointer

    @property
    def ready_event(self) -> int:
        return self._slot.event

    @property
    def base_ready_event(self) -> int:
        return self._slot.base_event

    @property
    def final_ready(self) -> bool:
        return self._final_ready

    @property
    def released(self) -> bool:
        return self._released

    def descriptor(self, *, request_id: int, device_uuid: str) -> dict[str, Any]:
        if not self._final_ready:
            raise ZeroCopyVlmError("VLM IPC image was published before final-ready")
        geometry = self.geometry
        contract = {
            "align_corners": True,
            "channels": 3,
            "dtype": "float32",
            "height": geometry.target_height,
            "image_max_tokens": geometry.image_max_tokens,
            "layout": "CHW",
            "merge_size": MERGE_SIZE,
            "mmproj_sha256": EXPECTED_MMPROJ_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "normalization": NORMALIZATION,
            "patch_size": PATCH_SIZE,
            "preprocessed": True,
            "protocol": LLAMA_IPC_PROTOCOL,
            "resize_mode": RESIZE_MODE,
            "source_height": geometry.source_height,
            "source_width": geometry.source_width,
            "width": geometry.target_width,
        }
        canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        return {
            **contract,
            "request_id": request_id,
            "device_id": self._slot.runtime.device_id,
            "device_uuid": device_uuid,
            "allocation_bytes": geometry.allocation_bytes,
            "memory_handle": base64.b64encode(self._slot.memory_handle).decode("ascii"),
            "ready_event_handle": base64.b64encode(self._slot.event_handle).decode("ascii"),
            "preprocess_fingerprint": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        }

    def mark_final_ready(self, stream: int) -> None:
        if self._released:
            raise ZeroCopyVlmError("cannot finalize a released VLM IPC image")
        if self._final_ready:
            raise ZeroCopyVlmError("VLM IPC image is already final-ready")
        self._slot.runtime.record_event(self._slot.event, stream)
        self._final_ready = True

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._slot.release(self.generation)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __copy__(self) -> None:
        raise TypeError("HipIpcImageLease is non-copyable")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("HipIpcImageLease is non-copyable")

    def __del__(self) -> None:
        self.release()


class ZeroCopyVlmPreprocessor:
    """Fused GPU preprocess plus a fixed pool of exportable HIP IPC buffers."""

    def __init__(
        self,
        *,
        kernel_library: str | Path,
        source_width: int,
        source_height: int,
        image_max_tokens: int = 512,
        pool_size: int = 2,
        device_id: int = 0,
    ) -> None:
        if pool_size < 2:
            raise ValueError("VLM IPC pool_size must be at least 2")
        if image_max_tokens != 512:
            raise ValueError("the locked Qwen3-VL IPC v2 contract requires image_max_tokens=512")
        self.runtime = _HipRuntime(device_id)
        self.kernels = _HipVlmKernels(Path(kernel_library))
        target_width, target_height = self.kernels.target_size(
            source_width, source_height, image_max_tokens
        )
        self.source_width = source_width
        self.source_height = source_height
        self.image_max_tokens = image_max_tokens
        self.target_width = target_width
        self.target_height = target_height
        self.allocation_bytes = target_width * target_height * 3 * 4
        self._slots: list[_IpcSlot] = []
        try:
            for _ in range(pool_size):
                self._slots.append(_IpcSlot(self.runtime, self.allocation_bytes))
        except BaseException:
            for slot in self._slots:
                slot.close()
            raise
        self._closed = False
        self._prepared = 0
        self._dropped_no_slot = 0

    def prepare_rgb8_pointer(
        self,
        *,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool = False,
        source_ready_event: int = 0,
        stream: int = 0,
        defer_final_ready: bool = False,
    ) -> HipIpcImageLease | None:
        if self._closed:
            raise ZeroCopyVlmError("VLM preprocessor is closed")
        if source_width != self.source_width or source_height != self.source_height:
            raise ZeroCopyVlmError(
                "VLM source dimensions changed; fixed IPC allocations cannot be silently replaced"
            )
        if source_pointer <= 0:
            raise ValueError("source_pointer must be a non-zero GPU address")
        slot = next((candidate for candidate in self._slots if candidate.try_acquire()), None)
        if slot is None:
            self._dropped_no_slot += 1
            return None
        try:
            geometry = self.kernels.preprocess(
                source_pointer=source_pointer,
                source_pitch=source_pitch,
                source_width=source_width,
                source_height=source_height,
                source_is_bgr=source_is_bgr,
                destination_pointer=slot.pointer,
                destination_width=self.target_width,
                destination_height=self.target_height,
                image_max_tokens=self.image_max_tokens,
                source_ready_event=source_ready_event,
                stream=stream,
            )
            self.runtime.record_event(slot.base_event, stream)
            self._prepared += 1
            lease = HipIpcImageLease(slot, geometry, final_ready=False)
            if not defer_final_ready:
                lease.mark_final_ready(stream)
            return lease
        except BaseException:
            slot.release(slot.generation)
            raise

    def runtime_info(self) -> dict[str, Any]:
        return {
            "protocol": LLAMA_IPC_PROTOCOL,
            "device_id": self.runtime.device_id,
            "device_uuid": self.runtime.device_uuid,
            "source_shape": [self.source_height, self.source_width, 3],
            "target_shape": [3, self.target_height, self.target_width],
            "dtype": "float32",
            "layout": "CHW",
            "normalization": NORMALIZATION,
            "resize_mode": RESIZE_MODE,
            "allocation_bytes_per_slot": self.allocation_bytes,
            "pool_size": len(self._slots),
            "device_pointers": [slot.pointer for slot in self._slots],
            "base_ready_events": [slot.base_event for slot in self._slots],
            "final_ready_events": [slot.event for slot in self._slots],
            "prepared": self._prepared,
            "dropped_no_slot": self._dropped_no_slot,
            "production_image_h2d_bytes": 0,
            "production_image_d2h_bytes": 0,
            "uses_global_device_synchronize": False,
            "two_stage_ready_supported": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        for slot in self._slots:
            slot.close()
        self._closed = True


@dataclass(frozen=True, slots=True)
class LlamaCppIpcConfig(LlamaCppConfig):
    server_path: str = ".build/llama-vlm-zerocopy-gfx1151/bin/llama-server"
    log_path: str = "output/realtime/llama-server-zerocopy.log"
    max_tokens: int = 32
    hip_stream_priority: int = 1
    prompt: str = (
        "Describe the main objects and action in this camera image. "
        "Reply with one English sentence of at most 16 words; do not speculate."
    )
    kernel_library: str = ".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0"
    llama_repository: str = "third_party/llama.cpp-vlm-zerocopy"
    llama_patch: str = "patches/llama-cpp-hip-ipc-device-input.patch"
    llama_patch_lock: str = "native-lock/llama-cpp-hip-ipc-device-input.sha256"
    ipc_pool_size: int = 2

    def __post_init__(self) -> None:
        LlamaCppConfig.__post_init__(self)
        if self.image_max_tokens != 512:
            raise ValueError("llama HIP IPC v2 currently requires image_max_tokens=512")
        if self.ipc_pool_size < 2:
            raise ValueError("llama HIP IPC pool must contain at least two slots")


class ZeroCopyLlamaCppVlm(LlamaCppVlm):
    """GPU-only llama.cpp service accepting final Qwen3-VL tensors via HIP IPC v2."""

    mode = "llamacpp-ipc"

    def __init__(self, config: LlamaCppIpcConfig, *, workspace: str | Path) -> None:
        super().__init__(config, workspace=workspace)
        self.config = config
        self.kernel_library = self._resolve_config_path(config.kernel_library)
        self.llama_repository = self._resolve_config_path(config.llama_repository)
        self.llama_patch = self._resolve_config_path(config.llama_patch)
        self.llama_patch_lock = self._resolve_config_path(config.llama_patch_lock)
        self._preprocessor: ZeroCopyVlmPreprocessor | None = None
        self._media_marker = ""
        self._request_sequence = 0
        self._successful_requests = 0
        self._request_latencies_ms: list[float] = []
        self._performance_lock = threading.Lock()
        self._last_request_performance: dict[str, float | int | None] = {
            "prompt_tokens": None,
            "prompt_tokens_per_second": None,
            "generated_tokens": None,
            "generated_tokens_per_second": None,
        }

    def _resolve_config_path(self, value: str) -> Path:
        path = Path(value)
        return (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()

    def _preflight(self) -> None:
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            raise VlmStartupError("HSA_OVERRIDE_GFX_VERSION is set; refusing architecture spoofing")
        for label, path in (
            ("llama-server", self.server_path),
            ("Qwen3-VL model", self.model_path),
            ("multimodal projector", self.mmproj_path),
            ("zero-copy HIP kernels", self.kernel_library),
            ("llama.cpp source", self.llama_repository),
            ("llama.cpp device-input patch", self.llama_patch),
            ("llama.cpp patch lock", self.llama_patch_lock),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        if _sha256(self.model_path) != EXPECTED_MODEL_SHA256:
            raise VlmStartupError("Qwen3-VL model SHA-256 mismatch")
        if _sha256(self.mmproj_path) != EXPECTED_MMPROJ_SHA256:
            raise VlmStartupError("mmproj SHA-256 mismatch")
        revision = subprocess.run(
            ["git", "-C", str(self.llama_repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != EXPECTED_LLAMA_IPC_COMMIT:
            raise VlmStartupError(f"unexpected llama.cpp device-input commit: {revision}")
        patch_lock = self.llama_patch_lock.read_text(encoding="ascii").strip()
        patch_digest = _sha256(self.llama_patch)
        source_diff = subprocess.run(
            ["git", "-C", str(self.llama_repository), "diff", "--binary"],
            check=True,
            capture_output=True,
        ).stdout
        source_digest = hashlib.sha256(source_diff).hexdigest()
        if not patch_lock or patch_digest != patch_lock or source_digest != patch_lock:
            raise VlmStartupError(
                "llama.cpp device-input patch/source mismatch: "
                f"lock={patch_lock}, patch={patch_digest}, source={source_digest}"
            )
        libraries = sorted(self.server_path.parent.glob("libggml-hip.so.*.*.*"))
        if len(libraries) != 1:
            raise VlmStartupError(f"expected one versioned libggml-hip, found {libraries}")
        cmake_cache = self.server_path.parent.parent / "CMakeCache.txt"
        if not cmake_cache.is_file():
            raise VlmStartupError(f"llama.cpp CMake cache is missing: {cmake_cache}")
        cmake_entries = set(cmake_cache.read_text(encoding="utf-8").splitlines())
        if ROCWMMA_CMAKE_ENTRY not in cmake_entries:
            raise VlmStartupError(
                "llama.cpp was not built with GGML_HIP_ROCWMMA_FATTN=ON"
            )
        code_objects = subprocess.run(
            ["roc-obj-ls", str(libraries[0])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        architectures = sorted(
            {
                fields[1]
                for line in code_objects.splitlines()
                if len(fields := line.split()) >= 2 and fields[1].startswith("hipv")
            }
        )
        if architectures != [EXPECTED_HIP_CODE_OBJECT]:
            raise VlmStartupError(f"unexpected llama.cpp HIP code objects: {architectures}")
        server_impl = self.server_path.parent / "libllama-server-impl.so"
        mtmd = next(iter(sorted(self.server_path.parent.glob("libmtmd.so.*.*.*"))), None)
        llama_lib = next(iter(sorted(self.server_path.parent.glob("libllama.so.*.*.*"))), None)
        if mtmd is None:
            raise VlmStartupError("versioned libmtmd is missing beside llama-server")
        if llama_lib is None:
            raise VlmStartupError("versioned libllama is missing beside llama-server")
        for path, marker in (
            (server_impl, LLAMA_IPC_PROTOCOL),
            (mtmd, "HIP IPC v2 D2D complete"),
            (mtmd, "HIP device embedding ready"),
            (llama_lib, "HIP embedding D2D complete"),
        ):
            strings = subprocess.run(
                ["strings", str(path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if marker not in strings:
                raise VlmStartupError(f"required device-input marker missing from {path.name}")
        hip_library = libraries[0]
        hip_strings = subprocess.run(
            ["strings", str(hip_library)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "GGML_HIP_STREAM_PRIORITY" not in hip_strings:
            raise VlmStartupError(
                f"required low-priority HIP stream support missing from {hip_library.name}"
            )
        if ROCWMMA_KERNEL_MARKER not in hip_strings:
            raise VlmStartupError(
                f"rocWMMA FlashAttention kernel marker missing from {hip_library.name}"
            )

    def start(self) -> Self:
        super().start()
        status, body = _request_json(
            f"{self.base_url}/props",
            timeout=5.0,
            api_key=self._api_key,
        )
        marker = body.get("media_marker") if status == 200 else None
        if not isinstance(marker, str) or not marker:
            self.stop()
            raise VlmStartupError("llama-server did not publish a multimodal media marker")
        self._media_marker = marker
        return self

    def _ensure_preprocessor(
        self, source_width: int, source_height: int
    ) -> ZeroCopyVlmPreprocessor:
        current = self._preprocessor
        if current is None:
            current = ZeroCopyVlmPreprocessor(
                kernel_library=self.kernel_library,
                source_width=source_width,
                source_height=source_height,
                image_max_tokens=self.config.image_max_tokens,
                pool_size=self.config.ipc_pool_size,
            )
            self._preprocessor = current
        elif current.source_width != source_width or current.source_height != source_height:
            raise ZeroCopyVlmError(
                "camera dimensions changed after the fixed VLM IPC pool was allocated"
            )
        return current

    def caption_gpu_pointer(
        self,
        *,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool = False,
        source_ready_event: int = 0,
        stream: int = 0,
    ) -> str:
        lease = self.prepare_gpu_pointer(
            source_pointer=source_pointer,
            source_pitch=source_pitch,
            source_width=source_width,
            source_height=source_height,
            source_is_bgr=source_is_bgr,
            source_ready_event=source_ready_event,
            stream=stream,
        )
        if lease is None:
            raise VlmRequestError("all fixed VLM IPC buffers are busy; newest request dropped")
        return self.caption_prepared(lease)

    def prepare_gpu_pointer(
        self,
        *,
        source_pointer: int,
        source_pitch: int,
        source_width: int,
        source_height: int,
        source_is_bgr: bool = False,
        source_ready_event: int = 0,
        stream: int = 0,
        defer_final_ready: bool = False,
    ) -> HipIpcImageLease | None:
        """Snapshot a GPU frame into a bounded IPC slot without issuing HTTP."""
        process = self._process
        if process is None or process.poll() is not None:
            raise VlmRequestError("zero-copy llama-server is not running")
        preprocessor = self._ensure_preprocessor(source_width, source_height)
        return preprocessor.prepare_rgb8_pointer(
            source_pointer=source_pointer,
            source_pitch=source_pitch,
            source_width=source_width,
            source_height=source_height,
            source_is_bgr=source_is_bgr,
            source_ready_event=source_ready_event,
            stream=stream,
            defer_final_ready=defer_final_ready,
        )

    def caption_prepared(
        self,
        lease: HipIpcImageLease,
        *,
        prompt: str | None = None,
        stop_at_first_sentence: bool = False,
        constrain_english_sentence: bool = True,
    ) -> str:
        """Consume a prepared IPC lease; ownership transfers to this call."""
        process = self._process
        if process is None or process.poll() is not None:
            lease.release()
            raise VlmRequestError("zero-copy llama-server is not running")
        preprocessor = self._preprocessor
        if preprocessor is None or lease._slot not in preprocessor._slots:
            lease.release()
            raise VlmRequestError("prepared image lease does not belong to this VLM engine")
        if lease.released:
            raise VlmRequestError("prepared image lease was already released")
        if not lease.final_ready:
            lease.release()
            raise VlmRequestError("prepared image lease is not final-ready")
        request_prompt = self.config.prompt if prompt is None else prompt.strip()
        if not request_prompt:
            lease.release()
            raise VlmRequestError("VLM request prompt must not be empty")
        if len(request_prompt) > 2048:
            lease.release()
            raise VlmRequestError("VLM request prompt exceeds the bounded 2048-character contract")
        with lease, self._request_lock:
            self._request_sequence += 1
            request_id = self._request_sequence
            descriptor = lease.descriptor(
                request_id=request_id,
                device_uuid=preprocessor.runtime.device_uuid,
            )
            # `/completions` exposes the custom multimodal_data control object directly,
            # so apply the locked Qwen3 chat wrapper here. The media marker itself is
            # replaced by mtmd with the model-specific vision token sequence.
            prompt = (
                "<|im_start|>user\n"
                + self._media_marker
                + request_prompt
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            payload = {
                "prompt": {
                    "prompt_string": prompt,
                    "multimodal_data": [descriptor],
                },
                "n_predict": self.config.max_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "cache_prompt": False,
            }
            if stop_at_first_sentence:
                payload["stop"] = [".", "\n", "。", "！", "？"]
                if constrain_english_sentence:
                    payload["grammar"] = SINGLE_SENTENCE_GBNF
            started = time.perf_counter()
            try:
                status, response = _request_json(
                    f"{self.base_url}/completions",
                    timeout=self.config.timeout_seconds,
                    api_key=self._api_key,
                    payload=payload,
                )
            except (OSError, TimeoutError) as exc:
                raise VlmRequestError(f"local GPU IPC VLM request failed: {exc}") from exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._request_latencies_ms.append(latency_ms)
            if status != 200:
                raise VlmRequestError(f"local GPU IPC VLM returned HTTP {status}: {response}")
            caption = str(response.get("content", "")).strip()
            if not caption:
                raise VlmRequestError(f"local GPU IPC VLM returned no content: {response}")
            timings = response.get("timings", {})
            if not isinstance(timings, dict):
                timings = {}

            def finite_float(key: str) -> float | None:
                value = timings.get(key)
                if isinstance(value, bool) or not isinstance(value, int | float):
                    return None
                converted = float(value)
                return converted if math.isfinite(converted) and converted >= 0.0 else None

            def nonnegative_int(key: str) -> int | None:
                value = timings.get(key)
                if isinstance(value, bool) or not isinstance(value, int | float):
                    return None
                converted = int(value)
                return converted if converted >= 0 else None

            generated_tokens = nonnegative_int("predicted_n")
            generated_tokens_per_second = finite_float("predicted_per_second")
            if generated_tokens_per_second is None:
                predicted_ms = finite_float("predicted_ms")
                if generated_tokens is not None and predicted_ms is not None and predicted_ms > 0:
                    generated_tokens_per_second = generated_tokens * 1000.0 / predicted_ms
            with self._performance_lock:
                self._last_request_performance = {
                    "prompt_tokens": nonnegative_int("prompt_n"),
                    "prompt_tokens_per_second": finite_float("prompt_per_second"),
                    "generated_tokens": generated_tokens,
                    "generated_tokens_per_second": generated_tokens_per_second,
                }
            self._successful_requests += 1
            normalized_caption = " ".join(caption.split())
            if stop_at_first_sentence and normalized_caption[-1] not in ".!?。！？":
                normalized_caption += "."
            return normalized_caption

    def last_request_performance(self) -> dict[str, float | int | None]:
        with self._performance_lock:
            return dict(self._last_request_performance)

    def runtime_info(self) -> dict[str, Any]:
        info = super().runtime_info()
        info.update(
            {
                "mode": self.mode,
                "llama_cpp_commit": EXPECTED_LLAMA_IPC_COMMIT,
                "llama_cpp_repository": str(self.llama_repository),
                "llama_cpp_patch": str(self.llama_patch),
                "llama_cpp_patch_sha256": self.llama_patch_lock.read_text(encoding="ascii").strip(),
                "ggml_hip_rocwmma_fattn": True,
                "rocwmma_kernel_marker": ROCWMMA_KERNEL_MARKER,
                "ipc_protocol": LLAMA_IPC_PROTOCOL,
                "successful_requests": self._successful_requests,
                "request_latencies_ms": list(self._request_latencies_ms),
                "last_request_performance": self.last_request_performance(),
                "image_bytes_in_http_requests": 0,
                "production_image_h2d_bytes": 0,
                "production_image_d2h_bytes": 0,
                "split_prepare_caption_api": True,
                "two_stage_ipc_ready_supported": True,
                "per_request_prompt_supported": True,
                "hip_stream_priority_policy": "low-priority-vlm-normal-priority-yolo",
                "preprocessor": (
                    self._preprocessor.runtime_info() if self._preprocessor is not None else None
                ),
            }
        )
        return info

    def stop(self) -> None:
        super().stop()
        if self._preprocessor is not None:
            self._preprocessor.close()
            self._preprocessor = None
