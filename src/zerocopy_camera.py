"""Strict V4L2 DMA-BUF camera whose Python boundary contains no pixels."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self

EXPECTED_ISP4_SOURCE_COMMIT = "32bed5152a6e284e02c0f803772bc48960f06e32"
EXPECTED_ISP4_PATCH_SHA256 = (
    "dcdb1fb2f3e1c611ab56fc2b8b51ca9e7abd965fbf440b2b128412d25a20b16b"
)
EXPECTED_PATCHED_MODULE_SRCVERSION = "CA94CB23673E748A9ECC5F2"
EXPECTED_PATCHED_MODULE_SHA256 = (
    "5a171b50ea6a0bd77c94de8221461dd7c666f5a328dcfe1bd7b1315f044e95f5"
)


class ZeroCopyCameraError(RuntimeError):
    """The strict camera contract or native capture operation failed."""


class ZeroCopyCameraTimeout(ZeroCopyCameraError):
    """No camera frame arrived before the bounded poll timeout."""


class _CaptureOptions(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_char_p),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("fps", ctypes.c_uint32),
        ("camera_buffers", ctypes.c_uint32),
        ("clean_rgb_buffers", ctypes.c_uint32),
        ("web_stream_buffers", ctypes.c_uint32),
        ("hip_device", ctypes.c_int),
        ("horizontal_flip", ctypes.c_uint32),
    ]


class _CaptureInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("fourcc", ctypes.c_uint32),
        ("bytes_per_line", ctypes.c_uint32),
        ("size_image", ctypes.c_uint32),
        ("camera_allocation_bytes", ctypes.c_size_t),
        ("web_stream_fourcc", ctypes.c_uint32),
        ("web_stream_bytes_per_line", ctypes.c_uint32),
        ("web_stream_size_image", ctypes.c_uint32),
        ("fps_numerator", ctypes.c_uint32),
        ("fps_denominator", ctypes.c_uint32),
        ("camera_buffers", ctypes.c_uint32),
        ("clean_rgb_buffers", ctypes.c_uint32),
        ("web_stream_buffers", ctypes.c_uint32),
        ("hip_device", ctypes.c_int),
        ("horizontal_flip", ctypes.c_uint32),
        ("frames_acquired", ctypes.c_uint64),
        ("frames_requeued", ctypes.c_uint64),
        ("frames_dropped_no_clean_slot", ctypes.c_uint64),
        ("web_stream_frames_converted", ctypes.c_uint64),
        ("web_stream_frames_released", ctypes.c_uint64),
        ("web_stream_frames_dropped_no_slot", ctypes.c_uint64),
        ("web_stream_gpu_bytes_written", ctypes.c_uint64),
        ("driver", ctypes.c_char_p),
        ("card", ctypes.c_char_p),
        ("memory_path", ctypes.c_char_p),
        ("color_conversion", ctypes.c_char_p),
    ]


class _NativeFrame(ctypes.Structure):
    _fields_ = [
        ("frame_id", ctypes.c_uint64),
        ("captured_monotonic_ns", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("sequence", ctypes.c_uint32),
        ("slot_index", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bytes_used", ctypes.c_uint32),
        ("rgb_pitch", ctypes.c_size_t),
        ("rgb_allocation_bytes", ctypes.c_size_t),
        ("rgb_device_pointer", ctypes.c_void_p),
        ("ready_event", ctypes.c_void_p),
        ("web_stream_generation", ctypes.c_uint64),
        ("web_stream_slot_index", ctypes.c_uint32),
        ("web_stream_dmabuf_fd", ctypes.c_int32),
        ("web_stream_fourcc", ctypes.c_uint32),
        ("web_stream_device_pointer", ctypes.c_void_p),
        ("web_stream_pitch", ctypes.c_size_t),
        ("web_stream_size_bytes", ctypes.c_size_t),
        ("web_stream_allocation_bytes", ctypes.c_size_t),
    ]


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", "replace") if value else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_component_preflight(
    *,
    workspace: str | Path,
    library: str | Path,
    require_active_patched_driver: bool,
) -> dict[str, Any]:
    """Validate all immutable camera artifacts before opening a V4L2 fd."""
    root = Path(workspace).resolve()
    library_path = Path(library)
    library_path = (
        (root / library_path).resolve() if not library_path.is_absolute() else library_path.resolve()
    )
    patch_path = root / "patches/linux-oem-6.17-amd-isp4-dmabuf-import.patch"
    source_dir = root / "third_party/linux-oem-6.17-amd-isp4"
    expected_opencv = root / ".local/opencv5-gfx1151"
    for label, path in (
        ("GPU camera library", library_path),
        ("ISP4 patch", patch_path),
        ("ISP4 source", source_dir),
        ("repo-local OpenCV 5 HIP", expected_opencv),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if _sha256(patch_path) != EXPECTED_ISP4_PATCH_SHA256:
        raise ZeroCopyCameraError("ISP4 DMA-BUF patch SHA-256 mismatch")
    source_commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != EXPECTED_ISP4_SOURCE_COMMIT:
        raise ZeroCopyCameraError(f"unexpected ISP4 source commit: {source_commit}")
    source_diff = subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "diff",
            "--binary",
            "--",
            "drivers/media/platform/amd/isp4/isp4_interface.c",
            "drivers/media/platform/amd/isp4/isp4_video.c",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(source_diff).hexdigest() != EXPECTED_ISP4_PATCH_SHA256:
        raise ZeroCopyCameraError("ISP4 source tree does not exactly match the locked patch")

    code_objects = subprocess.run(
        ["roc-obj-ls", str(library_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "hipv4-amdgcn-amd-amdhsa--gfx1151" not in code_objects:
        raise ZeroCopyCameraError("GPU camera library has no gfx1151 code object")

    import cv2

    cv2_path = Path(cv2.__file__).resolve()
    if not str(cv2_path).startswith(str(expected_opencv) + os.sep):
        raise ZeroCopyCameraError(f"cv2 resolved outside repo-local OpenCV 5 HIP: {cv2_path}")
    if not cv2.__version__.startswith("5.") or cv2.cuda.getCudaEnabledDeviceCount() != 1:
        raise ZeroCopyCameraError("repo-local OpenCV 5 HIP did not expose exactly one GPU")
    if not hasattr(cv2.cuda_GpuMat, "fromDevicePointer"):
        raise ZeroCopyCameraError("OpenCV 5 HIP lacks GpuMat.fromDevicePointer")

    loaded_srcversion_path = Path("/sys/module/amd_capture/srcversion")
    loaded_srcversion = (
        loaded_srcversion_path.read_text().strip() if loaded_srcversion_path.is_file() else None
    )
    active_driver_patched = loaded_srcversion == EXPECTED_PATCHED_MODULE_SRCVERSION
    if require_active_patched_driver and not active_driver_patched:
        raise ZeroCopyCameraError(
            "refusing to open the camera: the locked patched amd_capture module is not active "
            f"(loaded srcversion={loaded_srcversion!r}, "
            f"expected={EXPECTED_PATCHED_MODULE_SRCVERSION!r})"
        )
    resolved_module_result = subprocess.run(
        ["modinfo", "-n", "amd_capture"],
        check=False,
        capture_output=True,
        text=True,
    )
    resolved_module = (
        Path(resolved_module_result.stdout.strip()).resolve()
        if resolved_module_result.returncode == 0
        and resolved_module_result.stdout.strip()
        else None
    )
    expected_persistent_module = (
        Path("/lib/modules")
        / os.uname().release
        / "updates/vlm-camera-pipeline/amd_capture.ko"
    ).resolve()
    persistent_module_installed = bool(
        resolved_module
        and resolved_module == expected_persistent_module
        and resolved_module.is_file()
        and _sha256(resolved_module) == EXPECTED_PATCHED_MODULE_SHA256
    )
    return {
        "library": str(library_path),
        "library_sha256": _sha256(library_path),
        "gpu_code_object": "gfx1151",
        "opencv_version": cv2.__version__,
        "opencv_path": str(cv2_path),
        "opencv_pointer_factory": True,
        "isp4_source_commit": source_commit,
        "isp4_patch_sha256": EXPECTED_ISP4_PATCH_SHA256,
        "loaded_module_srcversion": loaded_srcversion,
        "expected_patched_module_srcversion": EXPECTED_PATCHED_MODULE_SRCVERSION,
        "active_driver_patched": active_driver_patched,
        "module_resolution": str(resolved_module) if resolved_module else None,
        "persistent_module_expected_path": str(expected_persistent_module),
        "persistent_module_installed": persistent_module_installed,
    }


class GpuYuyvFrameLease:
    """Non-copyable lease over one encoder-facing YUYV GPU DMA-BUF slot."""

    __slots__ = (
        "_owner",
        "_released",
        "allocation_bytes",
        "captured_monotonic_ns",
        "device_pointer",
        "dmabuf_fd",
        "fourcc",
        "frame_id",
        "generation",
        "height",
        "pitch",
        "size_bytes",
        "slot_index",
        "width",
    )

    def __init__(self, owner: StrictGpuCamera, native: _NativeFrame) -> None:
        self._owner = owner
        self._released = False
        self.frame_id = int(native.frame_id)
        self.captured_monotonic_ns = int(native.captured_monotonic_ns)
        self.generation = int(native.web_stream_generation)
        self.slot_index = int(native.web_stream_slot_index)
        self.dmabuf_fd = int(native.web_stream_dmabuf_fd)
        self.device_pointer = int(native.web_stream_device_pointer or 0)
        self.fourcc = int(native.web_stream_fourcc).to_bytes(4, "little").decode("ascii")
        self.pitch = int(native.web_stream_pitch)
        self.size_bytes = int(native.web_stream_size_bytes)
        self.allocation_bytes = int(native.web_stream_allocation_bytes)
        self.width = int(native.width)
        self.height = int(native.height)
        if (
            self.generation <= 0
            or self.captured_monotonic_ns <= 0
            or self.dmabuf_fd < 0
            or self.device_pointer <= 0
            or self.fourcc != "YUYV"
            or self.pitch < self.width * 2
            or self.size_bytes < self.pitch * self.height
            or self.allocation_bytes < self.size_bytes
        ):
            raise ZeroCopyCameraError("native camera returned an invalid web YUYV DMA-BUF")

    @property
    def released(self) -> bool:
        return self._released

    def duplicate_fd(self) -> int:
        if self._released:
            raise ZeroCopyCameraError("web NV12 frame lease is already released")
        return os.dup(self.dmabuf_fd)

    def release(self) -> None:
        if self._released:
            return
        self._owner._release_web(self)
        self._released = True

    def __copy__(self) -> None:
        raise TypeError("GpuYuyvFrameLease is non-copyable")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("GpuYuyvFrameLease is non-copyable")

    def __del__(self) -> None:
        if not getattr(self, "_released", True):
            try:
                self.release()
            except Exception:  # noqa: BLE001, S110 - destructor cannot report failures
                pass


class GpuFrameLease:
    """Non-copyable lease over one camera-owned clean RGB8 GPU slot."""

    __slots__ = (
        "_opencv_view",
        "_owner",
        "_released",
        "_web_stream",
        "bytes_used",
        "captured_monotonic_ns",
        "frame_id",
        "generation",
        "height",
        "ready_event",
        "rgb_allocation_bytes",
        "rgb_device_pointer",
        "rgb_pitch",
        "sequence",
        "slot_index",
        "width",
    )

    def __init__(self, owner: StrictGpuCamera, native: _NativeFrame, opencv_view: Any) -> None:
        self._owner = owner
        self._released = False
        self._opencv_view = opencv_view
        self.frame_id = int(native.frame_id)
        self.captured_monotonic_ns = int(native.captured_monotonic_ns)
        self.generation = int(native.generation)
        self.sequence = int(native.sequence)
        self.slot_index = int(native.slot_index)
        self.width = int(native.width)
        self.height = int(native.height)
        self.bytes_used = int(native.bytes_used)
        self.rgb_pitch = int(native.rgb_pitch)
        self.rgb_allocation_bytes = int(native.rgb_allocation_bytes)
        self.rgb_device_pointer = int(native.rgb_device_pointer or 0)
        self.ready_event = int(native.ready_event or 0)
        self._web_stream = (
            GpuYuyvFrameLease(owner, native) if int(native.web_stream_dmabuf_fd) >= 0 else None
        )

    @property
    def opencv_view(self) -> Any:
        if self._released:
            raise ZeroCopyCameraError("camera frame lease is already released")
        return self._opencv_view

    def detach_web_stream(self) -> GpuYuyvFrameLease | None:
        """Transfer ownership of the optional encoder-facing YUYV lease."""
        if self._released:
            raise ZeroCopyCameraError("camera frame lease is already released")
        lease = self._web_stream
        self._web_stream = None
        return lease

    def release(self, *, consumer_done_event: int = 0) -> None:
        if self._released:
            return
        if self._web_stream is not None:
            self._web_stream.release()
            self._web_stream = None
        self._owner._release(self, consumer_done_event=consumer_done_event)
        self._released = True
        self._opencv_view = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.release()

    def __copy__(self) -> None:
        raise TypeError("GpuFrameLease is non-copyable")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise TypeError("GpuFrameLease is non-copyable")

    def __del__(self) -> None:
        if not getattr(self, "_released", True):
            try:
                self.release()
            except Exception:  # noqa: BLE001, S110 - destructor cannot report failures
                pass


class StrictGpuCamera:
    """HIP-owned camera ring with V4L2 DMA-BUF and fixed clean RGB leases."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        library: str | Path,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        camera_buffers: int = 4,
        clean_rgb_buffers: int = 3,
        web_stream_buffers: int = 0,
        hip_device: int = 0,
        horizontal_flip: bool = False,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.device = device
        self.horizontal_flip = horizontal_flip
        self._preflight = camera_component_preflight(
            workspace=self.workspace,
            library=library,
            require_active_patched_driver=True,
        )
        self.library_path = Path(self._preflight["library"])
        self._library = ctypes.CDLL(str(self.library_path))
        self._bind_api()
        self._handle = ctypes.c_void_p()
        self._creator_thread = threading.get_ident()
        self._active: dict[int, int] = {}
        self._active_web: dict[int, int] = {}
        options = _CaptureOptions(
            device.encode("utf-8"),
            width,
            height,
            fps,
            camera_buffers,
            clean_rgb_buffers,
            web_stream_buffers,
            hip_device,
            int(horizontal_flip),
        )
        status = self._library.vlm_camera_gpu_capture_create(
            ctypes.byref(options), ctypes.byref(self._handle)
        )
        if status != 0 or not self._handle.value:
            self._raise("create strict GPU camera", status)

    def _bind_api(self) -> None:
        library = self._library
        library.vlm_camera_gpu_capture_last_error.argtypes = []
        library.vlm_camera_gpu_capture_last_error.restype = ctypes.c_char_p
        library.vlm_camera_gpu_capture_create.argtypes = [
            ctypes.POINTER(_CaptureOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.vlm_camera_gpu_capture_create.restype = ctypes.c_int
        library.vlm_camera_gpu_capture_acquire_rgb8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_NativeFrame),
        ]
        library.vlm_camera_gpu_capture_acquire_rgb8.restype = ctypes.c_int
        library.vlm_camera_gpu_capture_release_rgb8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        library.vlm_camera_gpu_capture_release_rgb8.restype = ctypes.c_int
        library.vlm_camera_gpu_capture_release_web_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]
        library.vlm_camera_gpu_capture_release_web_stream.restype = ctypes.c_int
        library.vlm_camera_gpu_capture_get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CaptureInfo),
        ]
        library.vlm_camera_gpu_capture_get_info.restype = ctypes.c_int
        library.vlm_camera_gpu_capture_destroy.argtypes = [ctypes.c_void_p]
        library.vlm_camera_gpu_capture_destroy.restype = None

    def _require_creator_thread(self) -> None:
        if threading.get_ident() != self._creator_thread:
            raise ZeroCopyCameraError("V4L2 capture operations must stay on the creator thread")

    def _raise(self, operation: str, status: int) -> None:
        detail = _decode(self._library.vlm_camera_gpu_capture_last_error())
        raise ZeroCopyCameraError(f"{operation} failed with status {status}: {detail}")

    def acquire(self, *, timeout_ms: int = 2000) -> GpuFrameLease | None:
        self._require_creator_thread()
        if not self._handle.value:
            raise ZeroCopyCameraError("camera is closed")
        native = _NativeFrame()
        status = self._library.vlm_camera_gpu_capture_acquire_rgb8(
            self._handle, timeout_ms, ctypes.byref(native)
        )
        if status == 1:
            return None
        if status == 2:
            raise ZeroCopyCameraTimeout(
                _decode(self._library.vlm_camera_gpu_capture_last_error())
            )
        if status != 0:
            self._raise("acquire strict GPU frame", status)
        pointer = int(native.rgb_device_pointer or 0)
        if pointer <= 0:
            raise ZeroCopyCameraError("native camera returned a null RGB GPU pointer")

        import cv2

        try:
            view = cv2.cuda_GpuMat.fromDevicePointer(
                pointer,
                int(native.height),
                int(native.width),
                cv2.CV_8UC3,
                int(native.rgb_pitch),
            )
            if int(view.cudaPtr()) != pointer:
                raise ZeroCopyCameraError("OpenCV GpuMat did not alias the native RGB pointer")
        except BaseException:
            if int(native.web_stream_dmabuf_fd) >= 0:
                self._library.vlm_camera_gpu_capture_release_web_stream(
                    self._handle,
                    native.web_stream_slot_index,
                    native.web_stream_generation,
                )
            self._library.vlm_camera_gpu_capture_release_rgb8(
                self._handle,
                native.slot_index,
                native.generation,
                ctypes.c_void_p(),
            )
            raise
        lease = GpuFrameLease(self, native, view)
        self._active[lease.slot_index] = lease.generation
        if lease._web_stream is not None:
            self._active_web[lease._web_stream.slot_index] = lease._web_stream.generation
        return lease

    def _release(self, lease: GpuFrameLease, *, consumer_done_event: int) -> None:
        self._require_creator_thread()
        if self._active.get(lease.slot_index) != lease.generation:
            raise ZeroCopyCameraError("stale or duplicate Python camera frame lease")
        status = self._library.vlm_camera_gpu_capture_release_rgb8(
            self._handle,
            lease.slot_index,
            lease.generation,
            ctypes.c_void_p(consumer_done_event),
        )
        if status:
            self._raise("release strict GPU frame", status)
        del self._active[lease.slot_index]

    def _release_web(self, lease: GpuYuyvFrameLease) -> None:
        self._require_creator_thread()
        if self._active_web.get(lease.slot_index) != lease.generation:
            raise ZeroCopyCameraError("stale or duplicate Python web stream frame lease")
        status = self._library.vlm_camera_gpu_capture_release_web_stream(
            self._handle,
            lease.slot_index,
            lease.generation,
        )
        if status:
            self._raise("release strict GPU web stream frame", status)
        del self._active_web[lease.slot_index]

    def runtime_info(self) -> dict[str, Any]:
        if not self._handle.value:
            raise ZeroCopyCameraError("camera is closed")
        native = _CaptureInfo()
        status = self._library.vlm_camera_gpu_capture_get_info(
            self._handle, ctypes.byref(native)
        )
        if status:
            self._raise("read strict GPU camera info", status)
        fourcc = int(native.fourcc).to_bytes(4, "little").decode("ascii", "replace")
        return {
            **self._preflight,
            "device": self.device,
            "driver": _decode(native.driver),
            "card": _decode(native.card),
            "format": fourcc,
            "width": int(native.width),
            "height": int(native.height),
            "bytes_per_line": int(native.bytes_per_line),
            "size_image": int(native.size_image),
            "camera_allocation_bytes": int(native.camera_allocation_bytes),
            "web_stream_format": int(native.web_stream_fourcc)
            .to_bytes(4, "little")
            .decode("ascii", "replace"),
            "web_stream_bytes_per_line": int(native.web_stream_bytes_per_line),
            "web_stream_size_image": int(native.web_stream_size_image),
            "fps": [int(native.fps_numerator), int(native.fps_denominator)],
            "camera_pool_size": int(native.camera_buffers),
            "clean_rgb_pool_size": int(native.clean_rgb_buffers),
            "web_stream_pool_size": int(native.web_stream_buffers),
            "memory_path": _decode(native.memory_path),
            "color_conversion": _decode(native.color_conversion),
            "horizontal_flip": bool(native.horizontal_flip),
            "orientation": (
                "front-camera physical view (GPU horizontal unmirror)"
                if native.horizontal_flip
                else "camera-native"
            ),
            "frames_acquired": int(native.frames_acquired),
            "frames_requeued": int(native.frames_requeued),
            "frames_dropped_no_clean_slot": int(native.frames_dropped_no_clean_slot),
            "web_stream_frames_converted": int(native.web_stream_frames_converted),
            "web_stream_frames_released": int(native.web_stream_frames_released),
            "web_stream_frames_dropped_no_slot": int(
                native.web_stream_frames_dropped_no_slot
            ),
            "web_stream_gpu_bytes_written": int(native.web_stream_gpu_bytes_written),
            "active_clean_leases": len(self._active),
            "active_web_stream_leases": len(self._active_web),
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "cpu_pixel_objects": False,
            "global_device_synchronize": False,
        }

    def close(self) -> None:
        if not self._handle.value:
            return
        self._require_creator_thread()
        if self._active or self._active_web:
            raise ZeroCopyCameraError(
                "cannot close camera with active frame leases: "
                f"clean={self._active}, web={self._active_web}"
            )
        self._library.vlm_camera_gpu_capture_destroy(self._handle)
        self._handle = ctypes.c_void_p()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        creator_thread = getattr(self, "_creator_thread", None)
        if handle is not None and handle.value and threading.get_ident() == creator_thread:
            self._library.vlm_camera_gpu_capture_destroy(handle)
