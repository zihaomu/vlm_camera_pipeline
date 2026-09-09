"""Strict HIP/OpenGL presenter for GPU-resident RGB frames, detections, and subtitles."""

from __future__ import annotations

import ctypes
import hashlib
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class ZeroCopyPresentError(RuntimeError):
    """The strict GPU-native presentation contract was violated."""


class _PresenterInfo(ctypes.Structure):
    _fields_ = [
        ("frame_width", ctypes.c_int),
        ("frame_height", ctypes.c_int),
        ("window_width", ctypes.c_int),
        ("window_height", ctypes.c_int),
        ("hip_device", ctypes.c_int),
        ("present_pool_size", ctypes.c_int),
        ("frames_presented", ctypes.c_uint64),
        ("subtitle_updates", ctypes.c_uint64),
        ("subtitle_h2d_bytes", ctypes.c_uint64),
        ("class_label_h2d_bytes", ctypes.c_uint64),
        ("performance_hud_updates", ctypes.c_uint64),
        ("performance_hud_h2d_bytes", ctypes.c_uint64),
        ("class_label_count", ctypes.c_int),
        ("class_label_width", ctypes.c_int),
        ("class_label_height", ctypes.c_int),
        ("performance_hud_visible", ctypes.c_int),
        ("performance_hud_width", ctypes.c_int),
        ("performance_hud_height", ctypes.c_int),
        ("gl_vendor", ctypes.c_char_p),
        ("gl_renderer", ctypes.c_char_p),
        ("gl_version", ctypes.c_char_p),
        ("interop_device_proof", ctypes.c_char_p),
        ("font_path", ctypes.c_char_p),
        ("window_title", ctypes.c_char_p),
        ("class_label_mode", ctypes.c_char_p),
        ("performance_hud_mode", ctypes.c_char_p),
    ]


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", "replace") if value else ""


class EglHipPresenter:
    """Two-slot DMA-BUF EGLImage presenter whose frame input is a HIP device pointer."""

    def __init__(
        self,
        *,
        library: str | Path,
        frame_width: int,
        frame_height: int,
        window_scale: float = 0.75,
        visible: bool = True,
        hip_device: int = 0,
        title: str = "YOLO26 + VLM - GPU Camera Demo",
        font_path: str | Path | None = None,
    ) -> None:
        library_path = Path(library).resolve()
        if not library_path.is_file():
            raise FileNotFoundError(
                f"EGL presenter library not found: {library_path}; "
                "run scripts/setup_egl_present_gfx1151.sh"
            )
        self.library_path = library_path
        self.library_sha256 = hashlib.sha256(library_path.read_bytes()).hexdigest()
        self._library = ctypes.CDLL(str(library_path))
        self._bind_api()
        self._handle = ctypes.c_void_p()
        self._creator_thread = threading.get_ident()
        encoded_font = None if font_path is None else str(Path(font_path).resolve()).encode()
        status = self._library.vlm_camera_egl_presenter_create(
            frame_width,
            frame_height,
            window_scale,
            int(visible),
            hip_device,
            title.encode("utf-8"),
            encoded_font,
            ctypes.byref(self._handle),
        )
        if status != 0 or not self._handle.value:
            self._raise("create EGL/HIP presenter", status)
        info = self.info()
        if info["present_pool_size"] != 2:
            self.close()
            raise ZeroCopyPresentError("strict presenter requires exactly two bounded DMA-BUF slots")

    def _bind_api(self) -> None:
        library = self._library
        library.vlm_camera_egl_presenter_last_error.argtypes = []
        library.vlm_camera_egl_presenter_last_error.restype = ctypes.c_char_p
        library.vlm_camera_egl_presenter_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.vlm_camera_egl_presenter_create.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_set_subtitle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        library.vlm_camera_egl_presenter_set_subtitle.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_set_performance_hud_text.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        library.vlm_camera_egl_presenter_set_performance_hud_text.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_set_performance_hud_visible.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.vlm_camera_egl_presenter_set_performance_hud_visible.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_present_rgb8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.vlm_camera_egl_presenter_present_rgb8.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_should_close.argtypes = [ctypes.c_void_p]
        library.vlm_camera_egl_presenter_should_close.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_PresenterInfo),
        ]
        library.vlm_camera_egl_presenter_get_info.restype = ctypes.c_int
        library.vlm_camera_egl_presenter_destroy.argtypes = [ctypes.c_void_p]
        library.vlm_camera_egl_presenter_destroy.restype = None

    def _raise(self, operation: str, status: int) -> None:
        detail = _decode(self._library.vlm_camera_egl_presenter_last_error())
        raise ZeroCopyPresentError(f"{operation} failed with status {status}: {detail}")

    def _require_open(self) -> None:
        if not self._handle.value:
            raise ZeroCopyPresentError("presenter is closed")

    def _require_creator_thread(self) -> None:
        if threading.get_ident() != self._creator_thread:
            raise ZeroCopyPresentError("EGL present/close must run on the creator thread")

    def set_subtitle(self, text: str) -> None:
        self._require_open()
        status = self._library.vlm_camera_egl_presenter_set_subtitle(
            self._handle, text.encode("utf-8")
        )
        if status:
            self._raise("set GPU subtitle resource", status)

    def set_performance_hud(self, text: str) -> None:
        self._require_open()
        status = self._library.vlm_camera_egl_presenter_set_performance_hud_text(
            self._handle, text.encode("utf-8")
        )
        if status:
            self._raise("set GPU performance HUD resource", status)

    def set_performance_visible(self, visible: bool) -> None:
        self._require_open()
        status = self._library.vlm_camera_egl_presenter_set_performance_hud_visible(
            self._handle, int(visible)
        )
        if status:
            self._raise("set GPU performance HUD visibility", status)

    def present_rgb8(
        self,
        *,
        source_pointer: int,
        source_pitch: int,
        source_is_bgr: bool,
        detections_pointer: int = 0,
        detection_count: int = 0,
        confidence_threshold: float = 0.5,
        source_ready_event: int = 0,
        detections_ready_event: int = 0,
        stream: int = 0,
    ) -> bool:
        self._require_open()
        self._require_creator_thread()
        status = self._library.vlm_camera_egl_presenter_present_rgb8(
            self._handle,
            ctypes.c_void_p(source_pointer),
            source_pitch,
            int(source_is_bgr),
            ctypes.c_void_p(detections_pointer),
            detection_count,
            confidence_threshold,
            ctypes.c_void_p(source_ready_event),
            ctypes.c_void_p(detections_ready_event),
            ctypes.c_void_p(stream),
        )
        if status == 1:
            return False
        if status:
            self._raise("present GPU frame", status)
        return True

    @property
    def should_close(self) -> bool:
        self._require_open()
        self._require_creator_thread()
        return bool(self._library.vlm_camera_egl_presenter_should_close(self._handle))

    def info(self) -> dict[str, Any]:
        self._require_open()
        native = _PresenterInfo()
        status = self._library.vlm_camera_egl_presenter_get_info(
            self._handle, ctypes.byref(native)
        )
        if status:
            self._raise("read presenter info", status)
        return {
            "frame_width": native.frame_width,
            "frame_height": native.frame_height,
            "window_width": native.window_width,
            "window_height": native.window_height,
            "hip_device": native.hip_device,
            "present_pool_size": native.present_pool_size,
            "frames_presented": native.frames_presented,
            "subtitle_updates": native.subtitle_updates,
            "subtitle_h2d_bytes": native.subtitle_h2d_bytes,
            "class_label_h2d_bytes": native.class_label_h2d_bytes,
            "performance_hud_updates": native.performance_hud_updates,
            "performance_hud_h2d_bytes": native.performance_hud_h2d_bytes,
            "class_label_count": native.class_label_count,
            "class_label_width": native.class_label_width,
            "class_label_height": native.class_label_height,
            "performance_hud_visible": bool(native.performance_hud_visible),
            "performance_hud_width": native.performance_hud_width,
            "performance_hud_height": native.performance_hud_height,
            "gl_vendor": _decode(native.gl_vendor),
            "gl_renderer": _decode(native.gl_renderer),
            "gl_version": _decode(native.gl_version),
            "interop_device_proof": _decode(native.interop_device_proof),
            "font_path": _decode(native.font_path),
            "window_title": _decode(native.window_title),
            "class_label_mode": _decode(native.class_label_mode),
            "performance_hud_mode": _decode(native.performance_hud_mode),
            "library": str(self.library_path),
            "library_sha256": self.library_sha256,
        }

    def close(self) -> None:
        if self._handle.value:
            self._require_creator_thread()
            self._library.vlm_camera_egl_presenter_destroy(self._handle)
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
            self._library.vlm_camera_egl_presenter_destroy(handle)
