"""Local web dashboard and GPU-only camera-to-H.264 presentation path."""

from __future__ import annotations

import asyncio
import copy
import ctypes
import hashlib
import json
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .zerocopy_camera import GpuYuyvFrameLease
from .zerocopy_hybrid import (
    DEFAULT_WEB_PROMPT,
    WEB_PROMPT_MAX_CHARACTERS,
    load_coco80_manifest,
    validate_web_prompt,
)
from .zerocopy_vlm import _HipRuntime
from .zerocopy_yolo import GpuDetectionLease

WEB_OVERLAY_CLASS_COUNT = 80
WEB_OVERLAY_LABEL_WIDTH = 224
WEB_OVERLAY_LABEL_HEIGHT = 36
WEB_OVERLAY_ATLAS_BYTES = (
    WEB_OVERLAY_CLASS_COUNT * WEB_OVERLAY_LABEL_WIDTH * WEB_OVERLAY_LABEL_HEIGHT
)
WEB_OVERLAY_RESOURCE_BYTES = WEB_OVERLAY_ATLAS_BYTES + WEB_OVERLAY_CLASS_COUNT * 4


class ZeroCopyWebError(RuntimeError):
    """The strict browser presentation contract failed."""


class _WebOverlayKernels:
    """ctypes boundary for the in-place GPU YUYV box/label compositor."""

    def __init__(self, library: str | Path) -> None:
        self.path = Path(library).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"zero-copy HIP kernel library not found: {self.path}; "
                "run scripts/setup_zerocopy_kernels_gfx1151.sh"
            )
        self._library = ctypes.CDLL(str(self.path))
        native = self._library
        native.vlm_camera_zerocopy_last_error.argtypes = []
        native.vlm_camera_zerocopy_last_error.restype = ctypes.c_char_p
        native.vlm_camera_web_overlay_class_count.argtypes = []
        native.vlm_camera_web_overlay_class_count.restype = ctypes.c_int
        native.vlm_camera_web_overlay_label_width.argtypes = []
        native.vlm_camera_web_overlay_label_width.restype = ctypes.c_int
        native.vlm_camera_web_overlay_label_height.argtypes = []
        native.vlm_camera_web_overlay_label_height.restype = ctypes.c_int
        native.vlm_camera_web_overlay_yuyv_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        native.vlm_camera_web_overlay_yuyv_f32.restype = ctypes.c_int
        abi = (
            int(native.vlm_camera_web_overlay_class_count()),
            int(native.vlm_camera_web_overlay_label_width()),
            int(native.vlm_camera_web_overlay_label_height()),
        )
        expected = (
            WEB_OVERLAY_CLASS_COUNT,
            WEB_OVERLAY_LABEL_WIDTH,
            WEB_OVERLAY_LABEL_HEIGHT,
        )
        if abi != expected:
            raise ZeroCopyWebError(f"native/Python Web overlay ABI mismatch: {abi}")

    def draw(
        self,
        *,
        destination_pointer: int,
        destination_pitch: int,
        width: int,
        height: int,
        detections_pointer: int,
        detection_count: int,
        confidence: float,
        class_label_pointer: int,
        detections_ready_event: int,
        stream: int,
    ) -> None:
        status = self._library.vlm_camera_web_overlay_yuyv_f32(
            ctypes.c_void_p(destination_pointer),
            destination_pitch,
            width,
            height,
            ctypes.c_void_p(detections_pointer),
            detection_count,
            confidence,
            ctypes.c_void_p(class_label_pointer),
            ctypes.c_void_p(class_label_pointer + WEB_OVERLAY_ATLAS_BYTES),
            ctypes.c_void_p(detections_ready_event),
            ctypes.c_void_p(stream),
        )
        if status:
            detail = self._library.vlm_camera_zerocopy_last_error()
            decoded = detail.decode("utf-8", "replace") if detail else f"status {status}"
            raise ZeroCopyWebError(f"Web GPU YUYV overlay failed: {decoded}")


def _resolve_overlay_font() -> Path:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ZeroCopyWebError("no local DejaVu Sans font is available for Web class labels")


def _build_class_label_resource(
    class_names: tuple[str, ...], font_path: Path
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    resource = np.zeros(WEB_OVERLAY_RESOURCE_BYTES, dtype=np.uint8)
    atlas = resource[:WEB_OVERLAY_ATLAS_BYTES].reshape(
        WEB_OVERLAY_CLASS_COUNT,
        WEB_OVERLAY_LABEL_HEIGHT,
        WEB_OVERLAY_LABEL_WIDTH,
    )
    widths = resource[WEB_OVERLAY_ATLAS_BYTES:].view(np.int32)
    font = ImageFont.truetype(str(font_path), 25)
    padding_x = 6
    for class_id, class_name in enumerate(class_names):
        label = Image.new(
            "L", (WEB_OVERLAY_LABEL_WIDTH, WEB_OVERLAY_LABEL_HEIGHT), color=0
        )
        draw = ImageDraw.Draw(label)
        bounds = draw.textbbox((0, 0), class_name, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        text_x = padding_x - bounds[0]
        text_y = (WEB_OVERLAY_LABEL_HEIGHT - text_height) // 2 - bounds[1]
        draw.text((text_x, text_y), class_name, font=font, fill=255)
        atlas[class_id] = np.asarray(label, dtype=np.uint8)
        widths[class_id] = min(
            WEB_OVERLAY_LABEL_WIDTH, text_width + padding_x * 2
        )
    return resource


class WebGpuYuyvOverlay:
    """Burn YOLO boxes and canonical class names into a GPU YUYV DMA-BUF."""

    def __init__(
        self,
        *,
        kernel_library: str | Path,
        coco_manifest: str | Path,
        model_names: Mapping[int, str],
        upload_stream: int,
        confidence: float = 0.5,
        device_id: int = 0,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Web overlay confidence must be between 0 and 1")
        self.class_names, self.manifest_sha256 = load_coco80_manifest(
            coco_manifest, model_names=model_names
        )
        self.confidence = confidence
        self.runtime = _HipRuntime(device_id)
        self.kernels = _WebOverlayKernels(kernel_library)
        self.font_path = _resolve_overlay_font()
        self.font_sha256 = hashlib.sha256(self.font_path.read_bytes()).hexdigest()
        self._class_label_pointer = 0
        self._done_event = 0
        host_pointer = 0
        upload_event = 0
        try:
            resource = _build_class_label_resource(self.class_names, self.font_path)
            self._class_label_pointer = self.runtime.allocate(WEB_OVERLAY_RESOURCE_BYTES)
            host_pointer = self.runtime.host_allocate(WEB_OVERLAY_RESOURCE_BYTES)
            upload_event = self.runtime.create_local_event()
            ctypes.memmove(host_pointer, int(resource.ctypes.data), WEB_OVERLAY_RESOURCE_BYTES)
            self.runtime.copy_host_to_device_async(
                self._class_label_pointer,
                host_pointer,
                WEB_OVERLAY_RESOURCE_BYTES,
                upload_stream,
            )
            self.runtime.record_event(upload_event, upload_stream)
            self.runtime.synchronize_event(upload_event)
            self._done_event = self.runtime.create_local_event()
        except BaseException:
            if self._done_event:
                self.runtime.destroy_event(self._done_event)
                self._done_event = 0
            if self._class_label_pointer:
                self.runtime.free(self._class_label_pointer)
                self._class_label_pointer = 0
            raise
        finally:
            if upload_event:
                self.runtime.destroy_event(upload_event)
            if host_pointer:
                self.runtime.host_free(host_pointer)
        self._draws = 0
        self._last_ms: float | None = None
        self._total_ms = 0.0
        self._max_ms = 0.0
        self._closed = False

    def draw(
        self,
        lease: GpuYuyvFrameLease,
        detection: GpuDetectionLease,
        *,
        stream: int,
    ) -> float:
        if self._closed:
            raise ZeroCopyWebError("Web GPU overlay is closed")
        if lease.released or lease.device_pointer <= 0:
            raise ZeroCopyWebError("Web GPU overlay received a released YUYV lease")
        started = time.perf_counter()
        self.kernels.draw(
            destination_pointer=lease.device_pointer,
            destination_pitch=lease.pitch,
            width=lease.width,
            height=lease.height,
            detections_pointer=detection.tensor.data_ptr(),
            detection_count=int(detection.tensor.shape[0]),
            confidence=self.confidence,
            class_label_pointer=self._class_label_pointer,
            detections_ready_event=detection.ready_event.cuda_event,
            stream=stream,
        )
        # VAAPI is an external consumer and cannot wait on a HIP event. Synchronize
        # only this overlay resource before importing the same allocation as DMA-BUF.
        self.runtime.record_event(self._done_event, stream)
        self.runtime.synchronize_event(self._done_event)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._draws += 1
        self._last_ms = elapsed_ms
        self._total_ms += elapsed_ms
        self._max_ms = max(self._max_ms, elapsed_ms)
        return elapsed_ms

    def runtime_info(self) -> dict[str, Any]:
        return {
            "mode": "in-place HIP YUYV box/class-label burn-in before VAAPI",
            "draws": self._draws,
            "confidence": self.confidence,
            "class_count": len(self.class_names),
            "class_label_size": [
                WEB_OVERLAY_LABEL_WIDTH,
                WEB_OVERLAY_LABEL_HEIGHT,
            ],
            "class_label_h2d_bytes": WEB_OVERLAY_RESOURCE_BYTES,
            "per_frame_control_d2h_bytes": 0,
            "full_detection_tensor_d2h_bytes": 0,
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "uses_global_device_synchronize": False,
            "external_consumer_boundary": "per-resource HIP event synchronize",
            "last_ms": self._last_ms,
            "mean_ms": self._total_ms / self._draws if self._draws else None,
            "max_ms": self._max_ms if self._draws else None,
            "coco80_manifest_sha256": self.manifest_sha256,
            "font": str(self.font_path),
            "font_sha256": self.font_sha256,
            "kernel_library": str(self.kernels.path),
            "kernel_library_sha256": hashlib.sha256(
                self.kernels.path.read_bytes()
            ).hexdigest(),
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._done_event:
            self.runtime.destroy_event(self._done_event)
            self._done_event = 0
        if self._class_label_pointer:
            self.runtime.free(self._class_label_pointer)
            self._class_label_pointer = 0
        self._closed = True


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    text: str
    version: int
    updated_unix_seconds: float


class PromptStore:
    """Thread-safe, versioned prompt state; prompt text is the sole UI control."""

    def __init__(self, initial: str = DEFAULT_WEB_PROMPT) -> None:
        self._lock = threading.Lock()
        self._text = validate_web_prompt(initial)
        self._version = 1
        self._updated = time.time()

    def snapshot(self) -> PromptSnapshot:
        with self._lock:
            return PromptSnapshot(self._text, self._version, self._updated)

    def update(self, value: str) -> PromptSnapshot:
        normalized = validate_web_prompt(value)
        with self._lock:
            if normalized != self._text:
                self._text = normalized
                self._version += 1
                self._updated = time.time()
            return PromptSnapshot(self._text, self._version, self._updated)


class _Mp4BoxParser:
    """Split arbitrary mp4mux buffers into complete top-level ISO BMFF boxes."""

    def __init__(self) -> None:
        self._pending = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[bytes, bytes]]:
        self._pending.extend(chunk)
        boxes: list[tuple[bytes, bytes]] = []
        while len(self._pending) >= 8:
            size = int.from_bytes(self._pending[0:4], "big")
            header_size = 8
            if size == 1:
                if len(self._pending) < 16:
                    break
                size = int.from_bytes(self._pending[8:16], "big")
                header_size = 16
            if size == 0:
                break
            if size < header_size:
                raise ZeroCopyWebError(f"invalid MP4 box size: {size}")
            if len(self._pending) < size:
                break
            box = bytes(self._pending[:size])
            del self._pending[:size]
            boxes.append((box[4:8], box))
        return boxes


class _Mp4FragmentAssembler:
    """Join one moof and its following boxes through mdat into an MSE segment."""

    def __init__(self) -> None:
        self._pending: bytearray | None = None

    def feed(self, kind: bytes, box: bytes) -> bytes | None:
        if kind == b"moof":
            if self._pending is not None:
                self._pending = bytearray(box)
                raise ZeroCopyWebError("received a new moof before the previous mdat")
            self._pending = bytearray(box)
            return None
        if self._pending is None:
            return None
        self._pending.extend(box)
        if kind != b"mdat":
            return None
        segment = bytes(self._pending)
        self._pending = None
        return segment


def _iter_mp4_boxes(
    data: bytes,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[bytes, int, int]]:
    limit = len(data) if end is None else end
    cursor = start
    boxes: list[tuple[bytes, int, int]] = []
    while cursor < limit:
        if cursor + 8 > limit:
            raise ZeroCopyWebError("truncated MP4 box header")
        size = int.from_bytes(data[cursor : cursor + 4], "big")
        kind = data[cursor + 4 : cursor + 8]
        header_size = 8
        if size == 1:
            if cursor + 16 > limit:
                raise ZeroCopyWebError("truncated extended MP4 box header")
            size = int.from_bytes(data[cursor + 8 : cursor + 16], "big")
            header_size = 16
        if size == 0:
            size = limit - cursor
        if size < header_size or cursor + size > limit:
            raise ZeroCopyWebError(f"invalid or truncated {kind!r} MP4 box")
        boxes.append((kind, cursor + header_size, cursor + size))
        cursor += size
    return boxes


def _full_box_flags(data: bytes, payload_start: int, payload_end: int) -> int:
    if payload_start + 4 > payload_end:
        raise ZeroCopyWebError("truncated MP4 full-box header")
    return int.from_bytes(data[payload_start + 1 : payload_start + 4], "big")


def _tfhd_default_sample_flags(
    data: bytes,
    payload_start: int,
    payload_end: int,
) -> int | None:
    flags = _full_box_flags(data, payload_start, payload_end)
    cursor = payload_start + 8  # full-box header plus track_ID
    for presence_flag, field_size in (
        (0x000001, 8),
        (0x000002, 4),
        (0x000008, 4),
        (0x000010, 4),
    ):
        if flags & presence_flag:
            cursor += field_size
    if not flags & 0x000020:
        return None
    if cursor + 4 > payload_end:
        raise ZeroCopyWebError("truncated tfhd default_sample_flags")
    return int.from_bytes(data[cursor : cursor + 4], "big")


def _trun_first_sample_flags(
    data: bytes,
    payload_start: int,
    payload_end: int,
    default_sample_flags: int | None,
) -> int:
    flags = _full_box_flags(data, payload_start, payload_end)
    if payload_start + 8 > payload_end:
        raise ZeroCopyWebError("truncated trun sample_count")
    sample_count = int.from_bytes(data[payload_start + 4 : payload_start + 8], "big")
    if sample_count == 0:
        raise ZeroCopyWebError("empty trun in browser media fragment")
    cursor = payload_start + 8
    if flags & 0x000001:
        cursor += 4
    if flags & 0x000004:
        if cursor + 4 > payload_end:
            raise ZeroCopyWebError("truncated trun first_sample_flags")
        return int.from_bytes(data[cursor : cursor + 4], "big")
    if flags & 0x000100:
        cursor += 4
    if flags & 0x000200:
        cursor += 4
    if flags & 0x000400:
        if cursor + 4 > payload_end:
            raise ZeroCopyWebError("truncated trun sample_flags")
        return int.from_bytes(data[cursor : cursor + 4], "big")
    if default_sample_flags is None:
        raise ZeroCopyWebError("fragment does not declare first-sample flags")
    return default_sample_flags


def _fragment_starts_with_sync_sample(media_segment: bytes) -> bool:
    """Return whether an fMP4 media segment begins with an independently decodable frame."""
    for top_kind, top_start, top_end in _iter_mp4_boxes(media_segment):
        if top_kind != b"moof":
            continue
        for child_kind, child_start, child_end in _iter_mp4_boxes(
            media_segment, top_start, top_end
        ):
            if child_kind != b"traf":
                continue
            default_sample_flags: int | None = None
            trun_boxes: list[tuple[int, int]] = []
            for traf_kind, traf_start, traf_end in _iter_mp4_boxes(
                media_segment, child_start, child_end
            ):
                if traf_kind == b"tfhd":
                    default_sample_flags = _tfhd_default_sample_flags(
                        media_segment, traf_start, traf_end
                    )
                elif traf_kind == b"trun":
                    trun_boxes.append((traf_start, traf_end))
            if trun_boxes:
                sample_flags = _trun_first_sample_flags(
                    media_segment,
                    *trun_boxes[0],
                    default_sample_flags,
                )
                return sample_flags & 0x00010000 == 0
    raise ZeroCopyWebError("media segment has no moof/traf/trun sample table")


@dataclass(slots=True)
class _VideoClient:
    websocket: Any
    init_sent: bool = False
    waiting_for_sync_sample: bool = True


class WebDashboard:
    """A localhost aiohttp dashboard serving state, prompt updates, and fMP4."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        host: str = "127.0.0.1",
        port: int = 8765,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.assets = self.workspace / "web"
        self.host = host
        self.port = port
        self.prompt_store = PromptStore()
        self._state_lock = threading.Lock()
        prompt = self.prompt_store.snapshot()
        self._state: dict[str, Any] = {
            "schema_version": 1,
            "pipeline": {
                "state": "starting",
                "detail": "Initializing GPU pipeline",
                "started_unix_seconds": time.time(),
            },
            "video": {
                "width": width,
                "height": height,
                "fps": fps,
                "transport": "WebSocket fragmented MP4",
                "codec": "H.264 constrained baseline / VAAPI",
                "clients": 0,
                "encoded_bytes": 0,
                "timing": {
                    "mapped_capture_monotonic_seconds": None,
                    "mapped_media_pts_seconds": None,
                    "timestamp_discontinuities": 0,
                    "latest_fragment_capture_monotonic_seconds": None,
                    "server_monotonic_seconds": time.monotonic(),
                },
            },
            "prompt": {
                "text": prompt.text,
                "version": prompt.version,
                "updated_unix_seconds": prompt.updated_unix_seconds,
                "scheduled_version": 0,
                "applied_version": 0,
                "max_characters": WEB_PROMPT_MAX_CHARACTERS,
            },
            "caption": {
                "text": "Waiting for the first 3-second VLM caption…",
                "request_id": 0,
                "source_frame_id": 0,
                "prompt_version": 0,
                "updated_unix_seconds": None,
            },
            "performance": {
                "camera_fps": None,
                "yolo_fps": None,
                "yolo_inference_ms": None,
                "vlm_latency_ms": None,
                "vlm_tokens_per_second": None,
                "vlm_state": "WARMING UP",
                "vlm_interval_seconds": 3.0,
            },
            "contract": {
                "camera": "V4L2 DMA-BUF",
                "preprocess": "OpenCV 5 HIP",
                "detector": "YOLO26 / MIGraphX",
                "vlm": "Qwen3-VL / llama.cpp HIP IPC",
                "overlay": "HIP in-place YUYV box/class-label burn-in",
                "encoder": "VAAPI H.264",
                "image_host_copies": 0,
            },
        }
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._box_parser = _Mp4BoxParser()
        self._fragment_assembler = _Mp4FragmentAssembler()
        self._video_queue: asyncio.Queue[tuple[bytes, bytes, int | None]] | None = None
        self._clients: dict[int, _VideoClient] = {}
        self._init_boxes: dict[bytes, bytes] = {}

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def video_client_count(self) -> int:
        with self._state_lock:
            return int(self._state["video"]["clients"])

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> Self:
        for asset in ("index.html", "app.css", "app.js", "amd-mark.svg"):
            path = self.assets / asset
            if not path.is_file():
                raise FileNotFoundError(f"web dashboard asset is missing: {path}")
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._thread_main, name="web-dashboard", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise ZeroCopyWebError("timed out starting the local web dashboard")
        if self._startup_error is not None:
            raise ZeroCopyWebError(f"web dashboard startup failed: {self._startup_error}")
        return self

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001 - forwarded to strict main thread
            self._startup_error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from aiohttp import web

        self._loop = asyncio.get_running_loop()
        self._video_queue = asyncio.Queue()
        app = web.Application(client_max_size=4096)
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/app.css", self._handle_css)
        app.router.add_get("/app.js", self._handle_js)
        app.router.add_get("/amd-mark.svg", self._handle_amd_mark)
        app.router.add_get("/api/state", self._handle_state)
        app.router.add_put("/api/prompt", self._handle_prompt)
        app.router.add_get("/video", self._handle_video)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        try:
            await site.start()
            broadcaster = asyncio.create_task(self._broadcast_video())
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
            broadcaster.cancel()
            await asyncio.gather(broadcaster, return_exceptions=True)
            for client in list(self._clients.values()):
                await client.websocket.close(code=1001, message=b"pipeline stopped")
            self._clients.clear()
        finally:
            await runner.cleanup()
            self._loop = None
            self._video_queue = None

    async def _asset_response(self, name: str, content_type: str) -> Any:
        from aiohttp import web

        return web.Response(
            body=(self.assets / name).read_bytes(),
            content_type=content_type,
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_index(self, request: Any) -> Any:
        del request
        return await self._asset_response("index.html", "text/html")

    async def _handle_css(self, request: Any) -> Any:
        del request
        return await self._asset_response("app.css", "text/css")

    async def _handle_js(self, request: Any) -> Any:
        del request
        return await self._asset_response("app.js", "application/javascript")

    async def _handle_amd_mark(self, request: Any) -> Any:
        del request
        return await self._asset_response("amd-mark.svg", "image/svg+xml")

    async def _handle_state(self, request: Any) -> Any:
        from aiohttp import web

        del request
        return web.json_response(self.state_snapshot(), headers={"Cache-Control": "no-store"})

    async def _handle_prompt(self, request: Any) -> Any:
        from aiohttp import web

        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {"prompt"}:
                raise ValueError("request body must contain only the prompt field")
            snapshot = self.prompt_store.update(payload["prompt"])
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        with self._state_lock:
            self._state["prompt"].update(
                {
                    "text": snapshot.text,
                    "version": snapshot.version,
                    "updated_unix_seconds": snapshot.updated_unix_seconds,
                }
            )
        return web.json_response(
            {"prompt": snapshot.text, "version": snapshot.version},
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_video(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        websocket = web.WebSocketResponse(heartbeat=15.0, max_msg_size=1024)
        await websocket.prepare(request)
        client = _VideoClient(websocket=websocket)
        client_id = id(websocket)
        self._clients[client_id] = client
        self._publish_client_count()
        await websocket.send_str(
            json.dumps(
                {
                    "type": "stream-info",
                    "mime": 'video/mp4; codecs="avc1.42C01F"',
                    "width": self._state["video"]["width"],
                    "height": self._state["video"]["height"],
                }
            )
        )
        await self._send_init_if_ready(client)
        try:
            async for message in websocket:
                if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        finally:
            self._clients.pop(client_id, None)
            self._publish_client_count()
        return websocket

    def _publish_client_count(self) -> None:
        with self._state_lock:
            self._state["video"]["clients"] = len(self._clients)

    async def _send_init_if_ready(self, client: _VideoClient) -> None:
        ftyp = self._init_boxes.get(b"ftyp")
        moov = self._init_boxes.get(b"moov")
        if ftyp is not None and moov is not None and not client.init_sent:
            await client.websocket.send_bytes(ftyp + moov)
            client.init_sent = True

    async def _broadcast_video(self) -> None:
        assert self._video_queue is not None
        while True:
            kind, box, latest_encoded_capture_ns = await self._video_queue.get()
            if kind in (b"ftyp", b"moov"):
                self._init_boxes[kind] = box
                for client in list(self._clients.values()):
                    await self._send_init_if_ready(client)
                continue
            try:
                media_segment = self._fragment_assembler.feed(kind, box)
            except ZeroCopyWebError as exc:
                self.set_pipeline_state("error", f"MP4 fragment assembler: {exc}")
                continue
            if media_segment is None:
                continue
            if latest_encoded_capture_ns is not None:
                with self._state_lock:
                    self._state["video"]["timing"][
                        "latest_fragment_capture_monotonic_seconds"
                    ] = latest_encoded_capture_ns / 1_000_000_000.0
            try:
                starts_with_sync_sample = _fragment_starts_with_sync_sample(media_segment)
            except ZeroCopyWebError as exc:
                self.set_pipeline_state("error", f"MP4 sync-sample parser: {exc}")
                continue
            stale: list[int] = []
            for client_id, client in list(self._clients.items()):
                if not client.init_sent:
                    continue
                if client.waiting_for_sync_sample:
                    if not starts_with_sync_sample:
                        continue
                    client.waiting_for_sync_sample = False
                try:
                    await asyncio.wait_for(
                        client.websocket.send_bytes(media_segment), timeout=0.25
                    )
                except (TimeoutError, ConnectionError, RuntimeError):
                    stale.append(client_id)
            for client_id in stale:
                client = self._clients.pop(client_id, None)
                if client is not None:
                    await client.websocket.close()
            if stale:
                self._publish_client_count()

    def publish_mp4_bytes(
        self,
        chunk: bytes,
        latest_encoded_capture_ns: int | None = None,
    ) -> None:
        loop = self._loop
        if loop is None or not chunk:
            return

        def enqueue() -> None:
            try:
                boxes = self._box_parser.feed(chunk)
            except BaseException as exc:  # noqa: BLE001
                self.set_pipeline_state("error", f"MP4 parser: {exc}")
                return
            target = self._video_queue
            if target is not None:
                for box in boxes:
                    target.put_nowait((*box, latest_encoded_capture_ns))
            with self._state_lock:
                self._state["video"]["encoded_bytes"] += len(chunk)

        loop.call_soon_threadsafe(enqueue)

    def prompt_snapshot(self) -> PromptSnapshot:
        return self.prompt_store.snapshot()

    def set_pipeline_state(self, state: str, detail: str) -> None:
        with self._state_lock:
            self._state["pipeline"].update({"state": state, "detail": detail})

    def mark_prompt_scheduled(self, version: int) -> None:
        with self._state_lock:
            self._state["prompt"]["scheduled_version"] = version

    def set_caption(
        self,
        text: str,
        *,
        request_id: int,
        source_frame_id: int,
        prompt_version: int,
    ) -> None:
        with self._state_lock:
            self._state["caption"] = {
                "text": text,
                "request_id": request_id,
                "source_frame_id": source_frame_id,
                "prompt_version": prompt_version,
                "updated_unix_seconds": time.time(),
            }
            self._state["prompt"]["applied_version"] = prompt_version

    def set_performance(
        self,
        *,
        camera_fps: float | None,
        yolo_fps: float | None,
        yolo_inference_ms: float | None,
        vlm_latency_ms: float | None,
        vlm_tokens_per_second: float | None,
        vlm_running: bool,
    ) -> None:
        with self._state_lock:
            self._state["performance"].update(
                {
                    "camera_fps": camera_fps,
                    "yolo_fps": yolo_fps,
                    "yolo_inference_ms": yolo_inference_ms,
                    "vlm_latency_ms": vlm_latency_ms,
                    "vlm_tokens_per_second": vlm_tokens_per_second,
                    "vlm_state": "RUNNING" if vlm_running else "IDLE",
                }
            )

    def state_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            snapshot = copy.deepcopy(self._state)
        snapshot["video"]["timing"]["server_monotonic_seconds"] = time.monotonic()
        return snapshot

    def set_video_timing_mapping(
        self,
        *,
        captured_monotonic_ns: int,
        media_pts_ns: int,
        discontinuities: int,
    ) -> None:
        if captured_monotonic_ns <= 0 or media_pts_ns < 0 or discontinuities < 0:
            raise ValueError("video timing mapping contains an invalid value")
        with self._state_lock:
            self._state["video"]["timing"].update(
                {
                    "mapped_capture_monotonic_seconds": (
                        captured_monotonic_ns / 1_000_000_000.0
                    ),
                    "mapped_media_pts_seconds": media_pts_ns / 1_000_000_000.0,
                    "timestamp_discontinuities": discontinuities,
                }
            )

    def info(self) -> dict[str, Any]:
        state = self.state_snapshot()
        return {
            "url": self.url,
            "host": self.host,
            "port": self.port,
            "video_clients": state["video"]["clients"],
            "prompt_version": state["prompt"]["version"],
            "prompt_applied_version": state["prompt"]["applied_version"],
            "encoded_bytes": state["video"]["encoded_bytes"],
        }

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: None)
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise ZeroCopyWebError("web dashboard thread did not stop")
        self._thread = None


class _CaptureTimestampClock:
    """Build a monotonic media timeline from real Camera capture intervals."""

    def __init__(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.frame_duration_ns = 1_000_000_000 // fps
        self._last_capture_ns: int | None = None
        self._last_pts_ns: int | None = None
        self.discontinuities = 0

    def mapping(self) -> tuple[int, int] | None:
        if self._last_capture_ns is None or self._last_pts_ns is None:
            return None
        return self._last_capture_ns, self._last_pts_ns

    def next(self, captured_ns: int) -> tuple[int, int]:
        if captured_ns <= 0:
            raise ValueError("Camera capture timestamp must be positive")
        if self._last_capture_ns is None:
            pts_ns = 0
            duration_ns = self.frame_duration_ns
        else:
            capture_delta = captured_ns - self._last_capture_ns
            minimum_delta = self.frame_duration_ns // 2
            maximum_delta = self.frame_duration_ns * 4
            if not minimum_delta <= capture_delta <= maximum_delta:
                capture_delta = self.frame_duration_ns
                self.discontinuities += 1
            assert self._last_pts_ns is not None
            pts_ns = self._last_pts_ns + capture_delta
            duration_ns = capture_delta
        self._last_capture_ns = captured_ns
        self._last_pts_ns = pts_ns
        return pts_ns, duration_ns


class VaapiMseStreamer:
    """Import HIP YUYV DMA-BUFs, GPU-convert/encode them, and emit fMP4 bytes."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        publish_mp4_bytes: Any,
        bitrate_kbps: int = 4000,
    ) -> None:
        if width != 1280 or height != 720 or fps != 30:
            raise ValueError("the web encoder contract requires 1280x720@30")
        os.environ.setdefault("LIBVA_DRIVER_NAME", "radeonsi")
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstAllocators", "1.0")
        gi.require_version("GstApp", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstAllocators, GstVideo

        Gst.init(None)
        self._Gst = Gst
        self._GstAllocators = GstAllocators
        self._GstVideo = GstVideo
        self.width = width
        self.height = height
        self.fps = fps
        self._publish_mp4_bytes = publish_mp4_bytes
        self._allocator = GstAllocators.DmaBufAllocator.new()
        self._completed_pts: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._inflight_lock = threading.Lock()
        self._inflight: dict[int, GpuYuyvFrameLease] = {}
        self._encoder_input_pts: deque[int] = deque()
        self._fatal_lock = threading.Lock()
        self._fatal_error: str | None = None
        self._submitted = 0
        self._rejected = 0
        self._encoded = 0
        self._encoded_bytes = 0
        self._encoder_inputs = 0
        self._encoder_outputs_without_input = 0
        self._encoder_input_mapping_misses = 0
        self._first_submitted_pts_ns: int | None = None
        self._last_submitted_pts_ns: int | None = None
        self._first_encoded_pts_ns: int | None = None
        self._last_encoded_pts_ns: int | None = None
        self._last_encoder_pts_offset_ns: int | None = None
        self._latest_encoded_capture_ns: int | None = None
        self._latest_capture_to_encoder_ms: float | None = None
        self._max_capture_to_encoder_ms = 0.0
        self._released = 0
        self._closed = False
        self._timestamp_clock = _CaptureTimestampClock(fps)
        pipeline_description = (
            "appsrc name=source is-live=true format=time block=false max-buffers=2 "
            "leaky-type=downstream "
            f"caps=video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=YUYV,"
            f"width={width},height={height},framerate=0/1 "
            "! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
            "! vapostproc name=vpp "
            f"! video/x-raw(memory:VAMemory),format=NV12,width={width},height={height},"
            f"framerate=0/1 "
            f"! vah264enc name=encoder bitrate={bitrate_kbps} key-int-max=6 b-frames=0 "
            "ref-frames=1 target-usage=7 rate-control=cbr cabac=false dct8x8=false "
            "! video/x-h264,profile=constrained-baseline,stream-format=byte-stream,alignment=au "
            "! h264parse config-interval=-1 "
            "! video/x-h264,profile=constrained-baseline,stream-format=avc,alignment=au "
            "! mp4mux fragment-duration=50 interleave-time=0 streamable=true "
            "! appsink name=sink emit-signals=true sync=false processing-deadline=0 "
            "max-buffers=8 drop=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_description)
        self._source = self._pipeline.get_by_name("source")
        self._encoder = self._pipeline.get_by_name("encoder")
        self._sink = self._pipeline.get_by_name("sink")
        if self._source is None or self._encoder is None or self._sink is None:
            raise ZeroCopyWebError("GStreamer did not create the required named elements")
        self._sink.connect("new-sample", self._on_new_sample)
        encoder_pad = self._encoder.get_static_pad("src")
        encoder_sink_pad = self._encoder.get_static_pad("sink")
        if encoder_pad is None or encoder_sink_pad is None:
            raise ZeroCopyWebError("VAAPI encoder has incomplete pads")
        encoder_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._on_encoder_input_buffer,
        )
        encoder_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_encoded_buffer)
        state_change = self._pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise ZeroCopyWebError("VAAPI web encoder refused PLAYING")

    @property
    def fatal_error(self) -> str | None:
        with self._fatal_lock:
            return self._fatal_error

    @property
    def timing_mapping(self) -> tuple[int, int] | None:
        return self._timestamp_clock.mapping()

    @property
    def timestamp_discontinuities(self) -> int:
        return self._timestamp_clock.discontinuities

    def _set_fatal(self, detail: str) -> None:
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = detail

    def _on_encoder_input_buffer(self, pad: Any, probe_info: Any) -> Any:
        del pad
        buffer = probe_info.get_buffer()
        if buffer is not None and buffer.pts != self._Gst.CLOCK_TIME_NONE:
            with self._inflight_lock:
                self._encoder_input_pts.append(int(buffer.pts))
                self._encoder_inputs += 1
        return self._Gst.PadProbeReturn.OK

    def _on_encoded_buffer(self, pad: Any, probe_info: Any) -> Any:
        del pad
        buffer = probe_info.get_buffer()
        if buffer is not None and buffer.pts != self._Gst.CLOCK_TIME_NONE:
            pts = int(buffer.pts)
            with self._inflight_lock:
                if self._encoder_input_pts:
                    input_pts = self._encoder_input_pts.popleft()
                else:
                    input_pts = None
                    self._encoder_outputs_without_input += 1
                lease = self._inflight.get(input_pts) if input_pts is not None else None
                if lease is not None:
                    self._latest_encoded_capture_ns = lease.captured_monotonic_ns
                    latency_ms = (
                        time.monotonic_ns() - lease.captured_monotonic_ns
                    ) / 1_000_000.0
                    self._latest_capture_to_encoder_ms = latency_ms
                    self._max_capture_to_encoder_ms = max(
                        self._max_capture_to_encoder_ms,
                        latency_ms,
                    )
                elif input_pts is not None:
                    self._encoder_input_mapping_misses += 1
                if self._first_encoded_pts_ns is None:
                    self._first_encoded_pts_ns = pts
                self._last_encoded_pts_ns = pts
                self._last_encoder_pts_offset_ns = (
                    pts - input_pts if input_pts is not None else None
                )
            if input_pts is not None:
                self._completed_pts.put(input_pts)
            self._encoded += 1
        return self._Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink: Any) -> Any:
        sample = sink.emit("pull-sample")
        if sample is None:
            return self._Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        mapped, mapping = buffer.map(self._Gst.MapFlags.READ)
        if not mapped:
            self._set_fatal("failed to map encoded fMP4 output")
            return self._Gst.FlowReturn.ERROR
        try:
            payload = bytes(mapping.data)
        finally:
            buffer.unmap(mapping)
        self._encoded_bytes += len(payload)
        with self._inflight_lock:
            latest_encoded_capture_ns = self._latest_encoded_capture_ns
        self._publish_mp4_bytes(payload, latest_encoded_capture_ns)
        return self._Gst.FlowReturn.OK

    def submit(self, lease: GpuYuyvFrameLease) -> bool:
        """Take ownership of a web frame lease and enqueue it without a host pixel map."""
        if self._closed:
            lease.release()
            raise ZeroCopyWebError("web encoder is closed")
        self.drain_completed()
        if self.fatal_error is not None:
            lease.release()
            return False
        with self._inflight_lock:
            if len(self._inflight) >= 7:
                self._rejected += 1
                lease.release()
                return False
        fd = lease.duplicate_fd()
        try:
            memory = self._GstAllocators.DmaBufAllocator.alloc(
                self._allocator,
                fd,
                lease.allocation_bytes,
            )
        except BaseException:
            os.close(fd)
            lease.release()
            raise
        buffer = self._Gst.Buffer.new()
        buffer.append_memory(memory)
        self._GstVideo.buffer_add_video_meta_full(
            buffer,
            self._GstVideo.VideoFrameFlags.NONE,
            self._GstVideo.VideoFormat.YUY2,
            lease.width,
            lease.height,
            1,
            [0, 0, 0, 0],
            [lease.pitch, 0, 0, 0],
        )
        pts, duration = self._timestamp_clock.next(lease.captured_monotonic_ns)
        buffer.pts = pts
        buffer.dts = pts
        buffer.duration = duration
        buffer.offset = lease.frame_id
        with self._inflight_lock:
            self._inflight[pts] = lease
            if self._first_submitted_pts_ns is None:
                self._first_submitted_pts_ns = pts
            self._last_submitted_pts_ns = pts
        flow = self._source.emit("push-buffer", buffer)
        if flow != self._Gst.FlowReturn.OK:
            with self._inflight_lock:
                owned = self._inflight.pop(pts, None)
            if owned is not None:
                owned.release()
            self._rejected += 1
            self._set_fatal(f"appsrc push failed: {flow.value_nick}")
            return False
        self._submitted += 1
        return True

    def _poll_bus(self) -> None:
        message = self._pipeline.get_bus().timed_pop_filtered(
            0,
            self._Gst.MessageType.ERROR,
        )
        if message is not None:
            error, debug = message.parse_error()
            self._set_fatal(f"GStreamer: {error}; {debug}")

    def drain_completed(self) -> int:
        """Release encoder-consumed leases on the camera-owning main thread."""
        self._poll_bus()
        newest_pts: int | None = None
        while True:
            try:
                pts = self._completed_pts.get_nowait()
            except queue.Empty:
                break
            newest_pts = pts if newest_pts is None else max(newest_pts, pts)
        if newest_pts is None:
            return 0
        with self._inflight_lock:
            ready = [pts for pts in self._inflight if pts <= newest_pts]
            leases = [self._inflight.pop(pts) for pts in sorted(ready)]
        for lease in leases:
            lease.release()
        self._released += len(leases)
        return len(leases)

    def info(self) -> dict[str, Any]:
        with self._inflight_lock:
            inflight = len(self._inflight)
        return {
            "backend": "GStreamer VAAPI H.264 -> fragmented MP4",
            "input": "HIP/HSA linear YUYV DMA-BUF",
            "gpu_vpp": "VAAPI YUYV -> native NV12",
            "encoder": "vah264enc",
            "timestamp_source": "Camera monotonic capture deltas",
            "timestamp_discontinuities": self._timestamp_clock.discontinuities,
            "submitted": self._submitted,
            "encoded": self._encoded,
            "encoded_bytes": self._encoded_bytes,
            "encoder_inputs": self._encoder_inputs,
            "encoder_outputs_without_input": self._encoder_outputs_without_input,
            "encoder_input_mapping_misses": self._encoder_input_mapping_misses,
            "first_submitted_pts_ns": self._first_submitted_pts_ns,
            "last_submitted_pts_ns": self._last_submitted_pts_ns,
            "first_encoded_pts_ns": self._first_encoded_pts_ns,
            "last_encoded_pts_ns": self._last_encoded_pts_ns,
            "last_encoder_pts_offset_ns": self._last_encoder_pts_offset_ns,
            "latest_capture_to_encoder_ms": self._latest_capture_to_encoder_ms,
            "max_capture_to_encoder_ms": self._max_capture_to_encoder_ms,
            "released": self._released,
            "rejected": self._rejected,
            "inflight": inflight,
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "encoded_output_host_bytes": self._encoded_bytes,
            "fatal_error": self.fatal_error,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._source.emit("end-of-stream")
        self._pipeline.get_bus().timed_pop_filtered(
            2 * self._Gst.SECOND,
            self._Gst.MessageType.ERROR | self._Gst.MessageType.EOS,
        )
        self._pipeline.set_state(self._Gst.State.NULL)
        self.drain_completed()
        with self._inflight_lock:
            leases = list(self._inflight.values())
            self._inflight.clear()
            self._encoder_input_pts.clear()
        for lease in leases:
            lease.release()
        self._released += len(leases)


class WebPipelinePresenter:
    """Facade used by the real-time loop for browser presentation."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.dashboard = WebDashboard(
            workspace=workspace,
            host=host,
            port=port,
            width=width,
            height=height,
            fps=fps,
        ).start()
        try:
            self.streamer = VaapiMseStreamer(
                width=width,
                height=height,
                fps=fps,
                publish_mp4_bytes=self.dashboard.publish_mp4_bytes,
            )
        except BaseException:
            self.dashboard.close()
            raise
        self._frames_seen = 0
        self._frames_submitted = 0
        self._frames_skipped_no_client = 0
        self._closed = False
        self.dashboard.set_pipeline_state("ready", "GPU pipeline ready; open the dashboard")

    @property
    def url(self) -> str:
        return self.dashboard.url

    @property
    def should_close(self) -> bool:
        error = self.streamer.fatal_error
        if error is not None:
            self.dashboard.set_pipeline_state("error", error)
            return True
        return False

    def prompt_snapshot(self) -> PromptSnapshot:
        return self.dashboard.prompt_snapshot()

    def present(self, lease: GpuYuyvFrameLease | None) -> bool:
        self._frames_seen += 1
        self.streamer.drain_completed()
        if lease is None:
            return True
        if self.dashboard.video_client_count == 0:
            lease.release()
            self._frames_skipped_no_client += 1
            return True
        if self.streamer.submit(lease):
            self._frames_submitted += 1
            timing_mapping = self.streamer.timing_mapping
            if timing_mapping is not None:
                captured_monotonic_ns, media_pts_ns = timing_mapping
                self.dashboard.set_video_timing_mapping(
                    captured_monotonic_ns=captured_monotonic_ns,
                    media_pts_ns=media_pts_ns,
                    discontinuities=self.streamer.timestamp_discontinuities,
                )
        return self.streamer.fatal_error is None

    def info(self) -> dict[str, Any]:
        return {
            "type": "web",
            "url": self.url,
            "frames_seen": self._frames_seen,
            "frames_presented": self._frames_seen,
            "frames_submitted": self._frames_submitted,
            "frames_skipped_no_client": self._frames_skipped_no_client,
            "performance_hud_visible": True,
            "dashboard": self.dashboard.info(),
            "streamer": self.streamer.info(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.dashboard.set_pipeline_state("stopped", "Pipeline stopped")
        self.streamer.close()
        self.dashboard.close()
        self._closed = True
