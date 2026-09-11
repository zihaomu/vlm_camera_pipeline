"""YOLO-guided VLM image composition with bounded GPU/CPU control metadata."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .zerocopy_vlm import HipIpcImageLease, _HipRuntime
from .zerocopy_yolo import GpuDetectionLease

HYBRID_INPUT_MODE = "hybrid-numbered-boxes-v1"
HYBRID_MAX_HINTS = 8
HYBRID_HINT_RECORD_BYTES = 32
HYBRID_HINT_BUFFER_BYTES = 260
COCO80_MANIFEST = "native-lock/coco80-classes.json"
WEB_PROMPT_MAX_CHARACTERS = 512
DEFAULT_WEB_PROMPT = "Describe the most important visible objects and actions concisely."
_SAFE_CLASS_NAME = re.compile(r"[a-z0-9 ]{1,32}")


class ZeroCopyHybridError(RuntimeError):
    """The strict Hybrid image/prompt contract was violated."""


class _HybridHintRecord(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("class_id", ctypes.c_int32),
        ("source_index", ctypes.c_int32),
        ("rank", ctypes.c_int32),
    ]


class _HybridHintBuffer(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("records", _HybridHintRecord * HYBRID_MAX_HINTS),
    ]


if ctypes.sizeof(_HybridHintRecord) != HYBRID_HINT_RECORD_BYTES:
    raise RuntimeError("Python HybridHintRecord ABI size changed")
if ctypes.sizeof(_HybridHintBuffer) != HYBRID_HINT_BUFFER_BYTES:
    raise RuntimeError("Python HybridHintBuffer ABI size changed")


@dataclass(frozen=True, slots=True)
class HybridHint:
    rank: int
    class_id: int
    class_name: str
    confidence: float
    source_index: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class HybridComposition:
    frame_id: int
    prompt: str
    hints: tuple[HybridHint, ...]
    compose_ms: float
    control_metadata_d2h_bytes: int
    input_mode: str = HYBRID_INPUT_MODE
    prompt_version: int = 0


def load_coco80_manifest(
    path: str | Path,
    *,
    model_names: Mapping[int, str] | None = None,
) -> tuple[tuple[str, ...], str]:
    manifest_path = Path(path).resolve()
    try:
        raw = manifest_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ZeroCopyHybridError(f"invalid COCO80 manifest: {manifest_path}") from exc
    names = payload.get("names") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("dataset") != "COCO80"
        or not isinstance(names, list)
        or len(names) != 80
        or any(not isinstance(name, str) or _SAFE_CLASS_NAME.fullmatch(name) is None for name in names)
        or len(set(names)) != 80
    ):
        raise ZeroCopyHybridError("COCO80 canonical manifest contract mismatch")
    locked_names = tuple(names)
    if model_names is not None:
        normalized = {int(key): str(value) for key, value in model_names.items()}
        expected = {index: name for index, name in enumerate(locked_names)}
        if normalized != expected:
            raise ZeroCopyHybridError("YOLO model class metadata differs from COCO80 manifest")
    return locked_names, hashlib.sha256(raw).hexdigest()


def build_hybrid_prompt(
    hints: Sequence[HybridHint],
    *,
    max_hints: int = HYBRID_MAX_HINTS,
    user_prompt: str | None = None,
) -> str:
    if not 1 <= max_hints <= HYBRID_MAX_HINTS:
        raise ValueError(f"max_hints must be between 1 and {HYBRID_MAX_HINTS}")
    if len(hints) > max_hints:
        raise ZeroCopyHybridError("Hybrid prompt received more hints than its fixed bound")
    entries: list[str] = []
    for expected_rank, hint in enumerate(hints, start=1):
        if hint.rank != expected_rank:
            raise ZeroCopyHybridError("Hybrid hint ranks must be contiguous and one-based")
        if not 0 <= hint.class_id < 80 or _SAFE_CLASS_NAME.fullmatch(hint.class_name) is None:
            raise ZeroCopyHybridError("Hybrid hint contains an invalid canonical class")
        if not math.isfinite(hint.confidence) or not 0.0 <= hint.confidence <= 1.0:
            raise ZeroCopyHybridError("Hybrid hint confidence is invalid")
        entries.append(f"#{hint.rank} {hint.class_name} {hint.confidence:.2f}")
    rendered_hints = "; ".join(entries) if entries else "none"
    if user_prompt is None:
        prompt = (
            "Answer in at most 16 English words. Visually describe the main objects/action in the "
            f"whole image. Object hints may be wrong/incomplete: {rendered_hints}. "
            "Check unboxed regions; never mention hints, boxes, IDs, scores, or speculate."
        )
    else:
        normalized = validate_web_prompt(user_prompt)
        if any("\u3400" <= character <= "\u9fff" for character in normalized):
            prompt = (
                f"{normalized}\n\n"
                f"同一画面的目标提示（可能错误或不完整）：{rendered_hints}。"
                "请结合完整画面核实，也检查未框选区域；不要提及提示、框、编号或分数。"
                "严格使用用户要求的语言和格式，回答简洁。"
            )
        else:
            prompt = (
                f"{normalized}\n\n"
                f"Object hints for this exact image may be wrong/incomplete: {rendered_hints}. "
                "Visually verify them and inspect unboxed regions. Never mention hints, boxes, IDs, "
                "or scores. Follow the language and format requested above; keep the response concise."
            )
    if len(prompt) > 2048:
        raise ZeroCopyHybridError("Hybrid prompt exceeds the fixed 2048-character bound")
    return prompt


def validate_web_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("web prompt must be a string")
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("web prompt must not be empty")
    if len(normalized) > WEB_PROMPT_MAX_CHARACTERS:
        raise ValueError(
            f"web prompt exceeds the {WEB_PROMPT_MAX_CHARACTERS}-character limit"
        )
    if "\x00" in normalized or any(
        ord(character) < 32 and character not in "\n\r\t" for character in normalized
    ):
        raise ValueError("web prompt contains an unsupported control character")
    return normalized


class _HybridKernels:
    def __init__(self, library: str | Path) -> None:
        self.path = Path(library).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"zero-copy HIP kernel library not found: {self.path}; "
                "run scripts/setup_zerocopy_kernels_gfx1151.sh"
            )
        self._library = ctypes.CDLL(str(self.path))
        library_handle = self._library
        library_handle.vlm_camera_zerocopy_last_error.argtypes = []
        library_handle.vlm_camera_zerocopy_last_error.restype = ctypes.c_char_p
        library_handle.vlm_camera_hybrid_max_hints.argtypes = []
        library_handle.vlm_camera_hybrid_max_hints.restype = ctypes.c_int
        library_handle.vlm_camera_hybrid_hint_record_size.argtypes = []
        library_handle.vlm_camera_hybrid_hint_record_size.restype = ctypes.c_size_t
        library_handle.vlm_camera_hybrid_hint_buffer_size.argtypes = []
        library_handle.vlm_camera_hybrid_hint_buffer_size.restype = ctypes.c_size_t
        library_handle.vlm_camera_hybrid_compact_and_overlay_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library_handle.vlm_camera_hybrid_compact_and_overlay_f32.restype = ctypes.c_int
        library_handle.vlm_camera_hybrid_compact_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library_handle.vlm_camera_hybrid_compact_f32.restype = ctypes.c_int
        native_abi = (
            int(library_handle.vlm_camera_hybrid_max_hints()),
            int(library_handle.vlm_camera_hybrid_hint_record_size()),
            int(library_handle.vlm_camera_hybrid_hint_buffer_size()),
        )
        if native_abi != (
            HYBRID_MAX_HINTS,
            HYBRID_HINT_RECORD_BYTES,
            HYBRID_HINT_BUFFER_BYTES,
        ):
            raise ZeroCopyHybridError(f"native/Python Hybrid ABI mismatch: {native_abi}")

    def compose(
        self,
        *,
        image_pointer: int,
        image_width: int,
        image_height: int,
        source_width: int,
        source_height: int,
        resized_width: int,
        resized_height: int,
        pad_left: int,
        pad_top: int,
        detections_pointer: int,
        detection_count: int,
        confidence: float,
        max_hints: int,
        hint_buffer_pointer: int,
        base_ready_event: int,
        detections_ready_event: int,
        stream: int,
    ) -> None:
        status = self._library.vlm_camera_hybrid_compact_and_overlay_f32(
            ctypes.c_void_p(image_pointer),
            image_width,
            image_height,
            source_width,
            source_height,
            resized_width,
            resized_height,
            pad_left,
            pad_top,
            ctypes.c_void_p(detections_pointer),
            detection_count,
            confidence,
            max_hints,
            ctypes.c_void_p(hint_buffer_pointer),
            ctypes.c_void_p(base_ready_event),
            ctypes.c_void_p(detections_ready_event),
            ctypes.c_void_p(stream),
        )
        if status:
            detail = self._library.vlm_camera_zerocopy_last_error()
            decoded = detail.decode("utf-8", "replace") if detail else f"status {status}"
            raise ZeroCopyHybridError(f"Hybrid GPU composition failed: {decoded}")

    def compact(
        self,
        *,
        detections_pointer: int,
        detection_count: int,
        confidence: float,
        max_hints: int,
        hint_buffer_pointer: int,
        detections_ready_event: int,
        stream: int,
    ) -> None:
        status = self._library.vlm_camera_hybrid_compact_f32(
            ctypes.c_void_p(detections_pointer),
            detection_count,
            confidence,
            max_hints,
            ctypes.c_void_p(hint_buffer_pointer),
            ctypes.c_void_p(detections_ready_event),
            ctypes.c_void_p(stream),
        )
        if status:
            detail = self._library.vlm_camera_zerocopy_last_error()
            decoded = detail.decode("utf-8", "replace") if detail else f"status {status}"
            raise ZeroCopyHybridError(f"Hybrid GPU compaction failed: {decoded}")


class _HybridHintSlot:
    __slots__ = (
        "control_ready_event",
        "device_pointer",
        "generation",
        "host_pointer",
        "in_use",
        "lock",
        "runtime",
    )

    def __init__(self, runtime: _HipRuntime) -> None:
        self.runtime = runtime
        self.device_pointer = runtime.allocate(HYBRID_HINT_BUFFER_BYTES)
        self.host_pointer = 0
        self.control_ready_event = 0
        try:
            self.host_pointer = runtime.host_allocate(HYBRID_HINT_BUFFER_BYTES)
            self.control_ready_event = runtime.create_local_event()
        except BaseException:
            if self.control_ready_event:
                runtime.destroy_event(self.control_ready_event)
            if self.host_pointer:
                runtime.host_free(self.host_pointer)
            runtime.free(self.device_pointer)
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
            if generation == self.generation:
                self.in_use = False

    def close(self) -> None:
        with self.lock:
            if self.in_use:
                raise ZeroCopyHybridError("cannot close an in-use Hybrid hint slot")
            event = self.control_ready_event
            host_pointer = self.host_pointer
            device_pointer = self.device_pointer
            self.control_ready_event = 0
            self.host_pointer = 0
            self.device_pointer = 0
        if event:
            self.runtime.destroy_event(event)
        if host_pointer:
            self.runtime.host_free(host_pointer)
        if device_pointer:
            self.runtime.free(device_pointer)


def _parse_hint_buffer(
    host_pointer: int,
    *,
    class_names: Sequence[str],
    max_hints: int,
) -> tuple[HybridHint, ...]:
    buffer = _HybridHintBuffer.from_address(host_pointer)
    count = int(buffer.count)
    if not 0 <= count <= max_hints:
        raise ZeroCopyHybridError(f"Hybrid GPU returned invalid hint count: {count}")
    hints: list[HybridHint] = []
    for index in range(count):
        record = buffer.records[index]
        class_id = int(record.class_id)
        rank = int(record.rank)
        confidence = float(record.confidence)
        coordinates = (
            float(record.x1),
            float(record.y1),
            float(record.x2),
            float(record.y2),
        )
        if (
            rank != index + 1
            or not 0 <= class_id < len(class_names)
            or not math.isfinite(confidence)
            or not all(math.isfinite(value) for value in coordinates)
        ):
            raise ZeroCopyHybridError("Hybrid GPU returned malformed control metadata")
        hints.append(
            HybridHint(
                rank=rank,
                class_id=class_id,
                class_name=class_names[class_id],
                confidence=confidence,
                source_index=int(record.source_index),
                x1=coordinates[0],
                y1=coordinates[1],
                x2=coordinates[2],
                y2=coordinates[3],
            )
        )
    return tuple(hints)


class HybridComposer:
    """Compose exact-frame numbered boxes and return only bounded prompt metadata."""

    def __init__(
        self,
        *,
        kernel_library: str | Path,
        coco_manifest: str | Path,
        model_names: Mapping[int, str],
        confidence: float = 0.5,
        max_hints: int = HYBRID_MAX_HINTS,
        pool_size: int = 2,
        device_id: int = 0,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Hybrid confidence must be between 0 and 1")
        if not 1 <= max_hints <= HYBRID_MAX_HINTS:
            raise ValueError(f"Hybrid max_hints must be between 1 and {HYBRID_MAX_HINTS}")
        if pool_size < 2:
            raise ValueError("Hybrid hint pool must contain at least two slots")
        self.class_names, self.manifest_sha256 = load_coco80_manifest(
            coco_manifest, model_names=model_names
        )
        self.runtime = _HipRuntime(device_id)
        self.kernels = _HybridKernels(kernel_library)
        self.confidence = confidence
        self.max_hints = max_hints
        self._slots: list[_HybridHintSlot] = []
        try:
            self._slots = [_HybridHintSlot(self.runtime) for _ in range(pool_size)]
        except BaseException:
            for slot in self._slots:
                slot.close()
            raise
        self._closed = False
        self._composed = 0
        self._dropped_no_slot = 0
        self._control_metadata_d2h_bytes = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _parse_hints(self, host_pointer: int) -> tuple[HybridHint, ...]:
        return _parse_hint_buffer(
            host_pointer,
            class_names=self.class_names,
            max_hints=self.max_hints,
        )

    def finalize(
        self,
        *,
        frame_id: int,
        image_lease: HipIpcImageLease,
        detection: GpuDetectionLease,
        stream: int,
        user_prompt: str | None = None,
        prompt_version: int = 0,
    ) -> HybridComposition:
        if self._closed:
            raise ZeroCopyHybridError("Hybrid composer is closed")
        if frame_id != detection.frame_id:
            raise ZeroCopyHybridError(
                f"Hybrid exact-frame mismatch: image={frame_id}, detection={detection.frame_id}"
            )
        if image_lease.released:
            raise ZeroCopyHybridError("Hybrid received a released VLM image lease")
        if image_lease.final_ready:
            raise ZeroCopyHybridError("Hybrid image was already published to llama.cpp")
        slot = next((candidate for candidate in self._slots if candidate.try_acquire()), None)
        if slot is None:
            self._dropped_no_slot += 1
            raise ZeroCopyHybridError("all fixed Hybrid hint slots are busy")
        generation = slot.generation
        started = time.perf_counter()
        try:
            geometry = image_lease.geometry
            self.kernels.compose(
                image_pointer=image_lease.device_pointer,
                image_width=geometry.target_width,
                image_height=geometry.target_height,
                source_width=geometry.source_width,
                source_height=geometry.source_height,
                resized_width=geometry.resized_width,
                resized_height=geometry.resized_height,
                pad_left=geometry.pad_left,
                pad_top=geometry.pad_top,
                detections_pointer=detection.tensor.data_ptr(),
                detection_count=int(detection.tensor.shape[0]),
                confidence=self.confidence,
                max_hints=self.max_hints,
                hint_buffer_pointer=slot.device_pointer,
                base_ready_event=image_lease.base_ready_event,
                detections_ready_event=detection.ready_event.cuda_event,
                stream=stream,
            )
            self.runtime.copy_device_to_host_async(
                slot.host_pointer,
                slot.device_pointer,
                HYBRID_HINT_BUFFER_BYTES,
                stream,
            )
            self.runtime.record_event(slot.control_ready_event, stream)
            image_lease.mark_final_ready(stream)
            self.runtime.synchronize_event(slot.control_ready_event)
            hints = self._parse_hints(slot.host_pointer)
            prompt = build_hybrid_prompt(
                hints,
                max_hints=self.max_hints,
                user_prompt=user_prompt,
            )
            compose_ms = (time.perf_counter() - started) * 1000.0
            self._composed += 1
            self._control_metadata_d2h_bytes += HYBRID_HINT_BUFFER_BYTES
            return HybridComposition(
                frame_id=frame_id,
                prompt=prompt,
                hints=hints,
                compose_ms=compose_ms,
                control_metadata_d2h_bytes=HYBRID_HINT_BUFFER_BYTES,
                prompt_version=prompt_version,
            )
        except BaseException:
            if not image_lease.final_ready:
                image_lease.release()
            raise
        finally:
            slot.release(generation)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "input_mode": HYBRID_INPUT_MODE,
            "max_hints": self.max_hints,
            "confidence": self.confidence,
            "hint_record_bytes": HYBRID_HINT_RECORD_BYTES,
            "hint_buffer_bytes": HYBRID_HINT_BUFFER_BYTES,
            "hint_pool_size": len(self._slots),
            "hint_device_pointers": [slot.device_pointer for slot in self._slots],
            "hint_host_pointers": [slot.host_pointer for slot in self._slots],
            "control_ready_events": [slot.control_ready_event for slot in self._slots],
            "composed": self._composed,
            "dropped_no_slot": self._dropped_no_slot,
            "control_metadata_d2h_bytes": self._control_metadata_d2h_bytes,
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "full_detection_tensor_d2h_bytes": 0,
            "uses_global_device_synchronize": False,
            "class_count": len(self.class_names),
            "coco80_manifest_sha256": self.manifest_sha256,
            "kernel_library": str(self.kernels.path),
            "kernel_library_sha256": hashlib.sha256(self.kernels.path.read_bytes()).hexdigest(),
        }

    def close(self) -> None:
        if self._closed:
            return
        for slot in self._slots:
            slot.close()
        self._closed = True


def default_coco80_manifest(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / COCO80_MANIFEST


def assert_presenter_coco80_source(workspace: str | Path, class_names: Sequence[str]) -> None:
    """Fail preflight when the compiled presenter source no longer matches the manifest."""
    source = (Path(workspace).resolve() / "native/egl_present/egl_present.hip").read_text(
        encoding="utf-8"
    )
    marker = "constexpr std::array<const char *, kDetectionClassCount> kCocoClassNames = {"
    try:
        body = source.split(marker, 1)[1].split("};", 1)[0]
    except IndexError as exc:
        raise ZeroCopyHybridError("presenter COCO80 class table is unavailable") from exc
    compiled_names = tuple(re.findall(r'"([a-z0-9 ]+)"', body))
    if compiled_names != tuple(class_names):
        raise ZeroCopyHybridError("presenter class atlas differs from COCO80 manifest")
