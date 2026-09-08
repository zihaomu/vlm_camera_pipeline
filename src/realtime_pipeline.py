"""Low-latency camera scheduling, PyTorch detection, display, and metrics."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .camera_io import CameraReader, CapturedFrame, LatestFrameSlot


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True, slots=True)
class DetectorOutput:
    detections: tuple[Detection, ...]
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    external_nms_ms: float | None = None


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    sequence: int
    captured_ns: int
    completed_ns: int
    detections: tuple[Detection, ...]
    source_bgr: np.ndarray | None = None


class Detector(Protocol):
    backend_name: str

    def warmup(self, frame_bgr: np.ndarray, iterations: int = 2) -> None: ...

    def detect_bgr(self, frame_bgr: np.ndarray) -> DetectorOutput: ...


class LatestResultSlot:
    """Thread-safe depth-one result publication slot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: DetectionSnapshot | None = None

    def publish(self, result: DetectionSnapshot) -> None:
        with self._lock:
            if self._latest is not None and result.sequence <= self._latest.sequence:
                raise ValueError(
                    "result sequences must increase monotonically: "
                    f"latest={self._latest.sequence}, new={result.sequence}"
                )
            self._latest = result

    def latest(self) -> DetectionSnapshot | None:
        with self._lock:
            return self._latest


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


class _RateWindow:
    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_ns = int(window_seconds * 1e9)
        self._events: deque[int] = deque()

    def mark(self, timestamp_ns: int) -> None:
        self._events.append(timestamp_ns)
        self._trim(timestamp_ns)

    def rate(self, now_ns: int) -> float:
        self._trim(now_ns)
        if not self._events:
            return 0.0
        if len(self._events) == 1:
            return 1.0
        elapsed_ns = max(1, self._events[-1] - self._events[0])
        return (len(self._events) - 1) * 1e9 / elapsed_ns

    def _trim(self, now_ns: int) -> None:
        cutoff = now_ns - self.window_ns
        while self._events and self._events[0] < cutoff:
            self._events.popleft()


class MetricsCollector:
    """Collect bounded-rate counters and latency samples for one run."""

    def __init__(self, backend: str, *, started_ns: int | None = None) -> None:
        self.backend = backend
        self.started_ns = started_ns or time.monotonic_ns()
        self.finished_ns: int | None = None
        self._lock = threading.Lock()
        self.processed_frames = 0
        self.displayed_frames = 0
        self.skipped_before_inference = 0
        self.detection_count_latest = 0
        self.camera_read_failures = 0
        self._infer_rate = _RateWindow()
        self._display_rate = _RateWindow()
        # 100k samples retain a complete 30-minute 30 FPS gate while bounding
        # memory for an indefinitely running demo.
        self._capture_to_infer_ms: deque[float] = deque(maxlen=100_000)
        self._capture_to_display_ms: deque[float] = deque(maxlen=100_000)
        self._capture_to_infer_recent: deque[float] = deque(maxlen=300)
        self._capture_to_display_recent: deque[float] = deque(maxlen=300)
        self._preprocess_ms: deque[float] = deque(maxlen=100_000)
        self._inference_ms: deque[float] = deque(maxlen=100_000)
        self._postprocess_ms: deque[float] = deque(maxlen=100_000)
        self._external_nms_ms: deque[float] = deque(maxlen=100_000)
        self._display_ms: deque[float] = deque(maxlen=100_000)
        self._rss_bytes: deque[int] = deque(maxlen=3_600)
        self._gpu_power_watts: deque[float] = deque(maxlen=3_600)
        self._gpu_temperature_c: deque[float] = deque(maxlen=3_600)

    def record_inference(
        self,
        frame: CapturedFrame,
        output: DetectorOutput,
        completed_ns: int,
        *,
        skipped: int = 0,
    ) -> None:
        with self._lock:
            self.processed_frames += 1
            self.skipped_before_inference += max(0, skipped)
            self.detection_count_latest = len(output.detections)
            self._infer_rate.mark(completed_ns)
            capture_to_infer_ms = (completed_ns - frame.captured_ns) / 1e6
            self._capture_to_infer_ms.append(capture_to_infer_ms)
            self._capture_to_infer_recent.append(capture_to_infer_ms)
            self._append_optional(self._preprocess_ms, output.preprocess_ms)
            self._append_optional(self._inference_ms, output.inference_ms)
            self._append_optional(self._postprocess_ms, output.postprocess_ms)
            self._append_optional(self._external_nms_ms, output.external_nms_ms)

    def record_display(
        self,
        frame: CapturedFrame,
        displayed_ns: int,
        display_ms: float,
    ) -> None:
        with self._lock:
            self.displayed_frames += 1
            self._display_rate.mark(displayed_ns)
            capture_to_display_ms = (displayed_ns - frame.captured_ns) / 1e6
            self._capture_to_display_ms.append(capture_to_display_ms)
            self._capture_to_display_recent.append(capture_to_display_ms)
            self._display_ms.append(display_ms)

    def sample_system(self) -> None:
        rss = _current_rss_bytes()
        power = _read_first_scaled("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average", 1e6)
        temperature = _read_first_scaled(
            "/sys/class/drm/card*/device/hwmon/hwmon*/temp1_input", 1e3
        )
        with self._lock:
            self._rss_bytes.append(rss)
            if power is not None:
                self._gpu_power_watts.append(power)
            if temperature is not None:
                self._gpu_temperature_c.append(temperature)

    def live_summary(self, now_ns: int | None = None) -> dict[str, Any]:
        current_ns = now_ns or time.monotonic_ns()
        with self._lock:
            return {
                "backend": self.backend,
                "inference_fps": round(self._infer_rate.rate(current_ns), 2),
                "display_fps": round(self._display_rate.rate(current_ns), 2),
                "processed_frames": self.processed_frames,
                "skipped_before_inference": self.skipped_before_inference,
                "detection_count": self.detection_count_latest,
                "capture_to_infer_p95_ms": _round_optional(
                    _percentile(self._capture_to_infer_recent, 95)
                ),
                "capture_to_display_p50_ms": _round_optional(
                    _percentile(self._capture_to_display_recent, 50)
                ),
                "capture_to_display_p95_ms": _round_optional(
                    _percentile(self._capture_to_display_recent, 95)
                ),
            }

    def finish(
        self,
        *,
        camera_frames: int,
        camera_read_failures: int,
        camera_format: dict[str, Any] | None,
        exit_reason: str,
    ) -> dict[str, Any]:
        self.sample_system()
        finished_ns = time.monotonic_ns()
        with self._lock:
            self.finished_ns = finished_ns
            self.camera_read_failures = camera_read_failures
            elapsed_seconds = max(1e-9, (finished_ns - self.started_ns) / 1e9)
            dropped_frames = max(0, camera_frames - self.processed_frames)
            return {
                "schema_version": 1,
                "backend": self.backend,
                "exit_reason": exit_reason,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "camera_format": camera_format,
                "camera_frames": camera_frames,
                "processed_frames": self.processed_frames,
                "displayed_frames": self.displayed_frames,
                "dropped_frames": dropped_frames,
                "skipped_before_inference": self.skipped_before_inference,
                "camera_read_failures": camera_read_failures,
                "capture_fps": round(camera_frames / elapsed_seconds, 3),
                "inference_fps": round(self.processed_frames / elapsed_seconds, 3),
                "display_fps": round(self.displayed_frames / elapsed_seconds, 3),
                "latency_ms": {
                    "capture_to_infer": _summary(self._capture_to_infer_ms),
                    "capture_to_display": _summary(self._capture_to_display_ms),
                    "preprocess": _summary(self._preprocess_ms),
                    "inference": _summary(self._inference_ms),
                    "postprocess": _summary(self._postprocess_ms),
                    "external_nms": _summary(self._external_nms_ms),
                    "display": _summary(self._display_ms),
                },
                "process_rss_bytes": _summary_int(self._rss_bytes),
                "gpu_power_watts": _summary(self._gpu_power_watts),
                "gpu_temperature_c": _summary(self._gpu_temperature_c),
                "record": {"mode": "off", "queue_depth": 0},
                "vlm": {"mode": "off", "requests": 0, "failures": 0},
            }

    @staticmethod
    def write_json(path: str | Path, metrics: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)

    @staticmethod
    def _append_optional(target: deque[float], value: float | None) -> None:
        if value is not None and math.isfinite(value):
            target.append(float(value))


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": _round_optional(_percentile(values, 50)),
        "p95": _round_optional(_percentile(values, 95)),
        "p99": _round_optional(_percentile(values, 99)),
    }


def _summary_int(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {
            "count": 0,
            "first": None,
            "current": None,
            "minimum": None,
            "maximum": None,
            "growth": None,
        }
    return {
        "count": len(values),
        "first": values[0],
        "current": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "growth": values[-1] - values[0],
    }


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def _read_first_scaled(pattern: str, divisor: float) -> float | None:
    for candidate in glob.glob(pattern):
        try:
            return float(Path(candidate).read_text(encoding="ascii").strip()) / divisor
        except (OSError, ValueError):
            continue
    return None


class PyTorchCameraDetector:
    """Ultralytics `.pt` detector pinned to a ROCm CUDA-compatible device."""

    backend_name = "pytorch"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        image_size: int = 640,
        confidence: float = 0.5,
        iou: float = 0.45,
        half: bool = True,
        nms_mode: str = "auto",
    ) -> None:
        if nms_mode == "external":
            raise NotImplementedError(
                "Route P consumes Ultralytics Results and never adds external NMS"
            )
        if nms_mode not in {"auto", "off"}:
            raise ValueError(f"unsupported NMS mode for Route P: {nms_mode}")
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"model not found: {model_file}")

        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch ROCm device is unavailable; refusing CPU fallback")
        properties = torch.cuda.get_device_properties(0)
        architecture = getattr(properties, "gcnArchName", "")
        if not str(architecture).startswith("gfx1151"):
            raise RuntimeError(
                f"expected gfx1151 ROCm device, got {architecture!r}; refusing to run"
            )

        self._torch = torch
        self._model = YOLO(str(model_file))
        self.model_path = model_file.resolve()
        with model_file.open("rb") as model_stream:
            self.model_sha256 = hashlib.file_digest(model_stream, "sha256").hexdigest()
        self.gpu_architecture = str(architecture)
        self.gpu_name = torch.cuda.get_device_name(0)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.half = half
        self.nms_mode = nms_mode

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "torch": self._torch.__version__,
            "hip": self._torch.version.hip,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "gpu_arch": self.gpu_architecture,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "model_end2end": bool(getattr(self._model.model, "end2end", False)),
            "image_size": self.image_size,
            "half": self.half,
            "confidence": self.confidence,
            "iou": self.iou,
            "nms_mode": self.nms_mode,
            "external_nms": False,
        }

    def warmup(self, frame_bgr: np.ndarray, iterations: int = 2) -> None:
        for _ in range(max(1, iterations)):
            self.detect_bgr(frame_bgr)
        model_device = next(self._model.model.parameters()).device
        if model_device.type != "cuda":
            raise RuntimeError(
                f"Ultralytics model ended warmup on {model_device}; refusing CPU fallback"
            )
        if self.nms_mode == "off" and not bool(getattr(self._model.model, "end2end", False)):
            raise RuntimeError("--nms-mode off requires a model declaring end2end=True")

    def detect_bgr(self, frame_bgr: np.ndarray) -> DetectorOutput:
        self._torch.cuda.synchronize()
        results = self._model.predict(
            source=frame_bgr,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            half=self.half,
            verbose=False,
        )
        self._torch.cuda.synchronize()
        if len(results) != 1:
            raise RuntimeError(f"expected one Ultralytics result, got {len(results)}")
        result = results[0]
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidence = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            names = result.names
            for coordinates, score, class_id in zip(xyxy, confidence, classes):
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = (
                    str(names.get(int(class_id), class_id))
                    if isinstance(names, dict)
                    else str(names[int(class_id)])
                )
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(score),
                        class_id=int(class_id),
                        label=label,
                    )
                )
        speed = getattr(result, "speed", {}) or {}
        return DetectorOutput(
            detections=tuple(detections),
            preprocess_ms=_as_optional_float(speed.get("preprocess")),
            inference_ms=_as_optional_float(speed.get("inference")),
            postprocess_ms=_as_optional_float(speed.get("postprocess")),
            external_nms_ms=None,
        )


def _as_optional_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class InferenceWorker:
    """Consume only the latest frame and publish only the latest detections."""

    def __init__(
        self,
        frame_slot: LatestFrameSlot,
        result_slot: LatestResultSlot,
        detector: Detector,
        metrics: MetricsCollector,
        *,
        retain_source_frame: bool = False,
        initial_sequence: int = -1,
    ) -> None:
        self.frame_slot = frame_slot
        self.result_slot = result_slot
        self.detector = detector
        self.metrics = metrics
        self.retain_source_frame = retain_source_frame
        self.initial_sequence = initial_sequence
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("InferenceWorker is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-inference", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        last_sequence = self.initial_sequence
        try:
            while not self._stop_event.is_set():
                frame = self.frame_slot.consume_after(last_sequence, timeout=0.5)
                if frame is None:
                    if self.frame_slot.closed or self._stop_event.is_set():
                        return
                    continue
                skipped = max(0, frame.sequence - last_sequence - 1)
                last_sequence = frame.sequence
                output = self.detector.detect_bgr(frame.bgr)
                completed_ns = time.monotonic_ns()
                self.metrics.record_inference(frame, output, completed_ns, skipped=skipped)
                self.result_slot.publish(
                    DetectionSnapshot(
                        sequence=frame.sequence,
                        captured_ns=frame.captured_ns,
                        completed_ns=completed_ns,
                        detections=output.detections,
                        source_bgr=frame.bgr if self.retain_source_frame else None,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - propagate worker failures to the owner.
            self._error = exc
            self._stop_event.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError(f"inference worker did not stop within {timeout:g} seconds")

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("inference worker failed") from self._error


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    display: bool = True
    display_mode: str = "live"
    max_latency_ms: float = 150.0
    metrics_json: str = "output/realtime/metrics.json"
    duration_seconds: float | None = None
    warmup_iterations: int = 2
    window_name: str = "YOLO26 on Ryzen AI MAX+ 395"

    def __post_init__(self) -> None:
        if self.display_mode not in {"live", "processed"}:
            raise ValueError("display_mode must be 'live' or 'processed'")
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive when provided")
        if self.warmup_iterations < 1:
            raise ValueError("warmup_iterations must be at least one")


class RealtimePipeline:
    """Orchestrate camera, inference, and main-thread OpenCV display."""

    def __init__(
        self,
        camera: CameraReader,
        detector: Detector,
        config: PipelineConfig,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.config = config
        self.result_slot = LatestResultSlot()
        self.metrics = MetricsCollector(detector.backend_name)
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> dict[str, Any]:
        import cv2

        exit_reason = "requested"
        camera_baseline = 0
        worker: InferenceWorker | None = None
        self.camera.start()
        try:
            first_frame = self.camera.slot.consume_after(-1, timeout=5.0)
            if first_frame is None:
                self.camera.raise_if_failed()
                raise RuntimeError("camera did not publish its first frame within 5 seconds")

            self.detector.warmup(first_frame.bgr, self.config.warmup_iterations)
            latest_after_warmup = self.camera.slot.latest()
            assert latest_after_warmup is not None
            camera_baseline = self.camera.published_frames
            self.metrics = MetricsCollector(self.detector.backend_name)
            worker = InferenceWorker(
                self.camera.slot,
                self.result_slot,
                self.detector,
                self.metrics,
                retain_source_frame=self.config.display_mode == "processed",
                initial_sequence=latest_after_warmup.sequence,
            )
            worker.start()
            exit_reason = self._main_loop(cv2, worker)
        except KeyboardInterrupt:
            exit_reason = "keyboard_interrupt"
        finally:
            self._stop_event.set()
            stop_errors: list[Exception] = []
            try:
                self.camera.stop()
            except Exception as exc:  # noqa: BLE001 - complete remaining cleanup first.
                stop_errors.append(exc)
            if worker is not None:
                try:
                    worker.stop()
                    worker.raise_if_failed()
                except Exception as exc:  # noqa: BLE001 - complete remaining cleanup first.
                    stop_errors.append(exc)
            if self.config.display:
                try:
                    cv2.destroyWindow(self.config.window_name)
                except cv2.error:
                    cv2.destroyAllWindows()

            negotiated = self.camera.negotiated
            camera_format = (
                {
                    "width": negotiated.width,
                    "height": negotiated.height,
                    "fps": negotiated.fps,
                    "fourcc": negotiated.fourcc,
                }
                if negotiated is not None
                else None
            )
            metrics = self.metrics.finish(
                camera_frames=max(0, self.camera.published_frames - camera_baseline),
                camera_read_failures=self.camera.read_failures,
                camera_format=camera_format,
                exit_reason=exit_reason,
            )
            runtime_info = getattr(self.detector, "runtime_info", None)
            metrics["detector"] = (
                runtime_info()
                if callable(runtime_info)
                else {"backend": self.detector.backend_name}
            )
            metrics["pipeline"] = {
                "display": self.config.display,
                "display_mode": self.config.display_mode,
                "max_latency_ms": self.config.max_latency_ms,
                "warmup_iterations": self.config.warmup_iterations,
            }
            MetricsCollector.write_json(self.config.metrics_json, metrics)
            if stop_errors:
                raise RuntimeError("pipeline shutdown failed") from stop_errors[0]
        return metrics

    def _main_loop(self, cv2: Any, worker: InferenceWorker) -> str:
        last_live_sequence = -1
        last_processed_sequence = -1
        next_report = time.monotonic() + 1.0
        started = time.monotonic()
        while not self._stop_event.is_set():
            if (
                self.config.duration_seconds is not None
                and time.monotonic() - started >= self.config.duration_seconds
            ):
                return "duration_elapsed"

            self.camera.raise_if_failed()
            worker.raise_if_failed()
            live_frame = self.camera.slot.consume_after(last_live_sequence, timeout=0.1)
            if live_frame is None:
                continue
            last_live_sequence = live_frame.sequence

            display_frame = live_frame
            result = self.result_slot.latest()
            if self.config.display_mode == "processed":
                if (
                    result is None
                    or result.sequence <= last_processed_sequence
                    or result.source_bgr is None
                ):
                    self._report_if_due(next_report)
                    if time.monotonic() >= next_report:
                        next_report = time.monotonic() + 1.0
                    continue
                last_processed_sequence = result.sequence
                display_frame = CapturedFrame(
                    result.sequence, result.captured_ns, result.source_bgr
                )

            now_ns = time.monotonic_ns()
            detection_age_ms = (
                (now_ns - result.captured_ns) / 1e6 if result is not None else math.inf
            )
            result_is_fresh = detection_age_ms <= self.config.max_latency_ms
            if self.config.display_mode == "processed":
                # Correctness/replay mode always binds boxes to their source frame.
                result_is_fresh = True
            compatible_result = (
                result if result is not None and result.sequence <= display_frame.sequence else None
            )
            usable_result = (
                compatible_result if compatible_result is not None and result_is_fresh else None
            )

            if self.config.display:
                live_metrics = self.metrics.live_summary(now_ns)
                live_metrics["capture_fps"] = round(self.camera.capture_fps, 2)
                preview = draw_on_host_frame(
                    display_frame.bgr,
                    usable_result.detections if usable_result else (),
                    live_metrics,
                    detection_sequence=(
                        compatible_result.sequence if compatible_result is not None else None
                    ),
                    detection_age_ms=(detection_age_ms if compatible_result is not None else None),
                )
                display_started_ns = time.monotonic_ns()
                cv2.imshow(self.config.window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                displayed_ns = time.monotonic_ns()
                self.metrics.record_display(
                    display_frame,
                    displayed_ns,
                    (displayed_ns - display_started_ns) / 1e6,
                )
                if key in {ord("q"), 27}:
                    return "user_exit"

            now = time.monotonic()
            if now >= next_report:
                self.metrics.sample_system()
                print(json.dumps(self.metrics.live_summary(), sort_keys=True), flush=True)
                next_report = now + 1.0
        return "requested"

    def _report_if_due(self, next_report: float) -> None:
        if time.monotonic() >= next_report:
            self.metrics.sample_system()
            print(json.dumps(self.metrics.live_summary(), sort_keys=True), flush=True)


def draw_on_host_frame(
    frame_bgr: np.ndarray,
    detections: Sequence[Detection],
    live_metrics: dict[str, Any],
    *,
    detection_sequence: int | None,
    detection_age_ms: float | None,
) -> np.ndarray:
    """Copy and annotate a host BGR frame; the captured input stays read-only."""

    import cv2

    preview = frame_bgr.copy()
    height, width = preview.shape[:2]
    for detection in detections:
        x1 = max(0, min(width - 1, round(detection.x1)))
        y1 = max(0, min(height - 1, round(detection.y1)))
        x2 = max(0, min(width - 1, round(detection.x2)))
        y2 = max(0, min(height - 1, round(detection.y2)))
        color = _class_color(detection.class_id)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            preview,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    age_text = "n/a" if detection_age_ms is None else f"{detection_age_ms:.1f} ms"
    sequence_text = "n/a" if detection_sequence is None else str(detection_sequence)
    lines = [
        f"backend: {live_metrics.get('backend', 'unknown')}",
        (
            f"capture/infer/display FPS: {live_metrics.get('capture_fps', 0):.1f} / "
            f"{live_metrics.get('inference_fps', 0):.1f} / "
            f"{live_metrics.get('display_fps', 0):.1f}"
        ),
        f"detections: {len(detections)}  result seq/age: {sequence_text} / {age_text}",
        f"dropped before inference: {live_metrics.get('skipped_before_inference', 0)}",
        (
            "capture-to-display p50/p95: "
            f"{_format_metric(live_metrics.get('capture_to_display_p50_ms'))} / "
            f"{_format_metric(live_metrics.get('capture_to_display_p95_ms'))} ms"
        ),
    ]
    panel_height = 24 * len(lines) + 10
    panel_width = min(width - 5, 720)
    # One captured-frame copy is required for drawing; avoid a second full-frame
    # overlay allocation in the hot UI loop.
    cv2.rectangle(preview, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (14, 28 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return preview


def _class_color(class_id: int) -> tuple[int, int, int]:
    return (
        int((37 * class_id + 80) % 205 + 50),
        int((17 * class_id + 130) % 205 + 50),
        int((29 * class_id + 30) % 205 + 50),
    )


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"
