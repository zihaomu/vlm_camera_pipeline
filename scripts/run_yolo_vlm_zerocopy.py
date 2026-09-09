#!/usr/bin/env python3
"""Run the strict GPU-resident camera + YOLO26 + Qwen3-VL + EGL demo."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ["ULTRALYTICS_MIGRAPHX_STRICT"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.zerocopy_camera import (
    StrictGpuCamera,
    ZeroCopyCameraError,
    camera_component_preflight,
)
from src.zerocopy_present import EglHipPresenter
from src.zerocopy_vlm import HipIpcImageLease, LlamaCppIpcConfig, ZeroCopyLlamaCppVlm
from src.zerocopy_yolo import GpuDetectionLease, PreparedYoloFrame, StrictMIGraphXYolo

CAMERA_LIBRARY = ".local/zerocopy-gfx1151/lib/libvlm_camera_gpu_capture.so.0.1.0"
KERNEL_LIBRARY = ".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0"
PRESENTER_LIBRARY = ".local/egl-present-gfx1151/lib/libvlm_camera_egl_present.so.0.1.0"
PERFORMANCE_HUD_UPDATE_SECONDS = 0.5
YOLO_HUD_ROLLING_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class VlmResult:
    request_id: int
    source_frame_id: int
    source_captured_ns: int
    scheduled_at: float
    completed_at: float
    caption: str | None
    error: str | None

    @property
    def latency_ms(self) -> float:
        return (self.completed_at - self.scheduled_at) * 1000.0


@dataclass(frozen=True, slots=True)
class _VlmTask:
    request_id: int
    source_frame_id: int
    source_captured_ns: int
    scheduled_at: float
    lease: HipIpcImageLease


class LatestOnlyVlmWorker:
    """One in-flight request, no backlog, with ownership transfer of IPC leases."""

    def __init__(self, engine: ZeroCopyLlamaCppVlm) -> None:
        self._engine = engine
        self._queue: queue.Queue[_VlmTask | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._outstanding = False
        self._closed = False
        self._latest: VlmResult | None = None
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped_busy = 0
        self._max_queue_depth = 0
        self._thread = threading.Thread(target=self._run, name="vlm-ipc-worker", daemon=True)
        self._thread.start()

    def submit(
        self,
        *,
        source_frame_id: int,
        source_captured_ns: int,
        scheduled_at: float,
        lease: HipIpcImageLease,
    ) -> bool:
        with self._lock:
            if self._closed:
                lease.release()
                raise RuntimeError("VLM worker is closed")
            if self._outstanding:
                self._dropped_busy += 1
                lease.release()
                return False
            self._outstanding = True
            self._submitted += 1
            request_id = self._submitted
        self._queue.put_nowait(
            _VlmTask(
                request_id=request_id,
                source_frame_id=source_frame_id,
                source_captured_ns=source_captured_ns,
                scheduled_at=scheduled_at,
                lease=lease,
            )
        )
        self._max_queue_depth = max(self._max_queue_depth, 1)
        return True

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            caption = None
            error = None
            try:
                caption = self._engine.caption_prepared(task.lease)
            except Exception as exc:  # noqa: BLE001 - surfaced through result/metrics
                task.lease.release()
                error = f"{type(exc).__name__}: {exc}"
            completed_at = time.monotonic()
            with self._lock:
                self._latest = VlmResult(
                    request_id=task.request_id,
                    source_frame_id=task.source_frame_id,
                    source_captured_ns=task.source_captured_ns,
                    scheduled_at=task.scheduled_at,
                    completed_at=completed_at,
                    caption=caption,
                    error=error,
                )
                self._completed += 1
                self._failed += int(error is not None)
                self._outstanding = False
            self._queue.task_done()

    def latest_after(self, request_id: int) -> VlmResult | None:
        with self._lock:
            if self._latest is None or self._latest.request_id <= request_id:
                return None
            return self._latest

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "dropped_busy": self._dropped_busy,
                "outstanding": self._outstanding,
                "queue_depth": self._queue.qsize(),
                "max_queue_depth": self._max_queue_depth,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()


@dataclass(frozen=True, slots=True)
class YoloResult:
    sequence_id: int
    source_frame_id: int
    preprocess_ms: float
    completed_at: float
    detection: GpuDetectionLease | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _YoloTask:
    sequence_id: int
    prepared: PreparedYoloFrame


class LatestOnlyYoloWorker:
    """Run MIGraphX off the UI thread with one newest pending frame and no backlog."""

    def __init__(self, engine: StrictMIGraphXYolo) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: _YoloTask | None = None
        self._running = False
        self._closed = False
        self._latest: YoloResult | None = None
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped_busy = 0
        self._dropped_superseded = 0
        self._max_queue_depth = 0
        self._thread = threading.Thread(target=self._run, name="yolo-migraphx-worker", daemon=True)
        self._thread.start()

    def submit(self, prepared: PreparedYoloFrame) -> bool:
        superseded: _YoloTask | None = None
        with self._condition:
            if self._closed:
                prepared.release()
                raise RuntimeError("YOLO worker is closed")
            self._submitted += 1
            sequence_id = self._submitted
            if self._pending is not None:
                superseded = self._pending
                self._dropped_superseded += 1
            self._pending = _YoloTask(sequence_id=sequence_id, prepared=prepared)
            self._max_queue_depth = max(self._max_queue_depth, 1)
            self._condition.notify()
        if superseded is not None:
            superseded.prepared.release()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                task = self._pending
                self._pending = None
                self._running = True
            assert task is not None
            detection = None
            error = None
            try:
                detection = self._engine.infer_prepared(task.prepared)
            except Exception as exc:  # noqa: BLE001 - surfaced on the strict main thread
                task.prepared.release()
                error = f"{type(exc).__name__}: {exc}"
            result = YoloResult(
                sequence_id=task.sequence_id,
                source_frame_id=task.prepared.frame_id,
                preprocess_ms=task.prepared.preprocess_ms,
                completed_at=time.monotonic(),
                detection=detection,
                error=error,
            )
            with self._lock:
                superseded = self._latest
                self._latest = result
                self._completed += 1
                self._failed += int(error is not None)
                self._running = False
            if superseded is not None and superseded.detection is not None:
                superseded.detection.release()

    def take_latest(self) -> YoloResult | None:
        with self._lock:
            result = self._latest
            self._latest = None
            return result

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "dropped_busy": self._dropped_busy,
                "dropped_superseded": self._dropped_superseded,
                "running": self._running,
                "pending": self._pending is not None,
                "outstanding": self._running or self._pending is not None,
                "latest_ready": self._latest is not None,
                "queue_depth": int(self._pending is not None),
                "max_queue_depth": self._max_queue_depth,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._thread.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-buffers", type=int, default=4)
    parser.add_argument("--clean-rgb-buffers", type=int, default=3)
    parser.add_argument("--yolo-model", default="models/yolo26x.onnx")
    parser.add_argument(
        "--yolo-cache",
        default="models/ort-migraphx-cache/gfx1151-yolo26x-strict-iobinding-v1",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--vlm-interval", type=float, default=3.0)
    parser.add_argument("--vlm-max-tokens", type=int, default=32)
    parser.add_argument("--window-scale", type=float, default=0.5)
    parser.add_argument(
        "--show-performance",
        "--show-speed",
        dest="show_performance",
        action="store_true",
        help="Show the YOLO/VLM performance HUD at startup; press H to toggle it",
    )
    parser.add_argument("--hidden", action="store_true", help="Create a hidden EGL window")
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until window close")
    parser.add_argument("--camera-timeout-ms", type=int, default=2000)
    parser.add_argument("--zero-copy", choices=("require",), default="require")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--metrics",
        default="output/realtime/metrics-yolo-vlm-zerocopy.jsonl",
    )
    return parser


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def rolling_yolo_performance(
    samples: deque[tuple[float, float]],
    *,
    now: float,
    window_seconds: float = YOLO_HUD_ROLLING_SECONDS,
) -> tuple[float | None, float | None]:
    cutoff = now - window_seconds
    while samples and samples[0][0] < cutoff:
        samples.popleft()
    if not samples:
        return None, None
    inference_ms = statistics.median(sample[1] for sample in samples)
    if len(samples) < 2:
        return None, inference_ms
    elapsed = samples[-1][0] - samples[0][0]
    fps = (len(samples) - 1) / elapsed if elapsed > 0 else None
    return fps, inference_ms


def format_performance_hud(
    *,
    yolo_fps: float | None,
    yolo_inference_ms: float | None,
    vlm_latency_ms: float | None,
    vlm_tokens_per_second: float | None,
    vlm_running: bool,
    vlm_interval_seconds: float,
) -> str:
    if yolo_fps is None or yolo_inference_ms is None:
        yolo_line = "YOLO warming up"
    else:
        yolo_line = f"YOLO {yolo_fps:.1f} FPS | {yolo_inference_ms:.1f} ms infer"
    state = "RUNNING" if vlm_running else "IDLE"
    if vlm_latency_ms is None:
        vlm_line = f"VLM waiting | {vlm_interval_seconds:.1f} s cadence | {state}"
    else:
        throughput = (
            f"{vlm_tokens_per_second:.1f} tok/s"
            if vlm_tokens_per_second is not None
            else "n/a tok/s"
        )
        vlm_line = f"VLM {vlm_latency_ms / 1000.0:.2f} s/req | {throughput} | {state}"
    return f"{yolo_line}\n{vlm_line}"


def write_json_line(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def strict_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.width != 1280 or args.height != 720 or args.fps != 30:
        raise ValueError("the locked camera contract requires NV12 1280x720@30")
    if args.camera_buffers < 4 or args.clean_rgb_buffers < 3:
        raise ValueError("strict mode requires camera pool >= 4 and clean RGB pool >= 3")
    if abs(args.vlm_interval - 3.0) > 1e-9:
        raise ValueError("strict mode fixes the VLM interval at exactly 3.0 seconds")
    if args.vlm_max_tokens != 32:
        raise ValueError("the validated VLM cadence contract fixes max tokens at 32")
    return camera_component_preflight(
        workspace=WORKSPACE,
        library=CAMERA_LIBRARY,
        require_active_patched_driver=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        camera_preflight = strict_preflight(args)
    except Exception as error:  # noqa: BLE001 - concise startup failure is intentional
        print(
            f"strict zero-copy preflight failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 2
    if args.preflight_only:
        print(json.dumps({"passed": True, "camera": camera_preflight}, indent=2))
        return 0

    import torch

    stop_requested = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    metrics_path = (WORKSPACE / args.metrics).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    yolo: StrictMIGraphXYolo | None = None
    vlm: ZeroCopyLlamaCppVlm | None = None
    presenter: EglHipPresenter | None = None
    camera: StrictGpuCamera | None = None
    vlm_worker: LatestOnlyVlmWorker | None = None
    yolo_worker: LatestOnlyYoloWorker | None = None
    latest_detection: GpuDetectionLease | None = None
    frame_latencies_ms: list[float] = []
    present_latencies_ms: list[float] = []
    yolo_preprocess_ms: list[float] = []
    yolo_latencies_ms: list[float] = []
    yolo_result_age_frames: list[int] = []
    vlm_latencies_ms: list[float] = []
    vlm_start_errors_ms: list[float] = []
    yolo_hud_samples: deque[tuple[float, float]] = deque()
    frame_count = 0
    yolo_drops = 0
    vlm_schedule_drops = 0
    missed_vlm_deadlines = 0
    latest_result_id = 0
    latest_vlm_latency_ms: float | None = None
    latest_vlm_tokens_per_second: float | None = None
    latest_caption = "Waiting for the first 3-second VLM caption…"
    latest_performance_hud = format_performance_hud(
        yolo_fps=None,
        yolo_inference_ms=None,
        vlm_latency_ms=None,
        vlm_tokens_per_second=None,
        vlm_running=False,
        vlm_interval_seconds=args.vlm_interval,
    )
    camera_start_attempts = 0
    started_at = time.monotonic()

    with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics:
        try:
            yolo = StrictMIGraphXYolo(
                WORKSPACE / args.yolo_model,
                cache_dir=WORKSPACE / args.yolo_cache,
                ultralytics_repository=WORKSPACE / "third_party/ultralytics",
                kernel_library=WORKSPACE / KERNEL_LIBRARY,
                confidence=args.confidence,
                input_pool_size=2,
                output_pool_size=3,
            )
            yolo.warmup(iterations=2)
            vlm = ZeroCopyLlamaCppVlm(
                LlamaCppIpcConfig(max_tokens=args.vlm_max_tokens),
                workspace=WORKSPACE,
            ).start()
            # Reserve the offset-zero camera DMA-BUF ring before EGL export
            # allocations. EGL can consume an explicit DMA-BUF offset; V4L2's
            # single-plane ABI cannot. A bounded retry handles transient HSA fd
            # exhaustion without changing memory path or falling back to host.
            for camera_start_attempts in range(1, 4):
                try:
                    camera = StrictGpuCamera(
                        workspace=WORKSPACE,
                        library=CAMERA_LIBRARY,
                        device=args.camera,
                        width=args.width,
                        height=args.height,
                        fps=args.fps,
                        camera_buffers=args.camera_buffers,
                        clean_rgb_buffers=args.clean_rgb_buffers,
                    )
                    break
                except ZeroCopyCameraError as error:
                    transient_export_error = "HSA camera DMA-BUF export failed" in str(error)
                    if not transient_export_error or camera_start_attempts == 3:
                        raise
                    time.sleep(0.1 * camera_start_attempts)
            if camera is None:
                raise RuntimeError("strict GPU camera startup exhausted without an error")
            presenter = EglHipPresenter(
                library=WORKSPACE / PRESENTER_LIBRARY,
                frame_width=args.width,
                frame_height=args.height,
                window_scale=args.window_scale,
                visible=not args.hidden,
                title="YOLO26 + VLM - GPU Camera Demo",
            )
            presenter.set_subtitle(latest_caption)
            vlm_worker = LatestOnlyVlmWorker(vlm)
            yolo_worker = LatestOnlyYoloWorker(yolo)
            presenter.set_performance_hud(latest_performance_hud)
            presenter.set_performance_visible(args.show_performance)
            stream = torch.cuda.current_stream(0)
            capability_manifest = {
                "type": "capability_manifest",
                "schema_version": 1,
                "zero_copy_mode": args.zero_copy,
                "zero_host_copy_runtime_enforced": True,
                "zero_host_copy_audit_verified": False,
                "component_copy_audits_verified": True,
                "full_camera_copy_audit_pending": True,
                "camera": camera.runtime_info(),
                "yolo": yolo.runtime_info(),
                "vlm": vlm.runtime_info(),
                "presenter": presenter.info(),
                "scheduler": {
                    "yolo_policy": "one-running-one-newest-pending-bounded",
                    "yolo_inference_thread": "background",
                    "vlm_interval_seconds": args.vlm_interval,
                    "vlm_policy": "latest-only-one-outstanding-no-backlog",
                },
                "performance_hud": {
                    "enabled_at_start": args.show_performance,
                    "toggle_key": "H",
                    "update_interval_seconds": PERFORMANCE_HUD_UPDATE_SECONDS,
                    "yolo_rolling_window_seconds": YOLO_HUD_ROLLING_SECONDS,
                },
                "startup": {"camera_attempts": camera_start_attempts},
            }
            write_json_line(metrics, capability_manifest)

            started_at = time.monotonic()
            next_vlm_at: float | None = None
            next_performance_hud_at = started_at
            while not stop_requested.is_set() and not presenter.should_close:
                if args.duration > 0 and time.monotonic() - started_at >= args.duration:
                    break
                frame_started = time.monotonic()
                frame = camera.acquire(timeout_ms=args.camera_timeout_ms)
                if frame is None:
                    continue
                vlm_lease: HipIpcImageLease | None = None
                vlm_consumer_event = 0
                try:
                    yolo_result = yolo_worker.take_latest()
                    if yolo_result is not None:
                        if yolo_result.error is not None or yolo_result.detection is None:
                            raise RuntimeError(f"YOLO worker failed: {yolo_result.error}")
                        if latest_detection is not None:
                            latest_detection.release()
                        latest_detection = yolo_result.detection
                        yolo_preprocess_ms.append(yolo_result.preprocess_ms)
                        yolo_latencies_ms.append(latest_detection.inference_ms)
                        yolo_hud_samples.append(
                            (yolo_result.completed_at, latest_detection.inference_ms)
                        )
                        write_json_line(
                            metrics,
                            {
                                "type": "yolo_result",
                                "sequence_id": yolo_result.sequence_id,
                                "source_frame_id": yolo_result.source_frame_id,
                                "inference_ms": latest_detection.inference_ms,
                                "preprocess_ms": yolo_result.preprocess_ms,
                            },
                        )

                    prepared_yolo = yolo.prepare_rgb8_pointer(
                        frame_id=frame.frame_id,
                        source_pointer=frame.rgb_device_pointer,
                        source_pitch=frame.rgb_pitch,
                        source_width=frame.width,
                        source_height=frame.height,
                        source_is_bgr=False,
                        source_ready_event=frame.ready_event,
                    )
                    if prepared_yolo is None or not yolo_worker.submit(prepared_yolo):
                        yolo_drops += 1

                    now = time.monotonic()
                    if next_vlm_at is None:
                        # Anchor cadence to the first delivered camera frame. Camera
                        # startup latency is not a missed 3-second VLM deadline.
                        next_vlm_at = now
                    if now >= next_vlm_at:
                        vlm_start_errors_ms.append(abs(now - next_vlm_at) * 1000.0)
                        periods_elapsed = max(1, int((now - next_vlm_at) // args.vlm_interval) + 1)
                        missed_vlm_deadlines += max(0, periods_elapsed - 1)
                        next_vlm_at += periods_elapsed * args.vlm_interval
                        vlm_lease = vlm.prepare_gpu_pointer(
                            source_pointer=frame.rgb_device_pointer,
                            source_pitch=frame.rgb_pitch,
                            source_width=frame.width,
                            source_height=frame.height,
                            source_is_bgr=False,
                            source_ready_event=frame.ready_event,
                            stream=stream.cuda_stream,
                        )
                        if vlm_lease is None:
                            vlm_schedule_drops += 1
                        else:
                            vlm_consumer_event = vlm_lease.ready_event
                            submitted = vlm_worker.submit(
                                source_frame_id=frame.frame_id,
                                source_captured_ns=frame.captured_monotonic_ns,
                                scheduled_at=now,
                                lease=vlm_lease,
                            )
                            if not submitted:
                                vlm_schedule_drops += 1
                                vlm_lease = None

                    result = vlm_worker.latest_after(latest_result_id)
                    if result is not None:
                        latest_result_id = result.request_id
                        vlm_latencies_ms.append(result.latency_ms)
                        latest_vlm_latency_ms = result.latency_ms
                        if result.error is not None:
                            raise RuntimeError(f"VLM worker failed: {result.error}")
                        if result.caption is not None:
                            latest_caption = result.caption
                            presenter.set_subtitle(latest_caption)
                        vlm_performance = vlm.last_request_performance()
                        measured_tokens_per_second = vlm_performance.get(
                            "generated_tokens_per_second"
                        )
                        latest_vlm_tokens_per_second = (
                            float(measured_tokens_per_second)
                            if isinstance(measured_tokens_per_second, int | float)
                            else None
                        )
                        write_json_line(
                            metrics,
                            {
                                "type": "vlm_result",
                                "request_id": result.request_id,
                                "source_frame_id": result.source_frame_id,
                                "latency_ms": result.latency_ms,
                                "generated_tokens_per_second": latest_vlm_tokens_per_second,
                                "caption": result.caption,
                                "error": result.error,
                            },
                        )

                    hud_now = time.monotonic()
                    if hud_now >= next_performance_hud_at:
                        yolo_hud_fps, yolo_hud_inference_ms = rolling_yolo_performance(
                            yolo_hud_samples, now=hud_now
                        )
                        hud_visible = bool(presenter.info()["performance_hud_visible"])
                        if hud_visible:
                            vlm_running = bool(vlm_worker.info()["outstanding"])
                            performance_hud = format_performance_hud(
                                yolo_fps=yolo_hud_fps,
                                yolo_inference_ms=yolo_hud_inference_ms,
                                vlm_latency_ms=latest_vlm_latency_ms,
                                vlm_tokens_per_second=latest_vlm_tokens_per_second,
                                vlm_running=vlm_running,
                                vlm_interval_seconds=args.vlm_interval,
                            )
                            if performance_hud != latest_performance_hud:
                                presenter.set_performance_hud(performance_hud)
                                latest_performance_hud = performance_hud
                        next_performance_hud_at = hud_now + PERFORMANCE_HUD_UPDATE_SECONDS

                    present_started = time.perf_counter()
                    frame_presented = presenter.present_rgb8(
                        source_pointer=frame.rgb_device_pointer,
                        source_pitch=frame.rgb_pitch,
                        source_is_bgr=False,
                        detections_pointer=(
                            latest_detection.tensor.data_ptr()
                            if latest_detection is not None
                            else 0
                        ),
                        detection_count=(
                            latest_detection.tensor.shape[0] if latest_detection is not None else 0
                        ),
                        confidence_threshold=args.confidence,
                        source_ready_event=frame.ready_event,
                        detections_ready_event=(
                            latest_detection.ready_event.cuda_event
                            if latest_detection is not None
                            else 0
                        ),
                        stream=stream.cuda_stream,
                    )
                    if frame_presented:
                        present_latencies_ms.append(
                            (time.perf_counter() - present_started) * 1000.0
                        )
                    if frame_presented and latest_detection is not None:
                        yolo_result_age_frames.append(frame.frame_id - latest_detection.frame_id)
                finally:
                    frame.release(consumer_done_event=vlm_consumer_event)
                if not frame_presented:
                    break
                frame_count += 1
                frame_latencies_ms.append((time.monotonic() - frame_started) * 1000.0)

            loop_ended_at = time.monotonic()
            if yolo_worker is not None:
                yolo_worker.close()
                final_yolo = yolo_worker.take_latest()
                if final_yolo is not None:
                    if final_yolo.error is not None or final_yolo.detection is None:
                        raise RuntimeError(f"YOLO worker failed: {final_yolo.error}")
                    yolo_preprocess_ms.append(final_yolo.preprocess_ms)
                    yolo_latencies_ms.append(final_yolo.detection.inference_ms)
                    final_yolo.detection.release()
            if vlm_worker is not None:
                vlm_worker.close()
                result = vlm_worker.latest_after(latest_result_id)
                if result is not None:
                    latest_result_id = result.request_id
                    vlm_latencies_ms.append(result.latency_ms)
                    if result.error is not None:
                        raise RuntimeError(f"VLM worker failed: {result.error}")
            loop_seconds = loop_ended_at - started_at
            yolo_worker_info = yolo_worker.info() if yolo_worker is not None else None
            vlm_worker_info = vlm_worker.info() if vlm_worker is not None else None
            camera_info = camera.runtime_info() if camera is not None else None
            yolo_info = yolo.runtime_info() if yolo is not None else None
            vlm_info = vlm.runtime_info() if vlm is not None else None
            presenter_info = presenter.info() if presenter is not None else None
            effective_fps = frame_count / max(loop_seconds, 1e-9)
            frame_loop_p95 = percentile(frame_latencies_ms, 0.95)
            yolo_completion_fps = (
                yolo_worker_info["completed"] / max(loop_seconds, 1e-9)
                if yolo_worker_info is not None
                else 0.0
            )
            vlm_start_error_p95 = percentile(vlm_start_errors_ms, 0.95)
            runtime_checks = {
                "capture_present_fps_gte_29": effective_fps >= 29.0,
                "capture_to_present_p95_lte_50ms": (
                    frame_loop_p95 is not None and frame_loop_p95 <= 50.0
                ),
                "yolo_completion_fps_gte_25": yolo_completion_fps >= 25.0,
                "yolo_failed_zero": (
                    yolo_worker_info is not None and yolo_worker_info["failed"] == 0
                ),
                "yolo_queue_depth_lte_1": (
                    yolo_worker_info is not None
                    and yolo_worker_info["max_queue_depth"] <= 1
                ),
                "vlm_requests_completed": (
                    vlm_worker_info is not None and vlm_worker_info["completed"] >= 1
                ),
                "vlm_failed_zero": (
                    vlm_worker_info is not None and vlm_worker_info["failed"] == 0
                ),
                "vlm_start_error_p95_lte_100ms": (
                    vlm_start_error_p95 is not None and vlm_start_error_p95 <= 100.0
                ),
                "vlm_schedule_drops_zero": vlm_schedule_drops == 0,
                "vlm_missed_deadlines_zero": missed_vlm_deadlines == 0,
                "camera_all_frames_requeued": (
                    camera_info is not None
                    and camera_info["frames_acquired"] == camera_info["frames_requeued"]
                ),
                "camera_clean_pool_not_dropped": (
                    camera_info is not None
                    and camera_info["frames_dropped_no_clean_slot"] == 0
                    and camera_info["active_clean_leases"] == 0
                ),
                "all_frames_presented": (
                    presenter_info is not None
                    and presenter_info["frames_presented"] == frame_count
                ),
                "production_image_host_copy_zero": (
                    camera_info is not None
                    and yolo_info is not None
                    and vlm_info is not None
                    and camera_info["image_h2d_bytes"] == 0
                    and camera_info["image_d2h_bytes"] == 0
                    and yolo_info["image_h2d_bytes"] == 0
                    and yolo_info["image_d2h_bytes"] == 0
                    and vlm_info["production_image_h2d_bytes"] == 0
                    and vlm_info["production_image_d2h_bytes"] == 0
                ),
            }
            summary = {
                "type": "final_summary",
                "schema_version": 1,
                "status": "completed",
                "duration_seconds": loop_seconds,
                "frames": frame_count,
                "effective_fps": effective_fps,
                "frame_loop_ms": {
                    "p50": statistics.median(frame_latencies_ms) if frame_latencies_ms else None,
                    "p95": frame_loop_p95,
                },
                "yolo_ms": {
                    "preprocess_p50": (
                        statistics.median(yolo_preprocess_ms) if yolo_preprocess_ms else None
                    ),
                    "preprocess_p95": percentile(yolo_preprocess_ms, 0.95),
                    "p50": statistics.median(yolo_latencies_ms) if yolo_latencies_ms else None,
                    "p95": percentile(yolo_latencies_ms, 0.95),
                    "dropped": yolo_drops
                    + (yolo_worker_info["dropped_superseded"] if yolo_worker_info else 0),
                    "prepare_drops": yolo_drops,
                    "superseded_pending": (
                        yolo_worker_info["dropped_superseded"] if yolo_worker_info else 0
                    ),
                    "completion_fps": yolo_completion_fps,
                    "result_age_frames_p95": percentile(
                        [float(value) for value in yolo_result_age_frames], 0.95
                    ),
                },
                "present_ms": {
                    "p50": statistics.median(present_latencies_ms)
                    if present_latencies_ms
                    else None,
                    "p95": percentile(present_latencies_ms, 0.95),
                },
                "vlm_ms": {
                    "p50": statistics.median(vlm_latencies_ms) if vlm_latencies_ms else None,
                    "p95": percentile(vlm_latencies_ms, 0.95),
                    "interval_seconds": args.vlm_interval,
                    "start_error_p95_ms": vlm_start_error_p95,
                    "schedule_drops": vlm_schedule_drops,
                    "missed_deadlines": missed_vlm_deadlines,
                    "latest_generated_tokens_per_second": latest_vlm_tokens_per_second,
                },
                "performance_hud": {
                    "enabled_at_start": args.show_performance,
                    "visible_at_exit": (
                        presenter_info["performance_hud_visible"]
                        if presenter_info is not None
                        else False
                    ),
                    "toggle_key": "H",
                    "update_interval_seconds": PERFORMANCE_HUD_UPDATE_SECONDS,
                    "yolo_rolling_window_seconds": YOLO_HUD_ROLLING_SECONDS,
                    "last_text": latest_performance_hud,
                },
                "worker": vlm_worker_info,
                "yolo_worker": yolo_worker_info,
                "camera": camera_info,
                "yolo": yolo_info,
                "vlm": vlm_info,
                "presenter": presenter_info,
                "acceptance": {
                    "runtime_checks": runtime_checks,
                    "short_runtime_passed": all(runtime_checks.values()),
                    "duration_gte_20_minutes": loop_seconds >= 1200.0,
                    "z5_soak_passed": all(runtime_checks.values()) and loop_seconds >= 1200.0,
                    "full_camera_copy_audit_passed": False,
                },
                "startup": {"camera_attempts": camera_start_attempts},
                "copy_contract": {
                    "image_h2d_bytes": 0,
                    "image_d2h_bytes": 0,
                    "framebuffer_readback": False,
                    "runtime_trace_required_for_z5": True,
                },
            }
            write_json_line(metrics, summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        finally:
            if yolo_worker is not None:
                yolo_worker.close()
                pending_yolo = yolo_worker.take_latest()
                if pending_yolo is not None and pending_yolo.detection is not None:
                    pending_yolo.detection.release()
            if latest_detection is not None:
                latest_detection.release()
            if vlm_worker is not None:
                vlm_worker.close()
            if camera is not None:
                camera.close()
            if presenter is not None:
                presenter.close()
            if vlm is not None:
                vlm.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
