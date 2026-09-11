#!/usr/bin/env python3
"""Run the strict GPU-resident camera + YOLO26 + Qwen3-VL demo."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import statistics
import subprocess
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
from src.zerocopy_hybrid import (
    HYBRID_INPUT_MODE,
    HybridComposer,
    HybridComposition,
    assert_presenter_coco80_source,
    default_coco80_manifest,
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
    started_at: float
    completed_at: float
    caption: str | None
    error: str | None
    hybrid: HybridComposition | None

    @property
    def latency_ms(self) -> float:
        return (self.completed_at - self.scheduled_at) * 1000.0

    @property
    def request_latency_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000.0


@dataclass(frozen=True, slots=True)
class _VlmTask:
    request_id: int
    source_frame_id: int
    source_captured_ns: int
    scheduled_at: float
    lease: HipIpcImageLease
    prompt: str | None
    hybrid: HybridComposition | None
    constrain_english_sentence: bool


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
        prompt: str | None = None,
        hybrid: HybridComposition | None = None,
        constrain_english_sentence: bool | None = None,
    ) -> int | None:
        with self._lock:
            if self._closed:
                lease.release()
                raise RuntimeError("VLM worker is closed")
            if self._outstanding:
                self._dropped_busy += 1
                lease.release()
                return None
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
                prompt=prompt,
                hybrid=hybrid,
                constrain_english_sentence=(
                    hybrid is not None
                    if constrain_english_sentence is None
                    else constrain_english_sentence
                ),
            )
        )
        self._max_queue_depth = max(self._max_queue_depth, 1)
        return request_id

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            caption = None
            error = None
            started_at = time.monotonic()
            try:
                if task.prompt is None:
                    caption = self._engine.caption_prepared(task.lease)
                else:
                    caption = self._engine.caption_prepared(
                        task.lease,
                        prompt=task.prompt,
                        stop_at_first_sentence=task.hybrid is not None,
                        constrain_english_sentence=task.constrain_english_sentence,
                    )
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
                    started_at=started_at,
                    completed_at=completed_at,
                    caption=caption,
                    error=error,
                    hybrid=task.hybrid,
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
    protected_for_vlm: bool


@dataclass(frozen=True, slots=True)
class _YoloTask:
    sequence_id: int
    prepared: PreparedYoloFrame
    protected_for_vlm: bool


@dataclass(frozen=True, slots=True)
class _PendingHybrid:
    deadline_at: float
    armed_at: float
    source_frame_id: int
    source_captured_ns: int
    lease: HipIpcImageLease
    user_prompt: str | None = None
    prompt_version: int = 0


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
        self._protected_latest: YoloResult | None = None
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped_busy = 0
        self._dropped_superseded = 0
        self._dropped_while_protected = 0
        self._protected_submitted = 0
        self._protected_completed = 0
        self._max_queue_depth = 0
        self._thread = threading.Thread(target=self._run, name="yolo-migraphx-worker", daemon=True)
        self._thread.start()

    def submit(
        self,
        prepared: PreparedYoloFrame,
        *,
        protected_for_vlm: bool = False,
    ) -> bool:
        superseded: _YoloTask | None = None
        rejected = False
        with self._condition:
            if self._closed:
                prepared.release()
                raise RuntimeError("YOLO worker is closed")
            if self._pending is not None and self._pending.protected_for_vlm:
                self._dropped_busy += 1
                self._dropped_while_protected += 1
                rejected = True
            else:
                self._submitted += 1
                sequence_id = self._submitted
                self._protected_submitted += int(protected_for_vlm)
            if not rejected and self._pending is not None:
                superseded = self._pending
                self._dropped_superseded += 1
            if not rejected:
                self._pending = _YoloTask(
                    sequence_id=sequence_id,
                    prepared=prepared,
                    protected_for_vlm=protected_for_vlm,
                )
                self._max_queue_depth = max(self._max_queue_depth, 1)
                self._condition.notify()
        if superseded is not None:
            superseded.prepared.release()
        if rejected:
            prepared.release()
            return False
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
                protected_for_vlm=task.protected_for_vlm,
            )
            with self._lock:
                if task.protected_for_vlm:
                    superseded = self._protected_latest
                    self._protected_latest = result
                    self._protected_completed += 1
                else:
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

    def take_protected(self) -> YoloResult | None:
        with self._lock:
            result = self._protected_latest
            self._protected_latest = None
            return result

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "dropped_busy": self._dropped_busy,
                "dropped_superseded": self._dropped_superseded,
                "dropped_while_protected": self._dropped_while_protected,
                "protected_submitted": self._protected_submitted,
                "protected_completed": self._protected_completed,
                "running": self._running,
                "pending": self._pending is not None,
                "outstanding": self._running or self._pending is not None,
                "latest_ready": self._latest is not None,
                "protected_latest_ready": self._protected_latest is not None,
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
    parser.add_argument(
        "--camera-horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="GPU-unmirror a front-facing camera before YOLO, VLM, and presentation",
    )
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
    parser.add_argument(
        "--vlm-log",
        default="output/realtime/llama-server-zerocopy.log",
        help="Per-run llama-server log; use a unique path for auditable runs",
    )
    parser.add_argument(
        "--vlm-input-mode",
        choices=("clean", "hybrid"),
        default="clean",
        help="Use the clean frame or exact-frame YOLO-guided Hybrid VLM input",
    )
    parser.add_argument("--hybrid-max-hints", type=int, default=8)
    parser.add_argument("--hybrid-arm-timeout-ms", type=float, default=100.0)
    parser.add_argument(
        "--presenter",
        choices=("egl", "web"),
        default="egl",
        help="Use the native EGL window or the prompt-enabled local web dashboard",
    )
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
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
    if not 1 <= args.hybrid_max_hints <= 8:
        raise ValueError("strict Hybrid mode requires --hybrid-max-hints in [1, 8]")
    if abs(args.hybrid_arm_timeout_ms - 100.0) > 1e-9:
        raise ValueError("strict Hybrid mode fixes the arm timeout at exactly 100 ms")
    if args.presenter == "web":
        if args.vlm_input_mode != "hybrid":
            raise ValueError("the prompt-enabled web presenter requires Hybrid VLM input")
        if args.web_host not in {"127.0.0.1", "localhost"}:
            raise ValueError("the unauthenticated prompt dashboard is restricted to localhost")
        if not 1024 <= args.web_port <= 65535:
            raise ValueError("--web-port must be between 1024 and 65535")
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
    hybrid_composer: HybridComposer | None = None
    presenter: Any | None = None
    web_overlay: Any | None = None
    camera: StrictGpuCamera | None = None
    vlm_worker: LatestOnlyVlmWorker | None = None
    yolo_worker: LatestOnlyYoloWorker | None = None
    latest_detection: GpuDetectionLease | None = None
    pending_hybrid: _PendingHybrid | None = None
    frame_latencies_ms: list[float] = []
    present_latencies_ms: list[float] = []
    web_overlay_latencies_ms: list[float] = []
    yolo_preprocess_ms: list[float] = []
    yolo_latencies_ms: list[float] = []
    yolo_result_age_frames: list[int] = []
    vlm_latencies_ms: list[float] = []
    vlm_start_errors_ms: list[float] = []
    hybrid_yolo_dependency_ms: list[float] = []
    hybrid_compose_ms: list[float] = []
    hybrid_hint_counts: list[int] = []
    yolo_hud_samples: deque[tuple[float, float]] = deque()
    frame_count = 0
    yolo_drops = 0
    vlm_schedule_drops = 0
    missed_vlm_deadlines = 0
    hybrid_arm_timeouts = 0
    hybrid_exact_frame_mismatches = 0
    hybrid_control_metadata_d2h_bytes = 0
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
            if args.vlm_input_mode == "hybrid":
                hybrid_composer = HybridComposer(
                    kernel_library=WORKSPACE / KERNEL_LIBRARY,
                    coco_manifest=default_coco80_manifest(WORKSPACE),
                    model_names=yolo.contract["names"],
                    confidence=args.confidence,
                    max_hints=args.hybrid_max_hints,
                    pool_size=2,
                )
                if args.presenter == "egl":
                    assert_presenter_coco80_source(WORKSPACE, hybrid_composer.class_names)
            vlm = ZeroCopyLlamaCppVlm(
                LlamaCppIpcConfig(max_tokens=args.vlm_max_tokens, log_path=args.vlm_log),
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
                        web_stream_buffers=8 if args.presenter == "web" else 0,
                        horizontal_flip=args.camera_horizontal_flip,
                    )
                    break
                except ZeroCopyCameraError as error:
                    transient_export_error = "HSA camera DMA-BUF export failed" in str(error)
                    if not transient_export_error or camera_start_attempts == 3:
                        raise
                    time.sleep(0.1 * camera_start_attempts)
            if camera is None:
                raise RuntimeError("strict GPU camera startup exhausted without an error")
            stream = torch.cuda.current_stream(0)
            if args.presenter == "web":
                from src.zerocopy_web import WebGpuYuyvOverlay, WebPipelinePresenter

                presenter = WebPipelinePresenter(
                    workspace=WORKSPACE,
                    host=args.web_host,
                    port=args.web_port,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                )
                web_overlay = WebGpuYuyvOverlay(
                    kernel_library=WORKSPACE / KERNEL_LIBRARY,
                    coco_manifest=default_coco80_manifest(WORKSPACE),
                    model_names=yolo.contract["names"],
                    upload_stream=stream.cuda_stream,
                    confidence=args.confidence,
                )
                print(f"Web dashboard: {presenter.url}", flush=True)
                if args.open_browser:
                    try:
                        browser_environment = os.environ.copy()
                        for inherited in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONNOUSERSITE"):
                            browser_environment.pop(inherited, None)
                        subprocess.Popen(
                            ["xdg-open", presenter.url],
                            env=browser_environment,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    except OSError as error:
                        print(f"Could not open browser automatically: {error}", file=sys.stderr)
            else:
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
            if args.presenter == "egl":
                presenter.set_performance_hud(latest_performance_hud)
                presenter.set_performance_visible(args.show_performance)
            if (
                args.vlm_input_mode == "hybrid"
                and args.presenter == "egl"
                and presenter.info()["class_label_count"] != 80
            ):
                raise RuntimeError("presenter class atlas does not match the Hybrid COCO80 contract")
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
                    "vlm_input_mode": args.vlm_input_mode,
                    "hybrid_policy": (
                        "exact-frame-protected-yolo-no-semantic-fallback"
                        if args.vlm_input_mode == "hybrid"
                        else None
                    ),
                    "hybrid_arm_timeout_ms": args.hybrid_arm_timeout_ms,
                },
                "hybrid": (
                    hybrid_composer.runtime_info() if hybrid_composer is not None else None
                ),
                "web_gpu_overlay": (
                    web_overlay.runtime_info() if web_overlay is not None else None
                ),
                "performance_hud": {
                    "enabled_at_start": args.presenter == "web" or args.show_performance,
                    "toggle_key": None if args.presenter == "web" else "H",
                    "read_only": args.presenter == "web",
                    "update_interval_seconds": PERFORMANCE_HUD_UPDATE_SECONDS,
                    "yolo_rolling_window_seconds": YOLO_HUD_ROLLING_SECONDS,
                },
                "startup": {"camera_attempts": camera_start_attempts},
            }
            write_json_line(metrics, capability_manifest)

            def publish_yolo_result(result: YoloResult) -> None:
                nonlocal latest_detection
                if result.error is not None or result.detection is None:
                    raise RuntimeError(f"YOLO worker failed: {result.error}")
                detection = result.detection
                yolo_preprocess_ms.append(result.preprocess_ms)
                yolo_latencies_ms.append(detection.inference_ms)
                yolo_hud_samples.append((result.completed_at, detection.inference_ms))
                write_json_line(
                    metrics,
                    {
                        "type": "yolo_result",
                        "sequence_id": result.sequence_id,
                        "source_frame_id": result.source_frame_id,
                        "inference_ms": detection.inference_ms,
                        "preprocess_ms": result.preprocess_ms,
                        "protected_for_vlm": result.protected_for_vlm,
                    },
                )
                if latest_detection is not None and detection.frame_id < latest_detection.frame_id:
                    detection.release()
                    return
                if latest_detection is not None:
                    latest_detection.release()
                latest_detection = detection

            def publish_vlm_result(result: VlmResult) -> None:
                nonlocal latest_result_id
                nonlocal latest_vlm_latency_ms
                nonlocal latest_vlm_tokens_per_second
                nonlocal latest_caption
                latest_result_id = result.request_id
                vlm_latencies_ms.append(result.latency_ms)
                latest_vlm_latency_ms = result.latency_ms
                start_error_ms = abs(result.started_at - result.scheduled_at) * 1000.0
                if result.hybrid is not None:
                    vlm_start_errors_ms.append(start_error_ms)
                if result.error is not None:
                    raise RuntimeError(f"VLM worker failed: {result.error}")
                if result.caption is not None:
                    latest_caption = result.caption
                    if args.presenter == "web":
                        presenter.dashboard.set_caption(
                            latest_caption,
                            request_id=result.request_id,
                            source_frame_id=result.source_frame_id,
                            prompt_version=(
                                result.hybrid.prompt_version
                                if result.hybrid is not None
                                else 0
                            ),
                        )
                    else:
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
                measured_prompt_tokens = vlm_performance.get("prompt_tokens")
                measured_generated_tokens = vlm_performance.get("generated_tokens")
                write_json_line(
                    metrics,
                    {
                        "type": "vlm_result",
                        "request_id": result.request_id,
                        "source_frame_id": result.source_frame_id,
                        "latency_ms": result.latency_ms,
                        "request_latency_ms": result.request_latency_ms,
                        "vlm_http_start_error_ms": start_error_ms,
                        "input_mode": (
                            result.hybrid.input_mode
                            if result.hybrid is not None
                            else "clean"
                        ),
                        "hint_count": (
                            len(result.hybrid.hints) if result.hybrid is not None else 0
                        ),
                        "prompt_tokens": (
                            int(measured_prompt_tokens)
                            if isinstance(measured_prompt_tokens, int | float)
                            else None
                        ),
                        "generated_tokens": (
                            int(measured_generated_tokens)
                            if isinstance(measured_generated_tokens, int | float)
                            else None
                        ),
                        "generated_tokens_per_second": latest_vlm_tokens_per_second,
                        "prompt_version": (
                            result.hybrid.prompt_version
                            if result.hybrid is not None
                            else 0
                        ),
                        "caption": result.caption,
                        "error": result.error,
                    },
                )

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
                web_lease = None
                vlm_consumer_event = 0
                try:
                    protected_yolo = yolo_worker.take_protected()
                    if protected_yolo is not None:
                        if args.vlm_input_mode != "hybrid" or hybrid_composer is None:
                            if protected_yolo.detection is not None:
                                protected_yolo.detection.release()
                            raise RuntimeError("protected YOLO result escaped Hybrid mode")
                        if pending_hybrid is None:
                            if protected_yolo.detection is not None:
                                protected_yolo.detection.release()
                            raise RuntimeError("protected YOLO result has no pending Hybrid image")
                        if protected_yolo.error is not None:
                            raise RuntimeError(f"YOLO worker failed: {protected_yolo.error}")
                        if protected_yolo.source_frame_id != pending_hybrid.source_frame_id:
                            hybrid_exact_frame_mismatches += 1
                            if protected_yolo.detection is not None:
                                protected_yolo.detection.release()
                            raise RuntimeError(
                                "Hybrid exact-frame mismatch: "
                                f"image={pending_hybrid.source_frame_id}, "
                                f"YOLO={protected_yolo.source_frame_id}"
                            )
                        if protected_yolo.detection is None:
                            raise RuntimeError("protected YOLO result has no detection lease")
                        composition = hybrid_composer.finalize(
                            frame_id=pending_hybrid.source_frame_id,
                            image_lease=pending_hybrid.lease,
                            detection=protected_yolo.detection,
                            stream=stream.cuda_stream,
                            user_prompt=pending_hybrid.user_prompt,
                            prompt_version=pending_hybrid.prompt_version,
                        )
                        dependency_ms = (
                            protected_yolo.completed_at - pending_hybrid.armed_at
                        ) * 1000.0
                        hybrid_yolo_dependency_ms.append(dependency_ms)
                        hybrid_compose_ms.append(composition.compose_ms)
                        hybrid_hint_counts.append(len(composition.hints))
                        hybrid_control_metadata_d2h_bytes += (
                            composition.control_metadata_d2h_bytes
                        )
                        request_id = vlm_worker.submit(
                            source_frame_id=pending_hybrid.source_frame_id,
                            source_captured_ns=pending_hybrid.source_captured_ns,
                            scheduled_at=pending_hybrid.deadline_at,
                            lease=pending_hybrid.lease,
                            prompt=composition.prompt,
                            hybrid=composition,
                            constrain_english_sentence=args.presenter != "web",
                        )
                        if request_id is not None and args.presenter == "web":
                            presenter.dashboard.mark_prompt_scheduled(
                                composition.prompt_version
                            )
                        write_json_line(
                            metrics,
                            {
                                "type": "hybrid_request",
                                "request_id": request_id,
                                "camera_frame_id": pending_hybrid.source_frame_id,
                                "yolo_source_frame_id": protected_yolo.source_frame_id,
                                "exact_frame_match": True,
                                "hint_count": len(composition.hints),
                                "hint_classes": [hint.class_name for hint in composition.hints],
                                "yolo_dependency_ms": dependency_ms,
                                "hybrid_compose_ms": composition.compose_ms,
                                "control_metadata_d2h_bytes": (
                                    composition.control_metadata_d2h_bytes
                                ),
                                "prompt_characters": len(composition.prompt),
                                "prompt_version": composition.prompt_version,
                                "input_mode": composition.input_mode,
                                "submitted": request_id is not None,
                            },
                        )
                        pending_hybrid = None
                        if request_id is None:
                            vlm_schedule_drops += 1
                        publish_yolo_result(protected_yolo)

                    yolo_result = yolo_worker.take_latest()
                    if yolo_result is not None:
                        publish_yolo_result(yolo_result)

                    now = time.monotonic()
                    if next_vlm_at is None:
                        # Anchor cadence to the first delivered camera frame. Camera
                        # startup latency is not a missed 3-second VLM deadline.
                        next_vlm_at = now
                    hybrid_due = (
                        args.vlm_input_mode == "hybrid"
                        and pending_hybrid is None
                        and now >= next_vlm_at
                    )
                    if hybrid_due and vlm_worker.info()["outstanding"]:
                        periods_elapsed = max(
                            1, int((now - next_vlm_at) // args.vlm_interval) + 1
                        )
                        missed_vlm_deadlines += periods_elapsed
                        vlm_schedule_drops += 1
                        next_vlm_at += periods_elapsed * args.vlm_interval
                        hybrid_due = False

                    prepared_yolo = yolo.prepare_rgb8_pointer(
                        frame_id=frame.frame_id,
                        source_pointer=frame.rgb_device_pointer,
                        source_pitch=frame.rgb_pitch,
                        source_width=frame.width,
                        source_height=frame.height,
                        source_is_bgr=False,
                        source_ready_event=frame.ready_event,
                    )
                    if hybrid_due:
                        if prepared_yolo is not None:
                            vlm_lease = vlm.prepare_gpu_pointer(
                                source_pointer=frame.rgb_device_pointer,
                                source_pitch=frame.rgb_pitch,
                                source_width=frame.width,
                                source_height=frame.height,
                                source_is_bgr=False,
                                source_ready_event=frame.ready_event,
                                stream=stream.cuda_stream,
                                defer_final_ready=True,
                            )
                            if vlm_lease is None:
                                if not yolo_worker.submit(prepared_yolo):
                                    yolo_drops += 1
                            else:
                                vlm_consumer_event = vlm_lease.base_ready_event
                                if yolo_worker.submit(
                                    prepared_yolo, protected_for_vlm=True
                                ):
                                    prompt_snapshot = (
                                        presenter.prompt_snapshot()
                                        if args.presenter == "web"
                                        else None
                                    )
                                    pending_hybrid = _PendingHybrid(
                                        deadline_at=next_vlm_at,
                                        armed_at=now,
                                        source_frame_id=frame.frame_id,
                                        source_captured_ns=frame.captured_monotonic_ns,
                                        lease=vlm_lease,
                                        user_prompt=(
                                            prompt_snapshot.text
                                            if prompt_snapshot is not None
                                            else None
                                        ),
                                        prompt_version=(
                                            prompt_snapshot.version
                                            if prompt_snapshot is not None
                                            else 0
                                        ),
                                    )
                                    vlm_lease = None
                                    periods_elapsed = max(
                                        1,
                                        int((now - next_vlm_at) // args.vlm_interval) + 1,
                                    )
                                    missed_vlm_deadlines += max(0, periods_elapsed - 1)
                                    next_vlm_at += periods_elapsed * args.vlm_interval
                                else:
                                    yolo_drops += 1
                        else:
                            yolo_drops += 1
                        if (
                            pending_hybrid is None
                            and now - next_vlm_at >= args.hybrid_arm_timeout_ms / 1000.0
                        ):
                            hybrid_arm_timeouts += 1
                            vlm_schedule_drops += 1
                            next_vlm_at += args.vlm_interval
                    else:
                        if prepared_yolo is None or not yolo_worker.submit(prepared_yolo):
                            yolo_drops += 1

                    if args.vlm_input_mode == "clean" and now >= next_vlm_at:
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
                            else:
                                vlm_lease = None

                    result = vlm_worker.latest_after(latest_result_id)
                    if result is not None:
                        publish_vlm_result(result)

                    hud_now = time.monotonic()
                    if hud_now >= next_performance_hud_at:
                        yolo_hud_fps, yolo_hud_inference_ms = rolling_yolo_performance(
                            yolo_hud_samples, now=hud_now
                        )
                        vlm_running = bool(vlm_worker.info()["outstanding"])
                        performance_hud = format_performance_hud(
                            yolo_fps=yolo_hud_fps,
                            yolo_inference_ms=yolo_hud_inference_ms,
                            vlm_latency_ms=latest_vlm_latency_ms,
                            vlm_tokens_per_second=latest_vlm_tokens_per_second,
                            vlm_running=vlm_running,
                            vlm_interval_seconds=args.vlm_interval,
                        )
                        if args.presenter == "web":
                            presenter.dashboard.set_performance(
                                camera_fps=frame_count / max(hud_now - started_at, 1e-9),
                                yolo_fps=yolo_hud_fps,
                                yolo_inference_ms=yolo_hud_inference_ms,
                                vlm_latency_ms=latest_vlm_latency_ms,
                                vlm_tokens_per_second=latest_vlm_tokens_per_second,
                                vlm_running=vlm_running,
                            )
                            latest_performance_hud = performance_hud
                        elif presenter.info()["performance_hud_visible"]:
                            if performance_hud != latest_performance_hud:
                                presenter.set_performance_hud(performance_hud)
                                latest_performance_hud = performance_hud
                        next_performance_hud_at = hud_now + PERFORMANCE_HUD_UPDATE_SECONDS

                    present_started = time.perf_counter()
                    if args.presenter == "web":
                        web_lease = frame.detach_web_stream()
                        if (
                            web_lease is not None
                            and latest_detection is not None
                            and presenter.dashboard.video_client_count > 0
                        ):
                            web_overlay_latencies_ms.append(
                                web_overlay.draw(
                                    web_lease,
                                    latest_detection,
                                    stream=stream.cuda_stream,
                                )
                            )
                        frame_presented = presenter.present(web_lease)
                        web_lease = None
                    else:
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
                                latest_detection.tensor.shape[0]
                                if latest_detection is not None
                                else 0
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
                    if vlm_lease is not None:
                        vlm_lease.release()
                    if web_lease is not None:
                        web_lease.release()
                    frame.release(consumer_done_event=vlm_consumer_event)
                if not frame_presented:
                    break
                frame_count += 1
                frame_latencies_ms.append((time.monotonic() - frame_started) * 1000.0)

            loop_ended_at = time.monotonic()
            if yolo_worker is not None:
                yolo_worker.close()
                final_protected_yolo = yolo_worker.take_protected()
                if final_protected_yolo is not None and final_protected_yolo.detection is not None:
                    final_protected_yolo.detection.release()
                if pending_hybrid is not None:
                    pending_hybrid.lease.release()
                    pending_hybrid = None
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
                    publish_vlm_result(result)
            if args.presenter == "web" and presenter is not None:
                presenter.close()
            loop_seconds = loop_ended_at - started_at
            yolo_worker_info = yolo_worker.info() if yolo_worker is not None else None
            vlm_worker_info = vlm_worker.info() if vlm_worker is not None else None
            camera_info = camera.runtime_info() if camera is not None else None
            yolo_info = yolo.runtime_info() if yolo is not None else None
            vlm_info = vlm.runtime_info() if vlm is not None else None
            hybrid_info = (
                hybrid_composer.runtime_info() if hybrid_composer is not None else None
            )
            presenter_info = presenter.info() if presenter is not None else None
            web_overlay_info = web_overlay.runtime_info() if web_overlay is not None else None
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
                "camera_web_pool_released": (
                    args.presenter != "web"
                    or (
                        camera_info is not None
                        and camera_info["web_stream_frames_converted"]
                        == camera_info["web_stream_frames_released"]
                        and camera_info["web_stream_frames_dropped_no_slot"] == 0
                        and camera_info["active_web_stream_leases"] == 0
                    )
                ),
                "web_encoder_failed_zero": (
                    args.presenter != "web"
                    or (
                        presenter_info is not None
                        and presenter_info["streamer"]["fatal_error"] is None
                    )
                ),
                "web_overlay_gpu_burn_in_active": (
                    args.presenter != "web"
                    or (
                        presenter_info is not None
                        and web_overlay_info is not None
                        and web_overlay_info["per_frame_control_d2h_bytes"] == 0
                        and (
                            presenter_info["frames_submitted"] == 0
                            or web_overlay_info["draws"] > 0
                        )
                    )
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
                "hybrid_exact_frame_match_100pct": (
                    args.vlm_input_mode != "hybrid"
                    or (
                        hybrid_exact_frame_mismatches == 0
                        and yolo_worker_info is not None
                        and yolo_worker_info["protected_completed"] == len(hybrid_hint_counts)
                    )
                ),
                "hybrid_arm_timeouts_zero": (
                    args.vlm_input_mode != "hybrid" or hybrid_arm_timeouts == 0
                ),
                "hybrid_control_metadata_bounded": (
                    args.vlm_input_mode != "hybrid"
                    or hybrid_control_metadata_d2h_bytes == len(hybrid_hint_counts) * 260
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
                "web_overlay_ms": {
                    "p50": (
                        statistics.median(web_overlay_latencies_ms)
                        if web_overlay_latencies_ms
                        else None
                    ),
                    "p95": percentile(web_overlay_latencies_ms, 0.95),
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
                "hybrid": {
                    "enabled": args.vlm_input_mode == "hybrid",
                    "input_mode": (
                        HYBRID_INPUT_MODE if args.vlm_input_mode == "hybrid" else "clean"
                    ),
                    "requests_composed": len(hybrid_hint_counts),
                    "exact_frame_mismatches": hybrid_exact_frame_mismatches,
                    "arm_timeouts": hybrid_arm_timeouts,
                    "hint_count": {
                        "min": min(hybrid_hint_counts) if hybrid_hint_counts else None,
                        "max": max(hybrid_hint_counts) if hybrid_hint_counts else None,
                        "median": (
                            statistics.median(hybrid_hint_counts)
                            if hybrid_hint_counts
                            else None
                        ),
                    },
                    "yolo_dependency_ms": {
                        "p50": (
                            statistics.median(hybrid_yolo_dependency_ms)
                            if hybrid_yolo_dependency_ms
                            else None
                        ),
                        "p95": percentile(hybrid_yolo_dependency_ms, 0.95),
                    },
                    "compose_ms": {
                        "p50": (
                            statistics.median(hybrid_compose_ms)
                            if hybrid_compose_ms
                            else None
                        ),
                        "p95": percentile(hybrid_compose_ms, 0.95),
                    },
                    "control_metadata_d2h_bytes": hybrid_control_metadata_d2h_bytes,
                    "runtime": hybrid_info,
                },
                "web_gpu_overlay": web_overlay_info,
                "performance_hud": {
                    "enabled_at_start": args.presenter == "web" or args.show_performance,
                    "visible_at_exit": (
                        presenter_info["performance_hud_visible"]
                        if presenter_info is not None
                        else False
                    ),
                    "toggle_key": None if args.presenter == "web" else "H",
                    "read_only": args.presenter == "web",
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
                    "hybrid_control_metadata_d2h_bytes": (
                        hybrid_control_metadata_d2h_bytes
                    ),
                    "full_detection_tensor_d2h_bytes": 0,
                    "web_detection_control_metadata_d2h_bytes": 0,
                    "web_class_label_startup_h2d_bytes": (
                        web_overlay_info["class_label_h2d_bytes"]
                        if web_overlay_info is not None
                        else 0
                    ),
                    "encoded_network_output_is_not_image_readback": True,
                    "runtime_trace_required_for_z5": True,
                },
            }
            write_json_line(metrics, summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        finally:
            if yolo_worker is not None:
                yolo_worker.close()
                pending_protected_yolo = yolo_worker.take_protected()
                if (
                    pending_protected_yolo is not None
                    and pending_protected_yolo.detection is not None
                ):
                    pending_protected_yolo.detection.release()
                pending_yolo = yolo_worker.take_latest()
                if pending_yolo is not None and pending_yolo.detection is not None:
                    pending_yolo.detection.release()
            if latest_detection is not None:
                latest_detection.release()
            if pending_hybrid is not None:
                pending_hybrid.lease.release()
            if vlm_worker is not None:
                vlm_worker.close()
            if presenter is not None:
                presenter.close()
            if camera is not None:
                camera.close()
            if web_overlay is not None:
                web_overlay.close()
            if hybrid_composer is not None:
                hybrid_composer.close()
            if vlm is not None:
                vlm.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
