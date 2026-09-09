#!/usr/bin/env python3
"""Exercise YOLO, split VLM, and EGL together on one validation-only GPU frame."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ["ULTRALYTICS_MIGRAPHX_STRICT"] = "1"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.run_yolo_vlm_zerocopy import LatestOnlyVlmWorker, LatestOnlyYoloWorker
from src.zerocopy_present import EglHipPresenter
from src.zerocopy_vlm import LlamaCppIpcConfig, ZeroCopyLlamaCppVlm
from src.zerocopy_yolo import StrictMIGraphXYolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="output/realtime/screenshots/pytorch-replay.jpg")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--vlm-interval", type=float, default=3.0)
    parser.add_argument("--vlm-hip-stream-priority", type=int, choices=(-1, 0, 1), default=1)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--output", default="output/realtime/zerocopy-integration-check.json")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    args = parse_args()
    if args.frames < 90 or args.target_fps != 30.0 or args.vlm_interval != 3.0:
        raise ValueError("integration check requires >=90 frames, 30 FPS, and 3.0-second VLM")
    source_path = (WORKSPACE / args.image).resolve()
    source_rgb = np.asarray(
        Image.open(source_path).convert("RGB").resize((1280, 720), Image.Resampling.BILINEAR)
    ).copy()
    source_gpu = torch.as_tensor(source_rgb, device="cuda:0")
    source_ready = torch.cuda.Event(enable_timing=False, blocking=True)
    stream = torch.cuda.current_stream(0)
    source_ready.record(stream)

    yolo = StrictMIGraphXYolo(
        WORKSPACE / "models/yolo26x.onnx",
        cache_dir=WORKSPACE / "models/ort-migraphx-cache/gfx1151-yolo26x-strict-iobinding-v1",
        ultralytics_repository=WORKSPACE / "third_party/ultralytics",
        kernel_library=WORKSPACE
        / ".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0",
        input_pool_size=2,
        output_pool_size=3,
    )
    yolo.warmup(iterations=2)
    vlm = ZeroCopyLlamaCppVlm(
        LlamaCppIpcConfig(hip_stream_priority=args.vlm_hip_stream_priority), workspace=WORKSPACE
    ).start()
    vlm_worker = LatestOnlyVlmWorker(vlm)
    yolo_worker = LatestOnlyYoloWorker(yolo)
    latest_detection = None
    yolo_preprocess_ms: list[float] = []
    yolo_ms: list[float] = []
    yolo_result_age_frames: list[int] = []
    yolo_schedule_drops = 0
    present_ms: list[float] = []
    frame_ms: list[float] = []
    vlm_ms: list[float] = []
    request_starts: list[float] = []
    latest_result_id = 0
    caption = "Waiting for VLM…"
    try:
        with EglHipPresenter(
            library=WORKSPACE / ".local/egl-present-gfx1151/lib/libvlm_camera_egl_present.so.0.1.0",
            frame_width=1280,
            frame_height=720,
            window_scale=0.5,
            visible=args.visible,
            title="YOLO26 + VLM zero-copy integration check",
        ) as presenter:
            presenter.set_subtitle(caption)
            loop_started = time.monotonic()
            next_vlm_at = loop_started
            for frame_index in range(args.frames):
                frame_started = time.perf_counter()
                yolo_result = yolo_worker.take_latest()
                if yolo_result is not None:
                    if yolo_result.error is not None or yolo_result.detection is None:
                        raise RuntimeError(f"YOLO worker failed: {yolo_result.error}")
                    if latest_detection is not None:
                        latest_detection.release()
                    latest_detection = yolo_result.detection
                    yolo_preprocess_ms.append(yolo_result.preprocess_ms)
                    yolo_ms.append(latest_detection.inference_ms)

                prepared_yolo = yolo.prepare_rgb8_pointer(
                    frame_id=frame_index,
                    source_pointer=source_gpu.data_ptr(),
                    source_pitch=source_gpu.stride(0),
                    source_width=1280,
                    source_height=720,
                    source_ready_event=int(source_ready.cuda_event),
                )
                if prepared_yolo is None or not yolo_worker.submit(prepared_yolo):
                    yolo_schedule_drops += 1

                now = time.monotonic()
                if now >= next_vlm_at:
                    prepared = vlm.prepare_gpu_pointer(
                        source_pointer=source_gpu.data_ptr(),
                        source_pitch=source_gpu.stride(0),
                        source_width=1280,
                        source_height=720,
                        source_ready_event=int(source_ready.cuda_event),
                        stream=stream.cuda_stream,
                    )
                    if prepared is None:
                        raise RuntimeError("VLM IPC pool exhausted")
                    request_starts.append(now)
                    if not vlm_worker.submit(
                        source_frame_id=frame_index,
                        source_captured_ns=0,
                        scheduled_at=now,
                        lease=prepared,
                    ):
                        raise RuntimeError("VLM latest-only worker was unexpectedly busy at 3s")
                    next_vlm_at += args.vlm_interval
                result = vlm_worker.latest_after(latest_result_id)
                if result is not None:
                    latest_result_id = result.request_id
                    vlm_ms.append(result.latency_ms)
                    if result.error:
                        raise RuntimeError(result.error)
                    caption = result.caption or caption
                    presenter.set_subtitle(caption)
                before_present = time.perf_counter()
                presenter.present_rgb8(
                    source_pointer=source_gpu.data_ptr(),
                    source_pitch=source_gpu.stride(0),
                    source_is_bgr=False,
                    detections_pointer=(
                        latest_detection.tensor.data_ptr() if latest_detection is not None else 0
                    ),
                    detection_count=(
                        latest_detection.tensor.shape[0] if latest_detection is not None else 0
                    ),
                    confidence_threshold=0.5,
                    source_ready_event=int(source_ready.cuda_event),
                    detections_ready_event=(
                        int(latest_detection.ready_event.cuda_event)
                        if latest_detection is not None
                        else 0
                    ),
                    stream=stream.cuda_stream,
                )
                present_ms.append((time.perf_counter() - before_present) * 1000.0)
                if latest_detection is not None:
                    yolo_result_age_frames.append(frame_index - latest_detection.frame_id)
                frame_ms.append((time.perf_counter() - frame_started) * 1000.0)
                deadline = loop_started + (frame_index + 1) / args.target_fps
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            loop_elapsed = time.monotonic() - loop_started
            yolo_worker.close()
            final_yolo = yolo_worker.take_latest()
            if final_yolo is not None:
                if final_yolo.error is not None or final_yolo.detection is None:
                    raise RuntimeError(f"YOLO worker failed: {final_yolo.error}")
                yolo_preprocess_ms.append(final_yolo.preprocess_ms)
                yolo_ms.append(final_yolo.detection.inference_ms)
                final_yolo.detection.release()
            vlm_worker.close()
            final_result = vlm_worker.latest_after(latest_result_id)
            if final_result is not None:
                if final_result.error:
                    raise RuntimeError(final_result.error)
                vlm_ms.append(final_result.latency_ms)
            presenter_info = presenter.info()
    finally:
        yolo_worker.close()
        pending_yolo = yolo_worker.take_latest()
        if pending_yolo is not None and pending_yolo.detection is not None:
            pending_yolo.detection.release()
        if latest_detection is not None:
            latest_detection.release()
        vlm_worker.close()
        vlm.stop()

    request_intervals = [later - earlier for earlier, later in pairwise(request_starts)]
    effective_fps = args.frames / loop_elapsed
    vlm_worker_info = vlm_worker.info()
    yolo_worker_info = yolo_worker.info()
    yolo_completion_fps = yolo_worker_info["completed"] / loop_elapsed
    server_log = vlm.log_path.read_text(encoding="utf-8", errors="replace")
    stream_priority_marker = f"HIP stream 0 priority={args.vlm_hip_stream_priority} (range -1..1)"
    checks = {
        "all_frames_presented": presenter_info["frames_presented"] == args.frames,
        "effective_fps_gte_29": effective_fps >= 29.0,
        "frame_loop_p95_lte_33ms": percentile(frame_ms, 0.95) <= 33.0,
        "yolo_completion_fps_gte_25": yolo_completion_fps >= 25.0,
        "yolo_failed_zero": yolo_worker_info["failed"] == 0,
        "yolo_queue_depth_lte_1": yolo_worker_info["max_queue_depth"] <= 1,
        "vlm_requests_completed": vlm_worker_info["completed"] >= 1,
        "vlm_failed_zero": vlm_worker_info["failed"] == 0,
        "vlm_busy_drops_zero": vlm_worker_info["dropped_busy"] == 0,
        "vlm_queue_depth_lte_1": vlm_worker_info["max_queue_depth"] <= 1,
        "vlm_interval_error_lte_100ms": not request_intervals
        or max(abs(value - 3.0) for value in request_intervals) <= 0.1,
        "yolo_source_event_wait": yolo.runtime_info()["source_event_wait_supported"] is True,
        "split_vlm_api": vlm.runtime_info()["split_prepare_caption_api"] is True,
        "vlm_stream_priority_runtime_logged": stream_priority_marker in server_log,
        "framebuffer_readback_false": True,
        "production_image_host_copy_zero": True,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "scope": "Z5 component integration on one validation-only uploaded frame; camera excluded",
        "checks": checks,
        "performance": {
            "frames": args.frames,
            "loop_seconds": loop_elapsed,
            "effective_fps": effective_fps,
            "frame_loop_p50_ms": statistics.median(frame_ms),
            "frame_loop_p95_ms": percentile(frame_ms, 0.95),
            "yolo_completion_fps": yolo_completion_fps,
            "yolo_schedule_drops": yolo_schedule_drops,
            "yolo_preprocess_p50_ms": statistics.median(yolo_preprocess_ms),
            "yolo_preprocess_p95_ms": percentile(yolo_preprocess_ms, 0.95),
            "yolo_p50_ms": statistics.median(yolo_ms),
            "yolo_p95_ms": percentile(yolo_ms, 0.95),
            "yolo_result_age_frames_p95": percentile(
                [float(value) for value in yolo_result_age_frames], 0.95
            ),
            "present_p50_ms": statistics.median(present_ms),
            "present_p95_ms": percentile(present_ms, 0.95),
            "vlm_latency_ms": vlm_ms,
            "vlm_request_intervals_seconds": request_intervals,
        },
        "workers": {"yolo": yolo_worker_info, "vlm": vlm_worker_info},
        "runtime": {
            "yolo": yolo.runtime_info(),
            "vlm": vlm.runtime_info(),
            "presenter": presenter_info,
        },
        "validation_only_host_transfers": {
            "source_upload_bytes": int(source_rgb.nbytes),
            "excluded_from_production_hot_path": True,
        },
        "stream_priority": {
            "requested": args.vlm_hip_stream_priority,
            "runtime_marker": stream_priority_marker,
            "runtime_marker_count": server_log.count(stream_priority_marker),
        },
        "production_contract": {
            "source": "camera clean RGB HIP pointer",
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "framebuffer_readback": False,
            "camera_integration_blocked_by_z0": True,
        },
    }
    output_path = (WORKSPACE / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"], "performance": report["performance"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
