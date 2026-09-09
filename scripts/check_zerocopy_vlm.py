#!/usr/bin/env python3
"""Validate Qwen3-VL GPU preprocess and the llama.cpp HIP IPC v2 request path."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import statistics
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.zerocopy_vlm import (
    LlamaCppIpcConfig,
    ZeroCopyLlamaCppVlm,
    ZeroCopyVlmPreprocessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="output/realtime/screenshots/pytorch-replay.jpg",
        help="Host image used only to validate the device component before camera Z0 is available",
    )
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=3.0)
    parser.add_argument("--output", default="output/realtime/zerocopy-vlm-check.json")
    parser.add_argument(
        "--server-log",
        default="output/realtime/llama-server-zerocopy.log",
    )
    parser.add_argument("--skip-server", action="store_true")
    return parser.parse_args()


def resize_bilinear_align_corners(source_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = source_rgb.shape[:2]
    x_ratio = np.float32((source_width - 1) / (width - 1)) if width > 1 else np.float32(0)
    y_ratio = np.float32((source_height - 1) / (height - 1)) if height > 1 else np.float32(0)
    source_x = np.arange(width, dtype=np.float32) * x_ratio
    source_y = np.arange(height, dtype=np.float32) * y_ratio
    x0 = np.minimum(source_x.astype(np.int32), source_width - 1)
    y0 = np.minimum(source_y.astype(np.int32), source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (source_x - x0)[None, :, None]
    wy = (source_y - y0)[:, None, None]
    pixels = source_rgb.astype(np.float32)
    p00 = pixels[np.ix_(y0, x0)]
    p10 = pixels[np.ix_(y0, x1)]
    p01 = pixels[np.ix_(y1, x0)]
    p11 = pixels[np.ix_(y1, x1)]
    top = p00 + (p10 - p00) * wx
    bottom = p01 + (p11 - p01) * wx
    return (top + (bottom - top) * wy).astype(np.uint8)


def reference_preprocess(source_bgr: np.ndarray, geometry: Any) -> np.ndarray:
    source_rgb = np.ascontiguousarray(source_bgr[:, :, ::-1])
    resized = resize_bilinear_align_corners(
        source_rgb,
        geometry.resized_width,
        geometry.resized_height,
    )
    canvas = np.zeros(
        (geometry.target_height, geometry.target_width, 3),
        dtype=np.uint8,
    )
    y0 = geometry.pad_top
    x0 = geometry.pad_left
    canvas[
        y0 : y0 + geometry.resized_height,
        x0 : x0 + geometry.resized_width,
    ] = resized
    normalized = (canvas.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def validation_readback(pointer: int, shape: tuple[int, int, int], ready_event: int) -> np.ndarray:
    """Explicit validation-only D2H; this function is not imported by the runtime."""
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipEventSynchronize.restype = ctypes.c_int
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemcpy.restype = ctypes.c_int
    output = np.empty(shape, dtype=np.float32)
    status = hip.hipEventSynchronize(ctypes.c_void_p(ready_event))
    if status != 0:
        raise RuntimeError(f"validation hipEventSynchronize failed: {status}")
    status = hip.hipMemcpy(
        ctypes.c_void_p(output.ctypes.data),
        ctypes.c_void_p(pointer),
        output.nbytes,
        2,  # hipMemcpyDeviceToHost, validation only
    )
    if status != 0:
        raise RuntimeError(f"validation hipMemcpy D2H failed: {status}")
    return output


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    args = parse_args()
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    workspace = WORKSPACE
    image_path = (workspace / args.image).resolve()
    output_path = (workspace / args.output).resolve()
    source_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise FileNotFoundError(f"validation image is unavailable: {image_path}")
    source_bgr = np.ascontiguousarray(source_bgr)
    source_height, source_width = source_bgr.shape[:2]
    source_gpu = torch.as_tensor(source_bgr, device="cuda:0")
    stream = torch.cuda.current_stream(0)

    kernel_library = workspace / (
        ".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0"
    )
    preprocessor = ZeroCopyVlmPreprocessor(
        kernel_library=kernel_library,
        source_width=source_width,
        source_height=source_height,
    )
    lease = preprocessor.prepare_rgb8_pointer(
        source_pointer=source_gpu.data_ptr(),
        source_pitch=source_gpu.stride(0) * source_gpu.element_size(),
        source_width=source_width,
        source_height=source_height,
        source_is_bgr=True,
        stream=stream.cuda_stream,
    )
    if lease is None:
        raise RuntimeError("fresh VLM IPC pool unexpectedly had no free slot")
    with lease:
        gpu_output = validation_readback(
            lease.device_pointer,
            (
                3,
                lease.geometry.target_height,
                lease.geometry.target_width,
            ),
            lease.ready_event,
        )
        reference = reference_preprocess(source_bgr, lease.geometry)
        absolute_error = np.abs(gpu_output - reference)
        numerical = {
            "source_shape": list(source_bgr.shape),
            "target_shape": list(gpu_output.shape),
            "resized_shape": [lease.geometry.resized_height, lease.geometry.resized_width],
            "padding_left_top": [lease.geometry.pad_left, lease.geometry.pad_top],
            "max_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
            "mismatched_elements": int(np.count_nonzero(absolute_error)),
            "elements": int(absolute_error.size),
            "validation_only_h2d_bytes": int(source_bgr.nbytes),
            "validation_only_d2h_bytes": int(gpu_output.nbytes),
        }
    preprocessor_info = preprocessor.runtime_info()
    preprocessor.close()
    numerical_passed = numerical["max_abs_error"] <= (2.0 / 255.0 + 1e-7)

    result: dict[str, Any] = {
        "schema_version": 1,
        "component": "qwen3-vl-hip-ipc-v2",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0],
        },
        "numerical_reference": {**numerical, "passed": numerical_passed},
        "preprocessor": preprocessor_info,
        "server": {"skipped": args.skip_server},
        "gate": {"status": "component_failed" if not numerical_passed else "component_passed"},
    }
    if not args.skip_server:
        config = LlamaCppIpcConfig(log_path=args.server_log)
        engine = ZeroCopyLlamaCppVlm(config, workspace=workspace)
        captions: list[str] = []
        wall_latencies_ms: list[float] = []
        request_starts: list[float] = []
        missed_deadlines = 0
        try:
            engine.start()
            next_request_at = time.monotonic()
            for index in range(args.requests):
                remaining = next_request_at - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                started = time.perf_counter()
                request_starts.append(started)
                torch.cuda.nvtx.range_push(f"VLM_IPC_V2_REQUEST_{index + 1}")
                try:
                    prepared = engine.prepare_gpu_pointer(
                        source_pointer=source_gpu.data_ptr(),
                        source_pitch=source_gpu.stride(0) * source_gpu.element_size(),
                        source_width=source_width,
                        source_height=source_height,
                        source_is_bgr=True,
                        stream=stream.cuda_stream,
                    )
                    if prepared is None:
                        raise RuntimeError("VLM IPC pool unexpectedly exhausted")
                    caption = engine.caption_prepared(prepared)
                finally:
                    torch.cuda.nvtx.range_pop()
                wall_latencies_ms.append((time.perf_counter() - started) * 1000.0)
                captions.append(caption)
                print(f"request={index + 1} latency_ms={wall_latencies_ms[-1]:.3f} {caption}")
                next_request_at += args.interval_seconds
                while next_request_at <= time.monotonic():
                    next_request_at += args.interval_seconds
                    missed_deadlines += 1
            runtime = engine.runtime_info()
        finally:
            engine.stop()
        log_text = (workspace / config.log_path).read_text(encoding="utf-8", errors="replace")
        mapped_count = log_text.count("HIP IPC v2 mapped request=")
        input_d2d_count = log_text.count("HIP IPC v2 D2D complete")
        embedding_ready_count = log_text.count("HIP device embedding ready")
        embedding_d2d_count = log_text.count("HIP embedding D2D complete")
        embedding_ready_bytes = [
            int(value)
            for value in re.findall(r"HIP device embedding ready[^\n]* bytes=(\d+)", log_text)
        ]
        embedding_d2d_bytes = [
            int(value)
            for value in re.findall(r"HIP embedding D2D complete[^\n]* bytes=(\d+)", log_text)
        ]
        server_source = (
            workspace
            / config.llama_repository
            / "tools/server/server-ipc.cpp"
        ).read_text(encoding="utf-8")
        forbidden_server_tokens = [
            token
            for token in ("hipMemcpyDeviceToHost", "hipDeviceSynchronize")
            if token in server_source
        ]
        server_passed = (
            len(captions) == args.requests
            and all(caption.strip() for caption in captions)
            and mapped_count == args.requests
            and input_d2d_count == args.requests
            and embedding_ready_count == args.requests
            and embedding_d2d_count >= args.requests
            and sum(embedding_d2d_bytes) == sum(embedding_ready_bytes)
            and not forbidden_server_tokens
            and runtime["gpu_proof"]["model_layers_offloaded"]
            == runtime["gpu_proof"]["model_layers_total"]
            and runtime["gpu_proof"]["mmproj_backend"] == "ROCm0"
        )
        start_intervals = [
            current - previous for previous, current in pairwise(request_starts)
        ]
        interval_errors_ms = [
            abs(interval - args.interval_seconds) * 1000.0 for interval in start_intervals
        ]
        cadence_passed = (
            missed_deadlines == 0
            and (not interval_errors_ms or max(interval_errors_ms) <= 100.0)
        )
        server_passed = server_passed and cadence_passed
        measured = wall_latencies_ms[1:] if len(wall_latencies_ms) > 1 else wall_latencies_ms
        result["server"] = {
            "skipped": False,
            "passed": server_passed,
            "runtime": runtime,
            "captions": captions,
            "request_latency_ms": wall_latencies_ms,
            "warm_request_excluded_from_steady_state": len(wall_latencies_ms) > 1,
            "steady_state_p50_ms": statistics.median(measured),
            "steady_state_p95_ms": percentile(measured, 0.95),
            "configured_interval_seconds": args.interval_seconds,
            "request_start_intervals_seconds": start_intervals,
            "start_interval_errors_ms": interval_errors_ms,
            "start_interval_error_p95_ms": percentile(interval_errors_ms, 0.95),
            "missed_deadlines": missed_deadlines,
            "max_queue_depth": 1,
            "cadence_passed": cadence_passed,
            "ipc_mapping_log_count": mapped_count,
            "ipc_input_d2d_log_count": input_d2d_count,
            "device_embedding_ready_log_count": embedding_ready_count,
            "device_embedding_d2d_log_count": embedding_d2d_count,
            "device_embedding_ready_bytes": embedding_ready_bytes,
            "device_embedding_d2d_bytes": embedding_d2d_bytes,
            "device_embedding_bytes_conserved": (
                sum(embedding_d2d_bytes) == sum(embedding_ready_bytes)
            ),
            "forbidden_server_ipc_tokens": forbidden_server_tokens,
            "http_image_payload_bytes": 0,
        }
        result["gate"] = {
            "status": "passed" if numerical_passed and server_passed else "failed",
            "z3_latency_target_ms": 2800.0,
            "z3_latency_target_met": bool(measured and percentile(measured, 0.95) < 2800.0),
            "camera_integration_blocked_by_z0": True,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["gate"], ensure_ascii=False))
    return 0 if result["gate"]["status"] in {"passed", "component_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
