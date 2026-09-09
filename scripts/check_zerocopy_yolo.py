#!/usr/bin/env python3
"""Validate the GPU-resident YOLO26 preprocess, MIGraphX I/O, and output pool."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

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
if Path("/opt/rocm/lib").is_dir() and "/opt/rocm/lib" not in sys.path:
    sys.path.append("/opt/rocm/lib")

from src.zerocopy_yolo import StrictMIGraphXYolo, hip_kernel_code_objects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo26x.onnx")
    parser.add_argument(
        "--cache-dir",
        default="models/ort-migraphx-cache/gfx1151-yolo26x-strict-iobinding-v1",
    )
    parser.add_argument(
        "--video",
        default="third_party/notebook/ultralytics_yolo26/data/sidewalk.mp4",
    )
    parser.add_argument(
        "--kernel-library",
        default=".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", default="output/realtime/zerocopy-yolo-check.json")
    return parser


def percentile(values: list[float], requested: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * requested / 100.0
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cpu_letterbox_reference(frame_bgr: object) -> object:
    import numpy as np
    from ultralytics.data.augment import LetterBox

    padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=frame_bgr)
    rgb_chw = padded[..., ::-1].transpose((2, 0, 1))
    return np.ascontiguousarray(rgb_chw, dtype=np.float32) / 255.0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations < 3:
        raise SystemExit("--iterations must be at least 3")

    import cv2
    import numpy as np
    import torch

    capture = cv2.VideoCapture(args.video)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read validation frame: {args.video}")

    # This upload is deliberately outside the production API and is counted below as test-only input setup.
    gpu_bgr = torch.as_tensor(np.ascontiguousarray(frame), device="cuda:0")
    runner = StrictMIGraphXYolo(
        args.model,
        cache_dir=args.cache_dir,
        ultralytics_repository=WORKSPACE / "third_party/ultralytics",
        kernel_library=args.kernel_library,
        confidence=args.confidence,
    )
    runner.warmup(iterations=2)

    input_pointers: set[int] = set()
    ort_output_pointers: set[int] = set()
    latencies: list[float] = []
    latest_cpu: np.ndarray | None = None
    latest_transform = None
    for frame_id in range(args.iterations):
        started = time.perf_counter()
        lease = runner.infer_rgb8_pointer(
            frame_id=frame_id,
            source_pointer=gpu_bgr.data_ptr(),
            source_pitch=gpu_bgr.stride(0),
            source_width=frame.shape[1],
            source_height=frame.shape[0],
            source_is_bgr=True,
        )
        if lease is None:
            raise RuntimeError("unexpected output-pool exhaustion during sequential validation")
        lease.ready_event.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        input_pointers.add(runner.input_tensor.data_ptr())
        ort_output_pointers.add(runner._backend.bindings[0].data_ptr())
        latest_cpu = lease.tensor.detach().cpu().numpy().copy()
        latest_transform = lease.transform
        lease.release()

    prepared_first = runner.prepare_rgb8_pointer(
        frame_id=args.iterations,
        source_pointer=gpu_bgr.data_ptr(),
        source_pitch=gpu_bgr.stride(0),
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        source_is_bgr=True,
    )
    prepared_second = runner.prepare_rgb8_pointer(
        frame_id=args.iterations + 1,
        source_pointer=gpu_bgr.data_ptr(),
        source_pitch=gpu_bgr.stride(0),
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        source_is_bgr=True,
    )
    prepared_exhausted = runner.prepare_rgb8_pointer(
        frame_id=args.iterations + 2,
        source_pointer=gpu_bgr.data_ptr(),
        source_pitch=gpu_bgr.stride(0),
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        source_is_bgr=True,
    )
    if prepared_first is None or prepared_second is None:
        raise RuntimeError("two-slot input pool did not provide two prepared leases")
    distinct_input_pointers = (
        prepared_first._input_slot.tensor.data_ptr()
        != prepared_second._input_slot.tensor.data_ptr()
    )
    input_pool_backpressure_passed = prepared_exhausted is None
    prepared_first_detection = runner.infer_prepared(prepared_first)
    prepared_second_detection = runner.infer_prepared(prepared_second)
    prepared_first_detection.release()
    prepared_second_detection.release()

    first = runner.infer_rgb8_pointer(
        frame_id=args.iterations + 3,
        source_pointer=gpu_bgr.data_ptr(),
        source_pitch=gpu_bgr.stride(0),
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        source_is_bgr=True,
    )
    second = runner.infer_rgb8_pointer(
        frame_id=args.iterations + 4,
        source_pointer=gpu_bgr.data_ptr(),
        source_pitch=gpu_bgr.stride(0),
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        source_is_bgr=True,
    )
    exhausted = runner.infer_rgb8_pointer(
        frame_id=args.iterations + 5,
        source_pointer=gpu_bgr.data_ptr(),
        source_pitch=gpu_bgr.stride(0),
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        source_is_bgr=True,
    )
    if first is None or second is None:
        raise RuntimeError("two-slot output pool did not provide two leases")
    first.ready_event.synchronize()
    second.ready_event.synchronize()
    pool_backpressure_passed = exhausted is None
    distinct_pool_pointers = first.tensor.data_ptr() != second.tensor.data_ptr()
    first.release()
    second.release()

    assert latest_cpu is not None and latest_transform is not None
    reference = cpu_letterbox_reference(frame)
    gpu_preprocessed = runner.input_tensor[0].detach().cpu().numpy()
    absolute_error = np.abs(gpu_preprocessed - reference)
    valid = latest_cpu[latest_cpu[:, 4] > 0]
    runtime = runner.runtime_info()
    code_objects = hip_kernel_code_objects(args.kernel_library)
    checks = {
        "provider_is_migraphx": runtime["provider"] == "MIGraphXExecutionProvider",
        "cpu_ep_fallback_disabled": runtime["cpu_ep_fallback_disabled"] is True,
        "migraphx_fp16": runtime["migraphx_fp16"] is True,
        "io_binding": runtime["io_binding"] is True,
        "input_pointer_stable": len(input_pointers) == 1,
        "two_distinct_input_pool_slots": distinct_input_pointers,
        "bounded_input_pool_backpressure": input_pool_backpressure_passed,
        "ort_output_pointer_stable": len(ort_output_pointers) == 1,
        "two_distinct_output_pool_slots": distinct_pool_pointers,
        "bounded_pool_backpressure": pool_backpressure_passed,
        "preprocess_mean_abs_error_lte_1_over_255": float(absolute_error.mean()) <= 1.0 / 255.0,
        "preprocess_max_abs_error_lte_3_over_255": float(absolute_error.max()) <= 3.0 / 255.0,
        "detections_remain_on_gpu_until_diagnostic": valid.shape[0] > 0,
        "gfx1151_kernel_code_object": any(label.endswith("--gfx1151") for label in code_objects),
        "hot_path_copy_counters_zero": all(
            runtime[key] == 0
            for key in (
                "image_h2d_bytes",
                "image_d2h_bytes",
                "tensor_h2d_bytes",
                "tensor_d2h_bytes",
            )
        ),
        "no_global_device_synchronize": runtime["global_device_synchronize"] is False,
        "camera_source_event_wait_supported": runtime["source_event_wait_supported"] is True,
    }
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "scope": "Z2 isolated GPU-pointer component gate; production camera integration is separate",
        "checks": checks,
        "runtime": runtime,
        "letterbox": {
            "source_shape": list(frame.shape),
            "target_shape": [1, 3, 640, 640],
            "gain": latest_transform.gain,
            "resized": [latest_transform.resized_width, latest_transform.resized_height],
            "pad_left": latest_transform.pad_left,
            "pad_top": latest_transform.pad_top,
            "mean_absolute_error": float(absolute_error.mean()),
            "max_absolute_error": float(absolute_error.max()),
        },
        "detections": {
            "valid_count": int(valid.shape[0]),
            "diagnostic_d2h_bytes": int(latest_cpu.nbytes),
            "values": valid.tolist(),
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "p95": round(percentile(latencies, 95), 3),
        },
        "validation_only_host_transfers": {
            "source_upload_bytes": int(frame.nbytes),
            "preprocess_download_bytes": int(gpu_preprocessed.nbytes),
            "detection_download_bytes": int(latest_cpu.nbytes),
            "excluded_from_production_hot_path": True,
        },
        "kernel_code_objects": code_objects,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
