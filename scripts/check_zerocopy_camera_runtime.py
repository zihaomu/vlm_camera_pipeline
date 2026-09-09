#!/usr/bin/env python3
"""Exercise the strict GPU camera ring without exposing frame pixels to Python."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from itertools import pairwise
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.zerocopy_camera import StrictGpuCamera

CAMERA_LIBRARY = ".local/zerocopy-gfx1151/lib/libvlm_camera_gpu_capture.so.0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--output", default="output/realtime/zerocopy-camera-runtime-check.json"
    )
    args = parser.parse_args()
    if args.frames < 8:
        parser.error("--frames must be at least 8")
    if args.timeout_ms < 100:
        parser.error("--timeout-ms must be at least 100")
    return args


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def main() -> int:
    args = parse_args()
    sequences: list[int] = []
    frame_ids: list[int] = []
    captured_ns: list[int] = []
    acquisition_ms: list[float] = []
    slot_pointers: dict[int, int] = {}
    allocation_bytes: set[int] = set()
    pitches: set[int] = set()

    camera = StrictGpuCamera(
        workspace=WORKSPACE,
        library=CAMERA_LIBRARY,
        device=args.camera,
        width=1280,
        height=720,
        fps=30,
        camera_buffers=4,
        clean_rgb_buffers=3,
    )
    started = time.monotonic()
    try:
        for _ in range(args.frames):
            before_acquire = time.perf_counter()
            frame = camera.acquire(timeout_ms=args.timeout_ms)
            acquisition_ms.append((time.perf_counter() - before_acquire) * 1000.0)
            if frame is None:
                raise RuntimeError("camera returned no clean RGB slot")
            with frame:
                sequences.append(frame.sequence)
                frame_ids.append(frame.frame_id)
                captured_ns.append(frame.captured_monotonic_ns)
                allocation_bytes.add(frame.rgb_allocation_bytes)
                pitches.add(frame.rgb_pitch)
                previous_pointer = slot_pointers.setdefault(
                    frame.slot_index, frame.rgb_device_pointer
                )
                if previous_pointer != frame.rgb_device_pointer:
                    raise RuntimeError(
                        f"clean RGB pointer changed for slot {frame.slot_index}: "
                        f"{previous_pointer:#x} -> {frame.rgb_device_pointer:#x}"
                    )
        elapsed = time.monotonic() - started
        runtime = camera.runtime_info()
    finally:
        camera.close()

    frame_intervals_ms = [
        (later - earlier) / 1_000_000.0
        for earlier, later in pairwise(captured_ns)
    ]
    sequence_consecutive = all(
        ((later - earlier) & 0xFFFFFFFF) == 1
        for earlier, later in pairwise(sequences)
    )
    checks = {
        "active_patched_driver": runtime["active_driver_patched"] is True,
        "nv12_1280x720_30": (
            runtime["format"] == "NV12"
            and runtime["width"] == 1280
            and runtime["height"] == 720
            and runtime["fps"] == [1, 30]
        ),
        "all_frames_acquired": len(frame_ids) == args.frames,
        "frame_ids_consecutive": frame_ids == list(range(1, args.frames + 1)),
        "v4l2_sequences_consecutive": sequence_consecutive,
        "timestamps_monotonic": all(
            later > earlier for earlier, later in pairwise(captured_ns)
        ),
        "effective_fps_gte_29": args.frames / elapsed >= 29.0,
        "capture_interval_p95_lte_40ms": percentile(frame_intervals_ms, 0.95) <= 40.0,
        "stable_bounded_clean_pointer_pool": (
            1 <= len(slot_pointers) <= runtime["clean_rgb_pool_size"]
        ),
        "expected_rgb_pitch": pitches == {1280 * 3},
        "expected_rgb_allocation": allocation_bytes == {1280 * 720 * 3},
        "camera_export_allocation_is_dedicated": (
            runtime["camera_allocation_bytes"] >= 4 * 1024 * 1024
            and runtime["camera_allocation_bytes"] >= runtime["size_image"]
        ),
        "camera_buffers_requeued": runtime["frames_requeued"] == args.frames,
        "no_clean_slot_drops": runtime["frames_dropped_no_clean_slot"] == 0,
        "no_active_leases": runtime["active_clean_leases"] == 0,
        "production_image_host_copy_zero": (
            runtime["image_h2d_bytes"] == 0 and runtime["image_d2h_bytes"] == 0
        ),
    }
    report = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": all(checks.values()),
        "scope": "real V4L2 GPU-export camera ring and NV12-to-RGB HIP path",
        "checks": checks,
        "performance": {
            "frames": args.frames,
            "elapsed_seconds": elapsed,
            "effective_fps": args.frames / elapsed,
            "acquire_p50_ms": statistics.median(acquisition_ms),
            "acquire_p95_ms": percentile(acquisition_ms, 0.95),
            "capture_interval_p50_ms": statistics.median(frame_intervals_ms),
            "capture_interval_p95_ms": percentile(frame_intervals_ms, 0.95),
        },
        "sequence": {
            "first": sequences[0],
            "last": sequences[-1],
            "sample_head": sequences[:8],
            "sample_tail": sequences[-8:],
        },
        "gpu_pool": {
            "clean_slot_pointers": {
                str(slot): f"0x{pointer:x}" for slot, pointer in sorted(slot_pointers.items())
            },
            "unique_pointers": len(set(slot_pointers.values())),
            "rgb_pitches": sorted(pitches),
            "rgb_allocation_bytes": sorted(allocation_bytes),
        },
        "runtime": runtime,
        "production_contract": {
            "python_pixel_objects": False,
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "diagnostic_pixel_readback_bytes": 0,
        },
    }
    output = (WORKSPACE / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"], **report["performance"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
