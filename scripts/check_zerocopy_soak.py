#!/usr/bin/env python3
"""Validate a long-running strict camera + YOLO + VLM metrics artifact."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
EXPECTED_PATCHED_MODULE_SRCVERSION = "CA94CB23673E748A9ECC5F2"
KERNEL_ERROR_TERMS = (
    "gpu fault",
    "gpu reset",
    "ring timeout",
    "ring stalled",
    "amd_isp_capture: error",
    "amd_isp_capture: timeout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="output/realtime/metrics-yolo-vlm-zerocopy-camera-soak-30m.jsonl",
    )
    parser.add_argument("--minimum-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--output", default="output/realtime/zerocopy-camera-soak-check.json"
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def read_final(path: Path) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("type") == "final_summary":
                result = record
    if result is None:
        raise ValueError(f"{path} has no final_summary")
    return result


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout + result.stderr


def main() -> int:
    args = parse_args()
    metrics_path = resolve(args.metrics)
    output_path = resolve(args.output)
    final = read_final(metrics_path)
    duration = float(final["duration_seconds"])
    frames = int(final["frames"])
    camera = final["camera"]
    presenter = final["presenter"]
    worker = final["worker"]
    yolo_worker = final["yolo_worker"]
    runtime_checks = final["acceptance"]["runtime_checks"]
    hybrid = final.get("hybrid") or {}
    hybrid_runtime = hybrid.get("runtime") or {}
    hybrid_enabled = bool(hybrid.get("enabled"))
    vlm_interval = float(final["vlm_ms"]["interval_seconds"])
    minimum_vlm_requests = max(1, math.floor(args.minimum_seconds / vlm_interval))

    file_end = datetime.fromtimestamp(metrics_path.stat().st_mtime).astimezone()
    approximate_start = file_end - timedelta(seconds=duration + 120.0)
    kernel_log = command_output(
        [
            "journalctl",
            "-k",
            "--since",
            approximate_start.isoformat(timespec="seconds"),
            "--no-pager",
        ]
    )
    kernel_error_lines = [
        line
        for line in kernel_log.splitlines()
        if any(term in line.lower() for term in KERNEL_ERROR_TERMS)
    ]
    camera_users = command_output(["fuser", "/dev/video0"]).strip()
    loaded_srcversion_path = Path("/sys/module/amd_capture/srcversion")
    loaded_srcversion = (
        loaded_srcversion_path.read_text(encoding="utf-8").strip()
        if loaded_srcversion_path.exists()
        else ""
    )
    checks = {
        "completed": final.get("status") == "completed",
        "duration_meets_user_approved_threshold": duration >= args.minimum_seconds,
        "all_runtime_checks_passed": bool(runtime_checks)
        and all(value is True for value in runtime_checks.values()),
        "all_frames_presented": int(presenter["frames_presented"]) == frames,
        "all_frames_requeued": (
            int(camera["frames_acquired"]) == frames
            and int(camera["frames_requeued"]) == frames
        ),
        "camera_drop_and_lease_zero": (
            int(camera["frames_dropped_no_clean_slot"]) == 0
            and int(camera["active_clean_leases"]) == 0
        ),
        "vlm_worker_clean": (
            int(worker["submitted"]) == int(worker["completed"])
            and int(worker["failed"]) == 0
            and int(worker["dropped_busy"]) == 0
            and worker["outstanding"] is False
            and int(worker["queue_depth"]) == 0
            and int(worker["max_queue_depth"]) <= 1
        ),
        "yolo_worker_clean": (
            int(yolo_worker["submitted"]) == int(yolo_worker["completed"])
            and int(yolo_worker["failed"]) == 0
            and yolo_worker["outstanding"] is False
            and int(yolo_worker["queue_depth"]) == 0
            and int(yolo_worker["max_queue_depth"]) <= 1
            and int(yolo_worker.get("dropped_busy", 0)) == 0
            and int(yolo_worker.get("dropped_superseded", 0)) == 0
            and int(yolo_worker.get("dropped_while_protected", 0)) == 0
        ),
        "vlm_cadence_clean": (
            int(final["vlm_ms"]["schedule_drops"]) == 0
            and int(final["vlm_ms"]["missed_deadlines"]) == 0
        ),
        "production_image_host_copy_zero": (
            int(final["copy_contract"]["image_h2d_bytes"]) == 0
            and int(final["copy_contract"]["image_d2h_bytes"]) == 0
            and int(final["copy_contract"].get("full_detection_tensor_d2h_bytes", 0))
            == 0
        ),
        "vlm_fully_gpu_offloaded": (
            final["vlm"]["gpu_proof"]["model_layers_offloaded"]
            == final["vlm"]["gpu_proof"]["model_layers_total"]
            and final["vlm"]["gpu_proof"]["mmproj_backend"] == "ROCm0"
            and final["vlm"]["gpu_proof"]["cpu_fallback_detected"] is False
        ),
        "minimum_vlm_request_count": int(worker["completed"]) >= minimum_vlm_requests,
        "hybrid_exact_frame_and_metadata_contract": (
            not hybrid_enabled
            or (
                int(hybrid["exact_frame_mismatches"]) == 0
                and int(hybrid["arm_timeouts"]) == 0
                and int(hybrid["requests_composed"]) == int(worker["completed"])
                and int(yolo_worker["protected_submitted"])
                == int(yolo_worker["protected_completed"])
                == int(worker["completed"])
                and int(hybrid["control_metadata_d2h_bytes"])
                == int(worker["completed"]) * 260
                and int(hybrid_runtime["control_metadata_d2h_bytes"])
                == int(worker["completed"]) * 260
            )
        ),
        "hybrid_fixed_pools_clean": (
            not hybrid_enabled
            or (
                int(hybrid_runtime["dropped_no_slot"]) == 0
                and int(final["vlm"]["preprocessor"]["dropped_no_slot"]) == 0
                and int(final["yolo"]["dropped_input_busy"]) == 0
                and int(final["yolo"]["dropped_no_output_slot"]) == 0
            )
        ),
        "patched_camera_module_still_active": (
            loaded_srcversion == EXPECTED_PATCHED_MODULE_SRCVERSION
        ),
        "camera_released_after_run": not camera_users,
        "kernel_gpu_camera_errors_zero": not kernel_error_lines,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "component": "strict-camera-yolo-vlm-soak",
        "status": "passed" if passed else "failed",
        "policy": {
            "minimum_seconds": args.minimum_seconds,
            "minimum_minutes": args.minimum_seconds / 60.0,
            "approved_by_user": True,
            "legacy_30_minute_flag_in_input_ignored": (
                "duration_gte_30_minutes" in final.get("acceptance", {})
            ),
        },
        "input": str(metrics_path.relative_to(WORKSPACE)),
        "performance": {
            "duration_seconds": duration,
            "frames": frames,
            "effective_present_fps": final["effective_fps"],
            "frame_loop_p95_ms": final["frame_loop_ms"]["p95"],
            "yolo_completed": yolo_worker["completed"],
            "yolo_completion_fps": final["yolo_ms"]["completion_fps"],
            "yolo_inference_p95_ms": final["yolo_ms"]["p95"],
            "vlm_completed": worker["completed"],
            "minimum_vlm_requests": minimum_vlm_requests,
            "vlm_latency_p95_ms": final["vlm_ms"]["p95"],
            "vlm_start_error_p95_ms": final["vlm_ms"]["start_error_p95_ms"],
            "hybrid_requests": hybrid.get("requests_composed"),
            "hybrid_metadata_d2h_bytes": hybrid.get("control_metadata_d2h_bytes"),
        },
        "resource_state": {
            "camera_frames_acquired": camera["frames_acquired"],
            "camera_frames_requeued": camera["frames_requeued"],
            "camera_clean_drops": camera["frames_dropped_no_clean_slot"],
            "camera_active_leases": camera["active_clean_leases"],
            "camera_users_after_run": camera_users,
            "loaded_module_srcversion": loaded_srcversion,
            "kernel_error_lines": kernel_error_lines,
        },
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": str(output_path)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
