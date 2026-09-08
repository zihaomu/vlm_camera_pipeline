#!/usr/bin/env python3
"""Evaluate a YOLO + Qwen3-VL realtime artifact against the M5 short gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_YOLO_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
EXPECTED_VLM_SHA256 = "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"
EXPECTED_MMPROJ_SHA256 = "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"
EXPECTED_CODE_OBJECT = "hipv4-amdgcn-amd-amdhsa--gfx1151"


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, actual: Any, expected: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})


def evaluate(
    metrics: dict[str, Any],
    *,
    minimum_duration: float,
    minimum_fps: float,
    maximum_display_p95_ms: float,
    maximum_vlm_p95_ms: float,
    minimum_vlm_requests: int,
    gpu_kernel_error_count: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    detector = metrics.get("detector", {})
    vlm = metrics.get("vlm", {})
    runtime = vlm.get("runtime", {})
    proof = runtime.get("gpu_proof", {})
    display_p95 = metrics.get("latency_ms", {}).get("capture_to_display", {}).get("p95")
    vlm_p95 = vlm.get("request_latency_ms", {}).get("p95")
    requests = int(vlm.get("requests", 0))
    successes = int(vlm.get("successes", 0))
    failures = int(vlm.get("failures", -1))
    cancelled = int(vlm.get("cancelled", -1))
    in_flight = int(vlm.get("in_flight", -1))

    add_check(
        checks,
        "clean_duration_exit",
        metrics.get("exit_reason") == "duration_elapsed",
        metrics.get("exit_reason"),
        "duration_elapsed",
    )
    add_check(
        checks,
        "duration_seconds",
        float(metrics.get("elapsed_seconds", 0)) >= minimum_duration,
        metrics.get("elapsed_seconds"),
        f">= {minimum_duration}",
    )
    add_check(
        checks,
        "camera_read_failures",
        int(metrics.get("camera_read_failures", -1)) == 0,
        metrics.get("camera_read_failures"),
        "0",
    )
    for field in ("capture_fps", "inference_fps", "display_fps"):
        add_check(
            checks,
            field,
            float(metrics.get(field, 0)) >= minimum_fps,
            metrics.get(field),
            f">= {minimum_fps}",
        )
    add_check(
        checks,
        "capture_to_display_p95_ms",
        display_p95 is not None and float(display_p95) <= maximum_display_p95_ms,
        display_p95,
        f"<= {maximum_display_p95_ms}",
    )
    add_check(
        checks,
        "yolo_gpu",
        str(detector.get("gpu_arch", "")).startswith("gfx1151")
        and str(detector.get("device", "")).startswith("cuda"),
        {"arch": detector.get("gpu_arch"), "device": detector.get("device")},
        "gfx1151 through the ROCm CUDA compatibility API",
    )
    add_check(
        checks,
        "yolo_model_sha256",
        detector.get("model_sha256") == EXPECTED_YOLO_SHA256,
        detector.get("model_sha256"),
        EXPECTED_YOLO_SHA256,
    )
    add_check(
        checks,
        "vlm_mode",
        vlm.get("mode") == "llamacpp",
        vlm.get("mode"),
        "llamacpp",
    )
    add_check(
        checks,
        "vlm_request_count",
        requests >= minimum_vlm_requests,
        requests,
        f">= {minimum_vlm_requests}",
    )
    add_check(
        checks,
        "vlm_all_requests_succeeded",
        requests == successes and failures == 0 and cancelled == 0 and in_flight == 0,
        {
            "requests": requests,
            "successes": successes,
            "failures": failures,
            "cancelled": cancelled,
            "in_flight": in_flight,
        },
        "requests=successes, failures=cancelled=in_flight=0",
    )
    add_check(
        checks,
        "vlm_queue_depth",
        int(vlm.get("queue_depth", -1)) == 1,
        vlm.get("queue_depth"),
        "1 (latest snapshot only)",
    )
    add_check(
        checks,
        "vlm_request_p95_ms",
        vlm_p95 is not None and float(vlm_p95) <= maximum_vlm_p95_ms,
        vlm_p95,
        f"<= {maximum_vlm_p95_ms}",
    )
    add_check(
        checks,
        "multimodal_capability",
        "multimodal" in runtime.get("capabilities", []),
        runtime.get("capabilities"),
        "contains multimodal",
    )
    add_check(
        checks,
        "vlm_device",
        runtime.get("device") == "ROCm0",
        runtime.get("device"),
        "ROCm0",
    )
    offloaded = int(proof.get("model_layers_offloaded", -1))
    total = int(proof.get("model_layers_total", -1))
    add_check(
        checks,
        "vlm_full_model_gpu_offload",
        offloaded > 0 and offloaded == total,
        f"{offloaded}/{total}",
        "all model layers",
    )
    add_check(
        checks,
        "vlm_mmproj_gpu_offload",
        proof.get("mmproj_backend") == "ROCm0",
        proof.get("mmproj_backend"),
        "ROCm0",
    )
    add_check(
        checks,
        "vlm_no_cpu_fallback",
        proof.get("cpu_fallback_detected") is False,
        proof.get("cpu_fallback_detected"),
        "false",
    )
    add_check(
        checks,
        "vlm_process_has_dev_kfd",
        proof.get("process_has_dev_kfd") is True,
        proof.get("process_has_dev_kfd"),
        "true",
    )
    add_check(
        checks,
        "vlm_hip_code_object",
        runtime.get("hip_code_objects") == [EXPECTED_CODE_OBJECT],
        runtime.get("hip_code_objects"),
        f"[{EXPECTED_CODE_OBJECT}]",
    )
    add_check(
        checks,
        "vlm_model_sha256",
        runtime.get("model_sha256") == EXPECTED_VLM_SHA256,
        runtime.get("model_sha256"),
        EXPECTED_VLM_SHA256,
    )
    add_check(
        checks,
        "vlm_mmproj_sha256",
        runtime.get("mmproj_sha256") == EXPECTED_MMPROJ_SHA256,
        runtime.get("mmproj_sha256"),
        EXPECTED_MMPROJ_SHA256,
    )
    add_check(
        checks,
        "optional_recording_off",
        metrics.get("record", {}).get("mode") == "off",
        metrics.get("record", {}).get("mode"),
        "off",
    )
    add_check(
        checks,
        "gpu_kernel_errors",
        gpu_kernel_error_count == 0,
        gpu_kernel_error_count,
        "0 reset/fault/hang/timeout events",
    )
    return {
        "schema_version": 1,
        "gate": "M5-yolo-qwen3-vl-short",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--minimum-duration", type=float, default=29.0)
    parser.add_argument("--minimum-fps", type=float, default=25.0)
    parser.add_argument("--maximum-display-p95-ms", type=float, default=150.0)
    parser.add_argument("--maximum-vlm-p95-ms", type=float, default=5000.0)
    parser.add_argument("--minimum-vlm-requests", type=int, default=4)
    parser.add_argument("--gpu-kernel-error-count", type=int, required=True)
    parser.add_argument("--output", default="output/realtime/m5-gate.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    result = evaluate(
        metrics,
        minimum_duration=args.minimum_duration,
        minimum_fps=args.minimum_fps,
        maximum_display_p95_ms=args.maximum_display_p95_ms,
        maximum_vlm_p95_ms=args.maximum_vlm_p95_ms,
        minimum_vlm_requests=args.minimum_vlm_requests,
        gpu_kernel_error_count=args.gpu_kernel_error_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
