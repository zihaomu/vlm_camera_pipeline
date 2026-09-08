from __future__ import annotations

from copy import deepcopy

from scripts.evaluate_m5_metrics import (
    EXPECTED_CODE_OBJECT,
    EXPECTED_MMPROJ_SHA256,
    EXPECTED_VLM_SHA256,
    EXPECTED_YOLO_SHA256,
    evaluate,
)


def valid_metrics() -> dict:
    return {
        "exit_reason": "duration_elapsed",
        "elapsed_seconds": 30.0,
        "camera_read_failures": 0,
        "capture_fps": 29.8,
        "inference_fps": 26.0,
        "display_fps": 29.6,
        "latency_ms": {"capture_to_display": {"p95": 30.0}},
        "detector": {
            "gpu_arch": "gfx1151",
            "device": "cuda:0",
            "model_sha256": EXPECTED_YOLO_SHA256,
        },
        "record": {"mode": "off"},
        "vlm": {
            "mode": "llamacpp",
            "requests": 6,
            "successes": 6,
            "failures": 0,
            "cancelled": 0,
            "in_flight": 0,
            "queue_depth": 1,
            "request_latency_ms": {"p95": 2500.0},
            "runtime": {
                "capabilities": ["completion", "multimodal"],
                "device": "ROCm0",
                "hip_code_objects": [EXPECTED_CODE_OBJECT],
                "model_sha256": EXPECTED_VLM_SHA256,
                "mmproj_sha256": EXPECTED_MMPROJ_SHA256,
                "gpu_proof": {
                    "model_layers_offloaded": 37,
                    "model_layers_total": 37,
                    "mmproj_backend": "ROCm0",
                    "cpu_fallback_detected": False,
                    "process_has_dev_kfd": True,
                },
            },
        },
    }


def run_evaluation(metrics: dict, gpu_errors: int = 0) -> dict:
    return evaluate(
        metrics,
        minimum_duration=29.0,
        minimum_fps=25.0,
        maximum_display_p95_ms=150.0,
        maximum_vlm_p95_ms=5000.0,
        minimum_vlm_requests=4,
        gpu_kernel_error_count=gpu_errors,
    )


def test_valid_m5_metrics_pass_all_checks() -> None:
    result = run_evaluation(valid_metrics())
    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_partial_offload_request_failure_and_kernel_fault_fail_gate() -> None:
    metrics = deepcopy(valid_metrics())
    metrics["vlm"]["successes"] = 5
    metrics["vlm"]["failures"] = 1
    metrics["vlm"]["runtime"]["gpu_proof"]["model_layers_offloaded"] = 30

    result = run_evaluation(metrics, gpu_errors=1)
    failed = {check["name"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert failed == {
        "vlm_all_requests_succeeded",
        "vlm_full_model_gpu_offload",
        "gpu_kernel_errors",
    }
