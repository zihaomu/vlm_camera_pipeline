from __future__ import annotations

from copy import deepcopy

from scripts.evaluate_m2_metrics import EXPECTED_MODEL_SHA256, evaluate, rss_growth_bytes


def valid_metrics() -> dict:
    return {
        "exit_reason": "duration_elapsed",
        "elapsed_seconds": 1800.1,
        "camera_read_failures": 0,
        "camera_frames": 54000,
        "processed_frames": 53000,
        "dropped_frames": 1000,
        "capture_fps": 29.9,
        "inference_fps": 29.4,
        "display_fps": 29.3,
        "latency_ms": {"capture_to_display": {"p95": 25.0}},
        "process_rss_bytes": {"growth": 32 * 1024 * 1024},
        "detector": {
            "gpu_arch": "gfx1151",
            "device": "cuda:0",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "external_nms": False,
        },
        "record": {"mode": "off"},
        "vlm": {"mode": "off"},
    }


def run_evaluation(metrics: dict) -> dict:
    return evaluate(
        metrics,
        minimum_duration=1790.0,
        minimum_fps=25.0,
        maximum_display_p95_ms=150.0,
        maximum_rss_growth_mib=128.0,
        rss_samples=None,
    )


def test_valid_metrics_pass_every_m2_check() -> None:
    result = run_evaluation(valid_metrics())
    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_cpu_fallback_and_high_latency_fail_m2() -> None:
    metrics = deepcopy(valid_metrics())
    metrics["detector"]["device"] = "cpu"
    metrics["latency_ms"]["capture_to_display"]["p95"] = 200.0

    result = run_evaluation(metrics)
    failed = {check["name"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert failed == {"gpu_device", "capture_to_display_p95_ms"}


def test_rss_growth_uses_external_samples_for_legacy_metrics(tmp_path) -> None:
    samples = tmp_path / "rss.csv"
    samples.write_text(
        "sampled_at,elapsed_seconds,rss_kib\n"
        "2026-09-08T17:00:00+08:00,0,100000\n"
        "2026-09-08T17:00:30+08:00,30,105000\n"
        "2026-09-08T17:01:00+08:00,60,103000\n",
        encoding="utf-8",
    )
    legacy_metrics = {
        "process_rss_bytes": {
            "count": 3,
            "current": 103000 * 1024,
            "maximum": 105000 * 1024,
        }
    }

    assert rss_growth_bytes(legacy_metrics, str(samples)) == 5000 * 1024
