#!/usr/bin/env python3
"""Evaluate a realtime metrics artifact against the documented M2 gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_MODEL_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, actual: Any, expected: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})


def rss_growth_bytes(metrics: dict[str, Any], samples_path: str | None) -> int | None:
    summary = metrics.get("process_rss_bytes", {})
    if summary.get("first") is not None and summary.get("maximum") is not None:
        return int(summary["maximum"]) - int(summary["first"])
    if summary.get("growth") is not None:
        return int(summary["growth"])
    if not samples_path:
        return None
    with Path(samples_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        return None
    first_rss = int(rows[0]["rss_kib"])
    maximum_rss = max(int(row["rss_kib"]) for row in rows)
    return (maximum_rss - first_rss) * 1024


def evaluate(
    metrics: dict[str, Any],
    *,
    minimum_duration: float,
    minimum_fps: float,
    maximum_display_p95_ms: float,
    maximum_rss_growth_mib: float,
    rss_samples: str | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    latency = metrics.get("latency_ms", {})
    display_p95 = latency.get("capture_to_display", {}).get("p95")
    detector = metrics.get("detector", {})
    camera_frames = int(metrics.get("camera_frames", 0))
    dropped_frames = int(metrics.get("dropped_frames", 0))
    drop_fraction = dropped_frames / camera_frames if camera_frames else 1.0
    rss_growth = rss_growth_bytes(metrics, rss_samples)

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
        "drop_fraction",
        drop_fraction <= 0.20,
        round(drop_fraction, 6),
        "<= 0.20 (latest-frame dropping is allowed)",
    )
    add_check(
        checks,
        "gpu_arch",
        str(detector.get("gpu_arch", "")).startswith("gfx1151"),
        detector.get("gpu_arch"),
        "gfx1151*",
    )
    add_check(
        checks,
        "gpu_device",
        str(detector.get("device", "")).startswith("cuda"),
        detector.get("device"),
        "cuda:0 (ROCm compatibility API)",
    )
    add_check(
        checks,
        "model_sha256",
        detector.get("model_sha256") == EXPECTED_MODEL_SHA256,
        detector.get("model_sha256"),
        EXPECTED_MODEL_SHA256,
    )
    add_check(
        checks,
        "external_nms",
        detector.get("external_nms") is False,
        detector.get("external_nms"),
        "false for end-to-end YOLO26 Route P",
    )
    add_check(
        checks,
        "optional_branches_off",
        metrics.get("record", {}).get("mode") == "off"
        and metrics.get("vlm", {}).get("mode") == "off",
        {
            "record": metrics.get("record", {}).get("mode"),
            "vlm": metrics.get("vlm", {}).get("mode"),
        },
        "record=off, vlm=off",
    )
    maximum_growth = int(maximum_rss_growth_mib * 1024 * 1024)
    add_check(
        checks,
        "rss_growth_bytes",
        rss_growth is not None and rss_growth <= maximum_growth,
        rss_growth,
        f"<= {maximum_growth} ({maximum_rss_growth_mib:g} MiB)",
    )
    return {
        "schema_version": 1,
        "gate": "M2-realtime-camera",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--rss-samples", default=None)
    parser.add_argument("--minimum-duration", type=float, default=1790.0)
    parser.add_argument("--minimum-fps", type=float, default=25.0)
    parser.add_argument("--maximum-display-p95-ms", type=float, default=150.0)
    parser.add_argument("--maximum-rss-growth-mib", type=float, default=128.0)
    parser.add_argument("--output", default="output/realtime/m2-gate.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    result = evaluate(
        metrics,
        minimum_duration=args.minimum_duration,
        minimum_fps=args.minimum_fps,
        maximum_display_p95_ms=args.maximum_display_p95_ms,
        maximum_rss_growth_mib=args.maximum_rss_growth_mib,
        rss_samples=args.rss_samples,
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
