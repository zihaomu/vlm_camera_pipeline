#!/usr/bin/env python3
"""Validate strict YOLO26 MIGraphX inference and emit a machine-readable report."""

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
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if Path("/opt/rocm/lib").is_dir() and "/opt/rocm/lib" not in sys.path:
    sys.path.append("/opt/rocm/lib")

from src.migraphx_detector import MIGraphXCameraDetector


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
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="output/realtime/migraphx-backend-check.json")
    return parser


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations < 3:
        raise SystemExit("--iterations must be at least 3")

    import cv2

    capture = cv2.VideoCapture(args.video)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read the validation frame from {args.video}")

    detector = MIGraphXCameraDetector(
        args.model,
        cache_dir=args.cache_dir,
        ultralytics_repository=WORKSPACE / "third_party/ultralytics",
    )
    detector.warmup(frame, iterations=2)
    elapsed_ms: list[float] = []
    latest = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        latest = detector.detect_bgr(frame)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    assert latest is not None
    steady = elapsed_ms[2:]
    report = {
        "schema_version": 1,
        "passed": True,
        "iterations": args.iterations,
        "total_latency_ms": {
            "mean": round(statistics.mean(steady), 3),
            "p50": round(statistics.median(steady), 3),
            "p95": round(_percentile(steady, 95), 3),
        },
        "latest_detection_count": len(latest.detections),
        "latest_detections": [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 4),
                "xyxy": [
                    round(detection.x1, 2),
                    round(detection.y1, 2),
                    round(detection.x2, 2),
                    round(detection.y2, 2),
                ],
            }
            for detection in latest.detections
        ],
        "runtime": detector.runtime_info(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
