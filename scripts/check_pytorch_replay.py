#!/usr/bin/env python3
"""Create a deterministic PyTorch ROCm detection artifact from locked video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.realtime_pipeline import PyTorchCameraDetector, draw_on_host_frame


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo26x.pt")
    parser.add_argument(
        "--video", default="third_party/notebook/ultralytics_yolo26/data/sidewalk.mp4"
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--output", default="output/realtime/pytorch-replay-correctness.json")
    parser.add_argument("--screenshot", default="output/realtime/screenshots/pytorch-replay.jpg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frame_index < 0:
        raise SystemExit("--frame-index must be non-negative")

    import cv2

    model_path = Path(args.model)
    video_path = Path(args.video)
    if sha256(model_path) != "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92":
        raise RuntimeError("unexpected yolo26x.pt SHA-256")
    if sha256(video_path) != "0194fb9a4b6590b5ba12e22f32978426efcd89da60ecc285ea9e4f00eb5bc658":
        raise RuntimeError("unexpected sidewalk.mp4 SHA-256")

    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"unable to decode frame {args.frame_index} from {video_path}")

    detector = PyTorchCameraDetector(
        model_path,
        confidence=args.confidence,
        iou=0.45,
        half=True,
        nms_mode="auto",
    )
    detector.warmup(frame, iterations=2)
    started_ns = time.monotonic_ns()
    result = detector.detect_bgr(frame)
    completed_ns = time.monotonic_ns()
    if not result.detections:
        raise RuntimeError("locked replay frame produced no detections")

    screenshot = draw_on_host_frame(
        frame,
        result.detections,
        {
            "backend": detector.backend_name,
            "capture_fps": 0.0,
            "inference_fps": 0.0,
            "display_fps": 0.0,
            "skipped_before_inference": 0,
            "capture_to_display_p50_ms": None,
            "capture_to_display_p95_ms": None,
        },
        detection_sequence=args.frame_index,
        detection_age_ms=(completed_ns - started_ns) / 1e6,
    )
    screenshot_path = Path(args.screenshot)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(screenshot_path), screenshot):
        raise RuntimeError(f"failed to write screenshot {screenshot_path}")

    payload = {
        "schema_version": 1,
        "gate": "M1-pytorch-replay-correctness",
        "passed": True,
        "video": {
            "path": str(video_path.resolve()),
            "sha256": sha256(video_path),
            "frame_index": args.frame_index,
            "frame_shape": list(frame.shape),
        },
        "detector": detector.runtime_info(),
        "timing_ms": {
            "wall": round((completed_ns - started_ns) / 1e6, 3),
            "preprocess": result.preprocess_ms,
            "inference": result.inference_ms,
            "postprocess": result.postprocess_ms,
        },
        "detections": [
            {
                "xyxy": [
                    round(detection.x1, 3),
                    round(detection.y1, 3),
                    round(detection.x2, 3),
                    round(detection.y2, 3),
                ],
                "confidence": round(detection.confidence, 6),
                "class_id": detection.class_id,
                "label": detection.label,
            }
            for detection in result.detections
        ],
        "screenshot": str(screenshot_path.resolve()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
