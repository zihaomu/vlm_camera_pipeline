#!/usr/bin/env python3
"""Run the low-latency Ryzen AI MAX+ 395 camera pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
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

from src.camera_io import CameraConfig, CameraReader, LatestFrameSlot
from src.realtime_pipeline import (
    PipelineConfig,
    PyTorchCameraDetector,
    RealtimePipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Latest-frame YOLO26 camera demo for AMD Ryzen AI MAX+ 395. "
            "MVP supports the PyTorch ROCm backend with recording and VLM disabled."
        )
    )
    parser.add_argument("--backend", choices=("pytorch", "migraphx"), default="pytorch")
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera path")
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--fourcc", choices=("NV12", "YUYV", "MJPG"), default="NV12")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--model", default="models/yolo26x.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nms-mode", choices=("auto", "off", "external"), default="auto")
    parser.add_argument("--display-mode", choices=("live", "processed"), default="live")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-camera-format",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--record", choices=("off", "vaapi", "cpu"), default="off")
    parser.add_argument("--record-path", default="output/realtime/camera.mp4")
    parser.add_argument("--vlm", choices=("off", "llamacpp"), default="off")
    parser.add_argument("--vlm-interval", type=float, default=6.0)
    parser.add_argument("--max-latency-ms", type=float, default=150.0)
    parser.add_argument("--metrics-json", default="output/realtime/metrics.json")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional bounded run duration for smoke/stability gates",
    )
    parser.add_argument("--warmup-iterations", type=int, default=2)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.backend != "pytorch":
        parser.error(
            "MIGraphX is an M3 experimental backend and is not enabled in the M2 implementation"
        )
    if args.record != "off":
        parser.error("recording belongs to M4 and is not enabled in the M2 implementation")
    if args.vlm != "off":
        parser.error("llama.cpp VLM captions belong to M5 and are not enabled yet")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    frame_slot = LatestFrameSlot()
    camera = CameraReader(
        CameraConfig(
            device=args.device,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            fourcc=args.fourcc,
            strict_format=args.strict_camera_format,
        ),
        frame_slot,
    )
    detector = PyTorchCameraDetector(
        args.model,
        device=args.gpu_device,
        image_size=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        half=args.half,
        nms_mode=args.nms_mode,
    )
    pipeline = RealtimePipeline(
        camera,
        detector,
        PipelineConfig(
            display=args.display,
            display_mode=args.display_mode,
            max_latency_ms=args.max_latency_ms,
            metrics_json=args.metrics_json,
            duration_seconds=args.duration_seconds,
            warmup_iterations=args.warmup_iterations,
        ),
    )

    previous_handlers: dict[int, signal.Handlers] = {}

    def request_stop(signum: int, _frame: object) -> None:
        print(f"received signal {signum}; requesting clean shutdown", file=sys.stderr)
        pipeline.request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        metrics = pipeline.run()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
