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
from src.vlm import LlamaCppConfig, LlamaCppVlm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Latest-frame YOLO26 + optional Qwen3-VL camera demo for AMD Ryzen AI MAX+ 395. "
            "Both inference paths reject CPU fallback."
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
    parser.add_argument(
        "--vlm-server",
        default="third_party/llama.cpp/build-gfx1151/bin/llama-server",
    )
    parser.add_argument("--vlm-model", default="models/Qwen3-VL-8B-Instruct-Q8_0.gguf")
    parser.add_argument("--vlm-mmproj", default="models/mmproj-F16.gguf")
    parser.add_argument("--vlm-port", type=int, default=0)
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument("--vlm-startup-timeout", type=float, default=180.0)
    parser.add_argument("--vlm-caption-expiry", type=float, default=15.0)
    parser.add_argument("--vlm-context-size", type=int, default=4096)
    parser.add_argument("--vlm-image-max-tokens", type=int, default=512)
    parser.add_argument("--vlm-max-tokens", type=int, default=64)
    parser.add_argument(
        "--vlm-prompt",
        default=(
            "Describe this camera image in one short English sentence. "
            "Mention the main objects and action; do not speculate."
        ),
    )
    parser.add_argument("--vlm-log", default="output/realtime/llama-server.log")
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
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.vlm_interval <= 0:
        parser.error("--vlm-interval must be positive")
    if args.vlm_timeout <= 0 or args.vlm_startup_timeout <= 0:
        parser.error("VLM timeouts must be positive")
    if args.vlm_caption_expiry <= 0:
        parser.error("--vlm-caption-expiry must be positive")
    if args.vlm_context_size < 1024:
        parser.error("--vlm-context-size must be at least 1024")
    if args.vlm_image_max_tokens <= 0 or args.vlm_max_tokens <= 0:
        parser.error("VLM token limits must be positive")
    if not 0 <= args.vlm_port <= 65535:
        parser.error("--vlm-port must be between 0 and 65535")


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
    vlm_engine = (
        LlamaCppVlm(
            LlamaCppConfig(
                server_path=args.vlm_server,
                model_path=args.vlm_model,
                mmproj_path=args.vlm_mmproj,
                log_path=args.vlm_log,
                port=args.vlm_port,
                context_size=args.vlm_context_size,
                image_max_tokens=args.vlm_image_max_tokens,
                max_tokens=args.vlm_max_tokens,
                timeout_seconds=args.vlm_timeout,
                startup_timeout_seconds=args.vlm_startup_timeout,
                prompt=args.vlm_prompt,
            ),
            workspace=WORKSPACE,
        )
        if args.vlm == "llamacpp"
        else None
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
            vlm_interval_seconds=args.vlm_interval,
            vlm_caption_expiry_seconds=args.vlm_caption_expiry,
        ),
        vlm_engine=vlm_engine,
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
