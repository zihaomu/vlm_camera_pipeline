#!/usr/bin/env python3
"""Run the separate YOLO26 MIGraphX + Qwen3-VL realtime camera demo."""

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
os.environ.setdefault("UV_CACHE_DIR", str(PROJECT_CACHE / "uv"))
os.environ.setdefault("UV_PYTHON_DOWNLOADS", "never")
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if Path("/opt/rocm/lib").is_dir() and "/opt/rocm/lib" not in sys.path:
    sys.path.append("/opt/rocm/lib")

from src.camera_io import (
    CameraConfig,
    CameraReader,
    LatestFrameSlot,
    VideoFileConfig,
    VideoFileReader,
)
from src.migraphx_detector import MIGraphXCameraDetector
from src.realtime_pipeline import PipelineConfig, RealtimePipeline
from src.vlm import LlamaCppConfig, LlamaCppVlm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent GPU-only YOLO26 ONNX Runtime MIGraphX + Qwen3-VL demo. "
            "YOLO continuously processes the latest frame while VLM refreshes a subtitle "
            "asynchronously. Neither path silently falls back to CPU."
        )
    )
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera path")
    parser.add_argument(
        "--video-file",
        default=None,
        help="Replay this file in real time instead of opening a camera",
    )
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--fourcc", choices=("NV12", "YUYV", "MJPG"), default="NV12")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--strict-camera-format",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--yolo-model", default="models/yolo26x.onnx")
    parser.add_argument(
        "--migraphx-cache-dir",
        default="models/ort-migraphx-cache/gfx1151-yolo26x",
    )
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--max-latency-ms", type=float, default=150.0)

    parser.add_argument("--vlm-interval", type=float, default=3.0)
    parser.add_argument("--vlm-caption-expiry", type=float, default=20.0)
    parser.add_argument(
        "--vlm-server",
        default="third_party/llama.cpp/build-gfx1151/bin/llama-server",
    )
    parser.add_argument("--vlm-model", default="models/Qwen3-VL-8B-Instruct-Q8_0.gguf")
    parser.add_argument("--vlm-mmproj", default="models/mmproj-F16.gguf")
    parser.add_argument("--vlm-port", type=int, default=0)
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument("--vlm-startup-timeout", type=float, default=180.0)
    parser.add_argument("--vlm-context-size", type=int, default=4096)
    parser.add_argument("--vlm-image-max-tokens", type=int, default=512)
    parser.add_argument("--vlm-max-tokens", type=int, default=64)
    parser.add_argument(
        "--vlm-prompt",
        default=(
            "请用简洁自然的中文描述这个摄像头画面，只输出一句适合视频显示的字幕。"
            "优先描述主要人物、物体和正在发生的动作，不要添加标题或猜测。"
        ),
    )
    parser.add_argument("--vlm-log", default="output/realtime/llama-server-yolo-vlm.log")

    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtitle-height", type=int, default=180)
    parser.add_argument(
        "--subtitle-font",
        default="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=0.75,
        help="Initial scale; drag the window or use +/- and 0 while it is focused",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional bounded duration for a smoke test; omit for a continuous demo",
    )
    parser.add_argument(
        "--metrics-json",
        default="output/realtime/metrics-yolo-migraphx-vlm.json",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.camera_fps <= 0:
        parser.error("camera width, height, and FPS must be positive")
    if args.video_fps is not None and args.video_fps <= 0:
        parser.error("--video-fps must be positive when provided")
    if args.imgsz != 640:
        parser.error("the locked YOLO26x ONNX graph requires --imgsz 640")
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--confidence and --iou must be between 0 and 1")
    if args.warmup_iterations < 1:
        parser.error("--warmup-iterations must be at least one")
    if args.max_latency_ms <= 0:
        parser.error("--max-latency-ms must be positive")
    if args.vlm_interval <= 0 or args.vlm_caption_expiry <= 0:
        parser.error("VLM interval and caption expiry must be positive")
    if args.vlm_timeout <= 0 or args.vlm_startup_timeout <= 0:
        parser.error("VLM timeouts must be positive")
    if args.vlm_context_size < 1024:
        parser.error("--vlm-context-size must be at least 1024")
    if args.vlm_image_max_tokens <= 0 or args.vlm_max_tokens <= 0:
        parser.error("VLM token limits must be positive")
    if not 0 <= args.vlm_port <= 65535:
        parser.error("--vlm-port must be between 0 and 65535")
    if args.subtitle_height < 120:
        parser.error("--subtitle-height must be at least 120")
    if not 0.25 <= args.window_scale <= 2.0:
        parser.error("--window-scale must be between 0.25 and 2.0")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive when provided")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    frame_slot = LatestFrameSlot()
    if args.video_file:
        source = VideoFileReader(
            VideoFileConfig(
                path=args.video_file,
                width=args.width,
                height=args.height,
                fps=args.video_fps,
                loop=args.loop_video,
            ),
            frame_slot,
        )
    else:
        source = CameraReader(
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

    detector = MIGraphXCameraDetector(
        args.yolo_model,
        cache_dir=args.migraphx_cache_dir,
        ultralytics_repository=WORKSPACE / "third_party/ultralytics",
        device=args.gpu_device,
        image_size=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        half=True,
    )
    vlm_engine = LlamaCppVlm(
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
    pipeline = RealtimePipeline(
        source,
        detector,
        PipelineConfig(
            display=args.display,
            display_mode="live",
            display_layout="subtitle",
            max_latency_ms=args.max_latency_ms,
            metrics_json=args.metrics_json,
            duration_seconds=args.duration_seconds,
            warmup_iterations=args.warmup_iterations,
            vlm_interval_seconds=args.vlm_interval,
            vlm_caption_expiry_seconds=args.vlm_caption_expiry,
            subtitle_panel_height=args.subtitle_height,
            subtitle_font_path=args.subtitle_font,
            window_scale=args.window_scale,
            window_name="YOLO26 MIGraphX + Qwen3-VL Realtime Camera",
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

    print(json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
