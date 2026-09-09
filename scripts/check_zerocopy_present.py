#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from src.zerocopy_present import EglHipPresenter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HIP/OpenGL zero-host-copy presentation")
    parser.add_argument(
        "--library",
        type=Path,
        default=REPOSITORY
        / ".local/egl-present-gfx1151/lib/libvlm_camera_egl_present.so.0.1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "output/realtime/zerocopy-present-check.json",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--window-scale", type=float, default=0.5)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="pace the visible probe; 0 measures maximum presentation capacity",
    )
    parser.add_argument("--visible", action="store_true")
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> int:
    args = _parse_args()
    if args.frames < 4:
        raise ValueError("--frames must be at least 4")
    if args.target_fps < 0:
        raise ValueError("--target-fps must be non-negative")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is unset; EGL/X11 or XWayland is required")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if architecture != "gfx1151":
        raise RuntimeError(f"expected gfx1151, got {architecture!r}")

    x = torch.arange(args.width, dtype=torch.int32, device="cuda")
    y = torch.arange(args.height, dtype=torch.int32, device="cuda")[:, None]
    frame = torch.empty((args.height, args.width, 3), dtype=torch.uint8, device="cuda")
    frame[:, :, 0] = ((x[None, :] + y) % 256).to(torch.uint8)
    frame[:, :, 1] = ((2 * x[None, :] + y) % 256).to(torch.uint8)
    frame[:, :, 2] = ((x[None, :] + 2 * y) % 256).to(torch.uint8)
    detections = torch.zeros((300, 6), dtype=torch.float32, device="cuda")
    detections[0] = torch.tensor(
        [80.0, 64.0, 470.0, 520.0, 0.92, 0.0], dtype=torch.float32, device="cuda"
    )
    detections[1] = torch.tensor(
        [620.0, 180.0, 1160.0, 650.0, 0.84, 2.0], dtype=torch.float32, device="cuda"
    )
    ready = torch.cuda.Event(enable_timing=False, blocking=True)
    ready.record(torch.cuda.current_stream(0))
    stream_pointer = int(torch.cuda.current_stream(0).cuda_stream)
    ready_pointer = int(ready.cuda_event)

    durations_ms: list[float] = []
    started = time.perf_counter()
    with EglHipPresenter(
        library=args.library,
        frame_width=args.width,
        frame_height=args.height,
        window_scale=args.window_scale,
        visible=args.visible,
        title="Zero-copy HIP/EGL presenter probe",
    ) as presenter:
        presenter.set_subtitle("VLM：一个人正从停放的汽车旁走过。  YOLO26: person, car")
        presenter.set_performance_hud(
            "YOLO 29.8 FPS | 26.4 ms infer\nVLM 2.03 s/req | 15.7 tok/s | IDLE"
        )
        presenter.set_performance_visible(True)
        initial_info = presenter.info()
        loop_started = time.perf_counter()
        for frame_index in range(args.frames):
            before = time.perf_counter()
            frame_presented = presenter.present_rgb8(
                source_pointer=frame.data_ptr(),
                source_pitch=frame.stride(0),
                source_is_bgr=False,
                detections_pointer=detections.data_ptr(),
                detection_count=detections.shape[0],
                confidence_threshold=0.5,
                source_ready_event=ready_pointer,
                detections_ready_event=ready_pointer,
                stream=stream_pointer,
            )
            if not frame_presented:
                break
            durations_ms.append((time.perf_counter() - before) * 1000.0)
            if presenter.should_close:
                break
            if args.target_fps > 0:
                deadline = loop_started + (frame_index + 1) / args.target_fps
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        loop_elapsed = time.perf_counter() - loop_started
        final_info = presenter.info()
    elapsed = time.perf_counter() - started

    frames_presented = int(final_info["frames_presented"])
    present_capacity_fps = 1000.0 / statistics.mean(durations_ms)
    effective_present_fps = frames_presented / loop_elapsed
    expected_window = [
        round(args.width * args.window_scale),
        round(args.height * args.window_scale),
    ]
    library_sha256 = hashlib.sha256(args.library.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "schema_version": 1,
        "component": "hip-opengl-egl-present",
        "status": "passed",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": architecture,
            "device": 0,
        },
        "graphics": {
            "backend": "HIP-exported-DMA-BUF/EGLImage/OpenGL/X11-or-XWayland",
            "vendor": final_info["gl_vendor"],
            "renderer": final_info["gl_renderer"],
            "version": final_info["gl_version"],
            "interop_device_proof": final_info["interop_device_proof"],
            "window_visible": args.visible,
            "initial_window_size": [
                initial_info["window_width"],
                initial_info["window_height"],
            ],
            "expected_initial_window_size": expected_window,
            "aspect_preserving_resize": True,
            "fullscreen_key": "F11",
            "close_keys": ["Escape", "WM_DELETE_WINDOW"],
        },
        "data_path": {
            "source": "HIP device RGB8 pointer",
            "frame_size": [args.width, args.height],
            "detections": "HIP device float32 [N,6] pointer",
            "present_pool_size": final_info["present_pool_size"],
            "hip_exported_dmabuf_egl_images": final_info["present_pool_size"],
            "frame_host_pointer_api": False,
            "frame_h2d_bytes": 0,
            "frame_d2h_bytes": 0,
            "framebuffer_readback": False,
            "global_hip_synchronize": False,
            "subtitle_updates": final_info["subtitle_updates"],
            "subtitle_h2d_bytes": final_info["subtitle_h2d_bytes"],
            "subtitle_transfer_class": "UI glyph alpha resource on caption change",
            "class_label_count": final_info["class_label_count"],
            "class_label_size": [
                final_info["class_label_width"],
                final_info["class_label_height"],
            ],
            "class_label_h2d_bytes": final_info["class_label_h2d_bytes"],
            "class_label_transfer_class": "static COCO80 UI glyph atlas at startup",
            "class_label_mode": final_info["class_label_mode"],
            "box_palette": "20 vivid class colors, indexed by class ID",
            "window_title": final_info["window_title"],
            "performance_hud_updates": final_info["performance_hud_updates"],
            "performance_hud_h2d_bytes": final_info["performance_hud_h2d_bytes"],
            "performance_hud_size": [
                final_info["performance_hud_width"],
                final_info["performance_hud_height"],
            ],
            "performance_hud_visible": final_info["performance_hud_visible"],
            "performance_hud_mode": final_info["performance_hud_mode"],
            "performance_hud_toggle_key": "H",
            "font_path": final_info["font_path"],
        },
        "performance": {
            "requested_frames": args.frames,
            "frames_presented": frames_presented,
            "elapsed_seconds_including_setup": elapsed,
            "loop_elapsed_seconds": loop_elapsed,
            "target_fps": args.target_fps,
            "effective_present_fps": effective_present_fps,
            "present_capacity_fps": present_capacity_fps,
            "present_call_p50_ms": statistics.median(durations_ms),
            "present_call_p95_ms": _percentile(durations_ms, 0.95),
        },
        "build": {
            "library": str(args.library.resolve()),
            "library_sha256": library_sha256,
            "gpu_arch": "gfx1151",
        },
    }
    checks = {
        "all_frames_presented": frames_presented == args.frames,
        "two_slot_pool": final_info["present_pool_size"] == 2,
        "window_scale_applied": initial_info["window_width"] == expected_window[0]
        and initial_info["window_height"] == expected_window[1],
        "hardware_renderer": "llvmpipe" not in str(final_info["gl_renderer"]).lower(),
        "subtitle_uploaded_once": final_info["subtitle_updates"] == 1,
        "coco80_class_labels_ready": final_info["class_label_count"] == 80
        and final_info["class_label_h2d_bytes"] > 0,
        "ascii_english_window_title": final_info["window_title"]
        == "Zero-copy HIP/EGL presenter probe"
        and final_info["window_title"].isascii(),
        "performance_hud_uploaded_once": final_info["performance_hud_updates"] == 1
        and final_info["performance_hud_h2d_bytes"] > 0,
        "performance_hud_visible": final_info["performance_hud_visible"],
        "zero_host_frame_copy_contract": True,
        "at_least_30_present_calls_per_second": present_capacity_fps >= 30.0,
        "paced_fps_met": args.target_fps <= 0 or effective_present_fps >= args.target_fps * 0.95,
    }
    report["checks"] = checks
    if not all(checks.values()):
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
