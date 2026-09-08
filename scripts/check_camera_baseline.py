#!/usr/bin/env python3
"""Run a bounded camera-only M1 stability gate and write JSON evidence."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.camera_io import CameraConfig, CameraReader


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", choices=("NV12", "YUYV", "MJPG"), default="NV12")
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="output/bringup/camera-baseline-10m.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be positive")

    import cv2

    reader = CameraReader(
        CameraConfig(
            device=args.device,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            fourcc=args.fourcc,
        )
    )
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    timestamps_ns: list[int] = []
    rss_samples: list[int] = []
    observed_frames = 0
    displayed_frames = 0
    exit_reason = "duration_elapsed"
    started = 0.0
    baseline_published = 0
    reader.start()
    try:
        first = reader.slot.consume_after(-1, timeout=5.0)
        if first is None:
            reader.raise_if_failed()
            raise RuntimeError("camera did not publish a frame within five seconds")
        last_sequence = first.sequence
        baseline_published = reader.published_frames
        started = time.monotonic()
        next_sample = started
        while True:
            now = time.monotonic()
            if stop_requested:
                exit_reason = "signal"
                break
            if now - started >= args.duration_seconds:
                break
            frame = reader.slot.consume_after(last_sequence, timeout=0.1)
            reader.raise_if_failed()
            if frame is None:
                continue
            last_sequence = frame.sequence
            timestamps_ns.append(frame.captured_ns)
            observed_frames += 1

            if args.display:
                cv2.imshow("M1 camera baseline - press q or Esc to stop", frame.bgr)
                key = cv2.waitKey(1) & 0xFF
                displayed_frames += 1
                if key in {ord("q"), 27}:
                    exit_reason = "user_exit"
                    break
            if now >= next_sample:
                rss_samples.append(rss_bytes())
                next_sample = now + 1.0
    finally:
        reader.stop()
        if args.display:
            cv2.destroyAllWindows()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    reader.raise_if_failed()

    finished = time.monotonic()
    elapsed = max(0.0, finished - started)
    published_frames = max(0, reader.published_frames - baseline_published)
    intervals_ms = [(second - first) / 1e6 for first, second in pairwise(timestamps_ns)]
    steady_fps = (
        (len(timestamps_ns) - 1) * 1e9 / (timestamps_ns[-1] - timestamps_ns[0])
        if len(timestamps_ns) >= 2 and timestamps_ns[-1] > timestamps_ns[0]
        else 0.0
    )
    rss_growth = rss_samples[-1] - rss_samples[0] if len(rss_samples) >= 2 else 0
    passed = (
        exit_reason == "duration_elapsed"
        and elapsed >= args.duration_seconds * 0.99
        and reader.read_failures == 0
        and steady_fps >= args.camera_fps * 0.93
        and rss_growth <= 128 * 1024 * 1024
    )
    negotiated = reader.negotiated
    payload = {
        "schema_version": 1,
        "gate": "M1-camera-baseline",
        "passed": passed,
        "exit_reason": exit_reason,
        "requested_duration_seconds": args.duration_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "negotiated": (
            {
                "width": negotiated.width,
                "height": negotiated.height,
                "fps": negotiated.fps,
                "fourcc": negotiated.fourcc,
            }
            if negotiated
            else None
        ),
        "published_frames": published_frames,
        "observed_latest_frames": observed_frames,
        "displayed_frames": displayed_frames,
        "read_failures": reader.read_failures,
        "steady_capture_fps": round(steady_fps, 3),
        "frame_interval_ms": {
            "p50": percentile(intervals_ms, 0.50),
            "p95": percentile(intervals_ms, 0.95),
            "p99": percentile(intervals_ms, 0.99),
        },
        "rss_bytes": {
            "samples": len(rss_samples),
            "first": rss_samples[0] if rss_samples else None,
            "last": rss_samples[-1] if rss_samples else None,
            "maximum": max(rss_samples) if rss_samples else None,
            "growth": rss_growth,
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
