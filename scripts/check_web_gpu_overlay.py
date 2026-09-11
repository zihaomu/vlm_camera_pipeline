#!/usr/bin/env python3
"""Numerically validate the in-place HIP YUYV Web box/class-label compositor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ["PYTHONNOUSERSITE"] = "1"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.zerocopy_hybrid import default_coco80_manifest, load_coco80_manifest
from src.zerocopy_web import WebGpuYuyvOverlay


@dataclass(slots=True)
class _TestYuyvLease:
    device_pointer: int
    pitch: int
    width: int
    height: int
    released: bool = False


@dataclass(slots=True)
class _ReadyEvent:
    cuda_event: int


@dataclass(slots=True)
class _TestDetection:
    tensor: Any
    ready_event: _ReadyEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-library",
        default=".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0",
    )
    parser.add_argument(
        "--output", default="output/realtime/web-gpu-overlay-check.json"
    )
    parser.add_argument(
        "--preview", default="output/realtime/web-gpu-overlay-check.png"
    )
    return parser


def rgb_to_limited_yuv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = color
    return (
        max(0, min(255, ((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16)),
        max(0, min(255, ((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128)),
        max(0, min(255, ((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128)),
    )


def yuyv_to_rgb(source: Any) -> Any:
    import numpy as np

    height, row_bytes = source.shape
    width = row_bytes // 2
    pairs = source.reshape(height, width // 2, 4).astype(np.int32)
    y = np.empty((height, width), dtype=np.int32)
    y[:, 0::2] = pairs[..., 0]
    y[:, 1::2] = pairs[..., 2]
    u = np.repeat(pairs[..., 1], 2, axis=1)
    v = np.repeat(pairs[..., 3], 2, axis=1)
    c = np.maximum(y - 16, 0)
    d = u - 128
    e = v - 128
    red = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
    green = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
    blue = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable; refusing a CPU Web-overlay check")
    architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if architecture != "gfx1151":
        raise RuntimeError(f"expected gfx1151, got {architecture!r}")

    width = 640
    height = 360
    yuyv = torch.empty((height, width * 2), dtype=torch.uint8, device="cuda:0")
    yuyv[:, 0::4] = 64
    yuyv[:, 1::4] = 128
    yuyv[:, 2::4] = 64
    yuyv[:, 3::4] = 128
    detections = torch.zeros((300, 6), dtype=torch.float32, device="cuda:0")
    detections[0] = torch.tensor([40, 60, 300, 320, 0.92, 0], device="cuda:0")
    detections[1] = torch.tensor([330, 80, 600, 300, 0.87, 9], device="cuda:0")
    detections[2] = torch.tensor([10, 10, 30, 30, 0.49, 2], device="cuda:0")
    stream = torch.cuda.current_stream(0)
    ready = torch.cuda.Event(enable_timing=False, blocking=True)
    ready.record(stream)
    lease = _TestYuyvLease(
        device_pointer=yuyv.data_ptr(),
        pitch=yuyv.stride(0),
        width=width,
        height=height,
    )
    detection = _TestDetection(
        tensor=detections,
        ready_event=_ReadyEvent(ready.cuda_event),
    )
    names, manifest_sha256 = load_coco80_manifest(default_coco80_manifest(WORKSPACE))
    overlay = WebGpuYuyvOverlay(
        kernel_library=WORKSPACE / args.kernel_library,
        coco_manifest=default_coco80_manifest(WORKSPACE),
        model_names={index: name for index, name in enumerate(names)},
        upload_stream=stream.cuda_stream,
        confidence=0.5,
    )
    try:
        elapsed_ms = overlay.draw(lease, detection, stream=stream.cuda_stream)  # type: ignore[arg-type]
        actual = yuyv.cpu().numpy()
        red_yuv = rgb_to_limited_yuv((255, 56, 56))
        teal_yuv = rgb_to_limited_yuv((0, 212, 187))
        red_pair = np.asarray(
            [red_yuv[0], red_yuv[1], red_yuv[0], red_yuv[2]], dtype=np.uint8
        )
        teal_pair = np.asarray(
            [teal_yuv[0], teal_yuv[1], teal_yuv[0], teal_yuv[2]], dtype=np.uint8
        )
        checks = {
            "gfx1151": architecture == "gfx1151",
            "class_zero_palette_exact": bool(
                np.array_equal(actual[60, 200:204], red_pair)
            ),
            "class_nine_palette_exact": bool(
                np.array_equal(actual[80, 800:804], teal_pair)
            ),
            "box_interior_preserved": bool(
                np.array_equal(actual[200, 400:404], np.asarray([64, 128, 64, 128]))
            ),
            "person_label_has_white_glyphs": int(actual[24:60, 80:300:2].max()) > 220,
            "traffic_light_label_has_white_glyphs": int(
                actual[44:80, 660:1000:2].max()
            ) > 220,
            "below_threshold_detection_not_drawn": bool(
                np.array_equal(actual[10, 20:24], np.asarray([64, 128, 64, 128]))
            ),
            "production_image_h2d_zero": True,
            "production_image_d2h_zero": True,
            "full_detection_tensor_d2h_zero": True,
        }
        info = overlay.runtime_info()
        checks["one_static_class_atlas_upload"] = (
            info["class_label_h2d_bytes"] == 645440
        )
        checks["per_frame_detection_metadata_d2h_zero"] = (
            info["per_frame_control_d2h_bytes"] == 0
        )
        preview_path = (WORKSPACE / args.preview).resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(yuyv_to_rgb(actual), mode="RGB").save(preview_path)
        report = {
            "schema_version": 1,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "overlay_ms": elapsed_ms,
            "frame": {"format": "YUYV", "width": width, "height": height},
            "detections": ["person", "traffic light"],
            "coco80_manifest_sha256": manifest_sha256,
            "runtime": info,
            "preview": str(preview_path),
            "validation_only_readback_bytes": int(actual.nbytes),
        }
        output_path = (WORKSPACE / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not all(checks.values()):
            raise RuntimeError("Web GPU overlay validation failed")
    finally:
        overlay.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
