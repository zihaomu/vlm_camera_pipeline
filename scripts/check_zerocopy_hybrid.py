#!/usr/bin/env python3
"""Validate Hybrid top-K metadata, exact-frame overlay, and two-stage IPC readiness."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
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

from src.zerocopy_hybrid import (
    HYBRID_HINT_BUFFER_BYTES,
    HybridComposer,
    ZeroCopyHybridError,
    default_coco80_manifest,
    load_coco80_manifest,
)
from src.zerocopy_vlm import ZeroCopyVlmPreprocessor


@dataclass(frozen=True, slots=True)
class _ValidationDetection:
    frame_id: int
    tensor: Any
    ready_event: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-library",
        default=".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0",
    )
    parser.add_argument("--output", default="output/realtime/zerocopy-hybrid-check.json")
    return parser


def validation_readback(pointer: int, shape: tuple[int, int, int], ready_event: int) -> Any:
    """Perform an explicitly test-only image D2H after the production API returns."""
    import numpy as np

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipEventSynchronize.restype = ctypes.c_int
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemcpy.restype = ctypes.c_int
    output = np.empty(shape, dtype=np.float32)
    status = hip.hipEventSynchronize(ctypes.c_void_p(ready_event))
    if status:
        raise RuntimeError(f"validation event wait failed: {status}")
    status = hip.hipMemcpy(
        ctypes.c_void_p(output.ctypes.data),
        ctypes.c_void_p(pointer),
        output.nbytes,
        2,
    )
    if status:
        raise RuntimeError(f"validation image D2H failed: {status}")
    return output


def prepare_image(
    preprocessor: ZeroCopyVlmPreprocessor,
    source: Any,
    stream: Any,
    *,
    defer_final_ready: bool,
) -> Any:
    lease = preprocessor.prepare_rgb8_pointer(
        source_pointer=source.data_ptr(),
        source_pitch=source.stride(0),
        source_width=source.shape[1],
        source_height=source.shape[0],
        source_is_bgr=False,
        stream=stream.cuda_stream,
        defer_final_ready=defer_final_ready,
    )
    if lease is None:
        raise RuntimeError("fresh VLM IPC pool unexpectedly had no free slot")
    return lease


def map_coordinate(
    source_coordinate: float, source_extent: int, resized_extent: int, pad_before: int
) -> int:
    scaled = source_coordinate * (resized_extent - 1) / (source_extent - 1)
    return pad_before + math.floor(scaled + 0.5)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable; refusing a CPU Hybrid check")
    architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if architecture != "gfx1151":
        raise RuntimeError(f"expected gfx1151, got {architecture!r}")

    kernel_library = (WORKSPACE / args.kernel_library).resolve()
    names, manifest_sha256 = load_coco80_manifest(default_coco80_manifest(WORKSPACE))
    model_names = {index: name for index, name in enumerate(names)}
    source_host = np.empty((720, 1280, 3), dtype=np.uint8)
    source_host[..., 0] = np.arange(1280, dtype=np.uint16)[None, :] % 256
    source_host[..., 1] = np.arange(720, dtype=np.uint16)[:, None] % 256
    source_host[..., 2] = 96
    source = torch.as_tensor(source_host, device="cuda:0")
    stream = torch.cuda.current_stream(0)
    preprocessor = ZeroCopyVlmPreprocessor(
        kernel_library=kernel_library,
        source_width=1280,
        source_height=720,
    )
    composer = HybridComposer(
        kernel_library=kernel_library,
        coco_manifest=default_coco80_manifest(WORKSPACE),
        model_names=model_names,
        confidence=0.5,
        max_hints=8,
    )
    clean_lease = None
    empty_lease = None
    mismatch_lease = None
    hybrid_lease = None
    try:
        clean_lease = prepare_image(
            preprocessor, source, stream, defer_final_ready=False
        )
        shape = (
            3,
            clean_lease.geometry.target_height,
            clean_lease.geometry.target_width,
        )
        clean = validation_readback(clean_lease.device_pointer, shape, clean_lease.ready_event)
        clean_lease.release()
        clean_lease = None

        empty_detections = torch.zeros((300, 6), dtype=torch.float32, device="cuda:0")
        empty_ready = torch.cuda.Event(enable_timing=False, blocking=True)
        empty_ready.record(stream)
        empty_detection = _ValidationDetection(
            frame_id=41,
            tensor=empty_detections,
            ready_event=empty_ready,
        )
        empty_lease = prepare_image(
            preprocessor, source, stream, defer_final_ready=True
        )
        empty_composition = composer.finalize(
            frame_id=41,
            image_lease=empty_lease,
            detection=empty_detection,  # type: ignore[arg-type]
            stream=stream.cuda_stream,
        )
        empty_hybrid = validation_readback(
            empty_lease.device_pointer, shape, empty_lease.ready_event
        )
        empty_lease.release()
        empty_lease = None

        mismatch_lease = prepare_image(
            preprocessor, source, stream, defer_final_ready=True
        )
        mismatch_rejected = False
        try:
            composer.finalize(
                frame_id=42,
                image_lease=mismatch_lease,
                detection=empty_detection,  # type: ignore[arg-type]
                stream=stream.cuda_stream,
            )
        except ZeroCopyHybridError as error:
            mismatch_rejected = "exact-frame mismatch" in str(error)
        mismatch_remained_unpublished = not mismatch_lease.final_ready
        mismatch_lease.release()
        mismatch_lease = None

        detections_host = np.zeros((300, 6), dtype=np.float32)
        detections_host[0] = [80, 90, 420, 680, 0.70, 0]
        detections_host[1] = [500, 200, 1000, 600, 0.95, 2]
        detections_host[2] = [10, 10, 20, 20, 0.49, 16]
        detections_host[3] = [100, 100, 300, 300, 0.95, 1]
        detections_host[4] = [200, 200, 200, 260, 0.99, 5]
        detections_host[5] = [250, 250, 400, 400, 0.88, 99]
        detections = torch.as_tensor(detections_host, device="cuda:0")
        detections_ready = torch.cuda.Event(enable_timing=False, blocking=True)
        detections_ready.record(stream)
        detection = _ValidationDetection(
            frame_id=42,
            tensor=detections,
            ready_event=detections_ready,
        )
        hybrid_lease = prepare_image(
            preprocessor, source, stream, defer_final_ready=True
        )
        if hybrid_lease.final_ready:
            raise RuntimeError("deferred Hybrid image was prematurely final-ready")
        composition = composer.finalize(
            frame_id=42,
            image_lease=hybrid_lease,
            detection=detection,  # type: ignore[arg-type]
            stream=stream.cuda_stream,
        )
        if not hybrid_lease.final_ready:
            raise RuntimeError("Hybrid image did not transition to final-ready")
        hybrid = validation_readback(
            hybrid_lease.device_pointer, shape, hybrid_lease.ready_event
        )
        geometry = hybrid_lease.geometry
        hybrid_lease.release()
        hybrid_lease = None

        changed = np.any(clean != hybrid, axis=0)
        content_left = geometry.pad_left
        content_right = geometry.pad_left + geometry.resized_width
        padding_changed = int(changed[:, :content_left].sum() + changed[:, content_right:].sum())
        changed_pixels = int(changed.sum())
        expected = [(1, 2, "car"), (2, 1, "bicycle"), (3, 0, "person")]
        observed = [(hint.rank, hint.class_id, hint.class_name) for hint in composition.hints]
        rank_colors = {
            1: (255, 56, 56),
            2: (255, 157, 151),
            3: (255, 178, 29),
        }
        box_mapping: list[dict[str, Any]] = []
        mapping_errors: list[int] = []
        for hint in composition.hints:
            expected_box = [
                map_coordinate(hint.x1, 1280, geometry.resized_width, geometry.pad_left),
                map_coordinate(hint.y1, 720, geometry.resized_height, geometry.pad_top),
                map_coordinate(hint.x2, 1280, geometry.resized_width, geometry.pad_left),
                map_coordinate(hint.y2, 720, geometry.resized_height, geometry.pad_top),
            ]
            color = (
                np.asarray(rank_colors[hint.rank], dtype=np.float32) - np.float32(127.5)
            ) / np.float32(127.5)
            color_mask = np.all(
                np.isclose(hybrid, color[:, None, None], rtol=0.0, atol=1e-7), axis=0
            )
            observed_y, observed_x = np.nonzero(color_mask)
            # The rank badge intentionally occupies x1..x1+27 above the box.
            # Inspect the top border outside that badge span so its 18-pixel
            # vertical offset is not mistaken for a coordinate error.
            top_border_y, _ = np.nonzero(
                color_mask[:, expected_box[0] + 28 : expected_box[2] + 1]
            )
            observed_box = [
                int(observed_x.min()),
                int(top_border_y.min()),
                int(observed_x.max()),
                int(observed_y.max()),
            ]
            errors = [
                abs(observed_coordinate - expected_coordinate)
                for observed_coordinate, expected_coordinate in zip(
                    observed_box, expected_box, strict=True
                )
            ]
            mapping_errors.extend(errors)
            box_mapping.append(
                {
                    "rank": hint.rank,
                    "expected": expected_box,
                    "observed": observed_box,
                    "errors": errors,
                }
            )
        box_mapping_max_error = max(mapping_errors, default=math.inf)
        no_detection_changed_elements = int(np.count_nonzero(clean != empty_hybrid))
        checks = {
            "gpu_arch_gfx1151": architecture == "gfx1151",
            "target_shape_locked": list(shape) == [3, 512, 960],
            "geometry_locked": (
                geometry.resized_width == 911
                and geometry.resized_height == 512
                and geometry.pad_left == 24
                and geometry.pad_top == 0
            ),
            "deterministic_topk": observed == expected,
            "confidence_order": all(
                first.confidence >= second.confidence
                for first, second in zip(composition.hints, composition.hints[1:], strict=False)
            ),
            "invalid_detections_filtered": len(composition.hints) == 3,
            "no_detection_has_no_hints": not empty_composition.hints,
            "no_detection_pixel_equal_to_clean": no_detection_changed_elements == 0,
            "wrong_frame_rejected_before_publish": (
                mismatch_rejected and mismatch_remained_unpublished
            ),
            "overlay_changed_pixels": changed_pixels > 1000,
            "box_coordinate_max_error_lte_1px": box_mapping_max_error <= 1,
            "padding_unchanged": padding_changed == 0,
            "prompt_number_mapping": (
                "#1 car 0.95; #2 bicycle 0.95; #3 person 0.70" in composition.prompt
            ),
            "metadata_copy_exactly_abi": (
                composition.control_metadata_d2h_bytes == HYBRID_HINT_BUFFER_BYTES
            ),
            "image_host_copy_counters_zero": (
                composer.runtime_info()["image_h2d_bytes"] == 0
                and composer.runtime_info()["image_d2h_bytes"] == 0
            ),
            "global_device_synchronize_absent": not composer.runtime_info()[
                "uses_global_device_synchronize"
            ],
        }
        result = {
            "schema_version": 1,
            "component": "yolo-guided-vlm-hybrid-v1",
            "gpu": {"name": torch.cuda.get_device_name(0), "architecture": architecture},
            "manifest_sha256": manifest_sha256,
            "kernel_library": str(kernel_library),
            "source_shape": list(source_host.shape),
            "target_shape": list(shape),
            "geometry": {
                "resized_width": geometry.resized_width,
                "resized_height": geometry.resized_height,
                "pad_left": geometry.pad_left,
                "pad_top": geometry.pad_top,
            },
            "hints": [
                {
                    "rank": hint.rank,
                    "class_id": hint.class_id,
                    "class_name": hint.class_name,
                    "confidence": hint.confidence,
                    "source_index": hint.source_index,
                    "box": [hint.x1, hint.y1, hint.x2, hint.y2],
                }
                for hint in composition.hints
            ],
            "prompt": composition.prompt,
            "changed_pixels": changed_pixels,
            "padding_changed_pixels": padding_changed,
            "no_detection_changed_elements": no_detection_changed_elements,
            "box_mapping": box_mapping,
            "box_mapping_max_error_pixels": box_mapping_max_error,
            "production_copy_contract": {
                "image_h2d_bytes": 0,
                "image_d2h_bytes": 0,
                "full_detection_tensor_d2h_bytes": 0,
                "control_metadata_d2h_bytes": (
                    composition.control_metadata_d2h_bytes
                    + empty_composition.control_metadata_d2h_bytes
                ),
            },
            "validation_only_copy_contract": {
                "source_h2d_bytes": int(source_host.nbytes),
                "detections_h2d_bytes": int(detections_host.nbytes),
                "image_d2h_bytes": int(clean.nbytes + empty_hybrid.nbytes + hybrid.nbytes),
            },
            "composer": composer.runtime_info(),
            "checks": checks,
            "gate": {"status": "passed" if all(checks.values()) else "failed"},
        }
    finally:
        if clean_lease is not None:
            clean_lease.release()
        if empty_lease is not None:
            empty_lease.release()
        if mismatch_lease is not None:
            mismatch_lease.release()
        if hybrid_lease is not None:
            hybrid_lease.release()
        composer.close()
        preprocessor.close()

    output = (WORKSPACE / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["gate"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
