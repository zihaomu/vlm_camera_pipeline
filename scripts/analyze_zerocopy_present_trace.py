#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DIRECTIONS = {
    0: "NONE",
    1: "H2H",
    2: "H2D",
    3: "D2H",
    4: "D2D",
}


def _parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit the strict HIP/EGL presenter ROCm trace")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "output/realtime/zerocopy-present-copy-audit.json",
    )
    return parser.parse_args()


def _tool(payload: dict[str, Any]) -> dict[str, Any]:
    tools = payload.get("rocprofiler-sdk-tool")
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
        raise ValueError("trace does not contain exactly one rocprofiler-sdk-tool payload")
    return tools[0]


def _operation_index(tool: dict[str, Any], kind: str, name: str) -> int | None:
    records = tool.get("strings", {}).get("buffer_records", [])
    for record in records:
        if record.get("kind") == kind:
            operations = record.get("operations", [])
            try:
                return operations.index(name)
            except ValueError:
                return None
    return None


def main() -> int:
    args = _parse_args()
    trace = _tool(json.loads(args.trace.read_text(encoding="utf-8")))
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    buffer_records = trace.get("buffer_records", {})
    copies = buffer_records.get("memory_copy", [])
    copy_counter = Counter((record.get("operation"), record.get("bytes")) for record in copies)
    copy_summary = [
        {
            "direction": DIRECTIONS.get(int(operation), f"UNKNOWN_{operation}"),
            "bytes": int(byte_count),
            "count": count,
            "total_bytes": int(byte_count) * count,
        }
        for (operation, byte_count), count in sorted(copy_counter.items())
    ]

    performance = validation["performance"]
    data_path = validation["data_path"]
    requested_frames = int(performance["requested_frames"])
    subtitle_bytes = int(data_path["subtitle_h2d_bytes"])
    class_label_bytes = int(data_path["class_label_h2d_bytes"])
    performance_hud_bytes = int(data_path["performance_hud_h2d_bytes"])
    frame_width, frame_height = map(int, data_path["frame_size"])
    image_sizes = {frame_width * frame_height * 3, frame_width * frame_height * 4}
    image_host_copies = [
        record
        for record in copies
        if record.get("operation") in {2, 3} and int(record.get("bytes", 0)) in image_sizes
    ]
    d2h_records = [record for record in copies if record.get("operation") == 3]
    ui_resource_sizes = {subtitle_bytes, class_label_bytes, performance_hud_bytes}
    non_ui_h2d = [
        record
        for record in copies
        if record.get("operation") == 2
        and int(record.get("bytes", 0)) not in ui_resource_sizes
    ]
    ui_h2d_counts = Counter(
        int(record.get("bytes", 0))
        for record in copies
        if record.get("operation") == 2
        and int(record.get("bytes", 0)) in ui_resource_sizes
    )

    kernel_symbols = {
        int(symbol["kernel_id"]): symbol.get("truncated_kernel_name", "")
        for symbol in trace.get("kernel_symbols", [])
        if "kernel_id" in symbol
    }
    kernel_counts: Counter[str] = Counter()
    for record in buffer_records.get("kernel_dispatch", []):
        kernel_id = int(record.get("dispatch_info", {}).get("kernel_id", -1))
        name = kernel_symbols.get(kernel_id, "")
        if name in {
            "rgb8_to_rgba_kernel",
            "detection_boxes_kernel",
            "subtitle_kernel",
            "performance_hud_kernel",
        }:
            kernel_counts[name] += 1

    hip_device_sync_index = _operation_index(
        trace, "HIP_RUNTIME_API", "hipDeviceSynchronize"
    )
    hip_device_sync_count = sum(
        1
        for record in buffer_records.get("hip_api", [])
        if record.get("operation") == hip_device_sync_index
    )
    hsa_export_index = _operation_index(
        trace, "HSA_AMD_EXT_API", "hsa_amd_portable_export_dmabuf"
    )
    hsa_export_count = sum(
        1
        for record in buffer_records.get("hsa_api", [])
        if record.get("operation") == hsa_export_index
    )
    checks = {
        "validation_passed": validation.get("status") == "passed",
        "only_three_ui_resource_h2d": len(copies) == 3
        and ui_h2d_counts[subtitle_bytes] == 1
        and ui_h2d_counts[class_label_bytes] == 1
        and ui_h2d_counts[performance_hud_bytes] == 1,
        "frame_image_h2d_d2h_zero": not image_host_copies,
        "all_d2h_zero": not d2h_records,
        "non_ui_resource_h2d_zero": not non_ui_h2d,
        "rgb_kernel_per_frame": kernel_counts["rgb8_to_rgba_kernel"] == requested_frames,
        "box_kernel_per_frame": kernel_counts["detection_boxes_kernel"] == requested_frames,
        "subtitle_kernel_per_frame": kernel_counts["subtitle_kernel"] == requested_frames,
        "performance_hud_kernel_per_frame": (
            kernel_counts["performance_hud_kernel"] == requested_frames
        ),
        "hip_device_synchronize_absent": hip_device_sync_count == 0,
        "two_hsa_dmabuf_exports": hsa_export_count == 2,
    }
    report = {
        "schema_version": 1,
        "component": "hip-egl-present-copy-audit",
        "status": "passed" if all(checks.values()) else "failed",
        "inputs": {
            "trace": str(args.trace),
            "validation_report": str(args.validation_report),
        },
        "memory_copy_summary": copy_summary,
        "classified_ui_control_transfer": {
            "subtitle_alpha_h2d_bytes": subtitle_bytes,
            "subtitle_count": ui_h2d_counts[subtitle_bytes],
            "class_label_atlas_h2d_bytes": class_label_bytes,
            "class_label_atlas_count": ui_h2d_counts[class_label_bytes],
            "performance_hud_h2d_bytes": performance_hud_bytes,
            "performance_hud_count": ui_h2d_counts[performance_hud_bytes],
        },
        "image_host_copy": {
            "h2d_bytes": sum(
                int(record.get("bytes", 0))
                for record in image_host_copies
                if record.get("operation") == 2
            ),
            "d2h_bytes": sum(
                int(record.get("bytes", 0))
                for record in image_host_copies
                if record.get("operation") == 3
            ),
        },
        "compositor_kernel_dispatches": dict(kernel_counts),
        "hip_device_synchronize_count": hip_device_sync_count,
        "hsa_dmabuf_export_count": hsa_export_count,
        "checks": checks,
        "notes": [
            "The only HIP memory-copy activity is one static COCO80 class-label atlas upload, one subtitle glyph-alpha upload, and one performance-HUD glyph upload.",
            "The frame and detections enter through existing HIP device pointers; present surfaces are HIP allocations exported as DMA-BUF EGLImages.",
            "Per-slot hipEventSynchronize is an explicit GPU-to-graphics ownership boundary and is not a global device synchronization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
