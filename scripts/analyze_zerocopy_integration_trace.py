#!/usr/bin/env python3
"""Audit the real-camera YOLO + VLM + EGL ROCm traces end to end."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
DIRECTIONS = {0: "NONE", 1: "H2H", 2: "H2D", 3: "D2H", 4: "D2D"}
IPC_INPUT_BYTES = 3 * 512 * 960 * 4
QWEN3_VL_EMBEDDING_BYTES = 480 * 4096 * 4 * 4
VISION_POSITION_CONTROL_BYTES = 1920 * 4 * 4
QWEN3_VL_LOGITS_BYTES = 151936 * 4
YOLO_OUTPUT_BYTES = 1 * 300 * 6 * 4
CAMERA_EXPORT_ALLOCATION_FLOOR = 4 * 1024 * 1024

KERNELS = {
    "nv12_to_rgb8_bt601_limited_kernel",
    "letterbox_rgb8_to_bchw_f32_kernel",
    "unletterbox_detections_kernel",
    "qwen3_vl_preprocess_rgb8_f32_kernel",
    "rgb8_to_rgba_kernel",
    "detection_boxes_kernel",
    "subtitle_kernel",
    "performance_hud_kernel",
    "hybrid_compact_hints_kernel",
    "hybrid_numbered_boxes_kernel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-trace", required=True)
    parser.add_argument("--server-trace", required=True)
    parser.add_argument(
        "--metrics",
        default="output/realtime/metrics-yolo-vlm-zerocopy-camera-profiled.jsonl",
    )
    parser.add_argument(
        "--server-log",
        default="output/realtime/llama-server-zerocopy.log",
    )
    parser.add_argument(
        "--output",
        default="output/realtime/zerocopy-camera-integration-copy-audit.json",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tool(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    roots = payload.get("rocprofiler-sdk-tool")
    if not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], dict):
        raise ValueError(f"{path} does not contain one rocprofiler-sdk-tool root")
    return roots[0]


def final_summary(path: Path) -> dict[str, Any]:
    final: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("type") == "final_summary":
                final = record
    if final is None:
        raise ValueError(f"{path} has no final_summary record")
    return final


def operation_index(tool: dict[str, Any], kind: str, name: str) -> int | None:
    for record in tool.get("strings", {}).get("buffer_records", []):
        if record.get("kind") != kind:
            continue
        try:
            return record.get("operations", []).index(name)
        except ValueError:
            return None
    return None


def hip_arguments(record: dict[str, Any]) -> dict[str, str]:
    return {
        str(argument["name"]): str(argument["value"])
        for argument in record.get("args", [])
        if isinstance(argument, dict) and "name" in argument and "value" in argument
    }


def hip_copy_counter(
    tool: dict[str, Any], *, start_timestamp: int
) -> Counter[tuple[str, int, str, str]]:
    result: Counter[tuple[str, int, str, str]] = Counter()
    for record in tool.get("buffer_records", {}).get("hip_api", []):
        if int(record.get("start_timestamp", 0)) < start_timestamp:
            continue
        arguments = hip_arguments(record)
        if "kind" not in arguments or "sizeBytes" not in arguments:
            continue
        result[
            (
                arguments["kind"],
                int(arguments["sizeBytes"]),
                arguments.get("src", "").lower(),
                arguments.get("dst", "").lower(),
            )
        ] += 1
    return result


def copy_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter((int(record["operation"]), int(record["bytes"])) for record in records)
    return [
        {
            "direction": DIRECTIONS.get(operation, f"UNKNOWN_{operation}"),
            "bytes": byte_count,
            "count": count,
            "total_bytes": byte_count * count,
        }
        for (operation, byte_count), count in sorted(counter.items())
    ]


def kernel_records(tool: dict[str, Any]) -> list[tuple[str, int]]:
    names = {
        int(symbol["kernel_id"]): str(symbol.get("truncated_kernel_name", ""))
        for symbol in tool.get("kernel_symbols", [])
        if "kernel_id" in symbol
    }
    return [
        (
            names.get(int(record.get("dispatch_info", {}).get("kernel_id", -1)), ""),
            int(record.get("start_timestamp", 0)),
        )
        for record in tool.get("buffer_records", {}).get("kernel_dispatch", [])
    ]


def traced_log_copies(
    pattern: str,
    log_text: str,
    traced: Counter[tuple[str, int, str, str]],
) -> tuple[list[dict[str, Any]], bool]:
    entries = [
        {
            "bytes": int(nbytes),
            "source": source.lower(),
            "destination": destination.lower(),
        }
        for nbytes, source, destination in re.findall(pattern, log_text)
    ]
    requested = Counter(
        (
            "DeviceToDevice",
            entry["bytes"],
            entry["source"],
            entry["destination"],
        )
        for entry in entries
    )
    return entries, all(traced[key] >= count for key, count in requested.items())


def analyze_parent(tool: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    kernels = kernel_records(tool)
    camera_timestamps = [
        timestamp
        for name, timestamp in kernels
        if name == "nv12_to_rgb8_bt601_limited_kernel"
    ]
    if not camera_timestamps:
        raise ValueError("parent trace has no camera NV12 kernel")
    hot_start = min(camera_timestamps)
    hot_kernel_counts = Counter(
        name for name, timestamp in kernels if timestamp >= hot_start and name in KERNELS
    )
    buffers = tool.get("buffer_records", {})
    hot_copies = [
        record
        for record in buffers.get("memory_copy", [])
        if int(record.get("start_timestamp", 0)) >= hot_start
    ]
    subtitle_updates = int(metrics["presenter"]["subtitle_updates"])
    subtitle_total_bytes = int(metrics["presenter"]["subtitle_h2d_bytes"])
    subtitle_bytes = subtitle_total_bytes // subtitle_updates if subtitle_updates else 0
    class_label_bytes = int(metrics["presenter"].get("class_label_h2d_bytes", 0))
    performance_hud_updates = int(metrics["presenter"].get("performance_hud_updates", 0))
    performance_hud_total_bytes = int(
        metrics["presenter"].get("performance_hud_h2d_bytes", 0)
    )
    performance_hud_bytes = (
        performance_hud_total_bytes // performance_hud_updates
        if performance_hud_updates
        else 0
    )
    hybrid = metrics.get("hybrid") or {}
    hybrid_runtime = hybrid.get("runtime") or {}
    hybrid_requests = int(hybrid.get("requests_composed", 0))
    hybrid_metadata_bytes = int(hybrid_runtime.get("hint_buffer_bytes", 0))
    hybrid_device_pointers = {
        hex(int(pointer)).lower()
        for pointer in hybrid_runtime.get("hint_device_pointers", [])
    }
    hybrid_host_pointers = {
        hex(int(pointer)).lower()
        for pointer in hybrid_runtime.get("hint_host_pointers", [])
    }
    hybrid_metadata_copies = [
        record
        for record in hot_copies
        if int(record["operation"]) == 3
        and hybrid_metadata_bytes > 0
        and int(record["bytes"]) == hybrid_metadata_bytes
    ]
    hot_subtitle_copies = [
        record
        for record in hot_copies
        if int(record["operation"]) == 2 and int(record["bytes"]) == subtitle_bytes
    ]
    hot_performance_hud_copies = [
        record
        for record in hot_copies
        if int(record["operation"]) == 2
        and performance_hud_bytes > 0
        and int(record["bytes"]) == performance_hud_bytes
    ]
    unclassified_host_copies = [
        record
        for record in hot_copies
        if not (
            int(record["operation"]) == 2
            and int(record["bytes"]) in {subtitle_bytes, performance_hud_bytes}
            or record in hybrid_metadata_copies
        )
    ]

    hip_copies = hip_copy_counter(tool, start_timestamp=hot_start)
    yolo_d2d_count = sum(
        count
        for (kind, nbytes, _source, _destination), count in hip_copies.items()
        if kind == "DeviceToDevice" and nbytes == YOLO_OUTPUT_BYTES
    )
    hybrid_metadata_api_count = sum(
        count
        for (kind, nbytes, source, destination), count in hip_copies.items()
        if kind == "DeviceToHost"
        and nbytes == hybrid_metadata_bytes
        and source in hybrid_device_pointers
        and destination in hybrid_host_pointers
    )
    unexpected_hip_host_copies = sum(
        count
        for (kind, nbytes, source, destination), count in hip_copies.items()
        if kind in {"HostToDevice", "DeviceToHost"}
        and not (
            kind == "HostToDevice"
            and nbytes in {subtitle_bytes, performance_hud_bytes}
            or kind == "DeviceToHost"
            and nbytes == hybrid_metadata_bytes
            and source in hybrid_device_pointers
            and destination in hybrid_host_pointers
        )
    )

    sync_index = operation_index(tool, "HIP_RUNTIME_API", "hipDeviceSynchronize")
    device_sync_count = sum(
        1
        for record in buffers.get("hip_api", [])
        if record.get("operation") == sync_index
        and int(record.get("start_timestamp", 0)) >= hot_start
    )
    event_record_index = operation_index(tool, "HIP_RUNTIME_API", "hipEventRecord")
    event_wait_index = operation_index(tool, "HIP_RUNTIME_API", "hipStreamWaitEvent")
    event_sync_index = operation_index(tool, "HIP_RUNTIME_API", "hipEventSynchronize")

    def count_event_calls(operation: int | None, handles: set[str]) -> int:
        if operation is None:
            return 0
        return sum(
            1
            for record in buffers.get("hip_api", [])
            if record.get("operation") == operation
            and int(record.get("start_timestamp", 0)) >= hot_start
            and hip_arguments(record).get("event", "").lower() in handles
        )

    vlm_preprocessor = metrics.get("vlm", {}).get("preprocessor") or {}
    base_ready_events = {
        hex(int(event)).lower() for event in vlm_preprocessor.get("base_ready_events", [])
    }
    final_ready_events = {
        hex(int(event)).lower() for event in vlm_preprocessor.get("final_ready_events", [])
    }
    control_ready_events = {
        hex(int(event)).lower()
        for event in hybrid_runtime.get("control_ready_events", [])
    }
    base_ready_record_count = count_event_calls(event_record_index, base_ready_events)
    final_ready_record_count = count_event_calls(event_record_index, final_ready_events)
    base_ready_wait_count = count_event_calls(event_wait_index, base_ready_events)
    control_ready_record_count = count_event_calls(event_record_index, control_ready_events)
    control_ready_sync_count = count_event_calls(event_sync_index, control_ready_events)
    export_index = operation_index(
        tool, "HSA_AMD_EXT_API", "hsa_amd_portable_export_dmabuf"
    )
    export_count = sum(
        1 for record in buffers.get("hsa_api", []) if record.get("operation") == export_index
    )

    frames = int(metrics["frames"])
    yolo_runs = int(metrics["yolo"]["runs"])
    vlm_prepared = int(metrics["vlm"]["preprocessor"]["prepared"])
    expected_exports = int(metrics["camera"]["camera_pool_size"]) + int(
        metrics["presenter"]["present_pool_size"]
    )
    performance_hud_enabled = bool(
        metrics.get("performance_hud", {}).get("enabled_at_start", False)
    )
    checks = {
        "hot_memory_copy_only_classified_control": (
            len(hot_copies)
            == len(hot_subtitle_copies) + len(hot_performance_hud_copies)
            + len(hybrid_metadata_copies)
            and len(hot_subtitle_copies) == max(0, subtitle_updates - 1)
            and len(hot_performance_hud_copies)
            == max(0, performance_hud_updates - 1)
        ),
        "hot_memory_activity_d2h_has_no_unclassified_transfer": (
            sum(int(record["operation"]) == 3 for record in hot_copies)
            == len(hybrid_metadata_copies)
        ),
        "hot_unclassified_host_copy_zero": not unclassified_host_copies,
        "hip_api_unclassified_host_copy_zero": unexpected_hip_host_copies == 0,
        "hybrid_metadata_pointer_ledger_matches": (
            hybrid_metadata_api_count == hybrid_requests
        ),
        "yolo_output_d2d_per_run": yolo_d2d_count == yolo_runs,
        "camera_nv12_kernel_per_frame": (
            hot_kernel_counts["nv12_to_rgb8_bt601_limited_kernel"] == frames
        ),
        "yolo_letterbox_kernel_per_frame": (
            hot_kernel_counts["letterbox_rgb8_to_bchw_f32_kernel"] == frames
        ),
        "yolo_unletterbox_kernel_per_run": (
            hot_kernel_counts["unletterbox_detections_kernel"] == yolo_runs
        ),
        "vlm_preprocess_kernel_per_request": (
            hot_kernel_counts["qwen3_vl_preprocess_rgb8_f32_kernel"] == vlm_prepared
        ),
        "hybrid_topk_kernel_per_request": (
            hot_kernel_counts["hybrid_compact_hints_kernel"] == hybrid_requests
        ),
        "hybrid_overlay_kernel_per_request": (
            hot_kernel_counts["hybrid_numbered_boxes_kernel"] == hybrid_requests
        ),
        "vlm_base_ready_event_record_per_request": (
            base_ready_record_count == vlm_prepared
        ),
        "vlm_final_ready_event_record_per_request": (
            final_ready_record_count == vlm_prepared
        ),
        "hybrid_waits_base_ready_event_per_request": (
            base_ready_wait_count == hybrid_requests
        ),
        "hybrid_control_event_record_per_request": (
            control_ready_record_count == hybrid_requests
        ),
        "hybrid_control_event_sync_per_request": (
            control_ready_sync_count == hybrid_requests
        ),
        "present_rgb_kernel_per_frame": hot_kernel_counts["rgb8_to_rgba_kernel"] == frames,
        "present_subtitle_kernel_per_frame": hot_kernel_counts["subtitle_kernel"] == frames,
        "present_performance_hud_kernel_expected": (
            hot_kernel_counts["performance_hud_kernel"]
            == (frames if performance_hud_enabled else 0)
        ),
        "present_box_kernel_bounded": (
            frames - 1 <= hot_kernel_counts["detection_boxes_kernel"] <= frames
        ),
        "hip_device_synchronize_absent": device_sync_count == 0,
        "camera_and_presenter_hsa_exports_exact": export_count == expected_exports,
        "camera_export_allocation_is_dedicated": (
            int(metrics["camera"]["camera_allocation_bytes"])
            >= CAMERA_EXPORT_ALLOCATION_FLOOR
        ),
    }
    return {
        "passed": all(checks.values()),
        "hot_window_start_timestamp": hot_start,
        "memory_copy_summary": copy_summary(hot_copies),
        "classified_ui_control": {
            "class_label_atlas_upload_before_hot_window": 1 if class_label_bytes else 0,
            "class_label_atlas_bytes": class_label_bytes,
            "initial_performance_hud_upload_before_hot_window": (
                1 if performance_hud_updates else 0
            ),
            "hot_performance_hud_uploads": len(hot_performance_hud_copies),
            "performance_hud_bytes_each": performance_hud_bytes,
            "initial_caption_upload_before_hot_window": 1 if subtitle_updates else 0,
            "hot_caption_uploads": len(hot_subtitle_copies),
            "bytes_each": subtitle_bytes,
            "hybrid_metadata_d2h": {
                "bytes_each": hybrid_metadata_bytes,
                "activity_count": len(hybrid_metadata_copies),
                "pointer_ledger_count": hybrid_metadata_api_count,
                "device_pointers": sorted(hybrid_device_pointers),
                "host_pointers": sorted(hybrid_host_pointers),
            },
        },
        "unclassified_hot_host_copy_count": len(unclassified_host_copies),
        "hip_api_unclassified_host_copy_count": unexpected_hip_host_copies,
        "yolo_output_d2d": {"bytes_each": YOLO_OUTPUT_BYTES, "count": yolo_d2d_count},
        "hybrid_event_ledger": {
            "vlm_prepared": vlm_prepared,
            "hybrid_requests": hybrid_requests,
            "base_ready_record_count": base_ready_record_count,
            "base_ready_wait_count": base_ready_wait_count,
            "final_ready_record_count": final_ready_record_count,
            "control_ready_record_count": control_ready_record_count,
            "control_ready_sync_count": control_ready_sync_count,
        },
        "kernel_dispatches": dict(sorted(hot_kernel_counts.items())),
        "hip_device_synchronize_count": device_sync_count,
        "hsa_dmabuf_export_count": export_count,
        "expected_hsa_dmabuf_export_count": expected_exports,
        "checks": checks,
    }


def analyze_server(
    tool: dict[str, Any], metrics: dict[str, Any], log_text: str
) -> dict[str, Any]:
    buffers = tool.get("buffer_records", {})
    copies = buffers.get("memory_copy", [])
    ipc_copies = [
        record
        for record in copies
        if int(record["operation"]) == 4 and int(record["bytes"]) == IPC_INPUT_BYTES
    ]
    if not ipc_copies:
        raise ValueError("server trace has no VLM IPC input D2D copy")
    hot_start = min(int(record["start_timestamp"]) for record in ipc_copies)
    hot_copies = [
        record for record in copies if int(record.get("start_timestamp", 0)) >= hot_start
    ]
    request_count = int(metrics["worker"]["completed"])
    control_h2d = [
        record
        for record in hot_copies
        if int(record["operation"]) == 2
        and int(record["bytes"]) == VISION_POSITION_CONTROL_BYTES
    ]
    logits_d2h = [
        record
        for record in hot_copies
        if int(record["operation"]) == 3 and int(record["bytes"]) == QWEN3_VL_LOGITS_BYTES
    ]
    unclassified_host_copies = [
        record
        for record in hot_copies
        if int(record["operation"]) in {2, 3}
        and record not in control_h2d
        and record not in logits_d2h
    ]
    kv_checkpoint_d2h = [
        record
        for record in hot_copies
        if int(record["operation"]) == 3 and int(record["bytes"]) == 1_087_488
    ]
    embedding_d2h = [
        record
        for record in hot_copies
        if int(record["operation"]) == 3
        and int(record["bytes"]) == QWEN3_VL_EMBEDDING_BYTES
    ]

    # The HIP API call begins a few microseconds before the corresponding
    # memory-copy activity's start timestamp. Use the full API ledger for
    # pointer matching; unique source/destination pairs keep startup traffic
    # from satisfying a request entry accidentally.
    traced = hip_copy_counter(tool, start_timestamp=0)
    input_log, input_pointer_match = traced_log_copies(
        r"HIP IPC v2 D2D complete[^\n]* bytes=(\d+) source=(0x[0-9a-f]+) "
        r"inp_raw=(0x[0-9a-f]+)",
        log_text,
        traced,
    )
    embedding_log, embedding_pointer_match = traced_log_copies(
        r"HIP embedding D2D complete[^\n]* bytes=(\d+) source=(0x[0-9a-f]+) "
        r"inp_embd=(0x[0-9a-f]+)",
        log_text,
        traced,
    )
    ready_bytes = [
        int(value)
        for value in re.findall(r"HIP device embedding ready[^\n]* bytes=(\d+)", log_text)
    ]
    cache_checkpoint_markers = re.findall(
        r"idle slots saved|saved[^\n]*prompt cache|saving[^\n]*prompt cache",
        log_text,
        flags=re.IGNORECASE,
    )
    ipc_hot = [
        record
        for record in hot_copies
        if int(record["operation"]) == 4 and int(record["bytes"]) == IPC_INPUT_BYTES
    ]
    checks = {
        "ipc_input_d2d_per_request": len(ipc_hot) == request_count,
        "vision_position_h2d_per_request": len(control_h2d) == request_count,
        "token_logits_d2h_present": bool(logits_d2h),
        "unclassified_host_copy_zero": not unclassified_host_copies,
        "image_embedding_d2h_zero": not embedding_d2h,
        "prompt_cache_kv_checkpoint_d2h_zero": not kv_checkpoint_d2h,
        "prompt_cache_disabled_in_log": "prompt cache is disabled" in log_text,
        "prompt_cache_checkpoint_log_zero": not cache_checkpoint_markers,
        "input_pointer_ledger_matches_hip_api": (
            len(input_log) == request_count and input_pointer_match
        ),
        "embedding_pointer_ledger_matches_hip_api": (
            len(embedding_log) == request_count * 2 and embedding_pointer_match
        ),
        "device_embedding_bytes_conserved": (
            ready_bytes == [QWEN3_VL_EMBEDDING_BYTES] * request_count
            and sum(entry["bytes"] for entry in embedding_log)
            == QWEN3_VL_EMBEDDING_BYTES * request_count
        ),
    }
    return {
        "passed": all(checks.values()),
        "hot_window_start_timestamp": hot_start,
        "memory_copy_summary": copy_summary(hot_copies),
        "classified_control_plane": {
            "vision_position_h2d": {
                "bytes_each": VISION_POSITION_CONTROL_BYTES,
                "count": len(control_h2d),
            },
            "token_sampling_logits_d2h": {
                "bytes_each": QWEN3_VL_LOGITS_BYTES,
                "count": len(logits_d2h),
            },
        },
        "image_and_embedding_host_copy": {
            "h2d_bytes": sum(
                int(record["bytes"])
                for record in unclassified_host_copies
                if int(record["operation"]) == 2
            ),
            "d2h_bytes": sum(
                int(record["bytes"])
                for record in unclassified_host_copies
                if int(record["operation"]) == 3
            ),
            "qwen3_vl_embedding_d2h_count": len(embedding_d2h),
        },
        "prompt_cache": {
            "configured_ram_mib": metrics["vlm"].get("prompt_cache_ram_mib"),
            "kv_checkpoint_d2h_bytes_each": 1_087_488,
            "kv_checkpoint_d2h_count": len(kv_checkpoint_d2h),
            "checkpoint_log_markers": cache_checkpoint_markers,
        },
        "pointer_ledger": {
            "ipc_input_entries": input_log,
            "ipc_input_matches_hip_api": input_pointer_match,
            "embedding_entries": embedding_log,
            "embedding_matches_hip_api": embedding_pointer_match,
            "embedding_ready_bytes": ready_bytes,
        },
        "checks": checks,
    }


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    parent_path = resolve(args.parent_trace)
    server_path = resolve(args.server_trace)
    metrics_path = resolve(args.metrics)
    log_path = resolve(args.server_log)
    output_path = resolve(args.output)
    metrics = final_summary(metrics_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    presenter_library_value = metrics.get("presenter", {}).get("library")
    presenter_library = (
        resolve(str(presenter_library_value)) if presenter_library_value else None
    )
    presenter_recorded_sha256 = metrics.get("presenter", {}).get("library_sha256")
    presenter_binary_identity_match = bool(
        presenter_library
        and presenter_library.is_file()
        and presenter_recorded_sha256
        and sha256(presenter_library) == presenter_recorded_sha256
    )

    parent_tool = load_tool(parent_path)
    parent = analyze_parent(parent_tool, metrics)
    parent_pid = int(parent_tool.get("metadata", {}).get("pid", 0))
    del parent_tool
    gc.collect()

    server_tool = load_tool(server_path)
    server = analyze_server(server_tool, metrics, log_text)
    server_pid = int(server_tool.get("metadata", {}).get("pid", 0))
    del server_tool
    gc.collect()

    runtime_checks = metrics.get("acceptance", {}).get("runtime_checks", {})
    runtime = {
        "completed": metrics.get("status") == "completed",
        "short_runtime_passed": metrics.get("acceptance", {}).get("short_runtime_passed")
        is True,
        "all_runtime_checks_passed": bool(runtime_checks)
        and all(value is True for value in runtime_checks.values()),
        "camera_start_attempts": metrics.get("startup", {}).get("camera_attempts"),
        "camera_allocation_bytes": metrics.get("camera", {}).get(
            "camera_allocation_bytes"
        ),
        "frames": metrics.get("frames"),
        "effective_fps": metrics.get("effective_fps"),
        "yolo_completed": metrics.get("yolo_worker", {}).get("completed"),
        "vlm_completed": metrics.get("worker", {}).get("completed"),
        "presenter_library": relative(presenter_library) if presenter_library else None,
        "presenter_library_sha256": presenter_recorded_sha256,
        "presenter_binary_identity_match": presenter_binary_identity_match,
    }
    passed = parent["passed"] and server["passed"] and all(
        runtime[key]
        for key in (
            "completed",
            "short_runtime_passed",
            "all_runtime_checks_passed",
            "presenter_binary_identity_match",
        )
    )
    report = {
        "schema_version": 1,
        "component": "real-camera-yolo-vlm-egl-copy-audit",
        "status": "passed" if passed else "failed",
        "scope": (
            "steady-state image path after first camera/VLM request; model-load transfers "
            "are outside the audited hot windows"
        ),
        "inputs": {
            "parent_trace": relative(parent_path),
            "parent_trace_sha256": sha256(parent_path),
            "parent_pid": parent_pid,
            "server_trace": relative(server_path),
            "server_trace_sha256": sha256(server_path),
            "server_pid": server_pid,
            "metrics": relative(metrics_path),
            "server_log": relative(log_path),
        },
        "runtime": runtime,
        "parent_camera_yolo_present": parent,
        "server_vlm": server,
        "production_contract": {
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "vision_embedding_d2h_bytes": 0,
            "global_device_synchronize_count": 0,
            "allowed_host_transfers": [
                "static COCO80 class-label glyph atlas (startup H2D)",
                "performance HUD glyph mask (startup/change H2D when enabled)",
                "subtitle glyph alpha control resource (H2D)",
                "Hybrid top-8 detection summary (one 260-byte D2H per request)",
                "VLM vision position metadata (H2D)",
                "VLM token sampling logits (D2H)",
            ],
        },
        "notes": [
            "Camera frames remain in HIP-owned DMA-BUFs imported by V4L2 and ISP PRIME/GART.",
            "YOLO input/output, VLM image preprocessing, and VLM embeddings remain device-resident.",
            "Hybrid reads back only a pointer-audited fixed 260-byte detection summary; full [300,6] output remains on the GPU.",
            "The prompt RAM cache is disabled because changing camera prompts caused avoidable KV checkpoint D2H traffic.",
            "Class-label, performance-HUD and caption glyphs, position metadata, and sampling logits are control-plane resources, not image payloads.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": str(output_path)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
