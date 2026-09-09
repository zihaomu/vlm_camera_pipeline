#!/usr/bin/env python3
"""Audit rocprofv3 copy records for the strict VLM HIP IPC request path."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
OPERATION_NAMES = {2: "H2D", 3: "D2H", 4: "D2D"}
IPC_INPUT_BYTES = 3 * 512 * 960 * 4
QWEN3_VL_EMBEDDING_BYTES = 480 * 4096 * 4 * 4
VISION_POSITION_CONTROL_BYTES = 1920 * 4 * 4
QWEN3_VL_LOGITS_BYTES = 151936 * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="rocprofv3 server-process JSON trace")
    parser.add_argument(
        "--server-log",
        default="output/realtime/llama-server-zerocopy-profiled.log",
    )
    parser.add_argument(
        "--validation-report",
        default="output/realtime/zerocopy-vlm-profiled-check.json",
    )
    parser.add_argument(
        "--output",
        default="output/realtime/zerocopy-vlm-copy-audit.json",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def buffer_records(trace: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    roots = trace.get("rocprofiler-sdk-tool")
    if not isinstance(roots, list) or len(roots) != 1:
        raise ValueError("expected exactly one rocprofiler-sdk-tool root")
    records = roots[0].get("buffer_records")
    if not isinstance(records, dict):
        raise TypeError("trace has no buffer_records")
    return records


def hip_arguments(record: dict[str, Any]) -> dict[str, str]:
    return {
        str(argument["name"]): str(argument["value"])
        for argument in record.get("args", [])
        if isinstance(argument, dict) and "name" in argument and "value" in argument
    }


def marker_bytes(pattern: str, text: str) -> list[int]:
    return [int(value) for value in re.findall(pattern, text)]


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for operation in sorted({int(record["operation"]) for record in records}):
        selected = [record for record in records if int(record["operation"]) == operation]
        sizes = Counter(int(record["bytes"]) for record in selected)
        result.append(
            {
                "operation": operation,
                "direction": OPERATION_NAMES.get(operation, "unknown"),
                "count": len(selected),
                "total_bytes": sum(int(record["bytes"]) for record in selected),
                "sizes": [
                    {"bytes": size, "count": count}
                    for size, count in sorted(sizes.items())
                ],
            }
        )
    return result


def main() -> int:
    args = parse_args()
    trace_path = resolve(args.trace)
    log_path = resolve(args.server_log)
    validation_path = resolve(args.validation_report)
    output_path = resolve(args.output)

    trace = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    buffers = buffer_records(trace)
    records = buffers.get("memory_copy")
    hip_api = buffers.get("hip_api")
    kernel_dispatches = buffers.get("kernel_dispatch")
    if not isinstance(records, list):
        raise TypeError("trace has no memory_copy records")
    if not isinstance(hip_api, list):
        raise TypeError("trace has no hip_api records; rerun rocprofv3 with --hip-trace")
    if not isinstance(kernel_dispatches, list):
        raise TypeError("trace has no kernel dispatches; rerun rocprofv3 with --kernel-trace")

    root = trace["rocprofiler-sdk-tool"][0]
    kernel_names = {
        int(symbol["kernel_id"]): str(symbol.get("formatted_kernel_name", ""))
        for symbol in root.get("kernel_symbols", [])
        if isinstance(symbol, dict) and "kernel_id" in symbol
    }
    rocwmma_dispatches = sum(
        1
        for record in kernel_dispatches
        if kernel_names.get(int(record["dispatch_info"]["kernel_id"]), "").startswith(
            "void flash_attn_ext_f16<"
        )
    )
    rocwmma_expected = (
        validation.get("server", {})
        .get("runtime", {})
        .get("ggml_hip_rocwmma_fattn")
        is True
    )

    ipc_copies = [
        record
        for record in records
        if int(record["operation"]) == 4 and int(record["bytes"]) == IPC_INPUT_BYTES
    ]
    if len(ipc_copies) != 1:
        raise ValueError(f"expected one {IPC_INPUT_BYTES}-byte IPC input D2D, found {len(ipc_copies)}")
    hot_start = int(ipc_copies[0]["start_timestamp"])
    hot_records = [record for record in records if int(record["start_timestamp"]) >= hot_start]

    ready_bytes = marker_bytes(r"HIP device embedding ready[^\n]* bytes=(\d+)", log_text)
    bridge_bytes = marker_bytes(r"HIP embedding D2D complete[^\n]* bytes=(\d+)", log_text)
    input_bytes = marker_bytes(r"HIP IPC v2 D2D complete[^\n]* bytes=(\d+)", log_text)
    input_pointer_log = re.findall(
        r"HIP IPC v2 D2D complete[^\n]* bytes=(\d+) source=(0x[0-9a-f]+) inp_raw=(0x[0-9a-f]+)",
        log_text,
    )
    embedding_pointer_log = re.findall(
        r"HIP embedding D2D complete[^\n]* bytes=(\d+) source=(0x[0-9a-f]+) inp_embd=(0x[0-9a-f]+)",
        log_text,
    )

    traced_d2d = []
    for record in hip_api:
        arguments = hip_arguments(record)
        if arguments.get("kind") != "DeviceToDevice" or "sizeBytes" not in arguments:
            continue
        traced_d2d.append(
            {
                "bytes": int(arguments["sizeBytes"]),
                "source": arguments.get("src", "").lower(),
                "destination": arguments.get("dst", "").lower(),
                "start_timestamp": int(record["start_timestamp"]),
                "correlation_id": int(record["correlation_id"]["internal"]),
            }
        )

    def pointer_copy_is_traced(entry: tuple[str, str, str]) -> bool:
        nbytes, source, destination = entry
        return any(
            copy["bytes"] == int(nbytes)
            and copy["source"] == source.lower()
            and copy["destination"] == destination.lower()
            for copy in traced_d2d
        )

    input_pointer_audit = [pointer_copy_is_traced(entry) for entry in input_pointer_log]
    embedding_pointer_audit = [pointer_copy_is_traced(entry) for entry in embedding_pointer_log]

    h2d_records = [record for record in hot_records if int(record["operation"]) == 2]
    d2h_records = [record for record in hot_records if int(record["operation"]) == 3]
    control_h2d = [
        record for record in h2d_records if int(record["bytes"]) == VISION_POSITION_CONTROL_BYTES
    ]
    sampling_d2h = [
        record for record in d2h_records if int(record["bytes"]) == QWEN3_VL_LOGITS_BYTES
    ]
    unclassified_h2d = [record for record in h2d_records if record not in control_h2d]
    unclassified_d2h = [record for record in d2h_records if record not in sampling_d2h]

    image_embedding_d2h = [
        record for record in hot_records
        if int(record["operation"]) == 3
        and int(record["bytes"]) == QWEN3_VL_EMBEDDING_BYTES
    ]
    bridge_bytes_conserved = (
        ready_bytes == [QWEN3_VL_EMBEDDING_BYTES]
        and sum(bridge_bytes) == QWEN3_VL_EMBEDDING_BYTES
    )
    profiled_semantics_passed = (
        validation.get("numerical_reference", {}).get("passed") is True
        and bool(validation.get("server", {}).get("captions"))
        and validation.get("server", {}).get("ipc_mapping_log_count") == 1
        and validation.get("server", {}).get("ipc_input_d2d_log_count") == 1
        and validation.get("server", {}).get("device_embedding_ready_log_count") == 1
        and validation.get("server", {}).get("device_embedding_bytes_conserved") is True
    )
    passed = (
        profiled_semantics_passed
        and input_bytes == [IPC_INPUT_BYTES]
        and bridge_bytes_conserved
        and input_pointer_audit == [True]
        and embedding_pointer_audit
        and all(embedding_pointer_audit)
        and not image_embedding_d2h
        and len(control_h2d) == 1
        and not unclassified_h2d
        and bool(sampling_d2h)
        and not unclassified_d2h
        and (not rocwmma_expected or rocwmma_dispatches > 0)
    )

    report = {
        "schema_version": 1,
        "component": "qwen3-vl-hip-ipc-v2-copy-audit",
        "status": "passed" if passed else "failed",
        "inputs": {
            "trace": str(trace_path.relative_to(WORKSPACE)),
            "server_log": str(log_path.relative_to(WORKSPACE)),
            "validation_report": str(validation_path.relative_to(WORKSPACE)),
            "profiled_semantics_passed": profiled_semantics_passed,
            "profiled_latency_gate_ignored": True,
        },
        "request_window": {
            "starts_at_first_ipc_input_d2d_timestamp": hot_start,
            "copy_records": len(hot_records),
            "summary": summarize(hot_records),
        },
        "pointer_ledger_log": {
            "ipc_input_d2d_bytes": input_bytes,
            "vision_embedding_ready_bytes": ready_bytes,
            "llm_embedding_d2d_chunks_bytes": bridge_bytes,
            "llm_embedding_bytes_conserved": bridge_bytes_conserved,
        },
        "hip_api_pointer_audit": {
            "input_copy_log_entries": [
                {"bytes": int(size), "source": source, "destination": destination}
                for size, source, destination in input_pointer_log
            ],
            "input_copy_matches_hip_api": input_pointer_audit,
            "embedding_copy_log_entries": [
                {"bytes": int(size), "source": source, "destination": destination}
                for size, source, destination in embedding_pointer_log
            ],
            "embedding_copy_matches_hip_api": embedding_pointer_audit,
            "matched": input_pointer_audit == [True]
            and bool(embedding_pointer_audit)
            and all(embedding_pointer_audit),
        },
        "rocwmma_flash_attention": {
            "build_expected": rocwmma_expected,
            "runtime_kernel_dispatches": rocwmma_dispatches,
            "runtime_verified": not rocwmma_expected or rocwmma_dispatches > 0,
            "kernel_prefix": "void flash_attn_ext_f16<",
        },
        "classified_control_plane": {
            "vision_position_h2d": {
                "count": len(control_h2d),
                "bytes_each": VISION_POSITION_CONTROL_BYTES,
                "total_bytes": sum(int(record["bytes"]) for record in control_h2d),
                "derivation": "1920 vision patches * 4 int32 position components",
            },
            "token_sampling_logits_d2h": {
                "count": len(sampling_d2h),
                "bytes_each": QWEN3_VL_LOGITS_BYTES,
                "total_bytes": sum(int(record["bytes"]) for record in sampling_d2h),
                "derivation": "151936 vocabulary logits * float32 per sampled token",
            },
        },
        "image_derived_host_copy": {
            "h2d_bytes": sum(int(record["bytes"]) for record in unclassified_h2d),
            "d2h_bytes": sum(int(record["bytes"]) for record in unclassified_d2h),
            "qwen3_vl_embedding_d2h_count": len(image_embedding_d2h),
            "qwen3_vl_embedding_bytes": QWEN3_VL_EMBEDDING_BYTES,
        },
        "notes": [
            "rocprofv3 records the IPC input copy as D2D.",
            "HIP API trace matches every logged input and embedding D2D by source pointer, destination pointer, byte count, and DeviceToDevice kind.",
            "This rocprofv3 build does not emit memory_copy activity records for the same-device internal ggml embedding copies, so their API records and byte conservation are audited explicitly.",
            "Vision position metadata and token-sampling logits are classified control-plane transfers, not image pixels or vision embeddings.",
            "On RDNA3, flash_attn_ext_f16 dispatches are the rocWMMA FlashAttention path selected by GGML_HIP_ROCWMMA_FATTN.",
            "Profiler overhead and generated-caption length can miss the 3-second cadence; cadence is validated separately without rocprof instrumentation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_path)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
