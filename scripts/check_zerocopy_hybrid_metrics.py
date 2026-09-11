#!/usr/bin/env python3
"""Validate a real-camera Hybrid JSONL run against the H4/H5 runtime contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
HYBRID_INPUT_MODE = "hybrid-numbered-boxes-v1"
HYBRID_HINT_BUFFER_BYTES = 260
FORBIDDEN_CAPTION_PATTERN = re.compile(
    r"(?:#\d+|\bbox(?:es)?\b|\bhints?\b|\bids?\b|\bscores?\b|"
    r"\bconfidence\b|\bdetectors?\b)",
    re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--minimum-seconds", type=float, default=21.0)
    parser.add_argument("--minimum-requests", type=int, default=7)
    parser.add_argument(
        "--output",
        default="output/realtime/zerocopy-hybrid-metrics-check.json",
    )
    return parser


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (WORKSPACE / candidate).resolve()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: record is not an object")
            records.append(record)
    return records


def _single_record(records: list[dict[str, Any]], record_type: str) -> dict[str, Any]:
    matching = [record for record in records if record.get("type") == record_type]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one {record_type!r} record, found {len(matching)}")
    return matching[0]


def _caption_contract(caption: Any) -> bool:
    if not isinstance(caption, str) or not caption:
        return False
    if not caption.isascii() or "\n" in caption or not caption.endswith("."):
        return False
    words = caption[:-1].split()
    return 1 <= len(words) <= 16 and FORBIDDEN_CAPTION_PATTERN.search(caption) is None


def audit_records(
    records: list[dict[str, Any]], *, minimum_seconds: float, minimum_requests: int
) -> dict[str, Any]:
    capability = _single_record(records, "capability_manifest")
    summary = _single_record(records, "final_summary")
    requests = [record for record in records if record.get("type") == "hybrid_request"]
    results = [record for record in records if record.get("type") == "vlm_result"]
    request_by_id = {
        record.get("request_id"): record
        for record in requests
        if isinstance(record.get("request_id"), int)
    }
    result_by_id = {
        record.get("request_id"): record
        for record in results
        if isinstance(record.get("request_id"), int)
    }
    matched_ids = sorted(set(request_by_id) & set(result_by_id))
    hybrid_summary = summary.get("hybrid") or {}
    hybrid_runtime = hybrid_summary.get("runtime") or {}
    copy_contract = summary.get("copy_contract") or {}
    runtime_checks = summary.get("acceptance", {}).get("runtime_checks", {})
    worker = summary.get("worker") or {}
    yolo_worker = summary.get("yolo_worker") or {}
    camera = summary.get("camera") or {}
    presenter = summary.get("presenter") or {}
    vlm = summary.get("vlm") or {}
    preprocessor = vlm.get("preprocessor") or {}
    request_count = len(requests)
    result_count = len(results)
    expected_metadata_bytes = request_count * HYBRID_HINT_BUFFER_BYTES

    checks = {
        "status_completed": summary.get("status") == "completed",
        "duration_gte_minimum": float(summary.get("duration_seconds", 0.0))
        >= minimum_seconds,
        "capability_declares_hybrid": (
            capability.get("scheduler", {}).get("vlm_input_mode") == "hybrid"
            and capability.get("hybrid", {}).get("input_mode") == HYBRID_INPUT_MODE
        ),
        "summary_declares_hybrid": hybrid_summary.get("input_mode") == HYBRID_INPUT_MODE,
        "minimum_request_count": request_count >= minimum_requests,
        "request_result_counts_equal": request_count == result_count,
        "request_ids_unique_and_correlated": (
            len(request_by_id) == request_count
            and len(result_by_id) == result_count
            and set(request_by_id) == set(result_by_id)
        ),
        "request_result_frame_ids_match": all(
            request_by_id[request_id].get("camera_frame_id")
            == request_by_id[request_id].get("yolo_source_frame_id")
            == result_by_id[request_id].get("source_frame_id")
            for request_id in matched_ids
        )
        and len(matched_ids) == request_count,
        "every_request_submitted": all(record.get("submitted") is True for record in requests),
        "every_request_exact_frame": all(
            record.get("exact_frame_match") is True for record in requests
        ),
        "hint_count_bounded": all(
            isinstance(record.get("hint_count"), int)
            and 0 <= int(record["hint_count"]) <= 8
            for record in requests
        ),
        "prompt_characters_bounded": all(
            isinstance(record.get("prompt_characters"), int)
            and 0 < int(record["prompt_characters"]) <= 2048
            for record in requests
        ),
        "metadata_exactly_one_abi_buffer_per_request": (
            all(
                record.get("control_metadata_d2h_bytes")
                == HYBRID_HINT_BUFFER_BYTES
                for record in requests
            )
            and hybrid_summary.get("control_metadata_d2h_bytes")
            == expected_metadata_bytes
            and hybrid_runtime.get("control_metadata_d2h_bytes")
            == expected_metadata_bytes
            and copy_contract.get("hybrid_control_metadata_d2h_bytes")
            == expected_metadata_bytes
        ),
        "result_errors_zero": all(record.get("error") is None for record in results),
        "caption_contract_passed": all(
            _caption_contract(record.get("caption")) for record in results
        ),
        "per_request_token_metrics_present": all(
            isinstance(record.get("prompt_tokens"), int)
            and int(record["prompt_tokens"]) > 0
            and isinstance(record.get("generated_tokens"), int)
            and 0 < int(record["generated_tokens"]) <= 32
            for record in results
        ),
        "fixed_three_second_cadence": summary.get("vlm_ms", {}).get("interval_seconds")
        == 3.0,
        "display_fps_gte_29": float(summary.get("effective_fps", 0.0)) >= 29.0,
        "yolo_completion_fps_gte_25": float(
            summary.get("yolo_ms", {}).get("completion_fps", 0.0)
        )
        >= 25.0,
        "vlm_start_error_p95_lte_100ms": float(
            summary.get("vlm_ms", {}).get("start_error_p95_ms", float("inf"))
        )
        <= 100.0,
        "worker_counts_exact": (
            worker.get("submitted") == request_count
            and worker.get("completed") == result_count
            and worker.get("failed") == 0
            and worker.get("dropped_busy") == 0
            and yolo_worker.get("protected_submitted") == request_count
            and yolo_worker.get("protected_completed") == request_count
        ),
        "all_runtime_checks_true": bool(runtime_checks)
        and all(value is True for value in runtime_checks.values()),
        "all_camera_frames_requeued_and_presented": (
            camera.get("frames_acquired") == camera.get("frames_requeued")
            == summary.get("frames")
            == presenter.get("frames_presented")
        ),
        "production_image_and_detection_host_copy_zero": (
            copy_contract.get("image_h2d_bytes") == 0
            and copy_contract.get("image_d2h_bytes") == 0
            and copy_contract.get("full_detection_tensor_d2h_bytes") == 0
            and hybrid_runtime.get("image_h2d_bytes") == 0
            and hybrid_runtime.get("image_d2h_bytes") == 0
            and preprocessor.get("production_image_h2d_bytes") == 0
            and preprocessor.get("production_image_d2h_bytes") == 0
        ),
        "global_device_synchronize_absent": (
            hybrid_runtime.get("uses_global_device_synchronize") is False
            and preprocessor.get("uses_global_device_synchronize") is False
            and camera.get("global_device_synchronize") is False
            and summary.get("yolo", {}).get("global_device_synchronize") is False
        ),
        "leases_and_queues_drained": (
            camera.get("active_clean_leases") == 0
            and worker.get("outstanding") is False
            and worker.get("queue_depth") == 0
            and yolo_worker.get("outstanding") is False
            and yolo_worker.get("queue_depth") == 0
        ),
    }
    return {
        "schema_version": 1,
        "scope": "real-camera Hybrid H4/H5 metrics contract",
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "duration_seconds": summary.get("duration_seconds"),
            "frames": summary.get("frames"),
            "effective_fps": summary.get("effective_fps"),
            "yolo_completion_fps": summary.get("yolo_ms", {}).get("completion_fps"),
            "vlm_latency_ms": summary.get("vlm_ms"),
            "hybrid_request_count": request_count,
            "vlm_result_count": result_count,
            "matched_request_ids": matched_ids,
            "metadata_d2h_bytes": expected_metadata_bytes,
            "captions": [record.get("caption") for record in results],
            "prompt_tokens": [record.get("prompt_tokens") for record in results],
            "generated_tokens": [record.get("generated_tokens") for record in results],
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.minimum_seconds <= 0 or args.minimum_requests <= 0:
        raise ValueError("minimum duration and request count must be positive")
    metrics_path = resolve(args.metrics)
    report = audit_records(
        load_records(metrics_path),
        minimum_seconds=args.minimum_seconds,
        minimum_requests=args.minimum_requests,
    )
    report["metrics"] = str(metrics_path)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
