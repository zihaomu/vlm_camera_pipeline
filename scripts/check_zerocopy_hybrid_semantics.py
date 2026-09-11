#!/usr/bin/env python3
"""Run fixed-image clean/Hybrid and adversarial-hint semantic validation on GPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ["ULTRALYTICS_MIGRAPHX_STRICT"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
KERNEL_LIBRARY = (
    WORKSPACE
    / ".local/zerocopy-gfx1151/lib/libvlm_camera_zerocopy_kernels.so.0.1.0"
)
FORBIDDEN_CAPTION_PATTERN = re.compile(
    r"(?:#\d+|\bbox(?:es)?\b|\bhints?\b|\bids?\b|\bscores?\b|"
    r"\bconfidence\b|\bdetectors?\b)",
    re.IGNORECASE,
)

from src.zerocopy_hybrid import (
    HYBRID_HINT_BUFFER_BYTES,
    HybridComposer,
    HybridComposition,
    default_coco80_manifest,
)
from src.zerocopy_vlm import LlamaCppIpcConfig, ZeroCopyLlamaCppVlm
from src.zerocopy_yolo import GpuDetectionLease, StrictMIGraphXYolo


@dataclass(frozen=True, slots=True)
class SemanticCase:
    name: str
    path: str
    expected_terms: tuple[str, ...]
    forbidden_objects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SyntheticDetection:
    frame_id: int
    tensor: Any
    ready_event: Any


CASES = (
    SemanticCase(
        "tiger",
        "third_party/notebook/opencv_amd_gpu/data/Bengal_tiger_small.jpg",
        ("tiger", "animal", "big cat"),
        ("airplane", "aircraft", "vehicle", "flower"),
    ),
    SemanticCase(
        "lotus",
        "third_party/llama.cpp-vlm-zerocopy/tools/ui/tests/stories/fixtures/assets/beautiful-flowers-lotus.webp",
        ("lotus", "flower", "blossom"),
        ("airplane", "aircraft", "person", "dog", "tiger"),
    ),
    SemanticCase(
        "dog",
        "third_party/opencv-hip/samples/data/chicky_512.png",
        ("dog", "puppy", "animal"),
        ("airplane", "aircraft", "flower", "tiger"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="output/realtime/zerocopy-hybrid-semantics-check.json"
    )
    parser.add_argument(
        "--server-log", default="output/realtime/llama-server-hybrid-semantics.log"
    )
    return parser


def caption_contract(caption: str, *, constrained: bool) -> bool:
    if not caption.strip() or FORBIDDEN_CAPTION_PATTERN.search(caption):
        return False
    if not constrained:
        return True
    return (
        caption.isascii()
        and "\n" not in caption
        and caption.endswith(".")
        and 1 <= len(caption[:-1].split()) <= 16
    )


def contains_expected(caption: str, case: SemanticCase) -> bool:
    normalized = caption.casefold()
    return any(term in normalized for term in case.expected_terms)


def avoids_forbidden_objects(caption: str, case: SemanticCase) -> bool:
    normalized = caption.casefold()
    return all(term not in normalized for term in case.forbidden_objects)


def load_gpu_image(case: SemanticCase) -> tuple[Any, int, Any]:
    path = WORKSPACE / case.path
    with Image.open(path) as source:
        resized = source.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
        contiguous = np.asarray(resized).copy()
    source_gpu = torch.as_tensor(contiguous, device="cuda:0")
    ready = torch.cuda.Event(enable_timing=False, blocking=True)
    ready.record(torch.cuda.current_stream(0))
    return source_gpu, int(contiguous.nbytes), ready


def clean_caption(
    vlm: ZeroCopyLlamaCppVlm, source: Any, ready_event: Any
) -> tuple[str, float, dict[str, Any]]:
    stream = torch.cuda.current_stream(0)
    lease = vlm.prepare_gpu_pointer(
        source_pointer=source.data_ptr(),
        source_pitch=source.stride(0),
        source_width=1280,
        source_height=720,
        source_is_bgr=False,
        source_ready_event=ready_event.cuda_event,
        stream=stream.cuda_stream,
    )
    if lease is None:
        raise RuntimeError("clean semantic request exhausted the fixed VLM IPC pool")
    started = time.perf_counter()
    caption = vlm.caption_prepared(lease)
    return caption, (time.perf_counter() - started) * 1000.0, vlm.last_request_performance()


def hybrid_caption(
    vlm: ZeroCopyLlamaCppVlm,
    composer: HybridComposer,
    source: Any,
    ready_event: Any,
    detection: GpuDetectionLease | SyntheticDetection,
) -> tuple[str, HybridComposition, float, dict[str, Any]]:
    stream = torch.cuda.current_stream(0)
    lease = vlm.prepare_gpu_pointer(
        source_pointer=source.data_ptr(),
        source_pitch=source.stride(0),
        source_width=1280,
        source_height=720,
        source_is_bgr=False,
        source_ready_event=ready_event.cuda_event,
        stream=stream.cuda_stream,
        defer_final_ready=True,
    )
    if lease is None:
        raise RuntimeError("Hybrid semantic request exhausted the fixed VLM IPC pool")
    composition = composer.finalize(
        frame_id=detection.frame_id,
        image_lease=lease,
        detection=detection,  # type: ignore[arg-type]
        stream=stream.cuda_stream,
    )
    started = time.perf_counter()
    caption = vlm.caption_prepared(
        lease,
        prompt=composition.prompt,
        stop_at_first_sentence=True,
    )
    return (
        caption,
        composition,
        (time.perf_counter() - started) * 1000.0,
        vlm.last_request_performance(),
    )


def synthetic_detection(
    *, frame_id: int, class_id: int | None
) -> SyntheticDetection:
    host = np.zeros((300, 6), dtype=np.float32)
    if class_id is not None:
        host[0] = [0.0, 0.0, 1279.0, 719.0, 0.99, float(class_id)]
    tensor = torch.as_tensor(host, device="cuda:0")
    ready = torch.cuda.Event(enable_timing=False, blocking=True)
    ready.record(torch.cuda.current_stream(0))
    return SyntheticDetection(frame_id=frame_id, tensor=tensor, ready_event=ready)


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable; refusing CPU semantic validation")
    architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if architecture != "gfx1151":
        raise RuntimeError(f"expected gfx1151, got {architecture!r}")

    yolo = StrictMIGraphXYolo(
        WORKSPACE / "models/yolo26x.onnx",
        cache_dir=WORKSPACE
        / "models/ort-migraphx-cache/gfx1151-yolo26x-strict-iobinding-v1",
        ultralytics_repository=WORKSPACE / "third_party/ultralytics",
        kernel_library=KERNEL_LIBRARY,
        confidence=0.5,
        input_pool_size=2,
        output_pool_size=3,
    )
    yolo.warmup(iterations=2)
    composer = HybridComposer(
        kernel_library=KERNEL_LIBRARY,
        coco_manifest=default_coco80_manifest(WORKSPACE),
        model_names=yolo.contract["names"],
        confidence=0.5,
        max_hints=8,
    )
    vlm = ZeroCopyLlamaCppVlm(
        LlamaCppIpcConfig(max_tokens=32, log_path=args.server_log), workspace=WORKSPACE
    ).start()
    records: list[dict[str, Any]] = []
    validation_h2d_bytes = 0
    try:
        tiger_source = None
        tiger_ready = None
        for frame_id, case in enumerate(CASES, start=1):
            source, source_bytes, ready = load_gpu_image(case)
            validation_h2d_bytes += source_bytes
            clean, clean_ms, clean_performance = clean_caption(vlm, source, ready)
            detection = yolo.infer_rgb8_pointer(
                frame_id=frame_id,
                source_pointer=source.data_ptr(),
                source_pitch=source.stride(0),
                source_width=1280,
                source_height=720,
                source_is_bgr=False,
                source_ready_event=ready.cuda_event,
            )
            if detection is None:
                raise RuntimeError("semantic YOLO validation exhausted a fixed pool")
            try:
                hybrid, composition, hybrid_ms, hybrid_performance = hybrid_caption(
                    vlm, composer, source, ready, detection
                )
            finally:
                detection.release()
            records.append(
                {
                    "case": case.name,
                    "image": case.path,
                    "expected_terms": list(case.expected_terms),
                    "clean": {
                        "caption": clean,
                        "latency_ms": clean_ms,
                        "performance": clean_performance,
                        "primary_hit": contains_expected(clean, case),
                        "avoids_forbidden_objects": avoids_forbidden_objects(clean, case),
                        "caption_contract": caption_contract(clean, constrained=False),
                    },
                    "hybrid": {
                        "caption": hybrid,
                        "latency_ms": hybrid_ms,
                        "performance": hybrid_performance,
                        "hints": [
                            {
                                "rank": hint.rank,
                                "class": hint.class_name,
                                "confidence": hint.confidence,
                            }
                            for hint in composition.hints
                        ],
                        "prompt": composition.prompt,
                        "primary_hit": contains_expected(hybrid, case),
                        "avoids_forbidden_objects": avoids_forbidden_objects(hybrid, case),
                        "caption_contract": caption_contract(hybrid, constrained=True),
                    },
                }
            )
            if case.name == "tiger":
                tiger_source = source
                tiger_ready = ready

        if tiger_source is None or tiger_ready is None:
            raise RuntimeError("tiger adversarial source was not retained")
        tiger_case = CASES[0]
        wrong_detection = synthetic_detection(frame_id=101, class_id=4)
        validation_h2d_bytes += 300 * 6 * 4
        wrong_caption, wrong_composition, wrong_ms, wrong_performance = hybrid_caption(
            vlm, composer, tiger_source, tiger_ready, wrong_detection
        )
        omitted_detection = synthetic_detection(frame_id=102, class_id=None)
        validation_h2d_bytes += 300 * 6 * 4
        omitted_caption, omitted_composition, omitted_ms, omitted_performance = hybrid_caption(
            vlm, composer, tiger_source, tiger_ready, omitted_detection
        )
        adversarial = {
            "wrong_hint": {
                "injected": "#1 airplane 0.99",
                "prompt": wrong_composition.prompt,
                "caption": wrong_caption,
                "latency_ms": wrong_ms,
                "performance": wrong_performance,
                "primary_hit": contains_expected(wrong_caption, tiger_case),
                "wrong_hint_not_inherited": all(
                    term not in wrong_caption.casefold() for term in ("airplane", "aircraft")
                ),
                "caption_contract": caption_contract(wrong_caption, constrained=True),
            },
            "omitted_hint": {
                "prompt": omitted_composition.prompt,
                "hint_count": len(omitted_composition.hints),
                "caption": omitted_caption,
                "latency_ms": omitted_ms,
                "performance": omitted_performance,
                "primary_hit": contains_expected(omitted_caption, tiger_case),
                "caption_contract": caption_contract(omitted_caption, constrained=True),
            },
        }
        clean_hits = sum(record["clean"]["primary_hit"] for record in records)
        hybrid_hits = sum(record["hybrid"]["primary_hit"] for record in records)
        runtime = {
            "yolo": yolo.runtime_info(),
            "vlm": vlm.runtime_info(),
            "hybrid": composer.runtime_info(),
        }
        checks = {
            "gpu_arch_gfx1151": architecture == "gfx1151",
            "clean_primary_target_hits_all": clean_hits == len(records),
            "hybrid_primary_target_hits_all": hybrid_hits == len(records),
            "hybrid_primary_hits_not_worse_than_clean": hybrid_hits >= clean_hits,
            "clean_captions_do_not_expose_detector_ui": all(
                record["clean"]["caption_contract"] for record in records
            ),
            "hybrid_captions_follow_contract": all(
                record["hybrid"]["caption_contract"] for record in records
            ),
            "clean_captions_avoid_forbidden_objects": all(
                record["clean"]["avoids_forbidden_objects"] for record in records
            ),
            "hybrid_captions_avoid_forbidden_objects": all(
                record["hybrid"]["avoids_forbidden_objects"] for record in records
            ),
            "wrong_airplane_hint_present_in_prompt": "#1 airplane 0.99"
            in wrong_composition.prompt,
            "wrong_hint_not_inherited": adversarial["wrong_hint"][
                "wrong_hint_not_inherited"
            ],
            "wrong_hint_visual_target_preserved": adversarial["wrong_hint"]["primary_hit"],
            "omitted_hint_count_zero": adversarial["omitted_hint"]["hint_count"] == 0,
            "omitted_hint_visual_target_preserved": adversarial["omitted_hint"][
                "primary_hit"
            ],
            "adversarial_captions_follow_contract": (
                adversarial["wrong_hint"]["caption_contract"]
                and adversarial["omitted_hint"]["caption_contract"]
            ),
            "production_image_host_copy_counters_zero": (
                runtime["yolo"]["image_h2d_bytes"] == 0
                and runtime["yolo"]["image_d2h_bytes"] == 0
                and runtime["vlm"]["production_image_h2d_bytes"] == 0
                and runtime["vlm"]["production_image_d2h_bytes"] == 0
                and runtime["hybrid"]["image_h2d_bytes"] == 0
                and runtime["hybrid"]["image_d2h_bytes"] == 0
            ),
            "metadata_copy_exact_per_hybrid_request": (
                runtime["hybrid"]["composed"] == len(records) + 2
                and runtime["hybrid"]["control_metadata_d2h_bytes"]
                == (len(records) + 2) * HYBRID_HINT_BUFFER_BYTES
            ),
        }
        report = {
            "schema_version": 1,
            "component": "hybrid-clean-fixed-image-semantic-ab",
            "status": "passed" if all(checks.values()) else "failed",
            "gpu": {"name": torch.cuda.get_device_name(0), "architecture": architecture},
            "cases": records,
            "adversarial": adversarial,
            "scores": {
                "clean_primary_hits": clean_hits,
                "hybrid_primary_hits": hybrid_hits,
                "case_count": len(records),
            },
            "validation_only_host_transfers": {
                "h2d_bytes": validation_h2d_bytes,
                "reason": "fixed-image and synthetic-detection semantic validation only",
                "excluded_from_production_hot_path": True,
            },
            "production_runtime": runtime,
            "checks": checks,
        }
    finally:
        vlm.stop()
        composer.close()

    output = Path(args.output)
    output = output.resolve() if output.is_absolute() else (WORKSPACE / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
