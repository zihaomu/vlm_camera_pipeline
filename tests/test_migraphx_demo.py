from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.run_yolo_vlm_demo import build_parser, validate_args
from src.migraphx_detector import (
    MIGraphXBackendError,
    _literal_metadata,
    _write_cache_identity,
)
from src.realtime_pipeline import Detection, draw_vlm_subtitle_frame


def test_combined_demo_has_a_separate_migraphx_entrypoint() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.yolo_model == "models/yolo26x.onnx"
    assert args.migraphx_cache_dir.endswith("gfx1151-yolo26x")
    assert args.vlm_interval == 3.0
    assert args.window_scale == 0.75
    validate_args(parser, args)


def test_combined_demo_rejects_a_shape_that_differs_from_locked_onnx() -> None:
    parser = build_parser()
    args = parser.parse_args(["--imgsz", "320"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)


def test_cache_identity_is_reused_only_when_all_fields_match(tmp_path) -> None:
    cache = tmp_path / "cache"
    identity = {"gpu_arch": "gfx1151", "model_sha256": "abc", "onnxruntime": "1.23.2"}

    _write_cache_identity(cache, identity)
    _write_cache_identity(cache, identity)

    assert json.loads((cache / "identity.json").read_text()) == identity
    with pytest.raises(MIGraphXBackendError, match="does not match"):
        _write_cache_identity(cache, {**identity, "gpu_arch": "gfx1100"})


def test_unidentified_compiled_cache_is_never_blessed(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "unknown.mxr").write_bytes(b"different target")

    with pytest.raises(MIGraphXBackendError, match="unidentified"):
        _write_cache_identity(cache, {"gpu_arch": "gfx1151"})


def test_metadata_parser_does_not_execute_arbitrary_text() -> None:
    assert _literal_metadata("{'nms': False}", field="args") == {"nms": False}
    with pytest.raises(MIGraphXBackendError, match="invalid ONNX args metadata"):
        _literal_metadata("__import__('os').getcwd()", field="args")


def test_subtitle_layout_draws_yolo_boxes_without_mutating_camera_frame() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    original = frame.copy()
    detection = Detection(20, 20, 140, 120, 0.95, 0, "person")
    rendered = draw_vlm_subtitle_frame(
        frame,
        {
            "backend": "onnxruntime-migraphx",
            "inference_fps": 24.0,
            "display_fps": 25.0,
            "detection_count": 1,
            "vlm_requests": 1,
            "vlm_successes": 0,
            "vlm_failures": 0,
            "vlm_in_flight": 1,
        },
        caption=None,
        detections=(detection,),
        panel_height=180,
    )

    assert rendered.shape == (360, 320, 3)
    assert np.array_equal(frame, original)
    assert np.any(rendered[18:23, 18:143] != 0)
    assert np.unique(rendered[180:].reshape(-1, 3), axis=0).shape[0] > 5
