from __future__ import annotations

import json
import time

import numpy as np

from src.camera_io import CapturedFrame
from src.realtime_pipeline import Detection, DetectorOutput, MetricsCollector


def test_metrics_json_contains_required_latency_and_drop_fields(tmp_path) -> None:
    started_ns = time.monotonic_ns()
    metrics = MetricsCollector("test", started_ns=started_ns)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image.setflags(write=False)

    for sequence, latency_ms in enumerate((10.0, 20.0, 30.0)):
        captured_ns = started_ns + sequence * 1_000_000
        frame = CapturedFrame(sequence, captured_ns, image)
        output = DetectorOutput(
            detections=(Detection(0, 0, 2, 2, 0.9, 0, "object"),),
            preprocess_ms=1.0,
            inference_ms=latency_ms - 3.0,
            postprocess_ms=2.0,
        )
        completed_ns = captured_ns + int(latency_ms * 1e6)
        metrics.record_inference(frame, output, completed_ns, skipped=sequence)
        metrics.record_display(frame, completed_ns + 1_000_000, 1.0)

    result = metrics.finish(
        camera_frames=9,
        camera_read_failures=0,
        camera_format={"width": 4, "height": 4, "fps": 30, "fourcc": "TEST"},
        exit_reason="test",
    )
    output_path = tmp_path / "metrics.json"
    metrics.write_json(output_path, result)
    loaded = json.loads(output_path.read_text())

    assert loaded["processed_frames"] == 3
    assert loaded["displayed_frames"] == 3
    assert loaded["dropped_frames"] == 6
    assert loaded["skipped_before_inference"] == 3
    assert loaded["latency_ms"]["capture_to_infer"]["p50"] == 20.0
    assert loaded["latency_ms"]["inference"]["count"] == 3
    assert loaded["record"] == {"mode": "off", "queue_depth": 0}
    assert loaded["vlm"]["failures"] == 0
