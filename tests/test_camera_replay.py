from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera_io import CameraConfig, CameraReader, CapturedFrame, LatestFrameSlot
from src.realtime_pipeline import (
    DetectorOutput,
    InferenceWorker,
    LatestResultSlot,
    MetricsCollector,
    PipelineConfig,
    RealtimePipeline,
)


class FakeCapture:
    """Small synthetic replay source implementing the OpenCV capture surface."""

    def __init__(self, width: int, height: int, fps: float, fourcc: str) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.opened = True
        self.released = False
        self.sequence = 0
        self._lock = threading.Lock()

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        elif property_id == cv2.CAP_PROP_FPS:
            self.fps = float(value)
        elif property_id == cv2.CAP_PROP_FOURCC:
            self.fourcc = int(value)
        return True

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FOURCC: self.fourcc,
        }
        return float(values.get(property_id, 0.0))

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.002)
        with self._lock:
            if self.released:
                return False, None
            value = self.sequence % 255
            self.sequence += 1
        return True, np.full((self.height, self.width, 3), value, dtype=np.uint8)

    def release(self) -> None:
        with self._lock:
            self.released = True


def test_camera_reader_negotiates_and_publishes_read_only_latest_frames() -> None:
    fake = FakeCapture(16, 12, 30.0, "NV12")
    reader = CameraReader(
        CameraConfig(width=16, height=12, fps=30.0, fourcc="NV12"),
        capture_factory=lambda _device, _backend: fake,
    )
    reader.start()
    try:
        cursor = -1
        deadline = time.monotonic() + 1.0
        frame = None
        while time.monotonic() < deadline:
            candidate = reader.slot.consume_after(cursor, timeout=0.1)
            if candidate is not None:
                frame = candidate
                cursor = candidate.sequence
                if cursor >= 3:
                    break
        assert frame is not None
        assert frame.bgr.shape == (12, 16, 3)
        assert not frame.bgr.flags.writeable
        assert reader.negotiated is not None
        assert reader.negotiated.fourcc == "NV12"
    finally:
        reader.stop()
    reader.raise_if_failed()


class SlowDetector:
    backend_name = "synthetic"

    def warmup(self, frame_bgr: np.ndarray, iterations: int = 2) -> None:
        return None

    def detect_bgr(self, frame_bgr: np.ndarray) -> DetectorOutput:
        time.sleep(0.02)
        return DetectorOutput(())


def test_slow_inference_skips_old_replay_frames_instead_of_queueing() -> None:
    frame_slot = LatestFrameSlot()
    result_slot = LatestResultSlot()
    metrics = MetricsCollector("synthetic")
    worker = InferenceWorker(frame_slot, result_slot, SlowDetector(), metrics)
    worker.start()
    try:
        for sequence in range(30):
            image = np.full((4, 4, 3), sequence, dtype=np.uint8)
            image.setflags(write=False)
            frame_slot.publish(CapturedFrame(sequence, time.monotonic_ns(), image))
            time.sleep(0.001)

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            latest = result_slot.latest()
            if latest is not None and latest.sequence == 29:
                break
            time.sleep(0.005)
        latest = result_slot.latest()
        assert latest is not None and latest.sequence == 29
        assert metrics.processed_frames < 30
        assert metrics.skipped_before_inference > 0
    finally:
        frame_slot.close()
        worker.stop()
    worker.raise_if_failed()


def test_realtime_pipeline_runs_synthetic_replay_and_writes_metrics(tmp_path) -> None:
    fake = FakeCapture(16, 12, 30.0, "NV12")
    reader = CameraReader(
        CameraConfig(width=16, height=12, fps=30.0, fourcc="NV12"),
        capture_factory=lambda _device, _backend: fake,
    )
    output = tmp_path / "replay-metrics.json"
    pipeline = RealtimePipeline(
        reader,
        SlowDetector(),
        PipelineConfig(
            display=False,
            duration_seconds=0.15,
            warmup_iterations=1,
            metrics_json=str(output),
        ),
    )

    metrics = pipeline.run()

    assert metrics["exit_reason"] == "duration_elapsed"
    assert metrics["camera_frames"] > metrics["processed_frames"] > 0
    assert metrics["dropped_frames"] > 0
    assert metrics["camera_read_failures"] == 0
    assert output.is_file()


def test_locked_sidewalk_video_feeds_the_latest_frame_scheduler() -> None:
    video_path = (
        Path(__file__).resolve().parents[1]
        / "third_party/notebook/ultralytics_yolo26/data/sidewalk.mp4"
    )
    assert video_path.is_file(), "run scripts/setup_native_gfx1151.sh first"
    with video_path.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == "0194fb9a4b6590b5ba12e22f32978426efcd89da60ecc285ea9e4f00eb5bc658"
        )

    slot = LatestFrameSlot()
    capture = cv2.VideoCapture(str(video_path))
    try:
        for sequence in range(32):
            ok, frame = capture.read()
            assert ok and frame is not None
            frame.setflags(write=False)
            slot.publish(CapturedFrame(sequence, time.monotonic_ns(), frame))
    finally:
        capture.release()

    latest = slot.consume_after(-1, timeout=0)
    assert latest is not None
    assert latest.sequence == 31
    assert latest.bgr.shape == (1080, 1920, 3)
