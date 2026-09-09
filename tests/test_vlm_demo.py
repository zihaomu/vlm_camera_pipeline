from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.run_vlm_demo import build_parser
from src.camera_io import CameraConfig, CameraReader, VideoFileConfig, VideoFileReader
from src.realtime_pipeline import (
    PipelineConfig,
    RealtimePipeline,
    _resize_display_window,
    draw_vlm_subtitle_frame,
)
from src.vlm import VlmCaption


class FakeCapture:
    def __init__(self) -> None:
        self.width = 320
        self.height = 180
        self.fps = 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*"NV12")
        self.released = False
        self.sequence = 0
        self._lock = threading.Lock()

    def isOpened(self) -> bool:
        return True

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


class FakeVlmEngine:
    mode = "llamacpp"

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> FakeVlmEngine:
        self.started = True
        return self

    def caption_bgr(self, frame_bgr: np.ndarray) -> str:
        assert frame_bgr.shape == (180, 320, 3)
        return "一个实时更新的测试画面。"

    def stop(self) -> None:
        self.stopped = True

    def runtime_info(self) -> dict[str, object]:
        return {"device": "ROCm0", "full_gpu_offload": True}


def test_vlm_demo_parser_defaults_to_smooth_three_second_refresh() -> None:
    args = build_parser().parse_args([])

    assert args.vlm_interval == 3.0
    assert args.display is True
    assert args.window_scale == 0.75
    assert "中文" in args.vlm_prompt


def test_window_scale_resizes_the_combined_video_and_subtitle_canvas() -> None:
    class FakeCv2:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int]] = []

        def resizeWindow(self, name: str, width: int, height: int) -> None:
            self.calls.append((name, width, height))

    cv2_api = FakeCv2()
    _resize_display_window(cv2_api, "demo", (900, 1280, 3), 0.75)

    assert cv2_api.calls == [("demo", 960, 675)]


def test_subtitle_panel_is_below_video_and_supports_chinese() -> None:
    frame = np.full((180, 320, 3), 90, dtype=np.uint8)
    original = frame.copy()
    now_ns = time.monotonic_ns()
    caption = VlmCaption(1, now_ns, now_ns, now_ns, "一个人正在镜头前挥手。")

    rendered = draw_vlm_subtitle_frame(
        frame,
        {
            "display_fps": 29.9,
            "vlm_requests": 2,
            "vlm_successes": 1,
            "vlm_failures": 0,
            "vlm_in_flight": 1,
            "vlm_caption_age_seconds": 0.2,
        },
        caption=caption,
        panel_height=180,
    )

    assert rendered.shape == (360, 320, 3)
    assert np.array_equal(rendered[:180], original)
    assert np.array_equal(frame, original)
    assert np.unique(rendered[180:].reshape(-1, 3), axis=0).shape[0] > 5


def test_realtime_pipeline_can_run_vlm_without_loading_a_detector(tmp_path) -> None:
    capture = FakeCapture()
    camera = CameraReader(
        CameraConfig(width=320, height=180, fps=30.0, fourcc="NV12"),
        capture_factory=lambda _device, _backend: capture,
    )
    engine = FakeVlmEngine()
    metrics_path = tmp_path / "vlm-only.json"
    pipeline = RealtimePipeline(
        camera,
        None,
        PipelineConfig(
            display=False,
            display_layout="subtitle",
            duration_seconds=0.12,
            metrics_json=str(metrics_path),
            vlm_interval_seconds=0.02,
        ),
        vlm_engine=engine,
    )

    metrics = pipeline.run()

    assert engine.started and engine.stopped
    assert metrics["backend"] == "off"
    assert metrics["detector"] == {"backend": "off", "loaded": False}
    assert metrics["pipeline"]["mode"] == "vlm-only"
    assert metrics["processed_frames"] == 0
    assert metrics["dropped_frames"] == 0
    assert metrics["vlm"]["successes"] >= 1
    assert metrics_path.is_file()


def test_video_file_reader_replays_at_configured_rate() -> None:
    video_path = (
        Path(__file__).resolve().parents[1]
        / "third_party/notebook/ultralytics_yolo26/data/sidewalk.mp4"
    )
    reader = VideoFileReader(
        VideoFileConfig(
            path=str(video_path),
            width=320,
            height=180,
            fps=50.0,
            loop=False,
        )
    )
    reader.start()
    try:
        frame = reader.slot.consume_after(-1, timeout=1.0)
        assert frame is not None
        next_frame = reader.slot.consume_after(frame.sequence, timeout=1.0)
        assert next_frame is not None
        assert next_frame.bgr.shape == (180, 320, 3)
        assert not next_frame.bgr.flags.writeable
        assert reader.negotiated is not None
        assert reader.negotiated.fps == 50.0
    finally:
        reader.stop()
    reader.raise_if_failed()
