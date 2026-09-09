from __future__ import annotations

import inspect
import threading

import pytest

from scripts import run_yolo_vlm_zerocopy as runtime
from src.zerocopy_vlm import ROCWMMA_CMAKE_ENTRY, ROCWMMA_KERNEL_MARKER


class _FakeLease:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _BlockingVlm:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.finish = threading.Event()

    def caption_prepared(self, lease: _FakeLease) -> str:
        self.started.set()
        assert self.finish.wait(timeout=2)
        lease.release()
        return "GPU caption"


class _FakePreparedYolo(_FakeLease):
    def __init__(self, frame_id: int) -> None:
        super().__init__()
        self.frame_id = frame_id
        self.preprocess_ms = 0.25


class _BlockingYolo:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.finish = threading.Event()
        self.detections: list[_FakeLease] = []

    def infer_prepared(self, prepared: _FakePreparedYolo) -> _FakeLease:
        self.started.set()
        assert self.finish.wait(timeout=2)
        prepared.release()
        detection = _FakeLease()
        self.detections.append(detection)
        return detection


def test_strict_entrypoint_locks_camera_and_vlm_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = runtime.build_parser()
    args = parser.parse_args([])
    assert args.camera == "/dev/video0"
    assert (args.width, args.height, args.fps) == (1280, 720, 30)
    assert args.vlm_interval == 3.0
    assert args.zero_copy == "require"
    assert args.show_performance is False

    speed_args = parser.parse_args(["--show-speed"])
    assert speed_args.show_performance is True

    expected = {"active_driver_patched": True}
    monkeypatch.setattr(runtime, "camera_component_preflight", lambda **_: expected)
    assert runtime.strict_preflight(args) is expected

    args.vlm_interval = 3.1
    with pytest.raises(ValueError, match="exactly 3.0"):
        runtime.strict_preflight(args)


def test_gpu_presenter_has_coco_labels_vivid_palette_and_ascii_demo_title() -> None:
    presenter_source = (
        runtime.WORKSPACE / "native/egl_present/egl_present.hip"
    ).read_text(encoding="utf-8")
    runtime_source = (
        runtime.WORKSPACE / "scripts/run_yolo_vlm_zerocopy.py"
    ).read_text(encoding="utf-8")
    default_title = inspect.signature(runtime.EglHipPresenter).parameters["title"].default

    assert "kDetectionClassCount = 80" in presenter_source
    assert '"person", "bicycle", "car"' in presenter_source
    assert "class_label_alpha" in presenter_source
    assert "detection_color(class_id)" in presenter_source
    assert "make_uchar4(255, 56, 56, 255)" in presenter_source
    assert default_title == "YOLO26 + VLM - GPU Camera Demo"
    assert default_title.isascii()
    assert 'title="YOLO26 + VLM - GPU Camera Demo"' in runtime_source
    assert "performance_hud_kernel" in presenter_source
    assert "key == XK_h || key == XK_H" in presenter_source


def test_performance_hud_formats_rolling_yolo_and_vlm_speed() -> None:
    samples = runtime.deque(
        [(10.0, 25.0), (10.5, 27.0), (11.0, 26.0), (11.5, 28.0)]
    )
    fps, inference_ms = runtime.rolling_yolo_performance(samples, now=11.5)
    assert fps == pytest.approx(2.0)
    assert inference_ms == pytest.approx(26.5)
    assert runtime.format_performance_hud(
        yolo_fps=29.62,
        yolo_inference_ms=26.74,
        vlm_latency_ms=2034.0,
        vlm_tokens_per_second=15.74,
        vlm_running=False,
        vlm_interval_seconds=3.0,
    ) == "YOLO 29.6 FPS | 26.7 ms infer\nVLM 2.03 s/req | 15.7 tok/s | IDLE"


def test_vlm_worker_allows_one_outstanding_request_and_no_backlog() -> None:
    engine = _BlockingVlm()
    worker = runtime.LatestOnlyVlmWorker(engine)  # type: ignore[arg-type]
    first = _FakeLease()
    second = _FakeLease()
    assert worker.submit(
        source_frame_id=7,
        source_captured_ns=100,
        scheduled_at=1.0,
        lease=first,  # type: ignore[arg-type]
    )
    assert engine.started.wait(timeout=1)
    assert not worker.submit(
        source_frame_id=8,
        source_captured_ns=200,
        scheduled_at=2.0,
        lease=second,  # type: ignore[arg-type]
    )
    assert second.released

    engine.finish.set()
    worker.close()
    result = worker.latest_after(0)
    assert result is not None
    assert result.caption == "GPU caption"
    assert result.source_frame_id == 7
    assert first.released
    assert worker.info() == {
        "submitted": 1,
        "completed": 1,
        "failed": 0,
        "dropped_busy": 1,
        "outstanding": False,
        "queue_depth": 0,
        "max_queue_depth": 1,
    }


def test_yolo_worker_keeps_one_newest_pending_frame_without_backlog() -> None:
    engine = _BlockingYolo()
    worker = runtime.LatestOnlyYoloWorker(engine)  # type: ignore[arg-type]
    first = _FakePreparedYolo(frame_id=11)
    second = _FakePreparedYolo(frame_id=12)
    third = _FakePreparedYolo(frame_id=13)
    assert worker.submit(first)  # type: ignore[arg-type]
    assert engine.started.wait(timeout=1)
    assert worker.submit(second)  # type: ignore[arg-type]
    assert worker.submit(third)  # type: ignore[arg-type]
    assert second.released

    engine.finish.set()
    worker.close()
    result = worker.take_latest()
    assert result is not None
    assert result.error is None
    assert result.source_frame_id == 13
    assert result.preprocess_ms == 0.25
    assert result.detection is engine.detections[-1]
    assert first.released
    assert third.released
    assert len(engine.detections) == 2
    assert engine.detections[0].released
    assert worker.info() == {
        "submitted": 3,
        "completed": 2,
        "failed": 0,
        "dropped_busy": 0,
        "dropped_superseded": 1,
        "running": False,
        "pending": False,
        "outstanding": False,
        "latest_ready": False,
        "queue_depth": 0,
        "max_queue_depth": 1,
    }


def test_strict_runtime_has_no_host_pixel_api_tokens() -> None:
    files = [
        runtime.WORKSPACE / "src/zerocopy_camera.py",
        runtime.WORKSPACE / "src/zerocopy_yolo.py",
        runtime.WORKSPACE / "src/zerocopy_vlm.py",
        runtime.WORKSPACE / "src/zerocopy_present.py",
        runtime.WORKSPACE / "scripts/run_yolo_vlm_zerocopy.py",
    ]
    forbidden = (
        ".cpu(",
        ".numpy(",
        ".download(",
        "cv2.VideoCapture",
        "cv2.imshow",
        "cv2.imencode",
        "Image.fromarray",
        "hipMemcpyDeviceToHost",
    )
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in forbidden), path


def test_llama_zerocopy_build_requires_rocwmma_flash_attention() -> None:
    setup = (
        runtime.WORKSPACE / "scripts/setup_llama_vlm_zerocopy_gfx1151.sh"
    ).read_text(encoding="utf-8")

    assert 'rocwmma_header="$hip_path/include/rocwmma/rocwmma.hpp"' in setup
    assert '-DGGML_HIP_ROCWMMA_FATTN=ON' in setup
    assert "GGML_HIP_ROCWMMA_FATTN:BOOL=ON" in setup
    assert ROCWMMA_CMAKE_ENTRY == "GGML_HIP_ROCWMMA_FATTN:BOOL=ON"
    assert ROCWMMA_KERNEL_MARKER == "ggml_cuda_flash_attn_ext_wmma_f16"


def test_persistent_module_scripts_preserve_stock_and_have_exact_rollback() -> None:
    install = (
        runtime.WORKSPACE / "scripts/install_amd_isp4_dmabuf_patch.sh"
    ).read_text(encoding="utf-8")
    uninstall = (
        runtime.WORKSPACE / "scripts/uninstall_amd_isp4_dmabuf_patch.sh"
    ).read_text(encoding="utf-8")
    expected_target = "updates/vlm-camera-pipeline"
    stock = "kernel/drivers/media/platform/amd/isp4/amd_capture.ko.zst"

    assert expected_target in install
    assert expected_target in uninstall
    assert stock in install
    assert stock in uninstall
    assert 'rm -f -- "$target_module"' in install
    assert 'rm -f -- "$target_module"' in uninstall
    assert 'rm -f -- "$stock_module"' not in install
    assert 'rm -f -- "$stock_module"' not in uninstall
    assert "rm -rf" not in install
    assert "rm -rf" not in uninstall
    assert "restore_stock" in install
    assert "restore_patch" in uninstall
