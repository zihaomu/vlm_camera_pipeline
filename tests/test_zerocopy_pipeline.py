from __future__ import annotations

import inspect
import threading

import pytest

from scripts import run_yolo_vlm_zerocopy as runtime
from src.zerocopy_vlm import ROCWMMA_CMAKE_ENTRY, ROCWMMA_KERNEL_MARKER


class _FakeLease:
    def __init__(self) -> None:
        self.released = False
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
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


class _FailingVlm:
    def caption_prepared(self, lease: _FakeLease) -> str:
        del lease
        raise TimeoutError("synthetic VLM timeout")


class _HybridVlm:
    def __init__(self) -> None:
        self.prompt: str | None = None
        self.stop_at_first_sentence = False

    def caption_prepared(
        self,
        lease: _FakeLease,
        *,
        prompt: str,
        stop_at_first_sentence: bool,
        constrain_english_sentence: bool,
    ) -> str:
        self.prompt = prompt
        self.stop_at_first_sentence = stop_at_first_sentence
        self.constrain_english_sentence = constrain_english_sentence
        lease.release()
        return "Hybrid caption."


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


class _FailingYolo:
    def infer_prepared(self, prepared: _FakePreparedYolo) -> _FakeLease:
        del prepared
        raise RuntimeError("synthetic YOLO failure")


def test_strict_entrypoint_locks_camera_and_vlm_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = runtime.build_parser()
    args = parser.parse_args([])
    assert args.camera == "/dev/video0"
    assert (args.width, args.height, args.fps) == (1280, 720, 30)
    assert args.vlm_interval == 3.0
    assert args.vlm_log == "output/realtime/llama-server-zerocopy.log"
    assert args.vlm_input_mode == "clean"
    assert args.hybrid_max_hints == 8
    assert args.hybrid_arm_timeout_ms == 100.0
    assert args.presenter == "egl"
    assert args.zero_copy == "require"
    assert args.show_performance is False
    assert args.camera_horizontal_flip is False

    speed_args = parser.parse_args(["--show-speed"])
    assert speed_args.show_performance is True
    assert parser.parse_args(["--camera-horizontal-flip"]).camera_horizontal_flip is True
    assert (
        parser.parse_args(["--no-camera-horizontal-flip"]).camera_horizontal_flip
        is False
    )
    assert (
        parser.parse_args(
            ["--camera-horizontal-flip", "--no-camera-horizontal-flip"]
        ).camera_horizontal_flip
        is False
    )

    expected = {"active_driver_patched": True}
    monkeypatch.setattr(runtime, "camera_component_preflight", lambda **_: expected)
    assert runtime.strict_preflight(args) is expected

    args.vlm_interval = 3.1
    with pytest.raises(ValueError, match="exactly 3.0"):
        runtime.strict_preflight(args)

    hybrid_args = parser.parse_args(["--vlm-input-mode", "hybrid"])
    monkeypatch.setattr(runtime, "camera_component_preflight", lambda **_: expected)
    assert runtime.strict_preflight(hybrid_args) is expected
    hybrid_args.hybrid_arm_timeout_ms = 101.0
    with pytest.raises(ValueError, match="exactly 100 ms"):
        runtime.strict_preflight(hybrid_args)

    web_args = parser.parse_args(["--presenter", "web", "--vlm-input-mode", "hybrid"])
    monkeypatch.setattr(runtime, "camera_component_preflight", lambda **_: expected)
    assert runtime.strict_preflight(web_args) is expected
    web_args.vlm_input_mode = "clean"
    with pytest.raises(ValueError, match="requires Hybrid"):
        runtime.strict_preflight(web_args)


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
    ) == 1
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
    assert first.release_calls == 1
    assert worker.info() == {
        "submitted": 1,
        "completed": 1,
        "failed": 0,
        "dropped_busy": 1,
        "outstanding": False,
        "queue_depth": 0,
        "max_queue_depth": 1,
    }


def test_vlm_worker_hybrid_prompt_path_and_failure_release_once() -> None:
    hybrid_engine = _HybridVlm()
    hybrid_worker = runtime.LatestOnlyVlmWorker(hybrid_engine)  # type: ignore[arg-type]
    hybrid_lease = _FakeLease()
    assert hybrid_worker.submit(
        source_frame_id=9,
        source_captured_ns=300,
        scheduled_at=3.0,
        lease=hybrid_lease,  # type: ignore[arg-type]
        prompt="bounded Hybrid prompt",
        hybrid=object(),  # type: ignore[arg-type]
    ) == 1
    hybrid_worker.close()
    hybrid_result = hybrid_worker.latest_after(0)
    assert hybrid_result is not None and hybrid_result.error is None
    assert hybrid_engine.prompt == "bounded Hybrid prompt"
    assert hybrid_engine.stop_at_first_sentence is True
    assert hybrid_engine.constrain_english_sentence is True
    assert hybrid_lease.release_calls == 1

    web_engine = _HybridVlm()
    web_worker = runtime.LatestOnlyVlmWorker(web_engine)  # type: ignore[arg-type]
    web_lease = _FakeLease()
    assert web_worker.submit(
        source_frame_id=11,
        source_captured_ns=500,
        scheduled_at=5.0,
        lease=web_lease,  # type: ignore[arg-type]
        prompt="用中文描述画面",
        hybrid=object(),  # type: ignore[arg-type]
        constrain_english_sentence=False,
    ) == 1
    web_worker.close()
    assert web_engine.prompt == "用中文描述画面"
    assert web_engine.stop_at_first_sentence is True
    assert web_engine.constrain_english_sentence is False

    failing_worker = runtime.LatestOnlyVlmWorker(_FailingVlm())  # type: ignore[arg-type]
    failing_lease = _FakeLease()
    assert failing_worker.submit(
        source_frame_id=10,
        source_captured_ns=400,
        scheduled_at=4.0,
        lease=failing_lease,  # type: ignore[arg-type]
    ) == 1
    failing_worker.close()
    failed_result = failing_worker.latest_after(0)
    assert failed_result is not None
    assert failed_result.caption is None
    assert failed_result.error == "TimeoutError: synthetic VLM timeout"
    assert failing_lease.release_calls == 1
    assert failing_worker.info()["failed"] == 1


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
        "dropped_while_protected": 0,
        "protected_submitted": 0,
        "protected_completed": 0,
        "running": False,
        "pending": False,
        "outstanding": False,
        "latest_ready": False,
        "protected_latest_ready": False,
        "queue_depth": 0,
        "max_queue_depth": 1,
    }


def test_yolo_worker_protects_hybrid_pending_task_from_newer_frames() -> None:
    engine = _BlockingYolo()
    worker = runtime.LatestOnlyYoloWorker(engine)  # type: ignore[arg-type]
    running = _FakePreparedYolo(frame_id=20)
    protected = _FakePreparedYolo(frame_id=21)
    rejected = _FakePreparedYolo(frame_id=22)

    assert worker.submit(running)  # type: ignore[arg-type]
    assert engine.started.wait(timeout=1)
    assert worker.submit(protected, protected_for_vlm=True)  # type: ignore[arg-type]
    assert not worker.submit(rejected)  # type: ignore[arg-type]
    assert rejected.released

    engine.finish.set()
    worker.close()
    normal_result = worker.take_latest()
    protected_result = worker.take_protected()
    assert normal_result is not None and normal_result.source_frame_id == 20
    assert protected_result is not None and protected_result.source_frame_id == 21
    assert protected_result.protected_for_vlm is True
    normal_result.detection.release()  # type: ignore[union-attr]
    protected_result.detection.release()  # type: ignore[union-attr]
    info = worker.info()
    assert info["submitted"] == 2
    assert info["completed"] == 2
    assert info["dropped_busy"] == 1
    assert info["dropped_while_protected"] == 1
    assert info["protected_submitted"] == 1
    assert info["protected_completed"] == 1


def test_yolo_worker_surfaces_protected_failure_and_releases_input_once() -> None:
    worker = runtime.LatestOnlyYoloWorker(_FailingYolo())  # type: ignore[arg-type]
    prepared = _FakePreparedYolo(frame_id=31)
    assert worker.submit(prepared, protected_for_vlm=True)  # type: ignore[arg-type]
    worker.close()
    result = worker.take_protected()
    assert result is not None
    assert result.source_frame_id == 31
    assert result.detection is None
    assert result.error == "RuntimeError: synthetic YOLO failure"
    assert result.protected_for_vlm is True
    assert prepared.release_calls == 1
    info = worker.info()
    assert info["failed"] == 1
    assert info["protected_submitted"] == info["protected_completed"] == 1


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


def test_hybrid_launcher_is_explicit_and_keeps_clean_default_separate() -> None:
    launcher = (runtime.WORKSPACE / "scripts/run_yolo_vlm_hybrid_zerocopy.sh").read_text(
        encoding="utf-8"
    )
    clean_launcher = (runtime.WORKSPACE / "scripts/run_yolo_vlm_zerocopy.sh").read_text(
        encoding="utf-8"
    )
    assert "--vlm-input-mode hybrid" in launcher
    assert "--hybrid-max-hints 8" in launcher
    assert "--hybrid-arm-timeout-ms 100" in launcher
    assert "--vlm-input-mode hybrid" not in clean_launcher


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
