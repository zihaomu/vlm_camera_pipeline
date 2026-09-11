from __future__ import annotations

import struct
from pathlib import Path

import pytest

from src.zerocopy_web import (
    WEB_OVERLAY_ATLAS_BYTES,
    WEB_OVERLAY_CLASS_COUNT,
    WEB_OVERLAY_LABEL_HEIGHT,
    WEB_OVERLAY_LABEL_WIDTH,
    WEB_OVERLAY_RESOURCE_BYTES,
    PromptStore,
    WebDashboard,
    ZeroCopyWebError,
    _CaptureTimestampClock,
    _fragment_starts_with_sync_sample,
    _Mp4BoxParser,
    _Mp4FragmentAssembler,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _full_box(kind: bytes, flags: int, payload: bytes) -> bytes:
    return _box(kind, bytes([0]) + flags.to_bytes(3, "big") + payload)


def _media_segment(sample_flags: int) -> bytes:
    tfhd = _full_box(b"tfhd", 0, struct.pack(">I", 1))
    trun = _full_box(
        b"trun",
        0x000701,
        struct.pack(">IIIII", 1, 0, 100, 4096, sample_flags),
    )
    moof = _box(b"moof", _box(b"traf", tfhd + trun))
    return moof + _box(b"mdat", b"encoded frame")


def test_prompt_store_is_bounded_normalized_and_versioned() -> None:
    store = PromptStore("  Describe the scene.  ")
    first = store.snapshot()
    assert first.text == "Describe the scene."
    assert first.version == 1
    assert store.update("Describe the scene.").version == 1
    second = store.update("Count all visible people.")
    assert second.text == "Count all visible people."
    assert second.version == 2
    with pytest.raises(ValueError, match="must not be empty"):
        store.update("  \n ")
    with pytest.raises(ValueError, match="512-character"):
        store.update("x" * 513)


def test_fragmented_mp4_parser_preserves_complete_top_level_boxes() -> None:
    source = _box(b"ftyp", b"demo") + _box(b"moov", b"metadata") + _box(b"moof", b"x")
    parser = _Mp4BoxParser()
    output = parser.feed(source[:7]) + parser.feed(source[7:19]) + parser.feed(source[19:])
    assert [kind for kind, _ in output] == [b"ftyp", b"moov", b"moof"]
    assert b"".join(box for _, box in output) == source


def test_mp4_fragment_assembler_emits_one_atomic_mse_media_segment() -> None:
    assembler = _Mp4FragmentAssembler()
    moof = _box(b"moof", b"metadata")
    auxiliary = _box(b"free", b"padding")
    mdat = _box(b"mdat", b"encoded frames")
    assert assembler.feed(b"mdat", mdat) is None
    assert assembler.feed(b"moof", moof) is None
    assert assembler.feed(b"free", auxiliary) is None
    assert assembler.feed(b"mdat", mdat) == moof + auxiliary + mdat

    with pytest.raises(ZeroCopyWebError, match="new moof"):
        assembler.feed(b"moof", moof)
        assembler.feed(b"moof", moof)


def test_capture_timestamp_clock_tracks_real_cadence_and_clamps_discontinuity() -> None:
    clock = _CaptureTimestampClock(30)
    frame_duration = 1_000_000_000 // 30
    assert clock.next(5_000_000_000) == (0, frame_duration)
    assert clock.next(5_034_000_000) == (34_000_000, 34_000_000)
    assert clock.next(7_034_000_000) == (
        34_000_000 + frame_duration,
        frame_duration,
    )
    assert clock.discontinuities == 1


def test_fragment_sync_sample_detection_uses_trun_sample_flags() -> None:
    assert _fragment_starts_with_sync_sample(_media_segment(0x00000040)) is True
    assert _fragment_starts_with_sync_sample(_media_segment(0x000100C0)) is False


def test_dashboard_exposes_prompt_as_the_only_frontend_control() -> None:
    html = (WORKSPACE / "web/index.html").read_text(encoding="utf-8")
    javascript = (WORKSPACE / "web/app.js").read_text(encoding="utf-8")
    stylesheet = (WORKSPACE / "web/app.css").read_text(encoding="utf-8")
    favicon = (WORKSPACE / "web/amd-mark.svg").read_text(encoding="utf-8")
    assert html.count("<textarea") == 1
    assert 'id="prompt-input"' in html
    assert "<select" not in html
    assert 'type="range"' not in html
    assert 'type="checkbox"' not in html
    assert "REAL-TIME PERFORMANCE" in html
    assert "GPU DATA PATH" in html
    assert "GPU BOX BURN-IN" in html
    assert 'id="amd-mark"' in html
    assert 'rel="icon" href="/amd-mark.svg"' in html
    assert 'class="amd-lockup"' in html
    assert '<span class="brand-mark" aria-hidden="true">' in html
    assert html.count('class="amd-tech"') == 5
    assert html.count('href="#amd-mark"') == 8
    assert "--amd-mark: #f5f7fa" in stylesheet
    assert 'fill="#f5f7fa"' in favicon
    assert 'fill="#ed1c24"' not in favicon
    assert "detection-layer" not in html
    assert "drawDetections" not in javascript
    assert "sourceBuffer.remove(oldest, keepFrom)" in javascript
    assert "STARTUP_BUFFER_SECONDS = 0.36" in javascript
    assert "TARGET_LIVE_LATENCY_SECONDS = 0.16" in javascript
    assert "REBUFFER_RESUME_SECONDS = 0.22" in javascript
    assert "MAX_LIVE_LATENCY_SECONDS = 0.40" in javascript
    assert "LOW_BUFFER_SECONDS = 0.12" in javascript
    assert "RECOVERED_BUFFER_SECONDS = 0.17" in javascript
    assert "BUFFER_RECOVERY_PLAYBACK_RATE = 0.985" in javascript
    assert "REBUFFER_DEBOUNCE_MS = 180" in javascript
    assert "captureToMediaEdgeSeconds" in javascript
    assert "captureToDisplaySeconds" in javascript
    assert 'video.addEventListener("waiting"' in javascript
    assert 'video.addEventListener("seeked"' in javascript
    assert "__vlmPlaybackDiagnostics" in javascript
    assert 'preload="auto"' in html


def test_dashboard_state_carries_caption_and_read_only_performance() -> None:
    dashboard = WebDashboard(workspace=WORKSPACE)
    dashboard.set_caption(
        "A person is standing.",
        request_id=2,
        source_frame_id=7,
        prompt_version=1,
    )
    dashboard.set_performance(
        camera_fps=30.0,
        yolo_fps=29.5,
        yolo_inference_ms=24.0,
        vlm_latency_ms=1900.0,
        vlm_tokens_per_second=16.0,
        vlm_running=False,
    )
    state = dashboard.state_snapshot()
    assert "detections" not in state
    assert state["caption"]["prompt_version"] == 1
    assert state["performance"]["yolo_fps"] == 29.5
    assert state["contract"]["image_host_copies"] == 0
    assert "YUYV" in state["contract"]["overlay"]


def test_web_video_path_is_bounded_and_low_latency() -> None:
    runtime = (WORKSPACE / "src/zerocopy_web.py").read_text(encoding="utf-8")
    assert "max-buffers=2" in runtime
    assert "key-int-max=6" in runtime
    assert "rate-control=cbr" in runtime
    assert "fragment-duration=50" in runtime
    assert "processing-deadline=0" in runtime
    assert "_on_encoder_input_buffer" in runtime
    assert "self._encoder_input_pts.popleft()" in runtime
    assert "latest_fragment_capture_monotonic_seconds" in runtime
    assert 'app.router.add_get("/amd-mark.svg"' in runtime


def test_web_overlay_is_gpu_yuyv_burn_in_with_fixed_label_abi() -> None:
    native = (WORKSPACE / "native/zerocopy_kernels/zerocopy_kernels.hip").read_text(
        encoding="utf-8"
    )
    camera_header = (
        WORKSPACE / "native/camera_dmabuf/camera_gpu_capture.h"
    ).read_text(encoding="utf-8")
    runtime = (WORKSPACE / "scripts/run_yolo_vlm_zerocopy.py").read_text(
        encoding="utf-8"
    )
    camera_native = (
        WORKSPACE / "native/camera_dmabuf/camera_gpu_capture.hip"
    ).read_text(encoding="utf-8")
    web_launcher = (
        WORKSPACE / "scripts/run_yolo_vlm_web_zerocopy.sh"
    ).read_text(encoding="utf-8")
    assert WEB_OVERLAY_CLASS_COUNT == 80
    assert WEB_OVERLAY_LABEL_WIDTH == 224
    assert WEB_OVERLAY_LABEL_HEIGHT == 36
    assert WEB_OVERLAY_ATLAS_BYTES == 645120
    assert WEB_OVERLAY_RESOURCE_BYTES == 645440
    assert "web_yuyv_detection_overlay_kernel" in native
    assert "vlm_camera_web_overlay_yuyv_f32" in native
    assert "web_write_yuyv_pair" in native
    assert "web_stream_device_pointer" in camera_header
    assert "web_overlay.draw(" in runtime
    assert '"web_detection_control_metadata_d2h_bytes": 0' in runtime
    assert "width - 1 - x" in camera_native
    assert "width - 2 - x" in camera_native
    assert "--camera-horizontal-flip" in web_launcher
