from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.camera_io import CapturedFrame, LatestFrameSlot
from src.vlm import LatestCaptionSlot, LlamaCppConfig, LlamaCppVlm, VlmCaption, VlmWorker


def make_frame(sequence: int) -> CapturedFrame:
    image = np.full((4, 6, 3), sequence, dtype=np.uint8)
    image.setflags(write=False)
    return CapturedFrame(sequence, time.monotonic_ns(), image)


class RecordingMetrics:
    def __init__(self) -> None:
        self.started = 0
        self.successes: list[int] = []
        self.failures = 0
        self.cancelled = 0
        self.changed = threading.Event()
        self._lock = threading.Lock()

    def record_vlm_started(self) -> None:
        with self._lock:
            self.started += 1

    def record_vlm_success(
        self,
        frame: CapturedFrame,
        requested_ns: int,
        completed_ns: int,
        text: str,
    ) -> None:
        assert completed_ns >= requested_ns
        assert text
        with self._lock:
            self.successes.append(frame.sequence)
            self.changed.set()

    def record_vlm_failure(
        self,
        requested_ns: int,
        completed_ns: int,
        error: BaseException,
    ) -> None:
        assert completed_ns >= requested_ns
        assert str(error)
        with self._lock:
            self.failures += 1
            self.changed.set()

    def record_vlm_cancelled(self, requested_ns: int, completed_ns: int) -> None:
        assert completed_ns >= requested_ns
        with self._lock:
            self.cancelled += 1
            self.changed.set()


class BlockingCaptioner:
    mode = "llamacpp"

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_finished = threading.Event()

    def caption_bgr(self, frame_bgr: np.ndarray) -> str:
        value = int(frame_bgr[0, 0, 0])
        self.calls.append(value)
        if len(self.calls) == 1:
            self.first_started.set()
            assert self.release_first.wait(1.0)
        if len(self.calls) == 2:
            self.second_finished.set()
        return f"frame {value}"


def test_vlm_worker_consumes_latest_snapshot_without_backlog() -> None:
    frame_slot = LatestFrameSlot()
    caption_slot = LatestCaptionSlot()
    metrics = RecordingMetrics()
    captioner = BlockingCaptioner()
    frame_slot.publish(make_frame(0))
    worker = VlmWorker(
        frame_slot,
        caption_slot,
        captioner,  # type: ignore[arg-type]
        metrics,
        interval_seconds=0.01,
    )
    worker.start()
    try:
        assert captioner.first_started.wait(1.0)
        for sequence in range(1, 10):
            frame_slot.publish(make_frame(sequence))
        captioner.release_first.set()
        assert captioner.second_finished.wait(1.0)
    finally:
        worker.request_stop()
        worker.join()
        worker.raise_if_failed()
        frame_slot.close()

    assert captioner.calls[:2] == [0, 9]
    assert metrics.successes[:2] == [0, 9]
    assert caption_slot.latest() is not None
    assert caption_slot.latest().sequence == 9  # type: ignore[union-attr]


class FailOnceCaptioner:
    mode = "llamacpp"

    def __init__(self) -> None:
        self.calls = 0

    def caption_bgr(self, frame_bgr: np.ndarray) -> str:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("synthetic timeout")
        return "recovered"


def test_vlm_request_failure_is_isolated_and_worker_recovers() -> None:
    frame_slot = LatestFrameSlot()
    caption_slot = LatestCaptionSlot()
    metrics = RecordingMetrics()
    captioner = FailOnceCaptioner()
    frame_slot.publish(make_frame(0))
    worker = VlmWorker(
        frame_slot,
        caption_slot,
        captioner,  # type: ignore[arg-type]
        metrics,
        interval_seconds=0.01,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 1.0
        while metrics.failures < 1 and time.monotonic() < deadline:
            metrics.changed.wait(0.05)
            metrics.changed.clear()
        frame_slot.publish(make_frame(1))
        while not metrics.successes and time.monotonic() < deadline:
            metrics.changed.wait(0.05)
            metrics.changed.clear()
    finally:
        worker.request_stop()
        worker.join()
        worker.raise_if_failed()
        frame_slot.close()

    assert metrics.failures == 1
    assert metrics.successes == [1]
    assert caption_slot.latest() is not None
    assert caption_slot.latest().text == "recovered"  # type: ignore[union-attr]


def test_caption_expiry_and_embedded_server_config_guards() -> None:
    now_ns = time.monotonic_ns()
    slot = LatestCaptionSlot()
    slot.publish(VlmCaption(3, now_ns - 2_000_000_000, now_ns, now_ns, "scene"))

    assert slot.latest_fresh(now_ns + 999_000_000, 1.0) is not None
    assert slot.latest_fresh(now_ns + 1_001_000_000, 1.0) is None
    with pytest.raises(ValueError, match="127.0.0.1"):
        LlamaCppConfig(host="0.0.0.0")
    with pytest.raises(ValueError, match="at least 1024"):
        LlamaCppConfig(context_size=512)


def test_llama_command_disables_prompt_cache_host_checkpoints(tmp_path: Path) -> None:
    server = LlamaCppVlm(LlamaCppConfig(), workspace=tmp_path)
    command = server._build_command()
    cache_index = command.index("--cache-ram")
    assert command[cache_index + 1] == "0"
