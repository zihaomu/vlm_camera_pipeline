from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.camera_io import CapturedFrame, LatestFrameSlot


def make_frame(sequence: int) -> CapturedFrame:
    image = np.full((4, 6, 3), sequence, dtype=np.uint8)
    image.setflags(write=False)
    return CapturedFrame(sequence, time.monotonic_ns(), image)


def test_latest_slot_is_depth_one_and_non_destructive() -> None:
    slot = LatestFrameSlot()
    slot.publish(make_frame(0))
    slot.publish(make_frame(1))
    slot.publish(make_frame(2))

    inference_view = slot.consume_after(-1, timeout=0)
    ui_view = slot.consume_after(1, timeout=0)

    assert inference_view is not None and inference_view.sequence == 2
    assert ui_view is inference_view
    assert slot.publish_count == 3
    assert slot.consume_after(2, timeout=0.001) is None


def test_consumer_waits_for_a_newer_sequence() -> None:
    slot = LatestFrameSlot()
    received: list[CapturedFrame | None] = []
    consumer = threading.Thread(target=lambda: received.append(slot.consume_after(4, timeout=1.0)))
    consumer.start()
    time.sleep(0.02)
    slot.publish(make_frame(5))
    consumer.join(timeout=1.0)

    assert not consumer.is_alive()
    assert received and received[0] is not None
    assert received[0].sequence == 5


def test_slot_rejects_non_monotonic_sequences_and_wakes_on_close() -> None:
    slot = LatestFrameSlot()
    slot.publish(make_frame(3))
    with pytest.raises(ValueError, match="increase monotonically"):
        slot.publish(make_frame(3))

    slot.close()
    assert slot.consume_after(3, timeout=None) is None
    with pytest.raises(RuntimeError, match="closed"):
        slot.publish(make_frame(4))
