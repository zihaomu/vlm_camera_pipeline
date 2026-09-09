"""Camera capture primitives with depth-one, latest-frame semantics."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import numpy as np


class CameraError(RuntimeError):
    """Base error raised by the camera capture layer."""


class CameraOpenError(CameraError):
    """The requested camera could not be opened or negotiated."""


class CameraReadError(CameraError):
    """The camera exceeded the configured consecutive-read failure limit."""


class CameraShutdownError(CameraError):
    """The capture thread did not stop within the configured timeout."""


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A BGR frame timestamped immediately after a successful host read."""

    sequence: int
    captured_ns: int
    bgr: np.ndarray

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.captured_ns <= 0:
            raise ValueError("captured_ns must be a positive monotonic timestamp")
        if not isinstance(self.bgr, np.ndarray):
            raise TypeError("bgr must be a numpy.ndarray")
        if self.bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 BGR frame, got {self.bgr.dtype}")
        if self.bgr.ndim != 3 or self.bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR frame, got {self.bgr.shape}")
        if not self.bgr.flags.c_contiguous:
            raise ValueError("BGR frame must be C-contiguous")


class LatestFrameSlot:
    """A non-destructive, condition-backed depth-one frame slot.

    Consumers track their own last sequence. Publishing a newer frame replaces
    the old reference, so a slow consumer never creates an unbounded backlog.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._closed = False
        self._publish_count = 0

    def publish(self, frame: CapturedFrame) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot publish to a closed LatestFrameSlot")
            if self._latest is not None and frame.sequence <= self._latest.sequence:
                raise ValueError(
                    "frame sequences must increase monotonically: "
                    f"latest={self._latest.sequence}, new={frame.sequence}"
                )
            self._latest = frame
            self._publish_count += 1
            self._condition.notify_all()

    def consume_after(self, sequence: int, timeout: float | None = None) -> CapturedFrame | None:
        """Return the latest frame newer than *sequence*, or ``None`` on timeout.

        Reading is non-destructive: independent UI and inference consumers can
        observe the same published frame using their own sequence cursors.
        """

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._latest is None or self._latest.sequence <= sequence:
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._latest

    def latest(self) -> CapturedFrame | None:
        with self._condition:
            return self._latest

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def publish_count(self) -> int:
        with self._condition:
            return self._publish_count


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "NV12"
    buffer_size: int = 1
    max_consecutive_failures: int = 30
    retry_delay_seconds: float = 0.01
    strict_format: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fps <= 0:
            raise ValueError("camera FPS must be positive")
        if len(self.fourcc) != 4:
            raise ValueError("camera FOURCC must contain exactly four characters")
        if self.buffer_size < 1:
            raise ValueError("camera buffer_size must be at least one")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least one")


@dataclass(frozen=True, slots=True)
class NegotiatedCameraFormat:
    width: int
    height: int
    fps: float
    fourcc: str


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str, int], CaptureLike]


def decode_fourcc(value: float) -> str:
    packed = int(value)
    return "".join(chr((packed >> (8 * index)) & 0xFF) for index in range(4))


class CameraReader:
    """Continuously capture a V4L2 camera into a :class:`LatestFrameSlot`."""

    def __init__(
        self,
        config: CameraConfig,
        slot: LatestFrameSlot | None = None,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.config = config
        self.slot = slot or LatestFrameSlot()
        self._capture_factory = capture_factory
        self._capture: CaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._error: BaseException | None = None
        self._published_frames = 0
        self._read_failures = 0
        self._capture_timestamps_ns: deque[int] = deque(maxlen=180)
        self._negotiated: NegotiatedCameraFormat | None = None

    def start(self) -> Self:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("CameraReader is already running")
            self._stop_event.clear()
            self._error = None
            self._capture = self._open_capture()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="camera-capture",
                daemon=False,
            )
            self._thread.start()
        return self

    def _open_capture(self) -> CaptureLike:
        import cv2

        factory = self._capture_factory or cv2.VideoCapture
        capture = factory(self.config.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(f"unable to open V4L2 camera {self.config.device}")

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        capture.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self.config.buffer_size))

        negotiated = NegotiatedCameraFormat(
            width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            fourcc=decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)).rstrip("\x00"),
        )
        try:
            self._validate_negotiated_format(negotiated)
        except BaseException:
            capture.release()
            raise
        self._negotiated = negotiated
        return capture

    def _validate_negotiated_format(self, actual: NegotiatedCameraFormat) -> None:
        if not self.config.strict_format:
            return
        mismatches: list[str] = []
        if actual.width != self.config.width:
            mismatches.append(f"width {actual.width} != {self.config.width}")
        if actual.height != self.config.height:
            mismatches.append(f"height {actual.height} != {self.config.height}")
        if abs(actual.fps - self.config.fps) > 0.5:
            mismatches.append(f"fps {actual.fps:g} != {self.config.fps:g}")
        if actual.fourcc.upper() != self.config.fourcc.upper():
            mismatches.append(f"FOURCC {actual.fourcc!r} != {self.config.fourcc!r}")
        if mismatches:
            raise CameraOpenError(
                f"camera {self.config.device} silently changed format: " + ", ".join(mismatches)
            )

    def _capture_loop(self) -> None:
        capture = self._capture
        assert capture is not None
        sequence = -1
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                captured_ns = time.monotonic_ns()
                if not ok or frame is None:
                    if self._stop_event.is_set():
                        break
                    consecutive_failures += 1
                    with self._stats_lock:
                        self._read_failures += 1
                    if consecutive_failures >= self.config.max_consecutive_failures:
                        raise CameraReadError(
                            f"camera {self.config.device} failed "
                            f"{consecutive_failures} consecutive reads"
                        )
                    time.sleep(self.config.retry_delay_seconds)
                    continue

                consecutive_failures = 0
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise CameraReadError(
                        "OpenCV camera output violated the BGR uint8 HxWx3 contract: "
                        f"shape={getattr(frame, 'shape', None)}, "
                        f"dtype={getattr(frame, 'dtype', None)}"
                    )
                negotiated = self._negotiated
                assert negotiated is not None
                expected_shape = (negotiated.height, negotiated.width)
                if frame.shape[:2] != expected_shape:
                    raise CameraReadError(
                        "captured frame shape changed after negotiation: "
                        f"{frame.shape[:2]} != {expected_shape}"
                    )
                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)
                frame.setflags(write=False)
                sequence += 1
                self.slot.publish(CapturedFrame(sequence, captured_ns, frame))
                with self._stats_lock:
                    self._published_frames += 1
                    self._capture_timestamps_ns.append(captured_ns)
        except Exception as exc:  # noqa: BLE001 - propagate worker failures to the owner.
            self._error = exc
            self._stop_event.set()
        finally:
            capture.release()
            self.slot.close()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        # Most V4L2 reads return promptly after the stop event. Avoid releasing
        # the same OpenCV handle concurrently from this thread and the capture
        # thread because amd_isp_capture can otherwise remain stuck in STREAMON.
        if thread is not None and thread.is_alive():
            capture = self._capture
            if capture is not None:
                capture.release()
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            raise CameraShutdownError(
                f"camera capture thread did not stop within {timeout * 2:g} seconds"
            )

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise CameraError("camera capture worker failed") from self._error

    @property
    def negotiated(self) -> NegotiatedCameraFormat | None:
        return self._negotiated

    @property
    def published_frames(self) -> int:
        with self._stats_lock:
            return self._published_frames

    @property
    def read_failures(self) -> int:
        with self._stats_lock:
            return self._read_failures

    @property
    def capture_fps(self) -> float:
        with self._stats_lock:
            if len(self._capture_timestamps_ns) < 2:
                return 0.0
            elapsed_ns = self._capture_timestamps_ns[-1] - self._capture_timestamps_ns[0]
            return (
                (len(self._capture_timestamps_ns) - 1) * 1e9 / elapsed_ns if elapsed_ns > 0 else 0.0
            )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


@dataclass(frozen=True, slots=True)
class VideoFileConfig:
    path: str
    width: int = 1280
    height: int = 720
    fps: float | None = None
    loop: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video width and height must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("video FPS override must be positive")


class VideoFileReader:
    """Replay a video at wall-clock speed into the same latest-frame contract."""

    def __init__(
        self,
        config: VideoFileConfig,
        slot: LatestFrameSlot | None = None,
    ) -> None:
        self.config = config
        self.slot = slot or LatestFrameSlot()
        self._capture: CaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._error: BaseException | None = None
        self._published_frames = 0
        self._read_failures = 0
        self._capture_timestamps_ns: deque[int] = deque(maxlen=180)
        self._negotiated: NegotiatedCameraFormat | None = None

    def start(self) -> Self:
        import cv2

        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("VideoFileReader is already running")
        path = Path(self.config.path)
        if not path.is_file():
            raise CameraOpenError(f"video file not found: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(f"unable to open video file {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        playback_fps = self.config.fps or (source_fps if source_fps > 0 else 30.0)
        packed_fourcc = capture.get(cv2.CAP_PROP_FOURCC)
        self._negotiated = NegotiatedCameraFormat(
            width=self.config.width,
            height=self.config.height,
            fps=playback_fps,
            fourcc=decode_fourcc(packed_fourcc).rstrip("\x00") or "FILE",
        )
        self._capture = capture
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="video-file-replay",
            daemon=False,
        )
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        import cv2

        capture = self._capture
        negotiated = self._negotiated
        assert capture is not None and negotiated is not None
        sequence = -1
        next_publish_at = time.monotonic()
        period_seconds = 1.0 / negotiated.fps
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if self.config.loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                        consecutive_failures += 1
                        if consecutive_failures > 2:
                            raise CameraReadError(
                                f"video {self.config.path} could not restart after EOF"
                            )
                        continue
                    return
                consecutive_failures = 0

                if frame.shape[:2] != (negotiated.height, negotiated.width):
                    frame = cv2.resize(
                        frame,
                        (negotiated.width, negotiated.height),
                        interpolation=cv2.INTER_AREA,
                    )
                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)

                wait_seconds = max(0.0, next_publish_at - time.monotonic())
                if self._stop_event.wait(wait_seconds):
                    return
                captured_ns = time.monotonic_ns()
                frame.setflags(write=False)
                sequence += 1
                self.slot.publish(CapturedFrame(sequence, captured_ns, frame))
                with self._stats_lock:
                    self._published_frames += 1
                    self._capture_timestamps_ns.append(captured_ns)
                next_publish_at += period_seconds
                now = time.monotonic()
                if next_publish_at < now - period_seconds:
                    next_publish_at = now
        except Exception as exc:  # noqa: BLE001 - propagate replay failures to the owner.
            self._error = exc
            self._stop_event.set()
        finally:
            capture.release()
            self.slot.close()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            capture = self._capture
            if capture is not None:
                capture.release()
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            raise CameraShutdownError(
                f"video replay thread did not stop within {timeout * 2:g} seconds"
            )

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise CameraError("video replay worker failed") from self._error

    @property
    def negotiated(self) -> NegotiatedCameraFormat | None:
        return self._negotiated

    @property
    def published_frames(self) -> int:
        with self._stats_lock:
            return self._published_frames

    @property
    def read_failures(self) -> int:
        with self._stats_lock:
            return self._read_failures

    @property
    def capture_fps(self) -> float:
        with self._stats_lock:
            if len(self._capture_timestamps_ns) < 2:
                return 0.0
            elapsed_ns = self._capture_timestamps_ns[-1] - self._capture_timestamps_ns[0]
            return (
                (len(self._capture_timestamps_ns) - 1) * 1e9 / elapsed_ns
                if elapsed_ns > 0
                else 0.0
            )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
