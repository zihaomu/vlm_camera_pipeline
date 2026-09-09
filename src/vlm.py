"""Strict GPU-only llama.cpp VLM service and latest-snapshot worker."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

from .camera_io import CapturedFrame, LatestFrameSlot

EXPECTED_LLAMA_COMMIT = "0b1bad14ff204627636aeb1de22ddcd5acb859d4"
EXPECTED_MODEL_SHA256 = "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"
EXPECTED_MMPROJ_SHA256 = "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"
EXPECTED_HIP_CODE_OBJECT = "hipv4-amdgcn-amd-amdhsa--gfx1151"


class VlmError(RuntimeError):
    """Base error for the local VLM path."""


class VlmStartupError(VlmError):
    """The local server failed a startup or GPU-only gate."""


class VlmRequestError(VlmError):
    """One caption request failed or returned an invalid response."""


@dataclass(frozen=True, slots=True)
class VlmCaption:
    sequence: int
    captured_ns: int
    requested_ns: int
    completed_ns: int
    text: str


class LatestCaptionSlot:
    """Thread-safe depth-one caption publication slot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: VlmCaption | None = None

    def publish(self, caption: VlmCaption) -> None:
        if not caption.text.strip():
            raise ValueError("caption text must not be empty")
        with self._lock:
            if self._latest is not None and caption.sequence <= self._latest.sequence:
                raise ValueError(
                    "caption sequences must increase monotonically: "
                    f"latest={self._latest.sequence}, new={caption.sequence}"
                )
            self._latest = caption

    def latest(self) -> VlmCaption | None:
        with self._lock:
            return self._latest

    def latest_fresh(self, now_ns: int, expiry_seconds: float) -> VlmCaption | None:
        latest = self.latest()
        if latest is None:
            return None
        if now_ns - latest.completed_ns > int(expiry_seconds * 1e9):
            return None
        return latest


class VlmMetrics(Protocol):
    def record_vlm_started(self) -> None: ...

    def record_vlm_success(
        self,
        frame: CapturedFrame,
        requested_ns: int,
        completed_ns: int,
        text: str,
    ) -> None: ...

    def record_vlm_failure(
        self,
        requested_ns: int,
        completed_ns: int,
        error: BaseException,
    ) -> None: ...

    def record_vlm_cancelled(self, requested_ns: int, completed_ns: int) -> None: ...


class VlmEngine(Protocol):
    mode: str

    def start(self) -> Self: ...

    def caption_bgr(self, frame_bgr: np.ndarray) -> str: ...

    def stop(self) -> None: ...

    def runtime_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    server_path: str = "third_party/llama.cpp/build-gfx1151/bin/llama-server"
    model_path: str = "models/Qwen3-VL-8B-Instruct-Q8_0.gguf"
    mmproj_path: str = "models/mmproj-F16.gguf"
    log_path: str = "output/realtime/llama-server.log"
    host: str = "127.0.0.1"
    port: int = 0
    context_size: int = 4096
    image_max_tokens: int = 512
    max_tokens: int = 64
    jpeg_quality: int = 85
    timeout_seconds: float = 120.0
    startup_timeout_seconds: float = 180.0
    hip_stream_priority: int = 0
    prompt: str = (
        "Describe this camera image in one short English sentence. "
        "Mention the main objects and action; do not speculate."
    )

    def __post_init__(self) -> None:
        if self.host != "127.0.0.1":
            raise ValueError("the embedded VLM server must bind only to 127.0.0.1")
        if not 0 <= self.port <= 65535:
            raise ValueError("VLM server port must be between 0 and 65535")
        if self.context_size < 1024:
            raise ValueError("VLM context size must be at least 1024")
        if self.image_max_tokens < 1 or self.max_tokens < 1:
            raise ValueError("VLM image and output token limits must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("VLM JPEG quality must be between 1 and 100")
        if self.timeout_seconds <= 0 or self.startup_timeout_seconds <= 0:
            raise ValueError("VLM timeouts must be positive")
        if self.hip_stream_priority not in (-1, 0, 1):
            raise ValueError("HIP stream priority must be -1 (high), 0 (normal), or 1 (low)")
        if not self.prompt.strip():
            raise ValueError("VLM prompt must not be empty")


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return (workspace / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _choose_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _request_json(
    url: str,
    *,
    timeout: float,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        try:
            response_body = json.loads(response_text)
        except json.JSONDecodeError:
            response_body = {"raw": response_text}
        return exc.code, response_body


def _capabilities(body: dict[str, Any]) -> set[str]:
    capabilities: set[str] = set()
    for key in ("models", "data"):
        for entry in body.get(key, []):
            if isinstance(entry, dict):
                capabilities.update(str(item) for item in entry.get("capabilities", []))
    return capabilities


class LlamaCppVlm:
    """Own a localhost llama-server and require full gfx1151 GPU offload."""

    mode = "llamacpp"

    def __init__(self, config: LlamaCppConfig, *, workspace: str | Path) -> None:
        self.config = config
        self.workspace = Path(workspace).resolve()
        self.server_path = _resolve(self.workspace, config.server_path)
        self.model_path = _resolve(self.workspace, config.model_path)
        self.mmproj_path = _resolve(self.workspace, config.mmproj_path)
        self.log_path = _resolve(self.workspace, config.log_path)
        self._process: subprocess.Popen[str] | None = None
        self._log_stream: Any = None
        self._log_thread: threading.Thread | None = None
        self._log_lock = threading.Lock()
        self._startup_log_lines: list[str] = []
        self._startup_verified = threading.Event()
        self._request_lock = threading.Lock()
        self._api_key = f"vlm-local-{secrets.token_urlsafe(24)}"
        self._port = config.port or _choose_free_port(config.host)
        self._startup_seconds: float | None = None
        self._gpu_proof: dict[str, Any] | None = None
        self._capability_set: set[str] = set()
        self._command: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self._port}"

    def start(self) -> Self:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("llama-server is already running")
        print("[vlm] stage=preflight sha256=verify code_object=gfx1151", flush=True)
        self._preflight()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._command = self._build_command()
        self._startup_log_lines.clear()
        self._startup_verified.clear()
        print("[vlm] stage=server_start device=ROCm0 gpu_layers=all fit=off", flush=True)
        self._log_stream = self.log_path.open("w", encoding="utf-8", buffering=1)
        process_environment = os.environ.copy()
        process_environment["GGML_HIP_STREAM_PRIORITY"] = str(self.config.hip_stream_priority)
        self._process = subprocess.Popen(
            self._command,
            cwd=self.workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=process_environment,
        )
        self._log_thread = threading.Thread(
            target=self._copy_server_log,
            name="llama-server-log",
            daemon=False,
        )
        self._log_thread.start()
        try:
            self._startup_seconds = self._wait_for_health()
            self._gpu_proof = self._validate_gpu_runtime()
            self._startup_verified.set()
            self._capability_set = self._read_capabilities()
            if "multimodal" not in self._capability_set:
                raise VlmStartupError(
                    "llama-server does not advertise its required multimodal capability"
                )
        except Exception:
            tail = self._log_tail()
            self.stop()
            if tail:
                print(tail, file=sys.stderr)
            raise
        print(
            f"[vlm] stage=ready seconds={self._startup_seconds:.3f} "
            f"layers={self._gpu_proof['model_layers_offloaded']}/"
            f"{self._gpu_proof['model_layers_total']} mmproj=ROCm0",
            flush=True,
        )
        return self

    def _copy_server_log(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            with self._log_lock:
                if not self._startup_verified.is_set():
                    self._startup_log_lines.append(line)
                # Full logs are needed for the startup proof but are too noisy for a
                # continuously running demo. Afterwards retain timings and faults only.
                keep_runtime_line = any(
                    marker in line
                    for marker in (
                        " W ",
                        " E ",
                        " F ",
                        "print_timing",
                        "cleaning up",
                        "common_memory_breakdown_print",
                        "HIP IPC v2",
                        "HIP device embedding",
                        "HIP embedding D2D",
                        "destination tensor",
                        "physical device mismatch",
                        "byte size mismatch",
                        "null tensor",
                        "error",
                        "failed",
                    )
                )
                if not self._startup_verified.is_set() or keep_runtime_line:
                    self._log_stream.write(line)

    def _preflight(self) -> None:
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            raise VlmStartupError("HSA_OVERRIDE_GFX_VERSION is set; refusing architecture spoofing")
        for label, path in (
            ("llama-server", self.server_path),
            ("Qwen3-VL model", self.model_path),
            ("multimodal projector", self.mmproj_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        if _sha256(self.model_path) != EXPECTED_MODEL_SHA256:
            raise VlmStartupError(f"Qwen3-VL model SHA-256 mismatch: {self.model_path}")
        if _sha256(self.mmproj_path) != EXPECTED_MMPROJ_SHA256:
            raise VlmStartupError(f"mmproj SHA-256 mismatch: {self.mmproj_path}")
        llama_dir = self.server_path.parents[2]
        try:
            head = subprocess.run(
                ["git", "-C", str(llama_dir), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise VlmStartupError("cannot verify the llama.cpp source identity") from exc
        if head != EXPECTED_LLAMA_COMMIT:
            raise VlmStartupError(f"unexpected llama.cpp commit: {head}")
        libraries = sorted(self.server_path.parent.glob("libggml-hip.so*"))
        if not libraries:
            raise VlmStartupError("libggml-hip.so is missing beside llama-server")
        try:
            output = subprocess.run(
                ["roc-obj-ls", str(libraries[0])],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise VlmStartupError("cannot inspect llama.cpp HIP code objects") from exc
        code_objects = sorted(
            {
                fields[1]
                for line in output.splitlines()
                if len(fields := line.split()) >= 2 and fields[1].startswith("hipv")
            }
        )
        if code_objects != [EXPECTED_HIP_CODE_OBJECT]:
            raise VlmStartupError(f"unexpected HIP code objects: {code_objects}")

    def _build_command(self) -> list[str]:
        return [
            str(self.server_path),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--alias",
            "qwen3-vl-8b-instruct",
            "--host",
            self.config.host,
            "--port",
            str(self._port),
            "--api-key",
            self._api_key,
            "--device",
            "ROCm0",
            "--gpu-layers",
            "all",
            "--fit",
            "off",
            "--mmproj-offload",
            "--ctx-size",
            str(self.config.context_size),
            "--parallel",
            "1",
            "--threads",
            "8",
            "--threads-batch",
            "8",
            "--batch-size",
            "512",
            "--ubatch-size",
            "256",
            "--image-max-tokens",
            str(self.config.image_max_tokens),
            "--mtmd-batch-max-tokens",
            str(self.config.image_max_tokens),
            "--reasoning",
            "off",
            # The default prompt cache checkpoints GPU KV state to host between
            # camera requests. Reuse is negligible for changing images, so keep
            # this control path disabled in the real-time server.
            "--cache-ram",
            "0",
            "--no-webui",
            "--no-slots",
            "--verbosity",
            "5",
        ]

    def _wait_for_health(self) -> float:
        process = self._process
        assert process is not None
        started = time.monotonic()
        deadline = started + self.config.startup_timeout_seconds
        last_status: int | str = "not contacted"
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise VlmStartupError(
                    f"llama-server exited during model load with code {return_code}"
                )
            try:
                status, body = _request_json(f"{self.base_url}/health", timeout=2.0)
                last_status = status
                if status == 200 and body.get("status") == "ok":
                    return time.monotonic() - started
            except (OSError, TimeoutError):
                last_status = "connection pending"
            time.sleep(0.25)
        raise VlmStartupError(
            f"llama-server health timeout after {self.config.startup_timeout_seconds:g}s "
            f"({last_status})"
        )

    def _validate_gpu_runtime(self) -> dict[str, Any]:
        process = self._process
        assert process is not None
        deadline = time.monotonic() + 5.0
        log_text = ""
        layer_matches: list[tuple[str, str]] = []
        while time.monotonic() < deadline:
            with self._log_lock:
                log_text = "".join(self._startup_log_lines)
            layer_matches = re.findall(
                r"offloaded\s+(\d+)/(\d+)\s+layers to GPU",
                log_text,
            )
            if layer_matches and re.search(r"CLIP using ROCm0 backend", log_text):
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        if not layer_matches:
            raise VlmStartupError("server log does not prove model-layer GPU offload")
        offloaded, total = (int(value) for value in layer_matches[-1])
        if offloaded <= 0 or offloaded != total:
            raise VlmStartupError(f"partial VLM model offload: {offloaded}/{total} layers")
        if not re.search(r"CLIP using ROCm0 backend", log_text):
            raise VlmStartupError("server log does not prove mmproj execution on ROCm0")
        if re.search(r"CLIP using CPU backend", log_text):
            raise VlmStartupError("mmproj CPU fallback detected")
        try:
            has_kfd = any(
                os.readlink(fd) == "/dev/kfd" for fd in Path(f"/proc/{process.pid}/fd").iterdir()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            has_kfd = False
        if not has_kfd:
            raise VlmStartupError("llama-server has no open /dev/kfd descriptor")
        return {
            "model_layers_offloaded": offloaded,
            "model_layers_total": total,
            "mmproj_backend": "ROCm0",
            "process_has_dev_kfd": True,
            "cpu_fallback_detected": False,
        }

    def _read_capabilities(self) -> set[str]:
        capabilities: set[str] = set()
        for endpoint in ("/v1/models", "/models"):
            status, body = _request_json(
                f"{self.base_url}{endpoint}",
                timeout=5.0,
                api_key=self._api_key,
            )
            if status == 200:
                capabilities.update(_capabilities(body))
        return capabilities

    def caption_bgr(self, frame_bgr: np.ndarray) -> str:
        process = self._process
        if process is None or process.poll() is not None:
            raise VlmRequestError("llama-server is not running")
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("VLM input must be an HxWx3 uint8 BGR frame")
        import cv2

        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not encoded:
            raise VlmRequestError("OpenCV could not encode the VLM snapshot")
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": "qwen3-vl-8b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config.prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": self.config.max_tokens,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with self._request_lock:
            try:
                status, response = _request_json(
                    f"{self.base_url}/v1/chat/completions",
                    timeout=self.config.timeout_seconds,
                    api_key=self._api_key,
                    payload=payload,
                )
            except (OSError, TimeoutError) as exc:
                raise VlmRequestError(f"local VLM request failed: {exc}") from exc
        if status != 200:
            raise VlmRequestError(f"local VLM returned HTTP {status}: {response}")
        try:
            caption = str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VlmRequestError(f"invalid local VLM response: {response}") from exc
        if not caption:
            raise VlmRequestError("local VLM returned an empty caption")
        return " ".join(caption.split())

    def runtime_info(self) -> dict[str, Any]:
        safe_command = ["<redacted>" if item == self._api_key else item for item in self._command]
        return {
            "mode": self.mode,
            "server": str(self.server_path),
            "model": str(self.model_path),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "mmproj": str(self.mmproj_path),
            "mmproj_sha256": EXPECTED_MMPROJ_SHA256,
            "llama_cpp_commit": EXPECTED_LLAMA_COMMIT,
            "device": "ROCm0",
            "hip_code_objects": [EXPECTED_HIP_CODE_OBJECT],
            "startup_seconds": self._startup_seconds,
            "capabilities": sorted(self._capability_set),
            "gpu_proof": self._gpu_proof,
            "command": safe_command,
            "timeout_seconds": self.config.timeout_seconds,
            "context_size": self.config.context_size,
            "image_max_tokens": self.config.image_max_tokens,
            "max_tokens": self.config.max_tokens,
            "hip_stream_priority_requested": self.config.hip_stream_priority,
            "prompt_cache_ram_mib": 0,
        }

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if self._log_thread is not None and self._log_thread.is_alive():
            self._log_thread.join(timeout=5.0)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if self._log_stream is not None:
            self._log_stream.close()
        self._process = None
        self._log_stream = None
        self._log_thread = None

    def _log_tail(self, lines: int = 80) -> str:
        try:
            with self._log_lock:
                return "\n".join(
                    self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                        -lines:
                    ]
                )
        except OSError:
            return ""

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


class VlmWorker:
    """Request captions from only the latest frame, with no request backlog."""

    def __init__(
        self,
        frame_slot: LatestFrameSlot,
        caption_slot: LatestCaptionSlot,
        engine: VlmEngine,
        metrics: VlmMetrics,
        *,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("VLM interval must be positive")
        self.frame_slot = frame_slot
        self.caption_slot = caption_slot
        self.engine = engine
        self.metrics = metrics
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("VlmWorker is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-vlm", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        next_request_at = time.monotonic()
        last_sequence = -1
        try:
            while not self._stop_event.is_set():
                remaining = max(0.0, next_request_at - time.monotonic())
                if self._stop_event.wait(remaining):
                    return
                frame = self.frame_slot.latest()
                if frame is None or frame.sequence <= last_sequence:
                    if self._stop_event.wait(0.05):
                        return
                    continue
                last_sequence = frame.sequence
                requested_ns = time.monotonic_ns()
                self.metrics.record_vlm_started()
                try:
                    text = self.engine.caption_bgr(frame.bgr)
                except Exception as exc:  # noqa: BLE001 - isolate VLM request failures.
                    completed_ns = time.monotonic_ns()
                    if self._stop_event.is_set():
                        self.metrics.record_vlm_cancelled(requested_ns, completed_ns)
                        return
                    self.metrics.record_vlm_failure(requested_ns, completed_ns, exc)
                    print(f"VLM request failed without stopping camera: {exc}", file=sys.stderr)
                else:
                    completed_ns = time.monotonic_ns()
                    self.caption_slot.publish(
                        VlmCaption(
                            sequence=frame.sequence,
                            captured_ns=frame.captured_ns,
                            requested_ns=requested_ns,
                            completed_ns=completed_ns,
                            text=text,
                        )
                    )
                    self.metrics.record_vlm_success(
                        frame,
                        requested_ns,
                        completed_ns,
                        text,
                    )
                next_request_at = max(
                    requested_ns / 1e9 + self.interval_seconds,
                    time.monotonic(),
                )
        except Exception as exc:  # noqa: BLE001 - propagate worker logic failures.
            self._error = exc
            self._stop_event.set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float = 15.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError(f"VLM worker did not stop within {timeout:g} seconds")

    def wait(self, timeout: float) -> bool:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("VLM worker failed") from self._error
