#!/usr/bin/env python3
"""Run one Qwen3-VL image request and reject any llama.cpp CPU fallback."""

from __future__ import annotations

import argparse
import base64
import csv
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
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]

EXPECTED_MODEL_SHA256 = "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"
EXPECTED_MMPROJ_SHA256 = "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"
EXPECTED_LLAMA_COMMIT = "0b1bad14ff204627636aeb1de22ddcd5acb859d4"
EXPECTED_CODE_OBJECT = "hipv4-amdgcn-amd-amdhsa--gfx1151"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default="third_party/llama.cpp/build-gfx1151/bin/llama-server",
    )
    parser.add_argument("--model", default="models/Qwen3-VL-8B-Instruct-Q8_0.gguf")
    parser.add_argument("--mmproj", default="models/mmproj-F16.gguf")
    parser.add_argument(
        "--image",
        default="output/realtime/screenshots/pytorch-replay.jpg",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in one short English sentence. Mention the main objects.",
    )
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--image-max-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log", default="output/bringup/logs/05-qwen3-vl-gpu-smoke.log")
    parser.add_argument("--output", default="output/bringup/qwen3-vl-gpu-smoke.json")
    return parser


def resolve_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = WORKSPACE / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose_port(requested: int) -> int:
    if requested:
        if not 1 <= requested <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def request_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float,
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
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = {"raw": response_body}
        return exc.code, parsed


def wait_for_health(process: subprocess.Popen[str], url: str, timeout: float) -> float:
    started = time.monotonic()
    deadline = started + timeout
    last_status: int | str = "not contacted"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"llama-server exited during model load with code {return_code}")
        try:
            status, body = request_json(url, timeout=2.0)
            last_status = status
            if status == 200 and body.get("status") == "ok":
                return time.monotonic() - started
        except (OSError, TimeoutError):
            last_status = "connection pending"
        time.sleep(0.25)
    raise TimeoutError(f"llama-server health timeout after {timeout:g}s ({last_status})")


def gpu_code_objects(server: Path) -> tuple[str, ...]:
    candidates = sorted(server.parent.glob("libggml-hip.so*"))
    if not candidates:
        raise FileNotFoundError(f"libggml-hip.so not found beside {server}")
    completed = subprocess.run(
        ["roc-obj-ls", str(candidates[0])],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return tuple(
        sorted(
            {
                fields[1]
                for line in completed.stdout.splitlines()
                if len(fields := line.split()) >= 2 and fields[1].startswith("hipv")
            }
        )
    )


def process_has_kfd(process: subprocess.Popen[str]) -> bool:
    fd_dir = Path(f"/proc/{process.pid}/fd")
    try:
        return any(os.readlink(fd) == "/dev/kfd" for fd in fd_dir.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def validate_gpu_log(log_text: str) -> dict[str, Any]:
    layer_matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log_text)
    if not layer_matches:
        raise RuntimeError("server log does not prove model-layer GPU offload")
    offloaded, total = (int(value) for value in layer_matches[-1])
    if offloaded <= 0 or offloaded != total:
        raise RuntimeError(f"partial model offload detected: {offloaded}/{total} layers")
    if not re.search(r"CLIP using ROCm0 backend", log_text):
        raise RuntimeError("server log does not prove mmproj execution on ROCm0")
    if re.search(r"CLIP using CPU backend", log_text):
        raise RuntimeError("mmproj CPU fallback detected")
    return {
        "model_layers_offloaded": offloaded,
        "model_layers_total": total,
        "mmproj_backend": "ROCm0",
        "cpu_fallback_detected": False,
    }


def model_capabilities(body: dict[str, Any]) -> set[str]:
    capabilities: set[str] = set()
    for key in ("models", "data"):
        for entry in body.get(key, []):
            if isinstance(entry, dict):
                capabilities.update(str(item) for item in entry.get("capabilities", []))
    return capabilities


@dataclass
class GpuSample:
    use_percent: int
    vram_used_bytes: int


class GpuMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[GpuSample] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="rocm-smi-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                )
                rows = list(csv.DictReader(completed.stdout.splitlines()))
                if rows:
                    row = rows[0]
                    self.samples.append(
                        GpuSample(
                            use_percent=int(row["GPU use (%)"]),
                            vram_used_bytes=int(row["VRAM Total Used Memory (B)"]),
                        )
                    )
            except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError):
                pass
            self._stop.wait(0.2)


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def log_tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise RuntimeError("HSA_OVERRIDE_GFX_VERSION is set; refusing architecture spoofing")
    if args.ctx_size < 1024 or args.image_max_tokens < 1 or args.max_tokens < 1:
        raise ValueError("context, image, and output token limits must be positive")
    if args.timeout <= 0 or args.startup_timeout <= 0:
        raise ValueError("timeouts must be positive")

    server = resolve_file(args.server, "llama-server")
    model = resolve_file(args.model, "Qwen3-VL model")
    mmproj = resolve_file(args.mmproj, "multimodal projector")
    image = resolve_file(args.image, "test image")
    if sha256(model) != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Qwen3-VL SHA-256 mismatch: {model}")
    if sha256(mmproj) != EXPECTED_MMPROJ_SHA256:
        raise RuntimeError(f"mmproj SHA-256 mismatch: {mmproj}")
    llama_head = git_head(server.parents[2])
    if llama_head != EXPECTED_LLAMA_COMMIT:
        raise RuntimeError(f"unexpected llama.cpp commit: {llama_head}")
    code_objects = gpu_code_objects(server)
    if code_objects != (EXPECTED_CODE_OBJECT,):
        raise RuntimeError(f"unexpected HIP code objects: {code_objects}")

    port = choose_port(args.port)
    api_key = f"vlm-local-{secrets.token_urlsafe(24)}"
    alias = "qwen3-vl-8b-instruct"
    log_path = Path(args.log)
    output_path = Path(args.output)
    if not log_path.is_absolute():
        log_path = WORKSPACE / log_path
    if not output_path.is_absolute():
        output_path = WORKSPACE / output_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server),
        "--model",
        str(model),
        "--mmproj",
        str(mmproj),
        "--alias",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--device",
        "ROCm0",
        "--gpu-layers",
        "all",
        "--fit",
        "off",
        "--mmproj-offload",
        "--ctx-size",
        str(args.ctx_size),
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
        str(args.image_max_tokens),
        "--mtmd-batch-max-tokens",
        str(args.image_max_tokens),
        "--reasoning",
        "off",
        "--no-webui",
        "--no-slots",
        "--verbosity",
        "5",
    ]
    safe_command = ["<redacted>" if value == api_key else value for value in command]
    print("[vlm 1/4] starting llama-server with forced ROCm0 offload", flush=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    process: subprocess.Popen[str] | None = None
    monitor = GpuMonitor()
    result: dict[str, Any] | None = None
    with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            health_seconds = wait_for_health(
                process,
                f"http://127.0.0.1:{port}/health",
                args.startup_timeout,
            )
            log_stream.flush()
            gpu_proof = validate_gpu_log(log_path.read_text(encoding="utf-8", errors="replace"))
            if not process_has_kfd(process):
                raise RuntimeError("llama-server has no open /dev/kfd descriptor")
            print(
                f"[vlm 2/4] ready in {health_seconds:.2f}s; "
                f"{gpu_proof['model_layers_offloaded']}/{gpu_proof['model_layers_total']} "
                "model layers + mmproj on ROCm0",
                flush=True,
            )

            _, v1_models = request_json(
                f"http://127.0.0.1:{port}/v1/models",
                api_key=api_key,
                timeout=5.0,
            )
            _, native_models = request_json(
                f"http://127.0.0.1:{port}/models",
                api_key=api_key,
                timeout=5.0,
            )
            capabilities = model_capabilities(v1_models) | model_capabilities(native_models)
            if "multimodal" not in capabilities:
                raise RuntimeError(f"server does not advertise multimodal capability: {capabilities}")

            mime_type = "image/jpeg" if image.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            image_url = f"data:{mime_type};base64,{base64.b64encode(image.read_bytes()).decode()}"
            payload = {
                "model": alias,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": args.max_tokens,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            }
            print("[vlm 3/4] submitting a real JPEG vision request", flush=True)
            monitor.start()
            request_started = time.monotonic()
            status, response = request_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                api_key=api_key,
                payload=payload,
                timeout=args.timeout,
            )
            request_seconds = time.monotonic() - request_started
            monitor.stop()
            if status != 200:
                raise RuntimeError(f"vision request failed with HTTP {status}: {response}")
            try:
                caption = str(response["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"invalid chat completion response: {response}") from exc
            if not caption:
                raise RuntimeError("VLM returned an empty caption")
            print(f"[vlm 4/4] caption: {caption}", flush=True)

            result = {
                "schema_version": 1,
                "started_at": started_at,
                "server_ready_seconds": round(health_seconds, 3),
                "request_seconds": round(request_seconds, 3),
                "caption": caption,
                "input_image": str(image.relative_to(WORKSPACE)),
                "llama_cpp": {
                    "commit": llama_head,
                    "server": str(server.relative_to(WORKSPACE)),
                    "command": safe_command,
                    "hip_code_objects": list(code_objects),
                },
                "models": {
                    "language": {
                        "path": str(model.relative_to(WORKSPACE)),
                        "bytes": model.stat().st_size,
                        "sha256": EXPECTED_MODEL_SHA256,
                    },
                    "mmproj": {
                        "path": str(mmproj.relative_to(WORKSPACE)),
                        "bytes": mmproj.stat().st_size,
                        "sha256": EXPECTED_MMPROJ_SHA256,
                    },
                },
                "gpu": {
                    **gpu_proof,
                    "device": "ROCm0",
                    "process_has_dev_kfd": True,
                    "sample_count": len(monitor.samples),
                    "use_percent_max": max(
                        (sample.use_percent for sample in monitor.samples), default=None
                    ),
                    "vram_used_bytes_max": max(
                        (sample.vram_used_bytes for sample in monitor.samples), default=None
                    ),
                },
                "server_capabilities": sorted(capabilities),
                "http_status": status,
            }
        except Exception:
            print(log_tail(log_path), file=sys.stderr)
            raise
        finally:
            monitor.stop()
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)

    if result is None:
        raise RuntimeError("VLM smoke produced no result")
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(f"wrote {output_path.relative_to(WORKSPACE)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
