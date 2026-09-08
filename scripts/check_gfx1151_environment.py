#!/usr/bin/env python3
"""Capture an auditable M0/M1 environment manifest without changing the host."""

from __future__ import annotations

import argparse
import grp
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_CACHE = WORKSPACE / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))


def run_command(arguments: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "output": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except OSError:
        return None


def current_groups() -> list[str]:
    return sorted({grp.getgrgid(group_id).gr_name for group_id in os.getgroups()})


def import_version(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - native import failures belong in the manifest.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", None)}


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - native import failures belong in the manifest.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    status: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count():
        properties = torch.cuda.get_device_properties(0)
        status.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "gcn_arch": getattr(properties, "gcnArchName", None),
            }
        )
    return status


def camera_status(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": device,
        "exists": Path(device).exists(),
        "readable": os.access(device, os.R_OK),
        "writable": os.access(device, os.W_OK),
    }
    query = run_command(["v4l2-ctl", "--device", device, "--get-fmt-video"])
    result["v4l2_query"] = query
    return result


def collect(device: str) -> dict[str, Any]:
    rocminfo = run_command(["rocminfo"])
    rocminfo_text = rocminfo["output"]
    architectures = sorted(set(re.findall(r"\bgfx\d+[a-z]*\b", rocminfo_text)))
    groups = current_groups()
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages_limit_text = read_text("/sys/module/ttm/parameters/pages_limit")
    pages_limit = int(pages_limit_text) if pages_limit_text else None
    return {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "groups": groups,
        "devices": {
            "kfd": {
                "exists": Path("/dev/kfd").exists(),
                "readable": os.access("/dev/kfd", os.R_OK),
                "writable": os.access("/dev/kfd", os.W_OK),
            },
            "render_nodes": sorted(str(path) for path in Path("/dev/dri").glob("renderD*")),
        },
        "rocm": {
            "rocminfo_ok": rocminfo["ok"],
            "architectures": architectures,
            "rocm_smi": run_command(["rocm-smi", "--showproductname", "--showmeminfo", "vram"]),
        },
        "ttm": {
            "page_size_bytes": page_size,
            "pages_limit": pages_limit,
            "bytes_limit": pages_limit * page_size if pages_limit is not None else None,
            "page_pool_size": _optional_int(read_text("/sys/module/ttm/parameters/page_pool_size")),
        },
        "camera": camera_status(device),
        "python_packages": {
            "numpy": import_version("numpy"),
            "opencv": import_version("cv2"),
            "ultralytics": import_version("ultralytics"),
            "torch": torch_status(),
        },
    }


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def evaluate(manifest: dict[str, Any], require_torch: bool) -> list[str]:
    failures: list[str] = []
    if "gfx1151" not in manifest["rocm"]["architectures"]:
        failures.append("rocminfo does not report gfx1151")
    if not manifest["devices"]["kfd"]["readable"]:
        failures.append("/dev/kfd is not readable")
    if not manifest["devices"]["render_nodes"]:
        failures.append("no /dev/dri/renderD* node was found")
    if not {"render", "video"}.issubset(manifest["groups"]):
        failures.append("current user is not in both render and video groups")
    if not manifest["camera"]["exists"] or not manifest["camera"]["readable"]:
        failures.append(f"camera {manifest['camera']['device']} is unavailable")
    if require_torch:
        torch = manifest["python_packages"]["torch"]
        if not torch.get("cuda_available"):
            failures.append("PyTorch ROCm device is unavailable")
        if not str(torch.get("gcn_arch", "")).startswith("gfx1151"):
            failures.append(f"PyTorch device arch is not gfx1151: {torch.get('gcn_arch')}")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-torch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = collect(args.device)
    failures = evaluate(manifest, args.require_torch)
    manifest["gate"] = {"passed": not failures, "failures": failures}
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
