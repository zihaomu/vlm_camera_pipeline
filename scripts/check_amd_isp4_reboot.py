#!/usr/bin/env python3
"""Capture and verify a controlled reboot for the persistent AMD ISP4 override."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KERNEL_RELEASE = "6.17.0-1032-oem"
EXPECTED_MODULE_SHA256 = "5a171b50ea6a0bd77c94de8221461dd7c666f5a328dcfe1bd7b1315f044e95f5"
EXPECTED_MODULE_SRCVERSION = "CA94CB23673E748A9ECC5F2"
EXPECTED_STOCK_SHA256 = "dc2db574c764210244c2f7617d435d6800499a06e7867db35113ecca9b8515fa"
PRE_REPORT = ROOT / "output/realtime/amd-isp4-controlled-reboot-before.json"
POST_REPORT = ROOT / "output/realtime/amd-isp4-controlled-reboot-after.json"
POST_PROBE_REPORT = ROOT / "output/realtime/amd-isp4-controlled-reboot-probe.json"
POST_RUNTIME_REPORT = ROOT / "output/realtime/amd-isp4-controlled-reboot-runtime.json"
POST_INTEGRATION_METRICS = (
    ROOT / "output/realtime/amd-isp4-controlled-reboot-integration.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("before", "after"), required=True)
    return parser.parse_args()


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_summary(path: Path) -> dict[str, Any]:
    final: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("type") == "final_summary":
                final = record
    if final is None:
        raise ValueError(f"no final_summary in {path}")
    return final


def snapshot() -> dict[str, Any]:
    kernel_release = platform.release()
    module_root = Path("/lib/modules") / kernel_release
    override = module_root / "updates/vlm-camera-pipeline/amd_capture.ko"
    stock = module_root / "kernel/drivers/media/platform/amd/isp4/amd_capture.ko.zst"
    resolved_result = run("modinfo", "-n", "amd_capture", check=False)
    resolved = (
        Path(resolved_result.stdout.strip())
        if resolved_result.returncode == 0 and resolved_result.stdout.strip()
        else None
    )
    loaded_srcversion_path = Path("/sys/module/amd_capture/srcversion")
    loaded_srcversion = (
        loaded_srcversion_path.read_text().strip()
        if loaded_srcversion_path.is_file()
        else None
    )
    refcount_path = Path("/sys/module/amd_capture/refcnt")
    refcount = int(refcount_path.read_text().strip()) if refcount_path.is_file() else None
    camera_users = run("fuser", "/dev/video0", check=False)
    return {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0]),
        "kernel_release": kernel_release,
        "system_state": run("systemctl", "is-system-running", check=False).stdout.strip(),
        "module_resolution": str(resolved) if resolved else None,
        "persistent_override": str(override),
        "persistent_override_sha256": sha256(override) if override.is_file() else None,
        "stock_module": str(stock),
        "stock_module_sha256": sha256(stock) if stock.is_file() else None,
        "loaded_srcversion": loaded_srcversion,
        "module_refcount": refcount,
        "camera_device_present": Path("/dev/video0").exists(),
        "camera_users": camera_users.stdout.split(),
        "module_resolution_matches_override": bool(
            resolved and override.is_file() and resolved.resolve() == override.resolve()
        ),
    }


def kernel_faults() -> list[str]:
    journal = run("journalctl", "-b", "-k", "--no-pager", check=False)
    if journal.returncode != 0:
        return ["<kernel-journal-unavailable>"]
    pattern = re.compile(
        r"(?:amdgpu.*\b(?:reset|fault|hang|timeout)\b|"
        r"amd[_ -]?isp.*\b(?:error|failed|failure|timeout|stall)\b|"
        r"amd_capture.*\b(?:error|failed|failure|timeout|stall)\b)",
        re.IGNORECASE,
    )
    expected_boot_messages = (
        "module verification failed: signature and/or required key missing",
    )
    return [
        line
        for line in journal.stdout.splitlines()
        if pattern.search(line)
        and not any(message in line.lower() for message in expected_boot_messages)
    ]


def main() -> int:
    args = parse_args()
    current = snapshot()
    before = None
    faults: list[str] = []
    probe: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    integration: dict[str, Any] | None = None
    if args.stage == "after":
        if not PRE_REPORT.is_file():
            raise FileNotFoundError(f"missing pre-reboot report: {PRE_REPORT}")
        before = json.loads(PRE_REPORT.read_text(encoding="utf-8"))
        faults = kernel_faults()
        probe = load_json(POST_PROBE_REPORT) if POST_PROBE_REPORT.is_file() else None
        runtime = load_json(POST_RUNTIME_REPORT) if POST_RUNTIME_REPORT.is_file() else None
        integration = (
            final_summary(POST_INTEGRATION_METRICS)
            if POST_INTEGRATION_METRICS.is_file()
            else None
        )

    checks = {
        "expected_kernel": current["kernel_release"] == EXPECTED_KERNEL_RELEASE,
        "module_resolution_matches_override": current[
            "module_resolution_matches_override"
        ],
        "persistent_override_sha256_locked": current["persistent_override_sha256"]
        == EXPECTED_MODULE_SHA256,
        "stock_module_sha256_unchanged": current["stock_module_sha256"]
        == EXPECTED_STOCK_SHA256,
        "loaded_srcversion_locked": current["loaded_srcversion"]
        == EXPECTED_MODULE_SRCVERSION,
        "module_unused": current["module_refcount"] == 0,
        "camera_device_present": current["camera_device_present"] is True,
        "camera_not_in_use": current["camera_users"] == [],
    }
    if args.stage == "after":
        checks.update(
            {
                "boot_id_changed": current["boot_id"] != before["snapshot"]["boot_id"],
                "boot_uptime_reset": current["uptime_seconds"]
                < before["snapshot"]["uptime_seconds"],
                "boot_kernel_faults_zero": faults == [],
                "post_reboot_gpu_export_coherency_passed": bool(
                    probe
                    and probe.get("passed") is True
                    and probe.get("frames_sampled") == 8
                    and probe.get("frames_changed_from_sentinel") == 8
                    and all(
                        buffer.get("export_offset") == 0
                        for buffer in probe.get("buffers", [])
                    )
                ),
                "post_reboot_camera_runtime_passed": bool(
                    runtime
                    and runtime.get("passed") is True
                    and runtime.get("checks")
                    and all(runtime["checks"].values())
                ),
                "post_reboot_full_integration_passed": bool(
                    integration
                    and integration.get("status") == "completed"
                    and integration.get("acceptance", {}).get("short_runtime_passed")
                    is True
                    and integration.get("camera", {}).get(
                        "persistent_module_installed"
                    )
                    is True
                ),
            }
        )

    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "stage": args.stage,
        "status": "passed" if passed else "failed",
        "snapshot": current,
        "before_report": str(PRE_REPORT.relative_to(ROOT)) if before else None,
        "before_boot_id": before["snapshot"]["boot_id"] if before else None,
        "kernel_faults": faults,
        "post_reboot_validation": {
            "gpu_export_probe": str(POST_PROBE_REPORT.relative_to(ROOT)),
            "gpu_export_frames": probe.get("frames_sampled") if probe else None,
            "camera_runtime": str(POST_RUNTIME_REPORT.relative_to(ROOT)),
            "camera_runtime_frames": runtime.get("performance", {}).get("frames")
            if runtime
            else None,
            "camera_runtime_effective_fps": runtime.get("performance", {}).get(
                "effective_fps"
            )
            if runtime
            else None,
            "integration_metrics": str(POST_INTEGRATION_METRICS.relative_to(ROOT)),
            "integration_frames": integration.get("frames") if integration else None,
            "integration_effective_fps": integration.get("effective_fps")
            if integration
            else None,
            "integration_yolo_completion_fps": integration.get("yolo_ms", {}).get(
                "completion_fps"
            )
            if integration
            else None,
            "integration_vlm_completed": integration.get("worker", {}).get(
                "completed"
            )
            if integration
            else None,
        }
        if args.stage == "after"
        else None,
        "checks": checks,
    }
    output = PRE_REPORT if args.stage == "before" else POST_REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
