#!/usr/bin/env python3
"""Verify the repo-local AMD ISP4 DMA-BUF patch and built module.

This checker is deliberately read-only with respect to the running kernel.  It
does not install, load, unload, unbind, or exercise the camera device.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "32bed5152a6e284e02c0f803772bc48960f06e32"
PATCH_SHA256 = "dcdb1fb2f3e1c611ab56fc2b8b51ca9e7abd965fbf440b2b128412d25a20b16b"
PATCH_PATH = ROOT / "patches/linux-oem-6.17-amd-isp4-dmabuf-import.patch"
SOURCE_DIR = Path(
    os.environ.get(
        "ZEROCOPY_AMD_ISP4_SOURCE_DIR",
        ROOT / "third_party/linux-oem-6.17-amd-isp4",
    )
).resolve()
KERNEL_RELEASE = os.environ.get("ZEROCOPY_AMD_ISP4_KERNEL_RELEASE", platform.release())
MODULE_DIR = SOURCE_DIR / "drivers/media/platform/amd/isp4"
MODULE_PATH = MODULE_DIR / "amd_capture.ko"
REPORT_PATH = Path(
    os.environ.get(
        "ZEROCOPY_AMD_ISP4_REPORT",
        ROOT / "output/realtime/amd-isp4-dmabuf-patch-build.json",
    )
).resolve()


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=True,
        text=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_field(path: Path, field: str) -> str:
    return run("modinfo", "-F", field, str(path)).stdout.strip()


def find_camera_processes() -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            stat_parts = (proc_dir / "stat").read_text().split()
            state = stat_parts[2]
            wchan = (proc_dir / "wchan").read_text().strip()
            video_fds: list[str] = []
            for fd_path in (proc_dir / "fd").iterdir():
                try:
                    target = os.readlink(fd_path)
                except OSError:
                    continue
                if target.startswith("/dev/video"):
                    video_fds.append(target)
            if "vb2_fop_release" not in wchan and not video_fds:
                continue
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            matches.append(
                {
                    "pid": int(proc_dir.name),
                    "state": state,
                    "wchan": wchan,
                    "video_fds": sorted(set(video_fds)),
                    "command": cmdline,
                }
            )
        except (FileNotFoundError, PermissionError, IndexError, ProcessLookupError):
            continue
    return sorted(matches, key=lambda entry: int(entry["pid"]))


def main() -> int:
    checks: dict[str, bool] = {}
    checks["patch_sha256_locked"] = PATCH_PATH.is_file() and sha256(PATCH_PATH) == PATCH_SHA256
    checks["source_commit_locked"] = (
        run("git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD").stdout.strip()
        == SOURCE_COMMIT
    )

    source_diff = run(
        "git",
        "-C",
        str(SOURCE_DIR),
        "diff",
        "--binary",
        "--",
        "drivers/media/platform/amd/isp4/isp4_interface.c",
        "drivers/media/platform/amd/isp4/isp4_video.c",
    ).stdout.encode()
    source_diff_sha256 = hashlib.sha256(source_diff).hexdigest()
    checks["source_matches_patch"] = source_diff_sha256 == PATCH_SHA256

    source_status = run(
        "git", "-C", str(SOURCE_DIR), "status", "--short", "--untracked-files=no"
    ).stdout.splitlines()
    expected_changes = {
        " M drivers/media/platform/amd/isp4/isp4_interface.c",
        " M drivers/media/platform/amd/isp4/isp4_video.c",
    }
    checks["no_extra_tracked_source_changes"] = set(source_status) == expected_changes
    checks["module_built"] = MODULE_PATH.is_file() and MODULE_PATH.stat().st_size > 0

    module_vermagic = module_field(MODULE_PATH, "vermagic") if MODULE_PATH.is_file() else ""
    module_name = module_field(MODULE_PATH, "name") if MODULE_PATH.is_file() else ""
    built_srcversion = module_field(MODULE_PATH, "srcversion") if MODULE_PATH.is_file() else ""
    checks["module_name"] = module_name == "amd_capture"
    checks["module_matches_running_kernel"] = module_vermagic.startswith(KERNEL_RELEASE + " ")

    installed_path_result = run("modinfo", "-n", "amd_capture", check=False)
    installed_path = (
        Path(installed_path_result.stdout.strip())
        if installed_path_result.returncode == 0 and installed_path_result.stdout.strip()
        else None
    )
    loaded_srcversion_path = Path("/sys/module/amd_capture/srcversion")
    loaded_srcversion = (
        loaded_srcversion_path.read_text().strip() if loaded_srcversion_path.is_file() else None
    )
    installed_srcversion = (
        module_field(installed_path, "srcversion")
        if installed_path is not None and installed_path.is_file()
        else None
    )
    active_module_matches_build = loaded_srcversion == built_srcversion
    persistent_override = (
        Path("/lib/modules")
        / KERNEL_RELEASE
        / "updates/vlm-camera-pipeline/amd_capture.ko"
    )
    persistent_override_installed = (
        persistent_override.is_file()
        and MODULE_PATH.is_file()
        and sha256(persistent_override) == sha256(MODULE_PATH)
    )
    checks["loaded_module_identity_recognized"] = loaded_srcversion in {
        None,
        built_srcversion,
        installed_srcversion,
    }

    camera_processes = find_camera_processes()
    passed = all(checks.values())
    status = "failed"
    if passed:
        if active_module_matches_build and persistent_override_installed:
            status = "patch_built_installed_and_active_runtime_validation_external"
        elif active_module_matches_build:
            status = "patch_built_and_active_runtime_validation_external"
        else:
            status = "patch_built_not_active"
    report = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "passed": passed,
        "runtime_camera_validated": False,
        "system_install_modified": persistent_override_installed,
        "checker_modified_running_module": False,
        "source": {
            "repository": "https://git.launchpad.net/ubuntu/+source/linux-oem-6.17",
            "commit": SOURCE_COMMIT,
            "checkout": str(SOURCE_DIR.relative_to(ROOT)),
            "tracked_changes": source_status,
            "diff_sha256": source_diff_sha256,
        },
        "patch": {
            "path": str(PATCH_PATH.relative_to(ROOT)),
            "sha256": sha256(PATCH_PATH),
            "fixes": [
                "treat foreign dma_buf private data as opaque",
                "import and pin foreign dma_buf through amdgpu PRIME for ISP GART address",
                "avoid CPU vmap and use an opaque completion cookie for GPU dma-bufs",
                "release imported BO on unmap and defensive detach",
                "publish V4L2 and firmware queue nodes before asynchronous completion",
            ],
        },
        "build": {
            "kernel_release": KERNEL_RELEASE,
            "kernel_build_dir": f"/lib/modules/{KERNEL_RELEASE}/build",
            "module": str(MODULE_PATH.relative_to(ROOT)),
            "module_size_bytes": MODULE_PATH.stat().st_size if MODULE_PATH.is_file() else 0,
            "module_sha256": sha256(MODULE_PATH) if MODULE_PATH.is_file() else None,
            "module_name": module_name,
            "srcversion": built_srcversion,
            "vermagic": module_vermagic,
        },
        "running_system": {
            "installed_module": str(installed_path) if installed_path else None,
            "installed_module_sha256": (
                sha256(installed_path) if installed_path and installed_path.is_file() else None
            ),
            "installed_srcversion": installed_srcversion,
            "persistent_override": str(persistent_override),
            "persistent_override_installed": persistent_override_installed,
            "loaded_srcversion": loaded_srcversion,
            "active_module_matches_build": active_module_matches_build,
            "camera_processes": camera_processes,
        },
        "checks": checks,
        "remaining": [
            "runtime camera validation is recorded by the separate camera reports",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
