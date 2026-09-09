#!/usr/bin/env python3
"""Read-only verification of the persistent AMD ISP4 module override."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import stat
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KERNEL_RELEASE = "6.17.0-1032-oem"
EXPECTED_MODULE_SHA256 = "5a171b50ea6a0bd77c94de8221461dd7c666f5a328dcfe1bd7b1315f044e95f5"
EXPECTED_MODULE_SRCVERSION = "CA94CB23673E748A9ECC5F2"
EXPECTED_STOCK_SHA256 = "dc2db574c764210244c2f7617d435d6800499a06e7867db35113ecca9b8515fa"
MODULE_RELATIVE = Path("updates/vlm-camera-pipeline/amd_capture.ko")
STOCK_RELATIVE = Path("kernel/drivers/media/platform/amd/isp4/amd_capture.ko.zst")
SOURCE_MODULE = (
    ROOT
    / "third_party/linux-oem-6.17-amd-isp4/drivers/media/platform/amd/isp4/amd_capture.ko"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/realtime/amd-isp4-persistent-install.json",
    )
    parser.add_argument(
        "--reload-performed",
        action="store_true",
        help="record that the installer cold-reloaded amd_capture through modprobe",
    )
    return parser.parse_args()


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_field(path: Path, field: str) -> str:
    result = run("modinfo", "-F", field, str(path), check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def initramfs_entries(kernel_release: str) -> list[str]:
    initrd = Path(f"/boot/initrd.img-{kernel_release}")
    if not initrd.is_file():
        return []
    result = run("lsinitramfs", str(initrd), check=False)
    if result.returncode != 0:
        return ["<lsinitramfs-failed>"]
    return sorted(
        line
        for line in result.stdout.splitlines()
        if Path(line).name.startswith("amd_capture.ko")
    )


def main() -> int:
    args = parse_args()
    kernel_release = platform.release()
    module_root = Path("/lib/modules") / kernel_release
    override = module_root / MODULE_RELATIVE
    stock = module_root / STOCK_RELATIVE
    resolved_result = run("modinfo", "-n", "amd_capture", check=False)
    resolved = (
        Path(resolved_result.stdout.strip())
        if resolved_result.returncode == 0 and resolved_result.stdout.strip()
        else None
    )
    show_depends = run("modprobe", "--show-depends", "amd_capture", check=False)
    loaded_srcversion_path = Path("/sys/module/amd_capture/srcversion")
    loaded_srcversion = (
        loaded_srcversion_path.read_text().strip()
        if loaded_srcversion_path.is_file()
        else None
    )
    module_refcount_path = Path("/sys/module/amd_capture/refcnt")
    module_refcount = (
        int(module_refcount_path.read_text().strip())
        if module_refcount_path.is_file()
        else None
    )
    camera_users = run("fuser", "/dev/video0", check=False)
    entries = initramfs_entries(kernel_release)
    depmod_config = Path("/etc/depmod.d/ubuntu.conf")
    depmod_text = depmod_config.read_text() if depmod_config.is_file() else ""
    override_stat = override.stat() if override.is_file() else None

    checks = {
        "running_expected_kernel": kernel_release == EXPECTED_KERNEL_RELEASE,
        "source_module_sha256_locked": SOURCE_MODULE.is_file()
        and sha256(SOURCE_MODULE) == EXPECTED_MODULE_SHA256,
        "persistent_override_installed": override.is_file(),
        "persistent_override_sha256_locked": override.is_file()
        and sha256(override) == EXPECTED_MODULE_SHA256,
        "persistent_override_srcversion_locked": module_field(override, "srcversion")
        == EXPECTED_MODULE_SRCVERSION,
        "persistent_override_owned_by_root": bool(
            override_stat and override_stat.st_uid == 0 and override_stat.st_gid == 0
        ),
        "persistent_override_mode_0644": bool(
            override_stat and stat.S_IMODE(override_stat.st_mode) == 0o644
        ),
        "stock_module_preserved": stock.is_file(),
        "stock_module_sha256_unchanged": stock.is_file()
        and sha256(stock) == EXPECTED_STOCK_SHA256,
        "depmod_updates_priority_configured": any(
            line.strip().startswith("search updates ") for line in depmod_text.splitlines()
        ),
        "modinfo_resolves_persistent_override": bool(
            resolved and resolved.resolve() == override.resolve()
        ),
        "modprobe_resolves_persistent_override": (
            show_depends.returncode == 0 and str(override) in show_depends.stdout
        ),
        "loaded_module_matches_persistent_override": loaded_srcversion
        == EXPECTED_MODULE_SRCVERSION,
        "camera_device_present_after_reload": Path("/dev/video0").exists(),
        "camera_not_in_use": camera_users.returncode != 0,
        "initramfs_has_no_stale_stock_module": entries != ["<lsinitramfs-failed>"]
        and all("updates/vlm-camera-pipeline/amd_capture.ko" in entry for entry in entries),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "scope": "persistent across reboot for the exact running kernel",
        "kernel_release": kernel_release,
        "kernel_upgrade_policy": (
            "fail closed after a kernel change until the locked patch is rebuilt and installed "
            "for that kernel"
        ),
        "installation": {
            "source_module": str(SOURCE_MODULE.relative_to(ROOT)),
            "source_sha256": sha256(SOURCE_MODULE) if SOURCE_MODULE.is_file() else None,
            "persistent_override": str(override),
            "persistent_override_sha256": sha256(override) if override.is_file() else None,
            "module_resolution": str(resolved) if resolved else None,
            "module_srcversion": module_field(override, "srcversion"),
            "module_vermagic": module_field(override, "vermagic"),
            "cold_reload_performed_by_installer": args.reload_performed,
            "loaded_srcversion": loaded_srcversion,
            "loaded_refcount": module_refcount,
            "depmod_database_updated": checks[
                "modinfo_resolves_persistent_override"
            ]
            and checks["modprobe_resolves_persistent_override"],
            "initramfs_entries": entries,
            "initramfs_update_required": bool(entries),
        },
        "rollback": {
            "stock_module": str(stock),
            "stock_sha256": sha256(stock) if stock.is_file() else None,
            "stock_srcversion": module_field(stock, "srcversion"),
            "script": "scripts/uninstall_amd_isp4_dmabuf_patch.sh",
        },
        "system": {
            "secure_boot": run("mokutil", "--sb-state", check=False).stdout.strip(),
            "kernel_tainted": int(Path("/proc/sys/kernel/tainted").read_text().strip()),
            "camera_users": camera_users.stdout.split(),
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
