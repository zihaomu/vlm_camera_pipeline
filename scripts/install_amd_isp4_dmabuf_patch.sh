#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_release="$(uname -r)"
expected_kernel_release="6.17.0-1032-oem"
expected_module_sha256="5a171b50ea6a0bd77c94de8221461dd7c666f5a328dcfe1bd7b1315f044e95f5"
expected_module_srcversion="CA94CB23673E748A9ECC5F2"
expected_stock_sha256="dc2db574c764210244c2f7617d435d6800499a06e7867db35113ecca9b8515fa"
source_module="$repo_dir/third_party/linux-oem-6.17-amd-isp4/drivers/media/platform/amd/isp4/amd_capture.ko"
module_root="/lib/modules/$kernel_release"
target_dir="$module_root/updates/vlm-camera-pipeline"
target_module="$target_dir/amd_capture.ko"
stock_module="$module_root/kernel/drivers/media/platform/amd/isp4/amd_capture.ko.zst"
initrd="/boot/initrd.img-$kernel_release"
initramfs_updated=false

run_root() {
  if (( EUID == 0 )); then
    "$@"
  else
    sudo -- "$@"
  fi
}

fail() {
  echo "Persistent amd_capture install refused: $*" >&2
  exit 1
}

wait_for_module() {
  local attempt
  for attempt in {1..50}; do
    if [[ -r /sys/module/amd_capture/srcversion && -e /dev/video0 ]]; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

restore_stock() {
  echo "Install validation failed; restoring stock module resolution." >&2
  if lsmod | awk '$1 == "amd_capture" { found=1 } END { exit !found }'; then
    run_root rmmod amd_capture || true
  fi
  run_root rm -f -- "$target_module"
  run_root depmod -a "$kernel_release"
  if [[ "$initramfs_updated" == true ]]; then
    run_root update-initramfs -u -k "$kernel_release"
  fi
  run_root modprobe amd_capture
  wait_for_module
}

[[ "$kernel_release" == "$expected_kernel_release" ]] || \
  fail "expected kernel $expected_kernel_release, got $kernel_release"
[[ -f "$source_module" ]] || fail "missing repo-local module $source_module"
[[ -f "$stock_module" ]] || fail "missing stock rollback module $stock_module"
[[ "$(sha256sum "$source_module" | awk '{print $1}')" == "$expected_module_sha256" ]] || \
  fail "repo-local module SHA-256 does not match the validated build"
[[ "$(modinfo -F srcversion "$source_module")" == "$expected_module_srcversion" ]] || \
  fail "repo-local module srcversion mismatch"
[[ "$(modinfo -F vermagic "$source_module")" == "$kernel_release "* ]] || \
  fail "repo-local module vermagic does not match the running kernel"
[[ "$(sha256sum "$stock_module" | awk '{print $1}')" == "$expected_stock_sha256" ]] || \
  fail "stock rollback module changed; refusing to install"
rg -q '^[[:space:]]*search[[:space:]]+updates([[:space:]]|$)' /etc/depmod.d/ubuntu.conf || \
  fail "depmod does not prioritize the updates directory"
[[ -x "$repo_dir/.venv/bin/python" ]] || fail "repo-local uv environment is missing"
if fuser /dev/video0 >/dev/null 2>&1; then
  fail "/dev/video0 is in use"
fi
if [[ -r /sys/module/amd_capture/refcnt && "$(< /sys/module/amd_capture/refcnt)" != "0" ]]; then
  fail "amd_capture has a non-zero reference count"
fi
if [[ -e "$target_module" && "$(sha256sum "$target_module" | awk '{print $1}')" != "$expected_module_sha256" ]]; then
  fail "an unknown module already exists at $target_module"
fi
sudo -n true 2>/dev/null || (( EUID == 0 )) || fail "passwordless sudo is unavailable"

run_root install -D -o root -g root -m 0644 -- "$source_module" "$target_module"
run_root depmod -a "$kernel_release"

resolved_module="$(modinfo -n amd_capture)"
if [[ "$resolved_module" != "$target_module" ]]; then
  restore_stock
  fail "modinfo resolved $resolved_module instead of $target_module"
fi

if [[ -r "$initrd" ]] && lsinitramfs "$initrd" | grep -E '(^|/)amd_capture\.ko(\.(xz|zst|gz))?$' >/dev/null; then
  run_root update-initramfs -u -k "$kernel_release"
  initramfs_updated=true
fi

if lsmod | awk '$1 == "amd_capture" { found=1 } END { exit !found }'; then
  if ! run_root rmmod amd_capture; then
    run_root rm -f -- "$target_module"
    run_root depmod -a "$kernel_release"
    fail "could not unload the active module; persistent override was rolled back"
  fi
fi
if ! run_root modprobe amd_capture || ! wait_for_module; then
  restore_stock
  fail "patched module did not cold-load cleanly; stock module was restored"
fi

loaded_srcversion="$(< /sys/module/amd_capture/srcversion)"
if [[ "$loaded_srcversion" != "$expected_module_srcversion" ]]; then
  restore_stock
  fail "loaded srcversion $loaded_srcversion does not match $expected_module_srcversion"
fi

ZEROCOPY_AMD_ISP4_REPORT="$repo_dir/output/realtime/amd-isp4-dmabuf-patch-build.json" \
  uv run --frozen --extra vlm --extra migraphx \
    python "$repo_dir/scripts/check_amd_isp4_patch.py" >/dev/null
uv run --frozen python "$repo_dir/scripts/check_amd_isp4_persistent_install.py" \
  --reload-performed

echo "Persistent module installed: $target_module"
echo "Stock module preserved: $stock_module"
echo "Cold modprobe reload passed with srcversion=$loaded_srcversion"
echo "initramfs_updated=$initramfs_updated"
