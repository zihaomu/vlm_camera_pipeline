#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_release="$(uname -r)"
expected_kernel_release="6.17.0-1032-oem"
expected_module_sha256="5a171b50ea6a0bd77c94de8221461dd7c666f5a328dcfe1bd7b1315f044e95f5"
expected_stock_sha256="dc2db574c764210244c2f7617d435d6800499a06e7867db35113ecca9b8515fa"
expected_stock_srcversion="B6E033F201D99AB0B7714F2"
source_module="$repo_dir/third_party/linux-oem-6.17-amd-isp4/drivers/media/platform/amd/isp4/amd_capture.ko"
module_root="/lib/modules/$kernel_release"
target_module="$module_root/updates/vlm-camera-pipeline/amd_capture.ko"
stock_module="$module_root/kernel/drivers/media/platform/amd/isp4/amd_capture.ko.zst"
initrd="/boot/initrd.img-$kernel_release"

run_root() {
  if (( EUID == 0 )); then
    "$@"
  else
    sudo -- "$@"
  fi
}

fail() {
  echo "Persistent amd_capture uninstall refused: $*" >&2
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

restore_patch() {
  echo "Stock reload failed; restoring the validated persistent patch." >&2
  run_root install -D -o root -g root -m 0644 -- "$source_module" "$target_module"
  run_root depmod -a "$kernel_release"
  run_root modprobe amd_capture
  wait_for_module
}

[[ "$kernel_release" == "$expected_kernel_release" ]] || \
  fail "expected kernel $expected_kernel_release, got $kernel_release"
[[ -f "$target_module" ]] || fail "persistent override is not installed"
[[ "$(sha256sum "$target_module" | awk '{print $1}')" == "$expected_module_sha256" ]] || \
  fail "installed override is not the known validated module"
[[ -f "$stock_module" ]] || fail "stock rollback module is missing"
[[ "$(sha256sum "$stock_module" | awk '{print $1}')" == "$expected_stock_sha256" ]] || \
  fail "stock rollback module SHA-256 changed"
if fuser /dev/video0 >/dev/null 2>&1; then
  fail "/dev/video0 is in use"
fi
if [[ -r /sys/module/amd_capture/refcnt && "$(< /sys/module/amd_capture/refcnt)" != "0" ]]; then
  fail "amd_capture has a non-zero reference count"
fi
sudo -n true 2>/dev/null || (( EUID == 0 )) || fail "passwordless sudo is unavailable"

if lsmod | awk '$1 == "amd_capture" { found=1 } END { exit !found }'; then
  run_root rmmod amd_capture
fi
run_root rm -f -- "$target_module"
run_root depmod -a "$kernel_release"
if ! run_root modprobe amd_capture || ! wait_for_module; then
  restore_patch
  fail "stock module did not reload; validated patch was restored"
fi
if [[ "$(< /sys/module/amd_capture/srcversion)" != "$expected_stock_srcversion" ]]; then
  run_root rmmod amd_capture || true
  restore_patch
  fail "stock module srcversion did not match; validated patch was restored"
fi

if [[ -r "$initrd" ]] && lsinitramfs "$initrd" | grep -E '(^|/)amd_capture\.ko(\.(xz|zst|gz))?$' >/dev/null; then
  run_root update-initramfs -u -k "$kernel_release"
fi

ZEROCOPY_AMD_ISP4_REPORT="$repo_dir/output/realtime/amd-isp4-dmabuf-patch-build.json" \
  uv run --frozen --extra vlm --extra migraphx \
    python "$repo_dir/scripts/check_amd_isp4_patch.py" >/dev/null

echo "Persistent override removed: $target_module"
echo "Stock module restored: $stock_module"
echo "Loaded stock srcversion=$expected_stock_srcversion"
