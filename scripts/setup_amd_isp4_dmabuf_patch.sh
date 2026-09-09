#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${AMD_ISP4_SOURCE_DIR:-$repo_dir/third_party/linux-oem-6.17-amd-isp4}"
source_url="https://git.launchpad.net/ubuntu/+source/linux-oem-6.17"
source_commit="32bed5152a6e284e02c0f803772bc48960f06e32"
patch_path="$repo_dir/patches/linux-oem-6.17-amd-isp4-dmabuf-import.patch"
patch_sha256="dcdb1fb2f3e1c611ab56fc2b8b51ca9e7abd965fbf440b2b128412d25a20b16b"
kernel_release="${AMD_ISP4_KERNEL_RELEASE:-$(uname -r)}"
kernel_build_dir="${AMD_ISP4_KERNEL_BUILD_DIR:-/lib/modules/$kernel_release/build}"
module_source_dir="$source_dir/drivers/media/platform/amd/isp4"

if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
  echo "Missing repo-local uv environment: run 'uv sync --frozen --extra vlm --extra migraphx'" >&2
  exit 1
fi
if [[ ! -d "$kernel_build_dir" ]]; then
  echo "Missing kernel headers at $kernel_build_dir" >&2
  exit 1
fi

actual_patch_sha256="$(sha256sum "$patch_path" | cut -d' ' -f1)"
if [[ "$actual_patch_sha256" != "$patch_sha256" ]]; then
  echo "Patch SHA-256 mismatch: expected $patch_sha256, got $actual_patch_sha256" >&2
  exit 1
fi

if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p "$(dirname "$source_dir")"
  git clone --filter=blob:none --no-checkout "$source_url" "$source_dir"
  git -C "$source_dir" sparse-checkout init --no-cone
  git -C "$source_dir" sparse-checkout set \
    /drivers/media/platform/amd/isp4/ \
    /drivers/media/common/videobuf2/ \
    /drivers/media/v4l2-core/ \
    /drivers/gpu/drm/amd/amdgpu/ \
    /include/drm/amd/isp.h
  git -C "$source_dir" checkout --detach "$source_commit"
fi

actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$source_commit" ]]; then
  echo "ISP4 source mismatch: expected $source_commit, got $actual_commit" >&2
  exit 1
fi

if git -C "$source_dir" apply --reverse --check "$patch_path" 2>/dev/null; then
  actual_diff_sha256="$(
    git -C "$source_dir" diff --binary -- \
      drivers/media/platform/amd/isp4/isp4_interface.c \
      drivers/media/platform/amd/isp4/isp4_video.c | sha256sum | cut -d' ' -f1
  )"
  if [[ "$actual_diff_sha256" != "$patch_sha256" ]]; then
    echo "Source contains edits beyond or different from the locked ISP4 patch" >&2
    exit 1
  fi
elif git -C "$source_dir" diff --quiet --exit-code; then
  git -C "$source_dir" apply --check "$patch_path"
  git -C "$source_dir" apply "$patch_path"
else
  echo "ISP4 source is dirty and does not exactly match the locked patch" >&2
  exit 1
fi

make -C "$kernel_build_dir" M="$module_source_dir" clean
make -C "$kernel_build_dir" M="$module_source_dir" modules

ZEROCOPY_AMD_ISP4_SOURCE_DIR="$source_dir" \
  ZEROCOPY_AMD_ISP4_KERNEL_RELEASE="$kernel_release" \
  uv run --frozen --extra vlm --extra migraphx \
    python "$repo_dir/scripts/check_amd_isp4_patch.py"

echo "Built only: $module_source_dir/amd_capture.ko"
echo "This script did not install or reload the module; the checker reports the current kernel state."
