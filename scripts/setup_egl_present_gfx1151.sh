#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_dir/native/egl_present"
build_dir="$repo_dir/.build/egl-present-gfx1151"
install_dir="$repo_dir/.local/egl-present-gfx1151"
build_jobs="${BUILD_JOBS:-8}"

for command_name in cmake ninja hipconfig rocminfo roc-obj-ls pkg-config; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing prerequisite command: $command_name" >&2
    exit 2
  fi
done
if test -n "${HSA_OVERRIDE_GFX_VERSION:-}"; then
  echo "HSA_OVERRIDE_GFX_VERSION is set; refusing architecture spoofing" >&2
  exit 3
fi
if ! rocminfo 2>/dev/null | grep -F 'gfx1151' >/dev/null; then
  echo "rocminfo does not report gfx1151" >&2
  exit 4
fi
for package_name in egl epoxy freetype2 fontconfig x11; do
  if ! pkg-config --exists "$package_name"; then
    echo "missing graphics development package: $package_name" >&2
    exit 5
  fi
done
if test -z "${DISPLAY:-}"; then
  echo "DISPLAY is unset; an EGL/X11 or XWayland session is required" >&2
  exit 6
fi

mkdir -p "$build_dir" "$install_dir"
hip_cxx="$(hipconfig -l)/clang"
hip_path="$(hipconfig -R)"
HIPCXX="$hip_cxx" HIP_PATH="$hip_path" cmake \
  -S "$source_dir" \
  -B "$build_dir" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_INSTALL_PREFIX="$install_dir"
cmake --build "$build_dir" --parallel "$build_jobs"
cmake --install "$build_dir"

library="$install_dir/lib/libvlm_camera_egl_present.so.0.1.0"
test -f "$library"
gpu_arches="$(roc-obj-ls "$library" 2>/dev/null | awk '$2 ~ /^hipv/ {print $2}' | sort -u)"
if test "$gpu_arches" != "hipv4-amdgcn-amd-amdhsa--gfx1151"; then
  echo "unexpected HIP code objects in $library: $gpu_arches" >&2
  exit 7
fi
if ldd "$library" | grep -F 'not found' >/dev/null; then
  echo "presenter has unresolved dynamic libraries" >&2
  exit 8
fi

echo "EGL/HIP zero-copy presenter installed at $install_dir"
