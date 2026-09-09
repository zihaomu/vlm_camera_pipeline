#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_dir/native/zerocopy_kernels"
build_dir="$repo_dir/.build/zerocopy-kernels-gfx1151"
install_dir="$repo_dir/.local/zerocopy-gfx1151"

command -v cmake >/dev/null
command -v ninja >/dev/null
test -d /opt/rocm

mkdir -p "$build_dir" "$install_dir"
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_INSTALL_PREFIX="$install_dir"
cmake --build "$build_dir" --parallel
cmake --install "$build_dir"

roc-obj-ls "$install_dir/lib/libvlm_camera_zerocopy_kernels.so" \
  | grep -F 'hipv4-amdgcn-amd-amdhsa--gfx1151' >/dev/null
printf 'zero-copy HIP kernels installed at %s\n' "$install_dir"
