#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_dir/native/camera_dmabuf"
build_dir="$repo_dir/.build/camera-dmabuf-gfx1151"
install_dir="$repo_dir/.local/zerocopy-gfx1151"

command -v cmake >/dev/null
command -v ninja >/dev/null
command -v uv >/dev/null
test -x "$repo_dir/.venv/bin/python"
test -d /opt/rocm

mkdir -p "$build_dir" "$install_dir"
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_INSTALL_PREFIX="$install_dir"
cmake --build "$build_dir" --parallel
cmake --install "$build_dir"

roc-obj-ls "$install_dir/lib/libvlm_camera_gpu_capture.so" \
  | grep -F 'hipv4-amdgcn-amd-amdhsa--gfx1151' >/dev/null

# shellcheck source=scripts/env_zerocopy_gfx1151.sh
source "$repo_dir/scripts/env_zerocopy_gfx1151.sh"
uv run --frozen --extra vlm --extra migraphx \
  python "$repo_dir/scripts/check_zerocopy_camera_component.py"

printf 'GPU camera component installed repo-locally at %s\n' "$install_dir"
printf 'No V4L2 device was opened; Z0 runtime validation remains a separate step.\n'
