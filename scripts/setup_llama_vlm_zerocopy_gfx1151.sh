#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_dir/third_party/llama.cpp-vlm-zerocopy"
build_dir="$repo_dir/.build/llama-vlm-zerocopy-gfx1151"
patch_file="$repo_dir/patches/llama-cpp-hip-ipc-device-input.patch"
patch_lock="$repo_dir/native-lock/llama-cpp-hip-ipc-device-input.sha256"
model_dir="$repo_dir/models"
source_url="https://github.com/zhangnju/llama.cpp.git"
source_branch="vlm_zerocopy"
source_commit="be84695622f7be5c307d726379ddd745993669b8"
model_repo="unsloth/Qwen3-VL-8B-Instruct-GGUF"
model_revision="b93a7ee713758252c555be4210c00540df954dc2"
model_name="Qwen3-VL-8B-Instruct-Q8_0.gguf"
mmproj_name="mmproj-F16.gguf"
model_sha="cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"
mmproj_sha="d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"
build_jobs="${BUILD_JOBS:-8}"

export UV_CACHE_DIR="$repo_dir/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
export HF_HOME="$repo_dir/.cache/huggingface"
export HF_HUB_CACHE="$repo_dir/.cache/huggingface/hub"

for command_name in uv git cmake ninja rocminfo hipconfig roc-obj-ls sha256sum strings; do
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
if ! [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUILD_JOBS must be a positive integer" >&2
  exit 5
fi
hip_path="$(hipconfig -R)"
rocwmma_header="$hip_path/include/rocwmma/rocwmma.hpp"
if ! test -r "$rocwmma_header"; then
  echo "rocWMMA header is required for the gfx1151 FlashAttention build: $rocwmma_header" >&2
  exit 11
fi

expected_patch_sha="$(tr -d '[:space:]' < "$patch_lock")"
printf '%s  %s\n' "$expected_patch_sha" "$patch_file" | sha256sum -c -
mkdir -p "$repo_dir/third_party" "$build_dir" "$model_dir" "$repo_dir/.cache/huggingface"

if test ! -d "$source_dir/.git"; then
  git clone --filter=blob:none --branch "$source_branch" "$source_url" "$source_dir"
fi
if test -n "$(git -C "$source_dir" status --porcelain)"; then
  if test "$(git -C "$source_dir" rev-parse HEAD)" != "$source_commit" \
    || test -n "$(git -C "$source_dir" ls-files --others --exclude-standard)" \
    || ! git -C "$source_dir" apply --reverse --check "$patch_file"; then
    echo "refusing an unrecognized dirty llama.cpp checkout: $source_dir" >&2
    exit 6
  fi
  actual_patch_sha="$(git -C "$source_dir" diff --binary | sha256sum | cut -d' ' -f1)"
  if test "$actual_patch_sha" != "$expected_patch_sha"; then
    echo "the applied llama.cpp patch differs from its lock" >&2
    exit 7
  fi
else
  if ! git -C "$source_dir" cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch --filter=blob:none origin "$source_commit"
  fi
  git -C "$source_dir" checkout --detach "$source_commit"
  git -C "$source_dir" apply "$patch_file"
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$source_commit"
if rg -n 'hipMemcpyDeviceToHost|hipDeviceSynchronize' \
  "$source_dir/tools/server/server-ipc.cpp" \
  "$source_dir/tools/mtmd/poc/ipc_client_poc.py" \
  "$source_dir/tools/mtmd/poc/ipc_preproc_client.py" >/dev/null; then
  # The legacy POCs are retained upstream as negative evidence; only the production server
  # source is required to be clean. Check it independently before continuing.
  if rg -n 'hipMemcpyDeviceToHost|hipDeviceSynchronize' \
    "$source_dir/tools/server/server-ipc.cpp" >/dev/null; then
    echo "forbidden host copy/global synchronize remains in production server IPC" >&2
    exit 8
  fi
fi

hip_cxx="$(hipconfig -l)/clang"
HIPCXX="$hip_cxx" HIP_PATH="$hip_path" cmake \
  -S "$source_dir" \
  -B "$build_dir" \
  -G Ninja \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGPU_TARGETS=gfx1151 \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release
grep -Fx 'GGML_HIP_ROCWMMA_FATTN:BOOL=ON' "$build_dir/CMakeCache.txt" >/dev/null
cmake --build "$build_dir" --parallel "$build_jobs" --target llama-server

hip_library="$(find "$build_dir/bin" -maxdepth 1 -type f -name 'libggml-hip.so.*.*.*' -print -quit)"
test -n "$hip_library"
gpu_arches="$(roc-obj-ls "$hip_library" 2>/dev/null | awk '$2 ~ /^hipv/ {print $2}' | sort -u)"
if test "$gpu_arches" != "hipv4-amdgcn-amd-amdhsa--gfx1151"; then
  echo "unexpected HIP code objects in $hip_library: $gpu_arches" >&2
  exit 9
fi
strings "$build_dir/bin/libllama-server-impl.so" | grep -F 'vlm-hip-ipc-v2' >/dev/null
strings "$build_dir/bin/libmtmd.so" | grep -F 'HIP IPC v2 D2D complete' >/dev/null
strings "$build_dir/bin/libmtmd.so" | grep -F 'HIP device embedding ready' >/dev/null
strings "$build_dir/bin/libllama.so" | grep -F 'HIP embedding D2D complete' >/dev/null
if ! "$build_dir/bin/llama-server" --list-devices | grep -F 'ROCm0:' >/dev/null; then
  echo "patched llama.cpp did not expose ROCm0" >&2
  exit 10
fi

cd "$repo_dir"
uv sync --frozen --extra vlm --extra migraphx
uv run --frozen --extra vlm --extra migraphx hf download "$model_repo" \
  "$model_name" "$mmproj_name" --revision "$model_revision" --local-dir "$model_dir"
printf '%s  %s\n' "$model_sha" "$model_dir/$model_name" | sha256sum -c -
printf '%s  %s\n' "$mmproj_sha" "$model_dir/$mmproj_name" | sha256sum -c -
bash "$repo_dir/scripts/setup_zerocopy_kernels_gfx1151.sh"

echo "Qwen3-VL HIP IPC v2 runtime is ready in $build_dir/bin"
