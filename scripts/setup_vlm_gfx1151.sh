#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="$workspace_dir/third_party/llama.cpp"
build_dir="$llama_dir/build-gfx1151"
model_dir="$workspace_dir/models"
llama_commit="0b1bad14ff204627636aeb1de22ddcd5acb859d4"
model_repo="unsloth/Qwen3-VL-8B-Instruct-GGUF"
model_revision="b93a7ee713758252c555be4210c00540df954dc2"
model_name="Qwen3-VL-8B-Instruct-Q8_0.gguf"
mmproj_name="mmproj-F16.gguf"
model_sha="cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"
mmproj_sha="d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"
build_jobs="${BUILD_JOBS:-8}"

export UV_CACHE_DIR="$workspace_dir/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
export HF_HOME="$workspace_dir/.cache/huggingface"
export HF_HUB_CACHE="$workspace_dir/.cache/huggingface/hub"

for required_command in uv git cmake ninja rocminfo hipconfig roc-obj-ls sha256sum; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "missing prerequisite command: $required_command" >&2
    exit 2
  fi
done

if test -n "${HSA_OVERRIDE_GFX_VERSION:-}"; then
  echo "HSA_OVERRIDE_GFX_VERSION is set; refusing an architecture-spoofed build" >&2
  exit 3
fi
if ! rocminfo 2>/dev/null | grep -F 'gfx1151' >/dev/null; then
  echo "rocminfo does not report gfx1151; refusing to build the locked VLM runtime" >&2
  exit 4
fi
if ! [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUILD_JOBS must be a positive integer" >&2
  exit 5
fi

mkdir -p "$workspace_dir/third_party" "$model_dir" "$workspace_dir/.cache/huggingface"

if test ! -e "$llama_dir"; then
  git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir"
elif test ! -d "$llama_dir/.git"; then
  echo "existing path is not a Git checkout: $llama_dir" >&2
  exit 6
fi
if test -n "$(git -C "$llama_dir" status --porcelain)"; then
  echo "refusing to change dirty third-party checkout: $llama_dir" >&2
  exit 7
fi
if ! git -C "$llama_dir" cat-file -e "${llama_commit}^{commit}" 2>/dev/null; then
  git -C "$llama_dir" fetch --filter=blob:none origin "$llama_commit"
fi
git -C "$llama_dir" checkout --detach "$llama_commit"
test "$(git -C "$llama_dir" rev-parse HEAD)" = "$llama_commit"

hip_cxx="$(hipconfig -l)/clang"
hip_path="$(hipconfig -R)"
HIPCXX="$hip_cxx" HIP_PATH="$hip_path" cmake \
  -S "$llama_dir" \
  -B "$build_dir" \
  -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$build_dir/bin"
cmake -E remove_directory "$build_dir/tools/ui/dist"
cmake -E rm -f "$build_dir/tools/ui/.ui-stamp"
cmake --build "$build_dir" --parallel "$build_jobs" \
  --target llama-server llama-mtmd-cli

hip_library="$build_dir/bin/libggml-hip.so.0.19.0"
test -f "$hip_library"
gpu_arches="$(roc-obj-ls "$hip_library" 2>/dev/null | awk '$2 ~ /^hipv/ {print $2}' | sort -u)"
if test "$gpu_arches" != "hipv4-amdgcn-amd-amdhsa--gfx1151"; then
  echo "unexpected HIP code objects in $hip_library: $gpu_arches" >&2
  exit 8
fi
if ! "$build_dir/bin/llama-server" --list-devices | grep -F 'ROCm0:' >/dev/null; then
  echo "llama.cpp did not expose the required ROCm0 device" >&2
  exit 9
fi

cd "$workspace_dir"
uv sync --frozen --extra vlm
uv run --frozen --extra vlm hf download "$model_repo" \
  "$model_name" "$mmproj_name" --revision "$model_revision" --local-dir "$model_dir"
printf '%s  %s\n' "$model_sha" "$model_dir/$model_name" | sha256sum -c -
printf '%s  %s\n' "$mmproj_sha" "$model_dir/$mmproj_name" | sha256sum -c -

echo "Qwen3-VL gfx1151 runtime is ready in $build_dir/bin"
