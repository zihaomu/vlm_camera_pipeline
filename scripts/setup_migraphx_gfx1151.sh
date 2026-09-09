#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wheel_dir="$workspace_dir/.cache/wheels/rocm7.2.1"
model_dir="$workspace_dir/models"
ultralytics_dir="$workspace_dir/third_party/ultralytics"
ultralytics_branch="add-onnx-migraphx-backend"
ultralytics_commit="34e213ca3ece4c18962f5bb922ec74da0c474d24"

export UV_CACHE_DIR="$workspace_dir/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
export YOLO_CONFIG_DIR="$workspace_dir/.cache"
export TORCH_HOME="$workspace_dir/.cache/torch"
export MPLCONFIGDIR="$workspace_dir/.cache/matplotlib"
export HF_HOME="$workspace_dir/.cache/huggingface"
export PYTHONPATH="/opt/rocm/lib${PYTHONPATH:+:$PYTHONPATH}"

for required_command in uv git curl sha256sum rocminfo dpkg-query; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "missing prerequisite command: $required_command" >&2
    exit 2
  fi
done

if ! rocminfo 2>/dev/null | grep -F 'gfx1151' >/dev/null; then
  echo "rocminfo does not report gfx1151; refusing the target-specific runtime" >&2
  exit 3
fi
for required_package in migraphx migraphx-dev half; do
  if ! dpkg-query -W -f='${Status}\n' "$required_package" 2>/dev/null \
    | grep -Fx 'install ok installed' >/dev/null; then
    echo "missing ROCm 7.2.1 native prerequisite: $required_package" >&2
    exit 4
  fi
done

mkdir -p "$wheel_dir" "$model_dir" "$workspace_dir/.cache"

download_verified() {
  local url="$1"
  local output="$2"
  local expected_sha="$3"
  if test -f "$output" && printf '%s  %s\n' "$expected_sha" "$output" \
    | sha256sum -c - >/dev/null 2>&1; then
    printf 'verified existing %s\n' "$output"
    return
  fi
  if test -e "$output"; then
    mv "$output" "$output.invalid-$(date +%Y%m%d-%H%M%S)"
  fi
  curl -fL --retry 5 --continue-at - --output "$output" "$url"
  printf '%s  %s\n' "$expected_sha" "$output" | sha256sum -c -
}

if test ! -d "$ultralytics_dir/.git"; then
  git clone --filter=blob:none --branch "$ultralytics_branch" \
    'https://github.com/zihaomu/ultralytics.git' "$ultralytics_dir"
fi
if test -n "$(git -C "$ultralytics_dir" status --porcelain)"; then
  echo "refusing to change dirty third-party checkout: $ultralytics_dir" >&2
  exit 5
fi
git -C "$ultralytics_dir" fetch origin "$ultralytics_branch"
if test "$(git -C "$ultralytics_dir" rev-parse FETCH_HEAD)" != "$ultralytics_commit"; then
  echo "the pinned Ultralytics branch moved; update the lock before installing" >&2
  exit 6
fi
if git -C "$ultralytics_dir" show-ref --verify --quiet "refs/heads/$ultralytics_branch"; then
  git -C "$ultralytics_dir" switch "$ultralytics_branch"
else
  git -C "$ultralytics_dir" switch --track -c "$ultralytics_branch" \
    "origin/$ultralytics_branch"
fi
test "$(git -C "$ultralytics_dir" rev-parse HEAD)" = "$ultralytics_commit"

download_verified \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
  "$wheel_dir/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
  '663bff4dc3f72582d69f12ad073eb5695dfb526d574376cc8e5b161c7d2f0f08'
download_verified \
  'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.onnx' \
  "$model_dir/yolo26x.onnx" \
  '88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a'

cd "$workspace_dir"
uv sync --frozen --extra migraphx --extra vlm
uv run --frozen --extra migraphx --extra vlm python scripts/check_migraphx_backend.py \
  --model models/yolo26x.onnx \
  --cache-dir models/ort-migraphx-cache/gfx1151-yolo26x \
  --output output/realtime/migraphx-backend-check.json

echo "MIGraphX uv environment is ready: $workspace_dir/.venv"
