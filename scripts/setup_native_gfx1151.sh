#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wheel_dir="$workspace_dir/.cache/wheels/rocm7.2.1"
model_dir="$workspace_dir/models"
third_party_dir="$workspace_dir/third_party"

export UV_CACHE_DIR="$workspace_dir/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
export YOLO_CONFIG_DIR="$workspace_dir/.cache"
export TORCH_HOME="$workspace_dir/.cache/torch"
export MPLCONFIGDIR="$workspace_dir/.cache/matplotlib"
export HF_HOME="$workspace_dir/.cache/huggingface"

for required_command in uv git curl sha256sum rocminfo; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "missing prerequisite command: $required_command" >&2
    exit 2
  fi
done

if ! rocminfo 2>/dev/null | grep -F 'gfx1151' >/dev/null; then
  echo "rocminfo does not report gfx1151; refusing to install the locked runtime" >&2
  exit 3
fi

mkdir -p "$wheel_dir" "$model_dir" "$third_party_dir" "$workspace_dir/.cache"

download_verified() {
  local url="$1"
  local output="$2"
  local expected_sha="$3"
  if test -f "$output" && printf '%s  %s\n' "$expected_sha" "$output" | sha256sum -c - >/dev/null 2>&1; then
    printf 'verified existing %s\n' "$output"
    return
  fi
  if test -e "$output"; then
    mv "$output" "$output.invalid-$(date +%Y%m%d-%H%M%S)"
  fi
  curl -fL --retry 5 --continue-at - --output "$output" "$url"
  printf '%s  %s\n' "$expected_sha" "$output" | sha256sum -c -
}

checkout_locked_repo() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  if test ! -d "$destination/.git"; then
    git clone --filter=blob:none "$url" "$destination"
  fi
  if test -n "$(git -C "$destination" status --porcelain)"; then
    echo "refusing to change dirty third-party checkout: $destination" >&2
    exit 4
  fi
  if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$destination" fetch --filter=blob:none origin "$commit"
  fi
  git -C "$destination" checkout --detach "$commit"
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

download_verified \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  "$wheel_dir/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
  'fb45ace0a27e9f0d0e3c4c6efd8932162743f8376f2aa4752a4d31ef5a1bd3d7'
download_verified \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' \
  "$wheel_dir/torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
  'd5fca8cda173235a3b7434baeebe04c3ebffec3c6fc191e79aa8aa300633f2c9'
download_verified \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl' \
  "$wheel_dir/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl" \
  '07787af1d28c273852f897bfeaa7bca29f2fa4a13ca0f28f535832b240ce7016'
download_verified \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl' \
  "$wheel_dir/torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
  '023d1ce5d847b2a0fbebacf52d35b4c7a233ca07b3dbd0f1cbde84362cbcf33d'

checkout_locked_repo \
  'https://github.com/zihaomu/notebook.git' \
  "$third_party_dir/notebook" \
  'd9d32c37a29272540937d3ee02b3f8f709046464'
checkout_locked_repo \
  'https://github.com/zihaomu/ultralytics.git' \
  "$third_party_dir/ultralytics" \
  '34e213ca3ece4c18962f5bb922ec74da0c474d24'

download_verified \
  'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt' \
  "$model_dir/yolo26x.pt" \
  '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'

cd "$workspace_dir"
uv sync --frozen
uv run --frozen python scripts/check_gfx1151_environment.py \
  --device /dev/video0 --require-torch --output native-lock/environment.json
uv run --frozen python -m pytest -q

echo "gfx1151 uv environment is ready: $workspace_dir/.venv"
