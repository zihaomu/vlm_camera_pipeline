#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_dir="$repo_dir/.local/zerocopy-gfx1151"
output_path="${ZEROCOPY_CAMERA_REPORT:-$repo_dir/output/realtime/zerocopy-camera-probe.json}"

"$repo_dir/scripts/setup_camera_dmabuf_gfx1151.sh"
mkdir -p "$(dirname "$output_path")"

exec "$install_dir/bin/camera_dmabuf_probe" \
  --device "${ZEROCOPY_CAMERA_DEVICE:-/dev/video0}" \
  --mode "${ZEROCOPY_CAMERA_MODE:-driver-export}" \
  --width "${ZEROCOPY_CAMERA_WIDTH:-1280}" \
  --height "${ZEROCOPY_CAMERA_HEIGHT:-720}" \
  --buffers "${ZEROCOPY_CAMERA_BUFFERS:-4}" \
  --frames "${ZEROCOPY_CAMERA_FRAMES:-8}" \
  --timeout-ms "${ZEROCOPY_CAMERA_TIMEOUT_MS:-2000}" \
  --output "$output_path"
