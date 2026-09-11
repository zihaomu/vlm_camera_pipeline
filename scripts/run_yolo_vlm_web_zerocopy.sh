#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test -x "$repo_dir/.venv/bin/python"

# shellcheck source=scripts/env_zerocopy_gfx1151.sh
source "$repo_dir/scripts/env_zerocopy_gfx1151.sh"
cd "$repo_dir"
exec uv run --frozen --extra vlm --extra migraphx --extra web \
  python scripts/run_yolo_vlm_zerocopy.py \
  --presenter web \
  --camera-horizontal-flip \
  --vlm-input-mode hybrid \
  --vlm-interval 3.0 \
  --hybrid-max-hints 8 \
  --hybrid-arm-timeout-ms 100 \
  --zero-copy require \
  --metrics output/realtime/metrics-yolo-vlm-web-zerocopy.jsonl \
  --vlm-log output/realtime/llama-server-web-zerocopy.log \
  --open-browser \
  "$@"
