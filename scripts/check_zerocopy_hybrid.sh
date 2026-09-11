#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test -x "$repo_dir/.venv/bin/python"

# shellcheck source=scripts/env_zerocopy_gfx1151.sh
source "$repo_dir/scripts/env_zerocopy_gfx1151.sh"
cd "$repo_dir"
exec uv run --frozen --extra vlm --extra migraphx \
  python scripts/check_zerocopy_hybrid.py "$@"
