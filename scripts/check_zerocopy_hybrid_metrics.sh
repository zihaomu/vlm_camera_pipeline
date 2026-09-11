#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test -x "$repo_dir/.venv/bin/python"

cd "$repo_dir"
exec uv run --frozen python scripts/check_zerocopy_hybrid_metrics.py "$@"
