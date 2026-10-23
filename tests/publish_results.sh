#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec uv run --no-project --python 3.12 python \
    "$REPO_ROOT/tests/test_utils/python_scripts/publish_test_results.py" "$@"
