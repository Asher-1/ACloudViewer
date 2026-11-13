#!/usr/bin/env bash
# Compatibility launcher; Windows can invoke run_upstream_parity.py directly.
set -euo pipefail
exec "${PYTHON:-python3}" "$(dirname "${BASH_SOURCE[0]}")/run_upstream_parity.py" "$@"
