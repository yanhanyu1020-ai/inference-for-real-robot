#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${COLLECTION_CONFIG:-${ROOT}/config/collection.json}"
exec "${COLLECTION_PYTHON:-python3}" "${ROOT}/onboard/collection_manager.py" --config "$CONFIG" "$@"
