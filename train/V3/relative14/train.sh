#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$code_dir/../../../env.sh"
cd "$HETEROVLA_ROOT"
exec python -m train.V3.relative14.train "$@"
