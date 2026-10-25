#!/usr/bin/env bash
set -e
exec "$(dirname "$0")/build/p450_estimator" "$@"
