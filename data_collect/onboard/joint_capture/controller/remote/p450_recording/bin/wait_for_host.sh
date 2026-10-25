#!/usr/bin/env bash
set -u

host="${1:?usage: wait_for_host.sh HOST [ATTEMPTS] [INTERVAL_SECONDS]}"
attempts="${2:-30}"
interval="${3:-1}"

case "$attempts" in
  ''|*[!0-9]*|0) echo "error: ATTEMPTS must be a positive integer" >&2; exit 2 ;;
esac

case "$interval" in
  ''|*[!0-9]*) echo "error: INTERVAL_SECONDS must be a non-negative integer" >&2; exit 2 ;;
esac

attempt=1
while [ "$attempt" -le "$attempts" ]; do
  attempt_started_ns=$(date +%s%N)
  if ping -c 1 -W 1 "$host" >/dev/null 2>&1; then
    echo "$host reachable on attempt $attempt/$attempts"
    exit 0
  fi
  echo "waiting for $host ($attempt/$attempts)"
  if [ "$attempt" -lt "$attempts" ]; then
    attempt_finished_ns=$(date +%s%N)
    interval_ns=$((10#$interval * 1000000000))
    remaining_ns=$((interval_ns - (attempt_finished_ns - attempt_started_ns)))
    if [ "$remaining_ns" -gt "$interval_ns" ]; then
      remaining_ns="$interval_ns"
    fi
    if [ "$remaining_ns" -gt 0 ]; then
      printf -v remaining_seconds '%d.%09d' \
        "$((remaining_ns / 1000000000))" "$((remaining_ns % 1000000000))"
      sleep "$remaining_seconds"
    fi
  fi
  attempt=$((attempt + 1))
done

echo "$host not reachable after $attempts attempts"
exit 1
