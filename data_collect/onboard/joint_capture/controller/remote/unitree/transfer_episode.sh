#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${COLLECTION_CONFIG:-${ROOT}/config/collection.json}"
PYTHON="${COLLECTION_PYTHON:-python3}"

config_value() {
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for part in sys.argv[2].split('.'):
    value = value[part]
print(value)
PY
}

usage() {
  echo "Usage: $0 EPISODE_ID [TASK]" >&2
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { usage; exit 2; }
episode_id="$1"
task="${2:-default}"
[[ "$episode_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "invalid episode id: $episode_id" >&2
  exit 2
}
[[ "$task" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "invalid task name: $task" >&2
  exit 2
}

data_root="$(config_value data_root)"
ssh_host="$(config_value a800.ssh_host)"
ssh_config="$(config_value a800.ssh_config)"
a800_root="$(config_value a800.data_root)"
episode_dir="$data_root/$task/$episode_id"

[ -d "$episode_dir" ] || { echo "episode not found: $episode_dir" >&2; exit 1; }
[ -f "$episode_dir/manifest.json" ] || {
  echo "episode has no manifest; stop and validate it before transfer" >&2
  exit 1
}
if [ -f "$ROOT/run/active_episode" ] && \
   [ "$(perl -0777 -pe 's/\s+$//' "$ROOT/run/active_episode")" = "$episode_dir" ]; then
  echo "cannot transfer an active episode" >&2
  exit 1
fi

incoming="$a800_root/incoming/$task/${episode_id}.partial"
destination="$a800_root/raw/$task/$episode_id"
remote_validator="${REMOTE_VALIDATOR:-$(dirname -- "$a800_root")/heterovla/data_collect/server/validate_transferred_episode.py}"

ssh -F "$ssh_config" "$ssh_host" \
  "mkdir -p '$incoming' '$a800_root/raw/$task'"
rsync -a --partial --info=progress2 \
  -e "ssh -F $ssh_config" \
  "$episode_dir/" "$ssh_host:$incoming/"
ssh -F "$ssh_config" "$ssh_host" \
  "python3 '$remote_validator' --episode '$incoming' --promote-to '$destination'"

echo "transfer complete: $ssh_host:$destination"
echo "local episode retained: $episode_dir"
