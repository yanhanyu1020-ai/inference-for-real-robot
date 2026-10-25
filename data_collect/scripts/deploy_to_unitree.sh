#!/usr/bin/env bash
set -euo pipefail

# Run on Unitree. It pulls the source of truth from A800, then compiles only
# the read-only Go2 DDS bridge. It does not start collection or touch robots.
SOURCE_HOST="${SOURCE_HOST:-heterovla-a800}"
SOURCE_ROOT="${SOURCE_ROOT:-/meta_eon_cfs/home/yhy/heterovla/data_collect}"
SSH_CONFIG="${SSH_CONFIG:-${HOME}/heterovla-control/ssh_config}"
DESTINATION="${DESTINATION:-${HOME}/heterovla-collection}"

ONBOARD_DIR="$DESTINATION/onboard/unitree_go2/collection"
mkdir -p "$DESTINATION" "$ONBOARD_DIR"
rsync -a --exclude '__pycache__' --exclude '*.pyc' -e "ssh -F $SSH_CONFIG" \
  "$SOURCE_HOST:$SOURCE_ROOT/config" \
  "$DESTINATION/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' -e "ssh -F $SSH_CONFIG" \
  "$SOURCE_HOST:$SOURCE_ROOT/onboard/unitree_go2/collection/" \
  "$ONBOARD_DIR/"

# Preserve previously documented ~/heterovla-collection/onboard/<file> entries.
for source in "$ONBOARD_DIR"/*; do
  ln -sfn "unitree_go2/collection/$(basename "$source")" \
    "$DESTINATION/onboard/$(basename "$source")"
done

chmod +x \
  "$ONBOARD_DIR/collection_ctl.sh" \
  "$ONBOARD_DIR/transfer_episode.sh" \
  "$ONBOARD_DIR/transfer_task.sh" \
  "$ONBOARD_DIR/hetero_pkl_recorder.py" \
  "$ONBOARD_DIR/validate_episode.py" \
  "$ONBOARD_DIR/play_episode.py" \
  "$ONBOARD_DIR/replay_episode.py"

cmake -S "$ONBOARD_DIR" -B "$DESTINATION/build"
cmake --build "$DESTINATION/build" --parallel 2

python3 -m py_compile \
  "$ONBOARD_DIR/hetero_pkl_recorder.py" \
  "$ONBOARD_DIR/validate_episode.py" \
  "$ONBOARD_DIR/play_episode.py" \
  "$ONBOARD_DIR/replay_episode.py"

echo "installed at $DESTINATION"
echo "no collection process was started"
