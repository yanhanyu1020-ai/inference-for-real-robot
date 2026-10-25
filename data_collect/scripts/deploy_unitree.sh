#!/usr/bin/env bash
set -euo pipefail

# Resolve compatibility entry-point symlinks before finding the module root.
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_PATH" ]; do
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
  SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
  [[ "$SCRIPT_PATH" = /* ]] || SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"
done
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
repo_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
source_dir="${repo_root}/data_collect/onboard/unitree_go2"
robot_host="${GO2_HOST:-unitree}"
remote_dir="${GO2_REMOTE_DIR:-/home/unitree/heterovla-recorder}"
sdk_root="${UNITREE_SDK2_ROOT:-/home/unitree/unitree_sdk2-main}"

if [[ ! -f "${source_dir}/recorder/go2_sdk_recorder.cpp" ]]; then
  echo "Unitree recorder source is missing: ${source_dir}/recorder"
  exit 1
fi

if ! git -C "${repo_root}" diff --quiet ||
   ! git -C "${repo_root}" diff --cached --quiet; then
  echo "Refusing to deploy from a dirty Git worktree."
  exit 1
fi

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"

ssh "${robot_host}" \
  "mkdir -p '${remote_dir}/bin' '${remote_dir}/build'"

scp \
  "${source_dir}/recorder/CMakeLists.txt" \
  "${source_dir}/recorder/go2_sdk_recorder.cpp" \
  "${source_dir}/recorder/go2_capture_ctl.sh" \
  "${source_dir}/recorder/README.md" \
  "${source_dir}/recorder/README_zh.md" \
  "${robot_host}:${remote_dir}/"

ssh "${robot_host}" \
  "printf '%s\n' '${git_commit}' > '${remote_dir}/source_git_commit.txt' && \
   chmod +x '${remote_dir}/go2_capture_ctl.sh' && \
   cmake -S '${remote_dir}' -B '${remote_dir}/build' \
     -DUNITREE_SDK2_ROOT='${sdk_root}' && \
   cmake --build '${remote_dir}/build' -j2 && \
   cp '${remote_dir}/build/go2_sdk_recorder' \
      '${remote_dir}/bin/go2_sdk_recorder'"

echo "Deployed commit ${git_commit} to ${robot_host}:${remote_dir}"
