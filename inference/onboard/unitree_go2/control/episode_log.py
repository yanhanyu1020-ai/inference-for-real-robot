#!/usr/bin/env python3
"""Attach capture loops to the active episode.

The Go2 DDS recorder (go2_capture_ctl.sh) writes the episode directory to
``~/heterovla-recorder/active_episode``.  Chunk / teleop / teach loops poll
that pointer and append command/state CSVs in the same folder, then write a
replay JSON on detach.  Logging lives in the process that owns Piper CAN or
the Go2 bridge so a second Piper process is not needed.

Teach capture sets ``replay_from_state=True`` so ``piper_replay.json`` comes
from measured joints, not from JointCtrl targets.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time


DEFAULT_POINTER = os.path.expanduser("~/heterovla-recorder/active_episode")
# piper.action_chunk is still 5000 rows per command; the replay sender splits.
MAX_REPLAY_ROWS = 60000
GO2_REPLAY_DT = 0.05
VX_LIMIT = 1.0
VY_LIMIT = 1.0
# Match go2_chunk_loop so a fast in-place turn is not clipped on replay.
VYAW_LIMIT = 3.5
VEL_DEADZONE = 0.04
YAW_DEADZONE = 0.08
YAW_INTEGRAL_TOL_RAD = math.radians(0.5)
# Measured body_height from sportmodestate (m).  Prone is ~0.07, standing ~0.30.
BODY_HEIGHT_LOW = 0.12
BODY_HEIGHT_HIGH = 0.22
POSE_EVENT_DEBOUNCE_ROWS = 40
POST_STAND_UP_SETTLE_ROWS = 40
POSE_CLUSTER_GAP_ROWS = 60
# Keep a short hold after the last moving row so STOP is not immediate.
TRAILING_IDLE_ROWS = 20
LOG = logging.getLogger("heterovla.episode_log")


def monotonic_ns():
    return time.monotonic_ns()


def wall_ns():
    return time.time_ns()


def read_pointer(path):
    try:
        with open(path) as handle:
            value = handle.read().strip()
    except OSError:
        return None
    if not value:
        return None
    return os.path.abspath(value)


def downsample_rows(rows, max_rows=MAX_REPLAY_ROWS):
    if not rows:
        return []
    if len(rows) <= max_rows:
        return [list(row) for row in rows]
    last = max_rows - 1
    n = len(rows) - 1
    picked = []
    seen = set()
    for i in range(max_rows):
        idx = int(round(i * n / float(last)))
        if idx in seen:
            continue
        seen.add(idx)
        picked.append(list(rows[idx]))
    if n not in seen:
        picked.append(list(rows[-1]))
    return picked[:max_rows]


def replay_from_csv(csv_path, fields, max_rows=MAX_REPLAY_ROWS):
    rows = []
    with open(csv_path, "rb") as handle:
        raw = handle.read().replace(b"\x00", b"")
    text = raw.decode("utf-8", "replace")
    reader = csv.DictReader(text.splitlines())
    for item in reader:
        try:
            rows.append([float(item[name]) for name in fields])
        except (TypeError, ValueError, KeyError):
            continue
    return downsample_rows(rows, max_rows)


def _clamp_go2(vx, vy, vyaw):
    return [
        max(-VX_LIMIT, min(VX_LIMIT, float(vx))),
        max(-VY_LIMIT, min(VY_LIMIT, float(vy))),
        max(-VYAW_LIMIT, min(VYAW_LIMIT, float(vyaw))),
    ]


def _deadzone_stationary(vx, vy, vyaw):
    """Zero only while standing still so a slow heading change is kept."""
    if (
        abs(vx) < VEL_DEADZONE
        and abs(vy) < VEL_DEADZONE
        and abs(vyaw) < YAW_DEADZONE
    ):
        return 0.0, 0.0, 0.0
    return vx, vy, vyaw


def _unwrap_angle(delta):
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _net_yaw(samples):
    total = 0.0
    for i in range(1, len(samples)):
        total += _unwrap_angle(samples[i]["yaw"] - samples[i - 1]["yaw"])
    return total


def _read_sport_samples(csv_path):
    samples = []
    with open(csv_path) as handle:
        for item in csv.DictReader(handle):
            sample = {
                "t": int(float(item["monotonic_ns"])),
                "vx": float(item["velocity_x"]),
                "vy": float(item["velocity_y"]),
                "vyaw": float(item["yaw_speed"]),
                "px": float(item["position_x"]),
                "py": float(item["position_y"]),
                "yaw": float(item["yaw"]),
            }
            if "body_height" in item:
                sample["body_height"] = float(item["body_height"])
            samples.append(sample)
    return samples


def _resample_poses(samples, dt):
    t0 = samples[0]["t"]
    t1 = samples[-1]["t"]
    duration_s = max(dt, (t1 - t0) / 1e9)
    n = max(1, int(round(duration_s / dt)))
    n = min(n, MAX_REPLAY_ROWS)
    j = 0
    poses = []
    for i in range(n):
        target = t0 + int(round(i * dt * 1e9))
        while j + 1 < len(samples) and samples[j + 1]["t"] <= target:
            j += 1
        poses.append(samples[j])
    return poses


def _body_velocity(prev, cur, dt):
    """World-frame pose delta → body-frame [vx, vy, vyaw] over one replay step."""
    dx = (cur["px"] - prev["px"]) / dt
    dy = (cur["py"] - prev["py"]) / dt
    yaw = prev["yaw"]
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx = c * dx + s * dy
    vy = -s * dx + c * dy
    vyaw = _unwrap_angle(cur["yaw"] - prev["yaw"]) / dt
    return vx, vy, vyaw


def _restore_yaw_integral(rows, target_dyaw, dt):
    """Nudge unsaturated frames so sum(vyaw)*dt stays close to measured Δyaw."""
    current = sum(row[2] for row in rows) * dt
    leftover = target_dyaw - current
    if abs(leftover) <= YAW_INTEGRAL_TOL_RAD:
        return
    sign = 1.0 if leftover > 0 else -1.0
    remaining = abs(leftover)

    def headroom(row):
        if sign > 0:
            return VYAW_LIMIT - row[2]
        return VYAW_LIMIT + row[2]

    order = list(range(len(rows)))
    order.sort(key=lambda i: (0 if rows[i][2] * sign > 0.05 else 1, -headroom(rows[i])))
    for i in order:
        room = headroom(rows[i])
        if room <= 1e-9:
            continue
        add = min(room, remaining / dt)
        rows[i][2] += sign * add
        remaining -= add * dt
        if remaining <= 1e-9:
            break


def _resample_body_heights(samples, dt, n_rows):
    if not samples or n_rows <= 0:
        return []
    t0 = samples[0]["t"]
    j = 0
    heights = []
    for i in range(n_rows):
        target = t0 + int(round(i * dt * 1e9))
        while j + 1 < len(samples) and samples[j + 1]["t"] <= target:
            j += 1
        heights.append(float(samples[j].get("body_height", 0.0)))
    return heights


def replay_pose_events_from_body_height(
    csv_path,
    n_rows,
    dt=GO2_REPLAY_DT,
    low=BODY_HEIGHT_LOW,
    high=BODY_HEIGHT_HIGH,
):
    """Map measured body_height to stand_up / stand_down at replay row indices.

    Go2 SportClient has no continuous BodyHeight API; discrete STAND_UP /
    STAND_DOWN through the velocity bridge is the closest replay match.
    """
    if n_rows <= 0:
        return []
    samples = _read_sport_samples(csv_path)
    if not samples or not any("body_height" in item for item in samples):
        return []

    heights = _resample_body_heights(samples, dt, n_rows)
    if not heights:
        return []

    def _state(height):
        if height > high:
            return "high"
        if height < low:
            return "low"
        return "mid"

    events = []
    state = _state(heights[0])
    last_idx = -POSE_EVENT_DEBOUNCE_ROWS
    for idx, height in enumerate(heights):
        hs = _state(height)
        if hs == "mid":
            continue
        if hs == "high" and state != "high":
            if state in ("low", "mid") and idx - last_idx >= POSE_EVENT_DEBOUNCE_ROWS:
                events.append({"index": idx, "pose": "stand_up"})
                last_idx = idx
            state = "high"
        elif hs == "low" and state == "high":
            if idx - last_idx >= POSE_EVENT_DEBOUNCE_ROWS:
                events.append({"index": idx, "pose": "stand_down"})
                last_idx = idx
            state = "low"
        elif hs == "low":
            state = "low"
    return events


def _is_moving_row(row, vx_eps=VEL_DEADZONE, vyaw_eps=YAW_DEADZONE):
    try:
        vx, vy, vyaw = (float(row[0]), float(row[1]), float(row[2]))
    except (TypeError, ValueError, IndexError):
        return False
    return abs(vx) > vx_eps or abs(vy) > vx_eps or abs(vyaw) > vyaw_eps


def _motion_clusters(actions, gap_rows=POSE_CLUSTER_GAP_ROWS):
    moving_idxs = [i for i, row in enumerate(actions) if _is_moving_row(row)]
    if not moving_idxs:
        return []
    clusters = []
    cluster = [moving_idxs[0]]
    for idx in moving_idxs[1:]:
        if idx - cluster[-1] <= gap_rows:
            cluster.append(idx)
        else:
            clusters.append(cluster)
            cluster = [idx]
    clusters.append(cluster)
    return clusters


def compact_pose_events_for_replay(
    actions,
    include_stand_down=False,
    gap_rows=POSE_CLUSTER_GAP_ROWS,
    sport_path=None,
):
    """Keep every measured stand_up / stand_down, including in-place squats.

    Move is gated off while prone, so a first test squat does not eat the
    later walk.  Without body_height, fall back to the longest motion cluster.
    """
    height_events = []
    if sport_path:
        height_events = replay_pose_events_from_body_height(sport_path, len(actions))
    if height_events:
        return [
            {"index": int(item["index"]), "pose": str(item["pose"]).strip().lower()}
            for item in height_events
        ]

    clusters = _motion_clusters(actions, gap_rows=gap_rows)
    if not clusters:
        return []
    main = max(clusters, key=len)
    events = [{"index": max(0, main[0]), "pose": "stand_up"}]
    if include_stand_down:
        events.append({
            "index": min(main[-1] + 1, len(actions) - 1),
            "pose": "stand_down",
        })
    return events


def trim_go2_replay_idle(
    actions,
    pose_events,
    trailing_rows=TRAILING_IDLE_ROWS,
):
    """Drop leading zeros before the first stand_up and trailing idle rows.

    Collection episodes often sit prone for tens of seconds before the real
    stand/walk.  Replaying that prefix makes the dog look frozen.
    """
    if not actions:
        return [], []
    events = [
        {"index": int(item["index"]), "pose": str(item.get("pose", "")).strip().lower()}
        for item in (pose_events or [])
    ]
    start = 0
    ups = [item["index"] for item in events if item["pose"] == "stand_up"]
    if ups:
        start = max(0, min(ups))
    last_move = start
    for idx, row in enumerate(actions):
        if _is_moving_row(row):
            last_move = idx
    for item in events:
        last_move = max(last_move, item["index"])
    end = min(len(actions), last_move + max(0, trailing_rows) + 1)
    if start >= end:
        return [list(row) for row in actions], events
    trimmed = [list(row) for row in actions[start:end]]
    shifted = []
    for item in events:
        idx = item["index"] - start
        if 0 <= idx < len(trimmed):
            shifted.append({"index": idx, "pose": item["pose"]})
    return trimmed, shifted


def replay_from_sport_state(csv_path, dt=GO2_REPLAY_DT, max_rows=MAX_REPLAY_ROWS):
    """Body-frame velocity trajectory from sport_mode_state.csv.

    Joystick commands are a poor replay source: releasing the stick logs zeros
    and we STOP, but the dog may keep walking.  Pose deltas (not yaw_speed)
    keep sum(vx,vy,vyaw)*dt close to the recorded path and heading, including
    fast in-place turns that would be clipped if we replayed IMU rate.
    """
    del max_rows  # resample length is duration/dt, capped inside _resample_poses
    samples = _read_sport_samples(csv_path)
    if not samples:
        return []

    poses = _resample_poses(samples, dt)
    rows = []
    prev = poses[0]
    for i, cur in enumerate(poses):
        if i == 0:
            vx, vy, vyaw = 0.0, 0.0, 0.0
        else:
            vx, vy, vyaw = _body_velocity(prev, cur, dt)
            prev = cur
        vx, vy, vyaw = _deadzone_stationary(vx, vy, vyaw)
        rows.append(_clamp_go2(vx, vy, vyaw))
    _restore_yaw_integral(rows, _net_yaw(samples), dt)
    return rows


def gate_go2_actions_for_pose_events(actions, pose_events):
    """Zero velocity rows while the replay state machine is prone.

    Go2 ignores Move() until StandUp completes.  Recorded pose deltas can
    still be non-zero while body_height is low (coasting / height change).
    """
    if not actions or not pose_events:
        return [list(row) for row in actions]
    events = sorted(
        (int(item["index"]), str(item["pose"]).strip().lower())
        for item in pose_events
    )
    standing = False
    event_i = 0
    gated = []
    for row_i, row in enumerate(actions):
        while event_i < len(events) and events[event_i][0] == row_i:
            pose = events[event_i][1]
            if pose == "stand_up":
                standing = True
            elif pose == "stand_down":
                standing = False
            event_i += 1
        if standing:
            gated.append(list(row))
        else:
            gated.append([0.0, 0.0, 0.0])
    return gated


def go2_replay_monotonic_span(csv_path, dt=GO2_REPLAY_DT):
    """Absolute monotonic window kept after idle trim, for aligning the arm."""
    actions = replay_from_sport_state(csv_path, dt=dt)
    if not actions:
        return None
    samples = _read_sport_samples(csv_path)
    if not samples:
        return None
    pose_events = compact_pose_events_for_replay(
        actions, include_stand_down=True, sport_path=csv_path)
    events = [
        {"index": int(item["index"]), "pose": str(item.get("pose", "")).strip().lower()}
        for item in (pose_events or [])
    ]
    start = 0
    ups = [item["index"] for item in events if item["pose"] == "stand_up"]
    if ups:
        start = max(0, min(ups))
    last_move = start
    for idx, row in enumerate(actions):
        if _is_moving_row(row):
            last_move = idx
    for item in events:
        last_move = max(last_move, item["index"])
    end = min(len(actions), last_move + max(0, TRAILING_IDLE_ROWS) + 1)
    t0 = samples[0]["t"]
    start_ns = t0 + int(round(start * dt * 1e9))
    end_ns = t0 + int(round(max(start, end - 1) * dt * 1e9))
    return start_ns, end_ns


def build_go2_replay_from_sport(
    csv_path,
    dt=GO2_REPLAY_DT,
    include_stand_down=True,
):
    """Return (actions, pose_events) ready for hardware replay.

    ``pose_events`` are every body_height stand/sit, including in-place
    squats.  Velocity is zeroed while prone so later Move rows still run
    after the next StandUp.
    """
    actions = replay_from_sport_state(csv_path, dt=dt)
    if not actions:
        return [], []
    pose_events = compact_pose_events_for_replay(
        actions, include_stand_down=include_stand_down, sport_path=csv_path)
    # Keep measured velocities in JSON; go2_chunk_loop pauses after each
    # stand_up / stand_down and skips Move while prone instead of zeroing rows.
    actions, pose_events = trim_go2_replay_idle(actions, pose_events)
    return actions, pose_events


def write_replay_json(path, actions, source=None, pose_events=None):
    payload = {"actions": actions}
    if source:
        payload["source"] = source
    if pose_events:
        payload["pose_events"] = pose_events
    with open(path, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def write_go2_replay(episode_dir, dt=GO2_REPLAY_DT):
    """Overwrite go2_replay.json from measured sport state when present."""
    sport_path = os.path.join(episode_dir, "sport_mode_state.csv")
    replay_path = os.path.join(episode_dir, "go2_replay.json")
    if not os.path.isfile(sport_path):
        return None
    actions, pose_events = build_go2_replay_from_sport(sport_path, dt=dt)
    if not actions:
        return None
    write_replay_json(
        replay_path, actions, source="sport_mode_state", pose_events=pose_events)
    LOG.info(
        "wrote %s (%d rows, %.1fs, %d pose events) from %s",
        replay_path, len(actions), len(actions) * dt, len(pose_events), sport_path,
    )
    return replay_path


class EpisodeLogger:
    """Poll an active-episode pointer and append timestamped CSV rows."""

    def __init__(
        self,
        pointer_path,
        cmd_filename,
        cmd_fields,
        state_filename=None,
        state_fields=None,
        replay_filename=None,
        replay_from_state=False,
    ):
        self.pointer_path = pointer_path
        self.cmd_filename = cmd_filename
        self.cmd_fields = list(cmd_fields)
        self.state_filename = state_filename
        self.state_fields = list(state_fields or [])
        self.replay_filename = replay_filename
        self.replay_from_state = bool(replay_from_state)
        self.episode_dir = None
        self._cmd_file = None
        self._state_file = None
        self._cmd_writer = None
        self._state_writer = None
        self._rows = 0

    def sync(self):
        target = read_pointer(self.pointer_path)
        if target == self.episode_dir:
            return self.episode_dir
        if self.episode_dir is not None:
            self.close()
        if target is None:
            return None
        if not os.path.isdir(target):
            LOG.warning("active episode is not a directory: %s", target)
            return None
        self._open(target)
        return self.episode_dir

    def log(self, cmd, state=None):
        if self.sync() is None:
            return
        now = monotonic_ns()
        wall = wall_ns()
        cmd_row = {"monotonic_ns": now, "wall_time_ns": wall}
        for name, value in zip(self.cmd_fields, cmd):
            cmd_row[name] = float(value)
        self._cmd_writer.writerow(cmd_row)
        self._rows += 1
        if self._state_writer is not None and state is not None:
            state_row = {"monotonic_ns": now, "wall_time_ns": wall}
            for name, value in zip(self.state_fields, state):
                state_row[name] = float(value)
            self._state_writer.writerow(state_row)
        if self._rows % 40 == 0:
            self.flush()

    def flush(self):
        if self._cmd_file is not None:
            self._cmd_file.flush()
        if self._state_file is not None:
            self._state_file.flush()

    def close(self):
        episode_dir = self.episode_dir
        cmd_path = None
        state_path = None
        if episode_dir is not None:
            cmd_path = os.path.join(episode_dir, self.cmd_filename)
            if self.state_filename:
                state_path = os.path.join(episode_dir, self.state_filename)
        self.flush()
        for handle in (self._cmd_file, self._state_file):
            if handle is not None:
                handle.close()
        self._cmd_file = None
        self._state_file = None
        self._cmd_writer = None
        self._state_writer = None
        self.episode_dir = None
        self._rows = 0
        source_path = cmd_path
        source_name = self.cmd_filename
        source_fields = self.cmd_fields
        if self.replay_from_state:
            source_path = state_path
            source_name = self.state_filename
            source_fields = self.state_fields
        if (
            episode_dir
            and self.replay_filename
            and source_path
            and os.path.isfile(source_path)
        ):
            try:
                actions = replay_from_csv(source_path, source_fields)
                replay_path = os.path.join(episode_dir, self.replay_filename)
                write_replay_json(replay_path, actions, source=source_name)
                LOG.info(
                    "wrote %s (%d rows) from %s",
                    replay_path, len(actions), source_path,
                )
            except (OSError, ValueError, KeyError, csv.Error) as exc:
                LOG.error("failed to write replay JSON: %s", exc)

    def _open(self, episode_dir):
        cmd_path = os.path.join(episode_dir, self.cmd_filename)
        self._cmd_file = open(cmd_path, "a", newline="")
        self._cmd_writer = csv.DictWriter(
            self._cmd_file,
            fieldnames=["monotonic_ns", "wall_time_ns"] + self.cmd_fields,
        )
        if self._cmd_file.tell() == 0:
            self._cmd_writer.writeheader()
        if self.state_filename:
            state_path = os.path.join(episode_dir, self.state_filename)
            self._state_file = open(state_path, "a", newline="")
            self._state_writer = csv.DictWriter(
                self._state_file,
                fieldnames=["monotonic_ns", "wall_time_ns"] + self.state_fields,
            )
            if self._state_file.tell() == 0:
                self._state_writer.writeheader()
        self.episode_dir = episode_dir
        self._rows = 0
        LOG.info("recording %s into %s", self.cmd_filename, episode_dir)
