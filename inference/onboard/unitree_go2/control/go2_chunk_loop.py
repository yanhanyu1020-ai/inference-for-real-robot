#!/usr/bin/env python3
"""Persistent Go2 velocity chunk executor.

Started via system.go2_control_start (alongside the C++ lease bridge).  Waits
on a unix socket for JSON action chunks:

    {"actions": [[vx, vy, vyaw], ...]}

Each row is forwarded to the velocity bridge at --dt.  Optional pose fields
(ignored by older copies of this file):

    {"actions": [...], "before": "stand_up", "after": "stand_down"}

Optional mid-chunk pose events (from measured body_height in sport CSV):

    {"actions": [...], "pose_events": [{"index": 288, "pose": "stand_up"}, ...]}

``before`` / ``after`` are stand_up, stand_down, or stop, and are sent on the
same lease-holding unix socket as MOVE.  Between chunks the dog is commanded
to STOP.  SIGTERM sends STOP and exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from go2_send import send_line  # noqa: E402
from episode_log import DEFAULT_POINTER, EpisodeLogger  # noqa: E402

VX_LIMIT = 1.0
VY_LIMIT = 1.0
VYAW_LIMIT = 3.5
POSE_MAP = {
    "stand_up": "STAND_UP",
    "stand_down": "STAND_DOWN",
    "stop": "STOP",
}
POSE_HOLD_S = {
    "STAND_UP": 3.0,
    "STAND_DOWN": 3.5,
    "STOP": 0.2,
}
# Brief zero-Move hold after StandUp only (JSON velocities stay intact).
POST_STAND_UP_SETTLE_S = 0.3
POST_STAND_DOWN_SETTLE_S = 0.0


def parse_args():
    here = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-sock", default=os.path.join(here, "go2.sock"))
    parser.add_argument("--socket", default=os.path.join(here, "go2_chunk.sock"))
    parser.add_argument("--dt", type=float, default=0.05,
                        help="seconds per action row (20 Hz default)")
    parser.add_argument("--hz", type=float, default=0.01,
                        help="seconds between repeated sends of one row "
                             "(default 0.01 -> 5 MOVE frames per 50 ms row). "
                             "Each unix write waits for SportClient.Move).")
    parser.add_argument(
        "--episode-pointer", default=DEFAULT_POINTER,
        help="path written by go2_capture_ctl / system.record_start",
    )
    return parser.parse_args()


def clamp(vx, vy, vyaw):
    return (
        max(-VX_LIMIT, min(VX_LIMIT, float(vx))),
        max(-VY_LIMIT, min(VY_LIMIT, float(vy))),
        max(-VYAW_LIMIT, min(VYAW_LIMIT, float(vyaw))),
    )


def apply_pose(bridge_sock, spec, when):
    if spec is None:
        return 0.0
    key = str(spec).strip().lower()
    if key not in POSE_MAP:
        logging.error("ignored %s pose %r (want stand_up/stand_down/stop)", when, spec)
        return 0.0
    line = POSE_MAP[key]
    logging.info("%s: %s", when, line)
    send_line(bridge_sock, line, timeout=3.0)
    hold = POSE_HOLD_S[line]
    if hold > 0:
        time.sleep(hold)
    logging.info("%s pose complete: %s", when, line)
    return hold


def _pause_after_pose(bridge_sock, pose, standing, hz):
    """Pause after stand/sit animation; JSON row velocities stay untouched."""
    pose = str(pose or "").strip().lower()
    if pose == "stand_up":
        settle_s = POST_STAND_UP_SETTLE_S
    elif pose == "stand_down":
        settle_s = POST_STAND_DOWN_SETTLE_S
    else:
        return 0.0
    if settle_s <= 0:
        return 0.0
    logging.info("post-%s pause %.1fs before rows resume", pose, settle_s)
    if standing:
        deadline = time.monotonic() + settle_s
        line = "MOVE 0 0 0"
        while time.monotonic() < deadline:
            send_line(bridge_sock, line)
            time.sleep(hz)
    else:
        time.sleep(settle_s)
    return settle_s


def _pose_event_map(pose_events):
    mapping = {}
    if not isinstance(pose_events, list):
        return mapping
    for item in pose_events:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
            pose = str(item["pose"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        if pose in POSE_MAP and index >= 0:
            mapping[index] = pose
    return mapping


def _standing_at_start(before, pose_events):
    """Whether MOVE is allowed before the first mid-chunk pose.

    A leftover stand_down means this chunk starts already standing (the
    opening StandUp ran as before=).  A later stand_up means the prefix
    is still prone: do not send Move until that animation finishes.
    """
    standing = str(before or "").strip().lower() == "stand_up"
    if not pose_events:
        return standing
    first_idx = min(pose_events)
    first_pose = pose_events[first_idx]
    if first_pose == "stand_down":
        return True
    if first_pose == "stand_up" and first_idx > 0:
        return False
    return standing


def _move_enabled_for_row(pose_events, idx, standing):
    """Send Move only while standing and after the walk-cluster stand_up.

    Episodes with two stand_up cycles (test squat, then walk) must not replay
    pose-delta velocities between them — that backward jitter after the first
    stand_up can leave Sport mode unable to walk after the second stand_up.
    """
    if not standing:
        return False
    ups = sorted(i for i, pose in pose_events.items() if pose == "stand_up")
    if len(ups) >= 2:
        return idx > ups[1]
    if len(ups) == 1:
        return idx > ups[0]
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logger = EpisodeLogger(
        args.episode_pointer,
        cmd_filename="go2_cmd.csv",
        cmd_fields=("vx", "vy", "vyaw"),
        replay_filename="go2_replay.json",
    )

    def release(signum, frame):
        logging.warning("signal %d: STOP then exit", signum)
        try:
            send_line(args.bridge_sock, "STOP", timeout=1.0)
        except Exception as exc:
            logging.error("failed to STOP on exit: %s", exc)
        logger.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, release)
    signal.signal(signal.SIGINT, release)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(4)
    server.settimeout(0.05)
    logging.info("go2 chunk executor listening on %s (bridge %s)",
                 args.socket, args.bridge_sock)
    logging.info("episode pointer %s", args.episode_pointer)

    while True:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            logger.log([0.0, 0.0, 0.0])
            continue
        raw = b""
        try:
            while True:
                part = conn.recv(1 << 20)
                if not part:
                    break
                raw += part
        finally:
            conn.close()
        try:
            msg = json.loads(raw.decode())
            rows = msg["actions"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("empty actions")
            before = msg.get("before")
            after = msg.get("after")
            pose_events = _pose_event_map(msg.get("pose_events"))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logging.error("bad chunk: %s", exc)
            continue

        logging.info("chunk received: %d rows before=%s after=%s pose_events=%d",
                     len(rows), before, after, len(pose_events))
        try:
            apply_pose(args.bridge_sock, before, "before")
            before_key = str(before or "").strip().lower()
            if before_key in ("stand_up", "stand_down"):
                _pause_after_pose(
                    args.bridge_sock, before_key, before_key == "stand_up", args.hz)
            t0 = time.monotonic()
            standing = _standing_at_start(before, pose_events)
            for idx, row in enumerate(rows):
                pose = pose_events.get(idx)
                if pose is not None:
                    apply_pose(args.bridge_sock, pose, "pose@%d" % idx)
                    standing = pose == "stand_up"
                    _pause_after_pose(args.bridge_sock, pose, standing, args.hz)
                    # Pose sleeps for seconds; keep the remaining MOVE rows on
                    # the original dt grid instead of permanently overrunning.
                    t0 = time.monotonic() - (idx + 1) * args.dt
                    continue
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    raise ValueError("row %d must be 3 numbers" % idx)
                vx, vy, vyaw = clamp(row[0], row[1], row[2])
                logger.log([vx, vy, vyaw])
                next_time = t0 + (idx + 1) * args.dt
                if _move_enabled_for_row(pose_events, idx, standing):
                    line = "MOVE %.6f %.6f %.6f" % (vx, vy, vyaw)
                    # The dog's sport controller ignores low-rate velocity
                    # commands; repeat each row at 200 Hz for its whole dt.
                    repeats = max(1, int(round(args.dt / args.hz)))
                    for _ in range(repeats):
                        send_line(args.bridge_sock, line)
                        time.sleep(args.hz)
                sleep_for = next_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif _move_enabled_for_row(pose_events, idx, standing):
                    logging.warning("chunk overran by %.0f ms at row %d",
                                    -sleep_for * 1000, idx)
        except Exception as exc:
            logging.exception("chunk failed: %s", exc)
        finally:
            try:
                send_line(args.bridge_sock, "STOP", timeout=1.0)
            except Exception as exc:
                logging.error("STOP after chunk failed: %s", exc)
            try:
                apply_pose(args.bridge_sock, after, "after")
            except Exception as exc:
                logging.error("after pose failed: %s", exc)
            logging.info("chunk done; STOP issued")
            logger.flush()


if __name__ == "__main__":
    main()
