#!/usr/bin/env python3
"""Stream a Go2 velocity trajectory from a (mock) policy server and execute it.

OpenPI-compatible client.  Each infer() returns actions shaped (H, 3):
    [vx, vy, vyaw] in m/s and rad/s

Requires the C++ velocity bridge (lease holder) already running on --bridge-sock.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np
from openpi_client import websocket_client_policy

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from go2_send import send_line  # noqa: E402

VX_LIMIT = 0.5
VY_LIMIT = 0.5
VYAW_LIMIT = 1.0


def parse_args():
    here = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--prompt", default="scripted dog motion")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--bridge-sock", default=os.path.join(here, "go2.sock"))
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="scales model velocities before clamp (safety)")
    return parser.parse_args()


def clamp_row(row):
    vx, vy, vyaw = (float(v) * 1.0 for v in row)
    return (
        max(-VX_LIMIT, min(VX_LIMIT, vx)),
        max(-VY_LIMIT, min(VY_LIMIT, vy)),
        max(-VYAW_LIMIT, min(VYAW_LIMIT, vyaw)),
    )


def run(args):
    stop_requested = False

    def graceful_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logging.warning("stopping Go2 stream")

    signal.signal(signal.SIGTERM, graceful_stop)
    signal.signal(signal.SIGINT, graceful_stop)

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("server metadata: %s", policy.get_server_metadata())

    observation = {
        "prompt": args.prompt,
        "observation/state": np.zeros(3, dtype=np.float32),
    }
    response = policy.infer(observation)
    actions = np.asarray(response["actions"], dtype=np.float32).reshape(-1, 3).copy()
    actions *= args.speed_scale
    logging.info("received action chunk: shape=%s prompt=%r",
                 actions.shape, response.get("echo_prompt"))

    start = time.monotonic()
    try:
        for i, row in enumerate(actions):
            if stop_requested:
                logging.info("trajectory stopped at step %d", i + 1)
                break
            vx, vy, vyaw = clamp_row(row)
            send_line(args.bridge_sock, "MOVE %.6f %.6f %.6f" % (vx, vy, vyaw))
            if i % 10 == 0 or i == len(actions) - 1:
                logging.info("step %d/%d vx=%.3f vy=%.3f vyaw=%.3f",
                             i + 1, len(actions), vx, vy, vyaw)
            next_time = start + (i + 1) * args.dt
            sleep_for = next_time - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                logging.warning("step %d overran by %.0f ms", i + 1, -sleep_for * 1000)
    finally:
        try:
            send_line(args.bridge_sock, "STOP", timeout=1.0)
        except Exception as exc:
            logging.error("STOP failed: %s", exc)
        logging.info("trajectory done; STOP issued")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        logging.warning("interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
