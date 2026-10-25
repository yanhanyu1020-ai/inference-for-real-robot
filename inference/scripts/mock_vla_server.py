#!/usr/bin/env python3
"""OpenPI-compatible mock VLA server streaming scripted arm trajectories.

Speaks the same websocket + msgpack protocol as openpi's serve_policy.py, so
the robot-side streaming client is unchanged when a real policy is attached
later.  This server never commands hardware.

Action convention (7 dims, one row per step at dt):
    [j1, j2, j3, j4, j5, j6, gripper]
    joints in degrees, gripper in mm.  The client converts to SDK units.
"""

import argparse
import asyncio
import http
import logging
import time

import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server

# Piper joint limits in degrees (from piper_sdk param manager).
JOINT_LIMITS_DEG = np.array([
    [-150.0, 150.0],
    [0.0, 180.0],
    [-170.0, 0.0],
    [-100.0, 100.0],
    [-70.0, 70.0],
    [-120.0, 120.0],
])
GRIPPER_RANGE_MM = (0.0, 70.0)


def smooth_step_waypoints(waypoints, steps_per_segment, dt):
    """Interpolate waypoints with smooth (sinusoidal) easing.

    Each waypoint is (H, 7): [j1..j6 deg, gripper mm].  The trajectory keeps
    the last waypoint until the requested duration is covered.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    chunks = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        for s in range(steps_per_segment):
            t = (s + 1) / steps_per_segment
            eased = 0.5 - 0.5 * np.cos(np.pi * t)
            chunks.append(a + (b - a) * eased)
    return np.asarray(chunks, dtype=np.float64)


def default_trajectory(steps_per_segment=100):
    """A 20s-ish smooth joint motion loop, all values within SDK limits.

    Gripper targets stay >= 2 mm: commanding 0 mm sits past the physical
    closed stop (feedback floor ~0.2 mm) and makes the driver hunt against
    the stop (shake + noise).
    """
    return smooth_step_waypoints(
        [
            [0.0, 10.0, -20.0, 0.0, 30.0, 0.0, 2.0],
            [45.0, 30.0, -60.0, 0.0, 60.0, 45.0, 30.0],
            [0.0, 20.0, -45.0, 10.0, 30.0, 0.0, 50.0],
            [0.0, 10.0, -20.0, 0.0, 30.0, 0.0, 2.0],
        ],
        steps_per_segment,
        dt=0.1,
    )


def dog_trajectory(steps_per_segment=100):
    """A Go2 velocity trajectory: [vx, vy, vyaw] in m/s and rad/s.

    Hold each phase so motion is obvious: forward -> stop -> yaw -> stop ->
    backward -> stop. Duplicate waypoints keep velocity constant for a full
    segment. Ends stationary so repeated runs are safe.
    """
    return smooth_step_waypoints(
        [
            [0.3, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        steps_per_segment,
        dt=0.05,
    )


def clamp_trajectory(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float32).copy()
    trajectory[:, :6] = np.clip(trajectory[:, :6], JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])
    trajectory[:, 6] = np.clip(trajectory[:, 6], GRIPPER_RANGE_MM[0], GRIPPER_RANGE_MM[1])
    return trajectory


async def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def handle_client(websocket, trajectory, action_dim):
    peer = websocket.remote_address
    logging.info("client connected: %s (trajectory steps=%d)", peer, len(trajectory))
    packer = msgpack_numpy.Packer()
    await websocket.send(
        packer.pack(
            {
                "service": "heterovla-mock-vla",
                "protocol": "openpi-msgpack-v1",
                "action_dim": action_dim,
                "horizon": len(trajectory),
                "dt_s": 0.05,
                "safe_test_only": True,
            }
        )
    )
    try:
        while True:
            observation = msgpack_numpy.unpackb(await websocket.recv())
            started = time.monotonic()
            response = {
                "actions": trajectory.astype(np.float32),
                "echo_prompt": str(observation.get("prompt", "")),
                "safe_test_only": True,
                "server_timing": {"infer_ms": (time.monotonic() - started) * 1000},
            }
            await websocket.send(packer.pack(response))
            logging.info("request from %s: prompt=%r", peer, str(observation.get("prompt", "")))
    except websockets.ConnectionClosed:
        logging.info("client disconnected: %s", peer)


async def run(host, port, trajectory, action_dim):
    async with websockets.asyncio.server.serve(
        lambda ws: handle_client(ws, trajectory, action_dim),
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        logging.info("mock VLA server listening on %s:%d (%d steps, %d dims)",
                     host, port, len(trajectory), action_dim)
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--robot", choices=("arm", "dog"), default="arm")
    parser.add_argument("--steps-per-segment", type=int, default=100,
                        help="steps between consecutive waypoints")
    args = parser.parse_args()
    if args.robot == "dog":
        trajectory = dog_trajectory(args.steps_per_segment)
        action_dim = 3
    else:
        trajectory = clamp_trajectory(default_trajectory(args.steps_per_segment))
        action_dim = 7
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args.host, args.port, trajectory, action_dim))


if __name__ == "__main__":
    main()
