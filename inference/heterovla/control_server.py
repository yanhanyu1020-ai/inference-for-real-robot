#!/usr/bin/env python3
"""WebSocket broker for operator-to-robot high-level commands."""

import argparse
import asyncio
import logging
import os
import time

import websockets

from heterovla.control_protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    encode,
    validate_command,
    validate_registration,
)


LOG = logging.getLogger("heterovla.control_server")


class ControlServer:
    def __init__(self, token="", command_timeout=15.0):
        self.token = token
        self.command_timeout = command_timeout
        self.robots = {}
        self.pending = {}

    async def handle(self, websocket, path=None):
        role = None
        robot_id = None
        try:
            registration = validate_registration(
                decode(await asyncio.wait_for(websocket.recv(), timeout=10.0))
            )
            if self.token and registration.get("token") != self.token:
                raise ProtocolError("authentication failed")
            role = registration["role"]
            robot_id = registration.get("robot_id")
            if role == "robot":
                old = self.robots.get(robot_id)
                if old is not None and old is not websocket:
                    await old.close(code=4001, reason="replaced by new robot session")
                self.robots[robot_id] = websocket
                LOG.info("robot connected: %s", robot_id)
            await websocket.send(
                encode({"type": "registered", "protocol": PROTOCOL_VERSION, "role": role})
            )

            async for payload in websocket:
                message = decode(payload)
                if role == "operator":
                    await self._from_operator(websocket, message)
                else:
                    await self._from_robot(robot_id, message)
        except asyncio.TimeoutError:
            await self._safe_close(websocket, 4008, "registration timeout")
        except ProtocolError as exc:
            await self._safe_send(websocket, {"type": "error", "error": str(exc)})
            await self._safe_close(websocket, 4000, "protocol error")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if role == "robot" and self.robots.get(robot_id) is websocket:
                del self.robots[robot_id]
                LOG.warning("robot disconnected: %s", robot_id)
                self._fail_robot_requests(robot_id, "robot disconnected")

    async def _from_operator(self, websocket, message):
        if message.get("type") == "list_robots":
            await websocket.send(
                encode({"type": "robots", "robots": sorted(self.robots.keys())})
            )
            return
        validate_command(message)
        robot_id = message["robot_id"]
        robot = self.robots.get(robot_id)
        if robot is None:
            await websocket.send(
                encode({"type": "result", "id": message["id"], "ok": False,
                        "error": "robot is offline"})
            )
            return
        request_id = message["id"]
        if request_id in self.pending:
            raise ProtocolError("duplicate command id")
        self.pending[request_id] = (websocket, robot_id, time.monotonic())
        await robot.send(encode(message))
        asyncio.create_task(self._expire(request_id))

    async def _from_robot(self, robot_id, message):
        message_type = message.get("type")
        if message_type == "heartbeat":
            return
        if message_type != "result":
            raise ProtocolError("robot may only send result or heartbeat")
        request_id = message.get("id")
        pending = self.pending.pop(request_id, None)
        if pending is None:
            LOG.warning("late or unknown result from %s: %s", robot_id, request_id)
            return
        operator, expected_robot, _ = pending
        if expected_robot != robot_id:
            raise ProtocolError("result robot mismatch")
        await self._safe_send(operator, message)

    async def _expire(self, request_id):
        await asyncio.sleep(self.command_timeout)
        pending = self.pending.pop(request_id, None)
        if pending is not None:
            operator, _, _ = pending
            await self._safe_send(
                operator,
                {"type": "result", "id": request_id, "ok": False,
                 "error": "command timed out"},
            )

    def _fail_robot_requests(self, robot_id, reason):
        for request_id, pending in list(self.pending.items()):
            operator, expected_robot, _ = pending
            if expected_robot == robot_id:
                del self.pending[request_id]
                asyncio.create_task(
                    self._safe_send(operator, {"type": "result", "id": request_id,
                                               "ok": False, "error": reason})
                )

    @staticmethod
    async def _safe_send(websocket, message):
        try:
            await websocket.send(encode(message))
        except websockets.exceptions.ConnectionClosed:
            pass

    @staticmethod
    async def _safe_close(websocket, code, reason):
        try:
            await websocket.close(code=code, reason=reason)
        except websockets.exceptions.ConnectionClosed:
            pass


async def serve(args):
    broker = ControlServer(args.token, args.command_timeout)
    async with websockets.serve(
        broker.handle,
        args.host,
        args.port,
        ping_interval=10,
        ping_timeout=10,
        max_size=1024 * 1024,
    ):
        LOG.info("control server listening on ws://%s:%d", args.host, args.port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--command-timeout", type=float, default=15.0)
    parser.add_argument("--token", default=os.environ.get("HETEROVLA_CONTROL_TOKEN", ""))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
