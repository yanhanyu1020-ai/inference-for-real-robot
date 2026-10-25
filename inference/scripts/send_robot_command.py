#!/usr/bin/env python3
"""Send one allowlisted command through a HeteroVLA control server."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

import websockets

# Support both canonical invocation and the repository-root compatibility link.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heterovla.control_protocol import COMMANDS, PROTOCOL_VERSION, encode, new_id


def load_arguments(data_text=None, data_file=None):
    if data_file:
        with open(data_file) as handle:
            data_text = handle.read()
    if not data_text:
        return {}
    arguments = json.loads(data_text)
    if not isinstance(arguments, dict):
        raise RuntimeError("command data must be a JSON object")
    return arguments


async def run(args):
    async with websockets.connect(args.uri, open_timeout=10, ping_interval=10) as websocket:
        await websocket.send(
            encode({"type": "register", "protocol": PROTOCOL_VERSION,
                    "role": "operator", "token": args.token})
        )
        registered = json.loads(await websocket.recv())
        if registered.get("type") != "registered":
            raise RuntimeError(registered.get("error", "registration failed"))
        request_id = new_id()
        arguments = load_arguments(args.data, args.data_file)
        await websocket.send(
            encode({"type": "command", "id": request_id, "robot_id": args.robot_id,
                    "command": args.command, "arguments": arguments})
        )
        while True:
            result = json.loads(await asyncio.wait_for(websocket.recv(), args.timeout))
            if result.get("type") == "result" and result.get("id") == request_id:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if not result.get("ok"):
                    raise SystemExit(2)
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--uri", default="ws://127.0.0.1:8766")
    parser.add_argument("--robot-id", default="unitree-go2-01")
    parser.add_argument("--data", default=None,
                        help="JSON object passed as command arguments, e.g. "
                             '{"actions": [[0,5,-10,0,15,0,2], ...]}')
    parser.add_argument("--data-file", default=None,
                        help="read command arguments from a JSON file or /dev/stdin")
    parser.add_argument("--token", default=os.environ.get("HETEROVLA_CONTROL_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=20.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
