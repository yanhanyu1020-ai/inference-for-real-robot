#!/usr/bin/env python3
"""Send one safe test request to an OpenPI-compatible policy server."""

import argparse

import numpy as np
from openpi_client import websocket_client_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prompt", default="communication test")
    parser.add_argument("--state-dim", type=int, default=6)
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print("server metadata:", policy.get_server_metadata())

    response = policy.infer(
        {
            "prompt": args.prompt,
            "observation/state": np.zeros(args.state_dim, dtype=np.float32),
        }
    )
    actions = np.asarray(response["actions"])
    print("echo prompt:", response.get("echo_prompt"))
    print("received state dim:", response.get("received_state_dim"))
    print("actions shape:", actions.shape)
    print("actions are all zero:", bool(np.all(actions == 0)))
    print("safe test only:", response.get("safe_test_only"))


if __name__ == "__main__":
    main()
