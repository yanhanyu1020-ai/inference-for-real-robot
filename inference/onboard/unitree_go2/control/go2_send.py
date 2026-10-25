#!/usr/bin/env python3
"""Send one velocity / stand command to the Go2 velocity bridge unix socket.

Text protocol (one line per command, space-separated):
  MOVE <vx> <vy> <vyaw>
  STAND_UP
  STAND_DOWN
  STOP

The bridge holds the SportClient lease; this client only writes commands.
"""

from __future__ import annotations

import argparse
import os
import socket
import time


DEFAULT_SOCK = os.path.expanduser("~/heterovla-control/go2.sock")


def send_line(sock_path, line, timeout=2.0):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(sock_path)
            sock.sendall((line.rstrip() + "\n").encode())
            sock.close()
            return
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError("bridge socket not ready (%s): %s" % (sock_path, last_error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sock", default=DEFAULT_SOCK)
    parser.add_argument("--stand-up", action="store_true")
    parser.add_argument("--stand-down", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="hold MOVE for this many seconds (20 Hz), then STOP")
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    flags = sum(bool(x) for x in (args.stand_up, args.stand_down, args.stop))
    if flags > 1:
        raise SystemExit("use only one of --stand-up / --stand-down / --stop")

    if args.stand_up:
        send_line(args.sock, "STAND_UP")
        print("STAND_UP")
        return
    if args.stand_down:
        send_line(args.sock, "STAND_DOWN")
        print("STAND_DOWN")
        return
    if args.stop:
        send_line(args.sock, "STOP")
        print("STOP")
        return

    line = "MOVE %.6f %.6f %.6f" % (args.vx, args.vy, args.vyaw)
    if args.seconds <= 0:
        send_line(args.sock, line)
        print(line)
        return

    dt = 1.0 / max(args.hz, 1.0)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        send_line(args.sock, line)
        time.sleep(dt)
    send_line(args.sock, "STOP")
    print("held %s for %.2fs then STOP" % (line, args.seconds))


if __name__ == "__main__":
    main()
