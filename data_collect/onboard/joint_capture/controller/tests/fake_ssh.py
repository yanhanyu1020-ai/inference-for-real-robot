"""Deterministic persistent responder used by clock probe tests."""

from __future__ import annotations

import queue
import argparse
import sys
import time


class FakePersistentProcess:
    """A line-oriented local stand-in for the remote SSH responder."""

    def __init__(self, offset_ns: int = 0, delay_ns: int = 0, drift_ns: int = 0,
                 outliers: tuple[int, ...] = ()) -> None:
        self.offset_ns = offset_ns
        self.delay_ns = delay_ns
        self.drift_ns = drift_ns
        self.outliers = outliers
        self.requests = 0
        self.returncode = None
        self.stdin = self
        self.stdout = self
        self.stderr = self
        self._responses: queue.Queue[str] = queue.Queue()

    def write(self, _request: str) -> int:
        index = self.requests
        self.requests += 1
        local_wall = time.time_ns()
        remote_receive = local_wall + self.offset_ns + index * self.drift_ns
        # The delay is transport delay in this local fake; remote processing is
        # intentionally zero so the four-timestamp RTT remains non-negative.
        remote_send = remote_receive
        if self.delay_ns:
            # Give the test process enough scheduling slack to keep RTT valid.
            time.sleep(self.delay_ns / 1_000_000_000)
        if index in self.outliers:
            remote_send += 100_000_000
        self._responses.put(f"{remote_receive} {remote_send} {time.monotonic_ns()}\n")
        return 1

    def flush(self) -> None:
        pass

    def readline(self, _limit: int = -1) -> str:
        return self._responses.get()

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def main(argv: list[str] | None = None) -> int:
    """Deterministic line responder for subprocess tests (never touches disk)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset-ns", type=int, default=0)
    parser.add_argument("--drift-ns", type=int, default=0)
    parser.add_argument("--delay-ns", type=int, default=0)
    parser.add_argument("--outliers", default="")
    args = parser.parse_args(argv)
    outliers = {int(x) for x in args.outliers.split(",") if x.strip()}
    for index, _line in enumerate(sys.stdin):
        local_wall = time.time_ns()
        remote_receive = local_wall + args.offset_ns + index * args.drift_ns
        remote_send = remote_receive + (100_000_000 if index in outliers else 0)
        if args.delay_ns:
            time.sleep(args.delay_ns / 1_000_000_000)
        print(remote_receive, remote_send, time.monotonic_ns(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
