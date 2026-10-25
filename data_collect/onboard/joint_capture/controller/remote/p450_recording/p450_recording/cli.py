import argparse
import json
import sys

from .exporter import export_session
from .manager import DEFAULT_DATA_ROOT, RecorderManager


def build_parser():
    parser = argparse.ArgumentParser(
        prog="p450-record",
        description="Record synchronized P450 camera and MAVROS topics.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="recording root directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="check topics and free disk space")
    start = subparsers.add_parser("start", help="start a recording session")
    start.add_argument("session_name")
    subparsers.add_parser("status", help="show recorder state")
    subparsers.add_parser("stop", help="stop the active recording cleanly")
    export = subparsers.add_parser(
        "export", help="export MP4, pose CSV and frame-pose alignment CSV"
    )
    export.add_argument("session_dir")
    export.add_argument("--max-delta-ms", type=float, default=50.0)
    export.add_argument("--max-interp-gap-ms", type=float, default=200.0)
    return parser


def _write_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def main(argv=None, stdout=None, stderr=None):
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    manager = RecorderManager(data_root=args.data_root)
    try:
        if args.command == "check":
            result = manager.check()
            _write_json(stdout, result)
            return 0 if result["ok"] else 2
        if args.command == "start":
            session_dir = manager.start(args.session_name)
            _write_json(stdout, {"started": True, "session_dir": str(session_dir)})
            return 0
        if args.command == "status":
            _write_json(stdout, manager.status())
            return 0
        if args.command == "stop":
            _write_json(stdout, manager.stop())
            return 0
        if args.command == "export":
            result = export_session(
                args.session_dir,
                max_delta_ms=args.max_delta_ms,
                max_interp_gap_ms=args.max_interp_gap_ms,
            )
            _write_json(stdout, result)
            return 0
    except (RuntimeError, ValueError, OSError) as exc:
        stderr.write(f"error: {exc}\n")
        return 2

    stderr.write(f"error: unsupported command {args.command}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
