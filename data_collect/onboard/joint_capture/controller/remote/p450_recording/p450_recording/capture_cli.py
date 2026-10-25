import argparse
import json
import sys

from .manager import DEFAULT_DATA_ROOT
from .orchestrator import CaptureOrchestrator


def build_parser():
    parser = argparse.ArgumentParser(
        prog="p450_capture",
        description="Prepare, record and export synchronized P450 datasets.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare",
        help="recovery only: start and validate a missing sensor stack",
    )
    start = commands.add_parser("start", help="start recording")
    start.add_argument("session_name")
    start.add_argument(
        "--raw-rgb", action="store_true",
        help="record uncompressed RGB (debug only; maximum 2 minutes)",
    )
    start.add_argument(
        "--max-minutes", type=float, default=None,
        help="automatic recording limit (default: 30 compressed, 2 raw)",
    )
    commands.add_parser("finish", help="stop recording and export the dataset")
    commands.add_parser("stop-fast", help="stop recording without exporting")
    finalize = commands.add_parser("finalize-raw", help="check/recover rosbag without video export")
    finalize.add_argument("session_dir")
    export = commands.add_parser("export", help="manually export video and CSV later")
    export.add_argument("session_dir")
    commands.add_parser("status", help="show stack and recording status")
    commands.add_parser("shutdown", help="stop capture-owned sensor processes")
    return parser


def _write_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def main(argv=None, stdout=None, stderr=None):
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    orchestrator = CaptureOrchestrator(data_root=args.data_root)
    try:
        if args.command == "prepare":
            result = orchestrator.prepare()
        elif args.command == "start":
            result = orchestrator.start(
                args.session_name,
                raw_rgb=args.raw_rgb,
                max_minutes=args.max_minutes,
            )
        elif args.command == "finish":
            result = orchestrator.finish()
        elif args.command == "stop-fast":
            result = orchestrator.stop_fast()
        elif args.command == "finalize-raw":
            result = orchestrator.finalize_raw(args.session_dir)
        elif args.command == "export":
            result = orchestrator.export(args.session_dir)
        elif args.command == "status":
            result = orchestrator.status()
        elif args.command == "shutdown":
            result = orchestrator.shutdown()
        else:
            raise RuntimeError(f"unsupported command {args.command}")
        _write_json(stdout, result)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
