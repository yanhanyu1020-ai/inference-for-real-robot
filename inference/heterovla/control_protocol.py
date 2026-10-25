"""Shared messages and validation for the HeteroVLA command channel.

The command channel intentionally carries named, high-level actions only.  Raw
joint targets and motor torques do not belong in this protocol.
"""

import json
import math
import uuid


PROTOCOL_VERSION = 1

COMMANDS = frozenset(
    {
        "system.ping",
        "system.status",
        "system.stop_all",
        "system.stream_start",
        "system.stream_stop",
        "system.stream_status",
        "system.control_start",
        "system.control_stop",
        "system.control_status",
        "system.go2_control_start",
        "system.go2_control_stop",
        "system.go2_control_status",
        "system.go2_stream_start",
        "system.go2_stream_stop",
        "system.go2_stream_status",
        "system.record_start",
        "system.record_stop",
        "system.record_status",
        "system.teleop_start",
        "system.teleop_stop",
        "system.teleop_status",
        "system.teach_start",
        "system.teach_stop",
        "system.teach_status",
        "piper.action_chunk",
        "go2.action_chunk",
        "go2.stand_up",
        "go2.stand_down",
        "go2.move_forward",
        "go2.move_backward",
        "go2.move_left",
        "go2.move_right",
        "go2.turn_left",
        "go2.turn_right",
        "go2.stop",
        "piper.enable",
        "piper.disable",
        "piper.go_zero",
        "piper.open_gripper",
        "piper.close_gripper",
        "piper.stop",
    }
)


class ProtocolError(ValueError):
    """Raised when a peer sends a malformed or forbidden message."""


def new_id():
    return uuid.uuid4().hex


def encode(message):
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def decode(payload):
    try:
        message = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("message is not valid JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    return message


def validate_registration(message):
    if message.get("type") != "register":
        raise ProtocolError("first message must be register")
    if message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    role = message.get("role")
    if role not in ("robot", "operator"):
        raise ProtocolError("role must be robot or operator")
    if role == "robot" and not _safe_name(message.get("robot_id")):
        raise ProtocolError("robot_id is required")
    return message


def validate_command(message):
    if message.get("type") != "command":
        raise ProtocolError("expected command message")
    if not _safe_name(message.get("id")):
        raise ProtocolError("command id is required")
    if not _safe_name(message.get("robot_id")):
        raise ProtocolError("robot_id is required")
    command = message.get("command")
    if command not in COMMANDS:
        raise ProtocolError("command is not in the allowlist: %r" % command)
    arguments = message.get("arguments", {})
    if command in ("piper.action_chunk", "go2.action_chunk"):
        dim = 7 if command == "piper.action_chunk" else 3
        actions = arguments.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= 5000:
            raise ProtocolError("%s requires actions list (1..5000 rows)" % command)
        for row in actions:
            if (not isinstance(row, (list, tuple)) or len(row) != dim
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)):
                raise ProtocolError("%s rows must be %d finite numbers" % (command, dim))
        if command == "go2.action_chunk":
            extra = set(arguments) - {"actions", "before", "after", "pose_events"}
            if extra:
                raise ProtocolError(
                    "unsupported go2.action_chunk keys: %s" % sorted(extra))
            for key in ("before", "after"):
                value = arguments.get(key)
                if value is None:
                    continue
                if value not in ("stand_up", "stand_down", "stop"):
                    raise ProtocolError(
                        "go2.action_chunk %s must be stand_up, stand_down, or stop"
                        % key)
            pose_events = arguments.get("pose_events")
            if pose_events is not None:
                if not isinstance(pose_events, list):
                    raise ProtocolError(
                        "go2.action_chunk pose_events must be a list")
                actions_len = len(actions)
                for item in pose_events:
                    if not isinstance(item, dict):
                        raise ProtocolError(
                            "go2.action_chunk pose_events entries must be objects")
                    pose = item.get("pose")
                    if pose not in ("stand_up", "stand_down", "stop"):
                        raise ProtocolError(
                            "go2.action_chunk pose_events.pose must be "
                            "stand_up, stand_down, or stop")
                    index = item.get("index")
                    if (not isinstance(index, int) or isinstance(index, bool)
                            or index < 0 or index >= actions_len):
                        raise ProtocolError(
                            "go2.action_chunk pose_events.index must be "
                            "0..%d" % (actions_len - 1))
    elif command == "system.record_start":
        extra = set(arguments) - {"episode_id", "instruction"}
        if extra:
            raise ProtocolError(
                "unsupported system.record_start keys: %s" % sorted(extra))
        episode_id = arguments.get("episode_id")
        if not _safe_episode_id(episode_id):
            raise ProtocolError(
                "system.record_start episode_id must be 1..64 "
                "letters, digits, '.', '_' or '-'")
        instruction = arguments.get("instruction", "")
        if instruction is None:
            instruction = ""
        if not isinstance(instruction, str) or len(instruction) > 512:
            raise ProtocolError(
                "system.record_start instruction must be a string of length 0..512")
    elif arguments != {}:
        raise ProtocolError("this protocol version does not accept command arguments")
    return message


def _safe_name(value):
    return isinstance(value, str) and 0 < len(value) <= 128


def _safe_episode_id(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(ch in allowed for ch in value) and value not in (".", "..")
