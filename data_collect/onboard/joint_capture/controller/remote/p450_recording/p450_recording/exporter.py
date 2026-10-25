import csv
import statistics
from pathlib import Path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

from .common import (
    COMPRESSED_IMAGE_TOPIC,
    RAW_IMAGE_TOPIC,
    interpolate_pose,
    quaternion_to_euler,
    stamp_to_ns,
    bag_sort_key,
    pending_bags,
    padded_bag_path,
)


POSE_TOPIC = "/Odometry"
IMAGE_TOPICS = (COMPRESSED_IMAGE_TOPIC, RAW_IMAGE_TOPIC)

POSE_FIELDS = [
    "stamp_ns",
    "time_s",
    "frame_id",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll",
    "pitch",
    "yaw",
]

ALIGNMENT_FIELDS = [
    "frame_index",
    "image_stamp_ns",
    "image_time_s",
    "pose_stamp_ns",
    "delta_ms",
    "valid",
    "alignment_method",
    "aligned_pose_stamp_ns",
    "prev_pose_stamp_ns",
    "next_pose_stamp_ns",
    "alpha",
    "prev_delta_ms",
    "next_delta_ms",
    "interp_valid",
    "frame_id",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll",
    "pitch",
    "yaw",
]


def _find_bags(session_dir):
    raw_dir = session_dir / "raw"
    unresolved = pending_bags(raw_dir) + pending_bags(session_dir)
    if unresolved:
        raise ValueError("unfinished .bag.active shards; run finalize-raw before export: " +
                         ", ".join(path.name for path in unresolved))
    bags = sorted(raw_dir.glob("*.bag"), key=bag_sort_key)
    if not bags:
        bags = sorted(session_dir.glob("*.bag"), key=bag_sort_key)
    if not bags:
        raise ValueError(f"no .bag files found under {session_dir}")
    canonical = [padded_bag_path(path) for path in bags]
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate shard names; resolve legacy/padded copies before export")
    return bags


def _pose_row(message):
    stamp_ns = stamp_to_ns(message.header.stamp)
    pose = message.pose.pose
    position = pose.position
    orientation = pose.orientation
    roll, pitch, yaw = quaternion_to_euler(
        orientation.x, orientation.y, orientation.z, orientation.w
    )
    return {
        "stamp_ns": stamp_ns,
        "time_s": f"{stamp_ns / 1_000_000_000.0:.9f}",
        "frame_id": message.header.frame_id,
        "x": position.x,
        "y": position.y,
        "z": position.z,
        "qx": orientation.x,
        "qy": orientation.y,
        "qz": orientation.z,
        "qw": orientation.w,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def _read_index(bag_paths):
    poses = []
    image_stamps = []
    image_topics = set()
    for bag_path in bag_paths:
        with rosbag.Bag(str(bag_path), "r") as bag:
            for topic, message, _ in bag.read_messages(
                topics=[POSE_TOPIC, *IMAGE_TOPICS]
            ):
                if topic == POSE_TOPIC:
                    row = _pose_row(message)
                    poses.append((row["stamp_ns"], row))
                else:
                    image_topics.add(topic)
                    image_stamps.append(stamp_to_ns(message.header.stamp))
    poses.sort(key=lambda item: item[0])
    if not poses:
        raise ValueError(f"no messages found on {POSE_TOPIC}")
    if not image_stamps:
        raise ValueError("no messages found on a supported RGB image topic")
    if len(image_topics) != 1:
        raise ValueError(
            "session contains multiple RGB image topics: "
            + ", ".join(sorted(image_topics))
        )
    return poses, image_stamps, image_topics.pop()


def _infer_fps(image_stamps, fallback=15.0):
    intervals = [
        (later - earlier) / 1_000_000_000.0
        for earlier, later in zip(image_stamps, image_stamps[1:])
        if later > earlier
    ]
    if not intervals:
        return fallback
    return min(120.0, max(1.0, 1.0 / statistics.median(intervals)))


def _write_pose_csv(path, poses):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=POSE_FIELDS)
        writer.writeheader()
        writer.writerows(row for _, row in poses)


def _formatted_number(value):
    return "" if value == "" else f"{value:.6f}"


def _write_alignment_csv(
    path, image_stamps, poses, max_delta_ms, max_interp_gap_ms
):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ALIGNMENT_FIELDS)
        writer.writeheader()
        for frame_index, image_stamp_ns in enumerate(image_stamps):
            alignment = interpolate_pose(
                poses,
                image_stamp_ns,
                max_delta_ms=max_delta_ms,
                max_interp_gap_ms=max_interp_gap_ms,
            )
            pose = alignment["pose"]
            pose_fields = {field: "" for field in POSE_FIELDS[2:]}
            if pose is not None:
                roll, pitch, yaw = quaternion_to_euler(
                    pose["qx"], pose["qy"], pose["qz"], pose["qw"]
                )
                pose_fields.update(
                    pose,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                )
            writer.writerow({
                "frame_index": frame_index,
                "image_stamp_ns": image_stamp_ns,
                "image_time_s": f"{image_stamp_ns / 1_000_000_000.0:.9f}",
                "pose_stamp_ns": alignment["pose_stamp_ns"],
                "delta_ms": _formatted_number(alignment["delta_ms"]),
                "valid": int(alignment["valid"]),
                "alignment_method": alignment["alignment_method"],
                "aligned_pose_stamp_ns": alignment["aligned_pose_stamp_ns"],
                "prev_pose_stamp_ns": alignment["prev_pose_stamp_ns"],
                "next_pose_stamp_ns": alignment["next_pose_stamp_ns"],
                "alpha": _formatted_number(alignment["alpha"]),
                "prev_delta_ms": _formatted_number(alignment["prev_delta_ms"]),
                "next_delta_ms": _formatted_number(alignment["next_delta_ms"]),
                "interp_valid": int(alignment["interp_valid"]),
                **pose_fields,
            })


def _decode_image(message, image_topic, bridge):
    if image_topic == COMPRESSED_IMAGE_TOPIC:
        image = cv2.imdecode(
            np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("cannot decode compressed RGB frame")
        return image
    return bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")


def _write_video(path, bag_paths, fps, image_topic):
    bridge = CvBridge()
    writer = None
    frame_count = 0
    try:
        for bag_path in bag_paths:
            with rosbag.Bag(str(bag_path), "r") as bag:
                for _, message, _ in bag.read_messages(topics=[image_topic]):
                    image = _decode_image(message, image_topic, bridge)
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot open video writer for {path}")
                    writer.write(image)
                    frame_count += 1
    finally:
        if writer is not None:
            writer.release()
    return frame_count


def export_session(session_dir, max_delta_ms=50.0, max_interp_gap_ms=200.0):
    session_dir = Path(session_dir).expanduser().resolve()
    if max_delta_ms < 0:
        raise ValueError("max_delta_ms must be non-negative")
    if max_interp_gap_ms < 0:
        raise ValueError("max_interp_gap_ms must be non-negative")
    bag_paths = _find_bags(session_dir)
    poses, image_stamps, image_topic = _read_index(bag_paths)
    fps = _infer_fps(image_stamps)

    export_dir = session_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    pose_csv = export_dir / "pose.csv"
    alignment_csv = export_dir / "frame_pose.csv"
    video_path = export_dir / "rgb.mp4"

    _write_pose_csv(pose_csv, poses)
    _write_alignment_csv(
        alignment_csv, image_stamps, poses, max_delta_ms, max_interp_gap_ms
    )
    frame_count = _write_video(video_path, bag_paths, fps, image_topic)
    if frame_count != len(image_stamps):
        raise RuntimeError(
            f"video frame count {frame_count} differs from indexed images {len(image_stamps)}"
        )
    return {
        "session_dir": str(session_dir),
        "bag_paths": [str(path) for path in bag_paths],
        "pose_count": len(poses),
        "frame_count": frame_count,
        "image_topic": image_topic,
        "fps": fps,
        "max_delta_ms": float(max_delta_ms),
        "max_interp_gap_ms": float(max_interp_gap_ms),
        "video_path": str(video_path),
        "pose_csv": str(pose_csv),
        "alignment_csv": str(alignment_csv),
    }
