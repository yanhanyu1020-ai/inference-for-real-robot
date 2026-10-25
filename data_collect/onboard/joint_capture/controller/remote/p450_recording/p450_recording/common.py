import bisect
import math
import re


RAW_IMAGE_TOPIC = "/uav1/camera/color/image_raw"
COMPRESSED_IMAGE_TOPIC = RAW_IMAGE_TOPIC + "/compressed"

BASE_REQUIRED_TOPICS = {
    "/uav1/camera/color/camera_info": "sensor_msgs/CameraInfo",
    "/Odometry": "nav_msgs/Odometry",
    "/uav1/mavros/state": "mavros_msgs/State",
    "/uav1/mavros/battery": "sensor_msgs/BatteryState",
}

REQUIRED_TOPICS = {
    COMPRESSED_IMAGE_TOPIC: "sensor_msgs/CompressedImage",
    **BASE_REQUIRED_TOPICS,
}

RAW_REQUIRED_TOPICS = {
    RAW_IMAGE_TOPIC: "sensor_msgs/Image",
    **BASE_REQUIRED_TOPICS,
}

OPTIONAL_TOPICS = {
    "/uav1/mavros/local_position/pose": "geometry_msgs/PoseStamped",
    "/uav1/mavros/local_position/velocity_local": "geometry_msgs/TwistStamped",
    "/uav1/mavros/imu/data": "sensor_msgs/Imu",
    "/uav1/mavros/global_position/global": "sensor_msgs/NavSatFix",
    "/uav1/prometheus/command": "prometheus_msgs/UAVCommand",
    "/uav1/mavros/rc/in": "mavros_msgs/RCIn",
    "/uav1/mavros/rc/out": "mavros_msgs/RCOut",
    "/uav1/mavros/manual_control/control": "mavros_msgs/ManualControl",
    "/uav1/mavros/setpoint_raw/local": "mavros_msgs/PositionTarget",
    "/uav1/mavros/setpoint_raw/target_local": "mavros_msgs/PositionTarget",
    "/uav1/mavros/timesync_status": "mavros_msgs/TimesyncStatus",
    "/uav1/mavros/extended_state": "mavros_msgs/ExtendedState",
    "/uav1/prometheus/control_state": "prometheus_msgs/UAVControlState",
    "/uav1/prometheus/state": "prometheus_msgs/UAVState",
}


def capture_topics(raw_rgb=False):
    return RAW_REQUIRED_TOPICS if raw_rgb else REQUIRED_TOPICS

SESSION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def validate_session_name(value):
    if not isinstance(value, str) or SESSION_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "session name must be 1-64 characters using only letters, numbers, '_' or '-'"
        )
    return value


def stamp_to_ns(stamp):
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)


def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def nearest_pose(poses, target_ns, max_delta_ms=50.0):
    if not poses:
        raise ValueError("cannot match a pose from an empty sequence")
    timestamps = [item[0] for item in poses]
    index = bisect.bisect_left(timestamps, target_ns)
    if index == 0:
        match = poses[0]
    elif index == len(poses):
        match = poses[-1]
    else:
        before = poses[index - 1]
        after = poses[index]
        match = before if target_ns - before[0] <= after[0] - target_ns else after
    delta_ms = abs(match[0] - target_ns) / 1_000_000.0
    return match[0], match[1], delta_ms, delta_ms <= max_delta_ms


def _normalized_pose(pose):
    try:
        xyz = [float(pose[key]) for key in ("x", "y", "z")]
        quat = [float(pose[key]) for key in ("qx", "qy", "qz", "qw")]
        if not all(math.isfinite(value) for value in xyz + quat):
            return None
        norm = math.hypot(*quat)
        if norm == 0.0 or not math.isfinite(norm):
            return None
        quat = [value / norm for value in quat]
        return {
            "frame_id": pose["frame_id"],
            **dict(zip(("x", "y", "z"), xyz)),
            **dict(zip(("qx", "qy", "qz", "qw"), quat)),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _result(
    method, target_ns, pose_stamp_ns, delta_ms, pose, valid,
    prev_stamp=None, next_stamp=None, alpha=None,
):
    return {
        "pose_stamp_ns": pose_stamp_ns,
        "aligned_pose_stamp_ns": (
            target_ns if method in ("exact", "interpolated", "nearest_boundary") else ""
        ),
        "delta_ms": delta_ms,
        "valid": valid,
        "interp_valid": method == "interpolated" and valid,
        "alignment_method": method,
        "prev_pose_stamp_ns": prev_stamp if prev_stamp is not None else "",
        "next_pose_stamp_ns": next_stamp if next_stamp is not None else "",
        "prev_delta_ms": (
            abs(target_ns - prev_stamp) / 1_000_000.0
            if prev_stamp is not None else ""
        ),
        "next_delta_ms": (
            abs(next_stamp - target_ns) / 1_000_000.0
            if next_stamp is not None else ""
        ),
        "alpha": alpha if alpha is not None else "",
        "pose": pose,
    }


def interpolate_pose(
    poses, target_ns, max_delta_ms=50.0, max_interp_gap_ms=200.0
):
    if not poses:
        raise ValueError("cannot interpolate a pose from an empty sequence")
    if max_delta_ms < 0 or max_interp_gap_ms < 0:
        raise ValueError("pose alignment thresholds must be non-negative")

    timestamps = [item[0] for item in poses]
    index = bisect.bisect_left(timestamps, target_ns)

    if index < len(poses) and timestamps[index] == target_ns:
        normalized = None
        for exact_index in range(
            index, bisect.bisect_right(timestamps, target_ns, lo=index)
        ):
            normalized = _normalized_pose(poses[exact_index][1])
            if normalized is not None:
                break
        return _result(
            "exact" if normalized is not None else "invalid",
            target_ns, target_ns, 0.0, normalized, normalized is not None,
            target_ns, target_ns, 0.0,
        )

    if index == 0 or index == len(poses):
        endpoint = poses[0] if index == 0 else poses[-1]
        delta_ms = abs(endpoint[0] - target_ns) / 1_000_000.0
        normalized = _normalized_pose(endpoint[1])
        valid = delta_ms <= max_delta_ms and normalized is not None
        return _result(
            "nearest_boundary" if valid else "invalid",
            target_ns, endpoint[0], delta_ms, normalized if valid else None, valid,
            endpoint[0] if index == len(poses) else None,
            endpoint[0] if index == 0 else None,
            0.0 if valid else None,
        )

    previous = poses[index - 1]
    following = poses[index]
    prev_stamp, next_stamp = previous[0], following[0]
    prev_pose = _normalized_pose(previous[1])
    next_pose = _normalized_pose(following[1])
    nearest_stamp = (
        prev_stamp
        if target_ns - prev_stamp <= next_stamp - target_ns
        else next_stamp
    )
    nearest_delta_ms = abs(nearest_stamp - target_ns) / 1_000_000.0
    gap_ms = (next_stamp - prev_stamp) / 1_000_000.0
    alpha = (
        (target_ns - prev_stamp) / float(next_stamp - prev_stamp)
        if next_stamp > prev_stamp else None
    )
    valid = (
        next_stamp > prev_stamp
        and gap_ms <= max_interp_gap_ms
        and prev_pose is not None
        and next_pose is not None
        and prev_pose["frame_id"] == next_pose["frame_id"]
    )
    if not valid:
        return _result(
            "invalid", target_ns, nearest_stamp, nearest_delta_ms, None, False,
            prev_stamp, next_stamp, alpha,
        )

    position = {
        key: prev_pose[key] + alpha * (next_pose[key] - prev_pose[key])
        for key in ("x", "y", "z")
    }
    keys = ("qx", "qy", "qz", "qw")
    q0 = [prev_pose[key] for key in keys]
    q1 = [next_pose[key] for key in keys]
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = [-value for value in q1]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        quat = [a + alpha * (b - a) for a, b in zip(q0, q1)]
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        quat = [
            math.sin((1.0 - alpha) * theta) / sin_theta * a
            + math.sin(alpha * theta) / sin_theta * b
            for a, b in zip(q0, q1)
        ]
    quat_norm = math.hypot(*quat)
    quat = [value / quat_norm for value in quat]
    interpolated = {
        "frame_id": prev_pose["frame_id"],
        **position,
        **dict(zip(keys, quat)),
    }
    return _result(
        "interpolated", target_ns, nearest_stamp, nearest_delta_ms, interpolated,
        True, prev_stamp, next_stamp, alpha,
    )


def bag_sort_key(path):
    """Order historical unpadded rosbag split suffixes numerically."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", path.name))


def padded_bag_path(path):
    match = re.fullmatch(r"flight_(\d+)\.bag", path.name)
    return path.with_name("flight_{:06d}.bag".format(int(match.group(1)))) if match else path


def pending_bags(directory):
    """Retain interrupted originals once a nonempty finalized copy exists."""
    return [path for path in sorted(directory.glob("*.bag.active"), key=bag_sort_key)
            if not any(target.is_file() and target.stat().st_size > 0
                       for target in (path.with_suffix(""), padded_bag_path(path.with_suffix(""))))]


def normalize_bag_names(directory):
    """rosbag generates unpadded suffixes; normalize completed shards at finalization."""
    changes = {}
    for source in sorted(directory.glob("*.bag"), key=bag_sort_key):
        target = padded_bag_path(source)
        if target != source:
            if target.exists():
                raise RuntimeError("duplicate bag shard: {} and {}".format(source, target))
            source.rename(target)
            changes[source] = target
    return changes
