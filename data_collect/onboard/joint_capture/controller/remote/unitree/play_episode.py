#!/usr/bin/env python3
import argparse
import pickle
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np


CAMERA_NAMES = ("front", "wrist")


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play the synchronized front and wrist images in a collected episode."
    )
    parser.add_argument("--episode", required=True, type=Path, help="episode directory")
    parser.add_argument(
        "--fps",
        type=positive_float,
        help="override the playback or exported video frame rate",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="write a side-by-side MP4 instead of opening a window")
    mode.add_argument("--check", action="store_true", help="decode all frames without displaying them")
    return parser.parse_args()


def frame_timestamp(path):
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"frame filename is not a timestamp: {path.name}") from exc


def find_frames(episode):
    frames_dir = episode.expanduser().resolve() / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")
    frames = sorted(frames_dir.glob("*.pkl"), key=frame_timestamp)
    if not frames:
        raise FileNotFoundError(f"no PKL frames found in: {frames_dir}")
    return frames


def infer_fps(frames):
    intervals = [
        (frame_timestamp(current) - frame_timestamp(previous)) / 1e9
        for previous, current in zip(frames, frames[1:])
    ]
    intervals = [interval for interval in intervals if interval > 0]
    return 1.0 / statistics.median(intervals) if intervals else 30.0


def decode_png(value, frame_path, camera_name):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"{frame_path}: camera.{camera_name}.rgb is {type(value).__name__}, expected PNG bytes"
        )
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{frame_path}: failed to decode camera.{camera_name}.rgb")
    return image


def read_frame(path):
    # Episode PKLs are trusted collector output; pickle must not be used with untrusted files.
    with path.open("rb") as handle:
        record = pickle.load(handle)
    try:
        images = [decode_png(record["camera"][name]["rgb"], path, name) for name in CAMERA_NAMES]
        timestamp_ns = int(record["timestamp_ns"])
        frame_index = int(record["frame_index"])
    except KeyError as exc:
        raise KeyError(f"{path}: missing field {exc}") from exc
    return timestamp_ns, frame_index, images


def resize_to_common_height(images):
    height = min(image.shape[0] for image in images)
    result = []
    for image in images:
        if image.shape[0] != height:
            width = round(image.shape[1] * height / image.shape[0])
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        result.append(image)
    return result


def compose_frame(images, frame_index, timestamp_ns):
    labeled = []
    for name, image in zip(CAMERA_NAMES, resize_to_common_height(images)):
        image = image.copy()
        cv2.putText(image, name, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, name, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(image)
    canvas = np.hstack(labeled)
    text = f"frame={frame_index}  time={timestamp_ns / 1e9:.3f}"
    cv2.putText(canvas, text, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def check_frames(frames, fps):
    shape = None
    for path in frames:
        timestamp_ns, frame_index, images = read_frame(path)
        canvas = compose_frame(images, frame_index, timestamp_ns)
        if shape is None:
            shape = canvas.shape
        elif canvas.shape != shape:
            raise ValueError(f"{path}: composed image shape changed from {shape} to {canvas.shape}")
    print(f"decoded {len(frames)} frames, shape={shape}, inferred_fps={fps:.3f}")


def export_video(frames, output, fps):
    output = output.expanduser().resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")
    writer = None
    try:
        for path in frames:
            timestamp_ns, frame_index, images = read_frame(path)
            canvas = compose_frame(images, frame_index, timestamp_ns)
            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {output}")
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    print(f"wrote {len(frames)} frames at {fps:.3f} FPS: {output}")


def wait_for_key(delay_ms):
    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    if key == ord("q"):
        return False
    if key == ord(" "):
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                break
    return True


def play_frames(frames, fps_override):
    cv2.namedWindow("heterovla episode", cv2.WINDOW_NORMAL)
    try:
        for index, path in enumerate(frames):
            started = time.monotonic()
            timestamp_ns, frame_index, images = read_frame(path)
            cv2.imshow("heterovla episode", compose_frame(images, frame_index, timestamp_ns))
            if fps_override is not None:
                interval_s = 1.0 / fps_override
            elif index + 1 < len(frames):
                interval_s = max(0.001, (frame_timestamp(frames[index + 1]) - frame_timestamp(path)) / 1e9)
            else:
                interval_s = 1.0 / 30.0
            elapsed_s = time.monotonic() - started
            if not wait_for_key(round(max(0.001, interval_s - elapsed_s) * 1000)):
                break
    finally:
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        frames = find_frames(args.episode)
        inferred_fps = infer_fps(frames)
        fps = args.fps or inferred_fps
        if args.check:
            check_frames(frames, inferred_fps)
        elif args.output:
            export_video(frames, args.output, fps)
        else:
            print(f"playing {len(frames)} frames at approximately {fps:.3f} FPS; Space=pause, q=quit")
            play_frames(frames, args.fps)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
