"""Versioned BGR8 records. Host reception times are not exposure timestamps."""
import json
import struct

import numpy as np

ENCODING = 'heterovla.raw_image.v1'
MAX_METADATA = 64 * 1024
MAX_PAYLOAD = 64 * 1024 * 1024


def image_schema():
    return {'encoding': ENCODING, 'layout': ['uint32_le metadata_bytes',
            'UTF-8 JSON metadata', 'contiguous uint8 BGR pixels'],
            'max_metadata_bytes': MAX_METADATA, 'max_payload_bytes': MAX_PAYLOAD,
            'required_metadata': ['camera', 'serial', 'seq', 'reader_seq', 'width',
                'height', 'stride', 'dtype', 'pixel_format', 'monotonic_ns',
                'wall_time_ns', 'boot_id', 'timestamp_source',
                'device_timestamp_ns', 'device_frame_number']}


def _validate(meta):
    if not isinstance(meta, dict) or any(k not in meta for k in image_schema()['required_metadata']):
        raise ValueError('missing raw image metadata')
    for key in ('camera', 'serial', 'boot_id'):
        if not isinstance(meta[key], str) or not meta[key]:
            raise ValueError('invalid ' + key)
    for key in ('seq', 'reader_seq', 'width', 'height', 'stride', 'monotonic_ns', 'wall_time_ns'):
        if type(meta[key]) is not int or not 0 <= meta[key] < 2**64:
            raise ValueError('invalid integer ' + key)
    if not meta['width'] or not meta['height'] or meta['stride'] != meta['width'] * 3:
        raise ValueError('invalid dimensions or stride')
    if meta['dtype'] != 'uint8' or meta['pixel_format'] != 'bgr8':
        raise ValueError('only uint8 bgr8 is supported')
    if meta['timestamp_source'] != 'host_receive':
        raise ValueError('unsupported timestamp source')
    if meta['device_timestamp_ns'] is not None or meta['device_frame_number'] is not None:
        raise ValueError('host_receive records must not invent device metadata')
    size = meta['height'] * meta['stride']
    if size > MAX_PAYLOAD:
        raise ValueError('image too large')
    return size


def encode_image(metadata, image):
    size = _validate(metadata)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (metadata['height'], metadata['width'], 3):
        raise ValueError('image type or dimensions do not match metadata')
    header = json.dumps(metadata, ensure_ascii=False, allow_nan=False,
                        separators=(',', ':')).encode('utf-8')
    if len(header) > MAX_METADATA or 4 + len(header) + size > MAX_PAYLOAD:
        raise ValueError('record too large')
    return struct.pack('<I', len(header)) + header + image.tobytes(order='C')


def decode_image(payload):
    if not 4 <= len(payload) <= MAX_PAYLOAD:
        raise ValueError('invalid payload length')
    length = struct.unpack_from('<I', payload)[0]
    if length > MAX_METADATA or 4 + length > len(payload):
        raise ValueError('invalid metadata length')
    try:
        metadata = json.loads(payload[4:4+length].decode('utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError('invalid metadata JSON') from exc
    size = _validate(metadata)
    if len(payload) != 4 + length + size:
        raise ValueError('pixel byte count does not match dimensions')
    pixels = np.frombuffer(payload, dtype=np.uint8, offset=4+length)
    return metadata, pixels.reshape(metadata['height'], metadata['width'], 3)
