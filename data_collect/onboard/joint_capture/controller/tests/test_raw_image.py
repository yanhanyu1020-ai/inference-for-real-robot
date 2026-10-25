import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'remote/unitree'))


def meta(seq=1, wall=1780000000000000001):
    return dict(camera='front', serial='123', seq=seq, reader_seq=100+seq,
                width=4, height=3, stride=12, dtype='uint8', pixel_format='bgr8',
                monotonic_ns=1000000000+seq, wall_time_ns=wall,
                boot_id='test-boot', timestamp_source='host_receive',
                device_timestamp_ns=None, device_frame_number=None)


def test_pixels_round_trip():
    from raw_image import encode_image, decode_image
    image = np.arange(72, dtype=np.uint8).reshape(3, 8, 3)[:, ::2]
    before = meta()
    actual, restored = decode_image(encode_image(before, image))
    assert actual == before
    np.testing.assert_array_equal(restored, image)


def test_integer_ns_survives_roundtrip():
    from raw_image import encode_image, decode_image
    actual, _ = decode_image(encode_image(meta(), np.zeros((3,4,3), dtype=np.uint8)))
    assert actual['wall_time_ns'] == 1780000000000000001


@pytest.mark.parametrize('mutation', ['truncated','long_header','negative','stride','float_ns','dtype','pixel_bytes'])
def test_rejects_malformed_size(mutation):
    from raw_image import decode_image
    value = meta()
    if mutation == 'negative': value['height'] = -3
    if mutation == 'stride': value['stride'] = 13
    if mutation == 'float_ns': value['wall_time_ns'] = 1.2
    if mutation == 'dtype': value['dtype'] = 'float32'
    header = json.dumps(value).encode()
    payload = struct.pack('<I', len(header)) + header + bytes(36)
    if mutation == 'truncated': payload = payload[:3]
    if mutation == 'long_header': payload = struct.pack('<I', 65537) + payload[4:]
    if mutation == 'pixel_bytes': payload = payload[:-1]
    with pytest.raises(ValueError): decode_image(payload)


def test_encode_rejects_wrong_type_and_shape():
    from raw_image import encode_image
    for image in [np.zeros((3,4,3),dtype=np.float32), np.zeros((4,4,3),dtype=np.uint8)]:
        with pytest.raises(ValueError): encode_image(meta(), image)


def test_wall_clock_step_does_not_change_sequence():
    from raw_image import encode_image, decode_image
    image = np.zeros((3,4,3), dtype=np.uint8)
    outputs = [decode_image(encode_image(meta(i, wall), image))[0]
               for i,wall in [(1,100),(2,50)]]
    assert [x['seq'] for x in outputs] == [1,2]
    assert [x['wall_time_ns'] for x in outputs] == [100,50]
