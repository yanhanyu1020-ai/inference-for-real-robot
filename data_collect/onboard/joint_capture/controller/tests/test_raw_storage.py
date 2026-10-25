from pathlib import Path
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'remote/unitree'))
from test_raw_image import meta


def rows(path):
    from mcap.reader import NonSeekingReader
    from raw_image import decode_image
    with path.open('rb') as f:
        return [decode_image(m.data) for _,_,m in NonSeekingReader(f, validate_crcs=True).iter_messages(log_time_order=False)]


def test_crc_roundtrip_and_rotation(tmp_path):
    from mcap_storage import CameraStore
    a = CameraStore(tmp_path, 'front', 'lz4', chunk_size=1, shard_size=1000)
    for seq in range(1,6):
        a.append(meta(seq), np.full((3,4,3), seq, np.uint8))
    result = a.close()
    paths = sorted(tmp_path.glob('front_*.mcap'))
    assert len(paths) > 1
    records = [row for p in paths for row in rows(p)]
    assert [m['seq'] for m,_ in records] == [1,2,3,4,5]
    assert all(np.all(img == m['seq']) for m,img in records)
    assert result['durable'] == 5
    assert not list(tmp_path.glob('*.active'))


def test_durable_advances_only_after_fsync(tmp_path):
    from mcap_storage import CameraStore
    fail = [True]
    def fsync(fd):
        if fail[0]: raise OSError('disk sync failed')
    a = CameraStore(tmp_path,'front','none',chunk_size=1,io_hooks={'fsync':fsync})
    a.append(meta(),np.zeros((3,4,3),np.uint8))
    assert a.stats()['written'] == 1
    assert a.stats()['durable'] == 0
    with pytest.raises(OSError): a.checkpoint()
    assert a.stats()['durable'] == 0
    fail[0] = False
    assert a.close()['durable'] == 1


def test_failed_rename_keeps_active_file(tmp_path):
    from mcap_storage import CameraStore
    def rename(src,dst): raise OSError('rename failed')
    a=CameraStore(tmp_path,'front','lz4',io_hooks={'rename':rename})
    a.append(meta(),np.zeros((3,4,3),np.uint8))
    with pytest.raises(OSError): a.close()
    assert list(tmp_path.glob('*.active'))
    assert a.stats()['closed'] is False


def test_checkpoint_without_new_frame(tmp_path):
    from mcap_storage import CameraStore
    a=CameraStore(tmp_path,'front','lz4')
    a.append(meta(),np.zeros((3,4,3),np.uint8))
    assert a.checkpoint()['durable'] == 0  # buffered partial chunk, not on disk
    assert a.close()['durable'] == 1
    assert a.close()['durable'] == 1


def test_existing_file_never_overwritten(tmp_path):
    from mcap_storage import CameraStore
    p=tmp_path/'front_000001.mcap'
    p.write_bytes(b'original')
    a=CameraStore(tmp_path,'front','none')
    with pytest.raises(FileExistsError): a.append(meta(),np.zeros((3,4,3),np.uint8))
    assert p.read_bytes() == b'original'


def test_directory_sync_failure_is_not_closed(tmp_path):
    from mcap_storage import CameraStore
    def sync(path): raise OSError('directory sync')
    a=CameraStore(tmp_path,'front','none',io_hooks={'sync_dir':sync})
    a.append(meta(),np.zeros((3,4,3),np.uint8))
    with pytest.raises(OSError): a.close()
    assert not a.stats()['closed']


def test_wrong_camera_or_sequence_rejected(tmp_path):
    from mcap_storage import CameraStore
    a=CameraStore(tmp_path,'front','none')
    a.append(meta(),np.zeros((3,4,3),np.uint8))
    with pytest.raises(ValueError): a.append(meta(),np.zeros((3,4,3),np.uint8))
    with pytest.raises(ValueError): a.append({**meta(2),'camera':'wrist'},np.zeros((3,4,3),np.uint8))
    a.close()


def test_failed_finish_latches_fault_and_preserves_partial(tmp_path, monkeypatch):
    from mcap_storage import CameraStore
    a = CameraStore(tmp_path, 'front', 'none')
    a.append(meta(), np.zeros((3,4,3), np.uint8))
    def fail():
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(a._writer, 'finish', fail)
    with pytest.raises(OSError): a.close()
    assert a.stats()['fault']
    assert a.stats()['durable'] == 0
    with pytest.raises(RuntimeError, match='recovery'): a.close()
    assert list(tmp_path.glob('*.active'))
    assert not list(tmp_path.glob('*.mcap'))


def test_crc_detects_pixel_corruption(tmp_path):
    from mcap_storage import CameraStore
    a = CameraStore(tmp_path, 'front', 'none', chunk_size=1)
    pixels = np.arange(36, dtype=np.uint8).reshape(3,4,3)
    a.append(meta(), pixels)
    a.close()
    path = next(tmp_path.glob('*.mcap'))
    content = bytearray(path.read_bytes())
    offset = content.index(pixels.tobytes())
    content[offset + 10] ^= 255
    path.write_bytes(content)
    with pytest.raises(ValueError): rows(path)


def test_cached_timing_summary_retains_wall_jumps(tmp_path):
    from mcap_storage import CameraStore
    a=CameraStore(tmp_path,'front','none')
    for seq,wall in enumerate((100,150,120),1):
        a.append({**meta(seq),'wall_time_ns':wall},np.zeros((3,4,3),np.uint8))
    report=a.close()['wall_summary']
    assert report=={'count':3,'first_ns':100,'last_ns':150,'max_gap_ns':50,'nonmonotonic_count':1}
