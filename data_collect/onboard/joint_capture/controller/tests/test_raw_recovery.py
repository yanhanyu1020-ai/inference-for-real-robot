import hashlib
import json
import struct
import numpy as np
import pytest
from test_raw_reader import episode_v2


def truncate_second_chunk(path):
    data=path.read_bytes()
    at=8
    chunks=0
    while at<len(data)-8:
        opcode,length=struct.unpack_from('<BQ',data,at)
        if opcode==6:
            chunks+=1
            if chunks==2:
                path.write_bytes(data[:at+9+length//2])
                return
        at+=9+length
    raise AssertionError('two chunks required')


def test_truncated_tail_recovers_complete_chunks_only(tmp_path):
    from recover_capture import recover_episode
    from episode_io import iter_camera_frames
    source=episode_v2(tmp_path/'source')
    front=next((source/'raw/cameras').glob('front*.mcap'))
    truncate_second_chunk(front)
    report=recover_episode(source,tmp_path/'recovered')
    assert len(list(iter_camera_frames(tmp_path/'recovered','front')))==1
    assert len(list(iter_camera_frames(tmp_path/'recovered','wrist')))==2
    assert not report['source_complete']
    assert report['streams']['front']['lost_upper_bound']=='unknown'
    assert report['errors']


def test_crc_failure_reports_loss(tmp_path):
    from recover_capture import recover_episode
    source=episode_v2(tmp_path/'source')
    path=next((source/'raw/cameras').glob('front*.mcap'))
    data=bytearray(path.read_bytes())
    at=data.index(np.full((3,4,3),2,np.uint8).tobytes())
    data[at]^=255
    path.write_bytes(data)
    report=recover_episode(source,tmp_path/'recovered')
    assert any('crc' in error.lower() for error in report['errors'])
    assert report['streams']['front']['recovered']==1


def test_recovery_never_modifies_source(tmp_path):
    from recover_capture import recover_episode
    source=episode_v2(tmp_path/'source')
    paths=list(source.rglob('*'))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    recover_episode(source,tmp_path/'recovered')
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    assert before==after
    with pytest.raises(FileExistsError): recover_episode(source,tmp_path/'recovered')
    with pytest.raises(ValueError): recover_episode(source,source/'nested')


def test_recovery_does_not_claim_complete_when_source_declares_loss(tmp_path):
    from recover_capture import recover_episode
    source=episode_v2(tmp_path/'source')
    summary=json.loads((source/'summary.json').read_text())
    summary['streams']['front']['rejected']=1
    (source/'summary.json').write_text(json.dumps(summary))
    report=recover_episode(source,tmp_path/'recovered')
    assert not report['source_complete']


def test_recovery_csv_sync_failure_does_not_claim_durability(tmp_path,monkeypatch):
    import os
    import recover_capture
    source=episode_v2(tmp_path/'source')
    (source/'raw/state.csv').write_text('monotonic_ns,value\n1,2\n')
    original=os.fsync
    # Cross-platform: track the specific destination descriptor by wrapping open.
    from pathlib import Path
    open_original=Path.open
    descriptors=set()
    def open_file(path,*args,**kwargs):
        handle=open_original(path,*args,**kwargs)
        if path==tmp_path/'recovered/raw/state.csv': descriptors.add(handle.fileno())
        return handle
    def sync(fd):
        if fd in descriptors: raise OSError('state fsync failed')
        return original(fd)
    monkeypatch.setattr(Path,'open',open_file)
    monkeypatch.setattr(os,'fsync',sync)
    with pytest.raises(OSError,match='state fsync failed'):
        recover_capture.recover_episode(source,tmp_path/'recovered')
    meta=tmp_path/'recovered/meta.json'
    assert not meta.exists() or not json.loads(meta.read_text()).get('durable_complete')
