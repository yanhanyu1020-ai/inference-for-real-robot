import io
import json
import pickle
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'remote/unitree'))
from test_raw_image import meta
from mcap_storage import CameraStore,durable_json


def episode_v2(directory, count=2):
    directory.mkdir(parents=True,exist_ok=True)
    durable_json(directory/'meta.json',{'format_version':2,'image_storage':'mcap','durable_complete':True})
    streams={}
    for camera in ('front','wrist'):
        store=CameraStore(directory/'raw/cameras',camera,'none',chunk_size=1)
        for seq in range(1,count+1):
            metadata={**meta(seq),'camera':camera,'monotonic_ns':seq*100_000_000}
            store.append(metadata,np.full((3,4,3),seq,np.uint8))
        streams[camera]={**store.close(),'received':count,'accepted':count,'rejected':0,'write_errors':0,'pending':0}
    durable_json(directory/'summary.json',{'streams':streams,'camera_durable_complete':True,'durable_complete':True,'quality_ok':True})
    return directory


@pytest.mark.parametrize('damage', ['index', 'footer_crc'])
def test_summary_crc_corruption_is_not_complete(tmp_path, damage):
    import struct
    from episode_io import inspect_episode
    from recover_capture import recover_episode
    source=episode_v2(tmp_path/'source')
    path=next((source/'raw/cameras').glob('front*.mcap'))
    data=bytearray(path.read_bytes())
    if damage=='footer_crc':
        data[-12] ^= 1
    else:
        offset=8
        while offset<len(data)-8:
            opcode,length=struct.unpack_from('<BQ',data,offset)
            if opcode==8:  # ChunkIndex, after message_start_time/message_end_time
                data[offset+9+16] ^= 1
                break
            offset+=9+length
        else: raise AssertionError('index required')
    path.write_bytes(data)
    report=inspect_episode(source)
    assert not report['application_integrity']
    assert any('summary CRC' in error for error in report['errors'])
    recovery=recover_episode(source,tmp_path/'recovered')
    assert not recovery['source_complete']
    assert recovery['streams']['front']['recovered']==2


def test_v1_and_v2_read_same_pixels(tmp_path):
    from episode_io import iter_camera_frames
    new=episode_v2(tmp_path/'new')
    old=tmp_path/'old'
    (old/'frames').mkdir(parents=True)
    for seq in (1,2):
        output=io.BytesIO()
        Image.fromarray(np.full((3,4,3),seq,np.uint8)).save(output,format='PNG')
        record={'timestamp_ns':seq,'frame_index':seq-1,'camera':{'front':{
            'serial':'123','wall_time_ns':seq,'monotonic_ns':seq*100_000_000,'rgb':output.getvalue()}}}
        (old/'frames'/f'{seq}.pkl').write_bytes(pickle.dumps(record))
    left=list(iter_camera_frames(old,'front'))
    right=list(iter_camera_frames(new,'front'))
    assert len(left)==len(right)==2
    assert all(np.array_equal(a[1],b[1]) for a,b in zip(left,right))


def test_corrupt_v2_never_falls_back_to_pkl(tmp_path):
    from episode_io import iter_camera_frames
    directory=episode_v2(tmp_path)
    next((directory/'raw/cameras').glob('front*.mcap')).write_bytes(b'broken')
    (directory/'frames').mkdir()
    (directory/'frames/one.pkl').write_bytes(pickle.dumps({'camera':{}}))
    with pytest.raises(Exception): list(iter_camera_frames(directory,'front'))


def test_boot_change_rejected_within_episode(tmp_path):
    from episode_io import iter_camera_frames
    from mcap.writer import Writer,CompressionType
    from raw_image import encode_image,ENCODING
    directory=episode_v2(tmp_path)
    path=next((directory/'raw/cameras').glob('front*.mcap'))
    with path.open('wb') as f:
        w=Writer(f,chunk_size=1,compression=CompressionType.NONE)
        w.start()
        ch=w.register_channel('/camera/front',ENCODING,0)
        for seq,boot in ((1,'one'),(2,'two')):
            m={**meta(seq),'boot_id':boot}
            w.add_message(ch,m['wall_time_ns'],encode_image(m,np.zeros((3,4,3),np.uint8)),m['wall_time_ns'])
        w.finish()
    with pytest.raises(ValueError,match='boot'): list(iter_camera_frames(directory,'front'))


def test_counts_include_queue_rejections(tmp_path):
    from episode_io import inspect_episode
    directory=episode_v2(tmp_path)
    summary=json.loads((directory/'summary.json').read_text())
    summary['streams']['front'].update(received=3,rejected=1)
    durable_json(directory/'summary.json',summary)
    report=inspect_episode(directory)
    assert report['raw_closed']
    assert not report['application_integrity']
    assert report['streams']['front']['rejected']==1
    assert report['alignment_valid'] is None
    assert report['sensor_integrity_unknown']


def test_wall_clock_backwards_keeps_sequence_order(tmp_path):
    from episode_io import iter_camera_frames
    directory=tmp_path
    durable_json(directory/'meta.json',{'format_version':2,'image_storage':'mcap'})
    store=CameraStore(directory/'raw/cameras','front','lz4',chunk_size=1)
    for seq in (1,2):
        store.append({**meta(seq),'wall_time_ns':10-seq},np.zeros((3,4,3),np.uint8))
    store.close()
    assert [m['seq'] for m,_ in iter_camera_frames(directory,'front')]==[1,2]


def test_v2_validator_uses_camera_streams_and_raw_monotonic_times(tmp_path):
    from validate_episode import validate,REQUIRED_RAW_FILES
    directory=episode_v2(tmp_path)
    for name in REQUIRED_RAW_FILES:
        (directory/'raw'/name).write_text('monotonic_ns,wall_time_ns\n100000000,200\n200000000,100\n')
    report=validate(directory,{'cameras':{'front':'123','wrist':'123'},'fps':10,'max_camera_skew_ms':50},False)
    assert report['ok']
    assert report['alignment_valid'] is None
    assert report['raw_closed'] and report['application_integrity']
    (directory/'raw/piper_state.csv').write_text('monotonic_ns,wall_time_ns\n900000000,100\n1000000000,200\n')
    report=validate(directory,{'cameras':{'front':'123','wrist':'123'},'fps':10,'max_camera_skew_ms':50},False)
    assert not report['ok']
    assert any('cover' in e for e in report['errors'])


def test_oversized_chunk_is_rejected_before_decompression():
    from mcap.records import Chunk
    from episode_io import chunk_records,MAX_CHUNK_BYTES
    chunk=Chunk(compression='lz4',data=b'not even compressed',message_start_time=0,
                message_end_time=0,uncompressed_crc=0,uncompressed_size=MAX_CHUNK_BYTES+1)
    with pytest.raises(ValueError,match='size limit'): chunk_records(chunk)


def test_inspector_rejects_wrong_episode_boot_id(tmp_path):
    from episode_io import inspect_episode
    directory=episode_v2(tmp_path)
    metadata=json.loads((directory/'meta.json').read_text())
    durable_json(directory/'meta.json',{**metadata,'boot_id':'other-boot'})
    assert not inspect_episode(directory)['application_integrity']


def test_quick_timing_summary_never_opens_image_files(tmp_path,monkeypatch):
    from episode_io import summarize_timing
    directory=episode_v2(tmp_path)
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.suffix=='.mcap': raise AssertionError('quick timing must not read images')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=summarize_timing(directory)
    assert [s['name'] for s in result]==['camera:front','camera:wrist']
    assert [s['count'] for s in result]==[2,2]
