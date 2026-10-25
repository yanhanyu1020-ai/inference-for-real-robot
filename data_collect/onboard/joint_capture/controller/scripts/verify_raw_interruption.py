"""Abrupt-exit recovery check; only the new child writer exits uncleanly."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from mcap_storage import CameraStore,durable_json
from recover_capture import recover_episode
from episode_io import iter_camera_frames


def hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def child(directory):
    output=directory/'raw/cameras';output.mkdir(parents=True)
    durable_json(directory/'meta.json',{'format_version':2,'image_storage':'mcap','boot_id':'interruption-test'})
    for name in ('front','wrist'):
        store=CameraStore(output,name,'lz4',chunk_size=1024)
        for seq in range(1,11):
            meta=dict(camera=name,serial='test',seq=seq,reader_seq=seq,width=4,height=3,
                stride=12,dtype='uint8',pixel_format='bgr8',monotonic_ns=seq,wall_time_ns=seq,
                boot_id='interruption-test',timestamp_source='host_receive',device_timestamp_ns=None,device_frame_number=None)
            store.append(meta,np.full((3,4,3),seq,np.uint8))
        store.checkpoint()
    os._exit(19)  # This child only; deliberately skip Python/MCAP cleanup.


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--child': child(Path(sys.argv[2]))
    root=Path(tempfile.mkdtemp(prefix='interruption-',dir=str(Path(__file__).resolve().parent)))
    source=root/'source'
    completed=subprocess.run([sys.executable,__file__,'--child',str(source)])
    assert completed.returncode==19
    before=hashes(source)
    report=recover_episode(source,root/'recovered')
    assert hashes(source)==before and not report['source_complete']
    counts={}
    for name in ('front','wrist'):
        frames=list(iter_camera_frames(root/'recovered',name));counts[name]=len(frames)
        assert frames and all(np.all(image==meta['source_seq']) for meta,image in frames)
    durable_json(root/'verification.json',{'source_hash_unchanged':True,'counts':counts,'report':report})
    print(json.dumps({'root':str(root),'counts':counts,'source_hash_unchanged':True,'loss_upper_bound':'unknown'}))
