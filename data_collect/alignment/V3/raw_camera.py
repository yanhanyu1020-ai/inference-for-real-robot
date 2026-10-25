"""Read CRC-verified raw camera frames using the capture project's reader."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def selected_raw_frames(source, camera, indices):
    if camera not in ('front','wrist'):
        raise ValueError('Raw MCAP selection is supported for front/wrist cameras')
    for parent in Path(__file__).resolve().parents:
        candidates=[parent/'data_collect/onboard/joint_capture/controller/remote/unitree',
                    parent/'heterovla/data_collect/onboard/joint_capture/controller/remote/unitree']
        directory=next((p for p in candidates if (p/'episode_io.py').is_file()),None)
        if directory is not None:
            break
    else:
        raise FileNotFoundError('Capture episode_io.py is required for raw MCAP decoding')
    sys.path.insert(0,str(directory))
    from episode_io import iter_mcap_frames
    mapping=pd.read_csv(source/'timing'/f'{camera}_raw_frames.csv')
    requested=set(map(int,indices));images={}
    count=0
    # Exhaust the iterator so its chunk and footer CRC checks all execute.
    for index,(metadata,bgr) in enumerate(iter_mcap_frames(source/'raw/cameras'/f'{camera}.mcap',camera)):
        row=mapping.iloc[index]
        assert int(row.raw_frame)==index
        assert metadata['seq']==int(row.seq) and metadata['monotonic_ns']==int(row.monotonic_ns)
        assert metadata['wall_time_ns']==int(row.wall_time_ns)
        if index in requested:
            images[index]=np.ascontiguousarray(bgr[:,:,::-1])
        count+=1
    assert count==len(mapping) and set(images)==requested
    return images
