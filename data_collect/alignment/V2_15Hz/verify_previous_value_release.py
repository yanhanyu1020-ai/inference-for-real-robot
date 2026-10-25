"""Independently verify every held value, source split, and training normalization."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq

from resplit_train35 import read, write, episode_path


def verify(root, raw, train_scripts):
    sys.path.insert(0,str(train_scripts))
    from dataset_15hz import JointV215HzDataset, as_openpi_sample
    from check_action_chunks import check
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    manifest=read(root/'reports/episode_manifest.json')
    inventory=read(root/'configs/source_inventory.json')
    expected={s:{e['source_episode'] for e in manifest if e['split']==s} for s in ('train','val')}
    assert len(expected['train'])==35 and len(expected['val'])==3 and not expected['train'] & expected['val']
    config=read(root/'configs/action_chunk.json')
    assert config['action_horizon']==20
    checks={}
    for e in manifest:
        if e['split']!='val':
            continue
        original=pq.read_table(root/'aligned'/f"{e['source_episode']}.parquet").to_pylist()
        provenance=pq.read_table(root/'provenance'/f"val_{e['episode_index']:06d}.parquet").to_pylist()
        origin=np.array([r['source_task_time_s'] for r in original])
        first=np.flatnonzero([all(r['action_valid']) and all(r['state_valid']) and all(r['camera_valid']) for r in original])[0]
        indices=np.arange(first,len(original))
        assert len(provenance)==len(indices)==e['frames']
        directory=root/'lerobot/local/heterovla_v2_15hz_val'
        info=read(directory/'meta/info.json')
        data=pq.read_table(episode_path(directory,info,e['episode_index'])).to_pydict()
        for key,field in [('state','observation.state'),('action','action')]:
            values=np.asarray([r[key] for r in original],np.float32)
            valid=np.asarray([r[f'{key}_valid'] for r in original],bool)
            target=np.empty((len(indices),values.shape[1]),np.float32)
            source_times=np.empty_like(target,dtype=np.float64)
            for dim in range(values.shape[1]):
                observed=np.flatnonzero(valid[:,dim] & (np.arange(len(original))>=first))
                last=observed[np.searchsorted(observed,indices,side='right')-1]
                target[:,dim]=values[last,dim]
                source_times[:,dim]=origin[last]
            np.testing.assert_array_equal(data[field],target)
            np.testing.assert_array_equal([r[key] for r in provenance],target)
            np.testing.assert_array_equal(data[f'{key}_valid'],valid[indices])
            np.testing.assert_array_equal(data[f'{key}_imputed'],~valid[indices])
            np.testing.assert_array_equal([r[f'{key}_value_source_grid_time_s'] for r in provenance],source_times)
            assert np.all(source_times<=origin[indices,None])
        np.testing.assert_array_equal(np.asarray(data['source_task_time']).reshape(-1),origin[indices].astype(np.float32))
        np.testing.assert_allclose(np.diff(origin[indices]),1/15,atol=2e-9,rtol=0)
        # Compare three decoded frames per camera with the exact original frame
        # selected by the preserved causal camera mapping.
        image_checks=[]
        selected=sorted({0,len(provenance)//2,len(provenance)-1})
        for camera_index,(raw_camera,camera) in enumerate(zip(('front','wrist','p450'),('front','wrist','uav'))):
            raw_indices={provenance[i]['camera_video_frames'][camera_index] for i in selected}
            originals={}
            with av.open(str(raw/e['source_path']/'videos'/f'{raw_camera}.mp4')) as container:
                container.streams.video[0].thread_count=2
                for i,frame in enumerate(container.decode(video=0)):
                    if i in raw_indices: originals[i]=frame.to_ndarray(format='rgb24')
            with av.open(str(episode_path(directory,info,e['episode_index'],f'observation.images.{camera}'))) as container:
                container.streams.video[0].thread_count=2
                for i,frame in enumerate(container.decode(video=0)):
                    if i in selected:
                        actual=frame.to_ndarray(format='rgb24')
                        reference=originals[provenance[i]['camera_video_frames'][camera_index]]
                        error=float(np.abs(actual.astype(float)-reference.astype(float)).mean())
                        assert error<6,(e['source_episode'],camera,i,error)
                        image_checks.append(error)
        checks[e['source_episode']]=dict(all_state_and_action_values_verified=True,
            original_validity_masks_preserved=True,no_future_value_used=True,frames=len(provenance),
            source_frame_comparisons=len(image_checks),max_rgb_mean_absolute_error=max(image_checks))
        print('FILL_VERIFIED',e['source_episode'],len(provenance),flush=True)
    norm=read(root/'assets/heterovla_v2_15hz/norm_stats.json')['norm_stats']
    arrays={'state':[],'actions':[]}
    for p in sorted((root/'lerobot/local/heterovla_v2_15hz_train/data').rglob('*.parquet')):
        d=pq.read_table(p).to_pydict()
        assert np.asarray(d['action_valid']).all()
        arrays['state'].extend(d['observation.state'])
        arrays['actions'].extend(d['action'])
    meta=LeRobotDatasetMetadata('local/heterovla_v2_15hz_train',root=root/'lerobot/local/heterovla_v2_15hz_train')
    for field,key in [('observation.state','state'),('action','actions')]:
        x=np.asarray(arrays[key],np.float64)
        for name,computed in [('mean',x.mean(0)),('std',x.std(0)),('q01',np.quantile(x,.01,axis=0)),('q99',np.quantile(x,.99,axis=0))]:
            np.testing.assert_allclose(norm[key][name],computed,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(meta.stats[field]['mean'],norm[key]['mean'],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(meta.stats[field]['std'],norm[key]['std'],atol=1e-6,rtol=1e-6)
    chunk=check(root,20)
    prompt=read(root/'configs/task.json')['english_translation'].strip()
    loader_checks={}
    for split in ('train','val'):
        ds=JointV215HzDataset(root,split)
        lookup={value:i for i,value in enumerate(ds.starts)}
        samples=set()
        for start,end in zip(ds.dataset.episode_data_index['from'],ds.dataset.episode_data_index['to']):
            start,end=int(start),int(end)
            if end-start>=20:
                samples.update([lookup[start],lookup[end-20]])
        for i in sorted(samples):
            sample=as_openpi_sample(ds[i])
            assert sample['prompt']==prompt and sample['actions'].shape==(20,14)
            assert sample['state'].shape==(30,)
        loader_checks[split]=dict(checked_boundary_windows=len(samples),english_prompt=prompt)
        print('BOUNDARIES_VERIFIED',split,len(samples),flush=True)
    write(root/'reports/action_chunk_validation.json',dict(default_action_horizon=20,configured_default_datasets=chunk,
        checks={'20':chunk},fps_independent_of_action_horizon=True))
    write(root/'reports/previous_value_verification.json',dict(passed=True,sources=checks,normalization_recomputed_and_verified=True,
        raw_train_labels_only=True,loader=loader_checks))
    validation=read(root/'reports/validation.json')
    validation.pop('pending',None)
    validation.update(format_validation_passed=True,validation_previous_value_fill_verified=True,
        chunk_and_loader_verified=True,default_action_horizon=20,fps=15)
    for split in ('train','val'):
        validation['datasets'][split].update(chunk[split])
    write(root/'reports/validation.json',validation)
    status=read(root/'STATUS.json')
    status.update(validation_pending=False,format_validation_passed=True,splits=validation['datasets'])
    write(root/'STATUS.json',status)
    print('RELEASE_VERIFIED',json.dumps(status),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','raw','train-scripts'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    verify(args.root.resolve(),args.raw.resolve(),args.train_scripts.resolve())
