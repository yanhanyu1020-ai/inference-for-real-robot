"""Export strict V3 training episodes with the established LeRobot pipeline."""
import json
from pathlib import Path
import shutil
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from alignment_core import FPS, HORIZON, TASK, CAMERAS, CAM_KEYS, segments, encode_h264, features, dump

def export_datasets(source, out, aligned, splits=('train',)):
    import cv2
    from lerobot.common.datasets import lerobot_dataset as ld
    ld.encode_video_frames=encode_h264
    manifest=[]
    for split in splits:
        root=out/'lerobot/local'/f'heterovla_v3_{split}'
        if root.exists():
            ds=ld.LeRobotDataset(f'local/heterovla_v3_{split}',root=root,video_backend='pyav')
            assert ds.fps==FPS and ds.meta.total_episodes>0
            ds.episode_buffer=ds.create_episode_buffer()
            ds.start_image_writer(num_threads=6)
        else:
            ds=ld.LeRobotDataset.create(repo_id=f'local/heterovla_v3_{split}',fps=FPS,root=root,
                robot_type='go2_piper_p450_joint_kinematic_v3',features=features(),use_videos=True,
                image_writer_threads=6,video_backend='pyav')
        completed=ds.meta.total_episodes
        # Temporary PNGs stay on local scratch; only release artifacts go to shared storage.
        scratch=Path(tempfile.mkdtemp(prefix=f'heterovla-15hz-{split}-',dir='/tmp'))
        def image_path(self,episode_index,image_key,frame_index):
            return scratch/f'episode_{episode_index:06d}'/image_key/f'frame_{frame_index:06d}.png'
        ds._get_image_file_path=types.MethodType(image_path,ds)
        def encode_cameras(self,episode_index):
            paths={key:str(self.root/self.meta.get_video_file_path(episode_index,key)) for key in self.meta.video_keys}
            def encode(key):
                path=Path(paths[key])
                if not path.exists():
                    images=self._get_image_file_path(episode_index,key,0).parent
                    encode_h264(images,path,self.fps)
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(encode,self.meta.video_keys))
            return paths
        ds.encode_episode_videos=types.MethodType(encode_cameras,ds)
        planned_index=0
        for task_rows,report in aligned:
            if report['split']!=split:continue
            e=source/report['source_path'];runs=segments(task_rows)
            if not runs:continue
            pending=[]
            for run in runs:
                ep=planned_index;planned_index+=1
                if ep>=completed:
                    pending.append(run);continue
                # Resume only completed episodes with numerically identical newly aligned labels.
                data=pq.read_table(root/ds.meta.get_data_file_path(ep)).to_pydict()
                assert np.array_equal(np.asarray(data['observation.state'],np.float32),np.asarray([r['state'] for r in run],np.float32))
                assert np.array_equal(np.asarray(data['action'],np.float32),np.asarray([r['action'] for r in run],np.float32))
                assert np.array_equal(np.asarray(data['source_task_time'],np.float32).reshape(-1),np.asarray([r['source_task_time_s'] for r in run],np.float32))
                provenance_path=out/'provenance'/f'{split}_{ep:06d}.parquet'
                if provenance_path.exists():
                    previous=pq.read_table(provenance_path).to_pylist()
                    assert previous==run
                else:
                    # Metadata is committed after all videos, but an interruption may precede provenance writing.
                    pq.write_table(pa.Table.from_pylist(run),provenance_path,compression='zstd')
                manifest.append(dict(split=split,episode_index=ep,source_episode=report['source_episode'],
                    source_path=report['source_path'],source_session=report['source_session'],
                    source_start_s=run[0]['source_task_time_s'],source_last_s=run[-1]['source_task_time_s'],
                    frames=len(run),valid_h15_windows=len(run)-HORIZON+1))
                print('RESUMED',split,ep,report['source_episode'],len(run),flush=True)
            runs=pending
            if not runs:continue
            wanted=[set(r['camera_video_frames'][i] for run in runs for r in run if r['camera_video_frames'][i]>=0) for i in range(3)]
            raw_images={}
            for ci,cam in enumerate(CAMERAS):
                selected={r.get('camera_raw_frames',[-1]*3)[ci] for run in runs for r in run}
                selected.discard(-1)
                if selected:
                    from raw_camera import selected_raw_frames
                    raw_images[ci]=selected_raw_frames(e,cam,selected)
            # Decode once per source/camera in temporal order. Sparse selected frames only.
            frames=[]
            for ci,cam in enumerate(CAMERAS):
                cap=cv2.VideoCapture(str(e/'videos'/f'{cam}.mp4'));bank={};i=0
                while True:
                    ok,img=cap.read()
                    if not ok:break
                    if i in wanted[ci]:bank[i]=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    i+=1
                cap.release()
                if set(bank)!=wanted[ci]:raise ValueError(f'Missing decoded video frames: {e.name}/{cam}')
                frames.append(bank)
            for run in runs:
                ep=ds.meta.total_episodes
                provenance=[]
                for row in run:
                    frame={'task':TASK,'observation.state':np.asarray(row['state'],np.float32),
                        'action':np.asarray(row['action'],np.float32),'action_valid':np.asarray(row['action_valid'],bool),
                        'source_task_time':np.array([row['source_task_time_s']],np.float32)}
                    for ci,key in enumerate(CAM_KEYS):
                        raw_index=row.get('camera_raw_frames',[-1]*3)[ci]
                        frame[key]=raw_images[ci][raw_index] if raw_index>=0 else frames[ci][row['camera_video_frames'][ci]]
                    ds.add_frame(frame)
                    provenance.append(row)
                ds.save_episode()
                shutil.rmtree(scratch/f'episode_{ep:06d}')
                record=dict(split=split,episode_index=ep,source_episode=report['source_episode'],source_path=report['source_path'],source_session=report['source_session'],
                    source_start_s=run[0]['source_task_time_s'],source_last_s=run[-1]['source_task_time_s'],
                    frames=len(run),valid_h15_windows=max(0,len(run)-HORIZON+1))
                manifest.append(record)
                pq.write_table(pa.Table.from_pylist(provenance),out/'provenance'/f'{split}_{ep:06d}.parquet',compression='zstd')
                print('EXPORTED',record,flush=True)
            del frames
        ds.stop_image_writer()
        shutil.rmtree(scratch)
        assert planned_index==ds.meta.total_episodes
    dump(out/'reports/episode_manifest.json',manifest)
    return manifest


def norm_stats(out):
    arrays={'state':[],'actions':[]}
    for f in sorted((out/'lerobot/local/heterovla_v3_train/data').rglob('*.parquet')):
        d=pq.read_table(f).to_pydict()
        arrays['state'].extend(d['observation.state']);arrays['actions'].extend(d['action'])
    stats={}
    for k,v in arrays.items():
        x=np.asarray(v,dtype=np.float64)
        stats[k]=dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),q01=np.quantile(x,.01,axis=0).tolist(),
            q99=np.quantile(x,.99,axis=0).tolist())
    dump(out/'assets/heterovla_v3/norm_stats.json',{'norm_stats':stats})


