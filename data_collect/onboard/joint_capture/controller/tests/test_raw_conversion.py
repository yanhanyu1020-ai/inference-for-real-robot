import csv
import io
import json
import pickle
import shutil
import numpy as np
from PIL import Image
import pytest
from test_raw_reader import episode_v2


def state_fixture(directory,enabled=True):
    common={'monotonic_ns':100000000,'wall_time_ns':100,'seq':1}
    inertial={**{k:0 for k in ('quat_w','quat_x','quat_y','quat_z','gyro_x','gyro_y','gyro_z',
                               'accel_x','accel_y','accel_z','roll','pitch','yaw')},
              **{'foot_force_'+str(i):0 for i in range(4)}}
    joint={**{'joint_'+str(i)+'.pos':0 for i in range(1,7)},'gripper.pos':0}
    rows={
        'piper_state':{**common,**joint},
        'piper_status':{**common,'ctrl_mode':1,'arm_status':0,'teach_status':0},
        'piper_gamepad':{**common,**joint,'arm_enabled':int(enabled),'arm_connected':1,'gamepad_connected':1,
            'command_mode':0,'movement_speed':1,'speed_factor':.1,'command_sent':1,'up_level_mode':'arm','low_level_mode':'joint'},
        'sport_mode_state':{**common,**inertial,**{k:0 for k in ('robot_sec','robot_nanosec','error_code','mode','gait_type','progress','foot_raise_height','body_height','yaw_speed')},
            **{v+'_'+a:0 for v in ('position','velocity') for a in 'xyz'},**{'range_'+str(i):0 for i in range(4)}},
        'low_state':{**common,**inertial,'tick':1,'power_v':25,'power_a':1,
            **{'foot_force_est_'+str(i):0 for i in range(4)},
            **{'motor_'+str(i)+'_'+v:0 for i in range(20) for v in ('q','dq','tau_est')}}}
    for name,row in rows.items():
        with (directory/'raw'/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            writer.writerow({**row,'seq':2,'monotonic_ns':200000000})


CONFIG={'max_camera_skew_ms':50,'piper_stale_ms':250,'go2_stale_ms':500,'piper_gamepad_stale_ms':1000,'fps':10}


def test_png_pkl_roundtrip_matches_raw_pixels(tmp_path):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    report=convert_episode(source,tmp_path/'export',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==2
    with next((tmp_path/'export/frames').glob('*.pkl')).open('rb') as f: record=pickle.load(f)
    with Image.open(io.BytesIO(record['camera']['front']['rgb'])) as img:
        assert np.all(np.asarray(img)==1)
    assert record['camera']['front']['monotonic_ns']==100000000
    assert not report['training_ready']


def test_asymmetric_rates_use_unique_nearest_frames():
    from convert_episode import pair_frames
    fronts=[({'monotonic_ns':t,'seq':i},None) for i,t in enumerate([100,150,200,300],1)]
    wrists=[({'monotonic_ns':t,'seq':i},None) for i,t in enumerate([90,110,290],1)]
    pairs=list(pair_frames(fronts,wrists,100))
    assert [w[0]['seq'] if w else None for _,w in pairs]==[1,2,3,None]


def test_wall_jump_pairs_by_monotonic():
    from convert_episode import pair_frames
    f=[({'monotonic_ns':100,'wall_time_ns':1000},None),({'monotonic_ns':200,'wall_time_ns':10},None)]
    w=[({'monotonic_ns':105,'wall_time_ns':1005},None),({'monotonic_ns':205,'wall_time_ns':15},None)]
    assert [b[0]['monotonic_ns'] for _,b in pair_frames(f,w,10)]==[105,205]


@pytest.mark.parametrize('missing', [True,False])
def test_missing_state_or_home_enable_rejects_training_sample(tmp_path,missing):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    if not missing: state_fixture(source,enabled=False)
    report=convert_episode(source,tmp_path/'export',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==0
    assert report['rejected']==2
    assert not report['training_ready']


def test_invalid_clock_mapping_does_not_claim_alignment(tmp_path):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    mapping=tmp_path/'alignment.json'
    mapping.write_text(json.dumps({'quality':{'degraded':True},'valid_interval':None}))
    with pytest.raises(ValueError,match='invalid training alignment mapping'):
        convert_episode(source,tmp_path/'export',target='pkl',mapping=mapping,config=CONFIG)
    assert not (tmp_path/'export').exists()


def test_video_export_keeps_timestamp_sidecar(tmp_path):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    report=convert_episode(source,tmp_path/'video',target='video',mapping=None,config=CONFIG)
    assert (tmp_path/'video/front.mp4').stat().st_size>0
    with (tmp_path/'video/front.timestamps.csv').open() as f: rows=list(csv.DictReader(f))
    assert [int(r['monotonic_ns']) for r in rows]==[100000000,200000000]
    assert report['streams']['front']['frames']==2


def test_malformed_numeric_state_is_not_a_training_sample(tmp_path):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    path=source/'raw/piper_state.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames
        rows=list(reader)
    for row in rows: row['joint_1.pos']='not-a-number'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    report=convert_episode(source,tmp_path/'export',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==0


def test_cartesian_command_without_raw_target_pose_is_rejected(tmp_path):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    path=source/'raw/piper_gamepad.csv'
    path.write_text(path.read_text().replace(',joint\n',',cartesian\n'))
    with path.open(newline='') as f:
        assert next(csv.DictReader(f))['low_level_mode']=='cartesian'
    report=convert_episode(source,tmp_path/'export',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==0


def test_valid_mapping_restores_integer_nanoseconds(tmp_path):
    from convert_episode import ClockMapping
    source=episode_v2(tmp_path/'source')
    meta=json.loads((source/'meta.json').read_text())
    (source/'meta.json').write_text(json.dumps({**meta,'episode_id':'test'}))
    path=tmp_path/'alignment.json'
    first=1780000000000000000
    path.write_text(json.dumps({'episode_id':'test','quality':{'degraded':False,'data_validated':True,'clock_estimated_error_ns':1000},
        'valid_interval':{'start_inclusive_ns':first-100,'end_exclusive_ns':first+10000},
        'mappings':{'unitree':{'host':'unitree','mapping_type':'piecewise_linear_offset','anchors':[
            {'remote_wall_ns':first,'offset_ns':100},{'remote_wall_ns':first+10000,'offset_ns':200}],
            'unsupported_gaps':[]}}}))
    clock=ClockMapping(path,source)
    assert clock.valid
    assert clock.restore(first+5000)==first+4850
    assert clock.restore(first-1) is None


def test_cartesian_pose_fields_survive_csv_and_conversion(tmp_path):
    from piper_gamepad_teleop import pose_columns
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    path=source/'raw/piper_gamepad.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    pose=pose_columns([.1,.2,.3,10,20,30])
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+list(pose))
        writer.writeheader()
        writer.writerows({**row,'low_level_mode':'cartesian',**pose} for row in rows)
    report=convert_episode(source,tmp_path/'export',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==2
    with next((tmp_path/'export/frames').glob('*.pkl')).open('rb') as f: record=pickle.load(f)
    assert record['piper']['command']['target_pose']==[.1,.2,.3,10,20,30]
