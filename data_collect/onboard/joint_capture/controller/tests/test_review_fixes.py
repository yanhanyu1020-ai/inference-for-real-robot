"""Regressions for stale CAN feedback, unsent actions and common-interval trimming."""
import csv
import json
import pickle
import sys
from types import SimpleNamespace

import pytest
from test_raw_conversion import state_fixture, CONFIG
from test_raw_reader import episode_v2


class FakeSDK:
    instances=[]

    def __init__(self,*args,**kwargs):
        self.instances.append(self)
        self.joints=SimpleNamespace(time_stamp=100.,joint_state=SimpleNamespace(
            **{'joint_'+str(i):i*1000 for i in range(1,7)}))
        self.gripper=SimpleNamespace(time_stamp=101.,gripper_state=SimpleNamespace(grippers_angle=10000))
        self.status=SimpleNamespace(time_stamp=102.,arm_status=SimpleNamespace(ctrl_mode=1,arm_status=0,teach_status=0))

    def ParseCANFrame(self,frame): pass
    def GetArmJointMsgs(self): return self.joints
    def GetArmGripperMsgs(self): return self.gripper
    def GetArmStatus(self): return self.status
    def isOk(self): return True

    def ConnectPort(self,*,piper_init):
        assert piper_init is False
        for key in self.feedback_ids:
            self.ParseCANFrame(can_frame(key))


def can_frame(key,**kwargs):
    return SimpleNamespace(arbitration_id=key,data=bytes(8),is_error_frame=False,
                           is_remote_frame=False,is_extended_id=False,**kwargs)


@pytest.fixture
def simulated_clock(monkeypatch):
    import hetero_pkl_recorder as recorder
    now=[1_000_000_000]
    clock=SimpleNamespace(monotonic_ns=lambda:now[0],time_ns=lambda:now[0]+100_000_000_000,
                          monotonic=lambda:now[0]/1e9,sleep=lambda seconds:None)
    monkeypatch.setattr(recorder,'time',clock)
    return recorder,now,clock


def feed_all(interface,now):
    for key in interface.feedback_ids:
        interface.ParseCANFrame(can_frame(key))
        now[0]+=1_000_000


def test_snapshot_uses_real_receive_times_and_retains_sdk_timestamps(simulated_clock):
    recorder,now,_=simulated_clock
    piper=recorder.feedback_piper(FakeSDK)('fake')
    feed_all(piper,now)
    first=piper.feedback_snapshot()
    now[0]+=100_000_000
    second=piper.feedback_snapshot()
    assert first[3:]==second[3:]
    assert first[3][0]<now[0]
    assert first[0].time_stamp==100.
    piper.joints.joint_state.joint_1=0
    assert first[0].joint_state.joint_1==1000


@pytest.mark.parametrize('frozen',[0x2A1,0x2A5,0x2A6,0x2A7,0x2A8])
def test_single_frozen_feedback_pair_cannot_hide_behind_other_updates(simulated_clock,frozen):
    recorder,now,_=simulated_clock
    piper=recorder.feedback_piper(FakeSDK)('fake')
    feed_all(piper,now)
    now[0]+=5_000_000_000
    for key in piper.feedback_ids:
        if key!=frozen: piper.ParseCANFrame(can_frame(key))
    with pytest.raises(RuntimeError,match=hex(frozen)):
        piper.feedback_snapshot()


def test_reader_does_not_emit_new_rows_from_an_unchanged_cache(tmp_path,monkeypatch,simulated_clock):
    recorder,now,clock=simulated_clock
    monkeypatch.setitem(sys.modules,'piper_sdk',SimpleNamespace(C_PiperInterface_V2=FakeSDK))
    reader=recorder.PiperReader('fake',tmp_path)
    polls=[0]
    def sleep(seconds):
        # Skip the initial startup wait, then poll the same cache for 25 ms.
        if seconds>=.5: return
        polls[0]+=1
        now[0]+=5_000_000
        if polls[0]>=5: reader._stop.set()
    clock.sleep=sleep
    reader._run()
    assert reader.error is None
    assert reader.rows==1
    with (tmp_path/'piper_state.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==1
    assert float(rows[0]['sdk_joint_time_stamp'])==100.
    assert int(rows[0]['monotonic_ns'])==1_000_000_000


def test_reader_stops_writing_when_feedback_expires(tmp_path,monkeypatch,simulated_clock):
    recorder,now,clock=simulated_clock
    monkeypatch.setitem(sys.modules,'piper_sdk',SimpleNamespace(C_PiperInterface_V2=FakeSDK))
    def sleep(seconds):
        if seconds>=.5: return
        now[0]+=5_000_000_000
    clock.sleep=sleep
    reader=recorder.PiperReader('fake',tmp_path)
    reader._run()
    assert reader.rows==1
    assert 'stale' in reader.error


@pytest.mark.parametrize('sent',['0',None])
def test_unsent_or_legacy_unknown_command_is_not_exported(tmp_path,sent):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    path=source/'raw/piper_gamepad.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames); rows=list(reader)
    if sent is None: fields.remove('command_sent')
    for row in rows:
        if sent is None: row.pop('command_sent')
        else: row['command_sent']=sent
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report=convert_episode(source,tmp_path/'out',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==0 and report['data_rejected']==2
    assert not report['training_ready']


def aligned_fixture(tmp_path,interval=(180,220)):
    source=episode_v2(tmp_path/'source',count=3)
    meta=json.loads((source/'meta.json').read_text())
    (source/'meta.json').write_text(json.dumps({**meta,'episode_id':'aligned-test'}))
    state_fixture(source)
    for path in (source/'raw').glob('*.csv'):
        with path.open(newline='') as f:
            reader=csv.DictReader(f); fields=list(reader.fieldnames); rows=list(reader)
        rows.append({**rows[-1],'seq':'3','monotonic_ns':'300000000'})
        for i,row in enumerate(rows,1): row['wall_time_ns']=str(i*100)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    from episode_io import iter_camera_frames
    frames={camera:list(iter_camera_frames(source,camera)) for camera in ('front','wrist')}
    for values in frames.values():
        for i,(metadata,_) in enumerate(values,1): metadata['wall_time_ns']=i*100
    mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps({'episode_id':'aligned-test','quality':{
        'degraded':False,'data_validated':True,'clock_estimated_error_ns':1000},
        'valid_interval':{'start_inclusive_ns':interval[0],'end_exclusive_ns':interval[1]},
        'mappings':{'unitree':{'host':'unitree','mapping_type':'piecewise_linear_offset',
            'anchors':[{'remote_wall_ns':0,'offset_ns':0},{'remote_wall_ns':400,'offset_ns':0}],
            'unsupported_gaps':[]}}}))
    return source,mapping,frames


def test_common_interval_exports_only_middle_frame_and_remains_training_ready(tmp_path,monkeypatch):
    import convert_episode as converter
    source,mapping,frames=aligned_fixture(tmp_path)
    monkeypatch.setattr(converter,'iter_camera_frames',lambda _,camera:iter(frames[camera]))
    report=converter.convert_episode(source,tmp_path/'out',target='pkl',mapping=mapping,config=CONFIG)
    assert report['exported']==1 and report['rejected']==report['trimmed']==2
    assert report['data_rejected']==0
    assert report['alignment_valid'] and report['training_ready']
    with next((tmp_path/'out/frames').glob('*.pkl')).open('rb') as f: record=pickle.load(f)
    assert record['frame_index']==0 and record['camera']['front']['seq']==2
    assert record['alignment']['valid']
    assert len(list((source/'raw/cameras').glob('*.mcap')))>0


def test_empty_common_interval_never_claims_training_ready(tmp_path,monkeypatch):
    import convert_episode as converter
    source,mapping,frames=aligned_fixture(tmp_path,interval=(350,390))
    monkeypatch.setattr(converter,'iter_camera_frames',lambda _,camera:iter(frames[camera]))
    report=converter.convert_episode(source,tmp_path/'out',target='pkl',mapping=mapping,config=CONFIG)
    assert report['exported']==0 and report['trimmed']==3
    assert not report['training_ready']

def test_future_sent_command_cannot_label_an_earlier_image(tmp_path):
    from convert_episode import convert_episode
    source=episode_v2(tmp_path/'source')
    state_fixture(source)
    path=source/'raw/piper_gamepad.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames); rows=list(reader)
    rows[0].update(monotonic_ns='50000000',command_sent='0')
    rows[1].update(monotonic_ns='110000000',command_sent='1')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report=convert_episode(source,tmp_path/'out',target='pkl',mapping=None,config=CONFIG)
    assert report['exported']==1 and report['data_rejected']==1
    with next((tmp_path/'out/frames').glob('*.pkl')).open('rb') as f: record=pickle.load(f)
    assert record['camera']['front']['monotonic_ns']==200000000
    assert record['piper']['command']['monotonic_ns']==110000000
