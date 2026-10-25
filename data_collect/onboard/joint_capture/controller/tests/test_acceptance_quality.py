from dataclasses import replace
import json

from jointctl.cli import build_alignment_report
from jointctl.inspectors import StreamSummary
from jointctl.models import ClockSample, EpisodeManifest, EpisodeState
from jointctl.manifest import ManifestStore


def episode(tmp_path):
    base=10**12
    store=ManifestStore(tmp_path)
    store.create(EpisodeManifest('quality', 'test', 'joint', base, EpisodeState.RECORDING,
                                 t0_desktop_ns=base+2*10**9, t1_desktop_ns=base+8*10**9))
    folder=store.episode_dir('quality')
    for host in ('p450','unitree'):
        rows=[]
        for i in range(101):
            t=base+i*100_000_000
            rows.append(ClockSample.from_exchange(host=host,sequence=i,local_send_wall_ns=t,
                local_send_mono_ns=t,local_receive_wall_ns=t+1_000_000,local_receive_mono_ns=t+1_000_000,
                remote_receive_wall_ns=t+500_000,remote_send_wall_ns=t+500_000,remote_monotonic_ns=t+500_000).to_dict())
        (folder/f'clock_{host}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    names={'p450':['/uav1/camera/color/image_raw/compressed','/Odometry','/uav1/mavros/imu/data'],
           'unitree':['pkl:front','pkl:wrist','raw/sport_mode_state.csv','raw/low_state.csv',
                      'raw/piper_state.csv','raw/piper_status.csv','raw/piper_gamepad.csv']}
    streams={h:[StreamSummary(n,151,base+500_000,base+10**10+500_000,
                'camera.wall_time_ns' if n.startswith('pkl:') else ('ros_header' if h=='p450' else 'wall_time_ns'),
                h,100_000_000,0) for n in ns] for h,ns in names.items()}
    return folder, streams


def test_complete_required_streams_pass_acceptance(tmp_path):
    folder, streams=episode(tmp_path)
    r=build_alignment_report(folder,streams)
    assert r['quality']['data_validated']
    assert r['valid_interval'] is not None


def test_no_cameras_cannot_be_a_valid_capture(tmp_path):
    folder, streams=episode(tmp_path)
    streams={h:[s for s in ss if 'camera' not in s.name and not s.name.startswith('pkl:')] for h,ss in streams.items()}
    r=build_alignment_report(folder,streams)
    assert r['valid_interval'] is None
    assert r['quality']['degraded']
    assert not r['quality']['data_validated']


def test_recorder_drop_counter_withdraws_joint_data_acceptance(tmp_path):
    folder, streams = episode(tmp_path)
    streams['unitree'][0] = replace(streams['unitree'][0], producer_dropped=245)
    streams['unitree'][1] = replace(streams['unitree'][1], producer_dropped=245)
    report = build_alignment_report(folder, streams)
    assert report['quality']['data_validated'] is False
    assert report['valid_interval'] is None
    assert any('245' in error and 'dropped' in error for error in report['quality']['data_errors'])
    assert sum('dropped' in error for error in report['quality']['data_errors']) == 1


def test_two_endpoint_frames_do_not_prove_continuous_coverage(tmp_path):
    folder,streams=episode(tmp_path)
    streams['unitree'][0]=replace(streams['unitree'][0],count=2,max_gap_ns=10**10)
    r=build_alignment_report(folder,streams)
    assert r['valid_interval'] is None
    assert any('gap' in reason for reason in r['quality']['data_errors'])


def test_missing_gap_evidence_is_not_assumed_continuous(tmp_path):
    folder,streams=episode(tmp_path)
    streams['unitree'][0]=replace(streams['unitree'][0],max_gap_ns=None)
    assert build_alignment_report(folder,streams)['valid_interval'] is None


def test_bad_clock_with_complete_data_withdraws_valid_interval(tmp_path):
    folder,streams=episode(tmp_path)
    # One anchor cannot prove drift over the episode.
    for host in ('p450', 'unitree'):
        path=folder/f'clock_{host}.jsonl'
        path.write_text(path.read_text().splitlines()[0]+'\n')
    report=build_alignment_report(folder,streams)
    assert report['quality']['degraded']
    assert report['valid_interval'] is None


def test_failed_model_replaces_stale_success_report(tmp_path, monkeypatch):
    from jointctl.cli import write_alignment_report
    folder,streams=episode(tmp_path)
    (folder/'alignment.json').write_text('{"valid_interval": {"old": true}}')
    def fail(*args, **kwargs):
        raise ValueError('clock model inconsistent with probe sequence 34')
    monkeypatch.setattr('jointctl.cli.build_alignment_report',fail)
    output=write_alignment_report(folder,stream_summaries=streams)
    report=json.loads(output.read_text())
    assert report['valid_interval'] is None
    assert report['quality']['degraded']
    assert 'inconsistent' in report['quality']['degradation_reasons'][0]
