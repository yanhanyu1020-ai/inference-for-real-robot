"""Read-only real-data experiment; exports timestamp indexes, not training media."""
import bisect
import csv
import hashlib
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import sys

from jointctl.alignment import build_clock_timeline
from jointctl.clock_sync import read_clock_records
from jointctl.models import EpisodeManifest
from jointctl.remote import run_ssh


def restore(mapping, timestamp):
    anchors = mapping['anchors']
    if len(anchors) < 2:
        raise ValueError('need at least two clock anchors')
    if not anchors[0]['remote_wall_ns'] <= timestamp <= anchors[-1]['remote_wall_ns']:
        raise ValueError('outside saved mapping anchors')
    for gap in mapping['unsupported_gaps']:
        if gap['remote_start_ns'] < timestamp < gap['remote_end_ns']:
            raise ValueError('unsupported clock gap')
    right = min(max(bisect.bisect_left([a['remote_wall_ns'] for a in anchors], timestamp), 1), len(anchors)-1)
    a, b = anchors[right-1], anchors[right]
    offset = a['offset_ns'] + round(Fraction(
        (b['offset_ns']-a['offset_ns'])*(timestamp-a['remote_wall_ns']),
        b['remote_wall_ns']-a['remote_wall_ns']))
    return timestamp-offset


COMMON = '''import os,sys,json,glob,re
root=sys.argv[1]; out=[]
def key(path): return [int(x) if x.isdigit() else x for x in re.split(r'(\\d+)',path)]
def add(stream,path,index,ns):
 if isinstance(ns,bool) or not isinstance(ns,int) or ns<=0: raise ValueError('invalid timestamp')
 out.append([stream,os.path.relpath(path,root),index,ns])
'''
SCRIPTS = {
 'p450': COMMON + '''import rosbag
topics=['/uav1/camera/color/image_raw/compressed','/Odometry']
for path in sorted(glob.glob(root+'/**/*.bag',recursive=True),key=key):
 counts={}
 with rosbag.Bag(path) as bag:
  for topic,msg,t in bag.read_messages(topics=topics):
   index=counts.get(topic,0); counts[topic]=index+1
   ns=int(msg.header.stamp.to_nsec())
   if abs(ns-int(t.to_nsec()))>1000000000: raise ValueError('header clock domain mismatch')
   add(topic,path,index,ns)
print(json.dumps(out,separators=(',',':')))
''',
 'unitree': COMMON + '''import pickle,csv
for path in sorted(glob.glob(root+'/**/*.pkl',recursive=True),key=key):
 with open(path,'rb') as f: record=pickle.load(f)
 for camera,metadata in record['camera'].items(): add('pkl:'+camera,path,0,metadata['wall_time_ns'])
for path in sorted(glob.glob(root+'/raw/*.csv')):
 with open(path,newline='') as f:
  for index,row in enumerate(csv.DictReader(f)):
   add('raw/'+os.path.basename(path),path,index,int(row['wall_time_ns']))
print(json.dumps(out,separators=(',',':')))
''',
}


def main(episode, output):
    output.mkdir(parents=True, exist_ok=False)
    report_path = episode/'alignment.json'
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report['quality']['degraded'] or not report['valid_interval']:
        raise ValueError('episode did not pass alignment validation')
    manifest = EpisodeManifest.from_dict(json.loads((episode/'manifest.json').read_text()))
    start = report['valid_interval']['start_inclusive_ns']
    end = report['valid_interval']['end_exclusive_ns']
    def fetch(host):
        command = 'python3 -c '+shlex.quote(SCRIPTS[host])+' '+shlex.quote(manifest.remote_directories[host])
        if host == 'p450':
            command = 'bash -c '+shlex.quote('source /opt/ros/noetic/setup.bash && '+command)
        result = run_ssh(host, command, 300)
        if not result.ok: raise RuntimeError(host+': '+result.stderr)
        rows = json.loads(result.stdout)
        (output/(host+'_source_timestamps.json')).write_text(json.dumps(rows),encoding='utf-8')
        print(host, 'raw timestamp rows:',len(rows),flush=True)
        return host,rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = dict(pool.map(fetch, ('p450','unitree')))
    samples = read_clock_records(episode,manifest)[0]
    stats = {}; kept={}; maximum_difference = 0
    with (output/'restored_timestamp_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['host','stream','relative_file','record_index','original_ns','desktop_ns','episode_relative_ns'])
        for host,rows in sources.items():
            mapping=report['mappings'][host]
            rebuilt=build_clock_timeline(samples[host],5_000_000_000,max_gap_ns=mapping['max_clock_gap_ns'])
            for stream,file,index,ns in rows:
                name=host+':'+stream
                stat=stats.setdefault(name,dict(total=0,inside_interval=0,outside_anchors=0,nonincreasing=0))
                stat['total']+=1
                try: desktop=restore(mapping,ns)
                except ValueError:
                    stat['outside_anchors']+=1
                    continue
                expected=rebuilt.remote_to_desktop(ns)
                if expected.degraded: raise ValueError('rebuilt mapping degraded')
                maximum_difference=max(maximum_difference,abs(desktop-expected.desktop_time_ns))
                if start<=desktop<end:
                    previous=kept.setdefault(name,[])
                    if previous and desktop<=previous[-1][0]: stat['nonincreasing']+=1
                    previous.append((desktop,file,index))
                    stat['inside_interval']+=1
                    writer.writerow([host,stream,file,index,ns,desktop,desktop-start])
    required=['p450:/uav1/camera/color/image_raw/compressed','p450:/Odometry','unitree:pkl:front','unitree:pkl:wrist']
    if any(not kept.get(name) for name in required): raise ValueError('missing restored camera/pose stream')
    pair_stats={}
    with (output/'camera_pairing_example.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['p450_ns','p450_bag','p450_message_index','target_stream','target_ns','target_file','target_index','difference_ns'])
        for name in required[1:]:
            candidates=kept[name]; stamps=[r[0] for r in candidates]; differences=[]
            for stamp,file,index in kept[required[0]]:
                pos=bisect.bisect_left(stamps,stamp)
                choices=candidates[max(0,pos-1):min(len(candidates),pos+1)]
                target=min(choices,key=lambda r:abs(r[0]-stamp))
                difference=target[0]-stamp; differences.append(abs(difference))
                writer.writerow([stamp,file,index,name,*target,difference])
            pair_stats[name]={'pairs':len(differences),'max_abs_delta_ms':max(differences)/1e6,
                              'mean_abs_delta_ms':sum(differences)/len(differences)/1e6}
    result={'episode_id':manifest.episode_id,'valid_duration_s':(end-start)/1e9,
            'mapping_sha256':hashlib.sha256(report_bytes).hexdigest(),
            'saved_vs_rebuilt_max_difference_ns':maximum_difference,'streams':stats,
            'nearest_pairing_example':pair_stats,
            'limitations':['Timestamp indexes only; no media copied or training actions transformed.',
                          'Nearest sample deltas are not measured physical synchronization error.',
                          'Rows outside saved anchors are excluded; raw data remains unchanged.']}
    assert maximum_difference == 0
    assert all(s['nonincreasing']==0 for s in stats.values())
    assert report_path.read_bytes()==report_bytes
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
