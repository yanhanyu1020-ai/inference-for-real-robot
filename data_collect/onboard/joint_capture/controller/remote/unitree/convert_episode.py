"""Explicit offline conversion; never called by Start or Stop."""
import argparse
import bisect
from contextlib import ExitStack
import csv
from fractions import Fraction
from io import BytesIO
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess

from episode_io import iter_camera_frames,load_object,inspect_episode
from mcap_storage import durable_json


def pair_frames(fronts,wrists,max_skew_ns):
    wrists=iter(wrists)
    right=next(wrists,None)
    left=None
    for front in fronts:
        stamp=front[0]['monotonic_ns']
        while right is not None and right[0]['monotonic_ns']<=stamp:
            left,right=right,next(wrists,None)
        choices=[item for item in (left,right) if item is not None]
        chosen=min(choices,key=lambda item:(abs(item[0]['monotonic_ns']-stamp),item[0]['monotonic_ns'])) if choices else None
        if chosen is not None and abs(chosen[0]['monotonic_ns']-stamp)>max_skew_ns:
            chosen=None
        if chosen is right and chosen is not None:
            right=next(wrists,None)
            left=None
        elif chosen is left:
            left=None
        yield front,chosen


class StateCursor:
    def __init__(self,handle):
        self.rows=iter(csv.DictReader(handle))
        self.previous=None
        self.left=None
        self.right=self._next()

    def _next(self):
        row=next(self.rows,None)
        if row is None: return None
        result={}
        for key,value in row.items():
            if key is None or value is None: raise ValueError('malformed state CSV')
            if key.endswith('_ns') or key=='seq': result[key]=int(value)
            elif key in ('gamepad_name','up_level_mode','low_level_mode'):
                result[key]=value
            else:
                numeric=float(value)
                if not math.isfinite(numeric): raise ValueError('nonfinite state value')
                result[key]=int(numeric) if numeric.is_integer() else numeric
        stamp=result['monotonic_ns']
        if self.previous is not None and stamp<self.previous: raise ValueError('state clock went backwards')
        self.previous=stamp
        return result

    def nearest(self,stamp,limit_ns):
        while self.right is not None and self.right['monotonic_ns']<=stamp:
            self.left,self.right=self.right,self._next()
        choices=[x for x in (self.left,self.right) if x is not None]
        chosen=min(choices,key=lambda x:(abs(x['monotonic_ns']-stamp),x['monotonic_ns'])) if choices else None
        if chosen is None or abs(chosen['monotonic_ns']-stamp)>limit_ns:
            raise ValueError('missing or stale state row')
        return chosen

    def at_or_before(self,stamp,limit_ns):
        while self.right is not None and self.right['monotonic_ns']<=stamp:
            self.left,self.right=self.right,self._next()
        if self.left is None or stamp-self.left['monotonic_ns']>limit_ns:
            raise ValueError('missing or stale command row at image time')
        return self.left


class ClockMapping:
    def __init__(self,path,episode):
        self.reason='no saved alignment mapping; preview only'
        self.valid=False
        if path is None: return
        try:
            report=load_object(path)
            quality=report['quality']
            source_meta=load_object(Path(episode)/'meta.json')
            if not source_meta.get('episode_id') or report['episode_id']!=source_meta['episode_id']:
                raise ValueError('mapping belongs to another episode')
            if quality.get('degraded') is not False or quality.get('data_validated') is not True:
                raise ValueError('mapping did not pass alignment validation')
            error=quality.get('clock_estimated_error_ns')
            if type(error) is not int or not 0<=error<=10_000_000:
                raise ValueError('relative clock uncertainty exceeds 10 ms or is unknown')
            self.interval=report['valid_interval']
            self.map=report['mappings']['unitree']
            self.anchors=self.map['anchors']
            self.times=[a['remote_wall_ns'] for a in self.anchors]
            if len(self.times)<2 or any(type(t) is not int for t in self.times) or any(b<=a for a,b in zip(self.times,self.times[1:])):
                raise ValueError('invalid clock anchors')
            if self.map.get('mapping_type')!='piecewise_linear_offset' or self.map.get('host')!='unitree':
                raise ValueError('unsupported mapping type or host')
            if not self.interval or self.interval['end_exclusive_ns']<=self.interval['start_inclusive_ns']:
                raise ValueError('empty alignment interval')
            self.valid=True
            self.reason=None
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.reason=str(exc)

    def restore(self,stamp):
        if not self.valid: return None
        if not self.times[0]<=stamp<=self.times[-1]: return None
        for gap in self.map.get('unsupported_gaps',[]):
            if gap['remote_start_ns']<stamp<gap['remote_end_ns']: return None
        index=min(max(bisect.bisect_left(self.times,stamp),1),len(self.times)-1)
        a,b=self.anchors[index-1:index+1]
        offset=a['offset_ns']+round(Fraction((b['offset_ns']-a['offset_ns'])*(stamp-a['remote_wall_ns']),b['remote_wall_ns']-a['remote_wall_ns']))
        desktop=stamp-offset
        return desktop if self.interval['start_inclusive_ns']<=desktop<self.interval['end_exclusive_ns'] else None


def go2_record(row,kind):
    result={k:row[k] for k in ('monotonic_ns','wall_time_ns','seq')}
    result['valid']=True
    for name,keys in {'quaternion':['quat_'+a for a in 'wxyz'],'gyroscope':['gyro_'+a for a in 'xyz'],
        'accelerometer':['accel_'+a for a in 'xyz'],'rpy':['roll','pitch','yaw'],
        'foot_force':['foot_force_'+str(i) for i in range(4)]}.items(): result[name]=[row[k] for k in keys]
    if kind=='sport_mode_state':
        for k in ('robot_sec','robot_nanosec','error_code','mode','gait_type','progress','foot_raise_height','body_height','yaw_speed'): result[k]=row[k]
        for name in ('position','velocity'): result[name]=[row[name+'_'+a] for a in 'xyz']
        result['range_obstacle']=[row['range_'+str(i)] for i in range(4)]
    else:
        for k in ('tick','power_v','power_a'): result[k]=row[k]
        result['foot_force_est']=[row['foot_force_est_'+str(i)] for i in range(4)]
        for field in ('q','dq','tau_est'): result['motor_'+field]=[row['motor_'+str(i)+'_'+field] for i in range(20)]
    return result


def png_bytes(image):
    from PIL import Image
    out=BytesIO()
    Image.fromarray(image[:,:,::-1]).save(out,format='PNG',compress_level=1)
    return out.getvalue()


def export_video(source,output,config,clock):
    ffmpeg=config.get('ffmpeg') or shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('ffmpeg is required only for explicit video export')
    streams={}
    for camera in ('front','wrist'):
        count=0
        child=None
        try:
            with (output/(camera+'.ffmpeg.log')).open('wb') as log, (output/(camera+'.timestamps.csv')).open('x',newline='') as sidecar:
                writer=csv.writer(sidecar)
                writer.writerow(['video_frame','seq','reader_seq','monotonic_ns','wall_time_ns','desktop_ns','width','height'])
                dimensions=None
                for m,image in iter_camera_frames(source,camera):
                    size=(m['width'],m['height'])
                    if dimensions is None:
                        dimensions=size
                        child=subprocess.Popen([str(ffmpeg),'-nostdin','-loglevel','error','-n','-f','rawvideo',
                            '-pixel_format','bgr24','-video_size',f'{size[0]}x{size[1]}','-framerate',str(config.get('fps',30)),
                            '-i','pipe:0','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','18',
                            '-pix_fmt','yuv420p','-movflags','+faststart',str(output/(camera+'.mp4'))],stdin=subprocess.PIPE,stdout=log,stderr=log)
                    if size!=dimensions: raise ValueError('video source dimensions changed')
                    child.stdin.write(image.tobytes(order='C'))
                    writer.writerow([count,m['seq'],m['reader_seq'],m['monotonic_ns'],m['wall_time_ns'],clock.restore(m['wall_time_ns']),*size])
                    count+=1
                if child is not None:
                    child.stdin.close()
                    if child.wait(timeout=120): raise RuntimeError('video encoder failed; inspect log')
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
        streams[camera]={'frames':count,'nominal_fps':config.get('fps',30),'timing':'original timestamps in sidecar; constant-rate preview'}
    return streams


def convert_episode(source,output,*,target,mapping,config):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('output must be separate from source')
    if target not in ('pkl','video'): raise ValueError('target must be pkl or video')
    if output.exists(): raise FileExistsError(str(output))
    clock=ClockMapping(mapping,source)
    if mapping is not None and not clock.valid:
        raise ValueError('invalid training alignment mapping: ' + str(clock.reason))
    output.mkdir(parents=True)
    report={'source':str(source),'target':target,'exported':0,'rejected':0,'errors':[],
            'alignment_valid':False,'alignment_reason':clock.reason,'training_ready':False,
            'export_kind':'aligned' if clock.valid else 'preview','trimmed':0,'data_rejected':0}
    if target=='video':
        report['streams']=export_video(source,output,config,clock)
        report['exported']=sum(s['frames'] for s in report['streams'].values())
    else:
        (output/'frames').mkdir()
        aligned=0
        with ExitStack() as stack:
            cursors={}
            missing=[]
            limits={'piper_state':config.get('piper_stale_ms',250),'piper_status':config.get('piper_stale_ms',250),
                'piper_gamepad':config.get('piper_gamepad_stale_ms',1000),'sport_mode_state':config.get('go2_stale_ms',500),'low_state':config.get('go2_stale_ms',500)}
            for name in limits:
                try: cursors[name]=StateCursor(stack.enter_context((source/'raw'/(name+'.csv')).open(newline='')))
                except (OSError,ValueError,KeyError) as exc: missing.append(name+': '+str(exc))
            rejects=csv.writer(stack.enter_context((output/'rejected.csv').open('x',newline='')))
            rejects.writerow(['front_seq','reason'])
            for front,wrist in pair_frames(iter_camera_frames(source,'front'),iter_camera_frames(source,'wrist'),int(config.get('max_camera_skew_ms',50)*1e6)):
                m=front[0]
                try:
                    # Normal pre/post-roll is excluded before matching state streams.
                    if clock.valid and clock.restore(m['wall_time_ns']) is None:
                        report['rejected']+=1
                        report['trimmed']+=1
                        rejects.writerow([m['seq'],'outside supported common alignment interval'])
                        continue
                    if wrist is None: raise ValueError('no unique wrist frame within skew limit')
                    if missing: raise ValueError('; '.join(missing))
                    states={n:(cursor.at_or_before if n=='piper_gamepad' else cursor.nearest)(
                        m['monotonic_ns'],int(limits[n]*1e6)) for n,cursor in cursors.items()}
                    gamepad=states['piper_gamepad']
                    if any(gamepad.get(k)!=1 for k in ('arm_enabled','arm_connected','gamepad_connected')):
                        raise ValueError('Home/arm enable or gamepad connection absent')
                    if gamepad.get('command_sent')!=1:
                        raise ValueError('control target was not sent')
                    joint_keys=['joint_'+str(i)+'.pos' for i in range(1,7)]+['gripper.pos']
                    piper=states['piper_state']
                    command={**gamepad,'valid':True,'target':{k:gamepad[k] for k in joint_keys},
                             'gamepad':{'connected':True}}
                    pose_keys=['target_pose_'+str(i) for i in range(6)]
                    if all(k in gamepad for k in pose_keys):
                        command['target_pose']=[gamepad[k] for k in pose_keys]
                    elif gamepad.get('low_level_mode')!='joint':
                        raise ValueError('Cartesian command target pose missing from raw CSV')
                    mapped={n:clock.restore(frame[0]['wall_time_ns']) for n,frame in (('front',front),('wrist',wrist))}
                    valid=all(v is not None for v in mapped.values()) and all(clock.restore(row['wall_time_ns']) is not None for row in states.values())
                    if clock.valid and not valid:
                        report['rejected']+=1
                        report['trimmed']+=1
                        rejects.writerow([m['seq'],'camera or state outside supported common alignment interval'])
                        continue
                    record={'timestamp_ns':max(m['wall_time_ns'],wrist[0]['wall_time_ns']),
                        'frame_index':report['exported'],'camera':{n:{**frame[0],'rgb':png_bytes(frame[1])} for n,frame in (('front',front),('wrist',wrist))},
                        'go2':{n:go2_record(states[n],n) for n in ('sport_mode_state','low_state')},
                        'piper':{'state':{k:piper[k] for k in joint_keys},'status':{k:states['piper_status'][k] for k in ('ctrl_mode','arm_status','teach_status')},
                            'monotonic_ns':piper['monotonic_ns'],'wall_time_ns':piper['wall_time_ns'],'command':command},
                        'diagnostics':{'camera_skew_ns':abs(m['monotonic_ns']-wrist[0]['monotonic_ns']),
                            'state_match_error_ns':{n:row['monotonic_ns']-m['monotonic_ns'] for n,row in states.items()}},
                        'alignment':{'valid':valid,'desktop_ns':mapped}}
                    with (output/'frames'/f'{report["exported"]:09d}.pkl').open('xb') as f: pickle.dump(record,f,protocol=4)
                    report['exported']+=1
                    aligned+=int(valid)
                except (ValueError,KeyError) as exc:
                    report['rejected']+=1
                    report['data_rejected']+=1
                    rejects.writerow([m['seq'],str(exc)])
                    if len(report['errors'])<100: report['errors'].append(str(exc))
        report['alignment_valid']=bool(report['exported'] and aligned==report['exported'])
        integrity=inspect_episode(source)
        report['training_ready']=bool(report['alignment_valid'] and integrity['application_integrity'] and not report['data_rejected'])
    if mapping is not None: shutil.copy2(mapping,output/'source_alignment.json')
    durable_json(output/'conversion_report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--target',required=True,choices=['pkl','video'])
    parser.add_argument('--mapping',type=Path)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(convert_episode(args.episode,args.output,target=args.target,mapping=args.mapping,config=load_object(args.config)),indent=2,ensure_ascii=False))
