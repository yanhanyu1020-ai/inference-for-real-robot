#!/usr/bin/env python3
"""Isolated two-writer throughput gate. Creates new files, never removes data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'remote/unitree'))
from mcap_storage import CameraStore, durable_json


def percentiles(values):
    return {'count':len(values), **{key:float(np.percentile(values,p)) if values else None
        for key,p in [('p50_ms',50),('p95_ms',95),('p99_ms',99),('max_ms',100)]}}


def run_workload(output, seconds, codec, samples):
    if seconds<=0 or not samples:
        raise ValueError('positive duration and image samples required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(output).free<20*1024**3:
        raise RuntimeError('less than 20 GiB free')
    append_ms=[]; sync_ms=[]; errors=[]; streams={}; raw_bytes=[0]
    lock=threading.Lock(); start=time.monotonic(); cpu=time.process_time(); stop=threading.Event()
    def fsync(fd):
        before=time.monotonic(); os.fsync(fd)
        with lock: sync_ms.append((time.monotonic()-before)*1000)
    def worker(name):
        store=CameraStore(output,name,codec,io_hooks={'fsync':fsync})
        seq=0; checkpoint=time.monotonic(); previous_ns=0
        try:
            while time.monotonic()-start<seconds and not stop.is_set():
                image=samples[seq%len(samples)]; seq+=1; h,w=image.shape[:2]
                now=max(previous_ns+1,time.monotonic_ns()); previous_ns=now
                meta=dict(camera=name,serial='benchmark',seq=seq,reader_seq=seq,width=w,height=h,
                    stride=w*3,dtype='uint8',pixel_format='bgr8',monotonic_ns=now,
                    wall_time_ns=time.time_ns(),boot_id='benchmark',timestamp_source='host_receive',
                    device_timestamp_ns=None,device_frame_number=None)
                before=time.monotonic(); store.append(meta,image)
                with lock:
                    append_ms.append((time.monotonic()-before)*1000); raw_bytes[0]+=image.nbytes
                if time.monotonic()-checkpoint>=1:
                    store.checkpoint(); checkpoint=time.monotonic()
                    if shutil.disk_usage(output).free<10*1024**3: raise RuntimeError('disk reserve reached')
        except Exception as exc:
            with lock: errors.append(name+': '+str(exc))
            stop.set()
        finally:
            try: streams[name]=store.close()
            except Exception as exc:
                with lock: errors.append(name+' close: '+str(exc))
    threads=[threading.Thread(target=worker,args=(n,)) for n in ('front','wrist')]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    elapsed=time.monotonic()-start; cpu_seconds=time.process_time()-cpu
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    except ImportError: rss=None
    result=dict(seconds=elapsed,codec=codec,raw_bytes=raw_bytes[0],raw_bytes_per_s=raw_bytes[0]/elapsed,
        file_bytes=sum(s['bytes'] for s in streams.values()),cpu_seconds=cpu_seconds,
        cpu_percent_total=100*cpu_seconds/elapsed/(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()),
        peak_rss_bytes=rss,append=percentiles(append_ms),fsync=percentiles(sync_ms),
        queue='not applicable: synchronous writer saturation test',streams=streams,errors=errors)
    durable_json(output/'benchmark.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--codec',choices=['lz4','none'],default='lz4')
    parser.add_argument('--samples',required=True,type=Path,help='real camera .npy samples; allow_pickle=False')
    parser.add_argument('--state-bytes-per-s',type=float,default=1024**2)
    args=parser.parse_args()
    if args.seconds<=0 or args.state_bytes_per_s<0: parser.error('invalid budget/duration')
    samples=[np.load(str(p),allow_pickle=False) for p in sorted(args.samples.glob('*.npy'))]
    if not samples or any(a.shape!=(480,640,3) or a.dtype!=np.uint8 for a in samples):
        parser.error('need real uint8 BGR 640x480 sample frames')
    args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.RandomState(20260926)
    random=[rng.randint(0,256,(480,640,3),dtype=np.uint8) for _ in range(8)]
    reports={name:run_workload(args.output/name,args.seconds/2,args.codec,images)
             for name,images in [('incompressible',random),('camera_samples',samples)]}
    required=2*(640*480*3*30*2+args.state_bytes_per_s)
    result=dict(required_bytes_per_s=required,state_budget_bytes_per_s=args.state_bytes_per_s,
        state_budget_is_estimate=True,workloads=reports,
        passed=all(not r['errors'] and r['raw_bytes_per_s']>=required for r in reports.values()))
    durable_json(args.output/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
