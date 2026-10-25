"""One resource-waiting resume job; exit permanently after its launch attempt.

No GPU allocation while waiting, no cron/desktop automation, no restart loop.
The already-tested queue remains the sole GPU and one-shot claim authority.
"""
from __future__ import annotations
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def main():
    import queue_training as q
    plan=json.loads((HERE/'resume_plan.json').read_text())
    queue=Path(plan['queue_dir']);queue.mkdir(parents=True,exist_ok=True)
    lock=os.open(queue/'resource_wait.lock',os.O_RDWR|os.O_CREAT,0o600)
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return 0
    run=Path('/meta_eon_cfs/home/hph/heterovla-model/V3/runs')/plan['run_name']
    ident=q.process_identity(os.getpid())
    waiter={'status':'waiting_for_idle_gpu','identity':ident,'started_utc':q.utcnow(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_actual_step':plan['source_actual_step'],'target_updates':20000,
        'check_interval_seconds':60,'desktop_automation':'PAUSED'}
    q.atomic_json(queue/'resource_wait.json',waiter)
    pointer={'run_name':plan['run_name'],'resume_id':plan['resume_id'],
        'queue_dir':str(queue),'code_dir':str(HERE),'source_checkpoint':plan['source_checkpoint'],
        'source_actual_step':plan['source_actual_step'],'started_unix':time.time(),
        'started_utc':q.utcnow(),'boot_id':waiter['boot_id'],'phase':'waiting_for_idle_gpu'}
    q.atomic_json(run/'active_resume.json',pointer)
    old=q.read_json(run/'runtime_status.json') or {}
    q.atomic_json(HERE/'runtime_before_resource_wait.json',old)
    q.atomic_json(run/'runtime_status.json',dict(old,status='waiting_for_idle_gpu',pid=None,
        step=plan['source_actual_step'],resume_id=plan['resume_id'],
        source_actual_step=plan['source_actual_step'],target_steps=20000,
        updated_utc=q.utcnow(),waiting_job_pid=os.getpid()))
    command=[sys.executable,str(HERE/'queue_training.py'),'check-and-start',
        '--code-dir',str(HERE),'--queue-dir',str(queue),'--run-dir',str(run),
        '--run-name',plan['run_name']]
    while True:
        # Any durable attempt ends this waiter, even if that attempt fails.
        before=q.read_json(queue/'attempt.json')
        if before:
            state=q.read_json(queue/'state.json') or {}
        else:
            result=subprocess.run(command,capture_output=True,text=True,timeout=180)
            try:state=json.loads(result.stdout)
            except ValueError:
                raise RuntimeError('Queue did not return JSON: '+result.stderr[-2000:])
        waiter.update(last_checked_utc=q.utcnow(),queue_status=state.get('status'))
        if state.get('training_identity'):
            pointer.update(phase='launched',gpu=state['gpu']['uuid'],
                started_utc=state.get('started_utc',q.utcnow()),
                started_unix=dt.datetime.fromisoformat(state['started_utc']).timestamp())
            q.atomic_json(run/'active_resume.json',pointer)
        if (queue/'attempt.json').exists():
            waiter.update(status='launch_attempt_made',finished_utc=q.utcnow())
            q.atomic_json(queue/'resource_wait.json',waiter)
            print(json.dumps(waiter),flush=True)
            return 0
        if state.get('status')=='blocked':
            waiter.update(status='failed_preflight',detail=state.get('detail'),finished_utc=q.utcnow())
            q.atomic_json(queue/'resource_wait.json',waiter)
            return 1
        q.atomic_json(queue/'resource_wait.json',waiter)
        print(json.dumps({'at':q.utcnow(),'status':'waiting_for_idle_gpu','source_step':plan['source_actual_step']}),flush=True)
        time.sleep(60)

if __name__=='__main__':
    raise SystemExit(main())
