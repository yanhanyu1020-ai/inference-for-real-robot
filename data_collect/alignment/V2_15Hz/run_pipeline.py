"""Build, validate and publish the independently rebuilt 15 Hz dataset on CPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from paths import BENCHMARK_ROOT, DATA_ROOT, REFERENCE_ROOT, TRAIN_SCRIPTS
from prepare import make_inventory


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def snapshot(source, reference):
    result = {}
    inventory = make_inventory(source,reference)
    for entry in inventory['sources']:
        for p in sorted((source/entry['source_path']).rglob('*')):
            if p.is_file() and not p.is_symlink():
                s=p.stat()
                result['raw/'+str(p.relative_to(source))] = [s.st_size,s.st_mtime_ns,s.st_mode]
    for p in sorted((source/'2026-10-18').rglob('*')):
        if p.is_file() and not p.is_symlink():
            s=p.stat();result['raw/'+str(p.relative_to(source))]=[s.st_size,s.st_mtime_ns,s.st_mode]
    for release in [reference.parent/'V1', reference]:
        # The published manifest lists only release artifacts, excluding environments.
        manifest = release/'SHA256SUMS.txt'
        result[release.name+'/manifest_sha256'] = sha(manifest)
        for line in manifest.read_text().splitlines():
            digest, rel = line.split('  ',1)
            p=release/rel
            assert p.resolve().is_relative_to(release.resolve())
            s=p.stat()
            result[release.name+'/'+rel]=[s.st_size,s.st_mtime_ns,s.st_mode]
    return result


def verify_release(release):
    count=0
    for line in (release/'SHA256SUMS.txt').read_text().splitlines():
        expected,rel=line.split('  ',1)
        p=release/rel
        assert p.resolve().is_relative_to(release.resolve())
        assert sha(p)==expected, f'Existing release checksum mismatch: {p}'
        count+=1
    return {'files':count,'sha256_all_match':True}


def verify_upload(source):
    root=source/'2026-10-18';manifest=read(root/'file_manifest.json');count=0;size=0
    for entry in manifest['files']:
        relative=Path(entry['path'].replace('\\','/'));p=root/relative
        assert not relative.is_absolute() and p.resolve().is_relative_to(root.resolve())
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],f'Upload checksum mismatch: {p}'
        count+=1;size+=entry['bytes']
    assert count==manifest['count'] and size==manifest['bytes']
    return dict(files=count,bytes=size,sha256_all_match=True,manifest_sha256=sha(root/'file_manifest.json'))


def publish_docs(stage, final):
    result=read(stage/'reports/validation.json')
    assert result['format_validation_passed']
    summary=read(stage/'reports/alignment_summary.json')
    conversion=read(stage/'reports/conversion.json');conversion['output']=str(final)
    write(stage/'reports/conversion.json',conversion)
    status=dict(version='V2_15Hz',fps=15,state_dim=30,action_dim=14,horizon=15,
        prediction_duration_s=1.0,format_ready=True,lerobot_loader_verified=True,
        requested_demonstrations=39,additional_upload_demonstrations=8,exportable_demonstrations=summary['included_sources_with_h15_segments'],
        non_exportable_source_episodes=summary['non_exportable_source_episodes'],
        splits=result['datasets'],reference_version='V2',all_sources_rebuilt_from_raw=True,
        training_target='joint_kinematic_imitation',operator_command_bc_ready=False,
        physical_time_alignment_independently_verified=False,full_vla_training_started=False,
        raw_unchanged=True,v1_unchanged=True,v2_unchanged=True)
    write(stage/'STATUS.json',status)
    lines=['# HeteroVLA V2_15Hz','',
        '按视频时间轴重新处理 V2 的原始来源：15Hz，state 30维，action 14维，H=15（约1秒）。',
        '任务：抓取桌子上的纸箱内的方块。所有旧、新来源均从 raw 重新计算，没有复用或插值旧10Hz标签。','',
        '## 时间和动作语义','',
        '- 逐行使用原始 `timing/frame_mapping.csv` 的 `task_time_s`，校验其15Hz网格。',
        '- 每路图像按实际映射采集时间选择不晚于当前时刻的最近帧；状态同样只使用过去记录。相机时间与视频名义时刻不同，因此选中帧号可以与网格行号不同。',
        '- Go2/UAV动作由 `[t,t+1/15s]` 位姿差计算，水平速度转换到t时刻朝向坐标；Piper使用截至区间终点最近的有效已发送绝对关节/夹爪目标。',
        '- 状态速度由过去约1/15秒的有效位姿计算，分母使用真实采样时间差。',
        '- 沿用V2的单位、坐标定义和有效性阈值；拒绝超出源时长的动作区间。缺口切段，少于15帧不导出，loader不补齐、不跨段。',
        '- 包含V2的31条请求来源和10月19日上传的10月18日8条保留任务，共39条；旧时钟隔离来源另外保留审计。已删除04、09、10不纳入。',
        '- 旧会话train/val划分保持；新增批次最早f4b0会话3条为val，其余5条为train。后续新增raw不自动纳入。',
        '- Go2/UAV是实际运动代理标签，Piper是已发送命令；跨设备物理曝光同步精度未独立验证。','',
        '## 导出结果','',
        '| split | 有效来源 | 片段 | 帧数 | H15窗口 |','|---|---:|---:|---:|---:|']
    for split,s in result['datasets'].items():
        lines.append(f"| {split} | {s['sources']} | {s['episodes']} | {s['frames']} | {s['h15_windows']} |")
    lines+=['','## 使用','',
        'LeRobot v2.1数据位于 `lerobot/local/heterovla_v2_15hz_train` 和 `heterovla_v2_15hz_val`。',
        '```python','import sys','from pathlib import Path',
        f"root = Path({str(final)!r})", "sys.path.insert(0, str(root/'scripts'))",
        'from dataset_15hz import JointV215HzDataset, as_openpi_sample',
        "dataset = JointV215HzDataset(root, split='train', horizon=15)",
        'sample = as_openpi_sample(dataset[0])  # actions.shape == (15, 14)','```','',
        '训练接入必须使用15Hz loader、action_horizon=15和本版本norm_stats；旧V2的H10 loader不适用。',
        '`assets/heterovla_v2_15hz/norm_stats.json`仅统计训练集。',
        '`aligned/`含所有视频网格行及拒绝原因；`provenance/`保存导出帧的来源时间和索引。',
        '`reports/validation.json`保存独立原始标签核对、全视频解码、真实loader、H15窗口和归一化验证。',
        '`reports/source_integrity.json`保存原始文件元数据、V1/V2发布校验结果。',
        '`scripts/`是构建源码快照；维护入口为主仓库`data_collect/alignment/V2_15Hz`及`train/V2_15Hz/scripts`。',
        '本次只重新处理及验证数据，没有启动训练或机器人执行。','']
    (stage/'README.md').write_text('\n'.join(lines))
    sums=[]
    for p in sorted(stage.rglob('*')):
        rel=p.relative_to(stage)
        if p.is_file() and not p.is_symlink() and p.name!='SHA256SUMS.txt' and not any(x.startswith('.') or x=='__pycache__' for x in rel.parts):
            sums.append(f'{sha(p)}  {rel}')
    (stage/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
    return status


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=BENCHMARK_ROOT)
    ap.add_argument('--reference-root',type=Path,default=REFERENCE_ROOT)
    ap.add_argument('--destination',type=Path,default=DATA_ROOT)
    ap.add_argument('--staging',type=Path)
    ap.add_argument('--align-only',action='store_true')
    args=ap.parse_args()
    source,reference,final=[x.expanduser().resolve() for x in [args.source,args.reference_root,args.destination]]
    stage=(args.staging or final.parent/('.'+final.name+'-building')).expanduser().resolve()
    assert stage.parent==final.parent and stage!=final
    for output in [stage,final]:
        for protected in [source,reference,reference.parent/'V1']:
            assert output!=protected and protected not in output.parents and output not in protected.parents
    if final.exists(): raise FileExistsError(f'Refusing to replace {final}')
    lock=(final.parent/'.V2_15Hz-build.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=final.parent/'V2_15Hz_build.json'
    def status(state,**extra):
        obj=dict(status=state,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),staging=str(stage),destination=str(final),**extra)
        write(status_path,obj);print(json.dumps(obj,ensure_ascii=False),flush=True)
    try:
        stage.mkdir(exist_ok=True);(stage/'reports').mkdir(exist_ok=True)
        status('checking_sources')
        before=stage/'reports/source_before.json'
        snap=snapshot(source,reference)
        if before.exists(): assert read(before)==snap,'Source files changed since build began'
        else: write(before,snap)
        upload_check=stage/'reports/new_upload_integrity.json'
        if not upload_check.exists():
            status('checking_new_upload_sha256')
            write(upload_check,verify_upload(source))
        else:
            assert read(upload_check)['manifest_sha256']==sha(source/'2026-10-18/file_manifest.json')
        scripts=stage/'scripts';scripts.mkdir(exist_ok=True)
        files=list(Path(__file__).resolve().parent.glob('*.py'))+[TRAIN_SCRIPTS/'dataset_15hz.py',TRAIN_SCRIPTS/'chunk_settings.py']
        frozen=(stage/'reports/alignment_summary.json').exists()
        for p in files:
            target=scripts/p.name
            if frozen and target.exists():
                assert sha(target)==sha(p), f'Frozen build script changed: {p.name}'
            else: shutil.copyfile(p,target)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_DATASETS_CACHE='/tmp/heterovla-v2-15hz-hf-cache')
        def run(name,*options):
            subprocess.run([sys.executable,'-u',str(scripts/name),*map(str,options)],env=env,check=True)
        run('test_alignment.py')
        if not frozen:
            status('aligning')
            run('prepare.py','--source',source,'--reference-root',reference,'--out',stage,'--aligned-only')
        if args.align_only:
            status('aligned');return
        if not (stage/'reports/conversion.json').exists():
            status('exporting')
            run('prepare.py','--source',source,'--reference-root',reference,'--out',stage,'--reuse-aligned')
        status('validating')
        run('verify.py','--root',stage,'--source',source,'--reference-root',reference)
        status('checking_integrity')
        assert read(before)==snapshot(source,reference),'Source files changed during build'
        integrity={'raw_metadata_unchanged':True,'v1':verify_release(reference.parent/'V1'),'v2':verify_release(reference),
            'new_upload':read(upload_check)}
        write(stage/'reports/source_integrity.json',integrity)
        result=publish_docs(stage,final)
        stage.rename(final)
        status('complete',result=result)
    except BaseException as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__': main()
