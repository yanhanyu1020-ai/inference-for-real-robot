#!/usr/bin/env python3
"""CPU-only V2 build, validation and publication; safe to run with nohup."""
from __future__ import annotations
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
from alignment_paths import BENCHMARK_ROOT, DATA_ROOT, V1_ROOT, PROJECT_ROOT, TRAIN_SCRIPTS
from prepare_v2 import TASK, dump


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def raw_snapshot(source):
    result = {}
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Unexpected raw symlink: {p}')
        if p.is_file():
            s = p.stat()
            result[str(p.relative_to(source))] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns, mode=s.st_mode)
    return result


def verify_v1(v1, expected_manifest_hash=None):
    manifest = v1/'SHA256SUMS.txt'
    digest = sha256(manifest)
    if expected_manifest_hash is not None:
        assert digest == expected_manifest_hash, 'V1 integrity manifest changed'
    count = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        p = (v1/relative).resolve()
        assert v1 in p.parents and p.is_file(), relative
        assert sha256(p) == expected, f'V1 checksum mismatch: {relative}'
        count += 1
    return dict(all_match=True, manifest_sha256=digest, files=count)


def publish_docs(root, final, source, v1):
    validation = json.loads((root/'reports/validation.json').read_text())
    assert validation['format_validation_passed']
    inventory = json.loads((root/'configs/source_inventory.json').read_text())
    alignment = json.loads((root/'reports/alignment_summary.json').read_text())
    manifest = json.loads((root/'reports/episode_manifest.json').read_text())
    actual = len({r['source_episode'] for r in manifest})
    excluded = alignment['non_exportable_source_episodes']
    summaries = {s: validation['datasets'][s] for s in ['train', 'val']}
    lines = []
    for e in inventory['sources']:
        report = json.loads((root/'reports'/f"{e['source_episode']}.json").read_text())
        lines.append(f"| {e['source_path']} | {e['split']} | {report['eligible_frames']} | {report['exported_segments']} | {report['exported_frames']} |")
    table = '\n'.join(lines)
    text = f'''# HeteroVLA V2：新26条与V1旧5条的合并数据

任务指令：**{TASK}**。

本版本请求合并31条示教（原5条、新26条），全部保留逐帧对齐审计表。
按照原V1质量规则，实际可导出来源 **{actual}条**；未生成完整有效H10片段的来源：{', '.join(excluded) or '无'}。
旧 `task_03_1144-1300` 继续独立隔离，不包含在请求合并的31条中。
源raw目录与V1在处理前后均检查，V1原校验清单逐项SHA-256复核通过。

## 数据入口

- 路径：`{final}`。
- train：`lerobot/local/heterovla_v2_train`，{summaries['train']['frames']}帧、{summaries['train']['episodes']}个连续片段、{summaries['train']['h10_windows']}个严格H10窗口。
- val：`lerobot/local/heterovla_v2_val`，{summaries['val']['frames']}帧、{summaries['val']['episodes']}个连续片段、{summaries['val']['h10_windows']}个严格H10窗口。
- 10Hz，三路640×480 RGB，state 30维，action 14维，H=10。
- 训练、验证按原始采集会话隔离；V1原划分保留，新Oct12 faf6会话4条整体为val，其余新22条为train。划分在检查有效样本前确定。
- 新来源ID含日期；`source_path` 是raw benchmark下的相对路径，避免每天重置task编号造成重名。

## 沿用V1的标签与过滤

Go2前3维与UAV后4维是未来0.1s运动的水平朝向坐标速度代理标签；Piper中间7维是已发送的绝对关节/夹爪目标。
保持14维动作的单位、阈值、mask和局部原点定义。没有将反馈状态冒充已发送命令。
输入状态和图像只取过去记录；原始时间映射下不读未来观测。跨设备物理曝光同步误差仍未经独立测定。
UAV混合模式FCU echoed target、逐维ignore-bit mask、type_mask和coordinate_frame另存于aligned/provenance，不直接作为统一速度命令。
`command_sent=0`、无效状态、过期观测、时钟告警与Go2起身准备等按V1规则过滤；不跨缺口拼接，不导出少于10帧的片段，不补造动作、success或reward标签。
旧5条直接复用V1逐帧对齐值，导出后的state/action/mask/source时间逐元素校验，与旧视频逐帧比较。

## 各来源

| raw相对路径 | split | 合格帧 | 导出片段 | 导出帧 |
|---|---|---:|---:|---:|
{table}

完整过滤原因见 `reports/<source_episode>.json`；未导出来源汇总见 `reports/alignment_summary.json`。

## 使用与复验

数据是固定LeRobot v2.1格式，与V1同一CPU转换环境生成。严格loader不会跨片段或在episode尾部补齐动作。

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home()/"heterovla/train/V2/scripts"))
from dataset_v2 import JointV2Dataset, as_openpi_sample
data = JointV2Dataset(Path.home()/"heterovla-data/aligned/V2", split="train", horizon=10)
sample = as_openpi_sample(data[0])
```

```bash
~/heterovla/.venvs/data-v1/bin/python ~/heterovla/data_collect/alignment/V2/verify_v2.py --root ~/heterovla-data/aligned/V2 --source ~/heterovla-data/raw/benchmark --v1-root ~/heterovla-data/aligned/V1
```

`assets/heterovla_v2/norm_stats.json`仅使用训练集计算。`configs/source_inventory.json`固定来源和会话划分。
`scripts/`保存本次转换和验证源码快照，维护入口是主仓库的`data_collect/alignment/V2/`和`train/V2/`。
`reports/validation.json`包含全视频解码、真实LeRobot/DataLoader、严格H10、归一化、旧V1一致性等结果。
`reports/source_integrity.json`记录raw元数据和V1 SHA复核；`SHA256SUMS.txt`为发布校验清单。
本次仅CPU转换与验证数据，没有启动模型训练。
'''
    (root/'README.md').write_text(text)
    dump(root/'STATUS.json', dict(task=TASK, version='V2', format_ready=True,
        lerobot_loader_verified=True, training_target='joint_kinematic_imitation',
        operator_command_bc_ready=False, raw_unchanged=True, v1_unchanged=True,
        requested_demonstrations=31, previous_demonstrations=5, new_demonstrations=26,
        exportable_demonstrations=actual, non_exportable_source_episodes=excluded,
        default_fps=10, state_dim=30, action_dim=14, horizon=10, splits=summaries,
        full_vla_training_started=False, physical_time_alignment_independently_verified=False,
        quarantined_source_episodes=inventory['quarantined_source_episodes']))
    conversion_path = root/'reports/conversion.json'
    conversion = json.loads(conversion_path.read_text())
    conversion['output'] = str(final)
    dump(conversion_path, conversion)
    checksums = []
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if not p.is_file() or p.is_symlink() or any(x.startswith('.') or x == '__pycache__' for x in rel.parts):
            continue
        if p.name == 'SHA256SUMS.txt' or p.suffix == '.log':
            continue
        checksums.append(f'{sha256(p)}  {rel}')
    (root/'SHA256SUMS.txt').write_text('\n'.join(checksums)+'\n')
    return dict(requested_sources=31, exportable_sources=actual, splits=summaries,
        non_exportable_source_episodes=excluded, checksummed_files=len(checksums))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=BENCHMARK_ROOT)
    ap.add_argument('--v1-root', type=Path, default=V1_ROOT)
    ap.add_argument('--destination', type=Path, default=DATA_ROOT)
    ap.add_argument('--staging', type=Path)
    ap.add_argument('--align-only', action='store_true')
    args = ap.parse_args()
    source, v1, final = (p.expanduser().resolve() for p in [args.source, args.v1_root, args.destination])
    staging = (args.staging or final.parent/'.V2-building-20261015').expanduser().resolve()
    assert staging.parent == final.parent and staging != final
    for output in [final, staging]:
        for protected in [source, v1, PROJECT_ROOT]:
            assert output != protected and protected not in output.parents and output not in protected.parents
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f'Refusing to replace nonempty V2: {final}')
    status_path = final.parent/'V2_build_20261015.json'
    lock = (final.parent/'.V2-build.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(state, **extra):
        value = dict(status=state, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(),
            staging=str(staging), destination=str(final), **extra)
        tmp = status_path.with_suffix('.tmp')
        dump(tmp, value)
        os.replace(tmp, status_path)
        print('STATUS', json.dumps(value, ensure_ascii=False), flush=True)
    try:
        staging.mkdir(exist_ok=True)
        (staging/'reports').mkdir(exist_ok=True)
        before_path = staging/'reports/raw_before.json'
        v1_path = staging/'reports/v1_before.json'
        status('checking_sources')
        if not before_path.exists():
            dump(before_path, raw_snapshot(source))
        else:
            assert json.loads(before_path.read_text()) == raw_snapshot(source), 'Raw changed since build began'
        if not v1_path.exists():
            dump(v1_path, verify_v1(v1))
        else:
            verify_v1(v1, json.loads(v1_path.read_text())['manifest_sha256'])
        scripts = staging/'scripts'
        scripts.mkdir(exist_ok=True)
        exported = (staging/'reports/conversion.json').exists()
        if not exported:
            for p in Path(__file__).resolve().parent.glob('*.py'):
                shutil.copyfile(p, scripts/p.name)
        loader = TRAIN_SCRIPTS/'dataset_v2.py'
        if loader.exists() and not exported:
            shutil.copyfile(loader, scripts/loader.name)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        def run(name, *options):
            subprocess.run([sys.executable, '-u', str(scripts/name), *map(str, options)], env=env, check=True)
        if not (staging/'reports/alignment_summary.json').exists():
            status('aligning')
            run('prepare_v2.py', '--source', source, '--out', staging, '--v1-root', v1, '--aligned-only')
        if args.align_only:
            status('aligned', summary=json.loads((staging/'reports/alignment_summary.json').read_text()))
            return
        frozen_loader = scripts/'dataset_v2.py'
        assert frozen_loader.is_file() and (scripts/'verify_v2.py').is_file(), 'V2 validator/loader is missing'
        loader_hash = sha256(frozen_loader)
        if not exported:
            status('exporting')
            run('prepare_v2.py', '--source', source, '--out', staging, '--v1-root', v1, '--reuse-aligned')
        status('validating')
        run('verify_v2.py', '--root', staging, '--source', source, '--v1-root', v1, '--train-scripts', scripts)
        assert sha256(frozen_loader) == loader_hash, 'Frozen loader changed during build'
        status('checking_integrity')
        before = json.loads(before_path.read_text())
        assert before == raw_snapshot(source), 'Raw changed during preparation'
        v1_check = verify_v1(v1, json.loads(v1_path.read_text())['manifest_sha256'])
        dump(staging/'reports/source_integrity.json', dict(raw_metadata_unchanged=True,
            raw_files_compared=len(before), v1_sha256_all_match=True, v1_check=v1_check,
            note='Raw upload SHA-256 was verified during transfer; this build also compares all source size/mtime/mode before and after.'))
        status('publishing')
        result = publish_docs(staging, final, source, v1)
        if final.exists():
            final.rmdir()  # rmdir refuses anything except the reserved empty V2.
        staging.rename(final)
        status('complete', result=result)
    except BaseException as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
