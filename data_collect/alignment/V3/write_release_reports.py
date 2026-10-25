"""Write current strict V3 membership and validity reports after verification."""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


def read(path):return json.loads(path.read_text())


def write_reports(root,final):
    validation=read(root/'reports/validation.json');assert validation['passed'] and not validation['imputation_applied']
    inventory=read(root/'configs/source_inventory.json');manifest=read(root/'reports/episode_manifest.json')
    horizon=read(root/'configs/action_chunk.json')['action_horizon']
    rows=[]
    for entry in inventory['sources']:
        name=entry['source_episode']
        audit=pq.read_table(root/'aligned'/f'{name}.parquet').to_pylist()
        episodes=[e for e in manifest if e['source_episode']==name]
        total=len(audit);bad=sum(not(all(r['state_valid']) and all(r['action_valid'])) for r in audit)
        rows.append(dict(source_episode=name,source_path=entry['source_path'],source_session=entry['source_session'],
            split=entry['split'],included=entry['include'],exported=bool(episodes),total_grid_frames=total,
            state_invalid_frames=sum(not all(r['state_valid']) for r in audit),
            action_invalid_frames=sum(not all(r['action_valid']) for r in audit),
            state_or_action_invalid_frames=bad,state_or_action_invalid_percent=round(100*bad/total,2),
            all_rule_rejected_frames=sum(not r['eligible'] for r in audit),
            episodes=len(episodes),exported_frames=sum(e['frames'] for e in episodes),
            h15_windows=sum(max(0,e['frames']-14) for e in episodes),
            h20_windows=sum(max(0,e['frames']-19) for e in episodes),imputation_applied=False))
    with (root/'reports/source_quality.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'reports/source_quality.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    text=['# V3：当前训练与验证数据','',
        '已合并2026-10-20与2026-10-21两批数据。10月20日任务11、13、14、15加入验证集，其余20条加入训练集，原有任务划分保留。','',
        f'LeRobot v2.1，15Hz，状态30维、动作14维，默认action chunk={horizon}。chunk独立于视频频率。模型输入使用configs/task.json中的英文任务描述。','',
        '|划分|有效录制条数|连续片段数|导出帧数|15步窗口数|','|---|---:|---:|---:|---:|']
    for split in ('train','val'):
        s=validation['datasets'][split]
        text.append(f"|{'训练' if split=='train' else '验证'}|{s['sources']}|{s['episodes']}|{s['frames']:,}|{s['windows']:,}|")
    text+=['','训练集和验证集均只保留原始有效连续片段，无前值补齐、插值补齐或跨缺口拼接。原有3条验证录制已经从未补值审计表重建，不再包含此前构造的保持值。',
        '导出最短片段仍为15帧；每个训练窗口只在一个片段内部生成。无效时间点保留在aligned审计表，其中无效维度的0是占位值，由有效性标记区分，不进入任何训练或验证窗口。','',
        '训练与验证按录制任务划分，没有重复来源任务。两侧仍有共同的采集会话，见configs/schema.json的overlapping_source_sessions；本划分不是采集会话独立验证。训练归一化仅由本次训练集重算。','',
        '## 当前验证集','',
        '|来源|导出帧数|连续片段数|15步窗口|','|---|---:|---:|---:|']
    for r in rows:
        if r['split']=='val' and r['exported']:
            text.append(f"|{r['source_episode']}|{r['exported_frames']}|{r['episodes']}|{r['h15_windows']}|")
    text+=['','## 文件与使用','',
        '```text','V3/','├── lerobot/local/heterovla_v3_train/  # 训练数据及三路视频',
        '├── lerobot/local/heterovla_v3_val/    # 原始有效验证数据',
        '├── aligned/                         # 所有原始时间点及有效性审计',
        '├── provenance/                      # 每个导出片段对应的原始记录',
        '├── configs/                         # 来源、划分、维度、阈值、chunk',
        '├── assets/heterovla_v3/norm_stats.json',
        '├── reports/                         # 当前检查与历史转换证据',
        '└── scripts/                         # 处理与loader代码快照','```','',
        '```python','import sys',f"sys.path.insert(0, {str(final/'scripts')!r})",'from dataset_v3 import OpenpiV3Dataset',
        f"train = OpenpiV3Dataset({str(final)!r}, 'train')",f"val = OpenpiV3Dataset({str(final)!r}, 'val')",
        f"assert train[0]['actions'].shape == ({horizon}, 14)",'```','',
        '[逐来源质量CSV](reports/source_quality.csv) · [完整核验结果](reports/validation.json) · [合并记录](reports/merge.json)','',
        '当前可导出数据为62条；来源清单另保留1条历史隔离记录和1条历史无可用连续片段记录，它们不进入训练或验证。',
        '所有导出表格逐值核对原始审计表；全部导出视频全量解码，并抽取首中尾及相机修复帧核对原始图像。保留的历史与新批次视频复制时还核对了原发布清单哈希。',
        '10月20日02使用原始相机帧的修复已保留，补充检查见reports/history/merge_parents/V3_20261020/reports/task02_camera_repair.json。',
        'reports/history/保存合并前的报告，仅作溯源，不代表当前成员和规则。原始录制仍保存在raw/benchmark。','']
    (root/'README.md').write_text('\n'.join(text))
    quality=['# V3逐条有效性统计','',
        '计数基于原始15Hz视频时间轴，包含首尾边界；状态或动作无效计数不重复累加。训练和验证均不补值。','',
        '|划分|来源|总时间点|状态或动作无效／比例|导出帧|片段|15步窗口|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        quality.append(f"|{r['split']}|{r['source_episode']}|{r['total_grid_frames']}|{r['state_or_action_invalid_frames']}／{r['state_or_action_invalid_percent']:.2f}%|{r['exported_frames']}|{r['episodes']}|{r['h15_windows']}|")
    (root/'reports/source_quality_zh.md').write_text('\n'.join(quality)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--final',type=Path,required=True)
    args=parser.parse_args();write_reports(args.root.resolve(),args.final.resolve())
