"""Write current-batch reports; training horizons are separate from data integrity."""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
from alignment_core import segments


def read(p):
    return json.loads(p.read_text())


def write_reports(root, final):
    summary=read(root/'reports/alignment_summary.json')
    upload=read(root/'reports/upload_integrity.json')
    checks=read(root/'reports/validation.json')
    assert checks['passed'] and upload['all_manifest_files_passed'] and upload['all_task_artifacts_passed']
    batch=read(root/'configs/schema.json')['batch']
    sources=summary['sources'];count=len(sources)
    frames=summary['exported_frames'];episodes=summary['exported_segments'];windows=summary['h20_windows']
    duration=sum(r['duration_s'] for r in sources)
    eligible=sum(r['eligible_frames'] for r in sources)
    short=sum(r['excluded_short_segment_frames'] for r in sources)
    rows=[]
    detail_dir=root/'reports/frame_quality'
    detail_dir.mkdir(exist_ok=True)
    for r in sources:
        audit=pq.read_table(root/'aligned'/f"{r['source_episode']}.parquet").to_pylist()
        exported={x['source_frame_index'] for run in segments(audit) for x in run}
        details=[]
        for x in audit:
            state_ok=all(x['state_valid']);action_ok=all(x['action_valid'])
            details.append(dict(source_frame_index=x['source_frame_index'],task_time_s=x['source_task_time_s'],
                action_target_time_s=x['action_target_time_s'],state_valid=state_ok,action_valid=action_ok,
                go2_state_valid=all(x['state_valid'][:10]),piper_state_valid=all(x['state_valid'][10:17]),
                uav_state_valid=all(x['state_valid'][17:]),go2_action_valid=all(x['action_valid'][:3]),
                piper_action_valid=all(x['action_valid'][3:10]),uav_action_valid=all(x['action_valid'][10:]),
                all_camera_mappings_valid=all(x['camera_valid']),eligible=x['eligible'],
                exported=x['source_frame_index'] in exported,
                rejection_reasons=';'.join(x['rejection_reasons'])))
        with (detail_dir/f"{r['source_episode']}.csv").open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
        total=r['total_grid_frames']
        action_bad=total-r['all_actions_valid_frames']
        row_bad=total-r['eligible_frames']
        state_bad=sum(not x['state_valid'] for x in details)
        state_action_bad=sum(not (x['state_valid'] and x['action_valid']) for x in details)
        assert len(details)==total and len(exported)==r['exported_frames']
        assert sum(not x['eligible'] for x in details)==row_bad
        rows.append(dict(task=f"{r['task_number']:02d}",source_episode=r['source_episode'],duration_s=r['duration_s'],
            file_integrity_passed=upload['tasks'][Path(r['source_path']).name]['file_integrity_passed'],
            upstream_recording_continuity_passed=r['upstream_recording_continuity_passed'],
            total_frames=total,action_invalid_frames=action_bad,action_invalid_percent=round(100*action_bad/total,2),
            state_invalid_frames=state_bad,state_or_action_invalid_frames=state_action_bad,
            state_or_action_invalid_percent=round(100*state_action_bad/total,2),
            row_invalid_frames=row_bad,row_invalid_percent=round(100*row_bad/total,2),
            internal_invalid_frames=r['internal_invalid_frames'],
            leading_invalid_frames=sum(x['frames'] for x in r['invalid_runs'] if x['position']=='leading'),
            trailing_invalid_frames=sum(x['frames'] for x in r['invalid_runs'] if x['position']=='trailing'),
            go2_action_invalid_frames=r['invalid_action_frames_by_robot']['go2'],
            piper_action_invalid_frames=r['invalid_action_frames_by_robot']['piper'],
            uav_action_invalid_frames=r['invalid_action_frames_by_robot']['uav'],
            exported_frames=r['exported_frames'],exported_percent=round(100*r['exported_frames']/total,2),
            short_segment_frames_excluded=r['excluded_short_segment_frames'],episodes=r['exported_segments'],
            longest_segment_frames=r['longest_segment_frames'],h20_windows=r['h20_windows'],
            verdict='无内部无效区间' if r['internal_invalid_runs']==0 else '存在内部记录间隔或观测条件不满足V3规则的区间'))
    with (root/'reports/completeness.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(batch=batch,
        standalone=True,active_v3_modified=False,source_count=count,raw_files_verified=upload['manifest_files'],
        raw_bytes_verified=upload['manifest_bytes'],raw_videos_decoded=count*3,all_file_integrity_passed=upload['all_task_artifacts_passed'],
        raw_frames=summary['raw_grid_frames'],exported_frames=frames,episodes=episodes,h20_windows=windows,
        h20_usable_sources=checks['h20_usable_sources'],tasks=rows,imputation_applied=False,
        action_horizon_is_training_parameter=True,physical_exposure_synchronization_independently_verified=False)
    (root/'reports/completeness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text=[f'# {batch}：逐条完整性检查','',
        f'单独保存于`{final}`，保留{count}条、{duration}秒。未合并或修改现有主V3。按15Hz时间轴保存单步state30、action14，训练chunk由loader决定，默认20步。','',
        f"当前原始目录清单的{upload['manifest_files']}个文件、{upload['manifest_bytes']:,}字节通过SHA256核对；{count*3}个原始MP4全量解码通过。文件完整性与时间连续性分别判断，H20窗口数不是完整性判据。",'',
        '|任务|总帧数|动作无效帧／比例|整行无效帧／比例|导出帧数／比例|最长段|H20窗口|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"|{r['task']}|{r['total_frames']}|{r['action_invalid_frames']}／{r['action_invalid_percent']:.2f}%|{r['row_invalid_frames']}／{r['row_invalid_percent']:.2f}%|{r['exported_frames']}／{r['exported_percent']:.2f}%|{r['longest_segment_frames']}|{r['h20_windows']}|")
    text+=['',
        '无效帧是该时间点未通过数据规则，并不代表视频图像丢失。整行无效计数包含首尾边界以及状态、动作、图像条件；同一帧有多个问题仍只计一次。',
        f'原始时间轴共{summary["raw_grid_frames"]:,}点，逐帧有效{eligible:,}点；其中{short}帧处于不足15帧的短段，仅留在审计表。最终导出{frames:,}帧、{episodes}个片段、{episodes*3}个MP4；默认H20有{windows:,}个窗口。','',
        '状态年龄≤75ms，运动标签位姿间隔≤120ms，Piper目标年龄≤100ms，图像年龄≤150ms，三路相机映射时间差≤100ms。无效处切段、不跨缺口拼接、不补值。最短15帧导出规则独立于训练chunk20；窗口在loader中按max(0,N-H+1)取样。','',
        '每个导出表格与审计、溯源值核对；全部导出视频解码，首、中、尾取样核对原始图像。训练归一化只使用本批导出帧计算。原始MCAP/ROS bag本次核对文件哈希，上传端的消息级检查见原始verification.json；本报告不将文件校验等同于独立的物理曝光同步或任务成功验证。','',
        '逐帧明细见reports/frame_quality/*.csv：保留全部视频时间点，分别列出三机状态、动作有效性、拒绝原因和是否导出。汇总CSV另列首部、尾部与内部无效帧数。时间缺口详情见reports/sources/*.json；其中数据流采样间隔的统计阈值与逐帧状态年龄检查含义不同。','']
    if (root/'reports/task02_camera_repair.json').exists():
        text+=['本批包含task02原始相机帧修复，详情见reports/task02_camera_repair.json。','']
    if (root/'reports/source_removal.json').exists():
        text+=['本批删除记录见reports/source_removal.json，当前成员以configs/source_inventory.json为准。','']
    text+=['## 首尾与内部时间连续性','',
        '|任务|首部无效帧|尾部无效帧|内部无效帧|Go2最大采样间隔ms|Piper状态最大间隔ms|Piper命令最大间隔ms|无人机位姿最大间隔ms|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r,s in zip(rows,sources):
        gaps=[s['streams'][k]['max_interval_s']*1000 for k in ('go2','piper_state','piper_command','uav_pose')]
        text.append(f"|{r['task']}|{r['leading_invalid_frames']}|{r['trailing_invalid_frames']}|{r['internal_invalid_frames']}|"+'|'.join(f'{g:.3f}' for g in gaps)+'|')
    text+=['','上传端完整性检查使用的阈值与V3逐帧对齐不同。例如上传端可能允许Go2采样间隔500ms、Piper命令间隔1000ms；通过上传端连续性检查并不代表通过V3的120ms运动标签间隔、100ms命令年龄检查。具体阈值见原始各任务completeness.json。','']
    (root/'reports/completeness_zh.md').write_text('\n'.join(text))
    readme=[f'# V3格式：{batch}独立批次','',
        f'当前保留{count}条录制，{frames:,}帧、{episodes}个Parquet、{episodes*3}个视频。15Hz、state30、action14，默认训练chunk20，可由loader覆盖；英文任务描述。','',
        f'LeRobot v2.1数据：`lerobot/local/heterovla_v3_train/`。无单独验证集，未合并现有主V3。H20有{windows:,}个窗口，覆盖{checks["h20_usable_sources"]}条来源。',
        '逐条文件完整性和时间连续性结果见下方报告；全部来源的原始时间点保存在aligned审计表，训练格式只导出满足当前V3规则的连续片段。','',
        '[逐条完整性报告](reports/completeness_zh.md) · [CSV汇总](reports/completeness.csv)','',
        '```python','import sys',f"sys.path.insert(0, {str(final/'scripts')!r})",'from dataset_v3 import OpenpiV3Dataset',
        f"data = OpenpiV3Dataset({str(final)!r}, 'train')",f'assert len(data) == {windows}',"assert data[0]['actions'].shape == (20, 14)",'```','',
        '数据集ID与V3一致，使用时显式指定本批root。configs记录当前来源及规则，aligned保留当前来源全部时间点审计，provenance对应导出片段。scripts为代码快照；报告与SHA256SUMS描述当前发布内容。','']
    (root/'README.md').write_text('\n'.join(readme))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--final',type=Path,required=True)
    args=parser.parse_args();write_reports(args.root.resolve(),args.final.resolve())
