#!/usr/bin/env python3
"""Publish documentation and integrity manifest only after validation succeeds."""
import argparse
from pathlib import Path
from alignment_paths import BENCHMARK_ROOT, DATA_ROOT
import hashlib,json,subprocess,sys
from prepare_v1 import dump,TASK

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('root',nargs='?',type=Path,help='Legacy positional output root')
ap.add_argument('--root',dest='root_option',type=Path)
ap.add_argument('--source',type=Path,default=BENCHMARK_ROOT)
args=ap.parse_args()
root=(args.root_option or args.root or DATA_ROOT).expanduser().resolve()
source=args.source.expanduser().resolve()
validation=json.loads((root/'reports/validation.json').read_text())
assert validation['format_validation_passed']
before=json.loads((root/'reports/benchmark_before.json').read_text())
after={str(f.relative_to(source)):{'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'mode':f.stat().st_mode}
       for f in sorted(source.rglob('*')) if f.is_file()}
assert before==after,'Source benchmark changed during preparation'
hash_check=json.loads((root/'reports/benchmark_sha256_check.json').read_text())
assert hash_check['all_match']
dump(root/'reports/benchmark_unchanged.json',{'unchanged':True,'metadata_files_compared':len(before),
    'sha256_manifest_files_checked':hash_check['files'],'all_source_sha256_match':True})
manifest=json.loads((root/'reports/episode_manifest.json').read_text())
summary={s:validation['datasets'][s] for s in ['train','val']}
table='\n'.join(f"| {r['source_episode']} | {r['split']} | {r['source_start_s']:.1f}–{r['source_last_s']:.1f} | {r['frames']} |"
    for r in manifest)
readme=f'''# HeteroVLA V1：三机联合运动模仿数据

任务指令（用户确认）：**{TASK}**。

本目录是独立派生版本。`{source}` 全程只读，其 {hash_check['files']} 个清单文件的 SHA-256 全部匹配，
全部 {len(before)} 个文件的大小、修改时间和权限与处理前一致。

## 使用入口

- 训练：`lerobot/local/heterovla_v1_train`，{summary['train']['frames']} 帧，{summary['train']['episodes']} 个连续片段。
- 验证：`lerobot/local/heterovla_v1_val`，{summary['val']['frames']} 帧，{summary['val']['episodes']} 个连续片段。
- 频率 10 Hz；三路 RGB 视频，640×480；状态 30 维；连续动作 14 维；推荐 H=10。
- 严格 H=10 窗口：训练 {summary['train']['h10_windows']}，验证 {summary['val']['h10_windows']}。
- 片段数不是独立示教数：共 6 段原始示教，默认使用 5 段；task_01 与 task_02 来自同一采集场次。
- `task_03` 因全段时钟告警隔离，不进入默认训练或验证。
- 训练与验证按原始采集场次分开；验证集不参与归一化统计。

## 必须区分的标签语义

**这是可加载训练的联合运动模仿数据，不是三台机器原始命令的完整行为克隆数据。**

`action[0:3]`：Go2 未来 0.1 秒里程计位移和偏航变化，转换为当前水平朝向坐标下的
平均 `vx, vy, yaw_rate`，单位 m/s、m/s、rad/s。坐标为前／左／逆时针，采用 yaw-only 变换，不是完整倾斜机体系。

`action[3:10]`：Piper 在区间终点之前最近一次已发送的 6 关节绝对目标（rad）与夹爪目标宽度（m）。
只接受 arm_connected、arm_enabled、command_sent 成立且为 joint 模式的记录。
目标最大陈旧时间 0.1 秒；这不是关节速度，也不是将实际反馈误用为命令。

`action[10:14]`：无人机未来 0.1 秒本地位置／偏航变化得到的水平朝向坐标平均
`vx, vy, vz, yaw_rate`，z 向上。其来源是实际运动，不能声称是原始飞控输入。

运动标签计算使用双端有效位姿插值；输入状态只取过去样本。
块内每一步的运动量相对于该步自己的朝向，不能把整个块都按块起点朝向转换。
采集中的 Go2 遥控输入到 SDK 命令映射尚未核实；无人机 `target_local` 是飞控回传的混合模式目标，
不直接改名成统一速度命令。`aligned/` 额外保留其目标值、逐维 ignore-bit mask、type_mask 和 coordinate_frame。
这些 FCU 目标经过 MAVROS 坐标转换；不能仅依据 coordinate_frame=1 就再盲目做一次 NED/ENU 交换。

默认训练导出只包含全部 14 维有效的样本，`action_valid` 全真；
逐维 `action_is_kinematic_proxy` 在 aligned/provenance 表中明确标出 3+4 个重建维度。
不可信目标只在审计表用数值零占位并置 mask=false，不进入默认训练。

## 状态与图像

完整字段顺序见 `configs/schema.json`。

- Go2：相对 episode 初始位置／朝向的水平位置、身体高度、roll/pitch、相对 yaw 的 sin/cos、过去位姿差分速度。
- Piper：6 关节实际角度与实际夹爪宽度。
- UAV：相对 episode 初始位置／朝向的水平位置、本地 z、roll/pitch、相对 yaw 的 sin/cos、过去位姿差分速度、armed、landed_state。
- 两台移动机器人的局部原点分别记录，**没有假设其共享世界坐标或空间标定**。
- 在线使用必须按同样定义保存 episode 原点、维护过去 0.1 秒状态历史，并按相同单位构建状态。
- front、wrist、uav 三路画面分别保存，openpi adapter 映射到 base_0_rgb、left_wrist_0_rgb、right_wrist_0_rgb。
  第三个槽名是兼容映射，不表示无人机相机是右腕相机。
- 按原始映射时间选取决策时刻之前最近图像；最大年龄 150 ms，三视角跨度不超过 100 ms。
  MP4 时间轴重新编码到 10 Hz，所有原帧索引和真实时间保留在 provenance。
- 这些规则保证“在提供的时钟映射下不读取未来观测”；源数据的物理跨设备同步误差尚未独立测定。

## 裁剪与划分

| 原录制 | 划分 | 源时间范围（秒，最后一帧时刻） | 帧数 |
|---|---|---|---:|
{table}

无效时间点把轨迹切断，绝不把缺口前后拼成连续运动；少于 H=10 的片段不导出。
Go2 body_height<0.23 m 的起身准备段、UAV 未连接／未解锁／里程计无效段、过期观测和无目标尾帧等被过滤。
有效等待和悬停保留，未进行 idle 删除或人为补造停止命令。
该 14 维动作不包含 Go2 站趴、飞控解锁或模式切换；不能用它声称学会这些离散控制。
成功率、最终抓取成功和完整任务边界未人工逐段标注；没有制造 success、reward 或失败标签。

## 加载与验证

数据按 openpi 锁定的 LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` 官方 writer 生成（v2.1）。
不要在未转换的情况下直接用仅支持 v3 的 reader。CPU 转换环境位于 `~/heterovla/.venvs/data-v1`（数据内 `.venv311` 为兼容入口），依赖冻结在 `configs/conversion_environment.txt`。

```bash
~/heterovla/.venvs/data-v1/bin/python ~/heterovla/data_collect/alignment/V1/verify_v1.py --root ~/heterovla-data/aligned/V1 --source ~/heterovla-data/raw/benchmark
```

```python
import sys
import argparse
from pathlib import Path
from alignment_paths import BENCHMARK_ROOT, DATA_ROOT
root = Path.home() / "heterovla-data/aligned/V1"
sys.path.insert(0, str(Path.home() / "heterovla/train/V1/scripts"))
from dataset_v1 import JointV1Dataset, as_openpi_sample
dataset = JointV1Dataset(root, split="train", horizon=10)
sample = dataset[0]                 # action: [10,14]; state: [30]
openpi_sample = as_openpi_sample(sample)
```

严格 loader 不返回 episode 尾部补齐的窗口，也不跨源缺口。
直接使用裸 LeRobotDataset 时，其尾部默认可能重复最后动作；V1 推荐使用上面的封装。
`reports/validation.json` 记录完整数值、视频解码、任务指令、分组、归一化及真实 DataLoader 检查。

## π0.5 LoRA 配置

`~/heterovla/train/V1/scripts/train_openpi_v1.py` 是现有 openpi JAX 环境的接入入口：注册配置、加载严格窗口、
读取只基于训练集计算的统计，再交给官方 trainer。
保留模型内部 32 维，padding 在模型变换中完成。没有额外关节 delta 变换。
此基线沿用官方逐维 flow-matching loss；若比较按机器人均衡的 loss，应作为单独实验实现，不能混淆。

```bash
# 在已经安装依赖、准备好 π0.5 权重的 openpi 环境中：
export OPENPI_ROOT=/absolute/path/to/openpi
python ~/heterovla/train/V1/scripts/train_openpi_v1.py --config-only
python ~/heterovla/train/V1/scripts/train_openpi_v1.py --exp-name v1_joint_lora
```

本次只准备与验证数据，未下载大模型权重、未启动完整 VLA 训练，也未将输出动作发送给机器人。
训练入口的完整运行状态以 `reports/openpi_config_check.json`（如存在）为准；数据 loader 验证不等于模型训练已验证。

## 文件用途

- `lerobot/`：可直接被固定版本 LeRobot loader 读取的训练／验证数据。
- `aligned/`：6 段录制的 10 Hz 逐帧派生表，包含 mask 和拒绝原因；隔离段不提供可信训练值。
- `provenance/`：每个导出 episode 对应的原始时间、源 CSV 行号、视频帧号、运动标签来源。
- `assets/heterovla_v1/norm_stats.json`：openpi 归一化统计，只用训练 split。
- `configs/`：任务、字段、阈值和版本信息。
- `scripts/`：原始发布脚本快照，仅作不可变 provenance；维护入口在 `~/heterovla/train/V1/` 与 `~/heterovla/data_collect/alignment/V1/`。
- `reports/`：各段质量、转换结果、源目录完整性、验证结果。
- `SHA256SUMS.txt`：V1 正式数据、脚本、配置和文档的校验清单；不含环境、缓存、日志和自身。

原始视频／高频 CSV／MCAP／bag 仍在 `~/heterovla-data/raw/benchmark`；本目录没有修改它们。
'''
(root/'README.md').write_text(readme)
dump(root/'STATUS.json',{'task':TASK,'format_ready':True,'lerobot_loader_verified':True,
    'training_target':'joint_kinematic_imitation','operator_command_bc_ready':False,
    'benchmark_unchanged':True,'default_fps':10,'state_dim':30,'action_dim':14,'horizon':10,
    'splits':summary,'full_vla_training_started':False,'physical_time_alignment_independently_verified':False,
    'quarantined_source_episodes':['task_03_1144-1300']})
lines=[]
# Do not traverse the package cache or virtual environments on the shared filesystem.
published_dirs=['lerobot','aligned','provenance','assets','configs','scripts','reports','previews']
published=[root/'README.md',root/'STATUS.json']
for name in published_dirs:published.extend((root/name).rglob('*'))
for p in sorted(published):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if any(x.startswith('.') or x=='__pycache__' for x in rel.parts):continue
    if p.name=='SHA256SUMS.txt' or p.suffix=='.log':continue
    if rel.parts[0] not in ['lerobot','aligned','provenance','assets','configs','scripts','reports','previews'] and p.name not in ['README.md','STATUS.json']:continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    lines.append(f'{h.hexdigest()}  {rel}')
(root/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'complete','root':str(root),'splits':summary,'checksummed_files':len(lines)},ensure_ascii=False,indent=2))
