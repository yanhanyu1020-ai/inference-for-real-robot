# HeteroVLA 独立数据采集

本文件记录原独立 Unitree Go2 + Piper + 双 RealSense 采集方案、独立传输和 A800 离线 LeRobot 转换。模块和运行设备总览见 [README.md](README.md)；最新 P450–Unitree 联合采集见 [joint_capture](onboard/joint_capture/README.md)。

## 固定硬件映射

| 名称 | 安装位置 | RealSense 序列号 |
| --- | --- | --- |
| `front` | Go2 头部 | `254843065497` |
| `wrist` | Piper 腕部 | `254843064363` |

默认采集两路 RGB `640x480@30 FPS`，不采深度。Piper 可使用物理按钮拖动示教。采集 recorder 只读状态；本目录也保留手柄遥操和回放入口，运行这些入口会涉及控制命令。

## 部署

源码事实来源位于 A800：

```text
~/heterovla/data_collect
```

在 Unitree 上从 A800 拉取并编译：

```bash
ssh unitree
ssh -F ~/heterovla-control/ssh_config heterovla-a800 \
  'perl -ne '\''print if $. <= 500'\'' \
  ~/heterovla/data_collect/scripts/deploy_to_unitree.sh' \
  > /tmp/deploy_to_unitree.sh
bash /tmp/deploy_to_unitree.sh
```

部署仅复制文件和编译只读 DDS bridge，不启动采集。

## 采集

腕部相机重新接入后：

```bash
ssh unitree
cd ~/heterovla-collection
./onboard/unitree_go2/collection/collection_ctl.sh start episode_001 "任务描述" task_name
```

操作员使用实体遥控器控制 Go2，使用 Piper 物理按钮拖动示教。软件不会限制
两台机器人是否同时运动。

查看状态：

```bash
./onboard/unitree_go2/collection/collection_ctl.sh status
```

结束采集，只做本地收尾：

```bash
./onboard/unitree_go2/collection/collection_ctl.sh stop
```

数据保存在：

```text
/home/unitree/heterovla-data/datasets/<task>/<episode_id>/
```

采集格式为逐帧 PKL（两路 PNG + 同步状态），并保留 Go2/Piper 高频 CSV。

## 独立传输

采集完成后可随时执行：

```bash
./onboard/unitree_go2/collection/transfer_episode.sh episode_001 task_name
```

传输使用断点续传，Unitree 本地数据不会删除。A800 原始数据位置：

```text
~/heterovla-data/raw/<task>/<episode_id>/
```

## 转换为 LeRobot

A800 的转换环境需要安装兼容版本的 LeRobot、PyArrow、OpenCV 和 ffmpeg。
转换器不会自动修改 Python 环境。

```bash
cd ~/heterovla
python3 data_collect/server/convert_to_lerobot.py \
  --input ~/heterovla-data/raw/task_name \
  --output ~/heterovla-data/lerobot/dataset_name \
  --action-offset-ms 50
```

最终输出标准 LeRobot Parquet、双路视频和 meta 文件。原始 episode 始终保留。

## 安全边界

- `go2_state_bridge` 只创建 DDS Subscriber。
- `hetero_pkl_recorder.py` 只读取 Piper CAN 反馈。
- 采集 recorder 不调用机械臂控制接口；`piper_gamepad_teleop.py` 和 `replay_episode.py` 属于会发送控制指令的独立入口。
- `stop` 不上传数据。
- 传输脚本不删除 Unitree 数据。
- 发现已有 Piper 写入进程时，启动脚本报错但不会自动停止它。
