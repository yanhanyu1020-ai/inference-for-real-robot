# 数据采集 `data_collect`

本模块集中采集、原始数据校验、传输、离线转换和采集回放准备代码。非服务器程序按实际运行设备放在 `onboard/`；仓库保存源码不代表这些程序在 A800 上运行。

| 目录 | 运行位置与用途 |
| --- | --- |
| `server/` | A800：已传输 episode 的完整性校验、LeRobot 转换、数据集检查 |
| `config/collection.json` | 原独立 Go2 + Piper 双相机采集配置，供 Unitree 使用 |
| `onboard/unitree_go2/collection/` | Unitree Jetson：原 PKL 双相机采集、手柄遥操、校验、回放、上传与 DDS bridge 源码 |
| `onboard/unitree_go2/recorder/` | Unitree Jetson：独立 SDK2 DDS 高频 CSV recorder |
| `onboard/joint_capture/` | 最新 P450–Unitree 联合采集业务核心：Mac / Windows 控制台，以及 Unitree 和 P450 部署源码 |
| `scripts/` | 部署或操作入口；每个脚本的运行设备见下表 |
| `tests/` | CPU 离线采集格式与转换测试 |
| `alignment/` | 数据时间对齐和后续训练数据准备，详见其 README |

最新联合采集优先从 [onboard/joint_capture/README.md](onboard/joint_capture/README.md) 进入。`unitree_go2/collection` 保留 A800 原有独立采集实现，两套采集程序使用各自配置及部署目录，部署前先选择对应方案。

| 入口 | 执行位置 |
| --- | --- |
| `scripts/deploy_to_unitree.sh` | Unitree：从 A800 拉取独立 collection 源码并编译；仅部署，不开始采集 |
| `scripts/deploy_unitree.sh` | 能通过 SSH 访问 Unitree 的操作端：部署独立 DDS recorder |
| `scripts/record_teach.py`、`record_teleop.py` | A800 / 控制端：经 `inference/scripts/robotctl.sh` 调用机载控制链录制，可能启动机器人运动控制 |
| `scripts/prepare_collection_replay.py`、`sport_to_go2_replay.py` | A800 / 离线工作站：CSV 转回放 JSON，复用 `inference/onboard/unitree_go2/control/episode_log.py` |

独立采集的硬件、录制和转换命令见 [README_zh.md](README_zh.md)。从仓库根运行服务器端转换示例：

```bash
python3 data_collect/server/convert_to_lerobot.py \
  --input ~/heterovla-data/raw/task_name \
  --output ~/heterovla-data/lerobot/dataset_name --action-offset-ms 50
```

独立采集在 Unitree 部署到 `~/heterovla-collection`，其中 `config/`、`run/`、`build/` 保持同一模块根。新源码位于 `onboard/unitree_go2/collection/`；旧 `onboard/<file>` 入口保留为相对符号链接。仓库根旧 `heterovla-collection`、采集 `scripts/<file>` 也由兼容链接保留。

本次归位仅涉及源码和路径，没有部署、连接机器人或启动采集。CPU 验证需已有 Python 环境提供 NumPy、OpenCV；C++ bridge 编译及实机采集另需设备上的 Unitree SDK2、Piper SDK、RealSense 和对应运行环境。
