# P450–Unitree 联合采集业务核心

本目录从本地 `joint-capture-system` 工作副本归入主仓库，保留 2026-10-15 时的业务源码及当前尚未提交的 Mac 启动修复。此处是独立的 Python 项目，未改变采集或控制逻辑，也未部署到机器人。原本地工作副本与其未提交修改保持原样。

| 路径 | 运行设备与用途 |
| --- | --- |
| `JointConsole-macOS.command` | Mac：双击准备本机环境并启动网页控制台 |
| `controller/src/jointctl/` | Mac / Windows：联合启停、状态、时钟映射、预览、恢复与离线检查；包括 `console_web/` 静态资源 |
| `controller/scripts/` | Mac `.command`、Windows `.bat` / PowerShell 入口，以及采集诊断脚本 |
| `controller/config/default.json` | 操作端默认模板；本机配置与采集 manifest 在各自本机维护 |
| `controller/remote/unitree/` | Unitree 部署源：raw MCAP / PKL 采集、相机会话、Piper 安全遥操、快速停止、转换、校验、恢复及 C++ DDS bridge |
| `controller/remote/unitree_config/collection.json` | Unitree 实机快照的 v2 raw 采集配置 |
| `controller/remote/p450_recording/` | P450 部署源：ROS bag 采集、编排、快速停止和离线导出；包含入口与 CPU 测试 |
| `controller/tests/` | 控制台、协议、时钟和采集业务离线测试；平台和实机测试需对应环境 |
| `docs/` | 从源仓库保留的 Mac、网络、恢复与依赖说明，描述原部署环境 |

`controller/remote/` 是归档后的部署源码位置，不会由复制或打开控制台自动安装到机器人。源码保留原部署路径约定：Unitree 业务文件对应 `~/heterovla-collection/onboard/`，`unitree_config/collection.json` 对应 `~/heterovla-collection/config/collection.json`；P450 的包目录对应 `/home/amov/p450_recording/`，其 `bin/` 入口对应 `/home/amov/bin/`。Unitree 的 `joint-can0.service`、`joint-can0-prepare`、`joint-can0.sudoers` 是单独的设备配置源，安装需按源项目说明处理。旧 A800 独立 collection 部署脚本不用于安装本目录的联合采集方案。

操作端需要 Python 3.10+ 和系统 OpenSSH。保持本目录内部结构即可保留 Mac / Windows 启动器及 `__file__` 相对路径；从仓库根可先做纯本机 CLI 检查：

```bash
PYTHONPATH=data_collect/onboard/joint_capture/controller/src \
  python3 -m jointctl --help
```

Mac 安装与启动见 [docs/MACOS.md](docs/MACOS.md)，控制器命令见 [controller/README.md](controller/README.md)。源文档中的“仓库根”指本 `joint_capture/` 目录；其中 `robots/` 镜像及上游 SDK 内容只作原部署参考，此次不纳入。

外部依赖仍由对应设备提供：Unitree SDK2、Piper SDK、RealSense、Gamepad_PiPER / cuRobo；P450 ROS Noetic、现有 `p450_experiment` / Prometheus / FAST_LIO / MAVROS / Livox 采集栈。MCAP、LZ4、Zstandard 的版本清单位于 `controller/remote/unitree/requirements-raw.txt`。本目录不包含第三方源码、SDK、二进制、虚拟环境、设备系统镜像、认证文件或采集数据，不是机器人整机恢复包。

原 A800 独立采集源仍保留在 `../unitree_go2/collection/`。两种实现各自保留，避免新业务源码被旧采集实现覆盖；实际部署应根据已有设备环境选择对应方案。
