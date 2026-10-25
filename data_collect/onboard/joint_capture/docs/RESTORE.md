# 恢复与使用

## 先决条件

先在健康存储上恢复。Unitree 当前文件系统有已记录的 ext4 错误；本仓库只保护代码，不能代替原始数据备份，也不证明文件系统已经修好。不要在原始数据尚未备份时根据本仓库执行格式化、刷机或离线修复。

机器人先保持未解锁、机械臂未使能。以下是恢复说明，本次归档没有执行这些安装/启动操作。

## macOS 控制台

安装 Python 3.10+ 后双击仓库根目录 `JointConsole-macOS.command`。本机环境、SSH 和网口设置见 [macOS 使用说明](MACOS.md)；机器人端仍使用下方原有部署与采集入口。

## Windows 控制台

安装 Python 3.10+ 与 OpenSSH，配置 SSH 别名 `p450` 和 `unitree`。私钥、known_hosts 与登录口令不在仓库中，需要使用你自己的凭据重新配置和核验主机身份。

先完成 [机器人与本机静态 IP 配置](NETWORK.md)。本机连接 P450 的网卡使用 `192.168.1.123/24`，连接 Unitree 的网卡使用 `192.168.123.222/24`，两张机器人专用网卡都不填写默认网关。上网网卡保留原有配置。

从仓库根目录运行，显式使用非 OneDrive 位置：

```powershell
& .\controller\scripts\Install-JointCapture.ps1 `
  -TargetRoot 'C:\RobotCapture' `
  -ManifestRoot 'D:\JointCaptureData\manifests' `
  -PythonPath 'C:\Path\To\python.exe'
```

安装后 `C:\RobotCapture\JointConsole.bat` 打开本机控制台，地址为 `http://127.0.0.1:8766/`。另有 JointStart、JointStop、JointStatus、JointRecover 入口。不要运行两套控制台争用同一状态目录。

当前配置将两端任务按本机时间生成同一 episode ID，并保存独立时钟映射；不把机器人系统时间强行改为本机时间。开始/停止的实际行为以所收录代码为准。历史 controller/README.md 中旧日期路径和旧版本说明仅作背景。

## Unitree

1. `robots/unitree/home/unitree/` 对应原 `/home/unitree/` 下的项目目录。先安装 DEPENDENCIES.md 中对应 JetPack、Python、SDK 和 ROS/系统依赖，再恢复代码。不要整目录覆盖一个仍在运行或已有新修改的机器人。
2. `heterovla-collection/config/collection.json` 保存现场路径、设备标识及遥操配置，换机时逐项核对。
3. C++ 桥由 onboard/CMakeLists.txt 构建；需要 SDK2 的 include、lib/<arch> 与 thirdparty/lib/<arch>。SDK2 完整副本目前未补齐，不能把部分目录当成可用 SDK。
4. `controller/remote/unitree/joint-can0-prepare`、`.service`、`.sudoers` 是受限 CAN 配置资源。helper 与 service 在盘点时与远端同名文件哈希一致；sudoers 是已有模板，不是本次读取 root 文件所得。部署时分别安装到 `/usr/local/sbin/joint-can0-prepare`、`/etc/systemd/system/joint-can0.service`、`/etc/sudoers.d/joint-can0`，按原权限设置并用 `visudo -cf` 检查；不要授予通用免密 sudo。
5. `Gamepad_PiPER_runtime/Gamepad_PiPER` 包含当前手柄代码和机械臂模型，cuRobo 包含模型与架构相关扩展。配对、Home 使能、关节模式与限速逻辑保留现状；本次没有调用机械臂控制命令。
6. `heterovla-control` / `heterovla-recorder` 是旧控制/录制工具，与当前采集主链分开保留。旧远程推理需要自行恢复 SSH 配置和 `/home/unitree/.config/heterovla/control-token`；仓库不提供真实凭据。

## P450

`robots/p450/` 的子路径对应原机器绝对路径。先按依赖表恢复上游仓库到原提交，再覆盖保存的现场修改。保留各项目许可证。

ROS 环境顺序是 `/opt/ros/noetic/setup.bash` → `/home/amov/prometheus_mavros/devel/setup.bash` → `/home/amov/p450_experiment/devel/setup.bash`。src/CMakeLists.txt 指向系统 catkin 的生成链接未归档，应在构建工作区时重新建立。

采集入口位于 `/home/amov/bin/p450_capture` 与 `/home/amov/p450_recording/`。完整无人机桌面启动链在 p450_experiment scripts/launch 下。视频流来自 spirecv-ros 的 video_streaming 与 SpireCV GStreamer，端口 8554；单独启动 p450_capture 并不等于启动 RTSP。

保留了桌面 autostart 两份配置：mid360_d435i 版本现场启用，另一份现场禁用。Fixcamera.service 现场为 disabled；不要因为文件存在就自动启用它。相机 launch 引用的 `/home/amov/emitter_off.json` 现场不存在，恢复后需核对该可选配置，不可假定已提供。

## 验收顺序

先检查两端 SSH、磁盘、相机/CAN 状态，再初始化，最后在获得操作者确认后做短采集、停止、文件校验和对时报告检查。代码快照单元测试通过不代表当前硬件、网络或文件系统通过实机验收。
