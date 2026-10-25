# Unitree Go2 机器狗端软件

[English README](README.md)

> **执行目标：**实际安装在 Unitree Go2 上的 Jetson 计算机（`Ubuntu
> 20.04`、`aarch64`）。本目录的源码会为了版本管理而保存在 h102 的仓库
> 副本中，但程序需要部署到机器狗并在机器狗系统内运行，不在 h102 上运行。

## 当前组件：SDK2 实时数据记录器

Recorder 直接订阅 Unitree SDK2 DDS Topic：

- `rt/wirelesscontroller`
- `rt/sportmodestate`
- `rt/lowstate`

记录内容包括遥控器输入、机身位置和速度、IMU、关节状态、估计力矩、
电机温度、足端力和遥控器原始数据。每条记录同时保存单调时钟和系统时间的
纳秒时间戳。

Recorder 只接收数据：它不创建控制命令 Publisher，也不会控制机器狗。

## 机器狗端依赖

- aarch64 平台上的 Ubuntu 20.04
- Unitree SDK2 2.0.0 源码，默认目录：
  `/home/unitree/unitree_sdk2-main`
- 通过 `eth0` 连接机器狗 DDS 网络
- CMake 和支持 C++17 的编译器

## 部署

在能够通过 SSH 访问机器狗的操作端电脑上执行：

```bash
GO2_HOST=unitree ./data_collect/scripts/deploy_unitree.sh
```

默认部署到机器狗：

```text
/home/unitree/heterovla-recorder
```

部署脚本会将当前仓库 commit 写入 `source_git_commit.txt`。新 episode 会
复制这个文件，使每条数据能够追溯到具体 Recorder 代码版本。

## 采集一条 episode

登录机器狗并启动：

```bash
ssh unitree
cd /home/unitree/heterovla-recorder
./go2_capture_ctl.sh start task_001 "机器狗站立再趴下"
```

使用实体遥控器操作机器狗，完成后停止：

```bash
./go2_capture_ctl.sh stop
```

随时查看状态：

```bash
./go2_capture_ctl.sh status
```

原始数据保存在机器狗内部：

```text
/home/unitree/heterovla-data/raw/<episode_id>/
```

episode 名称不可复用；控制脚本会拒绝覆盖已存在的数据目录。

## 输出文件

- `wireless_controller.csv`：摇杆和按键位掩码
- `sport_mode_state.csv`：机身位姿、速度、步态、IMU和足端力
- `low_state.csv`：高频 IMU、关节、电机、电源和遥控器原始状态
- `go2_cmd.csv`：下发给狗的速度指令（摇杆映射，松杆就是 0）
- `go2_replay.json`：复现用。`record_stop` 后从 `sport_mode_state.csv` 的位姿差分生成（速度积分对齐实测位移和朝向），不是摇杆指令
- `piper_cmd.csv`、`piper_state.csv`、`piper_replay.json`：机械臂指令与反馈
- `stick.json`：摇操时 recorder 写出的最新摇杆快照（无 `unitree_sdk2py` 时 teleop 读这个）
- `instruction.txt`：任务指令
- `start_time.txt`、`stop_time.txt`：采集起止时间
- `source_git_commit.txt`：部署时对应的源码版本
- `summary.json`：时长和消息数量
- `recorder.log`：运行状态和累计消息数量

这些异步原始数据必须保留。重采样以及 LeRobot/openpi 转换属于服务器端
离线任务，不能放进实时采集回调。
