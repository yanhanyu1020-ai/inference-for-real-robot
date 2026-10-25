# HeteroVLA

面向异构机器人协同的数据高效视觉—语言—动作学习项目。初始平台由 Unitree
Go2 移动底座、AgileX Piper 机械臂，以及无人机或外部视觉模块组成。

[English README](README.md)

## 运行环境边界

本仓库同时管理多台计算机使用的源代码。某份源码可以为了版本管理而保存在
GPU 服务器的仓库副本中，但这不代表它在 GPU 服务器上运行。

| 环境 | 目标系统 | 职责 | 仓库目录 |
| --- | --- | --- | --- |
| 机器人本体运行时 | Unitree Go2 Jetson，Ubuntu 20.04，aarch64 | DDS/SDK 数据采集、机器人适配、本地安全、动作执行 | `onboard/unitree_go2/`、`robot_gateway/` |
| GPU 服务器运行时 | A800（meta-eon-a800-006），x86_64，NVIDIA GPU | 数据转换、训练、策略服务、评测 | `heterovla/`、`data_tools/`、`scripts/` |
| 操作端电脑 | 能通过 SSH 访问机器狗和 GPU 服务器的电脑 | 部署、控制 episode、传输数据 | `scripts/` |

**`onboard/unitree_go2/` 和 `robot_gateway/` 下的代码部署并运行在机器狗
内部的 Jetson 系统上，不属于 GPU 服务器的运行程序。** GPU 服务器只保存这
部分源码的 Git 仓库副本，用于审查、版本管理和上传 GitHub。

## 仓库结构

- `onboard/unitree_go2`：执行目标为 Go2 机载 Jetson 的程序（命令 agent、
  Go2/Piper 命令执行器、采集器）
- `robot_gateway`：Piper 流式执行器（`piper_stream_loop.py` 面向策略服务器
  流式；`piper_chunk_loop.py` 面向分块远程控制）
- `heterovla`：协议定义与控制代理（`control_protocol.py`、`control_server.py`）
- `scripts`：部署和操作工具（`robotctl.sh`、`control_up.sh`、
  `mock_vla_server.py`、`send_go2_trajectory.py`、`send_hetero_demo.py`、
  `record_teleop.py`、`record_teach.py`）
- `tests`：协议测试
- `docs`：操作文档（`control_channel_zh.md`、
  `trajectory_control_zh.md` 轨迹从初始化到回位）
- `third_party/openpi`：上游 openpi 集成（计划中）

## 当前状态

远程控制与流式控制链路已端到端打通：

```
A800（模拟 VLA / 未来 π0.5 策略服务，ws://127.0.0.1:8765）
        │  动作块（openpi msgpack 协议，经 SSH 隧道 18765）
        ▼
Go2 Jetson：robot_command_agent（常驻，经 18766 broker）
        │  200Hz 流式 ModeCtrl+JointCtrl+GripperCtrl
        ▼
Piper 机械臂：关节 + 夹爪
```

Go2 速度流式通过一个**常驻 C++ 桥**实现（桥持有狗的控制租约，见下文）。

### 网络配置

Go2 Jetson 上建立两条 SSH 隧道，把 GPU 服务器的服务转发到本机端口
（`control_up.sh` 一键完成，且带断线自动重连看门狗）：

在unitree上一键恢复：
`bash ~/heterovla-control/control_up.sh`（CAN、双端口隧道看门狗、
干净环境启动 agent）。

```bash
ssh -F ~/heterovla-control/ssh_config -fNT \
  -L 127.0.0.1:18766:127.0.0.1:8766 \   # 控制 broker
  -L 127.0.0.1:18765:127.0.0.1:8765 \   # 策略服务
  heterovla-a800
```


### A800 操作命令

```bash
bash scripts/robotctl.sh <command>
```

- Go2：`go2.stand_up`、`go2.stand_down`、`go2.stop`
- Piper：`piper.enable`、`piper.disable`、`piper.go_zero`（安全回零位）、
  `piper.open_gripper`、`piper.close_gripper`、`piper.stop`
- 流式：`system.stream_start/stop/status`（对策略服务器运行
  `piper_stream_loop.py`）、`system.control_start/stop/status`（常驻块执行
  器）、`piper.action_chunk`（带 JSON `actions` 数据）
- Go2 速度：`system.go2_control_start/stop/status`（常驻 C++ 桥）、
  `system.go2_stream_start/stop/status`（策略流式）、`go2.action_chunk`
- 安全：`system.stop_all`
- 采集：`system.record_start/stop/status`、摇操 `system.teleop_*`、
  示教 `system.teach_*`（臂进拖动示教后只读关节，左摇杆仍开狗）

注意：`robotctl.sh` 会剔除 http(s)/all 代理环境变量后再连接本地 broker
（websockets 15 会读代理环境变量，否则本地连接会被劫持到 git 代理）。

### A800 上的 mock 策略服务器

```bash
python3 scripts/mock_vla_server.py                 # 机械臂：7 维轨迹
python3 scripts/mock_vla_server.py --robot dog     # 机器狗：3 维 [vx,vy,vyaw]
```

### Piper 机械臂流式

A800 mock VLA → websocket → Jetson 上 `piper_stream_loop.py`，以 200Hz
逐帧执行动作：

```bash
# A800
bash scripts/robotctl.sh system.stream_start       # 启动流式
bash scripts/robotctl.sh system.stream_status      # 查看日志
bash scripts/robotctl.sh system.stream_stop
```

### Go2 机器狗运动：完整流程（推荐从 A800 操作）

端到端链路：

```
A800: send_go2_trajectory.py / robotctl
        │  go2.action_chunk  [vx, vy, vyaw]（经控制 broker 18766↔8766）
        ▼
Jetson: go2_chunk_loop.py（按 50ms/行转发）
        │  unix socket → MOVE / STAND_UP / STOP
        ▼
Jetson: go2_velocity_bridge（常驻，持有 SportClient 租约）
        ▼
Go2 机身运动
```

#### 0. 前置条件

1. **关掉遥控器**（否则占租约，Move 返回成功但狗不动；错误码 3205/3207）。
2. A800 控制服务已起：`bash scripts/control_server_ctl.sh status`（监听 `127.0.0.1:8766`）。
3. Jetson 侧隧道 + agent 已上线（狗上执行 `~/heterovla-control/control_up.sh`），
   A800 上可见：
   ```bash
   bash scripts/robotctl.sh system.status
   # 期望：hostname=ubuntu execute=True；日志有 robot connected: unitree-go2-01
   ```
4. 注意：`system.control_start` 是 **Piper 臂**；狗要用 **`system.go2_control_start`**。

#### 1. 启动 Go2 控制栈（租约桥 + chunk 执行器）

在 **A800** 仓库根目录：

```bash
bash scripts/robotctl.sh system.go2_control_start
bash scripts/robotctl.sh system.go2_control_status
# 期望：bridge_pid=... chunk_pid=...（都不是 none）
```

`go2_control_start` 会在 Jetson 上拉起 `go2_velocity_bridge`（持租约）和
`go2_chunk_loop.py`。不要在 A800 本机跑 `./go2_velocity_bridge`（那是 Jetson
二进制，目录在狗上的 `~/heterovla-control/`）。

#### 2. 先站立（`send_go2_trajectory.py` 不会自动站立）

速度轨迹只发 `[vx,vy,vyaw]`，**趴着发速度无效**。站立必须走**已持租约的桥**：
`go2_control_start` 之后，`robotctl go2.stand_up` / `go2.stand_down` 会写入
`go2.sock`，不要另起 `go2_command_executor`。

```bash
# A800（桥已在跑时）
bash scripts/robotctl.sh go2.stand_up
# 或在 Jetson 上
python3 ~/heterovla-control/go2_send.py --stand-up
# 或干净环境：~/heterovla-control/go2ctl.sh stand_up
```

#### 3. 从 A800 下发速度轨迹

动作每行 3 维：`[vx, vy, vyaw]`（m/s、rad/s），1～5000 行，默认 50ms/行。
终端里 `ok: true / chunk delivered` 只表示**已交给机载执行器**；狗还要再跑
整段时长（例如 `--seconds 2` → 再看约 2 秒），可用 status 核对：

```bash
# 最稳妥的连通性验证：恒速前进 2 秒（约 0.6 m）
python3 scripts/send_go2_trajectory.py --vx 0.3 --seconds 2

# 内置演示（前进保持→停→转向→停→后退→停，各段约 2s；可加 --ensure-control 顺带确保控制栈已起）
python3 scripts/send_go2_trajectory.py --ensure-control

# 自定义 JSON（与 piper.action_chunk 同形，三维速度）
python3 scripts/send_go2_trajectory.py --file dog_traj.json
# 等价：bash scripts/robotctl.sh go2.action_chunk "$(cat dog_traj.json)"

# 执行后立刻看机载日志（有无 Move failed、chunk 是否跑满时长）
bash scripts/robotctl.sh system.go2_control_status
```

速度硬限幅：±0.5 m/s、±1 rad/s；桥在 400ms 无新指令时自动 STOP。

#### 3b. Go2 + Piper 协同演示（从 A800）

写自己的轨迹、以及从初始化到回位的完整命令（文首三条可直接跑的路径：脚本 demo / 摇操录制 / 摇操复现），见
[`docs/trajectory_control_zh.md`](docs/trajectory_control_zh.md)。

狗的速度通道和臂的关节通道是独立的，可以同时下发；下面这条默认场景是
**串行**的：先走完并等趴下动画结束，再动机械臂（不要把日志里的
`after=stand_down` 收包行当成已经趴好）。

1. 前进 2 s（`vx=0.3`）
2. 原地右转约 90°（开环，`vyaw=-0.5` rad/s × π s；打滑时角度会偏）
3. 趴下（`go2.stand_down`，**走已持租约的桥**，不要另起 executor）
4. 机械臂示教抓取（伸→合爪→抬；不是视觉抓取）

首次请在 **Jetson** 上把新的控制文件拉下来（A800 解析不了
`unitree`，`deploy_unitree_control.sh` 会失败）。**不要**再用
`robotctl go2.stand_up`：桥在跑时那会另起 `go2_command_executor` 抢租约，
卡 12 秒、狗也不动。

```bash
# 在狗上
bash ~/heterovla/scripts/pull_control_on_go2.sh
# 若该脚本还不在狗上，至少拉 chunk + episode_log；采集命令还要拉 agent 并 control_up.sh：
scp -F ~/heterovla-control/ssh_config \
  heterovla-a800:~/heterovla/onboard/unitree_go2/control/go2_chunk_loop.py \
  heterovla-a800:~/heterovla/onboard/unitree_go2/control/episode_log.py \
  heterovla-a800:~/heterovla/onboard/unitree_go2/control/robot_command_agent.py \
  heterovla-a800:~/heterovla/robot_gateway/piper_chunk_loop.py \
  ~/heterovla-control/

# 然后在 A800 重启桥 + 臂 chunk（加载新脚本）
bash scripts/robotctl.sh system.go2_control_stop
bash scripts/robotctl.sh system.control_stop
bash scripts/robotctl.sh system.go2_control_start
bash scripts/robotctl.sh system.control_start
```

```bash
# 默认整段：站立 → 前进 2s → 右转 90° → 趴下 → 臂抓取
python3 scripts/send_hetero_demo.py --ensure-control

# 边跑边记录狗+臂（写到 Jetson ~/heterovla-data/raw/<episode_id>/）
python3 scripts/send_hetero_demo.py --ensure-control --record
python3 scripts/send_hetero_demo.py --ensure-control --record demo_001

# 只跑机械臂（狗已经趴下/不动）
python3 scripts/send_hetero_demo.py --skip-dog
# 没有 piper_chunk_loop.py 时：回零 + 开爪 + 合爪
python3 scripts/send_hetero_demo.py --skip-dog --arm-basic

# 自定义轨迹（与各自 action_chunk 同形）
python3 scripts/send_hetero_demo.py --dog-file dog_traj.json --arm-file grasp.json

# 摇操录制 / 示教采集 / 复现：抄 docs/trajectory_control_zh.md 文首路径 B、D、C
python3 scripts/record_teleop.py --ensure-control --stand-up --record teleop_001
python3 scripts/record_teach.py --ensure-control --stand-up --record teach_001
python3 scripts/send_hetero_demo.py --ensure-control --together --no-stand-down \
  --dog-file go2_replay.json --arm-file piper_replay.json
```

机械臂还需要把 `piper_chunk_loop.py` 放到狗上（旧 agent 会 `system.control_start` 启动它）：

```bash
# 在狗上
scp -F ~/heterovla-control/ssh_config \
  heterovla-a800:~/heterovla/robot_gateway/piper_chunk_loop.py \
  ~/heterovla-control/piper_chunk_loop.py
```

没有这个文件时，演示会自动退回 `piper.go_zero` / 开合爪，夹爪仍会动。

急停仍用 `bash scripts/robotctl.sh system.stop_all`。臂的 `system.stream_start`
与 `system.control_start` 互斥；跑本演示前不要开流式。

#### 4. 停车 / 收尾

```bash
bash scripts/robotctl.sh go2.stop          # 桥在跑时走桥；否则才用独立 executor
# Jetson：python3 ~/heterovla-control/go2_send.py --stop
# Jetson：python3 ~/heterovla-control/go2_send.py --stand-down
bash scripts/robotctl.sh system.go2_control_stop   # 停 chunk + 桥（释放租约）
bash scripts/robotctl.sh system.stop_all           # 紧急停所有
```

#### 策略流式（可选，mock VLA → `go2_stream_loop.py`）

```bash
# A800
python3 scripts/mock_vla_server.py --robot dog
bash scripts/robotctl.sh system.go2_control_start
bash scripts/robotctl.sh system.go2_stream_start
bash scripts/robotctl.sh system.go2_stream_status
bash scripts/robotctl.sh system.go2_stream_stop
```

### Go2 速度控制（双终端，仅在 Jetson 上调试）

不经过 A800 `robotctl` 时，可在狗上直接：

终端 1 —— 租约持有者（保持运行）：
```bash
cd ~/heterovla-control && ./go2_velocity_bridge eth0 go2.sock
```

终端 2 —— 发指令：
```bash
python3 ~/heterovla-control/go2_send.py --stand-up
python3 ~/heterovla-control/go2_send.py --vx 0.5 --seconds 2
python3 ~/heterovla-control/go2_send.py --stand-down
python3 ~/heterovla-control/go2_send.py --stop
```

监视：`go2_monitor eth0`（`rt/wirelesscontroller` + `rt/sportmodestate`）。
SDK 二进制需干净环境（source 了 cyclonedds_ws/ROS 会 `free(): invalid pointer`）：
用 `~/heterovla-control/go2ctl.sh stand_up`。

## 工程经验（Piper 机械臂，踩坑总结）

1. **夹爪位置控制依赖整臂运动模式 0x01**：每帧重发
   `ModeCtrl(0x01, 0x01, speed, 0x00)` + 200Hz 持续流式 + 力矩 3000（3 N·m），
   否则夹爪无视指令（effort≈0），看起来像坏了。
2. **ModeCtrl 速度参数影响变化目标的跟踪**：速度 20 跟不上移动目标，控制器
   报 `REACH_TARGET_POS_FAILED` 并自动失能整臂。流式必须用速度 100。
3. **精确零位压在关节限位上**（J2 下限 0°、J3 上限 0°）：命令精确零位触发
   同样故障并坠落。`piper.go_zero` 已改为安全偏移位
   `[0, 0.5, -0.5, 0, 0.5, 0]`（度）。夹爪同理：永不发 0mm（物理闭合点约
   0.2mm），用 ≥1mm。
4. **单写者原则**：同一时刻只允许一个进程向 Piper 发指令流。两个并发
   200Hz 流会互相打架，导致掉臂或卡死。`system.stream_start` 与
   `system.control_start` 互斥。
5. **控制器状态可能谎报**（电机显示使能但控制器已瘫在 STANDBY）。流式执行
   器通过"先失能再使能"恢复；并有 30 步无响应即中止的活体检测。
6. **正常位姿下使能一次就够**：电机已使能时不要 `EmergencyStop` 再使能，否则会掉臂，看起来像“摔下来才开爪”。`piper.go_zero` / 开合爪 / chunk 执行器都会先读使能位，已使能则只刷新 ModeCtrl。

## 工程经验（Go2 机器狗，踩坑总结）

1. **运动需要控制租约**：`SportClient(true)` + `WaitLeaseApplied()`；没有
   租约时 Move 返回 rc=0 但狗无视。租约服务激活后站起/趴下也需要租约
   （错误码 3205 "Request denied by lease"；3207 "Lease already exists" =
   已被其他持有者占用，如遥控器）。需要一个常驻进程持有租约（双终端模式）。
2. **SDK 控制前必须关掉遥控器**——遥控器会拿走租约，且其锁定状态会阻止
   行走。
3. **SDK 二进制需要干净环境**（交互终端 source 了 cyclonedds_ws/ROS 会
   `free(): invalid pointer` 崩溃）。
4. **手柄原始输入在 `rt/wirelesscontroller`**（lx/ly/rx/ry/keys），狗的状态
   在 `rt/sportmodestate`（行走时 mode 仍为 0/IDLE——不要按 mode 判断，
   读速度反馈）。
5. **站立走桥，不要另起 executor**：`go2_control_start` 后租约已在
   `go2_velocity_bridge`。更新后的 agent 会把 `robotctl go2.stand_up` /
   `go2.stand_down` 转到桥上；旧 agent 仍会另起 `go2_command_executor` 并
   冲突。也可用 `go2_send.py --stand-up`。`send_go2_trajectory.py` 不自动站立；
   `send_hetero_demo.py` 默认会站起。
6. **`chunk delivered` ≠ 动作结束**：A800 立刻返回；机载按 50ms/行继续跑。
   用 `--vx 0.3 --seconds 2` 做连通性验证最直观；`system.control_start` 是臂不是狗。

## 数据管理原则

禁止向仓库提交原始 episode、视频、rosbag、模型 checkpoint、账号凭据、
设备地址和编译产物。即使仓库是 private，也只保存源码与文档，不保存机器人
采集数据或密钥。
