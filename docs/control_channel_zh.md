# GPU 服务器到 Go2/Piper 控制通道

这条通道用于从 GPU 服务器下发经过白名单限制的高层指令。它与
π0.5 的图像/状态/动作请求通道分开，便于独立验证通信和本体安全。

## 数据流

1. GPU 服务器在 `127.0.0.1:8766` 运行 WebSocket 控制服务。
2. Unitree 使用一把专用密钥，经 SSH 跳板机连接 GPU 服务器。
3. Unitree 把自己的 `127.0.0.1:18766` 转发到服务器的
   `127.0.0.1:8766`，不需要 Mac 参与运行时通信。
4. Unitree 上的机器端代理主动建立 WebSocket 长连接。
5. 操作者在 GPU 服务器运行 `robotctl.sh`，服务器转发指令并等待结果。

跳板机只转发端到端 SSH 流量，不运行 HeteroVLA 代码。

## 凭据和本地配置

令牌、真实主机地址和 SSH 私钥都不能提交到 Git。

- 服务器令牌：`~/.config/heterovla/control-token`
- Unitree 令牌：`~/.config/heterovla/control-token`
- Unitree 专用私钥：`~/.ssh/heterovla_tunnel_ed25519`
- 脱敏 SSH 模板：
  `onboard/unitree_go2/control/ssh_config.example`
- Unitree 实际 SSH 配置：`~/heterovla-control/ssh_config`

专用公钥需要加入跳板机和 GPU 服务器账号的
`authorized_keys`；私钥只保留在 Unitree。

## 部署

在操作端先把脱敏模板复制到 Git 仓库外，填入实际主机地址，然后：

```bash
GO2_SSH_CONFIG=/path/to/private/ssh_config bash scripts/deploy_unitree_control.sh
```

如果 Unitree 已经有 `~/heterovla-control/ssh_config`，可以不传
`GO2_SSH_CONFIG`，部署脚本不会覆盖现有私有配置。部署脚本只复制和编译，
不会启动代理或命令硬件。

## 启动顺序

在 GPU 服务器的仓库根目录启动控制服务：

```bash
bash scripts/control_server_ctl.sh start
bash scripts/control_server_ctl.sh status
```

在 Unitree 建立 SSH 转发（含策略通道；用 `control_up.sh` 一键完成，含看门狗自动重连）：

```bash
ssh -F ~/heterovla-control/ssh_config -fNT \
  -L 127.0.0.1:18766:127.0.0.1:8766 \
  -L 127.0.0.1:18765:127.0.0.1:8765 heterovla-a800
```

首次必须以 dry-run 启动机器端代理：

```bash
HETEROVLA_CONTROL_TOKEN="$(cat ~/.config/heterovla/control-token)" \
  ~/heterovla-client-env/bin/python \
  ~/heterovla-control/robot_command_agent.py \
  --uri ws://127.0.0.1:18766 --robot-id unitree-go2-01
```

在 GPU 服务器发送不会移动硬件的状态指令：

```bash
bash scripts/robotctl.sh system.status
```

只有在 dry-run 验证通过、Go2 四足着地、机械臂无人体或物体干涉、
Piper CAN 已正确激活后，才在 Unitree 启动命令末尾加 `--execute`。

## 当前白名单

- 无动作：`system.ping`、`system.status`
- 紧急停止：`system.stop_all`、`go2.stop`、`piper.stop`
- Go2：`go2.stand_up`、`go2.stand_down`（桥在跑时走 `go2.sock`，避免另起
  executor 抢租约）、`go2.action_chunk`
- Piper：`piper.enable`、`piper.disable`、`piper.go_zero`（安全偏移位
  `[0, 0.5°, -0.5°, 0, 0.5°, 0]`）、`piper.open_gripper`、`piper.close_gripper`
- 流式控制：`system.stream_start` / `system.stream_stop` / `system.stream_status`
  （启动/停止/查看 `piper_stream_loop.py`，连接策略服务器 18765）
- 分块控制：`system.control_start` / `system.control_stop` / `system.control_status`
  （常驻 `piper_chunk_loop.py`）+ `piper.action_chunk`（携带 JSON
  `{"actions": [[j1..j6°, 夹爪mm], ...]}`，每行 7 个有限数，1..5000 行）
- 采集：`system.record_start`（JSON `episode_id`，可选 `instruction`）/
  `system.record_stop` / `system.record_status`。开始后 chunk 执行器把
  `go2_cmd.csv`、`piper_cmd.csv`、`piper_state.csv` 写进同一 episode 目录。
- 摇操：`system.teleop_start` / `system.teleop_stop` / `system.teleop_status`
  （机载 `hetero_teleop_loop.py`：Unitree 遥控器 → 狗速度桥 + Piper CAN，
  并写入同一套 episode CSV）。与 Piper 流式/分块/示教采集互斥。
- 示教采集：`system.teach_start` / `system.teach_stop` / `system.teach_status`
  （机载 `hetero_teach_loop.py`：左摇杆 → 狗速度桥；Piper 使能后发一次
  `MotionCtrl_1` 拖动示教松闸，再只读关节。`ctrl_mode==2` 或
  `teach_status==recording` 时把实测关节写入同一套 episode CSV，
  `piper_replay.json` 来自 `piper_state.csv`）。与摇操/流式/分块互斥。
  不要在回放时再 record。

除 `piper.action_chunk`、`go2.action_chunk`、`system.record_start` 外，V1
不接受任意参数，不接受关节角、速度或力矩。
`piper.action_chunk` 的数据由 agent 本地限幅后执行，执行器侧还会按 SDK
限位与安全范围（关节限位、夹爪 0.5–65mm）二次钳位。

## 机械臂工程要点（踩坑记录）

1. 夹爪位置控制依赖整臂运动模式 0x01：每帧重发
   `ModeCtrl(0x01, 0x01, speed, 0x00)` + 200Hz 持续流式 + 力矩 3000（3 N·m），
   否则夹爪对位置指令完全无响应（曾误判为夹爪硬件故障）。
2. ModeCtrl 速度参数必须用 100：速度 20 跟不上变化目标，控制器报
   `REACH_TARGET_POS_FAILED` 并自动失能整臂导致坠落。
3. 精确零位在 J2/J3 限位边界，命令精确零位会触发同样故障并坠落；夹爪
   同理禁止命令 0mm（物理闭合点约 0.2mm）。`go_zero` 与夹爪指令均使用
   安全偏移。
4. 单写者原则：同一时刻只允许一个进程向 Piper 发指令流（流式/分块执行器
   互斥），并发 200Hz 流会打架导致掉臂或卡死。
5. 控制器状态可能谎报（电机位使能但控制器瘫在 STANDBY）：执行器先失能再
   使能恢复，并带 30 步无响应即中止的活体检测。
6. 正常位姿下使能一次就够：流式正常退出后臂保持位姿，不要对健康臂重复使能。

## 执行前提

- Go2 必须处于可以接收 SportClient 高层命令的模式。
- `--network-interface` 必须是连接 Go2 DDS 网络的网卡。
- Piper CAN 必须处于 `UP` 且波特率正确。控制程序不会自动改网卡或
  CAN 配置。
- 机器人旁必须有人能够使用实体急停；首次测试不能无人远程进行。
