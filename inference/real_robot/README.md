# V3 三机真机推理

2026-10-24 的联调结果、框架图、协议字段与下次调用顺序见 [16.7 真机推理流程](README_16_7_WORKFLOW.md)。

控制器运行在 10.240.16.7：读取 Go2 前视、Piper 腕部、P450 图像和 30 维状态，调用本机 V3 WebSocket 策略，再将 14 维动作分发给两个机载网关。执行后读取新图像，保存预测动作、实际限幅动作和执行回执。当前本地模拟测试通过；真实硬件的依赖、接口、时延和执行效果必须在连接后验收。

2026-10-24 现场验收：Go2/Piper 与 P450 的三路真实 640×480 图像、30 维真实状态，已送入 V3 模型并连续完成两轮共 4 次只读诊断推理。输出均为有限 `(15,14)` 动作，模型耗时 96–125 ms；观测请求耗时 275–587 ms。没有执行机器人动作。结果见 `outputs/real_robot/20261024_141645_diagnostic/`、`outputs/real_robot/20261024_141807_diagnostic/`，汇总见 `.cache/real_robot/real_interface_check.json`。三路预览见 `outputs/real_robot/20261024_141643/`。

同日 15:58 追加两轮真实动作下行验证：新增只观测 `POST /validate_actions`，模型前 3 步分别传到 Unitree `(3,10)` 和 P450 `(3,4)`，逐值回传一致，并取回回执之后的新三图；没有调用执行器。P450 动作数值/限幅检查通过；Unitree 收到动作但因 Piper 首步目标与实测反馈差超过 0.35 rad 拒绝动作校验。结果见 `outputs/real_robot/20261024_155818_diagnostic/`、`.cache/real_robot/action_downlink_check.json`。动作已到达机载网关，不代表已发送到 SDK/CAN/MAVROS 或完成运动。

Go2 当前未站立且报告 `error_code=1001`；P450 报告 connected=true、armed=false、odom_valid=true、MANUAL、landed_state=1。诊断完整保留了这些真实状态及就绪问题。接口通过不等同于真机执行验收，执行模式仍关闭。链路往返曾测得约 182–387 ms，超过当前执行时钟校准要求；图像预览和只读诊断不会授予动作执行权限。

P450 的系统时间原为 1970 年，造成 MAVROS 位姿源时间异常。已在未解锁状态下停止本包观测进程、校正系统时间并恢复观测，位姿时间戳检查和上述 4 次真实输入推理均通过。`rtc0` 写入并读回正确；开机使用的 `rtc1` 在写入核对期间 SSH 再次断线，因此重启后的时钟保持尚未验证，不能承诺断电后自动恢复。当前网络仍有间歇中断。

今后需要校时时，可在 16.7 交互终端运行 `bash inference/scripts/repair_p450_clock.sh`。脚本将 16.7 当前时间传给 P450，并在终端请求 sudo 密码；密码不进入项目配置。脚本检查飞控已连接且未解锁，停止本包观测网关、观测栈及已核对的相机进程，校时后恢复观测。完成后重新验证位姿源时间戳和完整诊断链路。脚本校正系统时间，不保证 RTC 或 NTP 的持久化。

## 网络

现场 Windows 保持 WLAN 连接服务器，同时用机器人网卡连接设备。当前已确认 Go2 有线 SSH 为 `192.168.123.18:22`，电脑“以太网 4”为 `192.168.123.51`。P450 目标为 `192.168.1.11:22`，电脑“以太网 5”为 `192.168.1.123`；该网卡必须先从 Disconnected 变成 Up。

Windows 检查：

```powershell
Get-NetAdapter -Name "以太网 4","以太网 5" | Format-Table Name,Status
Test-NetConnection 192.168.123.18 -Port 22
Test-NetConnection 192.168.1.11 -Port 22
```

两台都通后在 Windows 建立反向 SSH 转发并保持窗口打开：

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -R 127.0.0.1:22271:192.168.123.18:22 -R 127.0.0.1:22272:192.168.1.11:22 16.7
```

`-NT` 登录后没有命令提示符是正常现象。只通 Go2 时可先只建立 22271 转发；P450 恢复后在另一个窗口用相同选项建立 22272 转发。不要在已有转发占用端口时重复绑定。

16.7 的 `.cache/real_robot/ssh_config` 定义 `v3-unitree` 和 `v3-p450`。私钥留在本机，权限 600，不包含在部署包内。机器人还必须授权这把密钥；网络通不等于认证成功。

## 检查与部署

以下命令从 16.7 仓库根目录执行：

```bash
source env.sh
python -m real_robot.manage probe
python -m real_robot.manage pack
python -m real_robot.manage deploy
python -m real_robot.manage start
python -m real_robot.manage status
```

可以用 `--kind unitree` 或 `--kind p450` 单独检查或部署。部署目录为各机载电脑的 `~/heterovla-live`，已有代码会备份；运行中的本包网关必须先停止才能更新。

默认环境路径来自现有项目，需由 probe 和实际导入确认：

| 设备 | 依赖及默认路径 |
| --- | --- |
| Go2/Piper | `/usr/bin/python3`，numpy、OpenCV、Piper SDK；`can0`；双相机；Unitree SDK2 `/home/unitree/unitree_sdk2-main`；CMake/C++ 编译环境 |
| P450 | `/usr/bin/python3`，numpy、OpenCV；ROS Noetic；`/home/amov/p450_experiment/devel/setup.bash`；rospy、MAVROS、prometheus_msgs；相机和状态话题 |
| 16.7 | `source env.sh`，运行中的 V3 策略服务 `ws://127.0.0.1:8000` |

路径可用 manage 的 `--unitree-python`、`--p450-python`、`--sdk-root`、`--p450-workspace` 覆盖。设备接口、相机序列号和话题见 `config/unitree.json`、`config/p450.json`，部署前应与实机核对。认证 token 从本机 `.cache/real_robot/token` 单独部署到机器人 `~/.config/heterovla-live/token`，不写入日志。

P450 缺少状态/定位节点时，本包提供独立观测栈，链接已有 `libuav_estimator.so`，不实例化 `UAV_controller`。启动包含 MAVROS、MID360、FAST-LIO、官方估计器；估计器按原有逻辑将定位数据送给飞控。它不加载控制器、解锁、切换模式或设置飞控参数。若已有同名观测话题发布者会拒绝重复启动：

```bash
python -m real_robot.manage sensors-build --kind p450
python -m real_robot.manage sensors-start --kind p450
python -m real_robot.manage sensors-status --kind p450
# 停止本包启动的观测栈：
python -m real_robot.manage sensors-stop --kind p450
```

该入口需要当前实机的 `/dev/ttyTHS0:921600`、MID360 配置和已有工作空间。`started` 仅表示进程启动，还需检查 `~/heterovla-live/p450_sensors.log` 与真实话题。当前相机另由 RGB roslaunch 提供，进程信息在远端 `camera_process.json`。P450 系统时间已于 2026-10-24 修复，但重启后的保持尚未验证；下次启动需重新检查，不要在正在采集时直接跳变系统时间。跨机时间仍通过单调时钟测量。

分别在 16.7 两个终端保持网关转发运行：

```bash
source env.sh
python -m real_robot.manage tunnel --kind unitree
```

```bash
source env.sh
python -m real_robot.manage tunnel --kind p450
```

这会将本机 18701/18702 转发到设备回环地址上的 8701/8702。无需将策略服务或网关暴露到公网。

## 图像、推理与执行

第三个 16.7 终端先取三路图像，无需无人机解锁或起飞：

```bash
source env.sh
python -m real_robot.controller --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token --observe --cycles 10
```

结果在打印的 `outputs/real_robot/<时间>/` 下：`index.html`、`latest/front.jpg`、`wrist.jpg`、`uav.jpg` 和状态 JSON。预览不保证全部模型状态已经满足推理条件。

预览 JSON 中 `raw_sensors` 保存收到的原始数值，`health` 保存时间戳新鲜度；图像与各状态流独立读取。Go2 原始向量为 `[x,y,z,yaw,body_height,roll,pitch]`，Piper 为 `[q1..q6,gripper]`，P450 `pose` 为 `[x,y,z,roll,pitch,yaw]`、`flags` 为 `[connected,armed,odom_valid]`。未收到的状态不填假值。

仅验证接口数据流、机器人尚未站立或无人机未解锁时，用以下诊断入口：

```bash
python -m real_robot.diagnostic --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token --cycles 2
```

它需要两端观测网关都已部署诊断接口，并且全部状态流真实可读。它保留实际 armed 值、记录所有就绪问题、组装真实三图和 30 维状态，保存模型 `(15,14)` 输出。两端必须为只观测模式，诊断响应没有可执行的 observation_id，程序也没有 prepare/commit 路径。地面状态超出训练有效条件、网络时序未通过等都会记录，诊断成功不能作为控制效果验收。任务 prompt 当前为拿取纸箱内绿色方块，可直接修改 `config/controller.json`。

进一步验证动作下行与回执，同时保持不执行：

```bash
python -m real_robot.diagnostic --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token --validate-actions --cycles 2
```

两端 `/status` 须包含 `validate_actions_v1` 能力。客户端发送真实预测动作的前 `execute_steps` 步并核对 `request_id`、逐值动作回传和非执行标志。网关使用真实反馈计算动作限幅候选值，但不调用硬件控制、不生成 plan_id、不创建执行线程。`received=true` 表示传输成功，`validation_passed` 单独表示数值、范围及初始关节差等检查是否通过；它不检查或授予飞行/运动就绪资格。拒绝时保留原因，不修改模型预测以求通过。每轮回执写入 `downlink_000.json` 等文件，随后取得采集时间晚于机载接收时间的新图，写入 `latest/`，完整结果记录在 `cycles.jsonl`。

模型输入遵循 V3 训练合同：Go2 10 维、Piper 7 维、无人机 13 维；Piper 关节用弧度、夹爪用米；三路 RGB 为 640×480。无人机的有效连接/解锁/里程计标志与 Go2 站立状态必须满足合同，不以零值补齐。`controller.json` 中 prompt 是示例任务，执行前必须改为现场任务。

只获取模型动作：

```bash
python -m real_robot.controller --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token
```

首次预热结果丢弃，随后重新获取观测推理。真实执行入口：

```bash
python -m real_robot.manage stop
python -m real_robot.manage start --execute
python -m real_robot.controller --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token --execute --cycles 1
```

执行要求 Go2 已站立、Piper 已使能且反馈健康，P450 已由现场操作员准备到 armed/OFFBOARD/空中状态；程序不自动解锁、起飞或切换模式。P450 默认高度范围 0.3–2.0 米、距本次起点半径 1 米，并拒绝并存的 MAVROS 指令发布者。发现已有控制程序时，先确认其作用和交接方式。

### 仅 Go2 + Piper 执行，无人机只观测

现场有线网络延迟低、但服务器 SSH 中转延迟高时，可将控制器移到现场 Windows，服务器 GPU 0 只运行推理服务。轻量客户端、三条转发和检查/执行命令见 [现场部署说明](README_FIELD_WINDOWS.md)。

使用 `--execution-target unitree` 保留三路真实图像和状态输入，只向 Unitree 发送模型前 10 维动作；P450 的后 4 维预测仅记录，不下发执行。无人机网关必须保持默认只观测模式，客户端对它只请求状态、重置观测原点和诊断观测，不发送 prepare、commit 或 stop。保留无人机实际未解锁标志及诊断问题，不要求其起飞。

先用不带 `--execute` 的命令检查时钟、图像、Go2 严格观测和模型推理，不执行动作：

```bash
python -m real_robot.controller --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token --execution-target unitree --cycles 1
```

Go2 应先由现场操作员准备到站立、反馈正常；Piper 已使能并处于 CAN 控制，周围留出动作空间。任务 prompt 需与现场任务一致。保持模型服务及两条 HTTP 隧道运行，然后仅切换 Unitree 网关：

```bash
python -m real_robot.manage stop --kind unitree
python -m real_robot.manage start --kind unitree --execute
python -m real_robot.controller --config inference/real_robot/config/controller.json --token-file .cache/real_robot/token --execution-target unitree --execute --cycles 1
```

不要使用不带 `--kind unitree` 的 `manage start --execute`，它会同时开启无人机执行。首次仅运行一轮，默认执行前 3 步（0.2 秒）。Ctrl+C、运行结束或异常均仅向 Unitree 请求停止；停止会锁存故障，下一次执行前需检查并重启 Unitree 网关。结束后可恢复其只观测模式：

```bash
python -m real_robot.manage stop --kind unitree
python -m real_robot.manage start --kind unitree
```

该模式保留执行时钟、观测新鲜度、三图时间差及动作新鲜度检查。时钟不确定度超过 40 ms 时会拒绝执行；仅诊断推理成功不表示已满足这些条件。日志的 `execution_targets` 和 `observation_only` 明确记录动作范围，`observation_only_status` 保存无人机执行开关关闭的检查结果。

模型输出 15×14 动作，默认仅执行前 3 步，每步 1/15 秒。速度默认乘 0.3 后限幅；Piper 还检查关节范围和变化率。两端先 prepare 检查、再定时 commit，要求动作新鲜度不超过 0.75 秒；网络、时钟误差或推理延迟过大时拒绝执行。软件不能保证独立相机物理曝光同步。

`cycles.jsonl` 记录推理时延、预测/实际动作、执行回执和前后观测元数据。执行后的 JPG 必须采集于动作结束之后。异常或控制器退出会请求两端停止；本机网关也有独立超时机制。软件停指令不等同于硬件急停，P450 的零速度请求仍依赖飞控及通信正常。

停止本包网关：

```bash
python -m real_robot.manage stop
```

停止不会关闭用户已有的策略服务器。日志位于各机器人 `~/heterovla-live/gateway.log`。故障锁存后需排查原因并重启网关。

## 本地验证

```bash
source env.sh
python -m pytest -q inference/tests/test_real_robot.py
HETEROVLA_LOOPBACK_TEST=1 python -m pytest -q inference/tests/test_real_robot.py
```

后一条使用本机回环 HTTP 和模拟设备测试完整获取观测、推理、执行回执、返回图像流程，不驱动真实机器人。2026-10-24：17 项通过，涵盖地面诊断保留真实状态且不提供执行凭据、单个设备故障不隐藏相机。两次时限测试曾失败；发现 OpenCV 默认 128 线程后将其限制为 1，复测全部通过。Unitree 两个桥接程序及 P450 独立估计器均已在实机编译成功。实际执行尚未验收。
