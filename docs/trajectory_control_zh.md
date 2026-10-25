# 用轨迹控制 Go2 机器狗和 Piper 机械臂

两台机器分工固定：不要在 A800 上跑机载二进制，也不要在狗上跑 `robotctl`。

| 机器 | 目录 | 职责 |
| --- | --- | --- |
| **Unitree**（Go2 机载 Jetson） | `~/heterovla-control/` | CAN、SSH 隧道、agent、速度桥、chunk 执行器 |
| **A800**（GPU 服务器） | `~/heterovla/` | 控制 broker、写/下发轨迹、`robotctl` |

急停（随时，A800）：

```bash
bash scripts/robotctl.sh system.stop_all
```

安全：狗四足着地，臂周围无人体；旁侧有人能按实体急停。首次不要无人远程试。  
**关掉遥控器**（只有下面路径 B 摇操、路径 D 示教时才打开）。遥控占租约时 Move 成功但狗不动（3205/3207）。  
不要同时开 `system.stream_start` 和 `system.control_start`（两条 200 Hz 流会抢 Piper，掉臂）。
示教采集时也不要开 chunk / 摇操：它们每帧 `ModeCtrl(0x01)`，会把示教键顶掉。

---

## 四条路径（先抄这段）

控制栈没起来时，先做一次第 2 节的初始化，再跑下面的命令。  
`chunk delivered` 只表示已交给机载，狗/臂还要按行跑完整段。

### A. 脚本 demo（站立 → 走 → 趴 → 抓）

```bash
# Unitree（关遥控；机载目录 ~/heterovla-control/）
bash ~/heterovla-control/pull_control_on_go2.sh   # 首次或脚本有更新
bash ~/heterovla-control/control_up.sh

# A800，仓库根目录 ~/heterovla
bash scripts/control_server_ctl.sh start          # 已在跑可改 status
python3 scripts/send_hetero_demo.py --ensure-control
```

回位见第 6 节。轨迹格式见第 1 节。

### B. 摇操录制

遥控器顺序：**关** → 起桥站立 → **开** → 摇 → 停采后再**关**。反了会抢租约。

```bash
# Unitree
bash ~/heterovla-control/pull_control_on_go2.sh
bash ~/heterovla-control/control_up.sh

# A800
bash scripts/control_server_ctl.sh restart         # teleop/record 在白名单里
python3 scripts/record_teleop.py --ensure-control --stand-up --record teleop_001
# 终端提示后再开遥控；摇完按 Enter（无 TTY 必须加 --seconds 20）
```

按键、文件、排障见第 5c 节。episode id 不可复用，失败换 `teleop_002`。

### D. 示教采集（拖臂 + 左摇杆开狗，同时记）

臂在拖动示教里记实测关节（软件 `MotionCtrl_1` 松闸，或按键 `ctrl_mode=2`），**不是**回放后再录。狗用左摇杆，和臂同一 episode 时钟。遥控器顺序与路径 B 相同。

```bash
# Unitree
bash ~/heterovla-control/pull_control_on_go2.sh
bash ~/heterovla-control/control_up.sh

# A800
bash scripts/control_server_ctl.sh restart         # teach/teleop/record 在白名单里
python3 scripts/record_teach.py --ensure-control --stand-up --record teach_001
# 终端提示后：开遥控 → 拖臂（还发硬就按示教键）→ 边走边拖 → Enter 结束
```

狗上若还是旧的 `pull_control_on_go2.sh`，先从 A800 把该脚本拷过去再 pull（需要 `hetero_teach_loop.py`）。

录完若臂还在按键示教里，再按一次示教键**退出**（`ctrl_mode` 不要停在 2），再走路径 C 复现（把 `teleop_001` 换成 `teach_001`）。细节见第 5e 节。

### C. 摇操 / 示教复现（关遥控）

```bash
# Unitree：把三个文件拷到 A800（A800 解析不了主机名 unitree）
scp -F ~/heterovla-control/ssh_config \
  ~/heterovla-data/raw/teleop_001/go2_replay.json \
  ~/heterovla-data/raw/teleop_001/piper_replay.json \
  ~/heterovla-data/raw/teleop_001/sport_mode_state.csv \
  heterovla-a800:~/heterovla/

# A800
python3 scripts/sport_to_go2_replay.py sport_mode_state.csv -o go2_replay.json
python3 scripts/send_hetero_demo.py --ensure-control --together --no-stand-down \
  --dog-file go2_replay.json --arm-file piper_replay.json
```

`--together --no-stand-down` 必须都加：摇操时狗和臂是同时动的，不能走完再趴再动臂。  
转换原理、文件含义、变体见第 5d 节。回位见第 6 节。

---

## 1. 轨迹怎么写

两条独立通道，JSON 形状不同。可以串行（demo 默认：走完并趴下再动臂），也可以分别下发。

### 1.1 机器狗：速度轨迹

每行 `[vx, vy, vyaw]`，单位 m/s、rad/s。机载 **50 ms/行**（约 20 Hz）。

```json
{
  "actions": [
    [0.3, 0.0, 0.0],
    [0.3, 0.0, 0.0],
    [0.0, 0.0, 0.0]
  ],
  "before": "stand_up",
  "after": "stand_down"
}
```

| 字段 | 含义 |
| --- | --- |
| `actions` | 协议单块 1～5000 行；复现脚本会自动切块，总上限 60000 |
| `before` / `after` | 可选：`stand_up` / `stand_down` / `stop`，走路前/后经桥执行 |

手写连通性验证用窄限幅：`vx,vy` ±0.5 m/s，`vyaw` ±1 rad/s。  
摇操复现用宽限幅（`send_hetero_demo.py`）：`vx,vy` ±1，`vyaw` ±3.5。  
最后一行写成 `[0,0,0]`。趴着只发速度无效，先 `before=stand_up`。

开环转向：时长 ≈ `|角度(rad)| / |角速度|`。右转 90°：`vyaw=-0.5`，约 3.14 s → 63 行。打滑会偏，不是闭环朝向。

```bash
# A800：只写文件、不下发
python3 scripts/send_go2_trajectory.py --vx 0.3 --seconds 2 --dump dog_fwd.json
python3 scripts/send_hetero_demo.py --dump-dog dog_approach.json
```

### 1.2 机械臂：关节轨迹

每行 `[J1, J2, J3, J4, J5, J6, 夹爪]`。关节**度**，夹爪 **mm**。机载 **5 ms/行**（200 Hz）。

```json
{
  "actions": [
    [0.0, 10.0, -20.0, 0.0, 15.0, 0.0, 2.0],
    [0.0, 30.0, -50.0, 0.0, 25.0, 0.0, 2.0],
    [0.0, 30.0, -50.0, 0.0, 25.0, 0.0, 50.0]
  ]
}
```

| 关节 | 范围 |
| --- | --- |
| J1 | -150 ~ 150° |
| J2 | 0 ~ 180° |
| J3 | -170 ~ 0° |
| J4 | -100 ~ 100° |
| J5 | -70 ~ 70° |
| J6 | -120 ~ 120° |
| 夹爪 | **1 ~ 65 mm**（禁止 0 mm，物理闭合点约 0.2 mm） |

安全回零（`piper.go_zero`）是 `[0, 0.5, -0.5, 0, 0.5, 0]` 度，不是精确全零（精确零压在 J2/J3 限位上会掉臂）。

相邻行要密，200 Hz 下跳变太大会跟丢。不要第一帧就开爪。

```bash
python3 scripts/send_hetero_demo.py --dump-arm arm_grasp.json
```

默认示教抓取路点（不是视觉抓取）：

```text
紧凑收拢，爪关   [0, 10, -20, 0, 15, 0, 2]
悬停，爪仍关     [0, 30, -50, 0, 25, 0, 2]
悬停开爪         [0, 30, -50, 0, 25, 0, 50]
下落             [0, 45, -75, 0, 35, 0, 50]
合爪             [0, 45, -75, 0, 35, 0, 2]
抬起             [0, 25, -45, 0, 20, 0, 2]
```

回位路点见第 6 节。

---

## 2. 初始化（先 Unitree，再 A800）

路径 A/B/C 里已经包含这些命令。这里是拆开说明。

### 2.1 Unitree：隧道 + agent

A800 **解析不了**主机名 `unitree`，拉文件必须在狗上做。

```bash
# Unitree（机载代码在 ~/heterovla-control/，不是 ~/heterovla/）
bash ~/heterovla-control/pull_control_on_go2.sh
# 若该脚本还不在狗上：从 A800 仓库拷到 heterovla-control，再执行
scp -F ~/heterovla-control/ssh_config \
  heterovla-a800:~/heterovla/scripts/pull_control_on_go2.sh \
  ~/heterovla-control/
bash ~/heterovla-control/pull_control_on_go2.sh

bash ~/heterovla-control/control_up.sh    # CAN + 双端口隧道看门狗 + agent --execute
```

`control_up.sh` 会重启 agent。只更新了 chunk 脚本时，不必重启 agent，在 A800 重启对应 control 即可。

### 2.2 A800：broker + 控制栈

```bash
# A800，~/heterovla
bash scripts/control_server_ctl.sh status    # 没起来就 start；改了白名单要 restart
bash scripts/robotctl.sh system.status
# 期望：hostname=ubuntu  execute=True；日志有 robot connected: unitree-go2-01

bash scripts/robotctl.sh system.go2_control_start
bash scripts/robotctl.sh system.go2_control_status
# 期望：bridge_pid=...  chunk_pid=...（都不是 none）

bash scripts/robotctl.sh system.control_start
bash scripts/robotctl.sh system.control_status
# 期望：pid=...；不要和 system.stream_start 同时开
```

`robotctl.sh` 会清掉 http(s)/all 代理再连本地 broker，否则 websockets 会被 git 代理劫持。

- `system.go2_control_start`：机载 C++ 速度桥（持 Sport 租约）+ `go2_chunk_loop.py`
- `system.control_start`：机载 `piper_chunk_loop.py`（臂，不是狗）

桥在跑时，站起/趴下必须走桥（`go2.action_chunk` 的 `before`/`after`）。  
**旧 agent** 上的 `robotctl go2.stand_up` 会另起 `go2_command_executor` 抢租约，卡 12 秒、狗也不动。

---

## 3. 站立

趴着发速度无效。站立动画约 3 秒，再发走路轨迹。

```bash
# A800（推荐：经 chunk，不另起 executor）
bash scripts/robotctl.sh go2.action_chunk '{"actions":[[0,0,0]],"before":"stand_up"}'

# agent 已更新、桥在跑时也可以
bash scripts/robotctl.sh go2.stand_up

# Unitree 调试
python3 ~/heterovla-control/go2_send.py --stand-up
# 干净环境：~/heterovla-control/go2ctl.sh stand_up
```

---

## 4. 下发机器狗轨迹

连通性验证（路径 A 的狗段也可以单独跑）：

```bash
# A800
python3 scripts/send_go2_trajectory.py --vx 0.3 --seconds 2
python3 scripts/send_go2_trajectory.py --file dog_fwd.json
bash scripts/robotctl.sh go2.action_chunk '{"actions":[[0.3,0,0],[0,0,0]],"after":"stand_down"}'
bash scripts/robotctl.sh system.go2_control_status
```

| 日志 | 含义 | 能不能开始动臂 |
| --- | --- | --- |
| `chunk received: N rows ... after=stand_down` | 刚收到，**开始走** | 否 |
| `after: STAND_DOWN` | 走完，开始趴下 | 再等约 3.5 s 动画 |
| `after pose complete: STAND_DOWN` 或 `chunk done` | 趴下结束 | 可以 |

不要把收包行里的 `after=stand_down` 当成已经趴好。`send_hetero_demo.py` 会按上表等。

---

## 5. 下发机械臂轨迹

确认狗已停稳（或已趴下）后再发。时长 ≈ 行数 × 5 ms；`chunk delivered` 会立刻返回，臂还在动。

```bash
# A800
python3 scripts/send_hetero_demo.py --ensure-control          # 整段协同
python3 scripts/send_hetero_demo.py --skip-dog                # 狗已趴好，只跑臂
python3 scripts/send_hetero_demo.py --dog-file dog_approach.json --arm-file arm_grasp.json
python3 scripts/send_hetero_demo.py --skip-dog --arm-basic    # 没有 chunk 时：回零 + 开合爪
```

小 JSON 也可以：`bash scripts/robotctl.sh piper.action_chunk --data-file arm_grasp.json`。  
大文件不要 `$(cat ...)`，Linux 单参数上限 128KiB。

---

## 5b. 边跑 demo 边记录

chunk 执行器和 Go2 DDS recorder 共用 `~/heterovla-recorder/active_episode`，不要另起 Piper 采集进程。

```bash
# Unitree
bash ~/heterovla-control/pull_control_on_go2.sh
bash ~/heterovla-control/control_up.sh

# A800
bash scripts/control_server_ctl.sh restart
bash scripts/robotctl.sh system.go2_control_stop
bash scripts/robotctl.sh system.control_stop
python3 scripts/send_hetero_demo.py --ensure-control --record demo_001
```

demo 的 `go2_replay.json` 来自当时下发的速度指令；摇操的狗 JSON 来自实测位姿，不要混用语义，见第 5d 节。  
`--arm-basic` 不走 chunk，记不到臂。`system.control_start` 与 `system.stream_start` 仍互斥。

单独开关：`system.record_start '{"episode_id":"demo_001","instruction":"hetero demo"}'` / `record_status` / `record_stop`。

---

## 5c. 摇操录制

总流程就是文首路径 B。路径 D 示教采集同样要开遥控（只当摇杆输入）。这是文档里**要把遥控器打开**的两条流程。

| 步 | 遥控器 | 原因 |
| --- | --- | --- |
| 起桥、站立 | **关** | 桥先拿到 Sport 租约 |
| 摇操 | **开** | 只当摇杆输入；租约仍在桥上 |
| 复现 | **关** | 否则 Move 成功但狗不动 |

| 输入 | 作用 |
| --- | --- |
| 左摇杆 ly / lx | 狗前进 `vx` / 转向 `vyaw`（不横移） |
| 右摇杆 rx / ry | 臂 J1 / J2 |
| L1 + 右摇杆 | 臂 J3 / J4 |
| L2 + 右摇杆 | 臂 J5 / J6 |
| R1 / R2 | 夹爪开 / 关 |
| select | 狗 STOP，臂保持 |
| L2 + A | 趴下 / 站起切换（与出厂遥控相同，趴下后再按一次站起） |
| START | 仅在已趴下时站起 |

`record_teleop.py` 会停掉两个 chunk（避免两路 CAN/CSV 互抢），**桥保持**。站起来后再开遥控，看 `system.teleop_status` 里摇杆数值在变。

机载需要 `piper_sdk`（系统 Python）。摇杆优先订 `rt/wirelesscontroller`；没有 `unitree_sdk2py` 时改读 recorder 的 `stick.json`。  
`teleop.log` 仍报无法 import DDS 且进程退出：狗上还是旧脚本，再 `pull_control_on_go2.sh` + `control_up.sh`。

等价手动开关（一般不用）：

```bash
bash scripts/robotctl.sh system.teleop_start
bash scripts/robotctl.sh system.record_start '{"episode_id":"teleop_001","instruction":"joystick teleop"}'
# ... 摇操 ...
bash scripts/robotctl.sh system.record_stop
bash scripts/robotctl.sh system.teleop_stop
```

狗上目录 `/home/unitree/heterovla-data/raw/teleop_001/`。缺 `go2_replay.json` / `piper_replay.json` 说明摇操没在写，看 `~/heterovla-control/teleop.log`。录完复现走路径 C / 第 5d 节。

---

## 5e. 示教采集

总流程就是文首路径 D。和路径 B 共用狗的速度桥和 `record_start`，差别只在臂：

| | 路径 B 摇操 | 路径 D 示教 |
| --- | --- | --- |
| 臂怎么动 | 右摇杆积分后 `JointCtrl`（CAN 位置控制） | 软件拖动示教（`MotionCtrl_1` 松闸）或按示教键（`ctrl_mode=2`） |
| CAN | 每帧写 `ModeCtrl(0x01)+JointCtrl` | 启动时使能 + 一次 `MotionCtrl_1(0x01)`，之后**不再**发运动指令 |
| `piper_replay.json` 来源 | `piper_cmd.csv` | `piper_state.csv`（实测角） |
| 右摇杆 / R1 R2 | 控臂 / 夹爪 | **忽略**（循环里发 `GripperCtrl`/`ModeCtrl` 会把示教顶掉） |

按键示教（`ctrl_mode=0x02`）软件切不进去。`system.teach_start` 会停掉 chunk / stream / 摇操，否则它们每帧 `ModeCtrl(0x01)`，臂拖不动。只打开 CAN、不使能的话臂会停在 **standby（抱闸）**，按键也进不了示教——这就是 `teach.log` 一直 `ctrl_mode=0` 的原因。现在采集进程会先使能，再发一次固件拖动示教。

同一 `record_start` 窗口里：左摇杆开狗、拖臂，两路 CSV 共用时间戳。不要先走完再单独拖臂再拼文件——那才会在狗速度里多出一段 0。

```bash
bash scripts/robotctl.sh system.teach_start
bash scripts/robotctl.sh system.record_start '{"episode_id":"teach_001","instruction":"drag teach"}'
# 开遥控，拖臂（还发硬就按示教键），边走边拖
bash scripts/robotctl.sh system.record_stop
bash scripts/robotctl.sh system.teach_stop
```

看是否进了示教：`system.teach_status` 的 `teach.log` 里应有 `teach=recording` 或 `ctrl=2(teach)`。一直 `arm=brake` / `ctrl=standby` 就是没松闸：确认狗上已 `pull_control_on_go2.sh` 更新过采集脚本，再按一次示教键。

复现前必须再按示教键**退出**（`ctrl_mode=1`），否则 `JointCtrl` 无响应。然后走路径 C（`--together --no-stand-down`）。不要在回放时再 `--record` 当训练数据。

---

## 5d. 轨迹复现

总流程就是文首路径 C。和脚本 demo 同一条 chunk 通道，三点不同：轨迹来自 episode；`--together` 同时下发；`--no-stand-down` 不趴。摇操已停掉 chunk，`--ensure-control` 会重新拉起。复现是开环，打滑和起始朝向不同时落点不会完全重合。

### 复现到底喂哪两个文件

episode 里 CSV 是原始记录，**不下发**。真正喂给执行器的只有两个 JSON：

```text
摇杆指令     →  go2_cmd.csv            松杆就是 0，狗可能还在滑 → 复现不用
狗实际位姿   →  sport_mode_state.csv  ──► go2_replay.json     → 喂给狗（50 ms/行）
臂关节指令   →  piper_cmd.csv         ──► piper_replay.json   → 喂给臂（5 ms/行；摇操）
臂关节反馈   →  piper_state.csv        摇操对照用；示教采集时这一份才是复现输入
```

| 文件 | 角色 |
| --- | --- |
| `go2_replay.json` | **狗的复现输入**。`source` 应为 `sport_mode_state` |
| `piper_replay.json` | **臂的复现输入**。摇操 `source` 为 `piper_cmd.csv`；示教采集 `source` 为 `piper_state.csv`（拖动时的实测角） |
| `sport_mode_state.csv` | 狗实测位姿；`record_stop` 会用它重写 `go2_replay.json`。路径 C 再跑一遍转换，避免拿到摇杆版 JSON |
| `go2_cmd.csv` | 摇杆日志。若 JSON 里 `"source":"go2_cmd.csv"`，说明没转成功，必须用 CSV 重做 |

JSON 形状和第 1 节相同，多一个 `source`：

```json
{"actions": [[0.12, 0.0, -0.05]], "source": "sport_mode_state"}
{"actions": [[0.0, 12.3, -18.1, 0.0, 14.0, 0.0, 2.0]], "source": "piper_cmd.csv"}
```

核对（A800，`~/heterovla`）：

```bash
python3 - <<'PY'
import json
for name, dt in (("go2_replay.json", 0.05), ("piper_replay.json", 0.005)):
    d = json.load(open(name))
    n = len(d["actions"])
    print("%s  source=%s  rows=%d  %.1fs" % (name, d.get("source"), n, n * dt))
PY
```

两边时长差几秒正常（停采、死区）。单块 5000 行，总上限 60000 行；`send_hetero_demo.py` 会自动切块。

### `sport_to_go2_replay.py` 在算什么

脚本调用 `episode_log.replay_from_sport_state`。输入是 `sport_mode_state.csv` 的世界系位姿 `(px, py, yaw)`，输出是机载 `Move` 能吃的机体系速度 `[vx, vy, vyaw]`，网格 50 ms。

**不用** CSV 里的 `velocity_x` / `yaw_speed`，也**不用**摇杆。原因：松杆后指令是 0，狗还在滑；IMU 角速度噪声大、还会被钳，积分对不齐实测朝向。位姿差分保证 `sum(v)*dt` 对齐录制时走过的路和转过的角。

**1. 时间对齐。** CSV 采样不均匀。从首帧到末帧按 50 ms 画格子，每个格子取「不超过该时刻的最后一条样本」（零阶保持）。

**2. 世界系位移 → 机体系速度。** 相邻两帧，上一帧朝向为 yaw：

```text
vx_world = (px[k] - px[k-1]) / dt
vy_world = (py[k] - py[k-1]) / dt
dyaw     = unwrap(yaw[k] - yaw[k-1])     # 折进 (-π, π]，避免 ±π 跳变算成转了一圈

# 绕上一帧朝向转回机体系（SE(2) 正运动学的逆）
vx   =  vx_world * cos(yaw) + vy_world * sin(yaw)
vy   = -vx_world * sin(yaw) + vy_world * cos(yaw)
vyaw =  dyaw / dt
```

正运动学是：`ẋ = vx cosψ − vy sinψ`，`ẏ = vx sinψ + vy cosψ`，`ψ̇ = vyaw`。  
按 50 ms 执行这些 `[vx, vy, vyaw]`，开环积分会回到录制的位移和净转角（地面打滑除外）。

**3. 静止死区。** `|vx|,|vy| < 0.04 m/s` 且 `|vyaw| < 0.08 rad/s` 才写成 `[0,0,0]`，避免站着微抖在复现时蠕动；慢速转向不会被抹掉。

**4. 朝向回灌。** 速度钳到 `vx,vy ≤ 1`、`vyaw ≤ 3.5` 之后，`sum(vyaw)*dt` 可能小于整段实测 `yaw_end − yaw_start`。把差额加到还没顶满限幅的帧上，使积分转角对齐实测净转角。

这就是路径 C 里这一行：

```bash
python3 scripts/sport_to_go2_replay.py sport_mode_state.csv -o go2_replay.json
```

`record_stop` 在狗上也会自动跑一遍；已有 episode 或 JSON 仍是摇杆版时，在 A800 再跑即可。也可以对整目录：`python3 scripts/sport_to_go2_replay.py /path/to/teleop_001`。

臂不需要这套转换：`piper_replay.json` 就是 `piper_cmd.csv` 里的目标关节角，200 Hz 原样回放。

### 复现脚本做了什么

`send_hetero_demo.py --together --no-stand-down`：

1. 读两个 JSON 的 `actions`，按狗/臂限位再钳一次
2. `--ensure-control` 拉起两个 chunk（摇操把它们停过）
3. 默认先 `before=stand_up`（已站好加 `--no-stand-up`）
4. 两个线程同时 `go2.action_chunk` / `piper.action_chunk`
5. 超 5000 行按块切，块间按 `行数 × dt` 等；大 JSON 走 stdin，不塞 argv

**不要**用 `send_go2_trajectory.py --file go2_replay.json`：最多 5000 行，且 `vyaw` 钳到 ±1，转向会被削掉。  
**不要** `robotctl.sh piper.action_chunk "$(cat piper_replay.json)"`：会 `Argument list too long`。

只一侧：加 `--skip-arm` 或 `--skip-dog`。  
直接下发（不切块、不同时发臂）：`bash scripts/robotctl.sh go2.action_chunk --data-file go2_replay.json`。

看是否在跑：`system.go2_control_status` / `system.control_status`。`chunk received` 是开始跑这一块；`holding last pose` 是臂锁在最后一帧。结束后回位走第 6 节，不要直接 `piper.go_zero`。

| 现象 | 处理 |
| --- | --- |
| 狗不动，Move 成功 | 关遥控 |
| 走完才动臂 / 中途趴下 | 加上 `--together --no-stand-down` |
| 转向比录制小很多 | 不要用 `send_go2_trajectory.py`；JSON 的 `source` 应为 `sport_mode_state` |
| chunk `pid=none` | `--ensure-control` |
| 臂 chunk 不健康 | 狗上 `pull_control_on_go2.sh` + `control_up.sh` |
| 落点差一截 | 开环打滑，或 JSON 仍是摇杆版 → 重跑 `sport_to_go2_replay.py` |
| 臂开头一跳 | 当前角和录制第一帧差太远，先收到接近的姿态 |

---

## 6. 回位

chunk 会以 200 Hz **锁在最后一帧**。这时不要 `piper.go_zero`：会另起 CAN 写者抢总线，超时。

```bash
# A800：插值回位（抬起 → 收拢 → 安全回零，约 2 s）后下发
python3 - <<'PY'
import json, math
waypoints = [
    [0.0, 25.0, -45.0, 0.0, 20.0, 0.0, 2.0],
    [0.0, 10.0, -20.0, 0.0, 15.0, 0.0, 2.0],
    [0.0, 0.5, -0.5, 0.0, 0.5, 0.0, 2.0],
]
steps = 200
rows = []
for i in range(len(waypoints) - 1):
    a, b = waypoints[i], waypoints[i + 1]
    for s in range(steps):
        t = (s + 1) / float(steps)
        e = 0.5 - 0.5 * math.cos(math.pi * t)
        rows.append([round(a[j] + (b[j] - a[j]) * e, 3) for j in range(7)])
with open("arm_return.json", "w") as f:
    json.dump({"actions": rows}, f, separators=(",", ":"))
print(len(rows), "rows")
PY
bash scripts/robotctl.sh piper.action_chunk --data-file arm_return.json
```

等约 2 s。日志应有 `holding last pose`。

备选：`system.control_stop` 后 `piper.go_zero`（臂停在当前角，不会自己掉）。

狗要站起来：`go2.action_chunk '{"actions":[[0,0,0]],"before":"stand_up"}'`。  
走回出发点只能再发反向速度（开环，对不齐）。只停车：`robotctl go2.stop`。

收工：先按需 `after=stand_down`（等约 4 s），再

```bash
bash scripts/robotctl.sh system.control_stop
bash scripts/robotctl.sh system.go2_control_stop
```

---

## 7. Unitree 本机调试（不经 A800）

速度桥必须单独占一个终端（持租约）。

```bash
# 终端 1
cd ~/heterovla-control && ./go2_velocity_bridge eth0 go2.sock

# 终端 2
python3 ~/heterovla-control/go2_send.py --stand-up
python3 ~/heterovla-control/go2_send.py --vx 0.5 --seconds 2
python3 ~/heterovla-control/go2_send.py --stand-down
python3 ~/heterovla-control/go2_send.py --stop
```

source 了 cyclonedds_ws/ROS 会 `free(): invalid pointer`，用 `~/heterovla-control/go2ctl.sh stand_up`。  
监视：`go2_monitor eth0`。行走时 mode 仍可能是 0/IDLE，看速度反馈，不要用 mode 判断。

---

## 8. 常见失败

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 狗不动，Move 成功 | 遥控器占租约，或没站立 | 关遥控；先 `before=stand_up` |
| 录完遥控蹲不下 | 速度桥占 Sport 租约；`stop_all` 会另起 executor 抢租约 | 先 `teach_stop` + `go2_control_stop`；**不要用** `system.stop_all`；开遥控等 3 s |
| 录制中遥控 L2+A 蹲不下 | 速度桥持租约，出厂指令进不了 SportClient | 更新 `hetero_teleop_loop.py` 后 `pull_control_on_go2.sh`；L2+A / START 会转发到桥 |
| `system.teleop_start` 被拒 | A800 broker 还是旧白名单 | `control_server_ctl.sh restart` |
| `system.teach_start` 被拒 | 同上，或狗上 agent 未更新 | `pull_control_on_go2.sh` + `control_up.sh` + broker restart |
| 示教键按了仍拖不动 | 臂在 standby 抱闸，或 chunk/摇操还在发 `ModeCtrl(0x01)` | `pull` 新脚本后 `teach_start`；日志应有 `MotionCtrl_1 drag-teach` |
| teach.log 一直 standby / brake | 狗上还是只读旧脚本，或没使能 | `pull_control_on_go2.sh` + `control_up.sh`；看 `teach=` / `arm=` |
| teleop.log 无法 import DDS | 没有 `unitree_sdk2py` | 更新 `hetero_teleop_loop.py` 后改读 `stick.json` |
| episode 里没有 `go2_replay.json` | 摇操/示教进程没在跑 | `teleop.log` / `teach.log` |
| 复现时走完才动臂 / 中途趴下 | 忘了 `--together --no-stand-down` | 两个都加 |
| 复现转向比录制小很多 | 用了 `send_go2_trajectory.py` | 用 `send_hetero_demo.py`；`source=sport_mode_state` |
| 复现时 chunk `pid=none` | 摇操停过执行器 | `--ensure-control` |
| `stand_up` 卡 12 秒 | 旧 agent 另起 executor 抢桥 | `go2.action_chunk` + `before`；或更新 agent |
| 臂在走路时抓取 | 把收包日志当成趴好了 | 等 `after: STAND_DOWN` 再加 3.5 s |
| `piper.go_zero` 超时 | chunk 还在 200 Hz 锁最后一帧 | 发回位 chunk，或先 `system.control_stop` |
| 掉臂 / `REACH_TARGET_POS_FAILED` | 精确零位或夹爪 0 mm | 安全回零；夹爪 ≥ 1 mm |
| `system.status` 连不上 | 代理、隧道或 broker | `robotctl` 已清代理；查 `control_up.sh` |
| `Argument list too long` | 大 JSON 塞进 argv | `send_hetero_demo.py` 或 `--data-file` |

通道与令牌细节见 [control_channel_zh.md](control_channel_zh.md)。
