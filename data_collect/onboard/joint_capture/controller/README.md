# P450–Unitree 联合采集控制器

2026-09-14 实机验证补充：已完成联合短采集与原始数据检查，P450 保持 1970 年系统时间也可建立相对时间映射。最近一次实机样本 `joint_20260914_050432_18d7` 的共同区间为 41.881 秒。时钟模型误差估计不等于物理同步精度。本轮额外收紧了故障恢复、时钟模型一致性和数据完整性检查。

本目录提供 Windows / macOS 端的联合启动、快速停止、状态、恢复、只读时钟探测和离线对齐工具。运行时只依赖 Python 3.10 及以上版本的标准库和系统 OpenSSH。远端采集脚本增加了快速停止和独立后处理入口，不裁剪或覆盖 rosbag、PKL、CSV 等原始数据，也不改机器人运动控制或系统时间。

macOS：双击仓库根目录 `JointConsole-macOS.command`，或先运行本目录 `scripts/Install-JointCapture.command` 配置本机记录路径。完整步骤见 [macOS 使用说明](../docs/MACOS.md)。下方 PowerShell 与 BAT 安装说明适用于 Windows。

## 安装

两台采集机开机并不是安装前提。审核通过后，在 PowerShell 中从仓库执行：

```powershell
& .\scripts\Install-JointCapture.ps1
```

默认目标为 `D:\OneDriveData\Desktop`。安装器通过 `(Get-Command python).Source` 解析 Python 的绝对路径并写入 `D:\OneDriveData\Desktop\JointCapture\python_path.txt`。当前预期解析结果是 `C:\Users\1\anaconda3\python.exe`，但安装时仍以实际命令解析结果为准。

安装是可恢复的。替换前，已有的 `JointCapture` 整目录以及 `JointStart.bat`、`JointStop.bat`、`JointStatus.bat`、`JointRecover.bat`、`joint_start.ps1`、`joint_stop.ps1`、`Joint_episode_counter.txt` 会复制或移动到 UTC 时间戳目录 `JointCaptureBackup-<时间戳>`。`D:\OneDriveData\Desktop\joint_manifests` 完全不参与替换，既有 manifest、`raw` 内容和其他历史文件保持原位；旧计数器也保留在桌面并备份。

如需先在临时目录验证安装包，可显式给出目标，命令不会接触真实桌面：

```powershell
& .\scripts\Install-JointCapture.ps1 -TargetRoot 'C:\Temp\JointCaptureDesktop'
```

## 日常操作

### 重启后的自动准备（2026-09-14）

本机 Start 在远端录制前先调用 P450 的 `p450_capture prepare`，并在 Unitree 执行 `sudo -n /bin/systemctl start joint-can0.service`。准备失败不会继续启动该端录制。已有完整 P450 链路由远端复用，部分链路或无人机已解锁时仍保留远端拒绝机制。

Unitree 已部署 root 所有的 `/usr/local/sbin/joint-can0-prepare`、`/etc/systemd/system/joint-can0.service` 和 `/etc/sudoers.d/joint-can0`。授权仅允许 unitree 用户启动这一固定服务，不允许通用免密 sudo；CAN 固定为 can0、1 Mbps，不发送电机或机械臂运动指令。服务启用开机启动，Start 也会再次检查；已 UP 但速率不符会拒绝改动。未通过整机重启实验验证，不要将单次服务启动检查当成开机验收。

双击 Start 只需输入 Task，Instruction 默认使用 Task。冷启动准备加初始化可能需要数分钟；看到“联合采集已开始”后再执行采集动作。Stop 提示“录制已停止 · 待后处理”表示两端已停止，可以开始下一条；这不代表数据已通过验收。

### 本机网页控制台

安装后双击 `D:\OneDriveData\Desktop\JointConsole.bat`，或打开 `http://127.0.0.1:8766/`（后台须先启动）。无需 Node.js、云服务或外网。入口启动独立后台并打开浏览器；不会自动开始录制。网页关闭/刷新不终止采集，重新打开会读取当前实际状态。

无人机实时画面：设备初始化后，点击“开启预览”。控制台通过无人机现有 rosbridge 的 `sensor_msgs/CompressedImage` 话题直接订阅彩色 JPEG，默认每秒 5 帧、仅保留最新画面，不写入采集文件。关闭预览、切到后台页面会取消订阅；返回页面后恢复连接。源时间戳必须连续推进才能显示“实时”，2 秒无新画面会清除旧图，连接中断自动重连。该预览不启动相机或录制，关闭预览不停止采集。

默认配置为 `p450.preview.url=ws://192.168.1.11:9090`、`topic=/uav1/camera/color/image_raw/compressed`、`fps=5`；帧率可设为 1–10。地址须为私网 IPv4 的 ws/wss 服务，页面只允许连接这个明确地址。PC 有线口须能访问无人机网段，ROS 彩色相机和 rosbridge 须已运行。采集控制仍沿用原有 SSH 配置，预览可独立验证。协议参考：[rosbridge 官方说明](https://github.com/RobotWebTools/rosbridge_suite/blob/ros2/ROSBRIDGE_PROTOCOL.md)。

填写任务名称与采集说明，点击“开始联合采集”；等待双方就绪。点击“停止录制”后等待两端停止确认，即可开始下一条。设备空闲时再点击“处理待办”，执行原始数据收尾、完整性校验和对时报告生成。此流程不自动导出视频；需要视频时自行另行导出。准备、停止、处理待办过程中不要重复运行其他控制命令。网页和 CLI 共用操作锁，发生冲突会拒绝第二个操作而不是排队重放。

设备卡显示当前/上次远端数据路径、数据增长、整机 RAM 和数据盘剩余空间。RAM 已用为 MemTotal-MemAvailable。状态与资源目标每 2 秒采样，失败后退避；数据超过 6 秒或探测失败标记过期，未知不等于零。设备不在线时仍可打开界面。后台任务日志位于 `joint_manifests\.console\jobs`。

页面断线时不要把上次状态当成实时状态。后台被终止或电脑重启不会自动停止远端录制；重新打开后核对状态，必要时使用恢复与停止。若后处理过程中断，待设备空闲后重试“处理待办”；只有本次生成的报告通过，才可认定数据验收通过。

仅允许本机访问，变更接口要求同源与令牌，不开放局域网。不要把 8766 端口代理或转发给外部网络。更新程序前请先停止当前采集，再退出旧后台；安装不代替后台重启。P450 30 分钟限制仍保留，第一版没有自动联停看门狗。

### 备用命令入口

四个批处理入口可从任意当前目录运行。它们使用安装目录的绝对配置路径，并固定把 manifest 写入 `D:\OneDriveData\Desktop\joint_manifests`，不会落到当前目录的 `episodes`。

```powershell
D:\OneDriveData\Desktop\JointStart.bat
D:\OneDriveData\Desktop\JointStatus.bat
D:\OneDriveData\Desktop\JointStop.bat
D:\OneDriveData\Desktop\JointRecover.bat
```

`JointStart.bat` 双击后不再询问任务名称：本机生成带 UTC 偏移量的本地时间采集 ID（例如 `joint_20260926_143015_UTCp0800_ab12`），两端和本机 manifest 使用同一 ID。默认远端任务目录也以该 ID 命名；可选采集说明只记录在元数据中，不改变目录名。CLI 无参数启动同样不需交互；旧自动化仍可显式传 `--instruction`、`--task`：

```powershell
$app = 'D:\OneDriveData\Desktop\JointCapture'
$python = (Get-Content -LiteralPath "$app\python_path.txt" -Raw).Trim()
$env:PYTHONPATH = "$app\src;$env:PYTHONPATH"
& $python -m jointctl --config "$app\config\default.json" --manifest-root 'D:\OneDriveData\Desktop\joint_manifests' start
```

启动成功表示两端在同一个新 episode 下产出数据，并已通过至少三轮时钟探测的启动门槛、记录共同起点。启动质量不达标会回滚。请等启动命令返回后再停止；如果启动进程仍存活，停止会拒绝与它竞争。启动窗口意外关闭、原进程退出后，可以停止清理。

`JointStop.bat` 进入控制器时立即取桌面时间作为候选 `T1`，核对所有权后并行停止。重试保留首次已保存的 `T1`。一端不可达不再阻止停止另一端已确认属于本次采集的会话；失败保留 active 指针，恢复连接后可重试。明确发现其他 episode 时仍拒绝全部停止。已提前退出的本次会话会记录诊断；不会向无归属的空闲机器发送停止命令。

停止成功后仅完成两端录制进程退出、原始数据落盘和监控关闭，标记 `postprocess=pending`，不等待视频导出或完整校验。`manifest.state=complete` 只表示停止流程完成，数据验收仍以之后生成的 `alignment.json` 为准。设备空闲时使用网页“处理待办”，或运行 CLI `finalize [episode_id]`；失败可重试。该后处理不自动导出视频。

`JointRecover.bat` 可复用已存在的本次 active manifest。指针缺失但有本地历史 manifest 时，允许从单端明确匹配的 episode 恢复；没有本地记录时仍要求两端都在采集且 ID 一致。不会猜测归属或停止外来会话。

## Manifest 与有效区间

每个 episode 位于：

```text
D:\OneDriveData\Desktop\joint_manifests\<episode_id>\
```

`manifest.json` 保存 start/status/stop 的 stdout、stderr、退出码、远端原始目录、`T0`、`T1`、状态和诊断。`clock_p450.jsonl` 与 `clock_unitree.jsonl` 是持续时钟探测的原始样本，只追加、不重写。`alignment.json` 是可重复生成的派生报告。

共同有效区间严格定义为 `[T0,T1)`，即包含 `T0`，不包含 `T1`。两端进程可以更早启动、更晚退出；区间外的数据是保留的首尾冗余，不会被本工具删除。

对已完成 episode 生成对齐报告：

```powershell
& $python -m jointctl --config "$app\config\default.json" --manifest-root 'D:\OneDriveData\Desktop\joint_manifests' align <episode_id>
```

该命令会只读检查远端数据摘要并写入对应 episode 的 `alignment.json`。若只想基于已经保存的 manifest 和时钟样本生成报告，可增加 `--no-remote-inspect`，但缺少流摘要时会按设计返回 timing-degraded。

P450 优先使用 ROS header timestamp；bag timestamp 回退只供诊断，不能通过必需数据流验收。Unitree 分别读取 PKL 内 front、wrist 相机自己的 `wall_time_ns`，不再用文件名代替两路时间戳；CSV 使用行内 `wall_time_ns`。转换不改动源时间戳。

`coverage_interval` 仅为首尾覆盖交集。只有必需数据流齐全、至少两条样本、时间递增、最大间隙不超过 250 ms、完整覆盖请求区间且时钟质量通过时，才输出非空 `valid_interval`。缺少连续性证据不能算通过。Unitree 保存校验同时检查 summary 保存错误、实际帧数、帧率及连续状态流首尾覆盖。

已观测到的探测点与时钟模型不一致时，即使调整小于旧的 50 ms 阈值也会拒绝模型。网络误差只是观测约束，不能保证发现两次探测之间所有未观测跳变。

当前边界：P450 原脚本的 30 分钟上限仍保留，尚无自动联停看门狗；正式采集请在上限前主动停止。CAN/传感器须预先准备好。Unitree 保留含图像的原始 PKL，尚未自动导出双路 MP4 或裁剪共同区间视频。本机网页控制台见上方使用说明。

## 十分钟只读时钟探测

硬件在线后、正式采集前运行：

```powershell
& $python -m jointctl --config "$app\config\default.json" probe --duration 600 --hosts p450 unitree
```

此命令只建立两条独立 SSH 时钟响应通道，不调用任何 capture start、stop、finish 或 status 命令，也不在远端写文件。输出包括各主机样本数、RTT 最小值/中位数/P95、offset 总漂移、漂移 ppm、拟合残差、异常样本比例、每台主机估计不确定度，以及两台主机相对对齐的不确定度上界。保留兼容的单主机短探测：

```powershell
& $python -m jointctl probe --host p450 --count 5
```

两台主机由独立工作线程采样；一条链路超时不会降低另一条链路的采样节奏。若任一主机最终没有有效样本，命令输出报告后返回连接或远端命令错误码。

## 误差含义与告警

`quality.estimated_error_ns` 是由观测到的网络 RTT、时钟拟合残差、漂移和映射区间传播出的保守估计，目标小于 10 ms。它不是用共同可见物理事件测得的实际端到端误差，也不能证明网络上下行完全对称。无硬件模拟测试会把映射结果与注入的真实时钟比较；实机实际误差仍需在后续授权的短采集里通过两端共同可见事件验证。

报告出现 `quality.degraded: true` 或 CLI 返回码 5 时，应查看 `degradation_reasons`。缺少样本、单锚点无法证明漂移、映射外推、流覆盖缺失或相对误差超过 10 ms 都会显式降级，不能把降级报告解释为已达到 10 ms。

`status` 从两个 JSONL 文件重新计算最新滑动窗口的 offset、低 RTT 误差和样本年龄，不复用启动时冻结的估计。`clock_health` 显示各主机的年龄、监控进程存活情况、最近错误和降级原因。样本过期、监控进程缺失或退出、最近探测错误、窗口最低 RTT 比启动样本增加超过 2 ms，或两端相对误差超过 10 ms，均设置 `timing_degraded` 并返回 5。连接失败、远端 status 错误和所有权冲突优先返回对应错误码。此存活检查只检查记录的 PID；实际采样是否持续由样本新鲜度共同判定。

对齐会保留 JSONL 中的探测错误和无效记录诊断，并在 `mappings.<host>.unsupported_gaps` 标记缺测区间。若请求区间或有效区间跨越不支持的缺测，或者远离采样覆盖边界，该主机和相对对齐的区间误差字段为 `null`，报告降级。仍可查看换算时间，但这些时间不能用于精度承诺。导出映射的跨缺测段 `max_uncertainty_ns` 也为 `null`；直接调用映射 API 时，缺测内部返回 `degraded=True`，以整个缺测长度作为保守占位误差。600 秒仅有两端锚点的记录无法证明毫秒精度。所有原始日志保持原样。

可在 `config/default.json` 调整以下选项（纳秒字段使用整数）：

| 选项 | 默认值 | 用途 |
| --- | --- | --- |
| `clock_interval_s` | `0.2` | 每个监控通道的采样间隔，传给独立监控进程 |
| `ssh_timeout_s` | `10.0` | 状态查询和监控探测等待超时 |
| `start_timeout_s` | `120.0` | 采集启动命令超时，包含传感器初始化 |
| `stop_timeout_s` | `300.0` | 快速停止命令超时，包含录制进程退出和必要落盘，不含导出与完整校验 |
| `inspect_timeout_s` | `300.0` | 离线读取远端采集数据的超时 |
| `alignment_window_ns` | `5000000000` | 5 秒低 RTT 估计窗口 |
| `clock_freshness_ns` | `10000000000` | 状态页可接受的样本年龄与最近错误观察窗口 |
| `max_clock_gap_ns` | `10000000000` | 离线对齐支持的最长相邻采样缺口及边界外推新鲜度 |

两个 10 秒阈值相当于默认 200 ms 节奏下约 50 次采样或两个 5 秒窗口。它们用于发现采样证据中断，并非物理误差保证；调大阈值会接受更长的未观测时钟行为。配置 `p450.host` 或 `unitree.host` 为其他 SSH 别名时，传输使用这些别名，文件名和样本 `host` 仍分别为 `clock_p450.jsonl`/`p450`、`clock_unitree.jsonl`/`unitree`。

退出码约定为：0 成功，2 参数或配置错误，3 连接失败，4 active 状态或 episode 冲突，5 对时质量降级，6 远端命令失败。

停止命令只有两端成功、最终状态 `complete` 且监控未报告关闭失败才返回 0。SSH 退出 255 或传输异常返回 3，其他远端停止失败返回 6；监控关闭超时或剩余 `partial` 状态返回 5，即使两端停止命令本身均成功。

## 远端采集入口

### P450 无外网、日期为 1970 时

无需手动把日期改成今天。每轮采集重新测量两端相对本机的时钟偏差，并持续记录漂移；原始时间戳保留不变。P450 header 必须为正数、与同一消息的 bag 时间相差不超过 1 秒，才用于映射，不再要求年份处于 2000–2100。1 秒仅是排除明显错误时间域的检查阈值，绝不是同步精度保证；其他情况回退到 bag 时间并保留来源标记。同一 topic 的合格 header 倒退会拒绝本次离线检查。

离线映射按本机单调时钟恢复探测顺序，检查远端重启、时间倒退以及 wall/monotonic 的明显不连续。超过 50 ms 加间隔的 2000 ppm 容差时拒绝生成新映射；低于阈值的跳变仍可能无法检出。采集期间不要手工改时。重启或改时后开始新会话，不跨会话复用映射。这里只验证时间域合理性，不证明相机曝光时间或设备内部时间源与系统时钟完全一致。

控制器使用以下入口；快速停止和原始数据收尾是本轮新增的分离步骤：

```text
P450 status: /home/amov/bin/p450_capture status
P450 start:  /home/amov/bin/p450_capture start <episode_id>
P450 stop:   /home/amov/bin/p450_capture stop-fast
P450 finalize: /home/amov/bin/p450_capture finalize-raw <session_dir>
P450 manual video export: /home/amov/bin/p450_capture export <session_dir>
Unitree status: ~/heterovla-collection/onboard/collection_ctl.sh status
Unitree start:  ~/heterovla-collection/onboard/collection_ctl.sh start <episode_id> <instruction> <task>
Unitree stop:   ~/heterovla-collection/onboard/collection_ctl.sh stop
Unitree finalize: ~/heterovla-collection/onboard/collection_ctl.sh finalize <episode_dir>
```

安装、单元测试和模拟测试不会执行这些命令。真实 start、stop、status、probe 和远端数据检查必须等硬件在线，并按操作授权分别执行。
