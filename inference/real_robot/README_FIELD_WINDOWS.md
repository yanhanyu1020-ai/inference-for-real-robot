# 现场 Windows 控制 Go2/Piper，服务器 GPU 0 推理

控制程序在与两台机器人有线相连的 Windows 电脑运行。服务器只运行已有 `127.0.0.1:8000` 模型服务，继续使用 GPU 0。无人机只提供真实图像和状态，不接收执行或停止请求。

```text
Go2/Piper <--现场有线 SSH--> Windows 控制器 <--远程 SSH--> 服务器 GPU 0 模型
P450      --现场有线 SSH--> Windows 控制器
```

Windows 上的两个网关转发只走现场局域网，时钟校准不经过 GPU 服务器。图像上传、远程模型推理和结果下载仍经过远程网络；这部分延迟没有消失，仍需满足原有 0.75 秒动作新鲜度检查。此次迁移不放宽 40 ms 校时、三图时序、速度、机械臂限位、观测凭据或过期动作检查。

## 1. 下载并安装轻量客户端

服务器已生成 `.cache/real_robot/field-client.zip`，其中没有 token、SSH 私钥、模型权重或 GPU 依赖。先保持原来的 Windows 到服务器反向 SSH 隧道运行，便于服务器继续管理机器人。

现场客户端可使用 Python 3.11 或 3.12，服务器的 JAX 环境与这里独立。先在 Windows PowerShell 运行以下命令查看已有解释器（`16.7` 是已有的服务器 SSH 别名）：

```powershell
py -0p
```

优先复用已有的 64 位 Python 3.11/3.12。下方以已经找到的 3.12 为例；若列表只有 3.11，把 `py -3.12` 换成 `py -3.11`。某版本提示 `No suitable Python runtime found` 时，不应继续猜测它的安装路径，应使用 `py -0p` 实际列出的兼容版本。两者都没有时，再从 [Python 官方下载页](https://www.python.org/downloads/windows/) 安装兼容版本，并新开一个操作客户端的 PowerShell，保留已有 SSH 隧道窗口。虚拟环境创建成功后，`.venv\Scripts\python.exe` 才会存在。

```powershell
scp 16.7:/meta_eon_cfs/home/yhy/heterovla/.cache/real_robot/field-client.zip .
Expand-Archive .\field-client.zip -DestinationPath .
cd .\field-client
scp 16.7:/meta_eon_cfs/home/yhy/heterovla/.cache/real_robot/token .\token
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

每次运行客户端前进入这个 `field-client` 目录。直接使用 `.venv` 的 Python，无需 PowerShell 激活脚本。客户端只依赖 NumPy、OpenCV、msgpack 和 websockets，不安装 JAX、CUDA 或完整训练环境。

## 2. 在 Windows 开三个独立 PowerShell 窗口，保持运行

窗口 A：仅将本机 48080 转发到服务器的模型端口 8000。

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:48080:127.0.0.1:8000 16.7
```

窗口 B：直接通过现场有线网络访问 Go2 网关，**这里不要用服务器上的 22271 转发**。

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:18701:127.0.0.1:8701 unitree@192.168.123.18
```

窗口 C：直接通过现场有线网络访问 P450 的只观测网关。

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:18702:127.0.0.1:8702 amov@192.168.1.11
```

按现场 SSH 配置登录机器人；这些命令可能提示输入机器人 SSH 密码。首次连接应核对主机身份。`-NT` 登录后没有命令提示符是正常现象。模型服务和网关均仍仅监听各自回环地址。

客户端 `controller.json` 已配置为：

```json
{
  "unitree": "http://127.0.0.1:18701",
  "p450": "http://127.0.0.1:18702",
  "policy": "ws://127.0.0.1:48080",
  "execute_steps": 3,
  "prompt": "Move to the table and pick up the green block inside the cardboard box."
}
```

`prompt` 应与现场任务一致。保留三路真实图像和真实状态，无人机保持未解锁也可作为观测输入。

## 3. 在第四个 Windows 窗口先检查、只推理

此时两端网关均保持只观测模式。进入下载的 `field-client` 目录：

```powershell
.\.venv\Scripts\python.exe -m real_robot.field_client check
.\.venv\Scripts\python.exe -m real_robot.field_client infer --cycles 2
```

`check` 只读测量两端 HTTP 网关时钟误差、检查无人机执行开关关闭，并连接远端模型确认合同；不重置观测、不调用执行器。`passed` 只表示连通和校时通过，`unitree_sensor_error` 单独保留真实问题。**ping 1 ms 不替代这里的 HTTP 实测结果。**

`infer` 会重置两端观测原点，读取真实三图和状态，再请求远端模型；不执行动作。它要求两端都是只观测网关。输出 `inference_s` 在 Windows 上包含图像上传、模型推理和下载的总时间，不能再直接套用服务器本地约 0.11 秒的数字。首次请求可能包含编译；再观察热推理耗时。

网络和动作新鲜度需要分别通过。除了这个推理往返，还要计入观测采样年龄、本地 prepare/commit 和 0.2 秒启动预留；超过原有 0.75 秒上限时，机载网关会拒绝动作，客户端不会重放旧动作。

严格观测和完整推理的非执行检查：

```powershell
.\.venv\Scripts\python.exe -m real_robot.field_client dry-run --cycles 1
```

`dry-run` 保留 Go2 严格观测检查，使用 P450 的只观测诊断数据。它不执行 prepare/commit，所以不能代替真机执行验收。

现场操作者已确认本机 `SportModeState.error_code=100` 是正常返回值，板载 `unitree.json` 使用 `go2_normal_state_codes: [0, 100]` 兼容该设备。未配置此项的设备仍只接受 `0`；其他值仍阻止严格观测和执行。这是本机配置，不是所有固件的通用状态码定义。站姿、数据时效、机械臂状态和动作限幅检查继续生效。

## 4. 现场条件和检查通过后，仅开启 Unitree 执行

Go2 由现场操作者准备为站立，Piper 已使能、处于 CAN 控制且反馈正常，任务和动作空间已核对。程序不自动站立或使能。

在服务器仓库目录执行，始终指定 `--kind unitree`：

```bash
source env.sh
python -m real_robot.manage stop --kind unitree
python -m real_robot.manage start --kind unitree --execute
```

不要开启 P450 的执行模式。模型服务继续使用原 GPU 0，无需重新启动。

然后在现场 Windows `field-client` 目录执行一轮：

```powershell
.\.venv\Scripts\python.exe -m real_robot.field_client execute --cycles 1
```

该入口固定执行对象为 Go2 + Piper，只下发模型前 10 维；P450 的后 4 维只记录。默认只执行前 3 步（0.2 秒），机载线程按 15 Hz 调度动作，之后停止本轮运动。下一轮取新观测再推理，不在远程请求等待期间继续播放旧动作。诊断期间未启动任何动作执行器，本文命令需在实际现场验证。

## 5. 停止和恢复只观测

正在运行的现场客户端可按 Ctrl+C；它只向 Unitree 发送停止请求。也可在另一个 Windows 终端从相同目录执行：

```powershell
.\.venv\Scripts\python.exe -m real_robot.field_client stop
```

单独 `stop` 不连接 P450 或模型服务。网关本身有控制线程超时和故障锁存；它不替代现场硬件急停。停止/运行结束后，再次执行前需检查并重启 Unitree 网关。要恢复只观测，在服务器运行：

```bash
python -m real_robot.manage stop --kind unitree
python -m real_robot.manage start --kind unitree
```

结果保存在 Windows 客户端的 `outputs` 目录，包含三路图像、预测/实际动作、执行回执和无人机只观测状态。

## 在服务器重新打包

更新客户端源码后运行，现场再下载新版：

```bash
python inference/scripts/package_field_client.py
```

包内 `MANIFEST.json` 保存文件 SHA256。OpenPI 的 NumPy/msgpack 编解码器和许可证随包提供，客户端不依赖服务器仓库路径。
