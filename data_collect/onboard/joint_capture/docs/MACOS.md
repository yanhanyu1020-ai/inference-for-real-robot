# macOS 控制台

Mac 运行网页控制台，通过系统 SSH 调用两台机器人原有的采集入口。机器人端的准备、开始、停止、原始数据格式和对时逻辑保持原样。

## 打开控制台

电脑需要 Python 3.10 或以上版本及 `ssh` 命令。在 Finder 中双击仓库根目录的 **JointConsole-macOS.command**。首次运行会在 `controller/.venv-macos` 准备独立 Python 环境，生成本机路径配置，然后打开 `http://127.0.0.1:8766/`。无需 Node.js、pip 或下载 Python 包。

也可以在终端从仓库根目录执行：

```bash
./JointConsole-macOS.command
```

入口会查找 Apple Silicon 和 Intel Homebrew 的 Python 路径。若要指定已有 Python：

```bash
JOINTCAPTURE_PYTHON=/实际路径/python3 \
  ./controller/scripts/Install-JointCapture.command
./JointConsole-macOS.command
```

环境准备只写本机文件，不连接机器人。控制台打开后会查询设备状态；录制由网页上的操作按钮触发。设备未连接时也可以打开界面。

本机文件位置：

| 内容 | 默认位置 |
| --- | --- |
| 独立 Python 环境 | `controller/.venv-macos/` |
| 可编辑设备配置 | `controller/config/local.macos.json` |
| 控制台路径配置 | `controller/macos_paths.json` |
| 会话记录、对时报告和任务日志 | `~/Library/Application Support/JointCapture/episodes/` |

重复安装会保留本机设备配置与历史记录。这些本机文件已加入 Git 忽略规则。机器人原始数据仍保存到机器人上的原目录。

如需将本机记录放到其他磁盘，可以先执行：

```bash
./controller/scripts/Install-JointCapture.command \
  --manifest-root '/实际路径/JointCapture/episodes'
```

也可以用 `--config '/实际路径/custom.json'` 指定设备配置。路径会保存为绝对路径，双击启动不受当前目录影响。调整路径前先结束当前采集并退出旧控制台后台；新目录不会自动迁移旧记录。移动仓库后重新执行安装入口，并核对 `macos_paths.json` 中的路径。

## 连接机器人

Mac 的机器人专用网口沿用现有部署地址：

| 目标 | Mac 网口 IPv4 | 子网掩码 | 机器人 IPv4 |
| --- | --- | --- | --- |
| P450 | `192.168.1.123` | `255.255.255.0` | `192.168.1.11` |
| Unitree | `192.168.123.222` | `255.255.255.0` | `192.168.123.18` |

在 macOS 网络设置中选择对应的有线网卡，配置手动 IPv4；机器人专用网口不填写路由器或 DNS。先确认地址未被其他地面站占用。联合采集需要同时访问两个网段，通常使用两个有线网口；上网网卡保留原配置。设备地址与 P450 地面站地址依据见 [网络说明](NETWORK.md)。安装入口不会修改网络。

将下面的主机段合并到自己的 `~/.ssh/config`，保留其他已有主机配置：

```sshconfig
Host p450
    HostName 192.168.1.11
    User amov
    ServerAliveInterval 5
    ServerAliveCountMax 3

Host unitree
    HostName 192.168.123.18
    User unitree
    ServerAliveInterval 5
    ServerAliveCountMax 3
```

使用自己的登录密钥；若私钥位于非默认位置，在对应段添加 `IdentityFile`。首次手动连接时核对机器人主机指纹。随后确认控制台可以免交互登录：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 unitree hostname
ssh -o BatchMode=yes -o ConnectTimeout=5 p450 hostname
```

使用其他 SSH 别名时，修改 `local.macos.json` 的 `unitree.host` 或 `p450.host`。修改后退出并重新启动控制台后台。网页关闭不会退出后台，也不会停止远端采集。后台 PID 可从 `http://127.0.0.1:8766/api/health` 查看；确认 `root` 与 `config` 对应这套控制台后，可用 `kill <pid>` 退出该后台。后台日志位于记录目录中的 `.console/server.log`。

## 日常采集与 Unitree 单独试采

原联合网页流程保持不变：两台设备在线且空闲后初始化、开始采集、停止并保存，再校验待处理数据。浏览器只控制和显示状态，Mac 不需要安装 ROS、CUDA、相机 SDK 或 CAN 驱动。详细行为见 [控制器说明](../controller/README.md)。

暂时不连接 P450 时，使用 Unitree 既有独立采集入口。以下命令均在 Mac 终端执行：

```bash
# 只读状态查询
ssh unitree '~/heterovla-collection/onboard/collection_ctl.sh status'

# 准备设备，然后在操作现场确认相机、手柄和机械臂状态
ssh unitree '~/heterovla-collection/onboard/collection_ctl.sh prepare'

# 用新的 ID 开始一条独立试采
ssh unitree '~/heterovla-collection/onboard/collection_ctl.sh start mac_unitree_test_001 "Unitree test" "mac_test"'

# 停止录制并保存
ssh unitree '~/heterovla-collection/onboard/collection_ctl.sh stop'
```

每次试采使用新的 ID。手柄配对、Home 使能等操作沿用机器人现有流程。独立试采不会生成联合控制器的双机 manifest 或 P450–Unitree 对齐报告；联合网页上的“开始采集”仍要求两端就绪。不要同时从独立入口和联合控制台操作同一条录制。
