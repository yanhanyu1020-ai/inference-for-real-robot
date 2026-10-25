# 机器人 IP 与 Windows 静态 IP 配置

这份说明用于本仓库的 P450 / Unitree 部署。换一台 Windows 电脑时，先配置本机网卡，再测试 SSH，最后打开采集控制台。机器人已有地址通常不用修改。

配置网络会中断对应连接。请先结束采集和数据复制，并让机器人保持静止、未解锁或未使能。不要在传输途中拔网线、更改 IP、重启网卡或创建网络桥接。

## 1. 地址速查

| 用途 | 机器人地址 | Windows 对应网卡地址 | 子网掩码 | Windows 专用网卡网关 / DNS |
| --- | --- | --- | --- | --- |
| P450 机载电脑 | `192.168.1.11` | `192.168.1.123` | `255.255.255.0` | 均留空 |
| Unitree 机载电脑 | `192.168.123.18` | `192.168.123.222` | `255.255.255.0` | 均留空 |

前缀长度均为 `24`。这两段地址分别用于机器人专用局域网，上网仍走电脑原来的 Wi-Fi 或其他上网网卡。没有默认网关不会妨碍同网段的电脑与机器人通信。

同一局域网中，每台设备必须使用不同地址。不要把 Windows 设置成机器人的 `.11` 或 `.18`。若另一台地面站已占用 `192.168.1.123`，先安排唯一地址；更换本机地址时，还需同步核对 P450 的 `ground_station_ip`。

P450 网络里另有两个保留地址，不要拿来给 Windows 使用。

| 地址 | 归档配置中的用途 |
| --- | --- |
| `192.168.1.12` | G1 云台 / 相机检测目标，并非 Windows 地面站地址 |
| `192.168.1.100` | MID360 雷达 |

`192.168.1.123` 与仓库内两份 `p450_communication*.launch` 的 `ground_station_ip` 一致。随意换成本机 `.10` 或 `.222` 可能仍能 SSH，却收不到发往原地址的地面站数据。

这些是本套设备的部署地址，不能套用到所有型号的机器人。2026-09-29 已读取 Windows 的 Unitree 网卡配置 `192.168.123.222/24`；Unitree 的 `192.168.123.18/24` 及 `eth0` 在此前实机排查中确认。P450 地址来自当前 SSH 配置与归档部署文件，本次文档更新没有重新修改或实测 P450 网卡。

## 2. 接线与网卡选择

同时连接两台机器人时，使用两个独立的有线网口或 USB 网卡。

```text
Windows 机器人网卡 A  192.168.1.123/24   ── P450   192.168.1.11/24
Windows 机器人网卡 B  192.168.123.222/24 ── Unitree 192.168.123.18/24
Windows Wi-Fi        保留原来的 DHCP 配置 ── 实验室网络 / Internet
```

两台机器人不需要接入同一个网段，也不需要彼此直接通信。本机分别连接两端，采集程序通过保存的时钟映射处理时间对齐。静态 IP 本身不提供时间同步。

先在 PowerShell 中查看网卡和现有地址。

```powershell
Get-NetAdapter | Format-Table Name, InterfaceIndex, InterfaceDescription, Status, LinkSpeed
Get-NetIPAddress -AddressFamily IPv4 |
    Format-Table InterfaceAlias, InterfaceIndex, IPAddress, PrefixLength
```

现场曾使用名为 `以太网 3` 的 Realtek USB 网卡连接 Unitree，但名称和接口编号会随电脑、USB 口及驱动变化。不要直接照抄接口编号。应结合物理接线和网卡描述识别目标网卡；仅在没有采集或传输时，才可通过插拔对应网线观察状态变化。

不要修改 `WLAN`、WSL / Hyper-V、Tailscale、VPN 等无关接口。如果只有一张机器人网卡，来回换线只能分时连接两端，无法按上述接线同时联合采集。

## 3. Windows 图形界面设置

1. 按 `Win + R`，输入 `ncpa.cpl` 并回车，打开网络连接。
2. 找到连接目标机器人的有线网卡，右键进入“属性”。
3. 选择“Internet 协议版本 4（TCP/IPv4）”，点击“属性”。
4. 选择“使用下面的 IP 地址”，按下表填写。

| 填写项 | 连接 P450 的网卡 | 连接 Unitree 的网卡 |
| --- | --- | --- |
| IP 地址 | `192.168.1.123` | `192.168.123.222` |
| 子网掩码 | `255.255.255.0` | `255.255.255.0` |
| 默认网关 | 留空 | 留空 |
| 首选 DNS | 留空 | 留空 |
| 备用 DNS | 留空 | 留空 |

5. 检查“高级”页面，确认这张专用网卡没有残留的旧默认网关或冲突地址。不要删除其他网卡的配置。
6. 保存。另一张机器人网卡按同样步骤设置，上网网卡保持原状。
7. 重新运行上一节的查看命令，确认地址和掩码已经生效。

也可以使用 Windows“设置 → 网络和 Internet → 以太网 → IP 分配 → 编辑 → 手动 → IPv4”。界面若要求子网前缀长度，填写 `24`；若要求子网掩码，填写 `255.255.255.0`。微软的 [TCP/IP 设置说明](https://support.microsoft.com/en-us/windows/experience/connectivity-networking/essential-network-settings-and-tasks-in-windows) 给出了手动 IPv4 的入口。

专用网卡显示“无 Internet”并不表示机器人连接失败。不要为了消除该提示，把机器人 IP 填成网关，也不要启用 Internet 连接共享或桥接两张机器人网卡。

如使用 PowerShell 管理地址，先核对现有配置再操作。`New-NetIPAddress` 会在对应接口启用 DHCP 时自动关闭 DHCP，重复执行也可能与已有地址冲突；本说明不提供清空全部接口的批处理。详见 [微软 New-NetIPAddress 文档](https://learn.microsoft.com/en-us/powershell/module/nettcpip/new-netipaddress)。

## 4. 机器人端只在地址丢失时调整

已经能连接时，保留机器人原配置。可在机器人本机终端查看，下面的命令不会修改网络。

```bash
ip -br -4 address
ip -4 route
nmcli device status
nmcli connection show
```

Unitree 对应有线接口应有 `192.168.123.18/24`，P450 对应接口应有 `192.168.1.11/24`。Unitree 现场接口名为 `eth0`；换机后仍需核对，不能据此推定 P450 也叫 `eth0`。

若确实丢失静态配置，接上显示器和键盘，在 Ubuntu 的有线连接设置中选中对应物理接口，将 IPv4 改为手动并恢复该机器的地址和 `/24` 掩码。机器人直连这张网卡不配置默认路由；已有 Wi-Fi、其他内部接口和路由保持不变。

如果该接口未由 NetworkManager 管理，先确认实际生效的是 Netplan、systemd-networkd 还是厂商配置，再采用对应方法。不要同时创建多份互相覆盖的配置，也不要通过唯一一条远程 SSH 会话试改其 IP。NetworkManager 的接口与连接配置查看方法见 [nmcli 官方说明](https://networkmanager.dev/docs/api/latest/nmcli.html)。

## 5. 配置 SSH 别名

控制台默认使用 `p450` 和 `unitree` 两个 SSH 别名。将以下主机段合并到 Windows 用户的 `%USERPROFILE%\.ssh\config`，不要覆盖已有其他主机配置。

```sshconfig
Host p450
    HostName 192.168.1.11
    User amov
    Port 22
    ConnectTimeout 5
    ServerAliveInterval 5
    ServerAliveCountMax 3

Host unitree
    HostName 192.168.123.18
    User unitree
    Port 22
    ConnectTimeout 5
    ServerAliveInterval 5
    ServerAliveCountMax 3
```

如果使用非默认私钥，在对应主机段添加 `IdentityFile`，填写你自己实际存在的私钥路径；公钥必须已授权到对应机器人。仓库不提供私钥或口令。不要把机器人的主机密钥告警直接忽略，首次连接时核对主机指纹后再接受。

先检查别名解析结果，再测试免交互登录。

```powershell
ssh -G p450 | Select-String '^(hostname|user|port|identityfile|bindaddress|proxycommand) '
ssh -G unitree | Select-String '^(hostname|user|port|identityfile|bindaddress|proxycommand) '
ssh -o BatchMode=yes -o ConnectTimeout=5 p450 "hostname"
ssh -o BatchMode=yes -o ConnectTimeout=5 unitree "hostname"
```

返回机器人主机名才说明免交互登录成功。手动 SSH 每次输密码能进入，仍不代表自动采集程序能够登录。若继承了旧 `ProxyCommand`、`BindAddress` 或通配主机配置，应核对它们是否仍指向有效的网卡和本机脚本；不要把旧电脑的绝对路径直接照搬。

## 6. 开始采集前的连通性检查

在 Windows 执行以下只读检查。

```powershell
ping -n 4 192.168.1.11
ping -n 4 192.168.123.18
Test-NetConnection 192.168.1.11 -Port 22
Test-NetConnection 192.168.123.18 -Port 22
```

SSH 端口检查应显示 `TcpTestSucceeded : True`。需要定位从哪张网卡发包时，可使用对应的本机源地址。

```powershell
ping -n 4 -S 192.168.1.123 192.168.1.11
ping -n 4 -S 192.168.123.222 192.168.123.18
Get-NetRoute -AddressFamily IPv4 |
    Where-Object { $_.DestinationPrefix -in @('192.168.1.0/24', '192.168.123.0/24') } |
    Format-Table DestinationPrefix, InterfaceIndex, NextHop, RouteMetric
```

两条直连网段应分别使用对应的机器人网卡。随后再运行上一节的免交互 SSH 测试，通过后打开控制台。不要只凭一次 ping 成功就认定采集链路已经就绪。

## 7. 常见现象

| 现象 | 优先检查 |
| --- | --- |
| 网卡显示“未连接” | 电源、网线、转接器和实际接线，改 IP 无法恢复物理载波 |
| 专用接口只有 `169.254.*` | 检查静态配置是否设到了正确网卡，以及目标地址是否生效 |
| ping 超时 | 对应网卡地址、掩码、重复 IP、路由及 ICMP 防火墙规则；同时检查 TCP 22 |
| ping 通，SSH 显示拒绝连接 | SSH 服务未启动、正在关机或端口未监听 |
| TCP 22 通，SSH 认证失败 | 用户名、密钥授权、主机指纹及实际 SSH 配置 |
| 小命令成功，大文件传输频繁超时 | 检查持续丢包、USB 网卡 / 供电、网线、设备负载和内核日志；不据此反复更换静态 IP |
| P450 能 SSH，Prometheus 没有状态 | 核对本机 `192.168.1.123` 与 `ground_station_ip`，再检查桥接进程及应用防火墙规则 |
| RTSP 拉流失败 | 检查 `rtsp://192.168.1.11:8554/live` 对应服务是否启动、端口是否监听；IP 正确不保证有视频流 |

防火墙排查应针对所需程序、端口和机器人网段，不建议关闭整个 Windows 防火墙。

## 配置依据

- [P450 地面站地址，MID360 / D435i 版本](../robots/p450/home/amov/p450_experiment/src/p450_experiment/launch_basic/p450_communication_mid360_d435i.launch)
- [P450 地面站地址，通用版本](../robots/p450/home/amov/p450_experiment/src/p450_experiment/launch_basic/p450_communication.launch)
- [MID360 的机载电脑地址与雷达地址](../robots/p450/home/amov/p450_experiment/src/p450_experiment/config/mid360_config/MID360_config.json)
- [G1 云台 / 相机地址](../robots/p450/home/amov/p450_experiment/src/p450_experiment/launch_spirecv/gimbal_server.launch)

本次仅补充文档，没有更改任何实机 IP、路由、SSH 配置或采集程序。
