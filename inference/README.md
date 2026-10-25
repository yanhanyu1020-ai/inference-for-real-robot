# Inference：策略服务与机器人执行

本目录保存推理、控制协议、服务器操作工具和对应的机载执行器。所有示例均从主仓库根目录运行。

V3 的 Go2 + Piper + P450 三机实时观测、获取动作和执行后返回图像入口见 [real_robot/README.md](real_robot/README.md)。

2026-10-24 联调总结、16.7 与真机网络框架图，以及下次调用的完整步骤见 [16.7 真机推理流程](real_robot/README_16_7_WORKFLOW.md)。

14 维状态、块起点相对关节角和速度的新训练/推理入口见 [relative14](../train/V3/relative14/README.md)。

```text
inference/
├── heterovla/                  # Python 协议定义与 control broker
├── scripts/                    # GPU 服务器服务、部署和操作入口
├── onboard/
│   ├── unitree_go2/control/    # Go2/Piper agent、执行器、遥操作与日志
│   ├── robot_gateway/         # Piper chunk / policy stream 执行器
│   └── scripts/               # Go2 启动、拉取更新、policy client 检查
├── tests/                      # 纯 CPU 协议、轨迹、遥操作映射和日志测试
└── requirements-control.txt   # control broker 的已有依赖声明
```

## 运行位置

GPU 服务器运行 `heterovla.control_server`、mock / OpenPI 策略服务以及 `scripts/` 中的操作工具。Go2 Jetson 运行 `onboard/` 下的代码；实际部署位置仍为 `~/heterovla-control/`，录制输出位置也保持原配置。Piper SDK、Unitree SDK 与 CAN/DDS 运行环境沿用已有部署。

主仓库根目录启动 broker 的入口为：

```bash
bash inference/scripts/control_server_ctl.sh start
```

直接使用 Python 模块时，将 `inference/` 加到 `PYTHONPATH`：

```bash
PYTHONPATH="$PWD/inference${PYTHONPATH:+:$PYTHONPATH}" python3 -m heterovla.control_server --help
python3 inference/scripts/mock_vla_server.py --help
```

服务器侧操作入口为 `inference/scripts/robotctl.sh`、`send_go2_trajectory.py` 和 `send_hetero_demo.py`。采集操作入口 `record_teach.py`、`record_teleop.py`、`prepare_collection_replay.py`、`sport_to_go2_replay.py` 及 recorder 部署脚本位于 `data_collect/scripts/`。

机载侧使用 `inference/onboard/scripts/control_up.sh` 和 `pull_control_on_go2.sh`。`control_up.sh` 既支持仓库布局，也支持复制到 `~/heterovla-control/` 后与 agent 平铺运行。控制代码部署入口为 `inference/scripts/deploy_unitree_control.sh`。这些命令会连接机器人或启动控制链，应在实际执行机器人操作时使用。

## 兼容与依赖位置

根目录保留 `heterovla`、逐文件的 `scripts/`、`onboard/unitree_go2/control` 和 `robot_gateway` 兼容链接，原有命令仍可使用。代码通过物理文件路径定位规范目录，兼容入口与新入口指向同一份实现。

`inference/scripts/` 中 `control_up.sh`、`pull_control_on_go2.sh`、`test_policy_client.py` 是指向 `onboard/scripts/` 的相对兼容链接；机载代码实体集中在 `onboard/`。共享的 `third_party/openpi` 与 `.venvs` 留在主仓库根目录。此调整不安装依赖。

## 验证

在主仓库根目录运行已有 CPU 测试：

```bash
python3 -m unittest discover -s inference/tests -p 'test_*.py'
```

测试不会启动 broker、连接机器人、发出控制命令或运行训练。`piper_replay.json` 是可选的本地大 payload 样本，仍从主仓库根目录读取；不存在时跳过该项测试。

完整控制通道与操作步骤见主仓库 `docs/control_channel_zh.md` 和 `docs/trajectory_control_zh.md`。
