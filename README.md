# HeteroVLA

异构机器人联合采集、VLA 训练与推理代码主仓库。代码、数据和模型产物分别保存在三个相邻目录中；本仓库是当前现场代码快照，上传至 [yanhanyu1020-ai/inference-for-real-robot](https://github.com/yanhanyu1020-ai/inference-for-real-robot)，源项目为 [panhe-hu/heterovla](https://github.com/panhe-hu/heterovla)。

真机部署入口：[Windows 现场客户端](inference/real_robot/README_FIELD_WINDOWS.md)、[机器人网关与执行](inference/real_robot/README.md)。当前配置支持三路图像推理，现场客户端选择 Go2＋Piper 执行，P450 保持只观测。

迁移到 `yhy` 后的环境激活、检查和 V3 训练入口见 [本机运行说明](docs/local_setup_zh.md)。

```text
~/
├── heterovla/                         # Git 主仓库
│   ├── data_collect/                  # 采集、传输、数据转换和对齐
│   │   ├── server/                    # 服务器数据检查与转换
│   │   ├── alignment/V1/              # V1 对齐、导出与验证代码
│   │   ├── scripts/                   # 采集操作、传输和部署入口
│   │   ├── config/                    # 采集配置
│   │   └── onboard/                   # 不在 GPU 服务器运行的代码
│   │       ├── unitree_go2/            # 既有采集与 SDK recorder
│   │       └── joint_capture/          # 最新联合采集：Mac/Windows、Go2、P450
│   ├── train/
│   │   ├── V1/                        # V1 数据定义下的 LoRA / full / LR scan
│   │   │   ├── scripts/               # 训练、配置、评估与 supervisor
│   │   │   ├── configs/               # 参数参考，实际配置由代码生成
│   │   │   └── env.sh                 # 复用现有训练环境，配置外部路径
│   │   ├── V2/                        # 历史训练实验
│   │   ├── V3/dataset/                # 当前V3数据loader与模型配置
│   │   └── tools/                     # 预训练权重准备等共享工具
│   ├── inference/
│   │   ├── heterovla/                 # Python 控制协议与 broker
│   │   ├── scripts/                   # 服务器操作与部署入口
│   │   ├── onboard/                   # Go2、Piper 等设备执行器与启动入口
│   │   └── tests/                     # 现有 CPU 测试
│   ├── third_party/openpi/            # 现有 Git submodule，位置保持稳定
│   ├── docs/                          # 控制文档与历史说明
│   └── .venvs/                        # 本机环境，Git 忽略
├── heterovla-data/
│   ├── raw/
│   │   ├── benchmark/                 # 原始 benchmark：旧 6 条 + 新 26 条
│   │   └── legacy_replays/            # 历史采集、遥操作与回放数据
│   └── aligned/
│       ├── V1/                        # 既有对齐表、LeRobot、统计与 provenance
│       └── V3/                        # 当前35条训练、3条前值补齐验证数据
└── heterovla-model/
    ├── pretrained/pi05_base/           # 官方基座权重
    ├── V1/runs/pi05_v1/                # 既有 checkpoint、扫描结果、日志和报告
    │   ├── checkpoints/               # 常规训练恢复点
    │   ├── lr_scan_20261005/           # LoRA 扫描及 best checkpoint
    │   └── full_lr_scan_20261007/      # full 扫描及 best checkpoint
    └── V2/                            # 后续策略模型产物
```

## 约定

- `data_collect`、`train`、`inference` 是三个维护入口；`onboard` 是子目录，不是 Git branch。目录拼写使用 `inference`。
- V1、V2 表示数据定义、动作语义和训练方案的版本。同一版本内可以有多个 LoRA、full fine-tuning、学习率与 seed 实验，用独立 run name 区分。
- Mac/Windows 控制端、Go2/Jetson、P450/ROS 等非服务器代码放在对应部分的 `onboard/`。SDK、ROS、驱动、cuRobo 与系统镜像由设备环境管理。
- 原始数据写入 `heterovla-data/raw/`；可训练的对齐、转换结果写入 `heterovla-data/aligned/<版本>/`，并保存 schema、划分、统计与 provenance。
- checkpoint 与运行产物写入 `heterovla-model/<版本>/`。官方预训练权重单独放在 `pretrained/`，避免与本项目训练结果混淆。
- 源代码进入 Git；数据、权重、虚拟环境、缓存、W&B 运行记录、日志和认证文件由 `.gitignore` 排除。配置只提交可共享的参数及示例，不提交 token、密码、私钥或 `.env`。

## 运行入口

各部分的设备职责、依赖与命令见：

- [采集与对齐](data_collect/README.md)
- [V1 训练](train/V1/README.md)
- [V3 数据与模型接口](train/V3/dataset/README.md)
- [推理与机器人执行](inference/README.md)

在服务器上检查 V1 配置：

```bash
cd ~/heterovla
source train/V1/env.sh
python train/V1/scripts/launch_experiment.py --variant dual_lora
python train/V1/scripts/train_openpi_v1.py --config-only
```

这些命令只检查配置。实际训练需要使用对应脚本的显式执行参数和独立的实验名称。数据转换与训练是独立步骤；新上传的 26 条数据仍属于 raw，尚未并入 V1，V2 也尚未生成。

## 旧入口兼容

整理时保留了旧脚本、Python 包和数据路径的兼容链接。根目录的 `scripts/`、`onboard/`、`heterovla`、`robot_gateway`、`heterovla-collection` 与 `experiments` 不作为新的维护入口；脚本实体分别位于上面的三个部分，`experiments` 指向外部 V1 运行目录。

`heterovla-data/benchmark`、`heterovla-data/V1`及拼错的`hererovla-model`保留兼容链接。当前数据入口为`heterovla-data/V3`；旧V2数据和变体已清理。已有checkpoint与训练实验保留，其历史数据路径不再作为当前入口。对齐数据里的旧 `scripts/` 是当时发布内容的 provenance，保留原字节与原校验清单；维护代码位于主仓库。

历史根目录 README 保存在 `docs/legacy/`。迁移备份与操作报告保存在本机 `.layout_backups/`，该目录不进入 Git。整理不会自动 commit、push、启动训练或向机器人发送命令。
