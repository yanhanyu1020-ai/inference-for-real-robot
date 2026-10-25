# V1 训练策略

这里保存既有 π0.5 V1 训练、评估及实验脚本。`action_lora`、`dual_lora`、`expert_full`、`full`、batch profile、LR scan 与其历史修复均属于 V1；这次整理不启动训练。

## 目录

- `scripts/`：原 `experiments/pi05_v1/scripts/` 的全部脚本，以及严格窗口 `dataset_v1.py` 和原始接入入口 `train_openpi_v1.py`。
- `configs/`：原四个配置摘要，路径已更新；历史参数和实验结果保持不变，它们是参数参考快照，真实运行配置由 `experiment_configs.make_config()` 与环境变量生成。`preflight.py` 将审计记录写入 `RUN_ROOT/configs/`，不会改写这里的源码副本。
- `env.sh`：复用 `~/heterovla/.venvs/openpi-pi05`，设置路径和离线数据开关；已有环境变量优先。
- `openpi_memory_patch.patch`、`MEMORY_OPTIMIZATION.txt`、`PREPARATION.txt`：既有 openpi patch 和准备记录；不会自动重新应用 patch。

| 内容 | 默认路径 |
| --- | --- |
| openpi | `~/heterovla/third_party/openpi` |
| 对齐后的 V1 数据 | `~/heterovla-data/aligned/V1` |
| 预训练参数 | `~/heterovla-model/pretrained/pi05_base/params` |
| V1 运行、报告、缓存 | `~/heterovla-model/V1/runs/pi05_v1` |
| checkpoint | `~/heterovla-model/V1/runs/pi05_v1/checkpoints` |

`HETEROVLA_ROOT`、`OPENPI_ROOT`、`HETEROVLA_DATA_ROOT`、`PI05_PARAMS`、`HETEROVLA_RUN_ROOT` 可覆盖这些默认值。源码与输出已分开，supervisor 从本目录调用脚本，不再依赖运行输出下存在 `scripts/`。数据目录原 `scripts/` 保留为既有数据发布的 provenance，不再作为维护入口。

## 检查配置

在已配置依赖的服务器上执行：

```bash
source ~/heterovla/train/V1/env.sh
python "$HETEROVLA_ROOT/train/V1/scripts/launch_experiment.py" --variant dual_lora
python "$HETEROVLA_ROOT/train/V1/scripts/train_openpi_v1.py" --config-only
```

上述两条只注册或打印配置。完整 CPU 数据、tokenizer 与权重元数据检查使用 `scripts/preflight.py`，会在运行目录写报告；不会启动训练。实际训练入口 `launch_experiment.py` 需要显式传 `--execute`、指定 `--gpu` 与唯一 `--exp-name`。历史 `train_*`、`*_supervisor.py` 是可执行实验脚本，包含当时固定的 GPU UUID、run name 和恢复条件，不应在未调整任务参数时直接重跑。

数据转换维护入口为仓库根目录下的 `data_collect/alignment/V1/`。V1 的定义、划分、标签语义和现有数据不因目录整理而变化；新采集数据仍保存在 raw，未经单独转换不会并入 V1。

预训练权重下载工具保存于 `../tools/download_pi05_base.py`，默认写入 `~/heterovla-model/pretrained/`；可用 `HETEROVLA_MODEL_ROOT` 覆盖 model 根目录。脚本只在显式运行时下载，导入同样会执行下载逻辑，因此检查源码时不要 import；本次整理没有运行它。
