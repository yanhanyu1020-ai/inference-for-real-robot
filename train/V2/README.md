> 2026-10-19数据整理：旧V2数据集及其变体已清理，当前数据入口为`heterovla-data/V3`，当前接口见`heterovla/train/V3/dataset/README.md`。本目录原有内容是历史实验或转换过程，保留用于追溯；旧训练实验不能自动按新数据恢复。

# V2 数据 loader

`scripts/dataset_v2.py` 为 `heterovla-data/aligned/V2` 提供严格 H=10 的 LeRobot v2.1 loader。数据名为 `local/heterovla_v2_train`、`local/heterovla_v2_val`，归一化 asset ID 为 `heterovla_v2`。三路图像、30 维 state、14 维动作与 V1 保持相同含义。

```python
from dataset_v2 import JointV2Dataset, OpenpiV2Dataset
dataset = JointV2Dataset("~/heterovla-data/aligned/V2", split="train", horizon=10)
```

loader 排除 episode 尾部不足十帧的起点，拒绝 padded/invalid 动作与窗口中的源时间缺口。跨设备观测因果性、V1 数值与图像一致性、会话划分和归一化统计由 `data_collect/alignment/V2/verify_v2.py` 检查。

本目录包含数据接入和下述π0.5全量微调入口，使用 `OpenpiV2Dataset`、`heterovla_v2` assets 和独立的 V2 model 输出目录。CPU准备阶段不会生成训练 checkpoint。


# π0.5 全量微调：从基座重新训练20k

在16.7拿到GPU后执行一条命令（`0`是nvidia-smi中的物理卡号，可替换为分配到的GPU UUID）：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V2/launch_full.sh --gpu 0
```

默认后台运行，返回实验名、PID、日志与状态路径。SSH断开后继续运行。若在调度器job里运行，添加`--foreground`使训练进程留在job前台。

## 固定配置

- 初始化：`~/heterovla-model/pretrained/pi05_base/params`，新Adam状态、step=0；新实验不加载V1或旧best权重。
- 模型：pi05，Gemma 2B+300M action expert；视觉、语言、action expert全部训练；无LoRA。
- 数据：`~/heterovla-data/aligned/V2`，严格H=10；train9344窗、val3532窗；只用V2 train归一化统计。
- batch32、peak LR5e-5、warmup100、cosine衰减周期20000、末值5e-6、seed42。
- AdamW β=(0.9,0.95)、eps1e-8、weight_decay1e-10、global grad clip1；参数float32、计算bfloat16；无EMA。
- activation remat=`nothing_saveable`，单GPU，allocator0.9；不自动降低batch或学习率。
- 总步数20000；无早停。每25步记录metrics，每200步用完整固定val和128窗train probe评估。
- 每1000步保存可恢复checkpoint；保留最近恢复点及5000步里程碑；best单独保留当前最佳（以val real14_loss选择）。20000步保存最终checkpoint。
- W&B离线记录，无需网络或登录。没有自动上传数据或模型。
- 沿用原官方32维flow matching训练目标（14真实动作维，其余padding）；逐机器人/real14验证指标另记。

上次成功依据是`full_b32_lr5e-5_s42_20261007_r2`：同batch下5e-5的best val0.08902，优于1e-4的0.11228。见`configs/previous_best.json`。

## 输出与恢复

- 运行状态/指标/best：`~/heterovla-model/V2/runs/<name>/`。
- 常规可恢复checkpoint：`~/heterovla-model/V2/checkpoints/pi05_heterovla_v2_full/<name>/`。
- 后台日志与启动记录：`~/heterovla-model/V2/runs/_launch_logs/`。
- 评估缓存：`~/heterovla-model/V2/runs/_eval_cache/<fingerprint>/`，完整val3532窗、train probe128窗，不使用随机图像增强。

```bash
# 中断后，指定原实验名恢复该V2自己的参数/Adam状态/step：
bash /meta_eon_cfs/home/hph/heterovla/train/V2/launch_full.sh --gpu 0 --resume --name <原实验名>
```

恢复时核对数据、归一化、模型与超参契约；按seed和step重放loader以保持数据位置，重放可能需要时间。新启动默认生成新实验名，拒绝覆盖已有目录。

## CPU准备命令

以下两条不会查询或使用GPU、实例化大模型或计算梯度：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V2/launch_full.sh --preflight
bash /meta_eon_cfs/home/hph/heterovla/train/V2/launch_full.sh --prepare-cache
```

`--prepare-cache`允许中断后继续，校验缓存SHA和变换/数据指纹；报告在`~/heterovla-model/V2/runs/_preflight/preflight.json`。

启动器先检查选定GPU至少60GiB可用（历史batch32实测加载权重前需要约58.55GiB），训练入口还会按真实batch32编译估算显存并在分配权重前再检查。显存不足直接退出；不会自动抢占其他进程、轮询等卡或修改batch。

默认路径使用V2专属覆盖变量，避免继承V1环境：`HETEROVLA_V2_DATA_ROOT`、`HETEROVLA_V2_RUN_ROOT`、`HETEROVLA_PI05_BASE_PARAMS`、`HETEROVLA_V2_OPENPI_CACHE`、`HETEROVLA_V2_PYTHON`。正常使用无需设置。
