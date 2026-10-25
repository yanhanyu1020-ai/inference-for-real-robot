> 2026-10-19数据整理：旧V2数据集及其变体已清理，当前数据入口为`heterovla-data/V3`，当前接口见`heterovla/train/V3/dataset/README.md`。本目录原有内容是历史实验或转换过程，保留用于追溯；旧训练实验不能自动按新数据恢复。

# V3：冻结 ViT 的 LoRA 续训

V3 是独立训练策略。V2 的全量训练入口、运行记录和checkpoint保持原样；V3复用只读的 `heterovla-data/aligned/V2` 数据及其训练归一化统计。

| 内容 | 目录 |
|---|---|
| V3训练代码 | `heterovla/train/V3/scripts/` |
| V3交接程序 | `heterovla/train/V3/handoff_lora_v3.py` |
| V3运行记录 | `heterovla-model/V3/runs/<name>/` |
| V3恢复checkpoint | `heterovla-model/V3/checkpoints/pi05_heterovla_v3_lora_frozen_vit/<name>/` |
| V3最佳checkpoint | `heterovla-model/V3/runs/<name>/best/<global_step>/` |
| 父模型 | V2低LR运行 `full_b32_lr1e-5_s42_from600_20261016_170448` 的best1000 |

## 训练范围与超参

- 模型为 `gemma_2b_lora`（r16/alpha16）和 `gemma_300m_lora`（r32/alpha32）。
- ViT与Gemma非LoRA基座全部冻结；两组LoRA和 `action_in/out_proj`、`time_mlp_in/out` 继续训练，预期52,153,376个可训练参数。
- 冻结与可训练参数均保留FP32，计算使用bf16。独立initializer去掉官方冻结参数转bf16的操作，并严格加载父模型每个dense叶。
- 官方LoRA默认A/B均随机；V3保留A随机，显式清零所有B，确保起点没有额外LoRA增量。
- batch32、seed42、新Adam phase0；warmup100，peak LR1e-5，cosine到1e-6。global从1000继续到20000，实际新增19000次更新，sampler以seed42重新开始。
- 每200个global steps完整验证3532窗口及固定128训练probe；每1000保存恢复点，保留5000里程碑和最近恢复点；best按real14 loss。无早停，W&B离线。

## 交接与验收

交接程序只针对明确指定的旧V2运行。CPU阶段完成源参数SHA、V2数据/归一化、真实抽象参数树、严格冻结范围及缓存验证后，才允许停止身份匹配的旧PID。新进程持有原GPU的UUID合作锁，并通过JAX预分配90%显存后在池内编译、加载模型；不更改全局compute mode、不停止其他进程。

GPU接管后，先验证dense权重逐叶精确加载、所有LoRA B为零、官方loss一致，以及完整验证各loss组匹配父best1000。首个更新后及完成时校验冻结参数摘要不变。初始检查失败不会进行梯度更新；日志和源checkpoint均保留供排查。

V3 checkpoint包含完整冻结基座、LoRA、投影头和恢复状态。推理必须按V3 LoRA模型配置加载，不能用dense模型配置静默丢弃LoRA叶。它不是仅包含adapter的导出文件。

用户授权的首次交接由本聊天执行；后续恢复必须使用同一个V3入口、原始运行合同和 `--resume`，不能借用V2的 `launch_full.sh`。
