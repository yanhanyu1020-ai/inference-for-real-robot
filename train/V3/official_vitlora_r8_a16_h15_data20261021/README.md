# V3 新数据从头训练

远端代码：`/meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_data20261021`

运行名：`vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_data20261021`

在 16.7 上运行，自动选择当前空闲的 80GB 卡：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_data20261021/train.sh
```

指定物理卡号（示例为 0）：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_data20261021/train.sh --gpu 0
```

加 `--check` 仅检查，不启动。无空卡返回 `waiting`；指定忙卡不会回退到别的卡。每个实验仅允许一次实际启动，重复执行返回现有状态；不会自动恢复、重试或覆盖。

从官方 `pi05_base/params` 重新加载初始权重，Adam 和训练步数从 0 开始。沿用上一版官方 OpenPI 训练逻辑与配置：ViT LoRA rank8/alpha16，冻结 ViT dense，其他层全量；batch32、seed42、warmup500、peak LR 1e-5、cosine 总计 20000 步（含 warmup）到 1e-6、EMA0.99、15Hz/H15。

数据为 `/meta_eon_cfs/home/hph/heterovla-data/aligned/V3` 当前合并版：55 条训练源轨迹，34,013 个 H15 窗口；7 条验证源轨迹，743 个窗口。两部分均为原始有效连续段，不使用 imputation。归一化采用新版训练集资产，文件摘要由 manifest 锁定。官方训练入口不执行周期验证。

此目录和新运行输出独立于旧实验。CPU 预检核对训练策略、初始权重形状、数据和归一化、实际 batch、启动门槛及编译兼容性。GPU 编译与实际显存核验在用户启动后执行，CPU 预检不代表 GPU 已验证。

最终 checkpoint：`/meta_eon_cfs/home/hph/heterovla-model/V3/checkpoints/pi05_heterovla_v3_vitlora/vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_data20261021/19999`。官方目录编号 19999 对应 20000 次实际更新。
