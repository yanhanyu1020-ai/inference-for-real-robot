# 手动续训当前官方V3实验

在16.7的hph终端运行。没有自动监测或后台轮询；每次手动执行只尝试一次。启动后由独立监督器维持训练，断开SSH不会停止训练。

自动选择一张真正空闲的80GB卡：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V3/manual_resume/resume.sh
```

指定3号卡，若忙则返回waiting，不会换卡：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V3/manual_resume/resume.sh --gpu 3
```

仅检查GPU和最新完整checkpoint，不启动：

```bash
bash /meta_eon_cfs/home/hph/heterovla/train/V3/manual_resume/resume.sh --check
```

默认续训固定run `vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_20261019_r2`。每次查找最新已提交、具有params与train_state的checkpoint，并只读核对实际step和两组Adam计数。恢复raw权重、Adam、EMA和LR进度，目标仍为20000实际更新；不会重新warmup、重新加载基座或清零LoRA。

每次手动启动创建独立code/queue/永久attempt，旧失败记录保留。run级锁由监督器和训练子进程继承，已有训练时只返回状态，避免重复起训。GPU要求至少80000MiB、至少95%显存空闲且无compute进程；没有空卡就立即返回，不自动等待。

输出包含GPU、恢复点、真实PID及`log`。`status=running`仅代表进程已启动；实际权重恢复、显存编译和训练更新应再看该日志及run中的runtime_status.json。训练失败不会自动重启。

## 本次缓存修正与检查

JAX0.5.3的`compilation_cache.is_cache_used`会记住首次调用时的启用状态。之前在编译前临时关cache仍可能读取已缓存程序，并丢失donation alias内存统计。本入口在任何JAX导入前设置`JAX_ENABLE_COMPILATION_CACHE=false`，并监测整个进程无持久化缓存读取；训练数学、batch32、EMA0.99和官方Adam不变。

已完成真实CPU检查：使用官方TrainState和train_step、独立进程预热缓存后全进程禁cache、零cache读取、donation alias有效、下一次raw/Adam/EMA及指标逐bit一致。`--prepare-only`也已在16.7通过，读取最新12000目录对应实际12001步，未启动监督器或GPU。当前修正版的GPU编译和更新会在用户自行启动后验证。

当前数据sampler按官方resume行为重新建立seed42 shuffle，官方checkpoint不保存sampler位置，因此不保证恢复后batch顺序与未中断时完全相同。
