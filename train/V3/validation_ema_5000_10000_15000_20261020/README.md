# V3 EMA 验证：5000 / 10000 / 15000

当前仅准备脚本，不运行GPU验证或显存等待任务。`queue_eval.py` 默认只检查计划，必须显式添加 `--execute` 才会启动。

## 固定评测口径

- 服务器16.9，仅使用GPU5：`GPU-e66640c0-8e1e-3f09-8f3f-2d10f877ec04`。
- 顺序读取当前官方V3 r2运行的5000、10000、15000目录，只加载`params`中的EMA；这些目录分别对应实际5001、10001、15001次更新。
- 模型架构、归一化、FP32/bf16参数精度与训练checkpoint一致；ViT LoRA rank8/alpha16，其他层dense。
- 当前V3验证集15Hz/H15，2426个完整窗口；batch16，末批10个真实窗口补6行，仅统计真实样本。
- 关闭训练增强，固定seed42及`fold_in(key42, batch_window_start)`，所有checkpoint使用同一输入缓存和随机采样。
- 主指标`official32_loss`保留官方32维动作平均；另报告`real14_loss`及三个机器人分组loss。先检查首批与官方`compute_loss`的一致性。
- 不加载Adam、不重新初始化基座、不修改训练。GPU显存池24GiB，启动前至少要求28GiB空闲；当前图的实际显存要在首次GPU编译时核验。

训练日志是raw权重和训练增强下的loss；这里是EMA和无增强验证，只能将各checkpoint的验证结果直接相互比较，不能仅凭训练/验证两个绝对数值断定过拟合。

## 文件

- `build_cache.py`：仅用CPU构建并验证输入缓存；中途停下的同一缓存可核验后续建。
- `evaluate_checkpoint.py`：单checkpoint评估，保留原dtype，仅官方loss末尾维度平均被展开以报告分组指标。
- `queue_eval.py`：CPU单次队列，每个checkpoint先CPU检查结构，再等GPU5显存；遇到真实错误停止，不自动换卡或重试GPU任务。
- `test_queue_contract.py`：CPU测试GPU身份、显存门、永久claim、完整窗口数和默认不执行模式。
- `plan.json`：绝对路径、三份checkpoint、GPU5及所有脚本SHA锁定。

## 后续手动启动

在16.9执行；当前尚未执行此命令：

```bash
/meta_eon_cfs/home/hph/heterovla/.venvs/openpi-pi05/bin/python -B \
  /meta_eon_cfs/home/hph/heterovla/train/V3/validation_ema_5000_10000_15000_20261020/queue_eval.py \
  --plan /meta_eon_cfs/home/hph/heterovla/train/V3/validation_ema_5000_10000_15000_20261020/plan.json \
  --execute
```

命令在前台执行；需要长期保持时可自行放在tmux中。未加`--execute`只显示准备状态。启动后若15000尚未保存，会等待完整checkpoint；显存不足也只用CPU等待，不会占GPU。

报告根目录：`/meta_eon_cfs/home/hph/heterovla-model/V3/validation/official_ema_5000_10000_15000_20261020`。各checkpoint的结果位于`results/5000`、`results/10000`、`results/15000`，全部完成后生成`comparison.json`。

当前CPU缓存准备由用户要求停止，已完成的2240窗保留；后续显式启动队列时会核验并补齐，不影响训练数据或权重。
