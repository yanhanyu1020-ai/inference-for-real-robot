# V3：14 维状态与块起点相对关节角／速度

该实验使用已有 V3 原始数据，训练样本加载时选择 state14，再调用官方 `DeltaActions`。六个关节角、Go2/P450 线速度及偏航角速度都减去同一个动作块起点状态，夹爪保持绝对开度。推理反归一化后，官方 `AbsoluteActions` 加回这个起点，返回执行端使用的物理动作。

原 `train/V3/local_yhy` 配置、原始 Parquet 和原归一化资产保持不变。新训练、检查点及归一化资产使用独立名称。旧 V3 checkpoint 不能当作此新语义模型使用。

## 14 维顺序

以下维度从 1 开始：

| state14 / action14 | 当前状态 | 动作标签 | 单位 |
| --- | --- | --- | --- |
| 1–3 | Go2 vx、vy、yaw_rate | 未来速度减去块起点对应速度 | m/s、m/s、rad/s |
| 4–9 | Piper 实测 q1…q6 | 目标关节角减去块起点实测角 | rad |
| 10 | Piper 实测夹爪开度 | 绝对目标开度，保持原值 | m |
| 11–14 | P450 vx、vy、vz、yaw_rate | 未来速度减去块起点对应速度 | m/s、m/s、m/s、rad/s |

state30 的零起始索引为 `[7,8,9,10,11,12,13,14,15,16,24,25,26,27]`。数据集适配器先选出这 14 个值，服务端要求 state 长度恰好为 14。PI05 的状态编码明确开启，编码时只有这 14 个真实值；为了兼容基础模型，随后 state 和 action 张量都补零至 32，额外 18 维仅为接口填充。

设块起点状态为 `s_t`，第 k 步物理动作标签为 `a[k]`：

```python
relative[k, d] = a[k, d] - s_t[d]  # d != 9，k=0..14
relative[k, 9] = a[k, 9]           # 夹爪绝对开度
physical[k, d] = predicted_relative[k, d] + s_t[d]
```

整个块共享同一个 `s_t`，不减去未来状态、不逐步积分，也不在每一步改用新的实测基准。输出 0 相对速度意味着恢复为块起点速度，停车对应的相对值是 `-s_t[d]`。速度增量仍以 m/s 或 rad/s 表示，不是加速度。航向速度按原数据的通道分量做数值差分；这不等同于将整块速度旋转到统一世界坐标系。

这仍然是关节空间动作，不是末端相对位姿。Go2/P450 标签仍来自原数据的未来实际运动代理，转换没有把它们变成已记录的操作者指令。

## 数据与统计

在仓库根目录运行：

```bash
source env.sh
python -m train.V3.relative14.prepare
python -m train.V3.relative14.preflight
```

`prepare` 只读取 train 的 856 个片段、45,997 帧。state 统计覆盖 34,013 个 H15 窗口起点；action 统计覆盖这些窗口的 510,195 个转换后动作步。相同原始动作位于不同窗口时会有不同的参考状态，因此不能用逐帧相邻差分的统计代替。

独立资产位于 `../heterovla-model/V3/relative14/assets/heterovla_v3_relative14/`：

- `norm_stats.json`：14 维 state 与相对 action 的 mean/std/q01/q99。
- `contract.json`：字段顺序、转换掩码、块参考点及输出语义。
- `provenance.json`：计数、输入文件哈希及归一化哈希。

已有同名统计若发生数值变化，程序拒绝覆盖，需创建新的实验版本。验证集不参与统计。

预检使用 CPU 和真实训练/验证样本，验证选择索引、块起点转换、状态编码、32 维填充及反归一化后的物理动作恢复。预检报告在 `../heterovla-model/V3/relative14/preflight.json`。源代码或资产发生变化后，需要重新预检。

## 训练

```bash
# 仅 CPU 检查，不分配 GPU：
bash train/V3/relative14/train.sh --check

# 显式启动新训练；GPU 必须空闲且满足原 V3 的显存要求：
bash train/V3/relative14/train.sh --gpu 0
```

默认沿用 V3 的 ViT-LoRA r8/a16、其余 dense 参数训练、B32、H15、EMA .99 和 20,000 步，从 `pi05_base` 初始化。沿用冻结 ViT dense 的 EMA 修正及编译显存检查。训练器不自动抢占正在运行的策略服务，不自动改 batch size，不覆盖或恢复旧实验。

默认新 checkpoint 根目录：

```text
../heterovla-model/V3/checkpoints/pi05_heterovla_v3_relative14/
  vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_relative14/
```

可以通过 `--run-name` 使用另一个新实验名称。训练保存的每个 checkpoint 附带新的归一化数据和语义标记。

### 后台等待空闲 GPU

在 16.7 的仓库根目录运行：

```bash
source env.sh
python -m train.V3.relative14.watch_gpu start
python -m train.V3.relative14.watch_gpu status
# 仅取消尚未启动的等待任务：
python -m train.V3.relative14.watch_gpu stop
```

默认每 30 秒查询所有 GPU，不分配显存。连续两次检测到某张卡总显存至少 80,000 MiB、空闲显存至少 95%、没有计算进程后，再次检查并执行 `bash train/V3/relative14/train.sh --gpu GPU-…`。训练入口仍执行自己的显存、预检和数据校验。监听进程脱离终端运行；断开 SSH 不影响监听或训练，主机重启后需重新启动监听。

此监听器只调度上面的默认 relative14 实验。重复 `start` 不会重复启动；`attempt.json` 永久记录一次尝试，失败或进程崩溃也不会自动重试或覆盖结果。训练开始后 `stop` 不终止训练。GPU 查询暂时失败时继续等待；预检失效时停止并报告错误。轮询不能阻止其他用户同时分配 GPU，训练入口会在启动时再次检查。

状态和日志目录：

```text
../heterovla-model/V3/relative14/watchers/
  vitlora_r8_a16_dense_b32_lr1e-5_ema099_h15_s42_relative14/
    state.json       # 等待、启动、运行、完成或失败，含进程身份
    watcher.log      # 检查记录
    training.log     # 真正训练的标准输出和错误（启动后产生）
    attempt.json     # 一次性启动记录（启动后产生）
```

`status` 的 `watcher_alive` 检查 PID、启动时间与命令；`runtime` 显示训练器的编译、加载权重、训练进度等状态。查看日志可用 `tail -f` 加上述日志路径。

## 推理

训练产生 checkpoint 后，指定其具体 step 目录：

```bash
python -m train.V3.relative14.serve_policy --checkpoint /path/to/new/checkpoint/19999 --check-only
python -m train.V3.relative14.serve_policy --checkpoint /path/to/new/checkpoint/19999 --gpu 0 --port 8001
python -m train.V3.relative14.check_policy --url ws://127.0.0.1:8001
```

新服务默认端口 8001，要求相匹配的 state14、相对动作归一化和 checkpoint 标记。API 返回的 `(15,14)` 已恢复为物理速度、绝对关节目标和夹爪开度，执行端不能再次加 state。

16.7 真机客户端会根据新服务元数据把网关的原始 state30 投影成 state14 后发送。网关仍可以保留全部传感器用于状态检查；送入策略网络的仅为这 14 维。连接完成后可运行只读诊断：

```bash
python -m real_robot.diagnostic --config inference/real_robot/config/controller_relative14.json --token-file .cache/real_robot/token --cycles 2
```

诊断日志将记录实际送给模型的 14 维 state 及语义版本。已有 state30 服务仍按原方式工作。

## 验证范围

```bash
python -m pytest -q train/V3/relative14/test_relative14.py inference/tests/test_real_robot.py
```

测试包含固定块起点、所有速度/偏航角速度恢复、夹爪保持绝对值、真实样本归一化往返，以及使用真实 OpenPI Policy 包装器和确定性测试采样器的推理后处理检查。CPU 测试不等于新权重已经训练或真机控制已经验收；本次实现阶段没有启动 GPU 训练。
