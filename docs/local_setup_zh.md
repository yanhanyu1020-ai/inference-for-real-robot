# yhy 本机运行

项目根目录为 `/meta_eon_cfs/home/yhy/heterovla`。数据、模型分别位于相邻的 `heterovla-data`、`heterovla-model`。当前训练数据使用 `aligned/V3`。

2026-10-24 验收：240 个已安装包依赖兼容，58 项采集/控制测试及 7 项动作窗口测试通过，V1 两个配置入口通过，V3 完整 CPU 预检的 7 组合同检查通过。训练集 34,013 个窗口、验证集 743 个窗口可正常加载，真实 batch 与模型输入一致；现有 `19999` checkpoint 的 63 组参数键/形状及归一化数值匹配。汇总报告为 `.cache/migration/summary.json`。

```bash
cd /meta_eon_cfs/home/yhy/heterovla
source env.sh
python scripts/check_environment.py
```

`env.sh` 激活本地 `.venvs/openpi-pi05`，配置本地源码、数据、权重、tokenizer 缓存及离线 W&B。不依赖当前登录用户的 `$HOME`。Python 为 3.11.15，JAX/JAXlib 0.5.3、Flax 0.10.2、PyTorch 2.7.1、NumPy 1.26.4。环境来自原项目已安装版本；复制后修复了 Python 链接、激活脚本、命令 shebang 和 editable 安装路径。

基座权重位于 `../heterovla-model/pretrained/pi05_base`，tokenizer 位于 `.cache/openpi/big_vision/paligemma_tokenizer.model`。现有 V3 最终模型仍位于原来的 `19999` checkpoint 目录。

完整 CPU 验收：

```bash
bash scripts/check_local.sh
```

检查依赖、采集与控制测试、动作窗口、真实视频抽样解码、训练/验证数据加载、归一化、模型参数结构及训练合同。日志与环境报告写入 `.cache/migration/`，V3 验收报告为 `train/V3/local_yhy/preflight_report.json`。CPU 验收不等于 GPU 前反向传播验收。

CPU batch 验收使用 `num_workers=0`，以支持不提供多进程共享内存的容器；训练配置仍保留原来的 `num_workers=2`。

## V3 训练

适配入口为 `train/V3/local_yhy/`，从 `official_vitlora_r8_a16_h15_data20261021_fresh2` 复制并修正路径。沿用原来的 ViT LoRA r8/alpha16、其他层 dense、batch32、horizon15、EMA0.99、20,000 步方案，新实验名以 `_yhy` 结尾，输出到本地模型目录。

在分配到 GPU 的计算节点上先检查：

```bash
cd /meta_eon_cfs/home/yhy/heterovla
source env.sh
python scripts/check_environment.py --require-gpu
bash train/V3/local_yhy/train.sh --check --gpu 0
```

启动训练的命令为：

```bash
bash train/V3/local_yhy/train.sh --gpu 0
```

该入口启动完整新实验，要求至少 80,000 MiB 显存、95% 空闲且没有其他计算进程，不自动覆盖或恢复现有模型。仅检查时保留 `--check`。修改代码或数据后，重新执行 CPU 验收以重建 manifest 与预检报告。原 `official_*`、`manual_*`、`*_resume` 目录保留历史合同与路径，不能直接作为迁移后的入口。

## 已有模型推理

```bash
source env.sh
python train/V3/local_yhy/serve_policy.py --check-only
```

上面的 CPU 检查核对 `19999` checkpoint 的参数键、形状及训练归一化数值，不加载完整模型数组。

在有 GPU 的节点启动真实策略服务：

```bash
python train/V3/local_yhy/serve_policy.py --gpu 0 --host 127.0.0.1 --port 8000
```

该服务使用配套的 ViT LoRA 架构和 checkpoint 内归一化。客户端使用 OpenPI WebSocket 协议，输入为三路 RGB 图像、30 维 state 和 prompt，返回 `(15, 14)` 动作。实机接入还需要原有坐标、单位、执行频率和机载执行器配置。

看到 `server listening on 127.0.0.1:8000` 表示服务已经开始监听。在同一 GPU 服务器上另开终端，发一次真实数据推理请求：

```bash
cd /meta_eon_cfs/home/yhy/heterovla
source env.sh
python inference/scripts/check_v3_policy.py --host 127.0.0.1 --port 8000
```

脚本读取验证集样本，向服务发送三路图像、30 维 state 和英文 prompt，检查返回动作是否为有限值的 `(15, 14)` 数组。首次请求需要 JAX 编译，等待时间会较长。该脚本不连接机器人执行器。旧 `test_policy_client.py` 发送的是 mock 格式，不能用于这个真实 V3 服务。

2026-10-24 已通过真实服务推理：`val[0]` 返回 `(15, 14)` 动作，所有数值有限，首次请求耗时 23.497 秒（包含编译开销），没有发送机器人执行命令。结果见 `.cache/migration/v3_inference.json`；该耗时不能作为稳定推理延迟。

`127.0.0.1` 只接受同一网络环境中的本机连接。跨机器的机器人客户端需要 SSH 端口转发，或在服务器上绑定客户端可访问的监听地址。

## V1 独立配置检查

V1 是另一套训练配置。下面的命令只做配置检查，不启动训练或机器人，也不是连接上述 V3 服务的步骤。在新终端运行，避免继承 V3 的数据目录覆盖：

```bash
cd /meta_eon_cfs/home/yhy/heterovla
source train/V1/env.sh
python train/V1/scripts/launch_experiment.py --variant dual_lora
python train/V1/scripts/train_openpi_v1.py --config-only
```

打印 `CONFIGURATION ONLY`、`V1 configuration registered successfully` 后返回命令提示符，表示检查成功。V1 中的模型内部动作维度 32 包含补齐维度；它不表示机器人需要 32 维动作。

## 机器人侧

V3 三机闭环的新实现和部署步骤见 [真机推理说明](../inference/real_robot/README.md)。本地模拟 HTTP 闭环测试已通过；设备环境和真实执行仍待连接后验收。

控制 broker 使用同一环境，可用 `python -m heterovla.control_server --help` 检查入口。Go2/Piper/P450 机载侧仍需设备自己的 SDK、ROS/CAN/DDS、相机驱动和 SSH 连接。采集配置中的服务器数据目录及校验程序已改为本地路径；`heterovla-a800` SSH 别名及登录账号需与实际设备配置一致。此次服务器配置过程没有连接机器人或部署机载代码。

初次 CPU 验收所在沙箱没有可访问的 `/dev/nvidia*`。随后已连接用户启动的 GPU 策略服务并完成一次真实推理，确认该服务的权重加载与推理可运行。训练前反向传播及机器人执行仍未验收。
