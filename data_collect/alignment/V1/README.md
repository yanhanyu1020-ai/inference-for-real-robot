# V1 数据对齐与转换

这里维护原 V1 的 `prepare_v1.py`、`verify_v1.py`、`finalize_v1.py`。严格窗口 loader 与训练接入位于 `train/V1/scripts/`；数据目录的原脚本副本保留为不可变 provenance。

默认输入是 `~/heterovla-data/raw/benchmark`，默认输出是 `~/heterovla-data/aligned/V1`。`HETEROVLA_BENCHMARK_ROOT`、`HETEROVLA_RAW_ROOT`、`HETEROVLA_DATA_ROOT` 可以覆盖默认值；命令行参数优先。`verify_v1.py` 从主仓库 `train/V1/scripts/` 导入 loader，可用 `HETEROVLA_ROOT` 指定仓库。

```bash
# 仅在安装好原 CPU 转换依赖、明确要创建派生数据时运行：
python prepare_v1.py --source ~/heterovla-data/raw/benchmark --out /path/to/new-output
python verify_v1.py --root /path/to/new-output --source ~/heterovla-data/raw/benchmark
python finalize_v1.py --root /path/to/new-output --source ~/heterovla-data/raw/benchmark
```

`finalize_v1.py` 同时兼容旧的 positional output root，只有验证和源文件完整性报告均通过才发布 README 与清单。

转换器保留既有 V1 的 `task_*` 布局、固定任务定义、划分和隔离规则。它不会自动把新增 26 条录制接入 V1；新增数据需要先确认格式、任务与划分，再另行生成派生版本。默认路径只描述目录布局；现有 V1 已发布，无需因这次整理重新转换或覆盖。

三台机器的动作语义仍为 Go2/UAV 的 kinematic proxy 与 Piper 记录目标，物理跨设备同步精度及完整 operator-command BC 的限制见对齐数据 README。
