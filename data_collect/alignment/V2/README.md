> 2026-10-19数据整理：旧V2数据集及其变体已清理，当前数据入口为`heterovla-data/V3`，当前接口见`heterovla/train/V3/dataset/README.md`。本目录原有内容是历史实验或转换过程，保留用于追溯；旧训练实验不能自动按新数据恢复。

# V2：旧5条与新增26条的转换合并

沿用V1的10Hz、state30、action14、H10、因果观测与有效性过滤。V1只读复用，新source_episode使用日期前缀，source_path定位raw真实目录。train/val按原始session隔离，旧划分保留，新Oct12 faf6整场4条为val。

新26条全部生成aligned审计表；10月14日task_03整段Piper command_sent=0，保留审计但没有训练导出，不放宽原V1规则。具体数量以发布数据的STATUS.json和reports/validation.json为准。

在既有CPU转换环境运行（不会启动模型训练）：

```bash
~/heterovla/.venvs/data-v1/bin/python ~/heterovla/data_collect/alignment/V2/run_pipeline.py
```

默认先在aligned/.V2-building-20261015生成，自动执行转换、全视频解码、真实LeRobot/DataLoader、旧V1数值与图像一致性、train-only norm和源数据完整性检查。全部通过后rename发布为aligned/V2。拒绝覆盖非空V2；原始raw及V1不写入。

状态：`~/heterovla-data/aligned/V2_build_20261015.json`。错误会记录状态和traceback，未通过校验的目录不会发布。

`--align-only`只生成审计表；相同源内容下再运行默认pipeline会复用对齐表完成导出。若导出中断，脚本保留失败现场并拒绝覆盖已存在LeRobot目录，需先查明原因再以新staging路径重建。

维护源码在本目录与train/V2；数据中的scripts/是构建时的源码快照。
