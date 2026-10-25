> 2026-10-19数据整理：旧V2数据集及其变体已清理，当前数据入口为`heterovla-data/V3`，当前接口见`heterovla/train/V3/dataset/README.md`。本目录原有内容是历史实验或转换过程，保留用于追溯；旧训练实验不能自动按新数据恢复。

# V2_15Hz 数据读取

默认入口`/meta_eon_cfs/home/hph/heterovla-data/V2_15Hz_train35`，当前指向`aligned/V2_15Hz_train35_valhold`。35条训练、3条验证，15Hz，action chunk独立设为20。

```python
from dataset_15hz import OpenpiV215HzDataset
root = '/meta_eon_cfs/home/hph/heterovla-data/V2_15Hz_train35'
train = OpenpiV215HzDataset(root, 'train')
val = OpenpiV215HzDataset(root, 'val')
assert train[0]['actions'].shape == (20, 14)
```

训练集只接受原始有效动作，不补值；验证集依据`configs/imputation.json`接受上一有效值保持的状态和动作，保留原始有效标记及补值标记。验证集读取不会在片段末尾补造额外窗口，也不跨来源取样。

训练有30,111帧、348个片段、24,164个H20窗口；验证有2,468帧、3个片段、2,411个窗口。验证来源为10月13日task_12、task_13及10月15日task_04，和训练集存在采集会话重叠。原验证集10月18日的三条已换入训练。

任务prompt从`configs/task.json`的`english_translation`读取。`front → base_0_rgb`、`wrist → left_wrist_0_rgb`、`uav → right_wrist_0_rgb`。模型动作维度32，14维数据补齐后输入，输出截取前14维。

`pi05_config.py`读取相同chunk设置与本版本训练归一化统计；`install_dataset_adapter`保证模型与loader的chunk长度一致。使用`--data-root`或`HETEROVLA_DATA_ROOT`可指定其他版本，`--action-horizon`可覆盖步数。本次未启动训练。

检查：`check_action_chunks.py --data-root <数据根目录> --action-horizon 20`。完整文件树、模型路径和补值规则见[数据结构与π0.5输入输出说明](../../docs/data_structure_and_pi05_io_zh.md)。
