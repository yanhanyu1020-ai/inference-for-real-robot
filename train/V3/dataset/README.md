# V3 数据与 π0.5 接口

数据入口：`/meta_eon_cfs/home/hph/heterovla-data/V3`，对应`aligned/V3`。数据为15Hz、state30、action14，action chunk独立设置为15步。训练55条、45,997帧、856个片段、34,013个窗口；验证7条、4,257帧、251个片段、743个窗口。

```python
import sys
sys.path.insert(0, '/meta_eon_cfs/home/hph/heterovla/train/V3/dataset/scripts')
from dataset_v3 import OpenpiV3Dataset
root = '/meta_eon_cfs/home/hph/heterovla-data/V3'
train = OpenpiV3Dataset(root, 'train')
val = OpenpiV3Dataset(root, 'val')
assert train[0]['actions'].shape == (15, 14)
```

LeRobot ID为`local/heterovla_v3_train`和`local/heterovla_v3_val`，归一化asset为`heterovla_v3`。英文prompt读取`configs/task.json`的`english_translation`。训练和验证均仅接受有效原始状态及动作；`configs/imputation.json`关闭补值。无效时间点切段，窗口不跨缺口。

`pi05_config.py`提供从官方π0.5基座开始的full配置，模型接口32维，输出取前14维。`install_dataset_adapter`把模型chunk传给loader。默认checkpoint命名为`pi05_heterovla_v3_full`，与上级目录既有的`pi05_heterovla_v3_lora_frozen_vit`历史实验区分。本次没有启动训练。

检查入口：

```bash
~/heterovla/.venvs/data-v1/bin/python \
  ~/heterovla/train/V3/dataset/scripts/check_action_chunks.py \
  --data-root ~/heterovla-data/V3 --action-horizon 15 --check-horizons 15
```

训练与验证存在采集会话重叠；验证标签已经去除此前补值。详细文件树、模型路径和当前划分见[说明文档](../../../docs/data_structure_and_pi05_io_zh.md)。
