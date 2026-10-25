> 2026-10-19数据整理：旧V2数据集及其变体已清理，当前数据入口为`heterovla-data/V3`，当前接口见`heterovla/train/V3/dataset/README.md`。本目录原有内容是历史实验或转换过程，保留用于追溯；旧训练实验不能自动按新数据恢复。

# V2_15Hz：以视频时间轴重新对齐

输入包括已发布 V2 的31条请求示教、10月19日上传的10月18日8条保留示教，以及1条继续隔离的旧来源，共40条审计来源、39条请求示教。新增批次任务编号为01、02、03、05、06、07、08、11；已删除04、09、10不纳入。旧会话划分保持，新增最早f4b0会话3条作为验证集，其他5条作为训练集。所有来源从原始状态、命令、视频时间映射重新计算；不自动加入后续新采集数据。

每行时间来自 `timing/frame_mapping.csv` 的15Hz共同视频网格。图像按各路映射采集时间选择不晚于当前时刻的最近有效帧，状态按实际时间戳选择过去记录。Go2/UAV动作是未来1/15秒的运动速度代理；Piper动作是截至该区间终点最近的有效已发送绝对关节/夹爪目标。逐维单位、坐标系、过滤阈值和原始会话划分沿用V2。

H=15，覆盖约1秒。拒绝越过原录制末端的动作区间；无效帧和时间缺口切段，少于15帧的连续区间不导出。输出状态30维、动作14维、三路640×480 RGB，格式为LeRobot v2.1。

```bash
~/heterovla/.venvs/data-v1/bin/python \
  ~/heterovla/data_collect/alignment/V2_15Hz/run_pipeline.py
```

默认生成 `~/heterovla-data/aligned/V2_15Hz`。先在相邻暂存目录完成对齐、导出和验证，通过后重命名发布。拒绝覆盖已存在的发布目录。`--staging`可指定暂存目录，本次扩展来源后使用`.V2_15Hz-all-building`。`--align-only`仅完成对齐；随后不带该选项运行即可继续导出。已完成片段只有在state/action/provenance与重新计算结果逐值一致时才能续用。PNG中间文件位于本机`/tmp`并在每段编码后清理。构建脚本快照固定后，不允许静默更换已存在的脚本。

校验包含：6项时间与边界回归测试；所有有效行的原始状态、动作及图像时间戳核对；全部导出视频解码；每个导出片段首尾完整窗口对应的原视频像素抽查；真实LeRobot/DataLoader；H15窗口；训练集归一化；原始来源元数据与旧V1/V2发布SHA-256。标签验证使用单独的计算实现。

```bash
~/heterovla/.venvs/data-v1/bin/python \
  ~/heterovla/data_collect/alignment/V2_15Hz/verify.py \
  --root ~/heterovla-data/aligned/V2_15Hz
```

严格loader维护入口：`train/V2_15Hz/scripts/dataset_15hz.py`。构建过程仅使用CPU，不启动模型训练或机器人执行。

后续35/3划分通过`resplit_train35.py`生成`aligned/V2_15Hz_train35`：保留f4b0会话3条作为验证，将父版本另外5条验证来源转入训练。该步骤核对父版本校验清单，保持视频字节及状态/动作/时间不变，仅重排编号、更新split并重算训练归一化统计；H=15暂不调整。根目录`heterovla-data/V2_15Hz`当前指向该35/3版本。
