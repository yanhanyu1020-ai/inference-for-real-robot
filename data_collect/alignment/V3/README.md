# V3 发布检查

当前数据为`heterovla-data/V3`（实际目录`aligned/V3`），55条训练、7条验证，15Hz、H15；两侧均只包含原始有效片段，不补值。当前loader及模型配置位于`train/V3/dataset/scripts`。

`verify_release.py`根据配置分派到当前`verify_strict_release.py`，逐值核对原始审计表、全量解码导出视频、抽样核对源图像、重算核对训练归一化，并验证真实loader与H15边界。历史补值验证分支仅用于旧版记录。当前代码不依赖已删除的旧版数据目录。

```bash
PYTHONDONTWRITEBYTECODE=1 HF_DATASETS_CACHE=/tmp/heterovla-v3-hf-cache \
  ~/heterovla/.venvs/data-v1/bin/python \
  ~/heterovla/data_collect/alignment/V3/verify_release.py \
  --root ~/heterovla-data/V3 \
  --raw ~/heterovla-data/raw/benchmark \
  --train-scripts ~/heterovla/train/V3/dataset/scripts
```

此检查会更新reports和STATUS；更新后需重建SHA256SUMS清单。迁移时的文件一致性记录在`reports/migration_payloads.json`，此前转换证据保存在`reports/history/`。原始数据不因版本整理而改写。

## 新日期独立批次

10月20日上传数据先按用户要求单独保存，不合并现有V3。处理入口`process_batch.py`使用`alignment_core.py`中与当前V3一致的时间对齐和标签生成规则；`audit_upload.py`核对上传清单全部哈希并全量解码原始视频；`verify_batch.py`核对所有导出表格、视频、源图像对应及归一化和loader。`write_batch_reports.py`生成逐条中文与CSV报告。

```bash
# 原始上传核验
python audit_upload.py --source <benchmark/日期> --output <报告.json>
# 按V3配置对齐到新的暂存目录
python process_batch.py --source <benchmark/日期> --reference <V3目录> --out <暂存目录>
# 原始核验通过后导出（使用相同参数）
python process_batch.py --source <benchmark/日期> --reference <V3目录> --out <暂存目录> --export-only
# 暂存目录reports/upload_integrity.json应保存上传核验结果
python verify_batch.py --source <benchmark/日期> --root <暂存目录> --train-scripts <V3 loader目录>
```

入口依据当前export_manifest.json核对来源与时长；2026-10-20批次删除06、07、08后保留12条、742秒。导出最短片段15帧沿用V3历史格式；模型chunk仍独立为20步。未指定验证集的新批次只有`heterovla_v3_train`，只提供当前有效片段，不自动沿用原验证集的补值策略。归一化仅由本批次生成，不能直接覆盖现有V3的统计。

2026-10-21新批次同样单独处理，原始清单为12条、766秒。`write_batch_reports.py`从批次配置读取日期；只有本批确实存在修复或删除记录才引用这些记录，避免沿用10月20日的说明。报告包括逐任务的状态、动作、整行无效比例，首尾与内部无效帧数，以及`reports/frame_quality/*.csv`逐帧有效性和导出标记。上传端连续性检查与V3对齐阈值分别报告。

02相机修复入口为`repair_task02.py`：仅替换有误取帧，状态和动作沿用原审计值。`raw_camera.py`使用采集端CRC校验读取器读取MCAP，需要`mcap==1.3.0`及`lz4`；可安装到隔离目录并通过PYTHONPATH提供依赖。普通LeRobot训练loader不需要额外MCAP依赖。`camera_raw_frames`非负的溯源项表示直接取自原始MCAP，此时对应`camera_video_frames`和`camera_mapping_rows`为-1，避免误指向降采样视频。

按用户要求删除来源时，`remove_sources.py`先构建保留来源的暂存数据，逐字段检查数值保持、核对视频哈希、重新编号并重算归一化。原始目录删除后需更新其清单和汇总报告，再运行`audit_upload.py`验证剩余文件。最终以当前configs/source_inventory.json为成员清单，删除记录保存于reports/source_removal.json。

## 10月20日、21日并入主V3

`merge_batches.py`核对各父数据集清单，保持已有训练数值和视频字节，合并后重编号并重算训练归一化。10月20日11、13、14、15划入验证，其余20条新增训练。原有3条验证从未补值审计表重新切段，关闭全部补值。当前为55条训练、7条验证，默认chunk15。

`write_release_reports.py`生成当前README与逐来源质量报告；只有完整核验通过后再发布。发布记录和归档路径见主V3的reports/merge_publication.json。合并前主V3与两份独立批次归档于heterovla-data/archive/，原始录制不移动。历史`repair_task02.py`、`remove_sources.py`适用于当时独立批次，不能直接作为当前合并版的通用编辑入口。
