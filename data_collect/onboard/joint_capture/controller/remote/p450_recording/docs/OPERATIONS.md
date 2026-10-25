# P450 数据采集操作说明

## 推荐流程

Prometheus 已连接、定位源显示 `MID360 [Valid]` 且无人机未解锁时：

```bash
p450_capture start demo_name
# 遥操并采集
p450_capture finish
```

默认模式记录 `/uav1/camera/color/image_raw/compressed` 的 JPEG 消息，而不是未压缩 RGB。图像、`/Odometry`、MAVROS 状态、遥控器输入和 Prometheus 指令都保留各自 ROS 时间戳并进入同一 rosbag。`finish` 导出 `rgb.mp4`、`pose.csv` 和 `frame_pose.csv`。

## 存储保护

- 剩余空间低于 20 GiB 时拒绝开始。
- bag 每 1 GiB 分卷，单卷损坏不会影响所有数据。
- 默认总采集时长上限为 30 分钟；到时向 rosbag 发送 SIGINT，使当前 bag 正常建立索引。
- `p450_capture status` 在采集中显示 `session_bytes` 和 `free_bytes`。
- 仍应在每段结束后执行 `p450_capture finish`；即使自动到时，它也负责导出并清理状态。

自定义较短时长：

```bash
p450_capture start demo_name --max-minutes 10
```

调试相机原始像素时才使用：

```bash
p450_capture start camera_debug --raw-rgb
```

原始 RGB 约 12.2 MB/s，工具强制限制为最多 2 分钟；默认 JPEG 流实测约 0.5 MB/s。位姿和控制消息仍按原始数值无损保存。

## Prometheus 地面站录制按钮

地面站“录制(R)”可以在笔记本侧额外保存观看用视频，适合作为冗余备份，但它不是本训练数据的主记录：它不能替代同一 rosbag 中带源时间戳的图像、位姿和控制消息。可以同时开启，两者互不冲突。

## 数据位置

机载目录：

```text
/home/amov/p450_data/YYYYMMDD_HHMMSS_demo_name/
```

拉取到 Windows：

```console
p450 transfer demo_name
```

默认进入 `%USERPROFILE%\Downloads\P450\<完整会话名>`。
