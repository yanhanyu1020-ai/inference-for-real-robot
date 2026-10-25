# P450 RGB 与位姿同步录制

这套工具在 P450 机载电脑上把相机原始图像、MID360/FAST-LIO 里程计及基础飞行状态录入同一组 ROS bag。所有消息保留各自的 ROS `header.stamp`；飞行后可导出 RGB 视频、完整位姿表和逐帧最近位姿对齐表。

工具只订阅和录制数据，不发布控制指令，也不会解锁无人机。

## 安全与启动

首次台架验证前请拆除桨叶，或采用可靠的整机约束。确认急停和遥控器切换可用后，再启动官方飞控与相机节点。

推荐使用一条命令自动准备 MID360、MAVROS/Prometheus、FAST-LIO 和 D435i，并开始记录：

```bash
p450_capture start tea_table_01
```

命令会复用已运行的正确节点、启动缺失节点，并等待 RGB、MID360 位姿、状态和电池确实收到消息。只有 PX4 已连接、未解锁、`location_source=10` 且 `odom_valid=True` 时才开始 rosbag。它不会解锁、切换模式或发布飞行指令。

结束采集并立即导出：

```bash
p450_capture finish
```

`finish` 会先安全停止 rosbag，再导出 MP4、位姿 CSV 和逐帧对齐 CSV。MID360、FAST-LIO、MAVROS 和 D435i 保持运行，便于马上开始下一段。

查看状态或在没有活动录制时关闭本工具启动的节点：

```bash
p450_capture status
p450_capture shutdown
```

底层诊断仍可使用 `p450_record check`。必需话题包括 RGB、相机内参、FAST-LIO `/Odometry`、飞控状态和电池；MAVROS 位姿/速度、IMU、GPS 与 Prometheus 指令话题存在时会自动加入。

若需要手动诊断，节点顺序为 MID360 驱动、MAVROS/Prometheus、FAST-LIO、D435i。FAST-LIO 使用 MAVROS IMU；若 MAVROS 重启，必须重启 FAST-LIO mapping，使积分状态和局部原点重新初始化。

## 底层录制命令

通常无需直接调用本节命令；它们保留用于诊断。开始：

```bash
p450_record start tea_table_01
```

随后正常遥操无人机。结束时务必使用下面的命令，让 rosbag 完成索引并安全落盘：

```bash
p450_record stop
```

查看当前状态：

```bash
p450_record status
```

数据保存到：

```text
/home/amov/p450_data/YYYYMMDD_HHMMSS_tea_table_01/
├── metadata.yaml
├── record.log
└── raw/
    └── flight_*.bag
```

bag 使用 LZ4 压缩，并按 4 GiB 自动分卷。

## 导出训练可用文件

把下面的会话目录替换为实际目录：

```bash
p450_record export /home/amov/p450_data/YYYYMMDD_HHMMSS_tea_table_01
```

输出位于会话的 `export/`：

- `rgb.mp4`：RGB 视频。
- `pose.csv`：全部 MID360/FAST-LIO 局部位姿，坐标系来自 `/Odometry` 的 `camera_init` 局部坐标系。
- `frame_pose.csv`：每帧 RGB 对齐到最近的位姿。

`frame_pose.csv` 中：

- `image_stamp_ns` 和 `pose_stamp_ns` 是原始消息时间戳。
- `delta_ms` 是两者的时间差。
- `valid=1` 表示差值不超过默认的 50 ms；否则为 `0`。

可以指定更严格的阈值：

```bash
p450_record export SESSION_DIR --max-delta-ms 25
```

## 当前同步边界

相机和 MAVROS 数据都在同一台机载电脑、同一个 ROS master 内进入 rosbag，因此单机演示不依赖服务器与机载机的墙上时钟一致。以后把机器狗、机械臂和无人机跨机器联合采集时，再加入统一的 PTP/Chrony 时间源，并在每条数据中保留源设备时间戳。

## 回滚与测试

本地源码使用 Git 分阶段提交。机载机测试命令：

```bash
cd /home/amov/p450_recording
source /opt/ros/noetic/setup.bash
python3 -m unittest -v
```
