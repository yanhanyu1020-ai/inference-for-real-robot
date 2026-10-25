# 依赖与来源

## Unitree

| 组件 | 已观测版本 / 来源 | 保存方式 |
| --- | --- | --- |
| Gamepad_PiPER | https://github.com/kehuanjack/Gamepad_PiPER @ aa00c80e100d687a5f4a386b79b83ece06b3a30f | 当前源码与 MIT 许可证；main.py、main_virtual.py、src/gamepad_base.py 包含现场修改 |
| cuRobo | https://github.com/NVlabs/curobo @ d64c4b005459db10c5dd867d8b30a87d5bda9bdb | 源码、模型、LICENSE、LICENSE_ASSETS，以及现场 aarch64/Python 3.8 编译扩展；不能假设扩展可移植 |
| Piper SDK | https://github.com/agilexrobotics/piper_sdk @ c05c5454b1cf61c05ad26385e0c0a3aa6d3c7bad，0.6.1 | 原部署无本地修改；按精确提交安装 |
| Unitree SDK2 | 原目录 /home/unitree/unitree_sdk2-main，CMake 2.0.0，无 .git | 上游提交无法确定；完整现场源码/库快照待网络恢复补齐，不以最新上游替代已验证版本 |
| MCAP Python | 1.3.0 | onboard/mcap；补充官方 releases/python/mcap/v1.3.0 标签的 MIT LICENSE，标签对象 4b298de17d0f2cb4adb675437a026a45a7868d9e |
| openpi-client | 0.1.2，Apache 2.0 | 旧策略控制使用，按版本安装；现场包源码快照待补 |
| websockets | 13.1 | 旧策略控制依赖 |

采集环境：系统 Python 3.8、OpenCV 4.2、NumPy 1.17.4、LZ4 3.0.2；原始 MCAP 依赖详见 `controller/remote/unitree/requirements-raw.txt`。相机通过 OpenCV/V4L2 与 udev 定位。

手柄环境：`/home/unitree/miniconda3/envs/cuRobo/bin/python`，Python 3.8.20、Torch 2.0.0+nv23.5、NumPy 1.24.4、SciPy 1.10.1、pygame 2.6.1、viser 1.0.30、yourdfpy 0.0.60。Torch/CUDA 必须匹配 JetPack，不要在 Jetson 上直接安装通用 x86 CUDA wheel。

旧控制脚本还探测 unitree_sdk2py，现场未找到对应源码目录，恢复旧链路时需单独核实。`heterovla-recorder/source_git_commit.txt` 记录 c071a8dbeb5166968a5520da140a7943aa2a6a67。

## P450

以下未修改的上游组件按 URL + 提交恢复，再覆盖 `robots/p450/` 中的现场文件。没有复制 ROS、驱动的 build/devel 安装产物。

| 项目 | 来源 | 提交 |
| --- | --- | --- |
| p450_experiment | https://gitee.com/amovlab1/p450_experiment.git | 8d26fb83286766adc1e97a064af234de2bf78d4c |
| Prometheus | https://gitee.com/amovlab1/Prometheus.git | eacfde31400906053c9826b59b798755b3ed4f89 |
| realsense-ros | https://gitee.com/amovlab1/realsense-ros.git | b1069e2beff204c5e67f78cfb1ee7d6280a63195 |
| livox_ros_driver2 | https://gitee.com/amovlab1/livox_ros_driver2.git | 4dd44023dd3c2870c405d50811a19e8d5ec4f101 |
| spirecv-ros | https://gitee.com/amovlab1/spirecv-ros.git | 52e51c6579eed23c722cfcac0c515fc4c8f902cc |
| SpireCV | https://gitee.com/amovlab/SpireCV.git | 9122d8391f9327cb16f522c054324d072f4a8798 |
| Livox-SDK2 | https://github.com/Livox-SDK/Livox-SDK2.git | 6a940156dd7151c3ab6a52442d86bc83613bd11b |
| librealsense | https://github.com/IntelRealSense/librealsense.git | c94410a420b74e5fb6a414bd12215c05ddd82b69 |
| MAVROS | https://gitee.com/amovlab1/prometheus-mavros.git | 55baa38519bbe8e2bf3c2d4b00ae928b523123df |
| MAVLink | https://gitee.com/amovlab1/mavlink-gbp-release.git | 9cd1f5b551134ee08137a84ec7dc01f24f6432bf |

旧传感器可选依赖：amovlab1/bluesea2 @ 818295641cc5a14b57320008f28b21fe8cc85823，amovlab1/rplidar_ros @ bf4fbfbc9f0f7dc0f34f6e97c2bc225eb448d32d（均 Gitee）。

现场上游修改已保存：相机标定 YAML、p450_onboard.sh、FAST_LIO laserMapping.cpp 与 logging_safety.hpp/test、Livox quick-start main.cpp/MID360 配置/待机脚本、SpireCV 算法配置、MAVROS px4.launch。

Prometheus、RealSense、SpireCV 的 Apache 2.0 声明，以及 Livox、MAVROS 对应许可证保持原样。p450_experiment、mission、spirecv-ros 的部分许可证仍为上游 TODO，不应将其宣称为本项目新发布的 MIT 代码。
