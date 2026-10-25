# Unitree Go2 采集端源码

本目录保存到主仓库统一版本管理，运行目标是 Go2 上的 Unitree Jetson。

- `collection/`：原 PKL 双相机采集、Piper / Go2 状态、手柄遥操、回放和传输；配置位于 `../../config/collection.json`（相对于本文件所在目录）。
- `recorder/`：独立 Unitree SDK2 DDS 高频 CSV recorder，运行与依赖见其 README。
- 最新 P450–Unitree 联合采集的 Unitree raw MCAP 采集实现位于 `../joint_capture/controller/remote/unitree/`，见 `../joint_capture/README.md`。

服务器校验与 LeRobot 转换位于 `../../server/`，不运行在机器人实时采集循环中。
