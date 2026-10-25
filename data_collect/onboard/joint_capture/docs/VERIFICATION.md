# 快照核验记录

日期：2026-09-29；本机 UTC+08。

- 本机源码基于 joint_time_sync 的 4a0919b 及当前工作区全部选定未提交修改，未改原仓库历史。
- Windows 已安装 src/config/scripts/README/pyproject 共 33 文件与源码 SHA-256 一致；桌面五个 BAT 也一致。
- P450 实际 p450_recording 的八个生产模块与开发镜像哈希一致；另保留实际入口、配置、launch、上游修改和 Pi05 适配器。
- Unitree 正式 onboard 与开发镜像重合的 14 个 Python 文件及 collection_ctl.sh、CAN helper/service 哈希一致。
- Unitree 运行副本来自当日已验证归档，SHA-256 为 16450c7c4affb1336ad63ad6dcc9ce03309d3aedf9d5fa3c6945104c28f16687。已在本机重新确认同一哈希，选择性提取，未发布其旧日志、缓存、私有 SSH 配置和历史 Git 数据。
- Unitree heterovla-recorder 在连接恢复后另行复制并逐文件校验。
- Windows 测试：在 controller 下设置项目根、src、remote/unitree 和快照 onboard 的 PYTHONPATH 后运行 `python -m pytest -q tests`，**373 passed in 28.94s**。
- 最初不区分平台直接执行整个目录的 pytest，因 Windows 缺 ROS rospy 及未设置项目路径/MCAP 路径导致八个收集错误；调整测试范围和环境后通过以上测试，没有修改采集逻辑。
- P450 ROS 专属测试未在本次 Windows 归档环境执行；未启动任何实机采集或运动控制进行验证。

## 待补齐 / 明确限制

1. Unitree SDK2 目录下载再次遇到 SSH 断线。未完成目录被 .gitignore 排除，不会冒充完整依赖上传。
2. Piper SDK 0.6.1、openpi-client 0.1.2 等依赖的版本/来源已记录，当前不包含完整虚拟环境。
3. 旧 A800 上传脚本的服务器端 validator 尚未取得；SDK2 上游 commit、旧 unitree_sdk2py 来源尚未确定。
4. P450 emitter_off.json 在源机器上已缺失。
5. 这是代码恢复资料，不是约 320 GiB 采集数据的备份，也不是文件系统修复完成证明。
