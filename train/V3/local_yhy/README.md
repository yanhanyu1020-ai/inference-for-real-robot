# yhy V3 入口

从相邻的 `official_vitlora_r8_a16_h15_data20261021_fresh2` 适配，使用本地环境、数据和模型路径。官方训练器、ViT LoRA 和训练参数沿用原方案，独立实验名以 `_yhy` 结尾。

运行说明见 [本机运行](../../../docs/local_setup_zh.md)。CPU 验收使用仓库根目录的 `bash scripts/check_local.sh`；训练前先运行 `bash train/V3/local_yhy/train.sh --check --gpu 0`。
