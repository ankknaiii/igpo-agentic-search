# Changelog

## 0.2.1

- One-click Colab notebook (`Runtime → Run all`) with forced `main` sync and torchao cleanup.
- Prompt template switched to `string.Template` to avoid JSON brace collisions.
- Lazy PEFT import with automatic removal of incompatible Colab `torchao`.
- README rewritten in structured research-engineering style; official IGPO figures vendored under `assets/`.

## 0.2.0

- 修复采样 token 的 decode–encode 往返，恢复重要性采样有效性。
- 引入 PPO 多 epoch 内循环、无偏 KL 估计、LR warmup 与 cosine schedule。
- 修正 outcome 奖励归属、SQuAD 风格 F1、turn advantage 填充逻辑。
- 扩充离线检索语料库与离线 QA 训练集/评估集；新增评估器与消融脚本。
- 支持 ModelScope 优先、HuggingFace 回退的模型加载策略。
- 专业化 README 与 Colab Notebook 文本。

## 0.1.0

- 初始轻量级 IGPO 复现实现。
