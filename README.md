# Information Gain-based Policy Optimization for Multi-Turn Search Agents

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.14967-b31b1b)](https://arxiv.org/abs/2510.14967)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankknaiii/igpo-agentic-search/blob/main/notebooks/train_igpo_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于论文 IGPO 的轻量级复现实现，支持 Colab 运行环境。该实现对齐 Wang et al., ICLR 2026（[arXiv:2510.14967](https://arxiv.org/abs/2510.14967)）中的 turn-level 信息增益奖励与 GRPO 风格目标（Eq. 3–8）。

官方完整训练系统（多机 / veRL）：https://github.com/GuoqingWang1/IGPO

本仓库面向有限 GPU 资源下的算法验证与消融实验。

## 问题动机

纯结果驱动的 GRPO 对每条 rollout 仅分配单一标量奖励。在较小 group size 下，常见两类失效模式：

| 查询难度 | 组内结果奖励 | 组相对 advantage |
|----------|--------------|------------------|
| 简单 | 全部正确 | 坍缩为零 |
| 困难 | 全部错误 | 坍缩为零 |

零 advantage 无法产生策略梯度。多轮搜索场景下，工具调用轮次间缺乏有效的信用分配机制。

## 方法概述

每一轮智能体与环境的交互被视为对真实答案信念的增量更新。在 teacher forcing 条件下：

```text
π_θ(a | q, o_≤t) = exp( (1/L) Σ_j log π_θ(a_j | q, o_≤t, a_<j) )
r_t = π_θ(a | q, o_≤t) - π_θ(a | q, o_≤t-1) ,   1 ≤ t < T
r_T = F1(â, a)   # 或格式惩罚
```

默认采用 `prob_diff`，并支持 `log_prob_diff`。组内对信息增益奖励与结果奖励执行 z-normalization（`separate` 或 `joint`），再计算折扣 turn-level advantage：

```text
Ã_t = Σ_{k=t}^{T} γ^{k-t} A_k
```

`gamma` 控制 turn-level advantage 的衰减速率；当 `gamma=1.0` 时，各 turn 的 advantage 退化为后续奖励的简单求和。默认值为 `0.95`。工具响应在 surrogate loss 中被掩码。

## 环境依赖

| 项目 | 要求 |
|------|------|
| Python | >= 3.10 |
| CUDA | 建议 11.8+（CPU 可运行完整性校验） |
| 关键依赖 | `torch`, `transformers`, `peft`, `accelerate`, `modelscope`, `datasets`, `numpy`, `scipy`, `matplotlib`, `tqdm` |

```bash
pip install -e ".[dev]"
```

## 快速开始

### 本地运行

```bash
python scripts/integrity_check.py
pytest -q
```

轻量级训练验证：

```bash
python scripts/integrity_check.py --train-steps 1
```

### Colab 运行

打开上方 Colab 徽章对应 Notebook。运行环境选择 T4 GPU，按顺序执行单元格。首个代码单元格将自动克隆本仓库。

默认基座模型为 `Qwen/Qwen2.5-0.5B-Instruct`（LoRA）。在显存允许的情况下，可通过 `TrainConfig.model_name` 切换至 1.5B 或 3B 规模的 Instruct 模型。模型加载默认 `model_source="auto"`：优先 ModelScope，失败时回退至 HuggingFace。

## 配置说明

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `model_name` | str | `Qwen/Qwen2.5-0.5B-Instruct` | 基座模型标识 |
| `model_source` | str | `auto` | `modelscope` / `huggingface` / `auto` |
| `algo` | str | `igpo` | `igpo` 或 `grpo` |
| `max_steps` | int | 500 | 训练步数 |
| `prompts_per_step` | int | 4 | 每步采样的问题数 |
| `group_size` | int | 8 | 每个问题的 rollout 数 |
| `max_turns` | int | 3 | 最大交互轮数 |
| `ppo_epochs` | int | 4 | 每个 batch 的 PPO 内循环轮数 |
| `gamma` | float | 0.95 | turn-level advantage 折扣因子 |
| `clip_eps` | float | 0.2 | PPO clip 阈值 |
| `kl_coef` | float | 0.001 | KL 正则系数 |
| `info_gain_type` | str | `prob_diff` | `prob_diff` 或 `log_prob_diff` |
| `info_gain_norm_mode` | str | `separate` | `separate` 或 `joint` |
| `eval_every` | int | 50 | 评估间隔（步） |
| `eval_samples` | int | 100 | 每次评估样本数 |
| `learning_rate` | float | 1e-5 | 优化器学习率 |
| `seed` | int | 42 | 随机种子 |

算法切换：

```python
TrainConfig(algo="grpo")  # 纯结果奖励
TrainConfig(algo="igpo")  # 信息增益 + 结果奖励
```

主要记录指标：`collapse_rate`、`mean_f1`、`mean_abs_ig`、`mean_ratio`、`clipfrac`。

## 评估方法

- 训练集：`igpo/data/qa_offline.jsonl`（200 条）
- 评估集：`igpo/data/qa_eval.jsonl`（50 条 held-out，与训练集无重叠）
- 指标：word-level F1、Exact Match、outcome collapse rate、mean \|IG\|
- 流程：训练过程中按 `eval_every` 调用 `IGPOEvaluator`；评估阶段 `temperature=0` 贪心解码
- 结果写入：`outputs/*/eval_metrics.jsonl`

## 消融实验

```bash
python scripts/run_ablation.py --algo igpo --seed 1 --seed 2 --seed 3 --max_steps 50
python scripts/run_ablation.py --algo grpo --seed 1 --seed 2 --seed 3 --max_steps 50
python scripts/analyze_ablation.py --input_dir ./ablation_results --metric collapse_rate
```

对比方法保持模型、数据、超参数一致，仅切换 `algo`。多随机种子结果以均值±标准差报告，并执行 Welch t-test。

## 仓库结构

```text
igpo/
  rewards/       结果 F1；信息增益过程奖励
  advantage/     组归一化；折扣 turn advantage
  algo/          clipped surrogate 目标
  env/           离线检索语料库
  agent/         多轮 rollout 与信念跟踪
  train/         LoRA 训练器
  eval/          held-out 评估器
  data/          离线训练集与评估集
notebooks/train_igpo_colab.ipynb
scripts/integrity_check.py
scripts/run_ablation.py
scripts/analyze_ablation.py
tests/
```

## 与官方实现的差异说明

| 方面 | 本仓库 | 官方 IGPO |
|------|--------|-----------|
| 算力 | 单卡 / Colab T4 | 8×A100 |
| 框架 | 轻量级训练器 | veRL + Ray |
| 检索 | 离线检索语料库 | 在线 Web Search API |
| 模型 | 0.5B–3B + LoRA | Qwen2.5-7B |
| 数据规模 | 离线 200/50 | 大规模基准训练 |
| 算法 | 对齐 Eq. 3–8；`prob_diff` + `separate` | 完整训练系统 |

## 已知限制

1. 离线检索语料库无法覆盖开放域实时网页检索分布。
2. 默认 0.5B + LoRA 主要用于算法通路验证，不追求复现论文 7B 绝对指标。
3. Teacher-forcing 信息增益依赖真实答案，不适用于无标准答案的开放生成任务。
4. 完整消融需多 seed 与足够步数，Colab 免费配额下建议缩短 `max_steps`。

## Citation

```bibtex
@inproceedings{wang2026information,
  title={Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
  author={Guoqing Wang and Sunhao Dai and Guangze Ye and Zeyu Gan and Wei Yao and Yong Deng and Xiaofeng Wu and Zhenzhe Ying},
  booktitle={ICLR},
  year={2026}
}
```
