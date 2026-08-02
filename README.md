# IGPO Agentic Search（Colab 复现）

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.14967-b31b1b)](https://arxiv.org/abs/2510.14967)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/annikawang/igpo-agentic-search/blob/main/notebooks/train_igpo_colab.ipynb)

> 高标准、可解释的 **Information Gain-based Policy Optimization (IGPO)** 复现：  
> GRPO 框架 + Agentic Search 多轮过程奖励 + Colab/T4 可跑通。

论文：*Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents* (ICLR 2026)  
官方完整训练代码（8×A100 / veRL）：https://github.com/GuoqingWang1/IGPO

---

## 问题：Advantage Collapse

小 group size 时，常见两种情况：

| 样本 | group 内 rollout | outcome reward | group advantage |
|------|------------------|----------------|-----------------|
| 特别简单 | 全对 | 全是 1 | **全 0** |
| 特别困难 | 全错 | 全是 0 | **全 0** |

advantage=0 → 梯度=0 → 这批样本白采。多轮 search 还缺 turn-level credit assignment。

## 方法：信息增益过程奖励

把每一轮与环境交互看成获取信息。若本轮有效，策略对 GT 的置信应上升：

```text
π(a | q, o≤t) = exp( mean_j log π(a_j | q, o≤t, a_<j) )   # teacher forcing
r_t = π(a | q, o≤t) - π(a | q, o≤t-1)                       # t < T  (prob_diff)
r_T = F1(â, a) 或格式惩罚                                     # 终局
```

然后：

1. group 内对 IG / F1 **separate z-norm**（或 joint）
2. `Ã_t = Σ_{k≥t} γ^{k-t} A_k` 折扣回传
3. 用 turn-level advantage 做 GRPO-style clipped surrogate（tool response mask 掉）

特性：**内生、ground-truth-aware、低 cost、不易 reward hacking**（相对 MCTS / 外部 RM）。

---

## 快速开始

### 本地单测（无需 GPU）

```bash
pip install -e ".[dev]"
python scripts/smoke_test.py
pytest -q
```

### Colab 训练

点上方 **Open in Colab**，或打开：

`notebooks/train_igpo_colab.ipynb`

默认：`Qwen2.5-0.5B-Instruct` + LoRA，`group_size=4`，mock 检索库（无需 Serper）。

### 对照 GRPO

```python
TrainConfig(algo="grpo", ...)  # outcome-only
TrainConfig(algo="igpo", ...)  # IG + F1
```

关注指标：`collapse_rate`（越低越好）、`mean_f1`、`mean_abs_ig`。

---

## 仓库结构

```text
igpo/
  rewards/          # F1 outcome + info-gain process reward
  advantage/        # group z-norm + discounted turn advantage
  algo/             # clipped surrogate (GRPO/IGPO)
  env/              # mock KB search
  agent/            # multi-turn rollout + belief tracking
  train/            # Colab trainer
notebooks/train_igpo_colab.ipynb
tests/
scripts/smoke_test.py
```

---

## 与官方代码的关系

| | 本仓库 | 官方 IGPO |
|--|--------|-----------|
| 目标 | Colab 可跑、算法可讲清、作品集级复现 | 论文级多卡训练 |
| 框架 | 自研精简 trainer | veRL + Ray |
| 检索 | mock KB | Google/Bing API |
| 模型 | 0.5B/1.5B + LoRA | Qwen2.5-7B 全参 |
| 算法 | 与论文 Eq.3–8 / 官方 `prob_diff`+`separate` 对齐 | 完整实现 |

---

## Citation

```bibtex
@inproceedings{wang2026information,
  title={Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
  author={Guoqing Wang and Sunhao Dai and Guangze Ye and Zeyu Gan and Wei Yao and Yong Deng and Xiaofeng Wu and Zhenzhe Ying},
  booktitle={ICLR},
  year={2026}
}
```
