# Information Gain-based Policy Optimization for Multi-Turn Search Agents

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.14967-b31b1b)](https://arxiv.org/abs/2510.14967)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankknaiii/igpo-agentic-search/blob/main/notebooks/train_igpo_colab.ipynb)

Reproduction of [Wang et al., ICLR 2026](https://arxiv.org/abs/2510.14967): turn-level information-gain rewards for agentic search under a GRPO-style objective.

Reference implementation (multi-node / veRL): https://github.com/GuoqingWang1/IGPO

This repository is a compact, Colab-runnable reimplementation of the core algorithm (Eq. 3–8), intended for method verification and ablation on limited GPUs.

## Motivation

Outcome-only GRPO assigns a single scalar reward per rollout. With small group sizes, two failure modes are common:

| Query difficulty | Within-group outcomes | Group-relative advantage |
|------------------|----------------------|---------------------------|
| Easy | All correct | Collapse to zero |
| Hard | All incorrect | Collapse to zero |

Zero advantage yields no policy gradient. Multi-turn search additionally lacks credit assignment across tool-use turns.

## Method

Each agent–environment turn is treated as an incremental update of the policy’s belief about the ground-truth answer. Under teacher forcing:

```text
π_θ(a | q, o_≤t) = exp( (1/L) Σ_j log π_θ(a_j | q, o_≤t, a_<j) )
r_t = π_θ(a | q, o_≤t) - π_θ(a | q, o_≤t-1) ,   1 ≤ t < T
r_T = F1(â, a)   # or format penalty
```

`prob_diff` matches the paper’s primary definition; `log_prob_diff` is also supported. Rewards are z-normalized within each group (`separate` or `joint`), then converted to discounted turn-level advantages:

```text
Ã_t = Σ_{k=t}^{T} γ^{k-t} A_k
```

Tool responses are masked from the surrogate loss. The optimization objective follows GRPO with turn-indexed advantages (Eq. 8).

## Setup

### Local

```bash
pip install -e ".[dev]"
python scripts/smoke_test.py
pytest -q
```

### Colab

Open the notebook badge above. Select a T4 GPU runtime, then execute cells in order. Cell 1 clones this repository automatically.

Default backbone: `Qwen/Qwen2.5-0.5B-Instruct` with LoRA. Larger instruct checkpoints (1.5B / 3B) may be substituted via `TrainConfig.model_name` when memory allows.

### Algorithm switch

```python
TrainConfig(algo="grpo")  # outcome reward only
TrainConfig(algo="igpo")  # information gain + outcome
```

Primary logged metrics: `collapse_rate`, `mean_f1`, `mean_abs_ig`.

## Layout

```text
igpo/
  rewards/       outcome F1; information-gain process reward
  advantage/     group normalization; discounted turn advantages
  algo/          clipped surrogate objective
  env/           offline mock retrieval
  agent/         multi-turn rollout and belief tracking
  train/         LoRA trainer
notebooks/train_igpo_colab.ipynb
tests/
scripts/smoke_test.py
```

## Scope relative to the official release

| Aspect | This repo | Official IGPO |
|--------|-----------|---------------|
| Compute | Single GPU / Colab T4 | 8×A100 |
| Stack | Lightweight trainer | veRL + Ray |
| Retrieval | Deterministic mock KB | Live web search API |
| Model | 0.5B–3B + LoRA | Qwen2.5-7B |
| Algorithm | Aligned with Eq. 3–8; `prob_diff` + `separate` | Full training system |

## Citation

```bibtex
@inproceedings{wang2026information,
  title={Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
  author={Guoqing Wang and Sunhao Dai and Guangze Ye and Zeyu Gan and Wei Yao and Yong Deng and Xiaofeng Wu and Zhenzhe Ying},
  booktitle={ICLR},
  year={2026}
}
```
