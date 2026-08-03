"""LoRA trainer for IGPO / GRPO multi-turn search agents.

The trainer implements group-relative advantage estimation with optional
turn-level information-gain rewards, PPO-style multi-epoch updates, and
periodic held-out evaluation.

References:
    Wang et al., IGPO, ICLR 2026, arXiv:2510.14967.
    Schulman et al., PPO, arXiv:1707.06347.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from igpo.advantage.turn_level import (
    TurnRewardTrajectory,
    advantage_collapse_rate,
    compute_igpo_advantages,
    grpo_outcome_advantages,
)
from igpo.agent.rollout import RolloutResult, SearchRolloutEngine
from igpo.algo.losses import (
    broadcast_turn_advantages_to_tokens,
    clipped_surrogate_loss,
    compute_token_logprobs,
)
from igpo.eval.evaluator import IGPOEvaluator
from igpo.train.data import default_data_path, default_eval_data_path, load_qa_jsonl


AlgoName = Literal["igpo", "grpo"]
ModelSource = Literal["modelscope", "huggingface", "auto"]
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters for IGPO / GRPO training."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_source: ModelSource = "auto"
    algo: AlgoName = "igpo"
    output_dir: str = "./outputs/igpo_colab"
    data_path: str = ""
    eval_data_path: str = ""
    max_steps: int = 500
    prompts_per_step: int = 4
    group_size: int = 8
    max_turns: int = 3
    max_new_tokens: int = 192
    temperature: float = 1.0
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.001
    gamma: float = 0.95
    ppo_epochs: int = 4
    info_gain_type: str = "prob_diff"
    info_gain_norm_mode: str = "separate"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    log_every: int = 1
    eval_every: int = 50
    eval_samples: int = 100
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 1


@dataclass
class StepMetrics:
    """Scalar diagnostics logged after each training step."""

    step: int
    algo: str
    mean_f1: float
    mean_em: float
    mean_outcome: float
    collapse_rate: float
    mean_abs_ig: float
    pg_loss: float
    kl_loss: float
    clipfrac: float
    mean_ratio: float
    num_rollouts: int
    extra: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    """Configure deterministic random-number generators where supported."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_model_and_tokenizer(cfg: TrainConfig, dtype: torch.dtype):
    """Load model and tokenizer with ModelScope-first, HuggingFace-fallback policy.

    Args:
        cfg: Training configuration.
        dtype: Parameter dtype.

    Returns:
        Tuple ``(model, tokenizer)``.
    """
    sources: list[str]
    if cfg.model_source == "auto":
        sources = ["modelscope", "huggingface"]
    else:
        sources = [cfg.model_source]

    last_error: Exception | None = None
    for source in sources:
        try:
            if source == "modelscope":
                from modelscope import snapshot_download

                local_dir = snapshot_download(cfg.model_name)
                tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    local_dir,
                    dtype=dtype,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    use_cache=False,
                )
                return model, tokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                use_cache=False,
            )
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001 - explicit multi-source fallback
            last_error = exc
            logger.warning("Model load via %s failed; applying fallback. error=%s", source, exc)

    raise RuntimeError(f"Failed to load model from sources={sources}: {last_error}")


def build_model_and_tokenizer(cfg: TrainConfig, device: torch.device):
    """Construct a LoRA-adapted causal LM and tokenizer."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = _load_model_and_tokenizer(cfg, dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        model = model.to(device)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()
    return model, tokenizer


def _truncate_middle_tool_turns(
    prompt_ids: list[int],
    turns: list,
    decision_mask: list[int],
    turn_spans: list[tuple[int, int]],
    max_seq_len: int,
) -> tuple[list[int], list[int], list[int], list[tuple[int, int]]]:
    """Truncate intermediate tool responses while retaining prompt and recent turns."""
    response_ids: list[int] = []
    for t in turns:
        response_ids.extend(t.token_ids)
    total_len = len(prompt_ids) + len(response_ids)
    if total_len <= max_seq_len:
        return prompt_ids, response_ids, decision_mask, turn_spans

    # Drop earliest tool turns first.
    keep_turns = list(turns)
    while keep_turns and (len(prompt_ids) + sum(len(t.token_ids) for t in keep_turns)) > max_seq_len:
        drop_idx = next((i for i, t in enumerate(keep_turns) if not t.is_decision), None)
        if drop_idx is None:
            break
        keep_turns.pop(drop_idx)

    response_ids = []
    decision_mask = []
    turn_spans = []
    cursor = 0
    for turn in keep_turns:
        start = cursor
        response_ids.extend(turn.token_ids)
        if turn.is_decision:
            decision_mask.extend([1] * len(turn.token_ids))
            turn_spans.append((start, start + len(turn.token_ids)))
        else:
            decision_mask.extend([0] * len(turn.token_ids))
        cursor += len(turn.token_ids)

    # If still too long, keep the prompt intact and truncate the oldest response tokens.
    overflow = len(prompt_ids) + len(response_ids) - max_seq_len
    if overflow > 0:
        response_ids = response_ids[overflow:]
        decision_mask = decision_mask[overflow:]
        shifted: list[tuple[int, int]] = []
        for start, end in turn_spans:
            start2, end2 = start - overflow, end - overflow
            if end2 <= 0:
                continue
            shifted.append((max(0, start2), end2))
        turn_spans = shifted
    return prompt_ids, response_ids, decision_mask, turn_spans


def _pack_rollout_for_loss(
    rollout: RolloutResult,
    tokenizer,
    turn_advantages: list[float] | None,
    trajectory_advantage: float | None,
    *,
    max_seq_len: int,
    device: torch.device,
):
    """Pack one rollout into token tensors, masks, and advantages."""
    prompt_ids = list(rollout.prompt_ids)
    response_ids: list[int] = []
    decision_mask: list[int] = []
    turn_spans: list[tuple[int, int]] = []

    cursor = 0
    for turn in rollout.turns:
        start = cursor
        response_ids.extend(turn.token_ids)
        if turn.is_decision:
            decision_mask.extend([1] * len(turn.token_ids))
            turn_spans.append((start, start + len(turn.token_ids)))
        else:
            decision_mask.extend([0] * len(turn.token_ids))
        cursor += len(turn.token_ids)

    prompt_ids, response_ids, decision_mask, turn_spans = _truncate_middle_tool_turns(
        prompt_ids, rollout.turns, decision_mask, turn_spans, max_seq_len
    )

    input_ids = torch.tensor(prompt_ids + response_ids, device=device, dtype=torch.long)
    if input_ids.numel() < 2 or not response_ids:
        return None

    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)
    labels = torch.tensor(response_ids, device=device, dtype=torch.long)
    mask = torch.tensor(decision_mask, device=device, dtype=torch.float32)
    if mask.numel() != resp_len:
        mask = torch.ones(resp_len, device=device)

    if turn_advantages is not None and len(turn_spans) > 0:
        asst_indices = [i for i, t in enumerate(rollout.turns) if t.is_decision]
        adv_list = list(turn_advantages)
        if len(adv_list) < len(asst_indices):
            adv_list = adv_list + [0.0] * (len(asst_indices) - len(adv_list))
        adv_list = adv_list[: len(asst_indices)]
        advantages = broadcast_turn_advantages_to_tokens(
            adv_list, turn_spans[: len(asst_indices)], resp_len, device=device
        )
    else:
        a = float(trajectory_advantage or 0.0)
        advantages = torch.full((resp_len,), a, device=device)

    return {
        "input_ids": input_ids,
        "prompt_len": prompt_len,
        "labels": labels,
        "mask": mask,
        "advantages": advantages,
        "tokenizer": tokenizer,
    }


class IGPOTrainer:
    """End-to-end trainer for IGPO and outcome-only GRPO."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = build_model_and_tokenizer(cfg, self.device)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=cfg.learning_rate)
        num_warmup_steps = max(1, int(0.03 * cfg.max_steps))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=cfg.max_steps,
        )
        data_path = cfg.data_path or str(default_data_path())
        self.dataset = load_qa_jsonl(data_path)
        self.history: list[StepMetrics] = []
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._has_peft = isinstance(self.model, PeftModel)
        self.evaluator = IGPOEvaluator(
            self.model,
            self.tokenizer,
            eval_data_path=cfg.eval_data_path or str(default_eval_data_path()),
            max_turns=cfg.max_turns,
            max_new_tokens=cfg.max_new_tokens,
            info_gain_type=cfg.info_gain_type,
            device=self.device,
        )

    def _rollout_engine(self) -> SearchRolloutEngine:
        return SearchRolloutEngine(
            self.model,
            self.tokenizer,
            max_turns=self.cfg.max_turns,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            info_gain_type=self.cfg.info_gain_type,
            device=self.device,
        )

    @torch.no_grad()
    def _ref_logprobs(self, input_ids: torch.Tensor, prompt_len: int, labels: torch.Tensor):
        if self._has_peft:
            with self.model.disable_adapter():
                out = self.model(input_ids=input_ids.unsqueeze(0))
        else:
            out = self.model(input_ids=input_ids.unsqueeze(0))
        logits = out.logits[0, prompt_len - 1 : prompt_len - 1 + labels.numel(), :]
        return compute_token_logprobs(logits.unsqueeze(0), labels.unsqueeze(0)).squeeze(0)

    def _forward_logprobs(self, input_ids: torch.Tensor, prompt_len: int, labels: torch.Tensor):
        out = self.model(input_ids=input_ids.unsqueeze(0))
        logits = out.logits[0, prompt_len - 1 : prompt_len - 1 + labels.numel(), :]
        return compute_token_logprobs(logits.unsqueeze(0), labels.unsqueeze(0)).squeeze(0)

    def train(self) -> list[StepMetrics]:
        """Run the configured number of optimization steps."""
        cfg = self.cfg
        engine = self._rollout_engine()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        pbar = tqdm(range(1, cfg.max_steps + 1), desc=f"train-{cfg.algo}")

        for step in pbar:
            prompts = random.sample(self.dataset, k=min(cfg.prompts_per_step, len(self.dataset)))
            rollouts: list[RolloutResult] = []
            group_ids: list[int] = []

            self.model.eval()
            for gi, sample in enumerate(prompts):
                for _ in range(cfg.group_size):
                    r = engine.rollout(sample["question"], sample["answer"])
                    rollouts.append(r)
                    group_ids.append(gi)

            outcomes = [r.outcome["reward"] for r in rollouts]
            f1s = [r.outcome["f1"] for r in rollouts]
            ems = [r.outcome["em"] for r in rollouts]
            collapse = advantage_collapse_rate(outcomes, group_ids)
            mean_abs_ig = float(
                sum(abs(x) for r in rollouts for x in r.info_gains)
                / max(1, sum(len(r.info_gains) for r in rollouts))
            )

            if cfg.algo == "igpo":
                trajs = [
                    TurnRewardTrajectory(
                        info_gains=r.info_gains,
                        outcome=r.outcome["reward"],
                        group_id=gid,
                    )
                    for r, gid in zip(rollouts, group_ids)
                ]
                turn_advs = compute_igpo_advantages(
                    trajs,
                    gamma=cfg.gamma,
                    norm_mode=cfg.info_gain_norm_mode,  # type: ignore[arg-type]
                )
                traj_advs = [None] * len(rollouts)
            else:
                turn_advs = [None] * len(rollouts)
                traj_advs = grpo_outcome_advantages(outcomes, group_ids)

            packed = []
            old_lps = []
            ref_lps = []
            for r, tadv, sadv in zip(rollouts, turn_advs, traj_advs):
                item = _pack_rollout_for_loss(
                    r,
                    self.tokenizer,
                    tadv,
                    sadv,
                    max_seq_len=cfg.max_seq_len,
                    device=self.device,
                )
                if item is None:
                    continue
                with torch.no_grad():
                    old_lp = self._forward_logprobs(
                        item["input_ids"], item["prompt_len"], item["labels"]
                    )
                    ref_lp = self._ref_logprobs(
                        item["input_ids"], item["prompt_len"], item["labels"]
                    )
                packed.append(item)
                old_lps.append(old_lp)
                ref_lps.append(ref_lp)

            if not packed:
                metrics = StepMetrics(
                    step=step,
                    algo=cfg.algo,
                    mean_f1=float(sum(f1s) / max(1, len(f1s))),
                    mean_em=float(sum(ems) / max(1, len(ems))),
                    mean_outcome=float(sum(outcomes) / max(1, len(outcomes))),
                    collapse_rate=float(collapse),
                    mean_abs_ig=mean_abs_ig,
                    pg_loss=0.0,
                    kl_loss=0.0,
                    clipfrac=0.0,
                    mean_ratio=1.0,
                    num_rollouts=len(rollouts),
                    extra={"skipped": True},
                )
                self.history.append(metrics)
                continue

            self.model.train()
            epoch_logs = []
            for epoch in range(cfg.ppo_epochs):
                self.optimizer.zero_grad(set_to_none=True)
                total_loss_logs = []
                for item, old_lp, ref_lp in zip(packed, old_lps, ref_lps):
                    new_lp = self._forward_logprobs(
                        item["input_ids"], item["prompt_len"], item["labels"]
                    )
                    out = clipped_surrogate_loss(
                        new_lp.unsqueeze(0),
                        old_lp.unsqueeze(0),
                        item["advantages"].unsqueeze(0),
                        item["mask"].unsqueeze(0),
                        clip_eps=cfg.clip_eps,
                        kl_coef=cfg.kl_coef,
                        ref_logprobs=ref_lp.unsqueeze(0),
                    )
                    (out.loss / max(1, len(packed))).backward()
                    total_loss_logs.append(out)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                self.optimizer.step()
                epoch_logs.append(total_loss_logs)
                if step == 1 and epoch == 0:
                    ratio0 = float(
                        sum(x.mean_ratio.item() for x in total_loss_logs) / len(total_loss_logs)
                    )
                    print(f"[diag] step1 epoch0 mean_ratio={ratio0:.6f}")

            self.scheduler.step()
            last_logs = epoch_logs[-1]
            metrics = StepMetrics(
                step=step,
                algo=cfg.algo,
                mean_f1=float(sum(f1s) / max(1, len(f1s))),
                mean_em=float(sum(ems) / max(1, len(ems))),
                mean_outcome=float(sum(outcomes) / max(1, len(outcomes))),
                collapse_rate=float(collapse),
                mean_abs_ig=mean_abs_ig,
                pg_loss=float(sum(x.pg_loss.item() for x in last_logs) / len(last_logs)),
                kl_loss=float(sum(x.kl_loss.item() for x in last_logs) / len(last_logs)),
                clipfrac=float(sum(x.clipfrac.item() for x in last_logs) / len(last_logs)),
                mean_ratio=float(sum(x.mean_ratio.item() for x in last_logs) / len(last_logs)),
                num_rollouts=len(rollouts),
                extra={
                    "epoch_clipfrac": [
                        float(sum(x.clipfrac.item() for x in logs) / len(logs)) for logs in epoch_logs
                    ]
                },
            )
            self.history.append(metrics)
            pbar.set_postfix(
                f1=f"{metrics.mean_f1:.3f}",
                collapse=f"{metrics.collapse_rate:.2f}",
                ig=f"{metrics.mean_abs_ig:.4f}",
                ratio=f"{metrics.mean_ratio:.3f}",
            )

            if step % cfg.log_every == 0:
                with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(metrics)) + "\n")

            if cfg.eval_every > 0 and step % cfg.eval_every == 0:
                eval_metrics = self.evaluator.evaluate(n_samples=cfg.eval_samples)
                eval_metrics["step"] = step
                with (self.output_dir / "eval_metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_metrics) + "\n")

        self.model.save_pretrained(self.output_dir / "lora")
        self.tokenizer.save_pretrained(self.output_dir / "lora")
        with (self.output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
        return self.history


def run_training(cfg: TrainConfig | None = None) -> list[StepMetrics]:
    """Construct a trainer and execute the full training loop."""
    cfg = cfg or TrainConfig()
    trainer = IGPOTrainer(cfg)
    return trainer.train()
