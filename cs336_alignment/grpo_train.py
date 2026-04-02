"""
GRPO (Group Relative Policy Optimization) training loop.

This module implements the GRPO training algorithm as described in:
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

Key differences from Expert Iteration:
- Expert Iteration uses SFT on correct rollouts only
- GRPO uses policy gradient with group-normalized rewards
"""

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import get_linear_schedule_with_warmup

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.expert_iter import (
    FormattedExample,
    build_r1_zero_prompt,
    build_sampling_params,
    init_vllm,
    load_formatted_examples,
    load_policy_into_vllm_instance,
)
from cs336_alignment.fine_tuning import (
    get_response_log_probs,
    load_model,
    save_pretrained,
    tokenize_prompt_and_output,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA_PATH = "data/gsm8k/train.formatted.jsonl"
DEFAULT_VAL_DATA_PATH = "data/gsm8k/test.formatted.jsonl"
DEFAULT_SAVE_DIR = "results/grpo"


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    model_path: str
    train_data_path: str
    val_data_path: str
    save_dir: str

    # Training hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    epochs_per_rollout_batch: int = 1  # On-policy
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    cliprange: float = 0.2

    # Sampling hyperparameters
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    # Normalization
    advantage_eps: float = 1e-6
    use_std_normalization: bool = True

    # Model/sampling settings
    seed: int = 42
    gpu_memory_utilization: float = 0.85

    # Validation
    max_eval_samples: int = 1024
    eval_every_steps: int = 5


class RolloutDataset(Dataset):
    """Dataset for rollout examples."""
    def __init__(self, examples: list[FormattedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> FormattedExample:
        return self.examples[idx]


def _truncate_after_answer_tag(text: str) -> str:
    """Truncate text after the </answer> tag."""
    if "</answer>" in text:
        return text.split("</answer>", 1)[0] + "</answer>"
    return text


def _build_collate_fn(tokenizer):
    """Build collate function for rollouts."""
    def _collate(batch: list[FormattedExample]) -> dict[str, torch.Tensor]:
        prompt_strs = [build_r1_zero_prompt(example.question) for example in batch]
        output_strs = [example.answer for example in batch]
        return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    return _collate


def sample_rollouts(
    llm,
    examples: list[FormattedExample],
    group_size: int,
    temperature: float,
    max_tokens: int,
    min_tokens: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Sample rollouts from the policy using vLLM.

    Returns:
        rollout_responses: list of generated responses
        ground_truths: list of corresponding ground truths (repeated group_size times)
    """
    prompts = [build_r1_zero_prompt(example.question) for example in examples]
    sampling_params = build_sampling_params(
        group_size=group_size,
        temperature=temperature,
        top_p=1.0,  # No top-p filtering for training
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        seed=seed,
    )

    outputs = llm.generate(prompts, sampling_params)

    rollout_responses = []
    ground_truths = []

    for example, output in zip(examples, outputs):
        for candidate in output.outputs:
            rollout_responses.append(candidate.text)
            ground_truths.append(example.ground_truth)

    return rollout_responses, ground_truths


def compute_advantages(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute group-normalized advantages for GRPO.

    This function is imported from tests.adapters to use the student's implementation.
    """
    from tests.adapters import run_compute_group_normalized_rewards

    normalized_rewards, raw_rewards, metadata = run_compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )

    # In GRPO, the advantage is the normalized reward
    advantages = normalized_rewards

    return advantages, raw_rewards, metadata


@torch.no_grad()
def evaluate_policy(
    llm,
    policy_model,
    eval_examples: list[FormattedExample],
    max_new_tokens: int,
    seed: int,
) -> tuple[float, dict]:
    """Evaluate policy using vLLM and return accuracy and statistics."""
    load_policy_into_vllm_instance(policy_model, llm)
    prompts = [build_r1_zero_prompt(example.question) for example in eval_examples]
    sampling_params = build_sampling_params(
        group_size=1,
        temperature=0.0,  # Greedy for evaluation
        top_p=1.0,
        max_tokens=max_new_tokens,
        min_tokens=1,
        seed=seed,
    )

    outputs = llm.generate(prompts, sampling_params)

    total = 0
    correct = 0
    format_ok = 0
    total_reward = 0.0

    for example, output in zip(eval_examples, outputs):
        if not output.outputs:
            continue
        response = output.outputs[0].text
        reward = r1_zero_reward_fn(response, example.ground_truth)
        r = float(reward["reward"])
        total_reward += r
        correct += int(r == 1.0)
        format_ok += int(float(reward["format_reward"]) == 1.0)
        total += 1

    accuracy = 0.0 if total == 0 else correct / total
    stats = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "format_ok_rate": 0.0 if total == 0 else format_ok / total,
        "mean_reward": 0.0 if total == 0 else total_reward / total,
    }

    return accuracy, stats


def run_grpo_experiment(config: GRPOConfig) -> dict[str, object]:
    """Run the GRPO training loop."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Load data
    train_examples = load_formatted_examples(config.train_data_path)
    val_examples = load_formatted_examples(config.val_data_path)

    if len(train_examples) == 0 or len(val_examples) == 0:
        raise ValueError("Train or validation dataset is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy model
    model, tokenizer = load_model(model_path=config.model_path)
    model.to(device)
    model.train()

    # Initialize vLLM for rollouts
    llm = init_vllm(
        model_id=config.model_path,
        device="cuda",
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )

    # Create output directory
    output_dir = Path(config.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "grpo_metrics.jsonl"

    # Sanity checks
    assert config.train_batch_size % config.gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    assert config.rollout_batch_size % config.group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size
    assert config.train_batch_size >= config.group_size, \
        "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = config.rollout_batch_size // micro_train_batch_size

    logger.info("GRPO Configuration:")
    logger.info(f"  n_grpo_steps: {config.n_grpo_steps}")
    logger.info(f"  rollout_batch_size: {config.rollout_batch_size}")
    logger.info(f"  group_size: {config.group_size}")
    logger.info(f"  train_batch_size: {config.train_batch_size}")
    logger.info(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    logger.info(f"  loss_type: {config.loss_type}")
    logger.info(f"  learning_rate: {config.learning_rate}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # Training state
    all_step_metrics = []
    train_rng = random.Random(config.seed)
    total_start = time.time()

    # Initial evaluation
    clipped_val_examples = val_examples[:config.max_eval_samples] if config.max_eval_samples > 0 else val_examples
    initial_accuracy, initial_stats = evaluate_policy(
        llm=llm,
        policy_model=model,
        eval_examples=clipped_val_examples,
        max_new_tokens=256,
        seed=config.seed,
    )
    logger.info(f"Initial validation accuracy: {initial_accuracy:.4f}")

    # GRPO Training loop
    for step_idx in range(1, config.n_grpo_steps + 1):
        step_start_time = time.time()

        # Sample rollouts
        if config.rollout_batch_size >= len(train_examples):
            step_examples = list(train_examples)
        else:
            indices = train_rng.sample(range(len(train_examples)), k=config.rollout_batch_size)
            step_examples = [train_examples[i] for i in indices]

        # Sample prompts (n_prompts_per_rollout_batch)
        if n_prompts_per_rollout_batch >= len(step_examples):
            prompt_examples = list(step_examples)
        else:
            indices = train_rng.sample(range(len(step_examples)), k=n_prompts_per_rollout_batch)
            prompt_examples = [step_examples[i] for i in indices]

        # Generate rollouts using vLLM
        rollout_responses, ground_truths = sample_rollouts(
            llm=llm,
            examples=prompt_examples,
            group_size=config.group_size,
            temperature=config.sampling_temperature,
            max_tokens=config.sampling_max_tokens,
            min_tokens=config.sampling_min_tokens,
            seed=config.seed + step_idx,
        )

        # Compute advantages
        advantages, raw_rewards, reward_metadata = compute_advantages(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )

        # Move to device
        advantages = advantages.to(device)
        raw_rewards_tensor = raw_rewards.to(device)

        # Create rollout dataset
        rollout_examples = [
            FormattedExample(
                question=prompt_examples[i // config.group_size].question,
                answer=rollout_responses[i],
                ground_truth=ground_truths[i],
            )
            for i in range(len(rollout_responses))
        ]
        rollout_dataset = RolloutDataset(rollout_examples)
        rollout_loader = DataLoader(
            rollout_dataset,
            batch_size=micro_train_batch_size,
            shuffle=True,
            collate_fn=_build_collate_fn(tokenizer),
        )

        # Compute old_log_probs for grpo_clip (only needed for off-policy)
        # For on-policy (epochs_per_rollout_batch=1), we compute on-the-fly
        old_log_probs = None
        if config.loss_type == "grpo_clip":
            # For grpo_clip, we need old_log_probs from the current policy before update
            # Since we're on-policy, we compute them along with the new ones
            pass

        # Training for epochs_per_rollout_batch
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_losses = []
        epoch_clip_fractions = []

        for epoch_idx in range(config.epochs_per_rollout_batch):
            for batch_idx, batch in enumerate(rollout_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                response_mask = batch["response_mask"].to(device).float()

                # Get log probs from current policy
                policy_output = get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                policy_log_probs = policy_output["log_probs"]

                # For grpo_clip, we need old_log_probs
                # On-policy: we use the same policy's log probs from the previous step
                # Actually for on-policy GRPO with single epoch, old_log_probs == policy_log_probs
                if old_log_probs is None and config.loss_type == "grpo_clip":
                    # Detach to treat as old log probs for clipping
                    old_log_probs = policy_log_probs.detach()

                # Get advantages for this microbatch
                start_idx = batch_idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                batch_advantages = advantages[start_idx:end_idx].unsqueeze(-1)  # (micro_batch, 1)
                batch_raw_rewards = raw_rewards_tensor[start_idx:end_idx].unsqueeze(-1)

                # Compute loss using the student's implementation
                from tests.adapters import run_grpo_microbatch_train_step

                loss_kwargs = {
                    "policy_log_probs": policy_log_probs,
                    "response_mask": response_mask,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "loss_type": config.loss_type,
                }

                if config.loss_type == "no_baseline":
                    loss_kwargs["raw_rewards"] = batch_raw_rewards
                else:
                    loss_kwargs["advantages"] = batch_advantages

                if config.loss_type == "grpo_clip":
                    loss_kwargs["old_log_probs"] = old_log_probs
                    loss_kwargs["cliprange"] = config.cliprange

                loss, metadata = run_grpo_microbatch_train_step(**loss_kwargs)

                epoch_losses.append(float(loss.item()))
                if "clipped_ratio_fraction" in metadata:
                    epoch_clip_fractions.append(metadata["clipped_ratio_fraction"])

                # Note: backward is called inside run_grpo_microbatch_train_step

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_elapsed = time.time() - step_start_time

        # Evaluation
        if step_idx % config.eval_every_steps == 0 or step_idx == config.n_grpo_steps:
            val_accuracy, val_stats = evaluate_policy(
                llm=llm,
                policy_model=model,
                eval_examples=clipped_val_examples,
                max_new_tokens=256,
                seed=config.seed + 10000 + step_idx,
            )
        else:
            val_accuracy, val_stats = None, {}

        # Log metrics
        step_metrics = {
            "step": step_idx,
            "rollout_reward_mean": float(raw_rewards.mean().item()),
            "rollout_reward_std": float(raw_rewards.std().item()),
            "rollout_reward_max": float(raw_rewards.max().item()),
            "rollout_reward_min": float(raw_rewards.min().item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std().item()),
            "train_loss_mean": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
            "train_clip_fraction": sum(epoch_clip_fractions) / len(epoch_clip_fractions) if epoch_clip_fractions else 0.0,
            "val_accuracy": val_accuracy,
            "val_mean_reward": val_stats.get("mean_reward", 0.0),
            "val_format_ok_rate": val_stats.get("format_ok_rate", 0.0),
            "step_elapsed_sec": step_elapsed,
            "total_elapsed_sec": time.time() - total_start,
        }

        all_step_metrics.append(step_metrics)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_metrics) + "\n")

        logger.info(
            f"[step {step_idx}/{config.n_grpo_steps}] "
            f"rollout_reward={step_metrics['rollout_reward_mean']:.4f} "
            f"train_loss={step_metrics['train_loss_mean']:.4f} "
            f"val_acc={val_accuracy if val_accuracy is not None else 'N/A'} "
            f"elapsed={step_elapsed:.1f}s"
        )

        # Save checkpoint
        if step_idx % config.eval_every_steps == 0 or step_idx == config.n_grpo_steps:
            step_model_dir = output_dir / f"step_{step_idx}"
            save_pretrained(str(step_model_dir), model, tokenizer)

    # Summary
    summary = {
        "config": asdict(config),
        "initial_val_accuracy": initial_accuracy,
        "final_val_accuracy": all_step_metrics[-1]["val_accuracy"] if all_step_metrics else None,
        "steps": all_step_metrics,
        "metrics_path": str(metrics_path),
        "total_elapsed_sec": time.time() - total_start,
    }

    with open(output_dir / "grpo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO training experiments.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data-path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--val-data-path", type=str, default=DEFAULT_VAL_DATA_PATH)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)

    # Training hyperparameters
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--loss-type", type=str, default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--cliprange", type=float, default=0.2)

    # Sampling
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)

    # Normalization
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--use-std-normalization", action="store_true", default=True)
    parser.add_argument("--no-std-normalization", dest="use_std_normalization", action="store_false")

    # Other settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--eval-every-steps", type=int, default=5)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    config = GRPOConfig(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        save_dir=args.save_dir,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        loss_type=args.loss_type,
        cliprange=args.cliprange,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        advantage_eps=args.advantage_eps,
        use_std_normalization=args.use_std_normalization,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_eval_samples=args.max_eval_samples,
        eval_every_steps=args.eval_every_steps,
    )

    summary = run_grpo_experiment(config)
    logger.info(f"GRPO training complete. Summary saved to {Path(config.save_dir) / 'grpo_summary.json'}")


if __name__ == "__main__":
    main()
