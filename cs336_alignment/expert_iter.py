import argparse
import json
import logging
import math
import random
import subprocess
import time

import torch

from dataclasses import asdict, dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.fine_tuning import (
    DEFAULT_MODEL_PATH,
    get_response_log_probs,
    init_vllm,
    load_model,
    load_policy_into_vllm_instance,
    save_pretrained,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TRAIN_DATA_PATH = "data/gsm8k/train.formatted.jsonl"
DEFAULT_VAL_DATA_PATH = "data/gsm8k/test.formatted.jsonl"
DEFAULT_SAVE_DIR = "results/expert_iter"
R1_ZERO_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)


@dataclass
class FormattedExample:
    question: str
    answer: str
    ground_truth: str


@dataclass
class FeasibilityConfig:
    model_path: str
    data_path: str
    save_dir: str
    sample_size: int
    group_size: int
    temperature: float
    top_p: float
    max_tokens: int
    min_tokens: int
    seed: int
    gpu_memory_utilization: float
    n_ei_steps: int


@dataclass
class EIConfig:
    model_path: str
    train_data_path: str
    val_data_path: str
    save_dir: str
    db_size: int
    group_size: int
    n_ei_steps: int
    sft_epochs_per_step: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    warmup_ratio: float
    weight_decay: float
    max_rollout_tokens: int
    min_rollout_tokens: int
    max_eval_tokens: int
    temperature: float
    top_p: float
    seed: int
    gpu_memory_utilization: float
    max_eval_samples: int


class RolloutDataset(Dataset):
    def __init__(self, examples: list[FormattedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> FormattedExample:
        return self.examples[idx]


def load_formatted_examples(dataset_path: str) -> list[FormattedExample]:
    examples: list[FormattedExample] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            ground_truth = str(row.get("ground truth", "")).strip()
            if not question or not answer or not ground_truth:
                logger.warning("Skipping malformed row %d in %s", line_idx, dataset_path)
                continue
            examples.append(
                FormattedExample(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                )
            )
    return examples


def sample_examples(examples: list[FormattedExample], sample_size: int, seed: int) -> list[FormattedExample]:
    if sample_size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(examples)), k=sample_size))
    return [examples[i] for i in sampled_indices]


def build_r1_zero_prompt(question: str) -> str:
    prompt = question.strip()
    prompt = prompt.removesuffix("<think>").strip()
    return R1_ZERO_PROMPT_TEMPLATE.format(question=prompt)


def build_sampling_params(group_size: int, temperature: float, top_p: float, max_tokens: int, min_tokens: int, seed: int) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )


def estimate_sampling_runtime(seconds_per_question: float, group_size: int, n_ei_steps: int) -> list[dict[str, float | int]]:
    estimates: list[dict[str, float | int]] = []
    for db_size in (512, 1024, 2048):
        total_questions = db_size * n_ei_steps
        estimated_seconds = seconds_per_question * total_questions
        estimates.append(
            {
                "db_size": db_size,
                "group_size": group_size,
                "n_ei_steps": n_ei_steps,
                "estimated_sampling_seconds": estimated_seconds,
                "estimated_sampling_minutes": estimated_seconds / 60.0,
                "estimated_sampling_hours": estimated_seconds / 3600.0,
            }
        )
    return estimates


def _truncate_after_answer_tag(text: str) -> str:
    if "</answer>" in text:
        return text.split("</answer>", 1)[0] + "</answer>"
    return text


def _build_collate_fn(tokenizer):
    def _collate(batch: list[FormattedExample]) -> dict[str, torch.Tensor]:
        prompt_strs = [build_r1_zero_prompt(example.question) for example in batch]
        output_strs = [example.answer for example in batch]
        return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    return _collate


def run_feasibility_experiment(config: FeasibilityConfig) -> tuple[dict[str, object], list[dict[str, object]]]:
    all_examples = load_formatted_examples(config.data_path)
    if len(all_examples) == 0:
        raise ValueError(f"No examples found in {config.data_path}")

    eval_examples = sample_examples(all_examples, config.sample_size, config.seed)
    prompts = [build_r1_zero_prompt(example.question) for example in eval_examples]

    logger.info(
        "Loaded %d examples; evaluating %d prompts with group_size=%d",
        len(all_examples),
        len(eval_examples),
        config.group_size,
    )
    llm = init_vllm(
        model_id=config.model_path,
        device="cuda",
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    sampling_params = build_sampling_params(
        group_size=config.group_size,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        seed=config.seed,
    )

    generation_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_elapsed_sec = time.time() - generation_start

    results: list[dict[str, object]] = []
    num_reward_one = 0
    num_format_ok = 0
    num_questions_with_any_correct = 0

    for example, prompt, output in zip(eval_examples, prompts, outputs):
        question_has_correct = False
        for sample_idx, candidate in enumerate(output.outputs):
            response = candidate.text
            reward_result = r1_zero_reward_fn(response, example.ground_truth)
            is_correct = float(reward_result["reward"]) == 1.0
            has_valid_format = float(reward_result["format_reward"]) == 1.0
            num_reward_one += int(is_correct)
            num_format_ok += int(has_valid_format)
            question_has_correct = question_has_correct or is_correct
            results.append(
                {
                    "question": example.question,
                    "prompt": prompt,
                    "reference_answer": example.answer,
                    "ground_truth": example.ground_truth,
                    "sample_idx": sample_idx,
                    "response": response,
                    "reward_result": reward_result,
                    "is_correct": is_correct,
                    "has_valid_format": has_valid_format,
                }
            )
        num_questions_with_any_correct += int(question_has_correct)

    total_generations = len(results)
    total_questions = len(eval_examples)
    seconds_per_question = generation_elapsed_sec / max(total_questions, 1)
    seconds_per_generation = generation_elapsed_sec / max(total_generations, 1)

    summary = {
        "config": asdict(config),
        "dataset_size": len(all_examples),
        "num_questions": total_questions,
        "group_size": config.group_size,
        "total_generations": total_generations,
        "num_reward_one": num_reward_one,
        "reward_one_rate": 0.0 if total_generations == 0 else num_reward_one / total_generations,
        "num_format_ok": num_format_ok,
        "format_ok_rate": 0.0 if total_generations == 0 else num_format_ok / total_generations,
        "num_questions_with_any_correct": num_questions_with_any_correct,
        "question_success_rate": 0.0 if total_questions == 0 else num_questions_with_any_correct / total_questions,
        "generation_elapsed_sec": generation_elapsed_sec,
        "seconds_per_question": seconds_per_question,
        "seconds_per_generation": seconds_per_generation,
        "sampling_runtime_estimates": estimate_sampling_runtime(
            seconds_per_question=seconds_per_question,
            group_size=config.group_size,
            n_ei_steps=config.n_ei_steps,
        ),
    }
    return summary, results


def select_example_rows(results: list[dict[str, object]], predicate: str, limit: int) -> list[dict[str, object]]:
    if predicate == "correct":
        filtered = [row for row in results if row["is_correct"]]
    elif predicate == "formatted_wrong":
        filtered = [row for row in results if row["has_valid_format"] and not row["is_correct"]]
    else:
        filtered = [row for row in results if not row["has_valid_format"]]
    return filtered[:limit]


def save_results(summary: dict[str, object], results: list[dict[str, object]], save_dir: str) -> tuple[str, str]:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"feasibility_{timestamp}_summary.json"
    results_path = output_dir / f"feasibility_{timestamp}_samples.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    return str(summary_path), str(results_path)


def query_nvidia_smi() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "utilization_gpu_pct": int(parts[3]),
            }
        )
    return {"available": True, "gpus": rows}


def run_vllm_diagnostic(
    model_path: str,
    data_path: str,
    sample_size: int,
    group_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    min_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
) -> dict[str, object]:
    examples = load_formatted_examples(data_path)
    sampled_examples = sample_examples(examples, sample_size, seed)
    prompts = [build_r1_zero_prompt(example.question) for example in sampled_examples]

    diagnostic: dict[str, object] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_count": torch.cuda.device_count(),
        "gpu_before_init": query_nvidia_smi(),
        "num_prompts": len(prompts),
        "group_size": group_size,
    }
    if torch.cuda.is_available():
        diagnostic["torch_device_name"] = torch.cuda.get_device_name(0)

    init_start = time.time()
    llm = init_vllm(
        model_id=model_path,
        device="cuda",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    diagnostic["init_elapsed_sec"] = time.time() - init_start
    diagnostic["gpu_after_init"] = query_nvidia_smi()

    sampling_params = build_sampling_params(
        group_size=group_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        seed=seed,
    )
    generate_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generate_elapsed_sec = time.time() - generate_start
    diagnostic["generate_elapsed_sec"] = generate_elapsed_sec
    diagnostic["gpu_after_generate"] = query_nvidia_smi()
    diagnostic["total_outputs"] = sum(len(output.outputs) for output in outputs)
    diagnostic["seconds_per_prompt"] = generate_elapsed_sec / max(len(prompts), 1)
    diagnostic["seconds_per_output"] = generate_elapsed_sec / max(diagnostic["total_outputs"], 1)
    diagnostic["sample_response_preview"] = outputs[0].outputs[0].text[:400] if outputs and outputs[0].outputs else ""
    return diagnostic


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    masked_total = (values * mask).sum().item()
    denom = mask.sum().item()
    if denom <= 0:
        return 0.0
    return masked_total / denom


@torch.no_grad()
def estimate_mean_response_entropy(
    model,
    tokenizer,
    examples: list[FormattedExample],
    device: torch.device,
    batch_size: int = 2,
    max_examples: int = 16,
) -> float | None:
    if len(examples) == 0:
        return None

    entropy_values: list[float] = []
    subset = examples[:max_examples]
    collate_fn = _build_collate_fn(tokenizer)
    model.eval()

    for start_idx in range(0, len(subset), batch_size):
        batch_examples = subset[start_idx:start_idx + batch_size]
        batch = collate_fn(batch_examples)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device).float()
        output = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=True,
        )
        entropy_values.append(_masked_mean(output["token_entropy"], response_mask))

    model.train()
    if not entropy_values:
        return None
    return sum(entropy_values) / len(entropy_values)


def collect_rollouts_for_step(
    llm,
    policy_model,
    step_examples: list[FormattedExample],
    config: EIConfig,
    step_idx: int,
) -> tuple[list[FormattedExample], dict[str, object], list[dict[str, object]]]:
    load_policy_into_vllm_instance(policy_model, llm)
    prompts = [build_r1_zero_prompt(example.question) for example in step_examples]
    sampling_params = build_sampling_params(
        group_size=config.group_size,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_rollout_tokens,
        min_tokens=config.min_rollout_tokens,
        seed=config.seed + step_idx,
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_elapsed_sec = time.time() - start_time

    kept_examples: list[FormattedExample] = []
    sample_rows: list[dict[str, object]] = []
    total_generations = 0
    num_reward_one = 0
    num_format_ok = 0
    num_questions_with_any_correct = 0

    for example, prompt, output in zip(step_examples, prompts, outputs):
        question_has_correct = False
        for sample_idx, candidate in enumerate(output.outputs):
            total_generations += 1
            response = candidate.text
            reward_result = r1_zero_reward_fn(response, example.ground_truth)
            is_correct = float(reward_result["reward"]) == 1.0
            has_valid_format = float(reward_result["format_reward"]) == 1.0
            num_reward_one += int(is_correct)
            num_format_ok += int(has_valid_format)
            question_has_correct = question_has_correct or is_correct
            if is_correct:
                kept_examples.append(
                    FormattedExample(
                        question=example.question,
                        answer=response,
                        ground_truth=example.ground_truth,
                    )
                )
            sample_rows.append(
                {
                    "step": step_idx,
                    "question": example.question,
                    "prompt": prompt,
                    "ground_truth": example.ground_truth,
                    "sample_idx": sample_idx,
                    "response": response,
                    "reward_result": reward_result,
                    "is_correct": is_correct,
                    "has_valid_format": has_valid_format,
                }
            )
        num_questions_with_any_correct += int(question_has_correct)

    summary = {
        "step": step_idx,
        "db_size": len(step_examples),
        "group_size": config.group_size,
        "total_generations": total_generations,
        "num_correct_rollouts": len(kept_examples),
        "reward_one_rate": 0.0 if total_generations == 0 else num_reward_one / total_generations,
        "format_ok_rate": 0.0 if total_generations == 0 else num_format_ok / total_generations,
        "question_success_rate": 0.0 if len(step_examples) == 0 else num_questions_with_any_correct / len(step_examples),
        "generation_elapsed_sec": generation_elapsed_sec,
        "seconds_per_question": generation_elapsed_sec / max(len(step_examples), 1),
    }
    return kept_examples, summary, sample_rows


@torch.no_grad()
def evaluate_policy_with_vllm(
    llm,
    policy_model,
    eval_examples: list[FormattedExample],
    max_new_tokens: int,
    seed: int,
) -> tuple[float, float]:
    load_policy_into_vllm_instance(policy_model, llm)
    prompts = [build_r1_zero_prompt(example.question) for example in eval_examples]
    sampling_params = build_sampling_params(
        group_size=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        min_tokens=1,
        seed=seed,
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_sec = time.time() - start_time

    total = 0
    correct = 0
    for example, output in zip(eval_examples, outputs):
        if not output.outputs:
            continue
        response = output.outputs[0].text
        reward = r1_zero_reward_fn(response, example.ground_truth)
        correct += int(float(reward["reward"]) == 1.0)
        total += 1

    accuracy = 0.0 if total == 0 else correct / total
    return accuracy, elapsed_sec


@torch.no_grad()
def evaluate_policy(
    model,
    tokenizer,
    eval_examples: list[FormattedExample],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0

    for start_idx in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[start_idx:start_idx + batch_size]
        prompts = [build_r1_zero_prompt(example.question) for example in batch_examples]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_lengths = encoded["attention_mask"].sum(dim=1)

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        for i, example in enumerate(batch_examples):
            start = int(prompt_lengths[i].item())
            response = tokenizer.decode(generated[i, start:], skip_special_tokens=True)
            response = _truncate_after_answer_tag(response)
            reward = r1_zero_reward_fn(response, example.ground_truth)
            correct += int(float(reward["reward"]) == 1.0)
            total += 1

    model.train()
    return 0.0 if total == 0 else correct / total


def train_policy_on_rollouts(
    model,
    tokenizer,
    train_examples: list[FormattedExample],
    config: EIConfig,
    step_idx: int,
    device: torch.device,
) -> dict[str, object]:
    if len(train_examples) == 0:
        return {
            "step": step_idx,
            "num_sft_examples": 0,
            "optimizer_steps": 0,
            "mean_loss": None,
            "mean_entropy": None,
            "train_elapsed_sec": 0.0,
        }

    train_dataset = RolloutDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_build_collate_fn(tokenizer),
    )

    num_batches = len(train_loader)
    optimizer_steps = math.ceil((num_batches * config.sft_epochs_per_step) / config.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_steps = int(optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(optimizer_steps, 1),
    )

    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)

    losses: list[float] = []
    start_time = time.time()
    microbatch_idx = 0

    progress_bar = tqdm(
        range(config.sft_epochs_per_step),
        desc=f"EI step {step_idx} SFT",
        leave=False,
    )
    for _ in progress_bar:
        for batch in train_loader:
            microbatch_idx += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device).float()

            output = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            token_log_probs = output["log_probs"]
            response_token_count = response_mask.sum().clamp_min(1.0)
            normalize_constant = response_token_count / max(response_mask.shape[0], 1)

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=token_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                normalize_constant=normalize_constant,
            )
            losses.append(float(loss.item()))

            should_step = (
                microbatch_idx % config.gradient_accumulation_steps == 0
                or microbatch_idx == num_batches * config.sft_epochs_per_step
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

    train_elapsed_sec = time.time() - start_time
    torch.cuda.empty_cache()
    mean_entropy = estimate_mean_response_entropy(
        model=model,
        tokenizer=tokenizer,
        examples=train_examples,
        device=device,
    )
    model.config.use_cache = True
    return {
        "step": step_idx,
        "num_sft_examples": len(train_examples),
        "optimizer_steps": optimizer_steps,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "mean_entropy": mean_entropy,
        "train_elapsed_sec": train_elapsed_sec,
    }


def run_expert_iteration(config: EIConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    train_examples = load_formatted_examples(config.train_data_path)
    val_examples = load_formatted_examples(config.val_data_path)
    if len(train_examples) == 0 or len(val_examples) == 0:
        raise ValueError("Train or validation dataset is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path=config.model_path)
    model.to(device)
    model.train()

    llm = init_vllm(
        model_id=config.model_path,
        device="cuda",
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )

    output_dir = Path(config.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "ei_metrics.jsonl"
    rollouts_path = output_dir / "ei_rollouts.jsonl"

    all_step_metrics: list[dict[str, object]] = []
    train_rng = random.Random(config.seed)
    total_start = time.time()
    clipped_val_examples = val_examples[:config.max_eval_samples] if config.max_eval_samples > 0 else val_examples

    initial_val_accuracy, initial_val_elapsed_sec = evaluate_policy_with_vllm(
        llm=llm,
        policy_model=model,
        eval_examples=clipped_val_examples,
        max_new_tokens=config.max_eval_tokens,
        seed=config.seed,
    )
    logger.info(
        "Initial validation reward@1 accuracy: %.4f (%.2fs)",
        initial_val_accuracy,
        initial_val_elapsed_sec,
    )

    for step_idx in range(1, config.n_ei_steps + 1):
        if config.db_size >= len(train_examples):
            step_examples = list(train_examples)
        else:
            indices = train_rng.sample(range(len(train_examples)), k=config.db_size)
            step_examples = [train_examples[i] for i in indices]

        rollout_examples, rollout_summary, sample_rows = collect_rollouts_for_step(
            llm=llm,
            policy_model=model,
            step_examples=step_examples,
            config=config,
            step_idx=step_idx,
        )
        with open(rollouts_path, "a", encoding="utf-8") as f:
            for row in sample_rows:
                f.write(json.dumps(row) + "\n")

        logger.info(
            "[step %d] rollout correct=%d/%d question_success=%.4f",
            step_idx,
            rollout_summary["num_correct_rollouts"],
            rollout_summary["total_generations"],
            rollout_summary["question_success_rate"],
        )

        torch.cuda.empty_cache()
        train_summary = train_policy_on_rollouts(
            model=model,
            tokenizer=tokenizer,
            train_examples=rollout_examples,
            config=config,
            step_idx=step_idx,
            device=device,
        )

        torch.cuda.empty_cache()
        val_accuracy, val_elapsed_sec = evaluate_policy_with_vllm(
            llm=llm,
            policy_model=model,
            eval_examples=clipped_val_examples,
            max_new_tokens=config.max_eval_tokens,
            seed=config.seed + 10_000 + step_idx,
        )

        step_metrics = {
            "step": step_idx,
            "rollout": rollout_summary,
            "train": train_summary,
            "val_reward_accuracy": val_accuracy,
            "val_elapsed_sec": val_elapsed_sec,
            "elapsed_sec": time.time() - total_start,
        }
        all_step_metrics.append(step_metrics)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_metrics) + "\n")

        logger.info(
            "[step %d] val_reward_accuracy=%.4f val_elapsed=%.2fs num_sft_examples=%d mean_entropy=%s",
            step_idx,
            val_accuracy,
            val_elapsed_sec,
            train_summary["num_sft_examples"],
            "n/a" if train_summary["mean_entropy"] is None else f"{train_summary['mean_entropy']:.4f}",
        )

        step_model_dir = output_dir / f"step_{step_idx}"
        save_pretrained(str(step_model_dir), model, tokenizer)

    summary = {
        "config": asdict(config),
        "initial_val_reward_accuracy": initial_val_accuracy,
        "initial_val_elapsed_sec": initial_val_elapsed_sec,
        "steps": all_step_metrics,
        "metrics_path": str(metrics_path),
        "rollouts_path": str(rollouts_path),
        "total_elapsed_sec": time.time() - total_start,
    }
    with open(output_dir / "ei_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expert iteration experiments.")
    parser.add_argument("--mode", type=str, default="feasibility", choices=["feasibility", "minimal_ei", "vllm_diagnostic"])
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--train-data-path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--val-data-path", type=str, default=DEFAULT_VAL_DATA_PATH)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)

    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--db-size", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-rollout-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--min-rollout-tokens", type=int, default=4)
    parser.add_argument("--max-eval-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--num-preview-examples", type=int, default=3)

    parser.add_argument("--sft-epochs-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    if args.mode == "feasibility":
        config = FeasibilityConfig(
            model_path=args.model_path,
            data_path=args.data_path,
            save_dir=args.save_dir,
            sample_size=args.sample_size,
            group_size=args.group_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            n_ei_steps=args.n_ei_steps,
        )

        total_start = time.time()
        summary, results = run_feasibility_experiment(config)
        summary["total_elapsed_sec"] = time.time() - total_start
        preview_limit = max(args.num_preview_examples, 0)
        summary["preview_examples"] = {
            "correct": select_example_rows(results, "correct", preview_limit),
            "formatted_wrong": select_example_rows(results, "formatted_wrong", preview_limit),
            "unformatted": select_example_rows(results, "unformatted", preview_limit),
        }

        summary_path, results_path = save_results(summary, results, args.save_dir)
        logger.info(
            "reward==1: %d / %d (%.4f)",
            summary["num_reward_one"],
            summary["total_generations"],
            summary["reward_one_rate"],
        )
        logger.info(
            "format ok: %d / %d (%.4f)",
            summary["num_format_ok"],
            summary["total_generations"],
            summary["format_ok_rate"],
        )
        logger.info(
            "questions with any correct sample: %d / %d (%.4f)",
            summary["num_questions_with_any_correct"],
            summary["num_questions"],
            summary["question_success_rate"],
        )
        logger.info(
            "generation elapsed: %.2fs (%.3fs/question)",
            summary["generation_elapsed_sec"],
            summary["seconds_per_question"],
        )
        logger.info("Saved summary to %s", summary_path)
        logger.info("Saved samples to %s", results_path)
        return

    if args.mode == "vllm_diagnostic":
        diagnostic = run_vllm_diagnostic(
            model_path=args.model_path,
            data_path=args.data_path,
            sample_size=args.sample_size,
            group_size=args.group_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(json.dumps(diagnostic, indent=2))
        return

    config = EIConfig(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        save_dir=args.save_dir,
        db_size=args.db_size,
        group_size=args.group_size,
        n_ei_steps=args.n_ei_steps,
        sft_epochs_per_step=args.sft_epochs_per_step,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_rollout_tokens=args.max_rollout_tokens,
        min_rollout_tokens=args.min_rollout_tokens,
        max_eval_tokens=args.max_eval_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_eval_samples=args.max_eval_samples,
    )
    summary = run_expert_iteration(config)
    logger.info("Finished EI run. Summary saved to %s", Path(config.save_dir) / "ei_summary.json")
    if summary["steps"]:
        logger.info(
            "Final validation reward accuracy: %.4f",
            summary["steps"][-1]["val_reward_accuracy"],
        )


if __name__ == "__main__":
    main()
