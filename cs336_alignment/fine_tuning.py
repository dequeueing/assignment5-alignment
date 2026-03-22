import argparse
import json
import logging
import math
import random
import re
import time

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from typing import List
from pathlib import Path
from tqdm.auto import tqdm
from unittest.mock import patch
from dataclasses import asdict, dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import get_linear_schedule_with_warmup

from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA_PATH = "data/gsm8k/train.formatted.jsonl"
DEFAULT_VAL_DATA_PATH = "data/gsm8k/test.formatted.jsonl"


@dataclass
class SFTExample:
    question: str
    answer: str
    ground_truth: str
    has_correct_answer: bool


@dataclass
class SFTConfig:
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    eval_every_steps: int
    eval_batch_size: int
    warmup_ratio: float
    weight_decay: float
    max_new_tokens: int
    max_eval_samples: int
    seed: int
    save_dir: str


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen2 with FlashAttention requires left padding for batched generation.
    tokenizer.padding_side = "left"
    return model, tokenizer

def forward_once(train_batch, model):
    device = 'cuda'
    input_ids = train_batch["input_ids"].to(device)
    labels = train_batch["labels"].to(device)

    logits = model(input_ids).logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss



def save_pretrained(output_dir, model, tokenizer):
    # Save the model weights
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


def gradient_accumulation(model, data_loader, loss_fn, optimizer):
    gradient_accumulation_steps = 4
    for idx, (inputs, labels) in enumerate(data_loader):
        logits = model(inputs)

        loss = loss_fn(logits, labels) / gradient_accumulation_steps
        loss.backward()

        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            


def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer):
    """
    Tokenize prompt and output separately, concat together,
    and construct response mask.
    """
    assert len(prompt_strs) == len(output_strs), "input lengths do not match"
    
    
    input_ids_cache = []
    labels_cache = []
    max_len = 0
    batch_size = len(prompt_strs)

    for idx, (prompt, response) in enumerate(zip(prompt_strs, output_strs)):
        logger.debug(f"{idx} {prompt} {response}")

        prompt_tokenized   = tokenizer(prompt, return_tensors="pt")
        response_tokenized = tokenizer(response, return_tensors="pt")
        
        logger.debug(f"prompt_tokenized result:\n{prompt_tokenized}")
        logger.debug(f"response_tokenized result:\n{response_tokenized}")
        
        prompt_ids    = prompt_tokenized['input_ids'].squeeze(0)
        response_ids  = response_tokenized['input_ids'].squeeze(0)
        prompt_mask   = prompt_tokenized['attention_mask'].squeeze(0)
        response_mask = response_tokenized['attention_mask'].squeeze(0)
        
        prompt_len   = prompt_ids.shape[0]
        response_len = response_ids.shape[0]
        total_len = prompt_len + response_len

        ids_concat = torch.concat([prompt_ids, response_ids])
        logger.debug(f"After concat, the ids look like:\n{ids_concat}")

        input_ids_cache.append(ids_concat)
        labels_cache.append((prompt_len, response_len))

        max_len = max(max_len, total_len)

    pad_id = tokenizer.pad_token_id
    input_ids_batch = torch.full((batch_size, max_len), pad_id)
    labels_batch    = torch.full((batch_size, max_len), pad_id)
    mask_batch      = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i in range(batch_size):
        ids_concat = input_ids_cache[i]
        input_ids_batch[i, :len(ids_concat)] = ids_concat
        labels_batch[i, :len(ids_concat)] = ids_concat

        prompt_len, response_len = labels_cache[i]
        mask_batch[i, prompt_len - 1:prompt_len + response_len - 1] = True
        
    logger.debug(f"\nFinal result (before shift):")
    logger.debug(f"input_ids_batch:\n{input_ids_batch}")
    logger.debug(f"labels_batch:\n{labels_batch}")
    logger.debug(f"mask_batch:\n{mask_batch}")
    
    result = {
        "input_ids": input_ids_batch[:, :-1],
        "labels": labels_batch[:, 1:],
        "response_mask": mask_batch[:, :-1],
    }
    return result

def compute_entropy(logits:torch.Tensor) -> torch.Tensor:
    """
    logits has the shape: (batch_size,sequence_length,vocab_size)
    """
    log_p = torch.log_softmax(logits, dim=-1)
    entropy = -(log_p.exp() * log_p).sum(dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Shape of tensors:
    1. input_ids: (batch_size, seq_len)
    2. labels: (batch_size, seq_len)
    """
    model_outputs = model(input_ids=input_ids)
    logits = model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs[0]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {
        "log_probs": token_log_probs,
    }
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits=logits)
    return result


def masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    masked_tensor = tensor * mask
    
    if dim is None:
        total_sum = torch.sum(masked_tensor)
    else:
        total_sum = torch.sum(masked_tensor, dim=int(dim))
        
    return total_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_log_probs: per-token log-probabilities
    response_mask: 1 for response tokens, and 0 for prompt/padding
    gradient_accumulation_steps: number of microbatchs per optimizer step
    """
    masked_log_prob_sum = masked_normalize(
        policy_log_probs,
        response_mask,
        dim=None,
        normalize_constant=normalize_constant,
    )
    batch_size = policy_log_probs.shape[0]
    loss = -masked_log_prob_sum / (gradient_accumulation_steps * batch_size)
    loss.backward()
    metadata = {
        "masked_log_prob_sum": masked_log_prob_sum.detach(),
    }
    return loss.detach(), metadata




def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


class SFTJsonlDataset(Dataset):
    def __init__(self, examples: list[SFTExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self.examples[idx]


def _build_prompt(question: str) -> str:
    return question.strip()


def _extract_answer_span(text: str) -> str:
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def _normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _is_correct_reasoning_example(reasoning_answer: str, ground_truth: str) -> bool:
    predicted = _normalize_answer(_extract_answer_span(reasoning_answer))
    target = _normalize_answer(ground_truth)
    return predicted == target and predicted != ""


def _read_jsonl_examples(dataset_path: str, only_correct_reasoning: bool = False) -> list[SFTExample]:
    examples: list[SFTExample] = []
    unique_keys: set[tuple[str, str]] = set()

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue

            ground_truth = str(row.get("ground truth", _extract_answer_span(answer))).strip()
            has_correct_answer = _is_correct_reasoning_example(answer, ground_truth)
            if only_correct_reasoning and not has_correct_answer:
                continue

            key = (question, answer)
            if key in unique_keys:
                continue
            unique_keys.add(key)
            examples.append(
                SFTExample(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                    has_correct_answer=has_correct_answer,
                )
            )

    return examples


def _sample_examples(examples: list[SFTExample], subset_size: int | None, seed: int) -> list[SFTExample]:
    if subset_size is None or subset_size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(examples)), k=subset_size)
    sampled_indices.sort()
    return [examples[i] for i in sampled_indices]


def _build_collate_fn(tokenizer):
    def _collate(batch: list[SFTExample]) -> dict[str, torch.Tensor]:
        prompt_strs = [_build_prompt(example.question) for example in batch]
        output_strs = [example.answer for example in batch]
        tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
        return tokenized

    return _collate


@torch.no_grad()
def evaluate_exact_match_accuracy(
    model: PreTrainedModel,
    tokenizer,
    eval_examples: list[SFTExample],
    eval_batch_size: int,
    max_new_tokens: int,
    max_eval_samples: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    clipped_examples = eval_examples[:max_eval_samples] if max_eval_samples > 0 else eval_examples

    for start_idx in range(0, len(clipped_examples), eval_batch_size):
        batch_examples = clipped_examples[start_idx:start_idx + eval_batch_size]
        prompts = [_build_prompt(example.question) for example in batch_examples]
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
            generated_response_ids = generated[i, start:]
            generated_response = tokenizer.decode(generated_response_ids, skip_special_tokens=True)
            predicted = _normalize_answer(_extract_answer_span(generated_response))
            target = _normalize_answer(example.ground_truth)
            correct += int(predicted == target and predicted != "")
            total += 1

    return 0.0 if total == 0 else correct / total


def run_sft_experiment(
    train_examples: list[SFTExample],
    val_examples: list[SFTExample],
    model_path: str,
    config: SFTConfig,
    experiment_name: str,
    filtered_correct_only: bool = False,
) -> dict[str, object]:
    if len(train_examples) == 0:
        raise ValueError("No training examples found.")
    if len(val_examples) == 0:
        raise ValueError("No validation examples found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    model, tokenizer = load_model(model_path=model_path)
    model.to(device)

    train_dataset = SFTJsonlDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_build_collate_fn(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_steps = int(config.max_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.max_steps,
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{experiment_name}_metrics.jsonl"

    metrics_history: list[dict[str, object]] = []
    train_iter = iter(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    progress_bar = tqdm(range(1, config.max_steps + 1), desc=f"SFT {experiment_name}")

    for step in progress_bar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device).float()

        policy_output = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )
        token_log_probs = policy_output["log_probs"]
        response_token_count = response_mask.sum().clamp_min(1.0)
        normalize_constant = response_token_count / max(response_mask.shape[0], 1)
        loss, _ = sft_microbatch_train_step(
            policy_log_probs=token_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            normalize_constant=normalize_constant,
        )

        if step % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(loss=float(loss.item()), lr=float(lr))

        if step % config.eval_every_steps == 0 or step == config.max_steps:
            val_accuracy = evaluate_exact_match_accuracy(
                model=model,
                tokenizer=tokenizer,
                eval_examples=val_examples,
                eval_batch_size=config.eval_batch_size,
                max_new_tokens=config.max_new_tokens,
                max_eval_samples=config.max_eval_samples,
                device=device,
            )
            elapsed_sec = time.time() - start_time
            metrics = {
                "experiment_name": experiment_name,
                "step": step,
                "train_loss": float(loss.item()),
                "learning_rate": float(lr),
                "val_exact_match": float(val_accuracy),
                "elapsed_sec": float(elapsed_sec),
                "train_size": len(train_examples),
                "filtered_correct_only": filtered_correct_only,
            }
            metrics_history.append(metrics)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            logger.info("[%s] step=%d loss=%.5f val_em=%.4f lr=%.2e", experiment_name, step, loss.item(), val_accuracy, lr)
            model.train()

    model_out = save_dir / experiment_name
    save_pretrained(str(model_out), model, tokenizer)

    result = {
        "experiment_name": experiment_name,
        "model_output_dir": str(model_out),
        "metrics_log": str(log_path),
        "history": metrics_history,
        "config": asdict(config),
        "train_size": len(train_examples),
        "filtered_correct_only": filtered_correct_only,
    }
    with open(save_dir / f"{experiment_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def parse_sizes_arg(sizes: str, full_size: int) -> list[int | None]:
    values: list[int | None] = []
    seen: set[int | None] = set()

    for part in sizes.split(","):
        token = part.strip().lower()
        if token == "full":
            value = None
        else:
            value = int(token)
            if value <= 0:
                continue
            value = min(value, full_size)

        if value not in seen:
            seen.add(value)
            values.append(value)

    if None not in seen:
        values.append(None)
    return values


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT experiments for reasoning dataset.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data-path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--val-data-path", type=str, default=DEFAULT_VAL_DATA_PATH)
    parser.add_argument("--save-dir", type=str, default="results/sft")

    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=256)

    parser.add_argument(
        "--subset-sizes",
        type=str,
        default="128,256,512,1024,full",
        help="Comma-separated subset sizes, include 'full' for full dataset.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="size_sweep",
        choices=["single", "size_sweep", "tune_full", "filtered_full"],
    )
    parser.add_argument("--single-subset-size", type=int, default=1024)
    parser.add_argument("--only-correct-reasoning", action="store_true")

    parser.add_argument("--lr-candidates", type=str, default="1e-5,2e-5,5e-5")
    parser.add_argument("--batch-candidates", type=str, default="4,8")
    parser.add_argument("--target-val-accuracy", type=float, default=0.15)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    # For user-formatted datasets where ground truth is derived from answer,
    # filtered_full becomes a no-op and should use the full dataset directly.
    if args.mode == "filtered_full":
        all_train_examples = _read_jsonl_examples(
            dataset_path=args.train_data_path,
            only_correct_reasoning=False,
        )
        logger.info(
            "mode=filtered_full: skip pre-filtering because ground truth is derived from answer in this dataset."
        )
    else:
        all_train_examples = _read_jsonl_examples(
            dataset_path=args.train_data_path,
            only_correct_reasoning=args.only_correct_reasoning,
        )
    val_examples = _read_jsonl_examples(dataset_path=args.val_data_path, only_correct_reasoning=False)

    logger.info("Loaded train examples: %d", len(all_train_examples))
    logger.info("Loaded validation examples: %d", len(val_examples))

    base_config = SFTConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        eval_every_steps=args.eval_every_steps,
        eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_new_tokens=args.max_new_tokens,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    results: list[dict[str, object]] = []

    if args.mode == "single":
        train_examples = _sample_examples(all_train_examples, args.single_subset_size, args.seed)
        exp_name = f"single_n{len(train_examples)}_lr{args.learning_rate}_bs{args.batch_size}"
        results.append(run_sft_experiment(
            train_examples,
            val_examples,
            args.model_path,
            base_config,
            exp_name,
            filtered_correct_only=args.only_correct_reasoning,
        ))

    elif args.mode == "size_sweep":
        size_values = parse_sizes_arg(args.subset_sizes, len(all_train_examples))
        for subset_size in size_values:
            train_examples = _sample_examples(all_train_examples, subset_size, args.seed)
            size_tag = "full" if subset_size is None else str(subset_size)
            exp_name = f"size_{size_tag}_lr{args.learning_rate}_bs{args.batch_size}"
            results.append(run_sft_experiment(
                train_examples,
                val_examples,
                args.model_path,
                base_config,
                exp_name,
                filtered_correct_only=args.only_correct_reasoning,
            ))

    elif args.mode == "filtered_full":
        exp_name = f"filtered_full_lr{args.learning_rate}_bs{args.batch_size}"
        results.append(run_sft_experiment(
            all_train_examples,
            val_examples,
            args.model_path,
            base_config,
            exp_name,
            filtered_correct_only=True,
        ))

    else:
        lr_candidates = parse_float_list(args.lr_candidates)
        batch_candidates = parse_int_list(args.batch_candidates)
        full_train = list(all_train_examples)
        best_result = None
        best_acc = -math.inf

        for lr in lr_candidates:
            for batch_size in batch_candidates:
                current_config = SFTConfig(
                    learning_rate=lr,
                    batch_size=batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_steps=args.max_steps,
                    eval_every_steps=args.eval_every_steps,
                    eval_batch_size=args.eval_batch_size,
                    warmup_ratio=args.warmup_ratio,
                    weight_decay=args.weight_decay,
                    max_new_tokens=args.max_new_tokens,
                    max_eval_samples=args.max_eval_samples,
                    seed=args.seed,
                    save_dir=args.save_dir,
                )
                exp_name = f"tune_full_lr{lr}_bs{batch_size}"
                result = run_sft_experiment(
                    full_train,
                    val_examples,
                    args.model_path,
                    current_config,
                    exp_name,
                    filtered_correct_only=args.only_correct_reasoning,
                )
                final_acc = result["history"][-1]["val_exact_match"] if result["history"] else 0.0
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_result = result
                results.append(result)
                if final_acc >= args.target_val_accuracy:
                    logger.info("Reached target val accuracy %.4f with lr=%s batch=%s", final_acc, lr, batch_size)

        if best_result is not None:
            logger.info(
                "Best full-data tuning result: exp=%s val_em=%.4f",
                best_result["experiment_name"],
                best_acc,
            )

    summary_path = Path(args.save_dir) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved run summary to %s", summary_path)


if __name__ == '__main__':
    main()