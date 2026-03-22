import argparse
import json
import logging
import random
import time

from dataclasses import asdict, dataclass
from pathlib import Path

from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.fine_tuning import DEFAULT_MODEL_PATH, init_vllm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_DATA_PATH = "data/gsm8k/train.formatted.jsonl"
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


def build_sampling_params(config: FeasibilityConfig) -> SamplingParams:
    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        n=config.group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=config.seed,
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


def run_feasibility_experiment(config: FeasibilityConfig) -> tuple[dict[str, object], list[dict[str, object]]]:
    all_examples = load_formatted_examples(config.data_path)
    if len(all_examples) == 0:
        raise ValueError(f"No examples found in {config.data_path}")

    eval_examples = sample_examples(all_examples, config.sample_size, config.seed)
    prompts = [build_r1_zero_prompt(example.question) for example in eval_examples]

    if config.group_size <= 0:
        raise ValueError("group_size must be positive")

    logger.info("Loaded %d examples; evaluating %d prompts with group_size=%d", len(all_examples), len(eval_examples), config.group_size)
    llm = init_vllm(
        model_id=config.model_path,
        device="cuda",
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    sampling_params = build_sampling_params(config)

    # prompt llm
    generation_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_elapsed_sec = time.time() - generation_start

    results: list[dict[str, object]] = []
    num_reward_one = 0
    num_format_ok = 0
    num_questions_with_any_correct = 0

    # Evaluate the generated responses
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feasibility experiment for expert iteration rollouts.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--num-preview-examples", type=int, default=3)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
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
    logger.info("reward==1: %d / %d (%.4f)", summary["num_reward_one"], summary["total_generations"], summary["reward_one_rate"])
    logger.info("format ok: %d / %d (%.4f)", summary["num_format_ok"], summary["total_generations"], summary["format_ok_rate"])
    logger.info(
        "questions with any correct sample: %d / %d (%.4f)",
        summary["num_questions_with_any_correct"],
        summary["num_questions"],
        summary["question_success_rate"],
    )
    logger.info("generation elapsed: %.2fs (%.3fs/question)", summary["generation_elapsed_sec"], summary["seconds_per_question"])
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved samples to %s", results_path)


if __name__ == "__main__":
    main()
