"""Evaluate Llama3.18B zero-shot performance on MMLU."""
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.load_mmlu import load_mmlu_split
from tests.adapters import run_parse_mmlu_response


SYSTEM_PROMPT = """Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.

Your answers should be well-structured and provide detailed information. They should also
have an engaging tone.

Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic,
dangerous, or illegal content, even if it may be helpful.

Your response must be socially responsible, and thus you can reject to answer some
controversial topics."""

MMLU_INSTRUCTION_TEMPLATE = """Answer the following multiple choice question about {subject}. Respond with a single
sentence of the form "The correct answer is _", filling the blank with the letter
corresponding to the correct answer (i.e., A, B, C or D).
Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
"""

# Stop generation when we see this to avoid the model starting a new conversation turn
STOP_STR = "# Query:"


def format_prompt(example: dict) -> str:
    """Format full prompt with system prompt + instruction."""
    instruction = MMLU_INSTRUCTION_TEMPLATE.format(
        subject=example["subject"],
        question=example["question"],
        options=example["options"],
    )
    return f"{SYSTEM_PROMPT}\n# Query:\n{instruction}\n# Answer:"


def main():
    # Load model - use Llama3.1-8B from HuggingFace
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, tensor_parallel_size=1)

    # Load data
    print("Loading MMLU test data...")
    examples = load_mmlu_split("test")
    print(f"Loaded {len(examples)} examples")

    # Format prompts
    prompts = [format_prompt(ex) for ex in examples]

    # Generate
    print("Starting generation...")
    start_time = time.time()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, stop=STOP_STR)
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = len(examples) / elapsed
    print(f"Generation done in {elapsed:.2f}s ({throughput:.2f} examples/s)")

    # Parse and evaluate
    correct = 0
    parse_failures = 0
    parse_failure_examples = []
    results = []

    for ex, output in zip(examples, outputs):
        model_output = output.outputs[0].text
        parsed = run_parse_mmlu_response(ex, model_output)

        is_correct = parsed == ex["answer"] if parsed else False
        if parsed is None:
            parse_failures += 1
            if len(parse_failure_examples) < 5:
                parse_failure_examples.append({
                    "question": ex["question"],
                    "model_output": model_output,
                    "gold_answer": ex["answer"],
                })
        else:
            if is_correct:
                correct += 1

        results.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "parsed_answer": parsed,
            "is_correct": is_correct,
            "model_output": model_output,
        })

    # Summary
    total_parsed = len(examples) - parse_failures
    accuracy = correct / total_parsed if total_parsed > 0 else 0

    summary = {
        "total_examples": len(examples),
        "parse_failures": parse_failures,
        "accuracy": accuracy,
        "throughput": throughput,
        "elapsed_seconds": elapsed,
        "parse_failure_examples": parse_failure_examples,
    }

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "mmlu_baseline_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total: {len(examples)}")
    print(f"Parse failures: {parse_failures}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"Results saved to results/mmlu_baseline_results.json")


if __name__ == "__main__":
    main()