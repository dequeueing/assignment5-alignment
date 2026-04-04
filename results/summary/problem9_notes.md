# Problem 9: Prompt Ablation (grpo_prompt_ablation)

## Setup
- **Comparison**: r1_zero_reward_fn vs question_only_reward_fn
- **Base config**: LR=5e-6, grpo_clip, epochs=4, cliprange=0.2 (best from P7-P8)
- **Eval**: Always uses r1_zero_reward_fn (requires `</think> <answer>` format)

## Results

| Metric                    | r1_zero (format+answer) | question_only (answer only) |
|--------------------------|-------------------------|-----------------------------|
| Steps run                | 82                      | 52                          |
| Peak val_accuracy (r1)   | **71.48%** @ step 80    | 0.39% @ step 10             |
| Peak rollout_reward      | 57.42%                  | **87.50%**                  |
| Format compliance (val)  | ~95% by step 80         | **0%** throughout           |
| Response length           | 200–300 tokens          | 135 → 259 tokens (growing) |

## Key Findings

### 1. Format Reward is Critical for Structured Reasoning
The `question_only` reward function does NOT require `</think> <answer>` format.
Without this format constraint:
- The model **never learns** the structured thinking format (0% format compliance)
- Val accuracy stays at 0% because evaluation requires the format
- The model "hacks" the reward by producing unstructured answers

### 2. High Training Reward ≠ High Eval Accuracy
- `question_only` achieves **87.5% rollout reward** (much higher than r1_zero's 57%)
- But val_accuracy is **0%** vs r1_zero's **71.5%**
- The model learns to produce correct answers in unstructured format
- This is a form of **reward hacking**: optimizing training reward without useful behavior

### 3. The Format Serves Multiple Purposes
- **Chain-of-thought**: Forces the model to show reasoning steps in `<think>` block
- **Answer extraction**: `<answer>` tags enable reliable answer parsing
- **Evaluation compatibility**: Without format, even correct answers can't be evaluated
- **Generalization**: Structured reasoning likely transfers better to new problems

### 4. Response Length Behavior
- r1_zero: Responses grow as model learns to think through problems
- question_only: Responses also grow (135→259 tokens) as model produces verbose unstructured text
- Without format constraint, nothing prevents degenerate verbosity

## Conclusion
**The structured prompt (`r1_zero`) is essential.** Without format-aware reward:
1. Model solves math but can't communicate answers reliably
2. Training reward is misleadingly high
3. No chain-of-thought reasoning emerges spontaneously
4. The gap is enormous: 71.5% vs 0.4% on format-requiring evaluation

This demonstrates that **reward shaping** (requiring both format AND correctness) is
crucial for teaching models structured reasoning, not just correct answers.
