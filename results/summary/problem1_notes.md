# Problem 1: grpo_learning_rate

## Results Summary

| LR   | Steps Run | Initial Acc | Best Acc | Best @ Step | Final Acc | Status |
|------|-----------|-------------|----------|-------------|-----------|--------|
| 5e-6 | 176/200   | 17.97%      | 16.80%   | 70          | 12.11%    | Stable decline |
| 1e-5 | 90/200    | 17.97%      | 12.11%   | 10          | 5.47%     | Significant decline |
| 5e-5 | 30/200    | 17.97%      | 0.00%    | -           | 0.00%     | Complete collapse |

## Key Findings

1. **Best LR: 5e-6** — Least degradation, maintained ~15% accuracy range for most of training
2. **LR=1e-5** — Accuracy steadily declined to ~5%, format adherence dropped to ~20%
3. **LR=5e-5** — Catastrophic collapse by step 5-10. Model generated gibberish, zero rewards → zero gradients → no further learning
4. **None of the LRs improved over baseline** — The base model (Qwen2.5-Math-1.5B) starts at only ~18% GSM8K accuracy with sparse binary rewards (~3% rollout success), making GRPO optimization very challenging

## Analysis

- GRPO training with sparse rewards (binary 0/1 for correctness) on a base model is inherently difficult
- With only ~3% of rollouts getting reward=1, most groups have all-zero advantages → no learning signal
- Higher LRs amplify noise from the sparse reward signal, causing catastrophic forgetting
- LR=5e-6 best preserves the initial model capability while allowing some exploration
- Format adherence (think/answer tags) declines with all LRs, more severely with higher LRs

## Conclusion

For GRPO training on this task, conservative learning rates (5e-6) are essential. The optimization landscape is extremely noisy with sparse rewards, and aggressive LRs quickly destroy model capabilities.
