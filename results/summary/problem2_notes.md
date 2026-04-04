# Problem 2: grpo_baselines

## Results
| Loss Type | Best Acc | Best @ Step | Final Acc (last eval) | Format OK Rate Trend |
|-----------|----------|-------------|----------------------|---------------------|
| no_baseline | 21.09% | 70 | 19.53% (step 80) | Improving: 51% → 68% |
| reinforce_with_baseline | 16.80% | 70 | 12.11% (step 170) | Declining: 48% → 38% |

## Discussion (2 sentences)
The no_baseline approach significantly outperforms reinforce_with_baseline, achieving 21% accuracy (improving over baseline 18%) vs 17% (declining). Two key trends on other metrics: (1) gradient norms are notably lower with no_baseline (~0.05-0.13 vs ~0.13-0.44), suggesting more stable optimization, and (2) format adherence (format_ok_rate) improves with no_baseline (51%→68%) but declines with reinforce_with_baseline (48%→38%), indicating the baseline subtraction may be introducing harmful variance in this sparse-reward setting.

## Best loss type for subsequent experiments: no_baseline
