# Problem 5: Group Standard Deviation Normalization

## Key Insight
`use_std_normalization` only affects the `advantages` (group-normalized rewards), NOT raw rewards.
Since our best loss_type=`no_baseline` uses raw_rewards directly, std normalization has **zero effect** on it.
Therefore, this comparison is done with `reinforce_with_baseline` to isolate the effect.

## Experimental Setup
| Setting | std=True (Problem 1 run) | std=False (new run) |
|---------|--------------------------|---------------------|
| loss_type | reinforce_with_baseline | reinforce_with_baseline |
| LR | 5e-6 | 5e-6 |
| Steps completed | 176/200 | 105/200 |

## Results (eval steps)
| Metric | std=True | std=False |
|--------|----------|-----------|
| Best val_acc | 16.80% @step70-80 | 16.41% @step30 |
| Final val_acc | 12.11% @step170 | 13.28% @step100 |
| Avg gradient_norm | ~0.25 | ~0.09 (3x smaller) |
| Avg advantage_std | ~0.40 | ~0.15 (2.5x smaller) |

## Analysis
1. **Gradient scale**: Without std normalization, advantages retain their natural scale (~0.15).
   With std normalization, advantages are rescaled to unit variance (~1.0), producing ~3x larger gradients.
2. **Performance**: Both variants show minimal improvement over baseline (15.62%).
   std=True has a slight edge at peak performance (16.80% vs 16.41%).
3. **Stability**: std=True shows more variance in val_acc (both higher peaks and lower troughs).
   std=False is more stable but doesn't reach as high.

## Decision
**Use std_normalization=True** (default) — it provides slightly better peak performance.
However, for our best config (no_baseline), this setting is irrelevant.

## Carry-forward Config
- loss_type: no_baseline (from Problem 2)
- LR: 5e-6 (from Problem 1)
- use_length_normalization: False / masked_mean (from Problem 4)
- use_std_normalization: True (default, doesn't affect no_baseline)
