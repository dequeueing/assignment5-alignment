# Problem 8: Off-Policy Clip Ablation

## Setup
Both experiments use epochs=4, LR=5e-6, grpo_clip loss.
- **With clipping**: cliprange=0.2 (standard)
- **Without clipping**: cliprange=1e6 (effectively no clipping)

## Results Comparison
| Step | With Clip val_acc | No Clip val_acc | With Clip grad | No Clip grad |
|------|-------------------|-----------------|----------------|--------------|
| 10   | 49.22%            | 51.56%          | 0.29           | 0.44         |
| 20   | 62.11%            | 57.81%          | 0.88           | 2.33         |
| 30   | 66.41%            | 57.81%          | 1.69           | 9.25         |
| 40   | 66.41%            | 64.06%          | 2.52           | 5.94         |
| 50   | 66.80%            | 67.19%          | 3.02           | 6.47         |
| 60   | 68.36%            | **69.53%**      | 4.72           | 12.94        |
| 70   | 70.70%            | **72.66%**      | 8.75           | **63.75**    |
| 80   | 71.48%            | 72.66%          | 9.50           | 27.63        |

## Key Findings
1. **Accuracy**: No-clip eventually reaches HIGHER peak accuracy (72.7% vs 71.5%) but with much more volatility
2. **Gradient stability**: Clipping dramatically reduces gradient norms (8.75 vs 63.75 at step 70)
3. **Early training**: Clipping converges more smoothly — consistently ahead at steps 20-40
4. **Late training**: No-clip catches up and slightly surpasses clip by steps 60-70
5. **Risk**: No-clip gradient norms spike to 63.75x — one bad batch could cause divergence

## Analysis
- PPO-style clipping primarily acts as a **gradient stabilizer**, not an accuracy limiter
- The gradient norm clipping (`max_norm=1.0`) in `clip_grad_norm_` provides a safety net regardless
- In this regime (sparse rewards, small model), the gradient norm clipping alone seems sufficient
- In larger models or harder tasks, PPO clipping would likely be more important for stability

## Decision
**Use clipping (cliprange=0.2)** — safer training with comparable accuracy, worth the small cost
