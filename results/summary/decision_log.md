# Decision Log — GRPO Experiments (CS336 Assignment 5)

## Problem 1: Learning Rate
- **Sweep**: {5e-6, 1e-5, 5e-5}
- **Decision**: LR=5e-6 (least destructive)
- **Note**: Pre-shuffle-fix results; all LRs showed minimal improvement

## Problem 2: Baselines
- **Comparison**: no_baseline vs reinforce_with_baseline
- **Decision**: no_baseline (21.09% best vs 12.11% declining)

## Problem 3: Length Normalization Theory
- Analyzed masked_mean vs masked_normalize(constant=1024)
- masked_normalize amplifies short-response gradients, masked_mean is neutral

## Problem 4: Length Normalization Empirical
- **Decision**: masked_mean (no length normalization) — 21.09% vs 19.53%

## Problem 5: Standard Deviation Normalization
- use_std_normalization irrelevant for no_baseline (uses raw rewards)
- Tested with reinforce_with_baseline: std=True slightly better
- **Decision**: use_std_normalization=True (default)

## Problem 6: Off-Policy Implementation
- **CRITICAL BUG FIX**: DataLoader shuffle=True → shuffle=False
  - Misassigned advantages to wrong examples; fixing → 17% → 69%
- Implemented off-policy old_log_probs pre-computation
- Moved optimizer.step() inside epoch loop

## Problem 7: Off-Policy Sweep
- epochs=1: 68.75% @step190 | epochs=4: 71.48% @step80
- **Decision**: epochs=4 (3x faster convergence, higher peak)

## Problem 8: Clip Ablation
- With clip (ε=0.2): 71.48%, grad_norm ~9.5
- No clip: 72.66%, but grad_norm spikes to 63.75
- **Decision**: Use clipping for stability

## Problem 9: Prompt Ablation
- r1_zero (format+answer): 71.48% val accuracy
- question_only (answer only): 0.39% val accuracy, 87.5% rollout reward
- **Decision**: r1_zero format is essential — without it, model never learns structured reasoning

## Problem 10: MATH Leaderboard
- Best config applied to MATH dataset (7498 train, 5000 val)
- **Peak: 49.61%** at step 190 (~4.8 hours, single A100)
- Baseline: 18.75% → improvement of +30.86 pp

## Overall Best Configuration
```
LR=5e-6, loss=grpo_clip, epochs=4, cliprange=0.2
no_baseline, no_length_normalization, std_normalization=True
shuffle=False (CRITICAL), r1_zero reward function
```
