# Problem 6 & 7: Off-Policy GRPO Implementation & Sweep

## Problem 6: Implementation
Fixed the `pass` placeholder in grpo_train.py for off-policy GRPO:
1. Pre-compute `old_log_probs` for ALL batches using `torch.inference_mode()` before any gradient updates
2. Store per-batch old_log_probs for consistent indexing
3. Move `optimizer.step()` inside the epoch loop (after each epoch, not after all epochs)
   - This enables TRUE off-policy training where model params change between epochs
   - Ratio deviates from 1.0 → clipping mechanism activates
4. Changed DataLoader to `shuffle=False` for correct advantage-to-example mapping
   - **Critical bug fix**: original `shuffle=True` randomized advantage assignments!

## Problem 7: Off-Policy Sweep Results

### Configuration
| Setting | epochs=1 (on-policy) | epochs=4 (off-policy) |
|---------|---------------------|----------------------|
| loss_type | grpo_clip | grpo_clip |
| LR | 5e-6 | 5e-6 |
| cliprange | 0.2 | 0.2 |
| Steps completed | 200/200 | 82/200 (stopped early) |
| Time per step | ~30s | ~78s |

### Val Accuracy Comparison (by step)
| Step | epochs=1 | epochs=4 |
|------|----------|----------|
| 10   | 28.12%   | **49.22%** |
| 20   | 40.23%   | **62.11%** |
| 30   | 46.48%   | **66.41%** |
| 40   | 51.56%   | **66.41%** |
| 50   | 56.25%   | **66.80%** |
| 60   | 57.42%   | **68.36%** |
| 70   | 58.59%   | **70.70%** |
| 80   | 60.16%   | **71.48%** |
| 100  | 64.45%   | N/A |
| 150  | 66.41%   | N/A |
| 200  | 67.97%   | N/A |

### Key Findings
1. **epochs=4 converges ~3x faster per step**: reaches 66% at step 30 vs step 120 for epochs=1
2. **epochs=4 reaches higher peak**: 71.48% @step80 vs 68.75% @step190 for epochs=1
3. **epochs=4 is more wall-clock efficient**: 71.5% in ~104min vs 68.8% in ~100min
4. **Clipping is active**: clip fraction grows from 4% → 14% as model drifts from old policy
5. **Gradient explosion risk**: grad_norm grows to 9.5 at step 80 for epochs=4

### Shuffle Bug Discovery
The original code used `shuffle=True` in the DataLoader, but indexed advantages by `batch_idx * batch_size`.
With shuffling, this misassigns advantages to wrong examples, destroying the learning signal.
Fixing to `shuffle=False` caused dramatic improvement (17% → 69% val accuracy over 200 steps).

## Decision
**Use epochs=4 with grpo_clip** — faster convergence, higher peak accuracy.
Best config: LR=5e-6, grpo_clip, epochs=4, cliprange=0.2, shuffle=False
