# GRPO Experiments Results — Quick Navigation

## 📊 Main Report
- **[Final_Report.md](Final_Report.md)** — Complete 18k word analysis with all findings, decisions, and recommendations

## 🎯 Quick Summary

| Problem | Result | Key File |
|---------|--------|----------|
| **P1** | LR=5e-6 best (21.09%) | [plot](results/summary/problem1_lr_comparison.png) \| [notes](results/summary/problem1_notes.md) |
| **P2** | no_baseline > reinforce (21.09%) | [plot](results/summary/problem2_baselines.png) \| [notes](results/summary/problem2_notes.md) |
| **P3** | Length norm theory analysis | [notes](results/summary/problem3_notes.md) |
| **P4** | masked_mean best (21.09%) | [plot](results/summary/problem4_length_norm.png) \| [notes](results/summary/problem4_notes.md) |
| **P5** | std_norm irrelevant (for no_baseline) | [notes](results/summary/problem5_notes.md) |
| **P6** | **Shuffle bug fix: 17% → 69%** | [critical fix documented] |
| **P7** | epochs=4 (71.48%, 3x faster) | [plot](results/summary/problem7_off_policy.png) \| [notes](results/summary/problem7_notes.md) |
| **P8** | Clipping for stability (clip vs no-clip) | [plot](results/summary/problem8_clip_ablation.png) \| [notes](results/summary/problem8_notes.md) |
| **P9** | r1_zero essential (71.48% vs 0.39%) | [plot](results/summary/problem9_prompt_ablation.png) \| [notes](results/summary/problem9_notes.md) |
| **P10** | **MATH: 49.61% (+30.86pp from baseline)** | [plot](results/summary/problem10_leaderboard.png) \| [notes](results/summary/problem10_notes.md) |

## 📈 Plots Gallery

### Training Curves
- `problem1_lr_comparison.png` — Learning rate sweep
- `problem2_baselines.png` — Baseline comparison
- `problem4_length_norm.png` — Length normalization ablation
- `problem5_std_norm.png` — Standard deviation normalization
- `problem7_off_policy.png` — Off-policy epochs sweep
- `problem8_clip_ablation.png` — Clipping ablation
- `problem9_prompt_ablation.png` — Reward function comparison
- `problem10_leaderboard.png` — MATH leaderboard optimization

## 🔍 Detailed Analysis

All problem notes are in `results/summary/problem*_notes.md` with:
- Setup and configuration
- Results (tables + statistics)
- Key findings
- Conclusions

## 💾 Configuration Files

- **[best_config.json](results/summary/best_config.json)** — Final recommended config
- **[decision_log.md](results/summary/decision_log.md)** — All decisions made

## 🧮 Raw Data

All metrics available in JSONL format:
- `results/grpo_leaderboard/grpo_metrics.jsonl` — Full P10 leaderboard run (200 steps)
- All other problem directories contain `grpo_metrics.jsonl` with step-by-step metrics

## 🎓 Key Insights

### The Shuffle Bug (Problem 6)
**+52 percentage points** from fixing DataLoader shuffle=True → shuffle=False. This single fix was worth more than all other optimizations combined.

### Off-Policy Training (Problem 7)
Epochs=4 converges **3x faster** and reaches **2.7pp higher** accuracy than epochs=1.

### Stability vs Performance (Problem 8)
Clipping reduces gradient norm by **7x** (from 63 to 9), providing critical stability despite -1.2pp accuracy trade-off.

### Reward Design (Problem 9)
Format-aware rewards (r1_zero) essential: question_only achieves 87.5% training reward but 0.39% validation accuracy (reward hacking).

### MATH Leaderboard (Problem 10)
**49.61%** peak accuracy from 18.75% baseline = **+30.86 percentage points** improvement on significantly harder dataset.

## 📋 Reproducibility

To reproduce leaderboard results:
```bash
.venv/bin/python -m cs336_alignment.grpo_train \
  --train-data-path data/math/train.formatted.jsonl \
  --val-data-path data/math/validation.formatted.jsonl \
  --learning-rate 5e-6 \
  --n-grpo-steps 300 \
  --loss-type grpo_clip \
  --epochs-per-rollout-batch 4 \
  --cliprange 0.2 \
  --rollout-batch-size 128 \
  --train-batch-size 128
```

See Final_Report.md for detailed reproducibility section.

---

**All experiments complete** | **All 10 problems solved** | **Ready for submission**
