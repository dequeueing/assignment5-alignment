# GRPO (Group Relative Policy Optimization) Implementation Investigation Report

## Executive Summary

This repository contains a complete implementation of the GRPO training algorithm, as described in DeepSeekMath (arxiv.org/abs/2402.03300) and DeepSeek-R1 (arxiv.org/abs/2501.12948). The implementation provides a modern reinforcement learning approach for fine-tuning language models on mathematical reasoning tasks, contrasting with the traditional Expert Iteration (SFT-only) baseline also implemented in the codebase.

**Key Timeline**: All GRPO components were implemented in a series of 7 commits between April 2-2, 2026, starting with foundational policy gradient loss functions and culminating in the complete training loop.

---

## 1. Architecture Overview

### 1.1 Core Components

The GRPO implementation consists of five main layers:

1. **Reward Computation** (`drgrpo_grader.py`): Math problem grading using `r1_zero_reward_fn`
2. **Group Normalization** (`run_compute_group_normalized_rewards`): Per-group advantage computation
3. **Policy Gradient Loss Functions** (`run_compute_naive_policy_gradient_loss`, `run_compute_grpo_clip_loss`): Three loss types
4. **Microbatch Training Step** (`run_grpo_microbatch_train_step`): Single gradient update
5. **Training Loop** (`grpo_train.py`): Full on-policy RL training orchestration

### 1.2 Key Difference from Expert Iteration

| Aspect | Expert Iteration | GRPO |
|--------|-----------------|------|
| **Data** | Correct rollouts only (SFT) | All rollouts (RL) |
| **Supervision** | Ground truth answers | Computed rewards |
| **Optimization** | Maximum likelihood (L2 loss) | Policy gradient (RL) |
| **Advantage Baseline** | N/A | Group-normalized rewards |

---

## 2. Component Deep Dive

### 2.1 Group Normalized Rewards (`tests/adapters.py:38-120`)

**Purpose**: Convert raw scalar rewards into per-token advantages via group normalization.

**Implementation Pattern**:
```python
1. Compute raw rewards for each rollout via reward_fn
2. Group rollouts (reshape into n_groups × group_size)
3. Normalize within each group:
   - Subtract group mean
   - Optionally divide by group std (if normalize_by_std=True)
4. Flatten back to sequence order
5. Return: (normalized_rewards, raw_rewards, metadata_dict)
```

**Key Details**:
- Handles group boundaries correctly with `.view(n_groups, group_size)`
- Uses `keepdim=True` for broadcasting safety
- Tracks 8 metadata statistics: min/max/mean/std of both raw and normalized rewards
- Epsilon parameter (`advantage_eps`) prevents division by zero

**Commit**: `76cdc4c` - Implement compute_group_normalized_rewards for GRPO

---

### 2.2 Policy Gradient Loss Functions

#### 2.2.1 Naive Policy Gradient (`tests/adapters.py:162-180`)

**Formula**: `loss = -advantages * policy_log_probs`

**Implementation**:
```python
# Broadcast advantage from (batch_size, 1) -> (batch_size, sequence_length)
# Then element-wise multiply: -A * log(πθ)
return -raw_rewards_or_advantages * policy_log_probs
```

**Used For**:
- `loss_type="no_baseline"`: Uses raw rewards directly
- `loss_type="reinforce_with_baseline"`: Uses advantages (group-normalized rewards)

**Commit**: `3e01457` - Implement compute_naive_policy_gradient_loss

---

#### 2.2.2 GRPO-Clip Loss (`tests/adapters.py:183-229`)

**Formula**: `loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`

where `ratio = exp(log πθ - log πθ_old)` and A is the advantage.

**Implementation Steps**:
```python
1. Compute log ratio: log πθ - log πθ_old (numerically stable)
2. Take exp to get ratio
3. Broadcast advantages over sequence dimension
4. Compute clipped ratio: clamp(ratio, 1-ε, 1+ε)
5. Compute LHS = ratio * A, RHS = clipped_ratio * A
6. Loss = -min(LHS, RHS) per token
7. Track clipped_ratio_fraction for monitoring
```

**Benefits Over Naive**:
- Prevents policy from drifting too far from old policy
- Provides on-policy training guarantee
- Monitors exploration via clipped_ratio_fraction

**Commit**: `eab6f56` - Implement compute_grpo_clip_loss

---

#### 2.2.3 Policy Gradient Loss Wrapper (`tests/adapters.py:232-261`)

**Purpose**: Unified interface for all three loss types.

**Routing Logic**:
- `no_baseline` → naive loss with raw_rewards
- `reinforce_with_baseline` → naive loss with advantages
- `grpo_clip` → clipped PPO-style loss with old_log_probs

**Commit**: `c30fff7` - Implement compute_policy_gradient_loss wrapper

---

### 2.3 Masked Mean (`tests/adapters.py:264-291`)

**Purpose**: Compute mean over response tokens only (mask out prompt/padding).

**Implementation**:
```python
if dim is None:
    return masked_tensor.sum() / mask.sum()  # Global average
else:
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)  # Per-dimension average
```

**Used For**: Aggregating per-token losses into per-example losses

---

### 2.4 GRPO Microbatch Training Step (`tests/adapters.py:310-365`)

**Purpose**: Single gradient update for a microbatch of rollouts.

**Execution Flow**:
```
1. Compute per-token loss via run_compute_policy_gradient_loss
2. Aggregate to per-example loss: masked_mean(per_token_loss, response_mask, dim=-1)
3. Average over batch: per_example_loss.mean()
4. Scale for gradient accumulation: loss / gradient_accumulation_steps
5. Backprop: loss.backward()
```

**Key Design**:
- Supports gradient accumulation for effective larger batch sizes
- Properly masks out prompt tokens in loss computation
- Returns loss scalar and metadata dict for logging

**Commit**: `c9e2d8a` - Implement run_grpo_microbatch_train_step

---

### 2.5 GRPO Training Loop (`cs336_alignment/grpo_train.py`)

**Hyperparameter Configuration** (`GRPOConfig` dataclass):

| Category | Key Parameters | Default |
|----------|---------------|---------|
| **Training** | n_grpo_steps, learning_rate, epochs_per_rollout_batch | 200, 1e-5, 1 (on-policy) |
| **Rollouts** | rollout_batch_size, group_size | 256, 8 |
| **Gradient** | train_batch_size, gradient_accumulation_steps | 256, 128 |
| **Loss** | loss_type, cliprange | reinforce_with_baseline, 0.2 |
| **Sampling** | temperature, min/max_tokens | 1.0, 4/1024 |
| **Normalization** | advantage_eps, use_std_normalization | 1e-6, True |

**Main Training Loop** (`main()` function):

```
for step in range(n_grpo_steps):
    1. Sample rollouts using vLLM
       - Generate group_size candidates per prompt
       - Compute rewards via r1_zero_reward_fn
    
    2. Compute advantages
       - Group-normalize rewards: compute_group_normalized_rewards
    
    3. Training phase (on-policy, single epoch)
       - Compute old log probs (before update)
       - Create minibatches from rollouts
       - For each minibatch:
         - Compute policy log probs
         - Run grpo_microbatch_train_step with chosen loss_type
         - Accumulate gradients
       - Optimizer step and scheduler update
    
    4. Validation (every eval_every_steps)
       - Evaluate on validation set
       - Log metrics
    
    5. Checkpointing
       - Save model, optimizer, scheduler
```

**On-Policy Training Characteristic**:
- `epochs_per_rollout_batch = 1` means each rollout is used exactly once
- Old log probs computed before any training updates
- Advantage computed fresh for each step
- Traditional reinforcement learning (no replay buffer)

**Commit**: `8fcfcc5` - Add GRPO training loop implementation

---

## 3. Data Flow Analysis

### 3.1 Rollout to Advantage to Loss

```
FormattedExample (question + ground_truth)
    ↓
sample_rollouts() [uses vLLM]
    ↓
(rollout_responses, ground_truths)  [length = batch_size × group_size]
    ↓
compute_group_normalized_rewards()
    ├─ Call r1_zero_reward_fn for each rollout
    ├─ Group normalize: (μ - mean) / σ
    └─ Return advantages + metadata
    ↓
compute_policy_gradient_loss()
    ├─ Get policy_log_probs from forward pass
    ├─ Compute per-token loss based on loss_type
    └─ Return per-token loss + metadata
    ↓
grpo_microbatch_train_step()
    ├─ masked_mean over sequence → per-example loss
    ├─ mean over batch → scalar loss
    ├─ Scale by gradient accumulation
    └─ loss.backward()
```

### 3.2 Shapes Through Pipeline

```
Input: 256 rollouts from 32 prompts (group_size=8)
       └─ (batch_size=256, seq_len=512)

After compute_group_normalized_rewards:
       ├─ normalized_rewards: (256,)
       └─ raw_rewards: (256,)

After get_response_log_probs:
       └─ policy_log_probs: (256, 512)

After compute_policy_gradient_loss:
       └─ per_token_loss: (256, 512)

After masked_mean (response_mask is (256, 512) with ~400 ones per row):
       └─ per_example_loss: (256,)

After batch mean:
       └─ scalar loss
```

---

## 4. Loss Function Comparison

### 4.1 Mathematical Formulation

**Naive Policy Gradient**:
```
L(θ) = -A * log πθ(response | prompt)
```
Simple but unstable when policy updates are large.

**GRPO-Clip (Clipped Policy Gradient)**:
```
L(θ) = -min(r * A, clip(r, 1-ε, 1+ε) * A)
where r = πθ(·) / πθ_old(·)
```
Prevents large policy updates while maintaining on-policy training.

### 4.2 Metadata Tracking

| Loss Type | Metadata Returned |
|-----------|------------------|
| naive/reinforce_with_baseline | {} (empty) |
| grpo_clip | clipped_ratio_fraction |

The `clipped_ratio_fraction` tracks what percentage of tokens had their ratio clipped, indicating exploration intensity.

---

## 5. Recent Commit Timeline

```
Date: April 2, 2026

Commit 76cdc4c (10:12:35)
└─ Implement compute_group_normalized_rewards
   └─ 51 insertions, focuses on within-group normalization

Commit adb6df2 (10:15:XX)
└─ Implement run_masked_mean
   └─ Utility for computing masked averages

Commit 3e01457 (10:17:32)
└─ Implement compute_naive_policy_gradient_loss
   └─ Basic REINFORCE loss: -A * log p

Commit eab6f56 (10:22:15)
└─ Implement compute_grpo_clip_loss
   └─ PPO-style clipped loss with ratio tracking

Commit c30fff7 (10:XX:XX)
└─ Implement compute_policy_gradient_loss wrapper
   └─ Routes between three loss types

Commit c9e2d8a (10:51:51)
└─ Implement run_grpo_microbatch_train_step
   └─ Complete single gradient update with masking

Commit 8fcfcc5 (10:55:59, HEAD)
└─ Add GRPO training loop implementation
   └─ 588 insertions: full training orchestration
```

**Pattern**: Building bottom-up from utility functions → loss functions → training step → full loop

---

## 6. Testing Infrastructure

### 6.1 Unit Tests (`tests/test_grpo.py`)

- `test_compute_group_normalized_rewards_*`: Normalizes with/without std division
- `test_compute_naive_policy_gradient_loss`: Validates -A * log p computation
- `test_compute_grpo_clip_loss_*`: Tests with large (10.0) and small (0.1) clip ranges
- `test_compute_policy_gradient_loss_*`: All three loss types (no_baseline, reinforce_with_baseline, grpo_clip)
- `test_masked_mean_*`: Tests different dimensions (0, 1, -1, None)
- `test_grpo_microbatch_train_step_*`: Single step and 10-step gradient accumulation

### 6.2 Test Fixtures

From `conftest.py`, provides:
- `reward_fn`: Mocked reward function returning consistent scores
- `rollout_responses`, `repeated_ground_truths`: Pre-configured test data
- `policy_log_probs`, `old_log_probs`: Pre-configured log probability tensors
- `response_mask`: Binary mask for response tokens
- `raw_rewards`, `advantages`: Pre-computed reward tensors

### 6.3 Snapshot Testing

Uses `numpy_snapshot` for regression testing against saved outputs:
```python
numpy_snapshot.assert_match(output)  # Compares against ./tests/_snapshots/
```

This ensures numerical stability across changes.

---

## 7. Key Design Decisions

### 7.1 On-Policy Training

```python
epochs_per_rollout_batch: int = 1  # Each rollout used exactly once
```
**Rationale**: True RL requires fresh samples. Unlike SFT, GRPO needs on-policy data for valid policy gradients.

### 7.2 Group Normalization

```python
# Normalize within groups, not globally
normalized = (raw - group_mean) / (group_std + eps)
```
**Rationale**: Reduces high-variance reward estimates by normalizing within groups, a key GRPO innovation over naive REINFORCE.

### 7.3 Three Loss Types

```python
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]
```
**Rationale**: 
- `no_baseline`: Pure REINFORCE with raw rewards
- `reinforce_with_baseline`: REINFORCE with group-normalized advantages
- `grpo_clip`: Clipped version (safer, recommended)

### 7.4 Gradient Accumulation

```python
loss = loss / gradient_accumulation_steps
loss.backward()  # Accumulates into param.grad
```
**Rationale**: Allows simulating larger effective batch sizes (256 train_batch_size with 128 accumulation_steps = 32K effective).

---

## 8. Integration with Existing Components

### 8.1 Dependency on Expert Iteration Module

GRPO imports from `expert_iter.py`:
- `FormattedExample`: Data class for question/answer/ground_truth
- `build_r1_zero_prompt()`: Formats prompts with `<think>` tags
- `build_sampling_params()`: Creates vLLM sampling configurations
- `init_vllm()`: Initializes vLLM engine
- `load_formatted_examples()`: Loads JSONL data
- `load_policy_into_vllm_instance()`: Syncs model to vLLM

### 8.2 Dependency on Fine-Tuning Module

GRPO imports from `fine_tuning.py`:
- `get_response_log_probs()`: Computes policy log probabilities
- `load_model()`: Loads transformer model
- `save_pretrained()`: Saves checkpoints
- `tokenize_prompt_and_output()`: Tokenizes prompts and responses

### 8.3 Dependency on Grading Module

GRPO uses `drgrpo_grader.py`:
- `r1_zero_reward_fn(response, ground_truth)`: Returns reward dict with keys: "reward", "format_reward", "answer_reward"

---

## 9. Known Characteristics

### 9.1 Numerical Stability

1. **Log-space computation**: Uses `log_ratio = log p_new - log p_old` to avoid underflow/overflow
2. **Epsilon parameters**: `advantage_eps=1e-6` prevents division by zero in normalization
3. **Clipping**: Explicit clipping in GRPO-Clip loss bounds the ratio

### 9.2 Memory Efficiency

- Rollout batch size (256) sampled and stored in memory
- Train batch size (256) processed with gradient accumulation (128 steps)
- Effective per-GPU memory: ~2-3x train_batch_size

### 9.3 Monitoring Capabilities

- Logs per-group reward statistics
- Tracks clipped_ratio_fraction for loss_type="grpo_clip"
- Validation metrics every 5 steps by default
- Checkpoint saving strategy included

---

## 10. Comparison with Related Work

### DeepSeekMath vs Implementation

| Aspect | DeepSeekMath Paper | This Implementation |
|--------|-------------------|-------------------|
| **Group Size** | 8 (fixed) | 8 (configurable) |
| **Loss Type** | PPO-Clip | Three options: no_baseline, reinforce, grpo_clip |
| **Advantage** | Group-normalized | Group-normalized ✓ |
| **On-Policy** | Yes | Yes (epochs_per_rollout_batch=1) ✓ |
| **Sampling** | vLLM | vLLM ✓ |

### DeepSeek-R1 Extensions

R1 adds:
- Longer reasoning paths (chain-of-thought)
- Verification rewards for intermediate steps
- Multi-stage training with varying loss types

This implementation provides the foundational GRPO framework that R1 extends.

---

## 11. Potential Extensions

Based on the current codebase structure, future enhancements could include:

1. **Reward Model Integration**: Replace `r1_zero_reward_fn` with learned reward model
2. **PPO-like Advantage Estimation**: Add separate value network for baseline
3. **Multi-GPU Distributed Training**: DDP wrapper for parallel training
4. **Curriculum Learning**: Difficulty-based example ordering
5. **Verifier Integration**: Early stopping based on intermediate verification
6. **KL Penalty**: Explicit divergence control instead of clipping

---

## 12. Summary

The GRPO implementation is a complete, well-structured RL training system for math reasoning:

✅ **Strengths**:
- Clean separation of concerns (group norm → loss → step → loop)
- Three loss variants for flexibility
- Proper on-policy handling with single-epoch training
- Comprehensive metadata tracking for monitoring
- Solid integration with vLLM for efficient sampling

✅ **Key Innovation**:
- Group-normalized advantages reduce variance compared to naive REINFORCE
- Clipped loss variant provides stability without replay buffer

✅ **Production Readiness**:
- Full checkpoint/restore capability
- Validation loop with early stopping infrastructure
- Configurable via GRPOConfig dataclass
- Extensive unit tests with snapshot verification

This codebase provides a reference implementation for applying policy gradient optimization to language model alignment, specifically for mathematical reasoning tasks.
