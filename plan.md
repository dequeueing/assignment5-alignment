# GRPO 实验执行手册（Agent-Ready 版）

## 0. 任务目标

你的唯一目标是：**在单块 40GB A100 GPU 的资源约束下，完成 GRPO 系列实验、产出全部要求的曲线与分析、实现离线策略版本、完成必要消融，并在最后用 4 小时墙钟预算冲击 leaderboard。**

你不是来“提出建议”的；你是来**把整个实验真正做完**的。

只要还有未完成的实验、未验证的实现、未产出的图表、未写完的分析、未达成的 leaderboard 结果，你就**继续执行**，不要停下来等待进一步指示。

---

## 1. 完成标准（Definition of Done）

只有在以下所有条件都满足时，任务才算完成：

1. 问题 1–10 全部执行完毕，或明确记录了为什么某项无法完成以及替代方案。
2. 每个问题都具备：

   * 对应配置记录
   * 运行日志
   * 结构化指标文件
   * 可复现命令
   * 至少一段结论分析
3. 所有需要的对比实验都已有曲线图或表格。
4. 离线策略（off-policy）实现已经完成并验证。
5. leaderboard 任务在 4 小时墙钟预算下真正跑过，并给出最终准确率。
6. 最终报告 `Final_Report.md` 已写完，内容完整、可提交。

**禁止把“只做了一部分、等用户确认”视为完成。**

---

## 2. 运行原则（必须遵守）

### 2.1 总体原则

* **先做最关键、最能影响后续决策的实验。**
* **单卡不可并行时，严格串行执行。**
* **每做完一个实验，立即整理结果，再进入下一个。**
* **如果某个实验结果会影响后续配置选择，先完成它，再推进后续实验。**
* **不做无意义的大网格搜索；只做足够识别趋势的实验。**
* **任何实验一旦出现明显失败信号，立刻早停并记录原因。**
* **任何实现改动必须先做最小验证，再进入大规模实验。**

### 2.2 禁止行为

* 不要因为某个实验效果差就中断整个任务。
* 不要把“还没画图”“还没分析”留到最后一起补。
* 不要跳过失败实验的记录。
* 不要在关键前置实验（问题1/2/4/5/6/7）未完成时，提前开始依赖它们结论的后续实验。
* 不要等待人工确认下一步；你要自己根据决策规则继续推进。

### 2.3 决策优先级

如果多个方向都可做，优先级如下：

1. **解锁后续实验依赖的实验**
2. **可能显著提升 leaderboard 的实验**
3. **实现正确性验证**
4. **关键消融**
5. **低优先级分析或附加实验**

---

## 3. 已知约束

* GPU：**1 × 40GB A100**
* 训练方式：**vLLM 采样 + 策略训练共享同一张 GPU**
* 数据：**GSM8K train/test (`formatte.jsonl`)**，leaderboard 阶段使用 **MATH train / full 5K val**
* 模型：**Qwen/Qwen2.5-Math-1.5B**
* 所有实验默认固定随机种子：**42**

你必须始终围绕这些约束做决策，不要假设额外算力存在。

---

## 4. 目录与产物规范

所有内容统一写入 `results/`：

```text
results/
├── grpo_learning_rate_[lr]/
├── grpo_baselines_[loss_type]/
├── grpo_length_norm_[method]/
├── grpo_std_norm_[true_false]/
├── grpo_offpolicy_impl_check/
├── grpo_offpolicy_[epoch_batch]/
├── grpo_clip_ablation_[clip_mode]/
├── grpo_prompt_[r1_or_question_only]/
├── grpo_leaderboard/
└── summary/
```

每个实验目录必须至少包含：

```text
config.json          # 完整配置
command.sh           # 可复现实验命令
metrics.csv          # 每步指标
logs.txt             # 原始日志
notes.md             # 本实验结论、异常、观察
plot.png / *.png     # 相关图表
best_model/          # 最优checkpoint（如有）
```

另外维护以下全局文件：

```text
results/summary/experiment_index.csv
results/summary/best_config.json
results/summary/decision_log.md
results/summary/final_tables.md
Final_Report.md
```

---

## 5. 全局日志字段（所有实验统一）

所有实验必须按统一 schema 记录指标，至少包含：

```python
log_metrics = {
    'step': step,
    'wall_clock_sec': wall_clock_sec,
    'learning_rate': lr,
    'val_answer_reward': float,
    'val_accuracy': float,
    'loss': float,
    'entropy': float,
    'response_length': float,
    'gradient_norm': float,
    'gpu_memory_gb': float,
    'train_batch_size': int,
    'rollout_batch_size': int,
    'group_size': int,
    'gradient_accumulation_steps': int,
    'epochs_per_rollout_batch': int,
    'loss_type': str,
    'use_std_normalization': bool,
    'length_normalization': str,
    'prompt_style': str,
    'status': str,  # running / early_stop / completed / diverged / oom / failed
}
```

如果某项指标暂时拿不到，也必须在 `notes.md` 中说明缺失原因。

---

## 6. 全局早停与失败处理规则

满足以下任一条件，立即停止当前实验并标记失败或早停：

1. `loss` 为 `NaN` 或 `Inf`
2. `loss > 1e5`
3. `gradient_norm > 100`
4. 连续 **20 步** `val_answer_reward` 无任何正增长
5. 显存接近上限并反复 OOM
6. 训练逻辑明显错误（shape mismatch、old_log_probs 未冻结、reward 全为常数等）

### 失败后处理顺序

若实验失败，按以下顺序处理：

1. 记录失败现象与错误日志
2. 判断是：

   * **实现 bug**
   * **超参不合适**
   * **资源不足**
   * **设计本身不可行**
3. 只允许做一次最小修复重试：

   * bug → 修 bug 后重试一次
   * OOM → 降 batch / 降 ga / 降 epoch 后重试一次
   * divergence → 降学习率或缩短步数后重试一次
4. 若重试仍失败，则：

   * 记录最终失败原因
   * 标记该实验为失败
   * 继续推进下一个实验或替代方案

**禁止在一个失败配置上无限反复。**

---

## 7. 统一实验执行模板

对每一个实验，都按下面流程执行：

### Phase A：准备

1. 创建实验目录
2. 写 `config.json`
3. 写 `command.sh`
4. 检查显存预算
5. 确认本实验依赖的最优配置是否已确定

### Phase B：运行

1. 启动训练
2. 实时记录关键指标
3. 监控 loss / gradient / entropy / reward / length
4. 若触发早停规则，则立刻停止

### Phase C：收尾

1. 保存 `metrics.csv`
2. 保存 `logs.txt`
3. 保存最佳 checkpoint
4. 生成图表
5. 写 `notes.md`（至少包含：结论、异常、是否进入下一阶段）
6. 更新 `results/summary/experiment_index.csv`
7. 更新 `results/summary/decision_log.md`

### Phase D：决策

根据实验目标判断：

* 是否确定当前最优配置
* 是否需要补一个额外验证实验
* 是否可以推进到下游实验

---

## 8. 实验总顺序（必须按依赖推进）

严格按以下顺序推进：

1. **问题1：学习率调优**
2. **问题2：基线效应**
3. **问题3：长度标准化理论分析**（可与问题2/4并行写作，但不能阻塞实验）
4. **问题4：长度标准化实验**
5. **问题5：组标准差标准化**
6. **问题6：离线策略实现与验证**
7. **问题7：离线策略超参搜索**
8. **问题8：离线 clipping 消融**
9. **问题9：提示词消融**
10. **问题10：leaderboard 最优化**
11. **最终报告整理**

---

## 9. 问题1：学习率调优（grpo_learning_rate）

## 目标

找到在当前 GRPO 设置下效果最好的学习率，作为后续所有实验的默认学习率。

## 基础配置

```yaml
n_grpo_steps: 50
rollout_batch_size: 256
group_size: 8
train_batch_size: 256
gradient_accumulation_steps: 128
epochs_per_rollout_batch: 1
loss_type: reinforce_with_baseline
use_std_normalization: true
temperature: 1.0
max_tokens: 1024
```

## 候选学习率

```yaml
[5e-6, 1e-5, 2e-5, 5e-5]
```

## 执行规则

### Stage 1：快速扫描

至少跑以下 3 个配置：

* `5e-6`
* `1e-5`
* `5e-5`

如果这 3 个中最好结果出现在中间区间不明确，再补跑 `2e-5`。

每个配置先跑 **50 步**，记录：

* `val_answer_reward`
* `val_accuracy`
* `loss`
* `entropy`
* `response_length`
* `gradient_norm`

### Stage 2：完整确认

选择 Stage 1 最佳学习率，跑 **200 步** 完整训练。

## 选择规则

按以下优先级选择最佳学习率：

1. 最终 `val_accuracy` 更高
2. 如果精度接近，则看 `val_answer_reward` 更高者
3. 如果 reward 接近，则优先更稳定（无梯度爆炸、loss 更平滑）
4. 如果仍接近，优先较小学习率

## 产物

* 不同学习率的 reward 曲线图
* 最优学习率对应的完整 200 步曲线
* 最优模型 checkpoint
* `notes.md`：说明哪个学习率最好，以及为什么

## 完成标志

当 `results/summary/best_config.json` 中写入 `learning_rate`，且问题1图表与分析完成，问题1才算结束。

---

## 10. 问题2：基线效应（grpo_baselines）

## 目标

比较 `no_baseline` 与 `reinforce_with_baseline` 两种方式对训练稳定性和最终性能的影响。

## 配置

使用问题1得到的最优学习率。

```yaml
n_grpo_steps: 100
use_std_normalization: true
```

对比：

```yaml
A: loss_type = no_baseline
B: loss_type = reinforce_with_baseline
```

## 执行规则

* 两个条件都各跑 100 步
* 单卡串行执行
* 记录相同指标
* 必须额外画出 `gradient_norm` 对比图

## 判断规则

优先选择：

1. 最终 `val_accuracy` 更高者
2. 若接近，则看收敛更快者
3. 若仍接近，则选梯度更稳者

## 产物

* 两条 reward 曲线
* 两条 gradient norm 曲线
* 简短分析：谁更稳、谁收敛更快、谁更适合后续实验

## 完成标志

将最优 `loss_type` 写入 `results/summary/best_config.json`。

---

## 11. 问题3：长度标准化理论分析（think_about_length_normalization）

## 目标

在不依赖实验结果的前提下，清晰分析 `masked_mean` 与 `masked_normalize` 的梯度差异、长度偏差和任务影响。

## 必须输出的分析结构

1. 两种公式定义
2. 对短序列 / 长序列的梯度强度比较
3. 对简单题 / 难题的潜在偏置
4. 对数学推理任务的潜在影响
5. 预期实验现象
6. 最终判断：在 math 任务上谁更可能更优，为什么

## 交付要求

输出一份不少于 2 页等价内容的分析文档，写入：

```text
results/summary/length_normalization_theory.md
```

这部分不阻塞实验推进，但必须在最终报告前完成。

---

## 12. 问题4：长度标准化实验（grpo_length_normalization）

## 目标

验证 `masked_mean` 与 `masked_normalize` 在实际训练中的差异。

## 配置

使用：

* 问题1的最优学习率
* 问题2的最优 `loss_type`

```yaml
n_grpo_steps: 100
```

对比：

```yaml
A: length_normalization = masked_mean
B: length_normalization = masked_normalize
   constant = max_response_len = 1024
```

## 关键观测指标

* `val_answer_reward`
* `val_accuracy`
* `gradient_norm`
* `response_length`
* `loss`
* `entropy`

## 判断规则

优先选择：

1. 最终验证精度更高者
2. 如果接近，则 reward 更高者
3. 如果接近，则更稳者
4. 如果 masked_normalize 能明显缓解长度偏置或提升长推理表现，则优先它

## 产物

* reward 对比图
* gradient norm 对比图
* response length 趋势图
* 分析：哪种更稳定，哪种更适合后续实验

## 完成标志

将最优 `length_normalization` 写入 `best_config.json`。

---

## 13. 问题5：组标准差标准化（grpo_group_standard_deviation）

## 目标

比较 advantage 是否除以组内标准差。

## 配置

使用此前所有最优配置。

```yaml
n_grpo_steps: 100
```

对比：

```yaml
A: use_std_normalization = true
B: use_std_normalization = false
```

## 执行规则

* 各跑 100 步
* 如可行，记录组内 reward std 分布
* 统计 `std=0` 的组数量

## 判断规则

优先选择：

1. 验证精度更高者
2. 收敛更稳定者
3. 更少异常梯度者
4. 如果取消 std 标准化后性能更高且训练未变差，则优先 `false`

## 产物

* reward 曲线
* gradient norm 曲线
* 稳定性分析
* `std` 标准化是否必要的结论

## 完成标志

将最优 `use_std_normalization` 写入 `best_config.json`。

---

## 14. 问题6：离线策略实现（grpo_off_policy）

## 目标

正确实现支持 `epochs_per_rollout_batch > 1` 的离线策略训练。

## 这是实现题，不只是实验题。

必须真正检查并修正代码逻辑，而不是假设已有实现正确。

## 必查点

### 1. old_log_probs 计算时机

必须在 rollout 生成后、参数更新前计算，且只计算一次：

```python
with torch.inference_mode():
    old_log_probs = get_response_log_probs(...)
```

### 2. 多 epoch 训练循环

必须支持：

```python
for epoch in range(epochs_per_rollout_batch):
    ...
```

### 3. 离线损失

离线训练默认必须使用：

```yaml
loss_type = grpo_clip
```

### 4. gradient_accumulation_steps 调整

当 `epochs_per_rollout_batch > 1` 时，重新计算 GA，避免显存爆炸和有效 batch 混乱。

### 5. 显存检查

old log probs 会额外占内存，必须重新确认 batch 是否可行。

## 验证要求

### 验证1：等价性检查

设：

```yaml
epochs_per_rollout_batch = 1
```

此时 off-policy 版本应与 on-policy 结果近似一致。

### 验证2：正确性检查

确认：

* old_log_probs 在多个 epoch 中保持不变
* 当前策略 log_probs 会变化
* ratio 计算合理
* clip 生效

## 产物

* 代码实现
* 检查说明文档
* `epochs=1` 对照测试结果
* 若实现改动较多，给出关键 diff 说明

## 完成标志

`results/grpo_offpolicy_impl_check/notes.md` 中明确写出：实现已验证通过，可进入问题7。

---

## 15. 问题7：离线策略超参搜索（grpo_off_policy_sweep）

## 目标

找到最优的离线策略配置，并与 on-policy 做步数与墙钟时间双重对比。

## 阶段划分

### 阶段1：Broad Sweep

目的：快速识别最优区间

```yaml
rollout_batch_size: 256
n_grpo_steps: 50
epochs_per_rollout_batch: [1, 2, 4, 8]
train_batch_size: [128, 256, 512]
```

不要求跑满所有笛卡尔积，只跑最有信息量的关键点。

**至少覆盖以下配置：**

```yaml
(epoch=1, batch=256)
(epoch=2, batch=128)
(epoch=4, batch=128)
(epoch=8, batch=64)
(epoch=2, batch=256)
(epoch=4, batch=256)
```

如果某配置明显 OOM 或发散，立刻停止并记录。

### 阶段2：Fine Tune

基于阶段1最优区间继续搜索，跑完整 200 步。

例如若 `(epoch=4, batch=128)` 最优，则优先扩展：

```yaml
epochs_per_rollout_batch: [3, 4, 5, 6]
train_batch_size: [96, 128, 160, 192]
n_grpo_steps: 200
```

## 关键要求

### 必须记录两种横轴

1. `x = GRPO step`
2. `x = wall_clock_time`

因为离线策略的关键问题不是只看步数，而是要看**实际时间效率**。

## 判断规则

最优离线配置需同时满足：

1. 验证精度或 reward 明显优于 on-policy baseline，或
2. 在相近效果下，用更少墙钟时间达到同等性能

如果离线配置在步数上更好但时间上更差，必须在分析中如实写明。

## 产物

* 阶段1快速趋势图
* 阶段2完整曲线
* step 轴对比图
* wall-clock 轴对比图
* 最优离线配置总结
* 与 on-policy 的权衡分析

## 完成标志

将最优离线配置写入 `best_config.json`：

```json
{
  "offpolicy": {
    "epochs_per_rollout_batch": ...,
    "train_batch_size": ...,
    "gradient_accumulation_steps": ...
  }
}
```

---

## 16. 问题8：离线 clipping 消融（grpo_off_policy_clip_ablation）

## 目标

验证离线训练中的 clipping 是否必要。

## 配置

使用问题7最优离线配置。

```yaml
n_grpo_steps: 100
```

对比：

```yaml
A: loss_type = grpo_clip
B: loss_type = grpo_no_clip
```

## 实现要求

必须新增 `grpo_no_clip`：

```python
elif loss_type == "grpo_no_clip":
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    advantages_bc = advantages.expand_as(policy_log_probs)
    loss = -ratio * advantages_bc
```

## 重点观测

* reward 曲线平滑性
* gradient_norm
* loss 是否不稳定
* entropy
* response_length
* 是否发生 divergence

## 判断规则

若 `no_clip` 出现明显不稳定、reward 崩塌或梯度爆炸，则直接得出 clipping 必要。

## 产物

* reward 对比图
* gradient norm 对比图
* 结论：clipping 是否必要，为什么

---

## 17. 问题9：提示词消融（grpo_prompt_ablation）

## 目标

验证 prompt 设计对 GRPO 学习动态和最终性能的影响。

## 对比设置

```yaml
A: R1-Zero prompt
B: question_only prompt
```

训练和验证必须使用同一类 prompt，不允许 train/test prompt mismatch。

## 配置

优先使用问题7最优离线配置；若离线实现存在明显问题，则退回问题1-5的最优 on-policy 配置。

## 关键观测

* reward
* accuracy
* response_length
* entropy
* gradient_norm
* 收敛速度

## 判断规则

如果 R1 prompt 带来更长推理和更好最终效果，即使训练慢一些也可保留。

如果 question_only 明显更快但最终性能明显差，则在 leaderboard 中通常不优先。

## 产物

* reward 对比图
* response_length 对比图
* entropy 对比图
* 结论：哪个 prompt 更适合最终提交

## 完成标志

将 `prompt_style` 写入 `best_config.json`。

---

## 18. 问题10：leaderboard 最优化（4 小时墙钟）

## 目标

在 **4 小时墙钟时间** 限制内，使用前面实验得到的最优配置，在 **MATH full 5K val** 上获得尽可能高的准确率。

## 核心原则

* 不再做宽泛探索，改为**面向最终成绩的执行**。
* 所有选择以最终验证准确率为准。
* 优先使用已经被前面实验验证过的最佳组合。

## 默认首选配置

如果前面实验已完成，则默认使用：

* 问题1最优学习率
* 问题2最优 loss type
* 问题4最优长度标准化
* 问题5最优 std normalization
* 问题7最优离线超参
* 问题8得出的是否 clipping
* 问题9最优 prompt

## 若前面实验结论仍不明确

按如下 fallback：

### Fallback A

优先相信：

* 更稳定的配置
* 更少显存风险的配置
* 在 50–100 步里 already 最强的配置

### Fallback B

如离线策略并未表现出明显优势，则回退到最优 on-policy 配置直接冲 leaderboard。

## 时间分配

```text
0:00 - 0:15   最终配置确认、GPU预热、路径检查
0:15 - 3:15   正式训练
3:15 - 3:45   完整5K验证评估
3:45 - 4:00   图表、结果表、摘要整理
```

## 训练要求

* 实时记录每步 wall clock
* 定期保存 checkpoint
* 保留最佳 checkpoint
* 若发现中途配置明显失效，可在剩余时间内切换到 fallback 配置，但必须记录原因

## 最终必须输出

1. 最终准确率
2. 墙钟时间 vs 准确率曲线
3. 最优 checkpoint 的配置表
4. leaderboard 策略说明
5. 与前期实验结论的对应关系

## 成功标准

* `>= 15%`：合格
* `>= 20%`：强
* `>= 25%`：优秀

**但无论是否达到 25%，都必须完成一次真实的 4 小时 leaderboard 运行并给出最终报告。**

---

## 19. 动态决策树（执行中必须遵守）

### 如果问题1中所有学习率都发散

立即执行：

1. 补跑 `1e-6` 与 `2e-6`
2. 若仍发散，增加 warmup
3. 若仍发散，检查 reward、loss、实现是否有 bug

### 如果问题2两种 baseline 差异不明显

1. 看 gradient norm 是否不同
2. 若仍不明显，再补 100 步
3. 若仍不明显，选更简单、更稳定者

### 如果问题4两种长度标准化差异不明显

1. 优先选更稳定者
2. 再看 response length 是否更符合 math 推理需求
3. 若仍接近，优先 `masked_normalize`

### 如果问题5关闭 std normalization 后不稳定

则保留标准化，除非关闭后效果大幅提升且不致命。

### 如果问题6实现验证失败

不要进入问题7。先修正实现，直到 `epochs=1` 与 on-policy 近似一致。

### 如果问题7发现离线始终不如 on-policy

1. 先调小学习率重试最优离线配置
2. 再减少 epoch
3. 若仍不如，则后续 leaderboard 回到 on-policy

### 如果问题8中 no_clip 崩溃

直接定性为：clipping 必要，不再额外浪费时间。

### 如果问题9中 prompt 消融结果不明显

优先选 leaderboard 规则要求的 prompt 或更强者；若接近，则优先更短更快者仅在最终成绩不受损时使用。

---

## 20. 每完成一个问题后必须执行的动作

1. 更新 `best_config.json`
2. 更新 `decision_log.md`
3. 更新总表 `experiment_index.csv`
4. 检查下一个问题是否已具备依赖条件
5. 若具备，立刻进入下一个问题
6. 若不具备，先补最小必要实验

**不要在问题完成后停下来。**

---

## 21. 最终报告结构（必须按此输出）

文件：`Final_Report.md`

```markdown
# Executive Summary

# Experimental Setup
- GPU / model / dataset / shared constraints

# Q1. Learning Rate
- setup
- results
- conclusion

# Q2. Baseline Ablation
...

# Q3. Length Normalization Theory
...

# Q4. Length Normalization Experiment
...

# Q5. Group Std Normalization
...

# Q6. Off-Policy Implementation
...

# Q7. Off-Policy Sweep
...

# Q8. Clipping Ablation
...

# Q9. Prompt Ablation
...

# Q10. Leaderboard Run
- final config
- wall-clock constrained performance
- final accuracy

# Hyperparameter Evolution Table

# Resource Usage Summary

# Failure Cases and What Was Learned

# Final Recommendations
```

每一节至少包含：

* 本节目标
* 本节设置
* 本节结果
* 本节结论
* 对下游实验的影响

---

## 22. 任务闭环要求

你必须持续推进，直到满足 Definition of Done。

只有以下两种情况可以停止：

1. **所有实验、实现、图表、结论、最终报告都已完成**
2. **确实存在无法突破的外部阻塞**（例如数据缺失、代码库损坏、GPU 不可用），且你已经：

   * 明确记录阻塞点
   * 提供已完成成果
   * 提供下一步恢复建议

除此之外，不允许停止。

---

## 23. 一句话执行摘要

**先用最小成本找出最关键超参，再验证长度与标准化策略，随后完成离线实现与离线搜索，接着做必要消融，最后用前面得到的最优组合在 4 小时预算内完成 leaderboard，并把全部结果整理成可提交的最终报告。**
