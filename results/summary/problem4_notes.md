# Problem 4: grpo_length_normalization

## Results
| Normalization | Best Acc | Final Acc | Avg Grad Norm | Format OK Trend |
|---------------|----------|-----------|---------------|-----------------|
| masked_mean | 21.09% | 19.53% (step 80) | 0.0985 | 51% → 68% |
| masked_normalize | 19.53% | 17.97% (step 200) | 0.0132 | 49% → 59% |

## Findings
masked_mean (no length normalization) slightly outperforms masked_normalize in terms of peak accuracy (21.09% vs 19.53%), though masked_normalize is more stable (stays near initial 17.97% throughout).

The most notable trend is in **gradient norm stability**: masked_normalize produces ~7.5x smaller gradients (avg 0.013 vs 0.099), confirming its role as an implicit learning rate reducer. This makes training extremely stable but also slower to improve.

Format adherence improves with both approaches, but faster with masked_mean (68% vs 59% at comparable steps).

## Decision: Use masked_mean (no length normalization) for subsequent experiments
Rationale: Higher peak performance and faster format learning, while still maintaining stability.
