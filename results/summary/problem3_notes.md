# Problem 3: think_about_length_normalization

## Comparison: masked_mean vs masked_normalize

### masked_mean (averaging over unmasked tokens per sequence)
- **Behavior**: Each sequence's loss is averaged over its own number of response tokens
- **Pro**: Equal weighting per sequence regardless of length; each example contributes equally to the batch gradient
- **Con**: Implicitly upweights shorter sequences — a 4-token response gets per-token gradient of 1/4 vs 1/7 for a 7-token response (same advantage). This creates a length bias that can encourage the model to generate shorter responses

### masked_normalize (dividing by constant, e.g., max_gen_len)
- **Behavior**: Each sequence's loss sum is divided by the same constant (e.g., max_gen_len=1024)
- **Pro**: Uniform per-token gradient regardless of sequence length; no implicit length bias; longer correct responses get proportionally more total gradient (natural credit assignment)
- **Con**: Short sequences contribute less total gradient to the batch; if short correct answers are important, they receive less reinforcement signal

### When one approach is better:
- **masked_normalize is better when**: Response length varies significantly and you want to avoid length-reward hacking. In RL/GRPO settings, masked_mean can cause the model to game rewards by producing short outputs that receive disproportionate gradient per token.
- **masked_mean is better when**: All responses have similar lengths, or when you specifically want to ensure every example has equal influence on the batch gradient regardless of its length.
- **In GRPO context**: Since GRPO encourages exploration and correct responses tend to require reasoning chains of variable length, masked_normalize (constant normalizer) is generally preferred to avoid penalizing the model for generating longer, more detailed reasoning.
