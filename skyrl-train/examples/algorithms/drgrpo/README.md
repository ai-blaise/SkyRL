# Dr. GRPO (Token-Normalized GRPO)

GRPO variant with sequence-mean token-sum normalization for better token-level credit assignment.

## Overview

Dr. GRPO modifies the loss reduction strategy to normalize at the token level while maintaining sequence-level grouping. This addresses issues where long sequences can dominate training or short sequences are under-weighted.

## Key Concept

Standard GRPO loss:
```
loss = mean(token_losses)  # Simple mean
```

Dr. GRPO loss:
```
loss = mean_over_sequences(sum_over_tokens(token_loss) / num_tokens_in_seq)
```

This ensures:
1. Each sequence contributes equally (sequence mean)
2. Token contributions within each sequence are normalized (token sum normalized)

## Quick Start

```bash
# 1. Prepare dataset
python examples/gsm8k/gsm8k_dataset.py

# 2. Run Dr. GRPO training
export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/drgrpo/run_drgrpo_gsm8k.sh
```

## Configuration

```yaml
# Dr. GRPO specific settings
trainer.algorithm.advantage_estimator: "grpo"
trainer.algorithm.loss_reduction: "seq_mean_token_sum_norm"
trainer.algorithm.grpo_norm_by_std: false  # Disable std normalization
trainer.algorithm.use_kl_loss: false
```

## Loss Reduction Options

| Option | Formula | Use Case |
|--------|---------|----------|
| `token_mean` | `mean(all_tokens)` | Standard GRPO |
| `sequence_mean` | `mean(seq_means)` | GSPO |
| `seq_mean_token_sum_norm` | `mean(normalized_seq_sums)` | Dr. GRPO |

## When to Use Dr. GRPO

| Scenario | Recommendation |
|----------|---------------|
| Variable response lengths | Dr. GRPO (normalized) |
| Token-level rewards | Dr. GRPO |
| Sequence-level rewards only | Standard GRPO is fine |
| Math/reasoning tasks | Dr. GRPO often better |

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `loss_reduction` | `seq_mean_token_sum_norm` | Dr. GRPO normalization |
| `grpo_norm_by_std` | `false` | Don't normalize by std (done differently) |
| `use_kl_loss` | `false` | KL handled via reward |

## Comparison

| Algorithm | Normalization | Best For |
|-----------|--------------|----------|
| GRPO | Token mean | General |
| GSPO | Sequence mean | Equal sequence weight |
| Dr. GRPO | Seq-mean token-sum-norm | Token-level fairness |

## Related Documentation

- [GRPO Example](../../gsm8k/README.md)
- [GSPO Example](../gspo/README.md)
- [Algorithms Guide](../../../docs/ALGORITHMS.md)
