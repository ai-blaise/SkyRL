# GSPO (Group Sequence Policy Optimization)

Sequence-level policy optimization with mean reduction across sequences.

## Overview

GSPO applies policy loss at the **sequence level** rather than the token level. This means the loss is computed as the mean over sequences, giving equal weight to each sequence regardless of length.

## Key Concept

Standard GRPO uses token-mean loss:
```
loss = mean(token_losses)  # Long sequences contribute more
```

GSPO uses sequence-mean loss:
```
loss = mean(sequence_losses)  # Each sequence weighted equally
```

This is important when you want all prompts/responses to contribute equally to training, regardless of response length.

## Quick Start

```bash
# GSPO builds on the base GSM8K config
python examples/gsm8k/gsm8k_dataset.py

export WANDB_API_KEY=<your_key_here>
bash examples/algorithms/gspo/run_gspo_gsm8k.sh
```

## Configuration

```yaml
# GSPO-specific settings
trainer.algorithm.policy_loss_type: "gspo"
trainer.algorithm.loss_reduction: "sequence_mean"

# Batch size considerations (GSPO may need smaller micro-batches)
trainer.micro_forward_batch_size_per_gpu: 16
trainer.micro_train_batch_size_per_gpu: 16
```

## When to Use GSPO

| Scenario | Recommendation |
|----------|---------------|
| Variable-length responses | GSPO (equal sequence weighting) |
| Fixed-length responses | Standard GRPO is fine |
| Short responses dominate | GSPO prevents bias toward long |
| Length-based rewards | Consider GSPO |

## Comparison

| Algorithm | Loss Reduction | Effect |
|-----------|---------------|--------|
| GRPO | token_mean | Long sequences have more influence |
| GSPO | sequence_mean | All sequences equal weight |
| Dr. GRPO | seq_mean_token_sum_norm | Normalized token-level |

## Related Documentation

- [GRPO Example](../../gsm8k/README.md)
- [Dr. GRPO](../drgrpo/README.md)
- [Algorithms Guide](../../../docs/ALGORITHMS.md)
