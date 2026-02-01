# LLM as a Judge

Train models using an external LLM to provide reward signals instead of rule-based exact matching.

## When to Use LLM Judge

Use LLM judge when:
- Exact-match rewards are too strict
- Evaluating open-ended responses
- Multiple valid answers exist
- Semantic correctness matters more than exact format

Use rule-based rewards when:
- Clear correct/incorrect answers
- Speed is critical (no API calls)
- Cost is a concern

## How It Works

1. Model generates response to problem
2. External LLM (judge) evaluates response quality
3. Judge returns reward score (0.0 - 1.0)
4. Policy is updated based on judge feedback

## Files

| File | Purpose |
|------|---------|
| `gsm8k_dataset_judge.py` | Dataset preparation with judge prompts |
| `llm_judge_env.py` | Environment using LLM for rewards |
| `main_llm_judge.py` | Training entrypoint |
| `run_llm_judge.sh` | Training script |

## Quick Start

```bash
# 1. Prepare dataset
python examples/llm_as_a_judge/gsm8k_dataset_judge.py

# 2. Run training
bash examples/llm_as_a_judge/run_llm_judge.sh
```

## Configuration

Set up the judge model:

```python
# In llm_judge_env.py
JUDGE_MODEL = "gpt-4"  # or local model endpoint
```

For local judge:
```yaml
environment:
  judge_endpoint: "http://localhost:8000/v1/chat/completions"
  judge_model: "Qwen/Qwen2.5-72B-Instruct"
```

## Cost Considerations

- Each reward computation requires a judge API call
- For N samples per prompt with M prompts: N × M API calls per step
- Consider local judge models for cost reduction

## Related Documentation

- [Custom Environments Guide](../../docs/CUSTOM_ENVIRONMENTS.md)
- [GSM8K Example](../gsm8k/README.md)
