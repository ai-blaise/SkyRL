# Multi-Turn Training Guide

This guide covers training agents that interact over multiple turns (conversations, tool use, multi-step reasoning).

---

## Quick Reference

| Parameter | Required Value | Why |
|-----------|---------------|-----|
| `use_conversation_multi_turn` | `true` | **REQUIRED for SGLang backend** |
| `max_turns` | 1-10+ | Maximum conversation turns |
| `generator.backend` | `sglang` or `vllm` | Inference backend |

---

## When to Use Multi-Turn Training

Use multi-turn training when your task involves:

- **Conversational agents** - Chatbots, assistants
- **Tool-using agents** - SQL, code execution, search
- **Multi-step reasoning** - Chain-of-thought with feedback
- **Interactive tasks** - Games, simulations, environments

**Single-turn** is sufficient for:
- Direct Q&A (GSM8K basic)
- One-shot generation tasks
- Simple classification/extraction

---

## Configuration

### Basic Multi-Turn Setup

```yaml
generator:
  backend: sglang
  use_conversation_multi_turn: true  # REQUIRED for SGLang
  max_turns: 5                        # Maximum turns per episode

  sampling_params:
    max_generate_length: 1024
    temperature: 1.0
```

### SGLang Requirement

**CRITICAL:** When using SGLang backend, `use_conversation_multi_turn` must be `true`:

```yaml
generator:
  backend: sglang
  use_conversation_multi_turn: true  # SGLang REQUIRES this
```

If set to `false` with SGLang, you'll get:
```
NotImplementedError: `use_conversation_multi_turn=False` is not supported for SGLang backend
```

---

## Multi-Turn Examples

### 1. GSM8K with Turn-Level Rewards

Location: `examples/turn_level_rewards/`

Rewards are assigned per turn:
- Correct answer: `1.0`
- Well-formatted response: `0.2 / max_turns`
- Otherwise: `0.0`

```bash
# Prepare dataset
python examples/turn_level_rewards/gsm8k_multi_turn_dataset.py

# Train
bash examples/turn_level_rewards/run_gsm8k_multi_turn.sh
```

### 2. Text-to-SQL (Multi-Turn Tool Use)

Location: `examples/text_to_sql/`

Agent iteratively refines SQL queries based on execution feedback.

```bash
# See examples/text_to_sql/sql.md for setup
python -m skyrl_train.entrypoints.main_base \
  +experiment=text_to_sql \
  generator.backend=sglang \
  generator.max_turns=5
```

### 3. Search Agent

Location: `examples/search/`

Multi-turn search with FAISS retriever backend.

```bash
# Start retriever server first
# See examples/search/README.md for setup
```

### 4. Mini-SWE Agent

Location: `examples/mini_swe_agent/`

Software engineering agent with code execution feedback.

---

## Creating Custom Multi-Turn Environments

### Basic Structure

```python
from skyrl_train.environments.base_environment import BaseEnvironment

class MyMultiTurnEnv(BaseEnvironment):
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.current_turn = 0

    def step(self, response: str, ground_truth: dict) -> tuple:
        self.current_turn += 1

        # Check if task is complete
        if self.is_correct(response, ground_truth):
            return 1.0, None, True  # reward, observation, done

        # Continue with feedback
        if self.current_turn < self.max_turns:
            feedback = self.get_feedback(response)
            return 0.0, feedback, False  # partial reward, next obs, not done

        # Max turns reached
        return 0.0, None, True

    def reset(self):
        self.current_turn = 0
```

### Reward Distribution Patterns

**Final reward only:**
```python
def step(self, response, ground_truth):
    if self.is_correct(response, ground_truth):
        return 1.0, None, True
    if self.current_turn >= self.max_turns:
        return 0.0, None, True
    return 0.0, self.get_feedback(response), False
```

**Turn-level rewards:**
```python
def step(self, response, ground_truth):
    if self.is_correct(response, ground_truth):
        return 1.0, None, True
    if self.is_well_formatted(response):
        return 0.2 / self.max_turns, self.get_feedback(response), False
    return 0.0, self.get_feedback(response), False
```

**Intermediate progress rewards:**
```python
def step(self, response, ground_truth):
    progress = self.calculate_progress(response, ground_truth)
    if progress >= 1.0:
        return 1.0, None, True
    if self.current_turn >= self.max_turns:
        return progress, None, True  # Partial credit
    return progress * 0.1, self.get_feedback(response), False
```

---

## Configuration Details

### use_conversation_multi_turn Explained

When `true` (required for SGLang):
- Each turn is stored as a separate message in conversation
- Format: `[system, user, assistant, user, assistant, ...]`
- Enables prefix caching (RadixAttention)

When `false` (vLLM only):
- All turns stored in single assistant message
- Format: `[system, user, assistant (multi-turn content)]`
- No prefix caching benefit

### max_turns Setting

```yaml
generator:
  max_turns: 5  # Adjust based on task complexity
```

Guidelines:
- Simple tasks: 2-3 turns
- Medium complexity: 5-7 turns
- Complex agents: 10+ turns

**Warning:** More turns = more generation time and memory usage.

---

## Troubleshooting

### "use_conversation_multi_turn=False not supported"

**Cause:** SGLang backend requires multi-turn mode.

**Fix:**
```yaml
generator:
  use_conversation_multi_turn: true
```

### Conversations Not Continuing

**Cause:** Environment returning `done=True` too early.

**Fix:** Check your environment's `step()` method returns `done=False` when feedback should continue.

### Memory Issues with Many Turns

**Cause:** Long conversation histories consume KV cache.

**Fix:**
```yaml
generator:
  gpu_memory_utilization: 0.7  # Lower to leave headroom
  max_num_seqs: 256            # Reduce concurrent sequences
```

### Inconsistent Rewards Across Turns

**Cause:** Reward signal is sparse or delayed.

**Fix:** Consider turn-level rewards to provide learning signal at each step:
```python
# Give small reward for progress, large reward for completion
intermediate_reward = 0.1 if made_progress else 0.0
final_reward = 1.0 if correct else 0.0
```

---

## Best Practices

1. **Start simple**: Begin with 2-3 turns, increase as needed
2. **Dense rewards**: Provide feedback at each turn when possible
3. **Clear stopping**: Define clear success/failure conditions
4. **Prefix caching**: Use SGLang for multi-turn to benefit from RadixAttention
5. **Memory management**: Monitor GPU memory with long conversations

---

## Related Documentation

- [Custom Environments Guide](./CUSTOM_ENVIRONMENTS.md) - Full environment creation guide
- [SGLang Integration Guide](./SGLANG_INTEGRATION_GUIDE.md) - SGLang configuration
- [Turn-Level Rewards Example](../examples/turn_level_rewards/README.md) - GSM8K multi-turn
- [Text-to-SQL Example](../examples/text_to_sql/) - Tool-use multi-turn
- [Search Agent Example](../examples/search/README.md) - Retrieval-augmented agent
