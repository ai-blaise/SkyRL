# Structured Output for RL

This guide explains how to use SGLang's structured output features for constrained action generation in RL training.

## Overview

Structured output (constrained decoding) ensures the model generates valid actions according to a predefined schema. This is useful for:

- **Discrete action RL**: Force output to be one of N valid actions
- **Tool calling**: Ensure valid JSON tool call format
- **Multi-step reasoning**: Enforce structured thought format
- **Game agents**: Constrain to valid game moves

## Supported Constraint Types

### 1. JSON Schema

Force output to match a JSON schema:

```python
sampling_params = {
    "json_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["move_up", "move_down", "move_left", "move_right", "pick", "drop"]
            },
            "target": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["action"]
    },
    "max_new_tokens": 100,
}
```

### 2. Regex Patterns

Constrain output to match a regex:

```python
# Only allow specific action format
sampling_params = {
    "regex": r"(move|pick|drop|use)\([a-z_]+\)",
    "max_new_tokens": 50,
}
```

### 3. EBNF Grammar

Use formal grammar for complex constraints:

```python
sampling_params = {
    "ebnf": """
        root ::= action
        action ::= move | interact
        move ::= "move(" direction ")"
        direction ::= "north" | "south" | "east" | "west"
        interact ::= "interact(" object ")"
        object ::= [a-z]+
    """,
    "max_new_tokens": 50,
}
```

## RL Environment Integration

### Example: Grid World Agent

```python
from skyrl_gym import Env

class GridWorldEnv(Env):
    ACTIONS = ["up", "down", "left", "right"]

    def get_sampling_params(self) -> dict:
        """Return sampling params with action constraints."""
        return {
            "json_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": self.ACTIONS}
                },
                "required": ["action"]
            },
            "max_new_tokens": 20,
            "temperature": 1.0,
        }

    def parse_action(self, response: str) -> str:
        """Parse constrained JSON response."""
        import json
        data = json.loads(response)
        return data["action"]
```

### Example: Tool Calling Agent

```python
sampling_params = {
    "json_schema": {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool": {
                "type": "string",
                "enum": ["search", "calculator", "code_exec", "none"]
            },
            "tool_input": {"type": "string"}
        },
        "required": ["thought", "tool"]
    },
    "max_new_tokens": 200,
}
```

## Action Masking (Token-Level)

For finer control, use token-level action masking:

```python
# Only allow specific token IDs
sampling_params = {
    "action_mask": [token_id_1, token_id_2, token_id_3],  # Valid token IDs
    "max_new_tokens": 1,  # Single token action
}

# Or block specific tokens
sampling_params = {
    "disallowed_tokens": [unsafe_token_1, unsafe_token_2],
    "max_new_tokens": 100,
}
```

## Configuration

Enable structured output backend in config:

```yaml
generator:
  # Structured output backend
  structured_output:
    # Backend options: "xgrammar" (fastest, CUDA), "outlines", "llguidance"
    backend: "xgrammar"

    # Cache grammars for faster reuse
    cache_grammars: true

    # Maximum grammar cache size
    max_cache_size: 1000
```

## Best Practices

### 1. Keep Schemas Simple

Simpler schemas = faster constrained decoding:

```python
# Good: Simple enum constraint
{"type": "string", "enum": ["a", "b", "c"]}

# Avoid: Complex nested schemas (slower)
{"type": "object", "properties": {...deeply nested...}}
```

### 2. Use Temperature > 0

Constrained decoding works with all temperatures, but T > 0 allows exploration:

```python
sampling_params = {
    "json_schema": {...},
    "temperature": 1.0,  # Allow exploration within constraints
}
```

### 3. Batch Consistent Schemas

When using n > 1 sampling, all samples use the same schema:

```python
sampling_params = {
    "json_schema": action_schema,
    "n": 4,  # 4 samples, all constrained to schema
}
```

### 4. Combine with RL Exploration

Use structured output for valid actions, entropy for exploration:

```python
sampling_params = {
    "json_schema": action_schema,
    "temperature": 1.2,  # Higher temp for exploration
    "top_p": 0.95,       # Nucleus sampling within constraints
}
```

## Troubleshooting

### Grammar Compilation Errors

If you see grammar errors, check:
1. JSON schema is valid
2. Regex is valid Python regex
3. EBNF follows xgrammar syntax

### Performance Issues

Constrained decoding adds overhead. Mitigate with:
1. Use `xgrammar` backend (CUDA-accelerated)
2. Enable grammar caching
3. Keep schemas simple

### Empty Outputs

If model produces empty output with constraints:
1. Check schema allows valid completions
2. Increase `max_new_tokens`
3. Lower `min_new_tokens` to 0

## See Also

- [SGLang Structured Output Docs](https://docs.sglang.ai/backend/structured_output.html)
- [RL Action Masking](./rl_logit_processors.py)
- [SkyRL Gym Environments](../skyrl-gym/skyrl_gym/envs/)
