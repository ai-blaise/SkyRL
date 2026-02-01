# Creating Custom Environments for SkyRL

**A guide to implementing custom reward functions and multi-turn environments.**

---

## Overview

SkyRL uses a Gymnasium-inspired environment system for computing rewards during RL training. Each environment:

1. Receives the LLM's output (action)
2. Computes a reward based on correctness/quality
3. Optionally provides feedback for multi-turn interactions
4. Returns done status

---

## Environment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Loop                           │
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐ │
│  │   Dataset   │ ──►  │  Generator  │ ──►  │   Environment   │ │
│  │             │      │             │      │                 │ │
│  │  - prompt   │      │  Calls LLM  │      │  - init()       │ │
│  │  - env_class│      │  via SGLang │      │  - step()       │ │
│  │  - reward_  │      │             │      │  - get_reward() │ │
│  │    spec     │      │             │      │                 │ │
│  └─────────────┘      └─────────────┘      └─────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start: Simple Single-Turn Environment

### Step 1: Create Environment Class

```python
# my_project/envs/simple_env.py

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
import re

class SimpleQAEnv(BaseTextEnv):
    """
    Simple Q&A environment with exact match reward.
    """

    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()

        # Ground truth is passed from dataset via extras
        assert "reward_spec" in extras, "reward_spec is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required"

        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _parse_answer(self, action: str) -> str:
        """Extract answer from LLM response."""
        # Look for pattern: **Answer: X**
        match = re.search(r"\*\*Answer:\s*(.+?)\*\*", action)
        return match.group(1).strip() if match else None

    def _compute_reward(self, answer: str) -> float:
        """Compute reward based on answer correctness."""
        if answer is None:
            return 0.0
        return 1.0 if answer.lower() == str(self.ground_truth).lower() else 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process LLM output and return reward.

        Args:
            action: The LLM's generated response

        Returns:
            BaseTextEnvStepOutput with observations, reward, done, metadata
        """
        self.turns += 1

        answer = self._parse_answer(action)
        reward = self._compute_reward(answer)

        # Single-turn: always done after one response
        return BaseTextEnvStepOutput(
            observations=[],        # No further observations needed
            reward=reward,
            done=True,
            metadata={"parsed_answer": answer}
        )
```

### Step 2: Create Dataset

```python
# my_project/create_dataset.py

from datasets import Dataset
import os

def create_qa_dataset():
    examples = [
        {
            "prompt": [
                {"role": "user", "content": "What is the capital of France? Reply with **Answer: <your answer>**"}
            ],
            "env_class": "simple_qa",
            "reward_spec": {
                "method": "rule",
                "ground_truth": "Paris"
            },
            "extra_info": {"topic": "geography"}
        },
        {
            "prompt": [
                {"role": "user", "content": "What is 2 + 2? Reply with **Answer: <your answer>**"}
            ],
            "env_class": "simple_qa",
            "reward_spec": {
                "method": "rule",
                "ground_truth": "4"
            },
            "extra_info": {"topic": "math"}
        },
        # ... more examples
    ]

    dataset = Dataset.from_list(examples)
    os.makedirs("~/data/simple_qa", exist_ok=True)
    dataset.to_parquet("~/data/simple_qa/train.parquet")

if __name__ == "__main__":
    create_qa_dataset()
```

### Step 3: Register and Run

```python
# my_project/main.py

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register

@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register custom environment
    register(
        id="simple_qa",
        entry_point="my_project.envs.simple_env:SimpleQAEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()

@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))

if __name__ == "__main__":
    main()
```

---

## BaseTextEnv Interface

All environments inherit from `BaseTextEnv`:

```python
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

class BaseTextEnv:
    """Base class for text-in, text-out RL environments."""

    def __init__(self):
        self.turns = 0          # Current turn counter
        self.max_turns = 1      # Maximum turns (set from extras)
        self.tool_groups = []   # Tool groups for tool-calling envs

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict]:
        """
        Initialize environment with prompt.

        Args:
            prompt: Initial chat history (list of message dicts)

        Returns:
            Tuple of (modified prompt, metadata dict)
        """
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process one LLM response.

        Args:
            action: LLM's generated text

        Returns:
            BaseTextEnvStepOutput containing:
              - observations: New messages to add to chat (or [] if done)
              - reward: Float reward value
              - done: Whether episode is complete
              - metadata: Optional dict with extra info
              - postprocessed_action: Optional modified action text
        """
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        """Return episode metrics for logging."""
        return {}

    def close(self):
        """Clean up resources."""
        pass
```

### BaseTextEnvStepOutput

```python
class BaseTextEnvStepOutput(TypedDict):
    observations: ConversationType  # List of {"role": str, "content": str}
    reward: float                   # Reward for this step
    done: bool                      # Episode complete?
    metadata: Dict[str, Any]        # Extra info for logging
    postprocessed_action: Optional[str]  # Modified action (optional)
```

---

## Reward Computation Patterns

### Pattern 1: Exact Match

```python
def _compute_reward(self, action: str) -> float:
    answer = self._parse_answer(action)
    return 1.0 if answer == self.ground_truth else 0.0
```

### Pattern 2: Partial Credit

```python
def _compute_reward(self, action: str) -> float:
    answer = self._parse_answer(action)

    if answer is None:
        return 0.0  # No answer found
    elif answer == self.ground_truth:
        return 1.0  # Exact match
    elif answer.lower() == self.ground_truth.lower():
        return 0.8  # Case-insensitive match
    else:
        return 0.0  # Wrong answer
```

### Pattern 3: Numerical Tolerance

```python
def _compute_reward(self, action: str) -> float:
    try:
        answer = float(self._parse_answer(action))
        expected = float(self.ground_truth)

        if abs(answer - expected) < 1e-6:
            return 1.0  # Exact
        elif abs(answer - expected) / abs(expected) < 0.01:
            return 0.8  # Within 1%
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0
```

### Pattern 4: LLM-as-Judge

```python
from openai import OpenAI

class LLMJudgeEnv(BaseTextEnv):
    def __init__(self, env_config, extras):
        super().__init__()
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _compute_reward(self, action: str) -> float:
        prompt = f"""
        Compare these two answers:
        Expected: {self.ground_truth}
        Actual: {action}

        Are they semantically equivalent? Reply with just "1" or "0".
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.0
```

### Pattern 5: Code Execution

```python
import subprocess
import tempfile

class CodeEnv(BaseTextEnv):
    def __init__(self, env_config, extras):
        super().__init__()
        self.test_cases = extras["reward_spec"]["test_cases"]

    def _compute_reward(self, action: str) -> float:
        # Extract code from response
        code = self._extract_code(action)
        if code is None:
            return 0.0

        # Run tests
        passed = 0
        for test_input, expected_output in self.test_cases:
            try:
                result = self._run_code(code, test_input)
                if result.strip() == expected_output.strip():
                    passed += 1
            except Exception:
                continue

        return passed / len(self.test_cases)

    def _run_code(self, code: str, input_data: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            result = subprocess.run(
                ['python', f.name],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
```

---

## Multi-Turn Environments

For environments that require multiple interaction turns:

```python
class MultiTurnMathEnv(BaseTextEnv):
    """
    Math environment with multiple attempts and feedback.
    """

    def __init__(self, env_config, extras):
        super().__init__()
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras.get("max_turns", 3)
        self.attempts = []

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        answer = self._parse_answer(action)
        self.attempts.append(answer)

        # Check if correct
        is_correct = answer is not None and answer == str(self.ground_truth)

        # Determine if done
        done = is_correct or self.turns >= self.max_turns

        if is_correct:
            # Correct answer
            return BaseTextEnvStepOutput(
                observations=[],
                reward=1.0,
                done=True,
                metadata={"attempts": self.attempts, "success": True}
            )

        if done:
            # Out of attempts
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"attempts": self.attempts, "success": False}
            )

        # Provide feedback for next turn
        if answer is None:
            feedback = "I couldn't find your answer. Please format it as \\boxed{answer}."
        else:
            feedback = f"'{answer}' is incorrect. You have {self.max_turns - self.turns} attempts remaining. Try again."

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": feedback}],
            reward=0.0,  # No intermediate reward
            done=False,
            metadata={"attempt": self.turns}
        )
```

---

## Tool-Calling Environments

For environments with tool use (e.g., SQL execution, web search):

```python
from skyrl_gym.tools import ToolGroup
import re

class SQLToolGroup(ToolGroup):
    """Tool group for executing SQL queries."""

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path

    def get_name(self) -> str:
        return "sql_executor"

    def get_tool_names(self) -> list:
        return ["execute_sql"]

    def execute(self, tool_name: str, tool_input: Any) -> str:
        if tool_name == "execute_sql":
            return self._execute_sql(tool_input)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_sql(self, query: str) -> str:
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(query)
            results = cursor.fetchall()
            conn.close()
            return str(results)
        except Exception as e:
            return f"Error: {e}"


class SQLEnv(BaseTextEnv):
    """Multi-turn SQL environment with tool execution."""

    def __init__(self, env_config, extras):
        super().__init__()
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.db_path = env_config.get("db_path", "database.db")
        self.max_turns = extras.get("max_turns", 5)

        # Initialize tool group
        self.sql_tool = SQLToolGroup(self.db_path)
        self.init_tool_groups([self.sql_tool])

        self.chat_history = []

    def _parse_sql(self, action: str) -> str:
        """Extract SQL from <sql>...</sql> tags."""
        match = re.search(r"<sql>(.*?)</sql>", action, re.DOTALL)
        return match.group(1).strip() if match else None

    def _is_final_answer(self, action: str) -> bool:
        """Check if response contains final answer."""
        return "<solution>" in action and "</solution>" in action

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        # Check for final answer
        if self._is_final_answer(action):
            reward = self._compute_final_reward()
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"history": self.chat_history}
            )

        # Check turn limit
        if self.turns >= self.max_turns:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"history": self.chat_history, "reason": "max_turns"}
            )

        # Execute SQL if present
        sql = self._parse_sql(action)
        if sql:
            result = self.sql_tool.execute("execute_sql", sql)
            observation = f"Query result: {result}"
        else:
            observation = "Please provide a SQL query in <sql>...</sql> tags."

        new_obs = {"role": "user", "content": observation}
        self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=0.0,
            done=False,
            metadata={}
        )

    def _compute_final_reward(self) -> float:
        # Compare final SQL result with ground truth
        # Implementation depends on your comparison logic
        pass
```

---

## Dataset Format

### Required Fields

```python
{
    "prompt": [                      # OpenAI chat format
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "env_class": "your_env_id",      # Registered environment ID
    "reward_spec": {
        "method": "rule",            # or "llm_judge", etc.
        "ground_truth": "expected"   # Passed to environment
    }
}
```

### Optional Fields

```python
{
    ...,
    "extra_info": {                  # Additional context
        "difficulty": "easy",
        "topic": "math",
        "source": "custom"
    },
    "db_id": "database_name",        # For SQL environments
    "data": {...}                    # Task-specific data
}
```

### Creating Parquet Dataset

```python
from datasets import Dataset
import pandas as pd

# From list of dicts
examples = [{"prompt": [...], "env_class": "...", ...}, ...]
dataset = Dataset.from_list(examples)
dataset.to_parquet("train.parquet")

# From pandas
df = pd.DataFrame(examples)
df.to_parquet("train.parquet")
```

---

## Environment Registration

### Method 1: Runtime Registration (Recommended)

```python
from skyrl_gym.envs import register

# In your main entry point
register(
    id="my_env",
    entry_point="my_package.envs.my_env:MyEnvClass",
)
```

### Method 2: Built-in Registration

Add to `skyrl_gym/envs/registration.py`:

```python
register(id="my_env", entry_point="skyrl_gym.envs.my_env.env:MyEnvClass")
```

---

## Configuration

### Training Config for Custom Environment

```yaml
# config.yaml

data:
  train_data:
    - "~/data/my_dataset/train.parquet"
  val_data:
    - "~/data/my_dataset/validation.parquet"

environment:
  env_class: my_env  # Must match registered ID

# Optional: Environment-specific config
skyrl_gym:
  my_env:
    db_path: "~/databases/my_db.sqlite"
    timeout: 30
```

### Accessing Environment Config

```python
class MyEnv(BaseTextEnv):
    def __init__(self, env_config, extras):
        super().__init__()

        # env_config comes from skyrl_gym.my_env in config
        self.db_path = env_config.get("db_path", "default.db")
        self.timeout = env_config.get("timeout", 10)
```

---

## Testing Your Environment

### Unit Test

```python
import pytest
from my_package.envs.my_env import MyEnv

def test_correct_answer():
    env = MyEnv(
        env_config={},
        extras={
            "reward_spec": {"ground_truth": "42"},
            "max_turns": 3
        }
    )

    result = env.step("The answer is **Answer: 42**")

    assert result["reward"] == 1.0
    assert result["done"] == True

def test_wrong_answer():
    env = MyEnv(
        env_config={},
        extras={
            "reward_spec": {"ground_truth": "42"},
            "max_turns": 3
        }
    )

    result = env.step("The answer is **Answer: 43**")

    assert result["reward"] == 0.0

def test_multi_turn():
    env = MyEnv(
        env_config={},
        extras={
            "reward_spec": {"ground_truth": "42"},
            "max_turns": 3
        }
    )

    # First turn - wrong answer
    result1 = env.step("**Answer: 41**")
    assert result1["done"] == False
    assert len(result1["observations"]) > 0  # Should have feedback

    # Second turn - correct answer
    result2 = env.step("**Answer: 42**")
    assert result2["done"] == True
    assert result2["reward"] == 1.0
```

### Integration Test

```python
def test_with_dataset():
    from datasets import Dataset
    from skyrl_gym.envs import register
    import skyrl_gym

    # Register env
    register(id="test_env", entry_point="my_package.envs.my_env:MyEnv")

    # Create mini dataset
    data = [
        {
            "prompt": [{"role": "user", "content": "What is 6*7?"}],
            "env_class": "test_env",
            "reward_spec": {"ground_truth": "42"}
        }
    ]

    # Test environment creation
    env = skyrl_gym.make("test_env", extras=data[0])
    result = env.step("The answer is **Answer: 42**")

    assert result["reward"] == 1.0
```

---

## Best Practices

### 1. Always Validate Inputs

```python
def __init__(self, env_config, extras):
    super().__init__()

    # Validate required fields
    assert "reward_spec" in extras, "reward_spec is required"
    assert "ground_truth" in extras["reward_spec"], "ground_truth is required"
```

### 2. Handle Edge Cases

```python
def _parse_answer(self, action: str) -> Optional[str]:
    if not action or not isinstance(action, str):
        return None

    # Handle multiple matches
    matches = re.findall(r"\\boxed\{(.+?)\}", action)
    return matches[-1] if matches else None  # Take last match
```

### 3. Log Useful Metadata

```python
def step(self, action: str) -> BaseTextEnvStepOutput:
    answer = self._parse_answer(action)
    reward = self._compute_reward(answer)

    return BaseTextEnvStepOutput(
        observations=[],
        reward=reward,
        done=True,
        metadata={
            "parsed_answer": answer,
            "expected": self.ground_truth,
            "turns_used": self.turns,
            "action_length": len(action)
        }
    )
```

### 4. Implement get_metrics for Custom Logging

```python
def get_metrics(self) -> Dict[str, Any]:
    return {
        "total_turns": self.turns,
        "success": self.success,
        "answer_found": self.answer_found
    }
```

### 5. Clean Up Resources

```python
def close(self):
    if hasattr(self, 'db_connection'):
        self.db_connection.close()
    if hasattr(self, 'temp_files'):
        for f in self.temp_files:
            os.remove(f)
```

---

## Example: Complete Custom Environment

See the [multiply example](../examples/multiply/) for a complete working implementation:

- `env.py` - Environment implementation
- `multiply_dataset.py` - Dataset creation
- `main_multiply.py` - Training entry point
- `run_multiply.sh` - Run script

---

## Troubleshooting

### Environment Not Found

```
KeyError: 'my_env'
```

**Fix:** Ensure registration happens before training starts:
```python
@ray.remote
def skyrl_entrypoint(cfg):
    register(id="my_env", ...)  # Register first!
    exp = BasePPOExp(cfg)
    exp.run()
```

### Reward Always 0

**Debug steps:**
1. Print parsed answer: `print(f"Parsed: {answer}, Expected: {self.ground_truth}")`
2. Check regex pattern matches actual output
3. Verify ground_truth type matches (string vs int)

### Multi-turn Not Working

**Fix:** Ensure `max_turns` is set:
```yaml
generator:
  max_turns: 5
  use_conversation_multi_turn: true
```

And in environment:
```python
def __init__(self, env_config, extras):
    super().__init__()
    self.max_turns = extras.get("max_turns", 5)
```

---

## Built-in Environments

SkyRL provides 7 built-in environments for common RL training tasks:

### gsm8k

**ID:** `gsm8k`
**Type:** Single-turn math
**Entry Point:** `skyrl_gym.envs.gsm8k.env:GSM8kEnv`

Grade-school math problems with exact match scoring.

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "What is 3 + 5? Put answer in \\boxed{}"}],
    "env_class": "gsm8k",
    "reward_spec": {
        "method": "rule",
        "ground_truth": "8"
    }
}
```

**Reward:** 1.0 for correct answer (parsed from `\boxed{}`), 0.0 otherwise.

**Example:**
```bash
bash examples/gsm8k/run_gsm8k.sh
```

---

### gsm8k_multi_turn

**ID:** `gsm8k_multi_turn`
**Type:** Multi-turn math with feedback
**Entry Point:** `skyrl_gym.envs.gsm8k.multi_turn_env:GSM8kMultiTurnEnv`

Multi-turn GSM8k where the model gets feedback and can retry.

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "Solve step by step: ..."}],
    "env_class": "gsm8k_multi_turn",
    "reward_spec": {"ground_truth": "42"},
    "extra_info": {"max_turns": 5}
}
```

**Configuration:**
```yaml
generator:
  max_turns: 5
  use_conversation_multi_turn: true
```

**Reward:**
- 1.0 for correct answer (terminates immediately)
- 0.2/max_turns for well-formatted but incorrect (encourages proper format)
- 0.0 for malformed responses

**Feedback:** Each turn provides observation like:
- "Please provide your step-by-step reasoning, and also include a tentative numeric answer..."
- Final turn: "Now provide only the final numeric answer in the exact format: '#### ANSWER'."

---

### aime

**ID:** `aime`
**Type:** Single-turn competition math
**Entry Point:** `skyrl_gym.envs.aime.env:AIMEEnv`

AIME (American Invitational Mathematics Examination) problems.

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "Find the value of x..."}],
    "env_class": "aime",
    "reward_model": {  # Note: uses reward_model, not reward_spec
        "ground_truth": "42"
    }
}
```

**Reward:** Computed via `utils.compute_score()` with accuracy metadata.

**Metadata:** Returns `{"acc": bool, "pred": str}` for analysis.

---

### search

**ID:** `search`
**Type:** Multi-turn search-augmented QA
**Entry Point:** `skyrl_gym.envs.search.env:SearchEnv`

Question answering with web search tool access.

**Prerequisites:**
```bash
# Start retrieval server first
bash examples/search/retriever/retrieval_launch.sh
```

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "What is the capital of France? Use <search>query</search> to search."}],
    "env_class": "search",
    "reward_spec": {"ground_truth": "Paris"},
    "max_turns": 3
}
```

**Configuration:**
```yaml
# config.yaml
skyrl_gym:
  search:
    search_url: "http://127.0.0.1:8000/retrieve"
    topk: 3
    timeout: 30
    log_requests: true

generator:
  max_turns: 3
  stop: ["</search>", "</answer>"]
```

**Tools:** Uses `SearchToolGroup` with `<search>query</search>` format.

**Response Format:**
- Search: `<search>query here</search>` → Returns `<information>Doc 1: ...</information>`
- Answer: `<answer>final answer</answer>` → Terminates episode

**Reward:** Computed on full chat history when `<answer>` is provided.

---

### text2sql

**ID:** `text2sql`
**Type:** Multi-turn SQL generation
**Entry Point:** `skyrl_gym.envs.sql.env:SQLEnv`

SQL query generation with execution feedback.

**Prerequisites:**
- Database files for Spider, Bird, or SynSQL datasets

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "Find all customers..."}],
    "env_class": "text2sql",
    "db_id": "customers_db",
    "data": "spider",  # or "bird", "synsql"
    "reward_spec": {"ground_truth": "SELECT * FROM customers"},
    "max_turns": 5
}
```

**Configuration:**
```yaml
skyrl_gym:
  text2sql:
    db_path: "/path/to/databases"

generator:
  max_turns: 5
  stop: ["</sql>", "</solution>"]
```

**Response Format:**
- Execute SQL: `<sql>SELECT * FROM customers</sql>` → Returns query results
- Final answer: `<solution>SELECT * FROM customers</solution>` → Terminates

**Reward:** Execution accuracy comparing predicted vs gold SQL results.

**Supported Datasets:**
| Dataset | DB Path Structure |
|---------|-------------------|
| Spider | `{db_path}/spider/database/{db_id}/{db_id}.sqlite` |
| Bird | `{db_path}/bird/train/train_databases/{db_id}/{db_id}.sqlite` |
| SynSQL | `{db_path}/SynSQL-2.5M/databases/{db_id}/{db_id}.sqlite` |

---

### lcb (LiveCodeBench)

**ID:** `lcb`
**Type:** Single-turn code generation
**Entry Point:** `skyrl_gym.envs.lcb.env:LCBEnv`

Code generation with test case execution.

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "Write a function that..."}],
    "env_class": "lcb",
    "reward_spec": {
        "ground_truth": "[{\"input\": \"1 2\", \"output\": \"3\"}, ...]"  # JSON test cases
    }
}
```

**Reward:** Fraction of test cases passed (0.0 to 1.0).

**Metadata:** Returns `{"parsed_code": str}` with extracted code.

---

### searchcode

**ID:** `searchcode`
**Type:** Multi-turn with search + code execution
**Entry Point:** `skyrl_gym.envs.searchcode.env:SearchCodeEnv`

Combined search and Python code execution for complex reasoning tasks.

**Dataset Format:**
```python
{
    "prompt": [{"role": "user", "content": "Research and compute..."}],
    "env_class": "searchcode",
    "reward_spec": {"ground_truth": "42"},
    "max_turns": 5
}
```

**Tools:** Uses both `SearchToolGroup` and `PythonCodeExecutorToolGroup`.

**Response Format:**
```
<tool><search>query here</search></tool>  # Search
<tool><python>print(2+2)</python></tool>  # Code execution
<solution>final answer</solution>          # Terminates
```

**Reward:** Computed using GSM8k-style answer extraction on full chat history.

---

## Built-in Tool Groups

SkyRL provides 3 built-in tool groups for environment tool-calling:

### SearchToolGroup

**Location:** `skyrl_gym.tools.search`

Web search via retrieval server.

**Configuration:**
```python
from skyrl_gym.tools import SearchToolGroup

tool_group = SearchToolGroup(
    search_url="http://127.0.0.1:8000/retrieve",  # Retrieval server URL
    topk=3,                                        # Number of results
    timeout=30,                                    # Request timeout (seconds)
    log_requests=True                              # Log API calls
)
```

**Tool Name:** `search`

**Input:** Query string
**Output:** JSON with formatted search results

**Features:**
- Connection pooling (shared sessions)
- Automatic retries (up to 10 attempts)
- Exponential backoff on failures

---

### PythonCodeExecutorToolGroup

**Location:** `skyrl_gym.tools.python`

Execute Python code snippets.

**Configuration:**
```python
from skyrl_gym.tools import PythonCodeExecutorToolGroup

tool_group = PythonCodeExecutorToolGroup(
    timeout=10.0  # Execution timeout (seconds)
)
```

**Tool Name:** `python`

**Input:** Python code string
**Output:** stdout or error message

**Security Note:** Executes code in subprocess. Use with caution in production.

---

### SQLCodeExecutorToolGroup

**Location:** `skyrl_gym.tools.sql`

Execute SQL queries against SQLite databases.

**Configuration:**
```python
from skyrl_gym.tools import SQLCodeExecutorToolGroup

tool_group = SQLCodeExecutorToolGroup(
    db_file_path="/path/to/databases"  # Base path for database files
)
```

**Tool Name:** `sql`

**Input:** `(db_id, sql_query, turns_left)`
**Output:** Query results as formatted string

**Features:**
- Transaction rollback (read-only queries)
- Timeout protection (default 5 seconds)
- Result truncation (max 50 rows for long outputs)

---

## Environment Selection Guide

```
What type of task?

Math/Reasoning problems?
├── Single answer → gsm8k
├── Step-by-step with feedback → gsm8k_multi_turn
└── Competition-level → aime

Code generation?
├── With test cases → lcb
└── As part of reasoning → searchcode

Information retrieval?
├── Search-based QA → search
└── Search + code execution → searchcode

Database queries?
└── SQL generation → text2sql

Need custom logic?
└── Create custom environment (see above)
```

---

## Environment Configuration Summary

| Environment | Multi-turn | Tools | Stop Strings | Config Key |
|-------------|------------|-------|--------------|------------|
| `gsm8k` | No | None | None | - |
| `gsm8k_multi_turn` | Yes | None | None | - |
| `aime` | No | None | None | - |
| `search` | Yes | SearchToolGroup | `</search>`, `</answer>` | `skyrl_gym.search` |
| `text2sql` | Yes | SQLCodeExecutorToolGroup | `</sql>`, `</solution>` | `skyrl_gym.text2sql` |
| `lcb` | No | None | None | - |
| `searchcode` | Yes | Search + Python | `</tool>`, `</solution>` | - |
