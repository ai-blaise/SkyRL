"""
Comprehensive integration tests for SkyRL Gym environments with nmoe models.

This module tests the integration between SkyRL gym environments and nmoe
MoE (Mixture of Experts) models, covering:
1. Gym environment with nmoe policy
2. SQL environment with nmoe
3. Environment reset with nmoe state
4. Multi-step rollouts with nmoe
5. Environment vectorization with nmoe
6. Reward shaping with nmoe aux_loss
7. Observation/action space handling
8. Environment registration with nmoe

Run with:
    uv run --isolated --extra dev pytest tests/integration/test_gym_nmoe_integration.py -v

For GPU tests:
    uv run --isolated --extra dev pytest tests/integration/test_gym_nmoe_integration.py -v -m "integration and gpu"
"""

import pytest
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock
import json

import skyrl_gym
from skyrl_gym.core import Env, EnvStepOutput
from skyrl_gym.envs.registration import (
    EnvSpec,
    registry,
    register,
    make,
    spec,
)
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.tools.core import tool, ToolGroup
from skyrl_gym.metrics import default_aggregate_metrics
from skyrl_gym import error


# ==============================================================================
# Mock nmoe Components
# ==============================================================================


@dataclass
class MockNMoEConfig:
    """Mock nmoe config for testing without GPU."""
    dim: int = 256
    inter_dim: int = 512
    moe_inter_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_dense_layers: int = 1
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 2
    vocab_size: int = 1000
    max_position_embeddings: int = 2048
    batch_size: int = 4
    seq_len: int = 512
    dtype: str = "bf16"
    aux_loss_alpha: float = 0.001
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    attn: str = "mla"
    route_scale: float = 1.0
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class MockNMoEOutput:
    """Mock nmoe model output with aux_loss and expert loads."""

    def __init__(
        self,
        logits: "torch.Tensor",
        aux_loss: Optional["torch.Tensor"] = None,
        expert_loads: Optional[Dict[str, Any]] = None,
    ):
        self.logits = logits
        self.aux_loss = aux_loss
        self.expert_loads = expert_loads or {}


class MockNMoEPolicy:
    """Mock nmoe policy for gym environment integration tests."""

    def __init__(self, config: MockNMoEConfig):
        self.config = config
        self.last_aux_loss = None
        self.last_expert_loads = None
        self._generation_count = 0

    def generate(
        self,
        input_ids: Any,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Generate text response and return metadata."""
        self._generation_count += 1

        # Simulate nmoe-specific metadata
        self.last_aux_loss = 0.01 * self._generation_count
        self.last_expert_loads = {
            "mean_load": 0.125,  # 1/8 for 8 experts
            "load_imbalance": 0.05,
            "loads": [[0.12, 0.13, 0.11, 0.14, 0.12, 0.13, 0.12, 0.13]],
        }

        # Generate mock response based on input
        responses = [f"<think>Processing step {self._generation_count}</think>\n<sql>SELECT 1;</sql>"]

        metadata = {
            "aux_loss": self.last_aux_loss,
            "expert_loads": self.last_expert_loads,
            "tokens_generated": max_new_tokens,
        }

        return responses, metadata

    def get_router_aux_loss(self) -> float:
        """Get the aux loss from the last forward pass."""
        return self.last_aux_loss or 0.0

    def get_expert_load_stats(self) -> Dict[str, Any]:
        """Get expert load statistics."""
        return self.last_expert_loads or {}


class MockVectorizedNMoEPolicy:
    """Mock vectorized nmoe policy for parallel environment execution."""

    def __init__(self, config: MockNMoEConfig, num_envs: int = 4):
        self.config = config
        self.num_envs = num_envs
        self.policies = [MockNMoEPolicy(config) for _ in range(num_envs)]

    def batch_generate(
        self,
        batch_inputs: List[Any],
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """Generate responses for a batch of inputs."""
        all_responses = []
        all_metadata = []

        for i, (policy, inp) in enumerate(zip(self.policies, batch_inputs)):
            responses, metadata = policy.generate(inp, max_new_tokens, **kwargs)
            all_responses.append(responses)
            all_metadata.append(metadata)

        return all_responses, all_metadata

    def aggregate_aux_losses(self) -> float:
        """Aggregate aux losses across all policies."""
        return sum(p.get_router_aux_loss() for p in self.policies) / len(self.policies)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def nmoe_config():
    """Provide a mock nmoe configuration."""
    return MockNMoEConfig()


@pytest.fixture
def nmoe_policy(nmoe_config):
    """Provide a mock nmoe policy."""
    return MockNMoEPolicy(nmoe_config)


@pytest.fixture
def vectorized_nmoe_policy(nmoe_config):
    """Provide a mock vectorized nmoe policy."""
    return MockVectorizedNMoEPolicy(nmoe_config, num_envs=4)


@pytest.fixture
def clean_registry():
    """Fixture that provides a clean registry for testing and restores it after."""
    original_registry = registry.copy()
    yield registry
    registry.clear()
    registry.update(original_registry)


@pytest.fixture
def mock_db_file():
    """Mock database file existence."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists


@pytest.fixture
def mock_sqlite_connection():
    """Mock SQLite connection for SQL environment tests."""
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("result",)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_connect


# ==============================================================================
# NMoE-Aware Gym Environment Implementation
# ==============================================================================


class NMoETextEnv(BaseTextEnv):
    """A text environment that tracks nmoe-specific metrics."""

    def __init__(self, max_turns: int = 5, track_nmoe_metrics: bool = True):
        super().__init__()
        self.max_turns = max_turns
        self.track_nmoe_metrics = track_nmoe_metrics
        self.nmoe_aux_losses: List[float] = []
        self.nmoe_expert_loads: List[Dict[str, Any]] = []
        self.closed = False

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        done = self.turns >= self.max_turns or "<done>" in action

        # Calculate reward with potential nmoe aux_loss integration
        base_reward = 1.0 if done else 0.0

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"Response to: {action}"}],
            reward=base_reward,
            done=done,
            metadata={"turns": self.turns, "nmoe_tracked": self.track_nmoe_metrics},
        )

    def record_nmoe_metrics(self, aux_loss: float, expert_loads: Dict[str, Any]):
        """Record nmoe-specific metrics from policy execution."""
        if self.track_nmoe_metrics:
            self.nmoe_aux_losses.append(aux_loss)
            self.nmoe_expert_loads.append(expert_loads)

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment metrics including nmoe statistics."""
        metrics = {"turns": self.turns}
        if self.track_nmoe_metrics and self.nmoe_aux_losses:
            metrics["mean_aux_loss"] = sum(self.nmoe_aux_losses) / len(self.nmoe_aux_losses)
            if self.nmoe_expert_loads:
                imbalances = [
                    el.get("load_imbalance", 0.0)
                    for el in self.nmoe_expert_loads
                    if "load_imbalance" in el
                ]
                if imbalances:
                    metrics["mean_load_imbalance"] = sum(imbalances) / len(imbalances)
        return metrics

    def close(self):
        self.closed = True


class NMoERewardShapingEnv(BaseTextEnv):
    """Environment with reward shaping based on nmoe aux_loss."""

    def __init__(
        self,
        max_turns: int = 5,
        aux_loss_penalty_weight: float = 0.1,
        load_balance_bonus_weight: float = 0.05,
    ):
        super().__init__()
        self.max_turns = max_turns
        self.aux_loss_penalty_weight = aux_loss_penalty_weight
        self.load_balance_bonus_weight = load_balance_bonus_weight
        self.pending_aux_loss: Optional[float] = None
        self.pending_load_stats: Optional[Dict[str, Any]] = None

    def set_nmoe_stats(self, aux_loss: float, load_stats: Dict[str, Any]):
        """Set nmoe stats to be used in next reward calculation."""
        self.pending_aux_loss = aux_loss
        self.pending_load_stats = load_stats

    def _calculate_shaped_reward(self, base_reward: float) -> Tuple[float, Dict[str, float]]:
        """Calculate reward with nmoe-based shaping."""
        shaped_reward = base_reward
        components = {"base_reward": base_reward}

        # Penalty for high aux_loss (encourages load balancing)
        if self.pending_aux_loss is not None:
            aux_penalty = -self.aux_loss_penalty_weight * self.pending_aux_loss
            shaped_reward += aux_penalty
            components["aux_loss_penalty"] = aux_penalty

        # Bonus for balanced expert loads
        if self.pending_load_stats is not None:
            load_imbalance = self.pending_load_stats.get("load_imbalance", 1.0)
            # Bonus is higher when load_imbalance is lower
            balance_bonus = self.load_balance_bonus_weight * (1.0 - min(load_imbalance, 1.0))
            shaped_reward += balance_bonus
            components["balance_bonus"] = balance_bonus

        return shaped_reward, components

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        done = self.turns >= self.max_turns or "<done>" in action

        base_reward = 1.0 if done else 0.0
        shaped_reward, reward_components = self._calculate_shaped_reward(base_reward)

        # Reset pending stats
        self.pending_aux_loss = None
        self.pending_load_stats = None

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"Turn {self.turns}"}],
            reward=shaped_reward,
            done=done,
            metadata={"reward_components": reward_components},
        )


# ==============================================================================
# Section 1: Gym Environment with nmoe Policy Tests
# ==============================================================================


@pytest.mark.integration
class TestGymWithNMoEPolicy:
    """Tests for gym environment integration with nmoe policy."""

    def test_env_with_nmoe_policy_basic_step(self, nmoe_policy):
        """Test basic environment step with nmoe policy."""
        env = NMoETextEnv(max_turns=3)

        # Simulate nmoe policy generating action
        responses, metadata = nmoe_policy.generate(
            input_ids=[[1, 2, 3]],
            max_new_tokens=50,
        )

        # Step environment with generated action
        action = responses[0]
        result = env.step(action)

        assert not result["done"]
        assert result["reward"] == 0.0
        assert "turns" in result["metadata"]
        assert result["metadata"]["nmoe_tracked"] is True

    def test_env_records_nmoe_metrics_during_rollout(self, nmoe_policy):
        """Test that environment properly records nmoe metrics during rollout."""
        env = NMoETextEnv(max_turns=5, track_nmoe_metrics=True)

        for i in range(3):
            responses, metadata = nmoe_policy.generate(
                input_ids=[[1, 2, 3]],
                max_new_tokens=50,
            )

            # Record nmoe metrics
            env.record_nmoe_metrics(
                aux_loss=nmoe_policy.get_router_aux_loss(),
                expert_loads=nmoe_policy.get_expert_load_stats(),
            )

            env.step(responses[0])

        # Verify metrics were recorded
        assert len(env.nmoe_aux_losses) == 3
        assert len(env.nmoe_expert_loads) == 3
        assert all(loss > 0 for loss in env.nmoe_aux_losses)

    def test_env_nmoe_metrics_aggregation(self, nmoe_policy):
        """Test aggregation of nmoe metrics in environment."""
        env = NMoETextEnv(max_turns=5, track_nmoe_metrics=True)

        for i in range(3):
            responses, metadata = nmoe_policy.generate([[1, 2, 3]], 50)
            env.record_nmoe_metrics(
                nmoe_policy.get_router_aux_loss(),
                nmoe_policy.get_expert_load_stats(),
            )
            env.step(responses[0])

        metrics = env.get_metrics()

        assert "mean_aux_loss" in metrics
        assert "mean_load_imbalance" in metrics
        assert metrics["mean_aux_loss"] > 0
        assert 0 <= metrics["mean_load_imbalance"] <= 1.0

    def test_env_works_without_nmoe_tracking(self, nmoe_policy):
        """Test environment works when nmoe tracking is disabled."""
        env = NMoETextEnv(max_turns=3, track_nmoe_metrics=False)

        responses, _ = nmoe_policy.generate([[1]], 50)
        result = env.step(responses[0])

        assert result["metadata"]["nmoe_tracked"] is False
        assert env.nmoe_aux_losses == []
        assert env.nmoe_expert_loads == []


# ==============================================================================
# Section 2: SQL Environment with nmoe Tests
# ==============================================================================


@pytest.mark.integration
class TestSQLEnvWithNMoE:
    """Tests for SQL environment integration with nmoe."""

    def test_sql_env_creation_with_nmoe_context(self, mock_db_file, mock_sqlite_connection):
        """Test SQL environment can be created in nmoe context."""
        from omegaconf import DictConfig

        extras = {
            "db_id": "test_db",
            "reward_spec": {"ground_truth": "SELECT 1"},
            "data": "custom",
            "max_turns": 3,
        }
        env_config = DictConfig({"db_path": "/test/path"})

        env = skyrl_gym.make("text2sql", env_config=env_config, extras=extras)
        assert env is not None
        assert hasattr(env, "step")
        assert hasattr(env, "init")

    def test_sql_env_step_with_nmoe_generated_action(
        self, mock_db_file, mock_sqlite_connection, nmoe_policy
    ):
        """Test SQL environment step with nmoe-generated SQL action."""
        from omegaconf import DictConfig

        extras = {
            "db_id": "test_db",
            "reward_spec": {"ground_truth": "SELECT 1"},
            "data": "custom",
            "max_turns": 3,
        }
        env_config = DictConfig({"db_path": "/test/path"})
        env = skyrl_gym.make("text2sql", env_config=env_config, extras=extras)

        # Simulate nmoe generating SQL action
        responses, metadata = nmoe_policy.generate([[1, 2, 3]], 100)
        action = responses[0]  # Contains <sql>SELECT 1;</sql>

        result = env.step(action)

        assert "observations" in result
        assert "reward" in result
        assert "done" in result

    def test_sql_env_multi_turn_with_nmoe(
        self, mock_db_file, mock_sqlite_connection, nmoe_policy
    ):
        """Test multi-turn SQL interaction with nmoe policy."""
        from omegaconf import DictConfig

        extras = {
            "db_id": "test_db",
            "reward_spec": {"ground_truth": "SELECT 1"},
            "data": "custom",
            "max_turns": 5,
        }
        env_config = DictConfig({"db_path": "/test/path"})
        env = skyrl_gym.make("text2sql", env_config=env_config, extras=extras)

        aux_losses = []
        for i in range(3):
            responses, metadata = nmoe_policy.generate([[1]], 100)
            aux_losses.append(metadata.get("aux_loss", 0))

            if i < 2:
                action = f"<think>Step {i}</think>\n<sql>SELECT {i};</sql>"
            else:
                action = f"<think>Final</think>\n<solution>SELECT 1</solution>"

            result = env.step(action)
            if result["done"]:
                break

        # Verify aux_losses were tracked
        assert len(aux_losses) >= 2
        assert all(loss >= 0 for loss in aux_losses)


# ==============================================================================
# Section 3: Environment Reset with nmoe State Tests
# ==============================================================================


@pytest.mark.integration
class TestEnvResetWithNMoEState:
    """Tests for environment reset handling with nmoe model state."""

    def test_env_reset_clears_nmoe_metrics(self, nmoe_policy):
        """Test that environment reset clears nmoe metrics."""
        env = NMoETextEnv(max_turns=5, track_nmoe_metrics=True)

        # Run some steps
        for _ in range(2):
            responses, _ = nmoe_policy.generate([[1]], 50)
            env.record_nmoe_metrics(
                nmoe_policy.get_router_aux_loss(),
                nmoe_policy.get_expert_load_stats(),
            )
            env.step(responses[0])

        assert len(env.nmoe_aux_losses) == 2

        # Create new environment (simulating reset)
        env2 = NMoETextEnv(max_turns=5, track_nmoe_metrics=True)
        assert len(env2.nmoe_aux_losses) == 0
        assert env2.turns == 0

    def test_env_init_returns_correct_prompt(self, nmoe_policy):
        """Test environment init returns correct prompt for nmoe."""
        env = NMoETextEnv(max_turns=5)
        prompt = [{"role": "user", "content": "Write a SQL query"}]

        obs, info = env.init(prompt)

        assert obs == prompt
        assert isinstance(info, dict)

    def test_env_state_preserved_across_steps(self, nmoe_policy):
        """Test environment state is preserved across nmoe policy steps."""
        env = NMoETextEnv(max_turns=10, track_nmoe_metrics=True)

        states_snapshot = []
        for i in range(5):
            responses, metadata = nmoe_policy.generate([[1]], 50)
            env.record_nmoe_metrics(
                nmoe_policy.get_router_aux_loss(),
                nmoe_policy.get_expert_load_stats(),
            )
            env.step(responses[0])

            states_snapshot.append({
                "turns": env.turns,
                "num_aux_losses": len(env.nmoe_aux_losses),
                "num_expert_loads": len(env.nmoe_expert_loads),
            })

        # Verify state progression
        for i, state in enumerate(states_snapshot):
            assert state["turns"] == i + 1
            assert state["num_aux_losses"] == i + 1
            assert state["num_expert_loads"] == i + 1


# ==============================================================================
# Section 4: Multi-step Rollouts with nmoe Tests
# ==============================================================================


@pytest.mark.integration
class TestMultiStepRolloutsWithNMoE:
    """Tests for multi-step rollouts with nmoe policy."""

    def test_complete_episode_rollout(self, nmoe_policy):
        """Test complete episode rollout with nmoe policy."""
        env = NMoETextEnv(max_turns=5)

        episode_rewards = []
        episode_aux_losses = []
        done = False
        step_count = 0

        while not done and step_count < 10:
            responses, metadata = nmoe_policy.generate([[1]], 50)
            episode_aux_losses.append(metadata.get("aux_loss", 0))

            if step_count == 4:  # Last step
                action = responses[0].replace("</sql>", "") + " <done>"
            else:
                action = responses[0]

            result = env.step(action)
            episode_rewards.append(result["reward"])
            done = result["done"]
            step_count += 1

        assert done or step_count == env.max_turns
        assert len(episode_rewards) > 0
        assert len(episode_aux_losses) > 0

    def test_rollout_collects_trajectory_data(self, nmoe_policy):
        """Test rollout properly collects trajectory data for RL."""
        env = NMoETextEnv(max_turns=3, track_nmoe_metrics=True)

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "aux_losses": [],
            "expert_loads": [],
        }

        obs, _ = env.init([{"role": "user", "content": "Start"}])
        trajectory["observations"].append(obs)

        for i in range(3):
            responses, metadata = nmoe_policy.generate([[1]], 50)
            action = responses[0] if i < 2 else "<done>"

            env.record_nmoe_metrics(
                nmoe_policy.get_router_aux_loss(),
                nmoe_policy.get_expert_load_stats(),
            )

            result = env.step(action)

            trajectory["actions"].append(action)
            trajectory["rewards"].append(result["reward"])
            trajectory["dones"].append(result["done"])
            trajectory["aux_losses"].append(metadata.get("aux_loss", 0))
            trajectory["expert_loads"].append(metadata.get("expert_loads", {}))

            if result["observations"]:
                trajectory["observations"].append(result["observations"])

            if result["done"]:
                break

        assert len(trajectory["actions"]) == len(trajectory["rewards"])
        assert len(trajectory["aux_losses"]) == len(trajectory["actions"])

    def test_rollout_handles_early_termination(self, nmoe_policy):
        """Test rollout handles early termination correctly."""
        env = NMoETextEnv(max_turns=10)

        step_count = 0
        for i in range(10):
            responses, _ = nmoe_policy.generate([[1]], 50)

            if i == 2:
                action = "<done>"
            else:
                action = responses[0]

            result = env.step(action)
            step_count += 1

            if result["done"]:
                break

        assert step_count == 3  # Should terminate early at step 3
        assert env.turns == 3


# ==============================================================================
# Section 5: Environment Vectorization with nmoe Tests
# ==============================================================================


@pytest.mark.integration
class TestEnvVectorizationWithNMoE:
    """Tests for vectorized environments with nmoe."""

    def test_vectorized_env_batch_step(self, vectorized_nmoe_policy):
        """Test batch stepping across vectorized environments."""
        num_envs = 4
        envs = [NMoETextEnv(max_turns=5) for _ in range(num_envs)]

        # Generate batch of actions
        batch_inputs = [[[i]] for i in range(num_envs)]
        all_responses, all_metadata = vectorized_nmoe_policy.batch_generate(
            batch_inputs, max_new_tokens=50
        )

        # Step all environments
        results = []
        for env, responses in zip(envs, all_responses):
            result = env.step(responses[0])
            results.append(result)

        assert len(results) == num_envs
        assert all(not r["done"] for r in results)

    def test_vectorized_env_aggregates_aux_losses(self, vectorized_nmoe_policy):
        """Test that vectorized environments properly aggregate aux losses."""
        num_envs = 4
        envs = [NMoETextEnv(max_turns=5, track_nmoe_metrics=True) for _ in range(num_envs)]

        # Run steps
        batch_inputs = [[[i]] for i in range(num_envs)]
        all_responses, all_metadata = vectorized_nmoe_policy.batch_generate(
            batch_inputs, max_new_tokens=50
        )

        for env, responses, metadata, policy in zip(
            envs, all_responses, all_metadata, vectorized_nmoe_policy.policies
        ):
            env.record_nmoe_metrics(
                policy.get_router_aux_loss(),
                policy.get_expert_load_stats(),
            )
            env.step(responses[0])

        # Aggregate
        aggregated_aux_loss = vectorized_nmoe_policy.aggregate_aux_losses()

        assert aggregated_aux_loss > 0
        assert all(len(env.nmoe_aux_losses) == 1 for env in envs)

    def test_vectorized_env_independent_states(self, vectorized_nmoe_policy):
        """Test that vectorized environments maintain independent states."""
        num_envs = 4
        envs = [NMoETextEnv(max_turns=5) for _ in range(num_envs)]

        # Step different number of times for each env
        for i, env in enumerate(envs):
            for _ in range(i + 1):
                responses, _ = vectorized_nmoe_policy.policies[i].generate([[1]], 50)
                env.step(responses[0])

        # Verify independent state
        for i, env in enumerate(envs):
            assert env.turns == i + 1

    def test_vectorized_env_parallel_rollout(self, vectorized_nmoe_policy):
        """Test parallel rollout across vectorized environments."""
        num_envs = 4
        envs = [NMoETextEnv(max_turns=3) for _ in range(num_envs)]

        all_trajectories = [{"rewards": [], "aux_losses": []} for _ in range(num_envs)]

        for step in range(3):
            batch_inputs = [[[step]] for _ in range(num_envs)]
            all_responses, all_metadata = vectorized_nmoe_policy.batch_generate(
                batch_inputs, max_new_tokens=50
            )

            for i, (env, responses, metadata) in enumerate(
                zip(envs, all_responses, all_metadata)
            ):
                action = responses[0] if step < 2 else "<done>"
                result = env.step(action)
                all_trajectories[i]["rewards"].append(result["reward"])
                all_trajectories[i]["aux_losses"].append(metadata.get("aux_loss", 0))

        # Verify all trajectories collected data
        for traj in all_trajectories:
            assert len(traj["rewards"]) == 3
            assert len(traj["aux_losses"]) == 3


# ==============================================================================
# Section 6: Reward Shaping with nmoe aux_loss Tests
# ==============================================================================


@pytest.mark.integration
class TestRewardShapingWithNMoEAuxLoss:
    """Tests for reward shaping using nmoe aux_loss."""

    def test_reward_includes_aux_loss_penalty(self, nmoe_policy):
        """Test that reward includes aux_loss penalty."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            aux_loss_penalty_weight=0.1,
            load_balance_bonus_weight=0.0,
        )

        responses, metadata = nmoe_policy.generate([[1]], 50)
        env.set_nmoe_stats(
            aux_loss=0.5,
            load_stats={"load_imbalance": 0.1},
        )

        result = env.step(responses[0])

        # Base reward is 0.0 (not done), with aux_loss penalty
        expected_penalty = -0.1 * 0.5
        assert result["metadata"]["reward_components"]["aux_loss_penalty"] == expected_penalty

    def test_reward_includes_load_balance_bonus(self, nmoe_policy):
        """Test that reward includes load balance bonus."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            aux_loss_penalty_weight=0.0,
            load_balance_bonus_weight=0.1,
        )

        responses, _ = nmoe_policy.generate([[1]], 50)
        env.set_nmoe_stats(
            aux_loss=0.0,
            load_stats={"load_imbalance": 0.2},  # 80% balanced
        )

        result = env.step(responses[0])

        # Bonus = 0.1 * (1.0 - 0.2) = 0.08
        expected_bonus = 0.1 * (1.0 - 0.2)
        assert abs(result["metadata"]["reward_components"]["balance_bonus"] - expected_bonus) < 1e-6

    def test_combined_reward_shaping(self, nmoe_policy):
        """Test combined aux_loss penalty and load balance bonus."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            aux_loss_penalty_weight=0.1,
            load_balance_bonus_weight=0.05,
        )

        # Final step for base reward = 1.0
        for i in range(4):
            env.step(f"step {i}")

        responses, _ = nmoe_policy.generate([[1]], 50)
        env.set_nmoe_stats(
            aux_loss=0.3,
            load_stats={"load_imbalance": 0.1},
        )

        result = env.step("<done>")

        components = result["metadata"]["reward_components"]
        assert components["base_reward"] == 1.0
        assert components["aux_loss_penalty"] == -0.1 * 0.3
        assert abs(components["balance_bonus"] - 0.05 * 0.9) < 1e-6

        # Verify total
        expected_total = 1.0 + (-0.1 * 0.3) + (0.05 * 0.9)
        assert abs(result["reward"] - expected_total) < 1e-6

    def test_reward_shaping_without_nmoe_stats(self, nmoe_policy):
        """Test reward when nmoe stats are not provided."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            aux_loss_penalty_weight=0.1,
            load_balance_bonus_weight=0.05,
        )

        # Don't set nmoe stats
        result = env.step("action without nmoe stats")

        # Only base reward should be present
        assert "aux_loss_penalty" not in result["metadata"]["reward_components"]
        assert "balance_bonus" not in result["metadata"]["reward_components"]


# ==============================================================================
# Section 7: Observation/Action Space Handling Tests
# ==============================================================================


@pytest.mark.integration
class TestObservationActionSpaceHandling:
    """Tests for observation and action space handling with nmoe."""

    def test_conversation_observation_format(self):
        """Test that observations follow conversation format."""
        env = NMoETextEnv(max_turns=5)
        prompt = [{"role": "user", "content": "Initial prompt"}]

        obs, info = env.init(prompt)

        assert isinstance(obs, list)
        assert len(obs) == 1
        assert obs[0]["role"] == "user"
        assert "content" in obs[0]

    def test_step_returns_user_observation(self):
        """Test that step returns observation in user role."""
        env = NMoETextEnv(max_turns=5)
        result = env.step("<think>thinking</think>")

        assert len(result["observations"]) == 1
        assert result["observations"][0]["role"] == "user"

    def test_action_string_handling(self, nmoe_policy):
        """Test handling of various action string formats."""
        env = NMoETextEnv(max_turns=5)

        test_actions = [
            "<think>reasoning</think>\n<sql>SELECT 1;</sql>",
            "<think>step 1</think>",
            "plain text action",
            "",
        ]

        for action in test_actions:
            result = env.step(action)
            assert "observations" in result
            assert "reward" in result
            assert "done" in result

    def test_long_action_handling(self, nmoe_policy):
        """Test handling of very long actions."""
        env = NMoETextEnv(max_turns=5)

        long_action = "<think>" + "A" * 10000 + "</think>"
        result = env.step(long_action)

        assert result is not None
        assert "observations" in result

    def test_special_characters_in_action(self):
        """Test handling of special characters in actions."""
        env = NMoETextEnv(max_turns=5)

        special_actions = [
            "<sql>SELECT * FROM users WHERE name = 'O\\'Brien';</sql>",
            "<think>Unicode: \u4e2d\u6587</think>",
            "<sql>SELECT '\\n\\t\\r';</sql>",
        ]

        for action in special_actions:
            result = env.step(action)
            assert result is not None


# ==============================================================================
# Section 8: Environment Registration with nmoe Tests
# ==============================================================================


@pytest.mark.integration
class TestEnvRegistrationWithNMoE:
    """Tests for environment registration with nmoe support."""

    def test_register_nmoe_aware_env(self, clean_registry):
        """Test registering an nmoe-aware environment."""
        register(
            id="nmoe-text-env",
            entry_point=NMoETextEnv,
            kwargs={"max_turns": 10, "track_nmoe_metrics": True},
        )

        assert "nmoe-text-env" in registry
        env_spec = spec("nmoe-text-env")
        assert env_spec.id == "nmoe-text-env"

    def test_make_nmoe_aware_env(self, clean_registry):
        """Test making an nmoe-aware environment from registry."""
        register(
            id="nmoe-text-env-make",
            entry_point=NMoETextEnv,
            kwargs={"max_turns": 5},
        )

        env = make("nmoe-text-env-make")

        assert isinstance(env, NMoETextEnv)
        assert env.max_turns == 5
        assert env.track_nmoe_metrics is True  # default

    def test_make_nmoe_env_with_override_kwargs(self, clean_registry):
        """Test making nmoe environment with overridden kwargs."""
        register(
            id="nmoe-text-env-override",
            entry_point=NMoETextEnv,
            kwargs={"max_turns": 5, "track_nmoe_metrics": True},
        )

        env = make("nmoe-text-env-override", max_turns=10, track_nmoe_metrics=False)

        assert env.max_turns == 10
        assert env.track_nmoe_metrics is False

    def test_env_spec_contains_nmoe_kwargs(self, clean_registry):
        """Test that EnvSpec preserves nmoe-related kwargs."""
        register(
            id="nmoe-spec-test",
            entry_point=NMoETextEnv,
            kwargs={"max_turns": 5, "track_nmoe_metrics": True},
        )

        env = make("nmoe-spec-test")

        assert hasattr(env, "spec")
        assert env.spec.kwargs["max_turns"] == 5
        assert env.spec.kwargs["track_nmoe_metrics"] is True

    def test_register_reward_shaping_env(self, clean_registry):
        """Test registering reward shaping environment."""
        register(
            id="nmoe-reward-shaping",
            entry_point=NMoERewardShapingEnv,
            kwargs={
                "max_turns": 5,
                "aux_loss_penalty_weight": 0.1,
                "load_balance_bonus_weight": 0.05,
            },
        )

        env = make("nmoe-reward-shaping")

        assert isinstance(env, NMoERewardShapingEnv)
        assert env.aux_loss_penalty_weight == 0.1
        assert env.load_balance_bonus_weight == 0.05


# ==============================================================================
# Additional Integration Tests
# ==============================================================================


@pytest.mark.integration
class TestNMoEPolicyIntegrationPatterns:
    """Tests for common nmoe-policy integration patterns."""

    def test_policy_env_loop_pattern(self, nmoe_policy):
        """Test the standard policy-environment loop pattern."""
        env = NMoETextEnv(max_turns=5, track_nmoe_metrics=True)

        prompt = [{"role": "user", "content": "Start task"}]
        obs, info = env.init(prompt)

        total_reward = 0.0
        done = False

        while not done:
            # Policy generates action
            responses, metadata = nmoe_policy.generate([[1]], 50)
            action = responses[0]

            # Record nmoe metrics
            env.record_nmoe_metrics(
                nmoe_policy.get_router_aux_loss(),
                nmoe_policy.get_expert_load_stats(),
            )

            # Environment step
            result = env.step(action)
            total_reward += result["reward"]
            done = result["done"]

            if not done:
                obs = result["observations"]

        metrics = env.get_metrics()
        assert "mean_aux_loss" in metrics
        assert env.turns <= env.max_turns

    def test_batch_rollout_pattern(self, vectorized_nmoe_policy):
        """Test batch rollout pattern with vectorized environments."""
        num_envs = 4
        envs = [NMoETextEnv(max_turns=3, track_nmoe_metrics=True) for _ in range(num_envs)]

        # Initialize all environments
        prompts = [[{"role": "user", "content": f"Task {i}"}] for i in range(num_envs)]
        for env, prompt in zip(envs, prompts):
            env.init(prompt)

        dones = [False] * num_envs
        step = 0

        while not all(dones) and step < 10:
            # Batch generate
            batch_inputs = [[[step]] for _ in range(num_envs)]
            all_responses, all_metadata = vectorized_nmoe_policy.batch_generate(
                batch_inputs, max_new_tokens=50
            )

            # Batch step
            for i, (env, responses, metadata, policy) in enumerate(
                zip(envs, all_responses, all_metadata, vectorized_nmoe_policy.policies)
            ):
                if not dones[i]:
                    env.record_nmoe_metrics(
                        policy.get_router_aux_loss(),
                        policy.get_expert_load_stats(),
                    )
                    result = env.step(responses[0])
                    dones[i] = result["done"]

            step += 1

        # All environments should have metrics
        for env in envs:
            metrics = env.get_metrics()
            assert "turns" in metrics

    def test_grpo_style_grouped_rollout(self, nmoe_policy):
        """Test GRPO-style grouped rollout with multiple samples per prompt."""
        group_size = 4
        env = NMoETextEnv(max_turns=5, track_nmoe_metrics=True)

        prompt = [{"role": "user", "content": "Task"}]
        env.init(prompt)

        group_rewards = []
        group_aux_losses = []

        for sample_idx in range(group_size):
            # Generate with temperature for diversity
            responses, metadata = nmoe_policy.generate(
                [[1]], max_new_tokens=50, temperature=1.0
            )

            env.record_nmoe_metrics(
                nmoe_policy.get_router_aux_loss(),
                nmoe_policy.get_expert_load_stats(),
            )

            result = env.step(responses[0])
            group_rewards.append(result["reward"])
            group_aux_losses.append(metadata.get("aux_loss", 0))

        # Compute group statistics (as in GRPO)
        mean_reward = sum(group_rewards) / len(group_rewards)
        mean_aux_loss = sum(group_aux_losses) / len(group_aux_losses)

        assert len(group_rewards) == group_size
        assert mean_aux_loss > 0


@pytest.mark.integration
class TestEdgeCases:
    """Tests for edge cases in nmoe-gym integration."""

    def test_empty_observation_handling(self):
        """Test handling of empty observations."""
        env = NMoETextEnv(max_turns=5)

        # Force done state to get empty observations
        for _ in range(5):
            env.step("action")

        result = env.step("final action")
        assert result["done"] is True

    def test_zero_aux_loss_handling(self):
        """Test handling when aux_loss is zero."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            aux_loss_penalty_weight=0.1,
        )

        env.set_nmoe_stats(aux_loss=0.0, load_stats={})
        result = env.step("action")

        assert result["metadata"]["reward_components"]["aux_loss_penalty"] == 0.0

    def test_high_load_imbalance_handling(self):
        """Test handling of high load imbalance."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            load_balance_bonus_weight=0.1,
        )

        # Very imbalanced loads
        env.set_nmoe_stats(
            aux_loss=0.0,
            load_stats={"load_imbalance": 0.9},
        )

        result = env.step("action")

        # Bonus should be small for high imbalance
        assert result["metadata"]["reward_components"]["balance_bonus"] == 0.1 * 0.1

    def test_missing_load_stats(self):
        """Test handling when load_stats is missing expected keys."""
        env = NMoERewardShapingEnv(
            max_turns=5,
            load_balance_bonus_weight=0.1,
        )

        env.set_nmoe_stats(
            aux_loss=0.0,
            load_stats={},  # Missing load_imbalance
        )

        result = env.step("action")

        # Should default to 1.0 imbalance, giving 0 bonus
        assert result["metadata"]["reward_components"]["balance_bonus"] == 0.0

    def test_context_manager_cleanup(self, nmoe_policy):
        """Test that context manager properly cleans up."""
        with NMoETextEnv(max_turns=5) as env:
            responses, _ = nmoe_policy.generate([[1]], 50)
            env.step(responses[0])
            assert not env.closed

        assert env.closed


@pytest.mark.integration
class TestMetricsAggregation:
    """Tests for metrics aggregation across episodes."""

    def test_aggregate_nmoe_metrics_across_episodes(self, nmoe_policy):
        """Test aggregating nmoe metrics across multiple episodes."""
        all_metrics = []

        for episode in range(3):
            env = NMoETextEnv(max_turns=3, track_nmoe_metrics=True)

            for step in range(3):
                responses, metadata = nmoe_policy.generate([[1]], 50)
                env.record_nmoe_metrics(
                    nmoe_policy.get_router_aux_loss(),
                    nmoe_policy.get_expert_load_stats(),
                )
                env.step(responses[0])

            metrics = env.get_metrics()
            all_metrics.append(metrics)

        # Aggregate using default aggregator
        aggregated = default_aggregate_metrics(all_metrics)

        assert "turns" in aggregated
        assert "mean_aux_loss" in aggregated
        assert aggregated["turns"] == 3.0  # All episodes had 3 turns

    def test_aggregate_reward_components(self, nmoe_policy):
        """Test aggregating reward components across episodes."""
        reward_components_list = []

        for episode in range(4):
            env = NMoERewardShapingEnv(
                max_turns=2,
                aux_loss_penalty_weight=0.1,
                load_balance_bonus_weight=0.05,
            )

            responses, _ = nmoe_policy.generate([[1]], 50)
            env.set_nmoe_stats(
                aux_loss=0.1 * (episode + 1),
                load_stats={"load_imbalance": 0.1 * episode},
            )

            result = env.step(responses[0])
            reward_components_list.append(result["metadata"]["reward_components"])

        # Manual aggregation of components
        avg_penalty = sum(
            rc.get("aux_loss_penalty", 0) for rc in reward_components_list
        ) / len(reward_components_list)

        assert avg_penalty < 0  # Should be negative (penalty)
