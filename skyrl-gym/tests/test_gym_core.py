"""
Tests for SkyRL Gym core functionality.

Run with:
    uv run --isolated --extra dev pytest tests/test_gym_core.py -v
"""

import pytest
from typing import Any, Dict, Tuple, List
from unittest.mock import MagicMock, patch
import json

import skyrl_gym
from skyrl_gym.core import Env, EnvStepOutput
from skyrl_gym.envs.registration import (
    EnvSpec,
    registry,
    register,
    make,
    spec,
    pprint_registry,
    load_env_creator,
    _find_spec,
    _check_spec_register,
)
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.tools.core import tool, ToolGroup
from skyrl_gym.metrics import default_aggregate_metrics, aggregate_for_environment
from skyrl_gym import error


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def clean_registry():
    """Fixture that provides a clean registry for testing and restores it after."""
    original_registry = registry.copy()
    # Clear custom entries but keep built-in ones
    yield registry
    # Restore original registry
    registry.clear()
    registry.update(original_registry)


@pytest.fixture
def mock_env_class():
    """Create a mock environment class for testing."""

    class MockEnv(Env):
        def __init__(self, config_value: str = "default"):
            super().__init__()
            self.config_value = config_value
            self.closed = False

        def step(self, action: str) -> EnvStepOutput:
            return EnvStepOutput(
                observations=f"Observed: {action}",
                reward=1.0,
                done=False,
                metadata={"action": action},
            )

        def init(self, *kwargs) -> Tuple[str, Dict[str, Any]]:
            return "initial observation", {"initialized": True}

        def close(self):
            self.closed = True

    return MockEnv


@pytest.fixture
def mock_text_env_class():
    """Create a mock text-based environment for testing."""

    class MockTextEnv(BaseTextEnv):
        def __init__(self, max_turns: int = 5):
            super().__init__()
            self.max_turns = max_turns
            self.closed = False

        def step(self, action: str) -> BaseTextEnvStepOutput:
            self.turns += 1
            done = self.turns >= self.max_turns or "<done>" in action
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": f"Response to: {action}"}],
                reward=1.0 if done else 0.0,
                done=done,
                metadata={"turns": self.turns},
            )

        def close(self):
            self.closed = True

    return MockTextEnv


# ==============================================================================
# Section 1: Base Env Class Tests (core.py)
# ==============================================================================


class TestBaseEnvNotImplemented:
    """Tests for base Env class NotImplementedError behavior."""

    def test_step_raises_not_implemented(self):
        """Test that base Env.step() raises NotImplementedError."""
        env = Env()
        with pytest.raises(NotImplementedError) as exc_info:
            env.step("action")
        assert "step()" in str(exc_info.value)
        assert "must be implemented by subclasses" in str(exc_info.value)

    def test_init_raises_not_implemented(self):
        """Test that base Env.init() raises NotImplementedError."""
        env = Env()
        with pytest.raises(NotImplementedError) as exc_info:
            env.init()
        assert "init()" in str(exc_info.value)
        assert "must be implemented by subclasses" in str(exc_info.value)

    def test_close_does_not_raise(self):
        """Test that base Env.close() does not raise (default pass)."""
        env = Env()
        # Should not raise
        env.close()

    def test_str_representation(self):
        """Test that __str__ returns expected format."""
        env = Env()
        assert str(env) == "Env(Env)"


class TestEnvContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_enter_returns_self(self, mock_env_class):
        """Test that __enter__ returns the environment instance."""
        env = mock_env_class()
        with env as e:
            assert e is env

    def test_context_manager_calls_close_on_exit(self, mock_env_class):
        """Test that __exit__ calls close()."""
        env = mock_env_class()
        assert not env.closed
        with env:
            pass
        assert env.closed

    def test_context_manager_calls_close_on_exception(self, mock_env_class):
        """Test that __exit__ calls close() even when exception occurs."""
        env = mock_env_class()
        assert not env.closed
        with pytest.raises(ValueError):
            with env:
                raise ValueError("Test exception")
        assert env.closed

    def test_context_manager_propagates_exception(self, mock_env_class):
        """Test that __exit__ returns False to propagate exceptions."""
        env = mock_env_class()
        with pytest.raises(ValueError) as exc_info:
            with env:
                raise ValueError("Test exception")
        assert str(exc_info.value) == "Test exception"


class TestEnvStepOutputTypedDict:
    """Tests for EnvStepOutput TypedDict structure."""

    def test_env_step_output_has_required_fields(self):
        """Test that EnvStepOutput has expected fields."""
        output = EnvStepOutput(
            observations="obs",
            reward=1.0,
            done=True,
            metadata={"key": "value"},
        )
        assert output["observations"] == "obs"
        assert output["reward"] == 1.0
        assert output["done"] is True
        assert output["metadata"] == {"key": "value"}

    def test_env_step_output_metadata_optional(self):
        """Test that metadata field has a default value of None."""
        # EnvStepOutput allows metadata to be None by default annotation
        output = EnvStepOutput(
            observations="obs",
            reward=0.5,
            done=False,
            metadata=None,
        )
        assert output["metadata"] is None


# ==============================================================================
# Section 2: Registration Module Tests
# ==============================================================================


class TestEnvSpec:
    """Tests for EnvSpec dataclass."""

    def test_env_spec_creation(self):
        """Test basic EnvSpec creation."""
        spec = EnvSpec(id="test-env", entry_point="module:TestEnv")
        assert spec.id == "test-env"
        assert spec.name == "test-env"
        assert spec.entry_point == "module:TestEnv"
        assert spec.kwargs == {}

    def test_env_spec_with_kwargs(self):
        """Test EnvSpec creation with kwargs."""
        spec = EnvSpec(
            id="test-env",
            entry_point="module:TestEnv",
            kwargs={"param1": "value1", "param2": 42},
        )
        assert spec.kwargs == {"param1": "value1", "param2": 42}

    def test_env_spec_name_from_id(self):
        """Test that name is set from id in __post_init__."""
        spec = EnvSpec(id="my-environment", entry_point="module:Env")
        assert spec.name == "my-environment"

    def test_env_spec_to_json(self):
        """Test EnvSpec serialization to JSON."""
        spec = EnvSpec(
            id="test-env",
            entry_point="module:TestEnv",
            kwargs={"param": "value"},
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "test-env"
        assert parsed["entry_point"] == "module:TestEnv"
        assert parsed["kwargs"] == {"param": "value"}

    def test_env_spec_to_json_fails_with_callable(self):
        """Test that to_json raises error for callable entry points."""
        spec = EnvSpec(id="test-env", entry_point=lambda: None)
        with pytest.raises(ValueError) as exc_info:
            spec.to_json()
        assert "Callable found" in str(exc_info.value)

    def test_env_spec_from_json(self):
        """Test EnvSpec deserialization from JSON."""
        json_str = '{"id": "test-env", "entry_point": "module:TestEnv", "kwargs": {}}'
        spec = EnvSpec.from_json(json_str)
        assert spec.id == "test-env"
        assert spec.entry_point == "module:TestEnv"

    def test_env_spec_from_json_invalid(self):
        """Test EnvSpec.from_json with invalid JSON raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            EnvSpec.from_json('{"invalid": "spec"}')
        assert "issue occurred" in str(exc_info.value)

    def test_env_spec_pprint(self):
        """Test EnvSpec pretty print."""
        spec = EnvSpec(id="test-env", entry_point="module:TestEnv")
        output = spec.pprint(disable_print=True)
        assert "id=test-env" in output

    def test_env_spec_pprint_with_entry_points(self):
        """Test EnvSpec pretty print with entry points."""
        spec = EnvSpec(id="test-env", entry_point="module:TestEnv")
        output = spec.pprint(disable_print=True, include_entry_points=True)
        assert "id=test-env" in output
        assert "entry_point=module:TestEnv" in output


class TestRegistrationFunctions:
    """Tests for register(), make(), spec() functions."""

    def test_register_adds_env_to_registry(self, clean_registry, mock_env_class):
        """Test that register() adds environment to registry."""
        test_id = "test-register-env"
        register(id=test_id, entry_point=mock_env_class)
        assert test_id in registry
        assert registry[test_id].id == test_id

    def test_register_requires_entry_point(self, clean_registry):
        """Test that register() requires entry_point."""
        with pytest.raises(AssertionError):
            register(id="test-env", entry_point=None)

    def test_register_duplicate_name_raises_error(self, clean_registry, mock_env_class):
        """Test that registering duplicate name raises RegistrationError."""
        register(id="duplicate-env", entry_point=mock_env_class)
        with pytest.raises(error.RegistrationError) as exc_info:
            register(id="duplicate-env", entry_point=mock_env_class)
        assert "already registered" in str(exc_info.value)

    def test_make_creates_env_from_callable(self, clean_registry, mock_env_class):
        """Test that make() creates environment from callable entry point."""
        register(id="make-test-env", entry_point=mock_env_class)
        env = make("make-test-env")
        assert isinstance(env, mock_env_class)
        assert isinstance(env, Env)

    def test_make_creates_env_with_kwargs(self, clean_registry, mock_env_class):
        """Test that make() passes kwargs to environment."""
        register(id="make-kwargs-env", entry_point=mock_env_class)
        env = make("make-kwargs-env", config_value="custom")
        assert env.config_value == "custom"

    def test_make_creates_env_from_spec(self, mock_env_class):
        """Test that make() accepts EnvSpec directly."""
        env_spec = EnvSpec(id="spec-env", entry_point=mock_env_class)
        env = make(env_spec)
        assert isinstance(env, mock_env_class)

    def test_make_sets_env_spec_attribute(self, clean_registry, mock_env_class):
        """Test that make() sets spec attribute on environment."""
        register(id="spec-attr-env", entry_point=mock_env_class)
        env = make("spec-attr-env")
        assert hasattr(env, "spec")
        assert env.spec.id == "spec-attr-env"

    def test_make_unregistered_env_raises_error(self, clean_registry):
        """Test that make() raises error for unregistered environment."""
        with pytest.raises(error.Error) as exc_info:
            make("nonexistent-env")
        assert "No registered env with id" in str(exc_info.value)

    def test_make_none_entry_point_raises_error(self, clean_registry):
        """Test that make() raises error if entry_point is None."""
        # Directly manipulate registry to create invalid spec
        registry["invalid-spec-env"] = EnvSpec(id="invalid-spec-env", entry_point=None)
        with pytest.raises(error.Error) as exc_info:
            make("invalid-spec-env")
        assert "entry_point is not specified" in str(exc_info.value)

    def test_make_non_env_class_raises_type_error(self, clean_registry):
        """Test that make() raises TypeError for non-Env classes."""

        class NotAnEnv:
            pass

        register(id="not-env", entry_point=NotAnEnv)
        with pytest.raises(TypeError) as exc_info:
            make("not-env")
        assert "must inherit from" in str(exc_info.value)

    def test_spec_returns_env_spec(self, clean_registry, mock_env_class):
        """Test that spec() returns EnvSpec for registered environment."""
        register(id="spec-test-env", entry_point=mock_env_class)
        env_spec = spec("spec-test-env")
        assert isinstance(env_spec, EnvSpec)
        assert env_spec.id == "spec-test-env"

    def test_spec_unregistered_raises_error(self, clean_registry):
        """Test that spec() raises error for unregistered environment."""
        with pytest.raises(error.Error) as exc_info:
            spec("nonexistent-spec-env")
        assert "No registered env with id" in str(exc_info.value)


class TestLoadEnvCreator:
    """Tests for load_env_creator function."""

    def test_load_env_creator_valid_path(self):
        """Test loading environment from valid import path."""
        creator = load_env_creator("skyrl_gym.core:Env")
        assert creator is Env

    def test_load_env_creator_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            load_env_creator("invalid_format_no_colon")


class TestPprintRegistry:
    """Tests for pprint_registry function."""

    def test_pprint_registry_returns_string(self, clean_registry, mock_env_class):
        """Test that pprint_registry returns string when disable_print=True."""
        register(id="pprint-env-1", entry_point=mock_env_class)
        register(id="pprint-env-2", entry_point=mock_env_class)
        output = pprint_registry(disable_print=True)
        assert isinstance(output, str)

    def test_pprint_empty_registry(self):
        """Test pprint_registry with empty registry."""
        output = pprint_registry(print_registry={}, disable_print=True)
        assert output == "No environments registered."


# ==============================================================================
# Section 3: SQL Environment Tests
# ==============================================================================


class TestSQLEnvTaskPaths:
    """Tests for SQL environment task type path construction."""

    @pytest.mark.parametrize(
        "task,expected_path_suffix",
        [
            ("synsql", "SynSQL-2.5M/databases"),
            ("spider", "spider/database"),
            ("bird", "bird/train/train_databases"),
            ("bird_dev", "bird/dev/dev_databases"),
            ("spider_dev", "spider/dev_database"),
            ("cosql", "cosql/database"),
            ("sparc", "sparc/database"),
            ("wikisql", "wikisql/databases"),
        ],
    )
    def test_task_path_construction(self, task, expected_path_suffix):
        """Test that task types construct correct database paths."""
        # We test path construction logic by importing and checking the module
        from skyrl_gym.envs.sql.env import SQLEnv

        # Create mock to avoid actual file system access
        with patch("os.path.exists", return_value=True):
            from omegaconf import DictConfig

            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": task,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            assert expected_path_suffix in env.db_path

    def test_unsupported_task_raises_error(self):
        """Test that unsupported task type raises NotImplementedError."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "unsupported_task",
            }
            env_config = DictConfig({"db_path": "/base/path"})
            with pytest.raises(NotImplementedError) as exc_info:
                SQLEnv(env_config, extras)
            assert "not supported" in str(exc_info.value)
            assert "unsupported_task" in str(exc_info.value)

    def test_custom_task_uses_provided_path(self):
        """Test that custom task uses the provided db_path directly."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
            }
            env_config = DictConfig({"db_path": "/custom/path"})
            env = SQLEnv(env_config, extras)
            assert env.db_path == "/custom/path"


class TestSQLEnvIsDone:
    """Tests for SQL environment _is_done() logic."""

    def test_is_done_max_turns_reached(self):
        """Test that _is_done returns True when max_turns reached."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
                "max_turns": 2,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            env.turns = 2
            assert env._is_done("some action") is True

    def test_is_done_solution_tag_present(self):
        """Test that _is_done returns True when solution tags present."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
                "max_turns": 10,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            env.turns = 1
            assert env._is_done("<solution>SELECT 1</solution>") is True

    def test_is_done_incomplete_solution_tag(self):
        """Test that _is_done returns False for incomplete solution tags."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
                "max_turns": 10,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            env.turns = 1
            # Only start tag
            assert env._is_done("<solution>SELECT 1") is False
            # Only end tag
            assert env._is_done("SELECT 1</solution>") is False

    def test_is_done_neither_condition(self):
        """Test that _is_done returns False when no termination condition met."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
                "max_turns": 10,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            env.turns = 1
            assert env._is_done("<sql>SELECT 1</sql>") is False


class TestSQLEnvValidateAction:
    """Tests for SQL environment _validate_action() validation."""

    def test_validate_action_valid_trailing_tag(self):
        """Test that valid action with trailing tag passes validation."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            # Should not raise
            env._validate_action("<think>thinking</think>\n<sql>SELECT 1</sql>")
            env._validate_action("<think>thinking</think>\n<solution>SELECT 1</solution>")

    def test_validate_action_content_after_sql_tag(self):
        """Test that content after </sql> tag raises AssertionError."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            with pytest.raises(AssertionError) as exc_info:
                env._validate_action("<sql>SELECT 1</sql>extra content")
            assert "</sql> detected" in str(exc_info.value)

    def test_validate_action_content_after_solution_tag(self):
        """Test that content after </solution> tag raises AssertionError."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            with pytest.raises(AssertionError) as exc_info:
                env._validate_action("<solution>SELECT 1</solution>trailing")
            assert "</solution> detected" in str(exc_info.value)


class TestSQLEnvToolParsing:
    """Tests for SQL environment tool parsing."""

    def test_parse_action_extracts_sql(self):
        """Test that _parse_action extracts SQL from tags."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
                "max_turns": 5,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            env.turns = 1

            tool_group, tool_name, tool_input = env._parse_action("<sql>SELECT * FROM users</sql>")
            assert tool_group == "SQLCodeExecutorToolGroup"
            assert tool_name == "sql"
            assert tool_input[1] == "SELECT * FROM users"

    def test_parse_action_no_sql_tag(self):
        """Test that _parse_action returns None for tool_input when no SQL tag."""
        from skyrl_gym.envs.sql.env import SQLEnv
        from omegaconf import DictConfig

        with patch("os.path.exists", return_value=True):
            extras = {
                "db_id": "test_db",
                "reward_spec": {"ground_truth": "SELECT 1"},
                "data": "custom",
                "max_turns": 5,
            }
            env_config = DictConfig({"db_path": "/base/path"})
            env = SQLEnv(env_config, extras)
            env.turns = 1

            tool_group, tool_name, tool_input = env._parse_action("no sql tags here")
            assert tool_input[1] is None  # SQL query should be None


# ==============================================================================
# Section 4: Metrics Module Tests
# ==============================================================================


class TestDefaultAggregateMetrics:
    """Tests for default_aggregate_metrics function."""

    def test_aggregate_empty_list(self):
        """Test aggregation of empty list returns empty dict."""
        result = default_aggregate_metrics([])
        assert result == {}

    def test_aggregate_numeric_fields(self):
        """Test that numeric fields are averaged."""
        metrics = [
            {"score": 1.0, "accuracy": 0.8},
            {"score": 2.0, "accuracy": 0.6},
            {"score": 3.0, "accuracy": 1.0},
        ]
        result = default_aggregate_metrics(metrics)
        assert result["score"] == 2.0  # (1+2+3)/3
        assert abs(result["accuracy"] - 0.8) < 1e-9  # (0.8+0.6+1.0)/3

    def test_aggregate_boolean_fields(self):
        """Test that boolean fields are converted to float and averaged."""
        metrics = [
            {"success": True},
            {"success": False},
            {"success": True},
        ]
        result = default_aggregate_metrics(metrics)
        assert abs(result["success"] - (2.0 / 3.0)) < 1e-6

    def test_aggregate_ignores_non_numeric(self):
        """Test that non-numeric fields are ignored."""
        metrics = [
            {"score": 1.0, "name": "test1"},
            {"score": 2.0, "name": "test2"},
        ]
        result = default_aggregate_metrics(metrics)
        assert "score" in result
        assert "name" not in result

    def test_aggregate_mixed_presence(self):
        """Test aggregation when fields are not present in all dicts."""
        metrics = [
            {"score": 1.0, "extra": 10.0},
            {"score": 2.0},
            {"score": 3.0, "extra": 20.0},
        ]
        result = default_aggregate_metrics(metrics)
        assert result["score"] == 2.0  # (1+2+3)/3
        assert result["extra"] == 15.0  # (10+20)/2


class TestAggregateForEnvironment:
    """Tests for aggregate_for_environment function."""

    def test_aggregate_for_unregistered_env_raises_error(self, clean_registry):
        """Test that unregistered environment raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            aggregate_for_environment("nonexistent-env", [])
        assert "No registered env with id" in str(exc_info.value)

    def test_aggregate_for_registered_env(self, clean_registry, mock_text_env_class):
        """Test aggregation using registered environment's method."""
        register(id="agg-test-env", entry_point=mock_text_env_class)
        metrics = [{"score": 1.0}, {"score": 2.0}]
        result = aggregate_for_environment("agg-test-env", metrics)
        # BaseTextEnv uses default_aggregate_metrics
        assert result["score"] == 1.5


# ==============================================================================
# Section 5: Tools Module Tests
# ==============================================================================


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_tool_decorator_captures_function_name(self):
        """Test that @tool decorator captures function name."""

        class MyToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("MyTools")

            @tool
            def my_tool(self, arg: str) -> str:
                return f"Result: {arg}"

        group = MyToolGroup()
        assert "my_tool" in group.get_tool_names()

    def test_tool_execution_through_decorator(self):
        """Test that decorated tools can be executed."""

        class MyToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("MyTools")

            @tool
            def echo(self, message: str) -> str:
                return f"Echo: {message}"

        group = MyToolGroup()
        result = group.execute_tool("echo", "hello")
        assert result == "Echo: hello"


class TestToolGroup:
    """Tests for ToolGroup class."""

    def test_tool_group_creation(self):
        """Test basic ToolGroup creation."""

        class EmptyToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("Empty")

        group = EmptyToolGroup()
        assert group.get_name() == "Empty"
        assert group.get_tool_names() == []

    def test_tool_group_registers_tools(self):
        """Test that tools are registered automatically."""

        class TestToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("TestTools")

            @tool
            def tool_a(self):
                return "A"

            @tool
            def tool_b(self):
                return "B"

        group = TestToolGroup()
        tools = group.get_tool_names()
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_get_tool_returns_callable(self):
        """Test that get_tool returns the callable."""

        class TestToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("TestTools")

            @tool
            def my_tool(self):
                return "result"

        group = TestToolGroup()
        tool_func = group.get_tool("my_tool")
        assert callable(tool_func)
        assert tool_func() == "result"

    def test_get_tool_nonexistent_returns_none(self):
        """Test that get_tool returns None for nonexistent tool."""

        class TestToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("TestTools")

        group = TestToolGroup()
        assert group.get_tool("nonexistent") is None

    def test_execute_tool_nonexistent_raises_error(self):
        """Test that execute_tool raises ValueError for nonexistent tool."""

        class TestToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("TestTools")

        group = TestToolGroup()
        with pytest.raises(ValueError) as exc_info:
            group.execute_tool("nonexistent")
        assert "not found" in str(exc_info.value)

    def test_execute_tool_with_args(self):
        """Test that execute_tool passes arguments correctly."""

        class TestToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("TestTools")

            @tool
            def add(self, a: int, b: int) -> int:
                return a + b

        group = TestToolGroup()
        result = group.execute_tool("add", 3, 5)
        assert result == 8

    def test_get_tool_to_group_mapping(self):
        """Test that get_tool_to_group_mapping returns correct mapping."""

        class TestToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("MyGroup")

            @tool
            def tool_one(self):
                pass

            @tool
            def tool_two(self):
                pass

        group = TestToolGroup()
        mapping = group.get_tool_to_group_mapping()
        assert mapping == {"tool_one": "MyGroup", "tool_two": "MyGroup"}


# ==============================================================================
# Section 6: BaseTextEnv Tests
# ==============================================================================


class TestBaseTextEnv:
    """Tests for BaseTextEnv class."""

    def test_base_text_env_initialization(self, mock_text_env_class):
        """Test BaseTextEnv initialization."""
        env = mock_text_env_class(max_turns=10)
        assert env.turns == 0
        assert env.max_turns == 10
        assert env.tool_groups == []

    def test_base_text_env_init_method(self, mock_text_env_class):
        """Test BaseTextEnv init() method returns prompt."""
        env = mock_text_env_class()
        prompt = [{"role": "user", "content": "Hello"}]
        obs, info = env.init(prompt)
        assert obs == prompt
        assert info == {}

    def test_base_text_env_close(self, mock_text_env_class):
        """Test BaseTextEnv close() method."""
        env = mock_text_env_class()
        env.close()  # Should not raise
        assert env.closed

    def test_base_text_env_get_metrics(self, mock_text_env_class):
        """Test BaseTextEnv get_metrics() returns empty dict by default."""
        env = mock_text_env_class()
        metrics = env.get_metrics()
        assert metrics == {}

    def test_base_text_env_init_tool_groups(self, mock_text_env_class):
        """Test BaseTextEnv tool group initialization."""

        class SimpleToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("SimpleTools")

            @tool
            def simple_tool(self):
                return "simple"

        env = mock_text_env_class()
        tool_group = SimpleToolGroup()
        env.init_tool_groups([tool_group])

        assert len(env.tool_groups) == 1
        assert "simple_tool" in env.tool_to_toolgroup

    def test_base_text_env_execute_tool(self, mock_text_env_class):
        """Test BaseTextEnv tool execution."""

        class SimpleToolGroup(ToolGroup):
            def __init__(self):
                super().__init__("SimpleTools")

            @tool
            def greet(self, name: str):
                return f"Hello, {name}!"

        env = mock_text_env_class()
        tool_group = SimpleToolGroup()
        env.init_tool_groups([tool_group])

        result = env._execute_tool("SimpleTools", "greet", ("World",))
        assert result == "Hello, World!"

    def test_base_text_env_execute_tool_not_found(self, mock_text_env_class):
        """Test BaseTextEnv raises error for nonexistent tool group."""
        env = mock_text_env_class()
        with pytest.raises(ValueError) as exc_info:
            env._execute_tool("NonexistentGroup", "some_tool", ())
        assert "not found" in str(exc_info.value)


# ==============================================================================
# Section 7: Error Module Tests
# ==============================================================================


class TestErrorClasses:
    """Tests for error classes."""

    def test_error_is_exception(self):
        """Test that Error inherits from Exception."""
        err = error.Error("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"

    def test_registration_error_is_error(self):
        """Test that RegistrationError inherits from Error."""
        err = error.RegistrationError("registration failed")
        assert isinstance(err, error.Error)
        assert isinstance(err, Exception)


# ==============================================================================
# Section 8: Integration Tests
# ==============================================================================


class TestBuiltInEnvironments:
    """Tests for built-in environment registrations."""

    @pytest.mark.parametrize(
        "env_id",
        [
            "aime",
            "gsm8k",
            "gsm8k_multi_turn",
            "text2sql",
            "search",
            "lcb",
            "searchcode",
        ],
    )
    def test_builtin_envs_registered(self, env_id):
        """Test that built-in environments are registered."""
        assert env_id in registry

    @pytest.mark.parametrize(
        "env_id",
        [
            "aime",
            "gsm8k",
            "gsm8k_multi_turn",
            "text2sql",
            "search",
            "lcb",
            "searchcode",
        ],
    )
    def test_builtin_envs_have_valid_spec(self, env_id):
        """Test that built-in environments have valid specs."""
        env_spec = spec(env_id)
        assert env_spec.id == env_id
        assert env_spec.entry_point is not None


class TestEnvSpecMake:
    """Tests for EnvSpec.make() method."""

    def test_env_spec_make_creates_env(self, mock_env_class):
        """Test that EnvSpec.make() creates environment."""
        env_spec = EnvSpec(id="spec-make-test", entry_point=mock_env_class)
        env = env_spec.make()
        assert isinstance(env, mock_env_class)

    def test_env_spec_make_with_kwargs(self, mock_env_class):
        """Test that EnvSpec.make() passes kwargs."""
        env_spec = EnvSpec(
            id="spec-make-kwargs-test",
            entry_point=mock_env_class,
            kwargs={"config_value": "from_spec"},
        )
        env = env_spec.make()
        assert env.config_value == "from_spec"

    def test_env_spec_make_override_kwargs(self, mock_env_class):
        """Test that EnvSpec.make() allows overriding kwargs."""
        env_spec = EnvSpec(
            id="spec-make-override-test",
            entry_point=mock_env_class,
            kwargs={"config_value": "original"},
        )
        env = env_spec.make(config_value="overridden")
        assert env.config_value == "overridden"
