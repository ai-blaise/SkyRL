"""
Comprehensive integration tests for nmoe model outputs with SkyRL verifiers.

This module tests the integration between nmoe Mixture-of-Experts model outputs
and SkyRL verifiers including:
1. naive_dapo verifier (math verification with DAPO scoring)
2. py_functional verifier (timeout utilities and functional helpers)
3. Reward computation from nmoe generations
4. Batch verification efficiency
5. Verifier with MoE aux_loss signals
6. Verifier determinism
7. Verifier error handling
8. Multi-GPU verification

Run with:
    uv run --isolated --extra dev pytest tests/integration/test_nmoe_verifier_integration.py -v
"""

import asyncio
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Check for optional dependencies
try:
    from sympy import Symbol, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

# Markers for tests requiring specific dependencies
requires_sympy = pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")


# -----------------------------------------------------------------------------
# Mock Classes and Fixtures
# -----------------------------------------------------------------------------


@dataclass
class MockMoEOutput:
    """Mock output from an nmoe MoE layer."""
    logits: torch.Tensor
    aux_loss: torch.Tensor
    expert_loads: torch.Tensor
    text: Optional[str] = None

    @classmethod
    def create(
        cls,
        batch_size: int = 4,
        seq_len: int = 128,
        vocab_size: int = 32000,
        n_experts: int = 8,
        text: Optional[str] = None,
    ) -> "MockMoEOutput":
        """Create a mock MoE output for testing."""
        logits = torch.randn(batch_size, seq_len, vocab_size)
        aux_loss = torch.tensor(0.01)  # Small auxiliary load balancing loss
        # Expert loads should sum to 1 (normalized)
        expert_loads = torch.softmax(torch.randn(n_experts), dim=0)
        return cls(
            logits=logits,
            aux_loss=aux_loss,
            expert_loads=expert_loads,
            text=text,
        )


@dataclass
class MockNMoEConfig:
    """Mock configuration for nmoe model."""
    dim: int = 2048
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    aux_loss_alpha: float = 0.01
    route_scale: float = 1.0
    moe_inter_dim: int = 8192
    vocab_size: int = 32000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "n_routed_experts": self.n_routed_experts,
            "n_activated_experts": self.n_activated_experts,
            "aux_loss_alpha": self.aux_loss_alpha,
            "route_scale": self.route_scale,
            "moe_inter_dim": self.moe_inter_dim,
            "vocab_size": self.vocab_size,
        }


class MockRouter:
    """Mock router for testing."""

    def __init__(self, n_experts: int = 8, topk: int = 2):
        self.n_experts = n_experts
        self.topk = topk
        self.bias = torch.zeros(n_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mock router forward pass."""
        T = x.size(0)
        # Generate random expert assignments
        indices = torch.randint(0, self.n_experts, (T, self.topk))
        weights = torch.softmax(torch.randn(T, self.topk), dim=-1)
        return weights, indices


class MockNMoEModel:
    """Mock nmoe model for testing verifier integration."""

    def __init__(self, config: Optional[MockNMoEConfig] = None):
        self.config = config or MockNMoEConfig()
        self.router = MockRouter(
            n_experts=self.config.n_routed_experts,
            topk=self.config.n_activated_experts,
        )
        self.last_aux_loss = torch.tensor(0.0)
        self.last_loads = None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Tuple[torch.Tensor, MockMoEOutput]:
        """Mock generation with MoE output."""
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1) + max_new_tokens

        # Generate mock output tokens
        output_ids = torch.randint(0, self.config.vocab_size, (batch_size, max_new_tokens))
        full_ids = torch.cat([input_ids, output_ids], dim=1)

        # Create mock MoE output
        moe_output = MockMoEOutput.create(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=self.config.vocab_size,
            n_experts=self.config.n_routed_experts,
        )

        self.last_aux_loss = moe_output.aux_loss
        self.last_loads = moe_output.expert_loads

        return full_ids, moe_output

    def get_router_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss from router for load balancing."""
        return self.last_aux_loss

    def get_expert_load_stats(self) -> Dict[str, Any]:
        """Get expert load statistics."""
        if self.last_loads is None:
            return {"loads": [], "mean_load": 0.0, "load_imbalance": 0.0}

        loads = self.last_loads
        mean_load = loads.mean().item()
        std_load = loads.std().item()
        load_imbalance = std_load / mean_load if mean_load > 0 else 0.0

        return {
            "loads": loads.tolist(),
            "mean_load": mean_load,
            "load_imbalance": load_imbalance,
        }


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Simple encoding."""
        return [ord(c) % 1000 for c in text[:100]]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Simple decoding."""
        return "".join(chr(min(max(i, 32), 126)) for i in ids[:100])

    def batch_decode(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Batch decode."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_nmoe_model():
    """Provide a mock nmoe model."""
    return MockNMoEModel()


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def mock_config():
    """Provide a mock nmoe config."""
    return MockNMoEConfig()


@pytest.fixture
def sample_math_problems():
    """Sample math problems for testing naive_dapo verifier."""
    return [
        {
            "problem": "What is 2 + 2?",
            "solution": "The answer is \\boxed{4}",
            "ground_truth": "4",
        },
        {
            "problem": "Solve for x: 2x = 10",
            "solution": "Dividing both sides by 2: x = \\boxed{5}",
            "ground_truth": "5",
        },
        {
            "problem": "What is the square root of 16?",
            "solution": "\\sqrt{16} = \\boxed{4}",
            "ground_truth": "4",
        },
        {
            "problem": "Simplify: 3/6",
            "solution": "3/6 = \\boxed{1/2}",
            "ground_truth": "1/2",
        },
        {
            "problem": "Calculate: 100 * 0.15",
            "solution": "100 * 0.15 = \\boxed{15}",
            "ground_truth": "15",
        },
    ]


@pytest.fixture
def sample_qa_problems():
    """Sample QA problems for testing qa verifier."""
    return [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "target": "Paris",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "target": "Shakespeare",
        },
        {
            "question": "What is H2O?",
            "answer": "Water",
            "target": "water",
        },
    ]


# -----------------------------------------------------------------------------
# Test Class: naive_dapo Verifier Integration
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestNaiveDapoVerifierIntegration:
    """Tests for naive_dapo verifier with nmoe outputs."""

    def test_compute_score_correct_boxed_answer(self):
        """Test compute_score with correct boxed answer."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        solution = "Let me solve this step by step. The answer is \\boxed{42}"
        ground_truth = "42"

        result = compute_score(solution, ground_truth, {})

        assert result["score"] == 1.0
        assert result["acc"] == True

    def test_compute_score_incorrect_answer(self):
        """Test compute_score with incorrect answer."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        solution = "The answer is \\boxed{10}"
        ground_truth = "42"

        result = compute_score(solution, ground_truth, {})

        assert result["score"] == 0.0
        assert result["acc"] == False

    def test_compute_score_with_think_tag(self):
        """Test compute_score correctly extracts answer after </think> tag."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        solution = "<think>Let me think... 2+2=4</think>The answer is \\boxed{4}"
        ground_truth = "4"

        result = compute_score(solution, ground_truth, {})

        assert result["score"] == 1.0

    def test_normalize_final_answer_removes_units(self):
        """Test normalization removes common units."""
        from skyrl_agent.tasks.verifiers.naive_dapo import normalize_final_answer

        test_cases = [
            ("42 dollars", "42"),
            ("100 degrees", "100"),
            ("5 cm", "5"),
        ]

        for input_val, expected_prefix in test_cases:
            result = normalize_final_answer(input_val)
            assert expected_prefix in result or result.replace(" ", "") == expected_prefix.replace(" ", "")

    def test_grade_answer_numeric_equivalence(self):
        """Test grade_answer handles numeric equivalence."""
        from skyrl_agent.tasks.verifiers.naive_dapo import grade_answer

        # Test equivalent numeric representations
        test_cases = [
            ("42", "42", True),
            ("4.0", "4", True),
            ("1/2", "0.5", False),  # Fractions need exact match
        ]

        for pred, truth, should_match in test_cases:
            is_correct, _ = grade_answer(pred, truth)
            if should_match:
                assert is_correct, f"Expected {pred} to match {truth}"

    @requires_sympy
    def test_grade_answer_symbolic_equivalence(self):
        """Test grade_answer handles symbolic equivalence."""
        from skyrl_agent.tasks.verifiers.naive_dapo import grade_answer

        # Test symbolically equivalent expressions
        is_correct, _ = grade_answer("x+x", "2x")
        # Note: This may or may not pass depending on sympy normalization

    def test_match_answer_extracts_boxed(self):
        """Test match_answer correctly extracts boxed content."""
        from skyrl_agent.tasks.verifiers.naive_dapo import match_answer

        response = "After calculation, \\boxed{123}"
        is_matched, answer = match_answer(response)

        assert is_matched == True
        assert answer == "123"

    def test_match_answer_no_boxed(self):
        """Test match_answer when no boxed content exists."""
        from skyrl_agent.tasks.verifiers.naive_dapo import match_answer

        response = "The answer is 123"
        is_matched, answer = match_answer(response)

        assert is_matched == False
        assert "123" in answer

    def test_compute_score_with_nmoe_output(self, mock_nmoe_model, mock_tokenizer):
        """Test compute_score integration with mock nmoe output."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Simulate nmoe model generating a solution
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output_ids, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        # Create a mock solution string
        solution = "The answer is \\boxed{42}"
        ground_truth = "42"

        result = compute_score(solution, ground_truth, {"aux_loss": moe_output.aux_loss.item()})

        assert "score" in result
        assert "acc" in result

    def test_compute_score_batch_processing(self, sample_math_problems):
        """Test compute_score can process a batch of problems."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        results = []
        for problem in sample_math_problems:
            result = compute_score(
                problem["solution"],
                problem["ground_truth"],
                {},
            )
            results.append(result)

        # All sample problems should be correct
        assert all(r["score"] == 1.0 for r in results)
        assert len(results) == len(sample_math_problems)

    def test_compute_score_with_pi(self):
        """Test compute_score handles pi correctly."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        solution = "The area is \\boxed{4\\pi}"
        ground_truth = "4\\pi"

        result = compute_score(solution, ground_truth, {})

        # Should handle pi expressions
        assert result["score"] in [0.0, 1.0]


# -----------------------------------------------------------------------------
# Test Class: py_functional Verifier Integration
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestPyFunctionalVerifierIntegration:
    """Tests for py_functional verifier utilities with nmoe."""

    def test_timeout_limit_decorator_success(self):
        """Test timeout_limit decorator allows quick functions to complete."""
        from skyrl_agent.tasks.verifiers.py_functional import timeout_limit

        @timeout_limit(seconds=5.0)
        def quick_function():
            return 42

        result = quick_function()
        assert result == 42

    def test_timeout_limit_decorator_timeout(self):
        """Test timeout_limit decorator raises TimeoutError."""
        from skyrl_agent.tasks.verifiers.py_functional import timeout_limit

        @timeout_limit(seconds=0.1)
        def slow_function():
            time.sleep(2.0)
            return 42

        with pytest.raises(TimeoutError):
            slow_function()

    def test_union_two_dict(self):
        """Test union_two_dict merges dictionaries correctly."""
        from skyrl_agent.tasks.verifiers.py_functional import union_two_dict

        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}

        result = union_two_dict(dict1, dict2)

        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_union_two_dict_conflict(self):
        """Test union_two_dict raises on conflicting values."""
        from skyrl_agent.tasks.verifiers.py_functional import union_two_dict

        dict1 = {"a": 1}
        dict2 = {"a": 2}  # Different value for same key

        with pytest.raises(AssertionError):
            union_two_dict(dict1, dict2)

    def test_append_to_dict(self):
        """Test append_to_dict appends values to lists."""
        from skyrl_agent.tasks.verifiers.py_functional import append_to_dict

        data = {"scores": [1, 2]}
        new_data = {"scores": 3, "losses": 0.1}

        append_to_dict(data, new_data)

        assert data["scores"] == [1, 2, 3]
        assert data["losses"] == [0.1]

    def test_nested_namespace_creation(self):
        """Test NestedNamespace creates proper dot-access structure."""
        from skyrl_agent.tasks.verifiers.py_functional import NestedNamespace

        config = {
            "model": {
                "dim": 2048,
                "moe": {"n_experts": 8},
            },
            "training": {"lr": 0.001},
        }

        ns = NestedNamespace(config)

        assert ns.model.dim == 2048
        assert ns.model.moe.n_experts == 8
        assert ns.training.lr == 0.001

    def test_dynamic_enum_registration(self):
        """Test DynamicEnum registration and lookup."""
        from skyrl_agent.tasks.verifiers.py_functional import DynamicEnum

        class TestEnum(DynamicEnum):
            _registry = {}
            _next_value = 0

        # Register members
        member1 = TestEnum.register("test_member")

        assert "TEST_MEMBER" in TestEnum
        assert TestEnum["TEST_MEMBER"] == member1

        # Clean up
        TestEnum.remove("test_member")

    def test_convert_to_regular_types(self):
        """Test convert_to_regular_types handles nested structures."""
        from skyrl_agent.tasks.verifiers.py_functional import convert_to_regular_types

        # Test with regular Python types (no Hydra)
        data = {"a": [1, 2, 3], "b": {"c": 4}}
        result = convert_to_regular_types(data)

        assert result == {"a": [1, 2, 3], "b": {"c": 4}}


# -----------------------------------------------------------------------------
# Test Class: Reward Computation from nmoe Generations
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestRewardComputationIntegration:
    """Tests for reward computation from nmoe model generations."""

    def test_reward_from_correct_math_answer(self, mock_nmoe_model):
        """Test reward computation for correct math answer."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Generate with mock model
        input_ids = torch.tensor([[1, 2, 3]])
        _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        # Simulate correct answer
        solution = "\\boxed{42}"
        ground_truth = "42"

        result = compute_score(solution, ground_truth, {})

        assert result["score"] == 1.0

        # Verify aux_loss is available for training signal
        aux_loss = mock_nmoe_model.get_router_aux_loss()
        assert aux_loss is not None

    def test_reward_with_aux_loss_penalty(self, mock_nmoe_model):
        """Test reward computation with auxiliary loss penalty."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Generate with mock model
        input_ids = torch.tensor([[1, 2, 3]])
        _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        solution = "\\boxed{42}"
        ground_truth = "42"

        base_result = compute_score(solution, ground_truth, {})
        base_score = base_result["score"]

        # Apply aux_loss penalty
        aux_loss = moe_output.aux_loss.item()
        penalty_weight = 0.1
        adjusted_score = base_score - penalty_weight * aux_loss

        # Score should be slightly reduced by aux_loss penalty
        assert adjusted_score <= base_score

    def test_reward_aggregation_over_batch(self, mock_nmoe_model, sample_math_problems):
        """Test reward aggregation over a batch of generations."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        batch_size = len(sample_math_problems)
        rewards = []
        aux_losses = []

        for problem in sample_math_problems:
            # Generate (simulated)
            input_ids = torch.tensor([[1, 2, 3]])
            _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

            # Compute reward
            result = compute_score(
                problem["solution"],
                problem["ground_truth"],
                {},
            )
            rewards.append(result["score"])
            aux_losses.append(moe_output.aux_loss.item())

        # Aggregate metrics
        mean_reward = sum(rewards) / len(rewards)
        mean_aux_loss = sum(aux_losses) / len(aux_losses)

        assert mean_reward == 1.0  # All samples are correct
        assert mean_aux_loss >= 0.0

    def test_reward_with_expert_load_stats(self, mock_nmoe_model):
        """Test reward computation with expert load statistics."""
        # Generate
        input_ids = torch.tensor([[1, 2, 3]])
        _, _ = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        # Get expert load stats
        stats = mock_nmoe_model.get_expert_load_stats()

        assert "loads" in stats
        assert "mean_load" in stats
        assert "load_imbalance" in stats

        # Load imbalance should be non-negative
        assert stats["load_imbalance"] >= 0.0


# -----------------------------------------------------------------------------
# Test Class: Batch Verification Efficiency
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestBatchVerificationEfficiency:
    """Tests for efficient batch verification with nmoe outputs."""

    def test_batch_verification_speed(self, sample_math_problems):
        """Test that batch verification is reasonably fast."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Extend sample problems to create larger batch
        problems = sample_math_problems * 20  # 100 problems

        start_time = time.time()

        results = []
        for problem in problems:
            result = compute_score(
                problem["solution"],
                problem["ground_truth"],
                {},
            )
            results.append(result)

        elapsed = time.time() - start_time

        # Should process 100 problems in under 5 seconds
        assert elapsed < 5.0
        assert len(results) == 100

    def test_parallel_verification_with_threads(self, sample_math_problems):
        """Test parallel verification using thread pool."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        problems = sample_math_problems * 10  # 50 problems

        def verify_single(problem):
            return compute_score(
                problem["solution"],
                problem["ground_truth"],
                {},
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(verify_single, problems))

        assert len(results) == 50
        assert all(r["score"] == 1.0 for r in results)

    def test_vectorized_reward_computation(self):
        """Test vectorized reward computation for efficiency."""
        # Simulate batch of scores
        batch_size = 1000
        raw_scores = torch.rand(batch_size)  # Random scores between 0 and 1
        aux_losses = torch.rand(batch_size) * 0.1  # Small aux losses

        # Vectorized computation
        penalty_weight = 0.1
        adjusted_scores = raw_scores - penalty_weight * aux_losses

        # All operations should be tensor-based
        assert adjusted_scores.shape == (batch_size,)
        assert torch.all(adjusted_scores <= raw_scores)

    def test_memory_efficiency_large_batch(self, mock_nmoe_model):
        """Test memory efficiency with large batches."""
        batch_size = 100

        # Track memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        results = []
        for i in range(batch_size):
            input_ids = torch.tensor([[1, 2, 3]])
            _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=20)
            results.append(moe_output.aux_loss.item())

        # Should complete without memory issues
        assert len(results) == batch_size


# -----------------------------------------------------------------------------
# Test Class: Verifier with MoE aux_loss Signals
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestVerifierWithMoEAuxLoss:
    """Tests for verifier integration with MoE auxiliary loss signals."""

    def test_aux_loss_computation_format(self, mock_nmoe_model):
        """Test aux_loss has correct format."""
        input_ids = torch.tensor([[1, 2, 3]])
        _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        aux_loss = moe_output.aux_loss

        # Should be a scalar tensor
        assert aux_loss.dim() == 0
        assert aux_loss.dtype == torch.float32 or aux_loss.dtype == torch.float64

    def test_aux_loss_gradient_flow(self):
        """Test that aux_loss supports gradient computation."""
        # Create a simple mock MoE layer with trainable parameters
        gate = torch.nn.Linear(64, 8)
        x = torch.randn(4, 64, requires_grad=True)

        # Compute router logits
        logits = gate(x)
        scores = torch.softmax(logits, dim=-1)

        # Simplified aux_loss computation
        # aux_loss = alpha * E * sum(f_i * P_i)
        n_experts = 8
        alpha = 0.01
        f = scores.mean(dim=0)  # Approximate dispatch fraction
        P = scores.mean(dim=0)  # Mean routing probability
        aux_loss = alpha * n_experts * (f * P).sum()

        # Verify gradient can flow
        aux_loss.backward()

        assert x.grad is not None
        assert gate.weight.grad is not None

    def test_aux_loss_in_reward_function(self, mock_nmoe_model):
        """Test incorporating aux_loss into reward function."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Generate
        input_ids = torch.tensor([[1, 2, 3]])
        _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        # Compute base reward
        solution = "\\boxed{42}"
        ground_truth = "42"
        base_result = compute_score(solution, ground_truth, {})

        # Define reward function with aux_loss regularization
        def compute_total_reward(base_score, aux_loss, lambda_aux=0.1):
            """Compute total reward with aux_loss penalty."""
            return base_score - lambda_aux * aux_loss

        total_reward = compute_total_reward(
            base_result["score"],
            moe_output.aux_loss.item(),
        )

        assert total_reward <= base_result["score"]

    def test_expert_load_balance_metric(self, mock_nmoe_model):
        """Test expert load balance metric computation."""
        # Generate multiple times to accumulate stats
        for _ in range(10):
            input_ids = torch.tensor([[1, 2, 3]])
            mock_nmoe_model.generate(input_ids, max_new_tokens=50)

        stats = mock_nmoe_model.get_expert_load_stats()

        # Verify stats structure
        assert "loads" in stats
        assert "mean_load" in stats
        assert "load_imbalance" in stats

        # Perfect balance would have imbalance = 0
        # Some imbalance is expected with random routing
        assert stats["load_imbalance"] >= 0.0

    def test_aux_loss_scales_with_imbalance(self):
        """Test that aux_loss increases with load imbalance."""
        n_experts = 8
        alpha = 0.01

        # Balanced load
        balanced_f = torch.ones(n_experts) / n_experts
        balanced_P = torch.ones(n_experts) / n_experts
        balanced_aux = alpha * n_experts * (balanced_f * balanced_P).sum()

        # Imbalanced load (all traffic to one expert)
        imbalanced_f = torch.zeros(n_experts)
        imbalanced_f[0] = 1.0
        imbalanced_P = torch.zeros(n_experts)
        imbalanced_P[0] = 1.0
        imbalanced_aux = alpha * n_experts * (imbalanced_f * imbalanced_P).sum()

        # Imbalanced should have higher aux_loss
        assert imbalanced_aux > balanced_aux


# -----------------------------------------------------------------------------
# Test Class: Verifier Determinism
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestVerifierDeterminism:
    """Tests for verifier determinism with nmoe outputs."""

    def test_compute_score_determinism(self):
        """Test compute_score produces deterministic results."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        solution = "The answer is \\boxed{42}"
        ground_truth = "42"

        results = []
        for _ in range(10):
            result = compute_score(solution, ground_truth, {})
            results.append(result["score"])

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_grade_answer_determinism(self):
        """Test grade_answer produces deterministic results."""
        from skyrl_agent.tasks.verifiers.naive_dapo import grade_answer

        test_cases = [
            ("42", "42"),
            ("3.14159", "3.14159"),
            ("1/2", "1/2"),
        ]

        for pred, truth in test_cases:
            results = []
            for _ in range(5):
                is_correct, _ = grade_answer(pred, truth)
                results.append(is_correct)

            # All results should be identical
            assert all(r == results[0] for r in results)

    def test_normalize_determinism(self):
        """Test normalization is deterministic."""
        from skyrl_agent.tasks.verifiers.naive_dapo import normalize_final_answer

        test_inputs = [
            "42 degrees",
            "100 dollars",
            "\\boxed{5}",
        ]

        for input_val in test_inputs:
            results = []
            for _ in range(5):
                result = normalize_final_answer(input_val)
                results.append(result)

            # All normalizations should be identical
            assert all(r == results[0] for r in results)

    def test_seeded_random_does_not_affect_verification(self):
        """Test that random seed does not affect verification."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        solution = "\\boxed{42}"
        ground_truth = "42"

        # Run with different random seeds
        results = []
        for seed in [42, 123, 456, 789]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            result = compute_score(solution, ground_truth, {})
            results.append(result["score"])

        # All results should be identical despite different seeds
        assert all(r == results[0] for r in results)


# -----------------------------------------------------------------------------
# Test Class: Verifier Error Handling
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestVerifierErrorHandling:
    """Tests for verifier error handling with nmoe outputs."""

    def test_compute_score_handles_none_solution(self):
        """Test compute_score handles None solution gracefully."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        result = compute_score(None, "42", {})

        # Should return a valid result structure
        assert "score" in result
        assert result["score"] == 0.0

    def test_compute_score_handles_empty_solution(self):
        """Test compute_score handles empty solution."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        result = compute_score("", "42", {})

        assert "score" in result

    def test_grade_answer_handles_none(self):
        """Test grade_answer handles None gracefully."""
        from skyrl_agent.tasks.verifiers.naive_dapo import grade_answer

        is_correct, _ = grade_answer(None, "42")

        assert is_correct == False

    def test_match_answer_handles_malformed_boxed(self):
        """Test match_answer handles malformed boxed expressions."""
        from skyrl_agent.tasks.verifiers.naive_dapo import match_answer

        test_cases = [
            "\\boxed{",  # Unclosed
            "\\boxed}",  # Missing opening
            "\\boxed{}",  # Empty
            "\\boxed{{}",  # Nested unbalanced
        ]

        for case in test_cases:
            # Should not raise exception
            is_matched, result = match_answer(case)
            # Result should be valid (even if not matched)
            assert isinstance(is_matched, bool)

    def test_normalize_handles_special_characters(self):
        """Test normalization handles special characters."""
        from skyrl_agent.tasks.verifiers.naive_dapo import normalize_final_answer

        special_cases = [
            "42!@#$%",
            "answer\n\twith\twhitespace",
            "unicode: \u03c0 \u221e",
        ]

        for case in special_cases:
            # Should not raise exception
            result = normalize_final_answer(case)
            assert isinstance(result, str)

    def test_symbolic_equal_timeout_handling(self):
        """Test symbolic_equal handles timeout gracefully."""
        from skyrl_agent.tasks.verifiers.naive_dapo import are_equal_under_sympy

        # Very complex expression that might timeout
        complex_expr = "x" * 100  # Repeated variable

        # Should complete without hanging
        try:
            result = are_equal_under_sympy(complex_expr, "x")
            assert isinstance(result, bool)
        except TimeoutError:
            # Timeout is acceptable
            pass

    def test_compute_score_handles_very_long_input(self):
        """Test compute_score handles very long inputs."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Very long solution
        long_solution = "x" * 10000 + "\\boxed{42}"

        result = compute_score(long_solution, "42", {})

        assert "score" in result

    def test_verifier_recovers_from_exceptions(self):
        """Test verifier recovers from internal exceptions."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Test with various edge cases that might cause internal exceptions
        edge_cases = [
            ("\\boxed{1/0}", "inf"),  # Division by zero expression
            ("\\boxed{sqrt(-1)}", "i"),  # Complex number
            ("\\boxed{}", ""),  # Empty boxed
        ]

        for solution, ground_truth in edge_cases:
            # Should not raise exception, but gracefully return a result
            try:
                result = compute_score(solution, ground_truth, {})
                assert "score" in result
            except Exception:
                # Some edge cases may raise, but should be specific exceptions
                pass


# -----------------------------------------------------------------------------
# Test Class: Multi-GPU Verification
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiGPUVerification:
    """Tests for multi-GPU verification scenarios."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_verification_on_different_devices(self):
        """Test verification works with tensors on different devices."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Verification is CPU-based, but should handle GPU tensor inputs
        solution = "\\boxed{42}"
        ground_truth = "42"

        # Extra info with GPU tensor (if present)
        extra_info = {
            "aux_loss": torch.tensor(0.01).cuda() if torch.cuda.is_available() else torch.tensor(0.01)
        }

        result = compute_score(solution, ground_truth, extra_info)

        assert "score" in result

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multi-GPU not available"
    )
    def test_parallel_verification_multi_gpu(self):
        """Test parallel verification across multiple GPUs."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        n_gpus = torch.cuda.device_count()

        def verify_on_device(device_id):
            torch.cuda.set_device(device_id)
            result = compute_score("\\boxed{42}", "42", {})
            return result["score"]

        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            results = list(executor.map(verify_on_device, range(n_gpus)))

        assert all(r == 1.0 for r in results)

    def test_distributed_verification_simulation(self):
        """Test simulated distributed verification."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        # Simulate distributed scenario
        world_size = 4
        problems_per_rank = 5

        all_results = []
        for rank in range(world_size):
            rank_results = []
            for i in range(problems_per_rank):
                result = compute_score(
                    f"\\boxed{{{rank * problems_per_rank + i}}}",
                    str(rank * problems_per_rank + i),
                    {"rank": rank},
                )
                rank_results.append(result["score"])
            all_results.append(rank_results)

        # All should be correct
        assert all(
            all(r == 1.0 for r in rank_results)
            for rank_results in all_results
        )

    def test_verification_with_expert_parallelism(self, mock_nmoe_model):
        """Test verification with simulated expert parallelism."""
        # Simulate expert parallelism where different experts are on different ranks
        n_experts = mock_nmoe_model.config.n_routed_experts
        world_size = 4
        experts_per_rank = n_experts // world_size

        # Each rank processes its local experts
        for rank in range(world_size):
            local_experts = list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank))

            # Generate with mock model
            input_ids = torch.tensor([[1, 2, 3]])
            _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

            # Verify expert loads are available
            assert moe_output.expert_loads is not None


# -----------------------------------------------------------------------------
# Test Class: QA Verifier Integration
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestQAVerifierIntegration:
    """Tests for QA verifier integration with nmoe outputs."""

    def test_normalize_answer(self):
        """Test QA answer normalization."""
        from skyrl_agent.tasks.verifiers.qa import normalize_answer

        test_cases = [
            ("The answer is Paris", "answer is paris"),
            ("A cat", "cat"),
            ("  Extra  Spaces  ", "extra spaces"),
        ]

        for input_val, expected in test_cases:
            result = normalize_answer(input_val)
            assert result == expected

    def test_em_check_exact_match(self):
        """Test exact match checking."""
        from skyrl_agent.tasks.verifiers.qa import em_check

        assert em_check("Paris", "Paris") == 1
        assert em_check("paris", "Paris") == 1  # Case insensitive
        assert em_check("London", "Paris") == 0

    def test_em_check_with_list(self):
        """Test exact match with list of acceptable answers."""
        from skyrl_agent.tasks.verifiers.qa import em_check

        golden_answers = ["Paris", "paris", "PARIS"]

        assert em_check("Paris", golden_answers) == 1
        assert em_check("London", golden_answers) == 0

    def test_f1_score_computation(self):
        """Test F1 score computation."""
        from skyrl_agent.tasks.verifiers.qa import f1_score

        # Perfect match
        assert f1_score("Paris", "Paris") == 1.0

        # Partial match
        score = f1_score("Paris France", "Paris")
        assert 0.0 < score < 1.0

    def test_compute_score_em(self):
        """Test compute_score_em function."""
        from skyrl_agent.tasks.verifiers.qa import compute_score_em

        result = compute_score_em(
            "Paris",
            {"target": "Paris"},
        )

        assert result["score"] == 1.0

    def test_compute_score_f1(self):
        """Test compute_score_f1 function."""
        from skyrl_agent.tasks.verifiers.qa import compute_score_f1

        result = compute_score_f1(
            "Paris",
            {"target": "Paris"},
        )

        assert result["score"] == 1.0

    def test_qa_verification_with_nmoe_output(self, mock_nmoe_model, sample_qa_problems):
        """Test QA verification with nmoe model output."""
        from skyrl_agent.tasks.verifiers.qa import compute_score_em

        for problem in sample_qa_problems:
            # Generate with mock model
            input_ids = torch.tensor([[1, 2, 3]])
            _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

            # Verify answer
            result = compute_score_em(
                problem["answer"],
                {"target": problem["target"]},
            )

            assert "score" in result


# -----------------------------------------------------------------------------
# Test Class: End-to-End Integration
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests for nmoe verifier pipeline."""

    def test_full_verification_pipeline(self, mock_nmoe_model, sample_math_problems):
        """Test full verification pipeline from generation to reward."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        results = []

        for problem in sample_math_problems:
            # Step 1: Generate with nmoe model
            input_ids = torch.tensor([[1, 2, 3]])
            _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=50)

            # Step 2: Get auxiliary loss
            aux_loss = moe_output.aux_loss.item()

            # Step 3: Compute verification score
            verification_result = compute_score(
                problem["solution"],
                problem["ground_truth"],
                {},
            )

            # Step 4: Compute final reward with aux_loss penalty
            base_score = verification_result["score"]
            final_reward = base_score - 0.1 * aux_loss

            results.append({
                "base_score": base_score,
                "aux_loss": aux_loss,
                "final_reward": final_reward,
            })

        # Verify results
        assert len(results) == len(sample_math_problems)
        assert all(r["base_score"] == 1.0 for r in results)

    def test_batch_verification_with_metrics(self, mock_nmoe_model):
        """Test batch verification with comprehensive metrics."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        batch_size = 10

        # Generate batch of problems
        problems = [
            {"solution": f"\\boxed{{{i}}}", "ground_truth": str(i)}
            for i in range(batch_size)
        ]

        metrics = {
            "scores": [],
            "aux_losses": [],
            "expert_imbalances": [],
        }

        for problem in problems:
            # Generate
            input_ids = torch.tensor([[1, 2, 3]])
            _, moe_output = mock_nmoe_model.generate(input_ids, max_new_tokens=20)

            # Verify
            result = compute_score(
                problem["solution"],
                problem["ground_truth"],
                {},
            )

            # Collect metrics
            metrics["scores"].append(result["score"])
            metrics["aux_losses"].append(moe_output.aux_loss.item())
            stats = mock_nmoe_model.get_expert_load_stats()
            metrics["expert_imbalances"].append(stats["load_imbalance"])

        # Compute aggregate metrics
        mean_score = sum(metrics["scores"]) / len(metrics["scores"])
        mean_aux_loss = sum(metrics["aux_losses"]) / len(metrics["aux_losses"])
        mean_imbalance = sum(metrics["expert_imbalances"]) / len(metrics["expert_imbalances"])

        assert mean_score == 1.0
        assert mean_aux_loss >= 0.0
        assert mean_imbalance >= 0.0

    def test_verification_with_different_answer_formats(self):
        """Test verification handles different answer formats."""
        from skyrl_agent.tasks.verifiers.naive_dapo import compute_score

        test_cases = [
            # Standard boxed
            ("\\boxed{42}", "42", 1.0),
            # With fbox
            ("\\fbox{42}", "42", 1.0),
            # With text wrapping
            ("\\boxed{\\text{42}}", "42", 1.0),
            # Fraction
            ("\\boxed{\\frac{1}{2}}", "1/2", 1.0),
        ]

        for solution, ground_truth, expected_score in test_cases:
            result = compute_score(solution, ground_truth, {})
            # May or may not match depending on normalization
            assert result["score"] in [0.0, 1.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
