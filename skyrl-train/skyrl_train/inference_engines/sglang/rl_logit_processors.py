"""RL-specific logit processors for action masking and constraints.

This module provides logit processors designed for reinforcement learning use cases,
enabling constrained generation based on environment action spaces.

Usage:
    # In sampling_params, pass action mask for valid tokens:
    sampling_params = {
        "action_mask": [token_id_1, token_id_2, ...],  # Valid token IDs
        "max_new_tokens": 1,
        ...
    }

    # Or use disallowed tokens:
    sampling_params = {
        "disallowed_tokens": [bad_token_1, bad_token_2, ...],
        ...
    }

These are integrated into the generate() pipeline and converted to SGLang's
custom_logit_processor format automatically.
"""
from typing import List, Optional, Set, Union
import torch


class RLActionMaskProcessor:
    """Masks invalid actions based on environment constraints.

    Only allows generation of tokens in the valid_token_ids set.
    All other tokens are masked with -inf logits.

    This is useful for:
    - Discrete action RL where actions map to specific tokens
    - Tool calling with constrained tool names
    - Structured output with known vocabulary
    """

    def __init__(
        self,
        valid_token_ids: List[int],
        mask_value: float = float("-inf"),
    ):
        """Initialize the action mask processor.

        Args:
            valid_token_ids: List of token IDs that are valid actions.
            mask_value: Value to use for masked (invalid) tokens.
                Default is -inf which gives 0 probability.
        """
        self.valid_token_ids: Set[int] = set(valid_token_ids)
        self.mask_value = mask_value

    def __call__(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Apply action mask to logits.

        Args:
            logits: Raw logits from the model [vocab_size] or [batch, vocab_size].
            token_ids: Previously generated token IDs (context).

        Returns:
            Masked logits with invalid actions set to mask_value.
        """
        # Create mask tensor
        mask = torch.full_like(logits, self.mask_value)

        # Unmask valid tokens
        for tid in self.valid_token_ids:
            if tid < logits.shape[-1]:
                mask[..., tid] = 0

        return logits + mask

    def to_str(self) -> str:
        """Serialize to string format for SGLang custom_logit_processor."""
        # Format: "rl_action_mask:token1,token2,..."
        tokens_str = ",".join(str(t) for t in sorted(self.valid_token_ids))
        return f"rl_action_mask:{tokens_str}"

    @classmethod
    def from_str(cls, s: str) -> "RLActionMaskProcessor":
        """Deserialize from string format."""
        prefix = "rl_action_mask:"
        if not s.startswith(prefix):
            raise ValueError(f"Invalid format: {s}")
        tokens_str = s[len(prefix):]
        tokens = [int(t) for t in tokens_str.split(",") if t]
        return cls(tokens)


class DisallowedTokensProcessor:
    """Prevents generation of specific tokens.

    Sets logits for disallowed tokens to -inf, preventing their generation.

    This is useful for:
    - Filtering profanity or unsafe content
    - Preventing specific token patterns
    - Blocking certain action types in RL
    """

    def __init__(self, disallowed_token_ids: List[int]):
        """Initialize the disallowed tokens processor.

        Args:
            disallowed_token_ids: List of token IDs that should never be generated.
        """
        self.disallowed: Set[int] = set(disallowed_token_ids)

    def __call__(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Apply disallowed token masking to logits.

        Args:
            logits: Raw logits from the model.
            token_ids: Previously generated token IDs (context).

        Returns:
            Logits with disallowed tokens set to -inf.
        """
        for tid in self.disallowed:
            if tid < logits.shape[-1]:
                logits[..., tid] = float("-inf")
        return logits

    def to_str(self) -> str:
        """Serialize to string format for SGLang custom_logit_processor."""
        tokens_str = ",".join(str(t) for t in sorted(self.disallowed))
        return f"disallowed_tokens:{tokens_str}"

    @classmethod
    def from_str(cls, s: str) -> "DisallowedTokensProcessor":
        """Deserialize from string format."""
        prefix = "disallowed_tokens:"
        if not s.startswith(prefix):
            raise ValueError(f"Invalid format: {s}")
        tokens_str = s[len(prefix):]
        tokens = [int(t) for t in tokens_str.split(",") if t]
        return cls(tokens)


class TemperatureScaleProcessor:
    """Scales temperature dynamically based on context.

    This processor allows varying temperature based on generation progress,
    useful for:
    - Higher temperature early (exploration) vs lower later (exploitation)
    - Context-dependent exploration strategies
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        warmup_tokens: int = 10,
    ):
        """Initialize temperature scaling processor.

        Args:
            initial_temp: Temperature at start of generation.
            final_temp: Temperature after warmup_tokens.
            warmup_tokens: Number of tokens over which to decay temperature.
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.warmup_tokens = warmup_tokens

    def __call__(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Apply temperature scaling based on generation progress.

        Args:
            logits: Raw logits from the model.
            token_ids: Previously generated token IDs.

        Returns:
            Temperature-scaled logits.
        """
        n_generated = len(token_ids)

        if n_generated >= self.warmup_tokens:
            temp = self.final_temp
        else:
            # Linear interpolation
            progress = n_generated / self.warmup_tokens
            temp = self.initial_temp + progress * (self.final_temp - self.initial_temp)

        # Apply temperature (divide logits by temperature)
        if temp > 0:
            return logits / temp
        return logits


def create_rl_logit_processor(
    action_mask: Optional[List[int]] = None,
    disallowed_tokens: Optional[List[int]] = None,
) -> Optional[str]:
    """Create a logit processor string for RL constraints.

    This is a convenience function to create the appropriate logit processor
    based on the RL constraint type.

    Args:
        action_mask: List of valid token IDs (all others masked).
        disallowed_tokens: List of token IDs to block.

    Returns:
        Serialized logit processor string for SGLang, or None if no constraints.

    Example:
        # In generate() preprocessing:
        action_mask = sampling_params.pop("action_mask", None)
        if action_mask:
            sampling_params["custom_logit_processor"] = create_rl_logit_processor(
                action_mask=action_mask
            )
    """
    if action_mask:
        processor = RLActionMaskProcessor(action_mask)
        return processor.to_str()
    if disallowed_tokens:
        processor = DisallowedTokensProcessor(disallowed_tokens)
        return processor.to_str()
    return None


def parse_rl_logit_processor(processor_str: str) -> Union[
    RLActionMaskProcessor,
    DisallowedTokensProcessor,
    None,
]:
    """Parse a serialized logit processor string.

    Args:
        processor_str: Serialized processor from to_str().

    Returns:
        The appropriate processor instance, or None if unrecognized.
    """
    if processor_str.startswith("rl_action_mask:"):
        return RLActionMaskProcessor.from_str(processor_str)
    if processor_str.startswith("disallowed_tokens:"):
        return DisallowedTokensProcessor.from_str(processor_str)
    return None
