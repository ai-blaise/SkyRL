"""SGLang async server for SkyAgent with VeRL integration.

This module provides an SGLang-based async server that can be used as a rollout
backend in VeRL training with SkyAgent. It mirrors the functionality of
SkyAgentAsyncvLLMServer but uses SGLang as the inference engine.
"""

from typing import Any, Dict, Optional, Tuple

import ray

try:
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.server_args import ServerArgs
except ImportError:
    Engine = None
    GenerateReqInput = None
    ServerArgs = None

try:
    from verl.workers.rollout.async_server import AsyncServerBase
except ImportError:
    # Fallback base class if verl is not installed
    class AsyncServerBase:
        """Base class for async servers when verl is not installed."""
        def __init__(self, *args, **kwargs):
            pass


@ray.remote(num_cpus=1)
class SkyAgentAsyncSGLangServer(AsyncServerBase):
    """Async SGLang server for SkyAgent rollouts in VeRL training.

    This server provides an SGLang-based inference backend that can be used
    for agent rollouts during reinforcement learning training with VeRL.
    It supports:
    - Async generation from token IDs
    - Configurable sampling parameters
    - Output token extraction with logprobs
    - Stop reason tracking
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        **kwargs,
    ):
        """Initialize the SGLang async server.

        Args:
            model_path: Path to the model (HuggingFace model ID or local path).
            max_model_len: Maximum model context length. If None, uses model's default.
            gpu_memory_utilization: Fraction of GPU memory to use.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            trust_remote_code: Whether to trust remote code from HuggingFace.
            dtype: Data type for model weights ("auto", "float16", "bfloat16", "float32").
            **kwargs: Additional arguments passed to SGLang ServerArgs.
        """
        super().__init__()

        if Engine is None:
            raise ImportError(
                "SGLang is not installed. Install it with: pip install sglang"
            )

        self.model_path = model_path
        self.max_model_len = max_model_len

        # Build server args for SGLang
        server_args = ServerArgs(
            model_path=model_path,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs,
        )

        if max_model_len is not None:
            server_args.context_length = max_model_len

        # Initialize SGLang engine
        self.engine = Engine(server_args=server_args)

        # Get actual max model len from engine
        if self.max_model_len is None:
            self.max_model_len = getattr(
                self.engine.tokenizer_manager.hf_config,
                'max_position_embeddings',
                8192
            )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: Dict[str, Any],
        request_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text from token IDs.

        Args:
            prompt_ids: List of input token IDs.
            sampling_params: Sampling parameters dict. Common keys:
                - max_tokens: Maximum tokens to generate.
                - temperature: Sampling temperature.
                - top_p: Nucleus sampling probability.
                - top_k: Top-k sampling.
                - stop: Stop sequences (list of strings or token IDs).
            request_id: Unique request identifier for tracking.

        Returns:
            Tuple of (response_text, meta_info) where meta_info contains:
                - output_tokens: List of generated token IDs.
                - finish_reason: Why generation stopped ("stop", "length", etc.).
                - logprobs: Log probabilities if requested, else None.
        """
        # Defensive sanitization of sampling params
        sp: Dict[str, Any] = dict(sampling_params) if sampling_params is not None else {}

        # Ensure max_tokens exists and is valid
        if "max_tokens" not in sp or sp["max_tokens"] is None:
            max_tokens = self.max_model_len - len(prompt_ids)
            sp["max_tokens"] = int(max(1, max_tokens))
        else:
            try:
                sp["max_tokens"] = int(sp["max_tokens"])
            except (TypeError, ValueError):
                sp["max_tokens"] = int(self.max_model_len - len(prompt_ids))

        # Extract return_logprob before passing to SGLang
        return_logprob = sp.pop("return_logprob", False)

        # Create generation request
        obj = GenerateReqInput(
            input_ids=[prompt_ids],  # SGLang expects list of lists
            sampling_params=sp,
            return_logprob=return_logprob,
        )

        # Generate using SGLang engine
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        outputs = await generator.__anext__()

        # Extract response from SGLang output format
        # SGLang returns a list of outputs (one per prompt)
        output = outputs[0] if isinstance(outputs, list) else outputs

        # Handle different SGLang output formats
        if hasattr(output, 'text'):
            response_str = output.text
        elif hasattr(output, 'output_text'):
            response_str = output.output_text
        elif isinstance(output, dict):
            response_str = output.get('text', output.get('output_text', ''))
        else:
            response_str = str(output)

        # Extract output tokens
        if hasattr(output, 'output_ids'):
            output_tokens = list(output.output_ids)
        elif hasattr(output, 'token_ids'):
            output_tokens = list(output.token_ids)
        elif isinstance(output, dict):
            output_tokens = output.get('output_ids', output.get('token_ids', []))
        else:
            # Fallback: tokenize the response
            output_tokens = self.engine.tokenizer_manager.tokenizer.encode(
                response_str, add_special_tokens=False
            )

        # Extract finish reason
        if hasattr(output, 'finish_reason'):
            finish_reason = output.finish_reason
        elif hasattr(output, 'meta_info') and isinstance(output.meta_info, dict):
            finish_reason = output.meta_info.get('finish_reason', 'unknown')
        elif isinstance(output, dict):
            finish_reason = output.get('finish_reason', 'unknown')
        else:
            # Infer finish reason from output
            eos_token_id = getattr(
                self.engine.tokenizer_manager.tokenizer,
                'eos_token_id',
                None
            )
            if output_tokens and output_tokens[-1] == eos_token_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"

        # Extract logprobs if available
        logprobs = None
        if return_logprob:
            if hasattr(output, 'meta_info') and hasattr(output.meta_info, 'output_token_logprobs'):
                logprobs = output.meta_info.output_token_logprobs
            elif hasattr(output, 'logprobs'):
                logprobs = output.logprobs
            elif isinstance(output, dict):
                logprobs = output.get('logprobs', output.get('output_token_logprobs'))

        meta_info = {
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "logprobs": logprobs,
        }

        return response_str, meta_info

    async def health_check(self) -> bool:
        """Check if the server is healthy and ready to serve requests."""
        try:
            return self.engine is not None
        except Exception:
            return False

    def shutdown(self):
        """Shutdown the SGLang engine and release resources."""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:
                pass
            self.engine = None
