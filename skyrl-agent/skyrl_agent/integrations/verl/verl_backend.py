from skyrl_agent.integrations.base import AsyncInferBackend, GeneratorOutput, GeneratorInput
from typing import Any, List, Dict
from loguru import logger


class VeRLBackend(AsyncInferBackend):
    def __init__(self, infer_engine, tokenizer: Any = None, cfg: Dict[str, Any] = None):
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        self.cfg = cfg

    async def async_generate_ids(
        self,
        input_ids: List[int],
        sampling_params: Dict[str, Any],
        request_id: str,
        **kwargs,
    ):
        response_str, meta_info = await self.infer_engine.generate(
            request_id=request_id,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
        )
        return response_str, meta_info

    async def async_generate_prompts(
        self,
        prompts: Any,
        sampling_params: Dict[str, Any],
        request_id: str = None,
        **kwargs,
    ) -> List[str]:
        """Generate text from string prompts using the VeRL backend.

        Args:
            prompts: A string prompt or list of string prompts.
            sampling_params: Dict of sampling parameters for generation.
            request_id: Optional request ID for tracking. If None, generates one.
            **kwargs: Additional keyword arguments passed to generation.

        Returns:
            Tuple of (generated_message, meta_info) for single prompt,
            or list of such tuples for multiple prompts.
        """
        import uuid

        if self.tokenizer is None:
            raise ValueError(
                "VeRLBackend requires a tokenizer to generate from prompts. "
                "Pass tokenizer to constructor or use async_generate_ids with pre-tokenized input."
            )

        if isinstance(prompts, str):
            # Single prompt
            input_ids = self.tokenizer.encode(prompts, add_special_tokens=True)
            req_id = request_id or str(uuid.uuid4())
            return await self.async_generate_ids(input_ids, sampling_params, request_id=req_id, **kwargs)
        else:
            # Multiple prompts
            results = []
            for i, prompt in enumerate(prompts):
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                req_id = f"{request_id}_{i}" if request_id else str(uuid.uuid4())
                result = await self.async_generate_ids(input_ids, sampling_params, request_id=req_id, **kwargs)
                results.append(result)
            return results


class VeRLGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Dict[str, Any]):
        self.result = result


class VeRLGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch: List[Dict[str, Any]] = []
        non_tensor_batch = input_batch.non_tensor_batch
        first_key = next(iter(non_tensor_batch.keys()))
        num_entries = len(non_tensor_batch[first_key])
        for i in range(num_entries):
            data_item: dict = {key: non_tensor_batch[key][i] for key in non_tensor_batch.keys()}
            self.input_batch.append(data_item)
        logger.info(f"input batch of size: {len(self.input_batch)}")
        logger.info(f"keys: {self.input_batch[0].keys()}")
