import logging
import ray
from packaging import version
from ray.actor import ActorHandle
from typing import Any, List, Dict, Literal, TYPE_CHECKING

logger = logging.getLogger(__name__)


def _get_default_sglang_attention_backend() -> str:
    """Auto-detect the best attention backend based on GPU compute capability.

    FlashAttention v3 (fa3) requires SM >= 80 and SM <= 90 (Ampere to Hopper).
    For GPUs outside this range (e.g., older GPUs or Blackwell SM 100+),
    fall back to flashinfer which has broader compatibility.

    Returns:
        "fa3" if GPU supports it, "flashinfer" otherwise.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, defaulting to flashinfer attention backend")
            return "flashinfer"

        # Get compute capability of first GPU
        major, minor = torch.cuda.get_device_capability(0)
        sm_version = major * 10 + minor

        # FA3 requires SM 80-90 (Ampere A100=80, Hopper H100=90)
        # SM 100+ (Blackwell) and SM < 80 should use flashinfer
        if 80 <= sm_version <= 90:
            logger.debug(f"GPU SM {sm_version} supports FA3, using fa3 attention backend")
            return "fa3"
        else:
            logger.info(
                f"GPU SM {sm_version} outside FA3 range (80-90), using flashinfer attention backend. "
                f"To override, set generator.engine_init_kwargs.attention_backend explicitly."
            )
            return "flashinfer"
    except Exception as e:
        logger.warning(f"Failed to detect GPU capability: {e}, defaulting to flashinfer")
        return "flashinfer"

if TYPE_CHECKING:
    from skyrl_train.weight_sync.transfer_strategy import WeightSyncInitInfo
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl_train.weight_sync import WeightUpdateRequest
from skyrl_train.inference_engines.utils import get_rendezvous_addr_port


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    def tp_size(self):
        return ray.get(self.inference_engine_actor.tp_size.remote())

    def pp_size(self):
        return ray.get(self.inference_engine_actor.pp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    def ep_size(self):
        """Get expert parallel size (for MoE models)."""
        return ray.get(self.inference_engine_actor.ep_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(input_batch=input_batch)

    async def generate_stream(self, input_batch: InferenceEngineInput):
        """Generate responses with streaming output.

        Note: Streaming over Ray requires special handling. This method
        returns an async iterator that yields StreamingChunk objects.
        """
        # For Ray actors, we need to handle streaming differently
        # The actor method returns an async generator
        async for chunk in self.inference_engine_actor.generate_stream.remote(input_batch=input_batch):
            yield chunk

    def supports_streaming(self) -> bool:
        """Check if this engine supports streaming generation."""
        return ray.get(self.inference_engine_actor.supports_streaming.remote())

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(init_info)

    async def update_named_weights(self, request: WeightUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.chat_completion.remote(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.completion.remote(request_payload)

    async def pause_generation(
        self, mode: Literal["abort", "in_place", "retract"] = "abort"
    ) -> None:
        return await self.inference_engine_actor.pause_generation.remote(mode)

    async def continue_generation(self) -> None:
        return await self.inference_engine_actor.continue_generation.remote()

    async def abort_generation(self) -> None:
        return await self.inference_engine_actor.abort_generation.remote()

    # LoRA adapter management
    async def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
        pinned: bool = False,
    ):
        """Load a LoRA adapter at runtime."""
        return await self.inference_engine_actor.load_lora_adapter.remote(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )

    async def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter at runtime."""
        return await self.inference_engine_actor.unload_lora_adapter.remote(lora_name)

    # Weight management
    async def get_weight_version(self) -> str | None:
        """Get the current weight version identifier."""
        return await self.inference_engine_actor.get_weight_version.remote()

    async def update_weight_version(
        self,
        new_version: str,
        abort_all_requests: bool = True,
    ) -> None:
        """Update the weight version identifier."""
        return await self.inference_engine_actor.update_weight_version.remote(
            new_version=new_version,
            abort_all_requests=abort_all_requests,
        )

    async def load_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        flush_cache: bool = True,
    ) -> None:
        """Load weights from a checkpoint file or directory."""
        return await self.inference_engine_actor.load_weights_from_disk.remote(
            model_path=model_path,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    async def get_weights_by_name(self, name: str):
        """Get weights for a specific parameter by name."""
        return await self.inference_engine_actor.get_weights_by_name.remote(name)

    async def validate_weights(
        self,
        weight_names: List[str] | None = None,
        check_nan: bool = True,
        check_inf: bool = True,
    ) -> Dict[str, Any]:
        """Validate model weights for NaN/Inf values."""
        return await self.inference_engine_actor.validate_weights.remote(
            weight_names=weight_names,
            check_nan=check_nan,
            check_inf=check_inf,
        )

    async def check_weight_sync_integrity(
        self,
        expected_version: str | None = None,
    ) -> Dict[str, Any]:
        """Check weight synchronization integrity."""
        return await self.inference_engine_actor.check_weight_sync_integrity.remote(
            expected_version=expected_version
        )

    # Sleep/wake variants
    async def sleep_weights_only(self, **kwargs) -> None:
        """Put only weights to sleep (offload to CPU)."""
        return await self.inference_engine_actor.sleep_weights_only.remote(**kwargs)

    async def wake_up_weights_only(self, **kwargs) -> None:
        """Wake up only weights (load back to GPU)."""
        return await self.inference_engine_actor.wake_up_weights_only.remote(**kwargs)

    async def sleep_all(self, **kwargs) -> None:
        """Sleep both weights and KV cache."""
        return await self.inference_engine_actor.sleep_all.remote(**kwargs)

    async def wake_up_all(self, **kwargs) -> None:
        """Wake up both weights and KV cache."""
        return await self.inference_engine_actor.wake_up_all.remote(**kwargs)

    # Cache management
    async def clear_hicache_storage(self) -> bool:
        """Clear only the hierarchical cache storage tier (disk)."""
        return await self.inference_engine_actor.clear_hicache_storage.remote()

    # Overlapped weight sync API
    def supports_overlapped_weight_sync(self) -> bool:
        """Check if overlapped weight sync is supported."""
        return ray.get(self.inference_engine_actor.supports_overlapped_weight_sync.remote())

    async def start_weight_transfer(self, request: WeightUpdateRequest):
        """Start transferring weights in the background."""
        return await self.inference_engine_actor.start_weight_transfer.remote(request)

    async def finish_weight_transfer(self, handle, flush_cache: bool = True) -> None:
        """Complete the weight transfer and apply weights."""
        return await self.inference_engine_actor.finish_weight_transfer.remote(
            handle=handle,
            flush_cache=flush_cache,
        )

    async def overlapped_weight_sync(
        self,
        request: WeightUpdateRequest,
        pause_mode: str = "in_place",
    ) -> None:
        """Perform overlapped weight synchronization."""
        return await self.inference_engine_actor.overlapped_weight_sync.remote(
            request=request,
            pause_mode=pause_mode,
        )

    # Session management methods
    def supports_sessions(self) -> bool:
        return ray.get(self.inference_engine_actor.supports_sessions.remote())

    async def open_session(
        self,
        capacity_of_str_len: int = 8192,
        session_id: str | None = None,
    ) -> str | None:
        return await self.inference_engine_actor.open_session.remote(
            capacity_of_str_len=capacity_of_str_len,
            session_id=session_id,
        )

    async def close_session(self, session_id: str) -> None:
        return await self.inference_engine_actor.close_session.remote(session_id)

    async def generate_with_session(
        self,
        session_id: str,
        input_batch: InferenceEngineInput,
        rid: str | None = None,
        offset: int | None = None,
        replace: bool = False,
        drop_previous_output: bool = False,
    ) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate_with_session.remote(
            session_id=session_id,
            input_batch=input_batch,
            rid=rid,
            offset=offset,
            replace=replace,
            drop_previous_output=drop_previous_output,
        )

    # Session introspection methods
    async def get_session_info(self, session_id: str) -> Dict[str, Any] | None:
        """Get information about a specific session."""
        return await self.inference_engine_actor.get_session_info.remote(session_id)

    async def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return await self.inference_engine_actor.list_sessions.remote()

    async def fork_session(
        self,
        source_session_id: str,
        target_session_id: str | None = None,
        offset: int = -1,
    ) -> str | None:
        """Fork a session at a specific point."""
        return await self.inference_engine_actor.fork_session.remote(
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            offset=offset,
        )

    # Score API for reward models
    def supports_scoring(self) -> bool:
        """Check if this engine supports reward model scoring."""
        return ray.get(self.inference_engine_actor.supports_scoring.remote())

    async def score(
        self,
        input_ids: List[List[int]],
        output_ids: List[List[int]],
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Compute reward scores using a reward model."""
        return await self.inference_engine_actor.score.remote(
            input_ids=input_ids,
            output_ids=output_ids,
            return_hidden_states=return_hidden_states,
        )

    # Server info API
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and configuration."""
        return await self.inference_engine_actor.get_server_info.remote()

    async def get_memory_pool_size(self) -> Dict[str, int]:
        """Get KV cache memory pool size information."""
        return await self.inference_engine_actor.get_memory_pool_size.remote()

    # Model saving API
    async def save_sharded_model(
        self,
        save_path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> bool:
        """Save model weights to disk in sharded format."""
        return await self.inference_engine_actor.save_sharded_model.remote(
            save_path=save_path,
            pattern=pattern,
            max_size=max_size,
        )

    async def save_remote_model(
        self,
        remote_path: str,
        storage_type: str = "s3",
    ) -> bool:
        """Save model weights to remote storage."""
        return await self.inference_engine_actor.save_remote_model.remote(
            remote_path=remote_path,
            storage_type=storage_type,
        )

    # Decode API
    async def decode(
        self,
        token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode token IDs to text strings."""
        return await self.inference_engine_actor.decode.remote(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    # Profiling API
    async def start_profile(self) -> bool:
        """Start profiling the inference engine."""
        return await self.inference_engine_actor.start_profile.remote()

    async def stop_profile(self) -> Dict[str, Any] | None:
        """Stop profiling and collect results."""
        return await self.inference_engine_actor.stop_profile.remote()

    # Embedding API
    def supports_embeddings(self) -> bool:
        """Check if this engine supports embedding generation."""
        return ray.get(self.inference_engine_actor.supports_embeddings.remote())

    async def encode(
        self,
        texts: List[str],
        dimensions: int | None = None,
    ) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        return await self.inference_engine_actor.encode.remote(
            texts=texts,
            dimensions=dimensions,
        )

    async def encode_single(
        self,
        text: str,
        dimensions: int | None = None,
    ) -> List[float]:
        """Get embedding for a single text."""
        return await self.inference_engine_actor.encode_single.remote(
            text=text,
            dimensions=dimensions,
        )

    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        dimensions: int | None = None,
    ) -> float:
        """Compute cosine similarity between two texts."""
        return await self.inference_engine_actor.compute_similarity.remote(
            text1=text1,
            text2=text2,
            dimensions=dimensions,
        )


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    sleep_level=2,  # we only set to 1 for unit tests that do not explicitly sync weights or for LoRA
    enable_lora=False,
    max_lora_rank=64,
    max_loras=1,
    fully_sharded_loras=False,
    engine_init_kwargs: Dict[str, Any] = {},
    rope_scaling: Dict[str, Any] = {},
    rope_theta: float | None = None,
    speculative_decoding: Dict[str, Any] | None = None,
    kv_cache: Dict[str, Any] | None = None,
    quantization: Dict[str, Any] | None = None,
    custom_logit_processor: Dict[str, Any] | None = None,
    structured_output: Dict[str, Any] | None = None,
    cuda_graph: Dict[str, Any] | None = None,
    piecewise_cuda_graph: Dict[str, Any] | None = None,
    torch_compile: Dict[str, Any] | None = None,
    attention: Dict[str, Any] | None = None,
    lora_config: Dict[str, Any] | None = None,
    scheduling: Dict[str, Any] | None = None,
    disaggregation: Dict[str, Any] | None = None,
    multi_node: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
    deterministic_inference: Dict[str, Any] | None = None,
    load_balancing: Dict[str, Any] | None = None,
    health_checks: Dict[str, Any] | None = None,
    hierarchical_cache: Dict[str, Any] | None = None,
    cpu_offload: Dict[str, Any] | None = None,
    sessions: Dict[str, Any] | None = None,
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface
    instances.
    """
    from skyrl_train.utils import ray_noset_visible_devices, get_all_env_variables, get_ray_pg_ready_with_timeout
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import VLLMRayActor, AsyncVLLMRayActor

        # if a dev version is being used, skip the version check
        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    # NOTE: we use the ray backend for tensor parallel size > 1 or pipeline parallel size > 1
    # to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if (tensor_parallel_size == 1 and pipeline_parallel_size == 1) else "ray"
    data_parallel_backend = "mp"
    use_hybrid_engine = shared_pg is not None
    num_gpus_per_actor = int(tensor_parallel_size == 1 and pipeline_parallel_size == 1)

    if use_hybrid_engine and tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        # Every worker will use 0.2 GPU, so that we can schedule
        # inference and training workers on the same GPUs.
        num_gpus_per_actor = 0.2

    per_engine_gpu_count = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    if not use_hybrid_engine:
        # Create a big placement group to ensure that all inference engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * per_engine_gpu_count)]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    for i in range(num_inference_engines):
        base_pg_index = i * per_engine_gpu_count

        # Get DP group rendezvous (addr, port) on the same node as DP rank 0 for this engine.
        data_parallel_address, data_parallel_rpc_port = get_rendezvous_addr_port(shared_pg, base_pg_index)

        if backend == "vllm":
            if async_engine:
                actor_class = AsyncVLLMRayActor
            else:
                actor_class = VLLMRayActor

            lora_kwargs = {
                "enable_lora": enable_lora,
                "max_lora_rank": max_lora_rank,
                "max_loras": max_loras,
                "fully_sharded_loras": fully_sharded_loras,
            }

            rope_engine_kwargs = {}
            if rope_scaling:
                rope_engine_kwargs["rope_scaling"] = rope_scaling
                if "max_model_len" not in engine_init_kwargs:
                    rope_factor = rope_scaling.get("factor", None)
                    rope_max_pos = rope_scaling.get("original_max_position_embeddings", None)
                    assert rope_factor is not None, "Please provide rope scaling `factor` to compute model max length"
                    assert (
                        rope_max_pos is not None
                    ), "Please provide rope `original_max_position_embeddings` to compute model max length"
                    rope_engine_kwargs["max_model_len"] = int(rope_factor * rope_max_pos)
            if rope_theta is not None:
                rope_engine_kwargs["rope_theta"] = rope_theta

            # Launch one actor per DP rank
            for dp_rank in range(data_parallel_size):

                # Contiguous TP*PP slice reserved for a single DP rank.
                tp_pp_size = tensor_parallel_size * pipeline_parallel_size
                base_dp_pg_index = base_pg_index + dp_rank * tp_pp_size
                dp_rank_bundles = (
                    list(range(base_dp_pg_index, base_dp_pg_index + tp_pp_size)) if tp_pp_size > 1 else None
                )
                dp_rank_sched = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=base_dp_pg_index,
                )

                dp_kwargs = (
                    {
                        "data_parallel_backend": data_parallel_backend,
                        "data_parallel_size": data_parallel_size,
                        "data_parallel_rank": dp_rank,
                        "data_parallel_address": data_parallel_address,
                        "data_parallel_rpc_port": data_parallel_rpc_port,
                    }
                    if data_parallel_size > 1
                    else {}
                )

                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=dp_rank_sched,
                ).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    distributed_executor_backend=distributed_executor_backend,
                    seed=seed + i * data_parallel_size + dp_rank,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                    gpu_memory_utilization=gpu_memory_utilization,
                    bundle_indices=dp_rank_bundles,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    enable_sleep_mode=inference_engine_enable_sleep,
                    noset_visible_devices=noset_visible_devices,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_logprobs=1,  # only need chosen-token logprobs
                    **dp_kwargs,
                    **engine_init_kwargs,
                    **lora_kwargs,
                    **rope_engine_kwargs,
                )
                inference_engine_actors.append(engine)
        elif backend == "sglang":
            # NOTE: there is no async / sync engine distinction in SGLang

            # Warn about vLLM-specific params that are ignored for SGLang
            if enforce_eager:
                logger.warning("SGLang backend: 'enforce_eager' is ignored (SGLang handles eager mode differently)")
            if vllm_v1_disable_multiproc:
                logger.warning("SGLang backend: 'vllm_v1_disable_multiproc' is ignored (vLLM-specific param)")
            # SGLang supports LoRA, EP, PP, DP natively - params will be passed to engine
            if enable_lora:
                logger.info(f"SGLang backend: LoRA enabled with max_lora_rank from engine_init_kwargs")
            if expert_parallel_size > 1:
                logger.info(f"SGLang backend: Using expert parallelism with ep_size={expert_parallel_size}")
            if pipeline_parallel_size > 1:
                logger.info(f"SGLang backend: Using pipeline parallelism with pp_size={pipeline_parallel_size}")
            if data_parallel_size > 1:
                logger.info(f"SGLang backend: Using data parallelism with dp_size={data_parallel_size}")
            # RoPE scaling configuration is handled below with validation

            # Build LD_LIBRARY_PATH to propagate CUDA library paths to Ray worker subprocesses.
            # SGLang uses multiprocessing with "spawn" method, which doesn't inherit environment
            # variables automatically. The CUDA libraries (libcudart.so.12) are often installed
            # via pip packages (nvidia-cuda-runtime-cu12) in the venv's site-packages.
            import os
            import sys

            # Start with any existing LD_LIBRARY_PATH
            driver_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

            # Find CUDA library paths in site-packages (pip-installed nvidia packages)
            cuda_lib_paths = []
            for site_path in sys.path:
                if "site-packages" in site_path:
                    # Check for nvidia cuda_runtime package
                    cuda_runtime_lib = os.path.join(site_path, "nvidia", "cuda_runtime", "lib")
                    if os.path.isdir(cuda_runtime_lib):
                        cuda_lib_paths.append(cuda_runtime_lib)
                    # Check for nvidia cublas package
                    cublas_lib = os.path.join(site_path, "nvidia", "cublas", "lib")
                    if os.path.isdir(cublas_lib):
                        cuda_lib_paths.append(cublas_lib)
                    # Check for nvidia cudnn package
                    cudnn_lib = os.path.join(site_path, "nvidia", "cudnn", "lib")
                    if os.path.isdir(cudnn_lib):
                        cuda_lib_paths.append(cudnn_lib)
                    # Check for nvidia nccl package
                    nccl_lib = os.path.join(site_path, "nvidia", "nccl", "lib")
                    if os.path.isdir(nccl_lib):
                        cuda_lib_paths.append(nccl_lib)

            # Combine paths
            if cuda_lib_paths:
                if driver_ld_library_path:
                    driver_ld_library_path = ":".join(cuda_lib_paths) + ":" + driver_ld_library_path
                else:
                    driver_ld_library_path = ":".join(cuda_lib_paths)

            bundle_indices = None
            if per_engine_gpu_count > 1:
                bundle_indices = list(range(i * per_engine_gpu_count, (i + 1) * per_engine_gpu_count))

            # For SGLang with TP*PP > 1, we can't use placement group bundle scheduling because:
            # 1. The placement group has 1-GPU bundles (designed for vLLM's distributed executor)
            # 2. SGLang needs all TP*PP GPUs in one actor (it spawns child processes, not Ray actors)
            # 3. Ray can't fit a 2-GPU request into a 1-GPU bundle
            # Solution: For TP>1, don't use placement group - let Ray schedule with normal GPU allocation
            tp_pp_size = tensor_parallel_size * pipeline_parallel_size
            if tp_pp_size > 1:
                scheduling_strategy = "DEFAULT"  # Let Ray schedule freely with num_gpus request
            else:
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i * per_engine_gpu_count,
                )

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            # Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527

            # Convert OmegaConf DictConfig to regular dict to support .pop() operations
            from omegaconf import OmegaConf
            if hasattr(engine_init_kwargs, '_metadata'):  # Check if it's an OmegaConf object
                engine_init_kwargs_dict = OmegaConf.to_container(engine_init_kwargs, resolve=True)
            else:
                engine_init_kwargs_dict = dict(engine_init_kwargs)

            # Extract SGLang-specific params before creating the remote function
            # Auto-detect attention backend if not specified (fa3 requires SM 80-90)
            default_attention_backend = _get_default_sglang_attention_backend()
            sglang_attention_backend = engine_init_kwargs_dict.pop("attention_backend", default_attention_backend)
            sglang_mm_attention_backend = engine_init_kwargs_dict.pop("mm_attention_backend", default_attention_backend)
            sglang_max_lora_rank = engine_init_kwargs_dict.pop("max_lora_rank", 64)
            sglang_max_loras_per_batch = engine_init_kwargs_dict.pop("max_loras_per_batch", 8)
            sglang_lora_backend = engine_init_kwargs_dict.pop("lora_backend", "csgmv")

            # Process speculative decoding configuration
            speculative_kwargs = {}
            if speculative_decoding and speculative_decoding.get("enabled", False):
                algorithm = speculative_decoding.get("algorithm")
                if algorithm is None:
                    raise ValueError(
                        "speculative_decoding.algorithm must be specified when enabled=True. "
                        "Supported: 'eagle', 'eagle3', 'standalone', 'ngram'"
                    )

                algorithm = algorithm.upper()
                if algorithm not in ["EAGLE", "EAGLE3", "STANDALONE", "NGRAM"]:
                    raise ValueError(
                        f"Invalid speculative algorithm: {algorithm}. "
                        f"Supported: 'eagle', 'eagle3', 'standalone', 'ngram'"
                    )

                # Validate LoRA compatibility
                if enable_lora and algorithm not in ["NGRAM", "NONE"]:
                    raise ValueError(
                        f"LoRA is only compatible with NGRAM speculative decoding, got: {algorithm}"
                    )

                # Validate attention backend for topk > 1
                eagle_topk = speculative_decoding.get("eagle_topk")
                if eagle_topk is not None and eagle_topk > 1:
                    if sglang_attention_backend not in ["flashinfer", "fa3"]:
                        raise ValueError(
                            f"speculative_decoding.eagle_topk > 1 requires 'flashinfer' or 'fa3' attention backend, "
                            f"got: {sglang_attention_backend}"
                        )

                # Core speculative parameters
                speculative_kwargs["speculative_algorithm"] = algorithm

                draft_model_path = speculative_decoding.get("draft_model_path")
                if draft_model_path:
                    speculative_kwargs["speculative_draft_model_path"] = draft_model_path
                    speculative_kwargs["speculative_draft_model_revision"] = speculative_decoding.get(
                        "draft_model_revision", "main"
                    )
                elif algorithm in ["EAGLE", "EAGLE3", "STANDALONE"]:
                    # For non-MTP models, draft_model_path is required for EAGLE/STANDALONE
                    logger.warning(
                        f"No draft_model_path specified for {algorithm}. "
                        f"SGLang will auto-detect if model has built-in MTP support."
                    )

                # Decoding parameters (let SGLang auto-choose if None)
                if speculative_decoding.get("num_steps") is not None:
                    speculative_kwargs["speculative_num_steps"] = speculative_decoding["num_steps"]
                if eagle_topk is not None:
                    speculative_kwargs["speculative_eagle_topk"] = eagle_topk
                if speculative_decoding.get("num_draft_tokens") is not None:
                    speculative_kwargs["speculative_num_draft_tokens"] = speculative_decoding["num_draft_tokens"]

                # Acceptance thresholds
                speculative_kwargs["speculative_accept_threshold_single"] = speculative_decoding.get(
                    "accept_threshold_single", 1.0
                )
                speculative_kwargs["speculative_accept_threshold_acc"] = speculative_decoding.get(
                    "accept_threshold_acc", 1.0
                )

                # Advanced options
                if speculative_decoding.get("attention_mode"):
                    speculative_kwargs["speculative_attention_mode"] = speculative_decoding["attention_mode"]
                if speculative_decoding.get("draft_attention_backend"):
                    speculative_kwargs["speculative_draft_attention_backend"] = speculative_decoding["draft_attention_backend"]

                # N-gram specific options
                if algorithm == "NGRAM":
                    ngram_config = speculative_decoding.get("ngram", {})
                    if ngram_config.get("min_match_window_size") is not None:
                        speculative_kwargs["speculative_ngram_min_match_window_size"] = ngram_config["min_match_window_size"]
                    if ngram_config.get("max_match_window_size") is not None:
                        speculative_kwargs["speculative_ngram_max_match_window_size"] = ngram_config["max_match_window_size"]
                    if ngram_config.get("min_bfs_breadth") is not None:
                        speculative_kwargs["speculative_ngram_min_bfs_breadth"] = ngram_config["min_bfs_breadth"]
                    if ngram_config.get("max_bfs_breadth") is not None:
                        speculative_kwargs["speculative_ngram_max_bfs_breadth"] = ngram_config["max_bfs_breadth"]
                    if ngram_config.get("match_type"):
                        speculative_kwargs["speculative_ngram_match_type"] = ngram_config["match_type"]
                    if ngram_config.get("branch_length") is not None:
                        speculative_kwargs["speculative_ngram_branch_length"] = ngram_config["branch_length"]
                    if ngram_config.get("capacity") is not None:
                        speculative_kwargs["speculative_ngram_capacity"] = ngram_config["capacity"]

                # Multi-layer EAGLE and CPU backup
                if speculative_decoding.get("enable_multi_layer_eagle"):
                    speculative_kwargs["enable_multi_layer_eagle"] = True
                if speculative_decoding.get("enable_draft_weights_cpu_backup"):
                    speculative_kwargs["enable_draft_weights_cpu_backup"] = True

                logger.info(
                    f"SGLang backend: Speculative decoding enabled with algorithm={algorithm}, "
                    f"draft_model={draft_model_path or 'auto'}"
                )

            # Process FP8 KV cache configuration
            kv_cache_kwargs = {}
            if kv_cache is not None:
                kv_cache_dtype = kv_cache.get("dtype", "auto")

                # Validate kv_cache_dtype
                valid_kv_dtypes = ["auto", "fp8_e4m3", "fp8_e5m2", "bf16", "bfloat16"]
                if kv_cache_dtype not in valid_kv_dtypes:
                    raise ValueError(
                        f"Invalid kv_cache.dtype: {kv_cache_dtype}. "
                        f"Supported: {valid_kv_dtypes}"
                    )

                # Only pass if not auto (let SGLang use its default behavior for auto)
                if kv_cache_dtype != "auto":
                    kv_cache_kwargs["kv_cache_dtype"] = kv_cache_dtype

                    # Validate FlashAttention3 + fp8_e5m2 incompatibility
                    if kv_cache_dtype == "fp8_e5m2" and sglang_attention_backend == "fa3":
                        logger.warning(
                            "SGLang backend: FlashAttention3 only supports fp8_e4m3. "
                            "Using fp8_e5m2 will automatically fall back to triton backend."
                        )

                    # Log memory savings info
                    if kv_cache_dtype.startswith("fp8"):
                        logger.info(
                            f"SGLang backend: FP8 KV cache enabled with dtype={kv_cache_dtype}. "
                            "Expected ~50% memory reduction for KV cache."
                        )

                # Quantization param path for scaling factors
                quant_param_path = kv_cache.get("quantization_param_path")
                if quant_param_path is not None:
                    kv_cache_kwargs["quantization_param_path"] = quant_param_path
                    logger.info(
                        f"SGLang backend: Using KV cache scaling factors from {quant_param_path}"
                    )
                elif kv_cache_dtype and kv_cache_dtype.startswith("fp8"):
                    logger.warning(
                        "SGLang backend: Using FP8 KV cache without scaling factors. "
                        "This may reduce accuracy. Consider providing quantization_param_path."
                    )

                # FP8 GEMM backend
                fp8_gemm_backend = kv_cache.get("fp8_gemm_backend", "auto")
                valid_fp8_backends = ["auto", "deep_gemm", "flashinfer_trtllm", "cutlass", "triton", "aiter"]
                if fp8_gemm_backend not in valid_fp8_backends:
                    raise ValueError(
                        f"Invalid kv_cache.fp8_gemm_backend: {fp8_gemm_backend}. "
                        f"Supported: {valid_fp8_backends}"
                    )
                if fp8_gemm_backend != "auto":
                    kv_cache_kwargs["fp8_gemm_runner_backend"] = fp8_gemm_backend

            # Process model quantization configuration
            quantization_kwargs = {}
            if quantization is not None:
                quant_method = quantization.get("method")

                if quant_method is not None:
                    # Validate quantization method
                    valid_quant_methods = [
                        # Weight-only quantization
                        "awq", "awq_marlin", "gptq", "gptq_marlin", "marlin",
                        "gguf", "bitsandbytes", "auto-round",
                        # FP8 quantization
                        "fp8", "w8a8_fp8", "modelopt_fp8", "modelopt",
                        # FP4/INT4 quantization
                        "modelopt_fp4", "w4afp8", "mxfp4", "petit_nvfp4",
                        # INT8 quantization
                        "w8a8_int8",
                        # MoE and other
                        "moe_wna16", "qoq", "compressed-tensors", "modelslim",
                    ]
                    if quant_method not in valid_quant_methods:
                        raise ValueError(
                            f"Invalid quantization.method: {quant_method}. "
                            f"Supported methods: {valid_quant_methods}"
                        )

                    quantization_kwargs["quantization"] = quant_method
                    logger.info(f"SGLang backend: Using quantization method '{quant_method}'")

                # Load format
                load_format = quantization.get("load_format", "auto")
                valid_load_formats = [
                    "auto", "pt", "safetensors", "gguf", "bitsandbytes",
                    "flash_rl", "npcache", "dummy", "sharded_state",
                    "layered", "remote", "remote_instance", "fastsafetensors", "private",
                    "nmoe",  # nmoe native checkpoint format (rd.pt + dp_rank_*.pt)
                ]
                if load_format not in valid_load_formats:
                    raise ValueError(
                        f"Invalid quantization.load_format: {load_format}. "
                        f"Supported: {valid_load_formats}"
                    )
                if load_format != "auto":
                    quantization_kwargs["load_format"] = load_format

                    # Validate gguf requires load_format=gguf
                    if quant_method == "gguf" and load_format != "gguf":
                        raise ValueError(
                            "quantization.method='gguf' requires load_format='gguf'"
                        )

                # FP32 language model head
                if quantization.get("enable_fp32_lm_head", False):
                    quantization_kwargs["enable_fp32_lm_head"] = True
                    logger.info("SGLang backend: Keeping language model head in FP32 for accuracy")

                # ModelOpt-specific configuration
                modelopt_config = quantization.get("modelopt", {})
                if modelopt_config:
                    modelopt_quant_type = modelopt_config.get("quant_type")
                    if modelopt_quant_type:
                        valid_modelopt_types = ["fp8", "int4_awq", "w4a8_awq", "nvfp4", "nvfp4_awq"]
                        if modelopt_quant_type not in valid_modelopt_types:
                            raise ValueError(
                                f"Invalid quantization.modelopt.quant_type: {modelopt_quant_type}. "
                                f"Supported: {valid_modelopt_types}"
                            )
                        quantization_kwargs["modelopt_quant"] = modelopt_quant_type

                    if modelopt_config.get("checkpoint_restore_path"):
                        quantization_kwargs["modelopt_checkpoint_restore_path"] = modelopt_config["checkpoint_restore_path"]

                    if modelopt_config.get("checkpoint_save_path"):
                        quantization_kwargs["modelopt_checkpoint_save_path"] = modelopt_config["checkpoint_save_path"]

                    if modelopt_config.get("export_path"):
                        quantization_kwargs["modelopt_export_path"] = modelopt_config["export_path"]

                    if modelopt_config.get("quantize_and_serve", False):
                        quantization_kwargs["quantize_and_serve"] = True
                        logger.info("SGLang backend: Will quantize model on-the-fly (slower startup)")

                # Draft model quantization (for speculative decoding)
                draft_quant = quantization.get("draft_model_quantization")
                if draft_quant:
                    quantization_kwargs["speculative_draft_model_quantization"] = draft_quant

            # Process custom logit processor configuration
            custom_logit_processor_kwargs = {}
            if custom_logit_processor is not None:
                if custom_logit_processor.get("enabled", False):
                    custom_logit_processor_kwargs["enable_custom_logit_processor"] = True
                    logger.info(
                        "SGLang backend: Custom logit processor support enabled. "
                        "Processors can be passed via sampling_params.custom_logit_processor"
                    )

            # Process structured output / grammar backend configuration
            structured_output_kwargs = {}
            if structured_output is not None:
                grammar_backend = structured_output.get("grammar_backend")
                if grammar_backend is not None:
                    valid_backends = ["xgrammar", "outlines", "llguidance", "none"]
                    if grammar_backend not in valid_backends:
                        raise ValueError(
                            f"structured_output.grammar_backend must be one of {valid_backends}, "
                            f"got: {grammar_backend}"
                        )
                    structured_output_kwargs["grammar_backend"] = grammar_backend
                    logger.info(f"SGLang backend: Grammar backend set to '{grammar_backend}'")

            # Process CUDA graph configuration
            cuda_graph_kwargs = {}
            if cuda_graph is not None:
                # Disable CUDA graphs entirely
                if cuda_graph.get("disable", False):
                    cuda_graph_kwargs["disable_cuda_graph"] = True
                    logger.info("SGLang backend: CUDA graphs disabled")
                else:
                    # Maximum batch size for CUDA graph capture
                    max_bs = cuda_graph.get("max_bs")
                    if max_bs is not None:
                        if not isinstance(max_bs, int) or max_bs < 1:
                            raise ValueError(
                                f"cuda_graph.max_bs must be a positive integer, got: {max_bs}"
                            )
                        cuda_graph_kwargs["cuda_graph_max_bs"] = max_bs
                        logger.info(f"SGLang backend: CUDA graph max batch size set to {max_bs}")

                    # Explicit list of batch sizes to capture
                    batch_sizes = cuda_graph.get("batch_sizes")
                    if batch_sizes is not None:
                        if not isinstance(batch_sizes, (list, tuple)) or not all(isinstance(x, int) for x in batch_sizes):
                            raise ValueError(
                                f"cuda_graph.batch_sizes must be a list of integers, got: {batch_sizes}"
                            )
                        cuda_graph_kwargs["cuda_graph_bs"] = list(batch_sizes)
                        logger.info(f"SGLang backend: CUDA graph batch sizes: {batch_sizes}")

                    # Disable batch size padding optimization
                    if cuda_graph.get("disable_padding", False):
                        cuda_graph_kwargs["disable_cuda_graph_padding"] = True
                        logger.info("SGLang backend: CUDA graph padding disabled (capturing all batch sizes)")

                    # Enable profiling during CUDA graph capture
                    if cuda_graph.get("enable_profiling", False):
                        cuda_graph_kwargs["enable_profile_cuda_graph"] = True
                        logger.info("SGLang backend: CUDA graph profiling enabled")

                    # Enable garbage collection during CUDA graph capture
                    if cuda_graph.get("enable_gc", False):
                        cuda_graph_kwargs["enable_cudagraph_gc"] = True
                        logger.info("SGLang backend: GC enabled during CUDA graph capture")

            # Process piecewise CUDA graph configuration
            piecewise_cuda_graph_kwargs = {}
            if piecewise_cuda_graph is not None:
                if piecewise_cuda_graph.get("enabled", False):
                    piecewise_cuda_graph_kwargs["enable_piecewise_cuda_graph"] = True
                    logger.info("SGLang backend: Piecewise CUDA graphs enabled for prefill optimization")

                    # Maximum token count for piecewise CUDA graph capture
                    max_tokens = piecewise_cuda_graph.get("max_tokens")
                    if max_tokens is not None:
                        if not isinstance(max_tokens, int) or max_tokens < 1:
                            raise ValueError(
                                f"piecewise_cuda_graph.max_tokens must be a positive integer, got: {max_tokens}"
                            )
                        piecewise_cuda_graph_kwargs["piecewise_cuda_graph_max_tokens"] = max_tokens

                    # Explicit list of token counts to capture
                    token_counts = piecewise_cuda_graph.get("token_counts")
                    if token_counts is not None:
                        if not isinstance(token_counts, (list, tuple)) or not all(isinstance(x, int) for x in token_counts):
                            raise ValueError(
                                f"piecewise_cuda_graph.token_counts must be a list of integers, got: {token_counts}"
                            )
                        piecewise_cuda_graph_kwargs["piecewise_cuda_graph_tokens"] = list(token_counts)

                    # Compiler for piecewise CUDA graphs
                    compiler = piecewise_cuda_graph.get("compiler", "eager")
                    valid_compilers = ["eager", "inductor"]
                    if compiler not in valid_compilers:
                        raise ValueError(
                            f"piecewise_cuda_graph.compiler must be one of {valid_compilers}, got: {compiler}"
                        )
                    piecewise_cuda_graph_kwargs["piecewise_cuda_graph_compiler"] = compiler

            # Process torch.compile configuration
            torch_compile_kwargs = {}
            if torch_compile is not None:
                if torch_compile.get("enabled", False):
                    torch_compile_kwargs["enable_torch_compile"] = True
                    logger.info("SGLang backend: torch.compile optimization enabled")

                    # Debug mode for torch.compile
                    if torch_compile.get("debug_mode", False):
                        torch_compile_kwargs["enable_torch_compile_debug_mode"] = True
                        logger.info("SGLang backend: torch.compile debug mode enabled")

                    # Maximum batch size for torch.compile
                    max_bs = torch_compile.get("max_bs", 32)
                    if not isinstance(max_bs, int) or max_bs < 1:
                        raise ValueError(
                            f"torch_compile.max_bs must be a positive integer, got: {max_bs}"
                        )
                    torch_compile_kwargs["torch_compile_max_bs"] = max_bs

            # Process attention backend configuration
            attention_kwargs = {}
            if attention is not None:
                # Valid attention backends
                valid_backends = [
                    # Common backends
                    "triton", "torch_native", "flex_attention", "nsa",
                    # NVIDIA backends
                    "flashinfer", "fa3", "fa4", "flashmla", "cutlass_mla",
                    "trtllm_mha", "trtllm_mla", "dual_chunk_flash_attn",
                    # AMD backends
                    "aiter", "wave",
                    # Other platforms
                    "intel_amx", "intel_xpu", "ascend",
                ]

                # Main attention backend
                backend = attention.get("backend")
                if backend is not None:
                    if backend not in valid_backends:
                        raise ValueError(
                            f"Invalid attention.backend: {backend}. "
                            f"Supported: {valid_backends}"
                        )
                    attention_kwargs["attention_backend"] = backend
                    logger.info(f"SGLang backend: Using attention backend '{backend}'")

                # Prefill-specific backend
                prefill_backend = attention.get("prefill_backend")
                if prefill_backend is not None:
                    if prefill_backend not in valid_backends:
                        raise ValueError(
                            f"Invalid attention.prefill_backend: {prefill_backend}. "
                            f"Supported: {valid_backends}"
                        )
                    attention_kwargs["prefill_attention_backend"] = prefill_backend
                    logger.info(f"SGLang backend: Using prefill attention backend '{prefill_backend}'")

                # Decode-specific backend
                decode_backend = attention.get("decode_backend")
                if decode_backend is not None:
                    if decode_backend not in valid_backends:
                        raise ValueError(
                            f"Invalid attention.decode_backend: {decode_backend}. "
                            f"Supported: {valid_backends}"
                        )
                    attention_kwargs["decode_attention_backend"] = decode_backend
                    logger.info(f"SGLang backend: Using decode attention backend '{decode_backend}'")

                # Multimodal attention backend
                mm_backend = attention.get("mm_backend")
                if mm_backend is not None:
                    valid_mm_backends = ["sdpa", "fa3", "triton_attn", "ascend_attn", "aiter_attn"]
                    if mm_backend not in valid_mm_backends:
                        raise ValueError(
                            f"Invalid attention.mm_backend: {mm_backend}. "
                            f"Supported: {valid_mm_backends}"
                        )
                    attention_kwargs["mm_attention_backend"] = mm_backend
                    logger.info(f"SGLang backend: Using multimodal attention backend '{mm_backend}'")

                # Double sparsity attention
                if attention.get("enable_double_sparsity", False):
                    attention_kwargs["enable_double_sparsity"] = True
                    # Double sparsity requires triton backend
                    if backend and backend != "triton":
                        logger.warning(
                            f"SGLang backend: enable_double_sparsity works best with attention.backend='triton', "
                            f"current backend is '{backend}'"
                        )
                    logger.info("SGLang backend: Double sparsity attention enabled")

                # Native Sparse Attention (NSA) configuration
                nsa_config = attention.get("nsa", {})
                if nsa_config:
                    valid_nsa_backends = [
                        "flashmla_sparse", "flashmla_kv", "flashmla_auto",
                        "fa3", "tilelang", "aiter"
                    ]

                    nsa_prefill = nsa_config.get("prefill_backend")
                    if nsa_prefill is not None:
                        if nsa_prefill not in valid_nsa_backends:
                            raise ValueError(
                                f"Invalid attention.nsa.prefill_backend: {nsa_prefill}. "
                                f"Supported: {valid_nsa_backends}"
                            )
                        attention_kwargs["nsa_prefill_backend"] = nsa_prefill
                        logger.info(f"SGLang backend: Using NSA prefill backend '{nsa_prefill}'")

                    nsa_decode = nsa_config.get("decode_backend")
                    if nsa_decode is not None:
                        if nsa_decode not in valid_nsa_backends:
                            raise ValueError(
                                f"Invalid attention.nsa.decode_backend: {nsa_decode}. "
                                f"Supported: {valid_nsa_backends}"
                            )
                        attention_kwargs["nsa_decode_backend"] = nsa_decode
                        logger.info(f"SGLang backend: Using NSA decode backend '{nsa_decode}'")

            # Process LoRA hot-swapping configuration
            lora_kwargs = {}
            if lora_config is not None:
                # Pre-loaded LoRA adapter paths
                lora_paths = lora_config.get("paths")
                if lora_paths is not None:
                    # Convert various formats to list
                    if isinstance(lora_paths, dict):
                        # Dict format: {"name": "path"} -> ["name=path", ...]
                        lora_paths = [f"{name}={path}" for name, path in lora_paths.items()]
                    elif isinstance(lora_paths, (list, tuple)):
                        lora_paths = list(lora_paths)
                    else:
                        raise ValueError(
                            f"lora.paths must be a list or dict, got: {type(lora_paths)}"
                        )
                    lora_kwargs["lora_paths"] = lora_paths
                    logger.info(f"SGLang backend: Pre-loading {len(lora_paths)} LoRA adapter(s)")

                # Maximum LoRA rank
                max_rank = lora_config.get("max_rank")
                if max_rank is not None:
                    if not isinstance(max_rank, int) or max_rank < 1:
                        raise ValueError(
                            f"lora.max_rank must be a positive integer, got: {max_rank}"
                        )
                    lora_kwargs["max_lora_rank"] = max_rank
                    logger.info(f"SGLang backend: Max LoRA rank set to {max_rank}")

                # Target modules
                target_modules = lora_config.get("target_modules")
                if target_modules is not None:
                    valid_target_modules = [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "qkv_proj", "gate_up_proj",
                        "embed_tokens", "lm_head", "all"
                    ]
                    if isinstance(target_modules, str):
                        target_modules = [target_modules]
                    for module in target_modules:
                        if module not in valid_target_modules:
                            raise ValueError(
                                f"Invalid lora.target_modules: {module}. "
                                f"Supported: {valid_target_modules}"
                            )
                    lora_kwargs["lora_target_modules"] = list(target_modules)

                # Max adapters per batch
                max_loras_per_batch = lora_config.get("max_loras_per_batch", 8)
                if not isinstance(max_loras_per_batch, int) or max_loras_per_batch < 1:
                    raise ValueError(
                        f"lora.max_loras_per_batch must be a positive integer, got: {max_loras_per_batch}"
                    )
                lora_kwargs["max_loras_per_batch"] = max_loras_per_batch

                # Max loaded adapters in CPU memory
                max_loaded_loras = lora_config.get("max_loaded_loras")
                if max_loaded_loras is not None:
                    if not isinstance(max_loaded_loras, int) or max_loaded_loras < max_loras_per_batch:
                        raise ValueError(
                            f"lora.max_loaded_loras must be >= max_loras_per_batch ({max_loras_per_batch}), "
                            f"got: {max_loaded_loras}"
                        )
                    lora_kwargs["max_loaded_loras"] = max_loaded_loras

                # Eviction policy
                eviction_policy = lora_config.get("eviction_policy", "lru")
                valid_eviction_policies = ["lru", "fifo"]
                if eviction_policy not in valid_eviction_policies:
                    raise ValueError(
                        f"Invalid lora.eviction_policy: {eviction_policy}. "
                        f"Supported: {valid_eviction_policies}"
                    )
                lora_kwargs["lora_eviction_policy"] = eviction_policy

                # LoRA backend
                lora_backend = lora_config.get("backend", "csgmv")
                valid_lora_backends = ["csgmv", "triton", "ascend", "torch_native"]
                if lora_backend not in valid_lora_backends:
                    raise ValueError(
                        f"Invalid lora.backend: {lora_backend}. "
                        f"Supported: {valid_lora_backends}"
                    )
                lora_kwargs["lora_backend"] = lora_backend

                # Warn about csgmv limitations
                if lora_backend == "csgmv" and target_modules:
                    embed_modules = {"embed_tokens", "lm_head"}
                    if embed_modules.intersection(set(target_modules)):
                        logger.warning(
                            "SGLang backend: lora.backend='csgmv' doesn't support embed_tokens/lm_head. "
                            "These will be ignored. Use lora.backend='triton' for full support."
                        )

                # Max chunk size for CSGMV
                max_chunk_size = lora_config.get("max_chunk_size", 16)
                if not isinstance(max_chunk_size, int):
                    raise ValueError(f"lora.max_chunk_size must be an integer, got: {max_chunk_size}")
                if max_chunk_size < 16 or max_chunk_size > 128:
                    raise ValueError(
                        f"lora.max_chunk_size must be between 16 and 128, got: {max_chunk_size}"
                    )
                if max_chunk_size & (max_chunk_size - 1) != 0:
                    raise ValueError(
                        f"lora.max_chunk_size must be a power of 2, got: {max_chunk_size}"
                    )
                lora_kwargs["max_lora_chunk_size"] = max_chunk_size

                # Enable LoRA if we have paths or explicit config
                if lora_paths or max_rank or target_modules:
                    lora_kwargs["enable_lora"] = True
                    logger.info(
                        f"SGLang backend: LoRA hot-swapping enabled with backend='{lora_backend}', "
                        f"max_loras_per_batch={max_loras_per_batch}"
                    )

            # Process RoPE scaling configuration
            rope_kwargs = {}
            if rope_scaling:
                # Validate rope_type
                rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
                valid_rope_types = [
                    "linear", "dynamic", "yarn", "deepseek_yarn",
                    "longrope", "llama3", "default"
                ]
                if rope_type and rope_type not in valid_rope_types:
                    raise ValueError(
                        f"Invalid rope_type: {rope_type}. "
                        f"Supported: {valid_rope_types}"
                    )

                # Validate required parameters based on rope_type
                if rope_type in ["linear", "dynamic", "yarn", "deepseek_yarn", "llama3"]:
                    if "factor" not in rope_scaling:
                        raise ValueError(
                            f"rope_scaling.factor is required for rope_type='{rope_type}'"
                        )

                if rope_type in ["yarn", "deepseek_yarn", "llama3", "longrope"]:
                    if "original_max_position_embeddings" not in rope_scaling:
                        raise ValueError(
                            f"rope_scaling.original_max_position_embeddings is required for rope_type='{rope_type}'"
                        )

                rope_kwargs["rope_scaling"] = dict(rope_scaling)
                factor = rope_scaling.get("factor", 1.0)
                logger.info(
                    f"SGLang backend: RoPE scaling enabled with type='{rope_type}', factor={factor}"
                )

            if rope_theta is not None:
                rope_kwargs["rope_theta"] = rope_theta
                logger.info(f"SGLang backend: Using custom rope_theta={rope_theta}")

            # Process priority scheduling configuration
            scheduling_kwargs = {}
            if scheduling:
                # Schedule policy
                policy = scheduling.get("policy", "fcfs")
                valid_policies = ["fcfs", "lpm", "dfs-weight", "lof", "random"]
                if policy not in valid_policies:
                    raise ValueError(
                        f"Invalid scheduling.policy: '{policy}'. "
                        f"Supported policies: {valid_policies}"
                    )
                scheduling_kwargs["schedule_policy"] = policy

                # Priority scheduling
                enable_priority = scheduling.get("enable_priority", False)
                if enable_priority:
                    # Priority scheduling only works with FCFS or LOF policies
                    if policy not in ["fcfs", "lof"]:
                        raise ValueError(
                            f"Priority scheduling requires policy='fcfs' or 'lof', "
                            f"but got policy='{policy}'"
                        )
                    scheduling_kwargs["enable_priority_scheduling"] = True
                    logger.info("SGLang backend: Priority scheduling enabled")

                # Abort on priority when disabled
                abort_on_priority = scheduling.get("abort_on_priority_when_disabled", False)
                if abort_on_priority:
                    scheduling_kwargs["abort_on_priority_when_disabled"] = True

                # Priority value interpretation
                low_priority_first = scheduling.get("low_priority_values_first", False)
                if low_priority_first:
                    scheduling_kwargs["schedule_low_priority_values_first"] = True

                # Preemption threshold
                preemption_threshold = scheduling.get("preemption_threshold", 10)
                if not isinstance(preemption_threshold, int) or preemption_threshold < 0:
                    raise ValueError(
                        f"scheduling.preemption_threshold must be a non-negative integer, "
                        f"got: {preemption_threshold}"
                    )
                if preemption_threshold != 10:  # Only set if non-default
                    scheduling_kwargs["priority_scheduling_preemption_threshold"] = preemption_threshold
                    if enable_priority:
                        logger.info(
                            f"SGLang backend: Priority preemption threshold set to {preemption_threshold}"
                        )

                # Schedule conservativeness
                conservativeness = scheduling.get("conservativeness", 1.0)
                if not isinstance(conservativeness, (int, float)) or conservativeness < 0:
                    raise ValueError(
                        f"scheduling.conservativeness must be a non-negative number, "
                        f"got: {conservativeness}"
                    )
                if conservativeness != 1.0:  # Only set if non-default
                    scheduling_kwargs["schedule_conservativeness"] = float(conservativeness)
                    logger.info(
                        f"SGLang backend: Schedule conservativeness set to {conservativeness}"
                    )

                # Chunked prefill configuration
                chunked_prefill_size = scheduling.get("chunked_prefill_size")
                if chunked_prefill_size is not None:
                    if not isinstance(chunked_prefill_size, int) or chunked_prefill_size <= 0:
                        raise ValueError(
                            f"scheduling.chunked_prefill_size must be a positive integer, "
                            f"got: {chunked_prefill_size}"
                        )
                    scheduling_kwargs["chunked_prefill_size"] = chunked_prefill_size
                    logger.info(
                        f"SGLang backend: Chunked prefill size set to {chunked_prefill_size}"
                    )

                # Dynamic chunking
                enable_dynamic_chunking = scheduling.get("enable_dynamic_chunking", False)
                if enable_dynamic_chunking:
                    scheduling_kwargs["enable_dynamic_chunking"] = True
                    logger.info("SGLang backend: Dynamic chunking enabled")

                # Capacity limits
                max_running = scheduling.get("max_running_requests")
                if max_running is not None:
                    if not isinstance(max_running, int) or max_running <= 0:
                        raise ValueError(
                            f"scheduling.max_running_requests must be a positive integer, "
                            f"got: {max_running}"
                        )
                    scheduling_kwargs["max_running_requests"] = max_running
                    logger.info(
                        f"SGLang backend: Max running requests set to {max_running}"
                    )

                max_queued = scheduling.get("max_queued_requests")
                if max_queued is not None:
                    if not isinstance(max_queued, int) or max_queued <= 0:
                        raise ValueError(
                            f"scheduling.max_queued_requests must be a positive integer, "
                            f"got: {max_queued}"
                        )
                    scheduling_kwargs["max_queued_requests"] = max_queued
                    logger.info(
                        f"SGLang backend: Max queued requests set to {max_queued}"
                    )

                max_prefill = scheduling.get("max_prefill_tokens")
                if max_prefill is not None:
                    if not isinstance(max_prefill, int) or max_prefill <= 0:
                        raise ValueError(
                            f"scheduling.max_prefill_tokens must be a positive integer, "
                            f"got: {max_prefill}"
                        )
                    scheduling_kwargs["max_prefill_tokens"] = max_prefill
                    logger.info(
                        f"SGLang backend: Max prefill tokens set to {max_prefill}"
                    )

                max_total = scheduling.get("max_total_tokens")
                if max_total is not None:
                    if not isinstance(max_total, int) or max_total <= 0:
                        raise ValueError(
                            f"scheduling.max_total_tokens must be a positive integer, "
                            f"got: {max_total}"
                        )
                    scheduling_kwargs["max_total_tokens"] = max_total
                    logger.info(
                        f"SGLang backend: Max total tokens set to {max_total}"
                    )

                if policy != "fcfs":
                    logger.info(f"SGLang backend: Using '{policy}' scheduling policy")

            # Process disaggregated prefill/decode configuration
            disaggregation_kwargs = {}
            if disaggregation:
                # Disaggregation mode
                mode = disaggregation.get("mode", "null")
                valid_modes = ["null", "prefill", "decode"]
                if mode not in valid_modes:
                    raise ValueError(
                        f"Invalid disaggregation.mode: '{mode}'. "
                        f"Supported modes: {valid_modes}"
                    )

                if mode != "null":
                    disaggregation_kwargs["disaggregation_mode"] = mode
                    logger.info(f"SGLang backend: Disaggregation mode set to '{mode}'")

                    # Transfer backend
                    transfer_backend = disaggregation.get("transfer_backend", "mooncake")
                    valid_backends = ["mooncake", "nixl", "ascend", "fake"]
                    if transfer_backend not in valid_backends:
                        raise ValueError(
                            f"Invalid disaggregation.transfer_backend: '{transfer_backend}'. "
                            f"Supported backends: {valid_backends}"
                        )
                    disaggregation_kwargs["disaggregation_transfer_backend"] = transfer_backend
                    logger.info(
                        f"SGLang backend: Disaggregation transfer backend set to '{transfer_backend}'"
                    )

                    # Bootstrap port
                    bootstrap_port = disaggregation.get("bootstrap_port", 8998)
                    if not isinstance(bootstrap_port, int) or bootstrap_port <= 0 or bootstrap_port > 65535:
                        raise ValueError(
                            f"disaggregation.bootstrap_port must be a valid port (1-65535), "
                            f"got: {bootstrap_port}"
                        )
                    if bootstrap_port != 8998:  # Only set if non-default
                        disaggregation_kwargs["disaggregation_bootstrap_port"] = bootstrap_port

                    # InfiniBand device
                    ib_device = disaggregation.get("ib_device")
                    if ib_device is not None:
                        disaggregation_kwargs["disaggregation_ib_device"] = ib_device
                        logger.info(
                            f"SGLang backend: Using InfiniBand device '{ib_device}' for disaggregation"
                        )

                    # Reserved decode tokens
                    reserved_tokens = disaggregation.get("num_reserved_decode_tokens", 512)
                    if not isinstance(reserved_tokens, int) or reserved_tokens < 0:
                        raise ValueError(
                            f"disaggregation.num_reserved_decode_tokens must be non-negative, "
                            f"got: {reserved_tokens}"
                        )
                    if reserved_tokens != 512:  # Only set if non-default
                        disaggregation_kwargs["num_reserved_decode_tokens"] = reserved_tokens

                    # Decode worker configuration (for prefill mode)
                    decode_config = disaggregation.get("decode", {})
                    if decode_config:
                        # Decode TP size
                        decode_tp = decode_config.get("tp_size")
                        if decode_tp is not None:
                            if not isinstance(decode_tp, int) or decode_tp <= 0:
                                raise ValueError(
                                    f"disaggregation.decode.tp_size must be a positive integer, "
                                    f"got: {decode_tp}"
                                )
                            disaggregation_kwargs["disaggregation_decode_tp"] = decode_tp
                            logger.info(
                                f"SGLang backend: Decode worker TP size set to {decode_tp}"
                            )

                        # Decode DP size
                        decode_dp = decode_config.get("dp_size")
                        if decode_dp is not None:
                            if not isinstance(decode_dp, int) or decode_dp <= 0:
                                raise ValueError(
                                    f"disaggregation.decode.dp_size must be a positive integer, "
                                    f"got: {decode_dp}"
                                )
                            disaggregation_kwargs["disaggregation_decode_dp"] = decode_dp
                            logger.info(
                                f"SGLang backend: Decode worker DP size set to {decode_dp}"
                            )

                        # KV cache offloading
                        if decode_config.get("enable_offload_kvcache", False):
                            disaggregation_kwargs["disaggregation_decode_enable_offload_kvcache"] = True
                            logger.info(
                                "SGLang backend: KV cache offloading enabled for decode workers"
                            )

                        # Polling interval
                        polling_interval = decode_config.get("polling_interval", 1)
                        if not isinstance(polling_interval, int) or polling_interval <= 0:
                            raise ValueError(
                                f"disaggregation.decode.polling_interval must be a positive integer, "
                                f"got: {polling_interval}"
                            )
                        if polling_interval != 1:  # Only set if non-default
                            disaggregation_kwargs["disaggregation_decode_polling_interval"] = polling_interval

                        # Fake auto mode
                        if decode_config.get("enable_fake_auto", False):
                            disaggregation_kwargs["disaggregation_decode_enable_fake_auto"] = True

                    # Prefill worker configuration (for decode mode)
                    prefill_config = disaggregation.get("prefill", {})
                    if prefill_config:
                        # Prefill PP size
                        prefill_pp = prefill_config.get("pp_size", 1)
                        if not isinstance(prefill_pp, int) or prefill_pp <= 0:
                            raise ValueError(
                                f"disaggregation.prefill.pp_size must be a positive integer, "
                                f"got: {prefill_pp}"
                            )
                        if prefill_pp != 1:  # Only set if non-default
                            disaggregation_kwargs["disaggregation_prefill_pp"] = prefill_pp
                            logger.info(
                                f"SGLang backend: Prefill worker PP size set to {prefill_pp}"
                            )

                # DP Attention (can be used without disaggregation mode)
                if disaggregation.get("enable_dp_attention", False):
                    disaggregation_kwargs["enable_dp_attention"] = True
                    logger.info("SGLang backend: Data Parallel Attention (DP-Attention) enabled")

                # DP LM Head
                if disaggregation.get("enable_dp_lm_head", False):
                    disaggregation_kwargs["enable_dp_lm_head"] = True
                    logger.info("SGLang backend: Data Parallel LM Head enabled")

            # Process multi-node inference configuration
            multi_node_kwargs = {}
            multi_node_env_vars = {}
            if multi_node:
                # Number of nodes
                nnodes = multi_node.get("nnodes")
                if nnodes is not None:
                    if not isinstance(nnodes, int) or nnodes <= 0:
                        raise ValueError(
                            f"multi_node.nnodes must be a positive integer, got: {nnodes}"
                        )
                    multi_node_kwargs["nnodes"] = nnodes
                    if nnodes > 1:
                        logger.info(f"SGLang backend: Multi-node inference with {nnodes} nodes")

                # Node rank
                node_rank = multi_node.get("node_rank")
                if node_rank is not None:
                    if not isinstance(node_rank, int) or node_rank < 0:
                        raise ValueError(
                            f"multi_node.node_rank must be a non-negative integer, got: {node_rank}"
                        )
                    if nnodes is not None and node_rank >= nnodes:
                        raise ValueError(
                            f"multi_node.node_rank ({node_rank}) must be less than nnodes ({nnodes})"
                        )
                    multi_node_kwargs["node_rank"] = node_rank
                    logger.info(f"SGLang backend: Node rank set to {node_rank}")

                # Distributed initialization address
                dist_init_addr = multi_node.get("dist_init_addr")
                if dist_init_addr is not None:
                    if not isinstance(dist_init_addr, str):
                        raise ValueError(
                            f"multi_node.dist_init_addr must be a string, got: {type(dist_init_addr)}"
                        )
                    # Validate format (hostname:port or ip:port)
                    if ":" not in dist_init_addr:
                        raise ValueError(
                            f"multi_node.dist_init_addr must be in format 'hostname:port', "
                            f"got: {dist_init_addr}"
                        )
                    multi_node_kwargs["dist_init_addr"] = dist_init_addr
                    logger.info(
                        f"SGLang backend: Distributed init address set to {dist_init_addr}"
                    )

                # NCCL configuration
                nccl_config = multi_node.get("nccl", {})
                if nccl_config:
                    # Enable symmetric memory
                    if nccl_config.get("enable_symm_mem", False):
                        multi_node_kwargs["enable_symm_mem"] = True
                        multi_node_env_vars["NCCL_CUMEM_ENABLE"] = "1"
                        # NVLS is auto-enabled with symm_mem
                        multi_node_env_vars["NCCL_NVLS_ENABLE"] = "1"
                        logger.info(
                            "SGLang backend: NCCL symmetric memory enabled (NVLS auto-enabled)"
                        )

                    # Enable NVLS (NVLink Switch)
                    elif nccl_config.get("enable_nvls", False):
                        multi_node_kwargs["enable_nccl_nvls"] = True
                        multi_node_env_vars["NCCL_NVLS_ENABLE"] = "1"
                        logger.info("SGLang backend: NCCL NVLS enabled")

                    # NCCL timeout
                    nccl_timeout = nccl_config.get("timeout")
                    if nccl_timeout is not None:
                        if not isinstance(nccl_timeout, (int, float)) or nccl_timeout <= 0:
                            raise ValueError(
                                f"multi_node.nccl.timeout must be a positive number, "
                                f"got: {nccl_timeout}"
                            )
                        multi_node_env_vars["NCCL_TIMEOUT"] = str(int(nccl_timeout))
                        multi_node_env_vars["SKYRL_WORKER_NCCL_TIMEOUT_IN_S"] = str(int(nccl_timeout))
                        logger.info(
                            f"SGLang backend: NCCL timeout set to {nccl_timeout} seconds"
                        )

                    # NCCL debug level
                    debug_level = nccl_config.get("debug_level")
                    if debug_level is not None:
                        valid_levels = ["WARN", "INFO", "DEBUG", "TRACE"]
                        debug_level = debug_level.upper()
                        if debug_level not in valid_levels:
                            raise ValueError(
                                f"multi_node.nccl.debug_level must be one of {valid_levels}, "
                                f"got: {debug_level}"
                            )
                        multi_node_env_vars["NCCL_DEBUG"] = debug_level
                        logger.info(f"SGLang backend: NCCL debug level set to {debug_level}")

                # InfiniBand optimization
                if multi_node.get("enable_ib_optimization", False):
                    multi_node_env_vars["NCCL_IB_DISABLE"] = "0"
                    multi_node_env_vars["NCCL_NET_GDR_LEVEL"] = "5"
                    logger.info("SGLang backend: InfiniBand optimization enabled")

                # CUDA device max connections
                cuda_max_connections = multi_node.get("cuda_device_max_connections")
                if cuda_max_connections is not None:
                    if not isinstance(cuda_max_connections, int) or cuda_max_connections <= 0:
                        raise ValueError(
                            f"multi_node.cuda_device_max_connections must be a positive integer, "
                            f"got: {cuda_max_connections}"
                        )
                    multi_node_env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = str(cuda_max_connections)
                    logger.info(
                        f"SGLang backend: CUDA device max connections set to {cuda_max_connections}"
                    )

            # Process Prometheus metrics and observability configuration
            metrics_kwargs = {}
            if metrics:
                # Enable metrics
                if metrics.get("enabled", False):
                    metrics_kwargs["enable_metrics"] = True
                    logger.info("SGLang backend: Prometheus metrics enabled")

                    # Enable metrics for all schedulers
                    if metrics.get("enable_for_all_schedulers", False):
                        metrics_kwargs["enable_metrics_for_all_schedulers"] = True
                        logger.info(
                            "SGLang backend: Metrics enabled for all TP ranks"
                        )

                # Latency histogram buckets
                buckets_config = metrics.get("buckets", {})
                if buckets_config:
                    # Time-to-first-token buckets
                    ttft_buckets = buckets_config.get("time_to_first_token")
                    if ttft_buckets is not None:
                        if not isinstance(ttft_buckets, list):
                            raise ValueError(
                                f"metrics.buckets.time_to_first_token must be a list, "
                                f"got: {type(ttft_buckets)}"
                            )
                        metrics_kwargs["bucket_time_to_first_token"] = ttft_buckets

                    # Inter-token latency buckets
                    itl_buckets = buckets_config.get("inter_token_latency")
                    if itl_buckets is not None:
                        if not isinstance(itl_buckets, list):
                            raise ValueError(
                                f"metrics.buckets.inter_token_latency must be a list, "
                                f"got: {type(itl_buckets)}"
                            )
                        metrics_kwargs["bucket_inter_token_latency"] = itl_buckets

                    # E2E request latency buckets
                    e2e_buckets = buckets_config.get("e2e_request_latency")
                    if e2e_buckets is not None:
                        if not isinstance(e2e_buckets, list):
                            raise ValueError(
                                f"metrics.buckets.e2e_request_latency must be a list, "
                                f"got: {type(e2e_buckets)}"
                            )
                        metrics_kwargs["bucket_e2e_request_latency"] = e2e_buckets

                # Token histograms
                if metrics.get("collect_tokens_histogram", False):
                    metrics_kwargs["collect_tokens_histogram"] = True
                    logger.info("SGLang backend: Token histogram collection enabled")

                prompt_buckets = metrics.get("prompt_tokens_buckets")
                if prompt_buckets is not None:
                    metrics_kwargs["prompt_tokens_buckets"] = prompt_buckets

                gen_buckets = metrics.get("generation_tokens_buckets")
                if gen_buckets is not None:
                    metrics_kwargs["generation_tokens_buckets"] = gen_buckets

                # Export to file
                export_config = metrics.get("export_to_file", {})
                if export_config.get("enabled", False):
                    metrics_kwargs["export_metrics_to_file"] = True
                    export_dir = export_config.get("directory")
                    if export_dir is None:
                        raise ValueError(
                            "metrics.export_to_file.directory is required when "
                            "metrics.export_to_file.enabled is true"
                        )
                    metrics_kwargs["export_metrics_to_file_dir"] = export_dir
                    logger.info(
                        f"SGLang backend: Per-request metrics export enabled to {export_dir}"
                    )

                # Custom labels
                custom_labels = metrics.get("custom_labels", {})
                if custom_labels:
                    header = custom_labels.get("header", "x-custom-labels")
                    if header != "x-custom-labels":
                        metrics_kwargs["tokenizer_metrics_custom_labels_header"] = header

                    allowed = custom_labels.get("allowed")
                    if allowed is not None:
                        if not isinstance(allowed, list):
                            raise ValueError(
                                f"metrics.custom_labels.allowed must be a list, "
                                f"got: {type(allowed)}"
                            )
                        metrics_kwargs["tokenizer_metrics_allowed_custom_labels"] = allowed

                # OpenTelemetry tracing
                tracing_config = metrics.get("tracing", {})
                if tracing_config.get("enabled", False):
                    metrics_kwargs["enable_trace"] = True
                    otlp_endpoint = tracing_config.get("otlp_endpoint", "localhost:4317")
                    if ":" not in otlp_endpoint:
                        raise ValueError(
                            f"metrics.tracing.otlp_endpoint must be in format 'hostname:port', "
                            f"got: {otlp_endpoint}"
                        )
                    metrics_kwargs["otlp_traces_endpoint"] = otlp_endpoint
                    logger.info(
                        f"SGLang backend: OpenTelemetry tracing enabled with endpoint {otlp_endpoint}"
                    )

                # Request logging
                logging_config = metrics.get("logging", {})
                if logging_config.get("enabled", False):
                    metrics_kwargs["log_requests"] = True

                    log_level = logging_config.get("level", 2)
                    if not isinstance(log_level, int) or log_level < 0 or log_level > 3:
                        raise ValueError(
                            f"metrics.logging.level must be an integer 0-3, got: {log_level}"
                        )
                    metrics_kwargs["log_requests_level"] = log_level

                    log_format = logging_config.get("format", "text")
                    if log_format not in ["text", "json"]:
                        raise ValueError(
                            f"metrics.logging.format must be 'text' or 'json', got: {log_format}"
                        )
                    metrics_kwargs["log_requests_format"] = log_format

                    log_targets = logging_config.get("targets")
                    if log_targets is not None:
                        if not isinstance(log_targets, list):
                            raise ValueError(
                                f"metrics.logging.targets must be a list, got: {type(log_targets)}"
                            )
                        metrics_kwargs["log_requests_target"] = log_targets

                    logger.info(
                        f"SGLang backend: Request logging enabled (level={log_level}, format={log_format})"
                    )

            # Process deterministic inference configuration
            deterministic_kwargs = {}
            if deterministic_inference:
                # Enable deterministic inference mode
                if deterministic_inference.get("enabled", False):
                    deterministic_kwargs["enable_deterministic_inference"] = True
                    logger.info(
                        "SGLang backend: Deterministic inference enabled "
                        "(uses batch-invariant kernels, ~34% overhead mitigated by CUDA graphs)"
                    )

                # On-policy target backend alignment
                rl_target = deterministic_inference.get("rl_on_policy_target")
                if rl_target is not None:
                    valid_targets = ["fsdp"]
                    if rl_target not in valid_targets:
                        raise ValueError(
                            f"deterministic_inference.rl_on_policy_target must be one of "
                            f"{valid_targets}, got: {rl_target}"
                        )
                    deterministic_kwargs["rl_on_policy_target"] = rl_target
                    logger.info(
                        f"SGLang backend: RL on-policy target set to '{rl_target}' "
                        f"(uses native ops for zero KL divergence with training backend)"
                    )

            # Process load balancing and request routing configuration
            load_balancing_kwargs = {}
            if load_balancing:
                # Load balance method
                lb_method = load_balancing.get("method")
                if lb_method is not None:
                    valid_methods = [
                        "auto", "round_robin", "shortest_queue",
                        "minimum_tokens", "follow_bootstrap_room"
                    ]
                    if lb_method not in valid_methods:
                        raise ValueError(
                            f"load_balancing.method must be one of {valid_methods}, "
                            f"got: {lb_method}"
                        )
                    load_balancing_kwargs["load_balance_method"] = lb_method
                    logger.info(f"SGLang backend: Load balance method set to '{lb_method}'")

                # Expert Parallelism (EP) configuration
                ep_config = load_balancing.get("expert_parallelism", {})
                if ep_config:
                    # EP size
                    ep_size_val = ep_config.get("ep_size")
                    if ep_size_val is not None:
                        if not isinstance(ep_size_val, int) or ep_size_val < 1:
                            raise ValueError(
                                f"load_balancing.expert_parallelism.ep_size must be a "
                                f"positive integer, got: {ep_size_val}"
                            )
                        load_balancing_kwargs["ep_size"] = ep_size_val
                        logger.info(f"SGLang backend: Expert parallelism size set to {ep_size_val}")

                    # EP dispatch algorithm
                    dispatch_algo = ep_config.get("dispatch_algorithm")
                    if dispatch_algo is not None:
                        load_balancing_kwargs["ep_dispatch_algorithm"] = dispatch_algo
                        logger.info(
                            f"SGLang backend: EP dispatch algorithm set to '{dispatch_algo}'"
                        )

                    # Number of redundant experts
                    num_redundant = ep_config.get("num_redundant_experts")
                    if num_redundant is not None:
                        if not isinstance(num_redundant, int) or num_redundant < 0:
                            raise ValueError(
                                f"load_balancing.expert_parallelism.num_redundant_experts "
                                f"must be a non-negative integer, got: {num_redundant}"
                            )
                        load_balancing_kwargs["ep_num_redundant_experts"] = num_redundant
                        if num_redundant > 0:
                            logger.info(
                                f"SGLang backend: {num_redundant} redundant experts configured"
                            )

                    # Initial expert location
                    init_location = ep_config.get("init_expert_location")
                    if init_location is not None:
                        load_balancing_kwargs["init_expert_location"] = init_location
                        logger.info(
                            f"SGLang backend: Initial expert location strategy set to '{init_location}'"
                        )

                # Expert-Parallel Load Balancing (EPLB) configuration
                eplb_config = load_balancing.get("eplb", {})
                if eplb_config:
                    # Enable EPLB
                    if eplb_config.get("enabled", False):
                        load_balancing_kwargs["enable_eplb"] = True
                        logger.info("SGLang backend: Expert-Parallel Load Balancing enabled")

                        # EPLB algorithm
                        eplb_algo = eplb_config.get("algorithm")
                        if eplb_algo is not None:
                            load_balancing_kwargs["eplb_algorithm"] = eplb_algo
                            logger.info(
                                f"SGLang backend: EPLB algorithm set to '{eplb_algo}'"
                            )

                        # Rebalance iterations
                        rebalance_iters = eplb_config.get("rebalance_num_iterations")
                        if rebalance_iters is not None:
                            if not isinstance(rebalance_iters, int) or rebalance_iters < 1:
                                raise ValueError(
                                    f"load_balancing.eplb.rebalance_num_iterations must be "
                                    f"a positive integer, got: {rebalance_iters}"
                                )
                            load_balancing_kwargs["eplb_rebalance_num_iterations"] = rebalance_iters
                            logger.info(
                                f"SGLang backend: EPLB rebalance every {rebalance_iters} iterations"
                            )

                        # Layers per chunk
                        layers_per_chunk = eplb_config.get("rebalance_layers_per_chunk")
                        if layers_per_chunk is not None:
                            if not isinstance(layers_per_chunk, int) or layers_per_chunk < 1:
                                raise ValueError(
                                    f"load_balancing.eplb.rebalance_layers_per_chunk must be "
                                    f"a positive integer, got: {layers_per_chunk}"
                                )
                            load_balancing_kwargs["eplb_rebalance_layers_per_chunk"] = layers_per_chunk

                        # Minimum utilization threshold
                        min_util = eplb_config.get("min_rebalancing_utilization_threshold")
                        if min_util is not None:
                            if not isinstance(min_util, (int, float)) or min_util < 0.0 or min_util > 1.0:
                                raise ValueError(
                                    f"load_balancing.eplb.min_rebalancing_utilization_threshold "
                                    f"must be a float in [0.0, 1.0], got: {min_util}"
                                )
                            load_balancing_kwargs["eplb_min_rebalancing_utilization_threshold"] = float(min_util)

                # Expert distribution metrics
                expert_metrics_config = load_balancing.get("expert_metrics", {})
                if expert_metrics_config:
                    # Recorder mode
                    recorder_mode = expert_metrics_config.get("recorder_mode")
                    if recorder_mode is not None:
                        load_balancing_kwargs["expert_distribution_recorder_mode"] = recorder_mode
                        logger.info(
                            f"SGLang backend: Expert distribution recorder mode set to '{recorder_mode}'"
                        )

                    # Recorder buffer size
                    buffer_size = expert_metrics_config.get("recorder_buffer_size")
                    if buffer_size is not None:
                        if not isinstance(buffer_size, int) or buffer_size < 1:
                            raise ValueError(
                                f"load_balancing.expert_metrics.recorder_buffer_size must be "
                                f"a positive integer, got: {buffer_size}"
                            )
                        load_balancing_kwargs["expert_distribution_recorder_buffer_size"] = buffer_size

                    # Enable expert distribution metrics
                    if expert_metrics_config.get("enabled", False):
                        load_balancing_kwargs["enable_expert_distribution_metrics"] = True
                        logger.info("SGLang backend: Expert distribution metrics enabled")

                # Request batching configuration
                batching_config = load_balancing.get("batching", {})
                if batching_config:
                    # Max prefill tokens
                    max_prefill = batching_config.get("max_prefill_tokens")
                    if max_prefill is not None:
                        if not isinstance(max_prefill, int) or max_prefill < 1:
                            raise ValueError(
                                f"load_balancing.batching.max_prefill_tokens must be "
                                f"a positive integer, got: {max_prefill}"
                            )
                        load_balancing_kwargs["max_prefill_tokens"] = max_prefill
                        logger.info(
                            f"SGLang backend: Max prefill tokens set to {max_prefill}"
                        )

                    # Max total tokens
                    max_total = batching_config.get("max_total_tokens")
                    if max_total is not None:
                        if not isinstance(max_total, int) or max_total < 1:
                            raise ValueError(
                                f"load_balancing.batching.max_total_tokens must be "
                                f"a positive integer, got: {max_total}"
                            )
                        load_balancing_kwargs["max_total_tokens"] = max_total
                        logger.info(
                            f"SGLang backend: Max total tokens set to {max_total}"
                        )

                    # Tokenizer worker count
                    tokenizer_workers = batching_config.get("tokenizer_worker_num")
                    if tokenizer_workers is not None:
                        if not isinstance(tokenizer_workers, int) or tokenizer_workers < 1:
                            raise ValueError(
                                f"load_balancing.batching.tokenizer_worker_num must be "
                                f"a positive integer, got: {tokenizer_workers}"
                            )
                        load_balancing_kwargs["tokenizer_worker_num"] = tokenizer_workers
                        logger.info(
                            f"SGLang backend: {tokenizer_workers} tokenizer worker(s) configured"
                        )

            # Process health checks and Kubernetes probe configuration
            health_checks_kwargs = {}
            health_checks_env_vars = {}
            if health_checks:
                # Watchdog configuration
                watchdog_config = health_checks.get("watchdog", {})
                if watchdog_config:
                    # Hard watchdog timeout
                    watchdog_timeout = watchdog_config.get("timeout")
                    if watchdog_timeout is not None:
                        if not isinstance(watchdog_timeout, (int, float)) or watchdog_timeout <= 0:
                            raise ValueError(
                                f"health_checks.watchdog.timeout must be a positive number, "
                                f"got: {watchdog_timeout}"
                            )
                        health_checks_kwargs["watchdog_timeout"] = float(watchdog_timeout)
                        logger.info(
                            f"SGLang backend: Watchdog timeout set to {watchdog_timeout}s"
                        )

                    # Soft watchdog timeout
                    soft_timeout = watchdog_config.get("soft_timeout")
                    if soft_timeout is not None:
                        if not isinstance(soft_timeout, (int, float)) or soft_timeout <= 0:
                            raise ValueError(
                                f"health_checks.watchdog.soft_timeout must be a positive number, "
                                f"got: {soft_timeout}"
                            )
                        health_checks_kwargs["soft_watchdog_timeout"] = float(soft_timeout)
                        logger.info(
                            f"SGLang backend: Soft watchdog timeout set to {soft_timeout}s"
                        )

                # Distributed initialization timeout
                dist_timeout = health_checks.get("dist_timeout")
                if dist_timeout is not None:
                    if not isinstance(dist_timeout, int) or dist_timeout <= 0:
                        raise ValueError(
                            f"health_checks.dist_timeout must be a positive integer, "
                            f"got: {dist_timeout}"
                        )
                    health_checks_kwargs["dist_timeout"] = dist_timeout
                    logger.info(
                        f"SGLang backend: Distributed init timeout set to {dist_timeout}s"
                    )

                # Health endpoint configuration (environment variables)
                endpoint_config = health_checks.get("endpoint", {})
                if endpoint_config:
                    # Health check timeout
                    endpoint_timeout = endpoint_config.get("timeout")
                    if endpoint_timeout is not None:
                        if not isinstance(endpoint_timeout, int) or endpoint_timeout <= 0:
                            raise ValueError(
                                f"health_checks.endpoint.timeout must be a positive integer, "
                                f"got: {endpoint_timeout}"
                            )
                        health_checks_env_vars["SGLANG_HEALTH_CHECK_TIMEOUT"] = str(endpoint_timeout)
                        logger.info(
                            f"SGLang backend: Health check timeout set to {endpoint_timeout}s"
                        )

                    # Enable generation in health checks
                    enable_generation = endpoint_config.get("enable_generation")
                    if enable_generation is not None:
                        if not isinstance(enable_generation, bool):
                            raise ValueError(
                                f"health_checks.endpoint.enable_generation must be a boolean, "
                                f"got: {type(enable_generation)}"
                            )
                        health_checks_env_vars["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = (
                            "1" if enable_generation else "0"
                        )
                        logger.info(
                            f"SGLang backend: Health endpoint generation "
                            f"{'enabled' if enable_generation else 'disabled'}"
                        )

                # Startup configuration (environment variables)
                startup_config = health_checks.get("startup", {})
                if startup_config:
                    # Weights ready timeout
                    weights_timeout = startup_config.get("weights_ready_timeout")
                    if weights_timeout is not None:
                        if not isinstance(weights_timeout, int) or weights_timeout <= 0:
                            raise ValueError(
                                f"health_checks.startup.weights_ready_timeout must be a "
                                f"positive integer, got: {weights_timeout}"
                            )
                        health_checks_env_vars["SGLANG_WAIT_WEIGHTS_READY_TIMEOUT"] = str(weights_timeout)
                        logger.info(
                            f"SGLang backend: Weights ready timeout set to {weights_timeout}s"
                        )

                    # Warmup timeout
                    warmup_timeout = startup_config.get("warmup_timeout")
                    if warmup_timeout is not None:
                        if not isinstance(warmup_timeout, (int, float)):
                            raise ValueError(
                                f"health_checks.startup.warmup_timeout must be a number, "
                                f"got: {type(warmup_timeout)}"
                            )
                        health_checks_env_vars["SGLANG_WARMUP_TIMEOUT"] = str(warmup_timeout)
                        if warmup_timeout > 0:
                            logger.info(
                                f"SGLang backend: Warmup timeout set to {warmup_timeout}s"
                            )
                        else:
                            logger.info("SGLang backend: Warmup timeout disabled")

            # Process hierarchical cache configuration
            hierarchical_cache_kwargs = {}
            if hierarchical_cache:
                # Enable hierarchical cache
                if hierarchical_cache.get("enabled", False):
                    hierarchical_cache_kwargs["enable_hierarchical_cache"] = True
                    logger.info("SGLang backend: Hierarchical cache enabled (GPU→CPU→Storage)")

                    # Host memory configuration
                    host_mem_config = hierarchical_cache.get("host_memory", {})
                    if host_mem_config:
                        # Host/device ratio
                        ratio = host_mem_config.get("ratio")
                        if ratio is not None:
                            if not isinstance(ratio, (int, float)) or ratio <= 0:
                                raise ValueError(
                                    f"hierarchical_cache.host_memory.ratio must be a positive number, "
                                    f"got: {ratio}"
                                )
                            hierarchical_cache_kwargs["hicache_ratio"] = float(ratio)
                            logger.info(
                                f"SGLang backend: Host cache ratio set to {ratio}x GPU cache"
                            )

                        # Explicit size in GB
                        size_gb = host_mem_config.get("size_gb")
                        if size_gb is not None:
                            if not isinstance(size_gb, int) or size_gb <= 0:
                                raise ValueError(
                                    f"hierarchical_cache.host_memory.size_gb must be a positive integer, "
                                    f"got: {size_gb}"
                                )
                            hierarchical_cache_kwargs["hicache_size"] = size_gb
                            logger.info(
                                f"SGLang backend: Host cache size set to {size_gb}GB"
                            )

                    # Write policy
                    write_policy = hierarchical_cache.get("write_policy")
                    if write_policy is not None:
                        valid_policies = ["write_through", "write_back", "write_through_selective"]
                        if write_policy not in valid_policies:
                            raise ValueError(
                                f"hierarchical_cache.write_policy must be one of {valid_policies}, "
                                f"got: {write_policy}"
                            )
                        hierarchical_cache_kwargs["hicache_write_policy"] = write_policy
                        logger.info(
                            f"SGLang backend: Cache write policy set to '{write_policy}'"
                        )

                    # I/O backend
                    io_backend = hierarchical_cache.get("io_backend")
                    if io_backend is not None:
                        valid_backends = ["kernel", "direct", "kernel_ascend"]
                        if io_backend not in valid_backends:
                            raise ValueError(
                                f"hierarchical_cache.io_backend must be one of {valid_backends}, "
                                f"got: {io_backend}"
                            )
                        hierarchical_cache_kwargs["hicache_io_backend"] = io_backend
                        logger.info(
                            f"SGLang backend: Cache I/O backend set to '{io_backend}'"
                        )

                    # Memory layout
                    mem_layout = hierarchical_cache.get("mem_layout")
                    if mem_layout is not None:
                        valid_layouts = [
                            "layer_first", "page_first", "page_first_direct",
                            "page_head", "page_first_kv_split"
                        ]
                        if mem_layout not in valid_layouts:
                            raise ValueError(
                                f"hierarchical_cache.mem_layout must be one of {valid_layouts}, "
                                f"got: {mem_layout}"
                            )
                        hierarchical_cache_kwargs["hicache_mem_layout"] = mem_layout

                    # Storage backend configuration
                    storage_config = hierarchical_cache.get("storage", {})
                    if storage_config:
                        storage_backend = storage_config.get("backend")
                        if storage_backend is not None:
                            valid_storage = [
                                "file", "mooncake", "nixl", "hf3fs", "aibrix_kvcache", "eic"
                            ]
                            if storage_backend not in valid_storage:
                                raise ValueError(
                                    f"hierarchical_cache.storage.backend must be one of {valid_storage}, "
                                    f"got: {storage_backend}"
                                )
                            hierarchical_cache_kwargs["hicache_storage_backend"] = storage_backend
                            logger.info(
                                f"SGLang backend: Tier 3 storage backend set to '{storage_backend}'"
                            )

                        prefetch_policy = storage_config.get("prefetch_policy")
                        if prefetch_policy is not None:
                            hierarchical_cache_kwargs["hicache_storage_prefetch_policy"] = prefetch_policy

                        extra_config = storage_config.get("extra_config")
                        if extra_config is not None:
                            hierarchical_cache_kwargs["hicache_storage_backend_extra_config"] = extra_config

                # Eviction policy (can be set without enabling hierarchical cache)
                eviction_policy = hierarchical_cache.get("eviction_policy")
                if eviction_policy is not None:
                    valid_eviction = ["lru", "lfu"]
                    if eviction_policy not in valid_eviction:
                        raise ValueError(
                            f"hierarchical_cache.eviction_policy must be one of {valid_eviction}, "
                            f"got: {eviction_policy}"
                        )
                    hierarchical_cache_kwargs["radix_eviction_policy"] = eviction_policy
                    logger.info(
                        f"SGLang backend: Cache eviction policy set to '{eviction_policy}'"
                    )

                # KV cache dtype
                kv_dtype = hierarchical_cache.get("kv_cache_dtype")
                if kv_dtype is not None:
                    valid_dtypes = [
                        "auto", "float16", "bfloat16",
                        "fp8_e5m2", "fp8_e4m3", "fp4_e2m1"
                    ]
                    if kv_dtype not in valid_dtypes:
                        raise ValueError(
                            f"hierarchical_cache.kv_cache_dtype must be one of {valid_dtypes}, "
                            f"got: {kv_dtype}"
                        )
                    hierarchical_cache_kwargs["kv_cache_dtype"] = kv_dtype
                    if kv_dtype != "auto":
                        logger.info(
                            f"SGLang backend: KV cache dtype set to '{kv_dtype}'"
                        )

                # Page size
                page_size = hierarchical_cache.get("page_size")
                if page_size is not None:
                    if not isinstance(page_size, int) or page_size < 1:
                        raise ValueError(
                            f"hierarchical_cache.page_size must be a positive integer, "
                            f"got: {page_size}"
                        )
                    hierarchical_cache_kwargs["page_size"] = page_size

            # Process CPU offload configuration
            cpu_offload_kwargs = {}
            if cpu_offload:
                # CPU offload size
                size_gb = cpu_offload.get("size_gb")
                if size_gb is not None:
                    if not isinstance(size_gb, int) or size_gb < 0:
                        raise ValueError(
                            f"cpu_offload.size_gb must be a non-negative integer, "
                            f"got: {size_gb}"
                        )
                    cpu_offload_kwargs["cpu_offload_gb"] = size_gb
                    if size_gb > 0:
                        logger.info(
                            f"SGLang backend: CPU offload memory set to {size_gb}GB"
                        )

                # Enable CPU backup for weights
                if cpu_offload.get("enabled", False):
                    cpu_offload_kwargs["enable_weights_cpu_backup"] = True
                    logger.info("SGLang backend: CPU weight backup enabled")

                # Enable CPU backup for draft weights
                if cpu_offload.get("draft_weights_enabled", False):
                    cpu_offload_kwargs["enable_draft_weights_cpu_backup"] = True
                    logger.info("SGLang backend: Draft weights CPU backup enabled")

                # Offload mode
                offload_mode = cpu_offload.get("mode")
                if offload_mode is not None:
                    cpu_offload_kwargs["offload_mode"] = offload_mode

                # Layer grouping configuration
                group_config = cpu_offload.get("group", {})
                if group_config:
                    group_size = group_config.get("size")
                    if group_size is not None:
                        if not isinstance(group_size, int):
                            raise ValueError(
                                f"cpu_offload.group.size must be an integer, "
                                f"got: {type(group_size)}"
                            )
                        cpu_offload_kwargs["offload_group_size"] = group_size

                    num_offload = group_config.get("num_offload")
                    if num_offload is not None:
                        if not isinstance(num_offload, int) or num_offload < 1:
                            raise ValueError(
                                f"cpu_offload.group.num_offload must be a positive integer, "
                                f"got: {num_offload}"
                            )
                        cpu_offload_kwargs["offload_num_in_group"] = num_offload

                    prefetch_step = group_config.get("prefetch_step")
                    if prefetch_step is not None:
                        if not isinstance(prefetch_step, int) or prefetch_step < 1:
                            raise ValueError(
                                f"cpu_offload.group.prefetch_step must be a positive integer, "
                                f"got: {prefetch_step}"
                            )
                        cpu_offload_kwargs["offload_prefetch_step"] = prefetch_step

            # Log session configuration (sessions are managed at runtime via API, not at init)
            if sessions and sessions.get("enabled", False):
                default_capacity = sessions.get("default_capacity", 8192)
                pool_sessions = sessions.get("pool_sessions", False)
                max_pool_size = sessions.get("max_pool_size", 64)
                logger.info(
                    f"SGLang backend: Session-based generation enabled "
                    f"(capacity={default_capacity}, pool={pool_sessions}, max_pool={max_pool_size}). "
                    f"Use engine.open_session()/generate_with_session()/close_session() API."
                )

            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                # Set multi-node environment variables if configured
                for env_key, env_value in multi_node_env_vars.items():
                    os.environ[env_key] = env_value

                # Set health check environment variables if configured
                for env_key, env_value in health_checks_env_vars.items():
                    os.environ[env_key] = env_value

                # Fallback: Set LD_LIBRARY_PATH in the current process environment.
                # Note: The primary fix is runtime_env in actor_class.options() below,
                # which sets LD_LIBRARY_PATH at the OS level for SGLang's spawned subprocesses.
                # This os.environ modification is kept as a secondary measure for any code
                # that runs before the actor is created.
                if driver_ld_library_path:
                    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if current_ld_path:
                        os.environ["LD_LIBRARY_PATH"] = f"{driver_ld_library_path}:{current_ld_path}"
                    else:
                        os.environ["LD_LIBRARY_PATH"] = driver_ld_library_path

                actor_class = SGLangRayActor

                # Build engine kwargs, only including LoRA params when LoRA is enabled
                engine_kwargs = dict(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    pp_size=pipeline_parallel_size,
                    dp_size=data_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    mm_attention_backend=sglang_mm_attention_backend,
                    attention_backend=sglang_attention_backend,
                    enable_memory_saver=inference_engine_enable_sleep,
                    enable_lora=enable_lora,
                    # Expert parallelism for MoE
                    ep_size=expert_parallel_size,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    tokenizer=tokenizer,
                    **engine_init_kwargs_dict,
                )

                # Only add LoRA params when LoRA is enabled
                if enable_lora:
                    engine_kwargs["max_lora_rank"] = sglang_max_lora_rank
                    engine_kwargs["max_loras_per_batch"] = sglang_max_loras_per_batch
                    engine_kwargs["lora_backend"] = sglang_lora_backend
                    # SGLang requires lora_target_modules when no lora_paths is provided
                    # Default to common target modules for most models
                    if "lora_target_modules" not in engine_kwargs:
                        engine_kwargs["lora_target_modules"] = [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"
                        ]

                # Add speculative decoding params
                engine_kwargs.update(speculative_kwargs)

                # Add FP8 KV cache params
                engine_kwargs.update(kv_cache_kwargs)

                # Add model quantization params
                engine_kwargs.update(quantization_kwargs)

                # Add custom logit processor params
                engine_kwargs.update(custom_logit_processor_kwargs)

                # Add structured output / grammar backend params
                engine_kwargs.update(structured_output_kwargs)

                # Add RoPE scaling params
                engine_kwargs.update(rope_kwargs)

                # Add CUDA graph params
                engine_kwargs.update(cuda_graph_kwargs)

                # Add piecewise CUDA graph params
                engine_kwargs.update(piecewise_cuda_graph_kwargs)

                # Add torch.compile params
                engine_kwargs.update(torch_compile_kwargs)

                # Add attention backend params
                engine_kwargs.update(attention_kwargs)

                # Add LoRA hot-swapping params
                engine_kwargs.update(lora_kwargs)

                # Add priority scheduling params
                engine_kwargs.update(scheduling_kwargs)

                # Add disaggregated prefill/decode params
                engine_kwargs.update(disaggregation_kwargs)

                # Add multi-node inference params
                engine_kwargs.update(multi_node_kwargs)

                # Add Prometheus metrics and observability params
                engine_kwargs.update(metrics_kwargs)

                # Add deterministic inference params
                engine_kwargs.update(deterministic_kwargs)

                # Add load balancing and request routing params
                engine_kwargs.update(load_balancing_kwargs)

                # Add health check and watchdog params
                engine_kwargs.update(health_checks_kwargs)

                # Add hierarchical cache params
                engine_kwargs.update(hierarchical_cache_kwargs)

                # Add CPU offload params
                engine_kwargs.update(cpu_offload_kwargs)

                # Build runtime_env to propagate LD_LIBRARY_PATH to SGLang subprocesses.
                # SGLang uses multiprocessing with "spawn" method, which starts fresh processes
                # that don't inherit os.environ modifications. Using runtime_env sets the
                # environment at the OS level before the process starts.
                actor_runtime_env = {}
                if driver_ld_library_path:
                    actor_runtime_env["env_vars"] = {"LD_LIBRARY_PATH": driver_ld_library_path}

                # SGLang needs all TP*PP GPUs allocated to the actor because it spawns
                # child processes (not Ray actors) that share the parent's CUDA_VISIBLE_DEVICES.
                # Unlike vLLM's "ray" distributed executor which creates child Ray actors
                # with their own GPU allocations.
                sglang_tp_pp_gpus = tensor_parallel_size * pipeline_parallel_size
                if use_hybrid_engine:
                    # Hybrid engine shares GPUs with training - use fractional allocation
                    sglang_actor_num_gpus = 0.2 * sglang_tp_pp_gpus
                else:
                    sglang_actor_num_gpus = sglang_tp_pp_gpus

                engine = actor_class.options(
                    num_cpus=sglang_actor_num_gpus,
                    num_gpus=sglang_actor_num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=actor_runtime_env if actor_runtime_env else None,
                ).remote(**engine_kwargs)
                return engine

            # Apply runtime_env to the wrapper remote function as well to ensure
            # LD_LIBRARY_PATH is set before any subprocess spawning occurs.
            wrapper_runtime_env = {"env_vars": {"LD_LIBRARY_PATH": driver_ld_library_path}} if driver_ld_library_path else None
            engine = ray.get(get_sglang_engine.options(
                num_gpus=1,  # Need GPU to import SGLang without vLLM
                runtime_env=wrapper_runtime_env,
            ).remote())

            inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        if backend == "vllm":
            # NOTE(shu): set to 1 for LoRA
            sleep_level = 1 if enable_lora else sleep_level
            sleep_refs = [engine.inference_engine_actor.sleep.remote(level=sleep_level) for engine in engines]
        elif backend == "sglang":
            # NOTE(Charlie): we always need to sync weights after waking up,
            # see: https://github.com/sgl-project/sglang/issues/7939 for more details.
            # SGLang always discards weights, so sleep_level is effectively always 2.
            # Just warn if a different level is requested since it will be ignored.
            if sleep_level != 2:
                logger.warning(f"SGLang ignores sleep_level={sleep_level} (always uses level 2 - weights discarded)")
            sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
