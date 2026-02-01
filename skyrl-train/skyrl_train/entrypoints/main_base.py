"""
Main entrypoint for training.
"""

from ray.util.placement_group import placement_group, PlacementGroup

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from skyrl_train.dataset import PromptDataset
from skyrl_train.utils import validate_cfg

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from skyrl_train.utils.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl_train.generators.base import GeneratorInterface
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import ray

import os
import hydra
from loguru import logger
from skyrl_train.utils.tracking import Tracking
import multiprocessing as mp

# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


def create_ray_wrapped_inference_engines_from_config(cfg: DictConfig, colocate_pg, tokenizer: PreTrainedTokenizerBase):
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines

    engine_kwargs = {
        "num_inference_engines": cfg.generator.num_inference_engines,
        "tensor_parallel_size": cfg.generator.inference_engine_tensor_parallel_size,
        "pipeline_parallel_size": cfg.generator.inference_engine_pipeline_parallel_size,
        "model_dtype": cfg.generator.model_dtype,
        "pretrain": cfg.trainer.policy.model.path,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": cfg.generator.vllm_v1_disable_multiproc,
        "enable_prefix_caching": cfg.generator.enable_prefix_caching,
        "enforce_eager": cfg.generator.enforce_eager,
        "expert_parallel_size": cfg.generator.inference_engine_expert_parallel_size,
        "data_parallel_size": cfg.generator.inference_engine_data_parallel_size,
        "shared_pg": colocate_pg,
        "gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
        "inference_engine_enable_sleep": cfg.trainer.placement.colocate_all,
        "async_engine": cfg.generator.async_engine,
        "max_num_batched_tokens": cfg.generator.max_num_batched_tokens,
        "max_num_seqs": cfg.generator.max_num_seqs,
        "tokenizer": tokenizer,
        "backend": cfg.generator.backend,
        "engine_init_kwargs": cfg.generator.engine_init_kwargs,
    }

    # Conditionally add LoRA parameters if LoRA is enabled
    if cfg.trainer.policy.model.lora.rank > 0 and cfg.trainer.strategy != "megatron":
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = cfg.trainer.policy.model.lora.rank
        engine_kwargs["sleep_level"] = 1
        engine_kwargs["max_loras"] = 1
        engine_kwargs["fully_sharded_loras"] = cfg.generator.fully_sharded_loras

        # TODO(devpatel): Bandaid solution, replace this once we have a better
        # solution for LoRA performance degradation on the vLLM side
        if cfg.generator.enforce_eager and cfg.generator.backend == "vllm":
            logger.warning(
                "LoRA is enabled but generator.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            engine_kwargs["enforce_eager"] = False

    if (rope_scaling := cfg.generator.get("rope_scaling", None)) is not None:
        engine_kwargs["rope_scaling"] = rope_scaling
    if (rope_theta := cfg.generator.get("rope_theta", None)) is not None:
        engine_kwargs["rope_theta"] = rope_theta

    # Add speculative decoding config (SGLang only)
    if (speculative_decoding := cfg.generator.get("speculative_decoding", None)) is not None:
        if cfg.generator.backend != "sglang":
            if speculative_decoding.get("enabled", False):
                raise ValueError(
                    "Speculative decoding is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
        else:
            # Convert OmegaConf to dict for the wrapper
            from omegaconf import OmegaConf
            engine_kwargs["speculative_decoding"] = OmegaConf.to_container(
                speculative_decoding, resolve=True
            )

    # Add FP8 KV cache config (SGLang only)
    if (kv_cache := cfg.generator.get("kv_cache", None)) is not None:
        kv_cache_dtype = kv_cache.get("dtype", "auto")
        # Only validate/pass if non-default values are set
        if kv_cache_dtype != "auto" or kv_cache.get("quantization_param_path") or kv_cache.get("fp8_gemm_backend", "auto") != "auto":
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "FP8 KV cache configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["kv_cache"] = OmegaConf.to_container(
                    kv_cache, resolve=True
                )

    # Add model quantization config (SGLang only)
    if (quantization := cfg.generator.get("quantization", None)) is not None:
        quant_method = quantization.get("method")
        load_format = quantization.get("load_format", "auto")
        # Only validate/pass if non-default values are set
        if quant_method is not None or load_format != "auto" or quantization.get("enable_fp32_lm_head", False):
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Model quantization configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["quantization"] = OmegaConf.to_container(
                    quantization, resolve=True
                )

    # Add custom logit processor config (SGLang only)
    if (custom_logit_processor := cfg.generator.get("custom_logit_processor", None)) is not None:
        if custom_logit_processor.get("enabled", False):
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Custom logit processor configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["custom_logit_processor"] = OmegaConf.to_container(
                    custom_logit_processor, resolve=True
                )

    # Add structured output / grammar backend config (SGLang only)
    if (structured_output := cfg.generator.get("structured_output", None)) is not None:
        grammar_backend = structured_output.get("grammar_backend")
        if grammar_backend is not None:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Structured output grammar_backend is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["structured_output"] = OmegaConf.to_container(
                    structured_output, resolve=True
                )

    # Add CUDA graph config (SGLang only)
    if (cuda_graph := cfg.generator.get("cuda_graph", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            cuda_graph.get("disable", False) or
            cuda_graph.get("max_bs") is not None or
            cuda_graph.get("batch_sizes") is not None or
            cuda_graph.get("disable_padding", False) or
            cuda_graph.get("enable_profiling", False) or
            cuda_graph.get("enable_gc", False)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "CUDA graph configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["cuda_graph"] = OmegaConf.to_container(
                    cuda_graph, resolve=True
                )

    # Add piecewise CUDA graph config (SGLang only)
    if (piecewise_cuda_graph := cfg.generator.get("piecewise_cuda_graph", None)) is not None:
        if piecewise_cuda_graph.get("enabled", False):
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Piecewise CUDA graph configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["piecewise_cuda_graph"] = OmegaConf.to_container(
                    piecewise_cuda_graph, resolve=True
                )

    # Add torch.compile config (SGLang only)
    if (torch_compile := cfg.generator.get("torch_compile", None)) is not None:
        if torch_compile.get("enabled", False):
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "torch.compile configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["torch_compile"] = OmegaConf.to_container(
                    torch_compile, resolve=True
                )

    # Add attention backend config (SGLang only)
    if (attention := cfg.generator.get("attention", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            attention.get("backend") is not None or
            attention.get("prefill_backend") is not None or
            attention.get("decode_backend") is not None or
            attention.get("mm_backend") is not None or
            attention.get("enable_double_sparsity", False) or
            (attention.get("nsa", {}).get("prefill_backend") is not None) or
            (attention.get("nsa", {}).get("decode_backend") is not None)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Attention backend configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["attention"] = OmegaConf.to_container(
                    attention, resolve=True
                )

    # Add LoRA hot-swapping config (SGLang only)
    if (lora_config := cfg.generator.get("lora", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            lora_config.get("paths") is not None or
            lora_config.get("max_rank") is not None or
            lora_config.get("target_modules") is not None or
            lora_config.get("max_loras_per_batch", 8) != 8 or
            lora_config.get("max_loaded_loras") is not None or
            lora_config.get("eviction_policy", "lru") != "lru" or
            lora_config.get("backend", "csgmv") != "csgmv" or
            lora_config.get("max_chunk_size", 16) != 16
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "LoRA hot-swapping configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["lora_config"] = OmegaConf.to_container(
                    lora_config, resolve=True
                )

    # Add priority scheduling config (SGLang only)
    if (scheduling_config := cfg.generator.get("scheduling", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            scheduling_config.get("policy", "fcfs") != "fcfs" or
            scheduling_config.get("enable_priority", False) is True or
            scheduling_config.get("abort_on_priority_when_disabled", False) is True or
            scheduling_config.get("low_priority_values_first", False) is True or
            scheduling_config.get("preemption_threshold", 10) != 10 or
            scheduling_config.get("conservativeness", 1.0) != 1.0 or
            scheduling_config.get("chunked_prefill_size") is not None or
            scheduling_config.get("enable_dynamic_chunking", False) is True or
            scheduling_config.get("max_running_requests") is not None or
            scheduling_config.get("max_queued_requests") is not None or
            scheduling_config.get("max_prefill_tokens") is not None or
            scheduling_config.get("max_total_tokens") is not None
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Priority scheduling configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["scheduling"] = OmegaConf.to_container(
                    scheduling_config, resolve=True
                )

    # Add disaggregated prefill/decode config (SGLang only)
    if (disaggregation_config := cfg.generator.get("disaggregation", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            disaggregation_config.get("mode", "null") != "null" or
            disaggregation_config.get("transfer_backend", "mooncake") != "mooncake" or
            disaggregation_config.get("bootstrap_port", 8998) != 8998 or
            disaggregation_config.get("ib_device") is not None or
            disaggregation_config.get("num_reserved_decode_tokens", 512) != 512 or
            disaggregation_config.get("enable_dp_attention", False) is True or
            disaggregation_config.get("enable_dp_lm_head", False) is True or
            # Check decode sub-config
            (disaggregation_config.get("decode", {}).get("tp_size") is not None) or
            (disaggregation_config.get("decode", {}).get("dp_size") is not None) or
            (disaggregation_config.get("decode", {}).get("enable_offload_kvcache", False) is True) or
            (disaggregation_config.get("decode", {}).get("polling_interval", 1) != 1) or
            (disaggregation_config.get("decode", {}).get("enable_fake_auto", False) is True) or
            # Check prefill sub-config
            (disaggregation_config.get("prefill", {}).get("pp_size", 1) != 1)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Disaggregated prefill/decode configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["disaggregation"] = OmegaConf.to_container(
                    disaggregation_config, resolve=True
                )

    # Add multi-node inference config (SGLang only)
    if (multi_node_config := cfg.generator.get("multi_node", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            multi_node_config.get("nnodes") is not None or
            multi_node_config.get("node_rank") is not None or
            multi_node_config.get("dist_init_addr") is not None or
            # Check NCCL sub-config
            (multi_node_config.get("nccl", {}).get("enable_symm_mem", False) is True) or
            (multi_node_config.get("nccl", {}).get("enable_nvls", False) is True) or
            (multi_node_config.get("nccl", {}).get("timeout") is not None) or
            (multi_node_config.get("nccl", {}).get("debug_level") is not None) or
            multi_node_config.get("enable_ib_optimization", False) is True or
            multi_node_config.get("cuda_device_max_connections") is not None
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Multi-node inference configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["multi_node"] = OmegaConf.to_container(
                    multi_node_config, resolve=True
                )

    # Add Prometheus metrics and observability config (SGLang only)
    if (metrics_config := cfg.generator.get("metrics", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            metrics_config.get("enabled", False) is True or
            metrics_config.get("enable_for_all_schedulers", False) is True or
            metrics_config.get("collect_tokens_histogram", False) is True or
            metrics_config.get("prompt_tokens_buckets") is not None or
            metrics_config.get("generation_tokens_buckets") is not None or
            # Check buckets sub-config
            (metrics_config.get("buckets", {}).get("time_to_first_token") is not None) or
            (metrics_config.get("buckets", {}).get("inter_token_latency") is not None) or
            (metrics_config.get("buckets", {}).get("e2e_request_latency") is not None) or
            # Check export_to_file sub-config
            (metrics_config.get("export_to_file", {}).get("enabled", False) is True) or
            # Check custom_labels sub-config
            (metrics_config.get("custom_labels", {}).get("header", "x-custom-labels") != "x-custom-labels") or
            (metrics_config.get("custom_labels", {}).get("allowed") is not None) or
            # Check tracing sub-config
            (metrics_config.get("tracing", {}).get("enabled", False) is True) or
            # Check logging sub-config
            (metrics_config.get("logging", {}).get("enabled", False) is True)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Prometheus metrics configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["metrics"] = OmegaConf.to_container(
                    metrics_config, resolve=True
                )

    # Add deterministic inference config (SGLang only)
    # Enables reproducible inference for on-policy RL training
    if (deterministic_config := cfg.generator.get("deterministic_inference", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            deterministic_config.get("enabled", False) is True or
            deterministic_config.get("rl_on_policy_target") is not None
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Deterministic inference configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["deterministic_inference"] = OmegaConf.to_container(
                    deterministic_config, resolve=True
                )

    # Add load balancing and request routing config (SGLang only)
    if (load_balancing_config := cfg.generator.get("load_balancing", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            load_balancing_config.get("method") is not None or
            # Check expert_parallelism sub-config
            (load_balancing_config.get("expert_parallelism", {}).get("ep_size") is not None) or
            (load_balancing_config.get("expert_parallelism", {}).get("dispatch_algorithm") is not None) or
            (load_balancing_config.get("expert_parallelism", {}).get("num_redundant_experts") is not None) or
            (load_balancing_config.get("expert_parallelism", {}).get("init_expert_location") is not None) or
            # Check eplb sub-config
            (load_balancing_config.get("eplb", {}).get("enabled", False) is True) or
            # Check expert_metrics sub-config
            (load_balancing_config.get("expert_metrics", {}).get("recorder_mode") is not None) or
            (load_balancing_config.get("expert_metrics", {}).get("recorder_buffer_size") is not None) or
            (load_balancing_config.get("expert_metrics", {}).get("enabled", False) is True) or
            # Check batching sub-config
            (load_balancing_config.get("batching", {}).get("max_prefill_tokens") is not None) or
            (load_balancing_config.get("batching", {}).get("max_total_tokens") is not None) or
            (load_balancing_config.get("batching", {}).get("tokenizer_worker_num") is not None)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Load balancing configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["load_balancing"] = OmegaConf.to_container(
                    load_balancing_config, resolve=True
                )

    # Add health checks and Kubernetes probe config (SGLang only)
    if (health_checks_config := cfg.generator.get("health_checks", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            # Check watchdog sub-config
            (health_checks_config.get("watchdog", {}).get("timeout") is not None) or
            (health_checks_config.get("watchdog", {}).get("soft_timeout") is not None) or
            # Check dist_timeout
            (health_checks_config.get("dist_timeout") is not None) or
            # Check endpoint sub-config
            (health_checks_config.get("endpoint", {}).get("timeout") is not None) or
            (health_checks_config.get("endpoint", {}).get("enable_generation") is not None) or
            # Check startup sub-config
            (health_checks_config.get("startup", {}).get("weights_ready_timeout") is not None) or
            (health_checks_config.get("startup", {}).get("warmup_timeout") is not None)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Health checks configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["health_checks"] = OmegaConf.to_container(
                    health_checks_config, resolve=True
                )

    # Add hierarchical cache config (SGLang only)
    if (hicache_config := cfg.generator.get("hierarchical_cache", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            hicache_config.get("enabled", False) is True or
            # Check host_memory sub-config
            (hicache_config.get("host_memory", {}).get("ratio") is not None) or
            (hicache_config.get("host_memory", {}).get("size_gb") is not None) or
            # Check other params
            (hicache_config.get("write_policy") is not None) or
            (hicache_config.get("io_backend") is not None) or
            (hicache_config.get("mem_layout") is not None) or
            # Check storage sub-config
            (hicache_config.get("storage", {}).get("backend") is not None) or
            (hicache_config.get("storage", {}).get("prefetch_policy") is not None) or
            (hicache_config.get("storage", {}).get("extra_config") is not None) or
            # Check other cache params
            (hicache_config.get("eviction_policy") is not None) or
            (hicache_config.get("kv_cache_dtype") is not None) or
            (hicache_config.get("page_size") is not None)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Hierarchical cache configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["hierarchical_cache"] = OmegaConf.to_container(
                    hicache_config, resolve=True
                )

    # Add CPU offload config (SGLang only)
    if (cpu_offload_config := cfg.generator.get("cpu_offload", None)) is not None:
        # Check if any non-default values are set
        has_custom_config = (
            (cpu_offload_config.get("size_gb") is not None) or
            cpu_offload_config.get("enabled", False) is True or
            cpu_offload_config.get("draft_weights_enabled", False) is True or
            (cpu_offload_config.get("mode") is not None) or
            # Check group sub-config
            (cpu_offload_config.get("group", {}).get("size") is not None) or
            (cpu_offload_config.get("group", {}).get("num_offload") is not None) or
            (cpu_offload_config.get("group", {}).get("prefetch_step") is not None)
        )
        if has_custom_config:
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "CPU offload configuration is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["cpu_offload"] = OmegaConf.to_container(
                    cpu_offload_config, resolve=True
                )

    # Add session-based generation config (SGLang only)
    if (sessions_config := cfg.generator.get("sessions", None)) is not None:
        if sessions_config.get("enabled", False):
            if cfg.generator.backend != "sglang":
                raise ValueError(
                    "Session-based generation is only supported with SGLang backend. "
                    f"Current backend: {cfg.generator.backend}"
                )
            else:
                from omegaconf import OmegaConf
                engine_kwargs["sessions"] = OmegaConf.to_container(
                    sessions_config, resolve=True
                )

    return create_ray_wrapped_inference_engines(**engine_kwargs)


def create_remote_inference_engines_from_config(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase):
    # TODO(tgriggs): We may want a separate config for the model name in case
    # it's different from the name used in the OpenAI API
    return create_remote_inference_engines(
        urls=cfg.generator.remote_inference_engine_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=cfg.generator.backend,
        tokenizer=tokenizer,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
    )


class BasePPOExp:
    def __init__(self, cfg: DictConfig):
        """
        Initializes a PPO experiment.

        The `cfg` passed here will be the final config from Hydra, including CLI overrides.
        """
        self.cfg = cfg
        self.tokenizer = self.get_tokenizer()
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.get_eval_dataset()
        self.colocate_pg = self.get_colocate_pg()

    @staticmethod
    def get_cfg_as_str(dict_cfg: DictConfig) -> str:
        return OmegaConf.to_yaml(dict_cfg)

    def get_tokenizer(self, padding_side="left"):
        """Initializes a tokenizer for the given model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not self.cfg.trainer.disable_fast_tokenizer,
        )
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            PromptDataset: The training dataset.
        """
        prompts_dataset = PromptDataset(
            datasets=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=8,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            PromptDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = PromptDataset(
                datasets=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                num_workers=8,
            )
            return prompts_dataset
        return None

    def get_colocate_pg(self, timeout: int = SKYRL_RAY_PG_TIMEOUT_IN_S) -> PlacementGroup:
        """Initializes a placement group for colocated training.

        A single placement group that packs all the inference engines together is created.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            PlacementGroup: The placement group for colocated training.
        """
        if self.cfg.trainer.placement.colocate_all:
            pg = placement_group(
                [{"GPU": 1, "CPU": 1}]
                * self.cfg.generator.num_inference_engines
                * self.cfg.generator.inference_engine_tensor_parallel_size
                * self.cfg.generator.inference_engine_pipeline_parallel_size
                * self.cfg.generator.inference_engine_data_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator

        return SkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """Initializes the trainer.

        Returns:
            RayPPOTrainer: The trainer.
        """
        return RayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_tracker(self):
        """Initializes the tracker for experiment tracking.

        Returns:
            Tracking: The tracker.
        """
        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def _get_worker_classes(self):
        """Get worker classes based on strategy and model type.

        Returns worker classes appropriate for the configured strategy (FSDP/Megatron)
        and model type (HF/nmoe).

        Returns:
            Tuple of (PolicyWorker, CriticWorker, RefWorker) classes
        """
        from skyrl_train.model_factory import get_model_type

        strategy = self.cfg.trainer.strategy
        model_type = get_model_type(self.cfg)

        logger.info(f"[WorkerSelection] strategy={strategy}, model_type={model_type}")

        if strategy in ("fsdp", "fsdp2"):
            if model_type == "nmoe":
                # Use NMoE-specific FSDP workers
                from skyrl_train.workers.fsdp.fsdp_worker_nmoe import get_nmoe_workers
                PolicyWorker, CriticWorker, RefWorker = get_nmoe_workers()
                logger.info("[WorkerSelection] Using NMoE FSDP workers")
            else:
                # Use standard HF FSDP workers
                from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
                logger.info("[WorkerSelection] Using HuggingFace FSDP workers")
        elif strategy == "megatron":
            if model_type == "nmoe":
                # Use NMoE-specific Megatron workers with RDEP expert parallelism
                from skyrl_train.workers.megatron.nmoe_megatron_worker import (
                    NMoEMegatronPolicyWorker as PolicyWorker,
                    NMoEMegatronCriticWorker as CriticWorker,
                    NMoEMegatronRefWorker as RefWorker,
                )
                logger.info("[WorkerSelection] Using NMoE Megatron workers with RDEP")
            else:
                from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
                logger.info("[WorkerSelection] Using Megatron workers")
        else:
            raise ValueError(f"Unknown strategy type: {strategy}")

        return PolicyWorker, CriticWorker, RefWorker

    def _setup_trainer(self):
        """Setup and return the trainer.

        Instantiates the trainer and all the associated models for training.

        Returns:
            RayPPOTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        # Select worker classes based on strategy and model type
        PolicyWorker, CriticWorker, RefWorker = self._get_worker_classes()

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)

        generator: GeneratorInterface = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build the models
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker)
        return trainer

    def run(self):
        trainer = self._setup_trainer()
        # Start the training loop
        trainer.train()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
