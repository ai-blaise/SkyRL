import os

# Unset RAY_RUNTIME_ENV_HOOK before importing Ray to avoid editable install issues.
# Ray's UV hook tries to replicate editable install paths (e.g., sglang @ editable+../../sglang/python)
# that don't exist in Ray workers, causing "Failed to generate package metadata" errors.
# This must be done before Ray is imported.
if "RAY_RUNTIME_ENV_HOOK" in os.environ:
    del os.environ["RAY_RUNTIME_ENV_HOOK"]

import pytest
import ray
from loguru import logger
from functools import lru_cache
from skyrl_train.utils.utils import peer_access_supported


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()

    # Unset RAY_RUNTIME_ENV_HOOK to avoid issues with editable installs (e.g., sglang)
    # Ray's UV hook tries to replicate editable paths that don't exist in workers.
    # See: https://github.com/sgl-project/sglang/issues/9039
    if "RAY_RUNTIME_ENV_HOOK" in os.environ:
        logger.info("Unsetting RAY_RUNTIME_ENV_HOOK to avoid editable install issues")
        del os.environ["RAY_RUNTIME_ENV_HOOK"]

    # TODO (team): maybe we should use the default config and use prepare_runtime_environment in some way
    env_vars = {"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}

    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars.update(
            {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
            }
        )

    # needed for megatron tests
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars["NVTE_FUSED_ATTN"] = "0"

    # NCCL settings to avoid CUDA/NCCL issues during weight sync
    # These are used in SGLang's test suite for update_weights_from_distributed
    # See: sglang/test/registered/rl/test_update_weights_from_distributed.py
    env_vars["NCCL_CUMEM_ENABLE"] = "0"
    env_vars["NCCL_NVLS_ENABLE"] = "0"

    logger.info(f"Initializing Ray with environment variables: {env_vars}")
    # Exclude pyproject.toml and uv.lock to prevent uv from trying to install
    # editable dependencies (e.g., sglang @ editable+../../sglang/python) in workers.
    # The relative path doesn't exist in Ray's runtime resources directory.
    ray.init(
        runtime_env={
            "env_vars": env_vars,
            "excludes": ["pyproject.toml", "uv.lock", ".python-version"],
        }
    )

    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
