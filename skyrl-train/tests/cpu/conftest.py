import os

# Unset RAY_RUNTIME_ENV_HOOK before importing Ray to avoid editable install issues.
# Ray's UV hook tries to replicate editable install paths (e.g., sglang @ editable+../../sglang/python)
# that don't exist in Ray workers, causing "Failed to generate package metadata" errors.
# This must be done before Ray is imported.
if "RAY_RUNTIME_ENV_HOOK" in os.environ:
    del os.environ["RAY_RUNTIME_ENV_HOOK"]

import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    """Initialize Ray once for the entire test session."""
    if not ray.is_initialized():
        # Exclude pyproject.toml and uv.lock to prevent uv from trying to install
        # editable dependencies (e.g., sglang @ editable+../../sglang/python) in workers.
        # The relative path doesn't exist in Ray's runtime resources directory.
        ray.init(
            runtime_env={
                "excludes": ["pyproject.toml", "uv.lock", ".python-version"],
            }
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
