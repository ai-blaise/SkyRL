import os

# Unset RAY_RUNTIME_ENV_HOOK before importing Ray to avoid editable install issues.
if "RAY_RUNTIME_ENV_HOOK" in os.environ:
    del os.environ["RAY_RUNTIME_ENV_HOOK"]

import pytest
import ray
from tests.gpu.utils import ray_init_for_tests


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
