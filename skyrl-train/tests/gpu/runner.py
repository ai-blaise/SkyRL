"""Helper script to run tests on GPU as a ray task.
Useful if working on a multi-node cluster.

Example:

```
uv run --isolated tests/gpu/runner.py --num-gpus 4 --test-file tests/gpu/test_models.py
```
or you can also do:

```
uv run --isolated tests/gpu/runner.py --num-gpus 4 --test-file tests/gpu/
```
"""

import os

# Unset RAY_RUNTIME_ENV_HOOK before importing Ray to avoid editable install issues.
if "RAY_RUNTIME_ENV_HOOK" in os.environ:
    del os.environ["RAY_RUNTIME_ENV_HOOK"]

import ray
import pytest
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--test-file", type=str, default="tests/gpu/test_models.py")
args = parser.parse_args()


@ray.remote(num_gpus=args.num_gpus)
def run_tests():
    return pytest.main(["-s", "-vvv", args.test_file])


if __name__ == "__main__":
    ray.init(
        runtime_env={
            "excludes": ["pyproject.toml", "uv.lock", ".python-version"],
        }
    )
    ray.get(run_tests.remote())
    ray.shutdown()
