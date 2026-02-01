"""Pytest configuration for SkyRL tests.

Configures Ray to avoid issues with editable installs in Ray workers.
"""

import os

# Unset RAY_RUNTIME_ENV_HOOK before importing Ray to avoid editable install issues.
# Ray's UV hook tries to replicate editable install paths (e.g., sglang @ editable+../../sglang/python)
# that don't exist in Ray workers, causing "Failed to generate package metadata" errors.
# This must be done before Ray is imported anywhere.
if "RAY_RUNTIME_ENV_HOOK" in os.environ:
    del os.environ["RAY_RUNTIME_ENV_HOOK"]
