"""
Sandbox module for containerized code execution.

This module provides secure, isolated code execution in ephemeral Docker containers.
"""

from src.utils.sandbox.config import (
    RegistryConfig,
    ResourceLimits,
    RoleSandboxOverrides,
    SandboxConfig,
    get_role_sandbox_overrides,
    get_sandbox_config,
    resolve_effective_config,
)
from src.utils.sandbox.executor import (
    FileOutput,
    SandboxExecutor,
    SandboxResult,
)

__all__ = [
    # Config
    "RegistryConfig",
    "ResourceLimits",
    "RoleSandboxOverrides",
    "SandboxConfig",
    "get_role_sandbox_overrides",
    "get_sandbox_config",
    "resolve_effective_config",
    # Executor
    "FileOutput",
    "SandboxExecutor",
    "SandboxResult",
]
