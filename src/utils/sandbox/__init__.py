"""
Sandbox module for containerized code execution.

This module provides secure, isolated code execution in ephemeral Docker containers.
"""

from src.utils.sandbox.config import (
    ApprovalMode,
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
from src.utils.sandbox.approval import (
    ApprovalRequest,
    ApprovalStatus,
    cancel_approvals_for_trace,
    cleanup_old_requests,
    create_approval_request,
    get_approval_request,
    get_pending_approvals_for_conversation,
    get_pending_approvals_for_trace,
    register_approval_callback,
    resolve_approval,
    wait_for_approval,
)

__all__ = [
    # Config
    "ApprovalMode",
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
    # Approval
    "ApprovalRequest",
    "ApprovalStatus",
    "cancel_approvals_for_trace",
    "cleanup_old_requests",
    "create_approval_request",
    "get_approval_request",
    "get_pending_approvals_for_conversation",
    "get_pending_approvals_for_trace",
    "register_approval_callback",
    "resolve_approval",
    "wait_for_approval",
]
