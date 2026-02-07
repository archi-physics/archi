"""
Sandbox configuration schema and validation.

This module defines the configuration dataclasses for the containerized
sandbox code execution feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ApprovalMode(str, Enum):
    """Approval mode for sandbox command execution."""
    
    AUTO = "auto"
    """Commands are executed automatically without user approval."""
    
    MANUAL = "manual"
    """Each command requires explicit user approval before execution."""


@dataclass
class RegistryConfig:
    """Configuration for a custom Docker registry."""
    
    url: str = ""
    """Registry URL (e.g., 'registry.cern.ch', 'ghcr.io')."""
    
    username_env: str = ""
    """Environment variable name containing the registry username."""
    
    password_env: str = ""
    """Environment variable name containing the registry password/token."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryConfig":
        """Create RegistryConfig from a dictionary."""
        if not data:
            return cls()
        return cls(
            url=data.get("url", ""),
            username_env=data.get("username_env", ""),
            password_env=data.get("password_env", ""),
        )
    
    def is_configured(self) -> bool:
        """Check if registry is configured with credentials."""
        return bool(self.url and self.username_env and self.password_env)
    
    def get_credentials(self) -> Optional[Dict[str, str]]:
        """Get registry credentials from environment variables."""
        import os
        if not self.is_configured():
            return None
        
        username = os.environ.get(self.username_env)
        password = os.environ.get(self.password_env)
        
        if not username or not password:
            logger.warning(
                f"Registry credentials not found in environment. "
                f"Expected {self.username_env} and {self.password_env}"
            )
            return None
        
        return {
            "username": username,
            "password": password,
            "registry": self.url,
        }


@dataclass
class ResourceLimits:
    """Resource constraints for sandbox containers."""
    
    memory: str = "256m"
    """Docker memory limit format (e.g., '256m', '1g')."""
    
    cpu: float = 0.5
    """CPU cores limit (e.g., 0.5 = half a core)."""
    
    pids_limit: int = 100
    """Maximum number of processes to prevent fork bombs."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLimits":
        """Create ResourceLimits from a dictionary."""
        return cls(
            memory=data.get("memory", "256m"),
            cpu=float(data.get("cpu", 0.5)),
            pids_limit=int(data.get("pids_limit", 100)),
        )


@dataclass
class SandboxConfig:
    """Configuration for the sandbox execution system."""
    
    enabled: bool = False
    """Whether sandbox execution is enabled for this deployment."""
    
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    """Approval mode for sandbox commands: 'auto' or 'manual'."""
    
    default_image: str = "python:3.11-slim"
    """Default Docker image to use when not specified."""
    
    image_allowlist: List[str] = field(default_factory=lambda: ["python:3.11-slim"])
    """List of allowed Docker images. Only images in this list can be used."""
    
    timeout: float = 30.0
    """Default execution timeout in seconds."""
    
    max_timeout: float = 300.0
    """Maximum allowed timeout (roles cannot exceed this)."""
    
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    """Resource limits for containers."""
    
    network_enabled: bool = True
    """Whether containers have outbound network access."""
    
    output_max_chars: int = 100000
    """Maximum characters for stdout/stderr output."""
    
    output_max_file_size: int = 10 * 1024 * 1024  # 10MB
    """Maximum size in bytes for captured output files."""
    
    docker_socket: str = "/var/run/docker.sock"
    """Path to Docker socket."""
    
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    """Custom Docker registry configuration for private images."""
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SandboxConfig":
        """Create SandboxConfig from a dictionary (e.g., from YAML config)."""
        if not data:
            return cls()
        
        resource_limits_data = data.get("resource_limits", {})
        resource_limits = ResourceLimits.from_dict(resource_limits_data)
        
        registry_data = data.get("registry", {})
        registry = RegistryConfig.from_dict(registry_data)
        
        # Parse approval_mode from string
        approval_mode_str = data.get("approval_mode", "auto").lower()
        try:
            approval_mode = ApprovalMode(approval_mode_str)
        except ValueError:
            logger.warning(
                f"Invalid approval_mode '{approval_mode_str}', defaulting to 'auto'"
            )
            approval_mode = ApprovalMode.AUTO
        
        return cls(
            enabled=bool(data.get("enabled", False)),
            approval_mode=approval_mode,
            default_image=data.get("default_image", "python:3.11-slim"),
            image_allowlist=data.get("image_allowlist", ["python:3.11-slim"]),
            timeout=float(data.get("timeout", 30.0)),
            max_timeout=float(data.get("max_timeout", 300.0)),
            resource_limits=resource_limits,
            network_enabled=bool(data.get("network_enabled", True)),
            output_max_chars=int(data.get("output_max_chars", 100000)),
            output_max_file_size=int(data.get("output_max_file_size", 10 * 1024 * 1024)),
            docker_socket=data.get("docker_socket", "/var/run/docker.sock"),
            registry=registry,
        )
    
    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of errors.
        
        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        
        if self.max_timeout <= 0:
            errors.append("max_timeout must be positive")
        
        if self.timeout > self.max_timeout:
            errors.append("timeout cannot exceed max_timeout")
        
        if not self.image_allowlist:
            errors.append("image_allowlist cannot be empty when sandbox is enabled")
        
        if self.default_image not in self.image_allowlist:
            errors.append(f"default_image '{self.default_image}' must be in image_allowlist")
        
        if self.resource_limits.cpu <= 0:
            errors.append("resource_limits.cpu must be positive")
        
        if self.resource_limits.pids_limit <= 0:
            errors.append("resource_limits.pids_limit must be positive")
        
        return errors
    
    def is_image_allowed(self, image: str) -> bool:
        """Check if an image is in the allowlist."""
        return image in self.image_allowlist


@dataclass
class RoleSandboxOverrides:
    """Per-role sandbox configuration overrides."""
    
    allowed_images: Union[List[str], Literal["*"]] = field(default_factory=list)
    """
    Images this role can use. Either a list of image names (subset of deployment allowlist)
    or "*" to allow all images from deployment allowlist.
    """
    
    timeout: Optional[float] = None
    """Role-specific timeout override (capped at deployment max_timeout)."""
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RoleSandboxOverrides":
        """Create RoleSandboxOverrides from a dictionary."""
        if not data:
            return cls()
        
        allowed_images = data.get("allowed_images", [])
        timeout = data.get("timeout")
        
        return cls(
            allowed_images=allowed_images,
            timeout=float(timeout) if timeout is not None else None,
        )


def resolve_effective_config(
    base_config: SandboxConfig,
    role_overrides: Optional[RoleSandboxOverrides] = None,
) -> SandboxConfig:
    """
    Resolve effective sandbox config by applying role overrides to base config.
    
    Role overrides can only restrict (not expand) the base configuration.
    
    Args:
        base_config: The deployment-level sandbox configuration.
        role_overrides: Optional role-specific overrides.
    
    Returns:
        Effective SandboxConfig with role restrictions applied.
    """
    if not role_overrides:
        return base_config
    
    # Resolve timeout (role can set custom, but capped at max_timeout)
    effective_timeout = base_config.timeout
    if role_overrides.timeout is not None:
        effective_timeout = min(role_overrides.timeout, base_config.max_timeout)
    
    # Resolve allowed images
    if role_overrides.allowed_images == "*":
        effective_images = base_config.image_allowlist
    elif role_overrides.allowed_images:
        # Role can only use images that are in both role list AND deployment allowlist
        effective_images = [
            img for img in role_overrides.allowed_images
            if img in base_config.image_allowlist
        ]
    else:
        effective_images = base_config.image_allowlist
    
    # Resolve default image (must be in effective images)
    effective_default = base_config.default_image
    if effective_default not in effective_images and effective_images:
        effective_default = effective_images[0]
    
    return SandboxConfig(
        enabled=base_config.enabled,
        approval_mode=base_config.approval_mode,
        default_image=effective_default,
        image_allowlist=effective_images,
        timeout=effective_timeout,
        max_timeout=base_config.max_timeout,
        resource_limits=base_config.resource_limits,
        network_enabled=base_config.network_enabled,
        output_max_chars=base_config.output_max_chars,
        output_max_file_size=base_config.output_max_file_size,
        docker_socket=base_config.docker_socket,
    )


def get_sandbox_config() -> SandboxConfig:
    """
    Load sandbox configuration from the deployment config.
    
    Returns:
        SandboxConfig loaded from archi.sandbox config section.
    """
    try:
        from src.utils.config_access import get_archi_config
        archi_config = get_archi_config()
        sandbox_data = archi_config.get("sandbox", {})
        return SandboxConfig.from_dict(sandbox_data)
    except Exception as e:
        logger.warning(f"Failed to load sandbox config, using defaults: {e}")
        return SandboxConfig()


def get_role_sandbox_overrides(role_config: Dict[str, Any]) -> Optional[RoleSandboxOverrides]:
    """
    Extract sandbox overrides from a role configuration.
    
    Args:
        role_config: The role configuration dictionary.
    
    Returns:
        RoleSandboxOverrides if present, None otherwise.
    """
    sandbox_data = role_config.get("sandbox")
    if not sandbox_data:
        return None
    return RoleSandboxOverrides.from_dict(sandbox_data)
