"""
RBAC Registry - Centralized permission and role management

This module provides the core registry for role-based access control,
loading configuration from the main config (services.chat_app.auth.auth_roles)
or from a standalone auth_roles.yaml file.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry instance (singleton pattern)
_registry: Optional['RBACRegistry'] = None


class RBACConfigError(Exception):
    """Raised when RBAC configuration is invalid."""
    pass


class RBACRegistry:
    """
    Central registry for role-based access control.
    
    Manages:
    - Role definitions and inheritance
    - Permission-to-role mappings
    - Configuration validation
    - Permission lookups with caching
    
    Thread-safe singleton pattern - one registry per application.
    """
    
    def __init__(self, config: Dict[str, Any], app_name: Optional[str] = None):
        """
        Initialize the RBAC registry from configuration.
        
        Args:
            config: Dictionary loaded from auth_roles configuration
            app_name: Override app_name (e.g., from SSO_CLIENT_ID). If not provided,
                     uses config['app_name'] or defaults to 'archi-app'
        """
        self._config = config
        self._app_name = app_name or config.get('app_name', 'archi-app')
        self._default_role = config.get('default_role', 'base-user')
        self._sso_config = config.get('sso', {})
        self._roles: Dict[str, Dict] = config.get('roles', {})
        self._permissions: Dict[str, Dict] = config.get('permissions', {})
        
        # Cache for resolved permissions per role (including inherited)
        self._role_permissions_cache: Dict[str, Set[str]] = {}
        
        # Validate configuration on load
        self._validate_config()
        
        # Pre-compute permission sets for all roles
        self._build_permission_cache()
        
        logger.info(f"RBAC Registry initialized: {len(self._roles)} roles, {len(self._permissions)} permissions")
    
    def _validate_config(self) -> None:
        """
        Validate the RBAC configuration.
        
        Raises:
            RBACConfigError: If configuration is invalid
        """
        # Check for required fields
        if not self._roles:
            raise RBACConfigError("No roles defined in configuration")
        
        if self._default_role not in self._roles:
            raise RBACConfigError(f"Default role '{self._default_role}' is not defined in roles")
        
        # Check for circular inheritance
        for role_name in self._roles:
            visited = set()
            self._check_circular_inheritance(role_name, visited)
        
        # Validate inherited roles exist
        for role_name, role_config in self._roles.items():
            for parent in role_config.get('inherits', []):
                if parent not in self._roles:
                    raise RBACConfigError(
                        f"Role '{role_name}' inherits from undefined role '{parent}'"
                    )
        
        logger.debug("RBAC configuration validated successfully")
    
    def _check_circular_inheritance(self, role_name: str, visited: Set[str], path: List[str] = None) -> None:
        """
        Check for circular inheritance in role definitions.
        
        Args:
            role_name: Current role being checked
            visited: Set of roles already visited in this path
            path: Current inheritance path for error reporting
        """
        if path is None:
            path = []
        
        if role_name in visited:
            cycle = ' -> '.join(path + [role_name])
            raise RBACConfigError(f"Circular inheritance detected: {cycle}")
        
        visited.add(role_name)
        path.append(role_name)
        
        role_config = self._roles.get(role_name, {})
        for parent in role_config.get('inherits', []):
            if parent in self._roles:
                self._check_circular_inheritance(parent, visited.copy(), path.copy())
    
    def _build_permission_cache(self) -> None:
        """
        Pre-compute resolved permissions for each role including inheritance.
        """
        for role_name in self._roles:
            self._role_permissions_cache[role_name] = self._resolve_permissions(role_name)
        
        logger.debug(f"Permission cache built for {len(self._role_permissions_cache)} roles")
    
    def _resolve_permissions(self, role_name: str, visited: Set[str] = None) -> Set[str]:
        """
        Resolve all permissions for a role including inherited permissions.
        
        Args:
            role_name: Name of the role to resolve
            visited: Set of already-visited roles to prevent infinite loops
            
        Returns:
            Set of all permission strings granted to this role
        """
        if visited is None:
            visited = set()
        
        if role_name in visited:
            return set()  # Prevent infinite loop (shouldn't happen after validation)
        
        visited.add(role_name)
        
        role_config = self._roles.get(role_name, {})
        permissions = set(role_config.get('permissions', []))
        
        # Resolve inherited permissions
        for parent in role_config.get('inherits', []):
            if parent in self._roles:
                permissions.update(self._resolve_permissions(parent, visited.copy()))
        
        return permissions
    
    @property
    def app_name(self) -> str:
        """Get the SSO application name for role extraction."""
        return self._app_name
    
    @property
    def default_role(self) -> str:
        """Get the default role for users without configured roles."""
        return self._default_role
    
    @property
    def allow_anonymous(self) -> bool:
        """Check if anonymous access is allowed when SSO is enabled."""
        return self._sso_config.get('allow_anonymous', False)
    
    def get_role_permissions(self, role_name: str) -> Set[str]:
        """
        Get all permissions granted to a role (including inherited).
        
        Args:
            role_name: Name of the role
            
        Returns:
            Set of permission strings, empty set if role not found
        """
        return self._role_permissions_cache.get(role_name, set()).copy()
    
    def get_all_permissions_for_roles(self, roles: List[str]) -> Set[str]:
        """
        Get all permissions granted to a list of roles.
        
        Args:
            roles: List of role names
            
        Returns:
            Set of all permission strings granted by any of the roles
        """
        permissions = set()
        for role in roles:
            permissions.update(self.get_role_permissions(role))
        return permissions
    
    def has_permission(self, roles: List[str], permission: str) -> bool:
        """
        Check if any of the given roles has the specified permission.
        
        Args:
            roles: List of role names the user has
            permission: Permission string to check (e.g., 'upload:documents')
            
        Returns:
            True if any role grants the permission, False otherwise
        """
        for role in roles:
            role_perms = self._role_permissions_cache.get(role, set())
            
            # Check for wildcard permission (admin)
            if '*' in role_perms:
                return True
            
            # Check for specific permission
            if permission in role_perms:
                return True
        
        return False
    
    def is_valid_role(self, role_name: str) -> bool:
        """
        Check if a role name is defined in the configuration.
        
        Args:
            role_name: Role name to check
            
        Returns:
            True if role is defined, False otherwise
        """
        return role_name in self._roles
    
    def filter_valid_roles(self, roles: List[str]) -> List[str]:
        """
        Filter a list of roles to only include valid/configured roles.
        
        Args:
            roles: List of role names from JWT token
            
        Returns:
            List of roles that are defined in configuration
        """
        valid_roles = [r for r in roles if self.is_valid_role(r)]
        
        invalid_roles = set(roles) - set(valid_roles)
        if invalid_roles:
            logger.warning(f"Ignoring unmapped roles from JWT: {invalid_roles}")
        
        return valid_roles
    
    def get_roles_with_permission(self, permission: str) -> List[str]:
        """
        Get all roles that grant a specific permission.
        
        Useful for error messages ("You need role X or Y to do this").
        
        Args:
            permission: Permission string to check
            
        Returns:
            List of role names that grant this permission
        """
        roles_with_permission = []
        for role_name, perms in self._role_permissions_cache.items():
            if '*' in perms or permission in perms:
                roles_with_permission.append(role_name)
        return roles_with_permission
    
    def get_role_info(self, role_name: str) -> Optional[Dict]:
        """
        Get configuration info for a specific role.
        
        Args:
            role_name: Name of the role
            
        Returns:
            Role configuration dict or None if not found
        """
        return self._roles.get(role_name)

    @property
    def pass_descriptions_to_agent(self) -> bool:
        """
        Check if role descriptions should be passed to the agent.
        
        Requires SSO auth with auth_roles configured.
        """
        return self._config.get('pass_descriptions_to_agent', False)

    def get_role_descriptions(self, roles: List[str]) -> str:
        """
        Get a formatted string of role descriptions for the given roles.
        
        Used to append role context to agent system prompts when enabled.
        Falls back to role name if no description is configured.
        
        Args:
            roles: List of role names
            
        Returns:
            Formatted string like "role1 (description1), role2 (description2)"
            or empty string if no valid roles
        """
        if not roles:
            return ""
        
        descriptions = []
        for role in roles:
            role_info = self._roles.get(role)
            if role_info:
                desc = role_info.get('description', role)
                descriptions.append(f"{role} ({desc})")
            elif self.is_valid_role(role):
                descriptions.append(role)
        
        return ", ".join(descriptions)


def load_rbac_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load RBAC configuration from main config or YAML file.
    
    Priority order:
    1. Main config (services.chat_app.auth.auth_roles) if available
    2. Explicit config_path if provided
    3. Standard auth_roles.yaml locations
    4. Minimal defaults
    
    Note: app_name is preferably sourced from SSO_CLIENT_ID environment variable.
          Use get_registry() to automatically populate app_name from SSO config.
    
    Args:
        config_path: Optional path to auth_roles.yaml. If not provided,
                    checks main config first, then standard locations.
    
    Returns:
        Configuration dictionary
        
    Raises:
        RBACConfigError: If config is invalid
    """
    # First, try to load from main config
    try:
        from src.utils.config_access import get_full_config
        full_config = get_full_config()
        auth_roles_config = full_config.get('services', {}).get('chat_app', {}).get('auth', {}).get('auth_roles')
        
        if auth_roles_config:
            logger.info("Loading RBAC configuration from main config (services.chat_app.auth.auth_roles)")
            return auth_roles_config
    except Exception as e:
        logger.debug(f"Could not load auth_roles from main config: {e}")
    
    # Fallback: try standalone auth_roles.yaml file
    search_paths = [
        config_path,
        '/root/archi/configs/auth_roles.yaml',  # Runtime container path
        os.path.join(os.getcwd(), 'configs', 'auth_roles.yaml'),  # Local dev
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'configs', 'auth_roles.yaml'),
    ]
    
    config_file = None
    for path in search_paths:
        if path and os.path.isfile(path):
            config_file = path
            break
    
    if config_file:
        logger.info(f"Loading RBAC configuration from: {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    # Return minimal default config if no config found
    logger.warning("No auth_roles configuration found, using minimal defaults")
    return {
        'app_name': 'archi-app',
        'default_role': 'base-user',
        'sso': {'allow_anonymous': False},
        'roles': {
            'base-user': {
                'description': 'Default authenticated user',
                'permissions': ['chat:query', 'chat:history', 'chat:feedback', 'api-keys:manage']
            }
        },
        'permissions': {}
    }


def get_registry(config_path: Optional[str] = None, force_reload: bool = False) -> RBACRegistry:
    """
    Get the global RBAC registry instance (singleton).
    
    Automatically uses SSO_CLIENT_ID environment variable as app_name if available.
    
    Args:
        config_path: Optional path to configuration file
        force_reload: If True, reload configuration even if already loaded
        
    Returns:
        RBACRegistry instance
    """
    global _registry
    
    if _registry is None or force_reload:
        config = load_rbac_config(config_path)
        
        # Use SSO_CLIENT_ID as app_name if available
        from src.utils.env import read_secret
        app_name = read_secret('SSO_CLIENT_ID')
        
        _registry = RBACRegistry(config, app_name=app_name)
        
        if app_name:
            logger.info(f"RBAC registry initialized with app_name from SSO_CLIENT_ID: {app_name}")
    
    return _registry


def reset_registry() -> None:
    """
    Reset the global registry (for testing purposes).
    """
    global _registry
    _registry = None
