"""
RBAC Permissions - Permission checking utilities

This module provides utility functions for checking permissions
outside of the decorator context.
"""

from typing import List, Optional, Set
from flask import session

from src.utils.logging import get_logger
from src.utils.rbac.registry import get_registry
from src.utils.rbac.permission_enum import Permission

logger = get_logger(__name__)


def has_permission(permission: str, roles: Optional[List[str]] = None) -> bool:
    """
    Check if the current user (or provided roles) has a specific permission.
    
    This function can be used in templates or code to conditionally
    show/hide UI elements based on permissions.
    
    Args:
        permission: Permission string to check (e.g., 'upload:documents')
        roles: Optional list of roles to check. If not provided,
               uses roles from current session.
    
    Returns:
        True if permission is granted, False otherwise
    """
    if roles is None:
        if not session.get('logged_in'):
            return False
        roles = session.get('roles', [])
    
    # Ensure roles is a list
    if roles is None:
        roles = []
    
    registry = get_registry()
    return registry.has_permission(roles, permission)


def has_any_permission(permissions: List[str], roles: Optional[List[str]] = None) -> bool:
    """
    Check if the current user has ANY of the specified permissions.
    
    Args:
        permissions: List of permission strings to check
        roles: Optional list of roles (uses session if not provided)
    
    Returns:
        True if at least one permission is granted
    """
    if roles is None:
        if not session.get('logged_in'):
            return False
        roles = session.get('roles', [])
    
    # Ensure roles is a list
    if roles is None:
        roles = []
    
    registry = get_registry()
    for permission in permissions:
        if registry.has_permission(roles, permission):
            return True
    
    return False


def has_all_permissions(permissions: List[str], roles: Optional[List[str]] = None) -> bool:
    """
    Check if the current user has ALL of the specified permissions.
    
    Args:
        permissions: List of permission strings to check
        roles: Optional list of roles (uses session if not provided)
    
    Returns:
        True if all permissions are granted
    """
    if roles is None:
        if not session.get('logged_in'):
            return False
        roles = session.get('roles', [])
    
    # Ensure roles is a list
    if roles is None:
        roles = []
    
    registry = get_registry()
    for permission in permissions:
        if not registry.has_permission(roles, permission):
            return False
    
    return True


def check_permission(permission: str, roles: Optional[List[str]] = None) -> bool:
    """
    Alias for has_permission for compatibility.
    """
    return has_permission(permission, roles)


def get_user_permissions(roles: Optional[List[str]] = None) -> Set[str]:
    """
    Get all permissions available to the current user.
    
    Args:
        roles: Optional list of roles (uses session if not provided)
    
    Returns:
        Set of all permission strings granted to the user
    """
    if roles is None:
        if not session.get('logged_in'):
            return set()
        roles = session.get('roles', [])
    
    # Ensure roles is a list
    if roles is None:
        roles = []
    
    registry = get_registry()
    return registry.get_all_permissions_for_roles(roles)


def get_user_roles_from_session() -> List[str]:
    """
    Get the current user's roles from the session.
    
    Returns:
        List of role names, or empty list if not authenticated
    """
    if not session.get('logged_in'):
        return []
    
    return session.get('roles', [])


def get_role_descriptions(roles: Optional[List[str]] = None) -> str:
    """
    Get formatted role descriptions for the current user or provided roles.
    
    Requires SSO auth with auth_roles configured.
    
    Args:
        roles: Optional list of roles. If not provided, uses session roles.
    
    Returns:
        Formatted string like "role1 (description1), role2 (description2)"
        or empty string if not authenticated or no roles.
    """
    if roles is None:
        roles = get_user_roles_from_session()
    
    if not roles:
        return ""
    
    registry = get_registry()
    return registry.get_role_descriptions(roles)


def is_admin(roles: Optional[List[str]] = None) -> bool:
    """
    Check if the current user has admin role (wildcard permissions).
    
    Args:
        roles: Optional list of roles (uses session if not provided)
    
    Returns:
        True if user has admin-level access (any role with '*' permission)
    """
    if roles is None:
        if not session.get('logged_in'):
            return False
        roles = session.get('roles', [])
    
    # Ensure roles is a list
    if roles is None:
        roles = []
    
    # Check if any role has wildcard permission
    registry = get_registry()
    for role in roles:
        if role in registry._roles:
            role_perms = registry._role_permissions_cache.get(role, set())
            if '*' in role_perms:
                return True
    
    return False


def is_expert(roles: Optional[List[str]] = None) -> bool:
    """
    Check if the current user has expert/power user role.
    Expert is defined as having config:modify or upload:documents permissions.
    
    Args:
        roles: Optional list of roles (uses session if not provided)
    
    Returns:
        True if user has expert-level access
    """
    if roles is None:
        if not session.get('logged_in'):
            return False
        roles = session.get('roles', [])
    
    # Ensure roles is a list
    if roles is None:
        roles = []
    
    # Admin (wildcard) counts as expert
    if is_admin(roles):
        return True
    
    # Check for expert-level permissions
    return (has_permission(Permission.Config.MODIFY, roles) or
            has_permission(Permission.Upload.DOCUMENTS, roles))


def can_upload_documents(roles: Optional[List[str]] = None) -> bool:
    """
    Convenience function to check document upload permission.
    
    Returns:
        True if user can upload documents
    """
    return has_permission(Permission.Upload.DOCUMENTS, roles)


def can_modify_config(roles: Optional[List[str]] = None) -> bool:
    """
    Convenience function to check config modification permission.

    Returns:
        True if user can modify configuration
    """
    return has_permission(Permission.Config.MODIFY, roles)


def can_view_metrics(roles: Optional[List[str]] = None) -> bool:
    """
    Convenience function to check metrics viewing permission.

    Returns:
        True if user can view metrics
    """
    return has_permission(Permission.Metrics.VIEW, roles)


def get_permission_context() -> dict:
    """
    Get a context dictionary with all permission checks for templates.
    
    Useful for passing to Jinja2 templates to conditionally render UI.
    
    Returns:
        Dictionary with boolean flags for each major permission
    """
    if not session.get('logged_in'):
        return {
            'is_authenticated': False,
            'can_chat': False,
            'can_view_documents': False,
            'can_select_documents': False,
            'can_upload_documents': False,
            'can_manage_api_keys': False,
            'can_view_config': False,
            'can_modify_config': False,
            'can_view_metrics': False,
            'is_admin': False,
            'is_expert': False,
            'user_roles': [],
        }
    
    roles = session.get('roles', [])
    
    return {
        'is_authenticated': True,
        'can_chat': has_permission(Permission.Chat.QUERY, roles),
        'can_view_documents': has_permission(Permission.Documents.VIEW, roles),
        'can_select_documents': has_permission(Permission.Documents.SELECT, roles),
        'can_upload_documents': has_permission(Permission.Upload.DOCUMENTS, roles),
        'can_manage_api_keys': has_permission(Permission.ApiKeys.MANAGE, roles),
        'can_view_config': has_permission(Permission.Config.VIEW, roles),
        'can_modify_config': has_permission(Permission.Config.MODIFY, roles),
        'can_view_metrics': has_permission(Permission.Metrics.VIEW, roles),
        'is_admin': is_admin(roles),
        'is_expert': is_expert(roles),
        'user_roles': roles,
    }
