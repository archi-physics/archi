"""Base utilities and RBAC decorators for agent tools."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, TypeVar

from langchain.tools import tool

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Type variable for generic function signatures
F = TypeVar('F', bound=Callable)


def check_tool_permission(required_permission: str) -> tuple[bool, Optional[str]]:
    """
    Check if the current user has permission to use a tool.
    
    Uses the Flask session to get user roles and checks against the RBAC registry.
    This function is designed to fail open in non-web contexts (CLI, testing)
    and when the RBAC system is not configured.
    
    Args:
        required_permission: The permission string to check (e.g., 'tools:http_get')
    
    Returns:
        (has_permission, error_message) tuple where:
        - has_permission is True if access is granted
        - error_message is None if granted, or a user-friendly error string if denied
    """
    try:
        from flask import session, has_request_context
        from src.utils.rbac.registry import get_registry
        
        # If we're not in a request context, allow the tool (for testing/CLI usage)
        if not has_request_context():
            logger.debug("No request context, allowing tool access")
            return True, None
        
        # Get user roles from session
        if not session.get('logged_in'):
            logger.warning("User not logged in, denying tool access")
            return False, "You must be logged in to use this feature."
        
        user_roles = session.get('roles', [])
        
        # Check permission using RBAC registry
       
        try:
            
            registry = get_registry()
        except Exception as e:
            logger.warning(f"RBAC registry not available, allowing tool access: {e}")
            return True, None
        try:
            if registry.has_permission(user_roles, required_permission):
                logger.debug(f"User with roles {user_roles} granted permission '{required_permission}'")
                return True, None
            else:
                logger.info(f"User with roles {user_roles} denied permission '{required_permission}'")
                return False, (
                    f"Permission denied: This tool requires '{required_permission}' permission. "
                    f"Your current role(s) ({', '.join(user_roles) if user_roles else 'none'}) "
                    "do not have access to this feature. Please contact an administrator "
                    "if you believe you should have access."
                )
        except Exception as e:
            logger.warning(f"Error checking permission from registry : {e}")
            return False, "An unexpected error occurred while checking permissions with registry. Access denied."
            
    except ImportError as e:
        # Flask not available (e.g., running outside web context)
        logger.debug(f"Flask not available, allowing tool access: {e}")
        return True, None
    except Exception as e:
        logger.error(f"Unexpected error checking tool permission: {e}")
        # Fail closed for unexpected errors to avoid granting access on error
        return False, "An unexpected error occurred while checking permissions. Access denied."


def require_tool_permission(permission: Optional[str]) -> Callable[[F], F]:
    """
    Decorator that enforces RBAC permission check before tool execution.
    
    This decorator wraps a tool function and checks if the current user
    has the required permission before allowing the tool to execute.
    If permission is denied, returns an error message instead of executing the tool.
    
    Args:
        permission: The permission required to use the tool (e.g., Permission.Tools.HTTP_GET).
                   If None, no permission check is performed (allow all).

    Returns:
        A decorator function that wraps the tool with permission checking.

    Example:
        @require_tool_permission(Permission.Tools.HTTP_GET)
        def _http_get_tool(url: str) -> str:
            ...
    
    Note:
        - If permission is None, the decorator is a no-op (returns original function)
        - Permission checks fail open in non-web contexts (CLI, testing)
        - Permission checks fail open if RBAC registry is not configured
    """
    def decorator(func: F) -> F:
        if permission is None:
            # No permission required, return original function
            return func
        
        # Capture permission in closure for type checker (guaranteed non-None here)
        required_perm: str = permission
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            has_perm, error_msg = check_tool_permission(required_perm)
            if not has_perm:
                logger.warning(f"Tool '{func.__name__}' permission denied: {required_perm}")
                return f"Error: {error_msg}"
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def create_abstract_tool(
    *,
    name: str = "abstract_tool",
    description: str = "Abstract base tool.",
) -> Callable:

    @tool(name, description=description)
    def _abstract_tool(query: str) -> str:
        """An abstract tool that does nothing."""
        ...
        return "This is an abstract tool. Please implement specific functionality."
    
    return _abstract_tool
