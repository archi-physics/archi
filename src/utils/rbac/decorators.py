"""
RBAC Decorators - Route protection decorators for Flask endpoints

This module provides decorators to protect Flask routes with permission requirements.
Decorators handle authentication checks, permission validation, and audit logging.
"""

from functools import wraps
from typing import Callable, List, Optional, Union
from flask import session, jsonify, redirect, url_for, request, g

from src.utils.logging import get_logger
from src.utils.rbac.registry import get_registry
from src.utils.rbac.audit import log_permission_check

logger = get_logger(__name__)


class PermissionDeniedError(Exception):
    """Raised when a permission check fails."""
    def __init__(self, message: str, required_permission: str, user_roles: List[str]):
        super().__init__(message)
        self.required_permission = required_permission
        self.user_roles = user_roles


def get_current_user_roles() -> List[str]:
    """
    Get roles for the current user from session.
    
    Returns:
        List of role names, or empty list if not authenticated
    """
    if not session.get('logged_in'):
        return []
    
    return session.get('roles', [])


def is_authenticated() -> bool:
    """
    Check if current user is authenticated.
    
    Returns:
        True if user has an active session
    """
    return session.get('logged_in', False)


def require_authenticated(f: Callable) -> Callable:
    """
    Decorator that requires user to be authenticated.
    
    Does NOT check for specific permissions, only that user is logged in.
    Use this for routes that should be accessible to all authenticated users.
    
    Usage:
        @app.route('/profile')
        @require_authenticated
        def profile():
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            # Log the denial
            log_permission_check(
                user='anonymous',
                permission='authenticated',
                granted=False,
                endpoint=request.endpoint,
                roles=[]
            )
            
            # Check if this is an API request
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({
                    'error': 'Authentication required',
                    'message': 'Please log in to access this resource',
                    'status': 401
                }), 401
            
            # Redirect to login for browser requests
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_permission(permission: Union[str, List[str]]) -> Callable:
    """
    Decorator that requires specific permission(s) to access a route.
    
    If a list of permissions is provided, user must have ALL of them.
    For "any of" logic, use require_any_permission instead.
    
    Usage:
        @app.route('/api/upload')
        @require_permission('upload:documents')
        def upload():
            ...
        
        @app.route('/api/admin/config')
        @require_permission(['config:view', 'config:modify'])
        def admin_config():
            ...
    """
    # Normalize to list
    required_permissions = [permission] if isinstance(permission, str) else permission
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check authentication
            if not is_authenticated():
                log_permission_check(
                    user='anonymous',
                    permission=','.join(required_permissions),
                    granted=False,
                    endpoint=request.endpoint,
                    roles=[]
                )
                
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({
                        'error': 'Authentication required',
                        'message': 'Please log in to access this resource',
                        'status': 401
                    }), 401
                
                return redirect(url_for('login'))
            
            # Get user roles
            user_roles = get_current_user_roles()
            user_email = session.get('user', {}).get('email', 'unknown')
            
            # Get registry and check permissions
            registry = get_registry()
            
            # Check each required permission
            missing_permissions = []
            for perm in required_permissions:
                if not registry.has_permission(user_roles, perm):
                    missing_permissions.append(perm)
            
            if missing_permissions:
                # Log the denial
                log_permission_check(
                    user=user_email,
                    permission=','.join(required_permissions),
                    granted=False,
                    endpoint=request.endpoint,
                    roles=user_roles,
                    missing=missing_permissions
                )
                
                # Get roles that would grant the permission for helpful error message
                roles_with_permission = set()
                for perm in missing_permissions:
                    roles_with_permission.update(registry.get_roles_with_permission(perm))
                
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required_permissions': missing_permissions,
                        'user_roles': user_roles,
                        'roles_with_permission': list(roles_with_permission),
                        'message': f"You need one of these roles to access this feature: {', '.join(roles_with_permission)}",
                        'status': 403
                    }), 403
                
                # For browser requests, return 403 page
                from flask import render_template
                return render_template('error.html',
                    error_code=403,
                    error_title='Permission Denied',
                    error_message=f"You don't have permission to access this feature.",
                    required_roles=list(roles_with_permission)
                ), 403
            
            # Log successful access
            log_permission_check(
                user=user_email,
                permission=','.join(required_permissions),
                granted=True,
                endpoint=request.endpoint,
                roles=user_roles
            )
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def require_any_permission(permissions: List[str]) -> Callable:
    """
    Decorator that requires ANY ONE of the specified permissions.
    
    User only needs one of the listed permissions to access the route.
    
    Usage:
        @app.route('/api/settings')
        @require_any_permission(['config:view', 'config:modify', 'admin:system'])
        def settings():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check authentication
            if not is_authenticated():
                log_permission_check(
                    user='anonymous',
                    permission=f"any({','.join(permissions)})",
                    granted=False,
                    endpoint=request.endpoint,
                    roles=[]
                )
                
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({
                        'error': 'Authentication required',
                        'message': 'Please log in to access this resource',
                        'status': 401
                    }), 401
                
                return redirect(url_for('login'))
            
            # Get user roles
            user_roles = get_current_user_roles()
            user_email = session.get('user', {}).get('email', 'unknown')
            
            # Get registry and check permissions
            registry = get_registry()
            
            # Check if user has ANY of the permissions
            has_any = False
            for perm in permissions:
                if registry.has_permission(user_roles, perm):
                    has_any = True
                    break
            
            if not has_any:
                # Log the denial
                log_permission_check(
                    user=user_email,
                    permission=f"any({','.join(permissions)})",
                    granted=False,
                    endpoint=request.endpoint,
                    roles=user_roles
                )
                
                # Get roles that would grant any of the permissions
                roles_with_permission = set()
                for perm in permissions:
                    roles_with_permission.update(registry.get_roles_with_permission(perm))
                
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required_permissions': permissions,
                        'user_roles': user_roles,
                        'roles_with_permission': list(roles_with_permission),
                        'message': f"You need one of these roles: {', '.join(roles_with_permission)}",
                        'status': 403
                    }), 403
                
                from flask import render_template
                return render_template('error.html',
                    error_code=403,
                    error_title='Permission Denied',
                    error_message=f"You don't have permission to access this feature.",
                    required_roles=list(roles_with_permission)
                ), 403
            
            # Log successful access
            log_permission_check(
                user=user_email,
                permission=f"any({','.join(permissions)})",
                granted=True,
                endpoint=request.endpoint,
                roles=user_roles
            )
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def check_sso_required() -> Callable:
    """
    Decorator to enforce SSO authentication when configured.
    
    When SSO is enabled and allow_anonymous is False, this decorator
    redirects unauthenticated users to SSO login.
    
    Usage:
        @app.route('/')
        @check_sso_required()
        def landing():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            registry = get_registry()
            
            # If anonymous access is not allowed and user is not authenticated
            if not registry.allow_anonymous and not is_authenticated():
                log_permission_check(
                    user='anonymous',
                    permission='sso_required',
                    granted=False,
                    endpoint=request.endpoint,
                    roles=[]
                )
                
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({
                        'error': 'Authentication required',
                        'message': 'SSO authentication is required for this application',
                        'login_url': url_for('login', method='sso'),
                        'status': 401
                    }), 401
                
                # Redirect to SSO login
                return redirect(url_for('login', method='sso'))
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator
