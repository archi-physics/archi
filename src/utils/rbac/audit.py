"""
RBAC Audit Logging - Security event logging for access control

This module provides audit logging for all permission checks,
supporting security analysis and compliance requirements.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional

from src.utils.logging import get_logger

# Dedicated audit logger
audit_logger = get_logger('rbac.audit')


def log_permission_check(
    user: str,
    permission: str,
    granted: bool,
    endpoint: str,
    roles: List[str],
    missing: Optional[List[str]] = None,
    extra: Optional[dict] = None
) -> None:
    """
    Log a permission check event for audit trail.
    
    Args:
        user: Username or email of the user (or 'anonymous')
        permission: Permission(s) being checked
        granted: Whether access was granted
        endpoint: Flask endpoint name
        roles: User's current roles
        missing: Permissions that were missing (if denied)
        extra: Additional context information
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = 'GRANTED' if granted else 'DENIED'
    
    # Structured log entry
    log_entry = {
        'timestamp': timestamp,
        'user': user,
        'permission': permission,
        'result': result,
        'endpoint': endpoint,
        'roles': roles,
    }
    
    if missing:
        log_entry['missing_permissions'] = missing
    
    if extra:
        log_entry.update(extra)
    
    # Log level based on result
    log_message = f"{user} | {permission} | {result} | {endpoint} | roles: {roles}"
    
    if granted:
        audit_logger.debug(log_message)
    else:
        audit_logger.warning(log_message)
        # Also log structured JSON for easier parsing
        audit_logger.info(f"AUDIT: {json.dumps(log_entry)}")


def log_role_assignment(
    user: str,
    roles: List[str],
    source: str,
    is_default: bool = False
) -> None:
    """
    Log a role assignment event.
    
    Args:
        user: Username or email
        roles: Roles assigned to user
        source: Source of roles (e.g., 'jwt', 'default')
        is_default: Whether default role was assigned
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'event': 'role_assignment',
        'user': user,
        'roles': roles,
        'source': source,
        'is_default': is_default,
    }
    
    if is_default:
        audit_logger.warning(
            f"Default role assigned to {user}: {roles} (no JWT roles found)"
        )
    else:
        audit_logger.info(f"Roles assigned to {user}: {roles} (source: {source})")
    
    audit_logger.debug(f"AUDIT: {json.dumps(log_entry)}")


def log_authentication_event(
    user: str,
    event_type: str,
    success: bool,
    method: str,
    details: Optional[str] = None
) -> None:
    """
    Log an authentication event.
    
    Args:
        user: Username or email (or 'unknown')
        event_type: Type of event ('login', 'logout', 'token_refresh')
        success: Whether the event succeeded
        method: Authentication method ('sso', 'basic')
        details: Additional details or error message
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = 'SUCCESS' if success else 'FAILURE'
    
    log_entry = {
        'timestamp': timestamp,
        'event': event_type,
        'user': user,
        'result': result,
        'method': method,
    }
    
    if details:
        log_entry['details'] = details
    
    log_message = f"AUTH | {event_type} | {user} | {result} | method: {method}"
    if details:
        log_message += f" | {details}"
    
    if success:
        audit_logger.info(log_message)
    else:
        audit_logger.warning(log_message)
    
    audit_logger.debug(f"AUDIT: {json.dumps(log_entry)}")
