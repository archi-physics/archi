"""
JWT Parser - Extract roles from SSO provider JWT tokens

This module handles parsing JWT tokens from SSO providers to extract
user roles from the resource_access claim.
"""

from typing import Any, Dict, List, Optional
import jwt

from src.utils.logging import get_logger
from src.utils.rbac.registry import get_registry
from src.utils.rbac.audit import log_role_assignment

logger = get_logger(__name__)


def extract_roles_from_token(
    token: Dict[str, Any],
    app_name: Optional[str] = None
) -> List[str]:
    """
    Extract roles from a JWT token's resource_access claim.
    
    The expected token structure from Keycloak/CERN SSO:
    {
        "resource_access": {
            "<app_name>": {
                "roles": ["role1", "role2", ...]
            }
        }
    }
    
    Args:
        token: Decoded JWT token dictionary (or the raw token response from OAuth)
        app_name: Application name to look for in resource_access.
                  If not provided, uses the configured app_name from registry.
    
    Returns:
        List of role strings extracted from the token
    """
    if app_name is None:
        registry = get_registry()
        app_name = registry.app_name
    
    try:
        # Handle both raw OAuth token response and decoded JWT
        # OAuth libraries may wrap the token differently
        
        # Check if this is an OAuth token response with nested tokens
        access_token_data = token
        
        # If token has 'access_token' key, it might be encoded
        if 'access_token' in token and isinstance(token['access_token'], str):
            # Try to decode the access token (without verification for role extraction)
            try:
                access_token_data = jwt.decode(
                    token['access_token'],
                    options={"verify_signature": False}  # We trust the OAuth library verified it
                )
            except jwt.DecodeError:
                logger.warning("Could not decode access_token, using token as-is")
                access_token_data = token
        
        # Also check id_token which may contain roles
        id_token_data = {}
        if 'id_token' in token and isinstance(token['id_token'], str):
            try:
                id_token_data = jwt.decode(
                    token['id_token'],
                    options={"verify_signature": False}
                )
            except jwt.DecodeError:
                pass
        
        # Look for resource_access in access_token first, then id_token
        resource_access = access_token_data.get('resource_access', {})
        if not resource_access and id_token_data:
            resource_access = id_token_data.get('resource_access', {})
        
        # Also check userinfo if present
        if not resource_access:
            userinfo = token.get('userinfo', {})
            resource_access = userinfo.get('resource_access', {})
        
        if not resource_access:
            logger.warning(f"No resource_access claim found in token")
            logger.debug(f"Token keys: {list(access_token_data.keys())}")
            return []
        
        # Get roles for our application
        app_access = resource_access.get(app_name, {})
        if not app_access:
            logger.warning(
                f"No roles found for app '{app_name}' in resource_access. "
                f"Available apps: {list(resource_access.keys())}"
            )
            return []
        
        roles = app_access.get('roles', [])
        
        if not isinstance(roles, list):
            logger.warning(f"roles claim is not a list: {type(roles)}")
            return []
        
        logger.info(f"Extracted roles for app '{app_name}': {roles}")
        return roles
        
    except Exception as e:
        logger.error(f"Error extracting roles from token: {e}")
        return []


def get_user_roles(
    token: Dict[str, Any],
    user_email: str,
    app_name: Optional[str] = None
) -> List[str]:
    """
    Get validated user roles from JWT token, with default role fallback.
    
    This is the main entry point for role extraction. It:
    1. Extracts roles from the JWT token
    2. Filters to only configured/valid roles
    3. Assigns default role if no valid roles found
    4. Logs the role assignment
    
    Args:
        token: JWT token dictionary from OAuth callback
        user_email: User's email for logging
        app_name: Optional app name override
        
    Returns:
        List of validated role strings (never empty - at minimum returns default role)
    """
    registry = get_registry()
    
    # Extract raw roles from token
    raw_roles = extract_roles_from_token(token, app_name)
    
    # Filter to only valid/configured roles
    valid_roles = registry.filter_valid_roles(raw_roles)
    
    if valid_roles:
        # User has at least one valid role
        log_role_assignment(
            user=user_email,
            roles=valid_roles,
            source='jwt',
            is_default=False
        )
        return valid_roles
    else:
        # No valid roles - assign default
        return assign_default_role(user_email, raw_roles)


def assign_default_role(user_email: str, original_roles: List[str] = None) -> List[str]:
    """
    Assign the default role to a user who has no configured roles.
    
    Args:
        user_email: User's email for logging
        original_roles: Original roles from JWT (for logging, may be unmapped)
        
    Returns:
        List containing only the default role
    """
    registry = get_registry()
    default_role = registry.default_role
    
    log_role_assignment(
        user=user_email,
        roles=[default_role],
        source='default',
        is_default=True
    )
    
    if original_roles:
        logger.info(
            f"User {user_email} has no configured roles. "
            f"Original JWT roles {original_roles} are not mapped. "
            f"Assigning default role: {default_role}"
        )
    else:
        logger.info(
            f"User {user_email} has no roles in JWT. "
            f"Assigning default role: {default_role}"
        )
    
    return [default_role]


def decode_jwt_claims(token_string: str, verify: bool = False) -> Dict[str, Any]:
    """
    Decode a JWT token string to extract claims.
    
    Args:
        token_string: Encoded JWT token
        verify: Whether to verify signature (requires public key)
        
    Returns:
        Decoded token claims dictionary
    """
    try:
        return jwt.decode(
            token_string,
            options={"verify_signature": verify}
        )
    except jwt.DecodeError as e:
        logger.error(f"Failed to decode JWT: {e}")
        return {}
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        return {}
