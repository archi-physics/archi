from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_role_context() -> str:
    """
    Get role context string for the current user if enabled.

    Requires SSO auth with auth_roles configured and pass_descriptions_to_agent: true.
    Returns empty string if conditions not met or user not authenticated.
    """
    try:
        from flask import session, has_request_context
        if not has_request_context():
            return ""
        if not session.get('logged_in'):
            return ""

        from src.utils.rbac.registry import get_registry
        registry = get_registry()

        if not registry.pass_descriptions_to_agent:
            return ""

        roles = session.get('roles', [])
        if not roles:
            return ""

        descriptions = registry.get_role_descriptions(roles)
        if descriptions:
            return f"\n\nUser roles: {descriptions}."
        return ""
    except Exception as e:
        logger.debug(f"Could not get role context: {e}")
        return ""


def read_prompt(prompt_filepath: str) -> str:
    try:
        with open(prompt_filepath, "r") as f:
            raw_prompt = f.read()

        return "\n".join(
            line for line in raw_prompt.split("\n") if not line.lstrip().startswith("#")
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}")

