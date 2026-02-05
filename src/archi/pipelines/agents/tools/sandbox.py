"""
Sandbox code execution tool for agent pipelines.

This module provides the create_sandbox_tool function that creates a LangChain-compatible
tool for executing code in isolated Docker containers.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.utils.logging import get_logger
from src.archi.pipelines.agents.tools.base import require_tool_permission

logger = get_logger(__name__)


# Default permission required to use the sandbox tool
DEFAULT_REQUIRED_PERMISSION = "tools:sandbox"


class SandboxInput(BaseModel):
    """Input schema for the sandbox execution tool."""
    
    code: str = Field(
        description="The code to execute. Should be complete, runnable code."
    )
    language: str = Field(
        default="python",
        description="Programming language: 'python', 'bash', or 'sh'. Default is 'python'."
    )
    image: Optional[str] = Field(
        default=None,
        description="Docker image to use. If not specified, uses the default image. "
                    "Must be in the allowed images list."
    )


def _get_user_role_overrides():
    """
    Get sandbox overrides for the current user's role.
    
    Returns:
        (role_overrides, error_message) tuple
    """
    try:
        from flask import session, has_request_context
        
        if not has_request_context():
            return None, None
        
        if not session.get('logged_in'):
            return None, None
        
        # Get user roles
        user_roles = session.get('roles', [])
        if not user_roles:
            return None, None
        
        # Get role configuration from RBAC registry
        from src.utils.rbac.registry import get_registry
        from src.utils.sandbox.config import get_role_sandbox_overrides
        
        registry = get_registry()
        
        # Find the first role with sandbox overrides
        for role_name in user_roles:
            role_config = registry.get_role_config(role_name)
            if role_config:
                overrides = get_role_sandbox_overrides(role_config)
                if overrides:
                    return overrides, None
        
        return None, None
        
    except ImportError:
        return None, None
    except Exception as e:
        logger.warning(f"Error getting role sandbox overrides: {e}")
        return None, None


def _format_output_for_agent(result) -> str:
    """
    Format sandbox result for agent consumption.
    
    Produces a structured text output that the agent can parse and present to users.
    """
    from src.utils.sandbox.executor import SandboxResult
    
    if not isinstance(result, SandboxResult):
        return str(result)
    
    parts = []
    
    # Handle system errors
    if result.error:
        parts.append(f"**Execution Error**: {result.error}")
        if result.timed_out:
            parts.append("The execution was terminated due to timeout.")
        return "\n".join(parts)
    
    # Exit code
    if result.exit_code != 0:
        parts.append(f"**Exit Code**: {result.exit_code} (non-zero indicates an error in the code)")
    else:
        parts.append("**Exit Code**: 0 (success)")
    
    # Execution time
    parts.append(f"**Execution Time**: {result.execution_time:.2f}s")
    
    # Stdout
    if result.stdout:
        parts.append("\n**Standard Output**:")
        parts.append("```")
        parts.append(result.stdout)
        parts.append("```")
    else:
        parts.append("\n**Standard Output**: (empty)")
    
    # Stderr
    if result.stderr:
        parts.append("\n**Standard Error**:")
        parts.append("```")
        parts.append(result.stderr)
        parts.append("```")
    
    # Truncation warning
    if result.truncated:
        parts.append("\n⚠️ Output was truncated due to size limits.")
    
    # Generated files
    if result.files:
        parts.append(f"\n**Generated Files** ({len(result.files)}):")
        for f in result.files:
            if f.truncated:
                parts.append(f"- {f.filename} ({f.size} bytes) - TRUNCATED/SKIPPED (too large)")
            else:
                # For images, we could potentially render them inline
                # For now, just indicate they're available
                parts.append(f"- {f.filename} ({f.mimetype}, {f.size} bytes)")
                
                # Include small text files inline
                if f.mimetype.startswith("text/") and f.size < 5000:
                    try:
                        import base64
                        content = base64.b64decode(f.content_base64).decode("utf-8")
                        parts.append(f"  ```\n{content}\n  ```")
                    except Exception:
                        pass
    
    return "\n".join(parts)


def create_sandbox_tool(
    *,
    name: str = "execute_code",
    description: Optional[str] = None,
    required_permission: Optional[str] = DEFAULT_REQUIRED_PERMISSION,
) -> Callable:
    """
    Create a LangChain tool that executes code in an isolated sandbox container.
    
    This tool allows agents to run arbitrary code (Python, bash, etc.) in ephemeral
    Docker containers with resource limits and security isolation.
    
    Args:
        name: The name of the tool (used by the LLM when selecting tools).
        description: Human-readable description of what the tool does.
            If None, a default description is used.
        required_permission: The RBAC permission required to use this tool.
            Default is 'tools:sandbox'. Set to None to disable permission checks.
    
    Returns:
        A callable LangChain tool that accepts code and returns execution results.
    
    Example:
        >>> from src.archi.pipelines.agents.tools import create_sandbox_tool
        >>> sandbox_tool = create_sandbox_tool(
        ...     name="run_code",
        ...     description="Execute Python or bash code in a sandbox",
        ... )
        >>> # Add to agent's tool list
        >>> tools = [retriever_tool, sandbox_tool]
    
    Security Notes:
        - Code runs in ephemeral Docker containers that are destroyed after execution
        - Containers have resource limits (CPU, memory, time)
        - Only images from the configured allowlist can be used
        - RBAC permission check is enforced at tool invocation time
        - Containers are isolated from the host and internal services
    
    Output:
        The tool returns a structured text output containing:
        - Exit code
        - Execution time
        - Standard output (stdout)
        - Standard error (stderr)
        - List of generated files (if any)
    """
    tool_description = description or (
        "Execute code in an isolated sandbox container.\n"
        "Input: JSON with 'code' (required), 'language' (optional: python/bash/sh), "
        "'image' (optional: Docker image from allowlist).\n"
        "Output: Execution results including stdout, stderr, exit code, and generated files.\n"
        "\n"
        "Use this tool to:\n"
        "- Run Python scripts for data analysis\n"
        "- Execute shell commands (curl, rucio, etc.)\n"
        "- Process data and generate outputs\n"
        "\n"
        "Generated files should be written to /workspace/output/ to be captured.\n"
        "For plots: plt.savefig('/workspace/output/plot.png')\n"
        "\n"
        "Example input: {\"code\": \"print('Hello')\", \"language\": \"python\"}"
    )
    
    @tool(name, description=tool_description, args_schema=SandboxInput)
    @require_tool_permission(required_permission)
    def _sandbox_tool(code: str, language: str = "python", image: Optional[str] = None) -> str:
        """Execute code in an isolated sandbox container."""
        
        # Import here to avoid circular imports and allow lazy loading
        from src.utils.sandbox import (
            SandboxExecutor,
            get_sandbox_config,
            resolve_effective_config,
        )
        
        # Load base config
        base_config = get_sandbox_config()
        
        if not base_config.enabled:
            logger.warning("Sandbox tool invoked but sandbox is not enabled")
            return "Error: Sandbox execution is not enabled for this deployment."
        
        # Get role overrides and resolve effective config
        role_overrides, _ = _get_user_role_overrides()
        effective_config = resolve_effective_config(base_config, role_overrides)
        
        # Validate image against effective allowlist
        target_image = image or effective_config.default_image
        if not effective_config.is_image_allowed(target_image):
            allowed = ", ".join(effective_config.image_allowlist)
            logger.warning(f"User requested disallowed image '{target_image}'")
            return (
                f"Error: Image '{target_image}' is not allowed for your role.\n"
                f"Allowed images: {allowed}"
            )
        
        # Create executor and run
        try:
            executor = SandboxExecutor(config=effective_config)
            
            logger.info(
                f"Executing {language} code in sandbox (image={target_image}, "
                f"timeout={effective_config.timeout}s)"
            )
            
            result = executor.execute(
                code=code,
                language=language,
                image=target_image,
                timeout=effective_config.timeout,
                limits=effective_config.resource_limits,
            )
            
            # Format output for agent
            return _format_output_for_agent(result)
            
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}", exc_info=True)
            return f"Error: Sandbox execution failed - {str(e)}"
    
    return _sandbox_tool


def create_sandbox_tool_with_files(
    *,
    name: str = "execute_code_with_files", 
    description: Optional[str] = None,
    required_permission: Optional[str] = DEFAULT_REQUIRED_PERMISSION,
    return_files: bool = True,
) -> Callable:
    """
    Create a sandbox tool that returns structured output including file data.
    
    This variant is useful when you need programmatic access to generated files
    (e.g., for rendering images in chat).
    
    Args:
        name: The name of the tool.
        description: Tool description.
        required_permission: RBAC permission required.
        return_files: Whether to include base64-encoded files in output.
    
    Returns:
        A callable tool that returns JSON-serializable output.
    """
    tool_description = description or (
        "Execute code in a sandbox and return structured results including generated files.\n"
        "Returns JSON with stdout, stderr, exit_code, execution_time, and files array."
    )
    
    @tool(name, description=tool_description, args_schema=SandboxInput)
    @require_tool_permission(required_permission)
    def _sandbox_tool_with_files(
        code: str, 
        language: str = "python", 
        image: Optional[str] = None
    ) -> str:
        """Execute code and return structured output with files."""
        
        from src.utils.sandbox import (
            SandboxExecutor,
            get_sandbox_config,
            resolve_effective_config,
        )
        
        base_config = get_sandbox_config()
        
        if not base_config.enabled:
            return json.dumps({"error": "Sandbox execution is not enabled"})
        
        role_overrides, _ = _get_user_role_overrides()
        effective_config = resolve_effective_config(base_config, role_overrides)
        
        target_image = image or effective_config.default_image
        if not effective_config.is_image_allowed(target_image):
            return json.dumps({
                "error": f"Image '{target_image}' is not allowed",
                "allowed_images": effective_config.image_allowlist,
            })
        
        try:
            executor = SandboxExecutor(config=effective_config)
            result = executor.execute(
                code=code,
                language=language,
                image=target_image,
                timeout=effective_config.timeout,
                limits=effective_config.resource_limits,
            )
            
            output = result.to_dict()
            
            # Optionally strip file content to reduce size
            if not return_files:
                for f in output.get("files", []):
                    f["content_base64"] = None
            
            return json.dumps(output)
            
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}", exc_info=True)
            return json.dumps({"error": str(e)})
    
    return _sandbox_tool_with_files
