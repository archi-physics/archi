"""
Sandbox code execution tool for agent pipelines.

This module provides the create_sandbox_tool function that creates a LangChain-compatible
tool for executing code in isolated Docker containers.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.utils.logging import get_logger
from src.archi.pipelines.agents.tools.base import require_tool_permission

logger = get_logger(__name__)


# Default permission required to use the sandbox tool
DEFAULT_REQUIRED_PERMISSION = "tools:sandbox"

# ---------------------------------------------------------------------------
# Per-request context for sandbox artifact persistence.
#
# We use a module-level dictionary keyed by trace_id instead of thread-local
# or contextvars, because LangChain/LangGraph may execute tools in different
# threads. The trace_id is set as an environment variable which IS inherited
# by child threads.
# ---------------------------------------------------------------------------
_sandbox_contexts: Dict[str, Dict] = {}  # trace_id -> {data_path, artifacts}
_sandbox_lock = threading.Lock()

# Environment variable name for passing trace_id to tools
_TRACE_ID_ENV = "_ARCHI_SANDBOX_TRACE_ID"

# Environment variable for conversation_id (needed for approval requests)
_CONVERSATION_ID_ENV = "_ARCHI_SANDBOX_CONVERSATION_ID"

# Environment variable for session-level approval mode override
_APPROVAL_MODE_ENV = "_ARCHI_SANDBOX_APPROVAL_MODE"

# Safe filename pattern
_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")

# Type for approval request callbacks
ApprovalRequestCallback = Callable[[Dict], None]


def set_sandbox_context(
    trace_id: str,
    data_path: str,
    conversation_id: Optional[int] = None,
    approval_callback: Optional[ApprovalRequestCallback] = None,
    approval_mode_override: Optional[str] = None,
) -> None:
    """
    Set the sandbox context for the current request (call before streaming).
    
    This stores context in a module-level dict and sets an environment variable
    so that the trace_id can be retrieved from any thread.
    
    Args:
        trace_id: Unique identifier for this request.
        data_path: Path where artifacts should be stored.
        conversation_id: The conversation ID for approval requests.
        approval_callback: Optional callback to notify about approval requests.
        approval_mode_override: Optional session-level approval mode ("auto" or "manual").
    """
    with _sandbox_lock:
        _sandbox_contexts[trace_id] = {
            "data_path": data_path,
            "artifacts": [],
            "conversation_id": conversation_id,
            "approval_callback": approval_callback,
            "approval_mode_override": approval_mode_override,
        }
    # Set env vars so child threads can find the trace_id and conversation_id
    os.environ[_TRACE_ID_ENV] = trace_id
    if conversation_id is not None:
        os.environ[_CONVERSATION_ID_ENV] = str(conversation_id)
    if approval_mode_override is not None:
        os.environ[_APPROVAL_MODE_ENV] = approval_mode_override
    logger.debug(
        "Sandbox context set: trace_id=%s, data_path=%s, conversation_id=%s, approval_mode=%s",
        trace_id, data_path, conversation_id, approval_mode_override
    )


def get_sandbox_artifacts() -> List[Dict]:
    """Return artifact metadata collected during the current request."""
    trace_id = os.environ.get(_TRACE_ID_ENV)
    if not trace_id:
        return []
    with _sandbox_lock:
        ctx = _sandbox_contexts.get(trace_id)
        return ctx["artifacts"] if ctx else []


def clear_sandbox_context() -> None:
    """Clear sandbox context (call after consuming artifacts)."""
    trace_id = os.environ.pop(_TRACE_ID_ENV, None)
    os.environ.pop(_CONVERSATION_ID_ENV, None)
    os.environ.pop(_APPROVAL_MODE_ENV, None)
    if trace_id:
        with _sandbox_lock:
            _sandbox_contexts.pop(trace_id, None)
        logger.debug("Sandbox context cleared: trace_id=%s", trace_id)


def _get_sandbox_context() -> tuple:
    """Get current trace_id, data_path, or (None, None) if not set."""
    trace_id = os.environ.get(_TRACE_ID_ENV)
    if not trace_id:
        return None, None
    with _sandbox_lock:
        ctx = _sandbox_contexts.get(trace_id)
        if ctx:
            return trace_id, ctx["data_path"]
    return None, None


def _get_full_sandbox_context() -> Dict:
    """
    Get the full sandbox context including conversation_id and approval_callback.
    
    Returns:
        Dict with trace_id, data_path, conversation_id, approval_callback, approval_mode_override
        or empty dict if not set.
    """
    trace_id = os.environ.get(_TRACE_ID_ENV)
    if not trace_id:
        return {}
    with _sandbox_lock:
        ctx = _sandbox_contexts.get(trace_id)
        if ctx:
            return {
                "trace_id": trace_id,
                "data_path": ctx.get("data_path"),
                "conversation_id": ctx.get("conversation_id"),
                "approval_callback": ctx.get("approval_callback"),
                "approval_mode_override": ctx.get("approval_mode_override"),
            }
    return {}


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage."""
    name = os.path.basename(filename).strip()
    if not name or not _SAFE_FILENAME_RE.match(name):
        # Generate a safe fallback name
        ext = Path(filename).suffix if filename else ".bin"
        name = f"output{ext}"
    return name


def _persist_sandbox_file(filename: str, mimetype: str, content_base64: str) -> Optional[Dict]:
    """
    Persist a sandbox-generated file directly to disk.
    
    Returns artifact metadata dict with url, or None if context not available.
    """
    trace_id, data_path = _get_sandbox_context()
    
    if not trace_id or not data_path:
        logger.warning("Sandbox context not set - cannot persist file %s", filename)
        return None
    
    # Validate trace_id format (UUID)
    if not re.fullmatch(r"[0-9a-f\-]{36}", trace_id):
        logger.error("Invalid trace_id format: %s", trace_id)
        return None
    
    try:
        # Create artifact directory
        artifact_dir = Path(data_path) / "sandbox_artifacts" / trace_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize and deduplicate filename
        safe_name = _sanitize_filename(filename)
        dest = artifact_dir / safe_name
        counter = 1
        while dest.exists():
            stem, ext = os.path.splitext(safe_name)
            dest = artifact_dir / f"{stem}_{counter}{ext}"
            counter += 1
        
        # Decode and write
        raw = base64.b64decode(content_base64)
        dest.write_bytes(raw)
        
        # Build URL
        url = f"/api/sandbox-artifacts/{trace_id}/{dest.name}"
        
        artifact = {
            "filename": dest.name,
            "mimetype": mimetype,
            "url": url,
            "size": len(raw),
        }
        
        # Store in module-level dict for later retrieval
        with _sandbox_lock:
            ctx = _sandbox_contexts.get(trace_id)
            if ctx:
                ctx["artifacts"].append(artifact)
        
        logger.info(
            "Persisted sandbox artifact: %s (%s, %d bytes) -> %s",
            filename, mimetype, len(raw), url,
        )
        return artifact
        
    except Exception as e:
        logger.error("Failed to persist sandbox file %s: %s", filename, e, exc_info=True)
        return None


def _get_effective_approval_mode(config_approval_mode: "ApprovalMode") -> "ApprovalMode":
    """
    Get the effective approval mode, considering session-level override.
    
    Priority:
    1. Session-level override (set by user per session/query)
    2. Config-level setting (from deployment config)
    
    Args:
        config_approval_mode: The approval mode from the effective config.
        
    Returns:
        The approval mode to use for this execution.
    """
    from src.utils.sandbox.config import ApprovalMode
    
    ctx = _get_full_sandbox_context()
    override = ctx.get("approval_mode_override")
    
    if override:
        # Parse the override string to ApprovalMode
        override_lower = override.lower()
        if override_lower == "auto":
            logger.debug("Using session-level approval_mode override: auto")
            return ApprovalMode.AUTO
        elif override_lower == "manual":
            logger.debug("Using session-level approval_mode override: manual")
            return ApprovalMode.MANUAL
        else:
            logger.warning(
                "Invalid approval_mode_override '%s', falling back to config: %s",
                override, config_approval_mode.value
            )
    
    return config_approval_mode


def _request_approval_and_wait(
    code: str,
    language: str,
    image: str,
    tool_call_id: str,
    timeout_seconds: float = 300.0,
) -> tuple[bool, Optional[str]]:
    """
    Request user approval for sandbox execution and wait for response.
    
    Args:
        code: The code to execute.
        language: Programming language.
        image: Docker image.
        tool_call_id: The tool call ID.
        timeout_seconds: How long to wait for approval.
    
    Returns:
        Tuple of (approved: bool, error_message: Optional[str])
    """
    from src.utils.sandbox.approval import (
        ApprovalStatus,
        create_approval_request,
        wait_for_approval,
    )
    
    ctx = _get_full_sandbox_context()
    trace_id = ctx.get("trace_id")
    conversation_id = ctx.get("conversation_id")
    approval_callback = ctx.get("approval_callback")
    
    logger.debug(
        "Approval context: trace_id=%s, conversation_id=%s, has_callback=%s",
        trace_id, conversation_id, approval_callback is not None
    )
    
    if not trace_id or conversation_id is None:
        logger.warning(
            "Cannot request approval: missing context (trace_id=%s, conversation_id=%s)",
            trace_id, conversation_id
        )
        # Fall back to auto-approve if context is missing
        return True, None
    
    # Create the approval request
    request = create_approval_request(
        trace_id=trace_id,
        conversation_id=conversation_id,
        code=code,
        language=language,
        image=image,
        tool_call_id=tool_call_id,
        timeout_seconds=timeout_seconds,
    )
    
    # Notify via callback if available (so the streaming layer can emit an event)
    if approval_callback:
        try:
            logger.info("Invoking approval callback for request: %s", request.approval_id)
            approval_callback(request.to_dict())
            logger.info("Approval callback completed successfully")
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
    else:
        logger.warning("No approval_callback available - frontend will not receive approval_request event")
    
    # Wait for approval
    try:
        resolved = wait_for_approval(
            request.approval_id,
            timeout=timeout_seconds,
            poll_interval=0.5,
        )
        
        if resolved.status == ApprovalStatus.APPROVED:
            logger.info("Sandbox execution approved: %s", request.approval_id)
            return True, None
        elif resolved.status == ApprovalStatus.REJECTED:
            logger.info("Sandbox execution rejected: %s", request.approval_id)
            return False, "Execution rejected by user"
        elif resolved.status == ApprovalStatus.EXPIRED:
            logger.info("Sandbox approval expired: %s", request.approval_id)
            return False, "Approval request timed out"
        elif resolved.status == ApprovalStatus.CANCELLED:
            logger.info("Sandbox approval cancelled: %s", request.approval_id)
            return False, "Approval request cancelled"
        else:
            return False, f"Unknown approval status: {resolved.status}"
            
    except Exception as e:
        logger.error("Error waiting for approval: %s", e, exc_info=True)
        return False, f"Error waiting for approval: {e}"


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
            role_config = registry.get_role_info(role_name)
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
    
    # Stdout - filter out container paths to prevent agent from echoing them
    if result.stdout:
        # Remove lines containing /workspace paths
        filtered_lines = []
        for line in result.stdout.split('\n'):
            if '/workspace' not in line.lower():
                filtered_lines.append(line)
        filtered_stdout = '\n'.join(filtered_lines).strip()
        
        if filtered_stdout:
            parts.append("\n**Standard Output**:")
            parts.append("```")
            parts.append(filtered_stdout)
            parts.append("```")
        else:
            parts.append("\n**Standard Output**: (output contained only file path info, omitted)")
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
    
    # Generated files - keep output minimal to avoid agent echoing details
    if result.files:
        logger.info("Formatting %d output file(s) for agent", len(result.files))
        image_count = 0
        other_count = 0
        for f in result.files:
            logger.info(
                "Processing file: %s, mimetype=%s, size=%d, truncated=%s, has_content=%s",
                f.filename, f.mimetype, f.size, f.truncated, bool(f.content_base64),
            )
            if f.truncated:
                other_count += 1  # Skip truncated files
            elif f.content_base64:
                # Persist file directly to disk (images and other files)
                artifact = _persist_sandbox_file(f.filename, f.mimetype, f.content_base64)
                if artifact:
                    if f.mimetype.startswith("image/"):
                        image_count += 1
                    else:
                        other_count += 1
                        # Include small text files inline for agent context
                        if f.mimetype.startswith("text/") and f.size < 5000:
                            try:
                                content = base64.b64decode(f.content_base64).decode("utf-8")
                                parts.append(f"\n**Generated text file:**\n```\n{content}\n```")
                            except Exception:
                                pass
                else:
                    other_count += 1
            else:
                other_count += 1

        # Summarize what was generated without exposing filenames
        if image_count:
            parts.append(
                f"\n**Generated output:** {image_count} image(s) will be displayed to the user "
                f"automatically below your response. Do NOT mention file paths or filenames."
            )
        if other_count and not image_count:
            parts.append(f"\n**Generated output:** {other_count} file(s) saved.")

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
        - Summary of generated files (if any)
    """
    tool_description = description or (
        "Execute code in an isolated sandbox container.\n"
        "Input: JSON with 'code' (required), 'language' (optional: python/bash/sh), "
        "'image' (optional: Docker image from allowlist).\n"
        "Output: Execution results including stdout, stderr, and exit code.\n"
        "\n"
        "Use this tool to:\n"
        "- Run Python scripts for data analysis and plotting\n"
        "- Execute shell commands\n"
        "- Process data and generate outputs\n"
        "\n"
        "For plots, save to /workspace/output/. The directories are pre-created.\n"
        "Example: plt.savefig('/workspace/output/plot.png')\n"
        "\n"
        "CRITICAL RULES FOR YOUR RESPONSE TO THE USER:\n"
        "1. NEVER mention /workspace/, /workspace/output/, or any container paths\n"
        "2. NEVER include filenames like 'plot.png' or 'File: xyz.png' in your response\n"
        "3. Images are displayed AUTOMATICALLY - just describe what you plotted\n"
        "4. Say things like 'Here is the plot' or 'The chart below shows...'\n"
        "5. DO NOT echo the stdout if it contains paths - summarize the results instead\n"
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
        from src.utils.sandbox.config import ApprovalMode
        
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
        
        # Check approval mode and request approval if needed
        # Session-level override takes priority over config
        effective_approval_mode = _get_effective_approval_mode(effective_config.approval_mode)
        if effective_approval_mode == ApprovalMode.MANUAL:
            # Generate a tool_call_id for tracking
            import uuid
            tool_call_id = str(uuid.uuid4())
            
            approved, error_msg = _request_approval_and_wait(
                code=code,
                language=language,
                image=target_image,
                tool_call_id=tool_call_id,
                timeout_seconds=effective_config.timeout * 10,  # Allow longer for approval
            )
            
            if not approved:
                logger.info("Sandbox execution not approved: %s", error_msg)
                return f"Execution cancelled: {error_msg or 'User did not approve'}"
        
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
        from src.utils.sandbox.config import ApprovalMode
        
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
        
        # Check approval mode and request approval if needed
        # Session-level override takes priority over config
        effective_approval_mode = _get_effective_approval_mode(effective_config.approval_mode)
        if effective_approval_mode == ApprovalMode.MANUAL:
            import uuid
            tool_call_id = str(uuid.uuid4())
            
            approved, error_msg = _request_approval_and_wait(
                code=code,
                language=language,
                image=target_image,
                tool_call_id=tool_call_id,
                timeout_seconds=effective_config.timeout * 10,
            )
            
            if not approved:
                return json.dumps({
                    "error": f"Execution cancelled: {error_msg or 'User did not approve'}",
                    "approval_rejected": True,
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
