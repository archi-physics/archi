"""HTTP GET request tool for fetching live data from URLs."""

from __future__ import annotations

from typing import Callable, Optional
from urllib.parse import urlparse, urlunparse

import requests
from langchain.tools import tool

from src.utils.logging import get_logger
from src.archi.pipelines.agents.tools.base import require_tool_permission

logger = get_logger(__name__)


# Default permission required to use the HTTP GET tool
DEFAULT_REQUIRED_PERMISSION = "tools:http_get"


def _validate_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate that the URL is well-formed and uses HTTP/HTTPS.
    
    Returns:
        (is_valid, error_message) tuple
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are supported."
        if not parsed.netloc:
            return False, "Invalid URL: missing hostname."
        return True, None
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"


def _sanitize_url_for_error(url: str) -> str:
    """
    Remove credentials from URL for error messages.
    
    Example: http://user:pass@example.com -> http://***:***@example.com
    """
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            sanitized_netloc = f"***:***@{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")
            return urlunparse((
                parsed.scheme,
                sanitized_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ))
        return url
    except Exception:
        return "***"


def create_http_get_tool(
    *,
    name: str = "fetch_url",
    description: Optional[str] = None,
    timeout: float = 10.0,
    max_response_chars: int = 40000,
    required_permission: Optional[str] = DEFAULT_REQUIRED_PERMISSION,
) -> Callable[[str], str]:
    """
    Create a LangChain tool that makes HTTP GET requests to fetch live data from URLs.
    
    This tool allows agents to retrieve real-time information from web endpoints,
    APIs, or documentation URLs. Only GET requests are supported for security reasons.
    
    Args:
        name: The name of the tool (used by the LLM when selecting tools).
        description: Human-readable description of what the tool does. 
            If None, a default description is used.
        timeout: Maximum time in seconds to wait for a response. Default is 10 seconds.
        max_response_chars: Maximum number of characters to return from the response body.
            Responses longer than this are truncated with a "[truncated]" indicator.
            Default is 40000 characters.
        required_permission: The RBAC permission required to use this tool.
            Default is 'tools:http_get'. Set to None to disable permission checks.
    
    Returns:
        A callable LangChain tool that accepts a URL string and returns either:
        - The response body text (truncated if needed)
        - An error message describing what went wrong
    
    Example:
        >>> from src.archi.pipelines.agents.tools import create_http_get_tool
        >>> http_tool = create_http_get_tool(
        ...     name="fetch_endpoint",
        ...     description="Fetch data from a REST API endpoint",
        ...     timeout=15.0,
        ...     max_response_chars=60000,
        ... )
        >>> # Add to agent's tool list
        >>> tools = [retriever_tool, file_search_tool, http_tool]
    
    Security Notes:
        - Only HTTP and HTTPS URLs are accepted
        - Credentials in URLs are sanitized in error messages
        - No authentication/authorization is built-in (use with public endpoints)
        - Response size is limited to prevent context window overflow
        - Timeouts prevent hanging on slow/unresponsive endpoints
        - RBAC permission check is enforced at tool invocation time
    
    Error Handling:
        The tool returns descriptive error strings rather than raising exceptions,
        allowing the agent to handle failures gracefully and provide useful feedback
        to the user. Common error cases:
        - Permission denied (user lacks required RBAC permission)
        - Invalid or malformed URLs
        - Connection timeouts or failures
        - HTTP error status codes (4xx, 5xx)
        - Network errors
    """
    tool_description = description or (
        "Fetch content from a URL via HTTP GET request.\n"
        "Input: A valid HTTP or HTTPS URL string.\n"
        "Output: The response body text (up to {max_chars} characters) or an error message.\n"
        "Use this to retrieve live data from web endpoints, APIs, or documentation URLs.\n"
        "Example input: 'https://example.com/api/status'\n"
        "IMPORTANT: When using this tool, avoid providing general answers from your knowledge. "
        "Instead, if you fail to retrieve the data, inform the user with the error message returned by this tool and ask if they would like a general answer instead."
    ).format(max_chars=max_response_chars)
    
    @tool(name, description=tool_description)
    @require_tool_permission(required_permission)
    def _http_get_tool(url: str) -> str:
        """Fetch content from a URL via HTTP GET request."""
        # Validate URL
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            logger.warning(f"HTTP GET tool received invalid URL: {_sanitize_url_for_error(url)}")
            return f"Error: {error_msg}"
        
        # Make request with error handling
        try:
            logger.info(f"HTTP GET tool fetching: {_sanitize_url_for_error(url)}")
            
            response = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
            )
            
            # Check for authentication errors first
            if response.status_code == 401:
                logger.warning(
                    f"HTTP GET tool received 401 Unauthorized from {_sanitize_url_for_error(url)}"
                )
                return (
                    "Error: HTTP 401: Unauthorized. This endpoint requires authentication, "
                    "but the HTTP GET tool does not support authentication credentials. "
                    "Please use a public endpoint or provide the user with alternative access methods."
                )
            
            # Check for other HTTP errors (4xx, 5xx)
            if response.status_code >= 400:
                logger.warning(
                    f"HTTP GET tool received status {response.status_code} from {_sanitize_url_for_error(url)}"
                )
                status_text = response.reason or "Error"
                return f"Error: HTTP {response.status_code}: {status_text}"
            
            # Success - return response text (truncated if needed)
            response_text = response.text
            if len(response_text) > max_response_chars:
                truncated = response_text[:max_response_chars].rstrip()
                logger.info(
                    f"HTTP GET tool truncated response from {len(response_text)} to {max_response_chars} chars"
                )
                return f"{truncated}\n\n... [response truncated at {max_response_chars} characters]"
            
            logger.info(f"HTTP GET tool successfully fetched {len(response_text)} chars")
            return response_text
            
        except requests.exceptions.Timeout:
            logger.warning(f"HTTP GET tool timeout after {timeout}s: {_sanitize_url_for_error(url)}")
            return f"Error: Request timed out after {timeout} seconds. The endpoint may be slow or unresponsive."
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"HTTP GET tool connection error: {_sanitize_url_for_error(url)} - {str(e)}")
            return f"Error: Connection failed. The endpoint may be unreachable or the URL may be incorrect."
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"HTTP GET tool request error: {_sanitize_url_for_error(url)} - {str(e)}")
            return f"Error: Request failed - {type(e).__name__}. Please check the URL and try again."
            
        except Exception as e:
            logger.error(f"HTTP GET tool unexpected error: {_sanitize_url_for_error(url)} - {str(e)}")
            return f"Error: An unexpected error occurred while fetching the URL."
    
    return _http_get_tool
