"""
Example usage of the HTTP GET tool in an agent pipeline.

This demonstrates how to add the HTTP GET tool to an agent's tool list.
"""

# Example 1: Basic usage with defaults
from src.archi.pipelines.agents.tools import create_http_get_tool

http_tool = create_http_get_tool()
# Tool is now ready to be added to an agent's tool list

# Example 2: Customized configuration
http_tool_custom = create_http_get_tool(
    name="fetch_api_data",
    description="Fetch real-time data from REST API endpoints",
    timeout=15.0,
    max_response_chars=8000,
)

# Example 3: Adding to an agent (pseudocode based on cms_comp_ops_agent.py pattern)
"""
class MyAgent(BaseReActAgent):
    def _build_static_tools(self) -> List[Callable]:
        # Existing tools
        retriever_tool = create_retriever_tool(...)
        file_search_tool = create_file_search_tool(...)
        
        # Add HTTP GET tool
        http_tool = create_http_get_tool(
            name="fetch_url",
            description="Fetch live data from web endpoints or APIs",
            timeout=10.0,
            max_response_chars=4000,
        )
        
        return [retriever_tool, file_search_tool, http_tool]
"""

# Example 4: Tool invocation (when used by the agent)
"""
When the LLM decides to use the tool, it will call:
result = http_tool.invoke("https://example.com/api/status")

Possible returns:
- Success: "{'status': 'ok', 'version': '1.0'}"
- Error: "Error: Request timed out after 10.0 seconds..."
- Error: "Error: HTTP 404: Not Found"
- Error: "Error: Invalid URL scheme 'ftp'. Only HTTP and HTTPS are supported."
"""
