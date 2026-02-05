"""
MONIT OpenSearch client and LangChain tool for querying CMS Rucio events.

This module provides access to CERN's MONIT Grafana API for querying
OpenSearch indices containing CMS monitoring data.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import requests
from langchain.tools import tool

from src.utils.logging import get_logger
from src.utils.env import read_secret

logger = get_logger(__name__)


class MONITOpenSearchClient:
    """
    HTTP client for querying OpenSearch via CERN's MONIT Grafana API.
    
    This client handles authentication and query formatting for the
    _msearch endpoint used by Grafana datasource proxies.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        url: str = "https://monit-grafana.cern.ch/api/datasources/proxy/9269/_msearch",
        default_index: str = "monit_prod_cms_rucio_raw_events*",
        timeout: float = 60.0,
    ):
        """
        Initialize the MONIT OpenSearch client.
        
        Args:
            token: Bearer token for MONIT Grafana API authentication.
                   If not provided, reads from MONIT_GRAFANA_TOKEN env var.
            url: Full URL to the _msearch endpoint.
            default_index: Default index pattern for queries.
            timeout: Request timeout in seconds.
        """
        self.token = token or read_secret("MONIT_GRAFANA_TOKEN")
        if not self.token:
            raise ValueError(
                "MONIT Grafana token not provided. Set MONIT_GRAFANA_TOKEN environment variable."
            )
        
        self.url = url
        self.default_index = default_index
        self.timeout = timeout
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def query(
        self,
        es_query: Dict[str, Any],
        *,
        index: Optional[str] = None,
        search_type: str = "query_then_fetch",
    ) -> Dict[str, Any]:
        """
        Execute an Elasticsearch query against MONIT.
        
        Args:
            es_query: Elasticsearch Query DSL dictionary.
            index: Index pattern (defaults to self.default_index).
            search_type: Search type for meta query.
            
        Returns:
            Raw JSON response from OpenSearch.
            
        Raises:
            requests.HTTPError: On HTTP errors.
            requests.Timeout: On timeout.
        """
        index_pattern = index or self.default_index
        
        meta_query = {
            "search_type": search_type,
            "ignore_unavailable": True,
            "index": [index_pattern],
        }
        
        # Format as NDJSON (newline-delimited JSON) for _msearch
        payload = "\n".join([json.dumps(meta_query), json.dumps(es_query)]) + "\n"
        
        response = requests.post(
            self.url,
            headers=self.headers,
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search_with_lucene(
        self,
        lucene_query: str,
        *,
        from_time: str = "now-24h",
        to_time: str = "now",
        time_field: str = "metadata.timestamp",
        size: int = 10,
        index: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Lucene query with time range filtering.
        
        Args:
            lucene_query: Lucene query string (e.g., 'data.name="/store/..."').
            from_time: Start time in Elasticsearch date math.
            to_time: End time in Elasticsearch date math.
            time_field: Field to use for time range filtering.
            size: Maximum number of results to return.
            index: Index pattern (defaults to self.default_index).
            
        Returns:
            Raw JSON response from OpenSearch.
        """
        es_query = {
            "size": size,
            "_source": True,  # Ensure all source fields are returned
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "query": lucene_query,
                                "analyze_wildcard": True,
                            }
                        }
                    ],
                    "filter": [
                        {
                            "range": {
                                time_field: {
                                    "gte": from_time,
                                    "lte": to_time,
                                    "format": "strict_date_optional_time||epoch_millis",
                                }
                            }
                        }
                    ],
                }
            },
            "sort": [
                {time_field: {"order": "desc"}}
            ],
        }
        
        return self.query(es_query, index=index)


def _format_file_size(bytes_value: Any) -> str:
    """Format file size in human-readable format."""
    try:
        # Handle string with commas
        if isinstance(bytes_value, str):
            bytes_value = int(bytes_value.replace(",", ""))
        bytes_value = int(bytes_value)
        
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(bytes_value) < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    except (ValueError, TypeError):
        return str(bytes_value)


def _format_duration(seconds: Any) -> str:
    """Format duration in human-readable format."""
    try:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except (ValueError, TypeError):
        return str(seconds)


def _format_rucio_event(hit: Dict[str, Any], idx: int) -> str:
    """Format a single Rucio event for LLM consumption."""
    source = hit.get("_source", {})
    
    # Handle case where _source is a JSON string (MONIT sometimes returns this)
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError:
            logger.warning("Failed to parse _source as JSON: %s", source[:200])
            return f"[{idx}] Error: Could not parse event data"
    
    # If source is still not a dict, log and return error
    if not isinstance(source, dict):
        logger.warning("Unexpected _source type: %s", type(source))
        return f"[{idx}] Error: Unexpected event data format"
    
    # Log the first few keys for debugging
    logger.debug("_source keys: %s", list(source.keys())[:10])
    
    # Extract data fields - handle both nested and flat structures
    data = {}
    
    # Check for nested structure: {"data": {...}, "metadata": {...}}
    if "data" in source and isinstance(source["data"], dict):
        # Nested structure
        for key, value in source["data"].items():
            data[key] = value
        if "metadata" in source and isinstance(source["metadata"], dict):
            for key, value in source["metadata"].items():
                data[f"meta_{key}"] = value
    else:
        # Flat structure with prefixes like "data.name", "metadata.timestamp"
        for key, value in source.items():
            if key.startswith("data."):
                data[key[5:]] = value  # Remove 'data.' prefix
            elif key.startswith("metadata."):
                data[f"meta_{key[9:]}"] = value  # Prefix metadata fields
            else:
                data[key] = value
    
    # Build formatted output - use event_created_at or meta_timestamp for time
    timestamp = data.get("event_created_at") or data.get("meta_timestamp") or "unknown"
    lines = [f"[{idx}] Event at {timestamp}"]
    lines.append("    " + "─" * 50)
    
    # Event type is the primary status indicator (not "state" which doesn't exist)
    event_type = data.get("event_type", "unknown")
    lines.append(f"    event_type:    {event_type}")
    
    # Show reason for failed transfers
    reason = data.get("reason")
    if reason:
        # Truncate long reason messages
        if len(reason) > 100:
            reason = reason[:100] + "..."
        lines.append(f"    reason:        {reason}")
    
    # IDs (important for lookups)
    # transfer_id is the FTS job ID; request_id is the Rucio request ID
    if data.get("transfer_id"):
        lines.append(f"    transfer_id:   {data['transfer_id']}")
    if data.get("request_id"):
        lines.append(f"    request_id:    {data['request_id']}")
    
    # FTS monitoring link (very useful for debugging)
    if data.get("transfer_link"):
        lines.append(f"    transfer_link: {data['transfer_link']}")
    
    # File info
    lines.append("")
    lines.append("    File:")
    if data.get("name"):
        name = data["name"]
        # Truncate long file names for readability
        if len(name) > 80:
            name = name[:40] + "..." + name[-37:]
        lines.append(f"      name:        {name}")
    if data.get("scope"):
        lines.append(f"      scope:       {data['scope']}")
    if data.get("bytes") or data.get("file_size"):
        size = data.get("bytes") or data.get("file_size")
        lines.append(f"      size:        {_format_file_size(size)}")
    if data.get("checksum_adler"):
        lines.append(f"      checksum:    {data['checksum_adler']} (adler32)")
    
    # Transfer info
    src_rse = data.get("src_rse")
    dst_rse = data.get("dst_rse")
    if src_rse or dst_rse:
        lines.append("")
        lines.append("    Transfer:")
        if src_rse:
            lines.append(f"      src_rse:     {src_rse}")
        if dst_rse:
            lines.append(f"      dst_rse:     {dst_rse}")
        if data.get("activity"):
            lines.append(f"      activity:    {data['activity']}")
        if data.get("account"):
            lines.append(f"      account:     {data['account']}")
        if data.get("protocol"):
            lines.append(f"      protocol:    {data['protocol']}")
        if data.get("transfer_endpoint"):
            lines.append(f"      endpoint:    {data['transfer_endpoint']}")
    
    # Timing info
    timing_info = []
    if data.get("duration"):
        timing_info.append(f"duration: {_format_duration(data['duration'])}")
    if data.get("submitted_at"):
        timing_info.append(f"submitted: {data['submitted_at']}")
    if data.get("started_at"):
        timing_info.append(f"started: {data['started_at']}")
    if data.get("transferred_at"):
        timing_info.append(f"transferred: {data['transferred_at']}")
    
    if timing_info:
        lines.append("")
        lines.append("    Timing:")
        for info in timing_info:
            lines.append(f"      {info}")
    
    # Dataset info
    dataset = data.get("dataset")
    if dataset:
        lines.append("")
        lines.append("    Dataset:")
        # Truncate long dataset names
        if len(dataset) > 80:
            dataset = dataset[:40] + "..." + dataset[-37:]
        lines.append(f"      name:        {dataset}")
        if data.get("datasetScope"):
            lines.append(f"      scope:       {data['datasetScope']}")
    
    return "\n".join(lines)


def _format_monit_response(
    response: Dict[str, Any],
    query: str,
    max_results: int,
) -> str:
    """Format MONIT OpenSearch response for LLM consumption."""
    # Log response structure for debugging
    logger.debug("Response keys: %s", list(response.keys()))
    
    # Handle _msearch response format (array of responses)
    responses = response.get("responses", [response])
    if not responses:
        return f"No results found for query: {query}"
    
    first_response = responses[0]
    logger.debug("First response keys: %s", list(first_response.keys()))
    
    # Check for errors
    if first_response.get("error"):
        error = first_response["error"]
        error_type = error.get("type", "unknown")
        error_reason = error.get("reason", str(error))
        return f"Query error ({error_type}): {error_reason}\n\nQuery was: {query}"
    
    hits_obj = first_response.get("hits", {})
    hits = hits_obj.get("hits", [])
    total = hits_obj.get("total", {})
    
    # Debug log the first hit structure
    if hits:
        first_hit = hits[0]
        logger.debug("First hit keys: %s", list(first_hit.keys()))
        source = first_hit.get("_source")
        logger.debug("_source type: %s", type(source).__name__)
        if isinstance(source, dict):
            logger.debug("_source sample keys: %s", list(source.keys())[:5])
        elif isinstance(source, str):
            logger.debug("_source is string, first 200 chars: %s", source[:200])
    
    # Handle different total formats
    if isinstance(total, dict):
        total_count = total.get("value", 0)
        relation = total.get("relation", "eq")
        total_str = f"{total_count}+" if relation == "gte" else str(total_count)
    else:
        total_count = total
        total_str = str(total_count)
    
    if not hits:
        return f"No Rucio events found matching query: {query}"
    
    # Format header
    lines = [
        f"Found {total_str} Rucio event(s) matching query: {query}",
        f"Showing {min(len(hits), max_results)} result(s):",
        "",
    ]
    
    # Format each hit
    for idx, hit in enumerate(hits[:max_results], start=1):
        lines.append(_format_rucio_event(hit, idx))
        lines.append("")
    
    return "\n".join(lines)


def create_monit_opensearch_tool(
    client: MONITOpenSearchClient,
    *,
    name: str = "search_monit_rucio",
    description: Optional[str] = None,
    max_results: int = 10,
) -> Callable[..., str]:
    """
    Create a LangChain tool for querying MONIT OpenSearch.
    
    The tool accepts Lucene query syntax and returns formatted results
    suitable for LLM consumption.
    
    Args:
        client: MONITOpenSearchClient instance.
        name: Tool name for LangChain.
        description: Tool description (uses default if not provided).
        max_results: Default maximum results to return.
        
    Returns:
        LangChain tool function.
    """
    
    tool_description = description or (
        "Query CERN MONIT for CMS Rucio data transfer events using Lucene query syntax.\n"
        "Use this to find transfer status, file locations, IDs, and transfer metrics.\n\n"
        "Input parameters:\n"
        "- query: Lucene query string (required). Examples:\n"
        '  - data.name="/store/mc/Run3Summer22MiniAODv4/..."  (exact file path)\n'
        "  - data.name:*MiniAODSIM*  (wildcard search)\n"
        "  - data.event_type:transfer-submitted  (submitted transfers)\n"
        "  - data.event_type:transfer-failed  (failed transfers)\n"
        "  - data.event_type:transfer-done  (completed transfers)\n"
        "  - data.dst_rse:T1_IT_CNAF_Tape  (by destination RSE)\n"
        "  - data.event_type:transfer-failed AND data.dst_rse:T2_CH_CERN  (boolean)\n"
        "- from_time: Start time (default: 'now-24h'). Supports ES date math.\n"
        "- to_time: End time (default: 'now'). Supports ES date math.\n"
        "- max_results: Max documents to return (default: 10).\n\n"
        "Common data fields:\n"
        "- event_type: transfer-submitted, transfer-done, transfer-failed, etc.\n"
        "- name: file path, src_rse/dst_rse: source/destination RSE\n"
        "- transfer_id: FTS job ID, request_id: Rucio request ID\n"
        "- reason: error message for failed transfers\n"
        "- bytes, activity, scope, dataset, account, protocol"
    )
    
    @tool(name, description=tool_description)
    def _search_monit_rucio(
        query: str,
        from_time: str = "now-24h",
        to_time: str = "now",
        max_results_override: Optional[int] = None,
    ) -> str:
        """
        Query MONIT OpenSearch for CMS Rucio events.
        
        Args:
            query: Lucene query string (e.g., data.name="/store/..." or data.event_type:transfer-*)
            from_time: Start time in ES date math (default: now-24h)
            to_time: End time in ES date math (default: now)
            max_results_override: Override default max results
            
        Returns:
            Formatted string with matching Rucio events.
        """
        if not query or not query.strip():
            return "Please provide a non-empty Lucene query."
        
        effective_max = max_results_override or max_results
        
        try:
            response = client.search_with_lucene(
                lucene_query=query.strip(),
                from_time=from_time,
                to_time=to_time,
                size=effective_max,
            )
            return _format_monit_response(response, query.strip(), effective_max)
            
        except requests.exceptions.Timeout:
            logger.warning("MONIT query timed out for query: %s", query)
            return (
                "Query timed out. The MONIT service may be slow or the query too broad. "
                "Try narrowing the time range or making the query more specific."
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.warning("MONIT HTTP error %s for query: %s", status_code, query)
            if status_code == 401 or status_code == 403:
                return "Authentication failed. The MONIT token may be invalid or expired."
            return f"MONIT query failed with HTTP error {status_code}. Please try again."
        except Exception as e:
            logger.error("MONIT query error: %s", e, exc_info=True)
            return f"Error querying MONIT: {str(e)}"
    
    return _search_monit_rucio


__all__ = [
    "MONITOpenSearchClient",
    "create_monit_opensearch_tool",
]
