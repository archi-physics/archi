"""
MONIT OpenSearch client and LangChain tool factories for querying OpenSearch indices.

This module provides:
- ``MONITOpenSearchClient``: HTTP client for CERN's MONIT Grafana API.
- ``create_monit_opensearch_search_tool``: Factory for index-agnostic search tools.
- ``create_monit_opensearch_aggregation_tool``: Factory for index-agnostic aggregation tools.

The tool factories are designed to work with **any** OpenSearch index. Domain-specific
knowledge (field names, query patterns, etc.) is injected via *skill* markdown files
that are appended to the tool description so the LLM has rich context.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import requests
from langchain.tools import tool

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_TIME_FORMAT = "strict_date_optional_time||epoch_millis"

# Hard limits to prevent LLM context window overflow
MAX_RESULTS_HARD_LIMIT = 50
MAX_OUTPUT_CHARS = 50_000


# ── Client ───────────────────────────────────────────────────────────────────

class MONITOpenSearchClient:
    """
    HTTP client for querying OpenSearch via CERN's MONIT Grafana API.

    This client handles authentication and query formatting for the
    ``_msearch`` endpoint used by Grafana datasource proxies.
    """

    def __init__(
        self,
        *,
        token: str,
        url: str = "https://monit-grafana.cern.ch/api/datasources/proxy/9269/_msearch",
        timeout: float = 60.0,
    ):
        """
        Initialize the MONIT OpenSearch client.

        Args:
            token: Bearer token for MONIT Grafana API authentication.
            url: Full URL to the ``_msearch`` endpoint.
            timeout: Request timeout in seconds.
        """
        if not token:
            raise ValueError(
                "MONIT Grafana token must be provided."
            )

        self.url = url
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def query(
        self,
        opensearch_query: Dict[str, Any],
        *,
        index: str,
        search_type: str = "query_then_fetch",
    ) -> Dict[str, Any]:
        """
        Execute an OpenSearch query against MONIT.

        Args:
            opensearch_query: OpenSearch Query DSL dictionary.
            index: Index pattern to query.
            search_type: Search type for meta query.

        Returns:
            Raw JSON response from OpenSearch.

        Raises:
            requests.HTTPError: On HTTP errors.
            requests.Timeout: On timeout.
        """
        meta_query = {
            "search_type": search_type,
            "ignore_unavailable": True,
            "index": [index],
        }

        # Format as NDJSON (newline-delimited JSON) for _msearch
        payload = "\n".join([json.dumps(meta_query), json.dumps(opensearch_query)]) + "\n"

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
        index: str,
    ) -> Dict[str, Any]:
        """
        Execute a Lucene query with time range filtering.

        Args:
            lucene_query: Lucene query string (e.g., ``data.name="/store/..."``).
            from_time: Start time in date math (e.g., ``now-7d``, ``now-24h``).
            to_time: End time in date math (e.g., ``now``).
            time_field: Field to use for time range filtering.
            size: Maximum number of results to return.
            index: Index pattern to query.

        Returns:
            Raw JSON response from OpenSearch.
        """
        opensearch_query = {
            "size": size,
            "_source": True,
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
                                    "format": _TIME_FORMAT,
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

        return self.query(opensearch_query, index=index)

    def search_with_aggregation(
        self,
        lucene_query: str,
        *,
        group_by: str,
        agg_type: str = "terms",
        top_n: int = 10,
        from_time: str = "now-24h",
        to_time: str = "now",
        time_field: str = "metadata.timestamp",
        index: str,
    ) -> Dict[str, Any]:
        """
        Execute a Lucene query with an aggregation.

        Args:
            lucene_query: Lucene query string to filter documents.
            group_by: Field to aggregate on.
            agg_type: Aggregation type (``terms``, ``sum``, ``avg``,
                      ``min``, ``max``, ``cardinality``).
            top_n: Number of buckets for ``terms`` aggregation (max 100).
            from_time: Start time in date math (e.g., ``now-7d``).
            to_time: End time in date math (e.g., ``now``).
            time_field: Field to use for time range filtering.
            index: Index pattern to query.

        Returns:
            Raw JSON response from OpenSearch.
        """
        top_n = min(top_n, 100)

        # Build aggregation clause
        if agg_type == "terms":
            # For text fields, try .keyword sub-field first
            field = group_by if group_by.endswith(".keyword") else f"{group_by}.keyword"
            agg_clause = {"terms": {"field": field, "size": top_n}}
        elif agg_type == "cardinality":
            agg_clause = {"cardinality": {"field": group_by}}
        elif agg_type in ("sum", "avg", "min", "max"):
            agg_clause = {agg_type: {"field": group_by}}
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        opensearch_query = {
            "size": 0,
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
                                    "format": _TIME_FORMAT,
                                }
                            }
                        }
                    ],
                }
            },
            "aggs": {
                "result": agg_clause,
            },
        }

        return self.query(opensearch_query, index=index)


# ── Response formatting helpers ──────────────────────────────────────────────

def _format_opensearch_response(
    response: Dict[str, Any],
    query: str,
    index: str,
    max_results: int,
    from_time: str,
    to_time: str,
) -> str:
    """Format an OpenSearch search response for LLM consumption."""
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

    # Handle different total formats
    if isinstance(total, dict):
        total_count = total.get("value", 0)
        relation = total.get("relation", "eq")
        total_str = f"{total_count}+" if relation == "gte" else str(total_count)
    else:
        total_count = total
        total_str = str(total_count)

    if not hits:
        return (
            f"No documents found in '{index}' matching: {query}\n"
            f"Time window: {from_time} → {to_time}"
        )

    # Format header
    lines = [
        f"Found {total_str} document(s) in '{index}' matching: {query}",
        f"Time window: {from_time} → {to_time}",
        f"Showing {min(len(hits), max_results)} result(s):",
        "",
    ]

    # Format each hit generically
    for idx, hit in enumerate(hits[:max_results], start=1):
        source = hit.get("_source", {})

        # Handle case where _source is a JSON string
        if isinstance(source, str):
            try:
                source = json.loads(source)
            except json.JSONDecodeError:
                lines.append(f"[{idx}] Error: Could not parse document data")
                continue

        if not isinstance(source, dict):
            lines.append(f"[{idx}] Error: Unexpected document data format")
            continue

        score = hit.get("_score")
        lines.append(f"[{idx}] Document (score: {score})")
        lines.append("─" * 50)

        # Flatten and display key fields from the document
        _append_fields(lines, source, indent=2)
        lines.append("")

    output = "\n".join(lines)

    # Truncate if output is too large
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n\n... [OUTPUT TRUNCATED - too many results]"

    return output


def _append_fields(
    lines: list,
    obj: Any,
    *,
    indent: int = 0,
    prefix: str = "",
    max_depth: int = 4,
) -> None:
    """Recursively append flattened key-value pairs from a dict/list."""
    if max_depth <= 0:
        return
    pad = " " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                _append_fields(lines, value, indent=indent, prefix=full_key, max_depth=max_depth - 1)
            elif isinstance(value, list):
                if len(value) <= 5 and all(not isinstance(v, (dict, list)) for v in value):
                    lines.append(f"{pad}{full_key}: {value}")
                else:
                    lines.append(f"{pad}{full_key}: [{len(value)} items]")
            else:
                str_val = str(value)
                if len(str_val) > 200:
                    str_val = str_val[:200] + "..."
                lines.append(f"{pad}{full_key}: {str_val}")


def _format_aggregation_response(
    response: Dict[str, Any],
    query: str,
    index: str,
    group_by: str,
    agg_type: str,
    from_time: str,
    to_time: str,
) -> str:
    """Format an OpenSearch aggregation response for LLM consumption."""
    responses = response.get("responses", [response])
    if not responses:
        return f"No aggregation results for query: {query}"

    first_response = responses[0]

    if first_response.get("error"):
        error = first_response["error"]
        error_type = error.get("type", "unknown")
        error_reason = error.get("reason", str(error))
        return f"Aggregation error ({error_type}): {error_reason}\n\nQuery was: {query}"

    total = first_response.get("hits", {}).get("total", {})
    if isinstance(total, dict):
        total_count = total.get("value", 0)
    else:
        total_count = total

    agg_result = first_response.get("aggregations", {}).get("result", {})

    lines = [
        f"Aggregation on '{index}' — {agg_type}({group_by})",
        f"Filter: {query}",
        f"Time window: {from_time} → {to_time}",
        f"Total matching documents: {total_count}",
        "",
    ]

    if agg_type == "terms":
        buckets = agg_result.get("buckets", [])
        if not buckets:
            lines.append("No buckets returned.")
        else:
            # Table header
            max_key_len = max(len(str(b.get("key", ""))) for b in buckets)
            col_width = max(max_key_len, 20)
            lines.append(f"  {'Value':<{col_width}}  Count")
            lines.append(f"  {'─' * col_width}  ─────")
            for bucket in buckets:
                key = str(bucket.get("key", ""))
                count = bucket.get("doc_count", 0)
                lines.append(f"  {key:<{col_width}}  {count}")
            # Show "other" count if present
            other = agg_result.get("sum_other_doc_count", 0)
            if other > 0:
                lines.append(f"  {'(other values)':<{col_width}}  {other}")
    elif agg_type == "cardinality":
        value = agg_result.get("value", 0)
        lines.append(f"Distinct values of {group_by}: {value}")
    elif agg_type in ("sum", "avg", "min", "max"):
        value = agg_result.get("value")
        lines.append(f"{agg_type}({group_by}): {value}")
    else:
        lines.append(f"Raw result: {json.dumps(agg_result, indent=2)}")

    return "\n".join(lines)


# ── Tool description builders ────────────────────────────────────────────────

def _build_search_tool_description(index: str, skill: Optional[str] = None) -> str:
    """Build the description for a search tool, optionally appending a skill."""
    base = (
        f"Search the '{index}' OpenSearch index using Lucene query syntax.\n\n"
        "Input parameters:\n"
        "- query: Lucene query string (required).\n"
        "- from_time: Start time (default: 'now-24h'). Supports date math (e.g., now-7d, now-24h).\n"
        "- to_time: End time (default: 'now'). Supports date math.\n"
        f"- max_results: Max documents to return (default: 10, hard limit: {MAX_RESULTS_HARD_LIMIT}).\n"
    )
    if skill:
        base += f"\n--- Domain Knowledge ---\n{skill}"
    return base


def _build_aggregation_tool_description(index: str, skill: Optional[str] = None) -> str:
    """Build the description for an aggregation tool, optionally appending a skill."""
    base = (
        f"Run aggregation queries on the '{index}' OpenSearch index.\n\n"
        "Use this for counting, grouping, statistics — NOT for fetching individual documents.\n\n"
        "Input parameters:\n"
        "- query: Lucene query string to filter documents (required). Use '*' for all documents.\n"
        "- group_by: Field to aggregate on (required, e.g. 'data.reason').\n"
        "- agg_type: Aggregation type (default: 'terms'). One of: terms, sum, avg, min, max, cardinality.\n"
        "- top_n: Number of top buckets for terms aggregation (default: 10, max: 100).\n"
        "- from_time: Start time (default: 'now-24h'). Supports date math (e.g., now-7d, now-24h).\n"
        "- to_time: End time (default: 'now'). Supports date math.\n"
    )
    if skill:
        base += f"\n--- Domain Knowledge ---\n{skill}"
    return base


# ── Tool factories ───────────────────────────────────────────────────────────

def create_monit_opensearch_search_tool(
    client: MONITOpenSearchClient,
    *,
    tool_name: str = "search_opensearch",
    index: str,
    skill: Optional[str] = None,
) -> Callable[..., str]:
    """
    Create a LangChain tool for searching an OpenSearch index via MONIT.

    Args:
        client: ``MONITOpenSearchClient`` instance.
        tool_name: Tool name for LangChain.
        index: Index pattern to query.
        skill: Optional skill markdown to append to the tool description.

    Returns:
        LangChain tool function.
    """
    tool_description = _build_search_tool_description(index, skill)

    @tool(tool_name, description=tool_description)
    def _search_opensearch(
        query: str,
        from_time: str = "now-24h",
        to_time: str = "now",
        max_results: int = 10,
    ) -> str:
        """
        Search OpenSearch for documents matching a Lucene query.

        Args:
            query: Lucene query string.
            from_time: Start time in date math (default: now-24h).
            to_time: End time in date math (default: now).
            max_results: Maximum number of results to return.

        Returns:
            Formatted string with matching documents.
        """
        if not query or not query.strip():
            return "Please provide a non-empty Lucene query."

        effective_max = min(max_results, MAX_RESULTS_HARD_LIMIT)

        try:
            response = client.search_with_lucene(
                lucene_query=query.strip(),
                from_time=from_time,
                to_time=to_time,
                size=effective_max,
                index=index,
            )
            return _format_opensearch_response(
                response, query.strip(), index, effective_max,
                from_time=from_time, to_time=to_time,
            )

        except requests.exceptions.Timeout:
            logger.warning("OpenSearch query timed out for query: %s", query)
            return (
                "Query timed out. The service may be slow or the query too broad. "
                "Try narrowing the time range or making the query more specific."
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.warning("OpenSearch HTTP error %s for query: %s", status_code, query)
            if status_code in (401, 403):
                return "Authentication failed. The token may be invalid or expired."
            return f"Query failed with HTTP error {status_code}. Please try again."
        except Exception as e:
            logger.error("OpenSearch query error: %s", e, exc_info=True)
            return f"Error querying OpenSearch: {e}"

    return _search_opensearch


def create_monit_opensearch_aggregation_tool(
    client: MONITOpenSearchClient,
    *,
    tool_name: str = "aggregate_opensearch",
    index: str,
    skill: Optional[str] = None,
) -> Callable[..., str]:
    """
    Create a LangChain tool for running aggregation queries on an OpenSearch index.

    Args:
        client: ``MONITOpenSearchClient`` instance.
        tool_name: Tool name for LangChain.
        index: Index pattern to query.
        skill: Optional skill markdown to append to the tool description.

    Returns:
        LangChain tool function.
    """
    tool_description = _build_aggregation_tool_description(index, skill)

    @tool(tool_name, description=tool_description)
    def _aggregate_opensearch(
        query: str,
        group_by: str,
        agg_type: str = "terms",
        top_n: int = 10,
        from_time: str = "now-24h",
        to_time: str = "now",
    ) -> str:
        """
        Run an aggregation query on OpenSearch.

        Args:
            query: Lucene query string to filter documents.
            group_by: Field to aggregate on.
            agg_type: Aggregation type (terms, sum, avg, min, max, cardinality).
            top_n: Number of top buckets for terms aggregation.
            from_time: Start time in date math (default: now-24h).
            to_time: End time in date math (default: now).

        Returns:
            Formatted aggregation results.
        """
        if not query or not query.strip():
            return "Please provide a non-empty Lucene query (use '*' for all documents)."

        if not group_by or not group_by.strip():
            return "Please provide a field to aggregate on (group_by)."

        try:
            response = client.search_with_aggregation(
                lucene_query=query.strip(),
                group_by=group_by.strip(),
                agg_type=agg_type,
                top_n=top_n,
                from_time=from_time,
                to_time=to_time,
                index=index,
            )
            return _format_aggregation_response(
                response, query.strip(), index, group_by.strip(), agg_type,
                from_time=from_time, to_time=to_time,
            )

        except requests.exceptions.Timeout:
            logger.warning("OpenSearch aggregation timed out for query: %s", query)
            return (
                "Aggregation timed out. Try narrowing the time range or simplifying the query."
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.warning("OpenSearch HTTP error %s for aggregation: %s", status_code, query)
            if status_code in (401, 403):
                return "Authentication failed. The token may be invalid or expired."
            return f"Aggregation failed with HTTP error {status_code}. Please try again."
        except Exception as e:
            logger.error("OpenSearch aggregation error: %s", e, exc_info=True)
            return f"Error running aggregation: {e}"

    return _aggregate_opensearch


__all__ = [
    "MONITOpenSearchClient",
    "create_monit_opensearch_search_tool",
    "create_monit_opensearch_aggregation_tool",
]
