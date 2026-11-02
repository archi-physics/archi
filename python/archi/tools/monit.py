"""Live MONIT OpenSearch agent tools (search + aggregate).

Rewritten from okg-deployments ``cms/cms_tools/monit_live.py`` (665 LOC)
at ``main@f33a9c4`` for the archi v3 package (task.w2.sources-monit).
Changes from the original:

- **De-CMS-ified names**: the four wired entry points are
  :func:`archi_monit_rucio_search`, :func:`archi_monit_rucio_aggregate`,
  :func:`archi_monit_condor_search`, and
  :func:`archi_monit_condor_aggregate` (cms originals: ``rucio_search``
  / ``rucio_aggregate`` / ``condor_search`` / ``condor_aggregate``,
  wired as ``cms_monit_rucio_search`` etc.).
- **Endpoints/datasources as parameters**: the generic cores
  :func:`monit_search` and :func:`monit_aggregate` take
  ``datasource_id`` / ``index`` / ``summary_fields`` /
  ``grafana_base_url`` / ``token_env`` explicitly; the four named
  wrappers bind the CMS defaults (Rucio raw events datasource 9269,
  Condor raw metrics datasource 8787 on
  ``https://monit-grafana.cern.ch``). The wrappers deliberately keep
  the cms agent-facing signatures — endpoint knobs are *not* exposed
  as agent parameters, so an agent cannot re-point a wired tool.
- Credentials stay env-var references only: ``token_env`` (default
  ``MONIT_GRAFANA_TOKEN``) names the variable; the value never appears
  in payloads, and it is redacted out of echoed queries.
- As in the original, a missing token raises ``RuntimeError`` (the
  ``agent_tools`` wiring declares ``requires_env`` so the harness
  gates registration/calls); timeouts and HTTP errors come back as
  structured ``error`` payloads, and every payload carries
  ``boundary: external_live`` plus ``observed_at``, source/index, the
  sanitized query, and the explicit time window.
- 2026-08 adversarial review (see
  ``pact/changes/circleback-fixes/notes-live.md``): every
  ``requests.RequestException`` (ConnectionError, ...) and a non-JSON
  200 body (Grafana HTML error page) also come back as structured
  ``error`` payloads instead of escaping as raw exceptions; the echoed
  ``time_window`` is exactly the window the query body used (both are
  sanitized once, identically); and the terms ``.keyword``->raw retry
  fires only when the ``.keyword`` attempt failed with a field-related
  error or matched documents yet produced no buckets — a zero-match
  empty bucket list is a valid result and is returned as-is (and a
  failing raw retry never replaces a valid empty ``.keyword`` result).

Wiring — an instance registers the tools in its ``deployment.yaml``
``agent_tools`` block (adapted from the cms block; the mapping key is
the tool name agents see, ``module``/``function`` point here, and
``boundary: external_live`` is mandatory for credentialed live tools —
the deployment contract lints
``akmon.external_live_boundary_missing`` otherwise)::

    agent_tools:
      archi_monit_rucio_search:
        module: archi.tools.monit
        function: archi_monit_rucio_search
        boundary: external_live
        requires_env: [MONIT_GRAFANA_TOKEN]
        optional: true
        description: |
          Search live external MONIT Rucio raw OpenSearch events. Results are
          not generation-pinned OKG facts; each response includes observed_at,
          source/index, sanitized query, explicit time window, and bounded
          MONIT document evidence.
      archi_monit_rucio_aggregate:
        module: archi.tools.monit
        function: archi_monit_rucio_aggregate
        boundary: external_live
        requires_env: [MONIT_GRAFANA_TOKEN]
        optional: true
        description: |
          Aggregate live external MONIT Rucio raw OpenSearch events. Results
          are not generation-pinned OKG facts; each response includes
          observed_at, source/index, sanitized query, explicit time window,
          and bounded aggregation evidence.
      archi_monit_condor_search:
        module: archi.tools.monit
        function: archi_monit_condor_search
        boundary: external_live
        requires_env: [MONIT_GRAFANA_TOKEN]
        optional: true
        description: |
          Search live external MONIT HTCondor raw OpenSearch metrics. Results
          are not generation-pinned OKG facts; each response includes
          observed_at, source/index, sanitized query, explicit time window,
          and bounded MONIT document evidence.
      archi_monit_condor_aggregate:
        module: archi.tools.monit
        function: archi_monit_condor_aggregate
        boundary: external_live
        requires_env: [MONIT_GRAFANA_TOKEN]
        optional: true
        description: |
          Aggregate live external MONIT HTCondor raw OpenSearch metrics.
          Results are not generation-pinned OKG facts; each response includes
          observed_at, source/index, sanitized query, explicit time window,
          and bounded aggregation evidence.

An instance targeting different indices defines thin module-level
wrappers over :func:`monit_search` / :func:`monit_aggregate` in its own
deployment package (the ``agent_tools`` loader imports bare
``module.function`` references and passes only agent-supplied
arguments, so per-instance knobs must be bound in a wrapper, not in
YAML)::

    # <instance>_tools/monit.py
    from archi.tools.monit import monit_search

    def my_fts_search(query, from_time="now-24h", to_time="now",
                      max_results=5, page=1):
        return monit_search(
            query,
            source_name="monit_opensearch:my_fts_index",
            datasource_id=1234,
            index="monit_prod_my_fts-*",
            summary_fields=("data.event_type", "metadata.timestamp"),
            from_time=from_time,
            to_time=to_time,
            max_results=max_results,
            page=page,
        )
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import requests


DEFAULT_GRAFANA_BASE_URL = "https://monit-grafana.cern.ch"
DEFAULT_TOKEN_ENV = "MONIT_GRAFANA_TOKEN"
# CMS MONIT raw-event datasources (documented defaults for the wrappers).
RUCIO_DATASOURCE_ID = 9269
CONDOR_DATASOURCE_ID = 8787
RUCIO_INDEX = "monit_prod_cms_rucio_raw_events*"
CONDOR_INDEX = "monit_prod_condor_raw_metric*"
TIME_FORMAT = "strict_date_optional_time||epoch_millis"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_HARD_LIMIT = 10
DEFAULT_TOP_N = 10
MAX_TOP_N_HARD_LIMIT = 100
MAX_QUERY_CHARS = 800
DEFAULT_TIMEOUT_SECONDS = 30.0

RUCIO_SUMMARY_FIELDS = (
    "data.event_type",
    "data.scope",
    "data.name",
    "data.dataset",
    "data.src_rse",
    "data.dst_rse",
    "data.rse",
    "data.account",
    "data.activity",
    "data.reason",
    "data.bytes",
    "data.transfer_id",
    "data.request_id",
    "data.created_at",
    "data.started_at",
    "data.transferred_at",
    "metadata.timestamp",
)

CONDOR_SUMMARY_FIELDS = (
    "data.Site",
    "data.CMS_JobType",
    "data.Type",
    "data.Status",
    "data.JobStatus",
    "data.ExitCode",
    "data.ErrorType",
    "data.CpuEff",
    "data.WallClockHr",
    "data.CpuTimeHr",
    "data.RequestCpus",
    "data.Workflow",
    "data.WMAgent_TaskType",
    "data.ScheddName",
    "data.RecordTime",
    "metadata.timestamp",
)

SUPPORTED_AGG_TYPES = {"terms", "sum", "avg", "min", "max", "cardinality"}


def archi_monit_rucio_search(
    query: str,
    from_time: str = "now-24h",
    to_time: str = "now",
    max_results: int = DEFAULT_MAX_RESULTS,
    page: int = 1,
) -> dict[str, Any]:
    """Search live Rucio MONIT raw events with Lucene syntax.

    cms original: ``rucio_search`` (wired as ``cms_monit_rucio_search``).
    """
    return monit_search(
        query,
        source_name="monit_opensearch:rucio_raw_events",
        datasource_id=RUCIO_DATASOURCE_ID,
        index=RUCIO_INDEX,
        summary_fields=RUCIO_SUMMARY_FIELDS,
        from_time=from_time,
        to_time=to_time,
        max_results=max_results,
        page=page,
    )


def archi_monit_rucio_aggregate(
    query: str,
    group_by: str,
    agg_type: str = "terms",
    top_n: int = DEFAULT_TOP_N,
    from_time: str = "now-24h",
    to_time: str = "now",
) -> dict[str, Any]:
    """Aggregate live Rucio MONIT raw events with Lucene syntax.

    cms original: ``rucio_aggregate`` (wired as
    ``cms_monit_rucio_aggregate``).
    """
    return monit_aggregate(
        query,
        group_by,
        source_name="monit_opensearch:rucio_raw_events",
        datasource_id=RUCIO_DATASOURCE_ID,
        index=RUCIO_INDEX,
        agg_type=agg_type,
        top_n=top_n,
        from_time=from_time,
        to_time=to_time,
    )


def archi_monit_condor_search(
    query: str,
    from_time: str = "now-24h",
    to_time: str = "now",
    max_results: int = DEFAULT_MAX_RESULTS,
    page: int = 1,
) -> dict[str, Any]:
    """Search live HTCondor MONIT raw metrics with Lucene syntax.

    cms original: ``condor_search`` (wired as ``cms_monit_condor_search``).
    """
    return monit_search(
        query,
        source_name="monit_opensearch:condor_raw_metric",
        datasource_id=CONDOR_DATASOURCE_ID,
        index=CONDOR_INDEX,
        summary_fields=CONDOR_SUMMARY_FIELDS,
        from_time=from_time,
        to_time=to_time,
        max_results=max_results,
        page=page,
    )


def archi_monit_condor_aggregate(
    query: str,
    group_by: str,
    agg_type: str = "terms",
    top_n: int = DEFAULT_TOP_N,
    from_time: str = "now-24h",
    to_time: str = "now",
) -> dict[str, Any]:
    """Aggregate live HTCondor MONIT raw metrics with Lucene syntax.

    cms original: ``condor_aggregate`` (wired as
    ``cms_monit_condor_aggregate``).
    """
    return monit_aggregate(
        query,
        group_by,
        source_name="monit_opensearch:condor_raw_metric",
        datasource_id=CONDOR_DATASOURCE_ID,
        index=CONDOR_INDEX,
        agg_type=agg_type,
        top_n=top_n,
        from_time=from_time,
        to_time=to_time,
    )


def monit_search(
    query: str,
    *,
    source_name: str,
    datasource_id: int,
    index: str,
    summary_fields: tuple[str, ...],
    from_time: str = "now-24h",
    to_time: str = "now",
    max_results: int = DEFAULT_MAX_RESULTS,
    page: int = 1,
    grafana_base_url: str = DEFAULT_GRAFANA_BASE_URL,
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generic MONIT OpenSearch Lucene search (parameterized core)."""
    observed_at = _observed_at()
    clean_query = _clean_query(query, token_env=token_env)
    # Sanitize the window once; the query body and the echoed
    # time_window must use the exact same values.
    from_time = _sanitize_time(from_time, "now-24h")
    to_time = _sanitize_time(to_time, "now")
    effective_max = _bounded_int(max_results, default=DEFAULT_MAX_RESULTS,
                                 lower=1, upper=MAX_RESULTS_HARD_LIMIT)
    effective_page = _bounded_int(page, default=1, lower=1, upper=1000)
    offset = (effective_page - 1) * effective_max
    descriptor = {
        "lucene": clean_query,
        "index": index,
        "time_field": "metadata.timestamp",
        "max_results": effective_max,
        "page": effective_page,
        "offset": offset,
    }
    time_window = _time_window(from_time, to_time)
    if not clean_query:
        return _search_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            results=[],
            total={"value": 0, "relation": "eq"},
            error="query must be non-empty",
        )

    body = {
        "size": effective_max,
        "from": offset,
        "_source": list(summary_fields),
        "query": _query_filter(clean_query, from_time=from_time, to_time=to_time),
        "sort": [{"metadata.timestamp": {"order": "desc"}}],
    }
    try:
        response = _post_msearch(
            grafana_base_url=grafana_base_url,
            token_env=token_env,
            datasource_id=datasource_id,
            index=index,
            body=body,
            timeout=timeout,
        )
        first = _first_response(response)
        error = _response_error(first)
        if error:
            return _search_payload(
                source_name=source_name,
                observed_at=observed_at,
                query=descriptor,
                time_window=time_window,
                results=[],
                total={"value": 0, "relation": "eq"},
                error=error,
            )
        hits_obj = first.get("hits", {})
        hits = hits_obj.get("hits", [])
        results = [
            _format_hit(hit, summary_fields=summary_fields)
            for hit in hits[:effective_max]
        ]
        return _search_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            results=results,
            total=_total_descriptor(hits_obj.get("total", 0)),
        )
    except requests.exceptions.Timeout:
        return _search_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            results=[],
            total={"value": 0, "relation": "eq"},
            error="MONIT OpenSearch request timed out",
        )
    except requests.exceptions.HTTPError as exc:
        return _search_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            results=[],
            total={"value": 0, "relation": "eq"},
            error=_http_error_message(exc),
        )
    except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
        # ConnectionError, TLS failures, non-JSON 200 bodies (Grafana
        # HTML error pages), ...: same structured payload as the
        # timeout/HTTP paths — never a raw exception to the agent.
        return _search_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            results=[],
            total={"value": 0, "relation": "eq"},
            error=_request_error_message(exc),
        )


def monit_aggregate(
    query: str,
    group_by: str,
    *,
    source_name: str,
    datasource_id: int,
    index: str,
    agg_type: str = "terms",
    top_n: int = DEFAULT_TOP_N,
    from_time: str = "now-24h",
    to_time: str = "now",
    grafana_base_url: str = DEFAULT_GRAFANA_BASE_URL,
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generic MONIT OpenSearch aggregation (parameterized core)."""
    observed_at = _observed_at()
    clean_query = _clean_query(query, token_env=token_env)
    # Sanitize the window once; the query body and the echoed
    # time_window must use the exact same values.
    from_time = _sanitize_time(from_time, "now-24h")
    to_time = _sanitize_time(to_time, "now")
    clean_group_by = (group_by or "").strip()
    clean_agg_type = (agg_type or "terms").strip().lower()
    effective_top_n = _bounded_int(top_n, default=DEFAULT_TOP_N,
                                   lower=1, upper=MAX_TOP_N_HARD_LIMIT)
    descriptor = {
        "lucene": clean_query,
        "index": index,
        "time_field": "metadata.timestamp",
        "group_by": clean_group_by,
        "agg_type": clean_agg_type,
        "top_n": effective_top_n,
    }
    time_window = _time_window(from_time, to_time)
    validation_error = _aggregation_validation_error(
        query=clean_query,
        group_by=clean_group_by,
        agg_type=clean_agg_type,
    )
    if validation_error:
        return _aggregation_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            aggregation={
                "agg_type": clean_agg_type,
                "group_by": clean_group_by,
                "total_matching_documents": 0,
                "buckets": [],
                "error": validation_error,
            },
        )

    agg_field = _terms_field(clean_group_by) if clean_agg_type == "terms" else clean_group_by
    body = _aggregation_body(
        lucene_query=clean_query,
        field=agg_field,
        agg_type=clean_agg_type,
        top_n=effective_top_n,
        from_time=from_time,
        to_time=to_time,
    )
    try:
        response = _post_msearch(
            grafana_base_url=grafana_base_url,
            token_env=token_env,
            datasource_id=datasource_id,
            index=index,
            body=body,
            timeout=timeout,
        )
        first = _first_response(response)
        keyword_error = _response_error(first)
        matched_but_bucketless = (
            not keyword_error
            and not _aggregation_buckets(first)
            and _total_descriptor(
                first.get("hits", {}).get("total", 0)
            )["value"] > 0
        )
        if (
            clean_agg_type == "terms"
            and agg_field != clean_group_by
            and (
                (
                    keyword_error
                    and _looks_like_field_error(keyword_error, agg_field)
                )
                or matched_but_bucketless
            )
        ):
            # The auto-appended .keyword sub-field is missing or not
            # aggregatable (field-related error, or documents matched
            # yet produced zero buckets — e.g. a numeric field whose
            # .keyword is unmapped): retry with the raw field name.
            # A zero-bucket .keyword result with zero matching
            # documents is a valid empty aggregation and is returned
            # as-is — retrying it used to surface fielddata errors for
            # legitimately empty windows.
            body = _aggregation_body(
                lucene_query=clean_query,
                field=clean_group_by,
                agg_type=clean_agg_type,
                top_n=effective_top_n,
                from_time=from_time,
                to_time=to_time,
            )
            retry_response = _post_msearch(
                grafana_base_url=grafana_base_url,
                token_env=token_env,
                datasource_id=datasource_id,
                index=index,
                body=body,
                timeout=timeout,
            )
            retry_first = _first_response(retry_response)
            if keyword_error or not _response_error(retry_first):
                # Use the retry only when the .keyword attempt had
                # failed or the raw attempt succeeded; a failing raw
                # retry must not replace a valid empty .keyword result.
                first = retry_first
        error = _response_error(first)
        if error:
            aggregation = {
                "agg_type": clean_agg_type,
                "group_by": clean_group_by,
                "total_matching_documents": 0,
                "buckets": [],
                "error": error,
            }
        else:
            aggregation = _format_aggregation(
                first,
                agg_type=clean_agg_type,
                group_by=clean_group_by,
            )
        return _aggregation_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            aggregation=aggregation,
        )
    except requests.exceptions.Timeout:
        aggregation = {
            "agg_type": clean_agg_type,
            "group_by": clean_group_by,
            "total_matching_documents": 0,
            "buckets": [],
            "error": "MONIT OpenSearch request timed out",
        }
        return _aggregation_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            aggregation=aggregation,
        )
    except requests.exceptions.HTTPError as exc:
        aggregation = {
            "agg_type": clean_agg_type,
            "group_by": clean_group_by,
            "total_matching_documents": 0,
            "buckets": [],
            "error": _http_error_message(exc),
        }
        return _aggregation_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            aggregation=aggregation,
        )
    except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
        # ConnectionError, TLS failures, non-JSON 200 bodies (Grafana
        # HTML error pages), ...: same structured payload as the
        # timeout/HTTP paths — never a raw exception to the agent.
        aggregation = {
            "agg_type": clean_agg_type,
            "group_by": clean_group_by,
            "total_matching_documents": 0,
            "buckets": [],
            "error": _request_error_message(exc),
        }
        return _aggregation_payload(
            source_name=source_name,
            observed_at=observed_at,
            query=descriptor,
            time_window=time_window,
            aggregation=aggregation,
        )


def _post_msearch(
    *,
    grafana_base_url: str,
    token_env: str,
    datasource_id: int,
    index: str,
    body: dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is not set")
    meta = {
        "search_type": "query_then_fetch",
        "ignore_unavailable": True,
        "index": [index],
    }
    payload = "\n".join((json.dumps(meta), json.dumps(body))) + "\n"
    response = requests.post(
        f"{grafana_base_url.rstrip('/')}/api/datasources/proxy/"
        f"{datasource_id}/_msearch",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        data=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _query_filter(lucene_query: str, *, from_time: str, to_time: str) -> dict[str, Any]:
    return {
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
                        "metadata.timestamp": {
                            "gte": from_time,
                            "lte": to_time,
                            "format": TIME_FORMAT,
                        }
                    }
                }
            ],
        }
    }


def _aggregation_body(
    *,
    lucene_query: str,
    field: str,
    agg_type: str,
    top_n: int,
    from_time: str,
    to_time: str,
) -> dict[str, Any]:
    if agg_type == "terms":
        agg_clause = {"terms": {"field": field, "size": top_n}}
    elif agg_type == "cardinality":
        agg_clause = {"cardinality": {"field": field}}
    else:
        agg_clause = {agg_type: {"field": field}}
    return {
        "size": 0,
        "query": _query_filter(lucene_query, from_time=from_time, to_time=to_time),
        "aggs": {"result": agg_clause},
    }


def _format_hit(
    hit: dict[str, Any],
    *,
    summary_fields: tuple[str, ...],
) -> dict[str, Any]:
    source = hit.get("_source") or {}
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError:
            source = {}
    if not isinstance(source, dict):
        source = {}
    fields = {
        field: _extract_nested(source, field)
        for field in summary_fields
        if _extract_nested(source, field) is not None
    }
    return {
        "document_id": hit.get("_id"),
        "index": hit.get("_index"),
        "score": hit.get("_score"),
        "fields": fields,
    }


def _format_aggregation(
    response: dict[str, Any],
    *,
    agg_type: str,
    group_by: str,
) -> dict[str, Any]:
    agg_result = response.get("aggregations", {}).get("result", {})
    out: dict[str, Any] = {
        "agg_type": agg_type,
        "group_by": group_by,
        "total_matching_documents": _total_descriptor(
            response.get("hits", {}).get("total", 0),
        )["value"],
    }
    if agg_type == "terms":
        out["buckets"] = [
            {
                "key": bucket.get("key"),
                "doc_count": bucket.get("doc_count", 0),
            }
            for bucket in agg_result.get("buckets", [])
        ]
        out["sum_other_doc_count"] = agg_result.get("sum_other_doc_count", 0)
    else:
        out["value"] = agg_result.get("value")
        out["buckets"] = []
    return out


def _search_payload(
    *,
    source_name: str,
    observed_at: str,
    query: dict[str, Any],
    time_window: dict[str, str],
    results: list[dict[str, Any]],
    total: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "boundary": "external_live",
        "source": source_name,
        "observed_at": observed_at,
        "query": query,
        "time_window": time_window,
        "total": total,
        "results": results,
    }
    if error:
        payload["error"] = error
    return payload


def _aggregation_payload(
    *,
    source_name: str,
    observed_at: str,
    query: dict[str, Any],
    time_window: dict[str, str],
    aggregation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "boundary": "external_live",
        "source": source_name,
        "observed_at": observed_at,
        "query": query,
        "time_window": time_window,
        "aggregation": aggregation,
    }


def _extract_nested(source: dict[str, Any], dotted_key: str) -> Any:
    current: Any = source
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _first_response(response: dict[str, Any]) -> dict[str, Any]:
    responses = response.get("responses")
    if isinstance(responses, list) and responses:
        first = responses[0]
        return first if isinstance(first, dict) else {}
    return response


def _response_error(response: dict[str, Any]) -> str | None:
    error = response.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        reason = error.get("reason") or error.get("type") or "OpenSearch error"
        return str(reason)
    return str(error)


def _aggregation_buckets(response: dict[str, Any]) -> list[Any]:
    buckets = response.get("aggregations", {}).get("result", {}).get("buckets", [])
    return buckets if isinstance(buckets, list) else []


def _total_descriptor(total: Any) -> dict[str, Any]:
    if isinstance(total, dict):
        value = total.get("value", 0)
        relation = total.get("relation", "eq")
    else:
        value = total or 0
        relation = "eq"
    return {"value": value, "relation": relation}


def _terms_field(field: str) -> str:
    return field if field.endswith(".keyword") else f"{field}.keyword"


def _aggregation_validation_error(*, query: str, group_by: str, agg_type: str) -> str | None:
    if not query:
        return "query must be non-empty; use '*' for all documents"
    if not group_by:
        return "group_by must be non-empty"
    if agg_type not in SUPPORTED_AGG_TYPES:
        return (
            "agg_type must be one of "
            + ", ".join(sorted(SUPPORTED_AGG_TYPES))
        )
    return None


def _clean_query(query: str, *, token_env: str = DEFAULT_TOKEN_ENV) -> str:
    clean = (query or "").strip()
    if len(clean) > MAX_QUERY_CHARS:
        clean = clean[:MAX_QUERY_CHARS]
    token = os.environ.get(token_env)
    if token:
        clean = clean.replace(token, "[redacted]")
    return clean


def _sanitize_time(value: str | None, default: str) -> str:
    """One sanitization for both the query body and the echoed window."""
    return ((value or "").strip()) or default


def _time_window(from_time: str, to_time: str) -> dict[str, str]:
    """Echo of the exact window the query body used.

    Callers pass values already normalized by :func:`_sanitize_time`;
    no further defaulting/stripping happens here, so the echo cannot
    drift from what was actually queried.
    """
    return {
        "from": from_time,
        "to": to_time,
        "time_field": "metadata.timestamp",
    }


def _bounded_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, lower), upper)


def _observed_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_error_message(exc: requests.exceptions.HTTPError) -> str:
    response = exc.response
    status = response.status_code if response is not None else "unknown"
    if status in {401, 403}:
        return f"MONIT OpenSearch authentication failed with HTTP {status}"
    return f"MONIT OpenSearch request failed with HTTP {status}"


def _request_error_message(exc: Exception) -> str:
    """Structured message for transport/parse failures.

    Built from the exception *type* only — exception text can embed
    URLs/hosts, and these messages travel back to the agent verbatim.
    """
    if isinstance(exc, json.JSONDecodeError):
        return "MONIT OpenSearch returned a non-JSON response body"
    return f"MONIT OpenSearch request failed: {type(exc).__name__}"


_FIELD_ERROR_MARKERS = (
    "no mapping found",
    "fielddata",
    "field data",
    "unknown field",
    "failed to find field",
    "not aggregatable",
)


def _looks_like_field_error(error: str, field: str) -> bool:
    """True when an OpenSearch error is about the aggregation field
    itself (missing ``.keyword`` sub-field, non-aggregatable type) —
    the only case where retrying with the raw field name can help."""
    lowered = error.lower()
    if field and field.lower() in lowered:
        return True
    return any(marker in lowered for marker in _FIELD_ERROR_MARKERS)


__all__ = [
    "archi_monit_condor_aggregate",
    "archi_monit_condor_search",
    "archi_monit_rucio_aggregate",
    "archi_monit_rucio_search",
    "monit_aggregate",
    "monit_search",
]
